#!/usr/bin/env python3
"""Render deterministic 4096px CRCO_V1 visual-audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "research" / "f1_v29_crco_v1_lock.json"
FIXTURE_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_fixtures"
RESULT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "source_free_results" / "run_a"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v29-crco-v1" / "visual_audit_4096"

W, H = 4096, 2304
BG = (13, 18, 27)
PANEL = (25, 33, 46)
TEXT = (235, 241, 248)
MUTED = (156, 169, 186)
GREEN = (59, 214, 139)
RED = (248, 95, 106)
AMBER = (250, 188, 74)
BLUE = (80, 169, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((96, 66), title, font=_font(70), fill=TEXT)
    draw.text((98, 154), subtitle, font=_font(32), fill=MUTED)
    return image, draw


def _save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(fixture_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = json.loads((FIXTURE_ROOT / f"{fixture_id}.json").read_text(encoding="utf-8"))
    result = json.loads((RESULT_ROOT / fixture_id / "result.json").read_text(encoding="utf-8"))
    return fixture, result


def _surface_view(fixture_id: str, fixture: dict[str, Any], result: dict[str, Any]) -> Image.Image:
    image, draw = _canvas(f"{fixture_id} — analytic surface view", "Source-free geometry; teal=legacy, green=rescue, red=veto")
    box = (120, 270, W - 120, H - 110)
    draw.rounded_rectangle(box, 30, fill=PANEL, outline=(52, 66, 86), width=4)
    cx, cy = W // 2, H // 2 + 100
    scale = 28.0
    # Two translucent-looking ruled flanks make the ownership relationship legible.
    for offset in range(-8, 9):
        y = cy + int(offset * 25)
        draw.line((cx - 1150, y, cx + 1150, y), fill=(45, 75, 91), width=3)
        draw.line((cx + int(offset * 20), cy - 750, cx + int(offset * 20), cy + 600), fill=(49, 61, 91), width=3)
    decisions = {row["candidate"]["candidate_hash"]: row for row in result["decisions"]}
    for index, candidate in enumerate(fixture["candidate_records"]):
        # Use result order to recover the content-derived candidate ID.
        decision = result["decisions"][index] if len(result["decisions"]) == len(fixture["candidate_records"]) else None
        if decision is None:
            continue
        action = decision["action"]
        color = GREEN if action == "RESCUE" else RED if action == "VETO" else BLUE
        yoff = float(candidate["line_start"][1])
        p0 = (cx - int(24 * scale), cy - int(yoff * 18))
        p1 = (cx + int(24 * scale), cy - int(yoff * 18))
        draw.line((*p0, *p1), fill=(0, 0, 0), width=28)
        draw.line((*p0, *p1), fill=color, width=16)
        draw.ellipse((p0[0] - 12, p0[1] - 12, p0[0] + 12, p0[1] + 12), fill=color)
        draw.ellipse((p1[0] - 12, p1[1] - 12, p1[0] + 12, p1[1] + 12), fill=color)
        draw.text((p1[0] + 40, p1[1] - 25), f"{action}  C={decision['continuous_station_count']} J={decision['jump_station_count']}", font=_font(30), fill=TEXT)
    draw.text((160, H - 185), f"final lines: {len(result['final_lines'])}   queried pixels: {result['queried_pixel_count']}", font=_font(36), fill=TEXT)
    return image


def _range_keyframes(fixture_id: str, fixture: dict[str, Any], result: dict[str, Any]) -> Image.Image:
    image, draw = _canvas(f"{fixture_id} — selected calibrated range keyframes", "Valid depth is colored; invalid pixels are black; frame hashes are content-derived")
    margin, gap = 100, 54
    panel_w = (W - 2 * margin - 3 * gap) // 4
    y0, y1 = 330, H - 170
    for index, frame in enumerate(fixture["range_frames"]):
        x0 = margin + index * (panel_w + gap)
        x1 = x0 + panel_w
        depth = np.asarray(frame["depth_q"], dtype=np.float64)
        valid = np.asarray(frame["valid"], dtype=np.bool_)
        rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        if valid.any():
            values = depth[valid]
            lo, hi = float(values.min()), float(values.max())
            t = np.zeros_like(depth)
            if hi > lo:
                t[valid] = (depth[valid] - lo) / (hi - lo)
            rgb[..., 0] = np.where(valid, 30 + 210 * t, 0).astype(np.uint8)
            rgb[..., 1] = np.where(valid, 215 - 120 * t, 0).astype(np.uint8)
            rgb[..., 2] = np.where(valid, 245 - 180 * t, 0).astype(np.uint8)
        tile = Image.fromarray(rgb, mode="RGB").resize((panel_w, y1 - y0), Image.Resampling.NEAREST)
        image.paste(tile, (x0, y0))
        draw.rectangle((x0, y0, x1, y1), outline=(90, 108, 132), width=4)
        key = result["selected_keyframe_hashes"][index]
        draw.text((x0, y0 - 58), f"K{index + 1}  {key[:12]}", font=_font(30), fill=TEXT)
    return image


def _continuity_evidence(fixture_id: str, _fixture: dict[str, Any], result: dict[str, Any]) -> Image.Image:
    image, draw = _canvas(f"{fixture_id} — continuity evidence", "Rows are candidate × keyframe; columns are 24 fixed stations")
    color = {"C": GREEN, "J": RED, "O": AMBER, "U": (91, 101, 117)}
    x0, x1 = 520, W - 130
    y = 340
    cell_w = (x1 - x0) / 24.0
    for candidate_index, decision in enumerate(result["decisions"]):
        draw.text((100, y + 8), f"candidate {candidate_index + 1}", font=_font(34), fill=TEXT)
        for frame_index in range(4):
            row_y = y + frame_index * 86
            draw.text((330, row_y + 12), f"K{frame_index + 1}", font=_font(28), fill=MUTED)
            for station_index, station in enumerate(decision["stations"]):
                state = station["per_frame"][frame_index]["state"]
                left = int(x0 + station_index * cell_w)
                right = int(x0 + (station_index + 1) * cell_w - 4)
                draw.rounded_rectangle((left, row_y, right, row_y + 58), 8, fill=color[state])
        y += 420
    legend_y = H - 120
    for index, (state, fill) in enumerate(color.items()):
        lx = 1500 + index * 430
        draw.rectangle((lx, legend_y, lx + 45, legend_y + 45), fill=fill)
        draw.text((lx + 62, legend_y + 2), {"C": "continuous", "J": "jump", "O": "occluded", "U": "unknown"}[state], font=_font(28), fill=TEXT)
    return image


def _station_matrix(fixture_id: str, _fixture: dict[str, Any], result: dict[str, Any]) -> Image.Image:
    image, draw = _canvas(f"{fixture_id} — cross-view station matrix", "Final cross-view adjudication; decision thresholds are 16/24 stations and an 8-station run")
    color = {"C": GREEN, "J": RED, "O": AMBER, "U": (91, 101, 117)}
    x0, x1 = 600, W - 160
    y = 400
    cell_w = (x1 - x0) / 24.0
    for index, decision in enumerate(result["decisions"]):
        draw.text((100, y + 28), f"candidate {index + 1}", font=_font(34), fill=TEXT)
        for station_index, state in enumerate(decision["cross_view_states"]):
            left = int(x0 + station_index * cell_w)
            right = int(x0 + (station_index + 1) * cell_w - 5)
            draw.rounded_rectangle((left, y, right, y + 92), 10, fill=color[state])
            draw.text((left + 38, y + 22), state, font=_font(34), fill=(10, 15, 22))
        draw.text((x0, y + 125), f"action={decision['action']}   C={decision['continuous_station_count']}   J={decision['jump_station_count']}   O={decision['occluded_station_count']}   U={decision['unknown_station_count']}   runs C/J={decision['longest_continuous_run']}/{decision['longest_jump_run']}", font=_font(32), fill=TEXT)
        y += 300
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise SystemExit("V29 visual output root is immutable")
    fixture_ids = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["source_free_fixture_ids"]
    rows = []
    for fixture_id in fixture_ids:
        fixture, result = _load(fixture_id)
        builders = {
            "native_surface": _surface_view,
            "range_keyframes": _range_keyframes,
            "continuity_evidence": _continuity_evidence,
            "station_matrix": _station_matrix,
        }
        artifacts = []
        for kind, builder in builders.items():
            path = OUTPUT_ROOT / fixture_id / f"{kind}_4096.png"
            image = builder(fixture_id, fixture, result)
            if max(image.size) != 4096:
                raise SystemExit("visual long edge contract failure")
            artifacts.append({"kind": kind, "path": str(path.relative_to(ROOT)), "sha256": _save(image, path), "width": image.width, "height": image.height})
        rows.append({"fixture_id": fixture_id, "artifacts": artifacts})
    manifest = {"kind": "v29_crco_native_4096_visual_manifest", "visual_long_edge_pixels": 4096, "fixtures": rows}
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"fixture_count": len(rows), "artifact_count": sum(len(row["artifacts"]) for row in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
