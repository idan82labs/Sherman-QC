#!/usr/bin/env python3
"""Tile synthetic scored overlays with per-case TP/FP/FN labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    overlay_root = args.overlay_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    score = json.loads(args.score.read_text(encoding="utf-8"))
    rows = sorted(score["cases"], key=lambda row: str(row["case_id"]))
    sheets = []
    for sheet_index in range(5):
        subset = rows[sheet_index * 6 : (sheet_index + 1) * 6]
        canvas = Image.new("RGB", (2400, 1800), "#07101f")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(subset):
            case_id = str(row["case_id"])
            source = overlay_root / f"{case_id}.png"
            image = Image.open(source).convert("RGB")
            image.thumbnail((780, 820), Image.Resampling.LANCZOS)
            column, line = index % 3, index // 3
            x = column * 800 + (800 - image.width) // 2
            y = line * 900 + 58 + (800 - image.height) // 2
            canvas.paste(image, (x, y))
            status = "EXACT" if row["exact"] else "MISS"
            color = "#7CFFB2" if row["exact"] else "#FFB36B"
            label = (
                f"{case_id} {row['family_id']} {status} — "
                f"TP {row['true_positive']} / FP {row['false_positive']} / FN {row['false_negative']}"
            )
            draw.text((column * 800 + 20, line * 900 + 20), label, fill=color)
        first = sheet_index * 6 + 1
        last = (sheet_index + 1) * 6
        path = output_root / f"scored_{first:03d}_{last:03d}.png"
        canvas.save(path, quality=95)
        sheets.append({"path": str(path), "sha256": _sha(path), "case_ids": [str(row["case_id"]) for row in subset]})
    report = {
        "schema_version": 1,
        "kind": "synthetic_lidar_scored_visual_contact_sheets",
        "confirmatory_eligible": False,
        "sheets": sheets,
    }
    report_path = output_root / "contact_sheet_manifest.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sheet_count": len(sheets), "manifest": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
