#!/usr/bin/env python3
"""Render truth-free visual audits for calibrated raw-range bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


COLORS = plt.get_cmap("turbo")


def _load(bundle: Path) -> tuple[dict, np.ndarray, list[dict]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    source = np.asarray(
        o3d.io.read_point_cloud(str(bundle / manifest["source_geometry"]["file"])).points,
        dtype=np.float64,
    )
    frames = []
    for row in manifest["frames"]:
        frames.append({
            "ordinal": row["ordinal"],
            "depth": np.load(bundle / row["depth_file"], allow_pickle=False).astype(np.float64),
            "validity": np.load(bundle / row["validity_file"], allow_pickle=False).astype(bool),
            "K": np.asarray(row["K"], dtype=np.float64),
            "T": np.asarray(row["T_world_from_camera"], dtype=np.float64),
        })
    return manifest, source, frames


def _backproject(frame: dict) -> np.ndarray:
    depth, validity, K, T = frame["depth"], frame["validity"], frame["K"], frame["T"]
    flat = np.flatnonzero(validity.ravel())
    v = flat // depth.shape[1]
    u = flat % depth.shape[1]
    pixels = np.column_stack((u, v, np.ones(len(flat))))
    camera = (np.linalg.inv(K) @ pixels.T).T * depth.ravel()[flat, None]
    return camera @ T[:3, :3].T + T[:3, 3]


def _depth_rgb(depth: np.ndarray, validity: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((depth - low) / max(high - low, 1e-9), 0.0, 1.0)
    rgb = (COLORS(normalized)[..., :3] * 255.0).astype(np.uint8)
    rgb[~validity] = 0
    return rgb


def render_bundle(bundle: Path, output: Path) -> dict:
    manifest, source, frames = _load(bundle)
    output.mkdir(parents=True, exist_ok=False)
    valid_depths = np.concatenate([frame["depth"][frame["validity"]] for frame in frames])
    low, high = map(float, np.quantile(valid_depths, (0.01, 0.99)))

    native_paths = []
    tile_images = []
    for frame in frames:
        rgb = _depth_rgb(frame["depth"], frame["validity"], low, high)
        path = output / f"frame_{frame['ordinal']:03d}_depth_native.png"
        Image.fromarray(rgb, mode="RGB").save(path)
        native_paths.append(str(path.resolve()))
        tile_images.append(Image.fromarray(rgb, mode="RGB"))

    width, height = tile_images[0].size
    columns = min(4, len(tile_images))
    rows = (len(tile_images) + columns - 1) // columns
    caption = 24
    contact = Image.new("RGB", (columns * width, rows * (height + caption)), "white")
    draw = ImageDraw.Draw(contact)
    for index, tile in enumerate(tile_images):
        x = (index % columns) * width
        y = (index // columns) * (height + caption)
        contact.paste(tile, (x, y + caption))
        frame = frames[index]
        draw.text((x + 6, y + 5), f"view {index:02d} | valid {frame['validity'].mean():.3f}", fill="black")
    contact_path = output / "depth_contact_native_tiles.png"
    contact.save(contact_path)

    figure = plt.figure(figsize=(18, 6), dpi=120)
    sampled_source = source[np.linspace(0, len(source) - 1, min(25_000, len(source)), dtype=np.int64)]
    frame_points = []
    for frame in frames:
        points = _backproject(frame)
        points = points[np.linspace(0, len(points) - 1, min(1_800, len(points)), dtype=np.int64)]
        frame_points.append(points)
    view_specs = ((25, -60, "oblique"), (90, -90, "top"), (0, -90, "side"))
    for slot, (elevation, azimuth, title) in enumerate(view_specs, 1):
        axis = figure.add_subplot(1, 3, slot, projection="3d")
        axis.scatter(sampled_source[:, 0], sampled_source[:, 1], sampled_source[:, 2], s=0.18, c="#777777", alpha=0.16, depthshade=False)
        for index, points in enumerate(frame_points):
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.55, color=COLORS(index / max(1, len(frame_points) - 1)), alpha=0.70, depthshade=False)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(f"{title}: colored raw views on gray fused source")
        axis.set_box_aspect(np.maximum(np.ptp(source, axis=0), 1e-6))
        axis.set_axis_off()
    figure.suptitle(f"{manifest['opaque_case_id']} calibrated back-projection audit | truth-free")
    figure.tight_layout()
    alignment_path = output / "backprojection_alignment_3d.png"
    figure.savefig(alignment_path, bbox_inches="tight")
    plt.close(figure)

    report = {
        "schema_version": 1,
        "kind": "calibrated_raw_range_truth_free_visual_audit_v1",
        "opaque_case_id": manifest["opaque_case_id"],
        "frame_count": len(frames),
        "native_frame_dimensions": [width, height],
        "shared_depth_window_mm": [low, high],
        "native_depth_paths": native_paths,
        "contact_path": str(contact_path.resolve()),
        "alignment_path": str(alignment_path.resolve()),
        "truth_opened": False,
    }
    (output / "visual_manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render_bundle(args.bundle.resolve(), args.output.resolve())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
