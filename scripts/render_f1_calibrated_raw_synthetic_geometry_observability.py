#!/usr/bin/env python3
"""Render evaluator truth against fused and native depth geometry.

This is a corpus-construction audit, not an inference review.  It verifies that
intended synthetic bends remain visible after scanner-like rendering and fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_line(line: dict, frame: dict, sample_count: int = 256) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(line["line_start"], dtype=float)
    end = np.asarray(line["line_end"], dtype=float)
    samples = start[None, :] + np.linspace(0.0, 1.0, sample_count)[:, None] * (end - start)[None, :]
    intrinsics = np.asarray(frame["K"], dtype=float)
    pose = np.asarray(frame["T_world_from_camera"], dtype=float)
    camera = (samples - pose[:3, 3]) @ pose[:3, :3]
    projected = camera @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:3]
    return uv, camera[:, 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output root is not empty")
    args.output_root.mkdir(parents=True, exist_ok=True)

    bundle = args.corpus_root / "public" / args.case_id
    evaluator = args.corpus_root / "evaluator_only" / args.case_id
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    truth = json.loads((evaluator / "truth.json").read_text(encoding="utf-8"))["manual_bends"]
    source = np.asarray(o3d.io.read_point_cloud(str(bundle / "source.ply")).points, dtype=float)
    sampled = source[np.linspace(0, len(source) - 1, min(80_000, len(source)), dtype=int)]

    fig = plt.figure(figsize=(18, 7), dpi=180)
    for panel, (elev, azim, title) in enumerate(((25, -58, "oblique"), (90, -90, "top"), (4, -90, "side")), 1):
        axis = fig.add_subplot(1, 3, panel, projection="3d")
        axis.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=0.08, c="#aeb6c2", alpha=0.32, rasterized=True)
        for index, bend in enumerate(truth):
            line = np.asarray([bend["line_start"], bend["line_end"]], dtype=float)
            axis.plot(line[:, 0], line[:, 1], line[:, 2], color=plt.cm.turbo(index / max(1, len(truth) - 1)), linewidth=2.5)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title)
        axis.set_axis_off()
        axis.set_box_aspect(np.maximum(np.ptp(source, axis=0), 1e-6))
    fig.suptitle(f"{args.case_id} geometry observability | intended bend centers over fused scanner-like source")
    fused_path = args.output_root / "fused_truth_overlay.png"
    fig.savefig(fused_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    frame_count = len(manifest["frames"])
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), squeeze=False)
    visibility = []
    for frame_index, frame in enumerate(manifest["frames"]):
        axis = axes.flat[frame_index]
        depth = np.asarray(np.load(bundle / frame["depth_file"], allow_pickle=False), dtype=float)
        valid = np.asarray(np.load(bundle / frame["validity_file"], allow_pickle=False), dtype=bool)
        image = np.full_like(depth, np.nan)
        image[valid] = depth[valid]
        axis.imshow(image, cmap="gray_r", interpolation="nearest")
        bend_visibility = []
        for bend_index, bend in enumerate(truth):
            uv, z = _project_line(bend, frame)
            inside = (
                (z > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < depth.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < depth.shape[0])
            )
            bend_visibility.append(int(np.count_nonzero(inside)))
            if np.any(inside):
                axis.plot(
                    uv[inside, 0],
                    uv[inside, 1],
                    color=plt.cm.turbo(bend_index / max(1, len(truth) - 1)),
                    linewidth=1.6,
                )
        visibility.append(bend_visibility)
        axis.set_title(f"native view {frame_index} | valid {valid.mean():.3f}")
        axis.set_axis_off()
    for unused in range(frame_count, len(axes.flat)):
        axes.flat[unused].set_axis_off()
    fig.suptitle(f"{args.case_id} native calibrated depth | intended bend centers", fontsize=16)
    fig.tight_layout()
    native_path = args.output_root / "native_depth_truth_overlay.png"
    fig.savefig(native_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": 1,
        "kind": "synthetic_geometry_observability_audit",
        "case_id": args.case_id,
        "bend_count": len(truth),
        "frame_count": frame_count,
        "visibility_samples_by_frame_and_bend": visibility,
        "fused_overlay": str(fused_path.resolve()),
        "fused_overlay_sha256": _sha(fused_path),
        "native_overlay": str(native_path.resolve()),
        "native_overlay_sha256": _sha(native_path),
        "truth_used_for_corpus_construction_audit": True,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": args.case_id, "bend_count": len(truth), "frame_count": frame_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
