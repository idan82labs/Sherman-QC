#!/usr/bin/env python3
"""Render the aggregate cross-section perpendicular to long parallel bends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.scan_path))
    if not cloud.has_normals():
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=40))
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    frame = np.load(args.frame_path)
    center, basis = frame["center"], frame["basis"]
    local = (points - center) @ basis.T
    local_normals = normals @ basis.T
    labels = json.loads(args.labels_path.read_text(encoding="utf-8"))["manual_bends"]

    # Remove end caps and the two internal tabs.  What remains is the repeated
    # sheet profile whose vertices correspond to long bend axes.
    mask = np.abs(local[:, 1]) <= 70.0
    q = local[mask]
    n = local_normals[mask]
    rgb = np.clip(0.12 + 0.88 * np.abs(n), 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=220, constrained_layout=True)
    axes[0].scatter(q[:, 0], q[:, 2], c=rgb, s=1.0, linewidths=0, alpha=0.7)
    axes[0].set_title("Central cross-section: RGB = |surface normal|")
    axes[1].hexbin(q[:, 0], q[:, 2], gridsize=300, bins="log", cmap="magma", mincnt=1)
    axes[1].set_title("Central cross-section: log point density")

    for record in labels:
        raw = np.asarray([record["line_start"], record["line_end"]], dtype=np.float64)
        line = (raw - center) @ basis.T
        delta = line[1] - line[0]
        direction = delta / max(float(np.linalg.norm(delta)), 1e-12)
        if abs(float(direction[1])) < 0.9:
            continue
        midpoint = line.mean(axis=0)
        for ax in axes:
            ax.scatter(midpoint[0], midpoint[2], s=80, color="#32d7ff", edgecolor="black", zorder=5)
            ax.text(
                midpoint[0] + 1.5,
                midpoint[2] + 1.0,
                str(record["bend_id"]),
                color="black",
                fontsize=11,
                fontweight="bold",
                bbox={"facecolor": "#32d7ff", "edgecolor": "black", "pad": 1.5},
                zorder=6,
            )
    for ax in axes:
        ax.set_xlabel("local X (mm)")
        ax.set_ylabel("local Z (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
    fig.suptitle("38172000 — profile perpendicular to B03 (end artifacts excluded)", fontsize=18)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, facecolor="white")
    plt.close(fig)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
