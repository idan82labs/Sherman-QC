#!/usr/bin/env python3
"""Render a dominant-sheet scan in a PCA-aligned edge-profile audit."""

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
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path)
    parser.add_argument("--max-points", type=int, default=450_000)
    args = parser.parse_args()
    mesh = o3d.io.read_triangle_mesh(str(args.scan_path))
    points = np.asarray(mesh.vertices, dtype=float)
    stride = max(1, len(points) // args.max_points)
    points = points[::stride]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, _ = cloud.segment_plane(distance_threshold=0.35, ransac_n=3, num_iterations=5000)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    center = points.mean(axis=0)
    height = (points - center) @ normal
    flat = points - center - height[:, None] * normal
    _, vectors = np.linalg.eigh(np.cov(flat.T))
    long_axis = vectors[:, -1]
    cross_axis = np.cross(normal, long_axis)
    cross_axis /= np.linalg.norm(cross_axis)
    local = np.column_stack(((points - center) @ long_axis, (points - center) @ cross_axis, height))

    predictions: list[dict[str, object]] = []
    if args.prediction_path is not None:
        payload = json.loads(args.prediction_path.read_text(encoding="utf-8"))
        predictions = list(payload.get("predictions", []))

    projected_predictions: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index, prediction in enumerate(predictions, start=1):
        raw = np.asarray(
            [prediction["line_start"], prediction["line_end"]], dtype=float
        )
        line_height = (raw - center) @ normal
        line_flat = raw - center - line_height[:, None] * normal
        line_local = np.column_stack(
            (
                line_flat @ long_axis,
                line_flat @ cross_axis,
                line_height,
            )
        )
        projected_predictions.append((index, line_local, line_local.mean(axis=0)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=220, constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(local[:, 0], local[:, 1], c=np.clip(local[:, 2], -16, 7), cmap="coolwarm", s=.22, linewidths=0)
    ax.set_title("Dominant-sheet plan: color = signed height (mm)")
    ax.set_aspect("equal")
    ax.set_xlabel("long axis (mm)")
    ax.set_ylabel("cross axis (mm)")
    for index, line, _ in projected_predictions:
        ax.plot(line[:, 0], line[:, 1], linewidth=2.2, alpha=.85)
        ax.text(
            line[:, 0].mean(), line[:, 1].mean(), str(index), fontsize=10,
            fontweight="bold", ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "pad": 1.2},
        )

    for ax, mask, title in (
        (axes[0, 1], np.ones(len(local), dtype=bool), "All-points cross section"),
        (axes[1, 0], np.abs(local[:, 0]) < 20, "Central 40 mm cross-section slice"),
        (axes[1, 1], np.abs(local[:, 0] + 230) < 20, "Feature-end 40 mm cross-section slice"),
    ):
        q = local[mask]
        ax.scatter(q[:, 1], q[:, 2], c=q[:, 0], cmap="viridis", s=.35, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("cross axis (mm)")
        ax.set_ylabel("signed height (mm)")
        ax.set_ylim(-18, 9)
        ax.grid(alpha=.2)
        for index, line, midpoint in projected_predictions:
            direction = line[1] - line[0]
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            if abs(float(direction[0])) < .85:
                continue
            ax.scatter(
                midpoint[1], midpoint[2], s=70, color="#ff7a00",
                edgecolor="black", linewidth=.8, zorder=5,
            )
            ax.text(
                midpoint[1] + 1.2, midpoint[2] + .35, str(index),
                fontsize=10, fontweight="bold", color="black", zorder=6,
                bbox={"facecolor": "#ffb347", "edgecolor": "black", "pad": 1.0},
            )
    fig.suptitle(f"{args.scan_path.name} edge-profile audit — raw PLY only", fontsize=18)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, facecolor="white")
    plt.close(fig)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
