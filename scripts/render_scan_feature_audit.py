#!/usr/bin/env python3
"""Render close orthographic audits of small formed scan features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-path", required=True)
    parser.add_argument("--frame-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.scan_path)
    if not cloud.has_normals():
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=40))
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    frame = np.load(args.frame_path)
    center, basis = frame["center"], frame["basis"]
    local = (points - center) @ basis.T
    normals_local = normals @ basis.T
    plane, _ = cloud.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=5000)
    signed = points @ np.asarray(plane[:3]) + float(plane[3])

    crops = [
        ("upper formed feature", (-125.0, 15.0), (60.0, 100.0)),
        ("lower formed feature", (-100.0, 35.0), (-100.0, -60.0)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=200, constrained_layout=True)
    for row, (title, xr, yr) in enumerate(crops):
        mask = (
            (local[:, 0] >= xr[0]) & (local[:, 0] <= xr[1])
            & (local[:, 1] >= yr[0]) & (local[:, 1] <= yr[1])
        )
        q, n, h = local[mask], normals_local[mask], signed[mask]
        order = np.argsort(np.abs(h))
        q, n, h = q[order], n[order], h[order]

        ax = axes[row, 0]
        sc = ax.scatter(q[:, 0], q[:, 1], c=np.clip(h, -15, 15), cmap="coolwarm", s=1.0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="signed distance from dominant plate plane (mm)")
        ax.set_title(f"{title}: height residual")

        ax = axes[row, 1]
        rgb = np.clip(0.1 + 0.9 * np.abs(n), 0, 1)
        ax.scatter(q[:, 0], q[:, 1], c=rgb, s=1.0, linewidths=0)
        ax.set_title(f"{title}: surface-normal map")

        ax = axes[row, 2]
        ax.scatter(q[:, 0], h, c=rgb, s=1.0, linewidths=0)
        ax.set_ylim(-16, 16)
        ax.set_title(f"{title}: long-axis section")
        ax.set_ylabel("plane residual (mm)")

        for ax in axes[row, :2]:
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax.set_aspect("equal")
            ax.set_xlabel("local X (mm)")
            ax.set_ylabel("local Y (mm)")
        axes[row, 2].set_xlim(*xr)
        axes[row, 2].set_xlabel("local X (mm)")
        for ax in axes[row]:
            ax.grid(alpha=0.18)

    fig.suptitle("38172000 close feature audit — physical bend versus three-sided free cut", fontsize=18)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
