#!/usr/bin/env python3
"""Render deterministic high-resolution orthographic audits of a PLY scan.

The audit intentionally colors points by surface-normal orientation.  Physical
bends remain visible across views as a color transition; free outlines do not
acquire a second supported surface and are therefore easier to reject.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _canonical_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    sample_step = max(1, len(points) // 200_000)
    _, _, basis = np.linalg.svd(points[::sample_step] - center, full_matrices=False)
    # Keep the broad sheet in local XY and choose stable signs for repeatability.
    if basis[0, 0] < 0:
        basis[0] *= -1
    if basis[1, 1] < 0:
        basis[1] *= -1
    basis[2] = np.cross(basis[0], basis[1])
    return center, basis, (points - center) @ basis.T


def _normal_colors(normals_local: np.ndarray) -> np.ndarray:
    colors = 0.15 + 0.85 * np.abs(normals_local)
    return np.clip(colors, 0.0, 1.0)


def render_scan_multiview(
    scan_path: Path,
    output_path: Path,
    *,
    max_points: int = 260_000,
    display_name: Optional[str] = None,
    write_frame: bool = True,
) -> Path:
    """Render one scan without requiring its source identity in the image.

    ``display_name`` exists for blind review packs: callers can provide an
    opaque case ID instead of leaking the native filename into the title.
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    scan_path = Path(scan_path)
    output = Path(output_path)
    cloud = o3d.io.read_point_cloud(str(scan_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        raise ValueError(f"No points in {scan_path}")
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=40)
        )
        cloud.orient_normals_consistent_tangent_plane(20)
    normals = np.asarray(cloud.normals, dtype=np.float64)

    center, basis, local = _canonical_pca(points)
    normals_local = normals @ basis.T
    step = max(1, len(points) // max_points)
    local = local[::step]
    colors = _normal_colors(normals_local[::step])

    fig = plt.figure(figsize=(24, 18), dpi=180, constrained_layout=True)
    views = [
        (90, -90, "plan / normal map"),
        (24, -100, "front oblique"),
        (22, -10, "side oblique"),
        (12, 145, "reverse low oblique"),
        (8, 80, "upper-feature grazing view"),
        (-6, -100, "lower-feature grazing view"),
    ]
    span = np.ptp(local, axis=0)
    midpoint = (local.min(axis=0) + local.max(axis=0)) / 2.0
    radius = float(max(span[:2]) * 0.56)
    zradius = float(max(span[2] * 0.8, radius * 0.18))
    for index, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        ax.scatter(
            local[:, 0], local[:, 1], local[:, 2],
            c=colors, s=0.32, alpha=0.9, linewidths=0, rasterized=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
        ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
        ax.set_zlim(midpoint[2] - zradius, midpoint[2] + zradius)
        ax.set_box_aspect((2 * radius, 2 * radius, 2 * zradius))
        ax.set_title(f"{title}  (elev {elev}°, azim {azim}°)", fontsize=13)
        ax.set_xlabel("local X (mm)")
        ax.set_ylabel("local Y (mm)")
        ax.set_zlabel("local Z (mm)")
        ax.grid(False)
    fig.suptitle(
        f"{display_name or scan_path.name} scan geometry audit — "
        "RGB = |local surface normal|\n"
        "A bend needs two supported surface directions; a silhouette/free edge does not.",
        fontsize=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        facecolor="white",
        metadata={"Software": "ShermanQC deterministic multiview renderer"},
    )
    plt.close(fig)
    if write_frame:
        np.savez(
            output.with_suffix(".frame.npz"),
            center=center,
            basis=basis,
            local_min=local.min(axis=0),
            local_max=local.max(axis=0),
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-points", type=int, default=260_000)
    parser.add_argument("--display-name")
    parser.add_argument("--no-frame", action="store_true")
    args = parser.parse_args()

    output = render_scan_multiview(
        Path(args.scan_path),
        Path(args.output_path),
        max_points=args.max_points,
        display_name=args.display_name,
        write_frame=not args.no_frame,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
