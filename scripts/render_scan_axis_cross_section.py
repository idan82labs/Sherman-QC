#!/usr/bin/env python3
"""Render a deterministic point-cloud cross-section normal to a candidate axis.

This is a manual-adjudication aid.  It removes the axial end regions, projects
the retained central points into the plane perpendicular to the supplied axis,
and optionally overlays candidate anchors from an F1 candidate-pool payload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("Axis vector must be non-zero")
    return values / norm


def _parse_axis(raw: str) -> np.ndarray:
    values = np.asarray([float(value) for value in raw.split(",")], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("--axis must contain exactly three comma-separated numbers")
    return _unit(values)


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(axis @ helper)) > 0.85:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(np.cross(axis, helper))
    v = _unit(np.cross(axis, u))
    if u[np.argmax(np.abs(u))] < 0:
        u *= -1.0
    v = _unit(np.cross(axis, u))
    return u, v


def _candidate_anchor(candidate: dict[str, object]) -> np.ndarray | None:
    anchor = candidate.get("anchor")
    if isinstance(anchor, list) and len(anchor) == 3:
        return np.asarray(anchor, dtype=np.float64)
    endpoints = candidate.get("span_endpoints")
    if isinstance(endpoints, list) and len(endpoints) == 2:
        return np.asarray(endpoints, dtype=np.float64).mean(axis=0)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-path", type=Path, required=True)
    parser.add_argument("--axis", required=True, help="World-frame fold axis as x,y,z")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--candidate-sources", default="")
    parser.add_argument("--central-fraction", type=float, default=0.55)
    parser.add_argument(
        "--axial-center-mm",
        type=float,
        help="Optional world-axis coordinate relative to the scan median",
    )
    parser.add_argument(
        "--axial-half-width-mm",
        type=float,
        help="Optional half-width for an explicit axial slice",
    )
    parser.add_argument("--max-points", type=int, default=240_000)
    parser.add_argument("--title", default="Central fold-profile audit")
    args = parser.parse_args()

    if not 0.05 <= args.central_fraction <= 1.0:
        raise ValueError("--central-fraction must be between 0.05 and 1.0")

    cloud = o3d.io.read_point_cloud(str(args.scan_path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        raise ValueError(f"No points in {args.scan_path}")
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=40)
        )
    normals = np.asarray(cloud.normals, dtype=np.float64)

    axis = _parse_axis(args.axis)
    u, v = _perpendicular_basis(axis)
    center = np.median(points, axis=0)
    centered = points - center
    axial = centered @ axis
    transverse = np.column_stack((centered @ u, centered @ v))
    normal_rgb = np.clip(
        0.15
        + 0.85
        * np.abs(np.column_stack((normals @ u, normals @ v, normals @ axis))),
        0.0,
        1.0,
    )

    if (args.axial_center_mm is None) != (args.axial_half_width_mm is None):
        raise ValueError(
            "--axial-center-mm and --axial-half-width-mm must be supplied together"
        )
    half_span = (
        float(args.axial_half_width_mm)
        if args.axial_half_width_mm is not None
        else float(np.ptp(axial)) * args.central_fraction * 0.5
    )
    axial_center = (
        float(args.axial_center_mm)
        if args.axial_center_mm is not None
        else float(np.median(axial))
    )
    mask = np.abs(axial - axial_center) <= half_span
    selected = np.flatnonzero(mask)
    step = max(1, len(selected) // args.max_points)
    selected = selected[::step]

    sources = {
        source.strip() for source in args.candidate_sources.split(",") if source.strip()
    }
    candidates: list[dict[str, object]] = []
    if args.candidate_pool:
        payload = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
        candidates = [
            candidate
            for candidate in payload.get("candidates", [])
            if not sources or candidate.get("candidate_source") in sources
        ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=220, constrained_layout=True)
    axes[0].scatter(
        transverse[selected, 0],
        transverse[selected, 1],
        c=normal_rgb[selected],
        s=0.55,
        linewidths=0,
        alpha=0.75,
        rasterized=True,
    )
    axes[0].set_title("Surface-normal color")
    axes[1].hexbin(
        transverse[selected, 0],
        transverse[selected, 1],
        gridsize=320,
        bins="log",
        cmap="magma",
        mincnt=1,
    )
    axes[1].set_title("Log point density")

    palette = {
        "scan_mesh_adjacency": "#2dd4bf",
        "plane_lattice": "#f59e0b",
        "strict_blind_owned": "#60a5fa",
        "control_support_strip": "#f43f5e",
    }
    for candidate in candidates:
        direction = candidate.get("axis_direction")
        anchor = _candidate_anchor(candidate)
        if not isinstance(direction, list) or len(direction) != 3 or anchor is None:
            continue
        alignment = abs(float(_unit(np.asarray(direction, dtype=np.float64)) @ axis))
        if alignment < 0.96:
            continue
        local_anchor = anchor - center
        candidate_axial = float(local_anchor @ axis)
        if abs(candidate_axial - axial_center) > half_span + 4.0:
            continue
        position = np.asarray([local_anchor @ u, local_anchor @ v])
        source = str(candidate.get("candidate_source", "candidate"))
        candidate_id = str(candidate.get("candidate_id", "?"))
        short_id = candidate_id.split("::")[-1]
        color = palette.get(source, "#22c55e")
        for ax in axes:
            ax.scatter(
                position[0], position[1], s=72, color=color, edgecolor="white", zorder=5
            )
            ax.text(
                position[0] + 0.8,
                position[1] + 0.8,
                short_id,
                fontsize=8,
                color="white",
                fontweight="bold",
                bbox={"facecolor": color, "edgecolor": "white", "pad": 1.1},
                zorder=6,
            )

    for ax in axes:
        ax.set_xlabel("profile U (mm)")
        ax.set_ylabel("profile V (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.16)
    fig.suptitle(
        f"{args.title}\naxial center={axial_center:.1f} mm; "
        f"half-width={half_span:.1f} mm; "
        f"axis={','.join(f'{value:.3f}' for value in axis)}",
        fontsize=17,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, facecolor="white")
    plt.close(fig)
    print(args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
