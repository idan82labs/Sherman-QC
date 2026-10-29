#!/usr/bin/env python3
"""Render clean, human-readable 3D bend overlays for manual visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_manual_bend_review import _stable_pca, _view_basis


def _load_points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError(f"No finite geometry points in {path}")
    return points


def _sample(points: np.ndarray, budget: int = 70000) -> np.ndarray:
    if len(points) <= budget:
        return points
    indices = np.linspace(0, len(points) - 1, budget, dtype=np.int64)
    return points[indices]


def _tight_limits(ax, points_2d: np.ndarray, pad: float = 0.06) -> None:
    lo = points_2d.min(axis=0)
    hi = points_2d.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - span[0] * pad, hi[0] + span[0] * pad)
    ax.set_ylim(lo[1] - span[1] * pad, hi[1] + span[1] * pad)


def render(*, geometry: Path, precommit: Path, scan_id: str, output: Path, role: str) -> None:
    points = _load_points(geometry)
    payload = json.loads(precommit.read_text(encoding="utf-8"))
    scan = next(row for row in payload["scans"] if row["opaque_scan_id"] == scan_id)
    bends = scan["manual_bends"]

    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    shown = _sample(projected)
    lines = []
    for row in bends:
        endpoints = np.asarray((row["line_start"], row["line_end"]), dtype=np.float64)
        lines.append((str(row["bend_id"]), (endpoints - center) @ pca))

    views = (
        ("Oblique overview", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Top", np.eye(3)),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=220, facecolor="#f6f7f9")
    role_label = role.upper()
    fig.suptitle(
        f"{role_label} SCAN — {len(lines)} SOURCE-ONLY PROPOSED BEND LINES",
        fontsize=21,
        fontweight="bold",
        color="#111827",
        y=0.978,
    )
    fig.text(
        0.5,
        0.932,
        "Gray = scanned surface   •   Orange = proposed bend axis   •   Numbers are visual references",
        ha="center",
        va="center",
        fontsize=12,
        color="#4b5563",
    )

    for view_index, (ax, (title, view)) in enumerate(zip(axes.flat, views)):
        view_points = shown @ view
        depth = view_points[:, 2]
        depth_norm = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        colors = plt.get_cmap("Greys")(0.30 + 0.42 * depth_norm)
        ax.scatter(
            view_points[:, 0],
            view_points[:, 1],
            c=colors,
            s=0.75,
            alpha=0.52,
            linewidths=0,
            rasterized=True,
        )
        for number, (_bend_id, endpoints) in enumerate(lines, start=1):
            line = endpoints @ view
            ax.plot(
                line[:, 0], line[:, 1],
                color="#ffffff", linewidth=8.5, alpha=0.96, solid_capstyle="round", zorder=8,
            )
            ax.plot(
                line[:, 0], line[:, 1],
                color="#f97316", linewidth=4.8, alpha=1.0, solid_capstyle="round", zorder=9,
            )
        full_view = projected @ view
        _tight_limits(ax, full_view)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#111827", pad=8)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#f6f7f9")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.895, bottom=0.025, wspace=0.035, hspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--role", choices=("good", "bad"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        geometry=args.geometry.resolve(),
        precommit=args.precommit.resolve(),
        scan_id=args.scan_id,
        output=args.output.resolve(),
        role=args.role,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
