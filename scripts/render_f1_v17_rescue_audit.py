#!/usr/bin/env python3
"""Render native-resolution four-view V17 rescue audits."""

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

from research.f1_multifamily_selector_v2.model import endpoints
from scripts.render_manual_bend_review import _stable_pca, _view_basis


def _points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    values = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(values):
        cloud = o3d.io.read_point_cloud(str(path))
        values = np.asarray(cloud.points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        raise ValueError(f"no finite scan geometry in {path}")
    return values


def _limits(ax: plt.Axes, values: np.ndarray) -> None:
    lo, hi = values.min(axis=0), values.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - 0.05 * span[0], hi[0] + 0.05 * span[0])
    ax.set_ylim(lo[1] - 0.05 * span[1], hi[1] + 0.05 * span[1])


def render(
    *, geometry: Path, pool_path: Path, v16_path: Path, v17_path: Path, output: Path
) -> None:
    points = _points(geometry)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    v16 = json.loads(v16_path.read_text(encoding="utf-8"))
    v17 = json.loads(v17_path.read_text(encoding="utf-8"))
    row_by_id = {str(row["candidate_id"]): row for row in pool["line_objects"]}
    incumbent_ids = [str(row["source_hypothesis_id"]) for row in v16["predictions"]]
    rescue_ids = [
        str(row["source_hypothesis_id"])
        for row in v17["predictions"]
        if row.get("selection_route")
        == "full_resolution_mesh_v11_axial_hysteresis_rescue_v17"
    ]
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    shown = projected[
        np.linspace(0, len(projected) - 1, min(len(projected), 100000), dtype=np.int64)
    ]
    incumbents = [(value, (endpoints(row_by_id[value]) - center) @ pca) for value in incumbent_ids]
    rescues = [(value, (endpoints(row_by_id[value]) - center) @ pca) for value in rescue_ids]
    views = (
        ("Oblique", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Top", np.eye(3)),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=240, facecolor="#f8fafc")
    fig.suptitle(
        f"V17 U02 TERMINAL RESCUE AUDIT — {len(incumbents)} V16 INCUMBENTS + {len(rescues)} RESCUES",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.942,
        "Blue = frozen V16 axis   •   Red = V17 appended axis   •   circles = unshortened endpoints",
        ha="center",
        fontsize=12,
        color="#334155",
    )
    for ax, (title, view) in zip(axes.flat, views):
        cloud = shown @ view
        depth = cloud[:, 2]
        shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        ax.scatter(
            cloud[:, 0], cloud[:, 1], c=plt.get_cmap("Greys")(0.27 + 0.48 * shade),
            s=0.7, alpha=0.58, linewidths=0, rasterized=True,
        )
        for candidate_id, line in incumbents:
            line2 = line @ view
            ax.plot(line2[:, 0], line2[:, 1], color="white", linewidth=7, zorder=7)
            ax.plot(line2[:, 0], line2[:, 1], color="#2563eb", linewidth=3.4, zorder=8)
        for candidate_id, line in rescues:
            line2 = line @ view
            ax.plot(line2[:, 0], line2[:, 1], color="white", linewidth=9, zorder=9)
            ax.plot(line2[:, 0], line2[:, 1], color="#dc2626", linewidth=4.8, zorder=10)
            ax.scatter(line2[:, 0], line2[:, 1], s=62, facecolor="#fef2f2", edgecolor="#dc2626", linewidth=2.2, zorder=11)
            midpoint = line2.mean(axis=0)
            ax.text(midpoint[0], midpoint[1], candidate_id.rsplit("::", 1)[-1], fontsize=9, fontweight="bold", color="#991b1b", zorder=12)
        _limits(ax, projected @ view)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_facecolor("#f8fafc")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.905, bottom=0.02, wspace=0.02, hspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--v16", type=Path, required=True)
    parser.add_argument("--v17", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        geometry=args.geometry,
        pool_path=args.pool,
        v16_path=args.v16,
        v17_path=args.v17,
        output=args.output,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
