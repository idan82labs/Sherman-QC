#!/usr/bin/env python3
"""Render native-resolution multiview audits for source-only F1 predictions."""

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


def _points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    values = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(values):
        cloud = o3d.io.read_point_cloud(str(path))
        values = np.asarray(cloud.points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        raise ValueError(f"no finite geometry in {path}")
    return values


def _limits(ax: plt.Axes, values: np.ndarray) -> None:
    lo, hi = values.min(axis=0), values.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - 0.04 * span[0], hi[0] + 0.04 * span[0])
    ax.set_ylim(lo[1] - 0.04 * span[1], hi[1] + 0.04 * span[1])


def render(
    *, geometry: Path, prediction_path: Path, output: Path, max_points: int
) -> dict[str, object]:
    points = _points(geometry)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = list(payload.get("predictions") or [])
    audit_mode = "CAD-GUIDED" if payload.get("cad_guided_zero_shot") is True else "SOURCE-ONLY"
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    take = np.linspace(
        0, len(projected) - 1, min(len(projected), max_points), dtype=np.int64
    )
    shown = projected[take]
    lines = []
    for index, row in enumerate(predictions, start=1):
        endpoints = np.asarray(
            (row["line_start"], row["line_end"]), dtype=np.float64
        )
        lines.append(
            (
                index,
                str(row.get("source_hypothesis_id") or row.get("bend_id") or index),
                (endpoints - center) @ pca,
            )
        )

    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=float)),
        ("PC2 / PC3", np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float)),
        ("Oblique", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, max(len(lines), 1)))
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=220, facecolor="#f8fafc")
    fig.suptitle(
        f"F1 {audit_mode} VISUAL AUDIT — {geometry.name} — {len(lines)} predicted bends",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.954,
        f"route: {payload.get('route', 'unknown')}  •  full unshortened axes  •  labels are visual indices",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    for ax, (title, view) in zip(axes.flat, views):
        cloud = shown @ view
        depth = cloud[:, 2]
        shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            c=plt.get_cmap("Greys")(0.25 + 0.52 * shade),
            s=0.55,
            alpha=0.56,
            linewidths=0,
            rasterized=True,
        )
        for color, (index, _source_id, line) in zip(colors, lines):
            line2 = line @ view
            ax.plot(
                line2[:, 0], line2[:, 1], color="white", linewidth=8.0,
                solid_capstyle="round", zorder=7,
            )
            ax.plot(
                line2[:, 0], line2[:, 1], color=color, linewidth=4.0,
                solid_capstyle="round", zorder=8,
            )
            ax.scatter(
                line2[:, 0], line2[:, 1], s=28, facecolor="white",
                edgecolor=color, linewidth=1.8, zorder=9,
            )
            midpoint = line2.mean(axis=0)
            ax.text(
                midpoint[0], midpoint[1], str(index), fontsize=8,
                fontweight="bold", ha="center", va="center", zorder=10,
                bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 1.2},
            )
        _limits(ax, projected @ view)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_facecolor("#f8fafc")
    fig.subplots_adjust(
        left=0.012, right=0.988, top=0.925, bottom=0.015, wspace=0.018, hspace=0.07
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    return {
        "geometry": str(geometry),
        "prediction": str(prediction_path),
        "output": str(output),
        "prediction_count": len(lines),
        "route": payload.get("route"),
        "visual_index_to_source": {
            str(index): source_id for index, source_id, _line in lines
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=250_000)
    args = parser.parse_args()
    result = render(
        geometry=args.geometry.resolve(),
        prediction_path=args.prediction.resolve(),
        output=args.output.resolve(),
        max_points=args.max_points,
    )
    metadata = args.output.resolve().with_suffix(".json")
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
