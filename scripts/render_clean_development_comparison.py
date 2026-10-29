#!/usr/bin/env python3
"""Render a clean prediction-versus-trusted-bend comparison on a development case."""

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
        raise ValueError(f"No finite points in {path}")
    return points


def _sample(points: np.ndarray, budget: int = 80000) -> np.ndarray:
    if len(points) <= budget:
        return points
    return points[np.linspace(0, len(points) - 1, budget, dtype=np.int64)]


def _endpoints(row: dict) -> np.ndarray:
    if row.get("span_endpoints") is not None:
        return np.asarray(row["span_endpoints"], dtype=np.float64)
    if row.get("line_start") is not None and row.get("line_end") is not None:
        return np.asarray((row["line_start"], row["line_end"]), dtype=np.float64)
    raise ValueError(f"No endpoints in row {row.get('bend_id')}")


def _tight_limits(ax, points: np.ndarray, pad: float = 0.055) -> None:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - span[0] * pad, hi[0] + span[0] * pad)
    ax.set_ylim(lo[1] - span[1] * pad, hi[1] + span[1] * pad)


def render(
    *,
    catalog: Path,
    case_id: str,
    prediction: Path,
    output: Path,
    title: str,
    prediction_label: str,
) -> None:
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    case = next(row for row in catalog_payload["cases"] if row["case_id"] == case_id)
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    predicted_rows = prediction_payload.get("owned_bend_regions") or prediction_payload.get("predictions") or []

    geometry_path = Path(case["scan_path"])
    points = _load_points(geometry_path)
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    shown = _sample(projected)
    trusted = [(_endpoints(row) - center) @ pca for row in case["bends"]]
    predicted = [(_endpoints(row) - center) @ pca for row in predicted_rows]

    views = (
        ("Oblique overview", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Top", np.eye(3)),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=220, facecolor="#f6f7f9")
    fig.suptitle(title, fontsize=21, fontweight="bold", color="#111827", y=0.978)
    fig.text(
        0.5,
        0.932,
        f"Gray = scan   •   Cyan = trusted bend ({len(trusted)})   •   Orange dashed = {prediction_label} ({len(predicted)})",
        ha="center",
        fontsize=12,
        color="#4b5563",
    )

    for ax, (view_title, view) in zip(axes.flat, views):
        view_points = shown @ view
        depth = view_points[:, 2]
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        colors = plt.get_cmap("Greys")(0.31 + 0.41 * normalized)
        ax.scatter(view_points[:, 0], view_points[:, 1], s=0.72, c=colors, alpha=0.52, linewidths=0, rasterized=True)
        for line3d in trusted:
            line = line3d @ view
            ax.plot(line[:, 0], line[:, 1], color="white", linewidth=9.0, solid_capstyle="round", zorder=5)
            ax.plot(line[:, 0], line[:, 1], color="#0891b2", linewidth=5.4, solid_capstyle="round", zorder=6)
        for line3d in predicted:
            line = line3d @ view
            ax.plot(line[:, 0], line[:, 1], color="white", linewidth=5.0, linestyle=(0, (5, 3)), solid_capstyle="round", zorder=7)
            ax.plot(line[:, 0], line[:, 1], color="#f97316", linewidth=2.5, linestyle=(0, (5, 3)), solid_capstyle="round", zorder=8)
        full_view = projected @ view
        _tight_limits(ax, full_view)
        ax.set_title(view_title, fontsize=14, fontweight="bold", color="#111827", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_facecolor("#f6f7f9")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.895, bottom=0.025, wspace=0.035, hspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--prediction-label", default="prediction")
    args = parser.parse_args()
    render(
        catalog=args.catalog.resolve(),
        case_id=args.case_id,
        prediction=args.prediction.resolve(),
        output=args.output.resolve(),
        title=args.title,
        prediction_label=args.prediction_label,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
