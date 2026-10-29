#!/usr/bin/env python3
"""Render counted bends and explicit non-counted objects from a typed F1 graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_line_audit import _stable_pca


PALETTE = {
    "BEND_STRAIGHT_COUNTED": ("#16a34a", "-", 4.2),
    "CORNER_NONCOUNTED": ("#dc2626", "--", 3.2),
    "PERIMETER_NONCOUNTED": ("#ea580c", ":", 3.0),
    "HOLE_NONCOUNTED": ("#7c3aed", ":", 3.0),
    "JUNCTION_NONCOUNTED": ("#db2777", "--", 3.0),
    "CURVE_NONCOUNTED": ("#2563eb", "--", 3.0),
}


def _points(path: Path, max_points: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    step = max(1, int(np.ceil(len(points) / max(1, max_points))))
    return points[::step]


def render(graph_path: Path, scan_path: Path, output_path: Path, max_points: int) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    points = _points(scan_path, max_points)
    center, basis = _stable_pca(points)
    projected = (points - center) @ basis
    objects = [
        item
        for item in graph.get("latent_paths") or ()
        if str(item.get("semantic_label")) in PALETTE
    ]
    objects.sort(
        key=lambda item: (
            str(item["semantic_label"]),
            tuple(item.get("primary_panel_pair") or ()),
        )
    )
    lines = []
    counters: dict[str, int] = {}
    for item in objects:
        label = str(item["semantic_label"])
        prefix = "B" if label == "BEND_STRAIGHT_COUNTED" else "C"
        counters[prefix] = counters.get(prefix, 0) + 1
        pair = "-".join(item.get("primary_panel_pair") or ())
        display = f"{prefix}{counters[prefix]} {pair}"
        endpoints = (np.asarray(item["span_endpoints"], dtype=np.float64) - center) @ basis
        lines.append((display, label, endpoints))

    projections = ((0, 1, "Topological face view"), (0, 2, "Side view A"), (1, 2, "Side view B"))
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), dpi=120)
    fig.patch.set_facecolor("white")
    for axis, (left, right, title) in zip(axes, projections):
        axis.scatter(
            projected[:, left],
            projected[:, right],
            s=0.16,
            c="#64748b",
            alpha=0.16,
            linewidths=0,
        )
        for display, semantic, endpoints in lines:
            color, style, width = PALETTE[semantic]
            axis.plot(
                endpoints[:, left],
                endpoints[:, right],
                color=color,
                linestyle=style,
                linewidth=width,
                solid_capstyle="round",
                dash_capstyle="round",
            )
            midpoint = np.mean(endpoints, axis=0)
            axis.text(
                midpoint[left],
                midpoint[right],
                display,
                fontsize=7.2,
                fontweight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.92,
                },
            )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_aspect("equal", adjustable="datalim")
        axis.axis("off")
    legend = [
        Line2D([0], [0], color="#16a34a", linewidth=4, label="Counted bend"),
        Line2D([0], [0], color="#dc2626", linewidth=3, linestyle="--", label="Explicit non-counted corner"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(
        f"Structured F1 audit — {sum(label == 'BEND_STRAIGHT_COUNTED' for _, label, _ in lines)} counted bends + "
        f"{sum(label == 'CORNER_NONCOUNTED' for _, label, _ in lines)} non-counted corners",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--scan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()
    render(args.graph.resolve(), args.scan.resolve(), args.output.resolve(), args.max_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
