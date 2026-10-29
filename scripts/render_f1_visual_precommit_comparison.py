#!/usr/bin/env python3
"""Render frozen manual axes and predictions together for native visual QA."""

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
        values = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) > 90_000:
        values = values[np.linspace(0, len(values) - 1, 90_000, dtype=np.int64)]
    return values


def _limits(ax, points: np.ndarray) -> None:
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    ax.set_xlim(lo[0] - 0.05 * span[0], hi[0] + 0.05 * span[0])
    ax.set_ylim(lo[1] - 0.05 * span[1], hi[1] + 0.05 * span[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = _points(args.geometry)
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    scan = next(row for row in precommit["scans"] if row["opaque_scan_id"] == args.scan_id)
    profile_label = str(prediction.get("profile") or "selector")
    version_label = "V11" if "V11" in profile_label else ("V10" if "V10" in profile_label else "MODEL")
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    manual = [
        (np.asarray((row["line_start"], row["line_end"]), dtype=np.float64) - center) @ pca
        for row in scan["manual_bends"]
    ]
    predicted = [
        (np.asarray((row["line_start"], row["line_end"]), dtype=np.float64) - center) @ pca
        for row in prediction["predictions"]
    ]
    views = (
        ("Oblique", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Top", np.eye(3)),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=220, facecolor="#f7f8fa")
    fig.suptitle(
        f"UNTOUCHED CONFIRMATION — {len(manual)} MANUAL vs {len(predicted)} {version_label} AXES",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    fig.text(0.5, 0.935, f"Gray scan  •  Cyan manual precommit  •  Orange dashed {version_label}", ha="center", fontsize=12, color="#475569")
    for ax, (title, view) in zip(axes.flat, views):
        shown = projected @ view
        depth = shown[:, 2]
        shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        ax.scatter(shown[:, 0], shown[:, 1], c=plt.get_cmap("Greys")(0.28 + 0.38 * shade), s=0.7, alpha=0.48, linewidths=0, rasterized=True)
        for line in manual:
            item = line @ view
            ax.plot(item[:, 0], item[:, 1], color="#06b6d4", linewidth=5.5, alpha=0.95, solid_capstyle="round", zorder=7)
        for line in predicted:
            item = line @ view
            ax.plot(item[:, 0], item[:, 1], color="#f97316", linewidth=3.4, linestyle=(0, (5, 3)), alpha=1.0, solid_capstyle="round", zorder=8)
        _limits(ax, shown)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.035, hspace=0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
