#!/usr/bin/env python3
"""Render a native-resolution V51 prediction-only sheet before truth opening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.render_manual_bend_review import _points, _stable_pca, _view_basis


VIEWS = (
    ("PC1 / PC2", np.eye(3)),
    ("PC1 / PC3", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)),
    ("PC2 / PC3", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)),
    ("Oblique front", _view_basis(np.array([1.0, 1.0, 0.75]), np.array([0.0, 0.0, 1.0]))),
    ("Oblique reverse", _view_basis(np.array([-1.0, 0.8, 0.65]), np.array([0.0, 0.0, 1.0]))),
    ("Oblique underside", _view_basis(np.array([0.8, -1.0, -0.7]), np.array([0.0, 0.0, 1.0]))),
)


def _line(record: dict, center: np.ndarray, pca: np.ndarray) -> np.ndarray:
    endpoints = np.asarray([record["line_start"], record["line_end"]], dtype=float)
    return (endpoints - center) @ pca


def render(scan: Path, prediction: Path, output: Path, case_id: str) -> None:
    points = _points(scan, 200_000)
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    if payload.get("source_only") is not True:
        raise ValueError("prediction is not source-only")
    lines = [_line(row, center, pca) for row in payload.get("predictions") or []]
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=180)
    fig.patch.set_facecolor("#07101f")
    for ax, (title, view) in zip(axes.flat, VIEWS):
        view_points = projected @ view
        depth = view_points[:, 2]
        order = np.argsort(depth)
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        ax.scatter(
            view_points[order, 0], view_points[order, 1], s=0.26,
            c=plt.get_cmap("bone")(0.24 + 0.64 * normalized)[order],
            alpha=0.38, linewidths=0, rasterized=True,
        )
        for line in lines:
            shown = line @ view
            ax.plot(shown[:, 0], shown[:, 1], color="black", linewidth=8.0, alpha=0.82)
            ax.plot(shown[:, 0], shown[:, 1], color="#42d4ff", linewidth=4.0, alpha=0.98)
        ax.set_title(title, color="white", fontsize=15, fontweight="bold")
        ax.set_facecolor("#07101f")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    fig.suptitle(f"{case_id} — frozen V51 prediction only — {len(lines)} lines", color="white", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.scan.resolve(), args.prediction.resolve(), args.output.resolve(), args.case_id)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
