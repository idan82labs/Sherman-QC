#!/usr/bin/env python3
"""Render retained versus spatial-fold-rejected V51 predictions in six views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_source_only_prediction_audit import _limits, _points
from scripts.render_manual_bend_review import _stable_pca, _view_basis


def render(geometry: Path, parent_path: Path, candidate_path: Path, output: Path) -> dict:
    points = _points(geometry)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    kept_ids = {str(row["bend_id"]) for row in candidate["predictions"]}
    retained = [row for row in parent["predictions"] if str(row["bend_id"]) in kept_ids]
    removed = [row for row in parent["predictions"] if str(row["bend_id"]) not in kept_ids]
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    take = np.linspace(0, len(projected) - 1, min(len(projected), 250_000), dtype=np.int64)
    shown = projected[take]
    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=float)),
        ("PC2 / PC3", np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float)),
        ("Oblique", _view_basis(np.asarray((1.0, 1.0, 0.75)), np.asarray((0.0, 0.0, 1.0)))),
        ("Reverse", _view_basis(np.asarray((-1.0, 0.8, 0.65)), np.asarray((0.0, 0.0, 1.0)))),
        ("Underside", _view_basis(np.asarray((0.8, -1.0, -0.7)), np.asarray((0.0, 0.0, 1.0)))),
    )
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=220, facecolor="#08111f")
    fig.suptitle(
        f"EXP_SCAN_0001 — {geometry.parent.name} — cyan retained / red rejected",
        color="white", fontsize=20, fontweight="bold", y=0.985,
    )
    for axis, (title, view) in zip(axes.flat, views):
        cloud = shown @ view
        depth = cloud[:, 2]
        shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        axis.scatter(cloud[:, 0], cloud[:, 1], c=plt.get_cmap("bone")(0.15 + 0.50 * shade), s=0.5, alpha=0.48, linewidths=0, rasterized=True)
        for rows, color, style, width in ((retained, "#22d3ee", "-", 2.4), (removed, "#fb3b4b", "--", 4.0)):
            for index, row in enumerate(rows, 1):
                line = (np.asarray([row["line_start"], row["line_end"]], dtype=float) - center) @ pca @ view
                axis.plot(line[:, 0], line[:, 1], color="#020617", linewidth=width + 3.0, linestyle=style, zorder=6)
                axis.plot(line[:, 0], line[:, 1], color=color, linewidth=width, linestyle=style, zorder=7)
                if rows is removed:
                    midpoint = np.mean(line, axis=0)
                    axis.text(midpoint[0], midpoint[1], f"R{index}", color="white", fontsize=8, fontweight="bold", ha="center", va="center", bbox={"boxstyle":"round,pad=0.15","facecolor":"#b91c1c","edgecolor":"white"}, zorder=8)
        _limits(axis, projected @ view)
        axis.set_title(title, color="white", fontsize=13, fontweight="bold")
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        axis.set_facecolor("#08111f")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02, hspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
    plt.close(fig)
    sidecar = {
        "case_id": geometry.parent.name,
        "retained_count": len(retained),
        "removed_count": len(removed),
        "removed_bend_ids": [row["bend_id"] for row in removed],
        "truth_opened": False,
        "visual_ruling": "PENDING_MANUAL_REVIEW",
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.geometry.resolve(), args.parent.resolve(), args.candidate.resolve(), args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
