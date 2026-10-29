#!/usr/bin/env python3
"""Render EXP_SCAN_0003 additions against the retained V51 parent."""

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
from scripts.run_f1_full_pool_relational_lofo_v1 import _duplicate


def render(geometry: Path, baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    points = _points(geometry)
    baseline = json.loads(baseline_path.read_text())["predictions"]
    candidate = json.loads(candidate_path.read_text())["predictions"]
    used = set(); common = []; removed = []
    for parent in baseline:
        match = next((index for index, row in enumerate(candidate) if index not in used and _duplicate(parent, row)), None)
        if match is None:
            removed.append(parent)
        else:
            used.add(match); common.append(candidate[match])
    additions = [row for index, row in enumerate(candidate) if index not in used]
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    take = np.linspace(0, len(projected) - 1, min(len(projected), 250_000), dtype=np.int64)
    shown = projected[take]
    views = (
        ("PC1 / PC2", np.eye(3)),
        ("PC1 / PC3", np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), float)),
        ("PC2 / PC3", np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), float)),
        ("Oblique", _view_basis(np.asarray((1., 1., .75)), np.asarray((0., 0., 1.)))),
        ("Reverse", _view_basis(np.asarray((-1., .8, .65)), np.asarray((0., 0., 1.)))),
        ("Underside", _view_basis(np.asarray((.8, -1., -.7)), np.asarray((0., 0., 1.)))),
    )
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=220, facecolor="#08111f")
    fig.suptitle(f"EXP_SCAN_0003 — {geometry.parent.name} — cyan parent / lime ownership addition", color="white", fontsize=20, fontweight="bold", y=.985)
    styles = ((common, "#22d3ee", "", 2.4), (removed, "#fb3b4b", "R", 3.8), (additions, "#84cc16", "A", 3.8))
    for axis, (title, view) in zip(axes.flat, views):
        cloud = shown @ view; depth = cloud[:, 2]
        shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        axis.scatter(cloud[:, 0], cloud[:, 1], c=plt.get_cmap("bone")(.15 + .5 * shade), s=.5, alpha=.48, linewidths=0, rasterized=True)
        for rows, color, prefix, width in styles:
            for index, row in enumerate(rows, 1):
                line = (np.asarray([row["line_start"], row["line_end"]], float) - center) @ pca @ view
                axis.plot(line[:, 0], line[:, 1], color="#020617", linewidth=width + 3, zorder=6)
                axis.plot(line[:, 0], line[:, 1], color=color, linewidth=width, zorder=7)
                if prefix:
                    midpoint = np.mean(line, axis=0)
                    axis.text(midpoint[0], midpoint[1], f"{prefix}{index}", color="white", fontsize=7, fontweight="bold", ha="center", va="center", bbox={"boxstyle": "round,pad=.15", "facecolor": "#365314" if prefix == "A" else "#4c1d1d", "edgecolor": "white"}, zorder=8)
        _limits(axis, projected @ view)
        axis.set_title(title, color="white", fontsize=13, fontweight="bold")
        axis.set_aspect("equal", adjustable="box"); axis.set_axis_off(); axis.set_facecolor("#08111f")
    fig.subplots_adjust(left=.01, right=.99, top=.94, bottom=.01, wspace=.02, hspace=.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=.08, facecolor=fig.get_facecolor()); plt.close(fig)
    sidecar = {"case_id": output.stem, "common_count": len(common), "removed_count": len(removed), "added_count": len(additions), "removed_ids": [row["bend_id"] for row in removed], "added_ids": [row["bend_id"] for row in additions], "visual_ruling": "PENDING_MANUAL_REVIEW"}
    output.with_suffix(".json").write_text(json.dumps(sidecar, sort_keys=True, separators=(",", ":")) + "\n")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True); parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(render(args.geometry.resolve(), args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()), sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

