#!/usr/bin/env python3
"""Render native-resolution truth/prediction overlays after score sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_f1_against_manual_gold import Thresholds, _endpoints, match_lines
from scripts.render_f1_v51_new_corpus_prediction_only import VIEWS
from scripts.render_manual_bend_review import _points, _stable_pca


def render(scan: Path, prediction: Path, truth: Path, output: Path, case_id: str) -> dict:
    points = _points(scan, 200_000)
    center, pca = _stable_pca(points)
    projected = (points - center) @ pca
    prediction_rows = list(json.loads(prediction.read_text())["predictions"])
    truth_payload = json.loads(truth.read_text())
    truth_rows = next(row["manual_bends"] for row in truth_payload["cases"] if row["case_id"] == case_id)
    matched = match_lines(truth_rows, prediction_rows, Thresholds())
    matched_gold = {row["gold_id"] for row in matched["matches"]}
    matched_pred = {row["prediction_id"] for row in matched["matches"]}
    gold_lines = [(str(row["bend_id"]), (_endpoints(row) - center) @ pca) for row in truth_rows]
    pred_lines = [(str(row["bend_id"]), (_endpoints(row) - center) @ pca) for row in prediction_rows]
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), dpi=180)
    fig.patch.set_facecolor("#07101f")
    for ax, (title, view) in zip(axes.flat, VIEWS):
        view_points = projected @ view
        depth = view_points[:, 2]
        order = np.argsort(depth)
        normalized = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)
        ax.scatter(view_points[order, 0], view_points[order, 1], s=0.26,
                   c=plt.get_cmap("bone")(0.24 + 0.64 * normalized)[order],
                   alpha=0.38, linewidths=0, rasterized=True)
        for bend_id, line in gold_lines:
            shown = line @ view
            color = "#39ff88" if bend_id in matched_gold else "#ff9f1c"
            ax.plot(shown[:, 0], shown[:, 1], color="black", linewidth=9.0, alpha=0.78)
            ax.plot(shown[:, 0], shown[:, 1], color=color, linewidth=6.0, alpha=0.94)
        for bend_id, line in pred_lines:
            shown = line @ view
            color = "#42d4ff" if bend_id in matched_pred else "#ff3b6b"
            ax.plot(shown[:, 0], shown[:, 1], color=color, linewidth=2.8, linestyle=(0, (5, 2)))
        ax.set_title(title, color="white", fontsize=15, fontweight="bold")
        ax.set_facecolor("#07101f")
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    status = f"TP {matched['true_positive']}  FP {matched['false_positive']}  FN {matched['false_negative']}"
    fig.suptitle(f"{case_id} — truth solid / V51 dashed — {status}", color="white", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.scan.resolve(), args.prediction.resolve(), args.truth.resolve(), args.output.resolve(), args.case_id)
    print(json.dumps({key: result[key] for key in ("true_positive", "false_positive", "false_negative", "exact")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
