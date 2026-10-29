#!/usr/bin/env python3
"""Render arbitrary scored synthetic cases for post-score visual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROUTE_COLORS = {
    "calibrated_per_view_segment_consensus_v60": "#00bcd4",
    "calibrated_birth_ownership_guard_v61": "#00bcd4",
    "calibrated_distributed_turn_consensus_v62": "#00bcd4",
}


def _segment(row: dict) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=float)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--score-key", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("scored visual root is not empty")
    score = json.loads(args.score.read_text(encoding="utf-8"))
    score_rows = score[args.score_key]
    cases = score_rows["cases"] if isinstance(score_rows, dict) else score_rows
    scores = {row["case_id"]: row for row in cases}
    records = []

    for case_id in args.case_id:
        source = np.asarray(
            o3d.io.read_point_cloud(
                str(args.corpus_root / "public" / case_id / "source.ply")
            ).points,
            dtype=float,
        )
        if len(source) > 100_000:
            source = source[np.linspace(0, len(source) - 1, 100_000, dtype=int)]
        truth = json.loads(
            (args.corpus_root / "evaluator_only" / case_id / "truth.json").read_text(
                encoding="utf-8"
            )
        )
        prediction = json.loads(
            (args.prediction_root / case_id / "semantic_result.json").read_text(
                encoding="utf-8"
            )
        )
        row = scores[case_id]
        matched_gold = {match["gold_id"] for match in row["matches"]}
        matched_predictions = {
            match["prediction_id"] for match in row["matches"]
        }

        fig = plt.figure(figsize=(16, 9), dpi=256)
        views = ((24, -58, "oblique"), (90, -90, "top"), (8, -90, "side"))
        for panel, (elevation, azimuth, title) in enumerate(views, 1):
            axis = fig.add_subplot(1, 3, panel, projection="3d")
            axis.scatter(
                source[:, 0],
                source[:, 1],
                source[:, 2],
                s=0.08,
                c="#b7bec8",
                alpha=0.30,
                rasterized=True,
            )
            for bend in truth["manual_bends"]:
                line = _segment(bend)
                color = "#111111" if bend["bend_id"] in matched_gold else "#d81b60"
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=2.4)
            for bend in prediction["predictions"]:
                line = _segment(bend)
                matched = bend["bend_id"] in matched_predictions
                route = bend.get("selection_route", "")
                if route in ROUTE_COLORS:
                    color = ROUTE_COLORS[route] if matched else "#8e24aa"
                else:
                    color = "#2f855a" if matched else "#ff5722"
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=4.0)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(title)
            axis.set_axis_off()
            axis.set_box_aspect(np.maximum(np.ptp(source, axis=0), 1e-6))

        fig.suptitle(
            f"{case_id} {args.score_key} | TP {row['true_positive']} "
            f"FP {row['false_positive']} FN {row['false_negative']} | "
            "black matched truth, magenta missed truth, green/cyan TP, orange/purple FP",
            fontsize=13,
        )
        output = args.output_root / f"{case_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        records.append(
            {
                "case_id": case_id,
                "image": str(output.resolve()),
                "image_sha256": _sha(output),
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "truth_opened": True,
                "score_key": args.score_key,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(records), "score_key": args.score_key}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
