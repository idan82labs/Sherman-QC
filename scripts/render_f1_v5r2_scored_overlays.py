#!/usr/bin/env python3
"""Render all consumed V5R2 cases after scoring for manual visual audit."""

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("visual output root is not empty")

    score = json.loads(args.score.read_text(encoding="utf-8"))
    score_by_case = {row["case_id"]: row for row in score["v62"]["cases"]}
    records = []
    for case_id in sorted(score_by_case):
        source_path = args.corpus_root / "public" / case_id / "source.ply"
        truth_path = args.corpus_root / "evaluator_only" / case_id / "truth.json"
        prediction_path = args.prediction_root / case_id / "semantic_result.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["manual_bends"]
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))["predictions"]
        scored = score_by_case[case_id]
        matched_prediction_ids = {row["prediction_id"] for row in scored["matches"]}
        points = np.asarray(o3d.io.read_point_cloud(str(source_path)).points, dtype=np.float64)
        if len(points) > 120_000:
            points = points[np.linspace(0, len(points) - 1, 120_000, dtype=int)]

        figure = plt.figure(figsize=(18, 10), dpi=240, facecolor="#f8fafc")
        views = (
            (24, -58, "oblique"), (90, -90, "top"), (8, -90, "side"),
            (20, 32, "reverse oblique"), (2, 0, "end"), (-16, -58, "underside"),
        )
        for panel, (elevation, azimuth, title) in enumerate(views, 1):
            axis = figure.add_subplot(2, 3, panel, projection="3d")
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2], s=0.08,
                c="#aeb8c4", alpha=0.32, rasterized=True,
            )
            for bend in truth:
                line = _line(bend)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#7e22ce", linestyle="--", linewidth=2.5)
            for index, prediction in enumerate(predictions, start=1):
                matched = prediction["bend_id"] in matched_prediction_ids
                color = "#16a34a" if matched else "#ef4444"
                line = _line(prediction)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="white", linewidth=7.0)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=4.0)
                midpoint = line.mean(axis=0)
                axis.text(midpoint[0], midpoint[1], midpoint[2], f"P{index}", fontsize=8, fontweight="bold", color=color)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_axis_off()
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
        figure.suptitle(
            f"{case_id} V52=V62 | purple dashed truth | green TP | red FP | "
            f"TP {scored['true_positive']} / FP {scored['false_positive']} / "
            f"FN {scored['false_negative']}",
            fontsize=16, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        output = args.output_root / f"{case_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight", pad_inches=0.05, facecolor=figure.get_facecolor())
        plt.close(figure)
        records.append(
            {
                "case_id": case_id,
                "tp": scored["true_positive"],
                "fp": scored["false_positive"],
                "fn": scored["false_negative"],
                "image": str(output.resolve()), "image_sha256": _sha(output),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v5r2_consumed_pilot_scored_visual_audit",
        "truth_opened": True,
        "manual_visual_audit_required": True,
        "records": records,
    }
    path = args.output_root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(records), "tp": sum(r["tp"] for r in records), "fp": sum(r["fp"] for r in records), "fn": sum(r["fn"] for r in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
