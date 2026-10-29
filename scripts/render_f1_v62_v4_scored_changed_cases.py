#!/usr/bin/env python3
"""Render post-score overlays for every V62-changed prospective V4 case."""

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


ROUTE = "calibrated_distributed_turn_consensus_v62"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v61-root", type=Path, required=True)
    parser.add_argument("--v62-root", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("visual output root is not empty")

    score = json.loads(args.score.read_text(encoding="utf-8"))
    birth_match = {row["prediction_id"]: bool(row["matched"]) for row in score["v62_births"]}
    records = []
    for row in score["v62_births"]:
        case_id = row["case_id"]
        if any(record["case_id"] == case_id for record in records):
            continue
        source_path = args.corpus_root / "public" / case_id / "source.ply"
        truth_path = args.corpus_root / "evaluator_only" / case_id / "truth.json"
        v61_path = args.v61_root / case_id / "semantic_result.json"
        v62_path = args.v62_root / case_id / "semantic_result.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["manual_bends"]
        v61 = json.loads(v61_path.read_text(encoding="utf-8"))
        v62 = json.loads(v62_path.read_text(encoding="utf-8"))
        births = [prediction for prediction in v62["predictions"] if prediction.get("selection_route") == ROUTE]
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
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.08, c="#b8c0ca", alpha=0.28, rasterized=True)
            for bend in truth:
                line = _line(bend)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#7e22ce", linestyle="--", linewidth=2.2)
            for prediction in v61["predictions"]:
                line = _line(prediction)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="white", linewidth=5.0)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="#16a34a", linewidth=2.5)
            for birth_index, prediction in enumerate(births, start=1):
                matched = birth_match[prediction["bend_id"]]
                color = "#06b6d4" if matched else "#ef4444"
                line = _line(prediction)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color="white", linewidth=7.0)
                axis.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=4.0)
                midpoint = line.mean(axis=0)
                axis.text(midpoint[0], midpoint[1], midpoint[2], f"B{birth_index}", fontsize=8, fontweight="bold", color=color)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_axis_off()
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
        tp = sum(birth_match[prediction["bend_id"]] for prediction in births)
        fp = len(births) - tp
        figure.suptitle(
            f"{case_id} SCORED V62 REVIEW | purple dashed truth | green V61 | cyan TP | red FP | births {tp} TP / {fp} FP",
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
                "birth_true_positive": tp,
                "birth_false_positive": fp,
                "image": str(output.resolve()),
                "image_sha256": _sha(output),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v62_v4_scored_changed_case_visual_audit",
        "truth_opened": True,
        "records": records,
    }
    path = args.output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(records), "birth_true_positive": sum(row["birth_true_positive"] for row in records), "birth_false_positive": sum(row["birth_false_positive"] for row in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
