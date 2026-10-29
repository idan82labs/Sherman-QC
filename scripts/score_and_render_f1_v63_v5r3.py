#!/usr/bin/env python3
"""Open V5R3 truth once, score frozen V62/V63, and render every case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.score_f1_calibrated_raw_synthetic_v60 import _aggregate


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V5R3_CORPUS_CONDITIONED_LTCO_PROSPECTIVE"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([row["line_start"], row["line_end"]], dtype=np.float64)


def _draw(
    axis,
    points: np.ndarray,
    truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    score: dict[str, Any],
    *,
    elevation: float,
    azimuth: float,
    title: str,
) -> None:
    axis.scatter(points[:,0], points[:,1], points[:,2], s=0.08, c="#aeb8c4", alpha=0.30, rasterized=True)
    for bend in truth:
        line = _line(bend)
        axis.plot(line[:,0], line[:,1], line[:,2], color="#7e22ce", linestyle="--", linewidth=2.2)
    matched = {row["prediction_id"] for row in score["matches"]}
    for index, prediction in enumerate(predictions, 1):
        color = "#16a34a" if prediction["bend_id"] in matched else "#ef4444"
        line = _line(prediction)
        axis.plot(line[:,0], line[:,1], line[:,2], color="white", linewidth=6.0)
        axis.plot(line[:,0], line[:,1], line[:,2], color=color, linewidth=3.4)
        middle = np.mean(line, axis=0)
        axis.text(*middle, f"P{index}", color=color, fontsize=7, fontweight="bold")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_axis_off()
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--v63-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("exact V5R3 one-time truth-opening token required")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("score/render output is not empty")
    private = _read(args.corpus_root / "evaluator_only" / "private_manifest.json")
    if private.get("token") != TOKEN:
        raise ValueError("private manifest token mismatch")
    v63_execution = _read(args.v63_root / "execution_manifest.json")
    if not all(v63_execution.get("semantic_replay_identity", {}).values()):
        raise ValueError("V63 outputs were not frozen with replay identity")

    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    baseline_cases: list[dict[str, Any]] = []
    v63_cases: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for private_row in private["cases"]:
        case_id = str(private_row["opaque_case_id"])
        family_id = str(private_row["family_id"])
        # This is the single private-truth read for the case. Both models and
        # both visual rows consume the same in-memory object.
        truth_path = args.corpus_root / "evaluator_only" / case_id / "truth.json"
        truth = _read(truth_path)["manual_bends"]
        baseline_path = args.baseline_root / case_id / "semantic_result.json"
        v63_path = args.v63_root / "primary" / case_id / "semantic_result.json"
        baseline = _read(baseline_path)
        gated = _read(v63_path)
        baseline_score = match_lines(truth, baseline["predictions"], thresholds)
        v63_score = match_lines(truth, gated["predictions"], thresholds)
        baseline_score.update({"case_id":case_id,"family_id":family_id})
        v63_score.update({"case_id":case_id,"family_id":family_id})
        baseline_cases.append(baseline_score)
        v63_cases.append(v63_score)
        baseline_matched = {row["prediction_id"] for row in baseline_score["matches"]}
        gated_ids = {row["bend_id"] for row in gated["predictions"]}
        for prediction in baseline["predictions"]:
            if prediction["bend_id"] not in gated_ids:
                deletion_rows.append({
                    "case_id":case_id,
                    "prediction_id":prediction["bend_id"],
                    "selection_route":prediction.get("selection_route"),
                    "was_true_positive":prediction["bend_id"] in baseline_matched,
                    "line_start":prediction["line_start"],
                    "line_end":prediction["line_end"],
                })

        source_path = args.corpus_root / "public" / case_id / "source.ply"
        points = np.asarray(o3d.io.read_point_cloud(str(source_path)).points, dtype=np.float64)
        if len(points) > 120_000:
            points = points[np.linspace(0, len(points)-1, 120_000, dtype=np.int64)]
        figure = plt.figure(figsize=(18, 11), dpi=240, facecolor="#f8fafc")
        views = ((24,-58,"oblique"),(90,-90,"top"),(8,-90,"side"))
        for column, (elevation, azimuth, name) in enumerate(views, 1):
            _draw(
                figure.add_subplot(2,3,column,projection="3d"), points, truth,
                baseline["predictions"], baseline_score, elevation=elevation,
                azimuth=azimuth, title=f"V62 baseline — {name}",
            )
            _draw(
                figure.add_subplot(2,3,column+3,projection="3d"), points, truth,
                gated["predictions"], v63_score, elevation=elevation,
                azimuth=azimuth, title=f"V63 LTCO — {name}",
            )
        figure.suptitle(
            f"{case_id} | purple truth, green TP, red FP | "
            f"V62 {baseline_score['true_positive']}TP/{baseline_score['false_positive']}FP/{baseline_score['false_negative']}FN → "
            f"V63 {v63_score['true_positive']}TP/{v63_score['false_positive']}FP/{v63_score['false_negative']}FN",
            fontsize=15, fontweight="bold",
        )
        figure.tight_layout(rect=(0,0,1,0.96))
        image_path = output_root / "visual-audit" / f"{case_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(image_path, bbox_inches="tight", pad_inches=0.04, facecolor=figure.get_facecolor())
        plt.close(figure)
        visual_rows.append({
            "case_id":case_id,"path":str(image_path.resolve()),"sha256":_sha(image_path),
            "baseline":{key:baseline_score[key] for key in ("true_positive","false_positive","false_negative","exact")},
            "v63":{key:v63_score[key] for key in ("true_positive","false_positive","false_negative","exact")},
        })
        del truth

    baseline_aggregate = _aggregate(baseline_cases)
    v63_aggregate = _aggregate(v63_cases)
    deleted_tp = sum(bool(row["was_true_positive"]) for row in deletion_rows)
    activation = baseline_aggregate["true_positive"] >= 4 and baseline_aggregate["false_positive"] >= 4
    automatic_checks = {
        "activation_tp_at_least_4": baseline_aggregate["true_positive"] >= 4,
        "activation_fp_at_least_4": baseline_aggregate["false_positive"] >= 4,
        "tp_preserved": v63_aggregate["true_positive"] == baseline_aggregate["true_positive"],
        "fp_quartered": v63_aggregate["false_positive"] <= 0.25 * baseline_aggregate["false_positive"],
        "survivor_precision_at_least_0_90": v63_aggregate["precision"] >= 0.90,
        "overall_precision_increased": v63_aggregate["precision"] > baseline_aggregate["precision"],
        "recall_identical": v63_aggregate["recall"] == baseline_aggregate["recall"],
        "no_exact_loss": all(not before["exact"] or after["exact"] for before,after in zip(baseline_cases,v63_cases)),
        "at_least_one_exact_gain": any(not before["exact"] and after["exact"] for before,after in zip(baseline_cases,v63_cases)),
        "all_deleted_predictions_were_false_positive": deleted_tp == 0,
        "all_six_visuals_rendered": len(visual_rows) == 6,
    }
    report = {
        "schema_version":1,
        "kind":"sherman_qc_f1_v63_ltco_v5r3_one_time_score_and_visual_audit",
        "token":TOKEN,
        "evidence_class":"prospective_nonconfirmatory_corpus_conditioned_synthetic",
        "confirmatory_eligible":False,
        "truth_opened_once_after_output_freeze":True,
        "truth_opened_by_inference":False,
        "thresholds":{"angle_deg":5.0,"transverse_offset_mm":4.0,"gold_span_overlap":0.60,"prediction_span_overlap":0.60},
        "baseline":{"aggregate":baseline_aggregate,"cases":baseline_cases},
        "v63":{"aggregate":v63_aggregate,"cases":v63_cases},
        "activation_gate_passed":activation,
        "deletions":{"count":len(deletion_rows),"true_positive_deleted":deleted_tp,"false_positive_deleted":len(deletion_rows)-deleted_tp,"rows":deletion_rows},
        "automatic_retention_checks":automatic_checks,
        "manual_retention_checks_pending":["no_broad_roll_survivor","no_hat_or_channel_suppression","native_visual_agreement"],
        "visuals":visual_rows,
        "no_tuning_rerun_or_replacement":True,
    }
    score_path = output_root / "score.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline":baseline_aggregate,"v63":v63_aggregate,"activation":activation,
        "deleted_tp":deleted_tp,"deleted_fp":len(deletion_rows)-deleted_tp,
        "score_sha256":_sha(score_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
