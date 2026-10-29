#!/usr/bin/env python3
"""Consume the single post-freeze V56 engineering score and retention gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_canonical_turn_median_line_reconstruction_v56 import PROFILE, RESULT_KIND
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
EXACT_V52_CASES = ("S005", "S006", "S007", "S013", "S026", "S027")


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--v52-score", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"single V56 score already exists: {args.output}")
    manifest_path = args.execution_root / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V56 cannot be scored without a complete execution manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY_FOR_SINGLE_SCORE" or manifest.get("completed_execution_count") != 60:
        raise ValueError("V56 execution is not score-ready")
    if not all((manifest.get("semantic_replay_identity") or {}).values()):
        raise ValueError("V56 replay identity did not pass")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    baseline = json.loads(args.v52_score.read_text(encoding="utf-8"))
    if truth.get("token") != TOKEN or family_map.get("token") != TOKEN or baseline.get("token") != TOKEN:
        raise ValueError("synthetic token mismatch")
    expected_baseline = {
        "case_count": 30,
        "family_count": 15,
        "exact_case_count": 6,
        "true_positive": 52,
        "false_positive": 27,
        "false_negative": 48,
    }
    if baseline.get("aggregate_integers") != expected_baseline:
        raise ValueError("frozen V52 baseline score changed")
    truth_by_case = {str(row["case_id"]): row for row in truth["cases"]}
    family_for_case = {
        str(case_id): str(family_id)
        for family_id, case_ids in family_map["families"].items()
        for case_id in case_ids
    }
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    cases: list[dict[str, Any]] = []
    predictions_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id in sorted(truth_by_case):
        path = args.execution_root / "primary" / case_id / "semantic_result.json"
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if prediction.get("kind") != RESULT_KIND or prediction.get("profile") != PROFILE or prediction.get("source_only") is not True:
            raise ValueError(f"{case_id}: invalid V56 prediction")
        predictions_by_case[case_id] = prediction["predictions"]
        result = match_lines(truth_by_case[case_id]["manual_bends"], prediction["predictions"], thresholds)
        result.update({"case_id": case_id, "family_id": family_for_case[case_id]})
        cases.append(result)
    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    exact = sum(bool(row["exact"]) for row in cases)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    families: list[dict[str, Any]] = []
    baseline_family = {row["family_id"]: row for row in baseline["families"]}
    family_gain_count = 0
    no_family_regression = True
    for family_id in sorted(family_map["families"]):
        rows = [row for row in cases if row["family_id"] == family_id]
        family_tp = sum(row["true_positive"] for row in rows)
        family_fn = sum(row["false_negative"] for row in rows)
        family_recall = _ratio(family_tp, family_tp + family_fn)
        baseline_tp = int(baseline_family[family_id]["true_positive"])
        baseline_recall = float(baseline_family[family_id]["recall"])
        if family_tp > baseline_tp:
            family_gain_count += 1
        if family_recall < baseline_recall:
            no_family_regression = False
        families.append(
            {
                "family_id": family_id,
                "case_ids": [row["case_id"] for row in rows],
                "true_positive": family_tp,
                "false_negative": family_fn,
                "recall": family_recall,
                "v52_true_positive": baseline_tp,
                "v52_recall": baseline_recall,
                "net_true_positive_gain": family_tp - baseline_tp,
            }
        )

    baseline_matches = {
        (row["case_id"], str(match["gold_id"]))
        for row in baseline["cases"]
        for match in row["matches"]
    }
    current_matches = {
        (row["case_id"], str(match["gold_id"]))
        for row in cases
        for match in row["matches"]
    }
    retained_v52_truth = len(baseline_matches & current_matches)
    recovered_new_truth = len(current_matches - baseline_matches)
    lost_v52_truth = sorted([list(item) for item in baseline_matches - current_matches])
    exact_case_identity: dict[str, bool] = {}
    for case_id in EXACT_V52_CASES:
        v52 = json.loads((args.v52_root / "primary" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        exact_case_identity[case_id] = _canonical(v52["predictions"]) == _canonical(predictions_by_case[case_id])

    baseline_aggregate = baseline["aggregate_integers"]
    aggregate_gate = {
        "tp_at_least_62": tp >= 62,
        "fp_at_most_20": fp <= 20,
        "fn_at_most_38": fn <= 38,
        "precision_at_least_0p75": precision >= 0.75,
        "recall_at_least_0p62": recall >= 0.62,
        "f1_at_least_0p6810055866": f1 >= 0.6810055866,
        "exact_at_least_12": exact >= 12,
        "all_52_v52_truths_retained": retained_v52_truth == len(baseline_matches) == 52,
        "at_least_10_new_truths": recovered_new_truth >= 10,
        "at_least_7_net_false_positives_removed": int(baseline_aggregate["false_positive"]) - fp >= 7,
    }
    breadth_gate = {
        "tp_gain_in_at_least_5_families": family_gain_count >= 5,
        "no_family_recall_regression": no_family_regression,
        "minimum_family_recall_positive": min(row["recall"] for row in families) > 0.0,
    }
    exact_gate = {
        "all_exact_v52_arrays_byte_identical": all(exact_case_identity.values()),
        "case_identity": exact_case_identity,
    }
    operational_gate = {
        "primary_replay_success_30_of_30": manifest["completed_execution_count"] == 60,
        "byte_identical_replay_30_of_30": all(manifest["semantic_replay_identity"].values()),
        "p95_wall_at_most_300_seconds": float(manifest["p95_case_wall_seconds"]) <= 300.0,
        "peak_rss_below_16_gib": int(manifest["peak_process_rss_bytes"]) < 16 * 1024**3,
    }
    numerical_retention = all(aggregate_gate.values()) and all(breadth_gate.values()) and bool(exact_gate["all_exact_v52_arrays_byte_identical"]) and all(operational_gate.values())
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_synthetic_stress_score",
        "token": TOKEN,
        "profile": PROFILE,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened_after_output_freeze": True,
        "aggregate_integers": {
            "case_count": 30,
            "family_count": 15,
            "exact_case_count": exact,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "retained_v52_truth_instance_count": retained_v52_truth,
            "newly_recovered_truth_instance_count": recovered_new_truth,
            "net_v52_false_positive_removed": int(baseline_aggregate["false_positive"]) - fp,
            "family_true_positive_gain_count": family_gain_count,
        },
        "aggregate_ratios": {
            "exact_count_accuracy": _ratio(exact, 30),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "minimum_family_recall": min(row["recall"] for row in families),
        },
        "lost_v52_truth_instances": lost_v52_truth,
        "retention_gates": {
            "aggregate": aggregate_gate,
            "breadth": breadth_gate,
            "exact_case_zero_regression": exact_gate,
            "operational": operational_gate,
            "numerical_and_operational_pass": numerical_retention,
            "visual_gate_pending": True,
        },
        "families": families,
        "cases": cases,
    }
    _write(args.output, report)
    print(json.dumps({**report["aggregate_integers"], **report["aggregate_ratios"], "numerical_retention": numerical_retention}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
