#!/usr/bin/env python3
"""Single sealed score of frozen synthetic V54 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.score_f1_v51_new_untouched_corpus import _atomic_write_json


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
KIND = "sherman_qc_f1_v54_exclusive_normal_turn_coverage_result"
TERMINAL_FAILURE_TOKEN = "V54_ENTC_TERMINAL_RETIRE"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"score already exists: {args.output}")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_score.read_text(encoding="utf-8"))
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
    baseline_by_case = {str(row["case_id"]): row for row in baseline["cases"]}
    baseline_exact = {case_id for case_id, row in baseline_by_case.items() if row["exact"]}
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    cases = []
    changed_scans: list[str] = []
    changed_families: set[str] = set()
    new_true_positive = 0
    new_false_positive = 0
    retained_old_true_positive = 0
    exact_case_addition_violations: list[str] = []
    for case_id in sorted(truth_by_case):
        primary_path = args.prediction_root / "primary" / case_id / "semantic_result.json"
        replay_path = args.prediction_root / "replay" / case_id / "semantic_result.json"
        if primary_path.read_bytes() != replay_path.read_bytes():
            raise ValueError(f"{case_id}: primary/replay semantic bytes differ before score")
        prediction = json.loads(primary_path.read_text(encoding="utf-8"))
        if prediction.get("kind") != KIND or prediction.get("source_only") is not True:
            raise ValueError(f"{case_id}: invalid V54 prediction")
        audit = prediction["exclusive_normal_turn_coverage_audit"]
        additions = list(audit.get("recovered_candidates") or ())
        addition_ids = {"TRV54-" + str(row["candidate_sha256"])[:20] for row in additions}
        if additions:
            changed_scans.append(case_id)
            changed_families.add(family_for_case[case_id])
            if case_id in baseline_exact:
                exact_case_addition_violations.append(case_id)
        result = match_lines(
            truth_by_case[case_id]["manual_bends"],
            prediction["predictions"],
            thresholds,
        )
        matched_ids = {str(row["prediction_id"]) for row in result["matches"]}
        old_matched_ids = {
            str(row["prediction_id"]) for row in baseline_by_case[case_id]["matches"]
        }
        retained_old_true_positive += len(old_matched_ids & matched_ids)
        new_true_positive += len(addition_ids & matched_ids)
        new_false_positive += len(addition_ids - matched_ids)
        result.update(
            {
                "case_id": case_id,
                "family_id": family_for_case[case_id],
                "v54_addition_count": len(additions),
                "v54_matched_addition_count": len(addition_ids & matched_ids),
                "v54_unmatched_addition_count": len(addition_ids - matched_ids),
                "retained_v52_matched_prediction_count": len(old_matched_ids & matched_ids),
                "v52_matched_prediction_count": len(old_matched_ids),
            }
        )
        cases.append(result)
    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    exact = sum(row["exact"] for row in cases)
    families = []
    for family_id in sorted(family_map["families"]):
        rows = [row for row in cases if row["family_id"] == family_id]
        family_tp = sum(row["true_positive"] for row in rows)
        family_fn = sum(row["false_negative"] for row in rows)
        families.append(
            {
                "family_id": family_id,
                "case_ids": [row["case_id"] for row in rows],
                "true_positive": family_tp,
                "false_negative": family_fn,
                "recall": _ratio(family_tp, family_tp + family_fn),
            }
        )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gates = {
        "all_52_v52_true_positives_retained": retained_old_true_positive == 52,
        "at_least_8_new_true_positives": new_true_positive >= 8,
        "zero_new_false_positives": new_false_positive == 0,
        "additions_in_at_least_6_scans": len(changed_scans) >= 6,
        "additions_in_at_least_4_families": len(changed_families) >= 4,
        "true_positive_at_least_60": tp >= 60,
        "false_positive_exactly_27": fp == 27,
        "false_negative_at_most_40": fn <= 40,
        "precision_at_least_0_689655": precision >= 0.689655,
        "recall_at_least_0_600000": recall >= 0.600000,
        "f1_at_least_0_641711": f1 >= 0.641711,
        "exact_count_at_least_9": exact >= 9,
        "zero_additions_on_v52_exact_scans": not exact_case_addition_violations,
    }
    score_gate_passed = all(gates.values())
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v54_synthetic_stress_score",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "single_score_consumed": True,
        "status": "SCORE_GATE_PASS_PENDING_VISUAL" if score_gate_passed else "TERMINAL_RETIRED_SCORE_GATE",
        "terminal_failure_token": None if score_gate_passed else TERMINAL_FAILURE_TOKEN,
        "aggregate_integers": {
            "case_count": 30,
            "family_count": 15,
            "exact_case_count": exact,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "new_true_positive": new_true_positive,
            "new_false_positive": new_false_positive,
            "retained_v52_true_positive": retained_old_true_positive,
            "changed_scan_count": len(changed_scans),
            "changed_family_count": len(changed_families),
        },
        "aggregate_ratios": {
            "exact_count_accuracy": _ratio(exact, 30),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "minimum_family_recall": min(row["recall"] for row in families),
        },
        "changed_scans": changed_scans,
        "changed_families": sorted(changed_families),
        "v52_exact_case_ids": sorted(baseline_exact),
        "exact_case_addition_violations": exact_case_addition_violations,
        "retention_gates": gates,
        "families": families,
        "cases": cases,
    }
    _atomic_write_json(args.output, report)
    print(json.dumps({**report["aggregate_integers"], **report["aggregate_ratios"], "status": report["status"]}, sort_keys=True))
    return 0 if score_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

