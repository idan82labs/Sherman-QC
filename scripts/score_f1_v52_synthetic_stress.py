#!/usr/bin/env python3
"""Score source-only synthetic V52 predictions as development evidence only."""

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
KIND = "sherman_qc_f1_v52_cylindrical_consensus_recovery_result"


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"score already exists: {args.output}")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    if truth.get("token") != TOKEN or family_map.get("token") != TOKEN:
        raise ValueError("synthetic token mismatch")
    truth_by_case = {str(row["case_id"]): row for row in truth["cases"]}
    family_for_case = {
        str(case_id): str(family_id)
        for family_id, case_ids in family_map["families"].items()
        for case_id in case_ids
    }
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    cases = []
    for case_id in sorted(truth_by_case):
        prediction = json.loads((args.prediction_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        if prediction.get("kind") != KIND or prediction.get("source_only") is not True:
            raise ValueError(f"{case_id}: invalid V52 prediction")
        result = match_lines(truth_by_case[case_id]["manual_bends"], prediction["predictions"], thresholds)
        result.update({"case_id": case_id, "family_id": family_for_case[case_id]})
        cases.append(result)
    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    exact = sum(row["exact"] for row in cases)
    families = []
    for family_id in sorted(family_map["families"]):
        rows = [row for row in cases if row["family_id"] == family_id]
        ft, ff = sum(row["true_positive"] for row in rows), sum(row["false_negative"] for row in rows)
        families.append({"family_id": family_id, "case_ids": [row["case_id"] for row in rows], "true_positive": ft, "false_negative": ff, "recall": _ratio(ft, ft + ff)})
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v52_synthetic_stress_score",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "aggregate_integers": {"case_count": 30, "family_count": 15, "exact_case_count": exact, "true_positive": tp, "false_positive": fp, "false_negative": fn},
        "aggregate_ratios": {"exact_count_accuracy": _ratio(exact, 30), "precision": precision, "recall": recall, "f1": f1, "minimum_family_recall": min(row["recall"] for row in families)},
        "families": families,
        "cases": cases,
    }
    _atomic_write_json(args.output, report)
    print(json.dumps({**report["aggregate_integers"], **report["aggregate_ratios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
