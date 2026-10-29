#!/usr/bin/env python3
"""Open synthetic truth once and compare frozen V52/V58 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(cases: list[dict]) -> dict:
    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    exact = sum(bool(row["exact"]) for row in cases)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_case_count": exact,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--v58-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("score output already exists")
    private = json.loads((args.corpus_root / "evaluator_only" / "private_manifest.json").read_text(encoding="utf-8"))
    if private.get("token") != TOKEN:
        raise ValueError("private manifest token mismatch")
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    baseline_cases, candidate_cases, safe_cases = [], [], []
    raw_tp = raw_fp = pruned_tp = pruned_fp = 0
    family_rows: dict[str, dict[str, list[dict]]] = {}
    for private_row in private["cases"]:
        case_id = str(private_row["opaque_case_id"])
        family_id = str(private_row["family_id"])
        truth_path = args.corpus_root / "evaluator_only" / case_id / "truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        baseline = json.loads((args.v52_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        candidate = json.loads((args.v58_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        safe = json.loads((args.v59_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        baseline_score = match_lines(truth["manual_bends"], baseline["predictions"], thresholds)
        candidate_score = match_lines(truth["manual_bends"], candidate["predictions"], thresholds)
        safe_score = match_lines(truth["manual_bends"], safe["predictions"], thresholds)
        baseline_score.update({"case_id": case_id, "family_id": family_id})
        candidate_score.update({"case_id": case_id, "family_id": family_id})
        safe_score.update({"case_id": case_id, "family_id": family_id})
        baseline_cases.append(baseline_score)
        candidate_cases.append(candidate_score)
        safe_cases.append(safe_score)
        family_rows.setdefault(family_id, {"v52": [], "v58": [], "v59": []})["v52"].append(baseline_score)
        family_rows[family_id]["v58"].append(candidate_score)
        family_rows[family_id]["v59"].append(safe_score)

        matched_ids = {row["prediction_id"] for row in candidate_score["matches"]}
        raw_ids = {row["bend_id"] for row in candidate["predictions"] if row.get("selection_route") == "calibrated_multiview_turn_consensus_v58"}
        raw_tp += len(raw_ids & matched_ids)
        raw_fp += len(raw_ids - matched_ids)
        pruned_ordinals = {int(row["v52_ordinal"]) for row in candidate.get("pruned_v52") or []}
        baseline_matched = {row["prediction_id"] for row in baseline_score["matches"]}
        for ordinal in pruned_ordinals:
            bend_id = baseline["predictions"][ordinal]["bend_id"]
            if bend_id in baseline_matched:
                pruned_tp += 1
            else:
                pruned_fp += 1

    families = []
    for family_id in sorted(family_rows):
        row = {"family_id": family_id}
        for name in ("v52", "v58", "v59"):
            summary = _aggregate(family_rows[family_id][name])
            row[name] = summary
        families.append(row)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_calibrated_raw_synthetic_v1_score",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "v52": {"aggregate": _aggregate(baseline_cases), "cases": baseline_cases},
        "v58": {"aggregate": _aggregate(candidate_cases), "cases": candidate_cases},
        "v59": {"aggregate": _aggregate(safe_cases), "cases": safe_cases},
        "mechanism": {
            "raw_view_true_positive": raw_tp,
            "raw_view_false_positive": raw_fp,
            "raw_view_precision": _ratio(raw_tp, raw_tp + raw_fp),
            "pruned_v52_true_positive": pruned_tp,
            "pruned_v52_false_positive": pruned_fp,
        },
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v52": report["v52"]["aggregate"], "v58": report["v58"]["aggregate"], "v59": report["v59"]["aggregate"], "mechanism": report["mechanism"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
