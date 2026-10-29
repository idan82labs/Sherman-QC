#!/usr/bin/env python3
"""Open synthetic truth once and apply the frozen V55 retention gate once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_mesh_section_turn_locus_reselection_v55 import (
    TERMINAL_FAILURE_TOKEN,
    canonical_bytes,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
KIND = "sherman_qc_f1_v55_mesh_section_turn_locus_reselection_result"
EXACT_V52_CASES = frozenset(("S005", "S006", "S007", "S013", "S026", "S027"))


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload, newline=True))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"single score already exists: {args.output}")
    visual = json.loads(args.visual_audit.read_text(encoding="utf-8"))
    if (
        visual.get("kind") != "sherman_qc_f1_v55_blind_native_visual_audit"
        or visual.get("passed") is not True
        or visual.get("truth_opened") is not False
        or visual.get("signed") is not True
    ):
        raise ValueError("blind native-resolution visual audit is not sealed and passing")
    execution = json.loads((args.execution_root / "execution_manifest.json").read_text(encoding="utf-8"))
    if execution.get("status") != "SEALED_PRIMARY_REPLAY" or execution.get("truth_opened") is not False:
        raise ValueError("V55 execution is not sealed for its single score")

    # Truth is deliberately read only after the sealed source-only visual gate.
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_score.read_text(encoding="utf-8"))
    if any(payload.get("token") != TOKEN for payload in (truth, family_map, baseline)):
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
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    cases = []
    correct_additions = false_additions = correct_prunes = incorrect_prunes = 0
    changed_families: set[str] = set()
    exact_case_violations: list[str] = []
    family_deltas: dict[str, dict[str, int]] = {}

    for case_id in sorted(truth_by_case):
        primary_path = args.execution_root / "primary" / case_id / "semantic_result.json"
        replay_path = args.execution_root / "replay" / case_id / "semantic_result.json"
        if primary_path.read_bytes() != replay_path.read_bytes():
            raise ValueError(f"{case_id}: primary/replay bytes differ")
        prediction = json.loads(primary_path.read_text(encoding="utf-8"))
        if prediction.get("kind") != KIND or prediction.get("source_only") is not True:
            raise ValueError(f"{case_id}: invalid V55 prediction")
        v52 = json.loads((args.v52_root / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        baseline_result = baseline_by_case[case_id]
        result = match_lines(truth_by_case[case_id]["manual_bends"], prediction["predictions"], thresholds)
        new_ids = {
            str(row.get("bend_id"))
            for row in prediction["predictions"]
            if str(row.get("bend_id") or "").startswith("TRV55-")
        }
        final_ids = {str(row.get("bend_id")) for row in prediction["predictions"]}
        v52_ids = {str(row.get("bend_id")) for row in v52["predictions"]}
        matched_new = {str(row["prediction_id"]) for row in result["matches"]} & new_ids
        baseline_matched = {str(row["prediction_id"]) for row in baseline_result["matches"]}
        baseline_false = v52_ids - baseline_matched
        pruned_ids = v52_ids - final_ids
        case_correct_additions = len(matched_new)
        case_false_additions = len(new_ids - matched_new)
        case_correct_prunes = len(pruned_ids & baseline_false)
        case_incorrect_prunes = len(pruned_ids & baseline_matched)
        correct_additions += case_correct_additions
        false_additions += case_false_additions
        correct_prunes += case_correct_prunes
        incorrect_prunes += case_incorrect_prunes
        changed = bool(new_ids or pruned_ids)
        family_id = family_for_case[case_id]
        if case_correct_additions or case_correct_prunes:
            changed_families.add(family_id)
        if case_id in EXACT_V52_CASES and prediction["predictions"] != v52["predictions"]:
            exact_case_violations.append(case_id)
        result.update(
            {
                "case_id": case_id,
                "family_id": family_id,
                "changed": changed,
                "correct_additions": case_correct_additions,
                "false_additions": case_false_additions,
                "correct_prunes": case_correct_prunes,
                "incorrect_prunes": case_incorrect_prunes,
                "new_prediction_ids": sorted(new_ids),
                "pruned_prediction_ids": sorted(pruned_ids),
            }
        )
        cases.append(result)

    tp = sum(row["true_positive"] for row in cases)
    fp = sum(row["false_positive"] for row in cases)
    fn = sum(row["false_negative"] for row in cases)
    exact = sum(row["exact"] for row in cases)
    families = []
    family_regressions: list[str] = []
    for family_id in sorted(family_map["families"]):
        rows = [row for row in cases if row["family_id"] == family_id]
        baseline_rows = [baseline_by_case[row["case_id"]] for row in rows]
        family_tp = sum(row["true_positive"] for row in rows)
        family_fp = sum(row["false_positive"] for row in rows)
        family_fn = sum(row["false_negative"] for row in rows)
        baseline_tp = sum(row["true_positive"] for row in baseline_rows)
        baseline_fp = sum(row["false_positive"] for row in baseline_rows)
        if family_tp < baseline_tp or family_fp > baseline_fp:
            family_regressions.append(family_id)
        delta = {"true_positive": family_tp - baseline_tp, "false_positive": family_fp - baseline_fp}
        family_deltas[family_id] = delta
        families.append(
            {
                "family_id": family_id,
                "case_ids": [row["case_id"] for row in rows],
                "true_positive": family_tp,
                "false_positive": family_fp,
                "false_negative": family_fn,
                "recall": _ratio(family_tp, family_tp + family_fn),
                "baseline_delta": delta,
            }
        )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    replay_identity = execution.get("semantic_replay_identity") or {}
    gates = {
        "correct_additions_at_least_6": correct_additions >= 6,
        "false_additions_exactly_0": false_additions == 0,
        "correct_prunes_at_least_6": correct_prunes >= 6,
        "incorrect_prunes_exactly_0": incorrect_prunes == 0,
        "final_tp_at_least_58": tp >= 58,
        "final_fp_at_most_21": fp <= 21,
        "final_fn_at_most_42": fn <= 42,
        "precision_at_least_0_73": precision >= 0.73,
        "recall_at_least_0_58": recall >= 0.58,
        "f1_at_least_0_645": f1 >= 0.645,
        "exact_count_at_least_12": exact >= 12,
        "correct_change_in_at_least_5_families": len(changed_families) >= 5,
        "replay_30_of_30": len(replay_identity) == 30 and all(replay_identity.values()),
        "visual_gate_passed": visual.get("passed") is True,
        "six_exact_v52_cases_unchanged": not exact_case_violations,
        "no_family_lost_tp_or_gained_fp": not family_regressions,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_synthetic_stress_score",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "single_score_consumed": True,
        "truth_opened": True,
        "status": "RETAINED_RESEARCH_BASELINE" if passed else "TERMINAL_RETIRED_RETENTION_GATE",
        "terminal_failure_token": None if passed else TERMINAL_FAILURE_TOKEN,
        "aggregate_integers": {
            "case_count": 30,
            "family_count": 15,
            "exact_case_count": exact,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "correct_additions": correct_additions,
            "false_additions": false_additions,
            "correct_prunes": correct_prunes,
            "incorrect_prunes": incorrect_prunes,
            "correctly_changed_family_count": len(changed_families),
        },
        "aggregate_ratios": {
            "exact_count_accuracy": _ratio(exact, 30),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "minimum_family_recall": min(row["recall"] for row in families),
        },
        "retention_gates": gates,
        "exact_v52_case_ids": sorted(EXACT_V52_CASES),
        "exact_case_violations": exact_case_violations,
        "correctly_changed_families": sorted(changed_families),
        "family_regressions": family_regressions,
        "family_deltas": family_deltas,
        "families": families,
        "cases": cases,
    }
    _write_new(args.output, report)
    print(json.dumps({**report["aggregate_integers"], **report["aggregate_ratios"], "status": report["status"]}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
