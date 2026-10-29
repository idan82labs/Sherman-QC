#!/usr/bin/env python3
"""Open U01-U04 development pseudo-truth once after the V66 visual seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    V51_ROOT,
    atomic_json,
    opaque_source_id,
    sha256_file,
    source_paths,
)


PROFILE = "V66_TFTOC_U01_U04_PSEUDO_TRUTH_SCORE_ONCE_V1"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    true_positive = sum(int(row["true_positive"]) for row in rows)
    false_positive = sum(int(row["false_positive"]) for row in rows)
    false_negative = sum(int(row["false_negative"]) for row in rows)
    exact = sum(bool(row["exact"]) for row in rows)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact_case_count": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _verify_lock(execution_root: Path, adjudication_path: Path, lock: Mapping[str, Any]) -> None:
    if lock.get("pseudo_truth_open_authorized") is not True:
        raise RuntimeError("scoring lock does not authorize the one-time opening")
    if sha256_file(execution_root / "summary.json") != lock["execution_summary_sha256"]:
        raise RuntimeError("execution summary hash mismatch")
    if sha256_file(adjudication_path) != lock["visual_adjudication_sha256"]:
        raise RuntimeError("visual adjudication hash mismatch")
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if adjudication.get("pseudo_truth_opened") is not False or adjudication.get("complete") is not True:
        raise RuntimeError("visual adjudication is not a sealed complete pretruth artifact")


def score(execution_root: Path, adjudication_path: Path, scoring_lock_path: Path) -> Mapping[str, Any]:
    lock = json.loads(scoring_lock_path.read_text(encoding="utf-8"))
    _verify_lock(execution_root, adjudication_path, lock)
    execution = json.loads((execution_root / "summary.json").read_text(encoding="utf-8"))
    if execution.get("pseudo_truth_opened") is not False or execution.get("execution_passed") is not True:
        raise RuntimeError("execution is not an eligible sealed result")
    case_paths = {row["opaque_source_id"]: execution_root / row["result"] for row in execution["cases"]}
    references = {}
    reference_hashes = {}
    for case_id, pool_name in CASE_BINDINGS:
        source_path, _pool, _v51 = source_paths(case_id, pool_name)
        opaque_id = opaque_source_id(source_path)
        reference_path = V51_ROOT / f"{case_id}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        references[opaque_id] = list(reference.get("predictions") or ())
        reference_hashes[opaque_id] = sha256_file(reference_path)

    baseline_rows = []
    candidate_rows = []
    cases = []
    for opaque_id in sorted(case_paths):
        payload = json.loads(case_paths[opaque_id].read_text(encoding="utf-8"))
        truth = references[opaque_id]
        baseline = match_lines(truth, list(payload["identity"]["v52_final_predictions"]), THRESHOLDS)
        candidate = match_lines(truth, list(payload["identity"]["v66_final_predictions"]), THRESHOLDS)
        baseline_rows.append(baseline); candidate_rows.append(candidate)
        cases.append({
            "opaque_source_id": opaque_id,
            "v52": baseline,
            "v66": candidate,
            "tp_nonregression": int(candidate["true_positive"]) >= int(baseline["true_positive"]),
            "fn_nonregression": int(candidate["false_negative"]) <= int(baseline["false_negative"]),
            "previously_exact_preserved": not bool(baseline["exact"]) or bool(candidate["exact"]),
        })
    baseline_aggregate = _aggregate(baseline_rows)
    candidate_aggregate = _aggregate(candidate_rows)
    required_fp_reduction = max(2, math.ceil(0.20 * int(baseline_aggregate["false_positive"])))
    gates = {
        "every_case_tp_nonregression": all(row["tp_nonregression"] for row in cases),
        "every_case_fn_nonregression": all(row["fn_nonregression"] for row in cases),
        "fp_reduction": int(candidate_aggregate["false_positive"]) <= int(baseline_aggregate["false_positive"]) - required_fp_reduction,
        "f1_gain_at_least_0_03": float(candidate_aggregate["f1"]) >= float(baseline_aggregate["f1"]) + 0.03,
        "recall_nonregression": float(candidate_aggregate["recall"]) >= float(baseline_aggregate["recall"]),
        "exact_case_gain_at_least_one": int(candidate_aggregate["exact_case_count"]) >= int(baseline_aggregate["exact_case_count"]) + 1,
        "previously_exact_preserved": all(row["previously_exact_preserved"] for row in cases),
    }
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "evidence_class": "development_pseudo_truth_consistency_only",
        "locked_benchmark_eligible": False,
        "physical_generalization_claim_authorized": False,
        "human_gold": False,
        "pseudo_truth_opened_once": True,
        "scoring_lock_sha256": sha256_file(scoring_lock_path),
        "visual_adjudication_sha256": sha256_file(adjudication_path),
        "reference_sha256_by_opaque_source": reference_hashes,
        "thresholds": {
            "axis_angle_deg": THRESHOLDS.angle_deg,
            "transverse_offset_mm": THRESHOLDS.transverse_offset_mm,
            "gold_span_overlap": THRESHOLDS.gold_span_overlap,
            "prediction_span_overlap": THRESHOLDS.prediction_span_overlap,
        },
        "v52": baseline_aggregate,
        "v66": candidate_aggregate,
        "required_fp_reduction": required_fp_reduction,
        "gates": gates,
        "all_pseudo_truth_gates_passed": all(gates.values()),
        "cases": cases,
        "decision": "RETAIN_RESEARCH_ONLY" if all(gates.values()) else "TERMINAL_RETIRE_V66",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--visual-adjudication", type=Path, required=True)
    parser.add_argument("--scoring-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V66 pseudo-truth score is immutable")
    result = score(args.execution_root.resolve(), args.visual_adjudication.resolve(), args.scoring_lock.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"v52": result["v52"], "v66": result["v66"], "decision": result["decision"]}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
