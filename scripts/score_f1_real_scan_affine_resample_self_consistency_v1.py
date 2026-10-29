#!/usr/bin/env python3
"""Score transformed-reference self-consistency after the sealed visual audit.

The evaluator records are transformed, previously audited V51 predictions,
not human gold.  This script therefore reports development consistency only
and refuses to emit benchmark/promotion claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


CASE_IDS = tuple(f"R{index:03d}" for index in range(1, 9))
EXECUTION_MANIFEST_SHA256 = "b7045655bc58c5e7315828e739320239771e32d34710a0d9958efd15c791cde6"
PRETRUTH_VISUAL_AUDIT_SHA256 = "8229e2641cc9acb1bcb38e303e4cac7ccc2f963a5b7d1fdb77286065c105d991"
EVALUATOR_MANIFEST_SHA256 = "f6ff50099f5da870fb3a58a4cbe5557ce088f0a89a330f23a2b53267e331ab38"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_matcher_truth(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bend_id": str(row["line_id"]),
            "line_start": row["line_start"],
            "line_end": row["line_end"],
        }
        for row in rows
    ]


def score(
    *,
    execution_root: Path,
    execution_manifest: Path,
    visual_audit: Path,
    evaluator_manifest: Path,
    expected_execution_sha256: str = EXECUTION_MANIFEST_SHA256,
    expected_visual_audit_sha256: str = PRETRUTH_VISUAL_AUDIT_SHA256,
    expected_evaluator_sha256: str = EVALUATOR_MANIFEST_SHA256,
) -> dict[str, Any]:
    if _sha(execution_manifest) != expected_execution_sha256:
        raise ValueError("execution manifest hash mismatch")
    if _sha(visual_audit) != expected_visual_audit_sha256:
        raise ValueError("pre-truth visual audit hash mismatch")
    if _sha(evaluator_manifest) != expected_evaluator_sha256:
        raise ValueError("evaluator manifest hash mismatch")
    execution = json.loads(execution_manifest.read_text(encoding="utf-8"))
    audit = json.loads(visual_audit.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_manifest.read_text(encoding="utf-8"))
    if execution.get("locked_benchmark_eligible") is not False or execution.get("independent_f1_eligible") is not False:
        raise ValueError("execution must remain benchmark-ineligible")
    if execution.get("truth_opened_by_detector") is not False or execution.get("pseudo_truth_opened_before_visual_audit") is not False:
        raise ValueError("truth isolation contract violated")
    if audit.get("evaluator_pseudo_truth_opened") is not False:
        raise ValueError("visual audit was not completed before pseudo-truth opening")
    if list(audit.get("visually_inspected_case_ids") or []) != list(CASE_IDS):
        raise ValueError("visual audit case set mismatch")
    if evaluator.get("pseudo_truth_kind") != "development_pseudo_truth_from_visually_audited_v51_outputs":
        raise ValueError("unexpected reference type")
    rows = list(evaluator.get("cases") or [])
    if [str(row.get("opaque_case_id")) for row in rows] != list(CASE_IDS):
        raise ValueError("evaluator case ordering mismatch")

    case_results: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    for row in rows:
        case_id = str(row["opaque_case_id"])
        prediction_path = execution_root / "v52" / "primary" / case_id / "semantic_result.json"
        prediction_hashes[case_id] = _sha(prediction_path)
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        if prediction.get("source_only") is not True or prediction.get("research_only") is not True:
            raise ValueError(f"{case_id}: prediction is not source-only research output")
        predictions = list(prediction.get("predictions") or [])
        result = match_lines(_as_matcher_truth(list(row.get("pseudo_truth_lines") or [])), predictions, THRESHOLDS)
        by_id = {str(item.get("bend_id")): item for item in predictions}
        unmatched_route_counts: dict[str, int] = {}
        for prediction_id in result["unmatched_prediction_ids"]:
            prediction_row = by_id[prediction_id]
            route = str(prediction_row.get("selection_route") or prediction_row.get("source_kind") or "unknown")
            unmatched_route_counts[route] = unmatched_route_counts.get(route, 0) + 1
        result.update({
            "case_id": case_id,
            "parent_case_id": str(row["parent_case_id"]),
            "variant": str(row["variant"]),
            "reference_kind": str(row["pseudo_truth_kind"]),
            "unmatched_prediction_route_counts": dict(sorted(unmatched_route_counts.items())),
        })
        case_results.append(result)

    tp = sum(int(row["true_positive"]) for row in case_results)
    fp = sum(int(row["false_positive"]) for row in case_results)
    fn = sum(int(row["false_negative"]) for row in case_results)
    exact = sum(bool(row["exact"]) for row in case_results)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    pairs = []
    for parent_id in sorted({str(row["parent_case_id"]) for row in case_results}):
        pair = sorted((row for row in case_results if row["parent_case_id"] == parent_id), key=lambda row: row["variant"])
        pairs.append({
            "parent_case_id": parent_id,
            "case_ids": [row["case_id"] for row in pair],
            "prediction_counts": [int(row["prediction_count"]) for row in pair],
            "reference_counts": [int(row["gold_count"]) for row in pair],
            "true_positives": [int(row["true_positive"]) for row in pair],
        })
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_real_scan_affine_resample_self_consistency_v1",
        "evidence_class": "development_pseudo_truth_consistency_only",
        "locked_benchmark_eligible": False,
        "independent_f1_eligible": False,
        "human_gold": False,
        "promotion_eligible": False,
        "reference_kind": "development_pseudo_truth_from_visually_audited_v51_outputs",
        "thresholds": {
            "axis_angle_deg": THRESHOLDS.angle_deg,
            "transverse_offset_mm": THRESHOLDS.transverse_offset_mm,
            "gold_span_overlap": THRESHOLDS.gold_span_overlap,
            "prediction_span_overlap": THRESHOLDS.prediction_span_overlap,
            "matching": "one_to_one_minimum_cost",
        },
        "input_sha256": {
            "execution_manifest": expected_execution_sha256,
            "pretruth_visual_audit": expected_visual_audit_sha256,
            "evaluator_manifest": expected_evaluator_sha256,
            "predictions": prediction_hashes,
        },
        "aggregate_integers": {
            "case_count": len(case_results),
            "exact_case_count": exact,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
        "aggregate_ratios": {
            "pseudo_truth_precision": precision,
            "pseudo_truth_recall": recall,
            "pseudo_truth_f1": f1,
            "exact_count_accuracy": _ratio(exact, len(case_results)),
        },
        "pairs": pairs,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("self-consistency output is immutable")
    result = score(
        execution_root=args.execution_root.resolve(),
        execution_manifest=args.execution_manifest.resolve(),
        visual_audit=args.visual_audit.resolve(),
        evaluator_manifest=args.evaluator_manifest.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({**result["aggregate_integers"], **result["aggregate_ratios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
