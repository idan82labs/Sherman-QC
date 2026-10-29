#!/usr/bin/env python3
"""Score one sealed 30-scan/15-family V51 corpus in one transaction.

This reader is corpus-generic and deliberately contains no case plumbing.  It
reuses the frozen one-to-one matcher and accepts only opaque case/family tokens.
The detector process must never invoke this module because it opens truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
CASE_COUNT = 30
FAMILY_COUNT = 15
SCHEMA_VERSION = 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = _sha(path)
    if observed != expected_sha256:
        raise ValueError(f"sealed input hash mismatch for {path}: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema for {path}")
    if payload.get("token") != TOKEN:
        raise ValueError(f"wrong execution token for {path}")
    return payload


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prediction_path(root: Path, case_id: str) -> Path:
    """Accept the frozen runner tree, retaining flat fixtures for unit tests."""
    nested = root / case_id / "semantic_result.json"
    flat = root / f"{case_id}.json"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"missing sealed prediction for {case_id} under {root}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def score(
    *,
    truth_path: Path,
    truth_sha256: str,
    family_path: Path,
    family_sha256: str,
    prediction_root: Path,
) -> dict[str, Any]:
    truth = _load(truth_path, truth_sha256)
    family_map = _load(family_path, family_sha256)
    truth_rows = list(truth.get("cases") or [])
    families = dict(family_map.get("families") or {})
    if len(truth_rows) != CASE_COUNT:
        raise ValueError(f"truth must contain exactly {CASE_COUNT} cases")
    if len(families) != FAMILY_COUNT:
        raise ValueError(f"family map must contain exactly {FAMILY_COUNT} families")
    truth_by_case = {str(row["case_id"]): row for row in truth_rows}
    if len(truth_by_case) != CASE_COUNT:
        raise ValueError("truth contains duplicate opaque case IDs")
    family_for_case: dict[str, str] = {}
    for family_id, case_ids in sorted(families.items()):
        if len(case_ids) != 2:
            raise ValueError(f"{family_id}: expected exactly two cases")
        for case_id in case_ids:
            case_id = str(case_id)
            if case_id in family_for_case:
                raise ValueError(f"{case_id}: assigned to multiple families")
            family_for_case[case_id] = str(family_id)
    if set(family_for_case) != set(truth_by_case):
        raise ValueError("truth and family-map case sets differ")

    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    case_results: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    for case_id in sorted(truth_by_case):
        prediction_path = _prediction_path(prediction_root, case_id)
        prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        if prediction_payload.get("kind") != "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result":
            raise ValueError(f"{case_id}: wrong prediction kind")
        if prediction_payload.get("source_only") is not True:
            raise ValueError(f"{case_id}: prediction is not source-only")
        result = match_lines(
            list(truth_by_case[case_id].get("manual_bends") or []),
            list(prediction_payload.get("predictions") or []),
            thresholds,
        )
        result.update({"case_id": case_id, "family_id": family_for_case[case_id]})
        case_results.append(result)
        prediction_hashes[case_id] = _sha(prediction_path)

    tp = sum(int(row["true_positive"]) for row in case_results)
    fp = sum(int(row["false_positive"]) for row in case_results)
    fn = sum(int(row["false_negative"]) for row in case_results)
    exact = sum(bool(row["exact"]) for row in case_results)
    family_rows: list[dict[str, Any]] = []
    for family_id in sorted(families):
        rows = [row for row in case_results if row["family_id"] == family_id]
        family_tp = sum(int(row["true_positive"]) for row in rows)
        family_fn = sum(int(row["false_negative"]) for row in rows)
        family_rows.append(
            {
                "family_id": family_id,
                "case_ids": sorted(str(row["case_id"]) for row in rows),
                "true_positive": family_tp,
                "false_negative": family_fn,
                "recall": _ratio(family_tp, family_tp + family_fn),
            }
        )
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_locked_score",
        "token": TOKEN,
        "input_sha256": {
            "truth": truth_sha256,
            "family_map": family_sha256,
            "predictions": prediction_hashes,
        },
        "thresholds": {
            "axis_angle_deg": 5.0,
            "transverse_offset_mm": 4.0,
            "gold_span_overlap": 0.60,
            "prediction_span_overlap": 0.60,
            "matching": "one_to_one_minimum_cost",
        },
        "aggregate_integers": {
            "case_count": CASE_COUNT,
            "family_count": FAMILY_COUNT,
            "exact_case_count": exact,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
        "aggregate_ratios": {
            "exact_count_accuracy": _ratio(exact, CASE_COUNT),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "minimum_family_recall": min(row["recall"] for row in family_rows),
        },
        "semantic_gate": {
            "exact_count_pass": exact >= 27,
            "precision_pass": precision >= 0.90,
            "recall_pass": recall >= 0.90,
            "minimum_family_recall_pass": all(row["recall"] >= 0.80 for row in family_rows),
        },
        "families": family_rows,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--family-map-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"score is immutable and already exists: {output}")
    report = score(
        truth_path=args.truth.resolve(),
        truth_sha256=args.truth_sha256,
        family_path=args.family_map.resolve(),
        family_sha256=args.family_map_sha256,
        prediction_root=args.prediction_root.resolve(),
    )
    _atomic_write_json(output, report)
    print(json.dumps({"output": str(output), **report["aggregate_integers"], **report["aggregate_ratios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
