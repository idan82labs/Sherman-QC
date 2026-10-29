#!/usr/bin/env python3
"""Nested-LOFO additive selection constrained by explicit parent-pair ownership."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_f1_physical15_ownership_metadata_v1 import ownership_keys
from scripts.evaluate_f1_against_manual_gold import match_lines
from scripts.localize_cad_bends_on_scan_v1 import _atomic
from scripts.run_f1_full_pool_relational_lofo_v1 import (
    MATCH_THRESHOLDS,
    _aggregate,
    _as_prediction,
    _baseline,
    _candidate_keys,
    _duplicate,
    _fit,
    _load_cases,
    _probabilities,
)


THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01)


def _ownership_rows(pool: dict[str, Any]) -> list[dict[str, Any]]:
    candidates_by_id = {
        str(row.get("candidate_id")): row
        for row in pool["candidates"]
        if row.get("candidate_id") is not None
    }
    raw_keys: list[set[str]] = []
    for line in pool["line_objects"]:
        keys = ownership_keys(line)
        for member_id in line.get("cluster_member_ids") or []:
            member = candidates_by_id.get(str(member_id))
            if member is not None:
                keys.update(ownership_keys(member))
        raw_keys.append(keys)
    group_sizes: dict[str, int] = {}
    for keys in raw_keys:
        if len(keys) == 1:
            key = next(iter(keys))
            group_sizes[key] = group_sizes.get(key, 0) + 1
    return [
        {
            "keys": sorted(keys),
            "unique_key": next(iter(keys)) if len(keys) == 1 else None,
            "group_size": group_sizes.get(next(iter(keys)), 0) if len(keys) == 1 else 0,
        }
        for keys in raw_keys
    ]


def _attach_ownership(cases: list[dict[str, Any]], v51_root: Path) -> None:
    for case in cases:
        case_id = case["case_id"]
        pool = json.loads((v51_root / "primary" / case_id / "pool" / f"{case_id}.json").read_text())
        ownership = _ownership_rows(pool)
        if len(ownership) != len(case["candidates"]):
            raise ValueError(f"{case_id}: ownership/candidate length mismatch")
        case["ownership"] = ownership
        for feature, row in zip(case["features"], ownership):
            feature["explicit_ownership_key_count"] = float(len(row["keys"]))
            feature["explicit_unique_ownership"] = float(row["unique_key"] is not None)
            feature["explicit_ownership_group_size"] = float(row["group_size"])


def _claimed_parent_keys(case: dict[str, Any]) -> set[str]:
    baseline_keys = {
        value
        for prediction in case["baseline_predictions"]
        for value in _candidate_keys(prediction)
    }
    claimed: set[str] = set()
    for candidate, ownership in zip(case["candidates"], case["ownership"]):
        if _candidate_keys(candidate) & baseline_keys:
            claimed.update(ownership["keys"])
    return claimed


def _select(case: dict[str, Any], probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    selected = [copy.deepcopy(row) for row in case["baseline_predictions"]]
    claimed = _claimed_parent_keys(case)
    order = sorted(
        range(len(probabilities)),
        key=lambda index: (
            -float(probabilities[index]),
            str(case["candidates"][index].get("candidate_id") or case["candidates"][index].get("bend_id")),
        ),
    )
    baseline_keys = {
        value
        for prediction in case["baseline_predictions"]
        for value in _candidate_keys(prediction)
    }
    for index in order:
        if float(probabilities[index]) < threshold:
            continue
        candidate = case["candidates"][index]
        ownership = case["ownership"][index]
        key = ownership["unique_key"]
        if key is None or key in claimed or _candidate_keys(candidate) & baseline_keys:
            continue
        prediction = _as_prediction(case, candidate)
        prediction["selector_score"] = round(float(probabilities[index]), 12)
        prediction["selection_route"] = "explicit_parent_pair_ownership_lofo_v1_addition"
        prediction["ownership_key_sha256"] = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if any(_duplicate(prediction, prior) for prior in selected):
            continue
        selected.append(prediction)
        claimed.add(key)
    selected.sort(key=lambda row: str(row["bend_id"]))
    return selected


def _score(case: dict[str, Any], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = _select(case, probabilities, threshold)
    result = match_lines(case["gold"], predictions, MATCH_THRESHOLDS)
    result.update(case_id=case["case_id"], family_id=case["family_id"], predictions=predictions)
    return result


def _threshold(inner_predictions: list[tuple[dict[str, Any], np.ndarray]]) -> float:
    best = None
    for threshold in THRESHOLD_GRID:
        metrics = _aggregate([_score(case, probabilities, threshold) for case, probabilities in inner_predictions])
        key = (float(metrics["f1"] or 0.0), float(metrics["precision"] or 0.0), threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    assert best is not None
    return best[1]


def _crossfit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    outer = []
    for outer_index, outer_case in enumerate(cases):
        train = [case for index, case in enumerate(cases) if index != outer_index]
        inner_predictions = []
        for inner_index, inner_case in enumerate(train):
            inner_train = [case for index, case in enumerate(train) if index != inner_index]
            vectorizer, model = _fit(inner_train)
            inner_predictions.append((inner_case, _probabilities(vectorizer, model, inner_case)))
        threshold = _threshold(inner_predictions)
        vectorizer, model = _fit(train)
        scored = _score(outer_case, _probabilities(vectorizer, model, outer_case), threshold)
        scored["selected_threshold"] = threshold
        scored["outer_case_excluded_from_fit_and_threshold"] = True
        scored["added_prediction_count"] = len(scored["predictions"]) - len(outer_case["baseline_predictions"])
        outer.append(scored)
    return {"metrics": _aggregate(outer), "outer_cases": outer}


def _stable(value: dict[str, Any]) -> str:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--v51-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cases = _load_cases(json.loads(args.public_manifest.read_text()), json.loads(args.gold_catalog.read_text()), args.v51_root)
    _attach_ownership(cases, args.v51_root)
    primary = _crossfit(cases)
    replay = _crossfit(cases)
    if _stable(primary) != _stable(replay):
        raise RuntimeError("nested LOFO replay mismatch")
    baseline = _baseline(cases)
    before, after = baseline["metrics"], primary["metrics"]
    keys = ("precision", "recall", "f1", "exact_count_accuracy", "exact_localized_target_set_accuracy", "minimum_family_recall")
    deltas = {key: after[key] - before[key] for key in keys}
    changed_cases = sum(row["added_prediction_count"] > 0 for row in primary["outer_cases"])
    added_predictions = sum(row["added_prediction_count"] for row in primary["outer_cases"])
    for stage, result in (("primary", primary), ("replay", replay)):
        for row in result["outer_cases"]:
            _atomic(args.output_root / stage / row["case_id"] / "semantic_result.json", {
                "schema": "SHERMAN_QC_EXP_SCAN_0003_RESULT_V1",
                "experiment_id": "EXP_SCAN_0003",
                "case_id": row["case_id"],
                "predictions": row["predictions"],
                "selected_threshold": row["selected_threshold"],
                "added_prediction_count": row["added_prediction_count"],
                "outer_case_excluded_from_fit_and_threshold": True,
                "gold_or_expected_counts_used_at_inference": False,
            })
    protected_nonregression = all(deltas[key] >= 0.0 for key in keys if key != "f1")
    report = {
        "schema": "SHERMAN_QC_EXP_SCAN_0003_NESTED_LOFO_REPORT_V1",
        "experiment_id": "EXP_SCAN_0003",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL_CROSSFIT",
        "promotion_eligible": False,
        "baseline": baseline,
        "candidate": primary,
        "deltas": deltas,
        "changed_cases": changed_cases,
        "added_predictions": added_predictions,
        "replay_sha256": _stable(replay),
        "semantic_replay_byte_identical": True,
        "elapsed_seconds": time.perf_counter() - started,
        "success_criterion_passed_before_visual": bool(changed_cases > 0 and deltas["f1"] > 0.0 and protected_nonregression),
    }
    _atomic(args.output_root / "REPORT.json", report)
    print(json.dumps({
        "baseline_f1": before["f1"], "candidate_f1": after["f1"], "deltas": deltas,
        "changed_cases": changed_cases, "added_predictions": added_predictions,
        "elapsed_seconds": report["elapsed_seconds"], "success_before_visual": report["success_criterion_passed_before_visual"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

