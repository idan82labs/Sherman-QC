#!/usr/bin/env python3
"""Independent one-shot validator for the preregistered V17 intervention."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import endpoints
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.select_f1_full_mesh_adjacency_hysteresis_v17 import (
    FROZEN_V11_DUPLICATE_PREDICATE_SHA256,
    assert_frozen_duplicate_predicate,
)
from scripts.select_f1_mesh_route_v16 import ROUTE_FULL, build_result, select_unified
from scripts.select_f1_mesh_route_v17 import (
    PROFILE as V17_PROFILE,
    RESCUE_ROUTE,
    build_result_v17,
    select_unified_v17,
)
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, _similar


# Intentionally duplicated from the professor packet instead of importing the
# implementation constants.  This validator is an independent reconstruction.
_MIN_FINITE = 0.90
_MIN_RUN = 0.90
_MIN_ENDPOINT = 2.0 / 3.0
_MAX_P90 = 3.0
_MIN_RATIO = 10.0
_MIN_TRANSITION = 0.25
_MIN_NORMAL_ANGLE = 45.0
_MIN_BALANCE = 0.50
_MIN_SIDE = 0.20
_MIN_COHERENCE = 0.50


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def independent_eligible(row: dict[str, Any]) -> bool:
    if str(row.get("representative_source") or "") != ADJACENCY_SOURCE:
        return False
    finite = _number(row, "finite_support_fraction")
    if finite is None or finite < _MIN_FINITE:
        return False
    run = _number(row, "finite_support_longest_run_fraction")
    if run is None or run < _MIN_RUN:
        return False
    endpoint = _number(row, "endpoint_support_fraction")
    if endpoint is None or endpoint < _MIN_ENDPOINT:
        return False
    p90 = _number(row, "p90_line_distance_mm")
    if p90 is None or not (0.0 < p90 <= _MAX_P90):
        return False
    span = _number(row, "span_mm")
    if span is None or span < _MIN_RATIO * p90:
        return False
    transition = _number(row, "two_sided_transition_score")
    if transition is None or transition < _MIN_TRANSITION:
        return False
    angle = _number(row, "two_sided_normal_angle_deg")
    if angle is None or angle < _MIN_NORMAL_ANGLE:
        return False
    balance = _number(row, "two_sided_support_balance")
    if balance is None or balance < _MIN_BALANCE:
        return False
    side = _number(row, "two_sided_min_support_fraction")
    if side is None or side < _MIN_SIDE:
        return False
    coherence = _number(row, "two_sided_normal_coherence")
    if coherence is None or coherence < _MIN_COHERENCE:
        return False
    return True


def _independent_key(row: dict[str, Any]) -> tuple[Any, ...]:
    ratio = float(row["span_mm"]) / float(row["p90_line_distance_mm"])
    return (
        -ratio,
        -float(row["two_sided_transition_score"]),
        -float(row["finite_support_longest_run_fraction"]),
        -float(row["endpoint_support_fraction"]),
        -float(row["two_sided_normal_coherence"]),
        -float(row["two_sided_min_support_fraction"]),
        -float(row["two_sided_support_balance"]),
        -float(row["span_mm"]),
        str(row.get("candidate_id") or ""),
    )


def independent_rescue(
    rows: list[dict[str, Any]], incumbents: list[int]
) -> tuple[list[int], list[str]]:
    incumbent_set = set(incumbents)
    eligible = [
        index
        for index, row in enumerate(rows)
        if index not in incumbent_set and independent_eligible(row)
    ]
    eligible.sort(key=lambda index: _independent_key(rows[index]))
    retained = list(incumbents)
    rescued: list[str] = []
    for index in eligible:
        if any(_similar(rows[index], rows[prior]) for prior in retained):
            continue
        retained.append(index)
        rescued.append(str(rows[index].get("candidate_id") or ""))
    return retained, rescued


def _predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(row)[0].tolist(),
            "line_end": endpoints(row)[1].tolist(),
        }
        for index, row in enumerate(rows)
    ]


def _metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(score["true_positive"]) for score in scores)
    fp = sum(int(score["false_positive"]) for score in scores)
    fn = sum(int(score["false_negative"]) for score in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(score["exact"]) for score in scores),
        "case_count": len(scores),
    }


def _geometry_signature(row: dict[str, Any]) -> str:
    line = endpoints(row).tolist()
    return json.dumps(line, sort_keys=True, separators=(",", ":"))


def _serialize_result(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_case_source_only(pool: dict, v9: dict, v4: dict) -> dict[str, Any]:
    """Reconstruct V17 independently without reading labels or gold."""

    baseline = select_unified(pool, v9, v4)
    actual = select_unified_v17(pool, v9, v4)
    rows = baseline["rows"]
    baseline_ids = [str(row["candidate_id"]) for row in baseline["prediction_rows"]]
    row_index = {str(row["candidate_id"]): index for index, row in enumerate(rows)}
    incumbents = [row_index[candidate_id] for candidate_id in baseline_ids]
    if baseline["route"] == ROUTE_FULL:
        expected_indices, expected_rescues = independent_rescue(rows, incumbents)
    else:
        expected_indices, expected_rescues = incumbents, []
    expected_ids = [str(rows[index]["candidate_id"]) for index in expected_indices]
    actual_ids = [str(row["candidate_id"]) for row in actual["prediction_rows"]]
    actual_rescues = [
        str(item["candidate_id"])
        for item in actual["mesh_adjacency_axial_hysteresis_rescues"]
    ]
    if actual_ids != expected_ids or actual_rescues != expected_rescues:
        raise AssertionError("independent V17 reconstruction disagrees with implementation")

    baseline_result = build_result(pool, baseline)
    actual_result = build_result_v17(pool, actual)
    if not expected_rescues and _serialize_result(actual_result) != _serialize_result(baseline_result):
        raise AssertionError("V17 no-op result is not byte-identical to V16")
    return {
        "route": str(baseline["route"]),
        "baseline_rows": baseline["prediction_rows"],
        "v17_rows": actual["prediction_rows"],
        "baseline_ids": baseline_ids,
        "v17_ids": actual_ids,
        "rescue_ids": expected_rescues,
        "no_op_v16_bytes_identical": not expected_rescues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert_frozen_duplicate_predicate()
    pool_manifest_path = args.pool_dir / "manifest.json"
    manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_point_serialization") is not False:
        raise RuntimeError("frozen source-point serialization contract is not false")
    gold_payload = json.loads(args.gold_manifest.read_text(encoding="utf-8"))
    gold = {str(row["case_token"]): row for row in gold_payload["cases"]}
    v9 = joblib.load(args.v9_model)
    v4 = joblib.load(args.v4_model)

    cases: list[dict[str, Any]] = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool_path = Path(item.get("path") or args.pool_dir / f"{token}.json")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        source = validate_case_source_only(pool, v9, v4)
        baseline_score = match_lines(
            gold[token]["manual_bends"], _predictions(source["baseline_rows"]), Thresholds()
        )
        v17_score = match_lines(
            gold[token]["manual_bends"], _predictions(source["v17_rows"]), Thresholds()
        )
        incumbent_geometry_preserved = all(
            _geometry_signature(source["baseline_rows"][index])
            == _geometry_signature(source["v17_rows"][index])
            for index in range(len(source["baseline_rows"]))
        )
        cases.append(
            {
                "case_token": token,
                "part_key": str(gold[token]["part_key"]),
                "route": source["route"],
                "baseline_ids": source["baseline_ids"],
                "v17_ids": source["v17_ids"],
                "rescue_ids": source["rescue_ids"],
                "incumbent_geometry_preserved": incumbent_geometry_preserved,
                "no_op_v16_bytes_identical": source["no_op_v16_bytes_identical"],
                "v16": baseline_score,
                "v17": v17_score,
            }
        )

    v16_metrics = _metrics([case["v16"] for case in cases])
    v17_metrics = _metrics([case["v17"] for case in cases])
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        families[case["part_key"]].append(case)
    family_metrics = {
        key: _metrics([case["v17"] for case in values])
        for key, values in sorted(families.items())
    }
    c017 = next(case for case in cases if case["case_token"] == "C017")
    newly_added_count = sum(len(case["rescue_ids"]) for case in cases)
    tp_gain = v17_metrics["true_positive"] - v16_metrics["true_positive"]
    fp_gain = v17_metrics["false_positive"] - v16_metrics["false_positive"]
    previously_exact_preserved = all(
        not case["v16"]["exact"] or case["v17"]["exact"] for case in cases
    )
    gates = {
        "all_incumbent_geometry_preserved": all(
            case["incumbent_geometry_preserved"] for case in cases
        ),
        "no_previously_matched_tp_lost": all(
            int(case["v17"]["true_positive"]) >= int(case["v16"]["true_positive"])
            for case in cases
        ),
        "every_rescue_scores_tp_without_fp": tp_gain == newly_added_count and fp_gain == 0,
        "c017_exact_2tp_0fp_0fn": (
            c017["v17"]["true_positive"] == 2
            and c017["v17"]["false_positive"] == 0
            and c017["v17"]["false_negative"] == 0
        ),
        "pooled_tp_at_least_134": v17_metrics["true_positive"] >= 134,
        "pooled_fp_at_most_60": v17_metrics["false_positive"] <= 60,
        "pooled_fn_at_most_42": v17_metrics["false_negative"] <= 42,
        "pooled_f1_at_least_0_724324": v17_metrics["f1"] >= 0.724324,
        "minimum_family_recall_positive": min(
            (value["recall"] for value in family_metrics.values()), default=0.0
        ) > 0.0,
        "minimum_family_f1_positive": min(
            (value["f1"] for value in family_metrics.values()), default=0.0
        ) > 0.0,
        "exact_cases_at_least_5": v17_metrics["exact_case_count"] >= 5,
        "previously_exact_cases_preserved": previously_exact_preserved,
        "independent_reconstruction_passed": True,
        "source_point_serialization_false": manifest["source_point_serialization"] is False,
    }
    report = {
        "schema_version": 1,
        "kind": "f1_v17_mesh_adjacency_axial_hysteresis_one_shot_regression",
        "profile": V17_PROFILE,
        "case_count": len(cases),
        "family_count": len(families),
        "route_counts": dict(sorted(Counter(case["route"] for case in cases).items())),
        "activation_case_count": sum(bool(case["rescue_ids"]) for case in cases),
        "rescue_count": newly_added_count,
        "v16_metrics": v16_metrics,
        "v17_metrics": v17_metrics,
        "true_positive_gain": tp_gain,
        "false_positive_gain": fp_gain,
        "min_family_recall_v17": min(
            (value["recall"] for value in family_metrics.values()), default=0.0
        ),
        "min_family_f1_v17": min(
            (value["f1"] for value in family_metrics.values()), default=0.0
        ),
        "falsification_gates": gates,
        "all_falsification_gates_passed": all(gates.values()),
        "architecture_ruling": "RETAIN" if all(gates.values()) else "STOP",
        "frozen_inputs": {
            "pool_manifest_sha256": _sha(pool_manifest_path),
            "gold_manifest_sha256": _sha(args.gold_manifest),
            "v9_model_sha256": _sha(args.v9_model),
            "v4_model_sha256": _sha(args.v4_model),
            "v11_duplicate_predicate_sha256": FROZEN_V11_DUPLICATE_PREDICATE_SHA256,
        },
        "family_metrics": family_metrics,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "family_count",
                    "activation_case_count",
                    "rescue_count",
                    "v16_metrics",
                    "v17_metrics",
                    "true_positive_gain",
                    "false_positive_gain",
                    "all_falsification_gates_passed",
                    "architecture_ruling",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
