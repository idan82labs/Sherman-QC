#!/usr/bin/env python3
"""Apply and score the development-only V52 precision gate from cached V51 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_post_v51_precision_ownership_v52_development import (  # noqa: E402
    CALIBRATED_NUMERIC_CUTOFF_COUNT,
    CALIBRATION_TRUTH_TIER,
    PROFILE,
    compose_v52_development,
)
from scripts.evaluate_bend_line_gold_catalog import evaluate  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds  # noqa: E402


DEFAULT_V51_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2/primary"
)
DEFAULT_CATALOG = (
    ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output/research/f1-v52-today-precision-development-20260831"
)
EXPECTED_CASE_IDS = tuple(f"T{index:03d}" for index in range(1, 29))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _today_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(case.get("case_id")): case for case in catalog.get("cases") or []}
    if set(EXPECTED_CASE_IDS) - set(by_id):
        missing = sorted(set(EXPECTED_CASE_IDS) - set(by_id))
        raise ValueError(f"development catalog is missing today's cases: {missing}")
    cases = [by_id[case_id] for case_id in EXPECTED_CASE_IDS]
    unexpected_tier = [
        str(case["case_id"])
        for case in cases
        if str(case.get("truth_tier")) != CALIBRATION_TRUTH_TIER
    ]
    if unexpected_tier:
        raise ValueError(f"today cases have an unexpected truth tier: {unexpected_tier}")
    result = dict(catalog)
    result["cases"] = cases
    result["source_snapshot_sha256"] = hashlib.sha256(_canonical(cases)).hexdigest()
    return result


def _delta(successor: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    left = successor["metrics"]
    right = incumbent["metrics"]
    keys = (
        "prediction_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "exact_case_count",
        "exact_case_rate",
    )
    return {
        key: round(float(left[key]) - float(right[key]), 6)
        if isinstance(left[key], float) or isinstance(right[key], float)
        else int(left[key]) - int(right[key])
        for key in keys
    }


def run(v51_root: Path, catalog_path: Path, output_root: Path) -> dict[str, Any]:
    v51_root = v51_root.resolve()
    catalog_path = catalog_path.resolve()
    output_root = output_root.resolve()
    if output_root == v51_root or v51_root in output_root.parents:
        raise ValueError("output root must not be the frozen V51 artifact tree")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty successor output: {output_root}")

    catalog = _today_catalog(_read(catalog_path))
    v51_prediction_map: dict[str, str] = {}
    v52_prediction_map: dict[str, str] = {}
    input_manifest: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    family_retained: Counter[str] = Counter()

    for case_id in EXPECTED_CASE_IDS:
        case_root = v51_root / case_id
        v51_path = case_root / "semantic_result.json"
        pool_path = case_root / "pool" / f"{case_id}.json"
        if not v51_path.is_file() or not pool_path.is_file():
            raise FileNotFoundError(f"{case_id}: cached V51 result or candidate pool is missing")
        v51_payload = _read(v51_path)
        pool_payload = _read(pool_path)
        v52_payload = compose_v52_development(v51_payload, pool_payload)
        case_output = output_root / "cases" / case_id / "semantic_result.json"
        _write(case_output, v52_payload)
        v51_prediction_map[case_id] = str(v51_path)
        v52_prediction_map[case_id] = str(case_output)
        input_manifest.append(
            {
                "case_id": case_id,
                "v51_path": str(v51_path),
                "v51_sha256": _sha256(v51_path),
                "pool_path": str(pool_path),
                "pool_sha256": _sha256(pool_path),
                "v52_path": str(case_output),
                "v52_sha256": _sha256(case_output),
            }
        )
        reason_counts.update(
            str(row["reason"]) for row in v52_payload["precision_ownership_decisions"]
        )
        family_retained.update(
            str(row.get("source_kind") or "") for row in v52_payload["predictions"]
        )

    thresholds = Thresholds()
    incumbent_report = evaluate(
        catalog, v51_prediction_map, tier="all", thresholds=thresholds
    )
    successor_report = evaluate(
        catalog, v52_prediction_map, tier="all", thresholds=thresholds
    )
    delta = _delta(successor_report, incumbent_report)
    retained_count = int(successor_report["metrics"]["prediction_count"])
    # This interval is already optimistic because it assumes a preregistered
    # classifier.  Here the same 14 outcomes informed threshold selection.
    nominal_precision_lower_95 = (
        round(0.025 ** (1.0 / retained_count), 6) if retained_count else 0.0
    )
    summary = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v52_today_precision_cached_development_summary",
        "profile": PROFILE,
        "cohort": "2026-08-30_today_28",
        "case_count": len(EXPECTED_CASE_IDS),
        "runtime_inputs": "cached_source_only_v51_result_and_candidate_pool",
        "inference_labels_used": False,
        "development_tuning_labels_used": True,
        "development_tuning_truth_tier": CALIBRATION_TRUTH_TIER,
        "authoritative_ground_truth": False,
        "promotion_eligible": False,
        "incumbent_metrics": incumbent_report["metrics"],
        "successor_metrics": successor_report["metrics"],
        "delta_successor_minus_incumbent": delta,
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "retained_source_family_counts": dict(sorted(family_retained.items())),
        "resubstitution_risk": {
            "calibration_case_count": len(EXPECTED_CASE_IDS),
            "evaluation_case_count": len(EXPECTED_CASE_IDS),
            "overlapping_calibration_evaluation_case_count": len(EXPECTED_CASE_IDS),
            "overlap_fraction": 1.0,
            "independent_holdout_case_count": 0,
            "calibrated_numeric_cutoff_count": CALIBRATED_NUMERIC_CUTOFF_COUNT,
            "prediction_outcomes_available_for_tuning": int(
                incumbent_report["metrics"]["prediction_count"]
            ),
            "prediction_outcomes_per_numeric_cutoff": round(
                float(incumbent_report["metrics"]["prediction_count"])
                / CALIBRATED_NUMERIC_CUTOFF_COUNT,
                6,
            ),
            "observed_f1_lift": delta["f1"],
            "held_out_verified_f1_lift": None,
            "observed_lift_resubstitution_fraction": 1.0,
            "retained_predictions": retained_count,
            "retained_false_positives": int(successor_report["metrics"]["false_positive"]),
            "nominal_preregistered_exact_binomial_precision_interval_95": [
                nominal_precision_lower_95,
                1.0,
            ],
            "interval_valid_after_same_cohort_threshold_tuning": False,
        },
        "limitations": [
            "Thresholds were tuned and evaluated on the same 28-case AI-adjudicated development cohort.",
            "The result is an in-sample development upper bound, not held-out generalization evidence.",
            "The precision-first gate intentionally leaves V51 false negatives unrecovered.",
        ],
    }
    _write(output_root / "input_manifest.json", {"cases": input_manifest})
    _write(output_root / "prediction_map_v51.json", v51_prediction_map)
    _write(output_root / "prediction_map_v52.json", v52_prediction_map)
    _write(output_root / "today_accuracy_v51_reference.json", incumbent_report)
    _write(output_root / "today_accuracy_v52.json", successor_report)
    _write(output_root / "score_delta.json", delta)
    _write(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v51-root", type=Path, default=DEFAULT_V51_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    summary = run(args.v51_root, args.catalog, args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
