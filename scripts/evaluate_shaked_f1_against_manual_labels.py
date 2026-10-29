#!/usr/bin/env python3
"""Score the frozen Shaked blind F1 outputs against assistant-reviewed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_f1_against_manual_gold import Thresholds, match_lines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_ROOT = (
    ROOT / "output/enterprise_benchmark/shaked-count-backed-12-blind-baseline-20260717"
)
DEFAULT_GOLD_ROOT = (
    ROOT / "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output/manual_labels/shaked-count-backed-12-20260717/frozen_blind_f1_line_accuracy.json"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_path(prediction_root: Path, case_id: str) -> Path:
    matches = sorted(
        (prediction_root / "isolated_cases" / case_id).glob(
            "result/**/decomposition_result.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"{case_id}: expected one decomposition result, found {len(matches)}")
    return matches[0].resolve()


def evaluate(
    prediction_root: Path,
    gold_root: Path,
    thresholds: Thresholds,
) -> dict[str, Any]:
    manifest = _read(gold_root / "manifest.json")
    results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        gold_path = gold_root / case_id / "bend_instance_labels.json"
        prediction_path = _prediction_path(prediction_root, case_id)
        gold = _read(gold_path)
        prediction = _read(prediction_path)
        result = match_lines(
            gold["manual_bends"],
            prediction.get("owned_bend_regions") or [],
            thresholds,
        )
        result.update(
            {
                "case_id": case_id,
                "part_number": str(case["part_number"]),
                "gold_path": str(gold_path.resolve()),
                "prediction_path": str(prediction_path),
                "gold_review_status": gold["review_status"],
                "gold_authoritative_ground_truth": False,
            }
        )
        results.append(result)

    true_positive = sum(int(item["true_positive"]) for item in results)
    false_positive = sum(int(item["false_positive"]) for item in results)
    false_negative = sum(int(item["false_negative"]) for item in results)
    gold_count = true_positive + false_negative
    prediction_count = true_positive + false_positive
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_count_cases = sum(
        int(item["prediction_count"]) == int(item["gold_count"]) for item in results
    )
    exact_geometry_cases = sum(bool(item["exact"]) for item in results)
    return {
        "schema_version": 1,
        "kind": "shaked_12_frozen_blind_f1_manual_line_accuracy",
        "evaluation_status": "development_labels_user_review_pending",
        "prediction_status": "frozen_pre_label_blind_output_unchanged",
        "thresholds": {
            "axis_angle_deg": thresholds.angle_deg,
            "transverse_offset_mm": thresholds.transverse_offset_mm,
            "gold_span_overlap": thresholds.gold_span_overlap,
            "prediction_span_overlap": thresholds.prediction_span_overlap,
            "matching": "one_to_one_minimum_cost",
        },
        "metrics": {
            "case_count": len(results),
            "gold_bend_count": gold_count,
            "prediction_count": prediction_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "exact_count_case_count": exact_count_cases,
            "exact_geometry_case_count": exact_geometry_cases,
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--offset-mm", type=float, default=4.0)
    parser.add_argument("--gold-overlap", type=float, default=0.60)
    parser.add_argument("--prediction-overlap", type=float, default=0.60)
    args = parser.parse_args()
    report = evaluate(
        args.prediction_root.resolve(),
        args.gold_root.resolve(),
        Thresholds(
            args.angle_deg,
            args.offset_mm,
            args.gold_overlap,
            args.prediction_overlap,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
