#!/usr/bin/env python3
"""Score F1 bend-line instances against the adjudicated 13-scan gold set.

Matching is one-to-one.  Count equality alone cannot pass: a prediction must
also agree in undirected axis angle, symmetric transverse offset, and finite
gold-span overlap.  Defaults intentionally reject the 4.8 mm ysherman tongue
silhouette that visual review identified as the wrong parallel line.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "experiments" / "f1-manual-gold-baseline-20260716.json"

DEFAULT_CASES = {
    "01_10839000": "output/experiments/f1-10of13/full-zero-shot-current/10839000/10839000_01_full_scan.ply--1c826530ab64/decomposition_result.json",
    "02_10839000": "output/experiments/f1-10of13/full-zero-shot-current/10839000/10839000_01_scan_1.ply--98b00841230a/decomposition_result.json",
    "03_10839000": "output/experiments/f1-10of13/boundary-aware-render/03_10839000/10839000/decomposition_result.json",
    "04_15651004": "output/experiments/f1-10of13/zero-shot-auto/04_15651004/15651004/decomposition_result.json",
    "05_37361005": "output/experiments/f1-10of13/full-zero-shot-current/37361005/decomposition_result.json",
    "06_38172000": "output/experiments/f1-10of13/plane-lattice-probe/06_38172000/plane_lattice_manifest.json",
    "07_38172000": "output/experiments/f1-10of13/structural-render-v2/07_38172000/38172000/decomposition_result.json",
    "08_40500000": "output/experiments/f1-10of13/plane-lattice-probe/08_40500000/plane_lattice_manifest.json",
    "09_47959001": "output/experiments/f1-10of13/corridor-recovery/09_47959001/47959001/decomposition_result.json",
    "10_49001000": "output/experiments/f1-10of13/supported-strip/10_49001000/49001000/decomposition_result.json",
    "11_49024000": "output/experiments/f1-10of13/small-tight-marker/11_49024000/49024000/decomposition_result.json",
    "12_49125000": "output/experiments/f1-10of13/zero-shot-auto/12_49125000/49125000/decomposition_result.json",
    "13_ysherman": "output/experiments/f1-10of13/corridor-completion-integrated-v2/13_ysherman/ysherman/decomposition_result.json",
}

CASE_JOBS = {
    "01_10839000": "bend_label_01_10839000",
    "02_10839000": "bend_label_02_10839000",
    "03_10839000": "bend_label_03_10839000",
    "04_15651004": "bend_label_04_15651004",
    "05_37361005": "bend_label_05_37361005",
    "06_38172000": "bend_label_06_38172000",
    "07_38172000": "bend_label_07_38172000",
    "08_40500000": "bend_label_08_40500000",
    "09_47959001": "bend_label_09_47959001",
    "10_49001000": "bend_label_10_49001000",
    "11_49024000": "bend_label_11_49024000",
    "12_49125000": "bend_label_12_49125000",
    "13_ysherman": "bend_label_13_ysherman",
}


@dataclass(frozen=True)
class Thresholds:
    angle_deg: float = 5.0
    transverse_offset_mm: float = 4.0
    gold_span_overlap: float = 0.60
    prediction_span_overlap: float = 0.60


def _endpoints(record: dict[str, Any]) -> np.ndarray:
    endpoints = record.get("span_endpoints")
    if endpoints:
        return np.asarray(endpoints, dtype=np.float64)
    if record.get("line_start") is not None and record.get("line_end") is not None:
        return np.asarray([record["line_start"], record["line_end"]], dtype=np.float64)
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    direction = np.asarray(record["axis_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    half_span = direction * float(record["span_mm"]) / 2.0
    return np.asarray([anchor - half_span, anchor + half_span])


def pair_metrics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gold_midpoint = np.mean(gold, axis=0)
    prediction_midpoint = np.mean(prediction, axis=0)
    gold_delta = gold[1] - gold[0]
    prediction_delta = prediction[1] - prediction[0]
    gold_length = float(np.linalg.norm(gold_delta))
    prediction_length = float(np.linalg.norm(prediction_delta))
    if gold_length <= 1e-9 or prediction_length <= 1e-9:
        raise ValueError("bend lines must have nonzero length")
    gold_axis = gold_delta / gold_length
    prediction_axis = prediction_delta / prediction_length
    dot = float(np.clip(np.dot(gold_axis, prediction_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0:
        prediction_axis *= -1

    midpoint_delta = prediction_midpoint - gold_midpoint
    gold_axis_offset = float(np.linalg.norm(midpoint_delta - gold_axis * np.dot(midpoint_delta, gold_axis)))
    prediction_axis_offset = float(
        np.linalg.norm(midpoint_delta - prediction_axis * np.dot(midpoint_delta, prediction_axis))
    )
    transverse_offset = max(gold_axis_offset, prediction_axis_offset)

    gold_interval = sorted(float(np.dot(point - gold_midpoint, gold_axis)) for point in gold)
    prediction_interval = sorted(float(np.dot(point - gold_midpoint, gold_axis)) for point in prediction)
    overlap_length = max(
        0.0,
        min(gold_interval[1], prediction_interval[1]) - max(gold_interval[0], prediction_interval[0]),
    )
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": transverse_offset,
        "gold_span_overlap": overlap_length / gold_length,
        "prediction_span_overlap": overlap_length / prediction_length,
        "shorter_span_overlap": overlap_length / min(gold_length, prediction_length),
        "gold_length_mm": gold_length,
        "prediction_length_mm": prediction_length,
    }


def is_match(metrics: dict[str, float], thresholds: Thresholds) -> bool:
    return (
        metrics["axis_angle_deg"] <= thresholds.angle_deg
        and metrics["transverse_offset_mm"] <= thresholds.transverse_offset_mm
        and metrics["gold_span_overlap"] >= thresholds.gold_span_overlap
        and metrics["prediction_span_overlap"] >= thresholds.prediction_span_overlap
    )


def _match_cost(metrics: dict[str, float], thresholds: Thresholds) -> float:
    return (
        metrics["axis_angle_deg"] / thresholds.angle_deg
        + metrics["transverse_offset_mm"] / thresholds.transverse_offset_mm
        + (1.0 - min(metrics["gold_span_overlap"], 1.0))
        + (1.0 - min(metrics["prediction_span_overlap"], 1.0))
    )


def match_lines(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    thresholds: Thresholds,
) -> dict[str, Any]:
    gold = [(str(record["bend_id"]), _endpoints(record)) for record in gold_records]
    predictions = [(str(record["bend_id"]), _endpoints(record)) for record in prediction_records]
    matrix = np.full((len(gold), len(predictions)), 1e9, dtype=np.float64)
    pair_cache: dict[tuple[int, int], dict[str, float]] = {}
    for gold_index, (_gold_id, gold_line) in enumerate(gold):
        for prediction_index, (_prediction_id, prediction_line) in enumerate(predictions):
            metrics = pair_metrics(gold_line, prediction_line)
            pair_cache[(gold_index, prediction_index)] = metrics
            if is_match(metrics, thresholds):
                matrix[gold_index, prediction_index] = _match_cost(metrics, thresholds)

    matches = []
    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    if len(gold) and len(predictions):
        gold_indices, prediction_indices = linear_sum_assignment(matrix)
        for gold_index, prediction_index in zip(gold_indices, prediction_indices):
            if matrix[gold_index, prediction_index] >= 1e9:
                continue
            matched_gold.add(int(gold_index))
            matched_predictions.add(int(prediction_index))
            matches.append(
                {
                    "gold_id": gold[gold_index][0],
                    "prediction_id": predictions[prediction_index][0],
                    **{key: round(value, 6) for key, value in pair_cache[(gold_index, prediction_index)].items()},
                }
            )
    return {
        "gold_count": len(gold),
        "prediction_count": len(predictions),
        "true_positive": len(matches),
        "false_positive": len(predictions) - len(matches),
        "false_negative": len(gold) - len(matches),
        "exact": len(matches) == len(gold) == len(predictions),
        "matches": matches,
        "unmatched_gold_ids": [item[0] for index, item in enumerate(gold) if index not in matched_gold],
        "unmatched_prediction_ids": [
            item[0] for index, item in enumerate(predictions) if index not in matched_predictions
        ],
    }


def _load_case_map(path: Path | None) -> dict[str, Path]:
    values = DEFAULT_CASES if path is None else json.loads(path.read_text(encoding="utf-8"))
    return {case_id: (ROOT / value).resolve() for case_id, value in values.items()}


def _discover_isolated_case_map(prediction_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for case_id in CASE_JOBS:
        matches = sorted((prediction_root / "isolated_cases" / case_id).glob("result/**/decomposition_result.json"))
        if len(matches) != 1:
            raise ValueError(f"expected one decomposition result for {case_id}, found {len(matches)}")
        discovered[case_id] = matches[0].resolve()
    return discovered


def evaluate(case_map: dict[str, Path], thresholds: Thresholds) -> dict[str, Any]:
    results = []
    for case_id in sorted(case_map):
        gold_path = ROOT / "output" / CASE_JOBS[case_id] / "bend_instance_labels.json"
        gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
        prediction_payload = json.loads(case_map[case_id].read_text(encoding="utf-8"))
        result = match_lines(
            gold_payload["manual_bends"],
            prediction_payload.get("owned_bend_regions") or [],
            thresholds,
        )
        result.update(
            {
                "case_id": case_id,
                "gold_path": str(gold_path),
                "prediction_path": str(case_map[case_id]),
            }
        )
        results.append(result)

    true_positive = sum(result["true_positive"] for result in results)
    false_positive = sum(result["false_positive"] for result in results)
    false_negative = sum(result["false_negative"] for result in results)
    gold_count = true_positive + false_negative
    prediction_count = true_positive + false_positive
    precision = true_positive / prediction_count if prediction_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_manual_line_accuracy",
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
            "exact_part_count": sum(1 for result in results if result["exact"]),
            "exact_part_rate": round(sum(1 for result in results if result["exact"]) / len(results), 6),
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-map", type=Path, help="JSON object mapping case_id to prediction manifest")
    parser.add_argument("--prediction-root", type=Path, help="Root produced by the isolated blind runner")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angle-deg", type=float, default=5.0)
    parser.add_argument("--offset-mm", type=float, default=4.0)
    parser.add_argument("--gold-overlap", type=float, default=0.60)
    parser.add_argument("--prediction-overlap", type=float, default=0.60)
    args = parser.parse_args()
    if args.case_map and args.prediction_root:
        parser.error("use only one of --case-map or --prediction-root")
    case_map = (
        _discover_isolated_case_map(args.prediction_root.resolve())
        if args.prediction_root
        else _load_case_map(args.case_map)
    )
    report = evaluate(
        case_map,
        Thresholds(args.angle_deg, args.offset_mm, args.gold_overlap, args.prediction_overlap),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
