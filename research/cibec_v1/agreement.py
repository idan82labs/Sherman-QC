"""Frozen CIBEC_V1 inter-annotator line agreement.

Historical detector scoring intentionally remains in
``scripts.evaluate_f1_against_manual_gold``.  This module implements the
professor's distinct CIBEC rule: angle <= 5 degrees, transverse offset <= 4 mm,
and overlap / shorter span >= 0.60.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AgreementThresholds:
    angle_deg: float = 5.0
    transverse_offset_mm: float = 4.0
    shorter_span_overlap: float = 0.60
    minimum_span_mm: float = 8.0


def _line(value: Any, minimum_span_mm: float) -> np.ndarray:
    line = np.asarray(value, dtype=np.float64)
    if line.shape != (2, 3) or not np.all(np.isfinite(line)):
        raise ValueError("line must be a finite 2x3 array")
    # Decimal-mm endpoints can reconstruct an exact 8.000000 mm span as the
    # adjacent binary64 value below 8.0 (for example -34.94 to -26.94).
    # The tolerance is three orders smaller than the canonical 1e-6 mm grid,
    # so it accepts the specified boundary without admitting a representable
    # one-micrometre undercut.
    span = float(np.linalg.norm(line[1] - line[0]))
    boundary_tolerance = max(1e-12, abs(minimum_span_mm) * 1e-13)
    if span < minimum_span_mm - boundary_tolerance:
        raise ValueError(f"line span must be at least {minimum_span_mm:.1f} mm")
    return line


def pair_metrics(first_raw: Any, second_raw: Any, *, minimum_span_mm: float = 8.0) -> dict[str, float]:
    first = _line(first_raw, minimum_span_mm)
    second = _line(second_raw, minimum_span_mm)
    first_mid = first.mean(axis=0); second_mid = second.mean(axis=0)
    first_delta = first[1] - first[0]; second_delta = second[1] - second[0]
    first_length = float(np.linalg.norm(first_delta)); second_length = float(np.linalg.norm(second_delta))
    first_axis = first_delta / first_length; second_axis = second_delta / second_length
    dot = float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if dot < 0.0:
        second_axis *= -1.0
    midpoint_delta = second_mid - first_mid
    first_offset = float(np.linalg.norm(midpoint_delta - first_axis * np.dot(midpoint_delta, first_axis)))
    second_offset = float(np.linalg.norm(midpoint_delta - second_axis * np.dot(midpoint_delta, second_axis)))
    first_interval = sorted(float(np.dot(point - first_mid, first_axis)) for point in first)
    second_interval = sorted(float(np.dot(point - first_mid, first_axis)) for point in second)
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": max(first_offset, second_offset),
        "shorter_span_overlap": overlap / min(first_length, second_length),
        "first_span_overlap": overlap / first_length,
        "second_span_overlap": overlap / second_length,
        "first_length_mm": first_length,
        "second_length_mm": second_length,
    }


def is_agreement(metrics: dict[str, float], thresholds: AgreementThresholds = AgreementThresholds()) -> bool:
    return (
        metrics["axis_angle_deg"] <= thresholds.angle_deg
        and metrics["transverse_offset_mm"] <= thresholds.transverse_offset_mm
        and metrics["shorter_span_overlap"] >= thresholds.shorter_span_overlap
    )


def _cost(metrics: dict[str, float], thresholds: AgreementThresholds) -> float:
    return (
        metrics["axis_angle_deg"] / thresholds.angle_deg
        + metrics["transverse_offset_mm"] / thresholds.transverse_offset_mm
        + 1.0 - min(metrics["shorter_span_overlap"], 1.0)
    )


def match_lines(first: list[dict[str, Any]], second: list[dict[str, Any]],
                thresholds: AgreementThresholds = AgreementThresholds()) -> dict[str, Any]:
    """Maximum-cardinality deterministic one-to-one agreement matching."""
    first_rows = [(str(row["event_id"]), _line([row["line_start_mm"], row["line_end_mm"]], thresholds.minimum_span_mm)) for row in first]
    second_rows = [(str(row["event_id"]), _line([row["line_start_mm"], row["line_end_mm"]], thresholds.minimum_span_mm)) for row in second]
    matrix = np.full((len(first_rows), len(second_rows)), 1e9, dtype=np.float64)
    cache: dict[tuple[int, int], dict[str, float]] = {}
    width = max(len(second_rows), 1)
    for i, (_first_id, first_line) in enumerate(first_rows):
        for j, (_second_id, second_line) in enumerate(second_rows):
            metrics = pair_metrics(first_line, second_line, minimum_span_mm=thresholds.minimum_span_mm)
            cache[(i, j)] = metrics
            if is_agreement(metrics, thresholds):
                matrix[i, j] = _cost(metrics, thresholds) + (i * width + j) * 1e-12
    matches = []
    matched_first: set[int] = set(); matched_second: set[int] = set()
    if first_rows and second_rows:
        first_indices, second_indices = linear_sum_assignment(matrix)
        for i, j in zip(first_indices, second_indices):
            if matrix[i, j] >= 1e9:
                continue
            matched_first.add(int(i)); matched_second.add(int(j))
            matches.append({
                "first_event_id": first_rows[i][0],
                "second_event_id": second_rows[j][0],
                **{key: round(value, 6) for key, value in cache[(i, j)].items()},
            })
    matches.sort(key=lambda row: (row["first_event_id"], row["second_event_id"]))
    return {
        "first_count": len(first_rows),
        "second_count": len(second_rows),
        "true_positive": len(matches),
        "false_positive": len(second_rows) - len(matches),
        "false_negative": len(first_rows) - len(matches),
        "exact_localized_target_set": len(matches) == len(first_rows) == len(second_rows),
        "matches": matches,
        "unmatched_first_event_ids": [row[0] for index, row in enumerate(first_rows) if index not in matched_first],
        "unmatched_second_event_ids": [row[0] for index, row in enumerate(second_rows) if index not in matched_second],
    }
