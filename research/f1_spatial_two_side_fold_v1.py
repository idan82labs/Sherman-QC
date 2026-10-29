"""Spatial two-side fold evidence for frozen scan-only line predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


CORRIDOR_RADIUS_MM = 12.0
CENTER_DEADBAND_MM = 1.5
MINIMUM_POINTS_PER_SIDE = 40
MINIMUM_SIDE_FRACTION = 0.15
MINIMUM_AXIAL_COVERAGE_FRACTION = 0.50
MINIMUM_TURN_DEG = 8.0
MAXIMUM_PLANE_RESIDUAL_RATIO = 0.04
SPLIT_ORIENTATION_COUNT = 18
MAXIMUM_LOCAL_POINTS = 6000


def _canonical(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("zero vector")
    result = result / norm
    dominant = int(np.argmax(np.abs(result)))
    return -result if result[dominant] < 0 else result


def _plane(rows: np.ndarray) -> tuple[np.ndarray, float]:
    centered = rows - np.mean(rows, axis=0)
    covariance = centered.T @ centered / max(1, len(rows))
    values, vectors = np.linalg.eigh(covariance)
    denominator = max(float(np.sum(values)), 1e-12)
    return _canonical(vectors[:, 0]), float(max(values[0], 0.0) / denominator)


def spatial_fold_evidence(
    points: np.ndarray,
    line_start: list[float],
    line_end: list[float],
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    start = np.asarray(line_start, dtype=np.float64)
    end = np.asarray(line_end, dtype=np.float64)
    direction = end - start
    span = float(np.linalg.norm(direction))
    if span <= 1e-9:
        return {"state": "UNRESOLVED", "reason": "DEGENERATE_LINE", "passed": None}
    unit = direction / span
    axial_all = (points - start) @ unit
    closest = start + np.clip(axial_all, 0.0, span)[:, None] * unit
    radial_all = points - closest
    distance = np.linalg.norm(radial_all, axis=1)
    mask = (axial_all >= -1.0) & (axial_all <= span + 1.0) & (distance <= CORRIDOR_RADIUS_MM)
    local = points[mask]
    axial = axial_all[mask]
    radial = radial_all[mask]
    if len(local) > MAXIMUM_LOCAL_POINTS:
        order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        selected = np.linspace(0, len(order) - 1, MAXIMUM_LOCAL_POINTS, dtype=np.int64)
        chosen = order[selected]
        local, axial, radial = local[chosen], axial[chosen], radial[chosen]
    if len(local) < 2 * MINIMUM_POINTS_PER_SIDE:
        return {
            "state": "UNRESOLVED",
            "reason": "INSUFFICIENT_CORRIDOR_POINTS",
            "passed": None,
            "corridor_point_count": int(len(local)),
        }
    reference_axes = np.eye(3)
    reference = reference_axes[int(np.argmin(np.abs(reference_axes @ unit)))]
    basis_a = _canonical(np.cross(unit, reference))
    basis_b = _canonical(np.cross(unit, basis_a))
    candidates = []
    for ordinal in range(SPLIT_ORIENTATION_COUNT):
        theta = math.pi * ordinal / SPLIT_ORIENTATION_COUNT
        transverse = math.cos(theta) * basis_a + math.sin(theta) * basis_b
        signed = radial @ transverse
        left = signed < -CENTER_DEADBAND_MM
        right = signed > CENTER_DEADBAND_MM
        count_left, count_right = int(np.sum(left)), int(np.sum(right))
        if min(count_left, count_right) < MINIMUM_POINTS_PER_SIDE:
            continue
        side_fraction = min(count_left, count_right) / (count_left + count_right)
        if side_fraction < MINIMUM_SIDE_FRACTION:
            continue
        normal_left, residual_left = _plane(local[left])
        normal_right, residual_right = _plane(local[right])
        turn = math.degrees(math.acos(float(np.clip(abs(normal_left @ normal_right), -1.0, 1.0))))
        coverage_left = min(1.0, float(np.quantile(axial[left], 0.95) - np.quantile(axial[left], 0.05)) / span)
        coverage_right = min(1.0, float(np.quantile(axial[right], 0.95) - np.quantile(axial[right], 0.05)) / span)
        minimum_coverage = min(coverage_left, coverage_right)
        maximum_residual = max(residual_left, residual_right)
        quality = (
            turn
            * math.sqrt(side_fraction)
            * min(1.0, minimum_coverage / MINIMUM_AXIAL_COVERAGE_FRACTION)
            * max(0.0, 1.0 - maximum_residual / MAXIMUM_PLANE_RESIDUAL_RATIO)
        )
        candidates.append({
            "split_ordinal": ordinal,
            "turn_deg": turn,
            "minimum_side_fraction": side_fraction,
            "minimum_axial_coverage_fraction": minimum_coverage,
            "maximum_plane_residual_ratio": maximum_residual,
            "left_point_count": count_left,
            "right_point_count": count_right,
            "quality": quality,
        })
    if not candidates:
        return {
            "state": "UNRESOLVED",
            "reason": "NO_BALANCED_SPATIAL_SPLIT",
            "passed": None,
            "corridor_point_count": int(len(local)),
        }
    best = max(candidates, key=lambda row: (row["quality"], -row["split_ordinal"]))
    passed = bool(
        best["turn_deg"] >= MINIMUM_TURN_DEG
        and best["minimum_axial_coverage_fraction"] >= MINIMUM_AXIAL_COVERAGE_FRACTION
        and best["maximum_plane_residual_ratio"] <= MAXIMUM_PLANE_RESIDUAL_RATIO
    )
    return {
        "state": "EVALUATED_PASS" if passed else "EVALUATED_FAIL",
        "reason": "SPATIAL_TWO_SIDE_FOLD" if passed else "SPATIAL_FOLD_INVARIANT_FAILED",
        "passed": passed,
        "corridor_point_count": int(len(local)),
        **best,
    }


def filter_predictions(points: np.ndarray, predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained = []
    evidence_rows = []
    for prediction in predictions:
        evidence = spatial_fold_evidence(points, prediction["line_start"], prediction["line_end"])
        keep = evidence["passed"] is not False
        evidence_rows.append({"bend_id": prediction["bend_id"], "retained": keep, **evidence})
        if keep:
            retained.append({**prediction, "spatial_two_side_fold_state": evidence["state"]})
    return retained, evidence_rows
