"""Relocate a trusted CAD bend event to the intersection of observed scan flanks."""

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
MAXIMUM_RELOCATION_MM = 8.0
MAXIMUM_DIRECTION_CHANGE_DEG = 10.0


def _canonical(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("zero vector")
    vector = vector / norm
    dominant = int(np.argmax(np.abs(vector)))
    return -vector if vector[dominant] < 0 else vector


def _plane(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.mean(rows, axis=0)
    covariance = (rows - center).T @ (rows - center) / max(1, len(rows))
    values, vectors = np.linalg.eigh(covariance)
    residual = float(max(values[0], 0.0) / max(float(np.sum(values)), 1e-12))
    return center, _canonical(vectors[:, 0]), residual


def observed_intersection(points: np.ndarray, line_start: list[float], line_end: list[float]) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    start, end = np.asarray(line_start, dtype=np.float64), np.asarray(line_end, dtype=np.float64)
    nominal = end - start
    span = float(np.linalg.norm(nominal))
    if span <= 1e-9:
        return {"state": "ABSTAIN_RELOCATION", "reason": "DEGENERATE_PARENT_LINE", "relocated": False}
    unit = nominal / span
    axial_all = (points - start) @ unit
    closest = start + np.clip(axial_all, 0.0, span)[:, None] * unit
    radial_all = points - closest
    mask = (
        (axial_all >= -1.0) & (axial_all <= span + 1.0)
        & (np.linalg.norm(radial_all, axis=1) <= CORRIDOR_RADIUS_MM)
    )
    local, axial, radial = points[mask], axial_all[mask], radial_all[mask]
    if len(local) > MAXIMUM_LOCAL_POINTS:
        order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        selected = order[np.linspace(0, len(order) - 1, MAXIMUM_LOCAL_POINTS, dtype=np.int64)]
        local, axial, radial = local[selected], axial[selected], radial[selected]
    if len(local) < 2 * MINIMUM_POINTS_PER_SIDE:
        return {"state": "ABSTAIN_RELOCATION", "reason": "INSUFFICIENT_CORRIDOR_POINTS", "relocated": False, "corridor_point_count": int(len(local))}
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ unit)))]
    basis_a, basis_b = _canonical(np.cross(unit, reference)), None
    basis_b = _canonical(np.cross(unit, basis_a))
    candidates = []
    for ordinal in range(SPLIT_ORIENTATION_COUNT):
        theta = math.pi * ordinal / SPLIT_ORIENTATION_COUNT
        transverse = math.cos(theta) * basis_a + math.sin(theta) * basis_b
        signed = radial @ transverse
        left, right = signed < -CENTER_DEADBAND_MM, signed > CENTER_DEADBAND_MM
        counts = int(np.sum(left)), int(np.sum(right))
        if min(counts) < MINIMUM_POINTS_PER_SIDE:
            continue
        side_fraction = min(counts) / sum(counts)
        if side_fraction < MINIMUM_SIDE_FRACTION:
            continue
        center_a, normal_a, residual_a = _plane(local[left])
        center_b, normal_b, residual_b = _plane(local[right])
        turn = math.degrees(math.acos(float(np.clip(abs(normal_a @ normal_b), -1.0, 1.0))))
        coverage = min(
            min(1.0, float(np.quantile(axial[left], 0.95) - np.quantile(axial[left], 0.05)) / span),
            min(1.0, float(np.quantile(axial[right], 0.95) - np.quantile(axial[right], 0.05)) / span),
        )
        residual = max(residual_a, residual_b)
        quality = turn * math.sqrt(side_fraction) * min(1.0, coverage / MINIMUM_AXIAL_COVERAGE_FRACTION) * max(0.0, 1.0 - residual / MAXIMUM_PLANE_RESIDUAL_RATIO)
        candidates.append((quality, -ordinal, center_a, center_b, normal_a, normal_b, turn, side_fraction, coverage, residual))
    if not candidates:
        return {"state": "ABSTAIN_RELOCATION", "reason": "NO_BALANCED_SPATIAL_SPLIT", "relocated": False, "corridor_point_count": int(len(local))}
    quality, negative_ordinal, center_a, center_b, normal_a, normal_b, turn, side_fraction, coverage, residual = max(candidates, key=lambda row: (row[0], row[1]))
    evidence_pass = turn >= MINIMUM_TURN_DEG and coverage >= MINIMUM_AXIAL_COVERAGE_FRACTION and residual <= MAXIMUM_PLANE_RESIDUAL_RATIO
    if not evidence_pass:
        return {"state": "ABSTAIN_RELOCATION", "reason": "LOCAL_SURFACE_FIT_UNTRUSTED", "relocated": False, "turn_deg": turn, "minimum_axial_coverage_fraction": coverage, "maximum_plane_residual_ratio": residual}
    direction = np.cross(normal_a, normal_b)
    if float(np.linalg.norm(direction)) <= 1e-9:
        return {"state": "ABSTAIN_RELOCATION", "reason": "PARALLEL_FITTED_PLANES", "relocated": False}
    direction = _canonical(direction)
    if float(direction @ unit) < 0:
        direction = -direction
    direction_change = math.degrees(math.acos(float(np.clip(direction @ unit, -1.0, 1.0))))
    midpoint = 0.5 * (start + end)
    matrix = np.vstack([normal_a, normal_b, unit])
    target = np.asarray([normal_a @ center_a, normal_b @ center_b, unit @ midpoint], dtype=np.float64)
    intersection, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    delta = intersection - midpoint
    relocation = float(np.linalg.norm(delta - (delta @ unit) * unit))
    if direction_change > MAXIMUM_DIRECTION_CHANGE_DEG or relocation > MAXIMUM_RELOCATION_MM:
        return {"state": "ABSTAIN_RELOCATION", "reason": "INTERSECTION_OUTSIDE_TRUST_BOUND", "relocated": False, "direction_change_deg": direction_change, "relocation_mm": relocation}
    observed_start = intersection - 0.5 * span * direction
    observed_end = intersection + 0.5 * span * direction
    return {
        "state": "RELOCATED_TO_OBSERVED_INTERSECTION",
        "reason": "TRUSTED_LOCAL_TWO_PLANE_INTERSECTION",
        "relocated": True,
        "line_start": observed_start.tolist(),
        "line_end": observed_end.tolist(),
        "turn_deg": turn,
        "minimum_side_fraction": side_fraction,
        "minimum_axial_coverage_fraction": coverage,
        "maximum_plane_residual_ratio": residual,
        "direction_change_deg": direction_change,
        "relocation_mm": relocation,
        "split_ordinal": int(-negative_ordinal),
        "quality": quality,
    }


def relocate_predictions(points: np.ndarray, predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output, records = [], []
    for prediction in predictions:
        evidence = observed_intersection(points, prediction["line_start"], prediction["line_end"])
        row = dict(prediction)
        if evidence["relocated"]:
            row["line_start"] = evidence["line_start"]
            row["line_end"] = evidence["line_end"]
            row["detected_line_start"] = evidence["line_start"]
            row["detected_line_end"] = evidence["line_end"]
        row["scan_observed_intersection_state"] = evidence["state"]
        output.append(row)
        records.append({"bend_id": prediction["bend_id"], **evidence})
    return output, records
