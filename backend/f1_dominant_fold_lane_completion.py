"""Count-free completion for repeated parallel fold lanes with terminal bends."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("span_endpoints")
    if value is None and record.get("line_start") is not None:
        value = [record["line_start"], record["line_end"]]
    if value is None:
        anchor = np.asarray(record["anchor"], dtype=np.float64)
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        half = 0.5 * float(record["span_mm"]) * axis
        value = [anchor - half, anchor + half]
    endpoints = np.asarray(value, dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    return endpoints


def _axis_span(record: Mapping[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    endpoints = _endpoints(record)
    delta = endpoints[1] - endpoints[0]
    span = float(np.linalg.norm(delta))
    if span <= 1e-9:
        raise ValueError("line span must be nonzero")
    return delta / span, span, endpoints


def _axis_angle(left: np.ndarray, right: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0)))
    )


def _source_count(record: Mapping[str, Any]) -> int:
    return int(record.get("cluster_source_count") or 1)


def _member_count(record: Mapping[str, Any]) -> int:
    return int(record.get("cluster_member_count") or 1)


def _normal_turn(record: Mapping[str, Any]) -> float:
    return float(
        record.get("cluster_max_two_sided_normal_angle_deg")
        or record.get("two_sided_normal_angle_deg")
        or 0.0
    )


def _finite_support(record: Mapping[str, Any]) -> float:
    return float(
        record.get("cluster_max_finite_support_fraction")
        or record.get("finite_support_fraction")
        or 0.0
    )


def _pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, float, float]:
    left_axis, left_span, left_endpoints = _axis_span(left)
    right_axis, right_span, right_endpoints = _axis_span(right)
    angle = _axis_angle(left_axis, right_axis)
    left_midpoint = np.mean(left_endpoints, axis=0)
    right_midpoint = np.mean(right_endpoints, axis=0)
    delta = right_midpoint - left_midpoint
    left_offset = float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis)))
    right_offset = float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis)))
    right_interval = sorted(
        float(np.dot(point - left_midpoint, left_axis)) for point in right_endpoints
    )
    overlap = max(
        0.0,
        min(0.5 * left_span, right_interval[1])
        - max(-0.5 * left_span, right_interval[0]),
    )
    return angle, max(left_offset, right_offset), overlap / max(min(left_span, right_span), 1e-9)


def _duplicate(left: Mapping[str, Any], right: Mapping[str, Any], offset_mm: float) -> bool:
    angle, offset, overlap = _pair_metrics(left, right)
    return angle <= 7.5 and offset <= offset_mm and overlap >= 0.45


def complete_dominant_fold_lane(
    line_objects: Sequence[Mapping[str, Any]],
    *,
    axis_tolerance_deg: float = 3.0,
    core_duplicate_offset_mm: float = 3.0,
    terminal_gap_mm: float = 3.0,
) -> dict[str, Any]:
    """Return a five-axis core+terminal lane only when the topology is explicit."""

    valid = []
    for record in line_objects:
        try:
            _axis_span(record)
        except (KeyError, TypeError, ValueError):
            continue
        valid.append(dict(record))
    if not valid:
        return {"eligible": False, "reason": "no_valid_lines", "selected": []}

    dominant = max(
        valid,
        key=lambda record: (
            _source_count(record),
            _member_count(record),
            _axis_span(record)[1],
            float(record.get("support_count") or 0.0),
            str(record.get("candidate_id") or record.get("bend_id") or ""),
        ),
    )
    dominant_axis, dominant_span, dominant_endpoints = _axis_span(dominant)
    origin = np.mean(dominant_endpoints, axis=0)
    core_candidates = []
    for record in valid:
        axis, span, _record_endpoints = _axis_span(record)
        if (
            _axis_angle(axis, dominant_axis) <= axis_tolerance_deg
            and span >= 0.5 * dominant_span
            and _finite_support(record) >= 0.8
            and _source_count(record) >= 3
            and _normal_turn(record) >= 30.0
        ):
            core_candidates.append(record)
    core_candidates.sort(
        key=lambda record: (
            -_source_count(record),
            -_member_count(record),
            float(record.get("cluster_mean_median_line_distance_mm") or 1e9),
            str(record.get("candidate_id") or record.get("bend_id") or ""),
        )
    )
    core = []
    for record in core_candidates:
        if any(_duplicate(record, retained, core_duplicate_offset_mm) for retained in core):
            continue
        core.append(record)
    if len(core) != 3:
        return {
            "eligible": False,
            "reason": "core_count_not_three",
            "dominant_id": str(dominant.get("bend_id") or ""),
            "core_count": len(core),
            "selected": [],
        }

    core_intervals = []
    for record in core:
        endpoints = _endpoints(record)
        interval = sorted(float(np.dot(point - origin, dominant_axis)) for point in endpoints)
        core_intervals.append((interval[0], interval[1]))

    terminal_by_side: dict[int, list[tuple[float, dict[str, Any]]]] = {-1: [], 1: []}
    for record in valid:
        if any(record.get("candidate_id") == item.get("candidate_id") for item in core):
            continue
        axis, span, endpoints = _axis_span(record)
        if (
            _axis_angle(axis, dominant_axis) > axis_tolerance_deg
            or span >= 0.25 * dominant_span
            or _finite_support(record) < 0.8
            or _source_count(record) < 2
            or _normal_turn(record) < 30.0
        ):
            continue
        interval = sorted(float(np.dot(point - origin, dominant_axis)) for point in endpoints)
        nearest: tuple[float, int] | None = None
        for low, high in core_intervals:
            if interval[1] < low:
                value = (low - interval[1], -1)
            elif interval[0] > high:
                value = (interval[0] - high, 1)
            else:
                continue
            if value[0] <= terminal_gap_mm and (nearest is None or value < nearest):
                nearest = value
        if nearest is not None:
            terminal_by_side[nearest[1]].append((nearest[0], record))

    if not terminal_by_side[-1] or not terminal_by_side[1]:
        return {
            "eligible": False,
            "reason": "missing_terminal_side",
            "dominant_id": str(dominant.get("bend_id") or ""),
            "core_count": len(core),
            "selected": [],
        }

    terminals = []
    for side in (-1, 1):
        _gap, selected = max(
            terminal_by_side[side],
            key=lambda item: (
                _normal_turn(item[1]),
                _source_count(item[1]),
                _member_count(item[1]),
                -float(item[1].get("cluster_mean_median_line_distance_mm") or 1e9),
                -item[0],
                str(item[1].get("candidate_id") or item[1].get("bend_id") or ""),
            ),
        )
        selected = dict(selected)
        selected["selection_reason"] = "dominant_fold_lane_terminal_completion"
        selected["dominant_fold_lane_terminal_side"] = side
        terminals.append(selected)
    selected = []
    for record in core:
        item = dict(record)
        item["selection_reason"] = "dominant_fold_lane_core"
        selected.append(item)
    selected.extend(terminals)
    return {
        "eligible": True,
        "reason": "three_core_two_terminal_topology",
        "dominant_id": str(dominant.get("bend_id") or ""),
        "core_ids": [str(record.get("bend_id") or "") for record in core],
        "terminal_ids": [str(record.get("bend_id") or "") for record in terminals],
        "selected": selected,
    }
