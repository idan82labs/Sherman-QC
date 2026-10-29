"""Count-free recovery of strongly supported short scan-mesh bend axes."""

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
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1e-9:
        raise ValueError("line span must be nonzero")
    return endpoints


def _pair_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float, float, float]:
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    left_delta = left_endpoints[1] - left_endpoints[0]
    right_delta = right_endpoints[1] - right_endpoints[0]
    left_span = float(np.linalg.norm(left_delta))
    right_span = float(np.linalg.norm(right_delta))
    left_axis = left_delta / left_span
    right_axis = right_delta / right_span
    angle = math.degrees(
        math.acos(float(np.clip(abs(float(np.dot(left_axis, right_axis))), 0.0, 1.0)))
    )
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
    overlap_fraction = overlap / max(min(left_span, right_span), 1e-9)
    return angle, max(left_offset, right_offset), overlap_fraction


def _duplicate(
    left: Mapping[str, Any], right: Mapping[str, Any], offset_mm: float
) -> bool:
    angle, offset, overlap = _pair_metrics(left, right)
    return angle <= 7.5 and offset <= offset_mm and overlap >= 0.45


def _is_scan_mesh_only(record: Mapping[str, Any]) -> bool:
    sources = record.get("cluster_sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        return list(sources) == ["scan_mesh_adjacency"]
    return str(record.get("representative_source") or "") == "scan_mesh_adjacency"


def complete_short_scan_mesh_bends(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
    *,
    maximum_span_mm: float,
    minimum_normal_turn_deg: float,
    minimum_finite_support_fraction: float,
    duplicate_offset_mm: float = 4.0,
) -> dict[str, Any]:
    """Append nonduplicate short mesh axes under externally tuned thresholds.

    The rule uses only scan geometry and line-object evidence. It does not infer
    or consume an expected bend count. Passing a nonpositive maximum span
    disables the completion explicitly.
    """

    output = [dict(record) for record in selected]
    if maximum_span_mm <= 0.0:
        return {"enabled": False, "added_ids": [], "selected": output}

    eligible: list[dict[str, Any]] = []
    for source in line_objects:
        record = dict(source)
        try:
            endpoints = _endpoints(record)
        except (KeyError, TypeError, ValueError):
            continue
        span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        normal_turn = float(
            record.get("cluster_max_two_sided_normal_angle_deg")
            or record.get("two_sided_normal_angle_deg")
            or 0.0
        )
        finite_support = float(
            record.get("cluster_max_finite_support_fraction")
            or record.get("finite_support_fraction")
            or 0.0
        )
        if (
            not _is_scan_mesh_only(record)
            or span > maximum_span_mm
            or normal_turn < minimum_normal_turn_deg
            or finite_support < minimum_finite_support_fraction
        ):
            continue
        eligible.append(record)

    eligible.sort(
        key=lambda record: (
            -float(
                record.get("cluster_max_two_sided_normal_angle_deg")
                or record.get("two_sided_normal_angle_deg")
                or 0.0
            ),
            -float(
                record.get("cluster_max_finite_support_fraction")
                or record.get("finite_support_fraction")
                or 0.0
            ),
            str(record.get("candidate_id") or record.get("bend_id") or ""),
        )
    )
    added_ids = []
    for record in eligible:
        if any(_duplicate(record, retained, duplicate_offset_mm) for retained in output):
            continue
        record["selection_reason"] = "short_scan_mesh_bend_completion"
        output.append(record)
        added_ids.append(str(record.get("bend_id") or record.get("candidate_id") or ""))
    return {"enabled": True, "added_ids": added_ids, "selected": output}

