"""Fail-closed routing between mesh-only and combined scan-only bend axes.

The combined selector is less precise than the mesh selector in aggregate, so
it must never replace mesh output merely because it returns fewer lines.  This
module permits one conservative exception: both selectors describe a single
parallel line family and the combined result is a strict geometric subset of
the mesh result.  In that case the subset is independent-source consensus,
not an inferred bend count.
"""

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
        half_span = 0.5 * float(record["span_mm"]) * axis
        value = [anchor - half_span, anchor + half_span]
    endpoints = np.asarray(value, dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1e-9:
        raise ValueError("line span must be nonzero")
    return endpoints


def _axis(endpoints: np.ndarray) -> np.ndarray:
    delta = endpoints[1] - endpoints[0]
    return delta / float(np.linalg.norm(delta))


def _axis_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _corresponds(
    left: np.ndarray,
    right: np.ndarray,
    *,
    maximum_axis_angle_deg: float,
    maximum_offset_mm: float,
    minimum_overlap_fraction: float,
) -> bool:
    left_axis = _axis(left)
    right_axis = _axis(right)
    if _axis_angle_deg(left_axis, right_axis) > maximum_axis_angle_deg:
        return False

    left_midpoint = np.mean(left, axis=0)
    right_midpoint = np.mean(right, axis=0)
    delta = right_midpoint - left_midpoint
    left_offset = float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis)))
    right_offset = float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis)))
    if max(left_offset, right_offset) > maximum_offset_mm:
        return False

    left_span = float(np.linalg.norm(left[1] - left[0]))
    right_span = float(np.linalg.norm(right[1] - right[0]))
    right_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in right)
    overlap = max(
        0.0,
        min(0.5 * left_span, right_interval[1])
        - max(-0.5 * left_span, right_interval[0]),
    )
    return overlap / max(min(left_span, right_span), 1e-9) >= minimum_overlap_fraction


def _is_single_parallel_family(
    endpoint_rows: Sequence[np.ndarray], maximum_axis_angle_deg: float
) -> bool:
    if len(endpoint_rows) < 2:
        return False
    axes = [_axis(endpoints) for endpoints in endpoint_rows]
    return all(
        _axis_angle_deg(axes[left], axes[right]) <= maximum_axis_angle_deg
        for left in range(len(axes))
        for right in range(left)
    )


def _injective_correspondence(options: Sequence[Sequence[int]]) -> list[int] | None:
    """Return a deterministic one-to-one combined-to-mesh assignment."""

    assignment = [-1] * len(options)
    used: set[int] = set()

    def assign(combined_index: int) -> bool:
        if combined_index == len(options):
            return True
        for mesh_index in sorted(options[combined_index]):
            if mesh_index in used:
                continue
            used.add(mesh_index)
            assignment[combined_index] = mesh_index
            if assign(combined_index + 1):
                return True
            used.remove(mesh_index)
            assignment[combined_index] = -1
        return False

    return assignment if assign(0) else None


def route_scan_only_source_consensus(
    mesh_selected: Sequence[Mapping[str, Any]],
    combined_selected: Sequence[Mapping[str, Any]],
    *,
    parallel_family_angle_deg: float = 7.5,
    correspondence_offset_mm: float = 4.0,
    minimum_overlap_fraction: float = 0.45,
) -> dict[str, Any]:
    """Choose a conservative scan-only source result without a count oracle.

    The combined route is accepted only when it is a non-empty strict subset
    of a mesh result containing at least two axes, adds no unsupported axis,
    and both results form one parallel family.  Invalid geometry fails closed
    to the mesh result.
    """

    mesh = [dict(record) for record in mesh_selected]
    combined = [dict(record) for record in combined_selected]
    decision: dict[str, Any] = {
        "route": "mesh_only",
        "reason": "combined_consensus_not_established",
        "mesh_count": len(mesh),
        "combined_count": len(combined),
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
        "selected": mesh,
    }
    if len(mesh) < 2 or not combined or len(combined) >= len(mesh):
        return decision

    try:
        mesh_endpoints = [_endpoints(record) for record in mesh]
        combined_endpoints = [_endpoints(record) for record in combined]
    except (KeyError, TypeError, ValueError):
        decision["reason"] = "invalid_line_geometry"
        return decision

    if not _is_single_parallel_family(mesh_endpoints, parallel_family_angle_deg):
        decision["reason"] = "mesh_result_has_multiple_axis_families"
        return decision
    if not _is_single_parallel_family(combined_endpoints, parallel_family_angle_deg):
        decision["reason"] = "combined_result_has_multiple_axis_families"
        return decision

    correspondence = [
        [
            mesh_index
            for mesh_index, mesh_line in enumerate(mesh_endpoints)
            if _corresponds(
                combined_line,
                mesh_line,
                maximum_axis_angle_deg=parallel_family_angle_deg,
                maximum_offset_mm=correspondence_offset_mm,
                minimum_overlap_fraction=minimum_overlap_fraction,
            )
        ]
        for combined_line in combined_endpoints
    ]
    assignment = _injective_correspondence(correspondence)
    if assignment is None:
        decision["reason"] = "combined_result_introduces_unsupported_axis"
        return decision

    decision.update(
        {
            "route": "combined_consensus_subset",
            "reason": "strict_parallel_family_consensus_subset",
            "corresponding_mesh_indices": assignment,
            "selected": combined,
        }
    )
    return decision
