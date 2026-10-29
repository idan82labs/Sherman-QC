"""Prune near-coincident duplicate lines from the scan-mesh bend lane.

Surface segmentation can emit two representations of the same physical bend
whose fitted axes fall on opposite sides of a rounded transition.  The normal
F1 selector already collapses duplicates within 4 mm.  This postprocessor
handles the narrow numerical boundary immediately above that threshold while
failing closed for different sources, unshared surfaces, short overlap, or
materially different spans.

The rule is identity-free and does not inspect an expected bend count, manual
truth, CAD, or part-specific metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NearCoincidentMeshDuplicateThresholds:
    maximum_axis_angle_deg: float = 3.0
    maximum_transverse_offset_mm: float = 4.01
    minimum_shorter_span_overlap: float = 0.95
    maximum_span_ratio: float = 1.10


def _candidate_id(candidate: Mapping[str, Any]) -> Any:
    return candidate.get("candidate_id") or candidate.get("bend_id")


def _finite_float(candidate: Mapping[str, Any], key: str) -> float:
    try:
        value = float(candidate.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) else 0.0


def _endpoints(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    start = candidate.get("line_start")
    end = candidate.get("line_end")
    span = candidate.get("span_endpoints")
    if (start is None or end is None) and isinstance(span, Sequence) and len(span) == 2:
        start = span[0] if start is None else start
        end = span[1] if end is None else end
    if start is None or end is None:
        return None
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if start_array.shape != (3,) or end_array.shape != (3,):
        return None
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(end_array)):
        return None
    return start_array, end_array


def _surface_ids(candidate: Mapping[str, Any]) -> frozenset[Any]:
    return frozenset(
        value
        for value in (candidate.get("surface1_id"), candidate.get("surface2_id"))
        if value is not None
    )


def _pair_geometry(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float] | None:
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    if left_endpoints is None or right_endpoints is None:
        return None
    left_start, left_end = left_endpoints
    right_start, right_end = right_endpoints
    left_vector = left_end - left_start
    right_vector = right_end - right_start
    left_span = float(np.linalg.norm(left_vector))
    right_span = float(np.linalg.norm(right_vector))
    if left_span <= 1.0e-9 or right_span <= 1.0e-9:
        return None
    left_axis = left_vector / left_span
    right_axis = right_vector / right_span
    axis_angle = degrees(
        acos(float(np.clip(abs(np.dot(left_axis, right_axis)), 0.0, 1.0)))
    )

    left_midpoint = (left_start + left_end) / 2.0
    right_midpoint = (right_start + right_end) / 2.0
    midpoint_delta = right_midpoint - left_midpoint
    left_axis_offset = float(
        np.linalg.norm(
            midpoint_delta - left_axis * float(np.dot(midpoint_delta, left_axis))
        )
    )
    right_axis_offset = float(
        np.linalg.norm(
            midpoint_delta - right_axis * float(np.dot(midpoint_delta, right_axis))
        )
    )
    transverse_offset = max(left_axis_offset, right_axis_offset)
    projected = sorted(
        (
            float(np.dot(right_start - left_midpoint, left_axis)),
            float(np.dot(right_end - left_midpoint, left_axis)),
        )
    )
    left_interval = (-left_span / 2.0, left_span / 2.0)
    overlap = max(
        0.0,
        min(left_interval[1], projected[1])
        - max(left_interval[0], projected[0]),
    )
    return {
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": transverse_offset,
        "shorter_span_overlap": overlap / min(left_span, right_span),
        "span_ratio": max(left_span, right_span) / min(left_span, right_span),
    }


def _is_duplicate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    thresholds: NearCoincidentMeshDuplicateThresholds,
) -> tuple[bool, dict[str, float] | None]:
    if (
        left.get("candidate_source") != "scan_mesh_adjacency"
        or right.get("candidate_source") != "scan_mesh_adjacency"
    ):
        return False, None
    left_surfaces = _surface_ids(left)
    right_surfaces = _surface_ids(right)
    if not left_surfaces or not right_surfaces or not (left_surfaces & right_surfaces):
        return False, None
    geometry = _pair_geometry(left, right)
    if geometry is None:
        return False, None
    duplicate = (
        geometry["axis_angle_deg"] <= thresholds.maximum_axis_angle_deg
        and geometry["transverse_offset_mm"]
        <= thresholds.maximum_transverse_offset_mm
        and geometry["shorter_span_overlap"]
        >= thresholds.minimum_shorter_span_overlap
        and geometry["span_ratio"] <= thresholds.maximum_span_ratio
    )
    return duplicate, geometry


def _strength(candidate: Mapping[str, Any], index: int) -> tuple[float, ...]:
    return (
        _finite_float(candidate, "two_sided_transition_score"),
        _finite_float(candidate, "selection_probability"),
        _finite_float(candidate, "predicted_alignment_quality"),
        _finite_float(candidate, "confidence"),
        -float(index),
    )


def prune_near_coincident_mesh_duplicates(
    selected: Sequence[Mapping[str, Any]],
    thresholds: NearCoincidentMeshDuplicateThresholds = (
        NearCoincidentMeshDuplicateThresholds()
    ),
) -> dict[str, Any]:
    """Keep the strongest line in each strict near-coincident conflict."""

    retained_indices: list[int] = []
    pruned_indices: set[int] = set()
    conflicts: list[dict[str, Any]] = []
    ranked_indices = sorted(
        range(len(selected)),
        key=lambda index: _strength(selected[index], index),
        reverse=True,
    )
    for candidate_index in ranked_indices:
        candidate = selected[candidate_index]
        duplicate_witness: tuple[int, dict[str, float]] | None = None
        for witness_index in retained_indices:
            duplicate, geometry = _is_duplicate(
                candidate, selected[witness_index], thresholds
            )
            if duplicate and geometry is not None:
                duplicate_witness = (witness_index, geometry)
                break
        if duplicate_witness is None:
            retained_indices.append(candidate_index)
            continue
        witness_index, geometry = duplicate_witness
        pruned_indices.add(candidate_index)
        conflicts.append(
            {
                "pruned_candidate_id": _candidate_id(candidate),
                "retained_candidate_id": _candidate_id(selected[witness_index]),
                **geometry,
            }
        )

    return {
        "selected": [
            candidate
            for index, candidate in enumerate(selected)
            if index not in pruned_indices
        ],
        "pruned": [selected[index] for index in sorted(pruned_indices)],
        "pruned_candidate_ids": [
            _candidate_id(selected[index]) for index in sorted(pruned_indices)
        ],
        "conflicts": conflicts,
        "activated": bool(pruned_indices),
        "thresholds": {
            key: value for key, value in vars(thresholds).items()
        },
    }
