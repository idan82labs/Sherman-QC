"""Prefer a complete routed bend set over a truncated mesh representation.

The mesh lane can recover the right physical bend objects while fitting one
axis to a short, tilted surface fragment.  When the independent routed lane
has the same number of lines, every line has a one-to-one geometric witness,
and the routed lines consistently span more of the supported interface, the
routed representation is the safer line geometry.

This rule is identity-free and never sees CAD, an expected count, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CompleteRoutedConsensusThresholds:
    maximum_axis_angle_deg: float = 12.0
    maximum_transverse_offset_mm: float = 4.0
    minimum_shorter_span_overlap: float = 0.60
    minimum_each_routed_to_mesh_span_ratio: float = 1.05
    minimum_median_routed_to_mesh_span_ratio: float = 1.15
    minimum_material_axis_disagreement_deg: float = 5.0


def _candidate_id(candidate: Mapping[str, Any]) -> Any:
    return candidate.get("candidate_id") or candidate.get("bend_id")


def _endpoints(candidate: Mapping[str, Any]) -> np.ndarray | None:
    span = candidate.get("span_endpoints")
    if isinstance(span, Sequence) and len(span) == 2:
        endpoints = np.asarray(span, dtype=np.float64)
    elif candidate.get("line_start") is not None and candidate.get("line_end") is not None:
        endpoints = np.asarray(
            [candidate["line_start"], candidate["line_end"]], dtype=np.float64
        )
    else:
        return None
    if endpoints.shape != (2, 3) or not np.all(np.isfinite(endpoints)):
        return None
    return endpoints


def _pair_geometry(left: np.ndarray, right: np.ndarray) -> dict[str, float] | None:
    left_midpoint = np.mean(left, axis=0)
    right_midpoint = np.mean(right, axis=0)
    left_delta = left[1] - left[0]
    right_delta = right[1] - right[0]
    left_span = float(np.linalg.norm(left_delta))
    right_span = float(np.linalg.norm(right_delta))
    if left_span <= 1.0e-9 or right_span <= 1.0e-9:
        return None
    left_axis = left_delta / left_span
    right_axis = right_delta / right_span
    dot = float(np.clip(np.dot(left_axis, right_axis), -1.0, 1.0))
    axis_angle = degrees(acos(abs(dot)))
    if dot < 0.0:
        right_axis *= -1.0
    midpoint_delta = right_midpoint - left_midpoint
    left_offset = float(
        np.linalg.norm(
            midpoint_delta - left_axis * float(np.dot(midpoint_delta, left_axis))
        )
    )
    right_offset = float(
        np.linalg.norm(
            midpoint_delta
            - right_axis * float(np.dot(midpoint_delta, right_axis))
        )
    )
    left_interval = sorted(
        float(np.dot(point - left_midpoint, left_axis)) for point in left
    )
    right_interval = sorted(
        float(np.dot(point - left_midpoint, left_axis)) for point in right
    )
    overlap = max(
        0.0,
        min(left_interval[1], right_interval[1])
        - max(left_interval[0], right_interval[0]),
    )
    return {
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": max(left_offset, right_offset),
        "shorter_span_overlap": overlap / min(left_span, right_span),
        "routed_span_mm": left_span,
        "mesh_span_mm": right_span,
        "routed_to_mesh_span_ratio": left_span / right_span,
    }


def prefer_complete_routed_consensus(
    mesh_selected: Sequence[Mapping[str, Any]],
    routed_selected: Sequence[Mapping[str, Any]],
    thresholds: CompleteRoutedConsensusThresholds = (
        CompleteRoutedConsensusThresholds()
    ),
) -> dict[str, Any]:
    """Return routed lines only for a strict, complete cross-lane consensus."""

    mesh = [dict(item) for item in mesh_selected]
    routed = [dict(item) for item in routed_selected]
    inactive = {
        "selected": mesh,
        "activated": False,
        "pairings": [],
        "thresholds": {key: value for key, value in vars(thresholds).items()},
    }
    if not mesh or len(mesh) != len(routed):
        return {**inactive, "reason": "lane_counts_differ_or_empty"}
    if any(item.get("source_kind") != "zero_shot_support_strip" for item in routed):
        return {**inactive, "reason": "routed_lane_not_uniform_support_strips"}
    mesh_endpoints = [_endpoints(item) for item in mesh]
    routed_endpoints = [_endpoints(item) for item in routed]
    if any(value is None for value in mesh_endpoints + routed_endpoints):
        return {**inactive, "reason": "invalid_line_geometry"}

    cost = np.full((len(routed), len(mesh)), 1.0e9, dtype=np.float64)
    geometries: dict[tuple[int, int], dict[str, float]] = {}
    for routed_index, routed_line in enumerate(routed_endpoints):
        for mesh_index, mesh_line in enumerate(mesh_endpoints):
            geometry = _pair_geometry(routed_line, mesh_line)
            if geometry is None:
                continue
            geometries[(routed_index, mesh_index)] = geometry
            if (
                geometry["axis_angle_deg"] <= thresholds.maximum_axis_angle_deg
                and geometry["transverse_offset_mm"]
                <= thresholds.maximum_transverse_offset_mm
                and geometry["shorter_span_overlap"]
                >= thresholds.minimum_shorter_span_overlap
            ):
                cost[routed_index, mesh_index] = (
                    geometry["axis_angle_deg"] / thresholds.maximum_axis_angle_deg
                    + geometry["transverse_offset_mm"]
                    / thresholds.maximum_transverse_offset_mm
                    + 1.0 - min(geometry["shorter_span_overlap"], 1.0)
                )
    routed_indices, mesh_indices = linear_sum_assignment(cost)
    if any(cost[left, right] >= 1.0e9 for left, right in zip(routed_indices, mesh_indices)):
        return {**inactive, "reason": "incomplete_cross_lane_assignment"}

    pairings = []
    span_ratios = []
    angles = []
    for routed_index, mesh_index in zip(routed_indices, mesh_indices):
        geometry = geometries[(int(routed_index), int(mesh_index))]
        span_ratios.append(geometry["routed_to_mesh_span_ratio"])
        angles.append(geometry["axis_angle_deg"])
        pairings.append(
            {
                "routed_candidate_id": _candidate_id(routed[int(routed_index)]),
                "mesh_candidate_id": _candidate_id(mesh[int(mesh_index)]),
                **geometry,
            }
        )
    if min(span_ratios) < thresholds.minimum_each_routed_to_mesh_span_ratio:
        return {**inactive, "reason": "routed_span_not_consistently_longer", "pairings": pairings}
    if float(np.median(span_ratios)) < thresholds.minimum_median_routed_to_mesh_span_ratio:
        return {**inactive, "reason": "median_routed_span_advantage_too_small", "pairings": pairings}
    if max(angles) < thresholds.minimum_material_axis_disagreement_deg:
        return {**inactive, "reason": "no_material_axis_disagreement", "pairings": pairings}

    return {
        "selected": routed,
        "activated": True,
        "reason": "complete_cross_lane_consensus_with_routed_span_advantage",
        "pairings": pairings,
        "thresholds": {key: value for key, value in vars(thresholds).items()},
    }
