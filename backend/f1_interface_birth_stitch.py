"""Stitch adjacent interface-birth fragments into one finite bend axis."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _duplicates, _endpoints


def _pair_key(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in item.get("canonical_flange_pair") or ()))


def _compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = _endpoints(left)
    second = _endpoints(right)
    first_delta = first[1] - first[0]
    second_delta = second[1] - second[0]
    first_span = float(np.linalg.norm(first_delta))
    second_span = float(np.linalg.norm(second_delta))
    if min(first_span, second_span) <= 1e-9:
        return False
    axis = first_delta / first_span
    other_axis = second_delta / second_span
    dot = float(np.clip(np.dot(axis, other_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(abs(dot)))
    if angle > 4.0:
        return False
    midpoint_delta = np.mean(second, axis=0) - np.mean(first, axis=0)
    offset = max(
        float(np.linalg.norm(midpoint_delta - axis * np.dot(midpoint_delta, axis))),
        float(np.linalg.norm(midpoint_delta - other_axis * np.dot(midpoint_delta, other_axis))),
    )
    if offset > 2.0:
        return False
    first_interval = (-0.5 * first_span, 0.5 * first_span)
    second_interval = sorted(float(np.dot(point - np.mean(first, axis=0), axis)) for point in second)
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    if overlap / min(first_span, second_span) >= 0.20:
        return False
    gap = max(0.0, max(first_interval[0], second_interval[0]) - min(first_interval[1], second_interval[1]))
    widths = [float(item.get("cross_width_mm") or 0.0) for item in (left, right)]
    maximum_gap = max(4.0, 0.75 * max(widths))
    return gap <= maximum_gap


def _stitch(members: Sequence[Mapping[str, Any]], identifier: str) -> dict[str, Any]:
    reference = max(members, key=lambda item: float(item.get("span_mm") or 0.0))
    reference_line = _endpoints(reference)
    axis = reference_line[1] - reference_line[0]
    axis /= np.linalg.norm(axis)
    origin = np.mean(reference_line, axis=0)
    endpoints = np.concatenate([_endpoints(item) for item in members], axis=0)
    coordinates = endpoints @ axis
    start_coordinate = float(np.min(coordinates))
    end_coordinate = float(np.max(coordinates))
    perpendicular_center = np.mean(endpoints - np.outer(coordinates, axis), axis=0)
    start = perpendicular_center + start_coordinate * axis
    end = perpendicular_center + end_coordinate * axis
    result = dict(reference)
    result.update(
        {
            "bend_id": identifier,
            "anchor": ((start + end) * 0.5).tolist(),
            "axis_direction": axis.tolist(),
            "span_mm": float(np.linalg.norm(end - start)),
            "span_endpoints": [start.tolist(), end.tolist()],
            "candidate_source": "interface_birth_stitch",
            "selection_reason": "adjacent_same_interface_birth_fragments",
            "stitch_member_ids": sorted(str(item.get("bend_id") or "") for item in members),
        }
    )
    return result


def promote_stitched_interface_births(
    selected: Sequence[Mapping[str, Any]],
    bend_hypotheses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Promote collinear, near-adjacent births from the same flange interface."""

    transition_spans = sorted(
        (
            float(item.get("span_mm") or 0.0)
            for item in bend_hypotheses
            if str(item.get("source_kind") or "") == "detector_bend_seed"
            and str(item.get("bend_class") or "") == "transition_only"
        ),
        reverse=True,
    )
    dominant_transition_topology = (
        len(transition_spans) >= 2
        and transition_spans[0] >= 200.0
        and transition_spans[0] >= 2.0 * transition_spans[1]
    )
    if not dominant_transition_topology:
        return {
            "selected": [dict(item) for item in selected],
            "promoted": [],
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
        }
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for source in bend_hypotheses:
        if str(source.get("source_kind") or "") != "interface_birth":
            continue
        key = _pair_key(source)
        if len(key) != 2:
            continue
        groups.setdefault(key, []).append(dict(source))
    output = [dict(item) for item in selected]
    promoted = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: str(item.get("bend_id") or ""))
        if len(members) != 2 or not _compatible(members[0], members[1]):
            continue
        candidate = _stitch(members, f"IBS{len(promoted) + 1}")
        try:
            duplicate_indices = [
                index for index, retained in enumerate(output) if _duplicates(candidate, retained)
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if duplicate_indices:
            duplicate_spans = [
                float(np.linalg.norm(_endpoints(output[index])[1] - _endpoints(output[index])[0]))
                for index in duplicate_indices
            ]
            if float(candidate["span_mm"]) < 1.15 * max(duplicate_spans):
                continue
            candidate["replaced_bend_ids"] = sorted(
                str(output[index].get("bend_id") or "") for index in duplicate_indices
            )
            output = [item for index, item in enumerate(output) if index not in duplicate_indices]
        output.append(candidate)
        promoted.append(candidate)
    return {
        "selected": output,
        "promoted": promoted,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
