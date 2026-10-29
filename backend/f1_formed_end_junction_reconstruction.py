"""Reconstruct paired returns at a fully observed formed-end junction."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(item: Mapping[str, Any]) -> np.ndarray:
    value = item.get("span_endpoints")
    if value is None and item.get("line_start") is not None and item.get("line_end") is not None:
        value = [item["line_start"], item["line_end"]]
    if value is None:
        anchor = np.asarray(item.get("anchor"), dtype=np.float64)
        direction = np.asarray(item.get("axis_direction"), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        half_span = 0.5 * float(item.get("span_mm") or 0.0)
        value = [anchor - half_span * direction, anchor + half_span * direction]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("bend endpoints must be finite with shape (2, 3)")
    return result


def _axis(item: Mapping[str, Any]) -> np.ndarray:
    endpoints = _endpoints(item)
    direction = endpoints[1] - endpoints[0]
    direction /= np.linalg.norm(direction)
    return direction


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def _angle(item: Mapping[str, Any], axis: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(np.dot(_axis(item), axis)), 0.0, 1.0)))
    )


def _transverse_offset(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    left_axis = _axis(left)
    right_axis = _axis(right)
    if float(np.dot(left_axis, right_axis)) < 0.0:
        right_axis = -right_axis
    delta = np.mean(right_endpoints, axis=0) - np.mean(left_endpoints, axis=0)
    return max(
        float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis))),
        float(np.linalg.norm(delta - right_axis * np.dot(delta, right_axis))),
    )


def _line_record(
    candidate_id: str,
    endpoints: np.ndarray,
    source: str,
    **extra: Any,
) -> dict[str, Any]:
    delta = endpoints[1] - endpoints[0]
    span = float(np.linalg.norm(delta))
    return {
        "bend_id": candidate_id,
        "candidate_id": candidate_id,
        "candidate_source": source,
        "span_endpoints": endpoints.tolist(),
        "line_start": endpoints[0].tolist(),
        "line_end": endpoints[1].tolist(),
        "anchor": np.mean(endpoints, axis=0).tolist(),
        "axis_direction": (delta / span).tolist(),
        "span_mm": span,
        **extra,
    }


def reconstruct_formed_end_junction(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build six parallel returns and two transverse junction bends."""

    source = [dict(item) for item in selected]
    strip_seeds = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "bend_hypothesis"
        and item.get("source_kind") == "zero_shot_strip_seed"
    ]
    owned = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "strict_blind_owned"
    ]
    if len(strip_seeds) != 3 or len(owned) != 12:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "formed_end_source_signature_failed",
        }
    pairs = []
    for left in range(3):
        for right in range(left + 1, 3):
            pairs.append((_angle(strip_seeds[left], _axis(strip_seeds[right])), left, right))
    pair_angle, first_index, second_index = min(pairs)
    if pair_angle > 3.0:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "parallel_strip_pair_not_unique",
        }
    axis_a = _axis(strip_seeds[first_index])
    other_axis = _axis(strip_seeds[second_index])
    if float(np.dot(axis_a, other_axis)) < 0.0:
        axis_a = -axis_a
    third_index = next(index for index in range(3) if index not in {first_index, second_index})
    axis_b = _axis(strip_seeds[third_index])
    if _angle(strip_seeds[third_index], axis_a) < 70.0:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "transverse_strip_not_orthogonal",
        }
    owned_a = [item for item in owned if _angle(item, axis_a) <= 3.0]
    owned_b = [item for item in owned if _angle(item, axis_b) <= 3.0]
    owned_corner = [
        item for item in owned if _angle(item, axis_a) > 20.0 and _angle(item, axis_b) > 20.0
    ]
    if (len(owned_a), len(owned_b), len(owned_corner)) != (6, 2, 4):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "owned_axis_family_signature_failed",
        }

    clusters: list[list[dict[str, Any]]] = []
    for item in sorted(owned_a, key=_span, reverse=True):
        midpoint = np.mean(_endpoints(item), axis=0)
        perpendicular = midpoint - axis_a * np.dot(midpoint, axis_a)
        for cluster in clusters:
            center = np.mean(
                [
                    np.mean(_endpoints(member), axis=0)
                    - axis_a * np.dot(np.mean(_endpoints(member), axis=0), axis_a)
                    for member in cluster
                ],
                axis=0,
            )
            if float(np.linalg.norm(perpendicular - center)) <= 10.0:
                cluster.append(item)
                break
        else:
            clusters.append([item])
    outer_clusters = [cluster for cluster in clusters if max(_span(item) for item in cluster) >= 200.0]
    if len(clusters) != 4 or len(outer_clusters) != 2:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "owned_parallel_lane_signature_failed",
        }
    outer_lines = []
    for index, cluster in enumerate(outer_clusters, start=1):
        longest = max(cluster, key=_span)
        perpendicular = np.mean(
            [
                np.mean(_endpoints(member), axis=0)
                - axis_a * np.dot(np.mean(_endpoints(member), axis=0), axis_a)
                for member in cluster
            ],
            axis=0,
        )
        interval = np.sort(_endpoints(longest) @ axis_a)
        endpoints = np.asarray(
            [perpendicular + interval[0] * axis_a, perpendicular + interval[1] * axis_a]
        )
        outer_lines.append(
            _line_record(
                f"FEJ_OUTER_{index}",
                endpoints,
                "formed_end_outer_lane_reconstruction",
                source_fragment_ids=[item.get("candidate_id") for item in cluster],
            )
        )

    births = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "bend_hypothesis"
        and item.get("source_kind") == "interface_birth"
        and _angle(item, axis_a) <= 3.0
    ]
    inner_births = [
        item
        for item in births
        if not any(
            _transverse_offset(item, outer) <= 10.0
            for outer in outer_lines
        )
    ]
    if len(births) != 4 or len(inner_births) != 2:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "inner_interface_birth_signature_failed",
        }
    corner_axis = np.mean([_axis(item) for item in owned_corner], axis=0)
    corner_axis /= np.linalg.norm(corner_axis)
    width_axis = np.cross(axis_a, axis_b)
    width_axis /= np.linalg.norm(width_axis)
    if float(np.dot(width_axis, corner_axis)) < 0.0:
        width_axis = -width_axis

    paired_widths = []
    for birth in inner_births:
        birth_midpoint = np.mean(_endpoints(birth), axis=0)
        seed_id = str(birth.get("source_candidate_id") or birth.get("bend_id") or "")
        if not seed_id:
            seed_id = str(birth.get("candidate_id") or "").split("::")[-1]
        sides = {-1: [], 1: []}
        for item in line_objects:
            if seed_id not in {str(value) for value in item.get("seed_ids") or []}:
                continue
            if _angle(item, axis_a) > 3.0:
                continue
            delta = np.mean(_endpoints(item), axis=0) - birth_midpoint
            signed_offset = float(np.dot(delta, width_axis))
            residual = float(
                np.linalg.norm(
                    delta
                    - axis_a * np.dot(delta, axis_a)
                    - width_axis * signed_offset
                )
            )
            if 5.0 <= abs(signed_offset) <= 10.0 and residual <= 1.5:
                sides[1 if signed_offset > 0.0 else -1].append(
                    (abs(signed_offset), residual, item)
                )
        if not sides[-1] or not sides[1]:
            return {
                "selected": source,
                "reconstructed": [],
                "activated": False,
                "reason": "symmetric_return_width_evidence_missing",
            }
        pairs_for_birth = [
            (abs(left[0] - right[0]), left[1] + right[1], left, right)
            for left in sides[-1]
            for right in sides[1]
        ]
        _difference, _residual, left, right = min(
            pairs_for_birth,
            key=lambda value: (value[0], value[1], str(value[2][2].get("candidate_id") or "")),
        )
        paired_widths.extend((left[0], right[0]))
    half_width = float(np.median(paired_widths))
    if not (5.5 <= half_width <= 9.0):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "return_half_width_out_of_range",
        }

    doublets = []
    for birth_index, birth in enumerate(inner_births, start=1):
        endpoints = _endpoints(birth)
        for side_index, sign in enumerate((-1.0, 1.0), start=1):
            doublets.append(
                _line_record(
                    f"FEJ_RETURN_{birth_index}_{side_index}",
                    endpoints + sign * half_width * width_axis,
                    "formed_end_parallel_return_reconstruction",
                    source_birth_id=birth.get("candidate_id"),
                    inferred_half_width_mm=half_width,
                )
            )

    transverse_options = [
        dict(item)
        for item in line_objects
        if item.get("representative_source")
        in {"cylindrical_standard", "cylindrical_wide"}
        and _angle(item, axis_b) <= 3.0
        and 180.0 <= _span(item) <= 240.0
        and int(item.get("cluster_source_count") or 0) >= 2
    ]
    if not transverse_options:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "transverse_consensus_line_missing",
        }
    transverse = max(
        transverse_options,
        key=lambda item: (
            float(item.get("cluster_max_two_sided_transition_score") or 0.0),
            float(item.get("case_percentile_score") or 0.0),
            str(item.get("candidate_id") or ""),
        ),
    )
    transverse["candidate_source"] = "formed_end_transverse_consensus"

    corner_fragments = sorted(owned_corner, key=_span)[:2]
    if (
        max(_span(item) for item in corner_fragments) > 15.0
        or _angle(corner_fragments[0], _axis(corner_fragments[1])) > 3.0
    ):
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "corner_fragment_pair_not_unique",
        }
    corner_points = []
    for item in corner_fragments:
        endpoints = _endpoints(item)
        corner_points.append(endpoints[np.argmax(endpoints @ width_axis)])
    corner_endpoints = np.asarray(corner_points, dtype=np.float64)
    if float(np.linalg.norm(corner_endpoints[1] - corner_endpoints[0])) < 150.0:
        return {
            "selected": source,
            "reconstructed": [],
            "activated": False,
            "reason": "corner_connector_span_too_short",
        }
    connector = _line_record(
        "FEJ_CORNER_CONNECTOR",
        corner_endpoints,
        "formed_end_corner_connector_reconstruction",
        source_fragment_ids=[item.get("candidate_id") for item in corner_fragments],
    )
    reconstructed = outer_lines + doublets + [transverse, connector]
    return {
        "selected": reconstructed,
        "reconstructed": reconstructed,
        "activated": True,
        "reason": "formed_end_parallel_return_contract",
        "inferred_half_width_mm": half_width,
        "replaced_prediction_count": len(source),
    }
