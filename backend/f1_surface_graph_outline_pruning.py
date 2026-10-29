"""Prune mesh outline chords exposed by the selected surface graph.

Unary scan evidence can be strong on a free perimeter because the two nearby
sheet faces still produce a coherent normal transition.  This module only
acts when the *selected* mesh-interface graph supplies an additional,
deterministic contradiction.  It does not inspect a part identifier, expected
bend count, learned probability, or manual truth.

The three deliberately narrow contradictions are:

* an interface that over-closes two or more surface triangles whose fitted
  junctions all lie materially beyond its finite span;
* a short edge between similarly sized broad faces that closes a triangle and
  runs almost along one of its two long sides; and
* an isolated surface pair that is parallel to, and scan-evidence dominated
  by, an equal-span interface in the connected sheet graph.

Missing or malformed geometry fails closed and leaves the candidate selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SurfaceGraphOutlineThresholds:
    minimum_multiply_closed_junction_outside_ratio: float = 0.25
    minimum_broad_face_area_ratio: float = 0.90
    minimum_broad_triangle_bend_angle_deg: float = 110.0
    maximum_broad_triangle_span_ratio: float = 0.55
    maximum_broad_triangle_side_angle_deg: float = 20.0
    maximum_isolated_axis_angle_deg: float = 1.0
    minimum_isolated_offset_mm: float = 5.0
    maximum_isolated_offset_mm: float = 50.0
    minimum_isolated_span_overlap: float = 0.95
    minimum_isolated_transition_advantage: float = 0.10
    minimum_isolated_span_ratio: float = 0.80
    maximum_isolated_span_ratio: float = 1.25


def _value(record: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _surface_pair(record: Mapping[str, Any]) -> frozenset[Hashable] | None:
    first = record.get("surface1_id")
    second = record.get("surface2_id")
    if first is None or second is None or first == second:
        return None
    try:
        return frozenset((first, second))
    except TypeError:
        return None


def _is_mesh_interface(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("candidate_id") or "").startswith("scan_mesh_adjacency::")
        or str(record.get("candidate_source") or "") == "scan_mesh_adjacency"
        or str(record.get("representative_source") or "")
        == "scan_mesh_adjacency"
    )


def _endpoints(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    value = record.get("span_endpoints")
    if value is None and record.get("line_start") is not None:
        value = (record.get("line_start"), record.get("line_end"))
    try:
        endpoints = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        return None
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1.0e-9:
        return None
    return endpoints[0], endpoints[1]


def _axis_angle_deg(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float | None:
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    if left_endpoints is None or right_endpoints is None:
        return None
    left_axis = left_endpoints[1] - left_endpoints[0]
    right_axis = right_endpoints[1] - right_endpoints[0]
    left_axis /= np.linalg.norm(left_axis)
    right_axis /= np.linalg.norm(right_axis)
    cosine = float(np.clip(abs(np.dot(left_axis, right_axis)), 0.0, 1.0))
    return degrees(acos(cosine))


def _parallel_geometry(
    candidate: Mapping[str, Any], witness: Mapping[str, Any]
) -> dict[str, float] | None:
    candidate_endpoints = _endpoints(candidate)
    witness_endpoints = _endpoints(witness)
    if candidate_endpoints is None or witness_endpoints is None:
        return None
    candidate_start, candidate_end = candidate_endpoints
    witness_start, witness_end = witness_endpoints
    candidate_vector = candidate_end - candidate_start
    witness_vector = witness_end - witness_start
    candidate_span = float(np.linalg.norm(candidate_vector))
    witness_span = float(np.linalg.norm(witness_vector))
    candidate_axis = candidate_vector / candidate_span
    witness_axis = witness_vector / witness_span
    cosine = float(np.clip(abs(np.dot(candidate_axis, witness_axis)), 0.0, 1.0))
    displacement = np.mean(witness_endpoints, axis=0) - np.mean(
        candidate_endpoints, axis=0
    )
    transverse_offset = float(
        np.linalg.norm(
            displacement - float(np.dot(displacement, candidate_axis)) * candidate_axis
        )
    )
    candidate_midpoint = np.mean(candidate_endpoints, axis=0)
    projected = sorted(
        float(np.dot(point - candidate_midpoint, candidate_axis))
        for point in witness_endpoints
    )
    overlap = max(
        0.0,
        min(0.5 * candidate_span, projected[1])
        - max(-0.5 * candidate_span, projected[0]),
    )
    return {
        "axis_angle_deg": degrees(acos(cosine)),
        "transverse_offset_mm": transverse_offset,
        "shorter_span_overlap": overlap / min(candidate_span, witness_span),
        "span_ratio": candidate_span / witness_span,
    }


def _triangle_junction_geometry(
    candidate: Mapping[str, Any],
    first_witness: Mapping[str, Any],
    second_witness: Mapping[str, Any],
) -> dict[str, float] | None:
    records = (candidate, first_witness, second_witness)
    endpoints = [_endpoints(record) for record in records]
    if any(value is None for value in endpoints):
        return None
    normal_matrix = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    line_geometry: list[tuple[np.ndarray, np.ndarray, float]] = []
    for value in endpoints:
        assert value is not None
        start, end = value
        midpoint = 0.5 * (start + end)
        vector = end - start
        span = float(np.linalg.norm(vector))
        axis = vector / span
        projector = np.eye(3, dtype=np.float64) - np.outer(axis, axis)
        normal_matrix += projector
        right_hand_side += projector @ midpoint
        line_geometry.append((midpoint, axis, span))
    try:
        junction = np.linalg.lstsq(
            normal_matrix, right_hand_side, rcond=None
        )[0]
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(junction).all():
        return None
    distances = [
        float(
            np.linalg.norm(
                (np.eye(3, dtype=np.float64) - np.outer(axis, axis))
                @ (junction - midpoint)
            )
        )
        for midpoint, axis, _span in line_geometry
    ]
    candidate_midpoint, candidate_axis, candidate_span = line_geometry[0]
    candidate_outside = max(
        0.0,
        abs(float(np.dot(junction - candidate_midpoint, candidate_axis)))
        - 0.5 * candidate_span,
    )
    return {
        "junction_residual_rms_mm": float(
            np.sqrt(np.mean(np.square(distances)))
        ),
        "candidate_junction_outside_ratio": candidate_outside / candidate_span,
    }


def _candidate_id(record: Mapping[str, Any]) -> Any:
    return record.get("candidate_id") or record.get("bend_id")


def prune_surface_graph_outlines(
    selected: Sequence[Mapping[str, Any]],
    thresholds: SurfaceGraphOutlineThresholds = SurfaceGraphOutlineThresholds(),
) -> dict[str, Any]:
    """Remove only surface-graph-proven outline chords, preserving order."""

    graph: dict[Hashable, set[Hashable]] = {}
    edge_records: dict[frozenset[Hashable], Mapping[str, Any]] = {}
    ambiguous_pairs: set[frozenset[Hashable]] = set()
    for record in selected:
        pair = _surface_pair(record)
        if pair is None:
            continue
        first, second = tuple(pair)
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
        # Ambiguous duplicate pairs fail closed in triangle lanes.
        if pair in ambiguous_pairs:
            continue
        if pair in edge_records:
            edge_records.pop(pair, None)
            ambiguous_pairs.add(pair)
        else:
            edge_records[pair] = record

    pruned_indices: set[int] = set()
    conflicts: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected):
        pair = _surface_pair(candidate)
        if pair is None or not _is_mesh_interface(candidate):
            continue
        first, second = tuple(pair)
        common = graph.get(first, set()) & graph.get(second, set())
        span = _value(candidate, "span_mm")

        if len(common) >= 2:
            junctions: list[dict[str, float | Hashable]] = []
            for neighbor in sorted(common, key=str):
                first_witness = edge_records.get(frozenset((first, neighbor)))
                second_witness = edge_records.get(frozenset((second, neighbor)))
                if first_witness is None or second_witness is None:
                    junctions = []
                    break
                geometry = _triangle_junction_geometry(
                    candidate, first_witness, second_witness
                )
                if geometry is None:
                    junctions = []
                    break
                junctions.append({"common_surface_id": neighbor, **geometry})
            if junctions and all(
                float(item["candidate_junction_outside_ratio"])
                >= thresholds.minimum_multiply_closed_junction_outside_ratio
                for item in junctions
            ):
                pruned_indices.add(candidate_index)
                conflicts.append(
                    {
                        "pruned_candidate_id": _candidate_id(candidate),
                        "lane": "multiply_closed_finite_nonjunction_chord",
                        "common_neighbor_count": len(common),
                        "triangle_junctions": junctions,
                    }
                )
                continue

        first_area = _value(candidate, "surface1_area")
        second_area = _value(candidate, "surface2_area")
        bend_angle = _value(candidate, "bend_angle")
        if (
            common
            and span is not None
            and first_area is not None
            and second_area is not None
            and min(first_area, second_area) > 0.0
            and min(first_area, second_area) / max(first_area, second_area)
            >= thresholds.minimum_broad_face_area_ratio
            and bend_angle is not None
            and bend_angle >= thresholds.minimum_broad_triangle_bend_angle_deg
        ):
            for neighbor in sorted(common, key=str):
                first_witness = edge_records.get(frozenset((first, neighbor)))
                second_witness = edge_records.get(frozenset((second, neighbor)))
                if first_witness is None or second_witness is None:
                    continue
                witness_spans = (
                    _value(first_witness, "span_mm"),
                    _value(second_witness, "span_mm"),
                )
                witness_angles = (
                    _axis_angle_deg(candidate, first_witness),
                    _axis_angle_deg(candidate, second_witness),
                )
                if any(value is None for value in witness_spans + witness_angles):
                    continue
                if span > thresholds.maximum_broad_triangle_span_ratio * min(
                    float(witness_spans[0]), float(witness_spans[1])
                ):
                    continue
                minimum_side_angle = min(
                    float(witness_angles[0]), float(witness_angles[1])
                )
                if (
                    minimum_side_angle
                    > thresholds.maximum_broad_triangle_side_angle_deg
                ):
                    continue
                pruned_indices.add(candidate_index)
                conflicts.append(
                    {
                        "pruned_candidate_id": _candidate_id(candidate),
                        "lane": "short_broad_face_triangle_chord",
                        "common_surface_id": neighbor,
                        "minimum_side_angle_deg": minimum_side_angle,
                        "span_to_shorter_side_ratio": span
                        / min(float(witness_spans[0]), float(witness_spans[1])),
                        "surface_area_ratio": min(first_area, second_area)
                        / max(first_area, second_area),
                    }
                )
                break
        if candidate_index in pruned_indices:
            continue

        if len(graph.get(first, set())) != 1 or len(graph.get(second, set())) != 1:
            continue
        candidate_transition = _value(candidate, "two_sided_transition_score")
        if candidate_transition is None:
            continue
        for witness in selected:
            witness_pair = _surface_pair(witness)
            if (
                witness is candidate
                or witness_pair is None
                or not _is_mesh_interface(witness)
                or pair & witness_pair
            ):
                continue
            witness_first, witness_second = tuple(witness_pair)
            if max(
                len(graph.get(witness_first, set())),
                len(graph.get(witness_second, set())),
            ) < 2:
                continue
            witness_transition = _value(witness, "two_sided_transition_score")
            if (
                witness_transition is None
                or witness_transition
                < candidate_transition
                + thresholds.minimum_isolated_transition_advantage
            ):
                continue
            geometry = _parallel_geometry(candidate, witness)
            if geometry is None:
                continue
            if (
                geometry["axis_angle_deg"]
                > thresholds.maximum_isolated_axis_angle_deg
                or not (
                    thresholds.minimum_isolated_offset_mm
                    <= geometry["transverse_offset_mm"]
                    <= thresholds.maximum_isolated_offset_mm
                )
                or geometry["shorter_span_overlap"]
                < thresholds.minimum_isolated_span_overlap
                or not (
                    thresholds.minimum_isolated_span_ratio
                    <= geometry["span_ratio"]
                    <= thresholds.maximum_isolated_span_ratio
                )
            ):
                continue
            pruned_indices.add(candidate_index)
            conflicts.append(
                {
                    "pruned_candidate_id": _candidate_id(candidate),
                    "witness_candidate_id": _candidate_id(witness),
                    "lane": "isolated_parallel_dominated_interface",
                    "transition_advantage": witness_transition
                    - candidate_transition,
                    **geometry,
                }
            )
            break

    return {
        "selected": [
            record
            for index, record in enumerate(selected)
            if index not in pruned_indices
        ],
        "pruned_candidate_ids": [
            _candidate_id(selected[index]) for index in sorted(pruned_indices)
        ],
        "conflicts": conflicts,
        "thresholds": {key: value for key, value in vars(thresholds).items()},
    }
