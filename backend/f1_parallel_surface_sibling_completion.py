"""Count-free recovery of distinct close parallel scan-mesh interfaces.

Finite-line deduplication is intentionally aggressive, but two physical folds can
occasionally be closer than the geometric duplicate radius.  This module only
restores such a line when the scan-mesh surface graph proves that it is a
different interface: either the line bridges two disconnected selected surface
components, or it adds a leaf to an already explicit high-degree fold hub.
"""

from __future__ import annotations

import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


def _endpoints(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("span_endpoints")
    if value is None and record.get("line_start") is not None:
        value = [record["line_start"], record["line_end"]]
    endpoints = np.asarray(value, dtype=np.float64)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 1.0e-9:
        raise ValueError("line span must be nonzero")
    return endpoints


def _surface_pair(record: Mapping[str, Any]) -> frozenset[Hashable] | None:
    first = record.get("surface1_id")
    second = record.get("surface2_id")
    if first is None or second is None or first == second:
        return None
    try:
        return frozenset((first, second))
    except TypeError:
        return None


def _value(record: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = record.get(name)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return 0.0


def _line_metrics(
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
        math.acos(
            float(np.clip(abs(float(np.dot(left_axis, right_axis))), 0.0, 1.0))
        )
    )
    left_midpoint = np.mean(left_endpoints, axis=0)
    right_midpoint = np.mean(right_endpoints, axis=0)
    midpoint_delta = right_midpoint - left_midpoint
    offsets = (
        float(
            np.linalg.norm(
                midpoint_delta - left_axis * np.dot(midpoint_delta, left_axis)
            )
        ),
        float(
            np.linalg.norm(
                midpoint_delta - right_axis * np.dot(midpoint_delta, right_axis)
            )
        ),
    )
    projected = sorted(
        float(np.dot(point - left_midpoint, left_axis)) for point in right_endpoints
    )
    overlap = max(
        0.0,
        min(0.5 * left_span, projected[1])
        - max(-0.5 * left_span, projected[0]),
    )
    return (
        angle,
        max(offsets),
        overlap / max(min(left_span, right_span), 1.0e-9),
    )


def _selected_surface_graph(
    selected: Sequence[Mapping[str, Any]],
) -> dict[Hashable, set[Hashable]]:
    graph: dict[Hashable, set[Hashable]] = {}
    for record in selected:
        pair = _surface_pair(record)
        if pair is None:
            continue
        first, second = tuple(pair)
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
    return graph


def _connected(
    graph: Mapping[Hashable, set[Hashable]], first: Hashable, second: Hashable
) -> bool:
    pending = [first]
    visited: set[Hashable] = set()
    while pending:
        node = pending.pop()
        if node == second:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(graph.get(node, set()) - visited)
    return False


def _topology_reason(
    pair: frozenset[Hashable],
    sibling_pair: frozenset[Hashable],
    graph: Mapping[Hashable, set[Hashable]],
    *,
    minimum_hub_degree: int,
) -> str | None:
    shared = pair & sibling_pair
    if len(shared) != 1:
        return None
    first, second = tuple(pair)
    if first in graph and second in graph and not _connected(graph, first, second):
        return "disconnected_surface_component_bridge"
    new_surfaces = pair - graph.keys()
    if len(new_surfaces) != 1:
        return None
    common = next(iter(shared))
    if common in graph and len(graph[common]) >= minimum_hub_degree:
        return "high_degree_fold_hub_leaf"
    return None


def _is_mesh_interface(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("candidate_id") or "").startswith("scan_mesh_adjacency::")
        or str(record.get("candidate_source") or "") == "scan_mesh_adjacency"
        or str(record.get("representative_source") or "") == "scan_mesh_adjacency"
    )


def _has_complementary_turn(
    candidate: Mapping[str, Any], sibling: Mapping[str, Any]
) -> bool:
    candidate_angle = _value(candidate, "bend_angle", "angle_deg")
    sibling_angle = _value(sibling, "bend_angle", "angle_deg")
    return (
        min(candidate_angle, sibling_angle) <= 110.0
        and max(candidate_angle, sibling_angle) >= 120.0
    )


def complete_parallel_surface_siblings(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_hub_degree: int = 5,
    minimum_span_mm: float = 40.0,
    minimum_finite_support_fraction: float = 0.95,
    minimum_endpoint_support_fraction: float = 0.8,
    maximum_median_line_distance_mm: float = 1.1,
    maximum_p90_line_distance_mm: float = 2.5,
    minimum_normal_turn_deg: float = 80.0,
    minimum_normal_coherence: float = 0.7,
    maximum_surface_gap_mm: float = 2.0,
    maximum_axis_angle_deg: float = 2.0,
    minimum_offset_mm: float = 0.75,
    maximum_offset_mm: float = 2.0,
    minimum_span_overlap: float = 0.9,
) -> dict[str, Any]:
    """Append surface-proven close siblings without identity or count inputs.

    The evidence gates deliberately require a fully supported finite line.  The
    nonzero offset floor prevents a second representation of the same axis,
    while the surface-graph rule prevents a strong outline from being accepted
    merely because it runs parallel to a selected bend.
    """

    output = [dict(record) for record in selected]
    graph = _selected_surface_graph(output)
    selected_pairs = {
        pair for record in output if (pair := _surface_pair(record)) is not None
    }
    selected_ids = {
        str(record.get("candidate_id") or record.get("bend_id") or "")
        for record in output
    }
    ranked = sorted(
        (dict(record) for record in candidates),
        key=lambda record: str(
            record.get("candidate_id") or record.get("bend_id") or ""
        ),
    )
    added: list[dict[str, Any]] = []
    for candidate in ranked:
        candidate_id = str(
            candidate.get("candidate_id") or candidate.get("bend_id") or ""
        )
        pair = _surface_pair(candidate)
        if (
            not candidate_id
            or candidate_id in selected_ids
            or pair is None
            or pair in selected_pairs
            or not _is_mesh_interface(candidate)
        ):
            continue
        try:
            endpoints = _endpoints(candidate)
            span_mm = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        except (TypeError, ValueError):
            continue
        if not (
            span_mm >= minimum_span_mm
            and _value(candidate, "finite_support_fraction")
            >= minimum_finite_support_fraction
            and _value(candidate, "finite_support_longest_run_fraction")
            >= minimum_finite_support_fraction
            and _value(candidate, "endpoint_support_fraction")
            >= minimum_endpoint_support_fraction
            and _value(candidate, "median_line_distance_mm")
            <= maximum_median_line_distance_mm
            and _value(candidate, "p90_line_distance_mm")
            <= maximum_p90_line_distance_mm
            and _value(candidate, "two_sided_normal_angle_deg")
            >= minimum_normal_turn_deg
            and _value(candidate, "two_sided_normal_coherence")
            >= minimum_normal_coherence
            and 0.0 < _value(candidate, "min_surface_gap")
            <= maximum_surface_gap_mm
        ):
            continue

        witnesses: list[tuple[str, float, float, float, Mapping[str, Any]]] = []
        for sibling in output:
            sibling_pair = _surface_pair(sibling)
            if sibling_pair is None or not _has_complementary_turn(candidate, sibling):
                continue
            reason = _topology_reason(
                pair,
                sibling_pair,
                graph,
                minimum_hub_degree=minimum_hub_degree,
            )
            if reason is None:
                continue
            try:
                angle, offset, overlap = _line_metrics(candidate, sibling)
            except (TypeError, ValueError):
                continue
            if (
                angle <= maximum_axis_angle_deg
                and minimum_offset_mm <= offset <= maximum_offset_mm
                and overlap >= minimum_span_overlap
            ):
                witnesses.append((reason, angle, offset, overlap, sibling))
        if not witnesses:
            continue
        reason, angle, offset, overlap, sibling = min(
            witnesses,
            key=lambda item: (
                item[2],
                item[1],
                str(item[4].get("candidate_id") or item[4].get("bend_id") or ""),
            ),
        )
        candidate["selection_reason"] = "parallel_surface_sibling_completion"
        candidate["parallel_surface_sibling_topology"] = reason
        candidate["parallel_surface_sibling_witness_id"] = str(
            sibling.get("candidate_id") or sibling.get("bend_id") or ""
        )
        output.append(candidate)
        selected_ids.add(candidate_id)
        selected_pairs.add(pair)
        added.append(
            {
                "candidate_id": candidate_id,
                "surface_pair": sorted(
                    pair, key=lambda value: (type(value).__name__, repr(value))
                ),
                "topology": reason,
                "witness_id": candidate["parallel_surface_sibling_witness_id"],
                "axis_angle_deg": angle,
                "transverse_offset_mm": offset,
                "shorter_span_overlap": overlap,
            }
        )
    return {"enabled": True, "added": added, "selected": output}
