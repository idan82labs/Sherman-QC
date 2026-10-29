"""Count-free recovery from selected scan-mesh surface wedges."""

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


def _axis_angle(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    left_axis = left_endpoints[1] - left_endpoints[0]
    right_axis = right_endpoints[1] - right_endpoints[0]
    left_axis /= np.linalg.norm(left_axis)
    right_axis /= np.linalg.norm(right_axis)
    return math.degrees(
        math.acos(
            float(np.clip(abs(float(np.dot(left_axis, right_axis))), 0.0, 1.0))
        )
    )


def _duplicate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    angle_tolerance_deg: float = 7.5,
    offset_tolerance_mm: float = 4.0,
) -> bool:
    if _axis_angle(left, right) > angle_tolerance_deg:
        return False
    left_endpoints = _endpoints(left)
    right_endpoints = _endpoints(right)
    left_delta = left_endpoints[1] - left_endpoints[0]
    right_delta = right_endpoints[1] - right_endpoints[0]
    left_span = float(np.linalg.norm(left_delta))
    right_span = float(np.linalg.norm(right_delta))
    left_axis = left_delta / left_span
    right_axis = right_delta / right_span
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
        max(offsets) <= offset_tolerance_mm
        and overlap / max(min(left_span, right_span), 1.0e-9) >= 0.45
    )


def complete_surface_wedges(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
    *,
    minimum_source_count: int = 2,
    maximum_single_source_span_mm: float = 10.0,
    duplicate_offset_mm: float = 4.0,
) -> dict[str, Any]:
    """Complete selected two-edge surface wedges with generated mesh evidence.

    Two selected mesh edges ``A-B`` and ``A-C`` expose a missing surface-pair
    edge ``B-C``.  The missing edge is appended only when it already exists in
    the generated scan-mesh line-object pool and is backed by at least
    ``minimum_source_count`` generator families, or is a short single-source
    interface.  Only the original selected graph defines wedges, so additions
    cannot recursively create more additions.

    No part identity, manual truth, or expected bend count is accepted.  The
    operation fails closed on ambiguous selected edges and rejects geometric
    duplicates of existing selections.
    """

    output = [dict(record) for record in selected]
    if minimum_source_count < 1 or maximum_single_source_span_mm <= 0.0:
        return {"enabled": False, "added": [], "selected": output}

    selected_edges: dict[frozenset[Hashable], int] = {}
    for index, record in enumerate(output):
        pair = _surface_pair(record)
        if pair is None:
            continue
        try:
            _endpoints(record)
        except (TypeError, ValueError):
            continue
        if pair in selected_edges:
            selected_edges[pair] = -1
        else:
            selected_edges[pair] = index
    selected_pairs = {pair for pair, index in selected_edges.items() if index >= 0}
    selected_surfaces = set().union(*selected_pairs) if selected_pairs else set()

    ranked_by_pair: dict[frozenset[Hashable], list[dict[str, Any]]] = {}
    for source in line_objects:
        record = dict(source)
        pair = _surface_pair(record)
        if pair is None or pair in selected_pairs or not pair <= selected_surfaces:
            continue
        if not (
            str(record.get("candidate_id") or "").startswith(
                "scan_mesh_adjacency::"
            )
            or str(record.get("representative_source") or "")
            == "scan_mesh_adjacency"
        ):
            continue
        try:
            endpoints = _endpoints(record)
        except (TypeError, ValueError):
            continue
        span_mm = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        source_count = int(record.get("cluster_source_count") or 1)
        if (
            source_count < minimum_source_count
            and span_mm > maximum_single_source_span_mm
        ):
            continue
        first, second = tuple(pair)
        if not any(
            frozenset((first, middle)) in selected_pairs
            and frozenset((second, middle)) in selected_pairs
            for middle in selected_surfaces - pair
        ):
            continue
        ranked_by_pair.setdefault(pair, []).append(record)

    added: list[dict[str, Any]] = []
    for pair in sorted(
        ranked_by_pair,
        key=lambda value: tuple(
            (type(item).__name__, repr(item))
            for item in sorted(
                value, key=lambda item: (type(item).__name__, repr(item))
            )
        ),
    ):
        candidates = ranked_by_pair[pair]
        candidates.sort(
            key=lambda record: (
                -int(record.get("cluster_source_count") or 1),
                -float(
                    record.get("cluster_max_two_sided_normal_angle_deg")
                    or record.get("two_sided_normal_angle_deg")
                    or 0.0
                ),
                str(record.get("candidate_id") or record.get("bend_id") or ""),
            )
        )
        candidate = candidates[0]
        try:
            is_duplicate = any(
                _duplicate(
                    candidate,
                    retained,
                    offset_tolerance_mm=duplicate_offset_mm,
                )
                for retained in output
            )
        except (TypeError, ValueError):
            continue
        if is_duplicate:
            continue
        candidate["selection_reason"] = "surface_wedge_completion"
        output.append(candidate)
        added.append(
            {
                "candidate_id": str(
                    candidate.get("candidate_id") or candidate.get("bend_id") or ""
                ),
                "surface_pair": sorted(
                    pair, key=lambda value: (type(value).__name__, repr(value))
                ),
                "cluster_source_count": int(
                    candidate.get("cluster_source_count") or 1
                ),
            }
        )
    added.sort(key=lambda item: item["candidate_id"])
    return {"enabled": True, "added": added, "selected": output}
