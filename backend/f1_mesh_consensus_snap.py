"""Snap raw mesh selections to geometrically equivalent consensus axes."""

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


def _duplicate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    offset_tolerance_mm: float,
) -> bool:
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
    return (
        angle <= 7.5
        and max(left_offset, right_offset) <= float(offset_tolerance_mm)
        and overlap / max(min(left_span, right_span), 1e-9) >= 0.45
    )


def _rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("cluster_source_count") or 1),
        int(record.get("cluster_member_count") or 1),
        float(
            record.get("cluster_mean_finite_support_fraction")
            or record.get("finite_support_fraction")
            or 0.0
        ),
        -float(
            record.get("cluster_mean_median_line_distance_mm")
            or record.get("median_line_distance_mm")
            or 1.0e9
        ),
        str(record.get("candidate_id") or record.get("bend_id") or ""),
    )


def _is_raw_mesh_selection(record: Mapping[str, Any]) -> bool:
    """Return whether ``record`` is a direct scan-mesh candidate.

    Cross-source line objects can carry mesh-like geometry, but snapping those
    objects changes a selector decision made from different evidence.  The
    consensus snap is only a representation refinement for a raw mesh choice.
    Older payloads did not always serialize ``candidate_source``, so the
    canonical ID remains a fail-closed compatibility fallback only when the
    source field is absent.
    """

    source = record.get("candidate_source")
    if source is not None:
        return str(source) == "scan_mesh_adjacency"
    return str(record.get("candidate_id") or record.get("bend_id") or "").startswith(
        "scan_mesh_adjacency::"
    )


def snap_selected_mesh_consensus(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
    *,
    minimum_source_count: int = 3,
    duplicate_offset_mm: float = 4.0,
) -> dict[str, Any]:
    """Replace selected mesh aliases with stronger mesh consensus representatives.

    The operation preserves selection count unless two snapped representatives
    become duplicates. It uses only generated line geometry and source consensus;
    no expected bend count, part identifier, or manual truth is consumed.
    """

    candidates = []
    for source in line_objects:
        record = dict(source)
        if (
            not str(record.get("candidate_id") or "").startswith(
                "scan_mesh_adjacency::"
            )
            or int(record.get("cluster_source_count") or 1)
            < int(minimum_source_count)
        ):
            continue
        try:
            _endpoints(record)
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(record)
    candidates.sort(key=_rank, reverse=True)

    output: list[dict[str, Any]] = []
    replacements = []
    for source in selected:
        original = dict(source)
        if not _is_raw_mesh_selection(original):
            output.append(original)
            continue
        aliases = [
            record
            for record in candidates
            if _duplicate(
                original,
                record,
                offset_tolerance_mm=duplicate_offset_mm,
            )
        ]
        replacement = max([original, *aliases], key=_rank)
        replacement = dict(replacement)
        original_id = str(original.get("candidate_id") or original.get("bend_id") or "")
        replacement_id = str(
            replacement.get("candidate_id") or replacement.get("bend_id") or ""
        )
        if replacement_id != original_id:
            replacement["selection_reason"] = "mesh_cross_source_consensus_snap"
            replacement["mesh_consensus_original_candidate_id"] = original_id
            replacements.append(
                {
                    "original_candidate_id": original_id,
                    "replacement_candidate_id": replacement_id,
                }
            )
        if any(
            _duplicate(
                replacement,
                retained,
                offset_tolerance_mm=duplicate_offset_mm,
            )
            for retained in output
        ):
            continue
        output.append(replacement)
    return {"replacements": replacements, "selected": output}
