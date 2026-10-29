"""Finite-line topology completion for fragmented scan-only F1 markers."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np


def merge_collinear_fragments(
    lines: Sequence[Mapping[str, Any]],
    *,
    axis_tolerance_deg: float = 5.0,
    offset_tolerance_mm: float = 5.0,
    gap_tolerance_mm: float = 25.0,
) -> list[dict[str, Any]]:
    """Merge finite fragments that lie on the same observed supporting line."""

    canonical = [_canonical_line(item) for item in lines]
    parent = list(range(len(canonical)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in itertools.combinations(range(len(canonical)), 2):
        left_endpoints, left_axis, _left_length = canonical[left]
        right_endpoints, right_axis, _right_length = canonical[right]
        if _axis_angle_deg(left_axis, right_axis) > axis_tolerance_deg:
            continue
        if float(np.dot(left_axis, right_axis)) < 0.0:
            right_endpoints = right_endpoints[::-1]
        delta = np.mean(right_endpoints, axis=0) - np.mean(left_endpoints, axis=0)
        offset = float(np.linalg.norm(delta - left_axis * np.dot(delta, left_axis)))
        if offset > offset_tolerance_mm:
            continue
        left_interval = sorted(float(value) for value in left_endpoints @ left_axis)
        right_interval = sorted(float(value) for value in right_endpoints @ left_axis)
        gap = max(
            0.0,
            right_interval[0] - left_interval[1],
            left_interval[0] - right_interval[1],
        )
        if gap <= gap_tolerance_mm:
            union(left, right)

    merged: list[dict[str, Any]] = []
    for root in sorted({find(index) for index in range(len(canonical))}):
        indices = [index for index in range(len(canonical)) if find(index) == root]
        if len(indices) < 2:
            continue
        axis = canonical[indices[0]][1]
        endpoints = np.vstack([canonical[index][0] for index in indices])
        center = np.mean(endpoints, axis=0)
        projections = endpoints @ axis
        low, high = float(np.min(projections)), float(np.max(projections))
        anchor = center + axis * (0.5 * (low + high) - float(np.dot(center, axis)))
        half_span = 0.5 * (high - low)
        source_ids = sorted(str(lines[index].get("bend_id") or lines[index].get("candidate_id")) for index in indices)
        merged.append(
            {
                "bend_id": f"MERGED_{len(merged) + 1}",
                "source_kind": "collinear_fragment_merge",
                "source_ids": source_ids,
                "support_count": len(indices),
                "anchor": anchor.tolist(),
                "axis_direction": axis.tolist(),
                "span_mm": 2.0 * half_span,
                "span_endpoints": [
                    (anchor - axis * half_span).tolist(),
                    (anchor + axis * half_span).tolist(),
                ],
            }
        )
    return merged


def complete_parallel_endpoint_connectors(
    lines: Sequence[Mapping[str, Any]],
    *,
    axis_tolerance_deg: float = 5.0,
    minimum_length_ratio: float = 0.75,
    minimum_connector_mm: float = 8.0,
    maximum_connector_mm: float = 450.0,
) -> list[dict[str, Any]]:
    """Connect corresponding endpoints of two finite parallel line objects.

    This closes rectangular scan layouts after fragmented opposite sides have
    been merged.  It proposes geometry only; downstream evidence selection is
    responsible for accepting or rejecting each connector.
    """

    canonical = [_canonical_line(item) for item in lines]
    connectors: list[dict[str, Any]] = []
    for left, right in itertools.combinations(range(len(canonical)), 2):
        left_endpoints, left_axis, left_length = canonical[left]
        right_endpoints, right_axis, right_length = canonical[right]
        if _axis_angle_deg(left_axis, right_axis) > axis_tolerance_deg:
            continue
        if min(left_length, right_length) / max(left_length, right_length) < minimum_length_ratio:
            continue
        if float(np.dot(left_axis, right_axis)) < 0.0:
            right_endpoints = right_endpoints[::-1]
        direct = np.linalg.norm(left_endpoints[0] - right_endpoints[0]) + np.linalg.norm(
            left_endpoints[1] - right_endpoints[1]
        )
        crossed = np.linalg.norm(left_endpoints[0] - right_endpoints[1]) + np.linalg.norm(
            left_endpoints[1] - right_endpoints[0]
        )
        if crossed < direct:
            right_endpoints = right_endpoints[::-1]
        for endpoint_index in range(2):
            start = left_endpoints[endpoint_index]
            end = right_endpoints[endpoint_index]
            length = float(np.linalg.norm(end - start))
            if not minimum_connector_mm <= length <= maximum_connector_mm:
                continue
            connectors.append(
                {
                    "bend_id": f"CONNECTOR_{left + 1}_{right + 1}_{endpoint_index + 1}",
                    "source_kind": "parallel_endpoint_connector",
                    "source_ids": [
                        str(lines[left].get("bend_id") or lines[left].get("candidate_id")),
                        str(lines[right].get("bend_id") or lines[right].get("candidate_id")),
                    ],
                    "support_count": 2,
                    "anchor": (0.5 * (start + end)).tolist(),
                    "axis_direction": ((end - start) / length).tolist(),
                    "span_mm": length,
                    "span_endpoints": [start.tolist(), end.tolist()],
                }
            )
    return connectors


def _canonical_line(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    if record.get("span_endpoints") is not None:
        endpoints = np.asarray(record["span_endpoints"], dtype=np.float64)
    else:
        anchor = np.asarray(record["anchor"], dtype=np.float64)
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        half = 0.5 * float(record["span_mm"]) * axis
        endpoints = np.asarray((anchor - half, anchor + half))
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("line endpoints must be a finite 2x3 array")
    delta = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("line span must be nonzero")
    axis = delta / length
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
        endpoints = endpoints[::-1]
    return endpoints, axis, length


def _axis_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))))

