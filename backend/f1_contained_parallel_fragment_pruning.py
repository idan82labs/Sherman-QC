"""Conservative pruning for contained, parallel short-line fragments.

This post-selection rule is intentionally geometric.  It does not inspect a
part identifier, an expected bend count, model probabilities, or local scan
cross-section features.  A short line is removed only when it is almost fully
contained along a much longer, nearly parallel selected line and occupies a
narrow transverse-offset band.  The narrow contract keeps this helper inert
for ordinary terminal bends and for distinct, similarly sized parallel folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ContainedParallelFragmentThresholds:
    maximum_short_span_mm: float = 10.0
    minimum_long_to_short_ratio: float = 4.0
    maximum_axis_angle_deg: float = 3.0
    minimum_transverse_offset_mm: float = 5.0
    maximum_transverse_offset_mm: float = 8.0
    minimum_short_span_overlap: float = 0.90


def _endpoints(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    span = candidate.get("span_endpoints")
    start = candidate.get("line_start")
    end = candidate.get("line_end")
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


def _contained_parallel_geometry(
    short: Mapping[str, Any],
    long: Mapping[str, Any],
) -> dict[str, float] | None:
    short_endpoints = _endpoints(short)
    long_endpoints = _endpoints(long)
    if short_endpoints is None or long_endpoints is None:
        return None
    short_start, short_end = short_endpoints
    long_start, long_end = long_endpoints
    short_vector = short_end - short_start
    long_vector = long_end - long_start
    short_span = float(np.linalg.norm(short_vector))
    long_span = float(np.linalg.norm(long_vector))
    if short_span <= 1.0e-9 or long_span <= 1.0e-9:
        return None
    short_axis = short_vector / short_span
    long_axis = long_vector / long_span
    cosine = float(np.clip(abs(np.dot(short_axis, long_axis)), 0.0, 1.0))
    axis_angle = degrees(acos(cosine))

    displacement = short_start - long_start
    transverse = displacement - float(np.dot(displacement, long_axis)) * long_axis
    transverse_offset = float(np.linalg.norm(transverse))
    projected = sorted(
        (
            float(np.dot(short_start - long_start, long_axis)),
            float(np.dot(short_end - long_start, long_axis)),
        )
    )
    overlap = max(0.0, min(long_span, projected[1]) - max(0.0, projected[0]))
    return {
        "short_span_mm": short_span,
        "long_span_mm": long_span,
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": transverse_offset,
        "short_span_overlap": overlap / short_span,
    }


def prune_contained_parallel_fragments(
    selected: Sequence[Mapping[str, Any]],
    thresholds: ContainedParallelFragmentThresholds = ContainedParallelFragmentThresholds(),
) -> dict[str, Any]:
    """Remove only strict contained-parallel fragment conflicts.

    The operation is deterministic and order preserving.  It fails closed for
    malformed geometry and never uses the number of selected lines as a target.
    """

    pruned_indices: set[int] = set()
    conflicts: list[dict[str, Any]] = []
    for short_index, short in enumerate(selected):
        for long_index, long in enumerate(selected):
            if short_index == long_index:
                continue
            geometry = _contained_parallel_geometry(short, long)
            if geometry is None:
                continue
            if geometry["short_span_mm"] > thresholds.maximum_short_span_mm:
                continue
            if geometry["long_span_mm"] < (
                thresholds.minimum_long_to_short_ratio * geometry["short_span_mm"]
            ):
                continue
            if geometry["axis_angle_deg"] > thresholds.maximum_axis_angle_deg:
                continue
            if not (
                thresholds.minimum_transverse_offset_mm
                <= geometry["transverse_offset_mm"]
                <= thresholds.maximum_transverse_offset_mm
            ):
                continue
            if geometry["short_span_overlap"] < thresholds.minimum_short_span_overlap:
                continue
            pruned_indices.add(short_index)
            conflicts.append(
                {
                    "pruned_candidate_id": short.get("candidate_id") or short.get("bend_id"),
                    "dominating_candidate_id": long.get("candidate_id") or long.get("bend_id"),
                    **geometry,
                }
            )
            break

    return {
        "selected": [candidate for index, candidate in enumerate(selected) if index not in pruned_indices],
        "pruned_candidate_ids": [
            selected[index].get("candidate_id") or selected[index].get("bend_id")
            for index in sorted(pruned_indices)
        ],
        "conflicts": conflicts,
        "thresholds": {
            "maximum_short_span_mm": thresholds.maximum_short_span_mm,
            "minimum_long_to_short_ratio": thresholds.minimum_long_to_short_ratio,
            "maximum_axis_angle_deg": thresholds.maximum_axis_angle_deg,
            "minimum_transverse_offset_mm": thresholds.minimum_transverse_offset_mm,
            "maximum_transverse_offset_mm": thresholds.maximum_transverse_offset_mm,
            "minimum_short_span_overlap": thresholds.minimum_short_span_overlap,
        },
    }
