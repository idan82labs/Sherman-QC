"""Prune weak parallel surface-interface branches that behave like outlines.

The mesh selector occasionally emits two nearly parallel interfaces that share
one fitted surface.  A real fold branch has materially stronger two-sided scan
transition evidence, while the competing perimeter branch concentrates normal
variation only in the tightest corridor.  This helper removes only those narrow
pairwise conflicts.  It never inspects a part identifier, an expected bend
count, a model probability, or manual truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WeakParallelOutlineThresholds:
    maximum_axis_angle_deg: float = 3.0
    minimum_transverse_offset_mm: float = 6.0
    maximum_transverse_offset_mm: float = 22.0
    minimum_shorter_span_overlap: float = 0.90
    minimum_transition_advantage: float = 0.25
    maximum_narrow_transition_score: float = 0.15
    maximum_narrow_transition_angle_deg: float = 35.0
    minimum_narrow_tight_to_wide_ratio: float = 1.20
    maximum_narrow_tight_to_wide_ratio: float = 2.00
    maximum_concentrated_transition_score: float = 0.45
    minimum_concentrated_tight_to_wide_ratio: float = 1.20
    maximum_witness_tight_to_wide_ratio: float = 0.30


def _finite_float(candidate: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(candidate[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _surface_pair(candidate: Mapping[str, Any]) -> frozenset[Any] | None:
    first = candidate.get("surface1_id")
    second = candidate.get("surface2_id")
    if first is None or second is None or first == second:
        return None
    return frozenset((first, second))


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


def _parallel_geometry(
    candidate: Mapping[str, Any],
    witness: Mapping[str, Any],
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
    if candidate_span <= 1.0e-9 or witness_span <= 1.0e-9:
        return None
    candidate_axis = candidate_vector / candidate_span
    witness_axis = witness_vector / witness_span
    cosine = float(np.clip(abs(np.dot(candidate_axis, witness_axis)), 0.0, 1.0))
    axis_angle = degrees(acos(cosine))

    displacement = witness_start - candidate_start
    transverse = displacement - float(np.dot(displacement, candidate_axis)) * candidate_axis
    transverse_offset = float(np.linalg.norm(transverse))
    projected = sorted(
        (
            float(np.dot(witness_start - candidate_start, candidate_axis)),
            float(np.dot(witness_end - candidate_start, candidate_axis)),
        )
    )
    overlap = max(
        0.0,
        min(candidate_span, projected[1]) - max(0.0, projected[0]),
    )
    return {
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": transverse_offset,
        "shorter_span_overlap": overlap / min(candidate_span, witness_span),
    }


def _candidate_lane(
    candidate: Mapping[str, Any],
    witness: Mapping[str, Any],
    thresholds: WeakParallelOutlineThresholds,
) -> str | None:
    transition = _finite_float(candidate, "two_sided_transition_score")
    transition_angle = _finite_float(candidate, "two_sided_normal_angle_deg")
    tight_to_wide = _finite_float(candidate, "tight_to_wide_normal_second_ratio")
    witness_tight_to_wide = _finite_float(
        witness, "tight_to_wide_normal_second_ratio"
    )
    if transition is None or transition_angle is None or tight_to_wide is None:
        return None

    source = candidate.get("candidate_source")
    if (
        source == "scan_mesh_adjacency"
        and transition <= thresholds.maximum_narrow_transition_score
        and transition_angle <= thresholds.maximum_narrow_transition_angle_deg
        and thresholds.minimum_narrow_tight_to_wide_ratio
        <= tight_to_wide
        <= thresholds.maximum_narrow_tight_to_wide_ratio
    ):
        return "narrow_one_sided_transition"
    if (
        source == "cross_source_line_object"
        and witness_tight_to_wide is not None
        and transition <= thresholds.maximum_concentrated_transition_score
        and tight_to_wide >= thresholds.minimum_concentrated_tight_to_wide_ratio
        and witness_tight_to_wide <= thresholds.maximum_witness_tight_to_wide_ratio
    ):
        return "tight_corridor_concentration"
    return None


def prune_weak_parallel_outlines(
    selected: Sequence[Mapping[str, Any]],
    thresholds: WeakParallelOutlineThresholds = WeakParallelOutlineThresholds(),
) -> dict[str, Any]:
    """Remove only weak, parallel, shared-surface outline conflicts.

    The operation is deterministic and order preserving.  Missing geometry or
    evidence fails closed, leaving the candidate untouched.
    """

    pruned_indices: set[int] = set()
    conflicts: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected):
        candidate_surfaces = _surface_pair(candidate)
        if candidate_surfaces is None:
            continue
        candidate_transition = _finite_float(
            candidate, "two_sided_transition_score"
        )
        if candidate_transition is None:
            continue
        for witness_index, witness in enumerate(selected):
            if candidate_index == witness_index:
                continue
            witness_surfaces = _surface_pair(witness)
            if witness_surfaces is None or len(candidate_surfaces & witness_surfaces) != 1:
                continue
            witness_transition = _finite_float(
                witness, "two_sided_transition_score"
            )
            if (
                witness_transition is None
                or witness_transition
                < candidate_transition + thresholds.minimum_transition_advantage
            ):
                continue
            lane = _candidate_lane(candidate, witness, thresholds)
            if lane is None:
                continue
            geometry = _parallel_geometry(candidate, witness)
            if geometry is None:
                continue
            if geometry["axis_angle_deg"] > thresholds.maximum_axis_angle_deg:
                continue
            if not (
                thresholds.minimum_transverse_offset_mm
                <= geometry["transverse_offset_mm"]
                <= thresholds.maximum_transverse_offset_mm
            ):
                continue
            if (
                geometry["shorter_span_overlap"]
                < thresholds.minimum_shorter_span_overlap
            ):
                continue
            pruned_indices.add(candidate_index)
            conflicts.append(
                {
                    "pruned_candidate_id": candidate.get("candidate_id")
                    or candidate.get("bend_id"),
                    "witness_candidate_id": witness.get("candidate_id")
                    or witness.get("bend_id"),
                    "lane": lane,
                    "transition_advantage": witness_transition
                    - candidate_transition,
                    **geometry,
                }
            )
            break

    return {
        "selected": [
            candidate
            for index, candidate in enumerate(selected)
            if index not in pruned_indices
        ],
        "pruned_candidate_ids": [
            selected[index].get("candidate_id") or selected[index].get("bend_id")
            for index in sorted(pruned_indices)
        ],
        "conflicts": conflicts,
        "thresholds": {
            key: value for key, value in vars(thresholds).items()
        },
    }
