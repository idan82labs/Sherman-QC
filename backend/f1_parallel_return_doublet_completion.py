"""Recover a distinct parallel return bend suppressed as a duplicate.

Scan-mesh segmentation can produce two close, nearly parallel interfaces around
a narrow return.  The selector's ordinary spatial deduplication correctly
removes most such pairs because they are alternate outlines of one bend.  A
real two-bend return is distinguishable when the interfaces have complementary
bend angles and materially different supported spans while remaining strongly
supported transitions.

The rule is deliberately narrow and uses geometry only.  It does not consume a
part identifier, expected bend count, CAD, or manual truth.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _endpoints(item: Mapping[str, Any]) -> np.ndarray:
    endpoints = item.get("span_endpoints")
    if endpoints is None:
        endpoints = [item["line_start"], item["line_end"]]
    value = np.asarray(endpoints, dtype=np.float64)
    if value.shape != (2, 3):
        raise ValueError("bend endpoints must have shape (2, 3)")
    return value


def parallel_line_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    left_line = _endpoints(left)
    right_line = _endpoints(right)
    left_delta = left_line[1] - left_line[0]
    right_delta = right_line[1] - right_line[0]
    left_span = float(np.linalg.norm(left_delta))
    right_span = float(np.linalg.norm(right_delta))
    if left_span <= 1.0e-9 or right_span <= 1.0e-9:
        raise ValueError("bend lines must have nonzero length")
    left_axis = left_delta / left_span
    right_axis = right_delta / right_span
    dot = float(np.clip(np.dot(left_axis, right_axis), -1.0, 1.0))
    axis_angle = math.degrees(math.acos(abs(dot)))
    if dot < 0.0:
        right_axis *= -1.0
    left_midpoint = np.mean(left_line, axis=0)
    right_midpoint = np.mean(right_line, axis=0)
    midpoint_delta = right_midpoint - left_midpoint
    transverse_offset = max(
        float(np.linalg.norm(midpoint_delta - left_axis * np.dot(midpoint_delta, left_axis))),
        float(np.linalg.norm(midpoint_delta - right_axis * np.dot(midpoint_delta, right_axis))),
    )
    left_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in left_line)
    right_interval = sorted(float(np.dot(point - left_midpoint, left_axis)) for point in right_line)
    overlap = max(
        0.0,
        min(left_interval[1], right_interval[1]) - max(left_interval[0], right_interval[0]),
    )
    return {
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": transverse_offset,
        "shorter_span_overlap": overlap / min(left_span, right_span),
        "relative_span_difference": abs(left_span - right_span) / max(left_span, right_span),
    }


def complete_parallel_return_doublet(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add only strongly evidenced close interfaces that are distinct returns."""

    output = [dict(item) for item in selected]
    selected_ids = {str(item.get("candidate_id") or "") for item in output}
    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seeds = [
        item
        for item in output
        if item.get("candidate_source") == "scan_mesh_adjacency"
        and float(item.get("selection_probability") or 0.0) >= 0.80
        and float(item.get("two_sided_transition_score") or 0.0) >= 0.70
        and float(item.get("confidence") or 0.0) >= 0.80
    ]
    for candidate_record in candidates:
        candidate = dict(candidate_record)
        candidate_id = str(candidate.get("candidate_id") or "")
        if (
            not candidate_id
            or candidate_id in selected_ids
            or candidate.get("candidate_source") != "scan_mesh_adjacency"
            or float(candidate.get("two_sided_transition_score") or 0.0) < 0.70
            or float(candidate.get("confidence") or 0.0) < 0.80
        ):
            continue
        for seed in seeds:
            metrics = parallel_line_metrics(seed, candidate)
            bend_angle_difference = abs(
                float(seed.get("bend_angle") or 0.0)
                - float(candidate.get("bend_angle") or 0.0)
            )
            if not (
                metrics["axis_angle_deg"] <= 3.0
                and 1.0 <= metrics["transverse_offset_mm"] <= 3.5
                and metrics["shorter_span_overlap"] >= 0.85
                and metrics["relative_span_difference"] >= 0.05
                and bend_angle_difference >= 25.0
            ):
                continue
            candidate["candidate_source"] = "parallel_return_doublet_completion"
            candidate["completion_seed_id"] = seed.get("candidate_id")
            candidate["completion_evidence"] = {
                **metrics,
                "bend_angle_difference_deg": bend_angle_difference,
            }
            additions.append(candidate)
            evidence.append(
                {
                    "candidate_id": candidate_id,
                    "seed_id": seed.get("candidate_id"),
                    **candidate["completion_evidence"],
                }
            )
            selected_ids.add(candidate_id)
            break
    output.extend(additions)
    return {
        "selected": output,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }
