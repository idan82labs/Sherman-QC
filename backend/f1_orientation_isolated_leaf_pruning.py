"""Prune a weak, directionally isolated leaf from a dense surface star."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def prune_orientation_isolated_weak_leaf(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Remove unsupported spokes while preserving well-covered terminal bends."""

    output = [dict(item) for item in selected]
    degree: dict[Any, int] = {}
    for item in output:
        left = item.get("surface1_id")
        right = item.get("surface2_id")
        if left is None or right is None or left == right:
            continue
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1

    pruned: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, candidate in enumerate(output):
        left = candidate.get("surface1_id")
        right = candidate.get("surface2_id")
        if left is None or right is None:
            retained.append(candidate)
            continue
        hub_degree = max(degree.get(left, 0), degree.get(right, 0))
        leaf_degree = min(degree.get(left, 0), degree.get(right, 0))
        other_angles = [
            parallel_line_metrics(candidate, other)["axis_angle_deg"]
            for other_index, other in enumerate(output)
            if other_index != index
        ]
        minimum_axis_angle = min(other_angles, default=0.0)
        weak_isolated_leaf = (
            hub_degree == 4
            and leaf_degree == 1
            and float(candidate.get("span_mm") or 0.0) <= 35.0
            and float(candidate.get("two_sided_transition_score") or 0.0) < 0.60
            and float(candidate.get("finite_support_fraction") or 0.0) <= 0.92
            and float(candidate.get("endpoint_support_fraction") or 0.0) <= 2.0 / 3.0
            and minimum_axis_angle >= 30.0
        )
        if not weak_isolated_leaf:
            retained.append(candidate)
            continue
        pruned.append(candidate)
        evidence.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "hub_degree": hub_degree,
                "leaf_degree": leaf_degree,
                "minimum_axis_angle_deg": minimum_axis_angle,
            }
        )
    return {
        "selected": retained,
        "pruned": pruned,
        "activated": bool(pruned),
        "evidence": evidence,
    }

