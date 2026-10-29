"""Prune outline fans when exactly one candidate has physical transition support."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _transition_score(candidate: Mapping[str, Any]) -> float:
    return float(
        candidate.get("two_sided_transition_score")
        or candidate.get("cluster_max_two_sided_transition_score")
        or 0.0
    )


def prune_transition_dominant_outline_fan(
    selected: Sequence[Mapping[str, Any]],
    *,
    minimum_candidate_count: int = 4,
    minimum_dominant_score: float = 0.60,
    maximum_runner_up_score: float = 0.35,
) -> dict[str, Any]:
    """Keep one physical interface only under an unambiguous score margin.

    This lane targets small bracket scans where boundary outlines create a fan
    of plausible mesh edges. It abstains whenever two candidates have material
    two-sided normal-transition evidence, preserving real multi-bend parts.
    """

    output = [dict(item) for item in selected]
    if len(output) < minimum_candidate_count:
        return {"selected": output, "pruned": [], "activated": False, "reason": "candidate_count_below_gate"}
    ranked = sorted(
        output,
        key=lambda item: (-_transition_score(item), str(item.get("candidate_id") or item.get("bend_id") or "")),
    )
    dominant_score = _transition_score(ranked[0])
    runner_up_score = _transition_score(ranked[1])
    if dominant_score < minimum_dominant_score or runner_up_score >= maximum_runner_up_score:
        return {
            "selected": output,
            "pruned": [],
            "activated": False,
            "reason": "transition_dominance_margin_not_met",
            "dominant_score": dominant_score,
            "runner_up_score": runner_up_score,
        }
    retained = dict(ranked[0])
    retained["selection_reason"] = "unique_two_sided_transition_in_outline_fan"
    pruned = []
    for item in ranked[1:]:
        removed = dict(item)
        removed["pruning_reason"] = "outline_fan_without_material_two_sided_transition"
        pruned.append(removed)
    return {
        "selected": [retained],
        "pruned": pruned,
        "activated": True,
        "reason": "unique_transition_dominant_candidate",
        "dominant_score": dominant_score,
        "runner_up_score": runner_up_score,
    }
