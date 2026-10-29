"""Complete strong long and tiny bend consensus on established bend objects."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def _is_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    metrics = parallel_line_metrics(left, right)
    return (
        metrics["axis_angle_deg"] <= 7.5
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.45
    )


def _eligible(candidate: Mapping[str, Any]) -> tuple[bool, str]:
    source = str(candidate.get("representative_source") or "")
    if source not in {"cylindrical_standard", "plane_lattice"}:
        return False, "unsupported_source"
    if (
        float(candidate.get("finite_support_fraction") or 0.0) < 0.90
        or float(candidate.get("endpoint_support_fraction") or 0.0) < 0.80
    ):
        return False, "incomplete_scan_support"
    source_count = int(candidate.get("cluster_source_count") or 0)
    transition = float(
        candidate.get("cluster_max_two_sided_transition_score")
        or candidate.get("two_sided_transition_score")
        or 0.0
    )
    span = float(candidate.get("span_mm") or 0.0)
    score_percentile = float(candidate.get("case_percentile_score") or 0.0)
    if source_count >= 4 and transition >= 0.95 and span >= 100.0:
        return True, "long_four_source_consensus"
    if (
        source_count >= 4
        and transition >= 0.63
        and span <= 15.0
        and (source == "plane_lattice" or score_percentile >= 0.20)
    ):
        return True, "tiny_four_source_consensus"
    return False, "outside_multiscale_contract"


def complete_multiscale_consensus(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add consensus lines only after five bends establish a complex object."""

    output = [dict(item) for item in selected]
    if len(output) < 5:
        return {
            "selected": output,
            "added": [],
            "activated": False,
            "reason": "fewer_than_five_selected_bends",
            "evidence": [],
        }
    eligible = []
    modes: dict[str, str] = {}
    for record in line_objects:
        candidate = dict(record)
        accepted, mode = _eligible(candidate)
        if not accepted:
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        modes[candidate_id] = mode
        eligible.append(candidate)
    eligible.sort(
        key=lambda item: (
            -int(item.get("cluster_source_count") or 0),
            -float(
                item.get("cluster_max_two_sided_transition_score")
                or item.get("two_sided_transition_score")
                or 0.0
            ),
            -float(item.get("case_percentile_score") or 0.0),
            str(item.get("candidate_id") or ""),
        )
    )
    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for candidate in eligible:
        if any(_is_duplicate(candidate, item) for item in output + additions):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        candidate["candidate_source"] = "multiscale_consensus_completion"
        candidate["completion_mode"] = modes[candidate_id]
        additions.append(candidate)
        evidence.append(
            {
                "candidate_id": candidate_id,
                "completion_mode": modes[candidate_id],
                "cluster_source_count": int(candidate.get("cluster_source_count") or 0),
            }
        )
    output.extend(additions)
    return {
        "selected": output,
        "added": additions,
        "activated": bool(additions),
        "reason": "completed" if additions else "no_unrepresented_consensus_lines",
        "evidence": evidence,
    }

