"""Complete the final separated long consensus line in a mature bend object."""

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


def complete_residual_long_consensus(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add a maximum-ranked long line separated from duplicate outlines."""

    output = [dict(item) for item in selected]
    if len(output) < 9:
        return {
            "selected": output,
            "added": [],
            "activated": False,
            "reason": "fewer_than_nine_selected_bends",
            "evidence": [],
        }
    options = []
    for record in line_objects:
        candidate = dict(record)
        if (
            candidate.get("representative_source")
            not in {"cylindrical_standard", "cylindrical_wide"}
            or float(candidate.get("span_mm") or 0.0) < 180.0
            or int(candidate.get("cluster_source_count") or 0) < 2
            or float(candidate.get("cluster_max_two_sided_transition_score") or 0.0) < 0.90
            or float(candidate.get("case_percentile_score") or 0.0) < 1.0
            or float(candidate.get("finite_support_fraction") or 0.0) < 0.90
            or float(candidate.get("endpoint_support_fraction") or 0.0) < 0.80
            or any(_is_duplicate(candidate, item) for item in output)
        ):
            continue
        parallels = []
        for existing in output:
            metrics = parallel_line_metrics(candidate, existing)
            if (
                metrics["axis_angle_deg"] <= 3.0
                and metrics["shorter_span_overlap"] >= 0.60
            ):
                parallels.append(metrics)
        if len(parallels) < 2:
            continue
        nearest_offset = min(item["transverse_offset_mm"] for item in parallels)
        if nearest_offset < 10.0:
            continue
        options.append((candidate, nearest_offset, len(parallels)))
    if not options:
        return {
            "selected": output,
            "added": [],
            "activated": False,
            "reason": "no_separated_maximum_rank_consensus",
            "evidence": [],
        }
    options.sort(
        key=lambda item: (
            -float(item[0].get("cluster_max_two_sided_transition_score") or 0.0),
            str(item[0].get("candidate_id") or ""),
        )
    )
    candidate, nearest_offset, parallel_count = options[0]
    candidate["candidate_source"] = "residual_long_consensus_completion"
    candidate["completion_evidence"] = {
        "nearest_parallel_offset_mm": nearest_offset,
        "parallel_selected_count": parallel_count,
    }
    output.append(candidate)
    return {
        "selected": output,
        "added": [candidate],
        "activated": True,
        "reason": "completed",
        "evidence": [
            {
                "candidate_id": candidate.get("candidate_id"),
                **candidate["completion_evidence"],
            }
        ],
    }

