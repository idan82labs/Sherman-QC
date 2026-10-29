"""Complete the opposite member of a prismatic medium-span bend pair."""

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


def complete_opposite_fragment_pair(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add one strong, distant parallel mate for each cropped mesh fragment."""

    output = [dict(item) for item in selected]
    records: dict[str, dict[str, Any]] = {}
    for record in candidates:
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id:
            records[candidate_id] = dict(record)
    for record in line_objects:
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id:
            records[candidate_id] = dict(record)

    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for seed in output:
        if seed.get("candidate_source") != "overextended_mesh_fragment_replacement":
            continue
        seed_span = max(float(seed.get("span_mm") or 0.0), 1.0e-9)
        options = []
        for candidate in records.values():
            candidate_span = float(candidate.get("span_mm") or 0.0)
            transition = float(
                candidate.get("cluster_max_two_sided_transition_score")
                or candidate.get("two_sided_transition_score")
                or 0.0
            )
            if (
                not (50.0 <= candidate_span <= 110.0)
                or transition < 0.85
                or float(candidate.get("finite_support_fraction") or 0.0) < 0.90
                or float(candidate.get("endpoint_support_fraction") or 0.0) < 0.80
                or any(_is_duplicate(candidate, item) for item in output + additions)
            ):
                continue
            metrics = parallel_line_metrics(seed, candidate)
            relative_span_difference = abs(seed_span - candidate_span) / max(
                seed_span, candidate_span
            )
            if not (
                metrics["axis_angle_deg"] <= 3.0
                and metrics["transverse_offset_mm"] >= 100.0
                and metrics["shorter_span_overlap"] >= 0.60
                and relative_span_difference <= 0.40
            ):
                continue
            options.append((candidate, metrics, relative_span_difference, transition))
        if not options:
            continue
        options.sort(
            key=lambda item: (
                -item[3],
                -int(item[0].get("cluster_source_count") or 0),
                -float(item[0].get("case_percentile_score") or 0.0),
                str(item[0].get("candidate_id") or ""),
            )
        )
        candidate, metrics, relative_span_difference, _transition = options[0]
        candidate["candidate_source"] = "opposite_fragment_pair_completion"
        candidate["completion_seed_id"] = seed.get("candidate_id")
        candidate["completion_evidence"] = {
            **metrics,
            "relative_span_difference": relative_span_difference,
        }
        additions.append(candidate)
        evidence.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "seed_id": seed.get("candidate_id"),
                **candidate["completion_evidence"],
            }
        )
    output.extend(additions)
    return {
        "selected": output,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }

