"""Replace a weak overextended mesh line with a compact consensus fragment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def replace_overextended_mesh_fragments(
    selected: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use strong nested cylindrical evidence to crop low-probability mesh lines."""

    output = [dict(item) for item in selected]
    replacements: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for mesh_line in list(output):
        mesh_span = float(mesh_line.get("span_mm") or 0.0)
        if (
            mesh_line.get("candidate_source") != "scan_mesh_adjacency"
            or float(mesh_line.get("selection_probability") or 1.0) > 0.40
            or not (100.0 <= mesh_span <= 180.0)
        ):
            continue
        options = []
        for record in line_objects:
            candidate = dict(record)
            candidate_span = float(candidate.get("span_mm") or 0.0)
            if (
                candidate.get("representative_source")
                not in {"cylindrical_standard", "cylindrical_wide"}
                or not (50.0 <= candidate_span <= 100.0)
                or int(candidate.get("cluster_source_count") or 0) < 2
                or float(candidate.get("cluster_max_two_sided_transition_score") or 0.0) < 0.95
                or float(candidate.get("case_percentile_score") or 0.0) < 0.95
                or float(candidate.get("finite_support_fraction") or 0.0) < 0.90
                or float(candidate.get("endpoint_support_fraction") or 0.0) < 0.80
            ):
                continue
            metrics = parallel_line_metrics(mesh_line, candidate)
            if (
                metrics["axis_angle_deg"] <= 8.0
                and metrics["transverse_offset_mm"] <= 4.0
                and metrics["shorter_span_overlap"] >= 0.90
            ):
                options.append((candidate, metrics))
        if not options:
            continue
        options.sort(
            key=lambda item: (
                -float(item[0].get("cluster_max_two_sided_transition_score") or 0.0),
                -float(item[0].get("case_percentile_score") or 0.0),
                str(item[0].get("candidate_id") or ""),
            )
        )
        replacement, metrics = options[0]
        output.remove(mesh_line)
        replacement["candidate_source"] = "overextended_mesh_fragment_replacement"
        replacement["replaced_candidate_id"] = mesh_line.get("candidate_id")
        output.append(replacement)
        removed.append(mesh_line)
        replacements.append(replacement)
        evidence.append(
            {
                "removed_candidate_id": mesh_line.get("candidate_id"),
                "replacement_candidate_id": replacement.get("candidate_id"),
                **metrics,
            }
        )
    return {
        "selected": output,
        "removed": removed,
        "added": replacements,
        "activated": bool(replacements),
        "evidence": evidence,
    }

