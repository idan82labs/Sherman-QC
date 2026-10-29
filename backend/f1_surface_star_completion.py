"""Complete a strongly supported terminal interface on a surface star."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def _surface_pair(item: Mapping[str, Any]) -> tuple[Any, Any] | None:
    left = item.get("surface1_id")
    right = item.get("surface2_id")
    if left is None or right is None or left == right:
        return None
    return left, right


def _is_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    metrics = parallel_line_metrics(left, right)
    return (
        metrics["axis_angle_deg"] <= 7.5
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.45
    )


def complete_strong_surface_star_leaf(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add an isolated high-evidence leaf incident to an established hub.

    A selected three-edge (or larger) surface star establishes a physical face.
    A fourth interface is admitted only when it connects that hub to a new
    surface, has strong scan evidence, and is not a spatial duplicate of any
    already selected line.
    """

    output = [dict(item) for item in selected]
    selected_ids = {str(item.get("candidate_id") or "") for item in output}
    degree: dict[Any, int] = {}
    for item in output:
        pair = _surface_pair(item)
        if pair is None:
            continue
        for surface_id in pair:
            degree[surface_id] = degree.get(surface_id, 0) + 1

    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for candidate_record in candidates:
        candidate = dict(candidate_record)
        candidate_id = str(candidate.get("candidate_id") or "")
        pair = _surface_pair(candidate)
        if (
            pair is None
            or not candidate_id
            or candidate_id in selected_ids
            or candidate.get("candidate_source") != "scan_mesh_adjacency"
            or float(candidate.get("two_sided_transition_score") or 0.0) < 0.80
            or float(candidate.get("confidence") or 0.0) < 0.60
            or float(candidate.get("min_surface_gap") or 0.0) > 6.0
        ):
            continue
        hub_surfaces = [surface_id for surface_id in pair if degree.get(surface_id, 0) >= 3]
        new_surfaces = [surface_id for surface_id in pair if degree.get(surface_id, 0) == 0]
        if len(hub_surfaces) != 1 or len(new_surfaces) != 1:
            continue
        if any(_is_duplicate(candidate, existing) for existing in output):
            continue
        candidate["candidate_source"] = "strong_surface_star_leaf_completion"
        candidate["completion_hub_surface_id"] = hub_surfaces[0]
        candidate["completion_new_surface_id"] = new_surfaces[0]
        additions.append(candidate)
        evidence.append(
            {
                "candidate_id": candidate_id,
                "hub_surface_id": hub_surfaces[0],
                "prior_hub_degree": degree[hub_surfaces[0]],
                "new_surface_id": new_surfaces[0],
            }
        )
        selected_ids.add(candidate_id)
        degree[hub_surfaces[0]] += 1
        degree[new_surfaces[0]] = 1
        output.append(candidate)

    return {
        "selected": output,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }

