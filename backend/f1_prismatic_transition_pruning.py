"""Prune prismatic strips lacking a strong matched two-sided scan transition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_width_lattice_completion import _metrics


def _matched_transition(item: Mapping[str, Any], line_objects: Sequence[Mapping[str, Any]]) -> float | None:
    matches = []
    for line_object in line_objects:
        try:
            geometry = _metrics(item, line_object)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if geometry["axis_angle_deg"] > 3.0 or geometry["transverse_offset_mm"] > 3.0 or geometry["shorter_span_overlap"] < 0.80:
            continue
        matches.append((geometry["transverse_offset_mm"], geometry["axis_angle_deg"], -geometry["shorter_span_overlap"], line_object))
    if not matches:
        return None
    line_object = min(matches)[3]
    return float(line_object.get("cluster_max_two_sided_transition_score") or 0.0)


def prune_weak_prismatic_transitions(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = [dict(item) for item in selected]
    strips = {str(item.get("strip_id") or ""): item for item in support_strips}
    if not any(str(item.get("strip_family") or "") == "prismatic_plane_lattice_long" for item in strips.values()):
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "no_prismatic_lattice_strips"}
    retained = []
    pruned = []
    for item in output:
        strip = strips.get(str(item.get("bend_id") or ""))
        if strip is None or str(strip.get("strip_family") or "") != "prismatic_plane_lattice_long":
            retained.append(item)
            continue
        transition = _matched_transition(item, line_objects)
        if transition is not None and transition < 0.65:
            removed = dict(item)
            removed["pruning_reason"] = "matched_two_sided_transition_below_0_65"
            removed["matched_two_sided_transition_score"] = transition
            pruned.append(removed)
        else:
            retained.append(item)
    return {"selected": retained, "pruned": pruned, "activated": True, "activation_reason": "prismatic_matched_transition_contract"}
