"""Complete cylindrical lanes at a mixed prismatic junction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _duplicates, _endpoints
from backend.f1_parallel_width_lattice_completion import _metrics, _quality


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def _same_lane(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        geometry = _metrics(left, right)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return geometry["axis_angle_deg"] <= 8.0 and geometry["transverse_offset_mm"] <= 7.0 and geometry["shorter_span_overlap"] >= 0.80


def complete_prismatic_junction(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = [dict(item) for item in selected]
    if str(source_route.get("profile_family") or "") != "prismatic_plane_lattice_completion":
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "not_prismatic_lattice_profile"}
    strips = {str(item.get("strip_id") or ""): item for item in support_strips}
    selected_strips = [item for item in output if str(item.get("bend_id") or "") in strips]
    derived = [item for item in output if item not in selected_strips]
    if len(selected_strips) != 2 or len(derived) != 1 or str(derived[0].get("candidate_source") or "") != "interface_birth_consensus":
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "not_two_strip_one_interface_junction"}
    try:
        strip_geometry = _metrics(selected_strips[0], selected_strips[1])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "invalid_strip_junction_geometry"}
    if strip_geometry["axis_angle_deg"] < 70.0:
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "selected_strips_not_orthogonal"}
    anchor = min(selected_strips, key=_span)
    anchor_endpoints = _endpoints(anchor)
    candidates = []
    for item in line_objects:
        if list(item.get("cluster_sources") or []) != ["cylindrical_standard", "cylindrical_wide"]:
            continue
        if int(item.get("cluster_source_count") or 0) < 2:
            continue
        if float(item.get("cluster_max_finite_support_longest_run_fraction") or 0.0) < 0.80:
            continue
        if float(item.get("cluster_max_endpoint_support_fraction") or 0.0) < 0.80:
            continue
        if float(item.get("cluster_max_two_sided_transition_score") or 0.0) < 0.45:
            continue
        if float(item.get("tight_to_wide_normal_second_ratio") or 0.0) > 0.30:
            continue
        try:
            endpoints = _endpoints(item)
            span = _span(item)
        except (KeyError, TypeError, ValueError):
            continue
        endpoint_distance = min(float(np.linalg.norm(left - right)) for left in endpoints for right in anchor_endpoints)
        if not 35.0 <= span <= 120.0 or endpoint_distance > 4.0 or any(_duplicates(item, retained) for retained in output):
            continue
        candidates.append(dict(item))
    lanes = []
    for candidate in sorted(
        candidates,
        key=lambda item: (float(item.get("tight_to_wide_normal_second_ratio") or 0.0), -_quality(item), str(item.get("bend_id") or "")),
    ):
        if any(_same_lane(candidate, retained) for retained in lanes):
            continue
        lanes.append(candidate)
    if len(lanes) != 2:
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "endpoint_connected_cylindrical_lane_pair_not_unique"}
    promoted = []
    for index, item in enumerate(lanes, start=1):
        item["bend_id"] = f"PJC{index}"
        item["candidate_source"] = "prismatic_junction_completion"
        item["selection_reason"] = "unique_endpoint_connected_tight_normal_cylindrical_lane_pair"
        promoted.append(item)
    return {"selected": output + promoted, "promoted": promoted, "activated": True, "activation_reason": "mixed_prismatic_junction_contract"}
