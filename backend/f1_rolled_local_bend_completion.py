"""Recover stable local bend lanes around one retained rolled-surface strip."""

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


def complete_rolled_local_bends(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = [dict(item) for item in selected]
    if str(source_route.get("profile_family") or "") != "rolled_or_mixed_transition":
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "not_rolled_or_mixed_transition"}
    strips = {str(item.get("strip_id") or ""): item for item in support_strips}
    retained_strips = [item for item in output if str(item.get("bend_id") or "") in strips]
    if len(output) != 1 or len(retained_strips) != 1:
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "not_single_retained_local_strip"}
    anchor = retained_strips[0]
    anchor_midpoint = np.mean(_endpoints(anchor), axis=0)
    candidates = []
    for item in line_objects:
        if int(item.get("cluster_source_count") or 0) < 2:
            continue
        if float(item.get("cluster_max_finite_support_longest_run_fraction") or 0.0) < 0.80:
            continue
        if float(item.get("cluster_max_endpoint_support_fraction") or 0.0) < 0.80:
            continue
        if float(item.get("cluster_max_two_sided_transition_score") or 0.0) < 0.55:
            continue
        try:
            span = _span(item)
            midpoint = np.mean(_endpoints(item), axis=0)
            geometry = _metrics(anchor, item)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not 15.0 <= span <= 40.0 or float(np.linalg.norm(midpoint - anchor_midpoint)) > 50.0:
            continue
        if geometry["axis_angle_deg"] > 12.0 or any(_duplicates(item, retained) for retained in output):
            continue
        candidates.append(dict(item))
    lanes = []
    for candidate in sorted(candidates, key=lambda item: (-_quality(item), str(item.get("bend_id") or ""))):
        if any(_same_lane(candidate, retained) for retained in lanes):
            continue
        lanes.append(candidate)
    if len(lanes) != 2:
        return {"selected": output, "promoted": [], "activated": False, "activation_reason": "stable_local_lane_pair_not_unique"}
    promoted = []
    for index, item in enumerate(lanes, start=1):
        item["bend_id"] = f"RLC{index}"
        item["candidate_source"] = "rolled_local_bend_completion"
        item["selection_reason"] = "unique_stable_local_rolled_bend_lane_pair"
        promoted.append(item)
    return {"selected": output + promoted, "promoted": promoted, "activated": True, "activation_reason": "single_strip_local_lane_pair_contract"}
