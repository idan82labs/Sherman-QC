"""Complete a partial orthogonal lattice from two strips and four finite rails."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints
from backend.f1_parallel_width_lattice_completion import _metrics, _quality


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def _strip_line(strip: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    endpoints = strip.get("axis_line_hint", {}).get("supported_endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("support strip has no finite endpoints")
    return {
        "bend_id": identifier,
        "span_endpoints": endpoints,
        "source_kind": "zero_shot_support_strip",
        "strip_family": strip.get("strip_family"),
        "candidate_source": "partial_lattice_completion",
        "selection_reason": "orthogonal_prismatic_support_strip_completion",
    }


def complete_partial_lattice(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = [dict(item) for item in selected]
    strips = [dict(item) for item in support_strips if str(item.get("strip_family") or "") == "prismatic_plane_lattice_long"]
    if str(source_route.get("route") or "") != "patch_graph" or str(source_route.get("profile_family") or "") != "prismatic_plane_lattice_completion":
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "not_patch_graph_prismatic_partial_lattice"}
    if len(strips) != 2:
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "prismatic_strip_pair_not_unique"}
    try:
        strip_lines = [_strip_line(strip, f"PLC_STRIP{index}") for index, strip in enumerate(strips, start=1)]
        strip_geometry = _metrics(strip_lines[0], strip_lines[1])
        spans = {str(item.get("bend_id") or ""): _span(item) for item in output}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "invalid_partial_lattice_geometry"}
    if strip_geometry["axis_angle_deg"] > 3.0:
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "support_strips_not_parallel"}
    maximum = max(spans.values(), default=0.0)
    rails = [item for item in output if spans[str(item.get("bend_id") or "")] >= 0.60 * maximum]
    residuals = [item for item in output if item not in rails]
    if len(rails) != 4 or not residuals:
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "not_four_finite_rails_with_short_residuals"}
    try:
        if min(_metrics(strip_lines[0], rail)["axis_angle_deg"] for rail in rails) < 70.0:
            return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "support_strips_not_orthogonal_to_rails"}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "invalid_rail_orientation"}
    target = min(rails, key=_span)
    candidates = []
    for item in line_objects:
        if list(item.get("cluster_sources") or []) != ["cylindrical_standard", "cylindrical_wide"]:
            continue
        if float(item.get("case_percentile_score") or 0.0) < 0.92 or float(item.get("case_percentile_support_count") or 0.0) < 0.92:
            continue
        if float(item.get("tight_to_wide_normal_second_ratio") or 0.0) > 0.10:
            continue
        try:
            geometry = _metrics(target, item)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if geometry["axis_angle_deg"] <= 3.0 and geometry["transverse_offset_mm"] <= 8.0 and geometry["shorter_span_overlap"] >= 0.80:
            candidates.append(dict(item))
    if not candidates:
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "no_dense_refit_for_shortest_rail"}
    best = max(candidates, key=lambda item: (_quality(item), str(item.get("bend_id") or "")))
    if any(_metrics(best, candidate)["transverse_offset_mm"] > 4.0 for candidate in candidates):
        return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "dense_refit_lane_not_unique"}
    replacement = dict(best)
    replacement["bend_id"] = str(target.get("bend_id") or "PLC_RAIL")
    replacement["candidate_source"] = "partial_lattice_dense_refit"
    replacement["selection_reason"] = "unique_high_percentile_zero_tight_normal_rail_refit"
    retained = [replacement if item is target else dict(item) for item in rails]
    refitted_strips = []
    strip_refits = []
    for strip, strip_line in zip(strips, strip_lines):
        width = float(strip.get("cross_fold_width_mm") or 0.0)
        candidates = []
        for item in line_objects:
            if list(item.get("cluster_sources") or []) != ["cylindrical_standard", "cylindrical_wide"]:
                continue
            if float(item.get("case_percentile_score") or 0.0) < 0.95 or float(item.get("case_percentile_support_count") or 0.0) < 0.95:
                continue
            try:
                geometry = _metrics(strip_line, item)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            ratio = geometry["transverse_offset_mm"] / max(width, 1e-9)
            if geometry["axis_angle_deg"] <= 3.0 and geometry["shorter_span_overlap"] >= 0.80 and 0.35 <= ratio <= 0.65:
                candidates.append((abs(ratio - 0.50), -_quality(item), str(item.get("bend_id") or ""), dict(item), ratio))
        if not candidates:
            return {"selected": output, "promoted": [], "pruned": [], "refit": [], "activated": False, "activation_reason": "no_half_width_dense_strip_refit"}
        chosen = min(candidates)
        strip_replacement = chosen[3]
        strip_replacement["bend_id"] = str(strip_line["bend_id"])
        strip_replacement["candidate_source"] = "partial_lattice_half_width_strip_refit"
        strip_replacement["selection_reason"] = "dense_cylindrical_lane_nearest_half_cross_fold_width"
        strip_replacement["offset_to_cross_fold_width_ratio"] = chosen[4]
        refitted_strips.append(strip_replacement)
        strip_refits.append({
            "removed_bend_id": strip_line["bend_id"],
            "replacement_bend_id": strip_replacement["bend_id"],
            "replacement_candidate_id": strip_replacement.get("candidate_id"),
            "offset_to_cross_fold_width_ratio": chosen[4],
        })
    pruned = []
    for item in residuals:
        removed = dict(item)
        removed["pruning_reason"] = "short_partial_lattice_residual"
        pruned.append(removed)
    return {
        "selected": retained + refitted_strips,
        "promoted": refitted_strips,
        "pruned": pruned,
        "refit": [{"removed_bend_id": target.get("bend_id"), "replacement_bend_id": replacement["bend_id"], "replacement_candidate_id": best.get("candidate_id")}] + strip_refits,
        "activated": True,
        "activation_reason": "two_strip_four_rail_partial_lattice_contract",
    }
