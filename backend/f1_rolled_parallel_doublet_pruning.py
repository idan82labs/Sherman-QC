"""Prune ambiguous one-width parallel doublets on rolled scan surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_width_lattice_completion import _metrics


def prune_rolled_parallel_doublet(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed when a rolled route contains one isolated strip-width doublet."""

    output = [dict(item) for item in selected]
    if str(source_route.get("profile_family") or "") != "rolled_or_mixed_transition":
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "not_rolled_or_mixed_transition"}
    strips = {str(item.get("strip_id") or ""): item for item in support_strips}
    pairs = []
    for index, left in enumerate(output):
        left_strip = strips.get(str(left.get("bend_id") or ""))
        if left_strip is None or str(left_strip.get("strip_family") or "") != "discrete_fold_strip":
            continue
        for right in output[index + 1 :]:
            right_strip = strips.get(str(right.get("bend_id") or ""))
            if right_strip is None or str(right_strip.get("strip_family") or "") != "discrete_fold_strip":
                continue
            try:
                geometry = _metrics(left, right)
                width = 0.5 * (
                    float(left_strip.get("cross_fold_width_mm") or 0.0)
                    + float(right_strip.get("cross_fold_width_mm") or 0.0)
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            width_ratio = geometry["transverse_offset_mm"] / max(width, 1e-9)
            if (
                geometry["axis_angle_deg"] <= 4.0
                and geometry["shorter_span_overlap"] >= 0.80
                and 0.80 <= width_ratio <= 1.50
            ):
                pairs.append((str(left.get("bend_id") or ""), str(right.get("bend_id") or ""), geometry, width_ratio))
    if len(pairs) != 1:
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "parallel_doublet_not_unique"}
    left_id, right_id, geometry, width_ratio = pairs[0]
    pair_ids = {left_id, right_id}
    pruned = []
    retained = []
    for item in output:
        if str(item.get("bend_id") or "") in pair_ids:
            removed = dict(item)
            removed["pruning_reason"] = "unique_rolled_surface_parallel_strip_width_doublet"
            removed["doublet_axis_angle_deg"] = geometry["axis_angle_deg"]
            removed["doublet_offset_to_support_width_ratio"] = width_ratio
            pruned.append(removed)
        else:
            retained.append(item)
    return {"selected": retained, "pruned": pruned, "activated": True, "activation_reason": "unique_parallel_doublet_contract"}
