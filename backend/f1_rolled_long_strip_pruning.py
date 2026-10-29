"""Prune long strip extrapolations on rolled or mixed surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints


def prune_rolled_long_strips(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    support_strips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = [dict(item) for item in selected]
    if str(source_route.get("profile_family") or "") != "rolled_or_mixed_transition":
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "not_rolled_or_mixed_transition"}
    strips = {str(item.get("strip_id") or ""): item for item in support_strips}
    retained = []
    pruned = []
    for item in output:
        strip = strips.get(str(item.get("bend_id") or ""))
        if strip is None or str(strip.get("strip_family") or "") != "discrete_fold_strip":
            retained.append(item)
            continue
        try:
            endpoints = _endpoints(item)
            span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            width = float(strip.get("cross_fold_width_mm") or 0.0)
        except (KeyError, TypeError, ValueError):
            retained.append(item)
            continue
        aspect_ratio = span / max(width, 1e-9)
        if aspect_ratio > 10.0:
            removed = dict(item)
            removed["pruning_reason"] = "rolled_surface_strip_span_exceeds_ten_support_widths"
            removed["span_to_support_width_ratio"] = aspect_ratio
            pruned.append(removed)
        else:
            retained.append(item)
    return {"selected": retained, "pruned": pruned, "activated": True, "activation_reason": "rolled_strip_finite_span_contract"}
