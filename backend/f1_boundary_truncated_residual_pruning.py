"""Prune weak, short patch-graph closures at truncated scan boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def prune_boundary_truncated_residuals(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only jointly short and weak boundary closures on patch-graph routes."""

    output = [dict(item) for item in selected]
    if str(source_route.get("route") or "") != "patch_graph":
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "not_patch_graph_route"}
    internal = [item for item in output if str(item.get("visibility_state") or "") == "internal"]
    if len(internal) < 2:
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "insufficient_internal_reference_objects"}
    try:
        dominant_span = max(_span(item) for item in internal)
        dominant_mass = max(float(item.get("support_mass") or 0.0) for item in internal)
    except (KeyError, TypeError, ValueError):
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "invalid_reference_geometry"}
    if dominant_span <= 0.0 or dominant_mass <= 0.0:
        return {"selected": output, "pruned": [], "activated": False, "activation_reason": "invalid_reference_support"}
    pruned = []
    retained = []
    for item in output:
        try:
            span_ratio = _span(item) / dominant_span
        except (KeyError, TypeError, ValueError):
            retained.append(item)
            continue
        mass_ratio = float(item.get("support_mass") or 0.0) / dominant_mass
        if (
            str(item.get("visibility_state") or "") == "boundary_truncated"
            and span_ratio < 0.20
            and mass_ratio < 0.20
        ):
            removed = dict(item)
            removed["pruning_reason"] = "short_weak_boundary_truncated_patch_graph_residual"
            removed["span_to_dominant_ratio"] = span_ratio
            removed["support_mass_to_dominant_ratio"] = mass_ratio
            pruned.append(removed)
        else:
            retained.append(item)
    return {"selected": retained, "pruned": pruned, "activated": True, "activation_reason": "patch_graph_relative_support_contract"}
