"""Complete a two-rail patch-graph frame from unique supported endpoint connectors."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints
from backend.f1_parallel_width_lattice_completion import _metrics


def _span(item: Mapping[str, Any]) -> float:
    endpoints = _endpoints(item)
    return float(np.linalg.norm(endpoints[1] - endpoints[0]))


def complete_patch_graph_endpoint_frame(
    selected: Sequence[Mapping[str, Any]],
    source_route: Mapping[str, Any],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace short rail fragments only when two orthogonal connectors are unique."""

    output = [dict(item) for item in selected]
    if str(source_route.get("route") or "") != "patch_graph" or len(output) < 3:
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "not_patch_graph_fragmented_frame"}
    try:
        spans = {str(item.get("bend_id") or ""): _span(item) for item in output}
    except (KeyError, TypeError, ValueError):
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "invalid_selected_geometry"}
    maximum = max(spans.values(), default=0.0)
    dominant = [item for item in output if spans[str(item.get("bend_id") or "")] >= 0.70 * maximum]
    residuals = [item for item in output if item not in dominant]
    if len(dominant) != 2 or not residuals or any(spans[str(item.get("bend_id") or "")] > 0.40 * maximum for item in residuals):
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "not_two_dominant_rails_with_short_residuals"}
    try:
        rail_geometry = _metrics(dominant[0], dominant[1])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "invalid_rail_geometry"}
    if rail_geometry["axis_angle_deg"] > 3.0:
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "dominant_rails_not_parallel"}
    connectors = []
    for item in line_objects:
        if str(item.get("representative_source") or "") != "endpoint_connector":
            continue
        if float(item.get("cluster_max_finite_support_longest_run_fraction") or 0.0) < 0.80:
            continue
        if float(item.get("cluster_max_endpoint_support_fraction") or 0.0) < 0.80:
            continue
        try:
            geometry = _metrics(dominant[0], item)
            span = _span(item)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if geometry["axis_angle_deg"] < 70.0 or span < 1.20 * maximum:
            continue
        connectors.append(dict(item))
    if len(connectors) != 2:
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "supported_endpoint_connector_pair_not_unique"}
    try:
        connector_geometry = _metrics(connectors[0], connectors[1])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "invalid_connector_geometry"}
    if connector_geometry["axis_angle_deg"] > 3.0:
        return {"selected": output, "promoted": [], "pruned": [], "activated": False, "activation_reason": "endpoint_connectors_not_parallel"}
    promoted = []
    for index, item in enumerate(connectors, start=1):
        item["bend_id"] = f"PFC{index}"
        item["candidate_source"] = "patch_graph_endpoint_frame_completion"
        item["selection_reason"] = "unique_supported_orthogonal_endpoint_connector_pair"
        promoted.append(item)
    pruned = []
    for item in residuals:
        removed = dict(item)
        removed["pruning_reason"] = "short_same_axis_fragment_replaced_by_supported_frame_connector"
        pruned.append(removed)
    return {"selected": [dict(item) for item in dominant] + promoted, "promoted": promoted, "pruned": pruned, "activated": True, "activation_reason": "two_rail_supported_frame_contract"}
