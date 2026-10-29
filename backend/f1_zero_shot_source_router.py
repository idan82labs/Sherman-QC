"""Geometry-only routing between patch-graph and zero-shot bend-line sources."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def route_f1_zero_shot_sources(
    patch_graph_selected: Sequence[Mapping[str, Any]],
    zero_shot_selected: Sequence[Mapping[str, Any]],
    zero_shot_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose one complete source without part identity, truth, or a count oracle.

    Plane-lattice and rolled/mixed profiles have scan-supported finite axes that
    are preferable to fragmented patch candidates.  Formed-transition output
    is accepted only for sparse candidate pools, where the patch graph is
    demonstrably under-seeded.  Dense formed-transition pools fail closed to
    patch graph because broad wall components otherwise generate false axes.
    """

    patch = [dict(item) for item in patch_graph_selected]
    zero = [dict(item) for item in zero_shot_selected]
    profile = str(zero_shot_result.get("profile_family") or "unknown")
    summary = dict(zero_shot_result.get("candidate_generator_summary") or {})
    union_count = int(summary.get("union_candidate_count") or 0)
    aspect_ratios = [float(value) for value in summary.get("aspect_ratios") or ()]
    maximum_aspect = max(aspect_ratios, default=0.0)

    selected_source = "patch_graph"
    reason = "default_fail_closed_to_patch_graph"
    if not patch and zero:
        selected_source = "zero_shot"
        reason = "patch_graph_empty"
    elif profile == "prismatic_plane_lattice_completion" and zero:
        # A lattice can be internally coherent while selecting the wrong
        # thickness lane.  Replacement is safe only when the conservative
        # zero-shot policy already abstained and both independent sources
        # agree on object cardinality.  Otherwise preserve patch geometry.
        if bool(zero_shot_result.get("abstained")) and len(patch) == len(zero):
            selected_source = "zero_shot"
            reason = "abstained_prismatic_lattice_with_source_count_consensus"
        else:
            reason = "prismatic_lattice_without_safe_source_consensus"
    elif profile == "rolled_or_mixed_transition" and zero:
        selected_source = "zero_shot"
        reason = "rolled_or_mixed_finite_scan_support"
    elif profile == "interior_formed_transition_completion" and zero:
        sparse = union_count <= 2 or (union_count <= 4 and maximum_aspect >= 3.0)
        if sparse:
            selected_source = "zero_shot"
            reason = "sparse_pool_formed_transition_recovery"
        else:
            reason = "dense_pool_formed_transition_fail_closed"

    selected_all = zero if selected_source == "zero_shot" else patch
    formed_feature_selected = []
    if selected_source == "zero_shot":
        formed_feature_selected = [
            item
            for item in selected_all
            if str(item.get("strip_family") or "") == "interior_formed_transition_strip"
        ]
    formed_ids = {id(item) for item in formed_feature_selected}
    selected = [item for item in selected_all if id(item) not in formed_ids]
    return {
        "route": selected_source,
        "reason": reason,
        "profile_family": profile,
        "union_candidate_count": union_count,
        "maximum_aspect_ratio": maximum_aspect,
        "patch_graph_count": len(patch),
        "zero_shot_count": len(zero),
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
        "selected": selected,
        "formed_feature_selected": formed_feature_selected,
    }
