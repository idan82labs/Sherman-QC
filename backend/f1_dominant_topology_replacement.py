"""Replace uncertain baseline lanes with a complete geometry-derived feature graph."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_DERIVED_SOURCES = {
    "interface_birth_stitch",
    "transition_boundary_refit",
    "axial_support_split",
}


def replace_complete_dominant_topology(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use only derived finite axes when their topology is complete and unambiguous."""

    source = [dict(item) for item in selected]
    derived = [item for item in source if str(item.get("candidate_source") or "") in _DERIVED_SOURCES]
    counts = {
        key: sum(str(item.get("candidate_source") or "") == key for item in derived)
        for key in sorted(_DERIVED_SOURCES)
    }
    stitch_pairs = {
        tuple(sorted(str(value) for value in item.get("canonical_flange_pair") or ()))
        for item in derived
        if str(item.get("candidate_source") or "") == "interface_birth_stitch"
    }
    complete = (
        counts["transition_boundary_refit"] == 1
        and counts["axial_support_split"] == 2
        and counts["interface_birth_stitch"] == 2
        and len(stitch_pairs) == 2
        and all(item.get("span_endpoints") is not None for item in derived)
    )
    if not complete:
        return {
            "selected": source,
            "applied": False,
            "removed_bend_ids": [],
            "derived_source_counts": counts,
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
        }
    removed = [
        str(item.get("bend_id") or "")
        for item in source
        if str(item.get("candidate_source") or "") not in _DERIVED_SOURCES
    ]
    return {
        "selected": derived,
        "applied": True,
        "removed_bend_ids": sorted(removed),
        "derived_source_counts": counts,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
