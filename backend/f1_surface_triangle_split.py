"""Replace a central mesh outline with two supported sides of a thin triangle."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.f1_parallel_return_doublet_completion import parallel_line_metrics


def _surface_pair(item: Mapping[str, Any]) -> tuple[Any, Any] | None:
    left = item.get("surface1_id")
    right = item.get("surface2_id")
    if left is None or right is None or left == right:
        return None
    return left, right


def _is_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    metrics = parallel_line_metrics(left, right)
    return (
        metrics["axis_angle_deg"] <= 7.5
        and metrics["transverse_offset_mm"] <= 4.0
        and metrics["shorter_span_overlap"] >= 0.45
    )


def split_supported_surface_triangle(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Split one selected edge when two near-parallel edges close its triangle."""

    output = [dict(item) for item in selected]
    raw = [
        dict(item)
        for item in candidates
        if item.get("candidate_source") == "scan_mesh_adjacency"
        and _surface_pair(item) is not None
    ]
    represented = [item for item in raw if any(_is_duplicate(item, keep) for keep in output)]
    unrepresented = [item for item in raw if not any(_is_duplicate(item, keep) for keep in output)]
    unrepresented_ids = {str(item.get("candidate_id") or "") for item in unrepresented}
    by_pair = {frozenset(_surface_pair(item)): item for item in raw}
    removals: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    consumed_ids: set[str] = set()

    for center in represented:
        center_id = str(center.get("candidate_id") or "")
        center_pair = _surface_pair(center)
        assert center_pair is not None
        center_span = max(float(center.get("span_mm") or 0.0), 1.0e-9)
        center_transition = float(center.get("two_sided_transition_score") or 0.0)
        if (
            center_id in consumed_ids
            or float(center.get("confidence") or 0.0) < 0.90
            or float(center.get("min_surface_gap") or 1.0e9) > 0.50
            or not (0.30 <= center_transition <= 0.75)
        ):
            continue
        surfaces = {
            surface_id
            for item in raw
            for surface_id in (_surface_pair(item) or ())
            if surface_id not in center_pair
        }
        for apex in surfaces:
            side_a = by_pair.get(frozenset((center_pair[0], apex)))
            side_b = by_pair.get(frozenset((center_pair[1], apex)))
            if (
                side_a is None
                or side_b is None
                or str(side_a.get("candidate_id") or "") not in unrepresented_ids
                or str(side_b.get("candidate_id") or "") not in unrepresented_ids
            ):
                continue
            sides = (side_a, side_b)
            metrics = [parallel_line_metrics(center, side) for side in sides]
            relative_spans = [
                abs(float(side.get("span_mm") or 0.0) - center_span) / center_span
                for side in sides
            ]
            if not (
                max(item["axis_angle_deg"] for item in metrics) <= 7.5
                and max(item["transverse_offset_mm"] for item in metrics) <= 15.0
                and min(item["shorter_span_overlap"] for item in metrics) >= 0.70
                and max(relative_spans) <= 0.10
                and min(float(side.get("confidence") or 0.0) for side in sides) >= 0.80
                and max(float(side.get("confidence") or 0.0) for side in sides) >= 0.98
                and min(float(side.get("min_surface_gap") or 1.0e9) for side in sides) <= 0.50
                and max(float(side.get("min_surface_gap") or 1.0e9) for side in sides) <= 12.0
                and max(float(side.get("two_sided_transition_score") or 0.0) for side in sides)
                < center_transition
            ):
                continue
            removed = [item for item in output if _is_duplicate(center, item)]
            output = [item for item in output if not _is_duplicate(center, item)]
            side_records = []
            for side in sides:
                item = dict(side)
                item["candidate_source"] = "surface_triangle_split"
                item["split_center_candidate_id"] = center_id
                side_records.append(item)
            output.extend(side_records)
            removals.extend(removed)
            additions.extend(side_records)
            evidence.append(
                {
                    "center_candidate_id": center_id,
                    "apex_surface_id": apex,
                    "side_candidate_ids": [side.get("candidate_id") for side in sides],
                    "removed_prediction_count": len(removed),
                }
            )
            consumed_ids.add(center_id)
            break

    return {
        "selected": output,
        "removed": removals,
        "added": additions,
        "activated": bool(additions),
        "evidence": evidence,
    }

