"""Count-free stitching of selected fragments from one physical bend axis."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from scripts.evaluate_f1_against_manual_gold import pair_metrics
from research.f1_multifamily_selector_v2.model import endpoints


def _interval(line: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = line @ axis
    return float(values.min()), float(values.max())


def _axial_gap(first: np.ndarray, second: np.ndarray) -> float:
    first_axis = first[1] - first[0]
    second_axis = second[1] - second[0]
    if float(np.dot(first_axis, second_axis)) < 0.0:
        second_axis = -second_axis
    axis = first_axis / max(float(np.linalg.norm(first_axis)), 1e-12)
    other = second_axis / max(float(np.linalg.norm(second_axis)), 1e-12)
    axis = axis + other
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    a0, a1 = _interval(first, axis)
    b0, b1 = _interval(second, axis)
    return max(0.0, max(a0, b0) - min(a1, b1))


def _connected_components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = sorted(adjacency[current].intersection(unseen), reverse=True)
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _fit_component(rows: list[dict], indices: list[int]) -> dict:
    points = np.vstack([endpoints(rows[index]) for index in indices])
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis = vectors[0]
    reference = endpoints(rows[indices[0]])[1] - endpoints(rows[indices[0]])[0]
    if float(np.dot(axis, reference)) < 0.0:
        axis = -axis
    projections = (points - center) @ axis
    fitted = np.vstack(
        (center + projections.min() * axis, center + projections.max() * axis)
    )
    member_lengths = [float(np.linalg.norm(endpoints(rows[index])[1] - endpoints(rows[index])[0])) for index in indices]
    representative_index = indices[int(np.argmax(member_lengths))]
    result = deepcopy(rows[representative_index])
    result["span_endpoints"] = fitted.tolist()
    result["axis_direction"] = axis.tolist()
    result["span_mm"] = float(np.linalg.norm(fitted[1] - fitted[0]))
    member_ids = sorted(str(rows[index].get("candidate_id") or "") for index in indices)
    result["candidate_id"] = "collinear_fragment_stitch::" + "+".join(member_ids)
    result["candidate_source"] = "collinear_fragment_stitch"
    result["representative_source"] = "collinear_fragment_stitch"
    result["stitched_member_candidate_ids"] = member_ids
    return result


def stitch_collinear_selected_fragments(
    selected: list[dict],
    *,
    maximum_axis_angle_deg: float = 2.0,
    maximum_transverse_offset_mm: float = 6.0,
    maximum_axial_gap_mm: float = 20.0,
    minimum_span_gain: float = 1.25,
) -> dict:
    """Replace end-to-end selected fragments with one fitted physical axis.

    Overlapping alternative lanes do not qualify unless their union materially
    extends beyond every member.  No unselected candidate can be added.
    """

    if len(selected) < 2:
        return {"selected": list(selected), "changes": []}
    adjacency = [set() for _ in selected]
    for first in range(len(selected)):
        first_line = endpoints(selected[first])
        for second in range(first):
            second_line = endpoints(selected[second])
            metrics = pair_metrics(first_line, second_line)
            if (
                float(metrics["axis_angle_deg"]) <= maximum_axis_angle_deg
                and float(metrics["transverse_offset_mm"])
                <= maximum_transverse_offset_mm
                and _axial_gap(first_line, second_line) <= maximum_axial_gap_mm
            ):
                adjacency[first].add(second)
                adjacency[second].add(first)

    replacements: dict[int, tuple[list[int], dict]] = {}
    changes: list[dict] = []
    for component in _connected_components(adjacency):
        if len(component) < 2:
            continue
        fitted = _fit_component(selected, component)
        maximum_member_span = max(
            float(np.linalg.norm(endpoints(selected[index])[1] - endpoints(selected[index])[0]))
            for index in component
        )
        if float(fitted["span_mm"]) < minimum_span_gain * maximum_member_span:
            continue
        root = min(component)
        replacements[root] = (component, fitted)
        changes.append(
            {
                "member_candidate_ids": list(fitted["stitched_member_candidate_ids"]),
                "fitted_candidate_id": str(fitted["candidate_id"]),
                "member_count": len(component),
                "maximum_member_span_mm": round(maximum_member_span, 6),
                "fitted_span_mm": round(float(fitted["span_mm"]), 6),
            }
        )

    consumed = {index for component, _ in replacements.values() for index in component}
    output: list[dict] = []
    for index, row in enumerate(selected):
        if index in replacements:
            output.append(replacements[index][1])
        elif index not in consumed:
            output.append(row)
    return {"selected": output, "changes": changes}
