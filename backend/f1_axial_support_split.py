"""Split an overextended transition hypothesis at a central atom-support deficit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _duplicates, _endpoints


def _same_uncertain_lane(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    first = _endpoints(left)
    second = _endpoints(right)
    first_delta = first[1] - first[0]
    second_delta = second[1] - second[0]
    first_span = float(np.linalg.norm(first_delta))
    second_span = float(np.linalg.norm(second_delta))
    if min(first_span, second_span) <= 1e-9:
        return False
    first_axis = first_delta / first_span
    second_axis = second_delta / second_span
    angle = np.degrees(np.arccos(float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))))
    delta = np.mean(second, axis=0) - np.mean(first, axis=0)
    offset = max(
        float(np.linalg.norm(delta - first_axis * np.dot(delta, first_axis))),
        float(np.linalg.norm(delta - second_axis * np.dot(delta, second_axis))),
    )
    interval = sorted(float(np.dot(point - np.mean(first, axis=0), first_axis)) for point in second)
    overlap = max(0.0, min(0.5 * first_span, interval[1]) - max(-0.5 * first_span, interval[0]))
    return angle <= 4.0 and offset <= 5.0 and overlap / min(first_span, second_span) >= 0.80


def promote_axial_support_lobes(
    selected: Sequence[Mapping[str, Any]],
    bend_hypotheses: Sequence[Mapping[str, Any]],
    atom_graph: Mapping[str, Any],
    *,
    dominant_span_minimum_mm: float = 200.0,
    dominant_span_ratio: float = 2.0,
    split_span_minimum_mm: float = 100.0,
    split_span_maximum_mm: float = 250.0,
    minimum_confidence: float = 0.75,
    maximum_central_to_edge_density: float = 0.70,
) -> dict[str, Any]:
    """Promote two observed end lobes when one finite strip bridges weak support."""

    transition = sorted(
        (
            dict(item)
            for item in bend_hypotheses
            if str(item.get("source_kind") or "") == "detector_bend_seed"
            and str(item.get("bend_class") or "") == "transition_only"
        ),
        key=lambda item: float(item.get("span_mm") or 0.0),
        reverse=True,
    )
    output = [dict(item) for item in selected]
    if (
        len(transition) < 2
        or float(transition[0].get("span_mm") or 0.0) < dominant_span_minimum_mm
        or float(transition[0].get("span_mm") or 0.0)
        < dominant_span_ratio * float(transition[1].get("span_mm") or 0.0)
    ):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    candidates = [
        item
        for item in transition[1:]
        if split_span_minimum_mm <= float(item.get("span_mm") or 0.0) <= split_span_maximum_mm
        and float(item.get("confidence") or 0.0) >= minimum_confidence
    ]
    if not candidates:
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    primary = max(candidates, key=lambda item: (float(item.get("confidence") or 0.0), float(item.get("span_mm") or 0.0)))
    lane_family = [item for item in candidates if item is primary or _same_uncertain_lane(primary, item)]
    source = min(
        lane_family,
        key=lambda item: (
            len(item.get("incident_flange_ids") or []),
            float(item.get("cross_width_mm") or float("inf")),
            -float(item.get("confidence") or 0.0),
        ),
    )
    atoms = {str(item.get("atom_id") or ""): item for item in atom_graph.get("atoms") or []}
    try:
        axis = np.asarray(source["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        origin = np.asarray(source["anchor"], dtype=np.float64)
        points = np.asarray(
            [atoms[str(identifier)]["centroid"] for identifier in source.get("atom_ids") or [] if str(identifier) in atoms],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    if len(points) < 40 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    coordinates = (points - origin) @ axis
    minimum = float(np.min(coordinates))
    maximum = float(np.max(coordinates))
    extent = maximum - minimum
    if extent < split_span_minimum_mm:
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    counts, _edges = np.histogram(coordinates, bins=20, range=(minimum, maximum))
    edge_density = float(np.mean(np.concatenate((counts[:6], counts[-6:]))))
    central_density = float(np.mean(counts[6:14]))
    if edge_density <= 0.0 or central_density > maximum_central_to_edge_density * edge_density:
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    intervals = ((minimum, minimum + 0.35 * extent), (minimum + 0.65 * extent, maximum))
    promoted = []
    for interval_index, (start_coordinate, end_coordinate) in enumerate(intervals, start=1):
        start = origin + axis * start_coordinate
        end = origin + axis * end_coordinate
        candidate = dict(source)
        candidate.update(
            {
                "bend_id": f"ASS{interval_index}",
                "anchor": ((start + end) * 0.5).tolist(),
                "span_mm": float(end_coordinate - start_coordinate),
                "span_endpoints": [start.tolist(), end.tolist()],
                "candidate_source": "axial_support_split",
                "selection_reason": "central_atom_support_deficit",
                "central_to_edge_density": central_density / edge_density,
                "source_hypothesis_id": source.get("bend_id"),
            }
        )
        try:
            if any(_duplicates(candidate, retained) for retained in output):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        output.append(candidate)
        promoted.append(candidate)
    return {"selected": output, "promoted": promoted, "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
