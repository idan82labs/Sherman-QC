"""Recover a finite bend axis from a dominant transition strip's supported boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _duplicates


def _two_means(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 4:
        return None
    centered = points - np.mean(points, axis=0)
    _u, _s, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vectors[0]
    centers = np.asarray((points[int(np.argmin(projection))], points[int(np.argmax(projection))]))
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(32):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and _:
            break
        labels = updated
        if any(not np.any(labels == index) for index in (0, 1)):
            return None
        centers = np.asarray([np.mean(points[labels == index], axis=0) for index in (0, 1)])
    return labels


def promote_supported_transition_boundary(
    selected: Sequence[Mapping[str, Any]],
    bend_hypotheses: Sequence[Mapping[str, Any]],
    flange_hypotheses: Sequence[Mapping[str, Any]],
    atom_graph: Mapping[str, Any],
    *,
    minimum_span_mm: float = 200.0,
    minimum_span_ratio: float = 2.0,
    minimum_boundary_atoms: int = 12,
    minimum_boundary_shift_ratio: float = 1.25,
    maximum_relative_boundary_shift: float = 0.75,
) -> dict[str, Any]:
    """Promote one finite strip boundary only when planar-neighbor support dominates."""

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
        or float(transition[0].get("span_mm") or 0.0) < minimum_span_mm
        or float(transition[0].get("span_mm") or 0.0)
        < minimum_span_ratio * float(transition[1].get("span_mm") or 0.0)
    ):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    source = transition[0]
    atoms = {str(item.get("atom_id") or ""): item for item in atom_graph.get("atoms") or []}
    adjacency = atom_graph.get("adjacency") or {}
    source_ids = {str(value) for value in source.get("atom_ids") or [] if str(value) in atoms}
    try:
        axis = np.asarray(source["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        origin = np.asarray(source["anchor"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    if len(source_ids) < 2 * minimum_boundary_atoms or not np.isfinite(axis).all():
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    all_points = np.asarray([atoms[identifier]["centroid"] for identifier in sorted(source_ids)], dtype=np.float64)
    axial = (all_points - origin) @ axis
    low, high = np.quantile(axial, (0.05, 0.95))
    boundary_ids = []
    for identifier in sorted(source_ids):
        point = np.asarray(atoms[identifier]["centroid"], dtype=np.float64)
        coordinate = float(np.dot(point - origin, axis))
        if low <= coordinate <= high and any(str(neighbor) not in source_ids for neighbor in adjacency.get(identifier, [])):
            boundary_ids.append(identifier)
    if len(boundary_ids) < 2 * minimum_boundary_atoms:
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    boundary_points = np.asarray([atoms[identifier]["centroid"] for identifier in boundary_ids], dtype=np.float64)
    residual = boundary_points - origin
    residual -= np.outer(residual @ axis, axis)
    _u, _s, cross_basis = np.linalg.svd(residual, full_matrices=False)
    labels = _two_means(residual @ cross_basis[:2].T)
    if labels is None or any(np.count_nonzero(labels == index) < minimum_boundary_atoms for index in (0, 1)):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    flange_map = {str(item.get("flange_id") or ""): item for item in flange_hypotheses}
    incident = [flange_map.get(str(value)) for value in source.get("canonical_flange_pair") or []]
    incident = [item for item in incident if item is not None]
    if not incident:
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    dominant_flange = max(incident, key=lambda item: len(item.get("atom_ids") or []))
    dominant_ids = {str(value) for value in dominant_flange.get("atom_ids") or []}
    support = []
    outside_sets = []
    for cluster_index in (0, 1):
        outside = {
            str(neighbor)
            for identifier, label in zip(boundary_ids, labels)
            if label == cluster_index
            for neighbor in adjacency.get(identifier, [])
            if str(neighbor) not in source_ids
        }
        outside_sets.append(outside)
        support.append(len(outside & dominant_ids))
    perpendicular_centers = []
    boundary_shifts = []
    for cluster_index in (0, 1):
        cluster_points = boundary_points[labels == cluster_index]
        cluster_coordinates = (cluster_points - origin) @ axis
        center = np.mean(cluster_points - np.outer(cluster_coordinates, axis), axis=0)
        delta = center - origin
        perpendicular_centers.append(center)
        boundary_shifts.append(float(np.linalg.norm(delta - axis * np.dot(delta, axis))))
    best = int(np.argmin(boundary_shifts))
    other = 1 - best
    cross_width = float(source.get("cross_width_mm") or 0.0)
    if (
        support[best] < minimum_boundary_atoms
        or boundary_shifts[other] < minimum_boundary_shift_ratio * max(boundary_shifts[best], 1e-9)
        or cross_width <= 0.0
        or boundary_shifts[best] > maximum_relative_boundary_shift * cross_width
    ):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}

    points = boundary_points[labels == best]
    coordinates = (points - origin) @ axis
    perpendicular_center = np.mean(points - np.outer(coordinates, axis), axis=0)
    start = perpendicular_center + axis * float(np.quantile(coordinates, 0.02))
    end = perpendicular_center + axis * float(np.quantile(coordinates, 0.98))
    candidate = dict(source)
    candidate.update(
        {
            "bend_id": "TBR1",
            "anchor": ((start + end) * 0.5).tolist(),
            "span_mm": float(np.linalg.norm(end - start)),
            "span_endpoints": [start.tolist(), end.tolist()],
            "candidate_source": "transition_boundary_refit",
            "selection_reason": "dominant_planar_neighbor_boundary_support",
            "boundary_support_counts": support,
            "boundary_shift_mm": boundary_shifts,
            "dominant_support_flange_id": dominant_flange.get("flange_id"),
        }
    )
    try:
        if any(_duplicates(candidate, retained) for retained in output):
            return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    except (KeyError, TypeError, ValueError):
        return {"selected": output, "promoted": [], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
    output.append(candidate)
    return {"selected": output, "promoted": [candidate], "uses_part_identifier": False, "uses_expected_count": False, "uses_manual_truth": False}
