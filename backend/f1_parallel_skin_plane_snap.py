"""Snap stitched bend axes from a sparse thickness face to its supported skin plane."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints


def _normalized_plane(flange: Mapping[str, Any]) -> tuple[np.ndarray, float] | None:
    normal = np.asarray(flange.get("normal"), dtype=np.float64)
    if normal.shape != (3,):
        return None
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    try:
        distance = float(flange["d"]) / norm
    except (KeyError, TypeError, ValueError):
        return None
    return normal / norm, distance


def _intersection(first: tuple[np.ndarray, float], second: tuple[np.ndarray, float]) -> tuple[np.ndarray, np.ndarray] | None:
    first_normal, first_d = first
    second_normal, second_d = second
    axis = np.cross(first_normal, second_normal)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis /= norm
    matrix = np.vstack((first_normal, second_normal))
    try:
        anchor = matrix.T @ np.linalg.solve(
            matrix @ matrix.T,
            -np.asarray((first_d, second_d), dtype=np.float64),
        )
    except np.linalg.LinAlgError:
        return None
    return anchor, axis


def snap_stitches_to_supported_skin_plane(
    selected: Sequence[Mapping[str, Any]],
    flange_hypotheses: Sequence[Mapping[str, Any]],
    *,
    parallel_tolerance_deg: float = 2.0,
    minimum_plane_separation_mm: float = 1.0,
    maximum_plane_separation_mm: float = 6.0,
    minimum_support_ratio: float = 1.25,
    maximum_axis_shift_mm: float = 6.0,
    minimum_interface_atoms: int = 8,
) -> dict[str, Any]:
    """Use a dominant parallel face as the visible skin of a stitched interface."""

    flanges = {str(item.get("flange_id") or ""): dict(item) for item in flange_hypotheses}
    planes = {identifier: _normalized_plane(item) for identifier, item in flanges.items()}
    cosine_limit = math.cos(math.radians(parallel_tolerance_deg))
    output = []
    replacements = []
    pruned = []
    for source in selected:
        item = dict(source)
        if str(item.get("candidate_source") or "") != "interface_birth_stitch":
            output.append(item)
            continue
        pair = [str(value) for value in item.get("canonical_flange_pair") or ()]
        if len(pair) != 2 or any(identifier not in flanges for identifier in pair):
            output.append(item)
            continue
        interface_support = [len(flanges[identifier].get("atom_ids") or []) for identifier in pair]
        if min(interface_support) < minimum_interface_atoms:
            pruned.append(
                {
                    "bend_id": item.get("bend_id"),
                    "reason": "insufficient_interface_plane_support",
                    "interface_atom_counts": interface_support,
                }
            )
            continue
        line = _endpoints(item)
        source_axis = line[1] - line[0]
        source_span = float(np.linalg.norm(source_axis))
        if source_span <= 1e-9:
            output.append(item)
            continue
        source_axis /= source_span
        source_midpoint = np.mean(line, axis=0)
        proposals = []
        for side_index, source_id in enumerate(pair):
            other_id = pair[1 - side_index]
            source_plane = planes.get(source_id)
            other_plane = planes.get(other_id)
            if source_plane is None or other_plane is None:
                continue
            source_support = len(flanges[source_id].get("atom_ids") or [])
            for sibling_id, sibling in flanges.items():
                if sibling_id in pair:
                    continue
                sibling_plane = planes.get(sibling_id)
                if sibling_plane is None:
                    continue
                dot = float(np.dot(source_plane[0], sibling_plane[0]))
                if abs(dot) < cosine_limit:
                    continue
                aligned_d = sibling_plane[1] if dot >= 0.0 else -sibling_plane[1]
                separation = abs(source_plane[1] - aligned_d)
                sibling_support = len(sibling.get("atom_ids") or [])
                if not (minimum_plane_separation_mm <= separation <= maximum_plane_separation_mm):
                    continue
                if sibling_support < minimum_support_ratio * max(source_support, 1):
                    continue
                replacement_plane = (source_plane[0], aligned_d)
                intersection = _intersection(other_plane, replacement_plane)
                if intersection is None:
                    continue
                infinite_anchor, axis = intersection
                angle = math.degrees(math.acos(float(np.clip(abs(np.dot(axis, source_axis)), 0.0, 1.0))))
                if angle > 4.0:
                    continue
                if np.dot(axis, source_axis) < 0.0:
                    axis *= -1.0
                anchor = infinite_anchor + axis * np.dot(source_midpoint - infinite_anchor, axis)
                shift = float(np.linalg.norm(anchor - source_midpoint))
                if shift > maximum_axis_shift_mm:
                    continue
                proposals.append((sibling_support / max(source_support, 1), -shift, source_id, sibling_id, anchor, axis, separation))
        if proposals:
            _ratio, negative_shift, source_id, sibling_id, anchor, axis, separation = max(
                proposals, key=lambda proposal: (proposal[0], proposal[1], proposal[3])
            )
            half = 0.5 * source_span * axis
            item["pre_skin_snap_span_endpoints"] = item.get("span_endpoints")
            item["anchor"] = anchor.tolist()
            item["axis_direction"] = axis.tolist()
            item["span_endpoints"] = [(anchor - half).tolist(), (anchor + half).tolist()]
            item["selection_reason"] = "supported_parallel_skin_plane_snap"
            replacements.append(
                {
                    "bend_id": item.get("bend_id"),
                    "source_flange_id": source_id,
                    "skin_flange_id": sibling_id,
                    "plane_separation_mm": separation,
                    "axis_shift_mm": -negative_shift,
                }
            )
        output.append(item)
    return {
        "selected": output,
        "replacements": replacements,
        "pruned": pruned,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
