"""Complete prismatic bend lattices from measured cross-fold widths and dense evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _duplicates, _endpoints


def _metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    first = _endpoints(left)
    second = _endpoints(right)
    first_delta = first[1] - first[0]
    second_delta = second[1] - second[0]
    first_span = float(np.linalg.norm(first_delta))
    second_span = float(np.linalg.norm(second_delta))
    first_axis = first_delta / first_span
    second_axis = second_delta / second_span
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))))
    midpoint_delta = np.mean(second, axis=0) - np.mean(first, axis=0)
    offset = max(
        float(np.linalg.norm(midpoint_delta - first_axis * np.dot(midpoint_delta, first_axis))),
        float(np.linalg.norm(midpoint_delta - second_axis * np.dot(midpoint_delta, second_axis))),
    )
    interval = sorted(float(np.dot(point - np.mean(first, axis=0), first_axis)) for point in second)
    overlap = max(0.0, min(0.5 * first_span, interval[1]) - max(-0.5 * first_span, interval[0]))
    return {"axis_angle_deg": angle, "transverse_offset_mm": offset, "shorter_span_overlap": overlap / max(min(first_span, second_span), 1e-9)}


def _quality(item: Mapping[str, Any]) -> float:
    finite = float(item.get("cluster_max_finite_support_longest_run_fraction") or 0.0)
    endpoint = float(item.get("cluster_max_endpoint_support_fraction") or 0.0)
    transition = float(item.get("cluster_max_two_sided_transition_score") or 0.0)
    coherence = float(item.get("cluster_max_two_sided_normal_coherence") or 0.0)
    source_count = min(float(item.get("cluster_source_count") or 0.0) / 3.0, 1.0)
    distance = float(item.get("cluster_mean_median_line_distance_mm") or 1e9)
    return 15.0 * finite + 8.0 * endpoint + 22.0 * transition + 10.0 * coherence + 12.0 * source_count + 10.0 * math.exp(-max(distance, 0.0) / 4.0)


def _strip_aspect_ratio(line: Mapping[str, Any], strip: Mapping[str, Any]) -> float:
    endpoints = _endpoints(line)
    span = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    width = float(strip.get("cross_fold_width_mm") or 0.0)
    return span / max(width, 1e-9)


def complete_parallel_width_lattice(
    selected: Sequence[Mapping[str, Any]],
    support_strips: Sequence[Mapping[str, Any]],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add only unique companions supported at a measured physical strip width."""

    output = [dict(item) for item in selected]
    strips = {str(item.get("strip_id") or ""): dict(item) for item in support_strips}
    selected_pairs = [(item, strips[str(item.get("bend_id") or "")]) for item in selected if str(item.get("bend_id") or "") in strips]
    selected_strips = [strip for _, strip in selected_pairs]
    activation_reasons = []
    if len(selected_strips) < 2:
        activation_reasons.append("fewer_than_two_selected_support_strips")
    if len(selected_strips) != len(selected):
        activation_reasons.append("selected_topology_contains_non_strip_objects")
    if any(str(item.get("strip_family") or "") != "prismatic_plane_lattice_long" for item in selected_strips):
        activation_reasons.append("selected_strip_is_not_long_prismatic_plane_lattice")
    try:
        aspect_ratios = [_strip_aspect_ratio(line, strip) for line, strip in selected_pairs]
    except (KeyError, TypeError, ValueError):
        aspect_ratios = []
        activation_reasons.append("selected_strip_geometry_invalid")
    if aspect_ratios and min(aspect_ratios) < 18.0:
        activation_reasons.append("selected_strip_aspect_ratio_below_18")
    if activation_reasons:
        return {
            "selected": output,
            "promoted": [],
            "activated": False,
            "activation_reasons": activation_reasons,
            "selected_strip_aspect_ratios": aspect_ratios,
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
        }
    promoted = []
    for retained in selected:
        strip = strips.get(str(retained.get("bend_id") or ""))
        if strip is None:
            continue
        width = float(strip.get("cross_fold_width_mm") or 0.0)
        if width < 5.0:
            continue
        candidates = []
        for line_object in line_objects:
            if int(line_object.get("cluster_source_count") or 0) < 2:
                continue
            if float(line_object.get("cluster_max_finite_support_longest_run_fraction") or 0.0) < 0.80:
                continue
            try:
                geometry = _metrics(retained, line_object)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            ratio = geometry["transverse_offset_mm"] / width
            if geometry["axis_angle_deg"] > 3.0 or geometry["shorter_span_overlap"] < 0.80 or not 0.75 <= ratio <= 1.25:
                continue
            candidates.append((abs(ratio - 1.0), -_quality(line_object), str(line_object.get("bend_id") or ""), dict(line_object), ratio))
        if not candidates:
            continue
        best = min(candidates)
        if best[0] > 0.10:
            continue
        candidate = best[3]
        if any(_duplicates(candidate, item) for item in output):
            continue
        candidate["bend_id"] = f"PWC{len(promoted) + 1}"
        candidate["candidate_source"] = "parallel_width_companion"
        candidate["selection_reason"] = "unique_measured_cross_fold_width_companion"
        candidate["source_strip_id"] = retained.get("bend_id")
        candidate["offset_to_width_ratio"] = best[4]
        output.append(candidate)
        promoted.append(candidate)

    plane_candidates = []
    for line_object in line_objects:
        if "plane_lattice" not in (line_object.get("cluster_sources") or []):
            continue
        if int(line_object.get("cluster_source_count") or 0) < 2 or _quality(line_object) < 60.0:
            continue
        if float(line_object.get("cluster_max_finite_support_longest_run_fraction") or 0.0) < 0.80:
            continue
        if any(_duplicates(line_object, item) for item in output):
            continue
        compatible = False
        for retained in selected:
            try:
                geometry = _metrics(retained, line_object)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if geometry["axis_angle_deg"] <= 3.0 and geometry["shorter_span_overlap"] >= 0.80:
                compatible = True
                break
        if compatible:
            plane_candidates.append(dict(line_object))
    if len(plane_candidates) == 1:
        candidate = plane_candidates[0]
        candidate["bend_id"] = f"PWC{len(promoted) + 1}"
        candidate["candidate_source"] = "parallel_width_companion"
        candidate["selection_reason"] = "unique_nonduplicate_plane_lattice_companion"
        output.append(candidate)
        promoted.append(candidate)
    return {
        "selected": output,
        "promoted": promoted,
        "activated": True,
        "activation_reasons": [],
        "selected_strip_aspect_ratios": aspect_ratios,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
