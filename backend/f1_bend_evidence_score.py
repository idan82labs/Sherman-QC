"""Interpretable, identity-free evidence scores for finite F1 bend axes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.f1_interface_birth_consensus import _endpoints


def _pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
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
    return {
        "axis_angle_deg": angle,
        "transverse_offset_mm": offset,
        "shorter_span_overlap": overlap / max(min(first_span, second_span), 1e-9),
    }


def _topology_strength(bend: Mapping[str, Any]) -> float:
    source = str(bend.get("candidate_source") or "")
    reason = str(bend.get("selection_reason") or "")
    if source in {"transition_boundary_refit", "axial_support_split"}:
        return 1.0
    if source == "interface_birth_stitch":
        return 0.96 if reason == "supported_parallel_skin_plane_snap" else 0.92
    if source == "interface_birth_consensus":
        return 0.90
    if reason == "corridor_source_plane_pareto_snap":
        return 0.90
    return 0.60


def _rating(score: float) -> str:
    if score >= 80.0:
        return "A"
    if score >= 65.0:
        return "B"
    if score >= 50.0:
        return "C"
    return "D"


def _object_geometry_quality(line_object: Mapping[str, Any]) -> float:
    finite_run = float(line_object.get("cluster_max_finite_support_longest_run_fraction") or 0.0)
    endpoint = float(line_object.get("cluster_max_endpoint_support_fraction") or 0.0)
    transition = float(line_object.get("cluster_max_two_sided_transition_score") or 0.0)
    coherence = float(line_object.get("cluster_max_two_sided_normal_coherence") or 0.0)
    source_consensus = min(float(line_object.get("cluster_source_count") or 0.0) / 3.0, 1.0)
    median_distance = float(line_object.get("cluster_mean_median_line_distance_mm") or 1e9)
    distance_quality = math.exp(-max(median_distance, 0.0) / 4.0) if np.isfinite(median_distance) else 0.0
    return float(
        15.0 * np.clip(finite_run, 0.0, 1.0)
        + 8.0 * np.clip(endpoint, 0.0, 1.0)
        + 22.0 * np.clip(transition, 0.0, 1.0)
        + 10.0 * np.clip(coherence, 0.0, 1.0)
        + 12.0 * np.clip(source_consensus, 0.0, 1.0)
        + 10.0 * np.clip(distance_quality, 0.0, 1.0)
    )


def _parallel_lane_ownership(
    selected: Mapping[str, Any] | None,
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if selected is None:
        return {"state": "unavailable", "margin": None, "competitor_line_object_id": None}
    selected_quality = _object_geometry_quality(selected)
    competitors = []
    for candidate in line_objects:
        if candidate is selected or str(candidate.get("bend_id") or "") == str(selected.get("bend_id") or ""):
            continue
        try:
            geometry = _pair_metrics(selected, candidate)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if (
            geometry["axis_angle_deg"] <= 6.0
            and 2.0 <= geometry["transverse_offset_mm"] <= 30.0
            and geometry["shorter_span_overlap"] >= 0.55
        ):
            competitors.append((_object_geometry_quality(candidate), dict(candidate), geometry))
    if not competitors:
        return {
            "state": "isolated",
            "margin": None,
            "selected_geometry_quality": round(selected_quality, 3),
            "competitor_geometry_quality": None,
            "competitor_line_object_id": None,
        }
    competitor_quality, competitor, geometry = max(
        competitors, key=lambda item: (item[0], str(item[1].get("bend_id") or ""))
    )
    margin = selected_quality - competitor_quality
    state = "selected_stronger" if margin >= 5.0 else "competitor_stronger" if margin <= -5.0 else "ambiguous"
    return {
        "state": state,
        "margin": round(float(margin), 3),
        "selected_geometry_quality": round(selected_quality, 3),
        "competitor_geometry_quality": round(float(competitor_quality), 3),
        "competitor_line_object_id": competitor.get("bend_id"),
        "competitor_offset_mm": round(float(geometry["transverse_offset_mm"]), 3),
        "competitor_axis_angle_deg": round(float(geometry["axis_angle_deg"]), 3),
        "competitor_shorter_span_overlap": round(float(geometry["shorter_span_overlap"]), 3),
    }


def score_bend_evidence(
    bend: Mapping[str, Any],
    line_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a 0-100 evidence score; this is not a calibrated probability."""

    best = None
    for line_object in line_objects:
        try:
            geometry = _pair_metrics(bend, line_object)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        cost = (
            geometry["axis_angle_deg"] / 5.0
            + geometry["transverse_offset_mm"] / 4.0
            + (1.0 - min(geometry["shorter_span_overlap"], 1.0))
        )
        key = (cost, geometry["transverse_offset_mm"], geometry["axis_angle_deg"])
        if best is None or key < best[0]:
            best = (key, dict(line_object), geometry)
    matched = False
    line_object: dict[str, Any] = {}
    geometry = {"axis_angle_deg": 180.0, "transverse_offset_mm": 1e9, "shorter_span_overlap": 0.0}
    if best is not None:
        _key, line_object, geometry = best
        matched = (
            geometry["axis_angle_deg"] <= 6.0
            and geometry["transverse_offset_mm"] <= 8.0
            and geometry["shorter_span_overlap"] >= 0.45
        )
    matched_line_object = line_object if matched else None
    if not matched:
        line_object = {}

    finite_run = float(line_object.get("cluster_max_finite_support_longest_run_fraction") or 0.0)
    endpoint = float(line_object.get("cluster_max_endpoint_support_fraction") or 0.0)
    transition = float(line_object.get("cluster_max_two_sided_transition_score") or 0.0)
    coherence = float(line_object.get("cluster_max_two_sided_normal_coherence") or 0.0)
    source_count = float(line_object.get("cluster_source_count") or 0.0)
    source_consensus = min(source_count / 3.0, 1.0)
    median_distance = float(line_object.get("cluster_mean_median_line_distance_mm") or 1e9)
    distance_quality = math.exp(-max(median_distance, 0.0) / 4.0) if np.isfinite(median_distance) else 0.0
    stability = max(
        float(line_object.get("cluster_max_stability_score") or 0.0),
        float(bend.get("stability_score") or 0.0),
        float(bend.get("confidence") or 0.0),
        float(bend.get("debug_confidence") or 0.0),
    )
    topology = _topology_strength(bend)
    components = {
        "finite_run": 15.0 * np.clip(finite_run, 0.0, 1.0),
        "endpoint_support": 8.0 * np.clip(endpoint, 0.0, 1.0),
        "two_sided_transition": 22.0 * np.clip(transition, 0.0, 1.0),
        "normal_coherence": 10.0 * np.clip(coherence, 0.0, 1.0),
        "source_consensus": 12.0 * np.clip(source_consensus, 0.0, 1.0),
        "scan_distance": 10.0 * np.clip(distance_quality, 0.0, 1.0),
        "solver_stability": 5.0 * np.clip(stability, 0.0, 1.0),
        "topology": 18.0 * np.clip(topology, 0.0, 1.0),
    }
    score = round(float(sum(components.values())), 3)
    ownership = _parallel_lane_ownership(matched_line_object, line_objects)
    candidate_source = str(bend.get("candidate_source") or "")
    if candidate_source in {"interface_birth_stitch", "interface_birth_consensus", "transition_boundary_refit", "axial_support_split"} and score >= 65.0:
        review_queue = "supported_derived_topology"
    elif ownership["state"] == "competitor_stronger":
        review_queue = "semantic_ownership_review"
    elif ownership["state"] == "ambiguous":
        review_queue = "parallel_lane_review"
    elif score < 65.0:
        review_queue = "weak_evidence_review"
    else:
        review_queue = "supported_high_evidence"
    return {
        "bend_id": str(bend.get("bend_id") or ""),
        "evidence_score": score,
        "evidence_rating": _rating(score),
        "interpretation": "intrinsic_geometry_evidence_not_probability",
        "review_queue": review_queue,
        "components": {key: round(float(value), 3) for key, value in components.items()},
        "features": {
            "finite_support_longest_run_fraction": finite_run,
            "endpoint_support_fraction": endpoint,
            "two_sided_transition_score": transition,
            "two_sided_normal_coherence": coherence,
            "cluster_source_count": int(source_count),
            "median_line_distance_mm": None if not np.isfinite(median_distance) else median_distance,
            "stability": stability,
            "topology_strength": topology,
        },
        "line_object_match": {
            "matched": matched,
            "line_object_id": line_object.get("bend_id") if matched else None,
            **{key: round(float(value), 6) for key, value in geometry.items()},
        },
        "parallel_lane_ownership": ownership,
        "uses_part_identifier": False,
        "uses_expected_count": False,
        "uses_manual_truth": False,
    }
