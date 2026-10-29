"""Unlabelled position-region association and conditional radius proposals.

This bounded search generates hypotheses, not verified bends. Pairwise-possible
sample links are neither jointly realizable topology nor physical connectivity.
No CAD, patch labels, normals, exact axis, true radius or learned scores enter.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius
from research.geometry.position_axis_hypothesis import _fit_plane, _number
from research.geometry.position_plane_regions import discover_position_plane_regions


MAX_PAIR_EVALUATIONS = 2016  # All unordered pairs of at most 64 regions.
MAX_BRIDGE_EVALUATIONS = 256


def _integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an explicit integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return int(value)


def _components(adjacency, allowed):
    """Connected components of the INDUCED graph; no paths through excluded rows."""
    unseen = set(np.flatnonzero(allowed).tolist())
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, frontier = [start], [start]
        while frontier:
            current = frontier.pop()
            neighbors = unseen.intersection(np.flatnonzero(adjacency[current]).tolist())
            unseen.difference_update(neighbors)
            frontier.extend(sorted(neighbors))
            component.extend(neighbors)
        groups.append(tuple(sorted(component)))
    return sorted(groups)


def _distances(points):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        difference = points[:, None, :] - points[None, :, :]
        result = np.hypot(np.hypot(difference[:, :, 0], difference[:, :, 1]), difference[:, :, 2])
    return result if np.isfinite(result).all() else None


def _fixed_plane_membership(points, plane, epsilon, numerical):
    witness = plane["plane_witness"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offsets = points - np.asarray(witness["reference_point"])
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale == 0:
            return None
        signed = ((offsets / scale) @ np.asarray(witness["normal"])
                  - witness["offset_from_reference"] / scale) * scale
        budget = epsilon + max(numerical, plane["numerical_allowance"])
    if not np.isfinite(signed).all() or not math.isfinite(budget):
        return None
    return np.abs(signed) <= budget


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", candidates=[], extended_regions=[],
                  search_complete_within_declared_policy=complete, reasons=[reason])
    return result


def _unavailable_analysis(analysis):
    return any(token in reason for reason in analysis["reasons"] for token in (
        "NUMERICAL_RANGE_UNSUPPORTED", "PLANE_FIT_UNAVAILABLE", "NUMERICAL_ENCLOSURE_INCONSISTENT",
    ))


def _sample_link_bound(distance, uncertainty, gap):
    upper = distance + uncertainty
    if not math.isfinite(upper):
        return None
    return {"observed_min_distance": distance,
            "minimum_distance_interval": [max(0.0, distance - uncertainty), upper],
            "all_error_models_have_a_sample_link_within_gap": upper <= gap}


def _finite_payload(value):
    if isinstance(value, dict):
        return all(_finite_payload(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_payload(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def analyze_position_bend_candidates(
    points, *, unit, position_error_bound, neighborhood_radii,
    max_normal_angle_bound, min_region_points, max_regions, sample_gap,
    max_axis_angle_bound, max_axial_gap, max_radius_width,
    max_pair_evaluations, max_bridge_evaluations,
):
    """Enumerate position-only plane/transition radius hypotheses within caps.

    Neighborhood radii, graph gap and all resolution policies are caller supplied,
    not calibrated or selected here. The supplied 256-row cloud is the entire
    search scope; no outside points are discovered. All alternative associations
    are retained, including abstained bridge diagnostics. There is no ranking,
    residual trimming, bend count, formation state or production release.
    """
    gap = _number(sample_gap, "sample_gap")
    axis_limit = _number(max_axis_angle_bound, "max_axis_angle_bound")
    if axis_limit >= math.pi / 2:
        raise ValueError("max_axis_angle_bound must be less than pi/2")
    axial_gap = _number(max_axial_gap, "max_axial_gap")
    radius_width = _number(max_radius_width, "max_radius_width")
    pair_cap = _integer(max_pair_evaluations, "max_pair_evaluations", MAX_PAIR_EVALUATIONS)
    bridge_cap = _integer(max_bridge_evaluations, "max_bridge_evaluations", MAX_BRIDGE_EVALUATIONS)
    discovery = discover_position_plane_regions(
        points, unit=unit, position_error_bound=position_error_bound,
        neighborhood_radii=neighborhood_radii, max_normal_angle_bound=max_normal_angle_bound,
        min_region_points=min_region_points, max_regions=max_regions,
    )
    result = {
        "schema_version": 1, "kind": "POSITION_BEND_CANDIDATES_V1", "status": "ABSTAINED",
        "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": discovery["input_count"], "unique_count": discovery["unique_count"],
        "positions": discovery["positions"], "plane_discovery": discovery,
        "search_complete_within_declared_policy": False,
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "policy": {"sample_gap": gap, "max_axis_angle_bound_radians": axis_limit,
                   "max_axial_gap": axial_gap, "max_radius_width": radius_width,
                   "max_pair_evaluations": pair_cap, "max_bridge_evaluations": bridge_cap,
                   "extension_policy": "EACH_FIXED_PLANE_COMPATIBLE_COMPONENT_INTERSECTING_ANY_SEED_MEMBER",
                   "pair_policy": "DISJOINT_EXTENDED_SUPPORTS",
                   "transition_policy": "ENTIRE_REMAINDER_COMPONENT_WITH_PAIRWISE_POSSIBLE_LINKS_TO_BOTH_PLANES",
                   "transition_min_points": 3},
        "graph": None, "extended_regions": [], "candidates": [],
        "search_diagnostics": {"extended_region_count": 0, "disjoint_pair_count": 0,
                               "association_count": 0, "bridge_evaluation_count": 0,
                               "split_seed_region_count": 0, "subminimum_component_count": 0},
        "limitations": [
            "FINITE_DECLARED_SEARCH_NOT_DETECTOR_RECALL_GUARANTEE",
            "OBSERVED_BALL_MEMBERSHIP_CAN_CHANGE_UNDER_POSITION_ERROR",
            "FIXED_SEED_PLANE_EXTENSION_CAN_MISS_COMPATIBLE_TRUE_PLANE_SAMPLES",
            "SEED_SUPPORTS_SPLIT_BY_SAMPLED_COMPONENT_SUBMINIMUM_PIECES_NOT_PROPOSED",
            "NO_GREEDY_POINT_OWNERSHIP_OR_PHYSICAL_SEGMENTATION",
            "PAIRWISE_POSSIBLE_LINKS_NOT_JOINTLY_REALIZABLE_TOPOLOGY",
            "SAMPLED_GRAPH_NOT_CONTINUOUS_PHYSICAL_CONNECTIVITY",
            "NO_TANGENCY_OR_COMMON_OBJECT_VERIFICATION",
            "COMPLETE_REMAINDER_COMPONENT_CAN_INCLUDE_UNRELATED_SURFACES",
            "RELAXED_RADIUS_WITNESS_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "NO_PHYSICAL_AXIS_LOCATION_OR_FINITE_MATERIAL_EXTENT",
            "NO_FORMATION_THICKNESS_CONFORMANCE_OR_BEND_COUNT_CLAIM",
        ],
        "reasons": [],
    }
    if not discovery["search_complete_within_declared_policy"]:
        return _stop(result, "PLANE_DISCOVERY_INCOMPLETE")
    if not discovery["regions"]:
        return _stop(result, "NO_PLANE_REGION_HYPOTHESES", complete=True)

    p = np.asarray(discovery["positions"], dtype=float)
    epsilon = float(position_error_bound)
    numerical = discovery["distance_numerical_allowance"]
    uncertainty = 2 * epsilon + numerical
    threshold = gap + uncertainty
    distances = _distances(p)
    if distances is None or not math.isfinite(threshold):
        return _stop(result, "ASSOCIATION_NUMERICAL_RANGE_UNSUPPORTED")
    adjacency = distances <= threshold
    np.fill_diagonal(adjacency, False)
    result["graph"] = {
        "scope": "PAIRWISE_POSSIBLE_SAMPLED_LINKS_ONLY",
        "distance_uncertainty_allowance": uncertainty,
        "distance_numerical_allowance": numerical,
        "edge_distance_threshold": threshold,
        "edge_count": int(np.count_nonzero(np.triu(adjacency, 1))),
        "joint_realizability_is_verified": False, "physical_connectivity_is_verified": False,
    }

    # The seed plane is FIXED during extension. Reachability never passes through
    # samples incompatible with that plane. Disconnected compatible components
    # remain separate even if one original seed ball covered more than one.
    extensions = {}
    for seed in discovery["regions"]:
        compatible = _fixed_plane_membership(p, seed["plane_analysis"], epsilon, numerical)
        if compatible is None:
            return _stop(result, "EXTENSION_NUMERICAL_RANGE_UNSUPPORTED")
        seed_indices = set(seed["point_indices"])
        if not np.all(compatible[seed["point_indices"]]):
            return _stop(result, "SEED_MEMBERSHIP_NUMERICAL_INCONSISTENCY")
        components = [support for support in _components(adjacency, compatible)
                      if not seed_indices.isdisjoint(support)]
        result["search_diagnostics"]["split_seed_region_count"] += len(components) > 1
        for support in components:
            if len(support) < min_region_points:
                result["search_diagnostics"]["subminimum_component_count"] += 1
                continue
            if support in extensions:
                extensions[support]["seed_region_ids"].append(seed["region_id"])
                continue
            # Recompute a joint all-point witness and cone for every extension.
            # A successful fixed-normal selection does not excuse a failed refit.
            plane = _fit_plane(p[list(support)], epsilon)
            if _unavailable_analysis(plane):
                return _stop(result, "EXTENSION_PLANE_ANALYSIS_UNAVAILABLE")
            if plane["reasons"] or plane["normal_angle_bound_radians"] > max_normal_angle_bound:
                continue
            if len(extensions) >= max_regions:
                return _stop(result, "EXTENDED_REGION_BUDGET_EXCEEDED")
            extensions[support] = {"seed_region_ids": [seed["region_id"]], "plane_analysis": plane}
    regions = [
        {"region_id": index, "point_indices": list(support),
         "seed_region_ids": sorted(set(extensions[support]["seed_region_ids"])),
         "plane_analysis": extensions[support]["plane_analysis"],
         "physical_region_is_verified": False}
        for index, support in enumerate(sorted(extensions))
    ]
    result["extended_regions"] = regions
    result["search_diagnostics"]["extended_region_count"] = len(regions)
    pairs = [(first, second) for first, second in combinations(regions, 2)
             if set(first["point_indices"]).isdisjoint(second["point_indices"])]
    result["search_diagnostics"]["disjoint_pair_count"] = len(pairs)
    if len(pairs) > pair_cap:
        return _stop(result, "PAIR_EVALUATION_BUDGET_EXCEEDED")

    associations = []
    for first, second in pairs:
        one, two = first["point_indices"], second["point_indices"]
        remaining = np.ones(len(p), dtype=bool)
        remaining[one + two] = False
        for support in _components(adjacency, remaining):
            if len(support) < 3:
                continue
            support = list(support)
            nearest = [float(np.min(distances[np.ix_(support, region)])) for region in (one, two)]
            if any(distance > threshold for distance in nearest):
                continue
            link_bounds = [_sample_link_bound(distance, uncertainty, gap) for distance in nearest]
            if any(bound is None for bound in link_bounds):
                return _stop(result, "SAMPLE_LINK_BOUND_NUMERICAL_RANGE_UNSUPPORTED")
            associations.append((first, second, support, link_bounds))
            if len(associations) > bridge_cap:
                result["search_diagnostics"]["association_count"] = len(associations)
                return _stop(result, "BRIDGE_EVALUATION_BUDGET_EXCEEDED")
    result["search_diagnostics"]["association_count"] = len(associations)

    # Budget-check the entire association set BEFORE any radius evaluation. An
    # overflow never selects a traversal-order-dependent prefix of candidates.
    for first, second, support, link_bounds in associations:
        analysis = analyze_plane_conditioned_radius(
            p[first["point_indices"]], p[second["point_indices"]], p[support],
            unit=unit, position_error_bound=epsilon, max_axis_angle_bound=axis_limit,
            max_axial_gap=axial_gap, max_radius_width=radius_width,
        )
        result["search_diagnostics"]["bridge_evaluation_count"] += 1
        if _unavailable_analysis(analysis):
            return _stop(result, "BRIDGE_ANALYSIS_UNAVAILABLE")
        result["candidates"].append({
            "candidate_id": len(result["candidates"]),
            "plane_region_ids": [first["region_id"], second["region_id"]],
            "transition_point_indices": support, "bridge_analysis": analysis,
            "association": {
                "scope": "POSSIBLE_SAMPLED_GRAPH_ONLY",
                "physical_connectivity_is_verified": False,
                "joint_graph_realizability_is_verified": False,
                "tangency_is_verified": False, "same_object_is_verified": False,
                "entire_remainder_component_retained": True,
                "plane_sample_link_bounds": link_bounds,
            },
        })
    result["search_complete_within_declared_policy"] = True
    if any(candidate["bridge_analysis"]["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
           for candidate in result["candidates"]):
        result["status"] = "POSITION_GEOMETRY_CANDIDATES"
    else:
        result["reasons"] = ["NO_CONDITIONAL_RADIUS_CANDIDATE"]
    if not _finite_payload(result):
        return _stop(result, "CANDIDATE_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED")
    return result
