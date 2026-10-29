"""Whole-component bend hypotheses with additional full cap-ball plane seeds.

Only the declared seed family/minimum changes. Predecessors remain untouched;
all final-support, graph, uncertainty and whole-transition limitations remain.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from research.geometry.cap_supported_plane_regions import discover_cap_supported_plane_regions
from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius
from research.geometry.plane_conditioned_two_radius import (
    MAX_SPLIT_EVALUATIONS,
    _integer as _band_integer,
    analyze_plane_conditioned_two_radius,
)
from research.geometry.position_axis_hypothesis import _number
from research.geometry.position_bend_candidates import (
    MAX_BRIDGE_EVALUATIONS,
    MAX_PAIR_EVALUATIONS,
    _components,
    _finite_payload,
    _integer,
    _unavailable_analysis,
)


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", candidates=[], extended_regions=[],
                  search_complete_within_declared_policy=complete, reasons=[reason])
    return result


def analyze_cap_supported_bend_candidates(
    points, *, unit, position_error_bound, neighbor_counts, max_neighborhood_radius,
    max_normal_angle_bound, min_region_points, max_regions, graph_neighbor_count,
    max_link_distance, max_axis_angle_bound, max_axial_gap, max_radius_width,
    max_pair_evaluations, max_bridge_evaluations, min_band_points, max_split_evaluations,
    max_plane_evaluations, graph_symmetry, min_seed_points,
):
    """Retain bounded radius alternatives without another plane-extension pass.

    Every associated transition is a WHOLE remainder graph component. A single
    radius is tried first; a two-radius witness follows abstention without an
    explicit numerical-unavailability signal. The inherited circle proposal does
    not expose every solver failure. No candidate constitutes physical identity,
    topology, thickness or a production decision.
    """
    axis_limit = _number(max_axis_angle_bound, "max_axis_angle_bound")
    if axis_limit >= math.pi / 2:
        raise ValueError("max_axis_angle_bound must be less than pi/2")
    axial_gap = _number(max_axial_gap, "max_axial_gap")
    width = _number(max_radius_width, "max_radius_width")
    pair_cap = _integer(max_pair_evaluations, "max_pair_evaluations", MAX_PAIR_EVALUATIONS)
    bridge_cap = _integer(max_bridge_evaluations, "max_bridge_evaluations", MAX_BRIDGE_EVALUATIONS)
    band_minimum = _band_integer(min_band_points, "min_band_points", 3, 128)
    split_cap = _band_integer(max_split_evaluations, "max_split_evaluations", 1, MAX_SPLIT_EVALUATIONS)
    discovery = discover_cap_supported_plane_regions(
        points, unit=unit, position_error_bound=position_error_bound,
        neighbor_counts=neighbor_counts, max_neighborhood_radius=max_neighborhood_radius,
        max_normal_angle_bound=max_normal_angle_bound,
        min_region_points=min_region_points, max_regions=max_regions,
        graph_neighbor_count=graph_neighbor_count, max_link_distance=max_link_distance,
        max_plane_evaluations=max_plane_evaluations, graph_symmetry=graph_symmetry,
        min_seed_points=min_seed_points,
    )
    graph = discovery["graph"]
    result = {
        "schema_version": 1, "kind": "CAP_SUPPORTED_BEND_CANDIDATES_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": discovery["input_count"], "unique_count": discovery["unique_count"],
        "positions": discovery["positions"], "plane_discovery": discovery, "graph": graph,
        "search_complete_within_declared_policy": False,
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "policy": {
            "graph_neighbor_count": int(graph_neighbor_count),
            "graph_symmetry": graph_symmetry,
            "graph_edge_scope": "OUTER_PAIRWISE_INTERVAL_RANK_RELAXATION",
            "max_link_distance": float(max_link_distance),
            "max_axis_angle_bound_radians": axis_limit,
            "max_axial_gap": axial_gap, "max_radius_width": width,
            "max_pair_evaluations": pair_cap, "max_bridge_evaluations": bridge_cap,
            "min_band_points": band_minimum, "max_split_evaluations": split_cap,
            "max_plane_evaluations": int(max_plane_evaluations),
            "max_final_plane_regions": int(max_regions),
            "min_cap_ball_seed_points": int(min_seed_points),
            "seed_query_limit": 1024,
            "region_budget_scope": "FINAL_EXACT_DEDUPLICATED_JOINT_REFIT_SUPPORTS_NOT_V1_RAW_SEEDS",
            "plane_work_budget_scope": "SEED_AND_FULL_SUPPORT_REFIT_CALLS",
            "bridge_budget_scope": "ASSOCIATIONS_EACH_WITH_ONE_SINGLE_AND_AT_MOST_ONE_TWO_RADIUS_CALL",
            "split_budget_scope": "EACH_TWO_RADIUS_ASSOCIATION_INDEPENDENTLY",
            "extension_policy": "USE_JOINT_REFITTED_DISCOVERY_REGIONS_WITHOUT_REEXTENSION",
            "pair_policy": "DISJOINT_FINAL_PLANE_SUPPORTS",
            "transition_policy": "ENTIRE_REMAINDER_COMPONENT_WITH_RELAXED_GRAPH_EDGES_TO_BOTH_PLANES",
            "transition_min_points": 3,
            "model_policy": "SINGLE_RADIUS_FIRST_TWO_RADIUS_AFTER_ABSTENTION_WITHOUT_EXPLICIT_NUMERICAL_UNAVAILABILITY_SIGNAL",
        },
        # Name retained for result/oracle compatibility; these are already the
        # final jointly refitted discovery supports, not a second extension.
        "extended_regions": [], "candidates": [],
        "search_diagnostics": {
            "extended_region_count": 0, "disjoint_pair_count": 0,
            "association_count": 0, "bridge_evaluation_count": 0,
            "single_radius_evaluation_count": 0, "two_radius_evaluation_count": 0,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "FINITE_DECLARED_SEARCH_NOT_DETECTOR_RECALL_GUARANTEE",
            "ADDITIONAL_CAP_BALL_SEEDS_NOT_CONFIDENCE_OR_PHYSICAL_SURFACE_IDENTITY",
            "FINAL_REGION_CAP_IS_NOT_THE_V1_RAW_SEED_CAP",
            "CONE_MEMBERSHIP_GATE_IS_NECESSARY_NOT_JOINT_COMPATIBILITY",
            "REJECTED_COMPLETE_CONE_PROPOSALS_HAVE_NO_SEED_FALLBACK",
            "JOINT_PLANE_WITNESS_NOT_PHYSICAL_SURFACE_MEMBERSHIP",
            "KNN_SCALE_AND_GRAPH_DEPEND_ON_DENSITY_ANISOTROPY_AND_OTHER_SURFACES",
            "OUTER_GRAPH_NOT_EVEN_INDIVIDUAL_EDGE_REALIZABILITY_CERTIFICATE",
            "UPPER_GRAPH_COMPONENTS_NOT_JOINT_PHYSICAL_CONNECTIVITY",
            "RELAXED_GRAPH_LINKS_NOT_TRUE_GAP_OR_PHYSICAL_CONTACT_CERTIFICATES",
            "NO_GREEDY_POINT_OWNERSHIP_OR_PHYSICAL_SEGMENTATION",
            "SAMPLED_GRAPH_NOT_CONTINUOUS_PHYSICAL_CONNECTIVITY",
            "NO_TANGENCY_OR_COMMON_OBJECT_VERIFICATION",
            "COMPLETE_REMAINDER_COMPONENT_CAN_INCLUDE_UNRELATED_SURFACES",
            "RELAXED_RADIUS_WITNESS_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
            "TWO_RADIUS_ASSIGNMENTS_NOT_VERIFIED_SURFACE_MEMBERSHIP",
            "SINGLE_RADIUS_SUCCESS_DOES_NOT_EXCLUDE_COMPETING_TWO_RADIUS_MODELS",
            "LEGACY_CIRCLE_PROPOSAL_DOES_NOT_CLASSIFY_ALL_SOLVER_FAILURES",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "NO_PHYSICAL_AXIS_LOCATION_OR_FINITE_MATERIAL_EXTENT",
            "NO_FORMATION_THICKNESS_CONFORMANCE_OR_BEND_COUNT_CLAIM",
        ],
        "reasons": [],
    }
    if not discovery["search_complete_within_declared_policy"]:
        return _stop(result, "PLANE_DISCOVERY_INCOMPLETE")
    if not graph["search_complete_within_declared_policy"]:
        return _stop(result, "GRAPH_ENVELOPE_INCOMPLETE")
    if discovery["positions"] != graph["positions"]:
        return _stop(result, "CANONICAL_POSITION_INDEXING_INCONSISTENT")
    if not discovery["regions"]:
        return _stop(result, "NO_PLANE_REGION_HYPOTHESES", complete=True)

    p = np.asarray(discovery["positions"], dtype=float)
    epsilon = float(position_error_bound)
    adjacency = np.zeros((len(p), len(p)), dtype=bool)
    for first, second in graph["edges"]:
        adjacency[first, second] = adjacency[second, first] = True
    regions = discovery["regions"]
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
            links = [int(np.count_nonzero(adjacency[np.ix_(support, region)]))
                     for region in (one, two)]
            if not all(links):
                continue
            associations.append((first, second, support, links))
            if len(associations) > bridge_cap:
                result["search_diagnostics"]["association_count"] = len(associations)
                return _stop(result, "BRIDGE_EVALUATION_BUDGET_EXCEEDED")
    result["search_diagnostics"]["association_count"] = len(associations)

    for first, second, support, links in associations:
        samples = (p[first["point_indices"]], p[second["point_indices"]], p[support])
        arguments = dict(unit=unit, position_error_bound=epsilon,
                         max_axis_angle_bound=axis_limit, max_axial_gap=axial_gap,
                         max_radius_width=width)
        single = analyze_plane_conditioned_radius(*samples, **arguments)
        diagnostics = result["search_diagnostics"]
        diagnostics["bridge_evaluation_count"] += 1
        diagnostics["single_radius_evaluation_count"] += 1
        if _unavailable_analysis(single):
            return _stop(result, "SINGLE_RADIUS_ANALYSIS_UNAVAILABLE")
        two = None
        status = "ABSTAINED"
        if single["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS":
            status = "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"
        else:
            two = analyze_plane_conditioned_two_radius(
                *samples, **arguments, min_band_points=band_minimum,
                max_split_evaluations=split_cap,
            )
            diagnostics["two_radius_evaluation_count"] += 1
            if _unavailable_analysis(two):
                return _stop(result, "TWO_RADIUS_ANALYSIS_UNAVAILABLE")
            if not two["search_complete_within_declared_policy"]:
                return _stop(result, "TWO_RADIUS_SEARCH_INCOMPLETE")
            if two["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS":
                status = "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"
        result["candidates"].append({
            "candidate_id": len(result["candidates"]),
            "plane_region_ids": [first["region_id"], second["region_id"]],
            "transition_point_indices": support,
            "single_radius_analysis": single, "two_radius_analysis": two, "status": status,
            "association": {
                "scope": "PAIRWISE_INTERVAL_RANK_RELAXATION_ONLY",
                "physical_connectivity_is_verified": False,
                "joint_graph_realizability_is_verified": False,
                "tangency_is_verified": False, "same_object_is_verified": False,
                "entire_remainder_component_retained": True,
                "plane_relaxed_graph_edge_counts": links,
                "relaxed_links_establish_true_gap": False,
            },
        })
    result["search_complete_within_declared_policy"] = True
    if any(candidate["status"] != "ABSTAINED" for candidate in result["candidates"]):
        result["status"] = "POSITION_GEOMETRY_CANDIDATES"
    else:
        result["reasons"] = ["NO_CONDITIONAL_RADIUS_CANDIDATE"]
    if not _finite_payload(result):
        return _stop(result, "CANDIDATE_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED")
    return result
