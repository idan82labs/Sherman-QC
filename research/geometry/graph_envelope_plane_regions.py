"""Cone-closed all-point plane witnesses on a graph interval relaxation.

An additive, separately bounded graph hypothesis. The observed-node graph
relaxation is not physical connectivity or even a joint graph witness.
Predecessor source and policy remain untouched.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.adaptive_position_regions import (
    _counts, _input, _rank_radius, _within,
)
from research.geometry.position_axis_hypothesis import _fit_plane, _number
from research.geometry.position_bend_candidates import _components, _finite_payload, _unavailable_analysis
from research.geometry.position_plane_regions import MAX_POINTS, MAX_REGIONS, _distances, _integer
from research.geometry.cone_closed_plane_regions import MAX_PLANE_EVALUATIONS, _cone_membership
from research.geometry.position_graph_envelope import build_position_graph_envelope


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", regions=[], reasons=[reason],
                  search_complete_within_declared_policy=complete)
    return result


def discover_graph_envelope_plane_regions(
    points, *, unit, position_error_bound, neighbor_counts, max_neighborhood_radius,
    max_normal_angle_bound, min_region_points, max_regions, graph_neighbor_count,
    max_link_distance, max_plane_evaluations, graph_symmetry,
):
    """Stream every declared seed into fixed-cone components and full refits.

    No iterative cone updates, point trimming, greedy removal, approximate merges
    or seed fallback after failed closure. A complete proposed component must
    pass the ORIGINAL error budget. Failure loses that proposal; it never proves
    no feasible plane exists. Final overlap alternatives are all preserved.
    """
    counts = _counts(neighbor_counts)
    radius_cap = _number(max_neighborhood_radius, "max_neighborhood_radius")
    angle = _number(max_normal_angle_bound, "max_normal_angle_bound")
    if angle >= math.pi/2:
        raise ValueError("max_normal_angle_bound must be less than pi/2 radians")
    minimum = _integer(min_region_points, "min_region_points", 3, MAX_POINTS)
    region_cap = _integer(max_regions, "max_regions", 1, MAX_REGIONS)
    fit_cap = _integer(max_plane_evaluations, "max_plane_evaluations", 1, MAX_PLANE_EVALUATIONS)
    # Validate the graph policy even when the input is empty. Both paths enforce
    # the shared 256 RAW row cap before conversion and exact deduplication.
    graph = build_position_graph_envelope(
        points, unit=unit, position_error_bound=position_error_bound,
        neighbor_count=graph_neighbor_count, max_link_distance=max_link_distance, symmetry=graph_symmetry,
    )
    count, p, epsilon = _input(points, unit, position_error_bound)
    result = {
        "schema_version": 1, "kind": "GRAPH_ENVELOPE_PLANE_REGIONS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": count, "unique_count": len(p), "positions": p.tolist(), "graph": graph,
        "position_error_bound": epsilon, "search_complete_within_declared_policy": False,
        "physical_region_count": None, "physical_geometry_is_verified": False,
        "distance_numerical_allowance": 0.0, "queries": [], "regions": [], "reasons": [],
        "policy": {
            "neighbor_counts": counts, "neighbor_count_excludes_seed": True,
            "max_neighborhood_radius": radius_cap,
            "max_normal_angle_bound_radians": angle, "min_region_points": minimum,
            "max_regions": region_cap, "region_budget_scope": "FINAL_REFITTED_EXACT_SUPPORTS_NOT_RAW_SEEDS",
            "raw_seed_query_limit": MAX_POINTS*4, "max_plane_evaluations": fit_cap,
            "fit_budget_scope": "ALL_UNIQUE_SEED_AND_COMPONENT_PLANE_FITS",
            "different_budget_unit_from_adaptive_v1": True,
            "seed_policy": "EVERY_OBSERVED_COMPLETE_TIE_KTH_NEIGHBOR_BALL",
            "nominal_radius_cap_guard_count": 1, "tie_closed_radius_cap_guard_count": 2,
            "extension_policy": "FIXED_SEED_CONE_POSSIBLE_COMPONENT_THEN_ORIGINAL_ERROR_ALL_POINT_REFIT",
            "support_deduplication": "EXACT_INDEX_TUPLES_ONLY",
            "graph_neighbor_count": int(graph_neighbor_count),
            "max_link_distance": float(max_link_distance),
            "fallback_to_seed_after_closure_failure": False,
            "graph_symmetry": graph_symmetry,
            "graph_edge_scope": "OUTER_PAIRWISE_INTERVAL_RANK_RELAXATION",
            "graph_relaxation_uses_original_error_bound": True,
            "final_error_bound_is_relaxed": False,
        },
        "diagnostics": {
            "seed_query_count": len(p)*len(counts), "unique_seed_support_count": 0,
            "accepted_raw_seed_count": 0, "plane_evaluation_count": 0,
            "component_refit_count": 0, "unique_closed_support_count": 0,
            "rejected_closed_support_count": 0, "duplicate_closed_support_count": 0,
            "split_seed_count": 0, "subminimum_component_count": 0,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "BOUNDED_SEARCH_NOT_FULL_SCAN_DISCOVERY",
            "SEED_CONE_CONDITIONAL_ON_SIMULTANEOUS_ERROR_AND_PLANAR_COUNTERPARTS",
            "POINTWISE_CONE_POSSIBILITY_NOT_JOINT_PLANE_COMPATIBILITY",
            "FAILED_COMPLETE_CLOSURE_CAN_HIDE_VALID_SUBSETS_NO_SEED_FALLBACK",
            "TLS_PLANE_PROPOSAL_NOT_COMPLETE_MINIMAX_FEASIBILITY_SEARCH",
            "INTERVAL_GRAPH_CAN_MERGE_DISTINCT_SUPPORTS_OR_HIDE_VALID_SUBSETS",
            "OUTER_GRAPH_NOT_EVEN_INDIVIDUAL_EDGE_REALIZABILITY_CERTIFICATE",
            "UPPER_GRAPH_COMPONENTS_NOT_JOINT_PHYSICAL_CONNECTIVITY",
            "FINAL_REGION_CAP_NOT_THE_V1_RAW_SEED_CAP",
            "NEARBY_COPLANAR_OBJECTS_CAN_SHARE_ONE_SAMPLE_WITNESS",
            "NO_PHYSICAL_MEMBERSHIP_CONTINUITY_TANGENCY_OR_BEND_COUNT",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ],
    }
    if not graph["search_complete_within_declared_policy"]:
        return _stop(result, "GRAPH_ENVELOPE_INCOMPLETE")
    if result["positions"] != graph["positions"]:
        return _stop(result, "CANONICAL_POSITION_INDEXING_INCONSISTENT")
    if len(p) == 0:
        return _stop(result, "INSUFFICIENT_DISTINCT_POSITIONS", complete=True)
    distances, guard = _distances(p)
    if distances is None or not math.isfinite(guard):
        return _stop(result, "DISTANCE_NUMERICAL_RANGE_UNSUPPORTED")
    result["distance_numerical_allowance"] = guard
    adjacency = np.zeros((len(p), len(p)), dtype=bool)
    for first, second in graph["edges"]:
        adjacency[first, second] = adjacency[second, first] = True
    supports = {}
    for seed, row in enumerate(distances):
        for k in counts:
            query = {"query_id": len(result["queries"]), "seed_index": seed,
                     "neighbor_count": k, "radius": None, "status": "INSUFFICIENT_OTHER_POSITIONS"}
            result["queries"].append(query)
            if len(p) <= k:
                continue
            radius = _rank_radius(row, seed, k)
            query["radius"] = radius
            if radius > radius_cap and radius-radius_cap > guard:
                query["status"] = "NOMINAL_RADIUS_CAP_EXCEEDED"
                continue
            indices = tuple(np.flatnonzero(_within(row, radius, guard)).tolist())
            query.update(status="QUERIED", point_indices=list(indices))
            supports.setdefault(indices, []).append(query["query_id"])
    diagnostics = result["diagnostics"]
    diagnostics["unique_seed_support_count"] = len(supports)
    fits = {}

    def fit(support):
        if support in fits:
            return fits[support], None
        if diagnostics["plane_evaluation_count"] >= fit_cap:
            return None, "PLANE_EVALUATION_BUDGET_EXCEEDED"
        diagnostics["plane_evaluation_count"] += 1
        plane = _fit_plane(p[list(support)], epsilon)
        if _unavailable_analysis(plane):
            return None, "PLANE_ANALYSIS_UNAVAILABLE"
        fits[support] = plane
        return plane, None

    accepted, closed = {}, set()
    for seed, query_ids in sorted(supports.items()):
        if len(seed) < minimum:
            continue
        plane, error = fit(seed)
        if error:
            return _stop(result, error)
        if (plane["reasons"] or plane["normal_angle_bound_radians"] is None
                or plane["normal_angle_bound_radians"] > angle):
            continue
        diagnostics["accepted_raw_seed_count"] += 1
        proposal = _cone_membership(p, seed, plane, epsilon, guard)
        if proposal is None:
            return _stop(result, "CONE_PROPOSAL_NUMERICAL_RANGE_UNSUPPORTED")
        allowed, cone = proposal
        if not np.all(allowed[list(seed)]):
            return _stop(result, "SEED_CONE_MEMBERSHIP_NUMERICAL_INCONSISTENCY")
        for query_id in query_ids:
            result["queries"][query_id]["cone_proposal"] = cone
        seed_set = set(seed)
        components = [component for component in _components(adjacency, allowed)
                      if not seed_set.isdisjoint(component)]
        diagnostics["split_seed_count"] += int(len(components) > 1)
        for component in components:
            if len(component) < minimum:
                diagnostics["subminimum_component_count"] += 1
                continue
            if component in closed:
                diagnostics["duplicate_closed_support_count"] += 1
                if component in accepted:
                    accepted[component]["seed_queries"].extend(query_ids)
                continue
            closed.add(component)
            diagnostics["unique_closed_support_count"] += 1
            before = diagnostics["plane_evaluation_count"]
            refit, error = fit(component)
            diagnostics["component_refit_count"] += diagnostics["plane_evaluation_count"]-before
            if error:
                return _stop(result, error)
            if (refit["reasons"] or refit["normal_angle_bound_radians"] is None
                    or refit["normal_angle_bound_radians"] > angle):
                diagnostics["rejected_closed_support_count"] += 1
                continue
            if len(accepted) >= region_cap:
                return _stop(result, "FINAL_REGION_CANDIDATE_BUDGET_EXCEEDED")
            accepted[component] = {"plane_analysis": refit, "seed_queries": list(query_ids)}
    result["regions"] = [
        {"region_id": index, "point_indices": list(support),
         "plane_analysis": accepted[support]["plane_analysis"],
         "seed_queries": sorted(set(accepted[support]["seed_queries"])),
         "physical_region_is_verified": False}
        for index, support in enumerate(sorted(accepted))
    ]
    result["search_complete_within_declared_policy"] = True
    if result["regions"]:
        result["status"] = "PLANE_REGION_HYPOTHESES"
    else:
        result["reasons"] = ["NO_ALL_POINT_CONE_CLOSED_REGION_WITHIN_DECLARED_POLICY"]
    if not _finite_payload(result):
        return _stop(result, "CONE_REGION_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED")
    return result
