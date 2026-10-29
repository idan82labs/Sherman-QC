"""All-point plane witnesses from uncertainty-cone proposals, research only.

Unlike the retained adaptive V1, the 64-region budget here counts FINAL refitted,
exact-deduplicated components, not raw plane seeds. Up to 1,024 declared seed
queries and an explicit 2,048 fit budget bound this distinct search policy.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.adaptive_position_regions import (
    _counts, _input, _rank_radius, _within, build_mutual_position_graph,
)
from research.geometry.position_axis_hypothesis import _fit_plane, _number
from research.geometry.position_bend_candidates import _components, _finite_payload, _unavailable_analysis
from research.geometry.position_plane_regions import MAX_POINTS, MAX_REGIONS, _distances, _integer


MAX_PLANE_EVALUATIONS = 2048
_MACHINE_EPS = float(np.finfo(float).eps)


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", regions=[], reasons=[reason],
                  search_complete_within_declared_policy=complete)
    return result


def _cone_membership(points, seed_indices, plane, epsilon, numerical):
    """Necessary separate-point compatibility with a possible seed plane.

    The true seed centroid is at most epsilon from the observed centroid, NOT
    epsilon/sqrt(N). A candidate adds another epsilon. Minimum absolute dot
    product over the normal cone is max(0, a*cos(alpha)-b*sin(alpha)). Each point
    may use a different normal/offset; this is not a simultaneous plane witness.
    """
    normal = np.asarray(plane["plane_witness"]["normal"])
    alpha = plane["normal_angle_bound_radians"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reference = points[seed_indices[0]]
        offsets = points-reference
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale <= 0:
            return None
        local = offsets/scale
        centroid = np.mean(local[list(seed_indices)], axis=0)
        vectors = local-centroid
        signed = vectors @ normal
        tangent = vectors-signed[:, None]*normal
        b = np.hypot(np.hypot(tangent[:, 0], tangent[:, 1]), tangent[:, 2])
        lower = np.maximum(0.0, np.abs(signed)*math.cos(alpha)-b*math.sin(alpha))
        # Disclosed arithmetic allowance; not directed-rounded certification.
        guard = (4*max(numerical, plane["numerical_allowance"])
                 + 128*_MACHINE_EPS*(float(np.max(np.abs(points)))+scale))
        budget = 2*epsilon+guard
        centroid_world = reference+centroid*scale
        allowed = lower <= budget/scale
    if (not np.isfinite(lower).all() or not np.isfinite(centroid_world).all()
            or not math.isfinite(guard) or not math.isfinite(budget)):
        return None
    return allowed, {
        "observed_seed_centroid": centroid_world.tolist(),
        "normal_angle_bound_radians": alpha,
        "numerical_allowance": guard, "proposal_distance_bound": budget,
        "candidate_point_count": int(np.count_nonzero(allowed)),
        "same_plane_membership_is_verified": False,
        "joint_compatibility_is_verified": False,
    }


def discover_cone_closed_plane_regions(
    points, *, unit, position_error_bound, neighbor_counts, max_neighborhood_radius,
    max_normal_angle_bound, min_region_points, max_regions, graph_neighbor_count,
    max_link_distance, max_plane_evaluations,
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
    graph = build_mutual_position_graph(
        points, unit=unit, position_error_bound=position_error_bound,
        neighbor_count=graph_neighbor_count, max_link_distance=max_link_distance,
    )
    count, p, epsilon = _input(points, unit, position_error_bound)
    result = {
        "schema_version": 1, "kind": "CONE_CLOSED_PLANE_REGIONS_V1",
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
            "epsilon_expands_observed_graph": False, "final_error_bound_is_relaxed": False,
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
            "OBSERVED_KNN_TIES_AND_GRAPH_STILL_NOISE_DENSITY_DEPENDENT",
            "FINAL_REGION_CAP_NOT_THE_V1_RAW_SEED_CAP",
            "NEARBY_COPLANAR_OBJECTS_CAN_SHARE_ONE_SAMPLE_WITNESS",
            "NO_PHYSICAL_MEMBERSHIP_CONTINUITY_TANGENCY_OR_BEND_COUNT",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ],
    }
    if not graph["search_complete_within_declared_policy"]:
        return _stop(result, "OBSERVED_GRAPH_INCOMPLETE")
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
