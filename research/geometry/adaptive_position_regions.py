"""Bounded adaptive observed neighborhoods and sampled graphs, research only.

The kth OTHER observed distance determines each local radius; all distance ties
are included. These are declared sampling policies, not density-invariant shape
estimators or physically calibrated gaps. They do not replace or modify V1.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.position_axis_hypothesis import _count, _fit_plane, _number, _positions
from research.geometry.position_plane_regions import MAX_POINTS, MAX_REGIONS, _distances, _integer


MAX_NEIGHBOR_COUNTS = 4


def _counts(values):
    if (not isinstance(values, (list, tuple, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1
            or not 1 <= len(values) <= MAX_NEIGHBOR_COUNTS):
        raise ValueError("neighbor_counts must contain one to four explicit counts")
    counts = [_integer(value, "neighbor_count", 2, MAX_POINTS - 1) for value in values]
    if len(set(counts)) != len(counts):
        raise ValueError("neighbor_counts must be unique")
    return sorted(counts)


def _input(points, unit, position_error_bound):
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    count = _count(points)
    if count > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} input positions are permitted")
    return count, np.unique(_positions(points, count), axis=0), epsilon


def _within(values, radius, allowance):
    # Difference comparison avoids overflow in radius + allowance.
    return (values <= radius) | (values - radius <= allowance)


def _rank_radius(row, seed, count):
    other = np.concatenate((row[:seed], row[seed + 1:]))
    return float(np.partition(other, count - 1)[count - 1])


def discover_adaptive_position_plane_regions(
    points, *, unit, position_error_bound, neighbor_counts, max_neighborhood_radius,
    max_normal_angle_bound, min_region_points, max_regions,
):
    """Enumerate complete-tie kth-neighbor balls at every unique sample.

    Each k counts OTHER samples, excluding the seed after exact deduplication.
    A query with too few samples, or kth distance above the explicit nominal
    radius cap plus one numerical guard, is skipped in full. No smaller k,
    clipped support, fitted error, residual trimming or greedy ownership is
    substituted. Subsequent tie closure can extend the cap by a second guard;
    epsilon does not enlarge the cap or the observed ball.
    """
    counts = _counts(neighbor_counts)
    radius_cap = _number(max_neighborhood_radius, "max_neighborhood_radius")
    angle = _number(max_normal_angle_bound, "max_normal_angle_bound")
    if angle >= math.pi / 2:
        raise ValueError("max_normal_angle_bound must be less than pi/2 radians")
    minimum = _integer(min_region_points, "min_region_points", 3, MAX_POINTS)
    region_cap = _integer(max_regions, "max_regions", 1, MAX_REGIONS)
    count, canonical, epsilon = _input(points, unit, position_error_bound)
    result = {
        "schema_version": 1, "kind": "ADAPTIVE_POSITION_PLANE_REGIONS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "input_count": count, "unique_count": len(canonical),
        "positions": canonical.tolist(), "position_error_bound": epsilon,
        "distance_numerical_allowance": 0.0,
        "rank_membership_uncertainty_allowance": 0.0,
        "radius_cap_uncertainty_allowance": 0.0,
        "search_complete_within_declared_policy": False,
        "policy": {
            "neighbor_counts": counts, "neighbor_count_excludes_seed": True,
            "max_neighborhood_radius": radius_cap,
            "radius_cap_scope": "NOMINAL_OBSERVED_KTH_DISTANCE_WITH_ONE_NUMERICAL_GUARD",
            "tie_closed_support_cap": "MAX_NEIGHBORHOOD_RADIUS_PLUS_AT_MOST_TWO_NUMERICAL_GUARDS",
            "epsilon_expands_membership_or_cap": False,
            "max_normal_angle_bound_radians": angle,
            "min_region_points": minimum, "max_regions": region_cap,
            "seed_policy": "EVERY_UNIQUE_POSITION_AT_EVERY_DECLARED_NEIGHBOR_COUNT",
            "membership_policy": "COMPLETE_OBSERVED_KTH_DISTANCE_BALL_WITH_NUMERICAL_TIES",
            "support_deduplication": "EXACT_INDEX_TUPLES_ONLY",
            "rank_boundary_policy": "ABS_DISTANCE_MINUS_KTH_DISTANCE_LE_4_EPSILON_PLUS_2_GUARDS",
            "membership_is_physically_verified": False,
            "residual_trimming": False, "greedy_removal_or_merging": False,
        },
        "seed_query_count": len(canonical) * len(counts),
        "unique_support_query_count": 0, "queries": [], "regions": [], "reasons": [],
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "DECLARED_OBSERVED_SEARCH_NOT_ALL_ERROR_PERTURBED_SUPPORTS",
            "KNN_SCALE_DEPENDS_ON_DENSITY_ANISOTROPY_AND_OTHER_SURFACES",
            "NUMERICAL_TIE_CLOSURE_NOT_DIRECTED_ROUNDING_CERTIFICATE",
            "REGION_MEMBERSHIP_AND_PHYSICAL_FLATNESS_UNVERIFIED",
            "POSITION_ERROR_BOUND_UNCALIBRATED",
            "OVERLAPPING_AND_NESTED_HYPOTHESES_NOT_PHYSICAL_INSTANCES",
            "NO_CONTINUITY_FORMATION_RADIUS_THICKNESS_OR_BEND_COUNT_CLAIM",
        ],
    }
    if len(canonical) == 0:
        result["search_complete_within_declared_policy"] = True
        result["reasons"] = ["INSUFFICIENT_DISTINCT_POSITIONS"]
        return result
    distances, guard = _distances(canonical)
    rank_error, cap_error = 4 * epsilon + 2 * guard, 2 * epsilon + guard
    if distances is None or not math.isfinite(rank_error) or not math.isfinite(cap_error):
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    result.update(distance_numerical_allowance=guard,
                  rank_membership_uncertainty_allowance=rank_error,
                  radius_cap_uncertainty_allowance=cap_error)
    supports = {}
    for seed, row in enumerate(distances):
        for k in counts:
            query = {"seed_index": seed, "neighbor_count": k, "radius": None,
                     "status": "INSUFFICIENT_OTHER_POSITIONS", "support_count": 0,
                     "rank_boundary_sensitive_count": 0, "radius_cap_is_sensitive": False}
            result["queries"].append(query)
            if len(canonical) <= k:
                continue
            radius = _rank_radius(row, seed, k)
            query.update(radius=radius,
                         radius_cap_is_sensitive=bool(abs(radius - radius_cap) <= cap_error))
            if radius > radius_cap and radius - radius_cap > guard:
                query["status"] = "NOMINAL_RADIUS_CAP_EXCEEDED"
                continue
            indices = tuple(int(index) for index in np.flatnonzero(_within(row, radius, guard)))
            boundary = np.abs(row - radius) <= rank_error
            boundary[seed] = False
            query.update(status="QUERIED", support_count=len(indices),
                         rank_boundary_sensitive_count=int(np.count_nonzero(boundary)))
            supports.setdefault(indices, []).append(query)
    result["unique_support_query_count"] = len(supports)
    regions = []
    for indices in sorted(supports):
        if len(indices) < minimum:
            continue
        plane = _fit_plane(canonical[list(indices)], epsilon)
        if any(reason in ("NUMERICAL_RANGE_UNSUPPORTED", "PLANE_FIT_UNAVAILABLE")
               for reason in plane["reasons"]):
            result["reasons"] = ["REGION_PLANE_ANALYSIS_UNAVAILABLE"]
            return result
        if (plane["reasons"] or plane["normal_angle_bound_radians"] is None
                or plane["normal_angle_bound_radians"] > angle):
            continue
        regions.append({"region_id": len(regions), "point_indices": list(indices),
                        "plane_analysis": plane, "seed_queries": supports[indices]})
        if len(regions) > region_cap:
            result["reasons"] = ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
            return result
    result["regions"] = regions
    result["search_complete_within_declared_policy"] = True
    if regions:
        result["status"] = "PLANE_REGION_HYPOTHESES"
    else:
        result["reasons"] = ["NO_PLANE_REGION_HYPOTHESIS_WITHIN_DECLARED_POLICY"]
    return result


def build_mutual_position_graph(
    points, *, unit, position_error_bound, neighbor_count, max_link_distance,
):
    """All mutual complete-tie observed-kNN edges below an observed cap.

    Neither epsilon nor fit residuals enlarge adjacency. The error bound is used
    ONLY to disclose unstable rank/cap decisions. The explicit link cap is not a
    physical gap claim. A sample may have more than k neighbors because ties are
    kept. The 256-row input bound limits the output to at most 32,640 edges.
    """
    k = _integer(neighbor_count, "neighbor_count", 2, MAX_POINTS - 1)
    cap = _number(max_link_distance, "max_link_distance")
    count, canonical, epsilon = _input(points, unit, position_error_bound)
    result = {
        "schema_version": 1, "kind": "MUTUAL_OBSERVED_POSITION_GRAPH_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "input_count": count, "unique_count": len(canonical),
        "positions": canonical.tolist(), "position_error_bound": epsilon,
        "neighbor_count": k, "max_link_distance": cap,
        "distance_numerical_allowance": 0.0,
        "rank_membership_uncertainty_allowance": 0.0,
        "link_cap_uncertainty_allowance": 0.0,
        "search_complete_within_declared_policy": False,
        "scope": "MUTUAL_OBSERVED_NEIGHBORHOOD_SAMPLED_LINKS_ONLY",
        "link_cap_scope": "OBSERVED_POLICY_CAP_WITH_NUMERICAL_GUARD",
        "epsilon_expands_edges": False, "neighbor_count_excludes_seed": True,
        "joint_realizability_is_verified": False,
        "physical_connectivity_is_verified": False,
        "neighborhood_membership_is_physically_verified": False,
        "local_radii": [], "edges": [], "edge_count": 0,
        "diagnostics": {"directed_neighbor_count": 0, "nonmutual_neighbor_count": 0,
                        "mutual_pairs_beyond_cap": 0,
                        "rank_boundary_sensitive_directed_count": 0,
                        "link_cap_sensitive_pair_count": 0},
        "reasons": [],
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "DECLARED_OBSERVED_GRAPH_NOT_ALL_ERROR_PERTURBED_GRAPHS",
            "MUTUAL_KNN_CAN_FRAGMENT_ONE_SURFACE_OR_CONNECT_DISTINCT_SURFACES",
            "OBSERVED_LINK_CAP_NOT_PHYSICAL_CONTACT_OR_TRUE_GAP",
            "DENSITY_ANISOTROPY_AND_OTHER_SURFACES_CAN_CHANGE_GRAPH",
            "NO_JOINT_TOPOLOGY_CONTINUITY_OR_PHYSICAL_COUNT_CLAIM",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ],
    }
    if len(canonical) <= k:
        result["search_complete_within_declared_policy"] = True
        result["reasons"] = ["INSUFFICIENT_OTHER_POSITIONS"]
        return result
    distances, guard = _distances(canonical)
    rank_error, cap_error = 4 * epsilon + 2 * guard, 2 * epsilon + guard
    if distances is None or not math.isfinite(rank_error) or not math.isfinite(cap_error):
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    radii = np.asarray([_rank_radius(row, seed, k) for seed, row in enumerate(distances)])
    directed = _within(distances, radii[:, None], guard)
    np.fill_diagonal(directed, False)
    rank_sensitive = np.abs(distances - radii[:, None]) <= rank_error
    np.fill_diagonal(rank_sensitive, False)
    mutual = directed & directed.T
    in_cap = _within(distances, cap, guard)
    adjacency = mutual & in_cap
    edges = np.transpose(np.nonzero(np.triu(adjacency, 1))).tolist()
    result.update(
        status="OBSERVED_SAMPLED_GRAPH", search_complete_within_declared_policy=True,
        distance_numerical_allowance=guard,
        rank_membership_uncertainty_allowance=rank_error,
        link_cap_uncertainty_allowance=cap_error,
        local_radii=radii.tolist(), edges=edges, edge_count=len(edges),
    )
    result["diagnostics"] = {
        "directed_neighbor_count": int(np.count_nonzero(directed)),
        "nonmutual_neighbor_count": int(np.count_nonzero(directed & ~directed.T)),
        "mutual_pairs_beyond_cap": int(np.count_nonzero(np.triu(mutual & ~in_cap, 1))),
        "rank_boundary_sensitive_directed_count": int(np.count_nonzero(rank_sensitive)),
        "link_cap_sensitive_pair_count": int(np.count_nonzero(np.triu(
            np.abs(distances - cap) <= cap_error, 1))),
    }
    if not edges:
        result["reasons"] = ["NO_MUTUAL_OBSERVED_LINKS_WITHIN_DECLARED_CAP"]
    return result
