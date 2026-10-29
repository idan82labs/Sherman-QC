"""Exhaustive bounded observed-ball plane-region hypotheses, research only.

The input contains positions, not preselected plane labels. This is an explicit
local search policy, not complete physical segmentation: overlapping and nested
sample supports are retained, and a curved patch can be plane-compatible at the
declared error. All error and angular budgets remain caller stipulations.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.position_axis_hypothesis import _count, _fit_plane, _number, _positions


MAX_POINTS = 256
MAX_RADII = 4
MAX_REGIONS = 64
_MACHINE_EPS = float(np.finfo(float).eps)
_MIN_NORMAL = float(np.finfo(float).tiny)


def _integer(value, name, minimum, maximum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return int(value)


def _radii(values):
    if (not isinstance(values, (list, tuple, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1
            or not 1 <= len(values) <= MAX_RADII):
        raise ValueError("neighborhood_radii must contain one to four explicit radii")
    radii = [_number(value, "neighborhood_radius") for value in values]
    if len(set(radii)) != len(radii):
        raise ValueError("neighborhood_radii must be unique")
    return sorted(radii)


def _distances(points):
    """Use scaled differences, avoiding overflow in subtraction and squaring.

    The allowance includes the original coordinate magnitude because subtracting
    or normalizing translated coordinates can lose low-order bits. It is a
    deliberately disclosed engineering guard, not directed-rounding arithmetic.
    """
    if not len(points):
        return np.empty((0, 0), dtype=float), 0.0
    magnitude = float(np.max(np.abs(points)))
    if magnitude == 0:
        return np.zeros((len(points), len(points)), dtype=float), 0.0
    if magnitude < _MIN_NORMAL:
        return None, 0.0
    normalized = points / magnitude
    delta = normalized[:, None, :] - normalized[None, :, :]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distances = np.sqrt(np.sum(delta * delta, axis=2)) * magnitude
    if not np.isfinite(distances).all():
        return None, 0.0
    allowance = (128 * _MACHINE_EPS * magnitude
                 + 128 * _MACHINE_EPS * float(np.max(distances)))
    if not math.isfinite(allowance) or allowance == 0:
        return None, 0.0
    return distances, allowance


def discover_position_plane_regions(
    points, *, unit, position_error_bound, neighborhood_radii,
    max_normal_angle_bound, min_region_points, max_regions,
):
    """Enumerate sample plane hypotheses without supplied labels or normals.

    Every canonical unique position is a seed at every declared radius. A seed's
    support is its nominal observed Euclidean ball (including the disclosed
    numerical guard); it is never residual-trimmed. Only EXACT support tuples
    are deduplicated. Neither local error compatibility nor the number of these
    overlapping hypotheses establishes physical planes, flanges, or bend count.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    radii = _radii(neighborhood_radii)
    angle_limit = _number(max_normal_angle_bound, "max_normal_angle_bound")
    if angle_limit >= math.pi / 2:
        raise ValueError("max_normal_angle_bound must be less than pi/2 radians")
    minimum = _integer(min_region_points, "min_region_points", 3, MAX_POINTS)
    region_limit = _integer(max_regions, "max_regions", 1, MAX_REGIONS)
    count = _count(points)
    if count > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} input positions are permitted")
    # Reject oversize payloads before NumPy conversion or exact deduplication.
    canonical = np.unique(_positions(points, count), axis=0)
    result = {
        "schema_version": 1, "kind": "POSITION_PLANE_REGIONS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "input_count": count, "unique_count": len(canonical),
        "positions": canonical.tolist(), "position_error_bound": epsilon,
        "distance_numerical_allowance": 0.0,
        "search_complete_within_declared_policy": False,
        "policy": {
            "neighborhood_radii": radii,
            "max_normal_angle_bound_radians": angle_limit,
            "min_region_points": minimum, "max_regions": region_limit,
            "seed_policy": "EVERY_UNIQUE_POSITION_AT_EVERY_DECLARED_RADIUS",
            "membership_policy": "OBSERVED_EUCLIDEAN_BALL_WITH_NUMERICAL_GUARD",
            "support_deduplication": "EXACT_INDEX_TUPLES_ONLY",
            "boundary_sensitivity_policy": "ABS_DISTANCE_MINUS_RADIUS_LE_2_EPSILON_PLUS_NUMERICAL_ALLOWANCE",
            "numerical_allowance_policy": "128_MACHINE_EPS_TIMES_SUM_OF_MAX_COORDINATE_MAGNITUDE_AND_MAX_PAIR_DISTANCE",
            "membership_is_physically_verified": False,
            "residual_trimming": False, "greedy_removal_or_merging": False,
        },
        "seed_query_count": len(canonical) * len(radii),
        "unique_support_query_count": 0,
        "regions": [], "reasons": [],
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "LOCAL_SEARCH_POLICY_NOT_COMPLETE_PHYSICAL_SEGMENTATION",
            "REGIONS_AND_NEIGHBORHOOD_MEMBERSHIP_UNVERIFIED",
            "POSITION_ERROR_BOUND_UNCALIBRATED",
            "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "OVERLAPPING_AND_NESTED_HYPOTHESES_NOT_DISTINCT_PHYSICAL_INSTANCES",
            "PLANE_COMPATIBLE_SAMPLES_MAY_LIE_ON_CURVED_OR_DISCONNECTED_SURFACES",
            "NO_SURFACE_IDENTITY_CONTINUITY_OR_FINITE_SUPPORT_CLAIM",
            "NO_BEND_COUNT_FORMATION_RADIUS_THICKNESS_OR_CONFORMANCE_CLAIM",
        ],
    }
    if len(canonical) < minimum:
        result["search_complete_within_declared_policy"] = True
        result["reasons"] = ["INSUFFICIENT_DISTINCT_POSITIONS"]
        return result
    distances, allowance = _distances(canonical)
    result["distance_numerical_allowance"] = allowance
    boundary_error = 2 * epsilon + allowance
    if distances is None or not math.isfinite(boundary_error):
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result

    queries = {}
    for seed_index, seed_distances in enumerate(distances):
        for radius in radii:
            # Comparison by nonnegative difference avoids overflow in r + guard.
            indices = tuple(int(index) for index in np.flatnonzero(
                (seed_distances <= radius) | (seed_distances - radius <= allowance)
            ))
            boundary_count = int(np.count_nonzero(
                np.abs(seed_distances - radius) <= boundary_error
            ))
            query = {"seed_index": seed_index, "radius": radius,
                     "boundary_sensitive_count": boundary_count}
            queries.setdefault(indices, []).append(query)
    result["unique_support_query_count"] = len(queries)

    regions = []
    for indices in sorted(queries):
        if len(indices) < minimum:
            continue
        plane = _fit_plane(canonical[list(indices)], epsilon)
        if any(reason in ("NUMERICAL_RANGE_UNSUPPORTED", "PLANE_FIT_UNAVAILABLE")
               for reason in plane["reasons"]):
            result["reasons"] = ["REGION_PLANE_ANALYSIS_UNAVAILABLE"]
            return result
        if (plane["reasons"]
                or plane["normal_angle_bound_radians"] is None
                or plane["normal_angle_bound_radians"] > angle_limit):
            continue
        regions.append({"region_id": len(regions), "point_indices": list(indices),
                        "plane_analysis": plane, "seed_queries": queries[indices]})
        if len(regions) > region_limit:
            # Nothing is selected from an arbitrary prefix of an exhausted cap.
            result["reasons"] = ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
            return result
    result["regions"] = regions
    result["search_complete_within_declared_policy"] = True
    if regions:
        result["status"] = "PLANE_REGION_HYPOTHESES"
    else:
        result["reasons"] = ["NO_PLANE_REGION_HYPOTHESIS_WITHIN_DECLARED_POLICY"]
    return result
