"""Necessary same-face tangent-plane exclusion, not physical bend verification.

Every ideal flange centroid must lie outside/on its tangent cylinder's projected
disk. A bounded algebraic center enclosure can exclude that relation even when
the earlier independent plane/direction/radius relaxation succeeds. No fit is
snapped to tangency, no sample is trimmed, and passing is not joint feasibility.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.axis_conditioned_radius import _axis_basis
from research.geometry.plane_conditioned_radius import (
    MAX_POINTS, _count, _positions, analyze_plane_conditioned_radius,
)
from research.geometry.position_bend_candidates import _finite_payload


_EPS = float(np.finfo(float).eps)


def _center_ball(q, error, radius_upper):
    """All-point necessary center enclosure in one normalized 2D frame.

    For radial errors <= E and true R <= Rhi, square-distance discrepancies are
    bounded by D=2*Rhi*E+E**2. Centering cannot enlarge their Euclidean norm.
    A=2*Hq, b=H||q||**2, r=A*c_hat-b imply
    ||c-c_hat|| <= (sqrt(N)*D + ||r||)/sigma_min(A).
    Numerical padding is engineering arithmetic, not directed rounding.
    """
    if (q.ndim != 2 or q.shape[1] != 2 or not 3 <= len(q) <= MAX_POINTS
            or not np.isfinite(q).all() or not math.isfinite(error) or error < 0
            or not math.isfinite(radius_upper) or radius_upper <= 0):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = q - np.mean(q, axis=0)
        squares = np.sum(q*q, axis=1)
        matrix = 2*centered
        rhs = squares-np.mean(squares)
        discrepancy = 2*radius_upper*error+error*error
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all() or not math.isfinite(discrepancy):
        return None
    try:
        center, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank != 2 or not np.isfinite(center).all() or not np.isfinite(singular).all():
        return None
    root_n = math.sqrt(len(q))
    singular_guard = 256*_EPS*root_n*max(1.0, float(singular[0]))
    denominator = float(singular[-1])-singular_guard
    if denominator <= 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        residual = float(np.linalg.norm(matrix @ center-rhs))
        arithmetic = 256*_EPS*root_n*(1+float(np.linalg.norm(rhs))
                      +float(np.linalg.norm(matrix))*float(np.linalg.norm(center)))
        radius = (root_n*discrepancy+residual+arithmetic)/denominator
    if not all(math.isfinite(x) for x in (residual, arithmetic, radius)):
        return None
    return dict(center=center.tolist(), error_radius=radius,
                square_discrepancy_bound=discrepancy, algebraic_residual_norm=residual,
                minimum_singular_value=float(singular[-1]),
                singular_value_numerical_allowance=singular_guard,
                residual_numerical_allowance=arithmetic, all_point_count=len(q))


def _assess_same_face_tangency(first, second, cylinder, radius_analysis):
    """INTERNAL reuse of this engine's radius analysis; no external evidence API."""
    result = {
        "schema_version": 1, "kind": "SAME_FACE_TANGENCY_NECESSARY_CHECK_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": radius_analysis["unit"], "analysis_available": False,
        "single_same_face_tangent_model_excluded": False,
        "two_radius_or_multiple_surface_model_is_excluded": False,
        "original_joint_compatibility": "UNESTABLISHED",
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "tangency_is_verified": False, "contact_is_verified": False,
        "physical_axis_location_is_verified": False,
        "radius_interval": list(radius_analysis["radius_interval"]),
        "radius_interval_scope": radius_analysis["radius_interval_scope"],
        "position_error_bound": radius_analysis["position_error_bound"],
        "center_enclosure": None, "flange_checks": [], "reasons": [],
        "policy": {
            "model": "ONE_SAME_FACE_CIRCULAR_CYLINDER_TANGENT_TO_BOTH_STIPULATED_PLANES",
            "test": "ALL_POINT_FLANGE_CENTROID_OUTSIDE_PROJECTED_CYLINDER_DISK",
            "minimum_contact_sample_count": 0,
            "candidate_residual_selection_or_trimming": False,
            "centroid_error_averages_away": False,
            "center_solver_limit": 1, "flange_centroid_check_limit": 2,
            "total_raw_position_limit": MAX_POINTS,
            "source_radius_fit_is_replaced": False,
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION", "POSITION_ERROR_BOUND_UNCALIBRATED",
            "BOUND_NOT_NUMERICALLY_CERTIFIED", "SINGLE_SAME_FACE_MODEL_ONLY",
            "PLANE_AND_CYLINDER_MEMBERSHIP_STIPULATED_NOT_PHYSICALLY_VERIFIED",
            "CENTER_BALL_OUTER_RELAXATION_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
            "NOT_EXCLUDED_IS_NOT_TANGENCY_OR_CONTACT_OR_GLOBAL_FEASIBILITY",
            "TANGENT_INFINITE_PLANE_NOT_OBSERVED_FINITE_CONTACT",
            "OUTSIDE_CENTROIDS_CAN_HIDE_SECANT_PLANES_AND_DETACHED_CIRCLES",
            "INNER_OUTER_MIXTURES_AND_MULTIPLE_FACES_NOT_EXCLUDED_BY_THIS_CHECK",
            "NO_PHYSICAL_AXIS_LOCATION_FORMATION_THICKNESS_OR_CONFORMANCE",
        ],
    }
    if radius_analysis["status"] != "CONDITIONAL_RADIUS_HYPOTHESIS":
        result["reasons"] = ["CONDITIONAL_SINGLE_RADIUS_REQUIRED"]
        return result
    lower, upper = result["radius_interval"]
    if upper is None or not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        result["reasons"] = ["FINITE_POSITIVE_RADIUS_ENCLOSURE_REQUIRED"]
        return result
    regions = [np.unique(region, axis=0) for region in (first, second, cylinder)]
    if any(len(region) < 3 for region in regions):
        result["reasons"] = ["INSUFFICIENT_DISTINCT_REGION_POSITIONS"]
        return result
    propagation = radius_analysis["uncertainty_propagation"]
    pivot = np.asarray(propagation["pivot"], dtype=float)
    _, basis = _axis_basis(radius_analysis["hypothesis"]["axis"])
    epsilon = result["position_error_bound"]
    chord = propagation["rotation_chord_bound"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cylinder_offsets = regions[2]-pivot
        scale = float(np.max(np.abs(cylinder_offsets)))
    if not math.isfinite(scale) or scale <= 0:
        result["reasons"] = ["TANGENCY_NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude = max(float(np.max(np.abs(region))) for region in regions)
        projection_guard = 256*_EPS*(magnitude+float(np.max(np.abs(pivot)))+scale)
        error = propagation["radius_effective_error_bound"]+projection_guard
        q = (cylinder_offsets/scale) @ basis
        normalized_error, normalized_upper = error/scale, upper/scale
    if (not math.isfinite(scale) or scale <= 0 or not np.isfinite(q).all()
            or not all(math.isfinite(x) for x in (error, normalized_error, normalized_upper))):
        result["reasons"] = ["TANGENCY_NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    ball = _center_ball(q, normalized_error, normalized_upper)
    if ball is None:
        result["reasons"] = ["CENTER_ENCLOSURE_UNAVAILABLE"]
        return result
    center = np.asarray(ball["center"])
    center_error = ball["error_radius"]*scale
    result["center_enclosure"] = {
        "scope": "ROTATED_SURROGATE_PROJECTED_CENTER_OUTER_BALL_NOT_PHYSICAL_AXIS_LOCATION",
        "pivot": pivot.tolist(), "projection_basis": basis.tolist(),
        "projected_center": (center*scale).tolist(), "error_radius": center_error,
        "cylinder_projection_error_bound": error,
        "projection_numerical_allowance": projection_guard,
        "normalization_scale": scale, "normalized_diagnostics": ball,
        "all_point_count": len(regions[2]),
    }
    for index, region in enumerate(regions[:2]):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            offsets = region-pivot
            local_scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(local_scale) or local_scale <= 0:
            result.update(center_enclosure=None, flange_checks=[], reasons=["TANGENCY_NUMERICAL_RANGE_UNSUPPORTED"])
            return result
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # The corrected mean of ALL ideal flange points lies in its plane.
            # Bound the SAME rotation about the cylinder pivot, including a
            # correlated centroid displacement of epsilon, never epsilon/sqrt(N).
            mean_offset = np.mean(offsets/local_scale, axis=0)*local_scale
            centroid_error = epsilon+chord*(math.hypot(*mean_offset)+epsilon)
            centroid_guard = projection_guard+256*_EPS*local_scale
            projected_centroid = (mean_offset/scale) @ basis
            distance = math.hypot(*(projected_centroid-center))*scale
            upper_distance = distance+center_error+centroid_error+centroid_guard
        if (not math.isfinite(local_scale) or local_scale <= 0
                or not np.isfinite(mean_offset).all()
                or not all(math.isfinite(x) for x in (centroid_error, centroid_guard, upper_distance))):
            result.update(center_enclosure=None, flange_checks=[], reasons=["TANGENCY_NUMERICAL_RANGE_UNSUPPORTED"])
            return result
        result["flange_checks"].append({
            "flange_index": index, "all_point_count": len(region),
            "observed_centroid_offset_from_pivot": mean_offset.tolist(),
            "observed_projected_centroid_center_distance": distance,
            "centroid_original_position_error_bound": epsilon,
            "rotated_centroid_error_bound": centroid_error,
            "centroid_numerical_allowance": centroid_guard,
            "true_projected_centroid_center_distance_upper_bound": upper_distance,
            "necessary_minimum_distance_for_tangency": lower,
            "centroid_is_robustly_inside_every_admissible_disk": upper_distance < lower,
        })
    if not _finite_payload(result):
        result.update(center_enclosure=None, flange_checks=[], reasons=["TANGENCY_NUMERICAL_RANGE_UNSUPPORTED"])
        return result
    excluded = any(item["centroid_is_robustly_inside_every_admissible_disk"] for item in result["flange_checks"])
    result.update(analysis_available=True, single_same_face_tangent_model_excluded=excluded,
                  status="SINGLE_SAME_FACE_TANGENT_MODEL_EXCLUDED" if excluded else "TANGENCY_NOT_EXCLUDED",
                  reasons=["FLANGE_CENTROID_INSIDE_NECESSARY_CYLINDER_DISK"] if excluded else [])
    return result


def analyze_same_face_tangency(
    first_neighborhood, second_neighborhood, cylinder_region, *, unit,
    position_error_bound, max_axis_angle_bound, max_axial_gap, max_radius_width,
):
    """Stipulated regions and one common error bound, not inferred face labels.

    Exclusion concerns ONLY an additional same-face tangency relation. It does
    not invalidate a cylinder radius, exclude two-radius/multiple-face models,
    establish a defect, or prove a different physical explanation.
    """
    sources = (first_neighborhood, second_neighborhood, cylinder_region)
    counts = [_count(region) for region in sources]
    if sum(counts) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} total input positions are permitted")
    regions = [_positions(region, count) for region, count in zip(sources, counts)]
    radius = analyze_plane_conditioned_radius(
        *regions, unit=unit, position_error_bound=position_error_bound,
        max_axis_angle_bound=max_axis_angle_bound, max_axial_gap=max_axial_gap,
        max_radius_width=max_radius_width,
    )
    return _assess_same_face_tangency(*regions, radius)
