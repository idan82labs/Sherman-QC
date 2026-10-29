"""Uncertainty-preserving plane-pair to radius relaxation, research only.

Three position regions and their common-axis relation are stipulated, not found
or verified. A witness under the inflated budget is ONLY a rotated/error-relaxed
sample witness, never a witness of the original joint plane/cylinder model.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.axis_conditioned_radius import analyze_axis_conditioned_radius
from research.geometry.position_axis_hypothesis import analyze_position_axis_hypothesis


MAX_POINTS = 256  # Shared total, before conversion or exact-repeat removal.
_MACHINE_EPS = float(np.finfo(float).eps)


def _real(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def _number(value, name, *, allow_zero=False):
    if not _real(value):
        raise ValueError(f"{name} must be an explicit finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a finite number") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _count(points):
    if points is None:
        return 0
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise ValueError("each region must be an explicit bounded Nx3 array or None")
    if isinstance(points, np.ndarray) and (points.ndim != 2 or points.shape[1] != 3):
        raise ValueError("each region must have shape Nx3")
    return len(points)


def _positions(points, count):
    if count == 0:
        return np.empty((0, 3), dtype=float)
    if isinstance(points, (list, tuple)):
        for row in points:
            if (not isinstance(row, (list, tuple, np.ndarray))
                    or isinstance(row, np.ndarray) and row.ndim != 1
                    or len(row) != 3 or not all(_real(value) for value in row)):
                raise ValueError("each position must contain exactly three real numbers")
    try:
        raw = np.asarray(points)
        if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("each region must be a numeric Nx3 array")
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("positions must be finite real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError("positions must be finite")
    return result


def _pivot_and_footprint(points):
    """Compute a finite centroid pivot without summing large global coordinates."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offsets = points - points[0]
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale == 0:
            return None
        pivot = points[0] + scale * np.mean(offsets / scale, axis=0)
        # Measure the footprint about the ACTUALLY RETURNED floating-point pivot.
        distances = np.linalg.norm((points - pivot) / scale, axis=1) * scale
        footprint = float(np.max(distances))
        numerical = (128 * _MACHINE_EPS * float(np.max(np.abs(points)))
                     + 128 * _MACHINE_EPS * scale + 128 * _MACHINE_EPS * footprint)
    if not np.isfinite(pivot).all() or not math.isfinite(footprint) or not math.isfinite(numerical):
        return None
    return pivot, footprint, numerical


def _relaxed_diagnostic(diagnostic):
    """Do not export a rotated surrogate's fitted line as physical axis geometry."""
    circle = diagnostic["circle_witness"]
    plane = diagnostic["parallel_plane_witness"]
    return {
        "source_kind": diagnostic["kind"], "status": diagnostic["status"],
        "scope": "ROTATED_ERROR_RELAXATION_ONLY_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
        "radius_interval": list(diagnostic["radius_interval"]),
        "input_count": diagnostic["input_count"], "unique_count": diagnostic["unique_count"],
        "position_error_bound": diagnostic["position_error_bound"],
        "numerical_allowance": diagnostic["numerical_allowance"],
        "effective_error_bound": diagnostic["effective_error_bound"],
        "triangle_constraint_count": diagnostic["triangle_constraint_count"],
        "circle_witness": None if circle is None else {
            key: circle[key] for key in ("radius", "max_radial_residual", "rms_radial_residual",
                                        "compatible_with_error_bound")
        },
        "parallel_plane_witness": None if plane is None else {
            key: plane[key] for key in ("width", "max_residual", "compatible_with_error_bound")
        },
        "axial_sample_groups": [{
            "sample_min": group["sample_min"], "sample_max": group["sample_max"],
            "point_count": group["point_count"],
            "scope": "OBSERVED_PROJECTIONS_ONLY", "physical_support_is_verified": False,
            "group_membership_is_verified_for_true_axis": False,
        } for group in diagnostic["axial_sample_groups"]],
        "reasons": list(diagnostic["reasons"]),
    }


def analyze_plane_conditioned_radius(
    first_neighborhood, second_neighborhood, cylinder_region, *, unit,
    position_error_bound, max_axis_angle_bound, max_axial_gap, max_radius_width,
):
    """Return conditional radius outer constraints without an exact-axis input.

    One simultaneous Euclidean error budget applies to ALL three position inputs.
    The two ideal planes and the cylinder's direction are ASSUMED to share a
    parallel axis. A positive result is a relaxed radius hypothesis, not joint
    compatibility at the original budget, surface identity or conformance.
    Angular policy is in radians; all length policies use the explicit unit.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    angle_limit = _number(max_axis_angle_bound, "max_axis_angle_bound")
    if angle_limit >= math.pi/2:
        raise ValueError("max_axis_angle_bound must be less than pi/2 radians")
    gap = _number(max_axial_gap, "max_axial_gap")
    width = _number(max_radius_width, "max_radius_width")
    counts = [_count(region) for region in (first_neighborhood, second_neighborhood, cylinder_region)]
    if sum(counts) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} total input positions are permitted across all regions")
    raw = [_positions(region, count) for region, count in zip(
        (first_neighborhood, second_neighborhood, cylinder_region), counts
    )]
    points = np.unique(raw[2], axis=0)
    axis_analysis = analyze_position_axis_hypothesis(
        raw[0], raw[1], unit=unit, position_error_bound=epsilon,
        max_axis_angle_bound=angle_limit,
    )
    result = {
        "schema_version": 1, "kind": "PLANE_CONDITIONED_RADIUS_V1", "status": "ABSTAINED",
        "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": sum(counts),
        "input_counts": dict(zip(("first_neighborhood", "second_neighborhood", "cylinder_region"), counts)),
        "position_error_bound": epsilon, "radius_interval": [0.0, None],
        "radius_interval_scope": "NECESSARY_OUTER_CONSTRAINT_UNDER_ORIGINAL_JOINT_ASSUMPTIONS",
        "original_joint_compatibility": "UNESTABLISHED",
        "policy": {"max_axis_angle_bound_radians": angle_limit,
                   "max_axial_gap": gap, "max_radius_width": width},
        "hypothesis": {
            "axis": axis_analysis["hypothesis"]["axis"],
            "axis_angle_bound_radians": axis_analysis["hypothesis"]["axis_angle_bound_radians"],
            "axis_is_verified": False, "regions_are_verified": False,
            "common_axis_relation_is_verified": False, "original_joint_model_is_verified": False,
            "physical_support_is_verified": False, "axis_location": None,
            "model": "CYLINDER_DIRECTION_PARALLEL_TO_TWO_STIPULATED_IDEAL_PLANES",
        },
        "axis_analysis": axis_analysis, "uncertainty_propagation": None,
        "relaxed_radius_diagnostic": None,
        "limitations": [
            "SUPPLIED_REGIONS_NOT_REGION_DISCOVERY", "COMMON_AXIS_RELATION_STIPULATED_NOT_VERIFIED",
            "SAME_POSITION_FITS_NOT_INDEPENDENT_EVIDENCE", "POSITION_ERROR_BOUND_UNCALIBRATED",
            "RADIUS_OUTER_CONSTRAINTS_NOT_EXACT_FEASIBLE_SET", "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "RELAXED_WITNESS_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
            "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY", "NO_CONTINUITY_OR_PHYSICAL_EXTENT_CLAIM",
            "NO_PHYSICAL_AXIS_LOCATION_OR_MARKER", "NO_FORMATION_THICKNESS_OR_CONFORMANCE_CLAIM",
            "RESOLUTION_POLICIES_NOT_PHYSICAL_QUALIFICATION",
        ],
        "reasons": [],
    }
    if len(points) < 3:
        result["reasons"].append("MISSING_CYLINDER_REGION" if len(points) == 0
                                 else "INSUFFICIENT_DISTINCT_CYLINDER_POSITIONS")
    if axis_analysis["status"] != "BOUNDED_DIRECTION_HYPOTHESIS":
        result["reasons"].extend(f"AXIS_{reason}" for reason in axis_analysis["reasons"])
    if result["reasons"]:
        return result

    footprint_data = _pivot_and_footprint(points)
    if footprint_data is None:
        result["reasons"] = ["BRIDGE_NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    pivot, footprint, numerical = footprint_data
    delta = axis_analysis["hypothesis"]["axis_angle_bound_radians"]
    chord = 2 * math.sin(delta/2)
    true_footprint = footprint + epsilon
    rotation_displacement = chord * true_footprint
    propagated = epsilon + rotation_displacement
    radius_budget = propagated + numerical
    if not all(math.isfinite(value) for value in (true_footprint, rotation_displacement, propagated, radius_budget)):
        result["reasons"] = ["BRIDGE_NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    diagnostic = analyze_axis_conditioned_radius(
        raw[2], axis=axis_analysis["hypothesis"]["axis"], unit=unit,
        position_error_bound=radius_budget, max_axial_gap=gap, max_radius_width=width,
    )
    result["radius_interval"] = list(diagnostic["radius_interval"])
    result["relaxed_radius_diagnostic"] = _relaxed_diagnostic(diagnostic)
    result["uncertainty_propagation"] = {
        "pivot": pivot.tolist(), "observed_footprint_radius": footprint,
        "true_counterpart_footprint_bound": true_footprint,
        "axis_angle_bound_radians": delta, "rotation_chord_bound": chord,
        "rotation_displacement_bound": rotation_displacement,
        "propagated_position_error_bound": propagated,
        "bridge_numerical_allowance": numerical, "radius_input_error_bound": radius_budget,
        "radius_numerical_allowance": diagnostic["numerical_allowance"],
        "radius_effective_error_bound": diagnostic["effective_error_bound"],
        # The larger spatial rotation budget also bounds each true-axis scalar
        # coordinate relative to this common pivot and a sign-aligned true axis.
        "axial_sample_coordinate_error_bound": diagnostic["effective_error_bound"],
        "axial_coordinate_scope": "SAMPLES_ONLY_COMMON_PIVOT_SIGN_ALIGNED_TRUE_AXIS",
        "physical_support_is_verified": False,
    }
    result["reasons"] = [f"RELAXED_RADIUS_{reason}" for reason in diagnostic["reasons"]]
    if diagnostic["status"] == "SAMPLE_CIRCLE_WITNESS":
        result["status"] = "CONDITIONAL_RADIUS_HYPOTHESIS"
    return result
