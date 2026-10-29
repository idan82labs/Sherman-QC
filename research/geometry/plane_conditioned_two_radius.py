"""Two concentric-radius sample witnesses under a plane-derived axis relaxation.

All original input points must be accounted for. Algebraic coefficients propose
one center, not a certified estimator or a complete search over feasible centers.
Band radius constraints are CONDITIONAL ON THAT PROPOSED MEMBERSHIP. They never
enclose all possible partitions, establish inner/outer face identity or thickness.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.axis_conditioned_radius import _axis_basis, analyze_axis_conditioned_radius
from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius


MAX_SPLIT_EVALUATIONS = 255
_MACHINE_EPS = float(np.finfo(float).eps)


def _integer(value, name, minimum, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return int(value)


def _unavailable(reasons):
    return any(token in reason for reason in reasons for token in (
        "NUMERICAL_RANGE_UNSUPPORTED", "PLANE_FIT_UNAVAILABLE", "NUMERICAL_ENCLOSURE_INCONSISTENT",
    ))


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", models=[], search_complete_within_declared_policy=complete,
                  reasons=[reason])
    return result


def _algebraic_center(q):
    """Fit the product of two implicit circle polynomials; proposal ONLY.

    (s-2c.q+d1)(s-2c.q+d2) = s² + A*s*x+B*s*y + quadratic/linear/constant
    terms, with A=-4cx and B=-4cy. Unconstrained least squares does NOT enforce
    the remaining concentric-circle coefficient identities. All-point geometric
    verification below, not algebraic residual, determines witness acceptance.
    """
    s = np.sum(q*q, axis=1)
    x, y = q.T
    design = np.column_stack((s*x, s*y, x*x, x*y, y*y, x, y, np.ones(len(q))))
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(design, -s*s, rcond=64*_MACHINE_EPS)
    except np.linalg.LinAlgError:
        return None, "ALGEBRAIC_CENTER_NUMERICALLY_UNAVAILABLE", None
    if not np.isfinite(coefficients).all() or not np.isfinite(singular).all():
        return None, "ALGEBRAIC_CENTER_NUMERICALLY_UNAVAILABLE", None
    if rank != 8:
        return None, "ALGEBRAIC_CENTER_RANK_UNRESOLVED", int(rank)
    return -coefficients[:2]/4, None, int(rank)


def _band_diagnostic(diagnostic, indices, witness_radius):
    # Do not export surrogate fitted 3D axes/endpoints from the fixed-axis tool.
    return {
        "point_indices": indices, "witness_radius": witness_radius,
        "radius_interval": list(diagnostic["radius_interval"]),
        "radius_interval_scope": "CONDITIONAL_ON_PROPOSED_BAND_MEMBERSHIP_AND_AXIS_ROTATION_RELAXATION",
        "position_error_bound": diagnostic["position_error_bound"],
        "numerical_allowance": diagnostic["numerical_allowance"],
        "effective_error_bound": diagnostic["effective_error_bound"],
        "status": diagnostic["status"],
        "membership_is_verified": False, "physical_face_identity_is_verified": False,
        "axial_sample_groups": [
            {"sample_min": group["sample_min"], "sample_max": group["sample_max"],
             "point_count": group["point_count"], "physical_support_is_verified": False}
            for group in diagnostic["axial_sample_groups"]
        ],
        "reasons": list(diagnostic["reasons"]),
    }


def analyze_plane_conditioned_two_radius(
    first_neighborhood, second_neighborhood, mixed_region, *, unit,
    position_error_bound, max_axis_angle_bound, max_axial_gap, max_radius_width,
    min_band_points, max_split_evaluations,
):
    """Retain all sample-compatible two-radius proposals at one algebraic center.

    Three supplied regions share the old 256 RAW row cap before conversion.
    One simultaneous position budget applies to every row. The same one-rotation
    global error relaxation is used for both proposed bands, never a smaller
    per-band footprint. The fixed-center radial split search is exhaustive only
    within its declared policy, not over all centers or physical segmentations.
    """
    minimum = _integer(min_band_points, "min_band_points", 3, 128)
    split_cap = _integer(max_split_evaluations, "max_split_evaluations", 1, MAX_SPLIT_EVALUATIONS)
    baseline = analyze_plane_conditioned_radius(
        first_neighborhood, second_neighborhood, mixed_region, unit=unit,
        position_error_bound=position_error_bound, max_axis_angle_bound=max_axis_angle_bound,
        max_axial_gap=max_axial_gap, max_radius_width=max_radius_width,
    )
    result = {
        "schema_version": 1, "kind": "PLANE_CONDITIONED_TWO_RADIUS_V1", "status": "ABSTAINED",
        "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": baseline["input_count"], "position_error_bound": baseline["position_error_bound"],
        "axis_analysis": baseline["axis_analysis"],
        "uncertainty_propagation": baseline["uncertainty_propagation"],
        "original_joint_compatibility": "UNESTABLISHED",
        "physical_face_count": None, "thickness": None,
        "physical_geometry_is_verified": False, "partition_is_unique": False,
        "partition_uniqueness": "UNESTABLISHED",
        "search_complete_within_declared_policy": False, "models": [],
        "mixed_positions": [], "single_radius_alternative": None, "flat_alternative": None,
        "policy": {"min_band_points": minimum, "max_split_evaluations": split_cap,
                   "center_proposal": "ALL_POINT_QUARTIC_LINEAR_LEAST_SQUARES",
                   "split_policy": "ALL_RADIAL_ORDER_CUTS_AT_ONE_PROPOSED_CENTER",
                   "band_order": "WITNESS_RADIUS_ORDER_NOT_INNER_OUTER_FACE_IDENTITY",
                   "max_axial_gap": float(max_axial_gap), "max_radius_width": float(max_radius_width)},
        "search_diagnostics": {"algebraic_rank": None, "radial_cut_count": 0,
                               "geometrically_compatible_cuts": 0, "band_evaluation_count": 0},
        "limitations": [
            "ALGEBRAIC_CENTER_IS_ONE_PROPOSAL_NOT_CERTIFIED_PARAMETER_ESTIMATE",
            "NO_COMPLETE_SEARCH_OVER_FEASIBLE_CENTERS_OR_PARTITIONS",
            "BAND_INTERVALS_CONDITIONAL_ON_PROPOSED_MEMBERSHIP_NOT_GLOBAL_RADIUS_BOUNDS",
            "SEPARATED_WITNESS_BANDS_DO_NOT_ESTABLISH_UNIQUE_SEGMENTATION",
            "SINGLE_PLANE_ALTERNATIVE_NOT_EXHAUSTIVE_SURFACE_MODEL_SELECTION",
            "LEGACY_CIRCLE_PROPOSAL_DOES_NOT_CLASSIFY_ALL_SOLVER_FAILURES",
            "RELAXED_WITNESS_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
            "NO_PHYSICAL_INNER_OUTER_FACE_IDENTITY_OR_THICKNESS",
            "NO_CONTINUOUS_SUPPORT_TANGENCY_OR_FORMING_STATE",
            "SAME_ALL_COMPONENT_AXIS_ROTATION_BUDGET_FOR_BOTH_BANDS",
            "NO_POINT_TRIMMING_OR_UNEXPLAINED_OUTLIER_DISCARD",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        ],
        "reasons": [],
    }
    if _unavailable(baseline["reasons"]):
        return _stop(result, "AXIS_OR_ROTATION_NUMERICAL_RANGE_UNSUPPORTED")
    propagation = baseline["uncertainty_propagation"]
    if propagation is None:
        return _stop(result, "AXIS_OR_ROTATION_HYPOTHESIS_UNRESOLVED", complete=True)
    # Deliberately omit the pooled single-radius interval: that model may be
    # incompatible with a mixture, so its enclosure says nothing about both radii.
    alternative = baseline["relaxed_radius_diagnostic"]
    result["single_radius_alternative"] = alternative["circle_witness"]
    result["flat_alternative"] = alternative["parallel_plane_witness"]
    if (alternative["circle_witness"] is not None
            and alternative["circle_witness"]["compatible_with_error_bound"]):
        return _stop(result, "SINGLE_RADIUS_ALTERNATIVE_COMPATIBLE", complete=True)
    if (alternative["parallel_plane_witness"] is not None
            and alternative["parallel_plane_witness"]["compatible_with_error_bound"]):
        return _stop(result, "FLAT_ALTERNATIVE_COMPATIBLE", complete=True)

    # The baseline validated shapes, finiteness and the shared cap before this
    # second small conversion. All exact unique rows remain accounted for.
    p = np.unique(np.asarray(mixed_region, dtype=float), axis=0)
    result["mixed_positions"] = p.tolist()
    if len(p) < 2*minimum:
        return _stop(result, "INSUFFICIENT_TWO_BAND_SAMPLES", complete=True)
    cuts = len(p)-1
    result["search_diagnostics"]["radial_cut_count"] = cuts
    if cuts > split_cap:
        return _stop(result, "RADIAL_SPLIT_BUDGET_EXCEEDED")
    axis, basis = _axis_basis(baseline["hypothesis"]["axis"])
    pivot = np.asarray(propagation["pivot"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offsets = p-pivot
        scale = float(np.max(np.abs(offsets)))
        q = (offsets/scale) @ basis
        # Use the old full-component effective budget, including its numerical
        # guard, and add a small disclosed guard for this projection's arithmetic.
        numerical = 128*_MACHINE_EPS*(float(np.max(np.abs(p)))+scale)
        spatial_budget = propagation["radius_effective_error_bound"]+numerical
        e = spatial_budget/scale
    if (not np.isfinite(q).all() or not math.isfinite(scale) or scale <= 0
            or not math.isfinite(e) or e > 1e50):
        return _stop(result, "TWO_RADIUS_NUMERICAL_RANGE_UNSUPPORTED")
    result["two_radius_numerical_allowance"] = numerical
    result["global_band_input_error_bound"] = spatial_budget
    center, error, rank = _algebraic_center(q)
    result["search_diagnostics"]["algebraic_rank"] = rank
    if error:
        return _stop(result, error, complete=error == "ALGEBRAIC_CENTER_RANK_UNRESOLVED")
    with np.errstate(over="ignore", invalid="ignore"):
        radial = np.hypot(q[:, 0]-center[0], q[:, 1]-center[1])
    if not np.isfinite(radial).all():
        return _stop(result, "TWO_RADIUS_NUMERICAL_RANGE_UNSUPPORTED")
    order = np.argsort(radial, kind="stable")
    ordered = radial[order]
    for split in range(1, len(p)):
        if split < minimum or len(p)-split < minimum:
            continue
        groups = [sorted(order[:split].tolist()), sorted(order[split:].tolist())]
        radii = [(ordered[0]+ordered[split-1])/2, (ordered[split]+ordered[-1])/2]
        if radii[0] <= 0 or radii[1]-radii[0] <= 2*e:
            continue
        residuals = [float(np.max(np.abs(radial[group]-radius)))
                     for group, radius in zip(groups, radii)]
        if max(residuals) > e:
            continue
        if any(len(np.unique(q[group], axis=0)) < 3 for group in groups):
            continue
        result["search_diagnostics"]["geometrically_compatible_cuts"] += 1
        bands, admissible = [], True
        for indices, radius in zip(groups, radii):
            diagnostic = analyze_axis_conditioned_radius(
                p[indices], axis=axis, unit=unit, position_error_bound=spatial_budget,
                max_axial_gap=max_axial_gap, max_radius_width=max_radius_width,
            )
            result["search_diagnostics"]["band_evaluation_count"] += 1
            if _unavailable(diagnostic["reasons"]):
                return _stop(result, "BAND_ANALYSIS_NUMERICAL_RANGE_UNSUPPORTED")
            physical_radius = float(radius*scale)
            lower, upper = diagnostic["radius_interval"]
            if diagnostic["status"] != "SAMPLE_CIRCLE_WITNESS":
                admissible = False
            if upper is None or not lower <= physical_radius <= upper:
                admissible = False
            bands.append(_band_diagnostic(diagnostic, indices, physical_radius))
        if admissible:
            result["models"].append({
                "model_id": len(result["models"]),
                "scope": "TWO_CONCENTRIC_CIRCLE_WITNESS_IN_ROTATED_ERROR_RELAXATION_ONLY",
                "normalized_projection_center": center.tolist(),
                "projection_pivot": pivot.tolist(), "projection_basis": basis.tolist(),
                "projection_scale": scale, "witness_radii": [float(radius*scale) for radius in radii],
                "max_radial_residual": max(residuals)*scale,
                "all_points_accounted_for": True, "point_count": len(p),
                "bands": bands, "partition_is_verified": False,
                "physical_face_identity_is_verified": False,
                "axis_location_is_verified": False,
            })
    result["search_complete_within_declared_policy"] = True
    if result["models"]:
        result["status"] = "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    else:
        result["reasons"] = ["NO_TWO_RADIUS_WITNESS_WITHIN_DECLARED_SEARCH"]
    if not _finite(result):
        return _stop(result, "TWO_RADIUS_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED")
    return result
