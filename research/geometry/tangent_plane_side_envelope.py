"""Conditional signed-side bounds, not verification of contact or tangency.

A SINGLE same-face circular cylinder tangent to a plane lies on one side of it.
Robust straddling excludes only that model. A valid inner/outer radius mixture
can straddle an inner flange plane and must NOT be rejected by this criterion.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.position_axis_hypothesis import _count, _fit_plane, _number, _positions
from research.geometry.position_bend_candidates import _finite_payload


MAX_POINTS = 256
_MACHINE_EPS = float(np.finfo(float).eps)


def _signed_cone_intervals(plane_points, candidate_points, plane_analysis, epsilon):
    """Reuse an INTERNAL all-point plane analysis; return a finite dict or None.

    Candidate order is preserved, with no deduplication, subset selection or fit.
    Plane normals are sign-aligned mathematically to the fitted display normal;
    that display sign is not a physical inward/outward orientation. For v=p-m,
    the exact unit-normal cone extrema enclose n_true.v. The true candidate and
    true plane centroid contribute another epsilon EACH; no sqrt(N) reduction
    or independence premise is used. Numerical allowances are engineering guards,
    not directed-rounded interval arithmetic.
    """
    try:
        counts = [_count(plane_points), _count(candidate_points)]
        if not counts[0] or sum(counts) > MAX_POINTS:
            return None
        plane = _positions(plane_points, counts[0])
        candidates = _positions(candidate_points, counts[1])
        epsilon = _number(epsilon, "position_error_bound", allow_zero=True)
        if plane_analysis["reasons"]:
            return None
        normal = np.asarray(plane_analysis["plane_witness"]["normal"], dtype=float)
        alpha = float(plane_analysis["normal_angle_bound_radians"])
        plane_guard = float(plane_analysis["numerical_allowance"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not math.isfinite(alpha) or not 0 <= alpha < math.pi / 2
            or not math.isfinite(plane_guard) or plane_guard < 0):
        return None
    length = math.hypot(*normal)
    if not math.isfinite(length) or length <= 0:
        return None
    normal = normal / length
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        all_points = np.concatenate((plane, candidates))
        reference = plane[0]
        offsets = all_points - reference
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale <= 0:
            return None
        centroid = reference + np.mean(offsets[:len(plane)] / scale, axis=0) * scale
        # Bounds are anchored to the ACTUALLY RETURNED observed centroid, not an
        # ideal local mean whose conversion back to world coordinates may round.
        vectors = (candidates - centroid) / scale
        a = vectors @ normal
        tangent = vectors - a[:, None] * normal
        b = np.hypot(np.hypot(tangent[:, 0], tangent[:, 1]), tangent[:, 2])
        r = np.hypot(np.hypot(vectors[:, 0], vectors[:, 1]), vectors[:, 2])
        cosine, sine = math.cos(alpha), math.sin(alpha)
        # If +v or -v lies inside the cone, the corresponding extremum is +/-r;
        # otherwise it lies on the cone boundary. No inverse cosine is needed.
        cone_upper = np.where(a >= r * cosine, r, a * cosine + b * sine) * scale
        cone_lower = np.where(-a >= r * cosine, -r, a * cosine - b * sine) * scale
        guard = (4 * plane_guard
                 + 128 * _MACHINE_EPS * float(np.max(np.abs(all_points)))
                 + 128 * _MACHINE_EPS * scale)
        budget = 2 * epsilon + guard
        lower, upper = cone_lower - budget, cone_upper + budget
    if (not np.isfinite(centroid).all() or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or not np.isfinite(cone_lower).all()
            or not np.isfinite(cone_upper).all() or not math.isfinite(guard)
            or not math.isfinite(budget) or np.any(lower > upper)):
        return None
    positive = np.flatnonzero(lower > 0).tolist()
    negative = np.flatnonzero(upper < 0).tolist()
    unresolved = np.flatnonzero((lower <= 0) & (upper >= 0)).tolist()
    result = {
        "candidate_count": len(candidates),
        "observed_plane_centroid": centroid.tolist(), "plane_normal": normal.tolist(),
        "normal_angle_bound_radians": alpha,
        "plane_centroid_position_error_bound": epsilon,
        "candidate_position_error_bound": epsilon,
        "combined_position_error_bound": 2 * epsilon,
        "numerical_allowance": guard, "combined_position_and_numerical_allowance": budget,
        "normal_cone_signed_intervals": np.column_stack((cone_lower, cone_upper)).tolist(),
        "signed_intervals": np.column_stack((lower, upper)).tolist(),
        "robust_positive_indices": positive, "robust_negative_indices": negative,
        "unresolved_indices": unresolved,
        "positive_side_not_excluded_indices": np.flatnonzero(upper >= 0).tolist(),
        "negative_side_not_excluded_indices": np.flatnonzero(lower <= 0).tolist(),
        "single_same_face_tangent_cylinder_excluded": bool(positive and negative),
        "two_radius_or_multiple_surface_model_is_excluded": False,
        "interval_scope": "NECESSARY_SIGNED_SIDE_BOUNDS_CONDITIONAL_ON_IDEAL_PLANE_AND_SIMULTANEOUS_ERROR",
        "normal_sign_scope": "MATHEMATICAL_SIGN_ALIGNMENT_NOT_PHYSICAL_ORIENTATION",
        "candidate_order_is_preserved": True,
        "joint_halfspace_membership_is_verified": False,
        "tangency_is_verified": False, "contact_is_verified": False,
        "physical_membership_is_verified": False, "enclosure_is_numerically_certified": False,
    }
    return result if _finite_payload(result) else None


def analyze_tangent_plane_sides(
    plane_region, candidate_points, *, unit, position_error_bound, max_normal_angle_bound,
):
    """Fit ALL plane samples at original epsilon, then disclose candidate sides.

    The shared 256 RAW-row cap precedes conversion and exact within-input dedup.
    Caller supplies positions and explicit error/resolution policies only, never
    a verified-plane flag, normals, point labels or an asserted material face.
    A positive/negative side is conditional on the supplied neighborhood really
    corresponding to one ideal plane under that error model.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    angle = _number(max_normal_angle_bound, "max_normal_angle_bound")
    if angle >= math.pi / 2:
        raise ValueError("max_normal_angle_bound must be less than pi/2 radians")
    counts = [_count(plane_region), _count(candidate_points)]
    if sum(counts) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} total raw positions are permitted")
    raw = [_positions(points, count) for points, count in zip((plane_region, candidate_points), counts)]
    planes, candidates = [np.unique(points, axis=0) for points in raw]
    plane_analysis = _fit_plane(planes, epsilon)
    result = {
        "schema_version": 1, "kind": "TANGENT_PLANE_SIDE_ENVELOPE_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": sum(counts), "input_counts": {
            "plane_region": counts[0], "candidate_points": counts[1],
        },
        "plane_positions": planes.tolist(), "candidate_positions": candidates.tolist(),
        "position_error_bound": epsilon, "plane_analysis": plane_analysis,
        "side_envelope": None, "analysis_available": False,
        "single_same_face_tangent_cylinder_excluded": False,
        "two_radius_or_multiple_surface_model_is_excluded": False,
        "physical_bend_count": None, "physical_geometry_is_verified": False,
        "tangency_is_verified": False, "contact_is_verified": False,
        "policy": {
            "max_normal_angle_bound_radians": angle,
            "max_total_raw_positions": MAX_POINTS,
            "plane_fit_uses_original_error_bound": True,
            "candidate_residual_selection_or_trimming": False,
            "unresolved_side_membership": "BOTH_CLOSED_HALFSPACES_NOT_EXCLUDED",
            "exclusion_scope": "SINGLE_SAME_FACE_CIRCULAR_CYLINDER_TANGENT_TO_THIS_PLANE_ONLY",
        },
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "SUPPLIED_PLANE_REGION_MEMBERSHIP_UNVERIFIED",
            "SIMULTANEOUS_POSITION_ERROR_UNCALIBRATED",
            "ALL_POINT_TLS_PROPOSAL_NOT_COMPLETE_PLANE_FEASIBILITY_SEARCH",
            "CONE_AND_CENTROID_BOUNDS_NOT_INDEPENDENT_STATISTICAL_EVIDENCE",
            "NORMAL_DISPLAY_SIGN_NOT_PHYSICAL_INWARD_OR_OUTWARD",
            "ONE_SIDED_POINTS_DO_NOT_ESTABLISH_TANGENCY_OR_CONTACT",
            "STRADDLING_EXCLUDES_ONLY_SINGLE_SAME_FACE_TANGENT_CYLINDER",
            "VALID_INNER_OUTER_MIXTURE_CAN_STRADDLE_AN_INNER_FLANGE_PLANE",
            "NO_TWO_RADIUS_MULTIFACE_THICKNESS_OR_ADJACENT_BEND_EXCLUSION",
            "POINTWISE_SIDE_POSSIBILITY_NOT_JOINT_SEGMENTATION",
            "NO_CONTINUITY_VISIBILITY_FORMING_STATE_OR_CONFORMANCE_CLAIM",
            "ENGINEERING_NUMERICAL_GUARDS_NOT_DIRECTED_ROUNDING_CERTIFICATES",
        ],
        "reasons": [],
    }
    if plane_analysis["reasons"]:
        result["reasons"] = [f"PLANE_{reason}" for reason in plane_analysis["reasons"]]
        return result
    alpha = plane_analysis["normal_angle_bound_radians"]
    if alpha is None or alpha > angle:
        result["reasons"] = ["PLANE_NORMAL_PRECISION_NOT_ESTABLISHED"]
        return result
    envelope = _signed_cone_intervals(planes, candidates, plane_analysis, epsilon)
    if envelope is None:
        result["reasons"] = ["SIDE_ENVELOPE_NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    result["side_envelope"] = envelope
    result["analysis_available"] = True
    excluded = envelope["single_same_face_tangent_cylinder_excluded"]
    result["single_same_face_tangent_cylinder_excluded"] = excluded
    if not len(candidates):
        result["reasons"] = ["NO_CANDIDATE_POSITIONS"]
    elif excluded:
        result["status"] = "SINGLE_SAME_FACE_TANGENT_CYLINDER_EXCLUDED"
        result["reasons"] = ["ROBUST_OPPOSITE_SIDES_OF_EVERY_ADMISSIBLE_PLANE"]
    else:
        result["status"] = "PLANE_SIDE_ENVELOPE"
    return result
