"""Position-derived, unoriented plane-pair direction hypotheses, research only.

The caller supplies TWO neighborhoods, not labels that establish real planes or
flanges. All positions are retained, apart from exact within-neighborhood repeats.
The returned direction is not an axis location, a bend, or independent evidence
for the neighborhoods. Deterministic error cones are conditional on an unverified
simultaneous position-error bound; floating-point guards are not certificates.
"""
from __future__ import annotations

import math

import numpy as np


MAX_POINTS = 256  # Total input rows across both neighborhoods, before deduplication.
_MACHINE_EPS = float(np.finfo(float).eps)
_ANGLE_GUARD = 128 * _MACHINE_EPS


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
        raise ValueError("each neighborhood must be an explicit bounded Nx3 array or None")
    if isinstance(points, np.ndarray) and (points.ndim != 2 or points.shape[1] != 3):
        raise ValueError("each neighborhood must have shape Nx3")
    return len(points)


def _positions(points, count):
    if count == 0:
        return np.empty((0, 3), dtype=float)
    if isinstance(points, (list, tuple)):
        # Bound every row before NumPy can allocate/coerce any nested payload.
        for row in points:
            if (not isinstance(row, (list, tuple, np.ndarray))
                    or isinstance(row, np.ndarray) and row.ndim != 1
                    or len(row) != 3 or not all(_real(value) for value in row)):
                raise ValueError("each position must contain exactly three real numbers")
    try:
        raw = np.asarray(points)
        if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("each neighborhood must be a numeric Nx3 array")
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("positions must be finite real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError("positions must be finite")
    return result


def _canonical_unit(vector):
    """Pick a display sign only; comparisons must use the unoriented line."""
    vector = vector / math.hypot(*vector)
    if vector[int(np.argmax(np.abs(vector)))] < 0:
        vector = -vector
    return vector


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def _fit_plane(raw, epsilon):
    points = np.unique(raw, axis=0)
    result = {
        "input_count": len(raw), "unique_count": len(points),
        "numerical_allowance": 0.0, "effective_error_bound": epsilon,
        "singular_values": None, "normal_angle_bound_radians": None,
        "plane_witness": None, "reasons": [],
    }
    if len(points) < 3:
        result["reasons"] = ["MISSING_NEIGHBORHOOD" if len(points) == 0
                             else "INSUFFICIENT_DISTINCT_POSITIONS"]
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offsets = points - points[0]
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale == 0:
            result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
            return result
        local = offsets / scale
        centered = local - np.mean(local, axis=0)
        numerical = (64 * _MACHINE_EPS * float(np.max(np.abs(points)))
                     + 64 * _MACHINE_EPS * scale)
        effective = epsilon + numerical
        e = effective / scale
    if not math.isfinite(e) or not math.isfinite(effective):
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    result["numerical_allowance"] = numerical
    result["effective_error_bound"] = effective
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        result["reasons"] = ["PLANE_FIT_UNAVAILABLE"]
        return result
    if not np.isfinite(singular).all() or not np.isfinite(vt).all():
        result["reasons"] = ["PLANE_FIT_UNAVAILABLE"]
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = singular * scale
    if not np.isfinite(lengths).all():
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    result["singular_values"] = lengths.tolist()
    if singular[1] <= 64 * _MACHINE_EPS * singular[0]:
        result["reasons"] = ["RANK_DEFICIENT_NEIGHBORHOOD"]
        return result

    normal = _canonical_unit(vt[-1])
    projections = local @ normal
    midpoint = float((np.max(projections) + np.min(projections)) / 2)
    maximum = float(np.max(np.abs(projections - midpoint))) * scale
    result["plane_witness"] = {
        "normal": normal.tolist(), "reference_point": points[0].tolist(),
        "offset_from_reference": midpoint * scale, "max_residual": maximum,
        "compatible_with_error_bound": bool(maximum <= effective),
    }
    # If any true counterpart set is coplanar within epsilon, ||X n_true||
    # <= sqrt(N)*epsilon. Since ||X n_true|| >= s2*sin(angle(n_true,n_fit)),
    # this cone contains EVERY such plane normal, not just the fitted witness.
    denominator = float(singular[1]) / math.sqrt(len(points))
    ratio = 1.0 if e >= denominator else e / denominator
    alpha = min(math.pi / 2, math.asin(ratio) + _ANGLE_GUARD)
    result["normal_angle_bound_radians"] = alpha
    if not result["plane_witness"]["compatible_with_error_bound"]:
        # TLS orientation plus a mid-strip offset is not a minimax-plane solver.
        # Failure of THIS witness does not prove that all plane fits fail.
        result["reasons"].append("NO_ALL_POINT_PLANE_WITNESS")
    if alpha >= math.pi / 2:
        result["reasons"].append("PLANE_NORMAL_UNRESOLVED")
    if not _finite(result):
        result.update(singular_values=None, normal_angle_bound_radians=None,
                      plane_witness=None, reasons=["NUMERICAL_RANGE_UNSUPPORTED"])
    return result


def analyze_position_axis_hypothesis(
    first_neighborhood, second_neighborhood, *, unit, position_error_bound,
    max_axis_angle_bound,
):
    """Bound an unoriented direction conditional on two supplied plane regions.

    ``max_axis_angle_bound`` is an explicit angular resolution policy in RADIANS,
    not a confidence level. ``position_error_bound`` is a simultaneous Euclidean
    bound in ``unit`` for each position in both regions. ``None`` or an empty
    neighborhood is missing evidence and yields abstention, not a missing bend.
    A positive status still does not verify either neighborhood or the axis.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    angle_limit = _number(max_axis_angle_bound, "max_axis_angle_bound")
    if angle_limit >= math.pi / 2:
        raise ValueError("max_axis_angle_bound must be less than pi/2 radians")
    counts = [_count(first_neighborhood), _count(second_neighborhood)]
    if sum(counts) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} total input positions are permitted")
    # Enforce the shared cap before converting either neighborhood to an array.
    raw = [_positions(first_neighborhood, counts[0]),
           _positions(second_neighborhood, counts[1])]
    planes = [_fit_plane(points, epsilon) for points in raw]
    result = {
        "schema_version": 1, "kind": "POSITION_AXIS_HYPOTHESIS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "input_count": sum(counts),
        "unique_count": sum(plane["unique_count"] for plane in planes),
        "position_error_bound": epsilon,
        "policy": {"max_axis_angle_bound_radians": angle_limit},
        "plane_neighborhoods": planes,
        "hypothesis": {
            "axis": None, "axis_angle_bound_radians": None,
            "axis_is_verified": False, "regions_are_verified": False,
            "axis_location": None, "model": "PAIR_OF_IDEAL_PLANES",
        },
        "axis_diagnostics": {
            "normal_separation_radians": None, "cross_product_norm": None,
            "cross_product_error_bound": None, "numerical_angle_allowance": _ANGLE_GUARD,
            "parallel_alternative_not_excluded": True,
        },
        "limitations": [
            "SUPPLIED_NEIGHBORHOODS_NOT_REGION_DISCOVERY", "AXIS_AND_REGIONS_UNVERIFIED",
            "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
            "NORMALS_ESTIMATED_FROM_SAME_POSITIONS_NOT_INDEPENDENT_EVIDENCE",
            "DIRECTION_ONLY_NO_CYLINDER_AXIS_LOCATION", "UNORIENTED_AXIS_NO_SIGNED_BEND_ANGLE",
            "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY", "NO_CONTINUITY_OR_FINITE_SUPPORT_CLAIM",
            "NO_RADIUS_FORMATION_THICKNESS_OR_CONFORMANCE_CLAIM",
        ],
        "reasons": [f"{name}_{reason}" for name, plane in zip(("FIRST", "SECOND"), planes)
                    for reason in plane["reasons"]],
    }
    if any(plane["plane_witness"] is None for plane in planes):
        return result
    normals = [np.asarray(plane["plane_witness"]["normal"]) for plane in planes]
    alphas = [plane["normal_angle_bound_radians"] for plane in planes]
    cross = np.cross(*normals)
    cross_length = math.hypot(*cross)
    beta = math.atan2(cross_length, abs(float(normals[0] @ normals[1])))
    # Sign-align each true normal to its fitted normal. Telescoping the cross
    # products gives ||c_true-c_fit|| <= delta1+delta2 for UNIT normals.
    error = sum(2 * math.sin(alpha / 2) for alpha in alphas) + _ANGLE_GUARD
    diagnostics = result["axis_diagnostics"]
    diagnostics.update(normal_separation_radians=beta, cross_product_norm=cross_length,
                       cross_product_error_bound=error,
                       parallel_alternative_not_excluded=bool(beta <= sum(alphas) + _ANGLE_GUARD))
    if cross_length > _ANGLE_GUARD:
        result["hypothesis"]["axis"] = _canonical_unit(cross).tolist()
    if diagnostics["parallel_alternative_not_excluded"]:
        result["reasons"].append("PARALLEL_ALTERNATIVE_NOT_EXCLUDED")
    if cross_length <= error:
        result["reasons"].append("AXIS_DIRECTION_BOUND_UNRESOLVED")
    else:
        # A radius-error ball about c_fit excludes zero. Its tangent cone has
        # half-angle asin(error/||c_fit||), enclosing every admissible c_true.
        angle = min(math.pi / 2, math.asin(error / cross_length) + _ANGLE_GUARD)
        result["hypothesis"]["axis_angle_bound_radians"] = angle
        if angle > angle_limit:
            result["reasons"].append("AXIS_ANGULAR_PRECISION_NOT_ESTABLISHED")
    if not result["reasons"]:
        result["status"] = "BOUNDED_DIRECTION_HYPOTHESIS"
    return result
