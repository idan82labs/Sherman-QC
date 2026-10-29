"""Bounded, position-only cylinder-sample diagnostic, not a bend detector.

The supplied axis, single-face region and position-error bound are hypotheses.
No CAD radius, construction normal, patch label, hidden sample or count is used.
Bounds are analytical outer constraints with floating-point padding, NOT a
directed-rounded numerical certificate or calibrated physical uncertainty.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np


MAX_POINTS = 256
_MACHINE_EPS = float(np.finfo(float).eps)


def _real_scalar(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def _number(value, name, *, allow_zero=False):
    if not _real_scalar(value):
        raise ValueError(f"{name} must be an explicit finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a finite number") from exc
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return value


def _positions(points):
    if not isinstance(points, (list, tuple, np.ndarray)):
        raise ValueError("points must be an explicit bounded Nx3 array")
    if isinstance(points, np.ndarray) and points.ndim == 0:
        raise ValueError("points must be an explicit bounded Nx3 array")
    if len(points) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points are permitted")
    if isinstance(points, (list, tuple)):
        # Validate bounded row widths before NumPy can allocate a ragged or huge
        # nested payload. Do not let mixed numeric rows coerce booleans to 0/1.
        for row in points:
            if (not isinstance(row, (list, tuple, np.ndarray))
                    or isinstance(row, np.ndarray) and row.ndim != 1
                    or len(row) != 3 or not all(_real_scalar(value) for value in row)):
                raise ValueError("points must contain exactly three real numbers per row")
    try:
        raw = np.asarray(points)
        if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("points must be a numeric Nx3 array")
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must be a finite numeric Nx3 array") from exc
    if not np.isfinite(result).all():
        raise ValueError("points must be finite")
    return result


def _axis_basis(axis):
    try:
        if isinstance(axis, (list, tuple)) and (len(axis) != 3 or not all(_real_scalar(value) for value in axis)):
            raise ValueError("axis must contain exactly three real numbers")
        raw = np.asarray(axis)
        if raw.shape != (3,) or raw.dtype.kind not in "iuf":
            raise ValueError("axis must be a numeric unit vector")
        a = np.asarray(raw, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("axis must be a finite unit vector") from exc
    if not np.isfinite(a).all():
        raise ValueError("axis must be finite")
    norm = math.hypot(*a)
    if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("axis must already be a unit vector")
    a = a / norm  # Numerical cleanup, never estimation of an unknown axis.
    seed = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, seed)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, np.column_stack((u, v))


def _plane_strip(q):
    """Minimum enclosing strip, by testing every pair-edge orientation.

    An optimum strip for a finite planar set has a convex-hull edge orientation.
    All pairs include those edges; testing extra orientations cannot create a
    false witness because each width is evaluated against ALL input positions.
    Chunks bound workspace; no hull tolerance can erase a shallow-arc point.
    """
    best_width, best_normal, best_offset = math.inf, np.array([1.0, 0.0]), 0.0
    pairs = list(combinations(range(len(q)), 2))
    for start in range(0, len(pairs), 256):
        chunk = pairs[start:start + 256]
        d = np.array([q[j] - q[i] for i, j in chunk])
        lengths = np.hypot(d[:, 0], d[:, 1])
        d, lengths = d[lengths > 0], lengths[lengths > 0]
        if not len(d):
            continue
        normals = np.column_stack((-d[:, 1], d[:, 0])) / lengths[:, None]
        values = q @ normals.T
        low, high = np.min(values, axis=0), np.max(values, axis=0)
        widths = high - low
        index = int(np.argmin(widths))
        if widths[index] < best_width:
            best_width = float(widths[index])
            best_normal = normals[index]
            best_offset = float((low[index] + high[index]) / 2)
    if not math.isfinite(best_width):
        best_width = 0.0
        best_offset = float(q[0, 0])
    # Re-evaluate actual residuals at the chosen central plane.
    residual = float(np.max(np.abs(q @ best_normal - best_offset)))
    return best_width, best_normal, best_offset, residual


def _radius_outer_interval(q, epsilon):
    """Necessary radius bounds from a diameter fan, not a feasibility solver."""
    delta = q[:, None, :] - q[None, :, :]
    distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
    i, j = np.unravel_index(int(np.argmax(distance)), distance.shape)
    lower, upper = max(0.0, float(distance[i, j]) - 2 * epsilon) / 2, math.inf
    triangle_count = 0
    for k in range(len(q)):
        if k in (i, j):
            continue
        sides = np.array([distance[i, j], distance[i, k], distance[j, k]])
        u, v = q[j] - q[i], q[k] - q[i]
        determinant = abs(float(u[0] * v[1] - u[1] * v[0]))
        allowance = epsilon * float(np.sum(sides)) + 3 * epsilon**2
        edge_low, edge_high = np.maximum(0.0, sides - 2 * epsilon), sides + 2 * epsilon
        if determinant + allowance > 0:
            lower = max(lower, float(np.prod(edge_low)) / (2 * (determinant + allowance)))
        if determinant > allowance:
            upper = min(upper, float(np.prod(edge_high)) / (2 * (determinant - allowance)))
        triangle_count += 1
    return lower, upper, triangle_count


def _circle_witness(q):
    """An algebraic seed and finite geometric least-squares search.

    Failure to find a max-error witness never proves no compatible circle exists.
    No residual trimming: every position must satisfy the declared error bound.
    """
    matrix = np.column_stack((2 * q, np.ones(len(q))))
    rhs = np.sum(q * q, axis=1)
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank != 3 or not np.isfinite(solution).all():
        return None
    center = solution[:2]
    radius = float(np.mean(np.linalg.norm(q - center, axis=1)))
    if not math.isfinite(radius) or radius <= 0:
        return None
    # Bounded damped Gauss-Newton, in normalized coordinates.
    for _ in range(32):
        vectors = q - center
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= _MACHINE_EPS) or not np.isfinite(lengths).all():
            break
        residuals = lengths - radius
        jacobian = np.column_stack((-vectors / lengths[:, None], -np.ones(len(q))))
        try:
            step, _, rank, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        except np.linalg.LinAlgError:
            break
        if rank != 3 or not np.isfinite(step).all():
            break
        objective = float(residuals @ residuals)
        accepted = False
        for power in range(12):
            fraction = 0.5**power
            candidate_center = center + fraction * step[:2]
            candidate_radius = radius + fraction * float(step[2])
            if candidate_radius <= 0 or not math.isfinite(candidate_radius):
                continue
            candidate_residual = np.linalg.norm(q - candidate_center, axis=1) - candidate_radius
            if not np.isfinite(candidate_residual).all():
                continue
            if float(candidate_residual @ candidate_residual) <= objective:
                center, radius = candidate_center, candidate_radius
                accepted = True
                break
        if not accepted or float(np.linalg.norm(fraction * step)) <= 64 * _MACHINE_EPS * max(1.0, radius):
            break
    residuals = np.linalg.norm(q - center, axis=1) - radius
    if not np.isfinite(residuals).all():
        return None
    return center, radius, float(np.max(np.abs(residuals))), float(np.sqrt(np.mean(residuals**2)))


def _finite_payload(value):
    if isinstance(value, dict):
        return all(_finite_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_payload(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def analyze_axis_conditioned_radius(
    points, *, axis, unit, position_error_bound, max_axial_gap, max_radius_width,
):
    """Return conservative, conditional evidence for a finite position sample.

    ``max_axial_gap`` is a declared sampling policy, not a topology detector.
    ``max_radius_width`` is a requested precision, not a tuned confidence score.
    Axial group extents describe observed sample coordinates, NEVER a certified
    continuous component or admissible surface marker. The fitted radius belongs
    to an unidentified single face, not necessarily the inner or neutral radius.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    epsilon = _number(position_error_bound, "position_error_bound", allow_zero=True)
    gap = _number(max_axial_gap, "max_axial_gap")
    width_limit = _number(max_radius_width, "max_radius_width")
    raw = _positions(points)
    a, basis = _axis_basis(axis)
    p = np.unique(raw, axis=0)
    result = {
        "schema_version": 1, "kind": "AXIS_CONDITIONED_RADIUS_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED",
        "unit": unit, "input_count": len(raw), "unique_count": len(p),
        "position_error_bound": epsilon, "numerical_allowance": 0.0,
        "effective_error_bound": epsilon, "radius_interval": [0.0, None],
        "policy": {"max_axial_gap": gap, "max_radius_width": width_limit},
        "circle_witness": None, "parallel_plane_witness": None,
        "axial_sample_groups": [], "triangle_constraint_count": 0,
        "hypothesis": {"axis": a.tolist(), "axis_is_verified": False,
                       "region_is_verified": False, "model": "SINGLE_CIRCULAR_CYLINDER_FACE"},
        "limitations": ["AXIS_AND_REGION_CONDITIONAL", "POSITION_ERROR_BOUND_UNCALIBRATED",
                        "OUTER_BOUND_NOT_NUMERICALLY_CERTIFIED", "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY",
                        "NO_FORMATION_ANGLE_THICKNESS_OR_CONFORMANCE_CLAIM", "NO_CONTINUITY_FROM_SAMPLES"],
        "reasons": [],
    }
    if len(p) < 3:
        result["reasons"] = ["INSUFFICIENT_DISTINCT_POSITIONS"]
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offsets = p - p[0]
        scale = float(np.max(np.abs(offsets)))
        if not math.isfinite(scale) or scale == 0:
            result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
            return result
        local = offsets / scale
        local_center = np.mean(local, axis=0)
        origin = p[0] + scale * local_center
        centered = local - local_center
        q, axial = centered @ basis, centered @ a
        numerical = 64 * _MACHINE_EPS * float(np.max(np.abs(p))) + 64 * _MACHINE_EPS * scale
        effective = epsilon + numerical
        e = effective / scale
    if not np.isfinite(origin).all() or not math.isfinite(e) or e > 1e50:
        result["reasons"] = ["NUMERICAL_RANGE_UNSUPPORTED"]
        return result
    result["numerical_allowance"] = numerical
    result["effective_error_bound"] = effective
    reasons = []

    plane_width, normal2, plane_offset, plane_residual = _plane_strip(q)
    normal3 = basis @ normal2
    plane_compatible = plane_residual * scale <= effective
    result["parallel_plane_witness"] = {
        "width": plane_width * scale, "max_residual": plane_residual * scale,
        "normal": normal3.tolist(), "point": (origin + scale * plane_offset * normal3).tolist(),
        "compatible_with_error_bound": plane_compatible,
    }
    if plane_compatible:
        reasons.append("PARALLEL_PLANE_COMPATIBLE")

    lower, upper, triangles = _radius_outer_interval(q, e)
    result["triangle_constraint_count"] = triangles
    physical_lower, physical_upper = lower * scale, upper * scale
    result["radius_interval"] = [physical_lower, physical_upper if math.isfinite(physical_upper) else None]
    if lower > upper:
        reasons.append("EMPTY_ANALYTIC_RADIUS_ENCLOSURE")
    elif not math.isfinite(physical_upper):
        reasons.append("RADIUS_UPPER_BOUND_UNRESOLVED")
    elif physical_upper - physical_lower > width_limit:
        reasons.append("RADIUS_PRECISION_NOT_ESTABLISHED")

    fit = _circle_witness(q)
    axis_point = None
    if fit is None:
        reasons.append("NO_VERIFIED_CIRCLE_WITNESS")
    else:
        center2, radius, maximum, rms = fit
        axis_point = origin + scale * (basis @ center2)
        if not np.isfinite(axis_point).all() or not math.isfinite(radius * scale):
            reasons.append("NUMERICAL_RANGE_UNSUPPORTED")
            axis_point = None
        else:
            compatible = maximum * scale <= effective
            result["circle_witness"] = {
                "radius": radius * scale, "axis_point": axis_point.tolist(),
                "max_radial_residual": maximum * scale, "rms_radial_residual": rms * scale,
                "compatible_with_error_bound": compatible,
            }
            if not compatible:
                reasons.append("NO_VERIFIED_CIRCLE_WITNESS")
            elif not lower <= radius <= upper:
                # A feasible circle must satisfy every necessary radius bound.
                # Numeric disagreement is an abstention, not evidence against
                # the physical part. No tolerance repairs the inconsistent fit.
                reasons.append("NUMERICAL_ENCLOSURE_INCONSISTENT")

    # Split measured axial coordinates only; a run does not establish continuity.
    order = np.argsort(axial, kind="stable")
    values = axial[order] * scale
    # Compare adjacent original-position differences, avoiding subtraction of
    # rounded centred coordinates. No tolerance is allowed to fill a sample gap.
    measured_gaps = np.diff(p[order], axis=0) @ a
    cuts = np.flatnonzero(measured_gaps > gap) + 1
    for group in np.split(values, cuts):
        lo, hi = float(group[0]), float(group[-1])
        result["axial_sample_groups"].append({
            "sample_min": lo, "sample_max": hi, "point_count": len(group),
            "axis_geometry_source": "CIRCLE_FIT_HYPOTHESIS" if axis_point is not None else "UNAVAILABLE",
            "axis_endpoints": [
                (axis_point + lo * a).tolist(), (axis_point + hi * a).tolist(),
            ] if axis_point is not None else None,
        })
    if len(result["axial_sample_groups"]) > 1:
        reasons.append("AXIAL_SAMPLE_GAP")
    if float(values[-1] - values[0]) <= 2 * effective:
        reasons.append("AXIAL_EXTENT_UNRESOLVED")
    if not reasons:
        result["status"] = "SAMPLE_CIRCLE_WITNESS"
    result["reasons"] = sorted(set(reasons))
    if not _finite_payload(result):
        result.update(status="ABSTAINED", radius_interval=[0.0, None],
                      circle_witness=None, parallel_plane_witness=None,
                      axial_sample_groups=[], reasons=["NUMERICAL_RANGE_UNSUPPORTED"])
    return result
