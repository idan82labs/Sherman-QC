"""Bounded, CAD-only development of one connected planar/cylindrical skin.

This is a visual geometric skin development, NOT a neutral-axis blank.  No
K-factor, bend allowance, manufacturing order, scan, or verified unit claim is
made. Unsupported topology returns an empty, explicit UNAVAILABLE envelope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


MAX_CAD_BYTES = 5_000_000
MAX_FACES = 1024
MAX_TRIANGLES = 200_000
MAX_CANDIDATES = 32
MAX_EDGE_USES = 16384
MAX_EDGE_IDENTITY_COMPARISONS = 1_000_000
MAX_OVERLAP_COMPARISONS = 1_000_000
MAX_CURVE_DEGREE = 8
MAX_CURVE_POLES = 256
MAX_TOTAL_CURVE_POLES = 32768
MAX_SURFACE_POLES = 1024
MAX_REFINED_CURVE_POLES = 2 * MAX_CURVE_POLES
MAX_KNOT_INSERTIONS = 2 * MAX_CURVE_POLES
MAX_CLOSED_CIRCLE_RAILS = 32
MECHANISM_VERSION = 3
CURVE_COEFFICIENT_TOL = 1e-10
MODEL = "SELECTED_CAD_SKIN_DEVELOPMENT"
# Numerical geometry tolerance and published candidate coordinate rounding,
# not parameters fitted to any physical part or scan.
GEOM_TOL = 1e-5
CANDIDATE_ROUNDING_TOL = 0.002
MAX_SWEEP_DEG = 175.0


class UnfoldUnavailable(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise UnfoldUnavailable(reason)


def _unit(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == (3,) and np.isfinite(result).all(), "NONFINITE_GEOMETRY")
    length = float(np.linalg.norm(result))
    _require(length > GEOM_TOL, "DEGENERATE_GEOMETRY")
    return result / length


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    _require(math.isfinite(angle), "NONFINITE_GEOMETRY")
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1. - math.cos(angle)) * (skew @ skew)


@dataclass
class NativeCurve:
    """Exact bounded native curve coefficients, not sampled curve inference."""
    kind: str
    poles: np.ndarray
    degree: int = 1
    weights: np.ndarray | None = None
    knots: np.ndarray | None = None
    multiplicities: np.ndarray | None = None
    center: np.ndarray | None = None
    axis: np.ndarray | None = None
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None
    radius: float | None = None
    u0: float | None = None
    u1: float | None = None

    def clamped_spline(self) -> bool:
        return (self.kind == "spline" and self.knots is not None and self.multiplicities is not None
                and self.weights is not None and len(self.knots) >= 2
                and np.isfinite(self.knots).all() and self.knots[-1] > self.knots[0]
                and self.multiplicities[0] == self.degree + 1
                and self.multiplicities[-1] == self.degree + 1
                and int(self.multiplicities.sum()) == len(self.poles) + self.degree + 1)

    def straight_endpoints(self) -> np.ndarray | None:
        if self.kind == "line":
            return self.poles.copy()
        if self.kind != "spline" or not self.clamped_spline():
            return None
        endpoints = self.poles[[0, -1]]
        vector = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(vector))
        if length <= GEOM_TOL:
            return None
        projections = (self.poles - endpoints[0]) @ (vector / length)
        residual = self.poles - endpoints[0] - projections[:, None] * (vector / length)
        if (np.linalg.norm(residual, axis=1).max() <= GEOM_TOL
                and np.diff(projections).min(initial=0.) >= -GEOM_TOL
                and projections.min() >= -GEOM_TOL and projections.max() <= length + GEOM_TOL):
            return endpoints.copy()
        return None


def _translated_curves(left: NativeCurve, right: NativeCurve, offset: np.ndarray) -> bool:
    """Prove same curve locus after translation, including reversed direction.

    Spline comparison refines each exact native knot span into Bezier pieces
    and uses a global positive-weight rational coefficient deviation certificate.
    No sampled points or scale-insensitive coefficient tolerances are used.
    """
    left_line, right_line = left.straight_endpoints(), right.straight_endpoints()
    if left_line is not None and right_line is not None:
        return min(np.linalg.norm(left_line + offset - right_line, axis=1).max(),
                   np.linalg.norm(left_line + offset - right_line[::-1], axis=1).max()) <= GEOM_TOL
    if left.kind != right.kind:
        return False
    if left.kind == "circle":
        center_error = float(np.linalg.norm(left.center + offset - right.center))
        if (left.radius * abs(left.u1 - left.u0 - 2 * math.pi) <= GEOM_TOL
                and right.radius * abs(right.u1 - right.u0 - 2 * math.pi) <= GEOM_TOL):
            bound = (center_error + abs(left.radius - right.radius)
                     + max(left.radius, right.radius) * min(
                         np.linalg.norm(_unit(left.axis) - _unit(right.axis)),
                         np.linalg.norm(_unit(left.axis) + _unit(right.axis)))
                     + left.radius * abs(left.u1 - left.u0 - 2 * math.pi)
                     + right.radius * abs(right.u1 - right.u0 - 2 * math.pi))
            return bound <= GEOM_TOL
        left_start = math.cos(left.u0) * left.x_axis + math.sin(left.u0) * left.y_axis
        left_tangent = -math.sin(left.u0) * left.x_axis + math.cos(left.u0) * left.y_axis
        sweep_error = right.radius * abs((left.u1 - left.u0) - (right.u1 - right.u0))
        for reverse in (False, True):
            u = right.u1 if reverse else right.u0
            start = math.cos(u) * right.x_axis + math.sin(u) * right.y_axis
            tangent = (-math.sin(u) * right.x_axis + math.cos(u) * right.y_axis) * (-1. if reverse else 1.)
            bound = (center_error + np.linalg.norm(left.radius * left_start - right.radius * start)
                     + np.linalg.norm(left.radius * left_tangent - right.radius * tangent) + sweep_error)
            if bound <= GEOM_TOL:
                return True
        return False
    if left.kind == "spline":
        if not left.clamped_spline() or not right.clamped_spline() or left.degree != right.degree:
            return False
        for reverse in (False, True):
            if _spline_translation_bound(left, right, offset, reverse=reverse) <= GEOM_TOL:
                return True
    return False


def _curve_in_perpendicular_plane(curve: NativeCurve, direction: np.ndarray) -> bool:
    direction = _unit(direction)
    if curve.kind == "circle":
        return curve.radius * math.hypot(float(np.dot(curve.x_axis, direction)),
                                        float(np.dot(curve.y_axis, direction))) <= GEOM_TOL
    return np.ptp(curve.poles @ direction) <= GEOM_TOL


def _closed_circle_set_bound(curves: list[NativeCurve], cylinder: SkinFace,
                             plane: SkinFace) -> dict[str, float] | None:
    """Prove a bounded native rail set covers one entire cylindrical carrier.

    Intervals are placed in the wall's common angular chart, not merely summed.
    Positive gaps/overlaps beyond the arithmetic roundoff allowance refuse even
    if each defect is below GEOM_TOL. The remaining physical error is charged to
    the pair's single shared geometric budget below.
    """
    if not 0 < len(curves) <= MAX_CLOSED_CIRCLE_RAILS or any(curve.kind != "circle" for curve in curves):
        return None
    radius = float(cylinder.radius)
    if (not math.isfinite(radius) or radius <= GEOM_TOL
            or any(value is None or not math.isfinite(value) for value in (cylinder.u0, cylinder.u1))):
        return None
    wall_frame = [np.asarray(value, dtype=float) for value in (cylinder.axis, cylinder.x_axis, cylinder.y_axis)]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in wall_frame):
        return None
    axis, x, y = wall_frame
    origin = np.asarray(cylinder.origin, dtype=float)
    center = origin + axis * float(np.dot(plane.origin - origin, axis))
    tau = 2 * math.pi
    epsilon = np.finfo(float).eps
    frame_floor = 64 * epsilon
    if (max(abs(np.linalg.norm(value) - 1.) for value in wall_frame) > frame_floor
            or abs(np.dot(x, y)) > frame_floor or abs(np.dot(x, axis)) > frame_floor
            or abs(np.dot(y, axis)) > frame_floor or np.linalg.norm(np.cross(x, y) - axis) > frame_floor):
        return None
    intervals, carrier_error = [], 0.
    coordinate_scale = max(1., float(np.max(np.abs(origin))), float(np.max(np.abs(center))),
                           float(np.max(np.abs(plane.origin))))
    parameter_scale = max(1., abs(float(cylinder.u0)), abs(float(cylinder.u1)), tau)
    for curve in curves:
        values = [curve.radius, curve.u0, curve.u1]
        if (any(value is None or not math.isfinite(value) for value in values)
                or curve.radius <= 0. or curve.u1 <= curve.u0):
            return None
        frame = [np.asarray(value, dtype=float) for value in (curve.axis, curve.x_axis, curve.y_axis, curve.center)]
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in frame):
            return None
        native_axis, native_x, native_y, native_center = frame
        if (max(abs(np.linalg.norm(value) - 1.) for value in (native_axis, native_x, native_y)) > frame_floor
                or abs(np.dot(native_x, native_y)) > frame_floor
                or np.linalg.norm(np.cross(native_x, native_y) - native_axis) > frame_floor):
            return None
        sweep = float(curve.u1 - curve.u0)
        parameter_scale = max(parameter_scale, abs(curve.u0), abs(curve.u1))
        coordinate_scale = max(coordinate_scale, float(np.max(np.abs(native_center))))
        radial = math.cos(curve.u0) * native_x + math.sin(curve.u0) * native_y
        tangent = -math.sin(curve.u0) * native_x + math.cos(curve.u0) * native_y
        orientation = float(np.dot(np.cross(native_x, native_y), axis))
        if abs(orientation) < .5:
            return None
        sign = 1. if orientation > 0. else -1.
        phase = math.atan2(float(np.dot(radial, y)), float(np.dot(radial, x))) % tau
        reference_radial = math.cos(phase) * x + math.sin(phase) * y
        reference_tangent = sign * (-math.sin(phase) * x + math.cos(phase) * y)
        # This coefficient bound controls every point of the whole circular arc,
        # including shifted/tilted carriers, not only endpoints or midpoint.
        error = (float(np.linalg.norm(native_center - center))
                 + float(np.linalg.norm(curve.radius * radial - radius * reference_radial))
                 + float(np.linalg.norm(curve.radius * tangent - radius * reference_tangent)))
        carrier_error = max(carrier_error, error)
        start = phase if sign > 0. else (phase - sweep) % tau
        intervals.append((start, sweep))
    operations = 128 + 32 * len(curves)
    gamma = operations * epsilon / (1. - operations * epsilon)
    numerical_error = gamma * (coordinate_scale + radius * (parameter_scale + tau + 1.))
    if not math.isfinite(numerical_error) or numerical_error >= GEOM_TOL:
        return None
    angular_roundoff = numerical_error / radius
    wall_closure_error = radius * abs(float(cylinder.u1 - cylinder.u0) - tau)
    if wall_closure_error > numerical_error:
        return None  # No partial-cylinder or partial-arc union fallback.
    pieces = []
    for start, sweep in intervals:
        if sweep <= angular_roundoff or sweep > tau + angular_roundoff:
            return None  # Degenerate slivers and multiple turns cannot add coverage.
        end = start + sweep
        if end > tau:
            pieces.extend(((start, tau), (0., end - tau)))
        else:
            pieces.append((start, end))
    cursor, gap, overlap = 0., 0., 0.
    for start, stop in sorted(pieces):
        if start > cursor:
            gap += start - cursor
        elif start < cursor:
            overlap += max(0., min(stop, cursor) - start)
        cursor = max(cursor, stop)
    if cursor < tau:
        gap += tau - cursor
    physical_gap, physical_overlap = radius * gap, radius * overlap
    if physical_gap > numerical_error or physical_overlap > numerical_error:
        return None
    return {"carrier_error": carrier_error, "gap_error": physical_gap,
            "overlap_error": physical_overlap, "wall_closure_error": wall_closure_error,
            "numerical_error": numerical_error,
            "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2])}


def _closed_circular_rail_sets_proven(cylinder: SkinFace, left_plane: SkinFace, right_plane: SkinFace,
                                     left_curves: list[NativeCurve], right_curves: list[NativeCurve],
                                     offset: np.ndarray) -> bool:
    """Narrow complete-circle boundary fallback with one shared physical budget."""
    try:
        if cylinder.kind != "cylinder" or cylinder.radius is None or cylinder.radius <= GEOM_TOL:
            return False
        left = _closed_circle_set_bound(left_curves, cylinder, left_plane)
        right = _closed_circle_set_bound(right_curves, cylinder, right_plane)
        if left is None or right is None:
            return False
        centers = [np.array([value["center_x"], value["center_y"], value["center_z"]])
                   for value in (left, right)]
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            return False
        translation_error = float(np.linalg.norm(centers[0] + offset - centers[1]))
        translation_roundoff = 16 * np.finfo(float).eps * max(
            1., float(np.max(np.abs(offset))), *(float(np.max(np.abs(center))) for center in centers))
        bound = translation_error + sum(value["carrier_error"] + value["gap_error"]
            + value["overlap_error"] + value["wall_closure_error"]
            + value["numerical_error"] for value in (left, right)) + translation_roundoff
        return math.isfinite(bound) and bound <= GEOM_TOL
    except (UnfoldUnavailable, TypeError, ValueError, AttributeError):
        return False


def _spline_translation_bound(left: NativeCurve, right: NativeCurve, offset: np.ndarray,
                              *, reverse: bool = False) -> float:
    """Own-knot Boehm→Bezier refinement; certified translated-locus bound.

    Identical degree/multiplicity topology defines a monotone, piecewise-affine
    parameter correspondence; matching global knot coordinates is unnecessary.
    Each original knot stays untouched. On each nonnegative Bernstein basis,
    pole error is convex-hull bounded. Relative
    weight error rho changes normalized rational coefficients by at most
    2*rho/(1-rho). A conservative gamma_n roundoff allowance accounts for every
    homogeneous insertion/division; ill-conditioned certificates abstain.
    """
    if (left.knots.shape != right.knots.shape
            or not np.array_equal(left.multiplicities, right.multiplicities[::-1] if reverse else right.multiplicities)
            or np.any(left.multiplicities[1:-1] > left.degree)):
        return math.inf
    refined = []
    insertions = 0
    for curve in (left, right):
        native_parameter_scale = max(abs(float(curve.knots[0])), abs(float(curve.knots[-1])),
                                     float(curve.knots[-1] - curve.knots[0]))
        _require(np.diff(curve.knots).min() >= 64 * np.finfo(float).eps * native_parameter_scale,
                 "UNCERTIFIED_KNOT_INSERTION_CONDITIONING")
        expanded = np.repeat(curve.knots, curve.multiplicities)
        homogeneous = np.column_stack((curve.poles * curve.weights[:, None], curve.weights))
        own_insertions = 0
        for value in curve.knots[1:-1]:
            for _ in range(curve.degree - int(np.count_nonzero(expanded == value))):
                insertions += 1
                own_insertions += 1
                _require(insertions <= MAX_KNOT_INSERTIONS and len(homogeneous) < MAX_REFINED_CURVE_POLES,
                         "CURVE_CORRESPONDENCE_REFINEMENT_BUDGET_EXCEEDED")
                k = min(int(np.searchsorted(expanded, value, side="right")) - 1, len(homogeneous) - 1)
                s = int(np.count_nonzero(expanded == value))
                p = curve.degree
                result = np.zeros((len(homogeneous) + 1, 4))
                result[:k - p + 1] = homogeneous[:k - p + 1]
                result[k - s + 1:] = homogeneous[k - s:]
                for i in range(k - p + 1, k - s + 1):
                    denominator = expanded[i + p] - expanded[i]
                    _require(denominator >= 8 * np.finfo(float).tiny, "UNCERTIFIED_KNOT_INSERTION_CONDITIONING")
                    alpha = (value - expanded[i]) / denominator
                    beta = (expanded[i + p] - value) / denominator
                    _require(0. <= alpha <= 1. and 0. <= beta <= 1., "INVALID_KNOT_INSERTION_COEFFICIENT")
                    # Compute both positive ratios directly; 1-alpha loses
                    # relative accuracy near a knot and is not used here.
                    result[i] = alpha * homogeneous[i] + beta * homogeneous[i - 1]
                expanded = np.insert(expanded, k + 1, value)
                homogeneous = result
        _require(np.isfinite(homogeneous).all() and homogeneous[:, 3].min() > 0.,
                 "UNCERTIFIED_RATIONAL_REFINEMENT")
        expected = curve.degree * (len(curve.knots) - 1) + 1
        _require(len(homogeneous) == expected, "BEZIER_REFINEMENT_TOPOLOGY_NOT_VERIFIED")
        points = homogeneous[:, :3] / homogeneous[:, 3, None] if own_insertions else curve.poles
        refined.append((points, homogeneous[:, 3] if own_insertions else curve.weights,
                        homogeneous, own_insertions))
    if reverse:
        points, weights, homogeneous, own_insertions = refined[1]
        refined[1] = (points[::-1], weights[::-1], homogeneous, own_insertions)
    maximum = 0.
    for segment in range(len(left.knots) - 1):
        start, stop = segment * left.degree, (segment + 1) * left.degree + 1
        points0, points1 = refined[0][0][start:stop], refined[1][0][start:stop]
        weights0, weights1 = refined[0][1][start:stop], refined[1][1][start:stop]
        w0, w1 = weights0 / weights0.max(), weights1 / weights1.max()
        rho = float(np.max(np.abs(w1 / w0 - 1.)))
        if not math.isfinite(rho) or rho >= 1.:
            return math.inf
        pole_error = float(np.linalg.norm(points0 + offset - points1, axis=1).max())
        diameter = float(np.linalg.norm(np.ptp(np.concatenate((points0 + offset, points1)), axis=0)))
        roundoff = 0.
        for curve_index, (points, weights, homogeneous, own_insertions) in enumerate(refined):
            original_curve = (left, right)[curve_index]
            operations = 32 * max(1, left.degree) * own_insertions + 32
            epsilon = np.finfo(float).eps
            gamma = operations * epsilon / (1. - operations * epsilon)
            original_homogeneous_scale = max(
                float(np.max(np.abs(original_curve.poles * original_curve.weights[:, None]))),
                float(np.max(original_curve.weights)))
            scale = ((max(float(np.max(np.abs(homogeneous))), original_homogeneous_scale, 1.)
                      + max(float(np.max(np.abs(points))), float(np.max(np.abs(original_curve.poles))), 1.))
                     / float(weights.min())
                     if own_insertions else max(float(np.max(np.abs(points))), 1.))
            # Insertion is a sequence of positive homogeneous convex combinations.
            # Both numerator and weight-denominator perturbations receive the
            # conservative 1/min_weight allowance when refinement took place.
            roundoff += 2 * gamma * (scale + diameter + float(np.linalg.norm(offset)))
        maximum = max(maximum, pole_error + diameter * 2 * rho / (1. - rho) + roundoff)
    return maximum


def _curve_proof(adaptor: Any) -> NativeCurve:
    """Capture only analytically finite bounded native curve representations."""
    from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
    from OCP.Geom import Geom_Ellipse, Geom_TrimmedCurve
    from OCP.GeomConvert import GeomConvert

    def vector(value: Any) -> np.ndarray:
        return np.array([value.X(), value.Y(), value.Z()], dtype=np.float64)

    first, last = float(adaptor.FirstParameter()), float(adaptor.LastParameter())
    _require(math.isfinite(first) and math.isfinite(last) and last > first, "INVALID_NATIVE_CURVE_INTERVAL")
    kind = adaptor.GetType()
    native_endpoints = np.stack([vector(adaptor.Value(first)), vector(adaptor.Value(last))])
    if kind == GeomAbs_Line:
        proof = NativeCurve("line", np.stack([vector(adaptor.Value(first)), vector(adaptor.Value(last))]))
    elif kind == GeomAbs_Circle:
        circle = adaptor.Circle()
        _require(last - first <= 2 * math.pi + GEOM_TOL, "INVALID_NATIVE_CURVE_INTERVAL")
        proof = NativeCurve("circle", np.stack([vector(adaptor.Value(first)), vector(adaptor.Value(last))]),
            center=vector(circle.Location()), axis=vector(circle.Axis().Direction()),
            x_axis=vector(circle.Position().XDirection()), y_axis=vector(circle.Position().YDirection()),
            radius=float(circle.Radius()), u0=first, u1=last)
    else:
        if kind == GeomAbs_BSplineCurve:
            # The OCP adaptor currently copies on this getter; explicitly copy
            # again before Segment so native topology can never be mutated.
            spline = adaptor.BSpline().Copy()
        elif kind == GeomAbs_BezierCurve:
            bezier = adaptor.Bezier()
            _require(bezier.Degree() <= MAX_CURVE_DEGREE and bezier.NbPoles() <= MAX_CURVE_POLES,
                     "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
            spline = GeomConvert.CurveToBSplineCurve_s(bezier)
        elif kind == GeomAbs_Ellipse:
            spline = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(Geom_Ellipse(adaptor.Ellipse()), first, last))
            first, last = float(spline.FirstParameter()), float(spline.LastParameter())
        else:
            raise UnfoldUnavailable("UNSUPPORTED_NATIVE_TRIM_CURVE")
        _require(spline.Degree() <= MAX_CURVE_DEGREE and spline.NbPoles() <= MAX_CURVE_POLES,
                 "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
        spline.Segment(first, last, 0.)
        _require(spline.FirstParameter() == first and spline.LastParameter() == last,
                 "NATIVE_CURVE_TRIM_PARAMETER_MISMATCH")
        _require(spline.NbPoles() <= MAX_CURVE_POLES and spline.NbKnots() <= MAX_CURVE_POLES,
                 "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
        poles = np.array([vector(spline.Pole(i)) for i in range(1, spline.NbPoles() + 1)])
        weights = np.array([spline.Weight(i) for i in range(1, spline.NbPoles() + 1)])
        knots = np.array([spline.Knot(i) for i in range(1, spline.NbKnots() + 1)])
        _require(np.isfinite(weights).all() and weights.min() > 0.
                 and weights.max() / weights.min() <= 1e12, "INVALID_NATIVE_CURVE_WEIGHTS")
        _require(np.isfinite(knots).all() and knots[-1] > knots[0], "INVALID_NATIVE_CURVE_KNOTS")
        proof = NativeCurve("spline", poles, int(spline.Degree()), weights / weights.max(),
            knots.copy(),
            np.array([spline.Multiplicity(i) for i in range(1, spline.NbKnots() + 1)], dtype=np.int32))
        _require(not spline.IsPeriodic() and proof.clamped_spline() and np.all(np.diff(proof.knots) > 0.),
                 "UNPROVEN_NATIVE_CURVE_CLAMPING")
        _require(np.linalg.norm(proof.poles[[0, -1]] - native_endpoints, axis=1).max() <= GEOM_TOL,
                 "NATIVE_CURVE_TRIM_ENDPOINT_MISMATCH")
    _require(np.isfinite(proof.poles).all(), "NONFINITE_NATIVE_CURVE")
    return proof


def _surface_proof(surface: Any) -> dict[str, Any]:
    """Analytical chart or exact positive-weight affine extrusion proof."""
    from OCP.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_SurfaceOfExtrusion

    def vector(value: Any) -> np.ndarray:
        return np.array([value.X(), value.Y(), value.Z()], dtype=np.float64)

    kind = surface.GetType()
    if kind in (GeomAbs_Plane, GeomAbs_Cylinder):
        return {"kind": "plane" if kind == GeomAbs_Plane else "cylinder", "pole_cost": 0}
    if kind == GeomAbs_SurfaceOfExtrusion:
        curve = _curve_proof(surface.BasisCurve())
        axis = _unit(vector(surface.Direction()))
        _require(_curve_in_perpendicular_plane(curve, axis), "NONPLANAR_EXTRUSION_BASE_CURVE")
        return {"kind": "extruded_boundary", "origin": curve.poles[0].copy(), "axis": axis,
                "v0": float(surface.FirstVParameter()), "v1": float(surface.LastVParameter()),
                "extrusion_curve": curve, "pole_cost": len(curve.poles)}
    _require(kind == GeomAbs_BSplineSurface, "UNSUPPORTED_NATIVE_SURFACE")
    spline = surface.BSpline()
    nu, nv = int(spline.NbUPoles()), int(spline.NbVPoles())
    _require(nu * nv <= MAX_SURFACE_POLES and max(spline.UDegree(), spline.VDegree()) <= MAX_CURVE_DEGREE,
             "NATIVE_SURFACE_COMPLEXITY_BUDGET_EXCEEDED")
    poles = np.array([[vector(spline.Pole(i, j)) for j in range(1, nv + 1)] for i in range(1, nu + 1)])
    weights = np.array([[spline.Weight(i, j) for j in range(1, nv + 1)] for i in range(1, nu + 1)])
    _require(np.isfinite(poles).all() and np.isfinite(weights).all() and weights.min() > 0.
             and weights.max() / weights.min() <= 1e12, "INVALID_NATIVE_SURFACE_COEFFICIENTS")
    # Positive-weight tensor basis preserves any plane containing all poles.
    points = poles.reshape(-1, 3)
    first = points[0]
    delta = points - first
    farthest = delta[int(np.argmax(np.linalg.norm(delta, axis=1)))]
    crosses = np.cross(farthest, delta)
    normal_candidate = crosses[int(np.argmax(np.linalg.norm(crosses, axis=1)))]
    if np.linalg.norm(normal_candidate) > GEOM_TOL:
        normal = _unit(normal_candidate)
        if np.max(np.abs(delta @ normal)) <= GEOM_TOL:
            return {"kind": "plane", "origin": first.copy(), "normal": normal, "pole_cost": nu * nv}
    # Only a degree-one, two-row tensor direction with identical weights and
    # constant pole displacement is accepted as a straight extrusion. A merely
    # sampled-looking ruled/doubly curved surface cannot satisfy this identity.
    possibilities = []
    if (spline.VDegree() == 1 and nv == 2 and spline.NbVKnots() == 2
            and spline.VMultiplicity(1) == 2 and spline.VMultiplicity(2) == 2):
        possibilities.append((poles[:, 0], poles[:, 1], weights[:, 0], weights[:, 1],
            int(spline.UDegree()), [spline.UKnot(i) for i in range(1, spline.NbUKnots() + 1)],
            [spline.UMultiplicity(i) for i in range(1, spline.NbUKnots() + 1)],
            float(spline.VKnot(1)), float(spline.VKnot(2)),
            float(surface.FirstVParameter()), float(surface.LastVParameter())))
    if (spline.UDegree() == 1 and nu == 2 and spline.NbUKnots() == 2
            and spline.UMultiplicity(1) == 2 and spline.UMultiplicity(2) == 2):
        possibilities.append((poles[0], poles[1], weights[0], weights[1],
            int(spline.VDegree()), [spline.VKnot(i) for i in range(1, spline.NbVKnots() + 1)],
            [spline.VMultiplicity(i) for i in range(1, spline.NbVKnots() + 1)],
            float(spline.UKnot(1)), float(spline.UKnot(2)),
            float(surface.FirstUParameter()), float(surface.LastUParameter())))
    for base, second, w0, w1, degree, knots, mult, k0, k1, trim0, trim1 in possibilities:
        displacement = second - base
        vector_offset = displacement[0]
        length = float(np.linalg.norm(vector_offset))
        if (length <= GEOM_TOL or np.linalg.norm(displacement - vector_offset, axis=1).max() > GEOM_TOL
                or not np.array_equal(w0, w1)):
            continue
        axis = vector_offset / length
        knots = np.asarray(knots, dtype=float)
        curve = NativeCurve("spline", base.copy(), degree, w0 / w0.max(),
                            knots.copy(), np.asarray(mult, dtype=np.int32))
        if not _curve_in_perpendicular_plane(curve, axis):
            continue
        _require(k1 > k0 and trim1 > trim0 and trim0 >= k0
                 and trim1 <= k1, "INVALID_NATIVE_EXTRUSION_INTERVAL")
        return {"kind": "extruded_boundary", "origin": base[0].copy(), "axis": axis,
                "v0": length * (trim0 - k0) / (k1 - k0), "v1": length * (trim1 - k0) / (k1 - k0),
                "extrusion_curve": curve, "pole_cost": nu * nv}
    raise UnfoldUnavailable("UNPROVEN_NONPLANAR_BSPLINE_SURFACE")


@dataclass
class SkinFace:
    """Import-local native face chart and its unchanged trimmed triangulation."""
    index: int
    kind: str
    vertices: np.ndarray
    triangles: np.ndarray
    edges: dict[int, np.ndarray | None] = field(default_factory=dict)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray | None = None
    axis: np.ndarray | None = None
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None
    radius: float | None = None
    u0: float | None = None
    u1: float | None = None
    v0: float | None = None
    v1: float | None = None
    extrusion_curve: NativeCurve | None = None
    edge_curves: dict[int, NativeCurve] = field(default_factory=dict)

    @property
    def area(self) -> float:
        points = self.vertices[self.triangles]
        return float(np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                             points[:, 2] - points[:, 0]), axis=1).sum() / 2.)


@dataclass
class BendLink:
    cylinder: int
    planes: tuple[int, int]
    feature_id: str
    seams: dict[int, np.ndarray]
    axis_record: dict[str, Any]


def _envelope(source_hash: str | None, reason: str | None = None) -> dict[str, Any]:
    return {"status": "UNAVAILABLE" if reason else "AVAILABLE", "reason": reason,
            "model": MODEL, "units": "assumed_mm", "units_verified": False,
            "manufacturing_ready": False, "vertices": [], "triangles": [],
            "bend_lines": [], "flange_surfaces": [], "source_sha256": source_hash,
            "diagnostics": {"mechanism_version": MECHANISM_VERSION,
                "limitations": ["VISUAL_ONLY_SELECTED_CAD_SKIN",
                "NOT_NEUTRAL_AXIS_OR_K_FACTOR_BLANK", "UNITS_ASSUMED_NOT_VERIFIED",
                "PLANAR_FLANGES_AND_PARTIAL_CYLINDERS_ONLY", "TREE_TOPOLOGY_ONLY",
                "MAXIMUM_BEND_SWEEP_175_DEGREES", "SINGLE_VALID_SOLID_ONLY",
                "BOUNDED_ANALYTICAL_AND_RATIONAL_SPLINE_NATIVE_TRIMS",
                "NONPLANAR_BOUNDARY_WALLS_REQUIRE_EXACT_AFFINE_EXTRUSION_PROOF",
                "CIRCULAR_RAIL_SET_FALLBACK_REQUIRES_COMPLETE_CYLINDRICAL_HOLE_WALL",
                "COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED"]}}


def unavailable_unfold_preview(reason: str) -> dict[str, Any]:
    """Public empty envelope for callers without an eligible CAD source."""
    return _envelope(None, reason)


def _validate_face(face: SkinFace) -> None:
    points, triangles = np.asarray(face.vertices), np.asarray(face.triangles)
    _require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
             and np.isfinite(points).all(), "NONFINITE_GEOMETRY")
    _require(triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) > 0
             and np.issubdtype(triangles.dtype, np.integer)
             and triangles.min() >= 0 and triangles.max() < len(points), "INVALID_NATIVE_TRIANGLES")
    _require(np.isfinite(face.origin).all() and face.origin.shape == (3,), "NONFINITE_GEOMETRY")
    _require(math.isfinite(face.area) and face.area > GEOM_TOL ** 2, "DEGENERATE_NATIVE_FACE_MESH")
    if face.kind == "plane":
        normal = _unit(face.normal)
        _require(np.max(np.abs((points - face.origin) @ normal)) <= GEOM_TOL,
                 "NATIVE_PLANE_OWNERSHIP_MISMATCH")
    elif face.kind == "cylinder":
        axis, x, y = _unit(face.axis), _unit(face.x_axis), _unit(face.y_axis)
        values = [face.radius, face.u0, face.u1, face.v0, face.v1]
        _require(all(value is not None and math.isfinite(value) for value in values), "NONFINITE_GEOMETRY")
        _require(face.radius > GEOM_TOL and face.u1 > face.u0 and face.v1 > face.v0,
                 "DEGENERATE_GEOMETRY")
        _require(abs(np.dot(axis, x)) < GEOM_TOL and abs(np.dot(axis, y)) < GEOM_TOL
                 and np.dot(np.cross(x, y), axis) > 1. - GEOM_TOL, "INVALID_CYLINDER_FRAME")
        offsets = points - face.origin
        axial = offsets @ axis
        radial = offsets - axial[:, None] * axis
        _require(np.max(np.abs(np.linalg.norm(radial, axis=1) - face.radius)) <= GEOM_TOL,
                 "NATIVE_CYLINDER_OWNERSHIP_MISMATCH")
        _require(axial.min() >= face.v0 - GEOM_TOL and axial.max() <= face.v1 + GEOM_TOL,
                 "NATIVE_CYLINDER_TRIM_MISMATCH")
        if face.u1 - face.u0 < 2 * math.pi - GEOM_TOL:
            _cylinder_angles(face, points)
    elif face.kind == "extruded_boundary":
        axis = _unit(face.axis)
        _require(face.extrusion_curve is not None and _curve_in_perpendicular_plane(face.extrusion_curve, axis),
                 "UNPROVEN_NATIVE_EXTRUSION")
        _require(face.v0 is not None and face.v1 is not None and math.isfinite(face.v0)
                 and math.isfinite(face.v1) and face.v1 > face.v0, "INVALID_NATIVE_EXTRUSION_INTERVAL")
        axial = (points - face.origin) @ axis
        _require(axial.min() >= face.v0 - GEOM_TOL and axial.max() <= face.v1 + GEOM_TOL,
                 "NATIVE_EXTRUSION_OWNERSHIP_MISMATCH")
    else:
        raise UnfoldUnavailable("UNSUPPORTED_NATIVE_SURFACE")


def _cylinder_angles(face: SkinFace, points: np.ndarray) -> np.ndarray:
    offsets = points - face.origin
    raw = np.arctan2(offsets @ face.y_axis, offsets @ face.x_axis)
    middle = (face.u0 + face.u1) / 2.
    angles = raw + 2 * math.pi * np.rint((middle - raw) / (2 * math.pi))
    _require(angles.min() >= face.u0 - GEOM_TOL and angles.max() <= face.u1 + GEOM_TOL,
             "NATIVE_CYLINDER_TRIM_MISMATCH")
    return angles


def _matching_axes(candidates: list[dict[str, Any]], axes: list[dict[str, Any]]) -> list[int]:
    _require(len(candidates) > 0 and len(candidates) == len(axes), "CAD_CANDIDATE_COUNT_MISMATCH")
    identifiers = [row.get("feature_id") for row in candidates]
    _require(all(isinstance(value, str) and value for value in identifiers)
             and len(set(identifiers)) == len(identifiers), "DUPLICATE_OR_INVALID_CANDIDATE_IDS")
    matches = []
    for candidate in candidates:
        try:
            line = np.asarray([candidate["line_start"], candidate["line_end"]], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnfoldUnavailable("INVALID_CANDIDATE_LINE") from exc
        _require(line.shape == (2, 3) and np.isfinite(line).all(), "NONFINITE_CANDIDATE_LINE")
        found = []
        for index, axis in enumerate(axes):
            native = np.asarray([axis["canonical_line_start"], axis["canonical_line_end"]])
            _require(native.shape == (2, 3) and np.isfinite(native).all(), "INVALID_NATIVE_AXIS")
            error = min(np.linalg.norm(line - native, axis=1).max(),
                        np.linalg.norm(line - native[::-1], axis=1).max())
            if error <= CANDIDATE_ROUNDING_TOL:
                found.append(index)
        _require(len(found) == 1, "CAD_CANDIDATE_AXIS_MISMATCH")
        matches.append(found[0])
    _require(len(set(matches)) == len(matches), "DUPLICATE_CANDIDATE_AXIS_ASSIGNMENT")
    return matches


def _tangent_seam(plane: SkinFace, cylinder: SkinFace, owners: dict[int, set[int]]) -> np.ndarray | None:
    # End caps have normals parallel to the axis. Thickness caps have a radial
    # plane passing through the axis. Neither can satisfy this tangent test.
    normal, axis = _unit(plane.normal), _unit(cylinder.axis)
    if (abs(np.dot(normal, axis)) > GEOM_TOL
            or abs(abs(np.dot(cylinder.origin - plane.origin, normal)) - cylinder.radius) > GEOM_TOL):
        return None
    shared = [edge for edge in cylinder.edges if plane.index in owners[edge]]
    _require(len(shared) == 1, "SPLIT_OR_AMBIGUOUS_TANGENT_SEAM")
    seam = cylinder.edges[shared[0]]
    _require(seam is not None and seam.shape == (2, 3) and np.isfinite(seam).all(),
             "NON_STRAIGHT_TANGENT_SEAM")
    direction = _unit(seam[1] - seam[0])
    _require(abs(np.dot(direction, axis)) > 1. - GEOM_TOL, "NON_AXIAL_TANGENT_SEAM")
    angles = _cylinder_angles(cylinder, seam)
    _require(np.ptp(angles) <= GEOM_TOL and
             min(abs(angles.mean() - cylinder.u0), abs(angles.mean() - cylinder.u1)) <= GEOM_TOL,
             "TANGENT_SEAM_NOT_NATIVE_TRIM_BOUNDARY")
    return seam


def _skin_graph(faces: list[SkinFace], axes: list[dict[str, Any]],
                candidates: list[dict[str, Any]]) -> tuple[list[BendLink], dict[int, list[BendLink]]]:
    _require(len(faces) <= MAX_FACES and sum(len(face.triangles) for face in faces) <= MAX_TRIANGLES,
             "GEOMETRY_BUDGET_EXCEEDED")
    _require([face.index for face in faces] == list(range(len(faces))), "FACE_ENUMERATION_MISMATCH")
    for face in faces:
        _validate_face(face)
    owners: dict[int, set[int]] = {}
    for face in faces:
        for edge in face.edges:
            owners.setdefault(edge, set()).add(face.index)
    _require(all(len(value) <= 2 for value in owners.values()), "NON_MANIFOLD_EDGE_INCIDENCE")
    matches = _matching_axes(candidates, axes)
    cylinder_features: dict[int, tuple[str, dict[str, Any]]] = {}
    thicknesses = []
    for candidate, match in zip(candidates, matches):
        record = axes[match]
        indices = [int(value) - 1 for value in record["face_indices"]]
        _require(len(indices) == 2 and len(set(indices)) == 2
                 and all(0 <= index < len(faces) for index in indices), "UNSUPPORTED_BEND_CYLINDER_GROUP")
        cylinders = [faces[index] for index in indices]
        _require(all(face.kind == "cylinder" for face in cylinders), "NATIVE_AXIS_FACE_MISMATCH")
        first, second = cylinders
        _require(abs(np.dot(first.axis, second.axis)) > 1. - GEOM_TOL
                 and np.linalg.norm(np.cross(second.origin - first.origin, first.axis)) <= GEOM_TOL,
                 "UNEQUAL_OR_NONCOAXIAL_BEND_SKINS")
        actual = sorted(face.radius for face in cylinders)
        _require(np.allclose(actual, sorted(record["radii_mm"]), atol=GEOM_TOL, rtol=0.),
                 "NATIVE_AXIS_RADIUS_MISMATCH")
        thickness = actual[1] - actual[0]
        _require(thickness > GEOM_TOL, "EQUAL_RADIUS_BEND_SKINS")
        thicknesses.append(thickness)
        sweeps = [face.u1 - face.u0 for face in cylinders]
        _require(max(sweeps) <= math.radians(MAX_SWEEP_DEG) + GEOM_TOL,
                 "BEND_SWEEP_EXCEEDS_SUPPORTED_175_DEGREES")
        _require(abs(sweeps[0] - sweeps[1]) <= GEOM_TOL, "UNEQUAL_BEND_SKIN_SWEEPS")
        radial_pairs = [np.stack([math.cos(u) * face.x_axis + math.sin(u) * face.y_axis
                                 for u in (face.u0, (face.u0 + face.u1) / 2., face.u1)])
                        for face in cylinders]
        _require(max(actual) * min(np.max(np.linalg.norm(radial_pairs[0] - radial_pairs[1], axis=1)),
                     np.max(np.linalg.norm(radial_pairs[0] - radial_pairs[1][::-1], axis=1))) <= GEOM_TOL,
                 "UNEQUAL_BEND_SKIN_ANGULAR_TRIMS")
        spans = [np.stack([face.origin + face.axis * value for value in (face.v0, face.v1)])
                 for face in cylinders]
        _require(min(np.max(np.linalg.norm(spans[0] - spans[1], axis=1)),
                     np.max(np.linalg.norm(spans[0] - spans[1][::-1], axis=1))) <= GEOM_TOL,
                 "UNEQUAL_BEND_SKIN_AXIAL_TRIMS")
        for index in indices:
            _require(index not in cylinder_features, "DUPLICATE_BEND_FACE_ASSIGNMENT")
            cylinder_features[index] = (candidate["feature_id"], record)
    _require(max(thicknesses) - min(thicknesses) <= GEOM_TOL, "VARYING_SHEET_THICKNESS")
    links, adjacency = [], {}
    for face in faces:
        if face.kind != "cylinder":
            continue
        sweep = face.u1 - face.u0
        if sweep >= 2 * math.pi - GEOM_TOL:
            _require(face.index not in cylinder_features, "FULL_CYLINDER_IS_NOT_BEND")
            continue  # Native full-cylinder hole walls are not forming skin.
        if face.index not in cylinder_features:
            continue  # Only boundary proof below can explain a non-bend wall.
        tangent = {}
        adjacent = set().union(*(owners[edge] for edge in face.edges)) - {face.index}
        for index in sorted(adjacent):
            if faces[index].kind == "plane":
                seam = _tangent_seam(faces[index], face, owners)
                if seam is not None:
                    tangent[index] = seam
            elif index in cylinder_features:
                raise UnfoldUnavailable("DIRECT_CYLINDER_TO_CYLINDER_FORMING_EDGE")
        _require(len(tangent) == 2, "BEND_REQUIRES_TWO_TANGENT_PLANAR_FLANGES")
        feature, axis_record = cylinder_features[face.index]
        link = BendLink(face.index, tuple(tangent), feature, tangent, axis_record)
        links.append(link)
        for index in tangent:
            adjacency.setdefault(index, []).append(link)
    return links, adjacency


def _select_skin(faces: list[SkinFace], links: list[BendLink],
                 adjacency: dict[int, list[BendLink]], candidate_count: int) -> tuple[int, list[BendLink]]:
    seen, eligible = set(), []
    for start in sorted(adjacency):
        if start in seen:
            continue
        nodes, component_links, stack = set(), {}, [start]
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            for link in adjacency[node]:
                component_links[link.cylinder] = link
                stack.extend(index for index in link.planes if index not in nodes)
        seen.update(nodes)
        features = [link.feature_id for link in component_links.values()]
        _require(len(features) == len(set(features)), "DUPLICATE_BEND_ASSIGNMENT_IN_SKIN")
        _require(len(component_links) == len(nodes) - 1, "CYCLIC_SKIN_GRAPH")
        if len(features) == candidate_count:
            root = max(nodes, key=lambda index: (faces[index].area, -index))
            eligible.append((faces[root].area, -root, root, list(component_links.values())))
    _require(bool(eligible), "NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS")
    _, _, root, selected = max(eligible, key=lambda value: value[:2])
    return root, selected


def _verify_skin_boundary(faces: list[SkinFace], links: list[BendLink],
                          adjacency: dict[int, list[BendLink]], selected: list[BendLink]) -> None:
    """Prove excluded boundary neighbors are thickness caps or through-hole walls.

    Both paired skins must be complete trees. An arbitrary planar appendage is
    not a thickness cap merely because it is outside the bend-link graph.
    """
    components, seen = [], set()
    all_features = {link.feature_id for link in links}
    for start in sorted(adjacency):
        if start in seen:
            continue
        nodes, cylinders, features, stack = set(), set(), set(), [start]
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            for link in adjacency[node]:
                cylinders.add(link.cylinder)
                features.add(link.feature_id)
                stack.extend(index for index in link.planes if index not in nodes)
        seen.update(nodes)
        _require(features == all_features, "INCOMPLETE_OPPOSITE_SKIN_GRAPH")
        components.append(nodes | cylinders)
    _require(len(components) == 2, "TWO_COMPLETE_SKIN_LAYERS_REQUIRED")
    owners: dict[int, set[int]] = {}
    for face in faces:
        for edge in face.edges:
            owners.setdefault(edge, set()).add(face.index)
    paired = {}
    for link in links:
        other = [value.cylinder for value in links if value.feature_id == link.feature_id
                 and value.cylinder != link.cylinder]
        _require(len(other) == 1, "UNSUPPORTED_BEND_CYLINDER_GROUP")
        paired[link.cylinder] = other[0]
    thickness = abs(faces[selected[0].cylinder].radius - faces[paired[selected[0].cylinder]].radius)
    explained = set().union(*components)

    def native_curve(face: SkinFace, edge: int) -> NativeCurve | None:
        if edge in face.edge_curves:
            return face.edge_curves[edge]
        rail = face.edges[edge]
        return NativeCurve("line", rail) if rail is not None else None

    def translated_opposite_rail(face: SkinFace, edge: int, neighbor: SkinFace,
                                 opposite: set[int], adjacent: set[int]) -> bool:
        curve = native_curve(face, edge)
        if curve is None:
            return False
        normal = _unit(face.normal)
        for other_index in sorted(opposite & adjacent):
            other = faces[other_index]
            if other.kind != "plane" or abs(np.dot(_unit(other.normal), normal)) < 1. - GEOM_TOL:
                continue
            separation = float(np.dot(other.origin - face.origin, normal))
            if abs(abs(separation) - thickness) > GEOM_TOL:
                continue
            slab = (neighbor.vertices - face.origin) @ normal * math.copysign(1., separation)
            if slab.min() < -GEOM_TOL or slab.max() > thickness + GEOM_TOL:
                continue
            for other_edge in set(other.edges) & set(neighbor.edges):
                other_curve = native_curve(other, other_edge)
                if other_curve is not None and _translated_curves(curve, other_curve, separation * normal):
                    return True
            # Native STEP can split one closed hole rail differently on its two
            # plane skins. Only whole closed circular sets on a complete native
            # cylindrical wall have this bounded fallback; partial rounded-slot
            # walls and arbitrary curve unions retain the one-to-one proof.
            if neighbor.kind == "cylinder":
                left_curves = [native_curve(face, value) for value in sorted(set(face.edges) & set(neighbor.edges))]
                right_curves = [native_curve(other, value) for value in sorted(set(other.edges) & set(neighbor.edges))]
                if (all(value is not None for value in left_curves + right_curves)
                        and _closed_circular_rail_sets_proven(neighbor, face, other,
                            left_curves, right_curves, separation * normal)):
                    return True
        return False

    for layer_index, layer in enumerate(components):
        opposite = components[1 - layer_index]
        for index in sorted(layer):
            face = faces[index]
            for edge, rail in face.edges.items():
                _require(len(owners[edge]) == 2, "OPEN_SELECTED_SKIN_BOUNDARY")
                neighbor_index = next(iter(owners[edge] - {index}))
                if neighbor_index in layer:
                    continue
                _require(neighbor_index not in opposite, "SKIN_LAYERS_JOIN_WITHOUT_THICKNESS_CAP")
                neighbor = faces[neighbor_index]
                adjacent_to_cap = set().union(*(owners[value] for value in neighbor.edges))
                if neighbor.kind == "plane" and face.kind == "cylinder":
                    # An annular axial end cap must join this skin to its exact
                    # paired-radius counterpart, not another arbitrary patch.
                    _require(abs(np.dot(_unit(neighbor.normal), _unit(face.axis))) > 1. - GEOM_TOL
                             and paired[index] in adjacent_to_cap,
                             "UNKNOWN_CYLINDRICAL_SKIN_BOUNDARY")
                elif neighbor.kind == "plane" and face.kind == "plane":
                    normal = _unit(face.normal)
                    _require(abs(np.dot(_unit(neighbor.normal), normal)) <= GEOM_TOL
                             and rail is not None, "UNKNOWN_PLANAR_SKIN_BOUNDARY")
                    # An axial annular end cap spans the whole bent profile, not
                    # just this plane's thickness slab. Its linear rail proof is
                    # retained separately from the curved wall slab proof.
                    curve = native_curve(face, edge)
                    proven = False
                    for other_index in opposite & adjacent_to_cap:
                        other = faces[other_index]
                        if other.kind != "plane" or abs(np.dot(_unit(other.normal), normal)) < 1. - GEOM_TOL:
                            continue
                        separation = float(np.dot(other.origin - face.origin, normal))
                        if abs(abs(separation) - thickness) > GEOM_TOL:
                            continue
                        for other_edge in set(other.edges) & set(neighbor.edges):
                            other_curve = native_curve(other, other_edge)
                            if (curve is not None and other_curve is not None
                                    and _translated_curves(curve, other_curve, separation * normal)):
                                proven = True
                    _require(proven, "UNKNOWN_PLANAR_SKIN_BOUNDARY")
                elif neighbor.kind in ("cylinder", "extruded_boundary") and face.kind == "plane":
                    # A rounded outer edge, slot, or curved cutout wall must be a
                    # proven normal extrusion through the exact known thickness.
                    _require(abs(np.dot(_unit(neighbor.axis), _unit(face.normal))) > 1. - GEOM_TOL
                             and abs(neighbor.v1 - neighbor.v0 - thickness) <= GEOM_TOL,
                             "UNKNOWN_HOLE_WALL_BOUNDARY")
                    _require(translated_opposite_rail(face, edge, neighbor, opposite, adjacent_to_cap),
                             "UNKNOWN_HOLE_WALL_BOUNDARY")
                else:
                    raise UnfoldUnavailable("UNKNOWN_SELECTED_SKIN_BOUNDARY")
                explained.add(neighbor_index)
    # A boss, return, or material island reachable only through a cap is not
    # silently omitted. Every source face needs one of the proven skin/cap/hole
    # roles before this can be presented as a complete selected-skin preview.
    _require(explained == set(range(len(faces))), "UNEXPLAINED_NATIVE_MATERIAL_SURFACE")


def _child_transform(parent_rotation: np.ndarray, parent_translation: np.ndarray,
                     cylinder: SkinFace, parent_seam: np.ndarray, child_seam: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    parent_u = float(_cylinder_angles(cylinder, parent_seam).mean())
    child_u = float(_cylinder_angles(cylinder, child_seam).mean())
    delta = child_u - parent_u  # Actual native trimmed interval, not principal atan2.
    _require(GEOM_TOL < abs(delta) <= math.radians(MAX_SWEEP_DEG) + GEOM_TOL,
             "UNSUPPORTED_BEND_SWEEP")
    axis = _unit(cylinder.axis)
    radial = math.cos(parent_u) * cylinder.x_axis + math.sin(parent_u) * cylinder.y_axis
    p = cylinder.origin + cylinder.radius * radial
    q = cylinder.origin + cylinder.radius * (math.cos(child_u) * cylinder.x_axis
                                             + math.sin(child_u) * cylinder.y_axis)
    tangent = np.cross(axis, radial)
    native_rotation = _rotation(axis, -delta)
    rotation = parent_rotation @ native_rotation
    translation = parent_translation + parent_rotation @ (p + cylinder.radius * delta * tangent - native_rotation @ q)
    _require(abs(np.linalg.det(rotation) - 1.) <= GEOM_TOL, "REFLECTED_UNFOLD_TRANSFORM")
    return rotation, translation, delta, parent_u, tangent


def _unwrap_cylinder(face: SkinFace, rotation: np.ndarray, translation: np.ndarray,
                     parent_u: float) -> np.ndarray:
    angles = _cylinder_angles(face, face.vertices)
    radial = math.cos(parent_u) * face.x_axis + math.sin(parent_u) * face.y_axis
    tangent = np.cross(face.axis, radial)
    axial = (face.vertices - face.origin) @ face.axis
    native_flat = (face.origin + face.radius * radial + axial[:, None] * face.axis
                   + face.radius * (angles - parent_u)[:, None] * tangent)
    return native_flat @ rotation.T + translation


def _triangle_intersection_area(left: np.ndarray, right: np.ndarray) -> float:
    """Convex polygon clipping; shared seams/points have zero intersection area."""
    polygon = [point.copy() for point in left]
    def cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    signed = cross(right[1] - right[0], right[2] - right[0])
    if signed < 0:
        right = right[::-1]
    for i in range(3):
        a, b = right[i], right[(i + 1) % 3]
        result = []
        for j, point in enumerate(polygon):
            previous = polygon[j - 1]
            dp = cross(b - a, point - a)
            dq = cross(b - a, previous - a)
            if (dp >= 0.) != (dq >= 0.):
                result.append(previous + (point - previous) * dq / (dq - dp))
            if dp >= 0.:
                result.append(point)
        polygon = result
        if len(polygon) < 3:
            return 0.
    points = np.asarray(polygon)
    return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1)))) / 2.


def _check_overlap(flat: dict[int, np.ndarray], faces: list[SkinFace]) -> None:
    comparisons = 0
    ids = sorted(flat)
    for offset, left_id in enumerate(ids):
        left = flat[left_id][faces[left_id].triangles, :2]
        left_min, left_max = left.min(axis=1), left.max(axis=1)
        for right_id in ids[offset + 1:]:
            right = flat[right_id][faces[right_id].triangles, :2]
            right_min, right_max = right.min(axis=1), right.max(axis=1)
            if np.any(left_max.max(axis=0) <= right_min.min(axis=0) + GEOM_TOL) or np.any(
                    right_max.max(axis=0) <= left_min.min(axis=0) + GEOM_TOL):
                continue
            order = np.argsort(right_min[:, 0])
            sorted_min = right_min[order, 0]
            for i, triangle in enumerate(left):
                stop = int(np.searchsorted(sorted_min, left_max[i, 0] - GEOM_TOL))
                possible = order[:stop]
                possible = possible[(right_max[possible, 0] > left_min[i, 0] + GEOM_TOL)
                                    & (right_max[possible, 1] > left_min[i, 1] + GEOM_TOL)
                                    & (right_min[possible, 1] < left_max[i, 1] - GEOM_TOL)]
                comparisons += len(possible)
                _require(comparisons <= MAX_OVERLAP_COMPARISONS, "OVERLAP_CHECK_BUDGET_EXCEEDED")
                for j in possible:
                    _require(_triangle_intersection_area(triangle, right[j]) <= GEOM_TOL ** 2,
                             "DEVELOPED_SKIN_SELF_OVERLAP")


def _develop_faces(faces: list[SkinFace], axes: list[dict[str, Any]],
                   candidates: list[dict[str, Any]], source_hash: str) -> dict[str, Any]:
    """Pure chart/graph development seam, also used by bounded synthetic tests."""
    links, adjacency = _skin_graph(faces, axes, candidates)
    root, selected = _select_skin(faces, links, adjacency, len(candidates))
    selected_ids = {link.cylinder for link in selected}
    selected_planes = {index for link in selected for index in link.planes}
    selected_edges = {edge for index in selected_planes for edge in faces[index].edges}
    for face in faces:
        if face.kind != "plane" or face.index in selected_planes:
            continue
        if not selected_edges.intersection(face.edges):
            continue
        for index in selected_planes:
            plane = faces[index]
            if (set(plane.edges).intersection(face.edges)
                    and abs(np.dot(_unit(face.normal), _unit(plane.normal))) > 1. - GEOM_TOL
                    and abs(np.dot(face.origin - plane.origin, _unit(plane.normal))) <= GEOM_TOL):
                raise UnfoldUnavailable("COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED")
    _verify_skin_boundary(faces, links, adjacency, selected)
    root_face = faces[root]
    normal = _unit(root_face.normal)
    first_link = next(link for link in adjacency[root] if link.cylinder in selected_ids)
    x_axis = _unit(faces[first_link.cylinder].axis)
    rotation = np.stack((x_axis, _unit(np.cross(normal, x_axis)), normal))
    translation = -rotation @ root_face.origin
    transforms = {root: (rotation, translation)}
    flat = {root: root_face.vertices @ rotation.T + translation}
    bend_lines, overlays, done, stack = [], [], set(), [root]
    while stack:
        parent = stack.pop()
        parent_rotation, parent_translation = transforms[parent]
        for link in adjacency[parent]:
            if link.cylinder not in selected_ids or link.cylinder in done:
                continue
            done.add(link.cylinder)
            child = next(index for index in link.planes if index != parent)
            _require(child not in transforms, "CYCLIC_SKIN_GRAPH")
            cylinder = faces[link.cylinder]
            child_rotation, child_translation, delta, parent_u, tangent = _child_transform(
                parent_rotation, parent_translation, cylinder, link.seams[parent], link.seams[child])
            # Check planar material extends away from each side of the arc strip.
            p = link.seams[parent][0]
            q = link.seams[child][0]
            sign = math.copysign(1., delta)
            _require(np.max((faces[parent].vertices - p) @ tangent * sign) <= GEOM_TOL,
                     "PARENT_FLANGE_EXTENDS_INTO_DEVELOPED_BEND")
            child_tangent = _rotation(cylinder.axis, delta) @ tangent
            _require(np.min((faces[child].vertices - q) @ child_tangent * sign) >= -GEOM_TOL,
                     "CHILD_FLANGE_EXTENDS_INTO_DEVELOPED_BEND")
            child_flat = faces[child].vertices @ child_rotation.T + child_translation
            cylinder_flat = _unwrap_cylinder(cylinder, parent_rotation, parent_translation, parent_u)
            for plane_id, plane_rotation, plane_translation, seam in (
                    (parent, parent_rotation, parent_translation, link.seams[parent]),
                    (child, child_rotation, child_translation, link.seams[child])):
                seam_u = float(_cylinder_angles(cylinder, seam).mean())
                radial = math.cos(parent_u) * cylinder.x_axis + math.sin(parent_u) * cylinder.y_axis
                seam_axial = (seam - cylinder.origin) @ cylinder.axis
                arc_seam = (cylinder.origin + cylinder.radius * radial
                            + seam_axial[:, None] * cylinder.axis
                            + cylinder.radius * (seam_u - parent_u) * tangent)
                _require(np.max(np.linalg.norm(arc_seam @ parent_rotation.T + parent_translation
                                              - (seam @ plane_rotation.T + plane_translation), axis=1)) <= GEOM_TOL,
                         "DEVELOPED_SEAM_CLOSURE_FAILED")
            transforms[child] = (child_rotation, child_translation)
            flat[child], flat[link.cylinder] = child_flat, cylinder_flat
            middle = cylinder.origin + cylinder.radius * (
                math.cos(parent_u) * cylinder.x_axis + math.sin(parent_u) * cylinder.y_axis)
            line = np.stack([middle + cylinder.v0 * cylinder.axis + cylinder.radius * delta / 2 * tangent,
                             middle + cylinder.v1 * cylinder.axis + cylinder.radius * delta / 2 * tangent])
            line = line @ parent_rotation.T + parent_translation
            # Native plane normals can be reversed independently. The sign is
            # therefore deliberately not promoted to an authoritative direction.
            bend_lines.append({"feature_id": link.feature_id, "line_start": line[0].tolist(),
                               "line_end": line[1].tolist(), "bend_direction": "unknown",
                               "sweep_deg": math.degrees(abs(delta))})
            overlays.append({"feature_id": link.feature_id, "vertices": child_flat.reshape(-1).tolist(),
                             "triangles": faces[child].triangles.reshape(-1).tolist()})
            stack.append(child)
    _require(len(done) == len(candidates), "INCOMPLETE_SELECTED_SKIN")
    for points in flat.values():
        _require(np.isfinite(points).all() and np.max(np.abs(points[:, 2])) <= GEOM_TOL,
                 "DEVELOPED_SKIN_NOT_PLANAR")
        points[:, 2] = 0.
    for line in bend_lines:
        _require(max(abs(line["line_start"][2]), abs(line["line_end"][2])) <= GEOM_TOL,
                 "DEVELOPED_BEND_LINE_NOT_PLANAR")
        line["line_start"][2] = line["line_end"][2] = 0.
    _check_overlap(flat, faces)
    payload = _envelope(source_hash)
    vertex_offset = 0
    for index in sorted(flat):
        payload["vertices"].extend(flat[index].reshape(-1).tolist())
        payload["triangles"].extend((faces[index].triangles + vertex_offset).reshape(-1).tolist())
        vertex_offset += len(flat[index])
    payload["bend_lines"] = sorted(bend_lines, key=lambda row: row["feature_id"])
    payload["flange_surfaces"] = sorted(overlays, key=lambda row: row["feature_id"])
    payload["diagnostics"].update({"root_source_face_index": root,
        "selected_source_face_indices": sorted(flat), "selected_cylinder_face_indices": sorted(done),
        "candidate_count": len(candidates), "native_face_count": len(faces),
        "selected_skin_rule": "LARGEST_PLANAR_ROOT_OF_COMPLETE_TREE_SKIN",
        "seam_closure_checked": True, "native_surface_ownership_checked": True,
        "cross_face_overlap_checked": True, "native_surface_coverage_checked": True,
        "excluded_caps_and_hole_walls_checked": True,
        "direction_status": "UNKNOWN_NATIVE_NORMAL_SIGN_NOT_AUTHORITY"})
    return payload


def _read_native(path: Path) -> tuple[list[SkinFace], list[dict[str, Any]]]:
    """Read only an already byte-bound, private STEP snapshot, never at import."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from scripts.extract_step_topology_bend_axes import extract_step_bend_axes

    def vector(value: Any) -> np.ndarray:
        return np.array([value.X(), value.Y(), value.Z()], dtype=np.float64)

    reader = STEPControl_Reader()
    _require(reader.ReadFile(str(path)) == IFSelect_RetDone and reader.TransferRoots() > 0,
             "STEP_READ_UNAVAILABLE")
    shape = reader.OneShape()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    _require(solids.More(), "SINGLE_VALID_SOLID_REQUIRED")
    solid = TopoDS.Solid_s(solids.Current())
    solids.Next()
    _require(not solids.More() and BRepCheck_Analyzer(solid).IsValid(), "SINGLE_VALID_SOLID_REQUIRED")
    native_faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        _require(len(native_faces) < MAX_FACES, "FACE_BUDGET_EXCEEDED")
        native_faces.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    _require(bool(native_faces), "STEP_HAS_NO_FACES")
    solid_faces = []
    solid_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while solid_explorer.More():
        solid_faces.append(TopoDS.Face_s(solid_explorer.Current()))
        solid_explorer.Next()
    _require(len(solid_faces) == len(native_faces) and all(
        sum(face.IsSame(owner_face) for owner_face in solid_faces) == 1 for face in native_faces),
        "NATIVE_FACE_SOLID_OWNERSHIP_MISMATCH")
    # Refuse unknown/very highly subdivided geometry BEFORE tessellating it.
    # A huge-radius circular trim can otherwise force an unbounded mesh even
    # when its STEP serialization is small. This is a safety estimate, not a
    # manufacturing or inference tolerance.
    estimated_segments, preflight_uses, total_curve_poles = 0, 0, 0
    native_edges, edge_curve_proofs, face_edge_ids, surface_proofs = [], [], [], []
    identity_comparisons = 0
    for native_face in native_faces:
        surface = BRepAdaptor_Surface(native_face)
        descriptor = _surface_proof(surface)
        surface_proofs.append(descriptor)
        total_curve_poles += descriptor["pole_cost"]
        _require(total_curve_poles <= MAX_TOTAL_CURVE_POLES, "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
        ids = set()
        edges = TopExp_Explorer(native_face, TopAbs_EDGE)
        while edges.More():
            preflight_uses += 1
            _require(preflight_uses <= MAX_EDGE_USES, "EDGE_INCIDENCE_BUDGET_EXCEEDED")
            edge = TopoDS.Edge_s(edges.Current())
            match = None
            for i, previous in enumerate(native_edges):
                identity_comparisons += 1
                _require(identity_comparisons <= MAX_EDGE_IDENTITY_COMPARISONS,
                         "EDGE_IDENTITY_COMPARISON_BUDGET_EXCEEDED")
                if edge.IsSame(previous):
                    match = i
                    break
            if match is None:
                match = len(native_edges)
                curve = _curve_proof(BRepAdaptor_Curve(edge))
                native_edges.append(edge)
                edge_curve_proofs.append(curve)
                total_curve_poles += len(curve.poles)
                _require(total_curve_poles <= MAX_TOTAL_CURVE_POLES, "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
            else:
                curve = edge_curve_proofs[match]
            _require(not (match in ids and curve.kind == "circle"), "DUPLICATE_NATIVE_CIRCULAR_RAIL_USE")
            ids.add(match)
            if curve.kind == "circle":
                radius, sweep = curve.radius, curve.u1 - curve.u0
                _require(math.isfinite(radius) and radius > GEOM_TOL and math.isfinite(sweep),
                         "NONFINITE_GEOMETRY")
                angular_step = min(0.1, 2. * math.acos(max(-1., min(1., 1. - 0.05 / radius))))
                _require(angular_step > 0., "TESSELLATION_PREFLIGHT_BUDGET_EXCEEDED")
                estimated_segments += math.ceil(sweep / angular_step)
            elif curve.kind == "spline":
                # Positive-weight rational B-splines are variation diminishing;
                # the control-polygon length bounds curve length. This deliberately
                # conservative safety estimate does not fit empirical parts.
                length = float(np.linalg.norm(np.diff(curve.poles, axis=0), axis=1).sum())
                _require(math.isfinite(length), "NONFINITE_NATIVE_CURVE")
                estimated_segments += math.ceil(length / 0.05) + len(curve.knots)
            else:
                estimated_segments += 1
            _require(estimated_segments <= MAX_TRIANGLES, "TESSELLATION_PREFLIGHT_BUDGET_EXCEEDED")
            edges.Next()
        face_edge_ids.append(ids)
    # A single serial, fixed-precision native tessellation. No provider installs.
    mesher = BRepMesh_IncrementalMesh(shape, 0.05, False, 0.1, False)
    mesher.Perform()
    _require(mesher.IsDone(), "NATIVE_TESSELLATION_UNAVAILABLE")
    faces, total_triangles, total_nodes = [], 0, 0
    for index, native_face in enumerate(native_faces):
        surface = BRepAdaptor_Surface(native_face)
        descriptor = surface_proofs[index]
        location = TopLoc_Location()
        triangles = BRep_Tool.Triangulation_s(native_face, location)
        _require(triangles is not None and triangles.NbTriangles() > 0, "NATIVE_FACE_WITHOUT_TRIANGLES")
        total_triangles += triangles.NbTriangles()
        _require(total_triangles <= MAX_TRIANGLES, "TRIANGLE_BUDGET_EXCEEDED")
        total_nodes += triangles.NbNodes()
        _require(total_nodes <= 3 * MAX_TRIANGLES, "VERTEX_BUDGET_EXCEEDED")
        points = np.array([vector(triangles.Node(i).Transformed(location.Transformation()))
                           for i in range(1, triangles.NbNodes() + 1)])
        indices = np.array([triangles.Triangle(i).Get() for i in range(1, triangles.NbTriangles() + 1)],
                           dtype=np.int32) - 1
        if descriptor["kind"] == "plane":
            if surface.GetType() == GeomAbs_Plane:
                plane = surface.Plane()
                origin, normal = vector(plane.Location()), vector(plane.Axis().Direction())
            else:
                origin, normal = descriptor["origin"], descriptor["normal"]
            face = SkinFace(index, "plane", points, indices, origin=origin, normal=normal)
        elif descriptor["kind"] == "cylinder":
            cylinder = surface.Cylinder()
            position = cylinder.Position()
            face = SkinFace(index, "cylinder", points, indices, origin=vector(cylinder.Location()),
                axis=vector(cylinder.Axis().Direction()), x_axis=vector(position.XDirection()),
                y_axis=vector(position.YDirection()), radius=float(cylinder.Radius()),
                u0=float(surface.FirstUParameter()), u1=float(surface.LastUParameter()),
                v0=float(surface.FirstVParameter()), v1=float(surface.LastVParameter()))
            # gp_Ax3 can be indirect (e.g. mirrored CAD). Keep native U and
            # radial axes, but reverse axial coordinate so the chart is proper.
            if np.dot(np.cross(face.x_axis, face.y_axis), face.axis) < 0.:
                face.axis = -face.axis
                face.v0, face.v1 = -face.v1, -face.v0
        else:
            face = SkinFace(index, "extruded_boundary", points, indices,
                origin=descriptor["origin"], axis=descriptor["axis"], v0=descriptor["v0"],
                v1=descriptor["v1"], extrusion_curve=descriptor["extrusion_curve"])
            _require(triangles.HasUVNodes(), "NATIVE_EXTRUSION_UV_OWNERSHIP_UNAVAILABLE")
            for i, point in enumerate(points, start=1):
                uv = triangles.UVNode(i)
                _require(math.isfinite(uv.X()) and math.isfinite(uv.Y())
                         and np.linalg.norm(vector(surface.Value(uv.X(), uv.Y())) - point) <= GEOM_TOL,
                         "NATIVE_EXTRUSION_UV_OWNERSHIP_MISMATCH")
        face.edges = {edge: edge_curve_proofs[edge].straight_endpoints() for edge in face_edge_ids[index]}
        face.edge_curves = {edge: edge_curve_proofs[edge] for edge in face_edge_ids[index]}
        faces.append(face)
    axes = extract_step_bend_axes(path)["bend_axes"]
    return faces, axes


def build_unfold_preview(cad_path: str | Path, expected_hash: str,
                         candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a complete CAD-bound skin development, or no flat geometry at all.

    File access occurs only when this function is explicitly invoked. The
    source is bounded, hashed, then parsed through a private byte-exact snapshot.
    """
    source_hash = None
    try:
        _require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
                 "INVALID_EXPECTED_CAD_HASH")
        _require(isinstance(candidates, list) and 0 < len(candidates) <= MAX_CANDIDATES
                 and all(isinstance(row, dict) for row in candidates), "CANDIDATE_BUDGET_OR_SCHEMA_INVALID")
        path = Path(cad_path)
        _require(path.suffix.lower() in {".stp", ".step"} and path.is_file(), "STEP_SOURCE_REQUIRED")
        with path.open("rb") as source:
            data = source.read(MAX_CAD_BYTES + 1)
        _require(0 < len(data) <= MAX_CAD_BYTES, "CAD_BYTE_BUDGET_EXCEEDED")
        source_hash = hashlib.sha256(data).hexdigest()
        _require(source_hash == expected_hash, "CAD_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="sherman-unfold-") as directory:
            snapshot = Path(directory) / "source.step"
            snapshot.write_bytes(data)
            faces, axes = _read_native(snapshot)
            return _develop_faces(faces, axes, candidates, source_hash)
    except UnfoldUnavailable as exc:
        return _envelope(source_hash, str(exc))
    except (OSError, ImportError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
        payload = _envelope(source_hash, "NATIVE_DEVELOPMENT_UNAVAILABLE")
        payload["diagnostics"]["error_type"] = type(exc).__name__
        return payload
