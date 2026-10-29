"""Coefficient-certified normalization of bounded native spline wall charts.

This module does not read files, approximate surfaces from samples, or convert
general NURBS to cylinders. It extends the incumbent's affine wall proof to
degree-elevated and nonuniformly parameterized straight extrusions. The native
surface is retained; the returned chart is an import-local geometric descriptor.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend import vendor_unfold_preview as incumbent


def _scalar_spline(poles: np.ndarray, degree: int, knots: np.ndarray,
                   multiplicities: np.ndarray, parameter: float) -> float:
    """Evaluate one nonrational clamped scalar B-spline by de Boor recursion."""
    expanded = np.repeat(knots, multiplicities)
    n = len(poles) - 1
    incumbent._require(len(expanded) == n + degree + 2, "INVALID_EXTRUSION_KNOT_TOPOLOGY")
    incumbent._require(knots[0] <= parameter <= knots[-1], "INVALID_NATIVE_EXTRUSION_INTERVAL")
    if parameter == knots[-1]:
        return float(poles[-1])
    span = int(np.searchsorted(expanded, parameter, side="right") - 1)
    incumbent._require(degree <= span <= n, "INVALID_EXTRUSION_KNOT_TOPOLOGY")
    values = poles[span - degree:span + 1].copy()
    for level in range(1, degree + 1):
        for j in range(degree, level - 1, -1):
            low = expanded[span - degree + j]
            high = expanded[span + 1 + j - level]
            incumbent._require(high > low, "INVALID_EXTRUSION_KNOT_TOPOLOGY")
            fraction = (parameter - low) / (high - low)
            values[j] = (1. - fraction) * values[j - 1] + fraction * values[j]
    return float(values[degree])


def _direction_proof(poles: np.ndarray, weights: np.ndarray, degree: int,
                     knots: np.ndarray, mult: np.ndarray,
                     trim0: float, trim1: float) -> dict[str, Any] | None:
    """Prove C(t)+g(s)*axis using the positive tensor-basis convex-hull bound.

    Pole array dimensions are transverse, extrusion, xyz. Identical weights
    across extrusion rows make the rational denominator independent of s, so
    the exact tensor quotient separates. No relative weight threshold is used.
    """
    if not np.array_equal(weights, np.repeat(weights[:, :1], weights.shape[1], axis=1)):
        return None
    base = poles[:, 0]
    offsets = poles - base[:, None]
    endpoint = offsets[0, -1]
    length = float(np.linalg.norm(endpoint))
    if length <= incumbent.GEOM_TOL:
        return None
    axis = endpoint / length
    scalar = offsets[0] @ axis
    # Monotone scalar control coordinates prove a nonfolding traversal in s.
    # Clamped endpoint knots make the complete chart span exactly [0,length].
    if np.any(np.diff(scalar) < 0.) or scalar[0] != 0.:
        return None
    ideal = base[:, None] + scalar[None, :, None] * axis
    translation_bound = float(np.linalg.norm(poles - ideal, axis=2).max())
    base_axial = base @ axis
    transverse_bound = float(np.ptp(base_axial))
    error_bound = translation_bound + transverse_bound
    if not math.isfinite(error_bound) or error_bound > incumbent.GEOM_TOL:
        return None
    v0 = _scalar_spline(scalar, degree, knots, mult, trim0)
    v1 = _scalar_spline(scalar, degree, knots, mult, trim1)
    if not math.isfinite(v0) or not math.isfinite(v1) or v1 - v0 <= incumbent.GEOM_TOL:
        return None
    return {"axis": axis, "v0": v0, "v1": v1,
            "certificate_error_bound": error_bound,
            "tensor_translation_bound": translation_bound,
            "transverse_base_bound": transverse_bound}


def surface_proof_v1(surface: Any) -> dict[str, Any]:
    """Incumbent charts plus globally certified higher-degree spline extrusion.

    Safe integration is at the native surface-proof seam, before triangulation.
    Existing cylinder recognition and bend extraction stay unchanged. A spline
    whose locus is cylindrical receives an extrusion wall chart here, not a bend
    chart; converting it to a bend requires an additional carrier/trim proof.
    """
    try:
        return incumbent._surface_proof(surface)
    except incumbent.UnfoldUnavailable as failure:
        if str(failure) != "UNPROVEN_NONPLANAR_BSPLINE_SURFACE":
            raise
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.GeomAbs import GeomAbs_BSplineSurface

    incumbent._require(surface.GetType() == GeomAbs_BSplineSurface,
                       "UNSUPPORTED_NATIVE_SURFACE")
    spline = surface.BSpline()
    nu, nv = int(spline.NbUPoles()), int(spline.NbVPoles())
    incumbent._require(nu * nv <= incumbent.MAX_SURFACE_POLES
                       and max(spline.UDegree(), spline.VDegree()) <= incumbent.MAX_CURVE_DEGREE,
                       "NATIVE_SURFACE_COMPLEXITY_BUDGET_EXCEEDED")
    incumbent._require(not spline.IsUPeriodic() and not spline.IsVPeriodic(),
                       "UNPROVEN_NATIVE_SURFACE_PERIODICITY")
    poles = np.array([[[spline.Pole(i, j).X(), spline.Pole(i, j).Y(), spline.Pole(i, j).Z()]
                       for j in range(1, nv + 1)] for i in range(1, nu + 1)], dtype=float)
    weights = np.array([[spline.Weight(i, j) for j in range(1, nv + 1)]
                        for i in range(1, nu + 1)], dtype=float)
    incumbent._require(np.isfinite(poles).all() and np.isfinite(weights).all()
                       and weights.min() > 0. and weights.max() / weights.min() <= 1e12,
                       "INVALID_NATIVE_SURFACE_COEFFICIENTS")
    for transpose in (False, True):
        degree = int(spline.UDegree() if transpose else spline.VDegree())
        count = int(spline.NbUKnots() if transpose else spline.NbVKnots())
        knot = spline.UKnot if transpose else spline.VKnot
        multiplicity = spline.UMultiplicity if transpose else spline.VMultiplicity
        knots = np.array([knot(i) for i in range(1, count + 1)], dtype=float)
        mult = np.array([multiplicity(i) for i in range(1, count + 1)], dtype=np.int32)
        if (not np.isfinite(knots).all() or len(knots) < 2 or np.any(np.diff(knots) <= 0.)
                or mult[0] != degree + 1 or mult[-1] != degree + 1):
            continue
        trim0 = float(surface.FirstUParameter() if transpose else surface.FirstVParameter())
        trim1 = float(surface.LastUParameter() if transpose else surface.LastVParameter())
        incumbent._require(math.isfinite(trim0) and math.isfinite(trim1)
                           and knots[0] <= trim0 < trim1 <= knots[-1],
                           "INVALID_NATIVE_EXTRUSION_INTERVAL")
        proof = _direction_proof(poles.swapaxes(0, 1) if transpose else poles,
                                 weights.T if transpose else weights,
                                 degree, knots, mult, trim0, trim1)
        if proof is None:
            continue
        # Native iso extraction and native trim retain the transverse curve's
        # own weights/knots instead of pretending control endpoints are trims.
        curve = spline.UIso(float(knots[0])) if transpose else spline.VIso(float(knots[0]))
        c0 = float(surface.FirstVParameter() if transpose else surface.FirstUParameter())
        c1 = float(surface.LastVParameter() if transpose else surface.LastUParameter())
        edge = BRepBuilderAPI_MakeEdge(curve, c0, c1)
        incumbent._require(edge.IsDone(), "NATIVE_EXTRUSION_ISOCURVE_UNAVAILABLE")
        curve_proof = incumbent._curve_proof(BRepAdaptor_Curve(edge.Edge()))
        # The descriptor origin is the trimmed native curve's first control
        # point. v is physical displacement from the full extrusion start.
        proof.update({"kind": "extruded_boundary", "origin": curve_proof.poles[0].copy(),
                      "extrusion_curve": curve_proof, "pole_cost": nu * nv,
                      "normalization_certificate": "POSITIVE_TENSOR_BASIS_EXTRUSION_V1",
                      "extrusion_parameter_direction": "u" if transpose else "v"})
        return proof
    raise incumbent.UnfoldUnavailable("UNPROVEN_NONPLANAR_BSPLINE_SURFACE")
