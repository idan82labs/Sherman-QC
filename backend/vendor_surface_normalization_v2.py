"""Coefficient proof of a fixed-direction extrusion locus with varying speed.

V1 is preserved. This accepts X(u,v)=C(u)+h(u,v)*axis only when every
transverse control row has the SAME actual trimmed physical endpoints. A
varying full-knot parameterization is not permission to guess trimmed bounds.
No arbitrary NURBS cylinder promotion, file access or native work at import.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np

from backend import vendor_unfold_preview as native
from backend.vendor_surface_normalization_v1 import surface_proof_v1, _scalar_spline


def varying_generator_proof(poles, weights, degree, knots, multiplicities,
                            trim0, trim1) -> dict[str, Any] | None:
    """Prove the same extrusion locus, not a separable native parameterization.

    Positive weights independent of v make the denominator independent of v.
    All per-u-row scalar splines strictly increase, and their evaluations at
    both native trim endpoints agree. Their positive weighted mixtures thus
    cover the same physical interval continuously for every transverse u.
    """
    p, w = np.asarray(poles, dtype=float), np.asarray(weights, dtype=float)
    if (p.ndim != 3 or p.shape[2] != 3 or min(p.shape[:2]) < 2
            or p.shape[0] * p.shape[1] > native.MAX_SURFACE_POLES
            or w.shape != p.shape[:2] or not np.isfinite(p).all()
            or not np.isfinite(w).all() or np.any(w <= 0)
            or w.max() / w.min() > 1e12):
        return None
    if not np.array_equal(w, np.repeat(w[:, :1], w.shape[1], axis=1)):
        return None
    if (type(degree) is not int or not 1 <= degree <= native.MAX_CURVE_DEGREE
            or not math.isfinite(trim0) or not math.isfinite(trim1)):
        return None
    k, m = np.asarray(knots, dtype=float), np.asarray(multiplicities)
    if (k.ndim != 1 or m.shape != k.shape or len(k) < 2 or not np.isfinite(k).all()
            or np.any(np.diff(k) <= 0) or not np.issubdtype(m.dtype, np.integer)
            or np.any(m <= 0) or m[0] != degree + 1 or m[-1] != degree + 1
            or np.any(m[1:-1] > degree) or int(m.sum()) != p.shape[1] + degree + 1
            or not k[0] <= trim0 < trim1 <= k[-1]):
        return None
    base = p[:, 0]
    offsets = p - base[:, None]
    endpoint = offsets[0, -1]
    length = float(np.linalg.norm(endpoint))
    if not math.isfinite(length) or length <= native.GEOM_TOL:
        return None
    axis = endpoint / length
    scalar = offsets @ axis
    # Strict control increments give positive derivative wherever the clamped
    # native chart is regular. No sampled monotonicity or tiny-weight hiding.
    if np.any(np.diff(scalar, axis=1) <= 0):
        return None
    residual = offsets - np.expand_dims(scalar, 2) * axis
    translation_bound = float(np.linalg.norm(residual, axis=2).max())
    base_axial_bound = float(np.ptp(base @ axis))
    starts = np.array([_scalar_spline(row, degree, k, m, trim0) for row in scalar])
    ends = np.array([_scalar_spline(row, degree, k, m, trim1) for row in scalar])
    endpoint_bound = max(float(np.max(np.abs(starts - starts[0]))),
                         float(np.max(np.abs(ends - ends[0]))))
    error = translation_bound + base_axial_bound + endpoint_bound
    if (not math.isfinite(error) or error > native.GEOM_TOL
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or ends[0] - starts[0] <= native.GEOM_TOL):
        return None
    return {"axis": axis, "v0": float(starts[0]), "v1": float(ends[0]),
            "certificate_error_bound": error,
            "tensor_normal_fiber_bound": translation_bound,
            "transverse_base_bound": base_axial_bound,
            "trimmed_endpoint_constancy_bound": endpoint_bound,
            "generator_parameterization": "TRANSVERSE_DEPENDENT_MONOTONE_SPEED",
            "normalization_certificate": "POSITIVE_TENSOR_COMMON_TRIMMED_FIBERS_V2"}


def surface_proof_v2(surface: Any) -> dict[str, Any]:
    try:
        return surface_proof_v1(surface)
    except native.UnfoldUnavailable as failure:
        if str(failure) != "UNPROVEN_NONPLANAR_BSPLINE_SURFACE":
            raise
    from OCP.GeomAbs import GeomAbs_BSplineSurface
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

    native._require(surface.GetType() == GeomAbs_BSplineSurface, "UNSUPPORTED_NATIVE_SURFACE")
    spline = surface.BSpline()
    nu, nv = int(spline.NbUPoles()), int(spline.NbVPoles())
    native._require(nu * nv <= native.MAX_SURFACE_POLES
                    and max(spline.UDegree(), spline.VDegree()) <= native.MAX_CURVE_DEGREE,
                    "NATIVE_SURFACE_COMPLEXITY_BUDGET_EXCEEDED")
    native._require(not spline.IsUPeriodic() and not spline.IsVPeriodic(),
                    "UNPROVEN_NATIVE_SURFACE_PERIODICITY")
    p = np.array([[[spline.Pole(i, j).X(), spline.Pole(i, j).Y(), spline.Pole(i, j).Z()]
                   for j in range(1, nv + 1)] for i in range(1, nu + 1)], dtype=float)
    w = np.array([[spline.Weight(i, j) for j in range(1, nv + 1)]
                  for i in range(1, nu + 1)], dtype=float)
    for transpose in (False, True):
        degree = int(spline.UDegree() if transpose else spline.VDegree())
        count = int(spline.NbUKnots() if transpose else spline.NbVKnots())
        knot = spline.UKnot if transpose else spline.VKnot
        mult = spline.UMultiplicity if transpose else spline.VMultiplicity
        knots = np.array([knot(i) for i in range(1, count + 1)])
        multiplicities = np.array([mult(i) for i in range(1, count + 1)], dtype=np.int32)
        trim0 = float(surface.FirstUParameter() if transpose else surface.FirstVParameter())
        trim1 = float(surface.LastUParameter() if transpose else surface.LastVParameter())
        proof = varying_generator_proof(p.swapaxes(0, 1) if transpose else p,
                                        w.T if transpose else w, degree, knots,
                                        multiplicities, trim0, trim1)
        if proof is None:
            continue
        curve = spline.UIso(float(knots[0])) if transpose else spline.VIso(float(knots[0]))
        c0 = float(surface.FirstVParameter() if transpose else surface.FirstUParameter())
        c1 = float(surface.LastVParameter() if transpose else surface.LastUParameter())
        edge = BRepBuilderAPI_MakeEdge(curve, c0, c1)
        native._require(edge.IsDone(), "NATIVE_EXTRUSION_ISOCURVE_UNAVAILABLE")
        curve_proof = native._curve_proof(BRepAdaptor_Curve(edge.Edge()))
        proof.update(kind="extruded_boundary", origin=curve_proof.poles[0].copy(),
                     extrusion_curve=curve_proof, pole_cost=nu * nv,
                     extrusion_parameter_direction="u" if transpose else "v")
        return proof
    raise native.UnfoldUnavailable("UNPROVEN_NONPLANAR_BSPLINE_SURFACE")
