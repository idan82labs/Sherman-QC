"""Independently authored native coefficients; no corpus or product imports."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend import vendor_surface_normalization_v1 as candidate
from backend import vendor_unfold_preview as incumbent


def _surface(*, scalar=(0., 1., 7., 12.), warp=0., unequal=False,
             transpose=False, rational=False, transverse_trim=False, transform=None):
    pytest.importorskip("OCP")
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    from OCP.gp import gp_Pnt
    grid = np.array([[[s, y, z] for s in scalar] for y, z in [(0., 0.), (2., 1.), (1., 2.)]])
    grid[1, 2, 2] += warp
    weights = np.repeat(np.array([1., .7, 1.])[:, None] if rational else np.ones((3, 1)), 4, axis=1)
    if unequal:
        weights[1, 2] *= .5
    if transform is not None:
        grid = grid @ transform.T + np.array([21., -42., 11.])
    if transpose:
        grid, weights = grid.swapaxes(0, 1), weights.T
    nu, nv = grid.shape[:2]
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    native_weights = TColStd_Array2OfReal(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            poles.SetValue(i + 1, j + 1, gp_Pnt(*grid[i, j]))
            native_weights.SetValue(i + 1, j + 1, float(weights[i, j]))
    def arrays(values, degree):
        knots, mult = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2)
        for i, value in enumerate(values, 1):
            knots.SetValue(i, value)
            mult.SetValue(i, degree + 1)
        return knots, mult
    uk, um = arrays([3., 9.] if transpose else [-2., 5.], 3 if transpose else 2)
    vk, vm = arrays([-2., 5.] if transpose else [3., 9.], 2 if transpose else 3)
    native = Geom_BSplineSurface(poles, native_weights, uk, vk, um, vm,
                               3 if transpose else 2, 2 if transpose else 3)
    t0, t1 = (-1., 4.) if transverse_trim else (-2., 5.)
    face = (BRepBuilderAPI_MakeFace(native, 4., 8., t0, t1, 1e-7).Face() if transpose
            else BRepBuilderAPI_MakeFace(native, t0, t1, 4., 8., 1e-7).Face())
    return BRepAdaptor_Surface(face)


def _independent_cubic(t, values):
    return (1. - t) ** 3 * values[0] + 3 * t * (1. - t) ** 2 * values[1] + 3 * t ** 2 * (1. - t) * values[2] + t ** 3 * values[3]


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("rational", [False, True])
def test_high_degree_extrusion_uses_physical_nonlinear_trim_coordinates(transpose, rational):
    surface = _surface(transpose=transpose, rational=rational)
    with pytest.raises(incumbent.UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        incumbent._surface_proof(surface)
    result = candidate.surface_proof_v1(surface)
    assert result["kind"] == "extruded_boundary"
    assert result["extrusion_parameter_direction"] == ("u" if transpose else "v")
    assert result["axis"] == pytest.approx([1., 0., 0.])
    assert result["v0"] == pytest.approx(_independent_cubic(1. / 6., [0., 1., 7., 12.]))
    assert result["v1"] == pytest.approx(_independent_cubic(5. / 6., [0., 1., 7., 12.]))
    assert result["v0"] != pytest.approx(12. / 6.)
    assert result["certificate_error_bound"] == pytest.approx(0.)
    assert result["pole_cost"] == 12


@pytest.mark.parametrize("mirrored", [False, True])
def test_pose_and_mirror_invariance_with_native_transverse_trims(mirrored):
    a = np.array([1., 2., -3.]); a /= np.linalg.norm(a)
    skew = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
    rotation = np.eye(3) + math.sin(.6) * skew + (1. - math.cos(.6)) * (skew @ skew)
    if mirrored:
        rotation = rotation @ np.diag([-1., 1., 1.])
    surface = _surface(transform=rotation, transverse_trim=True)
    result = candidate.surface_proof_v1(surface)
    assert result["axis"] == pytest.approx(rotation @ np.array([1., 0., 0.]))
    assert result["certificate_error_bound"] < 1e-12
    curve = result["extrusion_curve"]
    assert curve.knots[[0, -1]] == pytest.approx([-1., 4.])
    for t in np.linspace(0., 1., 11):
        native = surface.Value(-1. + 5. * t, 4.)
        point = np.array([native.X(), native.Y(), native.Z()])
        base = (point - result["v0"] * result["axis"] - np.array([21., -42., 11.])) @ rotation
        assert base[0] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("fault", [dict(warp=.001), dict(unequal=True), dict(scalar=(0., 8., 2., 12.))])
def test_nonextrusion_weight_coupling_and_parameter_folds_refuse(fault):
    with pytest.raises(incumbent.UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        candidate.surface_proof_v1(_surface(**fault))


def test_large_control_net_errors_cannot_hide_behind_small_basis_weights():
    with pytest.raises(incumbent.UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        candidate.surface_proof_v1(_surface(warp=.01, rational=True))


def test_native_analytic_plane_is_retained_without_new_certificate():
    pytest.importorskip("OCP")
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    surface = BRepAdaptor_Surface(BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(1., 2., 3.), gp_Dir(0., 0., 1.)), -2., 2., -3., 3.).Face())
    assert candidate.surface_proof_v1(surface) == {"kind": "plane", "pole_cost": 0}


def test_knot_inserted_extrusion_proves_global_chart_without_degree_reduction():
    surface = _surface()
    spline = surface.BSpline()
    spline.InsertVKnot(5., 2, 1e-12)
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    adapted = BRepAdaptor_Surface(BRepBuilderAPI_MakeFace(spline, -2., 5., 4., 8., 1e-7).Face())
    result = candidate.surface_proof_v1(adapted)
    assert result["v0"] == pytest.approx(_independent_cubic(1. / 6., [0., 1., 7., 12.]))
    assert result["v1"] == pytest.approx(_independent_cubic(5. / 6., [0., 1., 7., 12.]))


def test_certificate_rejects_sum_of_individually_small_geometry_errors():
    poles = np.array([[[s, y, z] for s in [0., 4., 8., 12.]]
                      for y, z in [(0., 0.), (2., 1.), (1., 2.)]])
    poles[1, :, 0] += 7e-6
    poles[1, 2, 2] += 7e-6
    assert candidate._direction_proof(poles, np.ones((3, 4)), 3,
        np.array([0., 1.]), np.array([4, 4]), 0., 1.) is None
