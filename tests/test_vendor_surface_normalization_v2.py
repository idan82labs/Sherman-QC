"""Pure independently authored polynomial/tensor witnesses; no native jobs."""
import math
import os
import numpy as np
import pytest

from backend.vendor_surface_normalization_v1 import _direction_proof
from backend.vendor_surface_normalization_v2 import varying_generator_proof
from backend.vendor_surface_normalization_v2 import surface_proof_v2


def net(scalars=None, weights=None):
    if scalars is None:
        scalars = [[0., 1., 3., 4.], [0., .5, 2., 4.], [0., 1.5, 3.5, 4.]]
    scalar = np.asarray(scalars)
    base = np.array([[0., 0., 0.], [0., 2., 1.], [0., 1., 3.]])
    poles = base[:, None] + np.expand_dims(scalar, 2) * [1., 0., 0.]
    if weights is None:
        weights = np.repeat(np.array([1., .7, 1.])[:, None], scalar.shape[1], axis=1)
    return poles, np.asarray(weights)


def prove(p, w, trim=(0., 1.)):
    degree = p.shape[1] - 1
    return varying_generator_proof(p, w, degree, np.array([0., 1.]),
                                   np.array([degree + 1, degree + 1]), *trim)


def test_varying_speed_same_extrusion_locus_does_not_need_shared_scalar_profile():
    p, w = net()
    assert _direction_proof(p, w, 3, np.array([0., 1.]), np.array([4, 4]), 0., 1.) is None
    proof = prove(p, w)
    assert proof["v0"] == 0. and proof["v1"] == 4.
    assert proof["axis"] == pytest.approx([1., 0., 0.])
    assert proof["trimmed_endpoint_constancy_bound"] == 0.


def test_actual_interior_trim_endpoints_must_agree_not_just_full_knots():
    p, w = net()
    assert prove(p, w, (.2, .8)) is None


def test_interior_trim_with_independently_authored_vanishing_perturbation():
    # delta(v)=v*(1-v)*(v-.2)*(v-.8). Independent power-to-Bernstein
    # conversion gives distinct speeds with identical ACTUAL trim endpoints.
    power = np.polynomial.polynomial.polymul(
        np.polynomial.polynomial.polymul([0., 1.], [1., -1.]),
        np.polynomial.polynomial.polymul([-.2, 1.], [-.8, 1.]))
    delta = np.array([sum(power[k] * math.comb(i, k) / math.comb(4, k)
                         for k in range(i + 1)) for i in range(5)])
    base = np.arange(5, dtype=float)
    p, w = net([base, base + .2 * delta, base - .3 * delta])
    proof = prove(p, w, (.2, .8))
    assert proof is not None
    assert proof["v0"] == pytest.approx(.8)
    assert proof["v1"] == pytest.approx(3.2)


@pytest.mark.parametrize("fault", ["warp", "base", "end", "weight", "fold", "stall"])
def test_unproved_normal_fibers_and_parameterization_refuse(fault):
    p, w = net(); p = p.copy(); w = w.copy()
    if fault == "warp": p[1, 1, 1] += .001
    if fault == "base": p[1, :, 0] += .001
    if fault == "end": p[1, -1, 0] -= .001
    if fault == "weight": w[1, 1] *= .5
    if fault == "fold": p[1, 2, 0] = .1
    if fault == "stall": p[1, 2, 0] = p[1, 1, 0]
    assert prove(p, w) is None


@pytest.mark.parametrize("trim", [(-1e-12, 1.), (0., 1.+1e-12), (.8, .2)])
def test_no_silent_trim_clamping(trim):
    assert prove(*net(), trim=trim) is None


@pytest.mark.parametrize("reflected", [False, True])
def test_pose_invariance_and_input_preservation(reflected):
    p, w = net(); axis = np.array([1., 2., -3.]); axis /= np.linalg.norm(axis)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    r = np.eye(3) + math.sin(.3)*cross + (1-math.cos(.3))*cross@cross
    if reflected: r = r @ np.diag([-1., 1., 1.])
    p = p @ r.T + [13., -4., 9.]; before = p.copy()
    proof = prove(p, w)
    assert proof is not None and proof["axis"] == pytest.approx(r @ [1., 0., 0.])
    assert np.array_equal(p, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0., -1.])
def test_bad_weights_refuse(bad):
    p, w = net(); w[1, 1] = bad
    assert prove(p, w) is None


def test_sum_of_coefficient_bounds_not_independent_thresholds():
    p, w = net()
    p[1, :, 0] += 7e-6
    p[1, 2, 2] += 7e-6
    assert prove(p, w) is None


def test_degree_and_knot_budget_protected():
    p, w = net()
    assert varying_generator_proof(p, w, 999, [0., 1.], [4, 4], 0., 1.) is None
    assert varying_generator_proof(p, w, 3, [0., 1.], [4, 3], 0., 1.) is None
    assert varying_generator_proof(p, w, 3, [0., 1.], [4., 4.], 0., 1.) is None


def _authored_native_surface(trim=(3., 9.)):
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    p, w = net()
    poles = TColgp_Array2OfPnt(1, 3, 1, 4)
    weights = TColStd_Array2OfReal(1, 3, 1, 4)
    for i in range(3):
        for j in range(4):
            poles.SetValue(i+1, j+1, gp_Pnt(*p[i, j]))
            weights.SetValue(i+1, j+1, float(w[i, j]))
    def basis(bounds, degree):
        knots = TColStd_Array1OfReal(1, 2); mult = TColStd_Array1OfInteger(1, 2)
        for i, value in enumerate(bounds, 1):
            knots.SetValue(i, value); mult.SetValue(i, degree+1)
        return knots, mult
    uk, um = basis((-2., 5.), 2); vk, vm = basis((3., 9.), 3)
    spline = Geom_BSplineSurface(poles, weights, uk, vk, um, vm, 2, 3)
    builder = BRepBuilderAPI_MakeFace(spline, -2., 5., *trim, 1e-7)
    assert builder.IsDone()
    return BRepAdaptor_Surface(builder.Face())


@pytest.mark.skipif(os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE") != "1",
                    reason="root-coordinated bounded authored-native phase required")
def test_authored_native_varying_speed_wall_proves_full_physical_interval():
    from backend.vendor_surface_normalization_v1 import surface_proof_v1
    from backend.vendor_unfold_preview import UnfoldUnavailable
    surface = _authored_native_surface()
    with pytest.raises(UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        surface_proof_v1(surface)
    proof = surface_proof_v2(surface)
    assert proof["kind"] == "extruded_boundary"
    assert proof["v0"] == pytest.approx(0.) and proof["v1"] == pytest.approx(4.)
    assert proof["normalization_certificate"] == "POSITIVE_TENSOR_COMMON_TRIMMED_FIBERS_V2"
    for u in np.linspace(-2., 5., 7):
        for v in np.linspace(3., 9., 9):
            point = surface.Value(float(u), float(v))
            axial = (np.array([point.X(), point.Y(), point.Z()]) - proof["origin"]) @ proof["axis"]
            assert -1e-10 <= axial <= 4.+1e-10


@pytest.mark.skipif(os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE") != "1",
                    reason="root-coordinated bounded authored-native phase required")
def test_authored_native_interior_trim_with_variable_physical_endpoints_refuses():
    from backend.vendor_unfold_preview import UnfoldUnavailable
    with pytest.raises(UnfoldUnavailable, match="UNPROVEN_NONPLANAR_BSPLINE_SURFACE"):
        surface_proof_v2(_authored_native_surface((4., 8.)))
