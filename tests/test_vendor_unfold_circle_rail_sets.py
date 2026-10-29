"""V3 closed-hole rail sets: source-only, newly authored synthetic geometry."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

from backend import vendor_unfold_preview as engine


TAU = 2 * math.pi
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _fixture(radius=1.7):
    wall = engine.SkinFace(2, "cylinder", np.zeros((4, 3)), TRIANGLES.copy(),
        origin=np.zeros(3), axis=np.array([0., 0., 1.]),
        x_axis=np.array([1., 0., 0.]), y_axis=np.array([0., 1., 0.]),
        radius=radius, u0=0., u1=TAU, v0=0., v1=1.)
    planes = [engine.SkinFace(i, "plane", np.zeros((4, 3)), TRIANGLES.copy(),
        origin=np.array([0., 0., float(i)]), normal=np.array([0., 0., 1.])) for i in (0, 1)]
    return wall, planes[0], planes[1]


def _arc(plane, start, stop, *, radius=1.7, phase=0., reverse_axis=False):
    sign = -1. if reverse_axis else 1.
    x = np.array([math.cos(phase), math.sin(phase), 0.])
    y = sign * np.array([-math.sin(phase), math.cos(phase), 0.])
    return engine.NativeCurve("circle", np.zeros((2, 3)), center=plane.origin.copy(),
        axis=np.array([0., 0., sign]), x_axis=x, y_axis=y,
        radius=radius, u0=start, u1=stop)


def _prove(wall, left, right, a, b):
    return engine._closed_circular_rail_sets_proven(wall, left, right, a, b, right.origin - left.origin)


@pytest.mark.parametrize("count", [2, 3, 4, 7, 16, 32])
def test_full_circle_and_complete_split_set_in_both_directions(count):
    wall, left, right = _fixture()
    a = [_arc(left, 0., TAU)]
    b = [_arc(right, TAU * i / count, TAU * (i + 1) / count) for i in range(count)]
    assert _prove(wall, left, right, a, b)
    assert _prove(wall, right, left, b, a)
    assert not engine._translated_curves(a[0], b[0], right.origin - left.origin)


@pytest.mark.parametrize("reversed_axis", [False, True])
def test_unordered_wrapped_and_independently_phased_native_charts(reversed_axis):
    wall, left, right = _fixture()
    a = [_arc(left, -7.3, -7.3 + TAU, phase=1.2)]
    b = [_arc(right, -2.3 + TAU * i / 7, -2.3 + TAU * (i + 1) / 7,
              phase=.83, reverse_axis=reversed_axis) for i in range(7)]
    b = b[::2] + b[1::2]
    assert _prove(wall, left, right, a, b)


def test_random_rigid_poses_preserve_complete_carrier_proof():
    rng = np.random.default_rng(819)
    for _ in range(24):
        radius = float(10 ** rng.uniform(-1., 4.))
        wall, left, right = _fixture(radius)
        a = [_arc(left, -.6, TAU - .6, radius=radius)]
        b = [_arc(right, 1.2 + i * TAU / 5, 1.2 + (i + 1) * TAU / 5,
                  radius=radius, phase=.7, reverse_axis=True) for i in range(5)]
        rotation = engine._rotation(rng.normal(size=3), float(rng.uniform(-math.pi, math.pi)))
        translation = rng.uniform(-1000., 1000., size=3)
        for face in (wall, left, right):
            face.origin = rotation @ face.origin + translation
            for name in ("axis", "x_axis", "y_axis", "normal"):
                value = getattr(face, name)
                if value is not None:
                    setattr(face, name, rotation @ value)
        for curve in a + b:
            curve.center = rotation @ curve.center + translation
            for name in ("axis", "x_axis", "y_axis"):
                setattr(curve, name, rotation @ getattr(curve, name))
        assert _prove(wall, left, right, a, b)


@pytest.mark.parametrize("failure", ["balanced_gap_overlap", "missing_arc", "duplicate_half", "full_plus_half",
    "zero_sweep", "multiple_turn", "wrong_radius", "tilted_axis", "nonunit_frame", "wrong_handedness",
    "mixed_kind", "partial_wall", "too_many_rails", "huge_u", "huge_center", "nonfinite"])
def test_wrong_or_incomplete_closed_carrier_sets_refuse(failure):
    wall, left, right = _fixture()
    a = [_arc(left, 0., TAU)]
    b = [_arc(right, 0., math.pi), _arc(right, math.pi, TAU)]
    delta = .75 * engine.GEOM_TOL / wall.radius
    if failure == "balanced_gap_overlap":
        b[1].u0 += delta
        b[1].u1 += delta  # sum remains 2π, but gap+overlap exceed the shared tolerance.
    elif failure == "missing_arc":
        b[1].u1 -= 1e-6 / wall.radius
    elif failure == "duplicate_half":
        b[1] = copy.deepcopy(b[0])
    elif failure == "full_plus_half":
        b[0].u1 = TAU
    elif failure == "zero_sweep":
        b[0].u1 = b[0].u0
    elif failure == "multiple_turn":
        b[0].u1 = 2 * TAU
    elif failure == "wrong_radius":
        b[0].radius += 2 * engine.GEOM_TOL
    elif failure == "tilted_axis":
        rotation = engine._rotation(np.array([1., 0., 0.]), .001)
        b[0].axis, b[0].x_axis, b[0].y_axis = [rotation @ v for v in (b[0].axis, b[0].x_axis, b[0].y_axis)]
    elif failure == "nonunit_frame":
        b[0].x_axis *= 1.001
    elif failure == "wrong_handedness":
        b[0].axis *= -1.
    elif failure == "mixed_kind":
        b[0] = engine.NativeCurve("line", np.zeros((2, 3)))
    elif failure == "partial_wall":
        wall.u1 = math.pi
    elif failure == "too_many_rails":
        b = [_arc(right, i * TAU / 33, (i + 1) * TAU / 33) for i in range(33)]
    elif failure == "huge_u":
        b[0].u0, b[0].u1 = 1e12, 1e12 + math.pi
    elif failure == "huge_center":
        for face in (wall, left, right):
            face.origin += [1e12, 0., 0.]
        for curve in a + b:
            curve.center += [1e12, 0., 0.]
    else:
        b[0].center[0] = math.nan
    assert not _prove(wall, left, right, a, b)


def test_opposite_carrier_errors_share_one_physical_budget():
    wall, left, right = _fixture()
    a, b = [_arc(left, 0., TAU)], [_arc(right, 0., math.pi), _arc(right, math.pi, TAU)]
    a[0].center[0] += .75 * engine.GEOM_TOL
    for curve in b:
        curve.center[0] -= .75 * engine.GEOM_TOL
    assert not _prove(wall, left, right, a, b)


def _topological_hole_case():
    source = runpy.run_path(str(Path(__file__).with_name("test_vendor_unfold_preview.py")))
    faces, axes, candidates = source["_case"]()
    angles = np.linspace(0., TAU, 33)
    points = np.array([[2. + v, -5. + 1.7 * math.cos(u), 10. + 1.7 * math.sin(u)]
                       for u in angles for v in (0., 1.)])
    triangles = np.array([[2 * i, 2 * i + 2, 2 * i + 3] for i in range(32)]
                        + [[2 * i, 2 * i + 3, 2 * i + 1] for i in range(32)], dtype=np.int32)
    wall = engine.SkinFace(6, "cylinder", points, triangles, edges={80: None, 81: None, 82: None},
        origin=np.array([2., -5., 10.]), axis=np.array([1., 0., 0.]),
        x_axis=np.array([0., 1., 0.]), y_axis=np.array([0., 0., 1.]),
        radius=1.7, u0=0., u1=TAU, v0=0., v1=1.)
    for face_index, edge, start, stop in [(0, 80, 0., TAU), (3, 81, 0., math.pi), (3, 82, math.pi, TAU)]:
        face = faces[face_index]
        face.edges[edge] = None
        curve = engine.NativeCurve("circle", np.zeros((2, 3)),
            center=np.array([2. if face_index == 0 else 3., -5., 10.]),
            axis=wall.axis.copy(), x_axis=wall.x_axis.copy(), y_axis=wall.y_axis.copy(),
            radius=wall.radius, u0=start, u1=stop)
        face.edge_curves[edge] = wall.edge_curves[edge] = curve
    faces.append(wall)
    return faces, axes, candidates


def test_native_edge_identity_graph_full_vs_two_halves_uses_narrow_fallback():
    faces, axes, candidates = _topological_hole_case()
    payload = engine._develop_faces(faces, axes, candidates, "a" * 64)
    assert payload["status"] == "AVAILABLE"
    assert payload["diagnostics"]["mechanism_version"] == 3
    assert payload["diagnostics"]["native_surface_coverage_checked"]
    assert payload["diagnostics"]["excluded_caps_and_hole_walls_checked"]
    assert len(payload["triangles"]) == sum(face.triangles.size for face in faces[:3])
    assert np.max(np.abs(np.array(payload["vertices"]).reshape(-1, 3)[:, 2])) == 0.
    assert payload["bend_lines"][0]["feature_id"] == candidates[0]["feature_id"]


@pytest.mark.parametrize("failure", ["gap", "wrong_span", "outside_slab", "mixed_rail", "partial_wall"])
def test_integrated_boundary_proof_still_refuses_invalid_hole_walls(failure):
    faces, axes, candidates = _topological_hole_case()
    wall = faces[-1]
    if failure == "gap":
        faces[3].edge_curves[82].u0 += .001
    elif failure == "wrong_span":
        wall.v1 = 2.
    elif failure == "outside_slab":
        wall.vertices[:, 0] += .1
        wall.v1 = 1.1
    elif failure == "mixed_rail":
        faces[3].edge_curves[82] = engine.NativeCurve("line", np.zeros((2, 3)))
    else:
        wall.u1 = math.pi
    with pytest.raises(engine.UnfoldUnavailable, match="UNKNOWN_HOLE_WALL_BOUNDARY|NATIVE_CYLINDER_TRIM_MISMATCH"):
        engine._develop_faces(faces, axes, candidates, "a" * 64)


def test_actual_ocp_native_circle_trim_coefficients_have_same_closed_set_proof():
    pytest.importorskip("OCP")
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
    wall, left, right = _fixture()
    def native(plane, start, stop):
        circle = gp_Circ(gp_Ax2(gp_Pnt(*plane.origin), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), wall.radius)
        return engine._curve_proof(BRepAdaptor_Curve(BRepBuilderAPI_MakeEdge(circle, start, stop).Edge()))
    a = [native(left, 0., TAU)]
    b = [native(right, 0., math.pi), native(right, math.pi, TAU)]
    assert _prove(wall, left, right, a, b)
