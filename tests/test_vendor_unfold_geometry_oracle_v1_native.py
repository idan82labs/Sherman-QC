"""Explicit native synthetic primitives; no STEP files or physical inputs."""
import math

import numpy as np
import pytest

from backend.vendor_unfold_geometry_oracle_v1 import (
    DevelopedFace, capture_native_face_evidence, validate_development,
)


def _polygon(points):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt
    builder = BRepBuilderAPI_MakePolygon()
    for point in points:
        builder.Add(gp_Pnt(*point))
    builder.Close()
    return builder.Wire()


def _mesh(face):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    mesher = BRepMesh_IncrementalMesh(face, .05, False, .1, False)
    mesher.Perform()
    assert mesher.IsDone()


def _flat(source):
    triangles = source.triangles.copy()
    chart = source.metric_chart
    q = chart[triangles]
    signs = (q[:, 1, 0]-q[:, 0, 0])*(q[:, 2, 1]-q[:, 0, 1])-(q[:, 1, 1]-q[:, 0, 1])*(q[:, 2, 0]-q[:, 0, 0])
    triangles[signs < 0] = triangles[signs < 0][:, [0, 2, 1]]
    return DevelopedFace(source.source_index, np.column_stack((chart, np.zeros(len(chart)))), triangles)


@pytest.mark.parametrize("reverse", [False, True])
def test_native_planar_rectangular_hole_exact_area_and_orientation(reverse):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    outer = _polygon([(0., 0., 0.), (10., 0., 0.), (10., 6., 0.), (0., 6., 0.)])
    # Clockwise inner wire is a hole according to native face orientation.
    inner = _polygon([(3., 2., 0.), (3., 4., 0.), (7., 4., 0.), (7., 2., 0.)])
    builder = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), outer)
    builder.Add(inner)
    face = builder.Face()
    if reverse:
        face = TopoDS.Face_s(face.Reversed())
    assert BRepCheck_Analyzer(face).IsValid()
    _mesh(face)
    source, receipt = capture_native_face_evidence(face, 0, polygon_area_error_bound_mm2=0.)
    assert source.continuous_area_mm2 == pytest.approx(52., abs=1e-10)
    assert source.boundary_cycles == 2
    assert receipt["triangulation_used_for_area"] is False
    assert validate_development([source], [_flat(source)])["total_flat_area_mm2"] == pytest.approx(52.)


def test_native_cylinder_surface_area_not_chord_area():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Ax3, gp_Cylinder
    face = BRepBuilderAPI_MakeFace(gp_Cylinder(gp_Ax3(), 4.), .2, .2+math.pi/2, -3., 9.).Face()
    _mesh(face)
    source, _ = capture_native_face_evidence(face, 0, polygon_area_error_bound_mm2=0.)
    assert source.continuous_area_mm2 == pytest.approx(4*12*math.pi/2, abs=1e-10)
    report = validate_development([source], [_flat(source)])
    assert report["total_flat_area_mm2"] == pytest.approx(source.continuous_area_mm2, abs=1e-8)


def test_native_circular_hole_discrete_mode_keeps_quadrature_separate():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pln, gp_Pnt
    outer = _polygon([(0., 0., 0.), (10., 0., 0.), (10., 6., 0.), (0., 6., 0.)])
    circle = gp_Circ(gp_Ax2(gp_Pnt(5., 3., 0.), gp_Dir(0., 0., 1.)), 1.)
    edge = BRepBuilderAPI_MakeEdge(circle).Edge()
    inner = BRepBuilderAPI_MakeWire(edge).Wire()
    builder = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), outer)
    builder.Add(TopoDS.Wire_s(inner.Reversed()))
    face = builder.Face()
    _mesh(face)
    source, receipt = capture_native_face_evidence(face, 0)
    assert source.continuous_area_mm2 is None and source.area_error_bound_mm2 is None
    assert receipt["native_surface_area_mm2"] == pytest.approx(60.-math.pi, abs=1e-8)
    report = validate_development([source], [_flat(source)])
    assert report["continuous_area_check"] == "NOT_ESTABLISHED"
    assert report["faces"][0]["boundary_cycles"] == 2
    # Mesh hole polygon is inscribed: area remains visibly different from the
    # continuous circular hole. No observed difference becomes an allowance.
    assert abs(report["total_flat_area_mm2"]-receipt["native_surface_area_mm2"]) > .001
    assert receipt["polygon_area_error_bound_mm2"] is None
