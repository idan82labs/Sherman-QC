"""Synthetic sheet solid written/read as STEP; no physical corpus inputs."""
import math

import numpy as np
import pytest

from backend.vendor_unfold_native_v5 import read_native_v5
from backend.vendor_unfold_oracle_bridge_v1 import NativeOracleCapture, validate_captured_development
from backend.vendor_unfold_preview_v5 import develop_faces_v5


def _synthetic_sheet():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Pnt, gp_Vec
    def point(x, y):
        return gp_Pnt(float(x), float(y), 0.)
    wire = BRepBuilderAPI_MakeWire()
    def line(a, b):
        wire.Add(BRepBuilderAPI_MakeEdge(point(*a), point(*b)).Edge())
    def arc(radius, reverse=False):
        start, end = point(radius, 0.), point(0., radius)
        if reverse:
            start, end = end, start
        curve = GC_MakeArcOfCircle(start, point(radius/math.sqrt(2), radius/math.sqrt(2)), end).Value()
        wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
    line((4., -30.), (4., 0.)); arc(4.)
    line((0., 4.), (-10., 4.)); line((-10., 4.), (-10., 5.))
    line((-10., 5.), (0., 5.)); arc(5., True)
    line((5., 0.), (5., -30.)); line((5., -30.), (4., -30.))
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0., 0., 12.)).Shape()


@pytest.mark.parametrize("pose", ["identity", "posed", "mirrored_posed"])
def test_actual_v5_native_observer_and_development_complete_oracle(tmp_path, pose):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    shape = _synthetic_sheet()
    if pose != "identity":
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
        if pose == "mirrored_posed":
            reflection = gp_Trsf()
            reflection.SetMirror(gp_Ax2(gp_Pnt(0., 0., 0.), gp_Dir(1., 0., 0.)))
            shape = BRepBuilderAPI_Transform(shape, reflection, True).Shape()
        transform = gp_Trsf()
        transform.SetRotation(gp_Ax1(gp_Pnt(0., 0., 0.), gp_Dir(2., -3., 5.)), .71)
        transform.SetTranslationPart(gp_Vec(73., -21., 44.))
        shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    assert BRepCheck_Analyzer(shape).IsValid()
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    path = tmp_path/"synthetic.step"
    assert writer.Write(str(path)) == IFSelect_RetDone
    capture = NativeOracleCapture()
    faces, axes = read_native_v5(path, on_face=capture.on_face)
    assert len(axes) == 1
    candidates = [{"feature_id": "77d1721b-bc8f-4a99-b2f1-39d7fd339e9b",
                   "line_start": axes[0]["canonical_line_start"],
                   "line_end": axes[0]["canonical_line_end"]}]
    payload = develop_faces_v5(faces, axes, candidates, "a"*64)
    report = validate_captured_development(payload, capture, faces, axes, candidates)
    assert report["face_count"] == 3 and report["seam_count"] == 2 and report["bend_count"] == 1
    assert report["continuous_area_check"] == "NOT_ESTABLISHED"
    assert report["total_flat_area_mm2"] == pytest_approx_independent_area(payload, faces)
    assert report["max_bend_line_error_mm"] < 1e-8


def pytest_approx_independent_area(payload, faces):
    selected = payload["diagnostics"]["selected_native_face_indices"]
    radius = next(faces[i].radius for i in selected if faces[i].kind == "cylinder")
    return pytest.approx(12*(40+radius*math.pi/2), abs=1e-8)
