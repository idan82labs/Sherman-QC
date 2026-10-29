"""Bounded folded display V2 with indexed native edge identity.

V1 remains unchanged. TopTools identity includes native shape/location and
ignores orientation, matching IsSame. Edge incidence/geometry caps are retained.

The caller must supply a hash-bound private STEP snapshot inside a supervised
native process. This reader establishes display provenance only. It does not
classify a spline as a bend, certify wall ownership, or flatten an unknown face.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np

from backend.vendor_unfold_preview import (
    MAX_FACES, MAX_TRIANGLES, MAX_TOTAL_CURVE_POLES, MAX_EDGE_USES,
    GEOM_TOL, _require, _curve_proof,
)
from backend.vendor_unfold_contract_v2 import MAX_COORDINATE


@dataclass(frozen=True)
class DisplayFace:
    index: int
    vertices: np.ndarray
    triangles: np.ndarray
    native_surface_type: str
    native_orientation: str


def validate_control_net(poles, weights) -> int:
    """Size/finite-value preflight, without making a geometric carrier claim."""
    p = np.asarray(poles, dtype=float)
    w = np.asarray(weights, dtype=float)
    _require(p.ndim == 3 and p.shape[2] == 3 and min(p.shape[:2]) >= 2
             and w.shape == p.shape[:2], "DISPLAY_INVALID_CONTROL_NET")
    count = p.shape[0] * p.shape[1]
    _require(count <= MAX_TOTAL_CURVE_POLES, "DISPLAY_CONTROL_NET_BUDGET")
    _require(np.isfinite(p).all() and np.max(np.abs(p)) <= MAX_COORDINATE
             and np.isfinite(w).all() and np.min(w) > 0
             and np.max(w) / np.min(w) <= 1e12,
             "DISPLAY_INVALID_CONTROL_COEFFICIENTS")
    return count


def read_folded_native(path: Path) -> list[DisplayFace]:
    """Tessellate complete native material under the caller's time/RSS cap."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape

    def xyz(p):
        return [p.X(), p.Y(), p.Z()]

    reader = STEPControl_Reader()
    _require(reader.ReadFile(str(path)) == IFSelect_RetDone and reader.TransferRoots() > 0,
             "DISPLAY_STEP_READ_UNAVAILABLE")
    shape = reader.OneShape()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    _require(solids.More(), "DISPLAY_SINGLE_VALID_SOLID_REQUIRED")
    solid = TopoDS.Solid_s(solids.Current())
    solids.Next()
    _require(not solids.More() and BRepCheck_Analyzer(solid).IsValid(),
             "DISPLAY_SINGLE_VALID_SOLID_REQUIRED")
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        _require(len(faces) < MAX_FACES, "DISPLAY_FACE_BUDGET")
        faces.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    _require(bool(faces), "DISPLAY_NO_NATIVE_FACES")
    owned = []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while explorer.More():
        _require(len(owned) < MAX_FACES, "DISPLAY_FACE_BUDGET")
        owned.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    _require(len(owned) == len(faces) and all(
        sum(face.IsSame(member) for member in owned) == 1 for face in faces),
        "DISPLAY_FACE_SOLID_OWNERSHIP_MISMATCH")

    edge_map = TopTools_IndexedMapOfShape()
    curve_cost, edge_uses, estimated_segments = 0, 0, 0
    for face in faces:
        _require(face.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED),
                 "DISPLAY_NATIVE_FACE_ORIENTATION_UNSUPPORTED")
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
            spline = surface.BSpline() if surface.GetType() == GeomAbs_BSplineSurface else surface.Bezier()
            nu, nv = spline.NbUPoles(), spline.NbVPoles()
            _require(2 <= nu and 2 <= nv and nu * nv <= MAX_TOTAL_CURVE_POLES,
                     "DISPLAY_CONTROL_NET_BUDGET")
            poles = [[[v for v in xyz(spline.Pole(i, j))] for j in range(1, nv + 1)]
                     for i in range(1, nu + 1)]
            weights = [[spline.Weight(i, j) for j in range(1, nv + 1)] for i in range(1, nu + 1)]
            curve_cost += validate_control_net(poles, weights)
        _require(curve_cost <= MAX_TOTAL_CURVE_POLES, "DISPLAY_NATIVE_COMPLEXITY_BUDGET")
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            edge_uses += 1
            _require(edge_uses <= MAX_EDGE_USES, "DISPLAY_EDGE_INCIDENCE_BUDGET")
            edge = TopoDS.Edge_s(explorer.Current())
            if not edge_map.Contains(edge):
                edge_map.Add(edge)
                _require(edge_map.Extent() <= MAX_EDGE_USES, "DISPLAY_EDGE_INCIDENCE_BUDGET")
                curve = _curve_proof(BRepAdaptor_Curve(edge))
                curve_cost += len(curve.poles)
                _require(curve_cost <= MAX_TOTAL_CURVE_POLES, "DISPLAY_NATIVE_COMPLEXITY_BUDGET")
                if curve.kind == "circle":
                    radius, sweep = curve.radius, curve.u1 - curve.u0
                    _require(math.isfinite(radius) and radius > GEOM_TOL
                             and math.isfinite(sweep) and sweep > 0.,
                             "DISPLAY_INVALID_CIRCLE")
                    step = min(0.1, 2 * math.acos(max(-1., min(1., 1. - 0.05 / radius))))
                    _require(step > 0, "DISPLAY_MESH_PREFLIGHT_BUDGET")
                    estimated_segments += math.ceil(sweep / step)
                elif curve.kind == "spline":
                    length = float(np.linalg.norm(np.diff(curve.poles, axis=0), axis=1).sum())
                    _require(math.isfinite(length), "DISPLAY_INVALID_CURVE_LENGTH")
                    estimated_segments += math.ceil(length / 0.05) + len(curve.knots)
                else:
                    estimated_segments += 1
                _require(estimated_segments <= MAX_TRIANGLES, "DISPLAY_MESH_PREFLIGHT_BUDGET")
            explorer.Next()

    mesher = BRepMesh_IncrementalMesh(shape, 0.05, False, 0.1, False)
    mesher.Perform()
    _require(mesher.IsDone(), "DISPLAY_NATIVE_TESSELLATION_UNAVAILABLE")
    result, total_nodes, total_triangles = [], 0, 0
    for index, face in enumerate(faces):
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        _require(mesh is not None and mesh.NbTriangles() > 0, "DISPLAY_NATIVE_FACE_WITHOUT_TRIANGLES")
        total_nodes += mesh.NbNodes()
        total_triangles += mesh.NbTriangles()
        _require(total_nodes <= 3 * MAX_TRIANGLES and total_triangles <= MAX_TRIANGLES,
                 "DISPLAY_TESSELLATION_BUDGET")
        vertices = np.asarray([xyz(mesh.Node(i).Transformed(location.Transformation()))
                               for i in range(1, mesh.NbNodes() + 1)], dtype=np.float64)
        triangles = np.asarray([mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles() + 1)],
                               dtype=np.int32) - 1
        _require(np.isfinite(vertices).all() and np.max(np.abs(vertices)) <= MAX_COORDINATE,
                 "DISPLAY_COORDINATE_BUDGET")
        _require(np.min(triangles) >= 0 and np.max(triangles) < len(vertices),
                 "DISPLAY_INVALID_NATIVE_TRIANGLES")
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, [0, 2, 1]]
        result.append(DisplayFace(index, vertices, triangles,
            str(BRepAdaptor_Surface(face).GetType()), str(face.Orientation())))
    return result
