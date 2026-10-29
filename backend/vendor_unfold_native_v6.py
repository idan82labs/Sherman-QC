"""Indexed-identity native V6 reader with injectable chart/axis proof seams.

The incumbent reader source is retained; native safety, ownership and meshing
gates are copied unchanged except the pre-tessellation surface proof and a
post-chart native-face observer. No CAD is read at import.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from backend.vendor_unfold_preview import (SkinFace, _require, _curve_proof, MAX_FACES, MAX_TOTAL_CURVE_POLES, MAX_EDGE_USES, MAX_TRIANGLES, GEOM_TOL)
from backend.vendor_surface_normalization_v2 import surface_proof_v2



def indexed_native_identity(indexed_map, edge, previous_count):
    """First-encounter ID via native IsSame hashing, ignoring orientation.

    The returned zero-based index preserves the V5 observer's enumeration.
    This verifies each lookup against native IsSame once, without a Python
    all-prior-edges scan. The caller bounds each edge use before invoking it.
    """
    _require(type(previous_count) is int and 0 <= previous_count < MAX_EDGE_USES,
             "EDGE_INCIDENCE_BUDGET_EXCEEDED")
    _require(int(indexed_map.Extent()) == previous_count,
             "NATIVE_INDEXED_EDGE_IDENTITY_MISMATCH")
    native_id = int(indexed_map.Add(edge))
    extent = int(indexed_map.Extent())
    _require(1 <= native_id <= previous_count + 1
             and extent in (previous_count, previous_count + 1),
             "NATIVE_INDEXED_EDGE_IDENTITY_MISMATCH")
    is_new = native_id == previous_count + 1
    _require(extent == previous_count + int(is_new)
             and indexed_map.FindKey(native_id).IsSame(edge),
             "NATIVE_INDEXED_EDGE_IDENTITY_MISMATCH")
    return native_id - 1, is_new


def read_native_v6(path: Path, *, surface_proof=surface_proof_v2, on_face=None, axis_extractor=None) -> tuple[list[SkinFace], list[dict[str, Any]]]:
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
    from OCP.TopTools import TopTools_IndexedMapOfShape
    if axis_extractor is None:
        from scripts.extract_step_topology_bend_axes_v2 import extract_step_bend_axes
        axis_extractor = extract_step_bend_axes

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
    native_edge_index = TopTools_IndexedMapOfShape()
    for native_face in native_faces:
        surface = BRepAdaptor_Surface(native_face)
        descriptor = surface_proof(surface)
        surface_proofs.append(descriptor)
        total_curve_poles += descriptor["pole_cost"]
        _require(total_curve_poles <= MAX_TOTAL_CURVE_POLES, "NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
        ids = set()
        edges = TopExp_Explorer(native_face, TopAbs_EDGE)
        while edges.More():
            preflight_uses += 1
            _require(preflight_uses <= MAX_EDGE_USES, "EDGE_INCIDENCE_BUDGET_EXCEEDED")
            edge = TopoDS.Edge_s(edges.Current())
            match, is_new = indexed_native_identity(native_edge_index, edge, len(native_edges))
            if is_new:
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
        if on_face is not None:
            on_face(native_face, face)
        faces.append(face)
    axes = axis_extractor(path)["bend_axes"]
    return faces, axes
