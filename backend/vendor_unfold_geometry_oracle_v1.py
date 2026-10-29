"""Independent, pure checks of selected-skin development geometry.

No unfold implementation, diagnostics, native importer, files or raw CAD are
read. Evidence must be captured independently from the native faces. Areas
refer to the selected geometric skin, never a manufacturing neutral axis.

For a plane the metric chart is orthonormal XY. For a cylinder it is (R*u,v),
whose first fundamental form is dx**2+dy**2. A developed face must therefore
equal that independently supplied chart up to one Euclidean plane isometry.
This is stronger than preserving only triangle edge lengths: adjacent
triangles cannot hinge or independently reflect in the developed plane.

Discrete native metric-polygon area is always checked against the flat mesh.
A separate continuous-area reference/error model is optional, and must be
supplied together. Absent that model, continuous area remains NOT_ESTABLISHED.
Mesher deflection and observed quadrature differences do not establish a bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class GeometryOracleError(ValueError):
    pass


@dataclass(frozen=True)
class FaceEvidence:
    source_index: int
    vertices: np.ndarray
    # Native face orientation must already be applied to these triangles.
    triangles: np.ndarray
    metric_chart: np.ndarray
    # Native oriented surface normals, evaluated independently at triangle
    # centroids in UV, then transformed by the native Location.
    oriented_normals: np.ndarray
    boundary_cycles: int
    continuous_area_mm2: float | None = None
    # Includes quadrature estimate and any independently justified polygonal
    # approximation error. Never silently derived from an engine diagnostic.
    area_error_bound_mm2: float | None = None


@dataclass(frozen=True)
class DevelopedFace:
    source_index: int
    vertices: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class SeamEvidence:
    left_face: int
    left_vertices: tuple[int, ...]
    right_face: int
    right_vertices: tuple[int, ...]


@dataclass(frozen=True)
class BendEvidence:
    feature_id: str
    source_face: int
    # Independently chosen selected-skin cylinder chart: the two endpoints are
    # (R*(u_min+u_max)/2, v_min/max). They are not neutral-axis locations.
    metric_endpoints: np.ndarray


@dataclass(frozen=True)
class DevelopedBend:
    feature_id: str
    endpoints: np.ndarray


def capture_native_face_evidence(native_face, source_index: int, *,
                                 polygon_area_error_bound_mm2: float | None = None,
                                 quadrature_relative_precision: float = 1e-10):
    """Capture independent evidence from an ALREADY authorized/meshed face.

This does no file access or meshing. Call inside the enclosing worker's fixed
native budget. OCCT normals come from D1 + native face Orientation; charts come
from UV nodes; areas use surface quadrature with UseTriangulation disabled.
The numerical convergence estimate is recorded but not a mathematical bound.
The caller may separately justify a polygonal trim approximation allowance.
Without one, the returned evidence supports discrete geometry checks only;
quadrature area and convergence remain separate observations in the receipt.
"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Pnt, gp_Vec

    _require(type(source_index) is int and source_index >= 0
             and (polygon_area_error_bound_mm2 is None or
                  (type(polygon_area_error_bound_mm2) in (int, float)
                   and math.isfinite(polygon_area_error_bound_mm2) and polygon_area_error_bound_mm2 >= 0))
             and math.isfinite(quadrature_relative_precision) and 0 < quadrature_relative_precision <= 1e-6,
             "INVALID_ORACLE_CAPTURE_CONFIGURATION")
    orientation = native_face.Orientation()
    _require(orientation in (TopAbs_FORWARD, TopAbs_REVERSED), "UNSUPPORTED_ORACLE_FACE_ORIENTATION")
    surface = BRepAdaptor_Surface(native_face)
    kind = surface.GetType()
    _require(kind in (GeomAbs_Plane, GeomAbs_Cylinder), "UNSUPPORTED_ORACLE_NATIVE_SURFACE")
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(native_face, location)
    _require(triangulation is not None and triangulation.HasUVNodes()
             and 0 < triangulation.NbTriangles() <= 200_000 and triangulation.NbNodes() <= 600_000,
             "ORACLE_NATIVE_TRIANGULATION_UNAVAILABLE")
    vertices = np.array([[point.X(), point.Y(), point.Z()] for point in
                         (triangulation.Node(i).Transformed(location.Transformation())
                          for i in range(1, triangulation.NbNodes()+1))])
    triangles = np.array([triangulation.Triangle(i).Get()
                          for i in range(1, triangulation.NbTriangles()+1)], dtype=np.int32)-1
    if orientation == TopAbs_REVERSED:
        triangles = triangles[:, [0, 2, 1]]
    uv = np.array([[point.X(), point.Y()] for point in
                   (triangulation.UVNode(i) for i in range(1, triangulation.NbNodes()+1))])
    chart = uv.copy()
    if kind == GeomAbs_Cylinder:
        chart[:, 0] *= float(surface.Cylinder().Radius())
    normals = []
    for coordinate in uv[triangles].mean(axis=1):
        point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        surface.D1(float(coordinate[0]), float(coordinate[1]), point, du, dv)
        normal = du.Crossed(dv)
        if orientation == TopAbs_REVERSED:
            normal.Reverse()
        normals.append([normal.X(), normal.Y(), normal.Z()])
    wires, explorer = 0, TopExp_Explorer(native_face, TopAbs_WIRE)
    while explorer.More():
        wires += 1
        _require(wires <= 1024, "ORACLE_WIRE_BUDGET_EXCEEDED")
        explorer.Next()
    properties = GProp_GProps()
    estimated_error = float(BRepGProp.SurfaceProperties_s(native_face, properties,
                            quadrature_relative_precision, False))
    area = float(properties.Mass())
    _require(math.isfinite(area) and area > 0 and math.isfinite(estimated_error)
             and 0 <= estimated_error <= quadrature_relative_precision,
             "ORACLE_SURFACE_QUADRATURE_UNAVAILABLE")
    # Include requested precision as well as returned convergence estimate so
    # an exactly zero reported convergence does not imply exact arithmetic.
    comparison_allowance = (None if polygon_area_error_bound_mm2 is None else
                            polygon_area_error_bound_mm2
                            + area*max(estimated_error, quadrature_relative_precision))
    return FaceEvidence(source_index, vertices, triangles, chart, np.asarray(normals), wires,
                        None if comparison_allowance is None else area, comparison_allowance), {
        "area_reference": "OCCT_ADAPTIVE_SURFACE_QUADRATURE",
        "native_surface_area_mm2": area,
        "triangulation_used_for_area": False,
        "returned_relative_convergence_estimate": estimated_error,
        "requested_relative_precision": quadrature_relative_precision,
        "polygon_area_error_bound_mm2": polygon_area_error_bound_mm2,
        "continuous_area_check": "NOT_ESTABLISHED" if comparison_allowance is None else "DECLARED_ERROR_MODEL_AVAILABLE",
        "continuous_trim_proof": False,
    }


def _require(condition, reason):
    if not condition:
        raise GeometryOracleError(reason)


def _mesh(vertices, triangles, dimension):
    p, t = np.asarray(vertices, dtype=float), np.asarray(triangles)
    _require(p.ndim == 2 and p.shape[1] == dimension and 3 <= len(p) <= 600_000
             and np.isfinite(p).all(), "INVALID_ORACLE_VERTICES")
    _require(t.ndim == 2 and t.shape[1] == 3 and 0 < len(t) <= 200_000
             and np.issubdtype(t.dtype, np.integer) and t.min() >= 0
             and t.max() < len(p), "INVALID_ORACLE_TRIANGLES")
    _require(len({tuple(sorted(row)) for row in t}) == len(t), "DUPLICATE_ORACLE_TRIANGLE")
    return p, t


def _twice_area(points, triangles):
    q = points[triangles]
    a, b = q[:, 1]-q[:, 0], q[:, 2]-q[:, 0]
    return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]


def _boundary_cycles(triangles):
    """Oriented combinatorial boundary; holes cannot disappear by filling."""
    edges = {}
    for row in triangles:
        for a, b in zip(row, np.roll(row, -1)):
            key = tuple(sorted((int(a), int(b))))
            edges.setdefault(key, []).append((int(a), int(b)))
    _require(all(len(uses) <= 2 for uses in edges.values()), "NONMANIFOLD_ORACLE_MESH")
    _require(all(len(uses) == 1 or uses[0] == uses[1][::-1]
                 for uses in edges.values()), "INCONSISTENT_ORACLE_WINDING")
    boundary = [uses[0] for uses in edges.values() if len(uses) == 1]
    _require(bool(boundary), "MISSING_ORACLE_BOUNDARY")
    outgoing, incoming = {}, {}
    for a, b in boundary:
        _require(a not in outgoing and b not in incoming, "BRANCHED_ORACLE_BOUNDARY")
        outgoing[a], incoming[b] = b, a
    _require(set(outgoing) == set(incoming), "OPEN_ORACLE_BOUNDARY")
    remaining, count = set(outgoing), 0
    while remaining:
        start = min(remaining)
        cursor = start
        while cursor in remaining:
            remaining.remove(cursor)
            cursor = outgoing[cursor]
        _require(cursor == start, "SELF_JOINED_ORACLE_BOUNDARY")
        count += 1
    # A connected triangulated face with B boundaries has Euler chi=2-B.
    # This prevents disjoint islands from masquerading as additional holes.
    used = set(int(i) for i in triangles.flat)
    _require(len(used)-len(edges)+len(triangles) == 2-count,
             "DISCONNECTED_OR_HANDLE_ORACLE_FACE")
    return count


def validate_development(evidence: Sequence[FaceEvidence],
                         developed: Sequence[DevelopedFace],
                         seams: Sequence[SeamEvidence] = (), *,
                         bend_evidence: Sequence[BendEvidence] = (),
                         developed_bends: Sequence[DevelopedBend] = (),
                         length_tolerance_mm: float = 1e-5,
                         area_roundoff_mm2: float = 1e-8):
    """Return auditable numeric evidence, or refuse; never mutate inputs.

The selected face set and source-vertex correspondence are explicit inputs,
not accepted from engine diagnostics. Identity, metric, winding, boundaries,
discrete area and supplied seams are checked for every supplied face. Continuous
area is checked only when its independent reference and error model are given.
"""
    _require(math.isfinite(length_tolerance_mm) and 0 < length_tolerance_mm <= .01
             and math.isfinite(area_roundoff_mm2) and 0 <= area_roundoff_mm2 <= .01,
             "INVALID_ORACLE_TOLERANCE")
    _require(0 < len(evidence) <= 1024 and len(developed) == len(evidence), "ORACLE_FACE_COUNT_MISMATCH")
    _require(sum(len(row.triangles) for row in evidence) <= 200_000
             and sum(len(row.vertices) for row in evidence) <= 600_000,
             "ORACLE_SHARED_GEOMETRY_BUDGET_EXCEEDED")
    src = {row.source_index: row for row in evidence}
    dst = {row.source_index: row for row in developed}
    _require(len(src) == len(evidence) and len(dst) == len(developed)
             and set(src) == set(dst), "ORACLE_SOURCE_FACE_BIJECTION_MISMATCH")
    rows, chart_maps, global_triangle_parity = [], {}, None
    for index in sorted(src):
        source, output = src[index], dst[index]
        p, t = _mesh(source.vertices, source.triangles, 3)
        f, ft = _mesh(output.vertices, output.triangles, 3)
        chart = np.asarray(source.metric_chart, dtype=float)
        normals = np.asarray(source.oriented_normals, dtype=float)
        _require(chart.shape == (len(p), 2) and np.isfinite(chart).all()
                 and normals.shape == (len(t), 3) and np.isfinite(normals).all(), "INVALID_ORACLE_NATIVE_CHART")
        _require(f.shape == p.shape and ft.shape == t.shape
                 and np.array_equal(np.sort(ft, axis=1), np.sort(t, axis=1)),
                 "ORACLE_SOURCE_TRIANGLE_LINEAGE_MISMATCH")
        same = np.logical_or.reduce([np.all(ft == np.roll(t, shift, axis=1), axis=1)
                                     for shift in (0, 1, 2)])
        parity = np.where(same, 1, -1)
        _require(np.all(parity == parity[0]), "ORACLE_LOCAL_TRIANGLE_PARITY_MISMATCH")
        if global_triangle_parity is None:
            global_triangle_parity = int(parity[0])
        _require(np.all(parity == global_triangle_parity), "ORACLE_CROSS_FACE_TRIANGLE_PARITY_MISMATCH")
        nlength = np.linalg.norm(normals, axis=1)
        cross = np.cross(p[t[:, 1]]-p[t[:, 0]], p[t[:, 2]]-p[t[:, 0]])
        clength = np.linalg.norm(cross, axis=1)
        _require(np.all(nlength > 0) and np.all(clength > area_roundoff_mm2)
                 and np.all(np.sum(cross*normals, axis=1) > .5*clength*nlength),
                 "ORACLE_NATIVE_FACE_ORIENTATION_MISMATCH")
        _require(np.max(np.abs(f[:, 2])) <= length_tolerance_mm, "ORACLE_FLAT_NOT_PLANAR")
        signs = _twice_area(f[:, :2], ft)
        # The exported visual mesh contract chooses +Z consistently. No
        # DoubleSide rendering can substitute for coherent surface winding.
        _require(np.all(signs > area_roundoff_mm2), "ORACLE_FLAT_WINDING_MISMATCH")
        source_cycles = _boundary_cycles(t)
        output_cycles = _boundary_cycles(ft)
        _require(type(source.boundary_cycles) is int and source.boundary_cycles > 0
                 and source_cycles == output_cycles == source.boundary_cycles,
                 "ORACLE_NATIVE_HOLE_BOUNDARY_MISMATCH")
        a, b = chart-chart.mean(axis=0), f[:, :2]-f[:, :2].mean(axis=0)
        _require(np.linalg.matrix_rank(a) == 2, "DEGENERATE_ORACLE_CHART")
        left, _, right = np.linalg.svd(a.T @ b)
        orthogonal = left @ right
        chart_maps[index] = (orthogonal, f[:, :2].mean(axis=0)-chart.mean(axis=0) @ orthogonal)
        residual = float(np.max(np.linalg.norm(a @ orthogonal-b, axis=1)))
        _require(residual <= length_tolerance_mm, "ORACLE_INTRINSIC_METRIC_MISMATCH")
        chart_area = float(np.abs(_twice_area(chart, t)).sum()/2)
        flat_area = float(signs.sum()/2)
        _require(abs(chart_area-flat_area) <= area_roundoff_mm2,
                 "ORACLE_DISCRETE_METRIC_POLYGON_AREA_MISMATCH")
        exact, bound = source.continuous_area_mm2, source.area_error_bound_mm2
        _require((exact is None) == (bound is None), "INCOMPLETE_ORACLE_CONTINUOUS_AREA_MODEL")
        continuous_check = "NOT_ESTABLISHED"
        if exact is not None:
            _require(type(exact) in (int, float) and math.isfinite(exact) and exact > 0
                     and type(bound) in (int, float) and math.isfinite(bound) and bound >= 0,
                     "INVALID_ORACLE_AREA_REFERENCE")
            _require(abs(chart_area-exact) <= bound+area_roundoff_mm2
                     and abs(flat_area-exact) <= bound+area_roundoff_mm2,
                     "ORACLE_CONTINUOUS_AREA_MISMATCH")
            continuous_check = "PASSED_UNDER_DECLARED_ERROR_MODEL"
        rows.append({"source_index": index, "flat_area_mm2": flat_area,
                     "metric_chart_area_mm2": chart_area, "continuous_area_mm2": exact,
                     "discrete_metric_polygon_area_check": "PASSED",
                     "continuous_area_check": continuous_check,
                     "declared_area_error_bound_mm2": bound, "metric_residual_mm": residual,
                     "boundary_cycles": source_cycles})
    max_seam = 0.
    for seam in seams:
        _require(seam.left_face in dst and seam.right_face in dst
                 and seam.left_face != seam.right_face and len(seam.left_vertices) >= 2
                 and len(seam.left_vertices) == len(seam.right_vertices), "INVALID_ORACLE_SEAM")
        for face_id, indices in ((seam.left_face, seam.left_vertices), (seam.right_face, seam.right_vertices)):
            _require(all(type(i) is int and 0 <= i < len(dst[face_id].vertices) for i in indices),
                     "INVALID_ORACLE_SEAM_VERTEX")
        left = np.asarray(dst[seam.left_face].vertices)[list(seam.left_vertices)]
        right = np.asarray(dst[seam.right_face].vertices)[list(seam.right_vertices)]
        error = float(np.max(np.linalg.norm(left-right, axis=1)))
        max_seam = max(max_seam, error)
        _require(error <= length_tolerance_mm, "ORACLE_DEVELOPED_SEAM_MISMATCH")
    bend_src = {row.feature_id: row for row in bend_evidence}
    bend_dst = {row.feature_id: row for row in developed_bends}
    _require(len(bend_src) == len(bend_evidence) and len(bend_dst) == len(developed_bends)
             and set(bend_src) == set(bend_dst), "ORACLE_BEND_BIJECTION_MISMATCH")
    max_bend_error = 0.
    for ident, row in bend_src.items():
        chart_endpoints = np.asarray(row.metric_endpoints, dtype=float)
        actual = np.asarray(bend_dst[ident].endpoints, dtype=float)
        _require(type(ident) is str and bool(ident) and row.source_face in chart_maps
                 and chart_endpoints.shape == (2, 2) and actual.shape == (2, 3)
                 and np.isfinite(chart_endpoints).all() and np.isfinite(actual).all(),
                 "INVALID_ORACLE_BEND_LINE")
        rotation, translation = chart_maps[row.source_face]
        expected = np.column_stack((chart_endpoints @ rotation+translation, np.zeros(2)))
        error = min(float(np.max(np.linalg.norm(actual-expected, axis=1))),
                    float(np.max(np.linalg.norm(actual[::-1]-expected, axis=1))))
        max_bend_error = max(max_bend_error, error)
        _require(error <= length_tolerance_mm, "ORACLE_BEND_LOCATION_OR_SPAN_MISMATCH")
    return {"status": "VERIFIED_UNDER_DECLARED_GEOMETRY_MODEL", "manufacturing_ready": False,
            "verification_model": "DISCRETE_NATIVE_METRIC_CHART",
            "face_count": len(rows), "faces": rows, "max_seam_error_mm": max_seam,
            "seam_count": len(seams),
            "global_source_triangle_parity": global_triangle_parity,
            "bend_count": len(bend_src), "max_bend_line_error_mm": max_bend_error,
            "total_flat_area_mm2": sum(row["flat_area_mm2"] for row in rows),
            "continuous_area_check": "PASSED_UNDER_DECLARED_ERROR_MODEL" if
                all(row["continuous_area_check"] == "PASSED_UNDER_DECLARED_ERROR_MODEL" for row in rows)
                else "NOT_ESTABLISHED",
            "continuous_trim_proof": False, "independent_native_truth_required": True}
