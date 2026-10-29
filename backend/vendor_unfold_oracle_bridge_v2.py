"""Indexed native observer and unchanged complete source-slice geometry oracle.

No files are opened here. The caller supplies an already authorized native
reader's on_face callback and its original face/axis records. Native topology
identity and edge polygon nodes are captured before development; unavailable
seam polygons or unequal tessellations refuse rather than invent correspondences.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from backend.vendor_unfold_geometry_oracle_v1 import (
    BendEvidence, DevelopedBend, DevelopedFace, FaceEvidence, GeometryOracleError,
    SeamEvidence, capture_native_face_evidence, validate_development,
)


@dataclass
class CapturedFace:
    evidence: FaceEvidence
    kind: str
    edge_vertices: dict[int, tuple[int, ...] | None]
    numerical_area_receipt: dict[str, Any]
    cylinder_metric_endpoints: np.ndarray | None = None
    cylinder_folded_midpoints: np.ndarray | None = None
    cylinder_radius: float | None = None
    cylinder_sweep_deg: float | None = None


def _require(condition, reason):
    if not condition:
        raise GeometryOracleError(reason)


def _capture_discrete_plane(native_face, index):
    """Independent discrete planar test for a normalized spline plane.

Native D1 normals and all native mesh vertices must lie in one plane. This
proves the supplied tessellation model only; continuous spline planarity stays
with the separate native surface proof, and no continuous area bound is set.
"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Pnt, gp_Vec
    orientation = native_face.Orientation()
    _require(orientation in (TopAbs_FORWARD, TopAbs_REVERSED), "BRIDGE_NATIVE_ORIENTATION_UNSUPPORTED")
    loc = TopLoc_Location()
    mesh = BRep_Tool.Triangulation_s(native_face, loc)
    _require(mesh is not None and mesh.HasUVNodes() and 0 < mesh.NbTriangles() <= 200_000
             and 3 <= mesh.NbNodes() <= 600_000, "BRIDGE_NATIVE_PLANE_MESH_UNAVAILABLE")
    vertices = np.array([[p.X(), p.Y(), p.Z()] for p in
                         (mesh.Node(i).Transformed(loc.Transformation()) for i in range(1, mesh.NbNodes()+1))])
    triangles = np.array([mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles()+1)], dtype=np.int32)-1
    if orientation == TopAbs_REVERSED:
        triangles = triangles[:, [0, 2, 1]]
    uv = np.array([[mesh.UVNode(i).X(), mesh.UVNode(i).Y()] for i in range(1, mesh.NbNodes()+1)])
    surface = BRepAdaptor_Surface(native_face)
    normals = []
    for coord in uv[triangles].mean(axis=1):
        p, a, b = gp_Pnt(), gp_Vec(), gp_Vec()
        surface.D1(float(coord[0]), float(coord[1]), p, a, b)
        normal = a.Crossed(b)
        if orientation == TopAbs_REVERSED:
            normal.Reverse()
        normals.append([normal.X(), normal.Y(), normal.Z()])
    normals = np.asarray(normals)
    lengths = np.linalg.norm(normals, axis=1)
    _require(np.isfinite(vertices).all() and np.isfinite(normals).all() and np.all(lengths > 0),
             "BRIDGE_DEGENERATE_NATIVE_PLANE")
    unit = normals/lengths[:, None]
    normal = unit[0]
    _require(np.max(np.linalg.norm(unit-normal, axis=1)) <= 1e-5
             and np.max(np.abs((vertices-vertices[0]) @ normal)) <= 1e-5,
             "BRIDGE_DISCRETE_NATIVE_PLANE_NOT_PLANAR")
    row = vertices[triangles[0]]
    x = row[1]-row[0]
    x -= np.dot(x, normal)*normal
    _require(np.linalg.norm(x) > 1e-5, "BRIDGE_DEGENERATE_NATIVE_PLANE")
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    chart = (vertices-vertices[0]) @ np.stack((x, y)).T
    wires, explorer = 0, TopExp_Explorer(native_face, TopAbs_WIRE)
    while explorer.More():
        wires += 1
        _require(wires <= 1024, "BRIDGE_NATIVE_WIRE_BUDGET_EXCEEDED")
        explorer.Next()
    props = GProp_GProps()
    estimate = float(BRepGProp.SurfaceProperties_s(native_face, props, 1e-10, False))
    area = float(props.Mass())
    _require(math.isfinite(area) and area > 0 and math.isfinite(estimate) and estimate >= 0,
             "BRIDGE_NATIVE_NUMERICAL_AREA_UNAVAILABLE")
    return FaceEvidence(index, vertices, triangles, chart, normals, wires), {
        "area_reference": "OCCT_ADAPTIVE_SURFACE_QUADRATURE", "native_surface_area_mm2": area,
        "returned_relative_convergence_estimate": estimate, "requested_relative_precision": 1e-10,
        "polygon_area_error_bound_mm2": None, "continuous_area_check": "NOT_ESTABLISHED",
        "continuous_trim_proof": False, "plane_chart_model": "NATIVE_D1_AND_ALL_TESSELLATION_VERTICES",
    }


class NativeOracleCapture:
    """Observer for read_native_v6(on_face=capture.on_face); native imports deferred."""
    def __init__(self):
        self.faces: dict[int, CapturedFace] = {}
        self.observed: dict[int, str] = {}
        self.edge_owners: dict[int, set[int]] = {}
        self._native_edges = []
        self._native_edge_index = None
        self._identity_comparisons = 0
        self._edge_uses = 0


    def _capture_edge_identity(self, edge):
        """Independent first-encounter native index, checked with IsSame."""
        count = len(self._native_edges)
        _require(count < 16384 and self._native_edge_index is not None,
                 "BRIDGE_NATIVE_EDGE_USE_BUDGET_EXCEEDED")
        _require(self._native_edge_index.Extent() == count,
                 "BRIDGE_INDEXED_EDGE_IDENTITY_MISMATCH")
        self._identity_comparisons += 1
        _require(self._identity_comparisons <= 1_000_000,
                 "BRIDGE_EDGE_IDENTITY_BUDGET_EXCEEDED")
        native_id = int(self._native_edge_index.Add(edge))
        is_new = native_id == count + 1
        _require(1 <= native_id <= count + 1
                 and self._native_edge_index.Extent() == count + int(is_new)
                 and self._native_edge_index.FindKey(native_id).IsSame(edge),
                 "BRIDGE_INDEXED_EDGE_IDENTITY_MISMATCH")
        if is_new:
            self._native_edges.append(edge)
        return native_id - 1

    def on_face(self, native_face, source_face):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
        from OCP.TopTools import TopTools_IndexedMapOfShape
        if self._native_edge_index is None:
            self._native_edge_index = TopTools_IndexedMapOfShape()
        index = source_face.index
        _require(type(index) is int and index == len(self.observed) and index < 1024,
                 "BRIDGE_NATIVE_FACE_ENUMERATION_MISMATCH")
        loc = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(native_face, loc)
        edges, explorer = {}, TopExp_Explorer(native_face, TopAbs_EDGE)
        while explorer.More():
            self._edge_uses += 1
            _require(self._edge_uses <= 16384, "BRIDGE_NATIVE_EDGE_USE_BUDGET_EXCEEDED")
            edge = TopoDS.Edge_s(explorer.Current())
            ident = self._capture_edge_identity(edge)
            if ident in edges:
                # Periodic seam has two polygons on the same native face. It
                # cannot be used as an unambiguous selected-skin seam, but an
                # excluded cylindrical hole wall must not block capture.
                edges[ident] = None
                explorer.Next()
                continue
            owners = self.edge_owners.setdefault(ident, set())
            owners.add(index)
            _require(len(owners) <= 2 and sum(len(value) for value in self.edge_owners.values()) <= 16384,
                     "BRIDGE_NATIVE_EDGE_INCIDENCE_INVALID")
            polygon = None if mesh is None else BRep_Tool.PolygonOnTriangulation_s(edge, mesh, loc)
            if polygon is not None:
                nodes = polygon.Nodes()
                indices = tuple(int(nodes.Value(j))-1 for j in range(nodes.Lower(), nodes.Upper()+1))
                _require(len(indices) >= 2 and all(0 <= j < mesh.NbNodes() for j in indices),
                         "BRIDGE_NATIVE_EDGE_POLYGON_INVALID")
                edges[ident] = indices
            else:
                edges[ident] = None
            explorer.Next()
        _require(set(edges) == set(source_face.edges), "BRIDGE_IMPORTER_NATIVE_EDGE_IDENTITY_MISMATCH")
        self.observed[index] = source_face.kind
        if source_face.kind not in {"plane", "cylinder"}:
            return
        surface = BRepAdaptor_Surface(native_face)
        kind = surface.GetType()
        if source_face.kind == "plane" and kind != GeomAbs_Plane:
            evidence, receipt = _capture_discrete_plane(native_face, index)
        else:
            _require((source_face.kind == "plane" and kind == GeomAbs_Plane)
                     or (source_face.kind == "cylinder" and kind == GeomAbs_Cylinder),
                     "BRIDGE_NATIVE_SURFACE_KIND_MISMATCH")
            evidence, receipt = capture_native_face_evidence(native_face, index)
        _require(np.array_equal(evidence.vertices, source_face.vertices)
                 and np.array_equal(np.sort(evidence.triangles, axis=1), np.sort(source_face.triangles, axis=1)),
                 "BRIDGE_NATIVE_SOURCE_MESH_MISMATCH")
        midpoint = folded = radius = sweep = None
        if source_face.kind == "cylinder":
            radius = float(surface.Cylinder().Radius())
            middle = radius*(float(surface.FirstUParameter())+float(surface.LastUParameter()))/2
            midpoint = np.array([[middle, float(surface.FirstVParameter())],
                                 [middle, float(surface.LastVParameter())]])
            umid = (float(surface.FirstUParameter())+float(surface.LastUParameter()))/2
            folded = np.array([[p.X(), p.Y(), p.Z()] for p in
                               (surface.Value(umid, float(v)) for v in
                                (surface.FirstVParameter(), surface.LastVParameter()))])
            sweep = math.degrees(abs(float(surface.LastUParameter())-float(surface.FirstUParameter())))
        self.faces[index] = CapturedFace(evidence, source_face.kind, edges, receipt, midpoint,
                                         folded, radius, sweep)


def validate_captured_development(payload, capture: NativeOracleCapture, source_faces,
                                 axes, candidates, *, feature_face_ownership=None):
    """Verify V6 geometry contract, exact original slices and all native seams.

    Feature ownership comes from exact original axis face pairs, independently
    reconstructed native mid-radius lines and unique finite candidate matches.
    Optional explicit ownership is checked against the same independent lines;
    labels alone never establish cylinder ownership.
"""
    from backend.vendor_unfold_contract_v5 import validate_preview
    view = copy.deepcopy(payload)
    view.setdefault("diagnostics", {})["transport_version"] = 5
    validate_preview(view, payload.get("source_sha256"), candidates)
    _require(view["status"] == "AVAILABLE", "BRIDGE_DEVELOPMENT_NOT_AVAILABLE")
    diag = view["diagnostics"]
    _require(len(source_faces) == len(capture.observed) == diag["original_native_face_count"]
             and [face.index for face in source_faces] == list(range(len(source_faces)))
             and set(capture.observed) == set(range(len(source_faces))), "BRIDGE_COMPLETE_NATIVE_CAPTURE_MISSING")
    observed_owners = {}
    for face in source_faces:
        for edge in face.edges:
            observed_owners.setdefault(edge, set()).add(face.index)
        if face.index in capture.faces:
            _require(set(capture.faces[face.index].edge_vertices) == set(face.edges),
                     "BRIDGE_NATIVE_FACE_EDGE_CAPTURE_INCOMPLETE")
    _require(observed_owners == capture.edge_owners, "BRIDGE_COMPLETE_NATIVE_EDGE_INCIDENCE_MISMATCH")
    charts = diag["chart_topology"]["charts"]
    for row in charts:
        members = row["source_face_indices"]
        _require(all(source_faces[i].kind == row["kind"] == capture.observed[i] for i in members),
                 "BRIDGE_CHART_NATIVE_KIND_MISMATCH")
        inside = set(members)
        internal = sorted(edge for edge, owners in capture.edge_owners.items()
                          if len(owners) == 2 and owners.issubset(inside))
        exterior = sorted(set().union(*(set(source_faces[i].edges) for i in members))-set(internal))
        _require(row["removed_internal_edge_ids"] == internal and row["exterior_edge_ids"] == exterior,
                 "BRIDGE_CHART_NATIVE_EDGE_LEDGER_MISMATCH")
        for member in row["members"]:
            source = source_faces[member["source_face_index"]]
            _require(member["vertex_count"] == len(source.vertices)
                     and member["triangle_count"] == len(source.triangles), "BRIDGE_CHART_SOURCE_COUNTS_MISMATCH")
        if len(members) > 1:
            _require(row["kind"] == "plane", "BRIDGE_NONPLANAR_CHART_STITCH")
            _require(all(i in capture.faces for i in members), "BRIDGE_CHART_NATIVE_EVIDENCE_MISSING")
            anchor = capture.faces[members[0]].evidence
            normal = anchor.oriented_normals[0]/np.linalg.norm(anchor.oriented_normals[0])
            for ident in members:
                face = capture.faces[ident].evidence
                normals = face.oriented_normals/np.linalg.norm(face.oriented_normals, axis=1)[:, None]
                _require(np.max(np.abs((face.vertices-anchor.vertices[0]) @ normal)) <= 1e-5
                         and np.max(np.minimum(np.linalg.norm(normals-normal, axis=1),
                                                np.linalg.norm(normals+normal, axis=1))) <= 1e-5,
                         "BRIDGE_NATIVE_COPLANAR_CHART_DRIFT")
            reachable, stack = set(), [members[0]]
            while stack:
                node = stack.pop()
                if node in reachable:
                    continue
                reachable.add(node)
                stack.extend(other for edge in internal if node in capture.edge_owners[edge]
                             for other in capture.edge_owners[edge]-reachable)
            _require(reachable == inside, "BRIDGE_DISCONNECTED_NATIVE_CHART")
    points = np.asarray(view["vertices"]).reshape(-1, 3)
    triangles = np.asarray(view["triangles"], dtype=np.int64).reshape(-1, 3)
    developed, evidence = {}, []
    for row in diag["developed_native_face_slices"]:
        ident = row["source_face_index"]
        _require(ident in capture.faces, "BRIDGE_SELECTED_NATIVE_EVIDENCE_MISSING")
        original = source_faces[ident]
        native = capture.faces[ident]
        _require(np.array_equal(native.evidence.vertices, original.vertices)
                 and np.array_equal(np.sort(native.evidence.triangles, axis=1), np.sort(original.triangles, axis=1)),
                 "BRIDGE_CAPTURE_SOURCE_IDENTITY_MISMATCH")
        start, count = row["body_vertex_start"], row["body_vertex_count"]
        tstart, tcount = row["body_triangle_start"], row["body_triangle_count"]
        developed[ident] = DevelopedFace(ident, points[start:start+count], triangles[tstart:tstart+tcount]-start)
        evidence.append(native.evidence)
    selected = set(developed)
    seams, seam_rows = [], []
    for edge, owners in sorted(capture.edge_owners.items()):
        if len(owners) != 2 or not owners.issubset(selected):
            continue
        left_id, right_id = sorted(owners)
        a, b = capture.faces[left_id].edge_vertices[edge], capture.faces[right_id].edge_vertices[edge]
        _require(a is not None and b is not None and len(a) == len(b), "BRIDGE_NATIVE_SEAM_POLYGON_UNAVAILABLE_OR_UNEQUAL")
        left = capture.faces[left_id].evidence.vertices[list(a)]
        right = capture.faces[right_id].evidence.vertices[list(b)]
        direct, reverse = np.max(np.linalg.norm(left-right, axis=1)), np.max(np.linalg.norm(left-right[::-1], axis=1))
        _require(min(direct, reverse) <= 1e-5, "BRIDGE_NATIVE_SEAM_NODE_IDENTITY_MISMATCH")
        if reverse < direct:
            b = b[::-1]
        seams.append(SeamEvidence(left_id, a, right_id, b))
        seam_rows.append({"native_edge_id": edge, "source_faces": [left_id, right_id], "native_node_count": len(a)})
    reachable, stack = set(), [min(selected)]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(other for row in seam_rows if node in row["source_faces"] for other in row["source_faces"] if other not in reachable)
    _require(reachable == selected, "BRIDGE_SELECTED_NATIVE_SEAM_GRAPH_DISCONNECTED")
    candidate_ids = {row["feature_id"] for row in candidates}
    _require(len(axes) == len(candidate_ids), "BRIDGE_NATIVE_FEATURE_COUNT_MISMATCH")
    axis_pairs = set()
    expected_pairs = {}
    for row in axes:
        indices = row.get("face_indices")
        _require(type(indices) is list and len(indices) == 2 and all(type(i) is int and i > 0 for i in indices),
                 "BRIDGE_NATIVE_AXIS_PAIR_INVALID")
        pair = tuple(sorted(j-1 for j in indices))
        _require(len(set(pair)) == 2 and all(i in capture.faces and capture.faces[i].kind == "cylinder" for i in pair),
                 "BRIDGE_NATIVE_AXIS_PAIR_INVALID")
        first, second = (capture.faces[i] for i in pair)
        a, b = first.cylinder_folded_midpoints, second.cylinder_folded_midpoints
        _require(a is not None and b is not None and np.shape(a) == np.shape(b) == (2, 3)
                 and first.cylinder_radius is not None and second.cylinder_radius is not None
                 and first.cylinder_sweep_deg is not None and second.cylinder_sweep_deg is not None,
                 "BRIDGE_NATIVE_AXIS_MIDRADIUS_EVIDENCE_MISSING")
        if np.max(np.linalg.norm(a-b[::-1], axis=1)) < np.max(np.linalg.norm(a-b, axis=1)):
            b = b[::-1]
        expected = (a+b)/2
        native_axis_line = np.array([row["canonical_line_start"], row["canonical_line_end"]], dtype=float)
        error = min(np.max(np.linalg.norm(native_axis_line-expected, axis=1)),
                    np.max(np.linalg.norm(native_axis_line[::-1]-expected, axis=1)))
        _require(math.isfinite(error) and error <= .002, "BRIDGE_EXTRACTED_AXIS_NATIVE_MIDRADIUS_MISMATCH")
        _require(abs(first.cylinder_sweep_deg-second.cylinder_sweep_deg) <= .0005,
                 "BRIDGE_NATIVE_PAIRED_SWEEP_MISMATCH")
        expected_pairs[pair] = (expected, min(first.cylinder_radius, second.cylinder_radius),
                                (first.cylinder_sweep_deg+second.cylinder_sweep_deg)/2)
        axis_pairs.add(pair)
    _require(len(axis_pairs) == len(axes), "BRIDGE_DUPLICATE_NATIVE_AXIS_OWNERSHIP")
    ownership = {}
    for candidate in candidates:
        line = np.array([candidate["line_start"], candidate["line_end"]], dtype=float)
        admissible = []
        for pair, (expected, radius, sweep) in expected_pairs.items():
            error = min(np.max(np.linalg.norm(line-expected, axis=1)),
                        np.max(np.linalg.norm(line[::-1]-expected, axis=1)))
            if math.isfinite(error) and error <= .002:
                admissible.append(pair)
        _require(len(admissible) == 1, "BRIDGE_CANDIDATE_NATIVE_AXIS_NOT_UNIQUE")
        pair = admissible[0]
        _, radius, sweep = expected_pairs[pair]
        for key, expected in (("target_radius", radius), ("target_angle", sweep)):
            if key in candidate:
                _require(type(candidate[key]) in (int, float) and math.isfinite(candidate[key])
                         and abs(candidate[key]-expected) <= .002, "BRIDGE_CANDIDATE_NATIVE_METADATA_MISMATCH")
        ownership[candidate["feature_id"]] = pair
    _require(len(set(ownership.values())) == len(axes), "BRIDGE_CANDIDATE_NATIVE_AXIS_REUSED")
    if feature_face_ownership is not None:
        _require(set(feature_face_ownership) == candidate_ids and all(
            tuple(sorted(feature_face_ownership[ident])) == pair for ident, pair in ownership.items()),
            "BRIDGE_EXPLICIT_NATIVE_FEATURE_OWNERSHIP_MISMATCH")
    bend_evidence, bend_outputs, selected_cylinders = [], [], set()
    by_id = {row["feature_id"]: row for row in view["bend_lines"]}
    for ident, pair in ownership.items():
        _require(len(pair) == 2 and len(set(pair)) == 2 and tuple(sorted(pair)) in axis_pairs
                 and all(type(i) is int and i in capture.faces and capture.faces[i].kind == "cylinder" for i in pair),
                 "BRIDGE_NATIVE_FEATURE_CYLINDER_OWNERSHIP_INVALID")
        present = set(pair)&selected
        _require(len(present) == 1, "BRIDGE_SELECTED_NATIVE_BEND_SKIN_AMBIGUOUS")
        cylinder = present.pop()
        _require(cylinder not in selected_cylinders, "BRIDGE_SELECTED_NATIVE_BEND_REUSED")
        selected_cylinders.add(cylinder)
        proof = next(row for row in diag["finite_width_proofs"] if row["feature_id"] == ident)
        proof_chart = charts[proof["cylinder_source_face_index"]]
        _require(proof_chart["kind"] == "cylinder" and proof_chart["source_face_indices"] == [cylinder],
                 "BRIDGE_DEVELOPED_FEATURE_NATIVE_OWNERSHIP_MISMATCH")
        _require(capture.faces[cylinder].cylinder_metric_endpoints is not None,
                 "BRIDGE_NATIVE_CYLINDER_MIDSWEEP_UNAVAILABLE")
        bend_evidence.append(BendEvidence(ident, cylinder, capture.faces[cylinder].cylinder_metric_endpoints))
        bend_outputs.append(DevelopedBend(ident, np.array([by_id[ident]["line_start"], by_id[ident]["line_end"]])))
    _require(selected_cylinders == {i for i in selected if capture.faces[i].kind == "cylinder"},
             "BRIDGE_UNASSIGNED_SELECTED_NATIVE_CYLINDER")
    result = validate_development(evidence, list(developed.values()), seams,
                                  bend_evidence=bend_evidence, developed_bends=bend_outputs)
    result.update({"bridge_model": "EXACT_NATIVE_FACE_EDGE_POLYGON_AND_V5_SOURCE_SLICES",
                   "native_face_count": len(source_faces), "selected_native_face_indices": sorted(selected),
                   "all_selected_shared_native_edges_checked": True, "native_seams": seam_rows,
                   "native_numerical_area_observations": {str(i): capture.faces[i].numerical_area_receipt for i in sorted(selected)},
                   "continuous_area_check": "NOT_ESTABLISHED"})
    return result
