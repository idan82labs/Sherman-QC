"""Detection extension for finite cylindrical strips with curved end boundaries.

V7 rectangle outputs are retained verbatim when no extension qualifies. New
features retain ordered native curved boundaries and mark only a certified
interior span. Their receipts cannot enter the rectangular unfolding adapter.
"""
from collections import Counter
import copy
import math
from pathlib import Path
import numpy as np
from scripts import extract_step_topology_bend_axes_v7 as legacy
from backend.cad_finite_trimmed_cylinder_domain_v1 import certify_graph_strip, certify_single_edge_end_join_core


def _span(poles, weights, edge_id, provenance):
    return {"poles": poles, "weights": weights, "native_edge_id": edge_id,
            "actual_interval_proven": True, "active_interval_provenance": provenance}


def native_graph_certificate(face, surface, edge_map):
    """Read complete oriented wire occurrences; never snap native coefficients."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_BezierCurve
    from OCP.Geom2dConvert import Geom2dConvert_BSplineCurveToBezierCurve
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    receipt = {"status": "REFUSED", "type_proven": False, "detection_only": True,
               "unfold_certificate": False, "certificate_schema_version": 8}
    def refuse(reason):
        return {**receipt, "reason": reason}
    if not BRepCheck_Analyzer(face).IsValid():
        return refuse("INVALID_NATIVE_FACE")
    radius = float(surface.Cylinder().Radius())
    if not 0. < surface.LastUParameter()-surface.FirstUParameter() < 2*math.pi-legacy.DOMAIN_EPS:
        return refuse("SINGLE_TURN_UNWRAPPED_NATIVE_CHART_REQUIRED")
    wires = []
    vertices = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(face, TopAbs_VERTEX, vertices)
    source_tolerance = float(BRep_Tool.Tolerance_s(face))
    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while explorer.More():
        if wires:
            return refuse("HOLES_OR_MULTIPLE_WIRES_UNSUPPORTED")
        wire = TopoDS.Wire_s(explorer.Current())
        if not wire.IsSame(BRepTools.OuterWire_s(face)):
            return refuse("NATIVE_OUTER_WIRE_REQUIRED")
        census = TopExp_Explorer(wire, TopAbs_EDGE)
        expected = []
        while census.More():
            expected.append((edge_map.FindIndex(census.Current()), str(census.Current().Orientation())))
            if len(expected) > 256:
                return refuse("NATIVE_WIRE_BUDGET")
            census.Next()
        ordered, observed = [], []
        traversal = BRepTools_WireExplorer(wire, face)
        while traversal.More():
            edge = TopoDS.Edge_s(traversal.Current())
            edge_id = edge_map.FindIndex(edge)
            orientation = edge.Orientation()
            observed.append((edge_id, str(orientation)))
            if len(observed) > 256 or edge_id <= 0 or orientation not in (TopAbs_FORWARD, TopAbs_REVERSED):
                return refuse("NATIVE_WIRE_IDENTITY_OR_ORIENTATION")
            curve = BRepAdaptor_Curve2d(edge, face)
            first, last = float(curve.FirstParameter()), float(curve.LastParameter())
            if not math.isfinite(first) or not math.isfinite(last) or last <= first:
                return refuse("INVALID_NATIVE_ACTIVE_INTERVAL")
            kind = curve.GetType()
            provenance = {"actual_interval": [first, last], "representation": str(kind),
                          "edge_orientation": str(orientation), "copy_restricted": False}
            spans = []
            if kind == GeomAbs_Line:
                p, q = curve.Value(first), curve.Value(last)
                spans = [_span([[float(p.X()*radius), float(p.Y())],
                                [float(q.X()*radius), float(q.Y())]], [1., 1.], edge_id, provenance)]
            elif kind in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
                if kind == GeomAbs_BSplineCurve:
                    spline, restriction = legacy.restrict_bspline_actual_interval(curve.BSpline(), first, last)
                    if spline is None:
                        return refuse(restriction["reason"])
                    conversion = Geom2dConvert_BSplineCurveToBezierCurve(spline)
                    if not 1 <= conversion.NbArcs() <= 256:
                        return refuse("BEZIER_SPAN_BUDGET")
                    arcs = [conversion.Arc(i) for i in range(1, conversion.NbArcs()+1)]
                    provenance.update(copy_restricted=True, restriction=restriction,
                                      conversion="NATIVE_EXACT_KNOT_INSERTION_BEZIER_DECOMPOSITION")
                else:
                    if not 0. <= first < last <= 1.:
                        return refuse("BEZIER_ACTIVE_INTERVAL_OUTSIDE_NATIVE_DOMAIN")
                    arc = curve.Bezier().Copy()
                    if first != 0. or last != 1.:
                        arc.Segment(first, last)
                    arcs = [arc]
                    provenance.update(copy_restricted=True)
                for arc in arcs:
                    if not 2 <= arc.NbPoles() <= 256:
                        return refuse("BEZIER_POLE_BUDGET")
                    poles = [[float(arc.Pole(i).X()*radius), float(arc.Pole(i).Y())]
                             for i in range(1, arc.NbPoles()+1)]
                    weights = [float(arc.Weight(i)) for i in range(1, arc.NbPoles()+1)]
                    spans.append(_span(poles, weights, edge_id, copy.deepcopy(provenance)))
            else:
                return refuse("PCURVE_REPRESENTATION_UNSUPPORTED")
            if orientation == TopAbs_REVERSED:
                spans = [{**span, "poles": list(reversed(span["poles"])),
                          "weights": list(reversed(span["weights"]))} for span in reversed(spans)]
            first_vertex, last_vertex = TopExp.FirstVertex_s(edge, True), TopExp.LastVertex_s(edge, True)
            if first_vertex.IsNull() or last_vertex.IsNull():
                return refuse("FINITE_NATIVE_ENDPOINT_VERTICES_REQUIRED")
            spans[0]["native_start_vertex_id"] = vertices.FindIndex(first_vertex)
            spans[-1]["native_end_vertex_id"] = vertices.FindIndex(last_vertex)
            source_tolerance = max(source_tolerance, float(BRep_Tool.Tolerance_s(edge)),
                float(BRep_Tool.Tolerance_s(first_vertex)), float(BRep_Tool.Tolerance_s(last_vertex)))
            ordered.extend(spans)
            if len(ordered) > 256:
                return refuse("COMPLETE_BOUNDARY_SPAN_BUDGET")
            traversal.Next()
        if Counter(expected) != Counter(observed):
            return refuse("ORDERED_NATIVE_WIRE_TRAVERSAL_INCOMPLETE")
        wires.append(ordered)
        explorer.Next()
    proof = certify_graph_strip(wires, native_brep_valid=True, exact_active_domains=True)
    if proof["status"] != "PROVED" and wires and 0. < source_tolerance <= 2.5e-6 and radius > 4*source_tolerance:
        joins = []
        for i, span in enumerate(wires[0]):
            next_span = wires[0][(i+1)%len(wires[0])]
            joins.append(None if span["native_edge_id"] == next_span["native_edge_id"] else
                {"shared_vertex_id":span.get("native_end_vertex_id"),
                 "source_tolerance_mm":source_tolerance,
                 "chart_gap_budget_mm":4*source_tolerance,
                 "bound_rule":"TWO_ENDPOINT_SOURCE_BALLS_AND_CHORD_TO_INTRINSIC_FACTOR_TWO"})
        strict_refusal = proof
        proof = certify_single_edge_end_join_core(wires, native_brep_valid=True,
            exact_active_domains=True, join_receipts=joins)
        proof.update(strict_refusal=strict_refusal, native_join_receipts=joins)
    return {**receipt, **proof, "native_edge_occurrences": observed if wires else [],
            "native_face_orientation": str(face.Orientation())}


def _qualify_new_pair(axis, rows):
    """Opposite native material normals and enclosing cap identity are required."""
    ids = axis["face_indices"]
    patches = [rows[i] for i in ids]
    radii = sorted({p["radius_mm"] for p in patches})
    if len(radii) != 2:
        return False, "EXACTLY_TWO_RADII_REQUIRED"
    for patch in patches:
        expected = "TopAbs_Orientation.TopAbs_REVERSED" if patch["radius_mm"] == radii[0] else "TopAbs_Orientation.TopAbs_FORWARD"
        if patch.get("native_face_orientation") != expected:
            return False, "OPPOSITE_NATIVE_MATERIAL_NORMALS_REQUIRED"
    # Both skins' end boundaries must meet the same owned native cap faces.
    # Cap shape can be curved; this is finite topology correspondence, not a
    # statement that the entire radial solid has calibrated constant thickness.
    left = set().union(*(set(p.get("native_cap_neighbor_ids", [])) for p in patches if p["radius_mm"] == radii[0])) - set(ids)
    right = set().union(*(set(p.get("native_cap_neighbor_ids", [])) for p in patches if p["radius_mm"] == radii[1])) - set(ids)
    if len(left & right) < 2:
        return False, "TWO_SHARED_NATIVE_END_CAPS_REQUIRED"
    midpoint = (np.asarray(axis["canonical_line_start"])+np.asarray(axis["canonical_line_end"]))/2
    for patch in patches:
        if "graph_native_frame" not in patch:
            continue
        frame = patch["graph_native_frame"]
        delta = midpoint-np.asarray(frame["origin"])
        theta = math.atan2(float(np.dot(delta,frame["y"])),float(np.dot(delta,frame["x"])))
        low, high = patch["graph_core_angular_native_rad"]
        theta += 2*math.pi*math.floor((low-theta)/(2*math.pi))
        if theta < low:
            theta += 2*math.pi
        if not low < theta < high:
            return False, "MARKER_OUTSIDE_CERTIFIED_ANGULAR_CORE"
    return True, None


def extract_step_bend_axes(path: Path):
    original = legacy.extract_step_bend_axes(path)
    if not original["partial_cylindrical_faces"]:
        return {**original, "schema_version": 8, "extension_status": "NO_ELIGIBLE_NATIVE_ROWS"}
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise ValueError("SUPPLEMENTAL_NATIVE_IMPORT_FAILED")
    shape = reader.OneShape()
    face_map, edge_map = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_map)
    expected_faces = len(original["partial_cylindrical_faces"])+len(original["surface_diagnostics"])
    if face_map.Extent() != expected_faces or face_map.Extent() > legacy.MAX_NATIVE_FACES or edge_map.Extent() > legacy.MAX_NATIVE_EDGES:
        raise ValueError("SECOND_IMPORT_NATIVE_CENSUS_DISAGREEMENT")
    owned = set()
    solid_explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    if not solid_explorer.More():
        raise ValueError("SECOND_IMPORT_SOLID_MISSING")
    owner_faces = TopExp_Explorer(solid_explorer.Current(), TopAbs_FACE)
    while owner_faces.More():
        owned.add(face_map.FindIndex(owner_faces.Current())); owner_faces.Next()
    incidence = {}
    for index in range(1, face_map.Extent()+1):
        explorer = TopExp_Explorer(face_map.FindKey(index), TopAbs_EDGE)
        while explorer.More():
            incidence.setdefault(edge_map.FindIndex(explorer.Current()), set()).add(index)
            explorer.Next()
    rows = copy.deepcopy(original["partial_cylindrical_faces"])
    by_id = {row["face_index"]: row for row in rows}
    changed = set()
    for row in rows:
        face = TopoDS.Face_s(face_map.FindKey(row["face_index"]))
        surface = BRepAdaptor_Surface(face)
        native_edges = set()
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            native_edges.add(edge_map.FindIndex(edge_explorer.Current())); edge_explorer.Next()
        if native_edges != set(row["edge_ids"]):
            raise ValueError("SECOND_IMPORT_NATIVE_EDGE_IDENTITY_DISAGREEMENT")
        from OCP.GeomAbs import GeomAbs_Cylinder
        if surface.GetType() != GeomAbs_Cylinder or float(surface.Cylinder().Radius()) != row["radius_mm"]:
            raise ValueError("SECOND_IMPORT_NATIVE_CARRIER_DISAGREEMENT")
        cyl = surface.Cylinder()
        position, direction = cyl.Location(), cyl.Axis().Direction()
        axis = np.asarray(row["angular_frame"]["axis"])
        origin = np.array([position.X(),position.Y(),position.Z()])
        native = np.array([direction.X(),direction.Y(),direction.Z()])
        delta = origin-np.asarray(row["line_start"])
        if np.linalg.norm(np.cross(native,axis)) > legacy.CARRIER_DIRECTION_TOL or np.linalg.norm(delta-axis*np.dot(delta,axis)) > legacy.CARRIER_POSITION_TOL_MM:
            raise ValueError("SECOND_IMPORT_NATIVE_AXIS_DISAGREEMENT")
        row["native_face_orientation"] = str(face.Orientation())
        # Preserve the old accepted domain and all its original marker bounds.
        if row["domain_certified"]:
            cap_edges = set(row["edge_ids"]) - {r["edge_id"] for r in row["tangent_rails"]}
        else:
            try:
                proof = native_graph_certificate(face, surface, edge_map)
            except Exception as exc:
                proof = {"status": "REFUSED", "reason": "GRAPH_NATIVE_EXCEPTION:"+type(exc).__name__+":"+str(exc)[:160]}
            row["graph_strip_certificate"] = proof
            if proof["status"] != "PROVED":
                continue
            x0, x1, low, high = proof["safe_rectangular_interior_core_mm"]
            radius = float(surface.Cylinder().Radius())
            rail_x = [s["poles"][0][0] for s in proof["ordered_original_boundary"] if all(p[0] == s["poles"][0][0] for p in s["poles"])]
            if abs(min(rail_x)/radius-surface.FirstUParameter()) > legacy.DOMAIN_EPS or abs(max(rail_x)/radius-surface.LastUParameter()) > legacy.DOMAIN_EPS:
                proof.update(status="REFUSED", reason="ANGULAR_BOUNDARY_ADAPTOR_DISAGREEMENT")
                continue
            cyl = surface.Cylinder()
            axis = np.asarray(row["angular_frame"]["axis"])
            position = cyl.Location(); direction = cyl.Axis().Direction()
            origin = np.array([position.X(), position.Y(), position.Z()])
            native = np.array([direction.X(), direction.Y(), direction.Z()])
            core = sorted(float(np.dot(origin+native*v, axis)) for v in (low, high))
            row.update(domain_certified=True, axial_interval=core,
                       native_domain_certificate=proof,
                       graph_core_angular_native_rad=[x0/radius,x1/radius],
                       graph_native_frame={"origin":origin.tolist(),
                           "x":[float(cyl.Position().XDirection().X()),float(cyl.Position().XDirection().Y()),float(cyl.Position().XDirection().Z())],
                           "y":[float(cyl.Position().YDirection().X()),float(cyl.Position().YDirection().Y()),float(cyl.Position().YDirection().Z())]})
            cap_edges = set(row["edge_ids"]) - set(proof["isoangular_rail_native_edge_ids"])
            changed.add(row["face_index"])
        row["native_cap_neighbor_ids"] = sorted((set().union(*(incidence.get(e, set()) for e in cap_edges)) & owned) - {row["face_index"]})
    if not changed:
        return {**original, "schema_version": 8, "partial_cylindrical_faces": rows,
                "extension_status": "ALL_GRAPH_DOMAINS_REFUSED"}
    result = legacy.build_features(rows)
    accepted, rejected = copy.deepcopy(original["bend_axes"]), result["rejected_groups"]
    incumbent_faces = set().union(*(set(a["face_indices"]) for a in original["bend_axes"])) if original["bend_axes"] else set()
    for axis in result["bend_axes"]:
        if changed.intersection(axis["face_indices"]):
            if incumbent_faces.intersection(axis["face_indices"]):
                rejected.append({"face_indices":axis["face_indices"],"rejection_reason":"SUPPLEMENTAL_FEATURE_OVERLAPS_RETAINED_INCUMBENT"})
                continue
            qualified, reason = _qualify_new_pair(axis, by_id)
            if not qualified:
                rejected.append({"face_indices": axis["face_indices"], "rejection_reason": reason})
                continue
            axis["domain_receipt"].update(kind="CERTIFIED_FINITE_GRAPH_STRIP_CORE_DOMAIN",
                certificate_schema_version=8, detection_only=True, unfold_certificate=False,
                marker_extent_semantics="CERTIFIED_INTERIOR_CORE_NOT_COMPLETE_BEND_EXTENT",
                native_cap_correspondence_proven=True, opposite_material_normals_proven=True)
            accepted.append(axis)
    result["bend_axes"] = accepted
    return {**original, **result, "schema_version": 8,
            "legacy_rejected_groups":copy.deepcopy(original["rejected_groups"]),
            "extension_status": "GRAPH_STRIP_DOMAIN_EXTENSION",
            "contract": {**original["contract"], "graph_strip_policy": "CERTIFIED_CONNECTED_CURVED_END_STRIP_DETECTION_ONLY",
                         "complete_bend_extent_metrology": "UNSCORED",
                         "nonrectangular_unfolding": "UNSUPPORTED"}}
