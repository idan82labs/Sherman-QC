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
from backend.cad_finite_native_curve_graph_domain_v2 import certify_native_curve_graph_domain


def _span(poles, weights, edge_id, provenance):
    return {"poles": poles, "weights": weights, "native_edge_id": edge_id,
            "actual_interval_proven": True, "active_interval_provenance": provenance}


def native_graph_certificate(face, surface, edge_map):
    """Preserve original native rational curves; no subdivision creates joins."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Line,GeomAbs_BSplineCurve,GeomAbs_BezierCurve
    from OCP.TopAbs import TopAbs_WIRE,TopAbs_EDGE,TopAbs_VERTEX,TopAbs_FORWARD,TopAbs_REVERSED
    from OCP.TopExp import TopExp,TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    base={"status":"REFUSED","type_proven":False,"certificate_schema_version":9,
          "detection_only":True,"unfold_certificate":False}
    def refuse(reason):
        return {**base,"reason":reason}
    radius=float(surface.Cylinder().Radius())
    sweep=surface.LastUParameter()-surface.FirstUParameter()
    if not 0.<sweep<2*math.pi-legacy.DOMAIN_EPS:
        return refuse("SINGLE_TURN_UNWRAPPED_NATIVE_CHART_REQUIRED")
    if not BRepCheck_Analyzer(face).IsValid():
        return refuse("INVALID_NATIVE_FACE")
    wires=TopExp_Explorer(face,TopAbs_WIRE)
    if not wires.More():
        return refuse("NATIVE_OUTER_WIRE_REQUIRED")
    wire=TopoDS.Wire_s(wires.Current());wires.Next()
    if wires.More() or not wire.IsSame(BRepTools.OuterWire_s(face)):
        return refuse("SINGLE_NATIVE_OUTER_WIRE_REQUIRED")
    expected=[];census=TopExp_Explorer(wire,TopAbs_EDGE)
    while census.More():
        expected.append((edge_map.FindIndex(census.Current()),str(census.Current().Orientation())))
        if len(expected)>256:return refuse("NATIVE_WIRE_BUDGET")
        census.Next()
    vertex_map=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(face,TopAbs_VERTEX,vertex_map)
    curves=[];observed=[];total_poles=0
    traversal=BRepTools_WireExplorer(wire,face)
    while traversal.More():
        edge=TopoDS.Edge_s(traversal.Current());eid=edge_map.FindIndex(edge)
        orientation=edge.Orientation();observed.append((eid,str(orientation)))
        if len(observed)>256 or eid<=0 or orientation not in (TopAbs_FORWARD,TopAbs_REVERSED):
            return refuse("NATIVE_OCCURRENCE_IDENTITY_OR_ORIENTATION")
        adaptor=BRepAdaptor_Curve2d(edge,face)
        first,last=float(adaptor.FirstParameter()),float(adaptor.LastParameter())
        if not math.isfinite(first) or not math.isfinite(last) or last<=first:
            return refuse("INVALID_NATIVE_ACTIVE_INTERVAL")
        kind=adaptor.GetType()
        basis={"periodic":False,"actual_interval":[first,last],"restricted_interval":[first,last],
               "actual_interval_proven":True}
        if kind==GeomAbs_Line:
            p,q=adaptor.Value(first),adaptor.Value(last)
            poles=[[float(p.X()*radius),float(p.Y())],[float(q.X()*radius),float(q.Y())]]
            weights=[1.,1.];basis.update(kind="LINE",degree=1,knots=[first,last],multiplicities=[2,2])
        elif kind in (GeomAbs_BSplineCurve,GeomAbs_BezierCurve):
            if kind==GeomAbs_BSplineCurve:
                curve,restriction=legacy.restrict_bspline_actual_interval(adaptor.BSpline(),first,last)
                if curve is None:return refuse(restriction["reason"])
                basis.update(kind="BSPLINE",degree=curve.Degree(),
                    knots=[float(curve.Knot(k)) for k in range(1,curve.NbKnots()+1)],
                    multiplicities=[curve.Multiplicity(k) for k in range(1,curve.NbKnots()+1)],
                    restriction=restriction)
            else:
                if first!=0. or last!=1.:
                    return refuse("EXACT_BEZIER_FULL_PARAMETER_DOMAIN_REQUIRED")
                curve=adaptor.Bezier()
                basis.update(kind="BEZIER",degree=curve.Degree(),knots=[0.,1.],
                    multiplicities=[curve.Degree()+1,curve.Degree()+1])
            if not 2<=curve.NbPoles()<=4096:
                return refuse("NATIVE_CURVE_POLE_BUDGET")
            poles=[[float(curve.Pole(i).X()*radius),float(curve.Pole(i).Y())] for i in range(1,curve.NbPoles()+1)]
            weights=[float(curve.Weight(i)) for i in range(1,curve.NbPoles()+1)]
        else:
            return refuse("NATIVE_CURVE_REPRESENTATION_UNSUPPORTED")
        total_poles+=len(poles)
        if total_poles>4096:return refuse("COMPLETE_BOUNDARY_POLE_BUDGET")
        if orientation==TopAbs_REVERSED:
            poles=list(reversed(poles));weights=list(reversed(weights))
            if basis["kind"]=="BSPLINE":
                basis["knots"]=[first+last-k for k in reversed(basis["knots"])]
                basis["knots"][0],basis["knots"][-1]=first,last
                basis["multiplicities"]=list(reversed(basis["multiplicities"]))
            basis["reversed_for_native_traversal"]=True
        start,end=TopExp.FirstVertex_s(edge,True),TopExp.LastVertex_s(edge,True)
        if start.IsNull() or end.IsNull():
            return refuse("FINITE_NATIVE_VERTEX_IDENTITY_REQUIRED")
        curves.append({"native_edge_id":eid,"native_start_vertex_id":vertex_map.FindIndex(start),
            "native_end_vertex_id":vertex_map.FindIndex(end),"poles":poles,"weights":weights,"basis":basis,
            "kernel_vertex_tolerance_mm":max(float(BRep_Tool.Tolerance_s(start)),float(BRep_Tool.Tolerance_s(end))),
            "kernel_edge_tolerance_mm":float(BRep_Tool.Tolerance_s(edge))})
        traversal.Next()
    if Counter(expected)!=Counter(observed):
        return refuse("ORDERED_NATIVE_WIRE_TRAVERSAL_INCOMPLETE")
    proof=certify_native_curve_graph_domain(curves,face_tolerance_mm=float(BRep_Tool.Tolerance_s(face)),
        native_brep_valid=True,complete_outer_wire=True,chart_branch_proven=True)
    return {**base,**proof,"native_edge_occurrences":observed,"native_face_orientation":str(face.Orientation())}


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
        return {**original, "schema_version": 9, "extension_status": "NO_ELIGIBLE_NATIVE_ROWS"}
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
            rail_x = proof["native_isoangular_rail_centers_metric_mm"]
            if abs(min(rail_x)/radius-surface.FirstUParameter()) > proof["face_tolerance_mm"]/radius or abs(max(rail_x)/radius-surface.LastUParameter()) > proof["face_tolerance_mm"]/radius:
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
        return {**original, "schema_version": 9, "partial_cylindrical_faces": rows,
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
                certificate_schema_version=9, detection_only=True, unfold_certificate=False,
                marker_extent_semantics="CERTIFIED_INTERIOR_CORE_NOT_COMPLETE_BEND_EXTENT",
                native_cap_correspondence_proven=True, opposite_material_normals_proven=True)
            accepted.append(axis)
    result["bend_axes"] = accepted
    return {**original, **result, "schema_version": 9,
            "legacy_rejected_groups":copy.deepcopy(original["rejected_groups"]),
            "extension_status": "GRAPH_STRIP_DOMAIN_EXTENSION",
            "contract": {**original["contract"], "graph_strip_policy": "CERTIFIED_CONNECTED_CURVED_END_STRIP_DETECTION_ONLY",
                         "complete_bend_extent_metrology": "UNSCORED",
                         "nonrectangular_unfolding": "UNSUPPORTED"}}

