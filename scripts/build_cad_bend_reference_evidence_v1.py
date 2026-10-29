"""Neutral, complete indexed BRep census; never imports an extraction candidate.

Native execution requires a separately authorized input/run gate. UV rectangles
are adaptor bounds, not filled trim domains. Samples describe surfaces only;
they are not curvature/tangency proofs or bend labels. IDs are repeatable only
for the same STEP bytes, OCP version and import settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def vector(value):
    return [float(value.X()), float(value.Y()), float(value.Z())]


def finite_json(value):
    """Retain nonfinite evidence explicitly instead of invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def curve_coefficients(curve, dimension=3, pole_budget=200_000):
    """Original carrier coefficients, never an inferred trim or bend."""
    from OCP import GeomAbs
    kind = curve.GetType()
    point = vector if dimension == 3 else lambda p: [float(p.X()), float(p.Y())]
    result = {"curve_class": str(kind)}
    if kind == GeomAbs.GeomAbs_Line:
        line = curve.Line()
        result["line"] = {"origin": point(line.Location()), "direction": point(line.Direction())}
    elif kind == GeomAbs.GeomAbs_Circle:
        circle = curve.Circle()
        result["circle"] = {"center": point(circle.Location()), "radius": float(circle.Radius()),
                            "x_direction": point(circle.XAxis().Direction()),
                            "y_direction": point(circle.YAxis().Direction())}
        if dimension == 3:
            result["circle"]["axis_direction"] = vector(circle.Axis().Direction())
    elif kind == GeomAbs.GeomAbs_BSplineCurve:
        spline = curve.BSpline()
        count = spline.NbPoles()
        result["bspline"] = {"degree": spline.Degree(), "periodic": spline.IsPeriodic(),
                             "original_parameter_domain": [spline.FirstParameter(), spline.LastParameter()],
                             "pole_count": count}
        if count > pole_budget:
            result["coefficient_refusal"] = "pole_budget_exceeded"
        else:
            result["bspline"].update({"poles": [point(spline.Pole(j)) for j in range(1, count + 1)],
                                      "weights": [float(spline.Weight(j)) for j in range(1, count + 1)],
                                      "knots": [float(spline.Knot(j)) for j in range(1, spline.NbKnots() + 1)],
                                      "multiplicities": [spline.Multiplicity(j) for j in range(1, spline.NbKnots() + 1)]})
    else:
        result["coefficient_refusal"] = "unsupported_carrier_class"
    return result


def coverage(faces, edges):
    face_ids = {f["id"] for f in faces}
    edge_ids = {e["id"] for e in edges}
    errors = []
    if len(face_ids) != len(faces) or len(edge_ids) != len(edges):
        errors.append("duplicate_index")
    for face in faces:
        for edge_id in face.get("edge_ids", []):
            if edge_id not in edge_ids:
                errors.append("missing_edge_index")
    for edge in edges:
        if not set(edge["face_ids"]).issubset(face_ids):
            errors.append("missing_face_index")
    return {"face_count": len(faces), "edge_count": len(edges),
            "surface_census": dict(sorted(Counter(f["surface_class"] for f in faces).items())),
            "face_errors": [f["id"] for f in faces if f.get("errors")],
            "edge_errors": [e["id"] for e in edges if e.get("errors")],
            "pcurve_errors": [{"face_id": f["id"], "edge_id": occurrence["edge_id"], "errors": occurrence["errors"]}
                              for f in faces for w in f.get("wires", []) for occurrence in w["edge_occurrences"] if occurrence.get("errors")],
            "pcurve_coefficient_refusals": [{"face_id": f["id"], "edge_id": occurrence["edge_id"], "refusal": occurrence["pcurve"]["coefficient_refusal"]}
                              for f in faces for w in f.get("wires", []) for occurrence in w["edge_occurrences"] if occurrence.get("pcurve", {}).get("coefficient_refusal")],
            "topology_errors": errors,
            "bend_reference_complete": False}


def collect_shape(shape):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve, BRepAdaptor_Curve2d
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_SOLID
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP import GeomAbs
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopLoc import TopLoc_Location

    validity = bool(BRepCheck_Analyzer(shape).IsValid())
    # Display tessellation is approximation only; never used to assign labels.
    mesh_error = None
    try:
        BRepMesh_IncrementalMesh(shape, 0.1, False, 0.3, False).Perform()
    except Exception as exc:
        mesh_error = type(exc).__name__ + ":" + str(exc)

    fm, em = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fm)
    TopExp.MapShapes_s(shape, TopAbs_EDGE, em)
    sm = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, sm)
    owners = {i: [] for i in range(1, fm.Extent() + 1)}
    for sid in range(1, sm.Extent() + 1):
        ex = TopExp_Explorer(sm.FindKey(sid), TopAbs_FACE)
        while ex.More():
            owners[fm.FindIndex(ex.Current())].append(sid)
            ex.Next()
    surface_names = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BezierSurface",
                     "BSplineSurface", "SurfaceOfRevolution", "SurfaceOfExtrusion", "OffsetSurface", "OtherSurface"]
    classes = {getattr(GeomAbs, "GeomAbs_" + name): name for name in surface_names}
    faces, edges = [], []
    remaining_poles = 200_000
    incidence = {i: set() for i in range(1, em.Extent() + 1)}
    for i in range(1, fm.Extent() + 1):
        face = TopoDS.Face_s(fm.FindKey(i))
        row = {"id": i, "orientation": str(face.Orientation()), "surface_class": "Unclassified",
               "edge_ids": [], "solid_ids": sorted(set(owners[i])), "wires": [], "errors": []}
        try:
            ex = TopExp_Explorer(face, TopAbs_EDGE)
            while ex.More():
                eid = em.FindIndex(ex.Current())
                row["edge_ids"].append(eid)
                incidence[eid].add(i)
                ex.Next()
            wx = TopExp_Explorer(face, TopAbs_WIRE)
            while wx.More():
                wire_edges = []
                wire = TopoDS.Wire_s(wx.Current())
                expected = []
                census = TopExp_Explorer(wire, TopAbs_EDGE)
                while census.More():
                    expected.append((em.FindIndex(census.Current()), str(census.Current().Orientation())))
                    census.Next()
                ex = BRepTools_WireExplorer(wire, face)
                while ex.More():
                    occurrence = ex.Current()
                    item = {"edge_id": em.FindIndex(occurrence), "orientation": str(occurrence.Orientation()), "errors": []}
                    try:
                        pcurve = BRepAdaptor_Curve2d(occurrence, face)
                        item["pcurve"] = curve_coefficients(pcurve, 2, remaining_poles)
                        if "poles" in item["pcurve"].get("bspline", {}):
                            remaining_poles -= item["pcurve"]["bspline"]["pole_count"]
                        item["pcurve"]["actual_trim_parameter_interval"] = [float(pcurve.FirstParameter()), float(pcurve.LastParameter())]
                    except Exception as exc:
                        item["errors"].append("pcurve:" + type(exc).__name__ + ":" + str(exc))
                    wire_edges.append(item)
                    ex.Next()
                observed = [(item["edge_id"], item["orientation"]) for item in wire_edges]
                if Counter(expected) != Counter(observed):
                    row["errors"].append("WIRE_TRAVERSAL_INCOMPLETE")
                row["wires"].append({"edge_occurrences": wire_edges, "order_is_boundary_order": True})
                wx.Next()
        except Exception as exc:
            row["errors"].append("topology:" + type(exc).__name__ + ":" + str(exc))
        try:
            surf = BRepAdaptor_Surface(face, True)
            row["surface_class"] = classes.get(surf.GetType(), "Unclassified")
            row["uv_bounds_not_trim_fill"] = list(BRepTools.UVBounds_s(face))
            row["tolerance"] = float(BRep_Tool.Tolerance_s(face))
            if row["surface_class"] == "Cylinder":
                c = surf.Cylinder()
                row["cylinder"] = {"radius": float(c.Radius()), "axis_origin": vector(c.Location()),
                                   "axis_direction": vector(c.Axis().Direction()),
                                   "x_direction": vector(c.Position().XDirection()),
                                   "y_direction": vector(c.Position().YDirection())}
            elif row["surface_class"] == "Plane":
                p = surf.Plane()
                row["plane"] = {"origin": vector(p.Location()), "carrier_normal": vector(p.Axis().Direction()),
                                "normal_orientation_requires_face_orientation": True}
            elif row["surface_class"] == "BSplineSurface":
                s = surf.BSpline()
                nu, nv = s.NbUPoles(), s.NbVPoles()
                row["bspline_surface"] = {"u_degree": s.UDegree(), "v_degree": s.VDegree(),
                    "u_periodic": s.IsUPeriodic(), "v_periodic": s.IsVPeriodic(),
                    "u_pole_count": nu, "v_pole_count": nv, "original_parameter_bounds": list(s.Bounds())}
                if nu * nv > remaining_poles:
                    row["bspline_surface"]["coefficient_refusal"] = "pole_budget_exceeded"
                else:
                    remaining_poles -= nu * nv
                    row["bspline_surface"].update({
                        "poles": [[vector(s.Pole(u, v)) for v in range(1, nv + 1)] for u in range(1, nu + 1)],
                        "weights": [[float(s.Weight(u, v)) for v in range(1, nv + 1)] for u in range(1, nu + 1)],
                        "u_knots": [float(s.UKnot(j)) for j in range(1, s.NbUKnots() + 1)],
                        "v_knots": [float(s.VKnot(j)) for j in range(1, s.NbVKnots() + 1)],
                        "u_multiplicities": [s.UMultiplicity(j) for j in range(1, s.NbUKnots() + 1)],
                        "v_multiplicities": [s.VMultiplicity(j) for j in range(1, s.NbVKnots() + 1)]})
        except Exception as exc:
            row["errors"].append("surface:" + type(exc).__name__ + ":" + str(exc))
        faces.append(row)
        try:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            row["area"] = float(props.Mass())
            row["area_centroid"] = vector(props.CentreOfMass())
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is None:
                raise RuntimeError("missing triangulation")
            row["display_mesh"] = {
                "vertices": [vector(tri.Node(j).Transformed(location.Transformation())) for j in range(1, tri.NbNodes() + 1)],
                "triangles_one_based": [list(tri.Triangle(j).Get()) for j in range(1, tri.NbTriangles() + 1)],
                "approximation_only": True, "linear_deflection": 0.1, "angular_deflection": 0.3}
        except Exception as exc:
            row["errors"].append("display_measure:" + type(exc).__name__ + ":" + str(exc))
    for i in range(1, em.Extent() + 1):
        edge = TopoDS.Edge_s(em.FindKey(i))
        row = {"id": i, "face_ids": sorted(incidence[i]), "errors": []}
        try:
            row["degenerated"] = bool(BRep_Tool.Degenerated_s(edge))
            row["tolerance"] = float(BRep_Tool.Tolerance_s(edge))
            curve = BRepAdaptor_Curve(edge)
            row["curve_class"] = str(curve.GetType())
            row["carrier"] = curve_coefficients(curve, 3, remaining_poles)
            if "poles" in row["carrier"].get("bspline", {}):
                remaining_poles -= row["carrier"]["bspline"]["pole_count"]
            first, last = float(curve.FirstParameter()), float(curve.LastParameter())
            row["parameter_domain"] = [first, last]
            row["endpoint_coordinates"] = [vector(curve.Value(first)), vector(curve.Value(last))]
            if math.isfinite(first) and math.isfinite(last):
                row["display_curve_samples_not_proof"] = [vector(curve.Value(first + (last - first) * j / 16)) for j in range(17)]
        except Exception as exc:
            row["errors"].append("curve:" + type(exc).__name__ + ":" + str(exc))
        edges.append(row)
    return {"faces": faces, "edges": edges, "solid_count": sm.Extent(),
            "brep_valid": validity, "display_meshing_error": mesh_error,
            "coefficient_pole_budget": 200_000, "coefficient_poles_consumed": 200_000 - remaining_poles,
            "index_method": "TopExp.MapShapes_s global indexed native faces/edges, one-based, no reordering",
            "coverage": coverage(faces, edges)}


def collect_evidence(source: Path):
    """Collect authorized STEP; caller owns process budgets and output custody."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    import OCP
    if source.stat().st_size > 5_000_000:
        raise RuntimeError("STEP exceeds fixed 5,000,000-byte input cap")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise RuntimeError("STEP root transfer failed")
    evidence = collect_shape(reader.OneShape())
    evidence.update({"schema": "neutral-brep-evidence-v1", "input_sha256": digest,
                     "ocp_version": getattr(OCP, "__version__", "unknown"),
                     "transferred_roots": transferred, "automatic_bend_labels": None,
                     "units": "native STEP reader coordinates; manufacturing units unverified",
                     "limitations": ["UV bounds alone do not encode filled trim regions", "display mesh is not continuous geometric proof",
                                      "unsupported carriers and coefficient budget refusals require unresolved review",
                                      "seam pcurve branch identity depends on native oriented edge occurrence"]})
    return finite_json(evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; evidence is append-only")
    evidence = collect_evidence(args.step)
    serialized = json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(serialized.encode()) > 40_000_000:
        raise RuntimeError("neutral evidence exceeds fixed 40,000,000-byte output cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(serialized)


if __name__ == "__main__":
    main()
