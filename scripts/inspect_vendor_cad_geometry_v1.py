"""Bounded native refusal diagnostics, called only by a supervised snapshot job.

API only: no raw-input CLI, file discovery, tessellation or inference promotion.
Diagnostic success does not override an unfold refusal. No native work at import.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np

from backend import vendor_unfold_preview as native
from backend.vendor_surface_normalization_v2 import surface_proof_v2
from backend.vendor_unfold_native_v6 import indexed_native_identity


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _xyz(value):
    return [_number(value.X()), _number(value.Y()), _number(value.Z())]


def trim_knot_diagnostics(trim, knots) -> dict:
    """Report signed interval discrepancies; never clamp or retry a proof."""
    a, b = map(float, trim); lo, hi = map(float, knots)
    finite = all(math.isfinite(x) for x in (a, b, lo, hi))
    return {"trim": [_number(a), _number(b)], "knot_span": [_number(lo), _number(hi)],
            "finite": finite,
            "lower_overshoot": max(0., lo-a) if finite else None,
            "upper_overshoot": max(0., b-hi) if finite else None,
            "strictly_contained": finite and lo <= a < b <= hi,
            "clamping_performed": False}


def _safe(value):
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return _number(value)
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)): return value
    raise TypeError("unsupported diagnostic payload type")


def describe_surface(surface: Any, face_index: int, *, proof=surface_proof_v2) -> dict:
    """Small per-face summary with proof refusal retained as data."""
    kind = str(surface.GetType())
    uv = [float(surface.FirstUParameter()), float(surface.LastUParameter()),
          float(surface.FirstVParameter()), float(surface.LastVParameter())]
    row = {"face_index": face_index, "native_surface_type": kind,
           "uv_bounds": [_number(x) for x in uv],
           "uv_bounds_finite": all(math.isfinite(x) for x in uv)}
    if "BSplineSurface" in kind:
        spline = surface.BSpline()
        nu, nv = int(spline.NbUPoles()), int(spline.NbVPoles())
        native._require(0 < nu * nv <= native.MAX_SURFACE_POLES,
                        "INSPECTOR_SURFACE_POLE_BUDGET_EXCEEDED")
        uk = [float(spline.UKnot(1)), float(spline.UKnot(spline.NbUKnots()))]
        vk = [float(spline.VKnot(1)), float(spline.VKnot(spline.NbVKnots()))]
        weights = np.array([[float(spline.Weight(i, j)) for j in range(1, nv+1)]
                            for i in range(1, nu+1)])
        row["spline"] = {"pole_counts": [nu, nv],
                         "degrees": [int(spline.UDegree()), int(spline.VDegree())],
                         "periodic": [bool(spline.IsUPeriodic()), bool(spline.IsVPeriodic())],
                         "u_interval": trim_knot_diagnostics(uv[:2], uk),
                         "v_interval": trim_knot_diagnostics(uv[2:], vk),
                         "positive_finite_weights": bool(np.isfinite(weights).all() and np.all(weights > 0)),
                         "weights_identical_across_u": bool(np.array_equal(weights, np.repeat(weights[:1], nu, axis=0))),
                         "weights_identical_across_v": bool(np.array_equal(weights, np.repeat(weights[:, :1], nv, axis=1)))}
    elif "Cylinder" in kind:
        cylinder = surface.Cylinder()
        row["cylinder"] = {"radius": _number(cylinder.Radius()),
                           "axis": _xyz(cylinder.Axis().Direction()),
                           "origin": _xyz(cylinder.Axis().Location())}
    elif "Plane" in kind:
        plane = surface.Plane()
        row["plane"] = {"origin": _xyz(plane.Location()), "normal": _xyz(plane.Axis().Direction())}
    elif "SurfaceOfExtrusion" in kind:
        row["extrusion_direction"] = _xyz(surface.Direction())
    try:
        descriptor = proof(surface)
        row["surface_certificate"] = _safe({k: v for k, v in descriptor.items()
                                             if k != "extrusion_curve"})
        row["surface_proof_status"] = "PROVED_CARRIER_ONLY"
    except native.UnfoldUnavailable as exc:
        row["surface_proof_status"] = "REFUSED"
        row["surface_proof_reason"] = str(exc)
    return row


def inspect_geometry(path: Path) -> dict:
    """Inspect an already private, hash-bound snapshot under root's supervisor.

    The caller establishes byte hash binding, one-shot authorization, time/RSS
    budgets and output custody. This function never opens another source path.
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer

    reader = STEPControl_Reader()
    native._require(reader.ReadFile(str(path)) == IFSelect_RetDone and reader.TransferRoots() > 0,
                    "INSPECTOR_STEP_READ_UNAVAILABLE")
    shape = reader.OneShape()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    native._require(solids.More(), "INSPECTOR_SINGLE_SOLID_REQUIRED")
    solid = TopoDS.Solid_s(solids.Current()); solids.Next()
    native._require(not solids.More() and BRepCheck_Analyzer(solid).IsValid(),
                    "INSPECTOR_SINGLE_VALID_SOLID_REQUIRED")
    solid_faces = TopTools_IndexedMapOfShape()
    explorer = TopExp_Explorer(solid, TopAbs_FACE); count = 0
    while explorer.More():
        count += 1
        native._require(count <= native.MAX_FACES, "INSPECTOR_FACE_BUDGET_EXCEEDED")
        solid_faces.Add(explorer.Current()); explorer.Next()
    rows, edge_owners, indexed, edge_count, uses, pole_cost = [], {}, TopTools_IndexedMapOfShape(), 0, 0, 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    seen_faces = TopTools_IndexedMapOfShape()
    while explorer.More():
        native._require(len(rows) < native.MAX_FACES, "INSPECTOR_FACE_BUDGET_EXCEEDED")
        face = TopoDS.Face_s(explorer.Current())
        native._require(seen_faces.Add(face) == len(rows)+1 and solid_faces.Contains(face),
                        "INSPECTOR_FACE_SOLID_OWNERSHIP_MISMATCH")
        row = describe_surface(BRepAdaptor_Surface(face), len(rows))
        pole_cost += row.get("surface_certificate", {}).get("pole_cost", 0)
        if "spline" in row and row["surface_proof_status"] == "REFUSED":
            nu, nv = row["spline"]["pole_counts"]; pole_cost += nu * nv
        native._require(pole_cost <= native.MAX_TOTAL_CURVE_POLES,
                        "INSPECTOR_TOTAL_POLE_BUDGET_EXCEEDED")
        row["native_orientation"] = str(face.Orientation())
        row["solid_owner_ids"] = [1]
        row["edge_uses"] = []
        edges = TopExp_Explorer(face, TopAbs_EDGE)
        while edges.More():
            uses += 1
            native._require(uses <= native.MAX_EDGE_USES, "INSPECTOR_EDGE_USE_BUDGET_EXCEEDED")
            edge = TopoDS.Edge_s(edges.Current())
            edge_id, is_new = indexed_native_identity(indexed, edge, edge_count)
            edge_count += int(is_new)
            row["edge_uses"].append({"edge_index": edge_id, "orientation": str(edge.Orientation())})
            edge_owners.setdefault(edge_id, set()).add(len(rows))
            edges.Next()
        rows.append(row); explorer.Next()
    native._require(len(rows) == count == solid_faces.Extent(), "INSPECTOR_FACE_SOLID_OWNERSHIP_MISMATCH")
    for row in rows:
        ids = {use["edge_index"] for use in row["edge_uses"]}
        row["adjacent_face_indices"] = sorted(set().union(*(edge_owners[i] for i in ids)) - {row["face_index"]})
        row["edge_owner_faces"] = {str(i): sorted(edge_owners[i]) for i in sorted(ids)}
    return _safe({"schema": "VENDOR_NATIVE_GEOMETRY_INSPECTOR_V1",
                  "face_index_space": "ZERO_BASED_NATIVE_EXPLORER",
                  "face_count": len(rows), "edge_count": edge_count, "edge_use_count": uses,
                  "surface_pole_cost": pole_cost, "meshing_performed": False,
                  "inference_or_refusal_override_performed": False,
                  "faces": rows})
