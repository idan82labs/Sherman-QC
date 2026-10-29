"""Exclusive five-source diagnostic successor; no support or manufacturing score.

Reuses the previously audited whole-case parent process monitor, reservation,
nonce claim and archive code. Native work runs in the claimed case process under
the same durable admission lock. A killed worker leaves RUNNING and halts the
campaign. No physical bytes are read by --freeze or import.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evaluate_vendor_cad_upload_unfold_v1 as transport

PROFILE = "VENDOR_UNFOLD_GEOMETRY_DIAGNOSTIC_V1"
PRE_PROFILE = "VENDOR_UNFOLD_GEOMETRY_DIAGNOSTIC_PRE_V1"
PROTOCOL = ROOT / "docs/operations/vendor-unfold-geometry-successor-2026-09-30.md"


def configure_transport():
    # A separate interpreter owns this adapter. Incumbent source is unchanged.
    transport.PROFILE = PROFILE
    transport.PRE_PROFILE = PRE_PROFILE
    transport.PROTOCOL = PROTOCOL
    transport.__file__ = str(Path(__file__).resolve())
    transport.OWNED_SOURCE_FILES = tuple(dict.fromkeys((*transport.OWNED_SOURCE_FILES,
        "scripts/diagnose_vendor_unfold_geometry_v1.py",
        "docs/operations/vendor-unfold-geometry-successor-2026-09-30.md")))
    transport._actual_upload_result = diagnose_case


def freeze(root):
    """Exclusive source-only freeze with diagnostic-specific declarations."""
    root.mkdir(mode=0o700,parents=False,exist_ok=False)
    pins=transport._source_pins()
    archive=transport._archive_sources(root,pins)
    frozen={"profile":PROFILE,"state":"FROZEN_AWAITING_INDEPENDENT_PRE",
        "protocol_sha256":transport._digest(PROTOCOL.read_bytes()),
        "runner_sha256":transport._digest(Path(__file__).read_bytes()),
        "source_pins":pins,"source_pins_sha256":transport._digest(transport._json(pins)),
        **archive,"case_sources":transport._bindings(),"cad_root":str(transport.CAD_ROOT),
        "route_method":"CLAIMED_PRIVATE_NATIVE_DIAGNOSTIC_WITH_SHARED_ADMISSION",
        "runtime":"DIAGNOSTIC_ONLY_INCUMBENT_GEOMETRY_V4",
        "candidate_origin":"PAIRED_NATIVE_CYLINDERS_NO_FIXTURE_OR_GOLD",
        "case_order":[part for part,_,_ in transport.CASE_SOURCES],
        "limits":{"whole_case_wall_seconds":transport.WALL_SECONDS,
            "whole_case_process_tree_rss_kib":transport.RSS_KIB,
            "campaign_seconds":transport.CAMPAIGN_SECONDS,
            "preparation_cleanup_allowance_seconds":transport.PREPARATION_SECONDS,
            "source_bytes":transport.MAX_CAD_BYTES,"result_bytes":transport.MAX_RESULT_BYTES,
            "log_bytes":transport.MAX_LOG_BYTES,"cases_serial":True},
        "score_rule":"NO_SUPPORT_SCORE_ALL_FIVE_DIAGNOSTIC_COMPLETION_LEDGER",
        "visual_rule":"NO_VISUAL_PROMOTION_FROM_DIAGNOSTIC",
        "manufacturing_ready":False,"units":"assumed_mm","independent_truth":None}
    transport._write_result(root/"FROZEN.json",frozen)
    for index,(part,_,expected) in enumerate(transport.CASE_SOURCES):
        case=root/f"case-{index+1:02d}-{part}"
        case.mkdir(mode=0o700)
        transport._write_result(case/"RESERVED.json",{"part":part,"expected_sha256":expected,
            "state":"NOT_EXECUTED","case_index":index,"profile":PROFILE})
    return frozen


def _plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


def _native_surface_ledger(path, rows=None):
    """Bounded coefficient/ownership ledger before the strict importer runs."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GeomAbs import GeomAbs_BSplineSurface
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from backend import vendor_unfold_preview as geometry

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise ValueError("STEP_READ_UNAVAILABLE")
    shape = reader.OneShape()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    if not solids.More():
        raise ValueError("SINGLE_VALID_SOLID_REQUIRED")
    solid = TopoDS.Solid_s(solids.Current())
    solids.Next()
    if solids.More() or not BRepCheck_Analyzer(solid).IsValid():
        raise ValueError("SINGLE_VALID_SOLID_REQUIRED")
    rows = [] if rows is None else rows
    total_poles = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        if len(rows) >= geometry.MAX_FACES:
            raise ValueError("FACE_BUDGET_EXCEEDED")
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        row = {"source_face_index": len(rows), "surface_type": str(surface.GetType()),
            "native_orientation": str(face.Orientation()),
            "u_interval": [surface.FirstUParameter(), surface.LastUParameter()],
            "v_interval": [surface.FirstVParameter(), surface.LastVParameter()]}
        try:
            row["surface_proof"] = _plain(geometry._surface_proof(surface))
            row["surface_proof_status"] = "PROVED"
        except geometry.UnfoldUnavailable as exc:
            row.update(surface_proof_status="UNAVAILABLE", reason=str(exc))
        if surface.GetType() == GeomAbs_BSplineSurface:
            spline = surface.BSpline()
            nu, nv = spline.NbUPoles(), spline.NbVPoles()
            total_poles += nu * nv
            if total_poles > geometry.MAX_TOTAL_CURVE_POLES:
                raise ValueError("NATIVE_CURVE_COMPLEXITY_BUDGET_EXCEEDED")
            row["spline"] = {"u_degree": spline.UDegree(), "v_degree": spline.VDegree(),
                "u_periodic": spline.IsUPeriodic(), "v_periodic": spline.IsVPeriodic(),
                "u_knots": [spline.UKnot(i) for i in range(1, spline.NbUKnots()+1)],
                "v_knots": [spline.VKnot(i) for i in range(1, spline.NbVKnots()+1)],
                "u_multiplicities": [spline.UMultiplicity(i) for i in range(1, spline.NbUKnots()+1)],
                "v_multiplicities": [spline.VMultiplicity(i) for i in range(1, spline.NbVKnots()+1)],
                "poles": [[[spline.Pole(i,j).X(), spline.Pole(i,j).Y(), spline.Pole(i,j).Z()]
                    for j in range(1,nv+1)] for i in range(1,nu+1)],
                "weights": [[spline.Weight(i,j) for j in range(1,nv+1)] for i in range(1,nu+1)]}
        rows.append(row)
        explorer.Next()
    return rows


def _components(adjacency):
    seen, output = set(), []
    for start in sorted(adjacency):
        if start in seen:
            continue
        nodes, links, stack = set(), {}, [start]
        while stack:
            index = stack.pop()
            if index in nodes:
                continue
            nodes.add(index)
            for link in adjacency[index]:
                links[link.cylinder] = link
                stack.extend(link.planes)
        seen.update(nodes)
        output.append({"planes": sorted(nodes), "cylinders": sorted(links),
            "features": sorted(link.feature_id for link in links.values()),
            "tree": len(links) == len(nodes)-1})
    return output


def _diagnose_native(path, expected):
    import numpy as np
    from backend import vendor_unfold_preview as geometry
    from backend import vendor_unfold_preview_v4 as candidate
    from backend.vendor_native_step_upload_v1 import _axis_rows
    report = {"source_sha256": expected, "surface_ledger": [],
        "units": "assumed_mm", "units_verified": False, "manufacturing_ready": False,
        "purpose": "FAILURE_MECHANISM_DIAGNOSIS_ONLY"}
    try:
        _native_surface_ledger(path,report["surface_ledger"])
    except (geometry.UnfoldUnavailable,ValueError,RuntimeError,OSError) as exc:
        report.update(import_status="UNAVAILABLE",import_reason=str(exc),
            failed_stage="SURFACE_LEDGER",error_type=type(exc).__name__)
        return report
    try:
        faces, axes = geometry._read_native(path)
    except (geometry.UnfoldUnavailable,ValueError,RuntimeError,OSError) as exc:
        report.update(import_status="UNAVAILABLE", import_reason=str(exc),
            failed_stage="STRICT_NATIVE_IMPORT",error_type=type(exc).__name__)
        return report
    report.update(import_status="AVAILABLE", native_face_count=len(faces),
        native_faces=[_plain(face) for face in faces], axes=_plain(axes))
    try:
        candidates = _axis_rows(axes, 1.)
    except ValueError as exc:
        report.update(candidate_status="UNAVAILABLE",candidate_reason=str(exc),
            flat_status="NOT_RUN_CANDIDATE_REFUSAL")
        return report
    report["candidate_status"]="AVAILABLE"
    report["candidates"] = candidates
    # Record native-shared-edge coplanar neighbors, never proximity stitching.
    owners = {}
    for face in faces:
        for edge in face.edges:
            owners.setdefault(edge, []).append(face.index)
    pairs = []
    for edge, ids in sorted(owners.items()):
        if len(ids) != 2:
            continue
        a,b = (faces[i] for i in ids)
        if a.kind == b.kind == "plane":
            normals_parallel = abs(float(np.dot(geometry._unit(a.normal),geometry._unit(b.normal))))
            plane_distance = abs(float(np.dot(a.origin-b.origin,geometry._unit(a.normal))))
            pairs.append({"edge":edge,"faces":ids,"normal_abs_dot":normals_parallel,
                "plane_distance":plane_distance,
                "coplanar_at_declared_tolerance": normals_parallel>1.-geometry.GEOM_TOL
                    and plane_distance<=geometry.GEOM_TOL})
    report["planar_shared_edge_pairs"] = pairs
    try:
        links, adjacency = geometry._skin_graph(faces, axes, candidates)
        report["skin_components"] = _components(adjacency)
        report["bend_links"] = [_plain(link) for link in links]
    except geometry.UnfoldUnavailable as exc:
        report["skin_graph_reason"] = str(exc)
    side_attempts = []
    def record_side(face, cylinder, seam, tangent, side, budget):
        row={"face":face.index,"cylinder":cylinder.index,"side":side,
            "seam":_plain(seam),"signed_tangent":_plain(tangent),
            "axial_span":[cylinder.v0,cylinder.v1]}
        side_attempts.append(row)
        try:
            proof=candidate._side_certificate(face,cylinder,seam,tangent,side,budget)
            row["proof"]=_plain(proof)
            return proof
        except geometry.UnfoldUnavailable as exc:
            row["reason"]=str(exc)
            raise
    namespace=dict(candidate._develop_faces.__globals__)
    namespace["_side_certificate"]=record_side
    develop=FunctionType(candidate._develop_faces.__code__,namespace,
        candidate._develop_faces.__name__,candidate._develop_faces.__defaults__)
    try:
        flat=develop(faces,axes,candidates,expected)
        report["flat_preview"]=flat
        report["flat_status"]="AVAILABLE"
    except geometry.UnfoldUnavailable as exc:
        report.update(flat_status="UNAVAILABLE",flat_reason=str(exc))
    report["finite_side_attempts"]=side_attempts
    return report


def diagnose_case(case_dir, source_raw, part, expected_sha):
    from backend import vendor_unfold_worker_v3 as shared
    lock_path=Path(shared.ADMISSION_LOCK_PATH)
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"r+b") as lock:
        info=os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():
            raise ValueError("ADMISSION_JOURNAL_OWNERSHIP_UNVERIFIED")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=lock.read(1025)
        if raw and json.loads(raw)!={"state":"QUIESCENT","pgid":None}:
            raise ValueError("ADMISSION_NOT_QUIESCENT")
        shared._record(lock,{"state":"RUNNING","pgid":os.getpgrp()})
        try:
            report=_diagnose_native(case_dir/"UPLOAD_SOURCE.step",expected_sha)
            report.update(profile=PROFILE,part=part)
            transport._write_result(case_dir/"GEOMETRY_DIAGNOSTIC.json",report)
        finally:
            shared._record(lock,{"state":"QUIESCENT","pgid":None})
    return {"status":"AVAILABLE","reason":None,"diagnostic_completed":True,
        "preview_claim":False,"report_sha256":transport._digest(transport._json(report)),
        "import_status":report["import_status"],"flat_status":report.get("flat_status")}


def main():
    configure_transport()
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--case-worker",nargs=3,metavar=("ROOT","INDEX","NONCE"))
    parser.add_argument("--attempt-root",type=Path)
    args=parser.parse_args()
    if args.case_worker:
        root,index,nonce=args.case_worker
        transport._case_worker(Path(root),int(index),nonce)
        return
    if args.attempt_root is None:
        parser.error("--attempt-root is required")
    if args.freeze:
        freeze(args.attempt_root)
        print(json.dumps({"state":"FROZEN_AWAITING_INDEPENDENT_PRE","profile":PROFILE}))
    else:
        rows=transport.execute(args.attempt_root)
        print(json.dumps({"profile":PROFILE,"diagnostic_completed_count":sum(
            row.get("child_outcome",{}).get("diagnostic_completed") is True for row in rows),
            "terminal_rows":len(rows),"support_claim":False}))


if __name__=="__main__":
    main()
