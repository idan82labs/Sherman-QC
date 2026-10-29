"""Generic native chart normalization plus coherent visual flat winding.

V4 source is unchanged. Exact spline extrusion proofs run before tessellation;
native-edge coplanar charts run before the skin graph. The V4 closure, material,
finite-width and whole-mesh overlap gates remain mandatory. All output triangle
geometry and source rows remain attributable. This is a selected-skin preview.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import tempfile
from types import FunctionType, SimpleNamespace

import numpy as np

from backend import vendor_unfold_preview as native
from backend import vendor_unfold_preview_v4 as v4
from backend.vendor_unfold_native_v5 import read_native_v5
from backend.vendor_unfold_coplanar_stitch_v1 import stitch_coplanar_native_faces
from backend.vendor_unfold_local_strip_guard_v1 import (
    LocalStripBudget, LocalStripUnavailable, certify_local_finite_strip, MODEL as STRIP_MODEL)

MECHANISM_VERSION=5
UnfoldUnavailable=native.UnfoldUnavailable


def _envelope(source_hash,reason=None):
    payload=v4._envelope(source_hash,reason)
    payload["diagnostics"]["mechanism_version"]=MECHANISM_VERSION
    payload["diagnostics"].update(finite_width_model=STRIP_MODEL,
        local_strip_source_sha256="7268fede9a8817b86a18f749d6fb682088a709a5aae4fc0c31b6c00066c14fad",
        v4_component_source_sha256="01ff5236499273d28e36a75ebeb6e34c8172de2a0ee1d6b99cfb416de27d59dc")
    payload["diagnostics"]["limitations"].extend([
        "EXACT_POSITIVE_TENSOR_EXTRUSION_PROOF_ONLY",
        "NATIVE_EDGE_CONNECTED_COPLANAR_CHARTS_ONLY",
        "EXPORTED_FLAT_TRIANGLE_WINDING_POSITIVE_Z",
        "DEVELOPED_AREA_ORACLE_REQUIRED_BEFORE_GEOMETRY_PROMOTION",
        "LOCAL_FINITE_STRIP_GUARD_IS_NATIVE_TESSELLATION_ONLY",
        "LOCAL_STRIP_WIDTH_IS_SELECTED_NATIVE_RADIUS_TIMES_SWEEP",
        "LOCAL_NATIVE_SEAM_BOUNDARY_AND_MATERIAL_SIDE_REQUIRED"])
    payload["diagnostics"]["limitations"].remove("COPLANAR_SKIN_FACE_SPLITS_UNSUPPORTED")
    return payload


def _orient_flat(vertices,triangles):
    p=np.asarray(vertices,dtype=float).reshape(-1,3)
    t=np.asarray(triangles,dtype=np.int64).reshape(-1,3).copy()
    q=p[t]
    twice=(q[:,1,0]-q[:,0,0])*(q[:,2,1]-q[:,0,1])-(q[:,1,1]-q[:,0,1])*(q[:,2,0]-q[:,0,0])
    native._require(np.isfinite(twice).all() and np.all(np.abs(twice)>1e-12),
                    "DEGENERATE_DEVELOPED_TRIANGLE")
    negative=twice<0
    t[negative]=t[negative][:,[0,2,1]]
    return t.reshape(-1).tolist(),int(negative.sum())


def develop_faces_v5(source_faces,axes,candidates,source_hash):
    stitched=stitch_coplanar_native_faces(source_faces,axes)
    namespace=dict(v4._develop_faces.__globals__)
    namespace["_envelope"]=_envelope
    namespace["FiniteWidthBudget"]=LocalStripBudget
    # Record the ACTUAL native seam delta used by the unchanged proper chart
    # transform. No caller-adjustable strip width or principal-angle shortcut.
    active={}
    def child_transform(*args):
        result=native._child_transform(*args)
        active.update(cylinder=args[2].index,delta=result[2])
        return result
    legacy_view=SimpleNamespace(**vars(native))
    legacy_view._child_transform=child_transform
    namespace["legacy"]=legacy_view
    def side_certificate(face,cylinder,seam,signed_tangent,side,budget):
        native._require(active.get("cylinder")==cylinder.index,"LOCAL_STRIP_TRANSFORM_BINDING_MISSING")
        sweep=abs(active["delta"])
        try:
            proof=certify_local_finite_strip(face.vertices,face.triangles,
                cylinder_origin=cylinder.origin,cylinder_axis=cylinder.axis,
                signed_tangent=signed_tangent,seam_endpoints=seam,
                axial_min=cylinder.v0,axial_max=cylinder.v1,
                developed_strip_width=cylinder.radius*sweep,side=side,budget=budget)
        except LocalStripUnavailable as exc:
            raise UnfoldUnavailable(f"LOCAL_STRIP_{side.upper()}:{exc}") from exc
        proof.update(radius_mm=cylinder.radius,sweep_radians=sweep)
        return proof
    namespace["_side_certificate"]=side_certificate
    develop=FunctionType(v4._develop_faces.__code__,namespace,v4._develop_faces.__name__,
                         v4._develop_faces.__defaults__)
    payload=develop(stitched.faces,stitched.axes,candidates,source_hash)
    payload["triangles"],flips=_orient_flat(payload["vertices"],payload["triangles"])
    for surface in payload["flange_surfaces"]:
        surface["triangles"],_=_orient_flat(surface["vertices"],surface["triangles"])
    diag=payload["diagnostics"]
    selected=diag["selected_source_face_indices"]
    vertex_offset=triangle_offset=0
    slices=[]
    by_chart={row["chart_face_index"]:row for row in stitched.ledger["charts"]}
    for index in selected:
        chart=stitched.faces[index]
        for member in by_chart[index]["members"]:
            slices.append({"source_face_index":member["source_face_index"],
                "chart_face_index":index,
                "body_vertex_start":vertex_offset+member["vertex_start"],
                "body_vertex_count":member["vertex_count"],
                "body_triangle_start":triangle_offset+member["triangle_start"],
                "body_triangle_count":member["triangle_count"]})
        vertex_offset+=len(chart.vertices)
        triangle_offset+=len(chart.triangles)
    diag.update({"face_index_space":"STITCHED_NATIVE_CHART",
        "original_native_face_count":len(source_faces),
        "selected_chart_face_indices":selected,
        "selected_native_face_indices":sorted(row["source_face_index"] for row in slices),
        "chart_topology":stitched.ledger,"developed_native_face_slices":slices,
        "exported_triangle_winding":"POSITIVE_Z",
        "reoriented_triangle_count":flips,
        "orientation_changes_vertex_geometry":False,
        "independent_geometry_oracle":"REQUIRED_SEPARATE_EVALUATION"})
    return payload


def build_unfold_preview(cad_path,expected_hash,candidates):
    source_hash=None
    try:
        native._require(type(expected_hash) is str and re.fullmatch(r"[0-9a-f]{64}",expected_hash),
                        "INVALID_EXPECTED_CAD_HASH")
        native._require(type(candidates) is list and 0<len(candidates)<=native.MAX_CANDIDATES,
                        "CANDIDATE_BUDGET_OR_SCHEMA_INVALID")
        path=Path(cad_path)
        native._require(path.suffix.lower() in {".stp",".step"} and path.is_file(),"STEP_SOURCE_REQUIRED")
        with path.open("rb") as stream:
            raw=stream.read(native.MAX_CAD_BYTES+1)
        native._require(0<len(raw)<=native.MAX_CAD_BYTES,"CAD_BYTE_BUDGET_EXCEEDED")
        source_hash=hashlib.sha256(raw).hexdigest()
        native._require(source_hash==expected_hash,"CAD_HASH_MISMATCH")
        with tempfile.TemporaryDirectory(prefix="sherman-unfold-v5-") as folder:
            snapshot=Path(folder)/"source.step"
            snapshot.write_bytes(raw)
            from backend.vendor_unfold_oracle_bridge_v1 import NativeOracleCapture,validate_captured_development
            capture=NativeOracleCapture()
            faces,axes=read_native_v5(snapshot,on_face=capture.on_face)
            result=develop_faces_v5(faces,axes,candidates,source_hash)
            oracle=validate_captured_development(result,capture,faces,axes,candidates)
            result["diagnostics"]["geometry_oracle"]=oracle
            result["diagnostics"]["independent_geometry_oracle"]="VERIFIED_DISCRETE_NATIVE_METRIC_CHART"
            return result
    except UnfoldUnavailable as exc:
        return _envelope(source_hash,str(exc))
    except (OSError,ImportError,RuntimeError,TypeError,ValueError,KeyError,IndexError) as exc:
        payload=_envelope(source_hash,"NATIVE_V5_DEVELOPMENT_UNAVAILABLE")
        payload["diagnostics"]["error_type"]=type(exc).__name__
        payload["diagnostics"]["error_reason"]=str(exc)[:200]
        return payload
