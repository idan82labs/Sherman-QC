"""Transport5/mechanism6 preview admission with explicit native/chart lineage.

The separately versioned common6 gate preserves incumbent numeric checks;
finite-strip and topology/winding lineage are checked on the original payload.
This reads no native CAD and does not certify physical support or manufacture.
"""
from __future__ import annotations
import re
import numpy as np
from backend.vendor_unfold_contract_v2 import PreviewContractError,canonical_candidates,strict_json
from backend.vendor_unfold_common_contract_v6 import _validate_preview as validate_common

MECHANISM_VERSION=6
TRANSPORT_VERSION=5


def _require(value,reason):
    if not value:
        raise PreviewContractError(reason)


def unavailable(reason,source_hash=None):
    from backend.vendor_unfold_contract_v3 import unavailable as incumbent
    value=incumbent(reason,source_hash)
    value["diagnostics"].update(mechanism_version=6,transport_version=5)
    return value


def validate_preview(payload,expected_hash,candidates):
    _require(type(payload) is dict and type(payload.get("diagnostics")) is dict,"INVALID_V6_ENVELOPE")
    diag=payload["diagnostics"]
    _require(type(diag.get("mechanism_version")) is int and diag["mechanism_version"]==6
        and type(diag.get("transport_version")) is int and diag["transport_version"]==5,
        "INVALID_V5_TRANSPORT_OR_V6_MECHANISM")
    validate_common(payload,expected_hash,candidates,mechanism_version=6)
    if payload["status"]=="UNAVAILABLE":
        return payload
    from backend.vendor_unfold_local_strip_contract_v2 import validate_local_strip_diagnostics
    validate_local_strip_diagnostics(payload,candidates)
    _require(diag.get("face_index_space")=="STITCHED_NATIVE_CHART"
        and diag.get("exported_triangle_winding")=="POSITIVE_Z"
        and diag.get("orientation_changes_vertex_geometry") is False,
        "V6_CHART_OR_WINDING_DECLARATION_MISSING")
    topology=diag.get("chart_topology")
    _require(type(topology) is dict and topology.get("model")=="NATIVE_EDGE_CONNECTED_COPLANAR_FLANGE_SUPERNODES"
        and topology.get("source_vertices_preserved") is True
        and topology.get("source_triangles_preserved") is True
        and topology.get("geometry_fitted") is False,"V6_TOPOLOGY_LINEAGE_MISSING")
    native_count=diag.get("original_native_face_count")
    chart_count=diag.get("native_face_count")
    _require(type(native_count) is int and type(chart_count) is int and 0<chart_count<=native_count<=1024
        and topology.get("native_face_count")==native_count
        and topology.get("chart_face_count")==chart_count,"V6_SOURCE_CHART_COUNT_MISMATCH")
    charts=topology.get("charts")
    _require(type(charts) is list and len(charts)==chart_count,"V6_CHART_RECORD_COUNT_MISMATCH")
    source_ids=[]
    for index,row in enumerate(charts):
        _require(type(row) is dict and row.get("chart_face_index")==index,
            "V6_CHART_ENUMERATION_INVALID")
        ids=row.get("source_face_indices")
        _require(type(ids) is list and ids and all(type(i) is int and 0<=i<native_count for i in ids)
            and ids==sorted(set(ids)),"V6_NATIVE_FACE_OWNERSHIP_INVALID")
        _require(row.get("kind") in {"plane","cylinder","extruded_boundary"}
            and (row["kind"]=="plane" or len(ids)==1),"V6_CHART_KIND_OR_CYLINDER_STITCH_INVALID")
        source_ids.extend(ids)
        members=row.get("members")
        _require(type(members) is list and len(members)==len(ids),"V6_CHART_MEMBER_COUNT_INVALID")
        vc=tc=0
        for member,ident in zip(members,ids):
            _require(type(member) is dict and set(member)=={"source_face_index","vertex_start",
                "vertex_count","triangle_start","triangle_count"}
                and all(type(value) is int for value in member.values())
                and member["source_face_index"]==ident and member["vertex_start"]==vc
                and member["triangle_start"]==tc and member["vertex_count"]>=3
                and member["triangle_count"]>0,"V6_CHART_MEMBER_PARTITION_INVALID")
            vc+=member["vertex_count"]
            tc+=member["triangle_count"]
    _require(sorted(source_ids)==list(range(native_count)),"V6_NATIVE_FACE_BIJECTION_INVALID")
    cylinders=diag["selected_cylinder_face_indices"]
    _require(all(charts[i]["kind"]=="cylinder" and len(charts[i]["source_face_indices"])==1
        for i in cylinders) and all(charts[i]["kind"]=="plane" for i in diag["selected_source_face_indices"]
        if i not in cylinders),"V6_SELECTED_CHART_ROLES_INVALID")
    slices=diag.get("developed_native_face_slices")
    _require(type(slices) is list and slices,"V6_DEVELOPED_SOURCE_SLICES_MISSING")
    expected_sources=sorted(i for chart in diag["selected_source_face_indices"]
        for i in charts[chart]["source_face_indices"])
    _require(diag.get("selected_chart_face_indices")==diag["selected_source_face_indices"]
        and diag.get("selected_native_face_indices")==expected_sources,
        "V6_SELECTED_SOURCE_BIJECTION_MISMATCH")
    observed=[]
    vertex_cursor=triangle_cursor=0
    expected_slices=[]
    vbase=tbase=0
    chart_spans={}
    for ident in diag["selected_source_face_indices"]:
        chart_spans[ident]={"vertex_start":vbase,"triangle_start":tbase,
            "vertex_count":sum(m["vertex_count"] for m in charts[ident]["members"]),
            "triangle_count":sum(m["triangle_count"] for m in charts[ident]["members"])}
        for member in charts[ident]["members"]:
            expected_slices.append({"source_face_index":member["source_face_index"],
                "chart_face_index":ident,"body_vertex_start":vbase+member["vertex_start"],
                "body_vertex_count":member["vertex_count"],
                "body_triangle_start":tbase+member["triangle_start"],
                "body_triangle_count":member["triangle_count"]})
        vbase+=sum(m["vertex_count"] for m in charts[ident]["members"])
        tbase+=sum(m["triangle_count"] for m in charts[ident]["members"])
    _require(slices==expected_slices,"V6_SOURCE_SLICE_LEDGER_MISMATCH")
    for row in slices:
        _require(type(row) is dict and all(type(row.get(key)) is int for key in (
            "source_face_index","chart_face_index","body_vertex_start","body_vertex_count",
            "body_triangle_start","body_triangle_count")),"V6_SOURCE_SLICE_TYPE_INVALID")
        _require(row["body_vertex_start"]==vertex_cursor and row["body_triangle_start"]==triangle_cursor
            and row["body_vertex_count"]>=3 and row["body_triangle_count"]>0,
            "V6_SOURCE_SLICE_PARTITION_INVALID")
        _require(row["chart_face_index"] in diag["selected_source_face_indices"]
            and row["source_face_index"] in charts[row["chart_face_index"]]["source_face_indices"],
            "V6_SOURCE_SLICE_OWNERSHIP_INVALID")
        observed.append(row["source_face_index"])
        vertex_cursor+=row["body_vertex_count"]
        triangle_cursor+=row["body_triangle_count"]
    _require(sorted(observed)==expected_sources and vertex_cursor*3==len(payload["vertices"])
        and triangle_cursor*3==len(payload["triangles"]),"V6_SOURCE_SLICE_COVERAGE_INVALID")
    _require(type(diag.get("reoriented_triangle_count")) is int
        and 0<=diag["reoriented_triangle_count"]<=triangle_cursor,"V6_REORIENTATION_COUNT_INVALID")
    p=np.asarray(payload["vertices"],dtype=float).reshape(-1,3)
    t=np.asarray(payload["triangles"],dtype=np.int64).reshape(-1,3)
    for row in slices:
        local=t[row["body_triangle_start"]:row["body_triangle_start"]+row["body_triangle_count"]]
        _require(np.all(local>=row["body_vertex_start"])
            and np.all(local<row["body_vertex_start"]+row["body_vertex_count"]),
            "V6_TRIANGLE_CROSSES_NATIVE_FACE_SLICE")
    q=p[t]
    signs=(q[:,1,0]-q[:,0,0])*(q[:,2,1]-q[:,0,1])-(q[:,1,1]-q[:,0,1])*(q[:,2,0]-q[:,0,0])
    _require(np.isfinite(signs).all() and np.all(signs>1e-12),"V6_FLAT_WINDING_INVALID")
    child_by_feature={row["feature_id"]:row["child_source_face_index"]
        for row in diag["finite_width_proofs"]}
    for overlay in payload["flange_surfaces"]:
        child=child_by_feature[overlay["feature_id"]]
        span=chart_spans[child]
        start,count=span["vertex_start"],span["vertex_count"]
        tri_start,tri_count=span["triangle_start"],span["triangle_count"]
        _require(overlay["vertices"]==payload["vertices"][start*3:(start+count)*3]
            and overlay["triangles"]==(t[tri_start:tri_start+tri_count]-start).reshape(-1).tolist(),
            "V6_FLANGE_OVERLAY_CHILD_CHART_PARITY_INVALID")
    return payload
