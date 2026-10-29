#!/usr/bin/env python3
"""Extract paired-skin sharp planar folds from STEP B-Rep topology."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS


MIN_NORMAL_ANGLE_DEG = 15.0
MAX_NORMAL_ANGLE_DEG = 90.0
MAX_EDGE_ANGLE_DEG = 0.5
MIN_OVERLAP = 0.80
THICKNESS_FACTOR = 1.50


def _canon(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64); v /= np.linalg.norm(v)
    if v[int(np.argmax(np.abs(v)))] < 0: v = -v
    return v


def _line_metrics(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, float]:
    x=np.asarray([a["line_start"],a["line_end"]]); y=np.asarray([b["line_start"],b["line_end"]])
    d=_canon(x[1]-x[0]); e=_canon(y[1]-y[0])
    angle=math.degrees(math.acos(float(np.clip(abs(np.dot(d,e)),-1,1))))
    delta=y.mean(0)-x.mean(0); offset=float(np.linalg.norm(delta-d*np.dot(delta,d)))
    origin=x.mean(0); ai=sorted(float(np.dot(p-origin,d)) for p in x); bi=sorted(float(np.dot(p-origin,d)) for p in y)
    overlap=max(0.0,min(ai[1],bi[1])-max(ai[0],bi[0])); shorter=min(ai[1]-ai[0],bi[1]-bi[0])
    return angle,offset,overlap/shorter if shorter>1e-9 else 0.0


def _normal_pair_equivalent(a: list[list[float]], b: list[list[float]]) -> bool:
    left=[_canon(np.asarray(x)) for x in a]; right=[_canon(np.asarray(x)) for x in b]
    scores=[]
    for order in ((0,1),(1,0)):
        scores.append(min(abs(float(np.dot(left[i],right[order[i]]))) for i in (0,1)))
    return max(scores) >= math.cos(math.radians(2.0))


def extract_planar_fold_events(path: Path, *, material_thickness_mm: float) -> dict[str, Any]:
    reader=STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or int(reader.TransferRoots()) <= 0:
        raise ValueError(f"failed to read STEP: {path}")
    shape=reader.OneShape(); mapping=TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape,TopAbs_EDGE,TopAbs_FACE,mapping)
    raw=[]
    for index in range(1,mapping.Extent()+1):
        faces=[TopoDS.Face_s(x) for x in mapping.FindFromIndex(index)]
        if len(faces)!=2: continue
        surfaces=[BRepAdaptor_Surface(x) for x in faces]
        if any(s.GetType()!=GeomAbs_Plane for s in surfaces): continue
        normals=[]
        for s in surfaces:
            d=s.Plane().Axis().Direction(); normals.append([d.X(),d.Y(),d.Z()])
        angle=math.degrees(math.acos(float(np.clip(abs(np.dot(_canon(normals[0]),_canon(normals[1]))),-1,1))))
        if not MIN_NORMAL_ANGLE_DEG <= angle <= MAX_NORMAL_ANGLE_DEG: continue
        curve=BRepAdaptor_Curve(TopoDS.Edge_s(mapping.FindKey(index)))
        if curve.GetType() != GeomAbs_Line:
            continue
        p0=curve.Value(curve.FirstParameter()); p1=curve.Value(curve.LastParameter())
        line=[np.array([p0.X(),p0.Y(),p0.Z()]),np.array([p1.X(),p1.Y(),p1.Z()])]
        if np.linalg.norm(line[1]-line[0]) <= 1e-6: continue
        line_direction = _canon(line[1] - line[0])
        line.sort(key=lambda p:float(np.dot(p, line_direction)))
        raw.append({"edge_index":index,"line_start":line[0].tolist(),"line_end":line[1].tolist(),"plane_normals":normals,"normal_angle_deg":angle})
    used=set(); events=[]
    for i,left in enumerate(raw):
        candidates=[]
        for j in range(i+1,len(raw)):
            if j in used or not _normal_pair_equivalent(left["plane_normals"],raw[j]["plane_normals"]): continue
            angle,offset,overlap=_line_metrics(left,raw[j])
            if angle<=MAX_EDGE_ANGLE_DEG and 0.2<=offset<=THICKNESS_FACTOR*material_thickness_mm and overlap>=MIN_OVERLAP:
                candidates.append((abs(offset-material_thickness_mm),j,offset,overlap))
        if not candidates or i in used: continue
        _,j,offset,overlap=min(candidates); used.update((i,j)); right=raw[j]
        a=np.asarray([left["line_start"],left["line_end"]]); b=np.asarray([right["line_start"],right["line_end"]])
        if np.dot(a[1]-a[0],b[1]-b[0])<0:b=b[::-1]
        mid=(a+b)/2.0
        events.append({"event_id":f"STEP_SHARP_{len(events)+1:03d}","source_edge_indices":[left["edge_index"],right["edge_index"]],"cad_line_start_mm":mid[0].tolist(),"cad_line_end_mm":mid[1].tolist(),"skin_separation_mm":offset,"common_span_fraction":overlap,"normal_angle_deg":0.5*(left["normal_angle_deg"]+right["normal_angle_deg"]),"relation":"PAIRED_PLANAR_SKIN_EDGES_TO_ONE_NEUTRAL_EVENT"})
    return {"schema":"SHERMAN_QC_STEP_PLANAR_FOLD_EVENTS_V1","material_thickness_mm":material_thickness_mm,"raw_planar_dihedral_edges":len(raw),"event_count":len(events),"events":events}
