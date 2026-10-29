#!/usr/bin/env python3
"""Freeze V33 PNFM triangle bindings and all ordered source-free fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research"/"f1_v33_pnfm_v1_lock.json"
V32=ROOT/"output"/"research"/"f1-v32-avhfm-v1"/"source_free_fixtures"
OUT=ROOT/"output"/"research"/"f1-v33-pnfm-v1"
FIX=OUT/"source_free_fixtures"
BIND=OUT/"triangle_bindings"
MANIFEST=OUT/"fixture_manifest.json"


def write_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))


def key_parts(key:int)->tuple[int,int,int]:
    skin=key//10_000_000
    rem=key-skin*10_000_000-1
    return skin,rem//10_000,rem%10_000


def lattice_quads(keys:list[int])->list[tuple[int,int,int,int,int,int]]:
    by={key_parts(int(key)):index for index,key in enumerate(keys)}
    rows=[]
    for (skin,yi,ci),a in sorted(by.items()):
        b=by.get((skin,yi,ci+1));c=by.get((skin,yi+1,ci));d=by.get((skin,yi+1,ci+1))
        if b is not None and c is not None and d is not None:rows.append((skin,yi,ci,a,b,c,d))
    return rows


def faces_from_keys(keys:list[int],mode:str="base")->list[list[int]]:
    faces=[]
    for _skin,yi,ci,a,b,c,d in lattice_quads(keys):
        flip=mode=="flipped" or (mode=="checker" and (yi+ci)%2==1)
        faces.extend(([a,c,d],[a,d,b]) if flip else ([a,c,b],[b,c,d]))
    return faces


def raw_from_v32(payload:dict[str,Any],mode:str="base",skew:bool=False)->dict[str,Any]:
    source=payload["source"]
    positions=np.asarray(source["positions"],dtype=np.float64).copy()
    if skew:
        for index,key in enumerate(source["stable_source_keys"]):
            _skin,yi,_ci=key_parts(int(key))
            if 0<yi<64:positions[index,1]+=(0.375 if yi%2 else -0.375)
    return {"positions":positions.tolist(),"triangles":faces_from_keys(source["stable_source_keys"],mode)}


def descriptor(fid:str,source_path:Path,source_sha:str,tri_path:Path,tri_sha:str,candidates:list[dict[str,Any]],expected:list[int]|None,construction:str)->dict[str,Any]:
    return {"schema_version":1,"kind":"v33_pnfm_frozen_fixture_binding","fixture_id":fid,"source_payload_path":str(source_path.relative_to(ROOT)),"source_payload_sha256":source_sha,"triangle_binding_path":str(tri_path.relative_to(ROOT)),"triangle_binding_sha256":tri_sha,"candidate_records":candidates,"expected_pre_final":expected,"construction":construction}


def freeze_bound(fid:str,payload:dict[str,Any],mode:str="base",skew:bool=False,expected:list[int]|None=None,construction:str="") -> dict[str,Any]:
    raw=raw_from_v32(payload,mode,skew)
    side=BIND/f"{fid}.json";write_json(side,raw)
    source_path=V32/f"{payload['fixture_id']}.json"
    row=descriptor(fid,source_path,sha(source_path),side,sha(side),payload["candidate_records"],expected,construction)
    write_json(FIX/f"{fid}.json",row);return row


def generated(fid:str,raw:dict[str,Any],candidates:list[dict[str,Any]],expected:list[int]|None,construction:str)->dict[str,Any]:
    source_path=BIND/f"{fid}_generated_source.json";write_json(source_path,{"fixture_id":fid,"source":raw,"candidate_records":candidates})
    side=BIND/f"{fid}.json";write_json(side,raw)
    row=descriptor(fid,source_path,sha(source_path),side,sha(side),candidates,expected,construction)
    write_json(FIX/f"{fid}.json",row);return row


def main()->None:
    lock=load(LOCK);base=load(V32/"SF01_FOLD60_MESH_C2.json")
    rows=[]
    rows.append(freeze_bound("SF01_FOLD60_MESH_C2",base,"base",False,[1,1],"exact byte-frozen V32 positive plus stable-key triangle binding"))
    rows.append(freeze_bound("SF01_RM_ALL_DIAGONALS_FLIPPED",base,"flipped",False,[1,1],"same frozen vertices; all lattice diagonals flipped"))
    rows.append(freeze_bound("SF01_RM_CHECKERBOARD_DIAGONALS",base,"checker",False,[1,1],"same frozen vertices; alternating lattice diagonals"))
    rows.append(freeze_bound("SF01_RM_SKEW_LATTICE",base,"checker",True,[1,1],"same analytic surface with frozen +/-3s/8 axial row stagger"))

    base_raw=raw_from_v32(base)
    points=np.asarray(base_raw["positions"],dtype=np.float64)
    # Axial adversary: rotate each cross-section about the candidate axis as u varies.
    twisted=points.copy(); center=np.asarray(base["candidate_records"][0]["center_xyz"],dtype=np.float64)
    for i,p in enumerate(points):
        angle=np.deg2rad(55.0*(p[1]/16.0));x,z=p[0]-center[0],p[2]-center[2]
        twisted[i,0]=center[0]+np.cos(angle)*x-np.sin(angle)*z;twisted[i,2]=center[2]+np.sin(angle)*x+np.cos(angle)*z
    rows.append(generated("SF_PNFM_ADV01_AXIAL_TWISTED_FAN",{"positions":twisted.tolist(),"triangles":base_raw["triangles"]},base["candidate_records"],[0,0],"axially twisted ruled fan"))

    keys=base["source"]["stable_source_keys"]
    comp=[]
    for tri in base_raw["triangles"]:
        skin=key_parts(int(keys[tri[0]]))[0];u=float(np.mean(points[np.asarray(tri),1]));slab=min(5,max(0,int((u+12.0)//4.0)))
        if skin==slab%2:comp.append(tri)
    rows.append(generated("SF_PNFM_FRAG_COMPLEMENTARY_SKINS",{"positions":points.tolist(),"triangles":comp},base["candidate_records"],[1,1],"six alternating complete-fan skin slabs"))
    phase=[]
    for tri in base_raw["triangles"]:
        xyz=np.mean(points[np.asarray(tri)],axis=0);u=float(xyz[1]);q=float(xyz[0]-center[0])
        remove=(q<-2 and -10<u<-7) or (-2<=q<=2 and -1.5<u<1.5) or (q>2 and 7<u<10)
        if not remove:phase.append(tri)
    rows.append(generated("SF_PNFM_FRAG_PHASE_WINDOWS",{"positions":points.tolist(),"triangles":phase},base["candidate_records"],[1,1],"phase-specific noncoincident 3s windows"))

    for fid in lock["ordered_fixture_ids"][7:-4]:
        payload=load(V32/f"{fid}.json")
        rows.append(freeze_bound(fid,payload,"base",False,[0,0],"byte-frozen V32 nuisance/adversary plus stable-key triangle binding"))

    duplicate_raw={"positions":points.tolist(),"triangles":base_raw["triangles"]+base_raw["triangles"]+[list(reversed(t)) for t in base_raw["triangles"]]}
    rows.append(generated("SD01_DUPLICATE_REVERSED_FACES",duplicate_raw,base["candidate_records"],[1,1],"exact duplicate and reversed-winding faces"))
    candidates=json.loads(json.dumps(base["candidate_records"]));second=json.loads(json.dumps(candidates[0]));second["candidate_id"]+=1;second["frozen_v16_order"]+=1;candidates.append(second)
    rows.append(generated("SD02_TWO_EXACT_RESCUE_CANDIDATES",base_raw,candidates,[2,1],"two exact rescue lines"))
    legacy=json.loads(json.dumps(candidates));legacy[0]["selected_legacy"]=True;legacy[0]["rejected_before_final_nms"]=False
    rows.append(generated("SD03_LEGACY_PLUS_RESCUE_DUPLICATE",base_raw,legacy,[1,1],"legacy-first exact duplicate"))
    near=json.loads(json.dumps(candidates));near[1]["render_line_start"][0]+=0.5;near[1]["render_line_end"][0]+=0.5;near[1]["center_xyz"][0]+=0.5
    rows.append(generated("SD04_WITHIN_RADIUS_NEAR_DUPLICATE",base_raw,near,[2,1],"ordinary-NMS within-radius near duplicate"))

    ids=[row["fixture_id"] for row in rows]
    if ids!=lock["ordered_fixture_ids"]:raise SystemExit(f"fixture order drift: {ids}")
    manifest={"schema_version":1,"kind":"v33_pnfm_fixture_manifest","lock_sha256":sha(LOCK),"fixture_count":len(rows),"rows":[{"fixture_id":row["fixture_id"],"descriptor_path":str((FIX/f"{row['fixture_id']}.json").relative_to(ROOT)),"descriptor_sha256":sha(FIX/f"{row['fixture_id']}.json"),"source_payload_sha256":row["source_payload_sha256"],"triangle_binding_sha256":row["triangle_binding_sha256"],"expected_pre_final":row["expected_pre_final"]} for row in rows]}
    write_json(MANIFEST,manifest)
    print(json.dumps({"fixture_count":len(rows),"manifest":str(MANIFEST),"manifest_sha256":sha(MANIFEST),"base_triangle_count":len(base_raw["triangles"])},indent=2))


if __name__=="__main__":main()
