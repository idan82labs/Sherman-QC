#!/usr/bin/env python3
"""Render fixed-camera 4096px V33 PNFM pre/post review montages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output"/"research"/"f1-v33-pnfm-v1"
FIX=OUT/"source_free_fixtures"
PRE=OUT/"preinvoke_visuals"
POST=OUT/"postinvoke_visuals"


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound(fid:str)->tuple[dict[str,Any],dict[str,Any]]:
    descriptor=load(FIX/f"{fid}.json")
    raw=load(ROOT/descriptor["triangle_binding_path"])
    return descriptor,raw


def _candidate_frame(candidate:dict[str,Any])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    p0=np.asarray(candidate["render_line_start"],dtype=float);p1=np.asarray(candidate["render_line_end"],dtype=float)
    a=(p1-p0)/np.linalg.norm(p1-p0);o=.5*(p0+p1)
    helper=np.asarray([1.,0.,0.]) if abs(a[0])<.8 else np.asarray([0.,0.,1.])
    b=np.cross(a,helper);b/=np.linalg.norm(b);m=np.cross(b,a)
    return p0,p1,a,b,float(candidate["local_scale_s"])


def render(fid:str,destination:Path,decision:dict[str,Any]|None=None,visual:dict[str,Any]|None=None)->dict[str,Any]:
    descriptor,raw=load_bound(fid);points=np.asarray(raw["positions"],dtype=float);faces=np.asarray(raw["triangles"],dtype=np.int64)
    candidate=descriptor["candidate_records"][0];p0,p1,a,b,s=_candidate_frame(candidate);o=.5*(p0+p1);m=np.cross(b,a)
    rel=points-o;u=rel@a;q=rel@b;r=rel@m
    fig=plt.figure(figsize=(16,16),dpi=256,facecolor="#081018")
    panels=[fig.add_subplot(221,projection="3d"),fig.add_subplot(222,projection="3d"),fig.add_subplot(223),fig.add_subplot(224)]
    for ax in panels:ax.set_facecolor("#0d1822")
    used=np.unique(faces.reshape(-1)) if len(faces) else np.arange(len(points),dtype=np.int64)
    stride=max(1,len(used)//32000);sample=used[::stride]
    if len(faces):
        seen:set[tuple[int,int,int]]=set();unique_rows=[]
        for face in faces:
            key=tuple(sorted(int(value) for value in face))
            if key not in seen:
                seen.add(key);unique_rows.append(face)
        canonical_faces=np.asarray(unique_rows,dtype=np.int64)
        fs=canonical_faces
        xyz=np.column_stack((q,u,r));polys=xyz[fs]
        uc=np.mean(u[fs],axis=1);umin,umax=float(np.min(uc)),float(np.max(uc));den=max(umax-umin,1e-12)
        colors=plt.get_cmap("viridis")((uc-umin)/den)
        panels[0].add_collection3d(Poly3DCollection(polys,facecolors=colors,edgecolors="none",alpha=.92))
        panels[0].auto_scale_xyz(q[used],u[used],r[used])
        wstride=max(1,len(canonical_faces)//4000);wf=canonical_faces[::wstride]
        segments3=[]
        for tri in wf:
            tri3=xyz[tri];segments3.extend((tri3[[0,1]],tri3[[1,2]],tri3[[2,0]]))
        panels[1].add_collection3d(Line3DCollection(segments3,colors="#8cc7d8",linewidths=.22,alpha=.58))
        panels[1].auto_scale_xyz(q[used],u[used],r[used])
    else:
        panels[0].scatter(q[sample],u[sample],r[sample],c=u[sample],s=2,cmap="viridis",alpha=.8)
        panels[1].scatter(q[sample],u[sample],r[sample],c="#8cc7d8",s=1.5,alpha=.65)
    candidate_u=np.asarray((-.5*np.linalg.norm(p1-p0),.5*np.linalg.norm(p1-p0)))
    for ax in panels[:2]:
        ax.plot([0,0],candidate_u,[0,0],color="#ff5b5b",lw=2.5)
        ax.view_init(elev=24,azim=-58);ax.set_box_aspect((1.0,2.2,1.0));ax.set_xlabel("q");ax.set_ylabel("u");ax.set_zlabel("r")
    panels[0].set_title("SHADED NATIVE TRIANGLE SURFACE",color="white")
    panels[1].set_title("TRIANGLE WIREFRAME · frozen diagonalization",color="white")
    panels[2].scatter(q[sample],r[sample],c="#d7e4ea",s=.45,alpha=.55,rasterized=True)
    panels[2].axvline(0,color="#ff5b5b",lw=2);panels[2].axhline(0,color="#ff5b5b",lw=.7,alpha=.6)
    panels[2].set_title("ORTHOGRAPHIC CANDIDATE-NORMAL CROSS-SECTION",color="white")
    radial=np.sqrt(q*q+r*r)
    panels[3].scatter(u[sample],radial[sample],c=q[sample],s=.55,cmap="coolwarm",alpha=.65,rasterized=True)
    panels[3].axvline(-.5*np.linalg.norm(p1-p0),color="#ff5b5b",lw=1.4);panels[3].axvline(.5*np.linalg.norm(p1-p0),color="#ff5b5b",lw=1.4)
    panels[3].set_title("CANDIDATE-AXIS VIEW · axial material/fragmentation",color="white")
    for index,other in enumerate(descriptor["candidate_records"][1:],start=1):
        op0=np.asarray(other["render_line_start"],dtype=float);op1=np.asarray(other["render_line_end"],dtype=float)
        mid=.5*(op0+op1);d=mid-o;oq=float(np.dot(d,b));orr=float(np.dot(d,m));ou=float(np.dot(d,a))
        color=("#ffb347","#d77bff","#4de3a7")[min(index-1,2)]
        for ax in panels[:2]:ax.plot([oq,oq],candidate_u+ou,[orr,orr],color=color,lw=2.0)
        panels[2].axvline(oq,color=color,lw=1.5)
    if decision is not None and visual:
        palette={"low":"#35c8ff","core":"#ffd43b","high":"#ff6b9d"}
        by_face={int(piece["face"]):piece for piece in visual.get("pieces",[])}
        for name,face_ids in visual.get("groups",{}).items():
            rows=[by_face[int(face)] for face in face_ids if int(face) in by_face]
            if not rows:continue
            pp=np.asarray([[row["q"],row["u"],row["r"]] for row in rows],dtype=float)
            panels[2].scatter(pp[:,0],pp[:,2],c=palette.get(name,"white"),s=8,alpha=.62,label=name,rasterized=True)
            panels[3].scatter(pp[:,1]-.5*np.linalg.norm(p1-p0),np.sqrt(pp[:,0]**2+pp[:,2]**2),c=palette.get(name,"white"),s=8,alpha=.62,label=name,rasterized=True)
        panels[2].legend(loc="upper right",facecolor="#132431",labelcolor="white")
    for ax in panels:
        ax.tick_params(colors="#9fb2c0");ax.grid(color="#29404f",alpha=.25)
        if not hasattr(ax,"zaxis"):
            ax.set_aspect("equal",adjustable="datalim")
            for spine in ax.spines.values():spine.set_color("#314654")
    subtitle=f"{fid}  |  vertices={len(points):,} triangles={len(faces):,}  |  PRE-INVOKE"
    if decision is not None:
        subtitle=f"{fid}  |  POST-INVOKE  |  {decision['reason']}  |  action={decision['action']}\nmetrics={decision['metrics_i32']}"
    fig.suptitle(subtitle,color="white",fontsize=13,y=.985)
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=256,facecolor=fig.get_facecolor(),pil_kwargs={"compress_level":6});plt.close(fig)
    return {"fixture_id":fid,"path":str(destination.relative_to(ROOT)),"sha256":sha(destination),"width":4096,"height":4096,"vertices":len(points),"triangles":len(faces),"stage":"post" if decision is not None else "pre"}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--all-pre",action="store_true");parser.add_argument("--fixture");args=parser.parse_args()
    lock=load(ROOT/"research"/"f1_v33_pnfm_v1_lock.json")
    ids=lock["ordered_fixture_ids"] if args.all_pre else [args.fixture]
    rows=[render(fid,PRE/f"{fid}_pre_4096.png") for fid in ids]
    manifest={"schema_version":1,"kind":"v33_pnfm_preinvoke_visual_manifest","renderer_sha256":sha(Path(__file__)),"rows":rows}
    path=OUT/"preinvoke_visual_manifest.json";path.write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print(json.dumps({"rendered":len(rows),"manifest":str(path),"manifest_sha256":sha(path)},indent=2))


if __name__=="__main__":main()
