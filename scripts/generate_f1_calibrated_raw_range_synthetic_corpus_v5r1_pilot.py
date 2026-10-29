#!/usr/bin/env python3
"""High-fidelity finite-part morphology pilot following V5 visual rejection.

This generator adds finite sheet thickness, smooth punched apertures, rounded
or chamfered outlines, and finite-radius manufacturing folds.  It remains a
small visual-gate pilot: detector execution is forbidden until every fused and
native view passes manual morphology review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib.path as mpl_path
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 as base
from scripts.generate_f1_lidar_like_synthetic_corpus import _rotation


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V5R2_HIGH_FIDELITY_PILOT_NON_CONFIRMATORY"
CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_high_fidelity_finite_part_pilot_v5r2"
CAPTURE_INDICES = (11,)
THICKNESS = 1.45

FAMILIES: tuple[dict[str, Any], ...] = (
    {"id":"HR01","name":"rounded_asymmetric_control_bracket","radius":2.4,"challenge":"compound rounded outline, smooth holes, unequal radiused flanges and corner reliefs"},
    {"id":"HR02","name":"perforated_instrument_cover","radius":2.0,"challenge":"non-rectangular cover, patterned slots, long radiused wall and short mounting ears"},
    {"id":"HR03","name":"local_raised_access_channel","radius":2.2,"challenge":"finite local raised channel with terminated rounded folds and carrier apertures"},
    {"id":"HR04","name":"rolled_guard_with_mounting_tabs","radius":2.5,"challenge":"large smooth rolled form as negative control beside two true short tab bends"},
)


class Builder:
    def __init__(self) -> None:
        self.vertices: list[np.ndarray] = []
        self.faces: list[tuple[int,int,int]] = []

    def _surface(self, rows: list[list[np.ndarray]], reverse: bool = False) -> None:
        off = len(self.vertices)
        nr, nc = len(rows), len(rows[0])
        self.vertices.extend(point for row in rows for point in row)
        for j in range(nr-1):
            for i in range(nc-1):
                a=off+j*nc+i; b=a+1; d=off+(j+1)*nc+i; c=d+1
                self.faces.extend(((a,c,b),(a,d,c)) if reverse else ((a,b,c),(a,c,d)))

    def quad(self, p00, p10, p11, p01, *, nu=10, nv=5, reverse=False) -> None:
        p00,p10,p11,p01=(np.asarray(p,float) for p in (p00,p10,p11,p01))
        rows=[]
        for j in range(nv+1):
            v=j/nv; row=[]
            for i in range(nu+1):
                u=i/nu
                row.append((1-u)*(1-v)*p00+u*(1-v)*p10+u*v*p11+(1-u)*v*p01)
            rows.append(row)
        self._surface(rows,reverse=reverse)

    def mesh(self) -> o3d.geometry.TriangleMesh:
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(self.vertices,float)),o3d.utility.Vector3iVector(np.asarray(self.faces,np.int32)))
        mesh.remove_duplicated_triangles(); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
        return mesh


def _rounded_rect(x0:float,x1:float,y0:float,y1:float,r:float,n:int=8, *, chamfer:tuple[float,float,float,float]|None=None)->np.ndarray:
    points=[]
    for cx,cy,a0 in ((x1-r,y0+r,-90),(x1-r,y1-r,0),(x0+r,y1-r,90),(x0+r,y0+r,180)):
        for angle in np.linspace(a0,a0+90,n,endpoint=False):
            t=math.radians(angle); points.append((cx+r*math.cos(t),cy+r*math.sin(t)))
    points=np.asarray(points,float)
    if chamfer:
        # Pull selected extrema inward to break the perfect rounded rectangle.
        tl,tr,br,bl=chamfer
        for index,value in enumerate((tl,tr,br,bl)):
            if value<=0: continue
            corner=np.asarray(((x0,y1),(x1,y1),(x1,y0),(x0,y0))[index],float)
            dist=np.linalg.norm(points-corner,axis=1)
            mask=dist<r*1.8
            center=np.asarray(((x0+r,y1-r),(x1-r,y1-r),(x1-r,y0+r),(x0+r,y0+r))[index],float)
            points[mask]=center+(points[mask]-center)*max(0.55,1-value/(3*r))
    return points


def _circle(cx:float,cy:float,r:float,n:int=28)->np.ndarray:
    a=np.linspace(0,2*math.pi,n,endpoint=False)
    return np.column_stack((cx+r*np.cos(a),cy+r*np.sin(a)))


def _slot(cx:float,cy:float,length:float,width:float,angle_deg:float=0,n:int=12)->np.ndarray:
    rr=width/2; half=max(0.0,(length-width)/2)
    pts=[]
    for a in np.linspace(-90,90,n,endpoint=False):
        t=math.radians(a); pts.append((half+rr*math.cos(t),rr*math.sin(t)))
    for a in np.linspace(90,270,n,endpoint=False):
        t=math.radians(a); pts.append((-half+rr*math.cos(t),rr*math.sin(t)))
    pts=np.asarray(pts,float)
    t=math.radians(angle_deg); R=np.asarray(((math.cos(t),-math.sin(t)),(math.sin(t),math.cos(t))))
    return pts@R.T+np.asarray((cx,cy))


def _inside(poly:np.ndarray,points:np.ndarray)->np.ndarray:
    return mpl_path.Path(poly,closed=True).contains_points(points,radius=1e-7)


def add_sheet(builder:Builder, outer:np.ndarray, holes:list[np.ndarray], *, z:float=0.0, thickness:float=THICKNESS, spacing:float=5.0)->None:
    """Triangulate a finite thick sheet with smooth explicit boundaries."""
    lo=outer.min(axis=0); hi=outer.max(axis=0)
    gx=np.arange(lo[0]+spacing,hi[0],spacing); gy=np.arange(lo[1]+spacing,hi[1],spacing)
    grid=np.asarray([(x,y) for y in gy for x in gx],float)
    keep=_inside(outer,grid)
    for hole in holes: keep &= ~_inside(hole,grid)
    interior=grid[keep]
    rings=[outer,*holes]
    points=np.vstack([*rings,interior])
    # Remove accidental duplicates without perturbing boundary coordinates.
    _,indices=np.unique(np.round(points,8),axis=0,return_index=True); points=points[np.sort(indices)]
    tri=Delaunay(points).simplices
    cent=points[tri].mean(axis=1)
    valid=_inside(outer,cent)
    for hole in holes: valid &= ~_inside(hole,cent)
    tri=tri[valid]
    off=len(builder.vertices); count=len(points)
    builder.vertices.extend(np.column_stack((points,np.full(count,z))))
    builder.vertices.extend(np.column_stack((points,np.full(count,z-thickness))))
    builder.faces.extend(tuple(off+v for v in face) for face in tri)
    builder.faces.extend(tuple(off+count+v for v in face[::-1]) for face in tri)

    cursor=0
    for ring_index,ring in enumerate(rings):
        ring_count=len(ring)
        # unique() can only affect coincident rings; these authored rings do not overlap.
        for i in range(ring_count):
            a=off+cursor+i; b=off+cursor+(i+1)%ring_count
            ab=off+count+cursor+i; bb=off+count+cursor+(i+1)%ring_count
            if ring_index==0: builder.faces.extend(((a,ab,b),(b,ab,bb)))
            else: builder.faces.extend(((a,b,ab),(b,bb,ab)))
        cursor += ring_count


def add_rounded_flange_y(builder:Builder,x0:float,x1:float,y0:float,*,radius:float,length:float,angle_deg:float,up:bool=True,direction:float=1.0,nx:int=18,na:int=8)->tuple[np.ndarray,np.ndarray]:
    """Add a thick, finite-radius flange tangent to the parent z=0 sheet."""
    sign=1.0 if up else -1.0; angle=math.radians(angle_deg)
    def surface(z_shift:float,reverse:bool)->list[list[np.ndarray]]:
        rows=[]
        for a in np.linspace(0,angle,na+1):
            y=y0+direction*radius*math.sin(a); z=sign*radius*(1-math.cos(a))+z_shift
            rows.append([np.asarray((x,y,z),float) for x in np.linspace(x0,x1,nx+1)])
        ye=y0+direction*radius*math.sin(angle); ze=sign*radius*(1-math.cos(angle))+z_shift
        dy=direction*length*math.cos(angle); dz=sign*length*math.sin(angle)
        rows.append([np.asarray((x,ye+dy,ze+dz),float) for x in np.linspace(x0,x1,nx+1)])
        return rows
    top=surface(0,False); bottom=surface(-THICKNESS,True)
    builder._surface(top); builder._surface(bottom,reverse=True)
    # Long free edge and both finite ends make thickness visible.
    builder.quad(top[-1][0],top[-1][-1],bottom[-1][-1],bottom[-1][0],nu=nx,nv=1)
    builder.quad(top[0][0],top[-1][0],bottom[-1][0],bottom[0][0],nu=na+1,nv=1)
    builder.quad(top[-1][-1],top[0][-1],bottom[0][-1],bottom[-1][-1],nu=na+1,nv=1)
    mid=0.5*angle; y=y0+direction*radius*math.sin(mid); z=sign*radius*(1-math.cos(mid))
    return np.asarray((x0,y,z)),np.asarray((x1,y,z))


def add_rounded_flange_x(builder:Builder,y0:float,y1:float,x0:float,*,radius:float,length:float,angle_deg:float,up:bool=True,direction:float=1.0)->tuple[np.ndarray,np.ndarray]:
    temp=Builder(); s,e=add_rounded_flange_y(temp,y0,y1,x0,radius=radius,length=length,angle_deg=angle_deg,up=up,direction=direction)
    verts=np.asarray(temp.vertices); verts=verts[:,[1,0,2]]
    off=len(builder.vertices); builder.vertices.extend(verts); builder.faces.extend(tuple(off+v for v in f) for f in temp.faces)
    return s[[1,0,2]],e[[1,0,2]]


def truth_line(rows:list[dict[str,Any]],start:np.ndarray,end:np.ndarray)->None:
    rows.append({"bend_id":f"G{len(rows)+1:03d}","line_start":start.tolist(),"line_end":end.tolist()})


def case1():
    b=Builder(); truth=[]
    outer=_rounded_rect(-92,92,-57,57,14,chamfer=(3,0,5,1))
    holes=[_circle(-52,14,9),_slot(30,-17,35,11,8),_circle(67,28,5)]
    add_sheet(b,outer,holes)
    truth_line(truth,*add_rounded_flange_y(b,-73,50,-57,radius=2.4,length=20,angle_deg=78,up=True,direction=-1))
    truth_line(truth,*add_rounded_flange_x(b,-35,39,-92,radius=2.4,length=17,angle_deg=70,up=False,direction=-1))
    truth_line(truth,*add_rounded_flange_y(b,55,81,57,radius=2.4,length=13,angle_deg=62,up=True))
    return b.mesh(),truth,{"hole_count":len(holes),"rounded_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case2():
    b=Builder(); truth=[]
    outer=_rounded_rect(-98,98,-61,61,18,chamfer=(0,4,2,5))
    holes=[_circle(-72,-35,5),_circle(72,35,5),_slot(-35,12,24,7,-12),_slot(18,18,30,6,6),_slot(48,-25,22,6,0)]
    for row in range(3):
        for col in range(5): holes.append(_slot(-42+21*col,-25+10*row,12,3.8,0,n=8))
    add_sheet(b,outer,holes,spacing=4.6)
    truth_line(truth,*add_rounded_flange_y(b,-71,74,61,radius=2.0,length=25,angle_deg=86,up=True,nx=28))
    truth_line(truth,*add_rounded_flange_y(b,-85,-58,-61,radius=2.0,length=11,angle_deg=58,up=False,direction=-1))
    truth_line(truth,*add_rounded_flange_x(b,26,49,98,radius=2.0,length=13,angle_deg=66,up=True))
    return b.mesh(),truth,{"hole_count":len(holes),"rounded_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case3():
    b=Builder(); truth=[]
    outer=_rounded_rect(-105,105,-64,64,16,chamfer=(2,5,0,3))
    x0,x1=-33,51; y0,y1=-18,22
    holes=[_circle(-76,-33,6),_slot(70,31,28,8,90),_circle(79,-38,5),_rounded_rect(x0-2,x1+2,y0-1,y1+1,4)]
    add_sheet(b,outer,holes,spacing=5.2)
    # Finite raised access channel: two rounded lower folds, sloped walls, and
    # two rounded upper folds over only part of the carrier length.
    x0,x1=-33,51; y0,y1=-18,22; r=2.2
    a0,a1=add_rounded_flange_y(b,x0,x1,y0,radius=r,length=18,angle_deg=72,up=True,direction=1,nx=20); truth_line(truth,a0,a1)
    c0,c1=add_rounded_flange_y(b,x0,x1,y1,radius=r,length=18,angle_deg=72,up=True,direction=-1,nx=20); truth_line(truth,c0,c1)
    truth_line(truth,*add_rounded_flange_y(b,-94,-61,-64,radius=2.2,length=10,angle_deg=54,up=False,direction=-1))
    return b.mesh(),truth,{"hole_count":len(holes),"rounded_negative_count":0,"sheet_thickness_mm":THICKNESS}


def case4():
    b=Builder(); truth=[]
    outer=_rounded_rect(-96,96,-59,44,17,chamfer=(4,1,0,5))
    holes=[_circle(-62,-14,8),_slot(13,5,39,10,-4),_circle(70,-31,5)]
    add_sheet(b,outer,holes,spacing=4.8)
    # Broad smooth continuation is a deliberate negative; there is no truth
    # line at either tangent of this 18 mm radius roll.
    add_rounded_flange_y(b,-88,88,44,radius=18,length=4,angle_deg=88,up=True,nx=36,na=18)
    truth_line(truth,*add_rounded_flange_y(b,-77,-41,-59,radius=2.5,length=13,angle_deg=67,up=True,direction=-1))
    truth_line(truth,*add_rounded_flange_y(b,34,79,-59,radius=2.5,length=12,angle_deg=63,up=False,direction=-1))
    return b.mesh(),truth,{"hole_count":len(holes),"rounded_negative_count":2,"sheet_thickness_mm":THICKNESS}


BUILDERS=(case1,case2,case3,case4)


def build_case(family:dict[str,Any],capture_index:int):
    n=int(family["id"][-2:]); mesh,truth,meta=BUILDERS[n-1]()
    rng=np.random.default_rng(610_000+97*n+capture_index); R=_rotation(620_000+53*n+capture_index); t=rng.uniform([-14,-10,-6],[14,10,6])
    v=np.asarray(mesh.vertices)@R.T+t; mesh.vertices=o3d.utility.Vector3dVector(v); mesh.compute_vertex_normals()
    for row in truth:
        line=np.asarray([row["line_start"],row["line_end"]])@R.T+t
        row["line_start"]=line[0].round(9).tolist(); row["line_end"]=line[1].round(9).tolist()
    meta.update({"family_id":family["id"],"family_name":family["name"],"capture":capture_index,"spacing_mm":None,"normal_noise_sigma_mm":base.capture.DEPTH_NOISE_SIGMA_MM,"tangential_noise_sigma_mm":None,"vertex_count":len(mesh.vertices),"triangle_count":len(mesh.triangles),"bend_count":len(truth),"finite_sheet_topology":True,"constant_cross_section":False,"finite_bend_radius":True,"smooth_aperture_boundaries":True,"rotation":R.round(12).tolist(),"translation_mm":t.round(9).tolist()})
    return mesh,truth,meta


def generate(output_root:Path,*,width:int=base.capture.DEFAULT_WIDTH,height:int=base.capture.DEFAULT_HEIGHT,view_count:int=base.capture.DEFAULT_VIEW_COUNT,family_limit:int|None=None):
    previous=(base.TOKEN,base.CORPUS_KIND,base.CAPTURE_INDICES,base.FAMILIES,base.build_case)
    base.TOKEN,base.CORPUS_KIND,base.CAPTURE_INDICES,base.FAMILIES,base.build_case=TOKEN,CORPUS_KIND,CAPTURE_INDICES,FAMILIES,build_case
    try: report=base.generate(output_root,width=width,height=height,view_count=view_count,family_limit=family_limit)
    finally: base.TOKEN,base.CORPUS_KIND,base.CAPTURE_INDICES,base.FAMILIES,base.build_case=previous
    path=output_root/"evaluator_only"/"renderer_manifest.json"; renderer=json.loads(path.read_text())
    renderer.update({"generator_sha256":base.capture._sha256(Path(__file__).resolve()),"visual_acceptance_required_before_inference":True,"detector_execution_authorized":False,"finite_sheet_topology":True,"constant_cross_section":False,"sheet_thickness_mm":THICKNESS,"finite_bend_radius":True,"smooth_aperture_boundaries":True})
    base.capture._atomic_json(path,renderer); return report


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--width",type=int,default=base.capture.DEFAULT_WIDTH); p.add_argument("--height",type=int,default=base.capture.DEFAULT_HEIGHT); p.add_argument("--view-count",type=int,default=base.capture.DEFAULT_VIEW_COUNT); p.add_argument("--family-limit",type=int); a=p.parse_args()
    report=generate(a.output_root.resolve(),width=a.width,height=a.height,view_count=a.view_count,family_limit=a.family_limit)
    print(json.dumps({"output_root":str(a.output_root.resolve()),"case_count":report["case_count"],"family_count":report["family_count"],"detector_execution_authorized":False,"token":TOKEN},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
