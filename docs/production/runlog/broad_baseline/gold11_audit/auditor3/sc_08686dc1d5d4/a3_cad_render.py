#!/usr/bin/env python3
"""Auditor 3: tessellate the 10839000_01 CAD with OCP and render 4 views, colouring cylindrical
(bend) faces, to see the finished part's bends (read-only; writes PNG + JSON here)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

STEP = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp"
OUT = Path(__file__).resolve().parent

r = STEPControl_Reader()
assert r.ReadFile(STEP) == IFSelect_RetDone
r.TransferRoots()
shape = r.OneShape()
BRepMesh_IncrementalMesh(shape, 0.3, False, 0.3, True)

tris_all, kinds = [], []
faces = []
exp = TopExp_Explorer(shape, TopAbs_FACE)
fi = 0
while exp.More():
    f = TopoDS.Face_s(exp.Current())
    s = BRepAdaptor_Surface(f)
    st = s.GetType()
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f, loc)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, props)
    area = props.Mass()
    V = None
    if tri is not None:
        tr = loc.Transformation()
        V = np.array([[tri.Node(i).Transformed(tr).X(), tri.Node(i).Transformed(tr).Y(), tri.Node(i).Transformed(tr).Z()]
                      for i in range(1, tri.NbNodes() + 1)])
        T = np.array([[tri.Triangle(i).Value(j) - 1 for j in (1, 2, 3)] for i in range(1, tri.NbTriangles() + 1)])
        tris_all.append(V[T])
        kinds.extend([("cyl" if st == GeomAbs_Cylinder else "plane" if st == GeomAbs_Plane else "other", fi)] * len(T))
    row = {"face": fi, "type": int(st), "area": area}
    if st == GeomAbs_Cylinder and V is not None:
        c = s.Cylinder()
        ax = c.Axis()
        p0 = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        t = (V - p0) @ d
        rad = c.Radius()
        # angular span of the face around its axis
        rel = V - p0 - np.outer(t, d)
        a = np.arctan2(rel @ np.cross(d, rel[0] / max(1e-9, np.linalg.norm(rel[0]))), rel @ (rel[0] / max(1e-9, np.linalg.norm(rel[0]))))
        row.update({"radius": rad, "axis_dir": d.round(4).tolist(), "axis_pt_mid": (p0 + d * 0.5 * (t.min() + t.max())).round(2).tolist(),
                    "len_along_axis": float(t.max() - t.min()), "ang_span_deg": float(math.degrees(a.max() - a.min()))})
    if st == GeomAbs_Plane and V is not None:
        pl = s.Plane()
        nrm = pl.Axis().Direction()
        row.update({"normal": [round(nrm.X(), 4), round(nrm.Y(), 4), round(nrm.Z(), 4)],
                    "bbox_min": V.min(0).round(2).tolist(), "bbox_max": V.max(0).round(2).tolist()})
    faces.append(row)
    fi += 1
    exp.Next()

tris = np.concatenate(tris_all)
(OUT / "cad_faces.json").write_text(json.dumps(faces, indent=1))
allv = tris.reshape(-1, 3)
print("CAD bbox", allv.min(0).round(2), allv.max(0).round(2), "faces", len(faces))
for row in faces:
    if "radius" in row and row["len_along_axis"] > 5:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
big = sorted([f for f in faces if "normal" in f], key=lambda f: -f["area"])[:14]
for f in big:
    print("plane", f["face"], round(f["area"], 1), f["normal"], f["bbox_min"], f["bbox_max"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cen = allv.mean(0)
fig = plt.figure(figsize=(16, 13), dpi=90)
views = [(20, -60), (90, -90), (0, -90), (0, 0)]
col = np.array([("#d62728" if k == "cyl" else "#bbbbbb") for k, _ in kinds])
for i, (el, az) in enumerate(views):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    pc = Poly3DCollection(tris - cen, facecolors=col, edgecolors="none", alpha=0.9)
    ax.add_collection3d(pc)
    R = np.ptp(allv, 0).max() / 2
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.view_init(el, az)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"CAD 10839000_01 elev {el} azim {az} (red = cylindrical faces: bends + hole walls)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "a3_cad_views.png")
print("saved")
