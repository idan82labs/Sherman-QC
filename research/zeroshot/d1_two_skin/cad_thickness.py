"""Reference sheet thickness from a STEP B-rep (validation only; never fed to the scan path).

Independent of the scan method: planar faces are read exactly from the B-rep (OCP), and every
pair of antiparallel planar faces whose projections overlap gives a candidate wall distance.
t_CAD = the overlap-area-weighted mode of those distances. Cylindrical bend faces give a second
check: coaxial cylinder pairs (same axis, radii differing by d) vote with d.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _faces(path: str):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"cannot read STEP {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    planes, cyls = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        exp.Next()
        ad = BRepAdaptor_Surface(face)
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        c = props.CentreOfMass()
        centre = np.array([c.X(), c.Y(), c.Z()])
        st = ad.GetType()
        if st == GeomAbs_Plane:
            pl = ad.Plane()
            d = pl.Axis().Direction()
            n = np.array([d.X(), d.Y(), d.Z()])
            if not pl.Direct():
                n = -n
            if face.Orientation() == TopAbs_REVERSED:
                n = -n
            # bounding box for the overlap test
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib
            box = Bnd_Box()
            BRepBndLib.Add_s(face, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            planes.append({"n": n, "centre": centre, "area": area,
                           "lo": np.array([xmin, ymin, zmin]), "hi": np.array([xmax, ymax, zmax])})
        elif st == GeomAbs_Cylinder:
            cy = ad.Cylinder()
            ax = cy.Axis()
            loc, dr = ax.Location(), ax.Direction()
            cyls.append({"p": np.array([loc.X(), loc.Y(), loc.Z()]), "u": np.array([dr.X(), dr.Y(), dr.Z()]),
                         "r": cy.Radius(), "area": area})
    return planes, cyls


def cad_thickness(path: str, dmin: float = 0.2, dmax: float = 15.0) -> dict[str, Any]:
    planes, cyls = _faces(path)
    d_all, w_all = [], []
    for i in range(len(planes)):
        a = planes[i]
        for j in range(i + 1, len(planes)):
            b = planes[j]
            if float(a["n"] @ b["n"]) > -0.9995:
                continue
            d = abs(float((b["centre"] - a["centre"]) @ a["n"]))
            if not (dmin < d < dmax):
                continue
            # overlap of the two faces' boxes after removing the normal direction (rough lateral overlap)
            lo = np.maximum(a["lo"], b["lo"])
            hi = np.minimum(a["hi"], b["hi"])
            ext = hi - lo
            k = int(np.argmax(np.abs(a["n"])))       # the box axis closest to the face normal
            others = [ext[x] for x in range(3) if x != k]
            if min(others) <= 0:
                continue
            ov = others[0] * others[1]
            if ov <= 0:
                continue
            d_all.append(d)
            w_all.append(min(ov, a["area"], b["area"]))
    out: dict[str, Any] = {"planar_faces": len(planes), "cylindrical_faces": len(cyls)}
    if d_all:
        d_all, w_all = np.array(d_all), np.array(w_all)
        bins = np.round(d_all, 2)
        uniq = np.unique(bins)
        score = np.array([w_all[np.abs(d_all - u) < 0.03].sum() for u in uniq])
        t = float(np.average(d_all[np.abs(d_all - uniq[np.argmax(score)]) < 0.03],
                             weights=w_all[np.abs(d_all - uniq[np.argmax(score)]) < 0.03]))
        out.update(t_planar_mm=t, planar_vote_share=float(score.max() / score.sum()),
                   planar_pairs=int(len(d_all)))
    # coaxial cylinder pairs (bend inner/outer arcs)
    cd, cw = [], []
    for i in range(len(cyls)):
        for j in range(i + 1, len(cyls)):
            a, b = cyls[i], cyls[j]
            if abs(float(a["u"] @ b["u"])) < 0.9999:
                continue
            dp = b["p"] - a["p"]
            off = np.linalg.norm(dp - (dp @ a["u"]) * a["u"])
            if off > 1e-3:
                continue
            d = abs(a["r"] - b["r"])
            if dmin < d < dmax:
                cd.append(d)
                cw.append(min(a["area"], b["area"]))
    if cd:
        cd, cw = np.array(cd), np.array(cw)
        u = np.unique(np.round(cd, 2))
        sc = np.array([cw[np.abs(cd - x) < 0.03].sum() for x in u])
        out.update(t_cylinder_mm=float(u[np.argmax(sc)]), cylinder_vote_share=float(sc.max() / sc.sum()),
                   cylinder_pairs=int(len(cd)))
    t_p, t_c = out.get("t_planar_mm"), out.get("t_cylinder_mm")
    out["t_mm"] = t_p if t_p is not None else t_c
    out["planar_cylinder_agree"] = (None if t_p is None or t_c is None else bool(abs(t_p - t_c) < 0.05))
    return out


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(cad_thickness(sys.argv[1]), indent=1, default=float))
