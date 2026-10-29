"""Render a STEP solid (tessellated, read-only) with its V10 bend axes and the
lenient coaxial bend candidates, 4 views, to a PNG for visual cross-check.
Usage: render_cad.py <step> <raw_json> <out.png>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def tessellate(path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    r = STEPControl_Reader(); r.ReadFile(str(path)); r.TransferRoots(); shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)
    tris = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current()); loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(f, loc)
        if t is not None:
            tr = loc.Transformation()
            pts = [t.Node(i).Transformed(tr) for i in range(1, t.NbNodes() + 1)]
            P = np.array([[p.X(), p.Y(), p.Z()] for p in pts])
            for i in range(1, t.NbTriangles() + 1):
                a, b, c = t.Triangle(i).Get(); tris.append(P[[a - 1, b - 1, c - 1]])
        ex.Next()
    return np.array(tris)


def main():
    step, raw, out = sys.argv[1], sys.argv[2], sys.argv[3]
    R = json.loads(Path(raw).read_text())
    T = tessellate(step)
    allp = T.reshape(-1, 3); lo, hi = allp.min(0), allp.max(0); ctr = (lo + hi) / 2; rad = (hi - lo).max() / 2
    views = [(30, -60, "iso"), (90, -90, "top"), (0, -90, "front"), (0, 0, "right"), (-30, 120, "iso_under"), (30, 120, "iso_back")]
    fig = plt.figure(figsize=(24, 16))
    axes = R.get("v10_bend_axes", [])
    cols = plt.cm.tab20(np.linspace(0, 1, max(1, len(axes))))
    for k, (el, az, name) in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        pc = Poly3DCollection(T, facecolor=(0.75, 0.75, 0.78, 0.35), edgecolor=(0.3, 0.3, 0.3, 0.08), linewidth=0.1)
        ax.add_collection3d(pc)
        for i, a in enumerate(axes):
            s, e = np.array(a["line_start"]), np.array(a["line_end"])
            ax.plot(*zip(s, e), color=cols[i], lw=3)
            m = (s + e) / 2
            ax.text(*m, f"C{i+1} {a['angular_sweeps_deg'][0]:.0f}°/{a['axial_span_mm']:.0f}", color=cols[i], fontsize=9, weight="bold")
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad); ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.view_init(elev=el, azim=az); ax.set_title(f"{Path(step).name} | {name} | V10 bends={len(axes)}"); ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(); fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
