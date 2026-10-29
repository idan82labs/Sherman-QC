"""Sections of an E3 registered scan (registered_scan.ply, datum frame) with the CAD stage mesh, at given stations.

usage: section.py E3_OUT_DIR AXIS(x|y|z) STATIONS(comma) H_AXIS V_AXIS H_LO H_HI V_LO V_HI OUT.png [--slab 0.6] [--mesh display/final_mesh.ply]
Points with |coord_AXIS - station| < slab are plotted in (H_AXIS, V_AXIS). Measured / nominal bend polylines (bend_geometry.json)
crossing the section are marked.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from domains.production.engine.inspect.scan_io import read_scan

d = Path(sys.argv[1]); ax_name = sys.argv[2]; stations = [float(s) for s in sys.argv[3].split(",")]
hn, vn = sys.argv[4], sys.argv[5]
hlo, hhi, vlo, vhi = map(float, sys.argv[6:10]); out = sys.argv[10]
slab = float(sys.argv[sys.argv.index("--slab") + 1]) if "--slab" in sys.argv else 0.6
meshp = sys.argv[sys.argv.index("--mesh") + 1] if "--mesh" in sys.argv else "display/stage_mesh.ply"
I = {"x": 0, "y": 1, "z": 2}
ai, hi_, vi = I[ax_name], I[hn], I[vn]
pts = read_scan(d / "registered_scan.ply").points
mp = d / meshp
mesh = read_scan(mp).points if mp.exists() else None
geo = json.loads((d / "bend_geometry.json").read_text())
n = len(stations)
fig, axs = plt.subplots(1, n, figsize=(7 * n, 7))
axs = np.atleast_1d(axs)
for k, s0 in enumerate(stations):
    a = axs[k]
    sel = (np.abs(pts[:, ai] - s0) < slab) & (pts[:, hi_] > hlo) & (pts[:, hi_] < hhi) & (pts[:, vi] > vlo) & (pts[:, vi] < vhi)
    a.scatter(pts[sel, hi_], pts[sel, vi], s=0.5, c="k", label=f"scan ({int(sel.sum())})")
    if mesh is not None:
        ms = (np.abs(mesh[:, ai] - s0) < 2.0) & (mesh[:, hi_] > hlo) & (mesh[:, hi_] < hhi) & (mesh[:, vi] > vlo) & (mesh[:, vi] < vhi)
        a.scatter(mesh[ms, hi_], mesh[ms, vi], s=6, c="c", marker="x", label="CAD mesh verts")
    for b in geo["bends"]:
        for key, col, mk in (("measured_polyline_mm", "r", "o"), ("nominal_polyline_mm", "g", "s")):
            pl = b.get(key)
            if not pl or len(pl) < 2:
                continue
            p, q = np.asarray(pl[0]), np.asarray(pl[-1])
            if abs(q[ai] - p[ai]) < 1e-6:
                continue
            tt = (s0 - p[ai]) / (q[ai] - p[ai])
            if tt < -0.05 or tt > 1.05:
                continue
            x = p + tt * (q - p)
            if hlo < x[hi_] < hhi and vlo < x[vi] < vhi:
                a.plot(x[hi_], x[vi], col + mk, ms=8, mfc="none")
                a.annotate(b["label"] + ("m" if col == "r" else "n"), (x[hi_], x[vi]), color=col, fontsize=9)
    a.set_xlim(hlo, hhi); a.set_ylim(vlo, vhi); a.set_aspect("equal"); a.grid(True, lw=0.3)
    a.set_title(f"{d.parent.name} {ax_name}={s0} ({hn},{vn})")
fig.tight_layout(); fig.savefig(out, dpi=80)
print("wrote", out)
