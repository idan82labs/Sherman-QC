"""Part-frame planar slab sections (auditor 2).

usage: a2_partsection.py NAME AXIS VALUES HALF ILIM JLIM
  AXIS  = L | W | N  (slab normal); remaining two axes are plotted (in order L,W,N).
  VALUES= comma list of slab positions; HALF = slab half-thickness (mm)
  ILIM/JLIM = 'lo:hi' plot limits for the two in-plane axes.
Gold lines crossing (or lying in) the slab are drawn: crossing -> star, in-plane -> line.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a2_common import OUT, get_scan, gold_lines
from a2_frame import R, to_part

name, axis, values, half, ilim, jlim = sys.argv[1:7]
ax_i = "LWN".index(axis)
others = [k for k in range(3) if k != ax_i]
vals = [float(v) for v in values.split(",")]
half = float(half)
ilo, ihi = (float(x) for x in ilim.split(":"))
jlo, jhi = (float(x) for x in jlim.split(":"))
xyz, nrm, faces = get_scan()
P = to_part(xyz)
Nn = nrm @ R.T
G = gold_lines()
gl = {k: (to_part(a), to_part(b)) for k, (a, b) in G.items()}
fig, axes = plt.subplots(1, len(vals), figsize=(5.2 * len(vals), 5.6))
axes = np.atleast_1d(axes)
for ax, v in zip(axes, vals):
    k = (np.abs(P[:, ax_i] - v) < half) & (P[:, others[0]] > ilo - 1) & (P[:, others[0]] < ihi + 1) \
        & (P[:, others[1]] > jlo - 1) & (P[:, others[1]] < jhi + 1)
    q = P[k][:, others]
    m = Nn[k][:, others]
    ang = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    ax.scatter(q[:, 0], q[:, 1], s=3, c=ang, cmap="hsv", vmin=-180, vmax=180, linewidths=0)
    for bid, (pa, pb) in gl.items():
        da, db = pa[ax_i] - v, pb[ax_i] - v
        if da * db < 0:  # crosses
            f = da / (da - db)
            pc = pa + f * (pb - pa)
            ax.plot(pc[others[0]], pc[others[1]], "*", ms=13, color="#1a9c3a", mec="k", zorder=5)
            ax.text(pc[others[0]] + 0.3, pc[others[1]] + 0.3, bid, color="#1a9c3a", fontsize=9)
        elif abs(da) < 2.0 and abs(db) < 2.0:
            ax.plot([pa[others[0]], pb[others[0]]], [pa[others[1]], pb[others[1]]], "-", color="#1a9c3a", lw=2)
            ax.text(pa[others[0]], pa[others[1]], bid, color="#1a9c3a", fontsize=9)
    ax.set_xlim(ilo, ihi)
    ax.set_ylim(jlo, jhi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("LWN"[others[0]])
    ax.set_ylabel("LWN"[others[1]])
    ax.set_title(f"{axis}={v:g} +-{half}  (n={int(k.sum())})", fontsize=9)
fig.suptitle(f"auditor2 sc_4f8188b4d1a8 part-frame slab sections: {name}", fontsize=10)
fig.tight_layout()
outp = OUT / f"a2_sec_{name}.png"
fig.savefig(outp, dpi=90)
print(outp)
