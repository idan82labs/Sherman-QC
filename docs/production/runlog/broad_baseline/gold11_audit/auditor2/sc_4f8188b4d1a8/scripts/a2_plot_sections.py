"""Auditor-2 cross-section plots perpendicular to each gold line (own fit, not the shared renderer).

usage: a2_plot_sections.py BID [fractions...] [--half H] [--zoom Z] [--wide W]
Writes OUT/xsec_a2_<BID>.png.  Points are coloured by in-plane normal direction.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a2_common import OUT, get_scan, gold_lines, perp_basis, unit
from a2_section import analyse_section, cyl_select, frame2d

args = sys.argv[1:]
bid = args[0]
half = 0.5
zoom = 7.0
wide = 25.0
fr = []
i = 1
while i < len(args):
    if args[i] == "--half":
        half = float(args[i + 1]); i += 2
    elif args[i] == "--zoom":
        zoom = float(args[i + 1]); i += 2
    elif args[i] == "--wide":
        wide = float(args[i + 1]); i += 2
    else:
        fr.append(float(args[i])); i += 1
if not fr:
    fr = [0.05, 0.25, 0.5, 0.75, 0.95]

xyz, nrm, faces = get_scan()
G = gold_lines()
a, b = G[bid]
L = float(np.linalg.norm(b - a))
idx, t, d, _ = cyl_select(xyz, nrm, a, b, radius=wide * 1.5, ext=30.0)
e1, e2 = perp_basis(d)

fig, axes = plt.subplots(2, len(fr), figsize=(4.2 * len(fr), 8.6))
if len(fr) == 1:
    axes = axes.reshape(2, 1)
for j, f in enumerate(fr):
    s = f * L if f <= 1.5 else f
    k = np.abs(t - s) < half
    q, m, an = frame2d(xyz, nrm, idx[k], a, d, e1, e2, s)
    r = analyse_section(q, m, an, rmin=3.5, rmax=13.0)
    ang = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    for row, lim in ((0, wide), (1, zoom)):
        ax = axes[row, j]
        ax.scatter(q[:, 0], q[:, 1], s=1.5 if row == 0 else 4, c=ang, cmap="hsv", vmin=-180, vmax=180,
                   linewidths=0)
        ax.plot([0], [0], marker="*", ms=14, color="#1a9c3a", mec="k", zorder=5, label=f"gold {bid}")
        circ = plt.Circle((0, 0), 4.0, fill=False, ls=":", color="#1a9c3a")
        ax.add_patch(circ)
        if r.get("ok"):
            Xo = np.asarray(r["Xo"])
            ax.plot(*Xo, "x", color="tab:blue", ms=9, mew=2, label="outer virtual sharp")
            if r.get("Xi") is not None:
                ax.plot(*np.asarray(r["Xi"]), "x", color="tab:purple", ms=9, mew=2, label="inner virtual sharp")
            if r.get("apex") is not None:
                ax.plot(*np.asarray(r["apex"]), "D", color="cyan", mec="k", ms=6, label="outer apex (scan)")
            if r.get("inner_apex") is not None:
                ax.plot(*np.asarray(r["inner_apex"]), "s", color="yellow", mec="k", ms=5, label="inner apex (scan)")
            bis = np.asarray(r["bisector"])
            ax.plot([Xo[0] - 2 * bis[0], Xo[0] + 8 * bis[0]], [Xo[1] - 2 * bis[1], Xo[1] + 8 * bis[1]], "--",
                    color="gray", lw=0.7)
            for key, col in (("tA", "tab:blue"), ("tB", "tab:orange")):
                tv = np.asarray(r[key])
                ax.plot([Xo[0], Xo[0] + 30 * tv[0]], [Xo[1], Xo[1] + 30 * tv[1]], "-", color=col, lw=0.5, alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f"{bid} s={s:.1f} mm ({f:.2f}) slab +-{half}")
        else:
            if r.get("ok") and r.get("apex") is not None:
                ax.set_title(f"bend {r['bend_angle_deg']:.0f}deg  gold-apex bis {r['gold_minus_apex_bis']:.2f} "
                             f"lat {r['gold_minus_apex_lat']:.2f}\n|g-apex| {r['gold_to_apex']:.2f}  |g-Xo| "
                             f"{r['gold_to_Xo']:.2f}  near {r['gold_to_nearest_scan']:.2f}", fontsize=8)
            elif r.get("ok"):
                ax.set_title(f"bend {r['bend_angle_deg']:.0f}deg  NO outer-apex material on bisector\n|g-Xo| "
                             f"{r['gold_to_Xo']:.2f}  near {r['gold_to_nearest_scan']:.2f}", fontsize=8)
            else:
                ax.set_title(f"fit: {r.get('why')}", fontsize=8)
            if j == 0:
                ax.legend(fontsize=6, loc="lower left")
fig.suptitle(f"auditor2 sc_4f8188b4d1a8 {bid} L={L:.1f} mm  (frame: e1/e2 perpendicular to gold; colour = normal angle)",
             fontsize=10)
fig.tight_layout()
outp = OUT / f"xsec_a2_{bid}.png"
fig.savefig(outp, dpi=95)
print(outp)
