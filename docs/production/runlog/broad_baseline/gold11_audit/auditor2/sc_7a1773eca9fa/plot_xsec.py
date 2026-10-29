#!/usr/bin/env python3
"""Auditor 2 - plot raw cross-sections perpendicular to the TRUE bend axis (from stations_<id>.json).

usage: plot_xsec.py NPZ BEND_ID lam1,lam2,... [half_width_mm] [zoom_mm]
Points in the slab are coloured by their oriented vertex normal relative to the local bisector:
  blue = outward normal points away from the concave side (outer skin),
  orange = toward the concave side (inner skin), grey = edge / end faces.
Markers: gold point (green star), outer apex (cyan diamond), inner apex (magenta diamond),
outer virtual sharp Xo (blue x), inner virtual sharp Xi (purple x); dotted circle = 4 mm around gold.
Frame: origin = outer apex, +y = bisector toward the concave side, +x = toward flange A.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
npz, bid, lams = sys.argv[1], sys.argv[2], [float(v) for v in sys.argv[3].split(",")]
hw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
zoom = float(sys.argv[5]) if len(sys.argv) > 5 else 14.0
tag = sys.argv[6] if len(sys.argv) > 6 else ""
d = np.load(npz)
P, N = d["P"], d["N"]
R = json.loads((OUT / f"stations_{bid}.json").read_text())
axis = np.array(R["axis"])
e1, e2 = np.array(R["e1"]), np.array(R["e2"])
a, b = np.array(R["gold_start"]), np.array(R["gold_end"])
sa, sb = R["s_gold"]
st = {round(x["s"], 3): x for x in R["stations"]}
sP = P @ axis
n = len(lams)
cols = min(n, 3)
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.2 * rows), squeeze=False)
for k, lam in enumerate(lams):
    ax = axs[k // cols][k % cols]
    s = sa + lam * (sb - sa)
    s_r = float(np.round(s))
    rec = st.get(round(s_r, 3))
    m = np.abs(sP - s) <= hw
    G3 = a + lam * (b - a)
    g2 = np.array([G3 @ e1, G3 @ e2])
    q = np.c_[P[m] @ e1, P[m] @ e2]
    nn = np.c_[N[m] @ e1, N[m] @ e2]
    nax = N[m] @ axis
    if rec is None or not rec.get("ok"):
        # plot in a gold-centred frame
        org = g2
        bis = np.array([0.0, 1.0])
        ex = np.array([1.0, 0.0])
        ttl = f"{bid} lam={lam:.2f} s={s:.1f} (no fit: {None if rec is None else rec.get('why')})"
    else:
        org = np.array(rec["apex"])
        bis = np.array(rec["bis"])
        ex = np.array(rec["ex"])
        ttl = (f"{bid} lam={lam:.2f} s={s:.1f} bend={rec['bend_deg']:.0f}deg t={rec['thickness'] or float('nan'):.2f}\n"
               f"gold-apex: along-bis {rec['gold_minus_apex_bis']:+.2f}  across {rec['gold_minus_apex_across']:+.2f}"
               f"  |d|={rec['d_gold_apex']:.2f} mm")
    X = np.c_[(q - org) @ ex, (q - org) @ bis]
    close = np.linalg.norm(X, axis=1) <= zoom * 1.5
    nb = nn @ bis
    edge = np.abs(nax) >= 0.5
    c = np.where(edge, "0.6", np.where(nb < -0.2, "tab:blue", np.where(nb > 0.2, "tab:orange", "0.3")))
    ax.scatter(X[close, 0], X[close, 1], s=2.5, c=c[close], linewidths=0)
    gx = np.array([(g2 - org) @ ex, (g2 - org) @ bis])
    ax.plot(*gx, marker="*", ms=16, color="#1a9c3a", mec="k", zorder=5, label="gold")
    ax.add_patch(plt.Circle(gx, 4.0, fill=False, ls=":", color="#1a9c3a"))
    if rec is not None and rec.get("ok"):
        ax.plot(0, 0, marker="D", ms=8, color="cyan", mec="k", zorder=5, label="outer apex")
        for key, mk, colr, lab in (("iapex", "D", "magenta", "inner apex"), ("Xo", "x", "tab:blue", "outer sharp"),
                                   ("Xi", "x", "purple", "inner sharp")):
            if key in rec:
                p = np.array(rec[key])
                ax.plot((p - org) @ ex, (p - org) @ bis, marker=mk, ms=10, mew=2.5, color=colr, mec=colr if mk == "x" else "k",
                        zorder=5, label=lab)
        if "C_o" in rec:
            C = np.array(rec["C_o"])
            ax.add_patch(plt.Circle(((C - org) @ ex, (C - org) @ bis), rec["R_o"], fill=False, color="tab:brown", lw=0.6))
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom * 0.6, zoom * 1.4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_title(ttl, fontsize=9)
    if k == 0:
        ax.legend(fontsize=7, loc="lower left")
for k in range(n, rows * cols):
    axs[k // cols][k % cols].axis("off")
fig.suptitle(f"sc_7a1773eca9fa {bid} - auditor2 raw slab +-{hw} mm perpendicular to fitted axis "
             f"(gold-axis angle {R['angle_gold_vs_axis_deg']:.2f} deg). blue=outer-skin normals, orange=inner-skin, grey=edges",
             fontsize=10)
fig.tight_layout()
outp = OUT / f"xsec_{bid}{tag}.png"
fig.savefig(outp, dpi=110)
print(outp)
