"""Corner + joggle close-ups (part frame) to look for formed features without gold lines."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *

Q = np.load(SCR / "Q.npy"); NQ = np.load(SCR / "NQ.npy"); C0 = np.load(SCR / "center.npy")
gold = [(b, to_frame(a, C0), to_frame(e, C0)) for b, a, e in load_gold()]
corners = {"L-,S-": (-149, -99), "L-,S+": (-149, 107), "L+,S-": (164, -99), "L+,S+": (164, 107)}
fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=80)
for j, (nm, (lc, scn)) in enumerate(corners.items()):
    for i, hh in enumerate((14.0, 0.0, -14.0)):
        ax = axes[i, j]
        m = (np.abs(Q[:, 2] - hh) < 0.5) & (np.abs(Q[:, 0] - lc) < 14) & (np.abs(Q[:, 1] - scn) < 14)
        ax.scatter(Q[m, 0], Q[m, 1], s=1.5, c=np.abs(NQ[m, 0]), cmap="coolwarm", vmin=0, vmax=1, linewidths=0)
        for b, a, e in gold:
            d = e - a
            if abs(d[2]) > 1e-6:
                t = (hh - a[2]) / d[2]
            else:
                t = None
            # project gold endpoints near the corner
            for p in (a, e):
                if abs(p[0] - lc) < 14 and abs(p[1] - scn) < 14:
                    ax.plot(p[0], p[1], "g*", ms=12)
                    ax.annotate(f"{b} end H={p[2]:.1f}", (p[0], p[1]), fontsize=8, color="g")
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"corner {nm}  slice H={hh:+.0f}+-0.5 (colour |n_L|)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "a3_corner_slices.png"); plt.close(fig)

# corner seam elevation: points within 6 mm of each vertical corner, viewed from outside diagonal
fig, axes = plt.subplots(1, 4, figsize=(24, 8), dpi=80)
for ax, (nm, (lc, scn)) in zip(axes, corners.items()):
    m = (np.abs(Q[:, 0] - lc) < 12) & (np.abs(Q[:, 1] - scn) < 12)
    diag = unit([np.sign(-lc), np.sign(-scn)])  # inward diagonal in L-S
    x = (Q[m, 0] - lc) * (-diag[1]) + (Q[m, 1] - scn) * diag[0]
    dep = (Q[m, 0] - lc) * diag[0] + (Q[m, 1] - scn) * diag[1]
    o = np.argsort(-dep)
    ax.scatter(x[o], Q[m, 2][o], s=1.0, c=dep[o], cmap="viridis", linewidths=0)
    for b, a, e in gold:
        for p in (a, e):
            if abs(p[0] - lc) < 12 and abs(p[1] - scn) < 12:
                xx = (p[0] - lc) * (-diag[1]) + (p[1] - scn) * diag[0]
                ax.plot(xx, p[2], "r*", ms=12); ax.annotate(b, (xx, p[2]), color="r", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title(f"corner {nm}: view along the inward diagonal (colour = depth)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "a3_corner_elevations.png"); plt.close(fig)

# joggle close-up sections
fig, axes = plt.subplots(2, 3, figsize=(21, 9), dpi=85)
from a3_stations import section
for i, side in enumerate(("Sminus", "Splus")):
    for j, a0 in enumerate((-100.0, 0.0, 120.0)):
        ax = axes[i, j]
        u, h, nu, nh = section(side, a0)
        m = (h < -13) & (u > -3) & (u < 40)
        ax.scatter(u[m], h[m], s=2, c=nh[m], cmap="coolwarm", vmin=-1, vmax=1, linewidths=0)
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-3, 40); ax.set_ylim(-24, -12)
        ax.set_title(f"{side} bottom flange, L={a0:+.0f} (u from wall outer face; colour n_h)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "a3_joggle_sections.png"); plt.close(fig)
print("ok")
