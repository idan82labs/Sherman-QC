import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_08686dc1d5d4")
P = np.load(W / "plateframe.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
fig, axs = plt.subplots(2, 1, figsize=(24, 9))
for ax, side, yw, s in ((axs[0], "M01", 152.86, 1), (axs[1], "M02", -158.66, -1)):
    m = (np.abs(P[:, 1] - yw) < 3.5) & (P[:, 2] > -8)
    dev = s * (P[m, 1] - yw)
    sc = ax.scatter(P[m, 0], P[m, 2], c=np.clip(dev, -2, 2), s=0.3, cmap="coolwarm")
    a, b = G[side]
    ax.plot([a[0], b[0]], [a[2], b[2]], "-", color="#1a9c3a", lw=2, label=f"gold {side}")
    ax.axvline(-100, color="k", ls=":"); ax.axvline(101, color="k", ls=":")
    ax.set_title(f"{side}-side wall face-on (x along wall, z up); colour = outward deviation from wall mid-plane; dotted = wall x-extent"); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlim(-180, 180); ax.set_ylim(-9, 40)
plt.colorbar(sc, ax=axs, shrink=0.6)
plt.savefig(OUT / "a2_walls_faceon.png", dpi=70, bbox_inches="tight")
# lip and wall-top sections at several x
xs = [-95, -60, -20, 20, 60, 95]
fig, axs = plt.subplots(2, len(xs), figsize=(4 * len(xs), 8))
for i, (side, yw, s) in enumerate((("M01", 152.86, 1), ("M02", -158.66, -1))):
    for j, x0 in enumerate(xs):
        ax = axs[i, j]
        m = (np.abs(P[:, 0] - x0) < 0.5) & (np.abs(P[:, 1] - yw) < 20)
        u = s * (yw - P[m, 1])
        ax.scatter(u, P[m, 2], s=1, c="k")
        a, b = G[side]
        t = (x0 - a[0]) / (b[0] - a[0]); gp = a + t * (b - a)
        ax.plot(s * (yw - gp[1]), gp[2], "*", color="#1a9c3a", ms=12, mec="k")
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlim(-6, 18); ax.set_ylim(-9, 40)
        ax.set_title(f"{side} fold+wall+lip @x={x0}\n(u: + toward plate interior)", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "a2_wall_lip_sections.png", dpi=70)
print("ok")
