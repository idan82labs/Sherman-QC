"""Overview renders in the plate frame: top-view height map with gold, and full sections."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3common import load_mesh, load_gold, slice_mesh, OUT

V, N, F = load_mesh()
G = load_gold()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
X = (V - o) @ R.T
TC = V[F].mean(1)
GOLD_C = {"M01": "#1a9c3a", "M02": "#1a9c3a", "M03": "#d62728", "M04": "#ff7f0e", "M05": "#d62728",
          "M06": "#ff7f0e", "M07": "#1f77b4", "M08": "#9467bd"}
Gf = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in G.items()}

# --- top view height map
rng = np.random.default_rng(0)
sel = rng.choice(len(X), 250000, replace=False)
fig, ax = plt.subplots(figsize=(13, 11), dpi=100)
sc = ax.scatter(X[sel, 0], X[sel, 1], c=np.clip(X[sel, 2], -20, 15), s=0.3, cmap="coolwarm", linewidths=0)
plt.colorbar(sc, ax=ax, label="h above plate mid-surface (mm), clipped -20..15")
for k, (s, e) in Gf.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], color=GOLD_C[k], lw=1.4)
    ax.text(e[0] + 2, e[1], f"{k} h={0.5*(s[2]+e[2]):.1f}", color=GOLD_C[k], fontsize=8, va="center")
ax.set_aspect("equal")
ax.set_xlabel("a (mm, along M01 direction)")
ax.set_ylabel("b (mm)")
ax.set_title("sc_f2475286312b top view in plate frame (colour = h); gold lines with mean h")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a3_topview_height.png")
plt.close(fig)

# --- full sections at a = const (b-h profile) for folds parallel to a
def section(axis_vec, val, other):
    org = o + val * axis_vec
    segs = slice_mesh(V, F, org, axis_vec)
    return segs

stations_a = [-140, -100, -60, -20, 0, 20, 40, 60, 80, 100, 120]
fig, axes = plt.subplots(len(stations_a), 1, figsize=(16, 2.2 * len(stations_a)), dpi=90)
for ax, av in zip(axes, stations_a):
    segs = section(a, av, b)
    if len(segs):
        P = (segs.reshape(-1, 3) - o) @ R.T
        P = P.reshape(-1, 2, 3)
        for sgm in P:
            ax.plot(sgm[:, 1], sgm[:, 2], color="k", lw=0.6)
    for k, (s, e) in Gf.items():
        if abs(e[0] - s[0]) < 1e-6:
            continue
        t = (av - s[0]) / (e[0] - s[0])
        if -0.05 <= t <= 1.05:
            p = s + t * (e - s)
            ax.plot(p[1], p[2], "o", color=GOLD_C[k], ms=5)
            ax.annotate(k, (p[1], p[2]), fontsize=7, color=GOLD_C[k], xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"section a = {av} mm (b horizontal, h vertical)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-25, 18)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(OUT / "a3_sections_a_const.png")
plt.close(fig)

stations_b = [-110, -100, -80, -40, 0, 40, 80, 100, 110]
fig, axes = plt.subplots(len(stations_b), 1, figsize=(16, 3.2 * len(stations_b)), dpi=90)
for ax, bv in zip(axes, stations_b):
    segs = section(b, bv, a)
    if len(segs):
        P = (segs.reshape(-1, 3) - o) @ R.T
        P = P.reshape(-1, 2, 3)
        for sgm in P:
            ax.plot(sgm[:, 0], sgm[:, 2], color="k", lw=0.6)
    for k, (s, e) in Gf.items():
        if abs(e[1] - s[1]) < 1e-6:
            continue
        t = (bv - s[1]) / (e[1] - s[1])
        if -0.05 <= t <= 1.05:
            p = s + t * (e - s)
            ax.plot(p[0], p[2], "o", color=GOLD_C[k], ms=5)
            ax.annotate(k, (p[0], p[2]), fontsize=7, color=GOLD_C[k], xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"section b = {bv} mm (a horizontal, h vertical)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-155, 135)
    ax.set_ylim(-70, 18)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(OUT / "a3_sections_b_const.png")
plt.close(fig)
print("ok")
