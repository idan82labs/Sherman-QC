"""Elevation views of both walls (u-w), end views (v-w) and an oblique-normal (crease) map."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1_common import load_all, gold_lines, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T
g = gold_lines()
gp = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in g.items()}

fig, axs = plt.subplots(3, 1, figsize=(22, 17))
for ax, (lab, m) in zip(axs[:2], [("wall 1 side (v < -135), elevation u-w", p[:, 1] < -135),
                                   ("wall 2 side (v > 135), elevation u-w", p[:, 1] > 135)]):
    idx = np.where(m)[0]
    ax.scatter(p[idx, 0], p[idx, 2], s=0.2, c=np.abs(n[idx, 1]), cmap="viridis")
    for k, (s, e) in gp.items():
        ax.plot([s[0], e[0]], [s[2], e[2]], "-", color="r", lw=1.5)
        mm = (s + e) / 2
        ax.text(mm[0], mm[2] + 2, k, color="r", fontsize=11)
    ax.set_title(lab + "  (colour |n_v|: yellow = wall faces)  gold in red")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlim(-185, 185)
# crease map: points whose normal is >12 deg from every frame axis, plan view, coloured by w
ax = axs[2]
a = np.abs(n)
obl = a.max(1) < np.cos(np.radians(12))
idx = np.where(obl)[0]
sc = ax.scatter(p[idx, 0], p[idx, 1], s=0.3, c=np.clip(p[idx, 2], -42, 2), cmap="turbo")
plt.colorbar(sc, ax=ax, label="w (mm)")
for k, (s, e) in gp.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], "-", color="k", lw=1.0)
    mm = (s + e) / 2
    ax.text(mm[0], mm[1], k, fontsize=11, bbox=dict(fc="yellow", alpha=0.7))
ax.set_title("plan: points with normal > 12 deg from every frame axis (bends, fillets, rounded cut edges, holes), colour = w")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a1_elevations_crease.png", dpi=70)
print(obl.sum())
