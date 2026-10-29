"""Plan-view maps in the part frame: height (w) and normal class, with gold lines."""
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

fig, axs = plt.subplots(1, 2, figsize=(22, 10))
# panel 1: points coloured by w (height); subsample
idx = np.arange(0, len(p), 3)
sc = axs[0].scatter(p[idx, 0], p[idx, 1], c=np.clip(p[idx, 2], -6, 2), s=0.2, cmap="turbo")
plt.colorbar(sc, ax=axs[0], label="w (mm) clipped -6..2")
axs[0].set_title("plan (u,v), colour = height w")
# panel 2: normal class
cls = np.full(len(p), 3)
cls[np.abs(n[:, 2]) > 0.97] = 0     # plate-parallel
cls[np.abs(n[:, 1]) > 0.97] = 1     # faces normal to v (walls along u)
cls[np.abs(n[:, 0]) > 0.97] = 2     # faces normal to u
cols = np.array(["#cccccc", "#1f77b4", "#2ca02c", "#d62728"])
for k in (0, 1, 2, 3):
    m = (cls == k)
    mi = np.where(m)[0][:: (6 if k == 0 else 1)]
    axs[1].scatter(p[mi, 0], p[mi, 1], s=0.3 if k else 0.1, c=cols[k], label=["plate-parallel", "n||v", "n||u", "oblique"][k])
axs[1].legend(markerscale=20, loc="upper right")
axs[1].set_title("plan (u,v), colour = normal class (oblique = bends/curvature/edges)")
for ax in axs:
    for k, (s, e) in gp.items():
        ax.plot([s[0], e[0]], [s[1], e[1]], "-", color="k", lw=1.2)
        m = (s + e) / 2
        ax.text(m[0], m[1], k, fontsize=11, color="k", bbox=dict(fc="yellow", alpha=0.7))
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("u (mm)  [along M01/M02]"); ax.set_ylabel("v (mm)  [along M03/M04]")
fig.tight_layout()
fig.savefig(OUT / "a1_plan_maps.png", dpi=80)
print("ok")
