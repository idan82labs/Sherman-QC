#!/usr/bin/env python3
"""Whole-part crease map (normal variation within 2.5 mm) in the part PCA frame, gold lines overlaid,
plus a plate-label map from normal clusters. For the missing-bend check. Read-only."""
import sys
sys.dont_write_bytecode = True
import json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_sections as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V, N = A.load()
rng = np.random.default_rng(0)
sub = rng.choice(len(V), 220000, replace=False)
P, NP = V[sub], N[sub]
tree = cKDTree(P)
nb = tree.query_ball_point(P, 2.5)
crease = np.zeros(len(P))
for i, idx in enumerate(nb):
    if len(idx) < 4:
        continue
    m = np.abs(NP[idx] @ NP[i])
    crease[i] = np.degrees(np.arccos(np.clip(np.percentile(m, 10), -1, 1)))
c0 = V.mean(0); _u, _s, vt = np.linalg.svd(V[::20] - c0, full_matrices=False)
for k in range(3):
    if vt[k][np.argmax(np.abs(vt[k]))] < 0:
        vt[k] = -vt[k]
Q = (P - c0) @ vt.T
labels = json.loads(A.LABELS.read_text())
gl = [(b["bend_id"], (np.asarray(b["line_start"]) - c0) @ vt.T, (np.asarray(b["line_end"]) - c0) @ vt.T) for b in labels["manual_bends"]]
np.savez_compressed("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/crease.npz", Q=Q, crease=crease, vt=vt, c0=c0)
views = [("top: PC1 x PC2 (looking along base normal)", 0, 1, 2), ("front: PC1 x PC3", 0, 2, 1), ("side: PC2 x PC3", 1, 2, 0)]
fig, axes = plt.subplots(3, 1, figsize=(16, 21), dpi=80)
for ax, (name, i, j, k) in zip(axes, views):
    o = np.argsort(crease)
    sc = ax.scatter(Q[o, i], Q[o, j], c=np.clip(crease[o], 0, 40), s=0.6, cmap="inferno_r", vmin=0, vmax=40, lw=0, rasterized=True)
    for bid, s, e in gl:
        ax.plot([s[i], e[i]], [s[j], e[j]], "-", color="#00c000", lw=1.5, alpha=0.8)
        m = 0.5 * (s + e)
        ax.text(m[i], m[j], bid, color="#006000", fontsize=9, ha="center", va="bottom")
    ax.set_title(name + "   colour = normal variation within 2.5 mm (deg); green = gold", fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
fig.colorbar(sc, ax=axes, shrink=0.3)
fig.savefig(HERE / "a1_crease_map.png")
print("wrote crease map")
