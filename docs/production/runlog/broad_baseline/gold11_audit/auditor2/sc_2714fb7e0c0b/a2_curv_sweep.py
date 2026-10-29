"""Whole-part sweep for curved / non-axis-aligned surface patches not explained by known folds.

Voxel-downsample (0.8 mm), PCA normals (k=24); keep smooth points (surface variation < 0.02) whose normal
deviates > 10 deg from all of +-u, +-v, +-w. Known fold zones (gold lines + the two lower-plate end-flange
folds + the tab-end creases) are masked by distance; what remains is plotted for visual review.
"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a2_common import load, gold_lines, OUT

q, R, O, raw = load()
G = gold_lines(R, O)
vox = 0.45
key = np.floor(q / vox).astype(np.int64)
_, idx = np.unique(key, axis=0, return_index=True)
p = q[np.sort(idx)]
tree = cKDTree(p)
_, nb = tree.query(p, k=10)
X = p[nb] - p[nb].mean(1, keepdims=True)
C = np.einsum("nki,nkj->nij", X, X) / 10
w, V = np.linalg.eigh(C)
nrm = V[:, :, 0]
var = w[:, 0] / w.sum(1)
dev = np.degrees(np.arccos(np.clip(np.abs(nrm).max(1), -1, 1)))
sel = (dev > 10) & (var < 0.06)

def seg_dist(P, a, b):
    d = b - a; L2 = d @ d
    t = np.clip(((P - a) @ d) / L2, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)

known = {k: v for k, v in G.items()}
known["EF_v0"] = (np.array([0.5, 0.8, -42.6]), np.array([44.5, 0.8, -42.6]))
known["EF_v225"] = (np.array([0.5, 224.3, -42.6]), np.array([44.5, 224.3, -42.6]))
mask_known = np.zeros(len(p), bool)
for k, (a, b) in known.items():
    if k in ("B06", "B08"):
        continue
    mask_known |= seg_dist(p, a, b) < 5.0
rest = sel & ~mask_known
print("points", len(p), "non-axis smooth", int(sel.sum()), "unexplained", int(rest.sum()))
# cluster unexplained points
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
P = p[rest]
t2 = cKDTree(P)
pairs = t2.query_pairs(1.6, output_type="ndarray")
n = len(P)
A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
ncomp, lab = connected_components(A, directed=False)
clusters = []
for c in range(ncomp):
    m = lab == c
    if m.sum() < 15:
        continue
    Q = P[m]
    clusters.append({"n": int(m.sum()), "min": Q.min(0).round(1).tolist(), "max": Q.max(0).round(1).tolist(),
                     "mean_dev_deg": float(dev[rest][m].mean())})
clusters.sort(key=lambda c: -c["n"])
for c in clusters[:40]:
    print(c)
(OUT / "a2_curv_sweep.json").write_text(json.dumps({"voxel_mm": vox, "clusters": clusters}, indent=1))
fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=85)
for ax, (i, j, nm) in zip(axes, ((0, 1, "top u-v"), (0, 2, "side u-w"))):
    ax.scatter(p[::6, i], p[::6, j], s=0.2, c="#cccccc", linewidths=0, rasterized=True)
    ax.scatter(p[sel & mask_known, i], p[sel & mask_known, j], s=0.6, c="#9ecae1", linewidths=0, label="non-axis, inside known fold zones")
    sc = ax.scatter(P[:, i], P[:, j], s=2, c=dev[rest], cmap="autumn_r", vmin=10, vmax=60, linewidths=0, label="non-axis, unexplained")
    for k, (a, b) in G.items():
        ax.plot([a[i], b[i]], [a[j], b[j]], color="#2ca02c", lw=1)
        ax.text((a[i] + b[i]) / 2, (a[j] + b[j]) / 2, k, fontsize=7, color="g")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(nm); ax.legend(fontsize=8, loc="upper right")
plt.colorbar(sc, ax=axes[1], label="normal deviation from nearest axis (deg)")
fig.tight_layout(); fig.savefig(OUT / "a2_curv_sweep.png"); plt.close(fig)
