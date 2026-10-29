"""Sanity sweep for formed folds not covered by the 12 measured fold lines (a1_reference_and_match.json).
Flags scan points whose oriented normal is >12 deg from every dominant face normal (+-X, +-Z, +-N and
the two tab-kink faces), drops points within 4.5 mm of any of the 12 fold lines, then separates
fold-like curvature (normal swept between two faces ~90 deg apart on one skin, large radius) from
cut-edge rounding (tight radius ~ t/2) with a local normal-change rate. Renders what is left.
Read-only on inputs."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import a1common as A

pts, nrm, _ = A.load_scan()
g = A.load_gold()
ex, ez, en = A.part_frame(g)
c = pts.mean(0)
M = np.c_[ex, ez, en].T
P = (pts - c) @ M.T
Nn = nrm @ M.T
dom = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
for s in (1, -1):
    for k in (1, -1):
        a = np.radians(28.5)
        dom.append(np.array([s * np.cos(a), k * np.sin(a), 0.0]))
D = np.array(dom)
cosmax = (Nn @ D.T).max(1)
flag = cosmax < np.cos(np.radians(12))
ref = json.loads((HERE / "a1_reference_and_match.json").read_text())["reference_lines_scan_raw"]


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


near = np.zeros(len(pts), bool)
for r in ref:
    a, b = np.array(r["line_start"]), np.array(r["line_end"])
    near |= seg_dist(pts, a, b) < 4.5
cand = flag & ~near
# normal-change rate: mean angle between a point's normal and its neighbours' normals within 0.9 mm
idx = np.nonzero(cand)[0]
tree = cKDTree(pts)
rate = np.zeros(len(idx))
for j, i in enumerate(idx):
    nb = tree.query_ball_point(pts[i], 0.9)
    if len(nb) < 3:
        continue
    cs = np.clip(nrm[nb] @ nrm[i], -1, 1)
    rate[j] = np.degrees(np.arccos(cs)).mean()
# a fold with R~2.7 sweeps ~21 deg/mm -> mean neighbour angle within 0.9 mm ~ 8-12 deg;
# a cut-edge cap (R~0.75) sweeps ~76 deg/mm -> >= 25 deg
foldlike = idx[(rate > 3) & (rate < 16)]
edgelike = idx[rate >= 16]
print("flagged", int(flag.sum()), "after removing 12 folds", int(cand.sum()), "fold-like", len(foldlike), "edge-like", len(edgelike))
fig, axes = plt.subplots(1, 3, figsize=(24, 9), dpi=80)
rng = np.random.default_rng(0)
bg = P[rng.choice(len(P), 60000, replace=False)]
for ax, (i, j, name) in zip(axes, ((0, 1, "top X-Z"), (0, 2, "front X-N"), (1, 2, "side Z-N"))):
    ax.scatter(bg[:, i], bg[:, j], s=0.1, c="#cccccc")
    ax.scatter(P[edgelike, i], P[edgelike, j], s=0.3, c="#1f77b4", label=f"edge-like curvature ({len(edgelike)})")
    ax.scatter(P[foldlike, i], P[foldlike, j], s=1.5, c="#d62728", label=f"fold-like curvature not on the 12 folds ({len(foldlike)})")
    for r in ref:
        a = (np.array(r["line_start"]) - c) @ M.T; b = (np.array(r["line_end"]) - c) @ M.T
        ax.plot([a[i], b[i]], [a[j], b[j]], color="k", lw=0.8)
    ax.set_title(name); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("sc_2714fb7e0c0b: curvature not explained by the 12 measured folds (black lines)")
fig.tight_layout()
fig.savefig(HERE / "a1_unexplained.png")
# cluster fold-like leftovers
if len(foldlike):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    sub = pts[foldlike]
    t2 = cKDTree(sub)
    pairs = np.array(list(t2.query_pairs(1.5)))
    if len(pairs):
        G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(sub), len(sub)))
        ncomp, lab = connected_components(G, directed=False)
        sizes = np.bincount(lab)
        for k in np.argsort(sizes)[::-1][:12]:
            q = P[foldlike][lab == k]
            ext = q.max(0) - q.min(0)
            print("cluster", int(k), "n", int(sizes[k]), "centre", np.round(q.mean(0), 1).tolist(), "extent XZN", np.round(ext, 1).tolist())
