"""Auditor 3: find scan regions not explained by the six plates (both skins), the five fold zones, or edge faces."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_load import load  # noqa: E402

OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa"
PREP = OUT + "/prep"
v, n, f = load()
z = np.load(PREP + "/points_ds.npz")
print({k: z[k].shape for k in z.files})
P = z["points"] if "points" in z.files else z[z.files[0]]
labels = z["labels"] if "labels" in z.files else None
patches = json.load(open(PREP + "/patches.json"))["patches"]

# full-res assignment: nearest downsampled point label
tree = cKDTree(P)
dist, idx = tree.query(v, k=1)
lab = labels[idx]
plates = {"base": (1, 2), "right_wall": (3, 4), "front_flange": (5, 8), "left_wall": (6, 7), "left_tab": (9, 10), "right_tab": (11, 12)}
by = {p["id"]: p for p in patches}
# per-plate skin plane residual for points labelled with that plate's patches
explained = np.zeros(len(v), bool)
for name, ids in plates.items():
    for pid in ids:
        nrm = np.asarray(by[pid]["normal"]);
        m = lab == pid
        c = v[m].mean(0)
        r = (v[m] - c) @ nrm
        print(f"{name:13s} patch {pid:2d}: n={m.sum():6d} plane resid p50 {np.median(np.abs(r)):.3f} p99 {np.percentile(np.abs(r),99):.3f} max {np.abs(r).max():.3f}")
        explained[np.where(m)[0][np.abs(r) < 0.6]] = True
# fold zones: within 9 mm of any gold line (folds verified to coincide)
G = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"))["bends"]
for b in G:
    S, E = np.array(b["line_start"]), np.array(b["line_end"])
    L = np.linalg.norm(E - S); d = (E - S) / L
    rel = v - S; t = rel @ d
    lat = np.linalg.norm(rel - np.outer(t, d), axis=1)
    explained |= (lat < 9.5) & (t > -6) & (t < L + 6)
# edge faces: points whose normal is ~perpendicular to the local plate normal (cut edges) near patch boundaries
edge_like = np.zeros(len(v), bool)
for pid in (13, 15, 18, 22, 25):
    edge_like |= lab == pid
explained |= edge_like
# remaining: unexplained points; cluster them
U = np.where(~explained)[0]
print("unexplained points:", len(U), f"({100*len(U)/len(v):.2f}%)")
if len(U):
    ut = cKDTree(v[U])
    pairs = ut.query_pairs(1.5, output_type="ndarray")
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(U), len(U)))
    ncomp, cl = connected_components(A, directed=False)
    sizes = np.bincount(cl)
    order = np.argsort(-sizes)
    print("clusters:", ncomp)
    for k in order[:15]:
        pts = v[U[cl == k]]
        ext = pts.max(0) - pts.min(0)
        nn = n[U[cl == k]]
        # dominant normal
        w, Q = np.linalg.eigh(nn.T @ nn)
        print(f"  cluster {k}: {sizes[k]} pts centroid {np.round(pts.mean(0),1)} bbox {np.round(ext,1)} dom-normal {np.round(Q[:,-1],2)} (eig frac {w[-1]/w.sum():.2f})")
    # render
    c0 = v.mean(0)
    Wv, Uv = np.linalg.eigh(np.cov((v - c0).T))
    ax3 = Uv[:, ::-1]
    q = (v - c0) @ ax3
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for a, (i, j) in zip(axs, ((0, 1), (0, 2))):
        s = np.random.default_rng(0).choice(len(v), 120000, replace=False)
        a.scatter(q[s, i], q[s, j], s=0.2, c="0.8")
        a.scatter(q[U, i], q[U, j], s=1.5, c="red")
        a.set_aspect("equal"); a.set_title(f"unexplained points (red) PC{i+1}-PC{j+1}")
    fig.tight_layout(); fig.savefig(OUT + "/a3_unexplained.png", dpi=70)
