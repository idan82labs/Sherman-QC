"""Auditor 3 (resumed run): look for formed features without a gold line.

For every plate found by the bench prep (base, front flange, both walls, both tabs) fit each skin
plane and map the signed residual of all raw vertices whose normal is within 25 deg of that plate
normal and that lie over the plate's footprint. A joggle, offset, bead or small formed tab shows
up as a coherent residual region (> 0.5 mm) or as a cluster of normals tilted 5-40 deg from
every plate. Also lists normal-tilted clusters that are not within 9 mm of a gold line.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa/scripts")
from a3_load import load  # noqa: E402

AUD = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa"
GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"
PLATES = {"base": 1, "front_flange": 5, "right_wall": 3, "left_wall": 6, "left_tab": 9, "right_tab": 11}


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


v, n, f = load()
patches = {p["id"]: p for p in json.load(open(AUD + "/prep/patches.json"))["patches"]}
z = np.load(AUD + "/prep/points_ds.npz")
tree = cKDTree(z["points"])
_, idx = tree.query(v, k=1)
lab = z["labels"][idx]
gold = json.load(open(GOLD))["bends"]

# distance of every vertex to the nearest gold segment
dg = np.full(len(v), np.inf)
for b in gold:
    S, E = np.array(b["line_start"]), np.array(b["line_end"])
    L = np.linalg.norm(E - S); d = (E - S) / L
    t = np.clip((v - S) @ d, 0, L)
    dg = np.minimum(dg, np.linalg.norm(v - S - np.outer(t, d), axis=1))

nrm = {k: unit(patches[pid]["normal"]) for k, pid in PLATES.items()}
cos_to = np.stack([np.abs(n @ nrm[k]) for k in PLATES], 1)
best_plate = np.argmax(cos_to, 1)
best_ang = np.degrees(np.arccos(np.clip(cos_to.max(1), -1, 1)))
report = {}
# 1) residual maps per plate skin
fig, axs = plt.subplots(2, 3, figsize=(24, 13))
for ax, (k, pid) in zip(axs.ravel(), PLATES.items()):
    twin = patches[pid]["twin"]
    rows = {}
    m_plate = (best_plate == list(PLATES).index(k)) & (best_ang < 25) & (dg > 9)
    for sk in [pid] + ([twin] if twin else []):
        nn = unit(patches[sk]["normal"])
        core = (lab == sk)
        c = v[core].mean(0)
        # robust plane on the core labelled points
        P = v[core]
        for _ in range(3):
            w, U = np.linalg.eigh(np.cov((P - c).T)); q = U[:, 0]
            r = (P - c) @ q
            keep = np.abs(r) < max(0.3, 3 * np.median(np.abs(r)))
            P = P[keep]; c = P.mean(0)
        w, U = np.linalg.eigh(np.cov((P - c).T)); q = U[:, 0]
        if q @ nn < 0:
            q = -q
        # assign the plate-like points of this skin: normal sign agrees with this skin
        m = m_plate & ((n @ nn) > 0)
        r = (v[m] - c) @ q
        rows[sk] = dict(n=int(m.sum()), resid_p50=float(np.median(np.abs(r))), resid_p99=float(np.percentile(np.abs(r), 99)),
                        frac_gt_0p5=float((np.abs(r) > 0.5).mean()), frac_gt_1=float((np.abs(r) > 1.0).mean()))
        if sk == pid:
            # in-plane coordinates for the map
            e1 = unit(np.cross(q, [0, 0, 1]) if abs(q[2]) < 0.9 else np.cross(q, [1, 0, 0]))
            e2 = np.cross(q, e1)
            X = v[m] - c
            sc = ax.scatter(X @ e1, X @ e2, c=np.clip(r, -1.5, 1.5), s=0.3, cmap="coolwarm", vmin=-1.5, vmax=1.5)
            ax.set_aspect("equal")
            ax.set_title(f"{k} skin {sk}: residual to plane (mm), p50 {rows[sk]['resid_p50']:.2f} p99 {rows[sk]['resid_p99']:.2f}")
            plt.colorbar(sc, ax=ax, shrink=0.6)
    report[k] = rows
fig.tight_layout()
fig.savefig(AUD + "/a3x_plate_flatness.png", dpi=60)
plt.close(fig)

# 2) tilted-normal clusters away from gold lines: normals 5..40 deg from every plate, not edges (|n . plate| small for edges)
cand = (best_ang > 5) & (best_ang < 40) & (dg > 9)
U = np.where(cand)[0]
print("tilted-normal candidates away from gold:", len(U))
clusters = []
if len(U):
    t2 = cKDTree(v[U])
    pairs = t2.query_pairs(1.2, output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(U), len(U)))
    nc, cl = connected_components(A, directed=False)
    sizes = np.bincount(cl)
    for kk in np.argsort(-sizes)[:20]:
        ii = U[cl == kk]
        if len(ii) < 40:
            continue
        P = v[ii]
        c = P.mean(0)
        w, Q = np.linalg.eigh(np.cov((P - c).T))
        ext = np.sqrt(np.maximum(w, 0)) * 4
        nn = n[ii]
        wn, Qn = np.linalg.eigh(nn.T @ nn)
        mn = Qn[:, -1]
        angs = {k: round(float(np.degrees(np.arccos(min(1, abs(mn @ nrm[k]))))), 1) for k in PLATES}
        clusters.append(dict(size=int(len(ii)), centroid=np.round(c, 1).tolist(), extent_4sigma=np.round(ext[::-1], 1).tolist(),
                             mean_normal=np.round(mn, 3).tolist(), angle_to_plates=angs,
                             min_dist_to_gold=round(float(dg[ii].min()), 1)))
for c in clusters:
    print(c)
json.dump(dict(plate_flatness=report, tilted_clusters=clusters), open(AUD + "/a3x_flatness.json", "w"), indent=1)
for k, rows in report.items():
    print(k, {s: {kk: round(vv, 3) for kk, vv in r.items()} for s, r in rows.items()})
