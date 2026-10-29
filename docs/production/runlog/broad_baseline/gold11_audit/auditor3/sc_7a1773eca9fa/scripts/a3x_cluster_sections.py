"""Auditor 3 (resumed run): cross-sections through the ambiguous tilted-normal clusters from a3x_flatness.json
(is any of them an unlabeled formed fold, joggle or small tab?).

For each chosen cluster: long axis a (PCA of its points), section planes perpendicular to a at 3 stations,
all raw vertices within +-0.6 mm of the plane and 30 mm of the cluster centroid, coloured by the closest
plate normal; the cluster's own points in red. Distances from the cluster to every gold line are printed.
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
COLORS = {"base": "tab:blue", "front_flange": "tab:purple", "right_wall": "tab:green", "left_wall": "tab:olive",
          "left_tab": "tab:orange", "right_tab": "tab:brown"}


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


v, n, f = load()
patches = {p["id"]: p for p in json.load(open(AUD + "/prep/patches.json"))["patches"]}
gold = json.load(open(GOLD))["bends"]
nrm = {k: unit(patches[pid]["normal"]) for k, pid in PLATES.items()}
cos_to = np.stack([np.abs(n @ nrm[k]) for k in PLATES], 1)
best_plate = np.argmax(cos_to, 1)
best_ang = np.degrees(np.arccos(np.clip(cos_to.max(1), -1, 1)))
dg = np.full(len(v), np.inf)
for b in gold:
    S, E = np.array(b["line_start"]), np.array(b["line_end"])
    L = np.linalg.norm(E - S); d = (E - S) / L
    t = np.clip((v - S) @ d, 0, L)
    dg = np.minimum(dg, np.linalg.norm(v - S - np.outer(t, d), axis=1))
cand = (best_ang > 5) & (best_ang < 40) & (dg > 9)
U = np.where(cand)[0]
t2 = cKDTree(v[U])
pairs = t2.query_pairs(1.2, output_type="ndarray")
A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(U), len(U)))
nc, cl = connected_components(A, directed=False)
sizes = np.bincount(cl)
order = [k for k in np.argsort(-sizes) if sizes[k] >= 40][:20]
# clusters of interest by rank in a3x_flatness order (0-based): 4 (#5), 6 (#7), 11 (#12), 18 (#19); plus 1 (#2 flange-edge strip)
WANT = {1: "C2 flange-length strip", 4: "C5 near left tab 8deg", 6: "C7 base region 8.6deg", 11: "C12 strip near left notch",
        18: "C19 strip near right notch"}
fig, axs = plt.subplots(len(WANT), 3, figsize=(18, 5.6 * len(WANT)))
info = {}
for row, (rank, name) in enumerate(WANT.items()):
    ii = U[cl == order[rank]]
    P = v[ii]
    c = P.mean(0)
    w, Q = np.linalg.eigh(np.cov((P - c).T))
    a = Q[:, -1]
    tt = (P - c) @ a
    gd = {}
    for b in gold:
        S, E = np.array(b["line_start"]), np.array(b["line_end"])
        L = np.linalg.norm(E - S); d = (E - S) / L
        t = np.clip((P - S) @ d, 0, L)
        gd[b["bend_id"]] = round(float(np.linalg.norm(P - S - np.outer(t, d), axis=1).min()), 1)
    nn = n[ii]
    hist = np.bincount(best_plate[ii], minlength=len(PLATES))
    info[name] = dict(size=len(ii), centroid=np.round(c, 1).tolist(), long_axis=np.round(a, 3).tolist(),
                      length=round(float(tt.max() - tt.min()), 1), min_dist_to_gold=gd,
                      closest_plate_hist={k: int(h) for k, h in zip(PLATES, hist)},
                      best_ang_p10_p50_p90=np.round(np.percentile(best_ang[ii], [10, 50, 90]), 1).tolist())
    print(name, info[name])
    # section basis: e1 = cluster mean normal projected, e2 = a x e1
    wn, Qn = np.linalg.eigh(nn.T @ nn)
    mn = Qn[:, -1]
    e1 = unit(mn - (mn @ a) * a)
    e2 = np.cross(a, e1)
    for col, fr in enumerate((0.2, 0.5, 0.8)):
        s0 = np.percentile(tt, 100 * fr)
        o = c + s0 * a
        rel = v - o
        m = (np.abs(rel @ a) < 0.6) & (np.linalg.norm(rel, axis=1) < 32)
        x, y = rel[m] @ e1, rel[m] @ e2
        bp = best_plate[m]; ba = best_ang[m]
        ax = axs[row, col]
        for k, name_p in enumerate(PLATES):
            sel = (bp == k) & (ba <= 5)
            ax.scatter(y[sel], x[sel], s=2, c=COLORS[name_p], label=name_p if (row == 0 and col == 0) else None)
        sel = ba > 5
        ax.scatter(y[sel], x[sel], s=2, c="0.55", label="tilted >5deg" if (row == 0 and col == 0) else None)
        inc = np.isin(np.where(m)[0], ii)
        ax.scatter(y[inc], x[inc], s=4, c="red", label="cluster pts" if (row == 0 and col == 0) else None)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{name} @ {int(fr*100)}% (slab +-0.6 mm)\nnearest gold {min(gd, key=gd.get)} {min(gd.values())} mm")
axs[0, 0].legend(fontsize=7, loc="lower left", markerscale=4)
fig.tight_layout()
fig.savefig(AUD + "/a3x_cluster_sections.png", dpi=62)
json.dump(info, open(AUD + "/a3x_cluster_sections.json", "w"), indent=1)
