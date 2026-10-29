"""Auditor-2: planar region growing on the raw scan mesh (faces), to find every flat skin patch.

Seeds = flattest faces first; a face joins a region when its normal is within ANG deg of the region
normal and its centroid is within DIST mm of the region plane (plane refit as the region grows).
Regions >= MIN_AREA mm2 are reported in the (t, u, w) frame of the gold lines (t along the common
gold direction, origin = M02 start), and pairs of parallel regions ~thickness apart are paired into
plates. Output: a2_planes.json + a2_planes_views.png.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, perp_basis, unit  # noqa: E402

ANG, DIST, MIN_AREA = 4.0, 0.25, 15.0
m = load_mesh()
gold = load_gold()
d = unit(gold["M01"][1] - gold["M01"][0])
u, w = perp_basis(d)
o = gold["M02"][0]
fn = np.asarray(m.face_normals)
fa = np.asarray(m.area_faces)
fc = np.asarray(m.triangles_center)
nF = len(fn)
adj = np.asarray(m.face_adjacency)
A = coo_matrix((np.ones(len(adj) * 2), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(nF, nF)).tocsr()

# local flatness: max normal deviation to adjacent faces
dev = np.zeros(nF)
cosang = np.einsum("ij,ij->i", fn[adj[:, 0]], fn[adj[:, 1]])
np.maximum.at(dev, adj[:, 0], 1 - cosang)
np.maximum.at(dev, adj[:, 1], 1 - cosang)
order = np.argsort(dev)
label = -np.ones(nF, int)
regions = []
cos_thr = np.cos(np.radians(ANG))
for seed in order:
    if label[seed] >= 0:
        continue
    rid = len(regions)
    n0 = fn[seed].copy()
    p0 = fc[seed].copy()
    members = [seed]
    label[seed] = rid
    q = deque([seed])
    acc_n = fn[seed] * fa[seed]
    acc_p = fc[seed] * fa[seed]
    acc_a = fa[seed]
    cnt = 1
    while q:
        f = q.popleft()
        for g in A.indices[A.indptr[f]:A.indptr[f + 1]]:
            if label[g] >= 0:
                continue
            if fn[g] @ n0 < cos_thr:
                continue
            if abs((fc[g] - p0) @ n0) > DIST:
                continue
            label[g] = rid
            members.append(g)
            q.append(g)
            acc_n += fn[g] * fa[g]
            acc_p += fc[g] * fa[g]
            acc_a += fa[g]
            cnt += 1
            if cnt % 200 == 0:
                n0 = unit(acc_n)
                p0 = acc_p / acc_a
    regions.append(np.asarray(members))

areas = np.array([fa[r].sum() for r in regions])
big = [i for i in np.argsort(-areas) if areas[i] >= MIN_AREA]
rows = []
for i in big:
    r = regions[i]
    P = fc[r]
    W = fa[r]
    cen = (P * W[:, None]).sum(0) / W.sum()
    X = P - cen
    C = (X * W[:, None]).T @ X / W.sum()
    ev, evec = np.linalg.eigh(C)
    n = evec[:, 0]
    if n @ (fn[r] * W[:, None]).sum(0) < 0:
        n = -n
    resid = np.sqrt(ev[0])
    tuw = np.c_[(P - o) @ d, (P - o) @ u, (P - o) @ w]
    rows.append({
        "rid": int(i), "area": float(W.sum()), "faces": int(len(r)), "normal": n.tolist(),
        "centroid": cen.tolist(), "rms_mm": float(resid),
        "n_dot_d": float(n @ d),
        "n_uw_deg": float(np.degrees(np.arctan2(n @ w, n @ u))),
        "t_range": [float(tuw[:, 0].min()), float(tuw[:, 0].max())],
        "u_range": [float(tuw[:, 1].min()), float(tuw[:, 1].max())],
        "w_range": [float(tuw[:, 2].min()), float(tuw[:, 2].max())],
        "cen_tuw": [float((cen - o) @ d), float((cen - o) @ u), float((cen - o) @ w)],
    })
print(f"{len(regions)} regions, {len(big)} with area >= {MIN_AREA} mm2; covered area {areas[big].sum():.0f} of {fa.sum():.0f}")
for r in rows:
    print(f"R{r['rid']:5d} A={r['area']:7.1f} rms={r['rms_mm']:.3f} n.d={r['n_dot_d']:+.3f} n_uw={r['n_uw_deg']:7.1f} "
          f"cen_tuw=({r['cen_tuw'][0]:6.1f},{r['cen_tuw'][1]:6.1f},{r['cen_tuw'][2]:6.1f}) "
          f"t[{r['t_range'][0]:6.1f},{r['t_range'][1]:6.1f}] u[{r['u_range'][0]:6.1f},{r['u_range'][1]:6.1f}] w[{r['w_range'][0]:6.1f},{r['w_range'][1]:6.1f}]")

# pair parallel opposite skins into plates
plates = []
used = set()
for a_i, ra in enumerate(rows):
    if a_i in used:
        continue
    best = None
    for b_i, rb in enumerate(rows):
        if b_i == a_i or b_i in used:
            continue
        na, nb = np.array(ra["normal"]), np.array(rb["normal"])
        if na @ nb > -np.cos(np.radians(6)):
            continue
        sep = (np.array(rb["centroid"]) - np.array(ra["centroid"])) @ na
        # rb must lie behind ra (inside the material) at ~thickness
        if not (-4.0 < sep < -1.5):
            continue
        # overlapping extents in t
        ov = min(ra["t_range"][1], rb["t_range"][1]) - max(ra["t_range"][0], rb["t_range"][0])
        if ov < 2:
            continue
        score = rb["area"]
        if best is None or score > best[0]:
            best = (score, b_i, sep)
    if best is not None:
        used.add(a_i)
        used.add(best[1])
        plates.append((a_i, best[1], best[2]))
print("\nplates (skin pairs):")
for a_i, b_i, sep in plates:
    ra, rb = rows[a_i], rows[b_i]
    ang = np.degrees(np.arccos(min(1, -np.array(ra["normal"]) @ np.array(rb["normal"]))))
    print(f"  R{ra['rid']} + R{rb['rid']}: thickness {abs(sep):.3f} mm, skin non-parallelism {ang:.2f} deg, "
          f"areas {ra['area']:.0f}/{rb['area']:.0f}, n_uw {ra['n_uw_deg']:.1f}, n.d {ra['n_dot_d']:+.3f}")

json.dump({"frame": {"o": o.tolist(), "d": d.tolist(), "u": u.tolist(), "w": w.tolist()},
           "regions": rows, "plates": [[rows[a]["rid"], rows[b]["rid"], s] for a, b, s in plates]},
          open(OUT / "a2_planes.json", "w"), indent=1)
np.save(OUT / "face_region_label.npy", label)

# render: faces coloured by region (top 20), in (t,u,w) projections
cmap = plt.get_cmap("tab20")
col = np.full((nF, 4), 0.85)
for k, r in enumerate(rows[:20]):
    col[regions[r["rid"]]] = cmap(k % 20)
T = np.c_[(fc - o) @ d, (fc - o) @ u, (fc - o) @ w]
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
for ax, (a, b, ttl) in zip(axs, [(0, 1, "t vs u"), (0, 2, "t vs w"), (1, 2, "u vs w")]):
    ax.scatter(T[:, a], T[:, b], s=0.4, c=col, linewidths=0)
    for k, r in enumerate(rows[:20]):
        c = r["cen_tuw"]
        ax.text(c[a], c[b], f"R{r['rid']}", fontsize=7)
    for bid, (s, e) in gold.items():
        ps = [(s - o) @ d, (s - o) @ u, (s - o) @ w]
        pe = [(e - o) @ d, (e - o) @ u, (e - o) @ w]
        ax.plot([ps[a], pe[a]], [ps[b], pe[b]], "r-", lw=2)
        ax.text(pe[a], pe[b], bid, color="r")
    ax.set_title(ttl)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a2_planes_views.png", dpi=70)
print("wrote", OUT / "a2_planes_views.png")
