"""Auditor-2: global segmentation of sc_70b39f530f00 into flat facets and curved (bend) bands.

1. Area-weighted Gauss map of face normals, folded +-n (both skins of a plate share a direction),
   greedy peak picking -> dominant plate orientations.
2. Per-vertex normal-variation curvature proxy over a 1.2 mm neighbourhood (|dn|/|dx| median),
   which separates flat skins (~0), formed bend arcs (~1/R, R 3-12 mm -> 0.08-0.35 /mm) and sharp
   plate edges / hole rims (>> 0.5 /mm, only within ~1 mm of a rim).
3. Writes renders (4 views) coloured by orientation cluster / curvature, with the gold lines.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, unit  # noqa: E402

m = load_mesh()
gold = load_gold()
V = np.asarray(m.vertices)
F = np.asarray(m.faces)
fn = np.asarray(m.face_normals)
fa = np.asarray(m.area_faces)
fc = np.asarray(m.triangles_center)

# ---------------------------------------------------------------- 1. Gauss map peaks (folded)
fold = fn * np.sign(fn[:, 2:3] + 1e-9)  # fold to upper hemisphere (z>=0)
remaining = np.ones(len(fn), bool)
peaks = []
for _ in range(12):
    idx = np.where(remaining)[0]
    if len(idx) == 0:
        break
    # candidate centres: subsample
    cand = idx[:: max(1, len(idx) // 3000)]
    best, best_a = None, 0.0
    for c in cand:
        d = np.abs(fold[idx] @ fold[c])
        a = fa[idx][d > np.cos(np.radians(4))].sum()
        if a > best_a:
            best, best_a = c, a
    if best_a < 40.0:  # mm^2
        break
    d = np.abs(fold[idx] @ fold[best])
    sel = idx[d > np.cos(np.radians(4))]
    # refine centre: area-weighted mean with consistent sign
    s = np.sign(fold[sel] @ fold[best])[:, None]
    cen = unit((fold[sel] * s * fa[sel, None]).sum(0))
    d2 = np.abs(fold[idx] @ cen)
    sel = idx[d2 > np.cos(np.radians(5))]
    remaining[sel] = False
    peaks.append((cen, fa[sel].sum(), len(sel)))

print("dominant plate orientations (folded normals, 5 deg cone):")
for i, (c, a, n) in enumerate(peaks):
    print(f"  P{i}: n={np.round(c, 4)} area={a:.0f} mm2 faces={n}")
print("area not in any peak:", fa[remaining].sum().round(0), "of", fa.sum().round(0))
for i in range(len(peaks)):
    for j in range(i + 1, len(peaks)):
        ang = np.degrees(np.arccos(min(1, abs(peaks[i][0] @ peaks[j][0]))))
        print(f"  angle P{i}-P{j} = {ang:.1f} deg")

# gold directions vs peaks
for bid, (s, e) in gold.items():
    d = unit(e - s)
    print(bid, "dir", np.round(d, 4), "len", round(float(np.linalg.norm(e - s)), 2),
          "perp-to-peaks (|n.d|):", [round(float(abs(p[0] @ d)), 3) for p in peaks])

# ---------------------------------------------------------------- 2. curvature proxy per vertex
vn = np.asarray(m.vertex_normals)
tree = cKDTree(V)
R = 1.2
nb = tree.query_ball_point(V, R)
curv = np.zeros(len(V))
for i, idx in enumerate(nb):
    idx = np.asarray(idx)
    idx = idx[idx != i]
    if len(idx) < 3:
        continue
    dx = np.linalg.norm(V[idx] - V[i], axis=1)
    dn = np.linalg.norm(vn[idx] - vn[i], axis=1)
    ok = dx > 0.05
    if ok.sum() < 3:
        continue
    curv[i] = np.median(dn[ok] / dx[ok])
np.save(OUT / "vertex_curv_proxy.npy", curv)
print("curvature proxy percentiles 50/75/90/95/99:", np.round(np.percentile(curv, [50, 75, 90, 95, 99]), 3))

# label each vertex with nearest peak orientation (or none)
vfold = vn * np.sign(vn[:, 2:3] + 1e-9)
P = np.array([p[0] for p in peaks])
dots = np.abs(vfold @ P.T)
vlab = np.argmax(dots, 1)
vlab[dots.max(1) < np.cos(np.radians(6))] = -1
np.save(OUT / "vertex_peak_label.npy", vlab)

# ---------------------------------------------------------------- 3. renders
c0 = V.mean(0)
X = V - c0
w_, v_ = np.linalg.eigh(np.cov(X.T))
pcs = v_[:, ::-1]
Y = X @ pcs


def gproj(p):
    return (np.asarray(p) - c0) @ pcs


cmap = plt.get_cmap("tab10")
views = [("top PC1-PC2", 0, 1), ("front PC1-PC3", 0, 2), ("side PC2-PC3", 1, 2)]
fig, axs = plt.subplots(2, 3, figsize=(21, 13))
for row in range(2):
    for k, (title, a, b) in enumerate(views):
        ax = axs[row, k]
        if row == 0:
            cols = np.array([cmap(l % 10) if l >= 0 else (0.75, 0.75, 0.75, 1) for l in vlab])
            ax.scatter(Y[:, a], Y[:, b], s=0.3, c=cols, linewidths=0)
            ax.set_title(title + " - colour = plate orientation cluster (grey = none)")
        else:
            cc = np.clip(curv, 0, 0.6)
            ax.scatter(Y[:, a], Y[:, b], s=0.3, c=cc, cmap="viridis", vmin=0, vmax=0.6, linewidths=0)
            ax.set_title(title + " - colour = curvature proxy (0..0.6 /mm)")
        for bid, (s, e) in gold.items():
            ps, pe = gproj(s), gproj(e)
            ax.plot([ps[a], pe[a]], [ps[b], pe[b]], "-", color="red", lw=2)
            ax.text(pe[a], pe[b], bid, color="red", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
fig.suptitle("sc_70b39f530f00 auditor-2 segmentation (gold lines red); peaks: " +
             "; ".join(f"P{i} {a:.0f}mm2" for i, (c, a, n) in enumerate(peaks)), fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "a2_segmentation_views.png", dpi=80)
print("wrote", OUT / "a2_segmentation_views.png")
json.dump({"peaks": [{"n": c.tolist(), "area": float(a), "faces": int(n)} for c, a, n in peaks],
           "pcs": pcs.tolist(), "c0": c0.tolist()}, open(OUT / "a2_segmentation.json", "w"), indent=1)
