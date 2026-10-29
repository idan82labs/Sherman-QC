"""Verifier2: missed-formed-bend check over the whole part (uses the prep downsampled cloud, not the full scan)."""
import json, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "..", "..", "prep", "sc_07062eee17aa")
D = np.load(os.path.join(PREP, "points_ds.npz"))
X = D["points"]; N0 = D["normals"]
cfg = json.load(open(os.path.join(HERE, "..", "lines.json")))["bends"]
pat = {p["id"]: p for p in json.load(open(os.path.join(PREP, "patches.json")))["patches"]}

# own normals (PCA k=16) + surface variation
tree = cKDTree(X)
_, nn = tree.query(X, k=16)
Q = X[nn] - X[nn].mean(1, keepdims=True)
C = np.einsum("nki,nkj->nij", Q, Q)
ev, evec = np.linalg.eigh(C)
Nn = evec[:, :, 0]
curv = ev[:, 0] / ev.sum(1)

nw = np.array(pat[1]["normal"]); nw /= np.linalg.norm(nw)
ex = np.array(cfg[0]["direction"]); ex -= (ex @ nw) * nw; ex /= np.linalg.norm(ex)
ey = np.cross(nw, ex)
x, y, z = X @ ex, X @ ey, X @ nw

# distance to labelled lines
def dist_seg(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + np.outer(t, ab)), axis=1)
dl = np.min([dist_seg(X, np.array(b["label_line"]["start"]), np.array(b["label_line"]["end"])) for b in cfg], axis=0)

# known face families (unsigned normals)
fam = {"web": pat[1]["normal"], "flange+Y": pat[6]["normal"], "flange-Y": pat[3]["normal"], "tabA": pat[10]["normal"],
       "tabB": pat[8]["normal"], "earA": pat[19]["normal"], "earB": pat[17]["normal"]}
F = np.array([np.array(v) / np.linalg.norm(v) for v in fam.values()])
angs = np.degrees(np.arccos(np.clip(np.abs(Nn @ F.T), 0, 1)))
amin = angs.min(1)
# rim (cut-edge) faces: normal perpendicular to the local sheet; flag points whose normal is ~90 deg from web AND
# not matching any family -> rims. We keep them but report separately via thinness.
odd = (amin > 8) & (dl > 6.0) & (curv < 0.02)
idx = np.where(odd)[0]
pairs = cKDTree(X[idx]).query_pairs(1.6, output_type="ndarray")
A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
ncomp, lab = connected_components(A, directed=False)
clusters = []
for c in range(ncomp):
    m = idx[lab == c]
    if len(m) < 15:
        continue
    P = X[m]; cc = P.mean(0)
    s = np.linalg.svd(P - cc, compute_uv=False)
    nrm = Nn[m].mean(0); nrm /= np.linalg.norm(nrm)
    clusters.append(dict(n=int(len(m)), centroid=cc.tolist(), xyz_part=[float(cc @ ex), float(cc @ ey), float(cc @ nw)],
                         extent_sv=[float(v) for v in s / np.sqrt(len(m))], mean_normal=nrm.tolist(),
                         ang_to_web=float(np.degrees(np.arccos(min(1, abs(nrm @ F[0]))))),
                         min_ang_family=float(np.degrees(np.arccos(np.clip(np.abs(F @ nrm), 0, 1))).min()),
                         min_dist_label=float(dl[m].min())))
clusters.sort(key=lambda c: -c["n"])
json.dump(clusters, open(os.path.join(HERE, "v2_missed_clusters.json"), "w"), indent=1)
print("odd points", odd.sum(), "clusters>=15:", len(clusters))
for c in clusters[:25]:
    print(c["n"], [round(v, 1) for v in c["xyz_part"]], "sv", [round(v, 2) for v in c["extent_sv"]], "ang_web %.1f minfam %.1f dlab %.1f" % (c["ang_to_web"], c["min_ang_family"], c["min_dist_label"]))

# patches 15/16/21/22 location vs labelled lines
for pid in (15, 16, 21, 22):
    cc = np.array(pat[pid]["centroid"])
    ds = {b["bend_id"]: float(dist_seg(cc[None], np.array(b["label_line"]["start"]), np.array(b["label_line"]["end"]))[0]) for b in cfg}
    k = min(ds, key=ds.get)
    print("patch", pid, "nearest label", k, round(ds[k], 2), "mm; normal angle to web", round(np.degrees(np.arccos(abs(np.array(pat[pid]['normal']) @ nw))), 1))

# --- figures
fig, axs = plt.subplots(1, 2, figsize=(22, 8))
ax = axs[0]
ax.scatter(x, y, c="0.85", s=0.3)
ax.scatter(x[odd], y[odd], c=amin[odd], s=2, cmap="autumn", vmin=8, vmax=60)
for b in cfg:
    s = np.array(b["label_line"]["start"]); e = np.array(b["label_line"]["end"])
    ax.plot([s @ ex, e @ ex], [s @ ey, e @ ey], "b-", lw=1.5); ax.text(s @ ex, s @ ey, b["bend_id"], color="b")
for i, c in enumerate(clusters[:25]):
    ax.text(c["xyz_part"][0], c["xyz_part"][1], str(i), fontsize=7)
ax.set_aspect("equal"); ax.set_title("top view: points whose normal is >8 deg from every known face family and >6 mm from any label line (colour = min angle)")
# web flatness: points with normal within 10 deg of web normal, signed distance to best plane (two skins separated)
ax = axs[1]
wm = (np.degrees(np.arccos(np.clip(np.abs(Nn @ nw), 0, 1))) < 10)
P = X[wm]; c0 = P.mean(0); _, _, vt = np.linalg.svd(P - c0); n = vt[2]
if n @ nw < 0: n = -n
rr = (X - c0) @ n
top = wm & (rr > 0) & (np.abs(rr) < 4)
P = X[top]; c0 = P.mean(0); _, _, vt = np.linalg.svd(P - c0); n = vt[2]
r = (P - c0) @ n
sc = ax.scatter(x[top], y[top], c=r, s=0.5, cmap="coolwarm", vmin=-1.5, vmax=1.5)
plt.colorbar(sc, ax=ax, fraction=0.03, label="mm from best plane (one web skin)")
for b in cfg:
    s = np.array(b["label_line"]["start"]); e = np.array(b["label_line"]["end"])
    ax.plot([s @ ex, e @ ex], [s @ ey, e @ ey], "k-", lw=1)
ax.set_aspect("equal"); ax.set_title(f"web skin (+n side) flatness: rms {r.std():.3f} mm, range {r.min():.2f}..{r.max():.2f}")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "v2_missed_topview.png"), dpi=80); plt.close(fig)

# full-width sections across the folds at several x
xs = np.percentile(x, [3, 15, 30, 45, 55, 70, 85, 97])
fig, axs = plt.subplots(len(xs), 1, figsize=(20, 2.6 * len(xs)))
for ax, x0 in zip(axs, xs):
    m = np.abs(x - x0) < 0.7
    ax.scatter(y[m], z[m], s=1.5, c="k")
    ax.set_title(f"section x={x0:.1f} (perpendicular to fold axis); z exaggerated x2", fontsize=8)
    ax.set_aspect(2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "v2_missed_yz_sections.png"), dpi=75); plt.close(fig)
# sections along the fold axis (xz) at several y to look for bends with axis perpendicular to the folds
ys = np.percentile(y, [5, 25, 50, 75, 95])
fig, axs = plt.subplots(len(ys), 1, figsize=(20, 2.4 * len(ys)))
for ax, y0 in zip(axs, ys):
    m = np.abs(y - y0) < 0.7
    ax.scatter(x[m], z[m], s=1.5, c="k"); ax.grid(alpha=0.3)
    ax.set_title(f"section y={y0:.1f} (along fold axis); z exaggerated x2", fontsize=8); ax.set_aspect(2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "v2_missed_xz_sections.png"), dpi=75); plt.close(fig)
print("done")
