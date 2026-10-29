"""Auditor-2: independent planar-patch segmentation of the whole scan (mesh topology +
vertex normals) to inventory every planar facet and every formed fold.
Writes a2_patches.json + a2_patches_*.png.  Read-only on inputs."""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from a2_common import OUT, fit_plane, load_gold, load_ply, unit  # noqa: E402

pts, nrm, tri = load_ply(faces=True)
c0 = pts.mean(0)
_, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
E = vt.copy()
P = (pts - c0) @ E.T          # PCA coords
N = nrm @ E.T

# vertex adjacency from triangles
i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])

# region growing by normal agreement along mesh edges: keep edge if normals within 6 deg
cosmax = np.cos(np.radians(6.0))
keep = np.einsum("ij,ij->i", nrm[i], nrm[j]) > cosmax
A = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(len(pts),) * 2).tocsr()
ncomp, lab = connected_components(A, directed=False)
sizes = np.bincount(lab)
order = np.argsort(-sizes)
spacing2 = 0.52 ** 2 * 0.866  # rough area per vertex mm^2 (triangle mesh)
patches = []
for rank, cid in enumerate(order):
    if sizes[cid] < 60:
        break
    m = lab == cid
    q = pts[m]
    c, n, s = fit_plane(q)
    r = (q - c) @ n
    rms = float(np.sqrt(np.mean(r ** 2)))
    # orient normal like the mean vertex normal
    if n @ nrm[m].mean(0) < 0:
        n = -n
    pq = P[m]
    patches.append({
        "pid": rank, "n_vert": int(m.sum()), "area_mm2": round(float(m.sum() * spacing2), 1),
        "centroid_pca": np.round((c - c0) @ E.T, 2).tolist(),
        "normal_pca": np.round(n @ E.T, 4).tolist(),
        "plane_rms_mm": round(rms, 3),
        "bbox_pca_min": np.round(pq.min(0), 1).tolist(),
        "bbox_pca_max": np.round(pq.max(0), 1).tolist(),
        "centroid_raw": np.round(c, 3).tolist(), "normal_raw": np.round(n, 5).tolist(),
    })
lab_rank = np.full(len(pts), -1)
for rank, cid in enumerate(order[:len(patches)]):
    lab_rank[lab == cid] = rank
np.save(OUT / "a2_patch_labels.npy", lab_rank)
json.dump({"pca_centroid": c0.tolist(), "pca_axes": E.tolist(), "patches": patches},
          open(OUT / "a2_patches.json", "w"), indent=1)
print("components", ncomp, "patches>=60v", len(patches), "covered frac",
      round(float((lab_rank >= 0).mean()), 3))
for p in patches[:60]:
    print(p["pid"], p["n_vert"], p["area_mm2"], "n", p["normal_pca"], "c", p["centroid_pca"],
          "rms", p["plane_rms_mm"], "bb", p["bbox_pca_min"], p["bbox_pca_max"])
