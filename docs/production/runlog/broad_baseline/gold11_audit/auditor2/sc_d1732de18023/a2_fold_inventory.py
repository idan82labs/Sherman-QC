"""Auditor-2 fold inventory: every connected mesh region whose vertex normals deviate
>12 deg from all six principal skin directions is clustered and characterised by the
axis its normals rotate about (bend arc: axis lies in the sheet, normals sweep ~bend
angle, region elongated along that axis; hole/edge wall: axis = skin normal).
Writes a2_fold_inventory.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_ply, unit  # noqa: E402

pts, nrm, tri = load_ply(faces=True)
c0 = pts.mean(0)
_, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
e3 = vt[2]
X = json.loads((OUT / "a2_xsec.json").read_text())
f1 = np.array(X["M01"]["frame"]["f"])     # short-flange normal family
f2 = np.array(X["M02"]["frame"]["f"])     # long-flange normal family
dirs = np.array([e3, f1, f2])
dev = np.degrees(np.arccos(np.clip(np.max(np.abs(nrm @ dirs.T), axis=1), -1, 1)))
off = dev > 12.0
i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
k = off[i] & off[j]
A = coo_matrix((np.ones(k.sum()), (i[k], j[k])), shape=(len(pts),) * 2).tocsr()
nc, lab = connected_components(A, directed=False)
lab[~off] = -1
ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
G = load_gold()
gl = {b: (a, unit(e - a), np.linalg.norm(e - a)) for b, (a, e) in G.items()}
rows = []
P = (pts - c0) @ vt.T
for cid, n in sorted(zip(ids, cnt), key=lambda x: -x[1]):
    if n < 25:
        break
    m = lab == cid
    q, nn = pts[m], nrm[m]
    # axis about which normals rotate: smallest-variance direction of the normals
    _, s, v = np.linalg.svd(nn - nn.mean(0))
    ax = v[2]
    # sweep: angle range of normals projected onto plane perpendicular to ax
    b1 = unit(np.cross(ax, [1, 0, 0]) if abs(ax[0]) < 0.9 else np.cross(ax, [0, 1, 0]))
    b2 = np.cross(ax, b1)
    ang = np.degrees(np.arctan2(nn @ b2, nn @ b1))
    ang_sorted = np.sort(ang)
    gaps = np.diff(np.r_[ang_sorted, ang_sorted[0] + 360])
    sweep = 360 - gaps.max()
    # extent along the axis vs across
    t = (q - q.mean(0)) @ ax
    pc = (q - q.mean(0))
    _, sp, vp = np.linalg.svd(pc, full_matrices=False)
    elong = vp[0]
    length = float(np.ptp(pc @ elong))
    width = float(np.ptp(pc @ vp[1]))
    in_sheet = float(abs(ax @ e3))         # ~0 => axis lies in plate plane (bend between plate & flange)
    # nearest gold line (perp distance of centroid) if axis parallel
    near = None
    for b, (a0, dg, L) in gl.items():
        cc = q.mean(0) - a0
        tt = cc @ dg
        dperp = np.linalg.norm(cc - dg * tt)
        angd = np.degrees(np.arccos(min(1, abs(dg @ elong))))
        if near is None or dperp < near[1]:
            near = (b, float(dperp), float(angd), float(tt), float(L))
    rows.append({"cid": int(cid), "n": int(n), "centroid_pca": np.round(q.mean(0) @ vt.T - c0 @ vt.T, 1).tolist(),
                 "bbox_pca_min": np.round(P[m].min(0), 1).tolist(), "bbox_pca_max": np.round(P[m].max(0), 1).tolist(),
                 "normal_rot_axis_pca": np.round(vt @ ax, 3).tolist(), "axis_dot_plate_normal": round(in_sheet, 3),
                 "normal_sweep_deg": round(float(sweep), 1), "length_mm": round(length, 1), "width_mm": round(width, 1),
                 "elong_dir_pca": np.round(vt @ elong, 3).tolist(),
                 "nearest_gold": {"bend": near[0], "perp_mm": round(near[1], 1), "angle_deg": round(near[2], 1),
                                  "t_on_gold": round(near[3], 1), "gold_len": round(near[4], 1)}})
json.dump(rows, open(OUT / "a2_fold_inventory.json", "w"), indent=1)
print("off-axis vertices:", int(off.sum()), "clusters>=25:", len(rows))
for r in rows:
    print(r["cid"], r["n"], "c", r["centroid_pca"], "len", r["length_mm"], "w", r["width_mm"], "rotax", r["normal_rot_axis_pca"],
          "sweep", r["normal_sweep_deg"], "elong", r["elong_dir_pca"], "near", r["nearest_gold"])
