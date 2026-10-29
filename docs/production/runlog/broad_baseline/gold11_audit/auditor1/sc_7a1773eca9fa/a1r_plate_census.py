#!/usr/bin/env python3
"""Auditor-1 (resumed run) missing-bend check for sc_7a1773eca9fa (49001000). Light, read-only.

Plate census: area-weighted face-normal clustering (sign-insensitive, 4 deg tolerance), then spatial
connected components of each normal cluster (faces linked through shared vertices).  Every planar
patch >= 40 mm^2 that is not one of the six known plates (base, LF, LTW, RTW, LT, RT) is a candidate
for an unlabelled formed bend (a 15 deg tab / joggle step would show up here as its own patch).
Also reports per-plate planarity (RMS / p99 distance to the plate's own plane, per skin side) so a
shallow fold inside a plate (which would merge into one 4-deg cluster) shows up as a large p99.
Writes a1r_plate_census.json and a1r_plate_census.png next to this script.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply")
LABELS = Path("/Users/idantal/ShermanQC/FullProject/output/bend_label_10_49001000/bend_instance_labels.json")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KNOWN = {  # inward-or-outward plate normals from audit_sections.py seeds (sign-insensitive here)
    "base": [-0.3853, 0.2373, -0.8918],
    "LF": [0.6801, -0.5866, -0.4397],
    "LTW": [-0.8069, -0.5583, 0.1930],
    "RTW": [-0.2293, -0.9411, 0.2484],
    "RT": [0.3916, 0.9170, 0.0763],
    "LT": [-0.5827, -0.5960, 0.5525],
}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(SCAN))
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-9
    n = np.zeros_like(cr)
    n[ok] = cr[ok] / (2 * area[ok, None])
    cen = (a + b + c) / 3.0
    # sign-insensitive greedy clustering on the sphere, seeded by area-weighted density
    tol = np.cos(np.radians(4.0))
    lab = -np.ones(len(F), int)
    centres = []
    rem = np.where(ok)[0]
    rng = np.random.default_rng(0)
    for it in range(40):
        if len(rem) == 0:
            break
        # pick the densest direction among a sample of remaining faces
        smp = rng.choice(rem, min(4000, len(rem)), replace=False)
        dots = np.abs(n[smp] @ n[smp].T)
        dens = (dots > tol).astype(float) @ area[smp]
        seed = n[smp[np.argmax(dens)]]
        for _ in range(3):
            s = np.abs(n[rem] @ seed) > tol
            if s.sum() == 0:
                break
            w = area[rem][s]
            nn = n[rem][s] * np.sign(n[rem][s] @ seed)[:, None]
            seed = unit((nn * w[:, None]).sum(0))
        s = np.abs(n[rem] @ seed) > tol
        if area[rem][s].sum() < 20.0:
            break
        lab[rem[s]] = len(centres)
        centres.append(seed)
        rem = rem[~s]
    # connected components of each cluster (faces sharing a vertex)
    patches = []
    for k, cdir in enumerate(centres):
        idx = np.where(lab == k)[0]
        if len(idx) == 0:
            continue
        sub = F[idx]
        # face-vertex incidence -> face adjacency through vertices
        rows = np.repeat(np.arange(len(idx)), 3)
        cols = sub.ravel()
        uniq, inv = np.unique(cols, return_inverse=True)
        A = coo_matrix((np.ones(len(rows)), (rows, inv)), shape=(len(idx), len(uniq))).tocsr()
        adj = (A @ A.T)
        ncomp, comp = connected_components(adj, directed=False)
        for ci in range(ncomp):
            fi = idx[comp == ci]
            ar = float(area[fi].sum())
            if ar < 40.0:
                continue
            P = cen[fi]
            cd = cdir
            off = P @ cd
            # two skins? split offsets
            known = max(KNOWN, key=lambda nm: abs(unit(KNOWN[nm]) @ cd))
            kdot = abs(unit(KNOWN[known]) @ cd)
            ext = P.max(0) - P.min(0)
            # planarity: residual to the patch's own best plane (per side by face-normal sign)
            res = {}
            for sgn in (1, -1):
                sel = np.sign(n[fi] @ cd) == sgn
                if sel.sum() < 20:
                    continue
                Q = P[sel]
                c0 = Q.mean(0)
                _u, _s, vt = np.linalg.svd(Q - c0, full_matrices=False)
                r = np.abs((Q - c0) @ vt[2])
                res[str(sgn)] = {"n": int(sel.sum()), "rms": float(np.sqrt(np.mean(r ** 2))), "p99": float(np.percentile(r, 99)),
                                 "area": float(area[fi][sel].sum())}
            patches.append({"cluster": k, "normal": cd.tolist(), "area_mm2": ar, "n_faces": int(len(fi)),
                            "centroid": P.mean(0).tolist(), "bbox_extent": ext.tolist(),
                            "nearest_known_plate": known, "angle_to_known_deg": float(np.degrees(np.arccos(min(1.0, kdot)))),
                            "planarity_by_side": res})
    patches.sort(key=lambda p: -p["area_mm2"])
    out = {"n_normal_clusters": len(centres), "patches": patches}
    (HERE / "a1r_plate_census.json").write_text(json.dumps(out, indent=1))
    print("normal clusters:", len(centres))
    for p in patches:
        plan = ", ".join("%s: n%d rms%.2f p99 %.2f" % (k, v["n"], v["rms"], v["p99"]) for k, v in p["planarity_by_side"].items())
        print(f"  area {p['area_mm2']:9.1f}  faces {p['n_faces']:7d}  ~{p['nearest_known_plate']:4s} ({p['angle_to_known_deg']:5.1f} deg)"
              f"  centroid {np.round(p['centroid'],1)}  extent {np.round(p['bbox_extent'],1)}  planarity {{{plan}}}")
    # figure: patches coloured by cluster in the base-plane view
    nb = unit(KNOWN["base"])
    e1 = unit(np.cross(nb, [0, 0, 1.0]))
    e2 = np.cross(nb, e1)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=80)
    cmap = plt.get_cmap("tab20")
    for i, p in enumerate(patches):
        k = p["cluster"]
        idx = np.where(lab == k)[0]
        sel = idx[:: max(1, len(idx) // 20000)]
        ax.scatter(cen[sel] @ e1, cen[sel] @ e2, s=0.3, color=cmap(k % 20), lw=0)
    lab_json = json.loads(LABELS.read_text())
    for bb in lab_json["manual_bends"]:
        s_ = np.asarray(bb["line_start"]); e_ = np.asarray(bb["line_end"])
        ax.plot([s_ @ e1, e_ @ e1], [s_ @ e2, e_ @ e2], "-", color="k", lw=1.2)
        mm = 0.5 * (s_ + e_)
        ax.text(mm @ e1, mm @ e2, bb["bend_id"], fontsize=9)
    for p in patches:
        cc = np.asarray(p["centroid"])
        ax.text(cc @ e1, cc @ e2, f"c{p['cluster']}:{p['nearest_known_plate']}\n{p['area_mm2']:.0f}mm2", fontsize=7, color="#333")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_title("auditor1 (resumed) plate census: face-normal clusters (4 deg, sign-insensitive), patches >= 40 mm^2; view along base normal; black = gold")
    fig.savefig(HERE / "a1r_plate_census.png")


if __name__ == "__main__":
    main()
