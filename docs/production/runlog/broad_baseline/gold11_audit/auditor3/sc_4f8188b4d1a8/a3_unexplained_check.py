"""Auditor-3 completeness check for sc_4f8188b4d1a8: are there formed folds that neither the gold nor
the measured missing-bend list (a3_missing_bends.json) explains?

Points whose outward normal is 'intermediate' (12..78 deg away from every part axis) are the
candidates for bend arcs (plus oblique cut edges, hole rims, deburr rounds, oblique flat faces).
Points within EXCL mm of a known fold (gold B01-B06, B09, B10 over their measured fold extents, and
the measured M1-M4 apex lines) are removed; the rest is voxel-clustered and every cluster is
reported with its part-frame bbox, size and elongation, and rendered for visual inspection.
Read-only on data; writes JSON + PNG into this auditor folder only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402

EXCL = 5.0
ANG_LO, ANG_HI = 12.0, 78.0
VOX = 1.5


def seg_dist(q, a, b):
    ab = b - a
    t = np.clip(((q - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(q - (a + t[:, None] * ab), axis=1)


def main():
    p, n, _ = c.load_scan()
    R = np.load(c.CACHE / "part_R.npy")
    o = np.load(c.CACHE / "part_o.npy")
    q = (p - o) @ R.T
    nq = n @ R.T
    good = np.linalg.norm(nq, axis=1) > 0.5
    ang = np.degrees(np.arccos(np.clip(np.abs(nq).max(1), 0, 1)))
    cand = good & (ang > ANG_LO) & (ang < ANG_HI)
    ext = json.loads((c.OUT / "a3_extent_match.json").read_text())
    miss = json.loads((c.OUT / "a3_missing_bends.json").read_text())
    gold = {g["id"]: g for g in c.load_gold()[0]}
    known = {}
    for bid, g in gold.items():
        L = float(np.linalg.norm(g["e"] - g["s"]))
        d = (g["e"] - g["s"]) / L
        if bid in ext:
            lo, hi = ext[bid]["fold_extent_s_mm"]
            a_, b_ = ext[bid]["apex_ref_line"]
            known[bid] = (np.asarray(a_), np.asarray(b_))
        else:
            known[bid] = (g["s"], g["e"])
    for k, v in miss.items():
        if v.get("found"):
            known[k] = (np.asarray(v["apex_line_raw"][0]), np.asarray(v["apex_line_raw"][1]))
    near = np.zeros(len(p), bool)
    for k, (a_, b_) in known.items():
        if k in ("B07", "B08"):
            continue  # not used to explain anything (audited separately)
        near |= seg_dist(p, a_, b_) < EXCL
    left = cand & ~near
    idx = np.nonzero(left)[0]
    print("candidates", int(cand.sum()), "explained", int((cand & near).sum()), "left", len(idx))
    # voxel clustering (26-connectivity) in part frame
    lo = q[idx].min(0) - 2 * VOX
    vi = np.floor((q[idx] - lo) / VOX).astype(int)
    shape = vi.max(0) + 2
    grid = np.zeros(shape, bool)
    grid[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    lab, nlab = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    pl = lab[vi[:, 0], vi[:, 1], vi[:, 2]]
    clusters = []
    for k in range(1, nlab + 1):
        m = pl == k
        if m.sum() < 25:
            continue
        pts = q[idx[m]]
        cen = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - cen, full_matrices=False)
        ext_ax = np.ptp((pts - cen) @ vt.T, axis=0)
        nn = nq[idx[m]]
        clusters.append({
            "n": int(m.sum()), "centroid_pf": cen.round(1).tolist(),
            "bbox_pf": [pts.min(0).round(1).tolist(), pts.max(0).round(1).tolist()],
            "principal_extent_mm": ext_ax.round(1).tolist(),
            "principal_dir_pf": vt[0].round(2).tolist(),
            "median_abs_normal_pf": np.median(np.abs(nn), 0).round(2).tolist(),
        })
    clusters.sort(key=lambda x: -x["n"])
    for cl in clusters[:60]:
        print(cl)
    (c.OUT / "a3_unexplained_clusters.json").write_text(json.dumps(
        {"params": {"EXCL_mm": EXCL, "ang_window_deg": [ANG_LO, ANG_HI], "voxel_mm": VOX},
         "known_lines_used": sorted(k for k in known if k not in ("B07", "B08")),
         "n_candidates": int(cand.sum()), "n_left": int(len(idx)), "clusters": clusters}, indent=1))
    # render
    rng = np.random.default_rng(1)
    bg = rng.choice(len(q), 120000, replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, (i, j, title) in zip(axs, [(0, 1, "top X'-Y'"), (0, 2, "front X'-Z'"), (1, 2, "side Y'-Z'")]):
        ax.scatter(q[bg, i], q[bg, j], s=0.2, c="0.85", rasterized=True)
        ax.scatter(q[cand & near, i], q[cand & near, j], s=0.4, c="tab:green", rasterized=True)
        ax.scatter(q[idx, i], q[idx, j], s=1.2, c="red", rasterized=True)
        for k, (a_, b_) in known.items():
            a2, b2 = (a_ - o) @ R.T, (b_ - o) @ R.T
            col = "black" if k.startswith("M") else ("orange" if k in ("B07", "B08") else "blue")
            ax.plot([a2[i], b2[i]], [a2[j], b2[j]], "-", c=col, lw=1)
            ax.text((a2[i] + b2[i]) / 2, (a2[j] + b2[j]) / 2, k.split("_")[0], color=col, fontsize=9)
        for ci, cl in enumerate(clusters[:40]):
            ax.text(cl["centroid_pf"][i], cl["centroid_pf"][j], f"c{ci}", color="darkred", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    fig.suptitle("auditor3 sc_4f8188b4d1a8 completeness: intermediate-normal points (12-78 deg off every part axis); "
                 f"green = within {EXCL} mm of a known fold (gold blue / measured missing M black), red = unexplained; "
                 "orange = gold B07/B08 (not used)")
    fig.tight_layout()
    fig.savefig(c.OUT / "a3_unexplained_overview.png", dpi=70)
    print(c.OUT / "a3_unexplained_overview.png")


if __name__ == "__main__":
    main()
