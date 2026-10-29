"""Emulate bench_labeling plane_from_points on the prep ds cloud (no raw mapping) to choose web seeds."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.segment import fit_plane

PREP = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/prep/sc_71cf0b216f43")
A = Path(__file__).resolve().parent
fr = json.loads((A / "part_frame.json").read_text()); R = np.stack([fr["ex"], fr["ey"], fr["ez"]]); o = np.array(fr["origin"])
ds = np.load(PREP / "points_ds.npz"); P = ds["points"]; N = ds["normals"]
man = json.loads((PREP / "prep_manifest.json").read_text()); voxel = man["voxel_mm"]; t = 1.45
tree = cKDTree(P)
cos20 = np.cos(np.radians(20.0))

def probe(seed, radius, grow=True, tol=None):
    tol = tol if tol is not None else min(0.6, 0.45 * t)
    seed = np.asarray(seed, float)
    sd, si = tree.query(seed)
    if sd > max(2.0, 3 * voxel):
        return {"error": f"seed {sd:.2f} mm off"}
    ball = np.asarray(tree.query_ball_point(seed, radius), dtype=np.int64)
    n0 = N[si] / np.linalg.norm(N[si]); x0 = P[si]
    ok = lambda idx, n: (N[idx] @ n) >= cos20
    cand = ball[ok(ball, n0) & (np.abs((P[ball] - x0) @ n0) <= 0.45 * t)]
    if len(cand) < 20:
        return {"error": f"only {len(cand)} cand", "seed_dist": float(sd)}
    n, c, _, _ = fit_plane(P[cand]); n = n if n @ n0 > 0 else -n
    inl = ball[ok(ball, n) & (np.abs((P[ball] - c) @ n) <= tol)]
    if len(inl) < 20:
        return {"error": f"only {len(inl)} inliers"}
    n, c, rms, _ = fit_plane(P[inl]); n = n if n @ n0 > 0 else -n
    region = set(int(i) for i in inl)
    if grow:
        frontier = list(region)
        while frontier:
            nb = tree.query_ball_point(P[frontier], 2.5 * voxel)
            new = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in nb]))
            new = np.array([i for i in new if i not in region], dtype=np.int64)
            if not len(new):
                break
            new = new[ok(new, n) & (np.abs((P[new] - c) @ n) <= tol)]
            region.update(int(i) for i in new); frontier = [int(i) for i in new]
    reg = np.fromiter(region, dtype=np.int64)
    q = (P[reg] - o) @ R.T
    npart = R @ n
    return {"seed_dist": round(float(sd), 2), "cand": int(len(cand)), "inl": int(len(inl)), "region": int(len(reg)),
            "rms": round(float(rms), 3), "normal_part": np.round(npart, 3).tolist(),
            "tilt_deg": round(float(np.degrees(np.arccos(min(1, abs(npart[2]))))), 1),
            "x_range": [round(float(q[:, 0].min()), 1), round(float(q[:, 0].max()), 1)],
            "y_range": [round(float(q[:, 1].min()), 2), round(float(q[:, 1].max()), 2)],
            "z_range": [round(float(q[:, 2].min()), 2), round(float(q[:, 2].max()), 2)]}

webs = json.loads((A / "joggle_webs.json").read_text())
for key in ("S9_top", "S9_bottom", "S10_top", "S10_bottom"):
    for s in webs[key]["seeds"]:
        for r in (8.0, 15.0, 40.0):
            res = probe(s["raw"], r)
            print(key, "x", s["x"], "r", r, res)
