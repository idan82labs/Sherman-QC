"""Chirality of a compiled stage: best proper-rotation fit of the mirrored stage cloud onto itself.
usage: cad_chirality.py COMPILED_DIR [STAGE] -> json on stdout.
Independent of E1: mirror x, then ICP (point-to-point, scipy KD-tree) from 24 axis-permutation starts about the
centroid + PCA starts; reports RMS / P99 / max NN deviation of the best fit, and clusters of points that deviate
> 1 mm (the features that tell a part from its mirror)."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

cdir = sys.argv[1]
d = json.load(open(f"{cdir}/compiled_part.json"))
stage = sys.argv[2] if len(sys.argv) > 2 else d["stages"][-1]["stage_model_id"]
z = np.load(f"{cdir}/stages/{stage}.npz")
P = z["points"].astype(np.float64)
rng = np.random.default_rng(0)
sub = P[rng.choice(len(P), size=min(len(P), 40000), replace=False)]
tree = cKDTree(P)
c = P.mean(0)
Q0 = (sub - c) * np.array([-1.0, 1.0, 1.0])  # mirrored about the centroid's x plane


def icp(R, t, pts, iters=60):
    for _ in range(iters):
        X = pts @ R.T + t
        dist, idx = tree.query(X)
        keep = dist < max(np.percentile(dist, 90) * 3, 1.0)
        A, B = pts[keep], P[idx[keep]]
        ma, mb = A.mean(0), B.mean(0)
        H = (A - ma).T @ (B - mb)
        U, S, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mb - R @ ma
    X = pts @ R.T + t
    dist, _ = tree.query(X)
    return R, t, float(np.sqrt((dist ** 2).mean()))


starts = Rot.create_group("O").as_matrix()  # 24 proper rotations of the cube
best = None
for R0 in starts:
    R, t, rms = icp(R0, c.copy(), Q0, iters=25)
    if best is None or rms < best[2]:
        best = (R, t, rms)
R, t, _ = icp(best[0], best[1], Q0, iters=80)
# full-cloud deviation of the mirrored CAD after the best proper fit (both directions)
Qall = ((P - c) * np.array([-1.0, 1.0, 1.0])) @ R.T + t
d1, _ = tree.query(Qall)
d2, _ = cKDTree(Qall).query(P)
dd = np.maximum(d1, d2)
out = {"stage": stage, "points": int(len(P)), "rms_mm": float(np.sqrt((d1 ** 2).mean())),
       "rms_sym_mm": float(np.sqrt(((d1 ** 2).mean() + (d2 ** 2).mean()) / 2)),
       "p99_mm": float(np.percentile(dd, 99)), "p999_mm": float(np.percentile(dd, 99.9)), "max_mm": float(dd.max()),
       "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
       "frac_gt_0p5mm": float((dd > 0.5).mean()), "frac_gt_1mm": float((dd > 1.0).mean())}
# clusters of > 1 mm deviation (5 mm linkage)
bad = np.where(dd > 1.0)[0]
clusters = []
if len(bad):
    bt = cKDTree(P[bad]); lab = -np.ones(len(bad), int); k = 0
    for i in range(len(bad)):
        if lab[i] >= 0: continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            for n in bt.query_ball_point(P[bad[j]], 5.0):
                if lab[n] < 0: lab[n] = k; stack.append(n)
        k += 1
    for j in range(k):
        m = bad[lab == j]
        if len(m) < 20: continue
        ext = P[m].max(0) - P[m].min(0)
        clusters.append({"n": int(len(m)), "centre": [round(float(v), 1) for v in P[m].mean(0)],
                         "extent_mm": [round(float(v), 1) for v in ext], "max_dev_mm": round(float(dd[m].max()), 2),
                         "segment_kinds": sorted({int(v) for v in z["segment_kind"][m]})})
    clusters.sort(key=lambda r: -r["n"])
out["deviating_clusters_gt_1mm"] = clusters[:20]
out["n_clusters_gt_1mm"] = len(clusters)
out["bbox_mm"] = [round(float(v), 1) for v in P.max(0) - P.min(0)]
print(json.dumps(out))
