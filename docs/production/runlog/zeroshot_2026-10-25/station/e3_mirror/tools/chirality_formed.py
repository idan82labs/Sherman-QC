"""chirality_formed.py COMPILED_DIR LABEL,LABEL,...: chirality of an arbitrary formed set (out-of-order partial
scans): the formed-set cloud from E1's stage_points_from_compiled, mirrored in x and fitted back by ICP over proper
rotations (24 cube starts, 40k-point subsample), deviation clusters as in chirality_fast.py. JSON line."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject", "/Users/idantal/ShermanQC/FullProject/backend"]
from domains.production.engine.eval.engine_output import load_compiled  # noqa
from domains.production.engine.compile.stages import stage_points_from_compiled  # noqa
cdir, labels = sys.argv[1], [l for l in sys.argv[2].split(",") if l]
comp = load_compiled(f"{cdir}/compiled_part.json")
d = json.load(open(f"{cdir}/compiled_part.json"))
uid = {b["label"]: b["bend_uid"] for b in d["bends"]}
z = np.load(f"{cdir}/stages/{d['stages'][-1]['stage_model_id']}.npz")
step = max(1, len(z["points"]) // 300000)
P, _ = stage_points_from_compiled(comp, z["points"][::step], z["normals"][::step], z["segment_kind"][::step],
                                  z["segment_index"][::step], [uid[l] for l in labels])
P = np.asarray(P, float); tree = cKDTree(P)
rng = np.random.default_rng(0); sub = P[rng.choice(len(P), min(len(P), 40000), replace=False)]
c = P.mean(0); Q0 = (sub - c) * np.array([-1.0, 1, 1])


def icp(R, t, X0, it):
    for _ in range(it):
        X = X0 @ R.T + t; dist, idx = tree.query(X)
        keep = dist < max(np.percentile(dist, 90) * 3, 1.0)
        A, B = X0[keep], P[idx[keep]]; ma, mb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb)); Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ Dm @ U.T; t = mb - R @ ma
    dist, _ = tree.query(X0 @ R.T + t)
    return R, t, float(np.sqrt((dist ** 2).mean()))


best = min((icp(R0, c.copy(), Q0, 20) for R0 in Rot.create_group("O").as_matrix()), key=lambda r: r[2])
R, t, _ = icp(best[0], best[1], Q0, 60)
Q = ((P - c) * np.array([-1.0, 1, 1])) @ R.T + t
d1, _ = tree.query(Q); d2, _ = cKDTree(Q).query(P); dd = np.maximum(d1, d2)
bad = np.where(dd > 1.5)[0]; clusters = []
if len(bad):
    bt = cKDTree(P[bad]); lab = -np.ones(len(bad), int); k = 0
    for i in range(len(bad)):
        if lab[i] >= 0: continue
        st = [i]; lab[i] = k
        while st:
            j = st.pop()
            for n in bt.query_ball_point(P[bad[j]], 3.0):
                if lab[n] < 0: lab[n] = k; st.append(n)
        k += 1
    clusters = sorted([int((lab == j).sum()) for j in range(k) if (lab == j).sum() >= 20], reverse=True)
print(json.dumps({"cdir": cdir, "formed": labels, "rms": float(np.sqrt((d2 ** 2).mean())), "p999": float(np.percentile(dd, 99.9)),
                  "max": float(dd.max()), "clusters_gt_1p5mm": clusters[:10], "distinguishable": bool(clusters)}))
