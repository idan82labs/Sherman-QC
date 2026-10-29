"""chirality_fast.py COMPILED_DIR [STAGE ...]: per stage, the CAD against its mirror image under each of E1's MIRROR
candidates (the best one by our max-deviation metric is kept): RMS, P99.9, max, share > 1 mm, and the clusters of
> 1 mm deviation (features that tell the part from its mirror). 'distinguishable' = a cluster of >= 20 points
(>= ~5 mm2 at 0.5 mm sampling) deviating > 1.5 mm. JSON line per stage."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
cdir = sys.argv[1]
d = json.load(open(f"{cdir}/compiled_part.json"))
stages = sys.argv[2:] or [s["stage_model_id"] for s in d["stages"]]
for st in d["stages"]:
    if st["stage_model_id"] not in stages: continue
    P = np.load(f"{cdir}/stages/{st['stage_model_id']}.npz")["points"].astype(float)
    if len(P) > 300000: P = P[:: int(np.ceil(len(P) / 300000))]
    tree = cKDTree(P); best = None
    for c in st["symmetry"]["candidates"]:
        if c["kind"] != "MIRROR": continue
        M = np.asarray(c["transform"]); Q = P @ M[:3, :3].T + M[:3, 3]
        d1, _ = tree.query(Q); d2, _ = cKDTree(Q).query(P); dd = np.maximum(d1, d2)
        key = float(np.percentile(dd, 99.5))
        if best is None or key < best[0]: best = (key, c, dd)
    if best is None:
        print(json.dumps({"cdir": cdir, "stage": st["stage_model_id"], "no_mirror_candidate": True})); continue
    _, c, dd = best
    bad = np.where(dd > 1.5)[0]; clusters = []
    if len(bad):
        bt = cKDTree(P[bad]); lab = -np.ones(len(bad), int); k = 0
        for i in range(len(bad)):
            if lab[i] >= 0: continue
            stack = [i]; lab[i] = k
            while stack:
                j = stack.pop()
                for n in bt.query_ball_point(P[bad[j]], 3.0):
                    if lab[n] < 0: lab[n] = k; stack.append(n)
            k += 1
        for j in range(k):
            m = bad[lab == j]
            if len(m) >= 20: clusters.append(int(len(m)))
    print(json.dumps({"cdir": cdir, "stage": st["stage_model_id"], "e1_rms": c["rms_mm"], "e1_threshold": st["symmetry"]["threshold_rms_mm"],
                      "rms": float(np.sqrt((dd ** 2).mean())), "p999": float(np.percentile(dd, 99.9)), "max": float(dd.max()),
                      "frac_gt_1mm": float((dd > 1).mean()), "clusters_gt_1p5mm": sorted(clusters, reverse=True)[:10],
                      "distinguishable": bool(clusters)}))
