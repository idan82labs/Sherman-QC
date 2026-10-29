"""Diagnostic: independent pose of the CAD final stage in the raw scan by PCA seeds (8 sign combos, proper only) + ICP.
usage: icp_pose.py SCAN CDIR OUT.npz"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from domains.production.engine.inspect.scan_io import read_scan
S = read_scan(sys.argv[1]).points
cdir = Path(sys.argv[2]); cp = json.loads((cdir / "compiled_part.json").read_text())
C = np.load(cdir / cp["stages"][-1]["sample_cloud"]["rel_path"])["points"]
rng = np.random.default_rng(0)
s = S[rng.choice(len(S), 40000, replace=False)]; c = C[rng.choice(len(C), 20000, replace=False)]
def frame(P):
    m = P.mean(0); u, sv, vt = np.linalg.svd(P - m, full_matrices=False); return m, vt
ms, vs = frame(s); mc, vc = frame(c)
ts = cKDTree(s)
best = None
for sx in (1, -1):
    for sy in (1, -1):
        sz = sx * sy * np.linalg.det(vc) * np.linalg.det(vs)
        D = np.diag([sx, sy, sz])
        R = vs.T @ D @ vc  # cad -> scan
        t = ms - R @ mc
        for it in range(40):
            q = c @ R.T + t
            d, i = ts.query(q)
            keep = d < max(3.0, np.percentile(d, 70))
            A, B = c[keep], s[i[keep]]
            ma, mb = A.mean(0), B.mean(0)
            H = (A - ma).T @ (B - mb); U, _, Vt = np.linalg.svd(H)
            Rn = Vt.T @ U.T
            if np.linalg.det(Rn) < 0:
                Vt[-1] *= -1; Rn = Vt.T @ U.T
            R, t = Rn, mb - Rn @ ma
        q = c @ R.T + t; d, _ = ts.query(q)
        d2, _ = cKDTree(q).query(s)
        res = (float(np.mean(d < 2.0)), float(np.mean(d2 < 2.0)), float(np.median(d)))
        print(sx, sy, "cad->scan within 2mm %.3f  scan explained %.3f  median %.2f" % res)
        if best is None or res[1] > best[0][1]:
            best = (res, R, t)
np.savez(sys.argv[3], R=best[1], t=best[2])
print("best", best[0])
