"""Candidate folds NOT in the gold, measured with the same station method as a1_stations.py:
  E_neg / E_pos : lower-tray end-flange folds at X~-112 / X~+112 (along Z, N~30)
  W_neg / W_pos : step-wall ends at X~-112 / +112 (along N, Z~72.5) - checks the engine's
                  CADF_CAD_B5 / CADF_CAD_B7 claim of a fold there (expect none)
Probe lines are defined in the part frame and converted to raw. Also refines each found fold's
apex line (least-squares through station apexes) and reports its raw endpoints.
Tab-end 30-deg kinks are measured separately (fit of the two tab portions in N slabs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
import a1common as A
import a1_stations as S

pts, nrm = S.pts, S.nrm
gold = S.gold
X, Z, N = S.X, S.Z, S.N
c = pts.mean(0)
M = np.c_[X, Z, N].T


def raw(p):
    return c + np.asarray(p, float) @ M


tree = cKDTree(pts)
probes = {
    "E_neg": (raw([-112.4, 60.0, 29.9]), raw([-112.4, 130.0, 29.9]), (-N, -X)),
    "E_pos": (raw([111.6, 60.0, 29.9]), raw([111.6, 130.0, 29.9]), (-N, X)),
}
out = {}
for name, (s0, s1, (nA, nB)) in probes.items():
    summ, rows = S.analyze(name, s0, s1, nA, nB, tree=tree, pad=0.0)
    # refine apex line from present stations
    d = S.u(s1 - s0)
    o = S.u(nA + nB); g = -o; v = S.u(np.cross(d, g))
    P3 = []
    for r in rows:
        if r.get("present"):
            cpt = s0 + r["s"] * d
            ap = r["apex"]
            P3.append(cpt + ap[0] * g + ap[1] * v)
    P3 = np.array(P3)
    res = {"summary": summ}
    if len(P3) >= 5:
        m = P3.mean(0)
        _, _, vt = np.linalg.svd(P3 - m)
        dd = vt[0] if vt[0] @ d > 0 else -vt[0]
        t = (P3 - m) @ dd
        a_, b_ = m + t.min() * dd, m + t.max() * dd
        resid = np.linalg.norm((P3 - m) - np.outer(t, dd), axis=1)
        res.update({"apex_line_raw": [a_.tolist(), b_.tolist()], "apex_line_len": float(t.max() - t.min()),
                    "apex_line_part": [((a_ - c) @ M.T).tolist(), ((b_ - c) @ M.T).tolist()],
                    "apex_fit_resid_med": float(np.median(resid)), "R_med": float(np.median([r["R"] for r in rows if r.get("present")]))})
        # angle to gold lines
        for bid, (g0, g1) in gold.items():
            gd = S.u(g1 - g0)
            res.setdefault("angle_to_gold_deg", {})[bid] = float(np.degrees(np.arccos(min(1, abs(gd @ dd)))))
    out[name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != "summary"}))
    print("   fold runs", summ["fold_runs_s_mm"], "R", summ["R_outer_mm(med,p10,p90)"])

# step-wall ends: is there any fold along N at the wall ends? sample the wall end edge region
for name, xs in (("W_neg", -1), ("W_pos", 1)):
    # wall at Z~72.5 between N=-12 and N=28; look at slabs N = -5, 5, 15 around (X=xs*112, Z=72.5)
    rep = []
    P = (pts - c) @ M.T
    for nl in (-5.0, 5.0, 15.0):
        m = (np.abs(P[:, 2] - nl) < 0.4) & (np.abs(P[:, 0] - xs * 110) < 8) & (np.abs(P[:, 1] - 72.5) < 8)
        q = P[m]
        rep.append({"N": nl, "n": int(m.sum()), "X_range": [float(q[:, 0].min()), float(q[:, 0].max())] if len(q) else None,
                    "Z_range": [float(q[:, 1].min()), float(q[:, 1].max())] if len(q) else None})
    out[name] = rep
    print(name, rep)
(A.OUT / "a1_missing.json").write_text(json.dumps(out, indent=1))
