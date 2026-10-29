"""Tab-end crease, per-skin fit (fixes the two-skin PCA bias of a2_tab_crease.py)."""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, OUT

q, R, O, raw = load()

def fit(P):
    m = P.mean(0); _, _, vt = np.linalg.svd(P - m, full_matrices=False)
    t = vt[0]
    return m, (t if t[0] > 0 else -t)

def skins(P, m0, t0):
    n = np.array([-t0[1], t0[0]])
    off = (P - m0) @ n
    med = np.median(off)
    return P[off > med], P[off <= med]

out = {}
for name, vlo, vhi in (("B09_tab_end", 24.0, 34.0), ("B10_tab_end", 200.0, 212.0)):
    rows = []
    for w0 in np.arange(2.0, 14.01, 1.0):
        m = (np.abs(q[:, 2] - w0) < 0.3) & (q[:, 0] > -80) & (q[:, 0] < -50) & (q[:, 1] > vlo) & (q[:, 1] < vhi)
        P = q[m][:, :2]
        st = P[P[:, 0] < -62.0]
        sl = P[(P[:, 0] > -56.8) & (P[:, 0] < -53.6)]
        if len(st) < 12 or len(sl) < 10:
            continue
        ms, ts = fit(st); ml, tl = fit(sl)
        a_st = [fit(s)[1] for s in skins(st, ms, ts) if len(s) >= 5]
        a_sl = [fit(s)[1] for s in skins(sl, ml, tl) if len(s) >= 5]
        if len(a_st) < 2 or len(a_sl) < 2:
            continue
        tS = a_st[0] + a_st[1]; tS /= np.linalg.norm(tS)
        tL = a_sl[0] + a_sl[1]; tL /= np.linalg.norm(tL)
        ang = math.degrees(math.acos(np.clip(abs(tS @ tL), -1, 1)))
        skin_ang = [math.degrees(math.acos(np.clip(abs(a @ b), -1, 1))) for a in a_st for b in a_sl]
        rows.append({"w": float(w0), "angle_deg": ang, "skin_pair_angles": [round(x, 1) for x in skin_ang]})
    a = np.array([r["angle_deg"] for r in rows])
    print(name, "n heights", len(rows), "crease angle med %.1f  [%.1f..%.1f]" % (np.median(a), a.min(), a.max()))
    for r in rows:
        print("   w=%.0f angle %.1f skins %s" % (r["w"], r["angle_deg"], r["skin_pair_angles"]))
    out[name] = rows
(OUT / "a2_tab_crease2.json").write_text(json.dumps(out, indent=1))
