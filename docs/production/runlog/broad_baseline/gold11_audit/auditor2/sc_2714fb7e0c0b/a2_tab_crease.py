"""Measure the vertical crease at the +u end of each window tab (B09 / B10) per height w (light, read-only)."""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, OUT

q, R, O, raw = load()

def fit(P):
    m = P.mean(0); _, _, vt = np.linalg.svd(P - m, full_matrices=False); return m, vt[0]

res = {}
for name, vlo, vhi in (("B09_tab_end", 24.0, 34.0), ("B10_tab_end", 200.0, 212.0)):
    rows = []
    for w0 in np.arange(1.0, 15.01, 1.0):
        m = (np.abs(q[:, 2] - w0) < 0.3) & (q[:, 0] > -80) & (q[:, 0] < -50) & (q[:, 1] > vlo) & (q[:, 1] < vhi)
        P = q[m][:, :2]
        if len(P) < 30:
            rows.append({"w": float(w0), "n": int(len(P))}); continue
        st = P[P[:, 0] < -62.5]
        sl = P[(P[:, 0] > -57.0) & (P[:, 0] < -53.5)]
        if len(st) < 10 or len(sl) < 8:
            rows.append({"w": float(w0), "n": int(len(P)), "why": "few"}); continue
        m1, t1 = fit(st); m2, t2 = fit(sl)
        if t1[0] < 0: t1 = -t1
        if t2[0] < 0: t2 = -t2
        ang = math.degrees(math.acos(np.clip(abs(t1 @ t2), -1, 1)))
        M = np.array([t1, -t2]).T
        s, _ = np.linalg.solve(M, m2 - m1)
        X = m1 + s * t1
        tip_u = float(P[:, 0].max())
        rows.append({"w": float(w0), "n": int(len(P)), "crease_angle_deg": ang, "mid_sharp_uv": X.round(2).tolist(),
                     "straight_mid_v": float(m1[1]), "tip_u": tip_u})
    res[name] = rows
    good = [r for r in rows if "crease_angle_deg" in r]
    if good:
        a = np.array([r["crease_angle_deg"] for r in good]); xs = np.array([r["mid_sharp_uv"] for r in good])
        print(name, "heights with crease", [r["w"] for r in good])
        print("   angle med %.1f [%.1f..%.1f]  sharp u med %.2f [%.2f..%.2f]  v med %.2f  tip u med %.1f" % (
            np.median(a), a.min(), a.max(), np.median(xs[:, 0]), xs[:, 0].min(), xs[:, 0].max(), np.median(xs[:, 1]),
            np.median([r["tip_u"] for r in good])))
# lowest attachment: where does the tab bottom sit near u=-59 (is crease running down into the B09/B10 fold zone?)
for name, vlo, vhi in (("B09", 24.0, 34.0), ("B10", 200.0, 212.0)):
    for u0 in (-62.0, -60.0, -59.0, -58.0, -56.0):
        m = (np.abs(q[:, 0] - u0) < 0.3) & (q[:, 1] > vlo) & (q[:, 1] < vhi) & (q[:, 2] > -0.3) & (q[:, 2] < 4)
        W = q[m][:, 2]
        V = q[m][:, 1]
        # tab points = those above the plate top skin level (w > 1.2)
        tab = W > 1.2
        print(name, "u=%.0f" % u0, "n", int(m.sum()), "min w of tab-region pts above 1.2: %s" % (None if not tab.any() else round(float(W[tab].min()), 2)),
              "plate-top pts (w<1.2):", int((~tab).sum()))
(OUT / "a2_tab_crease.json").write_text(json.dumps(res, indent=1))
