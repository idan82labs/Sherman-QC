"""Joggle analysis at both plate ends (u ~ -165 and u ~ +163), stations along v.

Per station (plane v = v0, slab +-0.6 mm), using the top and bottom skins separately:
  main-plate level and strip level (median w of +w / -w normal points in flat windows),
  slope segment: points between the two levels; fit slope line; joggle angle;
  bend points: intersection of slope line with each level line (virtual sharps) for top and
  bottom skins; convex skin for the upper bend is the top skin, for the lower bend the bottom skin.
Also the gold M04 / M03 position at this v.
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from a1_common import load_all, gold_lines, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T
g = gold_lines()
gp = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in g.items()}


def gold_at_v(key, v0):
    s, e = gp[key]
    t = (v0 - s[1]) / (e[1] - s[1])
    return s + t * (e - s), t


def fit_line(P2):
    c = P2.mean(0)
    _, _, vt = np.linalg.svd(P2 - c, full_matrices=False)
    return c, vt[0]


def analyse(v0, side, half=0.6):
    # side -1: left end (u<0), +1: right end
    m = (np.abs(p[:, 1] - v0) < half) & (side * p[:, 0] > 140)
    P = p[m][:, [0, 2]]; N = n[m][:, [0, 2]]
    out = {"v": float(v0), "n": int(m.sum())}
    if m.sum() < 40:
        out["status"] = "no_points"; return out
    su = side * P[:, 0]
    res = {}
    for skin, sgn in (("top", 1), ("bot", -1)):
        sk = (sgn * N[:, 1] > 0.5)
        main = sk & (su > 142) & (su < 156) & (sgn * N[:, 1] > 0.98)
        strip = sk & (su > 169) & (su < 175) & (sgn * N[:, 1] > 0.98)
        if main.sum() < 5 or strip.sum() < 5:
            res[skin] = None; continue
        wm = float(np.median(P[main, 1])); ws = float(np.median(P[strip, 1]))
        # slope: skin points whose w is between levels (exclusive margins), su in 156..170
        lo, hi = min(wm, ws), max(wm, ws)
        sl = sk & (su > 156) & (su < 170) & (P[:, 1] > lo + 0.3 * (hi - lo)) & (P[:, 1] < hi - 0.3 * (hi - lo))
        if sl.sum() < 3:
            res[skin] = {"w_main": wm, "w_strip": ws, "slope": None}; continue
        c, d = fit_line(P[sl])
        if d[0] * side < 0:
            d = -d
        ang = float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))
        # virtual sharps: slope line meets w = wm and w = ws
        u_main = float(c[0] + (wm - c[1]) / d[1] * d[0]) if abs(d[1]) > 1e-6 else None
        u_strip = float(c[0] + (ws - c[1]) / d[1] * d[0]) if abs(d[1]) > 1e-6 else None
        res[skin] = {"w_main": wm, "w_strip": ws, "slope_deg": ang, "u_sharp_main": u_main,
                     "u_sharp_strip": u_strip, "n_slope": int(sl.sum())}
    out["skins"] = res
    out["status"] = "ok" if res.get("top") and res.get("bot") and res["top"].get("slope_deg") is not None and res["bot"].get("slope_deg") is not None else "partial"
    return out


rows = {"left": [], "right": []}
for side, lab, key in ((-1, "left", "M04"), (1, "right", "M03")):
    for v0 in np.arange(-156, 157, 4.0):
        st = analyse(v0, side)
        q, t = gold_at_v(key, v0)
        st["gold_key"] = key; st["gold_t"] = float(t); st["gold_uw"] = [float(q[0]), float(q[2])]
        rows[lab].append(st)
(OUT / "a1_joggles.json").write_text(json.dumps(rows, indent=1))
for lab in rows:
    print("=====", lab)
    for st in rows[lab]:
        if st["status"] in ("ok", "partial"):
            T = st["skins"]["top"]; B = st["skins"]["bot"]
            def f(x, k):
                return "None" if (x is None or x.get(k) is None) else "%.2f" % x[k]
            print("v=%6.1f n=%4d top: main %s strip %s slope %s sharp_main %s sharp_strip %s | bot: main %s strip %s slope %s sharp_main %s sharp_strip %s | %s(u,w)=(%.2f,%.2f) t=%.2f" % (
                st["v"], st["n"], f(T, "w_main"), f(T, "w_strip"), f(T, "slope_deg"), f(T, "u_sharp_main"), f(T, "u_sharp_strip"),
                f(B, "w_main"), f(B, "w_strip"), f(B, "slope_deg"), f(B, "u_sharp_main"), f(B, "u_sharp_strip"),
                st["gold_key"], st["gold_uw"][0], st["gold_uw"][1], st["gold_t"]))
        else:
            print("v=%6.1f %s" % (st["v"], st["status"]))
