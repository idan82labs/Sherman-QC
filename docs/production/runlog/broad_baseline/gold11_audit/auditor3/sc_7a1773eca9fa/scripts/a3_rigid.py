"""Auditor 3: is the gold set one locus family moved by a single rigid registration error?"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_loci import analyse, to3  # noqa: E402

pts = {k: [] for k in ("gold", "outer_apex", "mid", "inner_apex", "centre", "outer_sharp", "inner_sharp")}
D, ids = [], []
for bid in ("M01", "M02", "M03", "M04", "M05"):
    F, rows, (lo, hi) = analyse(bid)
    in_g = [r for r in rows if lo <= r["s"] <= hi and r["apex"] is not None and r["Ro"] is not None]
    # 5 stations per line at 10/30/50/70/90 %
    for fr in (0.1, 0.3, 0.5, 0.7, 0.9):
        target = lo + fr * (hi - lo)
        r = min(in_g, key=lambda q: abs(q["s"] - target))
        t = np.median([v for v in r["th"].values()]) if r["th"] else 2.03
        g = r["g"]
        apex = r["apex"]
        beta = r["bend"]
        Ro = r["Ro"]
        pts["gold"].append(to3(F, r["s"], r["gold"]))
        pts["outer_apex"].append(to3(F, r["s"], apex))
        pts["mid"].append(to3(F, r["s"], apex + 0.5 * t * g))
        pts["inner_apex"].append(to3(F, r["s"], apex + t * g))
        pts["centre"].append(to3(F, r["s"], apex + Ro * g))
        pts["outer_sharp"].append(to3(F, r["s"], r["Xo"]))
        Ri = max(Ro - t, 0.1)
        pts["inner_sharp"].append(to3(F, r["s"], apex + t * g - Ri * (1 / np.cos(np.radians(beta) / 2) - 1) * g))
        D.append(F.d)
        ids.append(bid)
D = np.array(D)
G = np.array(pts["gold"])
c0 = G.mean(0)


def fit(A, mode):
    def resid(p):
        if mode == "translation":
            X = G + p[:3]
        else:
            rot = Rotation.from_rotvec(p[:3]).as_matrix()
            X = (G - c0) @ rot.T + c0 + p[3:6]
        v = A - X
        v = v - (v * D).sum(1)[:, None] * D
        return v.ravel()
    p0 = np.zeros(3 if mode == "translation" else 6)
    sol = least_squares(resid, p0)
    v = sol.x
    r = resid(v).reshape(-1, 3)
    per = np.linalg.norm(r, axis=1)
    return v, per


out = {}
for locus in ("outer_apex", "mid", "inner_apex", "centre", "outer_sharp", "inner_sharp"):
    A = np.array(pts[locus])
    before = A - G
    before = before - (before * D).sum(1)[:, None] * D
    b = np.linalg.norm(before, axis=1)
    res = {"before_rms": float(np.sqrt(np.mean(b ** 2)))}
    for mode in ("translation", "rigid"):
        p, per = fit(A, mode)
        perline = {bid: round(float(np.max(per[[i for i, x in enumerate(ids) if x == bid]])), 2) for bid in dict.fromkeys(ids)}
        res[mode] = dict(rms=float(np.sqrt(np.mean(per ** 2))), max=float(per.max()), per_line_max=perline,
                         params=[round(float(x), 3) for x in p])
        if mode == "rigid":
            res[mode]["rot_deg"] = float(np.degrees(np.linalg.norm(p[:3])))
    out[locus] = res
    print(f"{locus:12s} before rms {res['before_rms']:.2f} | translation rms {res['translation']['rms']:.2f} max {res['translation']['max']:.2f} "
          f"t={res['translation']['params']} per-line {res['translation']['per_line_max']} | rigid rms {res['rigid']['rms']:.2f} "
          f"max {res['rigid']['max']:.2f} rot {res['rigid']['rot_deg']:.2f} deg per-line {res['rigid']['per_line_max']}")
json.dump(out, open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_rigid.json", "w"), indent=1)
