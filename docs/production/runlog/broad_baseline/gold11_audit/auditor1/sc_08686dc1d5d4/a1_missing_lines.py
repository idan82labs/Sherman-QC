"""Approximate outer-apex/virtual-sharp lines (raw frame) of the formed bends that carry no gold line:
both lip folds and the four joggle bends. Read-only; writes a1_missing_bends.json."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from a1_common import load_all, OUT

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T


def fit_line(P2):
    c = P2.mean(0)
    _, _, vt = np.linalg.svd(P2 - c, full_matrices=False)
    return c, vt[0]


def inter(c1, d1, c2, d2):
    M = np.array([d1, -d2]).T
    s, _ = np.linalg.solve(M, c2 - c1)
    return c1 + s * d1


def lip_station(u0, side, half=0.5):
    m = (np.abs(p[:, 0] - u0) < half) & (side * p[:, 1] > 125) & (p[:, 2] < -20)
    P = p[m][:, 1:3]; N = n[m][:, 1:3]
    wall_out = (side * N[:, 0] > 0.97) & (P[:, 1] > -36.5) & (P[:, 1] < -28)
    vw = np.median(P[wall_out, 0]) if wall_out.sum() else None
    if vw is None:
        return None
    lip_bot = (N[:, 1] < -0.97) & (P[:, 1] < -38) & (side * (vw - P[:, 0]) > 4)
    if wall_out.sum() < 8 or lip_bot.sum() < 8:
        return None
    c1, d1 = fit_line(P[wall_out]); c2, d2 = fit_line(P[lip_bot])
    X = inter(c1, d1, c2, d2)
    ang = np.degrees(np.arccos(abs(d1 @ d2)))
    return X, ang


out = {}
for lab, side, (u0, u1) in (("wall1_lip_fold", -1, (-69, 64)), ("wall2_lip_fold", 1, (-64, 68))):
    rows = []
    for u in np.arange(u0, u1 + 0.1, 2.0):
        r = lip_station(u, side)
        if r is not None:
            rows.append([u, r[0][0], r[0][1], r[1]])
    A = np.array(rows)
    cv = np.polyfit(A[:, 0], A[:, 1], 1); cw = np.polyfit(A[:, 0], A[:, 2], 1)
    ends = [np.array([u, np.polyval(cv, u), np.polyval(cw, u)]) for u in (u0, u1)]
    raw = [(o + e @ R).tolist() for e in ends]
    out[lab] = {"locus": "outer virtual sharp (wall outer skin x lip bottom skin)", "u_extent": [u0, u1],
                "length_mm": float(np.linalg.norm(ends[1] - ends[0])), "line_raw": raw,
                "included_angle_wall_lip_deg_median": float(np.median(A[:, 3])), "n_stations": len(A)}

J = json.loads((OUT / "a1_joggles.json").read_text())
for side_lab in ("left", "right"):
    rows = [r for r in J[side_lab] if r["status"] == "ok"]
    for bend, skin, key, wkey in (("upper", "top", "u_sharp_main", "w_main"), ("lower", "bot", "u_sharp_strip", "w_strip")):
        A = np.array([[r["v"], r["skins"][skin][key], r["skins"][skin][wkey], r["skins"][skin]["slope_deg"]] for r in rows])
        # robust: drop outliers > 2 mm from median trend
        cu = np.polyfit(A[:, 0], A[:, 1], 1)
        keep = np.abs(A[:, 1] - np.polyval(cu, A[:, 0])) < 1.0
        A = A[keep]
        cu = np.polyfit(A[:, 0], A[:, 1], 1); cw = np.polyfit(A[:, 0], A[:, 2], 1)
        v0, v1 = A[:, 0].min() - 2, A[:, 0].max() + 2
        ends = [np.array([np.polyval(cu, v), v, np.polyval(cw, v)]) for v in (v0, v1)]
        out[f"{side_lab}_joggle_{bend}_bend"] = {
            "locus": f"{skin}-skin virtual sharp (convex side) of the ~45 deg joggle bend",
            "v_extent_measured": [float(v0), float(v1)], "length_mm": float(np.linalg.norm(ends[1] - ends[0])),
            "line_raw": [(o + e @ R).tolist() for e in ends], "slope_deg_median": float(np.median(A[:, 3])),
            "n_stations": int(len(A)), "fit_rms_u": float(np.std(A[:, 1] - np.polyval(cu, A[:, 0])))}
(OUT / "a1_missing_bends.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, "len %.1f" % v["length_mm"], {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("line_raw", "locus")})
    print("    raw", [[round(x, 2) for x in e] for e in v["line_raw"]])
