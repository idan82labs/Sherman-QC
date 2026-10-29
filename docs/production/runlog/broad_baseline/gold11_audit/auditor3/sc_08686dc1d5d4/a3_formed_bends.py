#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4: measure every FORMED bend of scan_1 as an outer-apex line, then score
each gold line against them with the frozen matcher's pair_metrics (5 deg / 4 mm / 60 % both spans).

Formed bends measured from the scan (independent of gold and engine):
  wallA_base, wallB_base   plate -> wall 90 deg bends (apex from the dense section fits, extent = wall u-range)
  wallA_foot, wallB_foot   wall -> flange 90 deg bends at the wall bottom (extent = foot u-range)
  joggleL_1/2, joggleR_1/2 the two ~35 deg bends of each edge joggle (mid-surface breakpoints; for R_o ~1.6 mm
                           and 35 deg the apex sits < 0.1 mm from the sharp, so the breakpoint is used)
Writes formed_bends_apex.json and gold_vs_formed_metrics.json here (no write anywhere else).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/production/runlog/broad_baseline/gold11_audit/scripts"))
from a3_global import gold_lines, load_points  # noqa: E402
from render_gold_xsections import fit_section  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


def fit_line(pts, u_axis, o, ulo, uhi):
    """PCA line through pts, clipped to the plane-frame coordinate range [ulo, uhi] along u_axis."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] if vt[0] @ u_axis > 0 else -vt[0]
    t_lo = ((o + ulo * u_axis) - c) @ u_axis / (d @ u_axis)
    t_hi = ((o + uhi * u_axis) - c) @ u_axis / (d @ u_axis)
    rms = float(np.sqrt(np.mean(np.sum(((pts - c) - np.outer((pts - c) @ d, d)) ** 2, axis=1))))
    return (c + t_lo * d), (c + t_hi * d), rms


def main():
    from scipy.spatial import cKDTree

    P = load_points()
    o, e1, e2, n = np.load(HERE / "_plate_frame.npy")
    X = P - o
    U, V, H = X @ e1, X @ e2, X @ n
    tree = cKDTree(P)
    st = json.loads((HERE / "stations_all.json").read_text())
    prof = json.loads((HERE / "profiles.json").read_text())
    bends = {}

    # ---- base bends from the dense section fits along M01 / M02 (apex only where a fold was found)
    for name, bid, key in (("wallA_base", "M01", "wallA_M01_side"), ("wallB_base", "M02", "wallB_M02_side")):
        ap = np.array([r["apex_xyz"] for r in st[bid]["rows"] if r.get("fold_here") and r.get("apex_xyz")])
        ulo, uhi = prof[key]["wall_u_min_max"]
        a, b, rms = fit_line(ap, e1, o, ulo, uhi)
        bends[name] = {"start": a.tolist(), "end": b.tolist(), "n_apex": int(len(ap)), "rms_mm": rms,
                       "u_range": [ulo, uhi], "kind": "90deg plate->wall"}

    # ---- foot bends: sections perpendicular to u at the wall bottom
    for name, sgn, key in (("wallA_foot", -1, "wallA_M01_side"), ("wallB_foot", 1, "wallB_M02_side")):
        foot = (V * sgn > 130) & (V * sgn < 153) & (H < -37)
        ulo, uhi = float(U[foot].min()), float(U[foot].max())
        vwall = prof[key]["wall_v_median"]
        apx, angs = [], []
        for u0 in np.arange(ulo + 3, uhi - 2, 3.0):
            c = o + u0 * e1 + vwall * e2 - 39.5 * n
            idx = np.array(tree.query_ball_point(c, 30), int)
            Q = P[idx] - c
            m = np.abs(Q @ e1) < 0.6
            q = np.c_[Q[m] @ e2, Q[m] @ n]
            q = q[np.linalg.norm(q, axis=1) < 14]
            fit = fit_section(q, 0.52)
            if fit.get("ok") and fit.get("apex") is not None and 60 < fit["bend_angle_deg"] < 120:
                apx.append(c + fit["apex"][0] * e2 + fit["apex"][1] * n)
                angs.append(fit["bend_angle_deg"])
        apx = np.array(apx)
        a, b, rms = fit_line(apx, e1, o, ulo, uhi)
        bends[name] = {"start": a.tolist(), "end": b.tolist(), "n_apex": int(len(apx)), "rms_mm": rms,
                       "u_range": [ulo, uhi], "bend_angle_median": float(np.median(angs)), "kind": "90deg wall->flange"}

    # ---- joggles: mid-surface h(u) breakpoints in v-slabs
    def joggle_breaks(ua, ub, v0):
        m = (np.abs(V - v0) < 0.6) & (U > ua) & (U < ub) & (np.abs(H) < 4)
        uu, hh = U[m], H[m]
        bins = np.arange(ua, ub, 0.25)
        mid = []
        for b0 in bins:
            mm = (uu >= b0) & (uu < b0 + 0.25)
            if mm.sum() < 2:
                mid.append(np.nan)
                continue
            mid.append(0.5 * (hh[mm].max() + hh[mm].min()) if np.ptp(hh[mm]) > 1.0 else np.nan)
        x = bins + 0.125
        y = np.array(mid)
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        best = None
        # exhaustive 2-breakpoint continuous piecewise-linear fit
        for i in range(3, len(x) - 6):
            for j in range(i + 2, min(i + 30, len(x) - 3)):
                b1, b2 = x[i], x[j]
                A = np.c_[np.ones_like(x), x, np.clip(x - b1, 0, None), np.clip(x - b2, 0, None)]
                coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = float(np.sum((A @ coef - y) ** 2))
                if best is None or r < best[0]:
                    best = (r, b1, b2, coef)
        r, b1, b2, coef = best
        f = lambda t: coef[0] + coef[1] * t + coef[2] * max(t - b1, 0) + coef[3] * max(t - b2, 0)  # noqa: E731
        s0 = coef[1]
        s1 = coef[1] + coef[2]
        s2 = coef[1] + coef[2] + coef[3]
        ang1 = math.degrees(abs(math.atan(s1) - math.atan(s0)))
        ang2 = math.degrees(abs(math.atan(s2) - math.atan(s1)))
        return (b1, f(b1)), (b2, f(b2)), ang1, ang2, math.sqrt(r / len(x))

    for side, (ua, ub), key in (("joggleL", (-176, -152), None), ("joggleR", (150, 173), None)):
        pts1, pts2, a1s, a2s = [], [], [], []
        vlo = max(float(V[(U > ua) & (U < ub)].min()), -153) + 4
        vhi = min(float(V[(U > ua) & (U < ub)].max()), 153) - 4
        for v0 in np.arange(-145, 146, 10.0):
            # skip slabs crossing the edge holes (v ~ -133, 0, 133)
            if min(abs(v0 - hv) for hv in (-133.3, 0.0, 133.3)) < 6:
                continue
            (u1, h1), (u2, h2), a1, a2, rr = joggle_breaks(ua, ub, v0)
            pts1.append(o + u1 * e1 + v0 * e2 + h1 * n)
            pts2.append(o + u2 * e1 + v0 * e2 + h2 * n)
            a1s.append(a1)
            a2s.append(a2)
        m = (U > ua) & (U < ub) & (np.abs(H) < 4)
        vext = [float(np.percentile(V[m], 0.05)), float(np.percentile(V[m], 99.95))]
        for k, pts, angs in ((1, np.array(pts1), a1s), (2, np.array(pts2), a2s)):
            c = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
            d = vt[0] if vt[0] @ e2 > 0 else -vt[0]
            ta = ((o + vext[0] * e2) - c) @ e2 / (d @ e2)
            tb = ((o + vext[1] * e2) - c) @ e2 / (d @ e2)
            rms = float(np.sqrt(np.mean(np.sum(((pts - c) - np.outer((pts - c) @ d, d)) ** 2, axis=1))))
            uu = (pts - o) @ e1
            bends[f"{side}_{k}"] = {"start": (c + ta * d).tolist(), "end": (c + tb * d).tolist(), "n_sections": int(len(pts)),
                                    "rms_mm": rms, "u_mean": float(uu.mean()), "u_sd": float(uu.std()),
                                    "v_range": vext, "bend_angle_median": float(np.median(angs)),
                                    "kind": "joggle bend (mid-surface breakpoint ~ apex for R_o 1.6 mm)"}

    for k, b in bends.items():
        s, e = np.array(b["start"]), np.array(b["end"])
        b["length_mm"] = float(np.linalg.norm(e - s))
        b["uvh_start"] = [float((s - o) @ e1), float((s - o) @ e2), float((s - o) @ n)]
        b["uvh_end"] = [float((e - o) @ e1), float((e - o) @ e2), float((e - o) @ n)]
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in b.items() if kk not in ("start", "end")})
    (HERE / "formed_bends_apex.json").write_text(json.dumps(bends, indent=1))

    # ---- frozen matcher metrics: every gold line vs every formed bend
    th = Thresholds()
    rows = []
    for bid, s0, s1 in gold_lines():
        g = np.array([s0, s1])
        best = None
        for k, b in bends.items():
            p = np.array([b["start"], b["end"]])
            mtr = pair_metrics(g, p)
            ok = is_match(mtr, th)
            rows.append({"gold": bid, "formed": k, "match": bool(ok), **{kk: float(vv) for kk, vv in mtr.items()}})
            if mtr["axis_angle_deg"] <= 10 and (best is None or mtr["transverse_offset_mm"] < best[1]["transverse_offset_mm"]):
                best = (k, mtr, ok)
        print(bid, "nearest parallel formed bend:", best[0] if best else None,
              {kk: round(vv, 3) for kk, vv in best[1].items()} if best else None, "MATCH" if best and best[2] else "no match")
    (HERE / "gold_vs_formed_metrics.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
