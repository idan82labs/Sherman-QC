"""Auditor 2, sc_07062eee17aa: per-station outer-apex / virtual-sharp fit for every gold line.

Independent of render_gold_xsections.py. Uses the part frame (u, w = plate normal, s = gold axis)
and the bend topology read off the dense sections (a2_stations_*.png):
  M01 tab  : plate leg -u, tab leg -w  (convex side +u,+w)
  M02 flange: plate leg +u, flange leg +w (convex side -u,-w)
  M04 tab  : plate leg +u, tab leg -w  (convex side -u,+w)
  M05 flange: plate leg -u, flange leg +w (convex side +u,-w)
M03 / M06 are checked against the flange free edge (no bend topology assumed).

Per station (every STEP mm of s inside the gold span, slab +-HALF mm, window +-WIN mm):
  * bend present?  = enough second-leg points 3..10 mm from the corner
  * outer skins of both legs (TLS line fits on the convex-side skin, >= 3.5 mm from the corner)
  * X_o (outer virtual sharp), X_i (inner virtual sharp), outer apex = X_o + d_min * bisector_in
  * gold - apex decomposed along the inward bisector (+ = into the material / concave side) and across
Writes a2_fit.json (+ prints a summary).
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c  # noqa: E402
from a2_profiles import G, P  # noqa: E402

STEP, HALF, WIN = 2.5, 0.5, 12.0
TOPO = {
    "M01": ((-1.0, 0.0), (0.0, -1.0)),
    "M02": ((1.0, 0.0), (0.0, 1.0)),
    "M04": ((1.0, 0.0), (0.0, -1.0)),
    "M05": ((-1.0, 0.0), (0.0, 1.0)),
}

order = np.argsort(P[:, 2])
Ps = P[order]
s_sorted = Ps[:, 2]


def slab(s0, half=HALF):
    i0, i1 = np.searchsorted(s_sorted, [s0 - half, s0 + half])
    return Ps[i0:i1]


def tls_line(pts):
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m)
    return m, vt[0], np.sqrt(np.mean(((pts - m) @ vt[1]) ** 2))


def intersect(p1, t1, p2, t2):
    A = np.array([t1, -t2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x = np.linalg.solve(A, p2 - p1)
    return p1 + x[0] * t1


def tangents(q, r=1.0):
    tree = cKDTree(q)
    out = np.full((len(q), 2), np.nan)
    for i, nb in enumerate(tree.query_ball_point(q, r)):
        if len(nb) < 4:
            continue
        x = q[nb] - q[nb].mean(0)
        w_, v_ = np.linalg.eigh(x.T @ x)
        if w_[1] < 4 * max(w_[0], 1e-12):
            continue
        out[i] = v_[:, 1]
    return out


def two_skins(off):
    """split 1D offsets into two skins (gap split); returns (low_mask, high_mask) or None."""
    o = np.sort(off)
    if len(o) < 8:
        return None
    gaps = np.diff(o)
    k = int(np.argmax(gaps))
    if gaps[k] < 0.6:
        return None
    thr = (o[k] + o[k + 1]) / 2
    return off < thr, off >= thr


def fit_station(k, s0):
    a, b = G[k]
    lam = (s0 - a[2]) / (b[2] - a[2])
    gp = (a + lam * (b - a))[:2]
    q3 = slab(s0)
    q = q3[:, :2] - gp
    q = q[(np.abs(q[:, 0]) < WIN) & (np.abs(q[:, 1]) < WIN)]
    res = {"s": float(s0), "lam": float(lam), "n": int(len(q))}
    if len(q) < 30:
        res["status"] = "no_points"
        return res
    res["d_nearest_scan"] = float(np.min(np.linalg.norm(q, axis=1)))
    A_dir, B_dir = (np.array(x, float) for x in TOPO[k])
    T = tangents(q)
    ok = np.isfinite(T[:, 0])
    cosA = np.abs(T @ A_dir)
    cosB = np.abs(T @ B_dir)
    # corner guess = gold point (0,0); legs measured from it
    nA = B_dir  # normal of leg A pointing to the concave side (legs are ~orthogonal here)
    nB = A_dir
    legA = ok & (cosA > np.cos(np.radians(20))) & (q @ A_dir > 3.5)
    legB = ok & (cosB > np.cos(np.radians(25))) & (q @ B_dir > 3.5)
    nb_present = int(np.sum(ok & (cosB > np.cos(np.radians(25))) & (q @ B_dir > 3.0) & (q @ B_dir < 10.0)
                        & (np.abs(q @ A_dir) < 6.0)))
    res["legB_pts_3_10mm"] = nb_present
    if legA.sum() < 10 or legB.sum() < 10:
        res["status"] = "no_bend" if legA.sum() >= 10 else "no_leg_A"
        return res
    sa = two_skins(q[legA] @ nA)
    sb = two_skins(q[legB] @ nB)
    if sa is None or sb is None:
        res["status"] = "skins_unresolved"
        return res
    QA, QB = q[legA], q[legB]
    outerA, innerA = QA[sa[0]], QA[sa[1]]  # lower offset toward concave normal = convex side = outer
    outerB, innerB = QB[sb[0]], QB[sb[1]]
    pA, tA, rmsA = tls_line(outerA)
    pB, tB, rmsB = tls_line(outerB)
    piA, tiA, _ = tls_line(innerA)
    piB, tiB, _ = tls_line(innerB)
    Xo = intersect(pA, tA, pB, tB)
    Xi = intersect(piA, tiA, piB, tiB)
    if Xo is None:
        res["status"] = "parallel"
        return res
    # orient tangents away from the corner, bend angle between legs
    tA = tA if tA @ A_dir > 0 else -tA
    tB = tB if tB @ B_dir > 0 else -tB
    inc = np.degrees(np.arccos(np.clip(tA @ tB, -1, 1)))  # included angle between legs
    bis_in = (tA + tB) / np.linalg.norm(tA + tB)  # from the virtual sharp into the concave side
    # outer apex: walk from Xo along bis_in to the first scan point (points within 0.35 mm of bisector)
    rel = q - Xo
    along = rel @ bis_in
    perp = rel - np.outer(along, bis_in)
    near = (np.linalg.norm(perp, axis=1) < 0.35) & (along > -0.5)
    if near.sum() < 1:
        res["status"] = "apex_not_found"
        return res
    d_min = float(np.min(along[near]))
    apex = Xo + d_min * bis_in
    gvec = -apex  # gold is at origin
    res.update({
        "status": "ok",
        "bend_included_deg": float(inc), "bend_angle_deg": float(180 - inc),
        "outer_skin_rms": [float(rmsA), float(rmsB)],
        "Xo": Xo.tolist(), "Xi": None if Xi is None else Xi.tolist(), "apex": apex.tolist(),
        "sharp_to_apex_mm": d_min,
        "gold_to_apex_mm": float(np.linalg.norm(apex)),
        "gold_minus_apex_along_inward_bisector_mm": float(gvec @ bis_in),
        "gold_minus_apex_across_mm": float(gvec @ np.array([-bis_in[1], bis_in[0]])),
        "gold_to_outer_sharp_mm": float(np.linalg.norm(Xo)),
        "gold_to_inner_sharp_mm": None if Xi is None else float(np.linalg.norm(Xi)),
        "thickness_legA_mm": float(np.median(innerA @ nA) - np.median(outerA @ nA)),
    })
    return res


def free_edge_station(k, s0, flange_line):
    """M03/M06: where is the gold relative to the flange top edge (no bend topology assumed)."""
    a, b = G[k]
    lam = (s0 - a[2]) / (b[2] - a[2])
    gp = (a + lam * (b - a))[:2]
    q3 = slab(s0)
    q = q3[:, :2] - gp
    res = {"s": float(s0), "lam": float(lam)}
    win = (np.abs(q[:, 0]) < 6) & (q[:, 1] > -20) & (q[:, 1] < 8)
    qq = q[win]
    res["n"] = int(len(qq))
    if len(qq) < 10:
        res["status"] = "no_flange_here"
        res["d_nearest_scan"] = float(np.min(np.linalg.norm(q, axis=1))) if len(q) else None
        return res
    # flange material = points with |u| < 3 (relative to gold), top edge = max w
    fl = qq[np.abs(qq[:, 0]) < 3.0]
    res["d_nearest_scan"] = float(np.min(np.linalg.norm(q, axis=1)))
    if len(fl) < 5:
        res["status"] = "no_flange_here"
        return res
    top = float(np.max(fl[:, 1]))
    res["flange_top_w_rel_gold"] = top  # >0: flange reaches above gold; <0: gold is above the edge (air)
    # is there anything turning away near the top (lip/hem)? points with |u| in 3..6 within 3 mm of top
    lip = qq[(np.abs(qq[:, 0]) >= 2.2) & (qq[:, 1] > top - 3.0)]
    res["lip_points"] = int(len(lip))
    # width of material near the top (single thickness vs doubled = hem)
    band = fl[(fl[:, 1] < top - 0.6) & (fl[:, 1] > top - 3.0)]
    res["top_band_u_extent_mm"] = float(np.ptp(band[:, 0])) if len(band) > 3 else None
    res["status"] = "ok"
    return res


out = {}
for k in ["M01", "M02", "M04", "M05"]:
    a, b = G[k]
    ss = np.arange(min(a[2], b[2]) + 1.0, max(a[2], b[2]) - 1.0 + 1e-9, STEP)
    out[k] = [fit_station(k, s) for s in ss]
for k, fl in (("M03", "M02"), ("M06", "M05")):
    a, b = G[k]
    ss = np.arange(min(a[2], b[2]) + 1.0, max(a[2], b[2]) - 1.0 + 1e-9, STEP)
    out[k] = [free_edge_station(k, s, fl) for s in ss]

(c.OUT / "a2_fit.json").write_text(json.dumps(out, indent=1))


def summ(k):
    st = out[k]
    a, b = G[k]
    s_lo, s_hi = min(a[2], b[2]), max(a[2], b[2])
    if k in ("M03", "M06"):
        oks = [x for x in st if x["status"] == "ok"]
        tops = np.array([x["flange_top_w_rel_gold"] for x in oks])
        lips = np.array([x["lip_points"] for x in oks])
        print(k, f"stations {len(st)} with flange {len(oks)}; flange top rel gold (mm): min {tops.min():.2f} "
                 f"med {np.median(tops):.2f} max {tops.max():.2f}; lip pts med {np.median(lips):.0f} max {lips.max()}")
        bands = [x["top_band_u_extent_mm"] for x in oks if x["top_band_u_extent_mm"] is not None]
        print("    top band u-extent (single sheet ~1.3-1.6 mm; hem would be ~3):", np.round(np.percentile(bands, [5, 50, 95]), 2))
        d = np.array([x["d_nearest_scan"] for x in st if x.get("d_nearest_scan") is not None])
        print("    gold->nearest scan: med %.2f, frac<=1mm %.2f" % (np.median(d), np.mean(d <= 1.0)))
        return
    oks = [x for x in st if x["status"] == "ok"]
    bend_s = np.array([x["s"] for x in oks])
    print(k, f"span s [{s_lo:.1f},{s_hi:.1f}] len {s_hi - s_lo:.1f}; stations {len(st)}; bend fitted at {len(oks)}")
    stat = {}
    for x in st:
        stat[x["status"]] = stat.get(x["status"], 0) + 1
    print("    status counts", stat)
    if len(oks):
        # contiguous runs of bend presence
        runs, cur = [], [bend_s[0], bend_s[0]]
        for s in bend_s[1:]:
            if s - cur[1] <= STEP * 1.6:
                cur[1] = s
            else:
                runs.append(cur)
                cur = [s, s]
        runs.append(cur)
        print("    bend runs (s):", [(round(r[0], 1), round(r[1], 1)) for r in runs])
        g2a = np.array([x["gold_to_apex_mm"] for x in oks])
        al = np.array([x["gold_minus_apex_along_inward_bisector_mm"] for x in oks])
        ac = np.array([x["gold_minus_apex_across_mm"] for x in oks])
        ang = np.array([x["bend_angle_deg"] for x in oks])
        d_os = np.array([x["gold_to_outer_sharp_mm"] for x in oks])
        d_is = np.array([x["gold_to_inner_sharp_mm"] for x in oks if x["gold_to_inner_sharp_mm"] is not None])
        thk = np.array([x["thickness_legA_mm"] for x in oks])
        sa = np.array([x["sharp_to_apex_mm"] for x in oks])
        pr = lambda v: f"med {np.median(v):.2f} [p5 {np.percentile(v, 5):.2f}, p95 {np.percentile(v, 95):.2f}]"
        print("    gold->apex", pr(g2a), "| along inward bisector", pr(al), "| across", pr(ac))
        print("    gold->outer sharp", pr(d_os), "| gold->inner sharp", pr(d_is), "| sharp->apex", pr(sa))
        print("    bend angle", pr(ang), "| thickness", pr(thk))
        print("    frac stations with gold<=4mm of apex: %.2f" % np.mean(g2a <= 4.0))


for k in ["M01", "M02", "M03", "M04", "M05", "M06"]:
    summ(k)
