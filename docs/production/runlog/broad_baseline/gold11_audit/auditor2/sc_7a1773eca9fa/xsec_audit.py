#!/usr/bin/env python3
"""Auditor 2 - independent per-station cross-section audit of Gold11 sc_7a1773eca9fa (49001000).

Read-only on the scan and the gold. For every gold line:
  1. true bend axis from the two dominant flange-normal clusters near the line (not the gold direction);
  2. stations every 1 mm along the true axis (gold span +-80 mm): slab |s - s_k| <= HW,
     2D frame perpendicular to the axis; flange/skin classes from oriented vertex normals;
  3. per station: outer/inner flange lines, outer/inner virtual sharp (X_o, X_i), bisector,
     outer apex (first OUTER-skin point along the bisector from X_o = bench_labeling 'apex'),
     inner apex, arc circle fits (R_o, R_i), and the gold point's offset from the outer apex
     (along bisector, + = toward the concave side; across, + = toward flange A);
  4. fold extent along the axis (both flanges present next to the corner) -> coverage of the gold.
Writes stations_<id>.json + summary.json next to this file. Plots are made by plot_xsec.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else None
GOLD = {
    "M01": ([-33.424, -66.283, 242.807], [-285.081, -376.09, 269.038]),
    "M02": ([-340.909, -287.798, 309.126], [-364.389, -246.464, 329.952]),
    "M03": ([-171.434, -12.045, 312.006], [-127.978, -28.787, 289.098]),
    "M04": ([-290.48, -373.313, 264.218], [-312.633, -334.317, 283.867]),
    "M05": ([-77.895, -47.862, 262.064], [-36.896, -63.658, 240.452]),
}
HW = 0.5          # slab half width (mm)
WIN = 30.0        # 2D window radius around the corner (mm)
FLANGE_TOL = np.cos(np.radians(12.0))
T_DEFAULT = 2.05


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def basis(d):
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(d, a))
    e2 = np.cross(d, e1)
    return e1, e2


def axial_peaks(theta, nbins=180):
    """two dominant axial directions (mod pi) of 2D normal angles."""
    h, edges = np.histogram(np.mod(theta, np.pi), bins=nbins, range=(0, np.pi))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    i1 = int(np.argmax(hs))
    d = np.minimum(np.abs(np.arange(nbins) - i1), nbins - np.abs(np.arange(nbins) - i1))
    hs2 = np.where(d > 15, hs, -1)
    i2 = int(np.argmax(hs2))
    c = (edges[:-1] + edges[1:]) / 2
    return c[i1], c[i2], int(hs[i1]), int(hs2[i2])


def true_axis(P, N, a, b):
    u = unit(b - a)
    L = np.linalg.norm(b - a)
    axis = u
    for _ in range(3):
        rel = P - a
        s = rel @ axis
        w = rel - np.outer(s, axis)
        r = np.linalg.norm(w, axis=1)
        sel = (s > 0.08 * L) & (s < 0.92 * L) & (r > 2.0) & (r < 22.0)
        n = N[sel]
        n = n[np.abs(n @ axis) < 0.35]
        e1, e2 = basis(axis)
        th = np.arctan2(n @ e2, n @ e1)
        t1, t2, c1, c2 = axial_peaks(th)
        m1 = np.cos(t1) * e1 + np.sin(t1) * e2
        m2 = np.cos(t2) * e1 + np.sin(t2) * e2
        # refine each 3D normal cluster by averaging sign-aligned members
        g1 = np.abs(n @ m1) > np.cos(np.radians(8))
        g2 = np.abs(n @ m2) > np.cos(np.radians(8))
        n1 = unit((n[g1] * np.sign(n[g1] @ m1)[:, None]).mean(0))
        n2 = unit((n[g2] * np.sign(n[g2] @ m2)[:, None]).mean(0))
        new = unit(np.cross(n1, n2))
        if new @ u < 0:
            new = -new
        axis = new
    return axis, n1, n2, int(g1.sum()), int(g2.sum())


def fit_line2d(Q):
    c = Q.mean(0)
    X = Q - c
    w, v = np.linalg.eigh(X.T @ X)
    t = v[:, 1]
    nrm = v[:, 0]
    res = X @ nrm
    return c, t, nrm, float(np.sqrt(np.mean(res ** 2)))


def intersect(c1, t1, c2, t2):
    M = np.array([t1, -t2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    s, _ = np.linalg.solve(M, c2 - c1)
    return c1 + s * t1


def circle_fit(Q):
    if len(Q) < 6:
        return None
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    R = float(np.sqrt(sol[2] + c @ c))
    res = np.linalg.norm(Q - c, axis=1) - R
    return c, R, float(np.sqrt(np.mean(res ** 2)))


def analyse_line(bid, P, N, tree):
    a = np.array(GOLD[bid][0], float)
    b = np.array(GOLD[bid][1], float)
    L = float(np.linalg.norm(b - a))
    ug = unit(b - a)
    axis, n1, n2, c1, c2 = true_axis(P, N, a, b)
    ang_gold_axis = float(np.degrees(np.arccos(min(1.0, abs(ug @ axis)))))
    e1, e2 = basis(axis)
    # 2D flange normal directions
    m1 = unit(np.array([n1 @ e1, n1 @ e2]))
    m2 = unit(np.array([n2 @ e1, n2 @ e2]))
    s_a, s_b = float(a @ axis), float(b @ axis)
    s_lo, s_hi = min(s_a, s_b), max(s_a, s_b)
    rel_all = P @ axis
    stations = []
    pad = 80.0
    region = (rel_all > s_lo - pad - 2) & (rel_all < s_hi + pad + 2)
    idx_region = np.flatnonzero(region)
    Pr, Nr, sr = P[idx_region], N[idx_region], rel_all[idx_region]
    order = np.argsort(sr)
    Pr, Nr, sr = Pr[order], Nr[order], sr[order]
    q_all = np.c_[Pr @ e1, Pr @ e2]
    n2d_all = np.c_[Nr @ e1, Nr @ e2]
    nax_all = Nr @ axis
    # gold line param: point on gold where axis coordinate = s
    for s in np.arange(np.floor(s_lo - pad), np.ceil(s_hi + pad) + 1e-9, 1.0):
        lam = (s - s_a) / (s_b - s_a)
        G3 = a + lam * (b - a)
        g2 = np.array([G3 @ e1, G3 @ e2])
        i0, i1 = np.searchsorted(sr, [s - HW, s + HW])
        if i1 - i0 < 20:
            stations.append({"s": float(s), "lam": float(lam), "ok": False, "why": "few points"})
            continue
        q = q_all[i0:i1]
        nn = n2d_all[i0:i1]
        nax = nax_all[i0:i1]
        good = np.abs(nax) < 0.5
        nnl = np.linalg.norm(nn, axis=1)
        nn = nn / np.maximum(nnl, 1e-9)[:, None]
        near = np.linalg.norm(q - g2, axis=1) <= WIN
        sel = good & near
        q, nn = q[sel], nn[sel]
        rec = {"s": float(s), "lam": float(lam), "ok": False, "n_pts": int(len(q))}
        if len(q) < 20:
            rec["why"] = "few points near gold"
            stations.append(rec)
            continue
        d1 = nn @ m1
        d2 = nn @ m2
        cls = {"A+": d1 > FLANGE_TOL, "A-": d1 < -FLANGE_TOL, "B+": d2 > FLANGE_TOL, "B-": d2 < -FLANGE_TOL}
        lines = {}
        far = np.linalg.norm(q - g2, axis=1) >= 6.0
        for k, msk in cls.items():
            mm = msk & far
            if mm.sum() >= 8:
                c, t, nrm, rms = fit_line2d(q[mm])
                # orient normal like the class normal
                ref = m1 if k[0] == "A" else m2
                sgn = 1.0 if k[1] == "+" else -1.0
                if nrm @ (sgn * ref) < 0:
                    nrm = -nrm
                lines[k] = {"c": c, "t": t, "n": nrm, "rms": rms, "npts": int(mm.sum())}
        rec["skins"] = {k: v["npts"] for k, v in lines.items()}
        # corner from any available A and B skin (pick pair with most points) to orient flanges
        A_keys = [k for k in ("A+", "A-") if k in lines]
        B_keys = [k for k in ("B+", "B-") if k in lines]
        if not A_keys or not B_keys:
            rec["why"] = "missing a flange"
            stations.append(rec)
            continue
        kA = max(A_keys, key=lambda k: lines[k]["npts"])
        kB = max(B_keys, key=lambda k: lines[k]["npts"])
        X0 = intersect(lines[kA]["c"], lines[kA]["t"], lines[kB]["c"], lines[kB]["t"])
        if X0 is None:
            rec["why"] = "parallel"
            stations.append(rec)
            continue
        # flange ray directions from the corner toward the points
        mA = cls["A+"] | cls["A-"]
        mB = cls["B+"] | cls["B-"]
        ua = lines[kA]["t"] * np.sign(((q[mA & far] - X0) @ lines[kA]["t"]).mean())
        ub = lines[kB]["t"] * np.sign(((q[mB & far] - X0) @ lines[kB]["t"]).mean())
        bis = unit(ua + ub)
        ex = unit(ua - ub)  # across, + toward flange A
        # outer skin of A: the class whose normal points away from B side (n . ub < 0)
        def skin_of(key):
            ln = lines[key]
            other = ub if key[0] == "A" else ua
            return "outer" if ln["n"] @ other < 0 else "inner"
        sk = {k: skin_of(k) for k in lines}
        rec["skin_map"] = sk
        outA = [k for k in A_keys if sk[k] == "outer"]
        inA = [k for k in A_keys if sk[k] == "inner"]
        outB = [k for k in B_keys if sk[k] == "outer"]
        inB = [k for k in B_keys if sk[k] == "inner"]
        # thickness from flanges with both skins
        tks = []
        for o, i_ in ((outA, inA), (outB, inB)):
            if o and i_:
                lo_, li_ = lines[o[0]], lines[i_[0]]
                tks.append(abs((li_["c"] - lo_["c"]) @ lo_["n"]))
        tk = float(np.median(tks)) if tks else None
        rec["thickness"] = tk
        T = tk if tk else T_DEFAULT

        def skin_line(o, i_, want):
            # returns (c, t) of the wanted skin; synthesise from the other skin if missing
            if want == "outer":
                if o:
                    return lines[o[0]]["c"], lines[o[0]]["t"], "meas"
                ln = lines[i_[0]]
                return ln["c"] + _shift_outer(ln, T, ua, ub), ln["t"], "synth"
            if i_:
                return lines[i_[0]]["c"], lines[i_[0]]["t"], "meas"
            ln = lines[o[0]]
            return ln["c"] + _shift_inner(ln, T, ua, ub), ln["t"], "synth"

        def _shift_outer(ln, T, ua, ub):
            # inner skin normal points toward the concave side; outer skin is T opposite to it
            return -T * ln["n"]

        def _shift_inner(ln, T, ua, ub):
            # outer skin normal points away from the concave side; inner skin is T opposite to it
            return -T * ln["n"]

        cAo, tAo, srcAo = skin_line(outA, inA, "outer")
        cBo, tBo, srcBo = skin_line(outB, inB, "outer")
        cAi, tAi, srcAi = skin_line(outA, inA, "inner")
        cBi, tBi, srcBi = skin_line(outB, inB, "inner")
        Xo = intersect(cAo, tAo, cBo, tBo)
        Xi = intersect(cAi, tAi, cBi, tBi)
        if Xo is None or Xi is None:
            rec["why"] = "no sharp"
            stations.append(rec)
            continue
        rec["skin_src"] = {"Ao": srcAo, "Bo": srcBo, "Ai": srcAi, "Bi": srcBi}
        bend_deg = float(np.degrees(np.arccos(np.clip(-(ua @ ub), -1, 1))))
        rec["bend_deg"] = bend_deg
        # outer apex: first outer-skin point along +bis from Xo, lateral <= 0.6 (expand)
        rel = q - Xo
        h = rel @ bis
        lat = rel @ ex
        nb = nn @ bis
        apex = None
        for lw in (0.6, 1.2, 2.4):
            cand = (np.abs(lat) <= lw) & (h >= -1.0) & (h <= 12.0) & (nb < -0.2)
            if cand.sum() >= 3:
                hv = h[cand]
                floor = np.percentile(hv, 5.0)
                ns = hv[hv <= floor + 0.3]
                apex_h = float(np.median(ns))
                apex = Xo + apex_h * bis
                rec["apex_lat_window"] = lw
                rec["apex_h_from_Xo"] = apex_h
                rec["apex_n"] = int(len(ns))
                break
        # inner apex: first inner-skin point along +bis from Xi
        reli = q - Xi
        hi = reli @ bis
        lati = reli @ ex
        iapex = None
        for lw in (0.6, 1.2, 2.4):
            cand = (np.abs(lati) <= lw) & (hi >= -3.0) & (hi <= 12.0) & (nb > 0.2)
            if cand.sum() >= 3:
                hv = hi[cand]
                floor = np.percentile(hv, 5.0)
                ns = hv[hv <= floor + 0.3]
                iapex = Xi + float(np.median(ns)) * bis
                rec["iapex_lat_window"] = lw
                break
        # arc points: not within flange tolerance of either flange normal
        arcmask = (np.abs(d1) < np.cos(np.radians(15))) & (np.abs(d2) < np.cos(np.radians(15)))
        dist_X = np.linalg.norm(q - Xo, axis=1)
        outer_arc = arcmask & (nb < 0) & (dist_X < 15)
        inner_arc = arcmask & (nb > 0) & (dist_X < 15)
        co = circle_fit(q[outer_arc]) if outer_arc.sum() >= 6 else None
        ci = circle_fit(q[inner_arc]) if inner_arc.sum() >= 6 else None
        rec["n_outer_arc"] = int(outer_arc.sum())
        rec["n_inner_arc"] = int(inner_arc.sum())
        if co:
            rec["R_o"] = co[1]
            rec["C_o"] = co[0].tolist()
            rec["circ_rms_o"] = co[2]
        if ci:
            rec["R_i"] = ci[1]
            rec["C_i"] = ci[0].tolist()
        # gold relative
        rec["Xo"] = Xo.tolist()
        rec["Xi"] = Xi.tolist()
        rec["bis"] = bis.tolist()
        rec["ex"] = ex.tolist()
        rec["ua"] = ua.tolist()
        rec["ub"] = ub.tolist()
        rec["gold2d"] = g2.tolist()
        if apex is not None:
            dG = g2 - apex
            rec["apex"] = apex.tolist()
            rec["gold_minus_apex_bis"] = float(dG @ bis)
            rec["gold_minus_apex_across"] = float(dG @ ex)
            rec["d_gold_apex"] = float(np.linalg.norm(dG))
        if iapex is not None:
            rec["iapex"] = iapex.tolist()
            rec["d_gold_iapex"] = float(np.linalg.norm(g2 - iapex))
            if apex is not None:
                rec["iapex_minus_apex_bis"] = float((iapex - apex) @ bis)
        rec["d_gold_Xo"] = float(np.linalg.norm(g2 - Xo))
        rec["d_gold_Xi"] = float(np.linalg.norm(g2 - Xi))
        rec["gold_minus_Xo_bis"] = float((g2 - Xo) @ bis)
        rec["gold_minus_Xo_across"] = float((g2 - Xo) @ ex)
        # nearest scan vertex (3D) and its skin
        dn, jn = tree.query(G3)
        nrm_n = N[jn]
        n2n = unit(np.array([nrm_n @ e1, nrm_n @ e2])) if np.hypot(nrm_n @ e1, nrm_n @ e2) > 1e-6 else np.zeros(2)
        rec["d_gold_scan3d"] = float(dn)
        rec["nearest_vertex_skin"] = "outer" if n2n @ bis < -0.2 else ("inner" if n2n @ bis > 0.2 else "edge/flank")
        q_near = np.array([P[jn] @ e1, P[jn] @ e2])
        rec["nearest_vertex_across_from_Xo"] = float((q_near - Xo) @ ex)
        rec["ok"] = apex is not None
        stations.append(rec)
    return {
        "bend_id": bid,
        "gold_start": a.tolist(), "gold_end": b.tolist(), "gold_len": L,
        "axis": axis.tolist(), "angle_gold_vs_axis_deg": ang_gold_axis,
        "flange_normals": [n1.tolist(), n2.tolist()], "normal_cluster_sizes": [c1, c2],
        "e1": e1.tolist(), "e2": e2.tolist(), "s_gold": [s_a, s_b],
        "stations": stations,
    }


def main():
    d = np.load(NPZ)
    P, N = d["P"], d["N"]
    tree = cKDTree(P)
    allres = {}
    for bid in GOLD:
        res = analyse_line(bid, P, N, tree)
        (OUT / f"stations_{bid}.json").write_text(json.dumps(res, indent=1))
        allres[bid] = res
        ok = [r for r in res["stations"] if r.get("ok")]
        print(bid, "len", round(res["gold_len"], 1), "angle(gold,axis)", round(res["angle_gold_vs_axis_deg"], 2),
              "stations ok", len(ok), "/", len(res["stations"]))


if __name__ == "__main__":
    main()
