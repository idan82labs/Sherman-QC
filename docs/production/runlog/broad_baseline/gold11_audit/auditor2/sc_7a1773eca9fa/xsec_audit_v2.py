#!/usr/bin/env python3
"""Auditor 2 - v2: global skin planes per bend + per-station outer apex, gold offsets, fold extent.

Read-only. Per gold line:
  * true axis from the two flange-normal clusters (as v1);
  * four skin planes (flange A/B x outer/inner) fitted ONCE over the central part of the bend with
    trimmed least squares on points 5..22 mm from the gold line whose oriented normals match;
    outer/inner from the normal sign (outer-skin outward normal points away from the other flange);
  * outer virtual-sharp line X_o(s) = intersection of the outer planes; bisector b (constant);
  * per 1 mm station: outer apex = first OUTER-skin point along +b from X_o within |lateral|<=0.6 mm
    (bench_labeling 'apex' definition); inner apex likewise from X_i over inner-skin points;
  * gold point G(s) offsets from the outer apex (along b, + = concave side; across, + = toward flange A);
  * fold present at s when both flanges have points within 2..15 mm of X_o on their own plane
    (|plane distance| <= 0.8 mm, either skin) AND an outer or inner arc point exists near the bisector.
Writes v2_<id>.json and v2_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xsec_audit import GOLD, unit, true_axis  # noqa: E402

OUT = Path(__file__).resolve().parent


def fit_plane_trim(X, iters=4, keep=0.5):
    c = X.mean(0)
    for it in range(iters + 1):
        Y = X - c
        w, v = np.linalg.eigh(Y.T @ Y)
        n = v[:, 0]
        d = (X - c) @ n
        if it == iters:
            break
        thr = max(np.percentile(np.abs(d), 80), 0.15) if it < 2 else max(3 * np.median(np.abs(d)), 0.15)
        m = np.abs(d) <= thr
        if m.sum() < 30:
            break
        c = X[m].mean(0)
        X = X[m]
    rms = float(np.sqrt(np.mean(((X - c) @ n) ** 2)))
    return n, c, rms, len(X)


def fit_plane_mode(X, nref, bin_mm=0.2, band=0.45):
    """dominant plane among points sharing a normal class: mode of the offset along the class normal,
    then PCA refit on the +-band slab and a final trimmed pass (robust to parallel side surfaces such
    as a flange end face that shares the normal direction)."""
    nref = nref / np.linalg.norm(nref)
    dd = X @ nref
    lo, hi = dd.min(), dd.max()
    nb = max(1, int(np.ceil((hi - lo) / bin_mm)))
    h, e = np.histogram(dd, bins=nb, range=(lo, lo + nb * bin_mm))
    hs = h + np.r_[0, h[:-1]] + np.r_[h[1:], 0]
    k = int(np.argmax(hs))
    mode = (e[k] + e[k + 1]) / 2
    Y = X[np.abs(dd - mode) <= band]
    n, c, rms, cnt = fit_plane_trim(Y, iters=2)
    return n, c, rms, cnt


def plane_line(n1, c1, n2, c2, near):
    d = np.cross(n1, n2)
    d = d / np.linalg.norm(d)
    A = np.stack([n1, n2])
    bb = np.array([n1 @ c1, n2 @ c2])
    p0 = near + A.T @ np.linalg.solve(A @ A.T, bb - A @ near)
    return p0, d


def analyse(bid, P, N, tree, fit_lo=0.1, fit_hi=0.9, r_fit=(5.0, 22.0), pad=60.0):
    a = np.array(GOLD[bid][0], float)
    b = np.array(GOLD[bid][1], float)
    L = float(np.linalg.norm(b - a))
    ug = unit(b - a)
    axis, n1, n2, _, _ = true_axis(P, N, a, b)
    rel = P - a
    sg = rel @ ug
    wg = rel - np.outer(sg, ug)
    rg = np.linalg.norm(wg, axis=1)
    reg = (sg > fit_lo * L) & (sg < fit_hi * L) & (rg > r_fit[0]) & (rg < r_fit[1])
    Pr, Nr = P[reg], N[reg]
    planes = {}
    for name, nref in (("A", n1), ("B", n2)):
        for sgn in (+1, -1):
            m = (Nr @ (sgn * nref)) > np.cos(np.radians(10))
            if m.sum() < 60:
                continue
            n, c, rms, k = fit_plane_mode(Pr[m], sgn * nref)
            if n @ (sgn * nref) < 0:
                n = -n
            planes[f"{name}{'+' if sgn > 0 else '-'}"] = {"n": n, "c": c, "rms": rms, "npts": int(k)}
    out = {"bend_id": bid, "gold_len": L, "axis": axis.tolist(),
           "angle_gold_axis_deg": float(np.degrees(np.arccos(min(1, abs(ug @ axis))))),
           "planes": {k: {"n": v["n"].tolist(), "c": v["c"].tolist(), "rms": v["rms"], "npts": v["npts"]}
                      for k, v in planes.items()}}
    # flange side directions (perp to axis, in plane, pointing to flange points)
    mid = (a + b) / 2

    def side_dir(key):
        pl = planes[key]
        nrm = pl["n"]
        t = np.cross(axis, nrm)
        t = t / np.linalg.norm(t)
        # flange points of this plane: direction from the gold line to their centroid
        if (pl["c"] - mid) @ t < 0:
            t = -t
        return t
    keysA = [k for k in planes if k[0] == "A"]
    keysB = [k for k in planes if k[0] == "B"]
    if not keysA or not keysB:
        out["error"] = f"planes found: {list(planes)}"
        return out
    uA = side_dir(max(keysA, key=lambda k: planes[k]["npts"]))
    uB = side_dir(max(keysB, key=lambda k: planes[k]["npts"]))
    bis = unit(uA + uB)
    ex = unit(uA - uB)
    ex = unit(ex - (ex @ axis) * axis)
    bis = unit(bis - (bis @ axis) * axis)
    included = float(np.degrees(np.arccos(np.clip(uA @ uB, -1, 1))))
    out["bend_deg"] = 180.0 - included

    def skin(key):
        other = uB if key[0] == "A" else uA
        return "outer" if planes[key]["n"] @ other < 0 else "inner"
    skins = {k: skin(k) for k in planes}
    out["skins"] = skins
    th = []
    for f in ("A", "B"):
        ko = [k for k in planes if k[0] == f and skins[k] == "outer"]
        ki = [k for k in planes if k[0] == f and skins[k] == "inner"]
        if ko and ki:
            po, pi = planes[ko[0]], planes[ki[0]]
            th.append(abs((pi["c"] - po["c"]) @ po["n"]))
    T = float(np.median(th)) if th else 2.0
    out["thickness"] = T
    out["thickness_per_flange"] = th

    def skin_plane(f, want):
        k = [k for k in planes if k[0] == f and skins[k] == want]
        if k:
            return planes[k[0]]["n"], planes[k[0]]["c"], "meas"
        k2 = [k for k in planes if k[0] == f][0]
        n, c = planes[k2]["n"], planes[k2]["c"]
        return n, c - T * n, "synth"   # other skin lies T behind the outward normal
    nAo, cAo, sAo = skin_plane("A", "outer")
    nBo, cBo, sBo = skin_plane("B", "outer")
    nAi, cAi, sAi = skin_plane("A", "inner")
    nBi, cBi, sBi = skin_plane("B", "inner")
    out["skin_src"] = {"Ao": sAo, "Bo": sBo, "Ai": sAi, "Bi": sBi}
    Xo0, do = plane_line(nAo, cAo, nBo, cBo, mid)
    Xi0, di = plane_line(nAi, cAi, nBi, cBi, mid)
    if do @ axis < 0:
        do = -do
    if di @ axis < 0:
        di = -di
    out["sharp_dir_vs_axis_deg"] = float(np.degrees(np.arccos(min(1, abs(do @ axis)))))
    # parametrise everything by s along do (outer sharp line)
    sP = (P - Xo0) @ do
    order = np.argsort(sP)
    sP_sorted = sP[order]
    s_a, s_b = float((a - Xo0) @ do), float((b - Xo0) @ do)
    s_lo, s_hi = min(s_a, s_b), max(s_a, s_b)
    stations = []
    for s in np.arange(np.floor(s_lo - pad), np.ceil(s_hi + pad) + 1e-9, 1.0):
        i0, i1 = np.searchsorted(sP_sorted, [s - 0.5, s + 0.5])
        idx = order[i0:i1]
        Xo = Xo0 + s * do
        # inner sharp at the same station: point on Xi line closest to the slab plane
        Xi = Xi0 + ((Xo - Xi0) @ di) * di
        lam = (s - s_a) / (s_b - s_a)
        G = a + lam * (b - a)
        rec = {"s": float(s), "lam": float(lam)}
        if len(idx) < 5:
            rec.update(fold=False, n=0)
            stations.append(rec)
            continue
        Q = P[idx]
        Nq = N[idx]
        w = Q - Xo
        w = w - np.outer(w @ do, do)
        h = w @ bis
        lat = w @ ex
        nb = Nq @ bis
        # flange presence near the corner
        dA = np.abs(np.minimum(np.abs((Q - cAo) @ nAo), np.abs((Q - cAi) @ nAi)))
        dB = np.abs(np.minimum(np.abs((Q - cBo) @ nBo), np.abs((Q - cBi) @ nBi)))
        alongA = w @ uA
        alongB = w @ uB
        rad = np.linalg.norm(w, axis=1)
        nA = int(((dA <= 0.8) & (alongA > 2.0) & (rad <= 15.0) & (np.abs(Nq @ nAo) > 0.9)).sum())
        nB = int(((dB <= 0.8) & (alongB > 2.0) & (rad <= 15.0) & (np.abs(Nq @ nBo) > 0.9)).sum())
        rec["nA"], rec["nB"] = nA, nB
        apex_h = None
        for lw in (0.6, 1.2):
            cand = (np.abs(lat) <= lw) & (h >= -1.5) & (h <= 3.0 * T + 6.0) & (nb < -0.3)
            if cand.sum() >= 2:
                hv = h[cand]
                fl = np.percentile(hv, 5)
                apex_h = float(np.median(hv[hv <= fl + 0.3]))
                rec["apex_lw"] = lw
                break
        iapex_h = None
        wi = Q - Xi
        wi = wi - np.outer(wi @ do, do)
        hi_ = wi @ bis
        lati = wi @ ex
        for lw in (0.6, 1.2):
            cand = (np.abs(lati) <= lw) & (hi_ >= -3.0) & (hi_ <= 2.0 * T + 6.0) & (nb > 0.3)
            if cand.sum() >= 2:
                hv = hi_[cand]
                fl = np.percentile(hv, 5)
                iapex_h = float(np.median(hv[hv <= fl + 0.3]))
                break
        rec["fold"] = bool(nA >= 3 and nB >= 3 and (apex_h is not None or iapex_h is not None))
        gw = G - Xo
        gw = gw - (gw @ do) * do
        rec["gold_h_from_Xo"] = float(gw @ bis)
        rec["gold_lat_from_Xo"] = float(gw @ ex)
        xi_w = Xi - Xo
        rec["Xi_h_from_Xo"] = float(xi_w @ bis)
        rec["Xi_lat_from_Xo"] = float(xi_w @ ex)
        if apex_h is not None:
            rec["apex_h"] = apex_h
            rec["gold_minus_apex_bis"] = rec["gold_h_from_Xo"] - apex_h
            rec["gold_minus_apex_across"] = rec["gold_lat_from_Xo"]
            rec["d_gold_apex"] = float(np.hypot(rec["gold_minus_apex_bis"], rec["gold_minus_apex_across"]))
        if iapex_h is not None:
            ia = Xi + iapex_h * bis
            iw = ia - Xo
            iw = iw - (iw @ do) * do
            rec["iapex_h_from_Xo"] = float(iw @ bis)
            rec["iapex_lat_from_Xo"] = float(iw @ ex)
            rec["d_gold_iapex"] = float(np.hypot(rec["gold_h_from_Xo"] - rec["iapex_h_from_Xo"],
                                                 rec["gold_lat_from_Xo"] - rec["iapex_lat_from_Xo"]))
        dn, jn = tree.query(G)
        rec["d_gold_scan"] = float(dn)
        nn = N[jn]
        rec["nearest_skin"] = "outer" if nn @ bis < -0.3 else ("inner" if nn @ bis > 0.3 else "flank/edge")
        stations.append(rec)
    out["stations"] = stations
    out["s_gold"] = [s_a, s_b]
    out["Xo0"] = Xo0.tolist()
    out["do"] = do.tolist()
    out["Xi0"] = Xi0.tolist()
    out["di"] = di.tolist()
    out["bis"] = bis.tolist()
    out["ex"] = ex.tolist()
    out["uA"] = uA.tolist()
    out["uB"] = uB.tolist()
    return out


def summarise(res):
    st = res["stations"]
    s_a, s_b = res["s_gold"]
    s_lo, s_hi = min(s_a, s_b), max(s_a, s_b)
    fold = [x for x in st if x.get("fold")]
    summ = {"bend_id": res["bend_id"], "gold_len": res["gold_len"], "bend_deg": res.get("bend_deg"),
            "thickness": res.get("thickness"), "skin_src": res.get("skin_src"),
            "angle_gold_axis_deg": res["angle_gold_axis_deg"]}
    if not fold:
        summ["fold"] = None
        return summ
    fs = np.array([x["s"] for x in fold])
    # contiguous runs (gap tolerance 6 mm) to report fold segments
    runs = []
    start = prev = fs[0]
    for v in fs[1:]:
        if v - prev > 6.0:
            runs.append([float(start), float(prev)])
            start = v
        prev = v
    runs.append([float(start), float(prev)])
    summ["fold_runs_s"] = runs
    # the run(s) that overlap the gold span
    ov = [r for r in runs if r[1] >= s_lo - 5 and r[0] <= s_hi + 5]
    if ov:
        f0, f1 = min(r[0] for r in ov), max(r[1] for r in ov)
        f0 -= 0.5
        f1 += 0.5
        summ["fold_extent_s"] = [f0, f1]
        summ["fold_len"] = f1 - f0
        inter = max(0.0, min(f1, s_hi) - max(f0, s_lo))
        summ["gold_covers_fold_frac"] = inter / (f1 - f0)
        summ["gold_on_fold_frac"] = inter / (s_hi - s_lo)
        summ["gold_overhang_mm"] = [max(0.0, f0 - s_lo), max(0.0, s_hi - f1)]
        summ["fold_missing_from_gold_mm"] = [max(0.0, s_lo - f0), max(0.0, f1 - s_hi)]
        inside = [x for x in st if s_lo <= x["s"] <= s_hi and x.get("fold") and "d_gold_apex" in x]
        if inside:
            arr = lambda k: np.array([x[k] for x in inside if k in x])
            summ["n_stations_measured"] = len(inside)
            for k in ("d_gold_apex", "gold_minus_apex_bis", "gold_minus_apex_across", "d_gold_scan", "apex_h",
                      "d_gold_iapex"):
                v = arr(k)
                if len(v):
                    summ[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                               "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
            sk = [x["nearest_skin"] for x in inside]
            summ["nearest_skin_counts"] = {k: sk.count(k) for k in set(sk)}
    return summ


def main():
    d = np.load(sys.argv[1])
    P, N = d["P"], d["N"]
    tree = cKDTree(P)
    summary = {}
    for bid in GOLD:
        res = analyse(bid, P, N, tree)
        (OUT / f"v2_{bid}.json").write_text(json.dumps(res, indent=1))
        summary[bid] = summarise(res) if "stations" in res else res
        print(json.dumps(summary[bid], indent=None, default=float)[:1500])
    (OUT / "v2_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
