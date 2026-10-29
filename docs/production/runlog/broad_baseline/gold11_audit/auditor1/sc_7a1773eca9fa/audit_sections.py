#!/usr/bin/env python3
"""Auditor-1 independent section analysis for Gold11 scan sc_7a1773eca9fa (49001000).

Read-only on all inputs. Writes only next to this script.

For every gold line:
  * fit both skins of the two plates the line separates (3D planes, local to the line);
  * true bend axis d = nP x nQ; outer / inner / mid virtual sharps from the skin planes;
  * 1 mm stations along the true axis, from 25 mm before to 25 mm after the gold span:
      - outer apex (first scan surface on the bisector coming from the convex side),
        inner apex (first surface coming from the concave side), radii, tangent points,
      - fold presence (flange material + bend arc present at the station),
      - gold point in the section: offset from the outer apex split along the bisector
        (+ = toward the concave/inner side) and sideways (+ = toward the base plate P),
        distances to inner apex, virtual sharps, arc centre, tangent points, nearest scan.
  * fold extent, coverage of the fold by the gold span, overshoot, matcher metrics of the
    gold against the reconstructed outer-apex line over the fold extent.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SCAN = Path("/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply")
LABELS = Path("/Users/idantal/ShermanQC/FullProject/output/bend_label_10_49001000/bend_instance_labels.json")
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else None
VERBOSE = False

# seed plate normals (raw frame) from the area-weighted face-normal clustering
SEED = {
    "base": [-0.3853, 0.2373, -0.8918],
    "LF": [0.6801, -0.5866, -0.4397],   # long front flange (PCA y=-57, full length)
    "LTW": [-0.8069, -0.5583, 0.1930],  # left tall wall (PCA x=-190)
    "RTW": [-0.2293, -0.9411, 0.2484],  # right tall wall
    "RT": [0.3916, 0.9170, 0.0763],     # right small tab
    "LT": [-0.5827, -0.5960, 0.5525],   # left small tab
}
PAIR = {"M01": ("base", "LF"), "M02": ("base", "LT"), "M03": ("base", "RT"),
        "M04": ("base", "LTW"), "M05": ("base", "RTW")}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def load():
    if CACHE and CACHE.is_file():
        z = np.load(CACHE)
        return z["V"], z["N"]
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(SCAN))
    return np.asarray(m.vertices), np.asarray(m.vertex_normals)


def fit_plane(P, iters=3, keep=2.5, seed=None):
    m = np.ones(len(P), bool)
    if seed is not None:
        # mode-seeded start: the densest 0.25 mm offset bin along the seed normal is the skin;
        # this rejects e.g. a flange's free-edge face that shares the plate's normal
        h = P @ unit(seed)
        bins = np.arange(h.min() - 0.25, h.max() + 0.5, 0.25)
        hist, edges = np.histogram(h, bins=bins)
        k = int(np.argmax(hist))
        mode = 0.5 * (edges[k] + edges[k + 1])
        m = np.abs(h - mode) < 0.8
    for _ in range(iters):
        c = P[m].mean(0)
        _u, _s, vt = np.linalg.svd(P[m] - c, full_matrices=False)
        n = vt[2]
        r = (P - c) @ n
        sig = max(1.4826 * np.median(np.abs(r[m] - np.median(r[m]))), 0.05)
        m = np.abs(r - np.median(r[m])) < keep * sig + 0.05
    c = P[m].mean(0)
    _u, _s, vt = np.linalg.svd(P[m] - c, full_matrices=False)
    n = vt[2]
    r = (P[m] - c) @ n
    return n, c, float(np.sqrt(np.mean(r ** 2))), int(m.sum())


def plane_plane_line(n1, c1, n2, c2):
    d = unit(np.cross(n1, n2))
    A = np.array([n1, n2, d])
    b = np.array([n1 @ c1, n2 @ c2, 0.0])
    p = np.linalg.solve(A, b)
    return p, d


def analyse(bid, s, e, V, N, tree):
    pname, qname = PAIR[bid]
    dg = unit(e - s)
    L = float(np.linalg.norm(e - s))
    mid = 0.5 * (s + e)
    idx = np.array(tree.query_ball_point(mid, 0.5 * L + 60.0))
    P = V[idx]
    Nn = N[idx]
    t = (P - s) @ dg
    perp = np.linalg.norm((P - s) - np.outer(t, dg), axis=1)
    out = {"bend_id": bid, "plates": [pname, qname], "gold_len_mm": L}
    skins = {}
    for name in (pname, qname):
        n0 = unit(SEED[name])
        cosang = Nn @ n0
        flat = np.abs(cosang) > math.cos(math.radians(10))
        span_ok = (t > -2) & (t < L + 2)
        sel = flat & span_ok & (perp > 8.0) & (perp < (30.0 if name == "base" else 45.0))
        two = []
        for sgn in (1, -1):
            m = sel & (np.sign(cosang) == sgn)
            if m.sum() < 50:
                continue
            n, c, rms, cnt = fit_plane(P[m], seed=n0)
            r_all = np.abs((P[m] - c) @ n)
            pts = P[m][r_all < 0.6]
            two.append({"n": n, "c": c, "rms": rms, "count": cnt, "sign": sgn, "pts": pts})
            if VERBOSE:
                print(f"   [{bid}] {name} skin sign {sgn:+d}: pts {int(m.sum())} inliers {cnt} rms {rms:.3f} n {np.round(n,4)} offset_along_seed {float(c @ n0):.3f}")
        skins[name] = two
    # orientation helpers
    res = {}
    for name in (pname, qname):
        two = skins[name]
        if not two:
            raise RuntimeError(f"{bid}: no skins for {name}")
        nref = unit(np.mean([sk["n"] * np.sign(sk["n"] @ np.asarray(SEED[name])) for sk in two], axis=0))
        res[name] = nref
    nP, nQ = res[pname], res[qname]
    d = unit(np.cross(nP, nQ))
    if d @ dg < 0:
        d = -d
    # in-plane directions from the bend into each plate (toward the plate's material)
    def into(name, nn):
        a = unit(np.cross(d, nn))
        two = skins[name]
        c = np.mean([sk["c"] for sk in two], axis=0)
        rel = c - s
        rel = rel - d * (rel @ d)
        return a if a @ rel > 0 else -a

    aP, aQ = into(pname, nP), into(qname, nQ)
    included = math.degrees(math.acos(np.clip(aP @ aQ, -1, 1)))
    bend_angle = 180.0 - included
    g = unit(aP + aQ)  # concave-side bisector
    # orient normals toward the concave side and identify outer/inner skins
    def outer_inner(name, nn, a_other):
        n_in = nn if nn @ a_other > 0 else -nn
        two = skins[name]
        offs = sorted([(float(sk["c"] @ n_in), sk) for sk in two], key=lambda x: x[0])
        outer = offs[0][1]
        inner = offs[-1][1] if len(offs) > 1 else None
        thick = (offs[-1][0] - offs[0][0]) if len(offs) > 1 else None
        return n_in, outer, inner, thick

    nPi, Pout, Pin, tP = outer_inner(pname, nP, aQ)
    nQi, Qout, Qin, tQ = outer_inner(qname, nQ, aP)
    # project skin planes onto a common normal sign
    def plane(sk, n_in):
        return n_in, sk["c"]

    OVS_p, OVS_d = plane_plane_line(nPi, Pout["c"], nQi, Qout["c"])
    IVS = None
    if Pin is not None and Qin is not None:
        IVS_p, _ = plane_plane_line(nPi, Pin["c"], nQi, Qin["c"])
        IVS = IVS_p
    thick = np.nanmean([x for x in (tP, tQ) if x is not None]) if (tP or tQ) else None
    out.update({
        "true_axis_dir": d.tolist(), "gold_vs_true_axis_deg": math.degrees(math.acos(min(1, abs(dg @ d)))),
        "bend_angle_deg": bend_angle, "included_deg": included,
        "thickness_P_mm": tP, "thickness_Q_mm": tQ,
        "skin_fit": {k: [{"rms": sk["rms"], "count": sk["count"]} for sk in v] for k, v in skins.items()},
    })
    h = unit(np.cross(d, g))
    if h @ aP < 0:
        h = -h
    half = math.radians(included) / 2.0
    # station loop along the true axis, parameterised by projection of the gold endpoints
    s0 = float((s - OVS_p) @ d)
    s1 = float((e - OVS_p) @ d)
    lo, hi = min(s0, s1), max(s0, s1)
    stations = []
    for u in np.arange(lo - 25.0, hi + 25.0 + 1e-6, 1.0):
        X0 = OVS_p + u * d
        ii = np.array(tree.query_ball_point(X0, 32.0))
        if len(ii) == 0:
            stations.append({"u": float(u), "present": False, "why": "no_points"})
            continue
        Q = V[ii]
        NQ = N[ii]
        al = (Q - X0) @ d
        m = np.abs(al) < 0.6
        Q, NQ = Q[m], NQ[m]
        # local re-fit of the four skin planes within +-15 mm along the axis (plates can warp)
        loc = {}
        for key, sk, nn in (("Pout", Pout, nPi), ("Qout", Qout, nQi), ("Pin", Pin, nPi), ("Qin", Qin, nQi)):
            if sk is None:
                loc[key] = None
                continue
            pts = sk["pts"]
            w = np.abs((pts - X0) @ d) < 15.0
            if w.sum() >= 60:
                n_l, c_l, _r, _c = fit_plane(pts[w], seed=nn)
                n_l = n_l if n_l @ nn > 0 else -n_l
                loc[key] = (n_l, c_l)
            else:
                loc[key] = (nn, sk["c"])
        A = np.array([loc["Pout"][0], loc["Qout"][0], d])
        b = np.array([loc["Pout"][0] @ loc["Pout"][1], loc["Qout"][0] @ loc["Qout"][1], d @ X0])
        X0 = np.linalg.solve(A, b)   # local outer virtual sharp in this station plane
        IVS_loc = None
        if loc["Pin"] is not None and loc["Qin"] is not None:
            A = np.array([loc["Pin"][0], loc["Qin"][0], d])
            b = np.array([loc["Pin"][0] @ loc["Pin"][1], loc["Qin"][0] @ loc["Qin"][1], d @ X0])
            IVS_loc = np.linalg.solve(A, b)
        x = (Q - X0) @ g   # + toward concave
        y = (Q - X0) @ h   # + toward plate P
        st = {"u": float(u), "n_slab": int(len(Q))}
        # fold presence: flange-Q flat material 3..20 mm from the OVS along aQ, and plate P likewise
        def flat_present(nn, a, skin_out, skin_in, lo_mm=3.0, hi_mm=20.0):
            cosv = np.abs(NQ @ nn) > math.cos(math.radians(12))
            along = (Q - X0) @ a
            dist_out = np.abs((Q - skin_out["c"]) @ nn)
            near = dist_out < 0.8
            if skin_in is not None:
                near |= np.abs((Q - skin_in["c"]) @ nn) < 0.8
            mm = cosv & near & (along > lo_mm) & (along < hi_mm)
            return int(mm.sum()), (float(along[mm].max()) if mm.any() else 0.0)

        nqf, qreach = flat_present(nQi, aQ, Qout, Qin)
        npf, preach = flat_present(nPi, aP, Pout, Pin)
        st["q_flat_pts"], st["q_reach_mm"] = nqf, qreach
        st["p_flat_pts"], st["p_reach_mm"] = npf, preach
        # bisector hits
        bis = (np.abs(y) < 0.45) & (x > -2.0) & (x < 14.0)
        xb = np.sort(x[bis])
        st["bis_pts"] = int(bis.sum())
        apex_o = apex_i = None
        if len(xb) >= 2:
            # outer skin: lowest cluster; inner skin: highest cluster (if separated by > 0.6 mm)
            apex_o = float(np.median(xb[xb < xb[0] + 0.5]))
            top = xb[-1]
            if top - xb[0] > 0.8:
                apex_i = float(np.median(xb[xb > top - 0.5]))
        st["apex_o_x"] = apex_o
        st["apex_i_x"] = apex_i
        if apex_o is not None and 1e-3 < half < math.radians(89):
            k = 1 / math.sin(half) - 1
            st["R_o"] = apex_o / k if apex_o > 0 else 0.0
        if IVS_loc is not None and apex_i is not None:
            ivs_x = float((IVS_loc - X0) @ g)
            st["ivs_x"] = ivs_x
            k = 1 / math.sin(half) - 1
            st["R_i"] = (apex_i - ivs_x) / k if k > 1e-6 else None
        # arc presence: points with normals between the two plates, near the corner
        cP = np.abs(NQ @ nPi)
        cQ = np.abs(NQ @ nQi)
        arcm = (cP < math.cos(math.radians(15))) & (cQ < math.cos(math.radians(15))) & (np.hypot(x, y) < 15)
        st["arc_pts"] = int(arcm.sum())
        # direct circle fits (Kasa) to the two arcs: split arc-normal points by distance to a provisional centre
        if apex_o is not None and arcm.sum() >= 12:
            k = 1 / math.sin(half) - 1
            Ro0 = max(apex_o / k, 0.5) if k > 1e-6 else 5.0
            cx = apex_o + Ro0
            qa = np.c_[x[arcm], y[arcm]]
            rr = np.hypot(qa[:, 0] - cx, qa[:, 1])
            fits = {}
            thick_here = (tP or 2.0)
            for nm, msk in (("outer", rr > Ro0 - 0.5 * thick_here), ("inner", rr <= Ro0 - 0.5 * thick_here)):
                if msk.sum() < 6:
                    continue
                A_ = np.c_[2 * qa[msk], np.ones(msk.sum())]
                b_ = (qa[msk] ** 2).sum(1)
                sol, *_ = np.linalg.lstsq(A_, b_, rcond=None)
                ccx, ccy = sol[0], sol[1]
                R = math.sqrt(max(sol[2] + ccx ** 2 + ccy ** 2, 0.0))
                res_ = np.abs(np.hypot(qa[msk, 0] - ccx, qa[msk, 1] - ccy) - R)
                fits[nm] = {"cx": float(ccx), "cy": float(ccy), "R": float(R), "n": int(msk.sum()),
                            "rms": float(np.sqrt(np.mean(res_ ** 2)))}
            st["circle_fit"] = fits
        st["loc_planes"] = {kk: (None if loc[kk] is None else [loc[kk][0].tolist(), loc[kk][1].tolist()]) for kk in loc}
        st["ivs_xyz"] = None if IVS_loc is None else IVS_loc.tolist()
        st["ovs_local_xyz"] = X0.tolist()
        st["present"] = bool(nqf >= 6 and npf >= 6 and (st["arc_pts"] >= 4 or st["bis_pts"] >= 2))
        # gold point in this section (intersection of the gold line with the station plane)
        den = dg @ d
        tg = ((X0 - s) @ d) / den
        st["gold_t"] = float(tg)
        if -1e-6 <= tg <= L + 1e-6:
            G = s + tg * dg
            gx, gy = float((G - X0) @ g), float((G - X0) @ h)
            st["gold_xy"] = [gx, gy]
            dn, _ = tree.query(G)
            st["gold_nearest_scan_mm"] = float(dn)
            if apex_o is not None:
                st["gold_minus_apex_o"] = [gx - apex_o, gy]
                st["d_apex_o"] = float(math.hypot(gx - apex_o, gy))
            if apex_i is not None:
                st["d_apex_i"] = float(math.hypot(gx - apex_i, gy))
            st["d_ovs"] = float(math.hypot(gx, gy))
            if IVS_loc is not None:
                ivx = float((IVS_loc - X0) @ g)
                ivy = float((IVS_loc - X0) @ h)
                st["d_ivs"] = float(math.hypot(gx - ivx, gy - ivy))
            # signed distances to the four skin planes
            st["gold_to_skin"] = {
                k: (float((G - loc[kk][1]) @ loc[kk][0]) if loc[kk] is not None else None)
                for k, kk in (("P_outer", "Pout"), ("Q_outer", "Qout"), ("P_inner", "Pin"), ("Q_inner", "Qin"))
            }
            st["ovs_xyz"] = X0.tolist()
            # position along P and Q relative to the outer virtual sharp (in-plane distance)
            st["gold_along_aP"] = float((G - X0) @ aP)
            st["gold_along_aQ"] = float((G - X0) @ aQ)
        stations.append(st)
    for name in skins:
        for sk in skins[name]:
            sk.pop("pts", None)
    out["stations"] = stations
    out["frame"] = {"OVS_p": OVS_p.tolist(), "d": d.tolist(), "g": g.tolist(), "h": h.tolist(),
                    "aP": aP.tolist(), "aQ": aQ.tolist(), "nP_in": nPi.tolist(), "nQ_in": nQi.tolist(),
                    "IVS_p": None if IVS is None else IVS.tolist(), "half_rad": half,
                    "Pout_c": Pout["c"].tolist(), "Qout_c": Qout["c"].tolist(),
                    "Pin_c": None if Pin is None else Pin["c"].tolist(),
                    "Qin_c": None if Qin is None else Qin["c"].tolist(),
                    "gold_u": [s0, s1]}
    return out


def summarise(r):
    st = r["stations"]
    pres = np.array([x.get("present", False) for x in st])
    us = np.array([x["u"] for x in st])
    # longest run allowing gaps up to 4 mm (holes / reliefs inside one fold)
    runs, cur = [], None
    gap = 0
    for u, p in zip(us, pres):
        if p:
            if cur is None:
                cur = [u, u]
            else:
                cur[1] = u
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > 4:
                runs.append(cur)
                cur, gap = None, 0
    if cur is not None:
        runs.append(cur)
    g0, g1 = sorted(r["frame"]["gold_u"])
    best = max(runs, key=lambda ab: (min(ab[1], g1) - max(ab[0], g0), ab[1] - ab[0])) if runs else None
    summ = {"fold_runs_u": runs}
    if best:
        f0, f1 = best[0] - 0.5, best[1] + 0.5
        ov = max(0.0, min(f1, g1) - max(f0, g0))
        summ.update({"fold_u": [f0, f1], "fold_len_mm": f1 - f0, "gold_u": [g0, g1],
                     "coverage_pct_of_fold": 100.0 * ov / (f1 - f0),
                     "gold_inside_fold_pct": 100.0 * ov / (g1 - g0),
                     "gold_start_minus_fold_start_mm": g0 - f0, "gold_end_minus_fold_end_mm": g1 - f1})
    inside = [x for x in st if "gold_xy" in x and x.get("present")]
    def stat(key, idx=None):
        v = [x[key] if idx is None else x[key][idx] for x in inside if x.get(key) is not None]
        if not v:
            return None
        v = np.asarray(v, float)
        return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    summ["d_apex_o"] = stat("d_apex_o")
    summ["gold_minus_apex_o_along_bisector(+concave)"] = stat("gold_minus_apex_o", 0)
    summ["gold_minus_apex_o_sideways(+toward_P)"] = stat("gold_minus_apex_o", 1)
    summ["d_apex_i"] = stat("d_apex_i")
    summ["d_ovs"] = stat("d_ovs")
    summ["d_ivs"] = stat("d_ivs")
    summ["R_o"] = stat("R_o")
    summ["R_i"] = stat("R_i")
    summ["apex_o_x"] = stat("apex_o_x")
    summ["apex_i_x"] = stat("apex_i_x")
    summ["gold_nearest_scan_mm"] = stat("gold_nearest_scan_mm")
    for nm in ("outer", "inner"):
        v = [x["circle_fit"][nm]["R"] for x in st if x.get("present") and nm in (x.get("circle_fit") or {})
             and x["circle_fit"][nm]["rms"] < 0.25]
        summ[f"circlefit_R_{nm}"] = ({"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                      "p90": float(np.percentile(v, 90)), "min": float(np.min(v)), "max": float(np.max(v)),
                                      "n": len(v)} if v else None)
    summ["gold_along_aP"] = stat("gold_along_aP")
    summ["gold_along_aQ"] = stat("gold_along_aQ")
    for k in ("P_outer", "Q_outer", "P_inner", "Q_inner"):
        v = [x["gold_to_skin"][k] for x in inside if x.get("gold_to_skin", {}).get(k) is not None]
        summ[f"gold_to_{k}_plane_mm"] = float(np.median(v)) if v else None
    return summ


def apex_line_metrics(r, V, tree):
    """Reconstruct the outer-apex line over the fold extent and score the gold against it with the
    frozen pair_metrics formula (transverse midpoint offset, angle, overlaps)."""
    sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
    from scripts.evaluate_f1_against_manual_gold import pair_metrics

    f = r["frame"]
    OVS = np.asarray(f["OVS_p"]); d = np.asarray(f["d"]); g = np.asarray(f["g"])
    sm = r["summary"]
    if "fold_u" not in sm:
        return None
    f0, f1 = sm["fold_u"]
    ax = [x["apex_o_x"] for x in r["stations"] if x.get("present") and x.get("apex_o_x") is not None
          and f0 <= x["u"] <= f1]
    ax_i = [x["apex_i_x"] for x in r["stations"] if x.get("present") and x.get("apex_i_x") is not None
            and f0 <= x["u"] <= f1]
    a = float(np.median(ax)) if ax else 0.0
    apex_line = np.array([OVS + f0 * d + a * g, OVS + f1 * d + a * g])
    gold = np.array(r["gold_endpoints"])
    res = {"apex_offset_from_ovs_mm": a, "apex_line": apex_line.tolist(),
           "gold_vs_outer_apex_line": pair_metrics(apex_line, gold)}
    if ax_i:
        ai = float(np.median(ax_i))
        inner_line = np.array([OVS + f0 * d + ai * g, OVS + f1 * d + ai * g])
        res["inner_apex_offset_from_ovs_mm"] = ai
        res["gold_vs_inner_apex_line"] = pair_metrics(inner_line, gold)
    return res


def main():
    V, N = load()
    tree = cKDTree(V)
    labels = json.loads(LABELS.read_text())
    report = {"scan": str(SCAN), "labels": str(LABELS), "lines": []}
    for b in labels["manual_bends"]:
        s = np.asarray(b["line_start"], float)
        e = np.asarray(b["line_end"], float)
        r = analyse(b["bend_id"], s, e, V, N, tree)
        r["gold_endpoints"] = [s.tolist(), e.tolist()]
        r["summary"] = summarise(r)
        r["apex_line_check"] = apex_line_metrics(r, V, tree)
        report["lines"].append(r)
        sm = r["summary"]
        print(f"== {b['bend_id']} plates {r['plates']} bend {r['bend_angle_deg']:.1f} deg  "
              f"tP {r['thickness_P_mm']} tQ {r['thickness_Q_mm']}  gold vs true axis {r['gold_vs_true_axis_deg']:.2f} deg")
        for k in ("fold_u", "fold_len_mm", "gold_u", "coverage_pct_of_fold", "gold_inside_fold_pct",
                  "gold_start_minus_fold_start_mm", "gold_end_minus_fold_end_mm", "fold_runs_u"):
            print("   ", k, json.dumps(sm.get(k), default=float)[:200])
        for k in ("d_apex_o", "gold_minus_apex_o_along_bisector(+concave)", "gold_minus_apex_o_sideways(+toward_P)",
                  "d_apex_i", "d_ovs", "d_ivs", "R_o", "R_i", "circlefit_R_outer", "circlefit_R_inner", "apex_o_x", "apex_i_x", "gold_nearest_scan_mm",
                  "gold_along_aP", "gold_along_aQ"):
            v = sm.get(k)
            if v:
                print(f"    {k:48s} med {v['median']:7.2f}  p10 {v['p10']:7.2f}  p90 {v['p90']:7.2f}  n {v['n']}")
        for k in ("P_outer", "Q_outer", "P_inner", "Q_inner"):
            print(f"    gold_to_{k}_plane_mm", sm.get(f"gold_to_{k}_plane_mm"))
        al = r["apex_line_check"]
        if al:
            print("    matcher vs OUTER apex line:", {k: round(v, 3) for k, v in al["gold_vs_outer_apex_line"].items()})
            if "gold_vs_inner_apex_line" in al:
                print("    matcher vs INNER apex line:", {k: round(v, 3) for k, v in al["gold_vs_inner_apex_line"].items()})
    (HERE / "audit_sections.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
