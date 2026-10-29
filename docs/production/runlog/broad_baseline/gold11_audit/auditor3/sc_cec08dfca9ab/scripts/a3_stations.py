"""Dense station analysis of every gold line of sc_cec08dfca9ab (read-only on inputs).

For each gold line, cut 1 mm slabs perpendicular to the line every STEP mm, from 30 mm before
its start to 30 mm past its end. In each section fit the corner model independently of the
earlier audit script:
  * flange A / flange B outer + inner skin lines (both skins are scanned on this part),
  * outer virtual sharp Xo, inner virtual sharp Xi, mid-surface sharp Xm,
  * outer apex = first outer-arc scan point along the concave bisector from Xo,
  * arc centre from the tangent-arc relation,
  * bend_present = both flanges carry points within 3..15 mm of the corner AND the arc band
    between them carries points (so a flange that just ends is not a bend).
Gold position = station centre (the gold line pierces every section at its origin).

Outputs stations_<bend>.json + stations_summary.json (+ apex lines fitted through the
bend-present stations, with the measured fold extent).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math

import numpy as np
from scipy.spatial import cKDTree

from a3_common import OUT, load_gold, load_points, unit

STEP = 2.0
SLAB = 0.5
EXT = 30.0

# approximate in-section flange directions per bend (section coords: u = in-plane horizontal
# axis (part Y for HB1/HB2, part X for HB3..HB6), w = part Z). a, b point from corner along flange.
SPEC = {
    "HB1": {"a": (-1, 0), "b": (0, 1), "names": ("floor", "top wall")},
    "HB2": {"a": (0, -1), "b": (-1, 0), "names": ("top wall", "top flange")},
    "HB3": {"a": (-1, 0), "b": (0, 1), "names": ("floor", "right wall")},
    "HB4": {"a": (1, 0), "b": (0, 1), "names": ("floor", "left wall")},
    "HB5": {"a": (0, -1), "b": (1, 0), "names": ("left wall", "left flange")},
    "HB6": {"a": (0, -1), "b": (-1, 0), "names": ("right wall", "right flange")},
}


def line_intersect(p1, t1, p2, t2):
    m = np.array([t1, -t2]).T
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    s, _ = np.linalg.solve(m, p2 - p1)
    return p1 + s * t1


def fit_skin_lines(q, corner, t, n, lo=4.0, hi=22.0, band=4.0):
    """Points along flange direction t (from corner) at distance lo..hi and within +-band of
    the flange line; split into skins by offset along n (n points to the concave side).
    Returns list of (offset, line point, direction, count) sorted by offset (outer first)."""
    rel = q - corner
    s = rel @ t
    o = rel @ n
    sel = (s > lo) & (s < hi) & (np.abs(o) < band)
    if sel.sum() < 15:
        return []
    pts = q[sel]
    off = o[sel]
    # refine direction by PCA of all selected points (both skins parallel)
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    tt = vt[0] if vt[0] @ t > 0 else -vt[0]
    nn = np.array([-tt[1], tt[0]])
    if nn @ n < 0:
        nn = -nn
    off = (pts - corner) @ nn
    order = np.argsort(off)
    groups, cur = [], [order[0]]
    for i, j in zip(order[:-1], order[1:]):
        if off[j] - off[i] > 0.35:
            groups.append(cur)
            cur = []
        cur.append(j)
    groups.append(cur)
    skins = []
    for g in groups:
        if len(g) < 8:
            continue
        gp = pts[g]
        cc = gp.mean(0)
        _u, _s, vt = np.linalg.svd(gp - cc, full_matrices=False)
        td = vt[0] if vt[0] @ t > 0 else -vt[0]
        nd = np.array([-td[1], td[0]])
        if nd @ n < 0:
            nd = -nd
        res = (gp - cc) @ nd
        skins.append({"off": float((cc - corner) @ nn), "p": cc, "t": td, "n": nd, "count": len(g),
                      "rms": float(np.sqrt((res ** 2).mean()))})
    skins.sort(key=lambda d: d["off"])
    return skins


def analyse_section(q, corner_guess, a, b):
    """q: 2D points (section coords, gold at origin). Returns dict."""
    a = unit(np.asarray(a, float))
    b = unit(np.asarray(b, float))
    na = unit(b - (b @ a) * a)  # normal of flange A toward the concave side
    nb = unit(a - (a @ b) * b)
    res = {"ok": False}
    # iterate twice: corner guess -> fitted outer sharp
    corner = np.asarray(corner_guess, float)
    for _ in range(3):
        sa = fit_skin_lines(q, corner, a, na)
        sb = fit_skin_lines(q, corner, b, nb)
        if not sa or not sb:
            res["why"] = f"flange_missing(A={len(sa)},B={len(sb)})"
            return res
        Xo = line_intersect(sa[0]["p"], sa[0]["t"], sb[0]["p"], sb[0]["t"])
        if Xo is None:
            res["why"] = "parallel"
            return res
        if np.linalg.norm(Xo - corner) < 0.05:
            corner = Xo
            break
        corner = Xo
    ta, tb = sa[0]["t"], sb[0]["t"]
    g = unit(ta + tb)
    gperp = np.array([-g[1], g[0]])
    included = math.degrees(math.acos(np.clip(ta @ tb, -1, 1)))
    two = len(sa) >= 2 and len(sb) >= 2
    Xi = line_intersect(sa[-1]["p"], sa[-1]["t"], sb[-1]["p"], sb[-1]["t"]) if two else None
    th_a = sa[-1]["off"] - sa[0]["off"] if len(sa) >= 2 else None
    th_b = sb[-1]["off"] - sb[0]["off"] if len(sb) >= 2 else None
    Xm = None
    if two:
        pa = 0.5 * (sa[0]["p"] + sa[-1]["p"])
        pb = 0.5 * (sb[0]["p"] + sb[-1]["p"])
        Xm = line_intersect(pa, ta, pb, tb)
    # arc / apex: points near the bisector line through Xo
    rel = q - Xo
    al = rel @ g
    ac = rel @ gperp
    near = (np.abs(ac) < 0.45) & (al > -1.5) & (al < 12)
    apex = None
    Ro = None
    C = None
    arc_pts = 0
    if near.sum() >= 2:
        vals = np.sort(al[near])
        delta = float(vals[min(1, len(vals) - 1)])
        apex = Xo + g * delta
        half = math.radians(included) / 2
        k = 1 / math.sin(half) - 1
        if k > 1e-6 and delta > 0:
            Ro = delta / k
            C = Xo + g * (Ro / math.sin(half))
        # arc band = points within Ro+0.8 of C that are not on the flat parts (between tangents)
        arc_pts = int(near.sum())
    # flange presence close to the corner (3..15 mm) on both sides
    sA = (q - Xo) @ ta
    oA = (q - Xo) @ sa[0]["n"]
    sB = (q - Xo) @ tb
    oB = (q - Xo) @ sb[0]["n"]
    nearA = int(((sA > 3) & (sA < 15) & (np.abs(oA) < 3)).sum())
    nearB = int(((sB > 3) & (sB < 15) & (np.abs(oB) < 3)).sum())
    # extent of each flange away from the corner
    extA = float(np.percentile(sA[(np.abs(oA) < 3) & (sA > 0)], 99)) if ((np.abs(oA) < 3) & (sA > 0)).sum() > 5 else 0.0
    extB = float(np.percentile(sB[(np.abs(oB) < 3) & (sB > 0)], 99)) if ((np.abs(oB) < 3) & (sB > 0)).sum() > 5 else 0.0
    bend_present = nearA >= 10 and nearB >= 10 and arc_pts >= 2 and apex is not None
    gold = np.zeros(2)
    out = {"ok": True, "Xo": Xo, "Xi": Xi, "Xm": Xm, "apex": apex, "C": C, "R_outer": Ro, "g": g,
           "included_deg": included, "bend_deg": 180 - included, "thick": [th_a, th_b], "two_skins": two,
           "nearA": nearA, "nearB": nearB, "extA": extA, "extB": extB, "arc_pts": arc_pts,
           "bend_present": bool(bend_present), "skinsA": len(sa), "skinsB": len(sb),
           "rmsA": sa[0]["rms"], "rmsB": sb[0]["rms"]}
    if apex is not None:
        v = gold - apex
        out["gold_apex_along_mm"] = float(v @ g)  # + = concave side (inside the bend)
        out["gold_apex_across_mm"] = float(v @ gperp)
        out["d_apex"] = float(np.linalg.norm(v))
    out["d_outer_sharp"] = float(np.linalg.norm(gold - Xo))
    out["gold_outer_sharp_along_mm"] = float((gold - Xo) @ g)
    if Xi is not None:
        out["d_inner_sharp"] = float(np.linalg.norm(gold - Xi))
        out["gold_inner_sharp_along_mm"] = float((gold - Xi) @ g)
    if Xm is not None:
        out["d_mid_sharp"] = float(np.linalg.norm(gold - Xm))
    if C is not None:
        out["d_arc_centre"] = float(np.linalg.norm(gold - C))
        out["gold_arc_centre_along_mm"] = float((gold - C) @ g)
    # gold distances to the flange planes (outer skins)
    out["gold_to_flangeA_outer"] = float((gold - sa[0]["p"]) @ sa[0]["n"])
    out["gold_to_flangeB_outer"] = float((gold - sb[0]["p"]) @ sb[0]["n"])
    if two:
        out["gold_to_flangeA_inner"] = float((gold - sa[-1]["p"]) @ sa[-1]["n"])
        out["gold_to_flangeB_inner"] = float((gold - sb[-1]["p"]) @ sb[-1]["n"])
    return out


def main():
    fr = json.loads((OUT / "part_frame.json").read_text())
    c0 = np.array(fr["origin"])
    M = np.array([fr["X"], fr["Y"], fr["Z"]])
    P = (load_points() - c0) @ M.T
    tree = cKDTree(P)
    G = [(b, (s - c0) @ M.T, (e - c0) @ M.T) for b, s, e in load_gold()]
    summary = {"frame": "part frame (a3_frame.py); values in mm", "step_mm": STEP, "slab_half_mm": SLAB, "lines": {}}
    for bid, s, e in G:
        d = unit(e - s)
        L = float(np.linalg.norm(e - s))
        horiz = np.array([0, 1.0, 0]) if bid in ("HB1", "HB2") else np.array([1.0, 0, 0])
        eu = unit(horiz - (horiz @ d) * d)
        ew = unit(np.cross(d, eu))
        if ew[2] < 0:
            ew = -ew
        spec = SPEC[bid]
        rows = []
        prev_corner = None
        for t in np.arange(-EXT, L + EXT + 1e-6, STEP):
            cpt = s + t * d
            idx = tree.query_ball_point(cpt, 45.0)
            if not idx:
                rows.append({"t": float(t), "ok": False, "why": "empty"})
                continue
            Q = P[idx]
            alng = (Q - cpt) @ d
            m = np.abs(alng) < SLAB
            q = np.c_[(Q[m] - cpt) @ eu, (Q[m] - cpt) @ ew]
            # corner guess: gold point shifted 5 mm to the convex side (outer), refined by fit
            a = np.array(spec["a"], float)
            b = np.array(spec["b"], float)
            g0 = unit(a + b)
            guess = prev_corner if prev_corner is not None else -5.0 * g0
            r = analyse_section(q, guess, a, b)
            if not r.get("ok") and prev_corner is not None:
                r = analyse_section(q, -5.0 * g0, a, b)
            row = {"t": float(t), "inside_gold": bool(-1e-6 <= t <= L + 1e-6), "n_slab": int(m.sum())}
            nd, _ = tree.query(cpt)
            row["d_nearest_scan"] = float(nd)
            if r.get("ok"):
                prev_corner = r["Xo"] if np.linalg.norm(r["Xo"]) < 15 else None
                for k, v in r.items():
                    if isinstance(v, np.ndarray):
                        row[k] = [float(x) for x in v]
                    else:
                        row[k] = v
                # 3D apex point
                if r.get("apex") is not None:
                    ap = r["apex"]
                    row["apex_xyz_part"] = (cpt + ap[0] * eu + ap[1] * ew).tolist()
                    xo = r["Xo"]
                    row["outer_sharp_xyz_part"] = (cpt + xo[0] * eu + xo[1] * ew).tolist()
            else:
                prev_corner = None
                row.update({"ok": False, "why": r.get("why")})
            rows.append(row)
        (OUT / f"stations_{bid}.json").write_text(json.dumps({"bend_id": bid, "length_mm": L, "rows": rows}, indent=0))
        # --- summary
        inside = [r for r in rows if r.get("inside_gold")]
        present_in = [r for r in inside if r.get("ok") and r.get("bend_present")]
        present_all = [r for r in rows if r.get("ok") and r.get("bend_present")]
        ts = np.array([r["t"] for r in present_all])
        # contiguous fold extent (allow gaps <= 12 mm, e.g. holes)
        segs = []
        if len(ts):
            st = ts[0]
            pv = ts[0]
            for x in ts[1:]:
                if x - pv > 12.0:
                    segs.append((st, pv))
                    st = x
                pv = x
            segs.append((st, pv))
        main_seg = max(segs, key=lambda z: z[1] - z[0]) if segs else None
        along = np.array([r["gold_apex_along_mm"] for r in present_in if r.get("gold_apex_along_mm") is not None])
        across = np.array([r["gold_apex_across_mm"] for r in present_in if r.get("gold_apex_across_mm") is not None])
        dap = np.array([r["d_apex"] for r in present_in if r.get("d_apex") is not None])
        summ = {"length_mm": L, "stations_inside_gold": len(inside), "bend_present_inside_gold": len(present_in),
                "frac_gold_on_fold": len(present_in) / max(1, len(inside)),
                "fold_segments_t": [[float(a), float(b)] for a, b in segs],
                "fold_main_extent_t": None if main_seg is None else [float(main_seg[0]), float(main_seg[1])]}
        if main_seg is not None:
            f0, f1 = main_seg
            fl = f1 - f0
            ov = max(0.0, min(L, f1) - max(0.0, f0))
            summ.update({"fold_length_mm": float(fl), "gold_fold_overlap_mm": float(ov),
                         "fold_coverage_by_gold": float(ov / fl) if fl > 0 else None,
                         "gold_overshoot_before_mm": float(max(0.0, f0 - 0.0)),
                         "gold_overshoot_after_mm": float(max(0.0, L - f1)),
                         "fold_beyond_gold_before_mm": float(max(0.0, -f0)),
                         "fold_beyond_gold_after_mm": float(max(0.0, f1 - L))})
        if len(along):
            summ.update({"gold_minus_apex_along_bisector_mm": {"median": float(np.median(along)), "min": float(along.min()), "max": float(along.max())},
                         "gold_minus_apex_across_mm": {"median": float(np.median(across)), "min": float(across.min()), "max": float(across.max())},
                         "d_apex_mm": {"median": float(np.median(dap)), "p10": float(np.percentile(dap, 10)), "p90": float(np.percentile(dap, 90))}})
            for key in ("d_outer_sharp", "d_inner_sharp", "d_mid_sharp", "d_arc_centre", "R_outer", "bend_deg",
                        "gold_to_flangeA_outer", "gold_to_flangeB_outer", "gold_to_flangeA_inner", "gold_to_flangeB_inner",
                        "gold_outer_sharp_along_mm", "gold_inner_sharp_along_mm", "gold_arc_centre_along_mm"):
                vals = [r[key] for r in present_in if r.get(key) is not None]
                if vals:
                    summ[key] = {"median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
            th = [x for r in present_in for x in (r.get("thick") or []) if x is not None]
            if th:
                summ["thickness_mm_median"] = float(np.median(th))
        # apex line through all bend-present stations of the main fold segment
        if main_seg is not None:
            A = np.array([r["apex_xyz_part"] for r in present_all if main_seg[0] <= r["t"] <= main_seg[1] and r.get("apex_xyz_part")])
            if len(A) >= 5:
                cA = A.mean(0)
                _u, _s, vt = np.linalg.svd(A - cA, full_matrices=False)
                dA = vt[0] if vt[0] @ d > 0 else -vt[0]
                # extent: project fold extent endpoints (t on gold) onto apex line
                tt = (A - cA) @ dA
                res = A - cA - np.outer(tt, dA)
                p_start = cA + tt.min() * dA - STEP / 2 * dA
                p_end = cA + tt.max() * dA + STEP / 2 * dA
                summ["apex_line_part"] = {"start": p_start.tolist(), "end": p_end.tolist(), "n": int(len(A)),
                                          "rms_mm": float(np.sqrt((res ** 2).sum(1).mean())),
                                          "angle_to_gold_deg": float(math.degrees(math.acos(min(1.0, abs(dA @ d)))))}
                # back to raw scan frame
                summ["apex_line_raw"] = {"start": (p_start @ M + c0).tolist(), "end": (p_end @ M + c0).tolist()}
        summary["lines"][bid] = summ
        print(bid, json.dumps({k: v for k, v in summ.items() if k not in ("apex_line_raw",)}, default=float)[:1500], flush=True)
    (OUT / "stations_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
