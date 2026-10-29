"""Auditor-3 independent cross-section locus audit for sc_4f8188b4d1a8 (Gold11 CG1E0BBA2933, shaked_xv/C05).

For every gold line: slabs perpendicular to the gold axis every STEP mm from -EXT to L+EXT.
In each slab the two legs are found from the mesh normals (two dominant normal families
perpendicular to the gold axis). The mesh normals were verified 100 % outward-consistent
(10,695/10,695 skin points), so each leg's OUTER skin is identified directly as the skin whose
outward normal faces away from the other leg (a missing inner skin no longer breaks the fit).
Outer skins -> outer virtual sharp O and bisector b. The OUTER APEX is the outer-skin crossing of
the bisector (scan points within 0.35 mm of the bisector, outward normal facing -b). The gold
point of the slab is then expressed relative to that apex: along-bisector (+ = towards the
concave side) and across, plus a material-zone classification (convex air / in material /
concave air; on the arc sector or beside a flat leg). Nothing is fitted to the gold.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402

STEP = 2.0
EXT = 30.0
SLAB = 0.6      # half slab thickness along the axis (mm)
WIN = 11.0      # 2-D half window around the gold point (mm)
LEG_MIN = 2.6   # flat-leg fit uses points at least this far from the mid-plane corner
LEG_MAX = 10.0
COS_LEG = 0.985
T_NOM = 1.5     # AL5052 1.5 mm (drawing)


def leg_normals(p, n, s0, d, e1, e2, L):
    """Two dominant normal families (mod pi) perpendicular to d near the gold line."""
    rel = p - s0
    t = rel @ d
    u, v = rel @ e1, rel @ e2
    m = (t > -5) & (t < L + 5) & (u * u + v * v < 12.0 ** 2) & (np.abs(n @ d) < 0.25)
    ang = np.mod(np.arctan2(n[m] @ e2, n[m] @ e1), np.pi)
    h, edges = np.histogram(ang, bins=180, range=(0, np.pi))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    i1 = int(np.argmax(hs))
    dist = np.minimum(np.abs(np.arange(180) - i1), 180 - np.abs(np.arange(180) - i1))
    h2 = np.where(dist > 20, hs, 0)
    i2 = int(np.argmax(h2))
    out = []
    for i in (i1, i2):
        a0 = (edges[i] + edges[i + 1]) / 2
        sel = np.abs(np.angle(np.exp(2j * (ang - a0)))) < np.radians(4)
        a = 0.5 * np.angle(np.mean(np.exp(2j * ang[sel])))
        out.append(np.array([math.cos(a), math.sin(a)]))
    return out, int(m.sum()), (int(hs[i1]), int(hs[i2]))


def line_intersect(n1, c1, n2, c2):
    A = np.array([n1, n2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([c1, c2]))


def mode(vals, bw=0.1):
    vals = np.asarray(vals, float)
    if len(vals) < 3:
        return None
    lo = vals.min() - 0.3
    nb = max(3, int(math.ceil((vals.max() + 0.3 - lo) / bw)))
    h, edges = np.histogram(vals, bins=nb, range=(lo, lo + nb * bw))
    hs = np.convolve(h, [1, 2, 1], mode="same")
    k = int(np.argmax(hs))
    pk = (edges[k] + edges[k + 1]) / 2
    sel = vals[np.abs(vals - pk) < 0.3]
    return float(np.median(sel)) if len(sel) >= 3 else None


def station(p2, n2, nA, nB, gold2=np.zeros(2)):
    res = {"ok": False}
    legs = [np.abs(n2 @ nl) > COS_LEG for nl in (nA, nB)]
    if legs[0].sum() < 8 or legs[1].sum() < 8:
        res["why"] = f"leg points {int(legs[0].sum())}/{int(legs[1].sum())}"
        return res
    # pass 1: per leg, skins by outward-normal sign; rough mid-plane
    mids, skin_modes = [], []
    for nl, lm in zip((nA, nB), legs):
        off = p2[lm] @ nl
        sg = np.sign(n2[lm] @ nl)
        hi = mode(off[sg > 0])     # skin whose outward normal is +nl (the higher-offset skin)
        lo = mode(off[sg < 0])
        if hi is not None and lo is not None and 0.9 <= hi - lo <= 2.6:
            mid = (hi + lo) / 2
        elif hi is not None and (lo is None or not (0.9 <= hi - lo <= 2.6)):
            # pick the stronger single skin
            cnt_hi = int(np.sum(np.abs(off[sg > 0] - hi) < 0.3))
            cnt_lo = int(np.sum(np.abs(off[sg < 0] - lo) < 0.3)) if lo is not None else 0
            mid = hi - T_NOM / 2 if cnt_hi >= cnt_lo else lo + T_NOM / 2
        elif lo is not None:
            mid = lo + T_NOM / 2
        else:
            res["why"] = "no skin mode"
            return res
        mids.append(mid)
        skin_modes.append((lo, hi))
    X = line_intersect(nA, mids[0], nB, mids[1])
    if X is None:
        res["why"] = "parallel legs"
        return res
    dirs = []
    for nl, lm, mid in zip((nA, nB), legs, mids):
        ld = np.array([-nl[1], nl[0]])
        q = p2[lm]
        near = np.abs(q @ nl - mid) < 1.4
        proj = (q[near] - X) @ ld
        pos = int(np.sum((proj > 0.5) & (proj < LEG_MAX)))
        neg = int(np.sum((proj < -0.5) & (proj > -LEG_MAX)))
        if neg > pos:
            ld = -ld
        dirs.append(ld)
    # pass 2: outer / inner skin offsets from the flat part of each leg (own sheet only)
    outer, inner, measured, thick, leg_len = [], [], [], [], []
    for k, (nl, lm, ld, mid) in enumerate(zip((nA, nB), legs, dirs, mids)):
        other = dirs[1 - k]
        q = p2[lm]
        sg = np.sign(n2[lm] @ nl)
        off = q @ nl
        along = (q - X) @ ld
        own = np.abs(off - mid) < 1.4
        flat = own & (along > LEG_MIN) & (along < LEG_MAX)
        # outward normal of the outer skin faces away from the other leg: sign(nl.other) * sg < 0
        s_out = -np.sign(nl @ other)
        o_pts = off[flat & (sg == s_out)]
        i_pts = off[flat & (sg == -s_out)]
        o = float(np.median(o_pts)) if len(o_pts) >= 3 else None
        i = float(np.median(i_pts)) if len(i_pts) >= 3 else None
        if o is None and i is None:
            res["why"] = f"leg {'AB'[k]} no flat skin"
            return res
        if o is None:
            o = i + s_out * T_NOM
            measured.append("inner_only")
        elif i is None:
            i = o - s_out * T_NOM
            measured.append("outer_only")
        else:
            measured.append("both")
        outer.append(o)
        inner.append(i)
        thick.append(abs(o - i))
        leg_len.append(float(np.percentile(along[own], 98)) if own.any() else 0.0)
    O = line_intersect(nA, outer[0], nB, outer[1])
    I = line_intersect(nA, inner[0], nB, inner[1])
    b = dirs[0] + dirs[1]
    b /= np.linalg.norm(b)
    bperp = np.array([-b[1], b[0]])
    included = math.degrees(math.acos(np.clip(dirs[0] @ dirs[1], -1, 1)))
    rel = p2 - O
    along_b = rel @ b
    lat = rel @ bperp
    band = (np.abs(lat) < 0.35) & (along_b > -0.6) & (along_b < 6.0)
    nb_ = n2 @ b
    out_c = band & (nb_ < -0.3)
    in_c = band & (nb_ > 0.3)
    res.update({"O": O.tolist(), "I": I.tolist(), "dirs": [x.tolist() for x in dirs], "skins_measured": measured,
                "thickness": thick, "leg_len": leg_len, "included_deg": included, "bend_deg": 180 - included})
    if out_c.sum() < 2:
        res["why"] = "no outer-apex points"
        return res
    apex_b = float(np.median(along_b[out_c]))
    A = O + apex_b * b
    inner_b = float(np.median(along_b[in_c])) if in_c.sum() >= 2 else None
    # arc model: R_o from sharp->apex distance, centre on bisector
    phi = math.radians(included / 2)
    k_ = 1 / math.sin(phi) - 1
    R_o = apex_b / k_ if k_ > 1e-6 else float("nan")
    C = O + b * (R_o / math.sin(phi)) if np.isfinite(R_o) else None
    g = gold2
    gA = g - A
    zone = None
    if C is not None:
        t_used = float(np.median(thick))
        R_i = R_o - t_used
        rg = float(np.linalg.norm(g - C))
        # angular sector of the arc: between the two tangent directions (-normal to each leg, towards convex)
        vg = (g - C) / max(rg, 1e-9)
        # tangent-point directions from C: opposite of each leg's inward normal = perpendicular to leg towards outer
        t_dirs = []
        for k, nl in enumerate((nA, nB)):
            s_out = -np.sign(nl @ dirs[1 - k])
            t_dirs.append(nl * s_out)      # outward normal of the outer skin
        # vg inside the sector if it is a positive combination of the two tangent directions
        M = np.array(t_dirs).T
        try:
            coef = np.linalg.solve(M, vg)
            in_sector = bool(np.all(coef >= -1e-9))
        except np.linalg.LinAlgError:
            in_sector = False
        if in_sector:
            if rg > R_o + 0.15:
                zone = "ARC_CONVEX_AIR"
            elif rg >= R_i - 0.15:
                zone = "ARC_IN_MATERIAL"
            else:
                zone = "ARC_CONCAVE_AIR"
        else:
            # beside a flat leg: which leg (the one whose tangent side contains g)
            k = 0 if coef[0] < 0 and coef[1] >= 0 else 1 if coef[1] < 0 and coef[0] >= 0 else -1
            if k == -1:
                zone = "OUTSIDE_BOTH"
            else:
                kk = 1 - k   # g is beyond the tangent point of leg kk (lies alongside leg kk)
                nl = (nA, nB)[kk]
                s_out = -np.sign(nl @ dirs[1 - kk])
                d_out = (g @ nl - outer[kk]) * s_out     # >0 = beyond outer skin (convex air)
                d_in = (g @ nl - inner[kk]) * s_out      # <0 = beyond inner skin (concave air)
                pos = "CONVEX_AIR" if d_out > 0.15 else ("CONCAVE_AIR" if d_in < -0.15 else "IN_MATERIAL")
                zone = f"LEG{'AB'[kk]}_{pos}"
                res["gold_along_leg_from_outer_sharp"] = float((g - O) @ dirs[kk])
        res.update({"R_o": R_o, "C": C.tolist(), "gold_radius_from_C": rg})
    res.update({
        "ok": True, "A": A.tolist(), "b": b.tolist(),
        "apex_from_outer_sharp": apex_b, "inner_apex_from_outer_sharp": inner_b,
        "gold_minus_apex_along_bisector": float(gA @ b),
        "gold_minus_apex_across": float(gA @ bperp),
        "gold_to_apex_mm": float(np.linalg.norm(gA)),
        "gold_minus_outer_sharp_along_bisector": float((g - O) @ b),
        "gold_to_outer_sharp_mm": float(np.linalg.norm(g - O)),
        "gold_to_inner_sharp_mm": float(np.linalg.norm(g - I)),
        "gold_zone": zone,
        "n_outer_apex_pts": int(out_c.sum()),
    })
    return res


def run(only=None):
    p, n, _ = c.load_scan()
    gold, _ = c.load_gold()
    report = {}
    for gl in gold:
        if only and gl["id"] not in only:
            continue
        s0, e0 = gl["s"], gl["e"]
        L = float(np.linalg.norm(e0 - s0))
        d, e1, e2 = c.basis(e0 - s0)
        legs, nleg, peaks = leg_normals(p, n, s0, d, e1, e2, L)
        nA, nB = legs
        rel = p - s0
        t_all = rel @ d
        u_all, v_all = rel @ e1, rel @ e2
        pre = (t_all > -EXT - 1) & (t_all < L + EXT + 1) & (u_all * u_all + v_all * v_all < (WIN * 1.5) ** 2)
        idx = np.nonzero(pre)[0]
        tt, uu, vv = t_all[idx], u_all[idx], v_all[idx]
        nn2 = np.stack([n[idx] @ e1, n[idx] @ e2], 1)
        stations = []
        ext = min(EXT, 0.6 * L + 4)
        step = STEP if L > 40 else 1.0
        for sk in np.arange(-ext, L + ext + 1e-6, step):
            m = (np.abs(tt - sk) < SLAB) & (np.abs(uu) < WIN) & (np.abs(vv) < WIN)
            p2 = np.stack([uu[m], vv[m]], 1)
            r = station(p2, nn2[m], nA, nB)
            r["s"] = float(sk)
            r["frac"] = float(sk / L)
            r["in_gold"] = bool(0 <= sk <= L)
            r["n"] = int(m.sum())
            # 3-D apex point for this station (for the apex reference line)
            if r.get("ok"):
                A2 = np.array(r["A"])
                r["apex_xyz"] = (s0 + d * sk + e1 * A2[0] + e2 * A2[1]).tolist()
            stations.append(r)
        report[gl["id"]] = {"length_mm": L, "dir": d.tolist(), "e1": e1.tolist(), "e2": e2.tolist(),
                            "line_start": s0.tolist(), "line_end": e0.tolist(),
                            "leg_normals_2d": [x.tolist() for x in legs], "leg_hist_peaks": peaks,
                            "stations": stations}
        ok_in = [s for s in stations if s["ok"] and s["in_gold"]]
        if ok_in:
            ab = np.array([s["gold_minus_apex_along_bisector"] for s in ok_in])
            ac = np.array([s["gold_minus_apex_across"] for s in ok_in])
            dist = np.array([s["gold_to_apex_mm"] for s in ok_in])
            zones = {}
            for s in ok_in:
                zones[s["gold_zone"]] = zones.get(s["gold_zone"], 0) + 1
            print(f"{gl['id']} L={L:6.1f} ok {len(ok_in)}/{sum(s['in_gold'] for s in stations)} "
                  f"along_bis med {np.median(ab):+.2f} [{ab.min():+.2f},{ab.max():+.2f}]  across med {np.median(ac):+.2f} "
                  f"[{ac.min():+.2f},{ac.max():+.2f}]  |g-apex| med {np.median(dist):.2f} max {dist.max():.2f}  "
                  f"bend {np.median([s['bend_deg'] for s in ok_in]):.1f}  R_o {np.nanmedian([s.get('R_o', np.nan) for s in ok_in]):.2f}"
                  f"  zones {zones}")
        else:
            print(gl["id"], "no ok stations")
    return report


if __name__ == "__main__":
    only = sys.argv[1:] or None
    rep = run(only)
    out = c.OUT / ("a3_xsec_stations.json" if not only else f"a3_xsec_stations_{'_'.join(only)}.json")
    out.write_text(json.dumps(rep, indent=1))
    print(out)
