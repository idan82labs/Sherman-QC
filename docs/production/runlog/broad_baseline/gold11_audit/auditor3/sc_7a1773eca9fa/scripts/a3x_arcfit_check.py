"""Auditor 3 (resumed run): independent cross-check of where each Gold11 human line sits on its fold.

Method (independent of a3_measure / a3_loci, which used skin-line intersections):
  * plate normals come from the bench_labeling prep patches (patches.json), not from axial modes;
  * per station (slab +-0.5 mm perpendicular to the fold axis d = n_base x n_plate) the arc points
    (normals > 6 deg from both plates, |n.d| < 0.35, within 14 mm of a provisional centre) are fitted
    with ONE shared centre C and two radii (inner Ri, outer Ro) -- concentric-arc model;
  * the outer apex is C + Ro * n_apex, n_apex = bisector of the two plates' outward normals;
  * the gold point G (gold line pierced by the section plane) is expressed as
        r_G   = |G - C|                (Ri = inner skin, Ro = outer skin)
        theta = polar angle of G around C from n_apex (0 = apex, +-beta/2 = tangent lines)
    and, when it is beyond a tangent line, as signed depth below that plate's inner skin and the
    distance past the inner tangent line along the plate.
Outputs a JSON with per-station rows and per-line summaries.
"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa/scripts")
from a3_load import load  # noqa: E402

AUD = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa"
GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"
PATCHES = AUD + "/prep/patches.json"
# fold -> (base patch, other plate patch, plate name)
FOLDS = {"M01": (1, 5, "front flange"), "M02": (1, 9, "left tab"), "M03": (1, 11, "right tab"),
         "M04": (1, 6, "left wall"), "M05": (1, 3, "right wall")}


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def fit_concentric(pi, po, c0):
    """Shared-centre fit of inner and outer arc points (robust soft-L1)."""
    def res(p):
        c = p[:2]
        r_i, r_o = p[2], p[3]
        a = np.hypot(*(pi - c).T) - r_i if len(pi) else np.zeros(0)
        b = np.hypot(*(po - c).T) - r_o if len(po) else np.zeros(0)
        return np.r_[a, b]
    ri0 = np.median(np.hypot(*(pi - c0).T)) if len(pi) else 3.3
    ro0 = np.median(np.hypot(*(po - c0).T)) if len(po) else ri0 + 2.0
    sol = least_squares(res, np.r_[c0, ri0, ro0], loss="soft_l1", f_scale=0.15)
    r = res(sol.x)
    return sol.x[:2], float(sol.x[2]), float(sol.x[3]), float(np.sqrt(np.mean(r ** 2)))


def normal_line_centre(p, n):
    """Least-squares point closest to all normal lines (p_i + t n_i)."""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for pi, ni in zip(p, n):
        P = np.eye(2) - np.outer(ni, ni)
        A += P; b += P @ pi
    return np.linalg.solve(A, b)


def analyse(v, n, bid, patches, step=1.0, margin=12.0):
    pb, pp, pname = FOLDS[bid]
    by = {q["id"]: q for q in patches}
    nb, npl = unit(by[pb]["normal"]), unit(by[pp]["normal"])
    gold = {b["bend_id"]: b for b in json.load(open(GOLD))["bends"]}[bid]
    S, E = np.array(gold["line_start"], float), np.array(gold["line_end"], float)
    d = unit(np.cross(nb, npl))
    if d @ (E - S) < 0:
        d = -d
    mid = (S + E) / 2
    e1 = unit(nb - (nb @ d) * d)
    e2 = np.cross(d, e1)
    rel = v - mid
    s_all = rel @ d
    lat = np.linalg.norm(rel - np.outer(s_all, d), axis=1)
    sa, sb = (S - mid) @ d, (E - mid) @ d
    lo, hi = min(sa, sb), max(sa, sb)
    keep = (s_all > lo - margin - 1) & (s_all < hi + margin + 1) & (lat < 40)
    P = rel[keep]; N = n[keep]
    s = P @ d
    xy = np.c_[P @ e1, P @ e2]
    nxy = np.c_[N @ e1, N @ e2]
    nd = N @ d
    a2 = unit([nb @ e1, nb @ e2])
    b2 = unit([npl @ e1, npl @ e2])
    nn = np.linalg.norm(nxy, axis=1) + 1e-12
    cosA = np.abs(nxy @ a2) / nn
    cosB = np.abs(nxy @ b2) / nn
    beta_planes = float(np.degrees(np.arccos(min(1.0, abs(a2 @ b2)))))
    rows = []
    prevC = None
    for s0 in np.arange(lo - margin, hi + margin + 1e-9, step):
        m = np.abs(s - s0) <= 0.5
        row = dict(s=float(s0))
        lam = (s0 - sa) / (sb - sa)
        G3 = S + lam * (E - S) - mid
        G = np.array([G3 @ e1, G3 @ e2])
        row["lam"] = float(lam)
        if m.sum() < 30:
            row["ok"] = False
            rows.append(row)
            continue
        X, NX, ND = xy[m], nxy[m] / nn[m][:, None], nd[m]
        cA, cB = cosA[m], cosB[m]
        arc = (cA < np.cos(np.radians(6))) & (cB < np.cos(np.radians(6))) & (np.abs(ND) < 0.35)
        # provisional centre: previous station or the normal-line intersection of arc points near the gold point
        c0 = prevC if prevC is not None else G
        near = arc & (np.hypot(*(X - c0).T) < 14)
        if near.sum() < 10 and prevC is not None:
            c0 = G  # a bad previous centre must not poison the march
            near = arc & (np.hypot(*(X - c0).T) < 14)
        if near.sum() < 10:
            row["ok"] = False
            row["why"] = f"arc pts {int(near.sum())}"
            rows.append(row)
            continue
        C = normal_line_centre(X[near], NX[near])
        near = arc & (np.hypot(*(X - C).T) < 12)
        if near.sum() < 10:
            row["ok"] = False
            row["why"] = "arc pts after centre"
            rows.append(row)
            continue
        # inner vs outer arc: normal points toward (inner) or away from (outer) the centre
        radial = (X[near] - C) / (np.hypot(*(X[near] - C).T)[:, None] + 1e-12)
        side = np.einsum("ij,ij->i", radial, NX[near])
        pi_, po_ = X[near][side < -0.5], X[near][side > 0.5]
        if len(pi_) < 6 and len(po_) < 6:
            row["ok"] = False
            row["why"] = "no arc"
            rows.append(row)
            continue
        C, Ri, Ro, rms = fit_concentric(pi_, po_, C)
        row["inner_arc_scanned"] = bool(len(pi_) >= 6)
        row["outer_arc_scanned"] = bool(len(po_) >= 6)
        if len(pi_) < 6:
            Ri = Ro - 2.03  # bench thickness; inner arc not scanned at this station
        if len(po_) < 6:
            Ro = Ri + 2.03
        # orient plate normals away from C using flat points of each plate within 3..25 mm of C
        flatA = (cA > np.cos(np.radians(4))) & (np.hypot(*(X - C).T) < 25)
        flatB = (cB > np.cos(np.radians(4))) & (np.hypot(*(X - C).T) < 25)
        if flatA.sum() < 5 or flatB.sum() < 5:
            row["ok"] = False
            row["why"] = "flats missing"
            rows.append(row)
            continue
        aa = a2 if np.median((X[flatA] - C) @ a2) > 0 else -a2
        bb = b2 if np.median((X[flatB] - C) @ b2) > 0 else -b2
        n_apex = unit(aa + bb)
        beta = float(np.degrees(np.arccos(np.clip(aa @ bb, -1, 1))))
        # skin offsets of flats (distance from C along plate normal): inner ~Ri, outer ~Ro if tangent
        offA = (X[flatA] - C) @ aa
        offB = (X[flatB] - C) @ bb
        apex_o = C + Ro * n_apex
        apex_i = C + Ri * n_apex
        g = G - C
        rG = float(np.hypot(*g))
        across = np.array([-n_apex[1], n_apex[0]])
        if across @ bb < 0:
            across = -across  # +theta toward plate B (the non-base plate)
        theta = float(np.degrees(np.arctan2(g @ across, g @ n_apex)))
        # nearest scanned point (2D, in slab) and whether the scan exists near the outer apex
        dn = np.hypot(*(X - G).T)
        da = np.hypot(*(X - apex_o).T)
        row.update(ok=True, n_arc_in=int(len(pi_)), n_arc_out=int(len(po_)), C=C.tolist(), n_apex=n_apex.tolist(),
                   across=across.tolist(), Ri=Ri, Ro=Ro, t=Ro - Ri,
                   fit_rms=rms, beta=beta, gold=G.tolist(), r_gold=rG, theta_gold=theta,
                   half_sweep=beta / 2, gold_to_outer_apex=float(np.hypot(*(G - apex_o))),
                   gold_to_inner_apex=float(np.hypot(*(G - apex_i))),
                   gold_minus_apex_along_bisector=float((apex_o - G) @ n_apex),
                   gold_minus_apex_across=float((G - apex_o) @ across),
                   gold_nearest_scan=float(dn.min()), outer_apex_nearest_scan=float(da.min()),
                   flatA_inner_off=float(np.quantile(offA, 0.1)), flatA_outer_off=float(np.quantile(offA, 0.9)),
                   flatB_inner_off=float(np.quantile(offB, 0.1)), flatB_outer_off=float(np.quantile(offB, 0.9)))
        # where is G relative to the tangent lines / skins
        if abs(theta) <= beta / 2:
            row["gold_zone"] = "arc_sector"
            row["gold_depth_below_inner_skin"] = rG - Ri  # >0: inside material (toward outer)
        else:
            plate = bb if theta > 0 else aa
            u_along = np.array([-plate[1], plate[0]])
            # direction along the plate away from the arc: perpendicular to plate normal, pointing away from apex side
            if u_along @ (n_apex - plate) > 0:
                u_along = -u_along
            row["gold_zone"] = "past_tangent_on_" + ("B" if theta > 0 else "A")
            row["gold_depth_below_inner_skin"] = float(g @ plate - Ri)
            row["gold_past_inner_tangent"] = float(g @ u_along)  # tangent point is at along = 0 from C
        if row["fit_rms"] < 0.6 and 0.5 < Ri < 15:
            prevC = C
        else:
            prevC = None
            row["suspect_fit"] = True
        rows.append(row)
    return dict(bid=bid, plate=pname, d=d.tolist(), gold_s=[float(lo), float(hi)], beta_planes=beta_planes,
                angle_gold_vs_axis=float(np.degrees(np.arccos(min(1.0, abs(unit(E - S) @ d))))), rows=rows)


def summarise(res):
    lo, hi = res["gold_s"]
    ok = [r for r in res["rows"] if r.get("ok")]
    ing = [r for r in ok if lo <= r["s"] <= hi]
    f = lambda k: np.array([r[k] for r in ing if k in r], float)
    out = dict(bid=res["bid"], plate=res["plate"], angle_gold_vs_axis=res["angle_gold_vs_axis"], beta_planes=res["beta_planes"],
               n_stations_gold=len(ing))
    for k in ("gold_to_outer_apex", "gold_to_inner_apex", "gold_minus_apex_along_bisector", "gold_minus_apex_across", "Ri", "Ro", "t", "beta",
              "r_gold", "theta_gold", "gold_nearest_scan", "outer_apex_nearest_scan", "gold_depth_below_inner_skin", "fit_rms"):
        a = f(k)
        if len(a):
            out[k] = dict(med=round(float(np.median(a)), 2), p10=round(float(np.percentile(a, 10)), 2), p90=round(float(np.percentile(a, 90)), 2))
    zones = {}
    for r in ing:
        zones[r["gold_zone"]] = zones.get(r["gold_zone"], 0) + 1
    out["gold_zone_counts"] = zones
    pt = f("gold_past_inner_tangent")
    if len(pt):
        out["gold_past_inner_tangent"] = dict(med=round(float(np.median(pt)), 2), p10=round(float(np.percentile(pt, 10)), 2),
                                              p90=round(float(np.percentile(pt, 90)), 2), n=len(pt))
    # fold extent from arc-fit presence (longest run of ok stations)
    ss = np.array([r["s"] for r in res["rows"]])
    okm = np.array([bool(r.get("ok")) for r in res["rows"]])
    best, cur, start = (0, None, None), 0, None
    for i, o in enumerate(okm):
        if o:
            if start is None:
                start = i
            if i - start + 1 > best[0]:
                best = (i - start + 1, start, i)
        else:
            start = None
    if best[1] is not None:
        f_lo, f_hi = ss[best[1]] - 0.5, ss[best[2]] + 0.5
        out["arc_extent_s"] = [float(f_lo), float(f_hi)]
        out["arc_extent_len"] = float(f_hi - f_lo)
        out["gold_covers_pct_of_arc_extent"] = round(100 * max(0.0, min(hi, f_hi) - max(lo, f_lo)) / (f_hi - f_lo), 1)
        out["gold_overrun_mm"] = round(float(max(0, f_lo - lo) + max(0, hi - f_hi)), 2)
    return out


if __name__ == "__main__":
    v, n, f = load()
    patches = json.load(open(PATCHES))["patches"]
    allres, summ = {}, {}
    for bid in (sys.argv[1:] or list(FOLDS)):
        res = analyse(v, n, bid, patches)
        allres[bid] = res
        summ[bid] = summarise(res)
        print(json.dumps(summ[bid]))
    json.dump(dict(summary=summ, detail=allres), open(AUD + "/a3x_arcfit_check.json", "w"), default=float)
