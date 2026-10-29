"""Per-gold-line station analysis for sc_2714fb7e0c0b (auditor 1). Read-only on inputs.

For every gold line, cross-sections (slab +-1 mm) every 1 mm from 12 mm before the start to
12 mm after the end. In each section the outer bend arc is found from the ORIENTED scan normals
(Geomagic export): outer-skin arc points have normals within 35 deg of the outward bisector o.
A normal-based circle fit p_i = C + R n_i gives the arc centre C (bend axis) and outer radius R;
the outer apex is A = C + R o. Flange outer/inner faces (normals within 8 deg of the flange
normals) give outer and inner virtual sharps. The gold point is then expressed relative to the
apex: along the inward bisector g = -o (+ = concave/inside) and across.

Fold present at a station = >= 6 outer-arc points, 0.8 <= R <= 6 mm and both flanges' outer faces
seen within 15 mm. Fold extent = longest run of present stations (gaps <= 3 mm bridged).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
import a1common as A

pts, nrm, _ = A.load_scan()
gold = A.load_gold()
ex, ez, en = A.part_frame(gold)
X, Z, N = ex, ez, en
u = A.unit

# outer face normals of the two flanges (outward, raw frame) for each gold line, from the part layout
CFG = {
    "B01": (-N, -X),   # main panel (X>-112) outer face -N ; side flange (rises +N) outer face -X
    "B02": (-N, X),
    "B03": (-N, Z),    # main panel outer -N ; step wall outer +Z
    "B04": (N, Z),     # lower panel outer +N? -> see note below
    "B05": (N, -Z),    # lower panel outer +N ; step wall outer -Z
    "B07": (-N, -Z),   # main panel outer -N ; far flange (rises +N at Z=-138) outer -Z
    "B09": (N, X),     # main panel outer +N (tab hangs -N) ; tab outer +X (tab at window edge, panel on -X side)
    "B10": (N, -X),
    "B06": (-X, Z),    # IF the corner were a fold: end flange (X=-112) outer -X ; return flange (Z=118) outer +Z
    "B08": (X, Z),
}
# B04: lower panel (N~30, Z 73..118) -> return flange rising +N at Z=118: concave side is (+N,-Z),
# so the outer faces are the lower panel's -N face and the return flange's +Z face.
CFG["B04"] = (-N, Z)


def analyze(bid, s0, s1, nA, nB, pad=12.0, step=1.0, h=1.0, tree=None):
    d = u(s1 - s0)
    L = float(np.linalg.norm(s1 - s0))
    o = u(nA + nB)          # outward apex normal
    g = -o                   # inward bisector
    v = u(np.cross(d, g))    # across
    cA, cB = float(np.cos(np.radians(8))), float(np.cos(np.radians(35)))
    rows = []
    for s in np.arange(-pad, L + pad + 1e-9, step):
        c = s0 + s * d
        idx = np.array(tree.query_ball_point(c, 16.0), int)
        if len(idx) == 0:
            rows.append({"s": float(s), "n": 0}); continue
        p, n = pts[idx], nrm[idx]
        al = (p - c) @ d
        m = np.abs(al) < h
        p, n = p[m], n[m]
        q = np.c_[(p - c) @ g, (p - c) @ v]          # 2D section coords, gold at origin
        nq = np.c_[n @ g, n @ v]
        r = {"s": float(s), "n": int(len(p))}
        arc = (n @ o > cB) & (np.linalg.norm(q, axis=1) < 9.0)
        r["n_arc"] = int(arc.sum())
        fa = (n @ nA > cA) & (np.linalg.norm(q, axis=1) < 15.0)
        fb = (n @ nB > cA) & (np.linalg.norm(q, axis=1) < 15.0)
        r["n_faceA"], r["n_faceB"] = int(fa.sum()), int(fb.sum())
        r["d_nearest"] = float(np.min(np.linalg.norm(q, axis=1))) if len(q) else None
        if arc.sum() >= 6:
            qa, na = q[arc], nq[arc]
            # p = C + R n  (least squares in C (2) and R)
            Amat = np.zeros((2 * len(qa), 3))
            Amat[0::2, 0] = 1; Amat[1::2, 1] = 1
            Amat[0::2, 2] = na[:, 0]; Amat[1::2, 2] = na[:, 1]
            sol, *_ = np.linalg.lstsq(Amat, qa.reshape(-1), rcond=None)
            C2, R = sol[:2], float(sol[2])
            o2 = np.array([o @ g, o @ v])  # = (-1, 0)
            apex = C2 + R * o2
            r.update({"R": R, "C": C2.tolist(), "apex": apex.tolist(),
                      "gold_minus_apex_inward": float(-apex[0]), "gold_minus_apex_across": float(-apex[1]),
                      "gold_to_apex": float(np.linalg.norm(apex)), "gold_to_centre": float(np.linalg.norm(C2))})
        # outer / inner face lines -> virtual sharps
        def fit_line(sel, nvec):
            if sel.sum() < 5:
                return None
            nv2 = np.array([nvec @ g, nvec @ v]); nv2 /= np.linalg.norm(nv2)
            off = np.median(q[sel] @ nv2)
            return nv2, float(off)
        la, lb = fit_line(fa, nA), fit_line(fb, nB)
        ia = (n @ -nA > cA) & (np.linalg.norm(q, axis=1) < 15.0)
        ib = (n @ -nB > cA) & (np.linalg.norm(q, axis=1) < 15.0)
        lia, lib = fit_line(ia, -nA), fit_line(ib, -nB)
        def inter(l1, l2):
            if l1 is None or l2 is None:
                return None
            Mx = np.array([l1[0], l2[0]])
            if abs(np.linalg.det(Mx)) < 1e-3:
                return None
            return np.linalg.solve(Mx, np.array([l1[1], l2[1]]))
        xo, xi = inter(la, lb), inter(lia, lib)
        if xo is not None:
            r["outer_sharp"] = xo.tolist(); r["gold_to_outer_sharp"] = float(np.linalg.norm(xo))
        if xi is not None:
            r["inner_sharp"] = xi.tolist(); r["gold_to_inner_sharp"] = float(np.linalg.norm(xi))
        if la is not None and lia is not None:
            r["thickA"] = float(abs(la[1] + lia[1]))  # inner normal is -nA so offsets add
        r["present"] = bool(r["n_arc"] >= 6 and "R" in r and 0.8 <= r["R"] <= 6.0 and r["n_faceA"] >= 5 and r["n_faceB"] >= 5)
        rows.append(r)
    # extent: longest run of present stations (bridge gaps <= 3 mm)
    pres = np.array([x.get("present", False) for x in rows])
    ss = np.array([x["s"] for x in rows])
    runs, cur = [], None
    for i, ok in enumerate(pres):
        if ok:
            if cur is None:
                cur = [i, i]
            elif ss[i] - ss[cur[1]] <= 3.0 + 1e-9:
                cur[1] = i
            else:
                runs.append(cur); cur = [i, i]
    if cur is not None:
        runs.append(cur)
    runs_mm = [(float(ss[a]), float(ss[b])) for a, b in runs]
    best = max(runs_mm, key=lambda ab: ab[1] - ab[0]) if runs_mm else None
    inside = [x for x in rows if x.get("present") and 0 <= x["s"] <= L]
    def med(k):
        vals = [x[k] for x in inside if k in x]
        return (float(np.median(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))) if vals else None
    summ = {"bend_id": bid, "length_mm": L, "fold_runs_s_mm": runs_mm, "fold_extent_s_mm": best,
            "n_present_in_span": len(inside), "n_stations_in_span": int(sum(1 for x in rows if 0 <= x["s"] <= L)),
            "inward_offset_mm(med,p10,p90)": med("gold_minus_apex_inward"),
            "across_offset_mm(med,p10,p90)": med("gold_minus_apex_across"),
            "gold_to_apex_mm(med,p10,p90)": med("gold_to_apex"),
            "gold_to_centre_mm(med,p10,p90)": med("gold_to_centre"),
            "gold_to_outer_sharp_mm(med,p10,p90)": med("gold_to_outer_sharp"),
            "gold_to_inner_sharp_mm(med,p10,p90)": med("gold_to_inner_sharp"),
            "R_outer_mm(med,p10,p90)": med("R"), "thickA_mm(med)": med("thickA")}
    if best:
        f0, f1 = best
        ov = max(0.0, min(L, f1) - max(0.0, f0))
        summ["coverage_of_fold_pct"] = 100.0 * ov / max(1e-9, f1 - f0)
        summ["gold_inside_fold_pct"] = 100.0 * ov / L
        summ["overshoot_start_mm"] = max(0.0, f0 - 0.0)
        summ["overshoot_end_mm"] = max(0.0, L - f1)
        summ["fold_beyond_start_mm"] = max(0.0, 0.0 - f0)
        summ["fold_beyond_end_mm"] = max(0.0, f1 - L)
    return summ, rows


if __name__ == "__main__":
    tree = cKDTree(pts)
    only = sys.argv[1:] or list(CFG)
    out = {}
    for bid in only:
        s0, s1 = gold[bid]
        nA, nB = CFG[bid]
        summ, rows = analyze(bid, s0, s1, nA, nB, tree=tree)
        out[bid] = {"summary": summ, "stations": rows}
        print(json.dumps(summ))
    (A.OUT / "a1_stations.json").write_text(json.dumps(out, indent=0))
