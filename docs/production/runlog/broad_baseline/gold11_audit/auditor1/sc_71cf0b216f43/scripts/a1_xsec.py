"""Auditor-1 (sc_71cf0b216f43 / 10839000): dense perpendicular cross-sections of every fold.

Read-only on the scan and the gold. Works in a part frame (x long axis, y short axis, z flange
normal; a1_frame.py), which is a pure rotation+translation of the raw scan, so all distances are mm.

For every fold (the 8 wall folds the human labelled + the 4 joggle folds) and every station along
the fold axis (2 mm pitch), a 2D section (slab +-0.75 mm) is analysed:
  * outer / inner skin lines of both plates (robust line fits on points whose mesh normals agree
    with the plate normal and point to the convex / concave side),
  * outer virtual sharp Xo (outer skins), inner virtual sharp Xi (inner skins),
  * OUTER APEX = outer-skin point nearest Xo (= mid-point of a circular arc tangent to both skins),
    cross-checked by a circle fit of the outer arc (apex_c = C + R * outward bisector),
  * the gold line's crossing of the section: offset to the apex (total, along the bisector toward
    the concave side, across toward plate A/B), to Xo, Xi, the inner apex, nearest scan point and the
    normal of that point (inner/outer skin).
Fold extent = stations where both plates are present at the corner and an outer arc is seen.
Output: a1_xsec_measurements.json (+ plots by a1_render.py).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get, unit  # noqa: E402

H = 0.75  # slab half width mm

# fold definitions in the part frame.
# axis: 0 -> fold runs along x (section coords (y, z)); 1 -> fold runs along y (section coords (x, z)).
# corner0: approx outer corner (u, z) in section coords; plate A = wall, plate B = flange.
# A_out: outward normal of plate A outer skin (2D, section coords); A_dir: direction plate A extends from corner.
FOLDS = {
    # short walls, fold along y, section (x, z)
    "RT": dict(axis=1, gold="M01", corner=(149.0, 21.0), A_out=(1, 0), A_dir=(0, -1), B_out=(0, 1), B_dir=(-1, 0),
               desc="right short wall -> 12 mm top lip"),
    "LT": dict(axis=1, gold="M02", corner=(-163.7, 21.0), A_out=(-1, 0), A_dir=(0, -1), B_out=(0, 1), B_dir=(1, 0),
               desc="left short wall -> 12 mm top lip"),
    "RB": dict(axis=1, gold="M03", corner=(149.0, -20.9), A_out=(1, 0), A_dir=(0, 1), B_out=(0, -1), B_dir=(-1, 0),
               desc="right short wall -> 35 mm bottom frame flange"),
    "LB": dict(axis=1, gold="M04", corner=(-163.7, -20.9), A_out=(-1, 0), A_dir=(0, 1), B_out=(0, -1), B_dir=(1, 0),
               desc="left short wall -> 35 mm bottom frame flange"),
    # long walls, fold along x, section (y, z)
    "FB": dict(axis=0, gold="M05", corner=(-99.1, -20.9), A_out=(-1, 0), A_dir=(0, 1), B_out=(0, -1), B_dir=(1, 0),
               desc="front long wall -> 35 mm bottom frame flange"),
    "FT": dict(axis=0, gold="M06", corner=(-99.1, 21.6), A_out=(-1, 0), A_dir=(0, -1), B_out=(0, 1), B_dir=(1, 0),
               desc="front long wall -> 35 mm top flange (joggled)"),
    "BB": dict(axis=0, gold="M07", corner=(106.9, -20.9), A_out=(1, 0), A_dir=(0, 1), B_out=(0, -1), B_dir=(-1, 0),
               desc="back long wall -> 35 mm bottom frame flange"),
    "BT": dict(axis=0, gold="M08", corner=(106.9, 21.6), A_out=(1, 0), A_dir=(0, -1), B_out=(0, 1), B_dir=(-1, 0),
               desc="back long wall -> 35 mm top flange (joggled)"),
}


def section(P, VN, tree_axis_sorted, axis, s0, center2, rad=16.0):
    """Points of the slab |coord_axis - s0| < H within rad of center2 in section coords."""
    order, keys = tree_axis_sorted
    lo, hi = np.searchsorted(keys, [s0 - H, s0 + H])
    idx = order[lo:hi]
    cols = [1, 2] if axis == 0 else [0, 2]
    q = P[idx][:, cols]
    nq = VN[idx][:, cols]
    m = np.linalg.norm(q - np.asarray(center2), axis=1) < rad
    return q[m], nq[m], idx[m]


def fit_line(pts, d0):
    """Robust 2D line: returns point, unit direction (sign aligned with d0), rms."""
    if len(pts) < 6:
        return None
    keep = np.ones(len(pts), bool)
    for _ in range(4):
        c = pts[keep].mean(0)
        x = pts[keep] - c
        w, v = np.linalg.eigh(x.T @ x)
        d = v[:, 1]
        n = np.array([-d[1], d[0]])
        r = (pts - c) @ n
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-3
        keep = np.abs(r - np.median(r[keep])) < max(3.5 * 1.4826 * mad, 0.12)
        if keep.sum() < 6:
            return None
    d = d * np.sign(d @ np.asarray(d0, float))
    return c, d, float(np.sqrt(np.mean(r[keep] ** 2)))


def intersect(l1, l2):
    p1, t1 = l1[0], l1[1]
    p2, t2 = l2[0], l2[1]
    M = np.array([t1, -t2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    s, _ = np.linalg.solve(M, p2 - p1)
    return p1 + s * t1


def fit_circle(pts):
    if len(pts) < 5:
        return None
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = sol
    r = math.sqrt(max(k + cx * cx + cy * cy, 0))
    return np.array([cx, cy]), r


def analyse_station(q, nq, f, Xo_prev=None):
    A_out = unit(f["A_out"]); A_dir = unit(f["A_dir"]); B_out = unit(f["B_out"]); B_dir = unit(f["B_dir"])
    c0 = np.asarray(f["corner"], float)
    res = {"ok": False}
    if len(q) < 30:
        res["why"] = "few_points"
        return res
    nn = np.linalg.norm(nq, axis=1)
    nn[nn == 0] = 1
    nqu = nq / nn[:, None]
    rel = q - c0
    a_al = rel @ A_dir  # distance along plate A from the approx corner
    b_al = rel @ B_dir
    cosA = nqu @ A_out
    cosB = nqu @ B_out
    tol = math.cos(math.radians(15))
    # plate A skins: points 4..14 mm along A from corner, normal parallel to A_out
    selA = (a_al > 4.0) & (a_al < 14.0) & (np.abs(rel @ A_out) < 4.0)
    selB = (b_al > 3.5) & (b_al < 10.0) & (np.abs(rel @ B_out) < 4.0)
    lines = {}
    for name, sel, cs, d0 in (("A_out", selA & (cosA > tol), cosA, A_dir), ("A_in", selA & (cosA < -tol), cosA, A_dir),
                              ("B_out", selB & (cosB > tol), cosB, B_dir), ("B_in", selB & (cosB < -tol), cosB, B_dir)):
        lines[name] = fit_line(q[sel], d0) if sel.sum() >= 6 else None
        res["n_" + name] = int(sel.sum())
    if lines["A_out"] is None or lines["B_out"] is None:
        res["why"] = "missing_outer_skin_" + ("A" if lines["A_out"] is None else "B")
        return res
    Xo = intersect(lines["A_out"], lines["B_out"])
    if Xo is None or np.linalg.norm(Xo - c0) > 8:
        res["why"] = "bad_outer_sharp"
        return res
    a = lines["A_out"][1]  # direction into plate A
    b = lines["B_out"][1]
    g_in = unit(a + b)  # bisector toward concave side
    included = math.degrees(math.acos(np.clip(a @ b, -1, 1)))
    # outer arc / apex: points within 7 mm of Xo whose normal points to the convex side (n . -g_in > 0.3)
    relx = q - Xo
    dX = np.linalg.norm(relx, axis=1)
    outer_like = (nqu @ (-g_in)) > 0.3
    cand = outer_like & (dX < 7.0) & ((relx @ g_in) > -0.6)
    res["n_outer_near_Xo"] = int(cand.sum())
    if cand.sum() < 3:
        res["why"] = "no_outer_arc_points"
        return res
    # apex: mean of the 4 outer points nearest Xo (their projection on the bisector)
    order = np.argsort(dX[cand])
    near = q[cand][order[:4]]
    delta = float(np.median((near - Xo) @ g_in))
    apex = Xo + g_in * delta
    res["apex_perp_spread"] = float(np.median(np.abs((near - Xo) @ np.array([-g_in[1], g_in[0]]))))
    # perpendicular spread check: nearest points should sit on the bisector
    half = math.radians(included) / 2
    k = 1 / math.sin(half) - 1
    R_from_delta = delta / k if k > 1e-6 else None
    # circle fit of arc points (between tangent points): outer-like, normal not parallel to either plate
    arc_sel = outer_like & (dX < 7.0) & (nqu @ A_out < math.cos(math.radians(12))) & (nqu @ B_out < math.cos(math.radians(12)))
    circ = fit_circle(q[arc_sel]) if arc_sel.sum() >= 6 else None
    apex_c = None
    Rc = None
    if circ is not None and circ[1] < 15:
        C, Rc = circ
        apex_c = C - g_in * Rc
    Xi = None
    if lines["A_in"] is not None and lines["B_in"] is not None:
        Xi = intersect(lines["A_in"], lines["B_in"])
        if Xi is not None and np.linalg.norm(Xi - Xo) > 6:
            Xi = None
    # inner apex: inner-like points nearest Xi, else None
    inner_apex = None
    if Xi is not None:
        inner_like = (nqu @ g_in) > 0.3
        dXi = np.linalg.norm(q - Xi, axis=1)
        ci = inner_like & (dXi < 5.0)
        if ci.sum() >= 3:
            o2 = np.argsort(dXi[ci])
            nr = q[ci][o2[:4]]
            inner_apex = Xi + g_in * float(np.median((nr - Xi) @ g_in))  # moves toward convex side -> negative
    thickA = None
    if lines["A_in"] is not None:
        thickA = float(abs((lines["A_in"][0] - lines["A_out"][0]) @ np.array([-a[1], a[0]])))
    thickB = None
    if lines["B_in"] is not None:
        thickB = float(abs((lines["B_in"][0] - lines["B_out"][0]) @ np.array([-b[1], b[0]])))
    res.update(ok=True, Xo=Xo.tolist(), apex=apex.tolist(), apex_c=None if apex_c is None else apex_c.tolist(),
               Rc=Rc, R_from_delta=R_from_delta, delta_apex_from_Xo=delta, included_deg=included,
               bend_deg=180 - included, g_in=g_in.tolist(), a=a.tolist(), b=b.tolist(),
               Xi=None if Xi is None else Xi.tolist(), inner_apex=None if inner_apex is None else inner_apex.tolist(),
               thickA=thickA, thickB=thickB,
               rmsA=lines["A_out"][2], rmsB=lines["B_out"][2])
    return res


def gold_offsets(res, g2, q, nq):
    """Offsets of the gold crossing g2 (2D) relative to this station's fold model."""
    apex = np.array(res["apex"]); g_in = np.array(res["g_in"]); a = np.array(res["a"]); b = np.array(res["b"])
    d = g2 - apex
    perp = np.array([-g_in[1], g_in[0]])
    # sign 'across' positive toward plate A (wall)
    if perp @ a < 0:
        perp = -perp
    out = {"gold_2d": g2.tolist(), "d_apex": float(np.linalg.norm(d)), "along_bisector_concave": float(d @ g_in),
           "across_toward_A": float(d @ perp), "d_Xo": float(np.linalg.norm(g2 - np.array(res["Xo"])))}
    if res.get("apex_c") is not None:
        out["d_apex_circle"] = float(np.linalg.norm(g2 - np.array(res["apex_c"])))
    if res.get("Xi") is not None:
        out["d_Xi"] = float(np.linalg.norm(g2 - np.array(res["Xi"])))
    if res.get("inner_apex") is not None:
        out["d_inner_apex"] = float(np.linalg.norm(g2 - np.array(res["inner_apex"])))
    # distance of gold point along each plate from the outer sharp (which plate is it 'on')
    out["gold_along_A_from_Xo"] = float((g2 - np.array(res["Xo"])) @ a)
    out["gold_along_B_from_Xo"] = float((g2 - np.array(res["Xo"])) @ b)
    if len(q):
        dd = np.linalg.norm(q - g2, axis=1)
        j = int(np.argmin(dd))
        n = nq[j] / (np.linalg.norm(nq[j]) + 1e-12)
        out["d_nearest_section_pt"] = float(dd[j])
        out["nearest_pt_normal_dot_concave"] = float(n @ g_in)  # >0 inner (concave-facing) skin, <0 outer skin
        out["nearest_pt_normal_2d"] = n.tolist()
    return out


def gold_cross(Gs, Ge, axis, s0):
    d = Ge - Gs
    if abs(d[axis]) < 1e-9:
        return None, None
    t = (s0 - Gs[axis]) / d[axis]
    p = Gs + t * d
    cols = [1, 2] if axis == 0 else [0, 2]
    return p[cols], t


def main():
    P, VN, F, G, R, c = get()
    gold = {bid: (s, e) for bid, s, e in G}
    sorted_by = {}
    for ax in (0, 1):
        order = np.argsort(P[:, ax])
        sorted_by[ax] = (order, P[order, ax])
    tree_all = cKDTree(P)
    out = {"scan": "sc_71cf0b216f43", "frame": "part frame (a1_frame.py): x long, y short, z flange normal; "
           "rotation R and centre c map raw -> part as (p - c) @ R.T", "R": R.tolist(), "c": c.tolist(),
           "slab_half_mm": H, "folds": {}}
    for fid, f in FOLDS.items():
        ax = f["axis"]
        Gs, Ge = gold[f["gold"]]
        rng = (-170.0, 156.0) if ax == 0 else (-104.0, 112.0)
        stations = []
        for s0 in np.arange(rng[0], rng[1] + 0.01, 2.0):
            q, nq, idx = section(P, VN, sorted_by[ax], ax, s0, f["corner"], rad=16.0)
            r = analyse_station(q, nq, f)
            r["s"] = float(s0)
            r["n_pts"] = int(len(q))
            g2, t = gold_cross(Gs, Ge, ax, s0)
            r["gold_t"] = None if t is None else float(t)
            if g2 is not None and -0.001 <= t <= 1.001:
                g3 = np.zeros(3)
                cols = [1, 2] if ax == 0 else [0, 2]
                g3[cols] = g2
                g3[ax] = s0
                dn, _ = tree_all.query(g3)
                r["gold_d_nearest_scan_3d"] = float(dn)
                if r["ok"]:
                    r["gold"] = gold_offsets(r, g2, q, nq)
            stations.append(r)
        out["folds"][fid] = {"def": {k: (list(v) if isinstance(v, tuple) else v) for k, v in f.items()}, "stations": stations}
        ok = [s for s in stations if s["ok"]]
        g = [s for s in ok if "gold" in s]
        print(f"{fid} {f['gold']}: stations {len(stations)} ok {len(ok)} gold-ok {len(g)}")
    with open(OUT + "/a1_xsec_measurements.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
