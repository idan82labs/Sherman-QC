"""Per-station outer-apex fit for the two plate->wall folds (F_M01 side, F_M02 side), in the
plate frame (x along walls, y toward the M01 wall, z = plate normal, walls rise +z).
Gold position is expressed relative to the outer apex: along the inward bisector (+ = concave
side) and across it, plus distances to outer/inner virtual sharp, inner arc apex, arc centre."""
import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
from scipy.spatial import cKDTree
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy")
G = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.load(open(W / "gold_plateframe.json")).items()}
tree = cKDTree(P)

def fit_line(a, b):  # least squares b = c0 + c1 a
    A = np.c_[np.ones(len(a)), a]
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c

def split2(v):
    """split values into two clusters along v (largest gap near middle)."""
    o = np.sort(v)
    if len(o) < 10:
        return None
    gaps = np.diff(o)
    lo, hi = int(0.1 * len(o)), int(0.9 * len(o))
    k = lo + int(np.argmax(gaps[lo:hi]))
    if gaps[k] < 0.25:
        return None
    return 0.5 * (o[k] + o[k + 1])

def station(side, x0, slab=0.6):
    s = 1.0 if side == "M01" else -1.0
    yw = 152.86 if side == "M01" else -158.66
    m = np.abs(P[:, 0] - x0) < slab
    Q = P[m]
    u = s * (yw - Q[:, 1])           # u>0: toward plate interior; wall at u~0
    z = Q[:, 2]
    # plate band near the fold
    zp_guess = np.median(z[(u > 8) & (u < 20) & (z < 0)]) if np.any((u > 8) & (u < 20) & (z < 0)) else -5.0
    pm = (u > 6) & (u < 20) & (np.abs(z - zp_guess) < 2.5)
    wm = (u > -4) & (u < 4) & (z > zp_guess + 7) & (z < zp_guess + 24)
    if pm.sum() < 20 or wm.sum() < 20:
        return {"x": x0, "ok": False, "why": "no_plate_or_wall", "n_plate": int(pm.sum()), "n_wall": int(wm.sum())}
    zc = split2(z[pm]); uc = split2(u[wm])
    if zc is None or uc is None:
        return {"x": x0, "ok": False, "why": "single_skin"}
    pb = pm & (z < zc); pt = pm & (z >= zc)          # plate bottom (outer) / top (inner) skin
    wo = wm & (u < uc); wi = wm & (u >= uc)          # wall outer face (u more negative) / inner face
    cpb = fit_line(u[pb], z[pb]); cpt = fit_line(u[pt], z[pt])   # z = c0 + c1 u
    cwo = fit_line(z[wo], u[wo]); cwi = fit_line(z[wi], u[wi])   # u = c0 + c1 z
    def inter(cp, cw):
        # z = a0 + a1 u ; u = b0 + b1 z
        a0, a1 = cp; b0, b1 = cw
        zz = (a0 + a1 * b0) / (1 - a1 * b1); uu = b0 + b1 * zz
        return np.array([uu, zz])
    Xo = inter(cpb, cwo); Xi = inter(cpt, cwi)
    tA = np.array([1.0, cpb[1]]); tA /= np.linalg.norm(tA)          # plate direction (into plate)
    tB = np.array([cwo[1], 1.0]); tB /= np.linalg.norm(tB)          # wall direction (up)
    g = tA + tB; g /= np.linalg.norm(g)                              # inward bisector (concave side)
    incl = math.degrees(math.acos(np.clip(tA @ tB, -1, 1)))
    bend = 180 - incl
    q2 = np.c_[u, z]
    dX = np.linalg.norm(q2 - Xo, axis=1)
    near = dX < 8
    ia = np.argsort(dX[near])[:3]
    apex = q2[near][ia].mean(0)
    delta = float(np.linalg.norm(apex - Xo))
    half = math.radians(incl) / 2
    Ro = delta / (1 / math.sin(half) - 1)
    C = Xo + g * (Ro / math.sin(half))
    thick_p = abs(cpt[0] - cpb[0]) / math.sqrt(1 + cpb[1] ** 2)
    thick_w = abs(cwi[0] - cwo[0]) / math.sqrt(1 + cwo[1] ** 2)
    # inner arc apex: first scan point from Xi along -g? inner arc lies between Xi and C on the bisector
    rel = q2 - Xi
    al = rel @ g; ac = np.abs(rel @ np.array([-g[1], g[0]]))
    cand = (ac < 0.35) & (al > -0.5) & (al < Ro + 2) & (np.linalg.norm(q2 - C, axis=1) < Ro + 0.2)
    inner_apex = None
    if cand.any():
        # inner skin points on the bisector: the one closest to C (largest al that is still material) ~ inner arc midpoint
        cc = np.where(cand)[0]
        dC = np.linalg.norm(q2[cc] - C, axis=1)
        # inner arc radius ~ Ro - t; choose the point nearest C among the bisector band
        inner_apex = q2[cc[np.argmin(dC)]]
    # gold point at this x
    a, b = G[side]
    t = (x0 - a[0]) / (b[0] - a[0])
    gp = a + t * (b - a)
    gu = s * (yw - gp[1]); gz = gp[2]
    gv = np.array([gu, gz])
    d_apex_vec = gv - apex
    along = float(d_apex_vec @ g); across = float(d_apex_vec @ np.array([-g[1], g[0]]))
    dn, _ = tree.query(gp)
    res = {"x": x0, "ok": True, "t_gold": float(t), "bend_deg": bend, "R_outer": Ro, "thick_plate": thick_p, "thick_wall": thick_w,
           "gold_uz": gv.tolist(), "apex_uz": apex.tolist(), "Xo_uz": Xo.tolist(), "Xi_uz": Xi.tolist(), "C_uz": C.tolist(),
           "g": g.tolist(), "d_apex": float(np.linalg.norm(d_apex_vec)), "gold_minus_apex_along_bisector": along,
           "gold_minus_apex_across": across, "d_outer_sharp": float(np.linalg.norm(gv - Xo)), "d_inner_sharp": float(np.linalg.norm(gv - Xi)),
           "d_arc_centre": float(np.linalg.norm(gv - C)), "d_inner_apex": None if inner_apex is None else float(np.linalg.norm(gv - inner_apex)),
           "inner_apex_uz": None if inner_apex is None else inner_apex.tolist(), "gold_nearest_scan_mm": float(dn),
           "apex_xyz_plate": [x0, yw - s * apex[0], apex[1]]}
    return res

out = {}
for side in ("M01", "M02"):
    rows = []
    for x0 in np.arange(-98, 101, 4.0):
        rows.append(station(side, float(x0)))
    out[side] = rows
    ok = [r for r in rows if r["ok"]]
    print(f"== {side}: stations {len(rows)} ok {len(ok)}")
    for r in rows[::5]:
        if r["ok"]:
            print("x=%6.1f bend=%.1f Ro=%.2f tp=%.2f tw=%.2f | gold-apex %.2f (along %.2f across %.2f) | ->Xo %.2f ->Xi %.2f ->C %.2f ->innerApex %s | gold->scan %.2f" % (
                r["x"], r["bend_deg"], r["R_outer"], r["thick_plate"], r["thick_wall"], r["d_apex"], r["gold_minus_apex_along_bisector"],
                r["gold_minus_apex_across"], r["d_outer_sharp"], r["d_inner_sharp"], r["d_arc_centre"],
                None if r["d_inner_apex"] is None else round(r["d_inner_apex"], 2), r["gold_nearest_scan_mm"]))
        else:
            print("x=%6.1f FAIL %s" % (r["x"], r.get("why")))
    d = np.array([r["d_apex"] for r in ok]); al = np.array([r["gold_minus_apex_along_bisector"] for r in ok]); ac = np.array([r["gold_minus_apex_across"] for r in ok])
    print("  d_apex min/med/max %.2f %.2f %.2f ; along med %.2f range %.2f..%.2f ; across med %.2f range %.2f..%.2f ; frac d_apex<=4: %.2f" % (
        d.min(), np.median(d), d.max(), np.median(al), al.min(), al.max(), np.median(ac), ac.min(), ac.max(), (d <= 4).mean()))
    for k in ("d_outer_sharp", "d_inner_sharp", "d_arc_centre", "d_inner_apex", "R_outer", "bend_deg", "thick_plate", "thick_wall", "gold_nearest_scan_mm"):
        v = np.array([r[k] for r in ok if r[k] is not None])
        print("  %s med %.2f (min %.2f max %.2f)" % (k, np.median(v), v.min(), v.max()))
json.dump(out, open(W / "fold_stations.json", "w"), indent=1)
