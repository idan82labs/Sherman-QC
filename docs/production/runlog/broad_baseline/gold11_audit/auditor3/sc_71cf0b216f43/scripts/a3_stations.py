"""Per-station corner fits for the 8 wall bends (top: plate<->wall, bottom: wall<->bottom flange)
of sc_71cf0b216f43 and gold-line offsets from the OUTER APEX (outer-skin arc midline).

Section frame per side: u = inward distance from the wall's outer face reference, h = height.
Outer/inner skins are separated with the scan mesh normals (outward from material, checked).
Writes a3_stations.json (per bend per station) under auditor3/sc_71cf0b216f43/.
"""
import sys
sys.dont_write_bytecode = True
import json
import math
import numpy as np
from scipy.spatial import cKDTree
from a3_common import *

Q = np.load(SCR / "Q.npy"); NQ = np.load(SCR / "NQ.npy"); C0 = np.load(SCR / "center.npy")
GOLD = {b: (to_frame(a, C0), to_frame(e, C0)) for b, a, e in load_gold()}
TREE = cKDTree(Q)

SIDES = {
    # name: (axis index, perp index, u = sgn*(coord - ref), station range)
    "Sminus": dict(ax=0, pe=1, ref=-99.08, sgn=+1, rng=(-152, 167)),
    "Splus": dict(ax=0, pe=1, ref=106.83, sgn=-1, rng=(-152, 167)),
    "Lminus": dict(ax=1, pe=0, ref=-148.95, sgn=+1, rng=(-103, 111)),
    "Lplus": dict(ax=1, pe=0, ref=163.68, sgn=-1, rng=(-103, 111)),
}
# human gold line -> (side, corner) it sits beside (from the part-frame coordinates)
GOLD_SIDE = {"M01": ("Lminus", "bot"), "M03": ("Lminus", "top"), "M02": ("Lplus", "bot"), "M04": ("Lplus", "top"),
             "M05": ("Sminus", "top"), "M06": ("Sminus", "bot"), "M07": ("Splus", "top"), "M08": ("Splus", "bot")}
SLAB = 0.6


def fit_line(x, y, it=3, k=2.5):
    """robust y = a x + b"""
    m = np.ones(len(x), bool)
    a = b = None
    for _ in range(it):
        if m.sum() < 8:
            return None
        a, b = np.polyfit(x[m], y[m], 1)
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r[m])) + 1e-3
        m = np.abs(r) < k * s + 0.05
    return a, b, int(m.sum())


def section(side, a0):
    sd = SIDES[side]
    m = np.abs(Q[:, sd["ax"]] - a0) < SLAB
    P = Q[m]; Nn = NQ[m]
    u = sd["sgn"] * (P[:, sd["pe"]] - sd["ref"])
    h = P[:, 2]
    nu = sd["sgn"] * Nn[:, sd["pe"]]
    nh = Nn[:, 2]
    keep = (u > -6) & (u < 45)
    return u[keep], h[keep], nu[keep], nh[keep]


def corner_fit(u, h, nu, nh, corner, flange_u):
    """corner 'top' (wall <-> top plate at h~+20) or 'bot' (wall <-> bottom flange at h~-20)."""
    res = {"ok": False}
    hs = 1.0 if corner == "top" else -1.0          # direction from wall middle toward this corner
    # wall skins: u = a h + b, fitted on the straight middle part of the wall
    wz = (np.abs(h) < 11) & (u > -3) & (u < 5)
    wo = wz & (nu < -0.85); wi = wz & (nu > 0.85)
    fo = fit_line(h[wo], u[wo]) if wo.sum() > 10 else None
    fi = fit_line(h[wi], u[wi]) if wi.sum() > 10 else None
    # plate skins: h = c u + d on u in flange_u
    pz = (u > flange_u[0]) & (u < flange_u[1]) & (hs * h > 14)
    if corner == "top":
        po = pz & (nh > 0.85); pi = pz & (nh < -0.85)
    else:
        po = pz & (nh < -0.85); pi = pz & (nh > 0.85)
    go = fit_line(u[po], h[po]) if po.sum() > 10 else None
    gi = fit_line(u[pi], h[pi]) if pi.sum() > 10 else None
    res.update({"n_wall_outer": int(wo.sum()), "n_wall_inner": int(wi.sum()), "n_plate_outer": int(po.sum()),
                "n_plate_inner": int(pi.sum())})
    if fo is None or go is None:
        res["why"] = "missing_outer_skin"
        return res
    # outer virtual sharp: u = a h + b and h = c u + d
    a, b = fo[0], fo[1]; c, d = go[0], go[1]
    hX = (c * b + d) / (1 - c * a); uX = a * hX + b
    Xo = np.array([uX, hX])
    # flange directions away from the corner
    wdir = unit([-a * hs, -hs])          # along wall, away from corner (toward the wall middle)
    pdir = unit([1.0, c])                # along plate, toward +u (inward)
    g = unit(wdir + pdir)               # inward bisector (concave side)
    incl = math.degrees(math.acos(np.clip(wdir @ pdir, -1, 1)))
    gp = np.array([-g[1], g[0]])
    rel = np.c_[u, h] - Xo
    al = rel @ g; ac = rel @ gp
    nrm = np.c_[nu, nh]
    near = (np.abs(ac) < 0.35) & (al > -1.0) & (al < 6.0)
    outer_pts = near & ((nrm @ g) < -0.5)
    if outer_pts.sum() < 2:
        res.update({"why": "no_outer_arc_points", "Xo": Xo.tolist()})
        return res
    vals = np.sort(al[outer_pts])
    delta = float(np.median(vals[:5]))
    apex = Xo + g * delta
    half = math.radians(incl) / 2
    k = 1 / math.sin(half) - 1
    Ro = delta / k if k > 1e-6 else float("nan")
    out = {"ok": True, "Xo": Xo, "g": g, "gp": gp, "incl_deg": incl, "bend_deg": 180 - incl, "apex": apex,
           "delta_apex_sharp": delta, "R_outer": Ro, "wall_outer": (a, b), "plate_outer": (c, d)}
    # inner skins / inner sharp / inner apex
    if fi is not None and gi is not None:
        a2, b2 = fi[0], fi[1]; c2, d2 = gi[0], gi[1]
        hXi = (c2 * b2 + d2) / (1 - c2 * a2); uXi = a2 * hXi + b2
        Xi = np.array([uXi, hXi])
        t_wall = abs((b2 - b)) * math.cos(math.atan(a))
        t_plate = abs(d2 - d) * math.cos(math.atan(c))
        out.update({"Xi": Xi, "t_wall": t_wall, "t_plate": t_plate})
        inner_pts = near & ((nrm @ g) > 0.5) & (al > delta + 0.4)
        if inner_pts.sum() >= 2:
            v2 = np.sort(al[inner_pts])
            out["inner_apex"] = Xo + g * float(np.median(v2[:5]))
            out["inner_apex_n"] = int(inner_pts.sum())
    out["n_near_bisector"] = int(near.sum())
    res.update(out)
    return res


def gold_point_at(bid, side, a0):
    sd = SIDES[side]
    p0, p1 = GOLD[bid]
    dd = p1 - p0
    if abs(dd[sd["ax"]]) < 1e-9:
        return None
    t = (a0 - p0[sd["ax"]]) / dd[sd["ax"]]
    if t < 0 or t > 1:
        return None
    p = p0 + t * dd
    u = sd["sgn"] * (p[sd["pe"]] - sd["ref"])
    return t, p, np.array([u, p[2]])


def main():
    out = {"scan_id": "sc_71cf0b216f43", "frame": {"L": L0.tolist(), "S": S0.tolist(), "H": H0.tolist(), "center": C0.tolist()},
           "sides": {k: {kk: vv for kk, vv in v.items()} for k, v in SIDES.items()}, "bends": {}}
    for side in SIDES:
        for corner in ("top", "bot"):
            flange_u = (6.0, 25.0) if corner == "top" else ((5.0, 9.0) if side.startswith("L") else (5.0, 16.0))
            rows = []
            lo, hi = SIDES[side]["rng"]
            for a0 in np.arange(lo, hi + 0.01, 2.0):
                u, h, nu, nh = section(side, a0)
                if len(u) < 50:
                    rows.append({"a": float(a0), "ok": False, "why": "empty"})
                    continue
                r = corner_fit(u, h, nu, nh, corner, flange_u)
                row = {"a": float(a0), "ok": r["ok"], "why": r.get("why")}
                for k in ("n_wall_outer", "n_wall_inner", "n_plate_outer", "n_plate_inner"):
                    row[k] = r.get(k)
                if r["ok"]:
                    for k in ("Xo", "apex", "Xi", "inner_apex", "g"):
                        if k in r:
                            row[k] = [float(x) for x in r[k]]
                    for k in ("bend_deg", "delta_apex_sharp", "R_outer", "t_wall", "t_plate"):
                        if k in r:
                            row[k] = float(r[k])
                    # gold lines beside this corner
                    for bid, (gs, gc) in GOLD_SIDE.items():
                        if gs != side or gc != corner:
                            continue
                        gp = gold_point_at(bid, side, a0)
                        if gp is None:
                            continue
                        t, p3, q = gp
                        v = q - r["apex"]
                        nd, ni = TREE.query(p3)
                        nn_normal = NQ[ni]
                        row["gold"] = {"bend_id": bid, "t": float(t), "uh": q.tolist(),
                                       "d_apex": float(np.linalg.norm(v)),
                                       "along_bisector_concave": float(v @ r["g"]),
                                       "across_toward_plate": float(v @ unit(np.array([1.0, r['plate_outer'][0]]) - np.array([-r['wall_outer'][0] * (1 if corner == 'top' else -1), -(1 if corner == 'top' else -1)]))),
                                       "d_outer_sharp": float(np.linalg.norm(q - r["Xo"])),
                                       "d_inner_sharp": float(np.linalg.norm(q - r["Xi"])) if "Xi" in r else None,
                                       "d_inner_apex": float(np.linalg.norm(q - r["inner_apex"])) if "inner_apex" in r else None,
                                       "d_nearest_scan": float(nd),
                                       "nearest_scan_normal_uh": [float(SIDES[side]["sgn"] * nn_normal[SIDES[side]["pe"]]), float(nn_normal[2])]}
                rows.append(row)
            out["bends"][f"{side}_{corner}"] = rows
    def enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)
    (OUT / "a3_stations.json").write_text(json.dumps(out, indent=1, default=enc))
    print("wrote", OUT / "a3_stations.json")


if __name__ == "__main__":
    main()
