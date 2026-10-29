"""Auditor-1 independent cross-section measurement for every gold line of sc_d1732de18023.

For stations every STEP mm along each gold line (extended EXT mm past both ends) a +-SLAB mm slab of
raw scan vertices perpendicular to the gold direction is projected to 2D.  The two flanges are found
from the projected vertex normals (two dominant orientations), each flange's two skins are fitted as
lines, and from them:
  X_o  outer virtual sharp (outer skins intersect)      X_i inner virtual sharp
  A_o  outer apex = outer-skin vertices on the bisector (median)   A_i inner apex likewise
  fold present at a station = flanges + both skins + outer arc vertices on the bisector.
Gold point at the station = gold line point at that axial position.
Writes a1_xsec_measurements.json; read-only on every input.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from a1_common import OUT, gold_lines, load_all, perp_basis, unit  # noqa: E402

STEP = 2.0
SLAB = 0.6
RAD = 22.0
EXT = 40.0
ANG_TOL = 8.0


def fit_station(q, m):
    """q: (n,2) points, m: (n,2) unit normals (projected).  Returns dict or None."""
    if len(q) < 60:
        return None
    phi = np.degrees(np.arctan2(m[:, 1], m[:, 0])) % 180.0
    h, e = np.histogram(phi, bins=180, range=(0, 180))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    p1 = int(np.argmax(hs))
    d = np.abs(((np.arange(180) - p1) + 90) % 180 - 90)
    hs2 = np.where(d >= 30, hs, -1)
    p2 = int(np.argmax(hs2))
    if hs2[p2] < 15:
        return None
    out = {}
    fl = []
    for pk in (p1, p2):
        dphi = np.abs(((phi - (pk + 0.5)) + 90) % 180 - 90)
        sel = dphi < ANG_TOL
        # refine orientation as circular mean of doubled angle
        ang = np.radians(phi[sel] * 2)
        mean = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2 % 180
        nvec = np.array([math.cos(math.radians(mean)), math.sin(math.radians(mean))])
        s = q[sel] @ nvec
        fl.append({"sel": sel, "n": nvec, "s": s, "ang": mean})
    # two skins per flange: split offsets by largest gap / 2-means
    for f in fl:
        s = f["s"]
        if len(s) < 20:
            return None
        # skins = densest offset peak + densest peak 0.8..2.2 mm from it (free-edge faces of the
        # other flange share the orientation but sit far away and are ignored)
        lo, hi = s.min() - 0.5, s.max() + 0.5
        bins = np.arange(lo, hi + 0.1, 0.1)
        hh, ee = np.histogram(s, bins=bins)
        hh = np.convolve(hh, np.ones(3), "same")
        cen = (ee[:-1] + ee[1:]) / 2
        k1 = int(np.argmax(hh))
        dist = np.abs(cen - cen[k1])
        cand = np.where((dist > 0.8) & (dist < 2.2), hh, -1)
        k2 = int(np.argmax(cand))
        if cand[k2] < 5:
            return None
        peaks = sorted([cen[k1], cen[k2]])
        refined = []
        for pk in peaks:
            near = np.abs(s - pk) < 0.4
            refined.append(float(np.median(s[near])))
        f["skins"] = refined
        f["sel"] = f["sel"].copy()
        on_skin = (np.abs(s - refined[0]) < 0.45) | (np.abs(s - refined[1]) < 0.45)
        idx_sel = np.where(f["sel"])[0]
        f["sel"][idx_sel[~on_skin]] = False
        f["s"] = s[on_skin]
    f1, f2 = fl
    th1 = f1["skins"][1] - f1["skins"][0]
    th2 = f2["skins"][1] - f2["skins"][0]
    # mid-surface intersection = rough corner
    def inter(n1, s1, n2, s2):
        M = np.array([n1, n2])
        if abs(np.linalg.det(M)) < 1e-6:
            return None
        return np.linalg.solve(M, np.array([s1, s2]))
    corner = inter(f1["n"], np.mean(f1["skins"]), f2["n"], np.mean(f2["skins"]))
    if corner is None:
        return None
    # flange directions away from corner
    for f in fl:
        tdir = np.array([-f["n"][1], f["n"][0]])
        proj = (q[f["sel"]] - corner) @ tdir
        if np.median(proj) < 0:
            tdir = -tdir
        f["t"] = tdir
        f["len"] = float(np.percentile(proj * np.sign(np.median(proj)), 98))
    # orient normals toward the concave side (toward the other flange's extension)
    for f, g in ((f1, f2), (f2, f1)):
        if f["n"] @ g["t"] < 0:
            f["n"] = -f["n"]
            f["s"] = -f["s"]
            f["skins"] = sorted([-f["skins"][0], -f["skins"][1]])
    # now inner skin = larger s (toward concave), outer = smaller s
    Xo = inter(f1["n"], f1["skins"][0], f2["n"], f2["skins"][0])
    Xi = inter(f1["n"], f1["skins"][1], f2["n"], f2["skins"][1])
    if Xo is None or Xi is None:
        return None
    b_in = unit(f1["t"] + f2["t"])
    across = np.array([-b_in[1], b_in[0]])
    bend_angle = 180.0 - math.degrees(math.acos(np.clip(f1["t"] @ f2["t"], -1, 1)))
    rel = q - Xo
    al = rel @ b_in
    ac = rel @ across
    near = np.abs(ac) < 0.35
    outer_pts = near & (al > -0.5) & (al < 4.5) & (m @ (-b_in) > 0.6)
    inner_pts = near & (al > 0.5) & (al < 7.0) & (m @ b_in > 0.6)
    res = {"Xo": Xo, "Xi": Xi, "b_in": b_in, "across": across, "bend_angle": bend_angle,
           "th": [float(th1), float(th2)], "flange_len": [f1["len"], f2["len"]],
           "flange_t": [f1["t"], f2["t"]], "flange_n": [f1["n"], f2["n"]],
           "skins": [f1["skins"], f2["skins"]], "n_outer_bis": int(outer_pts.sum()),
           "n_inner_bis": int(inner_pts.sum())}
    res["Ao"] = Xo + b_in * float(np.median(al[outer_pts])) if outer_pts.sum() >= 2 else None
    res["Ai"] = Xo + b_in * float(np.median(al[inner_pts])) if inner_pts.sum() >= 2 else None
    return res


def main():
    xyz, nrm, _ = load_all()
    G = gold_lines()
    allres = {}
    for bid, (a, b) in G.items():
        d = unit(b - a)
        L = float(np.linalg.norm(b - a))
        u, w = perp_basis(d)
        rel = xyz - a
        t = rel @ d
        r2 = rel - np.outer(t, d)
        rad = np.linalg.norm(r2, axis=1)
        cyl = (rad < RAD + 5) & (t > -EXT - 2) & (t < L + EXT + 2)
        idx = np.where(cyl)[0]
        tt = t[idx]
        Q = np.stack([rel[idx] @ u, rel[idx] @ w], 1)
        M3 = nrm[idx]
        M = np.stack([M3 @ u, M3 @ w], 1)
        Mn = np.linalg.norm(M, axis=1)
        stations = []
        for s0 in np.arange(-EXT, L + EXT + 1e-6, STEP):
            sel = (np.abs(tt - s0) < SLAB) & (Mn > 0.5)
            q = Q[sel]
            mm = M[sel] / Mn[sel][:, None]
            g = np.zeros(2)  # gold point is the origin of the 2D frame (on the infinite gold line)
            # restrict to RAD around the gold point
            keep = np.linalg.norm(q - g, axis=1) < RAD
            r = fit_station(q[keep], mm[keep])
            row = {"t": float(s0), "in_gold_span": bool(0 <= s0 <= L), "n": int(keep.sum())}
            if r is not None and r["Ao"] is not None and 60 < r["bend_angle"] < 120 \
                    and min(r["th"]) > 0.8 and max(r["th"]) < 2.2 and min(r["flange_len"]) > 3.0:
                Ao, Xo, Xi, b_in, ac = r["Ao"], r["Xo"], r["Xi"], r["b_in"], r["across"]
                gv = g - Ao
                row.update({
                    "fold": True, "bend_angle": r["bend_angle"], "th": r["th"],
                    "flange_len": r["flange_len"], "n_outer_bis": r["n_outer_bis"],
                    "d_apex": float(np.linalg.norm(gv)),
                    "gold_minus_apex_along_concave_mm": float(gv @ b_in),
                    "gold_minus_apex_across_mm": float(gv @ ac),
                    "d_outer_sharp": float(np.linalg.norm(g - Xo)),
                    "d_inner_sharp": float(np.linalg.norm(g - Xi)),
                    "d_inner_apex": None if r["Ai"] is None else float(np.linalg.norm(g - r["Ai"])),
                    "apex_minus_outer_sharp": float(np.linalg.norm(Ao - Xo)),
                    "inner_apex_minus_outer_apex": None if r["Ai"] is None else float(np.linalg.norm(r["Ai"] - Ao)),
                    # position of gold along each flange measured from the outer sharp, and offset from skins
                    "gold_on_flange": [
                        {"along_from_Xo": float((g - Xo) @ r["flange_t"][k]),
                         "s_minus_outer_skin": float(g @ r["flange_n"][k] - r["skins"][k][0]),
                         "s_minus_inner_skin": float(g @ r["flange_n"][k] - r["skins"][k][1])}
                        for k in (0, 1)],
                    "apex_xyz": (a + d * s0 + u * Ao[0] + w * Ao[1]).tolist(),
                    "outer_sharp_xyz": (a + d * s0 + u * Xo[0] + w * Xo[1]).tolist(),
                    "inner_sharp_xyz": (a + d * s0 + u * Xi[0] + w * Xi[1]).tolist(),
                    "gold_xyz": (a + d * s0).tolist(),
                })
            else:
                row["fold"] = False
                if r is not None:
                    row["why"] = {"ang": r["bend_angle"], "th": r["th"], "len": r["flange_len"],
                                  "Ao": r["Ao"] is not None}
            stations.append(row)
        allres[bid] = {"length_mm": L, "dir": d.tolist(), "start": a.tolist(), "end": b.tolist(),
                       "stations": stations}
        fs = [s for s in stations if s.get("fold")]
        ins = [s for s in fs if s["in_gold_span"]]
        print(bid, "L", round(L, 1), "stations", len(stations), "fold", len(fs), "fold&in-span", len(ins))
        if ins:
            dap = np.array([s["d_apex"] for s in ins])
            print("   d_apex med %.2f min %.2f max %.2f" % (np.median(dap), dap.min(), dap.max()))
    (OUT / "a1_xsec_measurements.json").write_text(json.dumps(allres, indent=1, default=float))
    print("wrote", OUT / "a1_xsec_measurements.json")


if __name__ == "__main__":
    main()
