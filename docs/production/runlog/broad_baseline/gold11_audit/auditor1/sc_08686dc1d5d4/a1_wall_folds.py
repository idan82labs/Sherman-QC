"""Station-by-station analysis of the plate->wall folds (gold M01 at v~-155, M02 at v~+155).

For each station u0 (plane u = u0, slab +-0.6 mm) in the part frame:
  fit plate top/bottom skins and wall outer/inner skins (2D lines in (v,w)),
  outer/inner virtual sharp, bisector, outer apex (outer-skin arc point nearest the bisector),
  inner apex, bend angle; gold M0x point at this u, offset from outer apex (along inward
  bisector / across) and distances to inner skin, arc centre etc.
Read-only on inputs; writes a1_wall_folds.json.
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from scipy.spatial import cKDTree
from a1_common import load_all, gold_lines, OUT, unit

xyz, nrm, faces = load_all()
F = np.load(OUT / "a1_frame.npz"); o, R = F["o"], F["R"]
p = (xyz - o) @ R.T
n = nrm @ R.T
g = gold_lines()
gp = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in g.items()}


def fit_line(P2):
    c = P2.mean(0)
    _, _, vt = np.linalg.svd(P2 - c, full_matrices=False)
    d = vt[0]
    res = (P2 - c) @ vt[1]
    return c, d, float(res.std())


def intersect(c1, d1, c2, d2):
    M = np.array([d1, -d2]).T
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    s, _ = np.linalg.solve(M, c2 - c1)
    return c1 + s * d1


def station(u0, side, half=0.6):
    """side=-1: wall at v~-155 (outer normal -v); side=+1: wall at v~+155 (outer normal +v)."""
    m = (np.abs(p[:, 0] - u0) < half) & (side * p[:, 1] > 125)
    P = p[m][:, 1:3]; N = n[m][:, 1:3]
    out = {"u": float(u0), "n_pts": int(m.sum())}
    if m.sum() < 50:
        out["status"] = "no_points"; return out, P, N
    vedge = side * 155.0
    # plate skins: |n_w| > 0.97, w > -4, side*v < 150
    top = (N[:, 1] > 0.97) & (P[:, 1] > -4) & (side * P[:, 0] > 132) & (side * P[:, 0] < 150)
    bot = (N[:, 1] < -0.97) & (P[:, 1] > -5) & (side * P[:, 0] > 132) & (side * P[:, 0] < 150)
    # wall skins: |n_v| > 0.97, -28 < w < -7
    wout = (side * N[:, 0] > 0.97) & (P[:, 1] < -7) & (P[:, 1] > -28)
    win = (side * N[:, 0] < -0.97) & (P[:, 1] < -7) & (P[:, 1] > -28)
    out.update(n_top=int(top.sum()), n_bot=int(bot.sum()), n_wout=int(wout.sum()), n_win=int(win.sum()))
    if min(top.sum(), wout.sum()) < 8:
        out["status"] = "no_fold"
        # describe what is there near the plate edge
        pe = (np.abs(N[:, 1]) > 0.9) & (P[:, 1] > -5)
        if pe.sum():
            out["plate_edge_v"] = float(side * np.max(side * P[pe, 0]))
            out["plate_top_w"] = float(np.median(P[(N[:, 1] > 0.9) & (P[:, 1] > -5), 1])) if ((N[:, 1] > 0.9) & (P[:, 1] > -5)).sum() else None
        return out, P, N
    ct, dt, rt = fit_line(P[top]); cw, dw, rw = fit_line(P[wout])
    Xo = intersect(ct, dt, cw, dw)
    res = {"rms_top": rt, "rms_wout": rw}
    Xi = None
    if bot.sum() >= 8 and win.sum() >= 8:
        cb, db, rb = fit_line(P[bot]); ci, di, ri = fit_line(P[win])
        Xi = intersect(cb, db, ci, di)
        res.update(rms_bot=rb, rms_win=ri)
    # directions pointing away from the corner along each flange
    a_plate = dt if (dt @ np.array([-side, 0.0])) > 0 else -dt     # toward the part interior (v -> 0)
    a_wall = dw if (dw @ np.array([0.0, -1.0])) > 0 else -dw         # downward
    ang_incl = np.degrees(np.arccos(np.clip(a_plate @ a_wall, -1, 1)))
    bis_in = unit(a_plate + a_wall)          # from outer sharp into the concave side
    # outer-skin arc points: within 6 mm of Xo, normal pointing outward (n . -bis_in > 0.3)
    near = np.linalg.norm(P - Xo, axis=1) < 6.0
    outward = (N @ (-bis_in)) > 0.2
    arc = near & outward
    apex = None
    if arc.sum() >= 3:
        Q = P[arc]
        rel = Q - Xo
        along = rel @ bis_in
        across = rel @ np.array([-bis_in[1], bis_in[0]])
        # apex = outer-skin point with |across| small, nearest the sharp
        cand = np.abs(across) < 0.35
        if cand.sum() >= 1:
            k = np.argmin(along[cand]); apex = Q[cand][k]
            # refine: median of candidates within 0.15 of min along
            sel = cand & (along < along[cand][k] + 0.15)
            apex = np.array([np.median(Q[sel][:, 0]), np.median(Q[sel][:, 1])])
    inner_apex = None
    inward = (N @ bis_in) > 0.2
    arci = near & inward & (np.linalg.norm(P - Xo, axis=1) < 6.0)
    if arci.sum() >= 3:
        Q = P[arci]; rel = Q - Xo
        along = rel @ bis_in; across = rel @ np.array([-bis_in[1], bis_in[0]])
        cand = (np.abs(across) < 0.35) & (along > 0.5)
        if cand.sum():
            k = np.argmin(along[cand]); inner_apex = Q[cand][k]
    out.update(status="fold", Xo=Xo.tolist(), Xi=None if Xi is None else Xi.tolist(),
               bis_in=bis_in.tolist(), included_deg=float(ang_incl), bend_deg=float(180 - ang_incl),
               apex=None if apex is None else apex.tolist(),
               inner_apex=None if inner_apex is None else inner_apex.tolist(), **res)
    if apex is not None:
        out["apex_minus_sharp_along"] = float((apex - Xo) @ bis_in)
    return out, P, N


def gold_at(key, u0):
    s, e = gp[key]
    t = (u0 - s[0]) / (e[0] - s[0])
    q = s + t * (e - s)
    return q, float(t)


results = {}
for key, side in (("M01", -1), ("M02", +1)):
    rows = []
    s, e = gp[key]
    for u0 in np.arange(-180, 181, 4.0):
        st, P, N = station(u0, side)
        q, t = gold_at(key, u0)
        st["gold_t"] = t
        st["gold_in_span"] = bool(0 <= t <= 1)
        st["gold_vw"] = [float(q[1]), float(q[2])]
        if st.get("status") == "fold" and st.get("apex") is not None:
            apex = np.array(st["apex"]); b = np.array(st["bis_in"])
            dvec = np.array(q[1:3]) - apex
            st["gold_minus_apex_along_bis_in"] = float(dvec @ b)
            st["gold_minus_apex_across"] = float(dvec @ np.array([-b[1], b[0]]))
            st["gold_apex_dist"] = float(np.linalg.norm(dvec))
            Xo = np.array(st["Xo"])
            st["gold_sharp_dist"] = float(np.linalg.norm(np.array(q[1:3]) - Xo))
            if st.get("Xi") is not None:
                st["gold_inner_sharp_dist"] = float(np.linalg.norm(np.array(q[1:3]) - np.array(st["Xi"])))
            if st.get("inner_apex") is not None:
                st["gold_inner_apex_dist"] = float(np.linalg.norm(np.array(q[1:3]) - np.array(st["inner_apex"])))
            # nearest scan point in this slab
            st["gold_nearest_scan_2d"] = float(np.min(np.linalg.norm(P - np.array(q[1:3]), axis=1)))
        rows.append(st)
    results[key] = rows

(OUT / "a1_wall_folds.json").write_text(json.dumps(results, indent=1))
for key in results:
    print("=====", key)
    for st in results[key]:
        if st.get("status") == "fold" and st.get("apex") is not None:
            print("u=%6.1f t=%5.2f bend=%5.1f apex(v,w)=(%.2f,%.2f) sharp-apex=%.2f gold(v,w)=(%.2f,%.2f) d_apex=%.2f along_in=%.2f across=%.2f d_innerapex=%s scan=%.2f" % (
                st["u"], st["gold_t"], st["bend_deg"], st["apex"][0], st["apex"][1], st["apex_minus_sharp_along"],
                st["gold_vw"][0], st["gold_vw"][1], st["gold_apex_dist"], st["gold_minus_apex_along_bis_in"],
                st["gold_minus_apex_across"], None if "gold_inner_apex_dist" not in st else round(st["gold_inner_apex_dist"], 2), st["gold_nearest_scan_2d"]))
        else:
            print("u=%6.1f t=%5.2f status=%s n=%d %s" % (st["u"], st["gold_t"], st.get("status"), st["n_pts"],
                  {k: st.get(k) for k in ("n_top", "n_wout", "plate_edge_v", "plate_top_w")}))
