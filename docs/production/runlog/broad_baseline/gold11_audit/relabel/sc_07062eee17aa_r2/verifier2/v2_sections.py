"""Verifier2 (independent) per-line section analysis for sc_07062eee17aa r2.

For every labelled line: 1 mm (0.5 mm ears) stations along the line, extended
+-25 mm (+-8 mm ears) beyond its ends. In each station slab, outer-skin legs are
refit locally (2D lines), giving a local virtual sharp, local bisector and the
measured outer apex (first skin point along the local bisector from the sharp =
arc midline). The label point is compared with that measured apex.
"""
import json, sys, os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LINES = os.path.join(HERE, "..", "lines.json")
SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01_SCAN_1.ply"


def load_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline()
            hdr += line
            if line.strip() == b"end_header":
                break
        nv = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        v = np.frombuffer(f.read(nv * 24), dtype="<f4").reshape(nv, 6)
    return v[:, :3].astype(np.float64), v[:, 3:].astype(np.float64)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_line2d(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def station(tree, X, P0, d, u, w, sharp0, dir_a, dir_b, n_a, n_b, T, leg_len, h, R):
    idx = tree.query_ball_point(P0, np.sqrt(R * R + h * h))
    if len(idx) == 0:
        return None
    Q = X[idx] - P0
    s = Q @ d
    m = np.abs(s) < h
    Q = Q[m]
    q2 = np.c_[Q @ w, Q @ u]  # 2D (w,u)
    out = {"pts": q2, "n": len(q2)}
    legs = {}
    for nm, dr, nn in (("a", dir_a, n_a), ("b", dir_b, n_b)):
        rel = q2 - sharp0
        sa = rel @ dr
        o = rel @ nn
        strip = (sa > T + 0.8) & (sa < T + 0.8 + leg_len) & (np.abs(o) < 4.5)
        if strip.sum() < 8:
            legs[nm] = None
            continue
        omax = np.percentile(o[strip], 98)
        sel = strip & (o > omax - 0.7)
        Pl = q2[sel]
        if len(Pl) < 8:
            legs[nm] = None
            continue
        c, v = fit_line2d(Pl)
        for _ in range(2):
            nrm = np.array([-v[1], v[0]])
            r = (Pl - c) @ nrm
            k = np.abs(r) < max(0.25, 2.5 * np.std(r))
            if k.sum() < 6:
                break
            c, v = fit_line2d(Pl[k])
        if v @ dr < 0:
            v = -v
        legs[nm] = (c, v, int(sel.sum()))
    out["legs"] = legs
    if legs["a"] is None or legs["b"] is None:
        return out
    (ca, va, _), (cb, vb, _) = legs["a"], legs["b"]
    # intersection ca + x va = cb + y vb
    A = np.c_[va, -vb]
    try:
        xy = np.linalg.solve(A, cb - ca)
    except np.linalg.LinAlgError:
        return out
    S = ca + xy[0] * va
    bis = unit(va + vb)
    lat = np.array([-bis[1], bis[0]])
    rel = q2 - S
    pr = rel @ bis
    la = rel @ lat
    cand = (np.abs(la) < 0.35) & (pr > -1.0) & (pr < 4.0)
    out["sharp_loc"] = S
    out["bis_loc"] = bis
    out["angle_incl"] = float(np.degrees(np.arccos(np.clip(va @ vb, -1, 1))))
    if cand.sum() < 2:
        return out
    depth = np.percentile(pr[cand], 10)
    Apex = S + depth * bis
    out["apex_loc"] = Apex
    out["apex_depth"] = float(depth)
    L = np.zeros(2)
    dv = L - Apex
    out["off_bis"] = float(dv @ bis)  # + = label deeper into material than outer skin; - = toward sharp/air
    out["off_lat"] = float(dv @ lat)
    out["label_depth"] = float((L - S) @ bis)
    out["dist_nearest"] = float(np.min(np.linalg.norm(q2, axis=1)))
    return out


def main():
    cfg = json.load(open(LINES))
    X, N = load_ply(SCAN)
    tree = cKDTree(X)
    results = {}
    for b in cfg["bends"]:
        bid = b["bend_id"]
        ear = bid in ("B5", "B6")
        Ls = np.array(b["label_line"]["start"]); Le = np.array(b["label_line"]["end"])
        Ss = np.array(b["sharp"]["start"]); Se = np.array(b["sharp"]["end"])
        Llen = np.linalg.norm(Le - Ls)
        d = unit(Le - Ls)
        u = np.array(b["bisector"]); u = unit(u - (u @ d) * d)
        w = np.cross(d, u)
        def p2(v):
            v = np.asarray(v); v = v - (v @ d) * d
            return unit(np.array([v @ w, v @ u]))
        dir_a = p2(b["side_dirs"]["plane_a"]); dir_b = p2(b["side_dirs"]["plane_b"])
        def nrm(da, db):
            n = np.array([-da[1], da[0]])
            return n if n @ db < 0 else -n
        n_a = nrm(dir_a, dir_b); n_b = nrm(dir_b, dir_a)
        incl = np.degrees(np.arccos(np.clip(dir_a @ dir_b, -1, 1)))
        T = 2.3 / np.tan(np.radians(incl / 2))  # tangent length for outer R=2.3
        leg_len = 4.0 if ear else 8.0
        h = 0.4 if ear else 0.6
        R = 9.0 if ear else 14.0
        step = 0.5 if ear else 1.0
        ext = 8.0 if ear else 25.0
        ts = np.arange(-ext, Llen + ext + 1e-9, step)
        rows = []
        keep = {}
        for t in ts:
            P0 = Ls + t * d
            Sh = Ss + (t / Llen) * (Se - Ss)
            sh2 = np.array([(Sh - P0) @ w, (Sh - P0) @ u])
            r = station(tree, X, P0, d, u, w, sh2, dir_a, dir_b, n_a, n_b, T, leg_len, h, R)
            row = {"t": float(t), "frac": float(t / Llen)}
            if r is not None:
                la, lb = r.get("legs", {}).get("a"), r.get("legs", {}).get("b")
                row["leg_a_n"] = 0 if la is None else la[2]
                row["leg_b_n"] = 0 if lb is None else lb[2]
                for k in ("off_bis", "off_lat", "apex_depth", "label_depth", "dist_nearest", "angle_incl"):
                    if k in r:
                        row[k] = r[k]
                if "apex_loc" in r:
                    A = r["apex_loc"]
                    row["apex_xyz"] = (P0 + A[0] * w + A[1] * u).tolist()
                row["label_sharp_depth_from_label"] = float(sh2 @ np.array([0, 1]))
                row["label_sharp_2d"] = sh2.tolist()
            rows.append(row)
            for f in (0.25, 0.5, 0.75):
                if abs(t - f * Llen) < step / 2 + 1e-9:
                    keep[f] = (r, sh2)
        # exact 25/50/75 stations
        for f in (0.25, 0.5, 0.75):
            t = f * Llen
            P0 = Ls + t * d
            Sh = Ss + f * (Se - Ss)
            sh2 = np.array([(Sh - P0) @ w, (Sh - P0) @ u])
            keep[f] = (station(tree, X, P0, d, u, w, sh2, dir_a, dir_b, n_a, n_b, T, leg_len, h, R), sh2)
        results[bid] = {"L": Llen, "rows": rows, "label_incl": b["included_angle_deg"]}
        # figure: 3 sections
        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        for j, f in enumerate((0.25, 0.5, 0.75)):
            r, sh2 = keep[f]
            for row_i, zoom in ((0, 9.0 if not ear else 6.0), (1, 2.5)):
                ax = axs[row_i, j]
                if r is None:
                    ax.set_title(f"{bid} {int(f*100)}% no points"); continue
                q = r["pts"]
                ax.scatter(q[:, 0], q[:, 1], s=3 if row_i == 0 else 8, c="0.45", label="scan pts (slab)")
                ax.plot(*sh2, "kx", ms=12, mew=2, label="label virtual sharp")
                if "sharp_loc" in r:
                    ax.plot(*r["sharp_loc"], "+", color="tab:orange", ms=14, mew=2, label="my local sharp")
                    for nm in ("a", "b"):
                        c, v, _ = r["legs"][nm]
                        tt = np.linspace(-12, 12, 2)
                        ax.plot(c[0] + tt * v[0], c[1] + tt * v[1], "--", color="tab:orange", lw=0.8)
                if "apex_loc" in r:
                    ax.plot(*r["apex_loc"], "o", mfc="none", mec="tab:green", ms=14, mew=2, label="my measured outer apex")
                ax.plot(0, 0, "o", color="tab:red", ms=7, label="label point")
                c0 = np.zeros(2)
                ax.set_xlim(c0[0] - zoom, c0[0] + zoom); ax.set_ylim(c0[1] - zoom, c0[1] + zoom)
                ax.set_aspect("equal"); ax.grid(alpha=0.3)
                ttl = f"{bid} @{int(f*100)}% (t={f*Llen:.1f} mm)"
                if "off_bis" in r:
                    ttl += f"\noff_bis={r['off_bis']:+.2f} (+=into material) lat={r['off_lat']:+.2f} incl={r['angle_incl']:.1f}"
                ax.set_title(ttl, fontsize=9)
                if row_i == 0 and j == 0:
                    ax.legend(fontsize=7, loc="lower left")
                ax.set_xlabel("w (mm, across fold)"); ax.set_ylabel("u (mm, label bisector, + into material)")
        fig.suptitle(f"verifier2 {bid}: perpendicular sections at 25/50/75% (top: wide, bottom: 2.5 mm zoom)")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"v2_sections_{bid}.png"), dpi=90)
        plt.close(fig)
    # save
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError
    json.dump(results, open(os.path.join(HERE, "v2_stations.json"), "w"), default=conv)
    print("done")


if __name__ == "__main__":
    main()
