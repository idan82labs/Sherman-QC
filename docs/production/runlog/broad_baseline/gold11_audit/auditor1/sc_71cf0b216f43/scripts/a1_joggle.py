"""Auditor-1 (sc_71cf0b216f43): characterise the joggles in the two long-wall top flanges.

Per x station (2 mm pitch) in the y-z section (slab +-0.5 mm): top-skin (normal +z) profile z_top(y) and
bottom-skin profile z_bot(y) across the joggle; upper / lower levels, drop, web slope angle (from the
steepest part of the top skin), and the two fold (virtual sharp) positions = intersections of the
web line with the upper / lower top-skin levels. Fold extent = stations where a drop of >= 1 mm with
a web slope >= 10 deg is seen. Output a1_joggles.json (+ raw-frame endpoints of the 4 joggle fold lines).
"""
import json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get

P, VN, F, G, R, c = get()
order = np.argsort(P[:, 0]); keys = P[order, 0]
out = {"joggles": {}}
for name, (ylo, yhi), sgn in (("front_top_flange", (-90.0, -62.0), +1), ("back_top_flange", (72.0, 100.0), -1)):
    rows = []
    for x0 in np.arange(-166.0, 152.0, 2.0):
        a, b = np.searchsorted(keys, [x0 - 0.5, x0 + 0.5]); idx = order[a:b]
        q = P[idx]; n = VN[idx]
        m = (q[:, 1] > ylo) & (q[:, 1] < yhi) & (q[:, 2] > 16.5) & (q[:, 2] < 23.5)
        q, n = q[m], n[m]
        top = n[:, 2] > 0.3  # top skin (+z facing incl. web top)
        if top.sum() < 20:
            rows.append({"x": float(x0), "ok": False}); continue
        yt, zt = q[top, 1], q[top, 2]
        # profile on 0.25 mm bins in y
        bins = np.arange(ylo, yhi + 0.25, 0.25)
        k = np.digitize(yt, bins)
        prof = np.array([[bins[i - 1] + 0.125, np.median(zt[k == i])] for i in np.unique(k) if (k == i).sum() >= 1 and 0 < i < len(bins)])
        if len(prof) < 20:
            rows.append({"x": float(x0), "ok": False}); continue
        # wall side = toward the long wall (front: smaller y; back: larger y)
        ys, zs = prof[:, 0], prof[:, 1]
        zu = np.median(zs[zs > np.percentile(zs, 70)]); zl = np.median(zs[zs < np.percentile(zs, 30)])
        drop = zu - zl
        # slope: gradient over 0.75 mm windows
        o = np.argsort(ys); ys, zs = ys[o], zs[o]
        grads = []
        for i in range(len(ys)):
            w = np.abs(ys - ys[i]) < 0.6
            if w.sum() >= 3:
                A = np.c_[ys[w], np.ones(w.sum())]; g = np.linalg.lstsq(A, zs[w], rcond=None)[0][0]
                grads.append((ys[i], zs[i], g))
        grads = np.array(grads)
        if drop < 1.0 or len(grads) == 0:
            rows.append({"x": float(x0), "ok": False, "drop": float(drop)}); continue
        mid = 0.5 * (zu + zl)
        near = np.abs(grads[:, 1] - mid) < 0.35 * drop
        if near.sum() == 0:
            rows.append({"x": float(x0), "ok": False, "drop": float(drop)}); continue
        gmax = grads[near][np.argmax(np.abs(grads[near, 2]))]
        slope = math.degrees(math.atan(abs(gmax[2])))
        # web line through the mid-drop point with the steepest slope
        ym = np.median(grads[near, 0])
        gg = np.median(grads[near, 2][np.abs(grads[near, 2]) > 0.5 * abs(gmax[2])])
        y_upper = ym + (zu - mid) / gg  # where web meets the upper level
        y_lower = ym + (zl - mid) / gg
        rows.append({"x": float(x0), "ok": bool(slope >= 10), "z_upper_top": float(zu), "z_lower_top": float(zl),
                     "drop": float(drop), "web_slope_deg": float(slope), "y_mid": float(ym),
                     "y_fold_upper": float(y_upper), "y_fold_lower": float(y_lower)})
    ok = [r for r in rows if r.get("ok")]
    xs = np.array([r["x"] for r in ok])
    summ = {}
    for key in ("z_upper_top", "z_lower_top", "drop", "web_slope_deg", "y_mid", "y_fold_upper", "y_fold_lower"):
        v = np.array([r[key] for r in ok]); summ[key] = [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    # fold lines in part frame (virtual sharps on the TOP skin levels), extent = ok station range
    x0, x1 = float(xs.min()), float(xs.max())
    lines = {}
    for fold, ykey, zkey in (("J_upper", "y_fold_upper", "z_upper_top"), ("J_lower", "y_fold_lower", "z_lower_top")):
        # straight-line fit y(x), z(x)
        yv = np.array([r[ykey] for r in ok]); zv = np.array([r[zkey] for r in ok])
        py = np.polyfit(xs, yv, 1); pz = np.polyfit(xs, zv, 1)
        e0 = np.array([x0, np.polyval(py, x0), np.polyval(pz, x0)]); e1 = np.array([x1, np.polyval(py, x1), np.polyval(pz, x1)])
        lines[fold] = {"part": [e0.tolist(), e1.tolist()], "raw": [(e0 @ R + c).tolist(), (e1 @ R + c).tolist()],
                       "y_resid_mm": float(np.std(yv - np.polyval(py, xs)))}
    gaps = [[float(a), float(b)] for a, b in zip(xs[:-1], xs[1:]) if b - a > 2.5]
    out["joggles"][name] = {"stations_ok": len(ok), "x_extent": [x0, x1], "gaps_x": gaps, "summary_median_p5_p95": summ,
                            "fold_lines_topskin_virtual_sharp": lines, "stations": rows}
    print(name, "ok stations", len(ok), "x extent", x0, x1, "gaps", gaps)
    for k2, v in summ.items(): print("   ", k2, [round(t, 2) for t in v])
json.dump(out, open(OUT + "/a1_joggles.json", "w"), indent=1)
