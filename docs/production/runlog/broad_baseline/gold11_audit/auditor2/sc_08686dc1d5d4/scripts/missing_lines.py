import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
from scipy.optimize import least_squares
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
P = np.load(W / "plateframe.npy")
c, e1, e2, e3 = np.load(W / "pca.npy"); R = np.load(W / "R_plate.npy"); B = np.vstack([e1, e2, e3])
def to_world(q): return (np.asarray(q) @ R) @ B + c
def fitl(a, b):
    A = np.c_[np.ones(len(a)), a]; cc, *_ = np.linalg.lstsq(A, b, rcond=None); return cc
def pw3(params, x):
    x1, x2, z0, z1 = params
    return np.where(x < x1, z0, np.where(x > x2, z1, z0 + (z1 - z0) * (x - x1) / max(x2 - x1, 1e-3)))
res = {}
# ---------------- joggles: two bends per station from a flat-incline-flat fit to top and bottom skin envelopes
for side, lo, hi in (("LEFT", -174, -152), ("RIGHT", 152, 175)):
    rows = []
    for y0 in np.arange(-152, 149, 4.0):
        m = (np.abs(P[:, 1] - y0) < 0.4) & (P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 2] < 2) & (P[:, 2] > -10)
        Q = P[m]
        if len(Q) < 40: continue
        bins = np.arange(lo, hi + 0.3, 0.3); idx = np.digitize(Q[:, 0], bins)
        xc = 0.5 * (bins[:-1] + bins[1:])
        top = np.array([Q[idx == k, 2].max() if np.sum(idx == k) else np.nan for k in range(1, len(bins))])
        bot = np.array([Q[idx == k, 2].min() if np.sum(idx == k) else np.nan for k in range(1, len(bins))])
        out = {}
        for nm, prof in (("top", top), ("bot", bot)):
            ok = np.isfinite(prof)
            x = xc[ok]; z = prof[ok]
            if len(x) < 20: break
            zl = np.median(z[x < x.min() + 6]) if side == "RIGHT" else np.median(z[x > x.max() - 6])
            zh = np.median(z[x > x.max() - 6]) if side == "RIGHT" else np.median(z[x < x.min() + 6])
            xm0 = 163.5 if side == "RIGHT" else -163.0
            if side == "RIGHT":
                f = least_squares(lambda p: pw3(p, x) - z, [xm0 - 1, xm0 + 1, zl, zh], loss="soft_l1", f_scale=0.1)
            else:  # mirror: raised strip on the low-x side
                f = least_squares(lambda p: pw3(p, -x) - z, [-xm0 - 1, -xm0 + 1, zl, zh], loss="soft_l1", f_scale=0.1)
            x1, x2, z0, z1 = f.x
            if side == "LEFT": x1, x2 = -x1, -x2
            ang = math.degrees(math.atan2(abs(z1 - z0), abs(x2 - x1)))
            out[nm] = (x1, x2, z0, z1, ang)
        if len(out) < 2: continue
        rows.append({"y": float(y0), "top": out["top"], "bot": out["bot"]})
    # bend A = plate->flank (inner, nearer the plate centre), bend B = flank->raised strip (outer, nearer the edge)
    # convex skin: bend A convex on bottom skin, bend B convex on top skin (strip raised toward +z)
    A_pts, B_pts, angs, steps = [], [], [], []
    for r in rows:
        tx1, tx2, tz0, tz1, ta = r["top"]; bx1, bx2, bz0, bz1, ba = r["bot"]
        A_pts.append([bx1, r["y"], bz0]); B_pts.append([tx2, r["y"], tz1]); angs.append(0.5 * (ta + ba)); steps.append(0.5 * ((tz1 - tz0) + (bz1 - bz0)))
    A_pts = np.array(A_pts); B_pts = np.array(B_pts); angs = np.array(angs); steps = np.array(steps)
    def robust_line(Pts):
        # x = a + b y fit with outlier rejection; z = c + d y
        keep = np.ones(len(Pts), bool)
        for _ in range(3):
            cx = fitl(Pts[keep, 1], Pts[keep, 0]); r_ = Pts[:, 0] - (cx[0] + cx[1] * Pts[:, 1])
            s = 1.4826 * np.median(np.abs(r_[keep] - np.median(r_[keep]))) + 1e-6
            keep = np.abs(r_) < max(3 * s, 0.3)
        cz = fitl(Pts[keep, 1], Pts[keep, 2])
        return cx, cz, keep, r_
    for nm, Pts in (("A_plate_to_flank", A_pts), ("B_flank_to_strip", B_pts)):
        cx, cz, keep, r_ = robust_line(Pts)
        y0, y1 = -156.0, 150.0
        p0 = np.array([cx[0] + cx[1] * y0, y0, cz[0] + cz[1] * y0]); p1 = np.array([cx[0] + cx[1] * y1, y1, cz[0] + cz[1] * y1])
        print(f"{side} joggle bend {nm}: x = {cx[0]:.2f} + {cx[1]:.5f} y ; inliers {keep.sum()}/{len(keep)} ; resid MAD {np.median(np.abs(r_[keep])):.2f} mm ; plate-frame ends {p0.round(2)} -> {p1.round(2)}")
        res[f"{side}_JOGGLE_{nm}"] = {"plate_frame": [p0.tolist(), p1.tolist()], "world": [to_world(p0).tolist(), to_world(p1).tolist()], "length_mm": float(np.linalg.norm(p1 - p0))}
    print(f"{side} joggle: flank angle median {np.median(angs):.1f} deg (IQR {np.percentile(angs,25):.1f}-{np.percentile(angs,75):.1f}); step median {np.median(steps):.2f} mm; A-B spacing median {np.median(np.abs(B_pts[:,0]-A_pts[:,0])):.2f} mm")
    res[f"{side}_JOGGLE_meta"] = {"flank_angle_deg_med": float(np.median(angs)), "step_mm_med": float(np.median(steps))}
# ---------------- lips: wall->lip fold outer apex per x station
for side, yw, s in (("M01", 152.86, 1), ("M02", -158.66, -1)):
    ap, angs = [], []
    for x0 in np.arange(-64, 65, 4.0):
        m = (np.abs(P[:, 0] - x0) < 0.5) & (np.abs(P[:, 1] - yw) < 20) & (P[:, 2] > 12)
        Q = P[m]; u = s * (yw - Q[:, 1]); z = Q[:, 2]
        zl = np.median(z[(u > 5) & (u < 10) & (z > 28)]) if np.any((u > 5) & (u < 10) & (z > 28)) else None
        if zl is None: continue
        wm = (u > -3) & (u < 3) & (z > zl - 18) & (z < zl - 6)
        lm = (u > 5) & (u < 10) & (np.abs(z - zl) < 2)
        if wm.sum() < 15 or lm.sum() < 15: continue
        uc = np.median(u[wm]); zc = np.median(z[lm])
        wo = wm & (u < uc); lt = lm & (z > zc)
        if wo.sum() < 5 or lt.sum() < 5: continue
        cw = fitl(z[wo], u[wo]); cl = fitl(u[lt], z[lt])     # u = cw0 + cw1 z ; z = cl0 + cl1 u
        zz = (cl[0] + cl[1] * cw[0]) / (1 - cl[1] * cw[1]); uu = cw[0] + cw[1] * zz
        Xo = np.array([uu, zz])
        q2 = np.c_[u, z]; d = np.linalg.norm(q2 - Xo, axis=1)
        apex = q2[np.argsort(d)[:3]].mean(0)
        tw = np.array([cw[1], 1.0]); tw /= np.linalg.norm(tw); tl = np.array([1.0, cl[1]]); tl /= np.linalg.norm(tl)
        bend = 180 - math.degrees(math.acos(np.clip((-tw) @ tl, -1, 1)))
        angs.append(bend)
        ap.append([x0, yw - s * apex[0], apex[1]])
    ap = np.array(ap)
    m_ = ap.mean(0); u_, s_, vt = np.linalg.svd(ap - m_); d_ = vt[0] if vt[0][0] > 0 else -vt[0]
    rr = np.linalg.norm((ap - m_) - np.outer((ap - m_) @ d_, d_), axis=1)
    x0, x1 = -66.0, 66.0
    p0 = m_ + d_ * ((x0 - m_[0]) / d_[0]); p1 = m_ + d_ * ((x1 - m_[0]) / d_[0])
    print(f"{side} LIP fold (wall->return lip): stations {len(ap)} bend angle median {np.median(angs):.1f} deg (min {min(angs):.1f} max {max(angs):.1f}); apex line resid rms {math.sqrt((rr**2).mean()):.3f}; plate-frame ends {p0.round(2)} -> {p1.round(2)}")
    res[f"{side}_WALL_LIP"] = {"plate_frame": [p0.tolist(), p1.tolist()], "world": [to_world(p0).tolist(), to_world(p1).tolist()], "length_mm": float(np.linalg.norm(p1 - p0)), "bend_deg_med": float(np.median(angs))}
# true wall folds (from apex_lines_match)
am = json.load(open(W / "apex_lines_match.json"))
for side in ("M01", "M02"):
    res[f"{side}_WALL_FOLD_TRUE_APEX"] = {"world": am[side]["apex_line_world"]}
json.dump(res, open(W / "true_fold_lines.json", "w"), indent=1)
for k, v in res.items():
    if "world" in v: print(k, np.round(v["world"][0], 2), np.round(v["world"][1], 2))
