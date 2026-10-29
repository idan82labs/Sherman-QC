"""Auditor-1 (sc_cec08dfca9ab): robust joggle measurement with fixed level windows around the flank centre.

Per x station (+-0.6 mm y-z slab): per-skin z(y) from 0.25 mm bins (top = 90th pct, bottom = 10th pct of z in the
bin, bins with >=3 pts), levels = medians in windows [yc-9, yc-3] and [yc+3, yc+9] (tilt-corrected by a joint
line fit of both windows' slope), step = level difference, flank angle = max |slope| of the 3-bin-smoothed
tilt-corrected profile within yc+-2.5, flank centre = where the profile crosses mid-level.
"""
import json
import numpy as np
from a1frame import load, to_part, from_part, W

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
CFG = {"base_joggle": dict(yc=-148.6, zlo=-14.0, zhi=-7.0), "lip_joggle": dict(yc=95.3, zlo=23.5, zhi=30.5)}


def profile(m, y0, y1, pct, bw=0.25):
    y, z = P[m, 1], P[m, 2]
    e = np.arange(y0, y1 + bw, bw)
    i = np.digitize(y, e) - 1
    yc = 0.5 * (e[:-1] + e[1:])
    zp = np.full(len(yc), np.nan)
    for k in range(len(yc)):
        zz = z[i == k]
        if len(zz) >= 3:
            zp[k] = np.percentile(zz, pct)
    return yc, zp


def measure(yc_bins, zp, yc):
    ok = ~np.isnan(zp)
    wa = ok & (yc_bins > yc - 9) & (yc_bins < yc - 3)
    wb = ok & (yc_bins > yc + 3) & (yc_bins < yc + 9)
    if wa.sum() < 8 or wb.sum() < 8:
        return None
    # common tilt from both windows (levels assumed parallel)
    ya, za, yb, zb = yc_bins[wa], zp[wa], yc_bins[wb], zp[wb]
    A = np.c_[np.r_[ya, yb], np.r_[np.ones_like(ya), np.zeros_like(yb)], np.r_[np.zeros_like(ya), np.ones_like(yb)]]
    coef, *_ = np.linalg.lstsq(A, np.r_[za, zb], rcond=None)
    tilt, la, lb = coef
    step = lb - la
    w = ok & (np.abs(yc_bins - yc) < 4.0)
    yy = yc_bins[w]; zz = zp[w] - tilt * yy
    zs = np.convolve(zz, np.ones(3) / 3, mode="same")
    sl = np.gradient(zs, yy)
    core = np.abs(yy - yc) < 2.5
    k = np.argmax(np.abs(sl) * core)
    mid = 0.5 * (la + lb)
    cross = yy[np.argmin(np.abs(zs - mid) + 10 * ~core)]
    return dict(step=float(step), flank_deg=float(np.degrees(np.arctan(abs(sl[k])))), flank_centre_y=float(cross),
                tilt_deg=float(np.degrees(np.arctan(tilt))), level_a=float(la), level_b=float(lb))


out = {}
for name, cf in CFG.items():
    rows = []
    for x0 in np.arange(-156.0, 156.1, 4.0):
        m = (np.abs(P[:, 0] - x0) < 0.6) & (np.abs(P[:, 1] - cf["yc"]) < 12) & (P[:, 2] > cf["zlo"]) & (P[:, 2] < cf["zhi"])
        if m.sum() < 40:
            rows.append(dict(x=float(x0), n=int(m.sum()))); continue
        yb, zt = profile(m, cf["yc"] - 12, cf["yc"] + 12, 90)
        _, zb = profile(m, cf["yc"] - 12, cf["yc"] + 12, 10)
        rows.append(dict(x=float(x0), n=int(m.sum()), top=measure(yb, zt, cf["yc"]), bot=measure(yb, zb, cf["yc"])))
    pres = [r for r in rows if r.get("top") and abs(r["top"]["step"]) >= 1.0]
    agg = dict(n_present=len(pres), n_total=len(rows),
               x_extent=[min(r["x"] for r in pres), max(r["x"] for r in pres)] if pres else None)
    for sk in ("top", "bot"):
        vv = [r[sk] for r in pres if r.get(sk)]
        for key in ("step", "flank_deg", "flank_centre_y"):
            v = np.array([q[key] for q in vv])
            agg[f"{sk}_{key}"] = [round(float(np.median(v)), 2), round(float(np.percentile(v, 10)), 2), round(float(np.percentile(v, 90)), 2)]
    missing = [r["x"] for r in rows if not (r.get("top") and abs(r["top"]["step"]) >= 1.0)]
    agg["stations_without_step"] = missing
    if pres:
        yv = agg["top_flank_centre_y"][0]
        zv = float(np.median([0.5 * (r["top"]["level_a"] + r["top"]["level_b"]) + 0 for r in pres]))
        agg["flank_line_part"] = [[agg["x_extent"][0], yv], [agg["x_extent"][1], yv]]
    out[name] = dict(summary=agg, rows=rows)
    print("==", name, json.dumps(agg))
json.dump(out, open(W + "joggle_measure2_a1.json", "w"), indent=1, default=float)
