"""Measure the two unlabelled joggles (offset steps) of sc_cec08dfca9ab as 2 bends each.

Per station (fixed part-X, 1 mm slab) the profile is resampled in 0.2 mm Y bins; top skin =
max Z, bottom skin = min Z per bin. The joggle has an upper plate, an inclined web and a lower
plate. For each skin the web slope (max |dZ/dY| over a 1 mm baseline) gives the joggle bend
angle; bend J1 (convex on the TOP skin of the upper plate end) and bend J2 (convex on the
BOTTOM skin at the lower plate start) are located at the mid-point between the plate and the
web tangent lines on their convex (outer) skin, i.e. the outer-skin arc midpoint.
Writes joggle_lines.json (part + raw frame) and a3_joggle_fit.png.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math

import numpy as np

from a3_common import OUT, load_points

JOGGLES = {
    # name: (Y window, Z window, upper plate side ('neg' = upper plate at smaller Y))
    "bottom_strip_joggle": ((-128.0, -110.0), (-3.0, 5.0), "neg"),
    "top_flange_joggle": ((116.0, 134.0), (35.0, 44.0), "pos"),
}


def profile(y, z, lo, hi, step=0.2):
    bins = np.arange(lo, hi + step, step)
    idx = np.digitize(y, bins) - 1
    top = np.full(len(bins) - 1, np.nan)
    bot = np.full(len(bins) - 1, np.nan)
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() >= 2:
            top[k] = z[m].max()
            bot[k] = z[m].min()
    yc = 0.5 * (bins[:-1] + bins[1:])
    return yc, top, bot


def line_fit(y, z):
    A = np.c_[y, np.ones_like(y)]
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    return sol  # slope, intercept


def analyse(yc, zs, upper_side, web_lo, web_hi):
    """zs: one skin profile. Fit upper plate, lower plate and web lines; return intersections."""
    ok = np.isfinite(zs)
    y, z = yc[ok], zs[ok]
    if len(y) < 30:
        return None
    # web = region of steep slope
    dz = np.gradient(z, y)
    k = np.argmax(np.abs(np.convolve(dz, np.ones(5) / 5, mode="same")))
    yw = y[k]
    web = (y > yw - 0.6) & (y < yw + 0.6)
    if upper_side == "neg":
        up = y < yw - 2.2
        lo = y > yw + 2.2
    else:
        up = y > yw + 2.2
        lo = y < yw - 2.2
    if web.sum() < 3 or up.sum() < 5 or lo.sum() < 5:
        return None
    su = line_fit(y[up], z[up])
    sl = line_fit(y[lo], z[lo])
    sw = line_fit(y[web], z[web])
    # intersections (virtual sharps on this skin)
    def inter(a, b):
        if abs(a[0] - b[0]) < 1e-9:
            return None
        yy = (b[1] - a[1]) / (a[0] - b[0])
        return yy, a[0] * yy + a[1]
    i_up = inter(su, sw)
    i_lo = inter(sl, sw)
    ang_up = math.degrees(abs(math.atan(sw[0]) - math.atan(su[0])))
    ang_lo = math.degrees(abs(math.atan(sw[0]) - math.atan(sl[0])))
    return {"y_web": float(yw), "web_slope_deg": float(math.degrees(math.atan(abs(sw[0])))),
            "sharp_upper": i_up, "sharp_lower": i_lo, "bend_upper_deg": ang_up, "bend_lower_deg": ang_lo,
            "plate_offset": float((su[0] * yw + su[1]) - (sl[0] * yw + sl[1]))}


def nearest_on_skin(yc, zs, sharp, upper):
    """outer-skin apex ~ skin point nearest to the virtual sharp (small-angle arc midpoint)."""
    ok = np.isfinite(zs)
    y, z = yc[ok], zs[ok]
    d = np.hypot(y - sharp[0], z - sharp[1])
    k = int(np.argmin(d))
    return float(y[k]), float(z[k]), float(d[k])


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = json.loads((OUT / "part_frame.json").read_text())
    c0 = np.array(fr["origin"])
    M = np.array([fr["X"], fr["Y"], fr["Z"]])
    P = (load_points() - c0) @ M.T
    result = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=90)
    for row, (name, ((ylo, yhi), (zlo, zhi), upper)) in enumerate(JOGGLES.items()):
        sel = (P[:, 1] > ylo) & (P[:, 1] < yhi) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
        Q = P[sel]
        stations = []
        for x in np.arange(-150, 151, 5.0):
            m = np.abs(Q[:, 0] - x) < 0.5
            if m.sum() < 60:
                continue
            yc, top, bot = profile(Q[m, 1], Q[m, 2], ylo, yhi)
            rt = analyse(yc, top, upper, None, None)
            rb = analyse(yc, bot, upper, None, None)
            if rt is None or rb is None or rt["sharp_upper"] is None or rb["sharp_lower"] is None:
                continue
            # J1: convex on top skin at the upper-plate end; J2: convex on bottom skin at lower-plate start
            a1 = nearest_on_skin(yc, top, rt["sharp_upper"], True)
            a2 = nearest_on_skin(yc, bot, rb["sharp_lower"], False)
            stations.append({"x": float(x), "J1_apex_yz": a1[:2], "J2_apex_yz": a2[:2],
                             "J1_sharp_yz": list(rt["sharp_upper"]), "J2_sharp_yz": list(rb["sharp_lower"]),
                             "web_slope_top_deg": rt["web_slope_deg"], "web_slope_bot_deg": rb["web_slope_deg"],
                             "J1_bend_deg": rt["bend_upper_deg"], "J2_bend_deg": rb["bend_lower_deg"],
                             "offset_top_mm": rt["plate_offset"], "offset_bot_mm": rb["plate_offset"]})
            if x in (-100.0, 0.0, 100.0):
                ax = axes[row, [-100.0, 0.0, 100.0].index(x)]
                ax.scatter(Q[m, 1], Q[m, 2], s=2, c="#444")
                ax.plot(*a1[:2], "o", mfc="none", mec="r", ms=9, mew=2, label="J1 apex (top skin)")
                ax.plot(*a2[:2], "o", mfc="none", mec="b", ms=9, mew=2, label="J2 apex (bottom skin)")
                ax.plot(*rt["sharp_upper"], "x", c="r", ms=8)
                ax.plot(*rb["sharp_lower"], "x", c="b", ms=8)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)
                ax.set_xlim(ylo, yhi)
                ax.set_title(f"{name} X={x:.0f}: web slope {rt['web_slope_deg']:.0f}/{rb['web_slope_deg']:.0f} deg, "
                             f"offset {rt['plate_offset']:.2f} mm", fontsize=8)
                ax.legend(fontsize=7)
        if not stations:
            result[name] = {"stations": 0}
            continue
        out = {"stations": len(stations)}
        for key in ("J1", "J2"):
            pts = np.array([[s["x"], *s[f"{key}_apex_yz"]] for s in stations])
            cA = pts.mean(0)
            _u, _s, vt = np.linalg.svd(pts - cA, full_matrices=False)
            dA = vt[0] if vt[0][0] > 0 else -vt[0]
            tt = (pts - cA) @ dA
            res = pts - cA - np.outer(tt, dA)
            st, en = cA + tt.min() * dA, cA + tt.max() * dA
            out[key] = {"start_part": st.tolist(), "end_part": en.tolist(), "rms_mm": float(np.sqrt((res ** 2).sum(1).mean())),
                        "start_raw": (st @ M + c0).tolist(), "end_raw": (en @ M + c0).tolist(),
                        "length_mm": float(np.linalg.norm(en - st)),
                        "bend_deg_median": float(np.median([s[f"{key}_bend_deg"] for s in stations]))}
        out["web_slope_deg_median"] = float(np.median([0.5 * (s["web_slope_top_deg"] + s["web_slope_bot_deg"]) for s in stations]))
        out["offset_mm_median"] = float(np.median([0.5 * (s["offset_top_mm"] + s["offset_bot_mm"]) for s in stations]))
        out["J1_J2_separation_mm"] = float(np.median([math.hypot(s["J1_apex_yz"][0] - s["J2_apex_yz"][0], s["J1_apex_yz"][1] - s["J2_apex_yz"][1]) for s in stations]))
        out["x_range_part"] = [stations[0]["x"], stations[-1]["x"]]
        out["per_station"] = stations
        result[name] = out
        print(name, json.dumps({k: v for k, v in out.items() if k != "per_station"}, indent=1))
    fig.tight_layout()
    fig.savefig(OUT / "a3_joggle_fit.png")
    (OUT / "joggle_lines.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
