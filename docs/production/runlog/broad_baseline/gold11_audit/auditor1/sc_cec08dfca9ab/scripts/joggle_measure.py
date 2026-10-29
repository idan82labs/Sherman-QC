"""Auditor-1 (sc_cec08dfca9ab): measure the two joggles (base near y~-149, top lip near y~95) in the part frame.

For x stations, take a +-0.6 mm y-z slab, build upper/lower skin profiles z(y) by binning, then locate the
flank (|slope| > tan 10 deg) on each skin, the max flank angle, the step height and the two knee positions.
Knee = where the flank leaves/enters a level (|slope| crosses tan 10 deg). Reports per-station values and the
x-extent over which the joggle is present (step >= 1 mm). Read-only on all inputs.
"""
import json
import numpy as np
from a1frame import load, to_part, from_part, W

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)


def skin_profiles(m, ylo, yhi, binw=0.2):
    y, z = P[m, 1], P[m, 2]
    edges = np.arange(ylo, yhi + binw, binw)
    idx = np.digitize(y, edges) - 1
    top = np.full(len(edges) - 1, np.nan)
    bot = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        zz = z[idx == i]
        if len(zz) >= 2:
            top[i] = np.max(zz)
            bot[i] = np.min(zz)
    yc = 0.5 * (edges[:-1] + edges[1:])
    return yc, top, bot


def analyse(yc, prof, lvl_lo_y, lvl_hi_y):
    ok = ~np.isnan(prof)
    if ok.sum() < 20:
        return None
    yy, zz = yc[ok], prof[ok]
    # smooth with 5-bin running median then mean
    zs = np.array([np.median(zz[max(0, i - 2):i + 3]) for i in range(len(zz))])
    zs = np.convolve(zs, np.ones(5) / 5, mode="same")
    sl = np.gradient(zs, yy)
    core = (yy > yy[0] + 1.0) & (yy < yy[-1] - 1.0)
    steep = core & (np.abs(sl) > np.tan(np.radians(10)))
    if steep.sum() < 2:
        return dict(step=float(np.nan))
    ys = yy[steep]
    # level heights away from the flank
    a = zz[(yy < ys.min() - 2.0)]
    b = zz[(yy > ys.max() + 2.0)]
    step = float(np.median(b) - np.median(a)) if len(a) > 3 and len(b) > 3 else float("nan")
    k = np.argmax(np.abs(sl) * core)
    return dict(knee_lo_y=float(ys.min()), knee_hi_y=float(ys.max()), flank_run=float(ys.max() - ys.min()),
                max_flank_deg=float(np.degrees(np.arctan(np.abs(sl[k])))), flank_mid_y=float(yy[k]),
                level_a=float(np.median(a)) if len(a) else None, level_b=float(np.median(b)) if len(b) else None,
                step=step)


out = {}
for name, (ylo, yhi, zlo, zhi) in {"base_joggle": (-158.0, -138.0, -14.0, -7.0),
                                   "lip_joggle": (86.0, 104.0, 23.5, 30.5)}.items():
    rows = []
    for x0 in np.arange(-156.0, 156.1, 4.0):
        m = (np.abs(P[:, 0] - x0) < 0.6) & (P[:, 1] > ylo) & (P[:, 1] < yhi) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
        if m.sum() < 40:
            rows.append(dict(x=float(x0), n=int(m.sum()), status="few_points"))
            continue
        yc, top, bot = skin_profiles(m, ylo, yhi)
        rt, rb = analyse(yc, top, ylo, yhi), analyse(yc, bot, ylo, yhi)
        rows.append(dict(x=float(x0), n=int(m.sum()), top=rt, bot=rb))
    good = [r for r in rows if r.get("top") and np.isfinite(r["top"].get("step", np.nan)) and abs(r["top"]["step"]) >= 1.0]
    xs = [r["x"] for r in good]
    agg = {}
    if good:
        for key in ("knee_lo_y", "knee_hi_y", "flank_run", "max_flank_deg", "step", "flank_mid_y"):
            v = np.array([r["top"][key] for r in good])
            agg["top_" + key] = [float(np.median(v)), float(v.min()), float(v.max())]
        vb = [r for r in good if r.get("bot") and np.isfinite(r["bot"].get("step", np.nan))]
        for key in ("max_flank_deg", "step", "knee_lo_y", "knee_hi_y"):
            v = np.array([r["bot"][key] for r in vb])
            if len(v):
                agg["bot_" + key] = [float(np.median(v)), float(v.min()), float(v.max())]
        agg["x_extent_present"] = [float(min(xs)), float(max(xs))]
        agg["n_stations_present"] = len(good)
        agg["n_stations_total"] = len(rows)
        # raw-frame endpoints of the two knee lines (top skin, median y, median z of levels) for reference
        ymid = agg["top_flank_mid_y"][0]
        zmid = float(np.median([0.5 * (r["top"]["level_a"] + r["top"]["level_b"]) for r in good]))
        agg["flank_mid_line_raw"] = [from_part(np.array([min(xs), ymid, zmid]), R, c).tolist(),
                                     from_part(np.array([max(xs), ymid, zmid]), R, c).tolist()]
    out[name] = dict(summary=agg, rows=rows)
    print("==", name, json.dumps(agg, default=float))
    for r in rows[::8]:
        t = r.get("top") or {}
        print(f"   x={r['x']:7.1f} n={r['n']:5d} step={t.get('step', float('nan')):5.2f} flank={t.get('max_flank_deg', float('nan')):5.1f}deg "
              f"knees y=[{t.get('knee_lo_y', float('nan')):7.2f},{t.get('knee_hi_y', float('nan')):7.2f}]")
json.dump(out, open(W + "joggle_measure_a1.json", "w"), indent=1, default=float)
