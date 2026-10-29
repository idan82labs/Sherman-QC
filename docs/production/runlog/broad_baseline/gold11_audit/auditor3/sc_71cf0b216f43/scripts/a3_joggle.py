"""Measure the joggle (offset) in the long-side bottom flanges: per station a 3-segment
fit (flat - incline - flat) of the lower and upper skins; reports step height, incline angle,
breakpoint (bend) positions and the along-L extent. Writes a3_joggle.json."""
import sys
sys.dont_write_bytecode = True
import json
import math
import numpy as np
from a3_common import *
from a3_stations import SIDES, section

def fit3(u, h):
    """least-squares flat(h1) - incline - flat(h2) with breakpoints ua<ub (grid search)."""
    best = None
    us = np.sort(u)
    for ua in np.arange(us[0] + 2, us[-1] - 3, 0.25):
        for w in np.arange(1.0, 8.01, 0.25):
            ub = ua + w
            if ub > us[-1] - 1:
                break
            m1 = u <= ua; m3 = u >= ub; m2 = ~m1 & ~m3
            if m1.sum() < 5 or m3.sum() < 5:
                continue
            h1 = np.median(h[m1]); h2 = np.median(h[m3])
            pred = np.where(m1, h1, np.where(m3, h2, h1 + (u - ua) / (ub - ua) * (h2 - h1)))
            r = np.abs(h - pred)
            cost = np.sum(np.minimum(r, 0.5) ** 2)
            if best is None or cost < best[0]:
                best = (cost, ua, ub, h1, h2)
    return best

out = {}
for side in ("Sminus", "Splus"):
    rows = []
    for a0 in np.arange(-150, 166, 4.0):
        u, h, nu, nh = section(side, a0)
        z = (h < -15) & (u > 8) & (u < 34)
        lo = z & (nh < -0.6); up = z & (nh > 0.6)
        row = {"a": float(a0), "n_lower": int(lo.sum()), "n_upper": int(up.sum())}
        for nm, m in (("lower", lo), ("upper", up)):
            if m.sum() < 30:
                continue
            b = fit3(u[m], h[m])
            if b is None:
                continue
            _, ua, ub, h1, h2 = b
            ang = math.degrees(math.atan2(abs(h2 - h1), ub - ua))
            row[nm] = {"ua": ua, "ub": ub, "h1": float(h1), "h2": float(h2), "step": float(h2 - h1), "incline_deg": ang}
        rows.append(row)
    out[side] = rows
(OUT / "a3_joggle.json").write_text(json.dumps(out, indent=1))
for side, rows in out.items():
    good = [r for r in rows if "lower" in r and abs(r["lower"]["step"]) > 1.0]
    print(side, "stations", len(rows), "with joggle(lower skin, step>1mm)", len(good))
    if good:
        a = np.array([r["a"] for r in good])
        print("  L extent of joggle", a.min(), a.max())
        for k in ("ua", "ub", "step", "incline_deg", "h1", "h2"):
            v = np.array([r["lower"][k] for r in good])
            print("  lower", k, "median %.2f p10 %.2f p90 %.2f" % (np.median(v), *np.percentile(v, [10, 90])))
        goodu = [r for r in rows if "upper" in r and abs(r["upper"]["step"]) > 1.0]
        for k in ("ua", "ub", "step", "incline_deg"):
            v = np.array([r["upper"][k] for r in goodu])
            if len(v):
                print("  upper", k, "median %.2f p10 %.2f p90 %.2f" % (np.median(v), *np.percentile(v, [10, 90])))
    bad = [r["a"] for r in rows if not ("lower" in r and abs(r["lower"]["step"]) > 1.0)]
    print("  stations without a lower-skin joggle:", bad)
