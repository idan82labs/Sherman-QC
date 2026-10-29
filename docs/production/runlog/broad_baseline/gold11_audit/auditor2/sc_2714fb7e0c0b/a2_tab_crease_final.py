"""Tab-end crease angle by direct per-skin read-off at fixed u stations (SUPERSEDES a2_tab_crease*.json, whose
~46 deg values are biased by splitting two skins on a 3 mm, sparse segment). Writes a2_tab_crease_final.json."""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from a2_common import load, OUT

q, R, O, raw = load()
out = {}
for name, vlo, vhi in (("B09_tab_end_crease", 24, 34), ("B10_tab_end_crease", 200, 212)):
    per = []
    for wl, wh in ((3, 5), (6, 8), (9, 11), (12, 14)):
        mids = []
        for u0 in np.arange(-66, -52.4, 1.0):
            m = (q[:, 2] > wl) & (q[:, 2] < wh) & (np.abs(q[:, 0] - u0) < 0.3) & (q[:, 1] > vlo) & (q[:, 1] < vhi)
            v = np.sort(q[m, 1])
            if len(v) < 4:
                continue
            k = int(np.argmax(np.diff(v)))
            if k + 1 >= len(v):
                continue
            mids.append((u0, 0.5 * (np.median(v[:k + 1]) + np.median(v[k + 1:]))))
        sl = [(u, m) for u, m in mids if -56.5 <= u <= -53.5]
        U = np.array([u for u, _ in sl]); M = np.array([m for _, m in sl])
        slope = float(np.polyfit(U, M, 1)[0])
        per.append({"w_band": [wl, wh], "slant_slope": slope, "crease_angle_deg": math.degrees(math.atan(abs(slope)))})
    a = [p["crease_angle_deg"] for p in per]
    out[name] = {"per_height": per, "angle_deg_median": float(np.median(a)), "angle_deg_range": [float(min(a)), float(max(a))],
                 "crease_u_mm": -58.5, "straight_until_u_mm": -60.0, "tab_height_w_mm": [1.7, 14.5],
                 "note": "vertical (w-axis) crease near the +u end of the window tab; end wing ~5-6 mm long; "
                         "below w~1.7 a relief notch separates tab and wing at u -59..-56.3; wing base meets the slanted window edge "
                         "(sharp L in section, fold-vs-butt not resolvable)"}
    print(name, round(out[name]["angle_deg_median"], 1), [round(x, 1) for x in a])
(OUT / "a2_tab_crease_final.json").write_text(json.dumps(out, indent=1))
