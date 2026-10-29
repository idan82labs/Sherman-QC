"""Auditor-1 (sc_71cf0b216f43): close-up y-z sections of the two long-wall top-flange joggles (missing bends)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get

P, VN, F, G, R, c = get()
jg = json.load(open(OUT + "/a1_joggles.json"))["joggles"]
xs = [-158, -100, 0, 100, 144]
fig, axes = plt.subplots(2, len(xs), figsize=(5 * len(xs), 9), dpi=80)
for i, (name, (ylo, yhi)) in enumerate((("front_top_flange", (-86, -68)), ("back_top_flange", (76, 94)))):
    j = jg[name]
    for k, x0 in enumerate(xs):
        ax = axes[i, k]
        m = (np.abs(P[:, 0] - x0) < 0.4) & (P[:, 1] > ylo) & (P[:, 1] < yhi) & (P[:, 2] > 16) & (P[:, 2] < 24)
        col = np.where(VN[m, 2] > 0, "#d62728", "#1f77b4")
        ax.scatter(P[m, 1], P[m, 2], s=5, c=col, linewidths=0)
        st = [s for s in j["stations"] if abs(s["x"] - x0) < 1.01 and s.get("ok")]
        if st:
            s = st[0]
            ax.axvline(s["y_fold_upper"], color="magenta", ls="--", lw=1)
            ax.axvline(s["y_fold_lower"], color="cyan", ls="--", lw=1)
            ax.set_title(f"{name} x={x0}: drop {s['drop']:.2f} mm, web {s['web_slope_deg']:.0f} deg\nfold lines y={s['y_fold_upper']:.1f} (upper) / {s['y_fold_lower']:.1f} (lower)", fontsize=9)
        else:
            ax.set_title(f"{name} x={x0}", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
fig.suptitle("sc_71cf0b216f43 joggles in the long-wall top flanges (no gold line): red = +z-facing skin, blue = -z-facing skin; "
             "dashed = the two joggle fold lines (top-skin virtual sharps)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT + "/a1_joggles.png")
print("ok")
