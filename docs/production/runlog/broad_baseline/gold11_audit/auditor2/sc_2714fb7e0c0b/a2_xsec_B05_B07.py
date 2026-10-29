"""Close-up cross-sections of B05 and B07 at 5 stations with a2 apex / arc-centre fits (from a2_stations.json)."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a2_common import load, OUT

q, R, O, raw = load()
st = json.loads((OUT / "a2_stations.json").read_text())
cfg = {"B05": (1, (0, 2), [14.0, 60.0, 112.0, 164.0, 212.0], (-6, 8), (-48, -34)),
       "B07": (1, (0, 2), [14.0, 60.0, 112.0, 164.0, 212.0], (-218, -202), (-9, 4))}
fig, axes = plt.subplots(2, 5, figsize=(24, 10), dpi=85)
for row, (name, (ax_i, (i, j), stations, xr, yr)) in enumerate(cfg.items()):
    rows = {round(r["s"], 1): r for r in st[name]["stations"]}
    for ax, s in zip(axes[row], stations):
        m = (np.abs(q[:, ax_i] - s) < 0.5) & (q[:, i] > xr[0]) & (q[:, i] < xr[1]) & (q[:, j] > yr[0]) & (q[:, j] < yr[1])
        ax.scatter(q[m, i], q[m, j], s=3, c="#555", linewidths=0)
        r = rows.get(round(s, 1))
        g0, g1 = np.array(st[name]["gold"][0]), np.array(st[name]["gold"][1])
        t = (s - g0[ax_i]) / (g1[ax_i] - g0[ax_i])
        gp = g0 + t * (g1 - g0)
        ax.plot(gp[i], gp[j], "*", color="#1a9c3a", ms=16, mec="k", label="gold")
        ax.add_patch(plt.Circle((gp[i], gp[j]), 4.0, fill=False, ls=":", color="#1a9c3a"))
        if r and r.get("ok"):
            ax.plot(*r["apex"], "D", color="#17becf", mec="k", ms=8, label="outer apex (a2 fit)")
            ax.plot(*r["X"], "x", color="#1f77b4", ms=10, mew=2, label="outer virtual sharp")
            if r.get("C"):
                ax.plot(*r["C"], "+", color="#8c564b", ms=14, mew=2, label="arc centre (bend axis)")
            ax.set_title(f"{name} @ v={s:.0f}: gold->apex {r.get('gold_d_apex', float('nan')):.2f} mm, along {r.get('gold_along', float('nan')):+.2f}, "
                         f"across {r.get('gold_across', float('nan')):+.2f}, ->centre {r.get('gold_d_centre', float('nan')):.2f}", fontsize=8)
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("u"); ax.set_ylabel("w")
    axes[row][0].legend(fontsize=7, loc="lower left")
fig.suptitle("sc_2714fb7e0c0b auditor 2: B05 / B07 cross-sections (slab +-0.5 mm along v); dotted circle = 4 mm matcher radius", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "a2_xsec_B05_B07.png"); plt.close(fig)
print("ok")
