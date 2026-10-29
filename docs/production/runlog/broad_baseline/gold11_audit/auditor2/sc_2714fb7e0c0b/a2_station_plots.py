"""Plot a2_stations.json: per fold, gold offset from apex (along/across) and fold presence vs station."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
d = json.loads((OUT / "a2_stations.json").read_text())
names = list(d)
fig, axes = plt.subplots(len(names), 1, figsize=(14, 2.6 * len(names)), dpi=85)
for ax, n in zip(axes, names):
    st = d[n]["stations"]
    s = np.array([r["s"] for r in st])
    fold = np.array([r["ok"] and r.get("Ro") is not None and 0.8 < r["Ro"] < 8 and 60 < r["bend_deg"] < 120 for r in st])
    ro = np.array([r.get("Ro") if r.get("ok") and r.get("Ro") is not None else np.nan for r in st], float)
    ax.plot(s, np.where(fold, ro, np.nan), "k.", ms=3, label="R_o fit (fold present)")
    for key, col in (("gold_along", "#d62728"), ("gold_across", "#1f77b4"), ("gold_d_apex", "#2ca02c")):
        y = np.array([r.get(key, np.nan) if r.get("ok") else np.nan for r in st], float)
        ax.plot(s, y, ".-", color=col, ms=3, lw=0.8, label=key)
    g = d[n]["gold"]
    if g:
        axi = d[n]["spec"]["ax"]
        ax.axvspan(min(g[0][axi], g[1][axi]), max(g[0][axi], g[1][axi]), color="#d62728", alpha=0.07, label="gold span")
    ax.axhline(4, color="r", ls=":", lw=0.7); ax.axhline(-4, color="r", ls=":", lw=0.7); ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-5, 7); ax.grid(alpha=0.3)
    ax.set_title(f"{n}: station along fold axis ({'uvw'[d[n]['spec']['ax']]}) ; gold offset vs outer apex (mm); along + = concave/inside", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=5)
fig.tight_layout(); fig.savefig(OUT / "a2_station_profiles.png"); plt.close(fig)
print("ok")
