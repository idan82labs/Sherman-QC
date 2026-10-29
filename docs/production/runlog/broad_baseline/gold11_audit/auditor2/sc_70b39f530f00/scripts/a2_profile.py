"""Auditor-2: exact mesh sections perpendicular to the common gold direction (all 3 gold lines are
parallel to < 0.01 deg), at a series of stations along that direction. Each panel shows every
section polyline in the (u, w) plane plus where each gold line pierces the plane (star, only when
the station lies inside that gold line's extent; hollow marker = gold line extended)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, perp_basis, section, unit  # noqa: E402

m = load_mesh()
gold = load_gold()
d = unit(gold["M01"][1] - gold["M01"][0])
u, w = perp_basis(d)
# orient u/w so the plots are readable: w ~ the arm normal direction; keep right-handed
o = gold["M02"][0]

ext = {}
for bid, (s, e) in gold.items():
    ts, te = sorted([(s - o) @ d, (e - o) @ d])
    ext[bid] = (ts, te)
    print(bid, "extent along d [mm]", round(ts, 2), round(te, 2), "pierce (u,w) at start",
          np.round([(s - o) @ u, (s - o) @ w], 2))
V = np.asarray(m.vertices)
tv = (V - o) @ d
print("part extent along d:", round(tv.min(), 2), round(tv.max(), 2))

stations = np.arange(np.floor(min(e[0] for e in ext.values())) - 6, np.ceil(max(e[1] for e in ext.values())) + 7, 3.0)
cols = 4
rows = int(np.ceil(len(stations) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5.2 * rows))
axs = np.atleast_2d(axs)
gcol = {"M01": "tab:red", "M02": "tab:blue", "M03": "tab:green"}
for k, t in enumerate(stations):
    ax = axs[k // cols, k % cols]
    p0 = o + t * d
    polys = section(m, p0, d)
    for pl in polys:
        q = np.c_[(pl - o) @ u, (pl - o) @ w]
        ax.plot(q[:, 0], q[:, 1], "-", color="0.25", lw=0.8)
    for bid, (s, e) in gold.items():
        # pierce point of the infinite gold line with this plane
        a = (t - (s - o) @ d) / ((e - s) @ d)
        p = s + a * (e - s)
        inside = ext[bid][0] - 1e-6 <= t <= ext[bid][1] + 1e-6
        ax.plot((p - o) @ u, (p - o) @ w, "*" if inside else "o", ms=12 if inside else 7,
                mfc=gcol[bid] if inside else "none", mec=gcol[bid], label=bid if k == 0 else None)
    ax.set_title(f"t = {t:+.1f} mm along gold dir")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(-45, 30)
    ax.set_ylim(-30, 40)
for k in range(len(stations), rows * cols):
    axs[k // cols, k % cols].axis("off")
axs[0, 0].legend(loc="upper left")
fig.suptitle("sc_70b39f530f00: exact mesh sections perpendicular to the common gold direction "
             "(filled star = inside that gold line's extent; hollow = extension)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "a2_profile_stations.png", dpi=70)
print("wrote", OUT / "a2_profile_stations.png", "stations", stations)
