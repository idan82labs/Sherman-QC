"""Window tabs (gold B09 at X~-81.5, B10 at X~+91.3; tabs hang toward -N) and their ends:
looks for the 30-deg kink of the tab where the window edge turns (CAD C5/C7, 12.7 mm, 30 deg).
Part frame as a1common. Read-only on inputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import a1common as A

pts, nrm, _ = A.load_scan()
g = A.load_gold()
ex, ez, en = A.part_frame(g)
c = pts.mean(0)
M = np.c_[ex, ez, en].T
P = (pts - c) @ M.T
Nn = nrm @ M.T
G = {k: ((a - c) @ M.T, (b - c) @ M.T) for k, (a, b) in g.items()}

fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=85)
for row, bid in enumerate(("B09", "B10")):
    a, b = G[bid]
    x0 = 0.5 * (a[0] + b[0])
    # plan view of the tab (X-Z), points below the panel (N < -14.5) = the tab body only
    ax = axes[row, 0]
    m = (P[:, 2] < -14.5) & (P[:, 2] > -30) & (np.abs(P[:, 0] - x0) < 15) & (P[:, 1] > -75) & (P[:, 1] < 35)
    ax.scatter(P[m, 0], P[m, 1], s=1, c=P[m, 2], cmap="viridis")
    ax.plot([a[0], b[0]], [a[1], b[1]], color="r", lw=1.5, label=f"gold {bid}")
    ax.set_title(f"{bid} tab plan view (X-Z), points N<-14.5 (tab body)", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # zoom near the tab end (Z 0..30) at three N levels
    for ax, nlev in zip(axes[row, 1:], (-17.0, -21.0, -25.0)):
        m = (np.abs(P[:, 2] - nlev) < 0.35) & (np.abs(P[:, 0] - x0) < 12) & (P[:, 1] > -5) & (P[:, 1] < 32)
        ax.scatter(P[m, 0], P[m, 1], s=4, c="k")
        ax.quiver(P[m, 0], P[m, 1], Nn[m, 0], Nn[m, 1], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="r", lw=1.5, label=f"gold {bid} (projected)")
        ax.set_title(f"{bid} tab end: slab N={nlev}+-0.35 (X horiz, Z vert)", fontsize=9)
        ax.set_xlim(x0 - 12, x0 + 12); ax.set_ylim(-5, 32)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
# row 2: tab profile (X-N) at a few Z along each tab incl. beyond the gold end
for ax, (bid, zl) in zip(axes[2], (("B09", 0.0), ("B09", 16.0), ("B10", 0.0), ("B10", 16.0))):
    a, b = G[bid]
    x0 = 0.5 * (a[0] + b[0])
    m = (np.abs(P[:, 1] - zl) < 0.35) & (np.abs(P[:, 0] - x0) < 14) & (P[:, 2] < -5) & (P[:, 2] > -30)
    ax.scatter(P[m, 0], P[m, 2], s=4, c="k")
    ax.quiver(P[m, 0], P[m, 2], Nn[m, 0], Nn[m, 2], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
    t = (zl - a[1]) / (b[1] - a[1])
    gp = a + t * (b - a)
    if 0 <= t <= 1:
        ax.plot(gp[0], gp[2], "*", color="lime", mec="k", ms=14, label=f"gold {bid}")
    ax.set_title(f"{bid} tab profile slab Z={zl}+-0.35 (X horiz, N vert)", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle("sc_2714fb7e0c0b window tabs: plan view + tab-end slabs (blue ticks = oriented normals)")
fig.tight_layout()
fig.savefig(A.OUT / "a1_tabs.png")
print("ok")
