"""Lower-tray corners (gold B06 at X=-112, B08 at X=+112, both along N at Z~118) and the
lower-tray end flanges. Part frame as a1common. Slabs: +-0.35 mm. Read-only on inputs."""
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

for side, bid, xs in (("neg", "B06", -1), ("pos", "B08", +1)):
    a, b = G[bid]
    x0 = 0.5 * (a[0] + b[0])
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=85)
    # row 0: slices perpendicular to N (X-Z plane) at several heights along the gold line
    for ax, nlev in zip(axes[0], (31.0, 34.0, 38.0, 42.0)):
        m = (np.abs(P[:, 2] - nlev) < 0.35) & (np.abs(P[:, 0] - x0) < 14) & (np.abs(P[:, 1] - 116) < 14)
        ax.scatter(P[m, 0], P[m, 1], s=4, c="k")
        # normals as tiny ticks
        ax.quiver(P[m, 0], P[m, 1], Nn[m, 0], Nn[m, 1], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
        # gold point at this height
        t = (nlev - a[2]) / (b[2] - a[2])
        gp = a + t * (b - a)
        ax.plot(gp[0], gp[1], "*", color="lime", mec="k", ms=16, label=f"gold {bid} @N={nlev}")
        ax.add_patch(plt.Circle((gp[0], gp[1]), 4, fill=False, ls=":", color="g"))
        ax.set_title(f"{bid} corner: slab N={nlev}+-0.35 (X horiz, Z vert)", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # row 1: slices perpendicular to Z through the end flange (X-N plane) at Z = 85, 100, 112, 117
    for ax, zlev in zip(axes[1], (85.0, 100.0, 112.0, 117.0)):
        m = (np.abs(P[:, 1] - zlev) < 0.35) & (np.abs(P[:, 0] - x0) < 16) & (P[:, 2] > 20) & (P[:, 2] < 50)
        ax.scatter(P[m, 0], P[m, 2], s=4, c="k")
        ax.quiver(P[m, 0], P[m, 2], Nn[m, 0], Nn[m, 2], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
        ax.plot([a[0], b[0]], [a[2], b[2]], color="lime", lw=2, label=f"gold {bid} (projected, at Z~{a[1]:.0f})")
        ax.set_title(f"slab Z={zlev}+-0.35: end flange profile (X horiz, N vert)", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # row 2: slices perpendicular to X through the return flange near the corner (Z-N plane)
    for ax, xoff in zip(axes[2], (0.5, 2.0, 4.0, 8.0)):
        xl = x0 - xs * xoff
        m = (np.abs(P[:, 0] - xl) < 0.35) & (np.abs(P[:, 1] - 112) < 14) & (P[:, 2] > 20) & (P[:, 2] < 50)
        ax.scatter(P[m, 1], P[m, 2], s=4, c="k")
        ax.quiver(P[m, 1], P[m, 2], Nn[m, 1], Nn[m, 2], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
        ax.plot([a[1], b[1]], [a[2], b[2]], color="lime", lw=2, label=f"gold {bid} (projected)")
        ax.set_title(f"slab X={xl:.1f} ({xoff} mm inboard of gold X): Z horiz, N vert", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"sc_2714fb7e0c0b lower-tray corner {side} (gold {bid} at X={x0:.1f}); blue ticks = oriented scan normals")
    fig.tight_layout()
    fig.savefig(A.OUT / f"a1_corner_{bid}.png")
    plt.close(fig)
print("ok")
