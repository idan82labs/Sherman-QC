"""Gold B07 (far flange at Z~-138) near its ends: is the fold present where the gold line runs
past X=+-101? Profiles (Z-N) at several X and a plan view of the corners. Read-only on inputs."""
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
a, b = ((g["B07"][0] - c) @ M.T, (g["B07"][1] - c) @ M.T)

fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=85)
for row, sgn in ((0, -1), (1, 1)):
    for ax, xa in zip(axes[row], (95.0, 100.0, 103.0, 108.0)):
        xl = sgn * xa
        m = (np.abs(P[:, 0] - xl) < 0.35) & (np.abs(P[:, 1] + 141) < 10) & (P[:, 2] > -20) & (P[:, 2] < 5)
        ax.scatter(P[m, 1], P[m, 2], s=5, c="k")
        ax.quiver(P[m, 1], P[m, 2], Nn[m, 1], Nn[m, 2], angles="xy", scale=25, width=0.002, color="tab:blue", alpha=0.5)
        t = (xl - a[0]) / (b[0] - a[0])
        gp = a + t * (b - a)
        ax.plot(gp[1], gp[2], "*", color="lime", mec="k", ms=15, label="gold B07")
        ax.add_patch(plt.Circle((gp[1], gp[2]), 4, fill=False, ls=":", color="g"))
        ax.set_title(f"slab X={xl:+.0f}+-0.35 (Z horiz, N vert)", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
for ax, sgn in zip(axes[2, :2], (-1, 1)):
    m = (np.abs(P[:, 0] - sgn * 104) < 12) & (P[:, 1] < -125)
    o = np.argsort(P[m, 2])
    ax.scatter(P[m, 0][o], P[m, 1][o], s=1.5, c=P[m, 2][o], cmap="viridis")
    ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=1.5, label="gold B07")
    ax.set_xlim(sgn * 104 - 12, sgn * 104 + 12)
    ax.set_title(f"plan X-Z near corner X~{sgn*104} (colour = N)", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
for ax, sgn in zip(axes[2, 2:], (-1, 1)):
    m = (np.abs(P[:, 0] - sgn * 104) < 12) & (P[:, 1] < -125)
    o = np.argsort(-P[m, 1])
    ax.scatter(P[m, 0][o], P[m, 2][o], s=1.5, c=P[m, 1][o], cmap="viridis")
    ax.plot([a[0], b[0]], [a[2], b[2]], "r-", lw=1.5, label="gold B07")
    ax.set_xlim(sgn * 104 - 12, sgn * 104 + 12)
    ax.set_title(f"front X-N near corner X~{sgn*104} (colour = Z)", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.suptitle("sc_2714fb7e0c0b gold B07 near its ends (fold measured on s=11..214 mm of 225.5)")
fig.tight_layout()
fig.savefig(A.OUT / "a1_B07_ends.png")
print("ok")
