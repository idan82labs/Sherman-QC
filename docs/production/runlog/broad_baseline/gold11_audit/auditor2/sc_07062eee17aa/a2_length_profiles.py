"""Auditor 2: lengthwise profiles (s vs w) at several u, to check the part ends and the plate for
formed features perpendicular to the gold lines (end flanges, tabs) that no gold line covers."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c
from a2_profiles import G, P

U = [-111.5, -100, -82, -70, -40, 0, 40, 80, 92, 100, 111]
fig, axs = plt.subplots(len(U), 1, figsize=(18, 2.3 * len(U)), sharex=True)
for ax, uu in zip(axs, U):
    m = np.abs(P[:, 0] - uu) < 0.5
    q = P[m]
    ax.scatter(q[:, 2], q[:, 1], s=0.8, c="k")
    ax.set_ylim(-17, 17)
    ax.set_ylabel(f"u={uu}")
    ax.grid(alpha=0.3)
axs[0].set_title("lengthwise profiles (s horizontal, w vertical, true scale in w window +-17 mm)")
axs[-1].set_xlabel("s (mm)")
fig.tight_layout()
fig.savefig(c.OUT / "a2_length_profiles.png", dpi=75)

# full end-on views of each part end, all points within 12 mm of the end
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
smin, smax = P[:, 2].min(), P[:, 2].max()
for ax, (lo, hi, nm) in zip(axs, [(smin, smin + 6, "s-min end"), (smax - 6, smax, "s-max end")]):
    m = (P[:, 2] >= lo) & (P[:, 2] <= hi)
    q = P[m]
    sc = ax.scatter(q[:, 0], q[:, 1], s=0.5, c=q[:, 2], cmap="viridis")
    ax.set_title(f"{nm}: points with s in [{lo:.0f},{hi:.0f}] (colour = s)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax)
fig.tight_layout()
fig.savefig(c.OUT / "a2_part_ends.png", dpi=75)
print("ok", smin, smax)
