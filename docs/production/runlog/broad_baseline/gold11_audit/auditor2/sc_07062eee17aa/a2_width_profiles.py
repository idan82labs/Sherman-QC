"""Auditor 2: full-width profiles (u vs w, vertical exaggeration) at several stations s, to look for
low-angle creases/joggles across the plate that no gold line covers."""
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

S = [-150, -130, -110, -85, -50, -30, -5, 20, 45, 70, 90, 110, 130, 150]
fig, axs = plt.subplots(len(S), 1, figsize=(18, 2.2 * len(S)), sharex=True)
for ax, s in zip(axs, S):
    m = np.abs(P[:, 2] - s) < 0.5
    q = P[m]
    ax.scatter(q[:, 0], q[:, 1], s=0.8, c="k")
    for k, (a, b) in G.items():
        lam = (s - a[2]) / (b[2] - a[2])
        p = a + lam * (b - a)
        ax.plot(p[0], p[1], "*", ms=10, mfc="#1a9c3a", mec="k")
        ax.annotate(k, (p[0], p[1]), fontsize=7)
    ax.set_ylim(-4, 4)
    ax.set_ylabel(f"s={s}")
    ax.grid(alpha=0.3)
axs[0].set_title("full-width profiles, w (plate normal) exaggerated: ylim +-4 mm; flanges/tabs leave the window")
axs[-1].set_xlabel("u (mm)")
fig.tight_layout()
fig.savefig(c.OUT / "a2_width_profiles.png", dpi=75)
print("ok")
