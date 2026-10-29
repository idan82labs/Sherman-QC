"""Where do the short-side BOTTOM bends (wall -> ~12 mm bottom flange) end? Close-up L-H
sections stepping in S through the transition at both ends, both short walls."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *

Q = np.load(SCR / "Q.npy"); NQ = np.load(SCR / "NQ.npy")
for wall, lref, sgn in (("Lminus", -148.95, 1), ("Lplus", 163.68, -1)):
    svals = list(np.arange(-80, -54, 2.5)) + list(np.arange(60, 86, 2.5))
    fig, axes = plt.subplots(4, 6, figsize=(30, 15), dpi=70)
    for ax, s0 in zip(axes.ravel(), svals):
        m = (np.abs(Q[:, 1] - s0) < 0.4) & (np.abs(Q[:, 0] - lref) < 20) & (Q[:, 2] < -12)
        u = sgn * (Q[m, 0] - lref)
        ax.scatter(u, Q[m, 2], s=3, c=NQ[m, 2], cmap="coolwarm", vmin=-1, vmax=1, linewidths=0)
        ax.set_xlim(-1.5, 14); ax.set_ylim(-23, -13); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{wall} S={s0:.1f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"a3_shortbot_ends_{wall}.png"); plt.close(fig)
print("ok")
