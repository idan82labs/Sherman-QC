"""Bottom-flange overlap zones at the 4 box corners: sections perpendicular to S (L-H plane)
near each short wall, to see how the short-side bottom flange meets the long-side flange."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *

Q = np.load(SCR / "Q.npy"); NQ = np.load(SCR / "NQ.npy"); C0 = np.load(SCR / "center.npy")
gold = [(b, to_frame(a, C0), to_frame(e, C0)) for b, a, e in load_gold()]
fig, axes = plt.subplots(4, 5, figsize=(30, 16), dpi=75)
for i, (wall, lref, sgn) in enumerate((("L-", -148.95, 1), ("L-", -148.95, 1), ("L+", 163.68, -1), ("L+", 163.68, -1))):
    svals = (-97, -93, -85, -75, -40) if i % 2 == 0 else (40, 75, 85, 95, 102)
    for j, s0 in enumerate(svals):
        ax = axes[i, j]
        m = (np.abs(Q[:, 1] - s0) < 0.5) & (np.abs(Q[:, 0] - lref) < 30) & (Q[:, 2] < -10)
        u = sgn * (Q[m, 0] - lref)
        ax.scatter(u, Q[m, 2], s=2, c=NQ[m, 2], cmap="coolwarm", vmin=-1, vmax=1, linewidths=0)
        for b, a, e in gold:
            d = e - a
            if abs(d[1]) > 50:
                t = (s0 - a[1]) / d[1]
                if 0 <= t <= 1:
                    p = a + t * d
                    ax.plot(sgn * (p[0] - lref), p[2], "g*", ms=12); ax.annotate(b, (sgn * (p[0] - lref), p[2]), color="g")
        ax.set_xlim(-2, 28); ax.set_ylim(-24, -10); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{wall} wall, section S={s0} (u from short-wall outer face; colour n_H)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "a3_overlap_sections.png"); plt.close(fig)
print("ok")
