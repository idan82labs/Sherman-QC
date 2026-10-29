"""Auditor-1 (sc_cec08dfca9ab): thin-slab (+-0.3 mm) x-z close-ups of both end-wall tops, to check for any narrow
lip / return flange outside the measured HB5/HB6 fold run (y in [-51.5, 81.4]); gold HB5/HB6 crossing marked."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import load, to_part, OUTD

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}
ys = [-82, -75, -62, -54, -48, 70, 80, 84, 90, 100, 108, 113]
fig, axs = plt.subplots(2, len(ys), figsize=(2.3 * len(ys), 7.2))
for j, y0 in enumerate(ys):
    m = np.abs(P[:, 1] - y0) < 0.3
    for i, (lo, hi, gname) in enumerate(((-161, -145, "HB5"), (144, 158, "HB6"))):
        ax = axs[i, j]
        mm = m & (P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 2] > 17) & (P[:, 2] < 31.5)
        ax.scatter(P[mm, 0], P[mm, 2], s=1.2, c="k")
        a, b = gp[gname]
        t = (y0 - a[1]) / (b[1] - a[1])
        if 0 <= t <= 1:
            q = a + t * (b - a)
            ax.plot(q[0], q[2], "*", color="#d000d0", ms=11)
        inside = -51.5 <= y0 <= 81.4
        ax.set_title(f"{gname} y={y0}\n{'fold run' if inside else 'outside fold run'}", fontsize=9,
                     color="#0a8f2a" if inside else "#b00000")
        ax.set_xlim(lo, hi); ax.set_ylim(17, 31.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
fig.suptitle("End-wall tops, x-z sections (part frame, slab +-0.3 mm). Star = gold HB5/HB6. Any lip/return outside the fold run?", fontsize=11)
fig.tight_layout()
fig.savefig(OUTD + "a1_endwall_top_closeups.png", dpi=85)
print("ok")
