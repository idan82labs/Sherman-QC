"""Auditor-1: x-z sections at many y near the end-wall/channel corners (left and right) to see lip topology."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}
side = sys.argv[1]  # left/right
ys = [float(v) for v in sys.argv[2].split(",")]
xr = (-162, -128) if side == "left" else (126, 160)
ncol = 4
nrow = int(np.ceil(len(ys) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 6.0 * nrow))
for ax, y0 in zip(axs.ravel(), ys):
    m = (np.abs(P[:, 1] - y0) < 0.5) & (P[:, 0] > xr[0]) & (P[:, 0] < xr[1]) & (P[:, 2] > 14)
    ax.scatter(P[m, 0], P[m, 2], s=1.5, c="k")
    for k, (a, b) in gp.items():
        da, db = a[1] - y0, b[1] - y0
        if da * db <= 0:
            t = da / (da - db); qq = a + t * (b - a)
            if xr[0] < qq[0] < xr[1] and qq[2] > 14:
                ax.plot(qq[0], qq[2], "*", ms=12, color="#ff00ff"); ax.text(qq[0] + .5, qq[2] + .5, k, color="#ff00ff")
    ax.set_title(f"x-z @ y={y0}"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlim(*xr); ax.set_ylim(14, 32)
for ax in axs.ravel()[len(ys):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(OUTD + f"a1_corner_sections_{side}.png", dpi=60)
print("ok")
