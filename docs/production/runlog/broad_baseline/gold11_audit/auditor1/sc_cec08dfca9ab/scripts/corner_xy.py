"""Auditor-1: x-y slices (constant z) at the four corners + front edge of end walls."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *
xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
zs = [-8, 5, 20, 27.5]
wins = [("back-left", (-162, -140), (100, 120)), ("back-right", (138, 160), (100, 120)),
        ("front-left", (-162, -140), (-95, -75)), ("front-right", (138, 160), (-95, -75))]
fig, axs = plt.subplots(len(zs), len(wins), figsize=(5.0 * len(wins), 4.6 * len(zs)))
for i, z0 in enumerate(zs):
    m = np.abs(P[:, 2] - z0) < 0.4
    for j, (nm, xr, yr) in enumerate(wins):
        ax = axs[i, j]
        mm = m & (P[:, 0] > xr[0]) & (P[:, 0] < xr[1]) & (P[:, 1] > yr[0]) & (P[:, 1] < yr[1])
        ax.scatter(P[mm, 0], P[mm, 1], s=2, c="k")
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"{nm} x-y slice z={z0}")
fig.tight_layout(); fig.savefig(OUTD + "a1_corner_xy_slices.png", dpi=60)
print("ok")
