"""Auditor-1: zoomed y-z sections of the two joggles (base plate near y=-148, top plate near y=95) at several x."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *
xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
xs = [-140, -60, 0, 60, 140]
fig, axs = plt.subplots(2, len(xs), figsize=(5.4 * len(xs), 9))
for k, x0 in enumerate(xs):
    m = np.abs(P[:, 0] - x0) < 0.4
    ax = axs[0, k]
    mm = m & (P[:, 1] > -160) & (P[:, 1] < -136) & (P[:, 2] < 0)
    ax.scatter(P[mm, 1], P[mm, 2], s=2, c="k"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"base joggle y-z @ x={x0}"); ax.set_xlim(-160, -136); ax.set_ylim(-14, -6)
    ax = axs[1, k]
    mm = m & (P[:, 1] > 84) & (P[:, 1] < 108) & (P[:, 2] > 20)
    ax.scatter(P[mm, 1], P[mm, 2], s=2, c="k"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"top-plate joggle y-z @ x={x0}"); ax.set_xlim(84, 108); ax.set_ylim(23, 31)
fig.tight_layout(); fig.savefig(OUTD + "a1_joggles_yz.png", dpi=65)
print("ok")
