"""Profile sections in the part frame (raw scan, read-only)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1]); Q = d["Q"]; NQ = d["NQ"]
out = Path(sys.argv[2])
half = 0.4

def sec_long(ax, x0, ylo, yhi, zlo, zhi, title):
    m = (np.abs(Q[:, 0] - x0) < half) & (Q[:, 1] > ylo) & (Q[:, 1] < yhi) & (Q[:, 2] > zlo) & (Q[:, 2] < zhi)
    ax.scatter(Q[m, 1], Q[m, 2], s=2, c="k")
    ax.set_aspect("equal"); ax.grid(True, lw=0.3); ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)

def sec_short(ax, y0, xlo, xhi, zlo, zhi, title):
    m = (np.abs(Q[:, 1] - y0) < half) & (Q[:, 0] > xlo) & (Q[:, 0] < xhi) & (Q[:, 2] > zlo) & (Q[:, 2] < zhi)
    ax.scatter(Q[m, 0], Q[m, 2], s=2, c="k")
    ax.set_aspect("equal"); ax.grid(True, lw=0.3); ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)

xs = [-150, -100, -40, 40, 100, 140]
fig, axes = plt.subplots(4, 6, figsize=(24, 14))
for j, x0 in enumerate(xs):
    sec_long(axes[0, j], x0, -103, -70, 15, 25, f"S9 top (wall3) x={x0}: y vs z")
    sec_long(axes[1, j], x0, 75, 108, 15, 25, f"S10 top (wall4) x={x0}: y vs z")
    sec_long(axes[2, j], x0, -103, -85, -23, -12, f"S9 bottom x={x0}")
    sec_long(axes[3, j], x0, 92, 108, -23, -12, f"S10 bottom x={x0}")
fig.tight_layout(); fig.savefig(out / "sec_long.png", dpi=80)

ys = [-95, -85, -70, -60, 0, 60, 70, 80, 90, 100]
fig, axes = plt.subplots(4, len(ys), figsize=(34, 14))
for j, y0 in enumerate(ys):
    sec_short(axes[0, j], y0, 128, 152, 14, 25, f"wall8 top y={y0}: x vs z")
    sec_short(axes[1, j], y0, -167, -143, 14, 25, f"wall7 top y={y0}: x vs z")
    sec_short(axes[2, j], y0, 140, 152, -23, -12, f"wall8 bottom y={y0}")
    sec_short(axes[3, j], y0, -167, -155, -23, -12, f"wall7 bottom y={y0}")
fig.tight_layout(); fig.savefig(out / "sec_short.png", dpi=70)
print("ok")
