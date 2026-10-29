"""Auditor-1: whole-part profile sections in the part frame (y-z at fixed x; x-z at fixed y), gold crossings marked."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1frame import *

xyz, nrm, faces, R, c, G = load()
P = to_part(xyz, R, c)
gp = {k: (to_part(a, R, c), to_part(b, R, c)) for k, (a, b) in G.items()}


def gold_cross(axis, val):
    out = []
    for k, (a, b) in gp.items():
        da, db = a[axis] - val, b[axis] - val
        if da * db <= 0 and abs(b[axis] - a[axis]) > 1e-6:
            t = da / (da - db)
            out.append((k, a + t * (b - a), t))
    return out


def gold_near(axis, val, i, j, tol=30):
    """gold lines roughly parallel to the section normal axis?? -> lines lying IN the section plane are drawn as segments"""
    segs = []
    for k, (a, b) in gp.items():
        if abs(a[axis] - val) < tol and abs(b[axis] - val) < tol:
            segs.append((k, a, b))
    return segs


xs = [-130, -60, 0, 60, 130]
fig, axs = plt.subplots(len(xs), 1, figsize=(24, 5.2 * len(xs)))
for ax, x0 in zip(axs, xs):
    m = np.abs(P[:, 0] - x0) < 0.6
    ax.scatter(P[m, 1], P[m, 2], s=0.4, c="k", rasterized=True)
    for k, q, t in gold_cross(0, x0):
        ax.plot(q[1], q[2], "*", ms=14, color="#ff00ff")
        ax.text(q[1] + 1, q[2] + 1.5, f"{k} t={t:.2f}", color="#ff00ff", fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"y-z section at x={x0} (slab +-0.6 mm), part frame")
    ax.set_xlim(-170, 125); ax.set_ylim(-18, 36)
fig.tight_layout(); fig.savefig(OUTD + "a1_profiles_yz.png", dpi=70)

ys = [-155, -120, -70, 0, 60, 90, 105, 112]
fig, axs = plt.subplots(len(ys), 2, figsize=(22, 4.6 * len(ys)), gridspec_kw=dict(width_ratios=[1, 1]))
for row, y0 in zip(axs, ys):
    m = np.abs(P[:, 1] - y0) < 0.6
    for ax, (lo, hi) in zip(row, [(-162, -138), (136, 160)]):
        mm = m & (P[:, 0] > lo) & (P[:, 0] < hi)
        ax.scatter(P[mm, 0], P[mm, 2], s=1.0, c="k", rasterized=True)
        for k, q, t in gold_cross(1, y0):
            if lo < q[0] < hi:
                ax.plot(q[0], q[2], "*", ms=14, color="#ff00ff")
                ax.text(q[0] + 0.5, q[2] + 1.0, f"{k} t={t:.2f}", color="#ff00ff", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlim(lo, hi); ax.set_ylim(-16, 33)
        ax.set_title(f"x-z section at y={y0} (slab +-0.6 mm), x in [{lo},{hi}]")
fig.tight_layout(); fig.savefig(OUTD + "a1_profiles_xz_ends.png", dpi=70)
print("ok")
