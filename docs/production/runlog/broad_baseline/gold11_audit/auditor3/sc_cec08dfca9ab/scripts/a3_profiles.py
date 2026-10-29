"""Profile sections in the floor frame (read-only).

a3_profiles_fixed_x.png : YZ profiles at fixed X (top side C-channel + bottom edge feature)
a3_profiles_fixed_y.png : XZ profiles at fixed Y (left/right C-channels, wall extent)
Gold lines crossing each section are drawn as green stars.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json

import numpy as np

from a3_common import OUT, load_gold, load_points


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = json.loads((OUT / "part_frame.json").read_text())
    c = np.array(fr["origin"])
    M = np.array([fr["X"], fr["Y"], fr["Z"]])
    P = (load_points() - c) @ M.T
    G = [(b, (s - c) @ M.T, (e - c) @ M.T) for b, s, e in load_gold()]
    half = 0.6

    def gold_hits(axis, val):
        out = []
        for b, s, e in G:
            if (s[axis] - val) * (e[axis] - val) > 0 or abs(e[axis] - s[axis]) < 1e-6:
                continue
            t = (val - s[axis]) / (e[axis] - s[axis])
            out.append((b, s + t * (e - s)))
        return out

    # ---- fixed X: YZ profiles
    xs = [-140, -100, -50, 0, 50, 100, 140]
    fig, axes = plt.subplots(len(xs), 1, figsize=(16, 3.0 * len(xs)), dpi=90)
    for ax, xv in zip(axes, xs):
        m = np.abs(P[:, 0] - xv) < half
        ax.scatter(P[m, 1], P[m, 2], s=0.6, c="#333", linewidths=0, rasterized=True)
        for b, q in gold_hits(0, xv):
            ax.plot(q[1], q[2], "*", color="#1a9c3a", ms=13, mec="k")
            ax.annotate(b, (q[1], q[2]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="#1a9c3a")
        ax.set_aspect("equal")
        ax.set_title(f"section X = {xv} mm (slab +-{half}) : Y (horizontal) vs Z height (vertical)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-140, 152)
        ax.set_ylim(-5, 45)
    fig.tight_layout()
    fig.savefig(OUT / "a3_profiles_fixed_x.png")
    plt.close(fig)

    # ---- fixed Y: XZ profiles, zoom on both side walls
    ys = [-125, -100, -70, -56, -50, -30, 0, 50, 100, 130, 144]
    fig, axes = plt.subplots(len(ys), 2, figsize=(12, 3.2 * len(ys)), dpi=90)
    for row, yv in zip(axes, ys):
        m = np.abs(P[:, 1] - yv) < half
        for ax, (lo, hi) in zip(row, ((-165, -115), (115, 165))):
            mm = m & (P[:, 0] > lo) & (P[:, 0] < hi)
            ax.scatter(P[mm, 0], P[mm, 2], s=0.8, c="#333", linewidths=0, rasterized=True)
            for b, q in gold_hits(1, yv):
                if lo < q[0] < hi:
                    ax.plot(q[0], q[2], "*", color="#1a9c3a", ms=13, mec="k")
                    ax.annotate(b, (q[0], q[2]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="#1a9c3a")
            ax.set_aspect("equal")
            ax.set_xlim(lo, hi)
            ax.set_ylim(-5, 45)
            ax.grid(alpha=0.3)
            ax.set_title(f"section Y = {yv} (slab +-{half}) : X vs Z", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "a3_profiles_fixed_y.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
