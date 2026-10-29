"""Close-up sections of the candidate unlabelled features (read-only on inputs).

1. top-flange step near Y~124 (along X)       -> sections at fixed X, zoom Y 100..150, Z 30..45
2. bottom-strip step near Y~-120 (along X)    -> sections at fixed X, zoom Y -137..-100, Z -3..6
3. left/right narrow flanges (along Y)         -> sections at fixed Y, zoom on X +-(140..160), Z 30..45
Writes a3_joggle_top.png, a3_joggle_bottom.png, a3_side_flanges.png and joggle_measure.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json

import numpy as np

from a3_common import OUT, load_points


def skin_heights(q_y, q_z, y0, y1):
    m = (q_y > y0) & (q_y < y1)
    if m.sum() < 10:
        return None
    z = np.sort(q_z[m])
    # two skins: split at largest gap
    gaps = np.diff(z)
    k = int(np.argmax(gaps))
    if gaps[k] > 0.5:
        return float(np.median(z[: k + 1])), float(np.median(z[k + 1:])), int(m.sum())
    return float(np.median(z)), None, int(m.sum())


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = json.loads((OUT / "part_frame.json").read_text())
    c0 = np.array(fr["origin"])
    M = np.array([fr["X"], fr["Y"], fr["Z"]])
    P = (load_points() - c0) @ M.T
    half = 0.5
    meas = {"top_flange": [], "bottom_strip": []}

    xs = [-150, -120, -60, 0, 60, 120, 150]
    for key, ylim, zlim, fname, bands in (
        ("top_flange", (100, 150), (30, 46), "a3_joggle_top.png", ((128, 140), (111, 120))),
        ("bottom_strip", (-137, -100), (-3, 6), "a3_joggle_bottom.png", ((-133, -124), (-115, -102))),
    ):
        fig, axes = plt.subplots(len(xs), 1, figsize=(13, 2.6 * len(xs)), dpi=95)
        for ax, xv in zip(axes, xs):
            m = (np.abs(P[:, 0] - xv) < half) & (P[:, 1] > ylim[0]) & (P[:, 1] < ylim[1]) & (P[:, 2] > zlim[0]) & (P[:, 2] < zlim[1])
            ax.scatter(P[m, 1], P[m, 2], s=1.5, c="#222", linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlim(*ylim)
            ax.set_ylim(*zlim)
            ax.grid(alpha=0.3)
            h1 = skin_heights(P[m, 1], P[m, 2], *bands[0])
            h2 = skin_heights(P[m, 1], P[m, 2], *bands[1])
            meas[key].append({"x": xv, "outer_band_skins_z": h1, "inner_band_skins_z": h2})
            ax.set_title(f"{key}: section X={xv} (slab +-{half})  skins z in Y{bands[0]}: {h1}   in Y{bands[1]}: {h2}", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)

    ys = [-20, 0, 40, 80, 120, 140]
    fig, axes = plt.subplots(len(ys), 2, figsize=(12, 2.8 * len(ys)), dpi=95)
    for row, yv in zip(axes, ys):
        m0 = np.abs(P[:, 1] - yv) < half
        for ax, (lo, hi) in zip(row, ((-160, -135), (135, 160))):
            m = m0 & (P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 2] > 25) & (P[:, 2] < 45)
            ax.scatter(P[m, 0], P[m, 2], s=1.5, c="#222", linewidths=0)
            ax.set_aspect("equal")
            ax.set_xlim(lo, hi)
            ax.set_ylim(25, 45)
            ax.grid(alpha=0.3)
            ax.set_title(f"side flange: section Y={yv}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a3_side_flanges.png")
    plt.close(fig)
    (OUT / "joggle_measure.json").write_text(json.dumps(meas, indent=1))
    print(json.dumps(meas, indent=1))


if __name__ == "__main__":
    main()
