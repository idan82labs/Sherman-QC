#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4: plate height profiles h(u) in v-bands (joggles / pre-bends / flatness),
and the wall/foot extent along u for both formed walls. Read-only; writes PNG + JSON here."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a3_global import gold_lines, load_points  # noqa: E402


def main():
    P = load_points()
    o, e1, e2, n = np.load(HERE / "_plate_frame.npy")
    X = P - o
    u, v, h = X @ e1, X @ e2, X @ n
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {}
    bands = [(-140, -130), (-60, -50), (-5, 5), (50, 60), (130, 140)]
    fig, axes = plt.subplots(len(bands), 1, figsize=(16, 3.0 * len(bands)), dpi=90, sharex=True)
    for ax, (v0, v1) in zip(axes, bands):
        m = (v >= v0) & (v < v1) & (np.abs(h) < 5)
        ax.scatter(u[m], h[m], s=0.3, c="#444", linewidths=0, rasterized=True)
        # binned upper / lower skin
        bins = np.arange(-182, 180, 1.0)
        up, lo = [], []
        for b0 in bins:
            mm = m & (u >= b0) & (u < b0 + 1)
            if mm.sum() < 4:
                up.append(np.nan); lo.append(np.nan)
                continue
            hh = h[mm]
            up.append(np.percentile(hh, 90)); lo.append(np.percentile(hh, 10))
        up, lo = np.array(up), np.array(lo)
        ax.plot(bins + 0.5, up, color="r", lw=0.8)
        ax.plot(bins + 0.5, lo, color="b", lw=0.8)
        ax.set_ylim(-4, 3)
        ax.grid(alpha=0.3)
        ax.set_ylabel(f"h, v in [{v0},{v1})")
        for bid, s0, s1 in gold_lines():
            for p in (s0, s1):
                pass
        ax.axvline(-174.6, color="g", lw=0.8, ls="--")
        ax.axvline(91, color="g", lw=0.8, ls="--")
        res[f"v{v0}_{v1}"] = {"u": (bins + 0.5).tolist(), "upper_p90": np.nan_to_num(up, nan=-99).tolist(),
                              "lower_p10": np.nan_to_num(lo, nan=-99).tolist()}
    axes[-1].set_xlabel("u (mm) along plate e1; green dashed = gold M04 (u~-175) and M03 (u~87-95)")
    fig.suptitle("sc_08686dc1d5d4 plate height profiles h(u) per v-band (red p90 / blue p10 per 1 mm bin)")
    fig.tight_layout()
    fig.savefig(HERE / "a3_profiles_hu.png")
    plt.close(fig)

    # zoom profiles at both short edges
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=90)
    for row, (ua, ub) in enumerate(((-182, -145), (140, 178))):
        for col, (v0, v1) in enumerate(((-120, -118), (-1, 1), (118, 120))):
            ax = axes[row, col]
            m = (v >= v0) & (v < v1) & (u >= ua) & (u <= ub) & (np.abs(h) < 6)
            ax.scatter(u[m], h[m], s=3, c="#333", linewidths=0)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(f"section v in [{v0},{v1}) u in [{ua},{ub}]", fontsize=9)
            if row == 0:
                ax.axvline(-174.6, color="g", lw=1, ls="--", label="gold M04 (u at this v ~ -174.6..-176.5)")
                ax.legend(fontsize=7)
    fig.suptitle("short-edge sections (joggle check): h vs u, true scale", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "a3_joggle_sections.png")
    plt.close(fig)

    # wall / foot extents along u for each wall (v<0 = wall A at M01, v>0 = wall B at M02)
    for name, sgn in (("wallA_M01_side", -1), ("wallB_M02_side", 1)):
        side = (v * sgn > 120)
        wall = side & (h < -6) & (h > -30)
        foot = side & (h < -33)
        res[name] = {"wall_u_range_p0.2_p99.8": np.percentile(u[wall], [0.2, 99.8]).tolist(),
                     "wall_u_min_max": [float(u[wall].min()), float(u[wall].max())],
                     "foot_u_min_max": [float(u[foot].min()), float(foot.sum() and u[foot].max())],
                     "foot_h_median": float(np.median(h[foot])), "wall_v_median": float(np.median(v[wall])),
                     "foot_v_range": np.percentile(v[foot], [1, 99]).tolist()}
        # coverage of the wall along u (1 mm bins with any wall point)
        hist, edges = np.histogram(u[wall], bins=np.arange(-110, 110, 1.0))
        res[name]["wall_bins_empty_inside"] = [float(edges[i]) for i in range(len(hist)) if hist[i] == 0
                                               and res[name]["wall_u_min_max"][0] < edges[i] < res[name]["wall_u_min_max"][1]]
        print(name, json.dumps({k: v_ for k, v_ in res[name].items()}, default=float))
    (HERE / "profiles.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
