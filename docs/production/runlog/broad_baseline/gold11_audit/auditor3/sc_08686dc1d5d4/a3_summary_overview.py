#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4: one overview of gold lines vs the formed bends measured from the scan."""
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    P = load_points()
    o, e1, e2, n = np.load(HERE / "_plate_frame.npy")
    X = P - o
    U, V, H = X @ e1, X @ e2, X @ n
    B = json.loads((HERE / "formed_bends_apex.json").read_text())
    rng = np.random.default_rng(3)
    s = rng.choice(len(P), 300000, replace=False)
    fig, ax = plt.subplots(figsize=(17, 14), dpi=95)
    hs = np.clip(H[s], -42, 2)
    sc = ax.scatter(U[s], V[s], c=hs, cmap="cividis", s=0.25, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.025, label="h (mm): plate ~0, joggled strips ~-1.5, walls to -40, feet ~-40")
    for k, b in B.items():
        a = np.array(b["uvh_start"])
        e = np.array(b["uvh_end"])
        col = "#1f77b4" if "base" in k else "#ff7f0e" if "foot" in k else "#9467bd"
        ax.plot([a[0], e[0]], [a[1], e[1]], color=col, lw=2.2, alpha=0.9)
        off = {"wallA_base": (0, -7), "wallB_base": (0, 7), "wallA_foot": (0, 6), "wallB_foot": (0, -6)}.get(k, (0, 0))
        mx, my = 0.5 * (a[0] + e[0]) + off[0], 0.5 * (a[1] + e[1]) + off[1]
        if k.startswith("joggle"):
            my = -60 if k.endswith("_1") else -40
            mx += -12 if k.startswith("joggleL") else 4
        ax.text(mx, my, k, color=col, fontsize=8, ha="center", va="center", rotation=90 if k.startswith("joggle") else 0,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=col, alpha=0.85))
    for bid, s0, s1 in gold_lines():
        g = np.array([s0, s1]) - o
        gu, gv = g @ e1, g @ e2
        ax.plot(gu, gv, color="#1a9c3a", lw=1.6, ls="--")
        ax.text(gu.mean(), gv.mean() + (4 if bid in ("M01", "M02") else 0), bid, color="white", fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="#1a9c3a", ec="none"))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title("sc_08686dc1d5d4 (10839000 scan_1): gold (green dashed) vs FORMED bends measured from the scan\n"
                 "blue = plate->wall 90 deg (u -102..99), orange = wall->flange 90 deg (u -68..65, at wall bottom), "
                 "purple = joggle bends ~35-40 deg (2 per edge, full width)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "a3_summary_overview.png")
    print("ok")


if __name__ == "__main__":
    main()
