#!/usr/bin/env python3
"""Read-only: constant-n slices (a-b plane) at the four part corners of sc_4f8188b4d1a8, to tell bends from seams."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage_sc_4f8188b4d1a8 as T  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts, _ = T.load_ply(T.SCAN)
    gold = {b["bend_id"]: np.array([b["line_start"], b["line_end"]], float) for b in json.load(open(T.GOLD))["manual_bends"]}
    a = T.unit(gold["B02"][1] - gold["B02"][0]); b = T.unit(gold["B01"][1] - gold["B01"][0]); b = T.unit(b - (b @ a) * a); n = np.cross(a, b)
    Rm = np.stack([a, b, n]); c0 = pts.mean(0); P = (pts - c0) @ Rm.T
    levels = (-8.0, 10.0, 25.0, 38.0, 43.0)
    corners = {"b_neg_end": (-111.0, -1), "b_pos_end": (112.0, +1)}
    fig, axs = plt.subplots(len(corners), len(levels), figsize=(4.2 * len(levels), 8.5), dpi=90)
    for r, (cname, (bc, sgn)) in enumerate(corners.items()):
        for c, lv in enumerate(levels):
            m = (np.abs(P[:, 2] - lv) < 0.8) & (P[:, 0] < -40) & (np.abs(P[:, 1] - bc) < 25)
            ax = axs[r, c]
            ax.scatter(P[m, 0], P[m, 1], s=1.5, c="k")
            ax.set_xlim(-125, -40); ax.set_ylim(bc - 25, bc + 25); ax.set_aspect("equal")
            ax.set_title(f"{cname} slice n={lv}±0.8"); ax.set_xlabel("a"); ax.set_ylabel("b")
            ax.axvline(-70.5, color="r", lw=0.5, ls=":"); ax.axvline(-117.5, color="g", lw=0.5, ls=":")
    fig.tight_layout(); fig.savefig(T.OUT / "slices_corners_constant_n.png"); plt.close(fig)
    # constant-b slices just inside each end: shows whether end walls exist at riser (a=-70, n -12..30)
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=90)
    for r, (bc, sgn) in enumerate(((-111.0, -1), (112.0, 1))):
        for c, off in enumerate((0.0, -1.5 * sgn, -4.0 * sgn)):
            m = (np.abs(P[:, 1] - (bc + off)) < 0.6) & (P[:, 0] < -30)
            ax = axs[r, c]
            ax.scatter(P[m, 0], P[m, 2], s=1.0, c="k")
            ax.set_xlim(-125, -30); ax.set_ylim(-20, 50); ax.set_aspect("equal")
            ax.set_title(f"slice b={bc + off:.1f}±0.6"); ax.set_xlabel("a"); ax.set_ylabel("n")
    fig.tight_layout(); fig.savefig(T.OUT / "slices_ends_constant_b.png"); plt.close(fig)
    # slot tabs: constant-a slices through B09/B10 (cross-section of tab bend)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=90)
    for ax, gid in zip(axs, ("B09", "B10")):
        g = (gold[gid] - c0) @ Rm.T
        am = g[:, 0].mean(); bm = g[:, 1].mean()
        m = (np.abs(P[:, 0] - am) < 0.8) & (np.abs(P[:, 1] - bm) < 40)
        ax.scatter(P[m, 1], P[m, 2], s=1.5, c="k"); ax.plot(g[:, 1].mean(), g[:, 2].mean(), "g+", ms=20, mew=2)
        ax.set_aspect("equal"); ax.set_title(f"{gid}: slice a={am:.1f}±0.8 (green + = gold line)"); ax.set_xlabel("b"); ax.set_ylabel("n")
    fig.tight_layout(); fig.savefig(T.OUT / "slices_slot_tabs.png"); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
