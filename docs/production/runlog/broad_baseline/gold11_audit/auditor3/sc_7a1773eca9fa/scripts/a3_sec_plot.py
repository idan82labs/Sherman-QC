"""Auditor 3: raw 2D sections (with outward-normal ticks) at given fractions of each gold line."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_load import load  # noqa: E402
from a3_bend import axial_modes, unit  # noqa: E402

GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"
OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor3/sc_7a1773eca9fa"

v, n, f = load()
gold = {b["bend_id"]: b for b in json.load(open(GOLD))["bends"]}


def frame_for(bid):
    b = gold[bid]
    S, E = np.array(b["line_start"]), np.array(b["line_end"])
    L = np.linalg.norm(E - S); g = (E - S) / L
    rel = v - S; t = rel @ g; lat = np.linalg.norm(rel - np.outer(t, g), axis=1)
    sel = (t > 0) & (t < L) & (lat > 6) & (lat < 40)
    modes = axial_modes(n[sel], k=5)
    best = None
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            ax = np.cross(modes[i], modes[j])
            if np.linalg.norm(ax) < 0.2:
                continue
            ang = np.degrees(np.arccos(min(1, abs(unit(ax) @ g))))
            if best is None or ang < best[0]:
                best = (ang, i, j)
    mA, mB = modes[best[1]], modes[best[2]]
    d = unit(np.cross(mA, mB))
    if d @ g < 0:
        d = -d
    return S, E, L, g, d, mA, mB


def section(bid, frac, half=0.5, R=40.0, ax=None, zoom=None):
    S, E, L, g, d, mA, mB = frame_for(bid)
    G = S + frac * (E - S)
    e1 = unit(mA - (mA @ d) * d)
    e2 = np.cross(d, e1)
    rel = v - G
    s = rel @ d
    m = (np.abs(s) <= half)
    x = rel[m] @ e1; y = rel[m] @ e2
    keep = np.hypot(x, y) < R
    x, y = x[keep], y[keep]
    nn = n[m][keep]
    cA = np.abs(nn @ mA) > 0.97
    cB = np.abs(nn @ mB) > 0.97
    other = ~(cA | cB)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x[cA], y[cA], s=2, c="tab:blue", label="plate A normal")
    ax.scatter(x[cB], y[cB], s=2, c="tab:orange", label="plate B normal")
    ax.scatter(x[other], y[other], s=2, c="0.3", label="other (arc/edge)")
    # outward normal ticks (every k-th point)
    nx, ny = nn @ e1, nn @ e2
    k = max(1, len(x) // 250)
    ax.quiver(x[::k], y[::k], nx[::k], ny[::k], angles="xy", scale_units="xy", scale=1.0 / 1.2, width=0.002, color="0.6")
    ax.plot([0], [0], "*", ms=14, mfc="lime", mec="k", label=f"gold {bid} @ {frac:.2f}")
    ax.set_aspect("equal")
    if zoom:
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.grid(alpha=0.3)
    ax.set_title(f"{bid} frac {frac:.2f}  slab +-{half} mm  (x=plate-A normal dir, y=d x e1)")
    return ax


if __name__ == "__main__":
    for bid in gold:
        fig, axs = plt.subplots(2, 3, figsize=(21, 14))
        for k, fr in enumerate((0.1, 0.5, 0.9)):
            section(bid, fr, ax=axs[0, k], R=40)
            section(bid, fr, ax=axs[1, k], R=14, zoom=12)
        axs[0, 0].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/a3_rawsec_{bid}.png", dpi=80)
        plt.close(fig)
        print("wrote", bid)
