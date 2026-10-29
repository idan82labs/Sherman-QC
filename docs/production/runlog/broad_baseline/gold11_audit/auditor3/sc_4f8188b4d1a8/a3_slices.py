"""Auditor-3: part-frame slab slices / box views for topology checks (corners, ears, side flanges).

usage: a3_slices.py NAME  axis  centre_x centre_y centre_z  half_slab  half_w half_h  [axis ...]
Each panel: points with |coord[axis]-centre[axis]| <= half_slab, plotted in the other two part axes,
coloured by |normal| (R=X', G=Y', B=Z'); gold lines projected (crimson) if they cross the slab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402

AX = "X'Y'Z'"


def load_part():
    p, n, _ = c.load_scan()
    R = np.load(c.CACHE / "part_R.npy")
    o = np.load(c.CACHE / "part_o.npy")
    return (p - o) @ R.T, n @ R.T, R, o


def panel(ax, q, nq, gold_pf, axis, centre, half_slab, half_w, half_h, title=None, s=1.5):
    others = [i for i in range(3) if i != axis]
    i, j = others
    m = (np.abs(q[:, axis] - centre[axis]) <= half_slab) & (np.abs(q[:, i] - centre[i]) <= half_w) & (
        np.abs(q[:, j] - centre[j]) <= half_h)
    col = np.clip(np.abs(nq[m]), 0, 1)
    ax.scatter(q[m, i], q[m, j], s=s, c=col, rasterized=True)
    for gid, (gs, ge) in gold_pf.items():
        # sample the gold segment, keep samples in slab
        t = np.linspace(0, 1, 400)
        pts = gs[None] + t[:, None] * (ge - gs)[None]
        k = (np.abs(pts[:, axis] - centre[axis]) <= max(half_slab, 0.5)) & (np.abs(pts[:, i] - centre[i]) <= half_w) & (
            np.abs(pts[:, j] - centre[j]) <= half_h)
        if k.any():
            ax.plot(pts[k, i], pts[k, j], ".", color="crimson", ms=4)
            ax.text(pts[k, i].mean(), pts[k, j].mean(), gid, color="crimson", fontsize=10, weight="bold")
    ax.set_xlim(centre[i] - half_w, centre[i] + half_w)
    ax.set_ylim(centre[j] - half_h, centre[j] + half_h)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel(AX[2 * i:2 * i + 2])
    ax.set_ylabel(AX[2 * j:2 * j + 2])
    ax.set_title(title or f"slab {AX[2*axis:2*axis+2]}={centre[axis]:.1f}±{half_slab}  n={m.sum()}")
    return m.sum()


def main(argv):
    name = argv[0]
    specs = []
    rest = argv[1:]
    while rest:
        axis = int(rest[0])
        cx, cy, cz, hs, hw, hh = map(float, rest[1:7])
        specs.append((axis, np.array([cx, cy, cz]), hs, hw, hh))
        rest = rest[7:]
    q, nq, R, o = load_part()
    gold, _ = c.load_gold()
    gold_pf = {g["id"]: ((g["s"] - o) @ R.T, (g["e"] - o) @ R.T) for g in gold}
    ncol = min(3, len(specs))
    nrow = (len(specs) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 7 * nrow), squeeze=False)
    for ax, (axis, centre, hs, hw, hh) in zip(axs.ravel(), specs):
        panel(ax, q, nq, gold_pf, axis, centre, hs, hw, hh)
    for ax in axs.ravel()[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"auditor3 sc_4f8188b4d1a8 {name} (part frame; colour |n| R=X' G=Y' B=Z')")
    fig.tight_layout()
    out = c.OUT / f"a3_{name}.png"
    fig.savefig(out, dpi=75)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
