"""Auditor-3: oblique vertical slabs in the part frame (slab normal = horizontal unit vector at heading H deg
from +X'), plotted as (horizontal lateral, Z'). Used for the angled tab ends.

usage: a3_oblique.py NAME  heading_deg cx cy cz half_slab half_w half_h  [heading ...]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402


def main(argv):
    name = argv[0]
    rest = argv[1:]
    specs = []
    while rest:
        h, cx, cy, cz, hs, hw, hh = map(float, rest[:7])
        specs.append((h, np.array([cx, cy, cz]), hs, hw, hh))
        rest = rest[7:]
    p, n, _ = c.load_scan()
    R = np.load(c.CACHE / "part_R.npy")
    o = np.load(c.CACHE / "part_o.npy")
    q = (p - o) @ R.T
    nq = n @ R.T
    ncol = min(3, len(specs))
    nrow = (len(specs) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 6.5 * nrow), squeeze=False)
    for ax, (h, ctr, hs, hw, hh) in zip(axs.ravel(), specs):
        dn = np.array([math.cos(math.radians(h)), math.sin(math.radians(h)), 0.0])
        lat = np.array([-dn[1], dn[0], 0.0])
        rel = q - ctr
        m = (np.abs(rel @ dn) <= hs) & (np.abs(rel @ lat) <= hw) & (np.abs(rel[:, 2]) <= hh)
        col = np.clip(np.abs(nq[m]), 0, 1)
        ax.scatter((rel @ lat)[m], q[m, 2], s=3, c=col)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(-hw, hw)
        ax.set_ylim(ctr[2] - hh, ctr[2] + hh)
        ax.set_xlabel(f"lateral (heading {h + 90:.0f} deg) from ({ctr[0]:.1f},{ctr[1]:.1f})")
        ax.set_ylabel("Z'")
        ax.set_title(f"slab normal heading {h:.0f} deg, ±{hs} mm, n={m.sum()}")
    for ax in axs.ravel()[len(specs):]:
        ax.axis("off")
    fig.suptitle(f"auditor3 sc_4f8188b4d1a8 {name} (oblique vertical slabs; colour |n| R=X' G=Y' B=Z')")
    fig.tight_layout()
    out = c.OUT / f"a3_{name}.png"
    fig.savefig(out, dpi=75)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
