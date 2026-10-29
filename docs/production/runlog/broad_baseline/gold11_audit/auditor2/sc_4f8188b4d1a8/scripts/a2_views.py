"""Part-frame projections of the whole scan with gold lines (auditor 2)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a2_common import OUT, get_scan, gold_lines
from a2_frame import to_part

xyz, nrm, faces = get_scan()
P = to_part(xyz)
G = gold_lines()
rng = np.random.default_rng(0)
sub = rng.choice(len(P), size=min(len(P), 250000), replace=False)
Q = P[sub]
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
views = [(0, 1, 2, "top: L vs W (colour N)"), (0, 2, 1, "side: L vs N (colour W)"), (1, 2, 0, "front: W vs N (colour L)")]
for ax, (i, j, c, title) in zip(axes, views):
    sc = ax.scatter(Q[:, i], Q[:, j], s=0.3, c=Q[:, c], cmap="viridis", linewidths=0)
    for bid, (a, b) in sorted(G.items()):
        pa, pb = to_part(a), to_part(b)
        ax.plot([pa[i], pb[i]], [pa[j], pb[j]], "-", color="red", lw=1.6)
        ax.text((pa[i] + pb[i]) / 2, (pa[j] + pb[j]) / 2, bid, color="red", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.6)
fig.suptitle("auditor2 sc_4f8188b4d1a8 part frame (L = B06 dir, W = B02 dir orthog., N = L x W; origin = B06 start); red = gold")
fig.tight_layout()
fig.savefig(OUT / "a2_partframe_views.png", dpi=80)
print(OUT / "a2_partframe_views.png")
