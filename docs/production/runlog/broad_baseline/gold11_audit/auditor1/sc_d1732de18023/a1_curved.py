"""Where are the curved (fold-like) surfaces?  Points whose normal lies between two principal
families (20..70 deg from both) are split by the fold axis they imply."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from a1_common import OUT, gold_lines, load_all, pca_frame  # noqa: E402

xyz, nrm, faces = load_all()
c, vt = pca_frame(xyz)
P = (xyz - c) @ vt.T
N = nrm @ vt.T
G = {k: ((a - c) @ vt.T, (b - c) @ vt.T) for k, (a, b) in gold_lines().items()}
A = np.degrees(np.arccos(np.clip(np.abs(N), 0, 1)))  # angle to each PC axis
# normal ~ in plane (i,j) means angle to the third axis ~ 90
sets = {}
for axis, (i, j) in {"about PC2 (lip-type)": (0, 2), "about PC1 (wall-type)": (1, 2),
                     "about PC3 (vertical)": (0, 1)}.items():
    k = 3 - i - j
    m = (A[:, k] > 75) & (A[:, i] > 20) & (A[:, i] < 70) & (A[:, j] > 20) & (A[:, j] < 70)
    sets[axis] = m
    print(axis, m.sum())

fig, axs = plt.subplots(3, 1, figsize=(16, 26))
for ax, (name, m) in zip(axs, sets.items()):
    idx = np.where(m)[0]
    sc = ax.scatter(P[idx, 0], P[idx, 1], c=P[idx, 2], s=1.0, cmap="coolwarm", vmin=-16, vmax=20)
    for k, (a, b) in G.items():
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="k", lw=0.8)
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, k, fontsize=8, bbox=dict(fc="yellow", alpha=0.7, lw=0))
    ax.set_aspect("equal")
    ax.set_title(f"curved points, fold axis {name}: n={len(idx)}  (colour = PC3 height)")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.6)
fig.tight_layout()
fig.savefig(OUT / "a1_curved_points.png", dpi=90)
print("wrote")
