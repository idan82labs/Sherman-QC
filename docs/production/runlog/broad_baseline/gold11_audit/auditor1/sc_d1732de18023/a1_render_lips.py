"""Lip faces seen along PC1 with gold line and fitted apex line + fold extent (extent check), plus
wall bends seen along PC2.  Read-only."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from a1_common import OUT, gold_lines, load_all, pca_frame  # noqa: E402

xyz, nrm, _ = load_all()
c, vt = pca_frame(xyz)
P = (xyz - c) @ vt.T
N = nrm @ vt.T
G = {k: ((a - c) @ vt.T, (b - c) @ vt.T) for k, (a, b) in gold_lines().items()}
EM = json.loads((OUT / "a1_extent_match.json").read_text())
APEX = {k: (np.array(v["apex_line_xyz"]) - c) @ vt.T for k, v in EM.items()}

fig, axs = plt.subplots(4, 1, figsize=(20, 22))
specs = [
    ("M01", P[:, 0] < -128, 1, 2, "M01 lip region (PC1<-128) seen from the part interior (+PC1 side), PC2-PC3",
     (-112, 104), (-14, 4)),
    ("M04", P[:, 0] > 140, 1, 2, "M04 lip region (PC1>140) seen from the part interior (-PC1 side), PC2-PC3",
     (-85, 90), (-17, 4)),
    ("M02", P[:, 1] > 92, 0, 2, "M02 wall region (PC2>92) seen from the interior, PC1-PC3", (-145, 130), (-4, 21)),
    ("M03", P[:, 1] < -96, 0, 2, "M03 wall region (PC2<-96) seen from the interior, PC1-PC3", (-145, 130), (-4, 21)),
]
for ax, (k, m, i, j, title, xl, yl) in zip(axs, specs):
    idx = np.where(m)[0]
    # colour: inner-face (normal toward interior) vs others
    if k == "M01":
        col = N[idx, 0]
    elif k == "M04":
        col = -N[idx, 0]
    elif k == "M02":
        col = -N[idx, 1]
    else:
        col = N[idx, 1]
    ax.scatter(P[idx, i], P[idx, j], c=col, s=0.6, cmap="coolwarm_r", vmin=-1, vmax=1, lw=0)
    a, b = G[k]
    ax.plot([a[i], b[i]], [a[j], b[j]], "-", color="#1a9c3a", lw=3, label=f"gold {k}")
    ax.plot([a[i], b[i]], [a[j], b[j]], "o", color="#1a9c3a", ms=8)
    ap = APEX[k]
    ax.plot(ap[:, i], ap[:, j], "-", color="k", lw=1.5, label=f"fitted outer apex line over fold extent ({EM[k]['fold_length_mm']:.1f} mm)")
    ax.plot(ap[:, i], ap[:, j], "|", color="k", ms=18, mew=2)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title + f"\ncoverage of fold by gold {EM[k]['coverage_of_fold']*100:.1f}%; colour = normal "
                 "toward interior (blue) / away (red)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "a1_extent_views.png", dpi=80)
print("wrote", OUT / "a1_extent_views.png")
