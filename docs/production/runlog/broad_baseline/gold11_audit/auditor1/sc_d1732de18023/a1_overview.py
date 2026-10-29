"""Overview renders in the PCA frame, points coloured by normal class, gold lines overlaid."""
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

# colour by dominant normal axis and sign
cls = np.argmax(np.abs(N), 1)
sgn = np.sign(N[np.arange(len(N)), cls])
cols = np.zeros((len(N), 3))
pal = {(0, 1): (0.85, 0.2, 0.2), (0, -1): (0.55, 0.0, 0.0), (1, 1): (0.2, 0.6, 0.2), (1, -1): (0.0, 0.35, 0.0),
       (2, 1): (0.3, 0.45, 0.9), (2, -1): (0.1, 0.1, 0.45)}
for (a, s), col in pal.items():
    cols[(cls == a) & (sgn == s)] = col
weak = np.max(np.abs(N), 1) < 0.95  # curved / arc points
cols[weak] = (1.0, 0.65, 0.0)


def view(ax, i, j, depth_axis, depth_sign, sub=4, title="", xlim=None, ylim=None, mask=None, s=0.3):
    m = np.ones(len(P), bool) if mask is None else mask
    idx = np.where(m)[0][::sub]
    order = np.argsort(depth_sign * P[idx, depth_axis])  # far first, near last
    idx = idx[order]
    ax.scatter(P[idx, i], P[idx, j], c=cols[idx], s=s, linewidths=0)
    for k, (a, b) in G.items():
        ax.plot([a[i], b[i]], [a[j], b[j]], "-", color="k", lw=1.2)
        ax.text((a[i] + b[i]) / 2, (a[j] + b[j]) / 2, k, fontsize=8, color="k",
                bbox=dict(fc="yellow", alpha=0.7, lw=0))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)


fig, axs = plt.subplots(2, 2, figsize=(18, 13))
view(axs[0, 0], 0, 1, 2, +1, title="top: PC1-PC2 seen from +PC3 (near drawn last)")
view(axs[0, 1], 0, 1, 2, -1, title="bottom: PC1-PC2 seen from -PC3")
view(axs[1, 0], 0, 2, 1, -1, title="front: PC1-PC3 seen from -PC2", ylim=(-20, 25))
view(axs[1, 1], 1, 2, 0, +1, title="side: PC2-PC3 seen from +PC1", ylim=(-20, 25))
fig.suptitle("sc_d1732de18023 auditor1 overview; colours: blue=+-PC3 plate skins, green=+-PC2 walls, "
             "red=+-PC1 lips, orange=curved (|n|max<0.95); black = gold", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "a1_overview.png", dpi=110)
print("wrote", OUT / "a1_overview.png")

# end close-ups: M01 end (PC1 < -120) and M04 end (PC1 > 115)
fig, axs = plt.subplots(2, 2, figsize=(18, 13))
mL = P[:, 0] < -110
mR = P[:, 0] > 110
view(axs[0, 0], 1, 2, 0, -1, sub=1, title="M01 end (PC1<-110) seen from -PC1 (outside), PC2-PC3", mask=mL,
     ylim=(-20, 22), s=0.5)
view(axs[0, 1], 1, 2, 0, +1, sub=1, title="M04 end (PC1>110) seen from +PC1 (outside), PC2-PC3", mask=mR,
     ylim=(-20, 22), s=0.5)
view(axs[1, 0], 0, 1, 2, -1, sub=1, title="M01 end seen from below (-PC3)", mask=mL, s=0.5)
view(axs[1, 1], 0, 1, 2, -1, sub=1, title="M04 end seen from below (-PC3)", mask=mR, s=0.5)
fig.tight_layout()
fig.savefig(OUT / "a1_ends.png", dpi=110)
print("wrote", OUT / "a1_ends.png")
