"""Overview renders in the part frame (L long, S short, H height): top view coloured by H,
and full-width sections perpendicular to L and to S, with gold lines marked."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *

P, N = load_pts()
c = P.mean(0)
Q = to_frame(P, c)
NQ = to_frame(N)
gold = [(b, to_frame(a, c), to_frame(e, c)) for b, a, e in load_gold()]
np.save(SCR / "Q.npy", Q); np.save(SCR / "NQ.npy", NQ); np.save(SCR / "center.npy", c)

rng = np.random.default_rng(0)
sub = rng.choice(len(Q), 300000, replace=False)
fig, ax = plt.subplots(1, 2, figsize=(22, 8), dpi=90)
sc = ax[0].scatter(Q[sub, 0], Q[sub, 1], c=Q[sub, 2], s=0.2, cmap="viridis", linewidths=0, rasterized=True)
plt.colorbar(sc, ax=ax[0], label="H (mm)")
for b, a, e in gold:
    ax[0].plot([a[0], e[0]], [a[1], e[1]], "r-", lw=1.5)
    ax[0].text((a[0] + e[0]) / 2, (a[1] + e[1]) / 2, b, color="r", fontsize=9)
ax[0].set_aspect("equal"); ax[0].set_title("top view L-S coloured by H"); ax[0].grid(alpha=.3)
# view from below
sc = ax[1].scatter(Q[sub, 0], Q[sub, 1], c=np.abs(NQ[sub, 2]), s=0.2, cmap="coolwarm", linewidths=0, rasterized=True)
plt.colorbar(sc, ax=ax[1], label="|n_H|")
ax[1].set_aspect("equal"); ax[1].set_title("top view coloured by |normal H| (1 = horizontal face)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "a3_topview.png"); plt.close(fig)

# sections perpendicular to L (S-H plane) at several L, and perpendicular to S (L-H plane)
fig, axes = plt.subplots(2, 3, figsize=(24, 9), dpi=85)
for ax, l0 in zip(axes.ravel(), (-120, -60, 0, 40, 100, 150)):
    m = np.abs(Q[:, 0] - l0) < 0.6
    ax.scatter(Q[m, 1], Q[m, 2], s=0.6, c="k", linewidths=0)
    for b, a, e in gold:
        d = e - a
        if abs(d[0]) > 50:  # long-side gold lines cross this section
            t = (l0 - a[0]) / d[0]
            if 0 <= t <= 1:
                p = a + t * d
                ax.plot(p[1], p[2], "r*", ms=10); ax.annotate(b, (p[1], p[2]), color="r", fontsize=8)
    ax.set_aspect("equal"); ax.set_title(f"section L={l0} (S-H)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "a3_sections_perpL.png"); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(26, 8), dpi=85)
for ax, s0 in zip(axes.ravel(), (-80, -20, 30, 90)):
    m = np.abs(Q[:, 1] - s0) < 0.6
    ax.scatter(Q[m, 0], Q[m, 2], s=0.6, c="k", linewidths=0)
    for b, a, e in gold:
        d = e - a
        if abs(d[1]) > 50:
            t = (s0 - a[1]) / d[1]
            if 0 <= t <= 1:
                p = a + t * d
                ax.plot(p[0], p[2], "r*", ms=10); ax.annotate(b, (p[0], p[2]), color="r", fontsize=8)
    ax.set_aspect("equal"); ax.set_title(f"section S={s0} (L-H)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "a3_sections_perpS.png"); plt.close(fig)
print("ok")
