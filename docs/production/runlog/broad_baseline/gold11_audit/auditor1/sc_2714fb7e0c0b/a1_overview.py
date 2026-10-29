"""Overview of sc_2714fb7e0c0b in the part frame (X along step bends, Z along main-panel long edges,
N = panel normal) with gold lines; plus a Z-N profile at mid X. Read-only on inputs."""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import a1common as A

pts, nrm, _ = A.load_scan()
g = A.load_gold()
ex, ez, en = A.part_frame(g)
c = pts.mean(0)
M = np.c_[ex, ez, en].T


def tf(p):
    return (np.asarray(p) - c) @ M.T


P = tf(pts)
G = {k: (tf(a), tf(b)) for k, (a, b) in g.items()}
rng = np.random.default_rng(1)
sub = P[rng.choice(len(P), 160000, replace=False)]
fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=90)
views = [("top X-Z (look along N)", 0, 1, 2), ("front X-N (look along Z)", 0, 2, 1), ("side Z-N (look along X)", 1, 2, 0)]
for ax, (name, i, j, k) in zip(axes.ravel()[:3], views):
    o = np.argsort(sub[:, k])
    ax.scatter(sub[o, i], sub[o, j], s=0.2, c=sub[o, k], cmap="viridis", linewidths=0, rasterized=True)
    for bid, (a, b) in G.items():
        ax.plot([a[i], b[i]], [a[j], b[j]], color="red", lw=1.6)
        m = 0.5 * (a + b)
        ax.text(m[i], m[j], bid, fontsize=8, color="k", bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.5))
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
# profile Z-N in a thin slab at X=0 and X=-60
ax = axes[1, 1]
for x0, col in ((0.0, "k"), (-81.5, "tab:blue"), (100.0, "tab:orange")):
    m = np.abs(P[:, 0] - x0) < 0.4
    ax.scatter(P[m, 1], P[m, 2], s=0.6, c=col, label=f"slab X={x0}")
for bid, (a, b) in G.items():
    if abs(a[0] - b[0]) > 50:  # lines along X show as points in this profile
        m = 0.5 * (a + b)
        ax.plot(m[1], m[2], "r*", ms=10)
        ax.text(m[1] + 2, m[2] + 2, bid, color="r", fontsize=8)
ax.set_title("Z-N profile slabs (lines along X shown as red stars)")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.suptitle("sc_2714fb7e0c0b part frame; red = gold B01..B10 (catalog shaked_xv/C06)")
fig.tight_layout()
fig.savefig(A.OUT / "a1_overview_partframe.png")
print("ok")
