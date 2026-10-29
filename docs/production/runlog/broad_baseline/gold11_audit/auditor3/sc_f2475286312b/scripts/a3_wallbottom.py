"""Inspect the wall's bottom edge (h < -57) for any return flange / tab near M08."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3common import load_mesh, load_gold, slice_mesh, OUT

V, N, F = load_mesh()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
X = (V - o) @ R.T
G = load_gold()
s8, e8 = ((G["M08"][0] - o) @ R.T, (G["M08"][1] - o) @ R.T)
m = (X[:, 2] < -55) & (X[:, 0] > 105)
Q = X[m]
print("points with h<-55, a>105:", len(Q))
# points beyond the wall's outer face (a > 114.2) near the bottom
out = Q[(Q[:, 0] > 114.2)]
print("of which a>114.2 (outside the wall face):", len(out))
if len(out):
    print("  b range %.1f..%.1f  a range %.1f..%.1f  h range %.1f..%.1f" % (out[:, 1].min(), out[:, 1].max(), out[:, 0].min(), out[:, 0].max(), out[:, 2].min(), out[:, 2].max()))
below = Q[Q[:, 2] < -61.2]
print("points below h=-61.2:", len(below), (below[:, 1].min(), below[:, 1].max()) if len(below) else "")
fig, axes = plt.subplots(2, 1, figsize=(18, 8), dpi=95)
ax = axes[0]
ax.scatter(Q[:, 1], Q[:, 2], s=0.5, c=Q[:, 0], cmap="viridis", vmin=110, vmax=118)
ax.plot([s8[1], e8[1]], [s8[2], e8[2]], color="r", lw=1, label="M08")
ax.set_xlabel("b"); ax.set_ylabel("h"); ax.set_title("wall bottom region (colour = a); red = M08", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.scatter(Q[:, 1], Q[:, 0], s=0.5, c=Q[:, 2], cmap="viridis")
ax.plot([s8[1], e8[1]], [s8[0], e8[0]], color="r", lw=1)
ax.set_xlabel("b"); ax.set_ylabel("a"); ax.set_title("wall bottom region seen from below (colour = h)", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_wall_bottom.png")
plt.close(fig)
