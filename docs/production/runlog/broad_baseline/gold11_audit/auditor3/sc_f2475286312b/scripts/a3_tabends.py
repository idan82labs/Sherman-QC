"""Tab end details: h=const sections through both tabs and top-view zoom of the window ends."""
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
fig, axes = plt.subplots(2, 3, figsize=(21, 9), dpi=95)
for row, (blo, bhi) in enumerate(((60, 95), (-100, -65))):
    for col, hv in enumerate((-3.0, -8.0, -12.0)):
        ax = axes[row, col]
        segs = slice_mesh(V, F, o + hv * n, n)
        P = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
        for s in P:
            if blo - 5 < s[0, 1] < bhi + 5 and -80 < s[0, 0] < 40:
                ax.plot(s[:, 0], s[:, 1], color="k", lw=0.8)
        m = (X[:, 1] > blo) & (X[:, 1] < bhi) & (X[:, 0] > -80) & (X[:, 0] < 40) & (np.abs(X[:, 2]) < 1.2)
        ax.scatter(X[m, 0], X[m, 1], s=0.2, c="#bbbbbb", linewidths=0)
        ax.set_xlim(-80, 40)
        ax.set_ylim(blo, bhi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"h = {hv} section (black) over plate points (grey); b in [{blo},{bhi}]", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_tab_ends.png")
plt.close(fig)
print("ok")
