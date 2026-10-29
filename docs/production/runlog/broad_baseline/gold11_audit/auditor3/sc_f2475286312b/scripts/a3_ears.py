"""h = const sections at both wall ends (ears) and a = const sections across the wall bottom."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3common import load_mesh, load_gold, slice_mesh, OUT

V, N, F = load_mesh()
G = load_gold()
fr = json.loads((OUT / "frame.json").read_text())
o, a, b, n = (np.array(fr[k]) for k in ("origin", "a", "b", "n"))
R = np.array([a, b, n])
Gf = {k: ((s - o) @ R.T, (e - o) @ R.T) for k, (s, e) in G.items()}
VEC = {"a": a, "b": b, "h": n}
AX = {"a": 0, "b": 1, "h": 2}


def draw(ax, cut, val, x, y, xl, yl, title):
    segs = slice_mesh(V, F, o + val * VEC[cut], VEC[cut])
    if len(segs):
        P = ((segs.reshape(-1, 3) - o) @ R.T).reshape(-1, 2, 3)
        for s in P:
            ax.plot(s[:, AX[x]], s[:, AX[y]], color="k", lw=0.7)
    ci = AX[cut]
    for k, (s, e) in Gf.items():
        if abs(e[ci] - s[ci]) < 1e-6:
            # line lies in a constant-cut plane: draw its projection if within 3 mm of the cut
            if abs(s[ci] - val) < 3:
                ax.plot([s[AX[x]], e[AX[x]]], [s[AX[y]], e[AX[y]]], color="r", lw=1)
            continue
        t = (val - s[ci]) / (e[ci] - s[ci])
        if -0.02 <= t <= 1.02:
            p = s + t * (e - s)
            ax.plot(p[AX[x]], p[AX[y]], "*", color="r", ms=10, mec="k", mew=0.4)
            ax.annotate(k, (p[AX[x]], p[AX[y]]), fontsize=7, color="r", xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(x, fontsize=7)
    ax.set_ylabel(y, fontsize=7)
    ax.tick_params(labelsize=6)


fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=95)
for i, hv in enumerate([-5, -20, -35, -50]):
    draw(axes[0, i], "h", hv, "a", "b", (100, 132), (96, 120), f"ear +b end: h = {hv}")
    draw(axes[1, i], "h", hv, "a", "b", (100, 132), (-120, -96), f"ear -b end: h = {hv}")
fig.suptitle("sc_f2475286312b h=const sections at the wall ends (a horizontal, b vertical); red = gold crossings", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_ears_hconst.png")
plt.close(fig)

# ear outline: b = const sections through the ears (a horizontal, h vertical)
fig, axes = plt.subplots(1, 4, figsize=(20, 8), dpi=95)
for ax, bv in zip(axes, [111.5, 113.0, -110.0, -111.5]):
    draw(ax, "b", bv, "a", "h", (95, 135), (-66, 6), f"b = {bv}")
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_ears_bconst.png")
plt.close(fig)

# wall bottom: a = const sections near the wall bottom, looking along b (b horizontal, h vertical)
fig, axes = plt.subplots(3, 1, figsize=(18, 9), dpi=95)
for ax, av in zip(axes, [112.5, 113.0, 116.5]):
    draw(ax, "a", av, "b", "h", (-118, 118), (-66, 4), f"a = {av}")
fig.tight_layout()
fig.savefig(OUT / "a3_zoom_wall_aconst.png")
plt.close(fig)
print("ok")
