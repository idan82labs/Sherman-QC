"""Top view (u-v) coloured by w + gold/pred lines; side profiles at several v and u."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a2_common import load, gold_lines, pred_lines, OUT

q, R, O, raw = load()
G = gold_lines(R, O)
P = pred_lines(R, O)
C = pred_lines(R, O, "predictions_cad_raw_ungated.json")
rng = np.random.default_rng(0)
sub = q[rng.choice(len(q), 250000, replace=False)]
fig, ax = plt.subplots(1, 1, figsize=(16, 14), dpi=90)
o = np.argsort(sub[:, 2])
sc = ax.scatter(sub[o, 0], sub[o, 1], c=sub[o, 2], s=0.3, cmap="viridis", linewidths=0, rasterized=True)
plt.colorbar(sc, ax=ax, label="w (mm, + = above main plate outer... sign as frame)")
for k, (s, e) in G.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], color="#d62728", lw=2.2)
    m = (s + e) / 2
    ax.text(m[0], m[1], k, color="white", fontsize=9, bbox=dict(fc="#d62728", ec="none", pad=0.5))
for k, (s, e) in P.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], color="orange", lw=1, ls="--")
for k, (s, e) in C.items():
    ax.plot([s[0], e[0]], [s[1], e[1]], color="cyan", lw=1, ls=":")
    ax.text(e[0], e[1], k, color="c", fontsize=7)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("u (along B01)"); ax.set_ylabel("v (along B03)")
ax.set_title("sc_2714fb7e0c0b part frame top view, colour = w; red = gold, orange dashed = best_f1, cyan dotted = CAD raw")
fig.tight_layout(); fig.savefig(OUT / "a2_top_uvw.png"); plt.close(fig)

# profiles: u-w at fixed v ; v-w at fixed u
fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=90)
for ax, v0 in zip(axes[0], (12.0, 112.0, 212.0)):
    m = np.abs(q[:, 1] - v0) < 0.6
    ax.scatter(q[m, 0], q[m, 2], s=0.4, c="k", linewidths=0)
    for k, (s, e) in G.items():
        d = e - s
        if abs(d[1]) > 0.5 * np.linalg.norm(d):  # line runs along v -> crosses this slice
            t = (v0 - s[1]) / d[1]
            if -0.05 <= t <= 1.05:
                x = s + t * d
                ax.plot(x[0], x[2], "r*", ms=10); ax.text(x[0] + 1, x[2] + 1, k, color="r", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(f"u-w profile at v={v0}")
for ax, u0 in zip(axes[1], (-180.0, -98.0, 25.0)):
    m = np.abs(q[:, 0] - u0) < 0.6
    ax.scatter(q[m, 1], q[m, 2], s=0.4, c="k", linewidths=0)
    for k, (s, e) in G.items():
        d = e - s
        if abs(d[0]) > 0.5 * np.linalg.norm(d):
            t = (u0 - s[0]) / d[0]
            if -0.05 <= t <= 1.05:
                x = s + t * d
                ax.plot(x[1], x[2], "r*", ms=10); ax.text(x[1] + 1, x[2] + 1, k, color="r", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(f"v-w profile at u={u0}")
fig.tight_layout(); fig.savefig(OUT / "a2_profiles.png"); plt.close(fig)
print("ok")
