"""Corner region close-ups (side-wall -> side-flange fold near the top flange), read-only."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from a3_common import OUT, load_gold, load_points

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = json.loads((OUT / "part_frame.json").read_text())
    c0 = np.array(fr["origin"]); M = np.array([fr["X"], fr["Y"], fr["Z"]])
    P = (load_points() - c0) @ M.T
    G = {b: ((s - c0) @ M.T, (e - c0) @ M.T) for b, s, e in load_gold()}
    ys = [95, 104, 108, 112, 118, 126, 134, 142, 146]
    fig, axes = plt.subplots(len(ys), 2, figsize=(11, 3.0 * len(ys)), dpi=90)
    for row, yv in zip(axes, ys):
        m0 = np.abs(P[:, 1] - yv) < 0.5
        for ax, (lo, hi), gid in zip(row, ((-160, -140), (140, 160)), ("HB5", "HB6")):
            m = m0 & (P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 2] > 28) & (P[:, 2] < 45)
            ax.scatter(P[m, 0], P[m, 2], s=3, c="#222", linewidths=0)
            s, e = G[gid]
            if (s[1] - yv) * (e[1] - yv) <= 0:
                q = s + (yv - s[1]) / (e[1] - s[1]) * (e - s)
                ax.plot(q[0], q[2], "*", color="#1a9c3a", ms=14, mec="k")
                ax.add_patch(plt.Circle((q[0], q[2]), 4.0, fill=False, ls=":", color="#1a9c3a"))
            ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(28, 45); ax.grid(alpha=0.3)
            ax.set_title(f"{gid} side, section Y={yv} (slab +-0.5), X vs Z; star = gold, dotted = 4 mm", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "a3_corner_side_flange.png"); plt.close(fig)

if __name__ == "__main__":
    main()
