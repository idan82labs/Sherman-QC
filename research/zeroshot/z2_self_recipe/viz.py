"""Debug overlays: points coloured by flange, gold lines (green) and predicted lines (red)."""
from __future__ import annotations

import numpy as np


def render(scan, out_png: str, gold=None, pred=None, title: str = "", max_pts: int = 120_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = scan.P
    lab = scan.flange_of_point
    sel = np.arange(len(P))
    if len(sel) > max_pts:
        sel = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
    c = P.mean(0)
    Q = P - c
    # PCA view axes
    _, _, vt = np.linalg.svd(Q[sel[:20000]], full_matrices=False)
    views = [(vt[0], vt[1]), (vt[0], vt[2]), (vt[1], vt[2])]
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    rng = np.random.default_rng(1)
    cols = rng.random((max(1, lab.max() + 1), 3)) * 0.8 + 0.1
    for ax, (ex, ey) in zip(axs, views):
        q = Q[sel]
        l = lab[sel]
        un = l < 0
        ax.scatter(q[un] @ ex, q[un] @ ey, s=0.2, c="0.8")
        ax.scatter(q[~un] @ ex, q[~un] @ ey, s=0.2, c=cols[l[~un]])
        for lines, col in ((gold or [], "g"), (pred or [], "r")):
            for ln in lines:
                a = np.asarray(ln["line_start"]) - c
                b = np.asarray(ln["line_end"]) - c
                ax.plot([a @ ex, b @ ex], [a @ ey, b @ ey], col + "-", lw=1.5)
                ax.text(((a + b) / 2) @ ex, ((a + b) / 2) @ ey, str(ln.get("bend_id", ""))[:8], color=col, fontsize=7)
        ax.set_aspect("equal")
    for f in scan.flanges:
        m = P[f.idx].mean(0) - c
        axs[0].text(m @ views[0][0], m @ views[0][1], str(f.fid), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=70)
    plt.close(fig)
