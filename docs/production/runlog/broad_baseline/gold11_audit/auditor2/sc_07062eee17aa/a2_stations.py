"""Auditor 2, sc_07062eee17aa: dense perpendicular cross-sections along each gold line.

For each gold line, stations every STEP mm along the line; slab +-HALF mm perpendicular to the
line; points shown in the part frame (u horizontal, w = plate normal vertical), centred on the
gold point. Writes a contact sheet per line: a2_stations_<id>.png
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c  # noqa: E402
from a2_profiles import G, P, d, u, w, o, g  # noqa: E402

HALF = 0.5
WIN = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
ids = sys.argv[1].split(",") if len(sys.argv) > 1 else list(G)

for k in ids:
    a, b = G[k]
    L = np.linalg.norm(b - a)
    # stations: every ~25 mm including both ends' neighbourhood
    lams = np.linspace(0.02, 0.98, 13)
    fig, axs = plt.subplots(3, 5, figsize=(25, 15.5))
    axs = axs.ravel()
    for i, lam in enumerate(lams):
        p = a + lam * (b - a)  # (u, w, s)
        m = (np.abs(P[:, 2] - p[2]) < HALF) & (np.abs(P[:, 0] - p[0]) < WIN) & (np.abs(P[:, 1] - p[1]) < WIN)
        q = P[m]
        ax = axs[i]
        ax.scatter(q[:, 0] - p[0], q[:, 1] - p[1], s=1.5, c="k")
        # other gold lines at this s
        for kk, (aa, bb) in G.items():
            ll = (p[2] - aa[2]) / (bb[2] - aa[2])
            if -0.05 <= ll <= 1.05:
                pp = aa + ll * (bb - aa)
                if abs(pp[0] - p[0]) < WIN and abs(pp[1] - p[1]) < WIN:
                    ax.plot(pp[0] - p[0], pp[1] - p[1], "*", ms=14 if kk == k else 9,
                            mec="k", mfc="#1a9c3a" if kk == k else "#bbbbbb")
                    ax.annotate(kk, (pp[0] - p[0], pp[1] - p[1]), fontsize=8, xytext=(4, 4),
                                textcoords="offset points")
        circ = plt.Circle((0, 0), 4.0, fill=False, ls=":", color="#1a9c3a")
        ax.add_patch(circ)
        ax.set_xlim(-WIN, WIN)
        ax.set_ylim(-WIN, WIN)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{k} t={lam:.2f} s={p[2]:.0f} mm  n={len(q)}", fontsize=10)
    axs[-2].axis("off")
    axs[-1].axis("off")
    fig.suptitle(f"sc_07062eee17aa auditor2: {k} cross-sections perpendicular to d (u right, w=plate normal up),"
                 f" slab +-{HALF} mm, gold point at origin, dotted = 4 mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(c.OUT / f"a2_stations_{k}.png", dpi=80)
    plt.close(fig)
    print("wrote", k)
