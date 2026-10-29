"""Verifier2: inspect the ear regions (is the ear attached to the web by a fold? where does the ear bend end?)."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_sections import load_ply, unit, LINES, SCAN, HERE

cfg = {b["bend_id"]: b for b in json.load(open(LINES))["bends"]}
X, _ = load_ply(SCAN)
tree = cKDTree(X)
nw = unit([0.074209, -0.232659, 0.969723])
EARS = {"A": ("B5", "B3", [-0.915, 0.379961, 0.135667]), "B": ("B6", "B4", [0.044594, -0.937954, -0.343879])}
for nm, (be, bf, ne) in EARS.items():
    E = cfg[be]; F = cfg[bf]
    c = 0.5 * (np.array(E["label_line"]["start"]) + np.array(E["label_line"]["end"]))
    ex = unit(np.array(F["direction"]) - (np.array(F["direction"]) @ nw) * nw)
    ez = nw; ey = np.cross(ez, ex)
    idx = tree.query_ball_point(c, 26.0)
    P = X[idx] - c
    x, y, z = P @ ex, P @ ey, P @ ez
    def proj(p):
        p = np.asarray(p) - c; return p @ ex, p @ ey, p @ ez
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
    for ax, (a, b_, col, la, lb) in zip(axs, ((x, y, z, "x (tab fold axis)", "y"), (x, z, y, "x (tab fold axis)", "z (web normal)"), (y, z, x, "y", "z (web normal)"))):
        sc = ax.scatter(a, b_, c=col, s=1, cmap="viridis")
        plt.colorbar(sc, ax=ax, fraction=0.04)
        for bid, colr in ((be, "tab:red"), (bf, "tab:blue")):
            s = proj(cfg[bid]["label_line"]["start"]); e = proj(cfg[bid]["label_line"]["end"])
            ii = {"x": 0, "y": 1, "z": 2}
            ka = ii[la[0]]; kb = ii[lb[0]]
            ax.plot([s[ka], e[ka]], [s[kb], e[kb]], "-", color=colr, lw=2, label=bid)
            ax.plot(s[ka], s[kb], "o", color=colr); ax.plot(e[ka], e[kb], "s", color=colr)
        ax.set_aspect("equal"); ax.set_xlabel(la); ax.set_ylabel(lb); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"ear {nm}: scan pts within 26 mm of {be} midpoint (colour = third axis); o=start s=end of label lines")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"v2_ear{nm}_views.png"), dpi=80); plt.close(fig)
    # sections perpendicular to the would-be web-ear hinge axis
    a_ax = unit(np.cross(nw, np.array(ne)))
    # in-web-plane direction perpendicular to a
    b_ax = unit(np.cross(nw, a_ax))
    offs = np.arange(-10, 10.1, 2.0)
    fig, axs = plt.subplots(2, 6, figsize=(24, 9)); axs = axs.ravel()
    for k, o in enumerate(offs):
        ax = axs[k]
        sl = np.abs(P @ a_ax - o) < 0.4
        ax.scatter(P[sl] @ b_ax, P[sl] @ nw, s=3, c="0.3")
        ax.set_title(f"ear {nm}: slab at {o:+.0f} mm along web-ear axis", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)
        ax.set_xlabel("in-web dir perp. to web-ear axis"); ax.set_ylabel("web normal")
    axs[-1].axis("off")
    fig.suptitle(f"ear {nm}: sections perpendicular to the web/ear plane-intersection axis (would show a 90 deg arc if the ear top were folded from the web)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"v2_ear{nm}_hinge_sections.png"), dpi=75); plt.close(fig)
print("done")
