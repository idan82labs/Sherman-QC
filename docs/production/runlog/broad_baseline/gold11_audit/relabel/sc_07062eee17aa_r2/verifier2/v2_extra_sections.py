"""Verifier2: extra perpendicular sections at chosen t (B2 bow, fold-end checks)."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_sections import load_ply, unit, station, LINES, SCAN, HERE

PLAN = {
    "B2": [2, 20, 40, 95, 120, 170, 230, 270, 300],
    "B3": [-8, -6, -4, -2, 0, 1, 3, 76, 78, 80],
    "B4": [-8, -6, -4, -2, 0, 1, 3, 76, 78, 80],
    "B5": [-1.0, -0.5, 0.5, 11.0, 12.0, 12.5, 13.0, 13.5, 14.5],
    "B6": [-1.0, -0.5, 0.5, 11.0, 11.6, 12.5, 13.0, 13.5, 14.5],
    "B1": [-2, 0, 44, 46, 86, 88, 304, 306, 308],
}

cfg = {b["bend_id"]: b for b in json.load(open(LINES))["bends"]}
X, _ = load_ply(SCAN)
tree = cKDTree(X)
for bid, tl in PLAN.items():
    b = cfg[bid]
    ear = bid in ("B5", "B6")
    Ls = np.array(b["label_line"]["start"]); Le = np.array(b["label_line"]["end"])
    Ss = np.array(b["sharp"]["start"]); Se = np.array(b["sharp"]["end"])
    Llen = np.linalg.norm(Le - Ls); d = unit(Le - Ls)
    u = np.array(b["bisector"]); u = unit(u - (u @ d) * d); w = np.cross(d, u)
    def p2(v):
        v = np.asarray(v); v = v - (v @ d) * d
        return unit(np.array([v @ w, v @ u]))
    dir_a = p2(b["side_dirs"]["plane_a"]); dir_b = p2(b["side_dirs"]["plane_b"])
    def nrm(da, db):
        n = np.array([-da[1], da[0]]); return n if n @ db < 0 else -n
    n_a = nrm(dir_a, dir_b); n_b = nrm(dir_b, dir_a)
    incl = np.degrees(np.arccos(np.clip(dir_a @ dir_b, -1, 1)))
    T = 2.3 / np.tan(np.radians(incl / 2))
    leg_len = 4.0 if ear else 8.0; h = 0.4 if ear else 0.6; R = 9.0 if ear else 14.0
    zoom = 7.0 if ear else 10.0
    n = len(tl); nc = 5 if n > 6 else n; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.4 * nr), squeeze=False)
    for k, t in enumerate(tl):
        ax = axs[k // nc, k % nc]
        P0 = Ls + t * d; Sh = Ss + (t / Llen) * (Se - Ss)
        sh2 = np.array([(Sh - P0) @ w, (Sh - P0) @ u])
        r = station(tree, X, P0, d, u, w, sh2, dir_a, dir_b, n_a, n_b, T, leg_len, h, R)
        if r is None:
            ax.set_title(f"t={t}: empty"); continue
        q = r["pts"]
        ax.scatter(q[:, 0], q[:, 1], s=3, c="0.4")
        ax.plot(*sh2, "kx", ms=10, mew=2)
        if "sharp_loc" in r:
            ax.plot(*r["sharp_loc"], "+", color="tab:orange", ms=12, mew=2)
        if "apex_loc" in r:
            ax.plot(*r["apex_loc"], "o", mfc="none", mec="tab:green", ms=12, mew=2)
        ax.plot(0, 0, "o", color="tab:red", ms=6)
        # draw label outer-skin leg lines from label sharp
        for dr in (dir_a, dir_b):
            ax.plot([sh2[0], sh2[0] + 12 * dr[0]], [sh2[1], sh2[1] + 12 * dr[1]], ":", color="tab:blue", lw=0.8)
        ttl = f"{bid} t={t} ({t/Llen*100:.0f}%) n={r['n']}"
        if "off_bis" in r:
            ttl += f"\nbis={r['off_bis']:+.2f} lat={r['off_lat']:+.2f} incl={r['angle_incl']:.1f}"
        la, lb = r["legs"].get("a"), r["legs"].get("b")
        ttl += f"\nlegs a={0 if la is None else la[2]} b={0 if lb is None else lb[2]}"
        ax.set_title(ttl, fontsize=8)
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom * 0.6, zoom * 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    for k in range(n, nr * nc):
        axs[k // nc, k % nc].axis("off")
    fig.suptitle(f"verifier2 {bid} extra sections (red=label pt, x=label sharp, +=my local sharp, green=my outer apex, blue dotted=label outer legs)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"v2_extra_{bid}.png"), dpi=80); plt.close(fig)
print("done")
