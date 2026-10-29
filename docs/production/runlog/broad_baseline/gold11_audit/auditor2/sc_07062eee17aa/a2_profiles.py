"""Auditor 2, sc_07062eee17aa: whole-part frame + stacked cross-section profiles along the gold axis.

Frame: d = mean gold direction, w = main-plate normal (from scan normals), u = w x d.
Writes PNGs into this directory only. Read-only on scan and gold.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c  # noqa: E402

v, n, f = c.load_scan()
g = c.load_gold()
d = c.frame_from_gold(g)
nn = n / np.linalg.norm(n, axis=1, keepdims=True)
t = g["M03"][0] - g["M02"][0]
tu = t / np.linalg.norm(t)
cosv = nn @ tu
sel = np.abs(cosv) > np.cos(np.radians(10))
m = nn[sel] * np.sign(cosv[sel])[:, None]
w_, V = np.linalg.eigh(m.T @ m)
w = V[:, -1] * np.sign(V[:, -1] @ tu)
w = w - (w @ d) * d
w /= np.linalg.norm(w)
u = np.cross(w, d)
u /= np.linalg.norm(u)
o = v.mean(0)
P = np.c_[(v - o) @ u, (v - o) @ w, (v - o) @ d]  # (u, w, s)


def proj(x):
    x = np.asarray(x, float) - o
    return np.array([x @ u, x @ w, x @ d])


G = {k: (proj(s), proj(e)) for k, (s, e) in g.items()}
frame = {"origin": o.tolist(), "u": u.tolist(), "w": w.tolist(), "d": d.tolist()}
(c.OUT / "a2_frame.json").write_text(json.dumps(frame, indent=1))

print("s range of scan", P[:, 2].min(), P[:, 2].max())
for k, (a, b) in G.items():
    print(k, "u,w,s start", np.round(a, 2), "end", np.round(b, 2))


def gold_uw_at(k, s):
    a, b = G[k]
    lam = (s - a[2]) / (b[2] - a[2])
    p = a + lam * (b - a)
    return p[:2], lam


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "overview"
    if mode == "overview":
        # whole-part end view (projection along d) + plan views
        fig, ax = plt.subplots(2, 1, figsize=(16, 14))
        idx = np.random.default_rng(0).choice(len(P), 200000, replace=False)
        ax[0].scatter(P[idx, 0], P[idx, 1], s=0.2, c="k", alpha=0.3)
        for k, (a, b) in G.items():
            ax[0].plot([a[0], b[0]], [a[1], b[1]], "-o", ms=3, label=k)
            ax[0].annotate(k, a[:2], fontsize=9)
        ax[0].set_aspect("equal")
        ax[0].legend()
        ax[0].set_title("end view along gold axis d (u horizontal, w = plate normal vertical)")
        ax[1].scatter(P[idx, 2], P[idx, 0], s=0.2, c="k", alpha=0.3)
        for k, (a, b) in G.items():
            ax[1].plot([a[2], b[2]], [a[0], b[0]], "-", lw=1.5, label=k)
            ax[1].annotate(k, (a[2], a[0]), fontsize=9)
        ax[1].set_aspect("equal")
        ax[1].set_title("plan view (s along d horizontal, u vertical)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(c.OUT / "a2_overview_frame.png", dpi=110)
        print("wrote overview")
