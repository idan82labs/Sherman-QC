"""Part frame for sc_cec08dfca9ab: floor plane fit + in-plane axes along HB1 (X) and HB3 (Y).

Writes part_frame.json and a height-coded top view (top_heightmap.png) with gold lines.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json

import numpy as np

from a3_common import OUT, load_gold, load_points, unit


def fit_plane(Q):
    c = Q.mean(0)
    _u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    return c, vt[2], s


def main():
    P = load_points()
    G = {b: (s, e) for b, s, e in load_gold()}
    d1 = unit(G["HB1"][1] - G["HB1"][0])
    d3 = unit(G["HB3"][1] - G["HB3"][0])
    n0 = unit(np.cross(d1, d3))
    # seed: plane through the HB1/HB3/HB4 gold lines (floor-level lines)
    seed_pts = np.vstack([G[k][0] for k in ("HB1", "HB3", "HB4")] + [G[k][1] for k in ("HB1", "HB3", "HB4")])
    c0 = seed_pts.mean(0)
    # iterate: take points within 3 mm of plane and inside the footprint, refit, tighten
    c, n = c0, n0
    for tol in (6.0, 3.0, 1.5, 1.0):
        h = (P - c) @ n
        sel = np.abs(h) < tol
        c, n, s = fit_plane(P[sel])
        if n @ n0 < 0:
            n = -n
    h = (P - c) @ n
    # sheet: the floor has two skins ~t apart; report both
    hist, edges = np.histogram(h, bins=400, range=(-10, 50))
    X = unit(d1 - (d1 @ n) * n)
    Y = np.cross(n, X)
    if Y @ d3 < 0:
        Y = -Y
        X = -X
    # recompute so frame right-handed with Z = n
    Z = np.cross(X, Y)
    frame = {"origin": c.tolist(), "X": X.tolist(), "Y": Y.tolist(), "Z": Z.tolist(),
             "note": "Z = fitted floor-plane normal (pointing toward the walls), X along HB1, Y along HB3"}
    (OUT / "part_frame.json").write_text(json.dumps(frame, indent=1))
    L = (P - c) @ np.array([X, Y, Z]).T
    print("frame", json.dumps(frame, indent=1))
    print("local bbox", L.min(0).round(1), L.max(0).round(1))
    # height histogram peaks
    pk = np.argsort(hist)[::-1][:12]
    print("height peaks", sorted([(round(float((edges[i] + edges[i + 1]) / 2), 2), int(hist[i])) for i in pk]))
    for b, (s, e) in G.items():
        ls, le = (s - c) @ np.array([X, Y, Z]).T, (e - c) @ np.array([X, Y, Z]).T
        print(b, "start", ls.round(2), "end", le.round(2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    sub = L[rng.choice(len(L), 400000, replace=False)]
    fig, ax = plt.subplots(figsize=(14, 13), dpi=100)
    o = np.argsort(sub[:, 2])
    sc = ax.scatter(sub[o, 0], sub[o, 1], c=sub[o, 2], s=0.3, cmap="turbo", vmin=-3, vmax=40, linewidths=0, rasterized=True)
    for b, (s, e) in G.items():
        ls, le = (s - c) @ np.array([X, Y, Z]).T, (e - c) @ np.array([X, Y, Z]).T
        ax.plot([ls[0], le[0]], [ls[1], le[1]], color="k", lw=1.2)
        m = 0.5 * (ls + le)
        ax.text(m[0], m[1], b, fontsize=9, color="w", bbox=dict(fc="k", ec="none", pad=1))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="height above floor plane (mm)")
    ax.set_title("sc_cec08dfca9ab top view in floor frame, colour = height above floor plane; black = gold lines")
    fig.tight_layout()
    fig.savefig(OUT / "a3_top_heightmap.png")


if __name__ == "__main__":
    main()
