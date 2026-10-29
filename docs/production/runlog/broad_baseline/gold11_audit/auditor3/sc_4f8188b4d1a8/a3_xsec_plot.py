"""Auditor-3: draw selected cross-section stations with the independent fit (outer/inner sharps, outer apex, gold).

usage: a3_xsec_plot.py NAME BID:s[,s...] [BID:s ...]   (s = mm along the gold line from line_start)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402
import a3_xsec as X  # noqa: E402


def main(argv):
    name = argv[0]
    req = []
    for tok in argv[1:]:
        bid, ss = tok.split(":")
        req += [(bid, float(s)) for s in ss.split(",")]
    p, n, _ = c.load_scan()
    gold = {g["id"]: g for g in c.load_gold()[0]}
    ncol = min(4, len(req))
    nrow = (len(req) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 6.4 * nrow), squeeze=False)
    cache = {}
    for ax, (bid, sk) in zip(axs.ravel(), req):
        g = gold[bid]
        L = float(np.linalg.norm(g["e"] - g["s"]))
        d, e1, e2 = c.basis(g["e"] - g["s"])
        if bid not in cache:
            cache[bid] = X.leg_normals(p, n, g["s"], d, e1, e2, L)[0]
        nA, nB = cache[bid]
        rel = p - g["s"]
        t = rel @ d
        u, v = rel @ e1, rel @ e2
        W = X.WIN
        m = (np.abs(t - sk) < X.SLAB) & (np.abs(u) < W) & (np.abs(v) < W)
        p2 = np.stack([u[m], v[m]], 1)
        n2 = np.stack([n[m] @ e1, n[m] @ e2], 1)
        r = X.station(p2, n2, nA, nB)
        # colour: leg A / leg B / other
        la = np.abs(n2 @ nA) > X.COS_LEG
        lb = np.abs(n2 @ nB) > X.COS_LEG
        ax.scatter(p2[~(la | lb), 0], p2[~(la | lb), 1], s=4, c="0.35", label="arc/other")
        ax.scatter(p2[la, 0], p2[la, 1], s=3, c="tab:blue", label="leg A")
        ax.scatter(p2[lb, 0], p2[lb, 1], s=3, c="tab:orange", label="leg B")
        ax.plot([0], [0], "*", ms=16, mfc="limegreen", mec="k", label=f"gold {bid}")
        ttl = f"{bid} s={sk:.0f}/{L:.0f} mm"
        if r.get("O") is not None:
            O = np.array(r["O"])
            ax.plot(*O, "x", c="navy", ms=10, mew=2, label="outer virtual sharp")
        if r.get("I") is not None:
            I = np.array(r["I"])
            ax.plot(*I, "x", c="purple", ms=10, mew=2, label="inner virtual sharp")
        if r["ok"]:
            A = np.array(r["A"])
            b = np.array(r["b"])
            ax.plot(*A, "D", c="cyan", mec="k", ms=8, label="outer apex (measured)")
            ax.plot([O[0] - 2 * b[0], O[0] + 8 * b[0]], [O[1] - 2 * b[1], O[1] + 8 * b[1]], ":", c="k", lw=0.8)
            for dd in r["dirs"]:
                dd = np.array(dd)
                ax.plot([O[0], O[0] + 12 * dd[0]], [O[1], O[1] + 12 * dd[1]], "--", c="navy", lw=0.6)
            circ = plt.Circle(A, 4.0, fill=False, ls=":", color="green")
            ax.add_patch(circ)
            ttl += (f"\nbend {r['bend_deg']:.1f}  gold-apex: along-bis {r['gold_minus_apex_along_bisector']:+.2f}"
                    f"  across {r['gold_minus_apex_across']:+.2f}  |d| {r['gold_to_apex_mm']:.2f}")
        else:
            ttl += f"\nFIT FAIL: {r.get('why')}"
        ax.set_title(ttl, fontsize=9)
        lim = 7.5
        ctr = np.array(r["A"]) if r["ok"] else np.zeros(2)
        ax.set_xlim(ctr[0] - lim, ctr[0] + lim)
        ax.set_ylim(ctr[1] - lim, ctr[1] + lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axs.ravel()[0].legend(fontsize=7, loc="lower left")
    for ax in axs.ravel()[len(req):]:
        ax.axis("off")
    fig.suptitle(f"auditor3 sc_4f8188b4d1a8 {name}: slab ±{X.SLAB} mm perpendicular to gold; gold at origin; dotted circle = 4 mm around measured outer apex")
    fig.tight_layout()
    out = c.OUT / f"a3_xs_{name}.png"
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
