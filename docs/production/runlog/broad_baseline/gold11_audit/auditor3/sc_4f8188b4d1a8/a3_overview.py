"""Auditor-3: part-frame overview renders (normal-coloured) with gold lines labelled."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402


def part_frame(gold):
    g = {x["id"]: x for x in gold}
    def dirn(i):
        d = g[i]["e"] - g[i]["s"]
        return d / np.linalg.norm(d)
    a = dirn("B02") + dirn("B04")
    a /= np.linalg.norm(a)
    b = dirn("B01") + dirn("B03")
    b -= a * (a @ b)
    b /= np.linalg.norm(b)
    cc = np.cross(a, b)
    return np.vstack([a, b, cc])


def main():
    p, n, _ = c.load_scan()
    gold, _ = c.load_gold()
    R = part_frame(gold)
    o = p.mean(0)
    q = (p - o) @ R.T
    nq = n @ R.T
    np.save(c.CACHE / "part_R.npy", R)
    np.save(c.CACHE / "part_o.npy", o)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(q), 160000, replace=False)
    col = np.clip(np.abs(nq[sel]), 0, 1)
    views = [("top X'-Y' (main-plate normal = Z')", 0, 1), ("front X'-Z'", 0, 2), ("side Y'-Z'", 1, 2)]
    fig, axs = plt.subplots(1, 3, figsize=(30, 11))
    for ax, (title, i, j) in zip(axs, views):
        k = 3 - i - j
        order = np.argsort(q[sel, k]) if k != 2 else np.argsort(q[sel, k])
        ax.scatter(q[sel][order, i], q[sel][order, j], s=0.3, c=col[order], rasterized=True)
        for gl in gold:
            s = (gl["s"] - o) @ R.T
            e = (gl["e"] - o) @ R.T
            ax.plot([s[i], e[i]], [s[j], e[j]], "-", color="crimson", lw=1.6)
            m = (s + e) / 2
            ax.text(m[i], m[j], gl["id"], color="crimson", fontsize=11, weight="bold")
            ax.plot([s[i]], [s[j]], "o", color="crimson", ms=3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    fig.suptitle("auditor3 sc_4f8188b4d1a8 38172000_01_X  part frame: X'=B02/B04 dir, Y'=B01/B03 dir; colour=|normal| (R=X',G=Y',B=Z'); gold = crimson")
    fig.tight_layout()
    fig.savefig(c.OUT / "a3_overview_partframe.png", dpi=80)
    print("ok", R)
    for gl in gold:
        s = (gl["s"] - o) @ R.T
        e = (gl["e"] - o) @ R.T
        print(gl["id"], np.round(s, 1), np.round(e, 1))
    print("bbox part", q.min(0), q.max(0))


if __name__ == "__main__":
    main()
