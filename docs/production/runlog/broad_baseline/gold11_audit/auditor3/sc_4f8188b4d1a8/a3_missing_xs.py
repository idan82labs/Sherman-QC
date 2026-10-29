"""Auditor-3: perpendicular sections through the measured no-gold folds M1-M4 (a3_missing_bends.json),
with the measured outer-apex line point (cyan diamond) and a 4 mm circle, plus any gold line within 12 mm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_common as c  # noqa: E402


def main():
    p, n, _ = c.load_scan()
    miss = json.loads((c.OUT / "a3_missing_bends.json").read_text())
    gold = c.load_gold()[0]
    req = [(k, f) for k in miss for f in (0.25, 0.5, 0.75)]
    fig, axs = plt.subplots(len(miss), 3, figsize=(16, 5.4 * len(miss)), squeeze=False)
    for ax, (k, f) in zip(axs.ravel(), req):
        a_, b_ = (np.asarray(x) for x in miss[k]["apex_line_raw"])
        L = float(np.linalg.norm(b_ - a_))
        d, e1, e2 = c.basis(b_ - a_)
        P = a_ + d * f * L
        rel = p - P
        t, u, v = rel @ d, rel @ e1, rel @ e2
        m = (np.abs(t) < 0.4) & (np.abs(u) < 12) & (np.abs(v) < 12)
        nn = n[m]
        col = np.clip(np.abs(nn @ np.stack([d, e1, e2], 1)), 0, 1)[:, [1, 2, 0]]
        ax.scatter(u[m], v[m], s=3, c=col)
        ax.plot([0], [0], "D", mfc="cyan", mec="k", ms=9, label="measured outer apex (M line)")
        ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color="green"))
        for g in gold:
            gd = g["e"] - g["s"]
            gl = np.linalg.norm(gd)
            ts = np.linspace(0, 1, 400)
            pts = g["s"][None] + ts[:, None] * gd[None]
            r2 = pts - P
            k2 = (np.abs(r2 @ d) < 0.6) & (np.abs(r2 @ e1) < 12) & (np.abs(r2 @ e2) < 12)
            if k2.any():
                ax.plot(r2[k2] @ e1, r2[k2] @ e2, "*", c="crimson", ms=10)
                ax.text(float((r2[k2] @ e1).mean()), float((r2[k2] @ e2).mean()), g["id"], color="crimson")
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{k} @ {int(f * 100)}% (fold {L:.0f} mm, bend {miss[k]['bend_deg_median']:.0f} deg)", fontsize=9)
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("auditor3 sc_4f8188b4d1a8: sections (slab +-0.4 mm) perpendicular to the measured NO-GOLD folds; "
                 "cyan = measured outer apex; crimson = any gold line in the slab; dotted = 4 mm")
    fig.tight_layout()
    out = c.OUT / "a3_missing_xs.png"
    fig.savefig(out, dpi=70)
    print(out)


if __name__ == "__main__":
    main()
