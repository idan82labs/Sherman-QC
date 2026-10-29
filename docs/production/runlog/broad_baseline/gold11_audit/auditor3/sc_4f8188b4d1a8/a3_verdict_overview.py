"""Auditor-3 verdict overview for sc_4f8188b4d1a8: gold lines coloured by verdict, measured apex reference
lines (dashed) and the measured no-gold folds M1-M4 (black), in the part frame."""
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

VERDICT = {"B01": "OK", "B02": "CONVENTION_OFFSET", "B03": "OK", "B04": "CONVENTION_OFFSET", "B05": "OK",
           "B06": "CONVENTION_OFFSET", "B07": "NOT_A_BEND", "B08": "NOT_A_BEND", "B09": "REAL_ERROR",
           "B10": "REAL_ERROR"}
COL = {"OK": "tab:green", "CONVENTION_OFFSET": "tab:olive", "REAL_ERROR": "tab:orange", "NOT_A_BEND": "red"}


def main():
    p, n, _ = c.load_scan()
    R = np.load(c.CACHE / "part_R.npy")
    o = np.load(c.CACHE / "part_o.npy")
    q = (p - o) @ R.T
    gold = c.load_gold()[0]
    fin = json.loads((c.OUT / "a3_final_summary.json").read_text())
    miss = json.loads((c.OUT / "a3_missing_bends.json").read_text())
    pf = lambda x: (np.asarray(x) - o) @ R.T  # noqa: E731
    rng = np.random.default_rng(2)
    bg = rng.choice(len(q), 140000, replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax, (i, j, title) in zip(axs, [(0, 1, "top X'-Y'"), (0, 2, "front X'-Z'"), (1, 2, "side Y'-Z'")]):
        ax.scatter(q[bg, i], q[bg, j], s=0.2, c="0.8", rasterized=True)
        for g in gold:
            v = VERDICT[g["id"]]
            s, e = pf(g["s"]), pf(g["e"])
            ax.plot([s[i], e[i]], [s[j], e[j]], "-", c=COL[v], lw=2.4)
            ax.text((s[i] + e[i]) / 2, (s[j] + e[j]) / 2, f"{g['id']} {v}", color=COL[v], fontsize=9, weight="bold")
            r = fin[g["id"]]
            if "apex_ref_line_raw" in r:
                a_, b_ = (pf(x) for x in r["apex_ref_line_raw"])
                ax.plot([a_[i], b_[i]], [a_[j], b_[j]], "--", c="tab:blue", lw=1)
        for k, m in miss.items():
            a_, b_ = (pf(x) for x in m["apex_line_raw"])
            ax.plot([a_[i], b_[i]], [a_[j], b_[j]], "-", c="black", lw=2.4)
            ax.text((a_[i] + b_[i]) / 2, (a_[j] + b_[j]) / 2 - 4, f"{k.split('_')[0]} MISSING", color="black", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title)
    fig.suptitle("auditor3 sc_4f8188b4d1a8 (38172000_01_X, Gold11 CG1E0BBA2933): gold coloured by verdict "
                 "(green OK, olive CONVENTION_OFFSET inner-sharp ~1-1.4 mm, orange REAL_ERROR locus 3.4 mm, red NOT_A_BEND); "
                 "blue dashed = measured outer-apex fold lines; black = formed folds with no gold (M1-M4)")
    fig.tight_layout()
    out = c.OUT / "a3_verdict_overview.png"
    fig.savefig(out, dpi=70)
    print(out)


if __name__ == "__main__":
    main()
