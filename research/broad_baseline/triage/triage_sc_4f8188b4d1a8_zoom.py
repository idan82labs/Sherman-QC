#!/usr/bin/env python3
"""Read-only zoomed slices: is the upper-strip end piece bent from the strip (STEP C1/C9, along a at n~30)
or from the B06 flange (gold B07/B08, along n at a~-117)? A bend shows a continuous rounded L of constant
~1.5 mm thickness; a seam shows two separate sheet ends."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage_sc_4f8188b4d1a8 as T  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts, _ = T.load_ply(T.SCAN)
    gold = {b["bend_id"]: np.array([b["line_start"], b["line_end"]], float) for b in json.load(open(T.GOLD))["manual_bends"]}
    a = T.unit(gold["B02"][1] - gold["B02"][0]); b = T.unit(gold["B01"][1] - gold["B01"][0]); b = T.unit(b - (b @ a) * a); n = np.cross(a, b)
    Rm = np.stack([a, b, n]); c0 = pts.mean(0); P = (pts - c0) @ Rm.T
    raw = json.load(open(T.RUNS / "best_f1" / T.CASE / "predictions_cad_raw_ungated.json"))
    for r in raw["owned_bend_regions"]:
        print("cad_raw", r["bend_id"], np.round((np.array([r["line_start"], r["line_end"]]) - c0) @ Rm.T, 1).tolist())
    fig, axs = plt.subplots(2, 4, figsize=(22, 11), dpi=90)
    for r, bc in enumerate((-111.0, 112.0)):
        # constant-a slices (b-n plane) through the end piece: strip(n~30) -> end piece(b~bc) junction
        for c, av in enumerate((-105.0, -90.0)):
            m = (np.abs(P[:, 0] - av) < 0.4) & (np.abs(P[:, 1] - bc) < 12) & (np.abs(P[:, 2] - 36) < 12)
            ax = axs[r, c]; ax.scatter(P[m, 1], P[m, 2], s=3, c="k")
            ax.set_aspect("equal"); ax.set_title(f"a={av}±0.4: strip(n≈30) vs end piece (b≈{bc})  [STEP C1/C9 = this junction]")
            ax.set_xlabel("b"); ax.set_ylabel("n"); ax.grid(alpha=.3)
        # constant-n slices (a-b plane) at the B06-flange / end-piece corner: gold B07/B08 location
        for c, nv in enumerate((36.0, 41.0)):
            m = (np.abs(P[:, 2] - nv) < 0.4) & (np.abs(P[:, 0] + 117) < 10) & (np.abs(P[:, 1] - bc) < 10)
            ax = axs[r, 2 + c]; ax.scatter(P[m, 0], P[m, 1], s=3, c="k")
            g = "B07" if bc > 0 else "B08"
            gl = (gold[g] - c0) @ Rm.T
            ax.plot(gl[:, 0].mean(), gl[:, 1].mean(), "g+", ms=18, mew=2, label=f"gold {g}")
            ax.set_aspect("equal"); ax.set_title(f"n={nv}±0.4: B06 flange (a≈-117) vs end piece corner"); ax.legend()
            ax.set_xlabel("a"); ax.set_ylabel("b"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(T.OUT / "zoom_endpiece_bend_vs_seam.png"); plt.close(fig)


if __name__ == "__main__":
    main()
