#!/usr/bin/env python3
"""Read-only close-ups in a part-aligned frame for sc_4f8188b4d1a8 (triage of FN/FP).

Frame: a = along B02 (long edge, ~PC1), b = along B01 (short edge, ~PC2), n = a x b (plate normal).
For each region: three orthographic projections of the points inside a box, coloured by the hidden axis,
with gold (green) and best_f1 / runtime_with_placeholders predictions (red / blue) overlaid.
"""
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

    pts, _n = T.load_ply(T.SCAN)
    gold = {b["bend_id"]: np.array([b["line_start"], b["line_end"]], float) for b in json.load(open(T.GOLD))["manual_bends"]}
    preds = {}
    for lane, f, col in (("bf1", T.RUNS / "best_f1" / T.CASE / "predictions_best_f1.json", "red"),
                         ("rwp", T.RUNS / "runtime" / T.CASE / "predictions_runtime_with_placeholders.json", "blue")):
        for r in json.load(open(f))["owned_bend_regions"]:
            preds[f"{lane}:{r['bend_id']}"] = (np.array([r["line_start"], r["line_end"]], float), col)
    a = T.unit(gold["B02"][1] - gold["B02"][0])
    b = T.unit(gold["B01"][1] - gold["B01"][0])
    b = T.unit(b - (b @ a) * a)
    n = np.cross(a, b)
    Rm = np.stack([a, b, n])
    c0 = pts.mean(0)
    P = (pts - c0) @ Rm.T
    def tf(L):
        return (L - c0) @ Rm.T
    regions = {
        "corner_B03start_CAD_B9": tf(gold["B03"][1]),
        "corner_B03end_CAD_B5_CAD_B1": tf(gold["B03"][0]),
        "slot_B09": tf(gold["B09"].mean(0)),
        "slot_B10": tf(gold["B10"].mean(0)),
        "CAD_B8_placeholder": tf(preds["rwp:CAD_B8"][0].mean(0)),
        "CAD_B11_placeholder": tf(preds["rwp:CAD_B11"][0].mean(0)),
    }
    half = {"slot_B09": 60, "slot_B10": 60}
    names = ("a(long)", "b(short)", "n(normal)")
    out = T.OUT
    for reg, ctr in regions.items():
        h = half.get(reg, 55)
        m = np.all(np.abs(P - ctr) < h, axis=1)
        Q = P[m]
        if len(Q) > 80000:
            Q = Q[np.random.default_rng(0).choice(len(Q), 80000, replace=False)]
        fig, axs = plt.subplots(1, 3, figsize=(21, 7), dpi=80)
        for ax, (i, j, k) in zip(axs, ((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            ax.scatter(Q[:, i], Q[:, j], s=0.6, c=Q[:, k], cmap="viridis")
            for gid, L in gold.items():
                l2 = tf(L)
                ax.plot(l2[:, i], l2[:, j], color="lime", lw=3, alpha=0.8)
                mm = l2.mean(0)
                if np.all(np.abs(mm - ctr) < h * 1.6):
                    ax.text(mm[i], mm[j], gid, color="darkgreen", fontsize=11, weight="bold")
            for pid, (L, col) in preds.items():
                l2 = tf(L)
                ax.plot(l2[:, i], l2[:, j], color=col, lw=1.5, ls="--")
                mm = l2.mean(0)
                if np.all(np.abs(mm - ctr) < h * 1.6) and "CAD" in pid:
                    ax.text(mm[i], mm[j] + 2, pid.split(":")[0] + ":" + pid.split("_")[-1], color=col, fontsize=9)
            ax.set_xlim(ctr[i] - h, ctr[i] + h); ax.set_ylim(ctr[j] - h, ctr[j] + h)
            ax.set_aspect("equal"); ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.set_title(f"{reg}: colour={names[k]}")
        fig.tight_layout(); fig.savefig(out / f"part_frame_{reg}.png"); plt.close(fig)
    # global part-frame overview
    Q = P[np.random.default_rng(1).choice(len(P), 120000, replace=False)]
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), dpi=80)
    for ax, (i, j, k) in zip(axs, ((0, 1, 2), (0, 2, 1), (1, 2, 0))):
        ax.scatter(Q[:, i], Q[:, j], s=0.3, c=Q[:, k], cmap="viridis")
        for gid, L in gold.items():
            l2 = tf(L); ax.plot(l2[:, i], l2[:, j], color="lime", lw=3, alpha=0.8); ax.text(*l2.mean(0)[[i, j]], gid, color="darkgreen")
        for pid, (L, col) in preds.items():
            l2 = tf(L); ax.plot(l2[:, i], l2[:, j], color=col, lw=1.5, ls="--")
            if "CAD" in pid:
                ax.text(*l2.mean(0)[[i, j]], pid.split(":")[0] + ":" + pid.split("_")[-1], color=col, fontsize=8)
        ax.set_aspect("equal"); ax.set_xlabel(names[i]); ax.set_ylabel(names[j])
    fig.tight_layout(); fig.savefig(out / "part_frame_overview.png"); plt.close(fig)
    # extents per gold line in part frame (for reasoning)
    for gid, L in gold.items():
        print(gid, np.round(tf(L), 1).tolist())
    for pid, (L, _c) in preds.items():
        print(pid, np.round(tf(L), 1).tolist())
    print("bbox", np.round(P.min(0), 1).tolist(), np.round(P.max(0), 1).tolist())


if __name__ == "__main__":
    main()
