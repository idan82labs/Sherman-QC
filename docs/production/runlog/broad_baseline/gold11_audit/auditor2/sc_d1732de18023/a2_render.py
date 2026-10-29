"""Auditor-2 renders: perpendicular cross-sections (frame = true bend axis from skin
planes, NOT the gold direction) at 5 stations per gold line, a wide context view, and
the gold->outer-apex offset profile over the whole fold.  Light: matplotlib Agg."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_ply, unit  # noqa: E402

pts, nrm, _ = load_ply()
G = load_gold()
X = json.loads((OUT / "a2_xsec.json").read_text())
AL = json.loads((OUT / "a2_apex_lines.json").read_text())


def frame(fr):
    d, up, f, mid = (np.array(fr[k]) for k in ("d", "e3", "f", "mid"))
    u_in = unit(np.cross(d, up))
    if u_in @ f > 0:
        u_in = -u_in
    w = unit(np.cross(u_in, d))
    if w @ up < 0:
        w = -w
    return d, u_in, w, mid


def slab(fr, s, half=0.5, box=40):
    d, u_in, w, mid = frame(fr)
    O = mid + d * s
    rel = pts - O
    m = np.abs(rel @ d) < half
    A, B = rel[m] @ u_in, rel[m] @ w
    k = (np.abs(A) < box) & (np.abs(B) < box)
    N = nrm[m][k]
    return A[k], B[k], N @ u_in, N @ w


for bid in sorted(G):
    rec = X[bid]
    fr = rec["frame"]
    st = [s for s in rec["stations"] if s.get("ok") and "apex" in s and s.get("gold_inside")]
    pick = [st[int(round(q * (len(st) - 1)))] for q in (0.05, 0.3, 0.5, 0.7, 0.95)]
    al = AL[bid]
    fig = plt.figure(figsize=(22, 10.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 0.75])
    for i, s in enumerate(pick):
        ax = fig.add_subplot(gs[0, i])
        A, B, nA, nB = slab(fr, s["s"])
        nb = np.array(s["nbis"])
        N2 = np.c_[nA, nB]
        N2 /= np.linalg.norm(N2, axis=1, keepdims=True) + 1e-9
        convex = N2 @ nb > 0.05
        ax.scatter(A[convex], B[convex], s=3, c="#1f77b4", label="convex-skin pts (normal toward outside)")
        ax.scatter(A[~convex], B[~convex], s=3, c="#ff7f0e", label="concave-skin / edge pts")
        apex = np.array(s["apex"])
        Xo, Xi = np.array(s["Xo"]), np.array(s["Xi"])
        g = np.array(s["gold2d"])
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(apex[0] + 4 * np.cos(th), apex[1] + 4 * np.sin(th), ":", c="green", lw=1,
                label="4 mm matcher radius around OUTER APEX")
        ax.plot(*apex, "D", c="cyan", mec="k", ms=8, label="outer apex (this audit)")
        if "inner_apex" in s:
            ia = np.array(s["inner_apex"])
            ax.plot(*ia, "s", c="magenta", mec="k", ms=6, label="inner apex")
        ax.plot(*Xo, "x", c="blue", ms=10, mew=2, label="outer virtual sharp")
        ax.plot(*Xi, "x", c="purple", ms=10, mew=2, label="inner virtual sharp")
        ax.plot(*g, "*", c="red", mec="k", ms=16, label="HUMAN gold " + bid)
        cx = (apex + g) / 2
        ax.set_xlim(cx[0] - 9, cx[0] + 9)
        ax.set_ylim(cx[1] - 9, cx[1] + 9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{bid} s={s['s']:.0f} mm  slab +-0.5 mm\n"
                     f"gold->outer apex {s['gold_apex_dist']:.2f} mm  (bisector-in {s['gold_along_bis_in']:.2f}, "
                     f"across {s['gold_across']:.2f})\n"
                     f"gold->inner apex {s.get('gold_to_inner_apex', float('nan')):.2f}  gold->scan {s['gold_d_scan']:.2f}"
                     f"  t={s['t_plate']:.2f}", fontsize=8)
        ax.set_xlabel("A = inward along plate (mm)", fontsize=7)
        ax.set_ylabel("B = along plate normal (+ = side long flanges rise to)", fontsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="lower left")
    # wide context at mid station
    ax = fig.add_subplot(gs[0, 5])
    s = pick[2]
    A, B, nA, nB = slab(fr, s["s"], half=0.5, box=40)
    ax.scatter(A, B, s=1, c="k")
    ax.plot(*np.array(s["gold2d"]), "*", c="red", mec="k", ms=14)
    ax.plot(*np.array(s["apex"]), "D", c="cyan", mec="k", ms=7)
    ax.set_xlim(-12, 30)
    ax.set_ylim(-21, 21)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"{bid} context s={s['s']:.0f} (+-0.5 mm slab)", fontsize=8)
    # profile along fold
    ax = fig.add_subplot(gs[1, :])
    allst = [x for x in rec["stations"] if x.get("ok") and "apex" in x]
    ss = np.array([x["s"] for x in allst])
    dd = np.array([x["gold_apex_dist"] for x in allst])
    ins = np.array([x.get("gold_inside", False) for x in allst])
    ax.plot(ss[ins], dd[ins], "o-", c="red", ms=3, label="gold -> outer apex distance (gold inside its span)")
    ax.plot(ss[~ins], dd[~ins], "o", c="pink", ms=3, label="gold line extrapolated beyond its endpoints")
    ia = np.array([x.get("gold_to_inner_apex", np.nan) for x in allst])
    ax.plot(ss[ins], ia[ins], "s-", c="magenta", ms=2, lw=0.8, label="gold -> inner apex")
    ax.axhline(4.0, ls=":", c="green", label="4 mm frozen matcher limit")
    # fold extent and gold extent are in apex-line t; convert via station s ~ same axis (offset)
    c = np.array(al["apex_line"]["c"])
    v = np.array(al["apex_line"]["v"])
    d, u_in, w, mid = frame(fr)
    off = float((c - mid) @ d)
    fl = np.array(al["fold_extent_t"]) * float(v @ d) + off
    ge = np.array(al["gold_extent_t"]) * float(v @ d) + off
    ax.plot(sorted(fl), [-0.4, -0.4], lw=6, c="gray", label=f"fold extent (outer-arc pts) {al['fold_len_mm']:.0f} mm")
    ax.plot(sorted(ge), [-0.9, -0.9], lw=6, c="red", alpha=0.6,
            label=f"gold span {al['gold_len_mm']:.0f} mm -> covers {al['coverage_of_fold'] * 100:.0f}% of fold")
    ax.set_ylim(-1.5, 7)
    ax.set_xlabel("s along true bend axis (mm)")
    ax.set_ylabel("mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    m = al["frozen_matcher_vs_ideal_line_on_fold_extent"]
    ax.set_title(f"{bid}: gold vs ideal full-fold lines under FROZEN matcher (5deg/4mm/60%): "
                 + "  ".join(f"{k}: off {m[k]['transverse_offset_mm']:.2f} {'MATCH' if m[k]['is_match'] else 'MISS'}"
                             for k in m), fontsize=9)
    fig.suptitle(f"sc_d1732de18023 (37361005_01.ply) Gold11 CG4BDD1371B6  {bid}  - auditor 2 independent cross-sections "
                 f"(axis from skin planes; angle gold vs apex line {al['angle_gold_vs_apex_deg']:.2f} deg)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"a2_xsec_{bid}.png", dpi=95)
    plt.close(fig)
    print("wrote", OUT / f"a2_xsec_{bid}.png")
