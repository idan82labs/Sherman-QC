#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4: close-up perpendicular sections for every gold line + zoomed top views
of the line ends (read-only; PNGs into this directory)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/scripts")
from a3_global import engine_lines, gold_lines, load_points  # noqa: E402
from render_gold_xsections import fit_section  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GC = "#1a9c3a"


def unit(v):
    return v / np.linalg.norm(v)


def main():
    from scipy.spatial import cKDTree

    P = load_points()
    o, e1, e2, n = np.load(HERE / "_plate_frame.npy")
    tree = cKDTree(P)
    G = {b: (s0, s1) for b, s0, s1 in gold_lines()}
    E = engine_lines()

    def section(c, d, a, w, half=45.0, slab=0.6):
        idx = np.array(tree.query_ball_point(c, half * 1.5), int)
        Q = P[idx] - c
        m = np.abs(Q @ d) < slab
        return np.c_[Q[m] @ a, Q[m] @ w]

    def frame_for(bid):
        s0, s1 = G[bid]
        d = unit(s1 - s0)
        w = unit(n - d * (n @ d))
        a = np.cross(w, d)
        return s0, s1, d, a, w

    # ---------------- M01 / M02 sections
    for bid in ("M01", "M02"):
        s0, s1, d, a, w = frame_for(bid)
        L = float(np.linalg.norm(s1 - s0))
        on = [5.0, 100.0, 185.0]
        off = [215.0, 240.0, 262.0]
        fig, axes = plt.subplots(3, 3, figsize=(18, 17), dpi=85)
        for k, s in enumerate(on):
            c = s0 + s * d
            q = section(c, d, a, w)
            fit = fit_section(q[np.linalg.norm(q, axis=1) < 22], 0.52)
            for row, half in ((0, 45.0), (1, 7.0)):
                ax = axes[row, k]
                win = (np.abs(q[:, 0]) < half + 2) & (np.abs(q[:, 1]) < half + 2)
                ax.scatter(q[win, 0], q[win, 1], s=3 if row else 1.2, c="#444", linewidths=0)
                ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color=GC, lw=1.2))
                if fit.get("ok"):
                    for key, mk, col, lab in (("apex", "D", "#17becf", "outer apex"), ("Xo", "x", "#1f77b4", "outer virtual sharp"),
                                              ("Xi", "x", "#9467bd", "inner virtual sharp"), ("C", "+", "#8c564b", "arc centre")):
                        v = fit.get(key)
                        if v is not None:
                            ax.plot(v[0], v[1], mk, color=col, ms=8, mew=2, mec="k" if mk == "D" else col, label=lab)
                    if fit.get("apex") is not None:
                        ap = fit["apex"]
                        ax.add_patch(plt.Circle(tuple(ap), 4.0, fill=False, ls="--", color="#17becf", lw=0.9))
                ax.plot(0, 0, "*", color=GC, ms=16, mec="k", label=f"gold {bid}")
                # engine lines crossing this section
                for tag, pid, p0, p1 in E:
                    pd = unit(p1 - p0)
                    if abs(pd @ d) < math.cos(math.radians(8)):
                        continue
                    t = ((c - p0) @ d) / (pd @ d)
                    if t < -2 or t > np.linalg.norm(p1 - p0) + 2:
                        continue
                    x3 = p0 + t * pd - c
                    xy = np.array([x3 @ a, x3 @ w])
                    if np.all(np.abs(xy) < half):
                        ax.plot(*xy, "o", mfc="none", mec="#d62728", ms=7, mew=1.5)
                        ax.annotate(f"{tag}:{pid[:8]}", xy, fontsize=6, color="#d62728", xytext=(4, -9), textcoords="offset points")
                ax.set_xlim(-half, half)
                ax.set_ylim(-half, half)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(f"{bid} s={s:.0f} mm of {L:.0f} (ON wall, u={(c - o) @ e1:.0f})  +-45 mm\n"
                                 f"x = in-plate perpendicular, y = plate normal (wall hangs to -y)", fontsize=8)
                else:
                    da = None if fit.get("apex") is None else np.linalg.norm(fit["apex"])
                    ax.set_title(f"close-up +-7 mm: bend {fit.get('bend_angle_deg', float('nan')):.1f} deg, gold->outer apex "
                                 f"{da if da is None else round(da, 2)} mm\n(dashed cyan circle = 4 mm around apex)", fontsize=8)
                    if k == 0:
                        ax.legend(fontsize=7, loc="lower left")
        for k, s in enumerate(off):
            c = s0 + s * d
            q = section(c, d, a, w, half=30)
            ax = axes[2, k]
            win = (np.abs(q[:, 0]) < 25) & (np.abs(q[:, 1]) < 25)
            ax.scatter(q[win, 0], q[win, 1], s=3, c="#444", linewidths=0)
            ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color=GC, lw=1.2))
            ax.plot(0, 0, "*", color=GC, ms=16, mec="k")
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(f"{bid} s={s:.0f} (BEYOND the wall end, u={(c - o) @ e1:.0f}): flat flap edge only", fontsize=8)
        fig.suptitle(f"sc_08686dc1d5d4 auditor-3 sections perpendicular to gold {bid} (gold at origin, dotted green = 4 mm)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(HERE / f"a3_xsec_{bid}.png")
        plt.close(fig)

    # ---------------- M03 / M04 sections
    for bid, ss in (("M03", (30.0, 155.0, 280.0)), ("M04", (30.0, 150.0, 270.0))):
        s0, s1, d, a, w = frame_for(bid)
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=85)
        for k, s in enumerate(ss):
            c = s0 + s * d
            q = section(c, d, a, w, half=40)
            for row, (hx, hy) in enumerate(((35.0, 8.0), (12.0, 4.0))):
                ax = axes[row, k]
                win = (np.abs(q[:, 0]) < hx) & (np.abs(q[:, 1]) < hy)
                ax.scatter(q[win, 0], q[win, 1], s=3, c="#444", linewidths=0)
                ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color=GC, lw=1.2))
                ax.plot(0, 0, "*", color=GC, ms=16, mec="k", label=f"gold {bid}")
                ax.set_xlim(-hx, hx)
                ax.set_ylim(-hy, hy)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)
                ax.set_title(f"{bid} s={s:.0f} (u={(c - o) @ e1:.1f}, v={(c - o) @ e2:.1f})  x = in-plate perp (+x = -u), y = normal",
                             fontsize=8)
        fig.suptitle(f"sc_08686dc1d5d4 auditor-3 sections perpendicular to gold {bid} (true scale; dotted green = 4 mm)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(HERE / f"a3_xsec_{bid}.png")
        plt.close(fig)

    # ---------------- wall profiles (base bend + foot bend) perpendicular to u
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=85)
    for r, (sgn, nm) in enumerate(((-1, "wall A (gold M01 side)"), (1, "wall B (gold M02 side)"))):
        for k, u0 in enumerate((-90.0, 0.0, 90.0)):
            ax = axes[r, k]
            X = P - o
            m = (np.abs(X @ e1 - u0) < 0.5) & (X @ e2 * sgn > 120)
            vv, hh = (X @ e2)[m], (X @ n)[m]
            ax.scatter(vv, hh, s=2, c="#444", linewidths=0)
            for bid in (("M01",) if sgn < 0 else ("M02",)):
                s0, s1 = G[bid]
                dd = unit(s1 - s0)
                t = (u0 - (s0 - o) @ e1) / (dd @ e1)
                gp = s0 + t * dd - o
                ax.plot(gp @ e2, gp @ n, "*", color=GC, ms=14, mec="k", label=f"gold {bid} at this u")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlabel("v (mm)")
            ax.set_ylabel("h (mm)")
            ax.set_title(f"{nm}: section u={u0:.0f} (+-0.5 mm)", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle("sc_08686dc1d5d4 formed walls: base->wall bend (gold M01/M02) and wall->flange bend at h~-39 (u in [-68,65] only)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "a3_wall_profiles.png")
    plt.close(fig)

    # ---------------- zoomed top views at the gold corners
    X = P - o
    uu, vv, hh = X @ e1, X @ e2, X @ n
    boxes = [("far-left corner, wall A end (M01 end / M04 start)", (-186, -90), (-162, -120)),
             ("near-right corner, wall A end (M01 start / M03 start)", (60, 130), (-162, -120)),
             ("far-left corner, wall B end (M02 end / M04 end)", (-186, -90), (120, 162)),
             ("near-right corner, wall B end (M02 start / M03 end)", (60, 130), (120, 162))]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=85)
    for ax, (title, (ua, ub), (va, vb)) in zip(axes.ravel(), boxes):
        m = (uu > ua) & (uu < ub) & (vv > va) & (vv < vb)
        sc = ax.scatter(uu[m], vv[m], c=np.clip(hh[m], -6, 2), cmap="turbo", s=0.6, linewidths=0, vmin=-6, vmax=2)
        for bid, (s0, s1) in G.items():
            e = np.array([s0, s1]) - o
            ax.plot(e @ e1, e @ e2, color=GC, lw=2)
            ax.plot((s0 - o) @ e1, (s0 - o) @ e2, "o", color=GC, ms=5)
        for tag, pid, p0, p1 in E:
            if tag == "zs_decomp":
                continue
            e = np.array([p0, p1]) - o
            ax.plot(e @ e1, e @ e2, color="#d62728", lw=1, ls="--")
        ax.set_xlim(ua, ub)
        ax.set_ylim(va, vb)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title + " (colour = h, clipped -6..2; walls go deeper)", fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.025)
    fig.suptitle("sc_08686dc1d5d4 gold corners in plate frame (green = gold, red dashed = engine best_f1 / CAD lines)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "a3_corner_zooms.png")
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
