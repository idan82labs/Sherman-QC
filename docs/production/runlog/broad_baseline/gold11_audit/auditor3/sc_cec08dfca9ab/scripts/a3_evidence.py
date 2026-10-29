"""Auditor-3 evidence renders for sc_cec08dfca9ab (read-only on inputs).

a3_E1_gold_vs_apex_sections.png : HB1..HB6 perpendicular sections at 25/50/75 % of the measured
    fold, +-9 mm window, 1 mm slab: scan points, gold (star, 4 mm matcher circle), bench_labeling
    outer apex (cyan dot) and outer virtual sharp (blue x), with signed offsets.
a3_E2_extent_overshoot.png      : sections inside the gold span but outside the fold (HB3/HB4 at
    Y=-120/-90/-65, HB5/HB6 at Y=-50/-35/-25): no fold there.
a3_E3_missing_joggles.png       : top view + sections of the two joggles (2 bends each) with the
    bench_labeling lines JT1/JT2/JB1/JB2 (B7..B10).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math

import numpy as np

from a3_common import OUT, load_gold, load_points, unit

MAIN = {"HB1": "B1", "HB2": "B2", "HB3": "B3", "HB4": "B4", "HB5": "B5", "HB6": "B6"}
JOG = {"B7": "JT1", "B8": "JT2 (inner-skin pair)", "B9": "JB1", "B10": "JB2"}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = json.loads((OUT / "part_frame.json").read_text())
    c0 = np.array(fr["origin"])
    M = np.array([fr["X"], fr["Y"], fr["Z"]])
    to_part = lambda v: (np.asarray(v, float) - c0) @ M.T  # noqa: E731
    P = to_part(load_points())
    G = {b: (to_part(s), to_part(e)) for b, s, e in load_gold()}
    bench = {b["bend_id"]: b for b in json.loads((OUT / "bench_lines/lines.json").read_text())["bends"]}
    st = json.loads((OUT / "stations_summary.json").read_text())["lines"]

    def section_basis(bid, d):
        horiz = np.array([0, 1.0, 0]) if bid in ("HB1", "HB2") else np.array([1.0, 0, 0])
        eu = unit(horiz - (horiz @ d) * d)
        ew = unit(np.cross(d, eu))
        if ew[2] < 0:
            ew = -ew
        return eu, ew

    def pierce(line, c, d):
        """Point where 3D line (part frame) crosses the plane through c with normal d."""
        a, b = line
        den = (b - a) @ d
        if abs(den) < 1e-9:
            return None
        t = ((c - a) @ d) / den
        return a + t * (b - a)

    # ---------------- E1
    fig, axes = plt.subplots(6, 3, figsize=(15, 29), dpi=80)
    for r, gid in enumerate(MAIN):
        gs, ge = G[gid]
        d = unit(ge - gs)
        L = np.linalg.norm(ge - gs)
        eu, ew = section_basis(gid, d)
        f0, f1 = st[gid]["fold_main_extent_t"]
        f0, f1 = max(f0, 0.0), min(f1, L)
        ap = bench[MAIN[gid]]["apex"]
        sh = bench[MAIN[gid]]["sharp"]
        apl = (to_part(ap["start"]), to_part(ap["end"]))
        shl = (to_part(sh["start"]), to_part(sh["end"]))
        for k, frac in enumerate((0.25, 0.5, 0.75)):
            t = f0 + frac * (f1 - f0)
            c = gs + t * d
            m = (np.abs((P - c) @ d) < 0.5) & (np.linalg.norm(P - c, axis=1) < 20)
            q = np.c_[(P[m] - c) @ eu, (P[m] - c) @ ew]
            ax = axes[r, k]
            ax.scatter(q[:, 0], q[:, 1], s=3, c="#333", linewidths=0)
            ax.plot(0, 0, "*", color="#1a9c3a", ms=18, mec="k", zorder=5, label="gold " + gid)
            ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color="#1a9c3a", lw=1.5))
            pa = pierce(apl, c, d)
            ps = pierce(shl, c, d)
            txt = ""
            if pa is not None:
                va = np.array([(pa - c) @ eu, (pa - c) @ ew])
                ax.plot(*va, "o", color="#17becf", mec="k", ms=9, zorder=6, label="bench apex (outer skin)")
                txt += f"gold-apex {np.linalg.norm(va):.2f} mm"
            if ps is not None:
                vs = np.array([(ps - c) @ eu, (ps - c) @ ew])
                ax.plot(*vs, "x", color="#1f77b4", ms=10, mew=2.5, zorder=6, label="outer virtual sharp")
                txt += f"   gold-sharp {np.linalg.norm(vs):.2f} mm"
            ax.set_xlim(-9, 9)
            ax.set_ylim(-9, 9)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(f"{gid} at {int(frac * 100)}% of fold (gold t={t:.0f} mm)\n{txt}", fontsize=9)
            if k == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("sc_cec08dfca9ab auditor 3: gold vs outer apex, perpendicular sections (1 mm slab, mm grid; section axes = part frame)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "a3_E1_gold_vs_apex_sections.png")
    plt.close(fig)

    # ---------------- E2 extent overshoot
    cases = [("HB3", y) for y in (-120, -90, -65)] + [("HB4", y) for y in (-120, -90, -65)] + \
            [("HB5", y) for y in (-50, -35, -25)] + [("HB6", y) for y in (-50, -35, -25)]
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), dpi=80)
    for ax, (gid, yv) in zip(axes.ravel(), cases):
        gs, ge = G[gid]
        q = gs + (yv - gs[1]) / (ge[1] - gs[1]) * (ge - gs)
        m = (np.abs(P[:, 1] - yv) < 0.5) & (np.abs(P[:, 0] - q[0]) < 25) & (np.abs(P[:, 2] - q[2]) < 25)
        ax.scatter(P[m, 0], P[m, 2], s=3, c="#333", linewidths=0)
        ax.plot(q[0], q[2], "*", color="#1a9c3a", ms=18, mec="k")
        ax.add_patch(plt.Circle((q[0], q[2]), 4.0, fill=False, ls=":", color="#1a9c3a", lw=1.5))
        ax.set_xlim(q[0] - 20, q[0] + 20)
        ax.set_ylim(q[2] - 20, q[2] + 20)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        f0, f1 = st[gid]["fold_main_extent_t"]
        ax.set_title(f"{gid} at Y={yv} (inside gold span, outside measured fold)\nX vs Z, 1 mm slab", fontsize=9)
    fig.suptitle("Gold overshoot: HB3/HB4 run 78 mm past the side-wall end (Y<-55) along a flat free floor edge;\n"
                 "HB5/HB6 run 34 mm past the side-flange start (Y<-21) along the free top edge of the wall", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "a3_E2_extent_overshoot.png")
    plt.close(fig)

    # ---------------- E3 joggles
    fig = plt.figure(figsize=(16, 14), dpi=80)
    gsx = fig.add_gridspec(3, 4, height_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(gsx[0, :])
    rng = np.random.default_rng(3)
    sub = P[rng.choice(len(P), 250000, replace=False)]
    o = np.argsort(sub[:, 2])
    ax.scatter(sub[o, 0], sub[o, 1], c=sub[o, 2], s=0.25, cmap="turbo", vmin=-3, vmax=42, linewidths=0, rasterized=True)
    for gid, (s, e) in G.items():
        ax.plot([s[0], e[0]], [s[1], e[1]], color="k", lw=1.0)
        ax.text(*(0.5 * (s + e))[:2], gid, fontsize=8, color="w", bbox=dict(fc="k", ec="none", pad=1))
    for bid, nm in JOG.items():
        b = bench[bid]
        s, e = to_part(b["apex"]["start"]), to_part(b["apex"]["end"])
        ax.plot([s[0], e[0]], [s[1], e[1]], color="m", lw=1.6, ls="--")
        ax.text(e[0] + 2, e[1] + (2 if "JT" in nm else -6), nm, fontsize=8, color="m")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Top view (floor frame, colour = height). Black = gold HB1..HB6. Magenta dashed = unlabelled joggle bend lines "
                 "(bench_labeling auto extent; the joggles physically run the full ~306 mm width)", fontsize=9)
    for k, xv in enumerate((-100, 0, 60, 140)):
        for row, (ylo, yhi, zlo, zhi, bids) in enumerate(((118, 132, 35, 44, ("B7", "B8")), (-126, -112, -3, 5, ("B9", "B10")))):
            a2 = fig.add_subplot(gsx[1 + row, k])
            m = (np.abs(P[:, 0] - xv) < 0.5) & (P[:, 1] > ylo) & (P[:, 1] < yhi) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
            a2.scatter(P[m, 1], P[m, 2], s=4, c="#333", linewidths=0)
            dX = np.array([1.0, 0, 0])
            for bid in bids:
                b = bench[bid]
                ln = (to_part(b["apex"]["start"]), to_part(b["apex"]["end"]))
                pp = pierce(ln, np.array([xv, 0, 0.0]), dX)
                lo_x = min(ln[0][0], ln[1][0]) - 1
                hi_x = max(ln[0][0], ln[1][0]) + 1
                mk = "o" if lo_x <= xv <= hi_x else "s"
                if pp is not None:
                    a2.plot(pp[1], pp[2], mk, mfc="none", mec="m", ms=9, mew=2)
                    a2.annotate(JOG[bid].split()[0], (pp[1], pp[2]), xytext=(4, 6), textcoords="offset points", fontsize=7, color="m")
            a2.set_aspect("equal")
            a2.set_xlim(ylo, yhi)
            a2.set_ylim(zlo, zhi)
            a2.grid(alpha=0.3)
            a2.set_title(("top-flange joggle" if row == 0 else "bottom-strip joggle") + f", X={xv}: Y vs Z", fontsize=8)
    fig.suptitle("Unlabelled formed joggles (2 bends each, ~47-49 deg per bend): magenta circle = bench line at this X "
                 "(square = extrapolated beyond its auto extent)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "a3_E3_missing_joggles.png")
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
