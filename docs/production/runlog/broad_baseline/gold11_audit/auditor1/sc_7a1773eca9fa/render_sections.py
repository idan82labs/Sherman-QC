#!/usr/bin/env python3
"""Auditor-1 renders for sc_7a1773eca9fa: per gold line, perpendicular cross-sections at 25/50/75 %
(close-up +-11 mm, bisector up = concave side up) plus a wide view, and a longitudinal strip
(offset from the outer apex along the true axis, fold presence vs gold span). Read-only on inputs."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_sections as A  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rot2(g2):
    # rotate so that the bisector (x axis of the section frame) points up
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def section_points(V, N, tree, X0, d, g, h, half=0.6, radius=40.0):
    ii = np.array(tree.query_ball_point(X0, radius))
    Q = V[ii]
    NQ = N[ii]
    m = np.abs((Q - X0) @ d) < half
    Q, NQ = Q[m], NQ[m]
    return (Q - X0) @ h, (Q - X0) @ g, NQ  # screen x = sideways (+ toward P), screen y = bisector (+ concave)


def draw_line(ax, n_in, c, X0, g, h, color, ls="--", lw=0.8, L=60):
    # plane (n_in, c) intersected with the station plane -> 2D line in (h, g) coords
    n2 = np.array([n_in @ h, n_in @ g])
    off = float(n_in @ (c - X0))
    if np.linalg.norm(n2) < 1e-6:
        return
    n2u = n2 / np.linalg.norm(n2)
    p = n2u * off / np.linalg.norm(n2)
    t = np.array([-n2u[1], n2u[0]])
    seg = np.array([p - L * t, p + L * t])
    ax.plot(seg[:, 0], seg[:, 1], ls, color=color, lw=lw)


def main():
    V, N = A.load()
    tree = cKDTree(V)
    labels = json.loads(A.LABELS.read_text())
    rep = json.loads((HERE / "audit_sections.json").read_text())
    by = {r["bend_id"]: r for r in rep["lines"]}
    for b in labels["manual_bends"]:
        bid = b["bend_id"]
        r = by[bid]
        f = r["frame"]
        d = np.asarray(f["d"]); g = np.asarray(f["g"]); h = np.asarray(f["h"])
        st = [x for x in r["stations"] if "gold_xy" in x and x.get("present")]
        if not st:
            continue
        sm = r["summary"]
        fig = plt.figure(figsize=(18, 13), dpi=85)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.0, 0.55])
        picks = [st[int(round(q * (len(st) - 1)))] for q in (0.25, 0.5, 0.75)]
        for col, s in enumerate(picks):
            X0 = np.asarray(s["ovs_local_xyz"])
            xs, ys, NQ = section_points(V, N, tree, X0, d, g, h)
            lp = s["loc_planes"]
            for row, lim in ((0, 11.0), (1, 35.0)):
                ax = fig.add_subplot(gs[row, col])
                m = (np.abs(xs) < lim * 1.3) & (np.abs(ys - (2 if row == 0 else 10)) < lim * 1.3)
                ax.scatter(xs[m], ys[m], s=4.0 if row == 0 else 1.0, c="#222", lw=0, rasterized=True, zorder=3)
                for key, colr in (("Pout", "#1f77b4"), ("Qout", "#1f77b4"), ("Pin", "#9467bd"), ("Qin", "#9467bd")):
                    if lp.get(key):
                        draw_line(ax, np.asarray(lp[key][0]), np.asarray(lp[key][1]), X0, g, h, colr)
                # markers
                ax.plot(0, 0, "x", color="#1f77b4", ms=9, mew=2, label="outer virtual sharp (OVS)")
                if s.get("ivs_xyz") is not None:
                    iv = np.asarray(s["ivs_xyz"]) - X0
                    ax.plot(iv @ h, iv @ g, "x", color="#9467bd", ms=9, mew=2, label="inner virtual sharp")
                if s.get("apex_o_x") is not None:
                    ax.plot(0, s["apex_o_x"], "D", color="#17becf", mec="k", ms=7, label="outer apex (AI/bench locus)")
                    circ = plt.Circle((0, s["apex_o_x"]), 4.0, fill=False, ls=":", color="#d62728", lw=1.0)
                    ax.add_patch(circ)
                if s.get("apex_i_x") is not None:
                    ax.plot(0, s["apex_i_x"], "D", color="#ff7f0e", mec="k", ms=6, label="inner apex")
                cf = s.get("circle_fit") or {}
                for nm, colr in (("outer", "#2ca02c"), ("inner", "#8c564b")):
                    if nm in cf and cf[nm]["rms"] < 0.3:
                        c = cf[nm]
                        ax.add_patch(plt.Circle((c["cy"], c["cx"]), c["R"], fill=False, color=colr, lw=0.7, alpha=0.7))
                        ax.plot(c["cy"], c["cx"], "+", color=colr, ms=8)
                gx, gy = s["gold_xy"]
                ax.plot(gy, gx, "*", color="#00a000", mec="k", ms=16, zorder=6, label="gold point")
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim * 0.55 + (2 if row == 0 else 6), lim * 1.45 + (2 if row == 0 else 6))
                ax.set_aspect("equal")
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=7)
                if row == 0:
                    off = s.get("gold_minus_apex_o")
                    ax.set_title(f"{bid} u={s['u']:.0f} ({'25 50 75'.split()[col]}% of gold)\n"
                                 f"gold-outer apex {s.get('d_apex_o', float('nan')):.2f} mm "
                                 f"(bisector {off[0]:+.2f}, side {off[1]:+.2f}); gold-inner apex "
                                 f"{s.get('d_apex_i', float('nan')):.2f}\nnearest scan {s['gold_nearest_scan_mm']:.2f} mm; "
                                 f"circle fit Ro {cf.get('outer', {}).get('R', float('nan')):.2f} Ri {cf.get('inner', {}).get('R', float('nan')):.2f}",
                                 fontsize=8)
                    if col == 0:
                        ax.legend(fontsize=6, loc="upper left")
                else:
                    ax.set_title(f"wide +-{lim:.0f} mm  (x: + toward {r['plates'][0]}, y: bisector, + concave)", fontsize=8)
        # 4th column: text summary
        ax = fig.add_subplot(gs[0:2, 3])
        ax.axis("off")
        al = r.get("apex_line_check") or {}
        mo = al.get("gold_vs_outer_apex_line", {})
        lines = [
            f"{bid}: plates {r['plates'][0]} | {r['plates'][1]}",
            f"bend angle {r['bend_angle_deg']:.1f} deg, t {r['thickness_P_mm']:.2f}/{r['thickness_Q_mm']:.2f} mm",
            f"gold vs true axis {r['gold_vs_true_axis_deg']:.2f} deg",
            f"gold length {r['gold_len_mm']:.3f} mm",
            f"fold extent {sm.get('fold_len_mm', float('nan')):.1f} mm (1 mm stations)",
            f"gold covers {sm.get('coverage_pct_of_fold', float('nan')):.1f} % of fold",
            f"gold inside fold {sm.get('gold_inside_fold_pct', float('nan')):.1f} %",
            "",
            f"gold - outer apex: median {sm['d_apex_o']['median']:.2f} mm",
            f"   p10 {sm['d_apex_o']['p10']:.2f}  p90 {sm['d_apex_o']['p90']:.2f}",
            f"   along bisector (+concave) {sm['gold_minus_apex_o_along_bisector(+concave)']['median']:+.2f}",
            f"   sideways (+ toward {r['plates'][0]}) {sm['gold_minus_apex_o_sideways(+toward_P)']['median']:+.2f}",
            f"gold - inner apex: {sm['d_apex_i']['median']:.2f} mm" if sm.get("d_apex_i") else "",
            f"gold - outer virtual sharp: {sm['d_ovs']['median']:.2f} mm",
            f"gold - inner virtual sharp: {sm['d_ivs']['median']:.2f} mm" if sm.get("d_ivs") else "",
            f"gold to {r['plates'][0]} inner plane {sm.get('gold_to_P_inner_plane_mm', float('nan')):+.2f} mm",
            f"gold to {r['plates'][1]} inner plane {sm.get('gold_to_Q_inner_plane_mm', float('nan')):+.2f} mm",
            f"circle fit Ro {sm['circlefit_R_outer']['median']:.2f}  Ri {sm['circlefit_R_inner']['median']:.2f}" if sm.get("circlefit_R_outer") and sm.get("circlefit_R_inner") else "",
            "",
            "frozen pair_metrics vs reconstructed outer-apex line:",
            f"   transverse {mo.get('transverse_offset_mm', float('nan')):.2f} mm, angle {mo.get('axis_angle_deg', float('nan')):.2f} deg",
            f"   overlap gold {mo.get('gold_span_overlap', float('nan')):.2f} / pred {mo.get('prediction_span_overlap', float('nan')):.2f}",
            f"   -> {'MATCH' if mo and mo['transverse_offset_mm'] <= 4 and mo['axis_angle_deg'] <= 5 and mo['gold_span_overlap'] >= .6 and mo['prediction_span_overlap'] >= .6 else 'NO MATCH'} (4 mm / 5 deg / 60 %)",
        ]
        ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        # longitudinal strip
        ax = fig.add_subplot(gs[2, :])
        allst = r["stations"]
        us = np.array([x["u"] for x in allst])
        pres = np.array([1.0 if x.get("present") else 0.0 for x in allst])
        ax.fill_between(us, 0, pres * 0.6, step="mid", color="#bbb", alpha=0.6, label="fold present (flange + arc)")
        gu = sorted(f["gold_u"])
        ax.axvspan(gu[0], gu[1], color="#00a000", alpha=0.08, label="gold span")
        for key, colr, lab in (("d_apex_o", "#d62728", "|gold - outer apex|"), ("d_apex_i", "#ff7f0e", "|gold - inner apex|")):
            uu = [x["u"] for x in allst if x.get(key) is not None]
            vv = [x[key] for x in allst if x.get(key) is not None]
            ax.plot(uu, vv, ".", color=colr, ms=3, label=lab)
        ax.axhline(4.0, color="k", ls=":", lw=0.8)
        ax.set_ylim(0, max(8, 1.1 * (sm["d_apex_o"]["max"] if sm.get("d_apex_o") else 8)))
        ax.set_xlabel("position along true bend axis u (mm)")
        ax.set_ylabel("mm")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(alpha=0.25)
        fig.suptitle(f"auditor1  sc_7a1773eca9fa (49001000, Gold11 CG191DEC7F81)  gold {bid}: section view, bisector up (concave side up); "
                     f"red dotted circle = 4 mm around the OUTER apex", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = HERE / f"a1_xsec_{bid}.png"
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
