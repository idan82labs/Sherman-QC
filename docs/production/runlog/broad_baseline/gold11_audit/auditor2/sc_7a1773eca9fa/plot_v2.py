#!/usr/bin/env python3
"""Auditor 2 - evidence cross-sections in the v2 frame (global skin planes, axis = outer-sharp line).

usage: plot_v2.py NPZ [lams]   -> a2_xsec_<id>.png for every gold line
Frame: origin = outer apex at that station, +y = bisector toward the concave side, +x = toward flange A
(flange names printed in the title: BASE = base plate, FLANGE/TAB/LIP = formed flange).
Drawn: raw slab points (+-0.5 mm, blue outer-skin normals, orange inner-skin normals, grey edges),
outer arc circle (R_o) and the concentric inner circle (R_o - t) about the fitted bend axis, inner
tangent points (pink ticks), outer/inner virtual sharp, outer apex, gold (green star, 4 mm circle),
and the ideal outer-apex line point (red ring).
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from xsec_audit import GOLD  # noqa: E402

BASE_N = np.array([-0.42916515, 0.26944502, -0.86210014])
NAMES = {"M01": "LIP", "M02": "SMALL TAB", "M03": "SMALL TAB", "M04": "TALL FLANGE", "M05": "TALL FLANGE"}


def circle_fit(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(sol[2] + c @ c))


npz = sys.argv[1]
lams = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.1, 0.5, 0.9]
d = np.load(npz)
P, N = d["P"], d["N"]
v3 = json.loads((OUT / "v3_polar_matcher.json").read_text())
for bid in GOLD:
    r = json.loads((OUT / f"v2_{bid}.json").read_text())
    Xo0, do, bis, ex, Xi0, di = (np.array(r[k]) for k in ("Xo0", "do", "bis", "ex", "Xi0", "di"))
    uA, uB = np.array(r["uA"]), np.array(r["uB"])
    nameA = "BASE" if abs(np.cross(do, uA) @ BASE_N) > 0.9 else NAMES[bid]
    nameB = "BASE" if abs(np.cross(do, uB) @ BASE_N) > 0.9 else NAMES[bid]
    beta = math.radians(r["bend_deg"])
    T = r["thickness"]
    a, b = (np.array(v) for v in GOLD[bid])
    s_a, s_b = r["s_gold"]
    st = {round(x["s"]): x for x in r["stations"]}
    sP = (P - Xo0) @ do
    fig, axs = plt.subplots(1, len(lams), figsize=(6.4 * len(lams), 6.9), squeeze=False)
    for k, lam in enumerate(lams):
        ax = axs[0][k]
        s = round(s_a + lam * (s_b - s_a))
        rec = st.get(s)
        m = np.abs(sP - s) <= 0.5
        Q = P[m] - Xo0 - s * do
        q = np.c_[Q @ ex, Q @ bis]
        nb = N[m] @ bis
        nax = np.abs(N[m] @ do)
        apex_h = rec.get("apex_h", 0.0) if rec else 0.0
        org = np.array([0.0, apex_h])
        X = q - org
        col = np.where(nax > 0.5, "0.65", np.where(nb < -0.3, "tab:blue", np.where(nb > 0.3, "tab:orange", "0.35")))
        close = np.linalg.norm(X, axis=1) < 22
        ax.scatter(X[close, 0], X[close, 1], s=3, c=col[close], linewidths=0)
        lamg = (s - s_a) / (s_b - s_a)
        G = a + lamg * (b - a) - Xo0 - s * do
        g = np.array([G @ ex, G @ bis]) - org
        # outer arc fit (outer-skin points whose normal lies inside the arc sector)
        nxv = N[m] @ ex
        ang = np.degrees(np.arctan2(nxv, -nb))
        half = math.degrees(beta) / 2
        sel = (nb < 0) & (np.abs(ang) < half - 4) & (np.linalg.norm(X, axis=1) < 12)
        txt = ""
        if sel.sum() >= 8:
            c, Ro = circle_fit(X[sel])
            Ri = Ro - T
            ax.add_patch(plt.Circle(c, Ro, fill=False, color="tab:brown", lw=0.7))
            ax.add_patch(plt.Circle(c, Ri, fill=False, color="tab:brown", lw=0.7, ls="--"))
            ax.plot(*c, "+", color="tab:brown", ms=10)
            for sgn in (+1, -1):
                tp = c + Ri * np.array([sgn * math.sin(beta / 2), -math.cos(beta / 2)])
                ax.plot(*tp, marker="|", ms=14, mew=2.5, color="#e377c2")
            v = g - c
            phi = math.degrees(math.atan2(v[0], -v[1]))
            txt = f"R_o={Ro:.2f} R_i~{Ri:.2f}  gold r={np.linalg.norm(v):.2f} phi={phi:+.0f} (tangent +-{half:.0f})"
        ax.plot(0, 0, "D", color="cyan", mec="k", ms=8, zorder=6, label="outer apex")
        ax.plot(0, -apex_h, "x", color="tab:blue", ms=11, mew=2.5, zorder=6, label="outer virtual sharp")
        Xi = Xi0 + ((Xo0 + s * do - Xi0) @ di) * di - Xo0 - s * do
        xi = np.array([Xi @ ex, Xi @ bis]) - org
        ax.plot(*xi, "x", color="purple", ms=11, mew=2.5, zorder=6, label="inner virtual sharp")
        ax.plot(*g, "*", color="#1a9c3a", mec="k", ms=18, zorder=7, label="gold")
        ax.add_patch(plt.Circle(g, 4.0, fill=False, ls=":", color="#1a9c3a", lw=1.2))
        ax.plot(0, 0, "o", mfc="none", mec="red", ms=16, mew=1.5, zorder=5, label="ideal apex line")
        ax.set_xlim(-14, 14)
        ax.set_ylim(-8, 16)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel(f"across (mm)   <- flange B = {nameB}  |  flange A = {nameA} ->")
        ax.set_ylabel("along bisector (mm), + = concave side")
        dd = np.hypot(*g)
        ax.set_title(f"{bid} @ {lam:.0%} of gold span (s={s})\n gold - outer apex: across {g[0]:+.2f}, bisector {g[1]:+.2f}, "
                     f"|d|={dd:.2f} mm\n{txt}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="lower left")
    vm = v3[bid]["matcher_gold_vs_ideal_apex"]
    fig.suptitle(f"sc_7a1773eca9fa {bid} (bend {r['bend_deg']:.1f} deg, t={T:.2f}) auditor2 - frozen matcher gold vs ideal outer-apex line: "
                 f"offset {vm['transverse_offset_mm']:.2f} mm, angle {vm['axis_angle_deg']:.2f} deg, overlap gold {vm['gold_span_overlap']:.2f}"
                 f" / pred {vm['prediction_span_overlap']:.2f} -> {'MATCH' if vm['is_match'] else 'NO MATCH'}", fontsize=10)
    fig.tight_layout()
    outp = OUT / f"a2_xsec_{bid}.png"
    fig.savefig(outp, dpi=100)
    print(outp)
