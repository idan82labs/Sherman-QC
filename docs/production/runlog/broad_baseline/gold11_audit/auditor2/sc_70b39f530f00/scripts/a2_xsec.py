"""Auditor-2: perpendicular cross-section close-ups per gold line (exact mesh sections, no slab).

Rows = M01, M02, M03. Columns = stations at 10 / 50 / 90 % of the measured fold extent + a strip
chart of the gold position along the whole axis. Section frame: x = across bisector, y = along the
bisector with +y toward the CONCAVE side, origin = outer apex at that station.
Markers: outer apex (blue diamond), inner apex (orange diamond), outer virtual sharp (blue x),
inner virtual sharp (purple x), gold pierce point (green star), best_f1 prediction pierce points
(red circles), 4 mm matcher circle around the outer apex (dotted)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import OUT, load_gold, load_mesh, section, unit  # noqa: E402

m = load_mesh()
gold = load_gold()
rep = json.loads((OUT / "a2_apex_report.json").read_text())
LR = json.loads((OUT / "a2_lines_report.json").read_text())["bends"]
RUNS = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_repro/runs/sc_70b39f530f00/best_f1")
pred = {r["bend_id"]: (np.array(r["line_start"]), np.array(r["line_end"]))
        for r in json.loads((RUNS / "predictions_best_f1.json").read_text())["owned_bend_regions"]}

fig = plt.figure(figsize=(26, 19))
gsp = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.5])
for row, bid in enumerate(("M01", "M02", "M03")):
    r = rep[bid]
    ax_ = np.array(r["axis"])
    sdir = np.array(r["sdir"])
    cdir = np.array(r["cdir"])
    Xo = np.array(r["Xo"])
    xi_sep = r["Xi_sep_along_mm"]
    t0, t1 = LR[bid]["fold_extent_on_axis_mm"]
    gs, ge = gold[bid]
    gd = unit(ge - gs)
    for col, frac in enumerate((0.10, 0.50, 0.90)):
        t = t0 + frac * (t1 - t0)
        p0 = Xo + t * ax_
        polys = section(m, p0, ax_)
        ax = fig.add_subplot(gsp[row, col])
        # find OA/IA at this station (same method as a2_apex)
        hits = []
        for pl in polys:
            q = np.c_[(pl - p0) @ cdir, (pl - p0) @ sdir]
            ax.plot(q[:, 0], q[:, 1], "-", color="0.2", lw=1.2)
            for k in range(len(q) - 1):
                a_, b_ = q[k], q[k + 1]
                if (a_[0] > 0) == (b_[0] > 0) or a_[0] == b_[0]:
                    continue
                lam = a_[0] / (a_[0] - b_[0])
                y = a_[1] + lam * (b_[1] - a_[1])
                if -6 < y < 12:
                    hits.append(y)
        hits = sorted(hits)
        oa, ia = hits[0], hits[1]
        # shift so the outer apex is at the origin
        def P(x, y):
            return x, y - oa
        for pl in polys:
            pass
        ax.cla()
        for pl in polys:
            q = np.c_[(pl - p0) @ cdir, (pl - p0) @ sdir - oa]
            ax.plot(q[:, 0], q[:, 1], "-", color="0.2", lw=1.2)
        ax.plot(0, 0, "D", color="#1f77b4", ms=9, label="outer apex")
        ax.plot(0, ia - oa, "D", color="#ff7f0e", ms=8, label="inner apex")
        ax.plot(0, -oa, "x", color="#1f77b4", ms=11, mew=2, label="outer virtual sharp")
        ax.plot(0, xi_sep - oa, "x", color="#9467bd", ms=11, mew=2, label="inner virtual sharp")
        lam = ((p0 - gs) @ ax_) / (gd @ ax_)
        G = gs + lam * gd
        gx, gy = (G - p0) @ cdir, (G - p0) @ sdir - oa
        ax.plot(gx, gy, "*", color="#1a9c3a", ms=18, mec="k", mew=0.6, label=f"gold {bid}")
        for pid, (ps, pe) in pred.items():
            pdv = unit(pe - ps)
            if abs(pdv @ ax_) < 0.9:
                continue
            lp = ((p0 - ps) @ ax_) / (pdv @ ax_)
            Q = ps + lp * pdv
            qx, qy = (Q - p0) @ cdir, (Q - p0) @ sdir - oa
            if abs(qx) < 9 and abs(qy) < 9:
                ax.plot(qx, qy, "o", mfc="none", mec="red", ms=9, mew=1.6)
                ax.text(qx + 0.3, qy + 0.3, pid[:14], color="red", fontsize=7)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(4 * np.cos(th), 4 * np.sin(th), ":", color="0.4", lw=1)
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-5, 8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{bid} @ {int(frac * 100)}% of fold (t={t:.1f})\n"
                     f"gold-OA {gy:+.2f} mm along bisector, {gx:+.2f} across; gold-IA {gy - (ia - oa):+.2f}", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("across bisector (mm)", fontsize=8)
        ax.set_ylabel("along bisector, + = concave side (mm)", fontsize=8)
    # strip chart
    ax = fig.add_subplot(gsp[row, 3])
    st = [s for s in r["stations"] if "oa" in s]
    T = np.array([s["t"] for s in st])
    ax.plot(T, [s["g_minus_oa"] for s in st], "-", color="#1f77b4", label="gold - outer apex (along bisector)")
    ax.plot(T, [s["g_minus_ia"] for s in st], "-", color="#ff7f0e", label="gold - inner apex")
    ax.plot(T, [s["g_across"] for s in st], "-", color="0.5", label="gold across bisector")
    ax.plot(T, [s["thk_on_bisector"] for s in st], "--", color="k", lw=0.8, label="skin-to-skin on bisector")
    ax.axhline(4, color="r", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(t0, t1, color="#c7e9c0", alpha=0.5, label="fold extent (apex line)")
    g0, g1 = LR[bid]["gold_interval_on_axis_mm"]
    ax.plot([g0, g1], [-0.8, -0.8], "-", color="#1a9c3a", lw=5, label="gold extent")
    ax.set_xlim(min(g0, t0) - 3, max(g1, t1) + 3)
    ax.set_ylim(-1.5, 4.5)
    ax.grid(alpha=0.3)
    o = LR[bid]
    ax.set_title(f"{bid}: bend {o['bend_angle_deg']:.1f} deg, R_o~{o['R_outer_est_mm']:.1f}, coverage {100 * o['coverage_of_fold']:.0f}%, "
                 f"overshoot {o['gold_overshoot_mm']['start']:.1f}/{o['gold_overshoot_mm']['end']:.1f} mm; gold vs axis {o['gold_vs_true_axis_deg']:.2f} deg", fontsize=9)
    ax.set_xlabel("position along true bend axis (mm)")
    ax.legend(fontsize=7, loc="upper right")
fig.suptitle("sc_70b39f530f00 auditor-2: exact mesh cross-sections perpendicular to each true bend axis. "
             "Every gold line lies on the INNER (concave) skin apex, ~2.5-2.6 mm from the outer apex (inside the 4 mm circle).",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "a2_xsec_closeups.png", dpi=70)
print("wrote", OUT / "a2_xsec_closeups.png")
