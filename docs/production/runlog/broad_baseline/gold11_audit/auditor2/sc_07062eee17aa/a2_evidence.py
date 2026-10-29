"""Auditor 2, sc_07062eee17aa: one evidence sheet. Per gold line: (left) close-up perpendicular section at a
representative station with gold point, fitted outer apex, outer/inner virtual sharp; (right) along-line
strip: gold->outer-apex distance where a fold exists, fold extents shaded, gold span, 4 mm matcher limit."""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_common as c  # noqa: E402
from a2_profiles import G, P  # noqa: E402

fit = json.loads((c.OUT / "a2_fit.json").read_text())
ext = json.loads((c.OUT / "a2_extents.json").read_text())
REP = {"M01": 30.0, "M02": 30.0, "M03": 30.0, "M04": 30.0, "M05": 30.0, "M06": 30.0}
FOLD_OF = {"M01": "M01", "M02": "M02", "M03": "M02", "M04": "M04", "M05": "M05", "M06": "M05"}
order = np.argsort(P[:, 2])
Ps = P[order]
s_sorted = Ps[:, 2]

fig, axs = plt.subplots(6, 2, figsize=(17, 30), gridspec_kw={"width_ratios": [1, 2.6]})
for r, k in enumerate(["M01", "M02", "M03", "M04", "M05", "M06"]):
    a, b = G[k]
    s0 = REP[k]
    lam = (s0 - a[2]) / (b[2] - a[2])
    gp = (a + lam * (b - a))[:2]
    i0, i1 = np.searchsorted(s_sorted, [s0 - 0.5, s0 + 0.5])
    q = Ps[i0:i1, :2] - gp
    W = 6.0 if k not in ("M03", "M06") else 9.0
    cy = 0.0 if k not in ("M03", "M06") else -4.0
    m = (np.abs(q[:, 0]) < W) & (np.abs(q[:, 1] - cy) < W)
    ax = axs[r, 0]
    ax.scatter(q[m, 0], q[m, 1], s=4, c="k", label="scan (slab +-0.5 mm)")
    ax.plot(0, 0, "*", ms=20, mfc="#1a9c3a", mec="k", label=f"gold {k}")
    ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls=":", color="#1a9c3a"))
    kf = FOLD_OF[k]
    st = [x for x in fit[kf] if x.get("status") == "ok"]
    if st:
        x = min(st, key=lambda z: abs(z["s"] - s0))
        af, bf = G[kf]
        lamf = (x["s"] - af[2]) / (bf[2] - af[2])
        gpf = (af + lamf * (bf - af))[:2]
        sh = gpf - gp  # fold-line gold point relative to this gold point
        ap = np.array(x["apex"]) + sh
        xo = np.array(x["Xo"]) + sh
        ax.plot(*ap, "D", ms=9, mfc="#17becf", mec="k", label="outer apex (fit)")
        ax.plot(*xo, "x", ms=11, mew=2.5, color="#1f77b4", label="outer virtual sharp")
        if x.get("Xi"):
            xi = np.array(x["Xi"]) + sh
            ax.plot(*xi, "x", ms=11, mew=2.5, color="#9467bd", label="inner virtual sharp")
        ttl = (f"{k} @ s={x['s']:.0f}: gold->outer apex {np.linalg.norm(ap):.2f} mm\n"
               f"bend {x['bend_angle_deg']:.0f} deg, gold->outer sharp {np.linalg.norm(xo):.2f} mm")
    else:
        ttl = f"{k} @ s={s0:.0f}"
    ax.set_xlim(-W, W)
    ax.set_ylim(cy - W, cy + W)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    # strip
    ax = axs[r, 1]
    g0, g1 = sorted([a[2], b[2]])
    ax.axvspan(g0, g1, color="#1a9c3a", alpha=0.08, label=f"gold span {g1 - g0:.0f} mm")
    for (lo, hi) in ext[kf]["fold_runs_s"]:
        ax.axvspan(lo, hi, color="#ff7f0e", alpha=0.25)
    ax.axvspan(0, 0, color="#ff7f0e", alpha=0.25, label=f"fold {kf} present (scan)")
    if k in ("M03", "M06"):
        ss = np.array([x["s"] for x in fit[k]])
        ds = np.array([x.get("d_nearest_scan") or np.nan for x in fit[k]])
        top = np.array([x.get("flange_top_w_rel_gold", np.nan) for x in fit[k]])
        ax.plot(ss, ds, "-", color="k", label="gold -> nearest scan point (mm)")
        ax.plot(ss, np.clip(-top, -5, 15), "--", color="#d62728", label="gold height above flange free edge (mm)")
        # distance to the nearest fold apex = to the M02/M05 apex
        stf = [x for x in fit[kf] if x.get("status") == "ok"]
        af, bf = G[kf]
        dd = []
        for x in stf:
            lamf = (x["s"] - af[2]) / (bf[2] - af[2])
            gpf = (af + lamf * (bf - af))[:2]
            lam = (x["s"] - a[2]) / (b[2] - a[2])
            gpk = (a + lam * (b - a))[:2]
            dd.append((x["s"], np.linalg.norm(gpf + np.array(x["apex"]) - gpk)))
        dd = np.array(dd)
        ax.plot(dd[:, 0], dd[:, 1], ".", color="#17becf", label=f"gold -> nearest fold apex ({kf}) (mm)")
        ax.set_ylim(-3, 16)
    else:
        st = [x for x in fit[k] if x.get("status") == "ok"]
        ss = np.array([x["s"] for x in st])
        ax.plot(ss, [x["gold_to_apex_mm"] for x in st], "o-", ms=3, color="#17becf", label="gold -> outer apex (mm)")
        ax.plot(ss, [x["gold_minus_apex_along_inward_bisector_mm"] for x in st], ".", color="#9467bd",
                label="along inward bisector (+ = into sheet)")
        ax.plot(ss, [x["gold_to_outer_sharp_mm"] for x in st], ".", color="#1f77b4", label="gold -> outer virtual sharp")
        ax.set_ylim(-3, 6)
    ax.axhline(4.0, color="r", ls=":", lw=1, label="4 mm matcher limit")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlim(-170, 165)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    e = ext[kf]
    if k in ("M03", "M06"):
        ax.set_title(f"{k}: 14.000 mm copy of {kf} along the plate normal; sits on the free (cut) edge of the {kf} flange, "
                     f"no fold there", fontsize=10)
    else:
        ax.set_title(f"{k}: fold covered {e['fold_covered_by_gold_pct']}% by gold; {e['gold_span_on_fold_pct']}% of the "
                     f"gold span lies on fold material (matcher needs >=60% of gold span)", fontsize=10)
    ax.set_xlabel("s along gold axis (mm)")
fig.suptitle("sc_07062eee17aa (38172000 scan 1) Gold11 CG65CD575B1C - auditor 2 evidence", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(c.OUT / "a2_evidence.png", dpi=70)
print("ok")
