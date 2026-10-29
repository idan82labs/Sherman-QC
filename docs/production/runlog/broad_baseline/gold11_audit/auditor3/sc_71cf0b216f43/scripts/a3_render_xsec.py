"""Auditor-3 perpendicular cross-sections per gold line (part frame, u = inward distance from the
wall's outer face, h = height), at 25/50/75 % of the TRUE fold extent plus one extra station.
Marks: outer apex (cyan diamond), outer virtual sharp (blue x), inner virtual sharp (purple x),
arc centre (brown +, with R_o / R_i circles), gold point (green star, dotted 4 mm circle),
engine best_f1 crossing (red circle). Points coloured by normal . bisector (blue = faces the
convex/outer side, red = faces the concave/inner side)."""
import sys
sys.dont_write_bytecode = True
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3_common import *
from a3_stations import SIDES, GOLD_SIDE, section, corner_fit, gold_point_at

C0 = np.load(SCR / "center.npy")
VN = json.loads((OUT / "a3_verdict_numbers.json").read_text())
PRED = ROOT / "docs/production/runlog/broad_baseline/gold11_repro/runs/sc_71cf0b216f43/best_f1/predictions_best_f1.json"
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import _endpoints
preds = []
for r in json.loads(PRED.read_text()).get("owned_bend_regions") or []:
    e = _endpoints(dict(r))
    preds.append((str(r.get("bend_id")), to_frame(e[0], C0), to_frame(e[1], C0)))


def pred_cross(side, a0):
    sd = SIDES[side]
    out = []
    for pid, p0, p1 in preds:
        d = p1 - p0
        if abs(d[sd["ax"]]) < 0.9 * np.linalg.norm(d):
            continue
        t = (a0 - p0[sd["ax"]]) / d[sd["ax"]]
        if t < -0.02 or t > 1.02:
            continue
        p = p0 + t * d
        out.append((pid, np.array([sd["sgn"] * (p[sd["pe"]] - sd["ref"]), p[2]])))
    return out


for bid, (side, corner) in GOLD_SIDE.items():
    v = VN[bid]
    lo, hi = v["fold_extent_axis_mm"]
    L = hi - lo
    stations = [lo + f * L for f in (0.25, 0.5, 0.75)]
    if bid in ("M01", "M02"):
        stations.append(-80.0); extra = "OVERSHOOT zone (gold present, no fold)"
    elif bid in ("M03", "M04"):
        stations.append(lo + 0.95 * L); extra = "95 % of fold"
    else:
        stations.append(lo + 0.03 * L); extra = "3 % of fold"
    flange_u = (6.0, 25.0) if corner == "top" else ((5.0, 9.0) if side.startswith("L") else (5.0, 16.0))
    fig, axes = plt.subplots(2, 4, figsize=(24, 12.5), dpi=80)
    for k, a0 in enumerate(stations):
        u, h, nu, nh = section(side, a0)
        r = corner_fit(u, h, nu, nh, corner, flange_u)
        gp = gold_point_at(bid, side, a0)
        for row, half in ((0, 22.0), (1, 7.0)):
            ax = axes[row, k]
            if r.get("ok"):
                g = r["g"]
                col = np.c_[nu, nh] @ g
                cen = r["apex"] if row == 0 or gp is None else 0.5 * (r["apex"] + gp[2]) if np.linalg.norm(gp[2] - r["apex"]) < 8 else r["apex"]
            else:
                col = nh; cen = gp[2] if gp is not None else np.array([5.0, 18.0 if corner == "top" else -18.0])
            win = (np.abs(u - cen[0]) < half) & (np.abs(h - cen[1]) < half)
            ax.scatter(u[win], h[win], s=3 if row else 1.5, c=col[win], cmap="coolwarm", vmin=-1, vmax=1, linewidths=0)
            if r.get("ok"):
                Xo = r["Xo"]; apex = r["apex"]
                ax.plot(*Xo, "x", color="#1f77b4", ms=10, mew=2, label="outer virtual sharp")
                if r.get("Xi") is not None:
                    ax.plot(*r["Xi"], "x", color="#9467bd", ms=9, mew=2, label="inner virtual sharp")
                Ro = r["R_outer"]
                if np.isfinite(Ro) and Ro < 20:
                    Cc = apex + g * Ro
                    ax.plot(*Cc, "+", color="#8c564b", ms=12, mew=2, label=f"arc centre (R_o {Ro:.2f})")
                    ax.add_patch(plt.Circle(tuple(Cc), Ro, fill=False, color="#8c564b", lw=0.7, alpha=0.7))
                    ti = 0.5 * ((r.get("t_wall") or 1.45) + (r.get("t_plate") or 1.45))
                    if Ro - ti > 0.2:
                        ax.add_patch(plt.Circle(tuple(Cc), Ro - ti, fill=False, color="#8c564b", lw=0.5, ls="--", alpha=0.7))
                ax.plot(*apex, "D", color="#17becf", mec="k", ms=8, label="OUTER APEX (AI/engine locus)")
            for pid, pq in pred_cross(side, a0):
                if np.all(np.abs(pq - cen) < half):
                    ax.plot(*pq, "o", mfc="none", mec="#d62728", ms=8, mew=1.5)
                    ax.annotate(pid[:12], pq, fontsize=6, color="#d62728", xytext=(4, 4), textcoords="offset points")
            if gp is not None:
                q = gp[2]
                ax.plot(*q, "*", color="#1a9c3a", mec="k", ms=17, label=f"gold {bid}")
                ax.add_patch(plt.Circle(tuple(q), 4.0, fill=False, ls=":", color="#1a9c3a", lw=1.3))
            ax.set_xlim(cen[0] - half, cen[0] + half); ax.set_ylim(cen[1] - half, cen[1] + half)
            ax.set_aspect("equal"); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
            ax.set_xlabel("u: inward from wall outer face (mm)", fontsize=7); ax.set_ylabel("h (mm)", fontsize=7)
            tag = f"{(a0 - lo) / L * 100:.0f}% of fold" if k < 3 else extra
            if row == 0:
                ax.set_title(f"{bid} {side} {corner}-bend, axis={a0:.1f} ({tag}) slab +-0.6", fontsize=9)
            else:
                if r.get("ok") and gp is not None:
                    q = gp[2]; vv = q - r["apex"]
                    gpp = np.array([-r["g"][1], r["g"][0]]); gpp = gpp if gpp[0] > 0 else -gpp
                    ax.set_title(f"gold->apex {np.linalg.norm(vv):.2f} mm (along concave {vv @ r['g']:+.2f}, across {vv @ gpp:+.2f})\n"
                                 f"bend {r['bend_deg']:.1f} deg  R_o {r['R_outer']:.2f}", fontsize=8.5)
                elif gp is not None:
                    ax.set_title("no corner fit at this station (no fold)", fontsize=8.5)
                if k == 0:
                    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
    m = v["frozen_matcher_vs_apex_line"]
    fig.suptitle(f"sc_71cf0b216f43 gold {bid} vs its fold {v['fold']} (fold {v['fold_length_mm']:.0f} mm) | d_apex min/med/max "
                 f"{v['d_apex_mm']['min']}/{v['d_apex_mm']['median']}/{v['d_apex_mm']['max']} mm | frozen matcher vs apex line: "
                 f"angle {m['axis_angle_deg']:.2f} deg, offset {m['transverse_offset_mm']:.2f} mm, gold-overlap {m['gold_span_overlap']:.2f} "
                 f"-> {'MATCH' if v['frozen_match'] else 'NO MATCH'}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"a3_xsec_{bid}.png"); plt.close(fig)
    print("wrote", bid)

# offset profiles
st = json.loads((OUT / "a3_stations.json").read_text())
fig, axes = plt.subplots(2, 4, figsize=(24, 9), dpi=80)
for ax, (bid, (side, corner)) in zip(axes.ravel(), sorted(GOLD_SIDE.items())):
    key = f"{side}_{corner}"; lo, hi = VN[bid]["fold_extent_axis_mm"]
    rows = [r for r in st["bends"][key] if r.get("gold")]
    a = np.array([r["a"] for r in rows]); d = np.array([r["gold"]["d_apex"] for r in rows])
    infold = (a >= lo) & (a <= hi)
    ax.plot(a[infold], d[infold], "g.-", ms=3, label="gold -> outer apex (inside fold)")
    ax.plot(a[~infold], d[~infold], ".", color="#aaa", ms=3, label="gold beyond fold end (no fold)")
    ax.axhline(4.0, color="r", ls=":", lw=1, label="4 mm matcher limit")
    ax.axvspan(lo, hi, color="#e8f4ea", zorder=0)
    ax.set_ylim(0, max(6, d.max() * 1.1)); ax.grid(alpha=.3)
    ax.set_title(f"{bid} ({key})  offset along the fold", fontsize=9)
    ax.set_xlabel("position along fold axis (mm, part frame)", fontsize=8)
    if bid == "M01":
        ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "a3_offset_profiles.png"); plt.close(fig)
print("ok")
