#!/usr/bin/env python3
"""Auditor 2 (pass b) - consistency of every gold line along its full length, from the saved
station JSONs only (no scan load, no geometry rerun).

Per 1 mm station inside the gold span:
  * d_gold_apex, across / bisector offset from the outer apex      (v2 stations, global skin planes)
  * past_tangent_mm: signed distance of the gold point beyond the bend-zone edge, measured along
    the flange on the gold's side: (g - C) . u_side, C = fitted arc centre (inner fit, else outer;
    concentric arcs share the tangent coordinate), u_side = unit flange direction away from the corner.
    > 0 = on the flat flange beyond the inner/outer tangent line; < 0 = inside the bend arc zone.
  * radial position r = |g - C| vs R_i (inner skin) and R_i + t (outer skin) while inside the zone.
Writes a2b_consistency.json and a2b_consistency.png next to this file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
IDS = ["M01", "M02", "M03", "M04", "M05"]
NAMES = {"M01": "base-LIP", "M02": "base-SMALL TAB", "M03": "base-SMALL TAB",
         "M04": "base-TALL FLANGE", "M05": "base-TALL FLANGE"}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


summ = json.loads((OUT / "v2_summary.json").read_text())
report = {}
fig, axs = plt.subplots(len(IDS), 2, figsize=(15, 3.1 * len(IDS)), squeeze=False)
for row, bid in enumerate(IDS):
    v1 = json.loads((OUT / f"stations_{bid}.json").read_text())
    v2 = json.loads((OUT / f"v2_{bid}.json").read_text())
    T = v2["thickness"]
    # ---- v1: tangent / radial position (2D frame of the true axis)
    s_a1, s_b1 = v1["s_gold"]
    lo1, hi1 = min(s_a1, s_b1), max(s_a1, s_b1)
    rows1 = []
    for st in v1["stations"]:
        if not st.get("ok") or not (lo1 <= st["s"] <= hi1):
            continue
        C = st.get("C_i") or st.get("C_o")
        if C is None:
            continue
        C = np.array(C)
        g = np.array(st["gold2d"])
        Xo = np.array(st["Xo"])
        ex = np.array(st["ex"])
        ua, ub = unit(st["ua"]), unit(st["ub"])
        side = "A" if (g - Xo) @ ex > 0 else "B"
        u = ua if side == "A" else ub
        past = float((g - C) @ u)
        r = float(np.linalg.norm(g - C))
        Ri = st.get("R_i") if st.get("C_i") else (st["R_o"] - T if st.get("R_o") else None)
        rows1.append({"s": st["s"], "lam": st["lam"], "side": side, "past_tangent_mm": past,
                      "r_from_centre": r, "R_i": Ri, "centre_src": "inner" if st.get("C_i") else "outer",
                      "nearest_vertex_skin": st.get("nearest_vertex_skin"),
                      "d_gold_scan3d": st.get("d_gold_scan3d")})
    # ---- v2: offsets from the outer apex, fold flags
    Rs = np.array([x["R_i"] for x in rows1 if x["R_i"]])
    Rmed = float(np.median(Rs)) if len(Rs) else None
    n_all1 = len(rows1)
    rows1 = [x for x in rows1 if x["R_i"] and abs(x["R_i"] - Rmed) <= 0.8 and abs(x["past_tangent_mm"]) <= 6.0]
    s_a, s_b = v2["s_gold"]
    lo, hi = min(s_a, s_b), max(s_a, s_b)
    rows2 = [x for x in v2["stations"] if lo <= x["s"] <= hi and x.get("fold") and "d_gold_apex" in x]
    sm = summ[bid]
    sides = [x["side"] for x in rows1]
    side_major = max(set(sides), key=sides.count) if sides else None
    inside = [x for x in rows1 if x["past_tangent_mm"] <= 0 and x["R_i"]]
    rep = {
        "pair": NAMES[bid], "bend_deg": v2["bend_deg"], "thickness": T,
        "gold_len_mm": v2["gold_len"], "angle_gold_vs_axis_deg": v2["angle_gold_axis_deg"],
        "fold_extent_mm": sm.get("fold_len"), "gold_covers_fold_frac": sm.get("gold_covers_fold_frac"),
        "gold_on_fold_frac": sm.get("gold_on_fold_frac"),
        "d_gold_outer_apex_mm": stats([x["d_gold_apex"] for x in rows2]),
        "across_mm": stats([x["gold_minus_apex_across"] for x in rows2]),
        "bisector_mm(+concave)": stats([x["gold_minus_apex_bis"] for x in rows2]),
        "d_gold_inner_apex_mm": stats([x["d_gold_iapex"] for x in rows2 if "d_gold_iapex" in x]),
        "d_gold_scan_mm": stats([x["d_gold_scan"] for x in rows2]),
        "nearest_skin_counts": {k: [x["nearest_skin"] for x in rows2].count(k)
                                for k in set(x["nearest_skin"] for x in rows2)},
        "gold_side_flange": side_major,
        "gold_side_counts": {k: sides.count(k) for k in set(sides)},
        "past_tangent_mm(+beyond onto flange)": stats([x["past_tangent_mm"] for x in rows1]),
        "r_minus_Ri_mm_when_inside_zone": stats([x["r_from_centre"] - x["R_i"] for x in inside]),
        "n_stations_v1": len(rows1), "n_stations_v1_before_fit_filter": n_all1, "R_i_median_v1": Rmed, "n_stations_v2": len(rows2),
    }
    report[bid] = rep
    # ---- plots
    ax = axs[row][0]
    L = v2["gold_len"]
    to_x = lambda sv: (sv - s_a) / (s_b - s_a) * L
    s2 = np.array([to_x(x["s"]) for x in rows2])
    ax.plot(s2, [x["d_gold_apex"] for x in rows2], "-", color="#1b7f3b", label="|gold - outer apex|")
    ax.plot(s2, [x["gold_minus_apex_across"] for x in rows2], ":", color="#555", label="across (+ toward flange A)")
    ax.plot(s2, [x["gold_minus_apex_bis"] for x in rows2], "--", color="#c05a00", label="bisector (+ concave side)")
    ax.axhline(4.0, color="red", lw=0.8, ls=":")
    ax.axhline(-4.0, color="red", lw=0.8, ls=":")
    f0, f1 = sorted((to_x(sm["fold_extent_s"][0]), to_x(sm["fold_extent_s"][1])))
    ax.axvspan(f0, f1, color="#d8ecff", alpha=0.6, lw=0, label="detected fold extent")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(L, color="k", lw=0.8)
    ax.set_title(f"{bid} {NAMES[bid]} {v2['bend_deg']:.1f} deg: offsets from outer apex along the line "
                 f"(gold ends = black, 4 mm = red)", fontsize=9)
    ax.set_xlabel("position along gold line (mm, 0 = gold start)", fontsize=8)
    ax.set_ylabel("mm", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    if row == 0:
        ax.legend(fontsize=7, loc="lower left")
    ax = axs[row][1]
    s1 = np.array([x["lam"] * L for x in rows1])
    ax.plot(s1, [x["past_tangent_mm"] for x in rows1], "-", color="#6a3d9a",
            label="gold past bend-zone edge (+ = on flat flange)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvspan(f0, f1, color="#d8ecff", alpha=0.6, lw=0)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(L, color="k", lw=0.8)
    ax.set_title(f"{bid}: signed distance past the tangent line on the {side_major}-side flange", fontsize=9)
    ax.set_xlabel("position along gold line (mm, 0 = gold start)", fontsize=8)
    ax.set_ylabel("mm", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    if row == 0:
        ax.legend(fontsize=7, loc="lower left")
fig.suptitle("sc_7a1773eca9fa 49001000 Gold11 - auditor 2 pass b: gold locus consistency along each line "
             "(raw-vertex stations, 1 mm)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / "a2b_consistency.png", dpi=110)
(OUT / "a2b_consistency.json").write_text(json.dumps(report, indent=1))
for bid, rep in report.items():
    print(bid, json.dumps({k: (round(v["median"], 2) if isinstance(v, dict) and "median" in v else v)
                           for k, v in rep.items()}, default=str))
