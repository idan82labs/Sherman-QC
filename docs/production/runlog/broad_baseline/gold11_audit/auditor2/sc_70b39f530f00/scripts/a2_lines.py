"""Auditor-2: reference apex lines, fold extents, gold coverage, and frozen-matcher metrics.

From a2_apex_report.json (exact mesh sections every 0.25 mm along each true bend axis):
  fold extent      = axis interval where the bisector ray crosses both skins (any 2-skin hit)
  full extent      = sub-interval where the skin-to-skin distance on the bisector is >= 2.4 mm
  outer apex line  = PCA line through the outer-apex points of the full stations, clipped to fold extent
  inner apex line  = same for the inner-apex points
Then gold coverage of the fold, overshoot, and the frozen matcher's pair_metrics/is_match
(scripts/evaluate_f1_against_manual_gold.py, 5 deg / 4 mm / 60 %) for gold vs apex lines and for
the best_f1 / zero-shot / CAD prediction lines vs gold and vs the outer-apex line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

from a2_common import OUT, load_gold  # noqa: E402

rep = json.loads((OUT / "a2_apex_report.json").read_text())
gold = load_gold()
TH = Thresholds()
RUNS = ROOT / "docs/production/runlog/broad_baseline/gold11_repro/runs/sc_70b39f530f00/best_f1"
preds = {}
for f, tag in (("predictions_best_f1.json", "best_f1"), ("predictions_f1_zero_shot_decomposition.json", "zero_shot"),
               ("predictions_cad_raw_ungated.json", "cad_raw"), ("predictions_cad_gated.json", "cad_gated")):
    d = json.loads((RUNS / f).read_text())
    for r in d.get("owned_bend_regions") or []:
        preds[f"{tag}:{r['bend_id']}"] = np.array([r["line_start"], r["line_end"]], float)


def fit_line(P):
    P = np.asarray(P)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def fmt(mt):
    return (f"ang {mt['axis_angle_deg']:5.2f}  off {mt['transverse_offset_mm']:5.2f}  "
            f"gov {mt['gold_span_overlap']:4.2f}  pov {mt['prediction_span_overlap']:4.2f}")


out = {}
apex_lines = {}
for bid in ("M01", "M02", "M03"):
    r = rep[bid]
    ax = np.array(r["axis"])
    st = r["stations"]
    any2 = [s for s in st if "oa" in s]
    full = [s for s in any2 if s["thk_on_bisector"] >= 2.4]
    t_any = (min(s["t"] for s in any2) - 0.125, max(s["t"] for s in any2) + 0.125)
    t_full = (min(s["t"] for s in full), max(s["t"] for s in full))
    gs, ge = gold[bid]
    Xo = np.array(r["Xo"])
    g0, g1 = sorted([(gs - Xo) @ ax, (ge - Xo) @ ax])
    fold_len = t_any[1] - t_any[0]
    ov = max(0.0, min(g1, t_any[1]) - max(g0, t_any[0]))
    cov = ov / fold_len
    over_lo = max(0.0, t_any[0] - g0)
    over_hi = max(0.0, g1 - t_any[1])
    under_lo = max(0.0, g0 - t_any[0])
    under_hi = max(0.0, t_any[1] - g1)
    lines = {}
    for key in ("oa_xyz", "ia_xyz"):
        c, v = fit_line([s[key] for s in full])
        if v @ ax < 0:
            v = -v
        # clip to the fold extent (stations are planes perpendicular to ax through Xo + t ax)
        tc = (c - Xo) @ ax
        a_ = c + v * ((t_any[0] - tc) / (v @ ax))
        b_ = c + v * ((t_any[1] - tc) / (v @ ax))
        lines[key] = np.array([a_, b_])
        resid = np.array([np.linalg.norm((np.array(s[key]) - c) - ((np.array(s[key]) - c) @ v) * v) for s in full])
        lines[key + "_resid_max"] = float(resid.max())
    apex_lines[bid] = lines
    G = np.array([gs, ge])
    m_oa = pair_metrics(G, lines["oa_xyz"])
    m_ia = pair_metrics(G, lines["ia_xyz"])
    g_oa = [s["g_minus_oa"] for s in full if s["gold_inside"]]
    g_ia = [s["g_minus_ia"] for s in full if s["gold_inside"]]
    g_xi = [s["g_minus_xi"] for s in full if s["gold_inside"]]
    g_ac = [s["g_across"] for s in full if s["gold_inside"]]
    thk = [s["thk_on_bisector"] for s in full]
    sd = np.array(r["gold_signed_dist_to_surface"])
    out[bid] = {
        "bend_angle_deg": r["bend_angle_deg"], "R_outer_est_mm": r["R_outer_est"], "R_inner_est_mm": r["R_inner_est"],
        "gold_vs_true_axis_deg": r["gold_vs_axis_deg"],
        "fold_extent_on_axis_mm": list(t_any), "fold_full_thickness_extent_mm": list(t_full),
        "fold_length_mm": fold_len, "gold_interval_on_axis_mm": [g0, g1], "gold_length_mm": g1 - g0,
        "coverage_of_fold": cov, "gold_overshoot_mm": {"start": over_lo, "end": over_hi},
        "fold_uncovered_mm": {"start": under_lo, "end": under_hi},
        "gold_minus_outer_apex_mm": {"min": min(g_oa), "median": float(np.median(g_oa)), "max": max(g_oa)},
        "gold_minus_inner_apex_mm": {"min": min(g_ia), "median": float(np.median(g_ia)), "max": max(g_ia)},
        "gold_minus_inner_sharp_mm": {"min": min(g_xi), "median": float(np.median(g_xi)), "max": max(g_xi)},
        "gold_across_bisector_mm": {"min": min(g_ac), "median": float(np.median(g_ac)), "max": max(g_ac)},
        "skin_to_skin_on_bisector_mm": float(np.median(thk)),
        "gold_signed_dist_to_surface_mm": {"min": float(sd.min()), "median": float(np.median(sd)), "max": float(sd.max()),
                                            "frac_within_0.3": float(np.mean(np.abs(sd) <= 0.3))},
        "outer_apex_line": lines["oa_xyz"].tolist(), "inner_apex_line": lines["ia_xyz"].tolist(),
        "apex_line_fit_resid_max_mm": {"outer": lines["oa_xyz_resid_max"], "inner": lines["ia_xyz_resid_max"]},
        "matcher_gold_vs_outer_apex": {**m_oa, "is_match": bool(is_match(m_oa, TH))},
        "matcher_gold_vs_inner_apex": {**m_ia, "is_match": bool(is_match(m_ia, TH))},
    }
    o = out[bid]
    print(f"\n== {bid}  bend {o['bend_angle_deg']:.1f} deg  R_o~{o['R_outer_est_mm']:.1f}  R_i~{o['R_inner_est_mm']:.1f}  "
          f"skin-skin on bisector {o['skin_to_skin_on_bisector_mm']:.2f}")
    print(f"   fold extent on axis [{t_any[0]:.2f}, {t_any[1]:.2f}] ({fold_len:.2f} mm); full-thickness [{t_full[0]:.2f}, {t_full[1]:.2f}]")
    print(f"   gold interval [{g0:.2f}, {g1:.2f}] ({g1 - g0:.2f} mm); coverage {100 * cov:.1f}%; overshoot start {over_lo:.2f} end {over_hi:.2f}; "
          f"uncovered start {under_lo:.2f} end {under_hi:.2f}")
    print(f"   gold - outer apex along bisector: {o['gold_minus_outer_apex_mm']}")
    print(f"   gold - inner apex along bisector: {o['gold_minus_inner_apex_mm']}")
    print(f"   gold - inner sharp: {o['gold_minus_inner_sharp_mm']}")
    print(f"   gold across bisector: {o['gold_across_bisector_mm']}")
    print(f"   gold signed dist to surface: {o['gold_signed_dist_to_surface_mm']}")
    print(f"   apex line fit resid max: {o['apex_line_fit_resid_max_mm']}")
    print(f"   matcher gold vs OUTER apex line: {fmt(m_oa)}  match={is_match(m_oa, TH)}")
    print(f"   matcher gold vs INNER apex line: {fmt(m_ia)}  match={is_match(m_ia, TH)}")

print("\n== prediction lines vs gold and vs outer-apex reference (frozen pair_metrics) ==")
rows = []
for pid, P in preds.items():
    for bid in ("M01", "M02", "M03"):
        mg = pair_metrics(np.array(gold[bid]), P)
        mo = pair_metrics(np.array(out[bid]["outer_apex_line"]), P)
        if min(mg["transverse_offset_mm"], mo["transverse_offset_mm"]) > 12 or mg["axis_angle_deg"] > 30:
            continue
        rows.append({"pred": pid, "bend": bid, "vs_gold": {**mg, "is_match": bool(is_match(mg, TH))},
                     "vs_outer_apex": {**mo, "is_match": bool(is_match(mo, TH))}})
        print(f"  {pid:32s} {bid}: vs gold [{fmt(mg)} {'MATCH' if is_match(mg, TH) else 'no'}]   "
              f"vs outer apex [{fmt(mo)} {'MATCH' if is_match(mo, TH) else 'no'}]")
json.dump({"bends": out, "prediction_comparisons": rows}, open(OUT / "a2_lines_report.json", "w"), indent=1)
print("\nwrote", OUT / "a2_lines_report.json")
