import json, sys, numpy as np
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, is_match, match_lines, Thresholds
from common import R, O, gold, tof, unit
W = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2/"
OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_cec08dfca9ab/"
apex = json.load(open(W + "apex_lines.json"))
st2 = json.load(open(W + "stations2.json"))
jl = json.load(open(W + "joggle_lines.json"))
def to_raw(Pf): return np.asarray(Pf) @ R + O
def med(bid, k):
    v = np.array([r[k] for r in st2[bid] if r.get("ok") and r.get(k) is not None], float)
    v = v[np.isfinite(v)]
    return dict(median=round(float(np.median(v)), 2), min=round(float(v.min()), 2), max=round(float(v.max()), 2)) if len(v) else None
TRUE_EXT = {"HB1": (-155.0, 151.5), "HB2": (-155.0, 151.5), "HB3": (-85.0, 114.5), "HB4": (-85.5, 114.0), "HB5": (-52.0, 81.0), "HB6": (-52.0, 81.0)}
VERDICT = {b: "REAL_ERROR" for b in TRUE_EXT}
lines = {}
for bid in TRUE_EXT:
    gs, ge = gold[bid]; A0, A1 = map(np.array, apex[bid]["apex_line_raw"])
    m = pair_metrics(np.array([gs, ge]), np.array([A0, A1]))
    si = 0 if bid in ("HB1", "HB2") else 1
    glo, ghi = sorted([tof(gs)[si], tof(ge)[si]]); lo, hi = TRUE_EXT[bid]
    lines[bid] = dict(
        verdict=VERDICT[bid],
        gold_len_mm=round(float(np.linalg.norm(ge - gs)), 1),
        gold_span_part_frame=[round(glo, 1), round(ghi, 1)],
        fold_span_part_frame=[lo, hi],
        fold_len_mm=round(hi - lo, 1),
        fold_coverage_pct=round(100 * m["prediction_span_overlap"], 1),
        gold_on_fold_pct=round(100 * m["gold_span_overlap"], 1),
        axis_angle_deg=round(m["axis_angle_deg"], 2),
        frozen_matcher_transverse_mm=round(m["transverse_offset_mm"], 2),
        frozen_matcher_match_vs_apex=bool(is_match(m, Thresholds())),
        gold_to_outer_apex_mm=med(bid, "d_apex"),
        offset_along_concave_bisector_mm=med(bid, "along"),
        offset_across_mm=med(bid, "across"),
        gold_to_outer_virtual_sharp_mm=med(bid, "d_outer_sharp"),
        gold_to_inner_virtual_sharp_mm=med(bid, "d_inner_sharp"),
        gold_to_inner_skin_apex_mm=med(bid, "d_inner_apex"),
        gold_to_arc_centre_mm=med(bid, "d_arc_centre"),
        outer_radius_mm=med(bid, "arc_R"),
        bend_angle_deg=med(bid, "angle"),
        apex_line_raw=apex[bid]["apex_line_raw"],
        apex_fit_rms_mm=round(apex[bid]["apex_fit_rms_mm"], 3),
    )
# joggle bends (missing): refit direction in part frame, extend to measured X extents
JX = {"front_joggle": (-154.0, 152.0), "back_lip_joggle": (-158.0, 154.0)}
missing = []
for name, tags in jl.items():
    for tag in ("concave_up_bend(d2>0)", "convex_up_bend(d2<0)"):
        r = jl[name][tag]
        P0 = tof(np.array(r["raw_start"])); P1 = tof(np.array(r["raw_end"]))
        d = unit(P1 - P0); lo, hi = JX[name]
        Q0 = P0 + d * (lo - P0[0]) / d[0]; Q1 = P0 + d * (hi - P0[0]) / d[0]
        missing.append(dict(id=f"{name}:{tag.split('(')[0]}", part_frame_mean=r["frame_mean"],
                            raw_start=to_raw(Q0).round(2).tolist(), raw_end=to_raw(Q1).round(2).tolist(),
                            length_mm=round(float(np.linalg.norm(Q1 - Q0)), 1),
                            max_slope_deg=round(jl[name]["max_slope_deg_median"], 1),
                            locus_note="from upper-skin curvature extremum; within ~1.2 mm (sheet thickness) of the outer-skin apex"))
# perfect apex labelling (6 folds + 4 joggle bends) vs this gold, frozen matcher
gold_recs = [dict(bend_id=b, line_start=gold[b][0].tolist(), line_end=gold[b][1].tolist()) for b in TRUE_EXT]
pred = [dict(bend_id="A_" + b, line_start=lines[b]["apex_line_raw"][0], line_end=lines[b]["apex_line_raw"][1]) for b in TRUE_EXT]
pred += [dict(bend_id="J_" + x["id"], line_start=x["raw_start"], line_end=x["raw_end"]) for x in missing]
res = match_lines(gold_recs, pred, Thresholds())
# sensitivity: gold moved onto the apex locus but with the human extents kept
shifted = []
for b in TRUE_EXT:
    gs, ge = gold[b]; A0, A1 = map(np.array, lines[b]["apex_line_raw"]); d = unit(A1 - A0)
    proj = lambda p: A0 + d * ((p - A0) @ d)
    shifted.append(dict(bend_id="S_" + b, line_start=proj(gs).tolist(), line_end=proj(ge).tolist()))
res_shift = match_lines(shifted, pred, Thresholds())
def dist(a, b):
    a0, a1 = map(np.array, a); b0, b1 = map(np.array, b); d = unit(a1 - a0); v = (b0 + b1) / 2 - a0
    return float(np.linalg.norm(v - (v @ d) * d))
pairs = [("HB1", "HB2"), ("HB3", "HB6"), ("HB4", "HB5"), ("HB3", "HB4"), ("HB5", "HB6")]
frame_free = {f"{a}-{b}": dict(gold_mm=round(dist(gold[a], gold[b]), 2), apex_mm=round(dist(lines[a]["apex_line_raw"], lines[b]["apex_line_raw"]), 2)) for a, b in pairs}
for v in frame_free.values(): v["diff_mm"] = round(v["gold_mm"] - v["apex_mm"], 2)
out = dict(
    scan_id="sc_cec08dfca9ab", gold11_id="CG08506AE97D", catalog_case_id="corpus/03_10839000", auditor="auditor2",
    frame=dict(label_frame="reference_mesh_raw", display_cloud_is_exact_raw_vertex_subset=True, scan_sha256_prefix="cec08dfca9ab",
               note="200000 display points all coincide with scan_2 raw vertices (max nn dist 0); gold used as-is in raw frame"),
    part=dict(sheet_thickness_mm=1.25, outer_radius_mm="2.7-2.9", main_folds=6, joggles=2, joggle_bends=4,
              non_bend_features=["base window", "side-wall windows and 4 holes per wall", "front-strip holes", "corner reliefs (free edges) at back corners",
                                 "side-wall tops are free edges for Y<-52 and Y>81 (back-lip corner region)", "base side edges are free edges for Y<-85 (no side walls)"]),
    lines=lines,
    frame_free_interline_distance_test=dict(pairs=frame_free,
        reading="gold box is 7-9 mm smaller than the folded part in every direction; a rigid frame error preserves distances, a similarity would need one scale (ratios 0.81 vs 0.97), so no frame transform can put these lines on the bends"),
    perfect_apex_labeling_vs_this_gold_frozen_matcher=dict(tp=res["true_positive"], fp=res["false_positive"], fn=res["false_negative"]),
    gold_shifted_to_apex_keep_human_extents_vs_perfect=dict(tp=res_shift["true_positive"], fp=res_shift["false_positive"], fn=res_shift["false_negative"],
        reading="locus offset alone drives the miss; the over-long HB3-HB6 extents still pass the 60% span gates"),
    missing_bends=missing,
    renders=[p for p in ["a2_overview_frame.png", "a2_xsec_HB1.png", "a2_xsec_HB2.png", "a2_xsec_HB3.png", "a2_xsec_HB4.png", "a2_xsec_HB5.png", "a2_xsec_HB6.png",
                         "a2_corners_and_extents.png", "a2_left_walltop_zoom.png", "a2_right_walltop_zoom.png", "a2_left_corner_xs.png", "a2_right_corner_xs.png",
                         "a2_front_joggle_xs.png", "a2_backlip_joggle_xs.png", "a2_vertical_corners_xy.png", "a2_sidewall_inner_check.png"]],
)
json.dump(out, open(OUT + "a2_verdicts.json", "w"), indent=1)
print(json.dumps(dict(perfect=out["perfect_apex_labeling_vs_this_gold_frozen_matcher"], shifted=out["gold_shifted_to_apex_keep_human_extents_vs_perfect"]["tp"], frame_free=frame_free), indent=1))
for b, L in lines.items():
    print(b, L["verdict"], "apex %.2f" % L["gold_to_outer_apex_mm"]["median"], "cov %.1f" % L["fold_coverage_pct"], "on-fold %.1f" % L["gold_on_fold_pct"],
          "ang %.2f" % L["axis_angle_deg"], "tr %.2f" % L["frozen_matcher_transverse_mm"], "oSharp %.2f" % L["gold_to_outer_virtual_sharp_mm"]["median"],
          "iSharp %.2f" % L["gold_to_inner_virtual_sharp_mm"]["median"], "iApex", (L["gold_to_inner_skin_apex_mm"] or {}).get("median"), "centre %.2f" % L["gold_to_arc_centre_mm"]["median"])
for x in missing: print(x["id"], x["length_mm"], x["max_slope_deg"], x["raw_start"], x["raw_end"])
