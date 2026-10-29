import sys, json, math, numpy as np, shutil
sys.dont_write_bytecode = True
from pathlib import Path
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
OUT = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_08686dc1d5d4")
st = json.load(open(W / "fold_stations.json")); am = json.load(open(W / "apex_lines_match.json")); T = json.load(open(W / "true_fold_lines.json"))
def stats(side):
    ok = [r for r in st[side] if r["ok"]]
    d = np.array([r["d_apex"] for r in ok])
    g = lambda k: [float(np.median([r[k] for r in ok])), float(min(r[k] for r in ok)), float(max(r[k] for r in ok))]
    return {"stations_ok": len(ok), "stations_total": len(st[side]), "d_outer_apex_med_min_max": [float(np.median(d)), float(d.min()), float(d.max())],
            "frac_stations_within_4mm": float((d <= 4).mean()), "along_bisector_concave_positive_med_min_max": g("gold_minus_apex_along_bisector"),
            "across_med_min_max": g("gold_minus_apex_across"), "d_outer_virtual_sharp_med_min_max": g("d_outer_sharp"),
            "d_inner_virtual_sharp_med_min_max": g("d_inner_sharp"), "d_arc_centre_med_min_max": g("d_arc_centre"),
            "gold_to_nearest_scan_med_min_max": g("gold_nearest_scan_mm"), "bend_deg_med": g("bend_deg")[0], "R_outer_med": g("R_outer")[0]}
doc = {
    "scan_id": "sc_08686dc1d5d4", "gold11_id": "CGE4B0549E41", "catalog_case_id": "corpus/02_10839000", "auditor": "auditor2",
    "frame": "raw scan == gold frame (identity, verified: display cloud is an exact raw-vertex subset)",
    "plate_frame_note": "x along walls (fold axis), y toward the M01 wall, z plate normal (walls rise +z); walls span x -100..101; plate x -177..178, y -156..150",
    "label_job_seed": "bend_inspection_meta.json part_name '10839000 scan 1 - manual bend truth (4 expected)', expected_bends 4",
    "gold_lines": {
        "M01": {"verdict": "CONVENTION_OFFSET", "fold": "plate->M01-side wall, 90.7 deg, Ro ~3.07, t ~1.5, fold x -100..101 (201 mm)",
                "locus_stats_in_fold": stats("M01"), "frozen_matcher_vs_true_apex_line": am["M01"]["metrics_vs_gold"],
                "coverage_of_fold_pct": 97.9, "overrun_beyond_fold_mm": 72.6, "overrun_pct_of_gold": 27.0,
                "endpoint_start": "inner face of wall near fold, 0.77 mm off surface into the concave corner (x=-95.8, 4 mm inside wall start)",
                "endpoint_end": "plate right corner, underside of joggled edge strip, 0.54 mm off surface (x=173.6), 5.4 mm from extended apex line, no fold there"},
        "M02": {"verdict": "CONVENTION_OFFSET", "fold": "plate->M02-side wall, 91.7 deg, Ro ~2.79, fold x -99.4..100.8 (200 mm)",
                "locus_stats_in_fold": stats("M02"), "frozen_matcher_vs_true_apex_line": am["M02"]["metrics_vs_gold"],
                "coverage_of_fold_pct": 94.4, "overrun_beyond_fold_mm": 74.4, "overrun_pct_of_gold": 28.3,
                "endpoint_start": "ON the inner bend arc (0.13 mm, normal 57 deg), x=-88.1 (11 mm inside wall start)",
                "endpoint_end": "plate right corner, underside of joggled edge strip (x=175.2), no fold there"},
        "M03": {"verdict": "NOT_A_BEND", "where": "across flat plate at x -96..-88 joining the two wall-start points, floating 0.9-1.6 mm above the top skin",
                "flatness": "dihedral across M03 0.04-0.16 deg, step <= 0.023 mm, top-skin residual within +-0.033 mm over x -150..-70",
                "nearest_fold": "left joggle, frozen-matcher transverse 70.6 mm (A) / 71.3 mm (B)", "coverage_of_any_fold_pct": 0.0},
        "M04": {"verdict": "NOT_A_BEND", "where": "on/0.2-0.4 mm under the flat raised right edge strip, 2.8-4.3 mm from the free cut edge, joining the M01/M02 overrun endpoints",
                "flatness": "strip top skin slope change across M04 0.02-0.89 deg (flat)",
                "nearest_fold": "right joggle, parallel (0.27 deg) full-length but frozen-matcher transverse 10.6 mm (bend B) / 11.7 mm (bend A) -> unmatchable; if it was meant as the joggle it is an ~11 mm REAL_ERROR",
                "coverage_of_any_fold_pct": 0.0}},
    "true_folds_world_apex_lines": {k: v["world"] for k, v in T.items() if "world" in v},
    "true_fold_meta": {k: v for k, v in T.items() if "world" not in v},
    "missing_bends": ["M01_WALL_LIP", "M02_WALL_LIP", "LEFT_JOGGLE_A_plate_to_flank", "LEFT_JOGGLE_B_flank_to_strip", "RIGHT_JOGGLE_A_plate_to_flank", "RIGHT_JOGGLE_B_flank_to_strip"],
    "perfect_apex_prediction_vs_this_gold_frozen_matcher": {"true_positive": 2, "false_positive": 6, "false_negative": 2, "note": "8 measured true fold apex lines scored as predictions against M01-M04"},
    "engine_vs_gold_frozen_matcher": {"CADF_CAD_B1_vs_M01": "MATCH transverse 3.909 mm (0.32 mm from true apex)", "F1C_90211c283156a5d8_vs_M02": "MATCH transverse 1.863 mm"},
    "other_features_checked": "no plate embosses/ribs (0 pts >2.5 mm off plate outside wall/joggle zones), no tabs beyond wall ends, window rim = 1.5 mm edge cap only; wall holes/cut-out are not bends",
    "renders": sorted(p.name for p in OUT.glob("a2_*.png")),
}
json.dump(doc, open(OUT / "auditor2_measurements.json", "w"), indent=1, default=float)
sd = OUT / "scripts"; sd.mkdir(exist_ok=True)
for f in (W / "scripts").glob("*.py"):
    shutil.copy2(f, sd / f.name)
print("written", OUT / "auditor2_measurements.json", "scripts:", sorted(p.name for p in sd.glob("*.py")))
