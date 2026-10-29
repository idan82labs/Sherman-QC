"""Light apex/extent check of the sc_f2475286312b AI relabel lines (no scan load, no heavy geometry).

1. Frozen-matcher metrics of every AI line against the fold apex lines that auditors 1 and 3
   measured independently, station by station on cross-sections (a1 folds_scan_apex_lines,
   a3 corrected_apex_lines). The auditor line is the 'gold' argument, so gold_span_overlap is the
   fraction of the auditor-measured fold covered by the AI line.
2. Outer-arc midline check on the prep's downsampled cloud: outer-skin points whose outward
   normal is within TOL deg of -bisector, per 2 mm station. Reports their across and
   along-bisector offsets from the AI line; (0, 0) means the line is on the outer-arc midline.
   This is the only check for the two tab end kinks (B9, B10), which no auditor fitted
   independently.
Usage: python apex_verification.py <lines.json> <points_ds.npz> <a1_measurements.json> <a3_corrected_apex_lines.json> <out.json>
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

lines_path, ds_path, a1_path, a3_path, out_path = sys.argv[1:6]
TOL = 6.0
EXPECTED = {  # AI bend -> (auditor-1 fold, auditor-3 fold); asserted against nearest-line search
    "B1": ("F1_upflange_v-54", "LIP_+b"), "B2": ("F2_upflange_v-278", "LIP_-b"),
    "B3": ("F4_endwall_u-224", "WALL"), "B4": ("F3_leftflange_u-477", "SHORT_LIP"),
    "B5": ("F5_slotflange_v-85", "TAB_+b"), "B6": ("F6_slotflange_v-258", "TAB_-b"),
    "B7": ("F7_ear_v-54", "EAR_+b"), "B8": ("F8_ear_v-278", "EAR_-b"),
    "B9": (None, None), "B10": (None, None),
}
doc = json.load(open(lines_path))
a1 = {k: [v["start_raw"], v["end_raw"]] for k, v in json.load(open(a1_path))["folds_scan_apex_lines"].items()}
a3 = {k: [v["line_start"], v["line_end"]] for k, v in json.load(open(a3_path))["fold_lines"].items()}
ds = np.load(ds_path)
P, N = ds["points"], ds["normals"]
th = Thresholds()
KEEP = ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap",
        "gold_length_mm", "prediction_length_mm")


def metrics(ref, pred):
    m = pair_metrics(np.asarray(ref, float), np.asarray(pred, float))
    return {**{k: round(float(m[k]), 3) for k in KEEP}, "is_match": bool(is_match(m, th))}


def nearest(refs, ai):
    best = None
    for name, seg in refs.items():
        m = pair_metrics(np.asarray(seg, float), np.asarray(ai, float))
        if m["axis_angle_deg"] <= 5.0 and m["transverse_offset_mm"] <= 4.0:
            if best is None or m["transverse_offset_mm"] < best[1]:
                best = (name, m["transverse_offset_mm"])
    return best and best[0]


out = {"lines": lines_path, "midline_normal_tol_deg": TOL,
       "note": "reference = auditor measurement (as 'gold' arg), prediction = AI line; gold_span_overlap = "
               "fraction of the auditor-measured fold extent covered by the AI line", "bends": {}}
for b in doc["bends"]:
    if b.get("status") != "ok":
        continue
    bid = b["bend_id"]
    ai = [b["label_line"]["start"], b["label_line"]["end"]]
    rec = {"patches": b["patches"], "skin": b["skin"], "label_locus": b["label_locus"],
           "bend_angle_deg": round(b["bend_angle_deg"], 2), "length_mm": round(b["length_mm"], 2),
           "extent_source": b["extent_source"], "coverage_patch_pair": round(b["coverage"], 3),
           "fold_support_frac": b["fold_support_frac"], "fold_support_gaps_t": b["fold_support_gaps_t"],
           "confidence": b["confidence"], "auto_confidence": b["auto_confidence"],
           "confidence_reasons": b["confidence_reasons"],
           "apex_offset_from_sharp_mm": round(b["apex_offset_mm"], 3),
           "expected_outer_apex_offset_mm": round(b["expected_apex_offset_mm"]["outer"], 3)}
    want1, want3 = EXPECTED[bid]
    got1, got3 = nearest(a1, ai), nearest(a3, ai)
    rec["auditor_fold_match"] = {"a1": got1, "a3": got3, "as_expected": (got1, got3) == (want1, want3)}
    if want1:
        rec["vs_auditor1_apex"] = metrics(a1[want1], ai)
    if want3:
        rec["vs_auditor3_apex"] = metrics(a3[want3], ai)
    d = np.asarray(b["direction"])
    bis = np.asarray(b["bisector"])
    e2 = np.cross(d, bis)
    e2 /= np.linalg.norm(e2)
    s0 = np.asarray(ai[0])
    L = float(np.linalg.norm(np.asarray(ai[1]) - s0))
    rel = P - s0
    s = rel @ d
    w = rel - np.outer(s, d)
    near = (s >= 1.0) & (s <= L - 1.0) & (np.linalg.norm(w, axis=1) <= 6.0)
    mid = near & ((N @ (-bis)) >= np.cos(np.radians(TOL)))
    st = np.floor(s[mid] / 2.0)
    per = np.array([(np.median((w[mid] @ e2)[st == k]), np.median((w[mid] @ bis)[st == k]))
                    for k in np.unique(st) if (st == k).sum() >= 2]).reshape(-1, 2)
    rec["outer_arc_midline"] = {
        "stations_with_midline_points": int(len(per)), "stations_total": int(np.floor((L - 1.0) / 2.0)) + 1,
        "across_offset_median_mm": round(float(np.median(per[:, 0])), 3) if len(per) else None,
        "across_offset_p10_p90_mm": [round(float(np.percentile(per[:, 0], q)), 3) for q in (10, 90)] if len(per) else None,
        "bisector_offset_median_mm": round(float(np.median(per[:, 1])), 3) if len(per) else None,
        "bisector_offset_p10_p90_mm": [round(float(np.percentile(per[:, 1], q)), 3) for q in (10, 90)] if len(per) else None,
    }
    out["bends"][bid] = rec
json.dump(out, open(out_path, "w"), indent=2)
for bid, r in out["bends"].items():
    v1, v3, m = r.get("vs_auditor1_apex"), r.get("vs_auditor3_apex"), r["outer_arc_midline"]
    print(bid, "len", r["length_mm"], r["extent_source"], "fold_support", r["fold_support_frac"],
          "| auditor folds", r["auditor_fold_match"],
          "| a1 off/cov", v1 and (v1["transverse_offset_mm"], v1["gold_span_overlap"]),
          "| a3 off/cov", v3 and (v3["transverse_offset_mm"], v3["gold_span_overlap"]),
          "| midline st", m["stations_with_midline_points"], "/", m["stations_total"],
          "across", m["across_offset_median_mm"], "bis", m["bisector_offset_median_mm"])
