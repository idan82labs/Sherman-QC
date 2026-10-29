"""Light apex/extent check of the sc_7a1773eca9fa AI relabel lines (no scan load, no heavy geometry).

1. Frozen matcher metrics of every AI label line against the auditors' independently measured loci
   (auditor 2 per-station ideal outer apex; auditor 3 outer apex / inner apex / outer sharp).
2. Outer-arc midline check on the prep's downsampled cloud: outer-skin points whose outward normal
   is within TOL deg of -bisector, per 2 mm station; reports their across / along-bisector offset
   from the AI line (0, 0 = line on the outer-arc midline).
Usage: python apex_verification.py <lines.json> <points_ds.npz> <a2 v3_polar_matcher.json> <a3_loci.json> <out.json>
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

lines_path, ds_path, a2_path, a3_path, out_path = sys.argv[1:6]
TOL = 6.0
AI_TO_GOLD_FOLD = {"B1": "M01", "B2": "M02", "B3": "M03", "B4": "M04", "B5": "M05"}  # same physical fold
doc = json.load(open(lines_path))
a2 = json.load(open(a2_path))
a3 = json.load(open(a3_path))
ds = np.load(ds_path)
P, N = ds["points"], ds["normals"]
th = Thresholds()


def metrics(ref, pred):
    m = pair_metrics(np.asarray(ref, float), np.asarray(pred, float))
    keep = ("axis_angle_deg", "transverse_offset_mm", "gold_span_overlap", "prediction_span_overlap",
            "gold_length_mm", "prediction_length_mm")
    return {**{k: round(m[k], 3) for k in keep}, "is_match": bool(is_match(m, th))}


out = {"lines": lines_path, "midline_normal_tol_deg": TOL,
       "note": "reference = auditor measurement (as 'gold' arg), prediction = AI line; gold_span_overlap = "
               "fraction of the auditor-measured fold extent covered by the AI line", "bends": {}}
for b in doc["bends"]:
    if b.get("status") != "ok":
        continue
    bid, fold = b["bend_id"], AI_TO_GOLD_FOLD[b["bend_id"]]
    ai = [b["label_line"]["start"], b["label_line"]["end"]]
    rec = {"fold": fold, "patches": b["patches"], "skin": b["skin"], "length_mm": round(b["length_mm"], 2),
           "extent_source": b["extent_source"], "coverage_patch_pair": round(b["coverage"], 3),
           "fold_support_frac": b["fold_support_frac"], "confidence": b["confidence"],
           "auto_confidence": b["auto_confidence"], "confidence_reasons": b["confidence_reasons"]}
    ideal = a2[fold]["ideal_apex_line"]
    rec["vs_auditor2_ideal_outer_apex"] = metrics([ideal["start"], ideal["end"]], ai)
    for loc in ("outer_apex", "inner_apex", "outer_sharp"):
        seg = a3[fold]["loci_lines"].get(loc)
        if seg:
            rec[f"vs_auditor3_{loc}"] = metrics(seg["seg"], ai)
    # outer-arc midline on the downsampled prep cloud
    d = np.asarray(b["direction"])
    bis = np.asarray(b["bisector"])
    e2 = np.cross(d, bis)
    e2 /= np.linalg.norm(e2)
    s0 = np.asarray(ai[0])
    L = float(np.linalg.norm(np.asarray(ai[1]) - s0))
    rel = P - s0
    s = rel @ d
    w = rel - np.outer(s, d)
    near = (s >= 2.0) & (s <= L - 2.0) & (np.linalg.norm(w, axis=1) <= 12.0)
    mid = near & ((N @ (-bis)) >= np.cos(np.radians(TOL)))
    st = np.floor(s[mid] / 2.0)
    per = np.array([(np.median((w[mid] @ e2)[st == k]), np.median((w[mid] @ bis)[st == k]))
                    for k in np.unique(st) if (st == k).sum() >= 2]).reshape(-1, 2)
    rec["outer_arc_midline"] = {
        "stations_with_midline_points": int(len(per)), "stations_total": int(round((L - 4.0) / 2.0)),
        "across_offset_median_mm": round(float(np.median(per[:, 0])), 3) if len(per) else None,
        "across_offset_p10_p90_mm": [round(float(np.percentile(per[:, 0], q)), 3) for q in (10, 90)] if len(per) else None,
        "bisector_offset_median_mm": round(float(np.median(per[:, 1])), 3) if len(per) else None,
    }
    out["bends"][bid] = rec
json.dump(out, open(out_path, "w"), indent=2)
for bid, r in out["bends"].items():
    print(bid, r["fold"], "len", r["length_mm"], "skin", r["skin"], "| vs a2 apex off",
          r["vs_auditor2_ideal_outer_apex"]["transverse_offset_mm"], "cov", r["vs_auditor2_ideal_outer_apex"]["gold_span_overlap"],
          "| vs a3 apex off", r["vs_auditor3_outer_apex"]["transverse_offset_mm"], "cov", r["vs_auditor3_outer_apex"]["gold_span_overlap"],
          "| a3 inner apex off", r["vs_auditor3_inner_apex"]["transverse_offset_mm"],
          "| midline across/bis", r["outer_arc_midline"]["across_offset_median_mm"], r["outer_arc_midline"]["bisector_offset_median_mm"],
          "| fold_support", r["fold_support_frac"])
