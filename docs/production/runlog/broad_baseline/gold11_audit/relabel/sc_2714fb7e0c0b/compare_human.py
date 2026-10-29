"""Read-only comparison: human Gold11 lines vs the AI relabel lines (frozen matcher + geometry).

Usage: python compare_human.py <lines_dir> <out_json>
Reads the human gold (never writes it), the relabel lines.json and the prep's
downsampled cloud. Light (no full-scan load).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics  # noqa: E402

HUMAN = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json"


def samples(a, b, step=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def main(lines_dir, out_path):
    lines_dir = Path(lines_dir)
    doc = json.loads((lines_dir / "lines.json").read_text())
    human = json.loads(HUMAN.read_text())
    prep = Path(doc["prep_dir"])
    ds = np.load(prep / "points_ds.npz")
    tree = cKDTree(ds["points"])
    ai = []
    per_ai = {}
    for b in doc["bends"]:
        if b.get("status") != "ok":
            continue
        line = b["label_line"]
        ai.append({"bend_id": b["bend_id"], "line_start": line["start"], "line_end": line["end"]})
        pts = samples(line["start"], line["end"])
        d, _ = tree.query(pts)
        bis = np.asarray(b["bisector"])
        per_ai[b["bend_id"]] = {
            "patches": b["patches"], "note": b["note"], "locus": b["label_locus"],
            "skin": b["skin"], "confidence": b["confidence"],
            "included_angle_deg": round(b["included_angle_deg"], 2),
            "length_mm": round(b["length_mm"], 1), "coverage": round(b["coverage"], 3),
            "fold_support_frac": round(b["fold_support_frac"], 3),
            "fold_support_gaps_t": b["fold_support_gaps_t"],
            "extent_source": b["extent_source"], "t_range": [round(v, 2) for v in b["t_range"]],
            "apex_offset_from_sharp_mm": round(b["apex_offset_mm"], 3),
            "expected_outer_apex_offset_mm": round(b["expected_apex_offset_mm"]["outer"], 3),
            "line_to_ds_cloud_mm": {"median": round(float(np.median(d)), 3),
                                    "p95": round(float(np.percentile(d, 95)), 3),
                                    "max": round(float(d.max()), 3),
                                    "frac_within_0p6mm": round(float((d <= 0.6).mean()), 3)},
            "bisector": [round(v, 4) for v in bis.tolist()],
        }
    gold = [{"bend_id": g["bend_id"], "line_start": g["line_start"], "line_end": g["line_end"]}
            for g in human["bends"]]
    result = match_lines(gold, ai, Thresholds())
    # nearest AI line per human line (diagnostic, not the matcher): by 3D segment distance
    nearest = {}
    for g in gold:
        ge = np.asarray([g["line_start"], g["line_end"]], float)
        gs = samples(*ge, step=0.5)
        best = None
        for p in ai:
            pe = np.asarray([p["line_start"], p["line_end"]], float)
            ps = samples(*pe, step=0.5)
            seg_d = float(cKDTree(ps).query(gs)[0].min())
            m = pair_metrics(ge, pe)
            if best is None or seg_d < best[0]:
                best = (seg_d, p["bend_id"], m)
        gd, _ = tree.query(samples(g["line_start"], g["line_end"]))
        nearest[g["bend_id"]] = {
            "nearest_ai_by_segment_distance": best[1], "segment_distance_mm": round(best[0], 3),
            **{k: round(v, 3) for k, v in best[2].items()},
            "human_line_to_ds_cloud_mm": {"median": round(float(np.median(gd)), 3),
                                          "frac_within_0p6mm": round(float((gd <= 0.6).mean()), 3)},
        }
    out = {"lines_dir": str(lines_dir.resolve()), "human_label": str(HUMAN),
           "matcher": "scripts/evaluate_f1_against_manual_gold.match_lines Thresholds() "
                      "(5 deg / 4 mm / 60% both spans, Hungarian)",
           "human_count": len(gold), "ai_count": len(ai),
           "match": result, "per_ai": per_ai, "per_human_nearest": nearest}
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"tp": result["true_positive"], "fp": result["false_positive"],
                      "fn": result["false_negative"],
                      "unmatched_human": result["unmatched_gold_ids"],
                      "unmatched_ai": result["unmatched_prediction_ids"],
                      "matches": [(m["gold_id"], m["prediction_id"], round(m["axis_angle_deg"], 2),
                                   round(m["transverse_offset_mm"], 2), round(m["gold_span_overlap"], 2),
                                   round(m["prediction_span_overlap"], 2)) for m in result["matches"]]},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
