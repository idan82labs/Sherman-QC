"""Auditor-2 (sc_4f8188b4d1a8): gold lines vs the auditor's own bench_labeling apex lines.

Read-only on gold.  Uses the frozen matcher (scripts/evaluate_f1_against_manual_gold: 5 deg / 4 mm /
60 % span, Hungarian) exactly as research/broad_baseline/score.py does, plus per-line geometry:
perpendicular distance of gold start/q25/mid/q75/end to the bench apex line and to the bench
plane-sharp (outer virtual sharp) line, and the fraction of the bench fold span covered by gold.
Writes a2_match_bench.json next to this script's parent directory.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics  # noqa: E402

from a2_common import OUT, gold_lines  # noqa: E402

G = gold_lines()
lines12 = json.loads((OUT / "bench_lines12" / "lines.json").read_text())
seams = json.loads((OUT / "bench_lines_seams" / "lines.json").read_text())

bench = {}
for b in lines12["bends"]:
    if b.get("status") != "ok":
        continue
    bench["AI_" + b["bend_id"]] = b
seam = {}
for b in seams["bends"]:
    if b.get("status") == "ok":
        seam["SEAM_" + b["bend_id"]] = b


def seg(b, key="label_line"):
    s = b[key]
    return np.asarray(s["start"], float), np.asarray(s["end"], float)


gold_records = [{"bend_id": k, "line_start": v[0].tolist(), "line_end": v[1].tolist()} for k, v in sorted(G.items())]
pred_records = [{"bend_id": k, "line_start": seg(b)[0].tolist(), "line_end": seg(b)[1].tolist()} for k, b in bench.items()]
thr = Thresholds()
res = match_lines(gold_records, pred_records, thr)
out = {"thresholds": thr.__dict__, "frozen_match_gold_vs_ai12": {k: res[k] for k in res if k != "pairs"}}
print("frozen matcher gold(10) vs auditor2 AI apex lines(12):",
      {k: res[k] for k in ("true_positive", "false_positive", "false_negative", "unmatched_gold_ids",
                           "unmatched_prediction_ids")})
for m in res["matches"]:
    print("  match", m["gold_id"], "<->", m["prediction_id"], "ang %.2f off %.2f govl %.2f povl %.2f" % (
        m["axis_angle_deg"], m["transverse_offset_mm"], m["gold_span_overlap"], m["prediction_span_overlap"]))

INTENDED = {"B01": "AI_B1", "B02": "AI_B2", "B03": "AI_B3", "B04": "AI_B4", "B05": "AI_B5", "B06": "AI_B6",
            "B09": "AI_B9", "B10": "AI_B10", "B07": "SEAM_B1", "B08": "SEAM_B2"}


def perp_dist(p, a, b):
    d = (b - a) / np.linalg.norm(b - a)
    r = p - a
    r = r - d * (r @ d)
    return float(np.linalg.norm(r)), r


per = {}
for gid, (ga, gb) in sorted(G.items()):
    pid = INTENDED[gid]
    rec = bench.get(pid) or seam.get(pid)
    aa, ab = seg(rec, "label_line")
    sa, sb = seg(rec, "sharp")
    d = (ab - aa) / np.linalg.norm(ab - aa)
    bis = np.asarray(rec["bisector"], float)
    bis = bis - d * (bis @ d)
    bis /= np.linalg.norm(bis)
    lat = np.cross(d, bis)
    st = {}
    for tag, f in (("start", 0.0), ("q25", 0.25), ("mid", 0.5), ("q75", 0.75), ("end", 1.0)):
        p = ga + f * (gb - ga)
        da, ra = perp_dist(p, aa, ab)
        ds, _ = perp_dist(p, sa, sb)
        # sign convention: bench bisector points from the plane sharp into the concave side? report both comps
        st[tag] = {"to_apex": round(da, 3), "to_outer_sharp": round(ds, 3),
                   "apex_bis": round(float(ra @ bis), 3), "apex_lat": round(float(ra @ lat), 3)}
    # coverage of bench fold span by gold projection
    t0, t1 = 0.0, float(np.linalg.norm(ab - aa))
    g0, g1 = sorted([float((ga - aa) @ d), float((gb - aa) @ d)])
    ov = max(0.0, min(t1, g1) - max(t0, g0))
    pm_apex = pair_metrics(np.array([ga, gb]), np.array([aa, ab]))
    pm_sharp = pair_metrics(np.array([ga, gb]), np.array([sa, sb]))
    ok_apex = (pm_apex["axis_angle_deg"] <= thr.angle_deg and pm_apex["transverse_offset_mm"] <= thr.transverse_offset_mm
               and pm_apex["gold_span_overlap"] >= thr.gold_span_overlap
               and pm_apex["prediction_span_overlap"] >= thr.prediction_span_overlap)
    ok_sharp = (pm_sharp["axis_angle_deg"] <= thr.angle_deg and pm_sharp["transverse_offset_mm"] <= thr.transverse_offset_mm
                and pm_sharp["gold_span_overlap"] >= thr.gold_span_overlap
                and pm_sharp["prediction_span_overlap"] >= thr.prediction_span_overlap)
    per[gid] = {"counterpart": pid, "counterpart_len": t1, "counterpart_bend_angle": rec.get("bend_angle_deg"),
                "counterpart_apex_offset_mm": rec.get("apex_offset_mm"),
                "gold_len": float(np.linalg.norm(gb - ga)),
                "coverage_of_counterpart": ov / t1, "gold_frac_inside": ov / (g1 - g0),
                "gold_overhang_mm": [round(max(0.0, t0 - g0), 2), round(max(0.0, g1 - t1), 2)],
                "stations": st, "matcher_vs_apex": pm_apex, "matcher_ok_vs_apex": ok_apex,
                "matcher_vs_sharp": pm_sharp, "matcher_ok_vs_sharp": ok_sharp}
    s = st
    print(f"{gid} vs {pid}: len g {per[gid]['gold_len']:.1f} / ai {t1:.1f}; cov {100 * ov / t1:.1f}% "
          f"overhang {per[gid]['gold_overhang_mm']}; to_apex s/m/e {s['start']['to_apex']:.2f}/{s['mid']['to_apex']:.2f}/"
          f"{s['end']['to_apex']:.2f}; to_sharp s/m/e {s['start']['to_outer_sharp']:.2f}/{s['mid']['to_outer_sharp']:.2f}/"
          f"{s['end']['to_outer_sharp']:.2f}; matcher apex ang {pm_apex['axis_angle_deg']:.2f} off "
          f"{pm_apex['transverse_offset_mm']:.2f} ok={ok_apex}; vs sharp off {pm_sharp['transverse_offset_mm']:.2f} ok={ok_sharp}")

# missing: AI lines with no gold match
unmatched_ai = res["unmatched_prediction_ids"]
out["per_gold"] = per
out["ai_lines"] = {k: {"bend_angle_deg": b.get("bend_angle_deg"), "length_mm": b.get("length_mm"),
                       "patches": b.get("patches"), "note": b.get("note"),
                       "label_line": b.get("label_line"), "fold_support_frac": b.get("fold_support_frac")}
                   for k, b in bench.items()}
out["unmatched_ai_lines"] = unmatched_ai


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


(OUT / "a2_match_bench.json").write_text(json.dumps(_clean(out), indent=1))
print(OUT / "a2_match_bench.json")
