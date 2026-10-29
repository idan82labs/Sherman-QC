"""Frozen-matcher comparison (5 deg / 4 mm / 60 % span, scripts/evaluate_f1_against_manual_gold)
of the Gold11 human lines against independent references for sc_cec08dfca9ab:
  * bench_labeling apex lines (outer-skin pairs, locus apex)      -> bench_lines/lines.json
  * bench_labeling sharp lines (outer virtual sharp)              -> bench_lines_sharp/lines.json
  * auditor-3 station apex lines (dense perpendicular sections)   -> stations_summary.json
Also reports the per-line signed offset decomposition gold - apex in the bend bisector frame.
Writes match_report.json. Read-only on inputs.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json

import numpy as np

from a3_common import OUT, ROOT, load_gold

sys.path.insert(0, str(ROOT))
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, match_lines, pair_metrics  # noqa: E402

MAIN = {"HB1": "B1", "HB2": "B2", "HB3": "B3", "HB4": "B4", "HB5": "B5", "HB6": "B6"}


def rec(bid, s, e):
    return {"bend_id": bid, "line_start": [float(v) for v in s], "line_end": [float(v) for v in e]}


def main():
    th = Thresholds()
    gold = [rec(b, s, e) for b, s, e in load_gold()]
    apex = json.loads((OUT / "bench_lines/lines.json").read_text())
    sharp = json.loads((OUT / "bench_lines_sharp/lines.json").read_text())
    st = json.loads((OUT / "stations_summary.json").read_text())
    refs = {
        "bench_apex": [rec(b["bend_id"], b["apex"]["start"], b["apex"]["end"]) for b in apex["bends"] if b.get("apex")],
        "bench_sharp": [rec(b["bend_id"], b["sharp"]["start"], b["sharp"]["end"]) for b in sharp["bends"]],
        "a3_station_apex": [rec("S" + k[2:], v["apex_line_raw"]["start"], v["apex_line_raw"]["end"])
                            for k, v in st["lines"].items() if v.get("apex_line_raw")],
    }
    report = {"thresholds": th.__dict__ if hasattr(th, "__dict__") else str(th), "pairs": {}, "hungarian": {}}
    for name, lines in refs.items():
        report["hungarian"][name] = match_lines(gold, lines, th)
        rows = {}
        for g in gold:
            gl = np.array([g["line_start"], g["line_end"]])
            best = None
            for r in lines:
                rl = np.array([r["line_start"], r["line_end"]])
                m = pair_metrics(gl, rl)
                if best is None or (m["axis_angle_deg"] < 10 and m["transverse_offset_mm"] < best[1]["transverse_offset_mm"]):
                    if m["axis_angle_deg"] < 10:
                        best = (r["bend_id"], m)
            if best:
                rows[g["bend_id"]] = {"ref": best[0], **{k: round(v, 3) for k, v in best[1].items()},
                                      "is_match": bool(is_match(best[1], th))}
        report["pairs"][name] = rows
    # what-if: gold translated onto the matching bench apex line (pure offset removal) still
    # overshooting? -> span-only test using bench apex line extents
    span_only = {}
    for g in gold:
        ref = next(r for r in refs["bench_apex"] if r["bend_id"] == MAIN[g["bend_id"]])
        gl = np.array([g["line_start"], g["line_end"]])
        rl = np.array([ref["line_start"], ref["line_end"]])
        d = (rl[1] - rl[0]) / np.linalg.norm(rl[1] - rl[0])
        # project gold endpoints onto the apex line (removes the transverse offset only)
        proj = np.array([rl[0] + ((p - rl[0]) @ d) * d for p in gl])
        m = pair_metrics(proj, rl)
        span_only[g["bend_id"]] = {k: round(v, 3) for k, v in m.items()}
        span_only[g["bend_id"]]["is_match_after_offset_removed"] = bool(is_match(m, th))
    report["span_only_after_projecting_gold_onto_bench_apex"] = span_only
    (OUT / "match_report.json").write_text(json.dumps(report, indent=1))
    for name in refs:
        h = report["hungarian"][name]
        print(f"== {name}: TP {h['true_positive']} / gold {h['gold_count']} / ref {h['prediction_count']}")
        for gid, r in report["pairs"][name].items():
            print(f"   {gid} vs {r['ref']}: angle {r['axis_angle_deg']:.2f} offset {r['transverse_offset_mm']:.2f} "
                  f"gold_ov {r['gold_span_overlap']:.2f} ref_ov {r['prediction_span_overlap']:.2f} "
                  f"len g/r {r['gold_length_mm']:.1f}/{r['prediction_length_mm']:.1f} match={r['is_match']}")
    print("== span-only (gold projected onto bench apex line)")
    for gid, r in span_only.items():
        print(f"   {gid}: gold_ov {r['gold_span_overlap']:.2f} ref_ov {r['prediction_span_overlap']:.2f} match={r['is_match_after_offset_removed']}")


if __name__ == "__main__":
    main()
