"""Auditor-2 (sc_4f8188b4d1a8): where do the engine best_f1 lines sit relative to the auditor's own
apex lines, and how do gold / AI-apex lines score against them with the frozen matcher?  Read-only.
Writes a2_engine_check.json."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines, pair_metrics, _endpoints  # noqa: E402

from a2_common import OUT, SCAN_SHA, gold_lines  # noqa: E402

PRED = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/runs/best_f1/"
            "sc_4f8188b4d1a8/predictions_best_f1.json")
pred = json.loads(PRED.read_text())
assert (pred.get("scan") or {}).get("sha256") in (None, SCAN_SHA), pred.get("scan")
eng = [r for r in pred["owned_bend_regions"] if r.get("line_start") is not None or r.get("span_endpoints")]
G = gold_lines()
gold_records = [{"bend_id": k, "line_start": v[0].tolist(), "line_end": v[1].tolist()} for k, v in sorted(G.items())]
lines12 = json.loads((OUT / "bench_lines12" / "lines.json").read_text())
ai_records = [{"bend_id": "AI_" + b["bend_id"], "line_start": b["label_line"]["start"], "line_end": b["label_line"]["end"]}
              for b in lines12["bends"] if b.get("status") == "ok"]
thr = Thresholds()
out = {"pred_path": str(PRED), "pred_status": pred.get("status"), "engine_line_count": len(eng)}
for name, gref in (("gold_vs_engine", gold_records), ("ai_vs_engine", ai_records)):
    r = match_lines(gref, eng, thr)
    out[name] = {k: r[k] for k in ("true_positive", "false_positive", "false_negative", "unmatched_gold_ids",
                                   "unmatched_prediction_ids", "matches")}
    print(name, {k: r[k] for k in ("true_positive", "false_positive", "false_negative", "unmatched_gold_ids",
                                   "unmatched_prediction_ids")})
# each engine line vs nearest AI apex line and nearest gold line (by metrics, no gating)
rows = []
for e in eng:
    pe = _endpoints(e)
    best_ai = min(((pair_metrics(np.asarray([a["line_start"], a["line_end"]]), pe), a["bend_id"]) for a in ai_records),
                  key=lambda t: (t[0]["axis_angle_deg"] > 10, t[0]["transverse_offset_mm"]))
    best_g = min(((pair_metrics(np.asarray([g["line_start"], g["line_end"]]), pe), g["bend_id"]) for g in gold_records),
                 key=lambda t: (t[0]["axis_angle_deg"] > 10, t[0]["transverse_offset_mm"]))
    row = {"engine_id": e.get("bend_id"), "engine_len": float(np.linalg.norm(pe[1] - pe[0])),
           "nearest_ai": best_ai[1], "vs_ai": best_ai[0], "nearest_gold": best_g[1], "vs_gold": best_g[0]}
    rows.append(row)
    print(f"engine {row['engine_id']:>16s} len {row['engine_len']:6.1f} | nearest AI {best_ai[1]:>7s} ang "
          f"{best_ai[0]['axis_angle_deg']:5.2f} off {best_ai[0]['transverse_offset_mm']:6.2f} govl "
          f"{best_ai[0]['gold_span_overlap']:.2f} povl {best_ai[0]['prediction_span_overlap']:.2f} | nearest gold "
          f"{best_g[1]} ang {best_g[0]['axis_angle_deg']:5.2f} off {best_g[0]['transverse_offset_mm']:6.2f} govl "
          f"{best_g[0]['gold_span_overlap']:.2f} povl {best_g[0]['prediction_span_overlap']:.2f}")
out["engine_lines"] = rows
(OUT / "a2_engine_check.json").write_text(json.dumps(out, indent=1, default=float))
print(OUT / "a2_engine_check.json")
