import sys, json, math, numpy as np
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from scripts.evaluate_f1_against_manual_gold import pair_metrics, Thresholds, is_match, match_lines
W = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_08686")
T = json.load(open(W / "true_fold_lines.json"))
gold = {b["bend_id"]: np.array([b["line_start"], b["line_end"]]) for b in json.load(open("/Users/idantal/ShermanQC/FullProject/output/bend_label_02_10839000/bend_instance_labels.json"))["manual_bends"]}
true = {k: np.array(v["world"]) for k, v in T.items() if "world" in v}
print("true folds:", {k: round(float(np.linalg.norm(v[1] - v[0])), 1) for k, v in true.items()})
for g, gl in gold.items():
    best = None
    for k, tl in true.items():
        m = pair_metrics(gl, tl)
        if best is None or (m["axis_angle_deg"] < 10 and m["transverse_offset_mm"] < best[1]["transverse_offset_mm"]):
            if m["axis_angle_deg"] < 10:
                best = (k, m)
        if m["axis_angle_deg"] < 10:
            print(f"{g} vs {k}: angle {m['axis_angle_deg']:.2f} transverse {m['transverse_offset_mm']:.2f} gold_ov {m['gold_span_overlap']:.2f} pred_ov {m['prediction_span_overlap']:.2f} -> {'MATCH' if is_match(m, Thresholds()) else 'no'}")
# full Hungarian: gold (4) vs true folds (8) as if the true folds were a perfect apex-locus prediction
gl = [{"bend_id": k, "line_start": v[0].tolist(), "line_end": v[1].tolist()} for k, v in gold.items()]
pl = [{"bend_id": k, "line_start": v[0].tolist(), "line_end": v[1].tolist()} for k, v in true.items()]
import inspect
print(inspect.signature(match_lines))
try:
    r = match_lines(gl, pl, Thresholds())
    print(json.dumps({k: v for k, v in r.items() if k in ("summary", "matches", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1")}, default=str)[:1500])
except Exception as e:
    print("match_lines call failed:", e)
