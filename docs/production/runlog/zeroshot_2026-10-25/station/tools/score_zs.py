"""Score the zs lines of the DEV sets (A-D only; DEV gold of record, frozen matcher) per mesh. N1 is never scored."""
import glob
import json
import os
import sys

sys.path[:0] = ["/Users/idantal/ShermanQC/FullProject"]
from research.zeroshot.zs_lane_v2 import scoring as SC  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(f"{H}/meta.json"))
specs = SC.case_specs("dev")
out = {}
for tag in ("A", "B", "C", "D"):
    sid = meta[tag]["scan_id"]
    assert specs[sid]["tier"] == "DEV"
    for p in sorted(glob.glob(f"/private/tmp/claude-501/dense_ready/runs/{tag}_*_zs_t1/record.json")):
        rec = json.load(open(p))
        if rec.get("lines") is None:
            continue
        lines = [{"start": l["start_mm"], "end": l["end_mm"]} for l in rec["lines"]]
        r = SC.score_scan(specs[sid], lines, diagnose=False)
        out[os.path.basename(os.path.dirname(p))] = {k: r.get(k) for k in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative", "disposition")}
json.dump(out, open(f"{H}/zs_dev_scores.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
