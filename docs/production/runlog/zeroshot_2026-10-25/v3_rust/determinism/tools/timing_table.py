"""P50 / P95 / max process wall and peak RSS per run label from runs.json files (det_run / py_run format
{sid: rec} or portable_run format {sid/tN: rec}); only rc 0 (mesh scans run by the program) are counted.
usage: timing_table.py OUT.json label=path/runs.json[:tN] ..."""
import json, sys
import numpy as np
res = {}
for arg in sys.argv[2:]:
    lab, spec = arg.split("=", 1)
    path, _, key = spec.partition(":")
    runs = json.load(open(path))
    recs = {k.split("/")[0]: v for k, v in runs.items() if (not key or k.endswith("/" + key))}
    w = [v["wall_s"] for v in recs.values() if v["rc"] == 0]
    rss = [v.get("peak_rss_gib") or 0 for v in recs.values() if v["rc"] == 0]
    q = lambda p: round(float(np.percentile(w, p)), 3)
    res[lab] = {"n": len(w), "p50_s": q(50), "p95_s": q(95), "max_s": round(max(w), 3), "sum_s": round(sum(w), 1), "peak_rss_max_gib": round(max(rss), 3),
                "walls": recs and {k: v["wall_s"] for k, v in recs.items() if v["rc"] == 0}}
    print(lab, {k: v for k, v in res[lab].items() if k != "walls"})
json.dump(res, open(sys.argv[1], "w"), indent=1)
