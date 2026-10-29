"""Does the engine's unplanned crease on 28870000 FULL land on the recipe-excluded gold line? (report-only; frozen
matcher read-only; the crease is not a line prediction of the frozen metric). usage: crease_check.py PHYS_DIR"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
from domains.production.engine.eval.line_metric import pair_metrics, is_match  # noqa: E402
W = Path(sys.argv[1])
cid, bid = "pc_38091ac91d9d", "sc_38091ac91d9d"
res = json.loads((W / "physical/E3" / cid / "engine/result.json").read_text())
reg = res["inspection_validity"]["registration"]; M = np.asarray(reg["raw_to_frame"], dtype=float).reshape(4, 4)
uf = float(reg.get("unit_factor") or 1.0)
Minv = np.linalg.inv(M)
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
g = next(x for x in gl["cases"] if (x.get("prediction_case_id") or x["case_id"]) == bid)
gold = sc.load_gold(g["gold"], frame_records=None)["lines"]
for c in res.get("unplanned_creases") or []:
    p = np.asarray(c["polyline_mm"], dtype=float)
    raw = (np.c_[p[[0, -1]], np.ones(2)] @ Minv.T)[:, :3] / uf
    for gline in gold:
        L = np.asarray([gline["line_start"], gline["line_end"]], dtype=float)
        m = pair_metrics(L, raw)
        print(c["crease_id"], c["reason"], round(c["included_deg"], 1), "vs gold", gline["bend_id"],
              {k: round(v, 2) for k, v in m.items()}, "MATCH" if is_match(m) else "")
