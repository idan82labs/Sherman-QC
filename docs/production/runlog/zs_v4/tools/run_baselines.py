"""LOFO baselines on the v4 pool (PROTOCOL.md sections 3-4): A0 v1 pinned (stored), A1 v1 retrained, A2 v3 pinned
(stored), A3 v3 retrained. Writes results/LOFO_BASELINES.json and results/lines_<arm>.json.
usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/run_baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import baselines as B, evalkit as EK, lofo as LO  # noqa: E402

RES = EK.V4RUN / "results"


def main() -> None:
    t0 = time.time()
    items = LO.load_items()
    sc = EK.Scorer([it.s for it in items])
    print(f"loaded {len(items)} items in {time.time() - t0:.1f}s", flush=True)
    preds = {"A0_v1_pinned": {it.case_id: EK.pinned_lines(it.s, "zs_lane_v1") for it in items},
             "A2_v3_pinned": {it.case_id: EK.pinned_lines(it.s, "zs_lane_v3") for it in items}}
    preds["A1_v1_retrained"], _ = LO.run(items, B.a1_fit, B.a1_predict)
    preds["A3_v3_retrained"], _ = LO.run(items, B.a3_fit, B.a3_predict)
    out, recs = {}, {}
    for arm, pred in preds.items():
        agg, rr = LO.score(items, pred, sc)
        recs[arm] = rr
        out[arm] = {"ALL": LO.summary(agg)}
        fe1 = [r for r in rr if r["origin"] == "FE1"]
        out[arm]["FE1"] = LO.summary(EK.aggregate(fe1))
        LO.dump(RES / f"lines_{arm}.json", LO.lines_json(pred))
        a = out[arm]["ALL"]
        print(f"{arm:18s} TP/FP/FN {a['tp']}/{a['fp']}/{a['fn']} micro {a['micro_f1']} {a['micro_ci95']} "
              f"macro {a['family_macro_f1']} {a['macro_ci95']} ctrl {a['controls']}", flush=True)
    pairs = {}
    for b in ("A1_v1_retrained", "A2_v3_pinned", "A3_v3_retrained"):
        pairs[f"{b} - A0_v1_pinned"] = EK.paired(recs["A0_v1_pinned"], recs[b])
    pairs["A3_v3_retrained - A1_v1_retrained"] = EK.paired(recs["A1_v1_retrained"], recs["A3_v3_retrained"])
    for k, p in pairs.items():
        print(k, {x: p[x] for x in ("d_micro", "d_micro_ci95", "d_macro", "d_macro_ci95", "families_better",
                                    "families_worse")})
    LO.dump(RES / "LOFO_BASELINES.json", {"schema": "sherman.zs_v4.lofo_baselines.v1",
                                          "label": "LOFO on the consumed v4 pool (internal estimate, not held-out); "
                                                   "A0/A2 are stored pinned outputs (in-sample on DEV)",
                                          "arms": out, "paired": pairs,
                                          "per_case": {a: [{k: r.get(k) for k in ("case_id", "family_id", "origin",
                                                       "disposition", "true_positive", "false_positive", "false_negative",
                                                       "false_positive_lines")} for r in rr] for a, rr in recs.items()}})
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
