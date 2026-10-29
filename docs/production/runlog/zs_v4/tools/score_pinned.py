"""Score the stored pinned-lane outputs (zs_lane_v1 pin 0087fb23, zs_lane_v3 rev3 pin 711bd079) on the whole v4 pool.

Stored outputs only (coverage run seq 41 + FE1 RUN seq 44); nothing is re-run. Frozen matcher, family5 bootstrap
n 10000 seed 20261018. POST-HOC and partly IN-SAMPLE (both lanes' GBMs were fit on DEV; v3's fix values were set
on DEV with N1 in view) -- descriptive only, never a held-out claim.
Writes results/PINNED_POOL.json (+ per-case diagnosis rows).  usage (repo root): .venv/bin/python -B <this>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import evalkit as EK  # noqa: E402

OUT = EK.V4RUN / "results/PINNED_POOL.json"
LANES = ["zs_lane_v1", "zs_lane_v3"]


def main() -> None:
    pool = EK.load_pool()
    sc = EK.Scorer(pool)
    res, per_case = {}, {}
    for lane in LANES:
        recs = []
        for s in pool:
            r = sc.score(s["case_id"], EK.pinned_lines(s, lane), diagnose=True)
            recs.append(r)
        per_case[lane] = recs
        out = {"ALL": EK.slim(EK.aggregate(recs))}
        for org in ("DEV", "TEST-B", "N1", "FE1"):
            sub = [r for r in recs if r["origin"] == org]
            out[org] = EK.slim(EK.aggregate(sub))
        res[lane] = out
    res["paired_v3_minus_v1"] = {"ALL": EK.paired(per_case["zs_lane_v1"], per_case["zs_lane_v3"])}
    for org in ("DEV", "TEST-B", "FE1"):
        a = [r for r in per_case["zs_lane_v1"] if r["origin"] == org]
        b = [r for r in per_case["zs_lane_v3"] if r["origin"] == org]
        res["paired_v3_minus_v1"][org] = EK.paired(a, b)
    keep = ("case_id", "family_id", "origin", "disposition", "gold_count", "prediction_count", "true_positive",
            "false_positive", "false_negative", "false_positive_lines", "fn_rows", "fp_rows", "matches",
            "output_missing_or_failed")
    doc = {"schema": "sherman.zs_v4.pinned_pool.v1",
           "label": "POST-HOC on the consumed v4 pool; DEV in-sample for both pinned lanes; not held-out",
           "results": res, "per_case": {l: [{k: r.get(k) for k in keep} for r in per_case[l]] for l in LANES}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str))
    for lane in LANES:
        a = res[lane]["ALL"]
        print(lane, a["scans"], a["families"], a["tp"], a["fp"], a["fn"], a["micro_f1"], a["micro_ci95"],
              a["family_macro_f1"], a["macro_ci95"], a["controls"], a["fn_causes"], a["fp_causes"])
        for org in ("DEV", "TEST-B", "N1", "FE1"):
            b = res[lane][org]
            print("   ", org, b["scans"], b["tp"], b["fp"], b["fn"], b["micro_f1"], b["family_macro_f1"], b["controls"])
    p = res["paired_v3_minus_v1"]["ALL"]
    print("paired v3-v1", {k: p[k] for k in ("d_micro", "d_micro_ci95", "d_macro", "d_macro_ci95", "families",
                                             "families_better", "families_worse")})
    print(p["changed"])


if __name__ == "__main__":
    main()
