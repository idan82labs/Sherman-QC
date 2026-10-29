"""POST-HOC, CONSUMED: score zs_lane_v2 outputs on the consumed TEST-B (61 scans) and compare with zs_lane_v1's one-time
result. TEST-B was consumed by zs_lane_v1 (testb/RESULTS.md); this re-score is NOT a held-out result and nothing in
zs_lane_v2 was tuned on it. Also scores a DEV CLI run (--cohort dev) for the lane-vs-dump consistency check.

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v2.score_posthoc --runs DIR --out JSON [--cohort testb|dev]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research.zeroshot.zs_lane_v2 import scoring
from research.zeroshot.zs_lane_v2.train import class_budget, paired_bootstrap, slim

LABEL = "POSTHOC_CONSUMED_TESTB: not a held-out result (TEST-B consumed by zs_lane_v1, 2026-10-25)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default="testb", choices=("testb", "dev"))
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    res = {"label": LABEL if a.cohort == "testb" else "DEV CLI run (consumed development)", "cohorts": {}}
    cohorts = ("testb", "testb_old") if a.cohort == "testb" else ("dev",)
    for coh in cohorts:
        specs = scoring.case_specs(coh)
        recs = [scoring.score_scan(specs[s], scoring.eval_lines_of(runs / "zs" / s / "eval_lines.json")) for s in specs]
        agg = scoring.aggregate(recs)
        out = {"agg": agg, "classes": class_budget(recs, specs), "per_case": slim(recs)}
        if coh in ("testb", "testb_old"):
            v1 = json.loads((scoring.PROG / "testb/RESULTS_per_case.json").read_text())[f"zs|{coh}"]
            out["vs_v1"] = paired_bootstrap(v1, recs)
            out["classes_v1"] = class_budget(v1, specs)
        res["cohorts"][coh] = out
        print(f"[{coh}] {res['label'][:40]}: P={agg['P']} R={agg['R']} micro={agg['micro_f1']} {agg.get('micro_ci95')} "
              f"macro={agg['family_macro_f1']} {agg.get('macro_ci95')} tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} "
              f"failed={agg['failed_or_missing_scans']}", flush=True)
        if "vs_v1" in out:
            pb = out["vs_v1"]
            print(f"   vs v1 (post-hoc): d_micro {pb['d_micro']} {pb['d_micro_ci95']} d_macro {pb['d_macro']} "
                  f"{pb['d_macro_ci95']} +/- {pb['families_better']}/{pb['families_worse']}", flush=True)
            print(f"   classes v1 {out['classes_v1']}\n   classes v2 {out['classes']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
