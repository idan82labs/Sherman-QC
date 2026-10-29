"""Score a cohort run of zs_lane_v1 (run_cohort.sh output: <runs>/<variant>/<sid>/eval_lines.json + run_record.json).

TEST-B (once, after the pre-registered run): gold of record = re-audit (cohort 'testb'), secondary column = old FRESH
labels ('testb_old'). DEV: the CLI consistency check (cohort 'dev'). Missing / FAILED outputs = all FN.
usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v1.score_testb --runs DIR --cohorts testb,testb_old
       --variants zs,z3,z2 --out RESULTS.json [--per-case PER_CASE.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v1 import scoring


def runtime(runs: Path, variant: str, ids: list[str]) -> dict:
    w, rss, st = [], [], {}
    for s in ids:
        p = runs / variant / s / "run_record.json"
        if not p.is_file():
            st["MISSING"] = st.get("MISSING", 0) + 1
            continue
        r = json.loads(p.read_text())
        w.append(r["wall_s"])
        rss.append(r.get("peak_rss_gib_process") or 0)
        st[r["status"]] = st.get(r["status"], 0) + 1
    a = np.array(w) if w else np.zeros(1)
    return {"statuses": st, "wall_median_s": round(float(np.median(a)), 2), "wall_p95_s": round(float(np.percentile(a, 95)), 2),
            "wall_max_s": round(float(a.max()), 2), "peak_rss_gib_max": round(float(max(rss or [0])), 2),
            "note": "per-process wall incl. interpreter start-up and model load, single thread"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cohorts", default="testb,testb_old")
    ap.add_argument("--variants", default="zs,z3,z2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-case", default="")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    res, pcs = {"schema": "sherman.zeroshot.zs_lane_v1.cohort_scores.v1", "runs": str(runs), "results": {}}, {}
    for cohort in a.cohorts.split(","):
        specs = scoring.case_specs(cohort)
        for v in a.variants.split(","):
            recs = [scoring.score_scan(specs[s], scoring.eval_lines_of(runs / v / s / "eval_lines.json")) for s in specs]
            agg = scoring.aggregate(recs)
            agg["runtime"] = runtime(runs, v, list(specs))
            res["results"][f"{v}|{cohort}"] = agg
            pcs[f"{v}|{cohort}"] = [{k: r.get(k) for k in ("case_id", "family_id", "disposition", "gold_count",
                                                           "prediction_count", "true_positive", "false_positive",
                                                           "false_negative", "false_positive_lines", "fn_causes",
                                                           "fp_causes", "fn_rows", "fp_rows",
                                                           "output_missing_or_failed")} for r in recs]
            print(f"{v}|{cohort}: P={agg['P']} R={agg['R']} micro={agg['micro_f1']} {agg.get('micro_ci95')} "
                  f"macro={agg['family_macro_f1']} {agg.get('macro_ci95')} tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} "
                  f"failed={agg['failed_or_missing_scans']} ctrl={agg['controls']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    if a.per_case:
        Path(a.per_case).write_text(json.dumps(pcs, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
