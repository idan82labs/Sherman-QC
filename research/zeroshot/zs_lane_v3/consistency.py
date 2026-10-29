"""DEV CLI consistency and runtime for zs_lane_v3: the frozen CLI's outputs (run_cohort.sh, guard + slot,
--strict-pins, final models) against the dump pipeline with the same final models, plus the per-process runtime
next to zs_lane_v1's and zs_lane_v2's DEV CLI runs. Also scores the CLI outputs (DEV, final models = in-sample;
reported only as a consistency check, never as an accuracy claim).

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3.consistency --dump DIR --cli DIR --out JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v2 import scoring
from research.zeroshot.zs_lane_v2 import train as T2
from research.zeroshot.zs_lane_v3 import core, lane

PROG = scoring.PROG
CONFIG = core.REPO / "research/zeroshot/zs_lane_v3/config_v3.json"


def _key(lines):
    return sorted((tuple(np.round(a, 4)), tuple(np.round(b, 4))) for a, b in lines)


def _runtime(root: Path, ids: list[str]) -> dict:
    w, rss, sel = [], [], []
    for s in ids:
        p = root / s / "run_record.json"
        if p.is_file():
            r = json.loads(p.read_text())
            w.append(float(r["wall_s"]))
            rss.append(float(r.get("peak_rss_gib_process") or 0.0))
            sp = (r.get("stages") or {}).get("select_post") or {}
            if "wall" in sp:
                sel.append(float(sp["wall"]))
    if not w:
        return {"n": 0}
    out = {"n": len(w), "wall_median_s": round(float(np.median(w)), 2), "wall_p95_s": round(float(np.percentile(w, 95)), 2),
           "wall_max_s": round(float(np.max(w)), 2), "wall_sum_s": round(float(np.sum(w)), 1),
           "peak_rss_gib_max": round(float(np.max(rss)), 2)}
    if sel:
        out.update(select_post_median_s=round(float(np.median(sel)), 4), select_post_max_s=round(float(np.max(sel)), 4))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    v = cfg["variants"]["zs"]
    zs_model, z3_model = lane.load_models(v)
    specs = scoring.case_specs("dev")
    ids = list(specs)
    dumps = T2.load_dumps(Path(a.dump), ids)
    cli = Path(a.cli) / "zs"
    same, diff, pins_bad, missing, recs = 0, {}, [], [], []
    for s in ids:
        p = cli / s / "eval_lines.json"
        if not p.is_file():
            missing.append(s)
            recs.append(scoring.score_scan(specs[s], None, diagnose=False))
            continue
        pay = json.loads(p.read_text())
        if not pay.get("conversion", {}).get("pins_ok"):
            pins_bad.append(s)
        got = [(r["line_start"], r["line_end"]) for r in pay.get("owned_bend_regions") or []]
        r = dumps[s]
        want, _, _ = core.select(r["candidates"], r["X"], r.get("evidence"), r.get("z2_lines") or [], v, zs_model,
                                 z3_model)
        if _key(got) == _key([(l["start_s"], l["end_s"]) for l in want]):
            same += 1
        else:
            diff[s] = {"cli": len(got), "dump": len(want)}
        recs.append(scoring.score_scan(specs[s], scoring.eval_lines_of(p), diagnose=False))
    agg = scoring.aggregate(recs, ci=False)
    res = {"scans": len(ids), "identical_line_sets": same, "differ": diff, "missing": missing,
           "pins_not_ok": pins_bad, "code_fingerprint": cfg["code_fingerprint"]["sha256"],
           "in_sample_final_model_score_NOT_A_CLAIM": {k: agg[k] for k in ("micro_f1", "tp", "fp", "fn")},
           "runtime": {"v3": _runtime(cli, ids), "v2": _runtime(PROG / "v2/dev_cli/zs", ids),
                       "v1": _runtime(PROG / "integration/dev_cli/zs", ids)}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("scans", "identical_line_sets", "missing", "pins_not_ok", "runtime")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
