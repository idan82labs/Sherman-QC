"""Assert that evalkit.FastScorer gives exactly the frozen matcher's TP/FP/FN (and control FP lines) on: the stored
pinned outputs of both lanes on every pool scan, and 20 random candidate subsets per cached scan (seed 0).
usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/check_fastscorer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import cache_io as CI, evalkit as EK  # noqa: E402


def main() -> None:
    pool = EK.load_pool()
    items = [SimpleNamespace(case_id=s["case_id"], gold=EK.gold_lines(s), zero_bend=s["zero_bend"], family=s["family5"])
             for s in pool]
    fs = EK.FastScorer(items)
    sc = EK.Scorer(pool)
    rng = np.random.default_rng(0)
    n = bad = 0
    for s in pool:
        sets = [EK.pinned_lines(s, "zs_lane_v1"), EK.pinned_lines(s, "zs_lane_v3")]
        d = CI.load(s["case_id"])
        if d is not None:
            C = d["v4"]["candidates"]
            for _ in range(20):
                if not C:
                    break
                k = rng.integers(0, len(C) + 1)
                idx = rng.choice(len(C), size=k, replace=False)
                sets.append([{"start": C[i]["start_s"], "end": C[i]["end_s"]} for i in idx])
        for lines in sets:
            r = sc.score(s["case_id"], lines)
            ref = (r.get("true_positive", 0), r.get("false_positive", 0), r.get("false_negative", 0),
                   int(r.get("false_positive_lines") or 0)) if r["disposition"] == "SCORED_COMPLETE_LABEL" else (
                0, 0, 0, int(r.get("false_positive_lines") or 0))
            got = fs.counts(s["case_id"], lines)
            n += 1
            if tuple(ref) != tuple(got):
                bad += 1
                print("MISMATCH", s["case_id"], ref, got)
    print(json.dumps({"line_sets": n, "mismatches": bad}))


if __name__ == "__main__":
    main()
