"""Score a Z3 run directory on DEV: candidate recall (all candidates), and gated line-F1 with error budget.

usage: python -m research.zeroshot.z3_slicing.score_run RUN_DIR [--gate NAME] [--json OUT] [--ids ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, zscore, gate as G  # noqa: E402


def load_run(run: Path, ids: list[str]) -> dict[str, dict]:
    out = {}
    for s in ids:
        p = run / f"{s}.json"
        if p.is_file():
            out[s] = json.loads(p.read_text())
    return out


def evaluate(run_data: dict[str, dict], inv: dict, select, diagnose: bool = True) -> tuple[dict, list[dict]]:
    recs = []
    for sid, d in run_data.items():
        lines = select(d, sid)
        rec = zscore.score_lines(sid, inv[sid], lines, diagnose=diagnose)
        rec["n_pred"] = len(lines)
        recs.append(rec)
    return zscore.aggregate(recs), recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gate", default="none")
    ap.add_argument("--json", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--per-case", action="store_true")
    a = ap.parse_args(argv)
    inv = custody.dev_inventory()
    ids = [s for s in a.ids.split(",") if s] or custody.dev_ids()
    data = load_run(Path(a.run), ids)
    sel_all = lambda d, s: d.get("candidates", [])  # noqa: E731
    agg_all, recs_all = evaluate(data, inv, sel_all, diagnose=False)
    gold = agg_all["gold"]
    print(f"scans={len(data)} ALL candidates: recall={agg_all['R']:.3f} TP={agg_all['tp']} gold={gold} "
          f"cands={agg_all['pred']} ratio={agg_all['pred'] / max(gold, 1):.2f}")
    out = {"run": a.run, "all": agg_all}
    if a.gate != "none":
        sel = G.GATES[a.gate]
        agg, recs = evaluate(data, inv, sel)
        print(f"GATE {a.gate}: P={agg['P']:.3f} R={agg['R']:.3f} F1={agg['micro_f1']:.3f} macro={agg['family_macro_f1']:.3f} "
              f"tp={agg['tp']} fp={agg['fp']} fn={agg['fn']} ctrl_fp={agg['controls']['fp_lines']}")
        print("  FN", agg["fn_causes"], " FP", agg["fp_causes"])
        out["gate"] = {"name": a.gate, **agg}
        if a.per_case:
            for r in sorted(recs, key=lambda r: (r.get("f1") or 0)):
                if r.get("disposition") == "SCORED_COMPLETE_LABEL":
                    print(f"  {r['case_id']} {r['family_id']:>12s} tp={r['true_positive']} fp={r['false_positive']} fn={r['false_negative']} "
                          f"FN{r.get('fn_causes')} FP{r.get('fp_causes')}")
        out["per_case"] = recs
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
