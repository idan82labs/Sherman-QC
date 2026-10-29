"""Run one named v4 procedure variant under nested LOFO, score it with the frozen matcher and append it to
results/DESIGN_LOG.jsonl (PROTOCOL.md section 5). Variants are declared in research/zeroshot/zs_lane_v4/variants.py.
usage (repo root): .venv/bin/python -B docs/production/runlog/zs_v4/tools/run_v4.py <variant> [--note TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v4 import evalkit as EK, lofo as LO, train as TR, variants as VA  # noqa: E402

RES = EK.V4RUN / "results"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true", help="engineering smoke test only (never for a design decision)")
    a = ap.parse_args(argv)
    t0 = time.time()
    spec = VA.get(a.variant)
    items = LO.load_items()
    data = TR.V4Data(items, hem=VA.hem_lines(spec, items), hyp=spec.hyp, stab=VA.stab_of(spec, items))
    pred, fitted = TR.lofo(data, spec, log=print)
    sc = EK.Scorer([it.s for it in items])
    agg, recs = LO.score(items, pred, sc)
    base = json.loads((RES / "LOFO_BASELINES.json").read_text())
    pc = base["per_case"]
    out = {"variant": a.variant, "spec": VA.describe(spec), "note": a.note,
           "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ALL": LO.summary(agg), "FE1": LO.summary(EK.aggregate([r for r in recs if r["origin"] == "FE1"])),
           "folds": fitted}
    slim = [{k: r.get(k) for k in ("case_id", "family_id", "origin", "disposition", "true_positive",
                                   "false_positive", "false_negative", "false_positive_lines")} for r in recs]
    for b in ("A0_v1_pinned", "A1_v1_retrained", "A3_v3_retrained"):
        p = EK.paired(pc[b], slim)
        out[f"paired_vs_{b}"] = {k: p[k] for k in ("d_micro", "d_micro_ci95", "d_macro", "d_macro_ci95", "families",
                                                   "families_better", "families_worse")}
    s = out["ALL"]
    print(f"{a.variant}: TP/FP/FN {s['tp']}/{s['fp']}/{s['fn']} micro {s['micro_f1']} {s['micro_ci95']} macro "
          f"{s['family_macro_f1']} {s['macro_ci95']} ctrl {s['controls']}  ({time.time() - t0:.0f}s)")
    for b in ("A0_v1_pinned", "A1_v1_retrained"):
        p = out[f"paired_vs_{b}"]
        print(f"   vs {b}: d_micro {p['d_micro']} {p['d_micro_ci95']} d_macro {p['d_macro']} {p['d_macro_ci95']} "
              f"better/worse fam {p['families_better']}/{p['families_worse']}")
    LO.dump(RES / f"v4/{a.variant}.json", {**out, "per_case": slim, "lines": LO.lines_json(pred)})
    if not a.no_log:
        with open(RES / "DESIGN_LOG.jsonl", "a") as fh:
            fh.write(json.dumps({k: out[k] for k in ("variant", "spec", "note", "when", "ALL", "FE1",
                                                     "paired_vs_A0_v1_pinned", "paired_vs_A1_v1_retrained")},
                                default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
