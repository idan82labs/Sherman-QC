"""zs_lane_v3 revision 2: DEV out-of-fold check that the runtime / memory changes keep the accuracy.

The revision-2 DEV dump (dev_dump.py with the face budget, through the gold guard and the geometry slot) is put
through exactly the v3 procedure of train.py: per outer fold the pooled GBM and the v2 post setting by inner grouped
CV over v2's ``final`` grid, the Z3-alone GBM on the same 4 training folds, then EXT and Z3R with the frozen
FixParams. It is compared with the frozen v3 OOF (``v3/oof/OOF_v3_per_case.json``) by the paired family bootstrap,
per family, and on the zero-bend controls and rolled scans. A second arm keeps the v3 fold models and post settings
(fitted on the v3 dump) and only feeds them the revision-2 candidates, which isolates the budget's effect and mirrors
production. The final models are NOT refit: revision 2 keeps the v3 models (sha256 ade6cde4... / 62e99d13...).

usage: .venv/bin/python -B -m research.zeroshot.zs_lane_v3.rev2_eval --dump DIR --v3-dump DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from research.zeroshot.zs_lane_v2 import scoring
from research.zeroshot.zs_lane_v2 import train as T2
from research.zeroshot.zs_lane_v3 import train as T3

KEYS = ("candidates", "X", "z2_lines", "evidence", "status")


def dump_identity(a: Path, b: Path, ids: list[str]) -> dict:
    same, diff, budgeted = [], [], []
    for s in ids:
        fa, fb = a / f"{s}.json", b / f"{s}.json"
        if not (fa.is_file() and fb.is_file()):
            continue
        ra, rb = json.loads(fa.read_text()), json.loads(fb.read_text())
        if ((rb.get("stages") or {}).get("budget") or {}).get("applied"):
            budgeted.append(s)
        eq = all(json.dumps(ra.get(k), sort_keys=True) == json.dumps(rb.get(k), sort_keys=True) for k in KEYS)
        (same if eq else diff).append(s)
    return {"compared": len(same) + len(diff), "identical": len(same), "differ": diff, "budgeted": budgeted,
            "unbudgeted_identical": sum(1 for s in same if s not in budgeted),
            "unbudgeted": sum(1 for s in same + diff if s not in budgeted),
            "unbudgeted_differ": [s for s in diff if s not in budgeted]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--v3-dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=scoring.BOOT_N)
    a = ap.parse_args(argv)
    specs = scoring.case_specs("dev")
    folds = scoring.dev_folds()
    dumps = T2.load_dumps(Path(a.dump), list(specs))
    ids = [s for s in specs if s in dumps]
    missing = [s for s in specs if s not in dumps]
    sc = T2.Scorer(specs)
    ident = dump_identity(Path(a.v3_dump), Path(a.dump), list(specs))
    print("dump identity:", {k: v for k, v in ident.items() if k != "differ"}, flush=True)
    sel_v2, P, chosen, _ = T2.nested_oof(dumps, specs, folds, "base", T2.GRIDS["final"], sc)
    Y = {s: T2.labels(dumps[s]["candidates"], T2.gold_lines(specs[s])) for s in ids}
    P3, _ = T3.z3_fold_probs(dumps, ids, folds, Y)
    post_of = {s: chosen[folds[s]][1] for s in ids}
    sel = {s: T3.lines_for(dumps[s], P[s], P3[s], post_of[s], T3.FIX) for s in ids}
    ev = {s: T3.as_eval(sel[s]) for s in ids}
    agg, recs = T2.evaluate(specs, ev, sc, n_boot=a.boot)
    # arm 2 (production-faithful, isolates the budget): the v3 fold models and post settings, fitted on the v3 dump,
    # applied to the revision-2 candidates (unbudgeted scans: identical inputs, so identical lines by construction)
    d3 = T2.load_dumps(Path(a.v3_dump), list(specs))
    _s, _p, chosen3, models3 = T2.nested_oof(d3, specs, folds, "base", T2.GRIDS["final"], sc)
    Y3 = {s: T2.labels(d3[s]["candidates"], T2.gold_lines(specs[s])) for s in ids}
    _p3, z3m = T3.z3_fold_probs(d3, ids, folds, Y3)
    selF = {}
    for s in ids:
        r, f = dumps[s], folds[s]
        X = T2.features(r, "base")
        pz = models3[f].predict(X) if len(X) else np.zeros(0)
        p3 = np.full(len(r["candidates"]), np.nan)
        rows = T3.z3_rows(r)
        if rows:
            p3[rows] = z3m[f].predict(r["X"][rows][:, :T3.Z3])
        selF[s] = T3.lines_for(r, pz, p3, chosen3[f][1], T3.FIX)
    evF = {s: T3.as_eval(selF[s]) for s in ids}
    aggF, recsF = T2.evaluate(specs, evF, sc, n_boot=a.boot)
    prog = scoring.PROG
    v3 = json.loads((prog / "v3/oof/OOF_v3_per_case.json").read_text())
    v1 = json.loads(T3.V1_PER_CASE.read_text())["zs|fixed|snap"]
    v1_ctrl = {x["case_id"] for x in v1 if x.get("disposition") == "ZERO_BEND_CONTROL"}
    rolled = set(T3.rolled_dev(specs))
    by3 = {x["case_id"]: x for x in v3}
    per_scan_changed = []
    for x in recs:
        y = by3.get(x["case_id"])
        k = ("true_positive", "false_positive", "false_negative")
        if y is None or any(x.get(f) != y.get(f) for f in k):
            per_scan_changed.append({"case_id": x["case_id"], "family": x.get("family_id"),
                                     "v3": None if y is None else [y.get(f) for f in k], "rev2": [x.get(f) for f in k],
                                     "budgeted": x["case_id"] in ident["budgeted"]})
    v3_lines = json.loads((prog / "v3/oof/OOF_v3_lines.json").read_text())
    key = lambda L: sorted((tuple(np.round(l["start"], 4)), tuple(np.round(l["end"], 4))) for l in L)  # noqa: E731
    lines_identical = [s for s in ids if key(ev[s]) == key(v3_lines.get(s, []))]
    out = {
        "dump": a.dump, "v3_dump": a.v3_dump, "missing_dumps": missing, "dump_identity": ident,
        "chosen_post": {str(f): {"index": k, "params": p.key()} for f, (k, p, _s) in chosen.items()},
        "rev2": {"agg": agg, "vs_v3": T2.paired_bootstrap(v3, recs, n_boot=a.boot),
                 "vs_v1": T2.paired_bootstrap(v1, recs, n_boot=a.boot),
                 "control_fp_lines": T3.per_scan_fp(recs, v1_ctrl), "rolled_fp": T3.per_scan_fp(recs, rolled),
                 "fixes": dict(Counter(l.get("fix") for v in sel.values() for l in v if l.get("fix")))},
        "rev2_fixed_v3_fold_models": {
            "agg": aggF, "vs_v3": T2.paired_bootstrap(v3, recsF, n_boot=a.boot),
            "vs_v1": T2.paired_bootstrap(v1, recsF, n_boot=a.boot),
            "control_fp_lines": T3.per_scan_fp(recsF, v1_ctrl), "rolled_fp": T3.per_scan_fp(recsF, rolled),
            "line_sets_identical_to_v3_oof": sum(1 for s in ids if key(evF[s]) == key(v3_lines.get(s, []))),
            "line_sets_differ": sorted(s for s in ids if key(evF[s]) != key(v3_lines.get(s, []))),
            "per_scan_counts_changed": [
                {"case_id": x["case_id"], "family": x.get("family_id"),
                 "v3": [by3[x["case_id"]].get(f) for f in ("true_positive", "false_positive", "false_negative")],
                 "rev2": [x.get(f) for f in ("true_positive", "false_positive", "false_negative")]}
                for x in recsF if x["case_id"] in by3 and any(
                    x.get(f) != by3[x["case_id"]].get(f) for f in ("true_positive", "false_positive", "false_negative"))]},
        "v3_frozen": {"control_fp_lines": T3.per_scan_fp(v3, v1_ctrl), "rolled_fp": T3.per_scan_fp(v3, rolled)},
        "line_sets_identical_to_v3_oof": len(lines_identical), "scans": len(ids),
        "line_sets_differ": sorted(set(ids) - set(lines_identical)),
        "per_scan_counts_changed": per_scan_changed,
    }
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "REV2_OOF_SUMMARY.json").write_text(json.dumps(out, indent=1, default=str))
    (od / "OOF_v3r2_per_case.json").write_text(json.dumps(T2.slim(recs), indent=1, default=str))
    (od / "OOF_v3r2_lines.json").write_text(json.dumps(ev, default=str))
    r = out["rev2"]
    q = out["rev2_fixed_v3_fold_models"]
    print(f"rev2 fixed v3 fold models: F1={aggF['micro_f1']} macro={aggF['family_macro_f1']} "
          f"tp/fp/fn={aggF['tp']}/{aggF['fp']}/{aggF['fn']} ctrlFP={aggF['controls']['fp_lines']} | vs v3 "
          f"{q['vs_v3']['d_micro']} {q['vs_v3']['d_micro_ci95']} {q['vs_v3']['families_better']}/"
          f"{q['vs_v3']['families_worse']} | line sets identical {q['line_sets_identical_to_v3_oof']}/{len(ids)}", flush=True)
    print(f"rev2: F1={agg['micro_f1']} {agg.get('micro_ci95')} macro={agg['family_macro_f1']} {agg.get('family_macro_ci95')} "
          f"tp/fp/fn={agg['tp']}/{agg['fp']}/{agg['fn']} ctrlFP={agg['controls']['fp_lines']} "
          f"| vs v3 {r['vs_v3']['d_micro']} {r['vs_v3']['d_micro_ci95']} {r['vs_v3']['families_better']}/"
          f"{r['vs_v3']['families_worse']} | line sets identical {len(lines_identical)}/{len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
