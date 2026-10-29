"""DEV line sets of a zs_rust run against a reference run (the frozen Python DEV CLI), both scored with the frozen
matcher (zs_lane_v2.scoring.score_scan -> research/broad_baseline/score.py score_case: 5 deg / 4 mm / 0.6 of both
spans, one-to-one Hungarian). Per scan: disposition, TP / FP / FN and the matched (gold, prediction) pairs must be
equal; the micro / macro aggregates are reported for both. A scan the binary refuses as a point cloud (no output in
the run) is taken from the reference run: the adapter runs the frozen Python revision-2 lane for it, not the binary.

DEV labels are consumed development data (the OOF cohort); nothing here is a held-out measurement.
usage: .venv/bin/python -B docs/.../v3_rust/tools/portable_matcher.py RUN_T_DIR REF_DIR OUT.json"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v2 import scoring

run, ref, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
specs = scoring.case_specs("dev")
recs_ref, recs_run, diff, pc = [], [], {}, []


def key(rec):
    m = sorted((x.get("gold_id"), x.get("prediction_id")) for x in rec.get("matches") or [])
    return {"disposition": rec.get("disposition"), "tp": rec.get("true_positive"), "fp": rec.get("false_positive"),
            "fn": rec.get("false_negative"), "fp_lines": rec.get("false_positive_lines"), "matches": m}


for sid in sorted(specs):
    lr = scoring.eval_lines_of(ref / sid / "eval_lines.json")
    p = run / sid / "eval_lines.json"
    if not p.is_file():
        pc.append(sid)
        lt = lr
    else:
        lt = scoring.eval_lines_of(p)
    a = scoring.score_scan(specs[sid], lr, diagnose=False)
    b = scoring.score_scan(specs[sid], lt, diagnose=False)
    recs_ref.append(a)
    recs_run.append(b)
    ka, kb = key(a), key(b)
    if ka != kb:
        diff[sid] = {"ref": ka, "run": kb}
agg = lambda r: {k: v for k, v in scoring.aggregate(r, ci=False).items() if k in ("scans", "tp", "fp", "fn", "micro_f1", "family_macro_f1", "controls")}  # noqa: E731
res = {"scans": len(specs), "line_sets_equal": len(specs) - len(diff), "differ": diff, "point_cloud_from_reference": pc,
       "ref_aggregate": agg(recs_ref), "run_aggregate": agg(recs_run), "run": str(run), "ref": str(ref)}
outp.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "differ"}, indent=1))
for k, v in diff.items():
    print(k, v)
