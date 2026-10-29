"""Per-bend correctness given CAD over EVERY A1 case the engine ran (report-only; frozen eval code read-only).

usage: per_bend_all.py PHYS_DIR TAG  (reads PHYS_DIR/physical_TAG, analysis_TAG, physical_scorecard_TAG.json)
Expected state: the frozen G3 truth row when the case has one (staged scans), else FORMED (complete scans).
Correct (expected FORMED): engine FORMED-like, angle MEASURED, both flange sides MEASURED (strict; 'riser-aware'
also accepts a side UNKNOWN(UNSUPPORTED_FEATURE) on a joggle riser), and its published line matches a GOLD_SET_V1
gold line (A1 scans in the gold set; else the Gold11 relabel; else the line criterion is not applied).
Correct (expected NOT_FORMED): engine NOT_YET_FORMED. NOT_DETERMINABLE truth: not scored.
False-PASS proxy: a PASS decision on a bend expected NOT_FORMED, or on a formed bend whose line no gold confirms.
Custody: cases_A1.json only; nothing SEALED.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
import bb_common as bb  # noqa: E402
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
from domains.production.engine.eval.freeze.physical import _engine_line_predictions, _gold_case_objects  # noqa: E402
from domains.production.engine.eval.line_metric import match_lines, prepare_gold  # noqa: E402

P, TAG = Path(sys.argv[1]), sys.argv[2]
PHYS, AN = P / f"physical_{TAG}", P / f"analysis_{TAG}"
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
score = json.loads((P / f"physical_scorecard_{TAG}.json").read_text())
truth = {(r["case_ref"], r["bend_uid"]): r["truth"] for r in score["G3"]["rows"]}
e3, cdirs = {}, {}
for cid, c in cases.items():
    cdirs[cid] = P / "compiled" / "dev" / c["recipe_sku"]
    rp = PHYS / "E3" / cid / "record.json"
    if rp.is_file():
        e3[cid] = json.loads(rp.read_text())
preds = _engine_line_predictions(list(cases), e3, cdirs)
g2r_list, _ = relabel_gold_cases(doc["gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
usable_r, _ = prepare_gold(_gold_case_objects(g2r_list))
rel_match = {cid: {x["prediction_id"] for x in match_lines(gold, preds.get(cid) or [])["matches"]} for cid, gold in usable_r.items()}
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
a1_by_bid = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"] if (c.get("broad") or {}).get("broad_case_id")}
gs_match = {}
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    if pid not in a1_by_bid or not g.get("cad_path"):
        continue
    cid = a1_by_bid[pid]["case_ref"]
    path = AN / "lanes" / "e3" / pid / "predictions_e3.json"
    if not path.is_file():
        continue
    pred = bb.load_eval_lines(path)
    if pid != g["case_id"]:
        pred = dict(pred, scan={})
    r = sc.score_case({"case_id": g["case_id"], "tier": g.get("tier"), "family_id": g["family_id"], "gold": g["gold"],
                       "label_complete": g.get("label_complete", True)}, frame_records=None, preloaded_predictions=pred)
    if r.get("matches") is None:
        continue
    lines = pred.get("owned_bend_regions") or []
    ids = set()
    for mm in r["matches"]:
        pidx = mm.get("prediction_id")
        ids.add(lines[pidx]["bend_id"] if isinstance(pidx, int) and 0 <= pidx < len(lines) else str(pidx))
    gs_match[cid] = ids
FORMED = {"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"}
rows = []
for cid in sorted(e3):
    if e3[cid].get("status") != "OK":
        continue
    res = json.loads((PHYS / "E3" / cid / "engine" / "result.json").read_text())
    published = {x["bend_id"] if isinstance(x, dict) and "bend_id" in x else x.get("prediction_id") for x in (preds.get(cid) or []) if isinstance(x, dict)}
    for b in res["bends"]:
        uid = b["bend_uid"]
        t = truth.get((cid, uid))
        if t is None:
            if cases[cid]["stage_class"] != "COMPLETE":
                continue
            t = "FORMED"
        if t == "NOT_DETERMINABLE":
            continue
        chs = b["characteristics"]
        ang = [c for c in chs if c["kind"] == "INCLUDED_ANGLE"]
        fl = [c for c in chs if c["kind"] == "FLANGE_LENGTH"]
        decs = [c.get("decision") for c in chs]
        if cid in gs_match:
            line_ok, src = uid in gs_match[cid], "GOLD_SET_V1"
        elif cid in rel_match:
            line_ok, src = uid in rel_match[cid], "RELABEL"
        else:
            line_ok, src = None, None
        st = b["state"]
        if t == "FORMED":
            ang_ok = bool(ang) and ang[0].get("state") == "MEASURED"
            fl_strict = all(c.get("state") == "MEASURED" for c in fl)
            fl_riser = all(c.get("state") == "MEASURED" or c.get("reason") == "UNSUPPORTED_FEATURE" for c in fl)
            base = st in FORMED and ang_ok and line_ok is not False
            ok_strict, ok_riser = base and fl_strict, base and fl_riser
        else:
            ok_strict = ok_riser = st == "NOT_YET_FORMED"
        fp_proxy = "PASS" in decs and (t == "NOT_FORMED" or line_ok is False)
        rows.append({"case": cid, "family": cases[cid]["family"], "label": b["label"], "truth": t, "state": st,
                     "line_ok": line_ok, "line_src": src, "correct_strict": ok_strict, "correct_riser_aware": ok_riser,
                     "pass_proxy": fp_proxy, "decisions": dict(Counter(decs))})
n = len(rows)
fam = defaultdict(lambda: [0, 0])
for r in rows:
    fam[r["family"]][0] += 1
    fam[r["family"]][1] += int(r["correct_riser_aware"])
summary = {"bends": n, "formed_expected": sum(r["truth"] == "FORMED" for r in rows),
           "correct_strict": sum(r["correct_strict"] for r in rows), "correct_riser_aware": sum(r["correct_riser_aware"] for r in rows),
           "rate_strict": sum(r["correct_strict"] for r in rows) / n if n else None,
           "rate_riser_aware": sum(r["correct_riser_aware"] for r in rows) / n if n else None,
           "family_macro_riser_aware": sum(c / t for t, c in fam.values()) / len(fam) if fam else None,
           "false_pass_proxy": sum(r["pass_proxy"] for r in rows),
           "false_pass_proxy_bends": [f"{r['case']}:{r['label']}" for r in rows if r["pass_proxy"]],
           "not_correct": [f"{r['case']}:{r['label']}:{r['truth'][:3]}->{r['state'][:3]}:line={r['line_ok']}" for r in rows if not r["correct_riser_aware"]]}
(AN / "per_bend_all.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
