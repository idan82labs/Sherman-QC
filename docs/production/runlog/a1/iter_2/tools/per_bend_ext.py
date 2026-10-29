"""A1 iter-1 MEASURE: extended per-bend correctness given CAD (report-only; frozen eval code used read-only).

Adds to W/analysis/per_bend.json rows:
  - line_matches_relabel (Gold11 relabel v1, gold of record, when the case has one)
  - line_matches_goldset (GOLD_SET_V1 AI gold, when the A1 scan is in the gold set)
  - line_confirmed_any = BROAD or relabel or gold set
and summarises: correct-given-CAD under each gold, false-PASS proxies (strict = PASS on a published line that
no gold of record confirms; withheld = PASS on a bend with no published line), angle / flange sanity.
Custody: only cases_A1.json cases; gold-set rows only for A1 scans; DEV-CHECK / SEALED never read.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import score as sc  # noqa: E402
import bb_common as bb  # noqa: E402
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
from domains.production.engine.eval.freeze.physical import _engine_line_predictions, _gold_case_objects  # noqa: E402
from domains.production.engine.eval.line_metric import match_lines, prepare_gold  # noqa: E402

W = Path(sys.argv[1])
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
pb = json.loads((W / "analysis" / "per_bend.json").read_text())
rows = pb["rows"]
e3 = {}
cdirs = {}
for cid, c in cases.items():
    cdirs[cid] = W / "compiled" / "dev" / c["recipe_sku"]
    rp = W / "physical" / "E3" / cid / "record.json"
    if rp.is_file():
        e3[cid] = json.loads(rp.read_text())
preds = _engine_line_predictions(list(cases), e3, cdirs)

# relabel gold (gold of record for Gold11)
g2r_list, _ = relabel_gold_cases(doc["gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
usable_r, _ = prepare_gold(_gold_case_objects(g2r_list))
rel_match = {}
for cid, gold in usable_r.items():
    m = match_lines(gold, preds.get(cid) or [])
    rel_match[cid] = {x["prediction_id"] for x in m["matches"]}

# gold set v1 (A1 scans only)
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
a1_by_bid = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"] if (c.get("broad") or {}).get("broad_case_id")}
gs_match = {}
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    if pid not in a1_by_bid or not g.get("cad_path"):
        continue
    cid = a1_by_bid[pid]["case_ref"]
    path = W / "analysis" / "lanes" / "e3" / pid / "predictions_e3.json"
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
        # bb matcher ids are either the bend_id or the row index
        if isinstance(pidx, int) and 0 <= pidx < len(lines):
            ids.add(lines[pidx]["bend_id"])
        else:
            ids.add(str(pidx))
    gs_match[cid] = ids

for r in rows:
    cid, uid = r["case_id"], r["bend_uid"]
    r["line_matches_relabel"] = (uid in rel_match[cid]) if cid in rel_match else None
    r["line_matches_goldset"] = (uid in gs_match[cid]) if cid in gs_match else None
    r["line_confirmed_any"] = bool(r["line_matches_gold"] or r["line_matches_relabel"] or r["line_matches_goldset"])
    r["pass_on_line_no_gold_confirms"] = bool("PASS" in r["decisions"] and r["line_published"] and not r["line_confirmed_any"])

n = len(rows)
FORMED = {"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"}


def correct(r, key):
    ok = r[key]
    return bool(r["engine_state"] in FORMED and ok and r["angle_state"] == "MEASURED"
                and r["flange_A_state"] == "MEASURED" and r["flange_B_state"] == "MEASURED")


adev = [abs(r["angle_dev_deg"]) for r in rows if r["angle_dev_deg"] is not None]
fdev = [abs(x) for r in rows for x in r["flange_devs_mm"]]
dec = Counter(d for r in rows for d in r["decisions"])
summary = {
    "bends": n,
    "correct_vs_broad_gold": sum(correct(r, "line_matches_gold") for r in rows),
    "correct_vs_any_gold_of_record": sum(correct(r, "line_confirmed_any") for r in rows),
    "line_confirmed_any_rate": sum(r["line_confirmed_any"] for r in rows) / n if n else None,
    "rows_with_relabel_gold": sum(r["line_matches_relabel"] is not None for r in rows),
    "relabel_match": sum(bool(r["line_matches_relabel"]) for r in rows),
    "rows_with_goldset_gold": sum(r["line_matches_goldset"] is not None for r in rows),
    "goldset_match": sum(bool(r["line_matches_goldset"]) for r in rows),
    "false_pass_proxy_strict": sum(r["pass_on_line_no_gold_confirms"] for r in rows),
    "false_pass_proxy_strict_bends": [f"{r['case_id']}:{r['label']}" for r in rows if r["pass_on_line_no_gold_confirms"]],
    "pass_without_published_line": sum(r["pass_without_published_line"] for r in rows),
    "pass_without_published_line_bends": [f"{r['case_id']}:{r['label']}" for r in rows if r["pass_without_published_line"]],
    "angle_sanity": {"measured": len(adev), "abs_dev_gt_1deg": sum(a > 1 for a in adev), "abs_dev_gt_2deg": sum(a > 2 for a in adev),
                     "abs_dev_gt_5deg": sum(a > 5 for a in adev),
                     "p50_abs_deg": sorted(adev)[len(adev) // 2] if adev else None,
                     "p95_abs_deg": sorted(adev)[min(len(adev) - 1, int(0.95 * len(adev)))] if adev else None},
    "flange_sanity": {"measured_sides": len(fdev), "abs_dev_gt_0p5mm": sum(a > 0.5 for a in fdev), "abs_dev_gt_1mm": sum(a > 1 for a in fdev),
                      "abs_dev_gt_3mm": sum(a > 3 for a in fdev),
                      "p50_abs_mm": sorted(fdev)[len(fdev) // 2] if fdev else None,
                      "p95_abs_mm": sorted(fdev)[min(len(fdev) - 1, int(0.95 * len(fdev)))] if fdev else None},
    "decision_counts": dict(dec),
}
(W / "analysis" / "per_bend_ext.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
