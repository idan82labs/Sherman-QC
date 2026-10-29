"""A1 iteration-0 post-analysis (report-only, uses the frozen eval code read-only; edits nothing frozen).

Inputs: the A1 work dir W (physical/E3/<case>/record.json, compiled/dev/<sku>), the frozen A1 case document,
the frozen reference lanes. Outputs (into W/analysis/):
  - lanes/e3/<broad_case_id>/predictions_e3.json  (eval-lines files, raw frame, apex locus) for Jev / gold set
  - subset_scores.json  engine vs best_f1 lane on the SAME scans (BROAD runnable subset, G2(a) human + relabel)
  - per_bend.json       per-bend correctness given CAD (report-only proxy) + worst-bend ranking
Custody: only cases_A1.json cases; nothing *SEALED*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]

from domains.production.engine.eval.freeze.physical import (  # noqa: E402
    _engine_line_predictions, _gold_case_objects, broad_line_scores, planned_population, truth_relevant)
from domains.production.engine.eval.line_metric import match_lines, prepare_gold, score_line_metric  # noqa: E402

W = Path(sys.argv[1])  # the physical work dir (W/phys)
OUT = W / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
lanes = json.loads((FZ / "broad_reference_lanes_A1.json").read_text())["lanes"]
cases = {c["case_ref"]: c for c in doc["cases"]}

e3, cdirs, pops = {}, {}, {}
for cid, c in cases.items():
    cdirs[cid] = W / "compiled" / "dev" / c["recipe_sku"]
    pops[cid] = planned_population(FZ / "recipes" / f"{c['recipe_sku']}.recipe.json", cdirs[cid])
    rp = W / "physical" / "E3" / cid / "record.json"
    if rp.is_file():
        e3[cid] = json.loads(rp.read_text())
ran = sorted(e3)
ok = sorted(k for k, r in e3.items() if r.get("status") == "OK")
refused = sorted(cid for cid, c in cases.items() if truth_relevant(c))
preds = _engine_line_predictions(list(cases), e3, cdirs)

# ---- eval-lines files (raw frame, apex locus) per broad scan id
lane_root = OUT / "lanes" / "e3"
for cid, rows in preds.items():
    bid = (cases[cid].get("broad") or {}).get("broad_case_id")
    if not bid:
        continue
    d = lane_root / bid
    d.mkdir(parents=True, exist_ok=True)
    ext = {b["bend_uid"]: b for b in ((e3[cid].get("extraction") or {}).get("bends") or [])}
    obr = []
    for r in rows:
        b = ext.get(r["bend_id"], {})
        ang = b.get("angle") or {}
        obr.append({"bend_id": r["bend_id"], "line_start": r["line_start"], "line_end": r["line_end"],
                    "origin": "e3_recipe_guided", "status": b.get("state"),
                    "measured_angle": ang.get("actual"), "target_angle": ang.get("expected")})
    (d / "predictions_e3.json").write_text(json.dumps({
        "schema": "sherman.broad_baseline.eval_lines.v1", "lane": "e3", "status": "OK", "frame": "scan_raw",
        "frame_definition": "coordinates of the scan file exactly as stored",
        "scan": {"path": cases[cid]["scan_path"], "sha256": cases[cid]["scan_sha256"]},
        "locus": "OUTER_APEX (measured_outer_apex, frame_to_raw)", "line_count": len(obr),
        "owned_bend_regions": obr}, indent=1))


def lane_lines(lane: str, ids):
    return {k: (lanes[lane].get(k) or {}).get("lines") or [] for k in ids if k in lanes[lane]}


def brief(s):
    return {k: s.get(k) for k in ("line_cases", "families", "true_positive", "false_positive", "false_negative",
                                  "micro", "family_macro_f1", "family_macro_f1_with_controls")}


broad_ids = [c for c in doc["lists"]["BROAD"]]
gold_b = [g for g in doc["broad_gold_cases"] if g["case_id"] in broad_ids]
sub_ids = [c for c in broad_ids if c in ok]
gold_sub = [g for g in gold_b if g["case_id"] in sub_ids]
res = {"runnable_ok_cases": ok, "e3_records": ran, "refused_truth_relevant": refused,
       "BROAD_all26": {"engine": brief(broad_line_scores(gold_b, preds)),
                       "best_f1": brief(broad_line_scores(gold_b, lane_lines("best_f1", broad_ids)))},
       "BROAD_same_scans_engine_ran": {"cases": sub_ids,
                                       "engine": brief(broad_line_scores(gold_sub, preds)),
                                       "best_f1": brief(broad_line_scores(gold_sub, lane_lines("best_f1", sub_ids)))}}
# per-case BROAD rows side by side
eng_rows = {r["case_id"]: r for r in broad_line_scores(gold_b, preds)["cases"]}
ref_rows = {r["case_id"]: r for r in broad_line_scores(gold_b, lane_lines("best_f1", broad_ids))["cases"]}
res["BROAD_per_case"] = [{"case_id": k, "family": cases[k]["family"], "ran": k in ok,
                          "engine": {x: eng_rows.get(k, {}).get(x) for x in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative", "disposition")},
                          "best_f1": {x: ref_rows.get(k, {}).get(x) for x in ("gold_count", "prediction_count", "true_positive", "false_positive", "false_negative")}}
                         for k in broad_ids]


def lm(gold_cases, p):
    s = score_line_metric(_gold_case_objects(gold_cases), p)
    e = s.get("engine") or s
    return {k: e.get(k) for k in ("f1", "precision", "recall", "true_positive", "false_positive", "false_negative",
                                  "case_count")}


g2a_h = doc["gold_cases"]
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
g2a_r, snap = relabel_gold_cases(g2a_h, only_files=[f["file"] for f in doc["relabel_files"]])
ref_bf = lane_lines("best_f1", [g["case_id"] for g in g2a_h])
sub_g = [g["case_id"] for g in g2a_h if g["case_id"] in ok]
res["G2a"] = {
    "relabel_snapshot_now": snap.get("snapshot_sha256"), "relabel_snapshot_at_freeze": doc.get("relabel_snapshot_sha256"),
    "all9": {"human": {"engine": lm(g2a_h, preds), "best_f1": lm(g2a_h, ref_bf)},
             "relabel": {"engine": lm(g2a_r, preds), "best_f1": lm(g2a_r, ref_bf)}},
    "same_scans_engine_ran": {"cases": sub_g,
                              "human": {"engine": lm([g for g in g2a_h if g["case_id"] in sub_g], preds),
                                        "best_f1": lm([g for g in g2a_h if g["case_id"] in sub_g], ref_bf)},
                              "relabel": {"engine": lm([g for g in g2a_r if g["case_id"] in sub_g], preds),
                                          "best_f1": lm([g for g in g2a_r if g["case_id"] in sub_g], ref_bf)}}}

# ---- per-bend correctness given CAD (report-only proxy) on complete-label, complete-scan BROAD cases that ran
usable, _ = prepare_gold(_gold_case_objects(gold_b))
FORMED = {"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"}
bend_rows = []
for cid in sub_ids:
    c = cases[cid]
    g = next(x for x in gold_b if x["case_id"] == cid)
    if g.get("zero_bend_control") or not g.get("label_complete", True) or c["stage_class"] != "COMPLETE" \
            or c.get("provisional_complete"):
        continue
    m = match_lines(usable[cid], preds.get(cid) or [])
    matched = {x["prediction_id"]: x for x in m["matches"]}
    ext = {b["bend_uid"]: b for b in ((e3[cid].get("extraction") or {}).get("bends") or [])}
    n_gold = len(usable[cid])
    for b in pops[cid]["bends"]:
        uid = b["bend_uid"] if isinstance(b, dict) else b
        r = ext.get(uid, {})
        ang = r.get("angle") or {}
        fl = r.get("flange") or {}
        chars = [ang, fl.get("A") or {}, fl.get("B") or {}, r.get("position") or {}]
        decisions = [ch.get("decision") for ch in chars if ch]
        measured = lambda ch: (ch or {}).get("state") == "MEASURED"
        line_ok = uid in matched
        formed = r.get("state") in FORMED
        dev_a = None if not measured(ang) or ang.get("actual") is None or ang.get("expected") is None else \
            float(ang["actual"]) - float(ang["expected"])
        dev_f = [float(x["actual"]) - float(x["expected"]) for x in (fl.get("A"), fl.get("B"))
                 if measured(x) and x.get("actual") is not None and x.get("expected") is not None]
        correct = bool(formed and line_ok and measured(ang) and measured(fl.get("A")) and measured(fl.get("B")))
        pass_any = "PASS" in decisions
        bend_rows.append({
            "case_id": cid, "family": c["family"], "bend_uid": uid, "label": r.get("label"),
            "engine_state": r.get("state"), "line_published": r.get("line_published"),
            "line_matches_gold": line_ok, "angle_state": ang.get("state"), "angle_dev_deg": dev_a,
            "flange_A_state": (fl.get("A") or {}).get("state"), "flange_B_state": (fl.get("B") or {}).get("state"),
            "flange_devs_mm": dev_f, "decisions": decisions, "correct_given_cad": correct,
            "pass_on_unconfirmed_bend": bool(pass_any and not line_ok),
            "pass_on_wrong_line": bool(pass_any and r.get("line_published") and not line_ok),
            "pass_without_published_line": bool(pass_any and not r.get("line_published")),
            "pass_decisions": decisions.count("PASS"),
            "gold_lines_in_case": n_gold, "recipe_bends_in_case": len(pops[cid]["bends"])})
n = len(bend_rows)
res["per_bend_given_cad"] = {
    "definition": "complete-label COMPLETE BROAD scans the engine ran; every recipe bend assumed formed (complete scan); "
                  "correct = engine state FORMED-like AND its apex line matches a gold line (frozen 5deg/4mm/0.6 "
                  "matcher) AND angle + both flange sides MEASURED. False PASS proxy = any PASS decision on a bend whose "
                  "line matches no gold line (no physical dimensional truth exists; report-only)",
    "bends": n, "correct": sum(r["correct_given_cad"] for r in bend_rows),
    "correct_rate": (sum(r["correct_given_cad"] for r in bend_rows) / n) if n else None,
    "formed_rate": (sum(r["engine_state"] in FORMED for r in bend_rows) / n) if n else None,
    "line_match_rate": (sum(r["line_matches_gold"] for r in bend_rows) / n) if n else None,
    "angle_measured_rate": (sum(r["angle_state"] == "MEASURED" for r in bend_rows) / n) if n else None,
    "false_pass_proxy": sum(r["pass_on_unconfirmed_bend"] for r in bend_rows),
    "false_pass_proxy_split": {"pass_on_published_line_matching_no_gold": sum(r["pass_on_wrong_line"] for r in bend_rows),
                               "pass_without_published_line": sum(r["pass_without_published_line"] for r in bend_rows),
                               "pass_decisions_on_gold_confirmed_bends": sum(r["pass_decisions"] for r in bend_rows if r["line_matches_gold"]),
                               "pass_decisions_total": sum(r["pass_decisions"] for r in bend_rows)},
    "decision_counts": {k: sum(d == k for r in bend_rows for d in r["decisions"]) for k in
                        sorted({d for r in bend_rows for d in r["decisions"]}, key=str)},
    "state_counts": {k: sum(r["engine_state"] == k for r in bend_rows) for k in
                     sorted({r["engine_state"] for r in bend_rows}, key=str)}}


def badness(r):
    s = 0.0
    if not r["line_matches_gold"]:
        s += 10
    if r["engine_state"] not in FORMED:
        s += 5
    if r["angle_dev_deg"] is not None:
        s += min(abs(r["angle_dev_deg"]), 30) / 3
    else:
        s += 3
    s += sum(min(abs(x), 30) for x in r["flange_devs_mm"]) / 3
    if r["pass_on_unconfirmed_bend"]:
        s += 8
    return s


for r in bend_rows:
    r["badness"] = round(badness(r), 3)
worst = sorted(bend_rows, key=lambda r: -r["badness"])[:20]
(OUT / "per_bend.json").write_text(json.dumps({"rows": bend_rows, "worst20": worst}, indent=1, default=str))
(OUT / "subset_scores.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: res[k] for k in ("BROAD_all26", "BROAD_same_scans_engine_ran", "per_bend_given_cad")}, indent=1,
                 default=str))
print(json.dumps(res["G2a"], indent=1, default=str))
