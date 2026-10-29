"""DEV-CHECK final analysis (report-only; frozen eval code read-only). usage: devcheck_analyse.py W OUT.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(Path(__file__).parent), str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import pb_common as pb  # noqa: E402
import score as sc  # noqa: E402
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
from domains.production.engine.eval.freeze.physical import (_engine_line_predictions, _gold_case_objects,  # noqa: E402
                                                            planned_population)
from domains.production.engine.eval.line_metric import match_lines, prepare_gold  # noqa: E402

W = Path(sys.argv[1]) / "devcheck_SEALED"
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "devcheck_SEALED.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
card = json.loads((W / "physical_scorecard.json").read_text())
truth = {(r["case_ref"], r["bend_uid"]): r["truth"] for r in (card["G3"].get("rows") or [])}
e3, cdirs = {}, {}
for cid, c in cases.items():
    cdirs[cid] = W / "compiled" / "dev" / c["recipe_sku"]
    rp = W / "physical" / "E3" / cid / "record.json"
    if rp.is_file():
        e3[cid] = json.loads(rp.read_text())
preds = _engine_line_predictions(list(cases), e3, cdirs)


def refusal(cid):
    r = e3.get(cid)
    if r is None:
        return "NOT_RUN_AMENDMENT7_GATE"
    if r.get("status") != "OK":
        return f"E3_{r.get('status')}"
    res = json.loads((W / "physical" / "E3" / cid / "engine" / "result.json").read_text())
    iv = res.get("inspection_validity") or {}
    if (iv.get("registration") or {}).get("abstained") or iv.get("state") not in (None, "VALID", "OK", "QUALIFIED"):
        return f"ENGINE_REFUSED:{iv.get('state')}:{','.join(iv.get('reasons') or [])}"
    return None


status = {cid: refusal(cid) for cid in cases}
# ---- line-gold matching for the per-bend line criterion (iter-7 order: GOLD_SET_V1, then relabel)
g2r_list, _ = relabel_gold_cases(doc["relabel_gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]]) \
    if doc.get("relabel_gold_cases") else ([], None)
usable_r, _ = prepare_gold(_gold_case_objects(g2r_list)) if g2r_list else ({}, None)
rel_match = {cid: {x["prediction_id"] for x in match_lines(g, preds.get(cid) or [])["matches"]} for cid, g in usable_r.items()}
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
by_bid = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"] if (c.get("broad") or {}).get("broad_case_id")}
gs_match = {}
for g in gl["cases"]:
    pid = g.get("prediction_case_id") or g["case_id"]
    if pid not in by_bid or not g.get("cad_path"):
        continue
    cid = by_bid[pid]["case_ref"]
    if cid not in preds:
        continue
    obr = [{"bend_id": r["bend_id"], "line_start": r["line_start"], "line_end": r["line_end"]} for r in preds[cid]]
    pred = {"schema": "sherman.broad_baseline.eval_lines.v1", "lane": "e3", "status": "OK", "frame": "scan_raw",
            "scan": {} if pid != g["case_id"] else {"sha256": by_bid[pid]["scan_sha256"]}, "owned_bend_regions": obr}
    r = sc.score_case({"case_id": g["case_id"], "tier": g.get("tier"), "family_id": g["family_id"], "gold": g["gold"],
                       "label_complete": g.get("label_complete", True)}, frame_records=None, preloaded_predictions=pred)
    if r.get("matches") is None:
        continue
    ids = set()
    for mm in r["matches"]:
        p = mm.get("prediction_id")
        ids.add(obr[p]["bend_id"] if isinstance(p, int) and 0 <= p < len(obr) else str(p))
    gs_match[cid] = ids

rows, missing = [], []
for cid, c in sorted(cases.items()):
    complete = c["stage_class"] == "COMPLETE" and not c.get("provisional_complete")
    if status[cid] is not None:
        if complete:  # expected FORMED for every planned bend: a miss for each
            n = len((planned_population(REPO / doc["recipes_dir"] / f"{c['recipe_sku']}.recipe.json", cdirs[cid]) or {}).get("bends") or [])
            missing.append({"case": cid, "family": c["family"], "bends": n, "why": status[cid]})
        continue
    res = json.loads((W / "physical" / "E3" / cid / "engine" / "result.json").read_text())
    if cid in gs_match:
        lm, src = gs_match[cid], "GOLD_SET_V1"
    elif cid in rel_match:
        lm, src = rel_match[cid], "RELABEL"
    else:
        lm, src = None, None
    tf = lambda uid, cid=cid, c=c: truth.get((cid, uid)) or ("FORMED" if c["stage_class"] == "COMPLETE" else None)
    rows += pb.per_bend_rows(cid, c["family"], res, tf, lm, src)
pbs = pb.summarize(rows, missing)
pbs["bootstrap_headline"] = pb.bootstrap_bends(pbs["_fam_counts"])
er = {}
for r in rows:
    er.setdefault(r["family"], [0, 0, 0]); er[r["family"]][0] += 1; er[r["family"]][1] += int(r["correct_riser_aware"])
pbs["bootstrap_engine_ran"] = pb.bootstrap_bends(er) if er else None
pbs.pop("_fam_counts")

# ---- line-F1 (frozen matcher, from the frozen scorer's case rows) + family bootstrap
lmet = card["BROAD"]["line_metric"]
arms = {"engine": lmet["engine"], "best_f1": lmet["reference_lanes"]["best_f1"], "runtime": lmet["reference_lanes"]["runtime"]}
ran_ids = {cid for cid in cases if status[cid] is None}
line = {}
for arm, blk in arms.items():
    cr = [{"case": r["case_id"], "family": r["family"], "tp": r["true_positive"], "fp": r["false_positive"],
           "fn": r["false_negative"]} for r in blk["cases"] if r["disposition"] == "SCORED_COMPLETE_LABEL"]
    sub = [r for r in cr if r["case"] in ran_ids]
    line[arm] = {"all_scans": {**pb.line_stats(cr), "bootstrap": pb.bootstrap_lines(cr)},
                 "engine_ran_scans": {**pb.line_stats(sub), "bootstrap": pb.bootstrap_lines(sub) if sub else None},
                 "per_case_f1": {r["case"]: round(2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"]), 4)
                                 if (2 * r["tp"] + r["fp"] + r["fn"]) else None for r in cr}}
out = {"cases": len(cases), "status": status,
       "compile": {"frozen_recipes_devcheck": 7, "compiled": 7, "matches_freeze": True,
                   "case_compile_status": {cid: c["compile_status"] for cid, c in cases.items()}},
       "line_f1": line, "per_bend": pbs,
       "scorecard_verdict": {"verdict": card["verdict"], "final": card["final"], "qualifying": card["qualifying"],
                             "G2": {k: v for k, v in card["G2"]["gates"].items()}, "G3": card["G3"].get("verdict"),
                             "truth_table": card["truth_table"].get("status"), "runtime": card["BROAD"].get("runtime")}}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"status": status, "line": {a: {k: {kk: v[k][kk] for kk in ("cases", "micro_f1", "family_macro_f1", "tp", "fp", "fn")}
                                                    for k in ("all_scans", "engine_ran_scans") if v[k]["cases"]} for a, v in line.items()},
                  "per_bend": {k: pbs[k] for k in ("engine_ran", "headline_with_misses", "false_pass_proxy", "false_pass_bends", "bootstrap_headline", "bootstrap_engine_ran")}},
                 indent=1, default=str))
