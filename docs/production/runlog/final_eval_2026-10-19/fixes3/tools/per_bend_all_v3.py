"""Per-bend correctness given CAD over every A1 case the engine ran, scoring definition V3 (report-only).

V3 = V2 + the D4 amendment (scripts/production/per_bend_scoring_v3.py): a FLANGE_LENGTH NOT_APPLICABLE_AT_STAGE side
satisfies the flange criterion. V2 is recomputed alongside for the diff. Derived from fixes2/tools/per_bend_all_v2.py.
usage: per_bend_all_v3.py PHYS_DIR [OUT.json]
Same inputs, truth and gold-of-record order as the frozen V1 tool (a1/iter_7/measure/tools/per_bend_all.py, kept
runnable and untouched). The only change is the owner-approved amendment in scripts/production/per_bend_scoring_v2.py
(withheld position -> n/a, identity by the mapped CAD nominal line). V1 is recomputed alongside for the diff.
Custody: cases_A1.json only; nothing SEALED.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
import numpy as np  # noqa: E402
import score as sc  # noqa: E402
import bb_common as bb  # noqa: E402
from domains.production.engine.eval.engine_output import (decoded_stage, frame_to_raw, load_compiled,  # noqa: E402
                                                          load_engine_output, stage_locus)
from domains.production.engine.eval.freeze.gold import relabel_gold_cases  # noqa: E402
from domains.production.engine.eval.freeze.physical import _engine_line_predictions, _gold_case_objects  # noqa: E402
from domains.production.engine.eval.line_metric import match_lines, prepare_gold  # noqa: E402

_spec = importlib.util.spec_from_file_location("per_bend_scoring_v2", REPO / "scripts/production/per_bend_scoring_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
_spec3 = importlib.util.spec_from_file_location("per_bend_scoring_v3", REPO / "scripts/production/per_bend_scoring_v3.py")
v3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(v3)

P = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else P / "analysis" / "per_bend_all_v3.json"
PHYS, AN = P / "physical", P / "analysis"
FZ = REPO / "docs/production/runlog/freeze"
doc = json.loads((FZ / "cases_A1.json").read_text())
cases = {c["case_ref"]: c for c in doc["cases"]}
score = json.loads((P / "physical_scorecard.json").read_text())
truth = {(r["case_ref"], r["bend_uid"]): r["truth"] for r in score["G3"]["rows"]}
e3, cdirs = {}, {}
for cid, c in cases.items():
    cdirs[cid] = P / "compiled" / "dev" / c["recipe_sku"]
    rp = PHYS / "E3" / cid / "record.json"
    if rp.is_file():
        e3[cid] = json.loads(rp.read_text())
preds = _engine_line_predictions(list(cases), e3, cdirs)

# gold of record per case: GOLD_SET_V1 first, else the Gold11 relabel (V1 order). Published-line matches exactly as V1.
g2r_list, _ = relabel_gold_cases(doc["gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
usable_r, _ = prepare_gold(_gold_case_objects(g2r_list))
gold_of = {}  # cid -> (src, gold_lines_raw, published_match_ids, matched_gold_ids, match_fn)
for cid, gold in usable_r.items():
    m = match_lines(gold, preds.get(cid) or [])
    gold_of[cid] = ("RELABEL", gold, {x["prediction_id"] for x in m["matches"]}, {x["gold_id"] for x in m["matches"]},
                    match_lines)
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
a1_by_bid = {(c.get("broad") or {}).get("broad_case_id"): c for c in doc["cases"] if (c.get("broad") or {}).get("broad_case_id")}
thr, sc_match = sc._matcher()
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
    gold_lines = sc.load_gold(g["gold"], frame_records=None)["lines"]
    matched_gold = {str(x["gold_id"]) for x in r["matches"]}
    gold_of[cid] = ("GOLD_SET_V1", gold_lines, ids, matched_gold, lambda a, b: sc_match(a, b, thr))


def nominal_raw(res_obj, compiled, uid):
    """CAD nominal outer-apex line of the decoded stage, mapped to the raw scan frame by the E3 registration."""
    try:
        return frame_to_raw(res_obj, stage_locus(compiled, decoded_stage(res_obj, compiled), uid, "outer_apex"))
    except Exception:  # noqa: BLE001 - no registration transform / no locus -> identity not verifiable
        return None


rows_v1, rows_v2, withheld_log = [], [], []
for cid in sorted(e3):
    if e3[cid].get("status") != "OK":
        continue
    res = json.loads((PHYS / "E3" / cid / "engine" / "result.json").read_text())
    published = {x["bend_id"] for x in (preds.get(cid) or [])}
    src = gold_of.get(cid)

    def truth_of(uid, cid=cid):
        t = truth.get((cid, uid))
        if t is None and cases[cid]["stage_class"] != "COMPLETE":
            return None
        return t or "FORMED"

    withheld = [b["bend_uid"] for b in res["bends"] if b["bend_uid"] not in published and v2.position_withheld(b)]
    ident = {}
    if src is not None and withheld:
        eo = load_engine_output(e3[cid]["engine_dir"])
        comp = load_compiled(cdirs[cid] / "compiled_part.json")
        nom = {u: nominal_raw(eo.result, comp, u) for u in withheld}
        ident = v2.identity_by_nominal(withheld, nom, src[1], src[3], src[4])
        for u in withheld:
            withheld_log.append({"case": cid, "bend_uid": u, **ident[u]})

    def status_of(uid, src=src, published=published, ident=ident):
        return v2.line_status(uid, criterion_applied=src is not None, published=uid in published,
                              published_match_ids=src[2] if src else (), withheld=uid in ident,
                              identity_ok=(ident.get(uid) or {}).get("identity_ok"))

    fam = cases[cid]["family"]
    rows_v1 += v2.per_bend_rows(cid, fam, res, truth_of, status_of, src[0] if src else None)  # V2 (named v1 below)
    rows_v2 += v3.per_bend_rows(cid, fam, res, truth_of, status_of, src[0] if src else None)  # V3


def summ(rows):
    n = len(rows)
    fam = defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["family"]][0] += 1
        fam[r["family"]][1] += int(r["correct_riser_aware"])
    return {"bends": n, "correct_riser_aware": sum(r["correct_riser_aware"] for r in rows),
            "correct_strict": sum(r["correct_strict"] for r in rows),
            "rate_riser_aware": sum(r["correct_riser_aware"] for r in rows) / n if n else None,
            "family_macro_riser_aware": sum(c / t for t, c in fam.values()) / len(fam) if fam else None,
            "false_pass_proxy": sum(r["false_pass_proxy"] for r in rows),
            "false_pass_proxy_bends": [f"{r['case']}:{r['label']}" for r in rows if r["false_pass_proxy"]],
            "line_status_counts": dict(sorted(__import__("collections").Counter(r["line_status"] for r in rows).items()))}


out = {"definition": v3.DEFINITION, "v2": summ(rows_v1), "v3": summ(rows_v2), "withheld_identity": withheld_log,
       "changed_rows": [{"case": a["case"], "label": a["label"], "v2": [a["line_status"], a["correct_riser_aware"], a["false_pass_proxy"]],
                         "v3": [b["line_status"], b["correct_riser_aware"], b["false_pass_proxy"]]}
                        for a, b in zip(rows_v1, rows_v2) if (a["correct_riser_aware"], a["false_pass_proxy"], a["line_status"])
                        != (b["correct_riser_aware"], b["false_pass_proxy"], b["line_status"])],
       "rows_v3": rows_v2}
OUT.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("v2", "v3", "withheld_identity", "changed_rows")}, indent=1, default=str))
