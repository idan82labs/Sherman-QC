"""DIAGNOSTIC - consumed sets, not a held-out score. Never a RUN; never a new evaluation.
Per-bend scoring V1/V2/V3 (scripts/production/per_bend_scoring_v2.py, _v3.py) and line-F1 (frozen matcher 5 deg / 4 mm /
0.6, Hungarian, OUTER_APEX, raw frame) on the consumed FRESH and DEV-CHECK engine-ran cases, for one engine-output arm.
Per-bend population: complete scans only, every bend expected FORMED (a1/iter_7 convention; staged scans have no
authorized truth table). usage: consumed_diag.py CASES.json ARM(before|after) AFTER_ROOT OUT.json"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import (decoded_stage, frame_to_raw, load_compiled,  # noqa: E402
                                                          load_engine_output, stage_locus)
from domains.production.engine.eval.line_metric import engine_lines, match_lines  # noqa: E402


def _mod(name):
    sp = importlib.util.spec_from_file_location(name, REPO / f"scripts/production/{name}.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


v2, v3 = _mod("per_bend_scoring_v2"), _mod("per_bend_scoring_v3")
cases = json.loads(Path(sys.argv[1]).read_text()); arm = sys.argv[2]; after_root = Path(sys.argv[3])
rows = {"V1": [], "V2": [], "V3": []}
lines = []
per_case = []
for c in cases:
    ed = Path(c["before"]) if arm == "before" else after_root / c["case"] / "engine"
    res = json.loads((ed / "result.json").read_text())
    state = res["inspection_validity"]["state"]
    gold = c["gold"]
    rec = {"case": c["case"], "set": c["set"], "family": c["family"], "validity": state,
           "engine": res["engine"]["engine_version"], "complete": c["complete"], "gold": len(gold)}
    if state != "VALID":
        lines.append((c["set"], c["family"], 0, 0, len(gold), False))
        rec.update(tp=0, fp=0, fn=len(gold))
        if c["complete"]:
            rec["bends_missing"] = len(res["bends"])
        per_case.append(rec); continue
    eo = load_engine_output(str(ed)); comp = load_compiled(Path(c["compiled"]) / "compiled_part.json")
    preds = engine_lines(eo.result, comp, locus="OUTER_APEX", frame="raw")
    if gold:
        m = match_lines(gold, preds)
        tp, fp, fn = m["true_positive"], m["false_positive"], m["false_negative"]
        pub_ids, gold_used = {x["prediction_id"] for x in m["matches"]}, {x["gold_id"] for x in m["matches"]}
        lines.append((c["set"], c["family"], tp, fp, fn, True))
        rec.update(tp=tp, fp=fp, fn=fn)
    else:
        pub_ids, gold_used = set(), set()
        rec.update(tp=None, fp=None, fn=None, note="no line gold for this case: line criterion not applied")
    if c["complete"]:
        published = {p["bend_id"] for p in preds}
        withheld = [b["bend_uid"] for b in res["bends"] if b["bend_uid"] not in published and v2.position_withheld(b)]
        ident = {}
        if gold and withheld:
            stage = decoded_stage(eo.result, comp)
            nom = {}
            for u in withheld:
                try:
                    nom[u] = frame_to_raw(eo.result, stage_locus(comp, stage, u, "outer_apex"))
                except Exception:  # noqa: BLE001
                    nom[u] = None
            ident = v2.identity_by_nominal(withheld, nom, gold, gold_used, match_lines)
        applied = bool(gold)
        st2 = lambda u: v2.line_status(u, criterion_applied=applied, published=u in published, published_match_ids=pub_ids,  # noqa: E731
                                       withheld=u in ident, identity_ok=(ident.get(u) or {}).get("identity_ok"))
        st1 = lambda u: (v2.NOT_APPLIED if not applied else (v2.MATCH if u in pub_ids else v2.MISS))  # noqa: E731
        fam = c["family"]
        r1 = v2.per_bend_rows(c["case"], fam, res, lambda u: "FORMED", st1, "LABEL")
        r2 = v2.per_bend_rows(c["case"], fam, res, lambda u: "FORMED", st2, "LABEL")
        r3 = v3.per_bend_rows(c["case"], fam, res, lambda u: "FORMED", st2, "LABEL")
        for k, r in (("V1", r1), ("V2", r2), ("V3", r3)):
            for x in r:
                x["set"] = c["set"]
            rows[k] += r
        rec.update({k: {"correct": sum(x["correct_riser_aware"] for x in r), "bends": len(r),
                        "false_pass": [x["label"] for x in r if x["false_pass_proxy"]]} for k, r in (("V1", r1), ("V2", r2), ("V3", r3))})
    per_case.append(rec)


def line_summary(sel):
    tp = sum(x[2] for x in sel); fp = sum(x[3] for x in sel); fn = sum(x[4] for x in sel)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    fam = defaultdict(lambda: [0, 0, 0])
    for x in sel:
        for i in range(3):
            fam[x[1]][i] += x[2 + i]
    fm = [2 * a / (2 * a + b + d) if (2 * a + b + d) else 1.0 for a, b, d in fam.values()]
    return {"cases": len(sel), "families": len(fam), "tp": tp, "fp": fp, "fn": fn, "micro_f1": f1,
            "family_macro_f1": sum(fm) / len(fm) if fm else None}


out = {"label": "DIAGNOSTIC - consumed sets, not a held-out score", "arm": arm, "per_case": per_case, "sets": {}}
for s in ("FRESH", "DEV-CHECK"):
    sel = [x for x in lines if x[0] == s]
    missing = sum(p.get("bends_missing", 0) for p in per_case if p["set"] == s)
    out["sets"][s] = {"lines_all_cases": line_summary(sel), "lines_engine_ran": line_summary([x for x in sel if x[5]]),
                      "per_bend_engine_ran": {k: {"bends": len([r for r in rows[k] if r["set"] == s]),
                                                  "correct_riser_aware": sum(r["correct_riser_aware"] for r in rows[k] if r["set"] == s),
                                                  "correct_strict": sum(r["correct_strict"] for r in rows[k] if r["set"] == s),
                                                  "false_pass_proxy": [f"{r['case']}:{r['label']}" for r in rows[k] if r["set"] == s and r["false_pass_proxy"]]}
                                              for k in rows},
                      "complete_case_bends_refused": missing}
Path(sys.argv[4]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["sets"], indent=1, default=str))
