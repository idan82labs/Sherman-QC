"""POST-HOC diagnostic, scoring definition V2 vs V1 on one complete scan (every bend expected FORMED), against an
ai_studio label in the raw scan frame. Never a held-out score. Refuses any path containing SEALED.
usage: posthoc_v2.py CASE_ID ENGINE_DIR COMPILED_DIR LABEL_JSON"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import (decoded_stage, frame_to_raw, load_compiled,  # noqa: E402
                                                          load_engine_output, stage_locus)
from domains.production.engine.eval.line_metric import engine_lines, match_lines  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2", REPO / "scripts/production/per_bend_scoring_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
cid, ed, cdir, lab_p = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
if any("SEALED" in str(p).upper() for p in (ed, cdir, lab_p)):
    sys.exit("refused: SEALED path")
lab = json.loads(lab_p.read_text())
assert lab.get("coordinate_frame") == "source_ply_raw"
gold = [{"bend_id": b.get("bend_id") or f"B{i:02d}", "line_start": b["line_start"], "line_end": b["line_end"]}
        for i, b in enumerate(lab.get("bends") or [], 1)]
eo = load_engine_output(str(ed))
comp = load_compiled(cdir / "compiled_part.json")
preds = engine_lines(eo.result, comp, locus="OUTER_APEX", frame="raw")
m = match_lines(gold, preds)
pub_ids, gold_used = {x["prediction_id"] for x in m["matches"]}, {x["gold_id"] for x in m["matches"]}
published = {p["bend_id"] for p in preds}
res = json.loads((ed / "result.json").read_text())
withheld = [b["bend_uid"] for b in res["bends"] if b["bend_uid"] not in published and v2.position_withheld(b)]
stage = decoded_stage(eo.result, comp)
nom = {u: frame_to_raw(eo.result, stage_locus(comp, stage, u, "outer_apex")) for u in withheld}
ident = v2.identity_by_nominal(withheld, nom, gold, gold_used, match_lines)
st2 = lambda u: v2.line_status(u, criterion_applied=True, published=u in published, published_match_ids=pub_ids,  # noqa: E731
                               withheld=u in ident, identity_ok=(ident.get(u) or {}).get("identity_ok"))
st1 = lambda u: v2.MATCH if u in pub_ids else v2.MISS  # noqa: E731
r1 = v2.per_bend_rows(cid, "-", res, lambda u: "FORMED", st1, "LABEL")
r2 = v2.per_bend_rows(cid, "-", res, lambda u: "FORMED", st2, "LABEL")
lab_of = {b["bend_uid"]: b["label"] for b in res["bends"]}
print(json.dumps({"case": cid, "lines_tp_fp_fn": [m["true_positive"], m["false_positive"], m["false_negative"]],
                  "identity": {lab_of[u]: v for u, v in ident.items()},
                  "v1": {"correct": sum(r["correct_riser_aware"] for r in r1), "bends": len(r1),
                         "false_pass": [r["label"] for r in r1 if r["false_pass_proxy"]]},
                  "v2": {"correct": sum(r["correct_riser_aware"] for r in r2), "bends": len(r2),
                         "false_pass": [r["label"] for r in r2 if r["false_pass_proxy"]]}}, default=str))
