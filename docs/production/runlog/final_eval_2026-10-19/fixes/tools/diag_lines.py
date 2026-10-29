"""POST-HOC, CONSUMED diagnostic: per-bend engine line / nominal line vs gold lines for one DEV-CHECK case."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO / "research/broad_baseline"), str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze.gold import relabel_gold_cases
from domains.production.engine.eval.freeze.physical import _engine_line_predictions, _gold_case_objects
from domains.production.engine.eval.line_metric import prepare_gold, pair_metrics, line_endpoints
from domains.production.engine.eval.engine_output import frame_to_raw, load_engine_output
W = Path(sys.argv[1]); cid = sys.argv[2]
doc = json.loads((REPO / "docs/production/runlog/freeze/devcheck_SEALED.json").read_text())
c = {c["case_ref"]: c for c in doc["cases"]}[cid]
cdir = W / "compiled" / "dev" / c["recipe_sku"]
rec = json.loads((W / "physical/E3" / cid / "record.json").read_text())
if len(sys.argv) > 3:
    rec = dict(rec, engine_dir=sys.argv[3], status="OK")
preds = _engine_line_predictions([cid], {cid: rec}, {cid: cdir}).get(cid, [])
gold = None
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
bid = (c.get("broad") or {}).get("broad_case_id")
for g in gl["cases"]:
    if (g.get("prediction_case_id") or g["case_id"]) == bid:
        import score as sc
        gold = ("GOLD_SET_V1", sc.load_gold(g["gold"], frame_records=None)["lines"]); break
if gold is None and doc.get("relabel_gold_cases"):
    g2r, _ = relabel_gold_cases(doc["relabel_gold_cases"], only_files=[f["file"] for f in doc["relabel_files"]])
    u, _ = prepare_gold(_gold_case_objects(g2r))
    if cid in u: gold = ("RELABEL", u[cid])
print("gold src", gold[0] if gold else None, "case stage", c["stage_class"])
gls = gold[1] if gold else []
if isinstance(gls, dict): gls = gls.get("bend_lines") or gls.get("lines") or gls
print("gold keys sample", (list(gls[0].keys()) if gls else None))
eo = load_engine_output(rec["engine_dir"])
res = eo.result
def show(tag, pts):
    best = []
    for g in gls:
        try: m = pair_metrics(line_endpoints(g), pts)
        except Exception as e: continue
        best.append((m["transverse_offset_mm"], g.get("bend_id"), round(m["axis_angle_deg"],2), round(m["transverse_offset_mm"],2), round(m["gold_span_overlap"],2), round(m["prediction_span_overlap"],2)))
    best.sort()
    print(f"  {tag}: len={np.linalg.norm(pts[1]-pts[0]):.1f} nearest={best[:2]}")
for row in res.bends:
    print(row.label, row.state, [ (ch.kind, ch.state, ch.decision) for ch in row.characteristics if ch.kind in ("INCLUDED_ANGLE","BEND_LINE_POSITION")])
    if row.display.nominal_polyline_mm:
        n = np.asarray(row.display.nominal_polyline_mm); show("nominal(OML?)", frame_to_raw(res, np.asarray([n[0], n[-1]])))
    for p in preds:
        if p["bend_id"] == row.bend_uid: show("engine OUTER_APEX", np.asarray([p["line_start"], p["line_end"]]))
for g in gls:
    e = line_endpoints(g); print("gold", g.get("bend_id"), np.round(e,1).tolist())

from domains.production.engine.eval.line_metric import match_lines
m = match_lines(gls, preds)
print("POSTHOC lines: gold", len(gls), "pred", len(preds), "tp", m["true_positive"], "fp", m["false_positive"], "fn", m["false_negative"])
