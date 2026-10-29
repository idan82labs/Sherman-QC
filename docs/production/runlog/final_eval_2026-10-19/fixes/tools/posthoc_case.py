"""POST-HOC, CONSUMED diagnostic (not a new result): engine lines (frozen matcher, OUTER_APEX, raw) vs one scan's
gold label, plus validity and bend states. usage: posthoc_case.py ENGINE_DIR COMPILED_DIR GOLD_LABEL_JSON"""
import json, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.engine_output import load_compiled, load_engine_output
from domains.production.engine.eval.line_metric import engine_lines, match_lines
ed, cdir, lab = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
eo = load_engine_output(str(ed)); res = eo.result
gold = [{"bend_id": b.get("bend_id") or f"B{i:02d}", "line_start": b["line_start"], "line_end": b["line_end"]}
        for i, b in enumerate(lab.get("bends") or [], 1)]
preds = engine_lines(res, load_compiled(cdir / "compiled_part.json"), locus="OUTER_APEX", frame="raw")
m = match_lines(gold, preds)
dec = {}
for b in res.bends:
    for c in b.characteristics:
        dec[c.decision] = dec.get(c.decision, 0) + 1
matched = {x["prediction_id"] for x in m["matches"]}
fp_proxy = [b.label for b in res.bends if any(c.decision == "PASS" for c in b.characteristics) and b.bend_uid not in matched]
ang = [(b.label, next((round(c.actual, 2) for c in b.characteristics if c.kind == "INCLUDED_ANGLE" and c.actual is not None), None)) for b in res.bends]
print(json.dumps({"pass_without_matched_line": fp_proxy, "angles": ang, "validity": res.inspection_validity.state, "reasons": list(res.inspection_validity.reasons),
                  "states": [(b.label, b.state) for b in res.bends], "gold": len(gold), "pred": len(preds),
                  "tp": m["true_positive"], "fp": m["false_positive"], "fn": m["false_negative"], "decisions": dec}))
