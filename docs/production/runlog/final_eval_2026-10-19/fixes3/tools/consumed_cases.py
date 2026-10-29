"""DIAGNOSTIC - consumed sets, not a held-out score. Builds the case table for the consumed FRESH and DEV-CHECK
engine-ran cases (owner authorization 2026-10-19, ledger seq 8: read-only labels of ai_bend_lines_fresh_v1_SEALED/ and
devcheck_SEALED.json; nothing else sealed is opened). Engine inputs come only from non-sealed locations: FRESH compiled
parts and recipes of the one-time evaluation work dir (final_eval_w/fresh), or the round-1 E1 rebuild (fixes/compile_after1);
DEV-CHECK compiled parts and recipes from the round-1 rebuild of the same STEP (nominals and tolerances checked equal).
Line gold: FRESH = its sealed ai_studio label; DEV-CHECK = the gold of record (GOLD_SET_V1), else the BROAD gold.
usage: consumed_cases.py OUT.json"""
import json, sys
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
SP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze.common import dev_sku_for  # noqa: E402
W = SP / "final_eval_w"; A1 = SP / "fixes/compile_after1"; BEFORE = SP / "fixes/posthoc"
out = []
man = json.loads((W / "fresh" / "fresh_cases.json").read_text())
STAGED_FRESH = {"fr_c0ce5ecd2ba3"}  # not in the one-time eval's per-bend population (results.json); 0-bend control has no recipe
for p in man["parts"]:
    sku = dev_sku_for(p["step_sha256"])
    for s in p["scans"]:
        cid = "fr_" + s["scan_id"][3:]
        if not (BEFORE / cid / "engine" / "result.json").exists():
            continue
        sha = json.loads((BEFORE / cid / "engine" / "result.json").read_text())["engine"]["compiled_part_sha256"]
        fresh = W / "fresh" / "compiled" / "dev" / sku
        if (fresh / "compiled_part.json").exists() and json.loads((fresh / "compiled_part.json").read_text())["compiled_part_sha256"] == sha:
            cdir, rec = fresh, W / "fresh" / "recipes" / sku / "dev.recipe.json"
        else:
            cdir, rec = A1 / sku / "compiled", A1 / sku / "dev.recipe.json"
        lab = json.loads((REPO / s["studio_label"]).read_text())
        assert lab.get("coordinate_frame") == "source_ply_raw", cid
        gold = [{"bend_id": b.get("bend_id") or f"B{i:02d}", "line_start": [float(x) for x in b["line_start"]],
                 "line_end": [float(x) for x in b["line_end"]]} for i, b in enumerate(lab.get("bends") or [], 1)]
        out.append({"case": cid, "set": "FRESH", "family": p["part_id"], "complete": cid not in STAGED_FRESH,
                    "scan": s["path"], "compiled": str(cdir), "recipe": str(rec), "before": str(BEFORE / cid / "engine"), "gold": gold})
dc = json.loads((REPO / "docs/production/runlog/freeze/devcheck_SEALED.json").read_text())
gold_dc = {g["case_id"]: g for g in dc["broad_gold_cases"]}
# gold of record (owner direction 2026-10-19): GOLD_SET_V1 where the scan is in it (ai_bend_lines_gold_v1 / the Gold11
# relabel, non-sealed consumed labels), else the frozen BROAD gold carried in devcheck_SEALED.json
gl = json.loads((REPO / "docs/production/runlog/gold_set_v1/case_list_gold_v1.json").read_text())
gs_by_id = {}
for g in gl["cases"]:
    for k in {g.get("prediction_case_id"), g["case_id"]} - {None}:
        gs_by_id.setdefault(k, g)
for c in dc["cases"]:
    cid = c["case_ref"]
    if not (BEFORE / cid / "engine" / "result.json").exists():
        continue
    bid = (c.get("broad") or {}).get("broad_case_id")
    gs = gs_by_id.get(bid) or gs_by_id.get("sc_" + cid[3:])
    if gs is not None and isinstance(gs.get("gold"), dict) and gs["gold"].get("path"):
        lab = json.loads((REPO / gs["gold"]["path"]).read_text())
        assert lab.get("coordinate_frame") == "source_ply_raw", cid
        gold = [{"bend_id": b.get("bend_id") or f"B{i:02d}", "line_start": [float(x) for x in b["line_start"]],
                 "line_end": [float(x) for x in b["line_end"]]} for i, b in enumerate(lab.get("bends") or [], 1)]
        out.append({"case": cid, "set": "DEV-CHECK", "family": c["family"], "complete": c["stage_class"] == "COMPLETE",
                    "scan": c["scan_path"], "compiled": str(A1 / c["recipe_sku"] / "compiled"),
                    "recipe": str(A1 / c["recipe_sku"] / "dev.recipe.json"), "before": str(BEFORE / cid / "engine"),
                    "gold": gold, "gold_source": "GOLD_SET_V1 " + gs["gold"]["path"]})
        continue
    g = gold_dc.get(cid)
    M = None
    if g is not None and g["frame"] != "raw_scan":  # reference frame: back to raw through the recorded transform
        import numpy as np
        M = np.linalg.inv(np.asarray(g["raw_to_reference"], dtype=float).reshape(4, 4))
    def raw(v):
        if M is None:
            return [float(x) for x in v]
        return [float(x) for x in (M[:3, :3] @ v + M[:3, 3])]
    gold = [{"bend_id": b["bend_id"], "line_start": raw(b["line_start"]), "line_end": raw(b["line_end"])}
            for b in (g or {}).get("gold_lines", [])]
    out.append({"case": cid, "set": "DEV-CHECK", "family": c["family"], "complete": c["stage_class"] == "COMPLETE",
                "scan": c["scan_path"], "compiled": str(A1 / c["recipe_sku"] / "compiled"),
                "recipe": str(A1 / c["recipe_sku"] / "dev.recipe.json"), "before": str(BEFORE / cid / "engine"),
                "gold": gold, "gold_source": "devcheck_SEALED.json broad_gold_cases" if g else None})
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(len(out), [(o["case"], o["set"], o["complete"], len(o["gold"])) for o in out])
