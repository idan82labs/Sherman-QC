"""FRESH final analysis (report-only; frozen eval scorer read-only). usage: fresh_analyse.py F OUT.json
Line metric: frozen BROAD scorer broad_line_scores (5 deg / 4 mm / 0.6, raw frame, engine OUTER_APEX locus),
gold = sealed FRESH ai_studio labels (sha-bound to the scan via provenance), built exactly like freeze broad.broad_gold.
Per-bend: a1/iter_7 per_bend_all definition; expected FORMED on complete scans; staged scans skipped (no frozen
truth table); line criterion = the FRESH gold. Zero-bend control: every engine bend expected NOT_FORMED."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(Path(__file__).parent), str(REPO), str(REPO / "backend")]
import pb_common as pb  # noqa: E402
from domains.production.engine.eval.freeze.physical import (_engine_line_predictions, _gold_case_objects,  # noqa: E402
                                                            broad_line_scores)
from domains.production.engine.eval.line_metric import match_lines, prepare_gold  # noqa: E402

F = Path(sys.argv[1])
man = json.loads((F / "fresh_cases.json").read_text())
bmk = json.loads((REPO / "docs/production/runlog/benchmark_v3/BENCHMARK_V3.json").read_text())
sealed = {s["scan_id"]: s for s in json.loads((REPO / "docs/production/runlog/benchmark_v3/fresh_SEALED/BENCHMARK_V3_FRESH_SEALED.json").read_text())["scans"]}
from domains.production.engine.eval.freeze.common import dev_sku_for, sha256_file  # noqa: E402

gold, cases, e3, cdirs, comp = [], {}, {}, {}, {}
for p in man["parts"]:
    sku = dev_sku_for(p["step_sha256"])
    b = json.loads((F / "recipes" / sku / "build.json").read_text()) if (F / "recipes" / sku / "build.json").is_file() else {}
    c = json.loads((F / "recipes" / sku / "compile.json").read_text()) if (F / "recipes" / sku / "compile.json").is_file() else {}
    comp[p["part_id"]] = {"sku": sku, "recipe": b.get("status", "BUILD_FAILED"), "recipe_reason": b.get("reason"),
                          "recipe_bends": b.get("bends"), "compile": c.get("status") or (b.get("status") if b.get("status") != "RECIPE_BUILT" else "NOT_COMPILED"),
                          "compile_reason": c.get("reason"), "compiled_bends": c.get("bends")}
    for s in p["scans"]:
        cid = "fr_" + s["scan_id"][3:]
        lab = json.loads((REPO / s["studio_label"]).read_text())
        prov_p = (REPO / s["studio_label"]).parent / "provenance" / Path(s["studio_label"]).name
        scan_sha = sha256_file(s["path"])
        assert json.loads(prov_p.read_text()).get("asset_sha256") == scan_sha, cid
        assert lab.get("coordinate_frame") == "source_ply_raw", cid
        lines = [{"bend_id": bb.get("bend_id") or f"B{i:02d}", "line_start": [float(x) for x in bb["line_start"]],
                  "line_end": [float(x) for x in bb["line_end"]]} for i, bb in enumerate(lab.get("bends") or [], 1)]
        zero = bool(lab.get("no_bends_present")) and not lines
        se = sealed[s["scan_id"]]
        gold.append({"case_id": cid, "broad_case_id": s["scan_id"], "family": p["part_id"], "tier": "FRESH_AI_GOLD_ACCEPTED_SEALED",
                     "gold_kind": "ai_studio", "label_complete": bool(se.get("label_complete", True)), "flags": [],
                     "scan_sha256": scan_sha, "frame": "raw_scan", "gold_lines": lines, "raw_to_reference": None,
                     "residual_mm": 0.0, "transform_method": "SOURCE_PLY_RAW_IS_SCAN_RAW", "zero_bend_control": zero})
        cases[cid] = {"family": p["part_id"], "sku": sku, "scan_id": s["scan_id"], "staged": bool(se.get("staged_or_incomplete")),
                      "zero": zero, "gold_n": len(lines)}
        cdirs[cid] = F / "compiled" / "dev" / sku
        rp = F / "physical" / "E3" / cid / "record.json"
        if rp.is_file():
            e3[cid] = json.loads(rp.read_text())
preds = _engine_line_predictions(list(cases), e3, cdirs)


def status_of(cid):
    c = cases[cid]
    cs = comp[c["family"]]
    if cs["compile"] != "COMPILED":
        return f"COMPILE_{cs['compile']}:{cs.get('compile_reason') or cs.get('recipe_reason')}"
    r = e3.get(cid)
    if r is None:
        return "E3_NOT_RUN"
    if r["status"] != "OK":
        return f"E3_{r['status']}:{(r.get('detail') or '')[:120]}"
    res = json.loads((Path(r["engine_dir"]) / "result.json").read_text())
    iv = res.get("inspection_validity") or {}
    if (iv.get("registration") or {}).get("abstained") or iv.get("state") not in (None, "VALID", "OK", "QUALIFIED"):
        return f"ENGINE_REFUSED:{iv.get('state')}:{','.join(iv.get('reasons') or [])}"
    return None


status = {cid: status_of(cid) for cid in cases}
# ---- best_f1 lane predictions (same conversion as freeze broad.baseline_predictions)
lane, lane_status = {}, {}
for cid, c in cases.items():
    pth = F / "bb_runs" / "best_f1" / c["scan_id"] / "predictions_best_f1.json"
    rr = F / "bb_runs" / "best_f1" / c["scan_id"] / "run_record.json"
    lane_status[cid] = json.loads(rr.read_text()).get("status") if rr.is_file() else "NOT_RUN"
    if not pth.is_file():
        continue
    d = json.loads(pth.read_text())
    if (d.get("scan") or {}).get("sha256") not in (None, gold[[g["case_id"] for g in gold].index(cid)]["scan_sha256"]):
        continue
    ls = []
    for i, row in enumerate(d.get("owned_bend_regions") or [], 1):
        ends = row.get("span_endpoints")
        if ends:
            a, b = ends[0], ends[1]
        elif row.get("line_start") is not None and row.get("line_end") is not None:
            a, b = row["line_start"], row["line_end"]
        else:
            continue
        ls.append({"bend_id": row.get("bend_id") or f"P{i:02d}", "line_start": [float(x) for x in a], "line_end": [float(x) for x in b]})
    lane[cid] = ls


def arm_block(arm_preds):
    s = broad_line_scores(gold, arm_preds)
    cr = [{"case": r["case_id"], "family": r["family"], "tp": r["true_positive"], "fp": r["false_positive"], "fn": r["false_negative"]}
          for r in s["cases"] if r["disposition"] == "SCORED_COMPLETE_LABEL"]
    ran = {cid for cid in cases if status[cid] is None}
    sub = [r for r in cr if r["case"] in ran]
    return {"frozen_scorer": {k: s.get(k) for k in ("line_cases", "families", "true_positive", "false_positive", "false_negative",
                                                     "micro", "family_macro_f1", "family_macro_f1_with_controls", "zero_bend_controls",
                                                     "exact_part_rate", "excluded_before_scoring")},
            "all_scans": {**pb.line_stats(cr), "bootstrap": pb.bootstrap_lines(cr)},
            "engine_ran_scans": {**pb.line_stats(sub), "bootstrap": pb.bootstrap_lines(sub) if sub else None},
            "per_case_f1": {r["case"]: round(2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"]), 4) if (2 * r["tp"] + r["fp"] + r["fn"]) else None for r in cr},
            "_rows": cr}


line = {"engine": arm_block(preds), "best_f1": arm_block(lane)}
# ---- per-bend
usable, _ = prepare_gold(_gold_case_objects(gold))
rows, missing = [], []
for cid, c in sorted(cases.items()):
    if c["staged"]:
        continue
    if status[cid] is not None:
        if c["zero"]:
            continue
        n = comp[c["family"]].get("compiled_bends") or comp[c["family"]].get("recipe_bends") or c["gold_n"]
        missing.append({"case": cid, "family": c["family"], "bends": n, "why": status[cid],
                        "bends_source": "compiled" if comp[c["family"]].get("compiled_bends") else ("recipe" if comp[c["family"]].get("recipe_bends") else "gold_lines")})
        continue
    res = json.loads((Path(e3[cid]["engine_dir"]) / "result.json").read_text())
    lm = {x["prediction_id"] for x in match_lines(usable[cid], preds.get(cid) or [])["matches"]} if cid in usable and not c["zero"] else None
    tf = (lambda uid: "NOT_FORMED") if c["zero"] else (lambda uid: "FORMED")
    rows += pb.per_bend_rows(cid, c["family"], res, tf, lm, "FRESH_SEALED")
pbs = pb.summarize(rows, missing)
pbs["bootstrap_headline"] = pb.bootstrap_bends(pbs.pop("_fam_counts"))
er = {}
for r in rows:
    er.setdefault(r["family"], [0, 0, 0]); er[r["family"]][0] += 1; er[r["family"]][1] += int(r["correct_riser_aware"])
pbs["bootstrap_engine_ran"] = pb.bootstrap_bends(er) if er else None
n_parts = len(comp)
out = {"parts": n_parts, "scans": len(cases), "compile": {"per_part": comp, "compiled": sum(v["compile"] == "COMPILED" for v in comp.values()),
       "rate": sum(v["compile"] == "COMPILED" for v in comp.values()) / n_parts},
       "status": status, "lane_status": lane_status,
       "e3_wall_s": {cid: r.get("wall_s") for cid, r in e3.items()},
       "line_f1": {k: {kk: vv for kk, vv in v.items() if kk != "_rows"} for k, v in line.items()},
       "per_bend": pbs, "_rows": {k: v["_rows"] for k, v in line.items()}, "_bend_rows": rows, "_missing": missing,
       "zero_bend_controls": {cid: {"engine_lines": len(preds.get(cid) or []), "lane_lines": len(lane.get(cid) or [])} for cid, c in cases.items() if c["zero"]}}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"compile": {k: (v["compile"], v["compile_reason"]) for k, v in comp.items()}, "status": status, "lane_status": lane_status,
                  "line": {a: {k: {kk: v[k][kk] for kk in ("cases", "micro_f1", "family_macro_f1", "tp", "fp", "fn")} for k in ("all_scans", "engine_ran_scans")}
                           for a, v in line.items()},
                  "per_bend": {k: pbs[k] for k in ("engine_ran", "headline_with_misses", "false_pass_proxy", "false_pass_bends")},
                  "zero": out["zero_bend_controls"]}, indent=1, default=str))
