"""Build INVENTORY.json for the zero-shot programme (phase 1, 2026-10-25). Metadata only.

Reads (all non-sealed, plus the two owner-authorized consumed label sources of ledger seq 8):
  benchmark_v3/BENCHMARK_V3.json, benchmark_v3/inventory.json (not_queued), gold_set_v2 case list + manifest,
  freeze/cases_A1.json, freeze/devcheck_SEALED.json (authorized, ledger seq 8),
  data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED/ (authorized, ledger seq 8: bend counts only here),
  ai_bend_lines_v1 / gold_v1 / gold11_relabel_v1 (counts), final_eval results.json + fixes/compile_after.json.
Never opens: benchmark_v3/fresh_SEALED, recipes_devcheck_SEALED, *_devcheck_SEALED other than devcheck_SEALED.json,
HOLDOUT10, N001-N016, EHO4, rhino-preview. Writes INVENTORY.json next to tools/.
usage: python3 -I build_inventory.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
RL = REPO / "docs/production/runlog"
OUT = RL / "zeroshot_2026-10-25"
ANN = REPO / "data/bend_corpus/annotations"


def rj(p):
    return json.loads(Path(p).read_text())


def fam_group(part):
    m = re.match(r"(\d{5})", part or "")
    return m.group(1) if m else part


b3 = rj(RL / "benchmark_v3/BENCHMARK_V3.json")
inv3 = rj(RL / "benchmark_v3/inventory.json")
gl2 = rj(RL / "gold_set_v2/case_list_gold_v2_all.json")["cases"]
g2man = rj(RL / "gold_set_v2/GOLD_SET_V2.json")
a1 = rj(RL / "freeze/cases_A1.json")["cases"]
dc = rj(RL / "freeze/devcheck_SEALED.json")["cases"]
fres = rj(RL / "final_eval_2026-10-19/results.json")["fresh"]
comp = rj(RL / "final_eval_2026-10-19/fixes/compile_after.json")["rows"]
comp_by_sku = {r["sku"]: r for r in comp}

g2_by = {}
for c in gl2:
    for k in {c["case_id"], c.get("prediction_case_id")} - {None}:
        g2_by[k] = c
a1_ids = {"sc_" + c["case_ref"][3:]: c for c in a1}
dc_ids = {"sc_" + c["case_ref"][3:]: c for c in dc}
DC_NOT_RUN = {"sc_eaa89c6d1f1f", "sc_152f1b1aacb2", "sc_c623d2e45bff", "sc_9ef2cf4e52cb", "sc_7a1773eca9fa"}
fresh_sku = {p: v["sku"] for p, v in fres["compile"]["per_part"].items()}
fresh_e3_scans = {"sc_" + k[3:]: v for k, v in fres["status"].items()} if isinstance(fres.get("status"), dict) else {}
g2_excl = {e["scan_id"]: e["reason"] for e in g2man["excluded"]["candidate_stage_skips"]}


def label_info(rel):
    p = REPO / rel
    if not p.is_file():
        return None
    d = rj(p)
    prov = p.parent / "provenance" / p.name
    complete = True
    if prov.is_file():
        pv = rj(prov)
        complete = pv.get("label_complete", True) is not False
    return {"path": rel, "n_lines": len(d.get("bends") or []), "no_bends_present": bool(d.get("no_bends_present")),
            "annotator": d.get("annotator"), "review_status": d.get("review_status"), "label_complete": complete}


rows = []
for s in b3["scans"]:
    sid, part = s["scan_id"], s["part_id"]
    row = {"scan_id": sid, "part_id": part, "family": part, "family_group5": fam_group(part), "format": s["format"],
           "path": s["path"], "benchmark_split": s["split"], "benchmark_role": s["role"],
           "benchmark_truth_source": s["truth_source"]}
    # ---- custody
    if s["role"] != "benchmark_scan" and not (s["role"] == "FORMAT_ALTERNATE_PLY_TWIN" and sid in g2_by):
        row["custody"] = "NOT_A_BENCHMARK_SCAN"
        row["usable"] = False
        row["why_not_usable"] = s["role"]
    else:
        if s["split"] == "FRESH":
            row["custody"] = "CONSUMED_FRESH"
        elif sid in dc_ids:
            row["custody"] = "CONSUMED_DEVCHECK"
        else:
            row["custody"] = "DEV_CONSUMED"
        row["usable"] = True
    if sid in g2_excl and row["usable"]:
        reason = g2_excl[sid]
        if "CAD_REFERENCE_MESH" in reason:
            row.update(usable=False, why_not_usable="POSSIBLE_CAD_REFERENCE_MESH (not a physical scan)")
        elif "twin" in reason.lower() or "same mesh" in reason.lower():
            row.update(usable=False, why_not_usable="DUPLICATE_MESH_TWIN: " + reason[:120])
        else:
            row["gold_set_v2_exclusion"] = reason
    # ---- stage
    g = g2_by.get(sid)
    staged = None
    if g is not None:
        staged = bool(g.get("staged_or_incomplete"))
    elif sid in dc_ids:
        staged = dc_ids[sid]["stage_class"] != "COMPLETE"
    else:
        hints = [h for h in s.get("stage_hints") or [] if h not in ("PRELIM_REV",)]
        staged = bool([h for h in hints if "PARTIAL" in h or "SCAN_1" in h or "SCAN_2" in h or "SECOND" in h
                       or "THIRD" in h or "FLAT" in h]) if hints else False
    row["stage"] = "partial" if staged else "full"
    row["stage_hints"] = s.get("stage_hints") or []
    # ---- gold
    gold = None
    if g is not None and g.get("gold", {}).get("path"):
        li = label_info(g["gold"]["path"])
        gold = {"set": "GOLD_SET_V2" + ("/GOLD11_RELABEL" if g["tier"] == "GOLD11" else "/" + g["tier"]),
                "quality": "A_audited_vision (3 blind vision auditors; GOLD_SET_V2 rule)", **(li or {}),
                "label_complete": bool(g.get("label_complete", True))}
    elif s["truth_source"] in ("AI_GOLD_ACCEPTED_SEALED", "AI_GOLD_PARTIAL_SEALED") and s.get("studio_label"):
        li = label_info(s["studio_label"])
        if li:
            gold = {"set": "AI_FRESH_V1_SEALED (consumed; ledger seq 8 read authorization)",
                    "quality": "B_accepted_ai_vision (2 blind labelers + reconcile + verifiers + judge; not 3-auditor audited)",
                    **li}
            if s["truth_source"] == "AI_GOLD_PARTIAL_SEALED":
                gold["label_complete"] = False
    elif s["truth_source"] == "AI_GOLD_ACCEPTED" and s.get("studio_label"):
        li = label_info(s["studio_label"])
        if li:
            gold = {"set": "AI_BEND_LINES_V1 (benchmark v2/v3 accepted)", "quality": "B_accepted_ai_vision", **li}
    if gold is None and s["truth_source"] == "AI_GOLD_ACCEPTED" and row["usable"]:
        gold = {"set": None, "quality": "C_ai_lines_only_no_studio_label_or_excluded", "n_lines": None}
    row["gold"] = gold
    row["has_gold"] = bool(gold and gold.get("set") and gold.get("label_complete", True))
    if row.get("gold_set_v2_exclusion") and "verifier pass failed" in row["gold_set_v2_exclusion"]:
        row["has_gold"] = False
        row["gold_issue"] = "label failed GOLD_SET_V1 2-verifier pass (B3 on inner skin): needs re-mark"
    # ---- CAD
    has_step = bool(b3["parts"].get(part, {}).get("has_step"))
    cad = {"has_step": has_step}
    sku = None
    if g is not None and g.get("cad_sha256"):
        sku = "DEV-" + g["cad_sha256"][:10].upper()
        cad["step_path"] = g.get("cad_path")
    elif sid in dc_ids:
        sku = "DEV-" + dc_ids[sid]["cad_sha256"][:10].upper()
        cad["step_path"] = dc_ids[sid]["cad_path"]
    elif part in fresh_sku:
        sku = fresh_sku[part]
    elif has_step:
        for r in comp:
            if part in Path(r["path"]).name:
                sku = r["sku"]; cad["step_path"] = r["path"]
                break
    if sku:
        r = comp_by_sku.get(sku)
        cad.update(sku=sku, step_path=cad.get("step_path") or (r or {}).get("path"),
                   compile_status=(r or {}).get("status", "UNKNOWN"), compile_reason=(r or {}).get("reason"),
                   compiled_bends=(r or {}).get("bends"))
        cad["compiles"] = cad["compile_status"] == "COMPILED"
    elif has_step:
        cad.update(compile_status="STEP_NOT_LOCATED", compiles=False)
    else:
        cad.update(compile_status="NO_STEP", compiles=False)
    row["cad"] = cad
    # ---- engine history (who has seen this scan)
    hist = []
    if sid in a1_ids:
        hist.append("A1_TUNING_SET")
    if sid in dc_ids:
        hist.append("DEVCHECK_E3_NOT_RUN_(no frozen truth table)" if sid in DC_NOT_RUN else "DEVCHECK_E3_RAN")
    if s["split"] == "FRESH" and ("fr_" + sid[3:]) in fres["status"]:
        hist.append("FRESH_FINAL_EVAL_E3+best_f1")
    pid = (g or {}).get("prediction_case_id") or sid
    row["a0_prediction_case_id"] = pid
    if (RL / "broad_baseline/runs/best_f1" / pid / "predictions_a0.json").is_file():
        hist.append("A0_ZERO_SHOT_OUTPUT_EXISTS")
    row["engine_history"] = hist
    rows.append(row)

# ---- on-disk scans outside BENCHMARK_V3 (from the v3 inventory "not queued" list)
extra = []
for x in inv3["label_queue"]["not_queued"]:
    reason = x.get("reason", "")
    if reason.startswith("EYAL50_QUARANTINE"):
        extra.append({"scan_id": x["scan_id"], "part_id": x["part_id"], "path": x["path"], "format": x["format"],
                      "custody": "QUARANTINE_EYAL50 (owner decision: 20 dev / 30 locked holdout split file not read)",
                      "usable": False, "has_gold": False, "stage": "full", "cad": {"has_step": False}})
    elif reason.startswith("SKU_OVERLAPS_SEALED_SET"):
        extra.append({"scan_id": x["scan_id"], "part_id": x["part_id"], "path": x["path"], "format": x["format"],
                      "custody": "FORBIDDEN_ADJACENT (same SKU as a HOLDOUT10 scan)", "usable": False, "has_gold": False,
                      "stage": "full", "cad": {"has_step": True}})

usable = [r for r in rows if r["usable"]]
fmt_ok = [r for r in usable if r["format"] == "ply"]
summary = {
    "benchmark_v3_rows": len(rows),
    "usable_physical_scans": len(usable),
    "usable_by_custody": dict(Counter(r["custody"] for r in usable)),
    "usable_families": len({r["family"] for r in usable}),
    "usable_family_groups5": len({r["family_group5"] for r in usable}),
    "usable_ply": len(fmt_ok), "usable_non_ply": len(usable) - len(fmt_ok),
    "usable_full_partial": dict(Counter(r["stage"] for r in usable)),
    "with_complete_gold": sum(r["has_gold"] for r in usable),
    "with_complete_gold_by_custody": dict(Counter(r["custody"] for r in usable if r["has_gold"])),
    "with_complete_gold_by_quality": dict(Counter(r["gold"]["quality"][:1] for r in usable if r["has_gold"])),
    "gold_lines_total": sum((r["gold"] or {}).get("n_lines") or 0 for r in usable if r["has_gold"]),
    "zero_bend_gold_controls": sum(1 for r in usable if r["has_gold"] and (r["gold"].get("n_lines") == 0)),
    "usable_without_complete_gold": [{"scan_id": r["scan_id"], "part_id": r["part_id"], "custody": r["custody"],
                                      "why": r.get("gold_issue") or (r["gold"] or {}).get("quality")
                                      or r["benchmark_truth_source"]} for r in usable if not r["has_gold"]],
    "gold_with_step": sum(1 for r in usable if r["has_gold"] and r["cad"]["has_step"]),
    "gold_with_compiling_step": sum(1 for r in usable if r["has_gold"] and r["cad"].get("compiles")),
    "gold_no_cad": sum(1 for r in usable if r["has_gold"] and not r["cad"]["has_step"]),
    "gold_families": len({r["family"] for r in usable if r["has_gold"]}),
    "never_seen_by_any_engine_and_gold": sum(1 for r in usable if r["has_gold"] and not r["engine_history"]),
    "on_disk_outside_benchmark_not_usable": len(extra),
}
doc = {"schema": "sherman.zeroshot.inventory.v1", "created": "2026-10-25", "builder": "tools/build_inventory.py",
       "custody_note": "Every usable scan is CONSUMED (dev data). No unconsumed physical scan with gold exists locally.",
       "summary": summary, "scans": rows, "outside_benchmark": extra}
(OUT / "INVENTORY.json").write_text(json.dumps(doc, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
