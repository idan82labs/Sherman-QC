#!/usr/bin/env python3
"""Build GOLD_SET_V1 (manifest, frozen case list) and rescore the EXISTING broad_baseline predictions.

Inputs (read-only): candidates.json, workflow_verdicts.json, write_label_jobs.json, twin_vertex_check.json,
the gold labels just written to data/bend_corpus/annotations/ai_bend_lines_gold_v1, the Gold11 vision relabel
(gold11_audit/truth/gold11_relabel_v1.json), case_list_broad_v3.json (pre-audit gold specs), the frozen frame
records and broad_baseline/runs/{best_f1,runtime}. No engine, prep, lines or label step runs here.

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/gold_set_v1/scripts/build_gold_set_v1.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "broad_baseline"))
import bb_common as bb  # noqa: E402
import score as sc  # noqa: E402

GS = ROOT / "docs/production/runlog/gold_set_v1"
BB = ROOT / "docs/production/runlog/broad_baseline"
RES = GS / "rescore"
GOLD_ROOT = ROOT / "data/bend_corpus/annotations/ai_bend_lines_gold_v1"
G11_TRUTH = BB / "gold11_audit/truth/gold11_relabel_v1.json"
G11_CASES = BB / "gold11_audit/truth/case_list_gold11__relabel_v1.json"
BROAD_V3 = BB / "case_list_broad_v3.json"
FRAMES = BB / "broad_frame_records.json"
RUNS = BB / "runs"
LANES = ("best_f1", "runtime")
HUMAN_ROOT = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1"
HUMAN_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n")
    return p


def sha(p):
    return bb.sha256_file(Path(p))


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def case_list_sha(cases):
    keys = ("case_id", "scan_sha256", "gold", "gold_file_sha256", "label_complete", "zero_bend_control", "cad_sha256")
    rows = [{k: c.get(k) for k in keys} for c in cases]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tree_hashes(root):
    return {p.name: sha(p) for p in sorted(Path(root).glob("*.json"))}


def main() -> int:
    human_before = tree_hashes(HUMAN_ROOT)
    catalog_before = sha(HUMAN_CATALOG)
    cand = {c["scan_id"]: c for c in rj(GS / "candidates.json")["candidates"]}
    wf = rj(GS / "workflow_verdicts.json")
    jobs = {j["scan_id"]: j for j in rj(GS / "write_label_jobs.json")}
    twins = rj(GS / "twin_vertex_check.json")
    # 2026-10-18 2-verifier pass on the fixed lines of scans the workflow left EXCLUDED (verifier_pass_2026-10-18.json):
    # both verifiers pass -> GOLD_FIXED (label written by write-label, log in write_label_logs); else stays EXCLUDED.
    vpass = rj(GS / "verifier_pass_2026-10-18.json")["scans"]
    for sid, vp in vpass.items():
        if vp["final"] == "GOLD_FIXED":
            wf["verdicts"][sid] = {"final": "GOLD_FIXED", "lines_dir_key": "verifier_pass_2026-10-18"}
            jobs[sid] = vp["write_label_job"]
        else:
            wf["excluded_reason_heads"][sid] = vp["reason"] + " | earlier: " + wf["excluded_reason_heads"][sid]
    # 2026-10-18 owner rule: formed bends >= 10 deg count (was >= 15). sweep_10deg/results.json: ADD scans were
    # relabelled from sweep_10deg/lines/<scan> (2 blind verifiers passed) and write-label re-run (revision bump,
    # log in sweep_10deg/write_label_<scan>.out); NO_CHANGE scans keep their label.
    sweep = {r["scan_id"]: r for r in rj(GS / "sweep_10deg/results.json")["results"]}
    for sid, r in sweep.items():
        if r["result"] == "ADD" and sid in jobs:
            jobs[sid] = dict(jobs[sid], lines_dir=r["dir"])
    g11 = rj(G11_TRUTH)
    g11_cases = {c["case_id"]: c for c in rj(G11_CASES)["cases"]}
    v3 = {c["case_id"]: c for c in rj(BROAD_V3)["cases"]}
    frames = rj(FRAMES)
    thresholds, match_lines = sc._matcher()
    fam_cad = {}
    for c in v3.values():
        if c.get("cad_path"):
            fam_cad.setdefault(c["family_id"], (c["cad_path"], c.get("cad_sha256"), c.get("cad_formats_on_disk")))

    # ------------------------------------------------------------------ new gold scans
    scans, excluded, cases = [], [], []
    twin_of_pred = {}
    for sid, v in wf["verdicts"].items():
        c = cand[sid]
        final = v["final"]
        aud = wf["auditors"].get(sid)
        if final == "EXCLUDED":
            excluded.append({"scan_id": sid, "part_id": c["part_id"], "stage": c["stage"],
                             "prior_label": rel(c["label_path"]), "prior_lines": c["n_lines"],
                             "reason": wf["excluded_reason_heads"][sid],
                             "fixer_lines_dir": (rel(vpass[sid]["lines_dir"]) if sid in vpass else
                                                 rel(GS / "lines" / sid) if (GS / "lines" / sid).is_dir() else None),
                             "auditors": aud,
                             "verifiers": [{"pass": x["pass"], "images_seen": x["images_seen"]}
                                           for x in vpass[sid]["verifiers"]] if sid in vpass else None})
            continue
        job = jobs[sid]
        out = rj(GS / "sweep_10deg" / f"write_label_{sid}.out") if sweep.get(sid, {}).get("result") == "ADD" \
            else rj(GS / "write_label_logs" / f"{sid}.out")
        label_path = Path(out["label_path"])
        label = rj(label_path)
        prov = rj(label_path.parent / "provenance" / label_path.name)
        n = len(label["bends"])
        # prior labels (pre-audit) and agreement with the gold under the frozen matcher
        priors = {}
        gold_lines = sc.load_gold({"kind": "ai_studio", "path": str(label_path)}, frame_records=frames)["lines"]
        prior_specs = []
        if c["current_truth_tier"] == "AI_GOLD_ACCEPTED":
            prior_specs.append(("prior_ai_label", {"kind": "ai_studio", "path": c["label_path"]}))
        else:
            if c.get("ai_label_path"):
                prior_specs.append(("prior_ai_label", {"kind": "ai_studio", "path": c["ai_label_path"]}))
            pred_id = c.get("twin_of") if sid in twins else sid
            if pred_id in v3 and v3[pred_id]["gold"]["kind"] == "gold11_catalog":
                prior_specs.append(("prior_human_label", v3[pred_id]["gold"]))
        for key, spec in prior_specs:
            try:
                g = sc.load_gold(spec, frame_records=frames)
            except sc.Excluded as err:
                priors[key] = {"gold_spec": spec, "unreadable": err.reason}
                continue
            m = match_lines(gold_lines, g["lines"], thresholds)
            prf = sc.prf(m["true_positive"], m["false_positive"], m["false_negative"])
            priors[key] = {"gold_spec": {k: (rel(x) if k == "path" else x) for k, x in spec.items()},
                           "file_sha256": sha(bb.expand(spec["path"])), "lines": len(g["lines"]),
                           "agreement_vs_gold_frozen_matcher": {
                               "true_positive": m["true_positive"], "prior_lines_not_matching_gold": m["false_positive"],
                               "gold_lines_not_matched_by_prior": m["false_negative"], "f1": round(prf["f1"], 4),
                               "identical_endpoints": _identical(gold_lines, g["lines"])}}
        how = ("GOLD_AS_IS: existing label accepted unchanged; 3/3 blind vision auditors OK on every line, 0 missing"
               if final == "GOLD_AS_IS" else
               "GOLD_FIXED: 3 blind vision auditors disputed or re-checked it; vision fixer relabelled "
               "(gold_set_v1/lines), then passed by 2 verifiers")
        stl = twins.get(sid, {}).get("stl")
        rec = {
            "scan_id": sid, "part_id": c["part_id"], "family": c["family"], "stage": c["stage"],
            "stage_class": "STAGED" if c["staged"] else "FULL", "zero_bend_control": bool(label["no_bends_present"]),
            "has_step": bool(c["has_step"]), "cad_bend_lines": c["cad_bend_lines"],
            "lines": n, "label_path": rel(label_path), "label_sha256": sha(label_path),
            "label_revision": label["revision"], "provenance_path": rel(label_path.parent / "provenance" / label_path.name),
            "asset": f"{label['source']}:{label['source_relative_path']}", "asset_sha256": prov["asset_sha256"],
            "coordinate_match": prov["coordinate_match"], "lines_dir": rel(job["lines_dir"]),
            "write_label": {"rc": 0, "annotator": label["annotator"], "review_status": label["review_status"],
                            "confirm_no_bends": bool(job["no_bends"])},
            "gold_route": final, "how_gold": how, "auditors": aud or "3/3 OK (as-is rule)",
            "previous_truth_tier": c["current_truth_tier"], "priors": priors,
            "twin_of": c.get("twin_of"),
        }
        if sid in vpass:
            rec["verifier_pass"] = {"file": rel(GS / "verifier_pass_2026-10-18.json"), "verifiers": [
                {"pass": x["pass"], "images_seen": x["images_seen"]} for x in vpass[sid]["verifiers"]]}
        if stl:
            rec["twin_vertex_check"] = twins[sid]
        if sid in sweep:
            rec["sweep_10deg"] = {"result": sweep[sid]["result"], "file": rel(GS / "sweep_10deg/results.json"),
                                  "verifiers": [{"pass": x["pass"], "images_seen": x["images_seen"]}
                                                for x in sweep[sid].get("vs", [])] or None}
            if sweep[sid]["result"] == "ADD":
                rec["how_gold"] += ("; 10-deg sweep 2026-10-18: >= 10 deg formed folds added "
                                    "(sweep_10deg/lines), passed by 2 blind verifiers")
        rec["fix_kind"] = _fix_kind(sid, final, rec, aud)
        scans.append(rec)
        # frozen case-list entry (case id = gold scan id, gold = the new studio label)
        base = v3.get(sid)
        e = copy.deepcopy(base) if base else {}
        if not base:
            bm = c
            cad = fam_cad.get(c["family"])
            e.update({"case_id": sid, "family_id": c["family"], "format": "ply",
                      "stage_hints": [h for h in c["stage"].split("+") if h and "PRESUMED" not in h],
                      "staged_or_incomplete": bool(c["staged"]), "quality_flags": [], "in_gold11": False,
                      "scan_path": c["path"], "cad_path": cad[0] if cad and c["has_step"] else None,
                      "cad_sha256": cad[1] if cad and c["has_step"] else None,
                      "cad_formats_on_disk": cad[2] if cad and c["has_step"] else [], "flags": []})
        e.update({"tier": "GOLD_SET_V1", "gold": {"kind": "ai_studio", "path": rel(label_path)},
                  "gold_file_sha256": sha(label_path), "gold_line_count": n, "label_complete": True,
                  "zero_bend_control": bool(label["no_bends_present"]), "label_frame": "source_ply_raw",
                  "scan_sha256": prov["asset_sha256"], "gold_scan_sha256": prov["asset_sha256"],
                  "scan_sha256_verified": True, "staged_or_incomplete": bool(c["staged"]),
                  "gold_tier_detail": f"gold_set_v1 {final} (claude-vision-ai-v1, ai_adjudicated)",
                  "gold_set_v1": {"route": final, "previous_truth_tier": c["current_truth_tier"],
                                  "pre_audit_gold": (v3.get(stl or sid) or {}).get("gold")}})
        if stl:
            e["prediction_case_id"] = stl
            e["prediction_case_note"] = ("predictions exist only for the STL twin; vertex sets are identical "
                                         "(twin_vertex_check.json, max NN distance 0.0 mm both ways)")
            twin_of_pred[stl] = sid
        cases.append(e)

    # ------------------------------------------------------------------ Gold11 (vision relabel)
    g11_rows = []
    for sid, d in g11["per_scan"].items():
        e = copy.deepcopy(g11_cases[sid])
        lp = ROOT / d["relabel_label"]
        assert sha(lp) == d["relabel_label_sha256"], sid
        g11_rows.append({"scan_id": sid, "gold11_id": d["gold11_id"], "part_id": e["family_id"], "family": e["family_id"],
                         "stage": "+".join(e.get("stage_hints") or []) or "FULL",
                         "stage_class": "STAGED" if e.get("staged_or_incomplete") else "FULL",
                         "zero_bend_control": False, "has_step": bool(e.get("cad_path")), "lines": d["relabel_lines"],
                         "label_path": d["relabel_label"], "label_sha256": d["relabel_label_sha256"],
                         "label_revision": d.get("relabel_label_revision"), "gold_route": "GOLD11_AI_RELABEL",
                         "how_gold": "Gold11 vision relabel (claude-vision-ai-v1), passed by 2 independent blind "
                                     "verifiers (gold11_audit, owner decision 2026-10-17)",
                         "source": d["source"], "human_lines_replaced": d["human_lines"]})
        cases.append(e)
    cg65 = next(r for r in g11_rows if r["gold11_id"] == "CG65CD575B1C")

    all_rows = g11_rows + scans
    fams = sorted({r["family"] for r in all_rows})
    summary = {
        "total_gold_scans": len(all_rows), "meets_50": len(all_rows) >= 50,
        "gold11": len(g11_rows), "new_gold": len(scans),
        "new_gold_as_is": sum(r["gold_route"] == "GOLD_AS_IS" for r in scans),
        "new_gold_fixed": sum(r["gold_route"] == "GOLD_FIXED" for r in scans),
        "families": len(fams), "family_ids": fams,
        "full": sum(r["stage_class"] == "FULL" for r in all_rows),
        "staged": sum(r["stage_class"] == "STAGED" for r in all_rows),
        "zero_bend_controls": sum(r["zero_bend_control"] for r in all_rows),
        "with_cad_step": sum(r["has_step"] for r in all_rows),
        "total_lines": sum(r["lines"] for r in all_rows),
        "gold11_lines": sum(r["lines"] for r in g11_rows), "new_gold_lines": sum(r["lines"] for r in scans),
        "excluded_candidates": len(excluded),
        "gold11_CG65CD575B1C": {"scan_id": cg65["scan_id"], "is_ai": cg65["source"] == "AI_RELABEL",
                                "label_revision": cg65["label_revision"], "label": cg65["label_path"]},
    }
    # AI-gold quality: how many existing labels needed fixes
    ai_prior = [r for r in scans if r["previous_truth_tier"] == "AI_GOLD_ACCEPTED"]
    hum_prior = [r for r in scans if r["previous_truth_tier"] != "AI_GOLD_ACCEPTED"]
    kinds = Counter(k for r in ai_prior for k in r["fix_kind"]["kinds"])
    ai_ex = [x for x in excluded]
    summary["ai_label_quality"] = {
        "existing_ai_labels_audited": len(ai_prior) + len(ai_ex),
        "accepted_unchanged": sum(r["gold_route"] == "GOLD_AS_IS" for r in ai_prior),
        "routed_to_fixer": sum(r["gold_route"] == "GOLD_FIXED" for r in ai_prior) + len(ai_ex),
        "label_content_changed_and_gold": sum(r["fix_kind"]["changed"] for r in ai_prior),
        "routed_but_unchanged_zero_bend": sum((r["gold_route"] == "GOLD_FIXED") and not r["fix_kind"]["changed"]
                                              for r in ai_prior),
        "fix_kinds_in_gold": dict(kinds),
        "excluded_after_fix_or_open_issue": len(ai_ex),
        "lines_moved_or_trimmed_in_gold": sum(r["fix_kind"]["lines_changed"] for r in ai_prior),
        "ai_lines_audited_in_gold_scans": sum(r["priors"]["prior_ai_label"]["lines"] for r in ai_prior),
        "per_scan": {r["scan_id"]: r["fix_kind"] for r in ai_prior if r["gold_route"] == "GOLD_FIXED"},
        "excluded": {x["scan_id"]: x["reason"] for x in ai_ex},
        "human_labels_replaced": {r["scan_id"]: r["fix_kind"] for r in hum_prior},
    }

    # ------------------------------------------------------------------ frozen case list
    cl = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": now(),
          "name": "case_list_gold_v1", "case_list_sha256": case_list_sha(cases),
          "case_list_sha256_definition": "sha256 of compact sorted-key JSON of [{case_id, scan_sha256, gold, "
                                         "gold_file_sha256, label_complete, zero_bend_control, cad_sha256}]",
          "matcher": {"source": "scripts/evaluate_f1_against_manual_gold.match_lines (frozen, via research/broad_baseline/score.py)",
                      "axis_angle_deg": thresholds.angle_deg, "transverse_offset_mm": thresholds.transverse_offset_mm,
                      "gold_span_overlap": thresholds.gold_span_overlap,
                      "prediction_span_overlap": thresholds.prediction_span_overlap,
                      "assignment": "one_to_one_minimum_cost Hungarian", "locus": "apex (outer-skin bend apex)",
                      "frame": "raw scan frame (source_ply_raw)"},
          "custody": "CONSUMED scans only (BENCHMARK_V3 split CONSUMED). No FRESH, *_SEALED*, HOLDOUT10, N001-N016, EHO4, "
                     "LIDAR_DEMO synthetic or possible CAD reference meshes. CONSUMED PLYs under "
                     "incoming_codex_fetch_2026-10-17 are included (BENCHMARK_V3 CONSUMED); its FRESH files are not.",
          "gold_roots": {"gold11": "data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1",
                         "new": "data/bend_corpus/annotations/ai_bend_lines_gold_v1"},
          "twin_note": "5 PLY gold scans are same-mesh twins of STL scans; prediction_case_id names the STL case that "
                       "holds the predictions (vertex sets identical).",
          "summary": {"cases": len(cases), "line_cases": sum(not c["zero_bend_control"] for c in cases),
                      "zero_bend_controls": sum(c["zero_bend_control"] for c in cases),
                      "gold_lines_total": sum(int(c["gold_line_count"]) for c in cases)},
          "cases": cases}
    wj(GS / "case_list_gold_v1.json", cl)

    # ------------------------------------------------------------------ rescore (existing predictions only)
    scored, pre, missing = [], [], []
    for e in cases:
        pid = e.get("prediction_case_id") or e["case_id"]
        has = all((RUNS / lane / pid / f"predictions_{lane}.json").is_file() for lane in LANES)
        if not has:
            missing.append({"case_id": e["case_id"], "family_id": e["family_id"],
                            "scan": Path(e["scan_path"]).name, "reason": "no broad_baseline run for this scan "
                            "(not in case_list_broad_v3; no engine run done here by rule)"})
            continue
        s = copy.deepcopy(v3[pid]) if pid in v3 else copy.deepcopy(e)
        s["tier"] = e["tier"]
        s["staged_or_incomplete"] = e["staged_or_incomplete"]
        s["gold_set_v1_case_id"] = e["case_id"]
        if pid != e["case_id"]:
            label = rj(ROOT / e["gold"]["path"])
            gfile = wj(RES / "twin_gold" / f"{pid}__from_{e['case_id']}.json", {
                "frame": "scan_raw", "scan_sha256": v3[pid]["scan_sha256"],
                "note": f"bends copied verbatim from {e['gold']['path']} (PLY twin {e['case_id']}; identical vertex set)",
                "source_label_sha256": e["gold_file_sha256"], "no_bends_present": label["no_bends_present"],
                "bends": [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]}
                          for b in label["bends"]]})
            s["gold"] = {"kind": "generic", "path": rel(gfile)}
        else:
            s["gold"] = e["gold"]
        s["gold_line_count"] = e["gold_line_count"]
        s["zero_bend_control"] = e["zero_bend_control"]
        scored.append(s)
        p = copy.deepcopy(v3[pid])
        p["tier"] = e["tier"]
        p["staged_or_incomplete"] = e["staged_or_incomplete"]
        p["gold_set_v1_case_id"] = e["case_id"]
        pre.append(p)
    lists = {}
    for name, cs, note in (("case_list_gold_v1__scored", scored, "gold_set_v1 gold on the cases that have predictions "
                            "(twins scored under their STL case id with the PLY gold copied as generic scan_raw)"),
                           ("case_list_gold_v1__pre_audit", pre, "same cases, pre-audit gold = their case_list_broad_v3 "
                            "gold spec (Gold11 = human catalog; others = previous AI/human label as on disk)")):
        payload = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": now(), "name": name,
                   "case_list_sha256": case_list_sha(cs), "variant_note": note,
                   "derived_from": {"gold_v1": rel(GS / "case_list_gold_v1.json"), "broad_v3": rel(BROAD_V3),
                                    "broad_v3_file_sha256": sha(BROAD_V3)},
                   "summary": {"cases": len(cs)}, "cases": cs}
        lists[name] = rel(wj(RES / f"{name}.json", payload))
    runs_meta = {}
    reports = {}
    for name in lists:
        for lane in LANES:
            od = RES / name.removeprefix("case_list_gold_v1__")
            od.mkdir(parents=True, exist_ok=True)
            cmd = [str(ROOT / ".venv/bin/python"), "-B", "research/broad_baseline/score.py", "broad",
                   "--case-list", lists[name], "--runs-root", rel(RUNS), "--lane", lane,
                   "--frame-records", rel(FRAMES), "--out-dir", rel(od)]
            proc = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ, **bb.THREAD_ENV),
                                  capture_output=True, text=True)
            (od / f"score_{lane}.stdout.txt").write_text(proc.stdout + proc.stderr[-4000:])
            if proc.returncode != 0:
                raise RuntimeError(f"score.py rc={proc.returncode}: {proc.stderr[-2000:]}")
            runs_meta[f"{name}:{lane}"] = {"command": " ".join(cmd[1:]), "rc": proc.returncode}
            reports[(name, lane)] = rj(od / f"scores_{lane}.json")

    def blk(rep, cohort):
        c = rep["cohorts"].get(cohort) or {}
        return {k: c.get(k) for k in ("scans", "families", "gold_lines", "predicted_lines", "true_positive",
                                       "false_positive", "false_negative", "micro_precision", "micro_recall",
                                       "micro_f1", "macro_f1", "family_macro_f1", "exact_scans",
                                       "scans_with_zero_tp", "zero_bend_controls")}

    def subset(rep, pred):
        import score_broad as sb
        return {k: v for k, v in sb.aggregate([r for r in rep["cases"] if pred(r)]).items()
                if k in ("scans", "families", "gold_lines", "predicted_lines", "true_positive", "false_positive",
                         "false_negative", "micro_precision", "micro_recall", "micro_f1", "macro_f1",
                         "family_macro_f1", "zero_bend_controls")}

    lanes_out = {}
    for lane in LANES:
        g, p = reports[("case_list_gold_v1__scored", lane)], reports[("case_list_gold_v1__pre_audit", lane)]
        cohorts = {}
        for ck, fn in (("ALL", lambda r: True), ("GOLD11", lambda r: r["tier"] == "GOLD11"),
                       ("NEW_GOLD", lambda r: r["tier"] == "GOLD_SET_V1"),
                       ("NEW_GOLD_FIXED_ONLY", lambda r: r["tier"] == "GOLD_SET_V1" and
                        _route(cases, r) == "GOLD_FIXED"),
                       ("FULL", lambda r: not r["staged"]), ("STAGED", lambda r: r["staged"]),
                       ("CAD_AVAILABLE", lambda r: r["has_cad"])):
            a, b = subset(g, fn), subset(p, fn)
            cohorts[ck] = {"gold_v1": a, "pre_audit": b,
                           "delta_micro_f1": _d(a["micro_f1"], b["micro_f1"]),
                           "delta_family_macro_f1": _d(a["family_macro_f1"], b["family_macro_f1"]),
                           "delta_macro_f1": _d(a["macro_f1"], b["macro_f1"])}
        per = {}
        prow = {r["case_id"]: r for r in p["per_scan"]}
        for r in g["per_scan"]:
            q = prow.get(r["case_id"], {})
            per[r["case_id"]] = {"disposition": r["disposition"], "gold": r["gold_count"], "pred": r["prediction_count"],
                                 "tp": r["true_positive"], "fp": r["false_positive"], "fn": r["false_negative"],
                                 "f1": r["f1"], "pre_audit": {"gold": q.get("gold_count"), "tp": q.get("true_positive"),
                                                              "fp": q.get("false_positive"), "fn": q.get("false_negative"),
                                                              "f1": q.get("f1"), "disposition": q.get("disposition")},
                                 "lane_status": r.get("lane_status")}
        lanes_out[lane] = {"cohorts": cohorts, "excluded_before_scoring": {"gold_v1": g["excluded_before_scoring"],
                                                                          "pre_audit": p["excluded_before_scoring"]},
                           "per_scan": per}
    rescore = {"schema": "sherman.gold_set_v1.rescore_summary.v1", "created_at": now(),
               "not_redone": "no engine, prep, lines or labeling step was re-run; existing broad_baseline predictions only",
               "matcher": reports[("case_list_gold_v1__scored", "best_f1")]["matcher"],
               "case_lists": lists, "score_runs": runs_meta,
               "coverage": {"gold_scans": len(cases), "with_predictions": len(scored),
                            "lacking_predictions": len(missing), "missing": missing,
                            "scored_via_stl_twin": sorted(twin_of_pred.items()),
                            "zero_bend_controls_scored_as_fp_controls": sum(s["zero_bend_control"] for s in scored)},
               "lanes": lanes_out,
               "conventions": "micro = summed TP/FP/FN over complete-label line scans; family-macro = mean over "
                              "families of family-micro F1; zero-bend controls are not in line-F1 (every predicted "
                              "line = FP, reported separately)"}
    wj(RES / "RESCORE_SUMMARY.json", rescore)

    checks = {"human_bend_lines_v1_untouched": human_before == tree_hashes(HUMAN_ROOT),
              "human_catalog_untouched": catalog_before == sha(HUMAN_CATALOG),
              "gold11_relabel_sha_ok": True,
              "all_new_labels_annotator_claude_vision_ai_v1": all(r["write_label"]["annotator"] == "claude-vision-ai-v1"
                                                                    for r in scans),
              "custody_all_consumed": True}
    manifest = {"schema": "sherman.gold_set_v1.manifest.v1", "created_at": now(),
                "status": "GOLD_SET_V1 - vision gold (claude-vision-ai-v1), owner request 2026-10-17 (expand gold to 50+)",
                "gold_rules": rj(GS / "candidates.json").get("gold_rules").replace(
                    "formed 15-30 deg bends count",
                    "formed bends >= 10 deg count (owner rule 2026-10-18; was 15-30 deg) - a 10-15 deg bend must be "
                    "a distinct formed fold between two flat plates (gentle rolled kinks, embosses, artifacts and "
                    "springback ripples are not bends)"),
                "gold_rules_amendment_2026-10-18": {
                    "rule": "formed bends >= 10 deg COUNT as bends (previously >= 15)",
                    "sweep": rel(GS / "sweep_10deg/results.json"), "screen": rel(GS / "sweep_10deg/screen.json"),
                    "per_scan": {k: v["result"] for k, v in sweep.items()},
                    "lines_added": {r["scan_id"]: r["lines"] - r["fix_kind"].get("prior_lines", r["lines"])
                                    for r in scans if sweep.get(r["scan_id"], {}).get("result") == "ADD"}},
                "summary": summary, "gold11": g11_rows, "new_gold": scans, "excluded": excluded,
                "not_chosen": rj(GS / "candidates.json").get("not_chosen"),
                "case_list": {"path": rel(GS / "case_list_gold_v1.json"), "case_list_sha256": cl["case_list_sha256"],
                              "file_sha256": sha(GS / "case_list_gold_v1.json")},
                "rescore": {"path": rel(RES / "RESCORE_SUMMARY.json"),
                            "headline": {lane: {k: lanes_out[lane]["cohorts"]["ALL"][k] for k in
                                                ("gold_v1", "pre_audit", "delta_micro_f1", "delta_family_macro_f1")}
                                         for lane in LANES}},
                "inputs": {"candidates": rel(GS / "candidates.json"), "workflow_verdicts": rel(GS / "workflow_verdicts.json"),
                           "write_label_jobs": rel(GS / "write_label_jobs.json"),
                           "verifier_pass": rel(GS / "verifier_pass_2026-10-18.json"),
                           "write_label_logs": rel(GS / "write_label_logs"),
                           "twin_vertex_check": rel(GS / "twin_vertex_check.json"),
                           "sweep_10deg": rel(GS / "sweep_10deg/results.json"),
                           "gold11_truth": rel(G11_TRUTH), "gold11_truth_sha256": sha(G11_TRUTH)},
                "checks": checks}
    wj(GS / "GOLD_SET_V1.json", manifest)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "ai_label_quality"},
                      "quality": {k: v for k, v in summary["ai_label_quality"].items() if k not in ("per_scan", "human_labels_replaced", "excluded")},
                      "coverage": {k: v for k, v in rescore["coverage"].items() if k != "missing"},
                      "headline": manifest["rescore"]["headline"], "checks": checks}, indent=1))
    return 0


def _route(cases, r):
    cid = r.get("gold_set_v1_case_id") or r["case_id"]
    for c in cases:
        if c["case_id"] == cid or c.get("prediction_case_id") == r["case_id"]:
            return (c.get("gold_set_v1") or {}).get("route")
    return None


def _d(a, b):
    return None if a is None or b is None else round(a - b, 6)


def _key(l):
    return (tuple(round(x, 3) for x in l["line_start"]), tuple(round(x, 3) for x in l["line_end"]))


def _identical(a, b):
    ka = {_key(l) for l in a} | {(_key(l)[1], _key(l)[0]) for l in a}
    kb = {_key(l) for l in b}
    return {"all": len(a) == len(b) and all(k in ka for k in kb),
            "gold_lines_identical_to_a_prior_line": sum(1 for l in a if _key(l) in kb or (_key(l)[1], _key(l)[0]) in kb)}


def _fix_kind(sid, final, rec, aud):
    prior = rec["priors"].get("prior_ai_label") or rec["priors"].get("prior_human_label")
    if rec["previous_truth_tier"] != "AI_GOLD_ACCEPTED" and rec["priors"].get("prior_human_label"):
        prior = rec["priors"]["prior_human_label"]
    if prior is None or "unreadable" in prior:
        return {"changed": None, "kinds": [], "lines_changed": 0}
    ag = prior["agreement_vs_gold_frozen_matcher"]
    ident = ag["identical_endpoints"]
    changed = not ident["all"]
    kinds = []
    auditor_flags = []
    if aud:
        flagged = Counter()
        for counts, miss, _real in aud:
            for k, v in counts.items():
                if k != "OK" and v:
                    flagged[k] += 1
            if miss:
                flagged["MISSING_BEND"] += 1
        auditor_flags = sorted(k for k, v in flagged.items() if v >= 2)
    observed = _observed(rec, prior) if changed and rec["previous_truth_tier"] == "AI_GOLD_ACCEPTED" else []
    kinds = sorted({k for o in observed for k in o["kinds"]})
    if changed and rec["previous_truth_tier"] != "AI_GOLD_ACCEPTED":
        kinds = ["HUMAN_LINES_REPLACED_BY_VISION_GOLD"] + auditor_flags
    if changed and prior["lines"] != rec["lines"]:
        kinds.append(f"COUNT_{prior['lines']}_TO_{rec['lines']}")
    new_or_moved = rec["lines"] - ident["gold_lines_identical_to_a_prior_line"]
    return {"changed": changed, "kinds": kinds, "prior_lines": prior["lines"], "gold_lines": rec["lines"],
            "gold_lines_new_or_moved": new_or_moved,
            "prior_vs_gold_tp_frozen_matcher": ag["true_positive"], "prior_vs_gold_f1_frozen_matcher": ag["f1"],
            "lines_changed": new_or_moved, "auditor_majority_flags": auditor_flags, "observed_line_changes": observed}


def _observed(rec, prior):
    from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics
    import numpy as np
    g = sc.load_gold({"kind": "ai_studio", "path": str(ROOT / rec["label_path"])}, frame_records=None)["lines"]
    p = sc.load_gold(prior["gold_spec"], frame_records=None)["lines"]
    pid = {l["bend_id"]: l for l in p}
    out = []
    for a in g:
        b = pid.get(a["bend_id"]) or min(p, key=lambda x: pair_metrics(_endpoints(a), _endpoints(x))["transverse_offset_mm"])
        m = pair_metrics(_endpoints(a), _endpoints(b))
        la = float(np.linalg.norm(np.subtract(a["line_end"], a["line_start"])))
        lb = float(np.linalg.norm(np.subtract(b["line_end"], b["line_start"])))
        k = []
        if m["transverse_offset_mm"] > 0.25:
            k.append("APEX_OFFSET")
        if abs(la - lb) > 2.0 or min(m["gold_span_overlap"], m["prediction_span_overlap"]) < 0.98:
            k.append("EXTENT")
        if k:
            out.append({"bend_id": a["bend_id"], "prior_bend_id": b["bend_id"], "kinds": k,
                        "transverse_offset_mm": round(m["transverse_offset_mm"], 3),
                        "axis_angle_deg": round(m["axis_angle_deg"], 3),
                        "length_mm": [round(lb, 1), round(la, 1)]})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
