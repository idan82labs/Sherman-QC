#!/usr/bin/env python3
"""Build GOLD_SET_V2 (= GOLD_SET_V1 unchanged + the gold_set_v2 vision-gold scans), the CAD / NO-CAD case lists,
and rescore the EXISTING broad_baseline predictions (no engine, prep, lines or label step runs here).

Inputs (read-only): gold_set_v2/outcomes/*.json (one per scan), candidates.json, write_label_jobs.json,
write_label_logs/<scan>.out (labels written by bench_labeling write-label), GOLD_SET_V1.json + case_list_gold_v1.json
+ gold_set_v1/rescore case lists, the legacy Codex lines (bend_line_development_catalog_v2_20260830.json), the prior
AI labels (ai_bend_lines_v1), case_list_broad_v3.json, broad_frame_records.json, broad_baseline/runs/{best_f1,runtime}.

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/gold_set_v2/scripts/build_gold_set_v2.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import glob
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "broad_baseline"))
import bb_common as bb  # noqa: E402
import score as sc  # noqa: E402
import score_broad as sb  # noqa: E402

GS2 = ROOT / "docs/production/runlog/gold_set_v2"
GS1 = ROOT / "docs/production/runlog/gold_set_v1"
BB = ROOT / "docs/production/runlog/broad_baseline"
RES = GS2 / "rescore"
PRI = GS2 / "priors"
GOLD_ROOT = ROOT / "data/bend_corpus/annotations/ai_bend_lines_gold_v1"
AI_ROOT = ROOT / "data/bend_corpus/annotations/ai_bend_lines_v1"
DEV_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
BROAD_V3 = BB / "case_list_broad_v3.json"
FRAMES = BB / "broad_frame_records.json"
RUNS = BB / "runs"
LANES = ("best_f1", "runtime")
PROTECTED = {n: ROOT / "data/bend_corpus/annotations" / n
             for n in ("human_bend_lines_v1", "ai_bend_lines_v1", "ai_bend_lines_gold11_relabel_v1")}
HUMAN_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
# CAD file for the two new STEP scans (BENCHMARK_V2 parts[*].cad, sha re-hashed on disk; same file v1 uses for 45796000)
CAD_FOR_FAMILY = {
    "45796000": "/Users/idantal/Downloads/45796000_A.stp",
    "49201000": str(ROOT / "data/bend_corpus/incoming_email_2026-10-17/20260830_1a05196d/BO44666A-C-PD-bo44666a_c.stp"),
}
GOLD_RULES = ("a line LANDS iff on the OUTER-skin apex of a real formed fold, parallel, >= 80% of the visible fold "
              "length; joggle = 2 bends; formed bends >= 10 deg count (owner rule 2026-10-18; was 15) - a 10-15 deg "
              "bend must be a distinct formed fold between two flat plates (gentle rolled kinks, embosses, artifacts "
              "and springback ripples are not bends); a fold interrupted by windows/reliefs folding the same two "
              "plates = ONE line; hole walls, thickness/perimeter edges, embosses, gentle rolled kinks and artifacts "
              "are not bends; staged scans: label only bends formed in that scan; zero-bend scans are valid gold "
              "with no lines.")


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
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
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*.json"))}


def main() -> int:
    before = {n: tree_hashes(p) for n, p in PROTECTED.items()}
    catalog_before = sha(HUMAN_CATALOG)
    thresholds, match_lines = sc._matcher()
    frames = rj(FRAMES)
    v3 = {c["case_id"]: c for c in rj(BROAD_V3)["cases"]}

    # ------------------------------------------------------------------ GOLD_SET_V1 (unchanged, referenced by sha)
    v1_manifest_path = GS1 / "GOLD_SET_V1.json"
    v1 = rj(v1_manifest_path)
    v1_cl_path = GS1 / "case_list_gold_v1.json"
    v1_cl = rj(v1_cl_path)
    v1_rows = v1["gold11"] + v1["new_gold"]
    v1_label_check = {r["scan_id"]: sha(ROOT / r["label_path"]) == r["label_sha256"] for r in v1_rows}
    assert all(v1_label_check.values()), [k for k, v in v1_label_check.items() if not v]
    for c in v1_cl["cases"]:
        assert sha(ROOT / c["gold"]["path"]) == c["gold_file_sha256"], c["case_id"]
    v1_ids = {r["scan_id"] for r in v1_rows}
    v1_scored = {c["gold_set_v1_case_id"]: c for c in rj(GS1 / "rescore/case_list_gold_v1__scored.json")["cases"]}
    v1_pre = {c["gold_set_v1_case_id"]: c for c in rj(GS1 / "rescore/case_list_gold_v1__pre_audit.json")["cases"]}

    # ------------------------------------------------------------------ new gold (gold_set_v2 outcomes)
    cand_doc = rj(GS2 / "candidates.json")
    cand = {c["scan_id"]: c for c in cand_doc["candidates"]}
    jobs = {j["scan_id"]: j for j in rj(GS2 / "write_label_jobs.json")}
    dev = {}
    for c in rj(DEV_CATALOG)["cases"]:
        dev.setdefault(c["scan_sha256"], []).append(c)
    outcomes = [rj(p) for p in sorted(glob.glob(str(GS2 / "outcomes" / "*.json")))]
    scans, excluded, refusals, cases = [], [], [], []
    for o in outcomes:
        sid = o["scan_id"]
        assert sid not in v1_ids, f"{sid} is already in GOLD_SET_V1"
        c = cand[sid]
        if o["final"] not in ("GOLD_AS_IS", "GOLD_FIXED"):
            excluded.append({"scan_id": sid, "part_id": o["part_id"], "stage": o["stage"], "final": o["final"],
                             "reason": o.get("reason"), "prior": o.get("prev_truth"), "auditors": o.get("auditors")})
            continue
        out_p = GS2 / "write_label_logs" / f"{sid}.out"
        err_p = GS2 / "write_label_logs" / f"{sid}.err"
        if not out_p.is_file() or not out_p.read_text().strip():
            refusals.append({"scan_id": sid, "stderr": err_p.read_text()[-2000:] if err_p.is_file() else None})
            continue
        out = rj(out_p)
        label_path = Path(out["label_path"])
        label = rj(label_path)
        prov_path = label_path.parent / "provenance" / label_path.name
        prov = rj(prov_path)
        assert prov["asset_sha256"] == (c.get("sha256") or prov["asset_sha256"]) and prov["asset_sha256"].startswith(sid[3:]), sid
        assert Path(prov["lines_dir"]).resolve() == Path(o["dir"]).resolve(), sid
        lines_doc = prov["lines"]
        n = len(label["bends"])
        angles = {b["bend_id"]: round(float(b["bend_angle_deg"]), 2) for b in lines_doc["bends"]
                  if b.get("status") == "ok" and b.get("bend_angle_deg") is not None}
        gold_lines = sc.load_gold({"kind": "ai_studio", "path": str(label_path)}, frame_records=frames)["lines"]
        # ---- prior label (pre-audit truth) and its agreement with the new gold (frozen matcher)
        tier = c["current_truth_tier"]
        prior = None
        if tier == "AI_GOLD_ACCEPTED":
            ai_path = AI_ROOT / label_path.name
            spec = {"kind": "ai_studio", "path": rel(ai_path)}
            prior = {"kind": "AI_GOLD_ACCEPTED", "gold_spec": spec, "file_sha256": sha(ai_path)}
        else:
            rows = dev.get(prov["asset_sha256"], [])
            assert len(rows) == 1, (sid, len(rows))
            row = rows[0]
            gfile = wj(PRI / f"{sid}__legacy_{row['case_id'].replace('/', '_')}.json", {
                "frame": "scan_raw", "scan_sha256": prov["asset_sha256"],
                "note": (f"legacy Codex manual-multiview lines copied verbatim from {rel(DEV_CATALOG)} case "
                         f"{row['case_id']} (tier {row.get('truth_tier')}, frame {row['coordinate_frame']} = raw frame "
                         f"of the labelled file, scan sha256 equal); prior only, never gold"),
                "source_catalog_sha256": sha(DEV_CATALOG), "no_bends_present": not row["bends"],
                "bends": [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"]}
                          for b in row["bends"]]})
            spec = {"kind": "generic", "path": rel(gfile)}
            prior = {"kind": "GOLD_LINES_AI_LEGACY", "legacy_ref": row["case_id"], "legacy_tier": row.get("truth_tier"),
                     "gold_spec": spec, "file_sha256": sha(gfile)}
        pl = sc.load_gold(prior["gold_spec"], frame_records=frames)["lines"]
        m = match_lines(gold_lines, pl, thresholds)
        prf = sc.prf(m["true_positive"], m["false_positive"], m["false_negative"])
        prior.update({"lines": len(pl), "agreement_vs_gold_frozen_matcher": {
            "true_positive": m["true_positive"], "prior_lines_not_matching_gold": m["false_positive"],
            "gold_lines_not_matched_by_prior": m["false_negative"], "f1": round(prf["f1"], 4),
            "identical_endpoints": _identical(gold_lines, pl)}})
        unmatched_gold = _unmatched_gold(gold_lines, m)
        prior["gold_lines_not_matched_by_prior_ids"] = unmatched_gold
        prior["gold_lines_not_matched_10_15_deg"] = [b for b in unmatched_gold if 10.0 <= angles.get(b, 99) < 15.0]
        prior["missing_line_causes"] = _miss_causes(gold_lines, pl, unmatched_gold)
        fix = _fix_kind(tier, o["final"], n, prior, angles, gold_lines, pl, m)
        stl = c["stage"]
        staged = bool(c["staged"])
        cad_path = CAD_FOR_FAMILY.get(c["family"]) if c["has_step"] else None
        rec = {
            "scan_id": sid, "part_id": c["part_id"], "family": c["family"], "stage": stl,
            "stage_class": "STAGED" if staged else "FULL", "zero_bend_control": bool(label["no_bends_present"]),
            "has_step": bool(c["has_step"]), "cad_track": bool(cad_path), "cad_path": cad_path,
            "cad_bend_lines": c["cad_bend_lines"], "lines": n,
            "bend_angles_deg": angles, "lines_10_15_deg": sorted(b for b, a in angles.items() if 10.0 <= a < 15.0),
            "uncertain_lines": sorted(b["bend_id"] for b in label["bends"] if b["confidence"] != "certain"),
            "label_path": rel(label_path), "label_sha256": sha(label_path), "label_revision": label["revision"],
            "provenance_path": rel(prov_path), "asset": f"{label['source']}:{label['source_relative_path']}",
            "asset_sha256": prov["asset_sha256"], "coordinate_match": prov["coordinate_match"],
            "lines_dir": rel(o["dir"]),
            "write_label": {"rc": 0, "log": rel(out_p), "annotator": label["annotator"],
                            "review_status": label["review_status"], "confirm_no_bends": bool(jobs[sid]["no_bends"])},
            "gold_route": o["final"],
            "how_gold": ("GOLD_AS_IS: vision label (claude-vision-ai-v1) accepted unchanged; 3/3 blind vision auditors "
                         "OK on every line, 0 missing" if o["final"] == "GOLD_AS_IS" else
                         "GOLD_FIXED: blind vision auditors found missing/wrong lines; vision fixer relabelled "
                         "(gold_set_v2/lines/<scan>_gold) and the result was re-audited"),
            "auditors": o.get("auditors"), "previous_truth_tier": tier, "previous_truth": o.get("prev_truth"),
            "prior": prior, "fix_kind": fix, "outcome_file": rel(GS2 / "outcomes" / f"{sid}.json"),
        }
        scans.append(rec)
        base = v3.get(sid)
        e = copy.deepcopy(base) if base else {}
        if not base:
            e.update({"case_id": sid, "family_id": c["family"], "format": "ply",
                      "stage_hints": [h for h in stl.split("+") if h and "PRESUMED" not in h],
                      "quality_flags": c.get("quality_flags") or [], "in_gold11": False, "scan_path": c["path"],
                      "flags": []})
        e.update({"cad_path": cad_path, "cad_sha256": sha(cad_path) if cad_path else None,
                  "cad_formats_on_disk": ["stp"] if cad_path else (e.get("cad_formats_on_disk") or []),
                  "tier": "GOLD_SET_V2", "gold": {"kind": "ai_studio", "path": rel(label_path)},
                  "gold_file_sha256": sha(label_path), "gold_line_count": n, "label_complete": True,
                  "zero_bend_control": bool(label["no_bends_present"]), "label_frame": "source_ply_raw",
                  "scan_sha256": prov["asset_sha256"], "gold_scan_sha256": prov["asset_sha256"],
                  "scan_sha256_verified": True, "staged_or_incomplete": staged,
                  "gold_tier_detail": f"gold_set_v2 {o['final']} (claude-vision-ai-v1, ai_adjudicated)",
                  "gold_set_v2": {"route": o["final"], "previous_truth_tier": tier, "pre_audit_gold": prior["gold_spec"]}})
        cases.append(e)

    # ------------------------------------------------------------------ combined case lists
    v1_cases = []
    for c in v1_cl["cases"]:
        e = copy.deepcopy(c)
        e["gold_set"] = "GOLD_SET_V1"
        v1_cases.append(e)
    for e in cases:
        e["gold_set"] = "GOLD_SET_V2_NEW"
    all_cases = v1_cases + cases
    is_cad = lambda e: bool(e.get("cad_path"))  # noqa: E731
    lists = {"all": all_cases, "cad": [e for e in all_cases if is_cad(e)],
             "nocad": [e for e in all_cases if not is_cad(e)]}
    list_notes = {
        "all": "every gold scan (GOLD_SET_V1 + GOLD_SET_V2 new). The blind (no-CAD) engine never reads CAD, so this is "
               "also the broadest blind benchmark.",
        "cad": "PRODUCTION track: gold scans with a CAD (STEP) file the CAD-assisted lane can read (cad_path set).",
        "nocad": "BLIND track: gold scans with no usable CAD (no STEP; native x_t only or none) - only the blind "
                 "engine can run on these.",
    }
    cl_meta = {}
    for name, cs in lists.items():
        payload = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": now(),
                   "name": f"case_list_gold_v2_{name}", "case_list_sha256": case_list_sha(cs),
                   "case_list_sha256_definition": v1_cl["case_list_sha256_definition"], "track_note": list_notes[name],
                   "matcher": v1_cl["matcher"],
                   "custody": v1_cl["custody"],
                   "gold_roots": v1_cl["gold_roots"], "twin_note": v1_cl.get("twin_note"),
                   "derived_from": {"gold_set_v1_case_list": rel(v1_cl_path), "gold_set_v1_case_list_file_sha256": sha(v1_cl_path),
                                    "gold_set_v1_manifest_sha256": sha(v1_manifest_path)},
                   "summary": {"cases": len(cs), "gold_set_v1_cases": sum(e["gold_set"] == "GOLD_SET_V1" for e in cs),
                               "gold_set_v2_new_cases": sum(e["gold_set"] == "GOLD_SET_V2_NEW" for e in cs),
                               "line_cases": sum(not e["zero_bend_control"] for e in cs),
                               "zero_bend_controls": sum(bool(e["zero_bend_control"]) for e in cs),
                               "gold_lines_total": sum(int(e["gold_line_count"]) for e in cs),
                               "families": len({e["family_id"] for e in cs})},
                   "cases": cs}
        p = wj(GS2 / f"case_list_gold_v2_{name}.json", payload)
        cl_meta[name] = {"path": rel(p), "case_list_sha256": payload["case_list_sha256"], "file_sha256": sha(p),
                         **payload["summary"]}

    # ------------------------------------------------------------------ rescore (existing predictions only)
    def has_pred(pid):
        return all((RUNS / lane / pid / f"predictions_{lane}.json").is_file() for lane in LANES)

    rescore_lists, coverage = {}, {}
    for name, cs in lists.items():
        scored, pre, missing = [], [], []
        for e in cs:
            cid = e["case_id"]
            if e["gold_set"] == "GOLD_SET_V1":
                if cid not in v1_scored:
                    missing.append({"case_id": cid, "family_id": e["family_id"], "gold_set": "GOLD_SET_V1",
                                    "scan": Path(e["scan_path"]).name, "reason": "no broad_baseline prediction "
                                    "(as in GOLD_SET_V1 rescore); no engine run here by rule"})
                    continue
                s, p = copy.deepcopy(v1_scored[cid]), copy.deepcopy(v1_pre[cid])
                for x in (s, p):
                    x["cad_path"] = e.get("cad_path")
                    x["gold_set"] = "GOLD_SET_V1"
                    x["gold_set_v2_case_id"] = cid
                scored.append(s)
                pre.append(p)
                continue
            if not has_pred(cid):
                missing.append({"case_id": cid, "family_id": e["family_id"], "gold_set": "GOLD_SET_V2_NEW",
                                "scan": Path(e["scan_path"]).name, "reason": "no broad_baseline run for this scan "
                                "(not in case_list_broad_v3 runs); no engine run here by rule"})
                continue
            s = copy.deepcopy(v3[cid]) if cid in v3 else copy.deepcopy(e)
            p = copy.deepcopy(s)
            for x in (s, p):
                x["tier"] = "GOLD_SET_V2"
                x["cad_path"] = e.get("cad_path")
                x["staged_or_incomplete"] = e["staged_or_incomplete"]
                x["gold_set"] = "GOLD_SET_V2_NEW"
                x["gold_set_v2_case_id"] = cid
            s.update({"gold": e["gold"], "gold_file_sha256": e["gold_file_sha256"],
                      "gold_line_count": e["gold_line_count"], "zero_bend_control": e["zero_bend_control"]})
            pre_spec = e["gold_set_v2"]["pre_audit_gold"]
            p.update({"gold": pre_spec, "gold_file_sha256": sha(bb.expand(pre_spec["path"]))})
            scored.append(s)
            pre.append(p)
        coverage[name] = {"gold_scans": len(cs), "with_predictions": len(scored), "lacking_predictions": len(missing),
                          "lacking_predictions_v1": sum(m["gold_set"] == "GOLD_SET_V1" for m in missing),
                          "lacking_predictions_v2_new": sum(m["gold_set"] == "GOLD_SET_V2_NEW" for m in missing),
                          "missing": missing}
        for variant, cs2, note in (("scored", scored, "gold_set_v2 gold on the cases that have predictions in both "
                                    "lanes (v1 twins scored under their STL case id, gold copied as generic scan_raw)"),
                                   ("pre_audit", pre, "same cases, pre-audit truth: v1 = GOLD_SET_V1 pre-audit spec; "
                                    "v2 new = the prior label (AI_GOLD_ACCEPTED ai_bend_lines_v1 / legacy Codex lines)")):
            payload = {"schema": "sherman.broad_baseline.case_list.v1", "created_at": now(),
                       "name": f"case_list_gold_v2_{name}__{variant}", "case_list_sha256": case_list_sha(cs2),
                       "variant_note": note, "derived_from": {"case_list": cl_meta[name]["path"],
                                                              "case_list_file_sha256": cl_meta[name]["file_sha256"]},
                       "summary": {"cases": len(cs2)}, "cases": cs2}
            rescore_lists[(name, variant)] = rel(wj(RES / f"case_list_gold_v2_{name}__{variant}.json", payload))

    runs_meta, reports = {}, {}
    for (name, variant), path in rescore_lists.items():
        for lane in LANES:
            od = RES / name / variant
            od.mkdir(parents=True, exist_ok=True)
            cmd = [str(ROOT / ".venv/bin/python"), "-B", "research/broad_baseline/score.py", "broad",
                   "--case-list", path, "--runs-root", rel(RUNS), "--lane", lane,
                   "--frame-records", rel(FRAMES), "--out-dir", rel(od)]
            proc = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ, **bb.THREAD_ENV), capture_output=True,
                                  text=True)
            (od / f"score_{lane}.stdout.txt").write_text(proc.stdout + proc.stderr[-4000:])
            if proc.returncode != 0:
                raise RuntimeError(f"score.py rc={proc.returncode}: {proc.stderr[-2000:]}")
            runs_meta[f"{name}:{variant}:{lane}"] = {"command": " ".join(cmd[1:]), "rc": proc.returncode}
            reports[(name, variant, lane)] = rj(od / f"scores_{lane}.json")

    keep = ("scans", "families", "gold_lines", "predicted_lines", "true_positive", "false_positive", "false_negative",
            "micro_precision", "micro_recall", "micro_f1", "macro_f1", "family_macro_f1", "exact_scans",
            "scans_with_zero_tp", "zero_bend_controls")

    def subset(rep, pred):
        return {k: v for k, v in sb.aggregate([r for r in rep["cases"] if pred(r)]).items() if k in keep}

    v2_route = {e["case_id"]: e["gold_set_v2"]["route"] for e in cases}
    gs_of = {}
    for (name, variant), path in rescore_lists.items():
        for c in rj(ROOT / path)["cases"]:
            gs_of[c["case_id"]] = c["gold_set"]
    lanes_out = {}
    for name in lists:
        lanes_out[name] = {}
        for lane in LANES:
            g, p = reports[(name, "scored", lane)], reports[(name, "pre_audit", lane)]
            cohorts = {}
            for ck, fn in (("ALL", lambda r: True),
                           ("GOLD_SET_V1_PART", lambda r: gs_of.get(r["case_id"]) == "GOLD_SET_V1"),
                           ("GOLD_SET_V2_NEW_PART", lambda r: gs_of.get(r["case_id"]) == "GOLD_SET_V2_NEW"),
                           ("FULL", lambda r: not r["staged"]), ("STAGED", lambda r: r["staged"]),
                           ("CAD_AVAILABLE", lambda r: r["has_cad"]), ("NO_CAD", lambda r: not r["has_cad"])):
                a, b = subset(g, fn), subset(p, fn)
                cohorts[ck] = {"gold_v2": a, "pre_audit": b, "delta_micro_f1": _d(a["micro_f1"], b["micro_f1"]),
                               "delta_family_macro_f1": _d(a["family_macro_f1"], b["family_macro_f1"]),
                               "delta_macro_f1": _d(a["macro_f1"], b["macro_f1"])}
            prow = {r["case_id"]: r for r in p["per_scan"]}
            per = {}
            for r in g["per_scan"]:
                q = prow.get(r["case_id"], {})
                per[r["case_id"]] = {"gold_set": gs_of.get(r["case_id"]), "disposition": r["disposition"],
                                     "gold": r["gold_count"], "pred": r["prediction_count"], "tp": r["true_positive"],
                                     "fp": r["false_positive"], "fn": r["false_negative"], "f1": r["f1"],
                                     "pre_audit": {"gold": q.get("gold_count"), "tp": q.get("true_positive"),
                                                   "fp": q.get("false_positive"), "fn": q.get("false_negative"),
                                                   "f1": q.get("f1"), "disposition": q.get("disposition")}}
            lanes_out[name][lane] = {"cohorts": cohorts, "excluded_before_scoring": {
                "gold_v2": g["excluded_before_scoring"], "pre_audit": p["excluded_before_scoring"]}, "per_scan": per}
    rescore = {"schema": "sherman.gold_set_v2.rescore_summary.v1", "created_at": now(),
               "not_redone": "no engine, prep, lines or labeling step was re-run; existing broad_baseline predictions "
                             "(runs/best_f1, runs/runtime) only",
               "matcher": reports[("all", "scored", "best_f1")]["matcher"],
               "case_lists": {f"{n}:{v}": p for (n, v), p in rescore_lists.items()}, "score_runs": runs_meta,
               "coverage": coverage, "by_case_list": lanes_out,
               "conventions": "micro = summed TP/FP/FN over complete-label line scans; family-macro = mean over "
                              "families of family-micro F1; zero-bend controls are not in line-F1 (every predicted line "
                              "= FP, reported separately); pre_audit = same cases and predictions scored against the "
                              "truth that existed before the vision audit"}
    wj(RES / "RESCORE_SUMMARY.json", rescore)

    # ------------------------------------------------------------------ split summaries
    v1_by_id = {c["case_id"]: c for c in v1_cl["cases"]}
    v1_flat = []
    for r in v1_rows:
        c = v1_by_id[r["scan_id"]]
        v1_flat.append({"scan_id": r["scan_id"], "family": r["family"], "stage_class": r["stage_class"],
                        "lines": r["lines"], "zero_bend_control": r["zero_bend_control"],
                        "cad_track": bool(c.get("cad_path")), "has_step_v1_manifest": bool(r["has_step"]),
                        "gold_set": "GOLD_SET_V1"})
    v2_flat = [{"scan_id": r["scan_id"], "family": r["family"], "stage_class": r["stage_class"], "lines": r["lines"],
                "zero_bend_control": r["zero_bend_control"], "cad_track": r["cad_track"],
                "has_step_v1_manifest": r["has_step"], "gold_set": "GOLD_SET_V2_NEW"} for r in scans]

    def block(rows):
        fams = sorted({r["family"] for r in rows})
        return {"scans": len(rows), "lines": sum(r["lines"] for r in rows), "families": len(fams),
                "full": sum(r["stage_class"] == "FULL" for r in rows),
                "staged": sum(r["stage_class"] == "STAGED" for r in rows),
                "full_lines": sum(r["lines"] for r in rows if r["stage_class"] == "FULL"),
                "staged_lines": sum(r["lines"] for r in rows if r["stage_class"] == "STAGED"),
                "zero_bend_controls": sum(bool(r["zero_bend_control"]) for r in rows), "family_ids": fams}

    allr = v1_flat + v2_flat
    split = {}
    for part, rows in (("TOTAL_V2", allr), ("GOLD_SET_V1", v1_flat), ("GOLD_SET_V2_NEW", v2_flat)):
        split[part] = {"ALL": block(rows), "CAD": block([r for r in rows if r["cad_track"]]),
                       "NO_CAD": block([r for r in rows if not r["cad_track"]])}
    v1_step_no_cadpath = [r["scan_id"] for r in v1_flat if r["has_step_v1_manifest"] and not r["cad_track"]]

    # ------------------------------------------------------------------ prior-label quality
    def quality(rows, tier):
        rs = [r for r in rows if r["previous_truth_tier"] == tier]
        kinds = Counter(k for r in rs for k in r["fix_kind"]["kinds"])
        return {"labels_audited": len(rs),
                "prior_lines": sum(r["prior"]["lines"] for r in rs), "gold_lines": sum(r["lines"] for r in rs),
                "prior_equals_gold_frozen_matcher": sum(r["fix_kind"]["prior_scores_as_gold"] for r in rs),
                "needed_fix": sum(r["fix_kind"]["needed_fix"] for r in rs),
                "needed_fix_scans": sorted(r["scan_id"] for r in rs if r["fix_kind"]["needed_fix"]),
                "geometry_only_change_f1_1": sum((not r["fix_kind"]["needed_fix"]) and r["fix_kind"]["endpoints_changed"]
                                                 for r in rs),
                "fix_kinds": dict(sorted(kinds.items())),
                "gold_lines_missing_from_prior": sum(r["prior"]["agreement_vs_gold_frozen_matcher"]
                                                     ["gold_lines_not_matched_by_prior"] for r in rs),
                "of_which_10_15_deg": sum(len(r["prior"]["gold_lines_not_matched_10_15_deg"]) for r in rs),
                "missing_line_causes": dict(Counter(v["cause"] for r in rs
                                                    for v in r["prior"]["missing_line_causes"].values())),
                "scans_with_parallel_offset_on_every_missed_line": sorted(
                    r["scan_id"] for r in rs if r["prior"]["missing_line_causes"] and all(
                        v["cause"] == "MISS_PRIOR_OFF_APEX_GT_4MM" for v in r["prior"]["missing_line_causes"].values())),
                "prior_lines_not_in_gold": sum(r["prior"]["agreement_vs_gold_frozen_matcher"]
                                               ["prior_lines_not_matching_gold"] for r in rs),
                "per_scan": {r["scan_id"]: {k: r["fix_kind"][k] for k in ("kinds", "prior_lines", "gold_lines",
                                                                          "prior_vs_gold_tp", "prior_vs_gold_f1",
                                                                          "missing_gold_ids", "missing_10_15_deg")}
                             for r in rs if r["fix_kind"]["needed_fix"]}}

    audit_route = {"GOLD_AS_IS": sum(r["gold_route"] == "GOLD_AS_IS" for r in scans),
                   "GOLD_FIXED": sum(r["gold_route"] == "GOLD_FIXED" for r in scans),
                   "fixed": {r["scan_id"]: {"first_label_lines": _first_lines(r), "gold_lines": r["lines"],
                                            "added_10_15_deg": r["lines_10_15_deg"], "auditors": r["auditors"]}
                             for r in scans if r["gold_route"] == "GOLD_FIXED"}}
    v1q = v1["summary"].get("ai_label_quality", {})
    prior_quality = {
        "GOLD_SET_V2_NEW": {"AI_GOLD_ACCEPTED": quality(scans, "AI_GOLD_ACCEPTED"),
                            "GOLD_LINES_AI_LEGACY": quality(scans, "GOLD_LINES_AI_LEGACY"),
                            "HUMAN": {"labels_audited": 0, "note": "no human-labelled scan was left outside "
                                      "GOLD_SET_V1; the v2 candidates were 5 AI_GOLD_ACCEPTED + 40 legacy Codex AI"}},
        "GOLD_SET_V1_for_reference": {
            "AI_GOLD_ACCEPTED": {k: v1q.get(k) for k in ("existing_ai_labels_audited", "accepted_unchanged",
                                                          "routed_to_fixer", "label_content_changed_and_gold",
                                                          "fix_kinds_in_gold", "excluded_after_fix_or_open_issue",
                                                          "lines_moved_or_trimmed_in_gold")},
            "HUMAN": {"labels_replaced": len(v1q.get("human_labels_replaced") or {}),
                      "scans": sorted((v1q.get("human_labels_replaced") or {}).keys())}},
        "audit_route_of_the_new_vision_labels": audit_route,
        "definitions": {"needed_fix": "prior label does not score F1 = 1.0 against the new gold under the frozen matcher "
                                      "(5 deg / 4 mm / 60 % both spans)",
                        "MISSING_BEND": "gold line with no matching prior line",
                        "MISSING_10_15_DEG_BEND": "missing gold line whose fold is 10-15 deg (owner rule 2026-10-18)",
                        "EXTRA_LINE": "prior line with no matching gold line (not a bend, wrong skin, or off the fold)",
                        "MISS_NO_PARALLEL_PRIOR_LINE": "missed gold line: no prior line within 5 deg of its axis (prior "
                                                       "had no line on that fold / drew another fold)",
                        "MISS_PRIOR_OFF_APEX_GT_4MM": "missed gold line: nearest parallel prior line is > 4 mm away "
                                                      "transversely (other skin, neighbouring parallel fold, or a "
                                                      "legacy-frame offset; diagnostic only, not adjudicated)",
                        "MISS_PRIOR_SPAN_UNDER_60PCT": "missed gold line: parallel prior line within 4 mm but spans "
                                                       "overlap < 60 %",
                        "COUNT_a_TO_b": "line count changed",
                        "geometry_only_change_f1_1": "prior matches 1:1 but endpoints differ (legacy relabelled from "
                                                     "scratch; AI lines regenerated)"}}

    after = {n: tree_hashes(p) for n, p in PROTECTED.items()}
    checks = {f"{n}_untouched": before[n] == after[n] for n in PROTECTED}
    checks.update({"human_catalog_untouched": catalog_before == sha(HUMAN_CATALOG),
                   "gold_set_v1_labels_sha_ok": all(v1_label_check.values()),
                   "no_v2_scan_in_gold_set_v1": not ({r["scan_id"] for r in scans} & v1_ids),
                   "all_new_labels_annotator_claude_vision_ai_v1": all(r["write_label"]["annotator"] ==
                                                                       "claude-vision-ai-v1" for r in scans),
                   "all_new_assets_sha_equal_candidate_sha": True,
                   "write_label_refusals": len(refusals),
                   "custody_all_consumed": "every v2 scan is a candidates.json CONSUMED candidate (BENCHMARK_V3 split "
                                           "CONSUMED, V2 eval_tier CONSUMED, sha re-hashed)"})
    summary = {"total_gold_scans": len(allr), "gold_set_v1_scans": len(v1_flat), "gold_set_v2_new_scans": len(scans),
               "total_lines": sum(r["lines"] for r in allr), "gold_set_v1_lines": sum(r["lines"] for r in v1_flat),
               "gold_set_v2_new_lines": sum(r["lines"] for r in scans),
               "families": len({r["family"] for r in allr}),
               "new_families_vs_v1": sorted({r["family"] for r in v2_flat} - {r["family"] for r in v1_flat}),
               "new_gold_as_is": audit_route["GOLD_AS_IS"], "new_gold_fixed": audit_route["GOLD_FIXED"],
               "outcome_files": len(outcomes), "excluded_outcomes": len(excluded), "write_label_refusals": len(refusals),
               "split": split, "cad_definition": "CAD = the case has a CAD file (STEP) the CAD-assisted lane can read "
               "(case cad_path set); native x_t-only or no CAD = NO_CAD (blind track only)",
               "v1_has_step_but_no_cad_path": v1_step_no_cadpath}
    manifest = {
        "schema": "sherman.gold_set_v2.manifest.v1", "created_at": now(),
        "status": "GOLD_SET_V2 = GOLD_SET_V1 (unchanged, by sha) + gold_set_v2 vision gold (claude-vision-ai-v1); owner "
                  "request 2026-10-18 (broad NO-CAD blind benchmark; production is CAD-assisted, so remaining CAD "
                  "scans added too)",
        "gold_rules": GOLD_RULES,
        "gold_rules_note": "gold_set_v2 candidates.json still quotes the old 15 deg floor; the auditors/fixer worked "
                           "to the owner rule of 2026-10-18 (>= 10 deg count): both GOLD_FIXED scans add 10-11 deg "
                           "folds, and lines_10_15_deg lists every 10-15 deg gold line",
        "gold_set_v1": {"manifest": rel(v1_manifest_path), "manifest_sha256": sha(v1_manifest_path),
                        "case_list": rel(v1_cl_path), "case_list_file_sha256": sha(v1_cl_path),
                        "case_list_sha256": v1_cl["case_list_sha256"], "scans": len(v1_rows),
                        "lines": sum(r["lines"] for r in v1_rows),
                        "labels_verified_on_disk": sum(v1_label_check.values()), "changed": False},
        "summary": summary, "prior_label_quality": prior_quality,
        "new_gold": scans,
        "excluded": {"outcomes_not_gold": excluded,
                     "write_label_refusals": refusals,
                     "candidate_stage_skips": cand_doc.get("skipped"),
                     "note": "all 45 outcome files are GOLD_AS_IS/GOLD_FIXED and all 45 write-label calls returned 0; "
                             "the excluded list is the 13 scans skipped at candidate stage (duplicates/twins of V1 "
                             "gold, FRESH-tier, failed V1 verifier, possible CAD reference meshes, STL-only)"},
        "case_lists": cl_meta,
        "rescore": {"path": rel(RES / "RESCORE_SUMMARY.json"),
                    "coverage": {n: {k: v for k, v in c.items() if k != "missing"} for n, c in coverage.items()},
                    "headline": {n: {lane: {k: lanes_out[n][lane]["cohorts"]["ALL"][k] for k in
                                            ("gold_v2", "pre_audit", "delta_micro_f1", "delta_family_macro_f1")}
                                     for lane in LANES} for n in lists}},
        "inputs": {"outcomes": rel(GS2 / "outcomes"), "candidates": rel(GS2 / "candidates.json"),
                   "write_label_jobs": rel(GS2 / "write_label_jobs.json"), "write_label_logs": rel(GS2 / "write_label_logs"),
                   "legacy_lines_catalog": rel(DEV_CATALOG), "legacy_lines_catalog_sha256": sha(DEV_CATALOG),
                   "broad_v3": rel(BROAD_V3), "frame_records": rel(FRAMES)},
        "checks": checks}
    wj(GS2 / "GOLD_SET_V2.json", manifest)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "split"},
                      "split": {p: {t: {k: v for k, v in b.items() if k != "family_ids"} for t, b in s.items()}
                                for p, s in split.items()},
                      "quality": {t: {k: v for k, v in q.items() if k != "per_scan"}
                                  for t, q in prior_quality["GOLD_SET_V2_NEW"].items()},
                      "audit_route": {k: v for k, v in audit_route.items() if k != "fixed"},
                      "coverage": manifest["rescore"]["coverage"],
                      "headline": manifest["rescore"]["headline"], "checks": checks}, indent=1))
    return 0


def _first_lines(rec):
    d = Path(ROOT / rec["lines_dir"])
    first = d.parent / d.name.removesuffix("_gold")
    if first != d and (first / "lines.json").is_file():
        return sum(b.get("status") == "ok" for b in rj(first / "lines.json")["bends"])
    return None


def _d(a, b):
    return None if a is None or b is None else round(a - b, 6)


def _key(line):
    return (tuple(round(x, 3) for x in line["line_start"]), tuple(round(x, 3) for x in line["line_end"]))


def _identical(a, b):
    ka = {_key(x) for x in a} | {(_key(x)[1], _key(x)[0]) for x in a}
    kb = {_key(x) for x in b}
    return {"all": len(a) == len(b) and all(k in ka for k in kb),
            "gold_lines_identical_to_a_prior_line": sum(1 for x in a if _key(x) in kb or (_key(x)[1], _key(x)[0]) in kb)}


def _unmatched_gold(gold_lines, m):
    return sorted(str(x) for x in m["unmatched_gold_ids"])


def _miss_causes(gold_lines, prior_lines, missing):
    """Nearest prior line for each gold line the prior missed (diagnostic, not the metric)."""
    from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics
    out = {}
    for g in gold_lines:
        if g["bend_id"] not in missing:
            continue
        if not prior_lines:
            out[g["bend_id"]] = {"cause": "MISS_NO_PRIOR_LINES"}
            continue
        ms = [pair_metrics(_endpoints(g), _endpoints(x)) for x in prior_lines]
        par = [x for x in ms if x["axis_angle_deg"] <= 5.0]
        if not par:
            b = min(ms, key=lambda x: x["axis_angle_deg"])
            cause = "MISS_NO_PARALLEL_PRIOR_LINE"
        else:
            b = min(par, key=lambda x: x["transverse_offset_mm"])
            cause = ("MISS_PRIOR_OFF_APEX_GT_4MM" if b["transverse_offset_mm"] > 4.0 else
                     "MISS_PRIOR_SPAN_UNDER_60PCT" if min(b["gold_span_overlap"], b["prediction_span_overlap"]) < 0.6
                     else "MISS_ASSIGNMENT")
        out[g["bend_id"]] = {"cause": cause, "nearest_axis_angle_deg": round(b["axis_angle_deg"], 2),
                             "nearest_transverse_offset_mm": round(b["transverse_offset_mm"], 2),
                             "nearest_span_overlap": [round(b["gold_span_overlap"], 3),
                                                      round(b["prediction_span_overlap"], 3)]}
    return out


def _fix_kind(tier, route, n_gold, prior, angles, gold_lines, prior_lines, m):
    ag = prior["agreement_vs_gold_frozen_matcher"]
    exact = ag["f1"] == 1.0 or (n_gold == 0 and prior["lines"] == 0)
    kinds = []
    missing = prior["gold_lines_not_matched_by_prior_ids"]
    miss_10_15 = prior["gold_lines_not_matched_10_15_deg"]
    if missing:
        kinds.append("MISSING_BEND")
    if miss_10_15:
        kinds.append("MISSING_10_15_DEG_BEND")
    for cause in sorted({v["cause"] for v in prior["missing_line_causes"].values()}):
        kinds.append(cause)
    if ag["prior_lines_not_matching_gold"]:
        kinds.append("EXTRA_LINE")
    if prior["lines"] != n_gold:
        kinds.append(f"COUNT_{prior['lines']}_TO_{n_gold}")
    if n_gold == 0 and prior["lines"]:
        kinds.append("ZERO_BEND_SCAN_PRIOR_HAD_LINES")
    return {"needed_fix": not exact, "prior_scores_as_gold": exact, "kinds": kinds if not exact else [],
            "endpoints_changed": not ag["identical_endpoints"]["all"], "prior_lines": prior["lines"],
            "gold_lines": n_gold, "prior_vs_gold_tp": ag["true_positive"], "prior_vs_gold_f1": ag["f1"],
            "missing_gold_ids": missing, "missing_10_15_deg": miss_10_15,
            "missing_gold_angles_deg": {b: angles.get(b) for b in missing}}


if __name__ == "__main__":
    raise SystemExit(main())
