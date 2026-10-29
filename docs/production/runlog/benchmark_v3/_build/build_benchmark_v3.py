"""Build BENCHMARK_V3 (json + md) from BENCHMARK_V2 + the benchmark_v3 queue results. Metadata only.

Reads: benchmark_v2/BENCHMARK_V2.json (already merges review11 verdicts and the LIDAR_DEMO
exclusions; their files are re-checked here), benchmark_v3/inventory.json, and the SEALED
judge table + label-write log in benchmark_v3/fresh_SEALED/_build/. Never opens scan geometry.

Writes:
  docs/production/runlog/benchmark_v3/BENCHMARK_V3.json   (non-sealed: FRESH as counts/status only)
  docs/production/BENCHMARK_V3.md                         (non-sealed: same rule)
  docs/production/runlog/benchmark_v3/fresh_SEALED/BENCHMARK_V3_FRESH_SEALED.json (per-scan FRESH truth summary)
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
BV2 = REPO / "docs/production/runlog/benchmark_v2"
BV3 = REPO / "docs/production/runlog/benchmark_v3"
SEALED = BV3 / "fresh_SEALED"
V2 = BV2 / "BENCHMARK_V2.json"
REVIEW11 = BV2 / "manifest_build/review11_verdicts_2026-10-17.json"
SYNTH = BV2 / "manifest_build/synthetic_demo_exclusions_2026-10-17.json"
INV = BV3 / "inventory.json"
JUDGE = SEALED / "_build/judge_table_2026-10-17.json"
WRITES = SEALED / "_build/label_write_results_2026-10-17.json"
OUT_JSON = BV3 / "BENCHMARK_V3.json"
OUT_MD = REPO / "docs/production/BENCHMARK_V3.md"
OUT_SEALED = SEALED / "BENCHMARK_V3_FRESH_SEALED.json"
SEALED_LABELS = "data/bend_corpus/annotations/ai_bend_lines_fresh_v1_SEALED"
SEALED_LINES = "docs/production/runlog/benchmark_v3/fresh_SEALED"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p) -> str:
    return str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)


v2 = json.loads(V2.read_text())
review11 = json.loads(REVIEW11.read_text())
synth = json.loads(SYNTH.read_text())
inv = json.loads(INV.read_text())
judge = json.loads(JUDGE.read_text())
writes = {w["scan_id"]: w for w in json.loads(WRITES.read_text())}
queue = {e["scan_id"]: e for e in inv["label_queue"]["queue"]}
rows = {r["scan_id"]: r for r in judge["rows"]}
assert set(rows) == set(queue), "judge table and v3 queue differ"
TWINS = judge["ply_twins_of_v2_scans"]
CAD_FIX = {k: int(v) for k, v in judge["cad_count_corrections"].items()}
STAGED_EXPL = judge["staged_explained"]

# ---------- V2 consistency checks (review11 + LIDAR_DEMO already merged into V2) ----------
v2_scans = {}
for pid, part in v2["parts"].items():
    for s in part["scans"]:
        v2_scans[s["scan_id"]] = (pid, part, s)
r11_status = {r["scan_id"]: (r.get("final_status") or r.get("verdict")) for r in review11.get("results", [])}
synth_ids = set(synth["scan_ids"])
merge_checks = {
    "review11_scans_in_v2": sum(1 for s in r11_status if s in v2_scans),
    "review11_scans_total": len(r11_status),
    "lidar_demo_excluded_role_ok": all(v2_scans[s][2]["role"] == "EXCLUDED_NOT_A_SCAN" for s in synth_ids if s in v2_scans),
    "lidar_demo_excluded_count": len(synth_ids),
}

# ---------- per-scan records ----------
scans = []
for sid, (pid, part, s) in v2_scans.items():
    rec = {"scan_id": sid, "part_id": pid, "split": part["split"], "source": "BENCHMARK_V2",
           "path": s["paths"][0] if s.get("paths") else None, "format": s.get("format"),
           "role": s["role"], "truth_source": s.get("primary_truth"),
           "truth_sources_all": s.get("truth_sources"), "stage_hints": s.get("stage_hints"),
           "staged_or_incomplete": bool(s.get("staged_or_incomplete")),
           "studio_label": None, "role_note": s.get("role_note")}
    sl = (s.get("ai_label_run") or {}).get("studio_label") or s.get("studio_label")
    if isinstance(sl, dict) and sl.get("written"):
        rec["studio_label"] = sl.get("label_path")
    scans.append(rec)

sealed_rows = []
cons_rows = []
for sid, r in rows.items():
    q = queue[sid]
    w = writes.get(sid, {})
    fresh = r["split"] == "FRESH"
    cad_task = r["cad_bend_lines_task"]
    cad_inv = q.get("cad_bend_lines")
    cad = CAD_FIX.get(sid, cad_task)
    stage_hints = q.get("stage_hints") or []
    staged = sid in STAGED_EXPL
    if r["verdict"] == "ACCEPT":
        truth = "AI_GOLD_ACCEPTED_SEALED" if fresh else "AI_GOLD_ACCEPTED"
    elif r["verdict"] == "ACCEPT_PARTIAL":
        truth = "AI_GOLD_PARTIAL_SEALED" if fresh else "AI_GOLD_PARTIAL"
    else:
        truth = "NONE_NEEDS_REWORK"
    role = "benchmark_scan"
    if sid in TWINS:
        role = "FORMAT_ALTERNATE_PLY_TWIN"
    label_written = w.get("status") == "WRITTEN"
    detail = {"scan_id": sid, "part_id": r["part_id"], "split": r["split"], "verdict": r["verdict"],
              "label_complete": r["verdict"] == "ACCEPT", "real_bends_judged": r["real_bends"],
              "lines_written": w.get("bends_written"), "cad_bend_lines": cad,
              "cad_bend_lines_task_header": cad_task, "cad_bend_lines_inventory": cad_inv,
              "stage_hints": stage_hints, "staged_or_incomplete": staged,
              "staged_note": STAGED_EXPL.get(sid), "lines_dir": w.get("lines_dir"),
              "lines_json_sha256": w.get("lines_json_sha256"), "studio_label": w.get("label_path"),
              "label_revision": w.get("revision"), "write_status": w.get("status"),
              "write_error": w.get("error"), "twin_of_v2_scan": TWINS.get(sid)}
    if fresh:
        sealed_rows.append(detail)
    else:
        cons_rows.append(detail)
    rec = {"scan_id": sid, "part_id": r["part_id"], "split": r["split"], "source": "BENCHMARK_V3_queue",
           "path": q["path"], "format": q["format"], "role": role, "truth_source": truth,
           "studio_label": w.get("label_path"), "write_status": w.get("status")}
    if fresh:
        # non-sealed: no bend counts, no stage/CAD comparison per scan
        rec["truth_location"] = {"labels": SEALED_LABELS, "lines_dir": w.get("lines_dir")}
        if w.get("status") == "REFUSED":
            rec["write_note"] = "write-label refused (" + (w.get("error") or "") + "); truth exists as accepted lines.json only; needs a PLY export to get a studio label"
    else:
        rec.update(stage_hints=stage_hints, staged_or_incomplete=staged or bool(set(stage_hints) & {"PARTIAL_SCAN_1", "PARTIAL_SCAN_2"}) and r["real_bends"] < cad,
                   n_lines=w.get("bends_written"), cad_bend_lines=cad, twin_of_v2_scan=TWINS.get(sid))
    scans.append(rec)

# ---------- parts ----------
inv_parts = inv["parts"]
parts = defaultdict(lambda: {"scans": []})
for rec in scans:
    parts[rec["part_id"]]["scans"].append(rec["scan_id"])
part_split = {}
part_step = {}
for pid in parts:
    vp = v2["parts"].get(pid)
    ip = inv_parts.get(pid)
    split = (vp or {}).get("split") or (ip or {}).get("split")
    if ip and ip.get("split") and vp and ip["split"] != vp["split"] and vp["split"] != "FRESH":
        split = vp["split"]
    part_split[pid] = split
    step = False
    if vp and any(f in ("stp", "step") for f in vp.get("cad", {}).get("formats", [])):
        step = True
    if ip and (ip.get("has_step_in_download") or ip.get("has_step_in_corpus_v2")):
        step = True
    part_step[pid] = step

by_id = {r["scan_id"]: r for r in scans}
bench = [r for r in scans if r["role"] == "benchmark_scan"]
bench_parts = {r["part_id"] for r in bench}
fresh_bench = [r for r in bench if r["split"] == "FRESH"]
cons_bench = [r for r in bench if r["split"] != "FRESH"]
fresh_accepted = [r for r in fresh_bench if r["truth_source"] == "AI_GOLD_ACCEPTED_SEALED"]
fresh_partial = [r for r in fresh_bench if r["truth_source"] == "AI_GOLD_PARTIAL_SEALED"]
fresh_rework = [r for r in fresh_bench if r["truth_source"] == "NONE_NEEDS_REWORK"]
fresh_parts = {r["part_id"] for r in fresh_bench}
fresh_parts_acc = {r["part_id"] for r in fresh_accepted}
staged_v2 = sum(1 for r in bench if r["source"] == "BENCHMARK_V2" and r["staged_or_incomplete"])
staged_v3_cons = sum(1 for r in bench if r["source"] != "BENCHMARK_V2" and r["split"] != "FRESH" and r.get("staged_or_incomplete"))
staged_v3_fresh = sum(1 for d in sealed_rows if d["staged_or_incomplete"])
v3_labels_fresh = [d for d in sealed_rows if d["write_status"] == "WRITTEN"]
v3_labels_cons = [d for d in cons_rows if d["write_status"] == "WRITTEN"]


# ---------- AI vs CAD count agreement ----------
def cad_class(d):
    cad, ai = d["cad_bend_lines"], d["real_bends_judged"]
    if cad is None or cad < 0:
        return "CAD_UNKNOWN"
    if ai == cad:
        return "AGREE"
    if ai < cad:
        return "AI_LT_CAD_STAGED_UNFORMED" if d["staged_or_incomplete"] else "AI_LT_CAD_CONVENTION"
    return "AI_GT_CAD_EXTRACTOR_UNDERCOUNT"


CAD_EXPLAIN_CONS = {
    "sc_3c93a0bf6f97": "V10 9 keeps the two step folds split at a full-width window; owner window rule (ENGINE_COMPILE 'Windows', names 11979003) merges each into one line -> 7. All 11 V10 segments covered. The V2 STL twin label (sc_d45186653eed) still has 9 lines; owner should align it (merge its B2+B3 and B4+B5).",
    "sc_43a32fe3b4a3": STAGED_EXPL.get("sc_43a32fe3b4a3"),
    "sc_1a5377ac7c15": STAGED_EXPL.get("sc_1a5377ac7c15"),
    "sc_ef0abac61ae7": STAGED_EXPL.get("sc_ef0abac61ae7"),
    "sc_32099fb52638": STAGED_EXPL.get("sc_32099fb52638"),
    "sc_4057ba2144fb": "task header said CAD 3; inventory and step_v10/0d25231fea16.json give 6 (v10 = lenient = same-plates-merged = 6); labels match the 6 CAD bends one to one. PARTIAL_SCAN_2 tag, but every CAD bend is formed.",
}
FRESH_CAD_EXPLAIN = {  # sealed only
    "sc_c0ce5ecd2ba3": "partly bent part; 4 long-edge CAD bends not formed (flat width matches unfolded hooks)",
    "sc_ead8e407c174": "CAD 0 is a V10 refusal on a rolled sleeve; STEP models a 2-fold joggle; lenient census 2",
    "sc_1f1d06c24e9f": "V10 strict 6; lenient census 10, matches the labels one to one",
    "sc_217cfc0f4c57": "V10 accepted 2 axes, refused 4 cylinders; lenient census 6 matches one to one",
    "sc_6fa053514df8": "V10 returned 0 on a 2-body Creo weld assembly; direct STEP cylinder-pair count 10",
    "sc_ead69b8181a8": "CAD 0 = STEP recipe not compiled; drawing D050-2525-045 rev 00 shows 2 folds",
}
cad_cons = []
for d in cons_rows:
    c = cad_class(d)
    cad_cons.append({"scan_id": d["scan_id"], "part_id": d["part_id"], "stage_hints": d["stage_hints"],
                     "ai_real_bends": d["real_bends_judged"], "cad_bend_lines": d["cad_bend_lines"],
                     "class": c, "explained": c == "AGREE" or d["scan_id"] in CAD_EXPLAIN_CONS,
                     "note": CAD_EXPLAIN_CONS.get(d["scan_id"]), "twin_of_v2_scan": d["twin_of_v2_scan"]})
fresh_cad = Counter()
fresh_unexplained = 0
for d in sealed_rows:
    if d["verdict"] not in ("ACCEPT", "ACCEPT_PARTIAL"):
        continue
    c = cad_class(d)
    d["cad_class"] = c
    d["cad_note"] = FRESH_CAD_EXPLAIN.get(d["scan_id"])
    fresh_cad[c] += 1
    if c not in ("AGREE", "CAD_UNKNOWN") and d["scan_id"] not in FRESH_CAD_EXPLAIN:
        fresh_unexplained += 1

# ---------- rejected / excluded ----------
rejected = []
for r in scans:
    if r["source"] == "BENCHMARK_V2" and r["role"] not in ("benchmark_scan", "FORMAT_ALTERNATE"):
        rejected.append({"scan_id": r["scan_id"], "part_id": r["part_id"], "role": r["role"],
                         "basis": ("LIDAR_DEMO synthetic resample (synthetic_demo_exclusions)" if r["scan_id"] in synth_ids
                                   else (r.get("role_note") or ""))[:300],
                         "review11": r11_status.get(r["scan_id"])})
v3_not_accepted = [{"scan_id": r["scan_id"], "part_id": r["part_id"], "split": r["split"],
                    "status": "NEEDS_REWORK",
                    "note": "judge: every kept line lands, but the label set is incomplete; fixable without redoing labelers/verifiers (add the round-1 pairs kept in the _cont state, one lines run under the lock, check the new close-ups). Not written; no truth yet. Details sealed."}
                   for r in fresh_rework]
nq = Counter(e["reason"].split(" (")[0] for e in inv["label_queue"]["not_queued"])

summary = {
    "unique_parts_with_benchmark_scan": len(bench_parts),
    "benchmark_scans": len(bench),
    "parts_with_benchmark_scan_and_step": sum(1 for p in bench_parts if part_step.get(p)),
    "consumed": {"parts": len({r["part_id"] for r in cons_bench}), "scans": len(cons_bench),
                 "from_v2": sum(1 for r in cons_bench if r["source"] == "BENCHMARK_V2"),
                 "new_in_v3": sum(1 for r in cons_bench if r["source"] != "BENCHMARK_V2"),
                 "ply_twins_of_v2_scans_not_counted": len(TWINS)},
    "fresh": {"parts_with_scans": len(fresh_parts), "scans": len(fresh_bench),
              "scans_with_accepted_truth": len(fresh_accepted),
              "parts_with_accepted_truth": len(fresh_parts_acc),
              "scans_partial_truth": len(fresh_partial),
              "parts_only_partial_or_no_truth": sorted(fresh_parts - fresh_parts_acc),
              "scans_needs_rework_no_truth": len(fresh_rework),
              "accepted_bend_lines_total": sum(d["lines_written"] or 0 for d in v3_labels_fresh)
              + sum(d["real_bends_judged"] for d in sealed_rows if d["write_status"] == "REFUSED"),
              "scans_with_sealed_studio_label": len(v3_labels_fresh),
              "scans_accepted_without_studio_label": sum(1 for d in sealed_rows if d["write_status"] == "REFUSED"),
              "scans_with_step_part": sum(1 for r in fresh_bench if part_step.get(r["part_id"])),
              "parts_with_step": sum(1 for p in fresh_parts if part_step.get(p))},
    "staged_or_incomplete_scans": {"v2_consumed": staged_v2, "v3_consumed_new": staged_v3_cons,
                                    "v3_fresh": staged_v3_fresh,
                                    "total": staged_v2 + staged_v3_cons + staged_v3_fresh},
    "truth_source_counts": dict(Counter(r["truth_source"] for r in bench)),
    "v3_label_writes": {"fresh_written_sealed": len(v3_labels_fresh), "consumed_written_ai_bend_lines_v1": len(v3_labels_cons),
                        "fresh_bend_lines_written": sum(d["lines_written"] or 0 for d in v3_labels_fresh),
                        "consumed_bend_lines_written": sum(d["lines_written"] or 0 for d in v3_labels_cons),
                        "refused": [{"scan_id": d["scan_id"], "split": d["split"], "error": d["write_error"]}
                                    for d in sealed_rows + cons_rows if d["write_status"] == "REFUSED"],
                        "not_written_needs_rework": [d["scan_id"] for d in sealed_rows + cons_rows if d["verdict"] == "NEEDS_REWORK"],
                        "replaced_stale_sealed_labels": sum(1 for d in sealed_rows if (d["label_revision"] or 0) > 1),
                        "partial_label_complete_false": [d["scan_id"] for d in sealed_rows + cons_rows if d["verdict"] == "ACCEPT_PARTIAL"]},
    "ai_vs_cad": {"consumed_v3": dict(Counter(x["class"] for x in cad_cons)),
                  "fresh_v3": dict(fresh_cad), "fresh_v3_unexplained_mismatch": fresh_unexplained,
                  "v2_reference": "docs/production/runlog/broad_baseline/cad_crosscheck.json (V2 STEP parts; one confirmed AI error sc_2233372b8df7, fixed 6->8)"},
    "v3_queue_not_queued": dict(nq),
}

source_mismatch = [{
    "issue": "49182000 / 42725000 source mismatch",
    "detail": "The SharePoint link listed under email subject '49182000_B' (msg 1a0516d6677aab46, 2026-08-30) served 42725000_B.ply. The part is taken from the filename (42725000). The file is sha256-identical (d82bd4f4...) to the V2 CONSUMED scan sc_d82bd4f417a2 (42725000_B, GOLD_LINES_AI_LEGACY), so it adds nothing new. 49182000 still has no scan on disk and is not in the benchmark; its real scan must be requested again.",
    "file": "~/ShermanQC/FullProject/data/bend_corpus/incoming_codex_fetch_2026-10-17/20260830_77d65ded/42725000_B.ply",
}]

flags = [
    {"scan_ids": ["sc_3c93a0bf6f97", "sc_d45186653eed"], "issue": "Same mesh (PLY vs STL). V3 label 7 lines (window rule) vs V2 label 9 lines. Counted once (V2 scan). Owner: align the V2 label (merge B2+B3, B4+B5) or decide the convention."},
    {"scan_ids": ["sc_db487cbd29a7", "sc_9b9fd762f361"], "issue": "PLY copy of the V2 STL scan (same 131655 points); labels agree (1). Counted once."},
    {"scan_ids": ["sc_43a32fe3b4a3", "sc_927a2fa13067"], "issue": "PLY twin of the V2 3RD_SCAN STL; labels agree (2). Counted once. Gives the V2 STL-only scan a PLY studio label."},
    {"scan_ids": ["sc_1a5377ac7c15", "sc_8dfdab44605d"], "issue": "PLY twin of the V2 SECOND_SCAN STL (human gold n=1); AI label agrees (1). Counted once."},
    {"scan_ids": ["sc_4057ba2144fb"], "issue": "Workflow task header gave CAD 3; repo inventory/step_v10 give 6. Other judge prompts may carry wrong CAD counts; V3 uses inventory values."},
    {"scan_ids": ["sc_c27168a2de83"], "issue": "Part 28289002 is CONSUMED_UNVERIFIED; its label now sits in ai_bend_lines_v1 (tuning pool), so moving it to FRESH is no longer possible."},
    {"scan_ids": [r["scan_id"] for r in fresh_bench if r.get("write_status") == "REFUSED"], "issue": "FRESH STL scan: write-label refuses non-PLY assets; accepted truth exists only as the sealed lines.json. Export a PLY in the same frame to get a sealed studio label."},
    {"scan_ids": [], "issue": "Several verifiers saw the labeler/reconciler 'notes' field in lines.json before forming their count, so they were not fully blind; judges checked geometry independently. Strip notes from what verifiers read in future runs."},
    {"scan_ids": [], "issue": "One FRESH lines dir had pairs.json overwritten by a killed re-run reconciler; restored byte-identical from the preserved copy (lines.json unchanged). One FRESH lines dir holds a stale render PNG that lines.json does not reference (write-label ignores it)."},
]  # V2 review11 owner_flags (LIDAR demos, sc_9bd69bc6535c) were resolved in V2 (exclusions, relabel_r3)

fresh_rule = {
    "definition": "A part is FRESH iff it never appeared in any tuning or evaluation before (not Gold11, Gold37, physical15 or other-consumed per BENCHMARK_V2.json and merged_inventory.json). Split is per part, never per scan.",
    "use": "FRESH truth is for the single final evaluation only, with one frozen engine config hash. Never for tuning, thresholds, label edits, case selection or any accuracy-loop look. If FRESH truth or results are looked at earlier, or the engine changes after the FRESH run, those parts become CONSUMED and the run is reported as such.",
    "sealed_locations": [SEALED_LABELS, SEALED_LINES],
    "labeling_note": "Labeling agents looked at FRESH scans to create truth (truth creation is not tuning). The engine/accuracy loop must never read FRESH scans, labels or lines dirs.",
    "non_sealed_policy": "This manifest and BENCHMARK_V3.md list FRESH scans with split, truth source and status only. Per-scan FRESH bend counts, line geometry, CAD comparisons and judge details are only in fresh_SEALED/BENCHMARK_V3_FRESH_SEALED.json.",
}

manifest = {
    "schema": "sherman.benchmark_v3.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "generator": rel(Path(__file__).resolve()),
    "inputs": {k: {"path": rel(p), "sha256": sha(p)} for k, p in
               {"benchmark_v2": V2, "review11_verdicts": REVIEW11, "synthetic_demo_exclusions": SYNTH,
                "v3_inventory": INV, "v3_judge_table_SEALED": JUDGE, "v3_label_writes_SEALED": WRITES}.items()},
    "v2_merge_checks": merge_checks,
    "fresh_rule": fresh_rule,
    "rules": v2["rules"],
    "summary": summary,
    "ai_vs_cad_consumed_v3": cad_cons,
    "rejected_or_excluded_scans": {"v2": rejected, "v3_not_accepted": v3_not_accepted,
                                   "v3_not_queued_counts": dict(nq),
                                   "crdownload_incomplete_ignored": inv["custody"].get("crdownload_incomplete_ignored")},
    "source_mismatches": source_mismatch,
    "flags": flags,
    "parts": {pid: {"split": part_split[pid], "has_step": part_step[pid], "scans": sorted(v["scans"])}
              for pid, v in sorted(parts.items())},
    "scans": sorted(scans, key=lambda r: (r["part_id"], r["scan_id"])),
}
OUT_JSON.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")

sealed_doc = {
    "schema": "sherman.benchmark_v3.fresh_sealed.v1", "generated_at": manifest["generated_at"],
    "SEALED": "FRESH truth for the single final evaluation only. Never read from the engine/accuracy loop.",
    "fresh_rule": fresh_rule, "counts": summary["fresh"], "ai_vs_cad_fresh": dict(fresh_cad),
    "scans": sorted(sealed_rows, key=lambda d: (d["part_id"], d["scan_id"])),
}
OUT_SEALED.write_text(json.dumps(sealed_doc, indent=1, ensure_ascii=False) + "\n")

# ---------- markdown ----------
s = summary
L = []
A = L.append
A("# Benchmark V3: broadened bend benchmark with a sealed FRESH split")
A("")
A(f"Generated {manifest['generated_at'][:19]}Z from metadata only by `{manifest['generator']}`. "
  f"Machine-readable manifest: `{rel(OUT_JSON)}` (sha256 `{sha(OUT_JSON)[:16]}...`). "
  "It merges BENCHMARK_V2 (which already includes the review11 verdicts and the LIDAR_DEMO exclusions) "
  "with the 91-scan benchmark_v3 queue from the Codex SharePoint/email fetch of 2026-10-17 (292 files, 7.6 GB; 3 `.crdownload` remnants ignored).")
A("")
A("## FRESH rule (binding)")
A("")
A(f"- {fresh_rule['definition']}")
A(f"- {fresh_rule['use']}")
A(f"- FRESH truth is sealed under `{SEALED_LABELS}` and `{SEALED_LINES}`. {fresh_rule['labeling_note']}")
A(f"- {fresh_rule['non_sealed_policy']}")
A("")
A("## Summary counts")
A("")
A("| Count | Value |")
A("|---|---|")
A(f"| Unique parts with a benchmark scan | {s['unique_parts_with_benchmark_scan']} |")
A(f"| Benchmark scans | {s['benchmark_scans']} |")
A(f"| Parts with a benchmark scan and a STEP | {s['parts_with_benchmark_scan_and_step']} |")
A(f"| CONSUMED parts / scans | {s['consumed']['parts']} / {s['consumed']['scans']} ({s['consumed']['from_v2']} from V2, {s['consumed']['new_in_v3']} new; {s['consumed']['ply_twins_of_v2_scans_not_counted']} PLY twins of V2 scans not counted) |")
A(f"| FRESH parts / scans on disk | {s['fresh']['parts_with_scans']} / {s['fresh']['scans']} |")
A(f"| FRESH scans / parts with accepted truth | {s['fresh']['scans_with_accepted_truth']} / {s['fresh']['parts_with_accepted_truth']} |")
A(f"| FRESH accepted bend lines (sealed) | {s['fresh']['accepted_bend_lines_total']} |")
A(f"| FRESH partial truth (label_complete=false) | {s['fresh']['scans_partial_truth']} scan |")
A(f"| FRESH needs rework (no truth) | {s['fresh']['scans_needs_rework_no_truth']} scan |")
A(f"| FRESH parts / scans with a STEP | {s['fresh']['parts_with_step']} / {s['fresh']['scans_with_step_part']} |")
A(f"| Staged or incomplete scans | {s['staged_or_incomplete_scans']['total']} (V2 {s['staged_or_incomplete_scans']['v2_consumed']}, V3 consumed {s['staged_or_incomplete_scans']['v3_consumed_new']}, V3 FRESH {s['staged_or_incomplete_scans']['v3_fresh']}) |")
A("")
A("Truth sources over benchmark scans: " + ", ".join(f"{k} {v}" for k, v in sorted(s["truth_source_counts"].items())) + ".")
A("")
w = s["v3_label_writes"]
A("## V3 label writes (2026-10-17)")
A("")
A(f"- FRESH: {w['fresh_written_sealed']} sealed studio labels written to `{SEALED_LABELS}` ({w['fresh_bend_lines_written']} lines), "
  f"{w['replaced_stale_sealed_labels']} of them replacing stale pre-reconcile labels. None went to `ai_bend_lines_v1`.")
A(f"- CONSUMED: {w['consumed_written_ai_bend_lines_v1']} labels written to `ai_bend_lines_v1` ({w['consumed_bend_lines_written']} lines).")
A(f"- Refused by write-label: " + "; ".join(f"`{x['scan_id']}` ({x['split']}): {x['error']}" for x in w["refused"]) + ".")
A(f"- Not written (NEEDS_REWORK): " + ", ".join(f"`{x}`" for x in w["not_written_needs_rework"]) + ".")
A(f"- ACCEPT_PARTIAL with `label_complete=false` in the provenance sidecar: " + ", ".join(f"`{x}`" for x in w["partial_label_complete_false"]) + ".")
A("- No drop_labels were requested, so no `_final` lines runs were needed. The BENCHMARK_V2 rebuild for the 42288005 relabel was already done (a dry-run rebuild matched the on-disk V2 json and md exactly), so it was not rerun.")
A("")
A("## AI vs CAD bend-count agreement")
A("")
A("CONSUMED V3 scans (per scan):")
A("")
A("| Scan | Part | Stage | AI real bends | CAD bend lines | Class | Note |")
A("|---|---|---|---|---|---|---|")
for x in cad_cons:
    A(f"| `{x['scan_id']}` | {x['part_id']} | {', '.join(x['stage_hints']) or '-'} | {x['ai_real_bends']} | {x['cad_bend_lines']} | {x['class']} | {(x['note'] or '').replace('|', '/')} |")
A("")
A("FRESH V3 scans (counts only; per-scan rows are sealed):")
A("")
A("| Class | Scans |")
A("|---|---|")
for k, v in sorted(fresh_cad.items()):
    A(f"| {k} | {v} |")
A("")
A(f"Every FRESH mismatch with a known CAD count is explained by the judge (staged/unformed bends, or V10 extractor refusal/under-count confirmed by a lenient STEP census, a direct cylinder count or the drawing); unexplained: {fresh_unexplained}. "
  "V2 reference: `docs/production/runlog/broad_baseline/cad_crosscheck.json` (one confirmed AI error, `sc_2233372b8df7`, fixed 6 -> 8).")
A("")
A("## Rejected or excluded scans")
A("")
A("| Scan | Part | Role | Basis |")
A("|---|---|---|---|")
for x in rejected:
    A(f"| `{x['scan_id']}` | {x['part_id']} | {x['role']} | {x['basis'][:160].replace('|', '/')} |")
for x in v3_not_accepted:
    A(f"| `{x['scan_id']}` | {x['part_id']} | FRESH {x['status']} | {x['note']} |")
A("")
A("V3 inventory files not queued: " + ", ".join(f"{k} {v}" for k, v in sorted(nq.items())) + ".")
A("")
A("## Source mismatch: 49182000 / 42725000")
A("")
A(source_mismatch[0]["detail"])
A("")
A("## Flags for the owner")
A("")
for f in flags:
    ids = ", ".join(f"`{i}`" for i in f.get("scan_ids", []))
    A(f"- {ids + ': ' if ids else ''}{f['issue']}")
A("")
A("## Scans by part")
A("")
A("| Part | Split | STEP | Scan | Source | Role | Truth source |")
A("|---|---|---|---|---|---|---|")
for r in manifest["scans"]:
    A(f"| {r['part_id']} | {part_split.get(r['part_id'])} | {'yes' if part_step.get(r['part_id']) else 'no'} | `{r['scan_id']}` | "
      f"{'V2' if r['source'] == 'BENCHMARK_V2' else 'V3'} | {r['role']} | {r['truth_source']} |")
A("")
A("## Custody")
A("")
A("Excluded and not opened: HOLDOUT10, sealed truth, N001-N016, EHO4, held-out manifests, and the 11 sealed items Codex skipped "
  "(16231000_03, 38168001_02, 38173001_01, 26687013_A00, 31409000_01, 47215000_B, 42637000_B, 50476000_A, 31367000_A00, 30820000_A01). "
  "No V3 scan belongs to those parts. Nothing was reset, cleaned or committed.")
OUT_MD.write_text("\n".join(L) + "\n")
print(json.dumps(summary, indent=1))
