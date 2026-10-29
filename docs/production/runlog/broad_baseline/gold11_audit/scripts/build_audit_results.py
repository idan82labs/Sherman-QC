#!/usr/bin/env python3
"""Roll the finished Gold11 audit up into gold11_audit/audit_results.json.

Read-only over every input; no auditor, relabel, verifier, prep, lines or engine step is re-run.

1. Agent outputs of workflow wf_37989420-76c, taken verbatim from its subagent transcripts:
   33 auditors (last StructuredOutput; the 5 the run re-used from the killed run's cache are read
   from that cache and compared with the echo), 11 relabel reports (final text) and 22 relabel
   verifiers (last StructuredOutput). The extraction is checked against the compact record the
   workflow handed to this step (EXPECTED_COMPACT).
2. Per human line: the majority verdict of the 3 auditors and the auditors' median offset.
3. Per human line, a mechanical cross-check against the AI relabel apex lines (bench_labeling
   lines.json): nearest apex line, frozen pair metrics, and the signed offset of the human line
   (start / mid / end) split into the component along apex -> virtual sharp (positive = convex
   side, toward the outer virtual sharp; negative = concave side, toward the inner skin and arc
   centre) and the lateral component.
4. Agreement with the single-agent triage on the 6 Gold11 scans it covered.
5. The gold11_relabel_v1 decision per scan (AI relabel iff both verifiers passed).

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/build_audit_results.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g11a_common as C  # noqa: E402

import score as sc  # noqa: E402  (research/broad_baseline/score.py, read-only)
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402

VERDICTS = ("OK", "CONVENTION_OFFSET", "REAL_ERROR", "NOT_A_BEND", "UNCLEAR")
NEAR_ANGLE_DEG = 15.0   # same "nearest counterpart" window as score_broad.diagnose
NEAR_OFFSET_MM = 15.0


# ---------------------------------------------------------------- transcripts

def _transcript_outputs(path: Path) -> tuple[dict | None, str | None, int]:
    last_so, last_text, n_so = None, None, 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            content = (rec.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            texts = [p.get("text") for p in content if p.get("type") == "text" and p.get("text")]
            for part in content:
                if part.get("type") == "tool_use" and part.get("name") == "StructuredOutput":
                    last_so, n_so = part.get("input"), n_so + 1
            if texts:
                last_text = "\n".join(texts)
    return last_so, last_text, n_so


def extract_agents() -> dict[str, dict]:
    agents: dict[str, dict] = {}
    for meta_path in sorted(C.TRANSCRIPTS.glob("*.meta.json")):
        meta = C.read_json(meta_path)
        label = meta.get("description")
        agent_id = meta_path.name.removesuffix(".meta.json")
        transcript = meta_path.with_name(agent_id + ".jsonl")
        so, text, n_so = _transcript_outputs(transcript)
        agents[label] = {"agent_id": agent_id, "phase": meta.get("workflowPhase"),
                         "transcript": str(transcript), "structured_output": so, "final_text": text,
                         "structured_output_calls": n_so}
    return agents


def auditor_output(agents: dict[str, dict], k: int, scan: str) -> dict:
    label = f"aud{k}:{scan}"
    if label in agents:
        out = agents[label]["structured_output"]
        return {"source": "live", "agent_id": agents[label]["agent_id"], "output": out}
    cache = C.CACHE_DIR / f"aud{k}_{scan}.json"
    out = C.read_json(cache)
    echo = (agents.get("load:" + label) or {}).get("structured_output")
    return {"source": "reused_from_killed_run_cache", "cache_file": str(cache),
            "load_agent_id": (agents.get("load:" + label) or {}).get("agent_id"),
            "load_echo_identical": echo == out, "output": out}


# ------------------------------------------------------------------ geometry

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def mechanical(human: dict, bends: list[dict]) -> dict:
    """Nearest relabel apex line to one human line, with a signed offset decomposition."""
    g = np.array([human["line_start"], human["line_end"]], dtype=float)
    best = None
    for b in bends:
        a = np.array([b["label_line"]["start"], b["label_line"]["end"]], dtype=float)
        m = pair_metrics(g, a)
        if m["axis_angle_deg"] <= NEAR_ANGLE_DEG and m["transverse_offset_mm"] <= NEAR_OFFSET_MM:
            cost = (m["axis_angle_deg"] / 5.0 + m["transverse_offset_mm"] / 4.0
                    + (1.0 - min(m["gold_span_overlap"], m["prediction_span_overlap"])))
            if best is None or cost < best[0]:
                best = (cost, b, m)
    length = float(np.linalg.norm(g[1] - g[0]))
    if best is None:
        return {"nearest_apex_line": None, "human_length_mm": C.r(length, 1),
                "note": f"no relabel apex line within {NEAR_ANGLE_DEG:.0f} deg / {NEAR_OFFSET_MM:.0f} mm"}
    _, b, m = best
    a0 = np.array(b["label_line"]["start"], dtype=float)
    a1 = np.array(b["label_line"]["end"], dtype=float)
    d = _unit(a1 - a0)
    sharp_mid = (np.array(b["sharp"]["start"]) + np.array(b["sharp"]["end"])) / 2.0
    us = sharp_mid - (a0 + a1) / 2.0
    us = us - d * us.dot(d)
    sharp_dist = float(np.linalg.norm(us))
    us = us / sharp_dist
    w = np.cross(d, us)

    def split(p: np.ndarray) -> dict:
        v = p - a0
        v = v - d * v.dot(d)
        return {"toward_sharp_mm": C.r(v.dot(us), 2), "lateral_mm": C.r(v.dot(w), 2),
                "perp_mm": C.r(np.linalg.norm(v), 2)}

    ends = [split(g[0]), split(g.mean(axis=0)), split(g[1])]
    perps = [e["perp_mm"] for e in (ends[0], ends[2])]
    return {
        "nearest_apex_line": b["bend_id"],
        "frozen_pair_metrics": {k: C.r(v, 3) for k, v in m.items()},
        "frozen_match_if_engine_drew_apex_line": bool(m["axis_angle_deg"] <= 5.0 and m["transverse_offset_mm"] <= 4.0
                                                      and m["gold_span_overlap"] >= 0.6
                                                      and m["prediction_span_overlap"] >= 0.6),
        "offset_start_mid_end": ends,
        "end_to_end_offset_spread_mm": C.r(abs(perps[0] - perps[1]), 2),
        "side": ("concave" if ends[1]["toward_sharp_mm"] < -0.25 else
                 "convex" if ends[1]["toward_sharp_mm"] > 0.25 else "on_apex"),
        "apex_to_virtual_sharp_mm": C.r(sharp_dist, 2),
        "relabel_bend_angle_deg": C.r(b.get("bend_angle_deg"), 1),
        "human_length_mm": C.r(length, 1),
        "apex_line_length_mm": C.r(b.get("length_mm"), 1),
    }


# ---------------------------------------------------------------- provenance

PROVISIONAL_SCRIPT = C.ROOT / "scripts" / "adjudicate_remaining_verified_bends.py"
# How that script builds each case (read from its source, never run).
PROVISIONAL_METHOD = {
    "01_10839000": "F1 zero-shot decomposition owned_bend_regions OB1-OB8: OB4 extended over OB2's span, "
                   "corner fragments OB5-OB8 paired into 4 horizontal axes",
    "02_10839000": "F1 zero-shot decomposition OB1-OB4: OB1+OB4 and OB2+OB3 joined by PCA-x extremes, "
                   "2 lines added to close a rectangle",
    "05_37361005": "F1 zero-shot decomposition OB1-OB4 span endpoints, verbatim",
    "06_38172000": "plane-lattice intersection lines (PLI_*) plus a copy of each outer line shifted +14.000 mm",
    "07_38172000": "F1 decomposition OB1/OB2/OB6 verbatim; OB4/OB5 extended and split +-7 mm; "
                   "OB7/OB8 upper endpoints joined",
    "08_40500000": "plane-lattice intersection lines plus 4 hand-entered vertical 'wall-end corner folds'",
    "10_49001000": "hand-entered: STEP bend-cylinder axes aligned to the scan and offset onto the observed bend surfaces",
    "11_49024000": "hand-entered: STEP bend-cylinder axes rigidly aligned, then shifted to the inner-surface arc midlines",
}
PROVISIONAL_DIR = C.ROOT / "output" / "manual-bend-label-review" / "provisional"
CATALOG = C.ROOT / "data" / "bend_corpus" / "annotations" / "bend_line_gold_catalog_v1.json"


def provenance_check(case_list: dict) -> dict:
    """Compare every Gold11 catalog line with the drafts of the untracked provisional script.

    The script (read, never run) builds axes from F1 detector outputs (zero-shot decomposition
    owned_bend_regions, plane-lattice intersections) plus rule edits and writes them with
    source "human_two_point" into output/manual-bend-label-review/provisional/<case>.json.
    Only the 8 Gold11 corpus cases that script lists are compared (13_ysherman is not Gold11).
    """
    catalog = C.read_json(CATALOG)
    rows = {r["scan_sha256"]: r for r in catalog["cases"]}
    out = {"script": C.rel(PROVISIONAL_SCRIPT), "script_exists": PROVISIONAL_SCRIPT.is_file(),
           "script_git_status": "untracked (git status ??)", "drafts_dir": C.rel(PROVISIONAL_DIR),
           "catalog": C.rel(CATALOG), "catalog_sha256": C.bb.sha256_file(CATALOG), "per_scan": {}}
    for entry in case_list["cases"]:
        if entry["tier"] != "GOLD11":
            continue
        row = rows[entry["scan_sha256"]]
        case_key = row["case_id"].split("/", 1)[-1]
        draft_path = PROVISIONAL_DIR / f"{case_key}.json"
        record = {"catalog_case_id": row["case_id"], "truth_tier": row.get("truth_tier"),
                  "review_status": row.get("review_status"), "catalog_lines": len(row["bends"])}
        if not draft_path.is_file():
            record["provisional_draft"] = None
            out["per_scan"][entry["case_id"]] = record
            continue
        draft = C.read_json(draft_path)
        drafts = [(b["bend_id"], np.array(b["line_start"], float), np.array(b["line_end"], float), b.get("source"))
                  for b in draft["manual_bends"]]
        diffs = []
        for b in row["bends"]:
            ends = C.bb.endpoints(b)
            best = min(((min(max(np.linalg.norm(ends[0] - s), np.linalg.norm(ends[1] - e)),
                             max(np.linalg.norm(ends[0] - e), np.linalg.norm(ends[1] - s))), did, src)
                        for did, s, e, src in drafts), key=lambda x: x[0])
            diffs.append({"catalog_bend": b.get("bend_id"), "draft_bend": best[1], "draft_source_tag": best[2],
                          "max_endpoint_diff_mm": float(best[0])})
        record.update({"provisional_draft": C.rel(draft_path), "draft_lines": len(drafts),
                       "script_method": PROVISIONAL_METHOD.get(case_key),
                       "max_endpoint_diff_mm": max(d["max_endpoint_diff_mm"] for d in diffs),
                       "identical": len(drafts) == len(row["bends"]) and max(d["max_endpoint_diff_mm"] for d in diffs) == 0.0,
                       "lines": diffs})
        out["per_scan"][entry["case_id"]] = record
    ident = [k for k, v in out["per_scan"].items() if v.get("identical")]
    out["scans_identical_to_provisional_script_output"] = ident
    out["lines_identical_to_provisional_script_output"] = sum(out["per_scan"][k]["catalog_lines"] for k in ident)
    return out


def step_count_check(case_list: dict, scans_out: list[dict]) -> dict:
    """Bend COUNT per scan vs the STEP v10 count of its family (broad baseline cad_crosscheck_raw).

    Counts only, no positions; only meaningful on full (non-staged) scans."""
    by_case = {c["case_id"]: c for c in case_list["cases"]}
    rows = {}
    for s in scans_out:
        entry = by_case[s["scan_id"]]
        raw = C.BB / "cad_crosscheck_raw" / f"{entry['family_id']}.json"
        cad = C.read_json(raw) if raw.is_file() else {}
        rows[s["scan_id"]] = {"family_id": entry["family_id"], "staged": bool(entry.get("staged_or_incomplete")),
                              "stage_hints": entry.get("stage_hints"),
                              "step_v10_bend_count": cad.get("v10_bend_count"),
                              "step_lenient_bend_count": cad.get("lenient_bend_count"),
                              "human_lines": s["human_line_count"], "relabel_lines": s["relabel"]["line_count"]}
    full = {k: v for k, v in rows.items() if not v["staged"]}
    return {"source": C.rel(C.BB / "cad_crosscheck_raw"), "per_scan": rows,
            "full_scans_relabel_equals_step_v10_or_lenient": sorted(
                k for k, v in full.items() if v["relabel_lines"] in (v["step_v10_bend_count"], v["step_lenient_bend_count"])),
            "full_scans_human_equals_step_v10_or_lenient": sorted(
                k for k, v in full.items() if v["human_lines"] in (v["step_v10_bend_count"], v["step_lenient_bend_count"])),
            "full_scans": sorted(full)}


# ------------------------------------------------------------------ majority

def majority(values: list[str]) -> tuple[str, str]:
    counts = Counter(values)
    top, n = counts.most_common(1)[0]
    if n == 3:
        return top, "unanimous"
    if n == 2:
        return top, "2_of_3"
    return "SPLIT", "no_majority"


def count_vector(lines: list[dict]) -> dict[str, int]:
    return dict(Counter(l["verdict"] for l in lines))


def main() -> int:
    agents = extract_agents()
    case_list = C.read_json(C.CASE_LIST)
    frames = C.read_json(C.FRAME_RECORDS)
    by_case = {c["case_id"]: c for c in case_list["cases"]}

    scans_out = []
    compact_checks = {}
    for scan in C.SCANS:
        entry = by_case[scan]
        gold = sc.load_gold(entry["gold"], frame_records=frames)
        human_lines = gold["lines"]
        human_ids = [h["bend_id"] for h in human_lines]
        auditors = [auditor_output(agents, k, scan) for k in (1, 2, 3)]
        outs = [a["output"] for a in auditors]

        # check against the workflow's compact record
        exp_pass, exp_aud = C.EXPECTED_COMPACT[scan]
        got_aud = [(count_vector(o["lines"]), len(o["missing_bends"]),
                    [l.get("offset_mm") for l in o["lines"] if l["verdict"] == "CONVENTION_OFFSET"]) for o in outs]
        checks = [agents.get(f"chk{k}:{scan}", {}).get("structured_output") for k in (1, 2)]
        got_pass = len([c for c in checks if c]) == 2 and all(c["pass"] and c["images_seen"] > 0 for c in checks)
        compact_checks[scan] = {"auditors_match": got_aud == [(a, m, o) for a, m, o in exp_aud],
                                "relabel_pass_match": got_pass == exp_pass}

        relabel_lines_doc = C.read_json(C.AUDIT / "relabel" / scan / "lines.json")
        bends = [b for b in relabel_lines_doc["bends"] if b.get("status") == "ok"]
        label_path = C.RELABEL_ROOT / f"{C.RELABEL_ASSET[scan]}.json"
        label = C.read_json(label_path)
        label_vs_lines = (len(label["bends"]) == len(bends) and all(
            np.allclose(lb["line_start"], b["label_line"]["start"], atol=1e-3)
            and np.allclose(lb["line_end"], b["label_line"]["end"], atol=1e-3)
            for lb, b in zip(label["bends"], bends)))

        per_line = []
        for h in human_lines:
            hid = h["bend_id"]
            votes = []
            for k, o in enumerate(outs, start=1):
                row = next((l for l in o["lines"] if l["gold_id"] == hid), None)
                votes.append({"auditor": k, "verdict": row["verdict"] if row else "MISSING_FROM_OUTPUT",
                              "offset_mm": row.get("offset_mm") if row else None,
                              "coverage_pct": row.get("coverage_pct") if row else None})
            verdict, agreement = majority([v["verdict"] for v in votes])
            offs = [v["offset_mm"] for v in votes if isinstance(v["offset_mm"], (int, float))]
            mech = mechanical(h, bends)
            per_line.append({
                "human_id": hid, "majority_verdict": verdict, "agreement": agreement, "votes": votes,
                "auditor_median_offset_from_outer_apex_mm": C.r(statistics.median(offs), 2) if offs else None,
                "mechanical_vs_relabel_apex": mech,
            })
        nearest_used = {p["mechanical_vs_relabel_apex"].get("nearest_apex_line") for p in per_line}
        relabel_unclaimed = [b["bend_id"] for b in bends if b["bend_id"] not in nearest_used]
        unknown_ids = sorted({l["gold_id"] for o in outs for l in o["lines"]} - set(human_ids))
        majority_counts = dict(Counter(p["majority_verdict"] for p in per_line))
        missing_counts = [len(o["missing_bends"]) for o in outs]
        missing_majority = sorted(missing_counts)[1]
        vectors = [json.dumps(count_vector(o["lines"]), sort_keys=True) for o in outs]
        vector_agreement = max(Counter(vectors).values())

        triage = C.TRIAGE_GOLD11.get(scan)
        triage_cmp = None
        if triage:
            rows = []
            for p in per_line:
                triage_wrong = p["human_id"] in triage["wrong"]
                rows.append({"human_id": p["human_id"], "triage": "WRONG" if triage_wrong else "NOT_FLAGGED",
                             "auditor_majority": p["majority_verdict"]})
            confirmed = sum(1 for x in rows if x["triage"] == "WRONG"
                            and x["auditor_majority"] in ("REAL_ERROR", "NOT_A_BEND"))
            reclassified_convention = sum(1 for x in rows if x["triage"] == "WRONG"
                                          and x["auditor_majority"] == "CONVENTION_OFFSET")
            overturned_ok = sum(1 for x in rows if x["triage"] == "WRONG" and x["auditor_majority"] == "OK")
            added = sum(1 for x in rows if x["triage"] == "NOT_FLAGGED"
                        and x["auditor_majority"] in ("REAL_ERROR", "NOT_A_BEND", "CONVENTION_OFFSET"))
            triage_cmp = {"triage_wrong_lines": len(triage["wrong"]),
                          "triage_omitted_real_bends": triage["omitted_real_bends"],
                          "triage_omission_note": triage["omission_note"],
                          "auditor_majority_missing_bends": missing_majority,
                          "triage_wrong_confirmed_as_real_error_or_not_a_bend": confirmed,
                          "triage_wrong_reclassified_convention_offset": reclassified_convention,
                          "triage_wrong_overturned_to_ok": overturned_ok,
                          "not_flagged_by_triage_but_auditor_majority_non_ok": added,
                          "per_line": rows}

        chk_rows = []
        for k, c in enumerate(checks, start=1):
            a = agents.get(f"chk{k}:{scan}", {})
            chk_rows.append({"verifier": k, "agent_id": a.get("agent_id"), **(c or {"missing": True})})
        relabel_agent = agents.get(f"relabel:{scan}", {})
        scans_out.append({
            "scan_id": scan,
            "gold11_id": entry.get("gold11_id"),
            "family_id": entry.get("family_id"),
            "scan_name": Path(entry["scan_path"]).name,
            "gold_tier_note": ("Codex multiview gold, not human" if "GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN"
                               in (entry.get("flags") or []) else "human two-point gold"),
            "human_line_count": len(human_lines),
            "human_gold_frame": gold.get("frame_transform"),
            "defective_by_workflow_rule": sum(
                1 for o in outs if o["missing_bends"] or any(l["verdict"] in ("REAL_ERROR", "NOT_A_BEND",
                                                                                 "CONVENTION_OFFSET")
                                                              for l in o["lines"])) >= 2,
            "majority": {"line_verdict_counts": majority_counts,
                         "missing_bends": missing_majority,
                         "missing_bends_by_auditor": missing_counts,
                         "auditors_with_identical_count_vector": vector_agreement,
                         "split_lines": [p["human_id"] for p in per_line if p["majority_verdict"] == "SPLIT"]},
            "per_line": per_line,
            "auditor_ids_not_in_gold": unknown_ids,
            "auditors": [{"auditor": k, **a} for k, a in enumerate(auditors, start=1)],
            "relabel": {"agent_id": relabel_agent.get("agent_id"),
                        "report": relabel_agent.get("final_text"),
                        "lines_json": C.rel(C.AUDIT / "relabel" / scan / "lines.json"),
                        "lines_json_sha256": C.bb.sha256_file(C.AUDIT / "relabel" / scan / "lines.json"),
                        "label": C.rel(label_path), "label_sha256": C.bb.sha256_file(label_path),
                        "label_revision": label.get("revision"), "label_annotator": label.get("annotator"),
                        "label_matches_lines_json": bool(label_vs_lines),
                        "line_count": len(bends),
                        "lines_not_nearest_to_any_human_line": relabel_unclaimed},
            "relabel_verifiers": chk_rows,
            "relabel_pass": got_pass,
            "gold11_relabel_v1_source": "AI_RELABEL" if got_pass else "HUMAN_GOLD",
            "triage_comparison": triage_cmp,
        })

    # ---------------------------------------------------------------- roll-up
    tot = Counter()
    for s in scans_out:
        tot.update(s["majority"]["line_verdict_counts"])
    lines_all = [p for s in scans_out for p in s["per_line"]]

    def offsets(pred):
        return sorted(p["auditor_median_offset_from_outer_apex_mm"] for p in lines_all
                      if pred(p) and p["auditor_median_offset_from_outer_apex_mm"] is not None)

    def mech_mid(pred):
        out = []
        for p in lines_all:
            m = p["mechanical_vs_relabel_apex"]
            if pred(p) and m.get("nearest_apex_line"):
                out.append(m["offset_start_mid_end"][1]["toward_sharp_mm"])
        return sorted(out)

    def dist(values):
        if not values:
            return {"n": 0}
        q = np.percentile(values, [25, 50, 75])
        return {"n": len(values), "min": C.r(min(values), 2), "p25": C.r(q[0], 2), "median": C.r(q[1], 2),
                "p75": C.r(q[2], 2), "max": C.r(max(values), 2),
                "over_4mm": sum(abs(v) > 4.0 for v in values)}

    conv = lambda p: p["majority_verdict"] == "CONVENTION_OFFSET"  # noqa: E731
    real = lambda p: p["majority_verdict"] == "REAL_ERROR"  # noqa: E731
    ok = lambda p: p["majority_verdict"] == "OK"  # noqa: E731
    sides = Counter((p["majority_verdict"], p["mechanical_vs_relabel_apex"].get("side", "no_near_apex_line"))
                    for p in lines_all)
    triaged = [s for s in scans_out if s["triage_comparison"]]
    agreement_unanimous = sum(1 for p in lines_all if p["agreement"] == "unanimous")
    agreement_2of3 = sum(1 for p in lines_all if p["agreement"] == "2_of_3")

    payload = {
        "schema": "sherman.gold11_audit.audit_results.v1",
        "created_at": C.bb.utc_now(),
        "workflow_run": C.WORKFLOW_RUN,
        "workflow_script": "scratchpad/cont/gold11_audit_continue.js (sherman-gold11-label-audit-continue)",
        "not_redone": ("This roll-up re-ran no auditor, relabel, verifier, prep, lines, write-label or engine "
                       "step. Agent outputs are read verbatim from the run's transcripts; the 5 auditor results "
                       "the run re-used from the killed run wf_9c8183ed-61a are read from its cache."),
        "question": ("Are the Gold11 label errors claimed by the single-agent triage (27/49 lines wrong, 11 missing "
                     "bends on 6 scans) real, or a line-locus convention difference?"),
        "verdict_definitions": {
            "OK": "on the bend, within the 4 mm gate, extent fine",
            "CONVENTION_OFFSET": "parallel, full length, consistently offset from the outer apex (auditor judgement)",
            "REAL_ERROR": "right bend but wrong place or wrong extent",
            "NOT_A_BEND": "free edge, seam, flat face: no formed fold there (a REAL_ERROR subtype)",
            "MISSING_BEND": "formed fold with no gold line (counted per auditor in missing_bends)",
        },
        "conventions_owner_direction": ["joggle = 2 bends", "small formed bends (15 deg tabs) count",
                                        "a fold interrupted by windows/reliefs, same two plates, collinear = ONE line",
                                        "line locus = outer-skin arc midline (bend apex)"],
        "mechanical_sign_convention": ("toward_sharp_mm > 0: convex side (toward the outer virtual sharp, free "
                                       "air outside the part); < 0: concave side (inner skin at about -t, arc "
                                       "centre at about -(Ri+t)). side=on_apex if |mid| <= 0.25 mm."),
        "compact_record_check": {"per_scan": compact_checks,
                                 "all_match": all(v["auditors_match"] and v["relabel_pass_match"]
                                                  for v in compact_checks.values())},
        "provenance_check": provenance_check(case_list),
        "step_count_check": step_count_check(case_list, scans_out),
        "rollup": {
            "scans": len(scans_out),
            "human_lines": len(lines_all),
            "majority_line_verdicts": dict(tot),
            "majority_missing_bends": sum(s["majority"]["missing_bends"] for s in scans_out),
            "line_agreement": {"unanimous": agreement_unanimous, "2_of_3": agreement_2of3,
                               "split": len(lines_all) - agreement_unanimous - agreement_2of3},
            "scans_defective_by_workflow_rule": sum(s["defective_by_workflow_rule"] for s in scans_out),
            "scans_with_any_majority_non_ok_line_or_missing_bend": sum(
                1 for s in scans_out if s["majority"]["missing_bends"]
                or any(v != "OK" for v in [p["majority_verdict"] for p in s["per_line"]])),
            "relabel_passed_both_verifiers": [s["scan_id"] for s in scans_out if s["relabel_pass"]],
            "relabel_failed": [s["scan_id"] for s in scans_out if not s["relabel_pass"]],
            "auditor_offset_from_outer_apex_mm": {"CONVENTION_OFFSET": dist(offsets(conv)),
                                                   "REAL_ERROR": dist(offsets(real)),
                                                   "OK": dist(offsets(ok))},
            "mechanical_mid_toward_sharp_mm": {"CONVENTION_OFFSET": dist(mech_mid(conv)),
                                               "REAL_ERROR": dist(mech_mid(real)),
                                               "OK": dist(mech_mid(ok))},
            "mechanical_side_by_majority_verdict": {f"{v}|{s}": n for (v, s), n in sorted(sides.items())},
            "triage_agreement_6_scans": {
                "triage_wrong_lines": sum(s["triage_comparison"]["triage_wrong_lines"] for s in triaged),
                "triage_lines_total": sum(s["human_line_count"] for s in triaged),
                "triage_omitted_real_bends": sum(s["triage_comparison"]["triage_omitted_real_bends"] for s in triaged),
                "confirmed_real_error_or_not_a_bend": sum(
                    s["triage_comparison"]["triage_wrong_confirmed_as_real_error_or_not_a_bend"] for s in triaged),
                "reclassified_convention_offset": sum(
                    s["triage_comparison"]["triage_wrong_reclassified_convention_offset"] for s in triaged),
                "overturned_to_ok": sum(s["triage_comparison"]["triage_wrong_overturned_to_ok"] for s in triaged),
                "extra_non_ok_lines_found_by_auditors": sum(
                    s["triage_comparison"]["not_flagged_by_triage_but_auditor_majority_non_ok"] for s in triaged),
                "auditor_majority_missing_bends": sum(
                    s["triage_comparison"]["auditor_majority_missing_bends"] for s in triaged),
            },
        },
        "scans": scans_out,
    }
    out = C.write_json(C.AUDIT / "audit_results.json", payload)
    print("wrote", out)
    print(json.dumps(payload["compact_record_check"], indent=1))
    print(json.dumps(payload["rollup"], indent=1))
    return 0 if payload["compact_record_check"]["all_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
