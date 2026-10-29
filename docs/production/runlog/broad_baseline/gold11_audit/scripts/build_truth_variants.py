#!/usr/bin/env python3
"""Build the Gold11 truth variants and the case lists the rescoring uses (gold11_audit/truth/).

Nothing under data/bend_corpus/annotations/human_bend_lines_v1 or the Gold11 catalog is written;
the human lines are only read through score.load_gold (raw-scan frame, frozen frame records).

Variants (all 11 Gold11 scans; tier stays GOLD11 so the scorer's cohorts are unchanged):
  human                     the published human Gold11 (catalog), unchanged.
  gold11_relabel_v1         TRUTH VARIANT: AI relabel where both relabel verifiers passed, else human.
  all11_ai_relabel          sensitivity only: AI relabel on all 11 (incl. the one that failed a verifier).
  human_snapped_apex        locus diagnostic only: every human line whose nearest relabel apex line is
                            within 5 deg and 8 mm (midpoint transverse) is projected onto that apex line
                            (extent kept); all other human lines are left where they are.
  human_snapped_apex_conv   locus diagnostic only: as above, but only lines the auditor majority called
                            OK or CONVENTION_OFFSET are snapped.
and broad57_gold11_relabel_v1: the full broad case list (case_list_broad_v3) with gold11_relabel_v1
substituted for the 11 Gold11 entries.

The 8 mm snap window is 2x the frozen 4 mm gate and falls in the empty gap of the measured offsets
(largest parallel offset to a same-fold apex line 7.55 mm, next 8.90 mm = a free edge).

Run: OMP_NUM_THREADS=1 .venv/bin/python -B docs/production/runlog/broad_baseline/gold11_audit/scripts/build_truth_variants.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g11a_common as C  # noqa: E402

import score as sc  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import pair_metrics  # noqa: E402

TRUTH = C.AUDIT / "truth"
SNAP_ANGLE_DEG = 5.0
SNAP_OFFSET_MM = 8.0


def case_list_sha(cases: list[dict]) -> str:
    """Same definition as research/broad_baseline/build_case_list.py."""
    identity = [{k: c.get(k) for k in ("case_id", "scan_sha256", "gold", "gold_file_sha256", "label_complete",
                                        "zero_bend_control", "cad_sha256")} for c in cases]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_case_list(name: str, cases: list[dict], base: dict, note: str) -> dict:
    payload = {"schema": base["schema"], "created_at": C.bb.utc_now(),
               "case_list_sha256": case_list_sha(cases),
               "case_list_sha256_definition": base["case_list_sha256_definition"],
               "derived_from": {"path": C.rel(C.CASE_LIST), "case_list_sha256": base["case_list_sha256"],
                                "file_sha256": C.bb.sha256_file(C.CASE_LIST)},
               "variant_note": note, "custody": base.get("custody"),
               "summary": {"cases": len(cases),
                           "line_cases": sum(not c["zero_bend_control"] for c in cases),
                           "gold_lines_total": sum(int(c.get("gold_line_count") or 0) for c in cases
                                                   if not c["zero_bend_control"])},
               "cases": cases}
    path = C.write_json(TRUTH / f"{name}.json", payload)
    return {"path": C.rel(path), "case_list_sha256": payload["case_list_sha256"], "cases": len(cases),
            "gold_lines_total": payload["summary"]["gold_lines_total"]}


def ai_entry(entry: dict, scan: str, variant: str) -> dict:
    e = copy.deepcopy(entry)
    path = C.RELABEL_ROOT / f"{C.RELABEL_ASSET[scan]}.json"
    label = C.read_json(path)
    prov = C.read_json(path.parent / "provenance" / path.name)
    assert prov["asset_sha256"] == entry["scan_sha256"], scan
    assert label["coordinate_frame"] == "source_ply_raw", scan
    e["gold"] = {"kind": "ai_studio", "path": C.rel(path)}
    e["gold_file_sha256"] = C.bb.sha256_file(path)
    e["gold_line_count"] = len(label["bends"])
    e["label_frame"] = "source_ply_raw"
    e["gold_tier_detail"] = "ai_relabel_gold11_v1 (claude-vision-ai-v1, ai_adjudicated, 2 verifiers)"
    e["gold11_truth_variant"] = {"variant": variant, "source": "AI_RELABEL",
                                 "replaces": entry["gold"], "replaced_gold_file_sha256": entry["gold_file_sha256"],
                                 "replaced_gold_line_count": entry["gold_line_count"]}
    return e


def human_entry(entry: dict, variant: str, why: str) -> dict:
    e = copy.deepcopy(entry)
    e["gold11_truth_variant"] = {"variant": variant, "source": "HUMAN_GOLD", "why": why}
    return e


def snap_lines(human_lines: list[dict], bends: list[dict], allowed: set[str] | None,
               verdicts: dict[str, str]) -> tuple[list[dict], list[dict]]:
    out, log = [], []
    for h in human_lines:
        g = np.array([h["line_start"], h["line_end"]], dtype=float)
        best = None
        for b in bends:
            a = np.array([b["label_line"]["start"], b["label_line"]["end"]], dtype=float)
            m = pair_metrics(g, a)
            if m["axis_angle_deg"] <= 15.0 and m["transverse_offset_mm"] <= 15.0:
                cost = (m["axis_angle_deg"] / 5.0 + m["transverse_offset_mm"] / 4.0
                        + (1.0 - min(m["gold_span_overlap"], m["prediction_span_overlap"])))
                if best is None or cost < best[0]:
                    best = (cost, b, m, a)
        verdict = verdicts.get(h["bend_id"])
        row = {"human_id": h["bend_id"], "majority_verdict": verdict}
        if best is None:
            out.append(dict(h))
            log.append({**row, "snapped": False, "why": "no apex line within 15 deg / 15 mm"})
            continue
        _, b, m, a = best
        row.update({"nearest_apex_line": b["bend_id"], "pre_snap_offset_mm": C.r(m["transverse_offset_mm"], 3),
                    "pre_snap_angle_deg": C.r(m["axis_angle_deg"], 3)})
        in_window = m["axis_angle_deg"] <= SNAP_ANGLE_DEG and m["transverse_offset_mm"] <= SNAP_OFFSET_MM
        if not in_window or (allowed is not None and verdict not in allowed):
            out.append(dict(h))
            log.append({**row, "snapped": False,
                        "why": "outside the 5 deg / 8 mm window" if not in_window else f"verdict {verdict} not snapped"})
            continue
        d = (a[1] - a[0]) / np.linalg.norm(a[1] - a[0])
        snapped = [a[0] + d * float(np.dot(p - a[0], d)) for p in g]
        out.append(C.bb.as_line(h["bend_id"], snapped[0], snapped[1], snapped_to=b["bend_id"],
                                pre_snap_offset_mm=C.r(m["transverse_offset_mm"], 3)))
        log.append({**row, "snapped": True})
    return out, log


def main() -> int:
    base = C.read_json(C.CASE_LIST)
    frames = C.read_json(C.FRAME_RECORDS)
    audit = C.read_json(C.AUDIT / "audit_results.json")
    scans = {s["scan_id"]: s for s in audit["scans"]}
    entries = {c["case_id"]: c for c in base["cases"]}
    g11_ids = [c["case_id"] for c in base["cases"] if c["tier"] == "GOLD11"]
    assert sorted(g11_ids) == sorted(C.SCANS)

    decisions = {}
    for scan in g11_ids:
        s = scans[scan]
        decisions[scan] = {"gold11_id": s["gold11_id"], "relabel_pass": s["relabel_pass"],
                           "source": "AI_RELABEL" if s["relabel_pass"] else "HUMAN_GOLD",
                           "verifiers": [{"verifier": v["verifier"], "pass": v.get("pass"),
                                          "images_seen": v.get("images_seen"),
                                          "problems_head": (v.get("problems") or "")[:300]}
                                         for v in s["relabel_verifiers"]],
                           "human_lines": s["human_line_count"], "relabel_lines": s["relabel"]["line_count"],
                           "relabel_label": s["relabel"]["label"], "relabel_label_sha256": s["relabel"]["label_sha256"],
                           "human_gold": entries[scan]["gold"], "human_gold_file_sha256": entries[scan]["gold_file_sha256"]}

    # ---- Gold11-only case lists
    lists = {}
    human_cases = [human_entry(entries[i], "human", "published human Gold11") for i in g11_ids]
    lists["case_list_gold11__human"] = write_case_list(
        "case_list_gold11__human", human_cases, base, "11 Gold11 entries of case_list_broad_v3, gold unchanged")
    v1_cases = [ai_entry(entries[i], i, "gold11_relabel_v1") if decisions[i]["relabel_pass"]
                else human_entry(entries[i], "gold11_relabel_v1", "AI relabel failed a verifier: human gold kept")
                for i in g11_ids]
    lists["case_list_gold11__relabel_v1"] = write_case_list(
        "case_list_gold11__relabel_v1", v1_cases, base,
        "gold11_relabel_v1: AI relabel where both relabel verifiers passed (10), human gold otherwise (1)")
    all_ai = [ai_entry(entries[i], i, "all11_ai_relabel") for i in g11_ids]
    lists["case_list_gold11__all11_ai_relabel"] = write_case_list(
        "case_list_gold11__all11_ai_relabel", all_ai, base,
        "SENSITIVITY ONLY: AI relabel on all 11, including sc_07062eee17aa whose relabel failed both verifiers "
        "(B2 about 0.9 mm off the outer apex)")

    # ---- snapped human gold (diagnostic)
    snap_manifest = {"rule": {"targets": "AI relabel apex lines (gold11_audit/relabel/<scan>/lines.json, status ok, "
                                         "label_line); sc_07062eee17aa's unverified relabel is used as a target too",
                              "window": f"axis angle <= {SNAP_ANGLE_DEG} deg and midpoint transverse offset <= "
                                        f"{SNAP_OFFSET_MM} mm to the nearest apex line (nearest = score_broad cost)",
                              "projection": "both human endpoints projected orthogonally onto the apex line axis "
                                            "(direction and locus from the apex line, extent from the human line)",
                              "use": "locus-convention diagnostic only; never a truth variant"},
                     "variants": {}}
    for variant, allowed in (("human_snapped_apex", None), ("human_snapped_apex_conv", {"OK", "CONVENTION_OFFSET"})):
        cases, per_scan = [], {}
        for scan in g11_ids:
            entry = entries[scan]
            gold = sc.load_gold(entry["gold"], frame_records=frames)
            bends = [b for b in C.read_json(C.AUDIT / "relabel" / scan / "lines.json")["bends"]
                     if b.get("status") == "ok"]
            verdicts = {p["human_id"]: p["majority_verdict"] for p in scans[scan]["per_line"]}
            lines, log = snap_lines(gold["lines"], bends, allowed, verdicts)
            path = TRUTH / variant / f"{scan}.json"
            C.write_json(path, {"frame": "scan_raw", "scan_sha256": entry["scan_sha256"],
                                "variant": variant, "derived_from": entry["gold"],
                                "diagnostic_only": True, "bends": lines, "snap_log": log})
            e = copy.deepcopy(entry)
            e["gold"] = {"kind": "generic", "path": C.rel(path)}
            e["gold_file_sha256"] = C.bb.sha256_file(path)
            e["gold11_truth_variant"] = {"variant": variant, "source": "HUMAN_GOLD_SNAPPED_TO_APEX",
                                         "diagnostic_only": True}
            cases.append(e)
            per_scan[scan] = {"snapped": sum(r["snapped"] for r in log), "lines": len(log), "log": log}
        lists[f"case_list_gold11__{variant}"] = write_case_list(
            f"case_list_gold11__{variant}", cases, base, f"{variant}: locus diagnostic only (see snap rule)")
        snap_manifest["variants"][variant] = {
            "snap_only_verdicts": sorted(allowed) if allowed else "all",
            "snapped_lines": sum(v["snapped"] for v in per_scan.values()),
            "human_lines": sum(v["lines"] for v in per_scan.values()),
            "per_scan": per_scan}
    C.write_json(TRUTH / "human_snapped_apex_manifest.json", snap_manifest)

    # ---- full broad list with gold11_relabel_v1 substituted
    v1 = {c["case_id"]: c for c in v1_cases}
    broad = [v1.get(c["case_id"], c) for c in base["cases"]]
    lists["case_list_broad_v3__gold11_relabel_v1"] = write_case_list(
        "case_list_broad_v3__gold11_relabel_v1", broad, base,
        "case_list_broad_v3 with the 11 Gold11 gold specs replaced by gold11_relabel_v1 (10 AI relabels, 1 human); "
        "every non-Gold11 entry is byte-for-byte the v3 entry")

    override = {scan: (v1[scan]["gold"]) for scan in g11_ids}
    C.write_json(TRUTH / "gold11_relabel_v1_gold_override.json", override)
    manifest = {
        "schema": "sherman.gold11_audit.truth_variant.v1",
        "created_at": C.bb.utc_now(),
        "variant": "gold11_relabel_v1",
        "rule": "per Gold11 scan: the AI relabel (data/bend_corpus/annotations/ai_bend_lines_gold11_relabel_v1) "
                "if BOTH independent relabel verifiers passed it (pass=true, images_seen>0), else the human gold",
        "status": "PROPOSED - not for gate decisions until the owner signs off",
        "counts": {"ai_relabel": sum(d["source"] == "AI_RELABEL" for d in decisions.values()),
                   "human_gold": sum(d["source"] == "HUMAN_GOLD" for d in decisions.values()),
                   "gold_lines": sum(d["relabel_lines"] if d["source"] == "AI_RELABEL" else d["human_lines"]
                                     for d in decisions.values()),
                   "human_gold_lines_replaced": sum(d["human_lines"] for d in decisions.values()
                                                    if d["source"] == "AI_RELABEL")},
        "per_scan": decisions,
        "gold_override_map": C.rel(TRUTH / "gold11_relabel_v1_gold_override.json"),
        "case_lists": lists,
        "human_gold_untouched": {"catalog": C.rel(C.ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"),
                                 "catalog_sha256": C.bb.sha256_file(C.ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"),
                                 "expected_sha256": "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"},
    }
    C.write_json(TRUTH / "gold11_relabel_v1.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("per_scan",)}, indent=1))
    print({v: snap_manifest["variants"][v]["snapped_lines"] for v in snap_manifest["variants"]})
    assert manifest["human_gold_untouched"]["catalog_sha256"] == manifest["human_gold_untouched"]["expected_sha256"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
