#!/usr/bin/env python3
"""Measure the frozen physical15 V51 pool recall ceiling on consumed Gold12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


def analyze(public: dict[str, Any], catalog: dict[str, Any], execution_root: Path) -> dict[str, Any]:
    gold_by_sha = {str(row["scan_sha256"]): row for row in catalog["cases"]}
    thresholds = Thresholds()
    rows = []
    source_hits: dict[str, int] = {}
    for public_row in public["cases"]:
        gold = gold_by_sha.get(str(public_row["source_sha256"]))
        if gold is None or gold.get("truth_tier") != "codex_manual_multiview_count_backed":
            continue
        case_id = str(public_row["opaque_case_id"])
        case_root = execution_root / "primary" / case_id
        pool = json.loads((case_root / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        selected_payload = json.loads((case_root / "semantic_result.json").read_text(encoding="utf-8"))
        candidates = list(pool.get("line_objects") or [])
        ceiling = match_lines(gold["bends"], candidates, thresholds)
        selected = match_lines(gold["bends"], selected_payload["predictions"], thresholds)
        by_bend = {str(candidate["bend_id"]): candidate for candidate in candidates}
        match_sources = []
        for match in ceiling["matches"]:
            candidate = by_bend[str(match["prediction_id"])]
            source = str(candidate.get("representative_source") or candidate.get("candidate_source") or "unknown")
            source_hits[source] = source_hits.get(source, 0) + 1
            match_sources.append({"gold_id":match["gold_id"],"candidate_bend_id":match["prediction_id"],"representative_source":source,"axis_angle_deg":match["axis_angle_deg"],"transverse_offset_mm":match["transverse_offset_mm"],"shorter_span_overlap":match["shorter_span_overlap"]})
        rows.append({
            "case_id":case_id,
            "family_id":str(gold["part_id"]),
            "gold_count":ceiling["gold_count"],
            "pool_line_object_count":len(candidates),
            "candidate_ceiling_true_positive":ceiling["true_positive"],
            "candidate_ceiling_false_negative":ceiling["false_negative"],
            "candidate_ceiling_recall":ceiling["true_positive"]/ceiling["gold_count"] if ceiling["gold_count"] else 0.0,
            "selected_true_positive":selected["true_positive"],
            "selected_recall":selected["true_positive"]/selected["gold_count"] if selected["gold_count"] else 0.0,
            "selection_recall_loss":ceiling["true_positive"]-selected["true_positive"],
            "candidate_generation_complete":ceiling["false_negative"]==0,
            "unborn_gold_ids":ceiling["unmatched_gold_ids"],
            "candidate_matches":match_sources,
        })
    if len(rows) != 12:
        raise ValueError(f"expected 12 consumed count-backed physical cases, got {len(rows)}")
    gold_count = sum(row["gold_count"] for row in rows)
    ceiling_tp = sum(row["candidate_ceiling_true_positive"] for row in rows)
    selected_tp = sum(row["selected_true_positive"] for row in rows)
    return {
        "schema":"SHERMAN_QC_PHYSICAL15_CANDIDATE_CEILING_V1",
        "evidence_class":"DEVELOPMENT_CONSUMED_PHYSICAL",
        "promotion_eligible":False,
        "model_changed":False,
        "truth_opened_after_frozen_execution":True,
        "thresholds":{"axis_angle_deg":thresholds.angle_deg,"transverse_offset_mm":thresholds.transverse_offset_mm,"gold_span_overlap":thresholds.gold_span_overlap,"prediction_span_overlap":thresholds.prediction_span_overlap},
        "aggregate":{
            "case_count":len(rows),
            "gold_bend_count":gold_count,
            "pool_line_object_count":sum(row["pool_line_object_count"] for row in rows),
            "candidate_ceiling_true_positive":ceiling_tp,
            "candidate_ceiling_recall":ceiling_tp/gold_count,
            "selected_true_positive":selected_tp,
            "selected_recall":selected_tp/gold_count,
            "selection_recall_loss":ceiling_tp-selected_tp,
            "cases_with_complete_candidate_generation":sum(row["candidate_generation_complete"] for row in rows),
            "cases_with_zero_selected_recall":sum(row["selected_true_positive"]==0 for row in rows),
            "cases_with_zero_candidate_ceiling_recall":sum(row["candidate_ceiling_true_positive"]==0 for row in rows),
        },
        "matched_candidate_source_counts":dict(sorted(source_hits.items())),
        "cases":rows,
    }


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--public-manifest",type=Path,required=True);parser.add_argument("--gold-catalog",type=Path,required=True);parser.add_argument("--execution-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    report=analyze(json.loads(args.public_manifest.read_text()),json.loads(args.gold_catalog.read_text()),args.execution_root)
    if args.output.exists(): raise FileExistsError(f"output exists: {args.output}")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print(json.dumps(report["aggregate"],sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
