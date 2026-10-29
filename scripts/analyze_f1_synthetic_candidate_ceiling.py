#!/usr/bin/env python3
"""Measure source-only candidate recall ceiling on the synthetic stress set.

This post-execution analysis opens generator truth. It is never callable from
the detector lane and is non-confirmatory by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    selected_score = json.loads(args.score.read_text(encoding="utf-8"))
    for payload in (truth, family_map, selected_score):
        if payload.get("token") != TOKEN:
            raise ValueError("synthetic token mismatch")

    truth_by_case = {str(row["case_id"]): row for row in truth["cases"]}
    selected_by_case = {str(row["case_id"]): row for row in selected_score["cases"]}
    family_for_case = {
        str(case_id): str(family_id)
        for family_id, case_ids in family_map["families"].items()
        for case_id in case_ids
    }
    thresholds = Thresholds(5.0, 4.0, 0.60, 0.60)
    rows = []
    for case_id in sorted(truth_by_case):
        pool_path = args.execution_root / "primary" / case_id / "pool" / f"{case_id}.json"
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        candidates = list(pool.get("line_objects") or [])
        ceiling = match_lines(truth_by_case[case_id]["manual_bends"], candidates, thresholds)
        candidate_by_bend_id = {str(row.get("bend_id")): row for row in candidates}
        match_details = []
        for match in ceiling["matches"]:
            candidate = candidate_by_bend_id[str(match["prediction_id"])]
            endpoints = candidate.get("span_endpoints") or []
            match_details.append(
                {
                    **match,
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "representative_source": str(candidate.get("representative_source") or ""),
                    "cluster_sources": list(candidate.get("cluster_sources") or []),
                    "span_endpoints": endpoints,
                    "support_count": candidate.get("support_count"),
                    "cluster_source_count": candidate.get("cluster_source_count"),
                }
            )
        selected = selected_by_case[case_id]
        rows.append(
            {
                "case_id": case_id,
                "family_id": family_for_case[case_id],
                "gold_count": ceiling["gold_count"],
                "pool_line_object_count": len(candidates),
                "candidate_ceiling_true_positive": ceiling["true_positive"],
                "candidate_ceiling_false_negative": ceiling["false_negative"],
                "candidate_ceiling_recall": _ratio(ceiling["true_positive"], ceiling["gold_count"]),
                "selected_true_positive": int(selected["true_positive"]),
                "selected_false_negative": int(selected["false_negative"]),
                "selection_recall_loss": int(ceiling["true_positive"]) - int(selected["true_positive"]),
                "candidate_matches": match_details,
                "candidate_generation_complete": ceiling["false_negative"] == 0,
            }
        )

    gold = sum(row["gold_count"] for row in rows)
    ceiling_tp = sum(row["candidate_ceiling_true_positive"] for row in rows)
    selected_tp = sum(row["selected_true_positive"] for row in rows)
    families = []
    for family_id in sorted(family_map["families"]):
        family_rows = [row for row in rows if row["family_id"] == family_id]
        family_gold = sum(row["gold_count"] for row in family_rows)
        family_ceiling = sum(row["candidate_ceiling_true_positive"] for row in family_rows)
        family_selected = sum(row["selected_true_positive"] for row in family_rows)
        families.append(
            {
                "family_id": family_id,
                "case_ids": [row["case_id"] for row in family_rows],
                "gold_count": family_gold,
                "candidate_ceiling_true_positive": family_ceiling,
                "candidate_ceiling_recall": _ratio(family_ceiling, family_gold),
                "selected_true_positive": family_selected,
                "selected_recall": _ratio(family_selected, family_gold),
                "selection_recall_loss": family_ceiling - family_selected,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_synthetic_source_only_candidate_ceiling",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "thresholds": {
            "axis_angle_deg": 5.0,
            "transverse_offset_mm": 4.0,
            "gold_span_overlap": 0.60,
            "prediction_span_overlap": 0.60,
        },
        "aggregate": {
            "gold_bend_count": gold,
            "candidate_ceiling_true_positive": ceiling_tp,
            "candidate_ceiling_recall": _ratio(ceiling_tp, gold),
            "selected_true_positive": selected_tp,
            "selected_recall": _ratio(selected_tp, gold),
            "selection_recall_loss": ceiling_tp - selected_tp,
            "cases_with_complete_candidate_generation": sum(row["candidate_generation_complete"] for row in rows),
        },
        "families": families,
        "cases": rows,
    }
    if args.output.exists():
        raise FileExistsError(f"candidate ceiling output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
