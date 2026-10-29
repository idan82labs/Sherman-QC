#!/usr/bin/env python3
"""Score opaque physical-development results against frozen Gold12 labels.

The first twelve scans in the physical15 lane are byte-identical to the
count-backed July corpus and already have frozen multiview line annotations.
Those annotations are development truth, not an untouched holdout.  This
script joins solely by scan SHA-256 and reports the corrected metric contract:
count equality, exact localized target sets, and line precision/recall/F1 are
kept separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gold_by_scan_sha(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in catalog["cases"]:
        digest = str(row["scan_sha256"])
        if digest in result:
            raise ValueError(f"duplicate gold scan SHA-256: {digest}")
        result[digest] = row
    return result


def score_stage(
    *,
    stage: str,
    result_root: Path,
    public_manifest: dict[str, Any],
    gold_index: dict[str, dict[str, Any]],
    thresholds: Thresholds,
) -> dict[str, Any]:
    matched_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    for public_row in public_manifest["cases"]:
        case_id = str(public_row["opaque_case_id"])
        scan_sha = str(public_row["source_sha256"])
        gold = gold_index.get(scan_sha)
        if gold is None:
            continue
        if gold.get("truth_tier") != "codex_manual_multiview_count_backed":
            continue
        prediction_path = result_root / "primary" / case_id / "semantic_result.json"
        prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        result = match_lines(gold["bends"], prediction_payload["predictions"], thresholds)
        result["case_id"] = case_id
        matched_rows.append(result)
        detailed_rows.append(
            {
                **result,
                "family_id": str(gold["part_id"]),
                "scan_sha256": scan_sha,
                "gold_case_id": str(gold["case_id"]),
                "gold_truth_tier": str(gold["truth_tier"]),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": _sha(prediction_path),
            }
        )
    if len(detailed_rows) != 12:
        raise ValueError(f"expected 12 hash-matched count-backed cases, got {len(detailed_rows)}")

    aggregate = score_localized_cases(matched_rows)
    family_recalls = {
        row["family_id"]: row["true_positive"] / row["gold_count"]
        for row in detailed_rows
    }
    aggregate["minimum_family_recall"] = min(family_recalls.values())
    aggregate["family_recalls"] = family_recalls
    return {
        "stage": stage,
        "evidence_class": "development_consumed_physical_gold12",
        "promotion_eligible": False,
        "truth_independent_of_current_execution": True,
        "untouched_family_holdout": False,
        "metrics": aggregate,
        "cases": detailed_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-name", default="V51")
    parser.add_argument("--candidate-name", default="V52")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public_manifest = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.gold_catalog.read_text(encoding="utf-8"))
    thresholds = Thresholds()
    gold_index = _gold_by_scan_sha(catalog)
    baseline = score_stage(
        stage=args.baseline_name,
        result_root=args.baseline_root,
        public_manifest=public_manifest,
        gold_index=gold_index,
        thresholds=thresholds,
    )
    candidate = score_stage(
        stage=args.candidate_name,
        result_root=args.candidate_root,
        public_manifest=public_manifest,
        gold_index=gold_index,
        thresholds=thresholds,
    )
    before = baseline["metrics"]
    after = candidate["metrics"]
    deltas = {
        "f1": after["f1"] - before["f1"],
        "precision": after["precision"] - before["precision"],
        "recall": after["recall"] - before["recall"],
        "exact_count_accuracy": after["exact_count_accuracy"] - before["exact_count_accuracy"],
        "exact_localized_target_set_accuracy": (
            after["exact_localized_target_set_accuracy"]
            - before["exact_localized_target_set_accuracy"]
        ),
        "minimum_family_recall": after["minimum_family_recall"] - before["minimum_family_recall"],
    }
    sensitivity = []
    for angle_deg, offset_mm in ((5.0, 4.0), (7.5, 4.0), (5.0, 6.0), (7.5, 6.0)):
        alternate = Thresholds(angle_deg=angle_deg, transverse_offset_mm=offset_mm)
        alternate_baseline = score_stage(
            stage=args.baseline_name,
            result_root=args.baseline_root,
            public_manifest=public_manifest,
            gold_index=gold_index,
            thresholds=alternate,
        )["metrics"]
        alternate_candidate = score_stage(
            stage=args.candidate_name,
            result_root=args.candidate_root,
            public_manifest=public_manifest,
            gold_index=gold_index,
            thresholds=alternate,
        )["metrics"]
        sensitivity.append(
            {
                "axis_angle_deg_max": angle_deg,
                "transverse_offset_mm_max": offset_mm,
                "gold_span_overlap_min": alternate.gold_span_overlap,
                "prediction_span_overlap_min": alternate.prediction_span_overlap,
                "baseline_f1": alternate_baseline["f1"],
                "candidate_f1": alternate_candidate["f1"],
                "f1_delta": alternate_candidate["f1"] - alternate_baseline["f1"],
            }
        )
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_physical15_development_gold12_comparison",
        "protocol_valid": True,
        "leakage_free_model_path": True,
        "development_truth_opened_only_after_both_executions": True,
        "promotion_eligible": False,
        "matching": {
            "axis_angle_deg_max": thresholds.angle_deg,
            "transverse_offset_mm_max": thresholds.transverse_offset_mm,
            "gold_span_overlap_min": thresholds.gold_span_overlap,
            "prediction_span_overlap_min": thresholds.prediction_span_overlap,
            "assignment": "one_to_one_minimum_cost",
        },
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "matcher_sensitivity_non_authoritative": sensitivity,
        "development_direction": "REGRESSION" if deltas["f1"] < 0 else "NON_REGRESSION",
    }
    _write(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline_f1": before["f1"],
                "candidate_f1": after["f1"],
                "f1_delta": deltas["f1"],
                "development_direction": report["development_direction"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
