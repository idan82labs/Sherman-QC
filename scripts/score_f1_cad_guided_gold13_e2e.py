#!/usr/bin/env python3
"""Score blind CAD-guided scan detections against the matched Gold13 slice."""

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
from scripts.f1_physical_evaluation_contract_v2 import score_localized_cases


def _cad_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for row in result.get("matches") or []:
        start = row.get("detected_line_start")
        end = row.get("detected_line_end")
        if start is None or end is None:
            continue
        predictions.append(
            {
                "bend_id": str(row.get("bend_id") or f"CAD_SCAN_{len(predictions) + 1}"),
                "line_start": start,
                "line_end": end,
                "cad_target_id": str(row.get("bend_id") or ""),
                "assignment_source": row.get("assignment_source"),
                "measurement_method": row.get("measurement_method"),
                "observability_state": row.get("observability_state"),
            }
        )
    return predictions


def _family_minimum(cases: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        by_family.setdefault(str(row["family_id"]), []).append(row)
    recalls = {
        family: sum(row["true_positive"] for row in rows)
        / sum(row["gold_count"] for row in rows)
        for family, rows in by_family.items()
    }
    return min(recalls.values()), recalls


def _a0_subset(
    *, scan_shas: set[str], pool_manifest: dict[str, Any], a0_score: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    token_by_sha = {
        str(row["source_scan_sha256"]): str(row["case_token"])
        for row in pool_manifest["cases"]
    }
    covered_shas = scan_shas.intersection(token_by_sha)
    selected_tokens = {token_by_sha[sha] for sha in covered_shas}
    rows = [
        {**dict(row), "case_id": str(row["case_token"])}
        for row in a0_score["metrics"]["cases"]
        if str(row["case_token"]) in selected_tokens
    ]
    if len(rows) != len(covered_shas):
        raise ValueError("A0 baseline token join is incomplete")
    corrected = score_localized_cases(rows)
    corrected["minimum_family_recall"], corrected["family_recalls"] = _family_minimum(
        [{**row, "family_id": row["part_key"]} for row in rows]
    )
    return corrected, covered_shas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--cad-summary", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--a0-pool-manifest", type=Path, required=True)
    parser.add_argument("--a0-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()

    slice_payload = json.loads(args.slice.read_text(encoding="utf-8"))
    summary = json.loads(args.cad_summary.read_text(encoding="utf-8"))
    catalog = json.loads(args.gold_catalog.read_text(encoding="utf-8"))
    catalog_by_sha = {str(row["scan_sha256"]): row for row in catalog["cases"]}
    slice_by_path = {str(Path(row["scan_path"]).resolve()): row for row in slice_payload["entries"]}
    thresholds = Thresholds()
    matched_cases = []
    visual_records = []
    for result in summary["results"]:
        scan_path = str(Path(result["scan_path"]).resolve())
        entry = slice_by_path.get(scan_path)
        if entry is None:
            raise ValueError(f"unexpected inference result: {scan_path}")
        gold = catalog_by_sha[str(entry["scan_sha256"])]
        predictions = _cad_predictions(result)
        scored = match_lines(gold["bends"], predictions, thresholds)
        scored.update(
            {
                "case_id": str(entry["case_token"]),
                "family_id": str(gold["part_id"]),
                "truth_record_id": str(gold["case_id"]),
                "truth_tier": str(gold["truth_tier"]),
                "scan_sha256": str(entry["scan_sha256"]),
                "scan_path": scan_path,
                "cad_path": str(Path(result["cad_path"]).resolve()),
                "cad_strategy": (result.get("details") or {}).get("cad_strategy"),
                "cad_target_count": int((result.get("summary") or {}).get("total_bends") or 0),
                "detected_cad_target_count": len(predictions),
                "pipeline_latency_sec": float((result.get("timing") or {}).get("total_sec") or 0.0),
            }
        )
        matched_cases.append(scored)
        payload = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_cad_guided_scan_line_predictions",
            "cad_guided_zero_shot": True,
            "source_only": False,
            "research_only": True,
            "route": "blind_cad_ply_detected_lines",
            "predictions": predictions,
        }
        path = args.prediction_root / str(entry["case_token"]) / "semantic_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        visual_records.append(
            {
                "case_id": str(entry["case_token"]),
                "scan_path": scan_path,
                "prediction_path": str(path.resolve()),
                "prediction_count": len(predictions),
            }
        )
    if len(matched_cases) != 13:
        raise ValueError(f"expected 13 scored cases, got {len(matched_cases)}")

    cad_metrics = score_localized_cases(matched_cases)
    cad_metrics["minimum_family_recall"], cad_metrics["family_recalls"] = _family_minimum(matched_cases)
    baseline, baseline_shas = _a0_subset(
        scan_shas={str(row["scan_sha256"]) for row in slice_payload["entries"]},
        pool_manifest=json.loads(args.a0_pool_manifest.read_text(encoding="utf-8")),
        a0_score=json.loads(args.a0_score.read_text(encoding="utf-8")),
    )
    overlap_cases = [row for row in matched_cases if row["scan_sha256"] in baseline_shas]
    cad_overlap = score_localized_cases(overlap_cases)
    cad_overlap["minimum_family_recall"], cad_overlap["family_recalls"] = _family_minimum(overlap_cases)
    deltas = {
        "precision": cad_overlap["precision"] - baseline["precision"],
        "recall": cad_overlap["recall"] - baseline["recall"],
        "f1": cad_overlap["f1"] - baseline["f1"],
        "exact_count_accuracy": cad_overlap["exact_count_accuracy"] - baseline["exact_count_accuracy"],
        "exact_localized_target_set_accuracy": (
            cad_overlap["exact_localized_target_set_accuracy"]
            - baseline["exact_localized_target_set_accuracy"]
        ),
        "minimum_family_recall": cad_overlap["minimum_family_recall"] - baseline["minimum_family_recall"],
    }
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_guided_gold13_e2e_line_comparison",
        "cad_guided_zero_shot": True,
        "scan_only_zero_shot": False,
        "promotion_eligible": False,
        "blind_inference_contract_rate": summary["typed_release_metrics"]["blind_inference_contract_rate"],
        "job_success_rate": summary["typed_release_metrics"]["job_success_rate"],
        "p95_pipeline_latency_sec": summary["typed_release_metrics"]["p95_pipeline_latency_sec"],
        "full_cad_case_count": len(matched_cases),
        "matched_a0_case_count": len(overlap_cases),
        "a0_unavailable_scan_sha256": sorted(
            {str(row["scan_sha256"]) for row in slice_payload["entries"]} - baseline_shas
        ),
        "matching": {
            "axis_angle_deg_max": thresholds.angle_deg,
            "transverse_offset_mm_max": thresholds.transverse_offset_mm,
            "gold_span_overlap_min": thresholds.gold_span_overlap,
            "prediction_span_overlap_min": thresholds.prediction_span_overlap,
        },
        "scan_only_a0": baseline,
        "cad_guided_matched_a0_subset": cad_overlap,
        "cad_guided": cad_metrics,
        "deltas": deltas,
        "cases": matched_cases,
        "visual_records": visual_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "scan_only_f1": baseline["f1"], "cad_guided_f1": cad_metrics["f1"], "f1_delta": deltas["f1"], "cad_exact_count_accuracy": cad_metrics["exact_count_accuracy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
