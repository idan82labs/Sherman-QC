#!/usr/bin/env python3
"""Evaluate V51 recovery V1 from frozen cached artifacts only.

This development evaluator deliberately separates two phases:

* source-only application opens V51 semantic results, candidate pools, and the
  frozen feature tree only;
* scoring/lineage opens the AI-adjudicated catalog only after both primary and
  replay successor artifacts have been materialized and byte-compared.

The retained V51 cache and baseline outputs are read-only and never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_f1_v51_source_only_recovery_v1 import (  # noqa: E402
    _canonical_bytes,
    apply_recovery,
    candidate_endpoints,
)
from scripts.evaluate_f1_against_manual_gold import (  # noqa: E402
    Thresholds,
    is_match,
    match_lines,
    pair_metrics,
)


DEFAULT_CATALOG = (
    ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
)
DEFAULT_CACHE_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
)
DEFAULT_MODEL = (
    ROOT / "data/bend_corpus/models/f1_v51_today_source_recovery_tree_v1_r3.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output/research/f1-v51-today-source-recovery-v1-20260831-r3"
)
TODAY_SCAN_ROLE = "2026-08-30_ai_multiview_consensus_development"
STAGES = ("v16", "v43", "v44", "v50", "v47", "v48", "v49", "v51.raw")
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _today_cases(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [
        dict(row)
        for row in catalog.get("cases") or []
        if row.get("scan_role") == TODAY_SCAN_ROLE
    ]
    expected = [f"T{index:03d}" for index in range(1, 29)]
    if [row.get("case_id") for row in cases] != expected:
        raise ValueError("catalog does not contain ordered T001..T028 daily cases")
    return cases


def _candidate_prediction(row: dict[str, Any], index: int) -> dict[str, Any]:
    endpoints = candidate_endpoints(row)
    return {
        "bend_id": f"POOL::{index:05d}::{row.get('candidate_id')}",
        "line_start": endpoints[0].tolist(),
        "line_end": endpoints[1].tolist(),
    }


def _line_matches(gold: dict[str, Any], prediction: dict[str, Any]) -> bool:
    gold_endpoints = np.asarray([gold["line_start"], gold["line_end"]], dtype=np.float64)
    prediction_endpoints = np.asarray(
        [prediction["line_start"], prediction["line_end"]], dtype=np.float64
    )
    return is_match(pair_metrics(gold_endpoints, prediction_endpoints), THRESHOLDS)


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in results)
    fp = sum(int(row["false_positive"]) for row in results)
    fn = sum(int(row["false_negative"]) for row in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "case_count": len(results),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(row["exact"]) for row in results),
        "exact_case_rate": round(sum(bool(row["exact"]) for row in results) / len(results), 6),
    }


def _score_cases(
    cases: list[dict[str, Any]],
    prediction_by_case: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = []
    by_case = {}
    for case in cases:
        case_id = str(case["case_id"])
        result = match_lines(case["bends"], prediction_by_case[case_id], THRESHOLDS)
        result["case_id"] = case_id
        rows.append(result)
        by_case[case_id] = result
    return _metrics(rows), by_case


def _delta(successor: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "prediction_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "exact_case_count",
    )
    return {
        key: round(float(successor[key]) - float(baseline[key]), 6)
        for key in keys
    }


def _matched_gold_ids(result: dict[str, Any]) -> set[str]:
    return {str(row["gold_id"]) for row in result["matches"]}


def evaluate_cached(
    catalog_path: Path,
    cache_root: Path,
    model_path: Path,
    output_root: Path,
    profiles: tuple[str, ...],
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    cache_root = cache_root.resolve()
    model_path = model_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("cached recovery output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("training_catalog_sha256") != _sha(catalog_path):
        raise ValueError("model/catalog training binding mismatch")
    cases = _today_cases(catalog)

    # Phase 1: materialize primary and replay without passing catalog data into
    # apply_recovery.  Only cached source artifacts and the frozen model cross
    # this boundary.
    successor_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
        profile: {} for profile in profiles
    }
    baseline_predictions: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    replay_identity: dict[str, dict[str, bool]] = {profile: {} for profile in profiles}
    for case in cases:
        case_id = str(case["case_id"])
        pass_outputs: dict[str, dict[str, dict[str, Any]]] = {profile: {} for profile in profiles}
        input_hashes[case_id] = {}
        for pass_name in ("primary", "replay"):
            case_root = cache_root / pass_name / case_id
            incumbent_path = case_root / "semantic_result.json"
            pool_path = case_root / "pool" / f"{case_id}.json"
            if not incumbent_path.is_file() or not pool_path.is_file():
                raise FileNotFoundError(f"{case_id}/{pass_name}: cached V51 artifacts are missing")
            incumbent = json.loads(incumbent_path.read_text(encoding="utf-8"))
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            input_hashes[case_id][f"{pass_name}_incumbent_sha256"] = _sha(incumbent_path)
            input_hashes[case_id][f"{pass_name}_pool_sha256"] = _sha(pool_path)
            if pass_name == "primary":
                baseline_predictions[case_id] = [dict(row) for row in incumbent["predictions"]]
            for profile in profiles:
                result = apply_recovery(incumbent, pool, model, profile_name=profile)
                result_path = output_root / profile / pass_name / case_id / "semantic_result.json"
                _write_json(result_path, result)
                pass_outputs[profile][pass_name] = {
                    "payload": result,
                    "path": str(result_path),
                    "sha256": _sha(result_path),
                }
        for profile in profiles:
            primary = pass_outputs[profile]["primary"]
            replay = pass_outputs[profile]["replay"]
            identical = primary["sha256"] == replay["sha256"]
            replay_identity[profile][case_id] = identical
            if not identical:
                raise RuntimeError(f"{profile}/{case_id}: successor primary/replay bytes differ")
            successor_predictions[profile][case_id] = [
                dict(row) for row in primary["payload"]["predictions"]
            ]

    # Phase 2: score and inspect candidate lineage after deterministic materialization.
    baseline_metrics, baseline_cases = _score_cases(cases, baseline_predictions)
    profile_reports = {}
    for profile in profiles:
        metrics, scored_cases = _score_cases(cases, successor_predictions[profile])
        newly_recovered: list[str] = []
        regressed: list[str] = []
        case_deltas = []
        for case in cases:
            case_id = str(case["case_id"])
            baseline_matched = _matched_gold_ids(baseline_cases[case_id])
            successor_matched = _matched_gold_ids(scored_cases[case_id])
            newly_recovered.extend(sorted(successor_matched - baseline_matched))
            regressed.extend(sorted(baseline_matched - successor_matched))
            case_deltas.append(
                {
                    "case_id": case_id,
                    "baseline_tp": baseline_cases[case_id]["true_positive"],
                    "successor_tp": scored_cases[case_id]["true_positive"],
                    "baseline_prediction_count": baseline_cases[case_id]["prediction_count"],
                    "successor_prediction_count": scored_cases[case_id]["prediction_count"],
                    "newly_recovered_gold_ids": sorted(successor_matched - baseline_matched),
                    "regressed_gold_ids": sorted(baseline_matched - successor_matched),
                }
            )
        profile_reports[profile] = {
            "metrics": metrics,
            "delta_from_v51": _delta(metrics, baseline_metrics),
            "newly_recovered_gold_count": len(newly_recovered),
            "newly_recovered_gold_ids": newly_recovered,
            "regressed_gold_count": len(regressed),
            "regressed_gold_ids": regressed,
            "all_case_primary_replay_byte_identity": all(replay_identity[profile].values()),
            "case_primary_replay_byte_identity": replay_identity[profile],
            "cases": case_deltas,
        }

    lineage_rows = []
    stage_presence_counts = {stage: 0 for stage in STAGES}
    pool_ceiling_results = []
    for case in cases:
        case_id = str(case["case_id"])
        primary_root = cache_root / "primary" / case_id
        pool = json.loads((primary_root / "pool" / f"{case_id}.json").read_text(encoding="utf-8"))
        pool_rows = [
            _candidate_prediction(row, index)
            for index, row in enumerate(pool.get("candidates") or [])
        ]
        line_object_rows = [
            _candidate_prediction(row, index)
            for index, row in enumerate(pool.get("line_objects") or [])
        ]
        stage_rows: dict[str, list[dict[str, Any]]] = {}
        for stage in STAGES:
            stage_path = primary_root / f"{stage}.json"
            stage_rows[stage] = list(
                json.loads(stage_path.read_text(encoding="utf-8")).get("predictions") or []
            )
        ceiling = match_lines(case["bends"], pool_rows, THRESHOLDS)
        ceiling["case_id"] = case_id
        pool_ceiling_results.append(ceiling)
        baseline_unmatched = set(baseline_cases[case_id]["unmatched_gold_ids"])
        for gold in case["bends"]:
            gold_id = str(gold["bend_id"])
            if gold_id not in baseline_unmatched:
                continue
            pool_hits = [row["bend_id"] for row in pool_rows if _line_matches(gold, row)]
            line_object_hits = [
                row["bend_id"] for row in line_object_rows if _line_matches(gold, row)
            ]
            stages_present = [
                stage
                for stage in STAGES
                if any(_line_matches(gold, row) for row in stage_rows[stage])
            ]
            for stage in stages_present:
                stage_presence_counts[stage] += 1
            classification = (
                "CANDIDATE_POOL_BORN_SELECTOR_OR_OWNERSHIP_MISS"
                if pool_hits
                else "EARLIER_STAGE_ONLY"
                if stages_present
                else "NEVER_BORN_IN_CACHED_V51_LINE_STREAM"
            )
            lineage_rows.append(
                {
                    "case_id": case_id,
                    "gold_id": gold_id,
                    "classification": classification,
                    "candidate_pool_match_count": len(pool_hits),
                    "line_object_match_count": len(line_object_hits),
                    "stages_present": stages_present,
                }
            )

    classification_counts = {
        name: sum(row["classification"] == name for row in lineage_rows)
        for name in (
            "CANDIDATE_POOL_BORN_SELECTOR_OR_OWNERSHIP_MISS",
            "EARLIER_STAGE_ONLY",
            "NEVER_BORN_IN_CACHED_V51_LINE_STREAM",
        )
    }
    if len(lineage_rows) != int(baseline_metrics["false_negative"]):
        raise RuntimeError("lineage rows do not account for every baseline false negative")
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_today_source_recovery_cached_development_result_v1",
        "evidence_class": "CONSUMED_AI_ADJUDICATED_PHYSICAL_DEVELOPMENT",
        "promotion_eligible": False,
        "generalization_evidence": False,
        "authoritative_ground_truth": False,
        "source_pipeline_rerun": False,
        "cached_artifacts_only": True,
        "truth_available_to_runtime_application": False,
        "model_development_tuned": True,
        "catalog_sha256": _sha(catalog_path),
        "model_sha256": _sha(model_path),
        "cache_execution_manifest_sha256": _sha(cache_root / "development_execution_manifest.json"),
        "thresholds": {
            "axis_angle_deg": THRESHOLDS.angle_deg,
            "transverse_offset_mm": THRESHOLDS.transverse_offset_mm,
            "gold_span_overlap": THRESHOLDS.gold_span_overlap,
            "prediction_span_overlap": THRESHOLDS.prediction_span_overlap,
            "matching": "one_to_one_minimum_cost",
        },
        "v51_baseline": baseline_metrics,
        "profiles": profile_reports,
        "false_negative_lineage": {
            "baseline_false_negative_count": len(lineage_rows),
            "classification_counts": classification_counts,
            "stage_presence_counts_among_baseline_false_negatives": stage_presence_counts,
            "candidate_pool_one_to_one_ceiling": _metrics(pool_ceiling_results),
            "rows": lineage_rows,
        },
        "input_hashes": input_hashes,
    }
    _write_json(output_root / "cached_recovery_development_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["balanced", "recall_max"],
        choices=("balanced", "recall_max"),
    )
    args = parser.parse_args()
    report = evaluate_cached(
        args.catalog,
        args.cache_root,
        args.model,
        args.output_root,
        tuple(args.profiles),
    )
    print(
        json.dumps(
            {
                "baseline": report["v51_baseline"],
                "profiles": {
                    name: row["metrics"] for name, row in report["profiles"].items()
                },
                "lineage": report["false_negative_lineage"]["classification_counts"],
                "report": str(args.output_root.resolve() / "cached_recovery_development_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
