#!/usr/bin/env python3
"""Materialize and score cached axis/span recovery on the daily 28."""

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

from backend.f1_post_v51_precision_ownership_v52_development import compose_v52_development
from scripts.apply_f1_v51_source_only_recovery_v1 import candidate_endpoints
from scripts.apply_f1_v51_today_axis_span_recovery_v1 import (
    _canonical_bytes,
    compose_axis_span_recovery,
    generate_axis_span_variants,
)
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines


DEFAULT_CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"
DEFAULT_V51_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
DEFAULT_RECALL_ROOT = ROOT / "output/research/f1-v51-today-source-recovery-v1-20260831-r3/balanced"
DEFAULT_MODEL = ROOT / "data/bend_corpus/models/f1_v51_today_axis_span_recovery_v1_r2.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-v51-today-axis-span-recovery-v1-20260831-r2"
TODAY_SCAN_ROLE = "2026-08-30_ai_multiview_consensus_development"
THRESHOLDS = Thresholds(5.0, 4.0, 0.60, 0.60)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _today_cases(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [dict(row) for row in catalog.get("cases") or [] if row.get("scan_role") == TODAY_SCAN_ROLE]
    if [row.get("case_id") for row in cases] != [f"T{index:03d}" for index in range(1, 29)]:
        raise ValueError("catalog must contain ordered T001..T028 daily cases")
    return cases


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positive"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["false_negative"]) for row in rows)
    return {
        "case_count": len(rows),
        "gold_bend_count": tp + fn,
        "prediction_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
        "f1": round(2.0 * tp / (2.0 * tp + fp + fn), 6) if tp else 0.0,
        "exact_case_count": sum(bool(row["exact"]) for row in rows),
    }


def _delta(successor: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(float(successor[key]) - float(baseline[key]), 6)
        for key in (
            "prediction_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "exact_case_count",
        )
    }


def evaluate(
    catalog_path: Path,
    v51_root: Path,
    recall_root: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    v51_root = v51_root.resolve()
    recall_root = recall_root.resolve()
    model_path = model_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("axis/span output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("training_catalog_sha256") != _sha(catalog_path):
        raise ValueError("axis/span model/catalog binding mismatch")
    cases = _today_cases(catalog)

    primary_predictions: dict[str, list[dict[str, Any]]] = {}
    base_union_predictions: dict[str, list[dict[str, Any]]] = {}
    identity: dict[str, bool] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    emitted_count: dict[str, int] = {}
    for case in cases:
        case_id = str(case["case_id"])
        payloads = {}
        input_hashes[case_id] = {}
        for pass_name in ("primary", "replay"):
            case_root = v51_root / pass_name / case_id
            v51_path = case_root / "semantic_result.json"
            pool_path = case_root / "pool" / f"{case_id}.json"
            recall_path = recall_root / pass_name / case_id / "semantic_result.json"
            for name, path in (("v51", v51_path), ("pool", pool_path), ("recall", recall_path)):
                if not path.is_file():
                    raise FileNotFoundError(f"{case_id}/{pass_name}: missing {name} artifact")
                input_hashes[case_id][f"{pass_name}_{name}_sha256"] = _sha(path)
            v51 = json.loads(v51_path.read_text(encoding="utf-8"))
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            recall = json.loads(recall_path.read_text(encoding="utf-8"))
            result = compose_axis_span_recovery(v51, recall, pool, model)
            path = output_root / pass_name / case_id / "semantic_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_canonical_bytes(result))
            payloads[pass_name] = (result, path)
            if pass_name == "primary":
                precision = compose_v52_development(v51, pool)
                incumbent_ids = {str(row.get("bend_id")) for row in v51["predictions"]}
                additions = [
                    row for row in recall["predictions"] if str(row.get("bend_id")) not in incumbent_ids
                ]
                base_union_predictions[case_id] = [dict(row) for row in precision["predictions"]] + [dict(row) for row in additions]
        identity[case_id] = payloads["primary"][1].read_bytes() == payloads["replay"][1].read_bytes()
        if not identity[case_id]:
            raise RuntimeError(f"{case_id}: primary/replay axis-span bytes differ")
        primary_predictions[case_id] = [dict(row) for row in payloads["primary"][0]["predictions"]]
        emitted_count[case_id] = int(payloads["primary"][0]["axis_span_audit"]["emitted_variant_count"])

    base_rows = []
    successor_rows = []
    case_diagnostics = []
    raw_pool_ceiling_rows = []
    variant_ceiling_rows = []
    combined_ceiling_rows = []
    for case in cases:
        case_id = str(case["case_id"])
        base = match_lines(case["bends"], base_union_predictions[case_id], THRESHOLDS)
        successor = match_lines(case["bends"], primary_predictions[case_id], THRESHOLDS)
        base_rows.append(base)
        successor_rows.append(successor)
        base_gold = {str(row["gold_id"]) for row in base["matches"]}
        successor_gold = {str(row["gold_id"]) for row in successor["matches"]}
        case_diagnostics.append(
            {
                "case_id": case_id,
                "base_tp": base["true_positive"],
                "successor_tp": successor["true_positive"],
                "base_fp": base["false_positive"],
                "successor_fp": successor["false_positive"],
                "variant_emitted_count": emitted_count[case_id],
                "newly_recovered_gold_ids": sorted(successor_gold - base_gold),
                "regressed_gold_ids": sorted(base_gold - successor_gold),
            }
        )

        pool_path = v51_root / "primary" / case_id / "pool" / f"{case_id}.json"
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        raw_rows = []
        for index, candidate in enumerate(pool.get("candidates") or []):
            endpoints = candidate_endpoints(candidate)
            raw_rows.append(
                {
                    "bend_id": f"POOL-{index:05d}",
                    "line_start": endpoints[0].tolist(),
                    "line_end": endpoints[1].tolist(),
                }
            )
        variants = generate_axis_span_variants(pool)
        raw_pool_ceiling_rows.append(match_lines(case["bends"], raw_rows, THRESHOLDS))
        variant_ceiling_rows.append(match_lines(case["bends"], variants, THRESHOLDS))
        combined_ceiling_rows.append(
            match_lines(case["bends"], base_union_predictions[case_id] + variants, THRESHOLDS)
        )

    base_metrics = _aggregate(base_rows)
    successor_metrics = _aggregate(successor_rows)
    report = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_today_axis_span_recovery_cached_development_report_v1",
        "catalog_sha256": _sha(catalog_path),
        "model_sha256": _sha(model_path),
        "v51_execution_manifest_sha256": _sha(v51_root / "development_execution_manifest.json"),
        "recall_report_sha256": _sha(recall_root.parent / "cached_recovery_development_report.json"),
        "cached_artifacts_only": True,
        "source_pipeline_rerun": False,
        "source_only_runtime": True,
        "truth_available_to_runtime_application": False,
        "development_tuned": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "authoritative_ground_truth": False,
        "base_precision_plus_r3_recall_union": base_metrics,
        "axis_span_successor_resubstitution": successor_metrics,
        "delta_from_base_union": _delta(successor_metrics, base_metrics),
        "case_group_held_out_diagnostic": model["development_diagnostics"]["case_group_held_out_metrics"],
        "candidate_oracle_ceilings": {
            "raw_pool_candidates": _aggregate(raw_pool_ceiling_rows),
            "axis_span_variants": _aggregate(variant_ceiling_rows),
            "base_union_plus_axis_span_variants": _aggregate(combined_ceiling_rows),
        },
        "all_case_primary_replay_byte_identity": all(identity.values()),
        "case_primary_replay_byte_identity": identity,
        "case_diagnostics": case_diagnostics,
        "input_hashes": input_hashes,
        "overfit_caveat": model["development_diagnostics"]["overfit_caveat"],
    }
    report_path = output_root / "cached_axis_span_development_report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--v51-root", type=Path, default=DEFAULT_V51_ROOT)
    parser.add_argument("--recall-root", type=Path, default=DEFAULT_RECALL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.catalog, args.v51_root, args.recall_root, args.model, args.output_root)
    report_path = args.output_root.resolve() / "cached_axis_span_development_report.json"
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": _sha(report_path),
                "base": report["base_precision_plus_r3_recall_union"],
                "successor": report["axis_span_successor_resubstitution"],
                "held_out": report["case_group_held_out_diagnostic"],
                "oracle": report["candidate_oracle_ceilings"],
                "replay_identity": report["all_case_primary_replay_byte_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
