#!/usr/bin/env python3
"""Regression-check V14's guarded retry and seed-family coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints, select_indices_configurable
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.select_f1_mesh_route_v13 import select_with_no_output_retry
from scripts.select_f1_mesh_route_v14 import recover_continuous_long_seed_families
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh
from scripts.train_f1_bounded_mesh_router_v9 import NMS_ANGLE, NMS_DISTANCE, NMS_OVERLAP, SCORE_THRESHOLD, SMALL_POOL_MAXIMUM


def _score(gold: list[dict], rows: list[dict], selected: list[int]) -> dict:
    predictions = [
        {
            "bend_id": str(index),
            "line_start": endpoints(rows[index])[0].tolist(),
            "line_end": endpoints(rows[index])[1].tolist(),
        }
        for index in selected
    ]
    return match_lines(gold, predictions, Thresholds())


def _aggregate(cases: list[dict], key: str) -> dict:
    tp = sum(case[key]["true_positive"] for case in cases)
    fp = sum(case[key]["false_positive"] for case in cases)
    fn = sum(case[key]["false_negative"] for case in cases)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "f1": round(2.0 * tp / (2.0 * tp + fp + fn), 6) if 2 * tp + fp + fn else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pool_dir / "manifest.json").read_text(encoding="utf-8"))
    gold = {str(row["case_token"]): row for row in json.loads(args.gold_manifest.read_text(encoding="utf-8"))["cases"]}
    estimator = joblib.load(args.v9_model)["estimator"]
    cases = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool = json.loads((args.pool_dir / f"{token}.json").read_text(encoding="utf-8"))
        diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
        if not bool(diagnostics.get("bounded_decimation_used")):
            continue
        rows = [dict(row) for row in pool["line_objects"]]
        bbox_span = max(float(value) for value in pool["scan_bbox_mm"])
        scores = estimator.predict_proba(build_feature_matrix(rows))[:, 1]
        if len(rows) <= SMALL_POOL_MAXIMUM:
            baseline = select_bounded_mesh(rows, bbox_span)
            v14 = list(baseline)
            retried = False
            coverage = []
        else:
            baseline = select_indices_configurable(
                rows, scores, threshold=SCORE_THRESHOLD,
                max_angle=NMS_ANGLE, max_distance=NMS_DISTANCE, min_overlap=NMS_OVERLAP,
            )
            v14, retried = select_with_no_output_retry(rows, scores)
            coverage = []
            if retried and v14:
                v14, coverage = recover_continuous_long_seed_families(rows, scores, v14, bbox_span)
        cases.append(
            {
                "case_token": token,
                "baseline": _score(gold[token]["manual_bends"], rows, baseline),
                "v14": _score(gold[token]["manual_bends"], rows, v14),
                "retry_used": retried,
                "coverage_count": len(coverage),
                "coverage": coverage,
            }
        )
    baseline_metrics = _aggregate(cases, "baseline")
    v14_metrics = _aggregate(cases, "v14")
    report = {
        "schema_version": 1,
        "kind": "f1_v14_bounded_mesh_development_regression",
        "case_count": len(cases),
        "baseline_metrics": baseline_metrics,
        "v14_metrics": v14_metrics,
        "true_positive_loss": baseline_metrics["true_positive"] - v14_metrics["true_positive"],
        "retry_case_count": sum(case["retry_used"] for case in cases),
        "coverage_recovery_count": sum(case["coverage_count"] for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "baseline_metrics", "v14_metrics", "true_positive_loss", "retry_case_count", "coverage_recovery_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
