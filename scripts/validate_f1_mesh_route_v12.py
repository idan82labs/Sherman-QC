#!/usr/bin/env python3
"""Compare the V9 bounded-mesh route before and after the V12 repair."""

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
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh
from scripts.train_f1_bounded_mesh_router_v9 import (
    NMS_ANGLE,
    NMS_DISTANCE,
    NMS_OVERLAP,
    SCORE_THRESHOLD,
    SMALL_POOL_MAXIMUM,
)


def _prediction(rows: list[dict], selected: list[int]) -> list[dict]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(rows[index])[0].tolist(),
            "line_end": endpoints(rows[index])[1].tolist(),
        }
        for index in selected
    ]


def _row_prediction(rows: list[dict]) -> list[dict]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(row)[0].tolist(),
            "line_end": endpoints(row)[1].tolist(),
        }
        for index, row in enumerate(rows)
    ]


def _aggregate(cases: list[dict], key: str) -> dict:
    tp = sum(int(case[key]["true_positive"]) for case in cases)
    fp = sum(int(case[key]["false_positive"]) for case in cases)
    fn = sum(int(case[key]["false_negative"]) for case in cases)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_case_count": sum(bool(case[key]["exact"]) for case in cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = joblib.load(args.v9_model)
    gold = {
        str(row["case_token"]): row
        for row in json.loads(args.gold_manifest.read_text(encoding="utf-8"))["cases"]
    }
    manifest = json.loads((args.pool_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool = json.loads((args.pool_dir / f"{token}.json").read_text(encoding="utf-8"))
        diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
        if not bool(diagnostics.get("bounded_decimation_used")):
            continue
        rows = [dict(row) for row in pool["line_objects"]]
        bbox_span = max(float(value) for value in pool["scan_bbox_mm"])
        if len(rows) <= SMALL_POOL_MAXIMUM:
            selected = select_bounded_mesh(rows, bbox_span)
        else:
            scores = model["estimator"].predict_proba(build_feature_matrix(rows))[:, 1]
            selected = select_indices_configurable(
                rows,
                scores,
                threshold=SCORE_THRESHOLD,
                max_angle=NMS_ANGLE,
                max_distance=NMS_DISTANCE,
                min_overlap=NMS_OVERLAP,
            )
        repaired, changes = repair_bounded_cylindrical_representatives(rows, selected)
        cases.append(
            {
                "case_token": token,
                "v9": match_lines(gold[token]["manual_bends"], _prediction(rows, selected), Thresholds()),
                "v12": match_lines(gold[token]["manual_bends"], _row_prediction(repaired), Thresholds()),
                "repair_count": len(changes),
                "repairs": changes,
            }
        )
    v9_metrics = _aggregate(cases, "v9")
    v12_metrics = _aggregate(cases, "v12")
    report = {
        "schema_version": 1,
        "kind": "f1_v12_bounded_mesh_development_regression",
        "case_count": len(cases),
        "evaluation_note": "same final V9 model before/after; isolates only V12 representative repair",
        "v9_metrics": v9_metrics,
        "v12_metrics": v12_metrics,
        "true_positive_loss": v9_metrics["true_positive"] - v12_metrics["true_positive"],
        "repair_count": sum(case["repair_count"] for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "v9_metrics", "v12_metrics", "true_positive_loss", "repair_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
