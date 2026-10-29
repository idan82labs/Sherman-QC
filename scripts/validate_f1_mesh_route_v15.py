#!/usr/bin/env python3
"""Regression-check V15 against V14 on bounded development meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_mesh_route_v13 import select_with_no_output_retry
from scripts.select_f1_mesh_route_v14 import recover_continuous_long_seed_families
from scripts.select_f1_mesh_route_v15 import (
    recover_separated_parallel_plane_pairs,
    repair_tilted_adjacency_representatives,
    select_with_v15_nms,
)
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh
from scripts.train_f1_bounded_mesh_router_v9 import SMALL_POOL_MAXIMUM


def _score(gold: list[dict], rows: list[dict]) -> dict:
    predictions = []
    for index, row in enumerate(rows):
        line = endpoints(row)
        predictions.append({"bend_id": str(index), "line_start": line[0].tolist(), "line_end": line[1].tolist()})
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
            v14_selected = select_bounded_mesh(rows, bbox_span)
            v15_selected = list(v14_selected)
            pair_recoveries = []
            tilted_repairs = []
        else:
            v14_selected, retried = select_with_no_output_retry(rows, scores)
            if retried and v14_selected:
                v14_selected, _ = recover_continuous_long_seed_families(rows, scores, v14_selected, bbox_span)
            v15_selected, retried = select_with_v15_nms(rows, scores)
            if retried and v15_selected:
                v15_selected, _ = recover_continuous_long_seed_families(rows, scores, v15_selected, bbox_span)
            v15_selected, pair_recoveries = recover_separated_parallel_plane_pairs(rows, v15_selected, bbox_span)
            v15_selected, tilted_repairs = repair_tilted_adjacency_representatives(rows, v15_selected)
        v14_rows, _ = repair_bounded_cylindrical_representatives(rows, v14_selected)
        v15_rows, _ = repair_bounded_cylindrical_representatives(rows, v15_selected)
        cases.append(
            {
                "case_token": token,
                "v14": _score(gold[token]["manual_bends"], v14_rows),
                "v15": _score(gold[token]["manual_bends"], v15_rows),
                "pair_recovery_count": len(pair_recoveries),
                "tilted_repair_count": len(tilted_repairs),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "f1_v15_bounded_mesh_development_regression",
        "case_count": len(cases),
        "v14_metrics": _aggregate(cases, "v14"),
        "v15_metrics": _aggregate(cases, "v15"),
        "pair_recovery_count": sum(case["pair_recovery_count"] for case in cases),
        "tilted_repair_count": sum(case["tilted_repair_count"] for case in cases),
        "cases": cases,
    }
    report["true_positive_loss"] = report["v14_metrics"]["true_positive"] - report["v15_metrics"]["true_positive"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "v14_metrics", "v15_metrics", "true_positive_loss", "pair_recovery_count", "tilted_repair_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
