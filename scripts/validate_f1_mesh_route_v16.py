#!/usr/bin/env python3
"""Compare the unified V16 selector with the prior unified V12 route."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints, select_indices_configurable
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.select_f1_bounded_mesh_router_v9 import route as route_v9
from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives
from scripts.select_f1_full_mesh_consensus_v11 import prune_short_cylindrical_fragments
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_mesh_route_v16 import select_unified
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_bounded_mesh_router_v9 import NMS_ANGLE, NMS_DISTANCE, NMS_OVERLAP, SCORE_THRESHOLD
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE


def _predictions(rows: list[dict]) -> list[dict]:
    return [
        {
            "bend_id": str(index),
            "line_start": endpoints(row)[0].tolist(),
            "line_end": endpoints(row)[1].tolist(),
        }
        for index, row in enumerate(rows)
    ]


def _select_v12(pool: dict, v9: dict, v4: dict) -> tuple[str, list[dict]]:
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    v9_scores = v9["estimator"].predict_proba(features)[:, 1]
    core_scores = v4["core_estimator"].predict_proba(features)[:, 1]
    adjacency_scores = np.zeros(len(rows), dtype=float)
    adjacency_indices = [
        index for index, row in enumerate(rows) if row.get("representative_source") == ADJACENCY_SOURCE
    ]
    if adjacency_indices:
        adjacency_scores[adjacency_indices] = v4["adjacency_estimator"].predict_proba(
            features[adjacency_indices]
        )[:, 1]
    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    mesh_available = bool(diagnostics.get("available"))
    bounded_mesh = bool(diagnostics.get("bounded_decimation_used"))
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    if mesh_available and not bounded_mesh:
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        selected, _ = repair_full_mesh_representatives(rows, selected)
        selected, _ = prune_short_cylindrical_fragments(rows, selected)
        return "full_resolution_mesh_v11", [rows[index] for index in selected]
    if bounded_mesh:
        route = route_v9(bounded_mesh=True, line_object_count=len(rows))
        if route == "bounded_mesh_small_pool_recall":
            selected = select_bounded_mesh(rows, bbox_span)
        else:
            selected = select_indices_configurable(
                rows,
                v9_scores,
                threshold=SCORE_THRESHOLD,
                max_angle=NMS_ANGLE,
                max_distance=NMS_DISTANCE,
                min_overlap=NMS_OVERLAP,
            )
        repaired, _ = repair_bounded_cylindrical_representatives(rows, selected)
        return route + "_consensus_repair", repaired
    selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
    return "non_mesh_v6", [rows[index] for index in selected]


def _aggregate(scores: list[dict]) -> dict:
    tp = sum(int(score["true_positive"]) for score in scores)
    fp = sum(int(score["false_positive"]) for score in scores)
    fn = sum(int(score["false_negative"]) for score in scores)
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
        "exact_case_count": sum(bool(score["exact"]) for score in scores),
        "case_count": len(scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.pool_dir / "manifest.json").read_text(encoding="utf-8"))
    gold = {row["case_token"]: row for row in json.loads(args.gold_manifest.read_text(encoding="utf-8"))["cases"]}
    v9 = joblib.load(args.v9_model)
    v4 = joblib.load(args.v4_model)
    cases = []
    for item in manifest["cases"]:
        token = str(item["case_id"])
        pool = json.loads(Path(item.get("path") or args.pool_dir / f"{token}.json").read_text(encoding="utf-8"))
        baseline_route, baseline_rows = _select_v12(pool, v9, v4)
        selection = select_unified(pool, v9, v4)
        v16_rows = selection["prediction_rows"]
        cases.append(
            {
                "case_token": token,
                "part_key": str(gold[token]["part_key"]),
                "scan_state": str(gold[token]["scan_state"]),
                "v12_route": baseline_route,
                "v16_route": selection["route"],
                "v12": match_lines(gold[token]["manual_bends"], _predictions(baseline_rows), Thresholds()),
                "v16": match_lines(gold[token]["manual_bends"], _predictions(v16_rows), Thresholds()),
            }
        )
    families: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        families[case["part_key"]].append(case)
    family_metrics = {
        family: {
            "v12": _aggregate([case["v12"] for case in rows]),
            "v16": _aggregate([case["v16"] for case in rows]),
        }
        for family, rows in sorted(families.items())
    }
    v12_metrics = _aggregate([case["v12"] for case in cases])
    v16_metrics = _aggregate([case["v16"] for case in cases])
    report = {
        "schema_version": 1,
        "kind": "f1_v16_unified_mesh_route_development_regression",
        "case_count": len(cases),
        "family_count": len(families),
        "route_counts": dict(sorted(Counter(case["v16_route"] for case in cases).items())),
        "v12_metrics": v12_metrics,
        "v16_metrics": v16_metrics,
        "f1_gain": round(v16_metrics["f1"] - v12_metrics["f1"], 6),
        "true_positive_loss": v12_metrics["true_positive"] - v16_metrics["true_positive"],
        "min_family_f1_v12": min((row["v12"]["f1"] for row in family_metrics.values()), default=0.0),
        "min_family_f1_v16": min((row["v16"]["f1"] for row in family_metrics.values()), default=0.0),
        "family_metrics": family_metrics,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "family_count", "route_counts", "v12_metrics", "v16_metrics", "f1_gain", "true_positive_loss", "min_family_f1_v16")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
