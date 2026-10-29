#!/usr/bin/env python3
"""Unified source-only mesh router with the retained V15 bounded-complex lane.

V15 was intentionally a bounded-complex experiment.  This entry point makes
that scope explicit: native full-resolution meshes stay on the proven V11
lane, bounded small pools stay on the V7 recall lane, and only bounded complex
pools use the V15 paired-plane-fold selector.  Non-mesh inputs retain V6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, endpoints
from scripts.select_f1_bounded_mesh_router_v9 import route as route_v9
from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives
from scripts.select_f1_full_mesh_consensus_v11 import prune_short_cylindrical_fragments
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_mesh_route_v14 import recover_continuous_long_seed_families
from scripts.select_f1_mesh_route_v15 import (
    NMS_ANGLE_DEG,
    NMS_DISTANCE_MM,
    recover_separated_parallel_plane_pairs,
    repair_tilted_adjacency_representatives,
    select_with_v15_nms,
)
from scripts.select_f1_topology_bounded_mesh_v7 import _coverage_score, select_bounded_mesh
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_bounded_mesh_router_v9 import PROFILE as V9_PROFILE, SCORE_THRESHOLD
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


PROFILE = "F1_SOURCE_ONLY_UNIFIED_MESH_ROUTE_V16"
ROUTE_FULL = "full_resolution_mesh_v11"
ROUTE_SMALL = "bounded_mesh_small_pool_recall_consensus_repair"
ROUTE_COMPLEX = "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair"
ROUTE_NON_MESH = "non_mesh_v6"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_route(pool: dict, line_object_count: int) -> str:
    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    mesh_available = bool(diagnostics.get("available"))
    bounded_mesh = bool(diagnostics.get("bounded_decimation_used"))
    if mesh_available and not bounded_mesh:
        return ROUTE_FULL
    if bounded_mesh:
        routed = route_v9(bounded_mesh=True, line_object_count=line_object_count)
        return ROUTE_SMALL if routed == "bounded_mesh_small_pool_recall" else ROUTE_COMPLEX
    return ROUTE_NON_MESH


def select_unified(pool: dict, v9: dict, v4: dict) -> dict:
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    if v9.get("profile") != V9_PROFILE or tuple(v9.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V9 model bundle does not match the frozen feature schema")
    if v4.get("profile") != V4_PROFILE or tuple(v4.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")

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

    route = classify_route(pool, len(rows))
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    repairs: list[dict] = []
    prunes: list[dict] = []
    coverage: list[dict] = []
    pair_recoveries: list[dict] = []
    tilted_repairs: list[dict] = []
    retried = False

    if route == ROUTE_FULL:
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        selected, repairs = repair_full_mesh_representatives(rows, selected)
        selected, prunes = prune_short_cylindrical_fragments(rows, selected)
        prediction_rows = [dict(rows[index]) for index in selected]
    elif route == ROUTE_SMALL:
        selected = select_bounded_mesh(rows, bbox_span)
        prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
    elif route == ROUTE_COMPLEX:
        selected, retried = select_with_v15_nms(
            rows,
            v9_scores,
            score_threshold=float(v9.get("score_threshold", SCORE_THRESHOLD)),
        )
        if retried and selected:
            selected, coverage = recover_continuous_long_seed_families(
                rows, v9_scores, selected, bbox_span
            )
        selected, pair_recoveries = recover_separated_parallel_plane_pairs(rows, selected, bbox_span)
        selected, tilted_repairs = repair_tilted_adjacency_representatives(rows, selected)
        prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
    else:
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        prediction_rows = [dict(rows[index]) for index in selected]

    return {
        "rows": rows,
        "prediction_rows": prediction_rows,
        "v9_scores": v9_scores,
        "bbox_span": bbox_span,
        "route": route,
        "no_output_retry_used": retried,
        "representative_repairs": repairs,
        "short_fragment_prunes": prunes,
        "seed_family_coverage_recoveries": coverage,
        "paired_plane_fold_recoveries": pair_recoveries,
        "tilted_adjacency_repairs": tilted_repairs,
    }


def build_result(pool: dict, selection: dict) -> dict:
    rows = selection["rows"]
    prediction_rows = selection["prediction_rows"]
    scores = selection["v9_scores"]
    bbox_span = float(selection["bbox_span"])
    route = str(selection["route"])
    predictions = []
    for row in prediction_rows:
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        source_index = int(row.get("_selector_source_index", rows.index(row) if row in rows else 0))
        score = _coverage_score(row, bbox_span) if route == ROUTE_SMALL else float(scores[source_index])
        predictions.append(
            {
                "bend_id": f"TRV16-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": route,
                "selector_score": round(score, 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_source_only_unified_mesh_route_selection",
        "profile": PROFILE,
        "route": route,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "no_output_retry_used": bool(selection["no_output_retry_used"]),
        "initial_threshold": SCORE_THRESHOLD,
        "nms_angle_deg": NMS_ANGLE_DEG if route == ROUTE_COMPLEX else None,
        "nms_distance_mm": NMS_DISTANCE_MM if route == ROUTE_COMPLEX else None,
        "seed_family_coverage_recoveries": selection["seed_family_coverage_recoveries"],
        "paired_plane_fold_recoveries": selection["paired_plane_fold_recoveries"],
        "tilted_adjacency_repairs": selection["tilted_adjacency_repairs"],
        "representative_repairs": selection["representative_repairs"],
        "short_fragment_prunes": selection["short_fragment_prunes"],
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    selection = select_unified(pool, joblib.load(args.v9_model), joblib.load(args.v4_model))
    result = build_result(pool, selection)
    result["candidate_pool_sha256"] = _sha(args.input)
    result["v9_model_sha256"] = _sha(args.v9_model)
    result["v4_model_sha256"] = _sha(args.v4_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": result["candidate_count"], "route": result["route"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
