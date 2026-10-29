#!/usr/bin/env python3
"""Apply the source-only V9 complexity router without count or identity input."""

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

from research.f1_multifamily_selector_v2.model import FEATURE_NAMES, build_feature_matrix, endpoints, select_indices_configurable
from scripts.select_f1_topology_bounded_mesh_v7 import _coverage_score, select_bounded_mesh
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_bounded_mesh_router_v9 import (
    NMS_ANGLE,
    NMS_DISTANCE,
    NMS_OVERLAP,
    PROFILE,
    SCORE_THRESHOLD,
    SMALL_POOL_MAXIMUM,
)
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


def route(*, bounded_mesh: bool, line_object_count: int) -> str:
    if not bounded_mesh:
        return "non_mesh_v6"
    if line_object_count <= SMALL_POOL_MAXIMUM:
        return "bounded_mesh_small_pool_recall"
    return "bounded_mesh_complex_pool_precision"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    bundle = joblib.load(args.model)
    if bundle.get("profile") != PROFILE or tuple(bundle.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V9 model bundle does not match the frozen feature schema")
    v4 = joblib.load(args.v4_model)
    if v4.get("profile") != V4_PROFILE or tuple(v4.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")

    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    bounded_mesh = bool(diagnostics.get("bounded_decimation_used"))
    selected_route = route(bounded_mesh=bounded_mesh, line_object_count=len(rows))
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    v9_scores = bundle["estimator"].predict_proba(features)[:, 1]
    core_scores = v4["core_estimator"].predict_proba(features)[:, 1]
    adjacency_scores = np.zeros(len(rows), dtype=float)
    adjacency_indices = [
        index for index, row in enumerate(rows)
        if row.get("representative_source") == ADJACENCY_SOURCE
    ]
    if adjacency_indices:
        adjacency_scores[adjacency_indices] = v4["adjacency_estimator"].predict_proba(
            features[adjacency_indices]
        )[:, 1]

    if selected_route == "non_mesh_v6":
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
    elif selected_route == "bounded_mesh_small_pool_recall":
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

    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        score = _coverage_score(row, bbox_span) if selected_route == "bounded_mesh_small_pool_recall" else float(v9_scores[index])
        predictions.append(
            {
                "bend_id": f"TRV9-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": selected_route,
                "selector_score": round(score, 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bounded_mesh_complexity_router_selection",
        "profile": PROFILE,
        "route": selected_route,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "bounded_mesh_route": bounded_mesh,
        "candidate_pool_sha256": _sha(args.input),
        "model_sha256": _sha(args.model),
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "route": selected_route}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
