#!/usr/bin/env python3
"""V12 plus a guarded no-output retry for complex bounded meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints, select_indices_configurable
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.train_f1_bounded_mesh_router_v9 import NMS_ANGLE, NMS_DISTANCE, NMS_OVERLAP, SCORE_THRESHOLD


PROFILE = "F1_BOUNDED_MESH_NO_OUTPUT_RETRY_V13"
RETRY_THRESHOLD = 0.35


def select_with_no_output_retry(rows: list[dict], scores) -> tuple[list[int], bool]:
    selected = select_indices_configurable(
        rows,
        scores,
        threshold=SCORE_THRESHOLD,
        max_angle=NMS_ANGLE,
        max_distance=NMS_DISTANCE,
        min_overlap=NMS_OVERLAP,
    )
    retried = not selected
    if retried:
        selected = select_indices_configurable(
            rows,
            scores,
            threshold=RETRY_THRESHOLD,
            max_angle=NMS_ANGLE,
            max_distance=NMS_DISTANCE,
            min_overlap=NMS_OVERLAP,
        )
    return selected, retried


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    model = joblib.load(args.v9_model)
    scores = model["estimator"].predict_proba(build_feature_matrix(rows))[:, 1]
    selected, retried = select_with_no_output_retry(rows, scores)
    prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
    predictions = []
    for row in prediction_rows:
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        source_index = int(row.get("_selector_source_index", rows.index(row) if row in rows else 0))
        predictions.append(
            {
                "bend_id": f"TRV13-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": "bounded_mesh_complex_pool_no_output_retry" if retried else "bounded_mesh_complex_pool_precision",
                "selector_score": round(float(scores[source_index]), 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bounded_mesh_no_output_retry_selection",
        "profile": PROFILE,
        "route": "bounded_mesh_complex_pool_no_output_retry" if retried else "bounded_mesh_complex_pool_precision",
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "no_output_retry_used": retried,
        "initial_threshold": SCORE_THRESHOLD,
        "retry_threshold": RETRY_THRESHOLD,
        "representative_repairs": repairs,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "no_output_retry_used": retried, "repair_count": len(repairs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
