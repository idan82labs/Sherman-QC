#!/usr/bin/env python3
"""V13 plus continuous long seed-family coverage after a no-output retry."""

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

from research.f1_multifamily_selector_v2.model import build_feature_matrix, endpoints
from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives
from scripts.select_f1_mesh_route_v13 import RETRY_THRESHOLD, select_with_no_output_retry
from scripts.train_f1_bounded_mesh_router_v9 import SCORE_THRESHOLD


PROFILE = "F1_BOUNDED_MESH_CONTINUOUS_SEED_COVERAGE_V14"
MIN_SPAN_FRACTION = 0.70
MIN_SELECTOR_SCORE = 0.15
MIN_CLUSTER_SOURCES = 2
MIN_SUPPORT_FRACTION = 0.90
MIN_NORMAL_ANGLE_DEG = 45.0
MIN_ANGULAR_BIN_FRACTION = 0.75


def recover_continuous_long_seed_families(
    rows: list[dict], scores, selected: list[int], bbox_span: float
) -> tuple[list[int], list[dict]]:
    """Add one fully supported long axis for each unrepresented seed family."""

    represented = {
        tuple(str(value) for value in rows[index].get("seed_ids") or ())
        for index in selected
        if rows[index].get("seed_ids")
    }
    eligible: list[int] = []
    for index, row in enumerate(rows):
        seed_family = tuple(str(value) for value in row.get("seed_ids") or ())
        if not seed_family or seed_family in represented:
            continue
        if float(row.get("span_mm") or 0.0) < MIN_SPAN_FRACTION * float(bbox_span):
            continue
        if float(scores[index]) < MIN_SELECTOR_SCORE:
            continue
        if int(row.get("cluster_source_count") or 0) < MIN_CLUSTER_SOURCES:
            continue
        support_values = (
            float(row.get("finite_support_fraction") or 0.0),
            float(row.get("finite_support_longest_run_fraction") or 0.0),
            float(row.get("endpoint_support_fraction") or 0.0),
        )
        if min(support_values) < MIN_SUPPORT_FRACTION:
            continue
        if float(row.get("two_sided_normal_angle_deg") or 0.0) < MIN_NORMAL_ANGLE_DEG:
            continue
        if float(row.get("transverse_angular_bin_fraction") or 0.0) < MIN_ANGULAR_BIN_FRACTION:
            continue
        eligible.append(index)

    additions: list[int] = []
    changes: list[dict] = []
    for seed_family in sorted({tuple(str(value) for value in rows[index].get("seed_ids") or ()) for index in eligible}):
        members = [
            index for index in eligible
            if tuple(str(value) for value in rows[index].get("seed_ids") or ()) == seed_family
        ]
        chosen = sorted(
            members,
            key=lambda index: (
                -float(scores[index]),
                -float(rows[index].get("span_mm") or 0.0),
                str(rows[index].get("candidate_id") or ""),
            ),
        )[0]
        additions.append(chosen)
        changes.append(
            {
                "candidate_id": str(rows[chosen]["candidate_id"]),
                "seed_family": list(seed_family),
                "selector_score": round(float(scores[chosen]), 12),
                "span_fraction": round(float(rows[chosen]["span_mm"]) / float(bbox_span), 12),
                "repair_basis": "continuous_long_unrepresented_seed_family_after_no_output_retry",
            }
        )
    return list(selected) + additions, changes


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
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    coverage: list[dict] = []
    if retried and selected:
        selected, coverage = recover_continuous_long_seed_families(rows, scores, selected, bbox_span)
    prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
    predictions = []
    for row in prediction_rows:
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        source_index = int(row.get("_selector_source_index", rows.index(row) if row in rows else 0))
        predictions.append(
            {
                "bend_id": f"TRV14-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": "bounded_mesh_no_output_retry_continuous_seed_coverage" if retried else "bounded_mesh_complex_pool_precision",
                "selector_score": round(float(scores[source_index]), 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bounded_mesh_continuous_seed_coverage_selection",
        "profile": PROFILE,
        "route": "bounded_mesh_no_output_retry_continuous_seed_coverage" if retried else "bounded_mesh_complex_pool_precision",
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "no_output_retry_used": retried,
        "initial_threshold": SCORE_THRESHOLD,
        "retry_threshold": RETRY_THRESHOLD,
        "seed_family_coverage_recoveries": coverage,
        "representative_repairs": repairs,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "coverage_count": len(coverage), "no_output_retry_used": retried, "repair_count": len(repairs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
