#!/usr/bin/env python3
"""Apply V5 plus a bounded plane-lattice-only dead-zone route.

V6 changes exactly one behavior: if V5 remains empty and the pool contains
only strong, fully supported plane intersections, select those intersections
while suppressing near-parallel aliases.  No expected count or identity enters
the decision.
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
from scripts.select_f1_topology_coverage_v5 import select as select_v5
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


PROFILE = "F1_TOPOLOGY_PLANE_DEAD_ZONE_COVERAGE_V6"
PLANE_SOURCE = "plane_lattice"
MIN_TRANSITION_SCORE = 0.50
MIN_FINITE_SUPPORT = 0.95


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_eligible(row: dict) -> bool:
    return (
        str(row.get("representative_source") or "") == PLANE_SOURCE
        and float(row.get("two_sided_transition_score") or 0.0) >= MIN_TRANSITION_SCORE
        and float(row.get("finite_support_fraction") or 0.0) >= MIN_FINITE_SUPPORT
    )


def _plane_similar(first: dict, second: dict) -> bool:
    first_line = endpoints(first)
    second_line = endpoints(second)
    first_axis = first_line[1] - first_line[0]
    second_axis = second_line[1] - second_line[0]
    first_axis /= np.linalg.norm(first_axis)
    second_axis /= np.linalg.norm(second_axis)
    angle = np.degrees(np.arccos(np.clip(abs(float(first_axis @ second_axis)), -1.0, 1.0)))
    if angle > 2.0:
        return False
    first_mid = first_line.mean(axis=0)
    second_mid = second_line.mean(axis=0)
    delta = second_mid - first_mid
    first_distance = np.linalg.norm(delta - float(delta @ first_axis) * first_axis)
    second_distance = np.linalg.norm(delta - float(delta @ second_axis) * second_axis)
    if max(first_distance, second_distance) > 12.0:
        return False
    first_extent = np.sort((first_line - first_mid) @ first_axis)
    second_extent = np.sort((second_line - first_mid) @ first_axis)
    overlap = max(0.0, min(first_extent[1], second_extent[1]) - max(first_extent[0], second_extent[0]))
    return overlap / min(float(np.ptp(first_extent)), float(np.ptp(second_extent))) >= 0.80


def select(rows: list[dict], core_scores: np.ndarray, adjacency_scores: np.ndarray) -> tuple[list[int], set[int], set[int]]:
    accepted, v5_coverage = select_v5(rows, core_scores, adjacency_scores)
    if accepted:
        return accepted, v5_coverage, set()
    additions = [index for index, row in enumerate(rows) if _plane_eligible(row)]
    additions.sort(
        key=lambda index: (
            -float(rows[index].get("two_sided_transition_score") or 0.0),
            -float(rows[index].get("finite_support_fraction") or 0.0),
            str(rows[index].get("candidate_id") or ""),
        )
    )
    for index in additions:
        if all(not _plane_similar(rows[index], rows[prior]) for prior in accepted):
            accepted.append(index)
    return accepted, v5_coverage, set(accepted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    bundle = joblib.load(args.model)
    if bundle.get("profile") != V4_PROFILE or tuple(bundle.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")
    core_scores = bundle["core_estimator"].predict_proba(features)[:, 1]
    adjacency_scores = np.zeros(len(rows), dtype=float)
    adjacency_indices = [
        index for index, row in enumerate(rows)
        if row.get("representative_source") == ADJACENCY_SOURCE
    ]
    if adjacency_indices:
        adjacency_scores[adjacency_indices] = bundle["adjacency_estimator"].predict_proba(
            features[adjacency_indices]
        )[:, 1]
    selected, v5_coverage, plane_coverage = select(rows, core_scores, adjacency_scores)

    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        if index in plane_coverage:
            route = "source_only_plane_dead_zone_coverage"
            score = float(row.get("two_sided_transition_score") or 0.0)
        elif index in v5_coverage:
            route = "source_only_dead_zone_coverage"
            score = float(row.get("cluster_source_count") or len(row.get("cluster_sources") or ()))
        elif row.get("representative_source") == ADJACENCY_SOURCE and adjacency_scores[index] >= bundle["adjacency_threshold"]:
            route = "adjacency_recall"
            score = float(adjacency_scores[index])
        else:
            route = "v3_precision_core"
            score = float(core_scores[index])
        predictions.append(
            {
                "bend_id": f"TRV6-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": route,
                "selector_score": round(score, 12),
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_topology_plane_dead_zone_coverage_selection",
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "candidate_pool_sha256": sha(args.input),
        "model_sha256": sha(args.model),
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "c006_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(predictions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
