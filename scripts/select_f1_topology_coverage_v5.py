#!/usr/bin/env python3
"""Apply V4 plus a bounded source-only dead-zone coverage route.

V5 changes exactly one behavior: when V4 returns no bend at all, admit sparse
cross-source/mesh-supported topology hypotheses and geometrically suppress
aliases.  It never uses a part identifier, expected count, CAD, or labels.
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
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE, _similar, select as select_v4


PROFILE = "F1_TOPOLOGY_DEAD_ZONE_COVERAGE_V5"
MANUAL_TOPOLOGY_SOURCES = {"bend_hypothesis", "control_support_strip"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_count(row: dict) -> int:
    sources = set(str(value) for value in (row.get("cluster_sources") or ()))
    return int(row.get("cluster_source_count") or len(sources))


def _coverage_eligible(row: dict) -> bool:
    sources = set(str(value) for value in (row.get("cluster_sources") or ()))
    representative = str(row.get("representative_source") or "")
    return (
        (ADJACENCY_SOURCE in sources and _source_count(row) >= 2)
        or (representative in MANUAL_TOPOLOGY_SOURCES and _source_count(row) >= 4)
    )


def _coverage_key(row: dict) -> tuple:
    return (
        -_source_count(row),
        -int(row.get("cluster_member_count") or 0),
        -float(row.get("case_percentile_two_sided_transition_score") or 0.0),
        str(row.get("candidate_id") or ""),
    )


def select(rows: list[dict], core_scores: np.ndarray, adjacency_scores: np.ndarray) -> tuple[list[int], set[int]]:
    accepted = select_v4(rows, core_scores, adjacency_scores)
    if accepted:
        return accepted, set()
    additions = [index for index, row in enumerate(rows) if _coverage_eligible(row)]
    additions.sort(key=lambda index: _coverage_key(rows[index]))
    for index in additions:
        if all(not _similar(rows[index], rows[prior]) for prior in accepted):
            accepted.append(index)
    return accepted, set(accepted)


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
    selected, coverage_indices = select(rows, core_scores, adjacency_scores)

    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        if index in coverage_indices:
            route = "source_only_dead_zone_coverage"
            score = float(_source_count(row))
        elif row.get("representative_source") == ADJACENCY_SOURCE and adjacency_scores[index] >= bundle["adjacency_threshold"]:
            route = "adjacency_recall"
            score = float(adjacency_scores[index])
        else:
            route = "v3_precision_core"
            score = float(core_scores[index])
        predictions.append(
            {
                "bend_id": f"TRV5-{digest}",
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
        "kind": "sherman_qc_f1_topology_dead_zone_coverage_selection",
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
