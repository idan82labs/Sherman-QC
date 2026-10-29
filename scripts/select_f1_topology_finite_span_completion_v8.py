#!/usr/bin/env python3
"""Apply V7 plus a bounded cross-source finite-span completion tranche.

V8 adds only hypotheses independently supported by at least three source
generators and by an almost-complete finite scan corridor.  Additions are
geometrically de-duplicated against V7 and each other.  No identity, expected
count, CAD, or labels enter inference.
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
from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_topology_recall_v4 import (
    ADJACENCY_SOURCE,
    PROFILE as V4_PROFILE,
    _similar,
)


PROFILE = "F1_TOPOLOGY_FINITE_SPAN_COMPLETION_V8"
MIN_SOURCE_COUNT = 3
MIN_FINITE_SUPPORT = 0.95
REQUIRED_INDEPENDENT_SOURCE = "strict_blind_owned"
DISALLOWED_COMPLETION_REPRESENTATIVES = frozenset({"fragment_merge"})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_count(row: dict) -> int:
    sources = set(str(value) for value in (row.get("cluster_sources") or ()))
    return int(row.get("cluster_source_count") or len(sources))


def _eligible(row: dict) -> bool:
    sources = set(str(value) for value in (row.get("cluster_sources") or ()))
    return (
        str(row.get("representative_source") or "") not in DISALLOWED_COMPLETION_REPRESENTATIVES
        and
        REQUIRED_INDEPENDENT_SOURCE in sources
        and _source_count(row) >= MIN_SOURCE_COUNT
        and float(row.get("finite_support_fraction") or 0.0) >= MIN_FINITE_SUPPORT
        and float(row.get("finite_support_longest_run_fraction") or 0.0) >= MIN_FINITE_SUPPORT
    )


def _addition_key(row: dict) -> tuple:
    return (
        -_source_count(row),
        -int(row.get("cluster_member_count") or 0),
        -float(row.get("two_sided_transition_score") or 0.0),
        -float(row.get("case_percentile_score") or 0.0),
        str(row.get("candidate_id") or ""),
    )


def select(
    rows: list[dict],
    bbox_span: float,
    *,
    bounded_mesh_route: bool = True,
    core_scores: np.ndarray | None = None,
    adjacency_scores: np.ndarray | None = None,
) -> tuple[list[int], set[int]]:
    if bounded_mesh_route:
        accepted = select_bounded_mesh(rows, bbox_span)
    else:
        if core_scores is None or adjacency_scores is None:
            raise ValueError("non-mesh V8 selection requires V4 scores")
        accepted, _v5_coverage, _plane_coverage = select_v6(rows, core_scores, adjacency_scores)
        # The finite-span signal was validated only for decimated mesh pools.
        # On the 12 ordinary development pools it added 23 false positives and
        # no new true positives, so fail closed instead of exporting it across
        # an unsupported pool topology.
        return accepted, set()
    additions = [index for index, row in enumerate(rows) if _eligible(row) and index not in accepted]
    additions.sort(key=lambda index: _addition_key(rows[index]))
    added: set[int] = set()
    for index in additions:
        if all(not _similar(rows[index], rows[prior]) for prior in accepted):
            accepted.append(index)
            added.add(index)
    return accepted, added


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

    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    bounded_mesh_route = bool(diagnostics.get("bounded_decimation_used"))
    selected, additions = select(
        rows,
        bbox_span,
        bounded_mesh_route=bounded_mesh_route,
        core_scores=core_scores,
        adjacency_scores=adjacency_scores,
    )
    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        if index in additions:
            route = "source_only_cross_source_finite_span_completion"
            score = float(_source_count(row))
        elif row.get("representative_source") == ADJACENCY_SOURCE:
            route = "source_only_bounded_mesh_coverage"
            score = float(adjacency_scores[index])
        else:
            route = "source_only_bounded_mesh_coverage"
            score = float(core_scores[index])
        predictions.append(
            {
                "bend_id": f"TRV8-{digest}",
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
        "kind": "sherman_qc_f1_topology_finite_span_completion_selection",
        "profile": PROFILE,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "completion_count": len(additions),
        "bounded_mesh_route": bounded_mesh_route,
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
    print(json.dumps({"candidate_count": len(predictions), "completion_count": len(additions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
