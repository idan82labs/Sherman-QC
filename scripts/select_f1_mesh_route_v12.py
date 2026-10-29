#!/usr/bin/env python3
"""Route native and bounded meshes to their proven source-only selectors.

V11 is retained for full-resolution meshes.  Bounded-decimation meshes use
the V9 complexity router, followed by one narrow representative repair: a
selected cylindrical-only axis may be replaced by a nearby multi-source
scan-mesh-adjacency axis with strong two-sided transition evidence.  The
repair cannot add a bend or use part identity, counts, labels, CAD, or gold.
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

from research.f1_multifamily_selector_v2.model import (
    FEATURE_NAMES,
    build_feature_matrix,
    endpoints,
    select_indices_configurable,
)
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from scripts.select_f1_bounded_mesh_router_v9 import route as route_v9
from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives
from scripts.select_f1_full_mesh_consensus_v11 import prune_short_cylindrical_fragments
from scripts.select_f1_topology_bounded_mesh_v7 import _coverage_score, select_bounded_mesh
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_bounded_mesh_router_v9 import (
    NMS_ANGLE,
    NMS_DISTANCE,
    NMS_OVERLAP,
    PROFILE as V9_PROFILE,
    SCORE_THRESHOLD,
)
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


PROFILE = "F1_SOURCE_ONLY_MESH_ROUTE_V12"
CYLINDRICAL_SOURCES = frozenset(("cylindrical_standard", "cylindrical_wide"))
REPAIR_MAX_ANGLE_DEG = 4.0
REPAIR_MAX_DISTANCE_MM = 5.0
REPAIR_MIN_SHORTER_OVERLAP = 0.80
REPAIR_MIN_SOURCE_COUNT = 3
REPAIR_MIN_TRANSITION_SCORE = 0.85


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_bounded_cylindrical_representatives(
    rows: list[dict], selected: list[int]
) -> tuple[list[dict], list[dict]]:
    """Move cylindrical-only axes halfway toward nearby strong mesh consensus.

    The midpoint is deliberately conservative: the source axis can move no
    more than half of the five-millimetre evidence-neighbourhood threshold.
    """

    repaired: list[dict] = []
    changes: list[dict] = []
    for index in selected:
        row = rows[index]
        sources = frozenset(str(value) for value in row.get("cluster_sources") or ())
        if not sources or not sources.issubset(CYLINDRICAL_SOURCES):
            repaired.append(dict(row))
            continue
        candidates: list[tuple[float, float, float, int]] = []
        for candidate_index, candidate in enumerate(rows):
            if candidate.get("representative_source") != ADJACENCY_SOURCE:
                continue
            if int(candidate.get("cluster_source_count") or 0) < REPAIR_MIN_SOURCE_COUNT:
                continue
            transition_score = float(candidate.get("two_sided_transition_score") or 0.0)
            if transition_score < REPAIR_MIN_TRANSITION_SCORE:
                continue
            metrics = pair_metrics(endpoints(row), endpoints(candidate))
            if (
                metrics["axis_angle_deg"] <= REPAIR_MAX_ANGLE_DEG
                and metrics["transverse_offset_mm"] <= REPAIR_MAX_DISTANCE_MM
                and metrics["shorter_span_overlap"] >= REPAIR_MIN_SHORTER_OVERLAP
            ):
                candidates.append(
                    (
                        float(metrics["transverse_offset_mm"]),
                        float(metrics["axis_angle_deg"]),
                        -transition_score,
                        candidate_index,
                    )
                )
        replacement = sorted(candidates)[0][3] if candidates else None
        if replacement is None:
            repaired.append(dict(row))
        else:
            consensus = rows[replacement]
            source_line = endpoints(row)
            consensus_line = endpoints(consensus)
            direct = float(np.linalg.norm(source_line[0] - consensus_line[0]) + np.linalg.norm(source_line[1] - consensus_line[1]))
            reversed_distance = float(np.linalg.norm(source_line[0] - consensus_line[1]) + np.linalg.norm(source_line[1] - consensus_line[0]))
            if reversed_distance < direct:
                consensus_line = consensus_line[::-1]
            midpoint_line = 0.5 * (source_line + consensus_line)
            midpoint = dict(row)
            midpoint["candidate_id"] = f"consensus_midpoint::{row['candidate_id']}::{consensus['candidate_id']}"
            midpoint["span_endpoints"] = midpoint_line.tolist()
            midpoint["representative_source"] = "source_consensus_midpoint"
            midpoint["cluster_sources"] = sorted(
                set(row.get("cluster_sources") or ()) | set(consensus.get("cluster_sources") or ())
            )
            midpoint["cluster_source_count"] = len(midpoint["cluster_sources"])
            midpoint["_selector_source_index"] = index
            repaired.append(midpoint)
            changes.append(
                {
                    "from_candidate_id": str(row["candidate_id"]),
                    "toward_candidate_id": str(consensus["candidate_id"]),
                    "to_candidate_id": str(midpoint["candidate_id"]),
                    "repair_basis": "conservative_midpoint_toward_nearby_multisource_scan_mesh_transition_consensus",
                }
            )
    return repaired, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    v9 = joblib.load(args.v9_model)
    v4 = joblib.load(args.v4_model)
    if v9.get("profile") != V9_PROFILE or tuple(v9.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V9 model bundle does not match the frozen feature schema")
    if v4.get("profile") != V4_PROFILE or tuple(v4.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")

    v9_scores = v9["estimator"].predict_proba(features)[:, 1]
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

    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    mesh_available = bool(diagnostics.get("available"))
    bounded_mesh = bool(diagnostics.get("bounded_decimation_used"))
    bbox_span = max(float(value) for value in (pool.get("scan_bbox_mm") or (1.0,)))
    repairs: list[dict] = []
    prunes: list[dict] = []

    if mesh_available and not bounded_mesh:
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        selected, repairs = repair_full_mesh_representatives(rows, selected)
        selected, prunes = prune_short_cylindrical_fragments(rows, selected)
        selected_route = "full_resolution_mesh_v11"
    elif bounded_mesh:
        selected_route = route_v9(bounded_mesh=True, line_object_count=len(rows))
        if selected_route == "bounded_mesh_small_pool_recall":
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
        prediction_rows, repairs = repair_bounded_cylindrical_representatives(rows, selected)
        selected_route += "_consensus_repair"
    else:
        selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)
        selected_route = "non_mesh_v6"

    if not bounded_mesh:
        prediction_rows = [rows[index] for index in selected]
    predictions = []
    for row in prediction_rows:
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        score_index = int(row.get("_selector_source_index", rows.index(row) if row in rows else 0))
        score = (
            _coverage_score(row, bbox_span)
            if selected_route.startswith("bounded_mesh_small_pool_recall")
            else float(v9_scores[score_index])
        )
        predictions.append(
            {
                "bend_id": f"TRV12-{digest}",
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
        "kind": "sherman_qc_f1_source_only_mesh_route_selection",
        "profile": PROFILE,
        "route": selected_route,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "representative_repairs": repairs,
        "short_fragment_prunes": prunes,
        "candidate_pool_sha256": _sha(args.input),
        "v9_model_sha256": _sha(args.v9_model),
        "v4_model_sha256": _sha(args.v4_model),
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "repair_count": len(repairs), "prune_count": len(prunes), "route": selected_route}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
