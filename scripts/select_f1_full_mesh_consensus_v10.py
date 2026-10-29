#!/usr/bin/env python3
"""Apply V9 with a bounded full-resolution-mesh representative repair.

V9 treated ``bounded_decimation_used`` as the mesh-presence flag.  Native
meshes below the point budget therefore fell through to V6.  V10 preserves
that precision-first V6 selection, then replaces only a selected singleton
scan-mesh representative with a nearby, span-overlapping multi-source line.
All evidence is source-derived and the selected bend count cannot increase.
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
from scripts.evaluate_f1_against_manual_gold import pair_metrics
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


PROFILE = "F1_FULL_RESOLUTION_MESH_CONSENSUS_V10"
MAX_ANGLE_DEG = 12.0
MAX_DISTANCE_MM = 5.0
MIN_SHORTER_OVERLAP = 0.80
MIN_CONSENSUS_SOURCES = 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_full_mesh_representatives(rows: list[dict], selected: list[int]) -> tuple[list[int], list[dict]]:
    """Replace only isolated scan-mesh axes with nearby consensus axes."""

    repaired: list[int] = []
    changes: list[dict] = []
    for index in selected:
        row = rows[index]
        if (
            row.get("representative_source") != ADJACENCY_SOURCE
            or int(row.get("cluster_source_count") or 0) > 1
        ):
            repaired.append(index)
            continue
        source_line = endpoints(row)
        candidates = []
        for candidate_index, candidate in enumerate(rows):
            source_count = int(candidate.get("cluster_source_count") or 0)
            if source_count < MIN_CONSENSUS_SOURCES:
                continue
            metrics = pair_metrics(source_line, endpoints(candidate))
            if (
                metrics["axis_angle_deg"] <= MAX_ANGLE_DEG
                and metrics["transverse_offset_mm"] <= MAX_DISTANCE_MM
                and metrics["shorter_span_overlap"] >= MIN_SHORTER_OVERLAP
            ):
                candidates.append(
                    (
                        -source_count,
                        float(metrics["axis_angle_deg"]),
                        float(metrics["transverse_offset_mm"]),
                        candidate_index,
                    )
                )
        replacement = sorted(candidates)[0][3] if candidates else index
        repaired.append(replacement)
        if replacement != index:
            changes.append(
                {
                    "from_candidate_id": str(row["candidate_id"]),
                    "to_candidate_id": str(rows[replacement]["candidate_id"]),
                }
            )
    return list(dict.fromkeys(repaired)), changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in pool["line_objects"]]
    features = build_feature_matrix(rows)
    v4 = joblib.load(args.v4_model)
    if v4.get("profile") != V4_PROFILE or tuple(v4.get("feature_names") or ()) != FEATURE_NAMES:
        raise RuntimeError("V4 model bundle does not match the frozen feature schema")
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
    selected, _v5, _v6 = select_v6(rows, core_scores, adjacency_scores)

    diagnostics = dict(pool.get("generator_diagnostics") or {}).get("scan_mesh_adjacency") or {}
    full_mesh = bool(diagnostics.get("available")) and not bool(diagnostics.get("bounded_decimation_used"))
    if full_mesh:
        selected, changes = repair_full_mesh_representatives(rows, selected)
        route = "full_resolution_mesh_v6_consensus_repair"
    else:
        changes = []
        route = "non_full_mesh_v6_unchanged"

    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        predictions.append(
            {
                "bend_id": f"TRV10-{digest}",
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": row.get("representative_source"),
                "selection_route": route,
            }
        )
    predictions.sort(key=lambda row: str(row["bend_id"]))
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_full_resolution_mesh_consensus_selection",
        "profile": PROFILE,
        "route": route,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "representative_repairs": changes,
        "candidate_pool_sha256": _sha(args.input),
        "v4_model_sha256": _sha(args.v4_model),
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "repair_count": len(changes), "route": route}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
