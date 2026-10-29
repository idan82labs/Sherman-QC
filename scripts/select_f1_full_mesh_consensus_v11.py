#!/usr/bin/env python3
"""Apply V10 plus bounded cylindrical-only short-fragment pruning.

The pruning rule is count-free and source-only. It removes a selected line
only when every supporting family is cylindrical and its finite span is less
than 45% of the median span of the selected set. It cannot add bend lines.
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
from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives
from scripts.select_f1_topology_plane_coverage_v6 import select as select_v6
from scripts.train_f1_topology_recall_v4 import ADJACENCY_SOURCE, PROFILE as V4_PROFILE


PROFILE = "F1_FULL_RESOLUTION_MESH_CONSENSUS_SHORT_FRAGMENT_V11"
CYLINDRICAL_SOURCES = frozenset(("cylindrical_standard", "cylindrical_wide"))
MAX_MEDIAN_SPAN_FRACTION = 0.45


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prune_short_cylindrical_fragments(rows: list[dict], selected: list[int]) -> tuple[list[int], list[dict]]:
    """Remove only short lines supported exclusively by cylindrical aliases."""

    spans = np.asarray([float(rows[index].get("span_mm") or 0.0) for index in selected])
    median_span = float(np.median(spans)) if len(spans) else 0.0
    kept: list[int] = []
    changes: list[dict] = []
    for index in selected:
        row = rows[index]
        sources = frozenset(str(value) for value in row.get("cluster_sources") or ())
        span_mm = float(row.get("span_mm") or 0.0)
        cylindrical_only = bool(sources) and sources.issubset(CYLINDRICAL_SOURCES)
        short_fragment = median_span > 0.0 and span_mm < MAX_MEDIAN_SPAN_FRACTION * median_span
        if cylindrical_only and short_fragment:
            changes.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "span_mm": round(span_mm, 12),
                    "median_selected_span_mm": round(median_span, 12),
                    "median_span_fraction": round(span_mm / median_span, 12),
                    "cluster_sources": sorted(sources),
                }
            )
        else:
            kept.append(index)
    return kept, changes


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
        selected, repairs = repair_full_mesh_representatives(rows, selected)
        selected, pruned = prune_short_cylindrical_fragments(rows, selected)
        route = "full_resolution_mesh_v6_consensus_short_fragment"
    else:
        repairs, pruned = [], []
        route = "non_full_mesh_v6_unchanged"
    predictions = []
    for index in selected:
        row = rows[index]
        line = endpoints(row)
        source_id = str(row["candidate_id"])
        digest = hashlib.sha256((PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
        predictions.append(
            {
                "bend_id": f"TRV11-{digest}",
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
        "kind": "sherman_qc_f1_full_mesh_consensus_short_fragment_selection",
        "profile": PROFILE,
        "route": route,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "line_object_count": len(rows),
        "representative_repairs": repairs,
        "short_fragment_prunes": pruned,
        "candidate_pool_sha256": _sha(args.input),
        "v4_model_sha256": _sha(args.v4_model),
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "repair_count": len(repairs), "prune_count": len(pruned), "route": route}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
