#!/usr/bin/env python3
"""Run the research-only F1 union-graph relational baseline.

This command bypasses the v23 global lane router.  It consumes only the
strict-blind scan candidate pool, emits the full typed audit graph, and adapts
counted paths to the existing finite-line prediction schema.  It is not the
future exact CP-SAT solver and must not be promoted to runtime/main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_structured_union_graph import (  # noqa: E402
    BEND_STRAIGHT_COUNTED,
    RESIDUAL_NONCOUNTED,
    solve_structured_union_baseline,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MappingLike = dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    candidate_pool_path = args.candidate_pool.resolve()
    candidate_pool = _read(candidate_pool_path)
    solution = solve_structured_union_baseline(candidate_pool)

    counted = [
        {
            "bend_id": str(path["latent_path_id"]),
            "span_endpoints": path["span_endpoints"],
            "axis_direction": path["axis_direction"],
            "span_mm": path["span_mm"],
            "semantic_label": path["semantic_label"],
            "incident_panel_pair": path["primary_panel_pair"],
            "independent_source_groups": path["independent_source_groups"],
        }
        for path in solution["latent_paths"]
        if path["semantic_label"] == BEND_STRAIGHT_COUNTED
    ]
    audit_objects = [
        path
        for path in solution["latent_paths"]
        if path["semantic_label"] != RESIDUAL_NONCOUNTED
    ]

    output_dir = args.output_dir.resolve()
    graph_path = output_dir / "typed_union_graph.json"
    prediction_path = output_dir / "prediction_owned_regions.json"
    _write(graph_path, solution)
    _write(
        prediction_path,
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_structured_union_counted_paths",
            "owned_bend_regions": counted,
        },
    )
    summary = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_structured_union_research_baseline",
        "algorithm": "f1_structured_union_relational_baseline_v1",
        "research_only": True,
        "runtime_promotion_authorized": False,
        "solver_status": "RELATIONAL_BASELINE_NOT_EXACT",
        "exact_optimality_proven": False,
        "external_inference_contract": solution["inference_contract"],
        "inputs": {
            "candidate_pool_sha256": _sha256(candidate_pool_path),
        },
        "graph_sha256": solution["graph_sha256"],
        "latent_path_count": len(solution["latent_paths"]),
        "typed_audit_object_count": len(audit_objects),
        "counted_bend_component_count": int(solution["bend_count"]),
        "topology_factor_count": len(solution["topology_factors"]),
        "prediction_path": str(prediction_path),
        "typed_graph_path": str(graph_path),
    }
    _write(output_dir / "inference_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

