#!/usr/bin/env python3
"""Execute the cap-safe V42 sidecar on retained V16 real-scan pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_ordered_projective_gauss_transport import (
    OPGTContractError,
    canonical_result_bytes,
)
from backend.f1_single_shell_parent_bridge_v42 import select_unified_v42
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars
from scripts.run_f1_v41_v16_real_scan_compatibility import (
    _canonical,
    _sha_file,
    adapt_candidates,
    incumbent_first_merge,
    load_bounded_source,
)


PROFILE = "F1_V42_V16_REAL_SCAN_COMPATIBILITY_V1"
MEASUREMENT_TRIANGLE_BUDGET = 60_000
CONTRACT = {
    "profile": PROFILE,
    "measurement_mesh": "deterministic_quadric_decimation_of_retained_bounded_mesh",
    "measurement_triangle_budget": MEASUREMENT_TRIANGLE_BUDGET,
    "measurement_budget_reason": "strictly_below_frozen_65536_local_support_cap",
    "topology_mesh": "retained_pool_builder_global_point_budget_v1",
    "candidate_family": "rejected_plane_lattice_only",
    "parent_geometry": "candidate_pool.plane_table",
    "local_scale": "candidate_pool.transition_local_spacing_mm",
    "incumbent_policy": "all_retained_v16_predictions_first",
    "labels_used": False,
    "expected_counts_used": False,
    "cad_used": False,
    "identity_used": False,
}


def measurement_source(
    topology_source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int | bool]]:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(topology_source["positions"], dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(topology_source["triangles"], dtype=np.int32)
    )
    before_vertices = int(len(mesh.vertices))
    before_triangles = int(len(mesh.triangles))
    decimated = before_triangles > MEASUREMENT_TRIANGLE_BUDGET
    if decimated:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=MEASUREMENT_TRIANGLE_BUDGET
        )
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) > MEASUREMENT_TRIANGLE_BUDGET:
        raise RuntimeError("V42 measurement mesh exceeds frozen triangle budget")
    return {
        "positions": positions.tolist(),
        "triangles": triangles.tolist(),
    }, {
        "input_vertex_count": before_vertices,
        "input_triangle_count": before_triangles,
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "decimation_used": decimated,
    }


def run_case(
    *, pool_path: Path, retained_path: Path, scan_path: Path, output_path: Path,
    selector=select_unified_v42,
    contract: dict[str, Any] = CONTRACT,
    profile: str = PROFILE,
    result_label: str = "v42",
) -> dict[str, Any]:
    started = time.perf_counter()
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    candidates, mappings = adapt_candidates(pool, retained)
    mapping_by_id = {int(value["candidate_id"]): value for value in mappings}
    contract_sha = hashlib.sha256(_canonical(contract)).hexdigest()
    results: list[dict[str, Any]] = []
    result_hashes: list[str] = []
    additions: list[dict[str, Any]] = []
    resource_stops: list[dict[str, Any]] = []

    if candidates:
        topology, topology_replay = load_bounded_source(scan_path, pool)
        measurement, measurement_replay = measurement_source(topology)

        def analyze_batch(indices: list[int]) -> None:
            try:
                result = selector(
                    measurement, topology,
                    [candidates[index] for index in indices],
                    config_sha256=contract_sha,
                    worker_count=1,
                )
            except OPGTContractError as exc:
                if len(indices) > 1:
                    midpoint = len(indices) // 2
                    analyze_batch(indices[:midpoint])
                    analyze_batch(indices[midpoint:])
                    return
                mapping = mappings[indices[0]]
                resource_stops.append({
                    "candidate_id": mapping["candidate_id"],
                    "source_hypothesis_id": mapping["source_hypothesis_id"],
                    "token": exc.token,
                    "detail": exc.detail,
                })
                return
            public = normalize_numpy_scalars(
                {key: value for key, value in result.items() if key != "_ephemeral"}
            )
            results.append(public)
            result_hashes.append(hashlib.sha256(canonical_result_bytes(public)).hexdigest())
            additions.extend(
                dict(value) for value in result.get("final_lines") or ()
                if value.get("source_lane") != "legacy"
            )

        analyze_batch(list(range(len(candidates))))
    else:
        topology_replay = None
        measurement_replay = None

    predictions, merge_audit = incumbent_first_merge(
        list(retained.get("predictions") or ()), additions, mapping_by_id
    )
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v42_v16_real_scan_compatibility_result",
        "profile": profile,
        "route": retained.get("route"),
        "contract": contract,
        "contract_sha256": contract_sha,
        "candidate_pool_sha256": _sha_file(pool_path),
        "retained_v16_sha256": _sha_file(retained_path),
        "scan_sha256": _sha_file(scan_path),
        "candidate_mappings": mappings,
        "compatible_rejected_candidate_count": len(candidates),
        "topology_mesh_replay": topology_replay,
        "measurement_mesh_replay": measurement_replay,
        f"{result_label}_results": results,
        f"{result_label}_result_sha256": result_hashes,
        f"{result_label}_resource_stops": resource_stops,
        f"{result_label}_resource_stop_count": len(resource_stops),
        f"{result_label}_pre_merge_addition_count": len(additions),
        "incumbent_first_merge_audit": merge_audit,
        f"{result_label}_post_merge_addition_count": sum(value["accepted"] for value in merge_audit),
        "retained_v16_prediction_count": len(retained.get("predictions") or ()),
        "candidate_count": len(predictions),
        "predictions": predictions,
        "research_only": True,
        "source_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "part_identity_used": False,
        "cad_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(normalize_numpy_scalars(payload)) + b"\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(
        pool_path=args.pool.resolve(), retained_path=args.retained.resolve(),
        scan_path=args.scan.resolve(), output_path=args.output.resolve(),
    )
    print(json.dumps({
        "compatible_rejected_candidate_count": result["compatible_rejected_candidate_count"],
        "v42_resource_stop_count": result["v42_resource_stop_count"],
        "v42_pre_merge_addition_count": result["v42_pre_merge_addition_count"],
        "v42_post_merge_addition_count": result["v42_post_merge_addition_count"],
        "candidate_count": result["candidate_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
