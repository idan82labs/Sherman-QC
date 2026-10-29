#!/usr/bin/env python3
"""Research-only V41 compatibility sidecar for retained V16 scan pools.

The adapter exposes only already-observed scan mesh geometry and rejected
plane-lattice candidates to the frozen V41 certificate.  Retained V16 lines
are immutable incumbents and suppress duplicate V41 additions.  No labels,
expected counts, CAD, filenames-as-features, or part identity are consumed by
the decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_fragment_tolerant_parent_bridge_v41 import select_unified_v41
from backend.f1_ordered_projective_gauss_transport import (
    OPGTContractError,
    canonical_result_bytes,
)
from backend.f1_v34_artifact_scalar_normalizer import normalize_numpy_scalars


PROFILE = "F1_V41_V16_REAL_SCAN_COMPATIBILITY_V1"
GLOBAL_POINT_BUDGET = 200_000
TARGET_TRIANGLE_BUDGET = int(GLOBAL_POINT_BUDGET * 1.5)
NMS_ANGLE_DEGREES = 7.0
NMS_DISTANCE_MM = 6.0
NMS_MINIMUM_OVERLAP = 0.60
CONTRACT = {
    "profile": PROFILE,
    "candidate_family": "rejected_plane_lattice_only",
    "local_scale": "candidate_pool.transition_local_spacing_mm",
    "parent_geometry": "candidate_pool.plane_table",
    "mesh_reduction": "retained_pool_builder_global_point_budget_v1",
    "global_point_budget": GLOBAL_POINT_BUDGET,
    "target_triangle_budget": TARGET_TRIANGLE_BUDGET,
    "incumbent_policy": "all_retained_v16_predictions_first",
    "nms_angle_degrees": NMS_ANGLE_DEGREES,
    "nms_distance_mm": NMS_DISTANCE_MM,
    "nms_minimum_overlap": NMS_MINIMUM_OVERLAP,
    "labels_used": False,
    "expected_counts_used": False,
    "cad_used": False,
    "identity_used": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _unit(value: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid vector")
    return array / norm


def _plane_centroid(plane: dict[str, Any]) -> list[float]:
    normal = _unit(plane["normal"])
    centroid = -float(plane["offset"]) * normal
    basis = np.asarray(plane.get("tangent_basis") or (), dtype=np.float64)
    extent = np.asarray(plane.get("extent_uv") or (), dtype=np.float64)
    if basis.shape == (2, 3) and extent.shape == (2, 2):
        centroid = centroid + float(np.mean(extent[0])) * basis[0]
        centroid = centroid + float(np.mean(extent[1])) * basis[1]
    return centroid.tolist()


def _raw_candidate(
    row: dict[str, Any],
    *,
    frozen_order: int,
    scale: float,
    planes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plane_ids = tuple(str(value) for value in row.get("incident_plane_ids") or ())
    if len(plane_ids) != 2 or any(value not in planes for value in plane_ids):
        raise ValueError(f"invalid incident planes for {row.get('candidate_id')}")
    start, end = np.asarray(row["span_endpoints"], dtype=np.float64)
    axis = _unit(end - start)
    length = float(np.linalg.norm(end - start))
    if length <= 0.0:
        raise ValueError(f"zero span for {row.get('candidate_id')}")
    parent_rows = [planes[value] for value in plane_ids]
    source_id = str(row["candidate_id"])
    numeric_id = _stable_u64(source_id)
    candidate = {
        "candidate_id": numeric_id,
        "frozen_v16_order": int(frozen_order),
        "selected_legacy": False,
        "rejected_before_final_nms": True,
        "render_admissible": True,
        "center_xyz": (0.5 * (start + end)).tolist(),
        "unit_axis_xyz": axis.tolist(),
        "segment_length_L": length,
        "local_scale_s": float(scale),
        "signed_turn_alpha": float(row["angle_deg"]),
        "parent_A": {
            "parent_id": _stable_u64(plane_ids[0]),
            "unit_normal_xyz": _unit(parent_rows[0]["normal"]).tolist(),
            "offset": float(parent_rows[0]["offset"]),
            "centroid_xyz": _plane_centroid(parent_rows[0]),
        },
        "parent_B": {
            "parent_id": _stable_u64(plane_ids[1]),
            "unit_normal_xyz": _unit(parent_rows[1]["normal"]).tolist(),
            "offset": float(parent_rows[1]["offset"]),
            "centroid_xyz": _plane_centroid(parent_rows[1]),
        },
        "render_line_start": start.tolist(),
        "render_line_end": end.tolist(),
        "score": None,
    }
    if row.get("two_sided_min_support_fraction") is not None:
        candidate["two_sided_min_support_fraction"] = float(
            row["two_sided_min_support_fraction"]
        )
    mapping = {
        "candidate_id": numeric_id,
        "source_hypothesis_id": source_id,
        "incident_plane_ids": list(plane_ids),
        "frozen_v16_order": int(frozen_order),
    }
    return candidate, mapping


def adapt_candidates(
    pool: dict[str, Any], retained: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = {
        str(value["source_hypothesis_id"])
        for value in retained.get("predictions") or ()
    }
    planes = {str(value["plane_id"]): dict(value) for value in pool.get("plane_table") or ()}
    scale = float(pool["transition_local_spacing_mm"])
    candidates: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index, row in enumerate(pool.get("line_objects") or ()):
        if row.get("representative_source") != "plane_lattice":
            continue
        if str(row.get("candidate_id")) in selected:
            continue
        candidate, mapping = _raw_candidate(
            dict(row), frozen_order=index, scale=scale, planes=planes
        )
        candidates.append(candidate)
        mappings.append(mapping)
    return candidates, mappings


def load_bounded_source(
    scan_path: Path, pool: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    source_vertices = int(len(mesh.vertices))
    source_triangles = int(len(mesh.triangles))
    diagnostics = dict(pool.get("generator_diagnostics") or {}).get(
        "scan_mesh_adjacency"
    ) or {}
    decimated = False
    if source_vertices > GLOBAL_POINT_BUDGET and source_triangles >= 10:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=min(source_triangles, TARGET_TRIANGLE_BUDGET)
        )
        decimated = True
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    observed = {
        "available": bool(len(positions) and len(triangles)),
        "source_vertex_count": source_vertices,
        "source_triangle_count": source_triangles,
        "vertex_count": int(len(positions)),
        "triangle_count": int(len(triangles)),
        "bounded_decimation_used": decimated,
    }
    expected = {
        key: diagnostics.get(key)
        for key in (
            "available", "source_vertex_count", "source_triangle_count",
            "vertex_count", "triangle_count", "bounded_decimation_used",
        )
    }
    if observed != expected:
        raise RuntimeError(f"bounded mesh replay drift: {observed} != {expected}")
    return {"positions": positions.tolist(), "triangles": triangles.tolist()}, observed


def _line_metrics(
    first_start: Iterable[float], first_end: Iterable[float],
    second_start: Iterable[float], second_end: Iterable[float],
) -> tuple[float, float, float]:
    a0, a1 = np.asarray(tuple(first_start), dtype=float), np.asarray(tuple(first_end), dtype=float)
    b0, b1 = np.asarray(tuple(second_start), dtype=float), np.asarray(tuple(second_end), dtype=float)
    axis_a, axis_b = a1 - a0, b1 - b0
    length_a, length_b = float(np.linalg.norm(axis_a)), float(np.linalg.norm(axis_b))
    if length_a <= 0.0 or length_b <= 0.0:
        return 180.0, math.inf, 0.0
    axis_a, axis_b = axis_a / length_a, axis_b / length_b
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))))
    delta = 0.5 * (b0 + b1) - 0.5 * (a0 + a1)
    distance = float(np.linalg.norm(delta - np.dot(delta, axis_a) * axis_a))
    along = abs(float(np.dot(delta, axis_a)))
    overlap = max(0.0, 0.5 * (length_a + length_b) - along) / min(length_a, length_b)
    return angle, distance, overlap


def incumbent_first_merge(
    retained_predictions: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    mapping_by_id: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = [dict(value) for value in retained_predictions]
    audit: list[dict[str, Any]] = []
    for addition in additions:
        duplicate_of = None
        metrics = None
        for prior in accepted:
            observed = _line_metrics(
                addition["line_start"], addition["line_end"],
                prior["line_start"], prior["line_end"],
            )
            if (
                observed[0] <= NMS_ANGLE_DEGREES
                and observed[1] <= NMS_DISTANCE_MM
                and observed[2] >= NMS_MINIMUM_OVERLAP
            ):
                duplicate_of = str(prior.get("source_hypothesis_id") or prior.get("bend_id"))
                metrics = observed
                break
        numeric_id = int(addition["candidate_id"])
        mapping = mapping_by_id[numeric_id]
        row = {
            "candidate_id": numeric_id,
            "source_hypothesis_id": mapping["source_hypothesis_id"],
            "accepted": duplicate_of is None,
            "duplicate_of": duplicate_of,
            "duplicate_metrics": None if metrics is None else {
                "axis_angle_degrees": metrics[0],
                "transverse_distance_mm": metrics[1],
                "overlap_fraction": metrics[2],
            },
        }
        audit.append(row)
        if duplicate_of is None:
            accepted.append(
                {
                    "bend_id": f"V41C-{numeric_id:016x}",
                    "line_start": addition["line_start"],
                    "line_end": addition["line_end"],
                    "source_hypothesis_id": mapping["source_hypothesis_id"],
                    "source_kind": "v41_fragment_tolerant_parent_bridge_sidecar",
                    "selection_route": PROFILE,
                    "selector_score": None,
                }
            )
    return accepted, audit


def run_case(
    *, pool_path: Path, retained_path: Path, scan_path: Path, output_path: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    candidates, mappings = adapt_candidates(pool, retained)
    mapping_by_id = {int(value["candidate_id"]): value for value in mappings}
    contract_sha = _sha_bytes(_canonical(CONTRACT))

    if candidates:
        source, mesh_replay = load_bounded_source(scan_path, pool)
        public_v41 = []
        additions = []
        resource_stops = []
        v41_result_sha = []
        # A resource stop is candidate-local evidence, not permission to skip
        # the rest of the frozen stream.  Deterministic bisection isolates the
        # stopped candidate while avoiding needless mesh canonicalizations.
        def analyze_batch(indices: list[int]) -> None:
            try:
                result = select_unified_v41(
                    source, [candidates[index] for index in indices],
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
            public_v41.append(public)
            v41_result_sha.append(_sha_bytes(canonical_result_bytes(public)))
            additions.extend(
                dict(value) for value in result.get("final_lines") or ()
                if value.get("source_lane") != "legacy"
            )
        analyze_batch(list(range(len(candidates))))
    else:
        mesh_replay = None
        public_v41 = []
        additions = []
        resource_stops = []
        v41_result_sha = []

    predictions, merge_audit = incumbent_first_merge(
        list(retained.get("predictions") or ()), additions, mapping_by_id
    )
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v41_v16_real_scan_compatibility_result",
        "profile": PROFILE,
        "route": retained.get("route"),
        "contract": CONTRACT,
        "contract_sha256": contract_sha,
        "candidate_pool_sha256": _sha_file(pool_path),
        "retained_v16_sha256": _sha_file(retained_path),
        "scan_sha256": _sha_file(scan_path),
        "candidate_mappings": mappings,
        "compatible_rejected_candidate_count": len(candidates),
        "mesh_replay": mesh_replay,
        "v41_results": public_v41,
        "v41_result_sha256": v41_result_sha,
        "v41_resource_stops": resource_stops,
        "v41_resource_stop_count": len(resource_stops),
        "v41_pre_merge_addition_count": len(additions),
        "incumbent_first_merge_audit": merge_audit,
        "v41_post_merge_addition_count": sum(value["accepted"] for value in merge_audit),
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
        "v41_resource_stop_count": result["v41_resource_stop_count"],
        "v41_pre_merge_addition_count": result["v41_pre_merge_addition_count"],
        "v41_post_merge_addition_count": result["v41_post_merge_addition_count"],
        "candidate_count": result["candidate_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
