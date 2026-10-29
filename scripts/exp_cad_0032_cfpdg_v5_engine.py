#!/usr/bin/env python3
"""Source-free execution-equivalent engine for EXP_CAD_0032 CF-PDG V5.

This module contains no physical CLI and performs no file I/O.  It installs a
small, scoped set of exact replacements around the byte-frozen EXP30 science
runner.  The replacements preserve the parent's logical outputs while avoiding
diagnostic materialization and repeated work on the 38 full-node member fits.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import math
import time
from typing import Any, Iterator, Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from scripts import exp_cad_0029_cfpdg_core as CAD
from scripts import exp_cad_0029_cfpdg_source as FROZEN_SRC
from scripts import exp_cad_0031_source_interface as FROZEN_INTAKE
from scripts import run_exp_cad_0029_cfpdg_source as FROZEN_RUNNER


EXPERIMENT_ID = "EXP_CAD_0032"
ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_EXECUTION_EQUIVALENT"
EXPECTED_CAD_NODE_COUNT = 8077
EXPECTED_UNIQUE_MEMBER_FIT_COUNT = 38
COMPONENT_DIJKSTRA_SOURCE_BATCH_SIZE = 128


def _kd_tie_self_check() -> bool:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    queries = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    tree = cKDTree(points)
    scalar = [
        np.asarray(sorted(tree.query_ball_point(row, 1.0)), dtype=np.int64)
        for row in queries
    ]
    batched = [
        np.sort(np.asarray(row, dtype=np.int64), kind="mergesort")
        for row in tree.query_ball_point(queries, 1.0, return_sorted=True)
    ]
    return all(np.array_equal(first, second) for first, second in zip(scalar, batched))


def _raycast_tie_self_check() -> bool:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    tensor = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor)
    # The diagonal rows are primitive ties; the off-diagonal rows are ordinary.
    points = np.asarray(
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.5, 1.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]],
        dtype=np.float64,
    )
    regions = np.asarray([11, 29], dtype=np.int64)
    raw_first = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
    raw_second = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
    primitive_first = raw_first["primitive_ids"].numpy().astype(np.int64)
    primitive_second = raw_second["primitive_ids"].numpy().astype(np.int64)
    closest_first = raw_first["points"].numpy().astype(np.float64)
    closest_second = raw_second["points"].numpy().astype(np.float64)
    normal_first = raw_first["primitive_normals"].numpy().astype(np.float64)
    normal_second = raw_second["primitive_normals"].numpy().astype(np.float64)
    valid = primitive_first < len(regions)
    expected_regions = np.full(len(points), -1, dtype=np.int64)
    expected_regions[valid] = regions[primitive_first[valid]]
    signed = np.sum((points - closest_first) * normal_first, axis=1)
    expected_sheets = np.where(signed >= 0.0, 1, -1).astype(np.int8)
    expected_sheets[~valid] = 0
    bound_v5 = FROZEN_RUNNER.raycast_regions_and_sheets(scene, points, regions)
    primitive_and_geometry_exact = all(
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes() == second.tobytes()
        for first, second in (
            (primitive_first, primitive_second),
            (closest_first, closest_second),
            (normal_first, normal_second),
        )
    )
    ownership_exact = all(
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes() == second.tobytes()
        for first, second in zip(
            (expected_regions, expected_sheets, valid), bound_v5, strict=True
        )
    )
    # Ensure the fixture truly contains both primitive ownerships and explicit
    # diagonal tie rows, rather than vacuously checking one triangle.
    tie_fixture_exercised = set(primitive_first.tolist()) == {0, 1}
    return bool(primitive_and_geometry_exact and ownership_exact and tie_fixture_exercised)


class V5Engine:
    """Per-execution caches and non-scientific counters."""

    def __init__(self, *, expected_node_count: int | None = None) -> None:
        self.expected_node_count = expected_node_count
        self.counters: dict[str, int] = {
            "unique_member_fit_count": 0,
            "batched_kd_fit_count": 0,
            "kd_node_query_count": 0,
            "supplemental_diagnostic_batched_kd_fit_count": 0,
            "supplemental_diagnostic_kd_node_query_count": 0,
            "local_fit_call_count": 0,
            "compact_local_fit_call_count": 0,
            "diagnostic_local_fit_call_count": 0,
            "nuisance_design_build_count": 0,
            "nuisance_design_cache_hit_count": 0,
            "induced_components_call_count": 0,
            "induced_components_cache_hit_count": 0,
            "component_streamed_dijkstra_row_count": 0,
            "component_dijkstra_batch_call_count": 0,
        }
        self._member_fit_depth = 0
        self._nuisance_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._component_cache: OrderedDict[
            tuple[Any, ...], tuple[list[dict[str, Any]], np.ndarray]
        ] = OrderedDict()
        self.component_cache_limit = 64
        self.elapsed_wall_seconds: float | None = None
        self.robustness_row_count: int | None = None
        self.kd_tie_differential_passed = _kd_tie_self_check()
        self.raycast_tie_differential_passed = _raycast_tie_self_check()

    @property
    def in_member_fit(self) -> bool:
        return self._member_fit_depth > 0

    def metrics(self) -> dict[str, Any]:
        counters = dict(self.counters)
        operation_counters = {
            **counters,
            "nuisance_design_hit_count": counters[
                "nuisance_design_cache_hit_count"
            ],
            "induced_components_streamed_dijkstra_count": counters[
                "component_streamed_dijkstra_row_count"
            ],
            "robustness_row_count": self.robustness_row_count,
        }
        full_count = counters["unique_member_fit_count"]
        expected_queries = (
            None
            if self.expected_node_count is None
            else EXPECTED_UNIQUE_MEMBER_FIT_COUNT * int(self.expected_node_count)
        )
        return {
            "schema": "EXP_CAD_0032_CFPDG_V5_ENGINE_METRICS_V1",
            "experiment_id": EXPERIMENT_ID,
            "architecture": ARCHITECTURE,
            **counters,
            "nuisance_design_hit_count": counters[
                "nuisance_design_cache_hit_count"
            ],
            "induced_components_streamed_dijkstra_count": counters[
                "component_streamed_dijkstra_row_count"
            ],
            "robustness_row_count": self.robustness_row_count,
            "operation_counters": operation_counters,
            "expected_node_count_for_this_invocation": self.expected_node_count,
            "expected_unique_member_fit_count": EXPECTED_UNIQUE_MEMBER_FIT_COUNT,
            "expected_full_fit_kd_node_query_count": expected_queries,
            "full_fit_inventory_gate_passed": full_count
            == EXPECTED_UNIQUE_MEMBER_FIT_COUNT,
            "full_fit_batched_kd_gate_passed": counters["batched_kd_fit_count"]
            == EXPECTED_UNIQUE_MEMBER_FIT_COUNT,
            "full_fit_kd_query_gate_passed": (
                expected_queries is not None
                and counters["kd_node_query_count"] == expected_queries
            ),
            "elapsed_wall_seconds_non_scientific": self.elapsed_wall_seconds,
            "equivalence_checks": {
                "raycast_tie_differential_passed": self.raycast_tie_differential_passed,
                "raycast_basis": "actual ordinary-and-tied fixture bytes through the frozen routine and separately invoked V5-bound path",
                "kd_tie_differential_passed": self.kd_tie_differential_passed,
                "semantic_payload_logical_equivalence": None,
                "visual_payload_logical_equivalence": None,
            },
            "scientific_threshold_or_gate_changed": False,
            "fit_or_robustness_row_skipped": False,
        }


def canonical_unoriented_normals_v5(normals: np.ndarray) -> np.ndarray:
    """Vectorized, byte-equivalent frozen hemisphere canonicalization."""

    values = np.asarray(normals, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("invalid scan normals")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("zero scan normal")
    values /= lengths[:, None]
    # np.argmax(axis=1) preserves the parent's lowest-Cartesian-index tie rule.
    gauge_axis = np.argmax(np.abs(values), axis=1)
    rows = np.arange(len(values), dtype=np.intp)
    gauge = values[rows, gauge_axis]
    if np.any(np.abs(gauge) <= 1e-15):
        raise ValueError("normal has no canonical hemisphere")
    values[gauge < 0.0] *= -1.0
    values[values == 0.0] = 0.0
    return values


def _identity_hash_v5(
    identities: np.ndarray, *, sort_as_set: bool, known_increasing_u64: bool
) -> str:
    values = np.asarray(identities, dtype=np.uint64)
    if sort_as_set and not known_increasing_u64:
        values = np.sort(values, kind="mergesort")
    payload = np.ascontiguousarray(values, dtype="<u8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def filter_source_rows_v5(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    member_token: str,
    original_row_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact EXP31 interface report with compact uint8 reason storage."""

    point_array = np.asarray(points, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("source points must be N-by-3")
    if normal_array.shape != point_array.shape:
        raise ValueError("source normals must match source point shape")
    increasing_identity = original_row_ids is None
    if increasing_identity:
        identities = np.arange(len(point_array), dtype=np.uint64)
    else:
        supplied = np.asarray(original_row_ids)
        if supplied.shape != (len(point_array),):
            raise ValueError("original row identity count mismatch")
        # Delegate the uncommon externally-carried identity validation to the
        # exact frozen serializer, including its error behavior.
        FROZEN_INTAKE.identity_bytes(supplied, sort_as_set=False)
        identities = np.asarray(supplied, dtype=np.uint64)

    point_nonfinite = ~np.all(np.isfinite(point_array), axis=1)
    normal_components_nonfinite = ~np.all(np.isfinite(normal_array), axis=1)
    normal_norm = FROZEN_INTAKE.stable_normal_norm(normal_array)
    with np.errstate(over="ignore", invalid="ignore"):
        parent_normal_norm = np.linalg.norm(normal_array, axis=1)
    normal_nonfinite = (
        normal_components_nonfinite
        | ~np.isfinite(normal_norm)
        | ~np.isfinite(parent_normal_norm)
    )
    zero_normal = (~normal_nonfinite) & (
        normal_norm <= FROZEN_INTAKE.NORMAL_NORM_EPSILON
    )

    # 0=valid, 1=nonfinite point, 2=nonfinite normal, 3=near-zero normal.
    reason_code = np.zeros(len(point_array), dtype=np.uint8)
    reason_code[point_nonfinite] = 1
    unresolved = reason_code == 0
    reason_code[unresolved & normal_nonfinite] = 2
    unresolved = reason_code == 0
    reason_code[unresolved & zero_normal] = 3
    valid = reason_code == 0

    retained_points = np.ascontiguousarray(point_array[valid], dtype=np.float64)
    retained_normals = np.ascontiguousarray(normal_array[valid], dtype=np.float64)
    retained_ids = np.ascontiguousarray(identities[valid], dtype=np.uint64)
    rejected_ids = np.ascontiguousarray(identities[~valid], dtype=np.uint64)
    codes = dict(zip(FROZEN_INTAKE.REASON_ORDER, (1, 2, 3), strict=True))
    exclusive_counts = {
        reason: int(np.count_nonzero(reason_code == code))
        for reason, code in codes.items()
    }
    raw_predicate_counts = {
        "NONFINITE_POINT": int(np.count_nonzero(point_nonfinite)),
        "NONFINITE_NORMAL": int(np.count_nonzero(normal_nonfinite)),
        "NORMAL_NORM_LE_1E_12": int(np.count_nonzero(zero_normal)),
    }
    if sum(exclusive_counts.values()) != len(rejected_ids):
        raise RuntimeError("exclusive source rejection accounting mismatch")

    actionability = FROZEN_INTAKE.evaluate_actionability(
        original_count=len(point_array), retained_count=len(retained_points)
    )
    reason_identity_hashes = {
        reason: _identity_hash_v5(
            identities[reason_code == code],
            sort_as_set=True,
            known_increasing_u64=increasing_identity,
        )
        for reason, code in codes.items()
    }
    report: dict[str, Any] = {
        "schema": FROZEN_INTAKE.SCHEMA,
        "member_token": str(member_token),
        "original_row_identity_encoding": "LITTLE_ENDIAN_UINT64_ACQUISITION_ORDINAL",
        "original_row_count": int(len(point_array)),
        "retained_row_count": int(len(retained_points)),
        "rejected_row_count": int(len(rejected_ids)),
        "raw_predicate_counts": raw_predicate_counts,
        "exclusive_rejection_counts": exclusive_counts,
        "original_identity_sequence_sha256": _identity_hash_v5(
            identities, sort_as_set=False, known_increasing_u64=increasing_identity
        ),
        "original_identity_set_sha256": _identity_hash_v5(
            identities, sort_as_set=True, known_increasing_u64=increasing_identity
        ),
        "retained_identity_sequence_sha256": _identity_hash_v5(
            retained_ids, sort_as_set=False, known_increasing_u64=increasing_identity
        ),
        "retained_identity_set_sha256": _identity_hash_v5(
            retained_ids, sort_as_set=True, known_increasing_u64=increasing_identity
        ),
        "rejected_identity_set_sha256": _identity_hash_v5(
            rejected_ids, sort_as_set=True, known_increasing_u64=increasing_identity
        ),
        "rejected_identity_sha256_by_exclusive_reason": reason_identity_hashes,
        "rejected_original_row_ids_sorted": [
            int(value)
            for value in (
                rejected_ids
                if increasing_identity
                else np.sort(rejected_ids, kind="mergesort")
            )
        ],
        "retained_points_sha256": FROZEN_INTAKE.numeric_array_sha256(
            retained_points
        ),
        "retained_normals_sha256": FROZEN_INTAKE.numeric_array_sha256(
            retained_normals
        ),
        "normal_norm_epsilon_inclusive": FROZEN_INTAKE.NORMAL_NORM_EPSILON,
        "replacement_interpolation_or_normal_estimation_performed": False,
        **actionability,
    }
    return {
        "points": retained_points,
        "normals": retained_normals,
        "original_row_ids": retained_ids,
        "report": report,
    }


def _compact_local_surface_fit(
    *,
    node_point: np.ndarray,
    node_normal: np.ndarray,
    tangent_first: np.ndarray,
    tangent_second: np.ndarray,
    node_region: int,
    voxel_points: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_regions: np.ndarray,
    voxel_sheets: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Frozen local-fit numerics without diagnostic-only list creation."""

    del voxel_sheets  # diagnostic-only on the noncentral path
    relative = voxel_points - node_point
    u = relative @ tangent_first
    v = relative @ tangent_second
    height = relative @ node_normal
    tangent_radius = np.sqrt(u * u + v * v)
    normal_dot = np.abs(voxel_normals @ node_normal)
    eligible = (
        (tangent_radius <= float(config["tangent_radius_mm"]))
        & (np.abs(height) <= float(config["normal_height_limit_mm"]))
        & (
            normal_dot
            >= math.cos(
                math.radians(
                    float(config["maximum_scan_normal_to_cad_normal_angle_deg"])
                )
            )
        )
        & (np.asarray(voxel_regions, dtype=np.int64) == int(node_region))
    )
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return {"valid": False, "reason": "NO_ELIGIBLE_VOXELS"}
    order = np.lexsort(
        (
            voxel_keys[ids, 2],
            voxel_keys[ids, 1],
            voxel_keys[ids, 0],
            voxel_normals[ids, 2],
            voxel_normals[ids, 1],
            voxel_normals[ids, 0],
            voxel_points[ids, 2],
            voxel_points[ids, 1],
            voxel_points[ids, 0],
            height[ids],
        )
    )
    ids = ids[order]
    ordered_heights = height[ids]
    starts = np.r_[
        0,
        1
        + np.flatnonzero(
            np.diff(ordered_heights) > float(config["height_cluster_gap_mm"])
        ),
        len(ids),
    ]
    eligible_clusters: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for cluster_id, (start, stop) in enumerate(
        zip(starts[:-1], starts[1:], strict=True)
    ):
        selected = ids[int(start) : int(stop)]
        median_height = FROZEN_SRC.ordered_median(height[selected])
        score = abs(median_height)
        occupied_count, _occupied_cells = FROZEN_SRC.footprint_occupancy(
            np.column_stack((u[selected], v[selected])),
            float(config["tangent_radius_mm"]),
            int(config["footprint_angular_sector_count"]),
            int(config["footprint_radial_band_count"]),
        )
        if (
            len(selected) < int(config["minimum_points"])
            or occupied_count
            < int(config["minimum_occupied_footprint_cells"])
            or score
            > float(config["maximum_selected_cluster_median_abs_height_mm"])
        ):
            continue
        try:
            parameters, _residual, p90 = FROZEN_SRC.huber_plane_fit(
                np.column_stack((u[selected], v[selected])),
                height[selected],
                iterations=int(config["huber_iteration_count"]),
                delta_mm=float(config["huber_delta_mm"]),
            )
            fitted_normal = (
                node_normal
                - parameters[1] * tangent_first
                - parameters[2] * tangent_second
            )
            fitted_normal /= np.linalg.norm(fitted_normal)
            fitted_angle = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            abs(np.dot(fitted_normal, node_normal)), -1.0, 1.0
                        )
                    )
                )
            )
            if (
                p90 <= float(config["maximum_plane_fit_p90_mm"])
                and fitted_angle
                <= float(config["maximum_fitted_normal_to_cad_normal_angle_deg"])
            ):
                first_id = int(selected[0])
                canonical_first_row = tuple(
                    float(value)
                    for value in np.r_[
                        height[first_id],
                        voxel_points[first_id],
                        voxel_normals[first_id],
                    ]
                ) + tuple(int(value) for value in voxel_keys[first_id])
                canonical_key = (score, -int(len(selected)), canonical_first_row)
                eligible_clusters.append(
                    (
                        canonical_key,
                        cluster_id,
                        {
                            "score_mm": score,
                            "parameters": parameters,
                            "selected_voxel_count": int(len(selected)),
                            "selected_voxel_keys": voxel_keys[selected],
                        },
                    )
                )
        except ValueError:
            pass
    if not eligible_clusters:
        return {"valid": False, "reason": "NO_ELIGIBLE_HEIGHT_CLUSTER"}
    eligible_clusters.sort(key=lambda item: item[0])
    _winner_key, _winner_id, winner = eligible_clusters[0]
    if len(eligible_clusters) > 1:
        runner_difference = float(
            eligible_clusters[1][2]["score_mm"] - winner["score_mm"]
        )
        if runner_difference < float(
            config["height_cluster_runner_separation_mm"]
        ):
            return {"valid": False, "reason": "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS"}
    parameters = np.asarray(winner["parameters"], dtype=np.float64)
    feature = np.asarray(
        [
            parameters[0],
            float(config["tangent_radius_mm"]) * parameters[1],
            float(config["tangent_radius_mm"]) * parameters[2],
        ],
        dtype=np.float64,
    )
    return {
        "valid": True,
        "feature": feature,
        "selected_voxel_count": winner["selected_voxel_count"],
        "selected_voxel_keys": winner["selected_voxel_keys"],
    }


def fit_scan_to_nodes_v5(
    *,
    engine: V5Engine,
    nodes: np.ndarray,
    voxel_points: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_regions: np.ndarray,
    voxel_sheets: np.ndarray,
    config: Mapping[str, Any],
    keep_diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any] | None], np.ndarray]:
    """Exact batched neighbor query with diagnostic-aware local fitting."""

    node_points = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    node_normals = np.column_stack((nodes["nx"], nodes["ny"], nodes["nz"]))
    tangent_first = np.column_stack((nodes["e1x"], nodes["e1y"], nodes["e1z"]))
    tangent_second = np.column_stack((nodes["e2x"], nodes["e2y"], nodes["e2z"]))
    radius = math.sqrt(
        float(config["tangent_radius_mm"]) ** 2
        + float(config["normal_height_limit_mm"]) ** 2
    )
    features = np.full((len(nodes), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(nodes), dtype=bool)
    selected_counts = np.zeros(len(nodes), dtype=np.int64)
    selected_voxel_mask = np.zeros(len(voxel_points), dtype=bool)
    diagnostics: list[dict[str, Any] | None] = [None] * len(nodes)
    if len(voxel_points) == 0:
        if keep_diagnostics:
            diagnostics = [
                {
                    "valid": False,
                    "reason": "EMPTY_VOXEL_PARTITION",
                    "clusters": [],
                }
                for _ in range(len(nodes))
            ]
        return features, valid, selected_counts, diagnostics, selected_voxel_mask

    search = cKDTree(voxel_points)
    neighbor_rows = search.query_ball_point(node_points, radius, return_sorted=True)
    # Canonicalize explicitly even though current SciPy promises sorted output.
    neighbors = [
        np.sort(np.asarray(row, dtype=np.int64), kind="mergesort")
        for row in neighbor_rows
    ]
    if engine.in_member_fit:
        engine.counters["batched_kd_fit_count"] += 1
        engine.counters["kd_node_query_count"] += len(nodes)
    else:
        engine.counters["supplemental_diagnostic_batched_kd_fit_count"] += 1
        engine.counters["supplemental_diagnostic_kd_node_query_count"] += len(nodes)

    sheets = np.asarray(voxel_sheets)
    for node_id, local in enumerate(neighbors):
        if not len(local):
            if keep_diagnostics:
                diagnostics[node_id] = {
                    "valid": False,
                    "reason": "EMPTY_LOCAL_BALL",
                    "clusters": [],
                }
            continue
        engine.counters["local_fit_call_count"] += 1
        kwargs = {
            "node_point": node_points[node_id],
            "node_normal": node_normals[node_id],
            "tangent_first": tangent_first[node_id],
            "tangent_second": tangent_second[node_id],
            "node_region": int(nodes["smooth_region"][node_id]),
            "voxel_points": voxel_points[local],
            "voxel_normals": voxel_normals[local],
            "voxel_keys": voxel_keys[local],
            "voxel_regions": voxel_regions[local],
            "voxel_sheets": sheets[local],
            "config": config,
        }
        if keep_diagnostics:
            engine.counters["diagnostic_local_fit_call_count"] += 1
            result = FROZEN_SRC.local_surface_fit(**kwargs)
        else:
            engine.counters["compact_local_fit_call_count"] += 1
            result = _compact_local_surface_fit(**kwargs)
        if result["valid"]:
            valid[node_id] = True
            features[node_id] = result["feature"]
            selected_counts[node_id] = int(result["selected_voxel_count"])
            selected_keys = {
                tuple(int(value) for value in key)
                for key in result.get("selected_voxel_keys", [])
            }
            if selected_keys:
                selected_voxel_mask[local] |= np.asarray(
                    [
                        tuple(int(value) for value in voxel_keys[index])
                        in selected_keys
                        for index in local
                    ],
                    dtype=bool,
                )
        if keep_diagnostics:
            diagnostics[node_id] = result
    return features, valid, selected_counts, diagnostics, selected_voxel_mask


def nuisance_design_v5(
    engine: V5Engine, nodes: np.ndarray, fixed_centroid: np.ndarray
) -> np.ndarray:
    centroid = np.asarray(fixed_centroid, dtype=np.float64)
    key = (
        id(nodes),
        len(nodes),
        nodes.dtype.str,
        centroid.shape,
        centroid.tobytes(),
    )
    cached = engine._nuisance_cache.get(key)
    if cached is not None:
        engine.counters["nuisance_design_cache_hit_count"] += 1
        return cached
    engine.counters["nuisance_design_build_count"] += 1
    points = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    normals = np.column_stack((nodes["nx"], nodes["ny"], nodes["nz"]))
    first = np.column_stack((nodes["e1x"], nodes["e1y"], nodes["e1z"]))
    second = np.column_stack((nodes["e2x"], nodes["e2y"], nodes["e2z"]))
    design = np.zeros((len(nodes), 3, 6), dtype=np.float64)
    design[:, 0, :3] = normals
    design[:, 0, 3:] = np.cross(points - centroid, normals)
    design[:, 1, 3:] = 3.0 * np.cross(first, normals)
    design[:, 2, 3:] = 3.0 * np.cross(second, normals)
    engine._nuisance_cache[key] = design
    return design


def induced_components_v5(
    *,
    engine: V5Engine,
    active: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    minimum_node_count: int,
    minimum_area_mm2: float,
    minimum_span_mm: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Frozen components with dead allocations removed and streamed medoids."""

    engine.counters["induced_components_call_count"] += 1
    active_array = np.asarray(active, dtype=bool)
    edge_array = np.asarray(edges, dtype=np.int64)
    key = (
        np.packbits(active_array, bitorder="little").tobytes(),
        id(nodes),
        id(edges),
        id(edge_weights),
        int(minimum_node_count),
        float(minimum_area_mm2),
        float(minimum_span_mm),
    )
    cached = engine._component_cache.get(key)
    if cached is not None:
        engine.counters["induced_components_cache_hit_count"] += 1
        engine._component_cache.move_to_end(key)
        components, labels = cached
        return copy.deepcopy(components), labels.copy()

    kept_edges = edge_array[
        active_array[edge_array[:, 0]] & active_array[edge_array[:, 1]]
    ]
    labels = np.full(len(nodes), -1, dtype=np.int64)
    if not np.any(active_array):
        result = ([], labels)
        engine._component_cache[key] = ([], labels.copy())
        return result
    union = CAD.UnionFind(len(nodes))
    for first, second in kept_edges:
        union.union(int(first), int(second))
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id in np.flatnonzero(active_array):
        groups[union.find(int(node_id))].append(int(node_id))

    retained: list[dict[str, Any]] = []
    area = np.asarray(nodes["area_mm2"], dtype=np.float64)
    weight_array = np.asarray(edge_weights)
    for members in sorted(groups.values(), key=lambda row: min(row)):
        ids = np.asarray(sorted(members), dtype=np.int64)
        component_area = math.fsum(float(value) for value in area[ids])
        if len(ids) < int(minimum_node_count) or component_area < float(
            minimum_area_mm2
        ):
            continue
        member_mask = np.zeros(len(nodes), dtype=bool)
        member_mask[ids] = True
        component_edge_mask = (
            member_mask[edge_array[:, 0]] & member_mask[edge_array[:, 1]]
        )
        component_graph = CAD.weighted_node_graph(
            len(nodes),
            edge_array[component_edge_mask],
            weight_array[component_edge_mask],
        )
        anchor = int(ids[0])
        first_distance = dijkstra(component_graph, directed=False, indices=anchor)
        maximum = float(np.max(first_distance[ids]))
        first_endpoint = int(np.min(ids[first_distance[ids] == maximum]))
        second_distance = dijkstra(
            component_graph, directed=False, indices=first_endpoint
        )
        span = float(np.max(second_distance[ids]))
        if span < float(minimum_span_mm):
            continue
        costs = np.empty(len(ids), dtype=np.float64)
        for batch_start in range(0, len(ids), COMPONENT_DIJKSTRA_SOURCE_BATCH_SIZE):
            batch_stop = min(
                batch_start + COMPONENT_DIJKSTRA_SOURCE_BATCH_SIZE, len(ids)
            )
            source_ids = ids[batch_start:batch_stop]
            distance_rows = np.asarray(
                dijkstra(component_graph, directed=False, indices=source_ids)
            )[:, ids]
            engine.counters["component_dijkstra_batch_call_count"] += 1
            for offset, distance in enumerate(distance_rows):
                row = batch_start + offset
                engine.counters["component_streamed_dijkstra_row_count"] += 1
                if not np.all(np.isfinite(distance)):
                    raise ValueError("retained component is graph-disconnected")
                costs[row] = math.fsum(
                    float(distance[column] * area[node_id])
                    for column, node_id in enumerate(ids)
                )
        minimum_cost = float(np.min(costs))
        medoid = int(np.min(ids[costs == minimum_cost]))
        component_id = len(retained)
        labels[ids] = component_id
        packed_ids = np.asarray(ids, dtype="<i8").tobytes()
        retained.append(
            {
                "component_id": component_id,
                "node_ids": ids.tolist(),
                "node_ids_sha256": hashlib.sha256(packed_ids).hexdigest(),
                "node_count": int(len(ids)),
                "area_mm2": component_area,
                "two_sweep_span_mm": span,
                "area_medoid_node_id": medoid,
            }
        )
    engine._component_cache[key] = (copy.deepcopy(retained), labels.copy())
    if len(engine._component_cache) > engine.component_cache_limit:
        engine._component_cache.popitem(last=False)
    return retained, labels


@contextmanager
def optimized_science_context(engine: V5Engine) -> Iterator[None]:
    """Install exact V5 replacements only for the duration of one call."""

    original = {
        "canonical": FROZEN_SRC.canonical_unoriented_normals,
        "fit": FROZEN_SRC.fit_scan_to_nodes,
        "nuisance": FROZEN_SRC.nuisance_design,
        "components": FROZEN_SRC.induced_components,
        "member_fit": FROZEN_RUNNER.member_fit_variant,
        "runner_dijkstra": getattr(FROZEN_RUNNER, "dijkstra", None),
    }
    runner_had_dijkstra = hasattr(FROZEN_RUNNER, "dijkstra")

    def fit_bound(**kwargs: Any) -> Any:
        return fit_scan_to_nodes_v5(engine=engine, **kwargs)

    def nuisance_bound(nodes: np.ndarray, fixed_centroid: np.ndarray) -> np.ndarray:
        return nuisance_design_v5(engine, nodes, fixed_centroid)

    def components_bound(**kwargs: Any) -> Any:
        return induced_components_v5(engine=engine, **kwargs)

    def member_fit_bound(**kwargs: Any) -> Any:
        engine.counters["unique_member_fit_count"] += 1
        engine._member_fit_depth += 1
        try:
            return original["member_fit"](**kwargs)
        finally:
            engine._member_fit_depth -= 1

    FROZEN_SRC.canonical_unoriented_normals = canonical_unoriented_normals_v5
    FROZEN_SRC.fit_scan_to_nodes = fit_bound
    FROZEN_SRC.nuisance_design = nuisance_bound
    FROZEN_SRC.induced_components = components_bound
    FROZEN_RUNNER.member_fit_variant = member_fit_bound
    FROZEN_RUNNER.dijkstra = dijkstra
    try:
        yield
    finally:
        if runner_had_dijkstra:
            FROZEN_RUNNER.dijkstra = original["runner_dijkstra"]
        else:
            delattr(FROZEN_RUNNER, "dijkstra")
        FROZEN_RUNNER.member_fit_variant = original["member_fit"]
        FROZEN_SRC.induced_components = original["components"]
        FROZEN_SRC.nuisance_design = original["nuisance"]
        FROZEN_SRC.fit_scan_to_nodes = original["fit"]
        FROZEN_SRC.canonical_unoriented_normals = original["canonical"]


def execute_parent_science_v5(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Run the frozen parent science under exact V5 execution replacements."""

    nodes = np.asarray(kwargs["nodes"])
    engine = V5Engine(expected_node_count=len(nodes))
    started = time.perf_counter()
    with optimized_science_context(engine):
        semantic, visual = FROZEN_RUNNER.execute_source(**kwargs)
    engine.robustness_row_count = len(semantic.get("robustness", []))
    engine.elapsed_wall_seconds = time.perf_counter() - started
    return semantic, visual, engine.metrics()
