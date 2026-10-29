#!/usr/bin/env python3
"""Frozen source-only semantics reused by EXP_CAD_0030 CF-PDG V3.

The module is deliberately independent of corpus manifests, evaluator roles,
labels, gold, expected counts, filenames, and quality tokens.  The CLI wrapper
is responsible only for immutable byte bindings and scan I/O after a separate
audit approval.  Every function here accepts already opaque numeric arrays so
the full contract can be qualified without loading either physical scan.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from scripts import exp_cad_0029_cfpdg_core as CAD


def canonical_unoriented_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64).copy()
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.all(np.isfinite(normals)):
        raise ValueError("invalid scan normals")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("zero scan normal")
    normals /= lengths[:, None]
    for row in range(len(normals)):
        # np.argmax returns the lowest Cartesian index on an exact tie.
        gauge_axis = int(np.argmax(np.abs(normals[row])))
        if abs(float(normals[row, gauge_axis])) <= 1e-15:
            raise ValueError("normal has no canonical hemisphere")
        if normals[row, gauge_axis] < 0.0:
            normals[row] *= -1.0
    normals[normals == 0.0] = 0.0
    return normals


def apply_rigid_transform(
    points: np.ndarray, normals: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2 or points.shape[1] != 3
        or np.asarray(normals).shape != points.shape
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("invalid transform point/normal arrays")
    normals = canonical_unoriented_normals(normals)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("invalid transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("non-rigid transform")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("improper transform")
    transformed_points = points @ rotation.T + transform[:3, 3]
    transformed_normals = canonical_unoriented_normals(normals @ rotation.T)
    return transformed_points, transformed_normals


def rotate_about_center(
    points: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    axis: int,
    degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(float(degrees))
    unit = np.zeros(3, dtype=np.float64)
    unit[int(axis)] = 1.0
    skew = np.asarray(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return (
        (np.asarray(points) - center) @ rotation.T + center,
        canonical_unoriented_normals(np.asarray(normals) @ rotation.T),
    )


def canonical_identity_bits(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype="<f8").copy()
    if (
        points.ndim != 2 or points.shape[1] != 3
        or np.asarray(normals).shape != points.shape
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("invalid identity point/normal arrays")
    normals = canonical_unoriented_normals(normals).astype("<f8", copy=True)
    points[points == 0.0] = 0.0
    normals[normals == 0.0] = 0.0
    bits = np.empty(len(points), dtype=np.int8)
    for index in range(len(points)):
        digest = hashlib.sha256(points[index].tobytes() + normals[index].tobytes()).digest()
        bits[index] = int(digest[0] & 1)
    return bits


def spatial_checker_bits(points: np.ndarray, cell_mm: float) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    cell = float(cell_mm)
    if (
        point_array.ndim != 2
        or point_array.shape[1] != 3
        or not np.all(np.isfinite(point_array))
        or not math.isfinite(cell)
        or cell <= 0.0
    ):
        raise ValueError("invalid spatial checker input")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.floor(point_array / cell)
    limit = float(np.iinfo(np.int64).max)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > limit):
        raise ValueError("spatial checker cell index exceeds int64")
    keys = scaled.astype(np.int64)
    return np.mod(np.sum(keys, axis=1), 2).astype(np.int8)


def ordered_median(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64), kind="mergesort")
    if not len(values):
        raise ValueError("empty median")
    middle = len(values) // 2
    if len(values) & 1:
        return float(values[middle])
    return math.fsum((float(values[middle - 1]), float(values[middle]))) / 2.0


def percentile_type7(values: np.ndarray, probability: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64), kind="mergesort")
    if not len(values):
        raise ValueError("empty percentile")
    position = (len(values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float((1.0 - fraction) * values[lower] + fraction * values[upper])


def voxelize_partition(
    points: np.ndarray,
    normals: np.ndarray,
    voxel_mm: float,
    partition_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if partition_mask is not None:
        mask = np.asarray(partition_mask, dtype=bool)
        points = np.asarray(points)[mask]
        normals = np.asarray(normals)[mask]
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
    aggregated = CAD.permutation_invariant_voxel_aggregate(
        np.asarray(points, dtype=np.float64),
        canonical_unoriented_normals(normals),
        float(voxel_mm),
    )
    return tuple(
        np.asarray(value).reshape(-1, 3) for value in aggregated
    )


def footprint_occupancy(
    tangent_uv: np.ndarray, radius_mm: float, sectors: int, radial_bands: int
) -> tuple[int, list[int]]:
    tangent_uv = np.asarray(tangent_uv, dtype=np.float64)
    radii = np.linalg.norm(tangent_uv, axis=1)
    angles = np.mod(np.arctan2(tangent_uv[:, 1], tangent_uv[:, 0]), 2.0 * math.pi)
    sector = np.floor(angles * int(sectors) / (2.0 * math.pi)).astype(np.int64)
    sector = np.minimum(sector, int(sectors) - 1)
    if int(radial_bands) != 2:
        raise ValueError("EXP29 freezes exactly two radial bands")
    # Inner band owns the inclusive rho==1.5mm boundary.
    band = (radii > 0.5 * float(radius_mm)).astype(np.int64)
    cells = band * int(sectors) + sector
    occupied = sorted(set(int(value) for value in cells))
    return len(occupied), occupied


def huber_plane_fit(
    uv: np.ndarray,
    heights: np.ndarray,
    *,
    iterations: int,
    delta_mm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    uv = np.asarray(uv, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    design = np.column_stack((np.ones(len(uv)), uv))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("rank-deficient plane")
    parameters = np.linalg.lstsq(design, heights, rcond=None)[0]
    for _ in range(int(iterations)):
        residual = heights - design @ parameters
        absolute = np.abs(residual)
        weights = np.ones(len(residual), dtype=np.float64)
        nonzero = absolute > float(delta_mm)
        weights[nonzero] = float(delta_mm) / absolute[nonzero]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_heights = heights * np.sqrt(weights)
        if np.linalg.matrix_rank(weighted_design) != 3:
            raise ValueError("rank-deficient Huber plane")
        parameters = np.linalg.lstsq(weighted_design, weighted_heights, rcond=None)[0]
    residual = heights - design @ parameters
    return parameters, residual, percentile_type7(np.abs(residual), 0.90)


def local_surface_fit(
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
            >= math.cos(math.radians(float(config["maximum_scan_normal_to_cad_normal_angle_deg"])))
        )
        & (np.asarray(voxel_regions, dtype=np.int64) == int(node_region))
    )
    ids = np.flatnonzero(eligible)
    if not len(ids):
        return {"valid": False, "reason": "NO_ELIGIBLE_VOXELS", "clusters": []}
    order = np.lexsort(
        (
            voxel_keys[ids, 2], voxel_keys[ids, 1], voxel_keys[ids, 0],
            voxel_normals[ids, 2], voxel_normals[ids, 1], voxel_normals[ids, 0],
            voxel_points[ids, 2], voxel_points[ids, 1], voxel_points[ids, 0],
            height[ids],
        )
    )
    ids = ids[order]
    ordered_heights = height[ids]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(ordered_heights) > float(config["height_cluster_gap_mm"])), len(ids)]
    cluster_rows: list[dict[str, Any]] = []
    eligible_clusters: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    for cluster_id, (start, stop) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        selected = ids[int(start) : int(stop)]
        median_height = ordered_median(height[selected])
        score = abs(median_height)
        occupied_count, occupied_cells = footprint_occupancy(
            np.column_stack((u[selected], v[selected])),
            float(config["tangent_radius_mm"]),
            int(config["footprint_angular_sector_count"]),
            int(config["footprint_radial_band_count"]),
        )
        row: dict[str, Any] = {
            "cluster_id": cluster_id,
            "voxel_count": int(len(selected)),
            "median_height_mm": median_height,
            "score_mm": score,
            "occupied_cell_count": occupied_count,
            "occupied_cells": occupied_cells,
            "first_voxel_key": voxel_keys[int(selected[0])].tolist(),
            "height_values_mm": height[selected].tolist(),
            "xyz_mm": voxel_points[selected].tolist(),
            "sheet_ids": np.asarray(voxel_sheets, dtype=np.int8)[selected].tolist(),
            "selected_voxel_keys": voxel_keys[selected].tolist(),
            "eligible": False,
        }
        if (
            len(selected) >= int(config["minimum_points"])
            and occupied_count >= int(config["minimum_occupied_footprint_cells"])
            and score <= float(config["maximum_selected_cluster_median_abs_height_mm"])
        ):
            try:
                parameters, residual, p90 = huber_plane_fit(
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
                    math.acos(float(np.clip(abs(np.dot(fitted_normal, node_normal)), -1.0, 1.0)))
                )
                row.update(
                    {
                        "plane_parameters": parameters.tolist(),
                        "plane_fit_p90_mm": p90,
                        "fitted_normal_angle_deg": fitted_angle,
                    }
                )
                if (
                    p90 <= float(config["maximum_plane_fit_p90_mm"])
                    and fitted_angle <= float(config["maximum_fitted_normal_to_cad_normal_angle_deg"])
                ):
                    row["eligible"] = True
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
                    eligible_clusters.append((canonical_key, cluster_id, row))
            except ValueError:
                row["fit_failure"] = "RANK_OR_NUMERIC"
        cluster_rows.append(row)
    if not eligible_clusters:
        return {"valid": False, "reason": "NO_ELIGIBLE_HEIGHT_CLUSTER", "clusters": cluster_rows}
    eligible_clusters.sort(key=lambda item: item[0])
    winner_key, winner_id, winner = eligible_clusters[0]
    # A score tie at 1e-12 is resolved only by count and canonical row identity;
    # an identical remaining key is impossible because cluster IDs differ.
    if len(eligible_clusters) > 1:
        runner = eligible_clusters[1][2]
        runner_difference = float(runner["score_mm"] - winner["score_mm"])
        if runner_difference < float(config["height_cluster_runner_separation_mm"]):
            return {
                "valid": False,
                "reason": "HEIGHT_CLUSTER_RUNNER_AMBIGUOUS",
                "winner_cluster_id": winner_id,
                "runner_cluster_id": eligible_clusters[1][1],
                "runner_score_difference_mm": runner_difference,
                "clusters": cluster_rows,
            }
    parameters = np.asarray(winner["plane_parameters"], dtype=np.float64)
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
        "winner_cluster_id": winner_id,
        "runner_cluster_id": eligible_clusters[1][1] if len(eligible_clusters) > 1 else None,
        "clusters": cluster_rows,
        "plane_fit_p90_mm": winner["plane_fit_p90_mm"],
        "occupied_cells": winner["occupied_cells"],
        "selected_voxel_count": int(winner["voxel_count"]),
        "selected_voxel_keys": winner.get("selected_voxel_keys", []),
        "local_support_xyz_mm": voxel_points[ids].tolist(),
        "local_support_height_mm": height[ids].tolist(),
        "local_support_sheet_ids": np.asarray(voxel_sheets, dtype=np.int8)[ids].tolist(),
    }


def fit_scan_to_nodes(
    *,
    nodes: np.ndarray,
    voxel_points: np.ndarray,
    voxel_normals: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_regions: np.ndarray,
    voxel_sheets: np.ndarray,
    config: Mapping[str, Any],
    keep_diagnostics: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any] | None], np.ndarray]:
    node_points = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    node_normals = np.column_stack((nodes["nx"], nodes["ny"], nodes["nz"]))
    tangent_first = np.column_stack((nodes["e1x"], nodes["e1y"], nodes["e1z"]))
    tangent_second = np.column_stack((nodes["e2x"], nodes["e2y"], nodes["e2z"]))
    search = cKDTree(voxel_points)
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
                {"valid": False, "reason": "EMPTY_VOXEL_PARTITION", "clusters": []}
                for _ in range(len(nodes))
            ]
        return features, valid, selected_counts, diagnostics, selected_voxel_mask
    for node_id in range(len(nodes)):
        local = np.asarray(
            sorted(search.query_ball_point(node_points[node_id], radius)),
            dtype=np.int64,
        )
        if not len(local):
            if keep_diagnostics:
                diagnostics[node_id] = {"valid": False, "reason": "EMPTY_LOCAL_BALL", "clusters": []}
            continue
        result = local_surface_fit(
            node_point=node_points[node_id],
            node_normal=node_normals[node_id],
            tangent_first=tangent_first[node_id],
            tangent_second=tangent_second[node_id],
            node_region=int(nodes["smooth_region"][node_id]),
            voxel_points=voxel_points[local],
            voxel_normals=voxel_normals[local],
            voxel_keys=voxel_keys[local],
            voxel_regions=voxel_regions[local],
            voxel_sheets=np.asarray(voxel_sheets)[local],
            config=config,
        )
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
                    [tuple(int(value) for value in voxel_keys[index]) in selected_keys for index in local],
                    dtype=bool,
                )
        if keep_diagnostics:
            diagnostics[node_id] = result
    return features, valid, selected_counts, diagnostics, selected_voxel_mask


def nuisance_design(nodes: np.ndarray, fixed_centroid: np.ndarray) -> np.ndarray:
    points = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    normals = np.column_stack((nodes["nx"], nodes["ny"], nodes["nz"]))
    first = np.column_stack((nodes["e1x"], nodes["e1y"], nodes["e1z"]))
    second = np.column_stack((nodes["e2x"], nodes["e2y"], nodes["e2z"]))
    design = np.zeros((len(nodes), 3, 6), dtype=np.float64)
    design[:, 0, :3] = normals
    design[:, 0, 3:] = np.cross(points - np.asarray(fixed_centroid), normals)
    design[:, 1, 3:] = 3.0 * np.cross(first, normals)
    design[:, 2, 3:] = 3.0 * np.cross(second, normals)
    return design


def normalized_design_rank_condition(
    design: np.ndarray, area: np.ndarray
) -> tuple[int, float]:
    rows = np.asarray(design, dtype=np.float64).reshape(-1, 6)
    weights = np.repeat(np.sqrt(np.asarray(area, dtype=np.float64)), 3)
    weighted = rows * weights[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    if np.any(norms <= 1e-12):
        return 0, math.inf
    singular = np.linalg.svd(weighted / norms[None, :], compute_uv=False)
    return CAD.design_rank_condition(singular)


def weighted_nuisance_solve(
    design: np.ndarray, response: np.ndarray, area: np.ndarray
) -> np.ndarray:
    rows = design.reshape(-1, 6)
    target = response.reshape(-1)
    weights = np.repeat(np.sqrt(np.asarray(area, dtype=np.float64)), 3)
    return np.linalg.lstsq(rows * weights[:, None], target * weights, rcond=None)[0]


def least_trimmed_nuisance(
    *,
    design: np.ndarray,
    response: np.ndarray,
    area: np.ndarray,
    node_ids: np.ndarray,
    retained_area_fraction: float,
    refit_count: int,
    maximum_condition_number: float,
) -> tuple[np.ndarray, np.ndarray]:
    rank, condition = normalized_design_rank_condition(design, area)
    if rank != 6 or condition > float(maximum_condition_number):
        raise ValueError("initial LTS design gate")
    parameters = weighted_nuisance_solve(design, response, area)
    retained = np.ones(len(node_ids), dtype=bool)
    total_area = math.fsum(float(value) for value in area)
    for _ in range(int(refit_count)):
        residual_norm = np.linalg.norm(response - np.einsum("nij,j->ni", design, parameters), axis=1)
        order = np.lexsort((node_ids, residual_norm))
        retained = np.zeros(len(node_ids), dtype=bool)
        accumulated = 0.0
        target = float(retained_area_fraction) * total_area
        for local_id in order:
            retained[int(local_id)] = True
            accumulated += float(area[int(local_id)])
            if accumulated >= target:
                break
        rank, condition = normalized_design_rank_condition(
            design[retained], area[retained]
        )
        if rank != 6 or condition > float(maximum_condition_number):
            raise ValueError("retained LTS design gate")
        parameters = weighted_nuisance_solve(
            design[retained], response[retained], area[retained]
        )
    return parameters, retained


def crossfit_pair_field(
    *,
    nodes: np.ndarray,
    feature_first: np.ndarray,
    valid_first: np.ndarray,
    feature_second: np.ndarray,
    valid_second: np.ndarray,
    fold_training_masks: Sequence[np.ndarray],
    fixed_centroid: np.ndarray,
    config: Mapping[str, Any],
    pair_config: Mapping[str, Any],
) -> dict[str, Any]:
    area = np.asarray(nodes["area_mm2"], dtype=np.float64)
    folds = np.asarray(nodes["fold_id"], dtype=np.int64)
    common = np.asarray(valid_first, dtype=bool) & np.asarray(valid_second, dtype=bool)
    union = np.asarray(valid_first, dtype=bool) | np.asarray(valid_second, dtype=bool)
    common_area = math.fsum(float(value) for value in area[common])
    union_area = math.fsum(float(value) for value in area[union])
    one_sided_area = math.fsum(float(value) for value in area[union & ~common])
    interior_area = math.fsum(float(value) for value in area)
    if (
        np.count_nonzero(common) < int(pair_config["minimum_common_support_node_count"])
        or common_area / interior_area
        < float(pair_config["minimum_common_support_area_fraction_of_cad_interior"])
        or (union_area > 0.0 and one_sided_area / union_area > float(pair_config["maximum_one_sided_support_area_fraction_of_union"]))
    ):
        return {"actionable": False, "reason": "PAIR_SUPPORT_GATE", "common": common}
    design_all = nuisance_design(nodes, fixed_centroid)
    rank, condition = normalized_design_rank_condition(design_all[common], area[common])
    if (
        rank != int(pair_config["required_common_support_rigid_rank"])
        or condition > float(pair_config["maximum_common_support_rigid_condition_number"])
    ):
        return {"actionable": False, "reason": "PAIR_DESIGN_GATE", "common": common}
    response = np.asarray(feature_first) - np.asarray(feature_second)
    prediction = np.full_like(response, np.nan)
    parameters: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for held_fold, base_training in enumerate(fold_training_masks):
        training = np.asarray(base_training, dtype=bool) & common
        training_area = math.fsum(float(value) for value in area[training])
        fold_rank, fold_condition = normalized_design_rank_condition(
            design_all[training], area[training]
        )
        if (
            np.count_nonzero(training)
            < int(pair_config["per_fold_per_row_minimum_post_halo_training_node_count"])
            or training_area / common_area
            < float(pair_config["per_fold_per_row_minimum_post_halo_training_area_fraction_of_that_row_common_support"])
            or fold_rank != int(pair_config["per_fold_per_row_required_rigid_rank"])
            or fold_condition > float(pair_config["per_fold_per_row_maximum_condition_number"])
        ):
            return {
                "actionable": False,
                "reason": "FOLD_TRAINING_GATE",
                "failed_fold": held_fold,
                "common": common,
            }
        ids = np.flatnonzero(training)
        try:
            parameter, retained = least_trimmed_nuisance(
                design=design_all[training],
                response=response[training],
                area=area[training],
                node_ids=ids,
                retained_area_fraction=float(config["retained_training_area_fraction"]),
                refit_count=int(config["least_trimmed_iteration_count"]),
                maximum_condition_number=float(
                    pair_config["per_fold_per_row_maximum_condition_number"]
                ),
            )
        except ValueError:
            return {
                "actionable": False,
                "reason": "LTS_RETAINED_DESIGN_GATE",
                "failed_fold": held_fold,
                "common": common,
            }
        translation = parameter[:3]
        rotation = parameter[3:]
        if (
            np.any(np.abs(translation) > float(config["translation_bound_per_axis_mm"]))
            or np.linalg.norm(rotation)
            > math.radians(float(config["rotation_norm_bound_deg"]))
        ):
            return {
                "actionable": False,
                "reason": "NUISANCE_BOUND_GATE",
                "failed_fold": held_fold,
                "common": common,
            }
        held = common & (folds == held_fold)
        prediction[held] = np.einsum("nij,j->ni", design_all[held], parameter)
        parameters.append(parameter)
        fold_rows.append(
            {
                "held_fold": held_fold,
                "training_node_count": int(np.count_nonzero(training)),
                "training_area_mm2": training_area,
                "retained_node_count": int(np.count_nonzero(retained)),
                "parameters": parameter.tolist(),
                "rank": fold_rank,
                "condition": fold_condition,
            }
        )
    translations = np.asarray([row[:3] for row in parameters])
    rotations = np.asarray([row[3:] for row in parameters])
    translation_dispersion = max(
        float(np.linalg.norm(first - second))
        for first, second in itertools.combinations(translations, 2)
    )
    rotation_dispersion_deg = math.degrees(
        max(
            float(np.linalg.norm(first - second))
            for first, second in itertools.combinations(rotations, 2)
        )
    )
    if (
        translation_dispersion > float(config["maximum_fold_translation_dispersion_mm"])
        or rotation_dispersion_deg > float(config["maximum_fold_rotation_dispersion_deg"])
    ):
        return {"actionable": False, "reason": "FOLD_DISPERSION_GATE", "common": common}
    residual = np.full_like(response, np.nan)
    residual[common] = response[common] - prediction[common]
    return {
        "actionable": True,
        "common": common,
        "union": union,
        "common_area_mm2": common_area,
        "union_area_mm2": union_area,
        "one_sided_area_fraction": one_sided_area / union_area if union_area else 0.0,
        "rank": rank,
        "condition": condition,
        "prediction": prediction,
        "residual": residual,
        "fold_rows": fold_rows,
        "translation_dispersion_mm": translation_dispersion,
        "rotation_dispersion_deg": rotation_dispersion_deg,
    }


def induced_components(
    *,
    active: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    minimum_node_count: int,
    minimum_area_mm2: float,
    minimum_span_mm: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    active = np.asarray(active, dtype=bool)
    edges = np.asarray(edges, dtype=np.int64)
    kept_edges = edges[active[edges[:, 0]] & active[edges[:, 1]]]
    labels = np.full(len(nodes), -1, dtype=np.int64)
    if not np.any(active):
        return [], labels
    union = CAD.UnionFind(len(nodes))
    for first, second in kept_edges:
        union.union(int(first), int(second))
    groups: dict[int, list[int]] = defaultdict(list)
    for node_id in np.flatnonzero(active):
        groups[union.find(int(node_id))].append(int(node_id))
    graph = CAD.weighted_node_graph(len(nodes), edges, edge_weights)
    retained: list[dict[str, Any]] = []
    area = np.asarray(nodes["area_mm2"], dtype=np.float64)
    edge_weight_lookup = {
        (int(first), int(second)): float(weight)
        for (first, second), weight in zip(edges, edge_weights, strict=True)
    }
    for members in sorted(groups.values(), key=lambda row: min(row)):
        ids = np.asarray(sorted(members), dtype=np.int64)
        component_area = math.fsum(float(value) for value in area[ids])
        if len(ids) < int(minimum_node_count) or component_area < float(minimum_area_mm2):
            continue
        member_mask = np.zeros(len(nodes), dtype=bool)
        member_mask[ids] = True
        component_edge_mask = member_mask[edges[:, 0]] & member_mask[edges[:, 1]]
        component_edges = edges[component_edge_mask]
        component_weights = np.asarray(edge_weights)[component_edge_mask]
        component_graph = CAD.weighted_node_graph(len(nodes), component_edges, component_weights)
        anchor = int(ids[0])
        first_distance = dijkstra(component_graph, directed=False, indices=anchor)
        maximum = float(np.max(first_distance[ids]))
        first_endpoint = int(np.min(ids[first_distance[ids] == maximum]))
        second_distance = dijkstra(component_graph, directed=False, indices=first_endpoint)
        span = float(np.max(second_distance[ids]))
        if span < float(minimum_span_mm):
            continue
        distance_matrix = dijkstra(component_graph, directed=False, indices=ids)[:, ids]
        if not np.all(np.isfinite(distance_matrix)):
            raise ValueError("retained component is graph-disconnected")
        costs = np.asarray(
            [
                math.fsum(
                    float(distance_matrix[row, column] * area[node_id])
                    for column, node_id in enumerate(ids)
                )
                for row in range(len(ids))
            ],
            dtype=np.float64,
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
    return retained, labels


def attribute_components(
    *,
    components: list[dict[str, Any]],
    area: np.ndarray,
    feature_first: np.ndarray,
    feature_second: np.ndarray,
    prediction: np.ndarray,
    first_token: str,
    second_token: str,
    minimum_ratio: float,
    major_fraction: float,
    denominator_floor: float,
) -> dict[str, Any]:
    if not components:
        return {"actionable": False, "reason": "NO_CENTRAL_COMPONENTS", "components": components}
    pair_energies = []
    for component in components:
        ids = np.asarray(component["node_ids"], dtype=np.int64)
        corrected_first = feature_first[ids] - 0.5 * prediction[ids]
        corrected_second = feature_second[ids] + 0.5 * prediction[ids]
        component_area = math.fsum(float(value) for value in area[ids])
        energy_first = math.fsum(
            float(area[node_id] * np.dot(vector, vector))
            for node_id, vector in zip(ids, corrected_first, strict=True)
        ) / component_area
        energy_second = math.fsum(
            float(area[node_id] * np.dot(vector, vector))
            for node_id, vector in zip(ids, corrected_second, strict=True)
        ) / component_area
        ratio = max(energy_first, energy_second) / max(
            min(energy_first, energy_second), float(denominator_floor)
        )
        pair_energy = math.fsum(
            float(area[node_id] * np.dot(first - second, first - second))
            for node_id, first, second in zip(ids, corrected_first, corrected_second, strict=True)
        )
        higher = first_token if energy_first > energy_second else second_token
        if energy_first == energy_second:
            higher = None
        component.update(
            {
                "first_member_energy_mm2": energy_first,
                "second_member_energy_mm2": energy_second,
                "energy_ratio": ratio,
                "higher_energy_opaque_member": higher,
                "pair_energy_mm4": pair_energy,
            }
        )
        pair_energies.append(pair_energy)
    total = math.fsum(pair_energies)
    if total <= 0.0:
        return {"actionable": False, "reason": "ZERO_PAIRWISE_ENERGY", "components": components}
    major_tokens = []
    for component in components:
        fraction = float(component["pair_energy_mm4"] / total)
        component["pair_energy_fraction"] = fraction
        component["major"] = fraction >= float(major_fraction)
        if component["major"]:
            if component["energy_ratio"] < float(minimum_ratio) or component["higher_energy_opaque_member"] is None:
                return {"actionable": False, "reason": "MAJOR_COMPONENT_ATTRIBUTION_GATE", "components": components}
            major_tokens.append(component["higher_energy_opaque_member"])
    if not major_tokens:
        return {"actionable": False, "reason": "NO_MAJOR_COMPONENT", "components": components}
    if len(set(major_tokens)) != 1:
        return {"actionable": False, "reason": "MAJOR_COMPONENT_TOKEN_DISAGREEMENT", "components": components}
    return {
        "actionable": True,
        "attributed_opaque_member": major_tokens[0],
        "total_pair_energy_mm4": total,
        "components": components,
    }


def area_weighted_jaccard(
    first_ids: Sequence[int], second_ids: Sequence[int], area: np.ndarray
) -> float:
    first = set(int(value) for value in first_ids)
    second = set(int(value) for value in second_ids)
    if not first and not second:
        return 1.0
    union = sorted(first | second)
    if not union:
        return 1.0
    intersection = sorted(first & second)
    numerator = math.fsum(float(area[index]) for index in intersection)
    denominator = math.fsum(float(area[index]) for index in union)
    return numerator / denominator if denominator else 0.0


def canonical_component_bijection(
    central: Sequence[Mapping[str, Any]],
    variant: Sequence[Mapping[str, Any]],
    area: np.ndarray,
) -> tuple[list[int], np.ndarray] | None:
    count = len(central)
    if count != len(variant):
        return None
    if count == 0:
        return [], np.empty((0, 0), dtype=np.float64)
    matrix = np.asarray(
        [
            [area_weighted_jaccard(c["node_ids"], v["node_ids"], area) for v in variant]
            for c in central
        ],
        dtype=np.float64,
    )
    # Scalable lexicographic prefix refinement.  Hungarian supplies the maximum
    # remaining score; at each central row we select the lowest column that can
    # still attain the frozen optimum within the explicit binary64 tolerance.
    rows, columns = linear_sum_assignment(-matrix)
    optimum = math.fsum(float(matrix[int(row), int(column)]) for row, column in zip(rows, columns, strict=True))
    available = list(range(count))
    prefix_score = 0.0
    assignment: list[int] = []
    tolerance = 1e-12 * max(1.0, abs(optimum))
    for row in range(count):
        selected = None
        for column in available:
            remaining_columns = [value for value in available if value != column]
            if row + 1 < count:
                sub = matrix[row + 1 :, remaining_columns]
                sub_rows, sub_columns = linear_sum_assignment(-sub)
                remaining_score = math.fsum(
                    float(sub[int(local_row), int(local_column)])
                    for local_row, local_column in zip(sub_rows, sub_columns, strict=True)
                )
            else:
                remaining_score = 0.0
            candidate = math.fsum((prefix_score, float(matrix[row, column]), remaining_score))
            if abs(candidate - optimum) <= tolerance:
                selected = int(column)
                break
        if selected is None:
            return None
        assignment.append(selected)
        available.remove(selected)
        prefix_score = math.fsum((prefix_score, float(matrix[row, selected])))
    return assignment, matrix


def split_noise_from_rows(
    identity_zero: np.ndarray,
    identity_one: np.ndarray,
    spatial_zero: np.ndarray,
    spatial_one: np.ndarray,
    valid_masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    complete = np.logical_and.reduce([np.asarray(mask, dtype=bool) for mask in valid_masks])
    output = np.full(len(identity_zero), math.inf, dtype=np.float64)
    output[complete] = np.maximum(
        0.5 * np.linalg.norm(identity_zero[complete] - identity_one[complete], axis=1),
        0.5 * np.linalg.norm(spatial_zero[complete] - spatial_one[complete], axis=1),
    )
    return output, complete


def robustness_row_ids() -> list[str]:
    return [f"R{index:02d}" for index in range(32)]
