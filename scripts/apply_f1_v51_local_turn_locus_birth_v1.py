#!/usr/bin/env python3
"""Source-only local-turn-locus candidate births after frozen V51.

The algorithm is intentionally label-free and identity-free.  It extracts a
bounded, voxelized source mesh cloud, marks local normal-change ridges, splits
their longitudinal support at physical gaps, and snaps a supported ridge axis
to the cross product of its two normal modes when the directions agree.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from scripts.evaluate_f1_against_manual_gold import _endpoints, pair_metrics


RESULT_KIND = "sherman_qc_f1_v51_local_turn_locus_birth_result_v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _unit(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate direction")
    result = result / norm
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_unit(left), _unit(right)))), 0.0, 1.0))))


def _geometry_cloud(scan_path: Path, profile: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise ValueError("source scan is not a triangle mesh")
    source_vertices = int(len(mesh.vertices))
    source_triangles = int(len(mesh.triangles))
    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    cloud.normals = mesh.vertex_normals
    cloud = cloud.voxel_down_sample(float(profile["voxel_size_mm"]))
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-9)
    points = points[finite]
    normals = normals[finite] / lengths[finite, None]
    quantized = np.rint(points * 1_000.0).astype(np.int64)
    quantized_normals = np.rint(normals * 1_000_000.0).astype(np.int64)
    order = np.lexsort((
        quantized_normals[:, 2], quantized_normals[:, 1], quantized_normals[:, 0],
        quantized[:, 2], quantized[:, 1], quantized[:, 0],
    ))
    points = points[order]
    normals = normals[order]
    maximum = int(profile["maximum_working_points"])
    if len(points) > maximum:
        indexes = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        points = points[indexes]
        normals = normals[indexes]
    return points, normals, {
        "source_vertex_count": source_vertices,
        "source_triangle_count": source_triangles,
        "working_point_count": int(len(points)),
    }


def _local_turn_field(
    points: np.ndarray,
    normals: np.ndarray,
    profile: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(int(profile["local_neighbor_count"]), len(points))
    tree = cKDTree(points)
    second = np.zeros(len(points), dtype=np.float64)
    contrast = np.zeros(len(points), dtype=np.float64)
    neighbor_count = np.zeros(len(points), dtype=np.int32)
    radius = float(profile["local_radius_mm"])
    for start in range(0, len(points), 20_000):
        stop = min(start + 20_000, len(points))
        distances, indexes = tree.query(
            points[start:stop], k=count, distance_upper_bound=radius, workers=1
        )
        if count == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        valid = (indexes < len(points)) & np.isfinite(distances)
        safe_indexes = np.where(valid, indexes, 0)
        local_normals = normals[safe_indexes]
        weights = valid.astype(np.float64)
        counts = np.sum(weights, axis=1)
        moment = np.einsum("bki,bkj,bk->bij", local_normals, local_normals, weights)
        moment /= np.maximum(counts[:, None, None], 1.0)
        eigenvalues = np.linalg.eigvalsh(moment)
        second[start:stop] = eigenvalues[:, 1]
        dot = np.abs(np.einsum("bki,bi->bk", local_normals, normals[start:stop]))
        angles = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
        angles[~valid] = np.nan
        with np.errstate(all="ignore"):
            values = np.nanquantile(angles, 0.90, axis=1)
        contrast[start:stop] = np.nan_to_num(values, nan=0.0)
        neighbor_count[start:stop] = counts.astype(np.int32)
    mask = (
        (neighbor_count >= int(profile["minimum_local_neighbors"]))
        & (contrast >= float(profile["minimum_normal_change_deg"]))
        & (second >= float(profile["minimum_normal_second_eigenvalue"]))
    )
    return np.flatnonzero(mask), second, contrast


def _cell_components(indexes: np.ndarray, points: np.ndarray, profile: Mapping[str, Any]) -> list[np.ndarray]:
    cell_size = float(profile["component_cell_mm"])
    cells = np.floor(points[indexes] / cell_size).astype(np.int64)
    cell_rows: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for point_index, cell in zip(indexes.tolist(), cells.tolist()):
        cell_rows[tuple(int(value) for value in cell)].append(int(point_index))
    occupied = set(cell_rows)
    components = []
    minimum = int(profile["minimum_component_points"])
    while occupied:
        seed = min(occupied)
        occupied.remove(seed)
        stack = [seed]
        member_cells = []
        while stack:
            cell = stack.pop()
            member_cells.append(cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        if neighbor in occupied:
                            occupied.remove(neighbor)
                            stack.append(neighbor)
        members = np.asarray(sorted(index for cell in member_cells for index in cell_rows[cell]), dtype=np.int64)
        if len(members) >= minimum:
            components.append(members)
    components.sort(key=lambda row: (-len(row), int(row[0])))
    return components


def _circular_span_deg(angles: np.ndarray) -> float:
    values = np.sort(np.mod(angles, 2.0 * np.pi))
    if len(values) < 2:
        return 0.0
    gaps = np.diff(np.r_[values, values[0] + 2.0 * np.pi])
    return float(np.degrees(2.0 * np.pi - np.max(gaps)))


def _normal_cross_axis(local_normals: np.ndarray, locus_axis: np.ndarray, profile: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    oriented = local_normals.copy()
    reference = oriented[0]
    oriented[np.einsum("ij,j->i", oriented, reference) < 0.0] *= -1.0
    center = np.mean(oriented, axis=0)
    covariance = (oriented - center).T @ (oriented - center) / max(len(oriented) - 1, 1)
    direction = np.linalg.eigh(covariance)[1][:, -1]
    coordinate = oriented @ direction
    low = oriented[coordinate <= np.quantile(coordinate, 0.20)]
    high = oriented[coordinate >= np.quantile(coordinate, 0.80)]
    if not len(low) or not len(high):
        return locus_axis, 90.0
    first = _unit(np.mean(low, axis=0))
    second = _unit(np.mean(high, axis=0))
    cross = np.cross(first, second)
    if float(np.linalg.norm(cross)) <= 1e-9:
        return locus_axis, 90.0
    cross = _unit(cross)
    error = _angle_deg(cross, locus_axis)
    return (cross if error <= float(profile["axis_snap_maximum_angle_deg"]) else locus_axis), error


def _component_births(
    component: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    second: np.ndarray,
    contrast: np.ndarray,
    profile: Mapping[str, Any],
    component_index: int,
) -> list[dict[str, Any]]:
    component_points = points[component]
    center = np.mean(component_points, axis=0)
    covariance = (component_points - center).T @ (component_points - center) / max(len(component_points) - 1, 1)
    axis = _unit(np.linalg.eigh(covariance)[1][:, -1])
    projections = (component_points - center) @ axis
    width = float(profile["longitudinal_bin_mm"])
    bins = np.floor((projections - float(np.min(projections))) / width + 1e-12).astype(np.int64)
    unique = sorted(set(int(value) for value in bins))
    runs: list[list[int]] = []
    for value in unique:
        if not runs or (value - runs[-1][-1]) * width > float(profile["maximum_supported_gap_mm"]):
            runs.append([value])
        else:
            runs[-1].append(value)
    births = []
    for run_index, run in enumerate(runs, 1):
        if len(run) < int(profile["minimum_supported_bin_count"]):
            continue
        mask = np.isin(bins, np.asarray(run, dtype=np.int64))
        member_indexes = component[mask]
        member_points = points[member_indexes]
        centers = []
        for bin_index in run:
            local = member_points[bins[mask] == bin_index]
            centers.append(np.mean(local, axis=0))
        centers_array = np.asarray(centers, dtype=np.float64)
        locus_center = np.mean(centers_array, axis=0)
        locus_covariance = (centers_array - locus_center).T @ (centers_array - locus_center) / max(len(centers_array) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(locus_covariance)
        locus_axis = _unit(eigenvectors[:, -1])
        total = float(np.sum(np.maximum(eigenvalues, 0.0)))
        linearity = float(max(eigenvalues[-1], 0.0) / max(total, 1e-12))
        locus_axis, snap_error = _normal_cross_axis(normals[member_indexes], locus_axis, profile)
        center_projection = (centers_array - locus_center) @ locus_axis
        residual = centers_array - locus_center - center_projection[:, None] * locus_axis
        locus_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        member_projection = (member_points - locus_center) @ locus_axis
        radial = member_points - locus_center - member_projection[:, None] * locus_axis
        radial_distance = np.linalg.norm(radial, axis=1)
        transverse_q80 = float(np.quantile(radial_distance, 0.80))
        span_low = float(np.min(center_projection) - width / 2.0)
        span_high = float(np.max(center_projection) + width / 2.0)
        span = span_high - span_low
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, locus_axis))) > 0.8:
            reference = np.asarray([0.0, 1.0, 0.0])
        first = _unit(np.cross(locus_axis, reference))
        second_basis = np.cross(locus_axis, first)
        valid_radial = radial_distance > 1e-9
        angular_span = _circular_span_deg(np.arctan2(radial[valid_radial] @ second_basis, radial[valid_radial] @ first))
        median_contrast = float(np.median(contrast[member_indexes]))
        median_second = float(np.median(second[member_indexes]))
        passed = bool(
            span >= float(profile["minimum_span_mm"])
            and locus_rms <= float(profile["maximum_locus_rms_mm"])
            and transverse_q80 <= float(profile["maximum_transverse_q80_mm"])
            and linearity >= float(profile["minimum_locus_linearity"])
            and angular_span <= float(profile["maximum_radial_angular_span_deg"])
        )
        if not passed:
            continue
        start = locus_center + span_low * locus_axis
        end = locus_center + span_high * locus_axis
        identity = {
            "component": int(component_index), "run": int(run_index),
            "line_start": np.round(start, 9).tolist(), "line_end": np.round(end, 9).tolist(),
        }
        token = hashlib.sha256(canonical_bytes(identity)).hexdigest()[:20]
        births.append({
            "bend_id": f"V51LTLB1-{token}",
            "line_start": start.tolist(),
            "line_end": end.tolist(),
            "source_kind": "local_turn_locus_birth_v1",
            "selection_route": "normal_change_ridge_supported_span_axis_snap",
            "source_hypothesis_id": f"component-{component_index}-run-{run_index}",
            "selector_score": float(median_contrast * median_second * span / max(1.0 + locus_rms, 1e-9)),
            "local_turn_locus_evidence": {
                "component_point_count": int(len(component)),
                "run_point_count": int(len(member_indexes)),
                "supported_bin_count": int(len(run)),
                "span_mm": span,
                "locus_rms_mm": locus_rms,
                "transverse_q80_mm": transverse_q80,
                "locus_linearity": linearity,
                "median_normal_change_deg": median_contrast,
                "median_normal_second_eigenvalue": median_second,
                "axis_snap_error_deg": snap_error,
                "radial_angular_span_deg": angular_span,
            },
        })
    return births


def _duplicate(left: Mapping[str, Any], right: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    metrics = pair_metrics(_endpoints(left), _endpoints(right))
    return bool(
        metrics["axis_angle_deg"] <= float(profile["duplicate_angle_deg"])
        and metrics["transverse_offset_mm"] <= float(profile["duplicate_offset_mm"])
        and metrics["shorter_span_overlap"] >= float(profile["duplicate_shorter_span_overlap"])
    )


def generate_births(scan_path: Path, profile: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points, normals, diagnostics = _geometry_cloud(scan_path, profile)
    indexes, second, contrast = _local_turn_field(points, normals, profile)
    components = _cell_components(indexes, points, profile)
    raw = []
    for component_index, component in enumerate(components, 1):
        raw.extend(_component_births(component, points, normals, second, contrast, profile, component_index))
    raw.sort(key=lambda row: (-float(row["selector_score"]), str(row["bend_id"])))
    kept = []
    for row in raw:
        if any(_duplicate(row, incumbent, profile) for incumbent in kept):
            continue
        kept.append(row)
        if len(kept) >= int(profile["maximum_births_per_scan"]):
            break
    diagnostics.update({
        "turn_ridge_point_count": int(len(indexes)),
        "turn_ridge_component_count": int(len(components)),
        "qualified_birth_count_before_nms_and_cap": int(len(raw)),
        "birth_count_after_nms_and_cap": int(len(kept)),
    })
    return kept, diagnostics


def apply_births(
    v51_result: Mapping[str, Any],
    scan_path: Path,
    parameter_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if v51_result.get("source_only") is not True or v51_result.get("research_only") is not True:
        raise ValueError("V51 input must be source-only and research-only")
    if any(v51_result.get(name) is not False for name in ("labels_used", "cad_used", "part_identity_used")):
        raise ValueError("V51 input reports forbidden evidence")
    if parameter_lock.get("status") != "LOCKED_BEFORE_CASE_OUTCOME_EVALUATION":
        raise ValueError("parameter precommit is not locked")
    profile = dict(parameter_lock["parameters"])
    births, diagnostics = generate_births(scan_path, profile)
    incumbents = [deepcopy(row) for row in v51_result.get("predictions") or []]
    additions = []
    for row in births:
        if any(_duplicate(row, incumbent, profile) for incumbent in incumbents):
            continue
        if any(_duplicate(row, incumbent, profile) for incumbent in additions):
            continue
        additions.append(row)
    predictions = incumbents + additions
    return {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "predictions": predictions,
        "candidate_count": len(predictions),
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "source_only": True,
        "research_only": True,
        "promotion_eligible": False,
        "generalization_evidence": False,
        "local_turn_locus_birth_audit": {
            **diagnostics,
            "incumbent_count": len(incumbents),
            "generated_birth_count": len(births),
            "emitted_addition_count": len(additions),
        },
        "parameter_lock_sha256": hashlib.sha256(canonical_bytes(parameter_lock)).hexdigest(),
    }
