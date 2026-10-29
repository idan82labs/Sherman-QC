#!/usr/bin/env python3
"""Select bounded registration refinements without sacrificing CAD bend support.

EXP_CAD_0021 is a truth-denying registration diagnostic.  It generates a
small, deterministic set of rigid ICP refinements around the EXP15 alignment,
scores them on the *same CAD surface samples that were already visible under
the baseline*, and rejects any candidate that reduces a native EXP17 bend
carrier's occupied-axis-bin support.  It never opens labels, expected counts,
or an evaluator.

The frozen-visible reference set is deliberate: several authenticated SLDPRT
meshes are incomplete tessellations.  Optimizing all scan-to-mesh distances
would reward a transform for pulling unrepresented scan surfaces toward the
remaining CAD triangles.  Here the global comparison is reference-to-scan and
is restricted to reference samples that the baseline already observed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP18_PATH = ROOT / "scripts" / "run_exp_cad_0018_native_carrier_support.py"
EXP18_SPEC = importlib.util.spec_from_file_location("exp18_carrier_support", EXP18_PATH)
EXP18 = importlib.util.module_from_spec(EXP18_SPEC)
assert EXP18_SPEC and EXP18_SPEC.loader
EXP18_SPEC.loader.exec_module(EXP18)

from research.f1_deterministic_reference_surface_v1 import sample_surface  # noqa: E402


VISIBLE_REFERENCE_RADIUS_MM = 5.0
GLOBAL_CLOSE_RADIUS_MM = 2.0
MAX_TRANSLATION_MM = 2.5
MAX_ROTATION_DEG = 1.0
MIN_CARRIER_POINT_RATIO = 0.90
MAX_PROTECTED_MEDIAN_RESIDUAL_REGRESSION_MM = 0.10
GLOBAL_FRACTION_TOLERANCE = 0.001
GLOBAL_RESIDUAL_TOLERANCE_MM = 0.02
MIN_GLOBAL_FRACTION_GAIN = 0.005
MIN_GLOBAL_P90_GAIN_MM = 0.05
MIN_GLOBAL_MEDIAN_GAIN_MM = 0.03
EVALUATION_VOXEL_MM = 0.75
MIN_CURVATURE_ANGULAR_SPAN_DEG = 20.0
MIN_NORMAL_RADIAL_MEDIAN = math.cos(math.radians(25.0))
MIN_NORMAL_RADIAL_P10 = math.cos(math.radians(45.0))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        canonicalize_floats(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonicalize_floats(value: Any) -> Any:
    """Remove sub-nanometric library-order noise from persisted evidence."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)
    if isinstance(value, Mapping):
        return {str(key): canonicalize_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize_floats(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def canonicalize_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.round(np.asarray(transform, dtype=np.float64), decimals=10)
    matrix[3] = [0.0, 0.0, 0.0, 1.0]
    return matrix


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("points must be finite Nx3")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be finite 4x4")
    return source @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_delta_stats(transform: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_mm": float(np.linalg.norm(matrix[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(cosine))),
        "rotation_orthogonality_error": orthogonality,
        "rotation_determinant": determinant,
    }


def absolute_similarity_stats(transform: np.ndarray) -> dict[str, Any]:
    """Validate an absolute scan→CAD transform without hiding reflections."""
    matrix = np.asarray(transform, dtype=np.float64)
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular))
    normalized = linear / scale if scale > 0 else linear
    determinant = float(np.linalg.det(normalized))
    orthogonality = float(np.linalg.norm(normalized.T @ normalized - np.eye(3), ord="fro"))
    uniform_scale_error = float(np.max(np.abs(singular - scale)))
    proper = bool(
        scale > 0
        and determinant > 0
        and abs(determinant - 1.0) <= 1e-6
        and orthogonality <= 1e-6
        and uniform_scale_error <= 1e-6
    )
    return {
        "proper_similarity_transform": proper,
        "normalized_rotation_determinant": determinant,
        "normalized_rotation_orthogonality_error": orthogonality,
        "uniform_scale": scale,
        "uniform_scale_error": uniform_scale_error,
        "reflection_detected": determinant < 0,
    }


def density_capped_points_and_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cloud = _make_cloud(points)
    cloud = cloud.voxel_down_sample(EVALUATION_VOXEL_MM)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * EVALUATION_VOXEL_MM, 2.5), max_nn=50)
    )
    capped = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    order = np.lexsort((capped[:, 2], capped[:, 1], capped[:, 0]))
    capped = np.ascontiguousarray(capped[order], dtype=np.float64)
    normals = np.ascontiguousarray(normals[order], dtype=np.float64)
    if not len(capped) or not np.isfinite(capped).all() or not np.isfinite(normals).all():
        raise ValueError("density cap produced invalid points or normals")
    return capped, normals


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    anchor = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ direction)))]
    first = np.cross(direction, anchor)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return first, second


def _circular_90pct_span_deg(radial_vectors: np.ndarray, axis: np.ndarray) -> float:
    if len(radial_vectors) < 3:
        return 0.0
    first, second = _perpendicular_basis(axis)
    angles = np.mod(
        np.arctan2(radial_vectors @ second, radial_vectors @ first),
        2.0 * math.pi,
    )
    angles.sort()
    count = max(3, int(math.ceil(0.90 * len(angles))))
    doubled = np.concatenate((angles, angles + 2.0 * math.pi))
    widths = doubled[count - 1 : count - 1 + len(angles)] - doubled[: len(angles)]
    return float(np.degrees(np.min(widths)))


def _longest_true_run(bitset: Sequence[bool]) -> int:
    best = current = 0
    for value in bitset:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def build_frozen_event_neighborhoods(
    baseline_points: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    *,
    thickness_mm: float,
) -> list[dict[str, Any]]:
    neighborhoods: list[dict[str, Any]] = []
    broad_margin = max(3.0 * float(thickness_mm), 3.0)
    for event in events:
        start = np.asarray(event["line_start_mm"], dtype=np.float64)
        end = np.asarray(event["line_end_mm"], dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        axis = direction / length
        relative = baseline_points - start
        parameters = relative @ axis
        radial_vectors = relative - parameters[:, None] * axis[None, :]
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        max_radius = max(float(value) for value in event["radii_mm"])
        mask = (
            (parameters >= -broad_margin)
            & (parameters <= length + broad_margin)
            & (radial_distances <= max_radius + broad_margin)
        )
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"empty frozen neighborhood for {event['event_id']}")
        bin_count = max(8, min(48, int(math.ceil(length / max(4.0 * thickness_mm, 4.0)))))
        clipped = np.clip(parameters[indices], 0.0, np.nextafter(length, 0.0))
        frozen_bins = np.minimum((clipped / length * bin_count).astype(np.int64), bin_count - 1)
        neighborhoods.append(
            {
                "event_id": str(event["event_id"]),
                "point_indices": indices,
                "frozen_bin_indices": frozen_bins,
                "bin_count": bin_count,
            }
        )
    return neighborhoods


def evaluate_frozen_event_support(
    points: np.ndarray,
    normals: np.ndarray,
    event: Mapping[str, Any],
    neighborhood: Mapping[str, Any],
    *,
    thickness_mm: float,
) -> tuple[dict[str, Any], np.ndarray]:
    start = np.asarray(event["line_start_mm"], dtype=np.float64)
    end = np.asarray(event["line_end_mm"], dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    axis = direction / length
    indices = np.asarray(neighborhood["point_indices"], dtype=np.int64)
    bins = np.asarray(neighborhood["frozen_bin_indices"], dtype=np.int64)
    local = points[indices]
    local_normals = normals[indices]
    relative = local - start
    parameters = relative @ axis
    radial_vectors = relative - parameters[:, None] * axis[None, :]
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radii = np.asarray(event["radii_mm"], dtype=np.float64)
    residual_matrix = np.abs(radial_distances[:, None] - radii[None, :])
    assignments = np.argmin(residual_matrix, axis=1)
    residuals = residual_matrix[np.arange(len(local)), assignments]
    tolerance = max(0.8, 0.6 * float(thickness_mm))
    carrier = (
        (parameters >= -tolerance)
        & (parameters <= length + tolerance)
        & (residuals <= tolerance)
    )
    bin_count = int(neighborhood["bin_count"])
    occupied = np.bincount(bins[carrier], minlength=bin_count)
    supported = occupied >= 3
    qualified = np.zeros(bin_count, dtype=bool)
    per_bin: list[dict[str, Any]] = []
    plane_margin_threshold = max(0.05, 0.05 * float(thickness_mm))
    for bin_index in range(bin_count):
        bin_mask = carrier & (bins == bin_index)
        best: Optional[dict[str, Any]] = None
        for radius_index in range(len(radii)):
            radius_mask = bin_mask & (assignments == radius_index)
            group = local[radius_mask]
            if len(group) < 3:
                continue
            group_radial = radial_vectors[radius_mask]
            group_lengths = np.linalg.norm(group_radial, axis=1)
            radial_units = group_radial / np.maximum(group_lengths[:, None], 1e-12)
            normal_alignment = np.abs(np.sum(local_normals[radius_mask] * radial_units, axis=1))
            angular_span = _circular_90pct_span_deg(group_radial, axis)
            centered = group - np.median(group, axis=0)
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            plane_normal = vh[-1]
            signed_plane = centered @ plane_normal
            plane_mad = float(np.median(np.abs(signed_plane - np.median(signed_plane))))
            signed_cylinder = group_lengths - float(radii[radius_index])
            cylinder_mad = float(np.median(np.abs(signed_cylinder - np.median(signed_cylinder))))
            margin = plane_mad - cylinder_mad
            row = {
                "radius_index": radius_index,
                "point_count": int(len(group)),
                "angular_span_deg": angular_span,
                "normal_radial_median": float(np.median(normal_alignment)),
                "normal_radial_p10": float(np.percentile(normal_alignment, 10)),
                "plane_mad_mm": plane_mad,
                "cylinder_mad_mm": cylinder_mad,
                "plane_minus_cylinder_mad_mm": margin,
            }
            row["qualified"] = bool(
                angular_span >= MIN_CURVATURE_ANGULAR_SPAN_DEG
                and row["normal_radial_median"] >= MIN_NORMAL_RADIAL_MEDIAN
                and row["normal_radial_p10"] >= MIN_NORMAL_RADIAL_P10
                and margin >= plane_margin_threshold
            )
            if best is None or (
                bool(row["qualified"]), margin, angular_span, len(group)
            ) > (
                bool(best["qualified"]),
                float(best["plane_minus_cylinder_mad_mm"]),
                float(best["angular_span_deg"]),
                int(best["point_count"]),
            ):
                best = row
        if best is not None and bool(best["qualified"]):
            qualified[bin_index] = True
        per_bin.append(best or {"qualified": False, "point_count": 0})
    full_mask = np.zeros(len(points), dtype=bool)
    full_mask[indices[carrier]] = True
    carrier_residuals = residuals[carrier]
    metric = {
        "event_id": str(event["event_id"]),
        "span_mm": length,
        "native_radii_mm": radii.tolist(),
        "radial_tolerance_mm": tolerance,
        "bin_count": bin_count,
        "min_points_per_bin": 3,
        "frozen_neighborhood_point_count": int(len(indices)),
        "carrier_point_count": int(np.sum(carrier)),
        "supported_bin_count": int(np.sum(supported)),
        "carrier_coverage_fraction": float(np.mean(supported)),
        "occupied_bin_point_counts": occupied.tolist(),
        "supported_bin_bitset": supported.tolist(),
        "curvature_qualified_bin_bitset": qualified.tolist(),
        "curvature_qualified_bin_count": int(np.sum(qualified)),
        "curvature_qualified_longest_run": _longest_true_run(qualified),
        "median_carrier_radial_residual_mm": float(np.median(carrier_residuals)) if len(carrier_residuals) else None,
        "p90_carrier_radial_residual_mm": float(np.percentile(carrier_residuals, 90)) if len(carrier_residuals) else None,
        "curvature_bin_diagnostics": per_bin,
    }
    return metric, full_mask


def global_visible_reference_metrics(
    candidate_scan_points: np.ndarray,
    visible_reference_points: np.ndarray,
) -> dict[str, float | int]:
    tree = cKDTree(np.asarray(candidate_scan_points, dtype=np.float64))
    distances, _ = tree.query(np.asarray(visible_reference_points, dtype=np.float64), k=1, workers=1)
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError("visible-reference query produced invalid distances")
    return {
        "visible_reference_point_count": int(len(distances)),
        "within_1mm_fraction": float(np.mean(distances <= 1.0)),
        "within_2mm_fraction": float(np.mean(distances <= GLOBAL_CLOSE_RADIUS_MM)),
        "median_distance_mm": float(np.median(distances)),
        "p90_distance_mm": float(np.percentile(distances, 90)),
    }


def event_support_metrics(
    points: np.ndarray,
    normals: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    neighborhoods: Sequence[Mapping[str, Any]],
    *,
    thickness_mm: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    union = np.zeros(len(points), dtype=bool)
    rows: list[dict[str, Any]] = []
    for event, neighborhood in zip(events, neighborhoods):
        metric, mask = evaluate_frozen_event_support(
            points,
            normals,
            event,
            neighborhood,
            thickness_mm=thickness_mm,
        )
        rows.append(metric)
        union |= mask
    return rows, union


def protected_neighborhood_decision(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    if len(baseline) != len(candidate):
        return False, ["EVENT_COUNT_CHANGED"]
    reasons: list[str] = []
    for base, trial in zip(baseline, candidate):
        event_id = str(base["event_id"])
        if str(trial["event_id"]) != event_id:
            reasons.append(f"{event_id}:EVENT_ORDER_CHANGED")
            continue
        if int(trial["supported_bin_count"]) < int(base["supported_bin_count"]):
            reasons.append(f"{event_id}:SUPPORTED_BIN_LOSS")
        base_qualified = np.asarray(base.get("curvature_qualified_bin_bitset", []), dtype=bool)
        trial_qualified = np.asarray(trial.get("curvature_qualified_bin_bitset", []), dtype=bool)
        if base_qualified.shape != trial_qualified.shape:
            reasons.append(f"{event_id}:CURVATURE_BIN_SCHEMA_CHANGED")
        elif np.any(base_qualified & ~trial_qualified):
            reasons.append(f"{event_id}:CURVATURE_QUALIFIED_BIN_LOSS")
        if int(trial.get("curvature_qualified_bin_count", 0)) < int(base.get("curvature_qualified_bin_count", 0)):
            reasons.append(f"{event_id}:CURVATURE_QUALIFIED_COUNT_LOSS")
        if int(trial.get("curvature_qualified_longest_run", 0)) < int(base.get("curvature_qualified_longest_run", 0)):
            reasons.append(f"{event_id}:CURVATURE_QUALIFIED_RUN_LOSS")
        base_points = int(base["carrier_point_count"])
        trial_points = int(trial["carrier_point_count"])
        if base_points and trial_points + 1e-12 < MIN_CARRIER_POINT_RATIO * base_points:
            reasons.append(f"{event_id}:CARRIER_POINT_LOSS_GT_10_PERCENT")
        base_residual = base.get("median_carrier_radial_residual_mm")
        trial_residual = trial.get("median_carrier_radial_residual_mm")
        if base_residual is not None:
            if trial_residual is None:
                reasons.append(f"{event_id}:CARRIER_RESIDUAL_BECAME_NULL")
            elif (
                int(base["supported_bin_count"]) >= max(1, int(base["bin_count"]) // 2)
                and float(trial_residual)
                > float(base_residual) + MAX_PROTECTED_MEDIAN_RESIDUAL_REGRESSION_MM + 1e-12
            ):
                reasons.append(f"{event_id}:MEDIAN_CARRIER_RESIDUAL_REGRESSION")
    return not reasons, reasons


def global_improvement_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    fraction = float(candidate["within_2mm_fraction"])
    base_fraction = float(baseline["within_2mm_fraction"])
    median = float(candidate["median_distance_mm"])
    base_median = float(baseline["median_distance_mm"])
    p90 = float(candidate["p90_distance_mm"])
    base_p90 = float(baseline["p90_distance_mm"])
    if fraction < base_fraction - GLOBAL_FRACTION_TOLERANCE - 1e-12:
        reasons.append("VISIBLE_REFERENCE_WITHIN_2MM_FRACTION_REGRESSION")
    if median > base_median + GLOBAL_RESIDUAL_TOLERANCE_MM + 1e-12:
        reasons.append("VISIBLE_REFERENCE_MEDIAN_REGRESSION")
    if p90 > base_p90 + GLOBAL_RESIDUAL_TOLERANCE_MM + 1e-12:
        reasons.append("VISIBLE_REFERENCE_P90_REGRESSION")
    strict_improvement = (
        fraction >= base_fraction + MIN_GLOBAL_FRACTION_GAIN - 1e-12
        or median <= base_median - MIN_GLOBAL_MEDIAN_GAIN_MM + 1e-12
        or p90 <= base_p90 - MIN_GLOBAL_P90_GAIN_MM + 1e-12
    )
    if not strict_improvement:
        reasons.append("NO_MATERIAL_GLOBAL_IMPROVEMENT")
    return not reasons, reasons


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> str:
    baselines = [row for row in rows if row["candidate_id"] == "BASELINE_IDENTITY"]
    if len(baselines) != 1:
        raise ValueError("exactly one baseline candidate is required")
    eligible = [
        row
        for row in rows
        if row["candidate_id"] != "BASELINE_IDENTITY"
        and bool(row["bounded_rigid_delta"])
        and bool(row["protected_neighborhood_pass"])
        and bool(row["material_global_improvement_pass"])
    ]
    if not eligible:
        return "BASELINE_IDENTITY"
    baseline_metric = baselines[0]["global_visible_reference_metrics"]

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
        metric = row["global_visible_reference_metrics"]
        delta = row["delta_stats"]
        fraction_gain = (
            float(metric["within_2mm_fraction"])
            - float(baseline_metric["within_2mm_fraction"])
        ) / MIN_GLOBAL_FRACTION_GAIN
        median_gain = (
            float(baseline_metric["median_distance_mm"])
            - float(metric["median_distance_mm"])
        ) / MIN_GLOBAL_MEDIAN_GAIN_MM
        p90_gain = (
            float(baseline_metric["p90_distance_mm"])
            - float(metric["p90_distance_mm"])
        ) / MIN_GLOBAL_P90_GAIN_MM
        material_strength = max(fraction_gain, median_gain, p90_gain)
        return (
            -material_strength,
            -fraction_gain,
            -p90_gain,
            -median_gain,
            float(delta["translation_mm"]) + 0.1 * float(delta["rotation_deg"]),
            str(row["candidate_id"]),
        )

    return str(sorted(eligible, key=rank)[0]["candidate_id"])


def _make_cloud(points: np.ndarray, normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return cloud


def generate_icp_candidates(
    baseline_points: np.ndarray,
    reference_points: np.ndarray,
    reference_normals: np.ndarray,
) -> list[tuple[str, np.ndarray, dict[str, float]]]:
    source = _make_cloud(baseline_points)
    source = source.voxel_down_sample(voxel_size=1.5)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.5, max_nn=40))
    target = _make_cloud(reference_points, reference_normals)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-8,
        relative_rmse=1e-8,
        max_iteration=80,
    )
    configurations = (
        ("P2PL_T1P5", "point_to_plane", 1.5),
        ("P2PL_T2P5", "point_to_plane", 2.5),
        ("P2PL_T4P0", "point_to_plane", 4.0),
        ("P2P_T2P5", "point_to_point", 2.5),
        ("P2P_T4P0", "point_to_point", 4.0),
    )
    rows: list[tuple[str, np.ndarray, dict[str, float]]] = []
    for candidate_id, method, threshold in configurations:
        if method == "point_to_plane":
            loss = o3d.pipelines.registration.TukeyLoss(k=min(2.0, threshold))
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        else:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            np.eye(4),
            estimation,
            criteria,
        )
        rows.append(
            (
                candidate_id,
                canonicalize_transform(registration.transformation),
                {
                    "icp_fitness": float(registration.fitness),
                    "icp_inlier_rmse_mm": float(registration.inlier_rmse),
                    "max_correspondence_distance_mm": float(threshold),
                },
            )
        )
    return rows


def _mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"invalid reference mesh {path}")
    return vertices, triangles


def _axis_limits(scan: np.ndarray, reference: np.ndarray, horizontal: int, vertical: int) -> tuple[float, float, float, float]:
    combined = np.vstack((scan, reference))
    h0, h1 = np.percentile(combined[:, horizontal], [0.05, 99.95])
    v0, v1 = np.percentile(combined[:, vertical], [0.05, 99.95])
    hm = max((h1 - h0) * 0.04, 1.0)
    vm = max((v1 - v0) * 0.04, 1.0)
    return h0 - hm, h1 + hm, v0 - vm, v1 + vm


def render_before_after(
    path: Path,
    *,
    case_token: str,
    baseline_points: np.ndarray,
    selected_points: np.ndarray,
    reference_points: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    baseline_support: np.ndarray,
    selected_support: np.ndarray,
    selected_candidate_id: str,
    baseline_global: Mapping[str, Any],
    selected_global: Mapping[str, Any],
) -> None:
    scan_step = max(1, len(baseline_points) // 130000)
    ref_step = max(1, len(reference_points) // 50000)
    base = baseline_points[::scan_step]
    chosen = selected_points[::scan_step]
    base_support = baseline_support[::scan_step]
    chosen_support = selected_support[::scan_step]
    reference = reference_points[::ref_step]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(3, 2, figsize=(30, 24), dpi=100, facecolor="#0b1016")
    for row_index, (horizontal, vertical, title) in enumerate(projections):
        limits = _axis_limits(np.vstack((base, chosen)), reference, horizontal, vertical)
        for column, (points, support, label, metric) in enumerate(
            (
                (base, base_support, "EXP15 baseline", baseline_global),
                (chosen, chosen_support, selected_candidate_id, selected_global),
            )
        ):
            axis = axes[row_index, column]
            axis.set_facecolor("#0b1016")
            axis.scatter(reference[:, horizontal], reference[:, vertical], color="#aeb7c2", s=0.55, alpha=0.20, linewidths=0)
            axis.scatter(points[~support, horizontal], points[~support, vertical], color="#46d9ff", s=0.75, alpha=0.28, linewidths=0)
            axis.scatter(points[support, horizontal], points[support, vertical], color="#72ff70", s=2.1, alpha=0.78, linewidths=0)
            for event in events:
                start = np.asarray(event["line_start_mm"], dtype=np.float64)
                end = np.asarray(event["line_end_mm"], dtype=np.float64)
                axis.plot(
                    [start[horizontal], end[horizontal]],
                    [start[vertical], end[vertical]],
                    color="#ff2da8",
                    linewidth=3.2,
                    solid_capstyle="round",
                )
            axis.set_xlim(limits[0], limits[1])
            axis.set_ylim(limits[2], limits[3])
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
            axis.set_title(
                f"{title} — {label}\nvisible-CAD ≤2mm={float(metric['within_2mm_fraction']):.3f}, "
                f"p90={float(metric['p90_distance_mm']):.3f} mm",
                color="white",
                fontsize=16,
                pad=10,
            )
    figure.suptitle(
        f"{case_token} — protected-neighborhood registration selection\n"
        "grey: frozen-visible CAD samples; cyan: scan; green: native carrier support; magenta: native bend axes",
        color="white",
        fontsize=21,
        y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def render_global_delta(
    path: Path,
    *,
    case_token: str,
    baseline_points: np.ndarray,
    selected_points: np.ndarray,
    reference_points: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    selected_candidate_id: str,
) -> None:
    step = max(1, len(baseline_points) // 120000)
    ref_step = max(1, len(reference_points) // 50000)
    baseline = baseline_points[::step]
    selected = selected_points[::step]
    reference = reference_points[::ref_step]
    projections = ((0, 2, "front X–Z"), (1, 2, "side Y–Z"), (0, 1, "top X–Y"))
    figure, axes = plt.subplots(1, 3, figsize=(30, 18), dpi=100, facecolor="#0b1016")
    for axis, (horizontal, vertical, title) in zip(axes, projections):
        limits = _axis_limits(np.vstack((baseline, selected)), reference, horizontal, vertical)
        axis.set_facecolor("#0b1016")
        axis.scatter(reference[:, horizontal], reference[:, vertical], color="#aeb7c2", s=0.55, alpha=0.18, linewidths=0)
        axis.scatter(baseline[:, horizontal], baseline[:, vertical], color="#3085ff", s=0.8, alpha=0.28, linewidths=0)
        axis.scatter(selected[:, horizontal], selected[:, vertical], color="#ffb23f", s=0.8, alpha=0.28, linewidths=0)
        for event in events:
            start = np.asarray(event["line_start_mm"], dtype=np.float64)
            end = np.asarray(event["line_end_mm"], dtype=np.float64)
            axis.plot([start[horizontal], end[horizontal]], [start[vertical], end[vertical]], color="#ff2da8", linewidth=3.0)
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        axis.set_title(title, color="white", fontsize=18, pad=12)
    figure.suptitle(
        f"{case_token} — selected rigid delta {selected_candidate_id}\n"
        "blue: baseline scan; amber: selected scan; grey: frozen-visible CAD; magenta: fixed native axes",
        color="white",
        fontsize=22,
        y=0.96,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def render_event_sheets(
    output_root: Path,
    *,
    case_token: str,
    baseline_points: np.ndarray,
    baseline_normals: np.ndarray,
    selected_points: np.ndarray,
    selected_normals: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    neighborhoods: Sequence[Mapping[str, Any]],
    thickness_mm: float,
    selected_candidate_id: str,
) -> list[Path]:
    paths: list[Path] = []
    for sheet_index, batch_start in enumerate(range(0, len(events), 5), 1):
        batch_events = events[batch_start : batch_start + 5]
        batch_neighborhoods = neighborhoods[batch_start : batch_start + 5]
        figure, axes = plt.subplots(len(batch_events), 4, figsize=(30, 18), dpi=100, facecolor="#0b1016", squeeze=False)
        for row_index, (event, neighborhood) in enumerate(zip(batch_events, batch_neighborhoods)):
            base_metric, base_mask = evaluate_frozen_event_support(
                baseline_points, baseline_normals, event, neighborhood, thickness_mm=thickness_mm
            )
            trial_metric, trial_mask = evaluate_frozen_event_support(
                selected_points, selected_normals, event, neighborhood, thickness_mm=thickness_mm
            )
            indices = np.asarray(neighborhood["point_indices"], dtype=np.int64)
            start = np.asarray(event["line_start_mm"], dtype=np.float64)
            end = np.asarray(event["line_end_mm"], dtype=np.float64)
            direction = end - start
            length = float(np.linalg.norm(direction))
            axis_direction = direction / length
            first, second = _perpendicular_basis(axis_direction)
            base_local = baseline_points[indices]
            trial_local = selected_points[indices]
            base_relative = base_local - start
            trial_relative = trial_local - start
            coordinates = {
                "base_side": np.column_stack((base_relative @ axis_direction, base_relative @ first)),
                "trial_side": np.column_stack((trial_relative @ axis_direction, trial_relative @ first)),
                "base_end": np.column_stack((base_relative @ first, base_relative @ second)),
                "trial_end": np.column_stack((trial_relative @ first, trial_relative @ second)),
            }
            base_support_local = base_mask[indices]
            trial_support_local = trial_mask[indices]
            lost = base_support_local & ~trial_support_local
            gained = trial_support_local & ~base_support_local
            panels = (
                ("base_side", base_support_local, lost, "baseline axis-side"),
                ("trial_side", trial_support_local, gained, "candidate axis-side"),
                ("base_end", base_support_local, lost, "baseline axis-end"),
                ("trial_end", trial_support_local, gained, "candidate axis-end"),
            )
            for column, (key, support, changed, title) in enumerate(panels):
                panel = axes[row_index, column]
                panel.set_facecolor("#0b1016")
                values = coordinates[key]
                panel.scatter(values[~support, 0], values[~support, 1], color="#46d9ff", s=0.8, alpha=0.22, linewidths=0)
                panel.scatter(values[support, 0], values[support, 1], color="#72ff70", s=2.0, alpha=0.70, linewidths=0)
                panel.scatter(
                    values[changed, 0],
                    values[changed, 1],
                    color="#ff4b4b" if "base" in key else "#ffd23f",
                    s=3.0,
                    alpha=0.90,
                    linewidths=0,
                )
                if "side" in key:
                    panel.plot([0.0, length], [0.0, 0.0], color="#ff2da8", linewidth=2.8)
                else:
                    theta = np.linspace(0.0, 2.0 * math.pi, 200)
                    for radius in event["radii_mm"]:
                        panel.plot(float(radius) * np.cos(theta), float(radius) * np.sin(theta), color="#ff2da8", linewidth=1.5, alpha=0.8)
                combined_values = np.vstack((coordinates[key.replace("base", "trial")], coordinates[key.replace("trial", "base")]))
                x0, x1 = np.percentile(combined_values[:, 0], [0.5, 99.5])
                y0, y1 = np.percentile(combined_values[:, 1], [0.5, 99.5])
                xm = max((x1 - x0) * 0.08, 0.5)
                ym = max((y1 - y0) * 0.08, 0.5)
                panel.set_xlim(x0 - xm, x1 + xm)
                panel.set_ylim(y0 - ym, y1 + ym)
                panel.set_aspect("equal", adjustable="box")
                panel.set_axis_off()
                base_bits = "".join("1" if value else "0" for value in base_metric["supported_bin_bitset"])
                trial_bits = "".join("1" if value else "0" for value in trial_metric["supported_bin_bitset"])
                panel.set_title(
                    f"{event['event_id']} — {title}\nlocality bins {base_bits} → {trial_bits}",
                    color="white",
                    fontsize=10,
                    pad=5,
                )
        figure.suptitle(
            f"{case_token} — candidate-bound native event audit ({selected_candidate_id}), sheet {sheet_index}\n"
            "green: support; red: baseline support lost; yellow: candidate support gained; magenta: fixed CAD carrier",
            color="white",
            fontsize=18,
            y=0.995,
        )
        path = output_root / f"native_event_sheet_{sheet_index:02d}.png"
        figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
        paths.append(path)
    return paths


def _mesh_hash_index(mesh_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(mesh_root.glob("*.ply")):
        digest = sha256_path(path)
        if digest in index:
            raise ValueError(f"duplicate mesh hash {digest}")
        index[digest] = path.resolve()
    return index


def execute_case(
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    source_record: Mapping[str, Any],
    registration_root: Path,
    mesh_hash_index: Mapping[str, Path],
    output_root: Path,
    reference_sample_count: int,
) -> dict[str, Any]:
    case_token = str(source_record["case_token"])
    registration_path = registration_root / "cases" / case_token / "semantic_result.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    aligned_scan_path = Path(registration["artifacts"]["aligned_scan"]["path"])
    if sha256_path(aligned_scan_path) != registration["artifacts"]["aligned_scan"]["sha256"]:
        raise ValueError(f"aligned scan hash mismatch for {case_token}")
    mesh_hash = str(registration["engine_mesh_sha256"])
    if mesh_hash != str(summary["engine_mesh_sha256"]):
        raise ValueError(f"candidate/registration mesh mismatch for {case_token}")
    if mesh_hash not in mesh_hash_index:
        raise ValueError(f"mesh hash not present for {case_token}")
    mesh_path = mesh_hash_index[mesh_hash]
    full_points = np.asarray(o3d.io.read_point_cloud(str(aligned_scan_path)).points, dtype=np.float64)
    points, baseline_normals = density_capped_points_and_normals(full_points)
    vertices, triangles = _mesh_arrays(mesh_path)
    reference_points, reference_normals = sample_surface(vertices, triangles, reference_sample_count)
    baseline_tree = cKDTree(points)
    baseline_reference_distances, _ = baseline_tree.query(reference_points, k=1, workers=1)
    visible_mask = baseline_reference_distances <= VISIBLE_REFERENCE_RADIUS_MM
    visible_reference = reference_points[visible_mask]
    visible_normals = reference_normals[visible_mask]
    if len(visible_reference) < 1000:
        raise ValueError(f"too few frozen-visible reference samples for {case_token}: {len(visible_reference)}")
    events = list(summary["extraction"]["events"])
    thickness_mm = float(summary["extraction"]["sheet_thickness_mm"])
    frozen_neighborhoods = build_frozen_event_neighborhoods(
        points,
        events,
        thickness_mm=thickness_mm,
    )
    baseline_event_metrics, baseline_support = event_support_metrics(
        points,
        baseline_normals,
        events,
        frozen_neighborhoods,
        thickness_mm=thickness_mm,
    )
    baseline_global = global_visible_reference_metrics(points, visible_reference)
    baseline_transform = np.asarray(registration["raw_scan_to_reference_transform"], dtype=np.float64)
    absolute_transform_stats = absolute_similarity_stats(baseline_transform)
    normal_tensor = visible_normals.T @ visible_normals / float(len(visible_normals))
    normal_eigenvalues = np.sort(np.linalg.eigvalsh(normal_tensor))[::-1]
    reference_is_multidirectional = bool(
        len(normal_eigenvalues) >= 2
        and normal_eigenvalues[1] >= max(0.02 * normal_eigenvalues[0], 1e-6)
    )
    case_blockers: list[str] = []
    if not bool(absolute_transform_stats["proper_similarity_transform"]):
        case_blockers.append("BASELINE_ABSOLUTE_TRANSFORM_NOT_PROPER_SIMILARITY")
    if not reference_is_multidirectional:
        case_blockers.append("FROZEN_VISIBLE_REFERENCE_UNDERCONSTRAINED_SINGLE_DIRECTION")

    raw_candidates = [("BASELINE_IDENTITY", np.eye(4), {"icp_fitness": 1.0, "icp_inlier_rmse_mm": 0.0, "max_correspondence_distance_mm": 0.0})]
    if not case_blockers:
        raw_candidates.extend(generate_icp_candidates(points, visible_reference, visible_normals))
    candidate_rows: list[dict[str, Any]] = []
    candidate_points: dict[str, np.ndarray] = {}
    candidate_support_masks: dict[str, np.ndarray] = {}
    for candidate_id, delta, icp_diagnostic in raw_candidates:
        trial_points = points if candidate_id == "BASELINE_IDENTITY" else apply_transform(points, delta)
        trial_normals = baseline_normals if candidate_id == "BASELINE_IDENTITY" else baseline_normals @ delta[:3, :3].T
        event_metrics, support_mask = event_support_metrics(
            trial_points,
            trial_normals,
            events,
            frozen_neighborhoods,
            thickness_mm=thickness_mm,
        )
        global_metrics = global_visible_reference_metrics(trial_points, visible_reference)
        delta_stats = rigid_delta_stats(delta)
        bounded = (
            delta_stats["translation_mm"] <= MAX_TRANSLATION_MM + 1e-12
            and delta_stats["rotation_deg"] <= MAX_ROTATION_DEG + 1e-12
            and delta_stats["rotation_orthogonality_error"] <= 1e-6
            and abs(delta_stats["rotation_determinant"] - 1.0) <= 1e-6
        )
        if candidate_id == "BASELINE_IDENTITY":
            protected_pass, protected_reasons = True, []
            global_pass, global_reasons = True, []
        else:
            protected_pass, protected_reasons = protected_neighborhood_decision(baseline_event_metrics, event_metrics)
            global_pass, global_reasons = global_improvement_decision(baseline_global, global_metrics)
        row = {
            "candidate_id": candidate_id,
            "delta_transform": np.asarray(delta, dtype=np.float64).tolist(),
            "delta_stats": delta_stats,
            "bounded_rigid_delta": bool(bounded),
            "icp_diagnostic": icp_diagnostic,
            "global_visible_reference_metrics": global_metrics,
            "event_support_metrics": event_metrics,
            "protected_neighborhood_pass": bool(protected_pass),
            "protected_neighborhood_rejection_reasons": protected_reasons,
            "material_global_improvement_pass": bool(global_pass),
            "global_improvement_rejection_reasons": global_reasons,
        }
        candidate_rows.append(row)
        candidate_points[candidate_id] = trial_points
        candidate_support_masks[candidate_id] = support_mask
    selected_id = "BASELINE_IDENTITY" if case_blockers else select_candidate(candidate_rows)
    selected_row = next(row for row in candidate_rows if row["candidate_id"] == selected_id)
    selected_points = candidate_points[selected_id]
    selected_support = candidate_support_masks[selected_id]
    selected_delta = np.asarray(selected_row["delta_transform"], dtype=np.float64)
    selected_raw_transform = selected_delta @ baseline_transform
    selected_full_points = full_points if selected_id == "BASELINE_IDENTITY" else apply_transform(full_points, selected_delta)

    case_root = output_root / "cases" / case_token
    render_path = case_root / "native_before_after_registration.png"
    render_before_after(
        render_path,
        case_token=case_token,
        baseline_points=points,
        selected_points=selected_points,
        reference_points=visible_reference,
        events=events,
        baseline_support=baseline_support,
        selected_support=selected_support,
        selected_candidate_id=selected_id,
        baseline_global=baseline_global,
        selected_global=selected_row["global_visible_reference_metrics"],
    )
    delta_render_path: Optional[Path] = None
    event_sheet_paths: list[Path] = []
    if selected_id != "BASELINE_IDENTITY":
        delta_render_path = case_root / "native_global_delta.png"
        render_global_delta(
            delta_render_path,
            case_token=case_token,
            baseline_points=points,
            selected_points=selected_points,
            reference_points=visible_reference,
            events=events,
            selected_candidate_id=selected_id,
        )
        selected_normals = baseline_normals @ selected_delta[:3, :3].T
        event_sheet_paths = render_event_sheets(
            case_root,
            case_token=case_token,
            baseline_points=points,
            baseline_normals=baseline_normals,
            selected_points=selected_points,
            selected_normals=selected_normals,
            events=events,
            neighborhoods=frozen_neighborhoods,
            thickness_mm=thickness_mm,
            selected_candidate_id=selected_id,
        )
    selected_scan_path = case_root / "selected_aligned_scan.ply"
    selected_cloud = _make_cloud(selected_full_points)
    case_root.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(selected_scan_path), selected_cloud, write_ascii=False, compressed=False):
        raise ValueError(f"failed to write selected scan for {case_token}")
    payload = {
        "schema": "EXP_CAD_0021_PROTECTED_REGISTRATION_SELECTOR_CASE_V1",
        "experiment_id": "EXP_CAD_0021",
        "case_token": case_token,
        "source_candidate_summary_sha256": sha256_path(summary_path),
        "source_registration_semantic_sha256": sha256_path(registration_path),
        "source_aligned_scan_sha256": registration["artifacts"]["aligned_scan"]["sha256"],
        "source_engine_mesh_sha256": mesh_hash,
        "source_sldprt_sha256": summary["source_sldprt_sha256"],
        "reference_sample_count": int(reference_sample_count),
        "density_capped_scan_point_count": int(len(points)),
        "evaluation_voxel_mm": EVALUATION_VOXEL_MM,
        "frozen_visible_reference_radius_mm": VISIBLE_REFERENCE_RADIUS_MM,
        "frozen_visible_reference_point_count": int(len(visible_reference)),
        "visible_reference_normal_tensor_eigenvalues": normal_eigenvalues.tolist(),
        "visible_reference_multidirectional": reference_is_multidirectional,
        "baseline_absolute_transform_stats": absolute_transform_stats,
        "case_blockers": case_blockers,
        "selection_authorized": not case_blockers,
        "event_count": len(events),
        "candidate_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
        "selected_candidate_id": selected_id,
        "baseline_retained": selected_id == "BASELINE_IDENTITY",
        "selected_delta_transform": selected_delta.tolist(),
        "selected_raw_scan_to_reference_transform": selected_raw_transform.tolist(),
        "selection_policy": "BOUNDED_RIGID_PARETO_GLOBAL_IMPROVEMENT_WITH_EXACT_EVENT_BIN_NONREGRESSION_V1",
        "artifacts": {
            "selected_aligned_scan": {"path": str(selected_scan_path.resolve()), "sha256": sha256_path(selected_scan_path)},
            "native_before_after_render": {"path": str(render_path.resolve()), "sha256": sha256_path(render_path)},
            "native_global_delta_render": (
                {"path": str(delta_render_path.resolve()), "sha256": sha256_path(delta_render_path)}
                if delta_render_path is not None
                else None
            ),
            "native_event_sheets": [
                {"path": str(path.resolve()), "sha256": sha256_path(path)} for path in event_sheet_paths
            ],
        },
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    semantic_path = case_root / "semantic_result.json"
    atomic_json(semantic_path, payload)
    return {
        "case_token": case_token,
        "selected_candidate_id": selected_id,
        "baseline_retained": payload["baseline_retained"],
        "selection_authorized": payload["selection_authorized"],
        "case_blockers": case_blockers,
        "semantic_sha256": sha256_path(semantic_path),
        "render_sha256": sha256_path(render_path),
        "selected_scan_sha256": sha256_path(selected_scan_path),
        "baseline_global": baseline_global,
        "selected_global": selected_row["global_visible_reference_metrics"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", action="append", required=True, type=Path)
    parser.add_argument("--registration-root", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-sample-count", type=int, default=80000)
    args = parser.parse_args(argv)
    if args.reference_sample_count < 10000:
        raise ValueError("reference sample count must be at least 10000")
    mesh_index = _mesh_hash_index(args.mesh_root)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for summary_path in args.candidate_summary:
        summary_path = summary_path.resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("labels_or_expected_counts_used") is not False or summary.get("evaluator_data_opened") is not False:
            raise ValueError("truth-denying EXP17 summary required")
        for source_record in summary["records"]:
            case_token = str(source_record["case_token"])
            if case_token in seen:
                raise ValueError(f"duplicate case token {case_token}")
            seen.add(case_token)
            print(f"[{len(seen)}] {case_token}", flush=True)
            records.append(
                execute_case(
                    summary_path=summary_path,
                    summary=summary,
                    source_record=source_record,
                    registration_root=args.registration_root.resolve(),
                    mesh_hash_index=mesh_index,
                    output_root=args.output_root.resolve(),
                    reference_sample_count=args.reference_sample_count,
                )
            )
    output = {
        "schema": "EXP_CAD_0021_PROTECTED_REGISTRATION_SELECTOR_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0021",
        "case_count": len(records),
        "changed_case_count": int(sum(not row["baseline_retained"] for row in records)),
        "blocked_case_count": int(sum(not row["selection_authorized"] for row in records)),
        "records": records,
        "selection_constants": {
            "visible_reference_radius_mm": VISIBLE_REFERENCE_RADIUS_MM,
            "global_close_radius_mm": GLOBAL_CLOSE_RADIUS_MM,
            "max_translation_mm": MAX_TRANSLATION_MM,
            "max_rotation_deg": MAX_ROTATION_DEG,
            "minimum_carrier_point_ratio": MIN_CARRIER_POINT_RATIO,
            "max_protected_median_residual_regression_mm": MAX_PROTECTED_MEDIAN_RESIDUAL_REGRESSION_MM,
            "minimum_global_fraction_gain": MIN_GLOBAL_FRACTION_GAIN,
            "minimum_global_p90_gain_mm": MIN_GLOBAL_P90_GAIN_MM,
            "minimum_global_median_gain_mm": MIN_GLOBAL_MEDIAN_GAIN_MM,
            "evaluation_voxel_mm": EVALUATION_VOXEL_MM,
            "minimum_curvature_angular_span_deg": MIN_CURVATURE_ANGULAR_SPAN_DEG,
            "minimum_normal_radial_median": MIN_NORMAL_RADIAL_MEDIAN,
            "minimum_normal_radial_p10": MIN_NORMAL_RADIAL_P10,
        },
        "candidate_set": ["BASELINE_IDENTITY", "P2PL_T1P5", "P2PL_T2P5", "P2PL_T4P0", "P2P_T2P5", "P2P_T4P0"],
        "visual_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "generalization_progress_indicator": "GPI-C_PENDING_SOURCE_ONLY_VISUAL_GATE",
        "evaluator_data_opened": False,
        "labels_or_expected_counts_used": False,
        "is_metric_eligible": False,
        "generalization_evidence": False,
    }
    atomic_json(args.output_root / "SOURCE_ONLY_SUMMARY.json", output)
    print(json.dumps({"case_count": len(records), "changed_case_count": output["changed_case_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
