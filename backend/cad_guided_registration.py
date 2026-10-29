"""Bounded rigid registration of separately supplied stable surface datums.

SEARCH_BOUNDED_NOT_GLOBAL: the 24 proper PCA starts are a finite hypothesis
search, not a global optimizer or proof of unique physical pose. Datum clouds
must be separately acquired/segmented stable geometry in the same declared mm
units as the inspection scan/CAD. This module does not identify datums, calibrate
units, or use bend nominal geometry, scan labels, known correspondences or poses.
Local surface observability and competing generated poses are both checked.
Fit residuals / singular values are engineering diagnostics, not uncertainty.
"""

from dataclasses import dataclass
from itertools import permutations, product
import math

import numpy as np
from scipy.spatial import cKDTree


MAX_DATUM_POINTS = 5000
MIN_DATUM_POINTS = 36
MAX_ABS_POSITION_MM = 1e6


@dataclass(frozen=True)
class RegistrationPolicy:
    max_iterations: int = 40
    max_correspondence_distance_mm: float = 0.30
    max_p95_residual_mm: float = 0.10
    max_residual_mm: float = 0.50
    min_scan_coverage: float = 0.95
    min_cad_coverage: float = 0.30
    min_observability_ratio: float = 0.01
    pose_class_distance_mm: float = 0.05
    ambiguity_residual_margin_mm: float = 0.01
    ambiguity_coverage_margin: float = 0.02
    normal_consistency_ratio: float = 0.05

    def __post_init__(self):
        if type(self.max_iterations) is not int or not 4 <= self.max_iterations <= 40:
            raise ValueError("max_iterations must be an integer in [4, 40]")
        for name in ("max_correspondence_distance_mm", "max_p95_residual_mm",
                     "max_residual_mm", "pose_class_distance_mm",
                     "ambiguity_residual_margin_mm"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
                raise ValueError("registration lengths must be finite numbers")
            if not math.isfinite(value) or not 1e-8 <= value <= 1e4:
                raise ValueError("registration length outside bounded scope")
        if not (self.max_p95_residual_mm <= self.max_correspondence_distance_mm
                <= self.max_residual_mm):
            raise ValueError("residual limits must be ordered")
        for name in ("min_scan_coverage", "min_cad_coverage",
                     "min_observability_ratio", "normal_consistency_ratio"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
                raise ValueError("registration ratios must be finite numbers")
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("registration ratio outside (0, 1]")
        value = self.ambiguity_coverage_margin
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value < 1):
            raise ValueError("ambiguity coverage margin outside [0, 1)")


def _positions(value):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu"
            or not MIN_DATUM_POINTS <= len(raw) <= MAX_DATUM_POINTS):
        raise ValueError("DATUM_ARRAY_OUTSIDE_SCOPE")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.max(np.abs(result)) > MAX_ABS_POSITION_MM:
        raise ValueError("INVALID_DATUM_POSITION")
    return result


def _canonical_cloud(points):
    # np.unique sorts lexicographically and removes density inflation by copies.
    unique = np.unique(points, axis=0)
    if len(unique) < MIN_DATUM_POINTS:
        raise ValueError("INSUFFICIENT_UNIQUE_DATUM_POINTS")
    return unique


def _canonical_reference(points, normals):
    raw = np.asarray(normals)
    if raw.shape != points.shape or raw.dtype.kind not in "fiu":
        raise ValueError("INVALID_CAD_NORMALS")
    n = np.asarray(raw, dtype=np.float64)
    lengths = np.linalg.norm(n, axis=1)
    if (not np.all(np.isfinite(n)) or not np.all(np.isfinite(lengths))
            or np.any(np.abs(lengths - 1) > 1e-6)):
        raise ValueError("INVALID_CAD_NORMALS")
    n = n / lengths[:, None]
    pivot = np.argmax(np.abs(n), axis=1)
    n *= np.where(n[np.arange(len(n)), pivot] < 0, -1., 1.)[:, None]
    rows = np.column_stack((points, n))
    order = np.lexsort(tuple(rows[:, index] for index in range(5, -1, -1)))
    rows = rows[order]
    unique, indices, inverse = np.unique(rows[:, :3], axis=0, return_index=True,
                                         return_inverse=True)
    chosen = rows[indices, 3:]
    if np.any(np.linalg.norm(rows[:, 3:] - chosen[inverse], axis=1) > 1e-6):
        raise ValueError("COINCIDENT_CAD_NORMAL_AMBIGUITY")
    if len(unique) < MIN_DATUM_POINTS:
        raise ValueError("INSUFFICIENT_UNIQUE_DATUM_POINTS")
    return unique, chosen


def _normal_consistency(points, normals, policy):
    """Require small, two-dimensional tangent neighborhoods at every CAD row.

    The supported datum contract is locally planar surface patches, not edge
    vertices or strongly curved geometry. Sign reversal of a normal is harmless.
    """
    _, neighbor = cKDTree(points).query(points, k=min(24, len(points)), workers=1)
    for index, candidates in enumerate(neighbor):
        same = np.abs(normals[candidates] @ normals[index]) >= 1 - 1e-6
        delta = points[candidates[same]] - points[index]
        if len(delta) < 4:
            return False
        singular = np.linalg.svd(delta, compute_uv=False)
        if singular[0] <= 1e-8 or singular[1] <= .01 * singular[0]:
            return False
        normal_error = np.linalg.norm(delta @ normals[index])
        if normal_error > policy.normal_consistency_ratio * singular[1]:
            return False
    return True


def _pca(points):
    center = np.mean(points, axis=0)
    delta = points - center
    values, basis = np.linalg.eigh(delta.T @ delta / len(points))
    order = np.argsort(values, kind="stable")[::-1]
    basis = basis[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def _proper_starts(scan, cad):
    # Isolated math from EXP22; do not import its historical execution graph.
    sc, sb = _pca(scan)
    cc, cb = _pca(cad)
    matrices = []
    for permutation in permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in product((-1., 1.), repeat=3):
            matrix = np.diag(signs) @ base
            if np.linalg.det(matrix) > 0:
                matrices.append(matrix)
    matrices.sort(key=lambda value: tuple(value.ravel()))
    for matrix in matrices:
        rotation = cb @ matrix @ sb.T
        yield rotation, cc - rotation @ sc


def _kabsch(source, target):
    a, b = np.mean(source, axis=0), np.mean(target, axis=0)
    left, _, right = np.linalg.svd((source - a).T @ (target - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ correction @ left.T
    return rotation, b - rotation @ a


def _rotation_vector(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _surface_jacobian(points, normals):
    center = np.mean(points, axis=0)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", centered, centered))))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("DEGENERATE_DATUM_SCALE")
    jacobian = np.column_stack((np.cross(centered, normals) / scale, normals))
    return jacobian, center, scale


def _refine(scan, cad, normals, tree, rotation, translation, policy):
    coarse_count = policy.max_iterations // 2
    for iteration in range(policy.max_iterations):
        points = scan @ rotation.T + translation
        _, indices = tree.query(points, k=1, workers=1)
        target = cad[indices]
        if iteration < coarse_count:
            next_rotation, next_translation = _kabsch(scan, target)
        else:
            n = normals[indices]
            jacobian, center, scale = _surface_jacobian(points, n)
            residual = np.einsum("ij,ij->i", points - target, n)
            step = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)[0]
            omega = step[:3] / scale
            # Bound the local linearization step, never introduce affine scale.
            factor = min(1., .2 / max(float(np.linalg.norm(omega)), 1e-15))
            increment = _rotation_vector(factor * omega)
            next_rotation = increment @ rotation
            next_translation = increment @ (translation - center) + center + factor * step[3:]
        motion = np.max(np.linalg.norm(
            scan @ (next_rotation - rotation).T + next_translation - translation, axis=1))
        rotation, translation = next_rotation, next_translation
        # Always reach surface refinement even if point-to-point is stationary.
        if iteration >= coarse_count and motion < 1e-10:
            break
    return rotation, translation, iteration + 1


def _candidate(scan, cad, normals, tree, rotation, translation, policy, index):
    rotation, translation, iterations = _refine(
        scan, cad, normals, tree, rotation, translation, policy)
    if (not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation))
            or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-8
            or abs(float(np.linalg.det(rotation)) - 1.) > 1e-8):
        raise ValueError("NON_RIGID_RESULT")
    points = scan @ rotation.T + translation
    distances, indices = tree.query(points, k=1, workers=1)
    backward, _ = cKDTree(points).query(cad, k=1, workers=1)
    inliers = distances <= policy.max_correspondence_distance_mm
    scan_coverage = float(np.mean(inliers))
    cad_coverage = float(np.mean(backward <= policy.max_correspondence_distance_mm))
    p95, maximum = float(np.quantile(distances, .95)), float(np.max(distances))
    rms = float(np.sqrt(np.mean(distances ** 2)))
    singular = np.zeros(6)
    if np.count_nonzero(inliers) >= MIN_DATUM_POINTS:
        jacobian, _, _ = _surface_jacobian(points[inliers], normals[indices[inliers]])
        singular = np.linalg.svd(jacobian, compute_uv=False) / math.sqrt(len(jacobian))
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
    reasons = []
    if (scan_coverage < policy.min_scan_coverage or cad_coverage < policy.min_cad_coverage
            or p95 > policy.max_p95_residual_mm or maximum > policy.max_residual_mm):
        reasons.append("INSUFFICIENT_FINITE_DATUM_SUPPORT")
    if ratio < policy.min_observability_ratio:
        reasons.append("LOCAL_POSE_UNOBSERVABLE")
    return dict(index=index, rotation=rotation, translation=translation, points=points,
                iterations=iterations, scan_coverage=scan_coverage, cad_coverage=cad_coverage,
                p95_residual_mm=p95, max_residual_mm=maximum, rms_residual_mm=rms,
                observability_singular_values=singular.tolist(), observability_ratio=ratio,
                reasons=reasons)


def register_datum_clouds(scan_datum_points, cad_datum_points, cad_datum_normals, *, policy):
    """Return a scan→CAD SE(3) pose only when bounded surface checks pass.

    Missing reference coverage is allowed within policy, but the observed subset
    must constrain all six pose directions. Unobserved CAD is not a defect.
    Invalid arrays and numerical failures return UNAVAILABLE, never a partial
    transform. Inputs are not mutated and duplicate rows cannot inflate support.
    """
    evidence = dict(candidate_search_status="SEARCH_BOUNDED_NOT_GLOBAL",
                    datum_input_contract="SEPARATE_STABLE_PLANAR_SURFACE_DATUMS",
                    datum_segmentation_status="CALLER_SUPPLIED_NOT_AUTOMATICALLY_VERIFIED",
                    transform_model="SE3_RIGID_NO_FITTED_SCALE",
                    unit_status="DECLARED_COMMON_MM_NOT_CALIBRATED",
                    uncertainty_status="UNCALIBRATED", candidate_count=0)
    result = dict(status="UNAVAILABLE", transform_scan_to_cad=None,
                  reasons=[], evidence=evidence)
    if type(policy) is not RegistrationPolicy:
        result["reasons"] = ["INVALID_REGISTRATION_POLICY"]
        return result
    try:
        scan = _canonical_cloud(_positions(scan_datum_points))
        cad, normals = _canonical_reference(_positions(cad_datum_points), cad_datum_normals)
        evidence.update(scan_unique_point_count=len(scan), cad_unique_point_count=len(cad))
        if not _normal_consistency(cad, normals, policy):
            result["reasons"] = ["CAD_NORMAL_SURFACE_INCONSISTENCY"]
            return result
        tree = cKDTree(cad)
        candidates = [_candidate(scan, cad, normals, tree, rotation, translation, policy, index)
                      for index, (rotation, translation) in enumerate(_proper_starts(scan, cad))]
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        reason = str(error)
        result["reasons"] = [reason if reason.isupper() and len(reason) <= 80
                             else "INVALID_OR_NUMERIC_DATUM_INPUT"]
        return result
    evidence["candidate_count"] = len(candidates)
    evidence["maximum_iterations_used"] = max(row["iterations"] for row in candidates)
    fields = ("index", "iterations", "scan_coverage", "cad_coverage", "p95_residual_mm",
              "max_residual_mm", "rms_residual_mm", "observability_singular_values",
              "observability_ratio", "reasons")
    evidence["candidates"] = [{key: row[key] for key in fields} for row in candidates]
    eligible = [row for row in candidates if not row["reasons"]]
    evidence["eligible_candidate_count"] = len(eligible)
    if not eligible:
        result["reasons"] = ["NO_SUPPORTED_OBSERVABLE_POSE"]
        return result
    eligible.sort(key=lambda row: (row["p95_residual_mm"], row["rms_residual_mm"],
                                  -row["cad_coverage"], row["index"]))
    classes = []
    for row in eligible:
        if not any(np.max(np.linalg.norm(row["points"] - prior["points"], axis=1))
                   <= policy.pose_class_distance_mm for prior in classes):
            classes.append(row)
    best = classes[0]
    evidence["distinct_eligible_pose_class_count"] = len(classes)
    comparable = [row for row in classes[1:]
                  if row["p95_residual_mm"] <= best["p95_residual_mm"] + policy.ambiguity_residual_margin_mm
                  and row["scan_coverage"] >= best["scan_coverage"] - policy.ambiguity_coverage_margin
                  and row["cad_coverage"] >= best["cad_coverage"] - policy.ambiguity_coverage_margin]
    evidence["comparable_distinct_pose_count"] = len(comparable)
    if comparable:
        result["reasons"] = ["AMBIGUOUS_GENERATED_POSES"]
        return result
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
    result.update(status="ACCEPTED_BOUNDED", transform_scan_to_cad=transform.tolist(), reasons=[])
    evidence["selected_candidate_index"] = best["index"]
    evidence["selected_pose"] = {key: best[key] for key in fields}
    return result
