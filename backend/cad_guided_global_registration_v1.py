"""Bounded provisional whole-part surface best-fit, NOT stable-datum metrology.

Inputs are geometry only. No identities, labels, nominal bend values, initial pose
or correspondences can be supplied. The search is 24 proper PCA starts followed
by trimmed rigid ICP; it is neither global nor a proof of unique physical pose.
All lengths use the caller's explicit provisional common-mm assumption. No scale
is estimated. A good whole-part fit can absorb real deformation, so downstream
measurements remain conditional on this alignment and acquisition-side identity.

The normalized point-to-plane Jacobian is [(p-c) cross n / L, n]. Its first
three solved coordinates are L*omega, NOT a geometric scale parameter. Exact
Rodrigues increments preserve SE(3). Local observability is not uncertainty.
"""

from dataclasses import asdict, dataclass
from itertools import permutations, product
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class WholePartRegistrationPolicyV1:
    coarse_points: int = 2048
    refine_points: int = 8192
    coarse_iterations: int = 30
    refine_iterations: int = 30
    trim_fraction: float = .85
    huber_delta_mm: float = .25
    coarse_distance_fraction: float = .15
    max_correspondence_distance_mm: float = 2.
    max_inlier_plane_p95_mm: float = .50
    min_scan_coverage: float = .75
    min_cad_coverage: float = .25
    min_observability_ratio: float = .005
    pose_class_distance_mm: float = .10
    ambiguity_plane_rms_margin_mm: float = .05
    ambiguity_coverage_margin: float = .03
    max_rotation_step_radians: float = .15
    max_translation_step_mm: float = 5.

    def __post_init__(self):
        for field, low, high in (("coarse_points", 36, 8192),
                                  ("refine_points", 36, 20000),
                                  ("coarse_iterations", 1, 60),
                                  ("refine_iterations", 1, 60)):
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("INVALID_WHOLE_PART_BUDGET")
        if self.refine_points < self.coarse_points:
            raise ValueError("INVALID_WHOLE_PART_BUDGET")
        for name, value in asdict(self).items():
            if name.endswith("points") or name.endswith("iterations"):
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 < value <= 1e4):
                raise ValueError("INVALID_WHOLE_PART_POLICY")
        for name in ("trim_fraction", "coarse_distance_fraction", "min_scan_coverage",
                     "min_cad_coverage", "min_observability_ratio",
                     "ambiguity_coverage_margin"):
            if getattr(self, name) > 1:
                raise ValueError("INVALID_WHOLE_PART_RATIO")
        if not (.5 <= self.trim_fraction <= 1
                and self.huber_delta_mm <= self.max_inlier_plane_p95_mm
                <= self.max_correspondence_distance_mm
                and self.max_rotation_step_radians <= .25):
            raise ValueError("INVALID_WHOLE_PART_POLICY_ORDER")


def _positions(value, limit):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu"
            or not 36 <= len(raw) <= limit):
        raise ValueError("WHOLE_PART_ARRAY_OUTSIDE_SCOPE")
    points = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(points)) or np.max(np.abs(points)) > 1e6:
        raise ValueError("INVALID_WHOLE_PART_POSITION")
    return points


def _canonical_scan(value):
    points = np.unique(_positions(value, 200000), axis=0)
    if len(points) < 36:
        raise ValueError("INSUFFICIENT_UNIQUE_SURFACE_POINTS")
    return points


def _canonical_cad(value, normals):
    points = _positions(value, 200000)
    raw = np.asarray(normals)
    if raw.shape != points.shape or raw.dtype.kind not in "fiu":
        raise ValueError("INVALID_CAD_SURFACE_NORMALS")
    n = np.asarray(raw, dtype=np.float64).copy()
    lengths = np.linalg.norm(n, axis=1)
    if not np.all(np.isfinite(n)) or np.any(np.abs(lengths - 1.) > 1e-6):
        raise ValueError("INVALID_CAD_SURFACE_NORMALS")
    n /= lengths[:, None]
    # Unoriented tangent normals: sign flips are mathematically equivalent.
    pivot = np.argmax(np.abs(n), axis=1)
    n *= np.where(n[np.arange(len(n)), pivot] < 0, -1., 1.)[:, None]
    rows = np.column_stack((points, n))
    rows = np.unique(rows, axis=0)
    unique, indices, inverse = np.unique(rows[:, :3], axis=0, return_index=True,
                                         return_inverse=True)
    chosen = rows[indices, 3:]
    if np.any(np.linalg.norm(rows[:, 3:] - chosen[inverse], axis=1) > 1e-6):
        # Edge vertices need a face-side identity; arbitrary normal selection is
        # not an admissible deterministic tie breaker for surface fitting.
        raise ValueError("COINCIDENT_CAD_NORMAL_AMBIGUITY")
    if len(unique) < 36:
        raise ValueError("INSUFFICIENT_UNIQUE_SURFACE_POINTS")
    return unique, chosen


def _spatial_sample(points, count):
    """Return actual rows, spatially balanced by nested dyadic AABB cells.

    No fabricated centroid observations. Lexicographic geometry tie-breaking is
    invariant to input row permutation; full cloud remains available for audit.
    This is an optimization sample, not a complete surface-coverage certificate.
    """
    if len(points) <= count:
        return points
    lower = np.min(points, axis=0)
    side = float(np.max(np.ptp(points, axis=0)))
    if side <= 1e-10:
        raise ValueError("DEGENERATE_WHOLE_PART_EXTENT")
    selected = np.arange(1)
    for depth in range(1, 22):
        cells = np.floor((points - lower) / side * (2 ** depth)).astype(np.int64)
        cells = np.minimum(cells, 2 ** depth - 1)
        _, first = np.unique(cells, axis=0, return_index=True)
        if len(first) > count:
            break
        selected = np.sort(first)
    if len(selected) < 36:
        # Degenerate distributions still get bounded deterministic actual rows.
        selected = np.unique(np.r_[selected, np.linspace(0, len(points)-1, 36,
                                                       dtype=np.int64)])[:count]
    return points[selected]


def _pca(points):
    center = np.mean(points, axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    vectors = vectors[:, np.argsort(values, kind="stable")[::-1]]
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return center, vectors


def _starts(scan, cad):
    sc, sb = _pca(scan)
    cc, cb = _pca(cad)
    symmetries = []
    for p in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            matrix = np.diag(signs) @ np.eye(3)[:, p]
            if np.linalg.det(matrix) > 0:
                symmetries.append(matrix)
    symmetries.sort(key=lambda a: tuple(a.ravel()))
    for matrix in symmetries:
        r = cb @ matrix @ sb.T
        yield r, cc - r @ sc


def _rigid_fit(source, target):
    a, b = np.mean(source, axis=0), np.mean(target, axis=0)
    left, _, right = np.linalg.svd((source-a).T @ (target-b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(right.T @ left.T)
    r = right.T @ correction @ left.T
    return r, b - r @ a


def _rodrigues(omega):
    angle = float(np.linalg.norm(omega))
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = omega / angle
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k @ k)


def _jacobian(points, normals):
    center = np.mean(points, axis=0)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if not math.isfinite(scale) or scale <= 1e-10:
        raise ValueError("DEGENERATE_WHOLE_PART_EXTENT")
    return np.column_stack((np.cross(centered, normals)/scale, normals)), center, scale


def _trim_indices(distances, gate, fraction):
    rows = np.flatnonzero(distances <= gate)
    count = max(36, int(math.ceil(fraction * len(rows))))
    order = np.argsort(distances[rows], kind="stable")
    return rows[order[:count]]


def _refine(scan, cad, normals, tree, r, t, *, policy, coarse):
    iterations = policy.coarse_iterations if coarse else policy.refine_iterations
    diagonal = float(np.linalg.norm(np.ptp(cad, axis=0)))
    gate = (max(policy.max_correspondence_distance_mm,
                policy.coarse_distance_fraction * diagonal) if coarse
            else policy.max_correspondence_distance_mm)
    for iteration in range(iterations):
        transformed = scan @ r.T + t
        distance, nearest = tree.query(transformed, k=1, workers=1)
        chosen = _trim_indices(distance, gate, policy.trim_fraction)
        if len(chosen) < 36:
            return r, t, iteration, "INSUFFICIENT_REFINEMENT_SUPPORT"
        p, q = transformed[chosen], cad[nearest[chosen]]
        if coarse:
            nr, nt = _rigid_fit(scan[chosen], q)
        else:
            n = normals[nearest[chosen]]
            jac, center, scale = _jacobian(p, n)
            residual = np.einsum("ij,ij->i", p-q, n)
            root_weight = np.sqrt(np.minimum(1., policy.huber_delta_mm /
                                              np.maximum(np.abs(residual), 1e-15)))
            step = np.linalg.lstsq(jac * root_weight[:, None],
                                   -residual * root_weight, rcond=1e-10)[0]
            omega = step[:3] / scale
            factor = min(1., policy.max_rotation_step_radians /
                         max(float(np.linalg.norm(omega)), 1e-15),
                         policy.max_translation_step_mm /
                         max(float(np.linalg.norm(step[3:])), 1e-15))
            # Backtrack against the SAME finite correspondence set's robust
            # tangent residual, so the linearized step cannot silently diverge.
            delta = policy.huber_delta_mm
            def loss(res):
                a = np.abs(res)
                return float(np.mean(np.where(a <= delta, .5*a*a,
                                               delta*(a-.5*delta))))
            previous = loss(residual)
            nr, nt = r, t
            improved = False
            for backtrack in range(8):
                f = factor * (.5 ** backtrack)
                inc = _rodrigues(f * omega)
                trial_r = inc @ r
                trial_t = inc @ (t-center) + center + f * step[3:]
                trial = scan[chosen] @ trial_r.T + trial_t
                if loss(np.einsum("ij,ij->i", trial-q, n)) <= previous + 1e-14:
                    nr, nt = trial_r, trial_t
                    improved = True
                    break
            if not improved:
                return r, t, iteration+1, "NO_IMPROVING_FIXED_CORRESPONDENCE_STEP"
        motion = float(np.max(np.linalg.norm(scan @ (nr-r).T + nt-t, axis=1)))
        r, t = nr, nt
        if motion <= 1e-8:
            return r, t, iteration+1, "STATIONARY"
    return r, t, iterations, "ITERATION_LIMIT"


def _diagnose(scan, cad, normals, tree, r, t, policy, *, optimization_points=None):
    p = scan @ r.T + t
    distance, nearest = tree.query(p, k=1, workers=1)
    backward, _ = cKDTree(p).query(cad, k=1, workers=1)
    support = distance <= policy.max_correspondence_distance_mm
    forward = float(np.mean(support))
    reverse = float(np.mean(backward <= policy.max_correspondence_distance_mm))
    residual = np.einsum("ij,ij->i", p-cad[nearest], normals[nearest])
    count = int(np.count_nonzero(support))
    singular = np.zeros(6)
    fit_singular = np.zeros(6)
    weighted_singular = np.zeros(6)
    plane_p95 = plane_rms = None
    fit_scan = scan if optimization_points is None else optimization_points
    fit_points = fit_scan @ r.T + t
    fit_distance, fit_nearest = tree.query(fit_points, k=1, workers=1)
    fit_residual = np.einsum("ij,ij->i", fit_points-cad[fit_nearest], normals[fit_nearest])
    retained = _trim_indices(fit_distance, policy.max_correspondence_distance_mm,
                             policy.trim_fraction)
    if count >= 36:
        jac, _, _ = _jacobian(p[support], normals[nearest[support]])
        singular = np.linalg.svd(jac, compute_uv=False) / math.sqrt(count)
        plane_p95 = float(np.quantile(np.abs(residual[support]), .95))
        plane_rms = float(np.sqrt(np.mean(residual[support] ** 2)))
    if len(retained) >= 36:
        fit_jac, _, _ = _jacobian(fit_points[retained], normals[fit_nearest[retained]])
        weight = np.minimum(1., policy.huber_delta_mm /
                            np.maximum(np.abs(fit_residual[retained]), 1e-15))
        fit_singular = np.linalg.svd(fit_jac, compute_uv=False) / math.sqrt(len(retained))
        weighted_singular = np.linalg.svd(fit_jac*np.sqrt(weight)[:, None],
                                          compute_uv=False) / math.sqrt(len(retained))
    ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    fit_ratio = float(fit_singular[-1]/fit_singular[0]) if fit_singular[0] > 0 else 0.
    weighted_ratio = (float(weighted_singular[-1]/weighted_singular[0])
                      if weighted_singular[0] > 0 else 0.)
    reasons = []
    if (count < 36 or forward < policy.min_scan_coverage
            or reverse < policy.min_cad_coverage):
        reasons.append("INSUFFICIENT_FINITE_SURFACE_SUPPORT")
    if plane_p95 is None or plane_p95 > policy.max_inlier_plane_p95_mm:
        reasons.append("SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE")
    if min(ratio, fit_ratio, weighted_ratio) < policy.min_observability_ratio:
        reasons.append("LOCAL_POSE_UNOBSERVABLE")
    return dict(scan_coverage=forward, cad_coverage=reverse, inlier_count=count,
                inlier_plane_p95_mm=plane_p95, inlier_plane_rms_mm=plane_rms,
                all_scan_point_p95_mm=float(np.quantile(distance, .95)),
                all_scan_point_max_mm=float(np.max(distance)),
                all_cad_point_p95_mm=float(np.quantile(backward, .95)),
                all_cad_point_max_mm=float(np.max(backward)),
                all_scan_plane_p95_mm=float(np.quantile(np.abs(residual), .95)),
                observability_ratio=ratio, observability_singular_values=singular.tolist(),
                diagnostic_scan_count=len(scan), diagnostic_cad_count=len(cad),
                optimizer_scan_count=len(fit_scan), fit_subset_count=len(retained),
                fit_fraction_of_optimizer_scan=len(retained)/len(fit_scan),
                fit_fraction_of_scan=len(retained)/len(scan),
                fit_observability_ratio=fit_ratio,
                weighted_fit_observability_ratio=weighted_ratio,
                fit_observability_singular_values=fit_singular.tolist(),
                weighted_fit_observability_singular_values=weighted_singular.tolist(),
                reasons=reasons)


def register_whole_part_v1(scan_points, cad_points, cad_normals, *, policy,
                           unit_assumption):
    """Produce only a conditional pose from geometry under explicit assumptions.

    CAD samples must be valid surface-interior points with corresponding tangent
    normals; caller-owned CAD sampling evidence must accompany final integration.
    Coverage fractions describe ALL supplied unique rows, never just the internal
    optimizer subset. They are NOT area or an unsupplied raw cloud's coverage.
    Rejected candidates remain in the evidence record.
    """
    evidence = dict(registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT",
                    datum_status="NO_VERIFIED_STABLE_DATUM",
                    transform_model="SE3_RIGID_NO_FITTED_SCALE",
                    unit_status=unit_assumption if type(unit_assumption) is str else "INVALID",
                    candidate_search_status="24_PCA_STARTS_BOUNDED_NOT_GLOBAL",
                    uncertainty_status="UNCALIBRATED",
                    deformation_absorption_status="BEST_FIT_MAY_ABSORB_REAL_DEVIATION",
                    intrinsic_geometry_status="SE3_PRESERVES_RADIUS_AND_ANGLE_BUT_ROI_SELECTION_CAN_BIAS",
                    datum_relative_deviation_status="PROVISIONAL_NOT_STABLE_DATUM_METROLOGY",
                    acquisition_side_identity="NOT_VERIFIED_BY_GLOBAL_ALIGNMENT",
                    coverage_basis="ALL_SUPPLIED_UNIQUE_SCAN_AND_CAD_ROWS_NOT_AREA",
                    candidate_count=0, candidates=[])
    result = dict(status="UNAVAILABLE", transform_scan_to_cad=None,
                  reasons=[], evidence=evidence)
    if type(policy) is not WholePartRegistrationPolicyV1:
        result["reasons"] = ["INVALID_WHOLE_PART_REGISTRATION_POLICY"]
        return result
    if (type(unit_assumption) is not str
            or unit_assumption != "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"):
        result["reasons"] = ["EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED"]
        return result
    evidence["policy"] = asdict(policy)
    try:
        scan = _canonical_scan(scan_points)
        cad, normals = _canonical_cad(cad_points, cad_normals)
        coarse = _spatial_sample(scan, policy.coarse_points)
        refine = _spatial_sample(scan, policy.refine_points)
        cad_pca = _spatial_sample(cad, policy.refine_points)
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
                        coarse_scan_rows=len(coarse), refined_scan_rows=len(refine),
                        cad_pca_rows=len(cad_pca))
        tree = cKDTree(cad)
        candidates = []
        for index, (r, t) in enumerate(_starts(coarse, cad_pca)):
            row = dict(index=index)
            try:
                r, t, nc, coarse_reason = _refine(
                    coarse, cad, normals, tree, r, t, policy=policy, coarse=True)
                r, t, nf, fine_reason = _refine(
                    refine, cad, normals, tree, r, t, policy=policy, coarse=False)
                if (not np.all(np.isfinite(r)) or not np.all(np.isfinite(t))
                        or np.linalg.norm(r.T @ r-np.eye(3)) > 1e-8
                        or abs(float(np.linalg.det(r))-1.) > 1e-8):
                    raise ValueError("NON_RIGID_RESULT")
                row.update(_diagnose(scan, cad, normals, tree, r, t, policy,
                                     optimization_points=refine))
                row.update(coarse_iterations=nc, refine_iterations=nf,
                           coarse_convergence_state=coarse_reason,
                           refine_convergence_state=fine_reason,
                           rotation=r.tolist(), translation=t.tolist())
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                row.update(reasons=["CANDIDATE_NUMERICAL_FAILURE"])
            candidates.append(row)
        evidence["candidate_count"] = len(candidates)
        evidence["candidates"] = candidates
        eligible = [row for row in candidates if not row["reasons"]]
        evidence["eligible_candidate_count"] = len(eligible)
        if not eligible:
            result["reasons"] = ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
            return result
        eligible.sort(key=lambda row: (-row["scan_coverage"], row["inlier_plane_rms_mm"],
                                       -row["cad_coverage"], row["index"]))
        classes = []
        for row in eligible:
            p = refine @ np.asarray(row["rotation"]).T + row["translation"]
            if not any(float(np.max(np.linalg.norm(p-prior[1], axis=1)))
                       <= policy.pose_class_distance_mm for prior in classes):
                classes.append((row, p))
        best = classes[0][0]
        competing = [row for row, _ in classes[1:]
                     if row["scan_coverage"] >= best["scan_coverage"] - policy.ambiguity_coverage_margin
                     and row["cad_coverage"] >= best["cad_coverage"] - policy.ambiguity_coverage_margin
                     and row["inlier_plane_rms_mm"] <= best["inlier_plane_rms_mm"]
                     + policy.ambiguity_plane_rms_margin_mm]
        evidence.update(distinct_eligible_pose_class_count=len(classes),
                        comparable_distinct_pose_count=len(competing),
                        best_generated_candidate_index=best["index"])
        if competing:
            result["reasons"] = ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
            return result
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
        result.update(status="CONDITIONAL_BEST_FIT", transform_scan_to_cad=transform.tolist())
        evidence["selected_candidate_index"] = best["index"]
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        message = str(error)
        result["reasons"] = [message if message.isupper() and len(message) < 100
                             else "INVALID_OR_NUMERICAL_WHOLE_PART_INPUT"]
    return result
