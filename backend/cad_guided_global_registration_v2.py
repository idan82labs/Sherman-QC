"""Provisional finite-triangle whole-part fit, additive to frozen V1/R0.

V1 PCA starts, spatial subsets, global trim fraction and numerical acceptance
thresholds are retained. Correspondences are now to a bounded finite triangle
surface. Coarse updates use rigid Kabsch; fine updates use Euclidean-Huber IRLS
weighted Kabsch with the existing center-based motion caps and same-set loss
backtracking. This is bounded local search, not global pose or side identity.

For fixed retained correspondences, w=min(1,delta/||p-q||) gives the quadratic
majorizer of Huber(||p-q||). Weighted centroids AND covariance are necessary.
Nearest points can slide on surfaces, so fixed-q point-to-point Hessian rank is
NOT used as geometric observability. The latter is conservatively measured only
on unambiguous FACE-interior normals, before/after the actual global trim and
with actual Huber weights. Boundary evidence is not refilled or called absent.

Finite Euclidean P95 is gated at the legacy plane-P95 numeric threshold. It is
stronger than the plane residual for the SAME points/correspondences, but changed
correspondence/selection prevents claiming the whole acceptance set is nested.
CAD reverse coverage remains distance from initialization sample rows to supplied
scan rows, not continuous area, native-surface coverage or unsupplied raw coverage.
"""

from dataclasses import asdict, dataclass, fields
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import cad_guided_global_registration_v1 as _v1
from .cad_guided_triangle_surface_v2 import (
    TriangleSurfaceBudgets, build_triangle_surface, query_closest_surface,
)


@dataclass(frozen=True)
class SurfaceRegistrationBudgetsV2:
    max_query_calls: int = 4096
    max_query_rows: int = 64_000_000
    max_node_visits: int = 2_000_000_000
    max_triangle_pairs: int = 2_000_000_000
    max_normal_pairs: int = 500_000_000
    max_work_units: int = 20_000_000_000


class _SurfaceFailure(ValueError):
    pass


def _norm(values):
    return np.hypot(np.hypot(values[..., 0], values[..., 1]), values[..., 2])


def _weighted_rigid_increment(points, targets, weights):
    """Proper weighted Kabsch on CURRENT transformed points, no fitted scale.

    Return rotation, center, center displacement. The action is
    R*(p-center)+center+displacement, so the translation cap is coordinate-origin
    independent. All weights are positive; their common normalization cancels.
    """
    if (points.shape != targets.shape or points.ndim != 2 or points.shape[1] != 3
            or weights.shape != (len(points),) or len(points) < 3
            or not np.isfinite(points).all() or not np.isfinite(targets).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("INVALID_WEIGHTED_RIGID_SUPPORT")
    weight = weights / math.fsum(weights.tolist())
    center = np.sum(points * weight[:, None], axis=0)
    target_center = np.sum(targets * weight[:, None], axis=0)
    source = points - center
    target = targets - target_center
    left, _, right = np.linalg.svd((source * weight[:, None]).T @ target)
    correction = np.eye(3)
    correction[-1, -1] = 1. if np.linalg.det(right.T @ left.T) >= 0 else -1.
    rotation = right.T @ correction @ left.T
    return rotation, center, target_center - center


def _huber_loss(distances, delta):
    return float(np.mean(np.where(distances <= delta, .5 * distances * distances,
                                  delta * (distances - .5 * delta))))


def _validated_execution_budgets(budgets):
    if type(budgets) is not SurfaceRegistrationBudgetsV2:
        raise ValueError("INVALID_SURFACE_REGISTRATION_BUDGETS")
    maxima = SurfaceRegistrationBudgetsV2()
    for field in fields(budgets):
        value = getattr(budgets, field.name)
        if type(value) is not int or not 0 <= value <= getattr(maxima, field.name):
            raise ValueError("INVALID_SURFACE_REGISTRATION_BUDGETS")
    return budgets


class _Queries:
    def __init__(self, index, budgets, surface_budgets):
        _validated_execution_budgets(budgets)
        self.index, self.budgets, self.surface_budgets = index, budgets, surface_budgets
        self.statistics = dict(query_calls=0, query_rows=0, node_visits=0,
                               triangle_pairs=0, normal_pairs=0, work_units=0)

    def __call__(self, points):
        if (self.statistics["query_calls"] >= self.budgets.max_query_calls
                or self.statistics["query_rows"] + len(points) > self.budgets.max_query_rows):
            raise _SurfaceFailure("REGISTRATION_QUERY_BUDGET_EXCEEDED")
        self.statistics["query_calls"] += 1
        self.statistics["query_rows"] += len(points)
        available = asdict(self.surface_budgets)
        for key in ("node_visits", "triangle_pairs", "normal_pairs", "work_units"):
            available["max_" + key] = min(available["max_" + key],
                getattr(self.budgets, "max_" + key) - self.statistics[key])
        result = query_closest_surface(self.index, points,
            budgets=TriangleSurfaceBudgets(**available))
        for key in ("node_visits", "triangle_pairs", "normal_pairs", "work_units"):
            self.statistics[key] += result["diagnostics"].get(key, 0)
        if result["status"] != "COMPLETE":
            raise _SurfaceFailure("FINITE_SURFACE_QUERY_UNAVAILABLE:" + result["reason"])
        return result


def _unambiguous(query):
    # Different supplied face tags alone do not invalidate identical coplanar
    # geometry. Distinct nearest points or oriented-normal conflict do.
    return ~(query["normal_ambiguous"] | query["closest_point_ambiguous"])


def _chosen(query, gate, fraction):
    distance = np.where(_unambiguous(query), query["distances"], np.inf)
    return _v1._trim_indices(distance, gate, fraction)


def _refine(scan, query_surface, r, t, *, diagonal, policy, coarse):
    count = policy.coarse_iterations if coarse else policy.refine_iterations
    gate = (max(policy.max_correspondence_distance_mm, policy.coarse_distance_fraction * diagonal)
            if coarse else policy.max_correspondence_distance_mm)
    last = {}
    for iteration in range(count):
        transformed = scan @ r.T + t
        query = query_surface(transformed)
        chosen = _chosen(query, gate, policy.trim_fraction)
        last = dict(admissible_rows=int(np.count_nonzero(
            (query["distances"] <= gate) & _unambiguous(query))), retained_rows=len(chosen),
            ambiguous_rows=int(np.count_nonzero(~_unambiguous(query))),
            boundary_rows=int(np.count_nonzero(query["closest_regions"] != "FACE")))
        if len(chosen) < 36:
            return r, t, iteration, "INSUFFICIENT_REFINEMENT_SUPPORT", last
        p, q = transformed[chosen], query["closest_points"][chosen]
        distances = _norm(p - q)
        weight = (np.ones(len(chosen)) if coarse else
                  np.minimum(1., policy.huber_delta_mm / np.maximum(distances, 1e-15)))
        increment, center, displacement = _weighted_rigid_increment(p, q, weight)
        if coarse:
            nr = increment @ r
            nt = increment @ (t - center) + center + displacement
        else:
            omega = Rotation.from_matrix(increment).as_rotvec()
            factor = min(1., policy.max_rotation_step_radians / max(float(_norm(omega)), 1e-15),
                         policy.max_translation_step_mm / max(float(_norm(displacement)), 1e-15))
            previous = _huber_loss(distances, policy.huber_delta_mm)
            nr, nt, accepted = r, t, False
            for backtrack in range(8):
                fraction = factor * .5**backtrack
                partial = _v1._rodrigues(fraction * omega)
                trial_r = partial @ r
                trial_t = partial @ (t - center) + center + fraction * displacement
                trial_distance = _norm(scan[chosen] @ trial_r.T + trial_t - q)
                loss = _huber_loss(trial_distance, policy.huber_delta_mm)
                if loss <= previous + 1e-14:
                    nr, nt, accepted = trial_r, trial_t, True
                    last.update(fixed_set_huber_before=previous, fixed_set_huber_after=loss,
                        update_center=center.tolist(), update_fraction=fraction,
                        applied_rotation_radians=fraction * float(_norm(omega)),
                        applied_center_displacement_mm=fraction * float(_norm(displacement)))
                    break
            if not accepted:
                return r, t, iteration + 1, "NO_IMPROVING_FIXED_CORRESPONDENCE_STEP", last
        motion = float(np.max(_norm(scan @ (nr - r).T + nt - t)))
        r, t = nr, nt
        if motion <= 1e-8:
            return r, t, iteration + 1, "STATIONARY", last
    return r, t, count, "ITERATION_LIMIT", last


def _rank(points, normals, weights=None):
    if len(points) < 36:
        return 0., np.zeros(6).tolist()
    jacobian, _, _ = _v1._jacobian(points, normals)
    if weights is not None:
        jacobian = jacobian * np.sqrt(weights)[:, None]
    singular = np.linalg.svd(jacobian, compute_uv=False) / math.sqrt(len(points))
    return (float(singular[-1] / singular[0]) if singular[0] > 0 else 0.), singular.tolist()


def _diagnose(scan, cad, refine, query_surface, r, t, policy):
    points = scan @ r.T + t
    query = query_surface(points)
    distances = query["distances"]
    backward, _ = cKDTree(points).query(cad, k=1, workers=1)
    geometric = distances <= policy.max_correspondence_distance_mm
    support = geometric & _unambiguous(query)
    interior = support & (query["closest_regions"] == "FACE")
    forward, reverse = float(np.mean(support)), float(np.mean(backward <= policy.max_correspondence_distance_mm))
    residual = np.einsum("ij,ij->i", points - query["closest_points"], query["oriented_normals"])
    ratio, singular = _rank(points[interior], query["oriented_normals"][interior])
    fit_points = refine @ r.T + t
    fit_query = query_surface(fit_points)
    retained = _chosen(fit_query, policy.max_correspondence_distance_mm, policy.trim_fraction)
    informative = retained[fit_query["closest_regions"][retained] == "FACE"]
    fit_ratio, fit_singular = _rank(fit_points[informative], fit_query["oriented_normals"][informative])
    weight = np.minimum(1., policy.huber_delta_mm /
                        np.maximum(fit_query["distances"][informative], 1e-15))
    weighted_ratio, weighted_singular = _rank(fit_points[informative],
        fit_query["oriented_normals"][informative], weight)
    count = int(np.count_nonzero(support))
    surface_p95 = float(np.quantile(distances[support], .95)) if count else None
    surface_rms = float(np.sqrt(np.mean(distances[support]**2))) if count else None
    reasons = []
    if count < 36 or forward < policy.min_scan_coverage or reverse < policy.min_cad_coverage:
        reasons.append("INSUFFICIENT_FINITE_SURFACE_SUPPORT")
    if surface_p95 is None or surface_p95 > policy.max_inlier_plane_p95_mm:
        reasons.append("SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE")
    if min(ratio, fit_ratio, weighted_ratio) < policy.min_observability_ratio:
        reasons.append("LOCAL_POSE_UNOBSERVABLE")
    return dict(scan_coverage=forward, cad_coverage=reverse, geometric_scan_coverage=float(np.mean(geometric)),
        inlier_count=count, inlier_surface_p95_mm=surface_p95, inlier_surface_rms_mm=surface_rms,
        inlier_plane_p95_mm=float(np.quantile(np.abs(residual[support]), .95)) if count else None,
        inlier_plane_rms_mm=float(np.sqrt(np.mean(residual[support]**2))) if count else None,
        all_scan_point_p95_mm=float(np.quantile(distances, .95)), all_scan_point_max_mm=float(np.max(distances)),
        all_cad_point_p95_mm=float(np.quantile(backward, .95)), all_cad_point_max_mm=float(np.max(backward)),
        ambiguous_scan_rows=int(np.count_nonzero(~_unambiguous(query))),
        boundary_scan_rows=int(np.count_nonzero(query["closest_regions"] != "FACE")),
        diagnostic_scan_count=len(scan), diagnostic_cad_count=len(cad), optimizer_scan_count=len(refine),
        informative_scan_count=int(np.count_nonzero(interior)),
        informative_fraction_of_scan=float(np.mean(interior)),
        fit_subset_count=len(retained), informative_fit_subset_count=len(informative),
        fit_fraction_of_optimizer_scan=len(retained)/len(refine),
        informative_fit_fraction_of_retained=len(informative)/len(retained) if len(retained) else 0.,
        fit_fraction_of_scan=len(retained)/len(scan),
        observability_ratio=ratio, observability_singular_values=singular,
        fit_observability_ratio=fit_ratio, fit_observability_singular_values=fit_singular,
        weighted_fit_observability_ratio=weighted_ratio,
        weighted_fit_observability_singular_values=weighted_singular, reasons=reasons)


def register_whole_part_v2(scan_points, cad_triangles, cad_face_ids, *, cad_initialization_points,
                           policy, unit_assumption, surface_budgets=None, registration_budgets=None):
    """Geometry-only surface fit. No initial pose, case identity or Gold input.

    Initialization points must be accompanied by the caller's triangle/sample
    binding in the connected workflow; arbitrary caller points alone do not
    certify that provenance. The triangle kernel has stricter numerical support
    restrictions than native tessellation. Any query budget/numeric failure
    refuses the whole registration, retaining completed and unattempted candidates.
    """
    evidence = dict(schema="FINITE_TRIANGLE_WHOLE_PART_REGISTRATION_V2",
        registration_mode="PROVISIONAL_WHOLE_PART_BEST_FIT", datum_status="NO_VERIFIED_STABLE_DATUM",
        transform_model="SE3_RIGID_NO_FITTED_SCALE", uncertainty_status="UNCALIBRATED",
        unit_status=unit_assumption if type(unit_assumption) is str else "INVALID",
        candidate_search_status="24_PCA_STARTS_BOUNDED_NOT_GLOBAL", candidates=[], candidate_count=0,
        acquisition_side_identity="NOT_VERIFIED_BY_GLOBAL_ALIGNMENT",
        deformation_absorption_status="BEST_FIT_MAY_ABSORB_REAL_DEVIATION",
        coverage_basis="ALL_SUPPLIED_UNIQUE_SCAN_ROWS_TO_TRIANGLES_AND_CAD_SAMPLE_ROWS_TO_SCAN_ROWS",
        observability_basis="UNAMBIGUOUS_FACE_INTERIORS_WITH_ACTUAL_TRIM_AND_HUBER_WEIGHTS_NO_REFILL",
        finite_boundary_status="USED_IN_DISTANCE_FIT_EXCLUDED_FROM_CONSERVATIVE_NORMAL_RANK",
        surface_error_status="APPROXIMATE_TRIANGULATION_NOT_ANALYTIC_CAD_OR_CALIBRATION",
        residual_gate="FINITE_EUCLIDEAN_P95_USING_UNCHANGED_LEGACY_PLANE_P95_NUMERIC_THRESHOLD",
        production_use_allowed=False)
    result = dict(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=[], evidence=evidence)
    if type(policy) is not _v1.WholePartRegistrationPolicyV1:
        result["reasons"] = ["INVALID_WHOLE_PART_REGISTRATION_POLICY"]
        return result
    if type(unit_assumption) is not str or unit_assumption != "PROVISIONAL_COMMON_MM_NOT_CALIBRATED":
        result["reasons"] = ["EXPLICIT_PROVISIONAL_UNIT_ASSUMPTION_REQUIRED"]
        return result
    evidence["policy"] = asdict(policy)
    query_surface = None
    try:
        surface_budget = TriangleSurfaceBudgets() if surface_budgets is None else surface_budgets
        execution_budget = _validated_execution_budgets(
            SurfaceRegistrationBudgetsV2() if registration_budgets is None else registration_budgets)
        if type(surface_budget) is not TriangleSurfaceBudgets:
            raise ValueError("INVALID_SURFACE_QUERY_BUDGETS")
        index = build_triangle_surface(cad_triangles, cad_face_ids, budgets=surface_budget)
        evidence["surface_index"] = {key: value for key, value in index.items() if key != "index"}
        if index["status"] != "COMPLETE":
            result["reasons"] = ["FINITE_TRIANGLE_SURFACE_UNAVAILABLE", index["reason"]]
            return result
        query_surface = _Queries(index["index"], execution_budget, surface_budget)
        evidence.update(surface_query_budgets=asdict(surface_budget),
                        registration_budgets=asdict(execution_budget))
        scan = _v1._canonical_scan(scan_points)
        cad = _v1._canonical_scan(cad_initialization_points)
        coarse = _v1._spatial_sample(scan, policy.coarse_points)
        refine = _v1._spatial_sample(scan, policy.refine_points)
        cad_pca = _v1._spatial_sample(cad, policy.refine_points)
        diagonal = float(_norm(np.ptp(index["index"].triangles.reshape(-1, 3), axis=0)))
        evidence.update(scan_unique_rows=len(scan), cad_unique_rows=len(cad),
            coarse_scan_rows=len(coarse), refined_scan_rows=len(refine), cad_pca_rows=len(cad_pca))
        candidates = evidence["candidates"]
        for number, (r, t) in enumerate(_v1._starts(coarse, cad_pca)):
            row = dict(index=number, execution_status="ATTEMPTED")
            candidates.append(row)
            try:
                r, t, nc, cr, cd = _refine(coarse, query_surface, r, t, diagonal=diagonal, policy=policy, coarse=True)
                r, t, nf, fr, fd = _refine(refine, query_surface, r, t, diagonal=diagonal, policy=policy, coarse=False)
                if (not np.isfinite(r).all() or not np.isfinite(t).all()
                        or np.max(np.abs(r.T @ r - np.eye(3))) > 1e-8
                        or abs(float(np.linalg.det(r)) - 1.) > 1e-8):
                    raise ValueError("NON_RIGID_RESULT")
                row.update(_diagnose(scan, cad, refine, query_surface, r, t, policy))
                row.update(coarse_iterations=nc, refine_iterations=nf, coarse_convergence_state=cr,
                    refine_convergence_state=fr, coarse_last_step=cd, refine_last_step=fd,
                    rotation=r.tolist(), translation=t.tolist())
            except _SurfaceFailure as error:
                row["reasons"] = [str(error)]
                for skipped in range(number + 1, 24):
                    candidates.append(dict(index=skipped, execution_status="NOT_ATTEMPTED",
                        reasons=["NOT_ATTEMPTED_AFTER_SURFACE_QUERY_FAILURE"]))
                raise
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                row["reasons"] = ["CANDIDATE_NUMERICAL_FAILURE"]
        eligible = [row for row in candidates if not row["reasons"]]
        evidence["eligible_candidate_count"] = len(eligible)
        if not eligible:
            result["reasons"] = ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
            return result
        eligible.sort(key=lambda row: (-row["scan_coverage"], row["inlier_surface_rms_mm"],
                                       -row["cad_coverage"], row["index"]))
        classes = []
        for row in eligible:
            p = refine @ np.asarray(row["rotation"]).T + row["translation"]
            if not any(float(np.max(_norm(p - prior[1]))) <= policy.pose_class_distance_mm for prior in classes):
                classes.append((row, p))
        best = classes[0][0]
        competing = [row for row, _ in classes[1:]
            if row["scan_coverage"] >= best["scan_coverage"] - policy.ambiguity_coverage_margin
            and row["cad_coverage"] >= best["cad_coverage"] - policy.ambiguity_coverage_margin
            and row["inlier_surface_rms_mm"] <= best["inlier_surface_rms_mm"] + policy.ambiguity_plane_rms_margin_mm]
        evidence.update(distinct_eligible_pose_class_count=len(classes), comparable_distinct_pose_count=len(competing),
                        best_generated_candidate_index=best["index"])
        if competing:
            result["reasons"] = ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
            return result
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = best["rotation"], best["translation"]
        result.update(status="CONDITIONAL_BEST_FIT", transform_scan_to_cad=transform.tolist())
        evidence["selected_candidate_index"] = best["index"]
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        result["reasons"] = [str(error) if isinstance(error, _SurfaceFailure)
            or (str(error).isupper() and len(str(error)) < 100) else "INVALID_OR_NUMERICAL_WHOLE_PART_INPUT"]
    finally:
        evidence["candidate_count"] = len(evidence["candidates"])
        evidence["attempted_candidate_count"] = sum(
            row["execution_status"] == "ATTEMPTED" for row in evidence["candidates"])
        if query_surface is not None:
            evidence["query_resources"] = dict(query_surface.statistics)
    return result
