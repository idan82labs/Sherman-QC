"""Invented, pre-run contracts for provisional whole-part registration V1.

No physical bytes, app/native imports, known inference pose/correspondences, or
Gold labels. Exact asymmetric cases require rotation Frobenius error <=1e-8 and
translation error <=1e-7 mm. Bounded-noise error limits are .001 and .02 mm.
These are constructed numerical criteria, not manufacturing/rig tolerances.
Each optimizer matrix has <=2000 rows and <=20 coarse +20 fine iterations.
"""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_global_registration_v1.py"
SPEC = importlib.util.spec_from_file_location("_whole_part_v1_constructed", SOURCE)
assert SPEC is not None and SPEC.loader is not None
registration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = registration
SPEC.loader.exec_module(registration)
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


def policy(**changes):
    base = registration.WholePartRegistrationPolicyV1(
        coarse_points=1024, refine_points=1024, coarse_iterations=20, refine_iterations=20,
        max_correspondence_distance_mm=.25, max_inlier_plane_p95_mm=.05,
        huber_delta_mm=.01, min_scan_coverage=.95, min_cad_coverage=.30,
        pose_class_distance_mm=.05, ambiguity_plane_rms_margin_mm=1e-5)
    return replace(base, **changes)


def patches():
    """Unlabelled inference geometry; group tags are solely construction/scoring."""
    descriptions = (
        (0, -31., np.linspace(-23., -5., 12), np.linspace(-4., 20., 13)),
        (1, -27., np.linspace(-12., 17., 13), np.linspace(5., 18., 9)),
        (2, -19., np.linspace(-25., 4., 10), np.linspace(-9., 11., 11)),
    )
    points, normals, groups = [], [], []
    for group, (axis, offset, first, second) in enumerate(descriptions):
        other = [index for index in range(3) if index != axis]
        for a in first:
            for b in second:
                point, normal = np.zeros(3), np.zeros(3)
                point[axis], normal[axis] = offset, -1.
                point[other] = a, b
                points.append(point)
                normals.append(normal)
                groups.append(group)
    return np.asarray(points), np.asarray(normals), np.asarray(groups)


def independent_rotation(axis, angle):
    # Quaternion formula is independent of the runtime Rodrigues expression.
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    w = math.cos(angle / 2)
    x, y, z = unit * math.sin(angle / 2)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose(index):
    if index == 0:
        return np.eye(3), np.zeros(3)
    angle = (.93, 2.17, math.pi-.017)[index-1]
    axis = ((1., 2., 3.), (-3., 2., 1.), (2., -4., 1.))[index-1]
    return independent_rotation(axis, angle), np.array([83., -117., 41.]) * index


def run(scan, cad, normals, **changes):
    return registration.register_whole_part_v1(scan, cad, normals,
                                               policy=policy(**changes), unit_assumption=UNIT)


@pytest.mark.parametrize("index", (0, 1, 2, 3))
@pytest.mark.parametrize("partial", (False, True))
def test_independent_full_and_partial_proper_poses(index, partial):
    cad, normals, _ = patches()
    rotation, translation = pose(index)
    observed = cad[np.arange(len(cad)) % 7 != 0] if partial else cad
    scan = observed @ rotation.T + translation
    result = run(scan, cad, normals)
    assert result["status"] == "CONDITIONAL_BEST_FIT", result
    transform = np.asarray(result["transform_scan_to_cad"])
    assert np.linalg.norm(transform[:3, :3]-rotation.T) <= 1e-8
    assert np.linalg.norm(transform[:3, 3]+rotation.T @ translation) <= 1e-7
    evidence = result["evidence"]
    assert evidence["candidate_count"] == 24
    assert evidence["datum_status"] == "NO_VERIFIED_STABLE_DATUM"
    assert evidence["candidate_search_status"] == "24_PCA_STARTS_BOUNDED_NOT_GLOBAL"
    assert evidence["transform_model"] == "SE3_RIGID_NO_FITTED_SCALE"
    assert evidence["uncertainty_status"] == "UNCALIBRATED"
    assert evidence["acquisition_side_identity"] == "NOT_VERIFIED_BY_GLOBAL_ALIGNMENT"
    assert evidence["datum_relative_deviation_status"] == "PROVISIONAL_NOT_STABLE_DATUM_METROLOGY"
    json.dumps(result, allow_nan=False)


def test_small_bounded_noise_has_frozen_numerical_error_limits():
    cad, normals, _ = patches()
    rotation, translation = pose(2)
    rng = np.random.default_rng(202609091)
    scan = cad @ rotation.T + translation + rng.uniform(-.001, .001, cad.shape)
    result = run(scan, cad, normals)
    assert result["status"] == "CONDITIONAL_BEST_FIT", result
    transform = np.asarray(result["transform_scan_to_cad"])
    assert np.linalg.norm(transform[:3, :3]-rotation.T) <= .001
    assert np.linalg.norm(transform[:3, 3]+rotation.T @ translation) <= .02


def test_normalized_jacobian_matches_independent_finite_differences():
    rng = np.random.default_rng(202609092)
    points = rng.uniform(-17, 23, (71, 3)) + [230., -417., 119.]
    normals = rng.normal(size=(71, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    target = points + rng.uniform(-.7, .7, points.shape)
    jacobian, center, scale = registration._jacobian(points, normals)
    h = 1e-5
    for coordinate in range(6):
        step = np.zeros(6)
        step[coordinate] = h
        def residual(sign):
            omega = sign * step[:3] / scale
            angle = float(np.linalg.norm(omega))
            rotation = independent_rotation(omega, angle) if angle else np.eye(3)
            moved = (points-center) @ rotation.T + center + sign*step[3:]
            return np.einsum("ij,ij->i", moved-target, normals)
        numerical = (residual(1)-residual(-1))/(2*h)
        np.testing.assert_allclose(jacobian[:, coordinate], numerical, atol=5e-9, rtol=5e-9)


@pytest.mark.parametrize("angle", (0., 1e-10, .13, 1.7, math.pi-.01))
def test_rodrigues_is_proper_and_preserves_intrinsic_radius_and_angle(angle):
    axis = np.array([2., -1., 3.])
    axis /= np.linalg.norm(axis)
    rotation = registration._rodrigues(axis*angle)
    np.testing.assert_allclose(rotation, independent_rotation(axis, angle), atol=2e-14, rtol=0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-14, rtol=0)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=2e-14)
    theta = np.linspace(.1, 1.4, 31)
    arc = np.column_stack((6*np.cos(theta), 6*np.sin(theta), np.zeros(len(theta))))
    center = np.array([21., -19., 17.])
    moved = arc @ rotation.T + center
    np.testing.assert_allclose(np.linalg.norm(moved-center, axis=1), 6., atol=5e-14, rtol=0)
    normals = np.array([[1., 0., 0.], [math.cos(1.5), math.sin(1.5), 0.]])
    assert np.dot(*(normals @ rotation.T)) == pytest.approx(np.dot(*normals), abs=1e-14)


def test_centered_finite_pose_update_equals_direct_action():
    rng = np.random.default_rng(202609093)
    scan = rng.uniform(-4, 7, (59, 3))
    rotation, translation = pose(3)
    center = np.array([-87., 49., 116.])
    increment = registration._rodrigues(np.array([.07, -.09, .11]))
    dt = np.array([.21, -.37, .41])
    updated_r = increment @ rotation
    updated_t = increment @ (translation-center) + center + dt
    direct = ((scan @ rotation.T + translation)-center) @ increment.T + center + dt
    np.testing.assert_allclose(scan @ updated_r.T + updated_t, direct, atol=2e-13, rtol=0)


@pytest.mark.parametrize("reflection", (False, True))
def test_kabsch_is_rigid_and_cannot_exactly_fit_a_chiral_reflection(reflection):
    rng = np.random.default_rng(202609094)
    source = rng.normal(size=(79, 3)) * [1., 2., 4.]
    rotation, translation = pose(1)
    target = source @ rotation.T + translation
    if reflection:
        target[:, 0] *= -1
    fitted_r, fitted_t = registration._rigid_fit(source, target)
    np.testing.assert_allclose(fitted_r.T @ fitted_r, np.eye(3), atol=1e-13, rtol=0)
    assert np.linalg.det(fitted_r) == pytest.approx(1., abs=1e-13)
    residual = np.linalg.norm(source @ fitted_r.T+fitted_t-target, axis=1)
    if reflection:
        assert float(np.sqrt(np.mean(residual**2))) > .5
    else:
        assert float(residual.max()) < 1e-12


def test_weighted_observability_uses_huber_weights_on_actual_fit_subset():
    cad, normals, groups = patches()
    scan = cad.copy()
    scan[groups == 2, 2] += .4
    specified = policy(trim_fraction=1., huber_delta_mm=1e-8,
                       max_correspondence_distance_mm=1., max_inlier_plane_p95_mm=.5)
    tree = cKDTree(cad)
    diagnosed = registration._diagnose(scan, cad, normals, tree, np.eye(3), np.zeros(3), specified)
    assert diagnosed["fit_observability_ratio"] > specified.min_observability_ratio
    assert diagnosed["weighted_fit_observability_ratio"] < specified.min_observability_ratio
    assert "LOCAL_POSE_UNOBSERVABLE" in diagnosed["reasons"]
    _, nearest = tree.query(scan, workers=1)
    center = scan.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((scan-center)**2, axis=1)))
    jacobian = np.column_stack((np.cross(scan-center, normals[nearest])/scale, normals[nearest]))
    residual = np.einsum("ij,ij->i", scan-cad[nearest], normals[nearest])
    weights = np.minimum(1., specified.huber_delta_mm/np.maximum(np.abs(residual), 1e-15))
    expected = np.linalg.svd(jacobian*np.sqrt(weights)[:, None], compute_uv=False)/math.sqrt(len(scan))
    np.testing.assert_allclose(diagnosed["weighted_fit_observability_singular_values"], expected,
                               atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("families", (1, 2))
def test_exact_planar_sliding_is_unobservable_despite_finite_samples(families):
    cad, normals, groups = patches()
    selected = groups < families
    points, n = cad[selected], normals[selected]
    jacobian, _, _ = registration._jacobian(points, n)
    assert np.linalg.matrix_rank(jacobian) == (3 if families == 1 else 5)
    result = run(points + [71., -41., 23.], points, n)
    assert result["status"] == "UNAVAILABLE", result
    assert result["transform_scan_to_cad"] is None


def test_symmetric_cube_refuses_distinct_generated_poses():
    points, normals = [], []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in (-1., 1.):
            for a in np.linspace(-8., 8., 9):
                for b in np.linspace(-8., 8., 9):
                    point, normal = np.zeros(3), np.zeros(3)
                    point[axis], normal[axis] = 10*sign, sign
                    point[other] = a, b
                    points.append(point)
                    normals.append(normal)
    cad = np.asarray(points)
    result = run(cad+[13., -17., 19.], cad, np.asarray(normals))
    assert result["status"] == "UNAVAILABLE", result
    assert result["reasons"] == ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
    assert result["evidence"]["comparable_distinct_pose_count"] > 0


def test_reordering_duplicates_and_normal_sign_do_not_change_semantic_result():
    cad, normals, _ = patches()
    rotation, translation = pose(1)
    scan = cad @ rotation.T+translation
    expected = run(scan, cad, normals)
    rng = np.random.default_rng(202609095)
    a, b = rng.permutation(len(scan)), rng.permutation(len(cad))
    result = run(np.repeat(scan[a], 2, axis=0), np.repeat(cad[b], 3, axis=0),
                 np.repeat(-normals[b], 3, axis=0))
    assert result == expected


@pytest.mark.parametrize("kind", ("reflection", "scale", "gross_outliers"))
def test_geometry_incompatibilities_cannot_become_qualified_rigid_pose(kind):
    cad, normals, _ = patches()
    scan = cad.copy()
    if kind == "reflection":
        scan[:, 0] *= -1
    elif kind == "scale":
        scan *= 1.20
    else:
        rng = np.random.default_rng(202609096)
        scan = np.vstack((scan, rng.uniform(100., 300., (160, 3))))
    result = run(scan, cad, normals)
    assert result["status"] == "UNAVAILABLE", result
    assert result["transform_scan_to_cad"] is None


def test_full_cloud_diagnostics_are_not_replaced_by_internal_optimization_sample():
    cad, normals, _ = patches()
    rng = np.random.default_rng(202609097)
    scan = np.vstack((cad, rng.uniform(70., 110., (700, 3))))
    result = run(scan, cad, normals, coarse_points=96, refine_points=192)
    assert result["status"] == "UNAVAILABLE", result
    canonical = np.unique(scan, axis=0)
    tree = cKDTree(cad)
    checked = 0
    for row in result["evidence"]["candidates"]:
        if "rotation" not in row:
            continue
        posed = canonical @ np.asarray(row["rotation"]).T + row["translation"]
        forward, _ = tree.query(posed, workers=1)
        backward, _ = cKDTree(posed).query(cad, workers=1)
        assert row["diagnostic_scan_count"] == len(canonical)
        assert row["optimizer_scan_count"] <= 192
        assert row["all_scan_point_p95_mm"] == pytest.approx(np.quantile(forward, .95), abs=1e-10)
        assert row["all_cad_point_p95_mm"] == pytest.approx(np.quantile(backward, .95), abs=1e-10)
        assert row["scan_coverage"] == pytest.approx(np.mean(forward <= .25), abs=1e-14)
        checked += 1
    assert checked == 24


@pytest.mark.parametrize("unit", (None, "mm", True, 1, np.array([UNIT, UNIT]), np.array(UNIT)))
def test_unit_assumption_is_explicit_strict_and_json_safe(unit):
    cad, normals, _ = patches()
    result = registration.register_whole_part_v1(cad, cad, normals, policy=policy(), unit_assumption=unit)
    assert result["status"] == "UNAVAILABLE"
    assert result["transform_scan_to_cad"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ("nan", "bool", "int64_min", "too_few", "zero_normal", "conflicting_normal"))
def test_input_numerics_and_normal_identity_fail_closed(change):
    cad, normals, _ = patches()
    scan = cad.copy()
    if change == "nan":
        scan[0, 0] = np.nan
    elif change == "bool":
        scan = np.ones_like(scan, dtype=bool)
    elif change == "int64_min":
        scan = scan.astype(np.int64)
        scan[0, 0] = np.iinfo(np.int64).min
    elif change == "too_few":
        scan = scan[:35]
    elif change == "zero_normal":
        normals[0] = 0
    else:
        cad = np.vstack((cad, cad[0]))
        normals = np.vstack((normals, [0., 1., 0.]))
    result = run(scan, cad, normals)
    assert result["status"] == "UNAVAILABLE"
    assert result["transform_scan_to_cad"] is None


def test_refinement_reports_iteration_limit_separately_from_stationarity():
    cad, normals, _ = patches()
    initial_r = independent_rotation([1., 2., 3.], .08)
    one = policy(coarse_iterations=1, refine_iterations=1)
    _, _, iterations, state = registration._refine(
        cad, cad, normals, cKDTree(cad), initial_r, np.array([.1, .2, -.1]), policy=one, coarse=True)
    assert iterations == 1 and state == "ITERATION_LIMIT"
    _, _, iterations, state = registration._refine(
        cad, cad, normals, cKDTree(cad), np.eye(3), np.zeros(3), policy=one, coarse=False)
    assert iterations == 1 and state == "STATIONARY"


@pytest.mark.parametrize("field,value", (
    ("coarse_points", True), ("refine_points", 35), ("coarse_iterations", 61),
    ("trim_fraction", .49), ("min_scan_coverage", 1.01),
    ("huber_delta_mm", float("nan")), ("max_rotation_step_radians", .26),
))
def test_policy_admission(field, value):
    with pytest.raises(ValueError):
        replace(policy(), **{field: value})
