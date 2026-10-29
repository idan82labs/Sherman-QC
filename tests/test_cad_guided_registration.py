"""Fixed invented datum contracts; no physical inputs or application imports.

Before execution: exact asymmetric poses must recover R within 1e-8 Frobenius
and t within 1e-7 mm, including deterministic partial capture retaining all
three datum normals. Underconstrained, symmetric, inconsistent and gross-outlier
inputs must abstain. These are constructed numerical criteria, not rig accuracy.
"""

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "backend" / "cad_guided_registration.py"
_SPEC = importlib.util.spec_from_file_location("_constructed_datum_registration", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
registration = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = registration
_SPEC.loader.exec_module(registration)


def datum_patches():
    """Three unrelated finite plane rectangles; identities are test-only."""
    rows, normals, groups = [], [], []
    descriptions = (
        (0, -31., np.linspace(-23., -5., 12), np.linspace(-4., 20., 13)),
        (1, -27., np.linspace(-12., 17., 13), np.linspace(5., 18., 9)),
        (2, -19., np.linspace(-25., 4., 10), np.linspace(-9., 11., 11)),
    )
    for group, (axis, offset, first, second) in enumerate(descriptions):
        other = [index for index in range(3) if index != axis]
        for a in first:
            for b in second:
                point = np.zeros(3)
                point[axis] = offset
                point[other] = a, b
                normal = np.zeros(3)
                normal[axis] = -1
                rows.append(point)
                normals.append(normal)
                groups.append(group)
    return np.asarray(rows), np.asarray(normals), np.asarray(groups)


def independent_pose(index):
    if index == 0:
        return np.eye(3), np.zeros(3)
    # Independent quaternion-to-matrix construction, not runtime pose helpers.
    axis = np.array([(1., 2., 3.), (-3., 1., 2.), (2., -4., 1.)][index - 1])
    axis /= np.linalg.norm(axis)
    half = (0.91, 2.11, math.pi - .03)[index - 1] / 2
    w = math.cos(half)
    x, y, z = math.sin(half) * axis
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return rotation, np.array([83., -117., 41.]) * index


@pytest.mark.parametrize("pose", (0, 1, 2, 3))
@pytest.mark.parametrize("partial", (False, True))
def test_independently_posed_asymmetric_full_or_partial_scan(pose, partial):
    cad, normals, _ = datum_patches()
    rotation, translation = independent_pose(pose)
    observed = cad[np.arange(len(cad)) % 5 != 0] if partial else cad
    scan = observed @ rotation.T + translation
    result = registration.register_datum_clouds(scan, cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "ACCEPTED_BOUNDED", result
    transform = np.asarray(result["transform_scan_to_cad"])
    assert np.linalg.norm(transform[:3, :3] - rotation.T) < 1e-8
    assert np.linalg.norm(transform[:3, 3] + rotation.T @ translation) < 1e-7
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1., abs=1e-10)
    evidence = result["evidence"]
    assert evidence["candidate_count"] == 24
    assert evidence["maximum_iterations_used"] <= 40
    assert evidence["candidate_search_status"] == "SEARCH_BOUNDED_NOT_GLOBAL"
    assert evidence["uncertainty_status"] == "UNCALIBRATED"
    assert evidence["transform_model"] == "SE3_RIGID_NO_FITTED_SCALE"
    assert evidence["selected_pose"]["observability_ratio"] >= .01


def test_point_order_duplicate_rows_and_normal_sign_cannot_change_result():
    cad, normals, _ = datum_patches()
    rotation, translation = independent_pose(2)
    scan = cad @ rotation.T + translation
    policy = registration.RegistrationPolicy()
    expected = registration.register_datum_clouds(scan, cad, normals, policy=policy)
    rng = np.random.default_rng(914)
    scan_indices = rng.permutation(len(scan))
    cad_indices = rng.permutation(len(cad))
    got = registration.register_datum_clouds(
        np.repeat(scan[scan_indices], 2, axis=0), np.repeat(cad[cad_indices], 3, axis=0),
        np.repeat(-normals[cad_indices], 3, axis=0), policy=policy)
    assert got == expected


def test_missing_third_normal_family_is_axially_unobservable():
    cad, normals, groups = datum_patches()
    rotation, translation = independent_pose(1)
    scan = cad[groups != 2] @ rotation.T + translation
    result = registration.register_datum_clouds(scan, cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "UNAVAILABLE", result
    assert result["transform_scan_to_cad"] is None
    assert any("LOCAL_POSE_UNOBSERVABLE" in row["reasons"]
               for row in result["evidence"]["candidates"])


def test_two_surface_families_have_exact_null_translation_even_if_sample_points_are_finite():
    cad, normals, groups = datum_patches()
    points, n = cad[groups != 2], normals[groups != 2]
    jacobian, _, _ = registration._surface_jacobian(points, n)
    axial_translation = np.array([0., 0., 0., 0., 0., 1.])
    assert np.array_equal(jacobian @ axial_translation, np.zeros(len(points)))
    assert np.linalg.matrix_rank(jacobian) == 5
    all_jacobian, _, _ = registration._surface_jacobian(cad, normals)
    assert np.linalg.matrix_rank(all_jacobian) == 6


def cube_datums():
    points, normals = [], []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for sign in (-1., 1.):
            for first in np.linspace(-8., 8., 9):
                for second in np.linspace(-8., 8., 9):
                    point, normal = np.zeros(3), np.zeros(3)
                    point[axis], normal[axis] = 10 * sign, sign
                    point[other] = first, second
                    points.append(point)
                    normals.append(normal)
    return np.asarray(points), np.asarray(normals)


def test_symmetric_cube_has_distinct_supported_poses_and_must_abstain():
    cad, normals = cube_datums()
    result = registration.register_datum_clouds(cad + [20., -40., 60.], cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "UNAVAILABLE", result
    assert result["transform_scan_to_cad"] is None
    assert result["reasons"] == ["AMBIGUOUS_GENERATED_POSES"]
    assert result["evidence"]["distinct_eligible_pose_class_count"] >= 2


@pytest.mark.parametrize("kind", ("gross_outlier", "reflection", "scale", "wrong_normals"))
def test_incompatible_datum_inputs_do_not_become_rigid_alignment(kind):
    cad, normals, _ = datum_patches()
    scan = cad.copy()
    if kind == "gross_outlier":
        scan = np.vstack((scan, [[500., 700., -800.]]))
    elif kind == "reflection":
        scan[:, 0] *= -1
    elif kind == "scale":
        scan *= 1.10
    else:
        normals = np.tile([0., 0., 1.], (len(cad), 1))
    result = registration.register_datum_clouds(scan, cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "UNAVAILABLE", result
    assert result["transform_scan_to_cad"] is None
    assert result["reasons"]


def test_small_bounded_perturbations_remain_distinct_from_calibrated_uncertainty():
    cad, normals, _ = datum_patches()
    rotation, translation = independent_pose(1)
    rng = np.random.default_rng(915)
    scan = cad @ rotation.T + translation + rng.uniform(-.001, .001, cad.shape)
    result = registration.register_datum_clouds(scan, cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "ACCEPTED_BOUNDED", result
    transform = np.asarray(result["transform_scan_to_cad"])
    assert np.linalg.norm(transform[:3, :3] - rotation.T) < .001
    assert np.linalg.norm(transform[:3, 3] + rotation.T @ translation) < .02
    assert result["evidence"]["uncertainty_status"] == "UNCALIBRATED"


@pytest.mark.parametrize("invalid", ("nan", "inf", "too_many", "too_few", "bool", "int64_min",
                                     "all_duplicates", "zero_normal", "conflicting_normal"))
def test_input_admission_is_bounded_and_fails_closed(invalid):
    cad, normals, _ = datum_patches()
    scan = cad.copy()
    if invalid in ("nan", "inf"):
        scan[0, 0] = np.nan if invalid == "nan" else np.inf
    elif invalid == "too_many":
        scan = np.zeros((5001, 3))
    elif invalid == "too_few":
        scan = scan[:35]
    elif invalid == "bool":
        scan = np.ones_like(scan, dtype=bool)
    elif invalid == "int64_min":
        scan = np.zeros_like(scan, dtype=np.int64)
        scan[0, 0] = np.iinfo(np.int64).min
    elif invalid == "all_duplicates":
        scan[:] = scan[0]
    elif invalid == "zero_normal":
        normals[0] = 0
    else:
        cad = np.vstack((cad, cad[0]))
        normals = np.vstack((normals, [0., 1., 0.]))
    result = registration.register_datum_clouds(scan, cad, normals,
                                               policy=registration.RegistrationPolicy())
    assert result["status"] == "UNAVAILABLE"
    assert result["transform_scan_to_cad"] is None


@pytest.mark.parametrize("field,value", (
    ("max_iterations", True), ("max_iterations", 41),
    ("max_correspondence_distance_mm", float("nan")),
    ("min_scan_coverage", 0), ("min_observability_ratio", 0),
    ("normal_consistency_ratio", -1), ("ambiguity_coverage_margin", 1),
    ("pose_class_distance_mm", True), ("max_residual_mm", .01),
))
def test_policy_validation(field, value):
    with pytest.raises(ValueError):
        replace(registration.RegistrationPolicy(), **{field: value})


def test_proper_starts_are_24_rigid_matrices_without_scale():
    cad, _, _ = datum_patches()
    rotation, translation = independent_pose(3)
    starts = list(registration._proper_starts(cad @ rotation.T + translation, cad))
    assert len(starts) == 24
    for r, t in starts:
        assert np.linalg.norm(r.T @ r - np.eye(3)) < 1e-12
        assert np.linalg.det(r) == pytest.approx(1., abs=1e-12)
        assert np.all(np.isfinite(t))
