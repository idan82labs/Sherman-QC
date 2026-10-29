"""Independent finite-surface registration controls; no native or physical input.

The opt-in matrix is frozen at four cases: one scalene tetrahedron under two
independently calculated nonidentity poses MUST register within .02 mm maximum
action error; a plane MUST refuse; a rectangular box MUST refuse generated-pose
ambiguity. Analytic scan plane grids are independent of the explicit triangles
and use a different grid from CAD initialization. No pose, row correspondence or
case identity is supplied to public inference. All 24 candidates and all four
case outcomes remain in the complete JSON receipt, including every refusal.

The matrix uses the original strict 1024-row/20+20-step synthetic policy and
is NOT the physical-development policy or a physical/generalization claim.
Root supervision is required: 60 s wall, 45 s CPU, 1 GiB process-tree RSS, 1 MiB
receipt. No iteration-until-pass, threshold sweep, native import or file input.

Callback/monkeypatch tests below are explicitly isolated optimizer/diagnosis
seams, not constructed public-registration evidence. They permit independently
known correspondences or injected bad updates only to test exact arithmetic,
backtracking and refusal handling. The four public matrix cases never patch
starts, poses, correspondences, fitting, diagnostics or acceptance.
"""

import ast
from dataclasses import asdict, replace
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from backend import cad_guided_global_registration_v1 as v1
from backend import cad_guided_global_registration_v2 as v2
from backend.cad_guided_triangle_surface_v2 import TriangleSurfaceBudgets


UNITS = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
MATRIX_OPT_IN = "SHERMAN_RUN_SURFACE_REGISTRATION_V2_MATRIX"
RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, receipt_bytes=1024**2)
POSE_ACTION_ERROR_MM = .02
MATRIX_CASES = (
    ("tetrahedron_oblique_a", "tetrahedron", .73, (1., -2., 3.), (17., -9., 5.), "REGISTER"),
    ("tetrahedron_oblique_b", "tetrahedron", -.49, (-3., 1., 2.), (-31., 23., 19.), "REGISTER"),
    ("plane_unobservable", "plane", .73, (1., -2., 3.), (17., -9., 5.), "REFUSE"),
    ("box_generated_pose_ambiguity", "box", -.49, (-3., 1., 2.), (-31., 23., 19.), "AMBIGUOUS"),
)


def _policy(**changes):
    policy = v1.WholePartRegistrationPolicyV1(
        coarse_points=1024, refine_points=1024, coarse_iterations=20, refine_iterations=20,
        huber_delta_mm=.01, max_correspondence_distance_mm=.5,
        max_inlier_plane_p95_mm=.05, min_scan_coverage=.95, min_cad_coverage=.30,
        pose_class_distance_mm=.05, ambiguity_plane_rms_margin_mm=.005)
    return replace(policy, **changes)


def _quaternion_pose(angle=.73, axis=(1., -2., 3.), translation=(17., -9., 5.)):
    axis = np.asarray(axis) / math.sqrt(math.fsum(value * value for value in axis))
    w = math.cos(angle / 2)
    x, y, z = axis * math.sin(angle / 2)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return rotation, np.asarray(translation, dtype=float)


def _axis_pairs():
    # Six complete centrally symmetric groups, 36 distinct noncoplanar rows.
    return np.vstack([np.vstack((np.eye(3), -np.eye(3))) * radius for radius in range(1, 7)])


def _fixed_correspondence_query(targets, *, regions=None, normals=None):
    """Isolated arithmetic seam only; never used by the public geometry matrix."""
    calls = []

    def query(points):
        calls.append(points.copy())
        assert len(points) == len(targets)
        return dict(closest_points=targets.copy(), distances=np.linalg.norm(points - targets, axis=1),
                    oriented_normals=np.tile([0., 0., 1.], (len(points), 1)) if normals is None else normals.copy(),
                    closest_regions=np.full(len(points), "FACE") if regions is None else regions.copy(),
                    normal_ambiguous=np.zeros(len(points), dtype=bool),
                    closest_point_ambiguous=np.zeros(len(points), dtype=bool))
    return query, calls


def _proper(rotation):
    assert np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-12)
    assert abs(float(np.linalg.det(rotation)) - 1.) <= 2e-12


@pytest.mark.parametrize("multiplier", (1., 1e-12, 1e12))
def test_weighted_centroids_covariance_and_se3_action_are_exact(multiplier):
    points = _axis_pairs()
    weights = np.arange(1., len(points) + 1.) * multiplier
    expected_r, expected_t = _quaternion_pose()
    targets = points @ expected_r.T + expected_t
    rotation, center, displacement = v2._weighted_rigid_increment(points, targets, weights)
    denominator = math.fsum(weights.tolist())
    independent_center = np.array([math.fsum((points[:, column] * weights).tolist()) / denominator
                                   for column in range(3)])
    independent_target_center = np.array([math.fsum((targets[:, column] * weights).tolist()) / denominator
                                          for column in range(3)])
    _proper(rotation)
    assert np.allclose(center, independent_center, rtol=0, atol=2e-14)
    assert np.allclose(displacement, independent_target_center - independent_center, rtol=0, atol=2e-14)
    assert np.allclose(rotation, expected_r, rtol=0, atol=2e-14)
    assert np.allclose((points - center) @ rotation.T + center + displacement, targets, rtol=0, atol=2e-13)


def test_reflection_and_scale_cannot_enter_rigid_increment():
    points = _axis_pairs()
    for targets in (points * [-1., 1., 1.], points * 2.):
        rotation, center, displacement = v2._weighted_rigid_increment(points, targets, np.ones(len(points)))
        actual = (points - center) @ rotation.T + center + displacement
        _proper(rotation)
        assert np.linalg.norm(actual - targets) > 1.
        assert np.allclose(np.linalg.norm(actual[:, None] - actual[None, :], axis=2),
                           np.linalg.norm(points[:, None] - points[None, :], axis=2), atol=1e-12)


@pytest.mark.parametrize("bad", (0., -1., np.nan, np.inf))
def test_nonpositive_or_nonfinite_weight_refuses(bad):
    points = _axis_pairs()
    weights = np.ones(len(points))
    weights[3] = bad
    with pytest.raises(ValueError, match="INVALID_WEIGHTED_RIGID_SUPPORT"):
        v2._weighted_rigid_increment(points, points, weights)


def test_euclidean_huber_formula_and_continuity():
    values = np.array([0., .5, 1., 2., 3.])
    expected = math.fsum(.5*x*x if x <= 1 else x - .5 for x in values) / len(values)
    assert v2._huber_loss(values, 1.) == expected
    assert abs(v2._huber_loss(np.array([1. - 1e-7]), 1.) - (.5 - 1e-7)) < 1e-13
    assert abs(v2._huber_loss(np.array([1. + 1e-7]), 1.) - (.5 + 1e-7)) < 1e-13


@pytest.mark.parametrize("coarse,expected_shift", ((True, .5), (False, 3.6 / 15.3)))
def test_coarse_is_unweighted_and_fine_uses_euclidean_not_plane_weights(coarse, expected_shift):
    points = _axis_pairs()
    offsets = np.r_[np.full(30, .2), np.full(6, 2.)]
    targets = points + np.column_stack((offsets, np.zeros((36, 2))))
    # Normals point along z: point-to-plane residual is identically zero, so
    # substituting the legacy plane residual would produce the wrong weights.
    query, calls = _fixed_correspondence_query(targets)
    policy = _policy(coarse_iterations=1, refine_iterations=1, trim_fraction=1.,
                     huber_delta_mm=.1, max_inlier_plane_p95_mm=.5,
                     max_correspondence_distance_mm=3., max_translation_step_mm=5.)
    rotation, translation, count, state, last = v2._refine(
        points, query, np.eye(3), np.zeros(3), diagonal=12., policy=policy, coarse=coarse)
    assert len(calls) == 1 and count == 1 and state == "ITERATION_LIMIT"
    assert np.allclose(rotation, np.eye(3), atol=2e-14)
    assert np.allclose(translation, [expected_shift, 0., 0.], atol=2e-14)
    if not coarse:
        assert last["fixed_set_huber_after"] <= last["fixed_set_huber_before"]


@pytest.mark.parametrize("center_displacement", ((.1, .05, 0.), (2., -1., .5)))
def test_fine_motion_caps_are_center_based_and_coordinate_origin_invariant(center_displacement):
    points = _axis_pairs()
    desired_r, _ = _quaternion_pose(.4, (0., 0., 1.), (0., 0., 0.))
    targets = points @ desired_r.T + center_displacement
    policy = _policy(refine_iterations=1, trim_fraction=1., max_correspondence_distance_mm=10.,
                     max_rotation_step_radians=.05, max_translation_step_mm=.2)
    outputs = []
    for origin in (np.zeros(3), np.array([100000., -200000., 300000.])):
        query, calls = _fixed_correspondence_query(targets + origin)
        rotation, translation, _, _, last = v2._refine(points + origin, query, np.eye(3), np.zeros(3),
            diagonal=12., policy=policy, coarse=False)
        _proper(rotation)
        assert len(calls) == 1
        assert last["applied_rotation_radians"] <= .05 + 1e-14
        assert last["applied_center_displacement_mm"] <= .2 + 1e-14
        assert last["fixed_set_huber_after"] <= last["fixed_set_huber_before"] + 1e-14
        outputs.append((points + origin) @ rotation.T + translation - origin)
    assert np.allclose(outputs[0], outputs[1], rtol=0, atol=2e-9)


@pytest.mark.parametrize("direction,expected_fraction,expected_state", (
    (2., .25, "ITERATION_LIMIT"), (-1., None, "NO_IMPROVING_FIXED_CORRESPONDENCE_STEP")))
def test_backtracking_is_bounded_and_reuses_the_same_correspondences(monkeypatch, direction, expected_fraction, expected_state):
    # Fault-injection optimizer seam, not public registration/pose evidence.
    points = _axis_pairs()
    targets = points + [.4, 0., 0.]
    query, calls = _fixed_correspondence_query(targets)
    monkeypatch.setattr(v2, "_weighted_rigid_increment", lambda p, q, w:
                        (np.eye(3), np.zeros(3), np.array([direction, 0., 0.])))
    policy = _policy(refine_iterations=1, trim_fraction=1., max_translation_step_mm=5.)
    rotation, translation, iterations, state, last = v2._refine(
        points, query, np.eye(3), np.zeros(3), diagonal=12., policy=policy, coarse=False)
    assert len(calls) == 1 and iterations == 1 and state == expected_state
    if expected_fraction is None:
        assert np.array_equal(rotation, np.eye(3)) and np.array_equal(translation, np.zeros(3))
    else:
        assert last["update_fraction"] == expected_fraction
        assert np.allclose(translation, [.5, 0., 0.], atol=1e-14)
        assert last["fixed_set_huber_after"] < last["fixed_set_huber_before"]


def _box_planes(grid):
    sizes = (4., 2.5, 1.5)
    points, normals = [], []
    for axis in range(3):
        other = [value for value in range(3) if value != axis]
        for side in (0., sizes[axis]):
            for u, v in itertools.product(grid, repeat=2):
                point = np.zeros(3)
                point[axis], point[other[0]], point[other[1]] = side, sizes[other[0]]*u, sizes[other[1]]*v
                normal = np.zeros(3)
                normal[axis] = -1. if side == 0 else 1.
                points.append(point)
                normals.append(normal)
    return np.asarray(points), np.asarray(normals)


def test_normal_rank_matches_independent_normalized_jacobian_and_weights():
    points, normals = _box_planes((.2, .5, .8))
    centered = points - np.mean(points, axis=0)
    scale = math.sqrt(math.fsum(np.dot(row, row) for row in centered) / len(points))
    jacobian = np.column_stack((np.cross(centered, normals) / scale, normals))
    weights = np.where(np.abs(normals[:, 2]) > .5, 1e-8, 1.)
    for actual_weights in (None, weights):
        ratio, singular = v2._rank(points, normals, actual_weights)
        expected = np.linalg.svd(jacobian if actual_weights is None else jacobian*np.sqrt(weights)[:, None],
                                 compute_uv=False) / math.sqrt(len(points))
        assert np.allclose(singular, expected, rtol=1e-13, atol=1e-15)
        assert abs(ratio - expected[-1] / expected[0]) < 1e-14
    assert v2._rank(points, normals)[0] > .005
    assert v2._rank(points, normals, weights)[0] < .005
    assert v2._rank(points, np.tile([0., 0., 1.], (len(points), 1)))[0] < 1e-14
    assert v2._rank(points[:35], normals[:35]) == (0., [0.] * 6)


def test_ambiguous_rows_are_excluded_without_refilling_to_minimum_support():
    query = dict(distances=np.arange(60) / 100., normal_ambiguous=np.arange(60) < 13,
                 closest_point_ambiguous=(np.arange(60) >= 13) & (np.arange(60) < 25))
    retained = v2._chosen(query, 1., .85)
    assert retained.tolist() == list(range(25, 60))
    # FACE identity differences alone are not oriented-normal/point ambiguity.
    query["normal_ambiguous"][:] = False
    query["closest_point_ambiguous"][:] = False
    query["face_identity_ambiguous"] = np.ones(60, dtype=bool)
    assert len(v2._chosen(query, 1., .85)) == 51


def test_global_trim_boundary_rows_are_not_refilled_for_observability():
    # Exact diagnosis seam: all 54 genuine face-interior constraints come after
    # 306 lower-distance boundary rows and must NOT be refilled after trimming.
    interior, normals = _box_planes((.2, .5, .8))
    points = np.vstack((np.tile(interior, (6, 1))[:306], interior))
    directions = np.vstack((np.tile(normals, (6, 1))[:306], normals))
    distances = np.r_[np.full(306, .01), np.full(54, .02)]
    regions = np.asarray(["EDGE"] * 306 + ["FACE"] * 54)
    query, calls = _fixed_correspondence_query(points - directions*distances[:, None],
                                               normals=directions, regions=regions)
    result = v2._diagnose(points, points, points, query, np.eye(3), np.zeros(3), _policy())
    assert len(calls) == 2 and result["fit_subset_count"] == 306
    assert result["informative_scan_count"] == 54 and result["observability_ratio"] > .005
    assert result["informative_fit_subset_count"] == 0
    assert result["fit_observability_ratio"] == result["weighted_fit_observability_ratio"] == 0.
    assert "LOCAL_POSE_UNOBSERVABLE" in result["reasons"]


def _real_query(triangles):
    indexed = v2.build_triangle_surface(np.asarray(triangles, dtype=float), list(range(len(triangles))))
    assert indexed["status"] == "COMPLETE", indexed
    return v2._Queries(indexed["index"], v2.SurfaceRegistrationBudgetsV2(), TriangleSurfaceBudgets())


def test_finite_boundary_distance_rejects_zero_plane_residual():
    query = _real_query([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]])
    points = np.column_stack((np.linspace(1., 8., 64), np.full(64, -.2), np.zeros(64)))
    result = v2._diagnose(points, points, points, query, np.eye(3), np.zeros(3), _policy())
    assert abs(result["inlier_surface_p95_mm"] - .2) < 1e-12
    assert result["inlier_plane_p95_mm"] == 0.
    assert result["boundary_scan_rows"] == 64 and result["informative_scan_count"] == 0
    assert "SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE" in result["reasons"]


def test_finite_forward_coverage_does_not_relabel_sparse_reverse_sample_coverage():
    query = _real_query([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]])
    points = np.asarray([[x, y, 0.] for x, y in itertools.product(np.linspace(1., 2., 8), repeat=2)])
    cad_initialization = np.column_stack((np.linspace(8., 8.2, 36), np.full(36, .5), np.zeros(36)))
    result = v2._diagnose(points, cad_initialization, points, query, np.eye(3), np.zeros(3), _policy())
    assert result["scan_coverage"] == 1. and result["cad_coverage"] == 0.
    assert "INSUFFICIENT_FINITE_SURFACE_SUPPORT" in result["reasons"]


@pytest.mark.parametrize("kind,expected_coverage", (("parallel_sheets", 0.),
                                                   ("opposite_winding", 0.), ("coplanar_tags", 1.)))
def test_real_finite_query_ambiguity_propagates_without_confusing_face_tags(kind, expected_coverage):
    triangle = np.asarray([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]])
    if kind == "parallel_sheets":
        triangles = [triangle + [0., 0., -.2], triangle + [0., 0., .2]]
    elif kind == "opposite_winding":
        triangles = [triangle, triangle[[0, 2, 1]]]
    else:
        triangles = [triangle, triangle.copy()]
    query = _real_query(triangles)
    points = np.asarray([[x, y, 0.] for x, y in itertools.product(np.linspace(1., 2., 8), repeat=2)])
    result = v2._diagnose(points, points, points, query, np.eye(3), np.zeros(3), _policy())
    assert result["geometric_scan_coverage"] == 1.
    assert result["scan_coverage"] == expected_coverage
    assert result["ambiguous_scan_rows"] == (64 if expected_coverage == 0. else 0)
    if expected_coverage == 0.:
        assert result["inlier_count"] == 0 and result["fit_subset_count"] == 0
        assert "INSUFFICIENT_FINITE_SURFACE_SUPPORT" in result["reasons"]


def _tetrahedron_planes(denominator):
    # Analytic plane equations, not triangle/barycentric-query API output.
    uv = np.asarray([(i/denominator, j/denominator) for i in range(1, denominator)
                     for j in range(1, denominator-i)])
    u, v = uv.T
    zero = np.zeros(len(uv))
    return np.vstack((np.column_stack((zero, 2.5*u, 1.5*v)),
                      np.column_stack((4.*u, zero, 1.5*v)),
                      np.column_stack((4.*u, 2.5*v, zero)),
                      np.column_stack((4.*u, 2.5*v, 1.5*(1-u-v)))))


def _geometry(kind):
    if kind == "tetrahedron":
        triangles = np.asarray([
            [[0., 0., 0.], [0., 0., 1.5], [0., 2.5, 0.]],
            [[0., 0., 0.], [4., 0., 0.], [0., 0., 1.5]],
            [[0., 0., 0.], [0., 2.5, 0.], [4., 0., 0.]],
            [[4., 0., 0.], [0., 2.5, 0.], [0., 0., 1.5]],
        ])
        return triangles, _tetrahedron_planes(10), _tetrahedron_planes(11)
    if kind == "plane":
        triangles = np.asarray([[[0., 0., 0.], [4., 0., 0.], [4., 2.5, 0.]],
                                [[0., 0., 0.], [4., 2.5, 0.], [0., 2.5, 0.]]])
        points = np.asarray([[4*u, 2.5*v, 0.] for u, v in itertools.product(np.arange(1., 8.)/8, repeat=2)])
        initial = np.asarray([[4*u, 2.5*v, 0.] for u, v in itertools.product(np.arange(1., 9.)/9, repeat=2)])
        return triangles, points, initial
    assert kind == "box"
    triangles = []
    for axis in range(3):
        other = [value for value in range(3) if value != axis]
        for side in (0., (4., 2.5, 1.5)[axis]):
            corners = []
            for u, v in ((0., 0.), (1., 0.), (1., 1.), (0., 1.)):
                point = np.zeros(3)
                point[axis] = side
                point[other[0]], point[other[1]] = (4., 2.5, 1.5)[other[0]]*u, (4., 2.5, 1.5)[other[1]]*v
                corners.append(point)
            for ids in ((0, 1, 2), (0, 2, 3)):
                triangle = np.asarray([corners[index] for index in ids])
                wanted = -1. if side == 0 else 1.
                if np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])[axis] * wanted < 0:
                    triangle = triangle[[0, 2, 1]]
                triangles.append(triangle)
    return np.asarray(triangles), _box_planes(np.arange(1., 6.)/6)[0], _box_planes(np.arange(1., 7.)/7)[0]


@pytest.mark.parametrize("field", ("max_query_calls", "max_query_rows", "max_node_visits",
                                   "max_triangle_pairs", "max_work_units"))
def test_fatal_query_budget_refuses_whole_registration_and_retains_all_candidate_slots(field):
    triangles, points, initial = _geometry("tetrahedron")
    result = v2.register_whole_part_v2(points, triangles, np.arange(len(triangles)),
        cad_initialization_points=initial, policy=_policy(), unit_assumption=UNITS,
        registration_budgets=replace(v2.SurfaceRegistrationBudgetsV2(), **{field: 0}))
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    candidates = result["evidence"]["candidates"]
    assert result["evidence"]["candidate_count"] == 24
    assert [row["index"] for row in candidates] == list(range(24))
    assert "BUDGET" in candidates[0]["reasons"][0]
    assert all(row["reasons"] == ["NOT_ATTEMPTED_AFTER_SURFACE_QUERY_FAILURE"] for row in candidates[1:])
    assert result["evidence"]["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", (-1, True, 4097))
def test_invalid_aggregate_budget_refuses_before_any_candidate(bad):
    triangles, points, initial = _geometry("tetrahedron")
    result = v2.register_whole_part_v2(points, triangles, [0, 1, 2, 3],
        cad_initialization_points=initial, policy=_policy(), unit_assumption=UNITS,
        registration_budgets=replace(v2.SurfaceRegistrationBudgetsV2(), max_query_calls=bad))
    assert result["reasons"] == ["INVALID_SURFACE_REGISTRATION_BUDGETS"]
    assert result["evidence"]["candidate_count"] == 0


def test_budget_failure_after_completed_candidate_cannot_publish_an_earlier_pose():
    triangles, points, initial = _geometry("tetrahedron")
    result = v2.register_whole_part_v2(points, triangles, [0, 1, 2, 3],
        cad_initialization_points=initial, policy=_policy(coarse_iterations=1, refine_iterations=1),
        unit_assumption=UNITS, registration_budgets=replace(v2.SurfaceRegistrationBudgetsV2(), max_query_calls=4))
    assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
    candidates = result["evidence"]["candidates"]
    assert len(candidates) == 24 and "rotation" in candidates[0] and "scan_coverage" in candidates[0]
    assert "BUDGET" in candidates[1]["reasons"][0]
    assert all(row["reasons"] == ["NOT_ATTEMPTED_AFTER_SURFACE_QUERY_FAILURE"] for row in candidates[2:])
    assert result["evidence"]["query_resources"]["query_calls"] == 4


def test_one_invalid_triangle_refuses_instead_of_creating_a_surface_hole():
    triangles, points, initial = _geometry("tetrahedron")
    triangles[2, 2] = triangles[2, 1]
    result = v2.register_whole_part_v2(points, triangles, [0, 1, 2, 3],
        cad_initialization_points=initial, policy=_policy(), unit_assumption=UNITS)
    assert result["reasons"][0] == "FINITE_TRIANGLE_SURFACE_UNAVAILABLE"
    assert result["evidence"]["candidate_count"] == 0 and result["transform_scan_to_cad"] is None


def test_cumulative_query_call_and_row_budgets_do_not_reset_per_call():
    indexed = v2.build_triangle_surface(np.asarray([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]]), [0])
    query = v2._Queries(indexed["index"], replace(v2.SurfaceRegistrationBudgetsV2(),
                                               max_query_calls=1, max_query_rows=1), TriangleSurfaceBudgets())
    assert query(np.array([[1., 1., .2]]))["status"] == "COMPLETE"
    before = dict(query.statistics)
    with pytest.raises(v2._SurfaceFailure, match="REGISTRATION_QUERY_BUDGET_EXCEEDED"):
        query(np.array([[1., 1., .2]]))
    assert query.statistics == before and before["query_calls"] == before["query_rows"] == 1


def run_constructed_surface_registration_checks():
    records = []
    for name, kind, angle, axis, translation, expectation in MATRIX_CASES:
        record = dict(case=name, passed=False, expectation=expectation, phase="analytic_fixture")
        try:
            triangles, cad_frame_scan, initial = _geometry(kind)
            rotation, shift = _quaternion_pose(angle, axis, translation)
            raw = cad_frame_scan @ rotation.T + shift
            record.update(scan_rows=len(raw), triangles=len(triangles), initialization_rows=len(initial))
            record["phase"] = "public_geometry_only_registration"
            result = v2.register_whole_part_v2(raw, triangles, np.arange(len(triangles)),
                cad_initialization_points=initial, policy=_policy(), unit_assumption=UNITS)
            record["registration"] = result
            record["phase"] = "frozen_post_inference_assertions"
            evidence = result["evidence"]
            record["post_inference_candidate_action_errors"] = []
            for candidate in evidence["candidates"]:
                error_row = dict(index=candidate["index"], maximum_mm=None, rms_mm=None)
                if "rotation" in candidate and "translation" in candidate:
                    errors = np.linalg.norm(raw @ np.asarray(candidate["rotation"]).T
                                            + candidate["translation"] - cad_frame_scan, axis=1)
                    error_row.update(maximum_mm=float(errors.max()), rms_mm=float(np.sqrt(np.mean(errors**2))))
                record["post_inference_candidate_action_errors"].append(error_row)
            assert evidence["candidate_count"] == 24
            assert [row["index"] for row in evidence["candidates"]] == list(range(24))
            assert evidence["production_use_allowed"] is False
            assert evidence["acquisition_side_identity"] == "NOT_VERIFIED_BY_GLOBAL_ALIGNMENT"
            assert evidence["uncertainty_status"] == "UNCALIBRATED"
            if expectation == "REGISTER":
                assert result["status"] == "CONDITIONAL_BEST_FIT", result["reasons"]
                transform = np.asarray(result["transform_scan_to_cad"])
                _proper(transform[:3, :3])
                errors = np.linalg.norm(raw @ transform[:3, :3].T + transform[:3, 3] - cad_frame_scan, axis=1)
                record["post_inference_pose_action_max_mm"] = float(errors.max())
                assert record["post_inference_pose_action_max_mm"] <= POSE_ACTION_ERROR_MM
            else:
                assert result["status"] == "UNAVAILABLE" and result["transform_scan_to_cad"] is None
                if expectation == "AMBIGUOUS":
                    assert result["reasons"] == ["AMBIGUOUS_GENERATED_WHOLE_PART_POSES"]
                    assert evidence["comparable_distinct_pose_count"] >= 1
                else:
                    assert result["reasons"] == ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]
                    assert all(row["reasons"] for row in evidence["candidates"])
                    assert any("LOCAL_POSE_UNOBSERVABLE" in row["reasons"] for row in evidence["candidates"])
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        records.append(record)
    return dict(protocol="FINITE_TRIANGLE_REGISTRATION_V2_FIXED_ANALYTIC_MATRIX", cases=records,
                expected_case_count=4, expected_registered_cases=2, expected_refusal_cases=2,
                policy=asdict(_policy()), maximum_pose_action_error_mm=POSE_ACTION_ERROR_MM,
                external_resource_limits=RESOURCE_LIMITS,
                evidence_class="CONSTRUCTED_FINITE_SURFACE_NOT_NATIVE_OR_PHYSICAL_GENERALIZATION")


def test_fixed_matrix_scope_budget_and_no_pose_patch():
    assert len(MATRIX_CASES) == 4 and [case[-1] for case in MATRIX_CASES] == ["REGISTER", "REGISTER", "REFUSE", "AMBIGUOUS"]
    assert RESOURCE_LIMITS == dict(wall_seconds=60, cpu_seconds=45,
                                   process_tree_rss_bytes=1024**3, receipt_bytes=1024**2)
    tree = ast.parse(Path(__file__).read_text())
    matrix = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "run_constructed_surface_registration_checks")
    assert not any(isinstance(node, ast.Name) and node.id in ("monkeypatch", "setattr") for node in ast.walk(matrix))
    calls = [node for node in ast.walk(matrix) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "register_whole_part_v2"]
    assert len(calls) == 1
    assert [key.arg for key in calls[0].keywords] == ["cad_initialization_points", "policy", "unit_assumption"]


def test_fixed_public_finite_surface_registration_matrix():
    if os.environ.get(MATRIX_OPT_IN) != "1":
        pytest.skip("Root must opt in once under the frozen external pure-geometry budget")
    receipt = run_constructed_surface_registration_checks()
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    print(encoded, flush=True)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["receipt_bytes"]
    assert len(receipt["cases"]) == 4
    assert all(row["passed"] for row in receipt["cases"]), receipt
