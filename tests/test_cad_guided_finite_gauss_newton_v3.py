"""Independent local finite-Gauss–Newton controls; no native/global-pose claims.

This fixed, small suite uses pure arrays, declared analytic rectangle callbacks,
and the actual finite-triangle query kernel. It does not run a whole registration,
native reader, physical part, Gold, held-out, app, file fixture or parameter sweep.
Root owns execution under 60 s wall / 45 s CPU / 1 GiB process-tree RSS limits.

Frozen hypotheses: normalized center-chart Jacobians agree with independent
central differences; each point owns one robust weight, regardless of equation
count; finite boundary residuals do not vanish with infinite-plane distance;
actual finite trial loss cannot improve by dropping/retrimming trial rows.
The finite leaving-face example must reject its full proposal and accept a
descending half step. The analytic rare-cap example must correct h=.1 mm in
one FACE-GN step, versus the exact Kabsch shrinkage m*h/(N+m), for m=9 and
N=144 or576. This is a local mechanism comparison, not engine/floor progress.

Current iterates belong to these explicit fine-helper test seams. No public
registration is monkeypatched or supplied a known pose. Expanded optimizer
rank is never asserted to be physical observability or uncertainty.
The cap-isolation controls deliberately inject a least-squares step only at
this local helper seam; their nearest-vertex callback is not a public pose test.
"""

from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from backend import cad_guided_finite_gauss_newton_v3 as v3
from backend import cad_guided_global_registration_v1 as v1
from backend import cad_guided_global_registration_v2 as v2
from backend.cad_guided_triangle_surface_v2 import TriangleSurfaceBudgets


RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45, process_tree_rss_bytes=1024**3)


def _policy(**changes):
    return replace(v1.WholePartRegistrationPolicyV1(
        coarse_points=1024, refine_points=1024, refine_iterations=1, trim_fraction=1.,
        huber_delta_mm=.1, max_inlier_plane_p95_mm=.2, max_correspondence_distance_mm=.5), **changes)


def _points():
    return np.vstack([np.vstack((np.eye(3), -np.eye(3))) * radius for radius in range(1, 7)])


def _quaternion_rotation(vector):
    angle = math.sqrt(math.fsum(float(value)**2 for value in vector))
    if angle == 0.:
        return np.eye(3)
    w = math.cos(angle / 2)
    x, y, z = np.asarray(vector) * (math.sin(angle / 2) / angle)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _fixture(kind="mixed"):
    points = _points()
    index = np.arange(len(points), dtype=float)
    normals = np.column_stack((np.cos(index), np.sin(index), np.full(len(points), .3))) / math.sqrt(1.09)
    regions = (np.asarray(["FACE"] * 12 + ["EDGE"] * 12 + ["VERTEX"] * 12)
               if kind == "mixed" else np.full(len(points), kind))
    error = np.column_stack((.2 + .3*np.sin(index), .1 + .2*np.cos(index), np.full(len(points), .4)))
    face = regions == "FACE"
    error[face] = normals[face] * (.05 + .01*index[face])[:, None]
    return points, points - error, normals, regions


def _independent_chart_residual(points, targets, normals, regions, weights, chart):
    center = np.mean(points, axis=0)
    centered = points - center
    scale = math.sqrt(math.fsum(float(np.dot(row, row)) for row in centered) / len(points))
    rotation = _quaternion_rotation(chart[:3] / scale)
    error = centered @ rotation.T + center + chart[3:] - targets
    face = regions == "FACE"
    return np.concatenate((np.sum(error[face]*normals[face], axis=1)*np.sqrt(weights[face]),
                           (error[~face]*np.sqrt(weights[~face, None])).reshape(-1)))


@pytest.mark.parametrize("kind", ("FACE", "EDGE", "mixed"))
def test_normalized_center_chart_jacobian_matches_independent_finite_differences(kind):
    points, targets, normals, regions = _fixture(kind)
    matrix, rhs, center, scale, weights = v3._mixed_system(points, targets, normals, regions, .15)
    expected_weights = np.minimum(1., .15 / np.linalg.norm(points-targets, axis=1))
    assert np.allclose(weights, expected_weights, rtol=1e-14, atol=1e-15)
    assert np.allclose(center, np.mean(points, axis=0), rtol=0, atol=1e-15)
    assert abs(scale - math.sqrt(math.fsum(float(np.dot(row, row)) for row in points-center)/len(points))) < 1e-14
    zero = np.zeros(6)
    initial = _independent_chart_residual(points, targets, normals, regions, expected_weights, zero)
    assert np.allclose(rhs, -initial, rtol=1e-13, atol=1e-14)
    columns = []
    for coordinate in range(6):
        chart = np.zeros(6)
        chart[coordinate] = 1e-6
        plus = _independent_chart_residual(points, targets, normals, regions, expected_weights, chart)
        minus = _independent_chart_residual(points, targets, normals, regions, expected_weights, -chart)
        columns.append((plus-minus)/(2e-6))
    assert np.allclose(matrix, np.column_stack(columns), rtol=0, atol=2e-9)
    face_count = int(np.count_nonzero(regions == "FACE"))
    assert matrix.shape == (face_count + 3*(len(points)-face_count), 6)


def test_boundary_components_share_one_euclidean_point_weight():
    points = _points()
    normals = np.tile([0., 0., 1.], (len(points), 1))
    matrix, rhs, _, _, weights = v3._mixed_system(
        points, points-[3., 4., 0.], normals, np.full(len(points), "EDGE"), 1.)
    assert np.allclose(weights, .2, rtol=0, atol=1e-15)
    assert np.allclose(rhs.reshape(-1, 3), -math.sqrt(.2)*np.array([3., 4., 0.]), rtol=0, atol=1e-14)
    assert np.allclose(matrix.reshape(-1, 3, 6)[:, :, 3:], math.sqrt(.2)*np.eye(3), rtol=0, atol=1e-14)
    assert abs(float(rhs @ rhs) - len(points)*5.) < 1e-11
    assert abs(float(rhs @ rhs) - len(points)*7.) > 1.  # Coordinatewise Huber would give 3+4.


def test_expanded_equation_count_neither_triples_nor_dilutes_point_weights():
    points = _points()
    normals = np.tile([0., 0., 1.], (36, 1))
    regions = np.asarray(["FACE"]*18 + ["VERTEX"]*18)
    error = np.vstack((np.tile([0., 0., 5.], (18, 1)), np.tile([3., 4., 0.], (18, 1))))
    mixed = v3._mixed_system(points, points-error, normals, regions, 1.)
    scalar = v3._mixed_system(points, points-[0., 0., 5.], normals, np.full(36, "FACE"), 1.)
    assert len(mixed[1]) == 72 and len(scalar[1]) == 36
    assert np.array_equal(mixed[4], scalar[4])
    assert abs(float(mixed[1] @ mixed[1]) - 180.) < 1e-11
    assert abs(float(mixed[1] @ mixed[1]) - float(scalar[1] @ scalar[1])) < 1e-11


@pytest.mark.parametrize("unit_scale", (1e-3, 1., 1e3))
def test_length_unit_change_preserves_rotation_chart_and_point_weights(unit_scale):
    points, targets, normals, regions = _fixture()
    base = v3._mixed_system(points, targets, normals, regions, .15)
    changed = v3._mixed_system(points*unit_scale, targets*unit_scale, normals, regions, .15*unit_scale)
    assert np.allclose(changed[0], base[0], rtol=1e-12, atol=1e-13)
    assert np.allclose(changed[1], base[1]*unit_scale, rtol=1e-12, atol=1e-12*unit_scale)
    assert np.allclose(changed[4], base[4], rtol=1e-12, atol=1e-14)
    assert abs(changed[3]/base[3] - unit_scale) < 1e-12*unit_scale
    original_step = np.linalg.lstsq(base[0], base[1], rcond=1e-10)[0]
    changed_step = np.linalg.lstsq(changed[0], changed[1], rcond=1e-10)[0]
    assert np.allclose(changed_step, original_step*unit_scale, rtol=1e-10, atol=1e-11*unit_scale)
    assert np.allclose(changed_step[:3]/changed[3], original_step[:3]/base[3], rtol=1e-10, atol=1e-11)


def test_origin_and_frame_changes_transform_the_same_mixed_chart():
    points, targets, normals, regions = _fixture()
    matrix, rhs, center, scale, weights = v3._mixed_system(points, targets, normals, regions, .15)
    rotation = _quaternion_rotation(np.array([.31, -.22, .17]))
    shift = np.array([10000., -20000., 30000.])
    actual = v3._mixed_system(points@rotation.T+shift, targets@rotation.T+shift,
                              normals@rotation.T, regions, .15)
    chart_rotation = np.zeros((6, 6))
    chart_rotation[:3, :3] = chart_rotation[3:, 3:] = rotation
    face_count = int(np.count_nonzero(regions == "FACE"))
    residual_rotation = np.eye(len(rhs))
    for start in range(face_count, len(rhs), 3):
        residual_rotation[start:start+3, start:start+3] = rotation
    assert np.allclose(actual[0], residual_rotation@matrix@chart_rotation.T, rtol=0, atol=2e-10)
    assert np.allclose(actual[1], residual_rotation@rhs, rtol=0, atol=2e-10)
    assert np.allclose(actual[2], rotation@center+shift, rtol=0, atol=2e-11)
    assert abs(actual[3]-scale) < 2e-11
    assert np.allclose(actual[4], weights, rtol=0, atol=2e-10)


@pytest.mark.parametrize("unit_scale", (1e-3, 1., 1e3))
def test_nonzero_rigid_update_action_is_unit_origin_and_frame_invariant(unit_scale):
    # Declared nearest-VERTEX callback seam, not a triangle/global-pose claim.
    scan = _points()
    initial_r = _quaternion_rotation(np.array([.17, -.13, .21]))
    initial_t = np.array([7., -11., 3.])
    aligned = scan@initial_r.T+initial_t
    center = np.mean(aligned, axis=0)
    increment = _quaternion_rotation(np.array([.015, -.02, .01]))
    targets = (aligned-center)@increment.T+center+[.02, -.01, .03]

    def vertex_query(vertices):
        def query(points):
            distance = np.linalg.norm(points[:, None, :]-vertices[None, :, :], axis=2)
            ordered = np.sort(distance, axis=1)
            assert np.all(ordered[:, 1]-ordered[:, 0] > 1e-9)
            selected = np.argmin(distance, axis=1)
            return dict(closest_points=vertices[selected], distances=distance[np.arange(len(points)), selected],
                        oriented_normals=np.tile([0., 0., 1.], (len(points), 1)),
                        closest_regions=np.full(len(points), "VERTEX"),
                        normal_ambiguous=np.zeros(len(points), dtype=bool),
                        closest_point_ambiguous=np.zeros(len(points), dtype=bool))
        return query

    base = v3.refine_finite_surface_v3(scan, vertex_query(targets), initial_r, initial_t, policy=_policy())
    frame = _quaternion_rotation(np.array([.31, -.22, .17]))
    origin = np.array([10000., -20000., 30000.])
    changed_r = frame@initial_r@frame.T
    changed_t = unit_scale*(frame@initial_t)+origin-changed_r@origin
    changed_scan = unit_scale*(scan@frame.T)+origin
    changed_targets = unit_scale*(targets@frame.T)+origin
    policy = _policy(huber_delta_mm=.1*unit_scale, max_inlier_plane_p95_mm=.2*unit_scale,
                     max_correspondence_distance_mm=.5*unit_scale, max_translation_step_mm=5.*unit_scale)
    actual = v3.refine_finite_surface_v3(changed_scan, vertex_query(changed_targets),
                                       changed_r, changed_t, policy=policy)
    for result in (base, actual):
        assert np.allclose(result[0].T@result[0], np.eye(3), rtol=0, atol=2e-12)
        assert abs(float(np.linalg.det(result[0]))-1.) < 2e-12
        assert result[4]["fixed_rows_finite_huber_after"] < result[4]["fixed_rows_finite_huber_before"]
    assert np.linalg.norm(base[0]-initial_r) > 1e-3
    action = ((changed_scan@actual[0].T+actual[1]-origin)@frame)/unit_scale
    assert np.allclose(action, scan@base[0].T+base[1], rtol=0, atol=5e-8)
    assert np.allclose(initial_r, _quaternion_rotation(np.array([.17, -.13, .21])), rtol=0, atol=0)
    assert np.array_equal(initial_t, [7., -11., 3.])


@pytest.mark.parametrize("rotation_cap, displacement_cap, expected_fraction", (
    (.04, .6, .1),   # Rotation binds; displacement also exceeds its uncapped limit.
    (.08, .15, .05), # Center displacement binds; rotation is also oversized.
    (.04, .3, .1),   # Both bounds bind at the same factor.
))
def test_mocked_oversized_step_caps_rotation_and_center_motion_not_world_translation(
        monkeypatch, rotation_cap, displacement_cap, expected_fraction):
    # Deliberately MOCKED least-squares step, local helper only. This isolates
    # cap/composition arithmetic, not the correctness of a recovered pose/step.
    scan = _points()
    incoming_r = _quaternion_rotation(np.array([.17, -.13, .21]))
    incoming_t = np.array([7., -11., 3.])
    omega, displacement = np.array([0., 0., .4]), np.array([1., 2., -2.])
    assert np.linalg.norm(omega) > rotation_cap
    assert np.linalg.norm(displacement) > displacement_cap
    increment = _quaternion_rotation(expected_fraction*omega)
    aligned = scan@incoming_r.T+incoming_t
    expected_action = (aligned-incoming_t)@increment.T+incoming_t+expected_fraction*displacement
    results = []
    for origin in (np.zeros(3), np.array([10000., -20000., 30000.])):
        shifted_scan = scan+origin
        initial_t = incoming_t+origin-incoming_r@origin
        current = shifted_scan@incoming_r.T+initial_t
        center = np.mean(current, axis=0)
        scale = math.sqrt(math.fsum(float(np.dot(row, row)) for row in current-center)/len(current))
        targets = expected_action+origin
        query_calls, solver_calls = [], []

        def query(points):
            # Actual nearest distance to the declared finite vertex set. The
            # targets are a cap-arithmetic oracle, not independent CAD/scan data.
            query_calls.append(points.copy())
            distance = np.linalg.norm(points[:, None, :]-targets[None, :, :], axis=2)
            selected = np.argmin(distance, axis=1)
            return dict(closest_points=targets[selected], distances=distance[np.arange(len(points)), selected],
                        oriented_normals=np.tile([0., 0., 1.], (len(points), 1)),
                        closest_regions=np.full(len(points), "VERTEX"),
                        normal_ambiguous=np.zeros(len(points), dtype=bool),
                        closest_point_ambiguous=np.zeros(len(points), dtype=bool))

        def prescribed_local_step(matrix, rhs, *, rcond):
            solver_calls.append((matrix.shape, rhs.shape, rcond))
            return np.r_[scale*omega, displacement], np.empty(0), 6, np.ones(6)

        with monkeypatch.context() as patch:
            patch.setattr(v3.np.linalg, "lstsq", prescribed_local_step)
            result = v3.refine_finite_surface_v3(
                shifted_scan, query, incoming_r, initial_t,
                policy=_policy(max_correspondence_distance_mm=2.,
                               max_rotation_step_radians=rotation_cap,
                               max_translation_step_mm=displacement_cap))
        rotation, translation, iterations, state, last = result
        assert solver_calls == [((108, 6), (108,), 1e-10)]
        assert len(query_calls) == 2 and iterations == 1 and state == "ITERATION_LIMIT"
        assert last["backtracking_trials"] == 1
        assert abs(last["update_fraction"]-expected_fraction) < 1e-14
        assert abs(last["applied_rotation_radians"]-expected_fraction*.4) < 1e-14
        assert abs(last["applied_center_displacement_mm"]-expected_fraction*3.) < 1e-13
        assert last["applied_rotation_radians"] <= rotation_cap+1e-14
        assert last["applied_center_displacement_mm"] <= displacement_cap+1e-13
        assert np.allclose(rotation, increment@incoming_r, rtol=0, atol=2e-12)
        assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=2e-12)
        assert abs(float(np.linalg.det(rotation))-1.) < 2e-12
        action = shifted_scan@rotation.T+translation
        assert np.allclose(action-origin, expected_action, rtol=0, atol=5e-10)
        assert np.allclose(np.mean(action, axis=0)-center, expected_fraction*displacement,
                           rtol=0, atol=5e-10)
        assert last["fixed_rows_finite_huber_after"] < 1e-18
        assert last["fixed_rows_finite_huber_before"] > 1e-4
        if np.any(origin):
            # A legitimate center-based rotation can need a huge world-t change.
            # Clipping that world translation would destroy coordinate invariance.
            assert np.linalg.norm(translation-initial_t) > 100*displacement_cap
        results.append(action-origin)
    assert np.allclose(results[0], results[1], rtol=0, atol=5e-10)


def _rectangle(axis, level, bounds):
    other = [value for value in range(3) if value != axis]
    corners = []
    for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
        point = np.zeros(3)
        point[axis] = level
        point[other[0]], point[other[1]] = bounds[0][first], bounds[1][second]
        corners.append(point)
    return np.asarray([[corners[i] for i in ids] for ids in ((0, 1, 2), (0, 2, 3))])


def _real_query(triangles, budgets=None):
    surface = v2.build_triangle_surface(np.asarray(triangles), np.arange(len(triangles)))
    assert surface["status"] == "COMPLETE", surface
    query = v2._Queries(surface["index"], budgets or v2.SurfaceRegistrationBudgetsV2(), TriangleSurfaceBudgets())
    records = []

    def record(points):
        result = query(points)
        records.append(dict(points=points.copy(), result=result))
        return result
    return record, records, query


def test_real_finite_edge_residual_is_nonzero_when_infinite_plane_residual_is_zero():
    triangle = np.asarray([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]])
    query, records, _ = _real_query(triangle)
    points = np.column_stack((np.linspace(1., 8., 36), np.full(36, -.2), np.zeros(36)))
    result = query(points)
    assert np.all(result["closest_regions"] == "EDGE")
    assert np.max(np.abs(np.sum((points-result["closest_points"])*result["oriented_normals"], axis=1))) == 0.
    matrix, rhs, _, _, _ = v3._mixed_system(points, result["closest_points"], result["oriented_normals"],
                                            result["closest_regions"], .1)
    assert matrix.shape == (108, 6) and abs(float(rhs@rhs) - .72) < 1e-12
    # Separate helper invocation, not a hidden public registration pose.
    rotation, translation, iterations, state, last = v3.refine_finite_surface_v3(
        points, query, np.eye(3), np.zeros(3), policy=_policy())
    assert iterations == 1 and state == "ITERATION_LIMIT"
    assert np.allclose(rotation, np.eye(3), rtol=0, atol=1e-12)
    assert np.allclose(translation, [0., .2, 0.], rtol=0, atol=1e-12)
    assert last["boundary_optimizer_rank_is_observability"] is False
    assert len(records) == 3 and last["trial_query_rows"] == 36
    assert last["fixed_rows_finite_huber_after"] < 1e-20


def test_actual_finite_geometry_backtracks_a_face_proposal_that_leaves_the_floor():
    # Genuine finite-kernel geometry: the minority side plane requests -0.2 y.
    # The majority floor is initially exact but its finite edge is close behind.
    triangles = np.vstack((_rectangle(2, 0., ((-3., 3.), (0., 1.))),
                            _rectangle(1, 0., ((-3., 3.), (1., 3.)))))
    floor = np.asarray([[x, y, 0.] for x, y in itertools.product(
        np.linspace(-2., 2., 12), np.linspace(.01, .06, 12))])
    side = np.asarray([[x, .2, z] for x, z in itertools.product(
        np.linspace(-2., 2., 6), np.linspace(1.2, 2.8, 6))])
    points = np.vstack((floor, side))
    query, records, _ = _real_query(triangles)
    rotation, translation, iterations, state, last = v3.refine_finite_surface_v3(
        points, query, np.eye(3), np.zeros(3), policy=_policy())
    assert iterations == 1 and state == "ITERATION_LIMIT"
    assert len(records) == 3 and [len(row["points"]) for row in records] == [180, 180, 180]
    assert np.all(records[0]["result"]["closest_regions"] == "FACE")
    assert np.all(records[1]["result"]["closest_regions"][:144] == "EDGE")
    assert last["backtracking_trials"] == 2 and last["update_fraction"] == .5
    assert np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-12)
    assert abs(float(np.linalg.det(rotation))-1.) < 1e-12
    assert np.allclose(translation, [0., -.1, 0.], rtol=0, atol=1e-11)
    losses = [v2._huber_loss(row["result"]["distances"], .1) for row in records]
    assert losses[1] > losses[0] > losses[2]
    assert abs(last["fixed_rows_finite_huber_after"] - losses[2]) < 1e-14


def test_same_retained_rows_cannot_be_dropped_retrimmed_or_ambiguity_filtered_in_trials():
    # Adversarial callback seam: every trial has one difficult, ambiguous row.
    points = np.vstack((_points(), _points()+[20., 0., 0.]))
    records = []

    def query(current):
        records.append(current.copy())
        if len(records) == 1:
            distance = np.r_[np.full(36, .05), np.full(36, .1)]
        else:
            assert len(current) == 36
            distance = np.r_[np.zeros(35), 10.]
        normals = np.tile([0., 0., 1.], (len(current), 1))
        ambiguous = np.zeros(len(current), dtype=bool)
        if len(records) > 1:
            ambiguous[-1] = True
        return dict(closest_points=current+normals*distance[:, None], distances=distance,
                    oriented_normals=normals, closest_regions=np.full(len(current), "FACE"),
                    normal_ambiguous=ambiguous, closest_point_ambiguous=np.zeros(len(current), dtype=bool))

    rotation, translation, count, state, _ = v3.refine_finite_surface_v3(
        points, query, np.eye(3), np.zeros(3), policy=_policy(trim_fraction=.5))
    assert count == 1 and state == "NO_IMPROVING_FINITE_SURFACE_STEP"
    assert len(records) == 9 and all(len(row) == 36 for row in records[1:])
    for trial in records[1:]:
        assert np.allclose(trial[:, :2], points[:36, :2], rtol=0, atol=1e-12)
        assert np.allclose((trial-points[:36])[:, 2], (trial-points[:36])[0, 2], atol=1e-12)
    assert np.array_equal(rotation, np.eye(3)) and np.array_equal(translation, np.zeros(3))


def _analytic_sliding_scene(flange_grid, h):
    rectangles = [(0, level, ((-1., 1.), (-2., 2.))) for level in (-2., 2.)]
    rectangles += [(1, level, ((-1., 1.), (-2., 2.))) for level in (-2., 2.)]
    rectangles += [(2, 0., ((-1., 1.), (-1., 1.)))]
    points = []
    for axis, level, _ in rectangles[:4]:
        for lateral, axial in itertools.product(np.linspace(-.8, .8, flange_grid),
                                                 h+np.linspace(-1., 1., flange_grid)):
            point = [lateral, level, axial] if axis == 1 else [level, lateral, axial]
            points.append(point)
    points.extend([x, y, h] for x, y in itertools.product((-.5, 0., .5), repeat=2))
    calls = []

    def query(current):
        # Independent closest points on five FINITE analytic rectangles. No
        # scan-row labels or assumed pose are used to choose correspondence.
        calls.append(current.copy())
        candidates, normals, regions = [], [], []
        for axis, level, bounds in rectangles:
            other = [value for value in range(3) if value != axis]
            projected = current.copy()
            projected[:, axis] = level
            boundary_count = np.zeros(len(current), dtype=int)
            for coordinate, (low, high) in zip(other, bounds):
                projected[:, coordinate] = np.clip(current[:, coordinate], low, high)
                boundary_count += (projected[:, coordinate] == low) | (projected[:, coordinate] == high)
            candidates.append(projected)
            normal = np.zeros(3)
            normal[axis] = 1.
            normals.append(normal)
            regions.append(np.where(boundary_count == 0, "FACE", np.where(boundary_count == 1, "EDGE", "VERTEX")))
        candidates = np.asarray(candidates)
        distances = np.linalg.norm(candidates-current[None, :, :], axis=2)
        ordered = np.sort(distances, axis=0)
        assert np.all(ordered[1]-ordered[0] > 1e-6), "FIXED_ANALYTIC_SEAM_REQUIRES_UNIQUE_CORRESPONDENCES"
        selected, row = np.argmin(distances, axis=0), np.arange(len(current))
        return dict(closest_points=candidates[selected, row], distances=distances[selected, row],
                    oriented_normals=np.asarray(normals)[selected], closest_regions=np.asarray(regions)[selected, row],
                    normal_ambiguous=np.zeros(len(current), dtype=bool),
                    closest_point_ambiguous=np.zeros(len(current), dtype=bool))
    return np.asarray(points), query, calls


@pytest.mark.parametrize("flange_grid", (6, 12))
def test_rare_cap_sliding_removes_fixed_point_tangential_stiffness_in_one_step(flange_grid):
    h, cap_count = .1, 9
    points, query, calls = _analytic_sliding_scene(flange_grid, h)
    flange_count = 4*flange_grid**2
    assert len(points) == flange_count + cap_count
    policy = _policy(huber_delta_mm=.2)
    baseline = v2._refine(points, query, np.eye(3), np.zeros(3), diagonal=7., policy=policy, coarse=False)
    candidate = v3.refine_finite_surface_v3(points, query, np.eye(3), np.zeros(3), policy=policy)
    expected_kabsch = -cap_count*h/(flange_count+cap_count)
    assert len(calls) == 3  # V2 current query; V3 current + actual finite trial.
    assert np.allclose(baseline[0], np.eye(3), rtol=0, atol=1e-12)
    assert np.allclose(baseline[1], [0., 0., expected_kabsch], rtol=0, atol=1e-12)
    assert np.allclose(candidate[0], np.eye(3), rtol=0, atol=1e-12)
    assert np.allclose(candidate[1], [0., 0., -h], rtol=0, atol=1e-12)
    assert candidate[4]["backtracking_trials"] == 1
    assert candidate[4]["expanded_linear_equations"] == len(points)
    assert candidate[4]["fixed_rows_finite_huber_after"] < 1e-20
    assert abs(h+baseline[1][2]) > .09 and abs(h+candidate[1][2]) < 1e-12


@pytest.mark.parametrize("query_cap", (1, 2))
def test_fatal_actual_query_budget_propagates_including_after_an_accepted_prefix(query_cap):
    triangles = np.asarray([[[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]]])
    query, _, budgeted = _real_query(triangles, replace(v2.SurfaceRegistrationBudgetsV2(), max_query_calls=query_cap))
    points = np.column_stack((np.linspace(1., 8., 36), np.full(36, -.2), np.zeros(36)))
    with pytest.raises(v2._SurfaceFailure, match="REGISTRATION_QUERY_BUDGET_EXCEEDED"):
        v3.refine_finite_surface_v3(points, query, np.eye(3), np.zeros(3), policy=_policy(refine_iterations=2))
    assert budgeted.statistics["query_calls"] == query_cap
    assert budgeted.statistics["query_rows"] == 36*query_cap


@pytest.mark.parametrize("kind", ("policy_type", "bypassed_policy", "reflection", "scale", "shear",
                                  "rotation_shape", "rotation_nan", "translation_shape", "translation_nan", "short_scan"))
def test_invalid_policy_or_iterate_refuses_before_any_surface_query(kind):
    points, rotation, translation, policy = _points(), np.eye(3), np.zeros(3), _policy()
    if kind == "policy_type":
        policy = None
    elif kind == "bypassed_policy":
        object.__setattr__(policy, "refine_iterations", 0)
    elif kind == "reflection":
        rotation[0, 0] = -1.
    elif kind == "scale":
        rotation *= 2.
    elif kind == "shear":
        rotation[0, 1] = .1
    elif kind == "rotation_shape":
        rotation = np.eye(2)
    elif kind == "rotation_nan":
        rotation[0, 0] = np.nan
    elif kind == "translation_shape":
        translation = np.zeros(2)
    elif kind == "translation_nan":
        translation[0] = np.nan
    else:
        points = points[:35]
    calls = []

    def forbidden_query(value):
        calls.append(value)
        raise AssertionError("INVALID_ADMISSION_MUST_NOT_QUERY")

    with pytest.raises(ValueError):
        v3.refine_finite_surface_v3(points, forbidden_query, rotation, translation, policy=policy)
    assert calls == []


@pytest.mark.parametrize("kind", ("region", "normal", "point", "target", "delta"))
def test_invalid_mixed_linearization_refuses(kind):
    points, targets, normals, regions = _fixture()
    delta = .15
    if kind == "region":
        regions[0] = "BAD"
    elif kind == "normal":
        normals[0] *= 2.
    elif kind == "point":
        points[0, 0] = np.nan
    elif kind == "target":
        targets[0, 0] = np.inf
    else:
        delta = 0.
    with pytest.raises(ValueError, match="INVALID_MIXED_FINITE_LINEARIZATION"):
        v3._mixed_system(points, targets, normals, regions, delta)


@pytest.mark.parametrize("extent_scale", (0., 1e-11))
def test_coincident_or_subthreshold_extent_refuses_without_an_accepted_pose(extent_scale):
    points = _points()*extent_scale
    assert points.shape == (36, 3)
    extent = math.sqrt(math.fsum(float(np.dot(row, row)) for row in points)/len(points))
    assert extent <= 1e-10
    assert (np.unique(points, axis=0).shape[0] == 1) is (extent_scale == 0.)
    normals, regions = np.tile([0., 0., 1.], (36, 1)), np.full(36, "FACE")
    targets = points+.1*normals
    with pytest.raises(ValueError, match="DEGENERATE_WHOLE_PART_EXTENT"):
        v3._mixed_system(points, targets, normals, regions, .1)
    calls = []

    def query(current):
        calls.append(current.copy())
        return dict(closest_points=current+.1*normals, distances=np.full(36, .1),
                    oriented_normals=normals, closest_regions=regions,
                    normal_ambiguous=np.zeros(36, dtype=bool),
                    closest_point_ambiguous=np.zeros(36, dtype=bool))

    with pytest.raises(ValueError, match="DEGENERATE_WHOLE_PART_EXTENT"):
        v3.refine_finite_surface_v3(points, query, np.eye(3), np.zeros(3), policy=_policy())
    assert len(calls) == 1  # Initial support query only; no trial/accepted prefix.


def test_fixed_scope_and_limits():
    assert RESOURCE_LIMITS == dict(wall_seconds=60, cpu_seconds=45, process_tree_rss_bytes=1024**3)
    assert [4*grid**2 for grid in (6, 12)] == [144, 576]
