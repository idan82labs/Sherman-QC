"""Independent small V5 geometry gate; the prospective 108-case matrix is absent.

Invented coordinates only. Four fixed noisy grids are required recoveries, not
optional positives. Failure of one stays a failed case. No native/physical IO.
"""
import ast
from dataclasses import replace
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_noisy_mesh_planes_v5.py"
# Integer rows are orthogonal with squared length 81 and determinant +729.
R_NUMERATOR = np.array([[1, 8, 4], [8, 1, -4], [-4, 4, -7]], dtype=np.int64)
R = R_NUMERATOR.astype(float) / 9.
GATE = dict(tile_triangles=(128, 1024), halo_rings=1, min_support_rows=3,
            max_normal_angle_radians=math.pi/18, min_inplane_covariance_ratio=1e-8)
FIXED_GRID_FRAMES = (
    (np.eye(3), np.zeros(3), 1.),
    (R, np.array([3., -4., 5.]), 1.),
    (R, np.array([37., -21., 9.]), .25),
    (R.T, np.array([-9., 5., 11.]), 4.),
)


@pytest.fixture
def subject(monkeypatch):
    name = "_independent_noisy_mesh_planes_v5"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def grid(n=9, span=16., *, hole=False):
    coordinates = np.linspace(-span/2, span/2, n)
    points = np.array([[x, y, 0.] for y in coordinates for x in coordinates])
    triangles = []
    for j in range(n-1):
        for i in range(n-1):
            if hole and i in (3, 4) and j in (3, 4):
                continue
            a, b, c, d = j*n+i, j*n+i+1, (j+1)*n+i, (j+1)*n+i+1
            triangles.extend(((a, b, c), (b, d, c)))
    return points, np.asarray(triangles, dtype=np.int64)


def run(subject, points, triangles, epsilon=.05, *, policy=None, budgets=None):
    kwargs = dict(position_error_bounds_mm=epsilon)
    if policy is not None:
        kwargs["policy"] = policy
    if budgets is not None:
        kwargs["budgets"] = budgets
    return subject.extract_noisy_scan_plane_proposals_v5(points, triangles, **kwargs)


def budget(subject, **changes):
    return replace(subject.NoisyMeshPlaneBudgetsV5(), **changes)


def independent_edges(triangles):
    owners = {}
    for index, (a, b, c) in enumerate(triangles.tolist()):
        for edge in ((a, b), (b, c), (c, a)):
            owners.setdefault(tuple(sorted(edge)), []).append(index)
    neighbors = [set() for _ in triangles]
    for rows in owners.values():
        if len(rows) == 2:
            a, b = rows
            neighbors[a].add(b)
            neighbors[b].add(a)
    return owners, neighbors


def assert_boundary(result):
    assert result["search_exhaustive"] is False
    assert result["physical_plane_count"] is None and result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False and result["accepted_registration"] is False
    assert result["physical_flatness_verified"] is False and result["residual_trimming"] is False
    assert result["raw_coordinate_welding"] is False
    assert result["uncertainty_status"] == "STIPULATED_UNCALIBRATED"


def assert_unavailable(result, *, discarded_at_least=0):
    assert result["status"] == "UNAVAILABLE" and result["reasons"]
    assert all(result[key] == [] for key in ("cores", "closures", "proposals", "partitions"))
    stats = result["statistics"]
    assert stats["discarded_core_records"] >= discarded_at_least
    assert stats["core_construction_complete"] is False
    assert stats["core_evaluation_complete"] is False
    assert stats["scheduled_closure_evaluation_complete"] is False
    assert stats["all_qualified_core_closures_evaluated"] is False
    assert stats["evidence_scope"] == "ATTEMPTED_WORK_ONLY_ALL_PROPOSALS_INVALIDATED"
    assert_boundary(result)


def assert_witness(analysis, points, epsilon, true_normal=None):
    assert analysis["status"] == "PROPOSED"
    assert analysis["witness_status"] == "FOUND" and analysis["cone_status"] == "AVAILABLE"
    normal = np.asarray(analysis["normal"])
    anchor = np.asarray(analysis["witness_anchor_mm"])
    offset = analysis["witness_offset_from_anchor_mm"]
    residuals = np.abs((points-anchor) @ normal-offset)
    assert np.all(residuals <= epsilon)  # Original stipulated bounds: no added tolerance.
    assert np.linalg.norm(normal) == pytest.approx(1., abs=1e-14)
    assert analysis["raw_support_count"] == len(points)
    assert analysis["same_physical_plane_verified"] is False
    assert analysis["numerical"]["position_error_bound_was_increased"] is False
    if true_normal is not None:
        angle = math.atan2(float(np.linalg.norm(np.cross(normal, true_normal))),
                           abs(float(np.dot(normal, true_normal))))
        assert angle <= analysis["normal_angle_bound_radians"]


def assert_lineage(result, triangles):
    for record in [*result["cores"], *result["closures"]]:
        support = record["triangle_rows"]
        raw = record["raw_vertex_rows"]
        assert np.array_equal(raw, np.unique(triangles[support].reshape(-1)))
        assert len(np.unique(support)) == len(support)
        assert np.all((support >= 0) & (support < len(triangles)))
        for key in ("triangle_rows", "raw_vertex_rows"):
            array = record[key]
            assert array.dtype == np.dtype("<i8") and not array.flags.writeable
            assert hashlib.sha256(array.tobytes()).hexdigest() == record[key + "_sha256"]
            with pytest.raises(ValueError):
                array.setflags(write=True)
            current = array.base
            while isinstance(current, np.ndarray):
                with pytest.raises(ValueError):
                    current.setflags(write=True)
                current = current.base
    for proposal in result["proposals"]:
        key = "cores" if proposal["kind"] == "CORE" else "closures"
        assert result[key][proposal["record_id"]]["analysis"]["status"] == "PROPOSED"


def test_constructed_rotation_has_exact_integer_orthogonality_and_positive_orientation():
    assert np.array_equal(R_NUMERATOR @ R_NUMERATOR.T, 81*np.eye(3, dtype=np.int64))
    a, b, c = R_NUMERATOR.tolist()
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    assert determinant == 729


@pytest.mark.parametrize("frame", range(4))
def test_required_noisy_grid_core_and_closure_recovery(subject, frame):
    truth, triangles = grid()
    rotation, translation, scale = FIXED_GRID_FRAMES[frame]
    local = truth.copy()
    local[:, 2] = [.01*(-1 if (i//9+i%9) % 2 else 1) for i in range(81)]
    observed = (local*scale) @ rotation.T + translation
    world_truth = (truth*scale) @ rotation.T + translation
    epsilon = np.full(81, .05*scale)
    assert np.all(np.linalg.norm(observed-world_truth, axis=1) < epsilon)
    result = run(subject, observed, triangles, epsilon)
    assert result["status"] == "COMPLETE", result["reasons"]
    assert len(result["cores"]) == 2 and len(result["closures"]) == 1
    assert result["proposals"] == [dict(kind="CORE", record_id=0), dict(kind="CORE", record_id=1),
                                   dict(kind="CLOSURE", record_id=0)]
    assert result["closures"][0]["source_core_ids"] == [0, 1]
    for record in [*result["cores"], *result["closures"]]:
        assert np.array_equal(record["triangle_rows"], np.arange(128))
        assert np.array_equal(record["raw_vertex_rows"], np.arange(81))
        assert_witness(record["analysis"], observed, epsilon, rotation[:, 2])
    expected_counts = dict(edge_records=384, shared_edge_pairs=176, core_count=2,
        cores_evaluated=2, qualified_core_count=2, closures_scheduled=2, closures_evaluated=2,
        unique_closure_count=1, qualified_closure_count=1, fit_evaluations=3,
        full_support_row_tests=243, cone_row_tests=162, closure_triangle_tests=256,
        closure_edge_tests=352, triangle_memberships=640, raw_memberships=243,
        retained_array_bytes=9112, unscheduled_qualified_core_count=0)
    for key, expected in expected_counts.items():
        assert result["statistics"][key] == expected, key
    assert result["statistics"]["all_qualified_core_closures_evaluated"] is True
    assert_boundary(result)
    assert_lineage(result, triangles)


def test_heterogeneous_strictly_interior_noise_uses_original_per_row_bounds(subject):
    points, triangles = grid()
    epsilon = .04 + .005*(np.arange(81) % 3)
    points[:, 2] = .008 + .001*points[:, 0]  # Correlated tilt; not independent noise.
    assert np.all(np.abs(points[:, 2]) < epsilon)
    saved = epsilon.copy()
    result = run(subject, points, triangles, epsilon)
    assert result["status"] == "COMPLETE" and result["closures"]
    for record in [*result["cores"], *result["closures"]]:
        rows = record["raw_vertex_rows"]
        assert_witness(record["analysis"], points[rows], epsilon[rows], np.array([0., 0., 1.]))
    assert np.array_equal(epsilon, saved)
    assert result["input_digests"]["position_error_bounds_mm_sha256"] == hashlib.sha256(saved.astype("<f8").tobytes()).hexdigest()


def test_exact_coordinate_plane_zero_epsilon_is_a_separate_algebraic_control(subject):
    points, triangles = grid()
    result = run(subject, points, triangles, 0.)
    assert result["status"] == "COMPLETE" and result["proposals"]
    for record in [*result["cores"], *result["closures"]]:
        analysis = record["analysis"]
        assert_witness(analysis, points[record["raw_vertex_rows"]], 0., np.array([0., 0., 1.]))
        assert analysis["numerical"]["exact_coordinate_plane_witness"] is True


def test_rotated_zero_epsilon_refusal_is_not_counted_as_noisy_grid_recovery(subject):
    points, triangles = grid()
    points = points @ R.T + [3., -4., 5.]
    result = run(subject, points, triangles, 0.)
    assert result["status"] == "COMPLETE"
    assert result["proposals"] == [] and result["closures"] == []
    assert all(row["analysis"]["witness_status"] == "NUMERICALLY_UNRESOLVED_OFFSET_INTERVAL"
               for row in result["cores"])
    assert all(row["analysis"]["numerical"]["position_error_bound_was_increased"] is False
               for row in result["cores"])
    assert_boundary(result)


def test_five_point_feasible_plane_does_not_make_failed_tls_orientation_infeasible(subject):
    points = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [1., 0., 1.]])
    epsilon = np.array([0., 0., 0., 0., 1.])
    assert np.all(np.abs(points[:, 2]) <= epsilon)  # Independently witnessed z=0.
    analysis = subject._analyze_support(points, epsilon, dict(GATE))
    assert analysis["status"] == "REJECTED"
    assert analysis["witness_status"] == "WITNESS_NOT_FOUND"
    assert not any("INFEASIBLE" in reason for reason in analysis["reasons"])
    assert np.array_equal(epsilon, [0., 0., 0., 0., 1.])


@pytest.mark.parametrize("height,witness", [(.75, "FOUND"), (1., "NUMERICALLY_UNRESOLVED_OFFSET_INTERVAL"),
                                             (1.1, "WITNESS_NOT_FOUND")])
def test_fixed_normal_intervals_keep_strict_touching_and_disjoint_cases_distinct(subject, height, witness):
    points = np.array([[-10., -10., height], [10., -10., -height],
                       [-10., 10., -height], [10., 10., height]])
    epsilon = np.ones(4)
    gate = {**GATE, "max_normal_angle_radians": .25}
    analysis = subject._analyze_support(points, epsilon, gate)
    assert analysis["witness_status"] == witness
    assert analysis["status"] == ("PROPOSED" if height < 1 else "REJECTED")
    assert analysis["numerical"]["position_error_bound_was_increased"] is False


def test_pointwise_corner_cone_admits_poles_but_joint_support_has_no_plane(subject):
    corners = np.array([[-10., -10., 0.], [10., -10., 0.], [-10., 10., 0.], [10., 10., 0.]])
    epsilon = np.ones(4)
    gate = {**GATE, "max_normal_angle_radians": .25}
    core = subject._analyze_support(corners, epsilon, gate)
    assert core["status"] == "PROPOSED"
    assert core["normal_angle_bound_radians"] >= math.asin(.2)
    assert core["normal_angle_bound_radians"] < .25
    assert core["cone_B_squared_mm2"] == pytest.approx(4.)
    assert core["cone_B_squared_upper_mm2"] >= 4.  # Scalar epsilon=1: (1+1)^2.
    points = np.vstack((corners, [[0., 0., 1.], [0., 0., -1.]]))
    all_epsilon = np.r_[epsilon, 0., 0.]
    _, cap = subject._configuration(subject.NoisyMeshPlanePolicyV5(), subject.NoisyMeshPlaneBudgetsV5())
    stats = subject._base()["statistics"]
    mask, diagnostic = subject._cone_mask(points, all_epsilon, core, cap, stats)
    assert mask.all() and stats["cone_row_tests"] == 6
    assert diagnostic["pointwise_possibility_only"] is True
    assert diagnostic["joint_compatibility_verified"] is False
    joint = subject._analyze_support(points, all_epsilon, gate)
    assert joint["status"] == "REJECTED" and joint["witness_status"] == "WITNESS_NOT_FOUND"
    # The two exact poles force d=0,n_z=0. A unit xy normal has
    # max_corner |n.p| = 10(|n_x|+|n_y|) >= 10 > the corner error bound 1.
    assert 10. > 1.


def test_covariance_and_error_energy_have_independent_uniform_identity_oracles(subject):
    points, _ = grid()
    points[:, 2] = .01*np.where(np.arange(81) % 2, -1., 1.)
    epsilon = .04 + .005*(np.arange(81) % 3)
    analysis = subject._analyze_support(points, epsilon, dict(GATE))
    assert analysis["status"] == "PROPOSED"
    mean = np.array([math.fsum(points[:, j])/81 for j in range(3)])
    covariance = np.array([[math.fsum((points[:, j]-mean[j])*(points[:, k]-mean[k]))/81
                            for k in range(3)] for j in range(3)])
    error_mean = math.fsum(epsilon)/81
    energy = math.fsum((float(value)+error_mean)**2 for value in epsilon)/81
    np.testing.assert_allclose(analysis["covariance_mm2"], covariance, rtol=1e-12, atol=1e-13)
    assert analysis["cone_B_squared_mm2"] == pytest.approx(energy, rel=1e-13)
    assert analysis["cone_B_squared_upper_mm2"] >= energy
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert analysis["cone_gap_lower_mm2"] <= eigenvalues[1]-eigenvalues[0]
    assert analysis["numerical"]["computed_normal_angle_guard_radians"] >= 0
    repeated = subject._analyze_support(np.repeat(points, 2, axis=0), np.repeat(epsilon, 2), dict(GATE))
    assert repeated["status"] == "PROPOSED" and repeated["raw_support_count"] == 162
    assert repeated["cone_B_squared_mm2"] == pytest.approx(energy, rel=1e-13)
    assert repeated["normal_angle_bound_radians"] >= .99*analysis["normal_angle_bound_radians"]


@pytest.mark.parametrize("points", [np.array([[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]]),
    np.vstack((10*np.eye(3), -10*np.eye(3)))])
def test_unresolved_smallest_eigengap_never_becomes_zero_angle_success(subject, points):
    analysis = subject._analyze_support(points, np.full(len(points), .01), dict(GATE))
    assert analysis["status"] == "REJECTED" and analysis["cone_status"] == "UNRESOLVED"
    assert analysis["normal_angle_bound_radians"] is None


def test_computed_normal_angle_guard_covers_an_independently_perturbed_eigenvector(subject, monkeypatch):
    points = np.array([[-10., -20., 0.], [10., -20., 0.], [-10., 20., 0.], [10., 20., 0.]])
    theta = 1e-5
    mixing = np.array([[math.cos(theta), -math.sin(theta), 0.],
                       [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
    original = subject.np.linalg.eigh
    calls = []
    def perturbed(covariance):
        values, axes = original(covariance)
        calls.append(covariance.copy())
        return values, axes @ mixing
    monkeypatch.setattr(subject.np.linalg, "eigh", perturbed)
    analysis = subject._analyze_support(points, np.full(4, .5), dict(GATE))
    assert len(calls) == 1 and analysis["status"] == "PROPOSED"
    computed = np.asarray(analysis["normal"])
    truth = np.array([0., 0., 1.])
    actual_error = math.atan2(float(np.linalg.norm(np.cross(computed, truth))),
                              abs(float(computed @ truth)))
    assert actual_error == pytest.approx(theta, rel=1e-10)
    guard = analysis["numerical"]["computed_normal_angle_guard_radians"]
    assert guard >= actual_error
    assert guard < 2*theta  # The computed-normal guard, not the much broader model cone.


def test_near_smallest_eigengap_is_numerically_unresolved(subject):
    axes = np.diag([1., 1.+1e-12, 2.])
    points = np.vstack((axes, -axes))
    analysis = subject._analyze_support(points, np.ones(6), dict(GATE))
    assert analysis["status"] == "REJECTED" and analysis["cone_status"] == "UNRESOLVED"
    assert analysis["cone_gap_lower_mm2"] <= 0.
    assert analysis["normal_angle_bound_radians"] is None


def test_negative_cone_numerator_does_not_become_clamped_zero_angle(subject):
    axes = np.diag([1., 2., 3.])
    points = np.vstack((axes, -axes))
    analysis = subject._analyze_support(points, np.zeros(6), dict(GATE))
    assert analysis["cone_gap_lower_mm2"] > 0.
    assert analysis["cone_numerator_upper_mm2"] < 0.
    assert analysis["status"] == "REJECTED" and analysis["cone_status"] == "UNRESOLVED"
    assert analysis["normal_angle_bound_radians"] is None


def test_real_core_survives_its_jointly_failed_whole_closure(subject):
    points, triangles = grid(span=40.)
    points = np.vstack((points, [[20., 25., 0.], [25., 25., 0.], [0., 0., 1.], [0., 0., -1.]]))
    triangles = np.vstack((triangles, [[79, 80, 81], [80, 81, 82], [81, 82, 83], [82, 83, 84]]))
    epsilon = np.r_[np.ones(83), 0., 0.]
    result = run(subject, points, triangles, epsilon)
    assert result["status"] == "COMPLETE", result["reasons"]
    retained = [row for row in result["cores"] if row["closure_disposition"] == "WHOLE_CLOSURE_REJECTED_CORE_RETAINED"]
    assert retained, "Required actual core/closure counterexample was not recovered"
    for core in retained:
        assert core["analysis"]["status"] == "PROPOSED"
        assert dict(kind="CORE", record_id=core["core_id"]) in result["proposals"]
        closure = result["closures"][core["closure_id"]]
        assert np.array_equal(closure["triangle_rows"], np.arange(132))
        assert np.array_equal(closure["raw_vertex_rows"], np.arange(85))
        assert closure["analysis"]["status"] == "REJECTED"
        assert closure["analysis"]["witness_status"] == "WITNESS_NOT_FOUND"
        assert dict(kind="CLOSURE", record_id=core["closure_id"]) not in result["proposals"]
    assert_lineage(result, triangles)


def test_hole_remains_original_triangle_union_with_unreferenced_center_row(subject):
    points, triangles = grid(hole=True)
    assert len(triangles) == 120 and 40 not in triangles
    result = run(subject, points, triangles)
    assert result["status"] == "COMPLETE" and len(result["closures"]) == 1
    assert np.array_equal(result["raw_input_accounting"]["unreferenced_raw_vertex_rows"], [40])
    assert result["raw_input_accounting"]["unreferenced_disposition"] == "INPUT_RETAINED_NOT_TRIANGLE_SUPPORT"
    assert np.array_equal(result["closures"][0]["triangle_rows"], np.arange(120))
    assert result["finite_support_representation"] == "ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON"
    centroids = points[triangles].mean(axis=1)
    assert not np.any((np.abs(centroids[:, 0]) < 2) & (np.abs(centroids[:, 1]) < 2))
    assert_lineage(result, triangles)


def test_equal_xyz_distinct_raw_id_components_are_not_welded(subject):
    points, triangles = grid()
    points = np.vstack((points, points))
    triangles = np.vstack((triangles, triangles+81))
    result = run(subject, points, triangles)
    assert result["status"] == "COMPLETE" and len(result["closures"]) == 2
    supports = {tuple(row["triangle_rows"]) for row in result["closures"]}
    assert supports == {tuple(range(128)), tuple(range(128, 256))}
    assert {len(row["raw_vertex_rows"]) for row in result["closures"]} == {81}
    assert_lineage(result, triangles)
    assert_boundary(result)


def test_disconnected_two_triangle_perpendicular_cap_is_not_dropped_by_area(subject):
    points, triangles = grid()
    points = np.vstack((points, [[30., 0., 0.], [30., 2., 0.], [30., 0., 2.], [30., 2., 2.]]))
    triangles = np.vstack((triangles, [[81, 82, 83], [82, 84, 83]]))
    epsilon = np.r_[np.full(81, .05), np.full(4, .005)]
    result = run(subject, points, triangles, epsilon)
    assert result["status"] == "COMPLETE" and len(result["closures"]) == 2
    tiny = next(row for row in result["closures"] if np.array_equal(row["triangle_rows"], [128, 129]))
    assert_witness(tiny["analysis"], points[tiny["raw_vertex_rows"]], epsilon[tiny["raw_vertex_rows"]],
                   np.array([1., 0., 0.]))


def test_original_edge_bfs_partitions_and_one_ring_halos_are_complete(subject):
    points, triangles = grid(n=11, span=20.)
    _, neighbors = independent_edges(triangles)
    result = run(subject, points, triangles)
    assert result["status"] == "COMPLETE" and len(result["cores"]) == 3
    for partition in result["partitions"]:
        family = [row for row in result["cores"] if row["tile_triangles"] == partition["tile_triangles"]]
        bases = np.concatenate([row["base_triangle_rows"] for row in family])
        assert np.array_equal(np.sort(bases), np.arange(200))
        for core in family:
            base = set(core["base_triangle_rows"].tolist())
            expected = base | set().union(*(neighbors[index] for index in base))
            assert set(core["triangle_rows"].tolist()) == expected
            assert np.all(partition["triangle_core_labels"][core["base_triangle_rows"]] == core["core_id"])
    assert_lineage(result, triangles)


@pytest.mark.parametrize("quota", [0, 1])
def test_unscheduled_qualified_cores_remain_proposals(subject, quota):
    points, triangles = grid()
    result = run(subject, points, triangles, budgets=budget(subject, max_closure_schedules=quota))
    assert result["status"] == "COMPLETE"
    assert [row for row in result["proposals"] if row["kind"] == "CORE"] == [
        dict(kind="CORE", record_id=0), dict(kind="CORE", record_id=1)]
    assert len(result["scheduled_core_ids"]) == quota
    assert result["statistics"]["unscheduled_qualified_core_count"] == 2-quota
    assert result["statistics"]["scheduled_closure_evaluation_complete"] is True
    assert result["statistics"]["all_qualified_core_closures_evaluated"] is False
    assert sum(row["closure_disposition"] == "NOT_SCHEDULED_BY_POLICY" for row in result["cores"]) == 2-quota


def test_input_mutation_cannot_change_copied_analysis_or_immutable_lineage(subject):
    points, triangles = grid()
    epsilon = np.full(81, .05)
    result = run(subject, points, triangles, epsilon)
    assert result["status"] == "COMPLETE"
    digests = dict(result["input_digests"])
    normals = [list(row["analysis"]["normal"]) for row in result["cores"]]
    saved = triangles.copy()
    points[:] = 999.
    triangles[:] = 0
    epsilon[:] = 999.
    assert result["input_digests"] == digests
    assert [row["analysis"]["normal"] for row in result["cores"]] == normals
    assert_lineage(result, saved)


@pytest.mark.parametrize("changes", [dict(max_raw_rows=80), dict(max_triangles=127),
    dict(max_edge_records=383), dict(max_core_records=0), dict(max_core_triangle_rows=127),
    dict(max_fit_evaluations=0), dict(max_metadata_bytes=1)])
def test_early_fixed_caps_whole_refuse(subject, changes):
    points, triangles = grid()
    assert_unavailable(run(subject, points, triangles, budgets=budget(subject, **changes)))


@pytest.mark.parametrize("changes", [dict(max_core_records=1), dict(max_fit_evaluations=2),
    dict(max_full_support_row_tests=162), dict(max_cone_row_tests=81),
    dict(max_closure_triangle_tests=128), dict(max_closure_edge_tests=176),
    dict(max_triangle_memberships=512), dict(max_raw_memberships=162),
    dict(max_retained_array_bytes=7440), dict(max_metadata_bytes=8192)])
def test_late_fixed_caps_discard_every_qualified_prefix(subject, changes):
    points, triangles = grid()
    result = run(subject, points, triangles, budgets=budget(subject, **changes))
    assert_unavailable(result, discarded_at_least=1)
    assert result["statistics"]["qualified_core_count"] >= 1


@pytest.mark.parametrize("field,value", [("max_raw_rows", 1_000_001), ("max_triangles", True),
    ("max_fit_evaluations", -1), ("chunk_rows", 0)])
def test_bypass_mutated_budget_default_cannot_raise_or_corrupt_hard_caps(subject, field, value):
    default = subject.extract_noisy_scan_plane_proposals_v5.__kwdefaults__["budgets"]
    before = getattr(default, field)
    points, triangles = grid()
    try:
        object.__setattr__(default, field, value)
        assert_unavailable(run(subject, points, triangles))
    finally:
        object.__setattr__(default, field, before)


@pytest.mark.parametrize("field,value", [("tile_triangles", (64, 1024)), ("halo_rings", True),
    ("min_support_rows", 2), ("max_normal_angle_radians", float("nan")),
    ("min_inplane_covariance_ratio", 0.)])
def test_bypass_mutated_policy_default_is_revalidated(subject, field, value):
    default = subject.extract_noisy_scan_plane_proposals_v5.__kwdefaults__["policy"]
    before = getattr(default, field)
    points, triangles = grid()
    try:
        object.__setattr__(default, field, value)
        assert_unavailable(run(subject, points, triangles))
    finally:
        object.__setattr__(default, field, before)


def test_configuration_snapshot_is_detached_before_owned_input_work(subject, monkeypatch):
    points, triangles = grid()
    policy, budgets = subject.NoisyMeshPlanePolicyV5(), subject.NoisyMeshPlaneBudgetsV5()
    original = subject._owned
    def mutate_after_configuration(value, dtype):
        object.__setattr__(policy, "min_support_rows", 999_999)
        object.__setattr__(budgets, "max_raw_rows", 0)
        return original(value, dtype)
    monkeypatch.setattr(subject, "_owned", mutate_after_configuration)
    result = run(subject, points, triangles, policy=policy, budgets=budgets)
    assert result["status"] == "COMPLETE" and result["proposals"]
    assert result["policy"]["min_support_rows"] == 3
    assert result["budgets"]["max_raw_rows"] == 1_000_000


def test_caller_triangle_mutation_during_point_snapshot_is_refused_before_indexing(subject, monkeypatch):
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    original = subject._owned
    snapshots = []
    def mutate_triangle(value, dtype):
        copied = original(value, dtype)
        snapshots.append(dtype)
        if value is points:
            triangles[0, 2] = -1
        return copied
    monkeypatch.setattr(subject, "_owned", mutate_triangle)
    monkeypatch.setattr(subject, "_adjacency", lambda *a: pytest.fail("Invalid owned indices reached adjacency"))
    result = run(subject, points, triangles)
    assert snapshots
    assert_unavailable(result)
    assert "RAW_INDEX" in " ".join(result["reasons"])


@pytest.mark.parametrize("error", [RuntimeError("invented late fit"), IndexError("invented late fit")])
def test_unexpected_fit_failure_after_good_cores_whole_clears(subject, monkeypatch, error):
    points, triangles = grid()
    original = subject._analyze_support
    calls = []
    def fail_third(*args):
        calls.append(1)
        if len(calls) == 3:
            raise error
        return original(*args)
    monkeypatch.setattr(subject, "_analyze_support", fail_third)
    result = run(subject, points, triangles)
    assert len(calls) == 3
    assert_unavailable(result, discarded_at_least=2)
    assert result["statistics"]["qualified_core_count"] == 2
    assert result["statistics"]["discarded_proposals"] == 2
    assert result["statistics"]["fit_evaluations"] == 3
    assert result["statistics"]["full_support_row_tests"] == 243


def test_unexpected_closure_graph_failure_after_good_cores_whole_clears(subject, monkeypatch):
    points, triangles = grid()
    def fail_graph(*args, **kwargs):
        raise IndexError("invented closure graph")
    monkeypatch.setattr(subject, "connected_components", fail_graph)
    result = run(subject, points, triangles)
    assert_unavailable(result, discarded_at_least=2)
    assert result["statistics"]["qualified_core_count"] == 2
    assert result["statistics"]["closure_triangle_tests"] == 128
    assert result["statistics"]["closure_edge_tests"] == 176


def test_process_interrupt_is_not_swallowed_as_an_ordinary_geometry_refusal(subject, monkeypatch):
    points, triangles = grid()
    def interrupted(*args):
        raise KeyboardInterrupt("invented control-flow interrupt")
    monkeypatch.setattr(subject, "_cone_mask", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(subject, points, triangles)


@pytest.mark.parametrize("epsilon", [True, -1., float("inf"), float("nan"), 1000.001,
    [0.05]*81, np.full(80, .05), np.full(81, True), np.full(81, .05, dtype=object)])
def test_malformed_or_unsupported_error_stipulations_refuse(subject, epsilon):
    points, triangles = grid()
    assert_unavailable(run(subject, points, triangles, epsilon))


@pytest.mark.parametrize("kind", ["repeated", "out_of_range", "unsigned_wrap", "nonmanifold", "collinear"])
def test_invalid_raw_triangle_topology_is_not_repaired(subject, kind):
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., -2., 0.], [1., 1., 1.]])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    if kind == "repeated":
        triangles[0] = [0, 0, 2]
    elif kind == "out_of_range":
        triangles[0, 2] = 5
    elif kind == "unsigned_wrap":
        triangles = np.array([[0, 1, 2**64-1]], dtype=np.uint64)
    elif kind == "nonmanifold":
        triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    else:
        points[2] = [4., 0., 0.]
    assert_unavailable(run(subject, points, triangles))


def test_extreme_translation_resolution_refusal_is_not_plane_recovery(subject):
    points = np.array([[999999., 999999., 999999.],
                       [999999.+1e-9, 999999., 999999.],
                       [999999., 999999.+1e-9, 999999.]])
    result = run(subject, points, np.array([[0, 1, 2]]), .001)
    assert_unavailable(result)
    assert "NUMERICAL_COVARIANCE_RESOLUTION_UNSUPPORTED" in result["reasons"]


def test_graph_storage_uses_sparse_original_triangle_edges_not_point_pair_distances(subject, monkeypatch):
    points, triangles = grid(n=11, span=20.)
    original, allocations = subject.coo_matrix, []
    def sparse_graph(arg, *, shape):
        allocations.append((shape, len(arg[0])))
        return original(arg, shape=shape)
    monkeypatch.setattr(subject, "coo_matrix", sparse_graph)
    result = run(subject, points, triangles)
    assert result["status"] == "COMPLETE"
    assert allocations and all(shape == (200, 200) and entries <= 600 for shape, entries in allocations)
    tree = ast.parse(SOURCE.read_text())
    forbidden = {"pdist", "cdist", "distance_matrix", "pairwise_distances"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and (name.startswith("OCP") or name.startswith("open3d") or name.startswith("backend"))
                   for name in imported)
