"""Invented finite mesh controls for the additive fixed-seed proposal lane.

No CAD, native API, fixture labels or true pose enters the extractor. Known
planes and triangle labels are post-call oracles only. The selected schedule
is incomplete/order-dependent; these tests do not assert general permutation
invariance, calibrated uncertainty, physical surfaces, absence or bend counts.
"""

from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module():
    prefix = "isolated_seeded_plane_proposals_v1"
    def owned(key):
        return key in (prefix, "backend") or key.startswith(prefix+".") or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved: del sys.modules[key]
    for name in (prefix, "backend"):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT/"backend")]
        sys.modules[name] = package
    try:
        yield importlib.import_module(prefix+".cad_guided_seeded_plane_proposals_v1")
    finally:
        for key in tuple(sys.modules):
            if owned(key): del sys.modules[key]
        sys.modules.update(saved)


def _sha(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def _faces(specifications, alternate=False):
    """Weld only deliberately shared exact fixture coordinates, before noise."""
    points, triangles, labels, lookup = [], [], [], {}
    for face, (us, vs, position, omit) in enumerate(specifications):
        grid = []
        for v in vs:
            row = []
            for u in us:
                xyz = tuple(float(x) for x in position(u, v))
                if xyz not in lookup:
                    lookup[xyz] = len(points)
                    points.append(xyz)
                row.append(lookup[xyz])
            grid.append(row)
        for j in range(len(vs)-1):
            for i in range(len(us)-1):
                if (i, j) in omit: continue
                a, b, c, d = grid[j][i], grid[j][i+1], grid[j+1][i], grid[j+1][i+1]
                cell = [(a, b, c), (b, d, c)] if alternate else [(a, b, d), (a, d, c)]
                triangles.extend(cell)
                labels.extend([face, face])
    return np.array(points, dtype=np.float64), np.array(triangles, dtype=np.int64), np.array(labels)


def _welded(alternate=False):
    grid = [0., 4., 8., 12., 16.]
    return _faces([
        (grid, grid, lambda x, y: (x, y, 0.), set()),
        (grid, [0., .25, 4., 8., 12., 16.], lambda x, z: (x, 0., z), set()),
        (grid, [0., .25], lambda y, z: (0., y, z), set()),
    ], alternate)


def _pose(index):
    if index == 0: return np.eye(3), np.zeros(3)
    if index == 1:
        return np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]]), np.array([32., -16., 8.])
    # Rational proper rotation from the unit quaternion (1,2,2,0)/3.
    return np.array([[1., 8., 4.], [8., 1., -4.], [-4., 4., -7.]])/9., np.array([123., -45., 7.])


def _noise(points, enabled):
    if not enabled: return np.zeros_like(points)
    rows = np.arange(len(points))[:, None]
    return ((rows*np.array([3, 5, 7])+np.array([0, 1, 2])) % 5-2)*.0005


def _run(module, points, triangles, epsilon=.05, **kwargs):
    return module.extract_seeded_plane_proposals_v1(points, triangles,
        position_error_bounds_mm=epsilon, **kwargs)


def _complete(result, points, triangles, epsilon=.05):
    assert result["status"] == "COMPLETE", result
    assert result["production_use_allowed"] is result["accepted_registration"] is False
    assert result["physical_plane_count"] is result["observed_bend_count"] is None
    assert result["raw_coordinate_welding"] is result["residual_trimming"] is False
    assert result["finite_support_representation"] == "ORIGINAL_TRIANGLE_UNION_NOT_FILLED_POLYGON"
    assert result["search_exhaustive"] is False
    stats = result["statistics"]
    assert stats["selected_policy_evaluation_complete"] is True
    assert result["work_counter_scope"] == "LOGICAL_SUPPORT_EVALUATIONS_NOT_EVERY_ARRAY_PASS"
    assert stats["preparation_validated_raw_rows"] == len(points)
    assert stats["preparation_validated_triangles"] == stats["preparation_triangle_normals"] == len(triangles)
    assert stats["preparation_adjacency_edge_records"] == 3*len(triangles)
    schedule = result["scheduled_seed_triangle_rows"]
    assert len(schedule) == len(set(schedule)) == len(result["seeds"]) == stats["completed_seed_count"]
    assert all(type(i) is int and 0 <= i < len(triangles) for i in schedule)
    assert result["unscheduled_seed_triangle_count"]+len(schedule) == len(triangles)
    assert [row["seed_triangle_row"] for row in result["seeds"]] == schedule
    eps = np.full(len(points), epsilon) if np.isscalar(epsilon) else epsilon
    assert result["input_digests"] == dict(points_sha256=_sha(points, "<f8"),
        triangle_raw_rows_sha256=_sha(triangles, "<i8"), position_error_bounds_mm_sha256=_sha(eps, "<f8"))
    seen = set()
    for seed in result["seeds"]:
        assert seed["normal_is_uncertainty_bound"] is False
        assert seed["threshold_scope"] == "FIXED_SEED_PROPOSAL_ONLY_NOT_POSITION_UNCERTAINTY_ENCLOSURE"
        region = result["regions"][seed["region_id"]]
        assert seed["seed_triangle_row"] in region["triangle_rows"]
        expected = "EXACT_SUPPORT_DUPLICATE" if seed["region_id"] in seen else region["analysis"]["status"]
        assert seed["disposition"] == expected
        seen.add(seed["region_id"])
    assert seen == set(range(len(result["regions"])))
    for index, region in enumerate(result["regions"]):
        trirows, rawrows = region["triangle_rows"], region["raw_vertex_rows"]
        assert region["region_id"] == index
        assert np.array_equal(trirows, np.unique(trirows)) and len(trirows)
        assert np.array_equal(rawrows, np.unique(triangles[trirows].ravel()))
        assert region["triangle_rows_sha256"] == _sha(trirows, "<i8")
        assert region["raw_vertex_rows_sha256"] == _sha(rawrows, "<i8")
        for array in (trirows, rawrows):
            assert array.dtype == np.dtype("<i8") and not array.flags.writeable
            with pytest.raises(ValueError): array.setflags(write=True)
        analysis = region["analysis"]
        assert analysis["raw_support_count"] == len(rawrows)
        assert analysis["fit_measure"] == "UNIFORM_ORIGINAL_RAW_IDENTITIES_NOT_XYZ_DEDUPLICATION"
        assert analysis["same_physical_plane_verified"] is False
        if analysis["status"] == "PROPOSED":
            assert analysis["witness_status"] == "FOUND" and analysis["cone_status"] == "AVAILABLE"
            assert analysis["numerical"]["position_error_bound_was_increased"] is False
            assert analysis["normal_angle_bound_radians"] <= math.pi/18
            # Independent all-row dot calculation (longdouble is wider only
            # where provided). The 1e-12 comparison allowance is floating test
            # arithmetic, not an increase to the input epsilon or a certificate.
            residual = np.abs((points[rawrows].astype(np.longdouble)
                - np.array(analysis["witness_anchor_mm"], dtype=np.longdouble))
                @ np.array(analysis["normal"], dtype=np.longdouble)
                - np.longdouble(analysis["witness_offset_from_anchor_mm"]))
            assert np.all(residual <= eps[rawrows]+1e-12)
    assert result["proposals"] == [dict(kind="SEEDED_REGION", record_id=i)
        for i, row in enumerate(result["regions"]) if row["analysis"]["status"] == "PROPOSED"]
    assert stats["unique_region_count"] == stats["fit_evaluations"] == len(result["regions"])
    assert stats["qualified_region_count"] == len(result["proposals"])
    assert stats["triangle_memberships"] == sum(len(r["triangle_rows"]) for r in result["regions"])
    assert stats["raw_memberships"] == stats["full_support_row_tests"] == sum(len(r["raw_vertex_rows"]) for r in result["regions"])
    edges = Counter(tuple(sorted((int(a), int(b)))) for tri in triangles
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
    assert max(edges.values()) <= 2
    assert stats["seed_selection_triangle_visits"] == stats["proposal_triangle_visits"] == len(schedule)*len(triangles)
    assert stats["proposal_raw_row_visits"] == len(schedule)*len(points)
    assert stats["proposal_edge_visits"] == len(schedule)*sum(v == 2 for v in edges.values())
    absent = np.setdiff1d(np.arange(len(points)), triangles.ravel())
    assert np.array_equal(result["raw_input_accounting"]["unreferenced_raw_vertex_rows"], absent)
    assert stats["retained_array_bytes"] == 8*(len(absent)+stats["triangle_memberships"]+stats["raw_memberships"])


def _failed(result):
    assert result["status"] == "UNAVAILABLE", result
    assert result["seeds"] == result["regions"] == result["proposals"] == []
    assert result["raw_input_accounting"] is None
    assert result["statistics"]["selected_policy_evaluation_complete"] is False
    assert result["statistics"]["evidence_scope"] == "ATTEMPTED_WORK_ONLY_ALL_PROPOSALS_INVALIDATED"


@pytest.mark.parametrize("pose", [0, 1, 2])
@pytest.mark.parametrize("noisy", [False, True])
def test_actual_welded_broad_planes_and_thin_cap_use_all_raw_support(module, pose, noisy):
    truth, triangles, labels = _welded()
    rotation, translation = _pose(pose)
    assert np.allclose(rotation.T@rotation, np.eye(3), atol=1e-15, rtol=0)
    assert abs(np.linalg.det(rotation)-1.) < 1e-15
    points = (truth+_noise(truth, noisy))@rotation.T+translation
    latent = truth@rotation.T+translation
    assert np.max(np.linalg.norm(points-latent, axis=1)) < .002 < .05
    saved = points.copy(), triangles.copy()
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    for face, normal in ((0, [0., 0., 1.]), (1, [0., 1., 0.])):
        expected = np.flatnonzero(labels == face)
        matches = [row for row in result["regions"] if np.array_equal(row["triangle_rows"], expected)
                   and row["analysis"]["status"] == "PROPOSED"]
        assert matches, "The fixed broad-face positive may not be renamed as an abstention"
        analysis = matches[0]["analysis"]
        target = rotation@normal
        angle = math.atan2(np.linalg.norm(np.cross(analysis["normal"], target)),
                           abs(float(np.dot(analysis["normal"], target))))
        assert angle <= analysis["normal_angle_bound_radians"]+1e-10
    cap_rows = np.flatnonzero(labels == 2)
    caps = [row for row in result["regions"] if np.array_equal(row["triangle_rows"], cap_rows)]
    assert caps and all(row["analysis"]["status"] == "REJECTED" for row in caps)
    assert all("CONDITIONAL" in " ".join(row["analysis"]["reasons"]) for row in caps)
    assert np.array_equal(points, saved[0]) and np.array_equal(triangles, saved[1])


@pytest.mark.parametrize("alternate,permuted", [(False, True), (True, False), (True, True)])
def test_row_relabeling_and_finite_diagonal_change_preserve_broad_positive_not_seed_identity(module, alternate, permuted):
    points, triangles, labels = _welded(alternate)
    if permuted:
        order = np.arange(len(points))[::-1]
        inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
        points, triangles = points[order].copy(), inverse[triangles[::-1]].copy()
        labels = labels[::-1].copy()
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    for face in (0, 1):
        assert any(row["analysis"]["status"] == "PROPOSED"
            and np.array_equal(row["triangle_rows"], np.flatnonzero(labels == face)) for row in result["regions"])


@pytest.mark.parametrize("offset", [0., .04, 3.])
def test_disconnected_parallel_and_coincident_meshes_keep_distinct_original_ids(module, offset):
    square = np.array([[0., 0., 0.], [16., 0., 0.], [16., 16., 0.], [0., 16., 0.]])
    points = np.vstack((square, square+[0., 0., offset]))
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    assert len(result["proposals"]) == 2
    assert {tuple(r["raw_vertex_rows"]) for r in result["regions"]} == {(0, 1, 2, 3), (4, 5, 6, 7)}
    assert all(r["analysis"]["status"] == "PROPOSED" for r in result["regions"])


def test_seed_component_is_retained_even_when_another_eligible_component_is_larger(module):
    points, triangles, labels = _faces([
        ([0., 8., 16.], [0., 8., 16.], lambda x, y: (x, y, 0.), set()),
        ([40., 56.], [0., 16.], lambda x, y: (x, y, 0.), set()),
    ])
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    small_rows = np.flatnonzero(labels == 1)
    small = [row for row in result["regions"] if np.array_equal(row["triangle_rows"], small_rows)]
    assert len(small) == 1 and small[0]["analysis"]["status"] == "PROPOSED"
    assert len(small[0]["raw_vertex_rows"]) == 4 < 9
    for seed in result["seeds"]:
        if seed["seed_triangle_row"] in small_rows:
            assert seed["region_id"] == small[0]["region_id"]
            assert seed["compatible_triangle_count"] == len(triangles)
            assert seed["seed_connected_triangle_count"] == len(small_rows)


def test_point_one_proposal_slab_cannot_replace_point_zero_five_whole_support_witness(module):
    grid = [-16., -8., 0., 8., 16.]
    def position(x, y):
        height = 0. if abs(x) <= 8. and abs(y) <= 8. else .09*(-1.)**(int(x/8)+int(y/8))
        return x, y, height
    points, triangles, _ = _faces([(grid, grid, position, set())])
    # Reflection symmetry makes z orthogonal to x and y in the all-row
    # covariance. The outer ring has both +.09 and -.09: its z-normal offset
    # intervals under epsilon .05 have no intersection. A central flat seed
    # nevertheless admits every row under the separate .05+.05 slab.
    exact = [[Fraction.from_float(float(value)) for value in row] for row in points]
    assert sum(row[2] for row in exact) == 0
    assert sum(row[0]*row[2] for row in exact) == 0
    assert sum(row[1]*row[2] for row in exact) == 0
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    whole = [row for row in result["regions"] if np.array_equal(row["triangle_rows"], np.arange(len(triangles)))]
    assert len(whole) == 1
    analysis = whole[0]["analysis"]
    assert analysis["status"] == "REJECTED" and analysis["witness_status"] == "WITNESS_NOT_FOUND"
    assert "WITNESS_NOT_FOUND" in analysis["reasons"]
    assert analysis["offset_interval_mm"][0] > analysis["offset_interval_mm"][1]
    assert analysis["numerical"]["position_error_bound_was_increased"] is False
    flat = [seed for seed in result["seeds"] if seed["region_id"] == whole[0]["region_id"]
            and np.all(points[triangles[seed["seed_triangle_row"]], 2] == 0.)]
    assert flat and all(seed["seed_max_position_error_mm"] == .05 for seed in flat)
    assert max(abs(points[:, 2])) == .09 < .05+.05


def test_hole_is_an_original_triangle_union_and_center_row_remains_unreferenced(module):
    grid = [-12., -8., -4., 0., 4., 8., 12.]
    points, triangles, _ = _faces([(grid, grid, lambda x, y: (x, y, 0.), {(2, 2), (2, 3), (3, 2), (3, 3)})])
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    assert len(result["proposals"]) == 1
    assert np.array_equal(result["regions"][0]["triangle_rows"], np.arange(len(triangles)))
    xyz = points[triangles]
    area = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1).sum()/2
    assert area == 24.*24.-8.*8.
    center = int(np.flatnonzero(np.all(points == [0., 0., 0.], axis=1))[0])
    assert np.array_equal(result["raw_input_accounting"]["unreferenced_raw_vertex_rows"], [center])
    assert center not in result["regions"][0]["raw_vertex_rows"]


def test_curved_bridge_cannot_chain_neighbor_normals_into_one_plane(module):
    angles = np.radians(np.arange(-20., 21., 5.))
    points, triangles, _ = _faces([(angles, [-8., -4., 0., 4., 8.],
        lambda a, y: (100.*math.sin(a), y, 100.*(1.-math.cos(a))), set())])
    # All neighboring panel normals differ only 5 degrees, but a non-neighbor
    # panel cannot enter by chaining: every seed uses its original fixed slab.
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    assert result["proposals"]
    panel_ids = np.tile(np.repeat(np.arange(8), 2), 4)
    for region in result["regions"]:
        assert len(set(panel_ids[region["triangle_rows"]])) == 1
        assert len(region["triangle_rows"]) < len(triangles)


def test_warped_grid_cannot_publish_the_whole_grid_as_one_qualified_support(module):
    grid = [0., 4., 8., 12., 16.]
    points, triangles, _ = _faces([(grid, grid,
        lambda x, y: (x, y, .08*(-1.)**(int(x/4)+int(y/4))), set())])
    result = _run(module, points, triangles)
    _complete(result, points, triangles)
    assert not any(r["analysis"]["status"] == "PROPOSED" and len(r["triangle_rows"]) == len(triangles)
                   for r in result["regions"])
    # These adversarial observations are NOT stipulated to arise from one
    # latent plane within epsilon; only each retained proposal is qualified.


def _square():
    return (np.array([[0., 0., 0.], [16., 0., 0.], [16., 16., 0.], [0., 16., 0.]]),
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))


def test_exact_small_counter_boundary_includes_duplicate_seed_work(module):
    points, triangles = _square()
    budget = module.SeededPlaneBudgetsV1(max_seeds=2, max_seed_selection_triangle_visits=4,
        max_proposal_raw_row_visits=8, max_proposal_triangle_visits=4, max_proposal_edge_visits=2,
        max_full_support_row_tests=4, max_triangle_memberships=2, max_raw_memberships=4,
        max_fit_evaluations=1, max_retained_array_bytes=48, max_working_array_bytes=1328)
    result = _run(module, points, triangles, budgets=budget)
    _complete(result, points, triangles)
    assert result["statistics"]["working_array_reservation_bytes"] == 1328
    assert [r["disposition"] for r in result["seeds"]] == ["PROPOSED", "EXACT_SUPPORT_DUPLICATE"]


@pytest.mark.parametrize("field,cap", [
    ("max_seed_selection_triangle_visits", 3), ("max_proposal_raw_row_visits", 7),
    ("max_proposal_triangle_visits", 3), ("max_proposal_edge_visits", 1),
    ("max_full_support_row_tests", 3), ("max_triangle_memberships", 1),
    ("max_raw_memberships", 3), ("max_fit_evaluations", 0),
    ("max_retained_array_bytes", 47), ("max_metadata_bytes", 0), ("max_working_array_bytes", 1327)])
def test_one_under_independently_counted_minimum_clears_every_prefix(module, field, cap):
    points, triangles = _square()
    budget = module.SeededPlaneBudgetsV1(max_seeds=2, max_retained_array_bytes=48)
    result = _run(module, points, triangles, budgets=replace(budget, **{field: cap}))
    _failed(result)


@pytest.mark.parametrize("where", ["region", "fit", "finalize"])
def test_late_failure_retains_attempted_work_but_no_proposal_prefix(module, monkeypatch, where):
    points, triangles = _square()
    if where == "fit":
        points = np.vstack((points, points+[0., 0., 3.]))
        triangles = np.vstack((triangles, triangles+4))
    owner, name = ((module, "_region") if where == "region" else
                   (module._v5, "_fit") if where == "fit" else (module._codec, "_finalize"))
    original = getattr(owner, name)
    calls = []
    def injected(*args, **kwargs):
        calls.append(1)
        if where == "finalize" or len(calls) == 2:
            raise RuntimeError("INVENTED_LATE_"+where)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, injected)
    result = _run(module, points, triangles)
    _failed(result)
    assert result["statistics"]["discarded_region_records"] >= 1
    assert result["statistics"]["discarded_proposals"] >= 1
    assert result["statistics"]["full_support_row_tests"] >= 4
    assert result["scheduled_seed_triangle_rows"]
    assert "INVENTED_LATE_"+where in result["reasons"][0]


def test_schedule_is_frozen_before_any_fit_and_not_refilled_after_rejection(module, monkeypatch):
    points, triangles, _ = _welded()
    ordinary = _run(module, points, triangles)
    _complete(ordinary, points, triangles)
    original = module._v5._analyze_support
    def reject(*args, **kwargs):
        answer = original(*args, **kwargs)
        answer.update(status="REJECTED", reasons=["MOCK_REJECTION_TO_TEST_SCHEDULING_ONLY"])
        return answer
    monkeypatch.setattr(module._v5, "_analyze_support", reject)
    rejected = _run(module, points, triangles)
    _complete(rejected, points, triangles)
    assert rejected["proposals"] == []
    assert ordinary["scheduled_seed_triangle_rows"] == rejected["scheduled_seed_triangle_rows"]
    assert [tuple(r["triangle_rows"]) for r in ordinary["regions"]] == [tuple(r["triangle_rows"]) for r in rejected["regions"]]


def test_zero_seeds_is_complete_no_seed_evaluation_not_surface_absence(module):
    points, triangles = _square()
    result = _run(module, points, triangles, budgets=module.SeededPlaneBudgetsV1(max_seeds=0))
    _complete(result, points, triangles)
    assert result["scheduled_seed_triangle_rows"] == result["proposals"] == []
    assert result["unscheduled_seed_triangle_count"] == len(triangles)


def test_runtime_uses_owned_snapshot_if_caller_arrays_change_after_admission(module, monkeypatch):
    points, triangles = _square()
    expected_p, expected_t = points.copy(), triangles.copy()
    original = module._normals
    def mutate(p, t, chunk):
        assert p is not points and t is not triangles
        assert not p.flags.writeable and not t.flags.writeable
        points[:] = 999.
        triangles[:] = -1
        return original(p, t, chunk)
    monkeypatch.setattr(module, "_normals", mutate)
    result = _run(module, points, triangles)
    _complete(result, expected_p, expected_t)


def test_mutation_during_first_snapshot_cannot_introduce_negative_index_alias(module, monkeypatch):
    points, triangles = _square()
    original = module._v5._owned
    first = []
    def mutate(value, dtype):
        answer = original(value, dtype)
        if not first:
            first.append(True)
            triangles[0, 0] = -1
        return answer
    monkeypatch.setattr(module._v5, "_owned", mutate)
    result = _run(module, points, triangles)
    _failed(result)
    assert "OWNED_RAW_INDEX_OUT_OF_RANGE" in result["reasons"][0]


@pytest.mark.parametrize("fault", ["point_list", "triangle_list", "point_bool", "triangle_float",
    "point_shape", "triangle_shape", "nan", "large", "negative", "out_of_range", "repeated_vertex",
    "collinear", "nonmanifold", "epsilon_bool", "epsilon_negative", "epsilon_nan", "epsilon_shape"])
def test_strict_array_topology_and_error_admission_never_returns_partial_support(module, fault):
    points, triangles = _square()
    epsilon = .05
    if fault == "point_list": points = points.tolist()
    elif fault == "triangle_list": triangles = triangles.tolist()
    elif fault == "point_bool": points = points.astype(bool)
    elif fault == "triangle_float": triangles = triangles.astype(float)
    elif fault == "point_shape": points = points.T
    elif fault == "triangle_shape": triangles = triangles.T
    elif fault == "nan": points[0, 0] = np.nan
    elif fault == "large": points[0, 0] = 1e6+1.
    elif fault == "negative": triangles[0, 0] = -1
    elif fault == "out_of_range": triangles[0, 0] = len(points)
    elif fault == "repeated_vertex": triangles[0, 1] = triangles[0, 0]
    elif fault == "collinear": points[:, 1] = 0.
    elif fault == "nonmanifold":
        points = np.vstack((points, [8., -8., 0.]))
        triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    elif fault == "epsilon_bool": epsilon = True
    elif fault == "epsilon_negative": epsilon = -.01
    elif fault == "epsilon_nan": epsilon = np.nan
    elif fault == "epsilon_shape": epsilon = np.ones((len(points), 1))*.05
    _failed(_run(module, points, triangles, epsilon))


@pytest.mark.parametrize("value", [True, -1, 1.5, 33])
def test_seed_budget_rejects_noninteger_negative_and_above_literal_maximum(module, value):
    points, triangles = _square()
    _failed(_run(module, points, triangles, budgets=module.SeededPlaneBudgetsV1(max_seeds=value)))


def test_mutated_public_default_cannot_increase_source_maximum(module):
    points, triangles = _square()
    default = module.extract_seeded_plane_proposals_v1.__kwdefaults__["budgets"]
    saved = default.max_seeds
    try:
        object.__setattr__(default, "max_seeds", 33)
        _failed(_run(module, points, triangles))
    finally:
        object.__setattr__(default, "max_seeds", saved)


@pytest.mark.parametrize("fault", ["policy_type", "budget_type", "angle_bool", "angle_zero", "angle_large", "tile_change"])
def test_unchanged_whole_support_policy_is_admitted_strictly(module, fault):
    points, triangles = _square()
    policy, budgets = module._v5.NoisyMeshPlanePolicyV5(), module.SeededPlaneBudgetsV1()
    if fault == "policy_type": policy = {}
    elif fault == "budget_type": budgets = {}
    elif fault == "angle_bool": policy = replace(policy, max_normal_angle_radians=True)
    elif fault == "angle_zero": policy = replace(policy, max_normal_angle_radians=0.)
    elif fault == "angle_large": policy = replace(policy, max_normal_angle_radians=1.)
    elif fault == "tile_change": policy = replace(policy, tile_triangles=(64, 1024))
    _failed(_run(module, points, triangles, qualification_policy=policy, budgets=budgets))
