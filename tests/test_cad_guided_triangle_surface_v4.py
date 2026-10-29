"""Independent finite-search oracle matrix for the V4 surface successor.

All V3 oracle cases are retained (V4 must satisfy the same exact enclosure
properties); new cases cover the sound vertex pruning bound, 256 tie slots,
wide-interval competitors becoming per-row uncertainty, and generic-soup
equality with the frozen V3 primitive.

Pure invented geometry only. Exact KKT oracle comes from the separately frozen
helper tests; no production minimizer/interval code is reused as truth.
"""
from dataclasses import fields, replace
from fractions import Fraction
from itertools import permutations
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def surface():
    package_name = "isolated_surface_v4_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[package_name] = package
    try:
        yield _load(package_name+".cad_guided_triangle_surface_v4",
                    ROOT/"backend/cad_guided_triangle_surface_v4.py")
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name+"."):
                sys.modules.pop(name, None)


@pytest.fixture
def oracle():
    name = "retained_triangle_bounds_v3_exact_oracle"
    try:
        yield _load(name, ROOT/"tests/test_cad_guided_triangle_distance_bounds_v3.py")
    finally:
        sys.modules.pop(name, None)


def _fraction(value):
    return Fraction(*float(value).as_integer_ratio())


def _triangle(z=0., scale=1.):
    return np.array([[0., 0., z], [scale, 0., z], [0., scale, z]], dtype=np.float64)


def _build(surface, triangles, identities=None, **kwargs):
    if identities is None:
        identities = list(range(len(triangles)))
    result = surface.build_triangle_surface(triangles, identities, **kwargs)
    assert result["status"] == "COMPLETE", result
    return result["index"]


def _cleared(surface, result):
    assert result["status"] == "UNAVAILABLE", result
    assert all(result[key] is None for key in surface.RESULT_ARRAY_KEYS)


def _check_exact_soup(result, points, triangles, identities, oracle, *, scale=1.):
    assert result["status"] == "COMPLETE", result
    assert result["closest_points"].shape == (len(points), 3)
    selected_ids = result["triangle_indices"]
    assert isinstance(selected_ids, np.ndarray) and selected_ids.dtype.kind in "iu"
    assert selected_ids.shape == (len(points),)
    assert np.all(selected_ids >= 0) and np.all(selected_ids < len(triangles))
    for row, point in enumerate(points):
        exact = [oracle._exact_projection(point, triangle) for triangle in triangles]
        optimum = min(item[1] for item in exact)
        lo = _fraction(result["distance_lower"][row])
        hi = _fraction(result["distance_upper"][row])
        nominal = _fraction(result["distances"][row])
        assert 0 <= lo <= hi and lo*lo <= optimum <= hi*hi
        # Accuracy is independently bounded by LOCAL scale, not returned guard.
        error = _fraction(math.ldexp(float(scale), -32))
        assert max(Fraction(0), nominal-error)**2 <= optimum <= (nominal+error)**2
        chosen = int(result["triangle_indices"][row])
        assert result["source_face_ids"][row] == identities[chosen]
        # Exact-point error refers to the selected triangle's true optimizer,
        # separate from whether another triangle is indistinguishably nearer.
        restored = tuple(_fraction(value) for value in result["closest_points"][row])
        exact_point = exact[chosen][0]
        squared_error = sum((a-b)**2 for a, b in zip(restored, exact_point))
        radius = _fraction(result["closest_point_error"][row])
        assert squared_error <= radius*radius
        assert hi-lo <= error
    return result


@pytest.mark.parametrize("shape", ("axis", "oblique"))
@pytest.mark.parametrize("power", (-10, 0, 10))
@pytest.mark.parametrize("translated", (False, True))
def test_locally_accurate_finite_projection_survives_large_world_origins(surface, oracle, shape, power, translated):
    tri = np.array([[0., 0., 0.], [4., 0., 0.], [0., 2., 0.]]) if shape == "axis" else np.array(
        [[0., 0., 0.], [2., 1., -2.], [2., -2., 1.]])
    # Seven independent barycentric/normal/outside query locations, not mesh vertices.
    normal = np.array([0., 0., 1.]) if shape == "axis" else np.array([1., 2., 2.])
    pairs = np.array([[.25, .25], [.5, 0.], [0., .5], [.75, .75], [-.5, -.5], [2., -.5], [-.5, 2.]])
    points = pairs[:, :1]*tri[1]+pairs[:, 1:]*tri[2]+.25*normal
    scale = math.ldexp(1., power)
    offset = np.array([2.**30, -2.**30, 2.**30]) if translated else np.zeros(3)
    triangles, points = ((tri+offset)*scale)[None], (points+offset)*scale
    result = surface.query_closest_surface(_build(surface, triangles, ["finite"]), points)
    _check_exact_soup(result, points, triangles, ["finite"], oracle, scale=scale)
    assert result["near_tie_counts"].tolist() == [1]*len(points)
    assert not np.any(result["normal_ambiguous"] | result["closest_point_ambiguous"])
    assert result["closest_regions"].tolist() == ["FACE","EDGE","EDGE","EDGE","VERTEX","VERTEX","VERTEX"]


@pytest.mark.parametrize("power", (-10, 0, 10))
@pytest.mark.parametrize("reverse", (False, True))
def test_remote_small_and_large_triangles_do_not_poison_local_surface_queries(surface, oracle, power, reverse):
    scale = math.ldexp(1., power)
    triangles = [_triangle(scale=4.)]
    # Sixteen represented, valid remote triangles span10orders of edge scale.
    # Neither coordinate/edge extrema nor a broad leaf may destroy local support.
    for index in range(16):
        size = math.ldexp(1., (-16, -8, 8, 16)[index % 4])
        triangles.append(_triangle(scale=size)+np.array([2.**22+index*2.**18, 2.**22, 2.**22]))
    triangles = np.array(triangles)*scale
    identities = tuple("primitive:"+str(index) for index in range(len(triangles)))
    if reverse:
        triangles, identities = triangles[::-1].copy(), identities[::-1]
    points = np.array([[.5, .5, .25], [1., .5, 1.], [3., 3., .5], [-1., -1., .25]])*scale
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    _check_exact_soup(answer, points, triangles, identities, oracle, scale=scale)
    assert answer["source_face_ids"] == ("primitive:0",)*len(points)
    assert answer["near_tie_counts"].tolist() == [1]*len(points)


def test_hole_distance_and_multiple_closest_points_are_preserved(surface, oracle):
    triangles, identities = [], []
    for face, (x0, x1, y0, y1) in enumerate(((-2.,2.,-2.,-1.),(-2.,2.,1.,2.),
                                           (-2.,-1.,-1.,1.),(1.,2.,-1.,1.))):
        a,b,c,d = [x0,y0,0.],[x1,y0,0.],[x1,y1,0.],[x0,y1,0.]
        triangles.extend([[a,b,c],[a,c,d]])
        identities.extend([face,face])
    triangles, points = np.array(triangles), np.array([[0.,0.,.25]])
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["distances"][0] > 1.
    assert answer["closest_point_ambiguous"][0] and answer["face_identity_ambiguous"][0]
    assert not answer["normal_ambiguous"][0]
    assert answer["closest_regions"][0] == "EDGE"


@pytest.mark.parametrize("mode", ("same_face", "different_face", "opposite", "parallel"))
def test_geometric_ties_keep_point_normal_and_identity_evidence_separate(surface, oracle, mode):
    tri = _triangle(scale=2.)
    triangles = np.array([tri, tri.copy()])
    identities = [0, int(mode == "different_face")]
    if mode == "opposite":
        triangles[1] = tri[[0,2,1]]
    if mode == "parallel":
        triangles[0,:,2], triangles[1,:,2] = 1., -1.
    points = np.array([[.25,.25,0.]])
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["near_tie_counts"].tolist() == [2]
    assert bool(answer["normal_ambiguous"][0]) == (mode == "opposite")
    assert bool(answer["closest_point_ambiguous"][0]) == (mode == "parallel")
    assert bool(answer["face_identity_ambiguous"][0]) == (mode == "different_face")


@pytest.mark.parametrize("reverse", (False, True))
def test_tie_band_uses_final_nearest_upper_bound_not_stale_earlier_winner(surface, oracle, reverse):
    triangles = np.array([_triangle(z=1.), _triangle(z=-1.), _triangle(z=.125)])
    identities = [0,1,2]
    if reverse:
        triangles, identities = triangles[::-1].copy(), identities[::-1]
    points = np.array([[.25,.25,0.]])
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points, near_tie_distance=.03125)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["source_face_ids"] == (2,) and answer["near_tie_counts"].tolist() == [1]
    assert not any(answer[key][0] for key in ("normal_ambiguous","closest_point_ambiguous","face_identity_ambiguous"))


@pytest.mark.parametrize("reverse", (False, True))
def test_exact_tie_distance_boundary_is_inclusive(surface, oracle, reverse):
    triangles = np.array([_triangle(z=1.), _triangle(z=1.+2.**-5)])
    identities = [0,1]
    if reverse:
        triangles, identities = triangles[::-1].copy(), identities[::-1]
    points = np.array([[.25,.25,0.]])
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points, near_tie_distance=2.**-5)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["source_face_ids"] == (0,)
    assert answer["near_tie_counts"].tolist() == [2]
    assert answer["closest_point_ambiguous"].tolist() == [True]


def test_proven_irrelevant_first_pass_lower_bound_does_not_poison_final_enclosure(surface, oracle, monkeypatch):
    # V4: the remote primitive sits at z=.3, inside the nearest-vertex pruning
    # bound (.375 to (0,0,.125)), so pass 1 still evaluates it first with
    # max_pair_batch=1; V3 used z=1, which V4 now prunes before any certificate.
    triangles = np.array([_triangle(z=.3), _triangle(z=.125)])
    points = np.array([[.25,.25,0.]])
    original = surface.enclose_triangle_distances
    weakened_pairs = []
    def weaken_sound_lower(point_rows, triangle_rows, coefficients, **kwargs):
        result = original(point_rows, triangle_rows, coefficients, **kwargs)
        assert result["status"] == "COMPLETE", result
        remote = np.all(triangle_rows[:,:,2] == .3, axis=1)
        # Zero remains a sound distance lower bound. The upper witness and all
        # other exact-helper outputs are unchanged. AABB proves this primitive
        # irrelevant once the later near primitive establishes the final U*.
        result["distance_lower"][remote] = 0.
        weakened_pairs.extend(np.flatnonzero(remote).tolist())
        return result
    monkeypatch.setattr(surface, "enclose_triangle_distances", weaken_sound_lower)
    budget = replace(surface.TriangleSurfaceBudgetsV4(), max_pair_batch=1)
    answer = surface.query_closest_surface(_build(surface, triangles), points, budgets=budget)
    _check_exact_soup(answer, points, triangles, [0,1], oracle)
    assert len(weakened_pairs) == 4  # Four witnesses in pass1, none after proof-pruning.
    assert answer["triangle_indices"].tolist() == [1]
    assert answer["near_tie_counts"].tolist() == [1]
    assert answer["diagnostics"]["proof_pruned_triangle_pairs"] >= 1


def test_outside_infinite_plane_loses_to_a_genuinely_nearer_finite_triangle(surface, oracle):
    triangles = np.array([_triangle(), [[2.,2.,.25],[4.,2.,.25],[2.,4.,.25]]])
    points = np.array([[2.25,2.25,0.]])
    answer = surface.query_closest_surface(_build(surface, triangles), points)
    _check_exact_soup(answer, points, triangles, [0,1], oracle)
    assert answer["triangle_indices"].tolist() == [1]


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_all_near_normal_pairs_are_compared_not_only_against_selected_normal(surface, order):
    triangles = []
    for degrees in (0., 4., -4.):
        theta = math.radians(degrees)
        triangles.append([[0.,0.,0.],[1.,0.,0.],[0.,math.cos(theta),math.sin(theta)]])
    triangles = np.array(triangles)[list(order)]
    answer = surface.query_closest_surface(_build(surface, triangles), [[0.,0.,0.]])
    assert answer["status"] == "COMPLETE", answer
    assert answer["near_tie_counts"].tolist() == [3]
    assert answer["normal_ambiguous"].tolist() == [True]
    assert answer["diagnostics"]["normal_pairs"] == 3


@pytest.mark.parametrize("reverse", (False, True))
def test_equidistant_competitors_with_different_scales_are_both_retained(surface, oracle, reverse):
    triangles = np.array([_triangle(z=.5,scale=2.**-8), _triangle(z=-.5,scale=32.)])
    identities = [0,1]
    if reverse:
        triangles, identities = triangles[::-1].copy(), identities[::-1]
    points = np.array([[2.**-10,2.**-10,0.]])
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["near_tie_counts"].tolist() == [2]
    assert answer["closest_point_ambiguous"].tolist() == [True]


def test_long_triangle_is_not_pruned_by_its_distant_centroid(surface, oracle):
    triangles = np.array([_triangle(scale=2.**20),
                          *[_triangle(z=2.+index/16.)+np.array([index%4,index//4,0.])
                            for index in range(16)]])
    points = np.array([[.125,.125,.25]])
    answer = surface.query_closest_surface(_build(surface, triangles), points)
    # This is a2^20-unit triangle, not a unit-scale precision fixture. Certifying
    # the support function can amplify interval width by its own long edges.
    # Pin a separate1-micro-unit enclosure width and sub-nanounit nominal error
    # prospectively, while retaining exact containment and selected primitive.
    _check_exact_soup(answer, points, triangles, list(range(len(triangles))), oracle, scale=2.**20)
    assert _fraction(answer["distance_upper"][0])-_fraction(answer["distance_lower"][0]) <= Fraction(1,2**20)
    assert abs(_fraction(answer["distances"][0])-Fraction(1,4)) <= Fraction(1,2**32)
    assert answer["triangle_indices"].tolist() == [0]


@pytest.mark.parametrize("bad", (
    np.zeros((3,3)), [[0.,0.,0.],[1.,0.,0.],[0.,2.**-40,0.]],
    [[0.,0.,0.],[1.,0.,0.],[0.,float("nan"),0.]],
))
def test_one_unsupported_triangle_never_creates_a_silent_hole(surface, bad):
    answer = surface.build_triangle_surface(np.array([_triangle(), bad]), [0,1])
    assert answer["status"] == "UNAVAILABLE" and answer["index"] is None


@pytest.mark.parametrize("field", ("max_query_rows","max_node_visits","max_triangle_pairs",
                                  "max_certificate_pairs","max_normal_pairs","max_work_units"))
def test_bounded_search_clears_every_array_on_incomplete_work(surface, field):
    index = _build(surface, np.array([_triangle(),_triangle()]))
    budget = replace(surface.TriangleSurfaceBudgetsV4(), **{field:0})
    answer = surface.query_closest_surface(index, [[.25,.25,.5]], budgets=budget)
    _cleared(surface, answer)


@pytest.mark.parametrize("certificate_cap", (7,8))
def test_exact_two_pass_certificate_work_boundary(surface, certificate_cap):
    index = _build(surface, np.array([_triangle()]))
    budget = replace(surface.TriangleSurfaceBudgetsV4(), max_pair_batch=1,
        max_triangle_pairs=2, max_certificate_pairs=certificate_cap,
        max_node_visits=2, max_work_units=28+16)
    answer = surface.query_closest_surface(index, [[.25,.25,.5]], budgets=budget)
    if certificate_cap == 7:
        _cleared(surface, answer)
        assert answer["reason"] == "QUERY_BUDGET_EXCEEDED_CERTIFICATE_PAIRS"
    else:
        assert answer["status"] == "COMPLETE", answer
        assert answer["diagnostics"]["node_visits"] == 2
        assert answer["diagnostics"]["triangle_box_pairs"] == 2
        assert answer["diagnostics"]["triangle_pairs"] == 2
        assert answer["diagnostics"]["certificate_pairs"] == 8
        assert answer["diagnostics"]["vertex_lookup_rows"] == 1
        assert answer["diagnostics"]["work_units"] == 44


def test_vertex_lookup_work_is_charged_before_any_search(surface):
    index = _build(surface, np.array([_triangle()]))
    budget = replace(surface.TriangleSurfaceBudgetsV4(), max_work_units=15)
    answer = surface.query_closest_surface(index, [[.25,.25,.5]], budgets=budget)
    _cleared(surface, answer)
    assert answer["reason"] == "QUERY_BUDGET_EXCEEDED_VERTEX_LOOKUP_WORK"
    assert answer["diagnostics"]["vertex_lookup_rows"] == 0 and answer["diagnostics"]["node_visits"] == 0


def test_late_tie_overflow_clears_an_earlier_unique_query(surface):
    index = _build(surface, np.array([_triangle(1.),_triangle(-1.)]))
    budget = replace(surface.TriangleSurfaceBudgetsV4(), max_pair_batch=1, max_ties_per_query=1)
    result = surface.query_closest_surface(index, [[.25,.25,1.],[.25,.25,0.]], budgets=budget)
    _cleared(surface, result)


def test_index_snapshots_geometry_and_all_array_bases_are_immutable(surface):
    triangles, identities = np.array([_triangle()]), ["original"]
    index = _build(surface, triangles, identities)
    triangles[:] = 42.
    identities[0] = "changed"
    for field in fields(index):
        value = getattr(index, field.name)
        while isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)
            value = value.base
    result = surface.query_closest_surface(index, [[.25,.25,1.]])
    assert result["status"] == "COMPLETE" and result["source_face_ids"] == ("original",)


def test_empty_query_has_no_geometry_work(surface):
    answer = surface.query_closest_surface(_build(surface, np.array([_triangle()])), np.empty((0,3)))
    assert answer["status"] == "COMPLETE"
    assert answer["closest_points"].shape == (0,3)
    assert answer["distance_lower"].shape == answer["distance_upper"].shape == (0,)
    assert answer["diagnostics"]["work_units"] == 0
    assert answer["diagnostics"]["vertex_lookup_rows"] == 0


@pytest.fixture
def frozen_v3():
    package_name = "isolated_surface_v3_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT/"backend")]
    sys.modules[package_name] = package
    try:
        yield _load(package_name+".cad_guided_triangle_surface_v3",
                    ROOT/"backend/cad_guided_triangle_surface_v3.py")
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name+"."):
                sys.modules.pop(name, None)


def _fan(count, radius=1., z=0.):
    """A disc fan: `count` triangles sharing the origin vertex, all normals +z."""
    triangles = []
    for k in range(count):
        a, b = 2.*math.pi*k/count, 2.*math.pi*(k+1)/count
        triangles.append([[0., 0., z], [radius*math.cos(a), radius*math.sin(a), z],
                          [radius*math.cos(b), radius*math.sin(b), z]])
    return np.array(triangles)


@pytest.mark.parametrize("count", (33, 40, 96))
def test_fan_vertex_with_more_than_thirty_two_incident_triangles_completes(surface, frozen_v3, oracle, count):
    triangles = _fan(count)
    identities = list(range(count))
    points = np.array([[0., 0., .25], [0., 0., -.125]])
    # The frozen V3 primitive refuses this fixture at its 32-slot storage limit.
    previous = frozen_v3.query_closest_surface(frozen_v3.build_triangle_surface(triangles, identities)["index"], points)
    assert previous["status"] == "UNAVAILABLE" and previous["reason"] == "NEAR_TIE_BUDGET_EXCEEDED"
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    _check_exact_soup(answer, points, triangles, identities, oracle)
    assert answer["near_tie_counts"].tolist() == [count, count]
    assert not np.any(answer["normal_ambiguous"]) and not np.any(answer["normal_ambiguity_uncertain"])
    assert np.all(answer["face_identity_ambiguous"])
    assert answer["closest_regions"].tolist() == ["VERTEX", "VERTEX"]
    assert answer["diagnostics"]["normal_pairs"] == 2*count*(count-1)//2


def test_tie_storage_maximum_is_two_hundred_fifty_six(surface):
    assert surface.MAX_TIES_PER_QUERY == 256
    assert surface.TriangleSurfaceBudgetsV4().max_ties_per_query == 256
    with pytest.raises(surface._Refusal):
        surface._budgets(replace(surface.TriangleSurfaceBudgetsV4(), max_ties_per_query=257))
    index = _build(surface, _fan(300))
    answer = surface.query_closest_surface(index, [[0., 0., .25]])
    _cleared(surface, answer)
    assert answer["reason"] == "NEAR_TIE_BUDGET_EXCEEDED"


def _needle():
    """A represented needle whose competitive certificate is wider than 1e-5 of its shortest edge."""
    return np.array([[[0., 0., 0.], [100., 0., 0.], [100.+1e-5, 0., 1e-5]],
                     [[0., 0., 0.], [100., 0., 0.], [0., 5., 0.]]])


def test_wide_competitive_interval_marks_rows_uncertain_instead_of_refusing(surface, frozen_v3, oracle):
    triangles, identities = _needle(), [7, 8]
    points = np.array([[100., 2e-6, 4e-6], [50., 1., .5]])
    previous = frozen_v3.query_closest_surface(frozen_v3.build_triangle_surface(triangles, identities)["index"], points)
    assert previous["status"] == "UNAVAILABLE"
    assert previous["reason"] == "UNSUPPORTED_COMPETITIVE_TRIANGLE_INTERVAL_WIDTH"
    answer = surface.query_closest_surface(_build(surface, triangles, identities), points)
    assert answer["status"] == "COMPLETE", answer
    # The needle's certificate is legitimately wide, so only exact enclosure
    # (never the well-conditioned accuracy bound) is asserted here.
    for row, point in enumerate(points):
        optimum = min(oracle._exact_projection(point, triangle)[1] for triangle in triangles)
        lo, hi = _fraction(answer["distance_lower"][row]), _fraction(answer["distance_upper"][row])
        assert 0 <= lo <= hi and lo*lo <= optimum <= hi*hi
    assert answer["diagnostics"]["interval_width_uncertain_pairs"] >= 1
    assert bool(answer["normal_ambiguity_uncertain"][0]) and bool(answer["closest_point_ambiguity_uncertain"][0])
    assert not answer["normal_ambiguity_uncertain"][1] and not answer["closest_point_ambiguity_uncertain"][1]
    assert answer["interval_width_policy"] == "WIDE_COMPETITIVE_INTERVAL_MARKS_ROW_UNCERTAIN_NOT_WHOLE_QUERY_REFUSAL"


def _soup(count, seed):
    """Deterministic invented triangle soup with generic (tie-free) positions."""
    state = np.uint64(seed)
    values = []
    for _ in range(count*9):
        state = (state*np.uint64(6364136223846793005)+np.uint64(1442695040888963407)) & np.uint64(2**64-1)
        values.append(float(state >> np.uint64(11))/float(2**53))
    base = np.array(values).reshape(count, 3, 3)*np.array([40., 30., 6.])
    base[:, 1:, :] = base[:, :1, :]+(base[:, 1:, :]-base[:, :1, :])*.05
    return base


def test_generic_soup_result_bytes_equal_frozen_v3_on_this_fixture(surface, frozen_v3):
    """Byte equality of every result array on ONE generic tie-free fixture.

    This is fixture-specific evidence, not a general bitwise-V3 claim: V4's
    vertex-threshold pruning may select differently among ulp-level near-ties.
    Timings are printed as consumed development observations, not a gate.
    """
    import time
    triangles = _soup(400, 12345)
    identities = list(range(len(triangles)))
    grid = np.array([[x, y, z] for x in np.linspace(-2., 42., 9) for y in np.linspace(-2., 32., 7) for z in (-1., 3., 7.)])
    started = time.process_time()
    previous_index = frozen_v3.build_triangle_surface(triangles, identities)["index"]
    previous = frozen_v3.query_closest_surface(previous_index, grid)
    v3_cpu = time.process_time()-started
    started = time.process_time()
    answer = surface.query_closest_surface(_build(surface, triangles, identities), grid)
    v4_cpu = time.process_time()-started
    assert previous["status"] == answer["status"] == "COMPLETE", (previous.get("reason"), answer.get("reason"))
    for key in surface.RESULT_ARRAY_KEYS:
        if key == "source_face_ids":
            assert previous[key] == answer[key]
        else:
            assert previous[key].dtype == answer[key].dtype and previous[key].shape == answer[key].shape, key
            assert previous[key].tobytes(order="C") == answer[key].tobytes(order="C"), key
    old, new = previous["diagnostics"], answer["diagnostics"]
    assert new["triangle_pairs"] < old["triangle_pairs"] and new["certificate_pairs"] < old["certificate_pairs"]
    assert new["vertex_lookup_rows"] == len(grid)
    print("SURFACE_V4_VS_V3_WORK_JSON=" + str(dict(rows=len(grid), triangles=len(triangles),
        v3_triangle_pairs=old["triangle_pairs"], v4_triangle_pairs=new["triangle_pairs"],
        v3_certificate_pairs=old["certificate_pairs"], v4_certificate_pairs=new["certificate_pairs"],
        v3_work_units=old["work_units"], v4_work_units=new["work_units"],
        v3_build_query_cpu=round(v3_cpu, 4), v4_build_query_cpu=round(v4_cpu, 4))).replace("'", '"'))


def test_vertex_bound_is_a_sound_upper_bound_of_exact_distance(surface):
    points = _soup(64, 777)[:, 0, :]*np.array([1., 1., 1e3])+np.array([2.**30, -2.**20, 0.])
    vertices = _soup(64, 999)[:, 1, :]*np.array([1e-3, 1., 1e2])+np.array([2.**30, -2.**20, 0.])
    upper = surface._vertex_distance_upper(points, vertices)
    for p, v, u in zip(points, vertices, upper):
        exact = sum((Fraction(*float(a).as_integer_ratio())-Fraction(*float(b).as_integer_ratio()))**2 for a, b in zip(p, v))
        assert Fraction(*float(u).as_integer_ratio())**2 >= exact


def test_index_exposes_deterministic_vertex_tree_over_distinct_vertices(surface):
    triangles = np.array([_triangle(), _triangle(z=0.)+np.array([1., 0., 0.]), _triangle(z=2.)])
    index = _build(surface, triangles)
    assert index.vertex_points.shape == (8, 3)  # (1,0,0) is shared by the first two
    assert index.vertex_points.flags.writeable is False
    distances, nearest = index.vertex_tree.query(np.array([[1.01, 0., 0.]]), k=1, workers=1)
    assert np.array_equal(index.vertex_points[int(nearest[0])], [1., 0., 0.])
    again = _build(surface, triangles)
    np.testing.assert_array_equal(index.vertex_points, again.vertex_points)
