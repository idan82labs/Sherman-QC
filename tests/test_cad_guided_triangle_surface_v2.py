"""Invented finite-surface contracts; import-safe NumPy module, no native/data I/O."""

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_triangle_surface_v2.py"


@pytest.fixture
def surface():
    name = "isolated_cad_guided_triangle_surface_v2"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def triangle(z=0., scale=1.):
    return np.array([[0., 0., z], [scale, 0., z], [0., scale, z]])


def built(surface, triangles=None, ids=None, **kwargs):
    if triangles is None:
        triangles = np.array([triangle()])
    if ids is None:
        ids = list(range(len(triangles)))
    result = surface.build_triangle_surface(triangles, ids, **kwargs)
    assert result["status"] == "COMPLETE", result
    return result["index"]


def assert_cleared(surface, result):
    assert result["status"] == "UNAVAILABLE"
    assert all(result[key] is None for key in surface.RESULT_ARRAY_KEYS)


def exact_closest(point, vertices):
    """Independent exact rational simplex minimization, including all edges."""
    p = tuple(Fraction(value) for value in point)
    a, b, c = [tuple(Fraction(value) for value in row) for row in vertices]

    def subtract(x, y):
        return tuple(i - j for i, j in zip(x, y))

    def dot(x, y):
        return sum(i * j for i, j in zip(x, y))

    e, f, d = subtract(b, a), subtract(c, a), subtract(p, a)
    aa, ab, bb = dot(e, e), dot(e, f), dot(f, f)
    determinant = aa * bb - ab * ab
    assert determinant > 0
    u = (dot(d, e) * bb - dot(d, f) * ab) / determinant
    v = (dot(d, f) * aa - dot(d, e) * ab) / determinant
    candidates = []
    if u >= 0 and v >= 0 and u + v <= 1:
        candidates.append(tuple(x + u * y + v * z for x, y, z in zip(a, e, f)))
    for start, stop in ((a, b), (b, c), (c, a)):
        edge = subtract(stop, start)
        parameter = min(Fraction(1), max(Fraction(0), dot(subtract(p, start), edge) / dot(edge, edge)))
        candidates.append(tuple(x + parameter * y for x, y in zip(start, edge)))
    distances = [dot(subtract(p, item), subtract(p, item)) for item in candidates]
    chosen = min(range(len(candidates)), key=lambda index: distances[index])
    return candidates[chosen], distances[chosen]


@pytest.mark.parametrize("query,expected,region", [
    ([.25, .25, 2.], [.25, .25, 0.], "FACE"),
    ([.8, .8, 0.], [.5, .5, 0.], "EDGE"),
    ([2., .2, 0.], [1., 0., 0.], "VERTEX"),
    ([.25, -.5, .3], [.25, 0., 0.], "EDGE"),
    ([-2., -3., 4.], [0., 0., 0.], "VERTEX"),
    ([0., 0., 0.], [0., 0., 0.], "VERTEX"),
    ([.25, 0., 0.], [.25, 0., 0.], "EDGE"),
])
def test_finite_face_edge_vertex_regions(surface, query, expected, region):
    result = surface.query_closest_surface(built(surface), [query])
    assert result["status"] == "COMPLETE", result
    np.testing.assert_allclose(result["closest_points"][0], expected, rtol=0, atol=2e-15)
    assert result["distances"][0] == pytest.approx(math.dist(query, expected), abs=2e-15)
    assert result["closest_regions"].tolist() == [region]
    np.testing.assert_array_equal(result["oriented_normals"], [[0., 0., 1.]])


def test_sparse_vertices_are_not_the_cad_surface(surface):
    mesh = np.array([triangle(scale=100.)])
    point = np.array([[30., 30., 2.]])
    result = surface.query_closest_surface(built(surface, mesh, ["face:wide"]), point)
    assert result["status"] == "COMPLETE"
    assert result["distances"].tolist() == [2.]
    assert min(np.linalg.norm(point[0] - vertex) for vertex in mesh[0]) > 40.
    np.testing.assert_array_equal(result["closest_points"], [[30., 30., 0.]])
    assert result["source_face_ids"] == ("face:wide",)


def test_finite_boundary_distance_is_not_facet_plane_residual(surface):
    point = np.array([[.8, .8, 0.]])
    result = surface.query_closest_surface(built(surface), point)
    residual = np.sum((point - result["closest_points"]) * result["oriented_normals"], axis=1)
    assert residual[0] == 0.
    assert result["distances"][0] > .4
    assert result["closest_regions"][0] == "EDGE"


def test_rectangular_hole_is_not_filled_by_infinite_planes(surface):
    rectangles = [(-2., 2., -2., -1.), (-2., 2., 1., 2.),
                  (-2., -1., -1., 1.), (1., 2., -1., 1.)]
    triangles = []
    faces = []
    for face, (x0, x1, y0, y1) in enumerate(rectangles):
        a, b, c, d = ([x0, y0, 0.], [x1, y0, 0.], [x1, y1, 0.], [x0, y1, 0.])
        triangles.extend([[a, b, c], [a, c, d]])
        faces.extend([face, face])
    result = surface.query_closest_surface(built(surface, triangles, faces), [[0., 0., .2]])
    assert result["status"] == "COMPLETE", result
    assert result["distances"][0] == pytest.approx(math.hypot(1., .2), abs=2e-15)
    assert result["closest_regions"][0] == "EDGE"
    assert result["closest_point_ambiguous"][0]
    assert result["face_identity_ambiguous"][0]
    assert not result["normal_ambiguous"][0]


def test_parallel_equidistant_sheets_have_point_not_normal_ambiguity(surface):
    index = built(surface, np.array([triangle(1.), triangle(-1.)]), ["upper", "lower"])
    result = surface.query_closest_surface(index, [[.2, .2, 0.]])
    assert result["status"] == "COMPLETE"
    assert result["triangle_indices"].tolist() == [0]
    assert result["near_tie_counts"].tolist() == [2]
    assert result["closest_point_ambiguous"].tolist() == [True]
    assert result["normal_ambiguous"].tolist() == [False]
    assert result["face_identity_ambiguous"].tolist() == [True]


def test_opposite_winding_remains_oriented_normal_ambiguity(surface):
    tri = triangle()
    result = surface.query_closest_surface(built(surface, [tri, tri[[0, 2, 1]]], [0, 0]), [[.2, .2, 1.]])
    assert result["status"] == "COMPLETE"
    assert result["normal_ambiguous"].tolist() == [True]
    assert result["closest_point_ambiguous"].tolist() == [False]
    assert result["face_identity_ambiguous"].tolist() == [False]


def test_all_near_normal_pairs_not_just_winner_are_compared(surface):
    triangles = []
    for degrees in (0., 4., -4.):
        theta = math.radians(degrees)
        triangles.append([[0., 0., 0.], [1., 0., 0.], [0., math.cos(theta), math.sin(theta)]])
    result = surface.query_closest_surface(built(surface, triangles), [[0., 0., 0.]])
    assert result["status"] == "COMPLETE"
    assert result["triangle_indices"].tolist() == [0]
    assert result["normal_ambiguous"].tolist() == [True]
    assert result["diagnostics"]["normal_pairs"] == 3


def test_later_closer_witness_removes_old_near_tie_ambiguity(surface):
    triangles = np.array([triangle(1.), triangle(-1.), triangle(.1)])
    result = surface.query_closest_surface(built(surface, triangles), [[.2, .2, 0.]], near_tie_distance=.01)
    assert result["status"] == "COMPLETE"
    assert result["triangle_indices"].tolist() == [2]
    assert result["near_tie_counts"].tolist() == [1]
    assert not any(result[name][0] for name in ("normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous"))


def test_near_tie_policy_is_explicit_and_does_not_replace_nearest(surface):
    triangles = np.array([triangle(1.01), triangle(1.)])
    result = surface.query_closest_surface(built(surface, triangles), [[.2, .2, 0.]], near_tie_distance=.02)
    assert result["status"] == "COMPLETE"
    assert result["triangle_indices"].tolist() == [1]
    assert result["distances"].tolist() == [1.]
    assert result["near_tie_counts"].tolist() == [2]


def test_bvh_matches_exact_rational_oracle_on_fixed_invented_matrix(surface):
    triangles = []
    for x in range(8):
        for y in range(4):
            triangles.append([[3*x, 3*y, (x+y) % 3], [3*x+2, 3*y, (x+y) % 3],
                              [3*x, 3*y+2, (x+y) % 3]])
    queries = [[Fraction(3*i-7, 8), Fraction((i*7) % 103 - 9, 8), Fraction(i % 17 - 3, 8)]
               for i in range(64)]
    result = surface.query_closest_surface(built(surface, triangles), np.array(queries, dtype=float))
    assert result["status"] == "COMPLETE", result
    for row, query in enumerate(queries):
        exact = [exact_closest(query, tri) for tri in triangles]
        selected = min(range(len(exact)), key=lambda index: (exact[index][1], index))
        minimizers = [index for index, item in enumerate(exact) if item[1] == exact[selected][1]]
        assert any(np.allclose(result["closest_points"][row], [float(v) for v in exact[index][0]],
                               rtol=0, atol=3e-14) for index in minimizers)
        assert result["distances"][row] == pytest.approx(math.sqrt(float(exact[selected][1])), abs=3e-14)
        if len(minimizers) == 1:
            assert result["triangle_indices"][row] == selected


@pytest.mark.parametrize("power", [-200, -20, 0, 20, 200])
def test_scale_covariance_and_direct_hypot_distance(surface, power):
    scale = math.ldexp(1., power)
    tri = triangle(scale=scale)
    point = np.array([[.25, .25, 2.]]) * scale
    result = surface.query_closest_surface(built(surface, [tri]), point)
    assert result["status"] == "COMPLETE", result
    np.testing.assert_allclose(result["closest_points"] / scale, [[.25, .25, 0.]], rtol=0, atol=2e-15)
    assert result["distances"][0] / scale == 2.


def test_rigid_transform_and_constant_coordinate_preservation(surface):
    rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    translation = np.array([1024., -512., 256.])
    tri = triangle() @ rotation.T + translation
    point = np.array([[.2, .3, 2.]]) @ rotation.T + translation
    result = surface.query_closest_surface(built(surface, [tri]), point)
    assert result["status"] == "COMPLETE"
    expected = np.array([[.2, .3, 0.]]) @ rotation.T + translation
    np.testing.assert_allclose(result["closest_points"], expected, rtol=0, atol=2e-13)
    assert result["closest_points"][0, 1] == -512.
    np.testing.assert_array_equal(result["oriented_normals"], [[0., 1., 0.]])


def test_index_geometry_and_base_views_cannot_be_mutated(surface):
    triangles = np.array([triangle()])
    ids = ["original"]
    index = built(surface, triangles, ids)
    triangles[:] = 90.
    ids[0] = "changed"
    for attribute in fields(index):
        value = getattr(index, attribute.name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)
            base = value.base
            while isinstance(base, np.ndarray):
                with pytest.raises(ValueError):
                    base.setflags(write=True)
                base = base.base
    with pytest.raises(FrozenInstanceError):
        index.minimum_edge = 0.
    result = surface.query_closest_surface(index, [[.2, .2, 1.]])
    assert result["source_face_ids"] == ("original",)
    assert result["distances"].tolist() == [1.]


@pytest.mark.parametrize("name,value", [
    ("max_build_nodes", 0), ("max_build_triangle_visits", 0), ("max_triangles", 0),
])
def test_build_budget_refuses_whole_index(surface, name, value):
    result = surface.build_triangle_surface([triangle()], [0], budgets=replace(surface.TriangleSurfaceBudgets(), **{name: value}))
    assert result["status"] == "UNAVAILABLE"
    assert result["index"] is None


@pytest.mark.parametrize("name,value", [
    ("max_node_visits", 1), ("max_triangle_pairs", 1), ("max_work_units", 17),
    ("max_query_rows", 0), ("max_triangles", 0),
])
def test_query_budget_clears_nearest_prefix(surface, name, value):
    budget = replace(surface.TriangleSurfaceBudgets(), **{name: value})
    result = surface.query_closest_surface(built(surface), [[.2, .2, 1.]], budgets=budget)
    assert_cleared(surface, result)


def test_exact_small_work_boundary_and_pair_batch_cap(surface):
    budget = replace(surface.TriangleSurfaceBudgets(), max_node_visits=2, max_triangle_pairs=2,
                     max_work_units=18, max_pair_batch=1)
    result = surface.query_closest_surface(built(surface), [[.2, .2, 1.]], budgets=budget)
    assert result["status"] == "COMPLETE", result
    assert result["diagnostics"]["work_units"] == 18
    assert result["diagnostics"]["maximum_pair_batch"] == 1


def test_late_tie_batch_overflow_clears_earlier_completed_rows(surface):
    index = built(surface, [triangle(1.), triangle(-1.)])
    budget = replace(surface.TriangleSurfaceBudgets(), max_pair_batch=1, max_ties_per_query=1)
    result = surface.query_closest_surface(index, [[.2, .2, 1.], [.2, .2, 0.]], budgets=budget)
    assert_cleared(surface, result)
    assert result["reason"] == "NEAR_TIE_BUDGET_EXCEEDED"
    assert result["diagnostics"]["tie_batches_completed"] == 1


def test_tie_normal_pair_budget_clears_all_results(surface):
    result = surface.query_closest_surface(built(surface, [triangle(1.), triangle(-1.)]), [[.2, .2, 0.]],
                                           budgets=replace(surface.TriangleSurfaceBudgets(), max_normal_pairs=0))
    assert_cleared(surface, result)


def test_tie_storage_is_batched_independently_of_query_count(surface):
    budget = replace(surface.TriangleSurfaceBudgets(), max_pair_batch=64, max_ties_per_query=32)
    result = surface.query_closest_surface(built(surface), np.tile([.2, .2, 1.], (17, 1)), budgets=budget)
    assert result["status"] == "COMPLETE"
    assert result["diagnostics"]["maximum_tie_storage_rows"] == 2
    assert result["diagnostics"]["tie_batches_completed"] == 9


@pytest.mark.parametrize("bad", [
    [[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]],
    [[[0., 0., 0.], [1., 0., 0.], [0., 1e-12, 0.]]],
    [[[0., 0., 0.], [1., 0., 0.], [0., np.nan, 0.]]],
    [[[0., 0., 0.], [1., 0., 0.], [0., np.inf, 0.]]],
    [[[0., 0., 0.], [True, 0., 0.], [0., 1., 0.]]],
    [[[0., 0., 0.], [2**53 + 1, 0., 0.], [0., 1., 0.]]],
    np.ones((1, 3, 3), dtype=complex), np.zeros((1, 2, 3)), [],
])
def test_invalid_triangle_rejects_entire_surface(surface, bad):
    result = surface.build_triangle_surface(bad, [0])
    assert result["status"] == "UNAVAILABLE"
    assert result["index"] is None


def test_late_invalid_triangle_never_creates_a_silent_hole(surface):
    triangles = np.array([triangle(), triangle(2.), np.zeros((3, 3))])
    result = surface.build_triangle_surface(triangles, [0, 1, 2])
    assert result["status"] == "UNAVAILABLE" and result["index"] is None


@pytest.mark.parametrize("bad", [True, -1, 2**53 + 1, "", "a" * 129, "bad\nface", None, 1.25])
def test_invalid_face_identity_rejected(surface, bad):
    assert surface.build_triangle_surface([triangle()], [bad])["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [
    [[.2, .2, np.nan]], [[.2, .2, np.inf]], [[True, .2, 1.]], [[2**53 + 1, .2, 1.]],
    [[np.nextafter(0., 1.), .2, 1.]], [[1e101, 0., 0.]], [[1e12, 0., 0.]],
    np.zeros((1, 2)), np.zeros((1, 3), dtype=complex),
])
def test_invalid_or_unsupported_query_clears_all_rows(surface, bad):
    assert_cleared(surface, surface.query_closest_surface(built(surface), bad))


@pytest.mark.parametrize("kwargs", [
    {"near_tie_distance": -1.}, {"near_tie_distance": np.nan}, {"near_tie_distance": True},
    {"normal_similarity_cos": 2.}, {"normal_similarity_cos": np.inf}, {"normal_similarity_cos": False},
    {"budgets": {}},
])
def test_invalid_query_policy_is_not_coerced(surface, kwargs):
    assert_cleared(surface, surface.query_closest_surface(built(surface), [[.2, .2, 1.]], **kwargs))


def test_empty_query_is_complete_without_geometry_work(surface):
    result = surface.query_closest_surface(built(surface), np.empty((0, 3)))
    assert result["status"] == "COMPLETE"
    assert result["closest_points"].shape == (0, 3)
    assert result["closest_regions"].shape == (0,)
    assert result["diagnostics"]["work_units"] == 0


def test_index_and_query_results_are_deterministic(surface):
    triangles = np.array([triangle(z) + [3 * (z % 4), 0., 0.] for z in range(32)])
    first = built(surface, triangles)
    second = built(surface, triangles)
    np.testing.assert_array_equal(first.permutation, second.permutation)
    np.testing.assert_array_equal(first.lower, second.lower)
    queries = [[.2, .2, .5], [9.2, .2, 8.]]
    a, b = surface.query_closest_surface(first, queries), surface.query_closest_surface(second, queries)
    assert a["status"] == b["status"] == "COMPLETE"
    for key in surface.RESULT_ARRAY_KEYS:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["diagnostics"] == b["diagnostics"]


def test_bounded_64_by_64_grid_has_many_leaves_and_prunes_finite_queries(surface):
    triangles = []
    face_ids = []
    for x in range(64):
        for y in range(64):
            a, b, c, d = [x, y, 0.], [x+1, y, 0.], [x+1, y+1, 0.], [x, y+1, 0.]
            triangles.extend([[a, b, c], [a, c, d]])
            face_ids.extend([64*x+y, 64*x+y])
    # Generated separately from tessellation vertices, away from diagonal seams.
    cells = [(x, y) for x in range(2, 62, 4) for y in range(3, 61, 5)]
    queries = np.array([[x+.25, y+.125, .5 + ((x+y) % 5)/16.] for x, y in cells])
    budget = replace(surface.TriangleSurfaceBudgets(), max_pair_batch=64)
    build = surface.build_triangle_surface(triangles, face_ids, budgets=budget)
    assert build["status"] == "COMPLETE"
    assert len(triangles) == 8192 and len(queries) == 180
    assert build["diagnostics"]["build_nodes"] == 1023
    result = surface.query_closest_surface(build["index"], queries, budgets=budget)
    assert result["status"] == "COMPLETE", result
    expected = queries.copy()
    expected[:, 2] = 0.
    np.testing.assert_allclose(result["closest_points"], expected, atol=3e-14, rtol=0)
    np.testing.assert_array_equal(result["distances"], queries[:, 2])
    assert result["source_face_ids"] == tuple(64*x+y for x, y in cells)
    assert set(result["closest_regions"]) == {"FACE"}
    assert np.all(result["near_tie_counts"] == 1)
    assert result["diagnostics"]["triangle_pairs"] < len(triangles) * len(queries) // 10
    assert result["diagnostics"]["maximum_pair_batch"] <= 64
    assert result["diagnostics"]["maximum_tie_storage_rows"] <= 2
