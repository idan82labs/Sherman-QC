"""Independent analytic tests for finite triangulated-surface correspondence.

The expected closest points below are constructed explicitly; the implementation
kernel, a native CAD reader, surface samples and registration are never used as
an oracle. These are invented arrays, not physical or analytic-CAD qualification.
Root must review this file and the implementation before any test execution.
"""

from dataclasses import dataclass
from itertools import permutations
import math

import numpy as np
import pytest

from backend.cad_guided_triangle_surface_v2 import (
    TriangleSurfaceBudgets,
    build_triangle_surface,
    query_closest_surface,
)


TRIANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
ARRAY_KEYS = (
    "closest_points", "distances", "triangle_indices", "source_face_ids",
    "oriented_normals", "normal_ambiguous", "closest_point_ambiguous",
    "face_identity_ambiguous", "near_tie_counts", "numerical_distance_guard",
    "closest_regions",
)
ANALYTIC_REGIONS = (
    ([.5, .5, 1.], [.5, .5, 0.], "FACE"),
    ([.5, -1., .5], [.5, 0., 0.], "EDGE"),
    ([-1., .5, .5], [0., .5, 0.], "EDGE"),
    ([1.5, 1.5, .5], [1., 1., 0.], "EDGE"),
    ([-1., -1., .5], [0., 0., 0.], "VERTEX"),
    ([3., -.5, .5], [2., 0., 0.], "VERTEX"),
    ([-.5, 3., .5], [0., 2., 0.], "VERTEX"),
)


def built(triangles, ids=None):
    triangles = np.asarray(triangles, dtype=np.float64)
    if ids is None:
        ids = np.arange(len(triangles), dtype=np.int64) + 101
    result = build_triangle_surface(triangles, np.asarray(ids, dtype=np.int64))
    assert result["status"] == "COMPLETE", result
    assert result["index"] is not None
    return result["index"]


def complete(result, count):
    assert result["status"] == "COMPLETE", result
    for name in ARRAY_KEYS:
        assert result[name] is not None, name
    assert np.asarray(result["closest_points"]).shape == (count, 3)
    assert np.asarray(result["oriented_normals"]).shape == (count, 3)
    for name in ("distances", "triangle_indices", "source_face_ids", "normal_ambiguous",
                 "closest_point_ambiguous", "face_identity_ambiguous", "near_tie_counts", "closest_regions"):
        assert np.asarray(result[name]).shape == (count,), name
    for name in ("closest_points", "distances", "oriented_normals", "numerical_distance_guard"):
        assert np.all(np.isfinite(result[name])), name
    assert np.all(np.asarray(result["distances"]) >= 0.)
    assert np.all(np.asarray(result["numerical_distance_guard"]) > 0.)
    assert np.all(np.asarray(result["near_tie_counts"]) >= 1)
    np.testing.assert_allclose(np.linalg.norm(result["oriented_normals"], axis=1), 1., rtol=0., atol=1e-12)


def cleared(result):
    assert result["status"] == "UNAVAILABLE", result
    assert result["reason"]
    for name in ARRAY_KEYS:
        assert result[name] is None, name


@pytest.mark.parametrize("query,expected,region", ANALYTIC_REGIONS)
@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_all_seven_voronoi_regions_under_every_vertex_order(query, expected, region, order):
    triangle = TRIANGLE[list(order)]
    surface = built([triangle], [17])
    result = query_closest_surface(surface, np.array([query]))
    complete(result, 1)
    np.testing.assert_allclose(result["closest_points"], [expected], rtol=0., atol=1e-10)
    np.testing.assert_allclose(result["distances"], [math.dist(query, expected)], rtol=0., atol=1e-10)
    assert list(result["closest_regions"]) == [region]
    assert list(result["source_face_ids"]) == [17]
    assert list(result["near_tie_counts"]) == [1]
    assert not any(bool(result[key][0]) for key in (
        "normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous"))
    oriented = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    oriented /= np.linalg.norm(oriented)
    np.testing.assert_allclose(result["oriented_normals"], [oriented], rtol=0., atol=1e-12)


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
@pytest.mark.parametrize("rotation", [
    np.eye(3),
    np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
])
def test_finite_projection_is_rigid_and_uniform_scale_equivariant(scale, rotation):
    translation = np.array([13., -7., 11.]) * scale
    queries = np.array([row[0] for row in ANALYTIC_REGIONS])
    expected = np.array([row[1] for row in ANALYTIC_REGIONS])
    transform = lambda value: np.asarray(value) @ rotation.T * scale + translation
    surface = built([transform(TRIANGLE)])
    result = query_closest_surface(surface, transform(queries))
    complete(result, len(queries))
    np.testing.assert_allclose(result["closest_points"], transform(expected), rtol=0., atol=1e-10 * scale)
    np.testing.assert_allclose(result["distances"], np.linalg.norm(queries - expected, axis=1) * scale,
                               rtol=0., atol=1e-10 * scale)
    assert list(result["closest_regions"]) == [row[2] for row in ANALYTIC_REGIONS]


def test_aabb_membership_and_infinite_plane_are_not_finite_triangle_membership():
    # Query lies in triangle0's AABB and plane, but outside x+y<=10.
    far_finite = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]])
    near_finite = np.array([[8., 8., 1.], [11., 8., 1.], [8., 11., 1.]])
    result = query_closest_surface(built([far_finite, near_finite], [31, 47]), [[9., 9., 0.]])
    complete(result, 1)
    np.testing.assert_allclose(result["closest_points"], [[9., 9., 1.]], rtol=0., atol=1e-10)
    np.testing.assert_allclose(result["distances"], [1.], rtol=0., atol=1e-10)
    assert list(result["source_face_ids"]) == [47]


def test_long_triangle_cannot_be_pruned_by_its_far_centroid():
    long_triangle = np.array([[-100., 0., 0.], [100., 0., 0.], [0., 2., 0.]])
    nearby_centroid = np.array([[98., -1., 1.], [101., -1., 1.], [98., 2., 1.]])
    query = np.array([[99., .01, .1]])
    result = query_closest_surface(built([nearby_centroid, long_triangle], [2, 9]), query)
    complete(result, 1)
    np.testing.assert_allclose(result["closest_points"], [[99., .01, 0.]], rtol=0., atol=1e-10)
    np.testing.assert_allclose(result["distances"], [.1], rtol=0., atol=1e-10)
    assert list(result["source_face_ids"]) == [9]


def test_finite_cap_rejects_its_plane_extension_in_favor_of_nearer_side():
    cap = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    side = np.array([[2.2, 0., 0.], [2.2, 1., 0.], [2.2, 0., 1.]])
    result = query_closest_surface(built([cap, side], [101, 202]), [[2., .25, .1], [.2, .2, .3]])
    complete(result, 2)
    np.testing.assert_allclose(result["closest_points"], [[2.2, .25, .1], [.2, .2, 0.]], rtol=0., atol=1e-10)
    np.testing.assert_allclose(result["distances"], [.2, .3], rtol=0., atol=1e-10)
    assert list(result["source_face_ids"]) == [202, 101]
    np.testing.assert_allclose(np.abs(result["oriented_normals"]), [[1., 0., 0.], [0., 0., 1.]], atol=1e-12)


def test_boundary_distance_is_not_facet_point_to_plane_residual():
    query = np.array([[2., .2, 0.]])
    surface = built([np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])])
    result = query_closest_surface(surface, query)
    complete(result, 1)
    assert result["distances"][0] > 1.
    assert result["closest_regions"][0] == "VERTEX"
    residual = np.einsum("ij,ij->i", query - result["closest_points"], result["oriented_normals"])
    np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-12)


def test_parallel_near_surfaces_have_point_ambiguity_without_normal_ambiguity():
    base = np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]])
    result = query_closest_surface(built([base + [0., 0., -1.], base + [0., 0., 1.]], [11, 12]), [[0., 0., 0.]])
    complete(result, 1)
    np.testing.assert_allclose(result["distances"], [1.], atol=1e-10)
    assert result["near_tie_counts"][0] == 2
    assert result["closest_point_ambiguous"][0]
    assert result["face_identity_ambiguous"][0]
    assert not result["normal_ambiguous"][0]


def triangle_with_normal(angle):
    tangent = np.array([-math.sin(angle), math.cos(angle), 0.])
    axial = np.array([0., 0., 1.])
    return np.array([-tangent - axial, tangent - axial, axial])


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_normal_tie_compatibility_is_all_pairs_not_winner_relative(order):
    # Each +/-4deg normal agrees with the 0deg normal at a5deg threshold,
    # while the two extremes differ by8deg and must expose ambiguity.
    triangles = np.array([triangle_with_normal(angle) for angle in (0., math.radians(4.), -math.radians(4.))])
    ids = np.array([13, 17, 19])
    result = query_closest_surface(built(triangles[list(order)], ids[list(order)]), [[0., 0., 0.]])
    complete(result, 1)
    assert result["near_tie_counts"][0] == 3
    assert result["normal_ambiguous"][0]
    assert not result["closest_point_ambiguous"][0]
    assert result["face_identity_ambiguous"][0]


def test_opposite_winding_is_oriented_normal_ambiguity_not_an_averaged_normal():
    result = query_closest_surface(built([TRIANGLE, TRIANGLE[[0, 2, 1]]], [5, 5]), [[.5, .5, 1.]])
    complete(result, 1)
    assert result["near_tie_counts"][0] == 2
    assert result["normal_ambiguous"][0]
    assert not result["closest_point_ambiguous"][0]
    assert not result["face_identity_ambiguous"][0]
    assert abs(float(result["oriented_normals"][0, 2])) == pytest.approx(1.)


def test_coplanar_duplicate_geometry_preserves_supplied_face_identity_ambiguity():
    result = query_closest_surface(built([TRIANGLE, TRIANGLE.copy()], [5, 7]), [[.5, .5, 1.]])
    complete(result, 1)
    assert result["near_tie_counts"][0] == 2
    assert result["face_identity_ambiguous"][0]
    assert not result["normal_ambiguous"][0]
    assert not result["closest_point_ambiguous"][0]


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_tie_band_is_relative_to_final_minimum_not_stale_earlier_winner(order):
    horizontal = np.array([[-2., -2., 1.08], [2., -2., 1.08], [0., 2., 1.08]])
    side = np.array([[1., -2., -2.], [1., 2., -2.], [1., 0., 2.]])
    triangles = np.array([horizontal, side, side + [.04, 0., 0.]])
    ids = np.array([101, 103, 107])
    result = query_closest_surface(built(triangles[list(order)], ids[list(order)]), [[0., 0., 0.]],
                                   near_tie_distance=.05)
    complete(result, 1)
    assert result["near_tie_counts"][0] == 2
    assert result["source_face_ids"][0] == 103
    np.testing.assert_allclose(result["distances"], [1.], atol=1e-10)
    assert not result["normal_ambiguous"][0]
    assert result["closest_point_ambiguous"][0]


def test_inputs_are_snapshotted_and_query_order_does_not_change_geometry():
    triangles = np.array([TRIANGLE])
    ids = np.array([23], dtype=np.int64)
    surface = built(triangles, ids)
    triangles[:] = 999.
    ids[:] = 999
    queries = np.array([row[0] for row in ANALYTIC_REGIONS])
    before = queries.copy()
    first = query_closest_surface(surface, queries)
    second = query_closest_surface(surface, queries[::-1].copy())
    complete(first, len(queries))
    complete(second, len(queries))
    np.testing.assert_array_equal(queries, before)
    np.testing.assert_allclose(first["closest_points"], second["closest_points"][::-1], rtol=0., atol=1e-10)
    assert list(first["source_face_ids"]) == [23] * len(queries)


@pytest.mark.parametrize("bad_triangle", [
    [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
    [[0., 0., 0.], [1., 0., 0.], [1., 1e-12, 0.]],
    [[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]],
])
def test_any_degenerate_or_unsupported_skinny_triangle_invalidates_whole_index(bad_triangle):
    result = build_triangle_surface(np.array([TRIANGLE, bad_triangle]), np.array([1, 2]))
    assert result["status"] == "UNAVAILABLE", result
    assert result["index"] is None and result["reason"]


def test_late_nonfinite_query_clears_all_preceding_correspondences():
    result = query_closest_surface(built([TRIANGLE]), [[.5, .5, .1], [float("nan"), 0., 0.]])
    cleared(result)


def test_late_tie_budget_failure_clears_the_whole_query_batch():
    crowded = np.repeat(TRIANGLE[None, :, :], 33, axis=0)
    triangles = np.concatenate(([TRIANGLE + [10., 0., 0.]], crowded))
    result = query_closest_surface(built(triangles), [[10.5, .5, .1], [.5, .5, .1]])
    cleared(result)


@pytest.mark.parametrize("budget", [
    TriangleSurfaceBudgets(max_triangles=0),
    TriangleSurfaceBudgets(max_build_nodes=0),
    TriangleSurfaceBudgets(max_build_triangle_visits=0),
])
def test_zero_build_work_or_row_budget_never_returns_partial_index(budget):
    result = build_triangle_surface(np.array([TRIANGLE]), [1], budgets=budget)
    assert result["status"] == "UNAVAILABLE", result
    assert result["index"] is None and result["reason"]


@pytest.mark.parametrize("budget", [
    TriangleSurfaceBudgets(max_query_rows=0),
    TriangleSurfaceBudgets(max_node_visits=0),
    TriangleSurfaceBudgets(max_triangle_pairs=0),
    TriangleSurfaceBudgets(max_work_units=0),
])
def test_zero_query_work_or_row_budget_clears_every_output_array(budget):
    result = query_closest_surface(built([TRIANGLE]), [[.5, .5, .1]], budgets=budget)
    cleared(result)


def test_normal_pair_budget_failure_after_nearest_pass_clears_geometry():
    surface = built([TRIANGLE, TRIANGLE.copy()], [1, 2])
    result = query_closest_surface(surface, [[.5, .5, .1]],
                                   budgets=TriangleSurfaceBudgets(max_normal_pairs=0))
    cleared(result)


def test_one_pair_batches_preserve_results_and_charge_both_passes_and_tie_work():
    surface = built([TRIANGLE, TRIANGLE.copy()], [1, 2])
    queries = np.array([[.5, .5, .1], [.25, .25, -.3]])
    baseline = query_closest_surface(surface, queries)
    batched = query_closest_surface(surface, queries, budgets=TriangleSurfaceBudgets(max_pair_batch=1))
    complete(baseline, len(queries))
    complete(batched, len(queries))
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(baseline[key], batched[key])
    diagnostic = batched["diagnostics"]
    assert diagnostic["maximum_pair_batch"] <= 1
    assert diagnostic["passes_completed"] == 2
    assert diagnostic["triangle_pairs"] == 2 * len(queries) * 2
    assert diagnostic["normal_pairs"] == len(queries)
    assert diagnostic["work_units"] == (
        diagnostic["node_visits"] + 8 * diagnostic["triangle_pairs"] + 3 * diagnostic["normal_pairs"])


@pytest.mark.parametrize("bad_triangle", [
    [[0., 0., 0.], [True, 0., 0.], [0., 1., 0.]],
    [[2**53 + 1, 0., 0.], [0., 1e16, 0.], [0., 0., 1e16]],
])
def test_mixed_scalar_lists_cannot_coerce_boolean_or_overprecision_geometry(bad_triangle):
    # Leave lists unconverted: preserving the offending scalar type is the test.
    result = build_triangle_surface([bad_triangle], [1])
    assert result["status"] == "UNAVAILABLE", result
    assert result["index"] is None and result["reason"]


@pytest.mark.parametrize("bad_query", [[True, .25, .1], [2**53 + 1, 0., 1.]])
def test_query_scalar_validation_precedes_numpy_list_promotion(bad_query):
    surface = built([TRIANGLE * 1e16])
    cleared(query_closest_surface(surface, [bad_query]))


@pytest.mark.parametrize("budget", [
    {},
    TriangleSurfaceBudgets(max_query_rows=200001),
    TriangleSurfaceBudgets(max_work_units=-1),
    TriangleSurfaceBudgets(max_triangle_pairs=True),
    TriangleSurfaceBudgets(max_pair_batch=0),
])
def test_malformed_or_raised_query_policy_is_unavailable_not_an_exception(budget):
    cleared(query_closest_surface(built([TRIANGLE]), [[.5, .5, .1]], budgets=budget))


def test_extended_dataclass_budget_is_rejected_by_build_and_query_without_exception():
    @dataclass(frozen=True)
    class ExtendedBudgets(TriangleSurfaceBudgets):
        undeclared_work: int = 1

    budget = ExtendedBudgets()
    result = build_triangle_surface(np.array([TRIANGLE]), [1], budgets=budget)
    assert result["status"] == "UNAVAILABLE", result
    assert result["index"] is None and result["reason"] == "INVALID_BUDGETS"
    query = query_closest_surface(built([TRIANGLE]), [[.5, .5, .1]], budgets=budget)
    cleared(query)
    assert query["reason"] == "INVALID_BUDGETS"
