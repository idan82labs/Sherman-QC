"""Invented positions only; no runtime, native or physical-data imports."""
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry import adaptive_position_regions as module
from research.geometry.position_bend_candidates import _components
from research.geometry.position_plane_regions import discover_position_plane_regions


def cross():
    return np.array([[0., 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])


def discover(points=None, **changes):
    options = dict(unit="mm", position_error_bound=0., neighbor_counts=[2],
                   max_neighborhood_radius=3., max_normal_angle_bound=.1,
                   min_region_points=3, max_regions=64)
    options.update(changes)
    return module.discover_adaptive_position_plane_regions(cross() if points is None else points,
                                                          **options)


def graph(points=None, **changes):
    options = dict(unit="mm", position_error_bound=0., neighbor_count=2,
                   max_link_distance=3.)
    options.update(changes)
    return module.build_mutual_position_graph(cross() if points is None else points, **options)


def scopes(result):
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert "physical_bend_count" not in result
    for key, value in result.items():
        if key.endswith(("is_verified", "is_physically_verified")):
            assert value is False


def test_position_only_explicit_api_has_no_supplied_geometric_truth():
    for function in (module.discover_adaptive_position_plane_regions,
                     module.build_mutual_position_graph):
        params = inspect.signature(function).parameters
        assert all(param.kind == inspect.Parameter.KEYWORD_ONLY for key, param in params.items()
                   if key != "points")
        assert not set(params) & {"labels", "normals", "axis", "radius", "truth", "model"}


def test_complete_tie_shell_is_not_arbitrarily_truncated_to_k():
    result = discover()
    scopes(result)
    center = result["positions"].index([0., 0., 0.])
    query = next(query for query in result["queries"] if query["seed_index"] == center)
    assert query["radius"] == 1
    assert query["neighbor_count"] == 2
    assert query["support_count"] == 5  # 4 OTHER tied points, not just k=2.
    assert query["rank_boundary_sensitive_count"] == 4
    assert any(region["point_indices"] == list(range(5)) for region in result["regions"])
    assert len(result["regions"]) == 5
    assert result["seed_query_count"] == result["unique_support_query_count"] == 5
    assert result["policy"]["neighbor_count_excludes_seed"] is True
    assert result["policy"]["membership_is_physically_verified"] is False


def test_graph_mutual_ties_can_produce_degree_greater_than_k():
    result = graph()
    scopes(result)
    center = result["positions"].index([0., 0., 0.])
    assert sum(center in edge for edge in result["edges"]) == 4
    assert result["edge_count"] == 8
    assert result["edges"] == sorted(result["edges"])
    assert all(i < j for i, j in result["edges"])


@pytest.mark.parametrize("function", [discover, graph])
def test_canonical_exact_repeats_and_permutations_preserve_every_result(function):
    expected = function()
    actual = function(np.repeat(cross()[::-1], 3, axis=0))
    assert actual["input_count"] == 15
    actual["input_count"] = 5
    assert actual == expected


def test_neighbor_count_order_preserves_all_overlapping_alternatives():
    first = discover(neighbor_counts=[2, 4])
    second = discover(neighbor_counts=[4, 2])
    assert first == second
    assert len(first["regions"]) == 5
    assert sum(len(region["seed_queries"]) for region in first["regions"]) == 10
    supports = [set(region["point_indices"]) for region in first["regions"]]
    assert any(a < b for a in supports for b in supports)
    assert first["policy"]["residual_trimming"] is False


def test_nominal_radius_cap_skips_entire_query_never_clips_ties_or_lowers_k():
    result = discover(max_neighborhood_radius=.99)
    assert result["search_complete_within_declared_policy"] is True
    assert result["regions"] == []
    assert result["unique_support_query_count"] == 0
    assert len(result["queries"]) == 5
    assert {query["status"] for query in result["queries"]} == {"NOMINAL_RADIUS_CAP_EXCEEDED"}


def test_insufficient_other_samples_do_not_silently_lower_explicit_k():
    result = discover(neighbor_counts=[5])
    assert result["search_complete_within_declared_policy"] is True
    assert not result["regions"]
    assert all(query["radius"] is None and query["status"] == "INSUFFICIENT_OTHER_POSITIONS"
               for query in result["queries"])
    linked = graph(neighbor_count=5)
    assert linked["search_complete_within_declared_policy"] is True
    assert linked["edges"] == linked["local_radii"] == []
    assert linked["reasons"] == ["INSUFFICIENT_OTHER_POSITIONS"]


def test_exhausted_region_budget_returns_no_prefix():
    result = discover(max_regions=4)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
    assert len(result["queries"]) == 5


def test_density_asymmetry_does_not_turn_one_way_neighbors_into_edges():
    points = np.array([[x, 0, 0] for x in (0., .1, .2, 10., 11.)])
    result = graph(points, max_link_distance=20.)
    assert result["edges"] == [[0, 1], [0, 2], [1, 2], [3, 4]]
    # The sparse sample at 10 selects .2, but .2's dense neighbors exclude 10.
    assert result["diagnostics"]["nonmutual_neighbor_count"] == 2
    assert [2, 3] not in result["edges"]


def test_observed_link_cap_rejects_long_mutual_edges_even_in_sparse_sampling():
    points = [[x, 0, 0] for x in (0., 1., 10., 11., 20., 21.)]
    result = graph(points, max_link_distance=2.)
    assert result["edges"] == [[0, 1], [2, 3], [4, 5]]
    assert result["diagnostics"]["mutual_pairs_beyond_cap"] > 0
    assert max(result["local_radii"]) > result["max_link_distance"]


def test_epsilon_changes_only_rank_and_cap_diagnostics_not_observed_edges_or_radii():
    exact = graph(max_link_distance=.99)
    uncertain = graph(max_link_distance=.99, position_error_bound=.1)
    assert uncertain["edges"] == exact["edges"] == []
    assert uncertain["local_radii"] == exact["local_radii"]
    guard = uncertain["distance_numerical_allowance"]
    assert uncertain["rank_membership_uncertainty_allowance"] == .4 + 2 * guard
    assert uncertain["link_cap_uncertainty_allowance"] == .2 + guard
    assert uncertain["diagnostics"]["link_cap_sensitive_pair_count"] > 0
    assert uncertain["epsilon_expands_edges"] is False


def test_mutual_graph_may_connect_distinct_nearby_surfaces_without_physical_claim():
    triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    points = np.concatenate([triangle, triangle + [0, 0, .01]])
    result = graph(points, neighbor_count=5, max_link_distance=2.)
    assert result["edge_count"] == 15
    assert result["physical_connectivity_is_verified"] is False
    assert "MUTUAL_KNN_CAN_FRAGMENT_ONE_SURFACE_OR_CONNECT_DISTINCT_SURFACES" in result["limitations"]


def sparse_quarter():
    # Same sparse stratum geometry as the retained V1 structural matrix: five
    # stations across 2R, not R. Truth is used only below, never in either API.
    radius = 2.
    levels = np.linspace(-radius / 2, radius / 2, 5)
    one = np.array([[x, y, radius] for x in np.linspace(-2 * radius, 0, 5) for y in levels])
    two = np.array([[radius, y, z] for z in np.linspace(-2 * radius, 0, 5) for y in levels])
    arc = np.array([[radius * math.sin(j * math.pi / 16), y,
                     radius * math.cos(j * math.pi / 16)] for j in range(1, 8) for y in levels])
    return one, two, arc


def test_adaptive_birth_and_mutual_sampling_cover_retained_sparse_fixture():
    one, two, arc = sparse_quarter()
    points = np.concatenate([one, two, arc])
    # Explicit synthetic development policies; no geometric parameter enters API.
    shared = dict(unit="mm", position_error_bound=1e-6, max_normal_angle_bound=.1,
                  min_region_points=9, max_regions=64)
    baseline = discover_position_plane_regions(points, neighborhood_radii=[.8], **shared)
    adaptive = module.discover_adaptive_position_plane_regions(
        points, neighbor_counts=[8], max_neighborhood_radius=1.5, **shared)
    assert baseline["regions"] == []  # Do not erase the frozen V1 failure.
    assert adaptive["status"] == "PLANE_REGION_HYPOTHESES"
    assert len(adaptive["regions"]) == 32
    canonical = np.asarray(adaptive["positions"])
    true_supports = [set(np.flatnonzero(np.any(np.all(
        canonical[:, None, :] == surface[None, :, :], axis=2), axis=1))) for surface in (one, two)]
    for support in true_supports:
        selected = [set(region["point_indices"]) for region in adaptive["regions"]
                    if set(region["point_indices"]) <= support]
        assert selected and set.union(*selected) == support
    sampled = graph(points, neighbor_count=8, max_link_distance=1.5,
                    position_error_bound=1e-6)
    adjacency = np.zeros((len(points), len(points)), dtype=bool)
    for i, j in sampled["edges"]:
        adjacency[i, j] = adjacency[j, i] = True
    assert len(_components(adjacency, np.ones(len(points), bool))) == 1
    for support in true_supports:
        allowed = np.zeros(len(points), bool)
        allowed[list(support)] = True
        assert _components(adjacency, allowed) == [tuple(sorted(support))]
    scopes(adaptive)
    scopes(sampled)


def test_no_residual_trimming_of_complete_nonplanar_support():
    points = np.concatenate([cross(), [[0, 0, .1]]])
    result = discover(points, neighbor_counts=[5], position_error_bound=.001)
    assert result["regions"] == []
    assert result["unique_support_query_count"] == 1
    assert all(query["support_count"] == 6 for query in result["queries"])


def test_rank_selection_diagnostics_use_four_epsilon_not_two():
    points = np.concatenate([cross(), [[1.03, 0, 0]]])
    result = discover(points, position_error_bound=.01)
    center = result["positions"].index([0., 0., 0.])
    query = next(q for q in result["queries"] if q["seed_index"] == center)
    assert query["support_count"] == 5
    assert query["rank_boundary_sensitive_count"] == 5  # Outside 2eps, inside 4eps.
    guard = result["distance_numerical_allowance"]
    assert result["rank_membership_uncertainty_allowance"] == .04 + 2 * guard
    assert result["radius_cap_uncertainty_allowance"] == .02 + guard


def test_se3_and_units_preserve_complete_ties_away_from_caps():
    points = cross()
    q, _ = np.linalg.qr(np.array([[.3, .7, .5], [.8, -.2, .1], [.4, .5, -.9]]))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = (points @ q.T + [2., -3., 7.]) * 1000
    before, after = discover(), discover(moved, max_neighborhood_radius=3000., unit="m")
    def identities(result, original):
        canonical = np.asarray(result["positions"])
        return np.argmin(np.linalg.norm(canonical[:, None] - original[None], axis=2), axis=1)
    old_ids, new_ids = identities(before, points), identities(after, moved)
    def supports(result, ids):
        return {frozenset(ids[region["point_indices"]]) for region in result["regions"]}
    assert supports(before, old_ids) == supports(after, new_ids)
    g1, g2 = graph(), graph(moved, max_link_distance=3000., unit="m")
    assert {frozenset(old_ids[edge]) for edge in g1["edges"]} == {
        frozenset(new_ids[edge]) for edge in g2["edges"]}


def test_exact_nominal_cap_uses_only_guard_and_is_stable_across_units_and_transforms():
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    rng = np.random.default_rng(70932026)
    for scale in (1e-6, 1., 1000., 1e6):
        for _ in range(8):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            moved = (points @ rotation.T + [2., -3., 7.]) * scale
            result = discover(moved, neighbor_counts=[3],
                              max_neighborhood_radius=math.sqrt(2) * scale)
            assert result["status"] == "PLANE_REGION_HYPOTHESES"
            assert len(result["regions"]) == 1
            assert result["regions"][0]["point_indices"] == list(range(4))
            assert {query["status"] for query in result["queries"]} == {"QUERIED"}
            assert all(query["radius_cap_is_sensitive"] for query in result["queries"])
            cap = result["policy"]["max_neighborhood_radius"]
            guard = result["distance_numerical_allowance"]
            canonical = np.asarray(result["positions"])
            for query in result["queries"]:
                distances = np.linalg.norm(canonical - canonical[query["seed_index"]], axis=1)
                assert float(np.max(distances)) <= cap + 2 * guard
            assert result["policy"]["epsilon_expands_membership_or_cap"] is False
    # A physical error allowance cannot admit a radius genuinely above the cap.
    rejected = discover(points, neighbor_counts=[3], max_neighborhood_radius=1.4,
                        position_error_bound=.1)
    assert rejected["regions"] == []
    assert all(query["status"] == "NOMINAL_RADIUS_CAP_EXCEEDED"
               for query in rejected["queries"])


@pytest.mark.parametrize("points", [None, [], [[0, 0, 0]], [[0, 0, 0]] * 5])
@pytest.mark.parametrize("function", [discover, graph])
def test_empty_or_repeated_insufficient_inputs_are_unknown_and_complete(points, function):
    result = function(points)
    # Helpers reserve None as their default fixture, so call the API directly.
    if points is None:
        if function is discover:
            result = module.discover_adaptive_position_plane_regions(
                None, unit="mm", position_error_bound=0., neighbor_counts=[2],
                max_neighborhood_radius=3., max_normal_angle_bound=.1,
                min_region_points=3, max_regions=64)
        else:
            result = module.build_mutual_position_graph(
                None, unit="mm", position_error_bound=0., neighbor_count=2, max_link_distance=3.)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    scopes(result)


class NoIteration(list):
    def __iter__(self):
        raise AssertionError("oversize payload must be rejected before conversion")


@pytest.mark.parametrize("function", [discover, graph])
def test_raw_input_cap_precedes_conversion_and_deduplication(function):
    with pytest.raises(ValueError, match="at most 256"):
        function(NoIteration([object()] * 257))


def test_neighbor_count_budget_precedes_element_conversion():
    with pytest.raises(ValueError, match="one to four"):
        discover(neighbor_counts=NoIteration([object()] * 5))


@pytest.mark.parametrize("changes", [
    {"neighbor_counts": []}, {"neighbor_counts": [2, 2]}, {"neighbor_counts": [True]},
    {"neighbor_counts": [1]}, {"neighbor_counts": [256]}, {"neighbor_counts": [2.]},
    {"neighbor_counts": np.ones((1, 2))}, {"neighbor_counts": (x for x in [2])},
    {"max_neighborhood_radius": 0}, {"max_neighborhood_radius": float("inf")},
    {"max_neighborhood_radius": True}, {"max_neighborhood_radius": 10 ** 1000},
    {"max_normal_angle_bound": math.pi / 2}, {"max_normal_angle_bound": 0},
    {"max_normal_angle_bound": True}, {"min_region_points": 2}, {"max_regions": 65},
    {"min_region_points": 3.}, {"max_regions": True},
])
def test_discovery_rejects_invalid_explicit_policies(changes):
    with pytest.raises(ValueError):
        discover(**changes)


@pytest.mark.parametrize("changes", [
    {"neighbor_count": 1}, {"neighbor_count": 256}, {"neighbor_count": True},
    {"neighbor_count": 2.}, {"max_link_distance": 0}, {"max_link_distance": True},
    {"max_link_distance": float("nan")}, {"max_link_distance": 10 ** 1000},
])
def test_graph_rejects_invalid_explicit_policies(changes):
    with pytest.raises(ValueError):
        graph(**changes)


@pytest.mark.parametrize("function", [discover, graph])
@pytest.mark.parametrize("changes", [
    {"unit": "cm"}, {"unit": True}, {"position_error_bound": -1},
    {"position_error_bound": True}, {"position_error_bound": float("nan")},
    {"position_error_bound": float("inf")}, {"position_error_bound": 10 ** 1000},
])
def test_common_policy_validation(function, changes):
    with pytest.raises(ValueError):
        function(**changes)


@pytest.mark.parametrize("function", [discover, graph])
@pytest.mark.parametrize("points", [
    [[0, 0]], [[0, 0, 0, 0]], [[True, 0, 0]], [["0", 0, 0]],
    [[float("nan"), 0, 0]], np.ones((3, 4)), [[10 ** 1000, 0, 0]],
])
def test_malformed_position_payloads_raise(function, points):
    with pytest.raises(ValueError):
        function(points)


@pytest.mark.parametrize("function", [discover, graph])
@pytest.mark.parametrize("points,epsilon", [
    ([[-1e308, 0, 0], [1e308, 0, 0], [0, 1e308, 0]], 0.),
    ([[0, 0, 0], [1e-320, 0, 0], [0, 1e-320, 0]], 0.),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e308),
])
def test_unsupported_numerics_are_finite_incomplete_results(function, points, epsilon):
    result = function(points, position_error_bound=epsilon)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["NUMERICAL_RANGE_UNSUPPORTED"]
    scopes(result)


def test_plane_analysis_failure_does_not_publish_preceding_successes(monkeypatch):
    original = module._fit_plane
    calls = 0
    def fail_second(points, epsilon):
        nonlocal calls
        calls += 1
        result = original(points, epsilon)
        if calls == 2:
            result["reasons"] = ["PLANE_FIT_UNAVAILABLE"]
        return result
    monkeypatch.setattr(module, "_fit_plane", fail_second)
    result = discover()
    assert calls == 2
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["REGION_PLANE_ANALYSIS_UNAVAILABLE"]
