"""Independent invented fixed-node graph envelope checks; no protected inputs."""
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry.position_axis_hypothesis import _fit_plane
from research.geometry.position_graph_envelope import (
    _directed_interval_ranks, build_position_graph_envelope,
)


def analyze(points, **changes):
    policy = dict(unit="mm", position_error_bound=0.01, neighbor_count=2,
                  max_link_distance=2.0, symmetry="mutual")
    policy.update(changes)
    return build_position_graph_envelope(points, **policy)


def edges(value):
    return {tuple(edge) for edge in value}


def exact_fixed_node_graph(points, k, cap, symmetry):
    """Direct nominal definition. Never deduplicate hypothetical corrected rows."""
    p = np.asarray(points)
    distances = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    directed = np.zeros((len(p), len(p)), dtype=bool)
    for i in range(len(p)):
        others = [j for j in range(len(p)) if j != i]
        radius = sorted(float(distances[i, j]) for j in others)[k - 1]
        for j in others:
            directed[i, j] = distances[i, j] <= radius
    return {(i, j) for i in range(len(p)) for j in range(i + 1, len(p))
            if distances[i, j] <= cap and
            ((directed[i, j] and directed[j, i]) if symmetry == "mutual"
             else (directed[i, j] or directed[j, i]))}


def components(n, graph_edges):
    neighbors = {i: set() for i in range(n)}
    for i, j in graph_edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    remaining, result = set(range(n)), []
    while remaining:
        seen, frontier = set(), {min(remaining)}
        while frontier:
            seen |= frontier
            frontier = set.union(*(neighbors[i] for i in frontier)) - seen
        remaining -= seen
        result.append(seen)
    return result


def assert_scope(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["upper_edge_scope"] == "NOT_EXCLUDED_BY_PAIRWISE_INTERVAL_RANK_RELAXATION"
    assert result["edge_field_selects"] == "UPPER_PROPOSAL_GRAPH"
    for field in ("joint_realizability_is_verified", "individual_upper_edge_realizability_is_verified",
                  "physical_connectivity_is_verified", "enclosure_is_numerically_certified",
                  "neighborhood_membership_is_physically_verified", "measurement_error_bound_is_relaxed"):
        assert result[field] is False
    assert result["policy"]["latent_coincidences_remove_nodes"] is False
    assert result["policy"]["rank_competitors_exclude_query_and_target"] is True
    assert result["policy"]["complete_distance_ties_are_included"] is True
    assert result["policy"]["k_or_distance_cap_adaptation"] is False
    assert edges(result["lower_edges"]) <= edges(result["observed_edges"]) <= edges(result["edges"])
    for name, count in (("edges", "edge_count"), ("lower_edges", "lower_edge_count"),
                        ("observed_edges", "observed_edge_count")):
        assert result[count] == len(result[name]) <= 32640
        assert result[name] == sorted(result[name])
        assert all(0 <= i < j < result["unique_count"] for i, j in result[name])


def test_public_signature_requires_explicit_policy_and_positions_only():
    signature = inspect.signature(build_position_graph_envelope)
    assert set(signature.parameters) == {
        "points", "unit", "position_error_bound", "neighbor_count", "max_link_distance", "symmetry",
    }
    assert all(parameter.default == inspect.Parameter.empty for parameter in signature.parameters.values())
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY
               for name, parameter in signature.parameters.items() if name != "points")


def test_complete_tie_rank_counts_strictly_closer_nodes_and_excludes_target():
    # Query 0 has two equal closest OTHER nodes, then one more distant node.
    distance = np.array([[0, 1, 1, 3], [1, 0, 2, 4], [1, 2, 0, 2], [3, 4, 2, 0]], dtype=float)
    lower, outer = _directed_interval_ranks(distance, distance, 2)
    assert np.array_equal(lower, outer)
    assert lower[0].tolist() == [False, True, True, False]
    assert not np.diag(lower).any()
    # Positive interval width must not count the target as potentially closer to
    # itself. With k=N-1, every OTHER identity is always a member, including ties.
    lower, outer = _directed_interval_ranks(np.maximum(0, distance - 0.2), distance + 0.2, 3)
    assert np.array_equal(lower, ~np.eye(4, dtype=bool))
    assert np.array_equal(lower, outer)


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_observed_ties_can_produce_more_than_k_neighbors(symmetry):
    p = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
    result = analyze(p, position_error_bound=0, max_link_distance=1.1, symmetry=symmetry)
    assert_scope(result)
    center = result["positions"].index([0.0, 0.0, 0.0])
    assert sum(center in pair for pair in result["observed_edges"]) == 4


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_cap_has_separate_lower_and_outer_checks_without_tuning(symmetry):
    p = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    result = analyze(p, position_error_bound=0.1, max_link_distance=0.85, symmetry=symmetry)
    assert_scope(result)
    assert result["neighbor_count"] == 2
    assert result["max_link_distance"] == 0.85
    assert result["lower_edges"] == result["observed_edges"] == []
    assert len(result["edges"]) == 2
    assert result["distance_interval_radius"] >= 0.2


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
@pytest.mark.parametrize("seed", [0, 17, 81, 402, 20260907])
@pytest.mark.parametrize("epsilon", [0, 0.02, 0.25])
def test_lower_and_upper_enclose_25_independent_fixed_node_perturbations(symmetry, seed, epsilon):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(10, 3))
    result = analyze(p, position_error_bound=epsilon, neighbor_count=3,
                     max_link_distance=1.7, symmetry=symmetry)
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is True
    canonical = np.asarray(result["positions"])
    for _ in range(25):
        direction = rng.normal(size=canonical.shape)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        corrected = canonical + direction * rng.uniform(0, 0.99, size=(len(p), 1)) * epsilon
        assert np.all(np.linalg.norm(corrected - canonical, axis=1) <= epsilon + 1e-15)
        exact = exact_fixed_node_graph(corrected, 3, 1.7, symmetry)
        assert edges(result["lower_edges"]) <= exact <= edges(result["edges"])


def test_upper_rank_edge_can_be_impossible_even_individually_in_three_dimensions():
    epsilon = 0.3
    p = [[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [2, 0, 0]]
    result = analyze(p, position_error_bound=epsilon, max_link_distance=3)
    assert_scope(result)
    assert (0, 3) in edges(result["edges"])
    assert (0, 3) not in edges(result["observed_edges"])
    # For ANY corrected query q within epsilon of zero, compare the distances to
    # the observed competitor a and observed target 2. Rationalizing their
    # difference gives this uniform lower bound. It exceeds 2 epsilon, so even
    # moving competitor/target adversarially in full 3D cannot reverse their rank.
    # Both competitors are always closer: target 2 cannot be a k=2 neighbor.
    for a in (1.0, 1.1):
        distance_gap_lower = (2 - a) * (2 + a - 2 * epsilon) / (2 + a + 2 * epsilon)
        assert distance_gap_lower > 2 * epsilon
    assert result["individual_upper_edge_realizability_is_verified"] is False


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_individually_attainable_triangle_cap_edges_are_not_jointly_attainable(symmetry):
    epsilon, cap = 0.1, 0.8
    p = np.array([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]])
    result = analyze(p, position_error_bound=epsilon, max_link_distance=cap, symmetry=symmetry)
    assert_scope(result)
    assert len(result["edges"]) == 3
    assert result["lower_edges"] == []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        direction = (p[j] - p[i]) / np.linalg.norm(p[j] - p[i])
        corrected = p.copy()
        corrected[i] += epsilon * direction
        corrected[j] -= epsilon * direction
        assert np.linalg.norm(corrected[j] - corrected[i]) <= cap + 2e-15
    # Convexity of triangle perimeter at the equilateral triangle gives perimeter
    # >= 3 - 3*sqrt(3)*epsilon: each vertex gradient has norm sqrt(3).
    # All three cap edges would instead force perimeter <= 3*cap, impossible.
    assert 3 - 3 * math.sqrt(3) * epsilon > 3 * cap
    assert result["joint_realizability_is_verified"] is False


def test_union_is_a_changed_density_connection_policy_even_at_zero_error():
    p = [[x, 0, 0] for x in (0, 0.01, 0.02, 1, 2, 3)]
    mutual = analyze(p, position_error_bound=0, max_link_distance=1.1, symmetry="mutual")
    union = analyze(p, position_error_bound=0, max_link_distance=1.1, symmetry="union")
    assert_scope(mutual)
    assert_scope(union)
    assert (2, 3) not in edges(mutual["observed_edges"])
    assert (2, 3) in edges(union["observed_edges"])
    assert len(components(6, mutual["observed_edges"])) > 1
    assert len(components(6, union["observed_edges"])) == 1
    assert union["policy"]["union_is_not_mutual_graph"] is True
    assert edges(mutual["edges"]) < edges(union["edges"])


def test_zero_error_rank_and_cap_margins_cannot_be_repaired_by_an_envelope():
    p = [[x, 0, 0] for x in (0, 0.01, 0.02, 1, 2, 3)]
    result = analyze(p, position_error_bound=0, max_link_distance=0.5, symmetry="union")
    assert_scope(result)
    assert edges(result["edges"]) == edges(result["observed_edges"])
    assert len(components(6, result["edges"])) > 1


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_added_cap_edges_join_unrelated_planes_whose_whole_union_must_fail(symmetry):
    # Two parallel 3x3 patches; outer cap links join them although the nominal
    # cap does not. Their complete union spans a cube of width .1 > 2 epsilon.
    p = [[x, y, z] for z in (0, 0.1) for x in (0, 0.05, 0.1) for y in (0, 0.05, 0.1)]
    result = analyze(p, position_error_bound=0.01, neighbor_count=4,
                     max_link_distance=0.085, symmetry=symmetry)
    assert_scope(result)
    assert len(components(18, result["observed_edges"])) == 2
    assert len(components(18, result["edges"])) == 1
    plane = _fit_plane(np.asarray(result["positions"]), 0.01)
    assert "NO_ALL_POINT_PLANE_WITNESS" in plane["reasons"]
    assert plane["plane_witness"]["compatible_with_error_bound"] is False
    assert result["physical_connectivity_is_verified"] is False


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_latent_coincidences_keep_fixed_observed_nodes(symmetry):
    observed = [[x, 0, 0] for x in (0, 0.1, 1, 2)]
    corrected = [[x, 0, 0] for x in (0.05, 0.05, 1, 2)]
    result = analyze(observed, position_error_bound=0.1, symmetry=symmetry)
    assert_scope(result)
    assert result["unique_count"] == 4
    assert len({tuple(row) for row in corrected}) == 3
    exact = exact_fixed_node_graph(corrected, 2, 2, symmetry)
    assert edges(result["lower_edges"]) <= exact <= edges(result["edges"])


@pytest.mark.parametrize("seed", [0, 17, 20260907])
def test_permutation_and_observed_exact_dedup_are_deterministic(seed):
    p = np.random.default_rng(73).normal(size=(12, 3))
    reference = analyze(p)
    permuted = p[np.random.default_rng(seed).permutation(len(p))]
    assert analyze(permuted) == reference
    repeated = analyze(np.concatenate([permuted, permuted]))
    assert repeated["input_count"] == 24
    assert repeated["unique_count"] == 12
    for key in ("positions", "edges", "lower_edges", "observed_edges", "diagnostics"):
        assert repeated[key] == reference[key]


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_edge_correspondence_under_rigid_motion_and_unit_scale(symmetry):
    p = np.random.default_rng(701).normal(size=(12, 3))
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, -1], [0.4, 3, 0.7], [0.5, -1, 4]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved = (p @ rotation.T + [4, -5, 7]) / 1000
    first = analyze(p, symmetry=symmetry)
    second = analyze(moved, unit="m", position_error_bound=0.00001,
                     max_link_distance=0.002, symmetry=symmetry)
    assert_scope(first)
    assert_scope(second)
    first_ids = {tuple(row): i for i, row in enumerate(p)}
    second_ids = {tuple(row): i for i, row in enumerate(moved)}
    for key in ("edges", "lower_edges", "observed_edges"):
        one = {tuple(sorted(first_ids[tuple(first["positions"][i])] for i in edge)) for edge in first[key]}
        two = {tuple(sorted(second_ids[tuple(second["positions"][i])] for i in edge)) for edge in second[key]}
        assert one == two


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("position_error_bound", -1), ("position_error_bound", math.inf),
    ("position_error_bound", math.nan), ("position_error_bound", True),
    ("neighbor_count", 1), ("neighbor_count", 256), ("neighbor_count", True), ("neighbor_count", 2.5),
    ("max_link_distance", 0), ("max_link_distance", -1), ("max_link_distance", math.inf),
    ("max_link_distance", math.nan), ("max_link_distance", True),
    ("symmetry", "symmetric"), ("symmetry", "MUTUAL"), ("symmetry", None), ("symmetry", True),
])
def test_all_policies_validate_before_empty_input_return(field, value):
    with pytest.raises(ValueError):
        analyze([], **{field: value})


def test_raw_limit_precedes_deduplication_or_array_protocol():
    class Oversize(list):
        def __len__(self):
            return 257

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversized input converted")

    with pytest.raises(ValueError, match="256"):
        analyze(Oversize())
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 0]] * 257)


@pytest.mark.parametrize("points", [None, [], np.empty((0, 3)), [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]]])
def test_too_few_other_nodes_abstain_without_lowering_k(points):
    result = analyze(points)
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is True
    assert result["status"] == "ABSTAINED"
    assert result["reasons"] == ["INSUFFICIENT_OTHER_POSITIONS"]
    assert result["edges"] == result["lower_edges"] == result["observed_edges"] == []


@pytest.mark.parametrize("points,epsilon", [
    ([[-1.7e308, 0, 0], [1.7e308, 0, 0], [0, 1.7e308, 0]], 0),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e308),
    ([[0, 0, 0], [1e308, 0, 0], [0, 1e308, 0]], 5e307),
    ([[0, 0, 0], [1e-310, 0, 0], [0, 1e-310, 0]], 0),
])
def test_numerical_range_failure_clears_all_graph_variants(points, epsilon):
    result = analyze(points, position_error_bound=epsilon)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["edges"] == result["lower_edges"] == result["observed_edges"] == []
    assert result["reasons"] == ["NUMERICAL_RANGE_UNSUPPORTED"]
