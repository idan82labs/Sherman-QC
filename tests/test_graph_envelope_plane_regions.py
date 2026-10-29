"""Invented graph-overlink and cone-refit controls; no physical inputs.

The two negative constructions and their policies are fixed before execution.
They call the actual public discovery API, without patched graph/model helpers
or construction labels at inference. Predecessor helpers supply only a grid and
an independent serialized-plane residual/scope assertion.
"""
import json

import numpy as np
import pytest

from research.geometry.graph_envelope_plane_regions import discover_graph_envelope_plane_regions
from tests.test_cone_closed_plane_regions import plane_grid, scope as predecessor_scope


def analyze(points, **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6, neighbor_counts=[8],
        max_neighborhood_radius=1.5, max_normal_angle_bound=0.1,
        min_region_points=9, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5, max_plane_evaluations=2048, graph_symmetry="mutual",
    )
    policy.update(changes)
    return discover_graph_envelope_plane_regions(points, **policy)


def scope(result):
    predecessor_scope(result)
    assert result["kind"] == "GRAPH_ENVELOPE_PLANE_REGIONS_V1"
    assert result["policy"]["graph_edge_scope"] == "OUTER_PAIRWISE_INTERVAL_RANK_RELAXATION"
    assert result["policy"]["graph_relaxation_uses_original_error_bound"] is True
    graph = result["graph"]
    assert graph["edge_field_selects"] == "UPPER_PROPOSAL_GRAPH"
    assert graph["measurement_error_bound_is_relaxed"] is False
    assert graph["joint_realizability_is_verified"] is False
    assert graph["individual_upper_edge_realizability_is_verified"] is False
    assert graph["physical_connectivity_is_verified"] is False
    assert graph["neighborhood_membership_is_physically_verified"] is False
    assert graph["enclosure_is_numerically_certified"] is False
    json.dumps(result, allow_nan=False)


def graph_components(node_count, edges):
    adjacency = {index: set() for index in range(node_count)}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining, components = set(range(node_count)), []
    while remaining:
        reached, frontier = set(), {min(remaining)}
        while frontier:
            reached.update(frontier)
            frontier = set.union(*(adjacency[index] for index in frontier)) - reached
        remaining -= reached
        components.append(frozenset(reached))
    return components


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_cap_only_upper_link_can_merge_coplanar_patches_without_physical_identity(symmetry):
    epsilon = 0.001
    # Two 3x3 planar patches have nearest cross-patch distance cap+1.5*epsilon.
    # k=n-1 deliberately removes rank uncertainty from this CAP-ONLY control.
    # Observed components remain separate; the upper cap relaxation connects
    # them. A valid common sample plane does not establish one physical object.
    first = np.array([[x, y, 0.0] for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0)])
    second = first + [3.0+1.5*epsilon, 0.0, 0.0]
    points = np.concatenate([first, second])
    result = analyze(
        points, position_error_bound=epsilon, neighbor_counts=[8],
        max_neighborhood_radius=4.0, min_region_points=9,
        graph_neighbor_count=17, max_link_distance=1.0, graph_symmetry=symmetry,
    )
    scope(result)
    assert result["search_complete_within_declared_policy"] is True
    graph = result["graph"]
    assert sorted(map(len, graph_components(18, graph["observed_edges"]))) == [9, 9]
    assert list(map(len, graph_components(18, graph["edges"]))) == [18]
    canonical = np.asarray(result["positions"])
    side = canonical[:, 0] > 2.0
    assert not any(side[first] != side[second] for first, second in graph["observed_edges"])
    assert any(side[first] != side[second] for first, second in graph["edges"])
    assert graph["diagnostics"]["upper_only_edge_count"] >= 3
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(18))
    assert result["regions"][0]["physical_region_is_verified"] is False
    assert result["physical_region_count"] is None
    assert "NEARBY_COPLANAR_OBJECTS_CAN_SHARE_ONE_SAMPLE_WITNESS" in result["limitations"]


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_pointwise_cone_admission_does_not_relax_full_plane_error_or_fall_back(symmetry):
    epsilon = 0.001
    # Symmetry makes the full-support TLS plane exactly z=0. Its opposed points
    # have residual 1.5*epsilon, above the ORIGINAL measurement bound. Clean
    # corner seed triangles still admit both through the separate-point 2*epsilon
    # cone proposal; the complete connected proposal must then be rejected.
    grid = np.array([[x, y, 0.0] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    points = np.concatenate([grid, [[0.0, 0.0, 1.5*epsilon], [0.0, 0.0, -1.5*epsilon]]])
    result = analyze(
        points, position_error_bound=epsilon, neighbor_counts=[2],
        max_neighborhood_radius=1.5, min_region_points=3,
        graph_neighbor_count=10, max_link_distance=3.0, graph_symmetry=symmetry,
    )
    scope(result)
    assert result["position_error_bound"] == epsilon
    assert result["search_complete_within_declared_policy"] is True
    assert list(map(len, graph_components(11, result["graph"]["edges"]))) == [11]
    proposals = [query["cone_proposal"] for query in result["queries"] if "cone_proposal" in query]
    assert proposals
    assert all(proposal["candidate_point_count"] == 11 for proposal in proposals)
    assert all(proposal["proposal_distance_bound"] >= 2*epsilon for proposal in proposals)
    assert all(proposal["joint_compatibility_is_verified"] is False for proposal in proposals)
    assert result["diagnostics"]["accepted_raw_seed_count"] > 0
    assert result["diagnostics"]["rejected_closed_support_count"] == 1
    assert result["diagnostics"]["unique_closed_support_count"] == 1
    assert result["status"] == "ABSTAINED"
    assert result["regions"] == []
    assert result["reasons"] == ["NO_ALL_POINT_CONE_CLOSED_REGION_WITHIN_DECLARED_POLICY"]
    assert result["policy"]["fallback_to_seed_after_closure_failure"] is False
    assert result["policy"]["final_error_bound_is_relaxed"] is False


def test_one_plane_returns_one_full_exact_deduplicated_original_bound_witness():
    points = plane_grid()
    result = analyze(points)
    scope(result)
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(25))
    assert result["diagnostics"]["duplicate_closed_support_count"] > 0


def test_observed_row_permutation_and_duplicates_preserve_final_supports():
    points = plane_grid()
    reference = analyze(points)
    repeated = analyze(np.repeat(points[::-1], 2, axis=0))
    scope(reference)
    scope(repeated)
    assert repeated["input_count"] == 50 and repeated["unique_count"] == 25
    assert reference["positions"] == repeated["positions"]
    assert reference["regions"] == repeated["regions"]
    assert reference["graph"]["edges"] == repeated["graph"]["edges"]


def test_final_region_cap_clears_an_already_found_prefix():
    result = analyze(np.concatenate([plane_grid(), plane_grid(xshift=6.0)]), max_regions=1)
    scope(result)
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["FINAL_REGION_CANDIDATE_BUDGET_EXCEEDED"]


def test_seed_and_component_fit_budget_clears_an_already_found_prefix():
    result = analyze(plane_grid(), max_plane_evaluations=2)
    scope(result)
    assert result["diagnostics"]["plane_evaluation_count"] == 2
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["PLANE_EVALUATION_BUDGET_EXCEEDED"]


@pytest.mark.parametrize("changes", [
    {"graph_symmetry": "either"}, {"graph_symmetry": None},
    {"max_regions": 65}, {"max_plane_evaluations": 0},
    {"graph_neighbor_count": 1}, {"position_error_bound": -1},
])
def test_invalid_policy_is_rejected_before_empty_input_return(changes):
    with pytest.raises(ValueError):
        analyze([], **changes)


def test_empty_input_is_not_a_verified_empty_physical_part():
    result = analyze([])
    scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["regions"] == result["queries"] == []
    assert result["physical_region_count"] is None
