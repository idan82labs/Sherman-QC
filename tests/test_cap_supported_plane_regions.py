"""Independent invented cap-seed controls; no files, patching, or physical data.

These checks call the public discovery functions. Small cap seeds never replace
the declared minimum for an accepted final region or a legacy kth-neighbor seed.
The numerical-boundary construction is specified analytically before execution.
"""
import json
import math

import numpy as np
import pytest

from research.geometry.cap_supported_plane_regions import discover_cap_supported_plane_regions
from research.geometry.graph_envelope_plane_regions import discover_graph_envelope_plane_regions


def policy(**changes):
    arguments = dict(
        unit="mm", position_error_bound=1e-6, neighbor_counts=[2],
        max_neighborhood_radius=1.1, max_normal_angle_bound=0.1,
        min_region_points=9, min_seed_points=3, max_regions=64,
        graph_neighbor_count=8, max_link_distance=1.1,
        max_plane_evaluations=2048, graph_symmetry="union",
    )
    arguments.update(changes)
    return arguments


def analyze(points, **changes):
    return discover_cap_supported_plane_regions(points, **policy(**changes))


def predecessor(points, **changes):
    arguments = policy(**changes)
    arguments.pop("min_seed_points")
    return discover_graph_envelope_plane_regions(points, **arguments)


def grid(shift=0.0, size=3):
    return np.array([[x+shift, y, 0.0] for x in np.linspace(-1, 1, size)
                     for y in np.linspace(-1, 1, size)])


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["kind"] == "CAP_SUPPORTED_PLANE_REGIONS_V1"
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_region_count"] is None
    assert result["policy"]["cap_balls_are_failure_triggered"] is False
    assert result["policy"]["cap_ball_distance_is_relaxed_by_position_error"] is False
    assert result["policy"]["final_error_bound_is_relaxed"] is False
    assert result["policy"]["fallback_to_seed_after_closure_failure"] is False
    assert result["policy"]["raw_seed_query_limit"] == 1024
    assert result["diagnostics"]["plane_evaluation_count"] <= result["policy"]["max_plane_evaluations"]
    supports = [tuple(region["point_indices"]) for region in result["regions"]]
    assert len(set(supports)) == len(supports) <= result["policy"]["max_regions"]
    points = np.asarray(result["positions"])
    for region in result["regions"]:
        indices = region["point_indices"]
        assert len(indices) >= result["policy"]["min_region_points"]
        assert region["physical_region_is_verified"] is False
        plane = region["plane_analysis"]
        assert plane["reasons"] == []
        assert plane["effective_error_bound"] == result["position_error_bound"] + plane["numerical_allowance"]
        assert plane["normal_angle_bound_radians"] <= result["policy"]["max_normal_angle_bound_radians"]
        witness = plane["plane_witness"]
        residuals = np.abs((points[indices]-witness["reference_point"]) @ witness["normal"]
                           - witness["offset_from_reference"])
        assert np.max(residuals) <= plane["effective_error_bound"]
        for query_id in region["seed_queries"]:
            assert result["queries"][query_id]["seed_eligible_by_declared_size"] is True


def query_signature(query):
    return tuple((key, query.get(key)) for key in (
        "seed_index", "neighbor_count", "radius", "status", "point_indices",
    ))


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_small_cap_seeds_close_to_nine_without_relabeling_legacy_eligibility(symmetry):
    points = grid()
    old = predecessor(points, graph_symmetry=symmetry)
    result = analyze(points, graph_symmetry=symmetry)
    scope(result)
    assert old["regions"] == []
    assert old["diagnostics"]["accepted_raw_seed_count"] == 0
    assert result["graph"] == old["graph"]
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(9))
    assert result["diagnostics"]["accepted_subfinal_size_seed_count"] > 0
    kth = [q for q in result["queries"] if q["kind"] == "KTH_NEIGHBOR_BALL"]
    cap = [q for q in result["queries"] if q["kind"] == "OBSERVED_RADIUS_CAP_BALL"]
    assert [query_signature(q) for q in kth] == [query_signature(q) for q in old["queries"]]
    assert len(kth) == len(cap) == 9
    for legacy, added in zip(kth, cap):
        assert legacy["point_indices"] == added["point_indices"]
        assert 3 <= len(added["point_indices"]) < 9
        assert legacy["seed_minimum_points"] == 9
        assert legacy["seed_eligible_by_declared_size"] is False
        assert "cone_proposal" not in legacy
        assert added["seed_minimum_points"] == 3
        assert added["seed_eligible_by_declared_size"] is True
    assert all(result["queries"][i]["kind"] == "OBSERVED_RADIUS_CAP_BALL"
               for i in result["regions"][0]["seed_queries"])


def test_cap_balls_run_even_when_legacy_queries_and_final_region_already_succeed():
    points = grid(size=5)
    changes = dict(neighbor_counts=[8], max_neighborhood_radius=1.5, max_link_distance=1.5,
                   graph_neighbor_count=10)
    old = predecessor(points, **changes)
    result = analyze(points, **changes)
    scope(result)
    assert old["regions"] and result["regions"]
    kth = [q for q in result["queries"] if q["kind"] == "KTH_NEIGHBOR_BALL"]
    cap = [q for q in result["queries"] if q["kind"] == "OBSERVED_RADIUS_CAP_BALL"]
    assert [query_signature(q) for q in kth] == [query_signature(q) for q in old["queries"]]
    assert any("cone_proposal" in q for q in kth)
    assert len(cap) == 25
    assert result["diagnostics"]["seed_query_count"] == 50
    assert result["diagnostics"]["cap_ball_query_count"] == 25
    assert result["graph"] == old["graph"]
    assert [r["point_indices"] for r in result["regions"]] == [r["point_indices"] for r in old["regions"]]


def test_whole_cap_ball_keeps_incompatible_fourth_point_not_a_favorable_triple():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    result = analyze(points, max_neighborhood_radius=2., max_link_distance=2., graph_neighbor_count=3)
    scope(result)
    cap = [q for q in result["queries"] if q["kind"] == "OBSERVED_RADIUS_CAP_BALL"]
    assert len(cap) == 4
    assert all(q["point_indices"] == list(range(4)) for q in cap)
    assert all(q["seed_eligible_by_declared_size"] for q in cap)
    assert all("cone_proposal" not in q for q in cap)
    assert result["diagnostics"]["accepted_raw_seed_count"] == 0
    assert result["diagnostics"]["plane_evaluation_count"] == 1
    assert result["search_complete_within_declared_policy"] is True
    assert result["regions"] == []


@pytest.mark.parametrize("height", [0., 1e-8])
def test_collinear_or_badly_conditioned_three_point_ball_is_not_an_accepted_seed(height):
    points = np.array([[-1., 0., 0.], [0., height, 0.], [1., 0., 0.]])
    result = analyze(points, max_neighborhood_radius=3., max_link_distance=3., graph_neighbor_count=2)
    scope(result)
    cap = [q for q in result["queries"] if q["kind"] == "OBSERVED_RADIUS_CAP_BALL"]
    assert len(cap) == 3
    assert all(q["point_indices"] == list(range(3)) for q in cap)
    assert all(q["seed_eligible_by_declared_size"] for q in cap)
    assert all("cone_proposal" not in q for q in cap)
    assert result["diagnostics"]["plane_evaluation_count"] == 1
    assert result["diagnostics"]["accepted_raw_seed_count"] == 0
    assert result["regions"] == []


def test_qualified_triangle_cannot_bypass_final_nine_sample_minimum():
    result = analyze([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                     max_neighborhood_radius=2., max_link_distance=2., graph_neighbor_count=2)
    scope(result)
    assert result["diagnostics"]["accepted_subfinal_size_seed_count"] == 1
    assert result["diagnostics"]["subminimum_component_count"] == 1
    assert result["diagnostics"]["unique_closed_support_count"] == 0
    assert result["search_complete_within_declared_policy"] is True
    assert result["status"] == "ABSTAINED" and result["regions"] == []


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_small_clean_seed_cannot_fall_back_after_incompatible_full_closure(symmetry):
    epsilon = .001
    points = np.concatenate([grid(), [[0., 0., 1.5*epsilon], [0., 0., -1.5*epsilon]]])
    result = analyze(points, position_error_bound=epsilon, graph_neighbor_count=10,
                     max_link_distance=3., graph_symmetry=symmetry)
    scope(result)
    assert result["diagnostics"]["accepted_subfinal_size_seed_count"] > 0
    proposals = [q["cone_proposal"] for q in result["queries"] if "cone_proposal" in q]
    assert proposals and all(p["candidate_point_count"] == 11 for p in proposals)
    assert result["diagnostics"]["unique_closed_support_count"] == 1
    assert result["diagnostics"]["rejected_closed_support_count"] == 1
    assert result["position_error_bound"] == epsilon
    assert result["search_complete_within_declared_policy"] is True
    assert result["regions"] == []


def test_full_cap_one_guard_and_legacy_two_guard_tie_closure_remain_distinct():
    # With these unit-scale points the actual disclosed distance allowance is
    # g0 up to O(machine_epsilon**2). The two kth neighbors are 0.5*g0 beyond
    # the nominal cap. A third neighbor 1.25*g0 beyond the cap belongs to the
    # kth ball's full tie closure but NOT to the independent one-guard cap ball.
    g0 = 128*np.finfo(float).eps*(1+math.sqrt(2))
    r1, r2 = 1+.5*g0, 1+1.25*g0
    points = np.array([[0., 0., 0.], [r1, 0., 0.], [0., r1, 0.], [0., 0., r2]])
    changes = dict(max_neighborhood_radius=1., max_link_distance=3., graph_neighbor_count=3)
    result = analyze(points, **changes)
    old = predecessor(points, **changes)
    scope(result)
    origin = result["positions"].index([0., 0., 0.])
    target = result["positions"].index([0., 0., r2])
    queries = [q for q in result["queries"] if q["seed_index"] == origin]
    kth = next(q for q in queries if q["kind"] == "KTH_NEIGHBOR_BALL")
    cap = next(q for q in queries if q["kind"] == "OBSERVED_RADIUS_CAP_BALL")
    previous = next(q for q in old["queries"] if q["seed_index"] == origin)
    guard = result["distance_numerical_allowance"]
    assert 0 < r1-1 < guard
    assert guard < r2-1 < 2*guard
    assert r2-r1 < guard
    assert kth["status"] == "QUERIED"
    assert query_signature(kth) == query_signature(previous)
    assert target in kth["point_indices"] and target not in cap["point_indices"]
    assert len(kth["point_indices"]) == 4 and len(cap["point_indices"]) == 3
    assert result["policy"]["cap_ball_radius_guard_count"] == 1
    assert result["policy"]["tie_closed_radius_cap_guard_count"] == 2


@pytest.mark.parametrize("epsilon", [0., 1e-6, .01])
def test_position_error_never_expands_the_observed_cap_ball(epsilon):
    points = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [1.005, 0., 0.]])
    result = analyze(points, position_error_bound=epsilon, max_neighborhood_radius=1.,
                     max_link_distance=2., graph_neighbor_count=3)
    scope(result)
    origin = result["positions"].index([0., 0., 0.])
    target = result["positions"].index([1.005, 0., 0.])
    cap = next(q for q in result["queries"]
               if q["seed_index"] == origin and q["kind"] == "OBSERVED_RADIUS_CAP_BALL")
    assert len(cap["point_indices"]) == 3 and target not in cap["point_indices"]


@pytest.mark.parametrize("count,neighbors,complete", [
    (204, [2, 3, 4, 5], True), (205, [2, 3, 4, 5], False),
    (256, [2, 3, 4], True), (256, [2, 3, 4, 5], False),
])
def test_exact_query_ceiling_preflights_before_any_seed_or_component_fits(count, neighbors, complete):
    # Every cap ball contains one point, so allowed runs do no plane fits either.
    points = np.array([[float(index), 0., 0.] for index in range(count)])
    result = analyze(points, neighbor_counts=neighbors, max_neighborhood_radius=.25,
                     graph_neighbor_count=2)
    scope(result)
    planned = count*(len(neighbors)+1)
    assert result["diagnostics"]["seed_query_count"] == planned
    assert result["diagnostics"]["plane_evaluation_count"] == 0
    assert result["search_complete_within_declared_policy"] is complete
    assert result["regions"] == []
    if complete:
        assert len(result["queries"]) == planned <= 1024
    else:
        assert planned > 1024
        assert result["queries"] == []
        assert result["reasons"] == ["SEED_QUERY_BUDGET_EXCEEDED"]


@pytest.mark.parametrize("fit_cap", [1, 2])
def test_seed_and_whole_component_fit_cap_clears_every_partial_region(fit_cap):
    result = analyze(grid(), max_plane_evaluations=fit_cap)
    scope(result)
    assert result["diagnostics"]["plane_evaluation_count"] == fit_cap
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["PLANE_EVALUATION_BUDGET_EXCEEDED"]


def test_final_region_cap_does_not_return_the_first_disconnected_patch_only():
    result = analyze(np.concatenate([grid(), grid(10.)]), max_regions=1)
    scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["FINAL_REGION_CANDIDATE_BUDGET_EXCEEDED"]


def test_exact_duplicate_and_row_permutation_invariance():
    reference = analyze(grid())
    repeated = analyze(np.repeat(grid()[::-1], 2, axis=0))
    scope(reference)
    scope(repeated)
    assert repeated["input_count"] == 18 and repeated["unique_count"] == 9
    assert repeated["positions"] == reference["positions"]
    assert repeated["queries"] == reference["queries"]
    assert repeated["regions"] == reference["regions"]
    assert repeated["diagnostics"]["seed_query_count"] == 18


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
def test_rigid_reexpression_and_unit_scale_preserve_complete_support(scale):
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    points = (grid() @ rotation.T + [2., -3., 1.])*scale
    result = analyze(points, unit="m" if scale == 1e-3 else "mm",
                     position_error_bound=1e-6*scale,
                     max_neighborhood_radius=1.1*scale, max_link_distance=1.1*scale)
    scope(result)
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(9))


@pytest.mark.parametrize("value", [2, 10, True, 3.0, None])
def test_cap_seed_minimum_must_be_explicit_integer_between_three_and_final_minimum(value):
    with pytest.raises(ValueError):
        analyze([], min_seed_points=value)


def test_raw_input_cap_applies_even_when_every_row_is_an_exact_repeat():
    with pytest.raises(ValueError, match="256"):
        analyze([[0., 0., 0.]]*257)


def test_finite_but_unrepresentable_pair_distance_is_incomplete_not_empty_geometry():
    result = analyze([[1e308, 0., 0.], [-1e308, 0., 0.], [0., 1e308, 0.]],
                     graph_neighbor_count=2)
    scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["GRAPH_ENVELOPE_INCOMPLETE"]


def test_empty_cloud_does_not_assert_an_empty_physical_part():
    result = analyze([])
    scope(result)
    assert result["search_complete_within_declared_policy"] is True
    assert result["queries"] == result["regions"] == []
    assert result["physical_region_count"] is None
