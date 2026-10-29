"""Invented bounded-error cone proposals; no physical inputs or calibration."""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.geometry import cone_closed_plane_regions as module
from research.geometry.position_axis_hypothesis import _fit_plane
from research.geometry.position_bend_candidates import _fixed_plane_membership


def analyze(points, **changes):
    arguments = dict(unit="mm", position_error_bound=1e-6, neighbor_counts=[8],
                     max_neighborhood_radius=1.5, max_normal_angle_bound=0.1,
                     min_region_points=9, max_regions=64, graph_neighbor_count=10,
                     max_link_distance=1.5, max_plane_evaluations=2048)
    arguments.update(changes)
    return module.discover_cone_closed_plane_regions(points, **arguments)


def plane_grid(z=0, xshift=0):
    return np.array([[x+xshift, y, z] for x in np.linspace(-1, 1, 5)
                     for y in np.linspace(-1, 1, 5)])


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_region_count"] is None
    assert result["policy"]["region_budget_scope"] == "FINAL_REFITTED_EXACT_SUPPORTS_NOT_RAW_SEEDS"
    assert result["policy"]["different_budget_unit_from_adaptive_v1"] is True
    assert result["policy"]["final_error_bound_is_relaxed"] is False
    assert result["policy"]["fallback_to_seed_after_closure_failure"] is False
    assert result["diagnostics"]["plane_evaluation_count"] <= result["policy"]["max_plane_evaluations"]
    supports = [tuple(region["point_indices"]) for region in result["regions"]]
    assert len(set(supports)) == len(supports) <= result["policy"]["max_regions"]
    p = np.asarray(result["positions"])
    for region in result["regions"]:
        assert region["physical_region_is_verified"] is False
        plane = region["plane_analysis"]
        assert plane["reasons"] == []
        assert plane["effective_error_bound"] == result["position_error_bound"] + plane["numerical_allowance"]
        witness = plane["plane_witness"]
        residual = np.abs((p[region["point_indices"]]-witness["reference_point"]) @ witness["normal"]
                          - witness["offset_from_reference"])
        assert np.max(residual) <= plane["effective_error_bound"]


def test_opposed_bounded_offset_noise_requires_centroid_uncertainty_not_fixed_plane():
    epsilon = 0.001
    seed = plane_grid(epsilon)
    points = np.concatenate([seed, plane_grid(-epsilon)])
    plane = _fit_plane(seed, epsilon)
    fixed = _fixed_plane_membership(points, plane, epsilon, 0)
    assert fixed[:25].all() and not fixed[25:].any()
    possible, diagnostic = module._cone_membership(points, tuple(range(25)), plane, epsilon, 0)
    assert possible.all()
    assert diagnostic["proposal_distance_bound"] >= 2*epsilon
    assert diagnostic["joint_compatibility_is_verified"] is False
    union = _fit_plane(points, epsilon)
    assert union["plane_witness"]["compatible_with_error_bound"] is True


def test_pointwise_possible_cone_union_still_needs_original_error_joint_refit():
    epsilon = 0.001
    seed = plane_grid()
    points = np.concatenate([seed, plane_grid(2*epsilon), plane_grid(-2*epsilon)])
    plane = _fit_plane(seed, epsilon)
    possible, diagnostic = module._cone_membership(points, tuple(range(25)), plane, epsilon, 0)
    assert possible.all()
    assert diagnostic["joint_compatibility_is_verified"] is False
    refit = _fit_plane(points, epsilon)
    assert "NO_ALL_POINT_PLANE_WITNESS" in refit["reasons"]
    result = analyze(points, position_error_bound=epsilon, graph_neighbor_count=74,
                     max_link_distance=4, neighbor_counts=[24], max_neighborhood_radius=4)
    scope(result)
    assert all(len(region["point_indices"]) < len(points) for region in result["regions"])


def test_normal_uncertainty_can_retain_distant_true_plane_samples():
    epsilon = 0.001
    seed = plane_grid()
    seed[:, 2] = 0.0005*seed[:, 0]
    assert np.max(np.abs(seed[:, 2])) <= epsilon
    points = np.concatenate([seed, [[20, 0, 0]]])
    plane = _fit_plane(seed, epsilon)
    assert not _fixed_plane_membership(points, plane, epsilon, 0)[-1]
    possible, _ = module._cone_membership(points, tuple(range(25)), plane, epsilon, 0)
    assert possible.all()


@pytest.mark.parametrize("seed", range(24))
def test_cone_gate_contains_true_plane_points_under_correlated_bounded_noise(seed):
    rng = np.random.default_rng(seed)
    epsilon = 1e-4
    truth = np.concatenate([plane_grid(), np.array([[20, 2, 0], [-20, -2, 0]])])
    directions = rng.normal(size=truth.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    noise = directions * rng.uniform(0.1, 0.9, size=(len(truth), 1))*epsilon
    # A shared bias remains deterministic bounded noise, not an averaging gain.
    noise = 0.5*noise + [0, 0, 0.45*epsilon]
    assert np.max(np.linalg.norm(noise, axis=1)) <= epsilon
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    points = (truth+noise) @ rotation.T + rng.uniform(-10, 10, size=3)
    plane = _fit_plane(points[:25], epsilon)
    assert plane["normal_angle_bound_radians"] is not None
    possible, _ = module._cone_membership(points, tuple(range(25)), plane, epsilon, 0)
    assert possible.all()


def test_one_noisy_plane_yields_one_exact_deduplicated_whole_support():
    rng = np.random.default_rng(193)
    points = plane_grid()
    points += rng.uniform(-1, 1, size=points.shape)*1e-7
    result = analyze(points)
    scope(result)
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(25))
    assert result["diagnostics"]["duplicate_closed_support_count"] > 0
    assert len(result["regions"][0]["seed_queries"]) > 1


def test_exact_permutations_and_repetitions_keep_same_proposals():
    points = np.concatenate([plane_grid(), plane_grid(0, 6)])
    first = analyze(points)
    second = analyze(np.repeat(points[::-1], 2, axis=0))
    scope(first)
    scope(second)
    assert first["positions"] == second["positions"]
    assert first["regions"] == second["regions"]
    assert len(first["regions"]) == 2
    assert first["physical_region_count"] is None


@pytest.mark.parametrize("seed", [193, 521, 8003])
def test_rigid_transform_preserves_whole_sample_components(seed):
    points = np.concatenate([plane_grid(), plane_grid(0, 6)])
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    transformed = points @ rotation.T + rng.uniform(-10, 10, size=3)
    result = analyze(transformed)
    scope(result)
    lookup = {tuple(row): index for index, row in enumerate(transformed)}
    supports = {frozenset(lookup[tuple(result["positions"][index])] for index in region["point_indices"])
                for region in result["regions"]}
    assert supports == {frozenset(range(25)), frozenset(range(25, 50))}


def test_graph_connected_distinct_coplanar_objects_are_not_physical_identity():
    points = np.concatenate([plane_grid(), plane_grid(0, 2.5)])
    result = analyze(points)
    scope(result)
    assert len(result["regions"]) == 1
    assert len(result["regions"][0]["point_indices"]) == 50
    assert "NEARBY_COPLANAR_OBJECTS_CAN_SHARE_ONE_SAMPLE_WITNESS" in result["limitations"]


@pytest.mark.parametrize("scale", [1e-3, 1, 1e3])
def test_length_policies_and_plane_bounds_scale_together(scale):
    points = plane_grid()*scale
    result = analyze(points, unit="m" if scale == 1e-3 else "mm", position_error_bound=1e-6*scale,
                     max_neighborhood_radius=1.5*scale, max_link_distance=1.5*scale)
    scope(result)
    assert len(result["regions"]) == 1
    assert result["regions"][0]["point_indices"] == list(range(25))
    assert_allclose(result["regions"][0]["plane_analysis"]["plane_witness"]["normal"], [0, 0, 1])


def test_final_region_overflow_clears_all_previously_found_regions():
    result = analyze(np.concatenate([plane_grid(), plane_grid(0, 6)]), max_regions=1)
    scope(result)
    assert result["reasons"] == ["FINAL_REGION_CANDIDATE_BUDGET_EXCEEDED"]
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False


def test_fit_budget_clears_successful_prefix_and_counts_seed_and_refit_work():
    result = analyze(plane_grid(), max_plane_evaluations=2)
    scope(result)
    assert result["diagnostics"]["plane_evaluation_count"] == 2
    assert result["reasons"] == ["PLANE_EVALUATION_BUDGET_EXCEEDED"]
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False


def test_numerically_unavailable_refit_clears_candidate_prefix(monkeypatch):
    original = module._fit_plane
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            return {"reasons": ["PLANE_FIT_UNAVAILABLE"]}
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_plane", fail_second)
    result = analyze(plane_grid())
    scope(result)
    assert result["regions"] == []
    assert result["reasons"] == ["PLANE_ANALYSIS_UNAVAILABLE"]
    assert result["search_complete_within_declared_policy"] is False


def test_input_cap_precedes_numpy_conversion_even_for_repeated_rows(monkeypatch):
    monkeypatch.setattr(np, "asarray", lambda *args, **kwargs: pytest.fail("cap checked too late"))
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 0]]*257)


@pytest.mark.parametrize("field,value", [
    ("max_regions", 65), ("max_regions", True), ("max_plane_evaluations", 2049),
    ("max_plane_evaluations", 0), ("max_plane_evaluations", 2.0),
    ("max_normal_angle_bound", math.pi/2), ("neighbor_counts", [8, 8]),
    ("graph_neighbor_count", 1), ("max_link_distance", 0),
    ("unit", "cm"), ("position_error_bound", -1),
])
def test_policy_validation_precedes_empty_input_return(field, value):
    with pytest.raises(ValueError):
        analyze([], **{field: value})


def test_empty_input_is_not_verified_empty_part():
    result = analyze([])
    scope(result)
    assert result["search_complete_within_declared_policy"] is True
    assert result["status"] == "ABSTAINED"
