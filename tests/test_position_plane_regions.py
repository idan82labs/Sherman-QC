"""Pure synthetic checks; no native geometry, filesystem fixtures or runtime."""
import json
import math

import numpy as np
import pytest

from research.geometry.position_plane_regions import discover_position_plane_regions


def discover(points, **overrides):
    options = dict(unit="mm", position_error_bound=0.0, neighborhood_radii=[4.0],
                   max_normal_angle_bound=0.1, min_region_points=3, max_regions=64)
    options.update(overrides)
    return discover_position_plane_regions(points, **options)


def patches():
    # The discovery API receives only this unlabelled union of positions.
    first = [[x, y, 0.0] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
    second = [[30.0, y, z] for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)]
    return np.asarray(first + second)


def supports(result):
    return {tuple(region["point_indices"]) for region in result["regions"]}


def source_supports(result, source):
    canonical = np.asarray(result["positions"])
    identities = np.argmin(np.linalg.norm(canonical[:, None, :] - source[None, :, :], axis=2), axis=1)
    return {frozenset(int(identities[index]) for index in region["point_indices"])
            for region in result["regions"]}


def test_unlabelled_two_patch_discovery_exhausts_all_seeds_and_preserves_research_boundary():
    result = discover(patches())
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert result["search_complete_within_declared_policy"] is True
    assert result["input_count"] == result["unique_count"] == 18
    assert result["seed_query_count"] == 18
    assert result["unique_support_query_count"] == 2
    assert len(result["regions"]) == 2
    assert [region["region_id"] for region in result["regions"]] == [0, 1]
    assert sorted(sum((region["point_indices"] for region in result["regions"]), [])) == list(range(18))
    assert sum(len(region["seed_queries"]) for region in result["regions"]) == 18
    assert all(not region["plane_analysis"]["reasons"] for region in result["regions"])
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["policy"]["membership_is_physically_verified"] is False
    assert "NO_BEND_COUNT_FORMATION_RADIUS_THICKNESS_OR_CONFORMANCE_CLAIM" in result["limitations"]
    assert "OVERLAPPING_AND_NESTED_HYPOTHESES_NOT_DISTINCT_PHYSICAL_INSTANCES" in result["limitations"]
    json.dumps(result, allow_nan=False)


def test_input_permutation_and_radius_permutation_are_exactly_deterministic():
    points = patches()
    expected = discover(points, neighborhood_radii=[1.1, 4.0])
    permutation = np.random.default_rng(8274).permutation(len(points))
    actual = discover(points[permutation], neighborhood_radii=[4.0, 1.1])
    assert actual == expected


def test_exact_duplicates_only_change_input_count_not_supports_or_fits():
    points = patches()
    expected = discover(points)
    actual = discover(np.concatenate((points, points[[0, 0, 2, 9]])))
    assert actual["input_count"] == len(points) + 4
    actual["input_count"] = expected["input_count"]
    assert actual == expected


def test_all_overlaps_and_strict_subsets_are_retained_without_greedy_merging():
    points = [[x, y, 0.0] for x in range(3) for y in range(3)]
    result = discover(points, neighborhood_radii=[1.01, 4.0])
    regions = [set(region["point_indices"]) for region in result["regions"]]
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert len(regions) == 10  # Nine local supports and the single whole-patch support.
    assert any(first < second for first in regions for second in regions)
    assert any(first & second and not first <= second and not second <= first
               for first in regions for second in regions)
    assert result["policy"]["residual_trimming"] is False
    assert result["policy"]["greedy_removal_or_merging"] is False


def test_candidate_overflow_rejects_the_entire_result_instead_of_a_prefix():
    points = [[x, y, 0.0] for x in range(3) for y in range(3)]
    result = discover(points, neighborhood_radii=[1.01, 4.0], max_regions=9)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
    assert result["seed_query_count"] == 18
    assert result["unique_support_query_count"] == 10
    assert result["positions"] == sorted(points)


@pytest.mark.parametrize("points", [None, [], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], [[0, 0, 0]] * 4])
def test_missing_or_too_few_unique_positions_abstain_without_missing_bend_claim(points):
    result = discover(points)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["regions"] == []
    assert result["reasons"] == ["INSUFFICIENT_DISTINCT_POSITIONS"]
    json.dumps(result, allow_nan=False)


def test_collinear_positions_do_not_supply_plane_normals():
    result = discover([[i, 0, 0] for i in range(5)], neighborhood_radii=[10.0])
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["reasons"] == ["NO_PLANE_REGION_HYPOTHESIS_WITHIN_DECLARED_POLICY"]


def test_nonplanar_all_point_support_is_not_residual_trimmed():
    points = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0], [0, 0, 0.2]]
    result = discover(points, position_error_bound=0.001, neighborhood_radii=[10.0])
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["unique_support_query_count"] == 1
    assert result["regions"] == []


def test_explicit_normal_precision_limit_is_enforced():
    points = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    loose = discover(points, position_error_bound=0.01, max_normal_angle_bound=0.02)
    tight = discover(points, position_error_bound=0.01, max_normal_angle_bound=0.005)
    assert loose["status"] == "PLANE_REGION_HYPOTHESES"
    assert loose["regions"][0]["plane_analysis"]["normal_angle_bound_radians"] > 0.005
    assert tight["status"] == "ABSTAINED"
    assert tight["search_complete_within_declared_policy"] is True


def test_boundary_diagnostics_count_all_points_near_radius_not_just_support_members():
    points = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1.01, 0, 0], [1, 1, 0]]
    result = discover(points, position_error_bound=0.01, neighborhood_radii=[1.0])
    queries = [(region, query) for region in result["regions"] for query in region["seed_queries"]
               if query["seed_index"] == 0]
    assert len(queries) == 1
    region, query = queries[0]
    assert region["point_indices"] == [0, 1, 2]
    assert query["boundary_sensitive_count"] == 3  # The 1.01 point is outside but unstable.
    assert result["distance_numerical_allowance"] > 0


def test_exact_radius_membership_uses_the_disclosed_numerical_guard():
    result = discover([[0, 0, 0], [0, 1, 0], [1, 0, 0]], neighborhood_radii=[1.0])
    region = next(region for region in result["regions"] if region["point_indices"] == [0, 1, 2])
    query = next(query for query in region["seed_queries"] if query["seed_index"] == 0)
    assert query["boundary_sensitive_count"] == 2


def test_short_cylindrical_patch_can_be_plane_compatible_but_is_not_a_verified_flat():
    radius = 10.0
    angles = np.linspace(-0.01, 0.01, 5)
    points = [[radius * math.sin(theta), y, radius * (1 - math.cos(theta))]
              for theta in angles for y in (-0.1, 0.0, 0.1)]
    result = discover(points, position_error_bound=0.001, neighborhood_radii=[1.0],
                      max_normal_angle_bound=0.05)
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert len(result["regions"]) == 1
    analysis = result["regions"][0]["plane_analysis"]
    assert analysis["plane_witness"]["max_residual"] > 0
    assert analysis["plane_witness"]["compatible_with_error_bound"] is True
    assert "PLANE_COMPATIBLE_SAMPLES_MAY_LIE_ON_CURVED_OR_DISCONNECTED_SURFACES" in result["limitations"]
    assert result["policy"]["membership_is_physically_verified"] is False


def test_se3_equivariance_away_from_search_boundaries():
    points = patches()
    matrix = np.asarray([[0.4, 1.1, -0.2], [0.9, -0.7, 0.3], [0.2, 0.5, 1.3]])
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    transformed = points @ rotation.T + [13.0, -7.0, 4.0]
    original_result = discover(points, position_error_bound=1e-5)
    transformed_result = discover(transformed, position_error_bound=1e-5)
    assert transformed_result["status"] == "PLANE_REGION_HYPOTHESES"
    assert source_supports(original_result, points) == source_supports(transformed_result, transformed)
    for original_region in original_result["regions"]:
        original_support = frozenset(np.argmin(np.linalg.norm(
            np.asarray(original_result["positions"])[original_region["point_indices"], None, :]
            - points[None, :, :], axis=2), axis=1))
        corresponding = next(region for region in transformed_result["regions"]
                             if source_supports({"positions": transformed_result["positions"],
                                                "regions": [region]}, transformed) == {original_support})
        normal = np.asarray(original_region["plane_analysis"]["plane_witness"]["normal"])
        rotated_normal = np.asarray(corresponding["plane_analysis"]["plane_witness"]["normal"])
        assert abs(float((rotation @ normal) @ rotated_normal)) == pytest.approx(1.0, abs=1e-12)


def test_declared_unit_change_preserves_supports_and_scales_distance_budgets():
    points = patches()
    in_meters = discover(points, unit="m", position_error_bound=1e-5)
    in_mm = discover(points * 1000, unit="mm", position_error_bound=0.01,
                     neighborhood_radii=[4000.0])
    assert supports(in_meters) == supports(in_mm)
    assert in_mm["distance_numerical_allowance"] == pytest.approx(
        1000 * in_meters["distance_numerical_allowance"], rel=1e-13)
    for first, second in zip(in_meters["regions"], in_mm["regions"]):
        assert first["plane_analysis"]["normal_angle_bound_radians"] == pytest.approx(
            second["plane_analysis"]["normal_angle_bound_radians"], abs=1e-12)
        assert second["plane_analysis"]["effective_error_bound"] == pytest.approx(
            first["plane_analysis"]["effective_error_bound"] * 1000)


def test_fixed_seed_1000_case_position_only_rotation_scale_and_permutation_sweep():
    rng = np.random.default_rng(906072)
    points = patches()
    expected = {frozenset(range(9)), frozenset(range(9, 18))}
    for _ in range(1000):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        scale = 10. ** rng.uniform(-6, 6)
        transformed = (points @ rotation.T + rng.uniform(-10, 10, size=3)) * scale
        result = discover(transformed[rng.permutation(18)],
                          position_error_bound=1e-6 * scale,
                          neighborhood_radii=[3.1 * scale, 4. * scale, 5. * scale],
                          max_normal_angle_bound=0.01, min_region_points=6, max_regions=2)
        assert result["status"] == "PLANE_REGION_HYPOTHESES"
        assert result["search_complete_within_declared_policy"] is True
        assert result["seed_query_count"] == 54
        assert len(result["regions"]) == 2
        assert source_supports(result, transformed) == expected
        assert sum(len(region["seed_queries"]) for region in result["regions"]) == 54
        json.dumps(result, allow_nan=False)


class DoNotConvert(list):
    def __iter__(self):
        raise AssertionError("oversized payload must not be iterated or converted")


def test_input_cap_is_enforced_before_conversion_even_for_duplicates():
    with pytest.raises(ValueError, match="at most 256"):
        discover(DoNotConvert([object()] * 257))


def test_radius_cap_is_enforced_before_element_conversion():
    with pytest.raises(ValueError, match="one to four"):
        discover(patches(), neighborhood_radii=DoNotConvert([object()] * 5))


@pytest.mark.parametrize("overrides", [
    {"unit": "cm"}, {"unit": True},
    {"position_error_bound": -1}, {"position_error_bound": True},
    {"position_error_bound": float("nan")}, {"position_error_bound": float("inf")},
    {"position_error_bound": 10 ** 1000},
    {"neighborhood_radii": []}, {"neighborhood_radii": [1, 1.0]},
    {"neighborhood_radii": [0]}, {"neighborhood_radii": [-1]},
    {"neighborhood_radii": [True]}, {"neighborhood_radii": [float("inf")]},
    {"neighborhood_radii": [float("nan")]}, {"neighborhood_radii": [[1]]},
    {"neighborhood_radii": np.ones((1, 1))}, {"neighborhood_radii": (x for x in [1])},
    {"max_normal_angle_bound": 0}, {"max_normal_angle_bound": -1},
    {"max_normal_angle_bound": True}, {"max_normal_angle_bound": math.pi / 2},
    {"max_normal_angle_bound": float("inf")}, {"max_normal_angle_bound": float("nan")},
    {"min_region_points": 2}, {"min_region_points": 257},
    {"min_region_points": True}, {"min_region_points": 3.0},
    {"max_regions": 0}, {"max_regions": 65},
    {"max_regions": True}, {"max_regions": 1.0},
])
def test_invalid_explicit_policies_raise(overrides):
    with pytest.raises(ValueError):
        discover(patches(), **overrides)


@pytest.mark.parametrize("points", [
    [[0, 0]], [[0, 0, 0, 0]], [[True, 0, 0]], [["0", 0, 0]],
    [[float("nan"), 0, 0]], [[float("inf"), 0, 0]],
    np.ones((3, 4)), np.ones(3), [[10 ** 1000, 0, 0]],
    (point for point in [[0, 0, 0]]),
])
def test_invalid_position_payloads_raise(points):
    with pytest.raises(ValueError):
        discover(points)


@pytest.mark.parametrize("points,epsilon", [
    ([[-1e308, 0, 0], [1e308, 0, 0], [0, 1e308, 0]], 0.0),
    ([[0, 0, 0], [1e-320, 0, 0], [0, 1e-320, 0]], 0.0),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e308),
])
def test_unsupported_numerical_range_abstains_with_finite_json(points, epsilon):
    result = discover(points, position_error_bound=epsilon)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["regions"] == []
    assert result["reasons"] == ["NUMERICAL_RANGE_UNSUPPORTED"]
    json.dumps(result, allow_nan=False)


def test_large_but_supported_coordinates_avoid_naive_squared_distance_overflow():
    points = np.asarray([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]) * 1e150
    result = discover(points, neighborhood_radii=[4e150])
    assert result["status"] == "PLANE_REGION_HYPOTHESES"
    assert len(result["regions"]) == 1
    json.dumps(result, allow_nan=False)


def test_numerically_unavailable_plane_fit_cannot_leave_a_partial_result(monkeypatch):
    import research.geometry.position_plane_regions as module
    original = module._fit_plane
    calls = []

    def unavailable_after_one(points, epsilon):
        calls.append(len(points))
        if len(calls) == 2:
            return {"reasons": ["PLANE_FIT_UNAVAILABLE"]}
        return original(points, epsilon)

    monkeypatch.setattr(module, "_fit_plane", unavailable_after_one)
    result = discover(patches())
    assert len(calls) == 2
    assert result["status"] == "ABSTAINED"
    assert result["regions"] == []
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["REGION_PLANE_ANALYSIS_UNAVAILABLE"]
