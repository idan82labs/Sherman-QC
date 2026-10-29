"""Invented unlabeled positions for adaptive integration; no physical inputs.

The policies are explicit development settings, not scanner calibration. Fixture
construction labels are retained only in test assertions and never inference.
"""
import inspect
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.geometry.adaptive_bend_candidates import analyze_adaptive_bend_candidates
from research.geometry.position_bend_candidates import analyze_position_bend_candidates


def synthetic_bend(angle_degrees=90, radius=2.0, *, flange_extent=2.0, mixture=False):
    angle = math.radians(angle_degrees)
    sign = math.copysign(1, angle)
    axial = radius * np.linspace(-0.5, 0.5, 5)
    stations = radius * np.linspace(0, flange_extent, 5)
    first = np.array([[-sign * distance, y, radius] for distance in stations for y in axial])
    second = np.array([
        [radius * math.sin(angle) + sign * distance * math.cos(angle), y,
         radius * math.cos(angle) - sign * distance * math.sin(angle)]
        for distance in stations for y in axial
    ])
    inner = np.array([
        [radius * math.sin(j * angle / 8), y, radius * math.cos(j * angle / 8)]
        for j in range(1, 8) for y in axial
    ])
    transition = np.concatenate([inner, inner * [1.1, 1, 1.1]]) if mixture else inner
    return first, second, transition


def analyze(points=None, *, scale=1.0, **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6 * scale, neighbor_counts=[8],
        max_neighborhood_radius=1.5 * scale, max_normal_angle_bound=0.1,
        min_region_points=9, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5 * scale, max_axis_angle_bound=0.1,
        max_axial_gap=0.51 * scale, max_radius_width=0.1 * scale,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255,
    )
    policy.update(changes)
    points = np.concatenate(synthetic_bend()) * scale if points is None else points
    return analyze_adaptive_bend_candidates(points, **policy)


def point_set(points):
    return frozenset(map(tuple, np.round(np.asarray(points), 9)))


def assert_scope(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    assert result["graph"]["scope"] == "MUTUAL_OBSERVED_NEIGHBORHOOD_SAMPLED_LINKS_ONLY"
    assert result["graph"]["epsilon_expands_edges"] is False
    for candidate in result["candidates"]:
        association = candidate["association"]
        assert association["scope"] == "MUTUAL_OBSERVED_NEIGHBORHOOD_GRAPH_ONLY"
        assert association["entire_remainder_component_retained"] is True
        assert association["observed_links_establish_true_gap"] is False
        for key in ("physical_connectivity_is_verified", "joint_graph_realizability_is_verified",
                    "tangency_is_verified", "same_object_is_verified"):
            assert association[key] is False
        assert candidate["single_radius_analysis"]["original_joint_compatibility"] == "UNESTABLISHED"
        if candidate["two_radius_analysis"] is not None:
            assert candidate["two_radius_analysis"]["original_joint_compatibility"] == "UNESTABLISHED"
            assert candidate["two_radius_analysis"]["thickness"] is None
            assert candidate["two_radius_analysis"]["physical_face_count"] is None


def assert_whole_component(result, candidate):
    """Recover the induced graph independently from the serialized edge list."""
    regions = {region["region_id"]: set(region["point_indices"])
               for region in result["extended_regions"]}
    first, second = [regions[value] for value in candidate["plane_region_ids"]]
    support = set(candidate["transition_point_indices"])
    assert first.isdisjoint(second)
    assert support.isdisjoint(first | second)
    remaining = set(range(result["unique_count"])) - first - second
    adjacency = {index: set() for index in range(result["unique_count"])}
    for one, two in result["graph"]["edges"]:
        adjacency[one].add(two)
        adjacency[two].add(one)
    reached, frontier = set(), {min(support)}
    while frontier:
        reached.update(frontier)
        frontier = set.union(*(adjacency[index] for index in frontier)) & remaining - reached
    assert reached == support
    for region, count in zip((first, second), candidate["association"]["plane_observed_graph_edge_counts"]):
        assert sum(len(adjacency[index] & region) for index in support) == count > 0


def test_public_input_is_positions_and_explicit_sampling_and_resolution_policies_only():
    signature = inspect.signature(analyze_adaptive_bend_candidates)
    assert set(signature.parameters) == {
        "points", "unit", "position_error_bound", "neighbor_counts", "max_neighborhood_radius",
        "max_normal_angle_bound", "min_region_points", "max_regions", "graph_neighbor_count",
        "max_link_distance", "max_axis_angle_bound", "max_axial_gap", "max_radius_width",
        "max_pair_evaluations", "max_bridge_evaluations", "min_band_points", "max_split_evaluations",
    }
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY
               for name, parameter in signature.parameters.items() if name != "points")


def test_sparse_flange_birth_improves_named_v1_failure_without_editing_v1_policy():
    first, second, transition = synthetic_bend()
    points = np.concatenate((first, second, transition))
    old = analyze_position_bend_candidates(
        points, unit="mm", position_error_bound=1e-6, neighborhood_radii=[0.8],
        max_normal_angle_bound=0.1, min_region_points=9, max_regions=64,
        sample_gap=0.81, max_axis_angle_bound=0.1, max_axial_gap=0.51,
        max_radius_width=0.1, max_pair_evaluations=2016, max_bridge_evaluations=256,
    )
    assert old["plane_discovery"]["regions"] == []
    assert old["status"] == "ABSTAINED"
    result = analyze(points)
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["extended_regions"]) == 2
    canonical = np.asarray(result["positions"])
    assert {point_set(canonical[region["point_indices"]]) for region in result["extended_regions"]} == {
        point_set(first), point_set(second),
    }
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"
    assert candidate["two_radius_analysis"] is None
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 0
    assert point_set(canonical[candidate["transition_point_indices"]]) == point_set(transition)
    assert candidate["single_radius_analysis"]["radius_interval"][0] <= 2 <= candidate["single_radius_analysis"]["radius_interval"][1]
    assert_whole_component(result, candidate)


def test_mixed_complete_component_falls_back_to_two_radius_without_trimming():
    first, second, transition = synthetic_bend(mixture=True)
    result = analyze(np.concatenate((first, second, transition)))
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert_whole_component(result, candidate)
    assert candidate["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"
    assert candidate["single_radius_analysis"]["status"] == "ABSTAINED"
    assert len(candidate["transition_point_indices"]) == 70
    analysis = candidate["two_radius_analysis"]
    assert analysis["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert len(analysis["models"]) == 1
    model = analysis["models"][0]
    assert_allclose(model["witness_radii"], [2, 2.2], atol=1e-10, rtol=0)
    assert model["all_points_accounted_for"] is True
    memberships = [set(band["point_indices"]) for band in model["bands"]]
    assert memberships[0].isdisjoint(memberships[1])
    assert memberships[0] | memberships[1] == set(range(70))
    for band, radius in zip(model["bands"], (2, 2.2)):
        assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
        assert band["membership_is_verified"] is False
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 1
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 1


def test_denser_mixture_can_still_fragment_smaller_mutual_neighborhood_policy():
    # Recorded development failure: choosing k=8 rather than 10 changes mutual
    # graph seam links. Adaptive neighborhoods are NOT density invariance.
    points = np.concatenate(synthetic_bend(mixture=True))
    result = analyze(points, graph_neighbor_count=8)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["extended_regions"]) == 2
    assert result["search_diagnostics"]["association_count"] == 0
    assert result["candidates"] == []


def test_connected_unexplained_contaminant_remains_and_both_models_abstain():
    points = np.concatenate(synthetic_bend())
    contaminant = [[2.4 / math.sqrt(2), 0, 2.4 / math.sqrt(2)]]
    result = analyze(np.concatenate([points, contaminant]))
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert len(candidate["transition_point_indices"]) == 36
    assert_whole_component(result, candidate)
    assert point_set(contaminant) <= point_set(np.asarray(result["positions"])[candidate["transition_point_indices"]])
    assert candidate["single_radius_analysis"]["status"] == "ABSTAINED"
    assert candidate["two_radius_analysis"]["status"] == "ABSTAINED"
    assert candidate["status"] == "ABSTAINED"


@pytest.mark.parametrize("seed", [0, 17, 20260907])
def test_canonical_order_permutation_and_repeats_preserve_complete_results(seed):
    points = np.concatenate(synthetic_bend(mixture=True))
    reference = analyze(points)
    permuted = points[np.random.default_rng(seed).permutation(len(points))]
    assert analyze(permuted) == reference
    repeated = analyze(np.concatenate([permuted, permuted]))
    assert repeated["input_count"] == 240
    for key in ("positions", "extended_regions", "candidates", "search_diagnostics", "status"):
        assert repeated[key] == reference[key]


@pytest.mark.parametrize("mixture", [False, True])
def test_rigid_motion_and_units_preserve_constructed_geometry_not_axis_location(mixture):
    points = np.concatenate(synthetic_bend(mixture=mixture))
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, -1], [0.4, 3, 0.7], [0.5, -1, 4]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = np.array([4, -5, 7])
    result = analyze((points @ rotation.T + translation) / 1000, scale=0.001, unit="m")
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    candidate = result["candidates"][0]
    restored = (np.asarray(result["positions"]) * 1000 - translation) @ rotation
    assert point_set(restored[candidate["transition_point_indices"]]) == point_set(synthetic_bend(mixture=mixture)[2])
    if mixture:
        bands = candidate["two_radius_analysis"]["models"][0]["bands"]
        for band, radius in zip(bands, (0.002, 0.0022)):
            assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
    else:
        bounds = candidate["single_radius_analysis"]["radius_interval"]
        assert bounds[0] <= 0.002 <= bounds[1]


@pytest.mark.parametrize("changes,reason", [
    ({"max_regions": 31}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_pair_evaluations": 5}, "PAIR_EVALUATION_BUDGET_EXCEEDED"),
    ({"max_bridge_evaluations": 1}, "BRIDGE_EVALUATION_BUDGET_EXCEEDED"),
])
def test_entire_association_budget_preflights_before_either_radius_call(changes, reason):
    points = np.concatenate(synthetic_bend())
    doubled = np.concatenate([points, points + [15, 0, 0]])
    result = analyze(doubled, **changes)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 0
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 0
    assert result["reasons"] == [reason]


def test_split_budget_exhaustion_never_releases_a_successful_candidate_prefix():
    points = np.concatenate(synthetic_bend(mixture=True))
    result = analyze(points, max_split_evaluations=68)
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["TWO_RADIUS_SEARCH_INCOMPLETE"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 1
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 1


def test_later_mixed_split_exhaustion_clears_an_earlier_single_radius_success():
    points = np.concatenate([
        np.concatenate(synthetic_bend()),
        np.concatenate(synthetic_bend(mixture=True)) + [15, 0, 0],
    ])
    result = analyze(points, max_split_evaluations=68)
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["TWO_RADIUS_SEARCH_INCOMPLETE"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["association_count"] == 2
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 2
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 1


def test_real_overflowing_pair_distances_produce_finite_incomplete_result():
    points = [[-1.7e308, 0, 0], [1.7e308, 0, 0], [0, 1.7e308, 0]]
    result = analyze(points, graph_neighbor_count=2)
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["reasons"] == ["PLANE_DISCOVERY_INCOMPLETE"]
    assert result["graph"]["reasons"] == ["NUMERICAL_RANGE_UNSUPPORTED"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["bridge_evaluation_count"] == 0


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("position_error_bound", -1), ("position_error_bound", True),
    ("neighbor_counts", []), ("neighbor_counts", [True]), ("neighbor_counts", [1]),
    ("neighbor_counts", [8, 8]), ("neighbor_counts", [256]),
    ("max_neighborhood_radius", 0), ("max_neighborhood_radius", math.inf),
    ("max_normal_angle_bound", math.pi / 2), ("min_region_points", 2), ("max_regions", 65),
    ("graph_neighbor_count", 1), ("graph_neighbor_count", True), ("graph_neighbor_count", 256),
    ("max_link_distance", 0), ("max_link_distance", math.nan),
    ("max_axis_angle_bound", 0), ("max_axis_angle_bound", math.pi / 2),
    ("max_axial_gap", -1), ("max_radius_width", "1"),
    ("max_pair_evaluations", 0), ("max_pair_evaluations", 2017),
    ("max_bridge_evaluations", 257), ("max_bridge_evaluations", False),
    ("min_band_points", 2), ("min_band_points", 129), ("min_band_points", 3.5),
    ("max_split_evaluations", 0), ("max_split_evaluations", 256), ("max_split_evaluations", "255"),
])
def test_all_policies_are_validated_even_when_cloud_is_empty(field, value):
    with pytest.raises(ValueError):
        analyze([], **{field: value})


def test_raw_input_cap_is_before_deduplication_and_array_protocol():
    class Oversize(list):
        def __len__(self):
            return 257

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversize input array conversion attempted")

    with pytest.raises(ValueError, match="256"):
        analyze(Oversize())
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 0]] * 257)


@pytest.mark.parametrize("points", [[], np.empty((0, 3)), [[0, 0, 0]], [[1, 2, 3], [1, 2, 3]]])
def test_insufficient_distinct_positions_do_not_establish_an_empty_physical_part(points):
    result = analyze(points)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["candidates"] == []


def run_adaptive_structural_matrix():
    """150 fixed-policy clean/hole/mixed/sparse/combined cases, no per-angle tuning."""
    counts = {name: dict(cases=0, positive=0, abstained=0, incomplete=0,
                         single_candidates=0, two_candidates=0,
                         matched_single_containment_checks=0, matched_two_containment_checks=0,
                         evaluated_two_bands=0, proposed_two_band_intervals=0,
                         unmatched_two_band_intervals=0)
              for name in ("clean", "missing_middle_arc", "concentric_mixture",
                           "sparse_flange_stations", "sparse_concentric_mixture")}
    by_angle = {name: {angle: dict(positive=0, abstained=0) for angle in (30, 60, 90, 120, 150)}
                for name in counts}
    for name, current in counts.items():
        for angle in (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150):
            for radius in (0.5, 2, 20):
                first, second, transition = synthetic_bend(
                    angle, radius, flange_extent=2 if name.startswith("sparse_") else 1,
                    mixture=name.endswith("concentric_mixture"),
                )
                if name == "missing_middle_arc":
                    transition = np.concatenate([transition[:15], transition[20:]])
                result = analyze(np.concatenate([first, second, transition]), scale=radius / 2)
                assert_scope(result)
                current["cases"] += 1
                outcome = "positive" if result["status"] == "POSITION_GEOMETRY_CANDIDATES" else "abstained"
                current[outcome] += 1
                by_angle[name][abs(angle)][outcome] += 1
                current["incomplete"] += int(not result["search_complete_within_declared_policy"])
                canonical = np.asarray(result["positions"])
                regions = {region["region_id"]: point_set(canonical[region["point_indices"]])
                           for region in result["extended_regions"]}
                for candidate in result["candidates"]:
                    assert_whole_component(result, candidate)
                    if candidate["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS":
                        current["single_candidates"] += 1
                    elif candidate["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS":
                        current["two_candidates"] += 1
                    two_analysis = candidate["two_radius_analysis"]
                    if two_analysis is not None:
                        current["evaluated_two_bands"] += two_analysis["search_diagnostics"]["band_evaluation_count"]
                        current["proposed_two_band_intervals"] += sum(
                            len(model["bands"]) for model in two_analysis["models"]
                        )
                    if {regions[index] for index in candidate["plane_region_ids"]} != {point_set(first), point_set(second)}:
                        continue
                    support = canonical[candidate["transition_point_indices"]]
                    if point_set(support) != point_set(transition):
                        continue
                    if not name.endswith("concentric_mixture"):
                        lower, upper = candidate["single_radius_analysis"]["radius_interval"]
                        assert lower <= radius and (upper is None or upper >= radius)
                        current["matched_single_containment_checks"] += 1
                    elif candidate["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS":
                        analysis = candidate["two_radius_analysis"]
                        mixed = np.asarray(analysis["mixed_positions"])
                        for model in analysis["models"]:
                            for band in model["bands"]:
                                actual = np.hypot(mixed[band["point_indices"], 0], mixed[band["point_indices"], 2])
                                # Check oracle correspondence before interpreting a
                                # proposed-membership interval as a true-R bound.
                                if np.ptp(actual) <= 1e-10 * radius:
                                    truth = radius if abs(actual[0] - radius) < 1e-8 * radius else 1.1 * radius
                                    lower, upper = band["radius_interval"]
                                    assert lower <= truth and (upper is None or upper >= truth)
                                    current["matched_two_containment_checks"] += 1
        current["unmatched_two_band_intervals"] = (
            current["proposed_two_band_intervals"] - current["matched_two_containment_checks"]
        )
    return {"cases": sum(item["cases"] for item in counts.values()), "strata": counts,
            "outcomes_by_absolute_angle": by_angle}


def test_fixed_adaptive_structural_matrix_retains_stratified_receipt():
    receipt = run_adaptive_structural_matrix()
    assert receipt["cases"] == 150
    common = dict(cases=30, incomplete=0, unmatched_two_band_intervals=0)
    assert receipt["strata"] == {
        "clean": dict(common, positive=30, abstained=0, single_candidates=30, two_candidates=0,
                      matched_single_containment_checks=30, matched_two_containment_checks=0,
                      evaluated_two_bands=0, proposed_two_band_intervals=0),
        "missing_middle_arc": dict(common, positive=30, abstained=0, single_candidates=42,
                                   two_candidates=0, matched_single_containment_checks=30,
                                   matched_two_containment_checks=0,
                                   evaluated_two_bands=0, proposed_two_band_intervals=0),
        "concentric_mixture": dict(common, positive=30, abstained=0, single_candidates=0,
                                   two_candidates=30, matched_single_containment_checks=0,
                                   matched_two_containment_checks=60,
                                   evaluated_two_bands=60, proposed_two_band_intervals=60),
        "sparse_flange_stations": dict(common, positive=24, abstained=6, single_candidates=24,
                                      two_candidates=0, matched_single_containment_checks=24,
                                      matched_two_containment_checks=0,
                                      evaluated_two_bands=0, proposed_two_band_intervals=0),
        "sparse_concentric_mixture": dict(common, positive=18, abstained=12, single_candidates=0,
                                         two_candidates=18, matched_single_containment_checks=0,
                                         matched_two_containment_checks=36,
                                         evaluated_two_bands=36, proposed_two_band_intervals=36),
    }
    for name, angles in receipt["outcomes_by_absolute_angle"].items():
        for angle, outcome in angles.items():
            fails = (name == "sparse_flange_stations" and angle == 30
                     or name == "sparse_concentric_mixture" and angle in (30, 60))
            assert outcome == {"positive": 0 if fails else 6, "abstained": 6 if fails else 0}
    json.dumps(receipt, allow_nan=False)


def run_adaptive_noise_sensitivity_matrix():
    """150 bounded nonzero-noise cases using the unchanged development policy.

    Each of the five existing strata, ten signed angles and three radius scales
    appears once. Seed 2026090717 adds independent direction noise with magnitude
    25--95% of the already declared epsilon (1e-6 * radius/2). No noise-dependent
    policy change, coordinate transform, residual trimming or retry occurs.

    An interval gets a truth-containment check only after exact observed-row
    identity recovers both entire true flange supports and the entire constructed
    transition. Mixture bands additionally must match an entire constructed
    radius layer. Other hypotheses remain in the reported denominators but are
    NOT scored as correct geometry. This is sensitivity, not calibrated accuracy.
    """
    rng = np.random.default_rng(2026090717)
    names = ("clean", "missing_middle_arc", "concentric_mixture",
             "sparse_flange_stations", "sparse_concentric_mixture")
    counts = {name: dict(
        cases=0, conditional_cases=0, abstained_cases=0, incomplete_cases=0,
        candidate_associations=0, matched_joint_associations=0, unmatched_associations=0,
        conditional_single_candidates=0, conditional_two_candidates=0,
        proposed_single_intervals=0, oracle_matched_single_intervals=0,
        single_containment_failures=0, proposed_two_band_intervals=0,
        oracle_matched_two_band_intervals=0, two_band_containment_failures=0,
        reasons={},
    ) for name in names}
    noise_ratios = []
    position_count = 0
    for name, current in counts.items():
        mixture = name.endswith("concentric_mixture")
        for angle in (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150):
            for radius in (0.5, 2, 20):
                first, second, transition = synthetic_bend(
                    angle, radius, flange_extent=2 if name.startswith("sparse_") else 1,
                    mixture=mixture,
                )
                if name == "missing_middle_arc":
                    transition = np.concatenate([transition[:15], transition[20:]])
                noiseless = np.concatenate([first, second, transition])
                scale = radius / 2
                epsilon = 1e-6 * scale
                direction = rng.normal(size=noiseless.shape)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                noise = direction * rng.uniform(0.25, 0.95, size=(len(noiseless), 1)) * epsilon
                observed = noiseless + noise
                actual_error = np.linalg.norm(observed - noiseless, axis=1)
                assert np.all((actual_error > 0) & (actual_error <= epsilon))
                noise_ratios.extend((actual_error / epsilon).tolist())
                position_count += len(observed)
                result = analyze(observed, scale=scale)
                assert_scope(result)
                current["cases"] += 1
                conditional = result["status"] == "POSITION_GEOMETRY_CANDIDATES"
                current["conditional_cases" if conditional else "abstained_cases"] += 1
                current["incomplete_cases"] += int(not result["search_complete_within_declared_policy"])
                for reason in result["reasons"]:
                    current["reasons"][reason] = current["reasons"].get(reason, 0) + 1

                # Exact float rows survive canonical sorting/JSON conversion.
                # This lookup is an assertion-only correspondence oracle, never
                # a nearest-fit label assignment or an input to the analyzer.
                original_index = {tuple(row): index for index, row in enumerate(observed)}
                assert len(original_index) == len(observed)
                canonical_ids = [original_index[tuple(row)] for row in result["positions"]]
                region_ids = {
                    region["region_id"]: frozenset(canonical_ids[index] for index in region["point_indices"])
                    for region in result["extended_regions"]
                }
                first_ids = frozenset(range(len(first)))
                second_ids = frozenset(range(len(first), len(first) + len(second)))
                transition_start = len(first) + len(second)
                transition_ids = frozenset(range(transition_start, len(observed)))
                true_layers = ({frozenset(range(transition_start, transition_start + 35)): radius,
                                frozenset(range(transition_start + 35, len(observed))): 1.1 * radius}
                               if mixture else {})
                for candidate in result["candidates"]:
                    assert_whole_component(result, candidate)
                    current["candidate_associations"] += 1
                    if candidate["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS":
                        current["conditional_single_candidates"] += 1
                    elif candidate["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS":
                        current["conditional_two_candidates"] += 1
                    planes_match = {region_ids[index] for index in candidate["plane_region_ids"]} == {
                        first_ids, second_ids,
                    }
                    support_ids = frozenset(canonical_ids[index] for index in candidate["transition_point_indices"])
                    matched = planes_match and support_ids == transition_ids
                    current["matched_joint_associations" if matched else "unmatched_associations"] += 1
                    if not mixture:
                        current["proposed_single_intervals"] += 1
                        if matched:
                            current["oracle_matched_single_intervals"] += 1
                            lower, upper = candidate["single_radius_analysis"]["radius_interval"]
                            current["single_containment_failures"] += int(
                                not (lower <= radius and (upper is None or upper >= radius))
                            )
                    two = candidate["two_radius_analysis"]
                    if not mixture or two is None:
                        continue
                    mixed_ids = [original_index[tuple(row)] for row in two["mixed_positions"]]
                    for model in two["models"]:
                        for band in model["bands"]:
                            current["proposed_two_band_intervals"] += 1
                            band_ids = frozenset(mixed_ids[index] for index in band["point_indices"])
                            if not matched or band_ids not in true_layers:
                                continue
                            current["oracle_matched_two_band_intervals"] += 1
                            truth = true_layers[band_ids]
                            lower, upper = band["radius_interval"]
                            current["two_band_containment_failures"] += int(
                                not (lower <= truth and (upper is None or upper >= truth))
                            )
    for current in counts.values():
        current["unmatched_single_intervals"] = (
            current["proposed_single_intervals"] - current["oracle_matched_single_intervals"]
        )
        current["unmatched_two_band_intervals"] = (
            current["proposed_two_band_intervals"] - current["oracle_matched_two_band_intervals"]
        )
    return {
        "cases": sum(item["cases"] for item in counts.values()),
        "seed": 2026090717, "positions_perturbed": position_count,
        "epsilon_policy": "1e-6 * radius / 2; unchanged from the exact structural matrix",
        "actual_error_over_epsilon_min": min(noise_ratios),
        "actual_error_over_epsilon_max": max(noise_ratios),
        "strata": counts,
    }


def test_fixed_bounded_noise_sensitivity_receipt_has_honest_denominators():
    receipt = run_adaptive_noise_sensitivity_matrix()
    print(json.dumps(receipt, sort_keys=True))
    assert receipt["cases"] == 150
    assert 0 < receipt["actual_error_over_epsilon_min"] <= receipt["actual_error_over_epsilon_max"] <= 1
    for current in receipt["strata"].values():
        assert current["cases"] == current["conditional_cases"] + current["abstained_cases"] == 30
        assert current["incomplete_cases"] <= current["abstained_cases"]
        assert current["candidate_associations"] == current["matched_joint_associations"] + current["unmatched_associations"]
        assert current["oracle_matched_single_intervals"] <= current["proposed_single_intervals"]
        assert current["oracle_matched_two_band_intervals"] <= current["proposed_two_band_intervals"]
        assert current["proposed_single_intervals"] == current["oracle_matched_single_intervals"] + current["unmatched_single_intervals"]
        assert current["proposed_two_band_intervals"] == current["oracle_matched_two_band_intervals"] + current["unmatched_two_band_intervals"]
        assert current["single_containment_failures"] == current["two_band_containment_failures"] == 0
    # Intentionally no required positive rate or unchanged exact-matrix coverage.
    json.dumps(receipt, allow_nan=False)
