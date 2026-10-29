"""Fixed invented cases for the new final-region-budget policy; no real inputs.

Construction and independent serialized graph oracles are imported unchanged
from the earlier adaptive test source. Policy values, strata, case order and
noise seed are fixed before executing this new candidate.
"""
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry.cone_closed_bend_candidates import analyze_cone_closed_bend_candidates
from tests.test_adaptive_bend_candidates import (
    analyze as baseline_analyze,
    assert_scope, assert_whole_component, point_set, synthetic_bend,
)


def analyze(points=None, *, scale=1.0, **changes):
    # These are the SAME published baseline geometry/search settings. The only
    # new work setting is explicit; max_regions now counts FINAL joint supports.
    policy = dict(
        unit="mm", position_error_bound=1e-6 * scale, neighbor_counts=[8],
        max_neighborhood_radius=1.5 * scale, max_normal_angle_bound=0.1,
        min_region_points=9, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5 * scale, max_axis_angle_bound=0.1,
        max_axial_gap=0.51 * scale, max_radius_width=0.1 * scale,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255, max_plane_evaluations=2048,
    )
    policy.update(changes)
    points = np.concatenate(synthetic_bend()) * scale if points is None else points
    return analyze_cone_closed_bend_candidates(points, **policy)


def test_public_api_has_only_positions_and_explicit_policies():
    signature = inspect.signature(analyze_cone_closed_bend_candidates)
    assert set(signature.parameters) == {
        "points", "unit", "position_error_bound", "neighbor_counts", "max_neighborhood_radius",
        "max_normal_angle_bound", "min_region_points", "max_regions", "graph_neighbor_count",
        "max_link_distance", "max_axis_angle_bound", "max_axial_gap", "max_radius_width",
        "max_pair_evaluations", "max_bridge_evaluations", "min_band_points", "max_split_evaluations",
        "max_plane_evaluations",
    }
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY
               for name, parameter in signature.parameters.items() if name != "points")
    assert all(parameter.default == inspect.Parameter.empty for parameter in signature.parameters.values())


@pytest.mark.parametrize("mixture", [False, True])
def test_full_constructed_supports_reach_models_without_second_extension(mixture):
    first, second, transition = synthetic_bend(mixture=mixture)
    result = analyze(np.concatenate([first, second, transition]))
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert result["search_complete_within_declared_policy"] is True
    assert result["extended_regions"] == result["plane_discovery"]["regions"]
    assert result["policy"]["extension_policy"] == "USE_JOINT_REFITTED_DISCOVERY_REGIONS_WITHOUT_REEXTENSION"
    assert result["policy"]["region_budget_scope"] == "FINAL_EXACT_DEDUPLICATED_JOINT_REFIT_SUPPORTS_NOT_V1_RAW_SEEDS"
    assert result["plane_discovery"]["policy"]["different_budget_unit_from_adaptive_v1"] is True
    canonical = np.asarray(result["positions"])
    assert {point_set(canonical[region["point_indices"]]) for region in result["extended_regions"]} == {
        point_set(first), point_set(second),
    }
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert_whole_component(result, candidate)
    assert point_set(canonical[candidate["transition_point_indices"]]) == point_set(transition)
    if mixture:
        assert candidate["status"] == "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"
        for band, radius in zip(candidate["two_radius_analysis"]["models"][0]["bands"], (2, 2.2)):
            assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
    else:
        assert candidate["status"] == "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"
        assert candidate["two_radius_analysis"] is None
        lower, upper = candidate["single_radius_analysis"]["radius_interval"]
        assert lower <= 2 <= upper


def test_connected_contaminant_is_never_trimmed_to_rescue_a_radius():
    contaminant = [[2.4 / math.sqrt(2), 0, 2.4 / math.sqrt(2)]]
    result = analyze(np.concatenate([np.concatenate(synthetic_bend()), contaminant]))
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert_whole_component(result, candidate)
    assert len(candidate["transition_point_indices"]) == 36
    assert point_set(contaminant) <= point_set(np.asarray(result["positions"])[candidate["transition_point_indices"]])
    assert candidate["single_radius_analysis"]["status"] == candidate["two_radius_analysis"]["status"] == "ABSTAINED"


@pytest.mark.parametrize("seed", [0, 17, 20260907])
def test_permutation_and_exact_repeats_preserve_proposals(seed):
    points = np.concatenate(synthetic_bend(mixture=True))
    reference = analyze(points)
    permuted = points[np.random.default_rng(seed).permutation(len(points))]
    assert analyze(permuted) == reference
    repeated = analyze(np.concatenate([permuted, permuted]))
    assert repeated["input_count"] == 240
    for key in ("positions", "extended_regions", "candidates", "search_diagnostics", "status"):
        assert repeated[key] == reference[key]


@pytest.mark.parametrize("mixture", [False, True])
def test_se3_and_units_keep_matching_radius_enclosures(mixture):
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
        for band, radius in zip(candidate["two_radius_analysis"]["models"][0]["bands"], (0.002, 0.0022)):
            assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
    else:
        lower, upper = candidate["single_radius_analysis"]["radius_interval"]
        assert lower <= 0.002 <= upper


@pytest.mark.parametrize("changes,reason", [
    ({"max_regions": 3}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_plane_evaluations": 1}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_pair_evaluations": 5}, "PAIR_EVALUATION_BUDGET_EXCEEDED"),
    ({"max_bridge_evaluations": 1}, "BRIDGE_EVALUATION_BUDGET_EXCEEDED"),
])
def test_budget_failures_clear_every_candidate_before_model_calls(changes, reason):
    points = np.concatenate(synthetic_bend())
    result = analyze(np.concatenate([points, points + [15, 0, 0]]), **changes)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 0
    assert result["search_diagnostics"]["two_radius_evaluation_count"] == 0
    assert result["reasons"] == [reason]


def test_later_split_exhaustion_clears_earlier_single_radius_candidate():
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
    ("max_split_evaluations", 0), ("max_split_evaluations", 256),
    ("max_plane_evaluations", 0), ("max_plane_evaluations", 2049), ("max_plane_evaluations", True),
])
def test_policies_are_validated_even_for_empty_input(field, value):
    with pytest.raises(ValueError):
        analyze([], **{field: value})


def test_raw_cap_precedes_deduplication_or_array_protocol():
    class Oversize(list):
        def __len__(self):
            return 257

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversize conversion")

    with pytest.raises(ValueError, match="256"):
        analyze(Oversize())
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 0]] * 257)


@pytest.mark.parametrize("points", [[], np.empty((0, 3)), [[0, 0, 0]], [[1, 2, 3], [1, 2, 3]]])
def test_insufficient_positions_leave_physical_count_unknown(points):
    result = analyze(points)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True
    assert result["candidates"] == []


def run_cone_closed_matrix(*, noisy):
    """Same 150 development constructions and optional predeclared bounded noise.

    Every single-analysis interval is counted, INCLUDING a rejected pooled model
    on a mixture. A single interval receives a truth check only on a fully matched
    single-surface construction. Every returned two-band interval is counted;
    checks require BOTH complete flange supports, the complete transition and an
    entire constructed radial layer. Unmatched alternatives remain unscored.

    The unchanged adaptive V1 receives EXACTLY the same observed rows and old
    policy values once per case. Full-joint support retention and conditional
    full-joint retention are compared case by case; gains elsewhere cannot hide
    losing a baseline-matched case. No labels enter either analyzer.
    """
    rng = np.random.default_rng(2026090717)
    names = ("clean", "missing_middle_arc", "concentric_mixture",
             "sparse_flange_stations", "sparse_concentric_mixture")
    counts = {name: dict(
        cases=0, conditional_cases=0, abstained_cases=0, incomplete_cases=0,
        final_regions=0, max_final_regions=0, plane_evaluations=0, max_plane_evaluations=0,
        candidate_associations=0, matched_joint_associations=0, unmatched_associations=0,
        matched_joint_cases=0, matched_conditional_associations=0, matched_conditional_cases=0,
        conditional_single_candidates=0, conditional_two_candidates=0,
        proposed_single_intervals=0, oracle_matched_single_intervals=0,
        single_containment_failures=0, proposed_two_band_intervals=0,
        oracle_matched_two_band_intervals=0, two_band_containment_failures=0,
        baseline_candidate_associations=0, baseline_conditional_cases=0,
        baseline_incomplete_cases=0, baseline_matched_joint_associations=0,
        baseline_matched_joint_cases=0, baseline_matched_conditional_associations=0,
        baseline_matched_conditional_cases=0, lost_baseline_matched_cases=0,
        lost_baseline_matched_conditional_cases=0,
        reasons={},
    ) for name in names}
    noise_ratios, case_receipts = [], []
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
                scale, epsilon = radius / 2, 1e-6 * radius / 2
                observed = noiseless.copy()
                if noisy:
                    direction = rng.normal(size=noiseless.shape)
                    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                    observed += direction * rng.uniform(0.25, 0.95, size=(len(noiseless), 1)) * epsilon
                    actual_error = np.linalg.norm(observed - noiseless, axis=1)
                    assert np.all((actual_error > 0) & (actual_error <= epsilon))
                    noise_ratios.extend((actual_error / epsilon).tolist())
                position_count += len(observed)
                result = analyze(observed, scale=scale)
                assert_scope(result)
                baseline = baseline_analyze(observed, scale=scale)
                assert_scope(baseline)
                current["cases"] += 1
                conditional = result["status"] == "POSITION_GEOMETRY_CANDIDATES"
                current["conditional_cases" if conditional else "abstained_cases"] += 1
                incomplete = not result["search_complete_within_declared_policy"]
                current["incomplete_cases"] += int(incomplete)
                region_count = len(result["extended_regions"])
                fit_count = result["plane_discovery"]["diagnostics"]["plane_evaluation_count"]
                current["final_regions"] += region_count
                current["max_final_regions"] = max(current["max_final_regions"], region_count)
                current["plane_evaluations"] += fit_count
                current["max_plane_evaluations"] = max(current["max_plane_evaluations"], fit_count)
                for reason in result["reasons"]:
                    current["reasons"][reason] = current["reasons"].get(reason, 0) + 1
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
                matched_count = matched_conditional_count = 0
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
                    matched_count += int(matched)
                    matched_conditional_count += int(matched and candidate["status"] in (
                        "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS", "CONDITIONAL_TWO_RADIUS_HYPOTHESIS",
                    ))
                    current["matched_joint_associations" if matched else "unmatched_associations"] += 1
                    interval = candidate["single_radius_analysis"]["radius_interval"]
                    if interval is not None:
                        current["proposed_single_intervals"] += 1
                        if matched and not mixture:
                            current["oracle_matched_single_intervals"] += 1
                            lower, upper = interval
                            current["single_containment_failures"] += int(
                                not (lower <= radius and (upper is None or upper >= radius))
                            )
                    two = candidate["two_radius_analysis"]
                    if two is None:
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
                current["matched_joint_cases"] += int(matched_count > 0)
                current["matched_conditional_associations"] += matched_conditional_count
                current["matched_conditional_cases"] += int(matched_conditional_count > 0)

                # Independent exact-row oracle for the unchanged baseline on the
                # identical cloud. Matching requires both COMPLETE flange sets
                # and the COMPLETE transition, not just a nearby radius or label.
                assert baseline["positions"] == result["positions"]
                baseline_regions = {
                    region["region_id"]: frozenset(canonical_ids[index] for index in region["point_indices"])
                    for region in baseline["extended_regions"]
                }
                baseline_matched = baseline_matched_conditional = 0
                for candidate in baseline["candidates"]:
                    assert_whole_component(baseline, candidate)
                    planes_match = {baseline_regions[index] for index in candidate["plane_region_ids"]} == {
                        first_ids, second_ids,
                    }
                    support_ids = frozenset(canonical_ids[index] for index in candidate["transition_point_indices"])
                    matched = planes_match and support_ids == transition_ids
                    baseline_matched += int(matched)
                    baseline_matched_conditional += int(matched and candidate["status"] in (
                        "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS", "CONDITIONAL_TWO_RADIUS_HYPOTHESIS",
                    ))
                baseline_conditional = baseline["status"] == "POSITION_GEOMETRY_CANDIDATES"
                baseline_incomplete = not baseline["search_complete_within_declared_policy"]
                lost_matched = baseline_matched > 0 and matched_count == 0
                lost_conditional = baseline_matched_conditional > 0 and matched_conditional_count == 0
                current["baseline_candidate_associations"] += len(baseline["candidates"])
                current["baseline_conditional_cases"] += int(baseline_conditional)
                current["baseline_incomplete_cases"] += int(baseline_incomplete)
                current["baseline_matched_joint_associations"] += baseline_matched
                current["baseline_matched_joint_cases"] += int(baseline_matched > 0)
                current["baseline_matched_conditional_associations"] += baseline_matched_conditional
                current["baseline_matched_conditional_cases"] += int(baseline_matched_conditional > 0)
                current["lost_baseline_matched_cases"] += int(lost_matched)
                current["lost_baseline_matched_conditional_cases"] += int(lost_conditional)
                case_receipts.append(dict(stratum=name, angle=angle, radius=radius,
                    status=result["status"], incomplete=incomplete, reasons=result["reasons"],
                    final_regions=region_count, plane_evaluations=fit_count,
                    associations=len(result["candidates"]), matched_joint_associations=matched_count,
                    matched_conditional_associations=matched_conditional_count,
                    baseline_status=baseline["status"], baseline_incomplete=baseline_incomplete,
                    baseline_reasons=baseline["reasons"],
                    baseline_associations=len(baseline["candidates"]),
                    baseline_matched_joint_associations=baseline_matched,
                    baseline_matched_conditional_associations=baseline_matched_conditional,
                    lost_baseline_matched=lost_matched,
                    lost_baseline_matched_conditional=lost_conditional))
    for current in counts.values():
        current["unmatched_single_intervals"] = (
            current["proposed_single_intervals"] - current["oracle_matched_single_intervals"]
        )
        current["unmatched_two_band_intervals"] = (
            current["proposed_two_band_intervals"] - current["oracle_matched_two_band_intervals"]
        )
    return dict(cases=150, noisy=noisy, seed=2026090717 if noisy else None,
        positions=position_count, positions_perturbed=position_count if noisy else 0,
        actual_error_over_epsilon_min=min(noise_ratios) if noisy else None,
        actual_error_over_epsilon_max=max(noise_ratios) if noisy else None,
        single_denominator_scope="ALL_RETURNED_SINGLE_ANALYSIS_INTERVALS_INCLUDING_ABSTAINED_POOLED_MIXTURES",
        two_denominator_scope="ALL_RETURNED_TWO_RADIUS_MODEL_BAND_INTERVALS",
        paired_baseline_scope="UNCHANGED_ADAPTIVE_V1_ON_EXACT_SAME_OBSERVED_ROWS_AND_PRIOR_POLICIES",
        retention_scope="PER_CASE_COMPLETE_JOINT_SUPPORT_AND_CONDITIONAL_COMPLETE_JOINT_SUPPORT",
        strata=counts, case_receipts=case_receipts)


def run_cone_closed_structural_matrix():
    return run_cone_closed_matrix(noisy=False)


def run_cone_closed_noise_sensitivity_matrix():
    return run_cone_closed_matrix(noisy=True)


@pytest.mark.parametrize("noisy", [False, True])
def test_fixed_matrix_has_full_scoped_denominators_and_true_bound_checks(noisy):
    receipt = run_cone_closed_matrix(noisy=noisy)
    print(json.dumps({key: value for key, value in receipt.items() if key != "case_receipts"}, sort_keys=True))
    assert receipt["cases"] == len(receipt["case_receipts"]) == 150
    if noisy:
        assert 0 < receipt["actual_error_over_epsilon_min"] <= receipt["actual_error_over_epsilon_max"] <= 1
    for current in receipt["strata"].values():
        assert current["cases"] == current["conditional_cases"] + current["abstained_cases"] == 30
        assert current["incomplete_cases"] <= current["abstained_cases"]
        assert current["candidate_associations"] == current["matched_joint_associations"] + current["unmatched_associations"]
        assert current["proposed_single_intervals"] == current["oracle_matched_single_intervals"] + current["unmatched_single_intervals"]
        assert current["proposed_two_band_intervals"] == current["oracle_matched_two_band_intervals"] + current["unmatched_two_band_intervals"]
        assert current["single_containment_failures"] == current["two_band_containment_failures"] == 0
        assert current["max_final_regions"] <= 64
        assert current["max_plane_evaluations"] <= 2048
        assert current["lost_baseline_matched_cases"] == 0
        assert current["lost_baseline_matched_conditional_cases"] == 0
        assert current["matched_conditional_cases"] <= current["matched_joint_cases"]
        assert current["baseline_matched_conditional_cases"] <= current["baseline_matched_joint_cases"]
    assert all(not case["lost_baseline_matched"] and not case["lost_baseline_matched_conditional"]
               for case in receipt["case_receipts"])
    # Freeze the FIRST executed development receipt, including residual failures.
    # These exact counters are regression evidence, not physical success targets.
    expected = {
        "clean": (30, 30, 30, 30, 0),
        "missing_middle_arc": (30, 54, 30, 30, 0),
        "concentric_mixture": (30, 102, 30, 0, 60),
        "sparse_flange_stations": (24, 24, 24, 24, 0),
        "sparse_concentric_mixture": (16, 52, 16, 0, 32) if noisy else (18, 54, 18, 0, 36),
    }
    for name, counters in expected.items():
        current = receipt["strata"][name]
        assert tuple(current[key] for key in (
            "conditional_cases", "candidate_associations", "matched_joint_associations",
            "oracle_matched_single_intervals", "oracle_matched_two_band_intervals",
        )) == counters
        assert current["incomplete_cases"] == 0
        assert current["unmatched_two_band_intervals"] == 0
    json.dumps(receipt, allow_nan=False)
