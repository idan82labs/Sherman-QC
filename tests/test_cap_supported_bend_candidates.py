"""Paired cap-ball seed hypothesis checks on retained invented challenges.

The candidate adds every cap ball with seed minimum3, final minimum9. No old k,
radius, epsilon, graph policy, fit/pair/association/split/query cap is increased.
"""
from collections import Counter
import json
import math

import numpy as np
import pytest

from research.geometry.cap_supported_bend_candidates import analyze_cap_supported_bend_candidates
from tests.adjacent_bend_constructions import make_adjacent_bend_fixture
from tests.test_adaptive_bend_candidates import synthetic_bend
from tests.test_adjacent_bend_challenge import challenge_inputs, classify_result
from tests.test_graph_envelope_bend_candidates import (
    CONDITIONAL_STATUSES, STRATA, analyze as baseline_analyze, assert_scope,
    assert_whole_component, matched_candidates,
)


def analyze(points, *, scale=1.0, **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6*scale, neighbor_counts=[8],
        max_neighborhood_radius=1.5*scale, max_normal_angle_bound=0.1,
        min_region_points=9, min_seed_points=3, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5*scale, max_axis_angle_bound=0.1,
        max_axial_gap=0.51*scale, max_radius_width=0.1*scale,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255, max_plane_evaluations=2048,
        graph_symmetry="union",
    )
    policy.update(changes)
    return analyze_cap_supported_bend_candidates(points, **policy)


def previous_inputs():
    rng = np.random.default_rng(2026090717)
    for name in STRATA:
        mixture = name.endswith("concentric_mixture")
        for angle in (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150):
            for radius in (0.5, 2, 20):
                first, second, transition = synthetic_bend(
                    angle, radius, flange_extent=2 if name.startswith("sparse_") else 1,
                    mixture=mixture,
                )
                if name == "missing_middle_arc":
                    transition = np.concatenate([transition[:15], transition[20:]])
                exact = np.concatenate([first, second, transition])
                epsilon = 1e-6*radius/2
                direction = rng.normal(size=exact.shape)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                noisy = exact + direction*rng.uniform(0.25, 0.95, size=(len(exact), 1))*epsilon
                assert np.all(np.linalg.norm(noisy-exact, axis=1) <= epsilon)
                for noise, observed in (("exact", exact), ("bounded_noise", noisy)):
                    yield dict(stratum=name, angle=angle, radius=radius, noise=noise,
                               mixture=mixture), observed


def radius_checks(observed, result, matches, radius, mixture):
    counts = Counter(checked_single_intervals=0, checked_band_intervals=0, containment_failures=0,
                     all_proposed_single_intervals=sum(c["single_radius_analysis"]["radius_interval"] is not None for c in result["candidates"]),
                     all_proposed_band_intervals=sum(len(m["bands"]) for c in result["candidates"]
                         if c["two_radius_analysis"] is not None for m in c["two_radius_analysis"]["models"]))
    original = {tuple(p): i for i, p in enumerate(observed)}
    for candidate in matches:
        single, two = candidate["single_radius_analysis"], candidate["two_radius_analysis"]
        if not mixture and single["radius_interval"] is not None:
            lower, upper = single["radius_interval"]
            counts["checked_single_intervals"] += 1
            counts["containment_failures"] += int(not (lower <= radius and (upper is None or radius <= upper)))
        if two is not None:
            ids = [original[tuple(p)] for p in two["mixed_positions"]]
            truth = {frozenset(range(50, 85)): radius, frozenset(range(85, 120)): 1.1*radius} if mixture else {}
            for model in two["models"]:
                for band in model["bands"]:
                    support = frozenset(ids[i] for i in band["point_indices"])
                    if support in truth:
                        lower, upper = band["radius_interval"]
                        counts["checked_band_intervals"] += 1
                        counts["containment_failures"] += int(not (lower <= truth[support] and (upper is None or truth[support] <= upper)))
    counts["unmatched_band_intervals"] = counts["all_proposed_band_intervals"]-counts["checked_band_intervals"]
    return counts


def run_previous_comparison():
    receipts, summaries = [], {}
    for case, observed in previous_inputs():
        scale = case["radius"]/2
        baseline = baseline_analyze(observed, scale=scale, symmetry="union")
        candidate = analyze(observed, scale=scale)
        assert_scope(candidate)
        for item in candidate["candidates"]:
            assert_whole_component(candidate, item)
        old, _ = matched_candidates(observed, baseline, (25, 25))
        new, _ = matched_candidates(observed, candidate, (25, 25))
        old_positive = any(c["status"] in CONDITIONAL_STATUSES for c in old)
        new_positive = any(c["status"] in CONDITIONAL_STATUSES for c in new)
        counts = Counter(cases=1, baseline_matched_cases=int(bool(old)), matched_cases=int(bool(new)),
            baseline_conditional_cases=int(old_positive), conditional_cases=int(new_positive),
            lost_matched_cases=int(bool(old) and not new), lost_conditional_cases=int(old_positive and not new_positive),
            baseline_associations=len(baseline["candidates"]), associations=len(candidate["candidates"]),
            unmatched_associations=len(candidate["candidates"])-len(new),
            incomplete_cases=int(not candidate["search_complete_within_declared_policy"]))
        counts.update(radius_checks(observed, candidate, new, case["radius"], case["mixture"]))
        key = f"{case['noise']}/{case['stratum']}"
        summaries.setdefault(key, Counter()).update(counts)
        receipts.append(dict(case, **dict(counts), reasons=candidate["reasons"],
            final_regions=len(candidate["extended_regions"]),
            plane_evaluations=candidate["plane_discovery"]["diagnostics"]["plane_evaluation_count"]))
    return dict(receipts=receipts, summaries={k: dict(v) for k, v in summaries.items()})


def run_adjacent_comparison():
    receipts, summaries = [], {}
    for case, fixture, observed in challenge_inputs():
        baseline = baseline_analyze(observed, symmetry="union")
        candidate = analyze(observed)
        assert_scope(candidate)
        old = classify_result(fixture, observed, baseline)
        new = classify_result(fixture, observed, candidate)
        counts = Counter(cases=1, events=2, baseline_associations=old["associations"],
                         associations=new["associations"], unmatched_associations=new["unmatched_associations"],
                         unmatched_conditional_associations=new["unmatched_conditional_associations"],
                         incomplete_cases=int(new["incomplete"]), lost_matched_events=0, lost_conditional_events=0,
                         baseline_matched_events=0, matched_events=0, baseline_conditional_events=0,
                         conditional_events=0, checked_single_intervals=0, containment_failures=0)
        for first, second in zip(old["events"], new["events"]):
            counts["lost_matched_events"] += int(bool(first["matched_associations"]) and not second["matched_associations"])
            counts["lost_conditional_events"] += int(bool(first["conditional_matches"]) and not second["conditional_matches"])
            counts["baseline_matched_events"] += int(bool(first["matched_associations"]))
            counts["matched_events"] += int(bool(second["matched_associations"]))
            counts["baseline_conditional_events"] += int(bool(first["conditional_matches"]))
            counts["conditional_events"] += int(bool(second["conditional_matches"]))
            counts[second["classification"]] += 1
            counts["checked_single_intervals"] += second["checked_single_intervals"]
            counts["containment_failures"] += second["containment_failures"]
        for prefix, result, receipt in (("baseline_", baseline, old), ("", candidate, new)):
            singles = sum(c["single_radius_analysis"]["radius_interval"] is not None
                          for c in result["candidates"])
            bands = sum(len(model["bands"]) for c in result["candidates"]
                        if c["two_radius_analysis"] is not None
                        for model in c["two_radius_analysis"]["models"])
            checked = sum(event["checked_single_intervals"] for event in receipt["events"])
            counts[prefix+"all_proposed_single_intervals"] = singles
            counts[prefix+"unscored_single_intervals"] = singles-checked
            # Adjacent truth is two DIFFERENT-center events. Concentric bands
            # on alternative supports have no full-event radius truth assertion.
            counts[prefix+"all_proposed_band_intervals"] = bands
            counts[prefix+"unscored_band_intervals"] = bands
        key = f"{case['noise']}/{case['coverage']}/L{case['middle_length']}"
        summaries.setdefault(key, Counter()).update(counts)
        receipts.append(dict(case, **dict(counts), baseline=old, candidate=new))
    return dict(receipts=receipts, summaries={k: dict(v) for k, v in summaries.items()})


@pytest.fixture(scope="module")
def previous():
    return run_previous_comparison()


@pytest.fixture(scope="module")
def adjacent():
    return run_adjacent_comparison()


def test_all_300_previous_cases_retain_complete_support_and_conditional_bounds(previous):
    print("CAP_SUPPORTED_PREVIOUS="+json.dumps(previous["summaries"], sort_keys=True))
    assert len(previous["receipts"]) == 300
    for counts in previous["summaries"].values():
        assert counts["cases"] == 30
        assert counts["lost_matched_cases"] == counts["lost_conditional_cases"] == 0
        assert counts["incomplete_cases"] == counts["containment_failures"] == 0


def test_all_144_adjacent_clouds_retain_old_matches_and_record_new_failures(adjacent):
    print("CAP_SUPPORTED_ADJACENT="+json.dumps(adjacent["summaries"], sort_keys=True))
    assert len(adjacent["receipts"]) == 144
    assert len(adjacent["summaries"]) == 18
    for counts in adjacent["summaries"].values():
        assert counts["cases"] == 8 and counts["events"] == 16
        assert counts["lost_matched_events"] == counts["lost_conditional_events"] == 0
        assert counts["incomplete_cases"] == counts["containment_failures"] == 0


@pytest.mark.parametrize("noise", ["exact", "bounded_noise"])
def test_recorded_adjacent_gain_retains_ambiguity_and_missing_seam_denominators(adjacent, noise):
    # Persistent receipt of the first fixed-policy run, not a physical accuracy gate.
    totals = Counter()
    for key, counts in adjacent["summaries"].items():
        if key.startswith(noise+"/"):
            totals.update(counts)
    assert totals["cases"] == 72 and totals["events"] == 144
    assert totals["baseline_conditional_events"] == 80
    assert totals["conditional_events"] == 120
    assert totals["baseline_associations"] == 216
    assert totals["associations"] == 264
    assert totals["unmatched_associations"] == 144
    assert totals["unmatched_conditional_associations"] == 36
    assert totals["MISSING_OBSERVED_SEAM_LINK"] == 24
    assert totals["ABSENT_COMPLETE_FLANGE"] == 0
    for case in adjacent["receipts"]:
        if case["noise"] == noise and case["coverage"] == "arc1_first_half":
            first, second = case["candidate"]["events"]
            assert first["classification"] == "MISSING_OBSERVED_SEAM_LINK"
            assert first["conditional_matches"] == 0
            assert second["classification"] == "COMPLETE_OBSERVED_SUPPORT_CONDITIONAL"


@pytest.mark.parametrize("angles", [(90, 90), (90, -90)])
def test_long_flange_gain_survives_rigid_motion_and_meter_conversion(angles):
    rotation, _ = np.linalg.qr(np.random.default_rng(51).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    fixture = make_adjacent_bend_fixture(
        angles_degrees=angles, radii=(0.002, 0.003),
        flange_lengths=(0.002, 0.006, 0.002), axial_extent=0.002,
        unit="m", rotation=rotation, translation=(0.01, -0.02, 0.03),
    )
    result = analyze(fixture["points"], scale=0.001, unit="m")
    receipt = classify_result(fixture, fixture["points"], result)
    assert not receipt["incomplete"]
    for event in receipt["events"]:
        assert event["classification"] == "COMPLETE_OBSERVED_SUPPORT_CONDITIONAL"
        assert event["checked_single_intervals"] == 1
        assert event["containment_failures"] == 0


def test_work_remains_bounded_and_unmatched_bands_are_not_hidden(previous, adjacent):
    assert max(c["plane_evaluations"] for c in previous["receipts"]) <= 2048
    assert max(c["final_regions"] for c in previous["receipts"]) <= 64
    assert all(c["unmatched_band_intervals"] >= 0 for c in previous["receipts"])
    assert max(c["candidate"]["max_plane_evaluations"] for c in adjacent["receipts"]) <= 2048
    assert max(c["candidate"]["final_regions"] for c in adjacent["receipts"]) <= 64
    for counts in adjacent["summaries"].values():
        assert counts["unscored_single_intervals"] >= 0
        assert (counts["checked_single_intervals"]+counts["unscored_single_intervals"]
                == counts["all_proposed_single_intervals"])
        assert counts["unscored_band_intervals"] == counts["all_proposed_band_intervals"]
    json.dumps(previous, allow_nan=False)
    json.dumps(adjacent, allow_nan=False)
    print("CAP_SUPPORTED_WORK="+json.dumps({
        "previous_max_plane_fits": max(c["plane_evaluations"] for c in previous["receipts"]),
        "previous_max_regions": max(c["final_regions"] for c in previous["receipts"]),
        "adjacent_max_plane_fits": max(c["candidate"]["max_plane_evaluations"] for c in adjacent["receipts"]),
        "adjacent_max_regions": max(c["candidate"]["final_regions"] for c in adjacent["receipts"]),
    }, sort_keys=True))


@pytest.mark.parametrize("mixture", [False, True])
def test_permutation_and_repetition_preserve_all_proposals(mixture):
    points = np.concatenate(synthetic_bend(mixture=mixture))
    reference = analyze(points)
    permuted = points[np.random.default_rng(22).permutation(len(points))]
    assert analyze(permuted) == reference
    repeated = analyze(np.concatenate([permuted, permuted]))
    for key in ("positions", "extended_regions", "candidates", "search_diagnostics", "status"):
        assert repeated[key] == reference[key]


def test_connected_radius_contaminant_stays_in_whole_component():
    extra = [[2.4/math.sqrt(2), 0, 2.4/math.sqrt(2)]]
    result = analyze(np.concatenate([np.concatenate(synthetic_bend()), extra]))
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert len(result["candidates"]) == 1
    assert len(result["candidates"][0]["transition_point_indices"]) == 36
    assert_whole_component(result, result["candidates"][0])


@pytest.mark.parametrize("changes,reason", [
    ({"max_regions": 3}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_plane_evaluations": 1}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_pair_evaluations": 5}, "PAIR_EVALUATION_BUDGET_EXCEEDED"),
    ({"max_bridge_evaluations": 1}, "BRIDGE_EVALUATION_BUDGET_EXCEEDED"),
])
def test_early_budget_failures_clear_prefixes(changes, reason):
    points = np.concatenate(synthetic_bend())
    result = analyze(np.concatenate([points, points+[15, 0, 0]]), **changes)
    assert_scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["reasons"] == [reason]
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 0


def test_late_split_budget_failure_clears_earlier_success():
    points = np.concatenate([np.concatenate(synthetic_bend()),
                             np.concatenate(synthetic_bend(mixture=True))+[15, 0, 0]])
    result = analyze(points, max_split_evaluations=68)
    assert_scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["reasons"] == ["TWO_RADIUS_SEARCH_INCOMPLETE"]
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 2


@pytest.mark.parametrize("value", [0, 2, 10, True, 3.5, None])
def test_seed_minimum_is_explicit_and_cannot_lower_final_support(value):
    with pytest.raises(ValueError):
        analyze([], min_seed_points=value)
