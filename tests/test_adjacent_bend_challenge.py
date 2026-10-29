"""Predeclared adjacent-bend attribution on invented positions only.

72 geometries x exact/bounded noise x retained mutual/union = 288 analyzer calls.
No candidate changes or threshold search. Truth is an assertion channel only.
"""
from collections import Counter
import json

import numpy as np
import pytest

from tests.adjacent_bend_constructions import (
    FULL_ARC, FIRST_HALF_ARC, MIDDLE_GAP_ARC, make_adjacent_bend_fixture,
)
from tests.test_graph_envelope_bend_candidates import (
    CONDITIONAL_STATUSES, analyze, assert_scope, assert_whole_component,
)


ANGLE_PAIRS = ((90, 90), (-90, -90), (90, -90), (-90, 90))
RADIUS_PAIRS = ((2, 2), (2, 3))
MIDDLE_LENGTHS = (0.25, 2, 6)
COVERAGES = {
    "full": (FULL_ARC, FULL_ARC),
    "arc2_middle_gap": (FULL_ARC, MIDDLE_GAP_ARC),
    "arc1_first_half": (FIRST_HALF_ARC, FULL_ARC),
}
NOISE_SEED = 2026090702


def challenge_inputs():
    """Fixed order and bounded noise; construction metadata never enters engine."""
    rng = np.random.default_rng(NOISE_SEED)
    for angles in ANGLE_PAIRS:
        for radii in RADIUS_PAIRS:
            for coverage, arc_indices in COVERAGES.items():
                for middle in MIDDLE_LENGTHS:
                    fixture = make_adjacent_bend_fixture(
                        angles_degrees=angles, radii=radii, flange_lengths=(2, middle, 2),
                        axial_extent=2, arc_sample_indices=arc_indices, unit="mm",
                    )
                    exact = fixture["points"]
                    direction = rng.normal(size=exact.shape)
                    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                    noisy = exact + direction*rng.uniform(0.25, 0.95, size=(len(exact), 1))*1e-6
                    error = np.linalg.norm(noisy-exact, axis=1)
                    assert np.all((error > 0) & (error <= 1e-6))
                    for noise, points in (("exact", exact), ("bounded_noise", noisy)):
                        yield dict(angles=angles, radii=radii, coverage=coverage,
                                   middle_length=middle, noise=noise), fixture, points


def connected_components(adjacency, allowed):
    remaining, groups = set(allowed), []
    while remaining:
        seen, frontier = set(), {min(remaining)}
        while frontier:
            seen.update(frontier)
            frontier = set.union(*(adjacency[i] for i in frontier)) & remaining - seen
        remaining -= seen
        groups.append(frozenset(seen))
    return groups


def classify_result(fixture, observed, result):
    original = {tuple(p): i for i, p in enumerate(observed)}
    assert len(original) == len(observed)
    ids = [original[tuple(p)] for p in result["positions"]]
    regions = {r["region_id"]: frozenset(ids[i] for i in r["point_indices"])
               for r in result["extended_regions"]}
    graph_edges = {tuple(sorted((ids[i], ids[j]))) for i, j in result["graph"]["edges"]}
    adjacency = {i: set() for i in range(len(observed))}
    for i, j in graph_edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    true_supports = {name: frozenset(map(int, indices)) for name, indices in fixture["support_indices"].items()}
    candidates = []
    for candidate in result["candidates"]:
        assert_whole_component(result, candidate)
        candidates.append((candidate, {regions[i] for i in candidate["plane_region_ids"]},
                           frozenset(ids[i] for i in candidate["transition_point_indices"])))
    events, matched_candidate_ids = [], set()
    for event_id, bend in enumerate(fixture["truth"]["bends"]):
        flanges = [true_supports[name] for name in bend["adjacent_flange_names"]]
        arc = true_supports[bend["arc_support_name"]]
        complete_flange_present = [flange in regions.values() for flange in flanges]
        remainder = set(range(len(observed))) - set.union(*map(set, flanges))
        groups = [group for group in connected_components(adjacency, remainder) if group & arc]
        own_arc_groups = connected_components(adjacency, arc)
        compound = any(group - arc for group in groups)
        seam_links = [sum(len(adjacency[i] & flange) for i in arc) for flange in flanges]
        matches = [candidate for candidate, pair, support in candidates
                   if pair == set(flanges) and support == arc]
        matched_candidate_ids.update(candidate["candidate_id"] for candidate in matches)
        positives = [candidate for candidate in matches if candidate["status"] in CONDITIONAL_STATUSES]
        if not result["search_complete_within_declared_policy"]:
            classification = "SEARCH_INCOMPLETE"
        elif not all(complete_flange_present):
            classification = "ABSENT_COMPLETE_FLANGE"
        elif len(own_arc_groups) != 1:
            classification = "OBSERVED_ARC_FRAGMENTED"
        elif compound:
            classification = "COMPOUND_REMAINDER"
        elif not all(seam_links):
            classification = "MISSING_OBSERVED_SEAM_LINK"
        elif not matches:
            classification = "UNEXPLAINED_ASSOCIATION_ABSENCE"
        elif not positives:
            classification = "MATCHED_RADIUS_ABSTAINED"
        else:
            classification = "COMPLETE_OBSERVED_SUPPORT_CONDITIONAL"
        intervals, failures = 0, 0
        for candidate in matches:
            interval = candidate["single_radius_analysis"]["radius_interval"]
            if interval is not None:
                intervals += 1
                lower, upper = interval
                failures += int(not (lower <= bend["radius"] and (upper is None or bend["radius"] <= upper)))
        events.append(dict(
            event_id=event_id, classification=classification,
            complete_flange_present=complete_flange_present,
            observed_arc_sample_count=len(arc), observed_arc_component_sizes=sorted(map(len, own_arc_groups)),
            seam_links=seam_links, matched_associations=len(matches), conditional_matches=len(positives),
            checked_single_intervals=intervals, containment_failures=failures,
            parent_components=[dict(sample_count=len(group),
                support_memberships={name: len(group & support) for name, support in true_supports.items()}) for group in groups],
            original_arc_to_other_remainder_edges=sum(len(adjacency[i] & (remainder-set(arc))) for i in arc),
            physical_coverage_is_established=False,
        ))
    return dict(
        events=events, associations=len(candidates),
        unmatched_associations=len(candidates)-len(matched_candidate_ids),
        unmatched_conditional_associations=sum(c["status"] in CONDITIONAL_STATUSES and c["candidate_id"] not in matched_candidate_ids
                                              for c, _, _ in candidates),
        final_regions=len(regions), max_plane_evaluations=result["plane_discovery"]["diagnostics"]["plane_evaluation_count"],
        incomplete=not result["search_complete_within_declared_policy"], reasons=result["reasons"],
        arc_to_arc_edges=sum(len(adjacency[i] & true_supports["arc_1"]) for i in true_supports["arc_0"]),
    )


def run_adjacent_challenge():
    receipts, summaries = [], {}
    for case, fixture, observed in challenge_inputs():
        for symmetry in ("mutual", "union"):
            result = analyze(observed, symmetry=symmetry)
            assert_scope(result)
            receipt = dict(case, symmetry=symmetry, **classify_result(fixture, observed, result))
            receipts.append(receipt)
            key = f"{case['noise']}/{symmetry}/{case['coverage']}/L{case['middle_length']}"
            count = summaries.setdefault(key, Counter())
            count.update(cases=1, events=2, associations=receipt["associations"],
                         unmatched_associations=receipt["unmatched_associations"],
                         unmatched_conditional_associations=receipt["unmatched_conditional_associations"],
                         incomplete_cases=int(receipt["incomplete"]))
            for event in receipt["events"]:
                count[event["classification"]] += 1
                count["checked_single_intervals"] += event["checked_single_intervals"]
                count["containment_failures"] += event["containment_failures"]
    return dict(seed=NOISE_SEED, geometries=72, observed_clouds=144,
                analyzer_calls=288, receipts=receipts,
                summaries={k: dict(v) for k, v in summaries.items()})


@pytest.fixture(scope="module")
def challenge():
    return run_adjacent_challenge()


def test_predeclared_challenge_has_complete_case_and_event_denominators(challenge):
    assert len(challenge["receipts"]) == challenge["analyzer_calls"] == 288
    assert len(challenge["summaries"]) == 36
    assert sum(len(c["events"]) for c in challenge["receipts"]) == 576
    for counts in challenge["summaries"].values():
        assert counts["cases"] == 8 and counts["events"] == 16
    json.dumps(challenge, allow_nan=False)
    print("ADJACENT_CHALLENGE=" + json.dumps(challenge["summaries"], sort_keys=True))


def test_all_observed_support_matched_bounds_contain_constructed_radius(challenge):
    assert all(event["containment_failures"] == 0 for case in challenge["receipts"] for event in case["events"])
    assert all(event["classification"] != "UNEXPLAINED_ASSOCIATION_ABSENCE"
               for case in challenge["receipts"] for event in case["events"])


def test_graph_parent_receipts_account_for_every_named_construction_support(challenge):
    for case in challenge["receipts"]:
        assert case["final_regions"] <= 64 and case["max_plane_evaluations"] <= 2048
        for event in case["events"]:
            assert sum(event["observed_arc_component_sizes"]) == event["observed_arc_sample_count"]
            for parent in event["parent_components"]:
                assert sum(parent["support_memberships"].values()) == parent["sample_count"]
            assert event["physical_coverage_is_established"] is False
