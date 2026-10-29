"""Predeclared attribution of SAME-FACE alternatives on existing synthetic rows.

Exactly one direct cap-supported baseline and one audited-wrapper call per
prior300/adjacent144 cloud; the wrapper performs another baseline pass. A separate
tangency diagnostic is called once for EVERY conditional SINGLE candidate,
including complete-support controls. It recomputes the old radius analysis,
whereas the wrapper reuses its internal radius results. All extra work is
disclosed, not hidden inside a baseline timing claim. Conditional TWO candidates
are counted but never rejected by this same-face test.

Original construction row identities classify support AFTER inference. No score,
threshold tuning, sample trimming, new RNG draws, or modified baseline policies.
"""
from collections import Counter
import json

import numpy as np
import pytest

from tests.test_adjacent_bend_challenge import challenge_inputs
from tests.test_cap_supported_bend_candidates import analyze as cap_analyze, previous_inputs
from tests.test_graph_envelope_bend_candidates import (
    CONDITIONAL_STATUSES, assert_scope, assert_whole_component,
)


SINGLE_STATUS = "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS"
TWO_STATUS = "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"
DIAGNOSTIC_STATUSES = {
    "SINGLE_SAME_FACE_TANGENT_MODEL_EXCLUDED", "TANGENCY_NOT_EXCLUDED", "ABSTAINED",
}


def attribution_inputs():
    """Reuse both existing generators and their exact row/RNG order unchanged."""
    for case, observed in previous_inputs():
        supports = {"flange_0": frozenset(range(25)), "flange_1": frozenset(range(25, 50))}
        stations = (1, 2, 3, 5, 6, 7) if case["stratum"] == "missing_middle_arc" else tuple(range(1, 8))
        arc_names = ("arc_inner", "arc_outer") if case["mixture"] else ("arc_0",)
        profile_stations = {}
        offset = 50
        for name in arc_names:
            count = len(stations)*5
            supports[name] = frozenset(range(offset, offset+count))
            for station_index, station in enumerate(stations):
                for axial in range(5):
                    profile_stations[offset+station_index*5+axial] = (name, station)
            offset += count
        assert offset == len(observed)
        events = [dict(event_id=0, flange_names=("flange_0", "flange_1"),
                       transition_support=frozenset(range(50, len(observed))),
                       single_surface=not case["mixture"])]
        yield dict(dataset="previous", **case), observed, supports, profile_stations, events, case["radius"]/2
    for case, fixture, observed in challenge_inputs():
        supports = {name: frozenset(map(int, indices)) for name, indices in fixture["support_indices"].items()}
        profile_stations, events = {}, []
        for event_id, bend in enumerate(fixture["truth"]["bends"]):
            name = bend["arc_support_name"]
            indices = fixture["support_indices"][name]
            for local_station, station in enumerate(bend["visible_sample_indices"]):
                for index in indices[local_station*5:(local_station+1)*5]:
                    profile_stations[int(index)] = (name, station)
            events.append(dict(event_id=event_id,
                               flange_names=tuple(bend["adjacent_flange_names"]),
                               transition_support=supports[name], single_surface=True))
        yield dict(dataset="adjacent", **case), observed, supports, profile_stations, events, 1.0


def _plane_composition(support, named, profile_stations):
    memberships = {name: len(support & indices) for name, indices in named.items()}
    profile_ids = sorted({profile_stations[index] for index in support if index in profile_stations})
    for name, truth in named.items():
        if not name.startswith("flange_"):
            continue
        if support == truth:
            classification, owner = "COMPLETE_CONSTRUCTED_FLANGE", name
            break
        if support and support < truth:
            classification, owner = "PARTIAL_CONSTRUCTED_FLANGE", name
            break
    else:
        owner = next((name for name, truth in named.items()
                      if name.startswith("arc") and support and support <= truth), None)
        if owner is not None:
            # Two sampled generators of one ideal circular extrusion define a
            # secant sample plane. This is a construction role, NOT a verified
            # physical flat face or an inference input.
            classification = ("ARC_ONLY_TWO_GENERATOR_SECANT_SAMPLES" if len(profile_ids) == 2
                              else "ARC_ONLY_OTHER_PROFILE_SUPPORT")
        else:
            classification = "MIXED_CONSTRUCTION_SUPPORT"
    return dict(classification=classification, construction_owner=owner,
                sample_count=len(support), memberships=memberships,
                observed_profile_station_ids=[list(value) for value in profile_ids],
                physical_face_identity_is_verified=False)


def _transition_composition(support, named, profile_stations):
    memberships = {name: len(support & indices) for name, indices in named.items()}
    arcs = [name for name, count in memberships.items() if name.startswith("arc") and count]
    flange_points = sum(count for name, count in memberships.items() if name.startswith("flange_"))
    if flange_points and arcs:
        classification = "MIXED_ARC_FLANGE_SUPPORT"
    elif flange_points:
        classification = "FLANGE_ONLY_TRANSITION_SUPPORT"
    elif len(arcs) == 1:
        classification = "COMPLETE_OBSERVED_ARC_SUPPORT" if support == named[arcs[0]] else "PARTIAL_OBSERVED_ARC_SUPPORT"
    elif arcs:
        classification = ("COMPLETE_MULTIPLE_OBSERVED_ARC_SUPPORTS"
                          if all(named[name] <= support for name in arcs)
                          else "MULTIPLE_ARC_SUPPORTS_WITH_PARTIAL_MEMBERSHIP")
    else:
        classification = "EMPTY_SUPPORT"
    return dict(classification=classification, sample_count=len(support), memberships=memberships,
                observed_arc_names=arcs,
                observed_profile_station_ids=[list(value) for value in sorted({
                    profile_stations[index] for index in support if index in profile_stations})],
                physical_arc_coverage_is_verified=False)


def _candidate_receipts(observed, baseline, named, profile_stations, events, scale, diagnose):
    original = {tuple(row): index for index, row in enumerate(observed)}
    assert len(original) == len(observed)
    canonical_ids = [original[tuple(row)] for row in baseline["positions"]]
    positions = np.asarray(baseline["positions"])
    regions = {region["region_id"]: region for region in baseline["extended_regions"]}
    region_supports = {key: frozenset(canonical_ids[index] for index in region["point_indices"])
                       for key, region in regions.items()}
    result = []
    for candidate in baseline["candidates"]:
        assert_whole_component(baseline, candidate)
        plane_ids = candidate["plane_region_ids"]
        planes = [region_supports[index] for index in plane_ids]
        transition = frozenset(canonical_ids[index] for index in candidate["transition_point_indices"])
        matches = [event for event in events
                   if set(planes) == {named[name] for name in event["flange_names"]}
                   and transition == event["transition_support"]]
        diagnostic = None
        if candidate["status"] == SINGLE_STATUS:
            diagnostic = diagnose(
                positions[regions[plane_ids[0]]["point_indices"]],
                positions[regions[plane_ids[1]]["point_indices"]],
                positions[candidate["transition_point_indices"]],
                unit="mm", position_error_bound=1e-6*scale,
                max_axis_angle_bound=0.1, max_axial_gap=0.51*scale, max_radius_width=0.1*scale,
            )
            assert diagnostic["status"] in DIAGNOSTIC_STATUSES
            assert isinstance(diagnostic["single_same_face_tangent_model_excluded"], bool)
        two = candidate["two_radius_analysis"]
        result.append(dict(
            candidate_id=candidate["candidate_id"], baseline_status=candidate["status"],
            plane_region_ids=plane_ids,
            plane_compositions=[_plane_composition(support, named, profile_stations) for support in planes],
            transition_composition=_transition_composition(transition, named, profile_stations),
            oracle_matched_event_ids=[event["event_id"] for event in matches],
            oracle_full_single_surface_match=any(event["single_surface"] for event in matches),
            proposed_single_interval_count=int(candidate["single_radius_analysis"]["radius_interval"] is not None),
            proposed_two_band_interval_count=0 if two is None else sum(len(model["bands"]) for model in two["models"]),
            diagnostic=diagnostic,
            diagnostic_targets_only_single_same_face_model=True,
            baseline_candidate_is_removed=False,
        ))
    return result


def run_same_face_tangency_attribution():
    # Imported only when root executes this predeclared test after implementing
    # and reviewing the standalone diagnostic; module collection is source-safe.
    from research.geometry.same_face_tangency import analyze_same_face_tangency
    from tests.test_tangency_audited_bend_candidates import analyze as audit_analyze

    receipts, summaries = [], {}
    for case, observed, named, profile_stations, events, scale in attribution_inputs():
        baseline = cap_analyze(observed, scale=scale)
        assert_scope(baseline)
        candidates = _candidate_receipts(
            observed, baseline, named, profile_stations, events, scale, analyze_same_face_tangency,
        )
        audit = audit_analyze(observed, scale=scale)
        assert audit["base_analysis"] == baseline
        assert audit["search_complete_within_declared_policy"] is True
        assessments = {item["candidate_id"]: item["analysis"] for item in audit["assessments"]}
        assert len(assessments) == len(audit["assessments"])
        standalone = {candidate["candidate_id"]: candidate["diagnostic"] for candidate in candidates
                      if candidate["baseline_status"] == SINGLE_STATUS}
        assert assessments == standalone
        assert all(candidate["candidate_id"] not in assessments for candidate in candidates
                   if candidate["baseline_status"] == TWO_STATUS)
        assert audit["policy"]["radius_analysis_recomputed"] is False
        assert audit["policy"]["base_candidates_modified"] is False
        assert audit["policy"]["two_radius_candidates_checked_as_single_face"] is False
        counts = Counter(
            clouds=1, constructed_events=len(events), baseline_associations=len(candidates),
            baseline_matched_associations=0, baseline_unmatched_associations=0,
            baseline_conditional_single=0, baseline_conditional_two=0, baseline_abstained_associations=0,
            baseline_unmatched_conditionals=0, baseline_matched_single_controls=0,
            diagnostic_calls=0, diagnostic_excluded=0, diagnostic_not_excluded=0, diagnostic_abstained=0,
            unmatched_conditional_single_excluded=0, matched_single_controls_excluded=0,
            all_proposed_single_intervals=0, all_proposed_two_band_intervals=0,
            baseline_incomplete_clouds=int(not baseline["search_complete_within_declared_policy"]),
            wrapper_analyzer_calls=1,
            wrapper_center_enclosure_evaluations=audit["diagnostics"]["center_enclosure_evaluation_count"],
        )
        compositions = Counter()
        for candidate in candidates:
            matched = bool(candidate["oracle_matched_event_ids"])
            status = candidate["baseline_status"]
            counts["baseline_matched_associations" if matched else "baseline_unmatched_associations"] += 1
            counts["baseline_conditional_single" if status == SINGLE_STATUS else
                   "baseline_conditional_two" if status == TWO_STATUS else "baseline_abstained_associations"] += 1
            counts["baseline_unmatched_conditionals"] += int(not matched and status in CONDITIONAL_STATUSES)
            control = candidate["oracle_full_single_surface_match"] and status == SINGLE_STATUS
            counts["baseline_matched_single_controls"] += int(control)
            counts["all_proposed_single_intervals"] += candidate["proposed_single_interval_count"]
            counts["all_proposed_two_band_intervals"] += candidate["proposed_two_band_interval_count"]
            diagnostic = candidate["diagnostic"]
            if diagnostic is None:
                continue
            counts["diagnostic_calls"] += 1
            category = {"SINGLE_SAME_FACE_TANGENT_MODEL_EXCLUDED": "diagnostic_excluded",
                        "TANGENCY_NOT_EXCLUDED": "diagnostic_not_excluded", "ABSTAINED": "diagnostic_abstained"}[diagnostic["status"]]
            counts[category] += 1
            excluded = diagnostic["single_same_face_tangent_model_excluded"]
            counts["matched_single_controls_excluded"] += int(control and excluded)
            counts["unmatched_conditional_single_excluded"] += int(not matched and excluded)
            plane_classes = sorted(item["classification"] for item in candidate["plane_compositions"])
            signature = ("matched" if matched else "unmatched")+"/"+"+".join(plane_classes)+"/"+candidate["transition_composition"]["classification"]
            compositions[signature] += 1
        key = case["dataset"]+"/"+case["noise"]
        summaries.setdefault(key, Counter()).update(counts)
        receipts.append(dict(case, counts=dict(counts), conditional_support_compositions=dict(compositions),
                             candidates=candidates, baseline_reasons=baseline["reasons"],
                             wrapper_complete=audit["search_complete_within_declared_policy"],
                             wrapper_base_preserved_exactly=True,
                             wrapper_assessment_candidate_ids=list(assessments)))
    diagnostic_calls = sum(item["counts"]["diagnostic_calls"] for item in receipts)
    wrapper_calls = sum(item["counts"]["wrapper_analyzer_calls"] for item in receipts)
    return dict(
        baseline_analyzer_calls=len(receipts),
        wrapper_analyzer_calls=wrapper_calls,
        standalone_diagnostic_calls=diagnostic_calls,
        diagnostic_radius_recomputation_is_extra_work=True,
        call_ledger=dict(
            direct_baseline_passes=len(receipts), wrapper_internal_baseline_passes=wrapper_calls,
            standalone_radius_recomputations=diagnostic_calls, wrapper_radius_recomputations=0,
            top_level_analyzer_or_diagnostic_calls=len(receipts)+wrapper_calls+diagnostic_calls,
            wrapper_center_enclosure_evaluations=sum(item["counts"]["wrapper_center_enclosure_evaluations"] for item in receipts),
        ),
        original_noise_seeds={"previous": 2026090717, "adjacent": 2026090702},
        interpretation="FIXED_SAME_FACE_MODEL_ATTRIBUTION_NOT_PHYSICAL_FALSE_POSITIVE_SCORING",
        summaries={key: dict(value) for key, value in summaries.items()}, receipts=receipts,
    )


@pytest.fixture(scope="module")
def attribution():
    return run_same_face_tangency_attribution()


def test_baseline_denominators_and_rng_sources_are_retained_without_candidate_removal(attribution):
    assert attribution["baseline_analyzer_calls"] == 444
    assert attribution["wrapper_analyzer_calls"] == 444
    assert set(attribution["summaries"]) == {"previous/exact", "previous/bounded_noise",
                                           "adjacent/exact", "adjacent/bounded_noise"}
    for key, counts in attribution["summaries"].items():
        previous = key.startswith("previous/")
        assert counts["clouds"] == (150 if previous else 72)
        assert counts["baseline_associations"] == (294 if previous else 264)
        assert counts["baseline_unmatched_associations"] == 144
        assert counts["baseline_unmatched_conditionals"] == (12 if previous else 36)
        assert counts["baseline_incomplete_clouds"] == 0
        assert counts["baseline_associations"] == counts["baseline_matched_associations"]+counts["baseline_unmatched_associations"]
        assert counts["baseline_associations"] == counts["baseline_conditional_single"]+counts["baseline_conditional_two"]+counts["baseline_abstained_associations"]
        assert counts["diagnostic_calls"] == counts["baseline_conditional_single"]
        assert counts["diagnostic_calls"] == counts["diagnostic_excluded"]+counts["diagnostic_not_excluded"]+counts["diagnostic_abstained"]
        assert counts["wrapper_analyzer_calls"] == counts["clouds"]
        assert counts["wrapper_center_enclosure_evaluations"] == counts["diagnostic_calls"]
    json.dumps(attribution, allow_nan=False)


def test_every_complete_observed_single_surface_control_remains_not_excluded(attribution):
    for case in attribution["receipts"]:
        assert case["counts"]["matched_single_controls_excluded"] == 0
        assert case["wrapper_complete"] is True
        assert case["wrapper_base_preserved_exactly"] is True
        assert case["wrapper_assessment_candidate_ids"] == [
            candidate["candidate_id"] for candidate in case["candidates"]
            if candidate["baseline_status"] == SINGLE_STATUS]
        for candidate in case["candidates"]:
            diagnostic = candidate["diagnostic"]
            assert candidate["baseline_candidate_is_removed"] is False
            if candidate["baseline_status"] != SINGLE_STATUS:
                assert diagnostic is None
                continue
            assert diagnostic is not None
            assert diagnostic["single_same_face_tangent_model_excluded"] == (
                diagnostic["status"] == "SINGLE_SAME_FACE_TANGENT_MODEL_EXCLUDED")
            if candidate["oracle_full_single_surface_match"]:
                assert not diagnostic["single_same_face_tangent_model_excluded"]
            # ABSTAINED is counted separately, never reported as demonstrated
            # tangent compatibility; two-radius possibilities remain unjudged.


def test_composition_receipts_account_for_all_samples_and_preserve_every_candidate(attribution):
    for case in attribution["receipts"]:
        ids = [candidate["candidate_id"] for candidate in case["candidates"]]
        assert ids == list(range(len(ids)))
        assert len(ids) == case["counts"]["baseline_associations"]
        assert sum(case["conditional_support_compositions"].values()) == case["counts"]["baseline_conditional_single"]
        for candidate in case["candidates"]:
            for support in candidate["plane_compositions"]+[candidate["transition_composition"]]:
                assert sum(support["memberships"].values()) == support["sample_count"]
    print("SAME_FACE_TANGENCY_ATTRIBUTION="+json.dumps(attribution["summaries"], sort_keys=True))
    print("SAME_FACE_SUPPORT_COMPOSITIONS="+json.dumps([
        {key: case[key] for key in ("dataset", "noise", "conditional_support_compositions")}
        for case in attribution["receipts"] if case["counts"]["baseline_unmatched_conditionals"]
    ], sort_keys=True))


def test_recorded_development_relation_results_and_extra_work_remain_explicit(attribution):
    # These are persistent receipts of root's first fixed standalone run, NOT
    # accuracy targets, score selection, or a claim that exclusion finds defects.
    for key, counts in attribution["summaries"].items():
        previous = key.startswith("previous/")
        assert counts["diagnostic_calls"] == (102 if previous else 156)
        assert counts["diagnostic_excluded"] == (12 if previous else 36)
        assert counts["diagnostic_not_excluded"] == (90 if previous else 120)
        assert counts["diagnostic_abstained"] == 0
        assert counts["baseline_matched_single_controls"] == (90 if previous else 120)
        assert counts["matched_single_controls_excluded"] == 0
        assert counts["unmatched_conditional_single_excluded"] == (12 if previous else 36)
    assert attribution["standalone_diagnostic_calls"] == 516
    assert attribution["call_ledger"] == dict(
        direct_baseline_passes=444, wrapper_internal_baseline_passes=444,
        standalone_radius_recomputations=516, wrapper_radius_recomputations=0,
        top_level_analyzer_or_diagnostic_calls=1404, wrapper_center_enclosure_evaluations=516,
    )
    print("SAME_FACE_CALL_LEDGER="+json.dumps(attribution["call_ledger"], sort_keys=True))
