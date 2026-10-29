"""Invented scanner-scale INGESTION with sparse, complete observed-window payloads.

The fixed resource fixture streams two million rows in 8,192-row chunks. Eight
predeclared windows contain 860 unmodified construction rows in total; every
other row is deterministic background outside all windows. Foreground rows are
interleaved throughout the stream, not supplied as one convenient prefix.

This is NOT dense full-scan geometry qualification, automatic ROI discovery,
physical coverage, a global graph, calibrated scanner performance or accuracy.
Construction names and memberships are used only by test oracles. The dispatcher
receives positions, fixed spatial windows and the unchanged bounded core policy.
No data files, services, native geometry or random-number streams are involved.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import resource
import sys
import time

import numpy as np
import pytest

import research.geometry.streamed_window_bend_candidates as streamed
from tests.adjacent_bend_constructions import make_adjacent_bend_fixture
from tests.test_adaptive_bend_candidates import synthetic_bend
from tests.test_adjacent_bend_challenge import classify_result
from tests.test_cap_supported_bend_candidates import radius_checks
from tests.test_graph_envelope_bend_candidates import CONDITIONAL_STATUSES, matched_candidates
from tests.test_tangency_audited_bend_candidates import analyze as isolated_analyze


RESOURCE_TOTAL_ROWS = 2_000_000
RESOURCE_CHUNK_ROWS = 8_192
RESOURCE_SELECTED_ROWS = 860
RESOURCE_SECONDS_LIMIT = 120.0
RESOURCE_PEAK_RSS_LIMIT = 512*1024**2
UNCHANGED_PROCESS_TREE_HARD_LIMIT = 16*1024**3


def fixed_window_fixture():
    """Eight fixed, translated constructions; no fitting or window adaptation."""
    clouds, names, construction_truth = [], [], []
    for kind in ("single", "missing_middle_arc", "concentric_mixture"):
        for angle in (90, -90):
            first, second, transition = synthetic_bend(
                angle, 2.0, flange_extent=2.0, mixture=kind == "concentric_mixture",
            )
            if kind == "missing_middle_arc":
                # A priori visibility construction, not stream-time trimming.
                transition = np.concatenate([transition[:15], transition[20:]])
            clouds.append(np.concatenate([first, second, transition]))
            names.append(f"{kind}_{angle:+d}")
            construction_truth.append(dict(kind=kind, radius=2.0, mixture=kind == "concentric_mixture"))
    for angles, name in (((90, 90), "adjacent_U"), ((90, -90), "adjacent_Z")):
        fixture = make_adjacent_bend_fixture(
            angles_degrees=angles, radii=(2.0, 3.0),
            flange_lengths=(2.0, 6.0, 2.0), axial_extent=2.0,
        )
        clouds.append(fixture["points"])
        names.append(name)
        # Support indices and radii are translation invariant. The original
        # fixture is test-only metadata; classify_result receives translated
        # observations and compares row membership, not unshifted coordinates.
        construction_truth.append(dict(kind="adjacent", fixture=fixture))
    items = []
    for index, (name, cloud) in enumerate(zip(names, clouds)):
        offset = np.array([50.0*index, 0.0, 0.0])
        points = cloud+offset
        lower, upper = offset+[-16.0, -4.0, -16.0], offset+[16.0, 4.0, 16.0]
        assert np.all(points > lower) and np.all(points < upper)
        assert 0 < len(points) <= 256
        items.append(dict(name=name, points=points, lower=lower, upper=upper,
                          test_truth=construction_truth[index]))
    assert [len(item["points"]) for item in items] == [85, 85, 80, 80, 120, 120, 145, 145]
    assert sum(len(item["points"]) for item in items) == RESOURCE_SELECTED_ROWS
    return items


def interleaved_foreground(items, total_rows):
    """Independent row-ID schedule, round-robin windows and increasing local IDs."""
    ordered = [(window_index, local_index)
               for local_index in range(max(len(item["points"]) for item in items))
               for window_index, item in enumerate(items) if local_index < len(item["points"])]
    count = len(ordered)
    assert total_rows > 2*count
    # Integer arithmetic fixes the same rows independently of chunk boundaries.
    global_ids = ((np.arange(count, dtype=np.int64)+1)*(total_rows-1))//(count+1)
    assert len(np.unique(global_ids)) == count
    points = np.array([items[window]["points"][local] for window, local in ordered])
    ids_by_window = [[] for _ in items]
    for row_id, (window, _) in zip(global_ids, ordered):
        ids_by_window[window].append(int(row_id))
    for index, row_ids in enumerate(ids_by_window):
        assert len(row_ids) == len(items[index]["points"])
        assert row_ids[0] < total_rows//20
    return dict(global_ids=global_ids, points=points, ids_by_window=ids_by_window)


class InventedRowStream:
    """One-shot lazy chunks; only the small foreground oracle is retained.

    ``eof_observed`` becomes true only when the consumer asks past the final
    yielded chunk. A transparent core spy can therefore detect premature work.
    """

    def __init__(self, items, *, total_rows=RESOURCE_TOTAL_ROWS,
                 chunk_rows=RESOURCE_CHUNK_ROWS, late_invalid=False):
        self.items = items
        self.total_rows = total_rows
        self.chunk_rows = chunk_rows
        self.schedule = interleaved_foreground(items, total_rows)
        self.late_invalid = late_invalid
        self.started = False
        self.eof_observed = False
        self.chunks_yielded = 0
        self.rows_yielded = 0
        self.maximum_yielded_chunk_rows = 0

    def __iter__(self):
        assert not self.started, "the invented scanner stream must be consumed once"
        self.started = True
        foreground_ids = self.schedule["global_ids"]
        for start in range(0, self.total_rows, self.chunk_rows):
            stop = min(start+self.chunk_rows, self.total_rows)
            ids = np.arange(start, stop, dtype=np.int64)
            # Always outside every window along Y; no all-scan array is built.
            chunk = np.empty((len(ids), 3), dtype=np.float64)
            chunk[:, 0] = (ids % 997)/997.0
            chunk[:, 1] = 1000.0+(ids % 31)/31.0
            chunk[:, 2] = (ids % 101)/101.0
            lo, hi = np.searchsorted(foreground_ids, [start, stop])
            chunk[foreground_ids[lo:hi]-start] = self.schedule["points"][lo:hi]
            self.chunks_yielded += 1
            self.rows_yielded += len(chunk)
            self.maximum_yielded_chunk_rows = max(self.maximum_yielded_chunk_rows, len(chunk))
            yield chunk
        if self.late_invalid:
            # A separate trailing chunk arrives AFTER all valid rows have been
            # accepted, including every complete ROI payload. Its Y coordinate
            # is outside every window; invalid X must still reject the batch.
            self.chunks_yielded += 1
            self.rows_yielded += 1
            yield np.array([[np.nan, 1000.0, 0.0]])
        self.eof_observed = True


def process_lifetime_peak_rss_bytes():
    """ru_maxrss is bytes on macOS, KiB on Linux; it is not a local delta."""
    measured = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return measured if sys.platform == "darwin" else measured*1024


def fixed_engine_policy():
    """The retained radius-2/mm policy, unchanged for every selected window."""
    return dict(
        unit="mm", position_error_bound=1e-6, neighbor_counts=[8],
        max_neighborhood_radius=1.5, max_normal_angle_bound=0.1,
        min_region_points=9, min_seed_points=3, max_regions=64,
        graph_neighbor_count=10, max_link_distance=1.5,
        max_axis_angle_bound=0.1, max_axial_gap=0.51, max_radius_width=0.1,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255, max_plane_evaluations=2048,
        graph_symmetry="union", max_center_evaluations=256,
    )


def declared_windows(items):
    # Opaque identifiers, not the construction names retained by this test.
    return [dict(window_id=f"window_{index}", lower=item["lower"].tolist(),
                 upper=item["upper"].tolist()) for index, item in enumerate(items)]


def dispatch(stream, items, **changes):
    policy = dict(
        windows=declared_windows(items), engine_policy=fixed_engine_policy(),
        boundary_margin=2.0, max_total_rows=RESOURCE_TOTAL_ROWS,
        max_chunk_rows=65_536, max_chunks=4096, max_result_bytes=16*1024**2,
    )
    policy.update(changes)
    return streamed.analyze_streamed_window_bends(stream, **policy)


class CoreTrace:
    """Transparent test instrumentation; no core outputs/policies are changed."""

    def __init__(self, stream):
        self.stream = stream
        self.actual = streamed.analyze_tangency_audited_bend_candidates
        self.validation_calls = 0
        self.geometry_points = []
        self.geometry_policies = []

    def __call__(self, points, **policy):
        if len(points) == 0:
            # This separate policy-validation call fits no geometry and must
            # happen before consuming the producer, not be hidden in the cost.
            assert not self.stream.started
            self.validation_calls += 1
        else:
            assert self.stream.eof_observed, "geometry was called before successful EOF"
            self.geometry_points.append(np.array(points, copy=True))
            self.geometry_policies.append(policy)
        return self.actual(points, **policy)


def traced_dispatch(stream, items, **changes):
    trace = CoreTrace(stream)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(streamed, "analyze_tangency_audited_bend_candidates", trace)
        result = dispatch(stream, items, **changes)
    assert trace.validation_calls == 1
    assert result["diagnostics"]["policy_validation_empty_engine_calls"] == 1
    assert result["diagnostics"]["local_engine_calls"] == len(trace.geometry_points)
    return result, trace


def assert_research_scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    policy = result["policy"]
    assert policy["max_raw_rows_per_window"] == 256
    assert policy["max_windows"] == 16
    assert policy["engine_policy"] == fixed_engine_policy()
    assert policy["downsampling_or_residual_trimming"] is False
    assert policy["analysis_before_stream_eof"] is False
    assert policy["cross_window_event_merging"] is False
    assert policy["any_window_failure_clears_all_analyses"] is True
    assert policy["producer_allocation_or_blocking_time_is_bounded"] is False
    assert policy["local_analysis_scope"] == "DECLARED_OBSERVED_WINDOW_ONLY_NOT_GLOBAL_ENGINE_EQUIVALENCE"
    assert "SCANNER_SCALE_INGESTION_NOT_DENSE_FULL_SCAN_DISCOVERY" in result["limitations"]
    assert "LOCAL_WINDOW_GRAPH_NOT_GLOBAL_GRAPH_OR_COMPONENT_EQUIVALENCE" in result["limitations"]
    assert "EOF_IS_COMPLETE_SUPPLIED_STREAM_NOT_PHYSICAL_CAPTURE_COVERAGE" in result["limitations"]
    assert "OVERLAPPING_WINDOW_RESULTS_NOT_ADDITIVE_PHYSICAL_EVENT_COUNTS" in result["limitations"]


def assert_exact_isolated_outputs_and_global_ids(result, trace, stream, items, oracles):
    """Use construction rows and integer schedule, not router membership output."""
    assert result["search_complete_within_declared_policy"] is True
    assert result["stream_fully_consumed"] is stream.eof_observed is True
    assert len(result["analyses"]) == len(items) == len(trace.geometry_points)
    assert len(result["window_dispositions"]) == len(items)
    mapped_rows = []
    for index, (item, oracle, payload, state) in enumerate(zip(
            items, oracles, result["analyses"], result["window_dispositions"])):
        assert payload["window_id"] == state["window_id"] == f"window_{index}"
        assert payload["analysis"] == oracle
        assert payload["lower"] == item["lower"].tolist()
        assert payload["upper"] == item["upper"].tolist()
        np.testing.assert_array_equal(trace.geometry_points[index], item["points"])
        assert trace.geometry_policies[index] == fixed_engine_policy()
        expected_rows = stream.schedule["ids_by_window"][index]
        assert len(expected_rows) == len(item["points"])
        coordinate_to_global = {tuple(point): row for point, row in zip(item["points"], expected_rows)}
        assert len(coordinate_to_global) == len(expected_rows)
        expected_mapping = [[coordinate_to_global[tuple(position)]]
                            for position in oracle["base_analysis"]["positions"]]
        assert payload["canonical_to_stream_rows"] == expected_mapping
        actual_rows = [row for group in payload["canonical_to_stream_rows"] for row in group]
        assert sorted(actual_rows) == expected_rows
        mapped_rows.extend(actual_rows)
        assert state["status"] == "LOCAL_DIAGNOSTIC_COMPLETE"
        assert state["local_engine_complete"] is True
        assert state["selected_raw_count"] == len(item["points"])
        assert state["boundary_sensitive_raw_count"] == 0
        assert payload["scope"] == "LOCAL_OBSERVED_WINDOW_DIAGNOSTIC_NOT_EXPECTED_OR_OBSERVED_PHYSICAL_EVENT"
    assert sorted(mapped_rows) == stream.schedule["global_ids"].tolist()
    assert len(set(mapped_rows)) == sum(len(item["points"]) for item in items)


def full_constructed_support_radius_receipt(item, audit):
    """Score only exact complete test-support matches; retain every alternative.

    This consumes no analyzer calls. Labels and the original adjacent fixture
    enter only the independently retained test oracles after the full raw-window
    engine output is available. Missing arc stations do not imply full physical
    coverage, and adjacent events have radii 2 AND 3, not one global radius 2.
    """
    observed, truth, result = item["points"], item["test_truth"], audit["base_analysis"]
    assert audit["search_complete_within_declared_policy"] is True
    assert result["search_complete_within_declared_policy"] is True
    counts = Counter(
        windows=1, associations=len(result["candidates"]),
        conditional_associations=sum(c["status"] in CONDITIONAL_STATUSES for c in result["candidates"]),
        all_proposed_single_intervals=sum(c["single_radius_analysis"]["radius_interval"] is not None
                                         for c in result["candidates"]),
        all_proposed_band_intervals=sum(len(model["bands"]) for c in result["candidates"]
            if c["two_radius_analysis"] is not None for model in c["two_radius_analysis"]["models"]),
        checked_single_intervals=0, checked_band_intervals=0, containment_failures=0,
    )
    if truth["kind"] == "adjacent":
        support_receipt = classify_result(truth["fixture"], observed, result)
        # The older helper names the observed fit count "max_plane_evaluations".
        # Do not confuse this performed-work receipt with the configured 2,048 cap.
        support_receipt["plane_evaluation_count"] = support_receipt.pop("max_plane_evaluations")
        assert support_receipt["plane_evaluation_count"] <= fixed_engine_policy()["max_plane_evaluations"]
        events = support_receipt["events"]
        assert len(events) == len(truth["fixture"]["truth"]["bends"]) == 2
        assert not support_receipt["incomplete"]
        for event in events:
            assert event["classification"] == "COMPLETE_OBSERVED_SUPPORT_CONDITIONAL"
            assert event["conditional_matches"] >= 1
            assert event["checked_single_intervals"] >= 1
            assert event["containment_failures"] == 0
            assert event["physical_coverage_is_established"] is False
        counts.update(
            constructed_events=2, full_support_matched_events=2, conditional_full_support_matched_events=2,
            matched_associations=counts["associations"]-support_receipt["unmatched_associations"],
            unmatched_associations=support_receipt["unmatched_associations"],
            matched_conditional_associations=sum(event["conditional_matches"] for event in events),
            unmatched_conditional_associations=support_receipt["unmatched_conditional_associations"],
            checked_single_intervals=sum(event["checked_single_intervals"] for event in events),
            containment_failures=sum(event["containment_failures"] for event in events),
        )
    else:
        matches, _ = matched_candidates(observed, result, (25, 25))
        assert matches
        expected_status = ("CONDITIONAL_TWO_RADIUS_HYPOTHESIS" if truth["mixture"] else
                           "CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS")
        assert all(candidate["status"] == expected_status for candidate in matches)
        radius_receipt = radius_checks(observed, result, matches, truth["radius"], truth["mixture"])
        assert radius_receipt["containment_failures"] == 0
        if truth["mixture"]:
            assert radius_receipt["checked_single_intervals"] == 0
            assert radius_receipt["checked_band_intervals"] >= 2
        else:
            assert radius_receipt["checked_single_intervals"] >= 1
            assert radius_receipt["checked_band_intervals"] == 0
        for key in ("all_proposed_single_intervals", "all_proposed_band_intervals"):
            assert radius_receipt[key] == counts[key]
        counts.update(
            constructed_events=1, full_support_matched_events=1, conditional_full_support_matched_events=1,
            matched_associations=len(matches), unmatched_associations=counts["associations"]-len(matches),
            matched_conditional_associations=len(matches),
            unmatched_conditional_associations=counts["conditional_associations"]-len(matches),
            checked_single_intervals=radius_receipt["checked_single_intervals"],
            checked_band_intervals=radius_receipt["checked_band_intervals"],
            containment_failures=radius_receipt["containment_failures"],
        )
        support_receipt = dict(matched_candidate_ids=[candidate["candidate_id"] for candidate in matches],
                               checked_radius_or_radii=[2.0, 2.2] if truth["mixture"] else [2.0])
    counts["unscored_single_intervals"] = counts["all_proposed_single_intervals"]-counts["checked_single_intervals"]
    counts["unscored_band_intervals"] = counts["all_proposed_band_intervals"]-counts["checked_band_intervals"]
    assert counts["unscored_single_intervals"] >= 0 and counts["unscored_band_intervals"] >= 0
    assert counts["associations"] == counts["matched_associations"]+counts["unmatched_associations"]
    assert counts["conditional_associations"] == (counts["matched_conditional_associations"]
                                                  +counts["unmatched_conditional_associations"])
    return dict(construction=item["name"], scope="COMPLETE_CONSTRUCTED_SUPPORT_ORACLE_NOT_PHYSICAL_ACCURACY",
                counts=dict(counts), support_receipt=support_receipt)


@pytest.fixture(scope="module")
def resource_receipt():
    # Fixed at the declared 2M maximum before the first dispatcher run. Resource
    # Timing covers fixture/oracle construction, the nonempty streamed path,
    # and the complete-support oracle checks (which make no engine calls).
    # It is not reduced after observing either geometry or resource outcomes.
    start = time.perf_counter()
    items = fixed_window_fixture()
    oracles = [isolated_analyze(item["points"]) for item in items]
    stream = InventedRowStream(items)
    result, trace = traced_dispatch(stream, items)
    matched_support_receipts = [full_constructed_support_radius_receipt(item, payload["analysis"])
                               for item, payload in zip(items, result["analyses"])]
    matched_support_counts = Counter()
    for support in matched_support_receipts:
        matched_support_counts.update(support["counts"])
    elapsed = time.perf_counter()-start
    peak_rss = process_lifetime_peak_rss_bytes()
    candidate_counts, assessment_counts = Counter(), Counter()
    for payload in result["analyses"]:
        audit = payload["analysis"]
        candidate_counts.update(candidate["status"] for candidate in audit["base_analysis"]["candidates"])
        assessment_counts.update(item["analysis"]["status"] for item in audit["assessments"])
    receipt = dict(
        scope="SCANNER_SCALE_INGESTION_WITH_SPARSE_ROI_PAYLOAD_NOT_DENSE_FULL_SCAN_QUALIFICATION",
        total_rows=RESOURCE_TOTAL_ROWS, windows=len(items), selected_raw_rows=RESOURCE_SELECTED_ROWS,
        background_rows=RESOURCE_TOTAL_ROWS-RESOURCE_SELECTED_ROWS,
        nominal_chunk_rows=RESOURCE_CHUNK_ROWS, chunk_count=math.ceil(RESOURCE_TOTAL_ROWS/RESOURCE_CHUNK_ROWS),
        isolated_core_oracle_calls=len(oracles), dispatcher_local_core_calls=len(trace.geometry_points),
        empty_policy_validation_calls=trace.validation_calls,
        total_core_entry_calls=len(oracles)+len(trace.geometry_points)+trace.validation_calls,
        local_candidate_status_counts=dict(candidate_counts), local_assessment_status_counts=dict(assessment_counts),
        full_support_radius_counts=dict(matched_support_counts),
        full_support_radius_receipts=matched_support_receipts,
        elapsed_seconds=elapsed, elapsed_scope="FIXTURE_AND_8_DIRECT_ORACLES_AND_FULL_STREAMED_PATH_AND_SUPPORT_CHECKS",
        peak_rss_bytes=peak_rss, peak_rss_scope="CURRENT_PROCESS_LIFETIME_HIGH_WATER_NOT_DELTA_OR_PROCESS_TREE",
        soft_seconds_limit=RESOURCE_SECONDS_LIMIT, soft_peak_rss_bytes_limit=RESOURCE_PEAK_RSS_LIMIT,
        soft_time_acceptance=elapsed <= RESOURCE_SECONDS_LIMIT,
        soft_memory_acceptance=peak_rss <= RESOURCE_PEAK_RSS_LIMIT,
        unchanged_external_process_tree_hard_limit_bytes=UNCHANGED_PROCESS_TREE_HARD_LIMIT,
        result_payload_bytes=result["diagnostics"]["result_payload_bytes"],
    )
    print("STREAMED_WINDOW_RESOURCE_RECEIPT="+json.dumps(receipt, sort_keys=True, allow_nan=False))
    return dict(items=items, oracles=oracles, stream=stream, result=result, trace=trace, receipt=receipt)


def test_two_million_row_nonempty_ingestion_preserves_all_local_outputs_and_original_row_ids(resource_receipt):
    run = resource_receipt
    result, stream, receipt = run["result"], run["stream"], run["receipt"]
    assert_research_scope(result)
    assert_exact_isolated_outputs_and_global_ids(result, run["trace"], stream, run["items"], run["oracles"])
    assert result["input_row_count"] == stream.rows_yielded == RESOURCE_TOTAL_ROWS
    diagnostics = result["diagnostics"]
    assert diagnostics["processed_chunks"] == diagnostics["received_chunks"] == stream.chunks_yielded == 245
    assert diagnostics["peak_chunk_rows"] == stream.maximum_yielded_chunk_rows == RESOURCE_CHUNK_ROWS
    assert diagnostics["rows_in_any_window"] == diagnostics["total_window_row_memberships"] == RESOURCE_SELECTED_ROWS
    assert diagnostics["retained_window_rows"] == diagnostics["peak_retained_window_rows"] == RESOURCE_SELECTED_ROWS
    assert diagnostics["rows_outside_all_windows"] == 1_999_140
    assert diagnostics["rows_in_any_window"]+diagnostics["rows_outside_all_windows"] == RESOURCE_TOTAL_ROWS
    assert receipt["isolated_core_oracle_calls"] == receipt["dispatcher_local_core_calls"] == 8
    assert receipt["empty_policy_validation_calls"] == 1
    assert receipt["total_core_entry_calls"] == 17
    assert all(payload["analysis"]["base_analysis"]["candidates"] for payload in result["analyses"])
    assert sum(receipt["local_candidate_status_counts"].values()) > 0
    assert sum(receipt["local_assessment_status_counts"].values()) > 0
    support_counts = receipt["full_support_radius_counts"]
    assert len(receipt["full_support_radius_receipts"]) == support_counts["windows"] == 8
    assert support_counts["constructed_events"] == support_counts["full_support_matched_events"] == 10
    assert support_counts["conditional_full_support_matched_events"] == 10
    assert support_counts["checked_single_intervals"] >= 8
    assert support_counts["checked_band_intervals"] >= 4
    assert support_counts["containment_failures"] == 0
    assert support_counts["all_proposed_single_intervals"] == (
        support_counts["checked_single_intervals"]+support_counts["unscored_single_intervals"])
    assert support_counts["all_proposed_band_intervals"] == (
        support_counts["checked_band_intervals"]+support_counts["unscored_band_intervals"])
    for row_ids in stream.schedule["ids_by_window"]:
        assert row_ids[0] < RESOURCE_TOTAL_ROWS//20
        assert row_ids[-1] > 7*RESOURCE_TOTAL_ROWS//10
        assert len({row//RESOURCE_CHUNK_ROWS for row in row_ids}) > 1


def test_fixed_resource_soft_acceptance_is_separate_from_geometry_equivalence(resource_receipt):
    # Failure is a reported engineering budget miss, not an invitation to shrink
    # the fixture, widen a cap, hide an abstention or infer a physical defect.
    receipt = resource_receipt["receipt"]
    assert receipt["soft_time_acceptance"], receipt
    assert receipt["soft_memory_acceptance"], receipt


@pytest.mark.parametrize("repeated", [False, True])
def test_257_raw_rows_block_the_entire_batch_without_a_sampled_subset_or_geometry_call(repeated):
    first = fixed_window_fixture()[0]
    dense = (np.repeat([[50.0, 0.0, 0.0]], 257, axis=0) if repeated else
             np.column_stack([50.0+np.linspace(0.0, 1.0, 257), np.zeros((257, 2))]))
    items = [first, dict(name="invented_dense_raw_window", points=dense,
                         lower=np.array([34.0, -4.0, -16.0]), upper=np.array([66.0, 4.0, 16.0]))]
    stream = InventedRowStream(items, total_rows=4096, chunk_rows=257)
    result, trace = traced_dispatch(stream, items)
    assert_research_scope(result)
    assert result["stream_fully_consumed"] is stream.eof_observed is True
    assert not result["search_complete_within_declared_policy"]
    assert result["analyses"] == []
    assert result["reasons"] == ["UNSUPPORTED_OBSERVED_WINDOW"]
    assert trace.geometry_points == []
    assert result["diagnostics"]["local_engine_calls"] == 0
    states = result["window_dispositions"]
    assert states[0]["selected_raw_count"] == 85 and states[0]["status"] == "READY"
    assert states[1]["selected_raw_count"] == 257 and states[1]["status"] == "WINDOW_RAW_CAP_EXCEEDED"
    assert all(not state["local_engine_complete"] for state in states)
    assert result["diagnostics"]["rows_in_any_window"] == 342


def test_final_nonfinite_background_row_invalidates_all_preceding_valid_window_payloads():
    items = fixed_window_fixture()
    stream = InventedRowStream(items, total_rows=8192, chunk_rows=257, late_invalid=True)
    result, trace = traced_dispatch(stream, items)
    assert_research_scope(result)
    assert stream.rows_yielded == 8193
    assert not result["stream_fully_consumed"] and not stream.eof_observed
    assert not result["search_complete_within_declared_policy"]
    assert result["reasons"] == ["NONFINITE_CHUNK_COORDINATE"]
    assert result["analyses"] == []
    assert trace.geometry_points == []
    assert result["diagnostics"]["local_engine_calls"] == 0
    assert all(not state["local_engine_complete"] for state in result["window_dispositions"])
    assert [state["selected_raw_count"] for state in result["window_dispositions"]] == [
        85, 85, 80, 80, 120, 120, 145, 145,
    ]
    # The entire invalid chunk is rejected; its otherwise-valid rows are not
    # counted as accepted ingestion. Never call that a fully consumed scan.
    assert result["input_row_count"] == 8192
    assert result["diagnostics"]["processed_chunks"] == 32
    assert result["diagnostics"]["received_chunks"] == 33


def test_selected_boundary_row_censors_every_window_before_any_geometry_call():
    items = fixed_window_fixture()[:2]
    second = items[1]
    boundary_row = np.array([[second["lower"][0], 0.0, 0.0]])
    items[1] = dict(second, points=np.concatenate([second["points"], boundary_row]))
    stream = InventedRowStream(items, total_rows=4096, chunk_rows=257)
    result, trace = traced_dispatch(stream, items)
    assert_research_scope(result)
    assert result["stream_fully_consumed"] is stream.eof_observed is True
    assert not result["search_complete_within_declared_policy"]
    assert result["analyses"] == [] and trace.geometry_points == []
    assert result["reasons"] == ["UNSUPPORTED_OBSERVED_WINDOW"]
    first, second = result["window_dispositions"]
    assert first["status"] == "READY"
    assert second["status"] == "WINDOW_BOUNDARY_CENSORED"
    assert second["selected_raw_count"] == 86
    assert second["boundary_sensitive_raw_count"] == 1
    assert result["diagnostics"]["local_engine_calls"] == 0


def test_chunk_boundaries_do_not_change_any_selected_row_or_local_core_output():
    all_items = fixed_window_fixture()
    items = [all_items[0], all_items[4]]
    oracles = [isolated_analyze(item["points"]) for item in items]
    results = []
    for chunk_rows in (31, 257, 4096):
        stream = InventedRowStream(items, total_rows=4096, chunk_rows=chunk_rows)
        result, trace = traced_dispatch(stream, items)
        assert_research_scope(result)
        assert_exact_isolated_outputs_and_global_ids(result, trace, stream, items, oracles)
        assert result["diagnostics"]["rows_in_any_window"] == 205
        assert result["diagnostics"]["rows_outside_all_windows"] == 3891
        assert result["diagnostics"]["local_engine_calls"] == 2
        assert result["diagnostics"]["processed_chunks"] == math.ceil(4096/chunk_rows)
        results.append(result)
    for result in results[1:]:
        for key in ("analyses", "window_dispositions", "status", "reasons", "input_row_count",
                    "stream_fully_consumed", "search_complete_within_declared_policy", "policy"):
            assert result[key] == results[0][key]
