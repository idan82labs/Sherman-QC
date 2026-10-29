"""Independent synthetic stream/window contracts; no physical inputs or runtime.

The only spies transparently call the real pure geometry engine. They observe
ordering and attempted calls, never replace a geometric answer with a fixture.
"""
import json
import math

import numpy as np
import pytest

from research.geometry import streamed_window_bend_candidates as streamed
from research.geometry.position_graph_envelope import build_position_graph_envelope
from research.geometry.tangency_audited_bend_candidates import (
    analyze_tangency_audited_bend_candidates,
)


def policy(**changes):
    values = dict(
        unit="mm", position_error_bound=1e-6, neighbor_counts=[8],
        max_neighborhood_radius=1.5, max_normal_angle_bound=0.1,
        min_region_points=9, min_seed_points=3, max_regions=64,
        graph_neighbor_count=10, max_link_distance=1.5,
        max_axis_angle_bound=0.1, max_axial_gap=0.51, max_radius_width=0.1,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255,
        max_plane_evaluations=2048, graph_symmetry="union",
        max_center_evaluations=256,
    )
    values.update(changes)
    return values


def box(name="local", center=0.0):
    return dict(window_id=name, lower=[center-10., -10., -10.],
                upper=[center+10., 10., 10.])


def analyze(chunks, **changes):
    values = dict(windows=[box()], engine_policy=policy(), boundary_margin=2.,
                  max_total_rows=1000, max_chunk_rows=512, max_chunks=64,
                  max_result_bytes=16_777_216)
    values.update(changes)
    return streamed.analyze_streamed_window_bends(chunks, **values)


def triangle():
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])


def plane_grid():
    return np.asarray([[x, y, 0.] for x in range(4) for y in range(3)])


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_bend_count"] is None
    p = result["policy"]
    assert p["local_analysis_scope"] == "DECLARED_OBSERVED_WINDOW_ONLY_NOT_GLOBAL_ENGINE_EQUIVALENCE"
    assert p["window_membership"] == "CLOSED_BOX_ON_OBSERVED_COORDINATES_NOT_LATENT_CORRECTED_MEMBERSHIP"
    assert p["downsampling_or_residual_trimming"] is False
    assert p["cross_window_event_merging"] is False
    assert p["analysis_before_stream_eof"] is False
    assert p["any_window_failure_clears_all_analyses"] is True
    assert p["producer_allocation_or_blocking_time_is_bounded"] is False
    assert result["diagnostics"]["policy_validation_empty_engine_calls"] == 1


def failed(result, reason, *, eof, calls=0):
    scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["analyses"] == []
    assert result["reasons"] == [reason]
    assert result["search_complete_within_declared_policy"] is False
    assert result["stream_fully_consumed"] is eof
    assert result["diagnostics"]["local_engine_calls"] == calls
    assert result["diagnostics"]["result_payload_bytes"] == 0


def transparent_spy(monkeypatch, events):
    original = streamed.analyze_tangency_audited_bend_candidates

    def spy(points, **kwargs):
        events.append(("engine", len(points)))
        return original(points, **kwargs)

    monkeypatch.setattr(streamed, "analyze_tangency_audited_bend_candidates", spy)


def test_nonempty_engine_waits_for_actual_eof_and_matches_direct_engine(monkeypatch):
    events = []
    transparent_spy(monkeypatch, events)
    points = plane_grid()

    def source():
        events.append(("yield", 0))
        yield points[:5]
        events.append(("yield", 1))
        yield points[5:]
        events.append(("eof", None))

    result = analyze(source())
    scope(result)
    assert events == [("engine", 0), ("yield", 0), ("yield", 1),
                      ("eof", None), ("engine", len(points))]
    assert result["stream_fully_consumed"]
    assert result["search_complete_within_declared_policy"]
    assert result["analyses"][0]["analysis"] == analyze_tangency_audited_bend_candidates(points, **policy())
    assert result["diagnostics"]["local_engine_calls"] == 1


def test_chunk_partitioning_preserves_every_payload_and_stream_row_identity():
    points = np.vstack([triangle(), [[100., 0., 0.]], triangle()[[1, 0]]])
    partitions = [[points], [points[:1], points[1:4], points[4:]],
                  [points[i:i+1] for i in range(len(points))]]
    results = [analyze(chunks) for chunks in partitions]
    for result in results:
        scope(result)
        assert result["input_row_count"] == 6
        assert result["diagnostics"]["rows_in_any_window"] == 5
        assert result["diagnostics"]["rows_outside_all_windows"] == 1
        assert result["window_dispositions"][0]["selected_raw_count"] == 5
        payload = result["analyses"][0]
        mapping = dict(zip(map(tuple, payload["analysis"]["base_analysis"]["positions"]),
                           payload["canonical_to_stream_rows"]))
        assert mapping == {(0., 0., 0.): [0, 5], (1., 0., 0.): [1, 4], (0., 1., 0.): [2]}
    assert results[0]["analyses"] == results[1]["analyses"] == results[2]["analyses"]
    assert results[0]["window_dispositions"] == results[1]["window_dispositions"] == results[2]["window_dispositions"]


def test_permuting_stream_changes_only_raw_row_receipts_not_local_geometry():
    points = np.vstack([triangle(), triangle()[[1]], [[100., 0., 0.]]])
    order = [4, 3, 0, 2, 1]
    first, second = analyze([points]), analyze([points[order]])
    assert first["analyses"][0]["analysis"] == second["analyses"][0]["analysis"]
    for result, observed in [(first, points), (second, points[order])]:
        payload = result["analyses"][0]
        for position, rows in zip(payload["analysis"]["base_analysis"]["positions"],
                                  payload["canonical_to_stream_rows"]):
            assert rows == [i for i, row in enumerate(observed) if np.array_equal(row, position)]


def test_overlaps_retain_independent_memberships_not_additive_physical_events():
    points = triangle()
    result = analyze([points], windows=[box("a"), box("b", 1.)])
    scope(result)
    assert result["diagnostics"]["rows_in_any_window"] == 3
    assert result["diagnostics"]["total_window_row_memberships"] == 6
    assert result["diagnostics"]["peak_retained_window_rows"] == 6
    assert result["diagnostics"]["local_engine_calls"] == 2
    assert [item["window_id"] for item in result["analyses"]] == ["a", "b"]
    assert result["analyses"][0]["analysis"] == result["analyses"][1]["analysis"]
    assert result["analyses"][0]["canonical_to_stream_rows"] == result["analyses"][1]["canonical_to_stream_rows"]


def test_raw_repeat_cap_counts_across_chunks_before_exact_deduplication(monkeypatch):
    events = []
    transparent_spy(monkeypatch, events)
    result = analyze([np.zeros((128, 3)), np.zeros((128, 3)), np.zeros((1, 3))])
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert events == [("engine", 0)]
    assert result["window_dispositions"][0]["selected_raw_count"] == 257
    assert result["window_dispositions"][0]["status"] == "WINDOW_RAW_CAP_EXCEEDED"
    assert result["diagnostics"]["peak_retained_window_rows"] == 256
    assert result["diagnostics"]["retained_window_rows"] == 0


def test_exact_raw_cap_retains_all_duplicate_source_rows():
    result = analyze([np.zeros((256, 3))])
    scope(result)
    assert result["search_complete_within_declared_policy"]
    payload = result["analyses"][0]
    assert payload["analysis"]["base_analysis"]["positions"] == [[0., 0., 0.]]
    assert payload["canonical_to_stream_rows"] == [list(range(256))]


def test_later_overlapping_window_overflow_prevents_even_a_good_first_call(monkeypatch):
    events = []
    transparent_spy(monkeypatch, events)
    windows = [box("small"), dict(window_id="large", lower=[-10., -10., -10.], upper=[40., 10., 10.])]
    points = np.vstack([np.zeros((1, 3)), np.tile([20., 0., 0.], (256, 1))])
    result = analyze([points[:100], points[100:]], windows=windows)
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert events == [("engine", 0)]
    assert [s["selected_raw_count"] for s in result["window_dispositions"]] == [1, 257]
    assert result["diagnostics"]["total_window_row_memberships"] == 258


@pytest.mark.parametrize("coordinate", [-10., 10.])
def test_closed_box_faces_are_selected_and_censored(coordinate):
    result = analyze([np.asarray([[coordinate, 0., 0.]])])
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert result["window_dispositions"][0]["selected_raw_count"] == 1
    assert result["window_dispositions"][0]["boundary_sensitive_raw_count"] == 1


def test_immediately_outside_closed_box_is_not_latent_membership_or_coverage():
    point = np.asarray([[np.nextafter(10., math.inf), 0., 0.]])
    result = analyze([point])
    scope(result)
    assert result["search_complete_within_declared_policy"]
    assert result["analyses"] == []
    assert result["diagnostics"]["rows_outside_all_windows"] == 1
    assert result["window_dispositions"][0]["status"] == "EMPTY_OBSERVED_WINDOW"


@pytest.mark.parametrize("extra_guard_units,censored", [(0., True), (0.5, True), (2., False)])
def test_inner_margin_is_strict_and_includes_engineering_allowance(extra_guard_units, censored):
    guard = 512*np.finfo(float).eps*(10.+math.hypot(20., 20., 20.))
    point = np.asarray([[8.-extra_guard_units*guard, 0., 0.]])
    result = analyze([point])
    state = result["window_dispositions"][0]
    assert state["numerical_boundary_allowance"] == guard
    if censored:
        failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
        assert state["status"] == "WINDOW_BOUNDARY_CENSORED"
    else:
        scope(result)
        assert result["search_complete_within_declared_policy"]
        assert state["boundary_sensitive_raw_count"] == 0


def test_unrelated_boundary_row_censors_all_windows_before_geometry(monkeypatch):
    events = []
    transparent_spy(monkeypatch, events)
    points = np.vstack([triangle(), triangle()+[40., 0., 0.], [[48., 0., 0.]]])
    result = analyze([points], windows=[box("first"), box("second", 40.)])
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert events == [("engine", 0)]
    assert [s["boundary_sensitive_raw_count"] for s in result["window_dispositions"]] == [0, 1]
    assert result["window_dispositions"][1]["selected_raw_count"] == 4


def test_halo_chain_cannot_be_shortened_to_a_favorable_interior_support():
    chain = np.asarray([[float(x), 0., 0.] for x in range(9)])
    result = analyze([chain[:4], chain[4:]])
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert result["window_dispositions"][0]["selected_raw_count"] == 9
    assert result["window_dispositions"][0]["boundary_sensitive_raw_count"] == 1


@pytest.mark.parametrize("tail,reason", [
    (np.asarray([[100., math.nan, 0.]]), "NONFINITE_CHUNK_COORDINATE"),
    (np.asarray([[100., math.inf, 0.]]), "NONFINITE_CHUNK_COORDINATE"),
    (np.zeros((2, 2)), "INVALID_CHUNK_ARRAY"),
    (np.zeros((2, 3), dtype=bool), "INVALID_CHUNK_ARRAY"),
    (np.zeros((2, 3), dtype=complex), "INVALID_CHUNK_ARRAY"),
    (np.zeros((2, 3), dtype=object), "INVALID_CHUNK_ARRAY"),
    ([[100., 0., 0.]], "INVALID_CHUNK_ARRAY"),
])
def test_invalid_late_chunk_including_outside_coordinates_clears_everything(monkeypatch, tail, reason):
    events = []
    transparent_spy(monkeypatch, events)
    result = analyze([triangle(), tail])
    failed(result, reason, eof=False)
    assert events == [("engine", 0)]
    assert result["input_row_count"] == 3
    assert result["diagnostics"]["received_chunks"] == 2
    assert result["diagnostics"]["processed_chunks"] == 1


def test_iterator_failure_after_selected_rows_is_not_eof(monkeypatch):
    events = []
    transparent_spy(monkeypatch, events)

    def source():
        yield triangle()
        raise RuntimeError("invented producer failure")

    result = analyze(source())
    failed(result, "INPUT_STREAM_FAILED", eof=False)
    assert events == [("engine", 0)]
    assert result["input_row_count"] == 3


def test_iterator_creation_failure_is_sanitized_and_never_calls_nonempty_core():
    class BrokenIterator:
        def __iter__(self):
            raise RuntimeError("invented iterator-creation failure")

    result = analyze(BrokenIterator())
    failed(result, "INPUT_STREAM_FAILED", eof=False)
    assert result["diagnostics"]["received_chunks"] == 0
    with pytest.raises(ValueError, match="explicit cooperative iterable"):
        analyze(None)


def test_empty_stream_nonempty_outside_stream_and_failed_stream_are_distinct():
    empty = analyze([])
    outside = analyze([np.asarray([[100., 0., 0.]])])

    def broken():
        raise RuntimeError("invented failure before any rows")
        yield np.empty((0, 3))

    failure = analyze(broken())
    for result, count in [(empty, 0), (outside, 1)]:
        scope(result)
        assert result["input_row_count"] == count
        assert result["stream_fully_consumed"]
        assert result["search_complete_within_declared_policy"]
        assert result["reasons"] == ["NO_OBSERVED_ROWS_IN_DECLARED_WINDOWS"]
        assert result["analyses"] == []
        assert result["diagnostics"]["local_engine_calls"] == 0
    failed(failure, "INPUT_STREAM_FAILED", eof=False)


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.uint64, np.float16, np.float32, np.float64])
def test_bounded_numeric_arrays_are_accepted_without_changing_coordinates(dtype):
    result = analyze([triangle().astype(dtype)])
    scope(result)
    assert result["analyses"][0]["analysis"]["base_analysis"]["positions"] == [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]


@pytest.mark.parametrize("coordinate,dtype", [(2**53+1, np.int64), (-(2**53+1), np.int64), (2**53+1, np.uint64)])
def test_inexact_integer_conversion_is_rejected_even_outside_all_windows(coordinate, dtype):
    result = analyze([np.asarray([[coordinate, 0, 0]], dtype=dtype)])
    failed(result, "INTEGER_COORDINATE_PRECISION_UNSUPPORTED", eof=False)


def test_exact_integer_precision_endpoint_outside_windows_remains_valid():
    result = analyze([np.asarray([[2**53, 0, 0], [-2**53, 0, 0]], dtype=np.int64)])
    scope(result)
    assert result["stream_fully_consumed"]
    assert result["input_row_count"] == 2
    assert result["diagnostics"]["rows_outside_all_windows"] == 2


def test_total_row_cap_includes_outside_rows_and_requires_eof_at_exact_cap():
    chunks = [triangle(), np.asarray([[100., 0., 0.]])]
    accepted = analyze(chunks, max_total_rows=4)
    assert accepted["stream_fully_consumed"]
    assert accepted["input_row_count"] == 4
    rejected = analyze(chunks, max_total_rows=3)
    failed(rejected, "TOTAL_ROW_BUDGET_EXCEEDED", eof=False)
    assert rejected["input_row_count"] == 3
    assert rejected["diagnostics"]["local_engine_calls"] == 0


def test_chunk_row_cap_is_checked_before_spatial_filtering():
    result = analyze([np.tile([100., 0., 0.], (4, 1))], max_chunk_rows=3)
    failed(result, "CHUNK_ROW_BUDGET_EXCEEDED", eof=False)
    assert result["input_row_count"] == 0


def test_empty_chunks_consume_count_budget_and_cannot_form_an_unbounded_bypass():
    accepted = analyze([np.empty((0, 3)), np.empty((0, 3))], max_chunks=2)
    assert accepted["stream_fully_consumed"]
    assert accepted["diagnostics"]["processed_chunks"] == 2
    rejected = analyze([triangle(), np.empty((0, 3)), np.empty((0, 3))], max_chunks=2)
    failed(rejected, "CHUNK_COUNT_BUDGET_EXCEEDED", eof=False)
    assert rejected["diagnostics"]["received_chunks"] == 3
    assert rejected["diagnostics"]["processed_chunks"] == 2


def test_producer_mutation_cannot_change_copied_windows_policy_or_buffered_rows():
    lower, upper = np.full(3, -10.), np.full(3, 10.)
    windows = [dict(window_id="snapshot", lower=lower, upper=upper)]
    declared = policy(neighbor_counts=np.asarray([8], dtype=np.int64))
    declared["position_error_bound"] = np.float64(1e-6)
    reusable = triangle()

    def source():
        yield reusable
        reusable[:] = [100., 0., 0.]
        lower[:] = 99.
        upper[:] = 101.
        windows[0]["window_id"] = "mutated"
        declared["neighbor_counts"][0] = 255
        declared["unit"] = "invalid"
        yield reusable

    result = analyze(source(), windows=windows, engine_policy=declared)
    scope(result)
    assert result["unit"] == "mm"
    assert result["policy"]["engine_policy"]["neighbor_counts"] == [8]
    assert result["input_row_count"] == 6
    assert result["diagnostics"]["rows_outside_all_windows"] == 3
    payload = result["analyses"][0]
    assert payload["window_id"] == "snapshot"
    assert payload["lower"] == [-10., -10., -10.]
    assert payload["upper"] == [10., 10., 10.]
    assert result["declared_windows"] == [dict(window_id="snapshot", lower=[-10.]*3, upper=[10.]*3)]
    assert payload["analysis"] == analyze_tangency_audited_bend_candidates(triangle(), **policy())
    assert payload["canonical_to_stream_rows"] == [[0], [2], [1]]


def test_late_core_incompleteness_clears_an_already_computed_first_payload():
    points = np.vstack([np.zeros((1, 3)), plane_grid()+[40., 0., 0.]])
    result = analyze([points], windows=[box("first"), box("second", 40.)],
                     engine_policy=policy(max_plane_evaluations=1))
    failed(result, "WINDOW_ENGINE_INCOMPLETE", eof=True, calls=2)
    assert result["window_dispositions"][0]["local_engine_complete"] is True
    assert result["window_dispositions"][1]["status"] == "WINDOW_ENGINE_INCOMPLETE"
    assert result["diagnostics"]["discarded_result_payload_bytes"] > 0


def test_payload_byte_budget_is_exact_and_does_not_return_a_successful_prefix():
    first = analyze([triangle()])
    payload = first["analyses"][0]
    size = len(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    assert first["diagnostics"]["result_payload_bytes"] == size
    exact = analyze([triangle()], max_result_bytes=size)
    assert exact["analyses"] == first["analyses"]
    small = analyze([triangle()], max_result_bytes=size-1)
    failed(small, "RESULT_PAYLOAD_BUDGET_EXCEEDED", eof=True, calls=1)
    assert small["diagnostics"]["result_payload_bytes"] == 0
    assert small["diagnostics"]["discarded_result_payload_bytes"] == 0
    assert small["diagnostics"]["serialized_window_payload_bytes"] == size
    both = analyze([triangle()], windows=[box(), box("other")], max_result_bytes=size)
    failed(both, "RESULT_PAYLOAD_BUDGET_EXCEEDED", eof=True, calls=2)
    assert both["diagnostics"]["discarded_result_payload_bytes"] == size
    assert both["diagnostics"]["serialized_window_payload_bytes"] > size
    assert both["window_dispositions"][0]["local_engine_complete"] is True


def test_boundary_numerical_range_failure_is_atomic_before_core():
    huge = dict(window_id="huge", lower=[-5e307]*3, upper=[5e307]*3)
    # Each axis span is representable; the 3-D norm plus coordinate magnitude
    # is not. The dispatcher must abstain rather than omit a finite guard.
    result = analyze([np.zeros((1, 3))], windows=[box(), huge])
    failed(result, "UNSUPPORTED_OBSERVED_WINDOW", eof=True)
    assert result["window_dispositions"][1]["status"] == "WINDOW_BOUNDARY_NUMERICAL_UNSUPPORTED"


def test_interior_window_graph_is_not_the_full_graph_even_without_boundary_samples():
    points = triangle()
    declared = policy(graph_neighbor_count=3)
    routed = analyze([np.vstack([points, [[100., 0., 0.]]])], engine_policy=declared)
    scope(routed)
    assert routed["search_complete_within_declared_policy"]
    local_graph = routed["analyses"][0]["analysis"]["base_analysis"]["graph"]
    assert local_graph["reasons"] == ["INSUFFICIENT_OTHER_POSITIONS"]
    assert local_graph["edges"] == []
    whole_graph = build_position_graph_envelope(
        np.vstack([points, [[100., 0., 0.]]]), unit="mm", position_error_bound=1e-6,
        neighbor_count=3, max_link_distance=1.5, symmetry="union",
    )
    # The remote point changes n<=k without itself joining a capped edge.
    assert whole_graph["edges"] == [[0, 1], [0, 2], [1, 2]]
    assert all(3 not in edge for edge in whole_graph["edges"])
    assert routed["window_dispositions"][0]["boundary_sensitive_raw_count"] == 0
    assert routed["diagnostics"]["rows_outside_all_windows"] == 1


@pytest.mark.parametrize("name,value", [
    ("max_total_rows", 0), ("max_total_rows", 2_000_001), ("max_total_rows", True),
    ("max_chunk_rows", 0), ("max_chunk_rows", 65_537), ("max_chunk_rows", 3.5),
    ("max_chunks", 0), ("max_chunks", 4097),
    ("max_result_bytes", 0), ("max_result_bytes", 16_777_217),
    ("boundary_margin", 0.), ("boundary_margin", math.inf),
    ("boundary_margin", True), ("boundary_margin", 1.5),
])
def test_invalid_dispatch_policy_is_rejected_before_iterating_producer(name, value):
    class NeverRead:
        def __iter__(self):
            raise AssertionError("invalid policy consumed producer")

    with pytest.raises(ValueError):
        analyze(NeverRead(), **{name: value})


@pytest.mark.parametrize("changes", [
    {"unit": "px"}, {"neighbor_counts": [True]},
    {"max_center_evaluations": 257}, {"max_plane_evaluations": 0},
])
def test_invalid_inherited_engine_policy_is_rejected_before_producer(changes):
    class NeverRead:
        def __iter__(self):
            raise AssertionError("invalid engine policy consumed producer")

    with pytest.raises(ValueError):
        analyze(NeverRead(), engine_policy=policy(**changes))


@pytest.mark.parametrize("windows", [
    [], [box(str(i)) for i in range(17)], [box("same"), box("same")],
    [box("not a valid name")], [dict(box(), extra=1)],
    [dict(window_id="bad", lower=[0., 0., 0.], upper=[0., 1., 1.])],
    [dict(window_id="bad", lower=[0., 0., 0.], upper=[1., math.inf, 1.])],
    [dict(window_id="bad", lower=[False, 0., 0.], upper=[1., 1., 1.])],
    [dict(window_id="bad", lower=[0., 0.], upper=[1., 1., 1.])],
    [dict(window_id="bad", lower=[-1e308]*3, upper=[1e308]*3)],
])
def test_invalid_windows_are_rejected_before_iterating_producer(windows):
    class NeverRead:
        def __iter__(self):
            raise AssertionError("invalid windows consumed producer")

    with pytest.raises(ValueError):
        analyze(NeverRead(), windows=windows)


def test_sixteen_empty_declared_windows_are_supported_without_physical_claims():
    result = analyze([], windows=[box(str(i)) for i in range(16)])
    scope(result)
    assert result["search_complete_within_declared_policy"]
    assert len(result["window_dispositions"]) == 16
    assert all(s["status"] == "EMPTY_OBSERVED_WINDOW" for s in result["window_dispositions"])
    assert result["diagnostics"]["local_engine_calls"] == 0
