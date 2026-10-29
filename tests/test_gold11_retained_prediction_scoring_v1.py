"""74 prospective invented-data scorer controls; no physical/cached input reads.

These tests score independently specified line segments, never run inference or
authenticate a historical cohort. The 4096-row controls exercise normalization
only, avoiding a large assignment matrix. Source is frozen by root's later exact
review/protocol, not by this author claiming an unexecuted test passed.
"""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_FRAME = "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
UNITS = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"
CARRIER = "MEASURED_INNER_ARC_MID_ANGLE_GENERATOR"
COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6)
CASE = "invented:00"
ALIASES = (
    "span_endpoints", "anchor", "axis_direction", "span_mm", "endpoints", "start", "end",
    "p0", "p1", "lineStart", "lineEnd",
)


@pytest.fixture(scope="module")
def scoring():
    """Isolated reviewed scripts namespace; no CLI or application initializer."""
    prefix = "_independent_retained_scoring_contract"
    package = types.ModuleType(prefix)
    package.__path__ = [str(ROOT / "scripts")]
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix + ".gold11_retained_prediction_scoring_v1")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)


def _line(identifier, y, *, start=0., end=100., z=0.):
    return dict(bend_id=identifier, line_start=[start, y, z], line_end=[end, y, z])


def _cached(rows):
    return dict(schema_version=1, kind="sherman_qc_f1_cad_guided_scan_line_predictions",
        route="blind_cad_ply_detected_lines", research_only=True,
        cad_guided_zero_shot=True, source_only=False, predictions=deepcopy(rows))


def _engine(rows):
    return dict(status="PARTIAL", phase="COMPLETE", observed_bend_count=None,
        production_use_allowed=False, conformance="INDETERMINATE", uncertainty_status="UNCALIBRATED",
        prediction_frame=RAW_FRAME, units_status=UNITS,
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE",
        predictions=[dict(deepcopy(row), unit="mm", reference_frame=RAW_FRAME,
                          line_convention=CARRIER) for row in rows])


def _cohort():
    """Unequal manual denominators: families contain 10,12,14,16,12,7 references."""
    families, references, cached, engine = {}, {}, {}, {}
    for i, count in enumerate(COUNTS):
        case = f"invented:{i:02d}"
        families[case] = f"family:{i % 6}"
        rows = [_line(f"line:{j}", 50.*j) for j in range(count)]
        references[case] = rows
        cached[case], engine[case] = _cached(rows), _engine(rows)
    return dict(case_family_by_id=families, reference_records_by_case=references,
        cached_results_by_case=cached, engine_results_by_case=engine)


def _score(scoring, arguments):
    return scoring.score_gold11_retained_predictions_v1(**arguments)


def _assert_zero_case(report, *, lane="cached_control", disposition="MALFORMED_RESULT"):
    row = report[lane]["case_dispositions"][CASE]
    assert row["disposition"] == disposition
    assert (row["gold_count"], row["prediction_count"], row["true_positive"],
            row["false_positive"], row["false_negative"]) == (2, 0, 0, 0, 2)
    assert row["precision"] is None and row["recall"] == 0. and row["f1"] is None
    assert row["observed_bend_count"] is None
    assert report[lane]["geometry"]["case_count"] == 11
    assert report[lane]["geometry"]["reference_line_count"] == 71
    assert len(report[lane]["families"]) == 6
    assert report[lane]["geometry"]["true_positive"] == 69
    assert report[lane]["geometry"]["false_negative"] == 2
    return row


def test_perfect_geometry_is_not_historical_replay_measured_cache_or_physical_authority(scoring):
    arguments = _cohort()
    before = deepcopy(arguments)
    report = _score(scoring, arguments)
    assert arguments == before
    assert (report["case_count"], report["family_count"], report["reference_line_count"]) == (11, 6, 71)
    for lane in ("cached_control", "engine"):
        metrics = report[lane]["geometry"]
        assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (71, 0, 0)
        assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.
        assert metrics["exact_localized_target_set_accuracy"] == 1.
        assert report[lane]["minimum_family_recall"] == 1.
    cached = report["cached_control"]
    assert cached["geometry"]["cached_prediction_line_count"] == 71
    assert "emitted_support_line_count" not in cached["geometry"]
    assert cached["lane_context"] == "CACHED_SCAN_FRAME_GATED_CAD_NOT_FRESH_INFERENCE"
    assert cached["carrier_semantics"] == "LEGACY_CAD_PREDICTED_LINE_NOT_ASSERTED_MEASURED_SURFACE_CARRIER"
    assert cached["inference_execution"] == "NOT_EVALUATED_RETAINED_PAYLOAD_ONLY"
    assert cached["geometry"]["count_metric_semantics"] == "CACHED_LINE_CARDINALITY_NOT_PHYSICAL_OBSERVED_BENDS"
    assert report["engine"]["physical_observed_bend_count"]["unknown_case_count"] == 11
    assert report["control_identity"] == "NEW_CACHED_CAD_CONTROL_NOT_HISTORICAL_FUSION_REPLAY"
    assert report["historical_score_comparability"] == "UNESTABLISHED"
    assert report["observed_bend_count"] is None
    assert report["strict_physical_engine_targets_met"] is False
    assert report["production_use_allowed"] is report["floor_release_allowed"] is False
    assert report["conformance"] == "INDETERMINATE"
    assert report["uncertainty_status"] == "UNCALIBRATED"
    assert report["per_bend_measurement_accuracy"] == "NOT_EVALUATED_NO_MEASUREMENT_TRUTH"
    assert all(value == 0 for value in report["paired_deltas"].values())
    json.dumps(report, allow_nan=False)


def test_manual_mixed_totals_and_each_unequal_family_are_not_claimed_score_inputs(scoring):
    arguments = _cohort()
    for payload in arguments["cached_results_by_case"].values():
        payload["predictions"][-1] = _line("unmatched", 5000.)
        payload["claimed_metrics"] = dict(true_positive=999, false_negative=0, f1=1.)
    report = _score(scoring, arguments)
    metrics = report["cached_control"]["geometry"]
    # Each case independently loses exactly one match and gains one unmatched line.
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (60, 11, 11)
    assert metrics["precision"] == pytest.approx(60/71)
    assert metrics["recall"] == pytest.approx(60/71)
    assert metrics["f1"] == pytest.approx(60/71)
    assert metrics["exact_cached_line_count_cases"] == 11
    assert metrics["exact_localized_target_set_cases"] == 0
    for family, gold, cases in ((0, 10, 2), (1, 12, 2), (2, 14, 2), (3, 16, 2), (4, 12, 2), (5, 7, 1)):
        row = report["cached_control"]["families"][f"family:{family}"]
        assert (row["reference_line_count"], row["case_count"]) == (gold, cases)
        assert (row["true_positive"], row["false_positive"], row["false_negative"]) == (gold-cases, cases, cases)
        assert row["precision"] == row["recall"] == pytest.approx((gold-cases)/gold)
        assert row["f1"] == pytest.approx((gold-cases)/gold)
        assert report["family_paired_deltas"][f"family:{family}"]["f1"] == pytest.approx(cases/gold)
    assert report["cached_control"]["minimum_family_recall"] == .8
    delta = report["paired_deltas"]
    assert (delta["true_positive"], delta["false_positive"], delta["false_negative"]) == (11, -11, -11)
    assert delta["precision"] == delta["recall"] == pytest.approx(11/71)
    assert delta["f1"] == pytest.approx(11/71)


def test_singleton_precision_recall_and_f1_use_their_own_denominators(scoring):
    arguments = _cohort()
    arguments["cached_results_by_case"][CASE]["predictions"] = [
        _line("hit", 0.), _line("miss:1", 5000.), _line("miss:2", 5100.)]
    report = _score(scoring, arguments)
    row = report["cached_control"]["case_dispositions"][CASE]
    assert (row["true_positive"], row["false_positive"], row["false_negative"]) == (1, 2, 1)
    assert row["precision"] == pytest.approx(1/3)
    assert row["recall"] == .5 and row["f1"] == pytest.approx(2/5)
    total = report["cached_control"]["geometry"]
    assert (total["true_positive"], total["false_positive"], total["false_negative"]) == (70, 2, 1)
    assert total["precision"] == pytest.approx(70/72)
    assert total["recall"] == pytest.approx(70/71)
    assert total["f1"] == pytest.approx(140/143)
    family = report["cached_control"]["families"]["family:0"]
    assert (family["true_positive"], family["false_positive"], family["false_negative"]) == (9, 2, 1)
    assert family["f1"] == pytest.approx(18/21)


@pytest.mark.parametrize("fault", (
    "not_dict", "schema_bool", "kind", "route", "research", "cad_guided", "source_only",
    "status", "case_identity", "family_identity", "tuple_rows", "missing_rows",
))
def test_malformed_cached_payload_keeps_whole_reference_denominator(scoring, fault):
    arguments = _cohort()
    payload = arguments["cached_results_by_case"][CASE]
    if fault == "not_dict":
        arguments["cached_results_by_case"][CASE] = None
    elif fault == "schema_bool":
        payload["schema_version"] = True
    elif fault in ("kind", "route"):
        payload[fault] = "another-schema"
    elif fault == "research":
        payload["research_only"] = False
    elif fault == "cad_guided":
        payload["cad_guided_zero_shot"] = False
    elif fault == "source_only":
        payload["source_only"] = True
    elif fault == "status":
        payload["status"] = "PARTIAL"
    elif fault == "case_identity":
        payload["case_id"] = "another-case"
    elif fault == "family_identity":
        payload["family_id"] = "another-family"
    elif fault == "tuple_rows":
        payload["predictions"] = tuple(payload["predictions"])
    else:
        del payload["predictions"]
    report = _score(scoring, arguments)
    _assert_zero_case(report)
    assert report["cached_control"]["malformed_case_count"] == 1
    assert report["engine"]["geometry"]["true_positive"] == 71


@pytest.mark.parametrize("fault", (
    "nan", "infinity", "boolean", "string", "coordinate_cap", "degenerate", "duplicate_id",
    "missing_endpoint", "unit", "frame", "carrier",
))
def test_bad_cached_suffix_invalidates_good_prefix_without_sanitizing(scoring, fault):
    arguments = _cohort()
    rows = arguments["cached_results_by_case"][CASE]["predictions"]
    row = rows[-1]
    if fault in ("nan", "infinity", "boolean", "string", "coordinate_cap"):
        row["line_end"][0] = {"nan": math.nan, "infinity": math.inf, "boolean": True,
                              "string": "100", "coordinate_cap": 1000001.}[fault]
    elif fault == "degenerate":
        row["line_end"] = list(row["line_start"])
    elif fault == "duplicate_id":
        row["bend_id"] = rows[0]["bend_id"]
    elif fault == "missing_endpoint":
        del row["line_end"]
    elif fault == "unit":
        row["unit"] = "inch"
    elif fault == "frame":
        row["reference_frame"] = "CAD"
    else:
        row["line_convention"] = CARRIER
    report = _score(scoring, arguments)
    _assert_zero_case(report)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("alias", ALIASES)
def test_every_alternate_cached_endpoint_alias_rejects_even_if_empty_or_agreeing(scoring, alias):
    arguments = _cohort()
    row = arguments["cached_results_by_case"][CASE]["predictions"][-1]
    row[alias] = deepcopy(row["line_start"]) if alias in ("start", "p0", "lineStart") else None
    result = _score(scoring, arguments)
    assert _assert_zero_case(result)["reason"] == "ALTERNATE_LINE_REPRESENTATION_NOT_ADMITTED"


@pytest.mark.parametrize("count", (10, 12))
def test_case_cardinality_is_exactly_eleven(scoring, count):
    arguments = _cohort()
    if count == 10:
        del arguments["case_family_by_id"][CASE]
    else:
        arguments["case_family_by_id"]["extra-case"] = "family:0"
    with pytest.raises(ValueError, match="^REQUIRE_COMPLETE_ELEVEN_CASE_FAMILY_MAP$"):
        _score(scoring, arguments)


@pytest.mark.parametrize("count", (5, 7))
def test_family_cardinality_is_exactly_six(scoring, count):
    arguments = _cohort()
    arguments["case_family_by_id"]["invented:05" if count == 5 else "invented:10"] = (
        "family:4" if count == 5 else "family:6")
    with pytest.raises(ValueError, match="^REQUIRE_SIX_DECLARED_FAMILIES$"):
        _score(scoring, arguments)


def test_case_identifier_cannot_be_an_empty_declared_key(scoring):
    arguments = _cohort()
    for mapping in arguments.values():
        mapping[""] = mapping.pop(CASE)
    with pytest.raises(ValueError, match="^INVALID_CASE_OR_FAMILY_ID$"):
        _score(scoring, arguments)


@pytest.mark.parametrize("name", (
    "reference_records_by_case", "cached_results_by_case", "engine_results_by_case",
))
def test_missing_case_in_either_lane_or_references_blocks_whole_scoring(scoring, name):
    arguments = _cohort()
    del arguments[name][CASE]
    with pytest.raises(ValueError, match="^CASE_SET_MISMATCH$"):
        _score(scoring, arguments)


@pytest.mark.parametrize("fault", ("70_lines", "72_lines", "empty_case", "duplicate_id", "nan", "alias"))
def test_bad_reference_is_not_reclassified_as_a_cached_or_engine_failure(scoring, fault):
    arguments = _cohort()
    rows = arguments["reference_records_by_case"][CASE]
    if fault == "70_lines":
        rows.pop()
    elif fault == "72_lines":
        rows.append(_line("extra", 6000.))
    elif fault == "empty_case":
        rows.clear()
    elif fault == "duplicate_id":
        rows[1]["bend_id"] = rows[0]["bend_id"]
    elif fault == "nan":
        rows[-1]["line_end"][0] = math.nan
    else:
        rows[-1]["span_endpoints"] = None
    with pytest.raises(ValueError):
        _score(scoring, arguments)


@pytest.mark.parametrize("kind,value,match", (
    ("offset", 4., True), ("offset", 4.01, False),
    ("angle", 4.9, True), ("angle", 5.1, False), ("angle", 179., True),
    ("span", (40., 140.), True), ("span", (41., 141.), False),
    ("span", (0., 59.), False), ("span", (-100., 100.), False),
    ("reverse", None, True),
))
def test_cached_geometry_uses_unchanged_finite_matching_criteria(scoring, kind, value, match):
    arguments = _cohort()
    row = arguments["cached_results_by_case"][CASE]["predictions"][0]
    if kind == "offset":
        row["line_start"][2] = row["line_end"][2] = value
    elif kind == "angle":
        cosine, sine = math.cos(math.radians(value)), math.sin(math.radians(value))
        row["line_start"] = [50.-50.*cosine, -50.*sine, 0.]
        row["line_end"] = [50.+50.*cosine, 50.*sine, 0.]
    elif kind == "span":
        row["line_start"][0], row["line_end"][0] = value
    else:
        row["line_start"], row["line_end"] = row["line_end"], row["line_start"]
    report = _score(scoring, arguments)
    result = report["cached_control"]["case_dispositions"][CASE]
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (
        (2, 0, 0) if match else (1, 1, 1))
    assert any(pair["gold_id"] == "line:0" and pair["prediction_id"] == "line:0"
               for pair in result["matches"]) is match


def test_duplicate_geometry_with_distinct_ids_is_not_fused_or_matched_twice(scoring):
    arguments = _cohort()
    rows = arguments["cached_results_by_case"][CASE]["predictions"]
    rows.append({**deepcopy(rows[0]), "bend_id": "second-copy"})
    result = _score(scoring, arguments)["cached_control"]["case_dispositions"][CASE]
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (2, 1, 0)
    assert len(result["unmatched_prediction_ids"]) == 1
    assert len({row["gold_id"] for row in result["matches"]}) == 2


def test_assignment_is_global_one_to_one_not_first_admissible_greedy(scoring):
    arguments = _cohort()
    references = [_line("gold:0", 0.), _line("gold:1", 6.)]
    # shared is admissible to both references, exclusive only to gold:0.
    predictions = [_line("shared", 3.), _line("exclusive", 0.)]
    arguments["reference_records_by_case"][CASE] = references
    arguments["cached_results_by_case"][CASE] = _cached(predictions)
    arguments["engine_results_by_case"][CASE] = _engine(references)
    result = _score(scoring, arguments)["cached_control"]["case_dispositions"][CASE]
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (2, 0, 0)
    assert {(row["gold_id"], row["prediction_id"]) for row in result["matches"]} == {
        ("gold:0", "exclusive"), ("gold:1", "shared")}


def test_valid_empty_cached_lists_are_unknown_observation_with_null_precision_f1(scoring):
    arguments = _cohort()
    for payload in arguments["cached_results_by_case"].values():
        payload["predictions"] = []
    report = _score(scoring, arguments)
    cached = report["cached_control"]
    assert cached["malformed_case_count"] == 0
    assert cached["geometry"]["cached_prediction_line_count"] == 0
    assert cached["geometry"]["false_negative"] == 71
    assert cached["geometry"]["precision"] is None and cached["geometry"]["f1"] is None
    assert cached["geometry"]["recall"] == cached["minimum_family_recall"] == 0.
    assert report["paired_deltas"]["precision"] is report["paired_deltas"]["f1"] is None
    assert report["paired_deltas"]["recall"] == 1.
    assert all(row["disposition"] == "SCORED_CACHED_GEOMETRY" and row["observed_bend_count"] is None
               for row in cached["case_dispositions"].values())


def test_zero_matches_with_equal_cached_counts_does_not_manufacture_zero_f1(scoring):
    arguments = _cohort()
    for payload in arguments["cached_results_by_case"].values():
        for row in payload["predictions"]:
            row["line_start"][2] = row["line_end"][2] = 1000.
    report = _score(scoring, arguments)
    result = report["cached_control"]["geometry"]
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (0, 71, 71)
    assert result["precision"] == result["recall"] == 0. and result["f1"] is None
    assert result["exact_cached_line_count_accuracy"] == 1.
    assert result["exact_localized_target_set_accuracy"] == 0.
    assert report["paired_deltas"]["f1"] is None


@pytest.mark.parametrize("fault", ("explicit_refusal", "nonempty_refusal", "count_claim", "bad_suffix"))
def test_engine_lane_retains_frozen_refusal_and_malformed_semantics(scoring, fault):
    arguments = _cohort()
    payload = arguments["engine_results_by_case"][CASE]
    if fault == "explicit_refusal":
        arguments["engine_results_by_case"][CASE] = dict(status="UNAVAILABLE", predictions=[])
    elif fault == "nonempty_refusal":
        payload["status"] = "UNAVAILABLE"
    elif fault == "count_claim":
        payload["observed_bend_count"] = 2
    else:
        payload["predictions"][-1]["line_end"][0] = math.nan
    report = _score(scoring, arguments)
    _assert_zero_case(report, lane="engine", disposition="REFUSED" if fault == "explicit_refusal" else "MALFORMED_RESULT")
    assert report["cached_control"]["geometry"]["true_positive"] == 71
    assert report["engine"]["physical_observed_bend_count"]["unknown_case_count"] == 11


def test_ignored_legacy_metadata_neither_filters_geometry_nor_grants_measurement(scoring):
    arguments = _cohort()
    payload = arguments["cached_results_by_case"][CASE]
    payload.update(metrics={"f1": 999}, production_use_allowed=True, observed_bend_count=500)
    for row in payload["predictions"]:
        row.update(measurement_method="nominal CAD", observability_state="MISSING",
                   confidence=math.nan, conformance="PASS")
    report = _score(scoring, arguments)
    assert report["cached_control"]["geometry"]["true_positive"] == 71
    assert report["observed_bend_count"] is None and report["production_use_allowed"] is False
    result = report["cached_control"]["case_dispositions"][CASE]
    assert result["observed_bend_count"] is None and result["disposition"] == "SCORED_CACHED_GEOMETRY"
    assert "measurement_method" not in json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("count,accepted", ((4096, True), (4097, False)))
def test_cached_prediction_cap_is_exact_and_precedes_any_row_normalization(scoring, monkeypatch, count, accepted):
    rows = [_line(f"row:{index}", 0.) for index in range(count)]
    payload = _cached(rows)
    if not accepted:
        def forbidden(value):
            raise AssertionError("Over-cap rows must be refused before point normalization.")
        monkeypatch.setattr(scoring, "_point", forbidden)
    normalized, disposition, reason = scoring._cached_predictions(payload, CASE, "family:0")
    if accepted:
        assert disposition == "SCORED_CACHED_GEOMETRY" and reason is None
        assert len(normalized) == 4096 and normalized[-1]["bend_id"] == "row:4095"
        assert normalized == rows
    else:
        assert normalized == [] and disposition == "MALFORMED_RESULT"
        assert reason == "INVALID_LINE_CONTAINER_OR_BUDGET"


def test_matching_and_engine_lane_are_exact_frozen_callable_objects(scoring):
    matcher = importlib.import_module(scoring.__package__ + ".evaluate_f1_against_manual_gold")
    frozen = importlib.import_module(scoring.__package__ + ".gold11_mesh_scoring_v1")
    assert scoring.match_lines is frozen.match_lines is matcher.match_lines
    assert scoring.Thresholds is frozen.Thresholds is matcher.Thresholds
    assert scoring._lane is frozen._lane
    report = _score(scoring, _cohort())
    assert report["matching"] == dict(axis_angle_deg_max=5., transverse_offset_mm_max=4.,
        gold_span_overlap_min=.6, prediction_span_overlap_min=.6,
        method="UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST",
        per_case_exact_flag="EXACT_LOCALIZED_REFERENCE_SET_NOT_PHYSICAL_COUNT")


def test_unexpected_matcher_failure_is_not_converted_to_a_zero_prediction_control(scoring, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("invented-matcher-failure")
    monkeypatch.setattr(scoring, "match_lines", failed)
    with pytest.raises(RuntimeError, match="^invented-matcher-failure$"):
        _score(scoring, _cohort())
