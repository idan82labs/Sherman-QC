"""Invented-only post-inference scorer contracts; no data/native/inference I/O."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import sys
import types

import pytest


_PACKAGE = "_gold11_mesh_scoring_v1_tests"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [str(Path(__file__).resolve().parents[1] / "scripts")]
sys.modules[_PACKAGE] = _package
scoring = importlib.import_module(_PACKAGE + ".gold11_mesh_scoring_v1")


def fixture():
    families, references, results = {}, {}, {}
    for case_index in range(11):
        case = f"invented-case:{case_index:02d}"
        families[case] = f"invented-family:{case_index % 6}"
        count = 11 if case_index == 10 else 6
        references[case] = [dict(bend_id=f"line:{index}", line_start=[0., 20. * index, 0.],
                                 line_end=[100., 20. * index, 0.]) for index in range(count)]
        results[case] = {
            "status": "PARTIAL", "phase": "COMPLETE", "observed_bend_count": None,
            "production_use_allowed": False, "conformance": "INDETERMINATE",
            "uncertainty_status": "UNCALIBRATED", "prediction_frame": scoring.RAW_FRAME,
            "units_status": scoring.PROVISIONAL_UNITS,
            "reference_frame_compatibility": "NOT_ASSERTED_BY_INFERENCE",
            "predictions": [{**deepcopy(row), "unit": "mm", "reference_frame": scoring.RAW_FRAME,
                             "line_convention": scoring.CARRIER} for row in references[case]],
        }
    return dict(reference_records_by_case=references, control_results_by_case=deepcopy(results),
                challenger_results_by_case=results, case_family_by_id=families)


def first_case(arguments):
    return sorted(arguments["case_family_by_id"])[0]


def score(arguments):
    return scoring.score_gold11_mesh_workflows_v1(**arguments)


def assert_case_zeroed(report, case):
    row = report["challenger"]["case_dispositions"][case]
    assert row["prediction_count"] == 0
    assert row["true_positive"] == row["false_positive"] == 0
    assert row["false_negative"] == row["gold_count"] == 6
    assert row["precision"] is None
    assert row["recall"] == 0.
    assert row["f1"] is None
    assert row["exact_emitted_support_count"] is False
    assert row["observed_bend_count"] is None
    assert report["challenger"]["geometry"]["case_count"] == 11
    assert report["challenger"]["geometry"]["reference_line_count"] == 71
    return row


def test_perfect_geometry_remains_not_physical_count_or_measurement_qualification():
    arguments = fixture()
    before = deepcopy(arguments)
    report = score(arguments)
    assert arguments == before
    metrics = report["challenger"]["geometry"]
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (71, 0, 0)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.
    assert metrics["exact_emitted_support_count_accuracy"] == 1.
    assert metrics["count_metric_semantics"] == "EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS"
    assert report["challenger"]["physical_observed_bend_count"]["unknown_rate"] == 1.
    assert report["challenger"]["physical_observed_bend_count"]["strict_count_target_met"] is False
    assert report["strict_physical_engine_targets_met"] is False
    assert report["production_use_allowed"] is report["floor_release_allowed"] is False
    assert report["observed_bend_count"] is None
    assert report["per_bend_measurement_accuracy"] == "NOT_EVALUATED_NO_MEASUREMENT_TRUTH"
    assert report["historical_score_comparability"] == "UNESTABLISHED"
    assert report["paired_deltas"]["f1"] == 0.
    for lane in ("control", "challenger"):
        for row in report[lane]["case_dispositions"].values():
            assert row["precision"] == row["recall"] == row["f1"] == 1.
            assert row["exact_emitted_support_count"] is True
            assert row["exact"] is True
            assert row["observed_bend_count"] is None
            assert row["count_metric_semantics"] == "EMITTED_SUPPORT_LINES_NOT_PHYSICAL_OBSERVED_BENDS"
    json.dumps(report, allow_nan=False)


def test_explicit_refusal_retains_all_false_negatives_and_six_families():
    arguments = fixture()
    case = first_case(arguments)
    arguments["challenger_results_by_case"][case] = {
        "status": "UNAVAILABLE", "phase": "NATIVE_IMPORT", "predictions": [],
        "reasons": ["invented import refusal"], "observed_bend_count": None,
    }
    report = score(arguments)
    row = assert_case_zeroed(report, case)
    assert row["disposition"] == "REFUSED"
    assert row["reported_workflow_phase"] == "NATIVE_IMPORT"
    assert row["reported_workflow_reasons"] == ["invented import refusal"]
    assert report["challenger"]["refused_case_count"] == 1
    assert len(report["challenger"]["families"]) == 6
    assert report["challenger"]["geometry"]["true_positive"] == 65
    assert report["challenger"]["geometry"]["recall"] == pytest.approx(65 / 71)
    assert report["paired_deltas"]["false_negative"] == 6


def test_undefined_legacy_precision_and_f1_stay_none_and_deltas_stay_none():
    arguments = fixture()
    for case in arguments["case_family_by_id"]:
        arguments["control_results_by_case"][case] = {"status": "UNAVAILABLE", "predictions": []}
    report = score(arguments)
    control = report["control"]["geometry"]
    assert control["precision"] is None
    assert control["recall"] == 0.
    assert control["f1"] is None
    assert report["paired_deltas"]["precision"] is None
    assert report["paired_deltas"]["f1"] is None
    assert report["paired_deltas"]["recall"] == 1.
    assert control["exact_emitted_support_count_cases"] == 0
    assert report["control"]["minimum_family_recall"] == 0.


def test_zero_true_positives_preserves_legacy_undefined_f1_even_with_predictions():
    arguments = fixture()
    for payload in arguments["challenger_results_by_case"].values():
        for row in payload["predictions"]:
            row["line_start"][2] = row["line_end"][2] = 1000.
    report = score(arguments)
    metrics = report["challenger"]["geometry"]
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (0, 71, 71)
    assert metrics["precision"] == metrics["recall"] == 0.
    assert metrics["f1"] is None
    # Count equality is not geometric success or physical observed bend count.
    assert metrics["exact_emitted_support_count_accuracy"] == 1.
    assert metrics["exact_localized_target_set_accuracy"] == 0.
    assert report["strict_physical_engine_targets_met"] is False
    for row in report["challenger"]["case_dispositions"].values():
        assert row["precision"] == row["recall"] == 0.
        assert row["f1"] is None
        assert row["exact_emitted_support_count"] is True
        assert row["exact"] is False
        assert row["observed_bend_count"] is None


def test_per_case_partial_localization_metrics_use_singleton_not_corpus_ratios():
    arguments = fixture()
    case = first_case(arguments)
    payload = arguments["challenger_results_by_case"][case]
    payload["predictions"] = payload["predictions"][:2]
    report = score(arguments)
    row = report["challenger"]["case_dispositions"][case]
    assert (row["true_positive"], row["false_positive"], row["false_negative"]) == (2, 0, 4)
    assert row["precision"] == 1.
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["f1"] == pytest.approx(.5)
    assert row["recall"] != report["challenger"]["geometry"]["recall"]
    assert row["exact_emitted_support_count"] is False
    assert row["exact"] is False
    assert row["observed_bend_count"] is None


def test_per_case_count_equality_does_not_replace_partial_localization_score():
    arguments = fixture()
    case = first_case(arguments)
    rows = arguments["challenger_results_by_case"][case]["predictions"]
    for row in rows[2:]:
        row["line_start"][2] = row["line_end"][2] = 1000.
    report = score(arguments)
    row = report["challenger"]["case_dispositions"][case]
    assert (row["true_positive"], row["false_positive"], row["false_negative"]) == (2, 4, 4)
    assert row["precision"] == row["recall"] == pytest.approx(1 / 3)
    assert row["f1"] == pytest.approx(1 / 3)
    assert row["exact_emitted_support_count"] is True
    assert row["exact"] is False
    assert row["observed_bend_count"] is None


@pytest.mark.parametrize("mutation", [
    "missing_end", "nan", "bool_coordinate", "zero_length", "duplicate_id", "alternate_span",
    "wrong_unit", "wrong_frame", "wrong_carrier", "non_dict", "too_many",
])
def test_any_malformed_prediction_zeroes_entire_case_not_only_bad_suffix(mutation):
    arguments = fixture()
    case = first_case(arguments)
    rows = arguments["challenger_results_by_case"][case]["predictions"]
    if mutation == "missing_end":
        del rows[-1]["line_end"]
    elif mutation == "nan":
        rows[-1]["line_end"][0] = float("nan")
    elif mutation == "bool_coordinate":
        rows[-1]["line_start"][0] = False
    elif mutation == "zero_length":
        rows[-1]["line_end"] = rows[-1]["line_start"][:]
    elif mutation == "duplicate_id":
        rows[-1]["bend_id"] = rows[0]["bend_id"]
    elif mutation == "alternate_span":
        rows[-1]["span_endpoints"] = [rows[-1]["line_start"], rows[-1]["line_end"]]
    elif mutation == "wrong_unit":
        rows[-1]["unit"] = "inch"
    elif mutation == "wrong_frame":
        rows[-1]["reference_frame"] = "CAD_REFERENCE"
    elif mutation == "wrong_carrier":
        rows[-1]["line_convention"] = "CYLINDER_CENTER_AXIS_SUPPORT"
    elif mutation == "non_dict":
        rows[-1] = None
    else:
        rows.extend(deepcopy(rows[0]) for _ in range(4097))
    report = score(arguments)
    assert assert_case_zeroed(report, case)["disposition"] == "MALFORMED_RESULT"
    assert report["challenger"]["malformed_case_count"] == 1


@pytest.mark.parametrize("field,value", [
    ("status", "PASS"), ("phase", "NATIVE_IMPORT"), ("predictions", None),
    ("observed_bend_count", 6), ("production_use_allowed", True),
    ("units_status", "VERIFIED_MM"), ("prediction_frame", "CAD_REFERENCE"),
    ("case_id", "other-case"), ("family_id", "other-family"),
])
def test_malformed_workflow_or_unsupported_count_claim_cannot_score_a_valid_prefix(field, value):
    arguments = fixture()
    case = first_case(arguments)
    arguments["challenger_results_by_case"][case][field] = value
    report = score(arguments)
    assert assert_case_zeroed(report, case)["disposition"] == "MALFORMED_RESULT"


def test_unavailable_payload_with_predictions_is_not_rescued():
    arguments = fixture()
    case = first_case(arguments)
    arguments["challenger_results_by_case"][case]["status"] = "UNAVAILABLE"
    report = score(arguments)
    assert assert_case_zeroed(report, case)["disposition"] == "MALFORMED_RESULT"


@pytest.mark.parametrize("mapping", ["reference_records_by_case", "control_results_by_case", "challenger_results_by_case"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_case_map_mismatch_blocks_whole_scoring_call(mapping, change):
    arguments = fixture()
    if change == "missing":
        arguments[mapping].pop(first_case(arguments))
    else:
        arguments[mapping]["extra-case"] = []
    with pytest.raises(ValueError, match="CASE_SET_MISMATCH"):
        score(arguments)


@pytest.mark.parametrize("mutation", ["missing_line", "extra_line", "duplicate_id", "zero_case", "bad_point", "wrong_unit", "wrong_frame"])
def test_invalid_reference_geometry_or_count_is_not_converted_to_engine_failure(mutation):
    arguments = fixture()
    rows = arguments["reference_records_by_case"][first_case(arguments)]
    if mutation == "missing_line":
        rows.pop()
    elif mutation == "extra_line":
        rows.append({**deepcopy(rows[0]), "bend_id": "extra"})
    elif mutation == "duplicate_id":
        rows[-1]["bend_id"] = rows[0]["bend_id"]
    elif mutation == "zero_case":
        rows.clear()
    elif mutation == "bad_point":
        rows[-1]["line_end"] = [1., 2.]
    elif mutation == "wrong_unit":
        rows[-1]["unit"] = "inch"
    else:
        rows[-1]["reference_frame"] = "CAD_REFERENCE"
    with pytest.raises(ValueError):
        score(arguments)


def test_six_family_contract_cannot_silently_drop_or_relabel_weak_family():
    arguments = fixture()
    for case in arguments["case_family_by_id"]:
        arguments["case_family_by_id"][case] = "one-family"
    with pytest.raises(ValueError, match="REQUIRE_SIX_DECLARED_FAMILIES"):
        score(arguments)


def test_minimum_family_recall_uses_full_reference_denominator():
    arguments = fixture()
    for case, family in arguments["case_family_by_id"].items():
        if family == "invented-family:5":
            arguments["challenger_results_by_case"][case] = {"status": "UNAVAILABLE", "predictions": []}
    report = score(arguments)
    assert report["challenger"]["minimum_family_recall"] == 0.
    assert report["challenger"]["families"]["invented-family:5"]["reference_line_count"] == 6
    assert report["family_paired_deltas"]["invented-family:5"]["recall"] == -1.


@pytest.mark.parametrize("offset,expected_tp", [(4., 71), (4.001, 65)])
def test_frozen_transverse_offset_rule(offset, expected_tp):
    arguments = fixture()
    for row in arguments["challenger_results_by_case"][first_case(arguments)]["predictions"]:
        row["line_start"][2] += offset
        row["line_end"][2] += offset
    assert score(arguments)["challenger"]["geometry"]["true_positive"] == expected_tp


def test_both_span_overlap_is_required_not_shorter_span_fusion_rule():
    arguments = fixture()
    for row in arguments["challenger_results_by_case"][first_case(arguments)]["predictions"]:
        row["line_end"][0] = 200.  # Gold fully covered, prediction overlap only .50.
    report = score(arguments)
    assert report["challenger"]["geometry"]["true_positive"] == 65
    assert report["challenger"]["geometry"]["false_positive"] == 6
    assert report["challenger"]["geometry"]["false_negative"] == 6
    assert report["matching"]["prediction_span_overlap_min"] == .60


@pytest.mark.parametrize("angle,expected_tp", [(4., 71), (6., 65)])
def test_frozen_angle_rule_uses_undirected_five_degrees(angle, expected_tp):
    arguments = fixture()
    dx, dz = 50. * math.cos(math.radians(angle)), 50. * math.sin(math.radians(angle))
    for row in arguments["challenger_results_by_case"][first_case(arguments)]["predictions"]:
        y = row["line_start"][1]
        row["line_start"] = [50. + dx, y, dz]
        row["line_end"] = [50. - dx, y, -dz]  # Reversed direction must not imply180 degrees.
    assert score(arguments)["challenger"]["geometry"]["true_positive"] == expected_tp


def test_matching_is_one_to_one_and_aggregate_booleans_are_ignored():
    arguments = fixture()
    case = first_case(arguments)
    payload = arguments["challenger_results_by_case"][case]
    duplicate_geometry = {**deepcopy(payload["predictions"][0]), "bend_id": "distinct-id-same-line"}
    payload["predictions"].append(duplicate_geometry)
    payload.update(f1=1., exact_count_accuracy=1., targets_met=True, observed_support_instance_count=6)
    report = score(arguments)
    metrics = report["challenger"]["geometry"]
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (71, 1, 0)
    assert metrics["precision"] == pytest.approx(71 / 72)
    assert metrics["exact_emitted_support_count_accuracy"] == pytest.approx(10 / 11)
    assert report["strict_physical_engine_targets_met"] is False


def test_paired_positive_delta_is_recomputed_from_line_records_not_score_claims():
    arguments = fixture()
    case = first_case(arguments)
    control = arguments["control_results_by_case"][case]
    control["predictions"].append({**deepcopy(control["predictions"][0]), "bend_id": "extra-control-line"})
    control["f1"] = 1.  # Deliberately false aggregate assertion is unused.
    report = score(arguments)
    assert report["control"]["geometry"]["f1"] == pytest.approx(142 / 143)
    assert report["challenger"]["geometry"]["f1"] == 1.
    assert report["paired_deltas"]["f1"] == pytest.approx(1 / 143)
    assert report["paired_deltas"]["precision"] == pytest.approx(1 / 72)
    assert report["paired_deltas"]["recall"] == 0.
    assert report["paired_deltas"]["exact_emitted_support_count_accuracy"] == pytest.approx(1 / 11)
    assert report["historical_score_comparability"] == "UNESTABLISHED"
