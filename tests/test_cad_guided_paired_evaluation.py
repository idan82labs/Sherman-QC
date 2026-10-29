"""Invented paired outputs; no corpus/imported application or inference execution."""

import builtins
import copy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    scorer_path = ROOT / "backend/cad_guided_evaluation.py"
    scorer = ModuleType("paired_test_cad_guided_evaluation")
    scorer.__file__ = str(scorer_path)
    previous = sys.modules.get(scorer.__name__)
    sys.modules[scorer.__name__] = scorer
    try:
        exec(compile(scorer_path.read_text(), str(scorer_path), "exec"), scorer.__dict__)
    finally:
        if previous is None:
            del sys.modules[scorer.__name__]
        else:
            sys.modules[scorer.__name__] = previous
    paired_path = ROOT / "backend/cad_guided_paired_evaluation.py"
    module = ModuleType("paired_evaluation_isolated_test")
    module.__file__ = str(paired_path)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cad_guided_evaluation":
            return scorer
        raise AssertionError(f"unexpected paired evaluator import: {name}")

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": guarded_import}
    exec(compile(paired_path.read_text(), str(paired_path), "exec"), module.__dict__)
    return module


def _line(identity="bend", offset=0.):
    return {"bend_id": identity, "line_start": [offset, 0., 0.], "line_end": [offset, 0., 20.]}


def _payload(status="COMPLETE", offset=0.):
    instances = [] if status == "UNAVAILABLE" else [{
        **_line(offset=offset), "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT", "unit": "mm",
        "observation_status": "SAMPLE_SUPPORTED_CONDITIONAL",
        "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM", "physical_trim_endpoints": None}]
    return {"finite_support_instances": instances, "reference_frame": "CAD_REFERENCE", "unit": "mm",
            "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT", "instance_extraction_status": status}


def _inputs(count=1):
    ids = [f"case-{index}" for index in range(count)]
    return {
        "expected_cases": [{"case_id": case_id, "part_id": f"part-{index}", "family_id": "family-a"}
                           for index, case_id in enumerate(ids)],
        "declared_families": ["family-a"], "reference_records_by_case": {case_id: [_line()] for case_id in ids},
        "incumbent_payloads_by_case": {case_id: _payload() for case_id in ids},
        "challenger_payloads_by_case": {case_id: _payload() for case_id in ids},
        "reference_frame": "CAD_REFERENCE", "reference_unit": "mm",
        "reference_convention": "CYLINDER_CENTER_AXIS_SUPPORT",
    }


def test_actual_two_lane_scoring_ignores_detached_aggregate_and_expected_counts():
    module, inputs = _module(), _inputs()
    inputs["incumbent_payloads_by_case"]["case-0"] = _payload(offset=10.)
    inputs["incumbent_payloads_by_case"]["case-0"].update(
        observed_support_instance_count=900, expected_feature_count=800, true_positive=700, exact_count=True,
        production_use_allowed=True, promotion_status="AUTHORIZED")
    before = copy.deepcopy(inputs)
    result = module.score_paired_support_cases(**inputs)
    assert inputs == before
    assert result["incumbent"]["aggregate"]["true_positive"] == 0
    assert result["challenger"]["aggregate"]["true_positive"] == 1
    assert result["paired_metric_deltas"]["true_positive"] == 1
    assert result["paired_metric_deltas"]["false_positive"] == -1
    assert result["paired_metric_deltas"]["exact_count_accuracy"] == 0.
    assert result["paired_metric_deltas"]["exact_localized_target_set_accuracy"] == 1.
    assert result["paired_metric_deltas"]["f1"] is None  # retained undefined-zero scorer convention
    assert result["physical_bend_count"] is None
    assert result["evidence_verification"] == "NOT_PERFORMED"
    assert result["promotion_status"] == "NOT_AUTHORIZED"
    assert result["production_use_allowed"] is result["physical_access_authorized"] is False
    assert result["support_diagnostic_targets"]["challenger"]["diagnostic_targets_met"] is True
    assert result["support_diagnostic_targets"]["challenger"]["physical_target_qualification"] == "NOT_ESTABLISHED"
    json.dumps(result, allow_nan=False)


def test_failed_and_partial_cases_stay_in_same_paired_denominators():
    inputs = _inputs(3)
    inputs["challenger_payloads_by_case"]["case-1"] = _payload("UNAVAILABLE")
    inputs["challenger_payloads_by_case"]["case-2"] = _payload("PARTIAL")
    result = _module().score_paired_support_cases(**inputs)
    summary = result["challenger"]["aggregate"]
    assert summary["case_count"] == 3
    assert summary["true_positive"] == 2 and summary["false_negative"] == 1
    assert summary["exact_count_cases"] == summary["complete_case_count"] == 1
    assert summary["f1"] == pytest.approx(.8)
    assert result["per_family"][0]["challenger"] == summary
    assert result["support_diagnostic_targets"]["challenger"]["diagnostic_targets_met"] is False


@pytest.mark.parametrize("mapping_name", ["reference_records_by_case", "incumbent_payloads_by_case",
                                          "challenger_payloads_by_case"])
@pytest.mark.parametrize("change", ["missing", "extra", "renamed", "not_mapping"])
def test_exact_case_set_required_for_both_lanes_and_truth(mapping_name, change):
    inputs = _inputs()
    mapping = inputs[mapping_name]
    if change == "missing":
        del mapping["case-0"]
    elif change == "extra":
        mapping["extra"] = copy.deepcopy(mapping["case-0"])
    elif change == "renamed":
        mapping["different"] = mapping.pop("case-0")
    else:
        inputs[mapping_name] = list(mapping.values())
    with pytest.raises(ValueError):
        _module().score_paired_support_cases(**inputs)


@pytest.mark.parametrize("change", ["duplicate_case", "empty_cases", "oversized_cases", "missing_field",
                                    "extra_field", "blank_id", "long_id", "bool_id", "part_two_families",
                                    "undeclared_family", "duplicate_family", "empty_families", "blank_family"])
def test_inventory_is_explicit_bounded_and_part_family_consistent(change):
    inputs = _inputs(2)
    if change == "duplicate_case":
        inputs["expected_cases"][1]["case_id"] = "case-0"
    elif change == "empty_cases":
        inputs["expected_cases"] = []
    elif change == "oversized_cases":
        inputs["expected_cases"] = [inputs["expected_cases"][0]] * 4097
    elif change == "missing_field":
        del inputs["expected_cases"][0]["part_id"]
    elif change == "extra_field":
        inputs["expected_cases"][0]["qualified"] = True
    elif change in {"blank_id", "long_id", "bool_id"}:
        inputs["expected_cases"][0]["case_id"] = {"blank_id": " ", "long_id": "x" * 257, "bool_id": True}[change]
    elif change == "part_two_families":
        inputs["declared_families"].append("family-b")
        inputs["expected_cases"][1].update(part_id="part-0", family_id="family-b")
    elif change == "undeclared_family":
        inputs["expected_cases"][0]["family_id"] = "family-b"
    elif change == "duplicate_family":
        inputs["declared_families"].append("family-a")
    elif change == "empty_families":
        inputs["declared_families"] = []
    else:
        inputs["declared_families"] = [" "]
    with pytest.raises(ValueError):
        _module().score_paired_support_cases(**inputs)


def test_multiple_scans_of_same_part_are_not_counted_as_multiple_parts():
    inputs = _inputs(2)
    inputs["expected_cases"][1]["part_id"] = "part-0"
    result = _module().score_paired_support_cases(**inputs)
    assert result["case_count"] == 2 and result["part_count"] == 1
    assert result["per_family"][0]["part_count"] == 1


def test_absent_declared_family_remains_no_evidence_and_blocks_diagnostic_targets():
    inputs = _inputs()
    inputs["declared_families"].append("family-with-no-cases")
    result = _module().score_paired_support_cases(**inputs)
    family = result["per_family"][1]
    assert result["absent_declared_families"] == ["family-with-no-cases"]
    assert family["evidence_status"] == "NO_EVIDENCE" and family["case_count"] == 0
    assert family["incumbent"] is family["challenger"] is None
    assert all(value is None for value in family["paired_metric_deltas"].values())
    assert result["support_diagnostic_targets"]["challenger"]["minimum_family_recall"] is None
    assert result["support_diagnostic_targets"]["challenger"]["diagnostic_targets_met"] is False


def test_worst_family_failure_is_not_hidden_by_pooled_success():
    inputs = _inputs(100)
    inputs["declared_families"].append("family-b")
    inputs["expected_cases"][-1]["family_id"] = "family-b"
    inputs["challenger_payloads_by_case"]["case-99"] = _payload("UNAVAILABLE")
    result = _module().score_paired_support_cases(**inputs)
    audit = result["support_diagnostic_targets"]["challenger"]
    assert all(check["meets"] is True for check in audit["checks"].values())
    assert audit["minimum_family_recall"] == 0.
    assert audit["family_recall_checks"]["family-b"]["meets"] is False
    assert audit["diagnostic_targets_met"] is False
    assert result["per_family"][1]["challenger"]["false_negative"] == 1


@pytest.mark.parametrize("successes,expected", [(94, False), (95, True)])
def test_exact_count_diagnostic_threshold_uses_exact_integer_arithmetic(successes, expected):
    inputs = _inputs(100)
    for index in range(successes, 100):
        inputs["challenger_payloads_by_case"][f"case-{index}"] = _payload("UNAVAILABLE")
    result = _module().score_paired_support_cases(**inputs)
    audit = result["support_diagnostic_targets"]["challenger"]
    assert audit["checks"]["sample_support_exact_count"]["meets"] is expected
    assert audit["diagnostic_targets_met"] is expected
    assert result["production_use_allowed"] is False


@pytest.mark.parametrize("status", ["UNAVAILABLE", "PARTIAL", "COMPLETE"])
def test_empty_references_do_not_create_physical_or_undefined_ratio_qualification(status):
    inputs = _inputs()
    inputs["reference_records_by_case"]["case-0"] = []
    for lane in ("incumbent_payloads_by_case", "challenger_payloads_by_case"):
        inputs[lane]["case-0"] = _payload(status)
        inputs[lane]["case-0"]["finite_support_instances"] = []
    result = _module().score_paired_support_cases(**inputs)
    summary = result["challenger"]["aggregate"]
    assert summary["exact_count_cases"] == int(status == "COMPLETE")
    assert summary["precision"] is summary["recall"] is summary["f1"] is None
    assert result["support_diagnostic_targets"]["challenger"]["diagnostic_targets_met"] is False
    assert result["physical_bend_count"] is None


@pytest.mark.parametrize("lane", ["incumbent_payloads_by_case", "challenger_payloads_by_case"])
@pytest.mark.parametrize("field,value", [("reference_frame", "SCANNER_FRAME"), ("unit", "cm"),
                                         ("line_convention", "SHARP_INTERSECTION"),
                                         ("instance_extraction_status", "PASS")])
def test_each_lane_must_obey_same_explicit_frame_unit_convention_and_status(lane, field, value):
    inputs = _inputs()
    inputs[lane]["case-0"][field] = value
    with pytest.raises(ValueError):
        _module().score_paired_support_cases(**inputs)


@pytest.mark.parametrize("change", ["missing_truth_value", "truth_wrong_frame", "unavailable_with_prediction",
                                    "nan_prediction", "threshold_dictionary", "threshold_zero"])
def test_invalid_truth_predictions_or_thresholds_invalidate_evaluation(change):
    module, inputs = _module(), _inputs()
    if change == "missing_truth_value":
        inputs["reference_records_by_case"]["case-0"] = None
    elif change == "truth_wrong_frame":
        inputs["reference_records_by_case"]["case-0"][0]["reference_frame"] = "SCANNER_FRAME"
    elif change == "unavailable_with_prediction":
        inputs["challenger_payloads_by_case"]["case-0"]["instance_extraction_status"] = "UNAVAILABLE"
    elif change == "nan_prediction":
        inputs["challenger_payloads_by_case"]["case-0"]["finite_support_instances"][0]["line_start"][0] = float("nan")
    elif change == "threshold_dictionary":
        inputs["thresholds"] = {"angle_deg": 0}
    else:
        with pytest.raises(ValueError):
            module.Thresholds(angle_deg=0)
        return
    with pytest.raises(ValueError):
        module.score_paired_support_cases(**inputs)


def test_declared_custom_match_threshold_is_recorded_and_shared_by_both_lanes():
    module, inputs = _module(), _inputs()
    inputs["incumbent_payloads_by_case"]["case-0"] = _payload(offset=1.5)
    inputs["challenger_payloads_by_case"]["case-0"] = _payload(offset=1.5)
    inputs["thresholds"] = module.Thresholds(transverse_offset_mm=1.)
    result = module.score_paired_support_cases(**inputs)
    assert result["matching_thresholds"]["transverse_offset_mm"] == 1.
    assert result["incumbent"]["aggregate"] == result["challenger"]["aggregate"]
    assert result["challenger"]["aggregate"]["false_positive"] == 1
