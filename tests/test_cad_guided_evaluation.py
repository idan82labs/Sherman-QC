"""Synthetic evaluator bridge checks against the exact retained pure source."""

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/cad_guided_evaluation.py"
LEGACY_MATCHER = ROOT / "scripts/evaluate_f1_against_manual_gold.py"
LEGACY_AGGREGATOR = ROOT / "scripts/f1_physical_evaluation_contract_v2.py"


def _module():
    name = "cad_guided_evaluation_isolated_test"
    module = ModuleType(name)
    module.__file__ = str(SOURCE)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), module.__dict__)
    finally:
        if prior is None:
            del sys.modules[name]
        else:
            sys.modules[name] = prior
    return module


def _legacy():
    namespace = {"dataclass": dataclass, "np": np, "math": math, "Any": Any,
                 "Mapping": Mapping, "Sequence": Sequence,
                 "linear_sum_assignment": linear_sum_assignment, "SCHEMA_VERSION": 2}
    for path, names in (
        (LEGACY_MATCHER, {"Thresholds", "_endpoints", "pair_metrics", "is_match", "_match_cost", "match_lines"}),
        (LEGACY_AGGREGATOR, {"_ratio", "_required_nonnegative_int", "score_localized_cases"}),
    ):
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in selected} == names
        # No experiment import, path defaults, file access function, or main.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _line(bend_id="line", start=(0., 0., 0.), end=(0., 0., 20.)):
    return {"bend_id": bend_id, "line_start": list(start), "line_end": list(end)}


def _instance(bend_id="prediction", start=(0., 0., 0.), end=(0., 0., 20.)):
    return {**_line(bend_id, start, end), "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT",
            "unit": "mm", "endpoint_semantics": "SAMPLED_SUPPORT_NOT_PHYSICAL_TRIM",
            "observation_status": "SAMPLE_SUPPORTED_CONDITIONAL", "physical_trim_endpoints": None}


def _payload(instances, status="COMPLETE"):
    return {"finite_support_instances": instances, "reference_frame": "CAD_REFERENCE",
            "unit": "mm", "line_convention": "CYLINDER_CENTER_AXIS_SUPPORT",
            "instance_extraction_status": status}


def _score(module, truth=None, instances=None, status="COMPLETE", case_id="constructed", **kwargs):
    return module.score_support_case(
        case_id=case_id, reference_records=[_line()] if truth is None else truth,
        prediction_payload=_payload([_instance()] if instances is None else instances, status),
        reference_frame="CAD_REFERENCE", reference_unit="mm",
        reference_convention="CYLINDER_CENTER_AXIS_SUPPORT", **kwargs)


def test_actual_scorer_bridge_does_not_use_expected_feature_or_measurement_counts():
    module = _module()
    payload = _payload([_instance()])
    payload.update(expected_feature_count=900, measured_conditional_feature_count=800,
                   observed_support_instance_count=700, observed_bend_count=None)
    row = module.score_support_case(case_id="one", reference_records=[_line()],
                                    prediction_payload=payload, reference_frame="CAD_REFERENCE",
                                    reference_unit="mm", reference_convention=module.LINE_CONVENTION)
    assert row["gold_count"] == row["prediction_count"] == row["true_positive"] == 1
    assert row["exact_count"] and row["exact_localized_target_set"]
    assert row["physical_bend_count"] is None and row["promotion_eligible"] is False
    json.dumps(row, allow_nan=False)


def test_wrong_parallel_location_can_have_exact_count_without_localized_success():
    module = _module()
    row = _score(module, instances=[_instance(start=(4.8, 0, 0), end=(4.8, 0, 20))])
    assert row["true_positive"] == 0 and row["false_positive"] == row["false_negative"] == 1
    assert row["exact_count"] is True
    assert row["exact_localized_target_set"] is False


def test_duplicate_predictions_are_penalized_one_to_one():
    row = _score(_module(), instances=[_instance("one"), _instance("two")])
    assert row["true_positive"] == 1 and row["false_positive"] == 1
    assert row["false_negative"] == 0 and row["exact_count"] is False


@pytest.mark.parametrize("fraction,expected", [(.2, 0), (.59, 0), (.6, 1), (.8, 1), (1., 1), (2., 0)])
def test_finite_support_is_scored_without_padding_or_shortening_to_truth(fraction, expected):
    row = _score(_module(), instances=[_instance(end=(0, 0, 20 * fraction))])
    assert row["true_positive"] == expected
    assert row["prediction_count"] == 1


@pytest.mark.parametrize("status", ["PARTIAL", "UNAVAILABLE"])
def test_incomplete_empty_inference_never_earns_zero_reference_count_credit(status):
    module = _module()
    row = _score(module, truth=[], instances=[], status=status)
    assert row["raw_exact_localized_set"] is True
    assert row["exact"] is row["exact_count"] is row["exact_localized_target_set"] is False
    result = module.score_support_cases([row], expected_case_ids=["constructed"])
    assert result["case_count"] == 1
    assert result["exact_count_accuracy"] == result["exact_localized_target_set_accuracy"] == 0.
    assert result["precision"] is result["recall"] is result["f1"] is None


def test_partial_inference_keeps_true_positives_but_does_not_claim_complete_count():
    module = _module()
    row = _score(module, status="PARTIAL")
    assert row["true_positive"] == 1 and row["false_positive"] == row["false_negative"] == 0
    assert row["exact_count"] is row["exact_localized_target_set"] is False
    result = module.score_support_cases([row], expected_case_ids=["constructed"])
    assert result["f1"] == 1. and result["inference_complete_rate"] == 0.


def test_failed_case_is_retained_in_denominators_and_false_negatives():
    module = _module()
    rows = [_score(module, case_id="good"),
            _score(module, instances=[], status="UNAVAILABLE", case_id="failed")]
    result = module.score_support_cases(rows, expected_case_ids=["good", "failed"])
    assert result["true_positive"] == result["false_negative"] == 1
    assert result["false_positive"] == 0
    assert result["precision"] == 1. and result["recall"] == .5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["exact_count_accuracy"] == result["inference_complete_rate"] == .5


@pytest.mark.parametrize("expected", [["constructed", "missing"], ["wrong"], [], ["constructed", "constructed"]])
def test_aggregate_rejects_missing_extra_or_ambiguous_requested_case_set(expected):
    module = _module()
    with pytest.raises(ValueError):
        module.score_support_cases([_score(module)], expected_case_ids=expected)


@pytest.mark.parametrize("field,value", [("reference_frame", "OTHER"), ("reference_frame", None),
                                       ("unit", "cm"), ("line_convention", "SHARP_INTERSECTION")])
def test_aggregate_rejects_mixed_or_undeclared_reference_contract(field, value):
    module = _module()
    first, second = _score(module, case_id="one"), _score(module, case_id="two")
    second[field] = value
    with pytest.raises(ValueError):
        module.score_support_cases([first, second], expected_case_ids=["one", "two"])


@pytest.mark.parametrize("field,value", [("reference_frame", "SCAN_REFERENCE"), ("unit", "m"),
                                       ("line_convention", "TANGENT_CONTACT_LINE"),
                                       ("line_convention", None), ("instance_extraction_status", "PASS")])
def test_no_silent_reference_convention_frame_unit_or_status_coercion(field, value):
    module = _module()
    payload = _payload([_instance()])
    payload[field] = value
    with pytest.raises(ValueError):
        module.score_support_case(case_id="one", reference_records=[_line()], prediction_payload=payload,
                                  reference_frame="CAD_REFERENCE", reference_unit="mm",
                                  reference_convention=module.LINE_CONVENTION)


@pytest.mark.parametrize("role", ["reference", "prediction"])
def test_explicit_per_line_frame_cannot_contradict_declared_case_frame(role):
    module = _module()
    truth, predictions = [_line()], [_instance()]
    (truth if role == "reference" else predictions)[0]["reference_frame"] = "SCAN_REFERENCE"
    with pytest.raises(ValueError, match="per-line frame mismatch"):
        _score(module, truth=truth, instances=predictions)


@pytest.mark.parametrize("change", ["nominal_line", "nan", "degenerate", "duplicate_id", "missing_array",
                                    "unavailable_with_line", "claimed_trim", "budget"])
def test_invalid_or_nonobserved_predictions_cannot_be_scored(change):
    module = _module()
    payload = _payload([_instance()])
    if change == "nominal_line":
        payload["finite_support_instances"] = [_line()]
    elif change == "nan":
        payload["finite_support_instances"][0]["line_start"][0] = float("nan")
    elif change == "degenerate":
        payload["finite_support_instances"][0]["line_end"] = [0., 0., 0.]
    elif change == "duplicate_id":
        payload["finite_support_instances"].append(_instance())
    elif change == "missing_array":
        del payload["finite_support_instances"]
    elif change == "unavailable_with_line":
        payload["instance_extraction_status"] = "UNAVAILABLE"
    elif change == "claimed_trim":
        payload["finite_support_instances"][0]["physical_trim_endpoints"] = [[0, 0, 0], [0, 0, 20]]
    else:
        payload["finite_support_instances"] = [_instance(str(i)) for i in range(257)]
    with pytest.raises(ValueError):
        module.score_support_case(case_id="one", reference_records=[_line()], prediction_payload=payload,
                                  reference_frame="CAD_REFERENCE", reference_unit="mm",
                                  reference_convention=module.LINE_CONVENTION)


def test_exact_retained_matcher_compatibility_on_256_fixed_invented_line_sets():
    module, legacy = _module(), _legacy()
    rng = np.random.default_rng(20260909)
    rows = []
    for case in range(256):
        gold = []
        predicted = []
        for index in range(case % 9):
            start = rng.uniform(-40, 40, 3)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            end = start + rng.uniform(1, 30) * direction
            gold.append(_line(f"G{index}", start, end))
            if (case + index) % 5:
                shift = rng.uniform(-3, 3, 3)
                predicted.append(_instance(f"P{index}", start + shift, end + shift))
            if (case + index) % 7 == 0:
                predicted.append(_instance(f"D{index}", start, end))
        expected = legacy["match_lines"](gold, predicted, legacy["Thresholds"]())
        actual = _score(module, truth=gold, instances=predicted, case_id=str(case))
        for key, value in expected.items():
            assert actual[key] == value, (case, key, actual[key], value)
        rows.append(actual)
    actual = module.score_support_cases(rows, expected_case_ids=[str(i) for i in range(256)])
    expected = legacy["score_localized_cases"](rows)
    for key in ("case_count", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1",
                "exact_count_cases", "exact_count_accuracy", "exact_localized_target_set_cases",
                "exact_localized_target_set_accuracy"):
        assert actual[key] == expected[key], key


@pytest.mark.parametrize("field,value", [("angle_deg", 0.), ("transverse_offset_mm", -1.),
                                       ("gold_span_overlap", 1.1), ("prediction_span_overlap", True)])
def test_invalid_matching_thresholds_are_rejected(field, value):
    module = _module()
    with pytest.raises(ValueError):
        module.Thresholds(**{field: value})
