from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as runtime
from scripts import evaluate_f1_against_manual_gold as evaluator


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prediction(y: float = 0.0, *, token: str = "P0") -> dict[str, object]:
    return {
        "bend_id": f"B-{token}",
        "line_start": [0.0, y, 0.0],
        "line_end": [100.0, y, 0.0],
        "selection_route": "v51-test",
        "selector_score": None,
        "source_hypothesis_id": f"H-{token}",
        "source_kind": "plane_lattice",
        "generic_audit": {"source_partition_count": 3},
    }


def _semantic(predictions: list[dict[str, object]] | None = None) -> bytes:
    return runtime.canonical_bytes(
        {
            "audit": {"source_partition_count": 4},
            "cad_used": False,
            "candidate_count": len(predictions or [_prediction()]),
            "expected_count_used": False,
            "labels_used": False,
            "part_identity_used": False,
            "predictions": predictions or [_prediction()],
            "research_only": True,
            "source_only": True,
        }
    )


def _line_object(
    y: float,
    *,
    score: float | None = 0.8,
    source: str | None = "plane_lattice",
    token: str = "LO1",
    start: float = 0.0,
    end: float = 100.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": token,
        "span_endpoints": [[start, y, 0.0], [end, y, 0.0]],
        "finite_support_fraction": 0.8,
        "endpoint_support_fraction": 0.8,
        "two_sided_transition_score": 0.8,
        "normal_second_eigenvalue": 0.2,
    }
    if source is not None:
        row["representative_source"] = source
    if score is not None:
        row["score"] = score
    return row


def _pool(line_objects: list[dict[str, object]] | None = None) -> bytes:
    return runtime.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 99,
            "candidate_source_counts": {"ignored": 99},
            "candidates": [
                {
                    "candidate_id": "RAW-IGNORED",
                    "span_endpoints": [[0.0, 30.0, 0.0], [100.0, 30.0, 0.0]],
                    "candidate_source": "plane_lattice",
                    "score": 999.0,
                }
            ],
            "generator_diagnostics": {"candidate_count": 99},
            "inference_contract": {
                "cad_exposed": False,
                "expected_count_exposed": False,
                "manual_truth_exposed": False,
                "part_identifier_exposed": False,
            },
            "line_object_count": len(line_objects or [_line_object(8.0)]),
            "line_objects": line_objects or [_line_object(8.0)],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _validated_inputs(
    *,
    predictions: list[dict[str, object]] | None = None,
    line_objects: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], bytes, bytes]:
    semantic_bytes = _semantic(predictions)
    pool_bytes = _pool(line_objects)
    _semantic_value, validated_predictions = runtime._validate_semantic_bytes(
        semantic_bytes, expected_sha256=runtime.sha256_bytes(semantic_bytes)
    )
    _pool_value, candidates = runtime._validate_pool_bytes(
        pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
    )
    return validated_predictions, candidates, semantic_bytes, pool_bytes


def _rule(feature: str = "score", threshold: float = 0.5) -> dict[str, object]:
    return {"feature": feature, "operator": "ge", "threshold": threshold}


def _bindings(runtime_sha: str) -> dict[str, str]:
    values = {key: _sha(key) for key in runtime.BINDING_KEYS}
    values.update(runtime.KNOWN_BINDINGS)
    values["candidate6_runtime_source_sha256"] = runtime_sha
    return values


def _model_and_pass(
    rule: dict[str, object], runtime_sha: str
) -> tuple[dict[str, object], bytes, str]:
    bindings = _bindings(runtime_sha)
    checks = {key: True for key in runtime.GATE_CHECK_KEYS}
    pass_value = {
        "attempt_ordinal": 6,
        "bindings": bindings,
        "checks": checks,
        "gate_result_value_free": True,
        "holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "kind": runtime.PASS_KIND,
        "label_derived_metric_values_present": False,
        "pre_holdout_execution_anchor_created": False,
        "schema_version": 1,
        "selected_rule": rule,
        "status": "PASS",
        "training_gate_passed": True,
    }
    pass_bytes = runtime.canonical_bytes(pass_value)
    pass_sha = runtime.sha256_bytes(pass_bytes)
    model: dict[str, object] = {
        "additive_only": True,
        "attempt_ordinal": 6,
        "bindings": bindings,
        "deterministic_tie_break_direction": (
            runtime.DETERMINISTIC_TIE_BREAK_DIRECTION
        ),
        "deterministic_tie_break_inputs": list(
            runtime.DETERMINISTIC_TIE_BREAK_INPUTS
        ),
        "external_launcher_authentication_required": True,
        "fixed_eligibility_inputs": list(runtime.FIXED_ELIGIBILITY_INPUTS),
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "inference_features": (
            [rule["feature"]]
            if rule["feature"] == "source_kind"
            else [rule["feature"], "source_kind"]
        ),
        "kind": runtime.MODEL_KIND,
        "line_objects_only": True,
        "max_added_false_positives_train18": runtime.MAX_ADDED_FALSE_POSITIVES_TRAIN18,
        "max_additions_per_case": 1,
        "missing_evidence_abstains": True,
        "ownership_contract": runtime._ownership_contract(),
        "post_holdout_model_or_threshold_mutation": (
            "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        ),
        "pre_holdout_execution_anchor_created": False,
        "preserve_v51_prediction_values_and_order": True,
        "profile": runtime.PROFILE,
        "raw_candidates_ignored": True,
        "rule": rule,
        "schema_version": 1,
        "source_only": True,
        "train_gate_pass_binding": {"kind": runtime.PASS_KIND, "sha256": pass_sha},
        "training_gate_passed": True,
        **{field: False for field in runtime.FALSE_INFERENCE_DECLARATIONS},
        **{
            declaration: rule["feature"] == feature or feature == "source_kind"
            for feature, declaration in runtime.FEATURE_DECLARATION_KEYS.items()
        },
    }
    return model, pass_bytes, pass_sha


def test_runtime_top_level_imports_are_stdlib_only() -> None:
    path = Path(runtime.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "pandas", "scipy", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_fixed_rule_grid_is_exact_low_capacity_and_id_count_free() -> None:
    rules = runtime.fixed_rule_space()
    assert len(rules) == 32
    assert rules == runtime.fixed_rule_space()
    assert len({runtime.canonical_bytes(rule) for rule in rules}) == 32
    assert {rule["feature"] for rule in rules} == {
        "endpoint_support_fraction",
        "finite_support_fraction",
        "line_length_mm",
        "normal_second_eigenvalue",
        "score",
        "source_kind",
        "two_sided_transition_score",
    }
    encoded = runtime.canonical_bytes(rules).decode("utf-8")
    for forbidden in (
        "case_id",
        "candidate_id",
        "bend_id",
        "source_hypothesis_id",
        "count",
        "filename",
    ):
        assert forbidden not in encoded
    assert all(
        rule["operator"] == ("eq" if rule["feature"] == "source_kind" else "ge")
        for rule in rules
    )


def test_preflight_exercises_all_rules_without_source_volume_disclosure() -> None:
    _predictions, _candidates, semantic_bytes, pool_bytes = _validated_inputs()
    result = runtime.preflight_candidate6_source_bytes(
        semantic_bytes,
        pool_bytes,
        v51_semantic_sha256=runtime.sha256_bytes(semantic_bytes),
        pool_sha256=runtime.sha256_bytes(pool_bytes),
    )
    assert result["status"] == "SOURCE_COMPATIBLE"
    assert result["all_fixed_rule_forms_exercised"] is True
    assert result["v51_prediction_values_and_order_preserved"] is True
    assert result["candidate_metric_computed"] is False
    forbidden_keys = {
        "candidate_count",
        "feature_row_count",
        "prediction_row_count",
        "rule_path_check_count",
        "missing_numeric_feature_counts",
    }
    assert forbidden_keys.isdisjoint(result)


def test_v51_prefix_is_exact_and_one_addition_is_appended() -> None:
    predictions, candidates, _semantic_bytes, _pool_bytes = _validated_inputs(
        predictions=[_prediction(0.0, token="A"), _prediction(20.0, token="B")],
        line_objects=[_line_object(8.0), _line_object(40.0, token="LO2")],
    )
    selected = runtime._select_addition(predictions, candidates, _rule())
    assert selected is not None
    output = list(predictions) + [runtime._addition_prediction(selected)]
    assert runtime.canonical_bytes(output[:2]) == runtime.canonical_bytes(predictions)
    assert len(output) == 3
    assert output[-1]["selection_route"] == "candidate6_conservative_recall_recovery_v1"


def test_candidate_ids_and_list_order_never_affect_selection() -> None:
    rows = [
        _line_object(10.0, score=0.8, token="FIRST-ID"),
        _line_object(8.0, score=0.8, token="SECOND-ID"),
    ]
    predictions, first_candidates, _semantic_bytes, _pool_bytes = _validated_inputs(
        line_objects=rows
    )
    changed = deepcopy(rows)
    changed[0]["candidate_id"] = "ZZZ"
    changed[1]["candidate_id"] = "AAA"
    _p2, second_candidates, _s2, _pool2 = _validated_inputs(
        line_objects=list(reversed(changed))
    )
    first = runtime._select_addition(predictions, first_candidates, _rule())
    second = runtime._select_addition(predictions, second_candidates, _rule())
    assert first is not None and second is not None
    assert runtime._addition_prediction(first) == runtime._addition_prediction(second)


def test_numeric_rule_prefers_higher_value_and_categorical_ties_on_geometry() -> None:
    rows = [
        _line_object(8.0, score=0.6, token="A"),
        _line_object(10.0, score=0.9, token="B"),
    ]
    predictions, candidates, _semantic_bytes, _pool_bytes = _validated_inputs(
        line_objects=rows
    )
    numeric = runtime._select_addition(predictions, candidates, _rule())
    assert numeric is not None
    assert numeric["identity_projection"]["line_start"][1] == 10.0
    categorical_rule = {
        "feature": "source_kind",
        "operator": "eq",
        "value": "plane_lattice",
    }
    categorical = runtime._select_addition(predictions, candidates, categorical_rule)
    assert categorical is not None
    assert categorical["identity_projection"]["line_start"][1] == 10.0


def test_missing_selected_feature_abstains() -> None:
    predictions, candidates, _semantic_bytes, _pool_bytes = _validated_inputs(
        line_objects=[_line_object(8.0, score=None)]
    )
    assert runtime._select_addition(predictions, candidates, _rule()) is None
    source_missing = _line_object(8.0, source=None)
    predictions, candidates, _semantic_bytes, _pool_bytes = _validated_inputs(
        line_objects=[source_missing]
    )
    source_rule = {
        "feature": "source_kind",
        "operator": "eq",
        "value": "plane_lattice",
    }
    assert runtime._select_addition(predictions, candidates, source_rule) is None


def test_numeric_rule_discloses_fixed_source_kind_eligibility() -> None:
    runtime_sha = _sha("runtime")
    model, pass_bytes, pass_sha = _model_and_pass(_rule(), runtime_sha)
    assert model["inference_features"] == ["score", "source_kind"]
    assert model["source_kind_feature_used"] is True
    assert model["fixed_eligibility_inputs"] == [
        "finite_line_geometry",
        "source_kind",
    ]
    assert model["deterministic_tie_break_inputs"] == [
        "canonical_line_endpoints",
        "source_kind",
    ]
    runtime._validate_model(
        model,
        pass_bytes=pass_bytes,
        pass_sha256=pass_sha,
        verified_runtime_source_sha256=runtime_sha,
    )
    hidden = deepcopy(model)
    hidden["source_kind_feature_used"] = False
    with pytest.raises(ValueError, match="feature declaration"):
        runtime._validate_model(
            hidden,
            pass_bytes=pass_bytes,
            pass_sha256=pass_sha,
            verified_runtime_source_sha256=runtime_sha,
        )
    hidden_tie_break = deepcopy(model)
    hidden_tie_break["deterministic_tie_break_inputs"] = ["source_kind"]
    with pytest.raises(ValueError, match="status/source contract"):
        runtime._validate_model(
            hidden_tie_break,
            pass_bytes=pass_bytes,
            pass_sha256=pass_sha,
            verified_runtime_source_sha256=runtime_sha,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bend_id", "C6R-collides-with-runtime"),
        ("source_hypothesis_id", "C6G-collides-with-runtime"),
    ],
)
def test_v51_predictions_cannot_occupy_reserved_candidate6_id_namespaces(
    field: str, value: str
) -> None:
    prediction = _prediction()
    prediction[field] = value
    semantic_bytes = _semantic([prediction])
    with pytest.raises(ValueError, match="reserved Candidate6"):
        runtime._validate_semantic_bytes(
            semantic_bytes,
            expected_sha256=runtime.sha256_bytes(semantic_bytes),
        )


def test_ownership_boundary_is_four_mm_with_declared_conservative_numeric_band() -> None:
    incumbent = ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    at_boundary = ((0.0, 4.0, 0.0), (100.0, 4.0, 0.0))
    beyond = ((0.0, 4.000001, 0.0), (100.0, 4.000001, 0.0))
    short = ((0.0, 0.0, 0.0), (50.0, 0.0, 0.0))
    disjoint = ((200.0, 0.0, 0.0), (300.0, 0.0, 0.0))
    perpendicular_same_midpoint = ((50.0, -50.0, 0.0), (50.0, 50.0, 0.0))
    assert runtime._ownership_conflict(at_boundary, [incumbent]) is True
    assert runtime._ownership_conflict(beyond, [incumbent]) is False
    assert runtime._ownership_conflict(short, [incumbent]) is False
    assert runtime._ownership_conflict(disjoint, [incumbent]) is False
    assert runtime._ownership_conflict(perpendicular_same_midpoint, [incumbent]) is False


def test_every_evaluator_match_in_conditioned_boundary_grid_is_runtime_conflict() -> None:
    np = pytest.importorskip("numpy")
    thresholds = evaluator.Thresholds()
    for translation in (-999_000.0, 0.0, 999_000.0):
        incumbent = (
            (translation, 0.0, 0.0),
            (translation + 100.0, 0.0, 0.0),
        )
        for angle_deg in (0.0, 4.99999, 5.0, 5.00001, 5.1):
            radians = math.radians(angle_deg)
            axis = (math.cos(radians), math.sin(radians), 0.0)
            for offset in (3.999999, 4.0, 4.000000001, 4.0001):
                for shift in (39.9999, 40.0, 40.0001):
                    start = (translation + shift, offset, 0.0)
                    end = tuple(start[index] + 100.0 * axis[index] for index in range(3))
                    for candidate in ((start, end), (end, start)):
                        metrics = evaluator.pair_metrics(
                            np.asarray(incumbent), np.asarray(candidate)
                        )
                        if evaluator.is_match(metrics, thresholds):
                            assert runtime._ownership_conflict(candidate, [incumbent]) is True


def test_finite_but_overflowing_geometry_fails_before_ownership() -> None:
    huge = ((-1.0e154, 0.0, 0.0), (1.0e154, 0.0, 0.0))
    with pytest.raises(ValueError, match="conditioned"):
        runtime.line_pair_metrics(huge, huge)
    with pytest.raises(ValueError, match="conditioned"):
        runtime._ownership_conflict(huge, [huge])


def test_reversed_candidate_endpoints_have_identical_selection_and_output() -> None:
    forward = _line_object(8.0, token="FORWARD")
    reversed_row = deepcopy(forward)
    reversed_row["span_endpoints"] = list(reversed(reversed_row["span_endpoints"]))
    predictions, forward_candidates, _s1, _p1 = _validated_inputs(
        line_objects=[forward]
    )
    _predictions, reverse_candidates, _s2, _p2 = _validated_inputs(
        line_objects=[reversed_row]
    )
    first = runtime._select_addition(predictions, forward_candidates, _rule())
    second = runtime._select_addition(predictions, reverse_candidates, _rule())
    assert first is not None and second is not None
    assert runtime._addition_prediction(first) == runtime._addition_prediction(second)


def test_v51_ids_and_prediction_order_do_not_affect_selected_geometry() -> None:
    original_predictions = [_prediction(0.0, token="A"), _prediction(20.0, token="B")]
    predictions, candidates, _s1, _p1 = _validated_inputs(
        predictions=original_predictions,
        line_objects=[_line_object(8.0, score=0.9)],
    )
    changed_predictions = deepcopy(original_predictions)
    changed_predictions[0]["bend_id"] = "COMPLETELY-DIFFERENT-BEND"
    changed_predictions[0]["source_hypothesis_id"] = "COMPLETELY-DIFFERENT-SOURCE"
    changed_predictions[1]["bend_id"] = "OTHER-BEND"
    changed_predictions[1]["source_hypothesis_id"] = "OTHER-SOURCE"
    reversed_predictions, changed_candidates, _s2, _p2 = _validated_inputs(
        predictions=list(reversed(changed_predictions)),
        line_objects=[_line_object(8.0, score=0.9)],
    )
    first = runtime._select_addition(predictions, candidates, _rule())
    second = runtime._select_addition(
        reversed_predictions, changed_candidates, _rule()
    )
    assert first is not None and second is not None
    assert runtime._addition_prediction(first) == runtime._addition_prediction(second)


@pytest.mark.parametrize(
    "left,right",
    [
        (((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)), ((0.0, 4.0, 0.0), (100.0, 4.0, 0.0))),
        (((0.0, 0.0, 0.0), (50.0, 50.0, 0.0)), ((2.0, 0.0, 0.0), (52.0, 50.0, 0.0))),
        (((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)), ((20.0, 2.0, 0.0), (80.0, 2.0, 0.0))),
    ],
)
def test_ownership_metrics_match_bound_evaluator(left, right) -> None:  # type: ignore[no-untyped-def]
    np = pytest.importorskip("numpy")
    expected = evaluator.pair_metrics(np.asarray(left), np.asarray(right))
    observed = runtime.line_pair_metrics(left, right)
    assert observed["axis_angle_deg"] == pytest.approx(expected["axis_angle_deg"], abs=2e-6)
    assert observed["transverse_offset_mm"] == pytest.approx(
        expected["transverse_offset_mm"], abs=1e-10
    )
    assert observed["incumbent_span_overlap"] == pytest.approx(
        expected["gold_span_overlap"], abs=1e-10
    )
    assert observed["candidate_span_overlap"] == pytest.approx(
        expected["prediction_span_overlap"], abs=1e-10
    )


def test_raw_candidates_and_accounting_metadata_are_inert() -> None:
    pool_value = json.loads(_pool())
    first_bytes = runtime.canonical_bytes(pool_value)
    _pool_first, first = runtime._validate_pool_bytes(
        first_bytes, expected_sha256=runtime.sha256_bytes(first_bytes)
    )
    pool_value["candidate_count"] = 1_000_000
    pool_value["line_object_count"] = 1_000_000
    pool_value["candidate_source_counts"] = {"changed": 1_000_000}
    pool_value["candidates"] = [
        {
            "candidate_id": "CHANGED-RAW-ID",
            "span_endpoints": [[0.0, 999.0, 0.0], [100.0, 999.0, 0.0]],
            "candidate_source": "plane_lattice",
            "score": -999.0,
        }
    ]
    second_bytes = runtime.canonical_bytes(pool_value)
    _pool_second, second = runtime._validate_pool_bytes(
        second_bytes, expected_sha256=runtime.sha256_bytes(second_bytes)
    )
    assert first == second


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("finite_support_fraction", -0.01),
        ("finite_support_fraction", 1.01),
        ("endpoint_support_fraction", 1.01),
        ("two_sided_transition_score", -0.01),
    ],
)
def test_physical_feature_domains_are_fail_closed(feature: str, value: float) -> None:
    row = _line_object(8.0)
    row[feature] = value
    pool_bytes = _pool([row])
    with pytest.raises(ValueError, match=r"physical \[0,1\] domain"):
        runtime._validate_pool_bytes(
            pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
        )


def test_normal_second_eigenvalue_is_nonnegative_not_assumed_normalized() -> None:
    negative = _line_object(8.0)
    negative["normal_second_eigenvalue"] = -0.01
    negative_bytes = _pool([negative])
    with pytest.raises(ValueError, match="must be nonnegative"):
        runtime._validate_pool_bytes(
            negative_bytes,
            expected_sha256=runtime.sha256_bytes(negative_bytes),
        )
    unnormalized = _line_object(8.0)
    unnormalized["normal_second_eigenvalue"] = 1.01
    unnormalized_bytes = _pool([unnormalized])
    _pool_value, candidates = runtime._validate_pool_bytes(
        unnormalized_bytes,
        expected_sha256=runtime.sha256_bytes(unnormalized_bytes),
    )
    assert candidates[0]["features"]["normal_second_eigenvalue"] == 1.01


def test_conflicting_source_aliases_fail_closed_but_wrapper_alias_is_allowed() -> None:
    wrapper = _line_object(8.0)
    wrapper["candidate_source"] = "cross_source_line_object"
    pool_bytes = _pool([wrapper])
    _pool_value, candidates = runtime._validate_pool_bytes(
        pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
    )
    assert candidates[0]["identity_projection"]["source_kind"] == "plane_lattice"

    conflicting = deepcopy(wrapper)
    conflicting["source_kind"] = "cylindrical_standard"
    pool_bytes = _pool([conflicting])
    with pytest.raises(ValueError, match="aliases disagree"):
        runtime._validate_pool_bytes(
            pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
        )


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "gold",
        "manualTruth",
        "is_true_positive",
        "falseNegative",
        "expectedCount",
        "sender_count",
        "partFilename",
        "cadDrawing",
        "holdoutStatus",
        "outcome",
    ],
)
def test_recursive_forbidden_aliases_fail_closed(forbidden_key: str) -> None:
    semantic = json.loads(_semantic())
    semantic["audit"][forbidden_key] = False
    semantic_bytes = runtime.canonical_bytes(semantic)
    with pytest.raises(ValueError, match="forbidden key"):
        runtime._validate_semantic_bytes(
            semantic_bytes, expected_sha256=runtime.sha256_bytes(semantic_bytes)
        )
    pool = json.loads(_pool())
    pool["generator_diagnostics"][forbidden_key] = False
    pool_bytes = runtime.canonical_bytes(pool)
    with pytest.raises(ValueError, match="forbidden key"):
        runtime._validate_pool_bytes(
            pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(score=float("nan")),
        lambda row: row.update(span_endpoints=[[0, 0, 0], [0, 0, 0]]),
        lambda row: row.update(span_endpoints=[[0, 0, 0], [math.inf, 0, 0]]),
        lambda row: row.update(
            line_start=[0, 0, 0], line_end=[10, 0, 0], span_endpoints=[[0, 1, 0], [10, 1, 0]]
        ),
    ],
)
def test_nonfinite_degenerate_or_conflicting_line_object_fails(mutation) -> None:  # type: ignore[no-untyped-def]
    row = _line_object(8.0)
    mutation(row)
    pool_bytes = _pool([row])
    with pytest.raises(ValueError):
        runtime._validate_pool_bytes(
            pool_bytes, expected_sha256=runtime.sha256_bytes(pool_bytes)
        )


def test_apply_is_additive_and_preserves_original_prediction_objects(monkeypatch) -> None:
    predictions_input = [_prediction(0.0, token="A"), _prediction(20.0, token="B")]
    semantic_bytes = _semantic(predictions_input)
    pool_bytes = _pool([_line_object(8.0, score=0.9), _line_object(40.0, score=0.8)])
    runtime_sha = _sha("runtime")
    model, pass_bytes, pass_sha = _model_and_pass(_rule(), runtime_sha)
    monkeypatch.setattr(
        runtime, "_validate_candidate5_terminal", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        runtime, "_validate_candidate5_disposition", lambda *_args, **_kwargs: {}
    )
    result = runtime.apply_candidate6(
        semantic_bytes,
        pool_bytes,
        model,
        pass_bytes,
        b"terminal",
        b"disposition",
        v51_semantic_sha256=runtime.sha256_bytes(semantic_bytes),
        pool_sha256=runtime.sha256_bytes(pool_bytes),
        train_gate_pass_sha256=pass_sha,
        candidate5_terminal_sha256=runtime.CANDIDATE5_TERMINAL_SHA256,
        candidate5_disposition_sha256=runtime.CANDIDATE5_DISPOSITION_SHA256,
        verified_runtime_source_sha256=runtime_sha,
    )
    assert result["addition_made"] is True
    assert result["predictions"][:2] == predictions_input
    assert runtime.canonical_bytes(result["predictions"][:2]) == runtime.canonical_bytes(
        predictions_input
    )
    assert len(result["predictions"]) == 3
    assert all(result[field] is False for field in runtime.FALSE_INFERENCE_DECLARATIONS)


def test_model_rejects_any_forbidden_feature_declaration() -> None:
    runtime_sha = _sha("runtime")
    model, pass_bytes, pass_sha = _model_and_pass(_rule(), runtime_sha)
    for field in runtime.FALSE_INFERENCE_DECLARATIONS:
        changed = deepcopy(model)
        changed[field] = True
        with pytest.raises(ValueError, match="status/source contract"):
            runtime._validate_model(
                changed,
                pass_bytes=pass_bytes,
                pass_sha256=pass_sha,
                verified_runtime_source_sha256=runtime_sha,
            )


def test_rule_outside_exact_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        runtime.validate_rule(
            {"feature": "score", "operator": "ge", "threshold": 0.5000001}
        )
    with pytest.raises(ValueError, match="unsupported"):
        runtime.validate_rule(
            {"feature": "candidate_count", "operator": "ge", "threshold": 1.0}
        )
