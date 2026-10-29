from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import pytest

from backend import f1_train18_candidate5_conservative_deletion_v1 as c5


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prediction(
    bend_id: str = "B001", *, selector_score: object = 0.4
) -> dict[str, object]:
    row: dict[str, object] = {
        "bend_id": bend_id,
        "line_start": [0.0, 0.0, 0.0],
        "line_end": [10.0, 0.0, 0.0],
        "selection_route": "full_resolution",
        "source_hypothesis_id": f"H-{bend_id}",
        "source_kind": "plane_lattice",
    }
    if selector_score is not ...:
        row["selector_score"] = selector_score
    return row


def _semantic(predictions: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "expected_count_used": False,
        "candidate_count": 17,
        "audit": {
            "input_prediction_count": 4,
            "source_partition_count": 2,
            "notes": ["source-derived", {"retry_count": 1}],
        },
        "scan_sha256": _sha("ignored internal scan identity"),
        "case_id": "IGNORED_CASE_METADATA",
        "predictions": predictions if predictions is not None else [_prediction()],
    }


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _preflight(value: dict[str, object]) -> dict[str, object]:
    data = _bytes(value)
    return c5.preflight_candidate5_semantic_bytes(
        data, v51_semantic_sha256=c5.sha256_bytes(data)
    )


def test_preflight_accepts_generic_counts_and_audit_metadata_but_ignores_them() -> None:
    result = _preflight(_semantic())
    assert result["status"] == "SOURCE_COMPATIBLE"
    assert result["generic_source_count_and_audit_metadata_ignored"] is True
    assert result["prediction_output_projection_sanitized"] is True
    assert result["semantic_internal_scan_identity_used"] is False
    assert "prediction_count" not in result
    assert "feature_value" not in result


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "labels",
        "groundTruth",
        "gold_bends",
        "expectedBendCount",
        "sender_count",
        "cadBends",
        "drawing_metadata",
        "part_id",
        "fileName",
    ],
)
def test_recursive_forbidden_keys_are_rejected(forbidden_key: str) -> None:
    value = _semantic()
    value["audit"] = {"deep": [{"source": {forbidden_key: "forbidden"}}]}
    with pytest.raises(ValueError, match="forbidden key"):
        _preflight(value)


@pytest.mark.parametrize(
    "forbidden_key", ["label", "truth_class", "gold", "expected_count", "sender_id", "cad_id", "drawing", "partNumber", "filename"]
)
def test_forbidden_prediction_metadata_is_rejected(forbidden_key: str) -> None:
    row = _prediction()
    row["metadata"] = {forbidden_key: 1}
    with pytest.raises(ValueError, match="forbidden key"):
        _preflight(_semantic([row]))


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "outcome",
        "outcomes",
        "is_true_positive",
        "isFalsePositive",
        "is_true_negative",
        "falseNegative",
        "true_positive_count",
        "fp_count",
        "target_truth",
        "targetClass",
    ],
)
def test_truth_outcome_and_target_aliases_fail_closed(forbidden_key: str) -> None:
    row = _prediction()
    row[forbidden_key] = False
    with pytest.raises(ValueError, match="forbidden key"):
        _preflight(_semantic([row]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_only", False),
        ("research_only", 1),
        ("labels_used", 0),
        ("cad_used", None),
        ("part_identity_used", True),
        ("expected_count_used", None),
    ],
)
def test_required_top_level_flags_are_strict(field: str, value: object) -> None:
    semantic = _semantic()
    semantic[field] = value
    with pytest.raises(ValueError, match="requires"):
        _preflight(semantic)


def test_missing_required_top_level_flag_is_rejected() -> None:
    semantic = _semantic()
    semantic.pop("labels_used")
    with pytest.raises(ValueError, match="labels_used"):
        _preflight(semantic)


def test_optional_top_level_guard_must_be_false_when_present() -> None:
    semantic = _semantic()
    semantic["sender_count_used"] = None
    with pytest.raises(ValueError, match="sender_count_used"):
        _preflight(semantic)


@pytest.mark.parametrize("score", [..., None, -0.25, 0, 1.5])
def test_selector_score_may_be_absent_null_or_finite(score: object) -> None:
    row = _prediction(selector_score=score)
    result = _preflight(_semantic([row]))
    assert result["feature_extraction_complete"] is True
    semantic_bytes = _bytes(_semantic([row]))
    _semantic_value, sanitized = c5._validate_semantic_bytes(
        semantic_bytes, expected_sha256=c5.sha256_bytes(semantic_bytes)
    )
    if score is ...:
        assert "selector_score" not in sanitized[0]
    else:
        assert sanitized[0]["selector_score"] == score


@pytest.mark.parametrize("score", [True, False, math.inf, -math.inf, math.nan, "0.2"])
def test_invalid_selector_score_is_rejected(score: object) -> None:
    with pytest.raises(ValueError, match="selector_score|non-finite"):
        _preflight(_semantic([_prediction(selector_score=score)]))


def test_missing_selector_score_is_missing_evidence_and_keeps_prediction() -> None:
    features = c5.extract_prediction_features(_prediction(selector_score=...))
    for operator in ("le", "ge"):
        rule = {"feature": "selector_score", "operator": operator, "threshold": 0.0}
        assert c5._matches(rule, features) is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("line_start"),
        lambda row: row.__setitem__("line_end", [0.0, 0.0, 0.0]),
        lambda row: row.__setitem__("line_end", [1.0, 2.0]),
        lambda row: row.__setitem__("line_end", [1.0, True, 2.0]),
        lambda row: row.__setitem__("line_end", [1.0, math.inf, 2.0]),
    ],
)
def test_prediction_geometry_is_strict(mutation) -> None:  # type: ignore[no-untyped-def]
    row = _prediction()
    mutation(row)
    with pytest.raises(ValueError):
        _preflight(_semantic([row]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bend_id", ""),
        ("bend_id", " B001"),
        ("selection_route", "route\n"),
        ("source_kind", 7),
        ("source_hypothesis_id", "x" * 257),
    ],
)
def test_prediction_identity_and_source_fields_are_strict(field: str, value: object) -> None:
    row = _prediction()
    row[field] = value
    with pytest.raises(ValueError):
        _preflight(_semantic([row]))


def test_duplicate_bend_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _preflight(_semantic([_prediction("B001"), _prediction("B001")]))


def test_sanitized_projection_strips_all_ignored_metadata() -> None:
    row = _prediction()
    row.update(
        {
            "candidate_count": 9,
            "audit": {"retry_count": 2},
            "scan_sha256": _sha("ignored"),
            "case_id": "ignored",
            "anchor": [1.0, 2.0, 3.0],
        }
    )
    data = _bytes(_semantic([row]))
    _source, sanitized = c5._validate_semantic_bytes(
        data, expected_sha256=c5.sha256_bytes(data)
    )
    assert set(sanitized[0]) == {
        "bend_id",
        "line_start",
        "line_end",
        "selection_route",
        "selector_score",
        "source_hypothesis_id",
        "source_kind",
    }
    assert not ({"candidate_count", "audit", "scan_sha256", "case_id", "anchor"} & set(sanitized[0]))


@pytest.mark.parametrize(
    "rule",
    [
        {"feature": "line_length_mm", "operator": "le", "threshold": 5.0},
        {"feature": "selector_score", "operator": "ge", "threshold": 0.4},
        {"feature": "source_kind", "operator": "eq", "value": "plane_lattice"},
        {"feature": "selection_route", "operator": "eq", "value": "full_resolution"},
    ],
)
def test_single_rule_allowlist_accepts_only_source_prediction_features(rule) -> None:  # type: ignore[no-untyped-def]
    c5.validate_rule(rule)


@pytest.mark.parametrize(
    "rule",
    [
        {"feature": "candidate_count", "operator": "ge", "threshold": 2},
        {"feature": "case_id", "operator": "eq", "value": "T001"},
        {"feature": "line_length_mm", "operator": "eq", "threshold": 2},
        {"feature": "source_kind", "operator": "eq", "value": ""},
        {"feature": "selector_score", "operator": "le", "threshold": True},
    ],
)
def test_rule_rejects_counts_identity_and_bad_schema(rule) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        c5.validate_rule(rule)


def test_duplicate_json_keys_and_nonfinite_json_are_rejected() -> None:
    duplicate = b'{"source_only":true,"source_only":true}'
    with pytest.raises(ValueError, match="duplicate"):
        c5._parse_json_bytes(duplicate, name="duplicate", canonical_required=False)
    with pytest.raises(ValueError, match="non-finite"):
        c5._parse_json_bytes(b'{"x":NaN}', name="nan", canonical_required=False)


def _synthetic_terminals(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, str, bytes, str]:
    c3 = {
        "schema_version": 1,
        "kind": c5.TERMINAL_KIND,
        "candidate_id": 3,
        "attempt_ordinal": 3,
        "status": c5.CANDIDATE3_TERMINAL_STATUS,
        "candidate3_continuation_authorized": False,
        "holdout_execution_authorized": False,
        "holdout_access_count": 0,
        "holdout_score_opened": False,
        "promotion_eligible": False,
        "post_terminal_candidate3_model_or_threshold_mutation": "FORBIDDEN",
    }
    c3_bytes = c5.canonical_bytes(c3)
    c3_sha = c5.sha256_bytes(c3_bytes)
    c4 = {
        "schema_version": 1,
        "kind": c5.TERMINAL_KIND,
        "candidate_id": 4,
        "attempt_ordinal": 4,
        "status": c5.CANDIDATE4_TERMINAL_STATUS,
        "candidate4_continuation_authorized": False,
        "holdout_execution_authorized": False,
        "holdout_access_count": 0,
        "holdout_score_opened": False,
        "promotion_eligible": False,
        "post_terminal_candidate4_model_or_threshold_mutation": "FORBIDDEN",
        "artifact_bindings": {
            "candidate3_terminal": {
                "path": "research/f1_aug30_train18_candidate3_terminal_negative_v1.json",
                "sha256": c3_sha,
                "size_bytes": len(c3_bytes),
            }
        },
    }
    c4_bytes = c5.canonical_bytes(c4)
    c4_sha = c5.sha256_bytes(c4_bytes)
    monkeypatch.setattr(c5, "CANDIDATE3_TERMINAL_SHA256", c3_sha)
    monkeypatch.setattr(c5, "CANDIDATE3_TERMINAL_SIZE_BYTES", len(c3_bytes))
    monkeypatch.setattr(c5, "CANDIDATE4_TERMINAL_SHA256", c4_sha)
    monkeypatch.setattr(c5, "CANDIDATE4_TERMINAL_SIZE_BYTES", len(c4_bytes))
    monkeypatch.setitem(c5.KNOWN_BINDINGS, "candidate3_terminal_sha256", c3_sha)
    monkeypatch.setitem(c5.KNOWN_BINDINGS, "candidate4_terminal_sha256", c4_sha)
    return c3_bytes, c3_sha, c4_bytes, c4_sha


def _bindings(runtime_sha: str) -> dict[str, str]:
    bindings = {key: value for key, value in c5.KNOWN_BINDINGS.items()}
    for key in sorted(c5.BINDING_KEYS - set(bindings)):
        bindings[key] = _sha(key)
    bindings["candidate5_runtime_source_sha256"] = runtime_sha
    if bindings["fit_audit_core_sha256"] == bindings["train_read_ledger_sha256"]:
        bindings["train_read_ledger_sha256"] = _sha("distinct ledger")
    return bindings


def _model_and_pass(
    rule: dict[str, object], runtime_sha: str
) -> tuple[dict[str, object], bytes, str]:
    bindings = _bindings(runtime_sha)
    pass_artifact: dict[str, object] = {
        "schema_version": 1,
        "kind": c5.PASS_KIND,
        "status": "PASS",
        "attempt_ordinal": 5,
        "training_gate_passed": True,
        "holdout_score_opened": False,
        "holdout_execution_authorized": False,
        "pre_holdout_execution_anchor_created": False,
        "selected_rule": rule,
        "bindings": bindings,
        "checks": {key: True for key in c5.GATE_CHECK_KEYS},
        "gate_result_value_free": True,
        "label_derived_metric_values_present": False,
    }
    pass_bytes = c5.canonical_bytes(pass_artifact)
    pass_sha = c5.sha256_bytes(pass_bytes)
    model: dict[str, object] = {
        "schema_version": 1,
        "kind": c5.MODEL_KIND,
        "profile": c5.PROFILE,
        "attempt_ordinal": 5,
        "audit_metadata_feature_used": False,
        "bend_id_feature_used": False,
        "cad_feature_used": False,
        "candidate_count_feature_used": False,
        "training_gate_passed": True,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "pre_holdout_execution_anchor_created": False,
        "post_holdout_model_or_threshold_mutation": "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT",
        "source_only": True,
        "deletion_only": True,
        "v51_fallback_default": True,
        "observed_source_or_prediction_multiplicity_used": False,
        "case_identity_feature_used": False,
        "drawing_feature_used": False,
        "expected_count_feature_used": False,
        "filename_feature_used": False,
        "gold_or_truth_feature_used": False,
        "label_feature_used": False,
        "line_length_feature_used": rule["feature"] == "line_length_mm",
        "part_identity_feature_used": False,
        "runtime_identity_feature_used": False,
        "selection_route_feature_used": rule["feature"] == "selection_route",
        "selector_score_feature_used": rule["feature"] == "selector_score",
        "semantic_internal_scan_identity_feature_used": False,
        "sender_count_feature_used": False,
        "source_hypothesis_id_feature_used": False,
        "source_kind_feature_used": rule["feature"] == "source_kind",
        "rule": rule,
        "inference_features": [rule["feature"]],
        "bindings": bindings,
        "train_gate_pass_binding": {"kind": c5.PASS_KIND, "sha256": pass_sha},
    }
    return model, pass_bytes, pass_sha


def test_apply_requires_seals_deletes_by_one_rule_and_sanitizes(monkeypatch: pytest.MonkeyPatch) -> None:
    c3_bytes, c3_sha, c4_bytes, c4_sha = _synthetic_terminals(monkeypatch)
    runtime_sha = _sha("verified runtime")
    rule: dict[str, object] = {
        "feature": "line_length_mm",
        "operator": "ge",
        "threshold": 8.0,
    }
    model, pass_bytes, pass_sha = _model_and_pass(rule, runtime_sha)
    short = _prediction("B-SHORT", selector_score=...)
    short["line_end"] = [5.0, 0.0, 0.0]
    short["audit"] = {"candidate_count": 99}
    long = _prediction("B-LONG")
    semantic = _semantic([short, long])
    semantic_bytes = _bytes(semantic)
    result = c5.apply_candidate5(
        semantic_bytes,
        model,
        pass_bytes,
        c3_bytes,
        c4_bytes,
        v51_semantic_sha256=c5.sha256_bytes(semantic_bytes),
        train_gate_pass_sha256=pass_sha,
        candidate3_terminal_sha256=c3_sha,
        candidate4_terminal_sha256=c4_sha,
        verified_runtime_source_sha256=runtime_sha,
    )
    assert [row["bend_id"] for row in result["predictions"]] == ["B-SHORT"]
    assert "audit" not in result["predictions"][0]
    assert "selector_score" not in result["predictions"][0]
    assert result["deletion_audit"] == [
        {
            "bend_id": "B-LONG",
            "feature": "line_length_mm",
            "feature_value": 10.0,
            "reason": "frozen_train_nested_single_rule",
        }
    ]
    assert result["scan_identity_authority"] == "ATTESTED_CACHE_MANIFEST_ONLY"
    assert result["semantic_internal_scan_identity_used"] is False
    assert result["line_length_feature_used"] is True
    assert result["selection_route_feature_used"] is False
    assert result["selector_score_feature_used"] is False
    assert result["source_kind_feature_used"] is False
    assert result["case_id_used"] is False
    assert result["runtime_id_used"] is False
    assert result["gold_count_used"] is False
    assert result["label_count_used_at_inference"] is False
    assert result["post_holdout_model_or_threshold_mutation"] == (
        "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
    )
    for field in (
        "audit_metadata_feature_used",
        "bend_id_feature_used",
        "cad_feature_used",
        "candidate_count_feature_used",
        "case_identity_feature_used",
        "drawing_feature_used",
        "expected_count_feature_used",
        "filename_feature_used",
        "gold_or_truth_feature_used",
        "label_feature_used",
        "part_identity_feature_used",
        "runtime_identity_feature_used",
        "semantic_internal_scan_identity_feature_used",
        "sender_count_feature_used",
        "source_hypothesis_id_feature_used",
    ):
        assert result[field] is False


def test_apply_fails_closed_on_pass_or_terminal_tamper(monkeypatch: pytest.MonkeyPatch) -> None:
    c3_bytes, c3_sha, c4_bytes, c4_sha = _synthetic_terminals(monkeypatch)
    runtime_sha = _sha("verified runtime")
    rule: dict[str, object] = {
        "feature": "source_kind",
        "operator": "eq",
        "value": "plane_lattice",
    }
    model, pass_bytes, pass_sha = _model_and_pass(rule, runtime_sha)
    semantic_bytes = _bytes(_semantic())
    common = {
        "v51_semantic_sha256": c5.sha256_bytes(semantic_bytes),
        "train_gate_pass_sha256": pass_sha,
        "candidate3_terminal_sha256": c3_sha,
        "candidate4_terminal_sha256": c4_sha,
        "verified_runtime_source_sha256": runtime_sha,
    }
    with pytest.raises(ValueError):
        c5.apply_candidate5(
            semantic_bytes,
            model,
            pass_bytes + b" ",
            c3_bytes,
            c4_bytes,
            **common,
        )
    with pytest.raises(ValueError):
        c5.apply_candidate5(
            semantic_bytes,
            model,
            pass_bytes,
            c3_bytes,
            c4_bytes + b" ",
            **common,
        )


def test_module_has_no_filesystem_or_pool_runtime_dependency() -> None:
    assert not hasattr(c5, "Path")
    assert "pool" not in c5.apply_candidate5.__code__.co_varnames
    assert "pool" not in c5.preflight_candidate5_semantic_bytes.__code__.co_varnames
