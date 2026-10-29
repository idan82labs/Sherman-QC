from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from backend import f1_train18_candidate4_sidecar_burst_guard_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]
MISSING = object()
CANDIDATE3_TERMINAL_PATH = (
    ROOT / "research/f1_aug30_train18_candidate3_terminal_negative_v1.json"
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prediction(
    bend_id: str,
    *,
    sidecar: bool = False,
    geometry: str = "line",
    selector_score: object = MISSING,
    y: float = 0.0,
    extra: dict | None = None,
) -> dict:
    row = {
        "bend_id": bend_id,
        "source_kind": (
            runtime.SIDECAR_SOURCE_KIND if sidecar else "plane_lattice"
        ),
        "selection_route": (
            "F1_V41_V16_REAL_SCAN_COMPATIBILITY_V1"
            if sidecar
            else "v16_full_resolution"
        ),
        "source_hypothesis_id": f"hypothesis-{bend_id}",
    }
    if geometry == "line":
        row.update(
            {
                "line_start": [0.0, y, 0.0],
                "line_end": [10.0, y, 0.0],
            }
        )
    elif geometry == "span":
        row["span_endpoints"] = [[0.0, y, 0.0], [10.0, y, 0.0]]
    elif geometry == "axial":
        row.update(
            {
                "anchor": [5.0, y, 0.0],
                "axis_direction": [1.0, 0.0, 0.0],
                "span_mm": 10.0,
            }
        )
    else:
        raise AssertionError(geometry)
    if selector_score is not MISSING:
        row["selector_score"] = selector_score
    if extra:
        row.update(deepcopy(extra))
    return row


def _semantic(rows: list[dict], *, canonical: bool = True, **updates) -> bytes:
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_source_only_semantic_result",
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "predictions": rows,
    }
    payload.update(updates)
    if canonical:
        return runtime.canonical_bytes(payload)
    return json.dumps(payload, indent=2, sort_keys=False).encode("utf-8")


def _terminal() -> bytes:
    data = CANDIDATE3_TERMINAL_PATH.read_bytes()
    assert len(data) == runtime.CANDIDATE3_TERMINAL_SIZE_BYTES
    assert runtime.sha256_bytes(data) == runtime.CANDIDATE3_TERMINAL_SHA256
    return data


def _bindings(candidate3_terminal_sha256: str) -> dict[str, str]:
    bindings = {key: _hash(f"candidate4-binding|{key}") for key in runtime.BINDING_KEYS}
    bindings.update(runtime.KNOWN_BINDINGS)
    bindings["candidate3_terminal_sha256"] = candidate3_terminal_sha256
    bindings["candidate4_runtime_source_sha256"] = runtime.runtime_source_sha256()
    assert bindings["fit_audit_core_sha256"] != bindings["train_read_ledger_sha256"]
    return bindings


def _pass_bytes(rule: str, bindings: dict[str, str], **updates) -> bytes:
    payload = {
        "schema_version": 1,
        "kind": runtime.PASS_KIND,
        "status": "PASS",
        "attempt_ordinal": 4,
        "training_gate_passed": True,
        "holdout_score_opened": False,
        "holdout_execution_authorized": False,
        "pre_holdout_execution_anchor_created": False,
        "selected_rule": rule,
        "bindings": bindings,
        "checks": {key: True for key in runtime.GATE_CHECK_KEYS},
        "gate_result_value_free": True,
        "label_derived_metric_values_present": False,
    }
    payload.update(updates)
    return runtime.canonical_bytes(payload)


def _model(
    rule: str,
    bindings: dict[str, str],
    train_gate_pass_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "kind": runtime.MODEL_KIND,
        "profile": runtime.PROFILE,
        "attempt_ordinal": 4,
        "training_gate_passed": True,
        "freeze_status": "FROZEN_AFTER_TRAIN_GATE_PASS",
        "holdout_score_opened": False,
        "pre_holdout_execution_anchor_created": False,
        "post_holdout_model_or_threshold_mutation": (
            "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
        ),
        "source_only": True,
        "deletion_only": True,
        "v51_fallback_default": True,
        "rule": rule,
        "source_kind_used": True,
        "selection_route_used": False,
        "selector_score_used": False,
        "geometry_used_for_deletion": False,
        "observed_source_multiplicity_used": True,
        "case_id_feature_used": False,
        "runtime_id_feature_used": False,
        "filename_feature_used": False,
        "cad_feature_used": False,
        "expected_count_used": False,
        "sender_count_used": False,
        "gold_count_used": False,
        "label_count_used_at_inference": False,
        "bindings": bindings,
        "train_gate_pass_binding": {
            "kind": runtime.PASS_KIND,
            "sha256": train_gate_pass_sha256,
        },
    }


def _bundle(rule: str = runtime.EXPLOSION_RULES[0]):
    terminal = _terminal()
    terminal_sha = runtime.sha256_bytes(terminal)
    bindings = _bindings(terminal_sha)
    gate_pass = _pass_bytes(rule, bindings)
    gate_sha = runtime.sha256_bytes(gate_pass)
    model = _model(rule, bindings, gate_sha)
    return terminal, terminal_sha, bindings, gate_pass, gate_sha, model


def _apply(rows: list[dict], *, rule: str = runtime.EXPLOSION_RULES[0], canonical=True):
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle(rule)
    semantic = _semantic(rows, canonical=canonical)
    result = runtime.apply_candidate4(
        semantic,
        model,
        gate_pass,
        terminal,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        train_gate_pass_sha256=gate_sha,
        candidate3_terminal_sha256=terminal_sha,
    )
    return result, semantic, model, gate_pass, terminal


def test_frozen_search_space_contains_only_two_drop_all_rules() -> None:
    assert runtime.ATTEMPT_ORDINAL == 4
    assert runtime.EXPLOSION_RULES == (
        "DROP_ALL_SIDECARS_IF_AT_LEAST_2",
        "DROP_ALL_SIDECARS_IF_AT_LEAST_3",
    )
    assert (
        "fixed_holdout_stage_order_similar4_adjacent3_distant3"
        in runtime.GATE_CHECK_KEYS
    )
    assert (
        "deployed_model_exact_bytes_match_evaluated_predictions"
        in runtime.GATE_CHECK_KEYS
    )


@pytest.mark.parametrize(
    "rule,sidecar_count,expected_trigger",
    [
        (runtime.EXPLOSION_RULES[0], 1, False),
        (runtime.EXPLOSION_RULES[0], 2, True),
        (runtime.EXPLOSION_RULES[1], 2, False),
        (runtime.EXPLOSION_RULES[1], 3, True),
    ],
)
def test_rules_trigger_only_at_the_frozen_multiplicity(
    rule: str, sidecar_count: int, expected_trigger: bool
) -> None:
    rows = [_prediction("base")] + [
        _prediction(f"sidecar-{index}", sidecar=True)
        for index in range(sidecar_count)
    ]
    result, _semantic_bytes, _model_value, _pass, _terminal_bytes = _apply(
        rows, rule=rule
    )
    assert result["deletion_audit"]["triggered"] is expected_trigger
    expected_ids = ["base"] if expected_trigger else [row["bend_id"] for row in rows]
    assert [row["bend_id"] for row in result["predictions"]] == expected_ids


@pytest.mark.parametrize("score", [MISSING, None, -100.0, 100.0])
def test_selector_score_is_irrelevant_to_burst_deletion(score: object) -> None:
    rows = [
        _prediction("base", selector_score=score),
        _prediction("s1", sidecar=True, selector_score=score),
        _prediction("s2", sidecar=True, selector_score=score),
    ]
    result, *_ = _apply(rows)
    assert [row["bend_id"] for row in result["predictions"]] == ["base"]
    assert result["selector_score_used"] is False


@pytest.mark.parametrize("invalid", ["0.5", True])
def test_present_selector_score_still_obeys_flexible_v51_schema(invalid: object) -> None:
    rows = [_prediction("base", selector_score=invalid)]
    with pytest.raises(ValueError, match="selector_score"):
        _apply(rows)


@pytest.mark.parametrize("geometry", ["line", "span", "axial"])
def test_all_supported_v51_geometry_forms_are_accepted(geometry: str) -> None:
    row = _prediction(
        "base",
        geometry=geometry,
        extra={"upstream_audit": {"finite": True, "version": 51}},
    )
    result, *_ = _apply([row])
    assert runtime.canonical_bytes(result["predictions"][0]) == runtime.canonical_bytes(
        row
    )


def test_redundant_complete_geometry_forms_are_validated_and_retained() -> None:
    row = _prediction("base", geometry="line")
    row["span_endpoints"] = [row["line_start"], row["line_end"]]
    row.update(
        {
            "anchor": [5.0, 0.0, 0.0],
            "axis_direction": [1.0, 0.0, 0.0],
            "span_mm": 10.0,
        }
    )
    result, *_ = _apply([row])
    assert runtime.canonical_bytes(result["predictions"][0]) == runtime.canonical_bytes(
        row
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.pop("line_end"),
        lambda row: row.update(line_end=[0.0, 0.0, 0.0]),
        lambda row: row.update(line_start=[True, 0.0, 0.0]),
        lambda row: row.update(line_start=[0.0, 0.0]),
        lambda row: row.update(source_kind=""),
        lambda row: row.update(selection_route=""),
        lambda row: row.update(source_hypothesis_id=None),
    ],
)
def test_malformed_prediction_schema_is_rejected(mutator) -> None:
    row = _prediction("base")
    mutator(row)
    with pytest.raises(ValueError):
        _apply([row])


def test_finite_coordinates_with_overflowing_derived_geometry_are_rejected() -> None:
    row = _prediction("base")
    row["line_start"] = [-1e308, 0.0, 0.0]
    row["line_end"] = [1e308, 0.0, 0.0]
    with pytest.raises(ValueError, match="nondegenerate line"):
        _apply([row])

    row = _prediction("base", geometry="axial")
    row["axis_direction"] = [1.7e308, 1.7e308, 0.0]
    with pytest.raises(ValueError, match="degenerate"):
        _apply([row])


def test_missing_source_hypothesis_id_is_allowed_because_guard_does_not_use_it() -> None:
    row = _prediction("base")
    row.pop("source_hypothesis_id")
    result, *_ = _apply([row])
    assert result["predictions"] == [row]


def test_duplicate_prediction_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate V51 prediction"):
        _apply([_prediction("same"), _prediction("same", y=1.0)])


def test_route_token_alone_never_classifies_a_non_sidecar() -> None:
    route_only = _prediction("route-only")
    route_only["selection_route"] = "fragment_tolerant_parent_bridge_like_route"
    rows = [
        route_only,
        _prediction("s1", sidecar=True),
        _prediction("s2", sidecar=True),
    ]
    result, *_ = _apply(rows)
    assert result["predictions"] == [route_only]
    assert result["selection_route_used"] is False


def test_retained_rows_are_never_modified_and_keep_relative_input_order() -> None:
    rows = [
        _prediction("base-a", extra={"opaque": [1, {"x": "y"}]}),
        _prediction("s1", sidecar=True),
        _prediction("base-b", geometry="span", selector_score=None),
        _prediction("s2", sidecar=True),
    ]
    originals = {row["bend_id"]: runtime.canonical_bytes(row) for row in rows}
    result, semantic, *_ = _apply(rows)
    assert [row["bend_id"] for row in result["predictions"]] == ["base-a", "base-b"]
    for row in result["predictions"]:
        assert runtime.canonical_bytes(row) == originals[row["bend_id"]]
    assert semantic == _semantic(rows)
    assert result["deletion_audit"]["retained_predictions_canonical_byte_identical"] is True


def test_set_decision_is_permutation_invariant() -> None:
    rows = [
        _prediction("base-a"),
        _prediction("base-b", geometry="axial"),
        _prediction("s1", sidecar=True),
        _prediction("s2", sidecar=True),
    ]
    expected = {
        runtime.canonical_bytes(rows[0]),
        runtime.canonical_bytes(rows[1]),
    }
    for permutation in itertools.permutations(rows):
        result, *_ = _apply(list(permutation))
        assert {runtime.canonical_bytes(row) for row in result["predictions"]} == expected
        retained_input_order = [
            row["bend_id"] for row in permutation if row["source_kind"] != runtime.SIDECAR_SOURCE_KIND
        ]
        assert [row["bend_id"] for row in result["predictions"]] == retained_input_order


def test_empty_v51_prediction_set_is_valid_and_unchanged() -> None:
    result, *_ = _apply([])
    assert result["predictions"] == []
    assert result["deletion_audit"]["triggered"] is False


def test_noncanonical_semantic_bytes_are_allowed_when_externally_hash_bound() -> None:
    row = _prediction("base")
    result, semantic, *_ = _apply([row], canonical=False)
    assert result["input_v51_semantic_sha256"] == runtime.sha256_bytes(semantic)
    assert result["predictions"] == [row]


def test_semantic_hash_tamper_rejects_before_json_parsing() -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    valid = _semantic([_prediction("base")])
    tampered = b"not-json"
    with pytest.raises(ValueError, match="external SHA binding"):
        runtime.apply_candidate4(
            tampered,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(valid),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_duplicate_json_keys_are_rejected() -> None:
    semantic = (
        b'{"predictions":[],"research_only":true,"source_only":true,'
        b'"source_only":true}'
    )
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_nonfinite_json_number_is_rejected() -> None:
    semantic = (
        b'{"predictions":[{"bend_id":"x","line_end":[1,0,0],'
        b'"line_start":[0,0,0],"selection_route":"r","selector_score":NaN,'
        b'"source_kind":"plane_lattice"}],"research_only":true,"source_only":true}'
    )
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    with pytest.raises(ValueError, match="non-finite JSON number"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_present_selector_score_is_rejected(score: float) -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    semantic = _semantic([_prediction("base", selector_score=score)])
    with pytest.raises(ValueError, match="non-finite JSON number"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"source_only": False},
        {"research_only": False},
        {"labels_used": True},
        {"expected_count_used": True},
        {"cad_used": True},
        {"part_identity_used": True},
    ],
)
def test_semantic_source_only_contract_is_fail_closed(updates: dict) -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    semantic = _semantic([_prediction("base")], **updates)
    with pytest.raises(ValueError, match="source-only|forbidden input"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("manual_bends", []),
        ("expected_count", 2),
        ("filename", "case.ply"),
        ("part_id", "secret-part"),
    ],
)
def test_forbidden_payload_fields_are_rejected_even_when_usage_flags_are_false(
    field: str, value: object
) -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    semantic = _semantic([_prediction("base")], **{field: value})
    with pytest.raises(ValueError, match="forbidden fields"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_forbidden_prediction_truth_fields_are_rejected() -> None:
    row = _prediction("base", extra={"is_true_positive": True})
    with pytest.raises(ValueError, match="forbidden.*field"):
        _apply([row])


@pytest.mark.parametrize(
    "extra",
    [
        {"opaque": {"labels": [1]}},
        {"metadata": {"ground_truth": True}},
        {"audit": {"expected_count": 2}},
        {"nested": [{"cad": {"axis": [1, 0, 0]}}]},
        {"nested": {"drawing_count": 3}},
        {"opaque": {"groundTruth": True}},
        {"metadata": {"labelCount": 2}},
        {"nested": {"CADGeometry": {"axis": [1, 0, 0]}}},
        {"audit": {"drawingMetadata": {}}},
        {"opaque": {"manualBends": []}},
        {"metadata": {"partId": "secret"}},
        {"nested": {"isTruePositive": True}},
        {"metadata": {"caseId": "A30-secret"}},
        {"nested": {"runtimeCaseId": "T999"}},
    ],
)
def test_nested_forbidden_fields_cannot_be_copied_to_output(extra: dict) -> None:
    row = _prediction("base", extra=extra)
    with pytest.raises(ValueError, match="forbidden nested field"):
        _apply([row])


def test_unused_root_case_identity_may_exist_but_is_not_copied_or_used() -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    row = _prediction("base")
    semantic = _semantic(
        [row], case_id="T-internal", runtime_case_id="T-internal", scan_id="opaque"
    )
    result = runtime.apply_candidate4(
        semantic,
        model,
        gate_pass,
        terminal,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        train_gate_pass_sha256=gate_sha,
        candidate3_terminal_sha256=terminal_sha,
    )
    assert result["predictions"] == [row]
    assert all(key not in result for key in runtime.ROOT_UNUSED_IDENTITY_FIELDS)
    assert result["case_id_used"] is False
    assert result["runtime_id_used"] is False


def test_candidate3_terminal_is_the_exact_audited_lf_canonical_rejection() -> None:
    terminal = _terminal()
    payload = json.loads(terminal)
    assert runtime.canonical_bytes(payload) == terminal
    assert payload["kind"] == runtime.CANDIDATE3_TERMINAL_KIND
    assert payload["status"] == runtime.CANDIDATE3_TERMINAL_STATUS
    assert payload["candidate3_continuation_authorized"] is False
    assert payload["artifact_bindings"]["candidate3_custody_incident"] == {
        "path": runtime.CANDIDATE3_INCIDENT_PATH,
        "sha256": runtime.CANDIDATE3_INCIDENT_SHA256,
        "size_bytes": runtime.CANDIDATE3_INCIDENT_SIZE_BYTES,
    }


def test_candidate3_terminal_rejects_wrong_external_seal() -> None:
    with pytest.raises(ValueError, match="exact audited seal"):
        runtime._validate_candidate3_terminal(
            _terminal(), expected_sha256=_hash("wrong-candidate3-terminal")
        )


def test_candidate3_terminal_schema_and_incident_checks_are_fail_closed(
    monkeypatch,
) -> None:
    original = _terminal()
    payload = json.loads(original)
    payload["status"] = "PASS"
    tampered = runtime.canonical_bytes(payload)
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SHA256", runtime.sha256_bytes(tampered))
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SIZE_BYTES", len(tampered))
    with pytest.raises(ValueError, match="exact noncontinuing custody rejection"):
        runtime._validate_candidate3_terminal(
            tampered, expected_sha256=runtime.sha256_bytes(tampered)
        )

    payload = json.loads(original)
    payload["artifact_bindings"]["candidate3_custody_incident"]["sha256"] = _hash(
        "wrong-incident"
    )
    tampered = runtime.canonical_bytes(payload)
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SHA256", runtime.sha256_bytes(tampered))
    monkeypatch.setattr(runtime, "CANDIDATE3_TERMINAL_SIZE_BYTES", len(tampered))
    with pytest.raises(ValueError, match="exact custody incident"):
        runtime._validate_candidate3_terminal(
            tampered, expected_sha256=runtime.sha256_bytes(tampered)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_gate_passed", False),
        ("holdout_score_opened", True),
        ("pre_holdout_execution_anchor_created", True),
        ("post_holdout_model_or_threshold_mutation", "ALLOWED"),
        ("observed_source_multiplicity_used", False),
        ("expected_count_used", True),
        ("selector_score_used", True),
    ],
)
def test_model_status_and_feature_declarations_are_fail_closed(
    field: str, value: object
) -> None:
    terminal, terminal_sha, _bindings_value, gate_pass, gate_sha, model = _bundle()
    model[field] = value
    semantic = _semantic([])
    with pytest.raises(ValueError, match="model status|declaration"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_model_rejects_missing_extra_unknown_or_placeholder_bindings() -> None:
    terminal, terminal_sha, bindings, gate_pass, gate_sha, model = _bundle()
    semantic = _semantic([])

    missing = deepcopy(model)
    missing.pop("rule")
    with pytest.raises(ValueError, match="model schema"):
        runtime.apply_candidate4(
            semantic,
            missing,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )

    extra = deepcopy(model)
    extra["threshold"] = 2
    with pytest.raises(ValueError, match="model schema"):
        runtime.apply_candidate4(
            semantic,
            extra,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )

    placeholder_bindings = dict(bindings)
    placeholder_bindings["candidate4_preregistration_sha256"] = "0" * 64
    placeholder_pass = _pass_bytes(model["rule"], placeholder_bindings)
    placeholder_pass_sha = runtime.sha256_bytes(placeholder_pass)
    placeholder_model = _model(model["rule"], placeholder_bindings, placeholder_pass_sha)
    with pytest.raises(ValueError, match="zero commitment"):
        runtime.apply_candidate4(
            semantic,
            placeholder_model,
            placeholder_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=placeholder_pass_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_known_lineage_and_runtime_source_bindings_are_exact() -> None:
    terminal, terminal_sha, bindings, gate_pass, gate_sha, model = _bundle()
    semantic = _semantic([])
    for field in (*runtime.KNOWN_BINDINGS, "candidate4_runtime_source_sha256"):
        broken_bindings = dict(bindings)
        broken_bindings[field] = _hash(f"tampered-{field}")
        broken_pass = _pass_bytes(model["rule"], broken_bindings)
        broken_pass_sha = runtime.sha256_bytes(broken_pass)
        broken_model = _model(model["rule"], broken_bindings, broken_pass_sha)
        with pytest.raises(ValueError, match="immutable binding|runtime source"):
            runtime.apply_candidate4(
                semantic,
                broken_model,
                broken_pass,
                terminal,
                v51_semantic_sha256=runtime.sha256_bytes(semantic),
                train_gate_pass_sha256=broken_pass_sha,
                candidate3_terminal_sha256=terminal_sha,
            )


def test_train_pass_must_be_distinct_canonical_value_free_and_all_true() -> None:
    terminal, terminal_sha, bindings, gate_pass, gate_sha, model = _bundle()
    assert gate_sha not in set(bindings.values())
    semantic = _semantic([])

    failed_payload = json.loads(gate_pass)
    failed_payload["checks"][next(iter(runtime.GATE_CHECK_KEYS))] = False
    failed_pass = runtime.canonical_bytes(failed_payload)
    failed_sha = runtime.sha256_bytes(failed_pass)
    failed_model = _model(model["rule"], bindings, failed_sha)
    with pytest.raises(ValueError, match="failed gate"):
        runtime.apply_candidate4(
            semantic,
            failed_model,
            failed_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=failed_sha,
            candidate3_terminal_sha256=terminal_sha,
        )

    noncanonical = json.dumps(json.loads(gate_pass), indent=2).encode("utf-8")
    noncanonical_sha = runtime.sha256_bytes(noncanonical)
    noncanonical_model = _model(model["rule"], bindings, noncanonical_sha)
    with pytest.raises(ValueError, match="canonical JSON"):
        runtime.apply_candidate4(
            semantic,
            noncanonical_model,
            noncanonical,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=noncanonical_sha,
            candidate3_terminal_sha256=terminal_sha,
        )

    tampered = gate_pass + b" "
    with pytest.raises(ValueError, match="external SHA binding"):
        runtime.apply_candidate4(
            semantic,
            model,
            tampered,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_train_pass_hash_cannot_alias_any_other_bound_artifact(monkeypatch) -> None:
    terminal, terminal_sha, bindings, _gate_pass, _gate_sha, model = _bundle()
    aliased_sha = bindings["fit_audit_core_sha256"]
    model["train_gate_pass_binding"]["sha256"] = aliased_sha
    monkeypatch.setattr(runtime, "_validate_pass_artifact", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="must be distinct"):
        runtime._validate_model(
            model,
            train_gate_pass_bytes=b"not-opened",
            train_gate_pass_sha256=aliased_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_train_pass_rule_and_bindings_must_equal_the_model() -> None:
    terminal, terminal_sha, bindings, _gate_pass, _gate_sha, model = _bundle()
    semantic = _semantic([])
    other_rule = runtime.EXPLOSION_RULES[1]
    mismatched_pass = _pass_bytes(other_rule, bindings)
    mismatched_sha = runtime.sha256_bytes(mismatched_pass)
    model["train_gate_pass_binding"]["sha256"] = mismatched_sha
    with pytest.raises(ValueError, match="rule differs"):
        runtime.apply_candidate4(
            semantic,
            model,
            mismatched_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=mismatched_sha,
            candidate3_terminal_sha256=terminal_sha,
        )

    broken_bindings = dict(bindings)
    broken_bindings["candidate4_tests_source_sha256"] = _hash("different-tests")
    mismatched_pass = _pass_bytes(model["rule"], broken_bindings)
    mismatched_sha = runtime.sha256_bytes(mismatched_pass)
    model["train_gate_pass_binding"]["sha256"] = mismatched_sha
    with pytest.raises(ValueError, match="bindings differ"):
        runtime.apply_candidate4(
            semantic,
            model,
            mismatched_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=mismatched_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_unsupported_rule_is_rejected() -> None:
    terminal, terminal_sha, bindings, gate_pass, gate_sha, model = _bundle()
    model["rule"] = "KEEP_ONE_SIDECAR"
    semantic = _semantic([])
    with pytest.raises(ValueError, match="unsupported burst rule"):
        runtime.apply_candidate4(
            semantic,
            model,
            gate_pass,
            terminal,
            v51_semantic_sha256=runtime.sha256_bytes(semantic),
            train_gate_pass_sha256=gate_sha,
            candidate3_terminal_sha256=terminal_sha,
        )


def test_result_schema_and_hash_are_exact_and_truth_free() -> None:
    rows = [
        _prediction("base"),
        _prediction("s1", sidecar=True),
        _prediction("s2", sidecar=True),
    ]
    result, *_ = _apply(rows)
    expected_keys = {
        "schema_version",
        "kind",
        "profile",
        "attempt_ordinal",
        "input_v51_semantic_sha256",
        "candidate4_model_sha256",
        "train_gate_pass_sha256",
        "candidate3_terminal_sha256",
        "rule",
        "predictions",
        "candidate_count",
        "deletion_audit",
        "source_only",
        "research_only",
        "labels_used",
        "cad_used",
        "drawing_used",
        "part_identity_used",
        "case_id_used",
        "runtime_id_used",
        "filename_used",
        "source_kind_used",
        "selection_route_used",
        "selector_score_used",
        "geometry_used_for_deletion",
        "observed_source_multiplicity_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
        "deletion_only",
        "v51_fallback_default",
        "training_gate_passed",
        "holdout_score_opened",
        "post_holdout_model_or_threshold_mutation",
        "promotion_eligible",
        "candidate4_result_sha256",
    }
    assert set(result) == expected_keys
    result_without_hash = dict(result)
    observed_hash = result_without_hash.pop("candidate4_result_sha256")
    assert observed_hash == runtime.sha256_bytes(
        runtime.canonical_bytes(result_without_hash)
    )
    assert result["observed_source_multiplicity_used"] is True
    assert result["post_holdout_model_or_threshold_mutation"] == (
        "FORBIDDEN_AFTER_FIRST_HOLDOUT_SCORE_EVENT"
    )
    assert "count_feature_used" not in result
    for field in (
        "labels_used",
        "cad_used",
        "drawing_used",
        "part_identity_used",
        "case_id_used",
        "runtime_id_used",
        "filename_used",
        "expected_count_used",
        "sender_count_used",
        "gold_count_used",
        "label_count_used_at_inference",
        "holdout_score_opened",
        "promotion_eligible",
    ):
        assert result[field] is False


def test_runtime_has_no_training_scoring_or_label_module_imports() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = (
        "evaluate_f1_against_manual_gold",
        "optimize_f1_train18",
        "sklearn",
        "data.bend_corpus",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(str(node.module or ""))
    assert all(not name.startswith(forbidden) for name in imported)
