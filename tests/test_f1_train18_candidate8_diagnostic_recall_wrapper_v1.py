from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import f1_train18_candidate6_conservative_recall_recovery_v1 as hypothesis
from backend import f1_train18_candidate7_conservative_recall_recovery_v1 as candidate7
from backend import f1_train18_candidate8_diagnostic_recall_wrapper_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]


def _semantic() -> bytes:
    return hypothesis.canonical_bytes(
        {
            "cad_used": False,
            "candidate_count": 1,
            "expected_count_used": False,
            "labels_used": False,
            "part_identity_used": False,
            "predictions": [
                {
                    "bend_id": "V51-B0",
                    "line_end": [100.0, 0.0, 0.0],
                    "line_start": [0.0, 0.0, 0.0],
                    "selection_route": "v51-synthetic",
                    "selector_score": None,
                    "source_hypothesis_id": "V51-H0",
                    "source_kind": "plane_lattice",
                }
            ],
            "research_only": True,
            "source_only": True,
        }
    )


def _pool() -> bytes:
    return hypothesis.canonical_bytes(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_v3",
            "candidate_count": 99,
            "candidate_source_counts": {"ignored": 99},
            "candidates": [{"candidate_id": "RAW-IGNORED"}],
            "inference_contract": {
                "cad_exposed": False,
                "expected_count_exposed": False,
                "manual_truth_exposed": False,
                "part_identifier_exposed": False,
            },
            "line_object_count": 1,
            "line_objects": [
                {
                    "candidate_id": "IGNORED-ID",
                    "endpoint_support_fraction": 0.8,
                    "finite_support_fraction": 0.8,
                    "normal_second_eigenvalue": 0.2,
                    "representative_source": "plane_lattice",
                    "score": 0.8,
                    "span_endpoints": [[0.0, 8.0, 0.0], [100.0, 8.0, 0.0]],
                    "two_sided_transition_score": 0.8,
                }
            ],
            "schema_version": 3,
            "source_point_serialization": False,
            "working_point_budget": 200000,
        }
    )


def _disposition() -> bytes:
    return (ROOT / runtime.CANDIDATE7_DISPOSITION_PATH).read_bytes()


def _diagnose(
    *,
    semantic: bytes | None = None,
    pool: bytes | None = None,
    disposition: bytes | None = None,
    hypothesis_runtime=hypothesis,  # type: ignore[no-untyped-def]
    candidate7_runtime=candidate7,  # type: ignore[no-untyped-def]
) -> dict[str, object]:
    semantic_bytes = _semantic() if semantic is None else semantic
    pool_bytes = _pool() if pool is None else pool
    disposition_bytes = _disposition() if disposition is None else disposition
    return runtime.diagnose_source_bytes(
        semantic_bytes,
        pool_bytes,
        disposition_bytes,
        v51_semantic_sha256=runtime.sha256_bytes(semantic_bytes),
        pool_sha256=runtime.sha256_bytes(pool_bytes),
        candidate7_disposition_sha256=runtime.CANDIDATE7_DISPOSITION_SHA256,
        hypothesis_runtime=hypothesis_runtime,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        candidate7_runtime=candidate7_runtime,
        candidate7_runtime_source_sha256=runtime.CANDIDATE7_RUNTIME_SOURCE_SHA256,
    )


def test_runtime_is_stdlib_only_and_filesystem_blind() -> None:
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "pathlib", "scripts", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls.intersection({"open", "Path", "read_bytes", "write_bytes"})


def test_exact_hypothesis_and_phase_contract_are_frozen() -> None:
    runtime.verify_scientific_hypothesis_runtime(
        hypothesis, source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256
    )
    assert runtime.ATTEMPT_ORDINAL == 8
    assert runtime.PHASE_CONTRACT == (
        "D01_SEMANTIC_VALIDATION",
        "D02_POOL_PARSE_SANITIZE",
        "D03_OWNERSHIP_EXERCISE",
        "D04_FIXED_RULE_EXERCISE",
        "D05_WRAPPER_RESULT_VALIDATION",
    )
    assert runtime.PHASE_CONTRACT_SHA256 == runtime.sha256_bytes(
        runtime.canonical_bytes(list(runtime.PHASE_CONTRACT))
    )
    assert "LINEAGE" not in runtime.PHASE_CONTRACT
    assert set(runtime.FAILURE_CODES) == {"LINEAGE", *runtime.PHASE_CONTRACT}


def test_actual_shaped_source_envelopes_pass_all_diagnostic_phases() -> None:
    result = _diagnose()
    assert result["status"] == "SOURCE_COMPATIBLE"
    assert result["check_code"] == runtime.PASS_CODE
    assert result["failure_phase"] is None
    assert result["diagnostic_complete"] is True
    runtime._validate_result(result)


def test_lineage_failure_is_value_free() -> None:
    result = runtime.diagnose_source_bytes(
        _semantic(),
        _pool(),
        b"{}\n",
        v51_semantic_sha256=runtime.sha256_bytes(_semantic()),
        pool_sha256=runtime.sha256_bytes(_pool()),
        candidate7_disposition_sha256=runtime.CANDIDATE7_DISPOSITION_SHA256,
        hypothesis_runtime=hypothesis,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        candidate7_runtime=candidate7,
        candidate7_runtime_source_sha256=runtime.CANDIDATE7_RUNTIME_SOURCE_SHA256,
    )
    assert result["failure_phase"] == "LINEAGE"
    assert result["check_code"] == "C8_LINEAGE_VALIDATION_FAILED"
    assert result["lineage_gate_passed"] is False
    assert result["scientific_hypothesis_exactly_preserved"] is False
    assert result["candidate_rule_selected"] is False
    assert result["scientific_hypothesis_performance_tested"] is False


def test_semantic_validation_failure_has_only_fixed_code() -> None:
    value = json.loads(_semantic())
    value["predictions"][0]["ground_truth"] = "SECRET-VALUE"
    result = _diagnose(semantic=hypothesis.canonical_bytes(value))
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[0]
    assert result["check_code"] == "C8_SEMANTIC_VALIDATION_FAILED"
    assert "SECRET" not in json.dumps(result)


def test_pool_parse_sanitize_failure_has_only_fixed_code() -> None:
    value = json.loads(_pool())
    value["line_objects"][0]["outcome"] = "SECRET-VALUE"
    result = _diagnose(pool=hypothesis.canonical_bytes(value))
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[1]
    assert result["check_code"] == "C8_POOL_PARSE_SANITIZE_FAILED"
    assert "SECRET" not in json.dumps(result)


def test_ownership_failure_discards_exception_type_text_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise LookupError("SECRET-GEOMETRY-123")

    monkeypatch.setattr(hypothesis, "line_pair_metrics", fail)
    result = _diagnose()
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[2]
    assert result["check_code"] == "C8_OWNERSHIP_EXERCISE_FAILED"
    encoded = json.dumps(result)
    assert "LookupError" not in encoded and "SECRET" not in encoded


def test_runtime_threshold_match_without_conservative_conflict_is_d03(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = json.loads(_pool())
    pool["line_objects"][0]["span_endpoints"] = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ]
    monkeypatch.setattr(hypothesis, "_ownership_conflict", lambda *_args: False)
    result = _diagnose(pool=hypothesis.canonical_bytes(pool))
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[2]
    assert result["check_code"] == "C8_OWNERSHIP_EXERCISE_FAILED"


def test_rule_failure_discards_exception_type_text_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ArithmeticError("SECRET-RULE-0.75")

    monkeypatch.setattr(hypothesis, "validate_rule", fail)
    result = _diagnose()
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[3]
    assert result["check_code"] == "C8_FIXED_RULE_EXERCISE_FAILED"
    encoded = json.dumps(result)
    assert "ArithmeticError" not in encoded and "0.75" not in encoded


def test_numeric_rule_direction_or_missingness_failure_is_d04(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hypothesis, "_rule_matches", lambda *_args: False)
    result = _diagnose()
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[3]


def test_v51_prefix_mutation_during_rule_exercise_is_d04(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate_prefix(predictions, _candidates, _rule):  # type: ignore[no-untyped-def]
        predictions[0]["source_kind"] = "mutated-prefix"
        return None

    monkeypatch.setattr(hypothesis, "_select_addition", mutate_prefix)
    result = _diagnose()
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[3]


def test_wrapper_result_failure_is_separate_fixed_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        candidate7,
        "preflight_candidate7_source_bytes",
        lambda *_args, **_kwargs: {"status": "UNSAFE"},
    )
    result = _diagnose()
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[4]
    assert result["check_code"] == "C8_WRAPPER_RESULT_VALIDATION_FAILED"


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key",
        "missing_key",
        "wrong_kind",
        "wrong_schema",
        "wrong_semantic_binding",
        "wrong_pool_binding",
        "wrong_hypothesis_binding",
    ),
)
def test_d05_requires_exact_candidate7_aggregate_contract(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    semantic = _semantic()
    pool = _pool()
    upstream = candidate7.preflight_candidate7_source_bytes(
        semantic,
        pool,
        v51_semantic_sha256=runtime.sha256_bytes(semantic),
        pool_sha256=runtime.sha256_bytes(pool),
        hypothesis_runtime=hypothesis,
        hypothesis_source_sha256=runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
    )
    if mutation == "extra_key":
        upstream["unexpected"] = False
    elif mutation == "missing_key":
        upstream.pop("kind")
    elif mutation == "wrong_kind":
        upstream["kind"] = "wrong"
    elif mutation == "wrong_schema":
        upstream["schema_version"] = 2
    elif mutation == "wrong_semantic_binding":
        upstream["input_v51_semantic_sha256"] = "1" * 64
    elif mutation == "wrong_pool_binding":
        upstream["input_pool_sha256"] = "2" * 64
    elif mutation == "wrong_hypothesis_binding":
        upstream["scientific_hypothesis_source_sha256"] = "3" * 64
    else:  # pragma: no cover - parameter contract
        raise AssertionError(mutation)
    monkeypatch.setattr(
        candidate7,
        "preflight_candidate7_source_bytes",
        lambda *_args, **_kwargs: upstream,
    )
    result = _diagnose(semantic=semantic, pool=pool)
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[4]
    assert result["check_code"] == "C8_WRAPPER_RESULT_VALIDATION_FAILED"


def test_candidate7_reserved_namespace_failure_is_localized_to_d05() -> None:
    semantic = json.loads(_semantic())
    semantic["predictions"][0]["bend_id"] = "C7R-RESERVED-IN-SYNTHETIC"
    result = _diagnose(semantic=hypothesis.canonical_bytes(semantic))
    assert result["failure_phase"] == runtime.PHASE_CONTRACT[4]
    assert result["check_code"] == "C8_WRAPPER_RESULT_VALIDATION_FAILED"


def test_diagnostic_output_never_contains_counts_ids_values_or_exception_details() -> None:
    forbidden_key_fragments = (
        "bend_id",
        "candidate_count",
        "exception_class_name",
        "exception_message",
        "feature_value",
        "geometry",
        "line_object_count",
        "prediction_count",
        "predictions",
        "selector_score",
        "source_kind",
    )
    for result in (
        _diagnose(),
        runtime._result(failure_phase="LINEAGE"),
        runtime._result(failure_phase=runtime.PHASE_CONTRACT[0]),
        runtime._result(failure_phase=runtime.PHASE_CONTRACT[1]),
        runtime._result(failure_phase=runtime.PHASE_CONTRACT[2]),
        runtime._result(failure_phase=runtime.PHASE_CONTRACT[3]),
        runtime._result(failure_phase=runtime.PHASE_CONTRACT[4]),
    ):
        assert set(result) == runtime.RESULT_KEYS
        assert all(
            fragment not in key
            for key in result
            if not key.endswith("_exposed")
            for fragment in forbidden_key_fragments
        )
        assert result["exception_class_exposed"] is False
        assert result["exception_text_exposed"] is False
        assert result["identifier_or_geometry_exposed"] is False
        assert result["numeric_or_categorical_source_value_exposed"] is False
        assert result["candidate_rule_selected"] is False
        assert result["scientific_hypothesis_performance_tested"] is False


def test_ignored_ids_counts_and_list_order_do_not_change_diagnostic_result() -> None:
    first = _diagnose()
    semantic = json.loads(_semantic())
    semantic["candidate_count"] = 999999
    semantic["predictions"][0]["bend_id"] = "CHANGED-BEND-ID"
    semantic["predictions"][0]["source_hypothesis_id"] = "CHANGED-SOURCE-ID"
    pool = json.loads(_pool())
    pool["candidate_count"] = 888888
    pool["line_object_count"] = 777777
    pool["line_objects"][0]["candidate_id"] = "CHANGED-CANDIDATE-ID"
    pool["candidates"] = list(reversed(pool["candidates"]))
    second = _diagnose(
        semantic=hypothesis.canonical_bytes(semantic),
        pool=hypothesis.canonical_bytes(pool),
    )
    assert first == second


def test_mutated_hypothesis_contract_fails_lineage_before_source_phases() -> None:
    changed = SimpleNamespace(**hypothesis.__dict__)
    changed.MAX_ADDITIONS_PER_CASE = 2
    result = _diagnose(hypothesis_runtime=changed)
    assert result["failure_phase"] == "LINEAGE"
    assert result["lineage_gate_passed"] is False
    assert result["scientific_hypothesis_exactly_preserved"] is False
