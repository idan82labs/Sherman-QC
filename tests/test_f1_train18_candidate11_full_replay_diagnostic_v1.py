from __future__ import annotations

import ast
from dataclasses import replace
import json
import pickle
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from backend import f1_train18_candidate11_full_replay_diagnostic_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate11_full_replay_diagnostic_v1.py"
MANIFEST_SHA = runtime.sha256_bytes(b"synthetic-c11-manifest")

C8_RESULT_KEYS = frozenset(
    {
        "attempt_ordinal",
        "candidate_metric_computed",
        "candidate_rule_selected",
        "check_code",
        "diagnostic_complete",
        "engineering_attempt_id",
        "exception_class_exposed",
        "exception_text_exposed",
        "failure_phase",
        "holdout_score_opened",
        "identifier_or_geometry_exposed",
        "kind",
        "labels_used",
        "lineage_gate_passed",
        "model_fit_performed",
        "numeric_or_categorical_source_value_exposed",
        "phase_contract_sha256",
        "schema_version",
        "scientific_hypothesis_exactly_preserved",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "scientific_hypothesis_source_sha256",
        "source_only",
        "status",
    }
)


def _source_bindings() -> dict[str, str]:
    return {
        "builder": runtime.BUILDER_SHA256,
        "c10_core": runtime.C10_CORE_SHA256,
        "c10_core_tests": runtime.C10_CORE_TESTS_SHA256,
        "c10_runner": runtime.C10_RUNNER_SHA256,
        "c10_runner_tests": runtime.C10_RUNNER_TESTS_SHA256,
        "c6_hypothesis": runtime.C6_HYPOTHESIS_SHA256,
        "c7_runtime": runtime.C7_RUNTIME_SHA256,
        "c8_adapter": runtime.C8_ADAPTER_SHA256,
        "c8_runtime": runtime.C8_RUNTIME_SHA256,
        "c9_adapter": runtime.C9_ADAPTER_SHA256,
        "c9_adapter_tests": runtime.C9_ADAPTER_TESTS_SHA256,
        "cluster": runtime.CLUSTER_SHA256,
    }


def _c8_context_source_bindings() -> dict[str, dict[str, str]]:
    return {
        "adapter": {"sha256": runtime.C8_ADAPTER_SHA256},
        "adapter_tests": {"sha256": runtime.C8_ADAPTER_TESTS_SHA256},
        "candidate7_runtime": {"sha256": runtime.C7_RUNTIME_SHA256},
        "hypothesis": {"sha256": runtime.C6_HYPOTHESIS_SHA256},
        "runtime": {"sha256": runtime.C8_RUNTIME_SHA256},
        "runtime_tests": {"sha256": runtime.C8_RUNTIME_TESTS_SHA256},
    }


class Controls:
    def __init__(self) -> None:
        self.adapter_calls = 0
        self.diagnostic_calls = 0
        self.adapter_fail_call: int | None = None
        self.adapter_variant_call: int | None = None
        self.adapter_bytes_subclass_call: int | None = None
        self.diagnostic_fail_call: int | None = None
        self.diagnostic_phase_call: int | None = None
        self.diagnostic_phase: str | None = None


def _c8_result(*, phase: str | None) -> dict[str, object]:
    code = runtime.C8_PASS_CODE if phase is None else runtime.C8_FAILURE_CODES[phase]
    return {
        "attempt_ordinal": 8,
        "candidate_metric_computed": False,
        "candidate_rule_selected": False,
        "check_code": code,
        "diagnostic_complete": phase is None,
        "engineering_attempt_id": runtime.C8_ENGINEERING_ATTEMPT_ID,
        "exception_class_exposed": False,
        "exception_text_exposed": False,
        "failure_phase": phase,
        "holdout_score_opened": False,
        "identifier_or_geometry_exposed": False,
        "kind": runtime.C8_RESULT_KIND,
        "labels_used": False,
        "lineage_gate_passed": True,
        "model_fit_performed": False,
        "numeric_or_categorical_source_value_exposed": False,
        "phase_contract_sha256": runtime.C8_PHASE_CONTRACT_SHA256,
        "schema_version": 1,
        "scientific_hypothesis_exactly_preserved": True,
        "scientific_hypothesis_id": runtime.SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "scientific_hypothesis_source_sha256": runtime.C6_HYPOTHESIS_SHA256,
        "source_only": True,
        "status": "SOURCE_COMPATIBLE" if phase is None else "SOURCE_INCOMPATIBLE",
    }


def _modules(controls: Controls) -> tuple[ModuleType, ModuleType, ModuleType]:
    c9 = ModuleType("synthetic_c9")
    c9.ATTEMPT_ORDINAL = 9
    c9.ENGINEERING_ATTEMPT_ID = "CANDIDATE9_LOSSLESS_POOL_ALIAS_ADAPTER_V1"
    c9.ADAPTER_KIND = (
        "sherman_qc_f1_aug30_train18_candidate9_lossless_pool_alias_adapter_v1"
    )
    c9.SCIENTIFIC_HYPOTHESIS_ID = runtime.SCIENTIFIC_HYPOTHESIS_ID

    def validate_c9_context(context: object) -> None:
        if context != "verified-c9":
            raise PermissionError("synthetic context rejection")

    def adapt_pool_bytes(
        context: object, *, pool_bytes: bytes, expected_pool_sha256: str
    ) -> SimpleNamespace:
        validate_c9_context(context)
        controls.adapter_calls += 1
        if controls.adapter_calls == controls.adapter_fail_call:
            raise ValueError("private adapter detail")
        suffix = (
            b":variant"
            if controls.adapter_calls == controls.adapter_variant_call
            else b":stable"
        )
        adapted = b"ADAPTED:" + pool_bytes + suffix
        if controls.adapter_calls == controls.adapter_bytes_subclass_call:
            adapted = ExplosiveBytes(adapted)
        return SimpleNamespace(
            original_pool_sha256=expected_pool_sha256,
            adapted_pool_sha256=runtime.sha256_bytes(adapted),
            adapted_pool_bytes=adapted,
            projection_byte_equivalent=True,
        )

    c9._validate_context = validate_c9_context
    c9.adapt_pool_bytes = adapt_pool_bytes

    c8_adapter = ModuleType("synthetic_c8_adapter")
    c8_adapter.AdapterContext = SimpleNamespace

    def diagnose_bound_pair(context: object, **kwargs: object) -> dict[str, object]:
        if (
            getattr(context, "runtime", None) is not c8
            or not isinstance(kwargs["pool_bytes"], bytes)
        ):
            raise PermissionError("synthetic diagnostic context rejection")
        controls.diagnostic_calls += 1
        if controls.diagnostic_calls == controls.diagnostic_fail_call:
            raise RuntimeError("private diagnostic detail")
        phase = (
            controls.diagnostic_phase
            if controls.diagnostic_calls == controls.diagnostic_phase_call
            else None
        )
        return _c8_result(phase=phase)

    c8_adapter.diagnose_bound_pair = diagnose_bound_pair

    c8 = ModuleType("synthetic_c8_runtime")
    c8.ATTEMPT_ORDINAL = 8
    c8.ENGINEERING_ATTEMPT_ID = runtime.C8_ENGINEERING_ATTEMPT_ID
    c8.SCIENTIFIC_HYPOTHESIS_ID = runtime.SCIENTIFIC_HYPOTHESIS_ID
    c8.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256 = runtime.C6_HYPOTHESIS_SHA256
    c8.RESULT_KIND = runtime.C8_RESULT_KIND
    c8.PASS_CODE = runtime.C8_PASS_CODE
    c8.PHASE_CONTRACT = runtime.C8_PHASE_CONTRACT
    c8.FAILURE_CODES = {
        "LINEAGE": "C8_LINEAGE_VALIDATION_FAILED",
        **runtime.C8_FAILURE_CODES,
    }
    c8.RESULT_KEYS = C8_RESULT_KEYS
    c8.PHASE_CONTRACT_SHA256 = runtime.C8_PHASE_CONTRACT_SHA256

    def validate_result(value: object) -> None:
        if not isinstance(value, dict) or set(value) != C8_RESULT_KEYS:
            raise ValueError("synthetic C8 schema rejection")
        phase = value["failure_phase"]
        expected = (
            runtime.C8_PASS_CODE
            if phase is None
            else runtime.C8_FAILURE_CODES.get(str(phase))
        )
        if (
            expected is None
            or value["check_code"] != expected
            or value["phase_contract_sha256"] != runtime.C8_PHASE_CONTRACT_SHA256
        ):
            raise ValueError("synthetic C8 contract rejection")

    c8._validate_result = validate_result
    return c9, c8_adapter, c8


class ExplosiveBytes(bytes):
    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError("private replay comparison detail")


class ExplosiveStr(str):
    def __hash__(self) -> int:
        raise RuntimeError("private binding detail")


def _context(
    monkeypatch: pytest.MonkeyPatch, controls: Controls | None = None
) -> tuple[runtime.DiagnosticContext, Controls]:
    controls = controls or Controls()
    c7 = b"synthetic-candidate7-disposition"
    c10 = b"synthetic-candidate10-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    monkeypatch.setattr(runtime, "C10_DISPOSITION_SHA256", runtime.sha256_bytes(c10))
    c9, c8_adapter, c8 = _modules(controls)
    c8_context = SimpleNamespace(
        runtime=c8,
        hypothesis=object(),
        candidate7_runtime=object(),
        source_bindings=_c8_context_source_bindings(),
    )
    return (
        runtime.make_verified_context(
            c9_adapter=c9,
            c9_adapter_context="verified-c9",
            c8_adapter=c8_adapter,
            c8_adapter_context=c8_context,
            c8_runtime=c8,
            candidate7_disposition_bytes=c7,
            candidate10_disposition_bytes=c10,
            source_bindings=_source_bindings(),
        ),
        controls,
    )


def _records() -> tuple[runtime.RecordBytes, ...]:
    values: list[runtime.RecordBytes] = []
    for pair in range(runtime.PAIR_COUNT):
        semantic = runtime.canonical_bytes({"semantic_fixture": pair})
        pool = runtime.canonical_bytes({"pool_fixture": pair})
        for pass_name in runtime.PASS_NAMES:
            ordinal = len(values)
            values.append(
                runtime.RecordBytes(
                    ordinal=ordinal,
                    pair_ordinal=pair,
                    pass_name=pass_name,
                    manifest_record_binding_sha256=runtime.sha256_bytes(
                        runtime.canonical_bytes(
                            {"domain": "manifest-record", "ordinal": ordinal}
                        )
                    ),
                    semantic_file_binding_sha256=runtime.sha256_bytes(
                        runtime.canonical_bytes(
                            {"domain": "semantic-file", "ordinal": ordinal}
                        )
                    ),
                    pool_file_binding_sha256=runtime.sha256_bytes(
                        runtime.canonical_bytes(
                            {"domain": "pool-file", "ordinal": ordinal}
                        )
                    ),
                    semantic_sha256=runtime.sha256_bytes(semantic),
                    pool_sha256=runtime.sha256_bytes(pool),
                    semantic_size_bytes=len(semantic),
                    pool_size_bytes=len(pool),
                    semantic_bytes=semantic,
                    pool_bytes=pool,
                )
            )
    return tuple(values)


def _run(
    context: runtime.DiagnosticContext,
    records: tuple[runtime.RecordBytes, ...] | list[runtime.RecordBytes] | None = None,
) -> dict[str, object]:
    return runtime.diagnose_all_records(
        context,
        _records() if records is None else records,
        manifest_binding_sha256=MANIFEST_SHA,
    )


def test_source_is_stdlib_only_and_identity_is_distinct() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {"__future__", "dataclasses", "hashlib", "json", "types", "typing"}
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed
    assert runtime.ATTEMPT_ORDINAL == 11
    assert runtime.ENGINEERING_ATTEMPT_ID == "CANDIDATE11_FULL_TRAIN_REPLAY_DIAGNOSTIC_V1"
    assert runtime.STAGE_CONTRACT == (
        "CONTEXT_SCHEMA_BINDING",
        "ADAPTER_PROJECTION",
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "REPLAY_SEMANTIC_BYTES",
        "REPLAY_ORIGINAL_POOL_BYTES",
        "REPLAY_ADAPTED_POOL_BYTES",
        "REPLAY_DIAGNOSTIC_BYTES",
        "AGGREGATE_RESULT",
    )


def test_exact_36_receipt_success_and_primary_replay_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, controls = _context(monkeypatch)
    result = _run(context)
    assert controls.adapter_calls == 36
    assert controls.diagnostic_calls == 36
    assert set(result) == runtime.RESULT_KEYS
    assert result["check_code"] == runtime.PASS_CODE
    assert result["all_records_compatible"] is True
    assert result["primary_replay_identity"] is True
    assert result["diagnostic_complete"] is True
    assert result["stage_reached_all_bitset"] == "1" * len(runtime.STAGE_CONTRACT)
    assert result["stage_passed_all_bitset"] == "1" * len(runtime.STAGE_CONTRACT)


@pytest.mark.parametrize("size", [0, 35, 37])
def test_inventory_must_be_exactly_36(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _ = _context(monkeypatch)
    records = list(_records())
    if size == 37:
        records.append(records[-1])
    else:
        records = records[:size]
    with pytest.raises(runtime.DiagnosticFailure):
        _run(context, records)


def test_context_schema_binding_failure_continues_other_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, controls = _context(monkeypatch)
    records = list(_records())
    records[7] = replace(records[7], semantic_size_bytes=1)
    result = _run(context, records)
    assert result["check_code"] == runtime.FAIL_CODE
    assert controls.adapter_calls == 35
    assert controls.diagnostic_calls == 35
    assert result["stage_reached_all_bitset"] == "100000000001"
    assert result["stage_passed_all_bitset"] == "000000000001"


def test_adapter_projection_exception_is_caught_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = Controls()
    controls.adapter_fail_call = 6
    context, controls = _context(monkeypatch, controls)
    result = _run(context)
    assert result["check_code"] == runtime.FAIL_CODE
    assert controls.adapter_calls == 36
    assert controls.diagnostic_calls == 35
    assert result["stage_reached_all_bitset"] == "110000011001"
    assert result["stage_passed_all_bitset"] == "100000011001"


@pytest.mark.parametrize("phase", runtime.C8_PHASE_CONTRACT)
def test_every_d01_d05_failure_is_aggregated_and_later_records_continue(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls()
    controls.diagnostic_phase_call = 9
    controls.diagnostic_phase = phase
    context, controls = _context(monkeypatch, controls)
    result = _run(context)
    assert result["check_code"] == runtime.FAIL_CODE
    assert result["all_records_compatible"] is False
    assert controls.adapter_calls == 36
    assert controls.diagnostic_calls == 36
    phase_index = runtime.C8_PHASE_CONTRACT.index(phase)
    reached = result["stage_reached_all_bitset"]
    passed = result["stage_passed_all_bitset"]
    assert reached == "11" + "1" * (phase_index + 1) + "0" * (
        4 - phase_index
    ) + "11111"
    assert passed == "11" + "1" * phase_index + "0" * (
        5 - phase_index
    ) + "11101"


def test_ordinary_diagnostic_exception_is_caught_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = Controls()
    controls.diagnostic_fail_call = 4
    context, controls = _context(monkeypatch, controls)
    result = _run(context)
    assert result["check_code"] == runtime.FAIL_CODE
    assert controls.diagnostic_calls == 36
    assert result["stage_reached_all_bitset"] == "110000011101"
    assert result["stage_passed_all_bitset"] == "110000011101"


def test_hostile_bytes_subclasses_are_rejected_without_exception_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, controls = _context(monkeypatch)
    records = list(_records())
    records[1] = replace(
        records[1], semantic_bytes=ExplosiveBytes(records[1].semantic_bytes)
    )
    result = _run(context, records)
    assert result["check_code"] == runtime.FAIL_CODE
    assert controls.adapter_calls == 35
    assert "private replay comparison detail" not in json.dumps(result)

    subclass_controls = Controls()
    subclass_controls.adapter_bytes_subclass_call = 2
    subclass_context, subclass_controls = _context(monkeypatch, subclass_controls)
    subclass_result = _run(subclass_context)
    assert subclass_result["check_code"] == runtime.FAIL_CODE
    assert subclass_controls.adapter_calls == 36
    assert subclass_controls.diagnostic_calls == 35


def test_hostile_binding_string_subclass_is_contained_per_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, controls = _context(monkeypatch)
    records = list(_records())
    records[4] = replace(
        records[4],
        manifest_record_binding_sha256=ExplosiveStr(
            records[4].manifest_record_binding_sha256
        ),
    )
    result = _run(context, records)
    assert result["check_code"] == runtime.FAIL_CODE
    assert controls.adapter_calls == 35
    assert "private binding detail" not in json.dumps(result)

    with pytest.raises(runtime.DiagnosticFailure) as captured:
        runtime.diagnose_all_records(
            context,
            _records(),
            manifest_binding_sha256=ExplosiveStr(MANIFEST_SHA),
        )
    assert str(captured.value) == "C11_DIAGNOSTIC_INVOCATION_REJECTED"


@pytest.mark.parametrize("kind", ["semantic", "original_pool"])
def test_replay_source_byte_identity_failures(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, controls = _context(monkeypatch)
    records = list(_records())
    if kind == "semantic":
        data = b"different replay semantic"
        records[1] = replace(
            records[1],
            semantic_bytes=data,
            semantic_size_bytes=len(data),
            semantic_sha256=runtime.sha256_bytes(data),
        )
    else:
        data = b"different replay pool"
        records[1] = replace(
            records[1],
            pool_bytes=data,
            pool_size_bytes=len(data),
            pool_sha256=runtime.sha256_bytes(data),
        )
    result = _run(context, records)
    assert result["check_code"] == runtime.FAIL_CODE
    assert result["primary_replay_identity"] is False
    assert controls.diagnostic_calls == 36
    assert result["stage_reached_all_bitset"] == "1" * 12
    assert result["stage_passed_all_bitset"] == (
        "111111101111" if kind == "semantic" else "111111110011"
    )


def test_replay_adapted_pool_byte_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = Controls()
    controls.adapter_variant_call = 2
    context, controls = _context(monkeypatch, controls)
    result = _run(context)
    assert result["check_code"] == runtime.FAIL_CODE
    assert result["primary_replay_identity"] is False
    assert controls.diagnostic_calls == 36
    assert result["stage_reached_all_bitset"] == "1" * 12
    assert result["stage_passed_all_bitset"] == "111111111011"


def test_replay_diagnostic_byte_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = Controls()
    controls.diagnostic_phase_call = 2
    controls.diagnostic_phase = runtime.C8_PHASE_CONTRACT[-1]
    context, controls = _context(monkeypatch, controls)
    result = _run(context)
    assert result["check_code"] == runtime.FAIL_CODE
    assert result["primary_replay_identity"] is False
    assert controls.diagnostic_calls == 36
    assert result["stage_reached_all_bitset"] == "1" * 12
    assert result["stage_passed_all_bitset"] == "111111011101"


def test_aggregate_result_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _context(monkeypatch)
    result = _run(context)
    result["labels_opened"] = True
    with pytest.raises(runtime.DiagnosticFailure):
        runtime._validate_public_result(result)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("primary_replay_identity", False),
        ("stage_passed_all_bitset", "0" + "1" * 11),
        ("stage_reached_all_bitset", "1" * 11 + "0"),
        ("external_source_and_manifest_authentication_required", False),
    ],
)
def test_public_validator_rejects_inconsistent_identity_and_stage_projection(
    key: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _context(monkeypatch)
    result = _run(context)
    result[key] = value
    with pytest.raises(runtime.DiagnosticFailure):
        runtime._validate_public_result(result)


def test_public_result_is_fixed_shape_and_discloses_no_record_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = Controls()
    controls.adapter_fail_call = 3
    context, _ = _context(monkeypatch, controls)
    result = _run(context)
    encoded = json.dumps(result, sort_keys=True)
    for forbidden in (
        "semantic_fixture",
        "pool_fixture",
        "manifest_record_binding",
        "case_id",
        "pair_ordinal",
        '"ordinal"',
        "private adapter detail",
        "ValueError",
        "reached\": [",
        "passed\": [",
    ):
        assert forbidden not in encoded
    assert result["labels_opened"] is False
    assert result["scientific_hypothesis_performance_tested"] is False
    assert result["candidate_metric_computed"] is False
    assert result["fit_or_model_performed"] is False
    assert result["holdout_opened"] is False
    assert result["evaluator_opened"] is False
    assert result["identifier_or_ordinal_exposed"] is False
    assert result["exception_text_exposed"] is False
    assert result["exception_class_exposed"] is False
    assert result["external_source_and_manifest_authentication_required"] is True
    assert result["filesystem_or_artifact_discovery_capability_present"] is False
    assert len(result["stage_reached_all_bitset"]) == len(runtime.STAGE_CONTRACT)
    assert len(result["stage_passed_all_bitset"]) == len(runtime.STAGE_CONTRACT)


def test_commitments_are_deterministic_but_change_on_hidden_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_context, _ = _context(monkeypatch)
    first = _run(first_context)
    second_context, _ = _context(monkeypatch)
    second = _run(second_context)
    assert runtime.canonical_bytes(first) == runtime.canonical_bytes(second)

    controls = Controls()
    controls.adapter_fail_call = 10
    third_context, _ = _context(monkeypatch, controls)
    third = _run(third_context)
    assert third["aggregate_reached_passed_bitsets_commitment_sha256"] != first[
        "aggregate_reached_passed_bitsets_commitment_sha256"
    ]


def test_sealed_receipt_vector_refuses_serialization_and_representation() -> None:
    sealed = runtime._SealedReceiptVector(((('ordinal', 1),),))
    assert repr(sealed) == "<_SealedReceiptVector>"
    with pytest.raises(TypeError):
        pickle.dumps(sealed)
    with pytest.raises(TypeError):
        json.dumps(sealed)


def test_malformed_or_nonmodule_dependencies_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c7 = b"synthetic-candidate7-disposition"
    c10 = b"synthetic-candidate10-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    monkeypatch.setattr(runtime, "C10_DISPOSITION_SHA256", runtime.sha256_bytes(c10))
    controls = Controls()
    c9, c8_adapter, c8 = _modules(controls)
    c8_context = SimpleNamespace(
        runtime=c8,
        hypothesis=object(),
        candidate7_runtime=object(),
        source_bindings=_c8_context_source_bindings(),
    )
    with pytest.raises(runtime.DiagnosticFailure):
        runtime.make_verified_context(
            c9_adapter=SimpleNamespace(**vars(c9)),
            c9_adapter_context="verified-c9",
            c8_adapter=c8_adapter,
            c8_adapter_context=c8_context,
            c8_runtime=c8,
            candidate7_disposition_bytes=c7,
            candidate10_disposition_bytes=c10,
            source_bindings=_source_bindings(),
        )


def test_candidate8_context_and_candidate10_lineage_are_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c7 = b"synthetic-candidate7-disposition"
    c10 = b"synthetic-candidate10-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    monkeypatch.setattr(runtime, "C10_DISPOSITION_SHA256", runtime.sha256_bytes(c10))
    c9, c8_adapter, c8 = _modules(Controls())
    c8_context = SimpleNamespace(
        runtime=c8,
        hypothesis=object(),
        candidate7_runtime=object(),
        source_bindings=_c8_context_source_bindings(),
    )
    with pytest.raises(runtime.DiagnosticFailure):
        runtime.make_verified_context(
            c9_adapter=c9,
            c9_adapter_context="verified-c9",
            c8_adapter=c8_adapter,
            c8_adapter_context=SimpleNamespace(
                runtime=object(),
                hypothesis=object(),
                candidate7_runtime=object(),
                source_bindings=_c8_context_source_bindings(),
            ),
            c8_runtime=c8,
            candidate7_disposition_bytes=c7,
            candidate10_disposition_bytes=c10,
            source_bindings=_source_bindings(),
        )
    with pytest.raises(runtime.DiagnosticFailure):
        runtime.make_verified_context(
            c9_adapter=c9,
            c9_adapter_context="verified-c9",
            c8_adapter=c8_adapter,
            c8_adapter_context=c8_context,
            c8_runtime=c8,
            candidate7_disposition_bytes=c7,
            candidate10_disposition_bytes=b"wrong-candidate10-disposition",
            source_bindings=_source_bindings(),
        )
    del c8_adapter.diagnose_bound_pair
    with pytest.raises(runtime.DiagnosticFailure):
        runtime.make_verified_context(
            c9_adapter=c9,
            c9_adapter_context="verified-c9",
            c8_adapter=c8_adapter,
            c8_adapter_context=c8_context,
            c8_runtime=c8,
            candidate7_disposition_bytes=c7,
            candidate10_disposition_bytes=c10,
            source_bindings=_source_bindings(),
        )


def test_callable_drift_after_context_creation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _context(monkeypatch)

    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("must not run")

    context.c9_adapter.adapt_pool_bytes = replacement
    with pytest.raises(runtime.DiagnosticFailure):
        _run(context)
