from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import f1_train18_candidate10_full_replay_adapter_preflight_v1 as runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/f1_train18_candidate10_full_replay_adapter_preflight_v1.py"
MANIFEST_BINDING_SHA256 = runtime.sha256_bytes(b"synthetic-manifest-binding")

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
        "c6_hypothesis": runtime.C6_HYPOTHESIS_SHA256,
        "c7_runtime": runtime.C7_RUNTIME_SHA256,
        "c8_adapter": runtime.C8_ADAPTER_SHA256,
        "c8_runtime": runtime.C8_RUNTIME_SHA256,
        "c9_adapter": runtime.C9_ADAPTER_SHA256,
        "c9_adapter_tests": runtime.C9_ADAPTER_TESTS_SHA256,
        "c9_disposition": runtime.C9_DISPOSITION_SHA256,
        "cluster": runtime.CLUSTER_SHA256,
    }


class FakeAdapter:
    ATTEMPT_ORDINAL = 9
    SCIENTIFIC_HYPOTHESIS_ID = runtime.SCIENTIFIC_HYPOTHESIS_ID

    def __init__(self) -> None:
        self.calls = 0
        self.call_dependent = False
        self.false_projection = False
        self.bad_receipt = False
        self.raise_error: Exception | None = None
        self.malformed_receipt = False

    @staticmethod
    def _validate_context(context: object) -> None:
        if context != "verified-c9":
            raise PermissionError("bad synthetic context")

    def adapt_pool_bytes(
        self,
        context: object,
        *,
        pool_bytes: bytes,
        expected_pool_sha256: str,
    ) -> SimpleNamespace:
        self._validate_context(context)
        if self.raise_error is not None:
            raise self.raise_error
        self.calls += 1
        suffix = str(self.calls).encode() if self.call_dependent else b"stable"
        adapted = b"ADAPTED:" + pool_bytes + b":" + suffix
        adapted_sha = "0" * 64 if self.bad_receipt else runtime.sha256_bytes(adapted)
        receipt = SimpleNamespace(
            original_pool_sha256=expected_pool_sha256,
            adapted_pool_sha256=adapted_sha,
            adapted_pool_bytes=adapted,
            projection_byte_equivalent=not self.false_projection,
        )
        if self.malformed_receipt:
            del receipt.adapted_pool_bytes
        return receipt


class FakeDiagnostic:
    def __init__(self) -> None:
        self.calls = 0
        self.call_dependent = False
        self.code = "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
        self.phase: str | None = None
        self.raise_error: Exception | None = None
        self.malformed_result = False

    def diagnose_bound_pair(self, context: object, **kwargs: object) -> dict[str, object]:
        assert context == "verified-c8"
        if self.raise_error is not None:
            raise self.raise_error
        self.calls += 1
        failure = self.phase is not None
        value: dict[str, object] = {
            "attempt_ordinal": 8,
            "candidate_metric_computed": False,
            "candidate_rule_selected": False,
            "check_code": self.code,
            "diagnostic_complete": not failure,
            "engineering_attempt_id": "CANDIDATE8_DIAGNOSTIC_RECALL_WRAPPER_V1",
            "exception_class_exposed": False,
            "exception_text_exposed": False,
            "failure_phase": self.phase,
            "holdout_score_opened": False,
            "identifier_or_geometry_exposed": False,
            "kind": "sherman_qc_f1_aug30_train18_candidate8_diagnostic_result_v1",
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
            "status": "SOURCE_INCOMPATIBLE" if failure else "SOURCE_COMPATIBLE",
        }
        if self.call_dependent:
            value["phase_contract_sha256"] = f"{self.calls:064x}"
        if self.malformed_result:
            value["forbidden_extra"] = kwargs["pool_sha256"]
        return value


def _fake_c8_runtime(*, permissive: bool = False) -> SimpleNamespace:
    failure_codes = {
        "LINEAGE": "C8_LINEAGE_VALIDATION_FAILED",
        "D01_SEMANTIC_VALIDATION": "C8_SEMANTIC_VALIDATION_FAILED",
        "D02_POOL_PARSE_SANITIZE": "C8_POOL_PARSE_SANITIZE_FAILED",
        "D03_OWNERSHIP_EXERCISE": "C8_OWNERSHIP_EXERCISE_FAILED",
        "D04_FIXED_RULE_EXERCISE": "C8_FIXED_RULE_EXERCISE_FAILED",
        "D05_WRAPPER_RESULT_VALIDATION": "C8_WRAPPER_RESULT_VALIDATION_FAILED",
    }

    def validate(result: object) -> None:
        if not isinstance(result, dict) or set(result) != C8_RESULT_KEYS:
            raise ValueError("synthetic Candidate8 result schema mismatch")
        if permissive:
            return
        phase = result["failure_phase"]
        expected = (
            "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
            if phase is None
            else failure_codes[str(phase)]
        )
        if (
            result["check_code"] != expected
            or result["phase_contract_sha256"] != runtime.C8_PHASE_CONTRACT_SHA256
        ):
            raise ValueError("synthetic Candidate8 result contract mismatch")

    return SimpleNamespace(
        ATTEMPT_ORDINAL=8,
        ENGINEERING_ATTEMPT_ID="CANDIDATE8_DIAGNOSTIC_RECALL_WRAPPER_V1",
        PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
        PHASE_CONTRACT=(
            "D01_SEMANTIC_VALIDATION",
            "D02_POOL_PARSE_SANITIZE",
            "D03_OWNERSHIP_EXERCISE",
            "D04_FIXED_RULE_EXERCISE",
            "D05_WRAPPER_RESULT_VALIDATION",
        ),
        FAILURE_CODES=failure_codes,
        RESULT_KEYS=C8_RESULT_KEYS,
        SCIENTIFIC_HYPOTHESIS_ID=runtime.SCIENTIFIC_HYPOTHESIS_ID,
        SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256=runtime.C6_HYPOTHESIS_SHA256,
        RESULT_KIND="sherman_qc_f1_aug30_train18_candidate8_diagnostic_result_v1",
        PHASE_CONTRACT_SHA256=runtime.C8_PHASE_CONTRACT_SHA256,
        _validate_result=validate,
    )


def _context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: FakeAdapter | None = None,
    diagnostic: FakeDiagnostic | None = None,
    permissive_runtime: bool = False,
) -> tuple[runtime.PreflightContext, FakeAdapter, FakeDiagnostic]:
    c7 = b"synthetic-candidate7-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    adapter = adapter or FakeAdapter()
    diagnostic = diagnostic or FakeDiagnostic()
    context = runtime.make_verified_context(
        c9_adapter=adapter,
        c9_adapter_context="verified-c9",
        c8_adapter=diagnostic,
        c8_adapter_context="verified-c8",
        c8_runtime=_fake_c8_runtime(permissive=permissive_runtime),
        candidate7_disposition_bytes=c7,
        source_bindings=_source_bindings(),
    )
    return context, adapter, diagnostic


def _records() -> tuple[runtime.RecordBytes, ...]:
    records: list[runtime.RecordBytes] = []
    for pair in range(runtime.PAIR_COUNT):
        semantic = runtime.canonical_bytes({"semantic_fixture": pair})
        pool = runtime.canonical_bytes({"pool_fixture": pair})
        for pass_name in runtime.PASS_NAMES:
            ordinal = len(records)
            records.append(
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
    return tuple(records)


def _run(
    context: runtime.PreflightContext,
    records: tuple[runtime.RecordBytes, ...] | list[runtime.RecordBytes] | None = None,
    *,
    manifest_binding_sha256: str = MANIFEST_BINDING_SHA256,
) -> dict[str, object]:
    return runtime.preflight_all_records(
        context,
        _records() if records is None else records,
        manifest_binding_sha256=manifest_binding_sha256,
    )


def test_top_level_imports_are_stdlib_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {"__future__", "dataclasses", "hashlib", "json", "types", "typing"}
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed


def test_identity_and_fixed_inventory_contract() -> None:
    assert runtime.ATTEMPT_ORDINAL == 10
    assert runtime.RECORD_COUNT == 36
    assert runtime.PAIR_COUNT == 18
    assert runtime.SOURCE_FILE_COUNT == 72
    assert runtime.SCIENTIFIC_HYPOTHESIS_ID == "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"


def test_full_36_record_preflight_passes_and_calls_adapter_and_diagnostic_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    context, adapter, diagnostic = _context(monkeypatch)
    result = _run(context)
    assert adapter.calls == 36
    assert diagnostic.calls == 36
    assert set(result) == runtime.RESULT_KEYS
    assert result["status"] == "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
    assert result["check_code"] == runtime.PASS_CODE
    assert result["record_count"] == 36
    assert result["pair_count"] == 18
    assert result["source_file_count"] == 72
    assert result["all_d01_d05_diagnostics_passed"] is True
    assert result["projection_receipt_attestation_complete"] is True


def test_pass_result_is_aggregate_only_and_publishes_no_source_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    result = _run(context)
    encoded = json.dumps(result, sort_keys=True)
    for forbidden in (
        "semantic_fixture",
        "pool_fixture",
        "case_id",
        "runtime_case_id",
        "line_start",
        "line_end",
        "exception",
    ):
        assert forbidden not in encoded
    assert result["adapted_pool_bytes_published"] is False
    assert result["original_pool_bytes_published"] is False
    assert result["adapted_cache_published"] is False
    assert result["scientific_hypothesis_performance_tested"] is False
    assert result["final_or_model_rule_selected"] is False
    assert result["diagnostic_candidate_exercise_performed"] is True


def test_commitments_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    first_context, _adapter, _diagnostic = _context(monkeypatch)
    first = _run(first_context)
    second_context, _adapter, _diagnostic = _context(monkeypatch)
    second = _run(second_context)
    assert runtime.canonical_bytes(first) == runtime.canonical_bytes(second)


@pytest.mark.parametrize("size", [0, 35, 37])
def test_wrong_record_inventory_rejects(size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    records = _records()
    if size > len(records):
        records = (*records, replace(records[-1], ordinal=len(records), pass_name="primary"))
    else:
        records = records[:size]
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "INPUT_SCHEMA"


def test_wrong_ordinal_or_pass_order_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    records[4] = replace(records[4], ordinal=99)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "INPUT_SCHEMA"
    records = list(_records())
    records[1] = replace(records[1], pass_name="primary")
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "INPUT_SCHEMA"


def test_duplicate_or_permuted_pair_bindings_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    records[2] = replace(
        records[2],
        manifest_record_binding_sha256=records[0].manifest_record_binding_sha256,
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "INPUT_SCHEMA"

    records = list(_records())
    records[0], records[2] = (
        replace(records[2], ordinal=0, pair_ordinal=0),
        replace(records[0], ordinal=2, pair_ordinal=1),
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "REPLAY"


def test_manifest_or_file_binding_drift_changes_aggregate_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_context, _adapter, _diagnostic = _context(monkeypatch)
    first = _run(first_context)
    second_context, _adapter, _diagnostic = _context(monkeypatch)
    second = _run(
        second_context,
        manifest_binding_sha256=runtime.sha256_bytes(b"other manifest"),
    )
    assert first["aggregate_evidence_commitment_sha256"] != second[
        "aggregate_evidence_commitment_sha256"
    ]

    records = list(_records())
    records[0] = replace(
        records[0],
        semantic_file_binding_sha256=runtime.sha256_bytes(b"other file binding"),
    )
    third_context, _adapter, _diagnostic = _context(monkeypatch)
    third = _run(third_context, records)
    assert first["aggregate_evidence_commitment_sha256"] != third[
        "aggregate_evidence_commitment_sha256"
    ]


def test_semantic_hash_binding_rejects_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    context, adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    records[0] = replace(
        records[0], semantic_bytes=b"tampered", semantic_size_bytes=len(b"tampered")
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "SOURCE_BINDING"
    assert adapter.calls == 0


def test_pool_hash_binding_rejects_before_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    context, adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    records[0] = replace(
        records[0], pool_bytes=b"tampered", pool_size_bytes=len(b"tampered")
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "SOURCE_BINDING"
    assert adapter.calls == 0


def test_semantic_primary_replay_byte_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    changed = runtime.canonical_bytes({"semantic_fixture": "replay-drift"})
    records[1] = replace(
        records[1],
        semantic_bytes=changed,
        semantic_sha256=runtime.sha256_bytes(changed),
        semantic_size_bytes=len(changed),
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "REPLAY"


def test_original_pool_primary_replay_byte_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    records = list(_records())
    changed = runtime.canonical_bytes({"pool_fixture": "replay-drift"})
    records[1] = replace(
        records[1],
        pool_bytes=changed,
        pool_sha256=runtime.sha256_bytes(changed),
        pool_size_bytes=len(changed),
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, records)
    assert captured.value.stage == "REPLAY"


def test_independently_adapted_primary_replay_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter()
    adapter.call_dependent = True
    context, _adapter, _diagnostic = _context(monkeypatch, adapter=adapter)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "REPLAY"


def test_independent_diagnostic_primary_replay_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = FakeDiagnostic()
    diagnostic.call_dependent = True
    context, _adapter, _diagnostic = _context(
        monkeypatch, diagnostic=diagnostic, permissive_runtime=True
    )
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "REPLAY"


def test_false_projection_receipt_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter()
    adapter.false_projection = True
    context, _adapter, diagnostic = _context(monkeypatch, adapter=adapter)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "ADAPTER"
    assert diagnostic.calls == 0


def test_adapted_bytes_hash_receipt_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter()
    adapter.bad_receipt = True
    context, _adapter, diagnostic = _context(monkeypatch, adapter=adapter)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "ADAPTER"
    assert diagnostic.calls == 0


@pytest.mark.parametrize("malformed", [False, True])
def test_adapter_ordinary_exception_or_malformed_receipt_is_bounded(
    malformed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeAdapter()
    if malformed:
        adapter.malformed_receipt = True
    else:
        adapter.raise_error = RuntimeError("sensitive synthetic payload")
    context, _adapter, _diagnostic = _context(monkeypatch, adapter=adapter)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "ADAPTER"
    assert str(captured.value) == runtime.FAILURE_CODES["ADAPTER"]


@pytest.mark.parametrize("malformed", [False, True])
def test_diagnostic_ordinary_exception_or_malformed_result_is_bounded(
    malformed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = FakeDiagnostic()
    if malformed:
        diagnostic.malformed_result = True
    else:
        diagnostic.raise_error = RuntimeError("sensitive synthetic payload")
    context, _adapter, _diagnostic = _context(monkeypatch, diagnostic=diagnostic)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "DIAGNOSTIC"
    assert str(captured.value) == runtime.FAILURE_CODES["DIAGNOSTIC"]


@pytest.mark.parametrize(
    ("phase", "code"),
    [
        ("D01_SEMANTIC_VALIDATION", "C8_SEMANTIC_VALIDATION_FAILED"),
        ("D02_POOL_PARSE_SANITIZE", "C8_POOL_PARSE_SANITIZE_FAILED"),
        ("D03_OWNERSHIP_EXERCISE", "C8_OWNERSHIP_EXERCISE_FAILED"),
        ("D04_FIXED_RULE_EXERCISE", "C8_FIXED_RULE_EXERCISE_FAILED"),
        ("D05_WRAPPER_RESULT_VALIDATION", "C8_WRAPPER_RESULT_VALIDATION_FAILED"),
    ],
)
def test_any_d01_to_d05_failure_rejects_full_pass(
    phase: str, code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = FakeDiagnostic()
    diagnostic.code = code
    diagnostic.phase = phase
    context, _adapter, _diagnostic = _context(monkeypatch, diagnostic=diagnostic)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "DIAGNOSTIC"


def test_diagnostic_code_phase_mismatch_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic = FakeDiagnostic()
    diagnostic.code = "C8_POOL_PARSE_SANITIZE_FAILED"
    diagnostic.phase = "D05_WRAPPER_RESULT_VALIDATION"
    context, _adapter, _diagnostic = _context(monkeypatch, diagnostic=diagnostic)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "DIAGNOSTIC"


def test_context_rejects_wrong_source_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    c7 = b"synthetic-candidate7-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    bindings = _source_bindings()
    bindings["c9_disposition"] = "0" * 64
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime.make_verified_context(
            c9_adapter=FakeAdapter(),
            c9_adapter_context="verified-c9",
            c8_adapter=FakeDiagnostic(),
            c8_adapter_context="verified-c8",
            c8_runtime=_fake_c8_runtime(),
            candidate7_disposition_bytes=c7,
            source_bindings=bindings,
        )
    assert captured.value.stage == "CONTEXT"


def test_context_rejects_candidate7_disposition_byte_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(b"expected"))
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime.make_verified_context(
            c9_adapter=FakeAdapter(),
            c9_adapter_context="verified-c9",
            c8_adapter=FakeDiagnostic(),
            c8_adapter_context="verified-c8",
            c8_runtime=_fake_c8_runtime(),
            candidate7_disposition_bytes=b"wrong",
            source_bindings=_source_bindings(),
        )
    assert captured.value.stage == "CONTEXT"


@pytest.mark.parametrize(
    ("attribute", "wrong"),
    [
        ("SCIENTIFIC_HYPOTHESIS_ID", "WRONG_HYPOTHESIS"),
        ("SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256", "2" * 64),
        ("RESULT_KIND", "wrong_result_kind"),
        ("ENGINEERING_ATTEMPT_ID", "wrong_attempt"),
        ("PHASE_CONTRACT_SHA256", "3" * 64),
    ],
)
def test_context_rejects_wrong_candidate8_symbolic_contract(
    attribute: str, wrong: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    c7 = b"synthetic-candidate7-disposition"
    monkeypatch.setattr(runtime, "C7_DISPOSITION_SHA256", runtime.sha256_bytes(c7))
    c8_runtime = _fake_c8_runtime()
    setattr(c8_runtime, attribute, wrong)
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime.make_verified_context(
            c9_adapter=FakeAdapter(),
            c9_adapter_context="verified-c9",
            c8_adapter=FakeDiagnostic(),
            c8_adapter_context="verified-c8",
            c8_runtime=c8_runtime,
            candidate7_disposition_bytes=c7,
            source_bindings=_source_bindings(),
        )
    assert captured.value.stage == "CONTEXT"


def test_context_rejects_post_construction_dependency_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, adapter, _diagnostic = _context(monkeypatch)
    adapter.ATTEMPT_ORDINAL = 99
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context)
    assert captured.value.stage == "CONTEXT"


def test_invalid_manifest_binding_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    with pytest.raises(runtime.PreflightFailure) as captured:
        _run(context, manifest_binding_sha256="bad")
    assert captured.value.stage == "INPUT_SCHEMA"


@pytest.mark.parametrize("stage", sorted(runtime.FAILURE_CODES))
def test_terminal_payload_is_fixed_value_free_and_nonretryable(stage: str) -> None:
    error = runtime.PreflightFailure(stage=stage, code=runtime.FAILURE_CODES[stage])
    value = runtime.terminal_payload(error)
    assert set(value) == runtime.TERMINAL_KEYS
    assert value["failure_stage"] == stage
    assert value["check_code"] == runtime.FAILURE_CODES[stage]
    assert value["retry_authorized"] is False
    assert value["labels_opened"] is False
    assert value["scientific_hypothesis_performance_tested"] is False
    assert "exception" not in json.dumps(value, sort_keys=True)


def test_public_validator_rejects_count_or_boolean_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    context, _adapter, _diagnostic = _context(monkeypatch)
    result = _run(context)
    changed = dict(result)
    changed["record_count"] = 35
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime._validate_public_result(changed)
    assert captured.value.stage == "AGGREGATE"
    changed = dict(result)
    changed["schema_version"] = True
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime._validate_public_result(changed)
    assert captured.value.stage == "AGGREGATE"
    changed = dict(result)
    changed["aggregate_evidence_commitment_sha256"] = "0" * 64
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime._validate_public_result(changed)
    assert captured.value.stage == "AGGREGATE"
    changed = dict(result)
    changed["labels_opened"] = 0
    with pytest.raises(runtime.PreflightFailure) as captured:
        runtime._validate_public_result(changed)
    assert captured.value.stage == "AGGREGATE"
