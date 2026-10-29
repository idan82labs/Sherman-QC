"""Candidate10 label-incapable full TRAIN18 adapter/replay preflight core.

This module is filesystem blind.  A guarded outer runner must authenticate and
read the exact 36 manifest records (18 primary/replay pairs) before passing
immutable bytes here.  Candidate10 applies the already-cleared Candidate9
lossless representation projection independently to every pool, invokes the
unchanged Candidate8 D01--D05 diagnostic independently for every record, and
requires exact primary/replay identity for source bytes, adapted bytes, and
diagnostic results.  It computes no labels, metrics, fitted model, or holdout
score and publishes no source or adapted bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import ModuleType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 10
ENGINEERING_ATTEMPT_ID = "CANDIDATE10_FULL_TRAIN_ADAPTER_REPLAY_PREFLIGHT_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate10_full_adapter_replay_preflight_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate10_full_adapter_replay_preflight_terminal_v1"

RECORD_COUNT = 36
PAIR_COUNT = 18
SOURCE_FILE_COUNT = 72
PASS_NAMES = ("primary", "replay")

C9_ADAPTER_SHA256 = "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096"
C9_ADAPTER_TESTS_SHA256 = "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623"
C8_RUNTIME_SHA256 = "d878eb9141f287435890ed3290a45f40dd0730130585a6846c38431e3342f9b8"
C8_ADAPTER_SHA256 = "b91a112356dba3ed4aa9e3fe51fe36ea12bf93d2dd9ab8900f2b3461509211be"
C7_RUNTIME_SHA256 = "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d"
C6_HYPOTHESIS_SHA256 = "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
BUILDER_SHA256 = "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2"
CLUSTER_SHA256 = "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c"
C9_DISPOSITION_SHA256 = "66ccbadbe49d5563355d407f8c319e665a20dd937b3506c74402fc2c97165ea2"
C7_DISPOSITION_SHA256 = "1a64d3148d018b224a3a44d9d36d2620f1d87f18063e32df14050210aa03f56b"

C8_ENGINEERING_ATTEMPT_ID = "CANDIDATE8_DIAGNOSTIC_RECALL_WRAPPER_V1"
C8_RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate8_diagnostic_result_v1"
C8_PHASE_CONTRACT = (
    "D01_SEMANTIC_VALIDATION",
    "D02_POOL_PARSE_SANITIZE",
    "D03_OWNERSHIP_EXERCISE",
    "D04_FIXED_RULE_EXERCISE",
    "D05_WRAPPER_RESULT_VALIDATION",
)
C8_PHASE_CONTRACT_SHA256 = (
    "192bd96ca4f944ceb9b2eb1dc9308e0d9f73bb09a83705579d3883995c49d94a"
)

PASS_CODE = "C10_ALL_RECORDS_ADAPTER_DIAGNOSTICS_REPLAY_PASSED"
FAILURE_CODES = {
    "CONTEXT": "C10_VERIFIED_CONTEXT_REJECTED",
    "INPUT_SCHEMA": "C10_RECORD_INPUT_SCHEMA_REJECTED",
    "SOURCE_BINDING": "C10_SOURCE_BYTE_BINDING_REJECTED",
    "ADAPTER": "C10_LOSSLESS_ADAPTER_REJECTED",
    "DIAGNOSTIC": "C10_D01_D05_DIAGNOSTIC_REJECTED",
    "REPLAY": "C10_PRIMARY_REPLAY_IDENTITY_REJECTED",
    "AGGREGATE": "C10_AGGREGATE_RESULT_REJECTED",
}

RESULT_KEYS = frozenset(
    {
        "aggregate_evidence_commitment_sha256",
        "adapted_cache_published",
        "adapted_pool_bytes_published",
        "adapted_pool_primary_replay_byte_identity",
        "all_d01_d05_diagnostics_passed",
        "attempt_ordinal",
        "candidate_metric_computed",
        "check_code",
        "cluster_provenance_accounting_is_not_a_feature",
        "diagnostic_candidate_exercise_performed",
        "diagnostic_primary_replay_byte_identity",
        "engineering_attempt_id",
        "evaluator_opened",
        "expected_sender_gold_label_count_features_used",
        "fit_or_model_performed",
        "final_or_model_rule_selected",
        "full28_opened",
        "holdout_opened",
        "input_binding_commitment_sha256",
        "kind",
        "labels_opened",
        "manifest_binding_sha256",
        "manifest_record_count",
        "original_pool_bytes_published",
        "original_pool_primary_replay_byte_identity",
        "pair_count",
        "projection_receipt_attestation_commitment_sha256",
        "projection_receipt_attestation_complete",
        "promotion_authorized",
        "record_count",
        "replay_diagnostic_commitment_sha256",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "semantic_primary_replay_byte_identity",
        "source_lineage_commitment_sha256",
        "source_file_count",
        "source_only",
        "status",
    }
)

TERMINAL_KEYS = frozenset(
    {
        "adapted_cache_published",
        "adapted_pool_bytes_published",
        "attempt_ordinal",
        "candidate_metric_computed",
        "check_code",
        "engineering_attempt_id",
        "evaluator_opened",
        "failure_stage",
        "fit_or_model_performed",
        "final_or_model_rule_selected",
        "full28_opened",
        "holdout_opened",
        "kind",
        "labels_opened",
        "original_pool_bytes_published",
        "promotion_authorized",
        "retry_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_only",
        "status",
    }
)

_CONTEXT_SENTINEL = object()


class PreflightFailure(RuntimeError):
    def __init__(self, *, stage: str, code: str) -> None:
        if stage not in FAILURE_CODES or code != FAILURE_CODES[stage]:
            raise ValueError("Candidate10 failure is outside the frozen allowlist")
        super().__init__(code)
        self.stage = stage
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Candidate10 hashes immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


@dataclass(frozen=True)
class RecordBytes:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    manifest_record_binding_sha256: str
    semantic_file_binding_sha256: str
    pool_file_binding_sha256: str
    semantic_sha256: str
    pool_sha256: str
    semantic_size_bytes: int
    pool_size_bytes: int
    semantic_bytes: bytes
    pool_bytes: bytes


@dataclass(frozen=True)
class PreflightContext:
    c9_adapter: ModuleType
    c9_adapter_context: Any
    c8_adapter: ModuleType
    c8_adapter_context: Any
    c8_runtime: ModuleType
    candidate7_disposition_bytes: bytes
    source_bindings: tuple[tuple[str, str], ...]
    source_lineage_commitment_sha256: str
    _callable_snapshot: tuple[tuple[Any, Any], ...]
    _c8_contract_snapshot: tuple[Any, ...]
    _sentinel: object


def _callable_identity(value: Any) -> tuple[Any, Any]:
    if not callable(value):
        raise TypeError("Candidate10 controlled dependency must be callable")
    return (getattr(value, "__self__", None), getattr(value, "__func__", value))


def _source_lineage_commitment(
    bindings: tuple[tuple[str, str], ...]
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "attempt_ordinal": ATTEMPT_ORDINAL,
                "domain": "CANDIDATE10_SOURCE_LINEAGE_V1",
                "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
                "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
                "source_bindings": [
                    {"name": name, "sha256": sha256} for name, sha256 in bindings
                ],
            }
        )
    )


def _dependency_callable_snapshot(
    *, c9_adapter: Any, c8_adapter: Any, c8_runtime: Any
) -> tuple[tuple[Any, Any], ...]:
    return (
        _callable_identity(getattr(c9_adapter, "_validate_context")),
        _callable_identity(getattr(c9_adapter, "adapt_pool_bytes")),
        _callable_identity(getattr(c8_adapter, "diagnose_bound_pair")),
        _callable_identity(getattr(c8_runtime, "_validate_result")),
    )


def _c8_contract_snapshot(c8_runtime: Any) -> tuple[Any, ...]:
    try:
        return (
            c8_runtime.ATTEMPT_ORDINAL,
            c8_runtime.PASS_CODE,
            tuple(c8_runtime.PHASE_CONTRACT),
            tuple(sorted(dict(c8_runtime.FAILURE_CODES).items())),
            frozenset(c8_runtime.RESULT_KEYS),
            c8_runtime.SCIENTIFIC_HYPOTHESIS_ID,
            c8_runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
            c8_runtime.RESULT_KIND,
            c8_runtime.ENGINEERING_ATTEMPT_ID,
            c8_runtime.PHASE_CONTRACT_SHA256,
        )
    except (AttributeError, TypeError, ValueError):
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"]) from None


def make_verified_context(
    *,
    c9_adapter: ModuleType,
    c9_adapter_context: Any,
    c8_adapter: ModuleType,
    c8_adapter_context: Any,
    c8_runtime: ModuleType,
    candidate7_disposition_bytes: bytes,
    source_bindings: Mapping[str, str],
) -> PreflightContext:
    expected = {
        "builder": BUILDER_SHA256,
        "c6_hypothesis": C6_HYPOTHESIS_SHA256,
        "c7_runtime": C7_RUNTIME_SHA256,
        "c8_adapter": C8_ADAPTER_SHA256,
        "c8_runtime": C8_RUNTIME_SHA256,
        "c9_adapter": C9_ADAPTER_SHA256,
        "c9_adapter_tests": C9_ADAPTER_TESTS_SHA256,
        "c9_disposition": C9_DISPOSITION_SHA256,
        "cluster": CLUSTER_SHA256,
    }
    try:
        normalized = {
            str(name): _require_sha(value, name=f"Candidate10 source {name}")
            for name, value in source_bindings.items()
        }
    except (AttributeError, TypeError, ValueError):
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"]) from None
    if normalized != expected:
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"])
    try:
        if (
            getattr(c9_adapter, "ATTEMPT_ORDINAL", None) != 9
            or getattr(c9_adapter, "SCIENTIFIC_HYPOTHESIS_ID", None)
            != SCIENTIFIC_HYPOTHESIS_ID
            or getattr(c8_runtime, "ATTEMPT_ORDINAL", None) != 8
            or getattr(c8_runtime, "PASS_CODE", None)
            != "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
            or not isinstance(candidate7_disposition_bytes, bytes)
            or sha256_bytes(candidate7_disposition_bytes) != C7_DISPOSITION_SHA256
        ):
            raise ValueError("Candidate10 controlled dependency identity mismatch")
        callable_snapshot = _dependency_callable_snapshot(
            c9_adapter=c9_adapter,
            c8_adapter=c8_adapter,
            c8_runtime=c8_runtime,
        )
        contract_snapshot = _c8_contract_snapshot(c8_runtime)
        if (
            contract_snapshot[2]
            != C8_PHASE_CONTRACT
            or set(dict(contract_snapshot[3]))
            != {*contract_snapshot[2], "LINEAGE"}
            or contract_snapshot[5] != SCIENTIFIC_HYPOTHESIS_ID
            or contract_snapshot[6] != C6_HYPOTHESIS_SHA256
            or contract_snapshot[7] != C8_RESULT_KIND
            or contract_snapshot[8] != C8_ENGINEERING_ATTEMPT_ID
            or contract_snapshot[9] != C8_PHASE_CONTRACT_SHA256
        ):
            raise ValueError("Candidate10 Candidate8 contract mismatch")
        c9_adapter._validate_context(c9_adapter_context)
    except PreflightFailure:
        raise
    except Exception:
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"]) from None
    frozen_bindings = tuple(sorted(normalized.items()))
    return PreflightContext(
        c9_adapter=c9_adapter,
        c9_adapter_context=c9_adapter_context,
        c8_adapter=c8_adapter,
        c8_adapter_context=c8_adapter_context,
        c8_runtime=c8_runtime,
        candidate7_disposition_bytes=candidate7_disposition_bytes,
        source_bindings=frozen_bindings,
        source_lineage_commitment_sha256=_source_lineage_commitment(frozen_bindings),
        _callable_snapshot=callable_snapshot,
        _c8_contract_snapshot=contract_snapshot,
        _sentinel=_CONTEXT_SENTINEL,
    )


def _validate_context(context: PreflightContext) -> None:
    try:
        expected_bindings = tuple(
            sorted(
                {
                    "builder": BUILDER_SHA256,
                    "c6_hypothesis": C6_HYPOTHESIS_SHA256,
                    "c7_runtime": C7_RUNTIME_SHA256,
                    "c8_adapter": C8_ADAPTER_SHA256,
                    "c8_runtime": C8_RUNTIME_SHA256,
                    "c9_adapter": C9_ADAPTER_SHA256,
                    "c9_adapter_tests": C9_ADAPTER_TESTS_SHA256,
                    "c9_disposition": C9_DISPOSITION_SHA256,
                    "cluster": CLUSTER_SHA256,
                }.items()
            )
        )
        if (
            not isinstance(context, PreflightContext)
            or context._sentinel is not _CONTEXT_SENTINEL
            or context.source_bindings != expected_bindings
            or context.source_lineage_commitment_sha256
            != _source_lineage_commitment(expected_bindings)
            or sha256_bytes(context.candidate7_disposition_bytes)
            != C7_DISPOSITION_SHA256
            or context._callable_snapshot
            != _dependency_callable_snapshot(
                c9_adapter=context.c9_adapter,
                c8_adapter=context.c8_adapter,
                c8_runtime=context.c8_runtime,
            )
            or context._c8_contract_snapshot
            != _c8_contract_snapshot(context.c8_runtime)
            or getattr(context.c9_adapter, "ATTEMPT_ORDINAL", None) != 9
            or getattr(context.c9_adapter, "SCIENTIFIC_HYPOTHESIS_ID", None)
            != SCIENTIFIC_HYPOTHESIS_ID
        ):
            raise ValueError("Candidate10 verified context changed")
        context.c9_adapter._validate_context(context.c9_adapter_context)
    except PreflightFailure:
        raise
    except Exception:
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"]) from None


def _validate_records(records: Sequence[RecordBytes]) -> tuple[RecordBytes, ...]:
    if (
        not isinstance(records, (tuple, list))
        or len(records) != RECORD_COUNT
        or any(type(record) is not RecordBytes for record in records)
    ):
        raise PreflightFailure(
            stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
        )
    normalized = tuple(records)
    manifest_bindings: set[str] = set()
    file_bindings: set[str] = set()
    for ordinal, record in enumerate(normalized):
        if (
            isinstance(record.ordinal, bool)
            or record.ordinal != ordinal
            or isinstance(record.pair_ordinal, bool)
            or record.pair_ordinal != ordinal // 2
            or record.pass_name != PASS_NAMES[ordinal % 2]
            or not isinstance(record.semantic_bytes, bytes)
            or not isinstance(record.pool_bytes, bytes)
            or not record.semantic_bytes
            or not record.pool_bytes
            or isinstance(record.semantic_size_bytes, bool)
            or not isinstance(record.semantic_size_bytes, int)
            or record.semantic_size_bytes != len(record.semantic_bytes)
            or isinstance(record.pool_size_bytes, bool)
            or not isinstance(record.pool_size_bytes, int)
            or record.pool_size_bytes != len(record.pool_bytes)
        ):
            raise PreflightFailure(
                stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
            )
        try:
            for name, value in (
                ("manifest record binding", record.manifest_record_binding_sha256),
                ("semantic file binding", record.semantic_file_binding_sha256),
                ("pool file binding", record.pool_file_binding_sha256),
                ("semantic SHA", record.semantic_sha256),
                ("pool SHA", record.pool_sha256),
            ):
                _require_sha(value, name=name)
        except ValueError:
            raise PreflightFailure(
                stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
            ) from None
        if (
            record.manifest_record_binding_sha256 in manifest_bindings
            or record.semantic_file_binding_sha256 in file_bindings
            or record.pool_file_binding_sha256 in file_bindings
            or record.semantic_file_binding_sha256 == record.pool_file_binding_sha256
        ):
            raise PreflightFailure(
                stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
            )
        manifest_bindings.add(record.manifest_record_binding_sha256)
        file_bindings.update(
            (record.semantic_file_binding_sha256, record.pool_file_binding_sha256)
        )
    if len(manifest_bindings) != RECORD_COUNT or len(file_bindings) != SOURCE_FILE_COUNT:
        raise PreflightFailure(
            stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
        )
    return normalized


def _diagnostic_code_phase(context: PreflightContext) -> dict[str, str | None]:
    try:
        mapping = {
            context.c8_runtime.PASS_CODE: None,
            **{
                context.c8_runtime.FAILURE_CODES[phase]: phase
                for phase in context.c8_runtime.PHASE_CONTRACT
            },
        }
    except (AttributeError, KeyError, TypeError):
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"]) from None
    if set(mapping) != {
        "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
        "C8_SEMANTIC_VALIDATION_FAILED",
        "C8_POOL_PARSE_SANITIZE_FAILED",
        "C8_OWNERSHIP_EXERCISE_FAILED",
        "C8_FIXED_RULE_EXERCISE_FAILED",
        "C8_WRAPPER_RESULT_VALIDATION_FAILED",
    }:
        raise PreflightFailure(stage="CONTEXT", code=FAILURE_CODES["CONTEXT"])
    return mapping


def preflight_all_records(
    context: PreflightContext,
    records: Sequence[RecordBytes],
    *,
    manifest_binding_sha256: str,
) -> dict[str, Any]:
    _validate_context(context)
    try:
        manifest_binding_sha256 = _require_sha(
            manifest_binding_sha256, name="Candidate10 manifest binding"
        )
    except ValueError:
        raise PreflightFailure(
            stage="INPUT_SCHEMA", code=FAILURE_CODES["INPUT_SCHEMA"]
        ) from None
    bound = _validate_records(records)
    code_phase = _diagnostic_code_phase(context)
    input_receipts: list[dict[str, Any]] = []
    projection_receipts: list[dict[str, Any]] = []
    diagnostic_receipts: list[dict[str, Any]] = []
    outputs: list[tuple[bytes, bytes, bytes, bytes]] = []

    for record in bound:
        if (
            sha256_bytes(record.semantic_bytes) != record.semantic_sha256
            or sha256_bytes(record.pool_bytes) != record.pool_sha256
        ):
            raise PreflightFailure(
                stage="SOURCE_BINDING", code=FAILURE_CODES["SOURCE_BINDING"]
            )
        input_receipts.append(
            {
                "manifest_record_binding_sha256": record.manifest_record_binding_sha256,
                "ordinal": record.ordinal,
                "pair_ordinal": record.pair_ordinal,
                "pass": record.pass_name,
                "pool_file_binding_sha256": record.pool_file_binding_sha256,
                "pool_sha256": record.pool_sha256,
                "pool_size_bytes": record.pool_size_bytes,
                "semantic_file_binding_sha256": record.semantic_file_binding_sha256,
                "semantic_sha256": record.semantic_sha256,
                "semantic_size_bytes": record.semantic_size_bytes,
            }
        )
        try:
            adapted = context.c9_adapter.adapt_pool_bytes(
                context.c9_adapter_context,
                pool_bytes=record.pool_bytes,
                expected_pool_sha256=record.pool_sha256,
            )
            if (
                adapted.original_pool_sha256 != record.pool_sha256
                or adapted.projection_byte_equivalent is not True
                or not isinstance(adapted.adapted_pool_bytes, bytes)
                or not adapted.adapted_pool_bytes
                or sha256_bytes(adapted.adapted_pool_bytes)
                != _require_sha(
                    adapted.adapted_pool_sha256,
                    name="Candidate10 adapted pool SHA",
                )
            ):
                raise ValueError("Candidate10 lossless adapter receipt mismatch")
        except PreflightFailure:
            raise
        except Exception:
            raise PreflightFailure(
                stage="ADAPTER", code=FAILURE_CODES["ADAPTER"]
            ) from None
        projection_receipts.append(
            {
                "adapted_pool_sha256": adapted.adapted_pool_sha256,
                "ordinal": record.ordinal,
                "original_pool_sha256": adapted.original_pool_sha256,
                "pass": record.pass_name,
                "projection_byte_equivalent": True,
            }
        )
        try:
            diagnostic = context.c8_adapter.diagnose_bound_pair(
                context.c8_adapter_context,
                semantic_bytes=record.semantic_bytes,
                pool_bytes=adapted.adapted_pool_bytes,
                candidate7_disposition_bytes=context.candidate7_disposition_bytes,
                semantic_sha256=record.semantic_sha256,
                pool_sha256=adapted.adapted_pool_sha256,
                candidate7_disposition_sha256=C7_DISPOSITION_SHA256,
            )
            context.c8_runtime._validate_result(diagnostic)
            if not isinstance(diagnostic, Mapping):
                raise TypeError("Candidate10 diagnostic result must be a mapping")
            code = diagnostic.get("check_code")
            phase = diagnostic.get("failure_phase")
            if (
                set(diagnostic) != context._c8_contract_snapshot[4]
                or code not in code_phase
                or phase != code_phase[code]
                or code != context.c8_runtime.PASS_CODE
            ):
                raise ValueError("Candidate10 diagnostic result did not pass")
            diagnostic_bytes = canonical_bytes(dict(diagnostic))
        except PreflightFailure:
            raise
        except Exception:
            raise PreflightFailure(
                stage="DIAGNOSTIC", code=FAILURE_CODES["DIAGNOSTIC"]
            ) from None
        diagnostic_receipts.append(
            {
                "diagnostic_sha256": sha256_bytes(diagnostic_bytes),
                "ordinal": record.ordinal,
                "pass": record.pass_name,
            }
        )
        outputs.append(
            (
                record.semantic_bytes,
                record.pool_bytes,
                adapted.adapted_pool_bytes,
                diagnostic_bytes,
            )
        )

    for pair_ordinal in range(PAIR_COUNT):
        primary = bound[2 * pair_ordinal]
        replay = bound[2 * pair_ordinal + 1]
        primary_outputs = outputs[2 * pair_ordinal]
        replay_outputs = outputs[2 * pair_ordinal + 1]
        if (
            primary.semantic_sha256 != replay.semantic_sha256
            or primary.pool_sha256 != replay.pool_sha256
            or primary_outputs != replay_outputs
        ):
            raise PreflightFailure(stage="REPLAY", code=FAILURE_CODES["REPLAY"])

    input_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE10_BOUND_INPUT_RECEIPTS_V1",
                "manifest_binding_sha256": manifest_binding_sha256,
                "receipts": input_receipts,
                "source_lineage_commitment_sha256": context.source_lineage_commitment_sha256,
            }
        )
    )
    projection_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE10_LOSSLESS_PROJECTION_RECEIPTS_V1",
                "input_binding_commitment_sha256": input_commitment,
                "receipts": projection_receipts,
            }
        )
    )
    diagnostic_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE10_D01_D05_REPLAY_RECEIPTS_V1",
                "projection_receipt_attestation_commitment_sha256": projection_commitment,
                "receipts": diagnostic_receipts,
            }
        )
    )
    aggregate_commitment = sha256_bytes(
        canonical_bytes(
            {
                "diagnostic_commitment_sha256": diagnostic_commitment,
                "domain": "CANDIDATE10_AGGREGATE_EVIDENCE_V1",
                "input_commitment_sha256": input_commitment,
                "manifest_binding_sha256": manifest_binding_sha256,
                "projection_commitment_sha256": projection_commitment,
                "source_lineage_commitment_sha256": context.source_lineage_commitment_sha256,
            }
        )
    )
    result = {
        "aggregate_evidence_commitment_sha256": aggregate_commitment,
        "adapted_cache_published": False,
        "adapted_pool_bytes_published": False,
        "adapted_pool_primary_replay_byte_identity": True,
        "all_d01_d05_diagnostics_passed": True,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate_metric_computed": False,
        "check_code": PASS_CODE,
        "cluster_provenance_accounting_is_not_a_feature": True,
        "diagnostic_candidate_exercise_performed": True,
        "diagnostic_primary_replay_byte_identity": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "expected_sender_gold_label_count_features_used": False,
        "fit_or_model_performed": False,
        "final_or_model_rule_selected": False,
        "full28_opened": False,
        "holdout_opened": False,
        "input_binding_commitment_sha256": input_commitment,
        "kind": RESULT_KIND,
        "labels_opened": False,
        "manifest_binding_sha256": manifest_binding_sha256,
        "manifest_record_count": RECORD_COUNT,
        "original_pool_bytes_published": False,
        "original_pool_primary_replay_byte_identity": True,
        "pair_count": PAIR_COUNT,
        "projection_receipt_attestation_commitment_sha256": projection_commitment,
        "projection_receipt_attestation_complete": True,
        "promotion_authorized": False,
        "record_count": RECORD_COUNT,
        "replay_diagnostic_commitment_sha256": diagnostic_commitment,
        "schema_version": SCHEMA_VERSION,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "semantic_primary_replay_byte_identity": True,
        "source_lineage_commitment_sha256": context.source_lineage_commitment_sha256,
        "source_file_count": SOURCE_FILE_COUNT,
        "source_only": True,
        "status": "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING",
    }
    _validate_public_result(result)
    return result


def _validate_public_result(value: Mapping[str, Any]) -> None:
    if set(value) != RESULT_KEYS:
        raise PreflightFailure(stage="AGGREGATE", code=FAILURE_CODES["AGGREGATE"])
    required_true = {
        "adapted_pool_primary_replay_byte_identity",
        "all_d01_d05_diagnostics_passed",
        "cluster_provenance_accounting_is_not_a_feature",
        "diagnostic_candidate_exercise_performed",
        "diagnostic_primary_replay_byte_identity",
        "original_pool_primary_replay_byte_identity",
        "projection_receipt_attestation_complete",
        "semantic_primary_replay_byte_identity",
        "source_only",
    }
    required_false = {
        "adapted_cache_published",
        "adapted_pool_bytes_published",
        "candidate_metric_computed",
        "evaluator_opened",
        "expected_sender_gold_label_count_features_used",
        "fit_or_model_performed",
        "final_or_model_rule_selected",
        "full28_opened",
        "holdout_opened",
        "labels_opened",
        "original_pool_bytes_published",
        "promotion_authorized",
        "scientific_hypothesis_performance_tested",
    }
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind") != RESULT_KIND
        or type(value.get("attempt_ordinal")) is not int
        or value.get("attempt_ordinal") != 10
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("status") != "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
        or value.get("check_code") != PASS_CODE
        or type(value.get("record_count")) is not int
        or value.get("record_count") != RECORD_COUNT
        or type(value.get("manifest_record_count")) is not int
        or value.get("manifest_record_count") != RECORD_COUNT
        or type(value.get("pair_count")) is not int
        or value.get("pair_count") != PAIR_COUNT
        or type(value.get("source_file_count")) is not int
        or value.get("source_file_count") != SOURCE_FILE_COUNT
        or any(value.get(key) is not True for key in required_true)
        or any(value.get(key) is not False for key in required_false)
    ):
        raise PreflightFailure(stage="AGGREGATE", code=FAILURE_CODES["AGGREGATE"])
    for key in (
        "aggregate_evidence_commitment_sha256",
        "input_binding_commitment_sha256",
        "manifest_binding_sha256",
        "projection_receipt_attestation_commitment_sha256",
        "replay_diagnostic_commitment_sha256",
        "source_lineage_commitment_sha256",
    ):
        try:
            _require_sha(value.get(key), name=key)
        except ValueError:
            raise PreflightFailure(
                stage="AGGREGATE", code=FAILURE_CODES["AGGREGATE"]
            ) from None
    expected_aggregate = sha256_bytes(
        canonical_bytes(
            {
                "diagnostic_commitment_sha256": value[
                    "replay_diagnostic_commitment_sha256"
                ],
                "domain": "CANDIDATE10_AGGREGATE_EVIDENCE_V1",
                "input_commitment_sha256": value["input_binding_commitment_sha256"],
                "manifest_binding_sha256": value["manifest_binding_sha256"],
                "projection_commitment_sha256": value[
                    "projection_receipt_attestation_commitment_sha256"
                ],
                "source_lineage_commitment_sha256": value[
                    "source_lineage_commitment_sha256"
                ],
            }
        )
    )
    if value["aggregate_evidence_commitment_sha256"] != expected_aggregate:
        raise PreflightFailure(stage="AGGREGATE", code=FAILURE_CODES["AGGREGATE"])


def terminal_payload(error: PreflightFailure) -> dict[str, Any]:
    value = {
        "adapted_cache_published": False,
        "adapted_pool_bytes_published": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate_metric_computed": False,
        "check_code": error.code,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "failure_stage": error.stage,
        "fit_or_model_performed": False,
        "final_or_model_rule_selected": False,
        "full28_opened": False,
        "holdout_opened": False,
        "kind": TERMINAL_KIND,
        "labels_opened": False,
        "original_pool_bytes_published": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "schema_version": SCHEMA_VERSION,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_only": True,
        "status": "TERMINAL_ENGINEERING_FAILURE",
    }
    if (
        set(value) != TERMINAL_KEYS
        or value["failure_stage"] not in FAILURE_CODES
        or value["check_code"] != FAILURE_CODES[value["failure_stage"]]
        or any(
            value[key] is not False
            for key in (
                "adapted_cache_published",
                "adapted_pool_bytes_published",
                "candidate_metric_computed",
                "evaluator_opened",
                "fit_or_model_performed",
                "final_or_model_rule_selected",
                "full28_opened",
                "holdout_opened",
                "labels_opened",
                "original_pool_bytes_published",
                "promotion_authorized",
                "retry_authorized",
                "scientific_hypothesis_performance_tested",
            )
        )
    ):
        raise ValueError("Candidate10 terminal payload is unsafe")
    return value
