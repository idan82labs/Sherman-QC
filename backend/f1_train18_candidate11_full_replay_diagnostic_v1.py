"""Pure Candidate11 full-TRAIN18 replay diagnostic aggregation.

The guarded runner authenticates sources and supplies exactly 36 immutable
record receipts.  This module is filesystem blind.  It applies the frozen
Candidate9 lossless projection and Candidate8 D01--D05 diagnostic independently
to every receipt, catches ordinary per-record failures, and continues through
the complete inventory.  Public output is deliberately aggregate-only: record
identities, ordinals, values, exception details, and raw bitsets never leave the
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import ModuleType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 11
ENGINEERING_ATTEMPT_ID = "CANDIDATE11_FULL_TRAIN_REPLAY_DIAGNOSTIC_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_result_v1"

RECORD_COUNT = 36
PAIR_COUNT = 18
SOURCE_FILE_COUNT = 72
PASS_NAMES = ("primary", "replay")

C9_ADAPTER_SHA256 = "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096"
C9_ADAPTER_TESTS_SHA256 = "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623"
C8_RUNTIME_SHA256 = "d878eb9141f287435890ed3290a45f40dd0730130585a6846c38431e3342f9b8"
C8_ADAPTER_SHA256 = "b91a112356dba3ed4aa9e3fe51fe36ea12bf93d2dd9ab8900f2b3461509211be"
C8_RUNTIME_TESTS_SHA256 = "85c8b684dcaf4acaccb9800ba7bb7747c3546b24baae7d3c0685169431f52cfb"
C8_ADAPTER_TESTS_SHA256 = "20bb70082daf1e681dd2ce3a1367da59befce4eebabcfd7a75944c5246091b8b"
C7_RUNTIME_SHA256 = "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d"
C6_HYPOTHESIS_SHA256 = "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
BUILDER_SHA256 = "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2"
CLUSTER_SHA256 = "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c"
C7_DISPOSITION_SHA256 = "1a64d3148d018b224a3a44d9d36d2620f1d87f18063e32df14050210aa03f56b"
C10_CORE_SHA256 = "3ef5a1c26ba138ff3d085e89483da706288d683797977fa5ad8bc47b9a135e53"
C10_CORE_TESTS_SHA256 = "a612a5a6bd243daa6536bb1139d454f7435431291553c344ef20f72d975d5796"
C10_RUNNER_SHA256 = "dc47d8612b9dbdc01fe7d5432cae76b69b2e696061e488125e2c27bce77db8b5"
C10_RUNNER_TESTS_SHA256 = "c07c4192316ce8452310cb83b54119b0f20046fc7a6cebb5f985b7db5c53daa9"
C10_DISPOSITION_SHA256 = "eb6240e3b28149446f2a971e522c9ce73d880facf28a4496e528629cff03818c"

C8_ENGINEERING_ATTEMPT_ID = "CANDIDATE8_ENGINEERING_ATTEMPT_V1"
C8_RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate8_diagnostic_result_v1"
C8_PHASE_CONTRACT = (
    "D01_SEMANTIC_VALIDATION",
    "D02_POOL_PARSE_SANITIZE",
    "D03_OWNERSHIP_EXERCISE",
    "D04_FIXED_RULE_EXERCISE",
    "D05_WRAPPER_RESULT_VALIDATION",
)
C8_PHASE_CONTRACT_SHA256 = "192bd96ca4f944ceb9b2eb1dc9308e0d9f73bb09a83705579d3883995c49d94a"
C8_PASS_CODE = "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
C8_FAILURE_CODES = {
    "D01_SEMANTIC_VALIDATION": "C8_SEMANTIC_VALIDATION_FAILED",
    "D02_POOL_PARSE_SANITIZE": "C8_POOL_PARSE_SANITIZE_FAILED",
    "D03_OWNERSHIP_EXERCISE": "C8_OWNERSHIP_EXERCISE_FAILED",
    "D04_FIXED_RULE_EXERCISE": "C8_FIXED_RULE_EXERCISE_FAILED",
    "D05_WRAPPER_RESULT_VALIDATION": "C8_WRAPPER_RESULT_VALIDATION_FAILED",
}
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

STAGE_CONTRACT = (
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
PASS_CODE = "C11_ALL_RECORD_DIAGNOSTICS_REPLAY_COMPATIBLE"
FAIL_CODE = "C11_AGGREGATE_DIAGNOSTIC_INCOMPATIBLE"

RESULT_KEYS = frozenset(
    {
        "aggregate_reached_passed_bitsets_commitment_sha256",
        "all_records_compatible",
        "candidate_metric_computed",
        "check_code",
        "diagnostic_complete",
        "engineering_attempt_id",
        "evaluator_opened",
        "exception_class_exposed",
        "exception_text_exposed",
        "external_source_and_manifest_authentication_required",
        "fit_or_model_performed",
        "filesystem_or_artifact_discovery_capability_present",
        "holdout_opened",
        "identifier_or_ordinal_exposed",
        "input_receipt_vector_commitment_sha256",
        "kind",
        "labels_opened",
        "numeric_or_categorical_source_value_exposed",
        "primary_replay_identity",
        "promotion_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_lineage_commitment_sha256",
        "source_only",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "stage_contract_sha256",
        "status",
    }
)

_CONTEXT_SENTINEL = object()


class DiagnosticFailure(RuntimeError):
    """Fixed, non-data-bearing failure for invalid global invocation state."""

    def __init__(self) -> None:
        super().__init__("C11_DIAGNOSTIC_INVOCATION_REJECTED")


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
    if type(data) is not bytes:
        raise TypeError("Candidate11 hashes immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


STAGE_CONTRACT_SHA256 = sha256_bytes(canonical_bytes(list(STAGE_CONTRACT)))


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
class DiagnosticContext:
    c9_adapter: ModuleType
    c9_adapter_context: Any
    c8_adapter: ModuleType
    c8_adapter_context: Any
    c8_runtime: ModuleType
    candidate7_disposition_bytes: bytes
    candidate10_disposition_bytes: bytes
    source_bindings: tuple[tuple[str, str], ...]
    source_lineage_commitment_sha256: str
    _callable_snapshot: tuple[tuple[Any, Any], ...]
    _c8_adapter_context_snapshot: tuple[Any, ...]
    _c8_contract_snapshot: tuple[Any, ...]
    _sentinel: object


class _SealedReceiptVector:
    """Non-returned receipt container which refuses representation/serialization."""

    __slots__ = ("__entries",)

    def __init__(self, entries: tuple[tuple[tuple[str, Any], ...], ...]) -> None:
        self.__entries = entries

    def __repr__(self) -> str:
        return "<_SealedReceiptVector>"

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("Candidate11 sealed receipts are not serializable")

    def _commitment(self, *, manifest_binding_sha256: str, lineage: str) -> str:
        return sha256_bytes(
            canonical_bytes(
                {
                    "domain": "CANDIDATE11_SEALED_INPUT_RECEIPT_VECTOR_V1",
                    "manifest_binding_sha256": manifest_binding_sha256,
                    "receipts": [dict(entry) for entry in self.__entries],
                    "source_lineage_commitment_sha256": lineage,
                }
            )
        )


def _callable_identity(value: Any) -> tuple[Any, Any]:
    if not callable(value):
        raise TypeError("controlled dependency callable missing")
    return (getattr(value, "__self__", None), getattr(value, "__func__", value))


def _dependency_snapshot(
    c9_adapter: ModuleType, c8_adapter: ModuleType, c8_runtime: ModuleType
) -> tuple[tuple[Any, Any], ...]:
    return (
        _callable_identity(getattr(c9_adapter, "_validate_context")),
        _callable_identity(getattr(c9_adapter, "adapt_pool_bytes")),
        _callable_identity(getattr(c8_adapter, "diagnose_bound_pair")),
        _callable_identity(getattr(c8_runtime, "_validate_result")),
    )


def _c8_contract(c8_runtime: ModuleType) -> tuple[Any, ...]:
    return (
        c8_runtime.ATTEMPT_ORDINAL,
        c8_runtime.ENGINEERING_ATTEMPT_ID,
        c8_runtime.SCIENTIFIC_HYPOTHESIS_ID,
        c8_runtime.SCIENTIFIC_HYPOTHESIS_SOURCE_SHA256,
        c8_runtime.RESULT_KIND,
        c8_runtime.PASS_CODE,
        tuple(c8_runtime.PHASE_CONTRACT),
        tuple(sorted(dict(c8_runtime.FAILURE_CODES).items())),
        frozenset(c8_runtime.RESULT_KEYS),
        c8_runtime.PHASE_CONTRACT_SHA256,
    )


def _c8_adapter_context_snapshot(
    c8_adapter: ModuleType, c8_adapter_context: Any, c8_runtime: ModuleType
) -> tuple[Any, ...]:
    bindings = getattr(c8_adapter_context, "source_bindings", None)
    if (
        not hasattr(c8_adapter, "AdapterContext")
        or type(c8_adapter_context) is not c8_adapter.AdapterContext
        or getattr(c8_adapter_context, "runtime", None) is not c8_runtime
        or not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "adapter",
            "adapter_tests",
            "candidate7_runtime",
            "hypothesis",
            "runtime",
            "runtime_tests",
        }
        or not all(isinstance(bindings[name], Mapping) for name in bindings)
        or bindings["adapter"].get("sha256") != C8_ADAPTER_SHA256
        or bindings["adapter_tests"].get("sha256")
        != C8_ADAPTER_TESTS_SHA256
        or bindings["hypothesis"].get("sha256") != C6_HYPOTHESIS_SHA256
        or bindings["candidate7_runtime"].get("sha256") != C7_RUNTIME_SHA256
        or bindings["runtime"].get("sha256") != C8_RUNTIME_SHA256
        or bindings["runtime_tests"].get("sha256")
        != C8_RUNTIME_TESTS_SHA256
    ):
        raise ValueError("Candidate8 adapter context contract mismatch")
    return (
        c8_adapter.AdapterContext,
        c8_adapter_context.runtime,
        getattr(c8_adapter_context, "hypothesis", None),
        getattr(c8_adapter_context, "candidate7_runtime", None),
        tuple(
            sorted(
                (name, tuple(sorted(dict(value).items())))
                for name, value in bindings.items()
            )
        ),
    )


def _expected_source_bindings() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "builder": BUILDER_SHA256,
                "c10_core": C10_CORE_SHA256,
                "c10_core_tests": C10_CORE_TESTS_SHA256,
                "c10_runner": C10_RUNNER_SHA256,
                "c10_runner_tests": C10_RUNNER_TESTS_SHA256,
                "c6_hypothesis": C6_HYPOTHESIS_SHA256,
                "c7_runtime": C7_RUNTIME_SHA256,
                "c8_adapter": C8_ADAPTER_SHA256,
                "c8_runtime": C8_RUNTIME_SHA256,
                "c9_adapter": C9_ADAPTER_SHA256,
                "c9_adapter_tests": C9_ADAPTER_TESTS_SHA256,
                "cluster": CLUSTER_SHA256,
            }.items()
        )
    )


def _lineage_commitment(bindings: tuple[tuple[str, str], ...]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "attempt_ordinal": ATTEMPT_ORDINAL,
                "domain": "CANDIDATE11_SOURCE_LINEAGE_V1",
                "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
                "candidate10_disposition_sha256": C10_DISPOSITION_SHA256,
                "candidate7_disposition_sha256": C7_DISPOSITION_SHA256,
                "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
                "source_bindings": [
                    {"name": name, "sha256": digest} for name, digest in bindings
                ],
            }
        )
    )


def _validate_dependency_contracts(
    c9_adapter: ModuleType, c8_runtime: ModuleType
) -> None:
    contract = _c8_contract(c8_runtime)
    if (
        type(c9_adapter) is not ModuleType
        or type(c8_runtime) is not ModuleType
        or c9_adapter.ATTEMPT_ORDINAL != 9
        or c9_adapter.ENGINEERING_ATTEMPT_ID
        != "CANDIDATE9_LOSSLESS_POOL_ALIAS_ADAPTER_V1"
        or c9_adapter.ADAPTER_KIND
        != "sherman_qc_f1_aug30_train18_candidate9_lossless_pool_alias_adapter_v1"
        or c9_adapter.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or contract[0] != 8
        or contract[1] != C8_ENGINEERING_ATTEMPT_ID
        or contract[2] != SCIENTIFIC_HYPOTHESIS_ID
        or contract[3] != C6_HYPOTHESIS_SHA256
        or contract[4] != C8_RESULT_KIND
        or contract[5] != C8_PASS_CODE
        or contract[6] != C8_PHASE_CONTRACT
        or dict(contract[7])
        != {"LINEAGE": "C8_LINEAGE_VALIDATION_FAILED", **C8_FAILURE_CODES}
        or contract[8] != C8_RESULT_KEYS
        or contract[9] != C8_PHASE_CONTRACT_SHA256
    ):
        raise ValueError("controlled dependency contract mismatch")


def make_verified_context(
    *,
    c9_adapter: ModuleType,
    c9_adapter_context: Any,
    c8_adapter: ModuleType,
    c8_adapter_context: Any,
    c8_runtime: ModuleType,
    candidate7_disposition_bytes: bytes,
    candidate10_disposition_bytes: bytes,
    source_bindings: Mapping[str, str],
) -> DiagnosticContext:
    try:
        if type(c8_adapter) is not ModuleType:
            raise TypeError("Candidate8 adapter must be a module")
        bindings = tuple(sorted(source_bindings.items()))
        if bindings != _expected_source_bindings() or any(
            type(name) is not str or not _is_sha(digest)
            for name, digest in bindings
        ):
            raise ValueError("source binding mismatch")
        if (
            type(candidate7_disposition_bytes) is not bytes
            or sha256_bytes(candidate7_disposition_bytes) != C7_DISPOSITION_SHA256
            or type(candidate10_disposition_bytes) is not bytes
            or sha256_bytes(candidate10_disposition_bytes)
            != C10_DISPOSITION_SHA256
        ):
            raise ValueError("disposition binding mismatch")
        _validate_dependency_contracts(c9_adapter, c8_runtime)
        c9_adapter._validate_context(c9_adapter_context)
        snapshot = _dependency_snapshot(c9_adapter, c8_adapter, c8_runtime)
        c8_adapter_snapshot = _c8_adapter_context_snapshot(
            c8_adapter, c8_adapter_context, c8_runtime
        )
        c8_snapshot = _c8_contract(c8_runtime)
    except Exception:
        raise DiagnosticFailure() from None
    return DiagnosticContext(
        c9_adapter=c9_adapter,
        c9_adapter_context=c9_adapter_context,
        c8_adapter=c8_adapter,
        c8_adapter_context=c8_adapter_context,
        c8_runtime=c8_runtime,
        candidate7_disposition_bytes=candidate7_disposition_bytes,
        candidate10_disposition_bytes=candidate10_disposition_bytes,
        source_bindings=bindings,
        source_lineage_commitment_sha256=_lineage_commitment(bindings),
        _callable_snapshot=snapshot,
        _c8_adapter_context_snapshot=c8_adapter_snapshot,
        _c8_contract_snapshot=c8_snapshot,
        _sentinel=_CONTEXT_SENTINEL,
    )


def _validate_context(context: DiagnosticContext) -> None:
    try:
        if (
            type(context) is not DiagnosticContext
            or context._sentinel is not _CONTEXT_SENTINEL
            or context.source_bindings != _expected_source_bindings()
            or context.source_lineage_commitment_sha256
            != _lineage_commitment(context.source_bindings)
            or sha256_bytes(context.candidate7_disposition_bytes)
            != C7_DISPOSITION_SHA256
            or sha256_bytes(context.candidate10_disposition_bytes)
            != C10_DISPOSITION_SHA256
            or context._callable_snapshot
            != _dependency_snapshot(
                context.c9_adapter, context.c8_adapter, context.c8_runtime
            )
            or context._c8_adapter_context_snapshot
            != _c8_adapter_context_snapshot(
                context.c8_adapter,
                context.c8_adapter_context,
                context.c8_runtime,
            )
            or context._c8_contract_snapshot != _c8_contract(context.c8_runtime)
        ):
            raise ValueError("verified context changed")
        _validate_dependency_contracts(context.c9_adapter, context.c8_runtime)
        context.c9_adapter._validate_context(context.c9_adapter_context)
    except Exception:
        raise DiagnosticFailure() from None


def _slot_schema_valid(
    record: RecordBytes,
    slot: int,
    manifest_occurrences: Mapping[str, int],
    file_occurrences: Mapping[str, int],
) -> bool:
    return (
        type(record) is RecordBytes
        and type(record.ordinal) is int
        and record.ordinal == slot
        and type(record.pair_ordinal) is int
        and record.pair_ordinal == slot // 2
        and record.pass_name == PASS_NAMES[slot % 2]
        and type(record.semantic_bytes) is bytes
        and bool(record.semantic_bytes)
        and type(record.pool_bytes) is bytes
        and bool(record.pool_bytes)
        and type(record.semantic_size_bytes) is int
        and record.semantic_size_bytes == len(record.semantic_bytes)
        and type(record.pool_size_bytes) is int
        and record.pool_size_bytes == len(record.pool_bytes)
        and all(
            _is_sha(value)
            for value in (
                record.manifest_record_binding_sha256,
                record.semantic_file_binding_sha256,
                record.pool_file_binding_sha256,
                record.semantic_sha256,
                record.pool_sha256,
            )
        )
        and manifest_occurrences.get(record.manifest_record_binding_sha256) == 1
        and file_occurrences.get(record.semantic_file_binding_sha256) == 1
        and file_occurrences.get(record.pool_file_binding_sha256) == 1
        and record.semantic_file_binding_sha256
        != record.pool_file_binding_sha256
        and sha256_bytes(record.semantic_bytes) == record.semantic_sha256
        and sha256_bytes(record.pool_bytes) == record.pool_sha256
    )


def _bit_string(bits: Sequence[bool]) -> str:
    return "".join("1" if bit else "0" for bit in bits)


def diagnose_all_records(
    context: DiagnosticContext,
    records: Sequence[RecordBytes],
    *,
    manifest_binding_sha256: str,
) -> dict[str, Any]:
    _validate_context(context)
    if (
        not _is_sha(manifest_binding_sha256)
        or not isinstance(records, (tuple, list))
        or len(records) != RECORD_COUNT
        or any(type(record) is not RecordBytes for record in records)
    ):
        raise DiagnosticFailure()
    bound = tuple(records)
    manifest_occurrences: dict[str, int] = {}
    file_occurrences: dict[str, int] = {}
    for record in bound:
        for value, target in (
            (record.manifest_record_binding_sha256, manifest_occurrences),
            (record.semantic_file_binding_sha256, file_occurrences),
            (record.pool_file_binding_sha256, file_occurrences),
        ):
            if type(value) is str:
                target[value] = target.get(value, 0) + 1

    reached = [[False] * RECORD_COUNT for _ in STAGE_CONTRACT]
    passed = [[False] * RECORD_COUNT for _ in STAGE_CONTRACT]
    adapted_outputs: list[bytes | None] = [None] * RECORD_COUNT
    diagnostic_outputs: list[bytes | None] = [None] * RECORD_COUNT
    receipt_entries: list[tuple[tuple[str, Any], ...]] = []

    for slot, record in enumerate(bound):
        reached[0][slot] = True
        try:
            schema_ok = _slot_schema_valid(
                record, slot, manifest_occurrences, file_occurrences
            )
        except Exception:
            schema_ok = False
        if not schema_ok:
            receipt_entries.append((("slot", slot), ("schema_bound", False)))
            continue
        passed[0][slot] = True
        receipt_entries.append(
            tuple(
                sorted(
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
                    }.items()
                )
            )
        )

        reached[1][slot] = True
        try:
            adapted = context.c9_adapter.adapt_pool_bytes(
                context.c9_adapter_context,
                pool_bytes=record.pool_bytes,
                expected_pool_sha256=record.pool_sha256,
            )
            adapted_bytes = adapted.adapted_pool_bytes
            adapted_sha = adapted.adapted_pool_sha256
            if (
                adapted.original_pool_sha256 != record.pool_sha256
                or adapted.projection_byte_equivalent is not True
                or type(adapted_bytes) is not bytes
                or not adapted_bytes
                or not _is_sha(adapted_sha)
                or sha256_bytes(adapted_bytes) != adapted_sha
            ):
                raise ValueError("projection receipt mismatch")
        except Exception:
            # A raising or malformed adapter call provides no accepted C8
            # phase receipt.  Do not infer that D01 (or any later phase) was
            # reached merely because the call was attempted.
            continue
        passed[1][slot] = True
        adapted_outputs[slot] = adapted_bytes

        try:
            diagnostic = context.c8_adapter.diagnose_bound_pair(
                context.c8_adapter_context,
                semantic_bytes=record.semantic_bytes,
                pool_bytes=adapted_bytes,
                candidate7_disposition_bytes=context.candidate7_disposition_bytes,
                semantic_sha256=record.semantic_sha256,
                pool_sha256=adapted_sha,
                candidate7_disposition_sha256=C7_DISPOSITION_SHA256,
            )
            context.c8_runtime._validate_result(diagnostic)
            if not isinstance(diagnostic, Mapping):
                raise TypeError("diagnostic is not a mapping")
            phase = diagnostic.get("failure_phase")
            expected_code = (
                C8_PASS_CODE if phase is None else C8_FAILURE_CODES.get(phase)
            )
            if (
                set(diagnostic) != context._c8_contract_snapshot[8]
                or phase not in {None, *C8_PHASE_CONTRACT}
                or expected_code is None
                or diagnostic.get("check_code") != expected_code
            ):
                raise ValueError("diagnostic receipt mismatch")
            diagnostic_bytes = canonical_bytes(dict(diagnostic))
        except Exception:
            continue

        failure_index = (
            len(C8_PHASE_CONTRACT)
            if phase is None
            else C8_PHASE_CONTRACT.index(phase)
        )
        for phase_index in range(len(C8_PHASE_CONTRACT)):
            stage_index = 2 + phase_index
            if phase_index <= failure_index:
                reached[stage_index][slot] = True
            if phase_index < failure_index or phase is None:
                passed[stage_index][slot] = True
        diagnostic_outputs[slot] = diagnostic_bytes

    for pair in range(PAIR_COUNT):
        primary_slot = 2 * pair
        replay_slot = primary_slot + 1
        primary = bound[primary_slot]
        replay = bound[replay_slot]
        slots = (primary_slot, replay_slot)
        if passed[0][primary_slot] and passed[0][replay_slot]:
            try:
                semantic_same = (
                    primary.semantic_sha256 == replay.semantic_sha256
                    and primary.semantic_bytes == replay.semantic_bytes
                )
            except Exception:
                semantic_same = False
            try:
                pool_same = (
                    primary.pool_sha256 == replay.pool_sha256
                    and primary.pool_bytes == replay.pool_bytes
                )
            except Exception:
                pool_same = False
            for slot in slots:
                reached[7][slot] = True
                passed[7][slot] = semantic_same
                reached[8][slot] = True
                passed[8][slot] = pool_same
        if adapted_outputs[primary_slot] is not None and adapted_outputs[replay_slot] is not None:
            try:
                adapted_same = adapted_outputs[primary_slot] == adapted_outputs[replay_slot]
            except Exception:
                adapted_same = False
            for slot in slots:
                reached[9][slot] = True
                passed[9][slot] = adapted_same
        if diagnostic_outputs[primary_slot] is not None and diagnostic_outputs[replay_slot] is not None:
            try:
                diagnostic_same = (
                    diagnostic_outputs[primary_slot] == diagnostic_outputs[replay_slot]
                )
            except Exception:
                diagnostic_same = False
            for slot in slots:
                reached[10][slot] = True
                passed[10][slot] = diagnostic_same

    record_compatible = [
        all(passed[stage][slot] for stage in range(11))
        for slot in range(RECORD_COUNT)
    ]
    for slot in range(RECORD_COUNT):
        reached[11][slot] = True
        # This stage records that aggregate construction itself completed.  It
        # is not a second copy of the per-record compatibility conjunction.
        passed[11][slot] = True

    all_compatible = all(record_compatible)
    sealed = _SealedReceiptVector(tuple(receipt_entries))
    receipt_commitment = sealed._commitment(
        manifest_binding_sha256=manifest_binding_sha256,
        lineage=context.source_lineage_commitment_sha256,
    )
    bitset_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE11_AGGREGATE_REACHED_PASSED_BITSETS_V1",
                "input_receipt_vector_commitment_sha256": receipt_commitment,
                "passed": [_bit_string(bits) for bits in passed],
                "reached": [_bit_string(bits) for bits in reached],
                "stage_contract_sha256": STAGE_CONTRACT_SHA256,
            }
        )
    )
    stage_reached_all_bitset = _bit_string([all(bits) for bits in reached])
    stage_passed_all_bitset = _bit_string([all(bits) for bits in passed])
    result = {
        "aggregate_reached_passed_bitsets_commitment_sha256": bitset_commitment,
        "all_records_compatible": all_compatible,
        "candidate_metric_computed": False,
        "check_code": PASS_CODE if all_compatible else FAIL_CODE,
        "diagnostic_complete": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "exception_class_exposed": False,
        "exception_text_exposed": False,
        "external_source_and_manifest_authentication_required": True,
        "fit_or_model_performed": False,
        "filesystem_or_artifact_discovery_capability_present": False,
        "holdout_opened": False,
        "identifier_or_ordinal_exposed": False,
        "input_receipt_vector_commitment_sha256": receipt_commitment,
        "kind": RESULT_KIND,
        "labels_opened": False,
        "numeric_or_categorical_source_value_exposed": False,
        "primary_replay_identity": all(
            passed[stage][slot]
            for stage in (7, 8, 9, 10)
            for slot in range(RECORD_COUNT)
        ),
        "promotion_authorized": False,
        "schema_version": SCHEMA_VERSION,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_lineage_commitment_sha256": context.source_lineage_commitment_sha256,
        "source_only": True,
        "stage_passed_all_bitset": stage_passed_all_bitset,
        "stage_reached_all_bitset": stage_reached_all_bitset,
        "stage_contract_sha256": STAGE_CONTRACT_SHA256,
        "status": (
            "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
            if all_compatible
            else "SOURCE_REPLAY_INCOMPATIBLE_NONAUTHORIZING"
        ),
    }
    _validate_public_result(result)
    return result


def _validate_public_result(value: Mapping[str, Any]) -> None:
    required_false = {
        "candidate_metric_computed",
        "evaluator_opened",
        "exception_class_exposed",
        "exception_text_exposed",
        "fit_or_model_performed",
        "filesystem_or_artifact_discovery_capability_present",
        "holdout_opened",
        "identifier_or_ordinal_exposed",
        "labels_opened",
        "numeric_or_categorical_source_value_exposed",
        "promotion_authorized",
        "scientific_hypothesis_performance_tested",
    }
    try:
        compatible = value["all_records_compatible"]
        valid = (
            isinstance(value, Mapping)
            and set(value) == RESULT_KEYS
            and type(value["schema_version"]) is int
            and value["schema_version"] == SCHEMA_VERSION
            and value["kind"] == RESULT_KIND
            and value["engineering_attempt_id"] == ENGINEERING_ATTEMPT_ID
            and value["scientific_hypothesis_id"] == SCIENTIFIC_HYPOTHESIS_ID
            and type(compatible) is bool
            and value["diagnostic_complete"] is True
            and value["source_only"] is True
            and value["external_source_and_manifest_authentication_required"] is True
            and all(value[key] is False for key in required_false)
            and value["check_code"] == (PASS_CODE if compatible else FAIL_CODE)
            and value["status"]
            == (
                "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
                if compatible
                else "SOURCE_REPLAY_INCOMPATIBLE_NONAUTHORIZING"
            )
            and type(value["primary_replay_identity"]) is bool
            and isinstance(value["stage_reached_all_bitset"], str)
            and len(value["stage_reached_all_bitset"]) == len(STAGE_CONTRACT)
            and set(value["stage_reached_all_bitset"]) <= {"0", "1"}
            and isinstance(value["stage_passed_all_bitset"], str)
            and len(value["stage_passed_all_bitset"]) == len(STAGE_CONTRACT)
            and set(value["stage_passed_all_bitset"]) <= {"0", "1"}
            and all(
                not (passed == "1" and reached == "0")
                for passed, reached in zip(
                    value["stage_passed_all_bitset"],
                    value["stage_reached_all_bitset"],
                )
            )
            and value["stage_reached_all_bitset"][-1] == "1"
            and value["stage_passed_all_bitset"][-1] == "1"
            and compatible
            == (value["stage_passed_all_bitset"][:11] == "1" * 11)
            and (
                not compatible
                or value["primary_replay_identity"] is True
            )
            and value["primary_replay_identity"]
            == (value["stage_passed_all_bitset"][7:11] == "1" * 4)
            and value["stage_contract_sha256"] == STAGE_CONTRACT_SHA256
            and all(
                _is_sha(value[key])
                for key in (
                    "aggregate_reached_passed_bitsets_commitment_sha256",
                    "input_receipt_vector_commitment_sha256",
                    "source_lineage_commitment_sha256",
                    "stage_contract_sha256",
                )
            )
        )
    except Exception:
        valid = False
    if not valid:
        raise DiagnosticFailure()
