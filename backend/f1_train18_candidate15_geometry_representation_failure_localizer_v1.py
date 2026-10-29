"""Filesystem-blind Candidate15 geometry-representation failure localizer.

This source-only core accepts caller-supplied immutable pool bytes through a
verified dependency context.  It performs no path discovery or filesystem I/O,
publishes no per-pool details, and makes no authority, model, metric, or
performance claim.  The full sealed implementation follows the Candidate13
opaque-capability design and is intentionally bound to the terminal Candidate14
engineering disposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 15
ENGINEERING_ATTEMPT_ID = "CANDIDATE15_GEOMETRY_REPRESENTATION_FAILURE_LOCALIZER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = "sherman_qc_f1_aug30_candidate15_geometry_representation_failure_localizer_result_v1"

STAGE_CONTRACT = (
    "INPUT_HASH_AND_JSON",
    "C13_PREFIX_0_TO_5_REPRODUCTION",
    "SUPPORTED_GEOMETRY_PRESENT",
    "LINE_START_END_JOINT_OR_ABSENT",
    "C9_AUX_KEY_PRESENCE_REQUIRES_SPAN_WHEN_INCOMPLETE",
    "PRESENT_GEOMETRY_SHAPE_TYPE_FINITE",
    "CONDITIONED_DOMAIN_AND_NONDEGENERACY",
    "C9_PARTIAL_AUX_RECONSTRUCTION_SPAN_EQUIVALENCE",
    "C6_ORDERED_REFERENCE_REPRESENTATION_EQUIVALENCE",
    "EXACT_C13_AGGREGATE_STAGE6_PARITY",
    "EXACT_C9_FULL_OUTCOME_PARITY",
    "AGGREGATE_STAGE6_VIOLATION_COVERAGE",
)

VIOLATION_CONTRACT = (
    "NO_SUPPORTED_GEOMETRY",
    "INCOMPLETE_LINE_ENDPOINT_PAIR",
    "PARTIAL_AUX_WITHOUT_SPAN",
    "MALFORMED_GEOMETRY_VALUE",
    "OUT_OF_DOMAIN_OR_DEGENERATE",
    "PARTIAL_AUX_SPAN_CONFLICT",
    "C6_ORDERED_REFERENCE_REPRESENTATION_CONFLICT",
    "UNCLASSIFIED_OR_PARITY_MISMATCH",
)

FIXED_BINDINGS = {
    "builder_source": {
        "path": "scripts/build_f1_scan_only_candidate_pool.py",
        "sha256": "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2",
        "size_bytes": 57613,
    },
    "builder_tests": {
        "path": "tests/test_build_f1_scan_only_candidate_pool.py",
        "sha256": "43172b7711ce7ec3f9e70845e9c61e5732999cd2f9140f6681a7ba6a98291a0e",
        "size_bytes": 21708,
    },
    "c13_source": {
        "path": "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
        "sha256": "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
        "size_bytes": 65344,
    },
    "c13_tests": {
        "path": "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
        "sha256": "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
        "size_bytes": 43387,
    },
    "c14_terminal_disposition": {
        "path": "research/f1_aug30_train18_candidate14_terminal_disposition_v1.json",
        "sha256": "15235350abdb2cc803bc5fb4abca7b836347a500fa9f24a25fb16ea30945b360",
        "size_bytes": 8008,
    },
    "c6_source": {
        "path": "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
        "sha256": "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
        "size_bytes": 53837,
    },
    "c6_tests": {
        "path": "tests/test_f1_train18_candidate6_conservative_recall_recovery_v1.py",
        "sha256": "98df5af442627ce71bf539bb6a801cb30e3ad5b3a10f4e680f55a07c6e4c452f",
        "size_bytes": 26864,
    },
    "c9_source": {
        "path": "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "sha256": "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
        "size_bytes": 22354,
    },
    "c9_tests": {
        "path": "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "sha256": "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
        "size_bytes": 16598,
    },
    "clusterer_source": {
        "path": "backend/f1_line_object_clustering.py",
        "sha256": "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c",
        "size_bytes": 6817,
    },
    "clusterer_tests": {
        "path": "tests/test_f1_line_object_clustering.py",
        "sha256": "e47176604a465967c0e568c6388be6bec2fdaf5608818d7e44618a36e6c5b8d4",
        "size_bytes": 1252,
    },
}

C13_PREFIX_STAGES = (
    "CONTEXT_INPUT_HASH_AND_JSON",
    "HEADER",
    "ROW_IDENTITY_ORDER_SNAPSHOT",
    "RECURSIVE_FORBIDDEN_FIELDS",
    "SOURCE_AUTHORITY_ALIAS_SPELLING",
    "CLUSTER_PROVENANCE_NAMESPACE",
)
C13_GEOMETRY_STAGE_INDEX = 6
GEOMETRY_EQUIVALENCE_TOLERANCE_MM = 1.0e-6

EXPECTED_C14_C13_AGGREGATE = {
    "adapter_or_hypothesis_semantics_modified_by_core": False,
    "artifact_discovery_or_open_performed_by_core": False,
    "builder_or_clusterer_invoked_by_core": False,
    "candidate11_actual_record_cause_resolved": False,
    "candidate9_internal_failure_stage_claimed": False,
    "check_code": "C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION",
    "diagnostic_stage_precedence_is_candidate13_contract": True,
    "exact_c9_success_failure_equivalent_for_all": True,
    "first_failed_candidate13_diagnostic_stage": "GEOMETRY_ALIAS_EQUIVALENCE",
    "fit_performed": False,
    "full28_opened": False,
    "holdout_opened": False,
    "input_or_projection_bytes_published": False,
    "input_origin_authenticated_by_core": False,
    "kind": "sherman_qc_f1_aug30_candidate13_pool_adapter_stage_result_v1",
    "label_or_evaluator_opened": False,
    "metric_computed": False,
    "performance_tested": False,
    "real_or_control_input_opened_by_core": False,
    "rule_selected": False,
    "stage_contract_commitment_sha256": "e943acb1d86356bc17b02652b4b542e1ad1df9b5a50233da31e5a6c0f31b077f",
    "stage_passed_all_bitset": "1111110001",
    "stage_reached_all_bitset": "1111111001",
    "status": "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING",
}

C13_RUNTIME_CONSTANTS = (
    "ATTEMPT_ORDINAL",
    "ENGINEERING_ATTEMPT_ID",
    "SCIENTIFIC_HYPOTHESIS_ID",
    "STAGE_CONTRACT",
    "FIXED_BINDINGS",
)
C9_RUNTIME_CONSTANTS = (
    "ATTEMPT_ORDINAL",
    "SCIENTIFIC_HYPOTHESIS_ID",
    "HYPOTHESIS_SOURCE_SHA256",
    "BUILDER_SOURCE_SHA256",
    "CLUSTER_SOURCE_SHA256",
    "TARGET_CANDIDATE_SOURCE",
    "AUTHORITATIVE_SOURCE_FIELD",
    "STALE_ALIAS_FIELD",
    "AUXILIARY_GEOMETRY_FIELDS",
    "SOURCE_FIELDS",
    "POOL_ALGORITHM",
    "POOL_SCHEMA_VERSION",
    "WORKING_POINT_BUDGET",
    "REPRESENTATION_CONTRACT",
)
C6_RUNTIME_CONSTANTS = (
    "ATTEMPT_ORDINAL",
    "MAX_ABS_COORDINATE_MM",
    "MIN_CONDITIONED_LINE_LENGTH_MM",
    "MAX_CONDITIONED_LINE_LENGTH_MM",
    "MAX_ADDITIONS_PER_CASE",
    "OWNERSHIP_TRANSVERSE_OFFSET_MM",
    "ALLOWED_SOURCE_KINDS",
    "NUMERIC_FEATURES",
    "POOL_INFERENCE_CONTRACT",
    "FORBIDDEN_KEY_TOKENS",
    "FORBIDDEN_COMPACT_KEYS",
)
BUILDER_RUNTIME_CONSTANTS = ("SUPPORT_ATOM_ID_CAP_PER_CANDIDATE",)

_CONTEXT_STATE_READY = "READY"
_CONTEXT_STATE_IN_FLIGHT = "IN_FLIGHT"
_CONTEXT_STATE_TERMINAL = "TERMINAL"
_CONTEXT_STATE_AGGREGATED = "AGGREGATED"
_CONTEXT_STATES = (
    _CONTEXT_STATE_READY,
    _CONTEXT_STATE_IN_FLIGHT,
    _CONTEXT_STATE_TERMINAL,
    _CONTEXT_STATE_AGGREGATED,
)

# Every module-level value whose value can affect an issued context, a geometry
# decision, or the public aggregate is named here.  The tuple commits to itself
# so removing or reordering a name is also detected after context issuance.
LOCAL_RUNTIME_CONSTANTS = (
    "SCHEMA_VERSION",
    "ATTEMPT_ORDINAL",
    "ENGINEERING_ATTEMPT_ID",
    "SCIENTIFIC_HYPOTHESIS_ID",
    "RESULT_KIND",
    "STAGE_CONTRACT",
    "VIOLATION_CONTRACT",
    "FIXED_BINDINGS",
    "C13_PREFIX_STAGES",
    "C13_GEOMETRY_STAGE_INDEX",
    "GEOMETRY_EQUIVALENCE_TOLERANCE_MM",
    "EXPECTED_C14_C13_AGGREGATE",
    "C13_RUNTIME_CONSTANTS",
    "C9_RUNTIME_CONSTANTS",
    "C6_RUNTIME_CONSTANTS",
    "BUILDER_RUNTIME_CONSTANTS",
    "RESULT_KEYS",
    "_CONTEXT_STATE_READY",
    "_CONTEXT_STATE_IN_FLIGHT",
    "_CONTEXT_STATE_TERMINAL",
    "_CONTEXT_STATE_AGGREGATED",
    "_CONTEXT_STATES",
    "LOCAL_RUNTIME_CONSTANTS",
)


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
        raise TypeError("Candidate15 hashes exact immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _binding_projection(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Candidate15 source binding map mismatch")
    names = tuple(dict.keys(value))
    if any(type(name) is not str for name in names) or set(names) != set(FIXED_BINDINGS):
        raise ValueError("Candidate15 source binding map mismatch")
    result: dict[str, Any] = {}
    for name in sorted(names):
        binding = dict.__getitem__(value, name)
        if type(binding) is not dict:
            raise ValueError("Candidate15 source binding schema mismatch")
        fields = tuple(dict.keys(binding))
        if (
            any(type(field) is not str for field in fields)
            or set(fields) != {"path", "sha256", "size_bytes"}
            or type(dict.__getitem__(binding, "path")) is not str
            or type(dict.__getitem__(binding, "sha256")) is not str
            or type(dict.__getitem__(binding, "size_bytes")) is not int
        ):
            raise ValueError("Candidate15 source binding schema mismatch")
        result[name] = {
            "path": dict.__getitem__(binding, "path"),
            "sha256": dict.__getitem__(binding, "sha256"),
            "size_bytes": dict.__getitem__(binding, "size_bytes"),
        }
    return result


def _sealed_binding_projection(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    proxy_type = type(types.MappingProxyType({}))
    if type(value) is not proxy_type:
        raise ValueError("Candidate15 sealed source binding map mismatch")
    result: dict[str, Any] = {}
    names = tuple(value.keys())
    if set(names) != set(FIXED_BINDINGS) or any(type(name) is not str for name in names):
        raise ValueError("Candidate15 sealed source binding map mismatch")
    for name in sorted(names):
        binding = value[name]
        if type(binding) is not proxy_type or set(binding.keys()) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("Candidate15 sealed source binding schema mismatch")
        result[name] = dict(binding)
    return result


def _frozen_value(value: Any) -> Any:
    if type(value) in (type(None), bool, int, float, str):
        return ("scalar", type(value).__name__, value)
    if type(value) in (tuple, list):
        return ("sequence", type(value).__name__, tuple(_frozen_value(v) for v in value))
    if type(value) in (set, frozenset):
        children = [_frozen_value(v) for v in value]
        children.sort(key=canonical_bytes)
        return ("set", type(value).__name__, tuple(children))
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Candidate15 constant mapping keys must be exact strings")
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple((key, _frozen_value(child)) for key, child in sorted(value.items())),
        )
    raise TypeError(f"Candidate15 cannot freeze dependency value {type(value).__name__}")


def _function_contract(function: types.FunctionType) -> tuple[Any, ...]:
    if type(function) is not types.FunctionType:
        raise TypeError("Candidate15 controlled callable must be an exact function")
    return (
        function,
        function.__code__,
        _frozen_value(function.__defaults__ or ()),
        _frozen_value(function.__kwdefaults__ or {}),
    )


def _class_contract(value: type) -> tuple[Any, ...]:
    parameters = getattr(value, "__dataclass_params__", None)
    return (
        value,
        value.__module__,
        value.__name__,
        tuple(sorted(getattr(value, "__dataclass_fields__", {}))),
        None
        if parameters is None
        else (
            bool(parameters.init),
            bool(parameters.repr),
            bool(parameters.eq),
            bool(parameters.order),
            bool(parameters.unsafe_hash),
            bool(parameters.frozen),
        ),
        tuple(
            (name, *_function_contract(child))
            for name, child in sorted(vars(value).items())
            if type(child) is types.FunctionType
        ),
    )


def _module_symbol_snapshot(module: types.ModuleType) -> tuple[Any, ...]:
    if type(module) is not types.ModuleType:
        raise TypeError("Candidate15 dependency must be an exact module")
    result: list[tuple[Any, ...]] = []
    for name, value in sorted(vars(module).items()):
        if type(value) is types.FunctionType and value.__module__ == module.__name__:
            result.append(("function", name, *_function_contract(value)))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            result.append(("class", name, *_class_contract(value)))
    return tuple(result)


def _dependency_constant_projection(
    c13_localizer: types.ModuleType,
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
) -> dict[str, Any]:
    return {
        "builder": {name: _frozen_value(getattr(builder, name)) for name in BUILDER_RUNTIME_CONSTANTS},
        "c13": {name: _frozen_value(getattr(c13_localizer, name)) for name in C13_RUNTIME_CONSTANTS},
        "c6": {name: _frozen_value(getattr(c6_hypothesis, name)) for name in C6_RUNTIME_CONSTANTS},
        "c9": {name: _frozen_value(getattr(c9_adapter, name)) for name in C9_RUNTIME_CONSTANTS},
    }


def _dependency_constant_commitment(
    c13_localizer: types.ModuleType,
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE15_TRANSITIVE_DEPENDENCY_CONSTANTS_V1",
                "projection": _dependency_constant_projection(
                    c13_localizer, c9_adapter, c6_hypothesis, builder
                ),
            }
        )
    )


def _dependency_symbol_snapshot(
    c13_localizer: types.ModuleType,
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> tuple[Any, ...]:
    modules = (c13_localizer, c9_adapter, c6_hypothesis, builder, clusterer)
    return (
        tuple((module, module.__name__, _module_symbol_snapshot(module)) for module in modules),
        (
            ("local_hashlib", hashlib, hashlib.sha256),
            ("local_json", json, json.dumps, json.loads),
            ("local_math", math, math.isfinite, math.sqrt),
            ("c13_exports", c13_localizer.LocalizerContext,
             c13_localizer.make_verified_context,
             c13_localizer.localize_pool_adapter_contract,
             c13_localizer.aggregate_pool_adapter_receipts),
            ("c9_context_sentinel", c9_adapter._CONTEXT_SENTINEL),
            ("c9_runtime", c9_adapter.hashlib, c9_adapter.json, c9_adapter.re),
            ("c6_runtime", c6_hypothesis.hashlib, c6_hypothesis.json,
             c6_hypothesis.math, c6_hypothesis.re, c6_hypothesis._HEX64),
            ("builder_runtime", builder.hashlib, builder.json, builder.math, builder.np),
            ("clusterer_runtime", clusterer.math, clusterer.np),
        ),
    )


def _local_function_snapshot() -> tuple[Any, ...]:
    result: list[tuple[Any, ...]] = []
    for name, value in sorted(globals().items()):
        if type(value) is types.FunctionType and value.__module__ == __name__:
            result.append(("function", name, *_function_contract(value)))
        elif isinstance(value, type) and value.__module__ == __name__:
            result.append(("class", name, *_class_contract(value)))
    return tuple(result)


def _local_constant_commitment() -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE15_LOCAL_OUTCOME_CONSTANTS_V1",
                "projection": {
                    name: _frozen_value(globals()[name])
                    for name in LOCAL_RUNTIME_CONSTANTS
                },
            }
        )
    )


@dataclass(frozen=True)
class _ContextRecord:
    c13_localizer: types.ModuleType
    c13_context: Any
    c9_adapter: types.ModuleType
    c9_context: Any
    c6_hypothesis: types.ModuleType
    builder: types.ModuleType
    clusterer: types.ModuleType
    bindings: Mapping[str, Mapping[str, Any]]
    binding_commitment_sha256: str
    dependency_symbol_snapshot: tuple[Any, ...]
    dependency_constant_commitment_sha256: str
    local_function_snapshot: tuple[Any, ...]
    local_constant_commitment_sha256: str


@dataclass
class _PoolState:
    pool_bytes: bytes
    expected_pool_sha256: str
    pool: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    source_rows: list[dict[str, Any]] | None = None


def _dependency_contract_is_exact(
    c13_localizer: types.ModuleType,
    c13_context: Any,
    c9_adapter: types.ModuleType,
    c9_context: Any,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> bool:
    return bool(
        type(c13_localizer) is types.ModuleType
        and type(c9_adapter) is types.ModuleType
        and type(c6_hypothesis) is types.ModuleType
        and type(builder) is types.ModuleType
        and type(clusterer) is types.ModuleType
        and c13_localizer.ATTEMPT_ORDINAL == 13
        and tuple(c13_localizer.STAGE_CONTRACT[:6]) == C13_PREFIX_STAGES
        and c13_localizer.STAGE_CONTRACT[C13_GEOMETRY_STAGE_INDEX]
        == "GEOMETRY_ALIAS_EQUIVALENCE"
        and c9_adapter.ATTEMPT_ORDINAL == 9
        and c9_adapter.SCIENTIFIC_HYPOTHESIS_ID == SCIENTIFIC_HYPOTHESIS_ID
        and c9_adapter.HYPOTHESIS_SOURCE_SHA256 == FIXED_BINDINGS["c6_source"]["sha256"]
        and c9_adapter.BUILDER_SOURCE_SHA256 == FIXED_BINDINGS["builder_source"]["sha256"]
        and c9_adapter.CLUSTER_SOURCE_SHA256 == FIXED_BINDINGS["clusterer_source"]["sha256"]
        and c6_hypothesis.ATTEMPT_ORDINAL == 6
        and c6_hypothesis.MAX_ADDITIONS_PER_CASE == 1
        and type(c13_context) is c13_localizer.LocalizerContext
        and type(c9_context) is c9_adapter.AdapterContext
        and c9_context.hypothesis is c6_hypothesis
        and c9_context.builder is builder
        and c9_context.clusterer is clusterer
        and c9_context.allowed_source_kinds == c6_hypothesis.ALLOWED_SOURCE_KINDS
    )


def _make_context_record(
    *,
    c13_localizer: types.ModuleType,
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
    bindings: Mapping[str, Mapping[str, Any]],
) -> _ContextRecord:
    if _binding_projection(bindings) != FIXED_BINDINGS:
        raise ValueError("Candidate15 source binding map mismatch")
    if any(
        type(module) is not types.ModuleType
        for module in (c13_localizer, c9_adapter, c6_hypothesis, builder, clusterer)
    ):
        raise ValueError("Candidate15 requires exact dependency modules")
    c13_exports = vars(c13_localizer)
    if (
        type(c13_exports) is not dict
        or type(c13_exports.get("make_verified_context")) is not types.FunctionType
        or type(c13_exports.get("localize_pool_adapter_contract")) is not types.FunctionType
        or type(c13_exports.get("aggregate_pool_adapter_receipts")) is not types.FunctionType
        or type(c13_exports.get("FIXED_BINDINGS")) is not dict
    ):
        raise ValueError("Candidate15 C13 public dependency exports mismatch")
    try:
        c13_context = c13_localizer.make_verified_context(
            c9_adapter=c9_adapter,
            c6_hypothesis=c6_hypothesis,
            builder=builder,
            clusterer=clusterer,
            bindings=c13_localizer.FIXED_BINDINGS,
        )
        c9_context = c9_adapter.make_verified_context(
            hypothesis=c6_hypothesis,
            builder=builder,
            clusterer=clusterer,
            hypothesis_source_sha256=FIXED_BINDINGS["c6_source"]["sha256"],
            builder_source_sha256=FIXED_BINDINGS["builder_source"]["sha256"],
            cluster_source_sha256=FIXED_BINDINGS["clusterer_source"]["sha256"],
        )
    except Exception:
        raise ValueError("Candidate15 dependency context construction failed") from None
    if not _dependency_contract_is_exact(
        c13_localizer,
        c13_context,
        c9_adapter,
        c9_context,
        c6_hypothesis,
        builder,
        clusterer,
    ):
        raise ValueError("Candidate15 dependency contract mismatch")
    c9_adapter._validate_context(c9_context)
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in sorted(FIXED_BINDINGS.items())
        }
    )
    return _ContextRecord(
        c13_localizer=c13_localizer,
        c13_context=c13_context,
        c9_adapter=c9_adapter,
        c9_context=c9_context,
        c6_hypothesis=c6_hypothesis,
        builder=builder,
        clusterer=clusterer,
        bindings=immutable_bindings,
        binding_commitment_sha256=sha256_bytes(canonical_bytes(FIXED_BINDINGS)),
        dependency_symbol_snapshot=_dependency_symbol_snapshot(
            c13_localizer, c9_adapter, c6_hypothesis, builder, clusterer
        ),
        dependency_constant_commitment_sha256=_dependency_constant_commitment(
            c13_localizer, c9_adapter, c6_hypothesis, builder
        ),
        local_function_snapshot=_local_function_snapshot(),
        local_constant_commitment_sha256=_local_constant_commitment(),
    )


def _context_projection(context: _ContextRecord) -> tuple[Any, ...]:
    return (
        id(context),
        id(type(context)),
        id(context.c13_localizer),
        id(context.c13_context),
        id(context.c9_adapter),
        id(context.c9_context),
        id(context.c6_hypothesis),
        id(context.builder),
        id(context.clusterer),
        id(context.bindings),
        context.binding_commitment_sha256,
        id(context.dependency_symbol_snapshot),
        context.dependency_constant_commitment_sha256,
        id(context.local_function_snapshot),
        context.local_constant_commitment_sha256,
    )


def _validate_context_record(context: _ContextRecord, projection: tuple[Any, ...]) -> None:
    try:
        if type(context) is not _ContextRecord or type(projection) is not tuple:
            raise ValueError("context type mismatch")
        if _context_projection(context) != projection:
            raise ValueError("context issuance mismatch")
        if (
            _sealed_binding_projection(context.bindings) != FIXED_BINDINGS
            or context.binding_commitment_sha256 != sha256_bytes(canonical_bytes(FIXED_BINDINGS))
            or not _dependency_contract_is_exact(
                context.c13_localizer,
                context.c13_context,
                context.c9_adapter,
                context.c9_context,
                context.c6_hypothesis,
                context.builder,
                context.clusterer,
            )
            or context.dependency_symbol_snapshot
            != _dependency_symbol_snapshot(
                context.c13_localizer,
                context.c9_adapter,
                context.c6_hypothesis,
                context.builder,
                context.clusterer,
            )
            or context.dependency_constant_commitment_sha256
            != _dependency_constant_commitment(
                context.c13_localizer,
                context.c9_adapter,
                context.c6_hypothesis,
                context.builder,
            )
            or context.local_function_snapshot != _local_function_snapshot()
            or context.local_constant_commitment_sha256 != _local_constant_commitment()
        ):
            raise ValueError("context content mismatch")
        context.c9_adapter._validate_context(context.c9_context)
    except Exception:
        raise PermissionError("Candidate15 verified context is not intact") from None


def _strict_finite_number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_point(value: Any) -> tuple[float, float, float] | None:
    if type(value) is not list or len(value) != 3:
        return None
    values = tuple(_strict_finite_number(coordinate) for coordinate in value)
    if any(coordinate is None for coordinate in values):
        return None
    return values  # type: ignore[return-value]


def _subtract(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(left[i]) - float(right[i]) for i in range(3))  # type: ignore[return-value]


def _add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(left[i]) + float(right[i]) for i in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(component) * scalar for component in value)  # type: ignore[return-value]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(component) * float(component) for component in value))


def _conditioned_endpoints(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    c6_hypothesis: types.ModuleType,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if any(
        abs(component) > c6_hypothesis.MAX_ABS_COORDINATE_MM
        for point in (first, second)
        for component in point
    ):
        return None
    length = _norm(_subtract(second, first))
    if not (
        c6_hypothesis.MIN_CONDITIONED_LINE_LENGTH_MM
        <= length
        <= c6_hypothesis.MAX_CONDITIONED_LINE_LENGTH_MM
    ):
        return None
    return (first, second) if first <= second else (second, first)


def _equivalent(
    left: tuple[tuple[float, float, float], tuple[float, float, float]],
    right: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    return all(
        abs(left[endpoint][axis] - right[endpoint][axis])
        <= GEOMETRY_EQUIVALENCE_TOLERANCE_MM
        for endpoint in range(2)
        for axis in range(3)
    )


def _span_representation(
    row: dict[str, Any], c6_hypothesis: types.ModuleType
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    value = row.get("span_endpoints")
    if value is None or type(value) is not list or len(value) != 2:
        return None
    first = _finite_point(value[0])
    second = _finite_point(value[1])
    if first is None or second is None:
        return None
    return _conditioned_endpoints(first, second, c6_hypothesis)


def _line_representation(
    row: dict[str, Any], c6_hypothesis: types.ModuleType
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    first = _finite_point(row.get("line_start"))
    second = _finite_point(row.get("line_end"))
    if first is None or second is None:
        return None
    return _conditioned_endpoints(first, second, c6_hypothesis)


def _aux_representation(
    *,
    anchor: tuple[float, float, float],
    axis: tuple[float, float, float],
    span_mm: float,
    c6_hypothesis: types.ModuleType,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    axis_norm = _norm(axis)
    if axis_norm <= 1.0e-12 or span_mm <= 1.0e-9:
        return None
    half = _scale(axis, 0.5 * span_mm / axis_norm)
    return _conditioned_endpoints(
        _subtract(anchor, half), _add(anchor, half), c6_hypothesis
    )


def _complete_aux_representation(
    row: dict[str, Any], c6_hypothesis: types.ModuleType
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    anchor = _finite_point(row.get("anchor"))
    axis = _finite_point(row.get("axis_direction"))
    span_mm = _strict_finite_number(row.get("span_mm"))
    if anchor is None or axis is None or span_mm is None:
        return None
    if any(
        abs(component) > c6_hypothesis.MAX_ABS_COORDINATE_MM
        for point in (anchor, axis)
        for component in point
    ):
        return None
    return _aux_representation(
        anchor=anchor,
        axis=axis,
        span_mm=span_mm,
        c6_hypothesis=c6_hypothesis,
    )


def _partial_aux_representation(
    row: dict[str, Any],
    reference: tuple[tuple[float, float, float], tuple[float, float, float]],
    c6_hypothesis: types.ModuleType,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    delta = _subtract(reference[1], reference[0])
    reference_length = _norm(delta)
    reference_axis = _scale(delta, 1.0 / reference_length)
    reference_midpoint = _scale(_add(reference[0], reference[1]), 0.5)
    anchor = (
        _finite_point(row.get("anchor"))
        if row.get("anchor") is not None
        else reference_midpoint
    )
    axis = (
        _finite_point(row.get("axis_direction"))
        if row.get("axis_direction") is not None
        else reference_axis
    )
    span_mm = (
        _strict_finite_number(row.get("span_mm"))
        if row.get("span_mm") is not None
        else reference_length
    )
    if anchor is None or axis is None or span_mm is None:
        return None
    if any(
        abs(component) > c6_hypothesis.MAX_ABS_COORDINATE_MM
        for point in (anchor, axis)
        for component in point
    ):
        return None
    return _aux_representation(
        anchor=anchor,
        axis=axis,
        span_mm=span_mm,
        c6_hypothesis=c6_hypothesis,
    )


def _present_geometry_is_shape_type_finite(row: dict[str, Any]) -> bool:
    span = row.get("span_endpoints")
    if span is not None:
        if (
            type(span) is not list
            or len(span) != 2
            or _finite_point(span[0]) is None
            or _finite_point(span[1]) is None
        ):
            return False
    for field in ("line_start", "line_end", "anchor", "axis_direction"):
        if row.get(field) is not None and _finite_point(row.get(field)) is None:
            return False
    if row.get("span_mm") is not None and _strict_finite_number(row.get("span_mm")) is None:
        return False
    return True


def _present_geometry_is_conditioned(
    row: dict[str, Any], c6_hypothesis: types.ModuleType
) -> bool:
    for field in ("line_start", "line_end", "anchor", "axis_direction"):
        point = _finite_point(row.get(field)) if row.get(field) is not None else None
        if point is not None and any(
            abs(component) > c6_hypothesis.MAX_ABS_COORDINATE_MM
            for component in point
        ):
            return False
    span_value = row.get("span_endpoints")
    if span_value is not None and type(span_value) is list and len(span_value) == 2:
        first, second = _finite_point(span_value[0]), _finite_point(span_value[1])
        for point in (first, second):
            if point is not None and any(
                abs(component) > c6_hypothesis.MAX_ABS_COORDINATE_MM
                for component in point
            ):
                return False
        if first is not None and second is not None:
            if _conditioned_endpoints(first, second, c6_hypothesis) is None:
                return False
    line_start, line_end = _finite_point(row.get("line_start")), _finite_point(row.get("line_end"))
    if line_start is not None and line_end is not None:
        if _conditioned_endpoints(line_start, line_end, c6_hypothesis) is None:
            return False
    aux_key_present = any(field in row for field in ("anchor", "axis_direction", "span_mm"))
    aux_complete = all(row.get(field) is not None for field in ("anchor", "axis_direction", "span_mm"))
    anchor = _finite_point(row.get("anchor"))
    axis = _finite_point(row.get("axis_direction"))
    if axis is not None and _norm(axis) <= 1.0e-12:
        return False
    span_mm = _strict_finite_number(row.get("span_mm"))
    if span_mm is not None:
        if span_mm <= 1.0e-9:
            return False
    complete_aux_shape_valid = (
        aux_complete
        and anchor is not None
        and axis is not None
        and span_mm is not None
    )
    if complete_aux_shape_valid and _complete_aux_representation(row, c6_hypothesis) is None:
        return False
    if aux_key_present and not aux_complete:
        reference = _span_representation(row, c6_hypothesis)
        supplied_aux_shape_valid = bool(
            (row.get("anchor") is None or anchor is not None)
            and (row.get("axis_direction") is None or axis is not None)
            and (row.get("span_mm") is None or span_mm is not None)
        )
        if (
            reference is not None
            and supplied_aux_shape_valid
            and _partial_aux_representation(row, reference, c6_hypothesis) is None
        ):
            return False
    return True


def _geometry_violation_mask(
    rows: list[dict[str, Any]], c6_hypothesis: types.ModuleType
) -> int:
    mask = 0
    aux_fields = ("anchor", "axis_direction", "span_mm")
    for row in rows:
        span_present = row.get("span_endpoints") is not None
        line_start_present = row.get("line_start") is not None
        line_end_present = row.get("line_end") is not None
        aux_value_present = any(row.get(field) is not None for field in aux_fields)
        aux_key_present = any(field in row for field in aux_fields)
        aux_complete = all(row.get(field) is not None for field in aux_fields)
        if not (span_present or line_start_present or line_end_present or aux_value_present):
            mask |= 1 << 0
        if line_start_present is not line_end_present:
            mask |= 1 << 1
        if aux_key_present and not aux_complete and not span_present:
            mask |= 1 << 2
        if not _present_geometry_is_shape_type_finite(row):
            mask |= 1 << 3
        if not _present_geometry_is_conditioned(row, c6_hypothesis):
            mask |= 1 << 4
        if aux_key_present and not aux_complete and span_present:
            reference = _span_representation(row, c6_hypothesis)
            reconstructed = (
                _partial_aux_representation(row, reference, c6_hypothesis)
                if reference is not None
                else None
            )
            if reference is not None and reconstructed is not None and not _equivalent(
                reference, reconstructed
            ):
                mask |= 1 << 5
        try:
            ordered: list[
                tuple[tuple[float, float, float], tuple[float, float, float]]
            ] = []
            span = _span_representation(row, c6_hypothesis)
            line = _line_representation(row, c6_hypothesis)
            auxiliary = (
                _complete_aux_representation(row, c6_hypothesis)
                if aux_complete
                else None
            )
            if span is not None:
                ordered.append(span)
            if line is not None:
                ordered.append(line)
            if auxiliary is not None:
                ordered.append(auxiliary)
            if len(ordered) > 1 and any(
                not _equivalent(ordered[0], representation)
                for representation in ordered[1:]
            ):
                mask |= 1 << 6
        except Exception:
            mask |= 1 << 7
    return mask


def _decode_bound_pool(context: _ContextRecord, state: _PoolState) -> dict[str, Any]:
    if type(state.pool_bytes) is not bytes or type(state.expected_pool_sha256) is not str:
        raise TypeError("Candidate15 input binding requires exact bytes and string")
    expected = context.c9_adapter._require_sha(
        state.expected_pool_sha256, name="Candidate15 caller pool SHA"
    )
    if context.c9_adapter.sha256_bytes(state.pool_bytes) != expected:
        raise ValueError("Candidate15 caller pool differs from external binding")
    pool = context.c9_adapter._decode_pool(state.pool_bytes, name="Candidate15 caller pool")
    if type(pool) is not dict:
        raise ValueError("Candidate15 decoded pool must be an exact object")
    state.pool = pool
    raw_rows = pool.get("line_objects")
    if type(raw_rows) is list and all(type(row) is dict for row in raw_rows):
        state.rows = raw_rows
    return pool


def _reproduce_c13_prefix(context: _ContextRecord, state: _PoolState) -> bool:
    pool = state.pool
    if type(pool) is not dict:
        return False
    try:
        if (
            pool.get("schema_version") != context.c9_adapter.POOL_SCHEMA_VERSION
            or pool.get("algorithm") != context.c9_adapter.POOL_ALGORITHM
            or pool.get("inference_contract") != context.c6_hypothesis.POOL_INFERENCE_CONTRACT
            or pool.get("source_point_serialization") is not False
            or isinstance(pool.get("working_point_budget"), bool)
            or pool.get("working_point_budget") != context.c9_adapter.WORKING_POINT_BUDGET
        ):
            raise ValueError("header")
        rows = pool.get("line_objects")
        if type(rows) is not list:
            raise ValueError("rows")
        identities: set[str] = set()
        exact_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if type(row) is not dict:
                raise ValueError("row")
            identity = context.c9_adapter._line_object_identity(row, index=index)
            if identity in identities:
                raise ValueError("duplicate")
            identities.add(identity)
            exact_rows.append(row)
        state.rows = exact_rows
        context.c6_hypothesis._validate_no_forbidden_keys(pool, path="candidate pool")
        source_rows: list[dict[str, Any]] = []
        for index, row in enumerate(exact_rows):
            for key in row:
                if type(key) is not str:
                    raise ValueError("row key")
                if (
                    context.c9_adapter._source_alias_spelling_is_known(key)
                    and key not in context.c9_adapter.SOURCE_FIELDS
                ):
                    raise ValueError("source spelling")
            candidate_source = context.c9_adapter._strict_text(
                row.get("candidate_source"),
                name=f"Candidate15 line_objects[{index}].candidate_source",
            )
            output = dict(row)
            if candidate_source == context.c9_adapter.TARGET_CANDIDATE_SOURCE:
                representative = context.c9_adapter._strict_text(
                    row.get(context.c9_adapter.AUTHORITATIVE_SOURCE_FIELD),
                    name="Candidate15 authoritative source",
                )
                if representative not in context.c9_context.allowed_source_kinds:
                    raise ValueError("source")
                if context.c9_adapter.STALE_ALIAS_FIELD in row:
                    context.c9_adapter._strict_text(
                        row[context.c9_adapter.STALE_ALIAS_FIELD],
                        name="Candidate15 stale source alias",
                    )
                    del output[context.c9_adapter.STALE_ALIAS_FIELD]
                context.c9_adapter._validate_cluster_envelope(
                    row,
                    index=index,
                    representative_source=representative,
                    context=context.c9_context,
                )
            else:
                if candidate_source not in context.c9_context.allowed_source_kinds:
                    raise ValueError("source")
                for alias in (
                    context.c9_adapter.AUTHORITATIVE_SOURCE_FIELD,
                    context.c9_adapter.STALE_ALIAS_FIELD,
                ):
                    if alias in row:
                        value = context.c9_adapter._strict_text(
                            row[alias], name="Candidate15 source alias"
                        )
                        if value != candidate_source:
                            raise ValueError("source aliases")
            source_rows.append(output)
        state.source_rows = source_rows
        return True
    except Exception:
        return False


def _reproduce_c13_geometry_stage(context: _ContextRecord, state: _PoolState) -> bool:
    if type(state.source_rows) is not list or type(state.rows) is not list:
        return False
    try:
        for index, source_row in enumerate(state.source_rows):
            row = state.rows[index]
            output = dict(source_row)
            auxiliary_keys_present = any(
                field in row for field in context.c9_adapter.AUXILIARY_GEOMETRY_FIELDS
            )
            auxiliary_complete = all(
                row.get(field) is not None
                for field in context.c9_adapter.AUXILIARY_GEOMETRY_FIELDS
            )
            if auxiliary_keys_present and not auxiliary_complete:
                if row.get("span_endpoints") is None:
                    raise ValueError("incomplete geometry")
                context.c9_adapter._validate_partial_geometry_against_span(
                    row, index=index, context=context.c9_context
                )
                for field in context.c9_adapter.AUXILIARY_GEOMETRY_FIELDS:
                    output.pop(field, None)
            if context.c6_hypothesis._endpoints_from_row(
                output, name=f"Candidate15 line_objects[{index}]"
            ) is None:
                raise ValueError("missing geometry")
        return True
    except Exception:
        return False


def _exact_c9_accepts(context: _ContextRecord, state: _PoolState) -> bool:
    if type(state.pool_bytes) is not bytes or type(state.expected_pool_sha256) is not str:
        return False
    try:
        receipt = context.c9_adapter.adapt_pool_bytes(
            context.c9_context,
            pool_bytes=state.pool_bytes,
            expected_pool_sha256=state.expected_pool_sha256,
        )
        return type(receipt) is context.c9_adapter.AdaptedPoolBytes
    except Exception:
        return False


def _analyze_pool_record(
    context: _ContextRecord,
    projection: tuple[Any, ...],
    *,
    pool_bytes: bytes,
    expected_pool_sha256: str,
) -> tuple[tuple[bool, ...], tuple[bool, ...], int, bool, bool]:
    """Return private stages 0..8, mask, reproduced C13-stage6, exact-C9 outcome."""

    _validate_context_record(context, projection)
    state = _PoolState(pool_bytes=pool_bytes, expected_pool_sha256=expected_pool_sha256)
    reached = [False] * 9
    passed = [False] * 9
    reached[0] = True
    try:
        _decode_bound_pool(context, state)
        passed[0] = True
    except Exception:
        passed[0] = False
    prefix_pass = False
    if passed[0]:
        reached[1] = True
        prefix_pass = _reproduce_c13_prefix(context, state)
        passed[1] = prefix_pass
    violation_mask = 0
    if type(state.rows) is list:
        for index in range(2, 9):
            reached[index] = True
        try:
            violation_mask = _geometry_violation_mask(
                state.rows, context.c6_hypothesis
            )
        except Exception:
            violation_mask = 1 << 7
        passed[2] = not bool(violation_mask & (1 << 0))
        passed[3] = not bool(violation_mask & (1 << 1))
        passed[4] = not bool(violation_mask & (1 << 2))
        passed[5] = not bool(violation_mask & (1 << 3))
        passed[6] = not bool(violation_mask & (1 << 4))
        passed[7] = not bool(violation_mask & (1 << 5))
        passed[8] = not bool(violation_mask & (1 << 6))
    c13_stage6_pass = prefix_pass and _reproduce_c13_geometry_stage(context, state)
    c9_accepts = _exact_c9_accepts(context, state)
    _validate_context_record(context, projection)
    return tuple(reached), tuple(passed), violation_mask, c13_stage6_pass, c9_accepts


RESULT_KEYS = frozenset(
    {
        "adapter_or_hypothesis_semantics_modified_by_core",
        "aggregate_stage6_violation_coverage_for_cohort",
        "artifact_discovery_or_open_performed_by_core",
        "c13_aggregate_called_exactly_once_by_core",
        "c13_aggregate_stage6_parity_for_cohort",
        "c13_receipts_issued_internally_on_identical_inputs",
        "check_code",
        "exact36_ordered_receipts_required",
        "exact_c9_full_outcome_parity_for_cohort",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_geometry_values_published",
        "input_origin_authenticated_by_core",
        "kind",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "per_pool_c13_stage6_parity_claimed",
        "per_pool_violation_exhaustiveness_claimed",
        "public_result_is_aggregate_only",
        "real_or_control_input_opened_by_core",
        "rule_selected",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
        "violation_contract_commitment_sha256",
        "violation_or_bitset",
    }
)


def _validate_exact_c14_c13_aggregate(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or set(value) != set(EXPECTED_C14_C13_AGGREGATE):
        raise ValueError("Candidate15 C13 public aggregate schema mismatch")
    for key, expected in EXPECTED_C14_C13_AGGREGATE.items():
        actual = dict.__getitem__(value, key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("Candidate15 did not reproduce the exact C14 C13 aggregate")


def _validate_public_result(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or set(value) != RESULT_KEYS:
        raise ValueError("Candidate15 public result schema mismatch")
    reached = value["stage_reached_all_bitset"]
    passed = value["stage_passed_all_bitset"]
    violations = value["violation_or_bitset"]
    false_fields = (
        "adapter_or_hypothesis_semantics_modified_by_core",
        "artifact_discovery_or_open_performed_by_core",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_geometry_values_published",
        "input_origin_authenticated_by_core",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "per_pool_c13_stage6_parity_claimed",
        "per_pool_violation_exhaustiveness_claimed",
        "real_or_control_input_opened_by_core",
        "rule_selected",
    )
    true_fields = (
        "c13_aggregate_called_exactly_once_by_core",
        "c13_receipts_issued_internally_on_identical_inputs",
        "exact36_ordered_receipts_required",
        "public_result_is_aggregate_only",
    )
    if (
        any(value[field] is not False for field in false_fields)
        or any(value[field] is not True for field in true_fields)
        or any(
            type(value[field]) is not bool
            for field in (
                "aggregate_stage6_violation_coverage_for_cohort",
                "c13_aggregate_stage6_parity_for_cohort",
                "exact_c9_full_outcome_parity_for_cohort",
            )
        )
        or any(
            type(value[field]) is not str
            for field in (
                "check_code",
                "kind",
                "stage_contract_commitment_sha256",
                "stage_passed_all_bitset",
                "stage_reached_all_bitset",
                "status",
                "violation_contract_commitment_sha256",
                "violation_or_bitset",
            )
        )
        or value["kind"] != RESULT_KIND
        or len(reached) != len(STAGE_CONTRACT)
        or len(passed) != len(STAGE_CONTRACT)
        or len(violations) != len(VIOLATION_CONTRACT)
        or set(reached + passed + violations) - {"0", "1"}
        or any(p == "1" and r != "1" for r, p in zip(reached, passed))
        or reached[-3:] != "111"
        or value["stage_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(STAGE_CONTRACT)))
        or value["violation_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(VIOLATION_CONTRACT)))
        or value["c13_aggregate_stage6_parity_for_cohort"] is not (passed[9] == "1")
        or value["exact_c9_full_outcome_parity_for_cohort"] is not (passed[10] == "1")
        or value["aggregate_stage6_violation_coverage_for_cohort"] is not (passed[11] == "1")
        or (violations[7] == "1") is not (
            passed[9] == "0" or passed[10] == "0" or passed[11] == "0"
        )
    ):
        raise ValueError("Candidate15 public result invariant mismatch")
    engineering_equivalent = passed[-3:] == "111"
    diagnostic_clear = passed[:9] == "1" * 9
    if not engineering_equivalent:
        expected = (
            "C15_AGGREGATE_PARITY_OR_COVERAGE_MISMATCH",
            "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE",
        )
    elif diagnostic_clear:
        expected = (
            "C15_GEOMETRY_REPRESENTATION_CONTRACT_CLEAR",
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        )
    else:
        expected = (
            "C15_GEOMETRY_REPRESENTATION_FAILURE_LOCALIZED",
            "SOURCE_POOL_GEOMETRY_INCOMPATIBLE_NONAUTHORIZING",
        )
    if (value["check_code"], value["status"]) != expected:
        raise ValueError("Candidate15 public result status mismatch")


def _build_public_result(
    *,
    reached_values: tuple[bool, ...],
    passed_values: tuple[bool, ...],
    violation_mask: int,
) -> dict[str, Any]:
    if (
        type(reached_values) is not tuple
        or type(passed_values) is not tuple
        or len(reached_values) != len(STAGE_CONTRACT)
        or len(passed_values) != len(STAGE_CONTRACT)
        or any(type(value) is not bool for value in reached_values + passed_values)
        or type(violation_mask) is not int
        or not 0 <= violation_mask < (1 << len(VIOLATION_CONTRACT))
    ):
        raise RuntimeError("Candidate15 aggregate projection mismatch")
    reached = "".join("1" if value else "0" for value in reached_values)
    passed = "".join("1" if value else "0" for value in passed_values)
    violations = "".join(
        "1" if violation_mask & (1 << index) else "0"
        for index in range(len(VIOLATION_CONTRACT))
    )
    engineering_equivalent = passed[-3:] == "111"
    diagnostic_clear = passed[:9] == "1" * 9
    if not engineering_equivalent:
        check_code = "C15_AGGREGATE_PARITY_OR_COVERAGE_MISMATCH"
        status = "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE"
    elif diagnostic_clear:
        check_code = "C15_GEOMETRY_REPRESENTATION_CONTRACT_CLEAR"
        status = "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    else:
        check_code = "C15_GEOMETRY_REPRESENTATION_FAILURE_LOCALIZED"
        status = "SOURCE_POOL_GEOMETRY_INCOMPATIBLE_NONAUTHORIZING"
    result = {
        "adapter_or_hypothesis_semantics_modified_by_core": False,
        "aggregate_stage6_violation_coverage_for_cohort": passed_values[11],
        "artifact_discovery_or_open_performed_by_core": False,
        "c13_aggregate_called_exactly_once_by_core": True,
        "c13_aggregate_stage6_parity_for_cohort": passed_values[9],
        "c13_receipts_issued_internally_on_identical_inputs": True,
        "check_code": check_code,
        "exact36_ordered_receipts_required": True,
        "exact_c9_full_outcome_parity_for_cohort": passed_values[10],
        "fit_performed": False,
        "full28_opened": False,
        "holdout_opened": False,
        "input_or_geometry_values_published": False,
        "input_origin_authenticated_by_core": False,
        "kind": RESULT_KIND,
        "label_or_evaluator_opened": False,
        "metric_computed": False,
        "performance_tested": False,
        "per_pool_c13_stage6_parity_claimed": False,
        "per_pool_violation_exhaustiveness_claimed": False,
        "public_result_is_aggregate_only": True,
        "real_or_control_input_opened_by_core": False,
        "rule_selected": False,
        "stage_contract_commitment_sha256": sha256_bytes(canonical_bytes(list(STAGE_CONTRACT))),
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": status,
        "violation_contract_commitment_sha256": sha256_bytes(
            canonical_bytes(list(VIOLATION_CONTRACT))
        ),
        "violation_or_bitset": violations,
    }
    _validate_public_result(result)
    return result


def _build_opaque_capability_runtime() -> tuple[Any, ...]:
    context_registry: dict[object, tuple[Any, ...]] = {}
    context_marker_registry: dict[object, tuple[Any, ...]] = {}
    receipt_registry: dict[object, tuple[Any, ...]] = {}
    receipt_marker_registry: dict[object, tuple[Any, ...]] = {}
    paired_c13_registry: dict[int, tuple[Any, ...]] = {}
    runtime_token = object()
    capability_key = object()
    activated = False
    runtime_seal: tuple[Any, ...] | None = None

    def _blocked_serialization(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("Candidate15 opaque handles are not serializable or copyable")

    class LocalizerContext:
        __slots__ = ()
        __copy__ = _blocked_serialization
        __deepcopy__ = _blocked_serialization
        __getstate__ = _blocked_serialization
        __reduce__ = _blocked_serialization
        __reduce_ex__ = _blocked_serialization

        def __repr__(self) -> str:
            return "<Candidate15OpaqueContext>"

        def __str__(self) -> str:
            return "<Candidate15OpaqueContext>"

    class _GeometryDiagnosticReceipt:
        __slots__ = ()
        __copy__ = _blocked_serialization
        __deepcopy__ = _blocked_serialization
        __getstate__ = _blocked_serialization
        __reduce__ = _blocked_serialization
        __reduce_ex__ = _blocked_serialization

        def __repr__(self) -> str:
            return "<Candidate15OpaqueGeometryReceipt>"

        def __str__(self) -> str:
            return "<Candidate15OpaqueGeometryReceipt>"

    def _closure_cell_ids(function: Callable[..., Any]) -> tuple[int, ...]:
        return tuple(id(cell) for cell in (function.__closure__ or ()))

    def _context_tag(
        handle: LocalizerContext,
        marker: object,
        record: _ContextRecord,
        projection: tuple[Any, ...],
        next_slot: int,
        state: str,
    ) -> str:
        material = (
            "CANDIDATE15_CONTEXT_CAPABILITY_V1",
            id(capability_key),
            id(runtime_token),
            id(handle),
            id(marker),
            id(record),
            id(projection),
            next_slot,
            state,
            record.binding_commitment_sha256,
            record.dependency_constant_commitment_sha256,
            record.local_constant_commitment_sha256,
        )
        return hashlib.sha256(repr(material).encode("ascii")).hexdigest()

    def _receipt_tag(
        handle: _GeometryDiagnosticReceipt,
        marker: object,
        context: LocalizerContext,
        slot: int,
        reached: tuple[bool, ...],
        passed: tuple[bool, ...],
        violation_mask: int,
        c13_stage6_pass: bool,
        c9_accepts: bool,
        c13_receipt: object,
    ) -> str:
        material = (
            "CANDIDATE15_GEOMETRY_RECEIPT_CAPABILITY_V1",
            id(capability_key),
            id(runtime_token),
            id(handle),
            id(marker),
            id(context),
            slot,
            "".join("1" if value else "0" for value in reached),
            "".join("1" if value else "0" for value in passed),
            violation_mask,
            c13_stage6_pass,
            c9_accepts,
            id(c13_receipt),
        )
        return hashlib.sha256(repr(material).encode("ascii")).hexdigest()

    def _context_entry_is_valid(handle: object, entry: object) -> bool:
        if (
            type(handle) is not LocalizerContext
            or type(entry) is not tuple
            or len(entry) != 9
            or entry[0] is not handle
            or type(entry[1]) is not object
            or type(entry[2]) is not _ContextRecord
            or type(entry[3]) is not tuple
            or type(entry[4]) is not int
            or not 0 <= entry[4] <= 36
            or type(entry[5]) is not str
            or entry[5] not in _CONTEXT_STATES
            or (entry[5] in (_CONTEXT_STATE_IN_FLIGHT, _CONTEXT_STATE_TERMINAL)
                and entry[4] >= 36)
            or (entry[5] == _CONTEXT_STATE_AGGREGATED and entry[4] != 36)
            or type(entry[6]) is not str
            or entry[7] is not runtime_token
            or entry[8] is not capability_key
        ):
            return False
        reverse = context_marker_registry.get(entry[1])
        return bool(
            type(reverse) is tuple
            and len(reverse) == 3
            and reverse[0] is handle
            and reverse[1] == entry[6]
            and reverse[2] is runtime_token
            and entry[6]
            == _context_tag(handle, entry[1], entry[2], entry[3], entry[4], entry[5])
        )

    def _receipt_entry_is_valid(handle: object, entry: object) -> bool:
        if (
            type(handle) is not _GeometryDiagnosticReceipt
            or type(entry) is not tuple
            or len(entry) != 13
            or entry[0] is not handle
            or type(entry[1]) is not object
            or type(entry[2]) is not LocalizerContext
            or type(entry[3]) is not int
            or not 0 <= entry[3] < 36
            or type(entry[4]) is not tuple
            or type(entry[5]) is not tuple
            or len(entry[4]) != 9
            or len(entry[5]) != 9
            or any(type(value) is not bool for value in entry[4] + entry[5])
            or type(entry[6]) is not int
            or not 0 <= entry[6] < (1 << 8)
            or type(entry[7]) is not bool
            or type(entry[8]) is not bool
            or type(entry[10]) is not str
            or entry[11] is not runtime_token
            or entry[12] is not capability_key
        ):
            return False
        context_entry = context_registry.get(entry[2])
        if not _context_entry_is_valid(entry[2], context_entry):
            return False
        reverse = receipt_marker_registry.get(entry[1])
        paired = paired_c13_registry.get(id(entry[9]))
        return bool(
            type(reverse) is tuple
            and len(reverse) == 3
            and reverse[0] is handle
            and reverse[1] == entry[10]
            and reverse[2] is runtime_token
            and type(paired) is tuple
            and len(paired) == 7
            and paired[0] is entry[9]
            and paired[1] is handle
            and paired[2] is entry[1]
            and paired[3] is entry[2]
            and paired[4] == entry[3]
            and paired[5] == entry[10]
            and paired[6] is runtime_token
            and entry[10]
            == _receipt_tag(
                handle,
                entry[1],
                entry[2],
                entry[3],
                entry[4],
                entry[5],
                entry[6],
                entry[7],
                entry[8],
                entry[9],
            )
        )

    def _guard_runtime() -> None:
        if not activated or type(runtime_seal) is not tuple or len(runtime_seal) != 13:
            raise PermissionError("Candidate15 opaque runtime is not sealed")
        functions = (
            _blocked_serialization,
            _closure_cell_ids,
            _context_tag,
            _receipt_tag,
            _context_entry_is_valid,
            _receipt_entry_is_valid,
            _guard_runtime,
            make_verified_context,
            _resolve_context,
            localize_geometry_representation_contract,
            _resolve_receipt,
            aggregate_geometry_representation_receipts,
            _activate,
        )
        current_function_seal = tuple(
            (*_function_contract(function), _closure_cell_ids(function))
            for function in functions
        )
        if (
            current_function_seal != runtime_seal[0]
            or id(context_registry) != runtime_seal[1]
            or id(context_marker_registry) != runtime_seal[2]
            or id(receipt_registry) != runtime_seal[3]
            or id(receipt_marker_registry) != runtime_seal[4]
            or id(paired_c13_registry) != runtime_seal[5]
            or id(runtime_token) != runtime_seal[6]
            or id(capability_key) != runtime_seal[7]
            or runtime_seal[8] != (id(LocalizerContext), id(_GeometryDiagnosticReceipt))
            or runtime_seal[9]
            != (_class_contract(LocalizerContext), _class_contract(_GeometryDiagnosticReceipt))
            or runtime_seal[10] != (hashlib.sha256, json.loads, json.dumps, math.isfinite, math.sqrt)
            or runtime_seal[11]
            != (
                globals().get("LocalizerContext"),
                globals().get("_GeometryDiagnosticReceipt"),
                globals().get("make_verified_context"),
                globals().get("localize_geometry_representation_contract"),
                globals().get("aggregate_geometry_representation_receipts"),
            )
            or runtime_seal[12] is not runtime_token
            or len(context_registry) != len(context_marker_registry)
            or len(receipt_registry) != len(receipt_marker_registry)
            or len(receipt_registry) != len(paired_c13_registry)
            or any(not _context_entry_is_valid(handle, entry) for handle, entry in context_registry.items())
            or any(not _receipt_entry_is_valid(handle, entry) for handle, entry in receipt_registry.items())
            or sum(entry[4] for entry in context_registry.values()) != len(receipt_registry)
        ):
            raise PermissionError("Candidate15 opaque runtime integrity mismatch")

    def _replace_context_state(
        handle: LocalizerContext,
        entry: tuple[Any, ...],
        *,
        next_slot: int,
        state: str,
    ) -> None:
        tag = _context_tag(handle, entry[1], entry[2], entry[3], next_slot, state)
        replacement = (
            handle,
            entry[1],
            entry[2],
            entry[3],
            next_slot,
            state,
            tag,
            runtime_token,
            capability_key,
        )
        context_registry[handle] = replacement
        context_marker_registry[entry[1]] = (handle, tag, runtime_token)

    def make_verified_context(
        *,
        c13_localizer: types.ModuleType,
        c9_adapter: types.ModuleType,
        c6_hypothesis: types.ModuleType,
        builder: types.ModuleType,
        clusterer: types.ModuleType,
        bindings: Mapping[str, Mapping[str, Any]],
    ) -> LocalizerContext:
        _guard_runtime()
        record = _make_context_record(
            c13_localizer=c13_localizer,
            c9_adapter=c9_adapter,
            c6_hypothesis=c6_hypothesis,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )
        projection = _context_projection(record)
        handle = LocalizerContext()
        marker = object()
        tag = _context_tag(handle, marker, record, projection, 0, _CONTEXT_STATE_READY)
        context_registry[handle] = (
            handle,
            marker,
            record,
            projection,
            0,
            _CONTEXT_STATE_READY,
            tag,
            runtime_token,
            capability_key,
        )
        context_marker_registry[marker] = (handle, tag, runtime_token)
        _guard_runtime()
        return handle

    def _resolve_context(
        handle: LocalizerContext,
    ) -> tuple[_ContextRecord, tuple[Any, ...], tuple[Any, ...]]:
        _guard_runtime()
        if type(handle) is not LocalizerContext:
            raise PermissionError("Candidate15 context handle is not authentic")
        entry = context_registry.get(handle)
        if not _context_entry_is_valid(handle, entry):
            raise PermissionError("Candidate15 context handle is not authentic")
        _validate_context_record(entry[2], entry[3])
        return entry[2], entry[3], entry

    def localize_geometry_representation_contract(
        context: LocalizerContext,
        *,
        pool_bytes: bytes,
        expected_pool_sha256: str,
    ) -> _GeometryDiagnosticReceipt:
        _guard_runtime()
        if type(pool_bytes) is not bytes or type(expected_pool_sha256) is not str:
            raise TypeError("Candidate15 requires exact bytes and SHA string inputs")
        record, projection, context_entry = _resolve_context(context)
        if context_entry[5] != _CONTEXT_STATE_READY or context_entry[4] >= 36:
            raise PermissionError("Candidate15 context has no remaining receipt slot")
        slot = context_entry[4]
        # Reserve the context before entering C13.  Until every forward and
        # reverse registration has completed, no retry or aggregate can race
        # this slot.  Any BaseException after this point terminally consumes
        # the context and removes any partially registered C15 receipt.
        _replace_context_state(
            context,
            context_entry,
            next_slot=slot,
            state=_CONTEXT_STATE_IN_FLIGHT,
        )
        c13_receipt: Any = None
        receipt: _GeometryDiagnosticReceipt | None = None
        marker: object | None = None
        paired_identity: int | None = None
        try:
            c13_receipt = record.c13_localizer.localize_pool_adapter_contract(
                record.c13_context,
                pool_bytes=pool_bytes,
                expected_pool_sha256=expected_pool_sha256,
            )
            reached, passed, violation_mask, c13_stage6_pass, c9_accepts = (
                _analyze_pool_record(
                    record,
                    projection,
                    pool_bytes=pool_bytes,
                    expected_pool_sha256=expected_pool_sha256,
                )
            )
            receipt = _GeometryDiagnosticReceipt()
            marker = object()
            tag = _receipt_tag(
                receipt,
                marker,
                context,
                slot,
                reached,
                passed,
                violation_mask,
                c13_stage6_pass,
                c9_accepts,
                c13_receipt,
            )
            paired_identity = id(c13_receipt)
            if paired_identity in paired_c13_registry:
                raise RuntimeError("Candidate15 received a repeated opaque C13 receipt")
            receipt_registry[receipt] = (
                receipt,
                marker,
                context,
                slot,
                reached,
                passed,
                violation_mask,
                c13_stage6_pass,
                c9_accepts,
                c13_receipt,
                tag,
                runtime_token,
                capability_key,
            )
            receipt_marker_registry[marker] = (receipt, tag, runtime_token)
            paired_c13_registry[paired_identity] = (
                c13_receipt,
                receipt,
                marker,
                context,
                slot,
                tag,
                runtime_token,
            )
            _replace_context_state(
                context,
                context_entry,
                next_slot=slot + 1,
                state=_CONTEXT_STATE_READY,
            )
            _guard_runtime()
            return receipt
        except BaseException:
            if paired_identity is not None:
                paired = paired_c13_registry.get(paired_identity)
                if type(paired) is tuple and paired[0] is c13_receipt:
                    paired_c13_registry.pop(paired_identity, None)
            if marker is not None:
                receipt_marker_registry.pop(marker, None)
            if receipt is not None:
                receipt_registry.pop(receipt, None)
            _replace_context_state(
                context,
                context_entry,
                next_slot=slot,
                state=_CONTEXT_STATE_TERMINAL,
            )
            raise

    def _resolve_receipt(
        context: LocalizerContext,
        receipt: _GeometryDiagnosticReceipt,
        *,
        expected_slot: int,
    ) -> tuple[Any, ...]:
        _guard_runtime()
        if type(receipt) is not _GeometryDiagnosticReceipt or type(expected_slot) is not int:
            raise PermissionError("Candidate15 geometry receipt is not authentic")
        entry = receipt_registry.get(receipt)
        if (
            not _receipt_entry_is_valid(receipt, entry)
            or entry[2] is not context
            or entry[3] != expected_slot
        ):
            raise PermissionError("Candidate15 geometry receipt is not authentic or ordered")
        return entry

    def aggregate_geometry_representation_receipts(
        context: LocalizerContext,
        receipts: tuple[_GeometryDiagnosticReceipt, ...],
    ) -> dict[str, Any]:
        _guard_runtime()
        record, _projection, context_entry = _resolve_context(context)
        if context_entry[5] != _CONTEXT_STATE_READY:
            raise PermissionError("Candidate15 aggregate context is terminal or consumed")
        if context_entry[4] != 36:
            raise ValueError("Candidate15 aggregate requires all 36 issued slots")
        if type(receipts) is not tuple or len(receipts) != 36:
            raise ValueError("Candidate15 aggregate requires exactly 36 ordered receipts")
        if any(type(receipt) is not _GeometryDiagnosticReceipt for receipt in receipts):
            raise PermissionError("Candidate15 aggregate receipt type mismatch")
        if len({id(receipt) for receipt in receipts}) != 36:
            raise PermissionError("Candidate15 rejects duplicate geometry receipts")
        entries = tuple(
            _resolve_receipt(context, receipt, expected_slot=index)
            for index, receipt in enumerate(receipts)
        )
        c13_receipts = tuple(entry[9] for entry in entries)
        if len({id(receipt) for receipt in c13_receipts}) != 36:
            raise PermissionError("Candidate15 rejects duplicate paired C13 receipts")
        # Consume before the sole external aggregate call.  A failure is terminal;
        # this context can never replay or call the C13 aggregate a second time.
        _replace_context_state(
            context,
            context_entry,
            next_slot=36,
            state=_CONTEXT_STATE_AGGREGATED,
        )
        c13_result = record.c13_localizer.aggregate_pool_adapter_receipts(
            record.c13_context, c13_receipts
        )
        _validate_exact_c14_c13_aggregate(c13_result)
        c13_passed = c13_result["stage_passed_all_bitset"]
        c13_stage6_all = c13_passed[C13_GEOMETRY_STAGE_INDEX] == "1"
        reproduced_stage6_all = all(entry[7] for entry in entries)
        stage6_parity = c13_stage6_all is reproduced_stage6_all
        exact_c9_all_accept = all(entry[8] for entry in entries)
        c13_diagnostics_all_accept = c13_passed[:-1] == "1" * (len(c13_passed) - 1)
        exact_c9_parity = bool(
            c13_result["exact_c9_success_failure_equivalent_for_all"]
            and exact_c9_all_accept is c13_diagnostics_all_accept
        )
        coverage = all(
            not bool(entry[6] & (1 << 7))
            and (
                (not entry[5][1])
                or (entry[7] is ((entry[6] & 0x7F) == 0))
            )
            for entry in entries
        ) and stage6_parity
        violation_mask = 0
        for entry in entries:
            violation_mask |= entry[6]
        if not stage6_parity or not exact_c9_parity or not coverage:
            violation_mask |= 1 << 7
        reached = tuple(
            all(entry[4][index] for entry in entries) for index in range(9)
        ) + (True, True, True)
        passed = tuple(
            all(entry[5][index] for entry in entries) for index in range(9)
        ) + (stage6_parity, exact_c9_parity, coverage)
        result = _build_public_result(
            reached_values=reached,
            passed_values=passed,
            violation_mask=violation_mask,
        )
        _guard_runtime()
        return result

    def _activate() -> None:
        nonlocal activated, runtime_seal
        if activated or runtime_seal is not None:
            raise RuntimeError("Candidate15 opaque runtime is already activated")
        functions = (
            _blocked_serialization,
            _closure_cell_ids,
            _context_tag,
            _receipt_tag,
            _context_entry_is_valid,
            _receipt_entry_is_valid,
            _guard_runtime,
            make_verified_context,
            _resolve_context,
            localize_geometry_representation_contract,
            _resolve_receipt,
            aggregate_geometry_representation_receipts,
            _activate,
        )
        function_seal = tuple(
            (*_function_contract(function), _closure_cell_ids(function))
            for function in functions
        )
        runtime_seal = (
            function_seal,
            id(context_registry),
            id(context_marker_registry),
            id(receipt_registry),
            id(receipt_marker_registry),
            id(paired_c13_registry),
            id(runtime_token),
            id(capability_key),
            (id(LocalizerContext), id(_GeometryDiagnosticReceipt)),
            (_class_contract(LocalizerContext), _class_contract(_GeometryDiagnosticReceipt)),
            (hashlib.sha256, json.loads, json.dumps, math.isfinite, math.sqrt),
            (
                LocalizerContext,
                _GeometryDiagnosticReceipt,
                make_verified_context,
                localize_geometry_representation_contract,
                aggregate_geometry_representation_receipts,
            ),
            runtime_token,
        )
        activated = True

    return (
        LocalizerContext,
        _GeometryDiagnosticReceipt,
        make_verified_context,
        localize_geometry_representation_contract,
        aggregate_geometry_representation_receipts,
        _activate,
    )


(
    LocalizerContext,
    _GeometryDiagnosticReceipt,
    make_verified_context,
    localize_geometry_representation_contract,
    aggregate_geometry_representation_receipts,
    _activate_opaque_runtime,
) = _build_opaque_capability_runtime()
_activate_opaque_runtime()
del _activate_opaque_runtime
del _build_opaque_capability_runtime


def contract_declaration() -> dict[str, Any]:
    """Return the fixed, value-free, non-authorizing source contract."""

    return {
        "adapter_semantics_modified": False,
        "arbitrary_same_process_python_adversary_resistance_claimed": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "c13_aggregate_call_count_per_successful_context": 1,
        "c13_receipts_issued_internally_on_identical_inputs": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "exact36_ordered_slots_required": True,
        "filesystem_io_performed_by_core": False,
        "fixed_binding_commitment_sha256": sha256_bytes(canonical_bytes(FIXED_BINDINGS)),
        "full28_opened": False,
        "holdout_opened": False,
        "hypothesis_semantics_modified": False,
        "input_custody_authenticated_by_core": False,
        "input_domain_requires_exact_builtin_bytes_and_sha_string": True,
        "isolated_authenticated_worker_required": True,
        "label_or_evaluator_opened": False,
        "metric_or_fit_performed": False,
        "outer_controlled_import_and_preload_enforcement_required": True,
        "outer_no_untrusted_code_or_reflection_worker_required": True,
        "per_pool_c13_parity_claimed": False,
        "per_pool_violation_exhaustiveness_claimed": False,
        "performance_tested": False,
        "public_result_is_aggregate_only": True,
        "real_or_control_input_opened_by_core": False,
        "resource_bounds_enforced_by_core": False,
        "rule_selected": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "stage_contract": list(STAGE_CONTRACT),
        "tamper_evident_fresh_worker_runtime": True,
        "violation_contract": list(VIOLATION_CONTRACT),
    }
