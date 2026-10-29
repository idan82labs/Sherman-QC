"""Filesystem-blind Candidate13 pool-adapter stage localizer.

The caller supplies immutable candidate-pool bytes and an external SHA-256
binding.  This module does not discover or open any path.  It applies the exact
Candidate9 lossless projection and unchanged Candidate6 consumer through a
closed, value-free stage contract.  Detailed parsed values and receipts remain
private; the public result contains only fixed enums, aggregate bitsets, and
boundary booleans.

Stage ordering is Candidate13's fixed diagnostic precedence, not a claim about
Candidate9's internal exception order.  The always-reached final stage calls
exact Candidate9 normalize and adapt operations and proves only accept/reject
parity; receipt equivalence is additionally checked when Candidate9 accepts.

An outer launcher must authenticate this source, its tests, every fixed source
binding, and input custody before constructing a verified context.  This core
does not authorize real inputs or explain Candidate11's aggregate failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import types
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 13
ENGINEERING_ATTEMPT_ID = "CANDIDATE13_POOL_ADAPTER_STAGE_LOCALIZER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = "sherman_qc_f1_aug30_candidate13_pool_adapter_stage_result_v1"

FIXED_BINDINGS = {
    "builder": {
        "path": "scripts/build_f1_scan_only_candidate_pool.py",
        "sha256": "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2",
        "size_bytes": 57613,
    },
    "c11_core": {
        "path": "backend/f1_train18_candidate11_full_replay_diagnostic_v1.py",
        "sha256": "42c5ded051dd3b57dd48e3039a433afe63a0467edc13d4bd3b65620a6a279359",
        "size_bytes": 32967,
    },
    "c11_disposition": {
        "path": "research/f1_aug30_train18_candidate11_terminal_disposition_v1.json",
        "sha256": "58bd4fd18b1f6f4259e051ac1360cf6b204b4c339f1ff6a93e20a0f7d3f05bab",
        "size_bytes": 4150,
    },
    "c11_runner": {
        "path": "scripts/run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py",
        "sha256": "d18c583e176778974a9c85ede90ab287f23f5606e1856634c9a6c783c5effdd2",
        "size_bytes": 167000,
    },
    "c12_core": {
        "path": "backend/f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py",
        "sha256": "89712f14836e102ba8861b2068d464e98ccfdb670d3447201797373b9a2254f6",
        "size_bytes": 42434,
    },
    "c12_disposition": {
        "path": "research/f1_aug30_train18_candidate12_synthetic_contract_pass_disposition_v1.json",
        "sha256": "64c51a9421c39a0b3bdb54e8f30701ae58324127ed4dbb828dde7a83add44f75",
        "size_bytes": 3555,
    },
    "c12_tests": {
        "path": "tests/test_f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py",
        "sha256": "a3cf4013384bb9ef0e3077bb00f3ce7028dc1d3e7d0333bb9ef3382ca0e41340",
        "size_bytes": 16111,
    },
    "c6_hypothesis": {
        "path": "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
        "sha256": "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
        "size_bytes": 53837,
    },
    "c9_adapter": {
        "path": "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "sha256": "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
        "size_bytes": 22354,
    },
    "c9_adapter_tests": {
        "path": "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "sha256": "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
        "size_bytes": 16598,
    },
    "clusterer": {
        "path": "backend/f1_line_object_clustering.py",
        "sha256": "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c",
        "size_bytes": 6817,
    },
}

STAGE_CONTRACT = (
    "CONTEXT_INPUT_HASH_AND_JSON",
    "HEADER",
    "ROW_IDENTITY_ORDER_SNAPSHOT",
    "RECURSIVE_FORBIDDEN_FIELDS",
    "SOURCE_AUTHORITY_ALIAS_SPELLING",
    "CLUSTER_PROVENANCE_NAMESPACE",
    "GEOMETRY_ALIAS_EQUIVALENCE",
    "CANONICAL_PROJECTION_BIJECTION_ORDER",
    "UNCHANGED_C6_VALIDATION",
    "EXACT_C9_OUTCOME_AND_RECEIPT_EQUIVALENCE",
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
        raise TypeError("Candidate13 hashes exact immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _binding_projection(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Candidate13 source binding map mismatch")
    names = tuple(dict.keys(value))
    if any(type(name) is not str for name in names) or set(names) != set(FIXED_BINDINGS):
        raise ValueError("Candidate13 source binding map mismatch")
    projection: dict[str, Any] = {}
    for name in sorted(names):
        binding = dict.__getitem__(value, name)
        if type(binding) is not dict:
            raise ValueError("Candidate13 source binding schema mismatch")
        fields = tuple(dict.keys(binding))
        if (
            any(type(field) is not str for field in fields)
            or set(fields) != {"path", "sha256", "size_bytes"}
            or type(dict.__getitem__(binding, "path")) is not str
            or type(dict.__getitem__(binding, "sha256")) is not str
            or type(dict.__getitem__(binding, "size_bytes")) is not int
        ):
            raise ValueError("Candidate13 source binding schema mismatch")
        projection[name] = {
            "path": dict.__getitem__(binding, "path"),
            "sha256": dict.__getitem__(binding, "sha256"),
            "size_bytes": dict.__getitem__(binding, "size_bytes"),
        }
    return projection


def _sealed_binding_projection(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    proxy_type = type(types.MappingProxyType({}))
    if type(value) is not proxy_type:
        raise ValueError("Candidate13 sealed source binding map mismatch")
    names = tuple(value.keys())
    if any(type(name) is not str for name in names) or set(names) != set(FIXED_BINDINGS):
        raise ValueError("Candidate13 sealed source binding map mismatch")
    projection: dict[str, Any] = {}
    for name in sorted(names):
        binding = value[name]
        if type(binding) is not proxy_type:
            raise ValueError("Candidate13 sealed source binding schema mismatch")
        fields = tuple(binding.keys())
        if (
            any(type(field) is not str for field in fields)
            or set(fields) != {"path", "sha256", "size_bytes"}
            or type(binding["path"]) is not str
            or type(binding["sha256"]) is not str
            or type(binding["size_bytes"]) is not int
        ):
            raise ValueError("Candidate13 sealed source binding schema mismatch")
        projection[name] = dict(binding)
    return projection


def _frozen_value(value: Any) -> Any:
    if type(value) in (type(None), bool, int, float, str):
        return ("scalar", type(value).__name__, value)
    if type(value) in (tuple, list):
        return (
            "sequence",
            type(value).__name__,
            tuple(_frozen_value(child) for child in value),
        )
    if type(value) in (set, frozenset):
        children = [_frozen_value(child) for child in value]
        children.sort(key=canonical_bytes)
        return ("set", type(value).__name__, tuple(children))
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Candidate13 constant mapping keys must be exact strings")
        return (
            "mapping",
            tuple(
                (key, _frozen_value(child))
                for key, child in sorted(value.items())
            ),
        )
    raise TypeError(
        f"Candidate13 cannot freeze dependency value type {type(value).__name__}"
    )


def _function_contract(function: types.FunctionType) -> tuple[Any, ...]:
    if type(function) is not types.FunctionType:
        raise TypeError("Candidate13 controlled callable must be an exact function")
    return (
        function,
        function.__code__,
        _frozen_value(function.__defaults__ or ()),
        _frozen_value(function.__kwdefaults__ or {}),
    )


def _class_contract(value: type) -> tuple[Any, ...]:
    fields = tuple(sorted(getattr(value, "__dataclass_fields__", {})))
    parameters = getattr(value, "__dataclass_params__", None)
    parameter_contract = (
        None
        if parameters is None
        else (
            bool(parameters.init),
            bool(parameters.repr),
            bool(parameters.eq),
            bool(parameters.order),
            bool(parameters.unsafe_hash),
            bool(parameters.frozen),
        )
    )
    methods = tuple(
        (name, *_function_contract(child))
        for name, child in sorted(vars(value).items())
        if type(child) is types.FunctionType
    )
    return value, value.__module__, value.__name__, fields, parameter_contract, methods


def _module_symbol_snapshot(module: types.ModuleType) -> tuple[Any, ...]:
    if type(module) is not types.ModuleType:
        raise TypeError("Candidate13 dependency must be an exact module")
    symbols: list[tuple[Any, ...]] = []
    for name, value in sorted(vars(module).items()):
        if type(value) is types.FunctionType and value.__module__ == module.__name__:
            symbols.append(("function", name, *_function_contract(value)))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            symbols.append(("class", name, *_class_contract(value)))
    return tuple(symbols)


def _live_default_contract(value: Any) -> Any:
    if type(value) in (type(None), bool, int, float, str):
        return ("value", type(value), value)
    if type(value) in (tuple, list):
        return (
            type(value),
            tuple(_live_default_contract(child) for child in value),
        )
    if type(value) is dict:
        return (
            "dict",
            tuple(
                (key, _live_default_contract(child))
                for key, child in sorted(value.items())
            ),
        )
    return ("identity", id(value), type(value))


def _live_member_contract(value: Any) -> tuple[Any, ...]:
    if type(value) is types.FunctionType:
        return (
            "function",
            id(value),
            value.__code__,
            _live_default_contract(value.__defaults__ or ()),
            _live_default_contract(value.__kwdefaults__ or {}),
        )
    if isinstance(value, type):
        return (
            "class",
            id(value),
            value.__module__,
            value.__name__,
            tuple(sorted(getattr(value, "__dataclass_fields__", {}))),
        )
    if callable(value):
        return (
            "callable",
            id(value),
            type(value),
            getattr(value, "__module__", None),
            getattr(value, "__name__", None),
        )
    if type(value) in (type(None), bool, int, float, str):
        return ("value", type(value), value)
    return ("object", id(value), type(value))


def _resolve_live_member(root: Any, dotted_name: str) -> Any:
    value = root
    for component in dotted_name.split("."):
        value = getattr(value, component)
    return value


def _live_runtime_member_snapshot(
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> tuple[Any, ...]:
    specifications = (
        ("local_hashlib", hashlib, ("sha256",)),
        ("local_json", json, ("dumps", "loads")),
        ("c9_hashlib", c9_adapter.hashlib, ("sha256",)),
        (
            "c9_json",
            c9_adapter.json,
            ("JSONDecodeError", "dumps", "loads"),
        ),
        ("c9_re", c9_adapter.re, ("sub",)),
        ("c6_hashlib", c6_hypothesis.hashlib, ("sha256",)),
        (
            "c6_json",
            c6_hypothesis.json,
            ("JSONDecodeError", "dumps", "loads"),
        ),
        (
            "c6_math",
            c6_hypothesis.math,
            ("acos", "degrees", "isfinite", "sqrt"),
        ),
        ("c6_re", c6_hypothesis.re, ("compile", "split", "sub")),
        ("builder_hashlib", builder.hashlib, ("sha256",)),
        ("builder_json", builder.json, ("dumps", "loads")),
        ("builder_math", builder.math, ("inf", "isfinite")),
        (
            "builder_numpy",
            builder.np,
            (
                "abs",
                "arccos",
                "arctan2",
                "argmax",
                "argmin",
                "asarray",
                "clip",
                "column_stack",
                "cos",
                "count_nonzero",
                "cross",
                "degrees",
                "diff",
                "dot",
                "einsum",
                "eye",
                "flatnonzero",
                "float64",
                "floor",
                "int64",
                "isfinite",
                "lexsort",
                "linalg.eigh",
                "linalg.eigvalsh",
                "linalg.norm",
                "linspace",
                "max",
                "mean",
                "median",
                "mod",
                "ndarray",
                "pi",
                "ptp",
                "quantile",
                "r_",
                "radians",
                "rint",
                "sin",
                "sort",
                "sqrt",
                "sum",
                "unique",
                "zeros",
            ),
        ),
        ("cluster_math", clusterer.math, ("acos", "degrees")),
        (
            "cluster_numpy",
            clusterer.np,
            (
                "abs",
                "argmax",
                "asarray",
                "clip",
                "dot",
                "float64",
                "isfinite",
                "linalg.norm",
                "max",
                "mean",
                "ndarray",
            ),
        ),
    )
    return tuple(
        (
            label,
            root,
            tuple(
                (name, _live_member_contract(_resolve_live_member(root, name)))
                for name in names
            ),
        )
        for label, root, names in specifications
    )


@dataclass(frozen=True)
class _ContextRecord:
    c9_adapter: types.ModuleType
    c9_context: Any
    c6_hypothesis: types.ModuleType
    builder: types.ModuleType
    clusterer: types.ModuleType
    bindings: Mapping[str, Mapping[str, Any]]
    binding_commitment_sha256: str
    _dependency_symbol_snapshot: tuple[Any, ...]
    _dependency_constant_commitment_sha256: str
    _local_function_snapshot: tuple[Any, ...]
    _local_constant_commitment_sha256: str
    _stage_function_snapshot: tuple[Any, ...]


@dataclass
class _RecordState:
    pool_bytes: bytes
    expected_pool_sha256: str
    context_issuance_projection: tuple[Any, ...]
    context_validated: bool = False
    pool: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    input_identities: tuple[str, ...] | None = None
    source_rows: list[dict[str, Any]] | None = None
    normalized_rows: list[dict[str, Any]] | None = None
    projected: dict[str, Any] | None = None
    projected_bytes: bytes | None = None
    diagnostic_failed: bool = False
    receipt: Any = None


@dataclass(frozen=True)
class _StageReceipt:
    stage: str
    reached: bool
    passed: bool


def _dependency_modules(
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> tuple[types.ModuleType, ...]:
    return c9_adapter, c6_hypothesis, builder, clusterer


def _dependency_symbol_snapshot(
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> tuple[Any, ...]:
    modules = _dependency_modules(c9_adapter, c6_hypothesis, builder, clusterer)
    module_symbols = tuple(
        (module, module.__name__, _module_symbol_snapshot(module)) for module in modules
    )
    referenced_modules = (
        ("c9_context_sentinel", c9_adapter._CONTEXT_SENTINEL),
        ("c9_hashlib", c9_adapter.hashlib),
        ("c9_json", c9_adapter.json),
        ("c9_re", c9_adapter.re),
        ("c6_hashlib", c6_hypothesis.hashlib),
        ("c6_json", c6_hypothesis.json),
        ("c6_math", c6_hypothesis.math),
        ("c6_re", c6_hypothesis.re),
        ("c6_sha_pattern", c6_hypothesis._HEX64),
        ("builder_hashlib", builder.hashlib),
        ("builder_json", builder.json),
        ("builder_math", builder.math),
        ("builder_numpy", builder.np),
        ("cluster_math", clusterer.math),
        ("cluster_numpy", clusterer.np),
    )
    return (
        module_symbols,
        referenced_modules,
        _live_runtime_member_snapshot(
            c9_adapter, c6_hypothesis, builder, clusterer
        ),
    )


def _dependency_constant_projection(
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
) -> dict[str, Any]:
    return {
        "builder": {
            name: _frozen_value(getattr(builder, name))
            for name in BUILDER_RUNTIME_CONSTANTS
        },
        "c6": {
            name: _frozen_value(getattr(c6_hypothesis, name))
            for name in C6_RUNTIME_CONSTANTS
        },
        "c9": {
            name: _frozen_value(getattr(c9_adapter, name))
            for name in C9_RUNTIME_CONSTANTS
        },
    }


def _dependency_constant_commitment(
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE13_TRANSITIVE_DEPENDENCY_CONSTANTS_V1",
                "projection": _dependency_constant_projection(
                    c9_adapter, c6_hypothesis, builder
                ),
            }
        )
    )


def _local_function_snapshot() -> tuple[Any, ...]:
    snapshot: list[tuple[Any, ...]] = []
    for name, value in sorted(globals().items()):
        if type(value) is types.FunctionType and value.__module__ == __name__:
            snapshot.append(("function", name, *_function_contract(value)))
        elif isinstance(value, type) and value.__module__ == __name__:
            snapshot.append(("class", name, *_class_contract(value)))
    return tuple(snapshot)


def _local_constant_commitment() -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "attempt_ordinal": ATTEMPT_ORDINAL,
                "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
                "fixed_bindings": FIXED_BINDINGS,
                "result_kind": RESULT_KIND,
                "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
                "stage_contract": list(STAGE_CONTRACT),
            }
        )
    )


def _dependency_contract_is_exact(
    c9_adapter: types.ModuleType,
    c9_context: Any,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
) -> bool:
    return bool(
        type(c9_adapter) is types.ModuleType
        and type(c6_hypothesis) is types.ModuleType
        and type(builder) is types.ModuleType
        and type(clusterer) is types.ModuleType
        and c9_adapter.ATTEMPT_ORDINAL == 9
        and c9_adapter.SCIENTIFIC_HYPOTHESIS_ID == SCIENTIFIC_HYPOTHESIS_ID
        and c9_adapter.HYPOTHESIS_SOURCE_SHA256
        == FIXED_BINDINGS["c6_hypothesis"]["sha256"]
        and c9_adapter.BUILDER_SOURCE_SHA256
        == FIXED_BINDINGS["builder"]["sha256"]
        and c9_adapter.CLUSTER_SOURCE_SHA256
        == FIXED_BINDINGS["clusterer"]["sha256"]
        and c6_hypothesis.ATTEMPT_ORDINAL == 6
        and c6_hypothesis.MAX_ADDITIONS_PER_CASE == 1
        and c6_hypothesis.OWNERSHIP_TRANSVERSE_OFFSET_MM == 4.0
        and type(c9_context) is c9_adapter.AdapterContext
        and c9_context.hypothesis is c6_hypothesis
        and c9_context.builder is builder
        and c9_context.clusterer is clusterer
        and c9_context.allowed_source_kinds == c6_hypothesis.ALLOWED_SOURCE_KINDS
    )


def _context_fields_are_exact(context: _ContextRecord) -> bool:
    proxy_type = type(types.MappingProxyType({}))
    return bool(
        type(context) is _ContextRecord
        and type(context.c9_adapter) is types.ModuleType
        and type(context.c6_hypothesis) is types.ModuleType
        and type(context.builder) is types.ModuleType
        and type(context.clusterer) is types.ModuleType
        and type(context.bindings) is proxy_type
        and type(context.binding_commitment_sha256) is str
        and type(context._dependency_symbol_snapshot) is tuple
        and type(context._dependency_constant_commitment_sha256) is str
        and type(context._local_function_snapshot) is tuple
        and type(context._local_constant_commitment_sha256) is str
        and type(context._stage_function_snapshot) is tuple
        and type(context.c9_context) is context.c9_adapter.AdapterContext
        and type(context.c9_context.hypothesis) is types.ModuleType
        and type(context.c9_context.builder) is types.ModuleType
        and type(context.c9_context.clusterer) is types.ModuleType
        and type(context.c9_context.allowed_source_kinds) is frozenset
        and all(
            type(value) is str
            for value in context.c9_context.allowed_source_kinds
        )
        and type(context.c9_context._sentinel) is object
    )


def _context_issuance_projection(context: _ContextRecord) -> tuple[Any, ...]:
    return (
        id(context),
        id(type(context)),
        id(context.c9_adapter),
        id(context.c9_context),
        id(context.c6_hypothesis),
        id(context.builder),
        id(context.clusterer),
        id(context.bindings),
        id(context.binding_commitment_sha256),
        context.binding_commitment_sha256,
        id(context._dependency_symbol_snapshot),
        id(context._dependency_constant_commitment_sha256),
        context._dependency_constant_commitment_sha256,
        id(context._local_function_snapshot),
        id(context._local_constant_commitment_sha256),
        context._local_constant_commitment_sha256,
        id(context._stage_function_snapshot),
        id(type(context.c9_context)),
        id(context.c9_context.hypothesis),
        id(context.c9_context.builder),
        id(context.c9_context.clusterer),
        id(context.c9_context.allowed_source_kinds),
        tuple(sorted(context.c9_context.allowed_source_kinds)),
        id(context.c9_context._sentinel),
    )


def _make_context_record(
    *,
    c9_adapter: types.ModuleType,
    c6_hypothesis: types.ModuleType,
    builder: types.ModuleType,
    clusterer: types.ModuleType,
    bindings: Mapping[str, Mapping[str, Any]],
) -> _ContextRecord:
    if _binding_projection(bindings) != FIXED_BINDINGS:
        raise ValueError("Candidate13 source binding map mismatch")
    if any(
        type(module) is not types.ModuleType
        for module in (c9_adapter, c6_hypothesis, builder, clusterer)
    ):
        raise ValueError("Candidate13 requires exact dependency modules")
    c9_context = c9_adapter.make_verified_context(
        hypothesis=c6_hypothesis,
        builder=builder,
        clusterer=clusterer,
        hypothesis_source_sha256=FIXED_BINDINGS["c6_hypothesis"]["sha256"],
        builder_source_sha256=FIXED_BINDINGS["builder"]["sha256"],
        cluster_source_sha256=FIXED_BINDINGS["clusterer"]["sha256"],
    )
    if not _dependency_contract_is_exact(
        c9_adapter, c9_context, c6_hypothesis, builder, clusterer
    ):
        raise ValueError("Candidate13 dependency contract mismatch")
    c9_adapter._validate_context(c9_context)
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in sorted(FIXED_BINDINGS.items())
        }
    )
    context = _ContextRecord(
        c9_adapter=c9_adapter,
        c9_context=c9_context,
        c6_hypothesis=c6_hypothesis,
        builder=builder,
        clusterer=clusterer,
        bindings=immutable_bindings,
        binding_commitment_sha256=sha256_bytes(canonical_bytes(FIXED_BINDINGS)),
        _dependency_symbol_snapshot=_dependency_symbol_snapshot(
            c9_adapter, c6_hypothesis, builder, clusterer
        ),
        _dependency_constant_commitment_sha256=_dependency_constant_commitment(
            c9_adapter, c6_hypothesis, builder
        ),
        _local_function_snapshot=_local_function_snapshot(),
        _local_constant_commitment_sha256=_local_constant_commitment(),
        _stage_function_snapshot=_stage_function_snapshot(),
    )
    return context


def _validate_context_record(
    context: _ContextRecord, issued_projection: tuple[Any, ...]
) -> None:
    try:
        if (
            type(context) is not _ContextRecord
            or type(issued_projection) is not tuple
            or not _context_fields_are_exact(context)
            or _context_issuance_projection(context) != issued_projection
        ):
            raise ValueError("context issuance mismatch")
        intact = not (
            _sealed_binding_projection(context.bindings) != FIXED_BINDINGS
            or context.binding_commitment_sha256
            != sha256_bytes(canonical_bytes(FIXED_BINDINGS))
            or not _dependency_contract_is_exact(
                context.c9_adapter,
                context.c9_context,
                context.c6_hypothesis,
                context.builder,
                context.clusterer,
            )
            or context._dependency_symbol_snapshot
            != _dependency_symbol_snapshot(
                context.c9_adapter,
                context.c6_hypothesis,
                context.builder,
                context.clusterer,
            )
            or context._dependency_constant_commitment_sha256
            != _dependency_constant_commitment(
                context.c9_adapter, context.c6_hypothesis, context.builder
            )
            or context._local_function_snapshot != _local_function_snapshot()
            or context._local_constant_commitment_sha256
            != _local_constant_commitment()
            or context._stage_function_snapshot != _stage_function_snapshot()
        )
        if not intact:
            raise ValueError("context mismatch")
        context.c9_adapter._validate_context(context.c9_context)
    except Exception:
        raise PermissionError("Candidate13 verified context is not intact") from None


def _require_pool(state: _RecordState) -> dict[str, Any]:
    if type(state.pool) is not dict:
        raise RuntimeError("Candidate13 private pool state is unavailable")
    return state.pool


def _require_rows(state: _RecordState) -> list[dict[str, Any]]:
    if type(state.rows) is not list or any(type(row) is not dict for row in state.rows):
        raise RuntimeError("Candidate13 private row state is unavailable")
    return state.rows


def _stage_context_input_hash_json(
    context: _ContextRecord, state: _RecordState
) -> None:
    _validate_context_record(context, state.context_issuance_projection)
    state.context_validated = True
    if type(state.pool_bytes) is not bytes or type(state.expected_pool_sha256) is not str:
        raise TypeError("Candidate13 input binding requires exact bytes and string")
    expected = context.c9_adapter._require_sha(
        state.expected_pool_sha256, name="Candidate13 caller pool SHA"
    )
    if context.c9_adapter.sha256_bytes(state.pool_bytes) != expected:
        raise ValueError("Candidate13 caller pool differs from its external binding")
    pool = context.c9_adapter._decode_pool(
        state.pool_bytes, name="Candidate13 caller pool"
    )
    if type(pool) is not dict:
        raise ValueError("Candidate13 decoded pool must be an exact object")
    state.pool = pool


def _stage_header(context: _ContextRecord, state: _RecordState) -> None:
    pool = _require_pool(state)
    if (
        pool.get("schema_version") != context.c9_adapter.POOL_SCHEMA_VERSION
        or pool.get("algorithm") != context.c9_adapter.POOL_ALGORITHM
        or pool.get("inference_contract")
        != context.c6_hypothesis.POOL_INFERENCE_CONTRACT
        or pool.get("source_point_serialization") is not False
        or isinstance(pool.get("working_point_budget"), bool)
        or pool.get("working_point_budget")
        != context.c9_adapter.WORKING_POINT_BUDGET
    ):
        raise ValueError("Candidate13 pool header mismatch")


def _stage_row_identity_bijection_order(
    context: _ContextRecord, state: _RecordState
) -> None:
    rows = _require_pool(state).get("line_objects")
    if type(rows) is not list:
        raise ValueError("Candidate13 line_objects must be an exact list")
    normalized_rows: list[dict[str, Any]] = []
    identities: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise ValueError("Candidate13 line object must be an exact object")
        identity = context.c9_adapter._line_object_identity(row, index=index)
        if identity in seen:
            raise ValueError("Candidate13 rejects duplicate line-object identity")
        seen.add(identity)
        identities.append(identity)
        normalized_rows.append(row)
    if len(identities) != len(normalized_rows):
        raise AssertionError("Candidate13 input row bijection invariant failed")
    state.rows = normalized_rows
    state.input_identities = tuple(identities)


def _stage_recursive_forbidden_fields(
    context: _ContextRecord, state: _RecordState
) -> None:
    context.c6_hypothesis._validate_no_forbidden_keys(
        _require_pool(state), path="candidate pool"
    )


def _stage_source_authority_alias_spelling(
    context: _ContextRecord, state: _RecordState
) -> None:
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(_require_rows(state)):
        for key in row:
            if type(key) is not str:
                raise ValueError("Candidate13 row key must be an exact string")
            if (
                context.c9_adapter._source_alias_spelling_is_known(key)
                and key not in context.c9_adapter.SOURCE_FIELDS
            ):
                raise ValueError("Candidate13 unknown source alias spelling")
        candidate_source = context.c9_adapter._strict_text(
            row.get("candidate_source"),
            name=f"Candidate13 line_objects[{index}].candidate_source",
        )
        output = dict(row)
        if candidate_source == context.c9_adapter.TARGET_CANDIDATE_SOURCE:
            representative = context.c9_adapter._strict_text(
                row.get(context.c9_adapter.AUTHORITATIVE_SOURCE_FIELD),
                name="Candidate13 authoritative source",
            )
            if representative not in context.c9_context.allowed_source_kinds:
                raise ValueError("Candidate13 unknown authoritative source")
            if context.c9_adapter.STALE_ALIAS_FIELD in row:
                context.c9_adapter._strict_text(
                    row[context.c9_adapter.STALE_ALIAS_FIELD],
                    name="Candidate13 stale source alias",
                )
                del output[context.c9_adapter.STALE_ALIAS_FIELD]
        else:
            if candidate_source not in context.c9_context.allowed_source_kinds:
                raise ValueError("Candidate13 unknown candidate source")
            for alias in (
                context.c9_adapter.AUTHORITATIVE_SOURCE_FIELD,
                context.c9_adapter.STALE_ALIAS_FIELD,
            ):
                if alias in row:
                    value = context.c9_adapter._strict_text(
                        row[alias], name="Candidate13 source alias"
                    )
                    if value != candidate_source:
                        raise ValueError("Candidate13 ambiguous source aliases")
        output_rows.append(output)
    state.source_rows = output_rows


def _stage_cluster_provenance_namespace(
    context: _ContextRecord, state: _RecordState
) -> None:
    for index, row in enumerate(_require_rows(state)):
        candidate_source = context.c9_adapter._strict_text(
            row.get("candidate_source"), name="Candidate13 candidate source"
        )
        if candidate_source == context.c9_adapter.TARGET_CANDIDATE_SOURCE:
            representative = context.c9_adapter._strict_text(
                row.get(context.c9_adapter.AUTHORITATIVE_SOURCE_FIELD),
                name="Candidate13 representative source",
            )
            context.c9_adapter._validate_cluster_envelope(
                row,
                index=index,
                representative_source=representative,
                context=context.c9_context,
            )


def _stage_geometry_alias_equivalence(
    context: _ContextRecord, state: _RecordState
) -> None:
    if type(state.source_rows) is not list:
        raise RuntimeError("Candidate13 source projection state is unavailable")
    output_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(state.source_rows):
        row = _require_rows(state)[index]
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
                raise ValueError("Candidate13 incomplete nonredundant geometry")
            context.c9_adapter._validate_partial_geometry_against_span(
                row, index=index, context=context.c9_context
            )
            for field in context.c9_adapter.AUXILIARY_GEOMETRY_FIELDS:
                output.pop(field, None)
        if context.c6_hypothesis._endpoints_from_row(
            output, name=f"Candidate13 line_objects[{index}]"
        ) is None:
            raise ValueError("Candidate13 row lacks finite line geometry")
        output_rows.append(output)
    state.normalized_rows = output_rows


def _stage_canonical_projection(
    context: _ContextRecord, state: _RecordState
) -> None:
    pool = _require_pool(state)
    if type(state.normalized_rows) is not list or state.input_identities is None:
        raise RuntimeError("Candidate13 normalized projection state is unavailable")
    expected = dict(pool)
    expected["line_objects"] = state.normalized_rows
    projected = context.c9_adapter._authorized_projection(
        pool, context=context.c9_context
    )
    if type(projected) is not dict or projected != expected:
        raise AssertionError("Candidate13 staged projection differs from exact C9")
    output_rows = projected.get("line_objects")
    if type(output_rows) is not list or len(output_rows) != len(state.input_identities):
        raise AssertionError("Candidate13 projection row bijection changed")
    output_identities = tuple(
        context.c9_adapter._line_object_identity(row, index=index)
        for index, row in enumerate(output_rows)
    )
    if output_identities != state.input_identities:
        raise AssertionError("Candidate13 projection row order or identity changed")
    projected_bytes = context.c9_adapter.canonical_bytes(projected)
    if type(projected_bytes) is not bytes or not projected_bytes.endswith(b"\n"):
        raise AssertionError("Candidate13 projection is not canonical LF bytes")
    decoded = context.c9_adapter._decode_pool(
        projected_bytes, name="Candidate13 canonical projection"
    )
    if decoded != projected or context.c9_adapter.canonical_bytes(decoded) != projected_bytes:
        raise AssertionError("Candidate13 canonical projection round trip changed")
    state.projected = projected
    state.projected_bytes = projected_bytes


def _stage_unchanged_c6_validation(
    context: _ContextRecord, state: _RecordState
) -> None:
    if type(state.projected_bytes) is not bytes:
        raise RuntimeError("Candidate13 projected bytes are unavailable")
    pool, _candidates = context.c6_hypothesis._validate_pool_bytes(
        state.projected_bytes,
        expected_sha256=context.c6_hypothesis.sha256_bytes(state.projected_bytes),
    )
    if pool != state.projected:
        raise AssertionError("Candidate13 C6 pool projection changed")


def _stage_receipt_projection_equivalence(
    context: _ContextRecord, state: _RecordState
) -> None:
    normalized_ok = False
    normalized: bytes | None = None
    try:
        normalized = context.c9_adapter.normalize_pool_bytes(
            context.c9_context,
            pool_bytes=state.pool_bytes,
            expected_pool_sha256=state.expected_pool_sha256,
        )
        normalized_ok = True
    except Exception:
        normalized_ok = False
    receipt_ok = False
    receipt: Any = None
    try:
        receipt = context.c9_adapter.adapt_pool_bytes(
            context.c9_context,
            pool_bytes=state.pool_bytes,
            expected_pool_sha256=state.expected_pool_sha256,
        )
        receipt_ok = True
    except Exception:
        receipt_ok = False
    if normalized_ok is not receipt_ok:
        raise AssertionError("Candidate13 exact C9 normalize/adapt outcomes disagree")
    if not receipt_ok:
        if state.diagnostic_failed:
            return
        raise AssertionError("Candidate13 diagnostics accepted an exact C9 rejection")
    if state.diagnostic_failed:
        raise AssertionError("Candidate13 diagnostics rejected an exact C9 acceptance")
    if type(state.projected_bytes) is not bytes or type(normalized) is not bytes:
        raise RuntimeError("Candidate13 projected bytes are unavailable")
    if (
        type(receipt) is not context.c9_adapter.AdaptedPoolBytes
        or type(receipt.original_pool_sha256) is not str
        or type(receipt.adapted_pool_sha256) is not str
        or type(receipt.adapted_pool_bytes) is not bytes
        or receipt.original_pool_sha256 != state.expected_pool_sha256
        or normalized != state.projected_bytes
        or receipt.adapted_pool_bytes != state.projected_bytes
        or receipt.adapted_pool_sha256
        != context.c9_adapter.sha256_bytes(receipt.adapted_pool_bytes)
        or receipt.projection_byte_equivalent is not True
    ):
        raise AssertionError("Candidate13 exact C9 receipt contract changed")
    context.c9_adapter.verify_projection_byte_equivalence(
        context.c9_context,
        original_pool_bytes=state.pool_bytes,
        normalized_pool_bytes=receipt.adapted_pool_bytes,
    )
    state.receipt = receipt


_STAGE_FUNCTIONS: tuple[
    tuple[str, Callable[[_ContextRecord, _RecordState], None]], ...
] = (
    (STAGE_CONTRACT[0], _stage_context_input_hash_json),
    (STAGE_CONTRACT[1], _stage_header),
    (STAGE_CONTRACT[2], _stage_row_identity_bijection_order),
    (STAGE_CONTRACT[3], _stage_recursive_forbidden_fields),
    (STAGE_CONTRACT[4], _stage_source_authority_alias_spelling),
    (STAGE_CONTRACT[5], _stage_cluster_provenance_namespace),
    (STAGE_CONTRACT[6], _stage_geometry_alias_equivalence),
    (STAGE_CONTRACT[7], _stage_canonical_projection),
    (STAGE_CONTRACT[8], _stage_unchanged_c6_validation),
    (STAGE_CONTRACT[9], _stage_receipt_projection_equivalence),
)


def _stage_function_snapshot() -> tuple[Any, ...]:
    stages = tuple(stage for stage, _function in _STAGE_FUNCTIONS)
    if stages != STAGE_CONTRACT or len(set(stages)) != len(STAGE_CONTRACT):
        raise ValueError("Candidate13 stage/function contract mismatch")
    return tuple(
        (stage, function, *_function_contract(function))
        for stage, function in _STAGE_FUNCTIONS
    )


RESULT_KEYS = frozenset(
    {
        "adapter_or_hypothesis_semantics_modified_by_core",
        "artifact_discovery_or_open_performed_by_core",
        "builder_or_clusterer_invoked_by_core",
        "candidate11_actual_record_cause_resolved",
        "candidate9_internal_failure_stage_claimed",
        "check_code",
        "diagnostic_stage_precedence_is_candidate13_contract",
        "exact_c9_success_failure_equivalent_for_all",
        "first_failed_candidate13_diagnostic_stage",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_projection_bytes_published",
        "input_origin_authenticated_by_core",
        "kind",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "real_or_control_input_opened_by_core",
        "rule_selected",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)


def _expected_bit_contract(passed: str) -> tuple[str, str]:
    if type(passed) is not str or len(passed) != len(STAGE_CONTRACT):
        return "INVALID", "INVALID"
    diagnostic = passed[:-1]
    failed_index = next(
        (index for index, bit in enumerate(diagnostic) if bit == "0"), None
    )
    if failed_index is None:
        reached_prefix = "1" * len(diagnostic)
        failed_stage = "NONE" if passed[-1] == "1" else STAGE_CONTRACT[-1]
    else:
        if diagnostic != (
            "1" * failed_index + "0" * (len(diagnostic) - failed_index)
        ):
            return "INVALID", "INVALID"
        reached_prefix = "1" * (failed_index + 1) + "0" * (
            len(diagnostic) - failed_index - 1
        )
        failed_stage = STAGE_CONTRACT[failed_index]
    return reached_prefix + "1", failed_stage


def _validate_public_result(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or set(value) != RESULT_KEYS:
        raise ValueError("Candidate13 public result schema mismatch")
    reached = value["stage_reached_all_bitset"]
    passed = value["stage_passed_all_bitset"]
    expected_reached, expected_failed = _expected_bit_contract(passed)
    fixed_false = (
        "adapter_or_hypothesis_semantics_modified_by_core",
        "artifact_discovery_or_open_performed_by_core",
        "builder_or_clusterer_invoked_by_core",
        "candidate11_actual_record_cause_resolved",
        "candidate9_internal_failure_stage_claimed",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_projection_bytes_published",
        "input_origin_authenticated_by_core",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "real_or_control_input_opened_by_core",
        "rule_selected",
    )
    if (
        any(
            type(value[key]) is not str
            for key in (
                "check_code",
                "first_failed_candidate13_diagnostic_stage",
                "kind",
                "stage_contract_commitment_sha256",
                "stage_passed_all_bitset",
                "stage_reached_all_bitset",
                "status",
            )
        )
        or value["kind"] != RESULT_KIND
        or any(value[key] is not False for key in fixed_false)
        or value["diagnostic_stage_precedence_is_candidate13_contract"] is not True
        or type(value["exact_c9_success_failure_equivalent_for_all"]) is not bool
        or type(reached) is not str
        or type(passed) is not str
        or len(reached) != len(STAGE_CONTRACT)
        or len(passed) != len(STAGE_CONTRACT)
        or set(reached + passed) - {"0", "1"}
        or any(pass_bit == "1" and reach_bit != "1" for reach_bit, pass_bit in zip(reached, passed))
        or reached != expected_reached
        or value["first_failed_candidate13_diagnostic_stage"] != expected_failed
        or value["exact_c9_success_failure_equivalent_for_all"]
        is not (passed[-1] == "1")
        or value["stage_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(STAGE_CONTRACT)))
    ):
        raise ValueError("Candidate13 public result invariant mismatch")
    all_pass = passed == "1" * len(STAGE_CONTRACT)
    equivalent = value["exact_c9_success_failure_equivalent_for_all"]
    if not equivalent:
        expected = (
            "C13_LOCALIZER_EXACT_C9_OUTCOME_MISMATCH",
            "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE",
        )
    elif all_pass:
        expected = (
            "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL",
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        )
    else:
        expected = (
            "C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION",
            "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING",
        )
    if (value["check_code"], value["status"]) != expected:
        raise ValueError("Candidate13 result status mismatch")


def _build_public_result(
    *, reached_values: tuple[bool, ...], passed_values: tuple[bool, ...]
) -> dict[str, Any]:
    if (
        type(reached_values) is not tuple
        or type(passed_values) is not tuple
        or len(reached_values) != len(STAGE_CONTRACT)
        or len(passed_values) != len(STAGE_CONTRACT)
        or any(type(value) is not bool for value in reached_values + passed_values)
    ):
        raise RuntimeError("Candidate13 aggregate receipt projection mismatch")
    reached = "".join("1" if value else "0" for value in reached_values)
    passed = "".join("1" if value else "0" for value in passed_values)
    first_failed = next(
        (stage for stage, stage_passed in zip(STAGE_CONTRACT, passed_values) if not stage_passed),
        "NONE",
    )
    exact_outcome_equivalent = passed_values[-1]
    all_pass = passed == "1" * len(STAGE_CONTRACT)
    if not exact_outcome_equivalent:
        check_code = "C13_LOCALIZER_EXACT_C9_OUTCOME_MISMATCH"
        status = "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE"
    elif all_pass:
        check_code = "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL"
        status = "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
    else:
        check_code = "C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION"
        status = "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING"
    result = {
        "adapter_or_hypothesis_semantics_modified_by_core": False,
        "artifact_discovery_or_open_performed_by_core": False,
        "builder_or_clusterer_invoked_by_core": False,
        "candidate11_actual_record_cause_resolved": False,
        "candidate9_internal_failure_stage_claimed": False,
        "check_code": check_code,
        "diagnostic_stage_precedence_is_candidate13_contract": True,
        "exact_c9_success_failure_equivalent_for_all": exact_outcome_equivalent,
        "first_failed_candidate13_diagnostic_stage": first_failed,
        "fit_performed": False,
        "full28_opened": False,
        "holdout_opened": False,
        "input_or_projection_bytes_published": False,
        "input_origin_authenticated_by_core": False,
        "kind": RESULT_KIND,
        "label_or_evaluator_opened": False,
        "metric_computed": False,
        "performance_tested": False,
        "real_or_control_input_opened_by_core": False,
        "rule_selected": False,
        "stage_contract_commitment_sha256": sha256_bytes(
            canonical_bytes(list(STAGE_CONTRACT))
        ),
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": status,
    }
    _validate_public_result(result)
    return result


def _localize_pool_record(
    context: _ContextRecord,
    issued_context_projection: tuple[Any, ...],
    *,
    pool_bytes: bytes,
    expected_pool_sha256: str,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Run one record and return only private boolean vectors to the issuer."""

    _validate_context_record(context, issued_context_projection)
    state = _RecordState(
        pool_bytes=pool_bytes,
        expected_pool_sha256=expected_pool_sha256,
        context_issuance_projection=issued_context_projection,
    )
    receipts: list[_StageReceipt] = []
    failed = False
    for index, (stage, function) in enumerate(_STAGE_FUNCTIONS):
        is_oracle_stage = index == len(_STAGE_FUNCTIONS) - 1
        if failed and not is_oracle_stage:
            receipts.append(_StageReceipt(stage=stage, reached=False, passed=False))
            continue
        if is_oracle_stage:
            state.diagnostic_failed = failed
        passed = False
        try:
            function(context, state)
            passed = True
        except Exception:
            passed = False
            if not is_oracle_stage:
                failed = True
        receipts.append(_StageReceipt(stage=stage, reached=True, passed=passed))
    if (
        len(receipts) != len(STAGE_CONTRACT)
        or tuple(receipt.stage for receipt in receipts) != STAGE_CONTRACT
    ):
        raise RuntimeError("Candidate13 private stage receipt contract mismatch")
    _validate_context_record(context, issued_context_projection)
    return (
        tuple(receipt.reached for receipt in receipts),
        tuple(receipt.passed for receipt in receipts),
    )


def _build_opaque_capability_runtime() -> tuple[Any, ...]:
    context_registry: dict[object, tuple[Any, ...]] = {}
    context_marker_registry: dict[object, object] = {}
    receipt_registry: dict[object, tuple[Any, ...]] = {}
    receipt_marker_registry: dict[object, object] = {}
    runtime_token = object()
    capability_key = object()
    activated = False
    runtime_seal: tuple[Any, ...] | None = None

    def _blocked_serialization(*_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("Candidate13 opaque handles are not serializable or copyable")

    class LocalizerContext:
        __slots__ = ()
        __copy__ = _blocked_serialization
        __deepcopy__ = _blocked_serialization
        __getstate__ = _blocked_serialization
        __reduce__ = _blocked_serialization
        __reduce_ex__ = _blocked_serialization

        def __repr__(self) -> str:
            return "<Candidate13OpaqueContext>"

        def __str__(self) -> str:
            return "<Candidate13OpaqueContext>"

    class _PoolDiagnosticReceipt:
        __slots__ = ()
        __copy__ = _blocked_serialization
        __deepcopy__ = _blocked_serialization
        __getstate__ = _blocked_serialization
        __reduce__ = _blocked_serialization
        __reduce_ex__ = _blocked_serialization

        def __repr__(self) -> str:
            return "<Candidate13OpaquePoolReceipt>"

        def __str__(self) -> str:
            return "<Candidate13OpaquePoolReceipt>"

    def _closure_cell_ids(function: Callable[..., Any]) -> tuple[int, ...]:
        return tuple(id(cell) for cell in (function.__closure__ or ()))

    def _context_entry_tag(
        handle: LocalizerContext,
        marker: object,
        record: _ContextRecord,
        projection: tuple[Any, ...],
    ) -> str:
        material = (
            "CANDIDATE13_CONTEXT_CAPABILITY_V1",
            id(capability_key),
            id(runtime_token),
            id(handle),
            id(marker),
            id(record),
            id(projection),
            record.binding_commitment_sha256,
            record._dependency_constant_commitment_sha256,
            record._local_constant_commitment_sha256,
        )
        return hashlib.sha256(repr(material).encode("ascii")).hexdigest()

    def _receipt_entry_tag(
        handle: _PoolDiagnosticReceipt,
        marker: object,
        context: LocalizerContext,
        reached: tuple[bool, ...],
        passed: tuple[bool, ...],
    ) -> str:
        material = (
            "CANDIDATE13_POOL_RECEIPT_CAPABILITY_V1",
            id(capability_key),
            id(runtime_token),
            id(handle),
            id(marker),
            id(context),
            "".join("1" if value else "0" for value in reached),
            "".join("1" if value else "0" for value in passed),
        )
        return hashlib.sha256(repr(material).encode("ascii")).hexdigest()

    def _context_entry_is_valid(
        handle: object, entry: object
    ) -> bool:
        if (
            type(handle) is not LocalizerContext
            or type(entry) is not tuple
            or len(entry) != 6
            or entry[0] is not handle
            or type(entry[1]) is not object
            or type(entry[2]) is not _ContextRecord
            or type(entry[3]) is not tuple
            or type(entry[4]) is not str
            or entry[5] is not runtime_token
        ):
            return False
        reverse = context_marker_registry.get(entry[1])
        return bool(
            type(reverse) is tuple
            and len(reverse) == 2
            and reverse[0] is handle
            and type(reverse[1]) is str
            and reverse[1] == entry[4]
            and entry[4]
            == _context_entry_tag(handle, entry[1], entry[2], entry[3])
        )

    def _receipt_entry_is_valid(
        handle: object, entry: object
    ) -> bool:
        if (
            type(handle) is not _PoolDiagnosticReceipt
            or type(entry) is not tuple
            or len(entry) != 7
            or entry[0] is not handle
            or type(entry[1]) is not object
            or type(entry[2]) is not LocalizerContext
            or type(entry[3]) is not tuple
            or type(entry[4]) is not tuple
            or len(entry[3]) != len(STAGE_CONTRACT)
            or len(entry[4]) != len(STAGE_CONTRACT)
            or any(type(value) is not bool for value in entry[3] + entry[4])
            or type(entry[5]) is not str
            or entry[6] is not runtime_token
        ):
            return False
        reverse = receipt_marker_registry.get(entry[1])
        return bool(
            type(reverse) is tuple
            and len(reverse) == 2
            and reverse[0] is handle
            and type(reverse[1]) is str
            and reverse[1] == entry[5]
            and entry[5]
            == _receipt_entry_tag(
                handle, entry[1], entry[2], entry[3], entry[4]
            )
        )

    def _guard_runtime() -> None:
        if not activated or type(runtime_seal) is not tuple or len(runtime_seal) != 10:
            raise PermissionError("Candidate13 opaque runtime is not sealed")
        functions = (
            _blocked_serialization,
            _closure_cell_ids,
            _context_entry_tag,
            _receipt_entry_tag,
            _context_entry_is_valid,
            _receipt_entry_is_valid,
            _guard_runtime,
            make_verified_context,
            _resolve_context,
            localize_pool_adapter_contract,
            _resolve_receipt,
            aggregate_pool_adapter_receipts,
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
            or id(runtime_token) != runtime_seal[5]
            or id(capability_key) != runtime_seal[6]
            or runtime_seal[7]
            != (id(LocalizerContext), id(_PoolDiagnosticReceipt))
            or runtime_seal[8]
            != (
                _class_contract(LocalizerContext),
                _class_contract(_PoolDiagnosticReceipt),
            )
            or runtime_seal[9] != _live_member_contract(hashlib.sha256)
            or len(context_registry) != len(context_marker_registry)
            or len(receipt_registry) != len(receipt_marker_registry)
            or globals().get("LocalizerContext") is not LocalizerContext
            or globals().get("_PoolDiagnosticReceipt") is not _PoolDiagnosticReceipt
            or globals().get("make_verified_context") is not make_verified_context
            or globals().get("localize_pool_adapter_contract")
            is not localize_pool_adapter_contract
            or globals().get("aggregate_pool_adapter_receipts")
            is not aggregate_pool_adapter_receipts
        ):
            raise PermissionError("Candidate13 opaque runtime integrity mismatch")

    def make_verified_context(
        *,
        c9_adapter: types.ModuleType,
        c6_hypothesis: types.ModuleType,
        builder: types.ModuleType,
        clusterer: types.ModuleType,
        bindings: Mapping[str, Mapping[str, Any]],
    ) -> LocalizerContext:
        _guard_runtime()
        record = _make_context_record(
            c9_adapter=c9_adapter,
            c6_hypothesis=c6_hypothesis,
            builder=builder,
            clusterer=clusterer,
            bindings=bindings,
        )
        projection = _context_issuance_projection(record)
        handle = LocalizerContext()
        marker = object()
        tag = _context_entry_tag(handle, marker, record, projection)
        context_registry[handle] = (
            handle,
            marker,
            record,
            projection,
            tag,
            runtime_token,
        )
        context_marker_registry[marker] = (handle, tag)
        _guard_runtime()
        return handle

    def _resolve_context(
        handle: LocalizerContext,
    ) -> tuple[_ContextRecord, tuple[Any, ...]]:
        _guard_runtime()
        if type(handle) is not LocalizerContext:
            raise PermissionError("Candidate13 context handle is not authentic")
        entry = context_registry.get(handle)
        if not _context_entry_is_valid(handle, entry):
            raise PermissionError("Candidate13 context handle is not authentic")
        _validate_context_record(entry[2], entry[3])
        return entry[2], entry[3]

    def localize_pool_adapter_contract(
        context: LocalizerContext,
        *,
        pool_bytes: bytes,
        expected_pool_sha256: str,
    ) -> _PoolDiagnosticReceipt:
        _guard_runtime()
        record, projection = _resolve_context(context)
        reached, passed = _localize_pool_record(
            record,
            projection,
            pool_bytes=pool_bytes,
            expected_pool_sha256=expected_pool_sha256,
        )
        if (
            type(reached) is not tuple
            or type(passed) is not tuple
            or len(reached) != len(STAGE_CONTRACT)
            or len(passed) != len(STAGE_CONTRACT)
            or any(type(value) is not bool for value in reached + passed)
        ):
            raise RuntimeError("Candidate13 private diagnostic vector mismatch")
        receipt = _PoolDiagnosticReceipt()
        marker = object()
        tag = _receipt_entry_tag(receipt, marker, context, reached, passed)
        receipt_registry[receipt] = (
            receipt,
            marker,
            context,
            reached,
            passed,
            tag,
            runtime_token,
        )
        receipt_marker_registry[marker] = (receipt, tag)
        _guard_runtime()
        return receipt

    def _resolve_receipt(
        context: LocalizerContext,
        receipt: _PoolDiagnosticReceipt,
    ) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        _guard_runtime()
        if type(receipt) is not _PoolDiagnosticReceipt:
            raise PermissionError("Candidate13 pool receipt is not authentic")
        entry = receipt_registry.get(receipt)
        if not _receipt_entry_is_valid(receipt, entry) or entry[2] is not context:
            raise PermissionError("Candidate13 pool receipt is not authentic")
        return entry[3], entry[4]

    def aggregate_pool_adapter_receipts(
        context: LocalizerContext,
        receipts: tuple[_PoolDiagnosticReceipt, ...],
    ) -> dict[str, Any]:
        _guard_runtime()
        _resolve_context(context)
        if type(receipts) is not tuple or len(receipts) != 36:
            raise ValueError("Candidate13 aggregate requires exactly 36 receipts")
        if len({id(receipt) for receipt in receipts}) != 36:
            raise PermissionError("Candidate13 rejects duplicate pool receipts")
        vectors = tuple(_resolve_receipt(context, receipt) for receipt in receipts)
        reached = tuple(
            all(vector[0][index] for vector in vectors)
            for index in range(len(STAGE_CONTRACT))
        )
        passed = tuple(
            all(vector[1][index] for vector in vectors)
            for index in range(len(STAGE_CONTRACT))
        )
        result = _build_public_result(
            reached_values=reached,
            passed_values=passed,
        )
        _guard_runtime()
        return result

    def _activate() -> None:
        nonlocal activated, runtime_seal
        if activated or runtime_seal is not None:
            raise RuntimeError("Candidate13 opaque runtime is already activated")
        functions = (
            _blocked_serialization,
            _closure_cell_ids,
            _context_entry_tag,
            _receipt_entry_tag,
            _context_entry_is_valid,
            _receipt_entry_is_valid,
            _guard_runtime,
            make_verified_context,
            _resolve_context,
            localize_pool_adapter_contract,
            _resolve_receipt,
            aggregate_pool_adapter_receipts,
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
            id(runtime_token),
            id(capability_key),
            (id(LocalizerContext), id(_PoolDiagnosticReceipt)),
            (
                _class_contract(LocalizerContext),
                _class_contract(_PoolDiagnosticReceipt),
            ),
            _live_member_contract(hashlib.sha256),
        )
        activated = True

    return (
        LocalizerContext,
        _PoolDiagnosticReceipt,
        make_verified_context,
        localize_pool_adapter_contract,
        aggregate_pool_adapter_receipts,
        _activate,
    )


(
    LocalizerContext,
    _PoolDiagnosticReceipt,
    make_verified_context,
    localize_pool_adapter_contract,
    aggregate_pool_adapter_receipts,
    _activate_opaque_runtime,
) = _build_opaque_capability_runtime()
_activate_opaque_runtime()
del _activate_opaque_runtime
del _build_opaque_capability_runtime


def contract_declaration() -> dict[str, Any]:
    """Value-free source contract; this function performs no I/O."""

    return {
        "adapter_semantics_modified": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate11_actual_record_cause_resolved": False,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "filesystem_io_performed_by_core": False,
        "fixed_binding_commitment_sha256": sha256_bytes(
            canonical_bytes(FIXED_BINDINGS)
        ),
        "hypothesis_semantics_modified": False,
        "input_custody_authenticated_by_core": False,
        "input_domain_requires_exact_builtin_bytes_and_sha_string": True,
        "outer_verified_dependency_and_input_bytes_required": True,
        "outer_process_resource_enforcement_required": True,
        "outer_controlled_import_required": True,
        "performance_tested": False,
        "per_pool_receipt_publication_authorized": False,
        "public_result_is_aggregate_only": True,
        "resource_bounds_enforced_by_core": False,
        "rule_selected": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "stage_contract": list(STAGE_CONTRACT),
        "stage_precedence_is_candidate13_diagnostic_not_candidate9_internal": True,
    }
