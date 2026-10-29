"""Filesystem-blind Candidate12 synthetic adapter-contract localizer.

This module exercises the exact Candidate9 lossless pool projection and the
unchanged Candidate6 pool consumer on a fixed neutral schema-3 matrix.  It has
no filesystem, discovery, label, metric, fitting, holdout, or publication
capability.  The detailed synthetic receipts never cross the module boundary;
the public result is only a closed stage enum plus aggregate reached/pass
bitsets.  A future guarded launcher must authenticate all frozen source bytes
before constructing the context used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import types
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 12
ENGINEERING_ATTEMPT_ID = "CANDIDATE12_SYNTHETIC_ADAPTER_CONTRACT_LOCALIZER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = (
    "sherman_qc_f1_aug30_train18_candidate12_synthetic_adapter_contract_result_v1"
)

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
    "JSON_CANONICALIZATION_AND_HEADER",
    "RECURSIVE_FORBIDDEN_FIELDS",
    "ROW_ORDER_AND_IDENTITY",
    "SOURCE_AUTHORITY",
    "CLUSTER_NAMESPACE_AND_PROVENANCE",
    "AUXILIARY_GEOMETRY_SUBSETS",
    "FEATURE_MISSINGNESS_AND_TYPES",
    "FINAL_C6_POOL_VALIDATION",
    "RECEIPT_PROJECTION_EQUIVALENCE",
    "AGGREGATE_RESULT",
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

_CONTEXT_SENTINEL = object()
_RECEIPT_SENTINEL = object()
_ISSUED_CONTEXTS: dict[int, tuple[object, object, object]] = {}


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
        raise TypeError("Candidate12 hashes exact immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _binding_projection(value: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "path": str(binding["path"]),
            "sha256": str(binding["sha256"]),
            "size_bytes": int(binding["size_bytes"]),
        }
        for name, binding in sorted(value.items())
    }


def _function_snapshot(functions: tuple[Callable[..., Any], ...]) -> tuple[Any, ...]:
    if any(type(function) is not types.FunctionType for function in functions):
        raise ValueError("Candidate12 controlled callables must be exact functions")
    return tuple((function, function.__code__) for function in functions)


def _frozen_value(value: Any) -> Any:
    if type(value) in (type(None), bool, int, float, str):
        return value
    if type(value) in (tuple, list):
        return {
            "kind": type(value).__name__,
            "items": [_frozen_value(child) for child in value],
        }
    if type(value) in (set, frozenset):
        children = [_frozen_value(child) for child in value]
        children.sort(key=canonical_bytes)
        return {"kind": type(value).__name__, "items": children}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("Candidate12 constant mapping keys must be exact strings")
        return {
            "kind": "mapping",
            "items": {
                key: _frozen_value(child) for key, child in sorted(value.items())
            },
        }
    raise TypeError(
        f"Candidate12 cannot freeze dependency value type {type(value).__name__}"
    )


def _function_contract(function: types.FunctionType) -> tuple[Any, ...]:
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
        raise TypeError("Candidate12 dependency must be an exact module")
    symbols: list[tuple[Any, ...]] = []
    for name, value in sorted(vars(module).items()):
        if type(value) is types.FunctionType and value.__module__ == module.__name__:
            symbols.append(("function", name, *_function_contract(value)))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            symbols.append(("class", name, *_class_contract(value)))
    return tuple(symbols)


@dataclass(frozen=True)
class LocalizerContext:
    c9_adapter: types.ModuleType
    c9_context: Any
    c6_hypothesis: types.ModuleType
    bindings: Mapping[str, Mapping[str, Any]]
    binding_commitment_sha256: str
    _functions: tuple[Callable[..., Any], ...]
    _function_snapshot: tuple[Any, ...]
    _local_function_snapshot: tuple[Any, ...]
    _dependency_symbol_snapshot: tuple[Any, ...]
    _dependency_constant_commitment_sha256: str
    _stage_function_snapshot: tuple[Any, ...]
    _constant_commitment_sha256: str
    _sentinel: object
    _issuance_token: object


@dataclass(frozen=True)
class _StageReceipt:
    stage: str
    reached: bool
    passed: bool


@dataclass(frozen=True)
class _ReceiptVector:
    receipts: tuple[_StageReceipt, ...]
    _sentinel: object

    def __reduce__(self) -> Any:
        raise TypeError("Candidate12 private receipts are not serializable")


def _controlled_functions(c9_adapter: types.ModuleType, c9_context: Any, c6: types.ModuleType) -> tuple[Callable[..., Any], ...]:
    return (
        c9_adapter._validate_context,
        c9_adapter._decode_pool,
        c9_adapter._authorized_projection,
        c9_adapter.adapt_pool_bytes,
        c9_adapter.verify_projection_byte_equivalence,
        c6._validate_pool_bytes,
        c6._validate_no_forbidden_keys,
        c6._endpoints_from_row,
        c9_context.builder._tag,
        c9_context.clusterer.cluster_line_candidates,
    )


def _dependency_modules(
    c9_adapter: types.ModuleType, c9_context: Any, c6: types.ModuleType
) -> tuple[types.ModuleType, ...]:
    return c9_adapter, c6, c9_context.builder, c9_context.clusterer


def _dependency_symbol_snapshot(
    c9_adapter: types.ModuleType, c9_context: Any, c6: types.ModuleType
) -> tuple[Any, ...]:
    modules = _dependency_modules(c9_adapter, c9_context, c6)
    module_symbols = tuple(
        (module, module.__name__, _module_symbol_snapshot(module)) for module in modules
    )
    referenced_modules = (
        ("c9_context_sentinel", c9_adapter._CONTEXT_SENTINEL),
        ("c9_hashlib", c9_adapter.hashlib),
        ("c9_json", c9_adapter.json),
        ("c9_re", c9_adapter.re),
        ("c6_hashlib", c6.hashlib),
        ("c6_json", c6.json),
        ("c6_math", c6.math),
        ("c6_re", c6.re),
        ("c6_sha_pattern", c6._HEX64),
        ("builder_numpy", c9_context.builder.np),
        ("cluster_math", c9_context.clusterer.math),
        ("cluster_numpy", c9_context.clusterer.np),
    )
    return module_symbols, referenced_modules


def _dependency_constant_projection(
    c9_adapter: types.ModuleType, c9_context: Any, c6: types.ModuleType
) -> dict[str, Any]:
    return {
        "builder": {
            name: _frozen_value(getattr(c9_context.builder, name))
            for name in BUILDER_RUNTIME_CONSTANTS
        },
        "c6": {
            name: _frozen_value(getattr(c6, name))
            for name in C6_RUNTIME_CONSTANTS
        },
        "c9": {
            name: _frozen_value(getattr(c9_adapter, name))
            for name in C9_RUNTIME_CONSTANTS
        },
    }


def _dependency_constant_commitment(
    c9_adapter: types.ModuleType, c9_context: Any, c6: types.ModuleType
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE12_TRANSITIVE_DEPENDENCY_CONSTANTS_V1",
                "projection": _dependency_constant_projection(
                    c9_adapter, c9_context, c6
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


def _dependency_contract_is_exact(
    c9_adapter: types.ModuleType, c9_context: Any, c6_hypothesis: types.ModuleType
) -> bool:
    return bool(
        type(c9_adapter) is types.ModuleType
        and type(c6_hypothesis) is types.ModuleType
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
        and type(c9_context.builder) is types.ModuleType
        and type(c9_context.clusterer) is types.ModuleType
        and c9_context.allowed_source_kinds == c6_hypothesis.ALLOWED_SOURCE_KINDS
    )


def _constant_commitment(c9_adapter: types.ModuleType, c6: types.ModuleType) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "c6_allowed_source_kinds": sorted(c6.ALLOWED_SOURCE_KINDS),
                "c6_attempt_ordinal": c6.ATTEMPT_ORDINAL,
                "c6_pool_inference_contract": dict(c6.POOL_INFERENCE_CONTRACT),
                "c6_rule_space_sha256": sha256_bytes(
                    canonical_bytes([dict(rule) for rule in c6.FIXED_RULE_SPACE])
                ),
                "c9_attempt_ordinal": c9_adapter.ATTEMPT_ORDINAL,
                "c9_pool_algorithm": c9_adapter.POOL_ALGORITHM,
                "c9_pool_schema_version": c9_adapter.POOL_SCHEMA_VERSION,
                "c9_representation_contract": dict(c9_adapter.REPRESENTATION_CONTRACT),
                "c9_working_point_budget": c9_adapter.WORKING_POINT_BUDGET,
                "stage_contract": list(STAGE_CONTRACT),
            }
        )
    )


def make_verified_context(
    *,
    c9_adapter: types.ModuleType,
    c9_context: Any,
    c6_hypothesis: types.ModuleType,
    bindings: Mapping[str, Mapping[str, Any]],
) -> LocalizerContext:
    if type(c9_adapter) is not types.ModuleType or type(c6_hypothesis) is not types.ModuleType:
        raise ValueError("Candidate12 requires exact dependency modules")
    if _binding_projection(bindings) != FIXED_BINDINGS:
        raise ValueError("Candidate12 source binding map mismatch")
    if not _dependency_contract_is_exact(c9_adapter, c9_context, c6_hypothesis):
        raise ValueError("Candidate12 dependency contract mismatch")
    c9_adapter._validate_context(c9_context)
    functions = _controlled_functions(c9_adapter, c9_context, c6_hypothesis)
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in sorted(FIXED_BINDINGS.items())
        }
    )
    token = object()
    context = LocalizerContext(
        c9_adapter=c9_adapter,
        c9_context=c9_context,
        c6_hypothesis=c6_hypothesis,
        bindings=immutable_bindings,
        binding_commitment_sha256=sha256_bytes(canonical_bytes(FIXED_BINDINGS)),
        _functions=functions,
        _function_snapshot=_function_snapshot(functions),
        _local_function_snapshot=_local_function_snapshot(),
        _dependency_symbol_snapshot=_dependency_symbol_snapshot(
            c9_adapter, c9_context, c6_hypothesis
        ),
        _dependency_constant_commitment_sha256=_dependency_constant_commitment(
            c9_adapter, c9_context, c6_hypothesis
        ),
        _stage_function_snapshot=_stage_function_snapshot(),
        _constant_commitment_sha256=_constant_commitment(c9_adapter, c6_hypothesis),
        _sentinel=_CONTEXT_SENTINEL,
        _issuance_token=token,
    )
    _ISSUED_CONTEXTS[id(context)] = (context, token, _CONTEXT_SENTINEL)
    return context


def _validate_context(context: LocalizerContext) -> None:
    try:
        issued = _ISSUED_CONTEXTS.get(id(context))
        intact = not (
            type(context) is not LocalizerContext
            or context._sentinel is not _CONTEXT_SENTINEL
            or issued is None
            or issued[0] is not context
            or issued[1] is not context._issuance_token
            or issued[2] is not _CONTEXT_SENTINEL
            or not _dependency_contract_is_exact(
                context.c9_adapter, context.c9_context, context.c6_hypothesis
            )
            or _binding_projection(context.bindings) != FIXED_BINDINGS
            or context.binding_commitment_sha256
            != sha256_bytes(canonical_bytes(FIXED_BINDINGS))
            or context._functions
            != _controlled_functions(
                context.c9_adapter, context.c9_context, context.c6_hypothesis
            )
            or context._function_snapshot != _function_snapshot(context._functions)
            or context._local_function_snapshot != _local_function_snapshot()
            or context._dependency_symbol_snapshot
            != _dependency_symbol_snapshot(
                context.c9_adapter, context.c9_context, context.c6_hypothesis
            )
            or context._dependency_constant_commitment_sha256
            != _dependency_constant_commitment(
                context.c9_adapter, context.c9_context, context.c6_hypothesis
            )
            or context._stage_function_snapshot != _stage_function_snapshot()
            or context._constant_commitment_sha256
            != _constant_commitment(context.c9_adapter, context.c6_hypothesis)
        )
        if not intact:
            raise ValueError("context mismatch")
        context.c9_adapter._validate_context(context.c9_context)
    except Exception:
        raise PermissionError("Candidate12 verified context is not intact") from None


def _clone(value: Any) -> Any:
    return json.loads(canonical_bytes(value).decode("utf-8"))


def _neutral_cross_row() -> dict[str, Any]:
    return {
        "bend_id": "NEUTRAL-BEND-A",
        "candidate_id": "plane_lattice::NEUTRAL-A",
        "candidate_source": "cross_source_line_object",
        "cluster_member_count": 2,
        "cluster_member_ids": [
            "evidence_trace::NEUTRAL-B",
            "plane_lattice::NEUTRAL-A",
        ],
        "cluster_source_count": 2,
        "cluster_sources": ["evidence_trace", "plane_lattice"],
        "endpoint_support_fraction": 0.75,
        "finite_support_fraction": 0.75,
        "normal_second_eigenvalue": 0.05,
        "representative_source": "plane_lattice",
        "score": 0.5,
        "source_candidate_id": "NEUTRAL-A",
        "source_kind": "evidence_trace",
        "span_endpoints": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        "two_sided_transition_score": 0.75,
    }


def _neutral_noncluster_row() -> dict[str, Any]:
    return {
        "bend_id": "NEUTRAL-BEND-B",
        "candidate_source": "scan_mesh_adjacency",
        "endpoint_support_fraction": 0.5,
        "finite_support_fraction": 0.5,
        "normal_second_eigenvalue": 0.1,
        "representative_source": "scan_mesh_adjacency",
        "score": 0.25,
        "source_kind": "scan_mesh_adjacency",
        "span_endpoints": [[0.0, 20.0, 0.0], [20.0, 20.0, 0.0]],
        "two_sided_transition_score": 0.5,
    }


def _pool(context: LocalizerContext, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "algorithm": context.c9_adapter.POOL_ALGORITHM,
        "inference_contract": dict(context.c6_hypothesis.POOL_INFERENCE_CONTRACT),
        "line_objects": _clone(rows if rows is not None else [_neutral_cross_row()]),
        "schema_version": context.c9_adapter.POOL_SCHEMA_VERSION,
        "source_point_serialization": False,
        "working_point_budget": context.c9_adapter.WORKING_POINT_BUDGET,
    }


def _adapt(context: LocalizerContext, pool: Mapping[str, Any], *, raw: bytes | None = None) -> Any:
    data = canonical_bytes(pool) if raw is None else raw
    return context.c9_adapter.adapt_pool_bytes(
        context.c9_context,
        pool_bytes=data,
        expected_pool_sha256=sha256_bytes(data),
    )


def _accept(context: LocalizerContext, pool: Mapping[str, Any], *, raw: bytes | None = None) -> Any:
    receipt = _adapt(context, pool, raw=raw)
    if (
        type(receipt) is not context.c9_adapter.AdaptedPoolBytes
        or receipt.projection_byte_equivalent is not True
        or type(receipt.adapted_pool_bytes) is not bytes
        or receipt.adapted_pool_sha256 != sha256_bytes(receipt.adapted_pool_bytes)
    ):
        raise AssertionError("Candidate12 authentic adapter receipt mismatch")
    return receipt


def _reject(context: LocalizerContext, pool: Mapping[str, Any], *, raw: bytes | None = None) -> None:
    try:
        _adapt(context, pool, raw=raw)
    except Exception:
        return
    raise AssertionError("Candidate12 negative contract probe was accepted")


def _stage_json_header(context: LocalizerContext) -> None:
    pool = _pool(context)
    canonical = canonical_bytes(pool)
    _accept(context, pool)
    noncanonical = json.dumps(pool, indent=2, sort_keys=False).encode("utf-8")
    receipt = _accept(context, pool, raw=noncanonical)
    normalized = context.c9_adapter._decode_pool(
        receipt.adapted_pool_bytes, name="Candidate12 canonicalized output"
    )
    if (
        not receipt.adapted_pool_bytes.endswith(b"\n")
        or context.c9_adapter.canonical_bytes(normalized)
        != receipt.adapted_pool_bytes
    ):
        raise AssertionError("Candidate12 normalized output is not LF-canonical")
    duplicate = b'{"algorithm":"duplicate",' + canonical[1:]
    _reject(context, pool, raw=duplicate)
    _reject(context, pool, raw=canonical[:-2])
    _reject(context, pool, raw=b"\xff")
    _reject(context, pool, raw=b"[]\n")
    nonfinite = canonical.replace(b'"score":0.5', b'"score":NaN')
    _reject(context, pool, raw=nonfinite)
    for key, bad in (
        ("schema_version", 2),
        ("algorithm", "other"),
        ("inference_contract", {}),
        ("source_point_serialization", True),
        ("working_point_budget", 1),
        ("working_point_budget", True),
    ):
        mutated = _clone(pool)
        mutated[key] = bad
        _reject(context, mutated)


def _stage_forbidden(context: LocalizerContext) -> None:
    _accept(context, _pool(context))
    for key in (
        "label",
        "ground_truth",
        "expectedCount",
        "sender_count",
        "cad_bends",
        "drawing",
        "part_id",
        "filename",
        "outcome",
        "isTruePositive",
    ):
        mutated = _pool(context)
        mutated["line_objects"][0][key] = False
        _reject(context, mutated)
    nested = _pool(context)
    nested["line_objects"][0]["neutral_nested"] = {
        "items": [{"still_neutral": True}, {"ground_truth": False}]
    }
    _reject(context, nested)


def _stage_rows(context: LocalizerContext) -> None:
    rows = [_neutral_cross_row(), _neutral_noncluster_row()]
    for ordered in (rows, list(reversed(rows))):
        receipt = _accept(context, _pool(context, ordered))
        decoded = context.c9_adapter._decode_pool(
            receipt.adapted_pool_bytes, name="Candidate12 adapted row probe"
        )
        output_rows = decoded["line_objects"]
        if [row["bend_id"] for row in output_rows] != [
            row["bend_id"] for row in ordered
        ]:
            raise AssertionError("Candidate12 adapter changed row order or identity")
        for original, projected in zip(ordered, output_rows):
            for field in ("bend_id", "candidate_id", "source_candidate_id"):
                if field in original and projected.get(field) != original[field]:
                    raise AssertionError(
                        "Candidate12 adapter changed a candidate identifier"
                    )
    duplicate = _pool(context, rows)
    duplicate["line_objects"][1]["bend_id"] = rows[0]["bend_id"]
    _reject(context, duplicate)
    missing = _pool(context)
    del missing["line_objects"][0]["bend_id"]
    _reject(context, missing)
    nonrow = _pool(context)
    nonrow["line_objects"][0] = []
    _reject(context, nonrow)


def _stage_source_authority(context: LocalizerContext) -> None:
    cross = _pool(context)
    receipt = _accept(context, cross)
    projected = context.c9_adapter._decode_pool(
        receipt.adapted_pool_bytes, name="Candidate12 source projection"
    )["line_objects"][0]
    if "source_kind" in projected or projected["representative_source"] != "plane_lattice":
        raise AssertionError("Candidate12 cross-source authority projection changed")
    _accept(context, _pool(context, [_neutral_noncluster_row()]))
    for field, bad in (
        ("representative_source", "evidence_trace"),
        ("source_kind", "evidence_trace"),
        ("candidate_source", "unknown_source"),
    ):
        row = _neutral_noncluster_row()
        row[field] = bad
        _reject(context, _pool(context, [row]))
    for mutation in ("missing", "unknown", "alias"):
        value = _pool(context)
        row = value["line_objects"][0]
        if mutation == "missing":
            del row["representative_source"]
        elif mutation == "unknown":
            row["representative_source"] = "unknown_source"
        else:
            row["representativeSource"] = row["representative_source"]
        _reject(context, value)


def _stage_cluster(context: LocalizerContext) -> None:
    # Exercise the bound builder and clusterer on neutral synthetic geometry;
    # the hand-authored envelope probes below must not be the only provenance
    # evidence.  Exact source-byte authentication remains the outer runner's
    # responsibility.
    raw = {
        "candidate_id": "NEUTRAL-GENERATED",
        "endpoint_support_fraction": 0.75,
        "finite_support_fraction": 0.75,
        "normal_second_eigenvalue": 0.05,
        "score": 0.5,
        "source_kind": "scan_mesh_adjacency",
        "span_endpoints": [[0.0, 40.0, 0.0], [20.0, 40.0, 0.0]],
        "two_sided_transition_score": 0.75,
    }
    tagged = []
    for source in ("evidence_trace", "plane_lattice"):
        tagged.extend(context.c9_context.builder._tag([_clone(raw)], source))
    generated = context.c9_context.clusterer.cluster_line_candidates(tagged)
    if type(generated) is not list or len(generated) != 1:
        raise AssertionError("Candidate12 neutral builder-cluster projection changed")
    generated_receipt = _accept(context, _pool(context, generated))
    generated_projection = context.c9_adapter._decode_pool(
        generated_receipt.adapted_pool_bytes,
        name="Candidate12 generated cluster projection",
    )["line_objects"][0]
    if (
        generated_projection.get("candidate_source")
        != context.c9_adapter.TARGET_CANDIDATE_SOURCE
        or "source_kind" in generated_projection
    ):
        raise AssertionError("Candidate12 generated source authority changed")
    _accept(context, _pool(context))
    mutations = (
        ("cluster_member_ids", ["BAD", "plane_lattice::NEUTRAL-A"]),
        ("cluster_member_ids", ["evidence_trace::", "plane_lattice::NEUTRAL-A"]),
        ("cluster_member_ids", ["foreign::X", "plane_lattice::NEUTRAL-A"]),
        ("cluster_member_ids", ["plane_lattice::NEUTRAL-A", "evidence_trace::NEUTRAL-B"]),
        ("cluster_member_ids", ["plane_lattice::NEUTRAL-A", "plane_lattice::NEUTRAL-A"]),
        ("cluster_member_ids", ["evidence_trace::NEUTRAL-B"]),
        ("cluster_member_ids", "not-a-list"),
        ("cluster_sources", ["plane_lattice", "evidence_trace"]),
        ("cluster_sources", ["evidence_trace", "foreign"]),
        ("cluster_source_count", 1),
        ("cluster_source_count", 0),
        ("cluster_member_count", 1),
        ("cluster_member_count", True),
        ("candidate_id", "plane_lattice::OTHER"),
        ("source_candidate_id", ""),
    )
    for field, bad in mutations:
        value = _pool(context)
        mutated_row = value["line_objects"][0]
        mutated_row[field] = bad
        _reject(context, value)


def _matching_auxiliary() -> dict[str, Any]:
    return {
        "anchor": [10.0, 0.0, 0.0],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": 20.0,
    }


def _stage_auxiliary(context: LocalizerContext) -> None:
    fields = ("anchor", "axis_direction", "span_mm")
    matching = _matching_auxiliary()
    for mask in range(8):
        value = _pool(context)
        row = value["line_objects"][0]
        present = [field for index, field in enumerate(fields) if mask & (1 << index)]
        for field in present:
            row[field] = _clone(matching[field])
        receipt = _accept(context, value)
        projected = context.c9_adapter._decode_pool(
            receipt.adapted_pool_bytes, name="Candidate12 auxiliary projection"
        )["line_objects"][0]
        if mask != 7 and any(field in projected for field in fields):
            raise AssertionError("Candidate12 incomplete auxiliary aliases survived")
        if mask == 7 and any(projected.get(field) != matching[field] for field in fields):
            raise AssertionError("Candidate12 complete auxiliary aliases changed")
        for field in present:
            conflict = _pool(context)
            conflict_row = conflict["line_objects"][0]
            for retained in present:
                conflict_row[retained] = _clone(matching[retained])
            conflict_row[field] = {
                "anchor": [99.0, 0.0, 0.0],
                "axis_direction": [0.0, 1.0, 0.0],
                "span_mm": 99.0,
            }[field]
            _reject(context, conflict)
    missing_span = _pool(context)
    row = missing_span["line_objects"][0]
    del row["span_endpoints"]
    row["anchor"] = matching["anchor"]
    _reject(context, missing_span)
    complete_without_span = _pool(context)
    complete_row = complete_without_span["line_objects"][0]
    del complete_row["span_endpoints"]
    complete_row.update(_clone(matching))
    _accept(context, complete_without_span)
    reverse_axis = _pool(context)
    reverse_row = reverse_axis["line_objects"][0]
    reverse_row.update(_clone(matching))
    reverse_row["axis_direction"] = [-1.0, 0.0, 0.0]
    _accept(context, reverse_axis)
    partial_reverse_axis = _pool(context)
    partial_reverse_axis["line_objects"][0]["axis_direction"] = [-1.0, 0.0, 0.0]
    _accept(context, partial_reverse_axis)
    no_geometry = _pool(context)
    del no_geometry["line_objects"][0]["span_endpoints"]
    _reject(context, no_geometry)
    for mask in range(1, 7):
        incomplete = _pool(context)
        incomplete_row = incomplete["line_objects"][0]
        del incomplete_row["span_endpoints"]
        for index, field in enumerate(fields):
            if mask & (1 << index):
                incomplete_row[field] = _clone(matching[field])
        _reject(context, incomplete)
    for endpoints in (
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2_000_001.0, 0.0, 0.0]],
    ):
        malformed = _pool(context)
        malformed["line_objects"][0]["span_endpoints"] = endpoints
        _reject(context, malformed)
    for field, bad in (
        ("axis_direction", [0.0, 0.0, 0.0]),
        ("span_mm", -1.0),
        ("anchor", "not-a-point"),
    ):
        malformed = _pool(context)
        malformed_row = malformed["line_objects"][0]
        malformed_row.update(_clone(matching))
        malformed_row[field] = bad
        _reject(context, malformed)


def _stage_features(context: LocalizerContext) -> None:
    optional = (
        "score",
        "finite_support_fraction",
        "endpoint_support_fraction",
        "two_sided_transition_score",
        "normal_second_eigenvalue",
    )
    for missing in (optional, *[(field,) for field in optional]):
        value = _pool(context)
        for field in missing:
            value["line_objects"][0].pop(field, None)
        receipt = _accept(context, value)
        projected = context.c9_adapter._decode_pool(
            receipt.adapted_pool_bytes, name="Candidate12 missing feature projection"
        )["line_objects"][0]
        if any(field in projected for field in missing):
            raise AssertionError("Candidate12 missing feature was imputed")
    for field in optional:
        value = _pool(context)
        value["line_objects"][0][field] = None
        receipt = _accept(context, value)
        projected = context.c9_adapter._decode_pool(
            receipt.adapted_pool_bytes, name="Candidate12 null feature projection"
        )["line_objects"][0]
        if field not in projected or projected[field] is not None:
            raise AssertionError("Candidate12 null feature missingness changed")
    for field, boundary in (
        ("finite_support_fraction", 0.0),
        ("finite_support_fraction", 1.0),
        ("endpoint_support_fraction", 0.0),
        ("endpoint_support_fraction", 1.0),
        ("two_sided_transition_score", 0.0),
        ("two_sided_transition_score", 1.0),
        ("normal_second_eigenvalue", 100.0),
    ):
        value = _pool(context)
        value["line_objects"][0][field] = boundary
        receipt = _accept(context, value)
        projected = context.c9_adapter._decode_pool(
            receipt.adapted_pool_bytes, name="Candidate12 feature boundary projection"
        )["line_objects"][0]
        if projected[field] != boundary:
            raise AssertionError("Candidate12 valid feature boundary changed")
    for field, bad in (
        ("score", "0.5"),
        ("score", True),
        ("finite_support_fraction", 1.1),
        ("endpoint_support_fraction", -0.1),
        ("two_sided_transition_score", 2.0),
        ("normal_second_eigenvalue", -0.1),
    ):
        value = _pool(context)
        value["line_objects"][0][field] = bad
        _reject(context, value)


def _stage_c6_validation(context: LocalizerContext) -> None:
    pool = _pool(context)
    projected = context.c9_adapter._authorized_projection(pool, context=context.c9_context)
    data = canonical_bytes(projected)
    _pool_value, candidates = context.c6_hypothesis._validate_pool_bytes(
        data, expected_sha256=sha256_bytes(data)
    )
    if not candidates:
        raise AssertionError("Candidate12 neutral projection lacks a C6 candidate")
    invalid = _clone(projected)
    invalid["line_objects"][0]["score"] = "invalid"
    bad = canonical_bytes(invalid)
    try:
        context.c6_hypothesis._validate_pool_bytes(
            bad, expected_sha256=sha256_bytes(bad)
        )
    except Exception:
        return
    raise AssertionError("Candidate12 unchanged C6 validator accepted an invalid feature")


def _stage_receipt(context: LocalizerContext) -> None:
    original = canonical_bytes(_pool(context))
    receipt = _accept(context, _pool(context), raw=original)
    if receipt.original_pool_sha256 != sha256_bytes(original):
        raise AssertionError("Candidate12 receipt input binding changed")
    context.c9_adapter.verify_projection_byte_equivalence(
        context.c9_context,
        original_pool_bytes=original,
        normalized_pool_bytes=receipt.adapted_pool_bytes,
    )
    changed = receipt.adapted_pool_bytes.replace(b"NEUTRAL-BEND-A", b"NEUTRAL-BEND-Z")
    try:
        context.c9_adapter.verify_projection_byte_equivalence(
            context.c9_context,
            original_pool_bytes=original,
            normalized_pool_bytes=changed,
        )
    except Exception:
        return
    raise AssertionError("Candidate12 projection-equivalence check accepted changed bytes")


_STAGE_FUNCTIONS: tuple[tuple[str, Callable[[LocalizerContext], None]], ...] = (
    (STAGE_CONTRACT[0], _stage_json_header),
    (STAGE_CONTRACT[1], _stage_forbidden),
    (STAGE_CONTRACT[2], _stage_rows),
    (STAGE_CONTRACT[3], _stage_source_authority),
    (STAGE_CONTRACT[4], _stage_cluster),
    (STAGE_CONTRACT[5], _stage_auxiliary),
    (STAGE_CONTRACT[6], _stage_features),
    (STAGE_CONTRACT[7], _stage_c6_validation),
    (STAGE_CONTRACT[8], _stage_receipt),
)


def _stage_function_snapshot() -> tuple[Any, ...]:
    stages = tuple(stage for stage, _function in _STAGE_FUNCTIONS)
    functions = tuple(function for _stage, function in _STAGE_FUNCTIONS)
    if stages != STAGE_CONTRACT[:-1] or len(functions) != len(STAGE_CONTRACT) - 1:
        raise ValueError("Candidate12 stage/function contract mismatch")
    return tuple((stage, function, function.__code__) for stage, function in _STAGE_FUNCTIONS)

RESULT_KEYS = frozenset(
    {
        "check_code",
        "filesystem_capability_present",
        "first_failed_stage",
        "kind",
        "label_metric_fit_holdout_opened_or_started",
        "fit_performed",
        "holdout_opened",
        "metric_computed",
        "performance_tested",
        "real_input_opened",
        "rule_selected",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)


def _validate_public_result(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Candidate12 public result schema mismatch")
    reached = value["stage_reached_all_bitset"]
    passed = value["stage_passed_all_bitset"]
    failed = value["first_failed_stage"]
    expected_failed = next(
        (
            STAGE_CONTRACT[index]
            for index, bit in enumerate(passed)
            if bit == "0"
        ),
        "NONE",
    ) if type(passed) is str and len(passed) == len(STAGE_CONTRACT) else "INVALID"
    if (
        value["kind"] != RESULT_KIND
        or value["filesystem_capability_present"] is not False
        or value["fit_performed"] is not False
        or value["holdout_opened"] is not False
        or value["label_metric_fit_holdout_opened_or_started"] is not False
        or value["metric_computed"] is not False
        or value["performance_tested"] is not False
        or value["real_input_opened"] is not False
        or value["rule_selected"] is not False
        or type(reached) is not str
        or type(passed) is not str
        or len(reached) != len(STAGE_CONTRACT)
        or len(passed) != len(STAGE_CONTRACT)
        or set(reached + passed) - {"0", "1"}
        or any(pass_bit == "1" and reach_bit != "1" for reach_bit, pass_bit in zip(reached, passed))
        or reached != "1" * len(STAGE_CONTRACT)
        or passed[-1] != (
            "1" if all(bit == "1" for bit in passed[:-1]) else "0"
        )
        or failed not in {"NONE", *STAGE_CONTRACT}
        or failed != expected_failed
        or value["stage_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(STAGE_CONTRACT)))
    ):
        raise ValueError("Candidate12 public result invariant mismatch")
    all_pass = passed == "1" * len(STAGE_CONTRACT)
    if all_pass:
        if (
            failed != "NONE"
            or value["check_code"] != "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_PASSED"
            or value["status"] != "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
        ):
            raise ValueError("Candidate12 PASS result mismatch")
    elif (
        failed == "NONE"
        or value["check_code"] != "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_FAILED"
        or value["status"] != "FAILED_ENGINEERING_CONTRACT_LOCALIZATION"
    ):
        raise ValueError("Candidate12 failure result mismatch")


def run_synthetic_contract_matrix(context: LocalizerContext) -> dict[str, Any]:
    _validate_context(context)
    _stage_function_snapshot()
    receipts: list[_StageReceipt] = []
    for stage, function in _STAGE_FUNCTIONS:
        passed = False
        try:
            function(context)
            passed = True
        except Exception:
            passed = False
        receipts.append(_StageReceipt(stage=stage, reached=True, passed=passed))
    aggregate_passed = all(receipt.reached and receipt.passed for receipt in receipts)
    receipts.append(
        _StageReceipt(
            stage=STAGE_CONTRACT[-1], reached=True, passed=aggregate_passed
        )
    )
    sealed = _ReceiptVector(tuple(receipts), _RECEIPT_SENTINEL)
    if (
        len(sealed.receipts) != len(STAGE_CONTRACT)
        or tuple(receipt.stage for receipt in sealed.receipts) != STAGE_CONTRACT
    ):
        raise RuntimeError("Candidate12 sealed receipt vector contract mismatch")
    reached = "".join("1" if value.reached else "0" for value in sealed.receipts)
    passed = "".join("1" if value.passed else "0" for value in sealed.receipts)
    first_failed = next(
        (value.stage for value in sealed.receipts if not value.passed), "NONE"
    )
    result = {
        "check_code": (
            "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_PASSED"
            if aggregate_passed
            else "C12_SYNTHETIC_ADAPTER_CONTRACT_MATRIX_FAILED"
        ),
        "filesystem_capability_present": False,
        "first_failed_stage": first_failed,
        "fit_performed": False,
        "holdout_opened": False,
        "kind": RESULT_KIND,
        "label_metric_fit_holdout_opened_or_started": False,
        "metric_computed": False,
        "performance_tested": False,
        "real_input_opened": False,
        "rule_selected": False,
        "stage_contract_commitment_sha256": sha256_bytes(
            canonical_bytes(list(STAGE_CONTRACT))
        ),
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": (
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
            if aggregate_passed
            else "FAILED_ENGINEERING_CONTRACT_LOCALIZATION"
        ),
    }
    _validate_public_result(result)
    return result


def contract_declaration() -> dict[str, Any]:
    """Source-only declaration; it does not authenticate or open any path."""

    return {
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "bindings": _clone(FIXED_BINDINGS),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "filesystem_capability_present": False,
        "fit_performed": False,
        "holdout_opened": False,
        "metric_computed": False,
        "outer_verified_dependency_bytes_required": True,
        "performance_tested": False,
        "rule_selected": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "stage_contract": list(STAGE_CONTRACT),
    }
