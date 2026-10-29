"""Candidate9 lossless representation adapter for clustered pool provenance.

This module does not select, score, filter, add, delete, deduplicate, reorder,
or otherwise alter candidate semantics.  It removes a stale inherited
``source_kind`` on a builder-produced ``cross_source_line_object`` whose
authoritative ``representative_source`` is present and valid.  When
authoritative ``span_endpoints`` exist, it may also remove an incomplete
redundant ``anchor``/``axis_direction``/``span_mm`` group that Candidate6 would
otherwise mistake for a complete alternate geometry.  Every complete alternate
geometry must still agree under Candidate6's unchanged validator.  Every other
JSON value and line-object position is preserved exactly under canonical JSON
value semantics.  Ambiguous, unknown, duplicate, non-finite, or malformed
inputs fail closed.

The adapter is filesystem-blind.  A future guarded launcher must authenticate
the exact builder, clusterer, Candidate6 hypothesis, this source, and its tests
before passing bytes here.  Candidate9 is an engineering attempt under the
scientifically untested Candidate6 32-rule hypothesis; it does not test or
change that hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import ModuleType
from typing import Any, Mapping


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 9
ENGINEERING_ATTEMPT_ID = "CANDIDATE9_LOSSLESS_POOL_ALIAS_ADAPTER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
ADAPTER_KIND = "sherman_qc_f1_aug30_train18_candidate9_lossless_pool_alias_adapter_v1"

HYPOTHESIS_SOURCE_SHA256 = (
    "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
)
BUILDER_SOURCE_SHA256 = (
    "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2"
)
CLUSTER_SOURCE_SHA256 = (
    "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c"
)
CANDIDATE8_DISPOSITION_SHA256 = (
    "e095c84cf616cda855f21843dfbddb69e6d6de38aa8cddde906a1a6aca87f331"
)

TARGET_CANDIDATE_SOURCE = "cross_source_line_object"
AUTHORITATIVE_SOURCE_FIELD = "representative_source"
STALE_ALIAS_FIELD = "source_kind"
AUXILIARY_GEOMETRY_FIELDS = ("anchor", "axis_direction", "span_mm")
POOL_ALGORITHM = "f1_uniform_scan_only_candidate_pool_v3"
POOL_SCHEMA_VERSION = 3
WORKING_POINT_BUDGET = 200000
SOURCE_FIELDS = frozenset(
    {"candidate_source", AUTHORITATIVE_SOURCE_FIELD, STALE_ALIAS_FIELD}
)

REPRESENTATION_CONTRACT = {
    "added_rows": False,
    "candidate_features_changed": False,
    "authoritative_candidate_geometry_changed": False,
    "candidate_identifiers_changed": False,
    "candidate_scores_changed": False,
    "candidate_source_semantics_changed": False,
    "deduplication_performed": False,
    "dropped_rows": False,
    "feature_imputation_performed": False,
    "filtering_performed": False,
    "one_to_one_row_mapping_required": True,
    "order_preserved": True,
    "ownership_changed": False,
    "rule_or_threshold_changed": False,
    "selection_performed": False,
    "sole_authorized_field_removals": (
        "cross_source_line_object.source_kind_when_representative_source_is_authoritative",
        "incomplete_anchor_axis_span_aliases_when_span_endpoints_is_authoritative",
    ),
}

_CONTEXT_SENTINEL = object()


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
        raise TypeError("Candidate9 hashes immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _decode_pool(data: bytes, *, name: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be immutable bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _strict_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def _source_alias_spelling_is_known(key: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", key.lower())
    return compact in {
        "candidatesource",
        "representativesource",
        "sourcekind",
    }


@dataclass(frozen=True)
class AdapterContext:
    hypothesis: ModuleType
    builder: ModuleType
    clusterer: ModuleType
    allowed_source_kinds: frozenset[str]
    _sentinel: object


@dataclass(frozen=True)
class AdaptedPoolBytes:
    original_pool_sha256: str
    adapted_pool_sha256: str
    adapted_pool_bytes: bytes
    projection_byte_equivalent: bool


def make_verified_context(
    *,
    hypothesis: ModuleType,
    builder: ModuleType,
    clusterer: ModuleType,
    hypothesis_source_sha256: str,
    builder_source_sha256: str,
    cluster_source_sha256: str,
) -> AdapterContext:
    if _require_sha(
        hypothesis_source_sha256, name="Candidate6 hypothesis source SHA"
    ) != HYPOTHESIS_SOURCE_SHA256:
        raise ValueError("Candidate9 hypothesis source binding mismatch")
    if _require_sha(
        builder_source_sha256, name="candidate-pool builder source SHA"
    ) != BUILDER_SOURCE_SHA256:
        raise ValueError("Candidate9 builder source binding mismatch")
    if _require_sha(
        cluster_source_sha256, name="line-object cluster source SHA"
    ) != CLUSTER_SOURCE_SHA256:
        raise ValueError("Candidate9 cluster source binding mismatch")
    if (
        getattr(hypothesis, "ATTEMPT_ORDINAL", None) != 6
        or getattr(hypothesis, "MAX_ADDITIONS_PER_CASE", None) != 1
        or getattr(hypothesis, "OWNERSHIP_TRANSVERSE_OFFSET_MM", None) != 4.0
        or not callable(getattr(hypothesis, "_validate_pool_bytes", None))
        or not callable(getattr(hypothesis, "_validate_no_forbidden_keys", None))
        or not callable(getattr(hypothesis, "_endpoints_from_row", None))
        or not callable(getattr(hypothesis, "_canonical_endpoints", None))
        or not callable(getattr(hypothesis, "_endpoints_equivalent", None))
        or not callable(getattr(hypothesis, "_point", None))
        or not callable(getattr(hypothesis, "_strict_number", None))
        or not callable(getattr(hypothesis, "_norm", None))
        or not callable(getattr(hypothesis, "_scale", None))
        or not callable(getattr(hypothesis, "_vector_add", None))
        or not callable(getattr(hypothesis, "_vector_subtract", None))
        or not callable(getattr(builder, "_tag", None))
        or not callable(getattr(clusterer, "cluster_line_candidates", None))
    ):
        raise ValueError("Candidate9 bound source contract mismatch")
    allowed = getattr(hypothesis, "ALLOWED_SOURCE_KINDS", None)
    if not isinstance(allowed, frozenset) or not allowed or any(
        not isinstance(value, str) or not value for value in allowed
    ):
        raise ValueError("Candidate9 hypothesis source-kind contract mismatch")
    return AdapterContext(
        hypothesis=hypothesis,
        builder=builder,
        clusterer=clusterer,
        allowed_source_kinds=frozenset(allowed),
        _sentinel=_CONTEXT_SENTINEL,
    )


def _validate_context(context: AdapterContext) -> None:
    if (
        not isinstance(context, AdapterContext)
        or context._sentinel is not _CONTEXT_SENTINEL
        or context.allowed_source_kinds
        != frozenset(getattr(context.hypothesis, "ALLOWED_SOURCE_KINDS", ()))
    ):
        raise PermissionError("Candidate9 lacks an exact verified source context")


def _validate_header(pool: Mapping[str, Any], context: AdapterContext) -> None:
    if (
        pool.get("schema_version") != POOL_SCHEMA_VERSION
        or pool.get("algorithm") != POOL_ALGORITHM
        or pool.get("inference_contract")
        != getattr(context.hypothesis, "POOL_INFERENCE_CONTRACT", None)
        or pool.get("source_point_serialization") is not False
        or isinstance(pool.get("working_point_budget"), bool)
        or pool.get("working_point_budget") != WORKING_POINT_BUDGET
    ):
        raise ValueError("Candidate9 candidate-pool header mismatch")
    context.hypothesis._validate_no_forbidden_keys(pool, path="candidate pool")


def _line_object_identity(row: Mapping[str, Any], *, index: int) -> str:
    return _strict_text(row.get("bend_id"), name=f"line_objects[{index}].bend_id")


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_cluster_envelope(
    row: Mapping[str, Any],
    *,
    index: int,
    representative_source: str,
    context: AdapterContext,
) -> None:
    sources = row.get("cluster_sources")
    member_ids = row.get("cluster_member_ids")
    source_count = _strict_positive_int(
        row.get("cluster_source_count"),
        name=f"line_objects[{index}].cluster_source_count",
    )
    member_count = _strict_positive_int(
        row.get("cluster_member_count"),
        name=f"line_objects[{index}].cluster_member_count",
    )
    if not isinstance(sources, list) or not isinstance(member_ids, list):
        raise ValueError(f"line_objects[{index}] lacks exact cluster provenance")
    normalized_sources = [
        _strict_text(value, name=f"line_objects[{index}].cluster_sources")
        for value in sources
    ]
    normalized_member_ids = [
        _strict_text(value, name=f"line_objects[{index}].cluster_member_ids")
        for value in member_ids
    ]
    source_candidate_id = _strict_text(
        row.get("source_candidate_id"),
        name=f"line_objects[{index}].source_candidate_id",
    )
    representative_candidate_id = _strict_text(
        row.get("candidate_id"),
        name=f"line_objects[{index}].candidate_id",
    )
    expected_representative_candidate_id = (
        f"{representative_source}::{source_candidate_id}"
    )
    member_source_prefixes: set[str] = set()
    for member_id in normalized_member_ids:
        if "::" not in member_id:
            raise ValueError(
                f"line_objects[{index}] cluster member lacks builder namespace"
            )
        source_prefix, source_identifier = member_id.split("::", 1)
        if (
            not source_prefix
            or not source_identifier
            or source_prefix not in context.allowed_source_kinds
        ):
            raise ValueError(
                f"line_objects[{index}] cluster member namespace is invalid"
            )
        member_source_prefixes.add(source_prefix)
    if (
        normalized_sources != sorted(set(normalized_sources))
        or normalized_member_ids != sorted(normalized_member_ids)
        or len(normalized_member_ids) != len(set(normalized_member_ids))
        or source_count != len(normalized_sources)
        or member_count != len(normalized_member_ids)
        or representative_source not in normalized_sources
        or representative_candidate_id != expected_representative_candidate_id
        or representative_candidate_id not in normalized_member_ids
        or member_source_prefixes != set(normalized_sources)
        or any(
            source not in context.allowed_source_kinds
            for source in normalized_sources
        )
    ):
        raise ValueError(f"line_objects[{index}] cluster provenance is ambiguous")


def _normalize_line_object(
    row: Mapping[str, Any], *, index: int, context: AdapterContext
) -> dict[str, Any]:
    for key in row:
        if not isinstance(key, str):
            raise ValueError(f"line_objects[{index}] contains a non-string key")
        if _source_alias_spelling_is_known(key) and key not in SOURCE_FIELDS:
            raise ValueError(
                f"line_objects[{index}] contains an unknown source alias spelling"
            )
    candidate_source = _strict_text(
        row.get("candidate_source"),
        name=f"line_objects[{index}].candidate_source",
    )
    output = dict(row)
    if candidate_source == TARGET_CANDIDATE_SOURCE:
        representative = _strict_text(
            row.get(AUTHORITATIVE_SOURCE_FIELD),
            name=f"line_objects[{index}].{AUTHORITATIVE_SOURCE_FIELD}",
        )
        if representative not in context.allowed_source_kinds:
            raise ValueError(
                f"line_objects[{index}] has an unknown authoritative source"
            )
        _validate_cluster_envelope(
            row,
            index=index,
            representative_source=representative,
            context=context,
        )
        if STALE_ALIAS_FIELD in row:
            _strict_text(
                row[STALE_ALIAS_FIELD],
                name=f"line_objects[{index}].{STALE_ALIAS_FIELD}",
            )
            del output[STALE_ALIAS_FIELD]
    else:
        if candidate_source not in context.allowed_source_kinds:
            raise ValueError(f"line_objects[{index}] has an unknown candidate source")
        for alias in (AUTHORITATIVE_SOURCE_FIELD, STALE_ALIAS_FIELD):
            if alias in row:
                value = _strict_text(row[alias], name=f"line_objects[{index}].{alias}")
                if value != candidate_source:
                    raise ValueError(
                        f"line_objects[{index}] contains ambiguous source aliases"
                    )
    auxiliary_keys_present = any(
        field in row for field in AUXILIARY_GEOMETRY_FIELDS
    )
    auxiliary_complete = all(
        row.get(field) is not None for field in AUXILIARY_GEOMETRY_FIELDS
    )
    if auxiliary_keys_present and not auxiliary_complete:
        if row.get("span_endpoints") is None:
            raise ValueError(
                f"line_objects[{index}] has incomplete nonredundant geometry aliases"
            )
        _validate_partial_geometry_against_span(
            row, index=index, context=context
        )
        for field in AUXILIARY_GEOMETRY_FIELDS:
            output.pop(field, None)
    # This unchanged Candidate6 primitive proves finite geometry and exact
    # agreement among every complete redundant representation.  Partial
    # auxiliary aliases have already been removed only when span_endpoints is
    # authoritative; no coordinate is ever reconstructed or modified here.
    if context.hypothesis._endpoints_from_row(
        output, name=f"line_objects[{index}]"
    ) is None:
        raise ValueError(f"line_objects[{index}] lacks finite line geometry")
    return output


def _validate_partial_geometry_against_span(
    row: Mapping[str, Any], *, index: int, context: AdapterContext
) -> None:
    hypothesis_runtime = context.hypothesis
    reference = hypothesis_runtime._endpoints_from_row(
        {"span_endpoints": row["span_endpoints"]},
        name=f"line_objects[{index}].authoritative_span",
    )
    if reference is None:
        raise ValueError(f"line_objects[{index}] lacks authoritative span geometry")
    delta = hypothesis_runtime._vector_subtract(reference[1], reference[0])
    reference_length = hypothesis_runtime._norm(delta)
    reference_axis = hypothesis_runtime._scale(delta, 1.0 / reference_length)
    reference_midpoint = hypothesis_runtime._scale(
        hypothesis_runtime._vector_add(reference[0], reference[1]), 0.5
    )
    anchor = (
        hypothesis_runtime._point(
            row["anchor"], name=f"line_objects[{index}].anchor"
        )
        if row.get("anchor") is not None
        else reference_midpoint
    )
    axis = (
        hypothesis_runtime._point(
            row["axis_direction"],
            name=f"line_objects[{index}].axis_direction",
        )
        if row.get("axis_direction") is not None
        else reference_axis
    )
    span_mm = (
        hypothesis_runtime._strict_number(
            row["span_mm"], name=f"line_objects[{index}].span_mm"
        )
        if row.get("span_mm") is not None
        else reference_length
    )
    axis_norm = hypothesis_runtime._norm(axis)
    if axis_norm <= 1.0e-12 or span_mm <= 1.0e-9:
        raise ValueError(f"line_objects[{index}] has degenerate partial geometry")
    half = hypothesis_runtime._scale(axis, 0.5 * span_mm / axis_norm)
    reconstructed = hypothesis_runtime._canonical_endpoints(
        hypothesis_runtime._vector_subtract(anchor, half),
        hypothesis_runtime._vector_add(anchor, half),
    )
    # This is the exact Candidate6 1e-6 mm endpoint-equivalence primitive;
    # Candidate9 introduces no new geometry tolerance.
    if not hypothesis_runtime._endpoints_equivalent(reference, reconstructed):
        raise ValueError(
            f"line_objects[{index}] partial geometry conflicts with authoritative span"
        )


def _authorized_projection(
    pool: Mapping[str, Any], *, context: AdapterContext
) -> dict[str, Any]:
    _validate_context(context)
    _validate_header(pool, context)
    line_objects = pool.get("line_objects")
    if not isinstance(line_objects, list):
        raise ValueError("Candidate9 candidate pool line_objects must be a list")
    output_rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, row in enumerate(line_objects):
        if not isinstance(row, dict):
            raise ValueError(f"line_objects[{index}] must be one object")
        identity = _line_object_identity(row, index=index)
        if identity in identities:
            raise ValueError("Candidate9 rejects duplicate line-object identifiers")
        identities.add(identity)
        output_rows.append(_normalize_line_object(row, index=index, context=context))
    output = dict(pool)
    output["line_objects"] = output_rows
    if len(output_rows) != len(line_objects):
        raise AssertionError("Candidate9 one-to-one row mapping invariant failed")
    if [row["bend_id"] for row in output_rows] != [
        row["bend_id"] for row in line_objects
    ]:
        raise AssertionError("Candidate9 line-object order invariant failed")
    return output


def normalize_pool_bytes(
    context: AdapterContext,
    *,
    pool_bytes: bytes,
    expected_pool_sha256: str,
) -> bytes:
    _validate_context(context)
    if sha256_bytes(pool_bytes) != _require_sha(
        expected_pool_sha256, name="Candidate9 input pool SHA"
    ):
        raise ValueError("Candidate9 input pool differs from its external binding")
    pool = _decode_pool(pool_bytes, name="Candidate9 input pool")
    projected = _authorized_projection(pool, context=context)
    output = canonical_bytes(projected)
    parsed_output = _decode_pool(output, name="Candidate9 normalized pool")
    if parsed_output != projected:
        raise AssertionError("Candidate9 normalized pool projection changed")
    # The unchanged Candidate6 validator is the consumer-side representation
    # proof.  It may reject, but this adapter never catches or weakens it.
    context.hypothesis._validate_pool_bytes(
        output,
        expected_sha256=sha256_bytes(output),
    )
    return output


def adapt_pool_bytes(
    context: AdapterContext,
    *,
    pool_bytes: bytes,
    expected_pool_sha256: str,
) -> AdaptedPoolBytes:
    original_sha = _require_sha(
        expected_pool_sha256, name="Candidate9 original pool SHA"
    )
    output = normalize_pool_bytes(
        context,
        pool_bytes=pool_bytes,
        expected_pool_sha256=original_sha,
    )
    verify_projection_byte_equivalence(
        context,
        original_pool_bytes=pool_bytes,
        normalized_pool_bytes=output,
    )
    return AdaptedPoolBytes(
        original_pool_sha256=original_sha,
        adapted_pool_sha256=sha256_bytes(output),
        adapted_pool_bytes=output,
        projection_byte_equivalent=True,
    )


def verify_projection_byte_equivalence(
    context: AdapterContext,
    *,
    original_pool_bytes: bytes,
    normalized_pool_bytes: bytes,
) -> None:
    _validate_context(context)
    original = _decode_pool(original_pool_bytes, name="Candidate9 original pool")
    normalized = _decode_pool(normalized_pool_bytes, name="Candidate9 normalized pool")
    expected = canonical_bytes(_authorized_projection(original, context=context))
    if normalized_pool_bytes != expected or normalized != _decode_pool(
        expected, name="Candidate9 expected projection"
    ):
        raise ValueError("Candidate9 normalized pool is not projection-byte equivalent")


def contract_declaration() -> dict[str, Any]:
    return {
        "adapter_kind": ADAPTER_KIND,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "bound_source_sha256": {
            "builder": BUILDER_SOURCE_SHA256,
            "candidate6_hypothesis": HYPOTHESIS_SOURCE_SHA256,
            "clusterer": CLUSTER_SOURCE_SHA256,
        },
        "candidate8_disposition_sha256": CANDIDATE8_DISPOSITION_SHA256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "external_verified_source_bytes_required": True,
        "representation_contract": dict(REPRESENTATION_CONTRACT),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
    }
