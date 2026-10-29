"""Frozen V7 CIIP evidence-container framing and fixed-width row codecs.

This module is deliberately geometry-free.  It serializes the six allowlisted
tables, constructs the fixed one-MiB metadata region and validates the complete
container fail-closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


METADATA_REGION_BYTES = 1_048_576
MAX_METADATA_JSON_BYTES = METADATA_REGION_BYTES - 8
MAX_SERIALIZED_BYTES = 64 * 1_048_576
SCHEMA_VERSION = 1
NOT_EVALUATED_I32 = -2_147_483_648
EXPECTED_CONFIGURATION_SHA256 = "876cf418b6a6411a2ea34be969d4d9bd9e66bcc167cac79f3a83ac3a06c40e95"
EXPECTED_SCHEMA_SHA256 = "b101e8177a96bf0a623db77c9b867d913371894cd9128ca7f25a1c8930c53376"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_DIRECTORY_DESCRIPTOR = struct.Struct("<BQIQQ32s")

_TAG_CASE = b"CIIP_CASE_V1\0"
_TAG_CELL = b"CIIP_CELL_V1\0"
_TAG_EDGE = b"CIIP_EDGE_V1\0"
_TAG_ESSENTIAL_DEATH = b"CIIP_ESSENTIAL_DEATH_V1\0"
_TAG_FIELD_ROW = b"CIIP_FIELD_ROW_V1\0"
_TAG_INTERVAL = b"CIIP_INTERVAL_V1\0"
_TAG_NODE = b"CIIP_NODE_V1\0"
_TAG_PARTITION = b"CIIP_PARTITION_COMMIT_V1\0"
_TAG_DIRECTORY = b"CIIP_TABLE_DIRECTORY_V1\0"


class ContainerValidationError(ValueError):
    """Raised when framing, canonical bytes, bindings, or rows are invalid."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    size: int


@dataclass(frozen=True)
class TableSpec:
    name: str
    name_enum: int
    fields: tuple[FieldSpec, ...]
    row_size: int
    order_fields: tuple[str, ...]

    @property
    def struct(self) -> struct.Struct:
        return struct.Struct("<" + "".join(_STRUCT_FORMAT[field.kind] for field in self.fields))


@dataclass(frozen=True)
class DecodedContainer:
    metadata: dict[str, Any]
    table_bytes: dict[str, bytes]
    rows: dict[str, Any]


_STRUCT_FORMAT = {
    "BYTES6": "6s",
    "BYTES16": "16s",
    "BYTES32": "32s",
    "U8": "B",
    "U16_LE": "H",
    "U32_LE": "I",
    "U64_LE": "Q",
    "I32_LE": "i",
    "I64_LE": "q",
}


def _f(name: str, kind: str, size: int) -> FieldSpec:
    return FieldSpec(name, kind, size)


TABLE_SPECS: dict[str, TableSpec] = {
    "partition_node": TableSpec(
        "partition_node", 0,
        (_f("node_id_sha256", "BYTES32", 32), _f("morton_prefix_u64", "U64_LE", 8),
         _f("subtree_membership_digest_sha256", "BYTES32", 32),
         _f("subtree_point_count_u32", "U32_LE", 4), _f("occupied_leaf_count_u32", "U32_LE", 4),
         _f("depth_u8", "U8", 1), _f("child_mask_u8", "U8", 1),
         _f("status_u16", "U16_LE", 2), _f("reserved_zero_u32", "U32_LE", 4)),
        88, ("depth_u8", "morton_prefix_u64"),
    ),
    "leaf_aggregate": TableSpec(
        "leaf_aggregate", 1,
        (_f("cell_id_sha256", "BYTES32", 32), _f("morton_u64", "U64_LE", 8),
         _f("membership_digest_128", "BYTES16", 16), _f("unique_point_count_u32", "U32_LE", 4),
         _f("raw_record_multiplicity_sum_u32", "U32_LE", 4)),
        64, ("morton_u64",),
    ),
    "cell_scale_field": TableSpec(
        "cell_scale_field", 2,
        (_f("row_id_digest_128", "BYTES16", 16), _f("support_membership_digest_128", "BYTES16", 16),
         _f("morton_u64", "U64_LE", 8), _f("trace_q_i64", "I64_LE", 8),
         _f("lambda1_q_i64", "I64_LE", 8), _f("lambda2_q_i64", "I64_LE", 8),
         _f("lambda3_q_i64", "I64_LE", 8), _f("mass_q_i64", "I64_LE", 8),
         _f("arm_u8", "U8", 1), _f("scale_u8", "U8", 1), _f("status_u16", "U16_LE", 2),
         _f("support_count_u32", "U32_LE", 4), _f("raw_count_p10_u32", "U32_LE", 4),
         _f("raw_count_p90_u32", "U32_LE", 4), _f("balance_numerator_u32", "U32_LE", 4),
         _f("balance_denominator_u32", "U32_LE", 4), _f("growth12_gate_code_i32", "I32_LE", 4),
         _f("growth23_gate_code_i32", "I32_LE", 4), _f("anisotropy_q_u16", "U16_LE", 2),
         _f("filtration_u16", "U16_LE", 2), _f("stage_mask_u32", "U32_LE", 4),
         _f("first_rejection_u16", "U16_LE", 2), _f("flags_u16", "U16_LE", 2),
         _f("eigenvalue_denominator_u32", "U32_LE", 4)),
        128, ("arm_u8", "scale_u8", "morton_u64"),
    ),
    "space_scale_adjacency": TableSpec(
        "space_scale_adjacency", 3,
        (_f("edge_id_digest_128", "BYTES16", 16), _f("lower_vertex_u32", "U32_LE", 4),
         _f("upper_vertex_u32", "U32_LE", 4), _f("edge_kind_u8", "U8", 1),
         _f("axis_u8", "U8", 1), _f("status_u16", "U16_LE", 2),
         _f("reserved_zero_u32", "U32_LE", 4)),
        32, ("edge_kind_u8", "axis_u8", "lower_vertex_u32", "upper_vertex_u32"),
    ),
    "persistence_disposition": TableSpec(
        "persistence_disposition", 4,
        (_f("interval_id_digest_128", "BYTES16", 16), _f("birth_row_id_digest_128", "BYTES16", 16),
         _f("death_row_id_digest_128", "BYTES16", 16), _f("arm_u8", "U8", 1),
         _f("status_u8", "U8", 1), _f("reason_u16", "U16_LE", 2),
         _f("birth_f_u16", "U16_LE", 2), _f("death_f_or_essential_u16", "U16_LE", 2),
         _f("lifetime_u16", "U16_LE", 2), _f("scale_mask_u8", "U8", 1),
         _f("spatial_face_edge_count_u32", "U32_LE", 4), _f("flags_u8", "U8", 1)),
        64, ("arm_u8", "birth_vertex_index_u32"),
    ),
    "rejection": TableSpec(
        "rejection", 5,
        (_f("row_id_digest_128", "BYTES16", 16), _f("arm_u8", "U8", 1),
         _f("scale_u8", "U8", 1), _f("first_rejection_u16", "U16_LE", 2),
         _f("stage_mask_u32", "U32_LE", 4), _f("denominator_id_u16", "U16_LE", 2),
         _f("reserved_zero_bytes6", "BYTES6", 6)),
        32, ("arm_u8", "scale_u8", "morton_u64"),
    ),
}
TABLE_NAMES = tuple(TABLE_SPECS)


for _spec in TABLE_SPECS.values():
    if _spec.struct.size != _spec.row_size or sum(field.size for field in _spec.fields) != _spec.row_size:
        raise RuntimeError(f"invalid frozen row definition: {_spec.name}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContainerValidationError(f"metadata is not canonical-JSON encodable: {exc}") from exc


def _sha(*parts: bytes) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def case_token_digest(raw_source_sha256: bytes) -> bytes:
    return _sha(_TAG_CASE, _exact_bytes(raw_source_sha256, 32, "raw_source_sha256"))


def node_id(case_digest: bytes, depth: int, morton_prefix: int) -> bytes:
    return _sha(_TAG_NODE, _exact_bytes(case_digest, 32, "case_digest"), _pack_uint(depth, 1), _pack_uint(morton_prefix, 8))


def cell_id(case_digest: bytes, morton: int) -> bytes:
    return _sha(_TAG_CELL, _exact_bytes(case_digest, 32, "case_digest"), _pack_uint(morton, 8))


def field_row_id(case_digest: bytes, arm: int, scale: int, morton: int) -> bytes:
    return _sha(_TAG_FIELD_ROW, _exact_bytes(case_digest, 32, "case_digest"),
                _pack_uint(arm, 1), _pack_uint(scale, 1), _pack_uint(morton, 8))[:16]


def edge_id(case_digest: bytes, lower: int, upper: int, kind: int, axis: int) -> bytes:
    return _sha(_TAG_EDGE, _exact_bytes(case_digest, 32, "case_digest"), _pack_uint(lower, 4),
                _pack_uint(upper, 4), _pack_uint(kind, 1), _pack_uint(axis, 1))[:16]


def essential_death_row_id(case_digest: bytes, arm: int, birth_row_id: bytes) -> bytes:
    return _sha(_TAG_ESSENTIAL_DEATH, _exact_bytes(case_digest, 32, "case_digest"),
                _pack_uint(arm, 1), _exact_bytes(birth_row_id, 16, "birth_row_id"))[:16]


def interval_id(
    case_digest: bytes, arm: int, birth_row_id: bytes, death_row_id: bytes, birth_f: int, death_f: int,
) -> bytes:
    return _sha(_TAG_INTERVAL, _exact_bytes(case_digest, 32, "case_digest"), _pack_uint(arm, 1),
                _exact_bytes(birth_row_id, 16, "birth_row_id"),
                _exact_bytes(death_row_id, 16, "death_row_id"), _pack_uint(birth_f, 2),
                _pack_uint(death_f, 2))[:16]


def _pack_uint(value: int, size: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << (8 * size):
        raise ContainerValidationError(f"unsigned {size}-byte value out of range")
    return value.to_bytes(size, "little", signed=False)


def _exact_bytes(value: Any, size: int, label: str) -> bytes:
    if hasattr(value, "tobytes") and not isinstance(value, (bytes, bytearray, memoryview)):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ContainerValidationError(f"{label} must be {size} bytes")
    result = bytes(value)
    if len(result) != size:
        raise ContainerValidationError(f"{label} must be exactly {size} bytes")
    return result


def _plain_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContainerValidationError("integer row field required")
    return value


def _row_value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        try:
            return row[name]
        except KeyError as exc:
            raise ContainerValidationError(f"row is missing field {name}") from exc
    try:
        return row[name]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ContainerValidationError(f"row is missing field {name}") from exc


def pack_row(table_name: str, row: Any) -> bytes:
    spec = _table_spec(table_name)
    values: list[Any] = []
    for field in spec.fields:
        value = _row_value(row, field.name)
        if field.kind.startswith("BYTES"):
            values.append(_exact_bytes(value, field.size, field.name))
        else:
            value = _plain_int(value)
            bits = field.size * 8
            signed = field.kind.startswith("I")
            lo, hi = (-(1 << (bits - 1)), 1 << (bits - 1)) if signed else (0, 1 << bits)
            if not lo <= value < hi:
                raise ContainerValidationError(f"{field.name} is outside {field.kind}")
            values.append(value)
    try:
        packed = spec.struct.pack(*values)
    except struct.error as exc:
        raise ContainerValidationError(f"cannot pack {table_name} row: {exc}") from exc
    _validate_row(table_name, dict(zip((field.name for field in spec.fields), values)))
    return packed


def unpack_row(table_name: str, packed: bytes) -> dict[str, Any]:
    spec = _table_spec(table_name)
    packed = _exact_bytes(packed, spec.row_size, f"{table_name} row")
    row = dict(zip((field.name for field in spec.fields), spec.struct.unpack(packed)))
    _validate_row(table_name, row)
    return row


def numpy_dtype(table_name: str) -> Any:
    """Return the exact unaligned little-endian NumPy dtype for a frozen table."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - project runtime includes NumPy
        raise RuntimeError("NumPy is required for numpy_dtype") from exc
    mapping = {
        "BYTES6": "V6", "BYTES16": "V16", "BYTES32": "V32", "U8": "u1",
        "U16_LE": "<u2", "U32_LE": "<u4", "U64_LE": "<u8", "I32_LE": "<i4", "I64_LE": "<i8",
    }
    spec = _table_spec(table_name)
    dtype = np.dtype([(field.name, mapping[field.kind]) for field in spec.fields], align=False)
    if dtype.itemsize != spec.row_size:
        raise RuntimeError(f"NumPy dtype drift for {table_name}")
    return dtype


def canonical_sort_key(table_name: str, row: Any) -> tuple[int, ...]:
    spec = _table_spec(table_name)
    try:
        return tuple(_plain_int(_row_value(row, name)) for name in spec.order_fields)
    except ContainerValidationError:
        raise
    except Exception as exc:
        raise ContainerValidationError(f"missing canonical order hook for {table_name}") from exc


def encode_table_rows(
    table_name: str,
    rows: Iterable[Any],
    *,
    order_key_hook: Callable[[Any], tuple[int, ...]] | None = None,
) -> bytes:
    materialized = list(rows)
    if order_key_hook is not None:
        materialized.sort(key=order_key_hook)
    elif table_name not in {"persistence_disposition", "rejection"}:
        materialized.sort(key=lambda row: canonical_sort_key(table_name, row))
    return b"".join(pack_row(table_name, row) for row in materialized)


def _table_rows_from_input(value: Any) -> list[Any]:
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        return list(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ContainerValidationError("packed table blobs are not accepted; supply fixed-width rows")
    return list(value)


def _encode_table_input(
    table_name: str,
    value: Any,
    *,
    order_key_hook: Callable[[Any], tuple[int, ...]] | None,
) -> tuple[bytes, int]:
    """Use a zero-object fixed-array path for production structured arrays."""

    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("NumPy is required for fixed-array containers") from exc
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype != numpy_dtype(table_name) or not array.flags.c_contiguous:
            raise ContainerValidationError(f"{table_name} must use the exact contiguous frozen dtype")
        if order_key_hook is not None:
            raise ContainerValidationError("Python order hooks are forbidden for fixed-array production input")
        _validate_array_rows(table_name, array)
        return array.tobytes(order="C"), len(array)
    rows = _table_rows_from_input(value)
    return encode_table_rows(table_name, rows, order_key_hook=order_key_hook), len(rows)


def encode_container(
    metadata: Mapping[str, Any],
    tables: Mapping[str, Iterable[Any]],
    *,
    order_key_hooks: Mapping[str, Callable[[Any], tuple[int, ...]]] | None = None,
) -> bytes:
    if set(tables) != set(TABLE_NAMES):
        raise ContainerValidationError("exactly the six frozen tables are required")
    hooks = dict(order_key_hooks or {})
    unknown_hooks = set(hooks) - set(TABLE_NAMES)
    if unknown_hooks:
        raise ContainerValidationError(f"unknown table-order hooks: {sorted(unknown_hooks)}")

    table_bytes: dict[str, bytes] = {}
    row_counts: dict[str, int] = {}
    for name in TABLE_NAMES:
        table_bytes[name], row_counts[name] = _encode_table_input(
            name, tables[name], order_key_hook=hooks.get(name)
        )

    descriptors = _make_descriptors(table_bytes, row_counts)
    total_size = METADATA_REGION_BYTES + sum(len(table_bytes[name]) for name in TABLE_NAMES)
    if total_size > MAX_SERIALIZED_BYTES:
        raise ContainerValidationError("serialized container exceeds 64 MiB")

    built = copy.deepcopy(dict(metadata))
    _set_computed(built, "table_descriptors", descriptors)
    _set_computed(built, "table_directory_sha256", table_directory_digest(descriptors).hex())
    partition_digest = partition_commitment(row_counts["partition_node"], table_bytes["partition_node"]).hex()
    _set_computed(built, "partition_commitment_sha256", partition_digest)
    if "resource" not in built or not isinstance(built["resource"], dict):
        raise ContainerValidationError("metadata.resource object is required")
    if "serialized_bytes" in built["resource"]:
        _u64_json(built["resource"]["serialized_bytes"], "resource.serialized_bytes")
    _set_computed(built["resource"], "serialized_bytes", total_size)
    if "raw_source_sha256" not in built:
        raise ContainerValidationError("raw_source_sha256 is required")
    raw_digest = _hex_digest_bytes(built["raw_source_sha256"], "raw_source_sha256")
    _set_computed(built, "case_token_digest_sha256", case_token_digest(raw_digest).hex())

    region = encode_metadata_region(built)
    container = region + b"".join(table_bytes[name] for name in TABLE_NAMES)
    decode_container(container, materialize_rows=False)
    return container


def encode_metadata_region(metadata: Mapping[str, Any]) -> bytes:
    payload = canonical_json_bytes(metadata)
    if len(payload) > MAX_METADATA_JSON_BYTES:
        raise ContainerValidationError("canonical metadata exceeds the fixed one-MiB region")
    return _U64.pack(len(payload)) + payload + bytes(MAX_METADATA_JSON_BYTES - len(payload))


def decode_metadata_region(region: bytes) -> dict[str, Any]:
    region = _exact_bytes(region, METADATA_REGION_BYTES, "metadata region")
    payload_length = _U64.unpack_from(region, 0)[0]
    if payload_length > MAX_METADATA_JSON_BYTES:
        raise ContainerValidationError("metadata length prefix exceeds fixed region")
    end = 8 + payload_length
    payload = region[8:end]
    if any(region[end:]):
        raise ContainerValidationError("metadata padding contains a nonzero byte")
    try:
        text = payload.decode("utf-8")
        metadata = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ContainerValidationError) as exc:
        raise ContainerValidationError(f"invalid metadata JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ContainerValidationError("metadata JSON must be an object")
    if canonical_json_bytes(metadata) != payload:
        raise ContainerValidationError("metadata JSON is not byte-exact canonical JSON")
    return metadata


def decode_container(
    data: bytes,
    *,
    expected_code_sha256: str | None = None,
    expected_dependency_lock_sha256: str | None = None,
    materialize_rows: bool = True,
) -> DecodedContainer:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContainerValidationError("container must be bytes")
    data = bytes(data)
    if len(data) < METADATA_REGION_BYTES:
        raise ContainerValidationError("container is shorter than metadata region")
    if len(data) > MAX_SERIALIZED_BYTES:
        raise ContainerValidationError("serialized container exceeds 64 MiB")
    metadata = decode_metadata_region(data[:METADATA_REGION_BYTES])
    _validate_metadata_shape(metadata)
    if expected_code_sha256 is not None and metadata["code_sha256"] != expected_code_sha256:
        raise ContainerValidationError("code SHA-256 does not match signed lock")
    if expected_dependency_lock_sha256 is not None and metadata["dependency_lock_sha256"] != expected_dependency_lock_sha256:
        raise ContainerValidationError("dependency-lock SHA-256 does not match signed lock")

    descriptors = metadata["table_descriptors"]
    expected_offset = METADATA_REGION_BYTES
    table_bytes: dict[str, bytes] = {}
    rows: dict[str, Any] = {}
    for name, descriptor in zip(TABLE_NAMES, descriptors):
        spec = TABLE_SPECS[name]
        if descriptor["name_enum"] != spec.name_enum or descriptor["row_size_u32"] != spec.row_size:
            raise ContainerValidationError(f"descriptor identity mismatch for {name}")
        if descriptor["offset_u64"] != expected_offset:
            raise ContainerValidationError(f"noncontiguous table offset for {name}")
        byte_length = descriptor["byte_length_u64"]
        if byte_length != descriptor["row_count_u64"] * spec.row_size:
            raise ContainerValidationError(f"declared byte length mismatch for {name}")
        end = expected_offset + byte_length
        if end > len(data):
            raise ContainerValidationError(f"truncated table {name}")
        payload = data[expected_offset:end]
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ContainerValidationError(f"table SHA-256 mismatch for {name}")
        if materialize_rows:
            decoded_rows = tuple(
                unpack_row(name, payload[offset: offset + spec.row_size])
                for offset in range(0, len(payload), spec.row_size)
            )
        else:
            import numpy as np

            decoded_rows = np.frombuffer(payload, dtype=numpy_dtype(name))
            _validate_array_rows(name, decoded_rows)
        table_bytes[name] = payload
        rows[name] = decoded_rows
        expected_offset = end
    if expected_offset != len(data):
        raise ContainerValidationError("container has trailing bytes")
    if metadata["resource"]["serialized_bytes"] != len(data):
        raise ContainerValidationError("serialized-byte resource record mismatch")
    if metadata["resource"]["resource_cap_tripped"]:
        raise ContainerValidationError("resource cap is marked tripped")
    if table_directory_digest(descriptors).hex() != metadata["table_directory_sha256"]:
        raise ContainerValidationError("table-directory commitment mismatch")
    if partition_commitment(len(rows["partition_node"]), table_bytes["partition_node"]).hex() != metadata["partition_commitment_sha256"]:
        raise ContainerValidationError("partition commitment mismatch")
    raw_digest = _hex_digest_bytes(metadata["raw_source_sha256"], "raw_source_sha256")
    if case_token_digest(raw_digest).hex() != metadata["case_token_digest_sha256"]:
        raise ContainerValidationError("case-token digest mismatch")
    if materialize_rows:
        _validate_domains(metadata, rows)
    else:
        _validate_domains_arrays(metadata, rows)
    return DecodedContainer(metadata, table_bytes, rows)


def validate_container(
    data: bytes,
    *,
    expected_code_sha256: str | None = None,
    expected_dependency_lock_sha256: str | None = None,
) -> None:
    decode_container(
        data,
        expected_code_sha256=expected_code_sha256,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
        materialize_rows=False,
    )


def partition_commitment(row_count: int, canonical_partition_bytes: bytes) -> bytes:
    return _sha(_TAG_PARTITION, _pack_uint(row_count, 8), bytes(canonical_partition_bytes))


def table_directory_digest(descriptors: Sequence[Mapping[str, Any]]) -> bytes:
    if len(descriptors) != len(TABLE_NAMES):
        raise ContainerValidationError("table directory must contain six descriptors")
    payload = bytearray(_TAG_DIRECTORY + _U32.pack(len(descriptors)))
    for expected_name, descriptor in zip(TABLE_NAMES, descriptors):
        _validate_descriptor_shape(descriptor)
        spec = TABLE_SPECS[expected_name]
        payload.extend(_DIRECTORY_DESCRIPTOR.pack(
            descriptor["name_enum"], descriptor["row_count_u64"], descriptor["row_size_u32"],
            descriptor["offset_u64"], descriptor["byte_length_u64"],
            _hex_digest_bytes(descriptor["sha256"], "descriptor.sha256"),
        ))
        if descriptor["name_enum"] != spec.name_enum:
            raise ContainerValidationError("table directory is not in enum order")
    return hashlib.sha256(payload).digest()


def _make_descriptors(table_bytes: Mapping[str, bytes], row_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    offset = METADATA_REGION_BYTES
    descriptors: list[dict[str, Any]] = []
    for name in TABLE_NAMES:
        spec = TABLE_SPECS[name]
        payload = table_bytes[name]
        descriptor = {
            "name_enum": spec.name_enum,
            "row_count_u64": row_counts[name],
            "row_size_u32": spec.row_size,
            "offset_u64": offset,
            "byte_length_u64": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        descriptors.append(descriptor)
        offset += len(payload)
    return descriptors


def _set_computed(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target and target[key] != value:
        raise ContainerValidationError(f"supplied {key} disagrees with computed value")
    target[key] = value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContainerValidationError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _table_spec(table_name: str) -> TableSpec:
    try:
        return TABLE_SPECS[table_name]
    except KeyError as exc:
        raise ContainerValidationError(f"unknown table: {table_name}") from exc


def _hex_digest_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ContainerValidationError(f"{label} must be lowercase hexadecimal SHA-256")
    return bytes.fromhex(value)


def _require_exact_keys(value: Any, keys: Sequence[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ContainerValidationError(f"{label} keys are not the frozen exact set")


def _u64_json(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ContainerValidationError(f"{label} must be a u64 JSON integer")


_TOP_KEYS = (
    "schema_version", "case_token_digest_sha256", "raw_source_sha256", "configuration_sha256",
    "schema_sha256", "code_sha256", "dependency_lock_sha256", "partition_commitment_sha256",
    "table_directory_sha256", "source_points_serialized", "census", "resource", "attestation",
    "table_descriptors",
)
_CENSUS_KEYS = (
    "raw_records", "finite_records", "unique_records", "sampled_unique_records", "occupied_cells_O",
    "partition_rows", "field_rows", "adjacency_rows", "persistence_rows", "rejection_rows",
    "moment_valid_rows", "attributed_intervals", "isotropic_survivors",
)
_RESOURCE_KEYS = (
    "cpu_nanoseconds", "wall_nanoseconds", "peak_process_tree_rss_bytes", "scratch_bytes",
    "serialized_bytes", "watchdog_poll_milliseconds", "resource_cap_tripped",
)
_ATTESTATION_EXPECTED = {
    "semantic_core_imported_by_independent_validator": False,
    "filename_path_family_identity_semantics_used": False,
    "gold_truth_cad_render_benchmark_accessed": False,
    "source_points_serialized": False,
    "top_k_reservoir_tail_omission_or_clipping_used": False,
    "complete_baseline_and_isotropic_payload": True,
    "deterministic_rebuild_match": True,
    "input_order_match": True,
    "independent_validator_match": True,
    "all_truncated_digest_collisions_checked": True,
}
_DESCRIPTOR_KEYS = ("name_enum", "row_count_u64", "row_size_u32", "offset_u64", "byte_length_u64", "sha256")


def _validate_metadata_shape(metadata: dict[str, Any]) -> None:
    _require_exact_keys(metadata, _TOP_KEYS, "metadata")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ContainerValidationError("schema version mismatch")
    if metadata["configuration_sha256"] != EXPECTED_CONFIGURATION_SHA256:
        raise ContainerValidationError("configuration SHA-256 mismatch")
    if metadata["schema_sha256"] != EXPECTED_SCHEMA_SHA256:
        raise ContainerValidationError("schema SHA-256 mismatch")
    for key in ("case_token_digest_sha256", "raw_source_sha256", "configuration_sha256", "schema_sha256",
                "code_sha256", "dependency_lock_sha256", "partition_commitment_sha256", "table_directory_sha256"):
        _hex_digest_bytes(metadata[key], key)
    if metadata["source_points_serialized"] is not False:
        raise ContainerValidationError("source_points_serialized must be false")
    _require_exact_keys(metadata["census"], _CENSUS_KEYS, "census")
    for key, value in metadata["census"].items():
        _u64_json(value, f"census.{key}")
    _require_exact_keys(metadata["resource"], _RESOURCE_KEYS, "resource")
    for key, value in metadata["resource"].items():
        if key == "resource_cap_tripped":
            if type(value) is not bool:
                raise ContainerValidationError("resource.resource_cap_tripped must be boolean")
        else:
            _u64_json(value, f"resource.{key}")
    _require_exact_keys(metadata["attestation"], tuple(_ATTESTATION_EXPECTED), "attestation")
    if metadata["attestation"] != _ATTESTATION_EXPECTED:
        raise ContainerValidationError("attestation does not equal the frozen expected values")
    if not isinstance(metadata["table_descriptors"], list) or len(metadata["table_descriptors"]) != 6:
        raise ContainerValidationError("exactly six table descriptors are required")
    for descriptor in metadata["table_descriptors"]:
        _validate_descriptor_shape(descriptor)


def _validate_descriptor_shape(descriptor: Mapping[str, Any]) -> None:
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, "table descriptor")
    for key in ("name_enum", "row_count_u64", "row_size_u32", "offset_u64", "byte_length_u64"):
        _u64_json(descriptor[key], f"descriptor.{key}")
    if descriptor["name_enum"] > 255 or descriptor["row_size_u32"] >= 1 << 32:
        raise ContainerValidationError("descriptor integer width overflow")
    _hex_digest_bytes(descriptor["sha256"], "descriptor.sha256")


def _validate_row(table_name: str, row: Mapping[str, Any]) -> None:
    if table_name == "partition_node":
        if row["status_u16"] != 0 or row["reserved_zero_u32"] != 0:
            raise ContainerValidationError("partition status/reserved field is noncanonical")
    elif table_name == "leaf_aggregate":
        if row["unique_point_count_u32"] == 0 or row["raw_record_multiplicity_sum_u32"] < row["unique_point_count_u32"]:
            raise ContainerValidationError("invalid occupied-leaf counts")
    elif table_name == "cell_scale_field":
        if row["arm_u8"] not in (0, 1) or row["scale_u8"] not in (0, 1, 2):
            raise ContainerValidationError("invalid field arm/scale")
        if row["status_u16"] not in (0, 1) or row["first_rejection_u16"] not in range(8):
            raise ContainerValidationError("invalid field status/rejection code")
        if row["flags_u16"] != 0 or row["stage_mask_u32"] & ~0x7F:
            raise ContainerValidationError("invalid field flags/stage mask")
        if row["growth12_gate_code_i32"] not in (-2_147_483_648, -1, 0, 1) or row["growth23_gate_code_i32"] not in (-2_147_483_648, -1, 0, 1):
            raise ContainerValidationError("invalid growth gate code")
        if row["eigenvalue_denominator_u32"] != (1 if row["arm_u8"] == 0 else 3):
            raise ContainerValidationError("field eigenvalue denominator disagrees with arm")
    elif table_name == "space_scale_adjacency":
        if row["status_u16"] != 0 or row["reserved_zero_u32"] != 0 or row["lower_vertex_u32"] >= row["upper_vertex_u32"]:
            raise ContainerValidationError("invalid adjacency status/endpoints")
        valid_axis = row["axis_u8"] in (0, 1, 2) if row["edge_kind_u8"] == 0 else row["axis_u8"] == 255
        if row["edge_kind_u8"] not in (0, 1) or not valid_axis:
            raise ContainerValidationError("invalid adjacency kind/axis")
    elif table_name == "persistence_disposition":
        if row["arm_u8"] not in (0, 1) or row["status_u8"] not in (0, 1) or row["reason_u16"] not in range(6):
            raise ContainerValidationError("invalid persistence code")
        if row["scale_mask_u8"] & ~0x07 or row["flags_u8"] != 0:
            raise ContainerValidationError("invalid persistence scale mask/flags")
        if row["status_u8"] == 0:
            if row["death_f_or_essential_u16"] == 65535 or row["death_f_or_essential_u16"] < row["birth_f_u16"]:
                raise ContainerValidationError("invalid finite persistence death")
            if row["lifetime_u16"] != row["death_f_or_essential_u16"] - row["birth_f_u16"]:
                raise ContainerValidationError("invalid finite persistence lifetime")
        elif row["death_f_or_essential_u16"] != 65535 or row["lifetime_u16"] != 65535 - row["birth_f_u16"]:
            raise ContainerValidationError("invalid essential persistence encoding")
    elif table_name == "rejection":
        if row["arm_u8"] not in (0, 1) or row["scale_u8"] not in (0, 1, 2):
            raise ContainerValidationError("invalid rejection arm/scale")
        if row["first_rejection_u16"] not in range(8) or row["denominator_id_u16"] not in range(9):
            raise ContainerValidationError("invalid rejection code")
        if row["stage_mask_u32"] & ~0x7F or row["reserved_zero_bytes6"] != bytes(6):
            raise ContainerValidationError("invalid rejection stage/reserved bytes")


def _validate_array_rows(table_name: str, rows: Any) -> None:
    """Vectorized row-contract validation without per-row Python objects."""

    import numpy as np

    if rows.dtype != numpy_dtype(table_name) or rows.ndim != 1:
        raise ContainerValidationError(f"{table_name} array dtype/domain mismatch")
    if table_name == "partition_node":
        if np.any(rows["status_u16"] != 0) or np.any(rows["reserved_zero_u32"] != 0):
            raise ContainerValidationError("partition status/reserved field is noncanonical")
    elif table_name == "leaf_aggregate":
        if np.any(rows["unique_point_count_u32"] == 0) or np.any(
            rows["raw_record_multiplicity_sum_u32"] < rows["unique_point_count_u32"]
        ):
            raise ContainerValidationError("invalid occupied-leaf counts")
    elif table_name == "cell_scale_field":
        arm, scale = rows["arm_u8"], rows["scale_u8"]
        rejection, status, stage = (
            rows["first_rejection_u16"], rows["status_u16"], rows["stage_mask_u32"]
        )
        if np.any(arm > 1) or np.any(scale > 2) or np.any(status > 1) or np.any(rejection > 7):
            raise ContainerValidationError("invalid field enum")
        if np.any(rows["flags_u16"] != 0) or np.any(stage & np.uint32(0xFFFF_FF80)):
            raise ContainerValidationError("invalid field flags/stage mask")
        allowed_growth = np.isin(rows["growth12_gate_code_i32"], (NOT_EVALUATED_I32, -1, 0, 1)) & np.isin(
            rows["growth23_gate_code_i32"], (NOT_EVALUATED_I32, -1, 0, 1)
        )
        if not np.all(allowed_growth):
            raise ContainerValidationError("invalid growth gate code")
        if np.any(rows["eigenvalue_denominator_u32"] != np.where(arm == 0, 1, 3)):
            raise ContainerValidationError("field eigenvalue denominator disagrees with arm")
        expected_stage = np.asarray((0x7F, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F), dtype=np.uint32)[rejection]
        if np.any(stage != expected_stage) or np.any(status != (rejection != 0).astype(np.uint16)):
            raise ContainerValidationError("field rejection/status/stage relation is noncanonical")
        before_growth = (rejection >= 1) & (rejection < 4)
        if np.any(rows["growth12_gate_code_i32"][before_growth] != NOT_EVALUATED_I32) or np.any(
            rows["growth23_gate_code_i32"][before_growth] != NOT_EVALUATED_I32
        ):
            raise ContainerValidationError("unevaluated growth sentinel mismatch")
        after_growth = (rejection == 0) | (rejection > 4)
        if np.any(rows["growth12_gate_code_i32"][after_growth] != 0) or np.any(
            rows["growth23_gate_code_i32"][after_growth] != 0
        ):
            raise ContainerValidationError("passed growth code mismatch")
        growth_failure = rejection == 4
        if np.any(
            (rows["growth12_gate_code_i32"][growth_failure] == 0)
            & (rows["growth23_gate_code_i32"][growth_failure] == 0)
        ):
            raise ContainerValidationError("growth rejection has no failing gate")
        before_density = (rejection >= 1) & (rejection < 5)
        if np.any(rows["raw_count_p10_u32"][before_density] != 0) or np.any(
            rows["raw_count_p90_u32"][before_density] != 0
        ):
            raise ContainerValidationError("unevaluated density evidence is nonzero")
        before_balance = (rejection >= 1) & (rejection < 3)
        if np.any(rows["balance_numerator_u32"][before_balance] != 0) or np.any(
            rows["balance_denominator_u32"][before_balance] != 0
        ):
            raise ContainerValidationError("unevaluated balance evidence is nonzero")
        invalid_before_numeric = (rejection >= 1) & (rejection < 7)
        if np.any(rows["trace_q_i64"][invalid_before_numeric] != 0):
            raise ContainerValidationError("unevaluated trace evidence is nonzero")
        invalid = rejection != 0
        if np.any(rows["anisotropy_q_u16"][invalid] != 0) or np.any(
            rows["filtration_u16"][invalid] != 65_534
        ):
            raise ContainerValidationError("invalid-row anisotropy sentinel mismatch")
        baseline_invalid = invalid & (arm == 0)
        if any(np.any(rows[name][baseline_invalid] != 0) for name in (
            "lambda1_q_i64", "lambda2_q_i64", "lambda3_q_i64"
        )):
            raise ContainerValidationError("invalid baseline row retained unevaluated eigenvalues")
        if np.any(rows["mass_q_i64"] != rows["support_count_u32"].astype(np.int64)):
            raise ContainerValidationError("field mass/support mismatch")
    elif table_name == "space_scale_adjacency":
        kind, axis = rows["edge_kind_u8"], rows["axis_u8"]
        if (
            np.any(rows["status_u16"] != 0)
            or np.any(rows["reserved_zero_u32"] != 0)
            or np.any(rows["lower_vertex_u32"] >= rows["upper_vertex_u32"])
            or np.any(kind > 1)
            or np.any((kind == 0) & (axis > 2))
            or np.any((kind == 1) & (axis != 255))
        ):
            raise ContainerValidationError("invalid adjacency row")
    elif table_name == "persistence_disposition":
        finite = rows["status_u8"] == 0
        essential = rows["status_u8"] == 1
        if (
            np.any(rows["arm_u8"] > 1)
            or np.any(~(finite | essential))
            or np.any(rows["reason_u16"] > 5)
            or np.any(rows["scale_mask_u8"] & np.uint8(0xF8))
            or np.any(rows["flags_u8"] != 0)
        ):
            raise ContainerValidationError("invalid persistence enum/flags")
        if np.any(rows["death_f_or_essential_u16"][finite] == 65_535) or np.any(
            rows["death_f_or_essential_u16"][finite] < rows["birth_f_u16"][finite]
        ) or np.any(
            rows["lifetime_u16"][finite]
            != rows["death_f_or_essential_u16"][finite] - rows["birth_f_u16"][finite]
        ):
            raise ContainerValidationError("invalid finite persistence encoding")
        if np.any(rows["death_f_or_essential_u16"][essential] != 65_535) or np.any(
            rows["lifetime_u16"][essential] != 65_535 - rows["birth_f_u16"][essential]
        ):
            raise ContainerValidationError("invalid essential persistence encoding")
    elif table_name == "rejection":
        if (
            np.any(rows["arm_u8"] > 1)
            or np.any(rows["scale_u8"] > 2)
            or np.any(rows["first_rejection_u16"] > 7)
            or np.any(rows["denominator_id_u16"] > 8)
            or np.any(rows["stage_mask_u32"] & np.uint32(0xFFFF_FF80))
            or np.any(rows["reserved_zero_bytes6"] != np.void(bytes(6)))
        ):
            raise ContainerValidationError("invalid rejection row")


def _validate_strict_order(table_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [tuple(row[name] for name in TABLE_SPECS[table_name].order_fields) for row in rows]
    if any(left >= right for left, right in zip(keys, keys[1:])):
        raise ContainerValidationError(f"{table_name} is not in strict canonical order")


def _validate_domains(metadata: Mapping[str, Any], rows: Mapping[str, tuple[dict[str, Any], ...]]) -> None:
    census = metadata["census"]
    counts = {name: len(rows[name]) for name in TABLE_NAMES}
    o = census["occupied_cells_O"]
    if not 0 < o <= 32_768:
        raise ContainerValidationError("occupied cell count is outside frozen bound")
    expected_census = {
        "partition_rows": counts["partition_node"], "field_rows": counts["cell_scale_field"],
        "adjacency_rows": counts["space_scale_adjacency"], "persistence_rows": counts["persistence_disposition"],
        "rejection_rows": counts["rejection"],
    }
    if any(census[key] != value for key, value in expected_census.items()):
        raise ContainerValidationError("census table-row count mismatch")
    if counts["leaf_aggregate"] != o or not o <= counts["partition_node"] <= 2 * o - 1:
        raise ContainerValidationError("leaf/partition actual-O domain mismatch")
    if counts["cell_scale_field"] != 6 * o or counts["persistence_disposition"] != 6 * o or counts["rejection"] != 6 * o:
        raise ContainerValidationError("complete two-arm field/disposition/rejection domain mismatch")
    if not 2 * o <= counts["space_scale_adjacency"] <= 11 * o:
        raise ContainerValidationError("adjacency actual-O domain mismatch")
    if sum(row["unique_point_count_u32"] for row in rows["leaf_aggregate"]) != census["sampled_unique_records"]:
        raise ContainerValidationError("sampled-point leaf census mismatch")

    for name in ("partition_node", "leaf_aggregate", "cell_scale_field", "space_scale_adjacency"):
        _validate_strict_order(name, rows[name])

    case_digest = bytes.fromhex(metadata["case_token_digest_sha256"])
    for row in rows["partition_node"]:
        if row["node_id_sha256"] != node_id(case_digest, row["depth_u8"], row["morton_prefix_u64"]):
            raise ContainerValidationError("partition node ID mismatch")
    for row in rows["leaf_aggregate"]:
        if row["cell_id_sha256"] != cell_id(case_digest, row["morton_u64"]):
            raise ContainerValidationError("cell ID mismatch")
    for row in rows["cell_scale_field"]:
        if row["row_id_digest_128"] != field_row_id(case_digest, row["arm_u8"], row["scale_u8"], row["morton_u64"]):
            raise ContainerValidationError("field-row ID mismatch")
    for row in rows["space_scale_adjacency"]:
        expected = edge_id(case_digest, row["lower_vertex_u32"], row["upper_vertex_u32"], row["edge_kind_u8"], row["axis_u8"])
        if row["edge_id_digest_128"] != expected:
            raise ContainerValidationError("edge ID mismatch")

    field_rows = rows["cell_scale_field"]
    arm_size = 3 * o
    for baseline, isotropic in zip(field_rows[:arm_size], field_rows[arm_size:]):
        if baseline["arm_u8"] != 0 or isotropic["arm_u8"] != 1:
            raise ContainerValidationError("field arms are not canonical")
        if (baseline["scale_u8"], baseline["morton_u64"]) != (isotropic["scale_u8"], isotropic["morton_u64"]):
            raise ContainerValidationError("baseline/isotropic source-cell correspondence mismatch")
        if baseline["row_id_digest_128"] == isotropic["row_id_digest_128"]:
            raise ContainerValidationError("cross-arm field-row digest collision")

    rejection_rows = rows["rejection"]
    persistence_rows = rows["persistence_disposition"]
    if len(rejection_rows) != len(field_rows) or len(persistence_rows) != len(field_rows):
        raise ContainerValidationError("arm-bound companion table domain mismatch")
    adjacency_ids = {row["edge_id_digest_128"] for row in rows["space_scale_adjacency"]}
    for field, rejection, persistence in zip(field_rows, rejection_rows, persistence_rows):
        if (rejection["arm_u8"], rejection["scale_u8"], rejection["row_id_digest_128"]) != (
            field["arm_u8"], field["scale_u8"], field["row_id_digest_128"],
        ):
            raise ContainerValidationError("rejection row is not the own-arm field-row alias")
        if persistence["arm_u8"] != field["arm_u8"] or persistence["birth_row_id_digest_128"] != field["row_id_digest_128"]:
            raise ContainerValidationError("persistence row is not in canonical birth-vertex order")
        if persistence["status_u8"] == 0:
            if persistence["death_row_id_digest_128"] not in adjacency_ids:
                raise ContainerValidationError("finite death edge is absent from adjacency table")
        else:
            expected_death = essential_death_row_id(case_digest, persistence["arm_u8"], persistence["birth_row_id_digest_128"])
            if persistence["death_row_id_digest_128"] != expected_death:
                raise ContainerValidationError("essential death ID mismatch")
        expected_interval = interval_id(
            case_digest, persistence["arm_u8"], persistence["birth_row_id_digest_128"],
            persistence["death_row_id_digest_128"], persistence["birth_f_u16"],
            persistence["death_f_or_essential_u16"],
        )
        if persistence["interval_id_digest_128"] != expected_interval:
            raise ContainerValidationError("interval ID mismatch")


def _strict_array_order(rows: Any, fields: Sequence[str], table_name: str) -> None:
    import numpy as np

    if len(rows) < 2:
        return
    keys = tuple(rows[name] for name in reversed(fields))
    order = np.lexsort(keys)
    if not np.array_equal(order, np.arange(len(rows))):
        raise ContainerValidationError(f"{table_name} is not in strict canonical order")
    equal = np.ones(len(rows) - 1, dtype=np.bool_)
    for name in fields:
        equal &= rows[name][1:] == rows[name][:-1]
    if np.any(equal):
        raise ContainerValidationError(f"{table_name} contains duplicate canonical keys")


def _validate_domains_arrays(metadata: Mapping[str, Any], rows: Mapping[str, Any]) -> None:
    """Complete actual-O and arm-binding validation over fixed arrays."""

    import numpy as np

    census = metadata["census"]
    counts = {name: len(rows[name]) for name in TABLE_NAMES}
    o = census["occupied_cells_O"]
    if not 0 < o <= 32_768:
        raise ContainerValidationError("occupied cell count is outside frozen bound")
    expected_census = {
        "partition_rows": counts["partition_node"],
        "field_rows": counts["cell_scale_field"],
        "adjacency_rows": counts["space_scale_adjacency"],
        "persistence_rows": counts["persistence_disposition"],
        "rejection_rows": counts["rejection"],
    }
    if any(census[key] != value for key, value in expected_census.items()):
        raise ContainerValidationError("census table-row count mismatch")
    if counts["leaf_aggregate"] != o or not o <= counts["partition_node"] <= 2 * o - 1:
        raise ContainerValidationError("leaf/partition actual-O domain mismatch")
    if counts["cell_scale_field"] != 6 * o or counts["persistence_disposition"] != 6 * o or counts["rejection"] != 6 * o:
        raise ContainerValidationError("complete two-arm companion domain mismatch")
    if not 2 * o <= counts["space_scale_adjacency"] <= 11 * o:
        raise ContainerValidationError("adjacency actual-O domain mismatch")
    leaves = rows["leaf_aggregate"]
    if int(np.sum(leaves["unique_point_count_u32"], dtype=np.uint64)) != census["sampled_unique_records"]:
        raise ContainerValidationError("sampled-point leaf census mismatch")
    _strict_array_order(rows["partition_node"], ("depth_u8", "morton_prefix_u64"), "partition_node")
    _strict_array_order(leaves, ("morton_u64",), "leaf_aggregate")
    fields = rows["cell_scale_field"]
    _strict_array_order(fields, ("arm_u8", "scale_u8", "morton_u64"), "cell_scale_field")
    adjacency = rows["space_scale_adjacency"]
    _strict_array_order(
        adjacency,
        ("edge_kind_u8", "axis_u8", "lower_vertex_u32", "upper_vertex_u32"),
        "space_scale_adjacency",
    )

    case_digest = bytes.fromhex(metadata["case_token_digest_sha256"])
    # Full-hash identity loops allocate no retained row objects; the table
    # domains themselves remain contiguous fixed-width arrays.
    for index in range(len(rows["partition_node"])):
        row = rows["partition_node"][index]
        if bytes(row["node_id_sha256"]) != node_id(case_digest, int(row["depth_u8"]), int(row["morton_prefix_u64"])):
            raise ContainerValidationError("partition node ID mismatch")
    for index in range(o):
        if bytes(leaves["cell_id_sha256"][index]) != cell_id(case_digest, int(leaves["morton_u64"][index])):
            raise ContainerValidationError("cell ID mismatch")
    for index in range(len(fields)):
        row = fields[index]
        if bytes(row["row_id_digest_128"]) != field_row_id(
            case_digest, int(row["arm_u8"]), int(row["scale_u8"]), int(row["morton_u64"])
        ):
            raise ContainerValidationError("field-row ID mismatch")
    for index in range(len(adjacency)):
        row = adjacency[index]
        if bytes(row["edge_id_digest_128"]) != edge_id(
            case_digest,
            int(row["lower_vertex_u32"]),
            int(row["upper_vertex_u32"]),
            int(row["edge_kind_u8"]),
            int(row["axis_u8"]),
        ):
            raise ContainerValidationError("edge ID mismatch")

    arm_size = 3 * o
    baseline, isotropic = fields[:arm_size], fields[arm_size:]
    if np.any(baseline["arm_u8"] != 0) or np.any(isotropic["arm_u8"] != 1):
        raise ContainerValidationError("field arms are not canonical")
    if np.any(baseline["scale_u8"] != isotropic["scale_u8"]) or np.any(
        baseline["morton_u64"] != isotropic["morton_u64"]
    ):
        raise ContainerValidationError("baseline/isotropic correspondence mismatch")
    if np.any(baseline["row_id_digest_128"] == isotropic["row_id_digest_128"]):
        raise ContainerValidationError("cross-arm field-row digest collision")
    preserved = (
        "support_membership_digest_128", "morton_u64", "trace_q_i64", "mass_q_i64",
        "status_u16", "support_count_u32", "raw_count_p10_u32", "raw_count_p90_u32",
        "balance_numerator_u32", "balance_denominator_u32", "growth12_gate_code_i32",
        "growth23_gate_code_i32", "stage_mask_u32", "first_rejection_u16", "flags_u16",
    )
    if any(np.any(baseline[name] != isotropic[name]) for name in preserved):
        raise ContainerValidationError("isotropic arm changed preserved payload")
    valid_baseline = baseline["first_rejection_u16"] == 0
    baseline_sum = baseline["lambda1_q_i64"] + baseline["lambda2_q_i64"] + baseline["lambda3_q_i64"]
    if np.any(baseline_sum[valid_baseline] != baseline["trace_q_i64"][valid_baseline]):
        raise ContainerValidationError("baseline trace/eigenvalue evidence mismatch")
    if any(np.any(isotropic[name] != baseline["trace_q_i64"]) for name in (
        "lambda1_q_i64", "lambda2_q_i64", "lambda3_q_i64"
    )) or np.any(isotropic["anisotropy_q_u16"] != 0) or np.any(isotropic["filtration_u16"] != 65_534):
        raise ContainerValidationError("isotropic exact-rational evidence mismatch")
    if np.any(
        baseline["anisotropy_q_u16"][valid_baseline] + baseline["filtration_u16"][valid_baseline] != 65_534
    ):
        raise ContainerValidationError("baseline q/filtration relation mismatch")

    rejection = rows["rejection"]
    persistence = rows["persistence_disposition"]
    if (
        np.any(rejection["row_id_digest_128"] != fields["row_id_digest_128"])
        or np.any(rejection["arm_u8"] != fields["arm_u8"])
        or np.any(rejection["scale_u8"] != fields["scale_u8"])
        or np.any(rejection["first_rejection_u16"] != fields["first_rejection_u16"])
        or np.any(rejection["stage_mask_u32"] != fields["stage_mask_u32"])
    ):
        raise ContainerValidationError("rejection companion alias mismatch")
    expected_denominator = np.zeros(len(fields), dtype=np.uint16)
    codes = fields["first_rejection_u16"]
    expected_denominator[codes == 1] = 1
    expected_denominator[codes == 2] = 2
    expected_denominator[codes == 3] = 3
    expected_denominator[(codes == 4) & (fields["growth12_gate_code_i32"] != 0)] = 4
    expected_denominator[(codes == 4) & (fields["growth12_gate_code_i32"] == 0)] = 5
    expected_denominator[codes == 5] = 6
    expected_denominator[codes == 6] = 7
    expected_denominator[codes == 7] = 8
    if np.any(rejection["denominator_id_u16"] != expected_denominator):
        raise ContainerValidationError("rejection denominator mismatch")
    if np.any(persistence["arm_u8"] != fields["arm_u8"]) or np.any(
        persistence["birth_row_id_digest_128"] != fields["row_id_digest_128"]
    ) or np.any(persistence["birth_f_u16"] != fields["filtration_u16"]):
        raise ContainerValidationError("persistence birth alias mismatch")
    adjacency_ids = np.sort(adjacency["edge_id_digest_128"].copy(), kind="stable")
    finite_mask = persistence["status_u8"] == 0
    finite_deaths = persistence["death_row_id_digest_128"][finite_mask]
    if len(finite_deaths):
        positions = np.searchsorted(adjacency_ids, finite_deaths)
        if np.any(positions >= len(adjacency_ids)) or np.any(
            adjacency_ids[np.minimum(positions, len(adjacency_ids) - 1)] != finite_deaths
        ):
            raise ContainerValidationError("finite death edge is absent")
    for index in range(len(persistence)):
        row = persistence[index]
        birth_id = bytes(row["birth_row_id_digest_128"])
        death_id = bytes(row["death_row_id_digest_128"])
        if int(row["status_u8"]) == 0:
            pass
        elif death_id != essential_death_row_id(case_digest, int(row["arm_u8"]), birth_id):
            raise ContainerValidationError("essential death ID mismatch")
        if bytes(row["interval_id_digest_128"]) != interval_id(
            case_digest,
            int(row["arm_u8"]),
            birth_id,
            death_id,
            int(row["birth_f_u16"]),
            int(row["death_f_or_essential_u16"]),
        ):
            raise ContainerValidationError("interval ID mismatch")
