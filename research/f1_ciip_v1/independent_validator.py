"""Independent frozen-V7 PLY and evidence-container validator.

This module deliberately does not import the primary parser or semantic core.
It rereads an already-open PLY file descriptor, reconstructs the frozen source
diagnostics, decodes the neutral fixed-width container, and independently
checks content bindings, complete domains, arm relationships and resource caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import stat
import struct
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from . import container
from .config import load_frozen_v7, validate_frozen_allocation_plan
from .independent_semantic_rebuild import (
    rebuild_semantic_tables,
    validate_independent_known_answer_vectors,
)


MAX_RAW_BYTES = 64_382_075
MAX_VERTEX_RECORDS = 1_140_033
SAMPLE_CAP = 200_000
CPU_NANOSECONDS_CAP = 300_000_000_000
WALL_NANOSECONDS_CAP = 300_000_000_000
RSS_WATCHDOG_BYTES = 12_884_901_888
ENTERPRISE_RSS_FORBIDDEN_BYTES = 17_179_869_184
SCRATCH_BYTES_CAP = 268_435_456
SERIALIZED_BYTES_CAP = 67_108_864
WATCHDOG_POLL_MILLISECONDS = 10
COORDINATE_MULTISET_DOMAIN = b"PWTS_COORDINATE_MULTISET_V1\0"

_SCALARS: dict[str, tuple[str, int]] = {
    "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
}
_INTEGER_SCALARS = frozenset(
    {"char", "int8", "uchar", "uint8", "short", "int16", "ushort", "uint16",
     "int", "int32", "uint", "uint32"}
)


class IndependentValidationError(ValueError):
    """Fail-closed independent validation error with a stable reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


@dataclass(frozen=True)
class SourceDiagnostics:
    raw_byte_count: int
    raw_ply_sha256: str
    ply_header_byte_count: int
    ply_header_sha256: str
    ply_format: str
    parsed_vertex_count: int
    finite_coordinate_count: int
    exact_duplicate_count: int
    ignored_property_name_count: int
    ignored_property_name_digest: str
    canonical_coordinate_multiset_sha256: str


@dataclass(frozen=True)
class ParsedSource:
    diagnostics: SourceDiagnostics
    points: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class ValidationSummary:
    raw_source_sha256: str
    case_token_digest_sha256: str
    occupied_cells: int
    sampled_unique_records: int
    serialized_bytes: int


def _fail(code: str, detail: str = "") -> None:
    raise IndependentValidationError(code, detail)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _read_exact(stream: BinaryIO, size: int, code: str) -> bytes:
    if size < 0 or size > MAX_RAW_BYTES:
        _fail("PLY_SIZE_OVERFLOW")
    raw = stream.read(size)
    if len(raw) != size:
        _fail(code)
    return raw


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1_048_576, size - offset), offset)
        if not chunk:
            _fail("PLY_CHANGED_DURING_READ")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _read_header(stream: BinaryIO, expected_bytes: int) -> tuple[bytes, str, list[dict[str, Any]]]:
    if not 1 <= expected_bytes <= MAX_RAW_BYTES:
        _fail("PLY_HEADER_BOUND_INVALID")
    lines: list[bytes] = []
    total = 0
    while total <= expected_bytes:
        line = stream.readline(expected_bytes - total + 1)
        if not line:
            _fail("PLY_HEADER_TRUNCATED")
        lines.append(line)
        total += len(line)
        if line.rstrip(b"\r\n") == b"end_header":
            break
    header = b"".join(lines)
    if len(header) != expected_bytes:
        _fail("PLY_HEADER_LENGTH_MISMATCH")
    try:
        decoded = [line.decode("ascii") for line in header.splitlines()]
    except UnicodeDecodeError as exc:
        raise IndependentValidationError("PLY_HEADER_NONASCII") from exc
    if not decoded or decoded[0] != "ply":
        _fail("PLY_MAGIC_INVALID")

    formats = [line.split() for line in decoded if line.startswith("format ")]
    if len(formats) != 1 or formats[0][2:] != ["1.0"]:
        _fail("PLY_FORMAT_DECLARATION_INVALID")
    ply_format = formats[0][1]
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        _fail("PLY_FORMAT_UNSUPPORTED")

    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in decoded[1:]:
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info", "format", "end_header"}:
            continue
        if parts[0] == "element":
            if len(parts) != 3:
                _fail("PLY_ELEMENT_DECLARATION_INVALID")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise IndependentValidationError("PLY_ELEMENT_COUNT_INVALID") from exc
            if count < 0:
                _fail("PLY_ELEMENT_COUNT_INVALID")
            current = {"name": parts[1], "count": count, "properties": []}
            elements.append(current)
        elif parts[0] == "property":
            if current is None:
                _fail("PLY_PROPERTY_WITHOUT_ELEMENT")
            if len(parts) == 3:
                if parts[1] not in _SCALARS:
                    _fail("PLY_SCALAR_TYPE_UNSUPPORTED")
                current["properties"].append(("scalar", parts[1], parts[2]))
            elif len(parts) == 5 and parts[1] == "list":
                if parts[2] not in _INTEGER_SCALARS or parts[3] not in _SCALARS:
                    _fail("PLY_LIST_TYPE_UNSUPPORTED")
                current["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                _fail("PLY_PROPERTY_DECLARATION_INVALID")
        else:
            _fail("PLY_HEADER_DIRECTIVE_UNSUPPORTED", parts[0])

    vertices = [element for element in elements if element["name"] == "vertex"]
    if len(vertices) != 1 or not elements or elements[0] is not vertices[0]:
        _fail("PLY_VERTEX_ELEMENT_INVALID")
    vertex = vertices[0]
    if not 0 < int(vertex["count"]) <= MAX_VERTEX_RECORDS:
        _fail("PLY_VERTEX_COUNT_OUT_OF_BOUND")
    scalar_names = [prop[2] for prop in vertex["properties"] if prop[0] == "scalar"]
    if any(scalar_names.count(axis) != 1 for axis in ("x", "y", "z")):
        _fail("PLY_XYZ_DECLARATION_INVALID")
    if any(prop[0] == "list" and prop[3] in {"x", "y", "z"} for prop in vertex["properties"]):
        _fail("PLY_XYZ_DECLARATION_INVALID")
    return header, ply_format, elements


def _parse_vertices(
    stream: BinaryIO, ply_format: str, vertex: Mapping[str, Any],
) -> np.ndarray:
    count = int(vertex["count"])
    properties: Sequence[Sequence[str]] = vertex["properties"]
    points = np.empty((count, 3), dtype=np.float64)
    axes = {"x": 0, "y": 1, "z": 2}
    if ply_format == "ascii":
        for row_index in range(count):
            line = stream.readline(MAX_RAW_BYTES + 1)
            if not line or len(line) > MAX_RAW_BYTES:
                _fail("PLY_ASCII_VERTEX_TRUNCATED")
            tokens = line.split()
            cursor = 0
            for prop in properties:
                if prop[0] == "scalar":
                    if cursor >= len(tokens):
                        _fail("PLY_ASCII_VERTEX_TRUNCATED")
                    if prop[2] in axes:
                        try:
                            points[row_index, axes[prop[2]]] = float(tokens[cursor])
                        except ValueError as exc:
                            raise IndependentValidationError("PLY_XYZ_VALUE_INVALID") from exc
                    cursor += 1
                else:
                    if cursor >= len(tokens):
                        _fail("PLY_ASCII_LIST_TRUNCATED")
                    try:
                        item_count = int(tokens[cursor])
                    except ValueError as exc:
                        raise IndependentValidationError("PLY_LIST_COUNT_INVALID") from exc
                    if item_count < 0 or cursor + 1 + item_count > len(tokens):
                        _fail("PLY_ASCII_LIST_TRUNCATED")
                    cursor += 1 + item_count
            if cursor != len(tokens):
                _fail("PLY_ASCII_VERTEX_EXTRA_FIELDS")
    else:
        endian = "<" if ply_format == "binary_little_endian" else ">"
        for row_index in range(count):
            for prop in properties:
                if prop[0] == "scalar":
                    fmt, size = _SCALARS[prop[1]]
                    value = struct.unpack(endian + fmt, _read_exact(stream, size, "PLY_BINARY_VERTEX_TRUNCATED"))[0]
                    if prop[2] in axes:
                        points[row_index, axes[prop[2]]] = float(value)
                else:
                    count_fmt, count_size = _SCALARS[prop[1]]
                    item_count = int(struct.unpack(
                        endian + count_fmt,
                        _read_exact(stream, count_size, "PLY_BINARY_LIST_TRUNCATED"),
                    )[0])
                    if item_count < 0:
                        _fail("PLY_LIST_COUNT_INVALID")
                    item_size = _SCALARS[prop[2]][1]
                    _read_exact(stream, item_count * item_size, "PLY_BINARY_LIST_TRUNCATED")
    return points


def _coordinate_diagnostics(points: np.ndarray) -> tuple[int, int, str]:
    finite_mask = np.isfinite(points).all(axis=1)
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count != len(points):
        _fail("NONFINITE_COORDINATE")
    points[points == 0.0] = 0.0
    little = np.ascontiguousarray(points, dtype="<f8")
    packed = little.view(np.dtype("V24")).reshape(-1)
    ordered = np.sort(packed, kind="stable")
    unique_count = 0 if not len(ordered) else 1 + int(np.count_nonzero(ordered[1:] != ordered[:-1]))
    digest = hashlib.sha256(COORDINATE_MULTISET_DOMAIN)
    digest.update(memoryview(ordered).cast("B"))
    return finite_count, finite_count - unique_count, digest.hexdigest()


def _compare_source(expected: Mapping[str, Any], observed: SourceDiagnostics) -> None:
    fields = {
        "raw_byte_count": observed.raw_byte_count,
        "raw_ply_sha256": observed.raw_ply_sha256,
        "ply_header_byte_count": observed.ply_header_byte_count,
        "ply_header_sha256": observed.ply_header_sha256,
        "ply_format": observed.ply_format,
        "parsed_vertex_count": observed.parsed_vertex_count,
        "finite_coordinate_count": observed.finite_coordinate_count,
        "exact_duplicate_count": observed.exact_duplicate_count,
        "ignored_property_name_count": observed.ignored_property_name_count,
        "ignored_property_name_digest": observed.ignored_property_name_digest,
    }
    for key, value in fields.items():
        if expected.get(key) != value:
            _fail("FROZEN_SOURCE_DIAGNOSTIC_MISMATCH", key)
    alias = expected.get("canonical_coordinate_multiset_digest")
    if alias != observed.canonical_coordinate_multiset_sha256:
        _fail("FROZEN_SOURCE_DIAGNOSTIC_MISMATCH", "canonical_coordinate_multiset_digest")


def parse_and_verify_ply_fd(fd: int, expected_source: Mapping[str, Any]) -> ParsedSource:
    """Independently parse an already-open regular PLY fd and verify its frozen row."""

    validate_independent_known_answer_vectors()
    validate_frozen_allocation_plan(load_frozen_v7())
    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
        _fail("PLY_FD_INVALID")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        _fail("PLY_FD_NOT_REGULAR")
    if not 0 < info.st_size <= MAX_RAW_BYTES or expected_source.get("raw_byte_count") != info.st_size:
        _fail("PLY_RAW_BYTE_BOUND_MISMATCH")
    header_bytes = expected_source.get("ply_header_byte_count")
    if isinstance(header_bytes, bool) or not isinstance(header_bytes, int):
        _fail("PLY_HEADER_BOUND_INVALID")

    original_offset = os.lseek(fd, 0, os.SEEK_CUR)
    duplicate = os.dup(fd)
    try:
        with os.fdopen(duplicate, "rb", closefd=True) as stream:
            stream.seek(0)
            raw_digest = hashlib.sha256()
            while True:
                chunk = stream.read(1_048_576)
                if not chunk:
                    break
                raw_digest.update(chunk)
            stream.seek(0)
            header, ply_format, elements = _read_header(stream, header_bytes)
            vertex = elements[0]
            points = _parse_vertices(stream, ply_format, vertex)
            first_digest = raw_digest.hexdigest()
            stream.seek(0)
            second_digest = hashlib.sha256()
            while True:
                chunk = stream.read(1_048_576)
                if not chunk:
                    break
                second_digest.update(chunk)
    finally:
        os.lseek(fd, original_offset, os.SEEK_SET)

    final_info = os.fstat(fd)
    identity_before = (
        info.st_dev, info.st_ino, info.st_size,
        getattr(info, "st_mtime_ns", None), getattr(info, "st_ctime_ns", None),
    )
    identity_after = (
        final_info.st_dev, final_info.st_ino, final_info.st_size,
        getattr(final_info, "st_mtime_ns", None), getattr(final_info, "st_ctime_ns", None),
    )
    if identity_before != identity_after or second_digest.hexdigest() != first_digest:
        _fail("PLY_SOURCE_CHANGED_DURING_PARSE")

    finite_count, duplicates, multiset_digest = _coordinate_diagnostics(points)
    final_info = os.fstat(fd)
    identity_before = (
        info.st_dev, info.st_ino, info.st_size,
        getattr(info, "st_mtime_ns", None), getattr(info, "st_ctime_ns", None),
    )
    identity_after = (
        final_info.st_dev, final_info.st_ino, final_info.st_size,
        getattr(final_info, "st_mtime_ns", None), getattr(final_info, "st_ctime_ns", None),
    )
    if identity_after != identity_before or _sha256_fd(fd, int(info.st_size)) != raw_digest.hexdigest():
        _fail("PLY_CHANGED_DURING_READ")
    ignored = sorted(
        f"{element['name']}:{prop[-1]}"
        for element in elements
        for prop in element["properties"]
        if not (element["name"] == "vertex" and prop[0] == "scalar" and prop[2] in {"x", "y", "z"})
    )
    diagnostics = SourceDiagnostics(
        raw_byte_count=info.st_size,
        raw_ply_sha256=first_digest,
        ply_header_byte_count=len(header),
        ply_header_sha256=hashlib.sha256(header).hexdigest(),
        ply_format=ply_format,
        parsed_vertex_count=len(points),
        finite_coordinate_count=finite_count,
        exact_duplicate_count=duplicates,
        ignored_property_name_count=len(ignored),
        ignored_property_name_digest=hashlib.sha256(_canonical_json(ignored)).hexdigest(),
        canonical_coordinate_multiset_sha256=multiset_digest,
    )
    _compare_source(expected_source, diagnostics)
    return ParsedSource(diagnostics, points)


def _sha(tag: bytes, *parts: bytes) -> bytes:
    return hashlib.sha256(tag + b"".join(parts)).digest()


def _field_id(case: bytes, arm: int, scale: int, morton: int) -> bytes:
    return _sha(b"CIIP_FIELD_ROW_V1\0", case, struct.pack("<BBQ", arm, scale, morton))[:16]


def _edge_id(case: bytes, lower: int, upper: int, kind: int, axis: int) -> bytes:
    return _sha(b"CIIP_EDGE_V1\0", case, struct.pack("<IIBB", lower, upper, kind, axis))[:16]


def _morton_xyz(value: int) -> tuple[int, int, int]:
    coordinates = [0, 0, 0]
    for bit in range(5):
        for axis in range(3):
            coordinates[axis] |= ((value >> (3 * bit + axis)) & 1) << bit
    return tuple(coordinates)  # type: ignore[return-value]


def _validate_resources(metadata: Mapping[str, Any], serialized_bytes: int) -> None:
    resource = metadata["resource"]
    limits = {
        "cpu_nanoseconds": CPU_NANOSECONDS_CAP,
        "wall_nanoseconds": WALL_NANOSECONDS_CAP,
        "peak_process_tree_rss_bytes": RSS_WATCHDOG_BYTES,
        "scratch_bytes": SCRATCH_BYTES_CAP,
        "serialized_bytes": SERIALIZED_BYTES_CAP,
    }
    for key, limit in limits.items():
        if resource[key] > limit:
            _fail("RESOURCE_CAP_EXCEEDED", key)
    if resource["peak_process_tree_rss_bytes"] >= ENTERPRISE_RSS_FORBIDDEN_BYTES:
        _fail("ENTERPRISE_RSS_THRESHOLD_REACHED")
    if resource["serialized_bytes"] != serialized_bytes:
        _fail("SERIALIZED_BYTE_ACCOUNTING_MISMATCH")
    if resource["watchdog_poll_milliseconds"] != WATCHDOG_POLL_MILLISECONDS:
        _fail("WATCHDOG_POLL_MISMATCH")
    if resource["resource_cap_tripped"] is not False:
        _fail("RESOURCE_CAP_MARKED_TRIPPED")


def validate_decoded_container(
    decoded: container.DecodedContainer,
    source: SourceDiagnostics,
    *,
    expected_code_sha256: str,
    expected_dependency_lock_sha256: str,
) -> ValidationSummary:
    """Independently validate decoded identities, relationships and V7 caps."""

    metadata, rows = decoded.metadata, decoded.rows
    if metadata["raw_source_sha256"] != source.raw_ply_sha256:
        _fail("CONTAINER_SOURCE_BINDING_MISMATCH")
    if metadata["code_sha256"] != expected_code_sha256:
        _fail("CONTAINER_CODE_BINDING_MISMATCH")
    if metadata["dependency_lock_sha256"] != expected_dependency_lock_sha256:
        _fail("CONTAINER_DEPENDENCY_BINDING_MISMATCH")
    census = metadata["census"]
    unique_records = source.finite_coordinate_count - source.exact_duplicate_count
    expected_census = {
        "raw_records": source.parsed_vertex_count,
        "finite_records": source.finite_coordinate_count,
        "unique_records": unique_records,
        "sampled_unique_records": min(unique_records, SAMPLE_CAP),
    }
    for key, expected in expected_census.items():
        if census[key] != expected:
            _fail("CONTAINER_SOURCE_CENSUS_MISMATCH", key)
    _validate_resources(metadata, container.METADATA_REGION_BYTES + sum(len(v) for v in decoded.table_bytes.values()))

    leaves = rows["leaf_aggregate"]
    fields = rows["cell_scale_field"]
    persistence = rows["persistence_disposition"]
    rejection = rows["rejection"]
    adjacency = rows["space_scale_adjacency"]
    o = len(leaves)
    mortons = tuple(row["morton_u64"] for row in leaves)
    expected_field_keys = tuple(
        (arm, scale, morton) for arm in (0, 1) for scale in range(3) for morton in mortons
    )
    if tuple((row["arm_u8"], row["scale_u8"], row["morton_u64"]) for row in fields) != expected_field_keys:
        _fail("FIELD_DOMAIN_NOT_COMPLETE")
    case = bytes.fromhex(metadata["case_token_digest_sha256"])
    for row in fields:
        if row["row_id_digest_128"] != _field_id(case, row["arm_u8"], row["scale_u8"], row["morton_u64"]):
            _fail("FIELD_IDENTITY_MISMATCH")

    arm_size = 3 * o
    preserved = (
        "support_membership_digest_128", "morton_u64", "trace_q_i64", "mass_q_i64",
        "status_u16", "support_count_u32", "raw_count_p10_u32", "raw_count_p90_u32",
        "balance_numerator_u32", "balance_denominator_u32", "growth12_gate_code_i32",
        "growth23_gate_code_i32", "stage_mask_u32", "first_rejection_u16", "flags_u16",
    )
    for baseline, isotropic in zip(fields[:arm_size], fields[arm_size:], strict=True):
        if any(baseline[key] != isotropic[key] for key in preserved):
            _fail("ISOTROPIC_PRESERVED_PAYLOAD_MISMATCH")
        if (
            baseline["first_rejection_u16"] == 0
            and baseline["lambda1_q_i64"] + baseline["lambda2_q_i64"] + baseline["lambda3_q_i64"] != baseline["trace_q_i64"]
        ):
            _fail("BASELINE_TRACE_EIGENVALUE_MISMATCH")
        if any(isotropic[key] != baseline["trace_q_i64"] for key in ("lambda1_q_i64", "lambda2_q_i64", "lambda3_q_i64")):
            _fail("ISOTROPIC_EIGENVALUE_NUMERATOR_MISMATCH")
        if isotropic["anisotropy_q_u16"] != 0 or isotropic["filtration_u16"] != 65_534:
            _fail("ISOTROPIC_ANISOTROPY_NOT_ZERO")

    if len(persistence) != len(fields) or len(rejection) != len(fields):
        _fail("COMPANION_DOMAIN_MISMATCH")
    for field_row, persistence_row, rejection_row in zip(fields, persistence, rejection, strict=True):
        if persistence_row["birth_row_id_digest_128"] != field_row["row_id_digest_128"]:
            _fail("PERSISTENCE_BIRTH_ALIAS_MISMATCH")
        if persistence_row["birth_f_u16"] != field_row["filtration_u16"]:
            _fail("PERSISTENCE_BIRTH_FILTRATION_MISMATCH")
        if (
            rejection_row["row_id_digest_128"] != field_row["row_id_digest_128"]
            or rejection_row["arm_u8"] != field_row["arm_u8"]
            or rejection_row["scale_u8"] != field_row["scale_u8"]
            or rejection_row["first_rejection_u16"] != field_row["first_rejection_u16"]
            or rejection_row["stage_mask_u32"] != field_row["stage_mask_u32"]
        ):
            _fail("REJECTION_ALIAS_MISMATCH")

    ordinal = {morton: index for index, morton in enumerate(mortons)}
    expected_edges: list[tuple[int, int, int, int]] = []
    for scale in range(3):
        for index, morton in enumerate(mortons):
            xyz = _morton_xyz(morton)
            for axis in range(3):
                neighbor = list(xyz)
                neighbor[axis] += 1
                if neighbor[axis] >= 32:
                    continue
                neighbor_morton = 0
                for bit in range(5):
                    for component in range(3):
                        neighbor_morton |= ((neighbor[component] >> bit) & 1) << (3 * bit + component)
                if neighbor_morton in ordinal:
                    expected_edges.append((scale * o + index, scale * o + ordinal[neighbor_morton], 0, axis))
    for scale in range(2):
        for index in range(o):
            expected_edges.append((scale * o + index, (scale + 1) * o + index, 1, 255))
    expected_edges.sort(key=lambda item: (item[2], item[3], item[0], item[1]))
    observed_edges = [
        (row["lower_vertex_u32"], row["upper_vertex_u32"], row["edge_kind_u8"], row["axis_u8"])
        for row in adjacency
    ]
    if observed_edges != expected_edges:
        _fail("ADJACENCY_DOMAIN_NOT_COMPLETE")
    for row in adjacency:
        if row["edge_id_digest_128"] != _edge_id(
            case, row["lower_vertex_u32"], row["upper_vertex_u32"], row["edge_kind_u8"], row["axis_u8"],
        ):
            _fail("ADJACENCY_IDENTITY_MISMATCH")

    valid_moments = sum(row["status_u16"] == 0 for row in fields)
    attributed = sum(row["arm_u8"] == 0 and row["reason_u16"] == 0 for row in persistence)
    isotropic_survivors = sum(row["arm_u8"] == 1 and row["reason_u16"] == 0 for row in persistence)
    if census["moment_valid_rows"] != valid_moments:
        _fail("MOMENT_VALID_CENSUS_MISMATCH")
    if census["attributed_intervals"] != attributed:
        _fail("ATTRIBUTED_INTERVAL_CENSUS_MISMATCH")
    if census["isotropic_survivors"] != isotropic_survivors or isotropic_survivors != 0:
        _fail("ISOTROPIC_SURVIVOR_CENSUS_MISMATCH")
    if attributed < 1:
        _fail("BASELINE_ATTRIBUTED_DENOMINATOR_ZERO")

    return ValidationSummary(
        raw_source_sha256=source.raw_ply_sha256,
        case_token_digest_sha256=metadata["case_token_digest_sha256"],
        occupied_cells=o,
        sampled_unique_records=census["sampled_unique_records"],
        serialized_bytes=metadata["resource"]["serialized_bytes"],
    )


def validate_container_bytes(
    data: bytes,
    source: SourceDiagnostics,
    *,
    expected_code_sha256: str,
    expected_dependency_lock_sha256: str,
) -> ValidationSummary:
    decoded = container.decode_container(
        data,
        expected_code_sha256=expected_code_sha256,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
    )
    return validate_decoded_container(
        decoded,
        source,
        expected_code_sha256=expected_code_sha256,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
    )


def rebuild_and_validate_container_bytes(
    data: bytes,
    source: ParsedSource,
    *,
    expected_code_sha256: str,
    expected_dependency_lock_sha256: str,
) -> ValidationSummary:
    """Independently rebuild all six tables, compare bytes, then validate framing."""

    decoded = container.decode_container(
        data,
        expected_code_sha256=expected_code_sha256,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
        materialize_rows=False,
    )
    rebuilt = rebuild_semantic_tables(source.points, source.diagnostics.raw_ply_sha256)
    expected_census = decoded.metadata["census"]
    if rebuilt.unique_records != expected_census["unique_records"]:
        _fail("INDEPENDENT_REBUILD_CENSUS_MISMATCH", "unique_records")
    if rebuilt.sampled_unique_records != expected_census["sampled_unique_records"]:
        _fail("INDEPENDENT_REBUILD_CENSUS_MISMATCH", "sampled_unique_records")
    if rebuilt.occupied_cells != expected_census["occupied_cells_O"]:
        _fail("INDEPENDENT_REBUILD_CENSUS_MISMATCH", "occupied_cells_O")
    for name in container.TABLE_NAMES:
        if rebuilt.table_bytes[name] != decoded.table_bytes[name]:
            _fail("INDEPENDENT_REBUILD_TABLE_BYTES_MISMATCH", name)
    metadata = decoded.metadata
    if metadata["raw_source_sha256"] != source.diagnostics.raw_ply_sha256:
        _fail("CONTAINER_SOURCE_BINDING_MISMATCH")
    if metadata["code_sha256"] != expected_code_sha256:
        _fail("CONTAINER_CODE_BINDING_MISMATCH")
    if metadata["dependency_lock_sha256"] != expected_dependency_lock_sha256:
        _fail("CONTAINER_DEPENDENCY_BINDING_MISMATCH")
    _validate_resources(metadata, len(data))
    return ValidationSummary(
        raw_source_sha256=source.diagnostics.raw_ply_sha256,
        case_token_digest_sha256=metadata["case_token_digest_sha256"],
        occupied_cells=rebuilt.occupied_cells,
        sampled_unique_records=rebuilt.sampled_unique_records,
        serialized_bytes=metadata["resource"]["serialized_bytes"],
    )


__all__ = [
    "IndependentValidationError",
    "ParsedSource",
    "SourceDiagnostics",
    "ValidationSummary",
    "parse_and_verify_ply_fd",
    "rebuild_and_validate_container_bytes",
    "validate_container_bytes",
    "validate_decoded_container",
]
