"""Bounded, fd-only primary PLY parser for the frozen CIIP-V1 V7 lock.

The parser consumes only PLY structure and XYZ coordinates.  It never receives
or derives a path and performs no sampling, partitioning, geometry, or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import stat
import struct
from typing import Final, Iterable

import numpy as np


RAW_BYTES_MAXIMUM: Final = 64_382_075
VERTEX_RECORDS_MAXIMUM: Final = 1_140_033
HEADER_BYTES_MAXIMUM: Final = 1_048_576
HEADER_LINES_MAXIMUM: Final = 100_000
ASCII_LINE_BYTES_MAXIMUM: Final = 1_048_576
ASCII_VERTEX_BATCH_ROWS: Final = 8_192
BINARY_VERTEX_BATCH_ROWS: Final = 65_536
MULTISET_DOMAIN: Final = b"PWTS_COORDINATE_MULTISET_V1\x00"

_SCALAR_DTYPES: Final[dict[str, str]] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_INTEGER_TYPES: Final = frozenset(
    {"char", "int8", "uchar", "uint8", "short", "int16", "ushort", "uint16", "int", "int32", "uint", "uint32"}
)


class PlyParseError(ValueError):
    """Terminal failure of the frozen primary PLY grammar."""


@dataclass(frozen=True, slots=True)
class ParserLimits:
    raw_bytes_maximum: int = RAW_BYTES_MAXIMUM
    vertex_records_maximum: int = VERTEX_RECORDS_MAXIMUM
    header_bytes_maximum: int = HEADER_BYTES_MAXIMUM
    header_lines_maximum: int = HEADER_LINES_MAXIMUM
    ascii_line_bytes_maximum: int = ASCII_LINE_BYTES_MAXIMUM
    list_items_per_record_maximum: int = 1_000_000
    total_list_items_maximum: int = 10_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PlyDiagnostics:
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

    @property
    def canonical_coordinate_multiset_digest(self) -> str:
        return self.canonical_coordinate_multiset_sha256


@dataclass(frozen=True, slots=True)
class ParsedPly:
    xyz: np.ndarray
    diagnostics: PlyDiagnostics
    ignored_properties: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Property:
    name: str
    kind: str
    scalar_type: str | None = None
    count_type: str | None = None
    item_type: str | None = None


@dataclass(frozen=True, slots=True)
class _Element:
    name: str
    count: int
    properties: tuple[_Property, ...]


@dataclass(frozen=True, slots=True)
class _Header:
    raw: bytes
    ply_format: str
    elements: tuple[_Element, ...]


class _PreadReader:
    """Buffered positional reader that never changes the caller's fd offset."""

    __slots__ = ("fd", "position", "end", "buffer", "eof")

    def __init__(self, fd: int, start: int, end: int) -> None:
        self.fd = fd
        self.position = start
        self.end = end
        self.buffer = bytearray()
        self.eof = start >= end

    def _fill(self, minimum: int = 1) -> None:
        while len(self.buffer) < minimum and not self.eof:
            requested = min(64 * 1024, self.end - self.position)
            if requested <= 0:
                self.eof = True
                break
            chunk = os.pread(self.fd, requested, self.position)
            if not chunk:
                self.eof = True
                break
            self.position += len(chunk)
            self.buffer.extend(chunk)
            if self.position >= self.end:
                self.eof = True

    def read_exact(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("read size must be nonnegative")
        self._fill(size)
        if len(self.buffer) < size:
            raise PlyParseError("truncated PLY payload")
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def readline(self, maximum: int) -> bytes:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > maximum:
                    raise PlyParseError("ASCII PLY line exceeds the fixed byte cap")
                result = bytes(self.buffer[: newline + 1])
                del self.buffer[: newline + 1]
                return result
            if len(self.buffer) > maximum:
                raise PlyParseError("ASCII PLY line exceeds the fixed byte cap")
            if self.eof:
                if not self.buffer:
                    return b""
                result = bytes(self.buffer)
                self.buffer.clear()
                return result
            self._fill(len(self.buffer) + 1)

    def read_line_block(self, line_count: int, maximum_line_bytes: int) -> bytes:
        block = bytearray()
        for _ in range(line_count):
            line = self.readline(maximum_line_bytes)
            if not line:
                raise PlyParseError("truncated ASCII PLY vertex data")
            block.extend(line)
        return bytes(block)

    def remaining(self) -> bytes:
        self._fill(self.end - self.position + len(self.buffer))
        result = bytes(self.buffer)
        self.buffer.clear()
        return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise PlyParseError("file became truncated during hashing")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _read_header(fd: int, raw_size: int, limits: ParserLimits) -> bytes:
    collected = bytearray()
    offset = 0
    scan_start = 0
    line_count = 0
    while offset < raw_size and len(collected) < limits.header_bytes_maximum:
        chunk = os.pread(
            fd,
            min(64 * 1024, raw_size - offset, limits.header_bytes_maximum - len(collected)),
            offset,
        )
        if not chunk:
            break
        collected.extend(chunk)
        offset += len(chunk)
        while True:
            newline = collected.find(b"\n", scan_start)
            if newline < 0:
                break
            line_count += 1
            if line_count > limits.header_lines_maximum:
                raise PlyParseError("PLY header line cap exceeded")
            line = bytes(collected[scan_start:newline]).rstrip(b"\r")
            if line == b"end_header":
                return bytes(collected[: newline + 1])
            scan_start = newline + 1
    if len(collected) >= limits.header_bytes_maximum:
        raise PlyParseError("PLY header byte cap exceeded")
    raise PlyParseError("PLY header ended before a newline-terminated end_header")


def _parse_nonnegative_decimal(token: str, *, name: str, maximum: int) -> int:
    if not token or not token.isdecimal():
        raise PlyParseError(f"{name} must be an unsigned decimal integer")
    value = int(token, 10)
    if value > maximum:
        raise PlyParseError(f"{name} exceeds its frozen bound")
    return value


def _parse_header(raw: bytes, limits: ParserLimits) -> _Header:
    try:
        raw_lines = raw.split(b"\n")
        if not raw_lines or raw_lines[-1] != b"":
            raise PlyParseError("PLY header must end on a line boundary")
        lines: list[str] = []
        for raw_line in raw_lines[:-1]:
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if b"\r" in raw_line:
                raise PlyParseError("PLY header contains a malformed carriage return")
            lines.append(raw_line.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise PlyParseError("PLY header must be ASCII") from exc
    if not lines or lines[0] != "ply":
        raise PlyParseError("not a PLY file")

    format_seen = False
    ply_format = ""
    element_rows: list[tuple[str, int, list[_Property]]] = []
    current: tuple[str, int, list[_Property]] | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if not fields:
            raise PlyParseError(f"blank PLY header line at {line_number}")
        directive = fields[0]
        if directive in {"comment", "obj_info"}:
            continue
        if directive == "format":
            if line_number != 2 or format_seen or len(fields) != 3:
                raise PlyParseError("PLY must declare exactly one well-formed format")
            if fields[2] != "1.0":
                raise PlyParseError("only PLY version 1.0 is accepted")
            if fields[1] not in {"ascii", "binary_little_endian"}:
                raise PlyParseError("V7 accepts only ASCII or little-endian binary PLY")
            format_seen = True
            ply_format = fields[1]
            continue
        if directive == "element":
            if not format_seen:
                raise PlyParseError("PLY format must precede element declarations")
            if len(fields) != 3:
                raise PlyParseError("malformed PLY element declaration")
            name = fields[1]
            if not name or any(existing[0] == name for existing in element_rows):
                raise PlyParseError("PLY element names must be nonempty and unique")
            count_limit = limits.vertex_records_maximum if name == "vertex" else (1 << 32) - 1
            count = _parse_nonnegative_decimal(fields[2], name=f"{name} count", maximum=count_limit)
            properties: list[_Property] = []
            current = (name, count, properties)
            element_rows.append(current)
            continue
        if directive == "property":
            if current is None:
                raise PlyParseError("PLY property declared before an element")
            properties = current[2]
            if len(fields) == 3:
                scalar_type, name = fields[1], fields[2]
                if scalar_type not in _SCALAR_DTYPES:
                    raise PlyParseError(f"unsupported PLY scalar type: {scalar_type}")
                prop = _Property(name=name, kind="scalar", scalar_type=scalar_type)
            elif len(fields) == 5 and fields[1] == "list":
                count_type, item_type, name = fields[2], fields[3], fields[4]
                if count_type not in _INTEGER_TYPES or item_type not in _SCALAR_DTYPES:
                    raise PlyParseError("unsupported or noninteger PLY list type")
                prop = _Property(name=name, kind="list", count_type=count_type, item_type=item_type)
            else:
                raise PlyParseError("malformed PLY property declaration")
            if not prop.name or any(existing.name == prop.name for existing in properties):
                raise PlyParseError("PLY property names must be nonempty and unique per element")
            properties.append(prop)
            continue
        if directive == "end_header":
            if line != "end_header" or line_number != len(lines):
                raise PlyParseError("end_header must be the final exact header line")
            continue
        raise PlyParseError(f"unsupported PLY header directive: {directive}")

    if not format_seen:
        raise PlyParseError("PLY format declaration is missing")
    elements = tuple(_Element(name, count, tuple(properties)) for name, count, properties in element_rows)
    vertices = [element for element in elements if element.name == "vertex"]
    if len(vertices) != 1 or not elements or elements[0].name != "vertex":
        raise PlyParseError("PLY must declare exactly one first-position vertex element")
    vertex = vertices[0]
    if any(element.count > 0 and not element.properties for element in elements):
        raise PlyParseError("nonempty PLY elements must declare at least one property")
    names = [prop.name for prop in vertex.properties]
    if any(names.count(axis) != 1 for axis in ("x", "y", "z")):
        raise PlyParseError("PLY vertex must have exactly one x, y, and z property")
    if any(prop.kind != "scalar" for prop in vertex.properties):
        raise PlyParseError("V7 vertex records must be fixed-width scalar rows")
    return _Header(raw=raw, ply_format=ply_format, elements=elements)


def _numpy_dtype(scalar_type: str, *, little_endian: bool = True) -> np.dtype:
    code = _SCALAR_DTYPES[scalar_type]
    byte_order = "<" if little_endian and code[-1] != "1" else "|"
    return np.dtype(byte_order + code)


def _binary_vertex_dtype(properties: tuple[_Property, ...]) -> np.dtype:
    return np.dtype(
        [(f"p{index}", _numpy_dtype(str(prop.scalar_type))) for index, prop in enumerate(properties)],
        align=False,
    )


def _assign_binary_vertices(
    reader: _PreadReader, vertex: _Element, xyz: np.ndarray
) -> None:
    dtype = _binary_vertex_dtype(vertex.properties)
    if dtype.itemsize <= 0:
        raise PlyParseError("binary vertex record width must be positive")
    coordinate_indices = tuple(
        next(index for index, prop in enumerate(vertex.properties) if prop.name == axis)
        for axis in ("x", "y", "z")
    )
    written = 0
    while written < vertex.count:
        rows = min(BINARY_VERTEX_BATCH_ROWS, vertex.count - written)
        encoded = reader.read_exact(rows * dtype.itemsize)
        records = np.frombuffer(encoded, dtype=dtype, count=rows)
        block = xyz[written : written + rows]
        for axis, property_index in enumerate(coordinate_indices):
            block[:, axis] = records[f"p{property_index}"]
        if not bool(np.isfinite(block).all()):
            raise PlyParseError("nonfinite PLY coordinate is terminal")
        block[block == 0.0] = 0.0
        written += rows


def _validate_ascii_scalar_columns(values: np.ndarray, properties: tuple[_Property, ...]) -> None:
    for column, prop in enumerate(properties):
        scalar_type = str(prop.scalar_type)
        if scalar_type not in _INTEGER_TYPES:
            continue
        dtype = _numpy_dtype(scalar_type)
        info = np.iinfo(dtype)
        data = values[:, column]
        if not bool((np.isfinite(data) & (data == np.trunc(data)) & (data >= info.min) & (data <= info.max)).all()):
            raise PlyParseError(f"invalid ASCII integer token for property {prop.name}")


def _ascii_integer_token_is_valid(encoded: bytes, start: int, end: int) -> bool:
    if start >= end:
        return False
    if encoded[start] in (0x2B, 0x2D):  # + or -
        start += 1
    return start < end and all(0x30 <= encoded[index] <= 0x39 for index in range(start, end))


def _validate_ascii_vertex_rows(
    encoded: bytes, expected_rows: int, properties: tuple[_Property, ...]
) -> None:
    """Validate line widths and integer syntax without per-point containers."""

    width = len(properties)
    row_count = 0
    token_count = 0
    in_token = False
    token_start = 0
    for index, value in enumerate(encoded):
        if value == 0x0A:  # newline
            if in_token:
                if (
                    str(properties[token_count].scalar_type) in _INTEGER_TYPES
                    and not _ascii_integer_token_is_valid(encoded, token_start, index)
                ):
                    raise PlyParseError(
                        f"invalid ASCII integer token for property {properties[token_count].name}"
                    )
                token_count += 1
            if token_count != width:
                raise PlyParseError("ASCII PLY vertex row has malformed, missing, or extra fields")
            row_count += 1
            token_count = 0
            in_token = False
        elif value in (0x09, 0x0B, 0x0C, 0x0D, 0x20):
            if in_token:
                if (
                    str(properties[token_count].scalar_type) in _INTEGER_TYPES
                    and not _ascii_integer_token_is_valid(encoded, token_start, index)
                ):
                    raise PlyParseError(
                        f"invalid ASCII integer token for property {properties[token_count].name}"
                    )
                token_count += 1
                in_token = False
        else:
            if not in_token:
                if token_count >= width:
                    raise PlyParseError(
                        "ASCII PLY vertex row has malformed, missing, or extra fields"
                    )
                token_start = index
            in_token = True
    if in_token or token_count:
        if in_token:
            if (
                str(properties[token_count].scalar_type) in _INTEGER_TYPES
                and not _ascii_integer_token_is_valid(encoded, token_start, len(encoded))
            ):
                raise PlyParseError(
                    f"invalid ASCII integer token for property {properties[token_count].name}"
                )
            token_count += 1
        if token_count != width:
            raise PlyParseError("ASCII PLY vertex row has malformed, missing, or extra fields")
        row_count += 1
    if row_count != expected_rows:
        raise PlyParseError("ASCII PLY vertex row count mismatch")


def _assign_ascii_vertices(
    reader: _PreadReader, vertex: _Element, xyz: np.ndarray, limits: ParserLimits
) -> None:
    width = len(vertex.properties)
    if width <= 0:
        raise PlyParseError("ASCII vertex record width must be positive")
    coordinate_indices = tuple(
        next(index for index, prop in enumerate(vertex.properties) if prop.name == axis)
        for axis in ("x", "y", "z")
    )
    written = 0
    while written < vertex.count:
        rows = min(ASCII_VERTEX_BATCH_ROWS, vertex.count - written)
        encoded = reader.read_line_block(rows, limits.ascii_line_bytes_maximum)
        _validate_ascii_vertex_rows(encoded, rows, vertex.properties)
        try:
            text = encoded.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PlyParseError("ASCII PLY payload must contain ASCII bytes") from exc
        with np.errstate(all="ignore"):
            values = np.fromstring(text, dtype=np.float64, sep=" ")
        expected = rows * width
        if values.size != expected:
            raise PlyParseError("ASCII PLY vertex row has malformed, missing, or extra fields")
        values = values.reshape(rows, width)
        _validate_ascii_scalar_columns(values, vertex.properties)
        block = xyz[written : written + rows]
        for axis, property_index in enumerate(coordinate_indices):
            scalar_type = str(vertex.properties[property_index].scalar_type)
            column = values[:, property_index]
            if _numpy_dtype(scalar_type).itemsize < 8:
                column = column.astype(_numpy_dtype(scalar_type), copy=False).astype(np.float64)
            block[:, axis] = column
        if not bool(np.isfinite(block).all()):
            raise PlyParseError("nonfinite PLY coordinate is terminal")
        block[block == 0.0] = 0.0
        written += rows


def _binary_scalar(reader: _PreadReader, scalar_type: str) -> int | float:
    dtype = _numpy_dtype(scalar_type)
    raw = reader.read_exact(dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype, count=1)[0].item()


def _consume_binary_tail(
    reader: _PreadReader, elements: Iterable[_Element], limits: ParserLimits
) -> None:
    total_list_items = 0
    for element in elements:
        fixed_width = all(prop.kind == "scalar" for prop in element.properties)
        if fixed_width:
            row_width = sum(_numpy_dtype(str(prop.scalar_type)).itemsize for prop in element.properties)
            reader.read_exact(element.count * row_width)
            continue
        for _ in range(element.count):
            for prop in element.properties:
                if prop.kind == "scalar":
                    reader.read_exact(_numpy_dtype(str(prop.scalar_type)).itemsize)
                    continue
                count = int(_binary_scalar(reader, str(prop.count_type)))
                if count < 0 or count > limits.list_items_per_record_maximum:
                    raise PlyParseError("binary PLY list length exceeds its fixed bound")
                total_list_items += count
                if total_list_items > limits.total_list_items_maximum:
                    raise PlyParseError("binary PLY total list items exceed the fixed bound")
                reader.read_exact(count * _numpy_dtype(str(prop.item_type)).itemsize)
    if reader.remaining():
        raise PlyParseError("binary PLY contains trailing undeclared bytes")


def _parse_ascii_token(token: bytes, scalar_type: str, *, property_name: str) -> int | float:
    try:
        text = token.decode("ascii")
        if scalar_type in _INTEGER_TYPES:
            if re.fullmatch(r"[+-]?[0-9]+", text) is None:
                raise ValueError
            value = int(text, 10)
            info = np.iinfo(_numpy_dtype(scalar_type))
            if value < info.min or value > info.max:
                raise ValueError
            return value
        return float(text)
    except (UnicodeDecodeError, ValueError, OverflowError) as exc:
        raise PlyParseError(f"invalid ASCII scalar for property {property_name}") from exc


def _consume_ascii_tail(
    reader: _PreadReader, elements: Iterable[_Element], limits: ParserLimits
) -> None:
    total_list_items = 0
    for element in elements:
        for _ in range(element.count):
            line = reader.readline(limits.ascii_line_bytes_maximum)
            if not line:
                raise PlyParseError("truncated ASCII PLY element data")
            fields = line.split()
            offset = 0
            for prop in element.properties:
                if prop.kind == "scalar":
                    if offset >= len(fields):
                        raise PlyParseError("short ASCII PLY element row")
                    _parse_ascii_token(fields[offset], str(prop.scalar_type), property_name=prop.name)
                    offset += 1
                    continue
                if offset >= len(fields):
                    raise PlyParseError("short ASCII PLY list")
                count_value = _parse_ascii_token(
                    fields[offset], str(prop.count_type), property_name=prop.name
                )
                count = int(count_value)
                if count < 0 or count > limits.list_items_per_record_maximum:
                    raise PlyParseError("ASCII PLY list length exceeds its fixed bound")
                total_list_items += count
                if total_list_items > limits.total_list_items_maximum:
                    raise PlyParseError("ASCII PLY total list items exceed the fixed bound")
                if offset + 1 + count > len(fields):
                    raise PlyParseError("truncated ASCII PLY list")
                for item in fields[offset + 1 : offset + 1 + count]:
                    _parse_ascii_token(item, str(prop.item_type), property_name=prop.name)
                offset += 1 + count
            if offset != len(fields):
                raise PlyParseError("ASCII PLY element row contains extra fields")
    if reader.remaining().strip():
        raise PlyParseError("ASCII PLY contains trailing undeclared fields")


def _coordinate_diagnostics(xyz: np.ndarray) -> tuple[int, str]:
    packed = xyz.view(np.dtype("V24")).reshape(-1)
    ordered = np.empty(packed.shape, dtype=np.dtype("V24"))
    np.copyto(ordered, packed)
    ordered.sort(kind="stable")
    if ordered.size:
        unique_count = 1 + int(np.count_nonzero(ordered[1:] != ordered[:-1]))
    else:
        unique_count = 0
    digest = hashlib.sha256(MULTISET_DOMAIN)
    digest.update(memoryview(ordered).cast("B"))
    return int(ordered.size) - unique_count, digest.hexdigest()


def parse_ply_fd(fd: int, *, limits: ParserLimits = ParserLimits()) -> ParsedPly:
    """Parse one already-open regular-file descriptor without changing its offset."""

    if isinstance(fd, bool) or not isinstance(fd, int):
        raise TypeError("fd must be an already-open integer file descriptor")
    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise PlyParseError("fd is not an open file descriptor") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PlyParseError("fd must refer to a bounded regular file")
    raw_size = int(metadata.st_size)
    if raw_size <= 0 or raw_size > limits.raw_bytes_maximum:
        raise PlyParseError("raw PLY byte count is outside the frozen bound")

    raw_sha256 = _sha256_fd(fd, raw_size)
    header_raw = _read_header(fd, raw_size, limits)
    header = _parse_header(header_raw, limits)
    vertex = header.elements[0]
    try:
        xyz = np.empty((vertex.count, 3), dtype=np.dtype("<f8"), order="C")
    except (MemoryError, ValueError) as exc:
        raise PlyParseError("unable to allocate the bounded XYZ array") from exc

    reader = _PreadReader(fd, len(header_raw), raw_size)
    if header.ply_format == "binary_little_endian":
        _assign_binary_vertices(reader, vertex, xyz)
        _consume_binary_tail(reader, header.elements[1:], limits)
    else:
        _assign_ascii_vertices(reader, vertex, xyz, limits)
        _consume_ascii_tail(reader, header.elements[1:], limits)

    duplicate_count, multiset_sha256 = _coordinate_diagnostics(xyz)
    ignored = tuple(
        sorted(
            f"{element.name}:{prop.name}"
            for element in header.elements
            for prop in element.properties
            if not (element.name == "vertex" and prop.name in {"x", "y", "z"})
        )
    )
    ignored_digest = hashlib.sha256(_canonical_json_bytes(list(ignored))).hexdigest()
    try:
        final_metadata = os.fstat(fd)
    except OSError as exc:
        raise PlyParseError("source descriptor became unavailable after parsing") from exc
    identity_before = (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        getattr(metadata, "st_mtime_ns", None), getattr(metadata, "st_ctime_ns", None),
    )
    identity_after = (
        final_metadata.st_dev, final_metadata.st_ino, final_metadata.st_size,
        getattr(final_metadata, "st_mtime_ns", None), getattr(final_metadata, "st_ctime_ns", None),
    )
    if identity_after != identity_before or _sha256_fd(fd, raw_size) != raw_sha256:
        raise PlyParseError("source changed between hash and completed parse")
    xyz.flags.writeable = False
    diagnostics = PlyDiagnostics(
        raw_byte_count=raw_size,
        raw_ply_sha256=raw_sha256,
        ply_header_byte_count=len(header_raw),
        ply_header_sha256=hashlib.sha256(header_raw).hexdigest(),
        ply_format=header.ply_format,
        parsed_vertex_count=vertex.count,
        finite_coordinate_count=vertex.count,
        exact_duplicate_count=duplicate_count,
        ignored_property_name_count=len(ignored),
        ignored_property_name_digest=ignored_digest,
        canonical_coordinate_multiset_sha256=multiset_sha256,
    )
    return ParsedPly(xyz=xyz, diagnostics=diagnostics, ignored_properties=ignored)


__all__ = [
    "ParsedPly",
    "ParserLimits",
    "PlyDiagnostics",
    "PlyParseError",
    "parse_ply_fd",
]
