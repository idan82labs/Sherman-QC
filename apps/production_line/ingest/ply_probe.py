"""Cheap structural probe of a scan file: PLY (ascii / binary LE / binary BE,
mesh or points) and STL (binary / ascii).

The probe decides whether a file landed completely and is a format the engine
accepts. It never reads a whole file into Python memory:

- PLY: the header (at most 64 KiB) is parsed for elements and properties.
  - Binary: the minimum body size is computed from the element strides (face
    lists count at least a triangle). A file smaller than header + minimum
    body is ``PLY_TRUNCATED``.
  - ASCII: body lines are counted in 1 MiB chunks. Fewer lines than
    elements is ``PLY_TRUNCATED``. The final record (the last line of the
    last non-empty element) must also carry exactly its token count: the
    number of scalar properties, plus for each list ``1 + n`` tokens. A file
    cut inside its final line is therefore ``PLY_TRUNCATED`` too, unless the
    cut falls inside the final token (undetectable without a checksum).
- Binary STL: the size must equal ``84 + 50 n`` exactly (a binary STL whose
  80-byte header happens to start with ``solid`` is still binary when the size
  matches).
- ASCII STL: starts with ``solid``; ``endfacet`` occurrences are counted in
  chunks, and the file must end with ``endsolid`` (else truncated,
  ``STL_INVALID``).
- Zero vertices or facets is ``EMPTY_SCAN``. A PLY magic with a malformed
  header is ``PLY_HEADER_INVALID``; bytes that are neither PLY nor STL are
  ``UNSUPPORTED_FORMAT``; an OS error is ``READ_ERROR``.

The format is decided by content, not by extension; the extension only selects
the error vocabulary for unrecognised bytes (``.stl`` -> ``STL_INVALID``).

The optional bounding box uses a numpy memmap of the vertex coordinates
(binary PLY with float/double x, y, z, and binary STL), only when the vertex
count is at most 20 M. numpy is imported lazily, inside the bbox helper only.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ProbeResult",
    "probe",
    "PLY_TYPE_SIZES",
    "MAX_HEADER_BYTES",
    "BBOX_MAX_VERTICES",
]

MAX_HEADER_BYTES = 64 * 1024
BBOX_MAX_VERTICES = 20_000_000
_CHUNK = 1024 * 1024

#: PLY scalar type name -> (byte size, numpy kind code)
PLY_TYPE_SIZES: dict[str, tuple[int, str]] = {
    "char": (1, "i1"),
    "int8": (1, "i1"),
    "uchar": (1, "u1"),
    "uint8": (1, "u1"),
    "short": (2, "i2"),
    "int16": (2, "i2"),
    "ushort": (2, "u2"),
    "uint16": (2, "u2"),
    "int": (4, "i4"),
    "int32": (4, "i4"),
    "uint": (4, "u4"),
    "uint32": (4, "u4"),
    "float": (4, "f4"),
    "float32": (4, "f4"),
    "double": (8, "f8"),
    "float64": (8, "f8"),
}
_PLY_FORMATS = {"ascii": "PLY_ASCII", "binary_little_endian": "PLY_BINARY_LE", "binary_big_endian": "PLY_BINARY_BE"}
_FACE_LIST_NAMES = frozenset({"vertex_indices", "vertex_index"})


@dataclass(frozen=True)
class ProbeResult:
    """``ok`` is True only for a complete, non-empty, supported scan.

    ``fmt`` is one of PLY_ASCII, PLY_BINARY_LE, PLY_BINARY_BE, STL_BINARY,
    STL_ASCII (``None`` when unrecognised). ``reason`` is a quarantine reason
    when not ok; ``detail`` is a short human explanation.
    """

    ok: bool
    fmt: str | None
    size_bytes: int
    vertex_count: int | None = None
    face_count: int | None = None
    header_bytes: int | None = None
    expected_min_bytes: int | None = None
    bbox_mm: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    reason: str | None = None
    detail: str | None = None

    @property
    def extension(self) -> str | None:
        """Archive extension implied by the detected format."""
        if self.fmt is None:
            return None
        return ".ply" if self.fmt.startswith("PLY") else ".stl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "format": self.fmt,
            "size_bytes": self.size_bytes,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "header_bytes": self.header_bytes,
            "expected_min_bytes": self.expected_min_bytes,
            "bbox_mm": None if self.bbox_mm is None else [list(self.bbox_mm[0]), list(self.bbox_mm[1])],
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class _Prop:
    name: str
    type: str | None  # scalar type, or None for lists
    count_type: str | None = None
    item_type: str | None = None


@dataclass(frozen=True)
class _Element:
    name: str
    count: int
    props: tuple[_Prop, ...]

    @property
    def fixed_stride(self) -> int | None:
        sizes = [PLY_TYPE_SIZES[p.type][0] for p in self.props if p.type is not None]
        if len(sizes) != len(self.props):
            return None
        return sum(sizes)

    def min_record_bytes(self) -> int:
        total = 0
        for p in self.props:
            if p.type is not None:
                total += PLY_TYPE_SIZES[p.type][0]
            else:
                items = 3 if (self.name == "face" and p.name in _FACE_LIST_NAMES) else 0
                total += PLY_TYPE_SIZES[p.count_type][0] + items * PLY_TYPE_SIZES[p.item_type][0]  # type: ignore[index]
        return total


class _HeaderError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _fail(size: int, reason: str, detail: str, fmt: str | None = None, **kw: Any) -> ProbeResult:
    return ProbeResult(ok=False, fmt=fmt, size_bytes=size, reason=reason, detail=detail, **kw)


def probe(path: str | os.PathLike[str], *, compute_bbox: bool = True) -> ProbeResult:
    """Probe ``path`` (see module docstring). Never raises for file content."""
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            head = fh.read(MAX_HEADER_BYTES)
    except OSError as exc:
        return _fail(0, "READ_ERROR", f"cannot read file: {exc.strerror or exc}")
    if size == 0:
        return _fail(0, "EMPTY_SCAN", "file is empty")
    if head.startswith(b"ply\n") or head.startswith(b"ply\r\n"):
        return _probe_ply(p, size, head, compute_bbox)
    stl = _probe_stl(p, size, head, compute_bbox)
    if stl is not None:
        return stl
    if p.suffix.lower() == ".stl":
        return _fail(size, "STL_INVALID", "not a binary STL (size != 84 + 50n) and not an ASCII STL")
    if head[:3].lower() == b"ply":
        return _fail(size, "PLY_HEADER_INVALID", "PLY magic not followed by a newline")
    return _fail(size, "UNSUPPORTED_FORMAT", "neither PLY nor STL content")


# --------------------------------------------------------------------------- PLY
def _parse_ply_header(head: bytes) -> tuple[str, list[_Element], int]:
    marker = head.find(b"end_header")
    if marker < 0:
        raise _HeaderError("no end_header within the first 64 KiB")
    eol = head.find(b"\n", marker)
    if eol < 0:
        raise _HeaderError("end_header line is not terminated")
    if head[marker:eol].rstrip(b"\r") != b"end_header":
        raise _HeaderError("malformed end_header line")
    header_bytes = eol + 1
    try:
        text = head[:marker].decode("ascii")
    except UnicodeDecodeError:
        raise _HeaderError("header is not ASCII") from None
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    if not lines or lines[0] != "ply":
        raise _HeaderError("missing ply magic")
    fmt: str | None = None
    elements: list[_Element] = []
    current: tuple[str, int, list[_Prop]] | None = None
    for ln in lines[1:]:
        if not ln or ln.startswith("comment") or ln.startswith("obj_info"):
            continue
        parts = ln.split()
        key = parts[0]
        if key == "format":
            if len(parts) != 3 or parts[1] not in _PLY_FORMATS or parts[2] != "1.0":
                raise _HeaderError(f"unsupported format line {ln!r}")
            if fmt is not None:
                raise _HeaderError("duplicate format line")
            fmt = _PLY_FORMATS[parts[1]]
        elif key == "element":
            if len(parts) != 3 or not parts[2].isdigit():
                raise _HeaderError(f"malformed element line {ln!r}")
            if current is not None:
                elements.append(_Element(current[0], current[1], tuple(current[2])))
            current = (parts[1], int(parts[2]), [])
        elif key == "property":
            if current is None:
                raise _HeaderError("property before any element")
            if len(parts) == 5 and parts[1] == "list":
                if parts[2] not in PLY_TYPE_SIZES or parts[3] not in PLY_TYPE_SIZES:
                    raise _HeaderError(f"unknown list types in {ln!r}")
                if PLY_TYPE_SIZES[parts[2]][1].startswith("f"):
                    raise _HeaderError("list count type must be an integer type")
                current[2].append(_Prop(parts[4], None, parts[2], parts[3]))
            elif len(parts) == 3 and parts[1] in PLY_TYPE_SIZES:
                current[2].append(_Prop(parts[2], parts[1]))
            else:
                raise _HeaderError(f"malformed property line {ln!r}")
        else:
            raise _HeaderError(f"unknown header keyword {key!r}")
    if current is not None:
        elements.append(_Element(current[0], current[1], tuple(current[2])))
    if fmt is None:
        raise _HeaderError("missing format line")
    names = [e.name for e in elements]
    if len(set(names)) != len(names):
        raise _HeaderError("duplicate element names")
    vertex = next((e for e in elements if e.name == "vertex"), None)
    if vertex is None:
        raise _HeaderError("no vertex element")
    prop_names = [pr.name for pr in vertex.props]
    for axis in ("x", "y", "z"):
        if axis not in prop_names:
            raise _HeaderError(f"vertex element has no '{axis}' property")
    return fmt, elements, header_bytes


_MAX_RECORD_BYTES = 64 * 1024


def _count_ascii_lines(path: Path, offset: int, needed: int) -> tuple[int, bytes]:
    """Count non-blank body lines after ``offset``, stopping once ``needed`` is reached.

    Returns ``(count, last)`` where ``last`` is the text of the last counted
    line (capped at 64 KiB), so the caller can validate the final record.
    """
    count = 0
    current = bytearray()
    last = b""
    with open(path, "rb") as fh:
        fh.seek(offset)
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            start = 0
            while True:
                nl = chunk.find(b"\n", start)
                piece = chunk[start:] if nl < 0 else chunk[start:nl]
                if len(current) < _MAX_RECORD_BYTES:
                    current += piece[: _MAX_RECORD_BYTES - len(current)]
                if nl < 0:
                    break
                if current.strip():
                    count += 1
                    last = bytes(current)
                    if count >= needed:
                        return count, last
                current.clear()
                start = nl + 1
    if current.strip():
        count += 1  # last line without a trailing newline
        last = bytes(current)
    return count, last


def _record_token_problem(element: _Element, line: bytes) -> str | None:
    """None when ``line`` has exactly the tokens one ``element`` record needs."""
    tokens = line.split()
    i = 0
    for prop in element.props:
        if prop.type is not None:
            i += 1
            continue
        if i >= len(tokens):
            return f"list '{prop.name}' count missing"
        try:
            n = int(tokens[i])
        except ValueError:
            return f"list '{prop.name}' count {tokens[i][:20]!r} is not an integer"
        if n < 0:
            return f"list '{prop.name}' count is negative"
        i += 1 + n
    if i != len(tokens):
        return f"{len(tokens)} tokens where the record needs {i}"
    return None


def _probe_ply(path: Path, size: int, head: bytes, compute_bbox: bool) -> ProbeResult:
    try:
        fmt, elements, header_bytes = _parse_ply_header(head)
    except _HeaderError as exc:
        return _fail(size, "PLY_HEADER_INVALID", exc.detail)
    vertex = next(e for e in elements if e.name == "vertex")
    face = next((e for e in elements if e.name == "face"), None)
    vcount = vertex.count
    fcount = face.count if face is not None else 0
    common: dict[str, Any] = {"vertex_count": vcount, "face_count": fcount, "header_bytes": header_bytes}
    if vcount == 0:
        return _fail(size, "EMPTY_SCAN", "PLY has zero vertices", fmt=fmt, **common)
    if fmt == "PLY_ASCII":
        needed = sum(e.count for e in elements)
        try:
            lines, last = _count_ascii_lines(path, header_bytes, needed)
        except OSError as exc:
            return _fail(size, "READ_ERROR", f"cannot read body: {exc.strerror or exc}", fmt=fmt, **common)
        if lines < needed:
            return _fail(size, "PLY_TRUNCATED", f"ASCII body has {lines} of {needed} element lines", fmt=fmt, **common)
        final_element = [e for e in elements if e.count > 0][-1]
        problem = _record_token_problem(final_element, last)
        if problem is not None:
            return _fail(
                size, "PLY_TRUNCATED", f"final {final_element.name} record is incomplete: {problem}", fmt=fmt, **common
            )
        return ProbeResult(ok=True, fmt=fmt, size_bytes=size, **common)
    body_min = sum(e.count * e.min_record_bytes() for e in elements)
    expected_min = header_bytes + body_min
    if size < expected_min:
        return _fail(
            size,
            "PLY_TRUNCATED",
            f"binary PLY is {size} bytes, needs at least {expected_min}",
            fmt=fmt,
            expected_min_bytes=expected_min,
            **common,
        )
    bbox = None
    if compute_bbox and vcount <= BBOX_MAX_VERTICES:
        bbox = _ply_bbox(path, fmt, elements, header_bytes)
    return ProbeResult(ok=True, fmt=fmt, size_bytes=size, expected_min_bytes=expected_min, bbox_mm=bbox, **common)


def _finite_bbox(mins: list[float], maxs: list[float]) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not all(math.isfinite(v) for v in mins + maxs):
        return None
    return (tuple(mins), tuple(maxs))  # type: ignore[return-value]


def _ply_bbox(
    path: Path, fmt: str, elements: list[_Element], header_bytes: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    offset = header_bytes
    for e in elements:
        if e.name == "vertex":
            break
        stride = e.fixed_stride
        if stride is None:
            return None  # a variable-size element precedes the vertices
        offset += e.count * stride
    vertex = next(e for e in elements if e.name == "vertex")
    if vertex.fixed_stride is None:
        return None
    if any(PLY_TYPE_SIZES[p.type][1] not in ("f4", "f8") for p in vertex.props if p.name in ("x", "y", "z")):  # type: ignore[index]
        return None
    endian = "<" if fmt == "PLY_BINARY_LE" else ">"
    fields = [(p.name, endian + PLY_TYPE_SIZES[p.type][1]) for p in vertex.props]  # type: ignore[index]
    try:
        return _memmap_bbox(path, fields, offset, vertex.count, ("x", "y", "z"))
    except (OSError, ValueError):
        return None


def _memmap_bbox(
    path: Path, fields: list[tuple[str, Any]], offset: int, count: int, axes: tuple[str, ...]
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    import numpy as np  # lazy: only the bbox needs it

    dtype = np.dtype(fields)
    arr = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    try:
        mins = [math.inf, math.inf, math.inf]
        maxs = [-math.inf, -math.inf, -math.inf]
        step = 1_000_000
        for start in range(0, count, step):
            chunk = arr[start:start + step]
            for i, axis in enumerate(axes):
                values = np.asarray(chunk[axis], dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size:
                    mins[i] = min(mins[i], float(finite.min()))
                    maxs[i] = max(maxs[i], float(finite.max()))
        return _finite_bbox(mins, maxs)
    finally:
        mm = getattr(arr, "_mmap", None)
        del arr
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):  # pragma: no cover - views still alive
                pass


# --------------------------------------------------------------------------- STL
def _probe_stl(path: Path, size: int, head: bytes, compute_bbox: bool) -> ProbeResult | None:
    """Return a result when the bytes are STL (valid or not), else None."""
    if size >= 84:
        (n,) = struct.unpack_from("<I", head, 80)
        if 84 + 50 * n == size:
            if n == 0:
                return _fail(size, "EMPTY_SCAN", "binary STL has zero facets", fmt="STL_BINARY", vertex_count=0, face_count=0)
            bbox = None
            if compute_bbox and 3 * n <= BBOX_MAX_VERTICES:
                try:
                    bbox = _stl_bbox(path, n)
                except (OSError, ValueError):
                    bbox = None
            return ProbeResult(
                ok=True,
                fmt="STL_BINARY",
                size_bytes=size,
                vertex_count=3 * n,
                face_count=n,
                header_bytes=84,
                expected_min_bytes=size,
                bbox_mm=bbox,
            )
    if head.lstrip()[:5].lower() == b"solid":
        return _probe_ascii_stl(path, size)
    return None


def _stl_bbox(path: Path, n: int) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    import numpy as np  # lazy

    dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    arr = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(n,))
    try:
        mins = [math.inf, math.inf, math.inf]
        maxs = [-math.inf, -math.inf, -math.inf]
        step = 500_000
        for start in range(0, n, step):
            verts = np.asarray(arr[start:start + step]["v"], dtype=np.float64).reshape(-1, 3)
            for i in range(3):
                col = verts[:, i]
                finite = col[np.isfinite(col)]
                if finite.size:
                    mins[i] = min(mins[i], float(finite.min()))
                    maxs[i] = max(maxs[i], float(finite.max()))
        return _finite_bbox(mins, maxs)
    finally:
        mm = getattr(arr, "_mmap", None)
        del arr
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):  # pragma: no cover
                pass


def _probe_ascii_stl(path: Path, size: int) -> ProbeResult:
    token = b"endfacet"
    facets = 0
    tail = b""
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                data = (tail + chunk).lower()
                facets += data.count(token)
                # keep a tail shorter than the token so no match is counted twice
                tail = data[-(len(token) - 1):]
            fh.seek(max(0, size - 4096))
            last = fh.read().lower()
    except OSError as exc:
        return _fail(size, "READ_ERROR", f"cannot read ASCII STL: {exc.strerror or exc}", fmt="STL_ASCII")
    stripped = last.rstrip()
    final_line = stripped.rsplit(b"\n", 1)[-1].strip()
    if not final_line.startswith(b"endsolid"):
        return _fail(size, "STL_INVALID", "ASCII STL does not end with endsolid (truncated)", fmt="STL_ASCII")
    if facets == 0:
        return _fail(size, "EMPTY_SCAN", "ASCII STL has zero facets", fmt="STL_ASCII", vertex_count=0, face_count=0)
    return ProbeResult(ok=True, fmt="STL_ASCII", size_bytes=size, vertex_count=3 * facets, face_count=facets)
