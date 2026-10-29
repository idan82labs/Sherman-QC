"""Scan ingest (E3 step 1): PLY (points or mesh) and STL, read by content.

The runner stages the scan under an opaque name (``input/scan.ply|.stl``), and
the engine never looks at the name (plan §10 amendment 4): the format is
decided from the bytes, exactly as the C02 probe does.

- **PLY**: ``ascii``, ``binary_little_endian`` or ``binary_big_endian`` 1.0.
  The vertex element must carry ``x``, ``y`` and ``z`` (any numeric type);
  other vertex properties are ignored. Faces are counted, not used (the
  engine works on the vertex cloud; the mesh connectivity of a structured-light
  export adds nothing the metrology uses).
- **STL**: binary (``84 + 50 n`` bytes) or ASCII (``facet ... vertex``). STL is
  a triangle soup, so exact duplicate vertices are merged; the reported
  ``vertex_count`` is ``3 n`` and ``face_count`` is ``n`` (the C02 probe's
  convention for :class:`contracts.production.inspection_result.ScanInfoV1`).

Coordinates are returned as float64 in the file's own units; the explicit
unit factor (1, 25.4 or 1000) is applied by the caller. Non-finite vertices
are dropped and counted. Nothing here rescales anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ScanData", "ScanReadError", "read_scan"]

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
    "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}
_PLY_FORMATS = {"ascii": "PLY_ASCII", "binary_little_endian": "PLY_BINARY_LE", "binary_big_endian": "PLY_BINARY_BE"}
_MAX_HEADER = 1 << 16
_ASCII_STL_VERTEX = re.compile(
    rb"vertex\s+([-+0-9.eEinfINFaA]+)\s+([-+0-9.eEinfINFaA]+)\s+([-+0-9.eEinfINFaA]+)"
)


class ScanReadError(ValueError):
    """The scan bytes are not a readable PLY/STL (reported as UNSUPPORTED_INPUT)."""


@dataclass
class ScanData:
    #: (N, 3) float64 vertex coordinates in the file's units (unique for STL).
    points: np.ndarray
    #: ScanInfoV1.format
    format: str
    #: Vertex count as stored in the file (3 n for STL).
    vertex_count: int
    face_count: int
    #: Non-finite vertices dropped while reading.
    dropped_nonfinite: int = 0


def read_scan(path: str | Path) -> ScanData:
    """Read ``path`` (format decided from its content). Raises :class:`ScanReadError`."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ScanReadError(f"cannot read scan: {exc.strerror or exc}") from None
    if not raw:
        raise ScanReadError("scan file is empty")
    if raw.startswith(b"ply\n") or raw.startswith(b"ply\r\n"):
        return _read_ply(raw)
    stl = _read_stl(raw)
    if stl is not None:
        return stl
    raise ScanReadError("neither PLY nor STL content")


# --------------------------------------------------------------------------- PLY
def _read_ply(raw: bytes) -> ScanData:
    end = raw.find(b"end_header", 0, _MAX_HEADER)
    if end < 0:
        raise ScanReadError("PLY header has no end_header")
    eol = raw.find(b"\n", end)
    if eol < 0:
        raise ScanReadError("PLY end_header line is not terminated")
    body_offset = eol + 1
    try:
        header = raw[:end].decode("ascii")
    except UnicodeDecodeError:
        raise ScanReadError("PLY header is not ASCII") from None
    fmt = None
    elements: list[tuple[str, int, list[tuple]]] = []
    for line in header.replace("\r", "").split("\n")[1:]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        if parts[0] == "format":
            if len(parts) != 3 or parts[1] not in _PLY_FORMATS:
                raise ScanReadError(f"unsupported PLY format line {line.strip()!r}")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3 or not parts[2].isdigit():
                raise ScanReadError(f"malformed PLY element line {line.strip()!r}")
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise ScanReadError("PLY property before any element")
            if len(parts) == 5 and parts[1] == "list":
                if parts[2] not in _PLY_TYPES or parts[3] not in _PLY_TYPES:
                    raise ScanReadError(f"unknown PLY list types in {line.strip()!r}")
                elements[-1][2].append(("list", parts[4], parts[2], parts[3]))
            elif len(parts) == 3 and parts[1] in _PLY_TYPES:
                elements[-1][2].append(("scalar", parts[2], parts[1]))
            else:
                raise ScanReadError(f"malformed PLY property line {line.strip()!r}")
        else:
            raise ScanReadError(f"unknown PLY header keyword {parts[0]!r}")
    if fmt is None:
        raise ScanReadError("PLY header has no format line")
    vertex = next((e for e in elements if e[0] == "vertex"), None)
    if vertex is None:
        raise ScanReadError("PLY has no vertex element")
    names = [p[1] for p in vertex[2] if p[0] == "scalar"]
    if any(axis not in names for axis in ("x", "y", "z")):
        raise ScanReadError("PLY vertex element lacks x/y/z")
    if any(p[0] == "list" for p in vertex[2]):
        raise ScanReadError("PLY vertex element with list properties is not supported")
    face = next((e for e in elements if e[0] == "face"), None)
    face_count = face[1] if face is not None else 0
    if fmt == "ascii":
        points = _ply_ascii_vertices(raw[body_offset:], elements, vertex)
    else:
        points = _ply_binary_vertices(raw, body_offset, elements, vertex, "<" if fmt == "binary_little_endian" else ">")
    return _finish(points, _PLY_FORMATS[fmt], vertex[1], face_count)


def _ply_binary_vertices(raw: bytes, offset: int, elements, vertex, endian: str) -> np.ndarray:
    for name, count, props in elements:
        if name == "vertex":
            dt = np.dtype([(p[1], endian + _PLY_TYPES[p[2]]) for p in props])
            need = dt.itemsize * count
            if offset + need > len(raw):
                raise ScanReadError("PLY is truncated inside the vertex element")
            arr = np.frombuffer(raw, dtype=dt, count=count, offset=offset)
            return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        offset = _skip_binary_element(raw, offset, count, props, endian)
    raise ScanReadError("PLY has no vertex element")  # pragma: no cover - checked by the caller


def _skip_binary_element(raw: bytes, offset: int, count: int, props, endian: str) -> int:
    """Byte offset after an element that precedes the vertices (rare; slow path for lists)."""
    if all(p[0] == "scalar" for p in props):
        return offset + np.dtype([(p[1], endian + _PLY_TYPES[p[2]]) for p in props]).itemsize * count
    for _ in range(count):
        for p in props:
            if p[0] == "scalar":
                offset += np.dtype(_PLY_TYPES[p[2]]).itemsize
            else:
                ct = np.dtype(endian + _PLY_TYPES[p[2]])
                if offset + ct.itemsize > len(raw):
                    raise ScanReadError("PLY is truncated")
                n = int(np.frombuffer(raw, dtype=ct, count=1, offset=offset)[0])
                offset += ct.itemsize + n * np.dtype(_PLY_TYPES[p[3]]).itemsize
    return offset


def _ply_ascii_vertices(body: bytes, elements, vertex) -> np.ndarray:
    lines = body.split(b"\n")
    cursor = 0
    for name, count, props in elements:
        if name == "vertex":
            block = [ln for ln in lines[cursor:cursor + count]]
            if len(block) < count:
                raise ScanReadError("ASCII PLY is truncated inside the vertex element")
            names = [p[1] for p in props]
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            try:
                rows = [ln.split() for ln in block]
                arr = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows], dtype=np.float64)
            except (ValueError, IndexError):
                raise ScanReadError("ASCII PLY vertex record is malformed") from None
            return arr.reshape(-1, 3)
        cursor += count
    raise ScanReadError("PLY has no vertex element")  # pragma: no cover


# --------------------------------------------------------------------------- STL
def _read_stl(raw: bytes) -> ScanData | None:
    if len(raw) >= 84:
        n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
        if 84 + 50 * n == len(raw):
            if n == 0:
                raise ScanReadError("binary STL has zero facets")
            rec = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(raw, dtype=rec, count=n, offset=84)
            verts = arr["v"].reshape(-1, 3).astype(np.float64)
            return _finish(_unique_rows(verts), "STL_BINARY", 3 * n, n)
    head = raw[:512].lstrip().lower()
    if head.startswith(b"solid") and b"facet" in raw[:65536].lower():
        found = _ASCII_STL_VERTEX.findall(raw)
        if not found:
            raise ScanReadError("ASCII STL has no vertices")
        try:
            verts = np.array([[float(a), float(b), float(c)] for a, b, c in found], dtype=np.float64)
        except ValueError:
            raise ScanReadError("ASCII STL vertex is malformed") from None
        facets = len(verts) // 3
        return _finish(_unique_rows(verts), "STL_ASCII", 3 * facets, facets)
    return None


def _unique_rows(verts: np.ndarray) -> np.ndarray:
    """Exact-duplicate vertices merged (deterministic, sorted order)."""
    if len(verts) == 0:
        return verts
    return np.unique(verts, axis=0)


def _finish(points: np.ndarray, fmt: str, vertex_count: int, face_count: int) -> ScanData:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        pts = pts[finite]
    return ScanData(points=pts, format=fmt, vertex_count=int(vertex_count), face_count=int(face_count),
                    dropped_nonfinite=dropped)
