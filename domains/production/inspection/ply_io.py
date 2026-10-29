"""Minimal, strict PLY reader/writer for the production viewer assets.

Formats handled (ENGINE_IO §2 and §4.1):

- **Point clouds** (``registered_scan.ply``): ``element vertex N`` with float
  ``x y z`` (other scalar vertex properties are read past), no faces.
- **Display meshes** (``display/*.ply``, E1 ``stages/S<k>.ply``):
  ``element vertex`` + ``element face`` with a ``list <count> <index>
  vertex_indices`` (or ``vertex_index``) property. Polygons with more than
  three corners are fan-triangulated.

Readers accept ``ascii 1.0``, ``binary_little_endian 1.0`` and
``binary_big_endian 1.0``; extra elements are skipped (binary skips need the
element's properties to be parseable). Every coordinate must be finite and
every face index in range, otherwise :class:`PlyFormatError` is raised.

Writers always emit ``binary_little_endian 1.0`` with float32 ``x y z`` (and
``list uchar int vertex_indices`` faces) and a fixed header, so identical
inputs give identical bytes. Files are written to a temporary name and
renamed into place.

numpy is imported lazily (inside functions).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = [
    "PlyFormatError",
    "PlyData",
    "read_ply",
    "write_points_ply",
    "write_mesh_ply",
    "atomic_write_bytes",
]

_SCALAR = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_FORMATS = {"ascii": None, "binary_little_endian": "<", "binary_big_endian": ">"}
#: Header is ASCII and short; anything bigger is not a PLY we produce or accept.
_MAX_HEADER_BYTES = 64 * 1024


class PlyFormatError(ValueError):
    """The file is not a PLY this module accepts (malformed, truncated, non-finite...)."""


@dataclass(frozen=True)
class _Property:
    name: str
    dtype: str | None  # scalar dtype code, or None for a list
    count_dtype: str | None = None
    item_dtype: str | None = None


@dataclass(frozen=True)
class _Element:
    name: str
    count: int
    properties: tuple[_Property, ...]


@dataclass(frozen=True)
class PlyData:
    """Parsed PLY: ``vertices`` float64 (N, 3); ``faces`` int64 (M, 3) or ``None``."""

    vertices: "np.ndarray"
    faces: "np.ndarray | None"
    format: str


def _parse_header(raw: bytes) -> tuple[str, list[_Element], int]:
    end = raw.find(b"end_header", 0, _MAX_HEADER_BYTES)
    if not raw.startswith(b"ply") or end < 0:
        raise PlyFormatError("not a PLY file (missing 'ply' magic or 'end_header')")
    nl = raw.find(b"\n", end)
    if nl < 0:
        raise PlyFormatError("header not terminated by a newline")
    try:
        text = raw[:end].decode("ascii")
    except UnicodeDecodeError:
        raise PlyFormatError("header is not ASCII") from None
    fmt: str | None = None
    elements: list[_Element] = []
    current: list[_Property] | None = None
    name = ""
    count = 0
    for line in text.splitlines()[1:]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        if parts[0] == "format":
            if len(parts) != 3 or parts[1] not in _FORMATS or parts[2] != "1.0":
                raise PlyFormatError(f"unsupported format line {line!r}")
            fmt = parts[1]
        elif parts[0] == "element":
            if current is not None:
                elements.append(_Element(name, count, tuple(current)))
            if len(parts) != 3 or not parts[2].isdigit():
                raise PlyFormatError(f"bad element line {line!r}")
            name, count, current = parts[1], int(parts[2]), []
        elif parts[0] == "property":
            if current is None:
                raise PlyFormatError("property before any element")
            if len(parts) == 5 and parts[1] == "list":
                if parts[2] not in _SCALAR or parts[3] not in _SCALAR:
                    raise PlyFormatError(f"bad list property {line!r}")
                current.append(_Property(parts[4], None, _SCALAR[parts[2]], _SCALAR[parts[3]]))
            elif len(parts) == 3 and parts[1] in _SCALAR:
                current.append(_Property(parts[2], _SCALAR[parts[1]]))
            else:
                raise PlyFormatError(f"bad property line {line!r}")
        else:
            raise PlyFormatError(f"unexpected header line {line!r}")
    if current is not None:
        elements.append(_Element(name, count, tuple(current)))
    if fmt is None:
        raise PlyFormatError("missing format line")
    return fmt, elements, nl + 1


def _scalar_struct(el: _Element, endian: str) -> "np.dtype":
    import numpy as np

    return np.dtype([(f"p{i}_{p.name}", endian + p.dtype) for i, p in enumerate(el.properties)])  # type: ignore[operator]


def _read_binary_list_element(buf: memoryview, offset: int, el: _Element, endian: str) -> tuple[list[Any], int]:
    """Generic (slow) path for elements with list properties."""
    import numpy as np

    rows: list[Any] = []
    for _ in range(el.count):
        row: list[Any] = []
        for p in el.properties:
            if p.dtype is not None:
                dt = np.dtype(endian + p.dtype)
                if offset + dt.itemsize > len(buf):
                    raise PlyFormatError(f"truncated element {el.name!r}")
                row.append(np.frombuffer(buf, dt, 1, offset)[0])
                offset += dt.itemsize
            else:
                cdt = np.dtype(endian + p.count_dtype)  # type: ignore[operator]
                idt = np.dtype(endian + p.item_dtype)  # type: ignore[operator]
                if offset + cdt.itemsize > len(buf):
                    raise PlyFormatError(f"truncated element {el.name!r}")
                n = int(np.frombuffer(buf, cdt, 1, offset)[0])
                offset += cdt.itemsize
                if offset + n * idt.itemsize > len(buf):
                    raise PlyFormatError(f"truncated element {el.name!r}")
                row.append(np.frombuffer(buf, idt, n, offset).astype(np.int64))
                offset += n * idt.itemsize
        rows.append(row)
    return rows, offset


def _face_list_index(el: _Element) -> int:
    for i, p in enumerate(el.properties):
        if p.dtype is None and p.name in ("vertex_indices", "vertex_index"):
            return i
    raise PlyFormatError("face element has no vertex_indices list")


def _triangulate(polys: list["np.ndarray"]) -> "np.ndarray":
    import numpy as np

    tris: list[tuple[int, int, int]] = []
    for poly in polys:
        if len(poly) < 3:
            raise PlyFormatError("face with fewer than 3 vertices")
        for k in range(1, len(poly) - 1):
            tris.append((int(poly[0]), int(poly[k]), int(poly[k + 1])))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def read_ply(path: str | os.PathLike[str], *, require_faces: bool = False) -> PlyData:
    """Read a PLY point cloud or triangle mesh (module docstring)."""
    import numpy as np

    raw = Path(path).read_bytes()
    fmt, elements, offset = _parse_header(raw)
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    if fmt == "ascii":
        tokens = raw[offset:].split()
        pos = 0
        for el in elements:
            values: list[list[float]] = []
            polys: list[np.ndarray] = []
            for _ in range(el.count):
                row: list[float] = []
                for p in el.properties:
                    if p.dtype is not None:
                        if pos >= len(tokens):
                            raise PlyFormatError(f"truncated element {el.name!r}")
                        row.append(float(tokens[pos]))
                        pos += 1
                    else:
                        if pos >= len(tokens):
                            raise PlyFormatError(f"truncated element {el.name!r}")
                        n = int(tokens[pos])
                        pos += 1
                        if pos + n > len(tokens):
                            raise PlyFormatError(f"truncated element {el.name!r}")
                        polys.append(np.asarray([int(t) for t in tokens[pos:pos + n]], dtype=np.int64))
                        pos += n
                values.append(row)
            if el.name == "vertex":
                names = [p.name for p in el.properties if p.dtype is not None]
                try:
                    cols = [names.index(c) for c in ("x", "y", "z")]
                except ValueError:
                    raise PlyFormatError("vertex element lacks x/y/z") from None
                arr = np.asarray(values, dtype=np.float64).reshape(el.count, len(names))
                vertices = arr[:, cols]
            elif el.name == "face":
                _face_list_index(el)
                faces = _triangulate(polys)
    else:
        endian = _FORMATS[fmt]
        buf = memoryview(raw)
        for el in elements:
            if all(p.dtype is not None for p in el.properties):
                dt = _scalar_struct(el, endian)  # type: ignore[arg-type]
                size = dt.itemsize * el.count
                if offset + size > len(raw):
                    raise PlyFormatError(f"truncated element {el.name!r}")
                arr = np.frombuffer(buf, dt, el.count, offset)
                offset += size
                if el.name == "vertex":
                    names = {p.name: f"p{i}_{p.name}" for i, p in enumerate(el.properties)}
                    if not {"x", "y", "z"} <= names.keys():
                        raise PlyFormatError("vertex element lacks x/y/z")
                    vertices = np.stack([arr[names[c]].astype(np.float64) for c in ("x", "y", "z")], axis=1)
                continue
            if el.name == "face":
                li = _face_list_index(el)
                p = el.properties[li]
                if len(el.properties) == 1:
                    # Fast path: every face a triangle with a fixed-size record.
                    tri_dt = np.dtype([("n", endian + p.count_dtype), ("i", endian + p.item_dtype, (3,))])  # type: ignore[operator]
                    size = tri_dt.itemsize * el.count
                    if offset + size <= len(raw):
                        arr = np.frombuffer(buf, tri_dt, el.count, offset)
                        if el.count == 0 or bool(np.all(arr["n"] == 3)):
                            faces = arr["i"].astype(np.int64)
                            offset += size
                            continue
                rows, offset = _read_binary_list_element(buf, offset, el, endian)  # type: ignore[arg-type]
                faces = _triangulate([r[li] for r in rows])
            else:
                _, offset = _read_binary_list_element(buf, offset, el, endian)  # type: ignore[arg-type]
    if vertices is None:
        raise PlyFormatError("no vertex element")
    if not np.all(np.isfinite(vertices)):
        raise PlyFormatError("non-finite vertex coordinates")
    if faces is not None and faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise PlyFormatError("face index out of range")
    if require_faces and (faces is None or len(faces) == 0):
        raise PlyFormatError("mesh has no faces")
    return PlyData(vertices=vertices, faces=faces, format=fmt)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary sibling and ``os.replace``."""
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _header(n_vertices: int, n_faces: int | None) -> bytes:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment frame cad_model_mm",
        f"element vertex {n_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if n_faces is not None:
        lines += [f"element face {n_faces}", "property list uchar int vertex_indices"]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_points_ply(path: str | os.PathLike[str], points: "np.ndarray") -> None:
    """Binary LE float32 xyz point cloud, no faces."""
    import numpy as np

    pts = np.ascontiguousarray(np.asarray(points, dtype="<f4").reshape(-1, 3))
    atomic_write_bytes(path, _header(len(pts), None) + pts.tobytes())


def write_mesh_ply(path: str | os.PathLike[str], vertices: "np.ndarray", faces: "np.ndarray") -> None:
    """Binary LE float32 vertices + ``list uchar int`` triangle faces."""
    import numpy as np

    v = np.ascontiguousarray(np.asarray(vertices, dtype="<f4").reshape(-1, 3))
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    rec = np.empty(len(f), dtype=np.dtype([("n", "u1"), ("i", "<i4", (3,))]))
    rec["n"] = 3
    rec["i"] = f.astype("<i4")
    atomic_write_bytes(path, _header(len(v), len(f)) + v.tobytes() + rec.tobytes())
