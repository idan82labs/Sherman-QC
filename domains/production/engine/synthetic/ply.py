"""Minimal binary PLY I/O for synthetic scans (no third-party dependency).

Written files are ``binary_little_endian 1.0`` with ``element vertex N``
(``property float x/y/z``) and, for meshes, ``element face M`` with
``property list uchar int vertex_indices`` -- the shape ZEISS exports and the
contract display meshes use. The header carries ``comment`` lines (generator,
units); no timestamps, so bytes are deterministic. :func:`read_ply` reads the
same subset (binary little-endian or ASCII, float/double vertices, optional
triangle faces) and ignores any extra vertex properties.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

__all__ = ["write_ply", "read_ply"]


def write_ply(
    path: str | os.PathLike,
    points: np.ndarray,
    faces: np.ndarray | None = None,
    comments: tuple[str, ...] = (),
) -> str:
    """Write ``points`` (N,3) as float32 and optional triangle ``faces`` (M,3);
    returns the sha256 of the bytes written (tmp + rename)."""
    pts = np.ascontiguousarray(np.asarray(points, dtype="<f4"))
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("points must be a finite (N, 3) array")
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments:
        if "\n" in c or "\r" in c:
            raise ValueError("PLY comments must be single lines")
        lines.append(f"comment {c}")
    lines += [f"element vertex {len(pts)}", "property float x", "property float y", "property float z"]
    body = [pts.tobytes()]
    if faces is not None:
        f = np.asarray(faces, dtype=np.int64)
        if f.ndim != 2 or f.shape[1] != 3 or (len(f) and (f.min() < 0 or f.max() >= len(pts))):
            raise ValueError("faces must be (M, 3) indices into points")
        lines += [f"element face {len(f)}", "property list uchar int vertex_indices"]
        rec = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        rec["n"] = 3
        rec["i"] = f.astype("<i4")
        body.append(rec.tobytes())
    lines.append("end_header")
    data = ("\n".join(lines) + "\n").encode("ascii") + b"".join(body)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return hashlib.sha256(data).hexdigest()


_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
    "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
}


def read_ply(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Read vertices (N,3 float64), triangle faces (M,3 int64 or None) and comments."""
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        raise ValueError("not a PLY file")
    header_end = raw.index(b"\n", end) + 1
    header = raw[:header_end].decode("ascii").splitlines()
    fmt = None
    comments: list[str] = []
    elements: list[tuple[str, int, list[tuple]]] = []
    for line in header[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "comment":
            comments.append(line[len("comment "):])
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1][2].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1][2].append(("scalar", parts[1], parts[2]))
    if fmt not in ("binary_little_endian", "ascii"):
        raise ValueError(f"unsupported PLY format {fmt}")
    verts = None
    faces = None
    if fmt == "ascii":
        rows = raw[header_end:].decode("ascii").split("\n")
        cursor = 0
        for name, count, props in elements:
            block = rows[cursor:cursor + count]
            cursor += count
            if name == "vertex":
                names = [p[2] for p in props if p[0] == "scalar"]
                arr = np.array([[float(v) for v in r.split()[: len(names)]] for r in block])
                verts = arr[:, [names.index("x"), names.index("y"), names.index("z")]]
            elif name == "face":
                faces = np.array([[int(v) for v in r.split()[1:4]] for r in block], dtype=np.int64)
        return np.asarray(verts, dtype=np.float64), faces, comments
    offset = header_end
    for name, count, props in elements:
        if all(p[0] == "scalar" for p in props):
            dt = np.dtype([(p[2], _TYPES[p[1]]) for p in props])
            arr = np.frombuffer(raw, dtype=dt, count=count, offset=offset)
            offset += dt.itemsize * count
            if name == "vertex":
                verts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        else:
            if len(props) != 1 or props[0][0] != "list":
                raise ValueError("only single-list face elements are supported")
            _, ctype, itype, _ = props[0]
            cdt, idt = np.dtype(_TYPES[ctype]), np.dtype(_TYPES[itype])
            rec = np.dtype([("n", cdt), ("i", idt, (3,))])
            arr = np.frombuffer(raw, dtype=rec, count=count, offset=offset)
            if count and not np.all(arr["n"] == 3):
                raise ValueError("only triangle faces are supported")
            offset += rec.itemsize * count
            if name == "face":
                faces = arr["i"].astype(np.int64)
    if verts is None:
        raise ValueError("PLY has no vertex element")
    return verts, faces, comments
