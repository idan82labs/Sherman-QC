"""Minimal PLY / raw-array I/O for evaluation (numpy only, no third-party PLY library).

Reads the shapes this harness meets:

- E3 ``registered_scan.ply``: binary little-endian, float32 ``x y z``, no faces.
- E1 / E3 display meshes: float32 vertices plus ``list uchar int`` triangles.
- E2 synthetic scans and real ZEISS/Geomagic exports: binary little/big-endian or
  ASCII, float or double vertices with optional extra vertex properties
  (normals, colour, quality), optional triangle faces.

:func:`read_ply` returns vertices (float64), optional unit normals and optional
triangles; :func:`read_ply_vertices` stops after the vertex element (cheap for
large meshes). :func:`write_points_ply` writes the E3 registered-scan format.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PlyData",
    "PlyHeader",
    "read_ply_header",
    "read_ply",
    "read_ply_vertices",
    "write_points_ply",
    "write_mesh_ply",
    "read_f32",
    "read_u16",
    "sha256_file",
]

_SCALAR = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
    "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


@dataclass(frozen=True)
class PlyHeader:
    fmt: str  # ascii | binary_little_endian | binary_big_endian
    header_bytes: int
    #: (name, count, [(kind, type..., name)])
    elements: tuple[tuple[str, int, tuple[tuple[str, ...], ...]], ...]
    comments: tuple[str, ...]

    def count(self, name: str) -> int:
        for el, n, _ in self.elements:
            if el == name:
                return n
        return 0


@dataclass(frozen=True)
class PlyData:
    vertices: np.ndarray  # (N, 3) float64
    normals: np.ndarray | None  # (N, 3) float64 unit, or None
    faces: np.ndarray | None  # (M, 3) int64, or None
    comments: tuple[str, ...]


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_ply_header(path: str | os.PathLike) -> PlyHeader:
    with open(path, "rb") as fh:
        head = fh.read(1 << 16)
    if not head.startswith(b"ply"):
        raise ValueError(f"{path}: not a PLY file")
    end = head.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: PLY header too long or truncated")
    nl = head.index(b"\n", end) + 1
    lines = head[:nl].decode("ascii", errors="strict").splitlines()
    fmt = None
    comments: list[str] = []
    elements: list[list] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] in ("comment", "obj_info"):
            comments.append(line.split(" ", 1)[1] if " " in line else "")
        elif parts[0] == "element":
            elements.append([parts[1], int(parts[2]), []])
        elif parts[0] == "property":
            if not elements:
                raise ValueError(f"{path}: property before element")
            if parts[1] == "list":
                elements[-1][2].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1][2].append(("scalar", parts[1], parts[2]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
    return PlyHeader(fmt, nl, tuple((e[0], e[1], tuple(e[2])) for e in elements), tuple(comments))


def _endian(fmt: str) -> str:
    return ">" if fmt == "binary_big_endian" else "<"


def _vertex_arrays(rec: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    verts = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    normals = None
    if {"nx", "ny", "nz"} <= set(names):
        n = np.column_stack([rec["nx"], rec["ny"], rec["nz"]]).astype(np.float64)
        norm = np.linalg.norm(n, axis=1)
        good = norm > 1e-12
        n[good] /= norm[good, None]
        normals = n
    return verts, normals


def read_ply(path: str | os.PathLike, *, with_faces: bool = True) -> PlyData:
    """Vertices, optional normals and optional triangles (see module docstring)."""
    header = read_ply_header(path)
    raw = Path(path).read_bytes()
    if header.fmt == "ascii":
        return _read_ascii(raw, header, with_faces)
    e = _endian(header.fmt)
    offset = header.header_bytes
    verts = normals = faces = None
    for name, count, props in header.elements:
        if all(p[0] == "scalar" for p in props):
            dt = np.dtype([(p[2], e + _SCALAR[p[1]]) for p in props])
            need = dt.itemsize * count
            if offset + need > len(raw):
                raise ValueError(f"{path}: truncated element {name!r}")
            rec = np.frombuffer(raw, dtype=dt, count=count, offset=offset)
            offset += need
            if name == "vertex":
                verts, normals = _vertex_arrays(rec, [p[2] for p in props])
            continue
        if not with_faces and verts is not None:
            break  # nothing after the vertex element is needed
        if name != "face" or len(props) != 1 or props[0][0] != "list":
            raise ValueError(f"{path}: unsupported list element {name!r}")
        _, ctype, itype, _ = props[0]
        cdt = np.dtype(e + _SCALAR[ctype])
        idt = np.dtype(e + _SCALAR[itype])
        rec_dt = np.dtype([("n", cdt), ("i", idt, (3,))])
        need = rec_dt.itemsize * count
        if offset + need > len(raw):
            raise ValueError(f"{path}: truncated face element")
        rec = np.frombuffer(raw, dtype=rec_dt, count=count, offset=offset)
        if count and not np.all(rec["n"] == 3):
            raise ValueError(f"{path}: only triangle faces are supported")
        faces = rec["i"].astype(np.int64)
        offset += need
    if verts is None:
        raise ValueError(f"{path}: PLY has no vertex element")
    return PlyData(verts, normals, faces, header.comments)


def _read_ascii(raw: bytes, header: PlyHeader, with_faces: bool) -> PlyData:
    rows = raw[header.header_bytes:].decode("ascii").split("\n")
    cursor = 0
    verts = normals = faces = None
    for name, count, props in header.elements:
        block = rows[cursor:cursor + count]
        cursor += count
        if name == "vertex":
            names = [p[2] for p in props if p[0] == "scalar"]
            arr = np.array([[float(v) for v in r.split()[: len(names)]] for r in block], dtype=np.float64)
            rec = {n: arr[:, i] for i, n in enumerate(names)}
            verts = np.column_stack([rec["x"], rec["y"], rec["z"]])
            if {"nx", "ny", "nz"} <= set(names):
                n = np.column_stack([rec["nx"], rec["ny"], rec["nz"]])
                norm = np.linalg.norm(n, axis=1)
                good = norm > 1e-12
                n[good] /= norm[good, None]
                normals = n
        elif name == "face" and with_faces:
            faces = np.array([[int(v) for v in r.split()[1:4]] for r in block], dtype=np.int64)
    if verts is None:
        raise ValueError("PLY has no vertex element")
    return PlyData(verts, normals, faces, header.comments)


def read_ply_vertices(path: str | os.PathLike) -> PlyData:
    """Vertices (and normals) only; faces are not read."""
    return read_ply(path, with_faces=False)


def write_points_ply(path: str | os.PathLike, points: np.ndarray, comments: tuple[str, ...] = ()) -> str:
    """Binary LE float32 ``x y z`` point PLY (the E3 ``registered_scan.ply`` shape).
    Written tmp + rename; returns the sha256 of the bytes."""
    return write_mesh_ply(path, points, None, comments)


def write_mesh_ply(path: str | os.PathLike, points: np.ndarray, faces: np.ndarray | None,
                   comments: tuple[str, ...] = ()) -> str:
    pts = np.ascontiguousarray(np.asarray(points, dtype="<f4"))
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("points must be a finite (N, 3) array")
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments:
        if "\n" in c or "\r" in c:
            raise ValueError("PLY comments are single lines")
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


def read_f32(path: str | os.PathLike, count: int | None = None) -> np.ndarray:
    """float32 little-endian array (E3 ``deviations.f32``); checks ``count`` when given."""
    arr = np.fromfile(path, dtype="<f4")
    if count is not None and arr.size != count:
        raise ValueError(f"{path}: expected {count} float32 values, found {arr.size}")
    return arr


def read_u16(path: str | os.PathLike, count: int | None = None) -> np.ndarray:
    """uint16 little-endian array (E3 ``segment_labels.u16``); checks ``count`` when given."""
    arr = np.fromfile(path, dtype="<u2")
    if count is not None and arr.size != count:
        raise ValueError(f"{path}: expected {count} uint16 values, found {arr.size}")
    return arr
