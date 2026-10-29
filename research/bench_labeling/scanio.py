"""Scan loading in raw source coordinates (no transform is ever applied)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .custody import check_path


@dataclass
class Scan:
    path: Path
    sha256: str
    points: np.ndarray            # (N, 3) float64, source_ply_raw
    normals: np.ndarray | None    # (N, 3) unit, oriented (mesh winding / file normals)
    oriented: bool                # True when normals carry a trustworthy outward sign
    kind: str                     # "mesh" | "point_cloud"
    n_faces: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ply_has_faces_and_normals(path: Path) -> tuple[bool, bool]:
    with path.open("rb") as handle:
        header = handle.read(64 * 1024)
    end = header.find(b"end_header")
    text = header[: max(end, 0)].decode("ascii", errors="replace")
    has_faces = False
    has_normals = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "face"] and parts[2] != "0":
            has_faces = True
        if len(parts) >= 3 and parts[0] == "property" and parts[-1] == "nx":
            has_normals = True
    return has_faces, has_normals


def load_scan(path: str | Path) -> Scan:
    import open3d as o3d

    path = check_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in {".ply", ".stl", ".obj"}:
        raise ValueError(f"unsupported scan format {suffix!r}; expected .ply, .stl or .obj")
    sha = sha256_file(path)
    if suffix in {".stl", ".obj"}:
        has_faces, file_normals = True, False
    else:
        has_faces, file_normals = _ply_has_faces_and_normals(path)
    if has_faces:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if suffix in {".stl", ".obj"}:
            mesh.remove_duplicated_vertices()
        points = np.asarray(mesh.vertices, dtype=np.float64).copy()
        n_faces = int(len(mesh.triangles))
        if n_faces == 0:
            # Faceless OBJ/PLY: treat as an unoriented point cloud.
            normals, kind, oriented = None, "point_cloud", False
        else:
            if not (file_normals and mesh.has_vertex_normals()):
                mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
            kind = "mesh"
            oriented = True
    else:
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64).copy()
        n_faces = 0
        kind = "point_cloud"
        if file_normals and cloud.has_normals():
            normals = np.asarray(cloud.normals, dtype=np.float64).copy()
            oriented = True
        else:
            normals = None
            oriented = False
    if len(points) == 0:
        raise ValueError(f"scan has no points: {path.name}")
    finite = np.isfinite(points).all(axis=1)
    if normals is not None:
        lengths = np.linalg.norm(normals, axis=1)
        good = np.isfinite(lengths) & (lengths > 1e-9)
        normals = np.where(good[:, None], normals / np.where(good, lengths, 1.0)[:, None], 0.0)
    if not finite.all():
        points = points[finite]
        normals = normals[finite] if normals is not None else None
    return Scan(path=path, sha256=sha, points=points, normals=normals,
                oriented=oriented, kind=kind, n_faces=n_faces)
