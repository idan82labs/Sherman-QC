from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from cad_import import import_cad_file


CAD_BREP_EXTS = {".step", ".stp", ".iges", ".igs"}


def load_reference_geometry(path: str | Path, deflection: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if p.suffix.lower() in CAD_BREP_EXTS:
        mesh = import_cad_file(str(p), linear_deflection=deflection)
    else:
        mesh = o3d.io.read_triangle_mesh(str(p))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32) if mesh.has_triangles() else np.zeros((0, 3), dtype=np.int32)
    return vertices, triangles


def load_point_positions(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == '.pcd' or p.suffix.lower() == '.ply':
        pcd = o3d.io.read_point_cloud(str(p))
        if pcd.has_points():
            return np.asarray(pcd.points, dtype=np.float64)
    mesh = o3d.io.read_triangle_mesh(str(p))
    return np.asarray(mesh.vertices, dtype=np.float64) if mesh.has_vertices() else np.zeros((0, 3), dtype=np.float64)
