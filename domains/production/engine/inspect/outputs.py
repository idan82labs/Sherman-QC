"""E3 output files (ENGINE_IO §4.1), written tmp + rename.

- ``registered_scan.ply``: binary little-endian PLY, ``element vertex N``,
  float32 ``x y z``, no faces: every ingested point after ``unit_factor`` and
  ``raw_to_frame``.
- ``deviations.f32``: N float32 little-endian, NaN where no value.
- ``segment_labels.u16``: N uint16 little-endian (0 unassigned, ``1 + i`` panel
  ``i``, ``1001 + j`` bend ``j``, 65535 unplanned crease).
- ``display/stage_mesh.ply`` / ``display/final_mesh.ply``: byte copies of the
  compiled display meshes.
- ``bend_geometry.json`` and ``result.json``: canonical JSON (``result.json``
  last).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from contracts.production.canonical import canonical_json_bytes

__all__ = ["write_points_ply", "write_array", "write_json", "copy_file", "atomic_write"]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_points_ply(path: Path, points: np.ndarray, comment: str = "sherman_production_engine registered scan") -> None:
    pts = np.ascontiguousarray(np.asarray(points, dtype="<f4"))
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"comment {comment}", "comment frame cad_model_mm units mm",
        f"element vertex {len(pts)}", "property float x", "property float y", "property float z", "end_header",
    ]) + "\n"
    atomic_write(path, header.encode("ascii") + pts.tobytes())


def write_array(path: Path, arr: np.ndarray, dtype: str) -> None:
    atomic_write(path, np.ascontiguousarray(np.asarray(arr).astype(dtype)).tobytes())


def write_json(path: Path, obj) -> None:
    atomic_write(path, canonical_json_bytes(obj))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
