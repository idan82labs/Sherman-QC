"""Decimated, index-aligned viewer assets in ``result/`` (ENGINE_IO §5 step 2).

Inputs are the files the engine result names in ``viewer_assets`` (relative
to the engine directory). Outputs (``layout.RESULT_FILES``), all in
``cad_model_mm``:

- ``viewer_scan.ply``: binary LE float32 xyz, the decimated registered scan.
- ``viewer_deviations.f32``: float32 LE, NaN = no value. It is built from the
  **same index selection** as ``viewer_scan.ply``, so point *i* of the PLY
  and value *i* of the deviations always belong together.
- ``stage_mesh.ply`` / ``final_mesh.ply``: the display meshes, re-encoded
  (binary LE float32, ``list uchar int`` triangles).
- ``bend_geometry.json``: ``BendGeometryFileV1.from_result(result)`` as
  canonical JSON (bend polylines + label/side anchors + unplanned creases),
  so the viewer geometry always equals the result rows.

Decimation (deterministic): if the scan has more than ``cap_points`` points,
a voxel grid anchored at the frame origin starts at 0.5 mm and grows by 1.5x
until the number of occupied voxels is at most the cap. Each voxel keeps the
one point nearest to the voxel's point centroid (ties: lowest index), and the
kept indices are emitted in ascending original order.

Honesty: viewer assets are display only and never change a verdict. A missing
or malformed asset is dropped with a warning (``VIEWER_*``); a dropped heatmap
always carries ``heatmap_unavailable_reason``. A deviation array that is not
index-aligned with the scan is never published.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from domains.production.inspection.ply_io import (
    PlyFormatError,
    atomic_write_bytes,
    read_ply,
    write_mesh_ply,
    write_points_ply,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from contracts.production.inspection_result import ShopfloorInspectionResultV1, ViewerAssetsV1

__all__ = [
    "DEFAULT_CAP_POINTS",
    "VOXEL_START_MM",
    "VOXEL_GROWTH",
    "voxel_decimate_indices",
    "read_deviations",
    "ViewerBuild",
    "build_viewer_assets",
]

DEFAULT_CAP_POINTS = 400_000
VOXEL_START_MM = 0.5
VOXEL_GROWTH = 1.5
_MAX_VOXEL_ITERATIONS = 64


def voxel_decimate_indices(
    points: "np.ndarray", cap: int, *, start_mm: float = VOXEL_START_MM, growth: float = VOXEL_GROWTH
) -> tuple["np.ndarray", float | None]:
    """Indices (ascending int64) of the points kept, and the voxel size used
    (``None`` when no decimation was needed). See the module docstring."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if n <= cap:
        return np.arange(n, dtype=np.int64), None
    voxel = float(start_mm)
    for _ in range(_MAX_VOXEL_ITERATIONS):
        keys = np.floor(pts / voxel).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        if len(counts) <= cap:
            sums = np.zeros((len(counts), 3), dtype=np.float64)
            np.add.at(sums, inverse, pts)
            centroids = sums / counts[:, None]
            dist = np.einsum("ij,ij->i", pts - centroids[inverse], pts - centroids[inverse])
            order = np.lexsort((np.arange(n), dist, inverse))  # by voxel, then distance, then index
            first = np.ones(n, dtype=bool)
            first[1:] = inverse[order][1:] != inverse[order][:-1]
            return np.sort(order[first]).astype(np.int64), voxel
        voxel *= growth
    raise ValueError("voxel decimation did not converge")  # pragma: no cover - 1.5^64 grows past any part


def read_deviations(path: str | Path, expected_count: int) -> "np.ndarray":
    """float32 LE deviations; raises ``ValueError`` unless exactly
    ``expected_count`` values that are finite or NaN (no infinities)."""
    import numpy as np

    raw = Path(path).read_bytes()
    if len(raw) % 4:
        raise ValueError(f"deviation file size {len(raw)} is not a multiple of 4")
    dev = np.frombuffer(raw, dtype="<f4")
    if len(dev) != expected_count:
        raise ValueError(f"{len(dev)} deviations for {expected_count} scan points (not index-aligned)")
    if np.isinf(dev).any():
        raise ValueError("deviations contain infinities")
    return dev


@dataclass(frozen=True)
class ViewerBuild:
    """Outcome of :func:`build_viewer_assets`."""

    assets: "ViewerAssetsV1"
    warnings: tuple[str, ...] = ()
    source_points: int | None = None
    kept_points: int | None = None
    voxel_mm: float | None = None
    files: tuple[str, ...] = field(default_factory=tuple)


def _safe_child(base: Path, rel: str) -> Path:
    """``base/rel``; ``rel`` is a contract RelPath (no ``..``, not absolute)."""
    path = (base / rel).resolve()
    if base.resolve() not in path.parents and path != base.resolve():
        raise ValueError(f"asset path escapes the engine directory: {rel}")
    return path


def build_viewer_assets(
    result: "ShopfloorInspectionResultV1",
    engine_dir: str | Path,
    out_dir: str | Path,
    *,
    cap_points: int = DEFAULT_CAP_POINTS,
) -> ViewerBuild:
    """Write the viewer copies into ``out_dir`` and return the rewritten
    ``ViewerAssetsV1`` (paths relative to ``out_dir``)."""
    import numpy as np

    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.inspection_result import BendGeometryFileV1, ViewerAssetsV1
    from contracts.production.layout import RESULT_FILES

    engine = Path(engine_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    va = result.viewer_assets
    warnings: list[str] = []
    written: list[str] = []

    # Stale copies from an earlier attempt must never be served with this result.
    for key in ("viewer_scan", "viewer_deviations", "stage_mesh", "final_mesh", "bend_geometry"):
        stale = out / RESULT_FILES[key]
        if stale.exists():
            stale.unlink()

    # -- scan + deviations (one shared index selection)
    scan_rel: str | None = None
    dev_rel: str | None = None
    point_count: int | None = None
    source_points: int | None = None
    voxel: float | None = None
    unavailable = va.heatmap_unavailable_reason
    points: np.ndarray | None = None
    if va.registered_scan is not None:
        try:
            points = read_ply(_safe_child(engine, va.registered_scan)).vertices
        except FileNotFoundError:
            warnings.append(f"VIEWER_SCAN_MISSING: {va.registered_scan} not found in the engine output")
        except (PlyFormatError, ValueError, OSError) as exc:
            warnings.append(f"VIEWER_SCAN_INVALID: {va.registered_scan}: {exc}")
        if points is not None and va.point_count is not None and len(points) != va.point_count:
            warnings.append(
                f"VIEWER_SCAN_INVALID: {va.registered_scan} has {len(points)} points, result says {va.point_count}"
            )
            points = None
    deviations: np.ndarray | None = None
    if va.deviations is not None:
        if points is None:
            unavailable = "VIEWER_SCAN_UNAVAILABLE"
        else:
            try:
                deviations = read_deviations(_safe_child(engine, va.deviations), len(points))
            except FileNotFoundError:
                warnings.append(f"VIEWER_DEVIATIONS_MISSING: {va.deviations} not found in the engine output")
                unavailable = "VIEWER_DEVIATIONS_MISSING"
            except (ValueError, OSError) as exc:
                warnings.append(f"VIEWER_DEVIATIONS_INVALID: {exc}")
                unavailable = "VIEWER_DEVIATIONS_INVALID"
    if points is not None:
        source_points = len(points)
        idx, voxel = voxel_decimate_indices(points, cap_points)
        write_points_ply(out / RESULT_FILES["viewer_scan"], points[idx])
        scan_rel, point_count = RESULT_FILES["viewer_scan"], int(len(idx))
        written.append(scan_rel)
        if deviations is not None:
            atomic_write_bytes(
                out / RESULT_FILES["viewer_deviations"],
                np.ascontiguousarray(deviations[idx], dtype="<f4").tobytes(),
            )
            dev_rel = RESULT_FILES["viewer_deviations"]
            written.append(dev_rel)
    if dev_rel is None and not unavailable:
        unavailable = "HEATMAP_NOT_PUBLISHED"

    # -- display meshes (re-encoded, validated)
    meshes: dict[str, str | None] = {"stage_mesh": None, "final_mesh": None}
    for key in ("stage_mesh", "final_mesh"):
        rel = getattr(va, key)
        if rel is None:
            continue
        try:
            mesh = read_ply(_safe_child(engine, rel), require_faces=True)
        except FileNotFoundError:
            warnings.append(f"VIEWER_MESH_MISSING: {rel} not found in the engine output")
            continue
        except (PlyFormatError, ValueError, OSError) as exc:
            warnings.append(f"VIEWER_MESH_INVALID: {rel}: {exc}")
            continue
        write_mesh_ply(out / RESULT_FILES[key], mesh.vertices, mesh.faces)  # type: ignore[arg-type]
        meshes[key] = RESULT_FILES[key]
        written.append(RESULT_FILES[key])

    # -- bend geometry: always derived from the result rows
    geometry = BendGeometryFileV1.from_result(result)
    if va.bend_geometry is not None:
        try:
            engine_geo = BendGeometryFileV1.model_validate_json(_safe_child(engine, va.bend_geometry).read_bytes())
            if engine_geo != geometry:
                warnings.append("VIEWER_BEND_GEOMETRY_REGENERATED: engine bend_geometry.json differs from the result rows")
        except FileNotFoundError:
            warnings.append(f"VIEWER_BEND_GEOMETRY_REGENERATED: {va.bend_geometry} not found; derived from the result")
        except (ValueError, OSError):
            warnings.append("VIEWER_BEND_GEOMETRY_REGENERATED: engine bend_geometry.json invalid; derived from the result")
    atomic_write_bytes(out / RESULT_FILES["bend_geometry"], canonical_json_bytes(geometry))
    written.append(RESULT_FILES["bend_geometry"])

    assets = ViewerAssetsV1(
        registered_scan=scan_rel,
        point_count=point_count,
        deviations=dev_rel,
        deviation_reference_stage=va.deviation_reference_stage if dev_rel is not None else None,
        heatmap_band_mm=va.heatmap_band_mm,
        heatmap_unavailable_reason=None if dev_rel is not None else (unavailable or "HEATMAP_NOT_PUBLISHED")[:200],
        segment_labels=None,
        stage_mesh=meshes["stage_mesh"],
        final_mesh=meshes["final_mesh"],
        bend_geometry=RESULT_FILES["bend_geometry"],
    )
    return ViewerBuild(
        assets=assets,
        warnings=tuple(warnings),
        source_points=source_points,
        kept_points=point_count,
        voxel_mm=voxel,
        files=tuple(written),
    )
