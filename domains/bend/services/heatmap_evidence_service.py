from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


HEATMAP_EVIDENCE_CACHE_VERSION = "v1"


def _cache_key(cad_path: str, scan_path: str, *, tolerance_angle: float, runtime_cfg: Any) -> str:
    cad = Path(cad_path)
    scan = Path(scan_path)
    seed = int(getattr(runtime_cfg, "local_refinement_kwargs", {}).get("deterministic_seed", 17))
    payload = {
        "version": HEATMAP_EVIDENCE_CACHE_VERSION,
        "cad_path": str(cad.resolve()),
        "scan_path": str(scan.resolve()),
        "cad_size": cad.stat().st_size if cad.exists() else None,
        "scan_size": scan.stat().st_size if scan.exists() else None,
        "cad_mtime_ns": cad.stat().st_mtime_ns if cad.exists() else None,
        "scan_mtime_ns": scan.stat().st_mtime_ns if scan.exists() else None,
        "tolerance_angle": float(tolerance_angle),
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_aligned_points(aligned_path: Path) -> np.ndarray:
    import trimesh

    loaded = trimesh.load(str(aligned_path), process=False)
    vertices = np.asarray(getattr(loaded, "vertices", []), dtype=np.float32)
    if vertices.size == 0 and hasattr(loaded, "vertices"):
        return np.empty((0, 3), dtype=np.float32)
    return vertices.reshape((-1, 3)).astype(np.float32, copy=False)


def _materialize_cached_artifacts(cache_root: Path, output_dir: Optional[Path]) -> Dict[str, Optional[Path]]:
    aligned_cache = cache_root / "aligned_scan.ply"
    deviations_cache = cache_root / "deviations.npy"
    stats_cache = cache_root / "deviation_stats.json"
    if output_dir is None:
        return {
            "aligned_scan_path": aligned_cache if aligned_cache.exists() else None,
            "deviations_path": deviations_cache if deviations_cache.exists() else None,
            "deviation_stats_path": stats_cache if stats_cache.exists() else None,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_out = output_dir / "aligned_scan.ply"
    deviations_out = output_dir / "deviations.npy"
    stats_out = output_dir / "deviation_stats.json"
    if aligned_cache.exists() and not aligned_out.exists():
        shutil.copy2(aligned_cache, aligned_out)
    if deviations_cache.exists() and not deviations_out.exists():
        shutil.copy2(deviations_cache, deviations_out)
    if stats_cache.exists() and not stats_out.exists():
        shutil.copy2(stats_cache, stats_out)
    return {
        "aligned_scan_path": aligned_out if aligned_out.exists() else None,
        "deviations_path": deviations_out if deviations_out.exists() else None,
        "deviation_stats_path": stats_out if stats_out.exists() else None,
    }


def _load_cached_bundle(cache_root: Path, output_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    meta_path = cache_root / "bundle_meta.json"
    aligned_cache = cache_root / "aligned_scan.ply"
    deviations_cache = cache_root / "deviations.npy"
    if not (meta_path.exists() and aligned_cache.exists() and deviations_cache.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        aligned_points = _load_aligned_points(aligned_cache)
        deviations = np.load(str(deviations_cache))
        materialized = _materialize_cached_artifacts(cache_root, output_dir)
        return {
            "aligned_scan_path": materialized.get("aligned_scan_path"),
            "aligned_points": aligned_points,
            "deviations_path": materialized.get("deviations_path"),
            "deviations": deviations.astype(np.float32, copy=False),
            "deviation_stats": meta.get("deviation_stats"),
            "cache_hit": True,
            "cache_key": cache_root.name,
        }
    except Exception:
        return None


def _write_cache_bundle(
    cache_root: Path,
    *,
    aligned_points: np.ndarray,
    deviations: np.ndarray,
    deviation_stats: Dict[str, Any],
) -> None:
    import trimesh

    cache_root.mkdir(parents=True, exist_ok=True)
    aligned_path = cache_root / "aligned_scan.ply"
    deviations_path = cache_root / "deviations.npy"
    stats_path = cache_root / "deviation_stats.json"
    meta_path = cache_root / "bundle_meta.json"

    colors = np.tile(np.array([[56, 189, 248, 210]], dtype=np.uint8), (len(aligned_points), 1))
    cloud = trimesh.PointCloud(aligned_points, colors=colors)
    cloud.export(str(aligned_path), file_type="ply")
    np.save(str(deviations_path), deviations)
    stats_path.write_text(json.dumps(deviation_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "cache_version": HEATMAP_EVIDENCE_CACHE_VERSION,
                "deviation_stats": deviation_stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_bend_heatmap_evidence(
    cad_path: str,
    scan_path: str,
    *,
    tolerance_angle: float,
    runtime_cfg: Any,
    output_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cache_root: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir) / _cache_key(
            cad_path,
            scan_path,
            tolerance_angle=tolerance_angle,
            runtime_cfg=runtime_cfg,
        )
        cached = _load_cached_bundle(cache_root, output_dir)
        if cached is not None:
            return cached

    import trimesh

    from qc_engine import ScanQCEngine

    engine = ScanQCEngine(progress_callback=None)
    engine.load_reference(cad_path)
    engine.load_scan(scan_path)

    raw_scan_count = int(len(engine.scan_pcd.points)) if engine.scan_pcd is not None else 0
    voxel_size = 2.0 if raw_scan_count >= 650000 else 1.5 if raw_scan_count >= 450000 else 1.0
    engine.preprocess(voxel_size=voxel_size)

    seed = int(getattr(runtime_cfg, 'local_refinement_kwargs', {}).get('deterministic_seed', 17))
    engine.align(auto_scale=True, tolerance=float(tolerance_angle), random_seed=seed)
    aligned_cloud = engine.aligned_scan_raw if engine.aligned_scan_raw is not None else engine.aligned_scan
    if aligned_cloud is None:
        return {'aligned_scan_path': None, 'aligned_points': None, 'deviations_path': None, 'deviations': None, 'deviation_stats': None}

    aligned_points = np.asarray(aligned_cloud.points, dtype=np.float32)
    if len(aligned_points) == 0:
        return {'aligned_scan_path': None, 'aligned_points': None, 'deviations_path': None, 'deviations': None, 'deviation_stats': None}

    if engine.aligned_scan_raw is not None:
        engine.aligned_scan = engine.aligned_scan_raw
    deviations = np.asarray(engine.compute_deviations(), dtype=np.float32)
    abs_devs = np.abs(deviations)
    deviation_stats = {
        'count': int(len(deviations)),
        'mean_abs_mm': float(np.mean(abs_devs)) if len(abs_devs) else 0.0,
        'p95_abs_mm': float(np.percentile(abs_devs, 95)) if len(abs_devs) else 0.0,
        'max_abs_mm': float(np.max(abs_devs)) if len(abs_devs) else 0.0,
    }

    if cache_root is not None:
        _write_cache_bundle(
            cache_root,
            aligned_points=aligned_points,
            deviations=deviations,
            deviation_stats=deviation_stats,
        )

    aligned_path = None
    deviations_path = None
    if cache_root is not None:
        materialized = _materialize_cached_artifacts(cache_root, output_dir)
        aligned_path = materialized.get("aligned_scan_path")
        deviations_path = materialized.get("deviations_path")
    elif output_dir is not None:
        aligned_path = output_dir / 'aligned_scan.ply'
        colors = np.tile(np.array([[56, 189, 248, 210]], dtype=np.uint8), (len(aligned_points), 1))
        cloud = trimesh.PointCloud(aligned_points, colors=colors)
        cloud.export(str(aligned_path), file_type='ply')
        deviations_path = output_dir / 'deviations.npy'
        np.save(str(deviations_path), deviations)
        (output_dir / 'deviation_stats.json').write_text(
            json.dumps(deviation_stats, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

    return {
        'aligned_scan_path': aligned_path if aligned_path is not None and aligned_path.exists() else None,
        'aligned_points': aligned_points,
        'deviations_path': deviations_path if deviations_path is not None and deviations_path.exists() else None,
        'deviations': deviations,
        'deviation_stats': deviation_stats,
        'cache_hit': False,
        'cache_key': cache_root.name if cache_root is not None else None,
    }
