"""
Shared progressive bend-inspection pipeline utilities.

This module centralizes:
- Robust CAD/scan geometry loading (including STEP/IGES via cad_import)
- CAD bend-spec extraction
- Scan bend detection
- CAD-to-scan bend matching

It is used by API endpoints and offline optimization loops to ensure
consistent behavior and parameterization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import gc
import inspect
import itertools
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

from cad_import import import_cad_file
from cad_bend_extractor import CADBendExtractor as LegacyMeshCADBendExtractor
from feature_detection import (
    BendDetector,
    ProgressiveBendMatcher,
    CADBendExtractor,
    BendSpecification,
    BendInspectionReport,
    BendMatch,
    DetectedBend,
    PlaneSegment,
)

logger = logging.getLogger(__name__)

CAD_DIRECT_MESH_EXTS = {".stl", ".ply", ".obj"}
CAD_BREP_EXTS = {".step", ".stp", ".iges", ".igs"}
SCAN_EXTS = {".stl", ".ply", ".obj", ".pcd"}


def _phase_log_enabled() -> bool:
    value = str(os.environ.get("BEND_PHASE_STDOUT", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _emit_phase_log(message: str) -> None:
    line = f"[bend-phase] {message}"
    logger.info(line)
    phase_path = str(os.environ.get("BEND_PHASE_LOG_PATH", "")).strip()
    if phase_path:
        try:
            with open(phase_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            logger.debug("Failed to append bend phase log", exc_info=True)
    if _phase_log_enabled():
        print(line, flush=True)


@dataclass
class BendInspectionRuntimeConfig:
    """Runtime-tunable configuration for bend inspection."""

    cad_import_deflection: float = 0.05

    # Constructor kwargs for CADBendExtractor and BendDetector (CAD point-cloud fallback)
    cad_extractor_kwargs: Dict[str, Any] = field(default_factory=dict)
    cad_detector_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"min_plane_points": 50, "min_plane_area": 30.0}
    )
    cad_detect_call_kwargs: Dict[str, Any] = field(default_factory=lambda: {"preprocess": True})

    # Constructor kwargs for BendDetector (scan path)
    scan_detector_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"min_plane_points": 50, "min_plane_area": 30.0}
    )
    scan_detect_call_kwargs: Dict[str, Any] = field(default_factory=lambda: {"preprocess": True})

    # Constructor kwargs for ProgressiveBendMatcher
    matcher_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_angle_diff": 15.0})
    local_refinement_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "deterministic_seed": 17,
            "retry_seeds": [29, 43, 59],
            "retry_below_fitness": 0.65,
            "min_alignment_fitness": 0.55,
        }
    )
    scan_quality_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "coverage_good_pct": 70.0,
            "coverage_min_pct": 45.0,
            "density_good_pts_per_cm2": 20.0,
            "density_min_pts_per_cm2": 8.0,
            "spacing_good_mm": 2.0,
            "spacing_warn_mm": 3.5,
            "sample_points": 30000,
            "voxel_norm": 0.08,
        }
    )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BendInspectionRuntimeConfig":
        cfg = cls()
        if not isinstance(raw, dict):
            return cfg
        for field_name in (
            "cad_import_deflection",
            "cad_extractor_kwargs",
            "cad_detector_kwargs",
            "cad_detect_call_kwargs",
            "scan_detector_kwargs",
            "scan_detect_call_kwargs",
            "matcher_kwargs",
            "local_refinement_kwargs",
            "scan_quality_kwargs",
        ):
            if field_name in raw:
                setattr(cfg, field_name, copy.deepcopy(raw[field_name]))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cad_import_deflection": copy.deepcopy(self.cad_import_deflection),
            "cad_extractor_kwargs": copy.deepcopy(self.cad_extractor_kwargs),
            "cad_detector_kwargs": copy.deepcopy(self.cad_detector_kwargs),
            "cad_detect_call_kwargs": copy.deepcopy(self.cad_detect_call_kwargs),
            "scan_detector_kwargs": copy.deepcopy(self.scan_detector_kwargs),
            "scan_detect_call_kwargs": copy.deepcopy(self.scan_detect_call_kwargs),
            "matcher_kwargs": copy.deepcopy(self.matcher_kwargs),
            "local_refinement_kwargs": copy.deepcopy(self.local_refinement_kwargs),
            "scan_quality_kwargs": copy.deepcopy(self.scan_quality_kwargs),
        }


@dataclass
class BendInspectionRunDetails:
    """Diagnostics from a bend-inspection run."""

    cad_source: str
    cad_strategy: str
    cad_vertex_count: int
    cad_triangle_count: int
    scan_point_count: int
    cad_bend_count: int
    scan_bend_count: int
    expected_bend_count: int
    expected_progress_pct: float
    overdetected_count: int
    warnings: List[str] = field(default_factory=list)
    scan_quality_status: str = "UNKNOWN"
    scan_coverage_pct: float = 0.0
    scan_density_pts_per_cm2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cad_source": self.cad_source,
            "cad_strategy": self.cad_strategy,
            "cad_vertex_count": self.cad_vertex_count,
            "cad_triangle_count": self.cad_triangle_count,
            "scan_point_count": self.scan_point_count,
            "cad_bend_count": self.cad_bend_count,
            "scan_bend_count": self.scan_bend_count,
            "expected_bend_count": self.expected_bend_count,
            "expected_progress_pct": round(self.expected_progress_pct, 2),
            "overdetected_count": self.overdetected_count,
            "warnings": self.warnings,
            "scan_quality_status": self.scan_quality_status,
            "scan_coverage_pct": round(self.scan_coverage_pct, 2),
            "scan_density_pts_per_cm2": round(self.scan_density_pts_per_cm2, 2),
        }


def _filter_kwargs(callable_obj: Any, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter kwargs to only accepted names for the given callable."""
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def load_bend_runtime_config(config_path: Optional[str] = None) -> BendInspectionRuntimeConfig:
    """
    Load runtime config from JSON file.

    Resolution order:
    1) `config_path` argument (if provided)
    2) `BEND_INSPECTION_CONFIG` environment variable
    3) built-in defaults
    """
    resolved = config_path or os.environ.get("BEND_INSPECTION_CONFIG", "").strip()
    if not resolved:
        default_path = Path(__file__).resolve().parent.parent / "output" / "improvement_loop" / "best_config.latest.json"
        if default_path.exists():
            resolved = str(default_path)
    if not resolved:
        return BendInspectionRuntimeConfig()

    path = Path(resolved)
    if not path.exists():
        logger.warning("Bend config file not found: %s (using defaults)", path)
        return BendInspectionRuntimeConfig()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("runtime_config"), dict):
            raw = raw["runtime_config"]
        cfg = BendInspectionRuntimeConfig.from_dict(raw)
        logger.info("Loaded bend runtime config from %s", path)
        return cfg
    except Exception as exc:
        logger.warning("Failed to load bend config %s: %s (using defaults)", path, exc)
        return BendInspectionRuntimeConfig()


def load_expected_bend_overrides(path: Optional[str] = None) -> Dict[str, int]:
    """
    Load optional expected bend-count overrides derived from drawing cross-check.
    """
    resolved = path or os.environ.get("BEND_EXPECTED_OVERRIDES", "").strip()
    if not resolved:
        resolved = str(
            Path(__file__).resolve().parent.parent
            / "output"
            / "improvement_loop"
            / "manual_cad_validation"
            / "bend_baseline_crosscheck.json"
        )
    p = Path(resolved)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse expected bend overrides %s: %s", p, exc)
        return {}
    if not isinstance(raw, list):
        return {}
    out: Dict[str, int] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        part = str(row.get("part", "")).strip()
        expected = row.get("drawing_expected_bends", None)
        if not part or expected is None:
            continue
        try:
            expected_int = int(expected)
        except Exception:
            continue
        if expected_int > 0:
            out[part] = expected_int
    return out


def load_cad_geometry(
    cad_path: str,
    cad_import_deflection: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """
    Load CAD geometry from file.

    Returns:
        vertices, triangles_or_none, source_label
    """
    import open3d as o3d

    path = Path(cad_path)
    ext = path.suffix.lower()

    if ext in CAD_BREP_EXTS:
        result = import_cad_file(str(path), deflection=cad_import_deflection)
        if not result.success or result.mesh is None:
            raise ValueError(f"Failed to import CAD {path.name}: {result.error}")

        verts = np.asarray(result.mesh.vertices, dtype=np.float64)
        faces = np.asarray(result.mesh.faces, dtype=np.int32)
        if len(verts) == 0:
            raise ValueError(f"CAD import yielded no vertices: {path.name}")
        return verts, faces if len(faces) > 0 else None, "cad_import"

    # Direct mesh path (STL/PLY/OBJ)
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_vertices():
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32) if len(mesh.triangles) > 0 else None
        return verts, faces, "open3d_mesh"

    # Fallback: point cloud CAD (PLY)
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.has_points():
        verts = np.asarray(pcd.points, dtype=np.float64)
        return verts, None, "open3d_point_cloud"

    raise ValueError(f"Failed to load CAD geometry: {path.name}")


def load_scan_points(scan_path: str) -> np.ndarray:
    """Load scan as Nx3 points from PLY/PCD/STL/OBJ."""
    import open3d as o3d

    path = Path(scan_path)
    ext = path.suffix.lower()

    if ext not in SCAN_EXTS:
        raise ValueError(f"Unsupported scan format: {ext}")

    # Point cloud first for PLY/PCD
    if ext in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.has_points():
            pts = np.asarray(pcd.points, dtype=np.float64)
            if len(pts) > 0:
                return pts

    # Mesh fallback
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_vertices():
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        if len(pts) > 0:
            return pts

    raise ValueError(f"Failed to load scan geometry: {path.name}")


def export_bend_reference_visualization(
    cad_path: str,
    output_dir: str | Path,
    cad_import_deflection: float = 0.05,
) -> Dict[str, str]:
    """
    Export a viewer-friendly CAD reference artifact for bend inspection.

    The Bend Inspection UI needs a mesh/point-cloud representation of the
    original CAD model so bend status can be drawn directly on top of it.
    """
    import trimesh

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, triangles, _ = load_cad_geometry(
        cad_path,
        cad_import_deflection=cad_import_deflection,
    )
    ref_path = out_dir / "reference_mesh.ply"

    verts32 = np.asarray(vertices, dtype=np.float32)
    if triangles is not None and len(triangles) > 0:
        faces = np.asarray(triangles, dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=verts32, faces=faces, process=False)
        mesh.export(str(ref_path), file_type="ply")
    else:
        cloud = trimesh.PointCloud(verts32)
        cloud.export(str(ref_path), file_type="ply")

    return {"reference_mesh_path": str(ref_path)}


def extract_cad_bend_specs(
    cad_vertices: np.ndarray,
    cad_triangles: Optional[np.ndarray],
    tolerance_angle: float = 1.0,
    tolerance_radius: float = 0.5,
    cad_extractor_kwargs: Optional[Dict[str, Any]] = None,
    cad_detector_kwargs: Optional[Dict[str, Any]] = None,
    cad_detect_call_kwargs: Optional[Dict[str, Any]] = None,
) -> List[BendSpecification]:
    specs, _ = extract_cad_bend_specs_with_strategy(
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        tolerance_angle=tolerance_angle,
        tolerance_radius=tolerance_radius,
        cad_extractor_kwargs=cad_extractor_kwargs,
        cad_detector_kwargs=cad_detector_kwargs,
        cad_detect_call_kwargs=cad_detect_call_kwargs,
        expected_bend_count_hint=None,
    )
    return specs


def _build_detector_cad_specs(
    cad_vertices: np.ndarray,
    cad_triangles: Optional[np.ndarray],
    tolerance_angle: float,
    tolerance_radius: float,
    cad_extractor_kwargs: Optional[Dict[str, Any]],
    cad_detector_kwargs: Optional[Dict[str, Any]],
    cad_detect_call_kwargs: Optional[Dict[str, Any]],
) -> List[BendSpecification]:
    """Current runtime CAD extraction path."""
    if cad_triangles is not None and len(cad_triangles) > 0:
        extractor_ctor_kwargs = _filter_kwargs(CADBendExtractor.__init__, cad_extractor_kwargs or {})
        extractor = CADBendExtractor(
            default_tolerance_angle=tolerance_angle,
            default_tolerance_radius=tolerance_radius,
            **extractor_ctor_kwargs,
        )
        return extractor.extract_from_mesh(cad_vertices, cad_triangles)

    detector_ctor = _filter_kwargs(BendDetector.__init__, cad_detector_kwargs or {})
    detector = BendDetector(**detector_ctor)
    detect_kwargs = _filter_kwargs(detector.detect_bends, cad_detect_call_kwargs or {})
    detected = detector.detect_bends(cad_vertices, **detect_kwargs)

    return [
        BendSpecification(
            bend_id=b.bend_id,
            target_angle=b.measured_angle,
            target_radius=b.measured_radius,
            bend_line_start=b.bend_line_start,
            bend_line_end=b.bend_line_end,
            tolerance_angle=tolerance_angle,
            tolerance_radius=tolerance_radius,
            flange1_normal=b.flange1.normal,
            flange2_normal=b.flange2.normal,
            flange1_area=b.flange1.area,
            flange2_area=b.flange2.area,
        )
        for b in detected
    ]


def _build_legacy_mesh_cad_specs(
    cad_vertices: np.ndarray,
    cad_triangles: np.ndarray,
    tolerance_angle: float,
    tolerance_radius: float,
) -> List[BendSpecification]:
    """
    Alternative CAD baseline for triangulated meshes using face adjacency.

    This path is often better for tray-like parts imported from STEP where the
    detector-based CAD extractor can collapse multiple local folds into a few
    large rolled transitions.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(cad_vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(cad_triangles, dtype=np.int32))

    extractor = LegacyMeshCADBendExtractor(
        coplanar_threshold=15.0,
        min_area_pct=0.2,
        max_bend_gap=20.0,
        min_bend_angle=8.0,
        dedup_distance=8.0,
    )
    result = extractor.extract_from_mesh(mesh)

    specs: List[BendSpecification] = []
    for i, bend in enumerate(result.bends, start=1):
        specs.append(
            BendSpecification(
                bend_id=f"CAD_B{i}",
                target_angle=float(bend.bend_angle),
                target_radius=3.0,
                bend_line_start=np.asarray(bend.bend_line_start, dtype=np.float64),
                bend_line_end=np.asarray(bend.bend_line_end, dtype=np.float64),
                tolerance_angle=tolerance_angle,
                tolerance_radius=tolerance_radius,
                flange1_normal=np.asarray(bend.surface1_normal, dtype=np.float64),
                flange2_normal=np.asarray(bend.surface2_normal, dtype=np.float64),
                flange1_area=float(bend.surface1_area),
                flange2_area=float(bend.surface2_area),
                bend_form="FOLDED",
            )
        )
    return specs


def _angle_hint_distance(spec_angle: float, cad_angle: float) -> float:
    return min(
        abs(float(cad_angle) - float(spec_angle)),
        abs((180.0 - float(cad_angle)) - float(spec_angle)),
    )


def _spec_angle_coverage_score(
    cad_specs: List[BendSpecification],
    spec_bend_angles_hint: List[float],
    match_threshold_deg: float = 10.0,
) -> Tuple[int, float, int]:
    """
    Score how well a CAD baseline explains the available spec bend angles.

    Score order:
    1. More matched spec angles is better.
    2. Lower mean angle error on matched angles is better.
    3. Smaller bend-count gap vs available spec hint is better.
    """
    hints = [float(v) for v in spec_bend_angles_hint if isinstance(v, (int, float))]
    if not cad_specs or not hints:
        return (0, float("inf"), len(cad_specs))

    remaining = list(range(len(cad_specs)))
    matched_errors: List[float] = []
    for hint in hints:
        best_idx: Optional[int] = None
        best_err = float("inf")
        for idx in remaining:
            err = _angle_hint_distance(hint, cad_specs[idx].target_angle)
            if err < best_err:
                best_err = err
                best_idx = idx
        if best_idx is not None and best_err <= match_threshold_deg:
            remaining.remove(best_idx)
            matched_errors.append(best_err)

    mean_err = float(np.mean(matched_errors)) if matched_errors else float("inf")
    count_gap = abs(len(cad_specs) - len(hints))
    return (len(matched_errors), mean_err, count_gap)


def _spec_score_prefers(
    candidate_score: Tuple[int, float, int],
    incumbent_score: Tuple[int, float, int],
) -> bool:
    cand_matches, cand_err, cand_gap = candidate_score
    inc_matches, inc_err, inc_gap = incumbent_score
    if cand_matches > inc_matches:
        return True
    if cand_matches < inc_matches:
        return False
    if np.isfinite(cand_err) and np.isfinite(inc_err):
        if cand_err + 1.0 < inc_err:
            return True
        if inc_err + 1.0 < cand_err:
            return False
    if cand_gap + 2 < inc_gap:
        return True
    return False


def _normalize_spec_schedule_type(spec_schedule_type: Optional[str]) -> str:
    value = str(spec_schedule_type or "").strip().lower()
    if not value:
        return "unknown"
    return value


def _spec_schedule_uses_strong_angle_matching(spec_schedule_type: Optional[str]) -> bool:
    return _normalize_spec_schedule_type(spec_schedule_type) == "complete_or_near_complete"


def _spec_schedule_uses_compactness_hint(spec_schedule_type: Optional[str]) -> bool:
    return _normalize_spec_schedule_type(spec_schedule_type) == "partial_angle_schedule"


def _rolled_profile_guard(
    detector_specs: List[BendSpecification],
    legacy_specs: List[BendSpecification],
    spec_bend_angles_hint: Optional[List[float]] = None,
) -> bool:
    """
    Prefer the detector baseline when it represents a rounded/rolled profile
    and the legacy mesh path explodes that profile into many folded pseudo-bends.
    """
    if not detector_specs or not legacy_specs:
        return False

    detector_rolled_ratio = float(
        sum(1 for s in detector_specs if str(s.bend_form).upper() == "ROLLED")
    ) / max(1, len(detector_specs))
    legacy_rolled_ratio = float(
        sum(1 for s in legacy_specs if str(s.bend_form).upper() == "ROLLED")
    ) / max(1, len(legacy_specs))

    if detector_rolled_ratio < 0.80:
        return False
    if legacy_rolled_ratio > 0.25:
        return False
    if len(legacy_specs) < len(detector_specs) + 2:
        return False

    hint_angles = [float(v) for v in (spec_bend_angles_hint or []) if isinstance(v, (int, float))]
    acute_hint_ratio = (
        float(sum(1 for angle in hint_angles if angle < 90.0)) / len(hint_angles)
        if hint_angles else 0.0
    )
    detector_score = _spec_angle_coverage_score(detector_specs, hint_angles) if hint_angles else (0, float("inf"), len(detector_specs))
    legacy_score = _spec_angle_coverage_score(legacy_specs, hint_angles) if hint_angles else (0, float("inf"), len(legacy_specs))

    # Rolled parts often have section/callout angles rather than full bend schedules.
    # If the detector baseline is compact and rolled while the legacy path expands
    # into many folded pseudo-bends, keep the detector unless the legacy path
    # materially explains more of the spec angles.
    detector_matches, _, _ = detector_score
    legacy_matches, _, _ = legacy_score
    if legacy_matches >= detector_matches + 2:
        return False
    if hint_angles and acute_hint_ratio >= 0.35:
        return True
    if len(detector_specs) <= 2 and len(legacy_specs) >= max(4, len(detector_specs) * 3):
        return True
    if len(legacy_specs) >= len(detector_specs) * 2:
        return True
    return False


def _choose_mesh_cad_baseline_specs(
    detector_specs: List[BendSpecification],
    legacy_specs: List[BendSpecification],
    expected_bend_count_hint: Optional[int] = None,
    spec_bend_angles_hint: Optional[List[float]] = None,
    spec_schedule_type: Optional[str] = None,
) -> Tuple[List[BendSpecification], str]:
    """
    Choose the better CAD bend baseline for triangulated CAD meshes.

    Selection priority:
    1. If an expected bend count hint exists, prefer the closer baseline.
    2. Otherwise, for suspicious tray-like cases where the detector baseline is
       mostly rolled and significantly undercounts, prefer the legacy mesh path.
    """
    if not legacy_specs:
        return detector_specs, "detector_mesh_no_legacy"
    if not detector_specs:
        return legacy_specs, "legacy_mesh_primary_empty"

    if expected_bend_count_hint is not None and expected_bend_count_hint > 0:
        detector_err = abs(len(detector_specs) - int(expected_bend_count_hint))
        legacy_err = abs(len(legacy_specs) - int(expected_bend_count_hint))
        if legacy_err + 1 < detector_err:
            return legacy_specs, "legacy_mesh_expected_hint"
        if detector_err + 1 < legacy_err:
            return detector_specs, "detector_mesh_expected_hint"

    if _rolled_profile_guard(detector_specs, legacy_specs, spec_bend_angles_hint):
        return detector_specs, "detector_mesh_rolled_profile_hint"

    if spec_bend_angles_hint:
        detector_score = _spec_angle_coverage_score(detector_specs, spec_bend_angles_hint)
        legacy_score = _spec_angle_coverage_score(legacy_specs, spec_bend_angles_hint)
        if _spec_schedule_uses_strong_angle_matching(spec_schedule_type):
            if _spec_score_prefers(detector_score, legacy_score):
                return detector_specs, "detector_mesh_spec_hint"
            if _spec_score_prefers(legacy_score, detector_score):
                return legacy_specs, "legacy_mesh_spec_hint"
        elif _spec_schedule_uses_compactness_hint(spec_schedule_type):
            detector_matches, _, _ = detector_score
            legacy_matches, _, _ = legacy_score
            if detector_matches >= legacy_matches and len(detector_specs) + 2 <= len(legacy_specs):
                return detector_specs, "detector_mesh_partial_schedule_hint"

    detector_rolled_ratio = float(
        sum(1 for s in detector_specs if str(s.bend_form).upper() == "ROLLED")
    ) / max(1, len(detector_specs))
    detector_near_90_ratio = float(
        sum(1 for s in detector_specs if abs(float(s.target_angle) - 90.0) <= 15.0)
    ) / max(1, len(detector_specs))
    legacy_near_90_ratio = float(
        sum(1 for s in legacy_specs if abs(float(s.target_angle) - 90.0) <= 15.0)
    ) / max(1, len(legacy_specs))
    detector_median_radius = float(
        np.median([float(s.target_radius) for s in detector_specs])
    ) if detector_specs else 0.0

    legacy_advantage = len(legacy_specs) - len(detector_specs)
    legacy_ratio = float(len(legacy_specs) / max(1, len(detector_specs)))

    tray_like_undercount = (
        legacy_advantage >= 2
        and legacy_ratio >= 1.35
        and legacy_near_90_ratio >= 0.75
        and detector_near_90_ratio >= 0.75
        and detector_rolled_ratio >= 0.75
        and detector_median_radius >= 6.0
    )
    if tray_like_undercount:
        return legacy_specs, "legacy_mesh_tray_override"

    return detector_specs, "detector_mesh"


def extract_cad_bend_specs_with_strategy(
    cad_vertices: np.ndarray,
    cad_triangles: Optional[np.ndarray],
    tolerance_angle: float = 1.0,
    tolerance_radius: float = 0.5,
    cad_extractor_kwargs: Optional[Dict[str, Any]] = None,
    cad_detector_kwargs: Optional[Dict[str, Any]] = None,
    cad_detect_call_kwargs: Optional[Dict[str, Any]] = None,
    expected_bend_count_hint: Optional[int] = None,
    spec_bend_angles_hint: Optional[List[float]] = None,
    spec_schedule_type: Optional[str] = None,
) -> Tuple[List[BendSpecification], str]:
    """
    Extract CAD bend specs and return the selected strategy label.
    """
    detector_specs = _build_detector_cad_specs(
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        tolerance_angle=tolerance_angle,
        tolerance_radius=tolerance_radius,
        cad_extractor_kwargs=cad_extractor_kwargs,
        cad_detector_kwargs=cad_detector_kwargs,
        cad_detect_call_kwargs=cad_detect_call_kwargs,
    )
    if cad_triangles is None or len(cad_triangles) == 0:
        return detector_specs, "detector_point_cloud"

    try:
        legacy_specs = _build_legacy_mesh_cad_specs(
            cad_vertices=cad_vertices,
            cad_triangles=cad_triangles,
            tolerance_angle=tolerance_angle,
            tolerance_radius=tolerance_radius,
        )
    except Exception as exc:
        logger.info("Legacy mesh CAD extractor unavailable for this mesh: %s", exc)
        return detector_specs, "detector_mesh_legacy_failed"

    return _choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=expected_bend_count_hint,
        spec_bend_angles_hint=spec_bend_angles_hint,
        spec_schedule_type=spec_schedule_type,
    )


def detect_scan_bends(
    scan_points: np.ndarray,
    scan_detector_kwargs: Optional[Dict[str, Any]] = None,
    scan_detect_call_kwargs: Optional[Dict[str, Any]] = None,
):
    """Detect bends in scan points with filtered runtime kwargs."""
    detector_ctor = _filter_kwargs(BendDetector.__init__, scan_detector_kwargs or {})
    detector = BendDetector(**detector_ctor)
    detect_kwargs = _filter_kwargs(detector.detect_bends, scan_detect_call_kwargs or {})
    return detector.detect_bends(scan_points, **detect_kwargs)


def select_cad_bends_for_expected(
    cad_bends: List[BendSpecification],
    expected_bend_count: int,
) -> List[BendSpecification]:
    """
    Select the strongest CAD bend specs when drawing-validated expected count is lower.

    This guards against CAD extraction over-segmentation (common with noisy STL CAD)
    while preserving deterministic ordering.
    """
    if expected_bend_count <= 0 or len(cad_bends) <= expected_bend_count:
        return list(cad_bends)

    ranked = sorted(
        enumerate(cad_bends),
        key=lambda item: (
            float(item[1].flange1_area + item[1].flange2_area),
            float(item[1].bend_line_length),
            float(abs(item[1].target_angle)),
        ),
        reverse=True,
    )[:expected_bend_count]

    keep_indices = sorted(idx for idx, _ in ranked)
    selected: List[BendSpecification] = []
    for new_idx, idx in enumerate(keep_indices, start=1):
        bend = copy.deepcopy(cad_bends[idx])
        bend.bend_id = f"CAD_B{new_idx}"
        selected.append(bend)
    return selected


def _sample_points(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[idx]


def _canonicalize_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    center = np.mean(pts, axis=0)
    centered = pts - center
    scale = float(np.max(np.abs(centered)))
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(centered)
    normalized = centered / scale
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                cov = (normalized.T @ normalized) / max(1, len(normalized) - 1)
        if not np.isfinite(cov).all():
            raise ValueError("Non-finite covariance")
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                coords = normalized @ basis
        if not np.isfinite(coords).all():
            coords = normalized
    except Exception:
        coords = normalized

    lo = np.percentile(coords, 2.0, axis=0)
    hi = np.percentile(coords, 98.0, axis=0)
    mid = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1e-6)
    out = (coords - mid) / span
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _voxel_set(points: np.ndarray, voxel_norm: float) -> set:
    if len(points) == 0:
        return set()
    voxel = max(1e-4, float(voxel_norm))
    idx = np.floor(points / voxel).astype(np.int32)
    uniq = np.unique(idx, axis=0)
    return {tuple(map(int, row)) for row in uniq}


def _coverage_proxy(
    cad_points: np.ndarray,
    scan_points: np.ndarray,
    voxel_norm: float,
) -> Tuple[float, float]:
    cad_norm = _canonicalize_points(cad_points)
    scan_norm = _canonicalize_points(scan_points)
    cad_vox = _voxel_set(cad_norm, voxel_norm)
    if not cad_vox or len(scan_norm) == 0:
        return 0.0, 0.0

    best_cov = 0.0
    best_iou = 0.0
    for perm in itertools.permutations((0, 1, 2)):
        permuted = scan_norm[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            transformed = permuted * np.asarray(signs, dtype=np.float64)
            scan_vox = _voxel_set(transformed, voxel_norm)
            if not scan_vox:
                continue
            inter = len(cad_vox.intersection(scan_vox))
            cov = inter / max(1, len(cad_vox))
            iou = inter / max(1, len(cad_vox) + len(scan_vox) - inter)
            if cov > best_cov or (abs(cov - best_cov) < 1e-9 and iou > best_iou):
                best_cov = cov
                best_iou = iou

    return float(best_cov * 100.0), float(best_iou * 100.0)


def _scan_density(scan_points: np.ndarray, sample_points: int = 20000) -> Tuple[float, float, float]:
    pts = _sample_points(scan_points, max_points=max(1000, int(sample_points)), seed=7)
    if len(pts) < 10:
        return 0.0, 0.0, 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 1e-9)]
    if len(nn) == 0:
        return 0.0, 0.0, 0.0

    median_spacing = float(np.median(nn))
    p90_spacing = float(np.percentile(nn, 90))
    # Approximate area per sample point for near-uniform 2D sampling.
    area_per_point_mm2 = max(1e-9, 0.8660254 * (median_spacing ** 2))
    density_pts_per_cm2 = float(100.0 / area_per_point_mm2)
    return density_pts_per_cm2, median_spacing, p90_spacing


def estimate_scan_quality(
    scan_points: np.ndarray,
    cad_points: np.ndarray,
    expected_bends: int,
    completed_bends: int,
    passed_bends: int,
    quality_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate scan quality for operator gating.

    Metrics:
    - Coverage proxy (%): canonicalized CAD/scan voxel overlap.
    - Density (pts/cm^2): from median nearest-neighbor spacing.
    - Spacing (mm): median and P90 nearest-neighbor distance.
    """
    q = quality_kwargs or {}
    sample_points = int(q.get("sample_points", 30000))
    voxel_norm = float(q.get("voxel_norm", 0.08))
    coverage_good = float(q.get("coverage_good_pct", 70.0))
    coverage_min = float(q.get("coverage_min_pct", 45.0))
    density_good = float(q.get("density_good_pts_per_cm2", 20.0))
    density_min = float(q.get("density_min_pts_per_cm2", 8.0))
    spacing_good = float(q.get("spacing_good_mm", 2.0))
    spacing_warn = float(q.get("spacing_warn_mm", 3.5))

    scan_sample = _sample_points(scan_points, max_points=sample_points, seed=11)
    cad_sample = _sample_points(cad_points, max_points=sample_points, seed=13)

    coverage_pct, overlap_iou_pct = _coverage_proxy(
        cad_points=cad_sample,
        scan_points=scan_sample,
        voxel_norm=voxel_norm,
    )
    density_pts_per_cm2, median_spacing_mm, p90_spacing_mm = _scan_density(
        scan_points=scan_sample,
        sample_points=min(sample_points, len(scan_sample)),
    )

    expected = max(1, int(expected_bends))
    bend_evidence_pct = float(min(int(completed_bends), expected) / expected * 100.0)
    in_spec_ratio_pct = float((int(passed_bends) / max(1, int(completed_bends))) * 100.0) if completed_bends > 0 else 0.0

    cov_norm = np.clip((coverage_pct - coverage_min) / max(1e-6, (coverage_good - coverage_min)), 0.0, 1.0)
    dens_norm = np.clip((density_pts_per_cm2 - density_min) / max(1e-6, (density_good - density_min)), 0.0, 1.0)
    spacing_norm = 1.0 - np.clip((median_spacing_mm - spacing_good) / max(1e-6, (spacing_warn - spacing_good)), 0.0, 1.0)
    bend_norm = np.clip(bend_evidence_pct / 100.0, 0.0, 1.0)

    score = float(100.0 * (0.45 * cov_norm + 0.35 * dens_norm + 0.10 * spacing_norm + 0.10 * bend_norm))
    if coverage_pct >= coverage_good and density_pts_per_cm2 >= density_good and median_spacing_mm <= spacing_good:
        status = "GOOD"
    elif coverage_pct < coverage_min or density_pts_per_cm2 < density_min or median_spacing_mm > spacing_warn:
        status = "POOR"
    else:
        status = "FAIR"

    queue_label = {
        "GOOD": "Scan Quality: Good",
        "FAIR": "Scan Quality: Fair",
        "POOR": "Scan Quality: Poor",
    }[status]
    queue_color = {
        "GOOD": "green",
        "FAIR": "yellow",
        "POOR": "red",
    }[status]

    notes: List[str] = []
    if coverage_pct < coverage_min:
        notes.append("Coverage is low; add additional viewpoints (including opposite side flanges).")
    elif coverage_pct < coverage_good:
        notes.append("Coverage is moderate; one more sweep can improve bend confidence.")
    if density_pts_per_cm2 < density_min or median_spacing_mm > spacing_warn:
        notes.append("Scan density is low; move scanner closer or increase capture resolution.")
    elif density_pts_per_cm2 < density_good:
        notes.append("Density is acceptable but below preferred metrology target for edge/bend detail.")
    if not notes:
        notes.append("Coverage and density are within preferred range for bend-by-bend validation.")

    return {
        "status": status,
        "queue_label": queue_label,
        "queue_color": queue_color,
        "score": round(score, 1),
        "coverage_pct": round(coverage_pct, 1),
        "overlap_iou_pct": round(overlap_iou_pct, 1),
        "density_pts_per_cm2": round(density_pts_per_cm2, 1),
        "median_spacing_mm": round(median_spacing_mm, 3),
        "p90_spacing_mm": round(p90_spacing_mm, 3),
        "bend_evidence_pct": round(bend_evidence_pct, 1),
        "in_spec_ratio_pct": round(in_spec_ratio_pct, 1),
        "point_count": int(len(scan_points)),
        "notes": notes,
        "thresholds": {
            "coverage_good_pct": coverage_good,
            "coverage_min_pct": coverage_min,
            "density_good_pts_per_cm2": density_good,
            "density_min_pts_per_cm2": density_min,
            "spacing_good_mm": spacing_good,
            "spacing_warn_mm": spacing_warn,
        },
    }


def match_detected_bends(
    detected_bends,
    cad_bends: List[BendSpecification],
    part_id: str,
    matcher_kwargs: Optional[Dict[str, Any]] = None,
) -> BendInspectionReport:
    """Match detected bends to CAD specs using ProgressiveBendMatcher."""
    matcher_ctor = _filter_kwargs(ProgressiveBendMatcher.__init__, matcher_kwargs or {})
    matcher = ProgressiveBendMatcher(**matcher_ctor)
    return matcher.match(detected_bends, cad_bends, part_id=part_id)


def _fallback_scan_detection_config(
    runtime_config: BendInspectionRuntimeConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build a higher-recall detector config for hard scans.

    Intended for selective fallback only when primary detection under-recovers.
    """
    ctor = copy.deepcopy(runtime_config.scan_detector_kwargs or {})
    call = copy.deepcopy(runtime_config.scan_detect_call_kwargs or {})

    ctor["min_plane_points"] = max(15, int(ctor.get("min_plane_points", 50) * 0.7))
    ctor["min_plane_area"] = max(6.0, float(ctor.get("min_plane_area", 30.0) * 0.65))
    ctor["adjacency_threshold"] = float(ctor.get("adjacency_threshold", 5.0)) * 1.25
    ctor["ransac_iterations"] = max(260, int(ctor.get("ransac_iterations", 120)))
    ctor["min_bend_angle"] = max(5.0, float(ctor.get("min_bend_angle", 10.0)) - 2.0)

    call["preprocess"] = True
    call["voxel_size"] = max(0.55, float(call.get("voxel_size", 1.0)) * 0.75)
    call["outlier_nb_neighbors"] = max(8, int(call.get("outlier_nb_neighbors", 20) * 0.8))
    call["outlier_std_ratio"] = max(2.0, float(call.get("outlier_std_ratio", 2.0)) * 1.1)
    call["multiscale"] = True

    return ctor, call


def _scan_runtime_profile_name(point_count: int, fallback: bool = False) -> Optional[str]:
    """Return a stable runtime profile label for large full-scope scans."""
    if point_count >= 650000:
        return "full_scan_dense_fallback" if fallback else "full_scan_dense_primary"
    if point_count >= 450000:
        return "full_scan_medium_fallback" if fallback else "full_scan_medium_primary"
    return None


def _apply_scan_runtime_profile(
    scan_detector_kwargs: Optional[Dict[str, Any]],
    scan_detect_call_kwargs: Optional[Dict[str, Any]],
    point_count: int,
    *,
    fallback: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Adapt runtime detection settings for large full-resolution scans.

    We still process the original full scan, but we raise the preprocessing
    voxel size and disable expensive multiscale fallback when point counts are
    high enough to destabilize Open3D on memory-limited development laptops.
    """
    ctor = copy.deepcopy(scan_detector_kwargs or {})
    call = copy.deepcopy(scan_detect_call_kwargs or {})
    profile = _scan_runtime_profile_name(point_count, fallback=fallback)
    if profile is None:
        return ctor, call, None

    call["preprocess"] = True

    if point_count >= 650000:
        ctor["min_plane_points"] = max(55, int(ctor.get("min_plane_points", 50)))
        ctor["min_plane_area"] = max(32.0, float(ctor.get("min_plane_area", 30.0)))
        call["voxel_size"] = max(1.5 if fallback else 2.0, float(call.get("voxel_size", 0.3)))
        call["outlier_nb_neighbors"] = min(max(10, int(call.get("outlier_nb_neighbors", 20))), 12)
        call["outlier_std_ratio"] = max(2.5, float(call.get("outlier_std_ratio", 2.0)))
        call["multiscale"] = False
        return ctor, call, profile

    ctor["min_plane_points"] = max(50, int(ctor.get("min_plane_points", 50)))
    ctor["min_plane_area"] = max(30.0, float(ctor.get("min_plane_area", 30.0)))
    call["voxel_size"] = max(1.75 if fallback else 2.0, float(call.get("voxel_size", 0.3)))
    call["outlier_nb_neighbors"] = min(max(10, int(call.get("outlier_nb_neighbors", 20))), 14)
    call["outlier_std_ratio"] = max(2.3, float(call.get("outlier_std_ratio", 2.0)))
    call["multiscale"] = False
    return ctor, call, profile


def _report_rank(
    report: BendInspectionReport,
    expected_bend_count: int,
) -> Tuple[int, int, int, int, int]:
    """
    Ranking tuple for selecting better report under same CAD baseline.
    """
    target = max(1, int(expected_bend_count))
    detected_capped = min(int(report.detected_count), target)
    overdetected = max(0, int(report.detected_count) - target)
    return (
        detected_capped,
        int(report.pass_count),
        -overdetected,
        -int(report.fail_count),
        -int(report.warning_count),
    )


def _match_priority(match: Optional[BendMatch]) -> int:
    if match is None:
        return -1
    return {
        "PASS": 3,
        "WARNING": 2,
        "FAIL": 1,
        "NOT_DETECTED": 0,
    }.get(str(match.status).upper(), 0)


def _match_penalty(match: Optional[BendMatch]) -> Tuple[float, float, float, float]:
    if match is None or str(match.status).upper() == "NOT_DETECTED":
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    angle = abs(float(match.angle_deviation or 0.0))
    center = abs(float(match.line_center_deviation_mm or 0.0))
    radius = abs(float(match.radius_deviation or 0.0))
    confidence = -float(match.match_confidence or 0.0)
    return (angle, center, radius, confidence)


def _select_better_match(primary: Optional[BendMatch], candidate: Optional[BendMatch]) -> Optional[BendMatch]:
    if primary is None:
        return copy.deepcopy(candidate)
    if candidate is None:
        return copy.deepcopy(primary)
    primary_priority = _match_priority(primary)
    candidate_priority = _match_priority(candidate)
    if candidate_priority > primary_priority:
        return copy.deepcopy(candidate)
    if primary_priority > candidate_priority:
        return copy.deepcopy(primary)
    if _match_penalty(candidate) < _match_penalty(primary):
        return copy.deepcopy(candidate)
    return copy.deepcopy(primary)


def _match_has_position_verification(match: Optional[BendMatch]) -> bool:
    if match is None:
        return False
    return any(
        metric is not None
        for metric in (
            match.line_start_deviation_mm,
            match.line_end_deviation_mm,
            match.line_center_deviation_mm,
        )
    )


def _prefer_authoritative_local_absence(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
    local_alignment_fitness: Optional[float],
) -> bool:
    """
    For dense scans with a strong CAD-guided alignment, a local NOT_DETECTED result
    should override a weak unaligned global match that never had positional evidence.
    """
    if primary is None or candidate is None:
        return False
    if str(candidate.status).upper() != "NOT_DETECTED":
        return False
    if str(primary.status).upper() == "NOT_DETECTED":
        return False
    if local_alignment_fitness is None or float(local_alignment_fitness) < 0.80:
        return False
    candidate_observability = str(getattr(candidate, "observability_state", "")).upper()
    if candidate_observability not in {"OBSERVED_NOT_FORMED", "PARTIALLY_OBSERVED", "UNOBSERVED"}:
        return False
    if _match_has_position_verification(primary):
        return False
    if float(primary.match_confidence or 0.0) >= 0.85:
        return False
    return True


def merge_bend_reports(
    primary_report: BendInspectionReport,
    local_report: BendInspectionReport,
    part_id: str,
    local_alignment_fitness: Optional[float] = None,
) -> BendInspectionReport:
    """
    Merge per-bend evidence from primary and local reports.

    Dense local refinement should add or improve bend evidence, not erase a valid
    primary detection for a different bend on partial scans.
    """
    local_by_id = {m.cad_bend.bend_id: m for m in local_report.matches}
    merged_matches: List[BendMatch] = []
    seen: set[str] = set()

    for primary_match in primary_report.matches:
        bend_id = primary_match.cad_bend.bend_id
        local_match = local_by_id.get(bend_id)
        if _prefer_authoritative_local_absence(primary_match, local_match, local_alignment_fitness):
            merged_matches.append(copy.deepcopy(local_match))
        else:
            merged_matches.append(_select_better_match(primary_match, local_match))
        seen.add(bend_id)

    for local_match in local_report.matches:
        bend_id = local_match.cad_bend.bend_id
        if bend_id not in seen:
            merged_matches.append(copy.deepcopy(local_match))

    merged_unmatched = list(primary_report.unmatched_detections or [])
    if local_report.unmatched_detections:
        merged_unmatched.extend(local_report.unmatched_detections)

    merged_report = BendInspectionReport(
        part_id=part_id,
        matches=merged_matches,
        unmatched_detections=merged_unmatched,
    )
    merged_report.compute_summary()
    return merged_report


def _dense_scan_requires_local_refinement(
    scan_points: np.ndarray,
    report: BendInspectionReport,
    expected_bend_count: int,
) -> bool:
    """Dense scans should prefer CAD-guided local verification over brute-force fallback."""
    if len(scan_points) < 550000:
        return False
    target = max(1, int(expected_bend_count))
    return int(report.detected_count) < max(1, int(np.ceil(0.75 * target)))


def _align_engine_with_retries(
    engine: Any,
    part_id: str,
    tolerance_angle: float,
    refinement_cfg: Optional[Dict[str, Any]],
    phase_cb: Any,
) -> Tuple[Optional[np.ndarray], float, float, int, List[str]]:
    """
    Run deterministic CAD-guided alignment, retrying alternate seeds only when the
    first fit is weak. This keeps the local refinement reproducible and prevents a
    low-quality random alignment from poisoning per-bend measurements.
    """
    cfg = refinement_cfg or {}
    base_seed = int(cfg.get("deterministic_seed", 17))
    retry_seeds = [int(seed) for seed in cfg.get("retry_seeds", [29, 43, 59]) if seed is not None]
    retry_below_fitness = float(cfg.get("retry_below_fitness", 0.65))
    min_alignment_fitness = float(cfg.get("min_alignment_fitness", 0.55))

    seeds: List[int] = []
    for seed in [base_seed, *retry_seeds]:
        if seed not in seeds:
            seeds.append(seed)

    warnings_local: List[str] = []
    best_points: Optional[np.ndarray] = None
    best_fitness = -1.0
    best_rmse = float("inf")
    best_seed = seeds[0]
    initial_fitness: Optional[float] = None

    for idx, seed in enumerate(seeds):
        phase_cb(f"align attempt begin seed={seed}")
        fitness, rmse = engine.align(auto_scale=True, tolerance=tolerance_angle, random_seed=seed)
        phase_cb(f"align attempt end seed={seed} fitness={fitness:.4f} rmse={rmse:.4f}")
        if engine.aligned_scan is not None:
            aligned_points = np.asarray(engine.aligned_scan.points)
            if fitness > best_fitness:
                best_fitness = float(fitness)
                best_rmse = float(rmse)
                best_seed = int(seed)
                best_points = aligned_points.copy()
        if idx == 0:
            initial_fitness = float(fitness)
            if fitness >= retry_below_fitness:
                break

    if initial_fitness is not None and len(seeds) > 1 and initial_fitness < retry_below_fitness:
        warnings_local.append(
            "Dense-scan local refinement retried alternate deterministic alignment seeds "
            f"because initial fitness was {initial_fitness:.2%}; selected seed {best_seed} "
            f"with fitness {best_fitness:.2%}."
        )

    if best_points is None:
        return None, best_fitness, best_rmse, best_seed, warnings_local

    if best_fitness < min_alignment_fitness:
        warnings_local.append(
            "Dense-scan local refinement discarded: alignment fitness "
            f"{best_fitness:.2%} is below the minimum trusted threshold of {min_alignment_fitness:.0%}."
        )
        return None, best_fitness, best_rmse, best_seed, warnings_local

    return best_points, best_fitness, best_rmse, best_seed, warnings_local


def _measurement_confidence_score(label: Optional[str]) -> float:
    mapping = {
        "high": 0.9,
        "medium": 0.72,
        "low": 0.5,
        "very_low": 0.25,
    }
    return float(mapping.get(str(label or "").strip().lower(), 0.4))


def _local_bend_frame(cad_bend_dict: Dict[str, Any], cad_bend: BendSpecification) -> Dict[str, Any]:
    center = np.asarray(cad_bend_dict.get("bend_center", (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0), dtype=np.float64)
    tangent = np.asarray(cad_bend_dict.get("bend_direction", cad_bend.bend_line_direction), dtype=np.float64)
    n1 = np.asarray(cad_bend_dict.get("surface1_normal", cad_bend.flange1_normal if cad_bend.flange1_normal is not None else np.zeros(3)), dtype=np.float64)
    n2 = np.asarray(cad_bend_dict.get("surface2_normal", cad_bend.flange2_normal if cad_bend.flange2_normal is not None else np.zeros(3)), dtype=np.float64)

    def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            vec = fallback
            norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return vec / norm

    tangent = _unit(tangent, cad_bend.bend_line_end - cad_bend.bend_line_start)
    n1 = _unit(n1, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    n2 = _unit(n2, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    bisector = _unit(n1 + n2, n1)
    return {
        "center": [round(float(x), 4) for x in center.tolist()],
        "tangent": [round(float(x), 6) for x in tangent.tolist()],
        "surface1_normal": [round(float(x), 6) for x in n1.tolist()],
        "surface2_normal": [round(float(x), 6) for x in n2.tolist()],
        "bisector": [round(float(x), 6) for x in bisector.tolist()],
    }


def _local_measurement_observability(
    measurement: Dict[str, Any],
    cad_angle: float,
) -> Dict[str, Any]:
    point_count = int(measurement.get("point_count") or 0)
    side1_points = int(measurement.get("side1_points") or 0)
    side2_points = int(measurement.get("side2_points") or 0)
    observed_surface_count = int(side1_points >= 12) + int(side2_points >= 12)
    confidence = _measurement_confidence_score(measurement.get("measurement_confidence"))

    alignments = [
        float(value)
        for value in (measurement.get("cad_alignment1"), measurement.get("cad_alignment2"))
        if isinstance(value, (int, float))
    ]
    alignment_score = float(sum(alignments) / len(alignments)) if alignments else confidence
    point_score = min(1.0, point_count / 140.0)
    surface_score = min(1.0, observed_surface_count / 2.0)
    evidence_score = max(
        0.0,
        min(
            1.0,
            0.35 * surface_score + 0.25 * point_score + 0.20 * confidence + 0.20 * alignment_score,
        ),
    )

    measured_angle = measurement.get("measured_angle")
    success = bool(measurement.get("success"))
    state = "UNOBSERVED"
    if success and measured_angle is not None:
        flat_gap = abs(float(measured_angle) - 180.0)
        target_gap = abs(float(measured_angle) - float(cad_angle))
        if observed_surface_count >= 2 and point_count >= 24 and flat_gap + 8.0 < target_gap:
            state = "OBSERVED_NOT_FORMED"
        elif observed_surface_count >= 2:
            state = "OBSERVED_FORMED"
        else:
            state = "PARTIALLY_OBSERVED"
    elif observed_surface_count >= 2 or point_count >= 20:
        state = "PARTIALLY_OBSERVED"

    return {
        "state": state,
        "confidence": confidence,
        "observed_surface_count": observed_surface_count,
        "point_count": point_count,
        "evidence_score": evidence_score,
    }


def _placeholder_plane(
    plane_id: int,
    center: np.ndarray,
    normal: Optional[np.ndarray],
    area: float,
    inlier_count: int,
) -> PlaneSegment:
    n = np.asarray(normal if normal is not None else np.array([1.0, 0.0, 0.0]), dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / norm
    centroid = np.asarray(center, dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.float64)
    return PlaneSegment(
        plane_id=plane_id,
        normal=n,
        d=float(-np.dot(n, centroid)),
        centroid=centroid,
        points=empty,
        boundary_points=empty,
        area=max(1.0, float(area)),
        inlier_count=max(1, int(inlier_count)),
    )


def _build_not_detected_match(
    cad_bend: BendSpecification,
    matcher: ProgressiveBendMatcher,
    action_item: str,
    *,
    observability_state: str = "UNOBSERVED",
    observability_confidence: float = 0.0,
    observed_surface_count: int = 0,
    local_point_count: int = 0,
    local_evidence_score: float = 0.0,
    measurement_mode: str = "global_detection",
    measurement_method: Optional[str] = None,
    measurement_context: Optional[Dict[str, Any]] = None,
) -> BendMatch:
    return BendMatch(
        cad_bend=cad_bend,
        detected_bend=None,
        status="NOT_DETECTED",
        match_confidence=0.0,
        bend_form=matcher._classify_bend_form(cad_bend, None),
        expected_line_length_mm=cad_bend.bend_line_length,
        expected_arc_length_mm=matcher._compute_arc_length(cad_bend.target_angle, cad_bend.target_radius),
        issue_location="bend_line",
        action_item=action_item,
        observability_state=observability_state,
        observability_confidence=float(observability_confidence or 0.0),
        observed_surface_count=int(observed_surface_count or 0),
        local_point_count=int(local_point_count or 0),
        local_evidence_score=float(local_evidence_score or 0.0),
        measurement_mode=measurement_mode,
        measurement_method=measurement_method,
        measurement_context=measurement_context or {},
    )


def refine_dense_scan_bends_via_alignment(
    cad_path: str,
    scan_path: str,
    cad_bends: List[BendSpecification],
    part_id: str,
    tolerance_angle: float,
    matcher_kwargs: Optional[Dict[str, Any]] = None,
    local_refinement_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[BendInspectionReport], List[str]]:
    """
    Align a dense scan to CAD and measure bends locally at CAD bend locations.

    This uses the existing ICP alignment + CAD-guided local bend measurement path
    from the general QC engine, which is more scalable for large scans than a
    denser global bend detector sweep.
    """
    try:
        from qc_engine import ScanQCEngine
        from cad_bend_extractor import CADBendExtractor as MeshCADBendExtractor
        from bend_detector import measure_all_scan_bends
    except Exception as exc:  # pragma: no cover - import surface varies by env
        return None, [f"Dense-scan local refinement unavailable: {exc}"]

    warnings_local: List[str] = []
    refine_t0 = perf_counter()

    def _phase(message: str) -> None:
        elapsed = perf_counter() - refine_t0
        _emit_phase_log(
            f"refine_dense_scan_bends_via_alignment[{part_id}] +{elapsed:7.2f}s {message}"
        )

    def _engine_progress(update: Any) -> None:
        _phase(
            f"qc_engine stage={getattr(update, 'stage', '?')} "
            f"progress={getattr(update, 'progress', '?')} "
            f"message={getattr(update, 'message', '')}"
        )

    _phase(f"start cad={Path(cad_path).name} scan={Path(scan_path).name}")
    engine = ScanQCEngine(progress_callback=_engine_progress)

    _phase("load_reference begin")
    engine.load_reference(cad_path)
    _phase("load_reference end")

    _phase("load_scan begin")
    engine.load_scan(scan_path)
    _phase("load_scan end")
    raw_scan_count = int(len(engine.scan_pcd.points)) if engine.scan_pcd is not None else 0
    voxel_size = 2.0 if raw_scan_count >= 650000 else 1.5 if raw_scan_count >= 450000 else 1.0
    _phase(f"preprocess begin voxel={voxel_size:.2f} raw_scan_count={raw_scan_count}")
    engine.preprocess(voxel_size=voxel_size)
    processed_scan_count = int(len(engine.scan_pcd.points)) if engine.scan_pcd is not None else 0
    _phase(f"preprocess end processed_scan_count={processed_scan_count}")
    _phase("align begin")
    aligned_points_best, fitness, rmse, selected_seed, retry_warnings = _align_engine_with_retries(
        engine=engine,
        part_id=part_id,
        tolerance_angle=tolerance_angle,
        refinement_cfg=local_refinement_kwargs,
        phase_cb=_phase,
    )
    warnings_local.extend(retry_warnings)
    _phase(f"align end fitness={fitness:.4f} rmse={rmse:.4f} seed={selected_seed}")

    if aligned_points_best is None:
        _phase("abort alignment below trusted threshold")
        return None, warnings_local

    if engine.aligned_scan is None:
        _phase("abort no aligned scan produced")
        return None, ["Dense-scan local refinement failed: alignment produced no aligned scan/reference mesh"]
    engine.aligned_scan.points = o3d.utility.Vector3dVector(aligned_points_best)

    if engine.aligned_scan is None or engine.reference_mesh is None:
        _phase("abort no aligned scan/reference mesh")
        return None, ["Dense-scan local refinement failed: alignment produced no aligned scan/reference mesh"]

    extractor = MeshCADBendExtractor()
    _phase("extract_from_mesh begin")
    cad_extract = extractor.extract_from_mesh(engine.reference_mesh)
    _phase(f"extract_from_mesh end bends={cad_extract.total_bends}")
    if cad_extract.total_bends <= 0:
        return None, ["Dense-scan local refinement failed: CAD mesh extractor found no bends"]

    cad_bend_dicts = [bend.to_dict() for bend in cad_extract.bends]
    cad_vertices = np.asarray(engine.reference_mesh.vertices)
    cad_triangles = np.asarray(engine.reference_mesh.triangles)
    surface_face_map = {
        surface.surface_id: surface.face_indices
        for surface in cad_extract.surfaces
    }
    aligned_points = np.asarray(engine.aligned_scan.points)
    _phase(f"measure_all_scan_bends begin aligned_points={len(aligned_points)} cad_bends={len(cad_bend_dicts)}")
    measurements = measure_all_scan_bends(
        aligned_points,
        cad_bend_dicts,
        search_radius=35.0,
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        surface_face_map=surface_face_map,
        deviations=None,
    )
    _phase(f"measure_all_scan_bends end measurements={len(measurements)}")

    matcher_ctor = _filter_kwargs(ProgressiveBendMatcher.__init__, matcher_kwargs or {})
    matcher_ctor["position_mode"] = "enabled"
    matcher = ProgressiveBendMatcher(**matcher_ctor)

    measurement_entries = []
    for cad_bend_dict, measurement in zip(cad_bend_dicts, measurements):
        center = np.asarray(cad_bend_dict["bend_center"], dtype=np.float64)
        measurement_entries.append(
            {
                "center": center,
                "cad_angle": float(cad_bend_dict["bend_angle"]),
                "cad_normals": (
                    np.asarray(cad_bend_dict["surface1_normal"], dtype=np.float64),
                    np.asarray(cad_bend_dict["surface2_normal"], dtype=np.float64),
                ),
                "measurement": measurement,
            }
        )

    matches: List[BendMatch] = []
    used_measurements: set[int] = set()
    for cad_bend in cad_bends:
        cad_mid = (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0
        best_idx: Optional[int] = None
        best_score = float("inf")
        for idx, entry in enumerate(measurement_entries):
            if idx in used_measurements:
                continue
            center_dist = float(np.linalg.norm(entry["center"] - cad_mid))
            if center_dist > 45.0:
                continue
            angle_diff = abs(entry["cad_angle"] - cad_bend.target_angle)
            if angle_diff > 35.0:
                continue
            score = center_dist + (angle_diff * 2.0)
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            fallback_frame = {
                "local_frame": {
                    "center": [round(float(x), 4) for x in cad_mid.tolist()],
                    "tangent": [round(float(x), 6) for x in np.asarray(cad_bend.bend_line_direction, dtype=np.float64).tolist()],
                },
                "search_radius_mm": 35.0,
            }
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item="Dense-scan local verification found no evidence at this CAD bend location.",
                    measurement_mode="cad_local_neighborhood",
                    measurement_context=fallback_frame,
                )
            )
            continue

        used_measurements.add(best_idx)
        entry = measurement_entries[best_idx]
        measurement = entry["measurement"]
        measurement_frame = _local_bend_frame(cad_bend_dicts[best_idx], cad_bend)
        observability = _local_measurement_observability(measurement, cad_bend.target_angle)
        measurement_context = {
            "local_frame": measurement_frame,
            "search_radius_mm": 35.0,
            "point_count": int(measurement.get("point_count") or 0),
            "side1_points": int(measurement.get("side1_points") or 0),
            "side2_points": int(measurement.get("side2_points") or 0),
            "measurement_method": measurement.get("measurement_method"),
            "measurement_confidence_label": measurement.get("measurement_confidence"),
            "cad_alignment1": measurement.get("cad_alignment1"),
            "cad_alignment2": measurement.get("cad_alignment2"),
            "planarity1": measurement.get("planarity1"),
            "planarity2": measurement.get("planarity2"),
            "observability_state": observability["state"],
            "local_evidence_score": round(float(observability["evidence_score"]), 4),
        }
        if not measurement.get("success") or measurement.get("measured_angle") is None:
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=f"Dense-scan local verification failed at this CAD bend location: {measurement.get('error', 'insufficient local evidence')}",
                    observability_state=observability["state"],
                    observability_confidence=observability["confidence"],
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context=measurement_context,
                )
            )
            continue

        if observability["state"] == "OBSERVED_NOT_FORMED":
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=(
                        "Scan coverage reaches this bend, but the local angle is still closer "
                        "to a flat/open state than the target bend angle."
                    ),
                    observability_state=observability["state"],
                    observability_confidence=observability["confidence"],
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context=measurement_context,
                )
            )
            continue

        confidence = _measurement_confidence_score(measurement.get("measurement_confidence"))
        point_count = int(measurement.get("point_count") or (measurement.get("side1_points", 0) + measurement.get("side2_points", 0)))
        plane1 = _placeholder_plane(
            1,
            cad_mid,
            cad_bend.flange1_normal if cad_bend.flange1_normal is not None else entry["cad_normals"][0],
            cad_bend.flange1_area,
            point_count // 2,
        )
        plane2 = _placeholder_plane(
            2,
            cad_mid,
            cad_bend.flange2_normal if cad_bend.flange2_normal is not None else entry["cad_normals"][1],
            cad_bend.flange2_area,
            point_count // 2,
        )
        detected = DetectedBend(
            bend_id=f"L{cad_bend.bend_id}",
            measured_angle=float(measurement["measured_angle"]),
            measured_radius=float(cad_bend.target_radius),
            bend_line_start=np.asarray(cad_bend.bend_line_start, dtype=np.float64),
            bend_line_end=np.asarray(cad_bend.bend_line_end, dtype=np.float64),
            bend_line_direction=np.asarray(cad_bend.bend_line_direction, dtype=np.float64),
            confidence=confidence,
            flange1=plane1,
            flange2=plane2,
            inlier_count=max(1, point_count),
            radius_fit_error=0.0,
        )
        matches.append(
            matcher._build_match(
                cad_bend=cad_bend,
                detected=detected,
                cost=max(0.0, 1.0 - confidence),
                use_position_cost=True,
            )
        )
        matches[-1].observability_state = observability["state"]
        matches[-1].observability_confidence = observability["confidence"]
        matches[-1].observed_surface_count = observability["observed_surface_count"]
        matches[-1].local_point_count = observability["point_count"]
        matches[-1].local_evidence_score = observability["evidence_score"]
        matches[-1].measurement_mode = "cad_local_neighborhood"
        matches[-1].measurement_method = str(measurement.get("measurement_method") or "unknown")
        matches[-1].measurement_context = measurement_context

    report = BendInspectionReport(part_id=part_id, matches=matches, unmatched_detections=[])
    report.compute_summary()
    setattr(report, "local_alignment_fitness", fitness)
    setattr(report, "local_alignment_seed", selected_seed)
    warnings_local.append(
        "Dense-scan local refinement used CAD-guided ICP alignment "
        f"(voxel={voxel_size:.2f}mm, fitness={fitness:.2%}, rmse={rmse:.3f}mm, seed={selected_seed})"
    )
    _phase(
        "complete "
        f"detected={report.detected_count} pass={report.pass_count} "
        f"warnings={report.warning_count} fail={report.fail_count}"
    )
    return report, warnings_local


def detect_and_match_with_fallback(
    scan_points: np.ndarray,
    cad_bends: List[BendSpecification],
    part_id: str,
    runtime_config: BendInspectionRuntimeConfig,
    expected_bend_count: Optional[int] = None,
) -> Tuple[BendInspectionReport, int, bool]:
    """
    Detect and match bends with selective high-recall fallback for weak scans.

    Returns:
        (selected_report, selected_scan_bend_count, fallback_used)
    """
    primary_ctor, primary_call, _ = _apply_scan_runtime_profile(
        runtime_config.scan_detector_kwargs,
        runtime_config.scan_detect_call_kwargs,
        len(scan_points),
        fallback=False,
    )
    primary_detected = detect_scan_bends(
        scan_points=scan_points,
        scan_detector_kwargs=primary_ctor,
        scan_detect_call_kwargs=primary_call,
    )
    primary_report = match_detected_bends(
        detected_bends=primary_detected,
        cad_bends=cad_bends,
        part_id=part_id,
        matcher_kwargs=runtime_config.matcher_kwargs,
    )
    primary_report.compute_summary()

    target = int(expected_bend_count) if expected_bend_count is not None else len(cad_bends)
    target = max(1, target)
    dense_scan = _dense_scan_requires_local_refinement(
        scan_points=scan_points,
        report=primary_report,
        expected_bend_count=target,
    )

    # Trigger fallback only when recovery potential is meaningful.
    should_try_fallback = (
        len(cad_bends) > 0
        and len(scan_points) >= 2000
        and not dense_scan
        and int(primary_report.detected_count) < max(1, int(np.ceil(0.75 * target)))
    )
    if not should_try_fallback:
        return primary_report, int(len(primary_detected)), False

    fb_ctor, fb_call = _fallback_scan_detection_config(runtime_config)
    fb_ctor, fb_call, _ = _apply_scan_runtime_profile(
        fb_ctor,
        fb_call,
        len(scan_points),
        fallback=True,
    )
    fallback_detected = detect_scan_bends(
        scan_points=scan_points,
        scan_detector_kwargs=fb_ctor,
        scan_detect_call_kwargs=fb_call,
    )
    fallback_report = match_detected_bends(
        detected_bends=fallback_detected,
        cad_bends=cad_bends,
        part_id=part_id,
        matcher_kwargs=runtime_config.matcher_kwargs,
    )
    fallback_report.compute_summary()

    if _report_rank(fallback_report, target) > _report_rank(primary_report, target):
        return fallback_report, int(len(fallback_detected)), True
    return primary_report, int(len(primary_detected)), False


def run_progressive_bend_inspection(
    cad_path: str,
    scan_path: str,
    part_id: str,
    tolerance_angle: float,
    tolerance_radius: float,
    runtime_config: Optional[BendInspectionRuntimeConfig] = None,
    spec_bend_angles_hint: Optional[List[float]] = None,
    spec_schedule_type: Optional[str] = None,
) -> Tuple[BendInspectionReport, BendInspectionRunDetails]:
    """
    Run the full bend-inspection pipeline and return report + diagnostics.
    """
    cfg = runtime_config or BendInspectionRuntimeConfig()
    t0 = perf_counter()
    warnings: List[str] = []
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] start cad={Path(cad_path).name} scan={Path(scan_path).name}"
    )

    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] load_cad_geometry begin")
    cad_vertices, cad_triangles, cad_source = load_cad_geometry(
        cad_path,
        cad_import_deflection=cfg.cad_import_deflection,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] load_cad_geometry end "
        f"source={cad_source} vertices={len(cad_vertices)} triangles={0 if cad_triangles is None else len(cad_triangles)}"
    )
    cad_vertex_count = int(len(cad_vertices))
    cad_triangle_count = int(0 if cad_triangles is None else len(cad_triangles))
    overrides = load_expected_bend_overrides()
    expected_bend_hint = overrides.get(part_id)

    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] extract_cad_bend_specs begin")
    cad_bends, cad_strategy = extract_cad_bend_specs_with_strategy(
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        tolerance_angle=tolerance_angle,
        tolerance_radius=tolerance_radius,
        cad_extractor_kwargs=cfg.cad_extractor_kwargs,
        cad_detector_kwargs=cfg.cad_detector_kwargs,
        cad_detect_call_kwargs=cfg.cad_detect_call_kwargs,
        expected_bend_count_hint=expected_bend_hint,
        spec_bend_angles_hint=spec_bend_angles_hint,
        spec_schedule_type=spec_schedule_type,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] extract_cad_bend_specs end "
        f"strategy={cad_strategy} bends={len(cad_bends)}"
    )
    cad_bend_count_extracted = len(cad_bends)
    if not cad_bends:
        warnings.append("No bends extracted from CAD")
    expected_bend_count = int(overrides.get(part_id, len(cad_bends)))
    if expected_bend_count <= 0:
        expected_bend_count = int(len(cad_bends))
    if cad_strategy != "detector_point_cloud":
        warnings.append(f"CAD bend baseline strategy: {cad_strategy}")
    effective_cad_bends = select_cad_bends_for_expected(cad_bends, expected_bend_count)
    if len(effective_cad_bends) < len(cad_bends):
        warnings.append(
            f"CAD bends reduced by expected baseline for {part_id}: extracted={len(cad_bends)} effective={len(effective_cad_bends)}"
        )
    cad_bends = effective_cad_bends
    cad_points_for_quality = _sample_points(
        cad_vertices,
        max_points=max(5000, int(cfg.scan_quality_kwargs.get("sample_points", 30000))),
        seed=5,
    )
    del cad_vertices
    del cad_triangles
    gc.collect()

    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] load_scan_points begin")
    scan_points = load_scan_points(scan_path)
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] load_scan_points end scan_points={len(scan_points)}"
    )
    primary_profile = _scan_runtime_profile_name(len(scan_points), fallback=False)
    if primary_profile:
        warnings.append(
            f"Large scan runtime profile applied: {primary_profile} ({len(scan_points)} scan points)"
        )
    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] detect_and_match_with_fallback begin")
    report, selected_scan_bend_count, fallback_used = detect_and_match_with_fallback(
        scan_points=scan_points,
        cad_bends=cad_bends,
        part_id=part_id,
        runtime_config=cfg,
        expected_bend_count=expected_bend_count,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] detect_and_match_with_fallback end "
        f"detected={report.detected_count} pass={report.pass_count} fallback_used={fallback_used}"
    )
    if _dense_scan_requires_local_refinement(
        scan_points=scan_points,
        report=report,
        expected_bend_count=expected_bend_count,
    ):
        _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] dense local refinement begin")
        local_report, local_warnings = refine_dense_scan_bends_via_alignment(
            cad_path=cad_path,
            scan_path=scan_path,
            cad_bends=cad_bends,
            part_id=part_id,
            tolerance_angle=tolerance_angle,
            matcher_kwargs=cfg.matcher_kwargs,
            local_refinement_kwargs=cfg.local_refinement_kwargs,
        )
        warnings.extend(local_warnings)
        _emit_phase_log(
            f"run_progressive_bend_inspection[{part_id}] dense local refinement end "
            f"local_report={'yes' if local_report is not None else 'no'}"
        )
        if local_report is not None:
            local_alignment_fitness = getattr(local_report, "local_alignment_fitness", None)
            merged_report = merge_bend_reports(
                report,
                local_report,
                part_id=part_id,
                local_alignment_fitness=local_alignment_fitness,
            )
            current_rank = _report_rank(report, expected_bend_count)
            local_rank = _report_rank(local_report, expected_bend_count)
            merged_rank = _report_rank(merged_report, expected_bend_count)
            if merged_rank > max(current_rank, local_rank):
                report = merged_report
                selected_scan_bend_count = int(merged_report.detected_count)
                warnings.append("Dense-scan local refinement merged with primary global detection")
            elif local_rank > current_rank:
                report = local_report
                selected_scan_bend_count = int(local_report.detected_count)
                warnings.append("Dense-scan local refinement promoted over primary global detection")
    if selected_scan_bend_count == 0:
        warnings.append("No bends detected in scan")
    if fallback_used:
        warnings.append("Applied high-sensitivity scan fallback for under-recovered bends")
        fallback_profile = _scan_runtime_profile_name(len(scan_points), fallback=True)
        if fallback_profile:
            warnings.append(
                f"Fallback scan runtime profile applied: {fallback_profile} ({len(scan_points)} scan points)"
            )

    report.processing_time_ms = (perf_counter() - t0) * 1000.0
    report.compute_summary()

    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] estimate_scan_quality begin")
    scan_quality = estimate_scan_quality(
        scan_points=scan_points,
        cad_points=cad_points_for_quality,
        expected_bends=expected_bend_count,
        completed_bends=report.detected_count,
        passed_bends=report.pass_count,
        quality_kwargs=cfg.scan_quality_kwargs,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] estimate_scan_quality end "
        f"status={scan_quality.get('status')} coverage={scan_quality.get('coverage_pct')}"
    )
    report.scan_quality = scan_quality

    overdetected_count = max(0, int(report.detected_count) - int(expected_bend_count))
    detected_for_progress = min(int(report.detected_count), int(expected_bend_count))
    expected_progress_pct = (
        (detected_for_progress / expected_bend_count) * 100.0
        if expected_bend_count > 0 else 0.0
    )
    if expected_bend_count != cad_bend_count_extracted:
        warnings.append(
            f"Expected bend override applied for {part_id}: CAD={cad_bend_count_extracted} expected={expected_bend_count}"
        )
    if overdetected_count > 0:
        warnings.append(
            f"Over-detected bends vs expected baseline: +{overdetected_count}"
        )
    if scan_quality.get("status") == "POOR":
        warnings.append(
            "Scan quality is POOR (coverage/density below target); capture additional or denser scans before release."
        )
    elif scan_quality.get("status") == "FAIR":
        warnings.append(
            "Scan quality is FAIR; one additional sweep can improve confidence."
        )

    details = BendInspectionRunDetails(
        cad_source=cad_source,
        cad_strategy=cad_strategy,
        cad_vertex_count=cad_vertex_count,
        cad_triangle_count=cad_triangle_count,
        scan_point_count=int(len(scan_points)),
        cad_bend_count=int(cad_bend_count_extracted),
        scan_bend_count=int(selected_scan_bend_count),
        expected_bend_count=expected_bend_count,
        expected_progress_pct=expected_progress_pct,
        overdetected_count=overdetected_count,
        warnings=warnings,
        scan_quality_status=str(scan_quality.get("status", "UNKNOWN")),
        scan_coverage_pct=float(scan_quality.get("coverage_pct", 0.0)),
        scan_density_pts_per_cm2=float(scan_quality.get("density_pts_per_cm2", 0.0)),
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] complete "
        f"detected={report.detected_count} expected={expected_bend_count} "
        f"scan_quality={details.scan_quality_status} elapsed_ms={report.processing_time_ms:.1f}"
    )
    return report, details
