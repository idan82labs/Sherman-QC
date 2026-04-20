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
from domains.bend.services.runtime_semantics import normalize_observability_state
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
    scan_state: str = "unknown"
    scan_quality_status: str = "UNKNOWN"
    scan_coverage_pct: float = 0.0
    scan_density_pts_per_cm2: float = 0.0
    dense_local_refinement_decision: Optional[str] = None
    dense_local_refinement_report_ranks: Optional[Dict[str, List[int]]] = None
    dense_local_refinement_geometry_penalties: Optional[Dict[str, Optional[float]]] = None
    local_alignment_seed: Optional[int] = None
    local_alignment_fitness: Optional[float] = None
    local_alignment_rmse: Optional[float] = None
    # CAD geometry from the pipeline run (same vertices used for bend detection).
    # Used to export the reference mesh PLY in the SAME coordinate system as
    # the bend line coordinates, avoiding dual-import coordinate mismatches.
    cad_vertices: Optional[np.ndarray] = field(default=None, repr=False)
    cad_triangles: Optional[np.ndarray] = field(default=None, repr=False)

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
            "scan_state": self.scan_state,
            "scan_quality_status": self.scan_quality_status,
            "scan_coverage_pct": round(self.scan_coverage_pct, 2),
            "scan_density_pts_per_cm2": round(self.scan_density_pts_per_cm2, 2),
            "dense_local_refinement_decision": self.dense_local_refinement_decision,
            "dense_local_refinement_report_ranks": self.dense_local_refinement_report_ranks,
            "dense_local_refinement_geometry_penalties": self.dense_local_refinement_geometry_penalties,
            "local_alignment_seed": self.local_alignment_seed,
            "local_alignment_fitness": (
                round(float(self.local_alignment_fitness), 4)
                if self.local_alignment_fitness is not None
                else None
            ),
            "local_alignment_rmse": (
                round(float(self.local_alignment_rmse), 4)
                if self.local_alignment_rmse is not None
                else None
            ),
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
    out: Dict[str, int] = {}
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
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse expected bend overrides %s: %s", p, exc)
            raw = None
        if isinstance(raw, list):
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

    seed_labels_path = Path(__file__).resolve().parent.parent / "data" / "bend_corpus_seed_labels.json"
    if seed_labels_path.exists():
        try:
            seed_raw = json.loads(seed_labels_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse bend corpus seed labels %s: %s", seed_labels_path, exc)
            seed_raw = None
        if isinstance(seed_raw, dict):
            for part, payload in seed_raw.items():
                if not isinstance(payload, dict):
                    continue
                expected = payload.get("expected_total_bends", None)
                if expected is None:
                    continue
                try:
                    expected_int = int(expected)
                except Exception:
                    continue
                if expected_int > 0 and str(part).strip() not in out:
                    out[str(part).strip()] = expected_int
    return out


def load_part_feature_policies(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load optional part-specific feature-counting policies from the bend corpus seed labels.
    """
    resolved = path or os.environ.get("BEND_FEATURE_POLICIES", "").strip()
    if not resolved:
        resolved = str(
            Path(__file__).resolve().parent.parent / "data" / "bend_corpus_seed_labels.json"
        )
    p = Path(resolved)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse bend feature policies %s: %s", p, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    policies: Dict[str, Dict[str, Any]] = {}
    for part_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        policy = payload.get("feature_policy")
        if isinstance(policy, dict) and policy:
            policies[str(part_id)] = dict(policy)
    return policies


def _apply_part_feature_policy(
    cad_bends: List[BendSpecification],
    feature_policy: Optional[Dict[str, Any]],
) -> List[BendSpecification]:
    if not cad_bends or not feature_policy:
        return list(cad_bends)

    non_counting_forms = {
        str(value).upper()
        for value in (feature_policy.get("non_counting_feature_forms") or [])
        if str(value).strip()
    }
    countable_forms = {
        str(value).upper()
        for value in (feature_policy.get("countable_feature_forms") or [])
        if str(value).strip()
    }
    non_counting_bend_ids = {
        str(value)
        for value in (feature_policy.get("non_counting_bend_ids") or [])
        if str(value).strip()
    }
    countable_bend_ids = {
        str(value)
        for value in (feature_policy.get("countable_bend_ids") or [])
        if str(value).strip()
    }
    rolled_feature_type = str(feature_policy.get("rolled_feature_type") or "ROLLED_SECTION").upper()
    countable_bend_angle_overrides = {
        str(key): float(value)
        for key, value in (feature_policy.get("countable_bend_angle_overrides") or {}).items()
        if str(key).strip() and value is not None
    }

    out: List[BendSpecification] = []
    for bend in cad_bends:
        updated = copy.deepcopy(bend)
        bend_form = str(updated.bend_form or "").upper()
        if updated.bend_id in non_counting_bend_ids or (bend_form and bend_form in non_counting_forms):
            updated.feature_type = rolled_feature_type if bend_form == "ROLLED" else "PROCESS_FEATURE"
            updated.countable_in_regression = False
        elif updated.bend_id in countable_bend_ids or (bend_form and bend_form in countable_forms):
            updated.feature_type = "DISCRETE_BEND"
            updated.countable_in_regression = True
        if updated.bend_id in countable_bend_angle_overrides:
            updated.target_angle = float(countable_bend_angle_overrides[updated.bend_id])
        out.append(updated)
    return out


def _augment_feature_policy_from_part_truth(
    feature_policy: Optional[Dict[str, Any]],
    part_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not feature_policy:
        return feature_policy
    resolved_part_id = str(part_id or "").strip()
    if not resolved_part_id:
        return feature_policy

    metadata_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "bend_corpus"
        / "parts"
        / resolved_part_id
        / "metadata.json"
    )
    if not metadata_path.exists():
        return feature_policy
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return feature_policy
    user_truth = metadata.get("user_truth") or {}
    effective_policy = dict(feature_policy)
    part_family = str(user_truth.get("part_family") or "").strip()
    if part_family and not effective_policy.get("part_family"):
        effective_policy["part_family"] = part_family
    raw_overrides = user_truth.get("countable_bend_target_angles_deg") or {}
    if not isinstance(raw_overrides, dict) or not raw_overrides:
        return effective_policy
    if not effective_policy.get("countable_bend_angle_overrides"):
        effective_policy["countable_bend_angle_overrides"] = {
            str(key): float(value)
            for key, value in raw_overrides.items()
            if str(key).strip() and value is not None
        }
    return effective_policy


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


def classify_scan_geometry_kind(scan_path: str) -> str:
    """
    Classify the scan representation used by this pipeline.

    STL/OBJ scans are currently consumed as raw mesh vertices, not as measured
    point clouds. They should not inherit the aggressive large-scan point-cloud
    runtime profile, which can collapse stable CAD-equivalent geometry into a
    smaller set of incorrect bend detections.
    """
    ext = Path(scan_path).suffix.lower()
    if ext in {".stl", ".obj"}:
        return "mesh_vertices"
    return "point_cloud"


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


def export_reference_mesh_from_vertices(
    vertices: np.ndarray,
    triangles: Optional[np.ndarray],
    output_dir: str | Path,
) -> Dict[str, str]:
    """
    Export reference mesh PLY from pre-loaded CAD vertices/triangles.

    This avoids re-importing the CAD file, guaranteeing that the reference
    mesh PLY is in the SAME coordinate system as the bend detection coordinates.
    """
    import trimesh

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    extractor_overrides: Optional[Dict[str, Any]] = None,
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

    extractor_kwargs = {
        "coplanar_threshold": 15.0,
        "min_area_pct": 0.2,
        "max_bend_gap": 20.0,
        "min_bend_angle": 8.0,
        "dedup_distance": 8.0,
    }
    if extractor_overrides:
        extractor_kwargs.update({k: v for k, v in extractor_overrides.items() if v is not None})

    extractor = LegacyMeshCADBendExtractor(**extractor_kwargs)
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


def _recover_legacy_mesh_cad_specs_for_expected(
    cad_vertices: np.ndarray,
    cad_triangles: np.ndarray,
    tolerance_angle: float,
    tolerance_radius: float,
    expected_bend_count_hint: int,
    baseline_specs: List[BendSpecification],
) -> Tuple[Optional[List[BendSpecification]], Optional[str]]:
    """
    Try a small set of more permissive legacy mesh extractor settings when the
    default mesh baseline clearly undercounts against known truth.

    This is intentionally narrow. It only runs for severe undercount cases and
    only promotes a recovered baseline when it materially improves count fit.
    """
    target = int(expected_bend_count_hint or 0)
    baseline_count = len(baseline_specs)
    if target <= 0 or baseline_count >= target or baseline_count > max(12, target - 4):
        return None, None

    recovery_configs = [
        {"min_area_pct": 0.15, "max_bend_gap": 20.0, "dedup_distance": 6.0},
        {"min_area_pct": 0.15, "max_bend_gap": 30.0, "dedup_distance": 10.0},
        {"min_area_pct": 0.15, "max_bend_gap": 45.0, "dedup_distance": 15.0},
    ]

    trials: List[Tuple[Tuple[float, int, int], List[BendSpecification], Dict[str, Any]]] = []
    for config in recovery_configs:
        try:
            specs = _build_legacy_mesh_cad_specs(
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                tolerance_angle=tolerance_angle,
                tolerance_radius=tolerance_radius,
                extractor_overrides=config,
            )
        except Exception:
            continue
        if not specs:
            continue
        count = len(specs)
        score = (
            abs(count - target),
            1 if count < target else 0,
            abs(count - baseline_count),
        )
        trials.append((score, specs, config))

    if not trials:
        return None, None

    trials.sort(key=lambda item: item[0])
    best_score, best_specs, best_config = trials[0]
    baseline_err = abs(baseline_count - target)
    best_err = abs(len(best_specs) - target)
    if best_err + 1 > baseline_err:
        return None, None

    logger.info(
        "Legacy mesh expected recovery selected: baseline=%s target=%s recovered=%s config=%s",
        baseline_count,
        target,
        len(best_specs),
        best_config,
    )
    return best_specs, "legacy_mesh_expected_recovery"


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


def _bend_midpoint(spec: BendSpecification) -> np.ndarray:
    return 0.5 * (np.asarray(spec.bend_line_start, dtype=np.float64) + np.asarray(spec.bend_line_end, dtype=np.float64))


def _bend_direction_alignment(a: BendSpecification, b: BendSpecification) -> float:
    da = np.asarray(a.bend_line_direction, dtype=np.float64)
    db = np.asarray(b.bend_line_direction, dtype=np.float64)
    return float(abs(np.dot(da, db)))


def _renumber_cad_specs(specs: List[BendSpecification]) -> List[BendSpecification]:
    renumbered: List[BendSpecification] = []
    for idx, spec in enumerate(specs, start=1):
        clone = copy.deepcopy(spec)
        clone.bend_id = f"CAD_B{idx}"
        renumbered.append(clone)
    return renumbered


def _supplement_detector_with_legacy_specs(
    detector_specs: List[BendSpecification],
    legacy_specs: List[BendSpecification],
    expected_bend_count_hint: int,
) -> List[BendSpecification]:
    """
    Recover compact spatially-distinct bends from the legacy mesh path when the
    detector baseline clearly undercounts against confirmed truth.

    This is intentionally conservative: it only supplements the detector
    baseline up to the expected count, never replaces it wholesale.
    """
    target = int(expected_bend_count_hint)
    if target <= len(detector_specs) or not legacy_specs:
        return list(detector_specs)

    selected = [copy.deepcopy(spec) for spec in detector_specs]
    selected_midpoints = [_bend_midpoint(spec) for spec in selected]

    def _nearest_selected_distance(spec: BendSpecification) -> float:
        midpoint = _bend_midpoint(spec)
        if not selected_midpoints:
            return float("inf")
        return float(min(np.linalg.norm(midpoint - other) for other in selected_midpoints))

    candidates: List[BendSpecification] = []
    for spec in legacy_specs:
        if _nearest_selected_distance(spec) < 28.0:
            continue
        if spec.bend_line_length < 14.0:
            continue
        candidates.append(copy.deepcopy(spec))

    candidates.sort(
        key=lambda spec: (
            min(float(spec.flange1_area), float(spec.flange2_area)),
            float(spec.bend_line_length),
            _nearest_selected_distance(spec),
            -abs(float(spec.target_angle) - 90.0),
        ),
        reverse=True,
    )

    for candidate in candidates:
        midpoint = _bend_midpoint(candidate)
        too_close = False
        for existing in selected:
            dist = float(np.linalg.norm(midpoint - _bend_midpoint(existing)))
            if dist < 34.0 and _bend_direction_alignment(candidate, existing) >= 0.97:
                too_close = True
                break
        if too_close:
            continue
        selected.append(copy.deepcopy(candidate))
        selected_midpoints.append(midpoint)
        if len(selected) >= target:
            break

    return _renumber_cad_specs(selected)


def _build_feature_policy_hybrid_specs(
    detector_specs: List[BendSpecification],
    legacy_specs: List[BendSpecification],
    expected_bend_count_hint: Optional[int],
    feature_policy: Optional[Dict[str, Any]],
    spec_bend_angles_hint: Optional[List[float]] = None,
) -> Tuple[Optional[List[BendSpecification]], Optional[str]]:
    """
    Build a hybrid CAD baseline for special parts where rolled shell features
    should be preserved as process features while countable bends come from a
    small folded legacy subset.
    """
    if not feature_policy:
        return None, None
    if str(feature_policy.get("countable_source") or "").strip().lower() != "legacy_folded_hybrid":
        return None, None
    target = int(expected_bend_count_hint or 0)
    if target <= 0 or not detector_specs or not legacy_specs:
        return None, None

    non_counting_forms = {
        str(value).upper()
        for value in (feature_policy.get("non_counting_feature_forms") or [])
        if str(value).strip()
    }
    if not non_counting_forms:
        return None, None

    process_specs = [
        copy.deepcopy(spec)
        for spec in detector_specs
        if str(spec.bend_form or "").upper() in non_counting_forms
    ]
    if not process_specs:
        return None, None
    rolled_feature_type = str(feature_policy.get("rolled_feature_type") or "ROLLED_SECTION").upper()
    for spec in process_specs:
        spec.feature_type = rolled_feature_type if str(spec.bend_form or "").upper() == "ROLLED" else "PROCESS_FEATURE"
        spec.countable_in_regression = False

    countable_candidates = [
        copy.deepcopy(spec)
        for spec in legacy_specs
        if str(spec.bend_form or "").upper() not in non_counting_forms and float(spec.bend_line_length) >= 10.0
    ]
    if not countable_candidates:
        return process_specs, "feature_policy_process_only"

    hint_angles = [float(v) for v in (spec_bend_angles_hint or []) if isinstance(v, (int, float))]
    part_family = str(feature_policy.get("part_family") or "").strip().lower()

    process_direction_hint: Optional[np.ndarray] = None
    process_direction_weights = [
        max(float(spec.bend_line_length), 1.0)
        for spec in process_specs
    ]
    if process_specs:
        ref = np.asarray(process_specs[0].bend_line_direction, dtype=np.float64)
        acc = np.zeros(3, dtype=np.float64)
        for spec, weight in zip(process_specs, process_direction_weights):
            direction = np.asarray(spec.bend_line_direction, dtype=np.float64)
            if float(np.dot(direction, ref)) < 0.0:
                direction = -direction
            acc += direction * float(weight)
        acc_norm = float(np.linalg.norm(acc))
        if acc_norm > 1e-9:
            process_direction_hint = acc / acc_norm

    process_profile_center: Optional[np.ndarray] = None
    if process_direction_hint is not None and process_specs:
        profile_points = []
        for spec in process_specs:
            midpoint = _bend_midpoint(spec)
            profile_point = midpoint - process_direction_hint * float(np.dot(midpoint, process_direction_hint))
            profile_points.append(profile_point)
        if profile_points:
            process_profile_center = np.mean(np.asarray(profile_points, dtype=np.float64), axis=0)

    def _candidate_rank(spec: BendSpecification) -> Tuple[float, float, float, float, float]:
        min_area = min(float(spec.flange1_area), float(spec.flange2_area))
        max_area = max(float(spec.flange1_area), float(spec.flange2_area))
        if hint_angles:
            angle_gap = round(min(_angle_hint_distance(hint, float(spec.target_angle)) for hint in hint_angles), 3)
        else:
            angle_gap = round(abs(abs(float(spec.target_angle)) - 90.0), 3)
        process_alignment_penalty = 0.0
        if process_direction_hint is not None:
            alignment = abs(float(np.dot(np.asarray(spec.bend_line_direction, dtype=np.float64), process_direction_hint)))
            process_alignment_penalty = round(1.0 - alignment, 4)
        return (
            process_alignment_penalty,
            angle_gap,
            min_area,
            -float(spec.bend_line_length),
            -max_area,
        )

    if (
        part_family == "rolled_shell_with_two_flange_bends"
        and target == 2
        and len(countable_candidates) >= 2
    ):
        upper_hint = None
        lower_hint = None
        if hint_angles:
            upper_hint = float(max(hint_angles))
            lower_hint = float(min(hint_angles))

        top_surface_candidates = [
            spec
            for spec in countable_candidates
            if max(
                abs(float(np.asarray(spec.flange1_normal, dtype=np.float64)[2])) if spec.flange1_normal is not None else 0.0,
                abs(float(np.asarray(spec.flange2_normal, dtype=np.float64)[2])) if spec.flange2_normal is not None else 0.0,
            ) >= 0.5
        ]
        max_top_y = max((_bend_midpoint(spec)[1] for spec in top_surface_candidates), default=None)
        max_top_z = max((_bend_midpoint(spec)[2] for spec in top_surface_candidates), default=None)

        def _top_surface_strength(spec: BendSpecification) -> float:
            return max(
                abs(float(np.asarray(spec.flange1_normal, dtype=np.float64)[2])) if spec.flange1_normal is not None else 0.0,
                abs(float(np.asarray(spec.flange2_normal, dtype=np.float64)[2])) if spec.flange2_normal is not None else 0.0,
            )

        def _short_upper_rank(spec: BendSpecification) -> Tuple[float, float, float, float, float, float]:
            midpoint = _bend_midpoint(spec)
            top_strength = _top_surface_strength(spec)
            angle_gap = (
                min(_angle_hint_distance(upper_hint, float(spec.target_angle)), _angle_hint_distance(lower_hint, float(spec.target_angle)))
                if upper_hint is not None and lower_hint is not None
                else (min(_angle_hint_distance(h, float(spec.target_angle)) for h in hint_angles) if hint_angles else 0.0)
            )
            y_gap = abs(float(midpoint[1]) - float(max_top_y)) if max_top_y is not None else 0.0
            z_gap = abs(float(midpoint[2]) - float(max_top_z)) if max_top_z is not None else 0.0
            return (
                0.0 if top_strength >= 0.5 else 1.0,
                0.0 if float(spec.bend_line_length) <= 70.0 else 1.0,
                round(y_gap, 3),
                round(z_gap, 3),
                round(abs(float(spec.bend_line_length) - 40.0), 3),
                round(angle_gap, 3),
            )

        upper_candidates = sorted(countable_candidates, key=_short_upper_rank)
        selected_upper: Optional[BendSpecification] = copy.deepcopy(upper_candidates[0]) if upper_candidates else None

        selected_lower: Optional[BendSpecification] = None
        if selected_upper is not None:
            upper_midpoint = _bend_midpoint(selected_upper)
            upper_z = float(upper_midpoint[2])

            def _lower_flange_rank(spec: BendSpecification) -> Tuple[float, float, float, float, float, float, float]:
                midpoint = _bend_midpoint(spec)
                top_strength = _top_surface_strength(spec)
                angle_gap = (
                    _angle_hint_distance(lower_hint, float(spec.target_angle))
                    if lower_hint is not None
                    else (min(_angle_hint_distance(h, float(spec.target_angle)) for h in hint_angles) if hint_angles else 0.0)
                )
                axis_alignment_penalty = 0.0
                if process_direction_hint is not None:
                    axis_alignment_penalty = 1.0 - abs(float(np.dot(np.asarray(spec.bend_line_direction, dtype=np.float64), process_direction_hint)))
                y_penalty = 0.0 if float(midpoint[1]) <= float(upper_midpoint[1]) - 2.0 else abs(float(midpoint[1]) - float(upper_midpoint[1]))
                z_gap = abs(float(midpoint[2]) - upper_z)
                return (
                    0.0 if float(spec.bend_line_length) >= 80.0 else 1.0,
                    round(angle_gap, 3),
                    0.0 if top_strength < 0.5 else 1.0,
                    round(z_gap, 3),
                    round(y_penalty, 3),
                    round(axis_alignment_penalty, 4),
                    -round(max(float(spec.flange1_area), float(spec.flange2_area)), 3),
                )

            remaining = [
                spec
                for spec in countable_candidates
                if spec.bend_id != selected_upper.bend_id
            ]
            if remaining:
                selected_lower = copy.deepcopy(sorted(remaining, key=_lower_flange_rank)[0])

        if selected_upper is not None and selected_lower is not None:
            selected_upper.feature_type = "DISCRETE_BEND"
            selected_upper.countable_in_regression = True
            selected_lower.feature_type = "DISCRETE_BEND"
            selected_lower.countable_in_regression = True
            return _renumber_cad_specs(process_specs + [selected_upper, selected_lower]), "feature_policy_rolled_hybrid_flange_pair"

    if target == 2 and process_direction_hint is not None and len(countable_candidates) >= 2:
        best_pair: Optional[Tuple[BendSpecification, BendSpecification]] = None
        best_pair_score: Optional[Tuple[float, float, float, float, float]] = None
        for i, candidate_a in enumerate(countable_candidates):
            direction_a = np.asarray(candidate_a.bend_line_direction, dtype=np.float64)
            alignment_a = abs(float(np.dot(direction_a, process_direction_hint)))
            if alignment_a < 0.9:
                continue
            midpoint_a = _bend_midpoint(candidate_a)
            max_area_a = max(float(candidate_a.flange1_area), float(candidate_a.flange2_area))
            profile_a = midpoint_a - process_direction_hint * float(np.dot(midpoint_a, process_direction_hint))
            radial_a = (
                float(np.linalg.norm(profile_a - process_profile_center))
                if process_profile_center is not None
                else 0.0
            )
            if hint_angles:
                angle_gap_a = round(min(_angle_hint_distance(hint, float(candidate_a.target_angle)) for hint in hint_angles), 3)
            else:
                angle_gap_a = round(abs(abs(float(candidate_a.target_angle)) - 90.0), 3)
            for candidate_b in countable_candidates[i + 1:]:
                direction_b = np.asarray(candidate_b.bend_line_direction, dtype=np.float64)
                alignment_b = abs(float(np.dot(direction_b, process_direction_hint)))
                if alignment_b < 0.9:
                    continue
                midpoint_b = _bend_midpoint(candidate_b)
                max_area_b = max(float(candidate_b.flange1_area), float(candidate_b.flange2_area))
                profile_b = midpoint_b - process_direction_hint * float(np.dot(midpoint_b, process_direction_hint))
                radial_b = (
                    float(np.linalg.norm(profile_b - process_profile_center))
                    if process_profile_center is not None
                    else 0.0
                )
                if hint_angles:
                    angle_gap_b = round(min(_angle_hint_distance(hint, float(candidate_b.target_angle)) for hint in hint_angles), 3)
                else:
                    angle_gap_b = round(abs(abs(float(candidate_b.target_angle)) - 90.0), 3)
                has_top_surface = int(
                    max(abs(float(np.asarray(candidate_a.flange1_normal, dtype=np.float64)[2])), abs(float(np.asarray(candidate_a.flange2_normal, dtype=np.float64)[2])), abs(float(np.asarray(candidate_b.flange1_normal, dtype=np.float64)[2])), abs(float(np.asarray(candidate_b.flange2_normal, dtype=np.float64)[2]))) >= 0.5
                )
                pair_score = (
                    -round(0.5 * (radial_a + radial_b), 3),
                    0 if has_top_surface else 1,
                    round(angle_gap_a + angle_gap_b, 3),
                    round(abs(max_area_a - max_area_b), 3),
                    round(float(np.linalg.norm(profile_a - profile_b)), 3),
                    round(max(max_area_a, max_area_b), 3),
                    -round(float(candidate_a.bend_line_length + candidate_b.bend_line_length), 3),
                )
                if best_pair_score is None or pair_score < best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (copy.deepcopy(candidate_a), copy.deepcopy(candidate_b))
        if best_pair is not None:
            selected_countable = []
            for candidate in best_pair:
                candidate.feature_type = "DISCRETE_BEND"
                candidate.countable_in_regression = True
                selected_countable.append(candidate)
            return _renumber_cad_specs(process_specs + selected_countable), "feature_policy_rolled_hybrid"

    selected_countable: List[BendSpecification] = []
    for candidate in sorted(countable_candidates, key=_candidate_rank):
        midpoint = _bend_midpoint(candidate)
        duplicate = False
        for existing in selected_countable:
            dist = float(np.linalg.norm(midpoint - _bend_midpoint(existing)))
            if dist < 12.0 and _bend_direction_alignment(candidate, existing) >= 0.97:
                duplicate = True
                break
        if duplicate:
            continue
        candidate.feature_type = "DISCRETE_BEND"
        candidate.countable_in_regression = True
        selected_countable.append(candidate)
        if len(selected_countable) >= target:
            break

    if not selected_countable:
        return process_specs, "feature_policy_process_only"

    return _renumber_cad_specs(process_specs + selected_countable), "feature_policy_rolled_hybrid"


def _choose_mesh_cad_baseline_specs(
    detector_specs: List[BendSpecification],
    legacy_specs: List[BendSpecification],
    expected_bend_count_hint: Optional[int] = None,
    spec_bend_angles_hint: Optional[List[float]] = None,
    spec_schedule_type: Optional[str] = None,
    feature_policy: Optional[Dict[str, Any]] = None,
    part_id: Optional[str] = None,
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

    feature_policy = _augment_feature_policy_from_part_truth(feature_policy, part_id=part_id)

    hybrid_specs, hybrid_strategy = _build_feature_policy_hybrid_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=expected_bend_count_hint,
        feature_policy=feature_policy,
        spec_bend_angles_hint=spec_bend_angles_hint,
    )
    if hybrid_specs is not None and hybrid_strategy is not None:
        return hybrid_specs, hybrid_strategy

    if expected_bend_count_hint is not None and expected_bend_count_hint > 0:
        detector_err = abs(len(detector_specs) - int(expected_bend_count_hint))
        legacy_err = abs(len(legacy_specs) - int(expected_bend_count_hint))
        if legacy_err + 1 < detector_err:
            return legacy_specs, "legacy_mesh_expected_hint"
        if (
            len(detector_specs) < int(expected_bend_count_hint)
            and len(legacy_specs) >= int(expected_bend_count_hint)
            and _rolled_profile_guard(detector_specs, legacy_specs, spec_bend_angles_hint)
        ):
            supplemented = _supplement_detector_with_legacy_specs(
                detector_specs,
                legacy_specs,
                int(expected_bend_count_hint),
            )
            if len(supplemented) >= min(int(expected_bend_count_hint), len(detector_specs) + 1):
                return supplemented, "detector_mesh_expected_supplement"
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
    feature_policy: Optional[Dict[str, Any]] = None,
    part_id: Optional[str] = None,
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

    selected_specs, strategy = _choose_mesh_cad_baseline_specs(
        detector_specs=detector_specs,
        legacy_specs=legacy_specs,
        expected_bend_count_hint=expected_bend_count_hint,
        spec_bend_angles_hint=spec_bend_angles_hint,
        spec_schedule_type=spec_schedule_type,
        feature_policy=feature_policy,
        part_id=part_id,
    )
    if expected_bend_count_hint is not None and expected_bend_count_hint > 0:
        recovered_specs, recovery_strategy = _recover_legacy_mesh_cad_specs_for_expected(
            cad_vertices=cad_vertices,
            cad_triangles=cad_triangles,
            tolerance_angle=tolerance_angle,
            tolerance_radius=tolerance_radius,
            expected_bend_count_hint=int(expected_bend_count_hint),
            baseline_specs=selected_specs,
        )
        if recovered_specs is not None and recovery_strategy is not None:
            return recovered_specs, recovery_strategy

    return selected_specs, strategy


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
    if expected_bend_count <= 0:
        return list(cad_bends)

    non_counting_specs = [
        copy.deepcopy(bend)
        for bend in cad_bends
        if not bool(getattr(bend, "countable_in_regression", True))
    ]
    countable_with_index = [
        (idx, bend)
        for idx, bend in enumerate(cad_bends)
        if bool(getattr(bend, "countable_in_regression", True))
    ]
    if not countable_with_index:
        return list(cad_bends)
    if len(countable_with_index) <= expected_bend_count:
        return list(cad_bends)

    ranked = sorted(
        countable_with_index,
        key=lambda item: (
            float(item[1].flange1_area + item[1].flange2_area),
            float(item[1].bend_line_length),
            float(abs(item[1].target_angle)),
        ),
        reverse=True,
    )[:expected_bend_count]

    keep_indices = {idx for idx, _ in ranked}
    selected: List[BendSpecification] = []
    for idx, bend in enumerate(cad_bends):
        if not bool(getattr(bend, "countable_in_regression", True)) or idx in keep_indices:
            selected.append(copy.deepcopy(bend))
    return _renumber_cad_specs(selected)


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


def _scan_runtime_profile_name_for_state(
    point_count: int,
    scan_state: Optional[str],
    *,
    fallback: bool = False,
) -> Optional[str]:
    state = str(scan_state or "").strip().lower()
    if state == "partial":
        if point_count >= 650000:
            return "partial_scan_dense_fallback" if fallback else "partial_scan_dense_primary"
        if point_count >= 300000:
            return "partial_scan_medium_fallback" if fallback else "partial_scan_medium_primary"
        if point_count >= 150000:
            return "partial_scan_light_fallback" if fallback else "partial_scan_light_primary"
    return _scan_runtime_profile_name(point_count, fallback=fallback)


def _apply_scan_runtime_profile(
    scan_detector_kwargs: Optional[Dict[str, Any]],
    scan_detect_call_kwargs: Optional[Dict[str, Any]],
    point_count: int,
    *,
    scan_state: Optional[str] = None,
    fallback: bool = False,
    scan_geometry_kind: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    """
    Adapt runtime detection settings for large full-resolution scans.

    We still process the original full scan, but we raise the preprocessing
    voxel size and disable expensive multiscale fallback when point counts are
    high enough to destabilize Open3D on memory-limited development laptops.
    """
    ctor = copy.deepcopy(scan_detector_kwargs or {})
    call = copy.deepcopy(scan_detect_call_kwargs or {})
    if str(scan_geometry_kind or "").strip().lower() == "mesh_vertices":
        return ctor, call, None
    profile = _scan_runtime_profile_name_for_state(point_count, scan_state, fallback=fallback)
    if profile is None:
        return ctor, call, None

    call["preprocess"] = True
    state = str(scan_state or "").strip().lower()

    if state == "partial" and point_count >= 650000:
        ctor["min_plane_points"] = max(70, int(ctor.get("min_plane_points", 50)))
        ctor["min_plane_area"] = max(40.0, float(ctor.get("min_plane_area", 30.0)))
        call["voxel_size"] = max(2.15 if fallback else 2.6, float(call.get("voxel_size", 0.3)))
        call["outlier_nb_neighbors"] = min(max(8, int(call.get("outlier_nb_neighbors", 20))), 10)
        call["outlier_std_ratio"] = max(2.7, float(call.get("outlier_std_ratio", 2.0)))
        call["multiscale"] = False
        return ctor, call, profile

    if state == "partial" and point_count >= 300000:
        if fallback:
            # Keep medium partial fallback meaningfully higher-recall than the
            # primary pass. The old profile raised the fallback back to the
            # same coarse geometry thresholds as primary, which erased recall
            # gains on short stage bends. Preserve the lighter geometry gates
            # but keep preprocessing cheap enough for corpus iteration.
            ctor["min_plane_points"] = max(36, int(ctor.get("min_plane_points", 50)))
            ctor["min_plane_area"] = max(18.0, float(ctor.get("min_plane_area", 30.0)))
            call["voxel_size"] = max(1.7, min(2.0, float(call.get("voxel_size", 2.0))))
            call["outlier_nb_neighbors"] = min(max(8, int(call.get("outlier_nb_neighbors", 20))), 10)
            call["outlier_std_ratio"] = max(2.45, float(call.get("outlier_std_ratio", 2.0)))
            call["multiscale"] = False
            return ctor, call, profile
        ctor["min_plane_points"] = max(68, int(ctor.get("min_plane_points", 50)))
        ctor["min_plane_area"] = max(38.0, float(ctor.get("min_plane_area", 30.0)))
        call["voxel_size"] = max(2.45, float(call.get("voxel_size", 0.3)))
        call["outlier_nb_neighbors"] = min(max(8, int(call.get("outlier_nb_neighbors", 20))), 10)
        call["outlier_std_ratio"] = max(2.6, float(call.get("outlier_std_ratio", 2.0)))
        call["multiscale"] = False
        return ctor, call, profile

    if state == "partial" and point_count >= 150000:
        if fallback:
            ctor["min_plane_points"] = max(34, int(ctor.get("min_plane_points", 50)))
            ctor["min_plane_area"] = max(16.0, float(ctor.get("min_plane_area", 30.0)))
            call["voxel_size"] = max(1.55, min(1.85, float(call.get("voxel_size", 1.85))))
            call["outlier_nb_neighbors"] = min(max(8, int(call.get("outlier_nb_neighbors", 20))), 10)
            call["outlier_std_ratio"] = max(2.35, float(call.get("outlier_std_ratio", 2.0)))
            call["multiscale"] = False
            return ctor, call, profile
        ctor["min_plane_points"] = max(62, int(ctor.get("min_plane_points", 50)))
        ctor["min_plane_area"] = max(35.0, float(ctor.get("min_plane_area", 30.0)))
        call["voxel_size"] = max(2.25, float(call.get("voxel_size", 0.3)))
        call["outlier_nb_neighbors"] = min(max(8, int(call.get("outlier_nb_neighbors", 20))), 10)
        call["outlier_std_ratio"] = max(2.5, float(call.get("outlier_std_ratio", 2.0)))
        call["multiscale"] = False
        return ctor, call, profile

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


def _report_short_obtuse_geometry_penalty(
    report: BendInspectionReport,
) -> float:
    """
    Secondary tie-breaker for short obtuse countable bends.

    Lower is better. This is only used when aggregate pass/fail ranking ties,
    so it should reward physically plausible local geometry without changing the
    overall benchmark objective hierarchy.
    """
    penalties: List[float] = []
    for match in report.matches:
        cad_bend = getattr(match, "cad_bend", None)
        if cad_bend is None:
            continue
        if not bool(getattr(cad_bend, "countable_in_regression", True)):
            continue
        target_angle = abs(float(getattr(cad_bend, "target_angle", 0.0) or 0.0))
        if not (105.0 <= target_angle <= 140.0):
            continue
        line_length = float(
            np.linalg.norm(
                np.asarray(cad_bend.bend_line_end, dtype=np.float64)
                - np.asarray(cad_bend.bend_line_start, dtype=np.float64)
            )
        )
        if line_length > 70.0:
            continue
        if str(getattr(match, "status", "")).upper() == "NOT_DETECTED":
            penalties.append(1000.0)
            continue

        line_dev = abs(float(getattr(match, "line_length_deviation_mm", 0.0) or 0.0))
        angle_dev = abs(float(getattr(match, "angle_deviation", 0.0) or 0.0))
        penalties.append(line_dev + (2.0 * angle_dev))

    if not penalties:
        return float("inf")
    return float(sum(penalties))


def _report_decision_grade_rank(
    report: BendInspectionReport,
    expected_bend_count: int,
) -> Tuple[int, int, int, int]:
    """
    Rank only bend claims that are actually decision-ready under the current
    correspondence-gated semantics.

    This prevents a local report from outranking the primary report purely by
    converting additional bends into ambiguous PASS/FAIL/WARNING rows.
    """
    target = max(1, int(expected_bend_count))
    countable_matches = [
        match
        for match in report.matches
        if bool(getattr(getattr(match, "cad_bend", None), "countable_in_regression", True))
    ]
    decision_ready = [
        match
        for match in countable_matches
        if bool(match.completion_claim_eligible() or match.metrology_claim_eligible() or match.position_claim_eligible())
    ]
    ready_capped = min(len(decision_ready), target)
    return (
        ready_capped,
        sum(1 for match in decision_ready if str(match.status).upper() == "PASS"),
        -sum(1 for match in decision_ready if str(match.status).upper() == "FAIL"),
        -sum(1 for match in decision_ready if str(match.status).upper() == "WARNING"),
    )


def _should_promote_dense_local_report(
    current_report: BendInspectionReport,
    candidate_report: BendInspectionReport,
    expected_bend_count: int,
) -> bool:
    if _report_rank(candidate_report, expected_bend_count) <= _report_rank(current_report, expected_bend_count):
        return False
    return _report_decision_grade_rank(candidate_report, expected_bend_count) > _report_decision_grade_rank(
        current_report,
        expected_bend_count,
    )


def _should_promote_dense_local_merge(
    current_report: BendInspectionReport,
    merged_report: BendInspectionReport,
    expected_bend_count: int,
) -> bool:
    if _report_rank(merged_report, expected_bend_count) <= _report_rank(current_report, expected_bend_count):
        return False
    return _report_decision_grade_rank(merged_report, expected_bend_count) > _report_decision_grade_rank(
        current_report,
        expected_bend_count,
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


def _prefer_primary_with_local_position_signal(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
) -> bool:
    """
    Preserve a stronger primary bend claim while borrowing trusted CAD-local
    position evidence.

    This handles the common dense-scan case where global detection gives the
    right bend identity and metrology, but never emits line-based position
    evidence. When the local report can verify position for the same bend
    without providing a better overall bend claim, keep the primary match as the
    anchor and graft in the local position signal.
    """
    if primary is None or candidate is None:
        return False
    if str(getattr(primary, "assignment_source", "")).upper() != "GLOBAL_DETECTION":
        return False
    if str(getattr(candidate, "assignment_source", "")).upper() != "CAD_LOCAL_NEIGHBORHOOD":
        return False
    if str(getattr(primary, "status", "")).upper() == "NOT_DETECTED":
        return False
    if not bool(primary.correspondence_ready()) or not bool(candidate.correspondence_ready()):
        return False
    if bool(primary.position_claim_eligible()) or not bool(candidate.position_claim_eligible()):
        return False
    if _match_priority(candidate) > _match_priority(primary):
        return False
    return _match_has_position_verification(candidate)


def _merge_primary_with_local_position_signal(
    primary: BendMatch,
    candidate: BendMatch,
) -> BendMatch:
    merged = copy.deepcopy(primary)
    merged.line_start_deviation_mm = candidate.line_start_deviation_mm
    merged.line_end_deviation_mm = candidate.line_end_deviation_mm
    merged.line_center_deviation_mm = candidate.line_center_deviation_mm
    merged.measurement_context = copy.deepcopy(primary.measurement_context or {})
    merged.measurement_context["position_signal_source"] = "CAD_LOCAL_NEIGHBORHOOD"
    merged.measurement_context["position_signal_context"] = copy.deepcopy(candidate.measurement_context or {})
    if "trusted_alignment_for_release" not in merged.measurement_context:
        merged.measurement_context["trusted_alignment_for_release"] = bool(
            (candidate.measurement_context or {}).get("trusted_alignment_for_release", True)
        )
    return merged


def _prefer_authoritative_local_absence(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
    local_alignment_fitness: Optional[float],
    local_alignment_metadata: Optional[Dict[str, Any]] = None,
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
    metadata = dict(local_alignment_metadata or {})
    if metadata:
        if bool(metadata.get("alignment_abstained")):
            return False
        if str(metadata.get("alignment_source") or "").upper() != "PLANE_DATUM_REFINEMENT":
            return False
        if float(metadata.get("alignment_confidence") or 0.0) < 0.55:
            return False
    candidate_observability = normalize_observability_state(
        getattr(candidate, "observability_state", ""),
        physical_completion_state=getattr(candidate, "physical_completion_state", ""),
        status=getattr(candidate, "status", ""),
    )
    # Local absence is only authoritative when the bend neighborhood was
    # explicitly observed as still unformed. UNKNOWN remains neutral.
    if candidate_observability != "UNFORMED":
        return False
    if _match_has_position_verification(primary):
        return False
    if float(primary.match_confidence or 0.0) >= 0.85:
        return False
    return True


def _prefer_feature_policy_local_match(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
    feature_policy: Optional[Dict[str, Any]],
) -> bool:
    """
    On hybrid rolled parts, countable flange bends must be judged by the
    CAD-local verification path, not by generic global bend detections.
    """
    if primary is None or candidate is None or not feature_policy:
        return False
    if str((feature_policy or {}).get("countable_source") or "").strip().lower() != "legacy_folded_hybrid":
        return False
    if not bool(getattr(primary.cad_bend, "countable_in_regression", True)):
        return False
    if str(getattr(primary, "assignment_source", "")).upper() != "GLOBAL_DETECTION":
        return False
    if str(getattr(candidate, "measurement_mode", "")).lower() != "cad_local_neighborhood":
        return False
    if str(getattr(candidate, "status", "")).upper() != "NOT_DETECTED" and not bool(candidate.correspondence_ready()):
        return False
    return True


def _prefer_local_short_obtuse_geometry(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
) -> bool:
    """
    For short obtuse countable bends, prefer a CAD-local match when the global
    match has physically implausible bend-line geometry.
    """
    if primary is None or candidate is None:
        return False
    if str(getattr(primary, "assignment_source", "")).upper() != "GLOBAL_DETECTION":
        return False
    if str(getattr(candidate, "assignment_source", "")).upper() != "CAD_LOCAL_NEIGHBORHOOD":
        return False
    if str(getattr(primary, "status", "")).upper() == "NOT_DETECTED":
        return False
    if str(getattr(candidate, "status", "")).upper() == "NOT_DETECTED":
        return False
    if not bool(getattr(primary.cad_bend, "countable_in_regression", True)):
        return False

    target_angle = abs(float(getattr(primary.cad_bend, "target_angle", 0.0) or 0.0))
    if not (105.0 <= target_angle <= 140.0):
        return False

    line_length = float(
        np.linalg.norm(
            np.asarray(primary.cad_bend.bend_line_end, dtype=np.float64)
            - np.asarray(primary.cad_bend.bend_line_start, dtype=np.float64)
        )
    )
    if line_length > 70.0:
        return False

    primary_line_dev = abs(float(getattr(primary, "line_length_deviation_mm", 0.0) or 0.0))
    candidate_line_dev = abs(float(getattr(candidate, "line_length_deviation_mm", 0.0) or 0.0))
    if primary_line_dev <= max(20.0, 0.4 * line_length):
        return False
    if candidate_line_dev > 2.0:
        return False

    primary_angle_dev = abs(float(getattr(primary, "angle_deviation", 0.0) or 0.0))
    candidate_angle_dev = abs(float(getattr(candidate, "angle_deviation", 0.0) or 0.0))
    if candidate_angle_dev > primary_angle_dev + 2.0:
        return False

    return True


def _prefer_local_decision_ready_upgrade(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
) -> bool:
    """
    Allow a per-bend CAD-local upgrade when it adds a release-grade bend claim
    that the primary report cannot make.

    This is narrower than whole-report promotion: keep the primary report as the
    stability anchor, but preserve local confident ON_POSITION / IN_TOL evidence
    when it turns a previously non-eligible bend into a decision-ready PASS.
    """
    if primary is None or candidate is None:
        return False
    if str(getattr(primary, "assignment_source", "")).upper() != "GLOBAL_DETECTION":
        return False
    if str(getattr(candidate, "assignment_source", "")).upper() != "CAD_LOCAL_NEIGHBORHOOD":
        return False
    if not bool(candidate.correspondence_ready()):
        return False
    if not bool(candidate.position_claim_eligible()):
        return False
    if bool(primary.position_claim_eligible()):
        return False
    if str(getattr(candidate, "status", "")).upper() != "PASS":
        return False
    return True


def _prefer_primary_decision_ready_match_over_ambiguous_local(
    primary: Optional[BendMatch],
    candidate: Optional[BendMatch],
) -> bool:
    """
    Preserve a decision-ready primary bend claim when the CAD-local candidate is
    still correspondence-ambiguous.

    Dense local refinement is useful as diagnostic evidence, but an ambiguous
    local PASS/WARNING should not displace a confident primary bend-specific
    claim just because the raw status ranking looks better.
    """
    if primary is None or candidate is None:
        return False
    if str(getattr(candidate, "assignment_source", "")).upper() != "CAD_LOCAL_NEIGHBORHOOD":
        return False
    if str(getattr(primary, "status", "")).upper() == "NOT_DETECTED":
        return False
    if str(getattr(primary, "assignment_source", "")).upper() not in {"GLOBAL_DETECTION", "DETECTED_BEND_MATCH"}:
        return False
    if not bool(primary.correspondence_ready()):
        return False
    if bool(candidate.correspondence_ready()):
        return False
    return True


def merge_bend_reports(
    primary_report: BendInspectionReport,
    local_report: BendInspectionReport,
    part_id: str,
    local_alignment_fitness: Optional[float] = None,
    local_alignment_metadata: Optional[Dict[str, Any]] = None,
    feature_policy: Optional[Dict[str, Any]] = None,
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
        if _prefer_feature_policy_local_match(primary_match, local_match, feature_policy):
            merged_matches.append(copy.deepcopy(local_match))
        elif _prefer_local_short_obtuse_geometry(primary_match, local_match):
            merged_matches.append(copy.deepcopy(local_match))
        elif _prefer_authoritative_local_absence(
            primary_match,
            local_match,
            local_alignment_fitness,
            local_alignment_metadata=local_alignment_metadata,
        ):
            merged_matches.append(copy.deepcopy(local_match))
        elif _prefer_local_decision_ready_upgrade(
            primary_match,
            local_match,
        ):
            merged_matches.append(copy.deepcopy(local_match))
        elif _prefer_primary_with_local_position_signal(
            primary_match,
            local_match,
        ):
            merged_matches.append(_merge_primary_with_local_position_signal(primary_match, local_match))
        elif _prefer_primary_decision_ready_match_over_ambiguous_local(
            primary_match,
            local_match,
        ):
            merged_matches.append(copy.deepcopy(primary_match))
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
    scan_state: Optional[str] = None,
) -> bool:
    """Dense scans should prefer CAD-guided local verification over brute-force fallback."""
    point_count = int(len(scan_points))
    state = str(scan_state or "").strip().lower()
    target = max(1, int(expected_bend_count))
    detected = int(report.detected_count)
    countable_matches = [
        match for match in report.matches
        if bool(getattr(getattr(match, "cad_bend", None), "countable_in_regression", True))
    ]
    formed_confident_unknown_position = sum(
        1
        for match in countable_matches
        if match.completion_state() == "FORMED"
        and match.correspondence_state() == "CONFIDENT"
        and match.positional_state() == "UNKNOWN_POSITION"
    )
    if point_count >= 550000 and detected < max(1, int(np.ceil(0.75 * target))):
        return True
    if (
        state == "full"
        and point_count >= 300000
        and target >= 8
        and detected >= max(6, int(np.ceil(0.80 * target)))
        and formed_confident_unknown_position >= max(4, int(np.ceil(0.50 * target)))
    ):
        return True
    gap = max(0, target - detected)
    if gap == 0:
        return False
    # Near-complete dense scans still benefit from local verification when they
    # are only one or two bends short of the expected baseline.
    if point_count >= 550000 and gap <= max(2, int(np.ceil(0.15 * target))):
        return True
    # Medium-size full scans with large expected bend counts can still benefit
    # from CAD-guided local recovery when the global detector materially
    # under-recovers bends. This is intentionally narrow so it does not reopen
    # local refinement on already-stable full scans.
    if state == "full" and point_count >= 300000 and target >= 12 and detected < max(4, int(np.ceil(0.70 * target))):
        return True
    # Large partials with many expected bends are the main remaining MAE driver.
    # Let them use CAD-guided local recovery when the global pass materially
    # under-recovers, but keep the gate narrow so lower-count partials do not
    # reopen the expensive broad partial refinement path.
    if state == "partial" and point_count >= 300000 and target >= 12 and detected < max(4, int(np.ceil(0.45 * target))):
        return True
    return False


def _feature_policy_requires_local_refinement(
    cad_bends: List[BendSpecification],
    report: BendInspectionReport,
    feature_policy: Optional[Dict[str, Any]],
) -> bool:
    """
    Hybrid rolled parts need a CAD-guided local pass even when the scan is not
    globally dense, because the countable flange bends can be missed while the
    rolled shell dominates detector evidence.
    """
    if not feature_policy:
        return False
    if str(feature_policy.get("countable_source") or "").strip().lower() != "legacy_folded_hybrid":
        return False

    countable_targets = [
        bend for bend in cad_bends
        if bool(getattr(bend, "countable_in_regression", True))
    ]
    return bool(countable_targets)


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


def _local_assignment_runner_up_gap(
    assignment_candidates: List[Dict[str, Any]],
    selected_candidate: Dict[str, Any],
) -> Optional[float]:
    if str(selected_candidate.get("candidate_kind") or "").upper() != "MEASUREMENT":
        return None
    selected_id = str(selected_candidate.get("candidate_id") or "").strip()
    selected_score = float(selected_candidate.get("assignment_score") or 0.0)
    other_scores = [
        float(item.get("assignment_score") or 0.0)
        for item in assignment_candidates
        if str(item.get("candidate_kind") or "").upper() == "MEASUREMENT"
        and str(item.get("candidate_id") or "").strip() != selected_id
    ]
    runner_up = max(other_scores) if other_scores else 0.0
    return float(selected_score - runner_up)


def _local_assignment_confidence(
    observability: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    assignment_candidates: List[Dict[str, Any]],
    *,
    null_assignment_score: float,
    measurement_success: bool,
) -> float:
    base_confidence = float(observability.get("confidence") or 0.0)
    if not measurement_success:
        return base_confidence
    if str(selected_candidate.get("candidate_kind") or "").upper() != "MEASUREMENT":
        return base_confidence

    candidate_score = float(selected_candidate.get("assignment_score") or 0.0)
    visibility = float(observability.get("visibility_score") or 0.0)
    evidence = float(observability.get("evidence_score") or 0.0)
    runner_up_gap = max(0.0, float(_local_assignment_runner_up_gap(assignment_candidates, selected_candidate) or 0.0))
    null_margin = max(0.0, candidate_score - float(null_assignment_score or 0.0))
    boosted_confidence = max(
        base_confidence,
        min(
            0.9,
            0.20 * base_confidence
            + 0.30 * candidate_score
            + 0.20 * visibility
            + 0.20 * evidence
            + 0.10 * min(1.0, runner_up_gap / 0.25),
        ),
    )
    if (
        runner_up_gap >= 0.20
        and candidate_score >= 0.78
        and null_margin >= 0.45
        and int(observability.get("observed_surface_count") or 0) >= 2
        and visibility >= 0.85
        and evidence >= 0.85
    ):
        boosted_confidence = max(boosted_confidence, 0.82)
    return round(float(min(0.9, max(0.0, boosted_confidence))), 4)


def _trusted_parent_frame_metadata(alignment_metadata: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(alignment_metadata, dict):
        return False
    if bool(alignment_metadata.get("alignment_abstained")):
        return False
    if str(alignment_metadata.get("alignment_source") or "").upper() != "PLANE_DATUM_REFINEMENT":
        return False
    try:
        confidence = float(alignment_metadata.get("alignment_confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return confidence >= 0.55


def _allow_probe_region_implausible_completion(
    *,
    measurement: Dict[str, Any],
    cad_bend: BendSpecification,
    observability: Dict[str, Any],
    selected_candidate: Dict[str, Any],
    null_assignment_score: float,
    scan_state: Optional[str],
    alignment_metadata: Optional[Dict[str, Any]],
) -> bool:
    """
    Keep a narrow class of strong partial probe-region results as completed bends
    even when the raw probe angle degenerates toward a flat/open reading.
    """
    if str(scan_state or "").strip().lower() != "partial":
        return False
    if _trusted_parent_frame_metadata(alignment_metadata):
        return False
    if str(measurement.get("measurement_method") or "").strip().lower() != "probe_region":
        return False
    if not bool(measurement.get("success")) or measurement.get("measured_angle") is None:
        return False
    if not (70.0 <= float(cad_bend.target_angle) <= 110.0):
        return False
    measured_angle = float(measurement.get("measured_angle") or 0.0)
    if measured_angle < 150.0:
        return False
    if int(observability.get("observed_surface_count") or 0) < 2:
        return False
    if float(observability.get("surface_visibility_ratio") or 0.0) < 0.95:
        return False
    if float(observability.get("local_support_score") or 0.0) < 0.95:
        return False
    if float(observability.get("side_balance_score") or 0.0) < 0.12:
        return False
    candidate_score = float(selected_candidate.get("assignment_score") or 0.0)
    if candidate_score < 0.67:
        return False
    if float(null_assignment_score or 0.0) > 0.25:
        return False
    return True


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
    *,
    scan_state: Optional[str] = None,
    alignment_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    point_count = int(measurement.get("point_count") or 0)
    nearby_points = int(measurement.get("nearby_points") or point_count)
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
    support_points = max(point_count, side1_points + side2_points)
    support_point_count = max(support_points, nearby_points)
    point_score = min(1.0, support_point_count / 220.0)
    coverage_score = min(1.0, support_point_count / 90.0)
    surface_score = min(1.0, observed_surface_count / 2.0)
    expected_surface_count = 2.0
    surface_visibility_ratio = min(1.0, observed_surface_count / expected_surface_count)
    side_balance_score = 0.0
    if max(side1_points, side2_points) > 0:
        side_balance_score = float(min(side1_points, side2_points) / max(side1_points, side2_points))
    evidence_score = max(
        0.0,
        min(
            1.0,
            0.35 * surface_score + 0.25 * point_score + 0.20 * confidence + 0.20 * alignment_score,
        ),
    )
    visibility_score = max(
        0.0,
        min(
            1.0,
            0.40 * surface_visibility_ratio
            + 0.15 * point_score
            + 0.20 * coverage_score
            + 0.20 * side_balance_score
            + 0.15 * alignment_score,
        ),
    )

    measured_angle = measurement.get("measured_angle")
    success = bool(measurement.get("success"))
    measurement_method = str(measurement.get("measurement_method") or "").strip().lower()
    scan_state_norm = str(scan_state or "").strip().lower()

    if measurement_method == "parent_frame_profile_section":
        trusted_parent_frame = _trusted_parent_frame_metadata(alignment_metadata)
        slice_count = int(measurement.get("slice_count") or 0)
        planned_slice_count = int(measurement.get("planned_slice_count") or slice_count)
        effective_slice_count = float(measurement.get("effective_slice_count") or 0.0)
        angle_dispersion = float(measurement.get("angle_dispersion_deg") or 999.0)
        aggregate_angle = measurement.get("aggregate_angle_deg", measured_angle)
        visible_span_ratio = float(measurement.get("visible_span_ratio") or 0.0)
        alignment_confidence = 0.0
        if isinstance(alignment_metadata, dict):
            try:
                alignment_confidence = float(alignment_metadata.get("alignment_confidence") or 0.0)
            except Exception:
                alignment_confidence = 0.0
        slice_ratio = float(slice_count / planned_slice_count) if planned_slice_count > 0 else 0.0
        slice_score = min(1.0, slice_ratio / 0.7) if planned_slice_count > 0 else min(1.0, slice_count / 7.0)
        effective_score = min(1.0, effective_slice_count / 6.0)
        dispersion_score = max(0.0, min(1.0, 1.0 - (angle_dispersion / 8.0)))
        span_score = min(1.0, visible_span_ratio / 0.6)
        evidence_score = max(
            0.0,
            min(
                1.0,
                0.25 * slice_score
                + 0.25 * effective_score
                + 0.20 * dispersion_score
                + 0.15 * span_score
                + 0.15 * alignment_confidence
                + 0.10 * confidence,
            ),
        )
        visibility_score = max(
            0.0,
            min(
                1.0,
                0.25 * slice_score
                + 0.20 * effective_score
                + 0.20 * surface_visibility_ratio
                + 0.20 * alignment_confidence
                + 0.15 * span_score,
            ),
        )
        support_point_count = max(support_point_count, slice_count * 12)
        expected_zone_visible = bool(
            measurement.get("expected_zone_visible")
            or (
                visible_span_ratio >= 0.4
                and planned_slice_count > 0
                and slice_count >= max(3, int(np.ceil(0.5 * planned_slice_count)))
            )
        )
        detail_state = "UNOBSERVED"
        if slice_count > 0:
            detail_state = "PARTIALLY_OBSERVED"
        if (
            trusted_parent_frame
            and success
            and aggregate_angle is not None
            and slice_count >= 5
            and effective_slice_count >= 4.0
            and angle_dispersion <= 6.0
            and visible_span_ratio >= 0.35
        ):
            flat_gap = abs(float(aggregate_angle) - 180.0)
            target_gap = abs(float(aggregate_angle) - float(cad_angle))
            if expected_zone_visible and visible_span_ratio >= 0.5 and flat_gap <= 5.0 and angle_dispersion <= 4.0:
                detail_state = "OBSERVED_NOT_FORMED"
            elif slice_ratio >= 0.5 and flat_gap >= 8.0 and target_gap <= 35.0 and angle_dispersion <= 5.0:
                detail_state = "OBSERVED_FORMED"
            else:
                detail_state = "PARTIALLY_OBSERVED"
        elif trusted_parent_frame and expected_zone_visible:
            detail_state = "PARTIALLY_OBSERVED"

        state = normalize_observability_state(
            detail_state,
            status="PASS" if detail_state == "OBSERVED_FORMED" else "NOT_DETECTED",
        )
        if scan_state_norm == "partial" and detail_state == "OBSERVED_FORMED" and not trusted_parent_frame:
            state = "UNKNOWN"
            detail_state = "PARTIALLY_OBSERVED"
        return {
            "state": state,
            "detail_state": detail_state,
            "confidence": confidence,
            "observed_surface_count": observed_surface_count,
            "point_count": point_count,
            "support_points": support_points,
            "support_point_count": support_point_count,
            "evidence_score": evidence_score,
            "visibility_score": visibility_score,
            "surface_visibility_ratio": surface_visibility_ratio,
            "local_support_score": min(1.0, 0.6 * slice_score + 0.4 * span_score),
            "side_balance_score": side_balance_score,
            "visibility_score_source": "parent_frame_profile_section_v2",
        }

    detail_state = "UNOBSERVED"
    if success and measured_angle is not None:
        flat_gap = abs(float(measured_angle) - 180.0)
        target_gap = abs(float(measured_angle) - float(cad_angle))
        trusted_parent_frame = _trusted_parent_frame_metadata(alignment_metadata)
        weak_partial_profile_section = (
            scan_state_norm == "partial"
            and measurement_method == "profile_section"
            and not trusted_parent_frame
            and side_balance_score < 0.2
        )
        if weak_partial_profile_section:
            detail_state = "PARTIALLY_OBSERVED" if support_point_count >= 24 else "UNOBSERVED"
        elif observed_surface_count >= 2 and point_count >= 24 and flat_gap + 8.0 < target_gap:
            detail_state = "OBSERVED_NOT_FORMED"
        elif observed_surface_count >= 2:
            detail_state = "OBSERVED_FORMED"
        else:
            detail_state = "PARTIALLY_OBSERVED"
    elif observed_surface_count >= 2 or point_count >= 20:
        detail_state = "PARTIALLY_OBSERVED"
    elif support_point_count >= 40:
        detail_state = "PARTIALLY_OBSERVED"

    state = normalize_observability_state(
        detail_state,
        status="PASS" if success and measured_angle is not None else "NOT_DETECTED",
    )
    if success and measured_angle is not None:
        trusted_parent_frame = _trusted_parent_frame_metadata(alignment_metadata)
        weak_partial_profile_section = (
            scan_state_norm == "partial"
            and measurement_method == "profile_section"
            and not trusted_parent_frame
            and side_balance_score < 0.2
        )
        if weak_partial_profile_section:
            state = "UNKNOWN"

    return {
        "state": state,
        "detail_state": detail_state,
        "confidence": confidence,
        "observed_surface_count": observed_surface_count,
        "point_count": point_count,
        "support_points": support_points,
        "support_point_count": support_point_count,
        "evidence_score": evidence_score,
        "visibility_score": visibility_score,
        "surface_visibility_ratio": surface_visibility_ratio,
        "local_support_score": point_score,
        "side_balance_score": side_balance_score,
        "visibility_score_source": "heuristic_local_visibility_v0",
    }


def _physical_state_from_observability(observability_state: str, success: bool) -> str:
    state = normalize_observability_state(
        observability_state,
        status="PASS" if success else "NOT_DETECTED",
    )
    if state == "FORMED" and success:
        return "FORMED"
    if state == "UNFORMED":
        return "NOT_FORMED"
    return "UNKNOWN"


def _support_only_measurement(
    scan_tree: cKDTree,
    scan_points: np.ndarray,
    cad_bend: BendSpecification,
    *,
    search_radius: float = 35.0,
    probe_offset: float = 15.0,
    probe_radius: float = 10.0,
) -> Dict[str, Any]:
    center = np.asarray((cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0, dtype=np.float64)
    tangent = np.asarray(cad_bend.bend_line_direction, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm > 1e-9:
        tangent = tangent / tangent_norm
    n1 = np.asarray(
        cad_bend.flange1_normal if cad_bend.flange1_normal is not None else np.array([0.0, 1.0, 0.0]),
        dtype=np.float64,
    )
    n2 = np.asarray(
        cad_bend.flange2_normal if cad_bend.flange2_normal is not None else np.array([0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    tang1 = np.cross(tangent, n1)
    tang2 = np.cross(tangent, n2)
    if float(np.linalg.norm(tang1)) > 1e-9:
        tang1 = tang1 / np.linalg.norm(tang1)
    else:
        tang1 = n1
    if float(np.linalg.norm(tang2)) > 1e-9:
        tang2 = tang2 / np.linalg.norm(tang2)
    else:
        tang2 = n2
    if float(np.dot(tang1, tang2)) > 0.0:
        tang2 = -tang2

    nearby_idx = scan_tree.query_ball_point(center, search_radius)
    probe1_idx = scan_tree.query_ball_point(center + tang1 * probe_offset, probe_radius)
    probe2_idx = scan_tree.query_ball_point(center + tang2 * probe_offset, probe_radius)
    point_count = int(len(nearby_idx))
    return {
        "success": False,
        "measured_angle": None,
        "measurement_method": "support_only_probe",
        "measurement_confidence": "very_low",
        "point_count": point_count,
        "nearby_points": point_count,
        "side1_points": int(len(probe1_idx)),
        "side2_points": int(len(probe2_idx)),
        "cad_alignment1": None,
        "cad_alignment2": None,
    }


def _cad_spec_to_measurement_dict(
    cad_bend: BendSpecification,
    ordinal: int,
    *,
    selected_baseline_direct: bool = False,
) -> Dict[str, Any]:
    center = 0.5 * (
        np.asarray(cad_bend.bend_line_start, dtype=np.float64)
        + np.asarray(cad_bend.bend_line_end, dtype=np.float64)
    )
    n1 = np.asarray(
        cad_bend.flange1_normal if cad_bend.flange1_normal is not None else np.array([0.0, 1.0, 0.0]),
        dtype=np.float64,
    )
    n2 = np.asarray(
        cad_bend.flange2_normal if cad_bend.flange2_normal is not None else np.array([0.0, 0.0, 1.0]),
        dtype=np.float64,
    )
    return {
        "bend_id": cad_bend.bend_id,
        "bend_name": cad_bend.bend_id,
        "bend_angle": float(cad_bend.target_angle),
        "bend_center": center,
        "bend_direction": np.asarray(cad_bend.bend_line_direction, dtype=np.float64),
        "bend_line_start": np.asarray(cad_bend.bend_line_start, dtype=np.float64),
        "bend_line_end": np.asarray(cad_bend.bend_line_end, dtype=np.float64),
        "surface1_normal": n1,
        "surface2_normal": n2,
        "surface1_id": -(ordinal * 2 + 1),
        "surface2_id": -(ordinal * 2 + 2),
        "bend_form": cad_bend.bend_form,
        "feature_type": cad_bend.feature_type,
        "countable_in_regression": bool(cad_bend.countable_in_regression),
        "selected_baseline_direct": bool(selected_baseline_direct),
    }


def _match_selected_spec_to_extracted_bend(
    cad_bend: BendSpecification,
    extracted_bends: List[Any],
    *,
    used_indices: Optional[set[int]] = None,
) -> Optional[Any]:
    used_indices = used_indices or set()
    cad_mid = 0.5 * (
        np.asarray(cad_bend.bend_line_start, dtype=np.float64)
        + np.asarray(cad_bend.bend_line_end, dtype=np.float64)
    )
    cad_dir = np.asarray(cad_bend.bend_line_direction, dtype=np.float64)
    best_idx: Optional[int] = None
    best_score = float("inf")

    for idx, bend in enumerate(extracted_bends or []):
        if idx in used_indices:
            continue
        bend_mid = np.asarray(getattr(bend, "bend_center", cad_mid), dtype=np.float64)
        bend_dir = np.asarray(getattr(bend, "bend_direction", cad_dir), dtype=np.float64)
        bend_dir_norm = float(np.linalg.norm(bend_dir))
        if bend_dir_norm > 1e-9:
            bend_dir = bend_dir / bend_dir_norm
        center_dist = float(np.linalg.norm(bend_mid - cad_mid))
        direction_alignment = abs(float(np.dot(cad_dir, bend_dir)))
        angle_gap = abs(float(getattr(bend, "bend_angle", cad_bend.target_angle)) - float(cad_bend.target_angle))
        score = center_dist + (angle_gap * 1.5) + ((1.0 - direction_alignment) * 20.0)
        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None
    used_indices.add(best_idx)
    return extracted_bends[best_idx]


def _build_selected_measurement_dicts(
    cad_bends: List[BendSpecification],
    extracted_bends: List[Any],
) -> Tuple[List[Dict[str, Any]], int]:
    """Preserve real extracted CAD surface IDs for selected bends when possible."""
    used_indices: set[int] = set()
    measurement_dicts: List[Dict[str, Any]] = []
    matched_count = 0

    for ordinal, spec in enumerate(cad_bends, start=1):
        matched = _match_selected_spec_to_extracted_bend(
            spec,
            extracted_bends,
            used_indices=used_indices,
        )
        if matched is None:
            measurement_dicts.append(
                _cad_spec_to_measurement_dict(
                    spec,
                    ordinal=ordinal,
                    selected_baseline_direct=True,
                )
            )
            continue

        payload = matched.to_dict()
        payload["bend_id"] = spec.bend_id
        payload["bend_name"] = spec.bend_id
        payload["bend_angle"] = float(spec.target_angle)
        payload["bend_center"] = (
            0.5
            * (
                np.asarray(spec.bend_line_start, dtype=np.float64)
                + np.asarray(spec.bend_line_end, dtype=np.float64)
            )
        )
        payload["bend_direction"] = np.asarray(spec.bend_line_direction, dtype=np.float64)
        payload["bend_line_start"] = np.asarray(spec.bend_line_start, dtype=np.float64)
        payload["bend_line_end"] = np.asarray(spec.bend_line_end, dtype=np.float64)
        payload["surface1_normal"] = np.asarray(
            spec.flange1_normal if spec.flange1_normal is not None else payload.get("surface1_normal"),
            dtype=np.float64,
        )
        payload["surface2_normal"] = np.asarray(
            spec.flange2_normal if spec.flange2_normal is not None else payload.get("surface2_normal"),
            dtype=np.float64,
        )
        payload["bend_form"] = spec.bend_form
        payload["feature_type"] = spec.feature_type
        payload["countable_in_regression"] = bool(spec.countable_in_regression)
        payload["selected_baseline_direct"] = True
        measurement_dicts.append(payload)
        matched_count += 1

    return measurement_dicts, matched_count


def _build_local_assignment_candidates(
    cad_bend: BendSpecification,
    measurement_entries: List[Dict[str, Any]],
    *,
    used_measurements: Optional[set[int]] = None,
    preferred_idx: Optional[int] = None,
    distance_limit_mm: float = 45.0,
    angle_limit_deg: float = 35.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Materialize local candidate-or-null assignments for one CAD bend."""
    used_measurements = used_measurements or set()
    cad_mid = (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0
    cad_dir = np.asarray(cad_bend.bend_line_direction, dtype=np.float64)
    candidates: List[Dict[str, Any]] = []
    best_visibility = 0.0
    best_evidence = 0.0

    for idx, entry in enumerate(measurement_entries):
        center = np.asarray(entry.get("center"), dtype=np.float64)
        center_dist = float(np.linalg.norm(center - cad_mid))
        measurement = entry.get("measurement") or {}
        candidate_angle = _local_assignment_entry_angle(entry)
        cad_angle_gap = abs(float(candidate_angle or 0.0) - float(cad_bend.target_angle))
        if center_dist > distance_limit_mm or cad_angle_gap > angle_limit_deg:
            continue

        observability = entry.get("observability") or _local_measurement_observability(
            measurement,
            cad_bend.target_angle,
        )
        measurement_frame = entry.get("measurement_frame") or _local_bend_frame(
            entry.get("cad_bend_dict") or {},
            cad_bend,
        )
        tangent = np.asarray(measurement_frame.get("tangent") or cad_dir, dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-9:
            tangent = tangent / tangent_norm
        else:
            tangent = cad_dir
        direction_alignment = abs(float(np.dot(cad_dir, tangent)))
        confidence = _measurement_confidence_score(measurement.get("measurement_confidence"))
        visibility = float(observability.get("visibility_score") or 0.0)
        evidence = float(observability.get("evidence_score") or 0.0)
        used_penalty = 0.18 if idx in used_measurements else 0.0
        assignment_score = max(
            0.0,
            min(
                1.0,
                0.30 * max(0.0, 1.0 - (center_dist / max(distance_limit_mm, 1e-6)))
                + 0.18 * max(0.0, 1.0 - (cad_angle_gap / max(angle_limit_deg, 1e-6)))
                + 0.18 * visibility
                + 0.16 * evidence
                + 0.10 * confidence
                + 0.08 * direction_alignment
                - used_penalty,
            ),
        )
        candidate = {
            "candidate_id": f"local_candidate_{idx}",
            "candidate_kind": "MEASUREMENT",
            "measurement_index": idx,
            "assignment_score": round(float(assignment_score), 4),
            "center_distance_mm": round(center_dist, 4),
            "cad_angle_gap_deg": round(cad_angle_gap, 4),
            "direction_alignment": round(direction_alignment, 4),
            "visibility_score": round(visibility, 4),
            "evidence_score": round(evidence, 4),
            "measurement_confidence_score": round(confidence, 4),
            "observability_state": observability.get("state"),
            "physical_completion_state": _physical_state_from_observability(
                str(observability.get("state") or ""),
                bool(measurement.get("success")),
            ),
            "measurement_success": bool(measurement.get("success")),
            "measured_angle": (
                round(float(measurement["measured_angle"]), 4)
                if measurement.get("measured_angle") is not None
                else None
            ),
            "used_by_other_bend": idx in used_measurements,
            "measurement_method": measurement.get("measurement_method"),
        }
        candidates.append(candidate)
        best_visibility = max(best_visibility, visibility)
        best_evidence = max(best_evidence, evidence)

    null_score = max(
        0.05,
        min(
            1.0,
            0.55 * (1.0 - best_visibility) + 0.45 * (1.0 - best_evidence),
        ),
    )
    null_candidate = {
        "candidate_id": "null_candidate",
        "candidate_kind": "NULL",
        "measurement_index": None,
        "assignment_score": round(float(null_score), 4),
        "center_distance_mm": None,
        "cad_angle_gap_deg": None,
        "direction_alignment": None,
        "visibility_score": round(float(best_visibility), 4),
        "evidence_score": round(float(best_evidence), 4),
        "measurement_confidence_score": 0.0,
        "observability_state": "UNOBSERVED" if best_visibility < 0.2 else "PARTIALLY_OBSERVED",
        "physical_completion_state": "UNKNOWN",
        "measurement_success": False,
        "measured_angle": None,
        "used_by_other_bend": False,
        "measurement_method": None,
    }
    candidates.append(null_candidate)
    candidates.sort(key=lambda item: (float(item.get("assignment_score") or 0.0), item.get("candidate_kind") != "NULL"), reverse=True)

    selected = null_candidate
    if preferred_idx is not None:
        preferred = next(
            (
                item
                for item in candidates
                if item.get("candidate_kind") == "MEASUREMENT" and int(item.get("measurement_index")) == int(preferred_idx)
            ),
            None,
        )
        if preferred is not None:
            selected = preferred
    return candidates, copy.deepcopy(selected)


def _select_local_assignment_candidate(
    cad_bend: BendSpecification,
    measurement_entries: List[Dict[str, Any]],
    *,
    used_measurements: Optional[set[int]] = None,
    distance_limit_mm: float = 45.0,
    angle_limit_deg: float = 35.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, Optional[int]]:
    """
    Select the best local measurement candidate for a CAD bend.

    Unlike the older center+angle heuristic, this uses the same assignment score
    that is later exposed to runtime semantics, so selection and reporting are
    aligned.
    """
    assignment_candidates, selected_candidate = _build_local_assignment_candidates(
        cad_bend,
        measurement_entries,
        used_measurements=used_measurements,
        preferred_idx=None,
        distance_limit_mm=distance_limit_mm,
        angle_limit_deg=angle_limit_deg,
    )
    null_assignment_score = float(
        next(
            (item.get("assignment_score") for item in assignment_candidates if item.get("candidate_kind") == "NULL"),
            0.0,
        )
    )
    measurement_candidates = [
        item for item in assignment_candidates
        if str(item.get("candidate_kind") or "").upper() == "MEASUREMENT"
    ]
    if measurement_candidates:
        top_measurement = max(
            measurement_candidates,
            key=lambda item: float(item.get("assignment_score") or 0.0),
        )
        if float(top_measurement.get("assignment_score") or 0.0) > null_assignment_score:
            selected_candidate = copy.deepcopy(top_measurement)
    if (
        str(selected_candidate.get("candidate_kind") or "").upper() == "MEASUREMENT"
        and float(selected_candidate.get("assignment_score") or 0.0) > null_assignment_score
        and selected_candidate.get("measurement_index") is not None
    ):
        return (
            assignment_candidates,
            copy.deepcopy(selected_candidate),
            null_assignment_score,
            int(selected_candidate.get("measurement_index")),
        )
    return assignment_candidates, copy.deepcopy(selected_candidate), null_assignment_score, None


def _resolve_local_assignment_conflicts(
    plans: List[Dict[str, Any]],
    *,
    brute_force_limit: int = 8,
) -> List[Dict[str, Any]]:
    """
    Resolve CAD-local candidate reuse with a one-to-one assignment pass.

    Greedy bend-order selection can assign the same local measurement to
    multiple bends. Solve each connected bend/measurement component jointly and
    keep a unique measurement assignment when that improves total assignment
    score.
    """

    def _measurement_candidates(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            dict(item)
            for item in (plan.get("assignment_candidates") or [])
            if str(item.get("candidate_kind") or "").upper() == "MEASUREMENT"
        ]

    def _null_candidate(plan: Dict[str, Any]) -> Dict[str, Any]:
        return next(
            (
                dict(item)
                for item in (plan.get("assignment_candidates") or [])
                if str(item.get("candidate_kind") or "").upper() == "NULL"
            ),
            {
                "candidate_id": "null_candidate",
                "candidate_kind": "NULL",
                "measurement_index": None,
                "assignment_score": float(plan.get("null_assignment_score") or 0.0),
                "used_by_other_bend": False,
            },
        )

    def _objective_score(option: Dict[str, Any], *, null_score: float) -> float:
        base_score = float(option.get("assignment_score") or 0.0)
        if str(option.get("candidate_kind") or "").upper() != "MEASUREMENT":
            return base_score
        if bool(option.get("measurement_success")):
            return base_score
        # Failed probe-only candidates can still carry strong locality evidence.
        # Keep them slightly above null for diagnostics, but do not let them
        # crowd out a successful one-to-one assignment in the global objective.
        return min(base_score, float(null_score) + 1e-3)

    measurement_to_bends: Dict[int, set[int]] = {}
    for bend_idx, plan in enumerate(plans):
        for candidate in _measurement_candidates(plan):
            measurement_idx = candidate.get("measurement_index")
            if measurement_idx is None:
                continue
            measurement_to_bends.setdefault(int(measurement_idx), set()).add(bend_idx)

    visited_bends: set[int] = set()
    components: List[List[int]] = []
    for start_idx in range(len(plans)):
        if start_idx in visited_bends:
            continue
        stack = [start_idx]
        component: set[int] = set()
        linked_measurements: set[int] = set()
        while stack:
            bend_idx = stack.pop()
            if bend_idx in component:
                continue
            component.add(bend_idx)
            visited_bends.add(bend_idx)
            for candidate in _measurement_candidates(plans[bend_idx]):
                measurement_idx = candidate.get("measurement_index")
                if measurement_idx is None:
                    continue
                measurement_idx = int(measurement_idx)
                if measurement_idx in linked_measurements:
                    continue
                linked_measurements.add(measurement_idx)
                stack.extend(measurement_to_bends.get(measurement_idx, set()) - component)
        components.append(sorted(component))

    resolved_plans = [copy.deepcopy(plan) for plan in plans]
    for component_bends in components:
        options_by_bend: Dict[int, List[Dict[str, Any]]] = {}
        null_scores_by_bend: Dict[int, float] = {}
        unique_measurements: set[int] = set()
        for bend_idx in component_bends:
            null_candidate = _null_candidate(resolved_plans[bend_idx])
            null_score = float(null_candidate.get("assignment_score") or resolved_plans[bend_idx].get("null_assignment_score") or 0.0)
            options = _measurement_candidates(resolved_plans[bend_idx])
            options.append(null_candidate)
            options_by_bend[bend_idx] = options
            null_scores_by_bend[bend_idx] = null_score
            for option in options:
                measurement_idx = option.get("measurement_index")
                if measurement_idx is not None:
                    unique_measurements.add(int(measurement_idx))

        if len(component_bends) <= 1 or not unique_measurements:
            continue

        def _greedy_component_solution() -> Dict[int, Dict[str, Any]]:
            remaining_measurements = set(unique_measurements)
            resolved: Dict[int, Dict[str, Any]] = {}
            for bend_idx in sorted(
                component_bends,
                key=lambda idx: max(
                    _objective_score(item, null_score=null_scores_by_bend[idx])
                    for item in options_by_bend[idx]
                ),
                reverse=True,
            ):
                ordered = sorted(
                    options_by_bend[bend_idx],
                    key=lambda item: _objective_score(item, null_score=null_scores_by_bend[bend_idx]),
                    reverse=True,
                )
                chosen = next(
                    (
                        item for item in ordered
                        if item.get("measurement_index") is None or int(item.get("measurement_index")) in remaining_measurements
                    ),
                    ordered[-1],
                )
                resolved[bend_idx] = dict(chosen)
                measurement_idx = chosen.get("measurement_index")
                if measurement_idx is not None:
                    remaining_measurements.discard(int(measurement_idx))
            return resolved

        def _brute_force_component_solution() -> Dict[int, Dict[str, Any]]:
            search_order = sorted(component_bends, key=lambda idx: len(options_by_bend[idx]))
            best_total = float("-inf")
            best_selection: Optional[Dict[int, Dict[str, Any]]] = None

            def _backtrack(order_idx: int, used_measurements: set[int], current_total: float, selection: Dict[int, Dict[str, Any]]) -> None:
                nonlocal best_total, best_selection
                if order_idx >= len(search_order):
                    if current_total > best_total:
                        best_total = current_total
                        best_selection = {idx: dict(choice) for idx, choice in selection.items()}
                    return

                bend_idx = search_order[order_idx]
                ordered_options = sorted(
                    options_by_bend[bend_idx],
                    key=lambda item: _objective_score(item, null_score=null_scores_by_bend[bend_idx]),
                    reverse=True,
                )
                for option in ordered_options:
                    measurement_idx = option.get("measurement_index")
                    if measurement_idx is not None and int(measurement_idx) in used_measurements:
                        continue
                    selection[bend_idx] = option
                    if measurement_idx is not None:
                        used_measurements.add(int(measurement_idx))
                    _backtrack(
                        order_idx + 1,
                        used_measurements,
                        current_total + _objective_score(option, null_score=null_scores_by_bend[bend_idx]),
                        selection,
                    )
                    if measurement_idx is not None:
                        used_measurements.discard(int(measurement_idx))
                    selection.pop(bend_idx, None)

            _backtrack(0, set(), 0.0, {})
            return best_selection or {}

        if len(component_bends) > brute_force_limit or len(unique_measurements) > brute_force_limit + 2:
            resolved = _greedy_component_solution()
        else:
            resolved = _brute_force_component_solution()

        selected_measurements = {
            int(choice.get("measurement_index"))
            for choice in resolved.values()
            if choice.get("measurement_index") is not None
        }
        for bend_idx in component_bends:
            plan = resolved_plans[bend_idx]
            selected = dict(resolved.get(bend_idx) or _null_candidate(plan))
            updated_candidates: List[Dict[str, Any]] = []
            for candidate in plan.get("assignment_candidates") or []:
                candidate_copy = dict(candidate)
                measurement_idx = candidate_copy.get("measurement_index")
                candidate_copy["used_by_other_bend"] = (
                    measurement_idx is not None
                    and int(measurement_idx) in selected_measurements
                    and str(candidate_copy.get("candidate_id") or "") != str(selected.get("candidate_id") or "")
                )
                updated_candidates.append(candidate_copy)
                if str(candidate_copy.get("candidate_id") or "") == str(selected.get("candidate_id") or ""):
                    selected = dict(candidate_copy)
            plan["assignment_candidates"] = updated_candidates
            plan["selected_candidate"] = selected
            measurement_idx = selected.get("measurement_index")
            plan["best_idx"] = int(measurement_idx) if measurement_idx is not None else None
    return resolved_plans


def _local_assignment_entry_angle(entry: Dict[str, Any]) -> Optional[float]:
    measurement = entry.get("measurement") or {}
    for key in ("aggregate_angle_deg", "measured_angle"):
        value = measurement.get(key)
        if value is not None:
            return float(value)
    value = entry.get("cad_angle")
    if value is None:
        return None
    return float(value)


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
    physical_completion_state: str = "UNKNOWN",
    observability_state: str = "UNKNOWN",
    observability_detail_state: str = "UNOBSERVED",
    observability_confidence: float = 0.0,
    visibility_score: float = 0.0,
    visibility_score_source: str = "heuristic_local_visibility_v0",
    surface_visibility_ratio: float = 0.0,
    local_support_score: float = 0.0,
    side_balance_score: float = 0.0,
    assignment_source: str = "NONE",
    assignment_confidence: float = 0.0,
    assignment_candidate_id: Optional[str] = None,
    assignment_candidate_kind: str = "NONE",
    assignment_candidate_score: float = 0.0,
    assignment_null_score: float = 0.0,
    assignment_candidate_count: int = 0,
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
        physical_completion_state=physical_completion_state,
        observability_state=observability_state,
        observability_detail_state=observability_detail_state,
        observability_confidence=float(observability_confidence or 0.0),
        visibility_score=float(visibility_score or 0.0),
        visibility_score_source=str(visibility_score_source or "heuristic_local_visibility_v0"),
        surface_visibility_ratio=float(surface_visibility_ratio or 0.0),
        local_support_score=float(local_support_score or 0.0),
        side_balance_score=float(side_balance_score or 0.0),
        assignment_source=assignment_source,
        assignment_confidence=float(assignment_confidence or 0.0),
        assignment_candidate_id=assignment_candidate_id,
        assignment_candidate_kind=assignment_candidate_kind,
        assignment_candidate_score=float(assignment_candidate_score or 0.0),
        assignment_null_score=float(assignment_null_score or 0.0),
        assignment_candidate_count=int(assignment_candidate_count or 0),
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
    scan_state: Optional[str] = None,
    matcher_kwargs: Optional[Dict[str, Any]] = None,
    local_refinement_kwargs: Optional[Dict[str, Any]] = None,
    feature_policy: Optional[Dict[str, Any]] = None,
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
        from bend_detector import (
            measure_all_scan_bends,
            measure_scan_bend_via_surface_classification,
            _compute_triangle_centers_and_normals,
        )
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
    alignment_metadata = {}
    if hasattr(engine, "alignment_metadata") and engine.alignment_metadata is not None:
        try:
            alignment_metadata = engine.alignment_metadata.to_dict()
        except Exception:
            alignment_metadata = dict(engine.alignment_metadata or {})

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

    use_selected_bend_measurement = (
        len(cad_bends) >= cad_extract.total_bends + 4
        and cad_extract.total_bends <= max(8, int(0.75 * len(cad_bends)))
    )
    if str((feature_policy or {}).get("countable_source") or "").strip().lower() == "legacy_folded_hybrid":
        use_selected_bend_measurement = True
    if use_selected_bend_measurement:
        _phase(
            f"use_selected_bend_measurement selected={len(cad_bends)} extracted={cad_extract.total_bends}"
        )
        cad_bend_dicts, matched_selected_count = _build_selected_measurement_dicts(
            cad_bends,
            cad_extract.bends,
        )
        use_surface_classification = matched_selected_count > 0
        cad_vertices = np.asarray(engine.reference_mesh.vertices) if use_surface_classification else None
        cad_triangles = np.asarray(engine.reference_mesh.triangles) if use_surface_classification else None
        surface_face_map = (
            {
                surface.surface_id: surface.face_indices
                for surface in cad_extract.surfaces
            }
            if use_surface_classification
            else None
        )
        warnings_local.append(
            "Dense local refinement measured selected CAD baseline directly "
            f"({len(cad_bends)} bends) because mesh re-extraction undercounted at {cad_extract.total_bends}. "
            f"Matched {matched_selected_count} selected bends back to extracted CAD surfaces."
        )
    else:
        cad_bend_dicts = [bend.to_dict() for bend in cad_extract.bends]
        cad_vertices = np.asarray(engine.reference_mesh.vertices)
        cad_triangles = np.asarray(engine.reference_mesh.triangles)
        surface_face_map = {
            surface.surface_id: surface.face_indices
            for surface in cad_extract.surfaces
        }
    measurement_cloud = getattr(engine, "aligned_scan_raw", None)
    if str((feature_policy or {}).get("countable_source") or "").strip().lower() == "legacy_folded_hybrid":
        if measurement_cloud is None or len(measurement_cloud.points) == 0:
            measurement_cloud = engine.aligned_scan
    else:
        measurement_cloud = engine.aligned_scan
    aligned_points = np.asarray(measurement_cloud.points)
    _phase(
        f"measure_all_scan_bends begin aligned_points={len(aligned_points)} "
        f"cad_bends={len(cad_bend_dicts)} raw_cloud={'yes' if measurement_cloud is getattr(engine, 'aligned_scan_raw', None) else 'no'}"
    )
    measurements = measure_all_scan_bends(
        aligned_points,
        cad_bend_dicts,
        search_radius=35.0,
        cad_vertices=cad_vertices,
        cad_triangles=cad_triangles,
        surface_face_map=surface_face_map,
        deviations=None,
        alignment_metadata=alignment_metadata,
        scan_state=scan_state,
    )
    if cad_vertices is not None and cad_triangles is not None and len(measurements) == len(cad_bend_dicts):
        all_face_centers, all_face_normals = _compute_triangle_centers_and_normals(cad_vertices, cad_triangles)
        for idx, (cad_bend_dict, measurement) in enumerate(zip(cad_bend_dicts, measurements)):
            try:
                bend_angle = abs(float(cad_bend_dict.get("bend_angle") or 0.0))
                line_start = np.asarray(cad_bend_dict.get("bend_line_start", cad_bend_dict.get("bend_center")), dtype=np.float64)
                line_end = np.asarray(cad_bend_dict.get("bend_line_end", cad_bend_dict.get("bend_center")), dtype=np.float64)
                line_length = float(np.linalg.norm(line_end - line_start))
            except Exception:
                continue
            if not bool(cad_bend_dict.get("selected_baseline_direct")):
                continue
            if not (105.0 <= bend_angle <= 140.0 and line_length <= 70.0):
                continue
            recovered_surface_measurement = measure_scan_bend_via_surface_classification(
                scan_points=aligned_points,
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                surf1_face_indices=np.empty((0,), dtype=np.int32),
                surf2_face_indices=np.empty((0,), dtype=np.int32),
                cad_bend_center=np.asarray(cad_bend_dict["bend_center"], dtype=np.float64),
                cad_surf1_normal=np.asarray(cad_bend_dict["surface1_normal"], dtype=np.float64),
                cad_surf2_normal=np.asarray(cad_bend_dict["surface2_normal"], dtype=np.float64),
                cad_bend_angle=float(cad_bend_dict["bend_angle"]),
                search_radius=24.0,
                min_offset_mm=5.0,
                cad_bend_direction=np.asarray(cad_bend_dict["bend_direction"], dtype=np.float64),
                cad_bend_line_start=line_start,
                cad_bend_line_end=line_end,
                allow_line_locality=True,
                allow_face_recovery=True,
                all_face_centers=all_face_centers,
                all_face_normals=all_face_normals,
            )
            current_success = bool(measurement.get("success")) and measurement.get("measured_angle") is not None
            recovered_success = bool(recovered_surface_measurement.get("success")) and recovered_surface_measurement.get("measured_angle") is not None
            if not recovered_success:
                continue
            current_gap = abs(float(measurement.get("measured_angle") or 0.0) - bend_angle) if current_success else float("inf")
            recovered_gap = abs(float(recovered_surface_measurement["measured_angle"]) - bend_angle)
            if recovered_gap + 0.5 < current_gap:
                measurements[idx] = recovered_surface_measurement
                warnings_local.append(
                    f"Recovered-face surface classification replaced local measurement for {cad_bend_dict.get('bend_id')} "
                    f"({current_gap:.2f}deg -> {recovered_gap:.2f}deg)."
                )
    _phase(f"measure_all_scan_bends end measurements={len(measurements)}")

    matcher_ctor = _filter_kwargs(ProgressiveBendMatcher.__init__, matcher_kwargs or {})
    matcher_ctor["position_mode"] = "enabled"
    matcher = ProgressiveBendMatcher(**matcher_ctor)

    measurement_entries = []
    aligned_tree = cKDTree(aligned_points) if len(aligned_points) else None
    for cad_bend_dict, measurement in zip(cad_bend_dicts, measurements):
        center = np.asarray(cad_bend_dict["bend_center"], dtype=np.float64)
        observability = _local_measurement_observability(
            measurement,
            float(cad_bend_dict["bend_angle"]),
            scan_state=scan_state,
            alignment_metadata=alignment_metadata,
        )
        measurement_entries.append(
            {
                "center": center,
                "cad_angle": float(cad_bend_dict["bend_angle"]),
                "cad_normals": (
                    np.asarray(cad_bend_dict["surface1_normal"], dtype=np.float64),
                    np.asarray(cad_bend_dict["surface2_normal"], dtype=np.float64),
                ),
                "cad_bend_dict": cad_bend_dict,
                "measurement": measurement,
                "observability": observability,
            }
        )

    assignment_plans: List[Dict[str, Any]] = []
    for cad_bend in cad_bends:
        assignment_candidates, selected_candidate, null_assignment_score, best_idx = _select_local_assignment_candidate(
            cad_bend,
            measurement_entries,
            used_measurements=set(),
        )
        assignment_plans.append(
            {
                "cad_bend": cad_bend,
                "assignment_candidates": assignment_candidates,
                "selected_candidate": selected_candidate,
                "null_assignment_score": null_assignment_score,
                "best_idx": best_idx,
            }
        )

    assignment_plans = _resolve_local_assignment_conflicts(assignment_plans)

    matches: List[BendMatch] = []
    for plan in assignment_plans:
        cad_bend = plan["cad_bend"]
        cad_mid = (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0
        assignment_candidates = list(plan.get("assignment_candidates") or [])
        selected_candidate = dict(plan.get("selected_candidate") or {})
        null_assignment_score = float(plan.get("null_assignment_score") or 0.0)
        best_idx = plan.get("best_idx")
        if best_idx is None:
            support_measurement = (
                _support_only_measurement(aligned_tree, aligned_points, cad_bend)
                if aligned_tree is not None
                else {
                    "success": False,
                    "measured_angle": None,
                    "measurement_method": "support_only_probe",
                    "measurement_confidence": "very_low",
                    "point_count": 0,
                    "nearby_points": 0,
                    "side1_points": 0,
                    "side2_points": 0,
                }
            )
            observability = _local_measurement_observability(
                support_measurement,
                cad_bend.target_angle,
                scan_state=scan_state,
                alignment_metadata=alignment_metadata,
            )
            fallback_frame = {
                "local_frame": {
                    "center": [round(float(x), 4) for x in cad_mid.tolist()],
                    "tangent": [round(float(x), 6) for x in np.asarray(cad_bend.bend_line_direction, dtype=np.float64).tolist()],
                },
                "parent_frame": alignment_metadata,
                "search_radius_mm": 35.0,
                "point_count": int(support_measurement.get("point_count") or 0),
                "side1_points": int(support_measurement.get("side1_points") or 0),
                "side2_points": int(support_measurement.get("side2_points") or 0),
                "measurement_method": support_measurement.get("measurement_method"),
                "observability_state": observability["state"],
                "observability_detail_state": observability.get("detail_state", "UNOBSERVED"),
                "visibility_score": round(float(observability["visibility_score"]), 4),
                "visibility_score_source": observability["visibility_score_source"],
                "surface_visibility_ratio": round(float(observability["surface_visibility_ratio"]), 4),
                "local_support_score": round(float(observability["local_support_score"]), 4),
                "side_balance_score": round(float(observability["side_balance_score"]), 4),
                "local_evidence_score": round(float(observability["evidence_score"]), 4),
                "assignment_candidates": assignment_candidates,
                "selected_assignment_candidate_id": selected_candidate.get("candidate_id"),
                "selected_assignment_candidate_kind": selected_candidate.get("candidate_kind"),
                "selected_assignment_candidate_score": selected_candidate.get("assignment_score"),
                "null_assignment_candidate_score": round(null_assignment_score, 4),
            }
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item="Dense-scan local verification found no evidence at this CAD bend location.",
                    physical_completion_state="UNKNOWN",
                    observability_state=observability["state"],
                    observability_detail_state=observability.get("detail_state", "UNOBSERVED"),
                    observability_confidence=observability["confidence"],
                    visibility_score=observability["visibility_score"],
                    visibility_score_source=observability["visibility_score_source"],
                    surface_visibility_ratio=observability["surface_visibility_ratio"],
                    local_support_score=observability["local_support_score"],
                    side_balance_score=observability["side_balance_score"],
                    assignment_source="NONE",
                    assignment_confidence=observability["confidence"],
                    assignment_candidate_id=str(selected_candidate.get("candidate_id") or "null_candidate"),
                    assignment_candidate_kind=str(selected_candidate.get("candidate_kind") or "NULL"),
                    assignment_candidate_score=float(selected_candidate.get("assignment_score") or 0.0),
                    assignment_null_score=null_assignment_score,
                    assignment_candidate_count=len(assignment_candidates),
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["support_point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_context=fallback_frame,
                )
            )
            continue

        entry = measurement_entries[best_idx]
        measurement = entry["measurement"]
        measurement_frame = _local_bend_frame(cad_bend_dicts[best_idx], cad_bend)
        observability = entry.get("observability") or _local_measurement_observability(
            measurement,
            cad_bend.target_angle,
            scan_state=scan_state,
            alignment_metadata=alignment_metadata,
        )
        measurement_context = {
            "local_frame": measurement_frame,
            "parent_frame": alignment_metadata,
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
            "slice_count": measurement.get("slice_count"),
            "planned_slice_count": measurement.get("planned_slice_count"),
            "effective_slice_count": measurement.get("effective_slice_count"),
            "slice_angle_samples_deg": measurement.get("slice_angle_samples_deg"),
            "aggregate_angle_deg": measurement.get("aggregate_angle_deg"),
            "aggregate_radius_mm": measurement.get("aggregate_radius_mm"),
            "angle_dispersion_deg": measurement.get("angle_dispersion_deg"),
            "visible_span_mm": measurement.get("visible_span_mm"),
            "visible_span_ratio": measurement.get("visible_span_ratio"),
            "supported_span_start_mm": measurement.get("supported_span_start_mm"),
            "supported_span_end_mm": measurement.get("supported_span_end_mm"),
            "section_method": measurement.get("section_method"),
            "observability_state": observability["state"],
            "observability_detail_state": observability.get("detail_state", "UNOBSERVED"),
            "visibility_score": round(float(observability["visibility_score"]), 4),
            "visibility_score_source": observability["visibility_score_source"],
            "surface_visibility_ratio": round(float(observability["surface_visibility_ratio"]), 4),
            "local_support_score": round(float(observability["local_support_score"]), 4),
            "side_balance_score": round(float(observability["side_balance_score"]), 4),
            "local_evidence_score": round(float(observability["evidence_score"]), 4),
            "assignment_candidates": assignment_candidates,
            "selected_assignment_candidate_id": selected_candidate.get("candidate_id"),
            "selected_assignment_candidate_kind": selected_candidate.get("candidate_kind"),
            "selected_assignment_candidate_score": selected_candidate.get("assignment_score"),
            "null_assignment_candidate_score": round(null_assignment_score, 4),
        }
        local_assignment_confidence = _local_assignment_confidence(
            observability,
            selected_candidate,
            assignment_candidates,
            null_assignment_score=null_assignment_score,
            measurement_success=bool(measurement.get("success")) and measurement.get("measured_angle") is not None,
        )
        if not measurement.get("success") or measurement.get("measured_angle") is None:
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=f"Dense-scan local verification failed at this CAD bend location: {measurement.get('error', 'insufficient local evidence')}",
                    physical_completion_state="UNKNOWN",
                    observability_state=observability["state"],
                    observability_detail_state=observability.get("detail_state", "UNOBSERVED"),
                    observability_confidence=observability["confidence"],
                    visibility_score=observability["visibility_score"],
                    visibility_score_source=observability["visibility_score_source"],
                    surface_visibility_ratio=observability["surface_visibility_ratio"],
                    local_support_score=observability["local_support_score"],
                    side_balance_score=observability["side_balance_score"],
                    assignment_source="CAD_LOCAL_NEIGHBORHOOD",
                    assignment_confidence=local_assignment_confidence,
                    assignment_candidate_id=str(selected_candidate.get("candidate_id") or f"local_candidate_{best_idx}"),
                    assignment_candidate_kind=str(selected_candidate.get("candidate_kind") or "MEASUREMENT"),
                    assignment_candidate_score=float(selected_candidate.get("assignment_score") or 0.0),
                    assignment_null_score=null_assignment_score,
                    assignment_candidate_count=len(assignment_candidates),
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context=measurement_context,
                )
            )
            continue

        if observability["state"] == "UNFORMED":
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=(
                        "Scan coverage reaches this bend, but the local angle is still closer "
                        "to a flat/open state than the target bend angle."
                    ),
                    physical_completion_state="NOT_FORMED",
                    observability_state=observability["state"],
                    observability_detail_state=observability.get("detail_state", "OBSERVED_NOT_FORMED"),
                    observability_confidence=observability["confidence"],
                    visibility_score=observability["visibility_score"],
                    visibility_score_source=observability["visibility_score_source"],
                    surface_visibility_ratio=observability["surface_visibility_ratio"],
                    local_support_score=observability["local_support_score"],
                    side_balance_score=observability["side_balance_score"],
                    assignment_source="CAD_LOCAL_NEIGHBORHOOD",
                    assignment_confidence=local_assignment_confidence,
                    assignment_candidate_id=str(selected_candidate.get("candidate_id") or f"local_candidate_{best_idx}"),
                    assignment_candidate_kind=str(selected_candidate.get("candidate_kind") or "MEASUREMENT"),
                    assignment_candidate_score=float(selected_candidate.get("assignment_score") or 0.0),
                    assignment_null_score=null_assignment_score,
                    assignment_candidate_count=len(assignment_candidates),
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context=measurement_context,
                )
            )
            continue

        if (
            str(scan_state or "").strip().lower() == "partial"
            and observability["state"] == "UNKNOWN"
            and str(measurement.get("measurement_method") or "").strip().lower() == "profile_section"
            and not _trusted_parent_frame_metadata(alignment_metadata)
        ):
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=(
                        "Local profile-section evidence is not trusted enough on this partial scan "
                        "without a parent-frame datum. Re-scan or recover a stronger datum island."
                    ),
                    physical_completion_state="UNKNOWN",
                    observability_state=observability["state"],
                    observability_detail_state=observability.get("detail_state", "PARTIALLY_OBSERVED"),
                    observability_confidence=observability["confidence"],
                    visibility_score=observability["visibility_score"],
                    visibility_score_source=observability["visibility_score_source"],
                    surface_visibility_ratio=observability["surface_visibility_ratio"],
                    local_support_score=observability["local_support_score"],
                    side_balance_score=observability["side_balance_score"],
                    assignment_source="CAD_LOCAL_NEIGHBORHOOD",
                    assignment_confidence=local_assignment_confidence,
                    assignment_candidate_id=str(selected_candidate.get("candidate_id") or f"local_candidate_{best_idx}"),
                    assignment_candidate_kind=str(selected_candidate.get("candidate_kind") or "MEASUREMENT"),
                    assignment_candidate_score=float(selected_candidate.get("assignment_score") or 0.0),
                    assignment_null_score=null_assignment_score,
                    assignment_candidate_count=len(assignment_candidates),
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context=measurement_context,
                )
            )
            continue

        measured_angle = float(measurement["measured_angle"])
        local_angle_deviation = abs(measured_angle - float(cad_bend.target_angle))
        local_confidence_label = str(measurement.get("measurement_confidence") or "").strip().lower()
        keep_implausible_probe_completion = (
            local_confidence_label == "very_low"
            and local_angle_deviation > 20.0
            and _allow_probe_region_implausible_completion(
                measurement=measurement,
                cad_bend=cad_bend,
                observability=observability,
                selected_candidate=selected_candidate,
                null_assignment_score=null_assignment_score,
                scan_state=scan_state,
                alignment_metadata=alignment_metadata,
            )
        )
        if local_confidence_label == "very_low" and local_angle_deviation > 20.0 and not keep_implausible_probe_completion:
            matches.append(
                _build_not_detected_match(
                    cad_bend,
                    matcher,
                    action_item=(
                        "Dense-scan local verification found some evidence here, but the fitted angle "
                        "was too implausible to trust. Re-scan or improve local coverage before release."
                    ),
                    physical_completion_state="UNKNOWN",
                    observability_state="UNKNOWN",
                    observability_detail_state="PARTIALLY_OBSERVED" if observability["visibility_score"] >= 0.2 else observability.get("detail_state", "UNOBSERVED"),
                    observability_confidence=observability["confidence"],
                    visibility_score=observability["visibility_score"],
                    visibility_score_source=observability["visibility_score_source"],
                    surface_visibility_ratio=observability["surface_visibility_ratio"],
                    local_support_score=observability["local_support_score"],
                    side_balance_score=observability["side_balance_score"],
                    assignment_source="CAD_LOCAL_NEIGHBORHOOD",
                    assignment_confidence=local_assignment_confidence,
                    assignment_candidate_id=str(selected_candidate.get("candidate_id") or f"local_candidate_{best_idx}"),
                    assignment_candidate_kind=str(selected_candidate.get("candidate_kind") or "MEASUREMENT"),
                    assignment_candidate_score=float(selected_candidate.get("assignment_score") or 0.0),
                    assignment_null_score=null_assignment_score,
                    assignment_candidate_count=len(assignment_candidates),
                    observed_surface_count=observability["observed_surface_count"],
                    local_point_count=observability["point_count"],
                    local_evidence_score=observability["evidence_score"],
                    measurement_mode="cad_local_neighborhood",
                    measurement_method=measurement.get("measurement_method"),
                    measurement_context={
                        **measurement_context,
                        "rejected_as_implausible": True,
                        "implausible_angle_deviation_deg": round(local_angle_deviation, 3),
                    },
                )
            )
            continue

        confidence = local_assignment_confidence
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
            measured_angle=measured_angle,
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
        matches[-1].observability_detail_state = observability.get("detail_state", "OBSERVED_FORMED")
        matches[-1].observability_confidence = observability["confidence"]
        matches[-1].physical_completion_state = "FORMED"
        matches[-1].assignment_source = "CAD_LOCAL_NEIGHBORHOOD"
        matches[-1].assignment_confidence = confidence
        matches[-1].assignment_candidate_id = str(selected_candidate.get("candidate_id") or f"local_candidate_{best_idx}")
        matches[-1].assignment_candidate_kind = str(selected_candidate.get("candidate_kind") or "MEASUREMENT")
        matches[-1].assignment_candidate_score = float(selected_candidate.get("assignment_score") or 0.0)
        matches[-1].assignment_null_score = null_assignment_score
        matches[-1].assignment_candidate_count = len(assignment_candidates)
        matches[-1].visibility_score = observability["visibility_score"]
        matches[-1].visibility_score_source = observability["visibility_score_source"]
        matches[-1].surface_visibility_ratio = observability["surface_visibility_ratio"]
        matches[-1].local_support_score = observability["local_support_score"]
        matches[-1].side_balance_score = observability["side_balance_score"]
        matches[-1].observed_surface_count = observability["observed_surface_count"]
        matches[-1].local_point_count = observability["point_count"]
        matches[-1].local_evidence_score = observability["evidence_score"]
        matches[-1].measurement_mode = "cad_local_neighborhood"
        matches[-1].measurement_method = str(measurement.get("measurement_method") or "unknown")
        if keep_implausible_probe_completion:
            matches[-1].action_item = (
                "Counted as formed from strong local probe evidence, but the fitted angle is implausible. "
                "Treat this as completion evidence only and re-scan for trustworthy angle metrology."
            )
            matches[-1].measurement_context = {
                **measurement_context,
                "implausible_probe_completion_override": True,
                "implausible_angle_deviation_deg": round(local_angle_deviation, 3),
            }
        else:
            matches[-1].measurement_context = measurement_context

    if cad_vertices is not None and cad_triangles is not None and len(matches) == len(cad_bends):
        cad_bend_by_id = {bend.bend_id: bend for bend in cad_bends}
        all_face_centers, all_face_normals = _compute_triangle_centers_and_normals(cad_vertices, cad_triangles)
        for idx, match in enumerate(matches):
            cad_bend = cad_bend_by_id.get(match.cad_bend.bend_id)
            if cad_bend is None:
                continue
            if str(match.assignment_source or "") != "CAD_LOCAL_NEIGHBORHOOD":
                continue
            if str(match.measurement_method or "") != "profile_section":
                continue
            target_angle = abs(float(cad_bend.target_angle))
            if not (105.0 <= target_angle <= 140.0):
                continue
            line_length = float(np.linalg.norm(
                np.asarray(cad_bend.bend_line_end, dtype=np.float64)
                - np.asarray(cad_bend.bend_line_start, dtype=np.float64)
            ))
            if line_length > 70.0:
                continue

            recovered_measurement = measure_scan_bend_via_surface_classification(
                scan_points=aligned_points,
                cad_vertices=cad_vertices,
                cad_triangles=cad_triangles,
                surf1_face_indices=np.empty((0,), dtype=np.int32),
                surf2_face_indices=np.empty((0,), dtype=np.int32),
                cad_bend_center=np.asarray(
                    (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0,
                    dtype=np.float64,
                ),
                cad_surf1_normal=np.asarray(
                    cad_bend.flange1_normal if cad_bend.flange1_normal is not None else np.array([1.0, 0.0, 0.0]),
                    dtype=np.float64,
                ),
                cad_surf2_normal=np.asarray(
                    cad_bend.flange2_normal if cad_bend.flange2_normal is not None else np.array([0.0, 0.0, 1.0]),
                    dtype=np.float64,
                ),
                cad_bend_angle=float(cad_bend.target_angle),
                search_radius=24.0,
                min_offset_mm=5.0,
                cad_bend_direction=np.asarray(cad_bend.bend_line_direction, dtype=np.float64),
                cad_bend_line_start=np.asarray(cad_bend.bend_line_start, dtype=np.float64),
                cad_bend_line_end=np.asarray(cad_bend.bend_line_end, dtype=np.float64),
                allow_line_locality=True,
                allow_face_recovery=True,
                all_face_centers=all_face_centers,
                all_face_normals=all_face_normals,
            )
            if not bool(recovered_measurement.get("success")) or recovered_measurement.get("measured_angle") is None:
                continue

            current_angle = match.detected_bend.measured_angle if match.detected_bend is not None else None
            if current_angle is None:
                continue
            current_gap = abs(float(current_angle) - target_angle)
            recovered_gap = abs(float(recovered_measurement["measured_angle"]) - target_angle)
            if recovered_gap + 0.5 >= current_gap:
                continue

            recovered_confidence = _measurement_confidence_score(recovered_measurement.get("measurement_confidence"))
            recovered_point_count = int(
                recovered_measurement.get("point_count")
                or (recovered_measurement.get("side1_points", 0) + recovered_measurement.get("side2_points", 0))
            )
            recovered_plane1 = _placeholder_plane(
                1,
                (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0,
                cad_bend.flange1_normal,
                cad_bend.flange1_area,
                recovered_point_count // 2,
            )
            recovered_plane2 = _placeholder_plane(
                2,
                (cad_bend.bend_line_start + cad_bend.bend_line_end) / 2.0,
                cad_bend.flange2_normal,
                cad_bend.flange2_area,
                recovered_point_count // 2,
            )
            recovered_detected = DetectedBend(
                bend_id=f"L{cad_bend.bend_id}",
                measured_angle=float(recovered_measurement["measured_angle"]),
                measured_radius=float(cad_bend.target_radius),
                bend_line_start=np.asarray(cad_bend.bend_line_start, dtype=np.float64),
                bend_line_end=np.asarray(cad_bend.bend_line_end, dtype=np.float64),
                bend_line_direction=np.asarray(cad_bend.bend_line_direction, dtype=np.float64),
                confidence=recovered_confidence,
                flange1=recovered_plane1,
                flange2=recovered_plane2,
                inlier_count=max(1, recovered_point_count),
                radius_fit_error=0.0,
            )
            replacement = matcher._build_match(
                cad_bend=cad_bend,
                detected=recovered_detected,
                cost=max(0.0, 1.0 - recovered_confidence),
                use_position_cost=True,
            )
            replacement.observability_state = match.observability_state
            replacement.observability_detail_state = match.observability_detail_state
            replacement.observability_confidence = match.observability_confidence
            replacement.physical_completion_state = match.physical_completion_state
            replacement.assignment_source = match.assignment_source
            replacement.assignment_confidence = recovered_confidence
            replacement.assignment_candidate_id = match.assignment_candidate_id
            replacement.assignment_candidate_kind = match.assignment_candidate_kind
            replacement.assignment_candidate_score = match.assignment_candidate_score
            replacement.assignment_null_score = match.assignment_null_score
            replacement.assignment_candidate_count = match.assignment_candidate_count
            replacement.visibility_score = match.visibility_score
            replacement.visibility_score_source = match.visibility_score_source
            replacement.surface_visibility_ratio = match.surface_visibility_ratio
            replacement.local_support_score = match.local_support_score
            replacement.side_balance_score = match.side_balance_score
            replacement.observed_surface_count = match.observed_surface_count
            replacement.local_point_count = recovered_point_count
            replacement.local_evidence_score = match.local_evidence_score
            replacement.measurement_mode = match.measurement_mode
            replacement.measurement_method = str(recovered_measurement.get("measurement_method") or "surface_classification")
            replacement.measurement_context = {
                **(match.measurement_context or {}),
                "measurement_method": recovered_measurement.get("measurement_method"),
                "measurement_confidence_label": recovered_measurement.get("measurement_confidence"),
                "point_count": recovered_point_count,
                "side1_points": int(recovered_measurement.get("side1_points") or 0),
                "side2_points": int(recovered_measurement.get("side2_points") or 0),
                "cad_alignment1": recovered_measurement.get("cad_alignment1"),
                "cad_alignment2": recovered_measurement.get("cad_alignment2"),
                "planarity1": recovered_measurement.get("planarity1"),
                "planarity2": recovered_measurement.get("planarity2"),
                "recovered_face_surface_classification": True,
                "recovered_angle_deg": round(float(recovered_measurement["measured_angle"]), 3),
                "recovered_angle_gap_deg": round(float(recovered_gap), 3),
                "replaced_profile_section_angle_deg": round(float(current_angle), 3),
            }
            if replacement.warnings is None:
                replacement.warnings = []
            replacement.warnings.append(
                f"Recovered-face surface classification replaced profile section "
                f"({current_gap:.2f}deg -> {recovered_gap:.2f}deg CAD gap)."
            )
            matches[idx] = replacement

    report = BendInspectionReport(part_id=part_id, matches=matches, unmatched_detections=[])
    report.compute_summary()
    setattr(report, "local_alignment_fitness", fitness)
    setattr(report, "local_alignment_seed", selected_seed)
    setattr(report, "local_alignment_rmse", rmse)
    setattr(report, "alignment_metadata", alignment_metadata)
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
    scan_state: Optional[str] = None,
    scan_geometry_kind: Optional[str] = None,
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
        scan_state=scan_state,
        fallback=False,
        scan_geometry_kind=scan_geometry_kind,
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
        scan_state=scan_state,
    )

    # Trigger fallback only when recovery potential is meaningful.
    should_try_fallback = (
        len(cad_bends) > 0
        and len(scan_points) >= 2000
        and not dense_scan
        and int(primary_report.detected_count) < max(1, int(np.ceil(0.75 * target)))
    )
    if should_try_fallback and str(scan_state or "").strip().lower() == "partial" and int(primary_report.detected_count) == 0:
        should_try_fallback = False
    if not should_try_fallback:
        return primary_report, int(len(primary_detected)), False

    fb_ctor, fb_call = _fallback_scan_detection_config(runtime_config)
    fb_ctor, fb_call, _ = _apply_scan_runtime_profile(
        fb_ctor,
        fb_call,
        len(scan_points),
        scan_state=scan_state,
        fallback=True,
        scan_geometry_kind=scan_geometry_kind,
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
    expected_bend_override: Optional[int] = None,
    scan_state: Optional[str] = None,
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
    feature_policies = load_part_feature_policies()
    expected_bend_hint = (
        int(expected_bend_override)
        if expected_bend_override is not None and int(expected_bend_override) > 0
        else overrides.get(part_id)
    )
    feature_policy = feature_policies.get(part_id)

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
        feature_policy=feature_policy,
        part_id=part_id,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] extract_cad_bend_specs end "
        f"strategy={cad_strategy} bends={len(cad_bends)}"
    )
    cad_bend_count_extracted = len(cad_bends)
    if not cad_bends:
        warnings.append("No bends extracted from CAD")
    if feature_policy:
        cad_bends = _apply_part_feature_policy(cad_bends, feature_policy)
        warnings.append(f"Applied part feature policy for {part_id}")
    expected_bend_count = int(expected_bend_hint if expected_bend_hint is not None else len(cad_bends))
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
    # Preserve CAD geometry for reference mesh export (same coordinate system
    # as bend detection coordinates). Stored in BendInspectionRunDetails.
    _saved_cad_vertices = cad_vertices
    _saved_cad_triangles = cad_triangles
    del cad_vertices
    del cad_triangles
    gc.collect()

    _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] load_scan_points begin")
    scan_points = load_scan_points(scan_path)
    scan_geometry_kind = classify_scan_geometry_kind(scan_path)
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] load_scan_points end scan_points={len(scan_points)}"
    )
    primary_profile = _scan_runtime_profile_name_for_state(len(scan_points), scan_state, fallback=False)
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
        scan_state=scan_state,
        scan_geometry_kind=scan_geometry_kind,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] detect_and_match_with_fallback end "
        f"detected={report.detected_count} pass={report.pass_count} fallback_used={fallback_used}"
    )
    dense_local_refinement_decision: Optional[str] = "not_applicable"
    dense_local_refinement_report_ranks: Optional[Dict[str, List[int]]] = None
    dense_local_refinement_geometry_penalties: Optional[Dict[str, Optional[float]]] = None
    selected_local_alignment_seed: Optional[int] = None
    selected_local_alignment_fitness: Optional[float] = None
    selected_local_alignment_rmse: Optional[float] = None
    should_run_local_refinement = _dense_scan_requires_local_refinement(
        scan_points=scan_points,
        report=report,
        expected_bend_count=expected_bend_count,
        scan_state=scan_state,
    ) or _feature_policy_requires_local_refinement(
        cad_bends=cad_bends,
        report=report,
        feature_policy=feature_policy,
    )
    if should_run_local_refinement:
        _emit_phase_log(f"run_progressive_bend_inspection[{part_id}] dense local refinement begin")
        local_report, local_warnings = refine_dense_scan_bends_via_alignment(
            cad_path=cad_path,
            scan_path=scan_path,
            cad_bends=cad_bends,
            part_id=part_id,
            tolerance_angle=tolerance_angle,
            scan_state=scan_state,
            matcher_kwargs=cfg.matcher_kwargs,
            local_refinement_kwargs=cfg.local_refinement_kwargs,
            feature_policy=feature_policy,
        )
        warnings.extend(local_warnings)
        _emit_phase_log(
            f"run_progressive_bend_inspection[{part_id}] dense local refinement end "
            f"local_report={'yes' if local_report is not None else 'no'}"
        )
        if local_report is not None:
            local_alignment_fitness = getattr(local_report, "local_alignment_fitness", None)
            local_alignment_seed = getattr(local_report, "local_alignment_seed", None)
            local_alignment_rmse = getattr(local_report, "local_alignment_rmse", None)
            local_alignment_metadata = getattr(local_report, "alignment_metadata", None)
            merged_report = merge_bend_reports(
                report,
                local_report,
                part_id=part_id,
                local_alignment_fitness=local_alignment_fitness,
                local_alignment_metadata=local_alignment_metadata,
                feature_policy=feature_policy,
            )
            current_rank = _report_rank(report, expected_bend_count)
            local_rank = _report_rank(local_report, expected_bend_count)
            merged_rank = _report_rank(merged_report, expected_bend_count)
            current_geometry_penalty = _report_short_obtuse_geometry_penalty(report)
            local_geometry_penalty = _report_short_obtuse_geometry_penalty(local_report)
            merged_geometry_penalty = _report_short_obtuse_geometry_penalty(merged_report)
            dense_local_refinement_report_ranks = {
                "primary": list(current_rank),
                "local": list(local_rank),
                "merged": list(merged_rank),
            }
            dense_local_refinement_geometry_penalties = {
                "primary": round(float(current_geometry_penalty), 4) if np.isfinite(current_geometry_penalty) else None,
                "local": round(float(local_geometry_penalty), 4) if np.isfinite(local_geometry_penalty) else None,
                "merged": round(float(merged_geometry_penalty), 4) if np.isfinite(merged_geometry_penalty) else None,
            }
            selected_local_alignment_seed = local_alignment_seed
            selected_local_alignment_fitness = local_alignment_fitness
            selected_local_alignment_rmse = local_alignment_rmse
            if str((feature_policy or {}).get("countable_source") or "").strip().lower() == "legacy_folded_hybrid":
                report = merged_report
                selected_scan_bend_count = int(merged_report.detected_count)
                dense_local_refinement_decision = "hybrid_authoritative_local"
                warnings.append("Dense-scan local refinement is authoritative for hybrid rolled countable bends")
            else:
                if _should_promote_dense_local_merge(
                    report,
                    merged_report,
                    expected_bend_count,
                ):
                    report = merged_report
                    selected_scan_bend_count = int(merged_report.detected_count)
                    dense_local_refinement_decision = "merged_over_primary_and_local"
                    warnings.append("Dense-scan local refinement merged with primary global detection")
                elif (
                    merged_rank == current_rank
                    and np.isfinite(current_geometry_penalty)
                    and np.isfinite(merged_geometry_penalty)
                    and merged_geometry_penalty + 5.0 < current_geometry_penalty
                ):
                    report = merged_report
                    selected_scan_bend_count = int(merged_report.detected_count)
                    dense_local_refinement_decision = "merged_tie_break_geometry"
                    warnings.append(
                        "Dense-scan local refinement replaced tied primary report on short-obtuse geometry plausibility"
                    )
                elif _should_promote_dense_local_report(report, local_report, expected_bend_count):
                    report = local_report
                    selected_scan_bend_count = int(local_report.detected_count)
                    dense_local_refinement_decision = "local_over_primary"
                    warnings.append("Dense-scan local refinement promoted over primary global detection")
                else:
                    dense_local_refinement_decision = "primary_retained"
        else:
            dense_local_refinement_decision = "local_unavailable"
    if selected_scan_bend_count == 0:
        warnings.append("No bends detected in scan")
    if fallback_used:
        warnings.append("Applied high-sensitivity scan fallback for under-recovered bends")
        fallback_profile = _scan_runtime_profile_name_for_state(len(scan_points), scan_state, fallback=True)
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
        scan_state=str(scan_state or "unknown"),
        scan_quality_status=str(scan_quality.get("status", "UNKNOWN")),
        scan_coverage_pct=float(scan_quality.get("coverage_pct", 0.0)),
        scan_density_pts_per_cm2=float(scan_quality.get("density_pts_per_cm2", 0.0)),
        dense_local_refinement_decision=dense_local_refinement_decision,
        dense_local_refinement_report_ranks=dense_local_refinement_report_ranks,
        dense_local_refinement_geometry_penalties=dense_local_refinement_geometry_penalties,
        local_alignment_seed=selected_local_alignment_seed,
        local_alignment_fitness=selected_local_alignment_fitness,
        local_alignment_rmse=selected_local_alignment_rmse,
        cad_vertices=_saved_cad_vertices,
        cad_triangles=_saved_cad_triangles,
    )
    _emit_phase_log(
        f"run_progressive_bend_inspection[{part_id}] complete "
        f"detected={report.detected_count} expected={expected_bend_count} "
        f"scan_quality={details.scan_quality_status} elapsed_ms={report.processing_time_ms:.1f}"
    )
    return report, details
