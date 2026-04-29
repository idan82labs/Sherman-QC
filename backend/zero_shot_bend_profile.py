from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from feature_detection.bend_detector import BendDetector, DetectedBend
from zero_shot_bend_overlay import build_renderable_bend_objects


SCAN_EXTS = {".ply", ".pcd", ".stl", ".obj"}
PERTURBATION_VOXELS_MM = (0.8, 1.2)
REPEATED_90_FAMILY_TOLERANCE_DEG = 12.0
DOMINANT_FAMILY_STRONG_MASS = 0.65
ORTHOGONAL_DENSE_DOMINANT_MASS = 0.8
ORTHOGONAL_DENSE_DIRECTION_SIMILARITY = 0.99
ORTHOGONAL_DENSE_LENGTH_DELTA_MM = 50.0
ORTHOGONAL_DENSE_ORTHOGONAL_SEP_MM = 40.0
ORTHOGONAL_DENSE_SINGLE_SCALE_MIN = 7
POLICY_MARGIN_MIN = 0.1
EXACT_COUNT_CONFIDENCE_MIN = 0.75
EXACT_COUNT_STABILITY_MIN = 0.7
EXACT_COUNT_IDENTITY_STABILITY_MIN = 0.8
EXACT_COUNT_MODAL_SUPPORT_MIN = 2.0 / 3.0
STRIP_SPACING_SCALE = 6.0
STRIP_AXIS_ALIGNMENT_MIN = 0.985
STRIP_ANGLE_TOLERANCE_DEG = 12.0
STRIP_OVERLAP_GAP_SCALE = 4.0
SMALL_BEND_DOMINANT_MASS_MIN = 0.6
SMALL_BEND_MULTISCALE_GAIN_MIN = 4
SMALL_BEND_UNION_MIN = 8
SMALL_BEND_SUPPORT_SCORE_MIN = 0.35
SMALL_BEND_COUNT_DELTA_MIN = 3
SMALL_BEND_IDENTITY_MIN = 0.72
SMALL_BEND_MODAL_SUPPORT_MIN = 2.0 / 3.0
SMALL_BEND_EXACT_SCORE_MIN = 0.4
MIXED_LOWER_BOUND_DOMINANT_MASS_MAX = 0.65
MIXED_LOWER_BOUND_IDENTITY_MIN = 0.72
NEAR_ORTHOGONAL_RESCUE_DOMINANT_MASS_MIN = 0.85
NEAR_ORTHOGONAL_RESCUE_UNION_MIN = 16
NEAR_ORTHOGONAL_RESCUE_GAIN_MIN = 4
NEAR_ORTHOGONAL_RESCUE_IDENTITY_MIN = 0.55
NEAR_ORTHOGONAL_RESCUE_MATCH_FRACTION_MIN = 0.6
NEAR_ORTHOGONAL_EXACT_SCORE_MIN = 0.45
NEAR_ORTHOGONAL_EXACT_IDENTITY_MIN = 0.7
NEAR_ORTHOGONAL_EXACT_MATCH_FRACTION_MIN = 0.85
NEAR_ORTHOGONAL_EXACT_MAX_DROPOUT = 1
NEAR_ORTHOGONAL_EXACT_MAX_SPAN = 2
ROLLED_SUPPORT_RESCUE_SCORE_MIN = 0.65
ROLLED_SUPPORT_RESCUE_IDENTITY_MIN = 0.6
ROLLED_SUPPORT_RESCUE_MATCH_FRACTION_MIN = 0.68
ROLLED_SUPPORT_RESCUE_MIN_WIDTH = 4


@dataclass(frozen=True)
class BendEvidenceAtom:
    atom_id: str
    source_mode: str
    candidate_bend_id: str
    local_axis_direction: List[float]
    local_line_family_id: str
    local_flange_compatibility: Dict[str, Any]
    local_spacing_mm: float
    local_support_patch_id: str
    local_support_span_mm: float
    support_patch_midpoint: List[float]
    support_bbox_mm: List[float]
    local_normal_summary: Dict[str, Any]
    local_curvature_summary: Dict[str, Any]
    null_assignment_eligible: bool
    observability_flags: List[str]
    ambiguity_flags: List[str]
    trace_endpoints: List[List[float]] = field(default_factory=list)
    trace_length_mm: float = 0.0
    trace_continuity_score: float = 0.0
    fold_strength: float = 0.0
    branch_degree: int = 0
    connected_component_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "source_mode": self.source_mode,
            "candidate_bend_id": self.candidate_bend_id,
            "local_axis_direction": [round(float(v), 6) for v in self.local_axis_direction],
            "local_line_family_id": self.local_line_family_id,
            "local_flange_compatibility": _json_safe(self.local_flange_compatibility),
            "local_spacing_mm": round(float(self.local_spacing_mm), 4),
            "local_support_patch_id": self.local_support_patch_id,
            "local_support_span_mm": round(float(self.local_support_span_mm), 4),
            "support_patch_midpoint": [round(float(v), 4) for v in self.support_patch_midpoint],
            "support_bbox_mm": [round(float(v), 4) for v in self.support_bbox_mm],
            "local_normal_summary": _json_safe(self.local_normal_summary),
            "local_curvature_summary": _json_safe(self.local_curvature_summary),
            "null_assignment_eligible": bool(self.null_assignment_eligible),
            "observability_flags": list(self.observability_flags),
            "ambiguity_flags": list(self.ambiguity_flags),
            "trace_endpoints": _json_safe(self.trace_endpoints),
            "trace_length_mm": round(float(self.trace_length_mm), 4),
            "trace_continuity_score": round(float(self.trace_continuity_score), 4),
            "fold_strength": round(float(self.fold_strength), 4),
            "branch_degree": int(self.branch_degree),
            "connected_component_size": int(self.connected_component_size),
        }


@dataclass(frozen=True)
class BendSupportStrip:
    strip_id: str
    strip_family: str
    axis_direction: List[float]
    axis_line_hint: Dict[str, Any]
    adjacent_flange_normals: List[List[float]]
    dominant_angle_deg: float
    radius_mm: Optional[float]
    support_patch_ids: List[str]
    support_point_count: int
    cross_fold_width_mm: float
    along_axis_span_mm: float
    support_bbox_mm: List[float]
    observability_status: str
    merge_provenance: Dict[str, Any]
    stability_score: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strip_id": self.strip_id,
            "strip_family": self.strip_family,
            "axis_direction": [round(float(v), 6) for v in self.axis_direction],
            "axis_line_hint": _json_safe(self.axis_line_hint),
            "adjacent_flange_normals": [
                [round(float(v), 6) for v in normal]
                for normal in self.adjacent_flange_normals
            ],
            "dominant_angle_deg": round(float(self.dominant_angle_deg), 4),
            "radius_mm": None if self.radius_mm is None else round(float(self.radius_mm), 4),
            "support_patch_ids": list(self.support_patch_ids),
            "support_point_count": int(self.support_point_count),
            "cross_fold_width_mm": round(float(self.cross_fold_width_mm), 4),
            "along_axis_span_mm": round(float(self.along_axis_span_mm), 4),
            "support_bbox_mm": [round(float(v), 4) for v in self.support_bbox_mm],
            "observability_status": self.observability_status,
            "merge_provenance": _json_safe(self.merge_provenance),
            "stability_score": round(float(self.stability_score), 4),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class PolicyScore:
    score: float
    estimated_bend_count: int
    estimated_bend_count_range: Tuple[int, int]
    dominant_angle_families_deg: List[float]
    exact_count_eligible: bool
    angle_family_coherence: float
    duplicate_family_suppression: float
    structural_compactness: float
    perturbation_stability: float
    abstain_reasons: List[str] = field(default_factory=list)
    support_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "estimated_bend_count": int(self.estimated_bend_count),
            "estimated_bend_count_range": list(self.estimated_bend_count_range),
            "dominant_angle_families_deg": [round(float(v), 3) for v in self.dominant_angle_families_deg],
            "exact_count_eligible": bool(self.exact_count_eligible),
            "angle_family_coherence": round(float(self.angle_family_coherence), 4),
            "duplicate_family_suppression": round(float(self.duplicate_family_suppression), 4),
            "structural_compactness": round(float(self.structural_compactness), 4),
            "perturbation_stability": round(float(self.perturbation_stability), 4),
            "abstain_reasons": list(self.abstain_reasons),
            "support_breakdown": _json_safe(self.support_breakdown),
        }


@dataclass(frozen=True)
class ZeroShotBendProfileResult:
    scan_path: str
    part_id: Optional[str]
    profile_family: str
    estimated_bend_count: Optional[int]
    estimated_bend_count_range: Tuple[int, int]
    dominant_angle_families_deg: List[float]
    hypothesis_confidence: float
    abstained: bool
    abstain_reasons: List[str]
    candidate_generator_summary: Dict[str, Any]
    policy_scores: Dict[str, PolicyScore]
    bend_evidence_atoms: List[BendEvidenceAtom] = field(default_factory=list)
    bend_support_strips: List[BendSupportStrip] = field(default_factory=list)
    renderable_bend_objects: List[Dict[str, Any]] = field(default_factory=list)
    support_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_path": self.scan_path,
            "part_id": self.part_id,
            "profile_family": self.profile_family,
            "estimated_bend_count": self.estimated_bend_count,
            "estimated_bend_count_range": list(self.estimated_bend_count_range),
            "dominant_angle_families_deg": [round(float(v), 3) for v in self.dominant_angle_families_deg],
            "hypothesis_confidence": round(float(self.hypothesis_confidence), 4),
            "abstained": bool(self.abstained),
            "abstain_reasons": list(self.abstain_reasons),
            "candidate_generator_summary": _json_safe(self.candidate_generator_summary),
            "policy_scores": {name: score.to_dict() for name, score in self.policy_scores.items()},
            "bend_evidence_atoms": [atom.to_dict() for atom in self.bend_evidence_atoms],
            "bend_support_strips": [strip.to_dict() for strip in self.bend_support_strips],
            "renderable_bend_objects": _json_safe(self.renderable_bend_objects),
            "support_breakdown": _json_safe(self.support_breakdown),
        }


def run_zero_shot_bend_profile(
    scan_path: str,
    runtime_config: Any,
    part_id: Optional[str] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> ZeroShotBendProfileResult:
    points, normals = _load_scan_geometry(scan_path)
    detector_kwargs = dict(getattr(runtime_config, "scan_detector_kwargs", {}) or {})
    detector = BendDetector(**detector_kwargs)
    experiment_config = _normalize_experiment_config(experiment_config)

    candidate_bundles = _build_candidate_bundles(
        detector=detector,
        points=points,
        normals=normals,
        detect_call_kwargs=dict(getattr(runtime_config, "scan_detect_call_kwargs", {}) or {}),
    )
    scan_features = _extract_scan_features(
        points=points,
        candidate_bundles=candidate_bundles,
        scan_path=scan_path,
        part_id=part_id,
    )
    policy_scores = _score_policies(scan_features, candidate_bundles, experiment_config=experiment_config)
    result = _select_zero_shot_result(
        scan_path=scan_path,
        part_id=part_id,
        policy_scores=policy_scores,
        scan_features=scan_features,
        candidate_bundles=candidate_bundles,
        experiment_config=experiment_config,
    )
    return result


def _normalize_experiment_config(experiment_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(experiment_config or {})
    method_family = str(payload.get("method_family") or "").strip()
    flags = dict(payload.get("flags") or {})
    if method_family == "fold_contour_diagnostics":
        flags.setdefault("fold_contour_diagnostics", True)
    elif method_family == "trace_informed_evidence_atoms":
        flags.setdefault("fold_contour_diagnostics", True)
        flags.setdefault("trace_informed_atoms", True)
    elif method_family == "trace_aware_strip_grouping":
        flags.setdefault("fold_contour_diagnostics", True)
        flags.setdefault("trace_informed_atoms", True)
        flags.setdefault("trace_aware_strip_grouping", True)
    elif method_family == "deterministic_candidate_graph_pruning":
        flags.setdefault("fold_contour_diagnostics", True)
        flags.setdefault("trace_informed_atoms", True)
        flags.setdefault("trace_aware_strip_grouping", True)
        flags.setdefault("candidate_graph_pruning", True)
    payload["flags"] = {
        key: bool(value)
        for key, value in flags.items()
        if key in {
            "fold_contour_diagnostics",
            "trace_informed_atoms",
            "trace_aware_strip_grouping",
            "candidate_graph_pruning",
        }
    }
    return payload


def _experiment_enabled(experiment_config: Optional[Dict[str, Any]], flag_name: str) -> bool:
    return bool((experiment_config or {}).get("flags", {}).get(flag_name))


def _load_scan_geometry(scan_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import open3d as o3d

    path = Path(scan_path)
    ext = path.suffix.lower()
    if ext not in SCAN_EXTS:
        raise ValueError(f"Unsupported scan format: {ext}")

    if ext in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.has_points():
            points = np.asarray(pcd.points, dtype=np.float64)
            normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
            if len(points) > 0:
                return points, normals

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_vertices():
        points = np.asarray(mesh.vertices, dtype=np.float64)
        if len(points) > 0:
            return points, None

    raise ValueError(f"Failed to load scan geometry: {path.name}")


def _voxel_downsample(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    voxel_size_mm: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size_mm)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size_mm * 3.0, 2.0), max_nn=30)
        )
    ds_points = np.asarray(pcd.points, dtype=np.float64)
    ds_normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None
    return ds_points, ds_normals


def _build_candidate_bundles(
    detector: BendDetector,
    points: np.ndarray,
    normals: Optional[np.ndarray],
    detect_call_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, List[DetectedBend]]]:
    bundles: Dict[str, Dict[str, List[DetectedBend]]] = {}
    bundles["original"] = _run_candidate_generators(detector, points, normals, detect_call_kwargs)
    for voxel_size in PERTURBATION_VOXELS_MM:
        ds_points, ds_normals = _voxel_downsample(points, normals, voxel_size)
        bundles[f"voxel_{voxel_size:g}"] = _run_candidate_generators(detector, ds_points, ds_normals, {"preprocess": False})
    return bundles


def _run_candidate_generators(
    detector: BendDetector,
    points: np.ndarray,
    normals: Optional[np.ndarray],
    detect_call_kwargs: Dict[str, Any],
) -> Dict[str, List[DetectedBend]]:
    base_kwargs = dict(detect_call_kwargs)
    single_kwargs = dict(base_kwargs)
    single_kwargs["multiscale"] = False
    multi_kwargs = dict(base_kwargs)
    multi_kwargs["multiscale"] = True
    return {
        "single_scale": detector.detect_bends(points, normals=normals, **single_kwargs),
        "multiscale": detector.detect_bends(points, normals=normals, **multi_kwargs),
    }


def _extract_scan_features(
    points: np.ndarray,
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    scan_path: str,
    part_id: Optional[str] = None,
) -> Dict[str, Any]:
    bbox_min = points.min(axis=0) if len(points) else np.zeros(3)
    bbox_max = points.max(axis=0) if len(points) else np.zeros(3)
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    aspect_ratios = [
        float(extents[0] / extents[1]),
        float(extents[0] / extents[2]),
        float(extents[1] / extents[2]),
    ]
    single = candidate_bundles["original"]["single_scale"]
    multi = candidate_bundles["original"]["multiscale"]
    union = single + multi
    angles = [float(b.measured_angle) for b in union]
    dominant_center, dominant_mass = _dominant_angle_family(angles)
    line_groups = _cluster_line_families(union, direction_similarity=0.995, length_delta_mm=20.0, orthogonal_sep_mm=12.0)
    duplicate_rate = 0.0
    if union:
        duplicate_rate = max(0.0, float(len(union) - len(line_groups)) / float(len(union)))
    local_spacing_mm = _estimate_local_spacing(points)
    return {
        "scan_path": scan_path,
        "scan_state_hint": _resolve_scan_state_hint(part_id=part_id, scan_path=scan_path),
        "point_count": int(len(points)),
        "local_spacing_mm": float(local_spacing_mm),
        "bbox_mm": [float(v) for v in extents],
        "aspect_ratios": aspect_ratios,
        "dominant_angle_family_deg": dominant_center,
        "dominant_angle_family_mass": dominant_mass,
        "duplicate_line_family_rate": duplicate_rate,
        "single_scale_count": len(single),
        "multiscale_count": len(multi),
        "union_candidate_count": len(union),
        "union_angle_values": angles,
    }


def _score_policies(
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, PolicyScore]:
    return {
        "conservative_macro": _score_conservative_macro(scan_features, candidate_bundles, experiment_config=experiment_config),
        "orthogonal_dense_sheetmetal": _score_orthogonal_dense_sheetmetal(scan_features, candidate_bundles, experiment_config=experiment_config),
        "repeated_90_sheetmetal": _score_repeated_90_sheetmetal(scan_features, candidate_bundles, experiment_config=experiment_config),
        "rolled_or_mixed_transition": _score_rolled_or_mixed_transition(scan_features, candidate_bundles, experiment_config=experiment_config),
    }


def _profile_support_summary(
    bends: Sequence[DetectedBend],
    *,
    source_mode: str,
    spacing_mm: float,
    strip_family: str,
    direction_similarity: float,
    length_delta_mm: float,
    orthogonal_sep_mm: float,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    line_groups = _cluster_line_families(
        bends,
        direction_similarity=direction_similarity,
        length_delta_mm=length_delta_mm,
        orthogonal_sep_mm=orthogonal_sep_mm,
    )
    trace_graph = None
    if _experiment_enabled(experiment_config, "fold_contour_diagnostics") or _experiment_enabled(experiment_config, "trace_informed_atoms"):
        trace_graph = _build_candidate_trace_graph(bends, spacing_mm=spacing_mm)
    evidence_atoms = _build_evidence_atoms(
        bends,
        line_groups=line_groups,
        source_mode=source_mode,
        spacing_mm=spacing_mm,
        strip_family=strip_family,
        trace_graph=trace_graph,
        experiment_config=experiment_config,
    )
    pre_prune_count = len(evidence_atoms)
    if _experiment_enabled(experiment_config, "candidate_graph_pruning"):
        evidence_atoms = _prune_candidate_graph_atoms(evidence_atoms, spacing_mm=spacing_mm)
    strips, duplicate_merge_count = _infer_support_strips(
        evidence_atoms,
        spacing_mm=spacing_mm,
        strip_family=strip_family,
        experiment_config=experiment_config,
    )
    return {
        "line_groups": line_groups,
        "evidence_atoms": evidence_atoms,
        "strips": strips,
        "strip_count": len(strips),
        "duplicate_merge_count": duplicate_merge_count,
        "candidate_trace_diagnostics": None if trace_graph is None else trace_graph["summary"],
        "pre_prune_evidence_atom_count": int(pre_prune_count),
        "post_prune_evidence_atom_count": int(len(evidence_atoms)),
    }


def _effective_strip_count(
    strip_summary: Dict[str, Any],
    *,
    cap_to_line_groups: bool = False,
) -> int:
    strip_count = int(strip_summary.get("strip_count") or 0)
    if not cap_to_line_groups:
        return strip_count
    line_group_count = len(strip_summary.get("line_groups") or [])
    if line_group_count <= 0:
        return strip_count
    return min(strip_count, line_group_count)


def _score_conservative_macro(
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> PolicyScore:
    retained = list(candidate_bundles["original"]["single_scale"])
    original_count = len(retained)
    strip_summary = _profile_support_summary(
        retained,
        source_mode="single_scale",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        strip_family="discrete_fold_strip",
        direction_similarity=0.998,
        length_delta_mm=10.0,
        orthogonal_sep_mm=8.0,
        experiment_config=experiment_config,
    )
    retained_count = _effective_strip_count(strip_summary)
    dominant_angles = _angle_family_centers(_angles_for_bends(retained))
    angle_coherence = _angle_coherence(retained, dominant_angles, tolerance_deg=10.0)
    duplicate_suppression = _duplicate_suppression(original_count, retained_count)
    structural_compactness = _structural_compactness(original_count, retained_count)
    perturbation = _perturbation_summary(
        candidate_bundles,
        "conservative_macro",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        experiment_config=experiment_config,
    )
    stability = _stability_score(perturbation)
    score = _compose_policy_score(
        angle_coherence,
        duplicate_suppression,
        structural_compactness,
        stability,
        retained,
        dominant_angles,
        perturbation,
    )
    exact_eligible = (
        score >= EXACT_COUNT_CONFIDENCE_MIN
        and stability >= EXACT_COUNT_STABILITY_MIN
        and len(dominant_angles) == 1
        and (
            int(perturbation.get("count_span") or 0) == 0
            or _identity_exact_count_eligible(
                perturbation,
                dominant_mass=float(scan_features.get("dominant_angle_family_mass") or 0.0),
            )
        )
    )
    return PolicyScore(
        score=score,
        estimated_bend_count=retained_count,
        estimated_bend_count_range=(min(perturbation["counts"]), max(perturbation["counts"])),
        dominant_angle_families_deg=dominant_angles,
        exact_count_eligible=exact_eligible,
        angle_family_coherence=angle_coherence,
        duplicate_family_suppression=duplicate_suppression,
        structural_compactness=structural_compactness,
        perturbation_stability=stability,
        abstain_reasons=_abstain_reasons(score, perturbation, scan_features["dominant_angle_family_mass"]),
        support_breakdown={
            "retained_candidate_count": retained_count,
            "raw_candidate_count": original_count,
            "line_family_groups": _serialize_line_groups(strip_summary["line_groups"]),
            "bend_evidence_atoms": [atom.to_dict() for atom in strip_summary["evidence_atoms"]],
            "bend_support_strips": [strip.to_dict() for strip in strip_summary["strips"]],
            "bend_evidence_atom_count": len(strip_summary["evidence_atoms"]),
            "strip_duplicate_merge_count": int(strip_summary["duplicate_merge_count"]),
            "candidate_trace_diagnostics": _json_safe(strip_summary.get("candidate_trace_diagnostics")),
            "pre_prune_evidence_atom_count": int(strip_summary.get("pre_prune_evidence_atom_count") or 0),
            "post_prune_evidence_atom_count": int(strip_summary.get("post_prune_evidence_atom_count") or 0),
            "perturbation": perturbation,
        },
    )


def _score_orthogonal_dense_sheetmetal(
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> PolicyScore:
    union = list(candidate_bundles["original"]["single_scale"]) + list(candidate_bundles["original"]["multiscale"])
    dominant_center, dominant_mass = _dominant_angle_family(_angles_for_bends(union))
    dominant_angles: List[float] = [dominant_center] if dominant_center is not None else []
    guard_ok = (
        dominant_center is not None
        and abs(float(dominant_center) - 90.0) <= 15.0
        and dominant_mass >= ORTHOGONAL_DENSE_DOMINANT_MASS
        and int(scan_features.get("single_scale_count") or 0) >= ORTHOGONAL_DENSE_SINGLE_SCALE_MIN
    )
    filtered = list(union)
    if dominant_center is not None:
        filtered = [
            bend
            for bend in union
            if abs(float(bend.measured_angle) - float(dominant_center)) <= REPEATED_90_FAMILY_TOLERANCE_DEG
        ]
    strip_summary = _profile_support_summary(
        filtered,
        source_mode="single_scale+multiscale",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        strip_family="discrete_fold_strip",
        direction_similarity=ORTHOGONAL_DENSE_DIRECTION_SIMILARITY,
        length_delta_mm=ORTHOGONAL_DENSE_LENGTH_DELTA_MM,
        orthogonal_sep_mm=ORTHOGONAL_DENSE_ORTHOGONAL_SEP_MM,
        experiment_config=experiment_config,
    )
    retained_count = _effective_strip_count(strip_summary, cap_to_line_groups=True)
    angle_coherence = _angle_coherence(filtered, dominant_angles, tolerance_deg=REPEATED_90_FAMILY_TOLERANCE_DEG)
    duplicate_suppression = _duplicate_suppression(len(union), retained_count)
    structural_compactness = _structural_compactness(len(union), retained_count)
    perturbation = _perturbation_summary(
        candidate_bundles,
        "orthogonal_dense_sheetmetal",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        experiment_config=experiment_config,
    )
    stability = _stability_score(perturbation)
    score = _compose_policy_score(
        angle_coherence,
        duplicate_suppression,
        structural_compactness,
        stability,
        filtered,
        dominant_angles,
        perturbation,
    )
    if guard_ok:
        score = min(1.0, score + 0.12)
    else:
        score *= 0.5
    exact_eligible = (
        guard_ok
        and len(dominant_angles) == 1
        and (
            (
                score >= EXACT_COUNT_CONFIDENCE_MIN
                and stability >= EXACT_COUNT_STABILITY_MIN
                and int(perturbation.get("count_span") or 0) == 0
            )
            or _identity_exact_count_recovery_eligible(
                score=score,
                stability=stability,
                perturbation=perturbation,
                dominant_mass=float(dominant_mass),
                min_score=0.6,
            )
        )
        and retained_count <= int(scan_features.get("single_scale_count") or 0)
    )
    abstain_reasons = _abstain_reasons(score, perturbation, dominant_mass if dominant_center is not None else 0.0)
    if not guard_ok:
        abstain_reasons.append("orthogonal_family_not_supported")
    return PolicyScore(
        score=score,
        estimated_bend_count=retained_count,
        estimated_bend_count_range=(min(perturbation["counts"]), max(perturbation["counts"])),
        dominant_angle_families_deg=dominant_angles,
        exact_count_eligible=exact_eligible,
        angle_family_coherence=angle_coherence,
        duplicate_family_suppression=duplicate_suppression,
        structural_compactness=structural_compactness,
        perturbation_stability=stability,
        abstain_reasons=sorted(set(abstain_reasons)),
        support_breakdown={
            "raw_candidate_count": len(union),
            "retained_candidate_count": retained_count,
            "dominant_angle_family_mass": round(float(dominant_mass), 4) if dominant_center is not None else 0.0,
            "line_family_groups": _serialize_line_groups(strip_summary["line_groups"]),
            "bend_evidence_atoms": [atom.to_dict() for atom in strip_summary["evidence_atoms"]],
            "bend_support_strips": [strip.to_dict() for strip in strip_summary["strips"]],
            "bend_evidence_atom_count": len(strip_summary["evidence_atoms"]),
            "strip_duplicate_merge_count": int(strip_summary["duplicate_merge_count"]),
            "candidate_trace_diagnostics": _json_safe(strip_summary.get("candidate_trace_diagnostics")),
            "pre_prune_evidence_atom_count": int(strip_summary.get("pre_prune_evidence_atom_count") or 0),
            "post_prune_evidence_atom_count": int(strip_summary.get("post_prune_evidence_atom_count") or 0),
            "perturbation": perturbation,
            "guard_ok": guard_ok,
        },
    )


def _score_repeated_90_sheetmetal(
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> PolicyScore:
    union = list(candidate_bundles["original"]["single_scale"]) + list(candidate_bundles["original"]["multiscale"])
    dominant_center, dominant_mass = _dominant_angle_family(_angles_for_bends(union))
    if dominant_center is None:
        dominant_angles: List[float] = []
        filtered = union
    else:
        dominant_angles = [dominant_center]
        filtered = [
            bend
            for bend in union
            if abs(float(bend.measured_angle) - float(dominant_center)) <= REPEATED_90_FAMILY_TOLERANCE_DEG
        ]
        if dominant_mass < DOMINANT_FAMILY_STRONG_MASS:
            filtered = list(union)
    strip_summary = _profile_support_summary(
        filtered,
        source_mode="single_scale+multiscale",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        strip_family="discrete_fold_strip",
        direction_similarity=0.995,
        length_delta_mm=20.0,
        orthogonal_sep_mm=12.0,
        experiment_config=experiment_config,
    )
    retained_count = _effective_strip_count(strip_summary, cap_to_line_groups=True)
    angle_coherence = _angle_coherence(filtered, dominant_angles, tolerance_deg=REPEATED_90_FAMILY_TOLERANCE_DEG)
    duplicate_suppression = _duplicate_suppression(len(union), retained_count)
    structural_compactness = _structural_compactness(len(union), retained_count)
    perturbation = _perturbation_summary(
        candidate_bundles,
        "repeated_90_sheetmetal",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        experiment_config=experiment_config,
    )
    stability = _stability_score(perturbation)
    score = _compose_policy_score(
        angle_coherence,
        duplicate_suppression,
        structural_compactness,
        stability,
        filtered,
        dominant_angles,
        perturbation,
    )
    exact_eligible = (
        dominant_mass >= DOMINANT_FAMILY_STRONG_MASS
        and len(dominant_angles) == 1
        and (
            (
                score >= EXACT_COUNT_CONFIDENCE_MIN
                and stability >= EXACT_COUNT_STABILITY_MIN
                and int(perturbation.get("count_span") or 0) == 0
            )
            or _identity_exact_count_recovery_eligible(
                score=score,
                stability=stability,
                perturbation=perturbation,
                dominant_mass=float(dominant_mass),
                min_score=0.6,
            )
        )
    )
    abstain_mass = dominant_mass if dominant_center is not None else 0.0
    return PolicyScore(
        score=score,
        estimated_bend_count=retained_count,
        estimated_bend_count_range=(min(perturbation["counts"]), max(perturbation["counts"])),
        dominant_angle_families_deg=dominant_angles,
        exact_count_eligible=exact_eligible,
        angle_family_coherence=angle_coherence,
        duplicate_family_suppression=duplicate_suppression,
        structural_compactness=structural_compactness,
        perturbation_stability=stability,
        abstain_reasons=_abstain_reasons(score, perturbation, abstain_mass),
        support_breakdown={
            "raw_candidate_count": len(union),
            "retained_candidate_count": retained_count,
            "dominant_angle_family_mass": round(float(dominant_mass), 4),
            "line_family_groups": _serialize_line_groups(strip_summary["line_groups"]),
            "bend_evidence_atoms": [atom.to_dict() for atom in strip_summary["evidence_atoms"]],
            "bend_support_strips": [strip.to_dict() for strip in strip_summary["strips"]],
            "bend_evidence_atom_count": len(strip_summary["evidence_atoms"]),
            "strip_duplicate_merge_count": int(strip_summary["duplicate_merge_count"]),
            "candidate_trace_diagnostics": _json_safe(strip_summary.get("candidate_trace_diagnostics")),
            "pre_prune_evidence_atom_count": int(strip_summary.get("pre_prune_evidence_atom_count") or 0),
            "post_prune_evidence_atom_count": int(strip_summary.get("post_prune_evidence_atom_count") or 0),
            "perturbation": perturbation,
        },
    )


def _score_rolled_or_mixed_transition(
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> PolicyScore:
    raw = list(candidate_bundles["original"]["multiscale"]) or list(candidate_bundles["original"]["single_scale"])
    strip_summary = _profile_support_summary(
        raw,
        source_mode="multiscale",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        strip_family="rolled_transition_strip" if len(_angle_family_centers(_angles_for_bends(raw))) >= 2 else "mixed_transition_strip",
        direction_similarity=0.998,
        length_delta_mm=10.0,
        orthogonal_sep_mm=8.0,
        experiment_config=experiment_config,
    )
    retained_count = _effective_strip_count(strip_summary)
    dominant_angles = _angle_family_centers(_angles_for_bends(raw))
    angle_coherence = _angle_coherence(raw, dominant_angles, tolerance_deg=15.0)
    duplicate_suppression = _duplicate_suppression(len(raw), retained_count)
    structural_compactness = _structural_compactness(len(raw), retained_count)
    perturbation = _perturbation_summary(
        candidate_bundles,
        "rolled_or_mixed_transition",
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
        experiment_config=experiment_config,
    )
    stability = _stability_score(perturbation)
    score = _compose_policy_score(
        angle_coherence,
        duplicate_suppression,
        structural_compactness,
        stability,
        raw,
        dominant_angles,
        perturbation,
    )
    exact_eligible = (
        score >= EXACT_COUNT_CONFIDENCE_MIN
        and stability >= EXACT_COUNT_STABILITY_MIN
        and len(dominant_angles) == 1
        and (
            int(perturbation.get("count_span") or 0) == 0
            or _identity_exact_count_eligible(
                perturbation,
                dominant_mass=float(scan_features.get("dominant_angle_family_mass") or 0.0),
            )
        )
    )
    return PolicyScore(
        score=score,
        estimated_bend_count=retained_count,
        estimated_bend_count_range=(min(perturbation["counts"]), max(perturbation["counts"])),
        dominant_angle_families_deg=dominant_angles,
        exact_count_eligible=exact_eligible,
        angle_family_coherence=angle_coherence,
        duplicate_family_suppression=duplicate_suppression,
        structural_compactness=structural_compactness,
        perturbation_stability=stability,
        abstain_reasons=_abstain_reasons(score, perturbation, scan_features["dominant_angle_family_mass"]),
        support_breakdown={
            "raw_candidate_count": len(raw),
            "retained_candidate_count": retained_count,
            "line_family_groups": _serialize_line_groups(strip_summary["line_groups"]),
            "bend_evidence_atoms": [atom.to_dict() for atom in strip_summary["evidence_atoms"]],
            "bend_support_strips": [strip.to_dict() for strip in strip_summary["strips"]],
            "bend_evidence_atom_count": len(strip_summary["evidence_atoms"]),
            "strip_duplicate_merge_count": int(strip_summary["duplicate_merge_count"]),
            "candidate_trace_diagnostics": _json_safe(strip_summary.get("candidate_trace_diagnostics")),
            "pre_prune_evidence_atom_count": int(strip_summary.get("pre_prune_evidence_atom_count") or 0),
            "post_prune_evidence_atom_count": int(strip_summary.get("post_prune_evidence_atom_count") or 0),
            "perturbation": perturbation,
        },
    )


def _perturbation_summary(
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    policy_name: str,
    spacing_mm: float,
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    counts: List[int] = []
    dominant_angles: List[Optional[float]] = []
    duplicate_merges: List[int] = []
    strips_by_variant: Dict[str, List[BendSupportStrip]] = {}
    variant_summaries: Dict[str, Dict[str, Any]] = {}
    for variant, bundle in candidate_bundles.items():
        if policy_name == "conservative_macro":
            bends = list(bundle["single_scale"])
            dominant_center, _ = _dominant_angle_family(_angles_for_bends(bends))
            strip_summary = _profile_support_summary(
                bends,
                source_mode=f"{variant}:single_scale",
                spacing_mm=spacing_mm,
                strip_family="discrete_fold_strip",
                direction_similarity=0.998,
                length_delta_mm=10.0,
                orthogonal_sep_mm=8.0,
                experiment_config=experiment_config,
            )
            count_value = _effective_strip_count(strip_summary)
            variant_summaries[variant] = _perturbation_variant_summary(
                variant=variant,
                raw_bends=bends,
                dominant_center=dominant_center,
                strip_summary=strip_summary,
                effective_count=count_value,
            )
        elif policy_name == "repeated_90_sheetmetal":
            bends = list(bundle["single_scale"]) + list(bundle["multiscale"])
            dominant_center, dominant_mass = _dominant_angle_family(_angles_for_bends(bends))
            if dominant_center is not None and dominant_mass >= DOMINANT_FAMILY_STRONG_MASS:
                bends = [
                    bend
                    for bend in bends
                    if abs(float(bend.measured_angle) - float(dominant_center)) <= REPEATED_90_FAMILY_TOLERANCE_DEG
                ]
            strip_summary = _profile_support_summary(
                bends,
                source_mode=f"{variant}:single_scale+multiscale",
                spacing_mm=spacing_mm,
                strip_family="discrete_fold_strip",
                direction_similarity=0.995,
                length_delta_mm=20.0,
                orthogonal_sep_mm=12.0,
                experiment_config=experiment_config,
            )
            count_value = _effective_strip_count(strip_summary, cap_to_line_groups=True)
            variant_summaries[variant] = _perturbation_variant_summary(
                variant=variant,
                raw_bends=bends,
                dominant_center=dominant_center,
                strip_summary=strip_summary,
                effective_count=count_value,
            )
        elif policy_name == "orthogonal_dense_sheetmetal":
            bends = list(bundle["single_scale"]) + list(bundle["multiscale"])
            dominant_center, dominant_mass = _dominant_angle_family(_angles_for_bends(bends))
            if dominant_center is not None and dominant_mass >= ORTHOGONAL_DENSE_DOMINANT_MASS:
                bends = [
                    bend
                    for bend in bends
                    if abs(float(bend.measured_angle) - float(dominant_center)) <= REPEATED_90_FAMILY_TOLERANCE_DEG
                ]
            strip_summary = _profile_support_summary(
                bends,
                source_mode=f"{variant}:single_scale+multiscale",
                spacing_mm=spacing_mm,
                strip_family="discrete_fold_strip",
                direction_similarity=ORTHOGONAL_DENSE_DIRECTION_SIMILARITY,
                length_delta_mm=ORTHOGONAL_DENSE_LENGTH_DELTA_MM,
                orthogonal_sep_mm=ORTHOGONAL_DENSE_ORTHOGONAL_SEP_MM,
                experiment_config=experiment_config,
            )
            count_value = _effective_strip_count(strip_summary, cap_to_line_groups=True)
            variant_summaries[variant] = _perturbation_variant_summary(
                variant=variant,
                raw_bends=bends,
                dominant_center=dominant_center,
                strip_summary=strip_summary,
                effective_count=count_value,
            )
        else:
            bends = list(bundle["multiscale"]) or list(bundle["single_scale"])
            dominant_center, _ = _dominant_angle_family(_angles_for_bends(bends))
            strip_summary = _profile_support_summary(
                bends,
                source_mode=f"{variant}:multiscale",
                spacing_mm=spacing_mm,
                strip_family="rolled_transition_strip",
                direction_similarity=0.998,
                length_delta_mm=10.0,
                orthogonal_sep_mm=8.0,
                experiment_config=experiment_config,
            )
            count_value = _effective_strip_count(strip_summary)
            variant_summaries[variant] = _perturbation_variant_summary(
                variant=variant,
                raw_bends=bends,
                dominant_center=dominant_center,
                strip_summary=strip_summary,
                effective_count=count_value,
            )
        counts.append(int(count_value))
        dominant_angles.append(dominant_center)
        duplicate_merges.append(int(strip_summary["duplicate_merge_count"]))
        strips_by_variant[variant] = list(strip_summary["strips"])
    modal_count = None
    modal_count_support = 0.0
    if counts:
        modal_values, modal_freq = np.unique(np.asarray(counts, dtype=np.int32), return_counts=True)
        best_idx = int(np.argmax(modal_freq))
        modal_count = int(modal_values[best_idx])
        modal_count_support = float(modal_freq[best_idx]) / float(len(counts))
    match_stats = _perturbation_reference_match_summary(strips_by_variant, spacing_mm=spacing_mm)
    return {
        "counts": counts,
        "dominant_angle_families_deg": [None if v is None else round(float(v), 3) for v in dominant_angles],
        "count_span": (max(counts) - min(counts)) if counts else 0,
        "duplicate_merge_counts": duplicate_merges,
        "modal_count": modal_count,
        "modal_count_support": round(float(modal_count_support), 4),
        "identity_stability": round(float(_perturbation_identity_stability(strips_by_variant, spacing_mm=spacing_mm)), 4),
        "reference_strip_count": int(match_stats["reference_strip_count"]),
        "variant_match_counts": [int(v) for v in match_stats["variant_match_counts"]],
        "matched_reference_fraction_mean": round(float(match_stats["matched_reference_fraction_mean"]), 4),
        "variant_summaries": variant_summaries,
    }


def _perturbation_variant_summary(
    *,
    variant: str,
    raw_bends: Sequence[DetectedBend],
    dominant_center: Optional[float],
    strip_summary: Dict[str, Any],
    effective_count: int,
) -> Dict[str, Any]:
    return {
        "variant": str(variant),
        "raw_candidate_count": int(len(raw_bends)),
        "dominant_angle_family_deg": None if dominant_center is None else round(float(dominant_center), 3),
        "line_family_group_count": int(len(strip_summary.get("line_groups") or [])),
        "evidence_atom_count": int(len(strip_summary.get("evidence_atoms") or [])),
        "strip_count": int(strip_summary.get("strip_count") or 0),
        "effective_count": int(effective_count),
        "duplicate_merge_count": int(strip_summary.get("duplicate_merge_count") or 0),
        "candidate_trace_diagnostics": _json_safe(strip_summary.get("candidate_trace_diagnostics")),
        "pre_prune_evidence_atom_count": int(strip_summary.get("pre_prune_evidence_atom_count") or 0),
        "post_prune_evidence_atom_count": int(strip_summary.get("post_prune_evidence_atom_count") or 0),
    }


def _perturbation_identity_stability(
    strips_by_variant: Dict[str, List[BendSupportStrip]],
    *,
    spacing_mm: float,
) -> float:
    reference = list(strips_by_variant.get("original") or [])
    others = [
        strips
        for name, strips in strips_by_variant.items()
        if name != "original"
    ]
    if not reference and not others:
        return 0.0
    if not others:
        return 1.0
    if not reference:
        return 0.0
    scores = [
        _strip_set_identity_score(reference, other, spacing_mm=spacing_mm)
        for other in others
    ]
    return float(np.mean(scores)) if scores else 0.0


def _perturbation_reference_match_summary(
    strips_by_variant: Dict[str, List[BendSupportStrip]],
    *,
    spacing_mm: float,
) -> Dict[str, Any]:
    reference = list(strips_by_variant.get("original") or [])
    others = [
        strips
        for name, strips in strips_by_variant.items()
        if name != "original"
    ]
    if not reference or not others:
        return {
            "reference_strip_count": len(reference),
            "variant_match_counts": [],
            "matched_reference_fraction_mean": 0.0,
        }
    matched_counts: List[int] = []
    matched_fractions: List[float] = []
    for other in others:
        stats = _strip_set_identity_match_stats(reference, other, spacing_mm=spacing_mm)
        matched_counts.append(int(stats["matched_count"]))
        matched_fractions.append(float(stats["matched_fraction"]))
    return {
        "reference_strip_count": len(reference),
        "variant_match_counts": matched_counts,
        "matched_reference_fraction_mean": float(np.mean(matched_fractions)) if matched_fractions else 0.0,
    }


def _strip_set_identity_score(
    reference: Sequence[BendSupportStrip],
    candidate: Sequence[BendSupportStrip],
    *,
    spacing_mm: float,
) -> float:
    return float(_strip_set_identity_match_stats(reference, candidate, spacing_mm=spacing_mm)["score"])


def _renderable_strip_rank(strip: BendSupportStrip) -> Tuple[float, int, float, float, str]:
    return (
        float(strip.confidence),
        int(strip.support_point_count),
        float(strip.along_axis_span_mm),
        float(strip.stability_score),
        str(strip.strip_id),
    )


def _select_renderable_support_strips(
    strips: Sequence[BendSupportStrip],
    *,
    target_count: Optional[int],
    spacing_mm: float,
) -> List[BendSupportStrip]:
    if target_count is None or target_count <= 0:
        return list(strips)
    if len(strips) <= target_count:
        return list(strips)

    ranked = sorted(strips, key=_renderable_strip_rank, reverse=True)
    selected: List[BendSupportStrip] = []
    deferred: List[BendSupportStrip] = []
    for strip in ranked:
        if any(_strip_identity_match_score(strip, kept, spacing_mm=spacing_mm) >= 0.7 for kept in selected):
            deferred.append(strip)
            continue
        selected.append(strip)
        if len(selected) >= target_count:
            break
    if len(selected) < target_count:
        for strip in deferred:
            if len(selected) >= target_count:
                break
            selected.append(strip)

    def _strip_order_key(strip: BendSupportStrip) -> Tuple[int, str]:
        raw = str(strip.strip_id)
        digits = "".join(ch for ch in raw if ch.isdigit())
        return (int(digits) if digits else 10**9, raw)

    return sorted(selected[:target_count], key=_strip_order_key)


def _strip_set_identity_match_stats(
    reference: Sequence[BendSupportStrip],
    candidate: Sequence[BendSupportStrip],
    *,
    spacing_mm: float,
) -> Dict[str, Any]:
    if not reference and not candidate:
        return {"score": 1.0, "matched_count": 0, "matched_fraction": 1.0}
    if not reference or not candidate:
        return {"score": 0.0, "matched_count": 0, "matched_fraction": 0.0}
    remaining = list(candidate)
    matched_score = 0.0
    matched_count = 0
    for ref_strip in reference:
        best_index = None
        best_score = 0.0
        for index, cand_strip in enumerate(remaining):
            score = _strip_identity_match_score(ref_strip, cand_strip, spacing_mm=spacing_mm)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.65:
            matched_score += best_score
            matched_count += 1
            remaining.pop(best_index)
    return {
        "score": float(matched_score) / float(max(len(reference), len(candidate))),
        "matched_count": int(matched_count),
        "matched_fraction": float(matched_count) / float(len(reference)) if reference else 0.0,
    }


def _strip_identity_match_score(
    lhs: BendSupportStrip,
    rhs: BendSupportStrip,
    *,
    spacing_mm: float,
) -> float:
    lhs_axis = _normalize_direction(np.asarray(lhs.axis_direction, dtype=np.float64))
    rhs_axis = _normalize_direction(np.asarray(rhs.axis_direction, dtype=np.float64))
    axis_dot = abs(float(np.dot(lhs_axis, rhs_axis)))
    if axis_dot < 0.99:
        return 0.0
    angle_delta = abs(float(lhs.dominant_angle_deg) - float(rhs.dominant_angle_deg))
    if angle_delta > 8.0:
        return 0.0
    lhs_mid = np.asarray((lhs.axis_line_hint or {}).get("midpoint") or [0.0, 0.0, 0.0], dtype=np.float64)
    rhs_mid = np.asarray((rhs.axis_line_hint or {}).get("midpoint") or [0.0, 0.0, 0.0], dtype=np.float64)
    midpoint_delta = lhs_mid - rhs_mid
    midpoint_distance = float(np.linalg.norm(midpoint_delta))
    midpoint_limit = max(spacing_mm * 12.0, 20.0)
    if midpoint_distance > midpoint_limit:
        return 0.0
    span_delta = abs(float(lhs.along_axis_span_mm) - float(rhs.along_axis_span_mm))
    span_limit = max(25.0, 0.25 * max(float(lhs.along_axis_span_mm), float(rhs.along_axis_span_mm), 1.0))
    if span_delta > span_limit:
        return 0.0
    axis_score = max(0.0, min(1.0, (axis_dot - 0.99) / 0.01))
    angle_score = max(0.0, 1.0 - angle_delta / 8.0)
    midpoint_score = max(0.0, 1.0 - midpoint_distance / midpoint_limit)
    span_score = max(0.0, 1.0 - span_delta / span_limit)
    return float(0.35 * axis_score + 0.25 * angle_score + 0.25 * midpoint_score + 0.15 * span_score)


def _compose_policy_score(
    angle_coherence: float,
    duplicate_suppression: float,
    structural_compactness: float,
    stability: float,
    bends: Sequence[DetectedBend],
    dominant_angles: Sequence[float],
    perturbation: Dict[str, Any],
) -> float:
    score = (
        0.35 * angle_coherence
        + 0.25 * duplicate_suppression
        + 0.20 * structural_compactness
        + 0.20 * stability
    )
    if _has_odd_isolated_angle_family(bends, dominant_angles):
        score -= 0.2
    if duplicate_suppression < 0.65:
        score -= 0.15
    if int(perturbation.get("count_span") or 0) > 1:
        score -= 0.1
    return max(0.0, min(1.0, float(score)))




def _identity_exact_count_eligible(
    perturbation: Dict[str, Any],
    *,
    dominant_mass: float,
) -> bool:
    return (
        int(perturbation.get("count_span") or 0) <= 1
        and float(perturbation.get("identity_stability") or 0.0) >= EXACT_COUNT_IDENTITY_STABILITY_MIN
        and float(perturbation.get("modal_count_support") or 0.0) >= EXACT_COUNT_MODAL_SUPPORT_MIN
        and float(dominant_mass) >= DOMINANT_FAMILY_STRONG_MASS
    )


def _identity_exact_count_recovery_eligible(
    *,
    score: float,
    stability: float,
    perturbation: Dict[str, Any],
    dominant_mass: float,
    min_score: float,
) -> bool:
    return (
        float(score) >= float(min_score)
        and float(stability) >= EXACT_COUNT_STABILITY_MIN
        and _identity_exact_count_eligible(
            perturbation,
            dominant_mass=dominant_mass,
        )
    )


def _select_zero_shot_result(
    scan_path: str,
    part_id: Optional[str],
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    candidate_bundles: Dict[str, Dict[str, List[DetectedBend]]],
    experiment_config: Optional[Dict[str, Any]] = None,
) -> ZeroShotBendProfileResult:
    ranked = sorted(policy_scores.items(), key=lambda item: item[1].score, reverse=True)
    winner_name, winner = ranked[0]
    runner_up_candidates = ranked[1:]
    orthogonal_dense = policy_scores.get("orthogonal_dense_sheetmetal")
    orthogonal_override = _should_prefer_orthogonal_dense(
        winner_name=winner_name,
        winner=winner,
        orthogonal_dense=orthogonal_dense,
        scan_features=scan_features,
    )
    if orthogonal_override:
        winner_name = "orthogonal_dense_sheetmetal"
        winner = orthogonal_dense
        runner_up_candidates = sorted(
            ((name, score) for name, score in policy_scores.items() if name != winner_name),
            key=lambda item: item[1].score,
            reverse=True,
        )
    runner_up = runner_up_candidates[0][1] if runner_up_candidates else None
    runner_up_score = runner_up.score if runner_up is not None else 0.0
    margin = float(winner.score - runner_up_score)
    same_count_consensus = (
        runner_up is not None
        and winner.exact_count_eligible
        and runner_up.exact_count_eligible
        and winner.estimated_bend_count == runner_up.estimated_bend_count
        and winner.estimated_bend_count_range == runner_up.estimated_bend_count_range
    )
    structural_count_consensus = _structural_count_consensus(policy_scores)
    small_bend_rescue = _small_tight_bend_rescue_consensus(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        margin=margin,
    )
    mixed_lower_bound_rescue = _mixed_angle_lower_bound_rescue(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        margin=margin,
    )
    near_orthogonal_range_rescue = _near_orthogonal_family_range_rescue(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        margin=margin,
    )

    abstain_reasons = list(winner.abstain_reasons)
    abstained = False
    if (
        margin < POLICY_MARGIN_MIN
        and not same_count_consensus
        and structural_count_consensus is None
        and not orthogonal_override
    ):
        abstained = True
        abstain_reasons.append("policy_margin_low")
    if not winner.dominant_angle_families_deg:
        abstained = True
        abstain_reasons.append("dominant_angle_family_weak")
    if winner_name == "conservative_macro" and len(winner.dominant_angle_families_deg) > 1 and not winner.exact_count_eligible:
        abstained = True
        abstain_reasons.append("mixed_angle_family_ambiguous")
    if winner.perturbation_stability < 0.5:
        abstained = True
        abstain_reasons.append("perturbation_unstable")
    if _allow_structural_range_without_count_abstention(
        winner_name=winner_name,
        winner=winner,
        margin=margin,
        scan_features=scan_features,
        abstain_reasons=abstain_reasons,
    ):
        abstained = False
        abstain_reasons = [
            reason for reason in abstain_reasons if reason not in {"policy_confidence_low", "perturbation_count_unstable", "perturbation_unstable"}
        ]

    count_range = winner.estimated_bend_count_range
    exact_count = winner.estimated_bend_count if winner.exact_count_eligible and not abstained else None
    small_bend_exact_recovery = _small_tight_bend_exact_count_recovery(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        small_bend_rescue=small_bend_rescue,
    )
    near_orthogonal_exact_recovery = _near_orthogonal_exact_dropout_recovery(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        near_orthogonal_range_rescue=near_orthogonal_range_rescue,
    )
    rolled_structural_range_rescue = _rolled_structural_support_range_rescue(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
    )
    exact_recovery = small_bend_exact_recovery or near_orthogonal_exact_recovery
    if exact_count is None and exact_recovery is not None:
        exact_count = int(exact_recovery["exact_count"])
        abstained = False
        count_range = (exact_count, exact_count)
        abstain_reasons = [
            reason
            for reason in abstain_reasons
            if reason not in {"policy_confidence_low", "policy_margin_low", "dominant_angle_family_weak", "perturbation_count_unstable", "perturbation_unstable"}
        ]
    profile_family = _reported_profile_family(
        winner_name=winner_name,
        winner=winner,
        policy_scores=policy_scores,
        scan_features=scan_features,
        abstained=abstained,
    )
    if structural_count_consensus is not None and not abstained and exact_count is None:
        count_range = structural_count_consensus
    if rolled_structural_range_rescue is not None and not abstained and exact_count is None:
        count_range = rolled_structural_range_rescue["count_range"]
    winner_perturbation = dict((winner.support_breakdown or {}).get("perturbation") or {})
    winner_perturbation_consensus = None
    if (
        margin >= POLICY_MARGIN_MIN
        and "policy_margin_low" not in abstain_reasons
        and float(scan_features.get("dominant_angle_family_mass") or 0.0) >= 0.9
    ):
        winner_perturbation_consensus = _perturbation_count_consensus_range(winner_perturbation)
    if abstained:
        applied_targeted_rescue = False
        if mixed_lower_bound_rescue is not None:
            count_range = mixed_lower_bound_rescue["count_range"]
            applied_targeted_rescue = True
        elif near_orthogonal_range_rescue is not None:
            count_range = near_orthogonal_range_rescue["count_range"]
            applied_targeted_rescue = True
        elif small_bend_rescue is not None:
            count_range = small_bend_rescue["count_range"]
            applied_targeted_rescue = True
        elif winner_perturbation_consensus is not None:
            count_range = winner_perturbation_consensus
        else:
            plausible_delta = 0.12
            if "mixed_angle_family_ambiguous" in abstain_reasons:
                plausible_delta = 0.5
            plausible = [
                score
                for _, score in ranked
                if float(winner.score - score.score) <= plausible_delta
            ]
            if plausible:
                low = min(score.estimated_bend_count_range[0] for score in plausible)
                high = max(score.estimated_bend_count_range[1] for score in plausible)
                count_range = (low, high)
        if not applied_targeted_rescue and _strong_near_orthogonal_family(scan_features):
            count_range = (max(0, count_range[0] - 1), count_range[1])
        elif not applied_targeted_rescue and count_range[0] == count_range[1]:
            count_range = (max(0, count_range[0] - 1), count_range[1] + 1)

    candidate_summary = {
        "point_count": scan_features["point_count"],
        "local_spacing_mm": round(float(scan_features.get("local_spacing_mm") or 0.0), 4),
        "bbox_mm": scan_features["bbox_mm"],
        "aspect_ratios": scan_features["aspect_ratios"],
        "single_scale_count": scan_features["single_scale_count"],
        "multiscale_count": scan_features["multiscale_count"],
        "union_candidate_count": scan_features["union_candidate_count"],
        "duplicate_line_family_rate": round(float(scan_features["duplicate_line_family_rate"]), 4),
        "strip_duplicate_merge_count": int((winner.support_breakdown or {}).get("strip_duplicate_merge_count") or 0),
        "scan_state_hint": scan_features.get("scan_state_hint"),
        "perturbation_variants": {
            name: {
                "single_scale_count": len(bundle["single_scale"]),
                "multiscale_count": len(bundle["multiscale"]),
            }
            for name, bundle in candidate_bundles.items()
        },
        "experiment_config": _json_safe(experiment_config),
    }
    selected_atoms = [
        _bend_evidence_atom_from_dict(item)
        for item in list((winner.support_breakdown or {}).get("bend_evidence_atoms") or [])
    ]
    selected_strips = [
        _bend_support_strip_from_dict(item)
        for item in list((winner.support_breakdown or {}).get("bend_support_strips") or [])
    ]
    renderable_strips = _select_renderable_support_strips(
        selected_strips,
        target_count=exact_count,
        spacing_mm=float(scan_features.get("local_spacing_mm") or 1.0),
    )
    renderable_bend_objects = build_renderable_bend_objects(
        [strip.to_dict() for strip in renderable_strips],
        profile_family=profile_family,
        part_id=part_id,
    )
    partial_scan_exact_block = (
        str(scan_features.get("scan_state_hint") or "").lower() == "partial"
        and exact_count is not None
    )
    partial_scan_consensus_only = (
        str(scan_features.get("scan_state_hint") or "").lower() == "partial"
        and exact_count is None
        and structural_count_consensus is not None
        and not winner.exact_count_eligible
    )
    partial_scan_low_margin_abstain = (
        str(scan_features.get("scan_state_hint") or "").lower() == "partial"
        and abstained
        and exact_count is None
        and margin < POLICY_MARGIN_MIN
    )
    if partial_scan_consensus_only:
        abstained = True
        abstain_reasons.append("scan_state_partial")
        profile_family = "scan_limited_partial"
    if partial_scan_low_margin_abstain:
        abstain_reasons.append("scan_state_partial")
        profile_family = "scan_limited_partial"
    if partial_scan_exact_block:
        exact_count = None
        abstained = True
        abstain_reasons.append("scan_state_partial")
        profile_family = "scan_limited_partial"
    if selected_strips and all(strip.observability_status == "scan_limited" for strip in selected_strips):
        abstained = True
        abstain_reasons.append("strip_observability_insufficient")
        if profile_family == winner_name:
            profile_family = "scan_limited_partial"
    return ZeroShotBendProfileResult(
        scan_path=scan_path,
        part_id=part_id,
        profile_family=profile_family,
        estimated_bend_count=exact_count,
        estimated_bend_count_range=count_range,
        dominant_angle_families_deg=winner.dominant_angle_families_deg,
        hypothesis_confidence=winner.score,
        abstained=abstained,
        abstain_reasons=sorted(set(abstain_reasons)),
        candidate_generator_summary=candidate_summary,
        policy_scores=policy_scores,
        bend_evidence_atoms=selected_atoms,
        bend_support_strips=selected_strips,
        renderable_bend_objects=renderable_bend_objects,
        support_breakdown={
            "selected_policy": winner_name,
            "selected_policy_margin": round(margin, 4),
            "scan_dominant_angle_family_mass": round(float(scan_features["dominant_angle_family_mass"]), 4),
            "orthogonal_dense_override": orthogonal_override,
            "structural_count_consensus": list(structural_count_consensus) if structural_count_consensus else None,
            "perturbation_count_consensus": list(winner_perturbation_consensus) if winner_perturbation_consensus else None,
            "mixed_lower_bound_rescue": _json_safe(mixed_lower_bound_rescue),
            "near_orthogonal_range_rescue": _json_safe(near_orthogonal_range_rescue),
            "small_bend_rescue_range": list(small_bend_rescue["count_range"]) if small_bend_rescue else None,
            "small_bend_rescue_mode": str(small_bend_rescue["rescue_mode"]) if small_bend_rescue else None,
            "small_bend_rescue_policies": list(small_bend_rescue["source_policies"]) if small_bend_rescue else None,
            "small_bend_exact_recovery": _json_safe(small_bend_exact_recovery),
            "near_orthogonal_exact_recovery": _json_safe(near_orthogonal_exact_recovery),
            "rolled_structural_range_rescue": _json_safe(rolled_structural_range_rescue),
            "bend_support_strip_count": len(selected_strips),
            "renderable_bend_object_count": len(renderable_bend_objects),
            "bend_evidence_atom_count": len(selected_atoms),
            "strip_duplicate_merge_count": int((winner.support_breakdown or {}).get("strip_duplicate_merge_count") or 0),
            "candidate_trace_diagnostics": _json_safe((winner.support_breakdown or {}).get("candidate_trace_diagnostics")),
            "pre_prune_evidence_atom_count": int((winner.support_breakdown or {}).get("pre_prune_evidence_atom_count") or 0),
            "post_prune_evidence_atom_count": int((winner.support_breakdown or {}).get("post_prune_evidence_atom_count") or 0),
            "experiment_config": _json_safe(experiment_config),
        },
    )


def _should_prefer_orthogonal_dense(
    winner_name: str,
    winner: PolicyScore,
    orthogonal_dense: Optional[PolicyScore],
    scan_features: Dict[str, Any],
) -> bool:
    if winner_name != "conservative_macro" or orthogonal_dense is None:
        return False
    if winner.exact_count_eligible:
        return False
    if not orthogonal_dense.support_breakdown.get("guard_ok"):
        return False
    if float(scan_features.get("dominant_angle_family_mass") or 0.0) < ORTHOGONAL_DENSE_DOMINANT_MASS:
        return False
    if float(scan_features.get("duplicate_line_family_rate") or 0.0) < 0.35:
        return False
    if orthogonal_dense.score < 0.55:
        return False
    if orthogonal_dense.perturbation_stability + 0.05 < winner.perturbation_stability:
        return False
    return int(winner.estimated_bend_count) - int(orthogonal_dense.estimated_bend_count) >= 2


def _structural_count_consensus(policy_scores: Dict[str, PolicyScore]) -> Optional[Tuple[int, int]]:
    grouped: Dict[int, List[PolicyScore]] = {}
    for score in policy_scores.values():
        grouped.setdefault(int(score.estimated_bend_count), []).append(score)

    for count, group in grouped.items():
        if len(group) < 3:
            continue
        if any(score.exact_count_eligible for score in group):
            continue
        angle_centers: List[float] = []
        low = max(score.estimated_bend_count_range[0] for score in group)
        high = min(score.estimated_bend_count_range[1] for score in group)
        if low > high:
            continue
        valid_angles = True
        for score in group:
            if len(score.dominant_angle_families_deg) != 1:
                valid_angles = False
                break
            angle_centers.append(float(score.dominant_angle_families_deg[0]))
        if not valid_angles:
            continue
        if max(angle_centers) - min(angle_centers) > 5.0:
            continue
        consensus_low = min(score.estimated_bend_count_range[0] for score in group)
        consensus_high = max(score.estimated_bend_count_range[1] for score in group)
        return (consensus_low, consensus_high)
    return None


def _small_tight_bend_rescue_consensus(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    margin: float,
) -> Optional[Dict[str, Any]]:
    if winner_name != "conservative_macro":
        return None
    if float(margin) >= POLICY_MARGIN_MIN:
        return None
    dominant_center = scan_features.get("dominant_angle_family_deg")
    dominant_mass = float(scan_features.get("dominant_angle_family_mass") or 0.0)
    if dominant_center is None or abs(float(dominant_center) - 90.0) > 15.0:
        return None
    if dominant_mass < SMALL_BEND_DOMINANT_MASS_MIN:
        return None
    single_scale_count = int(scan_features.get("single_scale_count") or 0)
    multiscale_count = int(scan_features.get("multiscale_count") or 0)
    union_candidate_count = int(scan_features.get("union_candidate_count") or 0)
    if multiscale_count < single_scale_count + SMALL_BEND_MULTISCALE_GAIN_MIN:
        return None
    if union_candidate_count < SMALL_BEND_UNION_MIN:
        return None

    repeated = policy_scores.get("repeated_90_sheetmetal")
    rolled = policy_scores.get("rolled_or_mixed_transition")
    if repeated is None or rolled is None:
        return None
    if repeated.score < SMALL_BEND_SUPPORT_SCORE_MIN or rolled.score < SMALL_BEND_SUPPORT_SCORE_MIN:
        return None
    if repeated.estimated_bend_count < winner.estimated_bend_count + SMALL_BEND_COUNT_DELTA_MIN:
        return None
    if rolled.estimated_bend_count < winner.estimated_bend_count + SMALL_BEND_COUNT_DELTA_MIN:
        return None
    if len(repeated.dominant_angle_families_deg) != 1:
        return None
    if abs(float(repeated.dominant_angle_families_deg[0]) - float(dominant_center)) > REPEATED_90_FAMILY_TOLERANCE_DEG:
        return None

    repeated_range = tuple(int(v) for v in repeated.estimated_bend_count_range)
    rolled_range = tuple(int(v) for v in rolled.estimated_bend_count_range)
    overlap_low = max(repeated_range[0], rolled_range[0])
    overlap_high = min(repeated_range[1], rolled_range[1])
    repeated_perturbation = dict((repeated.support_breakdown or {}).get("perturbation") or {})
    rolled_perturbation = dict((rolled.support_breakdown or {}).get("perturbation") or {})
    rescue_range = (min(repeated_range[0], rolled_range[0]), max(repeated_range[1], rolled_range[1]))
    rescue_mode = "union"
    if (
        overlap_low <= overlap_high
        and float(repeated_perturbation.get("identity_stability") or 0.0) >= SMALL_BEND_IDENTITY_MIN
        and float(rolled_perturbation.get("identity_stability") or 0.0) >= SMALL_BEND_IDENTITY_MIN
        and float(repeated_perturbation.get("modal_count_support") or 0.0) >= SMALL_BEND_MODAL_SUPPORT_MIN
        and float(rolled_perturbation.get("modal_count_support") or 0.0) >= SMALL_BEND_MODAL_SUPPORT_MIN
        and repeated_perturbation.get("modal_count") is not None
        and repeated_perturbation.get("modal_count") == rolled_perturbation.get("modal_count")
        and overlap_low <= int(repeated_perturbation.get("modal_count")) <= overlap_high
    ):
        rescue_range = (overlap_low, overlap_high)
        rescue_mode = "structural_overlap"
    return {
        "count_range": rescue_range,
        "rescue_mode": rescue_mode,
        "source_policies": ["repeated_90_sheetmetal", "rolled_or_mixed_transition"],
    }


def _small_tight_bend_exact_count_recovery(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    small_bend_rescue: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if winner_name != "conservative_macro":
        return None
    if small_bend_rescue is None or small_bend_rescue.get("rescue_mode") != "structural_overlap":
        return None
    if str(scan_features.get("scan_state_hint") or "").lower() == "partial":
        return None
    conservative_retained = int((winner.support_breakdown or {}).get("retained_candidate_count") or winner.estimated_bend_count or 0)
    if conservative_retained > 2:
        return None
    repeated = policy_scores.get("repeated_90_sheetmetal")
    rolled = policy_scores.get("rolled_or_mixed_transition")
    if repeated is None or rolled is None:
        return None
    if repeated.score < SMALL_BEND_EXACT_SCORE_MIN or rolled.score < SMALL_BEND_EXACT_SCORE_MIN:
        return None
    repeated_perturbation = dict((repeated.support_breakdown or {}).get("perturbation") or {})
    rolled_perturbation = dict((rolled.support_breakdown or {}).get("perturbation") or {})
    repeated_modal = repeated_perturbation.get("modal_count")
    rolled_modal = rolled_perturbation.get("modal_count")
    if repeated_modal is None or rolled_modal is None or int(repeated_modal) != int(rolled_modal):
        return None
    if float(repeated_perturbation.get("identity_stability") or 0.0) < SMALL_BEND_IDENTITY_MIN:
        return None
    if float(rolled_perturbation.get("identity_stability") or 0.0) < SMALL_BEND_IDENTITY_MIN:
        return None
    if float(repeated_perturbation.get("modal_count_support") or 0.0) < SMALL_BEND_MODAL_SUPPORT_MIN:
        return None
    if float(rolled_perturbation.get("modal_count_support") or 0.0) < SMALL_BEND_MODAL_SUPPORT_MIN:
        return None
    exact_count = int(repeated_modal)
    rescue_range = tuple(int(v) for v in small_bend_rescue.get("count_range") or ())
    if len(rescue_range) != 2 or not (rescue_range[0] <= exact_count <= rescue_range[1]):
        return None
    if exact_count <= int(winner.estimated_bend_count):
        return None
    return {
        "exact_count": exact_count,
        "source_policies": ["repeated_90_sheetmetal", "rolled_or_mixed_transition"],
        "mode": "structural_overlap_modal_match",
    }


def _mixed_angle_lower_bound_rescue(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    margin: float,
) -> Optional[Dict[str, Any]]:
    if winner_name != "rolled_or_mixed_transition":
        return None
    if float(margin) >= POLICY_MARGIN_MIN:
        return None
    if str(scan_features.get("scan_state_hint") or "").lower() == "partial":
        return None
    if float(scan_features.get("dominant_angle_family_mass") or 0.0) > MIXED_LOWER_BOUND_DOMINANT_MASS_MAX:
        return None
    if len(winner.dominant_angle_families_deg) < 2:
        return None

    conservative = policy_scores.get("conservative_macro")
    if conservative is None or conservative.exact_count_eligible:
        return None
    conservative_perturbation = dict((conservative.support_breakdown or {}).get("perturbation") or {})
    winner_perturbation = dict((winner.support_breakdown or {}).get("perturbation") or {})
    conservative_modal = conservative_perturbation.get("modal_count")
    winner_modal = winner_perturbation.get("modal_count")
    if conservative_modal is None or winner_modal is None:
        return None
    if float(conservative_perturbation.get("modal_count_support") or 0.0) < SMALL_BEND_MODAL_SUPPORT_MIN:
        return None
    if float(winner_perturbation.get("modal_count_support") or 0.0) < SMALL_BEND_MODAL_SUPPORT_MIN:
        return None
    if float(winner_perturbation.get("identity_stability") or 0.0) < MIXED_LOWER_BOUND_IDENTITY_MIN:
        return None

    conservative_low, conservative_high = tuple(int(v) for v in conservative.estimated_bend_count_range)
    winner_low, winner_high = tuple(int(v) for v in winner.estimated_bend_count_range)
    conservative_modal = int(conservative_modal)
    winner_modal = int(winner_modal)
    if conservative_modal != conservative_low:
        return None
    if winner_modal != int(winner.estimated_bend_count):
        return None
    if conservative_low >= winner_modal:
        return None
    if conservative_high < winner_modal:
        return None
    if winner_low > winner_modal or winner_high < winner_modal:
        return None

    return {
        "count_range": (conservative_low, winner_modal),
        "mode": "mixed_lower_bound_modal_bridge",
        "source_policies": ["conservative_macro", "rolled_or_mixed_transition"],
    }


def _near_orthogonal_family_range_rescue(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    margin: float,
) -> Optional[Dict[str, Any]]:
    if winner_name != "conservative_macro":
        return None
    if float(margin) >= POLICY_MARGIN_MIN:
        return None
    if str(scan_features.get("scan_state_hint") or "").lower() == "partial":
        return None
    dominant_center = scan_features.get("dominant_angle_family_deg")
    if dominant_center is None or abs(float(dominant_center) - 90.0) > REPEATED_90_FAMILY_TOLERANCE_DEG:
        return None
    if float(scan_features.get("dominant_angle_family_mass") or 0.0) < NEAR_ORTHOGONAL_RESCUE_DOMINANT_MASS_MIN:
        return None
    if int(scan_features.get("union_candidate_count") or 0) < NEAR_ORTHOGONAL_RESCUE_UNION_MIN:
        return None
    if int(scan_features.get("multiscale_count") or 0) < int(scan_features.get("single_scale_count") or 0) + NEAR_ORTHOGONAL_RESCUE_GAIN_MIN:
        return None

    orthogonal = policy_scores.get("orthogonal_dense_sheetmetal")
    repeated = policy_scores.get("repeated_90_sheetmetal")
    if orthogonal is None or repeated is None:
        return None
    orthogonal_estimated = int(orthogonal.estimated_bend_count)
    repeated_estimated = int(repeated.estimated_bend_count)
    if orthogonal_estimated != repeated_estimated:
        return None
    if orthogonal_estimated <= int(winner.estimated_bend_count):
        return None
    orthogonal_perturbation = dict((orthogonal.support_breakdown or {}).get("perturbation") or {})
    repeated_perturbation = dict((repeated.support_breakdown or {}).get("perturbation") or {})
    if float(orthogonal_perturbation.get("identity_stability") or 0.0) < NEAR_ORTHOGONAL_RESCUE_IDENTITY_MIN:
        return None
    if float(repeated_perturbation.get("identity_stability") or 0.0) < NEAR_ORTHOGONAL_RESCUE_IDENTITY_MIN:
        return None
    if float(orthogonal_perturbation.get("matched_reference_fraction_mean") or 0.0) < NEAR_ORTHOGONAL_RESCUE_MATCH_FRACTION_MIN:
        return None
    if float(repeated_perturbation.get("matched_reference_fraction_mean") or 0.0) < NEAR_ORTHOGONAL_RESCUE_MATCH_FRACTION_MIN:
        return None

    upper_bound = max(
        int(orthogonal.estimated_bend_count_range[1]),
        int(repeated.estimated_bend_count_range[1]),
    )
    if upper_bound <= orthogonal_estimated:
        return None
    return {
        "count_range": (orthogonal_estimated, upper_bound),
        "mode": "near_orthogonal_family_bridge",
        "source_policies": [
            "orthogonal_dense_sheetmetal",
            "repeated_90_sheetmetal",
        ],
    }


def _near_orthogonal_exact_dropout_recovery(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    near_orthogonal_range_rescue: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if winner_name != "conservative_macro":
        return None
    if str(scan_features.get("scan_state_hint") or "").lower() == "partial":
        return None
    orthogonal = policy_scores.get("orthogonal_dense_sheetmetal")
    repeated = policy_scores.get("repeated_90_sheetmetal")
    if orthogonal is None or repeated is None:
        return None
    if orthogonal.score < NEAR_ORTHOGONAL_EXACT_SCORE_MIN or repeated.score < NEAR_ORTHOGONAL_EXACT_SCORE_MIN:
        return None

    orthogonal_perturbation = dict((orthogonal.support_breakdown or {}).get("perturbation") or {})
    repeated_perturbation = dict((repeated.support_breakdown or {}).get("perturbation") or {})
    exact_count = int(orthogonal.estimated_bend_count)
    if exact_count != int(repeated.estimated_bend_count):
        return None
    rescue_range = tuple(int(v) for v in near_orthogonal_range_rescue.get("count_range") or ()) if near_orthogonal_range_rescue else ()
    if rescue_range:
        if len(rescue_range) != 2 or rescue_range[0] != exact_count:
            return None
        if rescue_range[1] != exact_count:
            return None
    if exact_count <= int(winner.estimated_bend_count):
        return None

    for perturbation in (orthogonal_perturbation, repeated_perturbation):
        if float(perturbation.get("identity_stability") or 0.0) < NEAR_ORTHOGONAL_EXACT_IDENTITY_MIN:
            return None
        if float(perturbation.get("matched_reference_fraction_mean") or 0.0) < NEAR_ORTHOGONAL_EXACT_MATCH_FRACTION_MIN:
            return None
        if int(perturbation.get("count_span") or 0) > NEAR_ORTHOGONAL_EXACT_MAX_SPAN:
            return None
        modal_count = perturbation.get("modal_count")
        if modal_count is None:
            return None
        if exact_count - int(modal_count) > NEAR_ORTHOGONAL_EXACT_MAX_DROPOUT:
            return None
        counts = [int(v) for v in list(perturbation.get("counts") or [])]
        if counts and exact_count - min(counts) > NEAR_ORTHOGONAL_EXACT_MAX_DROPOUT:
            return None
    return {
        "exact_count": exact_count,
        "source_policies": ["orthogonal_dense_sheetmetal", "repeated_90_sheetmetal"],
        "mode": "near_orthogonal_dropout_match",
    }


def _rolled_structural_support_range_rescue(
    *,
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if winner_name != "rolled_or_mixed_transition":
        return None
    if str(scan_features.get("scan_state_hint") or "").lower() == "partial":
        return None
    if winner.exact_count_eligible or winner.score < ROLLED_SUPPORT_RESCUE_SCORE_MIN:
        return None
    low, high = tuple(int(v) for v in winner.estimated_bend_count_range)
    winner_perturbation = dict((winner.support_breakdown or {}).get("perturbation") or {})
    winner_consensus = _perturbation_count_consensus_range(winner_perturbation)
    if high - low < ROLLED_SUPPORT_RESCUE_MIN_WIDTH and winner_consensus is None:
        return None
    if float(winner_perturbation.get("identity_stability") or 0.0) < ROLLED_SUPPORT_RESCUE_IDENTITY_MIN:
        return None
    if float(winner_perturbation.get("matched_reference_fraction_mean") or 0.0) < ROLLED_SUPPORT_RESCUE_MATCH_FRACTION_MIN:
        return None
    reference_strip_count = int(winner_perturbation.get("reference_strip_count") or 0)
    strip_count = len(list((winner.support_breakdown or {}).get("bend_support_strips") or []))
    support_cap = max(reference_strip_count, strip_count, int(winner.estimated_bend_count))
    if support_cap < int(winner.estimated_bend_count):
        return None
    if winner_consensus is not None:
        consensus_low, consensus_high = winner_consensus
        lower_bound = max(low, int(consensus_low))
        upper_bound = min(high, int(consensus_high))
        if lower_bound < upper_bound and (lower_bound > low or upper_bound < high):
            return {
                "count_range": (lower_bound, upper_bound),
                "mode": "rolled_structural_consensus_bridge",
                "source_policies": ["rolled_or_mixed_transition"],
                "reference_strip_count": reference_strip_count,
                "support_strip_count": strip_count,
                "winner_consensus_range": list(winner_consensus),
            }

    conservative = policy_scores.get("conservative_macro")
    if conservative is None:
        return None
    conservative_perturbation = dict((conservative.support_breakdown or {}).get("perturbation") or {})
    if float(conservative_perturbation.get("matched_reference_fraction_mean") or 0.0) < ROLLED_SUPPORT_RESCUE_MATCH_FRACTION_MIN:
        return None
    lower_bound = max(low, int(conservative.estimated_bend_count))
    upper_bound = min(high, support_cap)
    rescue_mode = "rolled_structural_support_bridge"
    if lower_bound >= upper_bound:
        return None
    if lower_bound <= low and upper_bound >= high:
        return None
    if int(conservative.estimated_bend_count) >= int(winner.estimated_bend_count):
        return None
    return {
        "count_range": (lower_bound, upper_bound),
        "mode": rescue_mode,
        "source_policies": ["conservative_macro", "rolled_or_mixed_transition"],
        "reference_strip_count": reference_strip_count,
        "support_strip_count": strip_count,
        "winner_consensus_range": list(winner_consensus) if winner_consensus is not None else None,
    }


def _reported_profile_family(
    winner_name: str,
    winner: PolicyScore,
    policy_scores: Dict[str, PolicyScore],
    scan_features: Dict[str, Any],
    abstained: bool,
) -> str:
    if not abstained:
        return winner_name
    rolled = policy_scores.get("rolled_or_mixed_transition")
    if (
        winner_name == "conservative_macro"
        and rolled is not None
        and float(scan_features.get("dominant_angle_family_mass") or 0.0) < 0.65
        and len(winner.dominant_angle_families_deg) > 1
        and len(rolled.dominant_angle_families_deg) >= 2
        and rolled.score >= 0.2
    ):
        return "rolled_or_mixed_transition"
    return winner_name


def _allow_structural_range_without_count_abstention(
    winner_name: str,
    winner: PolicyScore,
    margin: float,
    scan_features: Dict[str, Any],
    abstain_reasons: Sequence[str],
) -> bool:
    if winner_name != "rolled_or_mixed_transition":
        return False
    if len(winner.dominant_angle_families_deg) < 2:
        return False
    if winner.score < 0.65:
        return False
    if margin < 0.15:
        return False
    disallowed = {"dominant_angle_family_weak", "mixed_angle_family_ambiguous", "policy_margin_low"}
    if any(reason in disallowed for reason in abstain_reasons):
        return False
    allowed = {"policy_confidence_low", "perturbation_count_unstable", "perturbation_unstable"}
    return all(reason in allowed for reason in abstain_reasons)


def _resolve_scan_state_hint(part_id: Optional[str], scan_path: str) -> Optional[str]:
    if not part_id:
        return None
    metadata_path = Path(__file__).resolve().parents[1] / "data" / "bend_corpus" / "parts" / str(part_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    target_name = Path(scan_path).name
    target_path = str(Path(scan_path).resolve())
    for scan in metadata.get("scan_files") or []:
        scan_name = str(scan.get("name") or "")
        scan_path_meta = str(scan.get("path") or "")
        if scan_name == target_name or scan_path_meta == target_path:
            state = str(scan.get("state") or "").strip().lower()
            return state or None
    return None


def _angles_for_bends(bends: Sequence[DetectedBend]) -> List[float]:
    return [float(b.measured_angle) for b in bends]


def _dominant_angle_family(angles: Sequence[float], tolerance_deg: float = REPEATED_90_FAMILY_TOLERANCE_DEG) -> Tuple[Optional[float], float]:
    if not angles:
        return None, 0.0
    values = sorted(float(v) for v in angles)
    best_center = values[0]
    best_support: List[float] = []
    for center in values:
        support = [v for v in values if abs(v - center) <= tolerance_deg]
        if len(support) > len(best_support):
            best_center = center
            best_support = support
    if not best_support:
        return None, 0.0
    return float(np.mean(best_support)), float(len(best_support)) / float(len(values))


def _angle_family_centers(angles: Sequence[float], tolerance_deg: float = 15.0) -> List[float]:
    remaining = sorted(float(v) for v in angles)
    centers: List[float] = []
    while remaining:
        center, _ = _dominant_angle_family(remaining, tolerance_deg=tolerance_deg)
        if center is None:
            break
        support = [v for v in remaining if abs(v - center) <= tolerance_deg]
        centers.append(float(np.mean(support)))
        remaining = [v for v in remaining if abs(v - center) > tolerance_deg]
    return [round(v, 3) for v in centers]


def _cluster_line_families(
    bends: Sequence[DetectedBend],
    direction_similarity: float,
    length_delta_mm: float,
    orthogonal_sep_mm: float,
) -> List[List[DetectedBend]]:
    groups: List[List[DetectedBend]] = []
    for bend in bends:
        placed = False
        for group in groups:
            rep = group[0]
            if _same_line_family(
                rep,
                bend,
                direction_similarity=direction_similarity,
                length_delta_mm=length_delta_mm,
                orthogonal_sep_mm=orthogonal_sep_mm,
            ):
                group.append(bend)
                placed = True
                break
        if not placed:
            groups.append([bend])
    return groups


def _same_line_family(
    lhs: DetectedBend,
    rhs: DetectedBend,
    direction_similarity: float,
    length_delta_mm: float,
    orthogonal_sep_mm: float,
) -> bool:
    lhs_dir = _normalize_direction(lhs.bend_line_direction)
    rhs_dir = _normalize_direction(rhs.bend_line_direction)
    if abs(float(np.dot(lhs_dir, rhs_dir))) < direction_similarity:
        return False
    if abs(float(lhs.bend_line_length - rhs.bend_line_length)) > length_delta_mm:
        return False
    midpoint_delta = _bend_midpoint(lhs) - _bend_midpoint(rhs)
    projected = midpoint_delta - np.dot(midpoint_delta, lhs_dir) * lhs_dir
    return float(np.linalg.norm(projected)) < orthogonal_sep_mm


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    unit = direction / norm
    if unit[0] < 0 or (abs(unit[0]) < 1e-8 and unit[1] < 0) or (abs(unit[0]) < 1e-8 and abs(unit[1]) < 1e-8 and unit[2] < 0):
        unit = -unit
    return unit


def _bend_midpoint(bend: DetectedBend) -> np.ndarray:
    return (np.asarray(bend.bend_line_start, dtype=np.float64) + np.asarray(bend.bend_line_end, dtype=np.float64)) / 2.0


def _angle_coherence(bends: Sequence[DetectedBend], dominant_angles: Sequence[float], tolerance_deg: float) -> float:
    if not bends:
        return 0.0
    if not dominant_angles:
        return 0.0
    coherent = 0
    for bend in bends:
        angle = float(bend.measured_angle)
        if any(abs(angle - float(center)) <= tolerance_deg for center in dominant_angles):
            coherent += 1
    return float(coherent) / float(len(bends))


def _duplicate_suppression(raw_count: int, retained_count: int) -> float:
    if raw_count <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(max(0, raw_count - retained_count)) / float(raw_count)))


def _structural_compactness(raw_count: int, retained_count: int) -> float:
    if raw_count <= 0:
        return 0.0
    ratio = float(retained_count) / float(raw_count)
    target = 0.7
    return max(0.0, 1.0 - min(1.0, abs(ratio - target) / target))


def _stability_score(perturbation: Dict[str, Any]) -> float:
    counts = list(perturbation.get("counts") or [])
    angles = [value for value in (perturbation.get("dominant_angle_families_deg") or []) if value is not None]
    identity = float(perturbation.get("identity_stability") or 0.0)
    if not counts:
        return 0.0
    count_span = max(counts) - min(counts)
    count_score = max(0.0, 1.0 - min(1.0, float(count_span) / 2.0))
    if not angles:
        return (count_score + identity) / 2.0
    angle_span = max(angles) - min(angles)
    angle_score = max(0.0, 1.0 - min(1.0, float(angle_span) / 15.0))
    return (count_score + angle_score + identity) / 3.0


def _perturbation_count_consensus_range(
    perturbation: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    counts = sorted(int(value) for value in (perturbation.get("counts") or []))
    if len(counts) < 3:
        return None
    median = counts[len(counts) // 2]
    inliers = [count for count in counts if abs(count - median) <= 1]
    if len(inliers) < 2:
        return None
    return (min(inliers), max(inliers))


def _has_odd_isolated_angle_family(
    bends: Sequence[DetectedBend],
    dominant_angles: Sequence[float],
) -> bool:
    if not bends or not dominant_angles:
        return False
    angles = [float(b.measured_angle) for b in bends]
    dominant = float(dominant_angles[0])
    raw_count = float(len(angles))
    for angle in angles:
        if abs(angle - dominant) <= 20.0:
            continue
        support = sum(1 for value in angles if abs(value - angle) <= 5.0) / raw_count
        if support < 0.15:
            return True
    return False


def _abstain_reasons(score: float, perturbation: Dict[str, Any], dominant_mass: float) -> List[str]:
    reasons: List[str] = []
    if score < EXACT_COUNT_CONFIDENCE_MIN:
        reasons.append("policy_confidence_low")
    if dominant_mass < DOMINANT_FAMILY_STRONG_MASS:
        reasons.append("dominant_angle_family_weak")
    if int(perturbation.get("count_span") or 0) > 1:
        reasons.append("perturbation_count_unstable")
    return reasons


def _strong_near_orthogonal_family(scan_features: Dict[str, Any]) -> bool:
    dominant_center = scan_features.get("dominant_angle_family_deg")
    dominant_mass = float(scan_features.get("dominant_angle_family_mass") or 0.0)
    if dominant_center is None:
        return False
    return dominant_mass >= 0.9 and abs(float(dominant_center) - 90.0) <= 15.0


def _serialize_line_groups(line_groups: Sequence[Sequence[DetectedBend]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for index, group in enumerate(line_groups, start=1):
        representative = max(group, key=lambda bend: float(bend.confidence or 0.0))
        serialized.append(
            {
                "family_id": f"F{index}",
                "candidate_count": len(group),
                "measured_angle_deg": round(float(representative.measured_angle), 3),
                "bend_line_length_mm": round(float(representative.bend_line_length), 3),
                "direction": [round(float(v), 6) for v in _normalize_direction(representative.bend_line_direction)],
            }
        )
    return serialized


def _estimate_local_spacing(points: np.ndarray, sample_size: int = 2048) -> float:
    if len(points) <= 1:
        return 1.0
    if len(points) > sample_size:
        indices = np.linspace(0, len(points) - 1, num=sample_size, dtype=int)
        sample = points[indices]
    else:
        sample = points
    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=min(2, len(sample)))
    if np.ndim(distances) == 1:
        nearest = distances
    else:
        nearest = distances[:, -1]
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(max(0.1, np.median(finite)))


def _bend_trace_endpoints(bend: DetectedBend) -> List[List[float]]:
    start = np.asarray(bend.bend_line_start, dtype=np.float64)
    end = np.asarray(bend.bend_line_end, dtype=np.float64)
    return [[float(v) for v in start.tolist()], [float(v) for v in end.tolist()]]


def _bend_projection_interval(
    endpoints: Sequence[Sequence[float]],
    axis: np.ndarray,
) -> Tuple[float, float]:
    points = np.asarray(endpoints, dtype=np.float64)
    projections = points @ axis
    return float(np.min(projections)), float(np.max(projections))


def _interval_overlap_gap(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> Tuple[float, float]:
    overlap = min(lhs[1], rhs[1]) - max(lhs[0], rhs[0])
    if overlap >= 0.0:
        return float(overlap), 0.0
    return 0.0, float(-overlap)


def _bend_pair_relation(
    lhs: DetectedBend,
    rhs: DetectedBend,
    *,
    spacing_mm: float,
) -> Dict[str, float]:
    axis = _normalize_direction(lhs.bend_line_direction)
    lhs_interval = _bend_projection_interval(_bend_trace_endpoints(lhs), axis)
    rhs_interval = _bend_projection_interval(_bend_trace_endpoints(rhs), axis)
    overlap_mm, gap_mm = _interval_overlap_gap(lhs_interval, rhs_interval)
    midpoint_delta = _bend_midpoint(lhs) - _bend_midpoint(rhs)
    orthogonal = midpoint_delta - np.dot(midpoint_delta, axis) * axis
    orthogonal_sep_mm = float(np.linalg.norm(orthogonal))
    return {
        "overlap_mm": float(overlap_mm),
        "gap_mm": float(gap_mm),
        "orthogonal_sep_mm": orthogonal_sep_mm,
        "axis_similarity": float(abs(np.dot(_normalize_direction(lhs.bend_line_direction), _normalize_direction(rhs.bend_line_direction)))),
        "overlap_ratio": float(
            overlap_mm / max(1.0, min(float(lhs.bend_line_length), float(rhs.bend_line_length)))
        ),
        "near_continuous": float(gap_mm <= max(spacing_mm * 2.0, 0.15 * min(float(lhs.bend_line_length), float(rhs.bend_line_length), 1e9))),
    }


def _bend_fold_strength(bend: DetectedBend) -> float:
    angle_term = abs(np.sin(np.deg2rad(float(bend.measured_angle or 0.0))))
    confidence_term = float(max(0.0, min(1.0, float(bend.confidence or 0.0))))
    flange1_area = float(getattr(getattr(bend, "flange1", None), "area", 0.0) or 0.0)
    flange2_area = float(getattr(getattr(bend, "flange2", None), "area", 0.0) or 0.0)
    area_balance = min(flange1_area, flange2_area) / max(flange1_area, flange2_area, 1.0)
    return float(max(0.0, min(1.0, 0.55 * angle_term + 0.35 * confidence_term + 0.10 * area_balance)))


def _build_candidate_trace_graph(
    bends: Sequence[DetectedBend],
    *,
    spacing_mm: float,
) -> Dict[str, Any]:
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(bends))}
    overlap_pairs = 0
    near_continuous_pairs = 0
    max_overlap_ratio = 0.0
    for lhs_index in range(len(bends)):
        for rhs_index in range(lhs_index + 1, len(bends)):
            relation = _bend_pair_relation(bends[lhs_index], bends[rhs_index], spacing_mm=spacing_mm)
            if relation["axis_similarity"] < STRIP_AXIS_ALIGNMENT_MIN:
                continue
            if relation["orthogonal_sep_mm"] > max(STRIP_OVERLAP_GAP_SCALE * spacing_mm, spacing_mm * 2.0):
                continue
            if relation["overlap_mm"] > 0.0:
                overlap_pairs += 1
                max_overlap_ratio = max(max_overlap_ratio, float(relation["overlap_ratio"]))
            if relation["near_continuous"]:
                near_continuous_pairs += 1
            if relation["overlap_mm"] > 0.0 or relation["near_continuous"]:
                adjacency[lhs_index].append(rhs_index)
                adjacency[rhs_index].append(lhs_index)
    components: List[List[int]] = []
    visited: set[int] = set()
    for index in range(len(bends)):
        if index in visited:
            continue
        stack = [index]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)
        components.append(sorted(component))
    trace_lengths = [float(bend.bend_line_length) for bend in bends]
    fold_strengths = [_bend_fold_strength(bend) for bend in bends]
    summary = {
        "candidate_trace_count": int(len(bends)),
        "connected_component_count": int(len(components)),
        "component_sizes": [int(len(component)) for component in components],
        "mean_branch_degree": round(float(np.mean([len(adjacency[index]) for index in adjacency])) if adjacency else 0.0, 4),
        "overlap_pair_count": int(overlap_pairs),
        "near_continuous_pair_count": int(near_continuous_pairs),
        "max_overlap_ratio": round(float(max_overlap_ratio), 4),
        "trace_length_mm_summary": {
            "min": round(float(min(trace_lengths)), 4) if trace_lengths else 0.0,
            "mean": round(float(np.mean(trace_lengths)), 4) if trace_lengths else 0.0,
            "max": round(float(max(trace_lengths)), 4) if trace_lengths else 0.0,
        },
        "fold_strength_summary": {
            "min": round(float(min(fold_strengths)), 4) if fold_strengths else 0.0,
            "mean": round(float(np.mean(fold_strengths)), 4) if fold_strengths else 0.0,
            "max": round(float(max(fold_strengths)), 4) if fold_strengths else 0.0,
        },
    }
    component_size_by_index: Dict[int, int] = {}
    for component in components:
        for index in component:
            component_size_by_index[index] = len(component)
    return {
        "adjacency": adjacency,
        "components": components,
        "component_size_by_index": component_size_by_index,
        "summary": summary,
    }


def _build_evidence_atoms(
    bends: Sequence[DetectedBend],
    *,
    line_groups: Sequence[Sequence[DetectedBend]],
    source_mode: str,
    spacing_mm: float,
    strip_family: str,
    trace_graph: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> List[BendEvidenceAtom]:
    family_lookup: Dict[int, str] = {}
    for group_index, group in enumerate(line_groups, start=1):
        for bend in group:
            family_lookup[id(bend)] = f"F{group_index}"
    atoms: List[BendEvidenceAtom] = []
    for index, bend in enumerate(bends, start=1):
        midpoint = _bend_midpoint(bend)
        direction = _normalize_direction(bend.bend_line_direction)
        bbox = _support_bbox_for_bend(bend, spacing_mm)
        flange_normals = _extract_flange_normals(bend)
        angle_offset = abs(float(bend.measured_angle) - 90.0)
        observability_flags = ["well_supported"]
        ambiguity_flags: List[str] = []
        trace_endpoints = _bend_trace_endpoints(bend)
        branch_degree = 0
        connected_component_size = 1
        trace_continuity_score = 0.0
        fold_strength = _bend_fold_strength(bend)
        if trace_graph is not None:
            bend_index = index - 1
            branch_degree = int(len(trace_graph["adjacency"].get(bend_index, [])))
            connected_component_size = int(trace_graph["component_size_by_index"].get(bend_index, 1))
            trace_continuity_score = max(0.0, min(1.0, branch_degree / 2.0))
            if connected_component_size > 1:
                trace_continuity_score = max(trace_continuity_score, min(1.0, 0.3 + 0.2 * (connected_component_size - 1)))
        if float(bend.confidence or 0.0) < 0.7:
            ambiguity_flags.append("low_confidence_candidate")
        if float(getattr(bend, "radius_fit_error", 0.0) or 0.0) > max(1.5, spacing_mm * 2.0):
            observability_flags.append("radius_fit_noisy")
        if strip_family != "discrete_fold_strip" or angle_offset > 20.0:
            ambiguity_flags.append("mixed_angle_candidate")
        if branch_degree == 0 and fold_strength < 0.5:
            ambiguity_flags.append("isolated_trace_candidate")
        atoms.append(
            BendEvidenceAtom(
                atom_id=f"A{index}",
                source_mode=source_mode,
                candidate_bend_id=str(getattr(bend, "bend_id", f"cand_{index}")),
                local_axis_direction=[float(v) for v in direction.tolist()],
                local_line_family_id=family_lookup.get(id(bend), f"F{index}"),
                local_flange_compatibility={
                    "flange_normal_delta_deg": round(_flange_normal_delta_deg(flange_normals), 4),
                    "strip_family": strip_family,
                },
                local_spacing_mm=float(spacing_mm),
                local_support_patch_id=f"{source_mode}:patch_{index}",
                local_support_span_mm=float(bend.bend_line_length),
                support_patch_midpoint=[float(v) for v in midpoint.tolist()],
                support_bbox_mm=bbox,
                local_normal_summary={
                    "flange_normals": flange_normals,
                },
                local_curvature_summary={
                    "radius_mm": float(getattr(bend, "measured_radius", 0.0) or 0.0),
                    "radius_fit_error": float(getattr(bend, "radius_fit_error", 0.0) or 0.0),
                },
                null_assignment_eligible=bool(float(bend.confidence or 0.0) < 0.65),
                observability_flags=observability_flags,
                ambiguity_flags=ambiguity_flags,
                trace_endpoints=trace_endpoints if _experiment_enabled(experiment_config, "trace_informed_atoms") else [],
                trace_length_mm=float(bend.bend_line_length) if _experiment_enabled(experiment_config, "trace_informed_atoms") else 0.0,
                trace_continuity_score=float(trace_continuity_score) if _experiment_enabled(experiment_config, "trace_informed_atoms") else 0.0,
                fold_strength=float(fold_strength) if _experiment_enabled(experiment_config, "trace_informed_atoms") else 0.0,
                branch_degree=int(branch_degree) if _experiment_enabled(experiment_config, "trace_informed_atoms") else 0,
                connected_component_size=int(connected_component_size) if _experiment_enabled(experiment_config, "trace_informed_atoms") else 1,
            )
        )
    return atoms


def _infer_support_strips(
    atoms: Sequence[BendEvidenceAtom],
    *,
    spacing_mm: float,
    strip_family: str,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BendSupportStrip], int]:
    if not atoms:
        return [], 0
    groups: List[List[BendEvidenceAtom]] = []
    for atom in atoms:
        placed = False
        for group in groups:
            if _same_support_strip(
                group[0],
                atom,
                spacing_mm=spacing_mm,
                trace_aware=_experiment_enabled(experiment_config, "trace_aware_strip_grouping"),
            ):
                group.append(atom)
                placed = True
                break
        if not placed:
            groups.append([atom])
    strips = [_build_support_strip(group, spacing_mm=spacing_mm, default_family=strip_family, index=index) for index, group in enumerate(groups, start=1)]
    duplicate_merge_count = max(0, len(atoms) - len(groups))
    return strips, duplicate_merge_count


def _same_support_strip(
    lhs: BendEvidenceAtom,
    rhs: BendEvidenceAtom,
    *,
    spacing_mm: float,
    trace_aware: bool = False,
) -> bool:
    lhs_dir = _normalize_direction(np.asarray(lhs.local_axis_direction, dtype=np.float64))
    rhs_dir = _normalize_direction(np.asarray(rhs.local_axis_direction, dtype=np.float64))
    if abs(float(np.dot(lhs_dir, rhs_dir))) < STRIP_AXIS_ALIGNMENT_MIN:
        return False
    lhs_mid = np.asarray(lhs.support_patch_midpoint, dtype=np.float64)
    rhs_mid = np.asarray(rhs.support_patch_midpoint, dtype=np.float64)
    midpoint_delta = lhs_mid - rhs_mid
    orthogonal = midpoint_delta - np.dot(midpoint_delta, lhs_dir) * lhs_dir
    if float(np.linalg.norm(orthogonal)) > max(STRIP_OVERLAP_GAP_SCALE * spacing_mm, spacing_mm * 2.0):
        return False
    lhs_angle = float(lhs.local_flange_compatibility.get("flange_normal_delta_deg") or 0.0)
    rhs_angle = float(rhs.local_flange_compatibility.get("flange_normal_delta_deg") or 0.0)
    if abs(lhs_angle - rhs_angle) > STRIP_ANGLE_TOLERANCE_DEG:
        return False
    if trace_aware and lhs.trace_endpoints and rhs.trace_endpoints:
        lhs_interval = _bend_projection_interval(lhs.trace_endpoints, lhs_dir)
        rhs_interval = _bend_projection_interval(rhs.trace_endpoints, lhs_dir)
        overlap_mm, gap_mm = _interval_overlap_gap(lhs_interval, rhs_interval)
        min_trace = min(float(lhs.trace_length_mm or 0.0), float(rhs.trace_length_mm or 0.0))
        gap_limit = max(spacing_mm * 2.0, 0.15 * max(min_trace, 1.0))
        if overlap_mm <= 0.0 and gap_mm > gap_limit:
            return False
        if (
            gap_mm > spacing_mm
            and float(lhs.fold_strength + rhs.fold_strength) / 2.0 < 0.55
            and float(lhs.trace_continuity_score + rhs.trace_continuity_score) / 2.0 < 0.4
        ):
            return False
    return True


def _prune_candidate_graph_atoms(
    atoms: Sequence[BendEvidenceAtom],
    *,
    spacing_mm: float,
) -> List[BendEvidenceAtom]:
    kept: List[BendEvidenceAtom] = []
    for atom in atoms:
        weak_isolated = (
            int(atom.connected_component_size) <= 1
            and int(atom.branch_degree) <= 0
            and float(atom.fold_strength) < 0.45
            and float(atom.trace_continuity_score) < 0.35
            and atom.null_assignment_eligible
            and "low_confidence_candidate" in atom.ambiguity_flags
        )
        if weak_isolated and float(atom.trace_length_mm or atom.local_support_span_mm) <= max(spacing_mm * 4.0, 12.0):
            continue
        kept.append(atom)
    return kept


def _build_support_strip(
    atoms: Sequence[BendEvidenceAtom],
    *,
    spacing_mm: float,
    default_family: str,
    index: int,
) -> BendSupportStrip:
    axis = _normalize_direction(
        np.mean([np.asarray(atom.local_axis_direction, dtype=np.float64) for atom in atoms], axis=0)
    )
    midpoints = np.asarray([atom.support_patch_midpoint for atom in atoms], dtype=np.float64)
    bbox_min = midpoints.min(axis=0) if len(midpoints) else np.zeros(3)
    bbox_max = midpoints.max(axis=0) if len(midpoints) else np.zeros(3)
    projected = midpoints @ axis
    along_axis_span = float(max(max(projected) - min(projected), max(atom.local_support_span_mm for atom in atoms)))
    dominant_angle = float(np.mean([atom.local_flange_compatibility.get("flange_normal_delta_deg") or 0.0 for atom in atoms]))
    radius_values = [
        float(atom.local_curvature_summary.get("radius_mm") or 0.0)
        for atom in atoms
        if float(atom.local_curvature_summary.get("radius_mm") or 0.0) > 0.0
    ]
    observability_status = "scan_limited" if any(atom.null_assignment_eligible for atom in atoms) else "supported"
    strip_family = default_family
    if default_family == "rolled_transition_strip" and len({tuple(atom.local_axis_direction) for atom in atoms}) == 1:
        strip_family = "rolled_transition_strip"
    elif default_family != "discrete_fold_strip" and len(atoms) > 1:
        strip_family = "mixed_transition_strip"
    normals: List[List[float]] = []
    for atom in atoms:
        for normal in atom.local_normal_summary.get("flange_normals") or []:
            if normal not in normals:
                normals.append(normal)
    support_point_count = int(sum(max(3, round(atom.local_support_span_mm / max(spacing_mm, 0.1))) for atom in atoms))
    confidence = float(np.mean([1.0 - min(1.0, len(atom.ambiguity_flags) * 0.2) for atom in atoms]))
    stability_score = float(max(0.0, min(1.0, 1.0 - (len(atoms) - 1) * 0.05)))
    return BendSupportStrip(
        strip_id=f"S{index}",
        strip_family=strip_family if observability_status != "scan_limited" else "scan_limited_strip",
        axis_direction=[float(v) for v in axis.tolist()],
        axis_line_hint={
            "midpoint": [float(v) for v in midpoints.mean(axis=0).tolist()],
            "source_atom_ids": [atom.atom_id for atom in atoms],
        },
        adjacent_flange_normals=normals[:2],
        dominant_angle_deg=dominant_angle,
        radius_mm=float(np.mean(radius_values)) if radius_values else None,
        support_patch_ids=[atom.local_support_patch_id for atom in atoms],
        support_point_count=support_point_count,
        cross_fold_width_mm=float(max(spacing_mm * STRIP_SPACING_SCALE, spacing_mm * (1.0 + len(atoms) * 0.5))),
        along_axis_span_mm=along_axis_span,
        support_bbox_mm=[float(v) for v in (bbox_max - bbox_min).tolist()],
        observability_status=observability_status,
        merge_provenance={
            "source_modes": sorted(dict.fromkeys(atom.source_mode for atom in atoms)),
            "merged_atom_ids": [atom.atom_id for atom in atoms],
        },
        stability_score=stability_score,
        confidence=confidence,
    )


def _support_bbox_for_bend(bend: DetectedBend, spacing_mm: float) -> List[float]:
    line = np.abs(np.asarray(bend.bend_line_end, dtype=np.float64) - np.asarray(bend.bend_line_start, dtype=np.float64))
    return [float(max(line[0], spacing_mm)), float(max(line[1], spacing_mm)), float(max(line[2], spacing_mm))]


def _extract_flange_normals(bend: DetectedBend) -> List[List[float]]:
    normals: List[List[float]] = []
    for flange_name in ("flange1", "flange2"):
        flange = getattr(bend, flange_name, None)
        normal = getattr(flange, "normal", None)
        if normal is None:
            continue
        normals.append([float(v) for v in _normalize_direction(np.asarray(normal, dtype=np.float64)).tolist()])
    return normals


def _flange_normal_delta_deg(normals: Sequence[Sequence[float]]) -> float:
    if len(normals) < 2:
        return 0.0
    lhs = _normalize_direction(np.asarray(normals[0], dtype=np.float64))
    rhs = _normalize_direction(np.asarray(normals[1], dtype=np.float64))
    dot = max(-1.0, min(1.0, float(np.dot(lhs, rhs))))
    return float(np.degrees(np.arccos(dot)))


def _bend_evidence_atom_from_dict(payload: Dict[str, Any]) -> BendEvidenceAtom:
    return BendEvidenceAtom(
        atom_id=str(payload.get("atom_id") or ""),
        source_mode=str(payload.get("source_mode") or ""),
        candidate_bend_id=str(payload.get("candidate_bend_id") or ""),
        local_axis_direction=list(payload.get("local_axis_direction") or []),
        local_line_family_id=str(payload.get("local_line_family_id") or ""),
        local_flange_compatibility=dict(payload.get("local_flange_compatibility") or {}),
        local_spacing_mm=float(payload.get("local_spacing_mm") or 0.0),
        local_support_patch_id=str(payload.get("local_support_patch_id") or ""),
        local_support_span_mm=float(payload.get("local_support_span_mm") or 0.0),
        support_patch_midpoint=list(payload.get("support_patch_midpoint") or []),
        support_bbox_mm=list(payload.get("support_bbox_mm") or []),
        local_normal_summary=dict(payload.get("local_normal_summary") or {}),
        local_curvature_summary=dict(payload.get("local_curvature_summary") or {}),
        null_assignment_eligible=bool(payload.get("null_assignment_eligible")),
        observability_flags=list(payload.get("observability_flags") or []),
        ambiguity_flags=list(payload.get("ambiguity_flags") or []),
        trace_endpoints=list(payload.get("trace_endpoints") or []),
        trace_length_mm=float(payload.get("trace_length_mm") or 0.0),
        trace_continuity_score=float(payload.get("trace_continuity_score") or 0.0),
        fold_strength=float(payload.get("fold_strength") or 0.0),
        branch_degree=int(payload.get("branch_degree") or 0),
        connected_component_size=int(payload.get("connected_component_size") or 1),
    )


def _bend_support_strip_from_dict(payload: Dict[str, Any]) -> BendSupportStrip:
    return BendSupportStrip(
        strip_id=str(payload.get("strip_id") or ""),
        strip_family=str(payload.get("strip_family") or ""),
        axis_direction=list(payload.get("axis_direction") or []),
        axis_line_hint=dict(payload.get("axis_line_hint") or {}),
        adjacent_flange_normals=list(payload.get("adjacent_flange_normals") or []),
        dominant_angle_deg=float(payload.get("dominant_angle_deg") or 0.0),
        radius_mm=None if payload.get("radius_mm") is None else float(payload.get("radius_mm")),
        support_patch_ids=list(payload.get("support_patch_ids") or []),
        support_point_count=int(payload.get("support_point_count") or 0),
        cross_fold_width_mm=float(payload.get("cross_fold_width_mm") or 0.0),
        along_axis_span_mm=float(payload.get("along_axis_span_mm") or 0.0),
        support_bbox_mm=list(payload.get("support_bbox_mm") or []),
        observability_status=str(payload.get("observability_status") or ""),
        merge_provenance=dict(payload.get("merge_provenance") or {}),
        stability_score=float(payload.get("stability_score") or 0.0),
        confidence=float(payload.get("confidence") or 0.0),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
