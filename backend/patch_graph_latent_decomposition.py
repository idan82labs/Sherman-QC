from __future__ import annotations

import os
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ai_analyzer import Model3DSnapshotRenderer
from feature_detection.bend_detector import BendDetector, DetectedBend
from zero_shot_bend_profile import (
    _build_candidate_bundles,
    _estimate_local_spacing,
    _extract_scan_features,
    _load_scan_geometry,
    _normalize_direction,
    run_zero_shot_bend_profile,
)
try:
    from scan_family_router import (
        ScanFamilyRoute,
        build_scan_formation_features,
        route_from_payload,
        route_scan_family,
        solver_options_for_route,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from backend.scan_family_router import (
        ScanFamilyRoute,
        build_scan_formation_features,
        route_from_payload,
        route_scan_family,
        solver_options_for_route,
    )


@dataclass(frozen=True)
class SurfaceAtom:
    atom_id: str
    voxel_key: Tuple[int, int, int]
    point_indices: Tuple[int, ...]
    centroid: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    local_spacing_mm: float
    curvature_proxy: float
    axis_direction: Tuple[float, float, float]
    weight: float
    area_proxy_mm2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "voxel_key": list(self.voxel_key),
            "point_indices": list(self.point_indices),
            "centroid": list(self.centroid),
            "normal": list(self.normal),
            "local_spacing_mm": round(float(self.local_spacing_mm), 4),
            "curvature_proxy": round(float(self.curvature_proxy), 6),
            "axis_direction": list(self.axis_direction),
            "weight": round(float(self.weight), 4),
            "area_proxy_mm2": round(float(self.area_proxy_mm2), 4),
        }


@dataclass(frozen=True)
class SurfaceAtomGraph:
    atoms: Tuple[SurfaceAtom, ...]
    adjacency: Dict[str, Tuple[str, ...]]
    edge_weights: Dict[str, float]
    voxel_size_mm: float
    local_spacing_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "adjacency": {key: list(values) for key, values in sorted(self.adjacency.items())},
            "edge_weights": dict(sorted((key, round(float(value), 6)) for key, value in self.edge_weights.items())),
            "voxel_size_mm": round(float(self.voxel_size_mm), 4),
            "local_spacing_mm": round(float(self.local_spacing_mm), 4),
        }


@dataclass(frozen=True)
class FlangeHypothesis:
    flange_id: str
    source_kind: str
    atom_ids: Tuple[str, ...]
    normal: Tuple[float, float, float]
    d: float
    centroid: Tuple[float, float, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flange_id": self.flange_id,
            "source_kind": self.source_kind,
            "atom_ids": list(self.atom_ids),
            "normal": list(self.normal),
            "d": round(float(self.d), 6),
            "centroid": list(self.centroid),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass(frozen=True)
class BendHypothesis:
    bend_id: str
    bend_class: str
    source_kind: str
    incident_flange_ids: Tuple[str, ...]
    canonical_flange_pair: Tuple[str, ...]
    canonical_bend_key: str
    atom_ids: Tuple[str, ...]
    anchor: Tuple[float, float, float]
    axis_direction: Tuple[float, float, float]
    angle_deg: float
    radius_mm: Optional[float]
    visibility_state: str
    confidence: float
    cross_width_mm: float
    span_mm: float
    candidate_classes: Tuple[str, ...] = field(default_factory=tuple)
    source_bend_ids: Tuple[str, ...] = field(default_factory=tuple)
    seed_incident_flange_ids: Tuple[str, ...] = field(default_factory=tuple)
    candidate_incident_flange_pairs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bend_id": self.bend_id,
            "bend_class": self.bend_class,
            "source_kind": self.source_kind,
            "incident_flange_ids": list(self.incident_flange_ids),
            "canonical_flange_pair": list(self.canonical_flange_pair),
            "canonical_bend_key": self.canonical_bend_key,
            "atom_ids": list(self.atom_ids),
            "anchor": list(self.anchor),
            "axis_direction": list(self.axis_direction),
            "angle_deg": round(float(self.angle_deg), 4),
            "radius_mm": None if self.radius_mm is None else round(float(self.radius_mm), 4),
            "visibility_state": self.visibility_state,
            "confidence": round(float(self.confidence), 4),
            "cross_width_mm": round(float(self.cross_width_mm), 4),
            "span_mm": round(float(self.span_mm), 4),
            "candidate_classes": list(self.candidate_classes),
            "source_bend_ids": list(self.source_bend_ids),
            "seed_incident_flange_ids": list(self.seed_incident_flange_ids),
            "candidate_incident_flange_pairs": [list(pair) for pair in self.candidate_incident_flange_pairs],
        }


@dataclass(frozen=True)
class TypedLabelAssignment:
    atom_labels: Dict[str, str]
    label_types: Dict[str, str]
    energy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_labels": dict(sorted(self.atom_labels.items())),
            "label_types": dict(sorted(self.label_types.items())),
            "energy": round(float(self.energy), 6),
        }


@dataclass(frozen=True)
class OwnedBendRegion:
    bend_id: str
    bend_class: str
    incident_flange_ids: Tuple[str, ...]
    canonical_bend_key: str
    owned_atom_ids: Tuple[str, ...]
    anchor: Tuple[float, float, float]
    axis_direction: Tuple[float, float, float]
    support_centroid: Tuple[float, float, float]
    span_endpoints: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    angle_deg: float
    radius_mm: Optional[float]
    visibility_state: str
    support_mass: float
    admissible: bool
    admissibility_reasons: Tuple[str, ...]
    post_bend_class: str
    debug_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bend_id": self.bend_id,
            "bend_class": self.bend_class,
            "incident_flange_ids": list(self.incident_flange_ids),
            "canonical_bend_key": self.canonical_bend_key,
            "owned_atom_ids": list(self.owned_atom_ids),
            "anchor": list(self.anchor),
            "axis_direction": list(self.axis_direction),
            "support_centroid": list(self.support_centroid),
            "span_endpoints": [list(self.span_endpoints[0]), list(self.span_endpoints[1])],
            "angle_deg": round(float(self.angle_deg), 4),
            "radius_mm": None if self.radius_mm is None else round(float(self.radius_mm), 4),
            "visibility_state": self.visibility_state,
            "support_mass": round(float(self.support_mass), 4),
            "admissible": bool(self.admissible),
            "admissibility_reasons": list(self.admissibility_reasons),
            "post_bend_class": self.post_bend_class,
            "debug_confidence": round(float(self.debug_confidence), 4),
        }


@dataclass(frozen=True)
class DecompositionResult:
    scan_path: str
    part_id: Optional[str]
    atom_graph: SurfaceAtomGraph
    flange_hypotheses: Tuple[FlangeHypothesis, ...]
    bend_hypotheses: Tuple[BendHypothesis, ...]
    assignment: TypedLabelAssignment
    owned_bend_regions: Tuple[OwnedBendRegion, ...]
    exact_bend_count: int
    bend_count_range: Tuple[int, int]
    candidate_solution_energies: Tuple[float, ...]
    candidate_solution_counts: Tuple[int, ...]
    object_level_confidence_summary: Dict[str, float]
    repair_diagnostics: Dict[str, Any]
    current_zero_shot_result: Dict[str, Any]
    current_zero_shot_count_range: Tuple[int, int]
    current_zero_shot_exact_count: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_path": self.scan_path,
            "part_id": self.part_id,
            "atom_graph": self.atom_graph.to_dict(),
            "flange_hypotheses": [item.to_dict() for item in self.flange_hypotheses],
            "bend_hypotheses": [item.to_dict() for item in self.bend_hypotheses],
            "assignment": self.assignment.to_dict(),
            "owned_bend_regions": [item.to_dict() for item in self.owned_bend_regions],
            "exact_bend_count": int(self.exact_bend_count),
            "bend_count_range": list(self.bend_count_range),
            "candidate_solution_energies": [round(float(value), 6) for value in self.candidate_solution_energies],
            "candidate_solution_counts": [int(value) for value in self.candidate_solution_counts],
            "object_level_confidence_summary": dict(sorted((key, round(float(value), 4)) for key, value in self.object_level_confidence_summary.items())),
            "repair_diagnostics": self.repair_diagnostics,
            "current_zero_shot_result": self.current_zero_shot_result,
            "current_zero_shot_count_range": list(self.current_zero_shot_count_range),
            "current_zero_shot_exact_count": self.current_zero_shot_exact_count,
        }


def run_patch_graph_latent_decomposition(
    *,
    scan_path: str,
    runtime_config: Any,
    part_id: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    route_mode: str = "off",
    scan_family_route: Optional[ScanFamilyRoute | Mapping[str, Any]] = None,
    solver_option_overrides: Optional[Mapping[str, Any]] = None,
) -> DecompositionResult:
    verbose_trace = os.environ.get("PATCH_GRAPH_F1_VERBOSE", "").strip().lower() not in {"", "0", "false", "no"}

    def _trace(message: str) -> None:
        if verbose_trace:
            print(f"[patch-graph-f1] {message}", flush=True)

    stage_started = perf_counter()
    stage_timings: Dict[str, float] = {}

    cache_hit = False
    points: np.ndarray
    current_result: Dict[str, Any]
    atom_graph: SurfaceAtomGraph
    flange_hypotheses: List[FlangeHypothesis]
    bend_hypotheses: List[BendHypothesis]
    cached_payload = _load_cached_research_inputs(cache_dir) if cache_dir is not None else None
    if cached_payload is not None:
        cache_hit = True
        points = np.asarray(cached_payload["bbox_points"], dtype=np.float64)
        current_result = dict(cached_payload["current_zero_shot_result"])
        atom_graph = _surface_atom_graph_from_dict(cached_payload["atom_graph"])
        flange_hypotheses = [FlangeHypothesis(**payload) for payload in cached_payload["flange_hypotheses"]]
        bend_hypotheses = _ensure_candidate_incident_flange_pairs(
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=[_bend_hypothesis_from_dict(payload) for payload in cached_payload["bend_hypotheses"]],
        )
        stage_timings["load_cached_research_inputs"] = perf_counter() - stage_started
        _trace(
            "load_cached_research_inputs:done seconds="
            f"{stage_timings['load_cached_research_inputs']:.3f} atoms={len(atom_graph.atoms)} bends={len(bend_hypotheses)}"
        )
    else:
        _trace("load_scan_geometry:start")
        points, normals = _load_scan_geometry(scan_path)
        stage_timings["load_scan_geometry"] = perf_counter() - stage_started
        _trace(f"load_scan_geometry:done seconds={stage_timings['load_scan_geometry']:.3f} points={len(points)}")

        stage_started = perf_counter()
        local_spacing_mm = _estimate_local_spacing(points)
        normals = _ensure_normals(points, normals, local_spacing_mm)
        voxel_size_mm = max(local_spacing_mm * 3.0, 2.5)
        atom_graph = build_surface_atom_graph(points, normals, voxel_size_mm=voxel_size_mm, local_spacing_mm=local_spacing_mm)
        stage_timings["build_atom_graph"] = perf_counter() - stage_started
        _trace(f"build_atom_graph:done seconds={stage_timings['build_atom_graph']:.3f} atoms={len(atom_graph.atoms)}")

        stage_started = perf_counter()
        detector_kwargs = dict(getattr(runtime_config, "scan_detector_kwargs", {}) or {})
        detector = BendDetector(**detector_kwargs)
        candidate_bundles = _build_candidate_bundles(
            detector=detector,
            points=points,
            normals=normals,
            detect_call_kwargs=dict(getattr(runtime_config, "scan_detect_call_kwargs", {}) or {}),
        )
        stage_timings["build_candidate_bundles"] = perf_counter() - stage_started
        _trace(f"build_candidate_bundles:done seconds={stage_timings['build_candidate_bundles']:.3f}")

        stage_started = perf_counter()
        current_result = run_zero_shot_bend_profile(scan_path=scan_path, runtime_config=runtime_config, part_id=part_id).to_dict()
        stage_timings["run_zero_shot_control"] = perf_counter() - stage_started
        _trace(f"run_zero_shot_control:done seconds={stage_timings['run_zero_shot_control']:.3f}")

        stage_started = perf_counter()
        flange_hypotheses = generate_flange_hypotheses(
            atom_graph=atom_graph,
            candidate_bundles=candidate_bundles,
            current_zero_shot_result=current_result,
        )
        bend_hypotheses = generate_bend_hypotheses(
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            candidate_bundles=candidate_bundles,
            current_zero_shot_result=current_result,
            bbox_points=points,
        )
        stage_timings["generate_hypotheses"] = perf_counter() - stage_started
        _trace(
            "generate_hypotheses:done seconds="
            f"{stage_timings['generate_hypotheses']:.3f} flanges={len(flange_hypotheses)} bends={len(bend_hypotheses)}"
        )
        if cache_dir is not None:
            _write_cached_research_inputs(
                cache_dir=cache_dir,
                payload={
                    "bbox_points": np.asarray(points, dtype=np.float64).tolist(),
                    "current_zero_shot_result": current_result,
                    "atom_graph": atom_graph.to_dict(),
                    "flange_hypotheses": [item.to_dict() for item in flange_hypotheses],
                    "bend_hypotheses": [item.to_dict() for item in bend_hypotheses],
                },
            )

    route_mode_normalized = str(route_mode or "off").strip().lower()
    if route_mode_normalized not in {"off", "observe", "apply"}:
        route_mode_normalized = "off"
    route_payload: Optional[Dict[str, Any]] = None
    route: Optional[ScanFamilyRoute] = None
    if scan_family_route is not None:
        route = scan_family_route if isinstance(scan_family_route, ScanFamilyRoute) else route_from_payload(scan_family_route)
    if route is None and route_mode_normalized != "off":
        formation_features = build_scan_formation_features(
            current_zero_shot_result=current_result,
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
        )
        route = route_scan_family(formation_features)
    if route is not None:
        route_payload = route.to_dict()
        route_payload["mode"] = route_mode_normalized
        route_payload["applied"] = bool(route_mode_normalized == "apply")

    solver_options = solver_options_for_route(route, route_mode_normalized)
    solver_options["scan_family_route"] = route.to_dict()
    current_range = _normalize_count_range(
        current_result.get("estimated_bend_count_range"),
        current_result.get("estimated_bend_count"),
    )
    if (
        solver_options.get("enable_selected_pair_duplicate_prune")
        and current_range
        and "selected_pair_duplicate_prune_min_active_bends" not in solver_options
    ):
        solver_options["selected_pair_duplicate_prune_min_active_bends"] = int(current_range[1]) + 1
    if solver_options.get("enable_birth_support_rescue"):
        solver_options["allow_no_contact_birth_rescue"] = bool(
            solver_options.get("allow_no_contact_birth_rescue")
            or _allow_no_contact_birth_rescue_for_range(current_range)
        )
    if solver_options.get("limit_birth_rescue_to_control_upper") and current_range:
        max_active_bends = int(current_range[1])
        if solver_options.get("limit_subspan_births_to_control_low_plus_one") and len(current_range) >= 2:
            max_active_bends = min(max_active_bends, int(current_range[0]) + 1)
        solver_options["birth_rescue_max_active_bends"] = max_active_bends
    if solver_option_overrides:
        solver_options.update(dict(solver_option_overrides))
    if bool(solver_options.get("enable_interface_birth")):
        birth_hypotheses = generate_interface_birth_hypotheses(
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            existing_bend_hypotheses=bend_hypotheses,
            enable_subspan_births=bool(solver_options.get("enable_interface_subspan_births", False)),
        )
        if birth_hypotheses:
            bend_hypotheses = [*bend_hypotheses, *birth_hypotheses]
        solver_options["interface_birth_hypothesis_count"] = len(birth_hypotheses)

    stage_started = perf_counter()
    solution_payload = solve_typed_latent_decomposition(
        atom_graph=atom_graph,
        flange_hypotheses=flange_hypotheses,
        bend_hypotheses=bend_hypotheses,
        solver_options=solver_options,
    )
    stage_timings["solve_typed_latent_decomposition"] = perf_counter() - stage_started
    _trace(
        "solve_typed_latent_decomposition:done seconds="
        f"{stage_timings['solve_typed_latent_decomposition']:.3f} "
        f"solution_counts={solution_payload.get('solution_counts')}"
    )

    stage_started = perf_counter()
    owned_regions = extract_owned_bend_regions(
        atom_graph=atom_graph,
        assignment=solution_payload["assignment"],
        flange_hypotheses=flange_hypotheses,
        bend_hypotheses=bend_hypotheses,
        solution_payload=solution_payload,
    )
    stage_timings["extract_owned_bend_regions"] = perf_counter() - stage_started
    _trace(
        "extract_owned_bend_regions:done seconds="
        f"{stage_timings['extract_owned_bend_regions']:.3f} owned_regions={len(owned_regions)}"
    )

    solution_payload["repair_diagnostics"] = {
        **dict(solution_payload.get("repair_diagnostics") or {}),
        "scan_family_route": route_payload,
        "solver_options": dict(solver_options),
        "stage_timings_seconds": {key: round(float(value), 4) for key, value in stage_timings.items()},
        "cache_hit": cache_hit,
        "problem_sizes": {
            "point_count": int(len(points)),
            "atom_count": int(len(atom_graph.atoms)),
            "flange_hypothesis_count": int(len(flange_hypotheses)),
            "bend_hypothesis_count": int(len(bend_hypotheses)),
        },
    }
    return DecompositionResult(
        scan_path=scan_path,
        part_id=part_id,
        atom_graph=atom_graph,
        flange_hypotheses=tuple(flange_hypotheses),
        bend_hypotheses=tuple(bend_hypotheses),
        assignment=solution_payload["assignment"],
        owned_bend_regions=tuple(owned_regions),
        exact_bend_count=sum(1 for item in owned_regions if item.admissible),
        bend_count_range=_solution_count_range(solution_payload["solution_counts"]),
        candidate_solution_energies=tuple(solution_payload["solution_energies"]),
        candidate_solution_counts=tuple(solution_payload["solution_counts"]),
        object_level_confidence_summary=solution_payload["object_level_confidence_summary"],
        repair_diagnostics=solution_payload["repair_diagnostics"],
        current_zero_shot_result=current_result,
        current_zero_shot_count_range=current_range,
        current_zero_shot_exact_count=_coerce_optional_int(current_result.get("estimated_bend_count")),
    )


def build_surface_atom_graph(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    voxel_size_mm: float,
    local_spacing_mm: float,
) -> SurfaceAtomGraph:
    keys = np.floor(points / max(voxel_size_mm, 1e-6)).astype(int)
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        buckets[(int(key[0]), int(key[1]), int(key[2]))].append(index)

    atoms: List[SurfaceAtom] = []
    key_to_atom_id: Dict[Tuple[int, int, int], str] = {}
    for atom_index, (key, indices) in enumerate(sorted(buckets.items()), start=1):
        pts = points[indices]
        nrm = normals[indices]
        centroid = np.mean(pts, axis=0)
        avg_normal = _normalize_direction(np.mean(nrm, axis=0))
        covariance = np.cov((pts - centroid).T) if len(indices) >= 3 else np.eye(3) * 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)
        curvature_proxy = float(max(eigenvalues[order[0]], 0.0) / max(np.sum(np.maximum(eigenvalues, 0.0)), 1e-6))
        axis_direction = _normalize_direction(eigenvectors[:, order[-1]])
        atom = SurfaceAtom(
            atom_id=f"A{atom_index}",
            voxel_key=key,
            point_indices=tuple(indices),
            centroid=tuple(float(v) for v in centroid.tolist()),
            normal=tuple(float(v) for v in avg_normal.tolist()),
            local_spacing_mm=float(local_spacing_mm),
            curvature_proxy=float(curvature_proxy),
            axis_direction=tuple(float(v) for v in axis_direction.tolist()),
            weight=float(len(indices)),
            area_proxy_mm2=float(len(indices) * voxel_size_mm * voxel_size_mm),
        )
        atoms.append(atom)
        key_to_atom_id[key] = atom.atom_id

    adjacency: Dict[str, List[str]] = {atom.atom_id: [] for atom in atoms}
    edge_weights: Dict[str, float] = {}
    for atom in atoms:
        for neighbor_key in _neighbor_keys(atom.voxel_key):
            neighbor_id = key_to_atom_id.get(neighbor_key)
            if not neighbor_id or neighbor_id == atom.atom_id:
                continue
            edge_key = _edge_key(atom.atom_id, neighbor_id)
            if edge_key in edge_weights:
                continue
            neighbor_atom = next(item for item in atoms if item.atom_id == neighbor_id)
            weight = _atom_adjacency_weight(atom, neighbor_atom, voxel_size_mm)
            adjacency[atom.atom_id].append(neighbor_id)
            adjacency[neighbor_id].append(atom.atom_id)
            edge_weights[edge_key] = float(weight)
    return SurfaceAtomGraph(
        atoms=tuple(atoms),
        adjacency={key: tuple(sorted(values)) for key, values in adjacency.items()},
        edge_weights=edge_weights,
        voxel_size_mm=float(voxel_size_mm),
        local_spacing_mm=float(local_spacing_mm),
    )


def generate_flange_hypotheses(
    *,
    atom_graph: SurfaceAtomGraph,
    candidate_bundles: Mapping[str, Mapping[str, Sequence[DetectedBend]]],
    current_zero_shot_result: Optional[Mapping[str, Any]] = None,
) -> List[FlangeHypothesis]:
    original = candidate_bundles.get("original") or {}
    union_bends = list(original.get("single_scale") or []) + list(original.get("multiscale") or [])
    plane_seeds: List[Tuple[str, Any]] = []
    for bend in union_bends:
        plane_seeds.append((f"{bend.bend_id}:f1", bend.flange1))
        plane_seeds.append((f"{bend.bend_id}:f2", bend.flange2))

    flange_hypotheses: List[FlangeHypothesis] = []
    atom_lookup = {atom.atom_id: atom for atom in getattr(atom_graph, "atoms", ())}
    next_index = 1
    for _, plane in plane_seeds:
        compatible = [
            atom.atom_id
            for atom in atom_graph.atoms
            if _point_to_plane_distance(atom.centroid, plane.normal, plane.d) <= atom_graph.local_spacing_mm * 2.5
            and _angle_deg(atom.normal, plane.normal) <= 22.5
        ]
        if not compatible:
            continue
        for component in _connected_components(compatible, atom_graph.adjacency):
            if len(component) < 2:
                continue
            pts = np.asarray([atom_lookup[atom_id].centroid for atom_id in component], dtype=np.float64)
            centroid = np.mean(pts, axis=0)
            normal = _normalize_direction(plane.normal)
            d = float(-np.dot(normal, centroid))
            confidence = float(min(1.0, len(component) / 8.0))
            flange_hypotheses.append(
                FlangeHypothesis(
                    flange_id=f"F{next_index}",
                    source_kind="detector_plane_seed",
                    atom_ids=tuple(sorted(component)),
                    normal=tuple(float(v) for v in normal.tolist()),
                    d=d,
                    centroid=tuple(float(v) for v in centroid.tolist()),
                    confidence=confidence,
                )
            )
            next_index += 1
    if _flange_orientation_family_count(flange_hypotheses) < 2:
        for strip in list((current_zero_shot_result or {}).get("bend_support_strips") or []):
            axis = _normalize_direction(strip.get("axis_direction") or [1.0, 0.0, 0.0])
            anchor = np.asarray((strip.get("axis_line_hint") or {}).get("midpoint") or [0.0, 0.0, 0.0], dtype=np.float64)
            cross_width_mm = float(strip.get("cross_fold_width_mm") or atom_graph.local_spacing_mm * 2.0 or 1.0)
            span_mm = float(strip.get("along_axis_span_mm") or cross_width_mm * 2.0 or 1.0)
            corridor_radius = max(cross_width_mm * 2.5, atom_graph.local_spacing_mm * 5.0, 4.0)
            half_span = max(span_mm * 0.85, corridor_radius)
            for raw_normal in strip.get("adjacent_flange_normals") or ():
                normal = _normalize_direction(raw_normal)
                compatible = [
                    atom.atom_id
                    for atom in atom_graph.atoms
                    if _angle_deg(atom.normal, normal) <= 28.0
                    and _distance_to_line(np.asarray(atom.centroid, dtype=np.float64), anchor, axis) <= corridor_radius
                    and abs(np.dot(np.asarray(atom.centroid, dtype=np.float64) - anchor, axis)) <= half_span
                ]
                if len(compatible) < 2:
                    continue
                for component in _connected_components(compatible, atom_graph.adjacency):
                    if len(component) < 2:
                        continue
                    pts = np.asarray([atom_lookup[atom_id].centroid for atom_id in component], dtype=np.float64)
                    centroid = np.mean(pts, axis=0)
                    d = float(-np.dot(normal, centroid))
                    confidence = float(min(0.72, 0.25 + len(component) / 20.0))
                    flange_hypotheses.append(
                        FlangeHypothesis(
                            flange_id=f"F{next_index}",
                            source_kind="zero_shot_strip_normal_seed",
                            atom_ids=tuple(sorted(component)),
                            normal=tuple(float(v) for v in normal.tolist()),
                            d=d,
                            centroid=tuple(float(v) for v in centroid.tolist()),
                            confidence=confidence,
                        )
                    )
                    next_index += 1
    if not flange_hypotheses:
        plane = _fit_plane(np.asarray([atom.centroid for atom in atom_graph.atoms], dtype=np.float64))
        flange_hypotheses.append(
            FlangeHypothesis(
                flange_id="F1",
                source_kind="fallback_global_plane",
                atom_ids=tuple(atom.atom_id for atom in atom_graph.atoms),
                normal=tuple(float(v) for v in plane["normal"].tolist()),
                d=float(plane["d"]),
                centroid=tuple(float(v) for v in plane["centroid"].tolist()),
                confidence=0.2,
            )
        )
    return _dedupe_flange_hypotheses(flange_hypotheses, atom_graph)


def generate_bend_hypotheses(
    *,
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    candidate_bundles: Mapping[str, Mapping[str, Sequence[DetectedBend]]],
    current_zero_shot_result: Mapping[str, Any],
    bbox_points: np.ndarray,
) -> List[BendHypothesis]:
    bend_hypotheses: List[BendHypothesis] = []
    atom_lookup = {atom.atom_id: atom for atom in getattr(atom_graph, "atoms", ())}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    next_index = 1
    for strip in list(current_zero_shot_result.get("bend_support_strips") or []):
        axis = _normalize_direction(strip.get("axis_direction") or [1.0, 0.0, 0.0])
        anchor = np.asarray((strip.get("axis_line_hint") or {}).get("midpoint") or [0.0, 0.0, 0.0], dtype=np.float64)
        angle_deg = float(strip.get("dominant_angle_deg") or 0.0)
        radius_mm = strip.get("radius_mm")
        cross_width_mm = float(strip.get("cross_fold_width_mm") or atom_graph.local_spacing_mm * 2.0 or 1.0)
        span_mm = float(strip.get("along_axis_span_mm") or cross_width_mm * 2.0 or 1.0)
        strip_normals = list(strip.get("adjacent_flange_normals") or [])
        incident_flange_ids = _match_flange_ids_for_normals(strip_normals, flange_hypotheses)
        support_atoms = [
            atom.atom_id
            for atom in atom_graph.atoms
            if _distance_to_line(np.asarray(atom.centroid, dtype=np.float64), anchor, axis) <= max(cross_width_mm * 0.8, atom_graph.voxel_size_mm * 1.5)
            and abs(np.dot(np.asarray(atom.centroid, dtype=np.float64) - anchor, axis)) <= max(span_mm * 0.75, atom_graph.voxel_size_mm * 1.5)
        ]
        visibility_state = _visibility_state(anchor, bbox_points, atom_graph.local_spacing_mm)
        class_variants = [("transition_only", None)]
        if radius_mm not in {None, 0, 0.0}:
            class_variants.append(("circular_strip", float(radius_mm)))
        if cross_width_mm > atom_graph.local_spacing_mm * 1.5:
            class_variants.append(("developable_strip", float(radius_mm) if radius_mm not in {None, 0, 0.0} else None))
        for bend_class, class_radius in class_variants:
            bend_hypotheses.append(
                BendHypothesis(
                    bend_id=f"B{next_index}",
                    bend_class=bend_class,
                    source_kind="zero_shot_strip_seed",
                    incident_flange_ids=tuple(incident_flange_ids),
                    canonical_flange_pair=tuple(sorted(incident_flange_ids)),
                    canonical_bend_key=f"RAW::{next_index}",
                    atom_ids=tuple(sorted(support_atoms)),
                    anchor=tuple(float(v) for v in anchor.tolist()),
                    axis_direction=tuple(float(v) for v in axis.tolist()),
                    angle_deg=angle_deg,
                    radius_mm=class_radius,
                    visibility_state=visibility_state,
                    confidence=float(strip.get("confidence") or 0.0),
                    cross_width_mm=cross_width_mm,
                    span_mm=span_mm,
                    candidate_classes=(bend_class,),
                    source_bend_ids=(str(strip.get("strip_id") or f"S{next_index}"),),
                )
            )
            next_index += 1
    original = candidate_bundles.get("original") or {}
    union_bends = list(original.get("single_scale") or []) + list(original.get("multiscale") or [])
    for bend in union_bends:
        axis = _normalize_direction(bend.bend_line_direction)
        anchor = (np.asarray(bend.bend_line_start, dtype=np.float64) + np.asarray(bend.bend_line_end, dtype=np.float64)) / 2.0
        incident_flange_ids = _match_flange_ids_for_normals(
            [getattr(bend.flange1, "normal", None), getattr(bend.flange2, "normal", None)],
            flange_hypotheses,
        )
        support_atoms = [
            atom.atom_id
            for atom in atom_graph.atoms
            if _distance_to_line(np.asarray(atom.centroid, dtype=np.float64), anchor, axis) <= max(atom_graph.voxel_size_mm * 2.0, bend.measured_radius + atom_graph.local_spacing_mm)
            and abs(np.dot(np.asarray(atom.centroid, dtype=np.float64) - anchor, axis)) <= max(bend.bend_line_length * 0.75, atom_graph.voxel_size_mm * 2.0)
        ]
        bend_hypotheses.append(
            BendHypothesis(
                bend_id=f"B{next_index}",
                bend_class="transition_only" if bend.measured_radius <= atom_graph.local_spacing_mm * 2.0 else "circular_strip",
                source_kind="detector_bend_seed",
                incident_flange_ids=tuple(incident_flange_ids),
                canonical_flange_pair=tuple(sorted(incident_flange_ids)),
                canonical_bend_key=f"RAW::{next_index}",
                atom_ids=tuple(sorted(support_atoms)),
                anchor=tuple(float(v) for v in anchor.tolist()),
                axis_direction=tuple(float(v) for v in axis.tolist()),
                angle_deg=float(bend.measured_angle),
                radius_mm=float(bend.measured_radius),
                visibility_state=_visibility_state(anchor, bbox_points, atom_graph.local_spacing_mm),
                confidence=float(bend.confidence),
                cross_width_mm=max(float(bend.measured_radius), atom_graph.local_spacing_mm * 1.25),
                span_mm=float(bend.bend_line_length),
                candidate_classes=(("transition_only" if bend.measured_radius <= atom_graph.local_spacing_mm * 2.0 else "circular_strip"),),
                source_bend_ids=(str(getattr(bend, "bend_id", f"D{next_index}")),),
            )
        )
        next_index += 1
    deduped = _dedupe_bend_hypotheses(bend_hypotheses, flange_lookup)
    return _ensure_candidate_incident_flange_pairs(
        atom_graph=atom_graph,
        flange_hypotheses=flange_hypotheses,
        bend_hypotheses=_normalize_bend_hypotheses(deduped),
    )


def generate_interface_birth_hypotheses(
    *,
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    existing_bend_hypotheses: Sequence[BendHypothesis],
    max_births: int = 24,
    enable_subspan_births: bool = False,
) -> List[BendHypothesis]:
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    if len(flange_lookup) < 2 or not atom_lookup:
        return []

    candidates: List[Tuple[float, Tuple[str, str], List[str], np.ndarray, np.ndarray, float]] = []
    flanges = sorted(flange_hypotheses, key=lambda item: item.flange_id)
    for left_index, left in enumerate(flanges):
        for right in flanges[left_index + 1 :]:
            pair = _canonical_flange_pair((left.flange_id, right.flange_id))
            if len(pair) != 2:
                continue
            theta = _angle_deg(left.normal, right.normal)
            if theta < 20.0 or theta > 165.0:
                continue
            atom_ids = _interface_candidate_atoms_for_pair(
                pair=pair,
                atom_graph=atom_graph,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
            )
            if len(atom_ids) < 3:
                continue
            for component in _connected_components(atom_ids, atom_graph.adjacency):
                if len(component) < 3:
                    continue
                points = np.asarray([atom_lookup[atom_id].centroid for atom_id in component], dtype=np.float64)
                centroid = np.mean(points, axis=0)
                pair_line = _plane_pair_line(left, right, centroid)
                if pair_line is None:
                    continue
                anchor, axis = pair_line
                projections = points @ axis
                span = float(np.max(projections) - np.min(projections)) if len(projections) else 0.0
                h = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
                if span < max(2.0 * h, 2.0):
                    continue
                component_variants = _interface_birth_component_variants(
                    component=component,
                    atom_graph=atom_graph,
                    atom_lookup=atom_lookup,
                    axis=axis,
                    enable_subspan_births=enable_subspan_births,
                )
                for variant in component_variants:
                    if len(variant) < 3:
                        continue
                    variant_points = np.asarray([atom_lookup[atom_id].centroid for atom_id in variant], dtype=np.float64)
                    variant_centroid = np.mean(variant_points, axis=0)
                    variant_line = _plane_pair_line(left, right, variant_centroid)
                    if variant_line is None:
                        continue
                    variant_anchor, variant_axis = variant_line
                    variant_projections = variant_points @ variant_axis
                    variant_span = float(np.max(variant_projections) - np.min(variant_projections)) if len(variant_projections) else 0.0
                    if variant_span < max(2.0 * h, 2.0):
                        continue
                    bendness = float(np.mean([min(1.0, float(atom_lookup[atom_id].curvature_proxy) * 20.0) for atom_id in variant]))
                    touches = _component_support_touches_flange_pair(
                        component=variant,
                        pair=pair,
                        atom_graph=atom_graph,
                        flange_lookup=flange_lookup,
                    )
                    subspan_bonus = 3.0 if enable_subspan_births and len(component_variants) > 1 else 0.0
                    score = float(len(variant)) + 2.0 * float(touches) + bendness + subspan_bonus
                    candidates.append((score, (pair[0], pair[1]), list(variant), variant_anchor, variant_axis, theta))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2][0]))
    births: List[BendHypothesis] = []
    used_components: List[set[str]] = []
    for _, pair, component, anchor, axis, theta in candidates:
        component_set = set(component)
        if any(len(component_set & used) / max(1, min(len(component_set), len(used))) > 0.6 for used in used_components):
            continue
        points = np.asarray([atom_lookup[atom_id].centroid for atom_id in component], dtype=np.float64)
        projections = points @ axis
        span = float(np.max(projections) - np.min(projections)) if len(projections) else 0.0
        h = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
        cross_width = max(4.0 * h, 8.0)
        birth_id = f"BIRTH_{pair[0]}_{pair[1]}_{len(births) + 1}"
        births.append(
            BendHypothesis(
                bend_id=birth_id,
                bend_class="transition_only",
                source_kind="interface_birth",
                incident_flange_ids=pair,
                canonical_flange_pair=pair,
                canonical_bend_key=f"interface_birth::{pair[0]}::{pair[1]}::{len(births) + 1}",
                atom_ids=tuple(sorted(component)),
                anchor=tuple(float(value) for value in anchor.tolist()),
                axis_direction=tuple(float(value) for value in axis.tolist()),
                angle_deg=float(theta),
                radius_mm=None,
                visibility_state="internal",
                confidence=0.55,
                cross_width_mm=float(cross_width),
                span_mm=float(max(span, cross_width)),
                candidate_classes=("transition_only",),
                source_bend_ids=(),
                seed_incident_flange_ids=pair,
                candidate_incident_flange_pairs=(pair,),
            )
        )
        used_components.append(component_set)
        if len(births) >= max_births:
            break
    return births


def _interface_birth_component_variants(
    *,
    component: Sequence[str],
    atom_graph: SurfaceAtomGraph,
    atom_lookup: Mapping[str, SurfaceAtom],
    axis: np.ndarray,
    enable_subspan_births: bool,
) -> List[List[str]]:
    base = list(component)
    if not enable_subspan_births or len(base) < 9:
        return [base]
    points = np.asarray([atom_lookup[atom_id].centroid for atom_id in base if atom_id in atom_lookup], dtype=np.float64)
    if len(points) != len(base):
        return [base]
    unit_axis = _normalize_direction(axis)
    projections_by_atom = {
        atom_id: float(np.dot(np.asarray(atom_lookup[atom_id].centroid, dtype=np.float64), unit_axis))
        for atom_id in base
    }
    projections = np.asarray(list(projections_by_atom.values()), dtype=np.float64)
    span = float(np.max(projections) - np.min(projections)) if len(projections) else 0.0
    h = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
    target_span = max(24.0, 10.0 * h)
    if span <= target_span * 1.35:
        return [base]
    bin_count = min(8, max(2, int(math.ceil(span / target_span))))
    low = float(np.min(projections))
    high = float(np.max(projections))
    if high <= low:
        return [base]
    variants: List[List[str]] = []
    for index in range(bin_count):
        bin_low = low + (high - low) * index / bin_count
        bin_high = low + (high - low) * (index + 1) / bin_count
        if index == bin_count - 1:
            atoms_in_bin = [atom_id for atom_id in base if bin_low <= projections_by_atom[atom_id] <= bin_high]
        else:
            atoms_in_bin = [atom_id for atom_id in base if bin_low <= projections_by_atom[atom_id] < bin_high]
        for subcomponent in _connected_components(atoms_in_bin, atom_graph.adjacency):
            if len(subcomponent) >= 3:
                variants.append(list(subcomponent))
    return variants if len(variants) >= 2 else [base]


def solve_typed_latent_decomposition(
    *,
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    solver_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    solver_options = dict(solver_options or {})
    label_types = {item.flange_id: "flange" for item in flange_hypotheses}
    label_types.update({item.bend_id: "bend" for item in bend_hypotheses})
    label_types["residual"] = "residual"
    label_types["outlier"] = "outlier"
    candidate_runs: List[Dict[str, Any]] = []
    initial_modes = ("flange_bias", "bend_bias", "proposal_bias")
    for initial_mode in initial_modes:
        labels = _solve_base_labeling(
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
            label_types=label_types,
            initial_mode=initial_mode,
        )
        labels, mode_diag = _apply_object_repair_moves(
            atom_graph=atom_graph,
            labels=labels,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
            label_types=label_types,
            solver_options=solver_options,
        )
        if solver_options.get("enable_final_duplicate_cluster_cleanup"):
            cleanup_diagnostics: Dict[str, Any] = {}
            labels, cleanup_pruned = _drop_selected_pair_duplicate_bends(
                atom_graph=atom_graph,
                labels=labels,
                bend_lookup={item.bend_id: item for item in bend_hypotheses},
                label_types=label_types,
                flange_lookup={item.flange_id: item for item in flange_hypotheses},
                diagnostics=cleanup_diagnostics,
                allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
                allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
                min_active_bends=int(solver_options.get("selected_pair_duplicate_prune_min_active_bends") or 0),
                alias_atom_count_max=int(solver_options.get("selected_pair_duplicate_prune_alias_atom_max") or 4),
            )
            mode_diag["final_duplicate_cluster_cleanup"] = {
                **cleanup_diagnostics,
                "pruned_bend_labels": list(cleanup_pruned),
            }
        assignment = TypedLabelAssignment(
            atom_labels=dict(sorted(labels.items())),
            label_types=label_types,
            energy=0.0,
        )
        energy = _total_energy(
            atom_graph=atom_graph,
            labels=assignment.atom_labels,
            flange_lookup={item.flange_id: item for item in flange_hypotheses},
            bend_lookup={item.bend_id: item for item in bend_hypotheses},
            label_types=label_types,
        )
        final_duplicate_diagnostics: Dict[str, Any] = {
            "enabled": bool(solver_options.get("enable_final_duplicate_cluster_scoring", False)),
            "penalty": 0.0,
            "clusters": [],
        }
        if solver_options.get("enable_final_duplicate_cluster_scoring"):
            final_duplicate_penalty, final_duplicate_diagnostics = _final_duplicate_cluster_penalty(
                atom_graph=atom_graph,
                labels=assignment.atom_labels,
                bend_lookup={item.bend_id: item for item in bend_hypotheses},
                flange_lookup={item.flange_id: item for item in flange_hypotheses},
                label_types=label_types,
                alias_atom_count_max=int(solver_options.get("final_duplicate_cluster_alias_atom_max") or 4),
                same_pair_penalty=float(solver_options.get("final_duplicate_cluster_same_pair_penalty") or 120.0),
                cross_pair_tiny_penalty=float(solver_options.get("final_duplicate_cluster_cross_pair_penalty") or 45.0),
                allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
                allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
            )
            energy += final_duplicate_penalty
        candidate_runs.append(
            {
                "assignment": TypedLabelAssignment(
                    atom_labels=assignment.atom_labels,
                    label_types=label_types,
                    energy=energy,
                ),
                "repair_diag": {
                    "initial_mode": initial_mode,
                    **mode_diag,
                    "energy": round(float(energy), 6),
                    "final_duplicate_cluster_scoring": final_duplicate_diagnostics,
                },
            }
        )

    candidate_runs.sort(key=lambda item: item["assignment"].energy)
    if solver_options.get("enable_control_exact_solution_tiebreak"):
        control_exact = _control_exact_count_from_solver_context(solver_options)
        if control_exact is not None:
            best_energy = float(candidate_runs[0]["assignment"].energy)
            gap = float(solver_options.get("control_exact_solution_tiebreak_energy_gap") or 3.0)
            for index, item in enumerate(candidate_runs):
                assignment = item["assignment"]
                count = _count_admissible_bend_components(
                    atom_graph=atom_graph,
                    labels=assignment.atom_labels,
                    bend_lookup={bend.bend_id: bend for bend in bend_hypotheses},
                    label_types=label_types,
                    flange_lookup={flange.flange_id: flange for flange in flange_hypotheses},
                    allow_geometry_backed_single_contact=bool(solver_options.get("allow_geometry_backed_single_contact", False)),
                    allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
                    allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
                    allow_mixed_transition_single_contact=bool(solver_options.get("allow_mixed_transition_single_contact", False)),
                )
                if count == int(control_exact) and float(assignment.energy) <= best_energy + gap:
                    if index != 0:
                        candidate_runs.insert(0, candidate_runs.pop(index))
                    candidate_runs[0]["repair_diag"]["control_exact_solution_tiebreak"] = {
                        "applied": True,
                        "control_exact": int(control_exact),
                        "selected_solution_count": int(count),
                        "selected_energy": round(float(candidate_runs[0]["assignment"].energy), 6),
                        "best_raw_energy": round(best_energy, 6),
                        "allowed_gap": round(gap, 6),
                    }
                    break
            else:
                candidate_runs[0]["repair_diag"]["control_exact_solution_tiebreak"] = {
                    "applied": False,
                    "control_exact": int(control_exact),
                    "best_raw_energy": round(best_energy, 6),
                    "allowed_gap": round(gap, 6),
                }
    best = candidate_runs[0]["assignment"]
    solution_label_maps = [item["assignment"].atom_labels for item in candidate_runs]
    solution_energies = [item["assignment"].energy for item in candidate_runs]
    solution_counts = [
        _count_admissible_bend_components(
            atom_graph=atom_graph,
            labels=assignment.atom_labels,
            bend_lookup={item.bend_id: item for item in bend_hypotheses},
            label_types=label_types,
            flange_lookup={item.flange_id: item for item in flange_hypotheses},
            allow_geometry_backed_single_contact=bool(solver_options.get("allow_geometry_backed_single_contact", False)),
            allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
            allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
            allow_mixed_transition_single_contact=bool(solver_options.get("allow_mixed_transition_single_contact", False)),
        )
        for assignment in [item["assignment"] for item in candidate_runs]
    ]
    object_level_confidence_summary = _build_object_level_confidence_summary(
        atom_graph=atom_graph,
        label_maps=solution_label_maps,
        bend_lookup={item.bend_id: item for item in bend_hypotheses},
        flange_lookup={item.flange_id: item for item in flange_hypotheses},
        solution_energies=solution_energies,
        label_types=label_types,
        allow_geometry_backed_single_contact=bool(solver_options.get("allow_geometry_backed_single_contact", False)),
        allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
        allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
        allow_mixed_transition_single_contact=bool(solver_options.get("allow_mixed_transition_single_contact", False)),
    )
    return {
        "assignment": best,
        "solution_energies": solution_energies,
        "solution_counts": solution_counts,
        "solution_label_maps": solution_label_maps,
        "object_level_confidence_summary": object_level_confidence_summary,
        "repair_diagnostics": {
            "iterations": [item["repair_diag"] for item in candidate_runs],
            "solver_options": dict(solver_options),
            "local_replacement_diagnostics": _build_local_replacement_diagnostics(
                atom_graph=atom_graph,
                labels=best.atom_labels,
                flange_hypotheses=flange_hypotheses,
                bend_hypotheses=bend_hypotheses,
                label_types=label_types,
            ),
        },
    }


def _control_exact_count_from_solver_context(solver_options: Mapping[str, Any]) -> Optional[int]:
    route = solver_options.get("scan_family_route")
    if isinstance(route, ScanFamilyRoute):
        snapshot = dict(route.feature_snapshot or {})
    elif isinstance(route, Mapping):
        snapshot = dict(route.get("feature_snapshot") or {})
    else:
        snapshot = dict(solver_options.get("scan_family_route_feature_snapshot") or {})
    exact = snapshot.get("zero_shot_exact_count")
    if exact is None:
        count_range = snapshot.get("zero_shot_count_range") or ()
        if isinstance(count_range, Sequence) and not isinstance(count_range, (str, bytes)) and len(count_range) >= 2:
            try:
                low, high = int(count_range[0]), int(count_range[1])
            except (TypeError, ValueError):
                return None
            if low == high:
                exact = low
    if exact is None:
        return None
    try:
        return int(exact)
    except (TypeError, ValueError):
        return None


def extract_owned_bend_regions(
    *,
    atom_graph: SurfaceAtomGraph,
    assignment: TypedLabelAssignment,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    solution_payload: Mapping[str, Any],
) -> List[OwnedBendRegion]:
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    bend_lookup = {item.bend_id: item for item in bend_hypotheses}
    bend_atom_ids = [atom_id for atom_id, label in assignment.atom_labels.items() if assignment.label_types.get(label) == "bend"]
    regions: List[OwnedBendRegion] = []
    next_index = 1
    for component in _connected_components(bend_atom_ids, atom_graph.adjacency, labels=assignment.atom_labels):
        label_ids = {assignment.atom_labels[atom_id] for atom_id in component}
        if len(label_ids) != 1:
            continue
        bend = bend_lookup[next(iter(label_ids))]
        centroids = np.asarray([atom_lookup[atom_id].centroid for atom_id in component], dtype=np.float64)
        support_centroid = np.mean(centroids, axis=0)
        axis = _normalize_direction(np.asarray(bend.axis_direction, dtype=np.float64))
        projections = centroids @ axis
        start = support_centroid + axis * (float(np.min(projections - np.dot(support_centroid, axis))))
        end = support_centroid + axis * (float(np.max(projections - np.dot(support_centroid, axis))))
        support_mass = float(sum(atom_lookup[atom_id].weight for atom_id in component))
        solver_options = ((solution_payload.get("repair_diagnostics") or {}).get("solver_options") or {})
        admissible, reasons = _bend_component_admissibility(
            component=component,
            bend=bend,
            atom_graph=atom_graph,
            labels=assignment.atom_labels,
            label_types=assignment.label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup,
            allow_geometry_backed_single_contact=bool(solver_options.get("allow_geometry_backed_single_contact", False)),
            allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
            allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
            allow_mixed_transition_single_contact=bool(solver_options.get("allow_mixed_transition_single_contact", False)),
        )
        attachment_eval = _bend_component_attachment_evaluation(
            component=component,
            bend=bend,
            atom_graph=atom_graph,
            labels=assignment.atom_labels,
            label_types=assignment.label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup,
            allow_hypothesis_support_contacts=bool(solver_options.get("allow_hypothesis_support_contacts", False)),
            allow_cross_axis_attachment_match=bool(solver_options.get("allow_cross_axis_attachment_match", False)),
        )
        confidence = float(solution_payload.get("object_level_confidence_summary", {}).get(bend.canonical_bend_key, _solution_confidence(solution_payload.get("solution_energies") or [])))
        regions.append(
            OwnedBendRegion(
                bend_id=f"OB{next_index}",
                bend_class=bend.bend_class,
                incident_flange_ids=tuple(attachment_eval["selected_pair"] or bend.incident_flange_ids),
                canonical_bend_key=bend.canonical_bend_key,
                owned_atom_ids=tuple(sorted(component)),
                anchor=tuple(float(v) for v in support_centroid.tolist()),
                axis_direction=tuple(float(v) for v in axis.tolist()),
                support_centroid=tuple(float(v) for v in support_centroid.tolist()),
                span_endpoints=(
                    tuple(float(v) for v in start.tolist()),
                    tuple(float(v) for v in end.tolist()),
                ),
                angle_deg=float(bend.angle_deg),
                radius_mm=bend.radius_mm,
                visibility_state=bend.visibility_state,
                support_mass=support_mass,
                admissible=admissible,
                admissibility_reasons=tuple(reasons),
                post_bend_class=_post_bend_class_for_region(bend, component, atom_graph, atom_lookup),
                debug_confidence=confidence,
            )
        )
        next_index += 1
    return _suppress_tiny_cross_pair_marker_clusters(
        [item for item in regions if item.admissible],
        local_spacing_mm=float(getattr(atom_graph, "local_spacing_mm", 1.0) or 1.0),
    )


def _suppress_tiny_cross_pair_marker_clusters(
    regions: Sequence[OwnedBendRegion],
    *,
    local_spacing_mm: float,
    tiny_atom_count_max: int = 5,
) -> List[OwnedBendRegion]:
    """Drop tiny duplicate-like cross-pair bend fragments before count/render export.

    These fragments are usually produced when a small residual support patch can be
    explained by several neighboring flange-pair attachments. Keeping the larger
    nearby support and suppressing the tiny fragment is safer than rendering two
    clustered bend markers on top of the same visible bend area.
    """
    active = set(range(len(regions)))
    threshold_mm = max(35.0, 10.0 * float(local_spacing_mm or 1.0))

    def atom_count(index: int) -> int:
        return len(regions[index].owned_atom_ids)

    def pair(index: int) -> Tuple[str, ...]:
        return tuple(sorted(str(value) for value in regions[index].incident_flange_ids))

    while True:
        candidate_edges: List[Tuple[int, int]] = []
        ordered = sorted(active)
        for offset, lhs in enumerate(ordered):
            lhs_pair = pair(lhs)
            if len(lhs_pair) != 2:
                continue
            lhs_anchor = np.asarray(regions[lhs].anchor, dtype=np.float64)
            for rhs in ordered[offset + 1 :]:
                rhs_pair = pair(rhs)
                if len(rhs_pair) != 2 or lhs_pair == rhs_pair:
                    continue
                min_atoms = min(atom_count(lhs), atom_count(rhs))
                if min_atoms > tiny_atom_count_max:
                    continue
                rhs_anchor = np.asarray(regions[rhs].anchor, dtype=np.float64)
                if float(np.linalg.norm(lhs_anchor - rhs_anchor)) <= threshold_mm:
                    candidate_edges.append((lhs, rhs))
        if not candidate_edges:
            break

        issue_degree: Dict[int, int] = defaultdict(int)
        for lhs, rhs in candidate_edges:
            if atom_count(lhs) <= tiny_atom_count_max:
                issue_degree[lhs] += 1
            if atom_count(rhs) <= tiny_atom_count_max:
                issue_degree[rhs] += 1
        if not issue_degree:
            break
        drop_index = max(
            issue_degree,
            key=lambda index: (
                issue_degree[index],
                -atom_count(index),
                -float(regions[index].support_mass),
                -float(regions[index].debug_confidence),
                regions[index].bend_id,
            ),
        )
        active.remove(drop_index)

    return [region for index, region in enumerate(regions) if index in active]


def build_owned_region_renderable_objects(
    owned_bend_regions: Sequence[OwnedBendRegion],
    atom_graph: Optional[SurfaceAtomGraph] = None,
) -> List[Dict[str, Any]]:
    renderable: List[Dict[str, Any]] = []
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms} if atom_graph is not None else {}
    for region in owned_bend_regions:
        owned_points = [
            np.asarray(atom_lookup[atom_id].centroid, dtype=np.float64)
            for atom_id in region.owned_atom_ids
            if atom_id in atom_lookup
        ]
        if owned_points:
            support_points = np.asarray(owned_points, dtype=np.float64)
            support_centroid = np.asarray(region.support_centroid, dtype=np.float64)
            distances = np.linalg.norm(support_points - support_centroid, axis=1)
            anchor_np = support_points[int(np.argmin(distances))]
            axis = np.asarray(region.axis_direction, dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-9:
                axis = np.asarray(region.span_endpoints[1], dtype=np.float64) - np.asarray(region.span_endpoints[0], dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis))
            axis = axis / max(axis_norm, 1e-9)
            projections = support_points @ axis
            if len(support_points) >= 6:
                low = float(np.quantile(projections, 0.05))
                high = float(np.quantile(projections, 0.95))
            else:
                low = float(np.min(projections))
                high = float(np.max(projections))
            start_np = support_points[int(np.argmin(np.abs(projections - low)))]
            end_np = support_points[int(np.argmin(np.abs(projections - high)))]
            start = [float(v) for v in start_np.tolist()]
            end = [float(v) for v in end_np.tolist()]
            anchor = [float(v) for v in anchor_np.tolist()]
            owned_support_points = [[float(v) for v in point.tolist()] for point in support_points]
        else:
            start = list(region.span_endpoints[0])
            end = list(region.span_endpoints[1])
            anchor = list(region.anchor)
            owned_support_points = []
        renderable.append(
            {
                "bend_id": region.bend_id,
                "label": region.bend_id,
                "status": "WARNING",
                "source_kind": "patch_graph_owned_region",
                "cad_line": {
                    "raw_start": start,
                    "raw_end": end,
                    "display_start": start,
                    "display_end": end,
                },
                "detected_line": {
                    "raw_start": start,
                    "raw_end": end,
                    "display_start": start,
                    "display_end": end,
                },
                "anchor": anchor,
                "detail_segments": [
                    region.bend_class.replace("_", " "),
                    f"θ≈{region.angle_deg:.1f}°",
                    f"atoms {len(region.owned_atom_ids)}",
                    f"conf {region.debug_confidence:.2f}",
                ],
                "metadata": {
                    "incident_flange_ids": list(region.incident_flange_ids),
                    "owned_atom_ids": list(region.owned_atom_ids),
                    "owned_support_points": owned_support_points,
                    "render_anchor_source": "owned_atom_medoid" if owned_support_points else "owned_region_anchor",
                    "angle_deg": region.angle_deg,
                    "radius_mm": region.radius_mm,
                    "visibility_state": region.visibility_state,
                    "debug_confidence": region.debug_confidence,
                },
            }
        )
    return renderable


def _render_owned_region_atom_projection(
    *,
    result: DecompositionResult,
    output_path: Path,
    width_px: int = 1920,
    height_px: int = 1080,
) -> None:
    """Render an exact atom-space ownership projection for F1 visual QA."""
    import matplotlib.pyplot as plt

    atoms = tuple(result.atom_graph.atoms)
    if not atoms:
        return

    atom_lookup = {atom.atom_id: atom for atom in atoms}
    points = np.asarray([atom.centroid for atom in atoms], dtype=np.float64)
    front = np.asarray([-0.52, -0.52, -0.67], dtype=np.float64)
    front = front / max(float(np.linalg.norm(front)), 1e-9)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(front, up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, front)
    up = up / max(float(np.linalg.norm(up)), 1e-9)

    xy = np.column_stack((points @ right, points @ up))
    region_colors = [
        "#f97316",
        "#0ea5e9",
        "#22c55e",
        "#eab308",
        "#ef4444",
        "#8b5cf6",
        "#14b8a6",
        "#f43f5e",
        "#84cc16",
        "#06b6d4",
        "#d946ef",
        "#fb923c",
    ]

    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax.set_facecolor("#eef4fb")
    fig.patch.set_facecolor("white")
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=2.0,
        c="#94a3b8",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    projected_labels: List[Tuple[float, float]] = []
    for index, region in enumerate(result.owned_bend_regions):
        color = region_colors[index % len(region_colors)]
        owned_points = np.asarray(
            [
                atom_lookup[atom_id].centroid
                for atom_id in region.owned_atom_ids
                if atom_id in atom_lookup
            ],
            dtype=np.float64,
        )
        if owned_points.size == 0:
            continue

        owned_xy = np.column_stack((owned_points @ right, owned_points @ up))
        centroid = np.asarray(region.support_centroid, dtype=np.float64)
        centroid_xy = (float(np.dot(centroid, right)), float(np.dot(centroid, up)))
        span_start = np.asarray(region.span_endpoints[0], dtype=np.float64)
        span_end = np.asarray(region.span_endpoints[1], dtype=np.float64)
        span_xy = np.asarray(
            [
                [float(np.dot(span_start, right)), float(np.dot(span_start, up))],
                [float(np.dot(span_end, right)), float(np.dot(span_end, up))],
            ],
            dtype=np.float64,
        )

        ax.scatter(
            owned_xy[:, 0],
            owned_xy[:, 1],
            s=18.0,
            c=color,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
        )
        ax.plot(
            span_xy[:, 0],
            span_xy[:, 1],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            solid_capstyle="round",
            zorder=6,
        )
        ax.scatter(
            [centroid_xy[0]],
            [centroid_xy[1]],
            s=70.0,
            c=color,
            edgecolors="#ffffff",
            linewidths=1.5,
            zorder=7,
        )

        label_dx = 12 if index % 2 == 0 else -12
        label_dy = 10 + (index % 4) * 8
        for prev_x, prev_y in projected_labels:
            if abs(prev_x - centroid_xy[0]) < 20 and abs(prev_y - centroid_xy[1]) < 20:
                label_dy += 12
        projected_labels.append(centroid_xy)
        ax.annotate(
            f"{region.bend_id}\n{len(region.owned_atom_ids)} atoms\n{region.angle_deg:.1f} deg",
            xy=centroid_xy,
            xytext=(label_dx, label_dy),
            textcoords="offset points",
            ha="left" if label_dx > 0 else "right",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#111827",
            linespacing=1.12,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#ffffff",
                edgecolor=color,
                linewidth=1.2,
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=1.0,
                alpha=0.92,
                shrinkA=3,
                shrinkB=3,
            ),
            zorder=8,
        )

    ax.set_title(
        f"Raw F1 Owned Bend Projection - {len(result.owned_bend_regions)} owned support regions",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.09)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, format="png", dpi=100, facecolor="white")
    plt.close(fig)


def render_decomposition_artifacts(
    *,
    result: DecompositionResult,
    output_dir: str | Path,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderable = build_owned_region_renderable_objects(result.owned_bend_regions, atom_graph=result.atom_graph)
    manifest = {
        "generated_at": str(np.datetime64("now")),
        "source_kind": "patch_graph_f1",
        "owned_region_semantics": "raw_f1_owned_support_regions_not_always_accepted_candidate_count",
        "scan_path": result.scan_path,
        "raw_f1_estimated_bend_count": result.exact_bend_count,
        "raw_f1_estimated_bend_count_range": list(result.bend_count_range),
        "raw_f1_owned_region_count": len(result.owned_bend_regions),
        "abstained": False,
        "abstain_reasons": [],
        "overview_image": "patch_graph_owned_region_overview.png",
        "atom_projection_image": "patch_graph_owned_region_atom_projection.png",
        "bends": renderable,
    }
    (out_dir / "owned_region_manifest.json").write_text(_json_dump(manifest), encoding="utf-8")
    renderer = Model3DSnapshotRenderer()
    atom_projection_path = out_dir / "patch_graph_owned_region_atom_projection.png"
    _render_owned_region_atom_projection(result=result, output_path=atom_projection_path)
    overview_path = out_dir / "patch_graph_owned_region_overview.png"
    overview_path.write_bytes(
        renderer.render_bend_status_overlay(
            bends=renderable,
            reference_mesh_path=result.scan_path,
            view="iso",
            width=1920,
            height=1080,
            focused_bend_ids=None,
            show_issue_callouts=False,
            compact_bend_labels=True,
            title_override=f"Raw F1 Owned Bend Regions - {len(renderable)} owned support regions",
        )
    )
    overview_view_paths: Dict[str, str] = {"iso": str(overview_path)}
    for view in ("front", "side", "top"):
        view_path = out_dir / f"patch_graph_owned_region_overview_{view}.png"
        view_path.write_bytes(
            renderer.render_bend_status_overlay(
                bends=renderable,
                reference_mesh_path=result.scan_path,
                view=view,
                width=1920,
                height=1080,
                focused_bend_ids=None,
                show_issue_callouts=False,
                compact_bend_labels=True,
                title_override=(
                    f"Raw F1 Owned Bend Regions - {len(renderable)} owned support regions "
                    f"({view.capitalize()} view)"
                ),
            )
        )
        overview_view_paths[view] = str(view_path)
    scan_context_path = out_dir / "patch_graph_scan_context_no_final_markers.png"
    scan_context_path.write_bytes(
        renderer.render_bend_status_overlay(
            bends=[],
            reference_mesh_path=result.scan_path,
            view="iso",
            width=1920,
            height=1080,
            focused_bend_ids=None,
            show_issue_callouts=False,
            compact_bend_labels=True,
            title_override="Scan Context - No Accepted Final Bend Markers",
        )
    )
    scan_context_view_paths: Dict[str, str] = {"iso": str(scan_context_path)}
    for view in ("front", "side", "top"):
        view_path = out_dir / f"patch_graph_scan_context_no_final_markers_{view}.png"
        view_path.write_bytes(
            renderer.render_bend_status_overlay(
                bends=[],
                reference_mesh_path=result.scan_path,
                view=view,
                width=1920,
                height=1080,
                focused_bend_ids=None,
                show_issue_callouts=False,
                compact_bend_labels=True,
                title_override=f"Scan Context - No Accepted Final Bend Markers ({view.capitalize()} view)",
            )
        )
        scan_context_view_paths[view] = str(view_path)
    focus_paths: List[str] = []
    for bend in renderable:
        bend_id = str(bend.get("bend_id") or "")
        if not bend_id:
            continue
        focus_path = out_dir / f"patch_graph_owned_region_{bend_id}.png"
        focus_path.write_bytes(
            renderer.render_bend_status_overlay(
                bends=renderable,
                reference_mesh_path=result.scan_path,
                view="iso",
                width=1920,
                height=1080,
                focused_bend_ids=[bend_id],
                show_issue_callouts=False,
                title_override=f"Raw F1 Owned Region {bend_id}",
            )
        )
        focus_paths.append(str(focus_path))
    return {
        "manifest_path": str(out_dir / "owned_region_manifest.json"),
        "overview_path": str(overview_path),
        "overview_view_paths": overview_view_paths,
        "scan_context_path": str(scan_context_path),
        "scan_context_view_paths": scan_context_view_paths,
        "atom_projection_path": str(atom_projection_path),
        "focus_paths": focus_paths,
    }


def _load_cached_research_inputs(cache_dir: str | Path) -> Optional[Dict[str, Any]]:
    cache_path = Path(cache_dir) / "research_inputs.json"
    if not cache_path.exists():
        return None
    import json

    return json.loads(cache_path.read_text(encoding="utf-8"))


def _write_cached_research_inputs(
    *,
    cache_dir: str | Path,
    payload: Mapping[str, Any],
) -> None:
    cache_path = Path(cache_dir) / "research_inputs.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _surface_atom_graph_from_dict(payload: Mapping[str, Any]) -> SurfaceAtomGraph:
    return SurfaceAtomGraph(
        atoms=tuple(
            SurfaceAtom(
                atom_id=str(item["atom_id"]),
                voxel_key=tuple(int(value) for value in item["voxel_key"]),
                point_indices=tuple(int(value) for value in item["point_indices"]),
                centroid=tuple(float(value) for value in item["centroid"]),
                normal=tuple(float(value) for value in item["normal"]),
                local_spacing_mm=float(item["local_spacing_mm"]),
                curvature_proxy=float(item["curvature_proxy"]),
                axis_direction=tuple(float(value) for value in item["axis_direction"]),
                weight=float(item["weight"]),
                area_proxy_mm2=float(item["area_proxy_mm2"]),
            )
            for item in payload.get("atoms") or []
        ),
        adjacency={str(key): tuple(str(value) for value in values) for key, values in (payload.get("adjacency") or {}).items()},
        edge_weights={str(key): float(value) for key, value in (payload.get("edge_weights") or {}).items()},
        voxel_size_mm=float(payload.get("voxel_size_mm") or 0.0),
        local_spacing_mm=float(payload.get("local_spacing_mm") or 0.0),
    )


def _bend_hypothesis_from_dict(payload: Mapping[str, Any]) -> BendHypothesis:
    incident = tuple(str(value) for value in payload.get("incident_flange_ids") or ())
    canonical_pair = _canonical_flange_pair(payload.get("canonical_flange_pair") or incident)
    seed_incident = tuple(str(value) for value in payload.get("seed_incident_flange_ids") or incident)
    candidate_pairs = _unique_flange_pairs(payload.get("candidate_incident_flange_pairs") or (canonical_pair,))
    return BendHypothesis(
        bend_id=str(payload["bend_id"]),
        bend_class=str(payload["bend_class"]),
        source_kind=str(payload["source_kind"]),
        incident_flange_ids=incident,
        canonical_flange_pair=canonical_pair,
        canonical_bend_key=str(payload["canonical_bend_key"]),
        atom_ids=tuple(str(value) for value in payload.get("atom_ids") or ()),
        anchor=tuple(float(value) for value in payload.get("anchor") or (0.0, 0.0, 0.0)),
        axis_direction=tuple(float(value) for value in payload.get("axis_direction") or (1.0, 0.0, 0.0)),
        angle_deg=float(payload.get("angle_deg") or 0.0),
        radius_mm=None if payload.get("radius_mm") is None else float(payload.get("radius_mm")),
        visibility_state=str(payload.get("visibility_state") or "internal"),
        confidence=float(payload.get("confidence") or 0.0),
        cross_width_mm=float(payload.get("cross_width_mm") or 0.0),
        span_mm=float(payload.get("span_mm") or 0.0),
        candidate_classes=tuple(str(value) for value in payload.get("candidate_classes") or (str(payload["bend_class"]),)),
        source_bend_ids=tuple(str(value) for value in payload.get("source_bend_ids") or ()),
        seed_incident_flange_ids=seed_incident,
        candidate_incident_flange_pairs=candidate_pairs,
    )


def _canonical_flange_pair(pair: Sequence[Any]) -> Tuple[str, ...]:
    values = tuple(sorted({str(value) for value in pair if value is not None and str(value)}))
    return values[:2] if len(values) > 2 else values


def _unique_flange_pairs(pairs: Sequence[Any]) -> Tuple[Tuple[str, str], ...]:
    unique: List[Tuple[str, str]] = []
    seen = set()
    for raw_pair in pairs:
        pair = _canonical_flange_pair(raw_pair)
        if len(pair) != 2 or pair in seen:
            continue
        seen.add(pair)
        unique.append((pair[0], pair[1]))
    return tuple(unique)


def _attachment_seed_delta(pair: Sequence[str], seed_pair: Sequence[str]) -> float:
    pair_set = set(_canonical_flange_pair(pair))
    seed_set = set(_canonical_flange_pair(seed_pair))
    if len(pair_set) != 2 or len(seed_set) != 2:
        return 0.6 if pair_set else 1.0
    if pair_set == seed_set:
        return 0.0
    if pair_set & seed_set:
        return 0.35
    return 0.8


def _ensure_normals(points: np.ndarray, normals: Optional[np.ndarray], local_spacing_mm: float) -> np.ndarray:
    if normals is not None and len(normals) == len(points):
        return np.asarray(normals, dtype=np.float64)
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(local_spacing_mm * 4.0, 3.0),
            max_nn=30,
        )
    )
    return np.asarray(pcd.normals, dtype=np.float64)


def _neighbor_keys(key: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                yield (key[0] + dx, key[1] + dy, key[2] + dz)


def _edge_key(lhs: str, rhs: str) -> str:
    return "::".join(sorted((lhs, rhs)))


def _rho(value: float) -> float:
    value = abs(float(value))
    return float(min(value * value, 4.0))


def _atom_adjacency_weight(lhs: SurfaceAtom, rhs: SurfaceAtom, voxel_size_mm: float) -> float:
    lhs_centroid = np.asarray(lhs.centroid, dtype=np.float64)
    rhs_centroid = np.asarray(rhs.centroid, dtype=np.float64)
    dist = float(np.linalg.norm(lhs_centroid - rhs_centroid))
    alignment = abs(float(np.dot(_normalize_direction(lhs.normal), _normalize_direction(rhs.normal))))
    return float(max(0.05, np.exp(-dist / max(voxel_size_mm, 1e-6)) * (0.5 + 0.5 * alignment)))


def _point_to_plane_distance(point: Sequence[float], normal: Sequence[float], d: float) -> float:
    return float(abs(np.dot(np.asarray(point, dtype=np.float64), _normalize_direction(normal)) + float(d)))


def _plane_pair_line(
    flange_a: FlangeHypothesis,
    flange_b: FlangeHypothesis,
    ref: Sequence[float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    n1 = _normalize_direction(flange_a.normal)
    n2 = _normalize_direction(flange_b.normal)
    axis = np.cross(n1, n2)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return None
    axis = axis / norm
    matrix = np.vstack([n1, n2])
    residual = matrix @ np.asarray(ref, dtype=np.float64) + np.asarray([flange_a.d, flange_b.d], dtype=np.float64)
    anchor = np.asarray(ref, dtype=np.float64) - matrix.T @ np.linalg.pinv(matrix @ matrix.T) @ residual
    return anchor, axis


def _interval_overlap_score(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    lhs_low, lhs_high = min(lhs), max(lhs)
    rhs_low, rhs_high = min(rhs), max(rhs)
    overlap = max(0.0, min(lhs_high, rhs_high) - max(lhs_low, rhs_low))
    denom = max(min(lhs_high - lhs_low, rhs_high - rhs_low), 1e-6)
    return float(max(0.0, min(1.0, overlap / denom)))


def _fit_plane(points: np.ndarray) -> Dict[str, Any]:
    centroid = np.mean(points, axis=0)
    covariance = np.cov((points - centroid).T) if len(points) >= 3 else np.eye(3) * 1e-6
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = _normalize_direction(eigenvectors[:, np.argmin(eigenvalues)])
    d = float(-np.dot(normal, centroid))
    return {"normal": normal, "centroid": centroid, "d": d}


def _connected_components(
    atom_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    labels: Optional[Mapping[str, str]] = None,
) -> List[List[str]]:
    remaining = set(atom_ids)
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        label = labels.get(start) if labels is not None else None
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in remaining:
                    continue
                if labels is not None and labels.get(neighbor) != label:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (len(item), item), reverse=True)


def _append_limited(items: List[Any], item: Any, limit: int = 32) -> None:
    if len(items) < limit:
        items.append(item)


def _dedupe_flange_hypotheses(
    hypotheses: Sequence[FlangeHypothesis],
    atom_graph: SurfaceAtomGraph,
) -> List[FlangeHypothesis]:
    kept: List[FlangeHypothesis] = []
    for candidate in hypotheses:
        duplicate = False
        for existing in kept:
            overlap = len(set(candidate.atom_ids) & set(existing.atom_ids))
            min_size = min(len(candidate.atom_ids), len(existing.atom_ids))
            if min_size <= 0:
                continue
            if overlap / min_size >= 0.8 and _angle_deg(candidate.normal, existing.normal) <= 8.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _flange_orientation_family_count(hypotheses: Sequence[FlangeHypothesis]) -> int:
    families: List[np.ndarray] = []
    for flange in hypotheses:
        normal = _normalize_direction(flange.normal)
        duplicate = any(abs(float(np.dot(normal, family))) >= 0.9659 for family in families)
        if not duplicate:
            families.append(normal)
    return len(families)


def _dedupe_bend_hypotheses(
    hypotheses: Sequence[BendHypothesis],
    flange_lookup: Mapping[str, FlangeHypothesis],
) -> List[BendHypothesis]:
    kept: List[BendHypothesis] = []
    for candidate in hypotheses:
        duplicate = False
        for existing in kept:
            if _angle_deg(candidate.axis_direction, existing.axis_direction) > 8.0:
                continue
            if abs(float(candidate.angle_deg) - float(existing.angle_deg)) > 12.0:
                continue
            anchor_delta = np.linalg.norm(np.asarray(candidate.anchor) - np.asarray(existing.anchor))
            candidate_atoms = set(candidate.atom_ids)
            existing_atoms = set(existing.atom_ids)
            overlap_ratio = 0.0
            if candidate_atoms and existing_atoms:
                overlap_ratio = len(candidate_atoms & existing_atoms) / max(1, min(len(candidate_atoms), len(existing_atoms)))
            candidate_pairs = set(_bend_candidate_attachment_pairs(candidate) or _unique_flange_pairs([candidate.incident_flange_ids]))
            existing_pairs = set(_bend_candidate_attachment_pairs(existing) or _unique_flange_pairs([existing.incident_flange_ids]))
            same_incident = bool(candidate_pairs & existing_pairs)
            if same_incident and anchor_delta <= max(candidate.cross_width_mm, existing.cross_width_mm, 4.0):
                duplicate = True
                break
            if overlap_ratio >= 0.45 and anchor_delta <= max(candidate.span_mm, existing.span_mm, 8.0) * 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _ensure_candidate_incident_flange_pairs(
    *,
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    max_pairs: int = 8,
) -> List[BendHypothesis]:
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    atom_to_flanges: Dict[str, List[str]] = defaultdict(list)
    for flange in flange_hypotheses:
        for atom_id in flange.atom_ids:
            atom_to_flanges[atom_id].append(flange.flange_id)

    normalized: List[BendHypothesis] = []
    for bend in bend_hypotheses:
        seed_pair = _canonical_flange_pair(bend.seed_incident_flange_ids or bend.canonical_flange_pair or bend.incident_flange_ids)
        pair_scores: Dict[Tuple[str, str], float] = {pair: 4.0 for pair in _unique_flange_pairs([seed_pair])}
        for pair in bend.candidate_incident_flange_pairs:
            canonical_pair = _canonical_flange_pair(pair)
            if len(canonical_pair) == 2:
                pair_scores[(canonical_pair[0], canonical_pair[1])] = max(pair_scores.get((canonical_pair[0], canonical_pair[1]), 0.0), 3.0)

        touched_scores: Dict[str, float] = defaultdict(float)
        for atom_id in bend.atom_ids:
            for flange_id in atom_to_flanges.get(atom_id, ()):
                touched_scores[flange_id] += 2.0
            for neighbor_id in atom_graph.adjacency.get(atom_id, ()):
                edge_weight = float(atom_graph.edge_weights.get(_edge_key(atom_id, neighbor_id), 1.0))
                for flange_id in atom_to_flanges.get(neighbor_id, ()):
                    touched_scores[flange_id] += edge_weight

        touched = sorted(touched_scores, key=lambda flange_id: (-touched_scores[flange_id], flange_id))[:10]
        for index, lhs in enumerate(touched):
            for rhs in touched[index + 1 :]:
                if lhs not in flange_lookup or rhs not in flange_lookup:
                    continue
                angle = _angle_deg(flange_lookup[lhs].normal, flange_lookup[rhs].normal)
                if angle < 15.0 or angle > 170.0:
                    continue
                pair = _canonical_flange_pair((lhs, rhs))
                if len(pair) != 2:
                    continue
                score = touched_scores[lhs] + touched_scores[rhs]
                if pair == seed_pair:
                    score += 4.0
                elif seed_pair and (pair[0] in seed_pair or pair[1] in seed_pair):
                    score += 1.0
                pair_scores[(pair[0], pair[1])] = max(pair_scores.get((pair[0], pair[1]), 0.0), score)

        ranked_pairs = tuple(
            pair
            for pair, _ in sorted(
                pair_scores.items(),
                key=lambda item: (-item[1], _attachment_seed_delta(item[0], seed_pair), item[0]),
            )[:max_pairs]
        )
        if not ranked_pairs and len(seed_pair) == 2:
            ranked_pairs = ((seed_pair[0], seed_pair[1]),)

        selected_pair = seed_pair if len(seed_pair) == 2 else (ranked_pairs[0] if ranked_pairs else tuple(bend.incident_flange_ids))
        normalized.append(
            BendHypothesis(
                bend_id=bend.bend_id,
                bend_class="transition_only",
                source_kind=bend.source_kind,
                incident_flange_ids=tuple(selected_pair),
                canonical_flange_pair=tuple(selected_pair),
                canonical_bend_key=bend.canonical_bend_key,
                atom_ids=bend.atom_ids,
                anchor=bend.anchor,
                axis_direction=bend.axis_direction,
                angle_deg=bend.angle_deg,
                radius_mm=bend.radius_mm,
                visibility_state=bend.visibility_state,
                confidence=bend.confidence,
                cross_width_mm=bend.cross_width_mm,
                span_mm=bend.span_mm,
                candidate_classes=tuple(sorted(set(bend.candidate_classes or ()) | {"transition_only"})),
                source_bend_ids=bend.source_bend_ids,
                seed_incident_flange_ids=tuple(seed_pair),
                candidate_incident_flange_pairs=ranked_pairs,
            )
        )
    return normalized


def _normalize_bend_hypotheses(
    hypotheses: Sequence[BendHypothesis],
) -> List[BendHypothesis]:
    grouped: Dict[str, List[BendHypothesis]] = defaultdict(list)
    for candidate in hypotheses:
        grouped[_bend_family_key(candidate)].append(candidate)

    normalized: List[BendHypothesis] = []
    next_index = 1
    for _, members in sorted(grouped.items()):
        member_atom_ids = sorted({atom_id for member in members for atom_id in member.atom_ids})
        if not member_atom_ids:
            continue
        anchors = np.asarray([member.anchor for member in members], dtype=np.float64)
        axis = _canonical_axis(np.mean(np.asarray([member.axis_direction for member in members], dtype=np.float64), axis=0))
        candidate_classes = tuple(sorted({candidate.bend_class for candidate in members}))
        canonical_flange_pair = max(
            (candidate.canonical_flange_pair for candidate in members),
            key=lambda item: (len(item), item),
        )
        raw_pairs: List[Sequence[str]] = []
        for candidate in members:
            raw_pairs.extend(candidate.candidate_incident_flange_pairs)
            raw_pairs.append(candidate.canonical_flange_pair)
            raw_pairs.append(candidate.incident_flange_ids)
        candidate_pairs = _unique_flange_pairs(raw_pairs)
        seed_incident = canonical_flange_pair or max(
            (candidate.seed_incident_flange_ids or candidate.incident_flange_ids for candidate in members),
            key=lambda item: (len(item), item),
        )
        mean_angle = float(np.mean([candidate.angle_deg for candidate in members]))
        radius_values = [candidate.radius_mm for candidate in members if candidate.radius_mm not in {None, 0.0}]
        confidence = float(max(candidate.confidence for candidate in members))
        source_bend_ids = tuple(sorted({source_id for candidate in members for source_id in candidate.source_bend_ids if source_id}))
        span_mm = float(max(candidate.span_mm for candidate in members))
        cross_width_mm = float(np.median([candidate.cross_width_mm for candidate in members]))
        visibility_state = "internal"
        if any(candidate.visibility_state != "internal" for candidate in members):
            visibility_state = members[0].visibility_state if members[0].visibility_state != "internal" else next(
                candidate.visibility_state for candidate in members if candidate.visibility_state != "internal"
            )
        canonical_key = _bend_family_key(members[0])
        normalized.append(
            BendHypothesis(
                bend_id=f"B{next_index}",
                bend_class="transition_only",
                source_kind="+".join(sorted({candidate.source_kind for candidate in members})),
                incident_flange_ids=canonical_flange_pair,
                canonical_flange_pair=canonical_flange_pair,
                canonical_bend_key=canonical_key,
                atom_ids=tuple(member_atom_ids),
                anchor=tuple(float(v) for v in np.mean(anchors, axis=0).tolist()),
                axis_direction=tuple(float(v) for v in axis.tolist()),
                angle_deg=mean_angle,
                radius_mm=float(np.median(radius_values)) if radius_values else None,
                visibility_state=visibility_state,
                confidence=confidence,
                cross_width_mm=cross_width_mm,
                span_mm=span_mm,
                candidate_classes=candidate_classes,
                source_bend_ids=source_bend_ids,
                seed_incident_flange_ids=tuple(seed_incident),
                candidate_incident_flange_pairs=candidate_pairs,
            )
        )
        next_index += 1
    return normalized


def _bend_family_key(candidate: BendHypothesis) -> str:
    pair = tuple(sorted(candidate.canonical_flange_pair or candidate.incident_flange_ids))
    axis = _canonical_axis(candidate.axis_direction)
    anchor = np.asarray(candidate.anchor, dtype=np.float64)
    anchor_bucket_scale = max(1.0, float(candidate.cross_width_mm) * 0.75)
    anchor_bucket = np.round(anchor / anchor_bucket_scale) * anchor_bucket_scale
    angle_bucket = round(float(candidate.angle_deg) / 5.0) * 5.0
    span_bucket = round(float(candidate.span_mm) / 20.0) * 20.0
    return "|".join(
        [
            ",".join(pair) if pair else "weak",
            f"{round(float(axis[0]), 2)}:{round(float(axis[1]), 2)}:{round(float(axis[2]), 2)}",
            f"{round(float(anchor_bucket[0]), 1)}:{round(float(anchor_bucket[1]), 1)}:{round(float(anchor_bucket[2]), 1)}",
            f"{round(float(angle_bucket), 1)}",
            f"{round(float(span_bucket), 1)}",
        ]
    )


def _canonical_axis(direction: Sequence[float]) -> np.ndarray:
    axis = _normalize_direction(direction)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0) or (abs(axis[0]) < 1e-6 and abs(axis[1]) < 1e-6 and axis[2] < 0):
        axis = -axis
    return axis


def _match_flange_ids_for_normals(
    normals: Sequence[Any],
    flange_hypotheses: Sequence[FlangeHypothesis],
) -> List[str]:
    matched: List[str] = []
    for normal in normals:
        if normal is None:
            continue
        target = _normalize_direction(normal)
        best = None
        best_score = -1.0
        for flange in flange_hypotheses:
            score = abs(float(np.dot(target, _normalize_direction(flange.normal))))
            if score > best_score:
                best_score = score
                best = flange.flange_id
        if best and best not in matched:
            matched.append(best)
    return matched[:2]


def _visibility_state(anchor: np.ndarray, bbox_points: np.ndarray, local_spacing_mm: float) -> str:
    bbox_min = np.min(bbox_points, axis=0)
    bbox_max = np.max(bbox_points, axis=0)
    margin = max(local_spacing_mm * 2.0, 3.0)
    if np.any(anchor <= bbox_min + margin) or np.any(anchor >= bbox_max - margin):
        return "boundary_truncated"
    return "internal"


def _distance_to_line(point: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> float:
    delta = point - anchor
    orth = delta - np.dot(delta, axis) * axis
    return float(np.linalg.norm(orth))


def _angle_deg(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    lhs_vec = _normalize_direction(lhs)
    rhs_vec = _normalize_direction(rhs)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(lhs_vec, rhs_vec)), -1.0, 1.0))))


def _normalize_count_range(raw_range: Any, estimated_count: Any) -> Tuple[int, int]:
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        low = _coerce_optional_int(raw_range[0])
        high = _coerce_optional_int(raw_range[1])
        if low is not None and high is not None:
            return (min(low, high), max(low, high))
    exact = _coerce_optional_int(estimated_count)
    if exact is None:
        return (0, 0)
    return (exact, exact)


def _allow_no_contact_birth_rescue_for_range(count_range: Sequence[int]) -> bool:
    if len(count_range) < 2:
        return False
    return bool(int(count_range[1]) - int(count_range[0]) >= 2)


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _initial_labels(
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    *,
    initial_mode: str,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    for atom in atom_graph.atoms:
        candidate_labels = _candidate_labels_for_atom(
            atom=atom,
            current_label=None,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
        )
        candidate_flanges = [item for item in flange_hypotheses if item.flange_id in candidate_labels]
        candidate_bends = [item for item in bend_hypotheses if item.bend_id in candidate_labels]
        best_flange = min(
            candidate_flanges,
            key=lambda item: _flange_data_cost(atom, item),
        ) if candidate_flanges else None
        best_bend = min(
            candidate_bends,
            key=lambda item: _bend_data_cost(atom, item, flange_lookup),
        ) if candidate_bends else None
        if initial_mode == "flange_bias":
            labels[atom.atom_id] = best_flange.flange_id if best_flange else "residual"
        elif initial_mode == "bend_bias" and best_bend is not None:
            bend_cost = _bend_data_cost(atom, best_bend, flange_lookup)
            flange_cost = _flange_data_cost(atom, best_flange) if best_flange else 999.0
            labels[atom.atom_id] = best_bend.bend_id if bend_cost <= flange_cost * 1.05 else (best_flange.flange_id if best_flange else "residual")
        else:
            proposal_ids = list(best_bend.atom_ids) if best_bend else []
            labels[atom.atom_id] = best_bend.bend_id if atom.atom_id in proposal_ids else (best_flange.flange_id if best_flange else "residual")
    return labels


def _solve_base_labeling(
    *,
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    label_types: Mapping[str, str],
    initial_mode: str,
) -> Dict[str, str]:
    labels = _initial_labels(
        atom_graph=atom_graph,
        flange_hypotheses=flange_hypotheses,
        bend_hypotheses=bend_hypotheses,
        initial_mode=initial_mode,
    )
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    bend_lookup = {item.bend_id: item for item in bend_hypotheses}
    for _ in range(6):
        changed = False
        active_labels = set(labels.values())
        next_labels = dict(labels)
        for atom in atom_graph.atoms:
            candidate_labels = _candidate_labels_for_atom(
                atom=atom,
                current_label=labels.get(atom.atom_id),
                flange_hypotheses=flange_hypotheses,
                bend_hypotheses=bend_hypotheses,
            )
            best_label = next(
                (
                    label
                    for label in candidate_labels
                    if label == labels.get(atom.atom_id)
                ),
                candidate_labels[0],
            )
            best_cost = None
            for label in candidate_labels:
                cost = _label_cost(
                    atom=atom,
                    label=label,
                    labels=labels,
                    atom_graph=atom_graph,
                    flange_lookup=flange_lookup,
                    bend_lookup=bend_lookup,
                    label_types=label_types,
                )
                if label not in active_labels:
                    cost += _activation_cost(label, flange_hypotheses, bend_hypotheses)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_label = label
            next_labels[atom.atom_id] = best_label
            changed = changed or best_label != labels.get(atom.atom_id)
        next_labels = _prune_and_refit_labels(
            labels=next_labels,
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
        )
        labels = next_labels
        if not changed:
            break
    return dict(sorted(labels.items()))


def _apply_object_repair_moves(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Dict[str, str],
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    label_types: Mapping[str, str],
    solver_options: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    solver_options = dict(solver_options or {})
    repair_iterations = max(1, int(solver_options.get("repair_iterations") or 4))
    allow_interface_activation = bool(solver_options.get("allow_interface_activation", True))
    repair_before_prune = bool(solver_options.get("repair_before_prune", False))
    enable_birth_support_rescue = bool(solver_options.get("enable_birth_support_rescue", False))
    allow_geometry_backed_single_contact = bool(solver_options.get("allow_geometry_backed_single_contact", False))
    allow_hypothesis_support_contacts = bool(solver_options.get("allow_hypothesis_support_contacts", False))
    allow_cross_axis_attachment_match = bool(solver_options.get("allow_cross_axis_attachment_match", False))
    allow_mixed_transition_single_contact = bool(solver_options.get("allow_mixed_transition_single_contact", False))
    allow_no_contact_birth_rescue = bool(solver_options.get("allow_no_contact_birth_rescue", False))
    birth_rescue_max_active_bends = int(solver_options.get("birth_rescue_max_active_bends") or 0)
    prioritize_residual_birth_rescue = bool(solver_options.get("prioritize_residual_birth_rescue", False))
    enable_selected_pair_duplicate_prune = bool(solver_options.get("enable_selected_pair_duplicate_prune", False))
    selected_pair_duplicate_prune_min_active_bends = int(solver_options.get("selected_pair_duplicate_prune_min_active_bends") or 0)
    selected_pair_duplicate_prune_alias_atom_max = int(solver_options.get("selected_pair_duplicate_prune_alias_atom_max") or 4)
    bend_lookup = {item.bend_id: item for item in bend_hypotheses}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    repaired = dict(labels)
    diagnostics: Dict[str, Any] = {
        "repair_steps": [],
        "solver_profile": solver_options.get("solver_profile", "default"),
        "enabled_moves": list(solver_options.get("enabled_moves") or ()),
        "allow_interface_activation": allow_interface_activation,
        "repair_before_prune": repair_before_prune,
        "enable_birth_support_rescue": enable_birth_support_rescue,
        "allow_geometry_backed_single_contact": allow_geometry_backed_single_contact,
        "allow_hypothesis_support_contacts": allow_hypothesis_support_contacts,
        "allow_cross_axis_attachment_match": allow_cross_axis_attachment_match,
        "allow_mixed_transition_single_contact": allow_mixed_transition_single_contact,
        "allow_no_contact_birth_rescue": allow_no_contact_birth_rescue,
        "birth_rescue_max_active_bends": birth_rescue_max_active_bends,
        "prioritize_residual_birth_rescue": prioritize_residual_birth_rescue,
        "enable_selected_pair_duplicate_prune": enable_selected_pair_duplicate_prune,
    }
    for _ in range(repair_iterations):
        previous = dict(repaired)
        born: List[str] = []
        pruned: List[str] = []
        if repair_before_prune and allow_interface_activation:
            repaired, born = _activate_interface_bends(
                atom_graph=atom_graph,
                labels=repaired,
                flange_hypotheses=flange_hypotheses,
                bend_hypotheses=bend_hypotheses,
                label_types=label_types,
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
            )
        prune_diagnostics: Dict[str, Any] = {}
        repaired, pruned = _drop_duplicate_and_inadmissible_bends(
            atom_graph=atom_graph,
            labels=repaired,
            bend_lookup=bend_lookup,
            flange_lookup=flange_lookup,
            label_types=label_types,
            diagnostics=prune_diagnostics,
            allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
            allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
        )
        selected_pair_pruned: List[str] = []
        selected_pair_prune_diagnostics: Dict[str, Any] = {}
        if enable_selected_pair_duplicate_prune:
            repaired, selected_pair_pruned = _drop_selected_pair_duplicate_bends(
                atom_graph=atom_graph,
                labels=repaired,
                bend_lookup=bend_lookup,
                label_types=label_types,
                flange_lookup=flange_lookup,
                diagnostics=selected_pair_prune_diagnostics,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                min_active_bends=selected_pair_duplicate_prune_min_active_bends,
                alias_atom_count_max=selected_pair_duplicate_prune_alias_atom_max,
            )
        if not repair_before_prune and allow_interface_activation and not born:
            repaired, born = _activate_interface_bends(
                atom_graph=atom_graph,
                labels=repaired,
                flange_hypotheses=flange_hypotheses,
                bend_hypotheses=bend_hypotheses,
                label_types=label_types,
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
            )
        rescued: List[str] = []
        rescue_diagnostics: Dict[str, Any] = {}
        if enable_birth_support_rescue:
            repaired, rescued = _rescue_interface_birth_supports(
                atom_graph=atom_graph,
                labels=repaired,
                flange_hypotheses=flange_hypotheses,
                bend_hypotheses=bend_hypotheses,
                label_types=label_types,
                diagnostics=rescue_diagnostics,
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_no_contact_birth_rescue=allow_no_contact_birth_rescue,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
                max_active_bends=birth_rescue_max_active_bends,
                prioritize_residual_support=prioritize_residual_birth_rescue,
            )
        repaired = _prune_and_refit_labels(
            labels=repaired,
            atom_graph=atom_graph,
            flange_hypotheses=flange_hypotheses,
            bend_hypotheses=bend_hypotheses,
        )
        diagnostics["repair_steps"].append(
            {
                "pruned_bend_labels": pruned,
                "prune_diagnostics": prune_diagnostics,
                "selected_pair_pruned_bend_labels": selected_pair_pruned,
                "selected_pair_prune_diagnostics": selected_pair_prune_diagnostics,
                "activated_bend_labels": born,
                "rescued_birth_labels": rescued,
                "birth_rescue_diagnostics": rescue_diagnostics,
                "active_bend_labels": sorted(
                    {
                        label
                        for label in repaired.values()
                        if label_types.get(label) == "bend"
                    }
                ),
            }
        )
        if repaired == previous:
            break
    return dict(sorted(repaired.items())), diagnostics


def _label_cost(
    *,
    atom: SurfaceAtom,
    label: str,
    labels: Mapping[str, str],
    atom_graph: SurfaceAtomGraph,
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    label_type = label_types.get(label, "outlier")
    if label_type == "flange":
        data_cost = _flange_data_cost(atom, flange_lookup[label])
    elif label_type == "bend":
        data_cost = _bend_data_cost(atom, bend_lookup[label], flange_lookup)
    elif label_type == "residual":
        data_cost = _residual_data_cost(atom)
    else:
        data_cost = 2.25
    smoothness = 0.0
    for neighbor_id in atom_graph.adjacency.get(atom.atom_id, ()):
        neighbor_label = labels.get(neighbor_id)
        if not neighbor_label:
            continue
        edge_weight = atom_graph.edge_weights.get(_edge_key(atom.atom_id, neighbor_id), 0.25)
        if neighbor_label != label:
            smoothness += 0.32 * edge_weight
    return float(data_cost + smoothness)


def _activation_cost(
    label: str,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
) -> float:
    if any(item.flange_id == label for item in flange_hypotheses):
        return 0.25
    if any(item.bend_id == label for item in bend_hypotheses):
        bend = next(item for item in bend_hypotheses if item.bend_id == label)
        if bend.bend_class == "transition_only":
            return 0.45
        if bend.bend_class == "circular_strip":
            return 0.6
        return 0.75
    return 0.0


def _flange_data_cost(atom: SurfaceAtom, flange: Optional[FlangeHypothesis]) -> float:
    if flange is None:
        return 999.0
    plane_distance = _point_to_plane_distance(atom.centroid, flange.normal, flange.d) / max(atom.local_spacing_mm * 2.0, 1.0)
    normal_delta = _angle_deg(atom.normal, flange.normal) / 25.0
    curvature = min(1.0, atom.curvature_proxy * 20.0)
    support_term = -0.4 if atom.atom_id in flange.atom_ids else 0.35
    return float(max(0.0, plane_distance + normal_delta + curvature + support_term))


def _residual_data_cost(atom: SurfaceAtom) -> float:
    curvature_signal = min(1.0, float(atom.curvature_proxy) * 20.0)
    return float(1.10 + 0.35 * curvature_signal)


def _bend_data_cost(
    atom: SurfaceAtom,
    bend: BendHypothesis,
    flange_lookup: Mapping[str, FlangeHypothesis],
) -> float:
    centroid = np.asarray(atom.centroid, dtype=np.float64)
    anchor = np.asarray(bend.anchor, dtype=np.float64)
    axis = _normalize_direction(bend.axis_direction)
    line_distance = _distance_to_line(centroid, anchor, axis) / max(bend.cross_width_mm * 0.75, atom.local_spacing_mm * 1.25, 1.0)
    axis_delta = _angle_deg(atom.axis_direction, axis) / 35.0
    normal_cost = 0.75
    normal_costs: List[float] = []
    for pair in _bend_candidate_attachment_pairs(bend):
        if len(pair) != 2 or pair[0] not in flange_lookup or pair[1] not in flange_lookup:
            continue
        flange_normals = [_normalize_direction(flange_lookup[flange_id].normal) for flange_id in pair]
        atom_normal = _normalize_direction(atom.normal)
        lhs = _angle_deg(atom_normal, flange_normals[0])
        rhs = _angle_deg(atom_normal, flange_normals[1])
        pair_angle = _angle_deg(flange_normals[0], flange_normals[1])
        low = min(lhs, rhs)
        high = max(lhs, rhs)
        if low >= 10.0 and high <= pair_angle + 12.0:
            normal_costs.append(0.15 + _attachment_seed_delta(pair, bend.seed_incident_flange_ids) * 0.05)
        else:
            normal_costs.append(min(1.5, abs((lhs + rhs) - pair_angle) / 35.0 + low / 45.0))
    if normal_costs:
        normal_cost = min(normal_costs)
    class_penalty = 0.1 if bend.bend_class == "transition_only" else 0.2 if bend.bend_class == "circular_strip" else 0.35
    support_term = -0.45 if atom.atom_id in bend.atom_ids else 0.45
    return float(max(0.0, line_distance + axis_delta + normal_cost + class_penalty + support_term))


def _drop_duplicate_and_inadmissible_bends(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Dict[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_mixed_transition_single_contact: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    next_labels = dict(labels)
    pruned: List[str] = []
    stats: Dict[str, Any] = {
        "duplicate_pruned_labels": [],
        "fragment_pruned_atoms": 0,
        "inadmissible_pruned_labels": [],
        "inadmissible_reason_counts": defaultdict(int),
        "inadmissible_label_reasons": {},
        "inadmissible_label_attachment_terms": {},
    }
    active_bends = sorted({label for label in next_labels.values() if label_types.get(label) == "bend"})
    groups: Dict[str, List[str]] = defaultdict(list)
    for bend_id in active_bends:
        groups[bend_lookup[bend_id].canonical_bend_key].append(bend_id)

    for bend_ids in groups.values():
        if len(bend_ids) <= 1:
            continue
        keep = max(
            bend_ids,
            key=lambda bend_id: (
                sum(1 for label in next_labels.values() if label == bend_id),
                bend_lookup[bend_id].confidence,
            ),
        )
        for bend_id in bend_ids:
            if bend_id == keep:
                continue
            if any(label == bend_id for label in next_labels.values()):
                pruned.append(bend_id)
                stats["duplicate_pruned_labels"].append(bend_id)
            for atom_id, label in list(next_labels.items()):
                if label == bend_id:
                    next_labels[atom_id] = "residual"

    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    active_bends = sorted({label for label in next_labels.values() if label_types.get(label) == "bend"})
    for bend_id in active_bends:
        bend_atoms = [atom_id for atom_id, label in next_labels.items() if label == bend_id]
        if not bend_atoms:
            continue
        components = _connected_components(bend_atoms, atom_graph.adjacency, labels=next_labels)
        if len(components) > 1:
            largest = max(components, key=len)
            for component in components:
                if component == largest:
                    continue
                stats["fragment_pruned_atoms"] += len(component)
                for atom_id in component:
                    next_labels[atom_id] = "residual"
        bend_atoms = [atom_id for atom_id, label in next_labels.items() if label == bend_id]
        if not bend_atoms:
            continue
        component = _connected_components(bend_atoms, atom_graph.adjacency, labels=next_labels)[0]
        admissible, reasons = _bend_component_admissibility(
            component=component,
            bend=bend_lookup[bend_id],
            atom_graph=atom_graph,
            labels=next_labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup or {},
            allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
            allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
        )
        if not admissible:
            attachment_eval = _bend_component_attachment_evaluation(
                component=component,
                bend=bend_lookup[bend_id],
                atom_graph=atom_graph,
                labels=next_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup or {},
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
            )
            pruned.append(bend_id)
            stats["inadmissible_pruned_labels"].append(bend_id)
            stats["inadmissible_label_reasons"][bend_id] = list(reasons)
            stats["inadmissible_label_attachment_terms"][bend_id] = {
                "selected_pair": list(attachment_eval.get("selected_pair") or ()),
                "penalty": float(attachment_eval.get("penalty", 999.0)),
                "quality": float(attachment_eval.get("quality", 0.0)),
                "terms": dict(attachment_eval.get("terms") or {}),
            }
            for reason in reasons:
                stats["inadmissible_reason_counts"][reason] += 1
            for atom_id in component:
                next_labels[atom_id] = "residual"
    if diagnostics is not None:
        diagnostics.update(stats)
        diagnostics["inadmissible_reason_counts"] = dict(stats["inadmissible_reason_counts"])
    return next_labels, sorted(set(pruned))


def _activate_interface_bends(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Dict[str, str],
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    label_types: Mapping[str, str],
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_mixed_transition_single_contact: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    next_labels = dict(labels)
    bend_lookup = {item.bend_id: item for item in bend_hypotheses}
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    activated: List[str] = []
    current_energy = _total_energy(
        atom_graph=atom_graph,
        labels=next_labels,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
    )
    for bend in bend_hypotheses:
        if any(label == bend.bend_id for label in next_labels.values()):
            continue
        candidate_pairs = [
            pair
            for pair in _bend_candidate_attachment_pairs(bend)
            if len(pair) == 2
            and pair[0] in flange_lookup
            and pair[1] in flange_lookup
            and _angle_deg(flange_lookup[pair[0]].normal, flange_lookup[pair[1]].normal) >= 20.0
        ]
        if not candidate_pairs:
            continue

        candidate_atoms = sorted(
            atom_id
            for atom_id in bend.atom_ids
            if next_labels.get(atom_id) != bend.bend_id
            and label_types.get(next_labels.get(atom_id)) in {"flange", "residual", "outlier", "bend"}
        )
        if not candidate_atoms:
            continue
        best_component_labels: Optional[Dict[str, str]] = None
        best_component_energy = current_energy
        for component in _connected_components(candidate_atoms, atom_graph.adjacency):
            touches_current_labels = any(
                _component_touches_designated_interface(
                    component=component,
                    designated_flanges=pair,
                    labels=next_labels,
                    adjacency=atom_graph.adjacency,
                    label_types=label_types,
                    visibility_state=bend.visibility_state,
                )
                for pair in candidate_pairs
            )
            touches_source_support = bool(
                bend.source_kind == "interface_birth"
                and any(
                    _component_support_touches_flange_pair(
                        component=component,
                        pair=pair,
                        atom_graph=atom_graph,
                        flange_lookup=flange_lookup,
                    )
                    >= (1 if bend.visibility_state != "internal" else 2)
                    for pair in candidate_pairs
                )
            )
            if not (touches_current_labels or touches_source_support):
                continue
            component_atoms = [atom_lookup[atom_id] for atom_id in component]
            promotion_margin = 0.05 if bend.source_kind == "interface_birth" else 0.3
            seed_atoms = [
                atom
                for atom in component_atoms
                if _promote_atom_to_bend(
                    atom=atom,
                    bend=bend,
                    labels=next_labels,
                    atom_graph=atom_graph,
                    flange_lookup=flange_lookup,
                    bend_lookup=bend_lookup,
                    label_types=label_types,
                    margin=promotion_margin,
                )
            ]
            if not seed_atoms:
                continue
            trial_labels = dict(next_labels)
            for atom_id in component:
                trial_labels[atom_id] = bend.bend_id
            admissible, _ = _bend_component_admissibility(
                component=component,
                bend=bend,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
            )
            if not admissible:
                continue
            trial_energy = _total_energy(
                atom_graph=atom_graph,
                labels=trial_labels,
                flange_lookup=flange_lookup,
                bend_lookup=bend_lookup,
                label_types=label_types,
            )
            if trial_energy + 1e-6 < best_component_energy:
                best_component_energy = trial_energy
                best_component_labels = trial_labels
        if best_component_labels is not None:
            next_labels = best_component_labels
            current_energy = best_component_energy
            activated.append(bend.bend_id)
    return next_labels, sorted(set(activated))


def _drop_selected_pair_duplicate_bends(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Dict[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    diagnostics: Optional[Dict[str, Any]] = None,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    min_active_bends: int = 0,
    alias_atom_count_max: int = 4,
) -> Tuple[Dict[str, str], List[str]]:
    next_labels = dict(labels)
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    bend_atoms = [atom_id for atom_id, label in next_labels.items() if label_types.get(label) == "bend"]
    components = _connected_components(bend_atoms, atom_graph.adjacency, labels=next_labels)
    if min_active_bends and len(components) < int(min_active_bends):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "selected_pair_duplicate_candidates": len(components),
                    "selected_pair_duplicate_pruned": [],
                    "skipped_below_min_active_bends": int(min_active_bends),
                }
            )
        return next_labels, []
    records: List[Dict[str, Any]] = []
    for component in components:
        bend_id = next_labels[component[0]]
        bend = bend_lookup.get(bend_id)
        if bend is None:
            continue
        points = np.asarray([atom_lookup[atom_id].centroid for atom_id in component if atom_id in atom_lookup], dtype=np.float64)
        if len(points) == 0:
            continue
        evaluation = _bend_component_attachment_evaluation(
            component=component,
            bend=bend,
            atom_graph=atom_graph,
            labels=next_labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
        )
        pair = tuple(evaluation.get("selected_pair") or ())
        if len(pair) != 2:
            continue
        mass = float(sum(atom_lookup[atom_id].weight for atom_id in component if atom_id in atom_lookup))
        centroid = np.mean(points, axis=0)
        records.append(
            {
                "bend_id": bend_id,
                "component": list(component),
                "pair": pair,
                "mass": mass,
                "atom_count": len(component),
                "centroid": centroid,
                "quality": float(evaluation.get("quality", 0.0)),
                "penalty": float(evaluation.get("penalty", 999.0)),
                "confidence": float(bend.confidence),
            }
        )

    pruned: List[str] = []
    decisions: List[Dict[str, Any]] = []
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[record["pair"]].append(record)

    local_spacing = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
    alias_atom_count_max = max(1, int(alias_atom_count_max))
    for pair_records in by_pair.values():
        if len(pair_records) <= 1:
            continue
        ordered = sorted(pair_records, key=lambda item: (-item["mass"], -item["atom_count"], item["penalty"], item["bend_id"]))
        kept: List[Dict[str, Any]] = []
        for record in ordered:
            duplicate_of: Optional[Dict[str, Any]] = None
            for kept_record in kept:
                centroid_distance = float(np.linalg.norm(record["centroid"] - kept_record["centroid"]))
                small_alias = min(record["atom_count"], kept_record["atom_count"]) <= alias_atom_count_max
                close = centroid_distance <= max(35.0, 6.0 * local_spacing)
                if small_alias and close:
                    duplicate_of = kept_record
                    break
            if duplicate_of is None:
                kept.append(record)
                continue
            pruned.append(record["bend_id"])
            decisions.append(
                {
                    "pruned": record["bend_id"],
                    "kept": duplicate_of["bend_id"],
                    "pair": list(record["pair"]),
                    "centroid_distance_mm": float(np.linalg.norm(record["centroid"] - duplicate_of["centroid"])),
                    "pruned_atom_count": int(record["atom_count"]),
                    "kept_atom_count": int(duplicate_of["atom_count"]),
                }
            )
            for atom_id in record["component"]:
                if next_labels.get(atom_id) == record["bend_id"]:
                    next_labels[atom_id] = "residual"

    if diagnostics is not None:
        diagnostics.update(
            {
                "selected_pair_duplicate_candidates": len(records),
                "selected_pair_duplicate_pruned": decisions,
            }
        )
    return next_labels, sorted(set(pruned))


def _final_duplicate_cluster_penalty(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    flange_lookup: Mapping[str, FlangeHypothesis],
    label_types: Mapping[str, str],
    alias_atom_count_max: int = 4,
    same_pair_penalty: float = 120.0,
    cross_pair_tiny_penalty: float = 45.0,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Score final MAP candidates that would render as clustered duplicate bend dots.

    This is deliberately separate from repair-time pruning: it lets the solver choose
    a cleaner local optimum without mutating the label map after convergence.
    """
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    components = _connected_components(bend_atoms, atom_graph.adjacency, labels=labels)
    local_spacing = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
    alias_atom_count_max = max(1, int(alias_atom_count_max))
    same_pair_close_mm = max(35.0, 6.0 * local_spacing)
    tiny_cross_pair_close_mm = max(18.0, 4.0 * local_spacing)

    records: List[Dict[str, Any]] = []
    for component in components:
        bend_id = labels[component[0]]
        bend = bend_lookup.get(bend_id)
        if bend is None:
            continue
        points = np.asarray(
            [atom_lookup[atom_id].centroid for atom_id in component if atom_id in atom_lookup],
            dtype=np.float64,
        )
        if len(points) == 0:
            continue
        evaluation = _bend_component_attachment_evaluation(
            component=component,
            bend=bend,
            atom_graph=atom_graph,
            labels=labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
        )
        pair = tuple(evaluation.get("selected_pair") or ())
        records.append(
            {
                "bend_id": bend_id,
                "pair": pair,
                "atom_count": int(len(component)),
                "mass": float(sum(atom_lookup[atom_id].weight for atom_id in component if atom_id in atom_lookup)),
                "centroid": np.mean(points, axis=0),
                "quality": float(evaluation.get("quality", 0.0)),
                "attachment_penalty": float(evaluation.get("penalty", 999.0)),
            }
        )

    penalty = 0.0
    clusters: List[Dict[str, Any]] = []
    for index, lhs in enumerate(records):
        for rhs in records[index + 1 :]:
            distance = float(np.linalg.norm(lhs["centroid"] - rhs["centroid"]))
            min_atom_count = int(min(lhs["atom_count"], rhs["atom_count"]))
            same_pair = len(lhs["pair"]) == 2 and lhs["pair"] == rhs["pair"]
            close_same_pair_alias = same_pair and min_atom_count <= alias_atom_count_max and distance <= same_pair_close_mm
            close_tiny_cross_pair = (
                not same_pair
                and min_atom_count <= 4
                and distance <= tiny_cross_pair_close_mm
            )
            if not close_same_pair_alias and not close_tiny_cross_pair:
                continue

            if close_same_pair_alias:
                reason = "selected_pair_duplicate_cluster"
                item_penalty = float(same_pair_penalty)
            else:
                reason = "tiny_cross_pair_marker_cluster"
                item_penalty = float(cross_pair_tiny_penalty)
            penalty += item_penalty
            clusters.append(
                {
                    "reason": reason,
                    "lhs_bend_id": lhs["bend_id"],
                    "rhs_bend_id": rhs["bend_id"],
                    "lhs_pair": list(lhs["pair"]),
                    "rhs_pair": list(rhs["pair"]),
                    "centroid_distance_mm": round(distance, 4),
                    "lhs_atom_count": int(lhs["atom_count"]),
                    "rhs_atom_count": int(rhs["atom_count"]),
                    "penalty": round(float(item_penalty), 6),
                }
            )

    return float(penalty), {
        "enabled": True,
        "penalty": round(float(penalty), 6),
        "component_count": len(records),
        "clusters": clusters,
        "same_pair_close_mm": round(float(same_pair_close_mm), 4),
        "tiny_cross_pair_close_mm": round(float(tiny_cross_pair_close_mm), 4),
        "alias_atom_count_max": int(alias_atom_count_max),
    }


def _active_bend_component_count(
    labels: Mapping[str, str],
    atom_graph: SurfaceAtomGraph,
    label_types: Mapping[str, str],
) -> int:
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    return len(_connected_components(bend_atoms, atom_graph.adjacency, labels=labels))


def _rescue_interface_birth_supports(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Dict[str, str],
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    label_types: Mapping[str, str],
    max_rescues: int = 8,
    diagnostics: Optional[Dict[str, Any]] = None,
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_no_contact_birth_rescue: bool = False,
    allow_mixed_transition_single_contact: bool = False,
    max_active_bends: int = 0,
    prioritize_residual_support: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    next_labels = dict(labels)
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    rescued: List[str] = []
    active_bend_atoms = {
        atom_id
        for atom_id, label in next_labels.items()
        if label_types.get(label) == "bend"
    }
    def _birth_rescue_sort_key(bend: BendHypothesis) -> Tuple[float, float, int, str]:
        valid_atoms = [atom_id for atom_id in bend.atom_ids if atom_id in atom_lookup]
        if not valid_atoms:
            return (0.0, 0.0, 0, bend.bend_id)
        residual_count = sum(1 for atom_id in valid_atoms if next_labels.get(atom_id) == "residual")
        residual_fraction = residual_count / float(max(1, len(valid_atoms)))
        if prioritize_residual_support and max_active_bends:
            return (-residual_fraction, -float(residual_count), -len(valid_atoms), bend.bend_id)
        return (0.0, 0.0, -len(valid_atoms), bend.bend_id)

    birth_candidates = sorted(
        (bend for bend in bend_hypotheses if bend.source_kind == "interface_birth"),
        key=_birth_rescue_sort_key,
    )
    stats: Dict[str, Any] = {
        "birth_candidates": len(birth_candidates),
        "skipped_already_active": 0,
        "skipped_no_candidate_atoms": 0,
        "skipped_component_too_small": 0,
        "skipped_active_overlap": 0,
        "trimmed_active_overlap_attempts": 0,
        "skipped_no_pair_contact": 0,
        "geometry_backed_no_contact_attempts": 0,
        "skipped_active_count_cap": 0,
        "prioritize_residual_support": bool(prioritize_residual_support and max_active_bends),
        "rejected_admissibility": 0,
        "admissibility_reason_counts": defaultdict(int),
        "rescued": [],
        "skipped_birth_details": [],
        "rejected_birth_details": [],
    }
    for birth in birth_candidates:
        if len(rescued) >= max_rescues:
            break
        if max_active_bends and _active_bend_component_count(next_labels, atom_graph, label_types) >= int(max_active_bends):
            stats["skipped_active_count_cap"] += 1
            _append_limited(
                stats["skipped_birth_details"],
                {"bend_id": birth.bend_id, "reason": "active_count_cap", "max_active_bends": int(max_active_bends)},
            )
            continue
        if any(label == birth.bend_id for label in next_labels.values()):
            stats["skipped_already_active"] += 1
            _append_limited(
                stats["skipped_birth_details"],
                {"bend_id": birth.bend_id, "reason": "already_active"},
            )
            continue
        flange_support_atoms = {
            atom_id
            for pair in _bend_candidate_attachment_pairs(birth)
            for flange_id in pair
            for atom_id in tuple(flange_lookup[flange_id].atom_ids if flange_id in flange_lookup else ())
        }
        interior_atoms = [atom_id for atom_id in birth.atom_ids if atom_id in atom_lookup and atom_id not in flange_support_atoms]
        candidate_atoms = interior_atoms if len(interior_atoms) >= 2 else [atom_id for atom_id in birth.atom_ids if atom_id in atom_lookup]
        if len(candidate_atoms) < 2:
            stats["skipped_no_candidate_atoms"] += 1
            _append_limited(
                stats["skipped_birth_details"],
                {"bend_id": birth.bend_id, "reason": "no_candidate_atoms", "candidate_atoms": len(candidate_atoms)},
            )
            continue
        for component in _connected_components(candidate_atoms, atom_graph.adjacency):
            if len(rescued) >= max_rescues:
                break
            if len(component) < 2:
                stats["skipped_component_too_small"] += 1
                _append_limited(
                    stats["skipped_birth_details"],
                    {"bend_id": birth.bend_id, "reason": "component_too_small", "component_atoms": len(component)},
                )
                continue
            overlap = len(set(component) & active_bend_atoms) / float(max(1, len(component)))
            if overlap > 0.25:
                original_component_atoms = len(component)
                trimmed_atoms = [atom_id for atom_id in component if atom_id not in active_bend_atoms]
                trimmed_components = _connected_components(trimmed_atoms, atom_graph.adjacency) if allow_no_contact_birth_rescue else []
                trimmed_components = [item for item in trimmed_components if len(item) >= 2]
                if trimmed_components:
                    component = max(trimmed_components, key=len)
                    stats["trimmed_active_overlap_attempts"] += 1
                    _append_limited(
                        stats["skipped_birth_details"],
                        {
                            "bend_id": birth.bend_id,
                            "reason": "active_overlap_trimmed",
                            "original_component_atoms": original_component_atoms,
                            "trimmed_component_atoms": len(component),
                            "overlap_ratio": round(float(overlap), 4),
                        },
                    )
                else:
                    stats["skipped_active_overlap"] += 1
                    _append_limited(
                        stats["skipped_birth_details"],
                        {
                            "bend_id": birth.bend_id,
                            "reason": "active_overlap",
                            "component_atoms": len(component),
                            "overlap_ratio": round(float(overlap), 4),
                        },
                    )
                    continue
            pair_contact = any(
                _component_support_touches_flange_pair(
                    component=component,
                    pair=pair,
                    atom_graph=atom_graph,
                    flange_lookup=flange_lookup,
                )
                >= (1 if birth.visibility_state != "internal" else 2)
                for pair in _bend_candidate_attachment_pairs(birth)
            )
            if not pair_contact and allow_no_contact_birth_rescue:
                geometry_eval = _bend_component_attachment_evaluation(
                    component=component,
                    bend=birth,
                    atom_graph=atom_graph,
                    labels=next_labels,
                    label_types=label_types,
                    atom_lookup=atom_lookup,
                    flange_lookup=flange_lookup,
                    allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                    allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                )
                terms = dict(geometry_eval.get("terms") or {})
                geometry_backed = (
                    not bool(geometry_eval.get("hard_reject", False))
                    and float(geometry_eval.get("quality", 0.0)) >= 0.62
                    and float(terms.get("line_norm", 999.0)) <= 1.35
                    and float(terms.get("span_overlap", 0.0)) >= 0.18
                    and float(terms.get("axis_deg", 999.0)) <= 32.0
                    and float(geometry_eval.get("penalty", 999.0)) <= 4.60
                )
                if geometry_backed:
                    stats["geometry_backed_no_contact_attempts"] += 1
                    pair_contact = True
            if not pair_contact:
                stats["skipped_no_pair_contact"] += 1
                _append_limited(
                    stats["skipped_birth_details"],
                    {"bend_id": birth.bend_id, "reason": "no_pair_contact", "component_atoms": len(component)},
                )
                continue
            trial_labels = dict(next_labels)
            for atom_id in component:
                trial_labels[atom_id] = birth.bend_id
            admissible, reasons = _bend_component_admissibility(
                component=component,
                bend=birth,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
            )
            if not admissible:
                stats["rejected_admissibility"] += 1
                for reason in reasons:
                    stats["admissibility_reason_counts"][reason] += 1
                _append_limited(
                    stats["rejected_birth_details"],
                    {
                        "bend_id": birth.bend_id,
                        "component_atoms": len(component),
                        "reasons": list(reasons),
                    },
                )
                continue
            next_labels = trial_labels
            active_bend_atoms.update(component)
            rescued.append(birth.bend_id)
            stats["rescued"].append(birth.bend_id)
            if max_active_bends and _active_bend_component_count(next_labels, atom_graph, label_types) >= int(max_active_bends):
                break
            break
    if diagnostics is not None:
        diagnostics.update(stats)
        diagnostics["admissibility_reason_counts"] = dict(stats["admissibility_reason_counts"])
    return next_labels, sorted(set(rescued))


def _atom_touches_designated_interface(
    *,
    atom_id: str,
    designated_flanges: Sequence[str],
    labels: Mapping[str, str],
    adjacency: Mapping[str, Sequence[str]],
    label_types: Mapping[str, str],
) -> bool:
    touching = {
        labels.get(neighbor_id)
        for neighbor_id in adjacency.get(atom_id, ())
        if label_types.get(labels.get(neighbor_id)) == "flange"
    }
    return sum(1 for flange_id in designated_flanges if flange_id in touching) >= min(2, len(designated_flanges))


def _component_touches_designated_interface(
    *,
    component: Sequence[str],
    designated_flanges: Sequence[str],
    labels: Mapping[str, str],
    adjacency: Mapping[str, Sequence[str]],
    label_types: Mapping[str, str],
    visibility_state: str,
) -> bool:
    touching = set()
    for atom_id in component:
        for neighbor_id in adjacency.get(atom_id, ()):
            neighbor_label = labels.get(neighbor_id)
            if label_types.get(neighbor_label) == "flange":
                touching.add(str(neighbor_label))
    required_contacts = min(2, len(designated_flanges))
    if visibility_state != "internal":
        required_contacts = min(required_contacts, 1)
    return sum(1 for flange_id in designated_flanges if flange_id in touching) >= required_contacts


def _interface_candidate_atoms_for_pair(
    *,
    pair: Tuple[str, str],
    atom_graph: SurfaceAtomGraph,
    atom_lookup: Mapping[str, SurfaceAtom],
    flange_lookup: Mapping[str, FlangeHypothesis],
) -> List[str]:
    left = flange_lookup.get(pair[0])
    right = flange_lookup.get(pair[1])
    if left is None or right is None:
        return []
    theta = _angle_deg(left.normal, right.normal)
    if theta < 20.0 or theta > 165.0:
        return []

    candidates = set()
    for atom in atom_graph.atoms:
        centroid = np.asarray(atom.centroid, dtype=np.float64)
        pair_line = _plane_pair_line(left, right, centroid)
        if pair_line is None:
            continue
        line_anchor, line_axis = pair_line
        h = max(float(atom.local_spacing_mm), float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
        line_dist = _distance_to_line(centroid, line_anchor, line_axis)
        near_line = line_dist <= max(12.0, 6.0 * h)

        atom_normal = _normalize_direction(atom.normal)
        lhs = _angle_deg(atom_normal, left.normal)
        rhs = _angle_deg(atom_normal, right.normal)
        normal_arc = abs((lhs + rhs) - theta) / 24.0
        bend_like = float(atom.curvature_proxy) * 20.0 >= 0.2
        touches_pair = _atom_support_touches_flange_pair(
            atom_id=atom.atom_id,
            pair=pair,
            atom_graph=atom_graph,
            flange_lookup=flange_lookup,
        )
        if touches_pair >= 2 or (touches_pair >= 1 and normal_arc <= 1.0 and near_line) or (normal_arc <= 0.75 and near_line and bend_like):
            candidates.add(atom.atom_id)
    return sorted(atom_id for atom_id in candidates if atom_id in atom_lookup)


def _atom_support_touches_flange_pair(
    *,
    atom_id: str,
    pair: Tuple[str, str],
    atom_graph: SurfaceAtomGraph,
    flange_lookup: Mapping[str, FlangeHypothesis],
) -> int:
    touching = set()
    for flange_id in pair:
        flange = flange_lookup.get(flange_id)
        if flange is None:
            continue
        support = set(flange.atom_ids)
        if atom_id in support or any(neighbor_id in support for neighbor_id in atom_graph.adjacency.get(atom_id, ())):
            touching.add(flange_id)
    return len(touching)


def _component_support_touches_flange_pair(
    *,
    component: Sequence[str],
    pair: Tuple[str, str],
    atom_graph: SurfaceAtomGraph,
    flange_lookup: Mapping[str, FlangeHypothesis],
) -> int:
    touching = set()
    component_set = set(component)
    for flange_id in pair:
        flange = flange_lookup.get(flange_id)
        if flange is None:
            continue
        support = set(flange.atom_ids)
        if component_set & support:
            touching.add(flange_id)
            continue
        if any(neighbor_id in support for atom_id in component for neighbor_id in atom_graph.adjacency.get(atom_id, ())):
            touching.add(flange_id)
    return len(touching)


def _promote_atom_to_bend(
    *,
    atom: SurfaceAtom,
    bend: BendHypothesis,
    labels: Mapping[str, str],
    atom_graph: SurfaceAtomGraph,
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    margin: float = 0.3,
) -> bool:
    current_label = labels.get(atom.atom_id, "residual")
    if current_label == bend.bend_id:
        return False
    current_label_type = label_types.get(current_label)
    current_cost = _label_cost(
        atom=atom,
        label=current_label,
        labels=labels,
        atom_graph=atom_graph,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
    )
    bend_cost = _label_cost(
        atom=atom,
        label=bend.bend_id,
        labels=labels,
        atom_graph=atom_graph,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
    )
    effective_margin = margin
    if current_label_type == "bend":
        incumbent = bend_lookup.get(str(current_label))
        if incumbent is None:
            return False
        if incumbent.canonical_bend_key == bend.canonical_bend_key:
            return False
        effective_margin += 0.4
    return bend_cost + effective_margin < current_cost


def _candidate_labels_for_atom(
    *,
    atom: SurfaceAtom,
    current_label: Optional[str],
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
) -> Tuple[str, ...]:
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    labels: List[str] = ["residual", "outlier"]
    if current_label:
        labels.append(current_label)

    supported_flanges = [item.flange_id for item in flange_hypotheses if atom.atom_id in item.atom_ids]
    supported_bends = [item.bend_id for item in bend_hypotheses if atom.atom_id in item.atom_ids]
    labels.extend(supported_flanges)
    labels.extend(supported_bends)

    if not supported_flanges and flange_hypotheses:
        labels.append(min(flange_hypotheses, key=lambda item: _flange_data_cost(atom, item)).flange_id)
    if not supported_bends and bend_hypotheses:
        labels.append(min(bend_hypotheses, key=lambda item: _bend_data_cost(atom, item, flange_lookup)).bend_id)

    return tuple(dict.fromkeys(labels))


def _typed_adjacency_penalty(
    label: str,
    neighbor_label: str,
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    lhs_type = label_types.get(label)
    rhs_type = label_types.get(neighbor_label)
    if lhs_type == rhs_type:
        return 0.0
    if lhs_type == "bend" and rhs_type == "flange":
        incident = {flange_id for pair in _bend_candidate_attachment_pairs(bend_lookup[label]) for flange_id in pair}
        return 0.0 if neighbor_label in incident else 0.35
    if lhs_type == "flange" and rhs_type == "bend":
        incident = {flange_id for pair in _bend_candidate_attachment_pairs(bend_lookup[neighbor_label]) for flange_id in pair}
        return 0.0 if label in incident else 0.35
    if lhs_type == "flange" and rhs_type == "flange":
        return 0.2
    if lhs_type == "bend" and rhs_type == "bend":
        return 0.15
    return 0.05


def _prune_and_refit_labels(
    *,
    labels: Dict[str, str],
    atom_graph: SurfaceAtomGraph,
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
) -> Dict[str, str]:
    next_labels = dict(labels)
    counts = defaultdict(int)
    for label in next_labels.values():
        counts[label] += 1
    for atom_id, label in list(next_labels.items()):
        if label.startswith("F") and counts[label] < 2:
            next_labels[atom_id] = "residual"
        elif label.startswith("B") and counts[label] < 2:
            next_labels[atom_id] = "residual"
    for bend in bend_hypotheses:
        bend_atoms = [atom_id for atom_id, label in next_labels.items() if label == bend.bend_id]
        if not bend_atoms:
            continue
        min_component_size = 2 if bend.bend_class != "transition_only" else 3
        for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=next_labels):
            if len(component) < min_component_size:
                for atom_id in component:
                    next_labels[atom_id] = "residual"
    return next_labels


def _total_energy(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    energy = 0.0
    active_labels = set(labels.values())
    for atom in atom_graph.atoms:
        energy += _label_cost(
            atom=atom,
            label=labels[atom.atom_id],
            labels=labels,
            atom_graph=atom_graph,
            flange_lookup=flange_lookup,
            bend_lookup=bend_lookup,
            label_types=label_types,
        )
    for label in active_labels:
        if label in flange_lookup:
            energy += 0.25
        elif label in bend_lookup:
            bend = bend_lookup[label]
            energy += 0.45 if bend.bend_class == "transition_only" else 0.6 if bend.bend_class == "circular_strip" else 0.75
    energy += _incidence_penalty(atom_graph, labels, bend_lookup, label_types, flange_lookup=flange_lookup)
    energy += _fragmentation_penalty(atom_graph, labels, bend_lookup, label_types)
    energy += _duplicate_conflict_penalty(labels, bend_lookup, label_types)
    energy += _support_mass_penalty(atom_graph, labels, bend_lookup, label_types)
    return float(energy)


def _incidence_penalty(
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
) -> float:
    penalty = 0.0
    atom_lookup = {atom.atom_id: atom for atom in getattr(atom_graph, "atoms", ())}
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=labels):
        bend_label = labels[component[0]]
        bend = bend_lookup[bend_label]
        evaluation = _bend_component_attachment_evaluation(
            component=component,
            bend=bend,
            atom_graph=atom_graph,
            labels=labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup or {},
        )
        penalty += float(evaluation["penalty"])
    return float(penalty)


def _fragmentation_penalty(
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    penalty = 0.0
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    components_by_label: Dict[str, List[List[str]]] = defaultdict(list)
    for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=labels):
        components_by_label[labels[component[0]]].append(component)
    for bend_label, components in components_by_label.items():
        if len(components) <= 1:
            continue
        bend = bend_lookup[bend_label]
        penalty += (len(components) - 1) * (0.65 if bend.bend_class == "transition_only" else 0.5)
    return float(penalty)


def _duplicate_conflict_penalty(
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    penalty = 0.0
    active_bends = sorted({label for label in labels.values() if label_types.get(label) == "bend"})
    grouped: Dict[str, List[str]] = defaultdict(list)
    for bend_id in active_bends:
        grouped[bend_lookup[bend_id].canonical_bend_key].append(bend_id)
    for bend_ids in grouped.values():
        if len(bend_ids) > 1:
            penalty += (len(bend_ids) - 1) * 0.8
    return float(penalty)


def _support_mass_penalty(
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> float:
    penalty = 0.0
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=labels):
        bend = bend_lookup[labels[component[0]]]
        support_mass = float(sum(atom_lookup[atom_id].weight for atom_id in component))
        min_support = _minimum_bend_support_mass(atom_graph, bend)
        if support_mass < min_support:
            penalty += (min_support - support_mass) * 0.35
    return float(penalty)


def _build_local_replacement_diagnostics(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    flange_hypotheses: Sequence[FlangeHypothesis],
    bend_hypotheses: Sequence[BendHypothesis],
    label_types: Mapping[str, str],
    max_records: int = 16,
) -> Dict[str, Any]:
    flange_lookup = {item.flange_id: item for item in flange_hypotheses}
    bend_lookup = {item.bend_id: item for item in bend_hypotheses}
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    current_energy = _total_energy(
        atom_graph=atom_graph,
        labels=labels,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
    )
    active_bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    active_components = _connected_components(active_bend_atoms, atom_graph.adjacency, labels=labels)
    active_labels = {str(label) for label in labels.values() if label_types.get(label) == "bend"}
    inactive_bends = [bend for bend in bend_hypotheses if bend.bend_id not in active_labels]

    replacement_records: List[Dict[str, Any]] = []
    add_records: List[Dict[str, Any]] = []
    replacement_moves: List[Dict[str, Any]] = []
    add_moves: List[Dict[str, Any]] = []

    for component in active_components:
        active_label = str(labels[component[0]])
        active_bend = bend_lookup.get(active_label)
        if active_bend is None:
            continue
        component_set = set(component)
        active_eval = _bend_component_attachment_evaluation(
            component=component,
            bend=active_bend,
            atom_graph=atom_graph,
            labels=labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup,
        )
        for candidate in inactive_bends:
            candidate_atoms = set(candidate.atom_ids)
            overlap_atoms = component_set & candidate_atoms
            if not overlap_atoms:
                continue
            overlap_min = len(overlap_atoms) / float(max(1, min(len(component_set), len(candidate_atoms))))
            overlap_candidate = len(overlap_atoms) / float(max(1, len(candidate_atoms)))
            if overlap_min < 0.15 and overlap_candidate < 0.10:
                continue
            trial_labels = dict(labels)
            for atom_id in component:
                trial_labels[atom_id] = candidate.bend_id
            admissible, reasons = _bend_component_admissibility(
                component=component,
                bend=candidate,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
            )
            candidate_eval = _bend_component_attachment_evaluation(
                component=component,
                bend=candidate,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
            )
            trial_energy = _total_energy(
                atom_graph=atom_graph,
                labels=trial_labels,
                flange_lookup=flange_lookup,
                bend_lookup=bend_lookup,
                label_types=label_types,
            )
            record = {
                "move": "replace_active_component",
                "active_bend_id": active_label,
                "active_source_kind": active_bend.source_kind,
                "active_selected_pair": list(active_eval.get("selected_pair") or ()),
                "candidate_bend_id": candidate.bend_id,
                "candidate_source_kind": candidate.source_kind,
                "candidate_selected_pair": list(candidate_eval.get("selected_pair") or ()),
                "component_atom_count": len(component),
                "overlap_atom_count": len(overlap_atoms),
                "overlap_min_ratio": round(float(overlap_min), 4),
                "overlap_candidate_ratio": round(float(overlap_candidate), 4),
                "candidate_admissible": bool(admissible),
                "candidate_reasons": list(reasons),
                "energy_delta": round(float(trial_energy - current_energy), 6),
                "candidate_attachment_penalty": round(float(candidate_eval.get("penalty", 999.0)), 6),
                "candidate_attachment_quality": round(float(candidate_eval.get("quality", 0.0)), 6),
            }
            replacement_records.append(record)
            replacement_moves.append({**record, "_component": tuple(component), "_candidate_bend_id": candidate.bend_id})

    non_bend_labels = {
        atom_id
        for atom_id, label in labels.items()
        if label_types.get(label) in {"flange", "residual", "outlier"}
    }
    for candidate in inactive_bends:
        available_atoms = [atom_id for atom_id in candidate.atom_ids if atom_id in non_bend_labels]
        for component in _connected_components(available_atoms, atom_graph.adjacency):
            if len(component) < 2:
                continue
            trial_labels = dict(labels)
            for atom_id in component:
                trial_labels[atom_id] = candidate.bend_id
            admissible, reasons = _bend_component_admissibility(
                component=component,
                bend=candidate,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
            )
            candidate_eval = _bend_component_attachment_evaluation(
                component=component,
                bend=candidate,
                atom_graph=atom_graph,
                labels=trial_labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup,
            )
            trial_energy = _total_energy(
                atom_graph=atom_graph,
                labels=trial_labels,
                flange_lookup=flange_lookup,
                bend_lookup=bend_lookup,
                label_types=label_types,
            )
            record = {
                "move": "add_inactive_candidate_support",
                "candidate_bend_id": candidate.bend_id,
                "candidate_source_kind": candidate.source_kind,
                "candidate_selected_pair": list(candidate_eval.get("selected_pair") or ()),
                "component_atom_count": len(component),
                "candidate_support_atom_count": len(candidate.atom_ids),
                "candidate_admissible": bool(admissible),
                "candidate_reasons": list(reasons),
                "energy_delta": round(float(trial_energy - current_energy), 6),
                "candidate_attachment_penalty": round(float(candidate_eval.get("penalty", 999.0)), 6),
                "candidate_attachment_quality": round(float(candidate_eval.get("quality", 0.0)), 6),
            }
            add_records.append(record)
            add_moves.append({**record, "_component": tuple(component), "_candidate_bend_id": candidate.bend_id})

    replacement_records.sort(key=lambda item: (not bool(item["candidate_admissible"]), float(item["energy_delta"]), -float(item["overlap_min_ratio"])))
    add_records.sort(key=lambda item: (not bool(item["candidate_admissible"]), float(item["energy_delta"]), -int(item["component_atom_count"])))
    multi_object_candidates = _score_pairwise_local_replacement_moves(
        atom_graph=atom_graph,
        labels=labels,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
        current_energy=current_energy,
        replacement_moves=replacement_moves,
        add_moves=add_moves,
        max_records=max_records,
    )
    relabel_candidates = _score_local_relabel_neighborhoods(
        atom_graph=atom_graph,
        labels=labels,
        flange_lookup=flange_lookup,
        bend_lookup=bend_lookup,
        label_types=label_types,
        current_energy=current_energy,
        replacement_moves=replacement_moves,
        add_moves=add_moves,
        max_records=max_records,
    )
    return {
        "current_energy": round(float(current_energy), 6),
        "active_bend_component_count": len(active_components),
        "active_bend_label_count": len(active_labels),
        "inactive_bend_label_count": len(inactive_bends),
        "replacement_candidates": replacement_records[:max_records],
        "add_candidates": add_records[:max_records],
        "multi_object_candidates": multi_object_candidates,
        "local_relabel_candidates": relabel_candidates,
    }


def _score_local_relabel_neighborhoods(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    current_energy: float,
    replacement_moves: Sequence[Mapping[str, Any]],
    add_moves: Sequence[Mapping[str, Any]],
    max_records: int,
) -> List[Dict[str, Any]]:
    current_count = _bend_component_count(atom_graph, labels, label_types)
    seed_moves = sorted(
        [*replacement_moves, *add_moves],
        key=lambda item: (not bool(item.get("candidate_admissible")), float(item.get("energy_delta", 999.0))),
    )[:10]
    neighborhoods: List[Dict[str, Any]] = []
    seen: set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    for seed in seed_moves:
        seed_atoms = set(seed.get("_component") or ())
        seed_candidate = str(seed.get("_candidate_bend_id") or seed.get("candidate_bend_id") or "")
        if not seed_atoms or not seed_candidate:
            continue
        cluster_atoms = set(seed_atoms)
        candidate_labels = {seed_candidate}
        active_label = str(seed.get("active_bend_id") or "")
        if active_label:
            candidate_labels.add(active_label)
        for other in seed_moves:
            other_atoms = set(other.get("_component") or ())
            if not other_atoms:
                continue
            touches = bool(cluster_atoms & other_atoms) or any(
                neighbor_id in other_atoms
                for atom_id in cluster_atoms
                for neighbor_id in atom_graph.adjacency.get(atom_id, ())
            )
            if not touches:
                continue
            other_candidate = str(other.get("_candidate_bend_id") or other.get("candidate_bend_id") or "")
            other_active = str(other.get("active_bend_id") or "")
            if other_candidate:
                candidate_labels.add(other_candidate)
            if other_active:
                candidate_labels.add(other_active)
            cluster_atoms.update(other_atoms)
            if len(candidate_labels) >= 6:
                break
        if len(cluster_atoms) > 220 or len(candidate_labels) < 2:
            continue
        key = (tuple(sorted(cluster_atoms)), tuple(sorted(candidate_labels)))
        if key in seen:
            continue
        seen.add(key)
        trial = _solve_frozen_local_relabel(
            atom_graph=atom_graph,
            labels=labels,
            cluster_atoms=cluster_atoms,
            candidate_labels=candidate_labels,
            flange_lookup=flange_lookup,
            bend_lookup=bend_lookup,
            label_types=label_types,
        )
        changed_atoms = [atom_id for atom_id in cluster_atoms if trial.get(atom_id) != labels.get(atom_id)]
        if not changed_atoms:
            continue
        trial_energy = _total_energy(
            atom_graph=atom_graph,
            labels=trial,
            flange_lookup=flange_lookup,
            bend_lookup=bend_lookup,
            label_types=label_types,
        )
        trial_count = _bend_component_count(atom_graph, trial, label_types)
        bend_labels_before = {str(labels.get(atom_id)) for atom_id in cluster_atoms if label_types.get(labels.get(atom_id)) == "bend"}
        bend_labels_after = {str(trial.get(atom_id)) for atom_id in cluster_atoms if label_types.get(trial.get(atom_id)) == "bend"}
        neighborhoods.append(
            {
                "move": "frozen_local_relabel",
                "energy_delta": round(float(trial_energy - current_energy), 6),
                "count_before": int(current_count),
                "count_after": int(trial_count),
                "count_delta": int(trial_count) - int(current_count),
                "cluster_atom_count": len(cluster_atoms),
                "changed_atom_count": len(changed_atoms),
                "candidate_labels": sorted(candidate_labels),
                "bend_labels_before": sorted(bend_labels_before),
                "bend_labels_after": sorted(bend_labels_after),
                "seed_move": _public_local_move_summary(seed),
            }
        )
    neighborhoods.sort(key=lambda item: float(item["energy_delta"]))
    return neighborhoods[:max_records]


def _solve_frozen_local_relabel(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    cluster_atoms: set[str],
    candidate_labels: set[str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
) -> Dict[str, str]:
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    allowed_by_atom: Dict[str, Tuple[str, ...]] = {}
    for atom_id in cluster_atoms:
        current_label = str(labels.get(atom_id) or "residual")
        allowed = {"residual", current_label}
        for label in candidate_labels:
            if label in bend_lookup and atom_id in bend_lookup[label].atom_ids:
                allowed.add(label)
            elif label == current_label:
                allowed.add(label)
        allowed_by_atom[atom_id] = tuple(sorted(label for label in allowed if label in label_types))

    trial = dict(labels)
    active_labels = set(labels.values())
    for _ in range(5):
        changed = False
        for atom_id in sorted(cluster_atoms):
            atom = atom_lookup.get(atom_id)
            if atom is None:
                continue
            best_label = trial.get(atom_id, "residual")
            best_cost: Optional[float] = None
            for label in allowed_by_atom.get(atom_id, ("residual",)):
                cost = _label_cost(
                    atom=atom,
                    label=label,
                    labels=trial,
                    atom_graph=atom_graph,
                    flange_lookup=flange_lookup,
                    bend_lookup=bend_lookup,
                    label_types=label_types,
                )
                if label not in active_labels:
                    cost += 0.45 if label in bend_lookup else 0.25 if label in flange_lookup else 0.0
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_label = label
            if best_label != trial.get(atom_id):
                trial[atom_id] = best_label
                changed = True
        if not changed:
            break
    return dict(sorted(trial.items()))


def _score_pairwise_local_replacement_moves(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    current_energy: float,
    replacement_moves: Sequence[Mapping[str, Any]],
    add_moves: Sequence[Mapping[str, Any]],
    max_records: int,
) -> List[Dict[str, Any]]:
    plausible_replacements = sorted(
        (
            move
            for move in replacement_moves
            if bool(move.get("candidate_admissible")) or float(move.get("energy_delta", 999.0)) <= 20.0
        ),
        key=lambda item: (not bool(item.get("candidate_admissible")), float(item.get("energy_delta", 999.0))),
    )[:8]
    plausible_adds = sorted(
        (
            move
            for move in add_moves
            if bool(move.get("candidate_admissible")) or float(move.get("energy_delta", 999.0)) <= 25.0
        ),
        key=lambda item: (not bool(item.get("candidate_admissible")), float(item.get("energy_delta", 999.0))),
    )[:8]
    moves: List[Mapping[str, Any]] = [*plausible_replacements, *plausible_adds]
    records: List[Dict[str, Any]] = []
    for left_index, left in enumerate(moves):
        for right in moves[left_index + 1 :]:
            left_atoms = set(left.get("_component") or ())
            right_atoms = set(right.get("_component") or ())
            if not left_atoms or not right_atoms:
                continue
            left_candidate = str(left.get("_candidate_bend_id") or left.get("candidate_bend_id") or "")
            right_candidate = str(right.get("_candidate_bend_id") or right.get("candidate_bend_id") or "")
            if not left_candidate or not right_candidate:
                continue
            if left_candidate == right_candidate:
                continue
            if left_atoms & right_atoms:
                continue
            left_active = str(left.get("active_bend_id") or "")
            right_active = str(right.get("active_bend_id") or "")
            if left_active and right_active and left_active == right_active:
                continue

            trial_labels = dict(labels)
            for atom_id in left_atoms:
                trial_labels[atom_id] = left_candidate
            for atom_id in right_atoms:
                trial_labels[atom_id] = right_candidate
            trial_energy = _total_energy(
                atom_graph=atom_graph,
                labels=trial_labels,
                flange_lookup=flange_lookup,
                bend_lookup=bend_lookup,
                label_types=label_types,
            )
            records.append(
                {
                    "move": "pairwise_local_neighborhood",
                    "moves": [
                        _public_local_move_summary(left),
                        _public_local_move_summary(right),
                    ],
                    "energy_delta": round(float(trial_energy - current_energy), 6),
                    "individual_energy_delta_sum": round(float(left.get("energy_delta", 0.0)) + float(right.get("energy_delta", 0.0)), 6),
                    "component_atom_count": int(len(left_atoms) + len(right_atoms)),
                    "candidate_bend_ids": [left_candidate, right_candidate],
                    "active_bend_ids": [value for value in (left_active, right_active) if value],
                }
            )
    records.sort(key=lambda item: float(item["energy_delta"]))
    return records[:max_records]


def _public_local_move_summary(move: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "move": str(move.get("move") or ""),
        "active_bend_id": move.get("active_bend_id"),
        "candidate_bend_id": move.get("candidate_bend_id"),
        "candidate_source_kind": move.get("candidate_source_kind"),
        "candidate_admissible": bool(move.get("candidate_admissible")),
        "energy_delta": float(move.get("energy_delta", 0.0)),
        "component_atom_count": int(move.get("component_atom_count", 0)),
        "candidate_selected_pair": list(move.get("candidate_selected_pair") or ()),
    }


def _minimum_bend_support_mass(
    atom_graph: SurfaceAtomGraph,
    bend: BendHypothesis,
) -> float:
    voxel_size_mm = max(float(getattr(atom_graph, "voxel_size_mm", 1.0)), 1e-6)
    return float(max(1.0, min(3.0, bend.cross_width_mm / voxel_size_mm)))


def _bend_candidate_attachment_pairs(bend: BendHypothesis) -> Tuple[Tuple[str, str], ...]:
    return _unique_flange_pairs(
        [
            *bend.candidate_incident_flange_pairs,
            bend.seed_incident_flange_ids,
            bend.canonical_flange_pair,
            bend.incident_flange_ids,
        ]
    )


def _touching_flange_weights(
    *,
    component: Sequence[str],
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    include_hypothesis_support: bool = False,
) -> Dict[str, float]:
    touching: Dict[str, float] = defaultdict(float)
    component_set = set(component)
    for atom_id in component:
        for neighbor_id in atom_graph.adjacency.get(atom_id, ()):
            neighbor_label = labels.get(neighbor_id)
            if label_types.get(neighbor_label) == "flange":
                touching[str(neighbor_label)] += float(
                    getattr(atom_graph, "edge_weights", {}).get(_edge_key(atom_id, neighbor_id), 1.0)
                )
    if include_hypothesis_support and flange_lookup:
        for flange_id, flange in flange_lookup.items():
            if touching.get(str(flange_id), 0.0) > 0.0:
                continue
            support = set(flange.atom_ids)
            if not support:
                continue
            support_contact = 0.0
            if component_set & support:
                support_contact += 0.5
            for atom_id in component:
                for neighbor_id in atom_graph.adjacency.get(atom_id, ()):
                    if neighbor_id in support:
                        support_contact += float(getattr(atom_graph, "edge_weights", {}).get(_edge_key(atom_id, neighbor_id), 1.0))
            if support_contact > 0.0:
                touching[str(flange_id)] += support_contact * 0.65
    return dict(touching)


def _component_candidate_attachment_pairs(
    *,
    component: Sequence[str],
    bend: BendHypothesis,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    atom_lookup: Mapping[str, SurfaceAtom],
    max_pairs: int = 12,
    allow_hypothesis_support_contacts: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    seed_pair = _canonical_flange_pair(bend.seed_incident_flange_ids or bend.canonical_flange_pair or bend.incident_flange_ids)
    scores: Dict[Tuple[str, str], float] = defaultdict(float)
    for pair in _bend_candidate_attachment_pairs(bend):
        if len(pair) == 2:
            scores[pair] += 3.0
    if len(seed_pair) == 2:
        scores[(seed_pair[0], seed_pair[1])] += 5.0

    touching = _touching_flange_weights(
        component=component,
        atom_graph=atom_graph,
        labels=labels,
        label_types=label_types,
        flange_lookup=flange_lookup,
        include_hypothesis_support=allow_hypothesis_support_contacts,
    )
    touched = sorted(touching, key=lambda flange_id: (-touching[flange_id], flange_id))[:8]
    for index, lhs in enumerate(touched):
        for rhs in touched[index + 1 :]:
            pair = _canonical_flange_pair((lhs, rhs))
            if len(pair) == 2:
                scores[(pair[0], pair[1])] += touching.get(lhs, 0.0) + touching.get(rhs, 0.0)

    known_component = [atom_lookup[atom_id].centroid for atom_id in component if atom_id in atom_lookup]
    if known_component and flange_lookup:
        support_centroid = np.mean(np.asarray(known_component, dtype=np.float64), axis=0)
        h = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
        width = max(float(bend.cross_width_mm), 2.0 * h, 4.0)
        flanges = sorted(flange_lookup.values(), key=lambda item: item.flange_id)
        for left_index, left in enumerate(flanges):
            for right in flanges[left_index + 1 :]:
                pair = _canonical_flange_pair((left.flange_id, right.flange_id))
                if len(pair) != 2:
                    continue
                theta = _angle_deg(left.normal, right.normal)
                if theta < 18.0 or theta > 165.0:
                    continue
                pair_line = _plane_pair_line(left, right, support_centroid)
                if pair_line is None:
                    continue
                pair_anchor, pair_axis = pair_line
                line_dist = _distance_to_line(support_centroid, pair_anchor, pair_axis)
                if line_dist <= max(3.0 * width, 8.0 * h, 25.0):
                    scores[(pair[0], pair[1])] += 1.0 / (1.0 + line_dist / max(width, 1.0))

    ranked = [
        (pair, score)
        for pair, score in scores.items()
        if len(pair) == 2 and (not flange_lookup or (pair[0] in flange_lookup and pair[1] in flange_lookup))
    ]
    ranked.sort(key=lambda item: (-item[1], _attachment_seed_delta(item[0], seed_pair), item[0]))
    return tuple(pair for pair, _ in ranked[:max_pairs])


def _score_attachment_pair(
    *,
    component: Sequence[str],
    bend: BendHypothesis,
    pair: Tuple[str, str],
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
    flange_lookup: Mapping[str, FlangeHypothesis],
    atom_lookup: Mapping[str, SurfaceAtom],
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
) -> Dict[str, Any]:
    touching = _touching_flange_weights(
        component=component,
        atom_graph=atom_graph,
        labels=labels,
        label_types=label_types,
        flange_lookup=flange_lookup,
        include_hypothesis_support=allow_hypothesis_support_contacts,
    )
    total_contact = float(sum(touching.values()))
    seed_pair = _canonical_flange_pair(bend.seed_incident_flange_ids or bend.canonical_flange_pair or bend.incident_flange_ids)
    designated = tuple(pair)
    designated_contacts = [touching.get(flange_id, 0.0) for flange_id in designated]
    present_designated = sum(1 for value in designated_contacts if value > 0.0)
    required_contacts = 2 if bend.visibility_state == "internal" else 1
    missing_contacts = max(0, required_contacts - present_designated)
    extra_contact = float(sum(value for flange_id, value in touching.items() if flange_id not in designated))
    extra_ratio = extra_contact / total_contact if total_contact > 0.0 else 0.0
    imbalance_ratio = 0.0
    if bend.visibility_state == "internal" and len(designated_contacts) == 2:
        lhs, rhs = designated_contacts
        if lhs + rhs > 0.0:
            imbalance_ratio = abs(lhs - rhs) / (lhs + rhs)
    contact_ratio = sum(designated_contacts) / (total_contact + 1e-6)

    reasons: List[str] = []
    if missing_contacts > 0:
        reasons.append("missing_designated_flange_contact")
    if total_contact > 0.0 and extra_ratio > 0.45:
        reasons.append("boundary_leakage")
    if bend.visibility_state == "internal" and imbalance_ratio > 0.88:
        reasons.append("contact_imbalance")

    fallback_penalty = (
        1.20 * missing_contacts
        + 0.90 * extra_ratio
        + 0.55 * max(0.0, imbalance_ratio - 0.70)
        + 1.10 * _attachment_seed_delta(designated, seed_pair)
    )
    fallback = {
        "selected_pair": designated,
        "touching": touching,
        "reasons": reasons,
        "penalty": float(fallback_penalty),
        "quality": float(contact_ratio),
        "hard_reject": False,
        "present_designated": present_designated,
        "required_contacts": required_contacts,
        "terms": {
            "contact_ratio": float(contact_ratio),
            "extra_ratio": float(extra_ratio),
            "imbalance_ratio": float(imbalance_ratio),
            "seed_delta": float(_attachment_seed_delta(designated, seed_pair)),
        },
    }
    if not flange_lookup or designated[0] not in flange_lookup or designated[1] not in flange_lookup:
        return fallback

    known_points = [atom_lookup[atom_id].centroid for atom_id in component if atom_id in atom_lookup]
    if not known_points:
        return {**fallback, "hard_reject": True, "penalty": 999.0, "reasons": reasons + ["empty_support_component"]}

    support_points = np.asarray(known_points, dtype=np.float64)
    support_centroid = np.mean(support_points, axis=0)
    flange_a = flange_lookup[designated[0]]
    flange_b = flange_lookup[designated[1]]
    pair_line = _plane_pair_line(flange_a, flange_b, support_centroid)
    if pair_line is None:
        return {**fallback, "hard_reject": True, "penalty": 999.0, "reasons": reasons + ["parallel_flange_pair"]}

    pair_anchor, pair_axis = pair_line
    bend_axis = _normalize_direction(bend.axis_direction)
    theta_pair = _angle_deg(flange_a.normal, flange_b.normal)
    raw_axis_deg = _angle_deg(bend_axis, pair_axis)
    axis_deg = min(raw_axis_deg, abs(90.0 - raw_axis_deg)) if allow_cross_axis_attachment_match else raw_axis_deg
    angle_delta = abs(float(bend.angle_deg) - theta_pair)
    h = max(float(getattr(atom_graph, "local_spacing_mm", 1.0)), 1.0)
    width = max(float(bend.cross_width_mm), 2.0 * h, 4.0)
    line_distances = [_distance_to_line(point, pair_anchor, pair_axis) for point in support_points]
    median_line_dist = float(np.median(line_distances)) if line_distances else 999.0
    sigma_line = max(1.5 * width, 4.0 * h, 6.0)
    line_norm = median_line_dist / sigma_line
    support_proj = support_points @ pair_axis
    support_interval = (float(np.min(support_proj)), float(np.max(support_proj)))
    anchor = np.asarray(bend.anchor, dtype=np.float64)
    seed_center = float(np.dot(anchor, pair_axis))
    seed_span = max(float(bend.span_mm), width)
    seed_interval = (seed_center - 0.5 * seed_span, seed_center + 0.5 * seed_span)
    span_overlap = _interval_overlap_score(support_interval, seed_interval)
    seed_delta = _attachment_seed_delta(designated, seed_pair)
    geom_quality = float(
        np.exp(-0.5 * ((axis_deg / 18.0) ** 2 + (angle_delta / 22.0) ** 2 + line_norm**2))
        * max(span_overlap, contact_ratio)
    )
    seed_cost = 1.10 * seed_delta * (0.25 + 0.75 * (1.0 - geom_quality))
    penalty = (
        1.20 * missing_contacts
        + 0.90 * extra_ratio
        + 0.55 * max(0.0, imbalance_ratio - 0.70)
        + 0.75 * _rho(axis_deg / 18.0)
        + 0.35 * _rho(angle_delta / 22.0)
        + 0.90 * _rho(line_norm)
        + 0.50 * max(0.0, 0.25 - span_overlap) / 0.25
        + seed_cost
    )
    hard_reject = False
    if theta_pair < 18.0 or theta_pair > 165.0:
        hard_reject = True
        reasons.append("invalid_pair_angle")
    if axis_deg > 40.0 and seed_delta >= 0.8:
        hard_reject = True
        reasons.append("axis_mismatch")
    if line_norm > 3.0 and seed_delta >= 0.8:
        hard_reject = True
        reasons.append("interface_corridor_far_from_support")
    if span_overlap < 0.10 and seed_delta >= 0.8 and geom_quality < 0.35:
        hard_reject = True
        reasons.append("span_mismatch")
    if seed_delta >= 0.8 and geom_quality < 0.55:
        hard_reject = True
        reasons.append("unsupported_full_reattachment")
    return {
        "selected_pair": designated,
        "touching": touching,
        "reasons": reasons,
        "penalty": float(penalty),
        "quality": float(geom_quality),
        "hard_reject": bool(hard_reject),
        "present_designated": present_designated,
        "required_contacts": required_contacts,
        "terms": {
            "axis_deg": float(axis_deg),
            "raw_axis_deg": float(raw_axis_deg),
            "angle_deg": float(angle_delta),
            "line_norm": float(line_norm),
            "span_overlap": float(span_overlap),
            "contact_ratio": float(contact_ratio),
            "extra_ratio": float(extra_ratio),
            "imbalance_ratio": float(imbalance_ratio),
            "seed_delta": float(seed_delta),
            "geom_quality": float(geom_quality),
        },
    }


def _bend_component_attachment_evaluation(
    *,
    component: Sequence[str],
    bend: BendHypothesis,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
    atom_lookup: Optional[Mapping[str, SurfaceAtom]] = None,
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
) -> Dict[str, Any]:
    touching = _touching_flange_weights(
        component=component,
        atom_graph=atom_graph,
        labels=labels,
        label_types=label_types,
        flange_lookup=flange_lookup or {},
        include_hypothesis_support=allow_hypothesis_support_contacts,
    )
    atom_lookup = atom_lookup or {atom.atom_id: atom for atom in getattr(atom_graph, "atoms", ())}
    flange_lookup = flange_lookup or {}
    candidate_pairs = _component_candidate_attachment_pairs(
        component=component,
        bend=bend,
        atom_graph=atom_graph,
        labels=labels,
        label_types=label_types,
        flange_lookup=flange_lookup,
        atom_lookup=atom_lookup,
        allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
    )
    if not candidate_pairs and len(touching) >= 2:
        ranked = sorted(touching, key=lambda flange_id: (-touching[flange_id], flange_id))[:2]
        candidate_pairs = _unique_flange_pairs([ranked])
    best: Optional[Dict[str, Any]] = None
    rejected: List[Dict[str, Any]] = []
    for pair in candidate_pairs:
        candidate = _score_attachment_pair(
            component=component,
            bend=bend,
            pair=pair,
            atom_graph=atom_graph,
            labels=labels,
            label_types=label_types,
            flange_lookup=flange_lookup,
            atom_lookup=atom_lookup,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
        )
        if candidate.get("hard_reject"):
            rejected.append(candidate)
            continue
        if best is None or (candidate["penalty"], -float(candidate.get("quality", 0.0)), candidate["selected_pair"]) < (
            best["penalty"],
            -float(best.get("quality", 0.0)),
            best["selected_pair"],
        ):
            best = candidate

    if best is not None:
        best["rejected_pairs"] = len(rejected)
        return best
    if rejected:
        rejected.sort(key=lambda item: float(item.get("penalty", 999.0)))
        fallback = dict(rejected[0])
        fallback["reasons"] = list(fallback.get("reasons", ())) + ["all_candidate_pairs_rejected"]
        fallback["rejected_pairs"] = len(rejected)
        return fallback
    if best is None:
        return {
            "selected_pair": tuple(),
            "touching": touching,
            "reasons": ["weak_incident_flange_pair"],
            "penalty": 999.0,
            "quality": 0.0,
            "hard_reject": True,
            "present_designated": 0,
            "required_contacts": 2 if bend.visibility_state == "internal" else 1,
            "terms": {},
        }


def _bend_component_admissibility(
    *,
    component: Sequence[str],
    bend: BendHypothesis,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
    atom_lookup: Mapping[str, SurfaceAtom],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_mixed_transition_single_contact: bool = False,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    support_mass = float(sum(atom_lookup[atom_id].weight for atom_id in component))
    if support_mass < _minimum_bend_support_mass(atom_graph, bend):
        reasons.append("support_mass_too_small")

    evaluation = _bend_component_attachment_evaluation(
        component=component,
        bend=bend,
        atom_graph=atom_graph,
        labels=labels,
        label_types=label_types,
        atom_lookup=atom_lookup,
        flange_lookup=flange_lookup or {},
        allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
        allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
    )
    reasons.extend(evaluation["reasons"])
    severe = {
        "invalid_pair_angle",
        "axis_mismatch",
        "interface_corridor_far_from_support",
        "unsupported_full_reattachment",
        "all_candidate_pairs_rejected",
        "weak_incident_flange_pair",
        "parallel_flange_pair",
        "empty_support_component",
    }
    if any(reason in severe for reason in reasons):
        return False, sorted(set(reasons))

    present_designated = int(evaluation["present_designated"])
    required_contacts = int(evaluation["required_contacts"])
    quality = float(evaluation.get("quality", 0.0))
    geometry_backed_single_contact = (
        bool(allow_geometry_backed_single_contact)
        and required_contacts >= 2
        and present_designated >= 1
        and support_mass >= _minimum_bend_support_mass(atom_graph, bend)
        and quality >= 0.58
        and float(evaluation.get("penalty", 999.0)) <= 3.60
    )
    geometry_backed_imbalanced_contact = (
        bool(allow_geometry_backed_single_contact)
        and present_designated >= required_contacts
        and support_mass >= _minimum_bend_support_mass(atom_graph, bend)
        and quality >= 0.58
        and float(evaluation.get("penalty", 999.0)) <= 3.60
    )
    terms = dict(evaluation.get("terms") or {})
    geometry_backed_birth_contact = (
        bool(allow_geometry_backed_single_contact)
        and str(bend.source_kind) == "interface_birth"
        and support_mass >= _minimum_bend_support_mass(atom_graph, bend)
        and quality >= 0.62
        and float(terms.get("line_norm", 999.0)) <= 1.35
        and float(terms.get("span_overlap", 0.0)) >= 0.18
        and float(terms.get("axis_deg", 999.0)) <= 32.0
        and float(evaluation.get("penalty", 999.0)) <= 4.60
    )
    mixed_transition_single_contact = (
        bool(allow_mixed_transition_single_contact)
        and required_contacts >= 2
        and present_designated >= 1
        and support_mass >= _minimum_bend_support_mass(atom_graph, bend)
        and float(terms.get("contact_ratio", 0.0)) >= 0.90
        and float(terms.get("extra_ratio", 1.0)) <= 0.10
        and float(terms.get("line_norm", 999.0)) <= 0.80
        and float(terms.get("span_overlap", 0.0)) >= 0.75
        and float(terms.get("axis_deg", 999.0)) <= 25.0
        and float(evaluation.get("penalty", 999.0)) <= 3.20
    )
    if present_designated < required_contacts:
        if not (geometry_backed_single_contact or geometry_backed_birth_contact or mixed_transition_single_contact):
            return False, sorted(set(reasons))
        reasons = [reason for reason in reasons if reason not in {"missing_designated_flange_contact", "contact_imbalance"}]

    if "boundary_leakage" in reasons and quality >= 0.75:
        reasons.remove("boundary_leakage")
    if "contact_imbalance" in reasons and quality >= 0.75:
        reasons.remove("contact_imbalance")
    if (geometry_backed_single_contact or geometry_backed_imbalanced_contact or geometry_backed_birth_contact or mixed_transition_single_contact) and "contact_imbalance" in reasons:
        reasons.remove("contact_imbalance")
    penalty_limit = 4.60 if geometry_backed_birth_contact else 4.00
    if float(evaluation.get("penalty", 999.0)) > penalty_limit:
        reasons.append("attachment_penalty_too_high")
    return (len(reasons) == 0, sorted(set(reasons)))


def _count_admissible_bend_components(
    *,
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    bend_lookup: Mapping[str, BendHypothesis],
    label_types: Mapping[str, str],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_mixed_transition_single_contact: bool = False,
) -> int:
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    count = 0
    for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=labels):
        bend_id = labels[component[0]]
        admissible, _ = _bend_component_admissibility(
            component=component,
            bend=bend_lookup[bend_id],
            atom_graph=atom_graph,
            labels=labels,
            label_types=label_types,
            atom_lookup=atom_lookup,
            flange_lookup=flange_lookup or {},
            allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
            allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
            allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
            allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
        )
        count += int(admissible)
    return count


def _build_object_level_confidence_summary(
    *,
    atom_graph: SurfaceAtomGraph,
    label_maps: Sequence[Mapping[str, str]],
    bend_lookup: Mapping[str, BendHypothesis],
    solution_energies: Sequence[float],
    label_types: Mapping[str, str],
    flange_lookup: Optional[Mapping[str, FlangeHypothesis]] = None,
    allow_geometry_backed_single_contact: bool = False,
    allow_hypothesis_support_contacts: bool = False,
    allow_cross_axis_attachment_match: bool = False,
    allow_mixed_transition_single_contact: bool = False,
) -> Dict[str, float]:
    if not label_maps or not solution_energies:
        return {}
    best = min(float(value) for value in solution_energies)
    scale = max(1.0, float(np.std(solution_energies)) + 1.0)
    raw_weights = [float(np.exp(-(float(energy) - best) / scale)) for energy in solution_energies]
    norm = sum(raw_weights) or 1.0
    weights = [value / norm for value in raw_weights]
    atom_lookup = {atom.atom_id: atom for atom in atom_graph.atoms}
    summary: Dict[str, float] = defaultdict(float)
    for labels, weight in zip(label_maps, weights):
        bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
        present_keys = set()
        for component in _connected_components(bend_atoms, atom_graph.adjacency, labels=labels):
            bend = bend_lookup[labels[component[0]]]
            admissible, _ = _bend_component_admissibility(
                component=component,
                bend=bend,
                atom_graph=atom_graph,
                labels=labels,
                label_types=label_types,
                atom_lookup=atom_lookup,
                flange_lookup=flange_lookup or {},
                allow_geometry_backed_single_contact=allow_geometry_backed_single_contact,
                allow_hypothesis_support_contacts=allow_hypothesis_support_contacts,
                allow_cross_axis_attachment_match=allow_cross_axis_attachment_match,
                allow_mixed_transition_single_contact=allow_mixed_transition_single_contact,
            )
            if admissible:
                present_keys.add(bend.canonical_bend_key)
        for key in present_keys:
            summary[key] += weight
    return {key: float(value) for key, value in sorted(summary.items())}


def _post_bend_class_for_region(
    bend: BendHypothesis,
    component: Sequence[str],
    atom_graph: SurfaceAtomGraph,
    atom_lookup: Mapping[str, SurfaceAtom],
) -> str:
    candidate_classes = set(bend.candidate_classes or ()) | {bend.bend_class}
    if "developable_strip" in candidate_classes and len(component) >= 5 and bend.span_mm >= max(bend.cross_width_mm * 2.5, atom_graph.local_spacing_mm * 6.0):
        return "developable_strip"
    if "circular_strip" in candidate_classes and bend.radius_mm not in {None, 0.0} and len(component) >= 2:
        return "circular_strip"
    return "transition_only"


def _bend_component_count(
    atom_graph: SurfaceAtomGraph,
    labels: Mapping[str, str],
    label_types: Mapping[str, str],
) -> int:
    bend_atoms = [atom_id for atom_id, label in labels.items() if label_types.get(label) == "bend"]
    return len(_connected_components(bend_atoms, atom_graph.adjacency, labels=labels))


def _solution_count_range(solution_counts: Sequence[int]) -> Tuple[int, int]:
    if not solution_counts:
        return (0, 0)
    return (int(min(solution_counts)), int(max(solution_counts)))


def _solution_confidence(solution_energies: Sequence[float]) -> float:
    if not solution_energies:
        return 0.0
    if len(solution_energies) == 1:
        return 1.0
    best = float(solution_energies[0])
    second = float(solution_energies[1])
    gap = max(0.0, second - best)
    return float(max(0.05, min(1.0, 0.35 + gap / max(abs(best), 1.0))))


def _json_dump(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
