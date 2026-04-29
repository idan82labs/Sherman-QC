#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _vec(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _unit(values: Sequence[Any]) -> np.ndarray:
    vector = _vec(values)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any], *, signless: bool = True) -> float:
    dot = float(np.dot(_unit(lhs), _unit(rhs)))
    if signless:
        dot = abs(dot)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


def _flange_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(flange.get("flange_id")): flange
        for flange in decomposition.get("flange_hypotheses") or ()
        if flange.get("flange_id")
    }


def _assignment(decomposition: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    assignment = decomposition.get("assignment") or {}
    return (
        {str(key): str(value) for key, value in (assignment.get("atom_labels") or {}).items()},
        {str(key): str(value) for key, value in (assignment.get("label_types") or {}).items()},
    )


def _owned_regions(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(region.get("bend_id")): region
        for region in decomposition.get("owned_bend_regions") or ()
        if region.get("bend_id")
    }


def _manual_sets(decomposition: Mapping[str, Any], feature_labels: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    regions = _owned_regions(decomposition)
    accepted_ids: set[str] = set()
    rejected_ids: set[str] = set()
    accepted_regions: Dict[str, Mapping[str, Any]] = {}
    if not feature_labels:
        return {"accepted_atoms": accepted_ids, "rejected_atoms": rejected_ids, "accepted_regions": accepted_regions}
    for row in feature_labels.get("region_labels") or ():
        bend_id = str(row.get("bend_id"))
        region = regions.get(bend_id)
        if not region:
            continue
        atoms = {str(value) for value in region.get("owned_atom_ids") or ()}
        if row.get("human_feature_label") == "conventional_bend":
            accepted_ids.update(atoms)
            accepted_regions[bend_id] = region
        elif row.get("human_feature_label") in {"fragment_or_duplicate", "false_positive", "non_bend"}:
            rejected_ids.update(atoms)
    return {"accepted_atoms": accepted_ids, "rejected_atoms": rejected_ids, "accepted_regions": accepted_regions}


def _normal_arc_cost(normal: Sequence[Any], flange_a: Mapping[str, Any], flange_b: Mapping[str, Any]) -> float:
    theta = _angle_deg(flange_a.get("normal") or (1, 0, 0), flange_b.get("normal") or (1, 0, 0), signless=False)
    lhs = _angle_deg(normal, flange_a.get("normal") or (1, 0, 0), signless=False)
    rhs = _angle_deg(normal, flange_b.get("normal") or (1, 0, 0), signless=False)
    return abs((lhs + rhs) - theta) / max(abs(theta), 1.0)


def _pca_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        return centroid, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), 0.0, 0.0
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = _unit(vh[0])
    projections = (points - centroid) @ axis
    closest = centroid + np.outer(projections, axis)
    distances = np.linalg.norm(points - closest, axis=1)
    span = float(np.max(projections) - np.min(projections))
    median_residual = float(np.median(distances)) if len(distances) else 0.0
    return centroid, axis, span, median_residual


def _touching_counts(
    atom_ids: Sequence[str],
    pair: Tuple[str, str],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
) -> Dict[str, int]:
    counts = {pair[0]: 0, pair[1]: 0}
    for atom_id in atom_ids:
        for neighbor in adjacency.get(atom_id, ()) or ():
            label = atom_labels.get(str(neighbor))
            if label in counts and label_types.get(label) == "flange":
                counts[label] += 1
    return counts


def _connected_components(atom_ids: Iterable[str], adjacency: Mapping[str, Sequence[str]]) -> List[List[str]]:
    remaining = {str(atom_id) for atom_id in atom_ids}
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            atom_id = stack.pop()
            for neighbor in adjacency.get(atom_id, ()) or ():
                neighbor_id = str(neighbor)
                if neighbor_id not in remaining:
                    continue
                remaining.remove(neighbor_id)
                stack.append(neighbor_id)
                component.append(neighbor_id)
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values))
    return components


def _component_line_quality(atom_ids: Sequence[str], atoms: Mapping[str, Mapping[str, Any]]) -> Tuple[float, float]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    _, _, span, residual = _pca_line(points)
    return span, residual


def _best_path_component(
    atom_ids: Sequence[str],
    *,
    atoms: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    min_atoms: int = 4,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    components = _connected_components(atom_ids, adjacency)
    diagnostics: List[Dict[str, Any]] = []
    best_component: List[str] = []
    best_score = -1.0
    for component in components:
        span, residual = _component_line_quality(component, atoms)
        score = min(1.0, len(component) / 24.0) + min(1.0, span / 24.0) + max(0.0, 1.0 - residual / 8.0)
        diagnostics.append(
            {
                "atom_count": len(component),
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "component_score": round(float(score), 6),
            }
        )
        if len(component) >= min_atoms and score > best_score:
            best_score = score
            best_component = list(component)
    return best_component or (components[0] if components else []), diagnostics


def _line_core_atoms(
    atom_ids: Sequence[str],
    *,
    atoms: Mapping[str, Mapping[str, Any]],
    locked: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    if locked or len(atom_ids) < 8:
        return sorted(atom_ids), {"core_trimmed": False, "input_atom_count": len(atom_ids), "core_atom_count": len(atom_ids)}
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis, span, residual = _pca_line(points)
    projections = (points - centroid) @ axis
    closest = centroid + np.outer(projections, axis)
    distances = np.linalg.norm(points - closest, axis=1)
    distance_gate = max(float(np.median(distances)) * 1.75, 3.5)
    lo, hi = np.quantile(projections, [0.06, 0.94])
    core = [
        atom_id
        for atom_id, distance, projection in zip(atom_ids, distances, projections)
        if float(distance) <= distance_gate and float(lo) <= float(projection) <= float(hi)
    ]
    if len(core) < 6:
        core = [
            atom_id
            for atom_id, distance in zip(atom_ids, distances)
            if float(distance) <= max(distance_gate, 5.5)
        ]
    if len(core) < 6:
        return sorted(atom_ids), {
            "core_trimmed": False,
            "input_atom_count": len(atom_ids),
            "core_atom_count": len(atom_ids),
            "trim_rejected": "too_few_core_atoms",
        }
    return sorted(core), {
        "core_trimmed": len(core) < len(atom_ids),
        "input_atom_count": len(atom_ids),
        "core_atom_count": len(core),
        "distance_gate_mm": round(float(distance_gate), 6),
        "projection_quantile_interval": [round(float(lo), 6), round(float(hi), 6)],
        "pretrim_span_mm": round(float(span), 6),
        "pretrim_median_line_residual_mm": round(float(residual), 6),
    }


def _support_for_pair(
    *,
    pair: Tuple[str, str],
    ridge_atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    score_by_atom: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    flange_a = flanges[pair[0]]
    flange_b = flanges[pair[1]]
    pair_axis = _unit(np.cross(_unit(flange_a.get("normal") or (1, 0, 0)), _unit(flange_b.get("normal") or (0, 1, 0))))
    selected: List[str] = []
    for atom_id in ridge_atom_ids:
        atom = atoms.get(atom_id)
        if not atom:
            continue
        normal_cost = _normal_arc_cost(atom.get("normal") or (1, 0, 0), flange_a, flange_b)
        axis_cost = _angle_deg(atom.get("axis_direction") or pair_axis, pair_axis) / 55.0
        ridge_score = float((score_by_atom.get(atom_id) or {}).get("ridge_score") or 0.0)
        if normal_cost <= 0.50 and axis_cost <= 1.10 and ridge_score >= 0.58:
            selected.append(atom_id)
    return selected


def _line_candidate(
    *,
    candidate_id: str,
    pair: Tuple[str, str],
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    locked_pair: Tuple[str, str],
    preserve_support_components: bool = False,
) -> Dict[str, Any]:
    flange_a = flanges[pair[0]]
    flange_b = flanges[pair[1]]
    is_locked = pair == locked_pair
    if preserve_support_components:
        path_atom_ids = [str(value) for value in atom_ids]
        span, residual = _component_line_quality(path_atom_ids, atoms)
        path_component_diagnostics = [
            {
                "atom_count": len(path_atom_ids),
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "component_score": None,
                "preserved_merged_support": True,
            }
        ]
    else:
        path_atom_ids, path_component_diagnostics = _best_path_component(
            atom_ids,
            atoms=atoms,
            adjacency=adjacency,
        )
    core_atom_ids, core_diagnostics = _line_core_atoms(path_atom_ids, atoms=atoms, locked=is_locked)
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in core_atom_ids], dtype=np.float64)
    centroid, axis, span, residual = _pca_line(points)
    projections = (points - centroid) @ axis if len(points) else np.asarray([], dtype=np.float64)
    if len(projections):
        endpoint_min = centroid + axis * float(np.min(projections))
        endpoint_max = centroid + axis * float(np.max(projections))
    else:
        endpoint_min = centroid
        endpoint_max = centroid
    expected_axis = _unit(np.cross(_unit(flange_a.get("normal") or (1, 0, 0)), _unit(flange_b.get("normal") or (0, 1, 0))))
    axis_delta = _angle_deg(axis, expected_axis)
    touch_counts = _touching_counts(
        core_atom_ids,
        pair,
        adjacency=adjacency,
        atom_labels=atom_labels,
        label_types=label_types,
    )
    present_contacts = sum(1 for value in touch_counts.values() if value > 0)
    normal_costs = [_normal_arc_cost(atoms[atom_id].get("normal") or (1, 0, 0), flange_a, flange_b) for atom_id in core_atom_ids]
    mean_normal_cost = float(sum(normal_costs) / max(1, len(normal_costs)))
    support_mass = float(sum(float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in core_atom_ids))
    is_chord = pair == tuple(sorted((locked_pair[0], "F1"))) or pair == tuple(sorted(("F1", "F17")))
    # Diagnostic score: higher is better, not calibrated as final energy.
    score = (
        min(1.0, len(core_atom_ids) / 35.0)
        + min(1.0, span / 20.0)
        + max(0.0, 1.0 - residual / 6.0)
        + max(0.0, 1.0 - axis_delta / 35.0)
        + max(0.0, 1.0 - mean_normal_cost / 0.45)
        + present_contacts / 2.0
    )
    penalty = 0.0
    reasons: List[str] = []
    if len(core_atom_ids) < 6:
        penalty += 0.8
        reasons.append("short_support")
    if span < 10.0:
        penalty += 0.8
        reasons.append("short_span")
    if residual > 8.0:
        penalty += 0.6
        reasons.append("weak_line_fit")
    if axis_delta > 40.0:
        penalty += 0.6
        reasons.append("axis_mismatch")
    if present_contacts < 1:
        penalty += 0.5
        reasons.append("no_flange_contact")
    if is_chord and not is_locked:
        penalty += 0.7
        reasons.append("possible_chord")
    if not reasons:
        reasons.append("bend_line_candidate")
    return {
        "candidate_id": candidate_id,
        "source_kind": "ridge_pair_support",
        "flange_pair": list(pair),
        "atom_ids": sorted(core_atom_ids),
        "proposal_atom_ids": sorted(str(value) for value in atom_ids),
        "path_atom_ids": sorted(path_atom_ids),
        "atom_count": len(core_atom_ids),
        "proposal_atom_count": len(atom_ids),
        "path_atom_count": len(path_atom_ids),
        "support_mass": round(float(support_mass), 6),
        "centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "endpoints": [
            [round(float(value), 6) for value in endpoint_min.tolist()],
            [round(float(value), 6) for value in endpoint_max.tolist()],
        ],
        "expected_axis": [round(float(value), 6) for value in expected_axis.tolist()],
        "axis_delta_deg": round(float(axis_delta), 6),
        "span_mm": round(float(span), 6),
        "median_line_residual_mm": round(float(residual), 6),
        "mean_normal_arc_cost": round(float(mean_normal_cost), 6),
        "touch_counts": touch_counts,
        "present_contact_count": present_contacts,
        "locked_accepted": is_locked,
        "possible_chord": is_chord and not is_locked,
        "candidate_score": round(float(score - penalty), 6),
        "raw_score": round(float(score), 6),
        "penalty": round(float(penalty), 6),
        "reason_codes": reasons,
        "path_component_diagnostics": path_component_diagnostics,
        "line_core_diagnostics": core_diagnostics,
        "admissible": score - penalty >= 3.25 and span >= 10.0 and len(core_atom_ids) >= 6 and not (is_chord and not is_locked),
    }


def _interior_birth_incidence_status(candidate: Mapping[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if int(candidate.get("present_contact_count") or 0) < 2:
        reasons.append("missing_two_sided_flange_contact")
    axis_delta = 999.0 if candidate.get("axis_delta_deg") is None else float(candidate.get("axis_delta_deg"))
    normal_arc = 999.0 if candidate.get("mean_normal_arc_cost") is None else float(candidate.get("mean_normal_arc_cost"))
    span = 0.0 if candidate.get("span_mm") is None else float(candidate.get("span_mm"))
    atom_count = 0 if candidate.get("atom_count") is None else int(candidate.get("atom_count"))
    if axis_delta > 28.0:
        reasons.append("axis_mismatch_for_interior_birth")
    if normal_arc > 0.95:
        reasons.append("weak_normal_arc_for_interior_birth")
    if span < 18.0:
        reasons.append("short_interior_birth_span")
    if atom_count < 8:
        reasons.append("small_interior_birth_support")
    if not reasons:
        return "incidence_validated", ["interior_birth_incidence_validated"]
    if reasons == ["missing_two_sided_flange_contact"] or (
        "missing_two_sided_flange_contact" in reasons
        and axis_delta <= 25.0
        and normal_arc <= 0.75
        and span >= 18.0
    ):
        return "incidence_pending", reasons
    return "incidence_rejected", reasons


def _interior_birth_candidates(
    *,
    interior_gaps_payload: Optional[Mapping[str, Any]],
    candidate_pairs: Sequence[Tuple[str, str]],
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    locked_pair: Tuple[str, str],
    allow_interior_birth_counting: bool,
) -> List[Dict[str, Any]]:
    if not interior_gaps_payload:
        return []
    candidates: List[Dict[str, Any]] = []
    birth_index = 1
    for hypothesis in interior_gaps_payload.get("merged_hypotheses") or ():
        atom_ids = [str(value) for value in hypothesis.get("atom_ids") or () if str(value) in atoms]
        if len(atom_ids) < 4:
            continue
        for pair in candidate_pairs:
            line_candidate = _line_candidate(
                candidate_id=f"IBL{birth_index}",
                pair=pair,
                atom_ids=atom_ids,
                atoms=atoms,
                flanges=flanges,
                adjacency=adjacency,
                atom_labels=atom_labels,
                label_types=label_types,
                locked_pair=locked_pair,
                preserve_support_components=True,
            )
            birth_index += 1
            status, incidence_reasons = _interior_birth_incidence_status(line_candidate)
            line_candidate["source_kind"] = "interior_gap_birth"
            line_candidate["interior_hypothesis_id"] = hypothesis.get("hypothesis_id")
            line_candidate["interior_source_candidate_ids"] = list(hypothesis.get("source_candidate_ids") or ())
            line_candidate["interior_incidence_status"] = status
            line_candidate["interior_incidence_reason_codes"] = incidence_reasons
            line_candidate["admissible_without_interior_gate"] = bool(line_candidate.get("admissible"))
            # Interior births must prove incidence before they can enter count selection.
            line_candidate["admissible"] = bool(
                allow_interior_birth_counting
                and line_candidate.get("admissible")
                and status == "incidence_validated"
            )
            candidates.append(line_candidate)
    candidates.sort(
        key=lambda row: (
            {"incidence_validated": 0, "incidence_pending": 1, "incidence_rejected": 2}.get(
                str(row.get("interior_incidence_status")),
                3,
            ),
            -float(row.get("candidate_score") or 0.0),
            row.get("candidate_id"),
        )
    )
    return candidates


def _interior_source_lookup(interior_gaps_payload: Optional[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    if not interior_gaps_payload:
        return {}
    rows: Dict[str, Mapping[str, Any]] = {}
    for candidate in interior_gaps_payload.get("candidates") or ():
        if candidate.get("candidate_id"):
            rows[str(candidate["candidate_id"])] = candidate
    for hypothesis in interior_gaps_payload.get("merged_hypotheses") or ():
        if hypothesis.get("hypothesis_id"):
            rows[str(hypothesis["hypothesis_id"])] = hypothesis
    return rows


def _owned_region_duplicate_diagnostics(
    *,
    candidate: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    atoms: Mapping[str, Mapping[str, Any]],
    same_pair_only: bool = True,
) -> Dict[str, Any]:
    candidate_pair = tuple(str(value) for value in candidate.get("flange_pair") or ())
    candidate_atoms = {str(value) for value in candidate.get("atom_ids") or ()}
    candidate_points = [_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in candidate_atoms if atom_id in atoms]
    if not candidate_points:
        return {"status": "no_candidate_points", "same_pair_owned_regions": []}
    candidate_centroid = np.mean(np.asarray(candidate_points, dtype=np.float64), axis=0)
    candidate_axis = _unit(candidate.get("axis_direction") or (1, 0, 0))
    candidate_projections = (np.asarray(candidate_points, dtype=np.float64) - candidate_centroid) @ candidate_axis
    candidate_interval = (
        float(np.min(candidate_projections)) if len(candidate_projections) else 0.0,
        float(np.max(candidate_projections)) if len(candidate_projections) else 0.0,
    )
    same_pair_rows: List[Dict[str, Any]] = []
    for region in decomposition.get("owned_bend_regions") or ():
        region_pair = tuple(str(value) for value in region.get("incident_flange_ids") or ())
        if same_pair_only and tuple(sorted(region_pair)) != tuple(sorted(candidate_pair)):
            continue
        region_atoms = {str(value) for value in region.get("owned_atom_ids") or ()}
        region_points = [_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in region_atoms if atom_id in atoms]
        if not region_points:
            continue
        region_points_array = np.asarray(region_points, dtype=np.float64)
        region_centroid, region_axis, region_span, region_residual = _pca_line(region_points_array)
        region_projections_on_candidate = (region_points_array - candidate_centroid) @ candidate_axis
        region_interval_on_candidate = (
            float(np.min(region_projections_on_candidate)) if len(region_projections_on_candidate) else 0.0,
            float(np.max(region_projections_on_candidate)) if len(region_projections_on_candidate) else 0.0,
        )
        atom_iou = len(candidate_atoms & region_atoms) / max(1, len(candidate_atoms | region_atoms))
        centroid_distance = float(np.linalg.norm(candidate_centroid - region_centroid))
        axis_delta = _angle_deg(candidate_axis, region_axis)
        line_distance = _line_to_line_distance(candidate_centroid, candidate_axis, region_centroid, region_axis)
        interval_overlap = _interval_overlap_fraction(candidate_interval, region_interval_on_candidate)
        same_pair_rows.append(
            {
                "bend_id": region.get("bend_id"),
                "incident_flange_ids": list(region_pair),
                "same_flange_pair": tuple(sorted(region_pair)) == tuple(sorted(candidate_pair)),
                "atom_iou": round(float(atom_iou), 6),
                "centroid_distance_mm": round(float(centroid_distance), 6),
                "axis_delta_deg": round(float(axis_delta), 6),
                "line_distance_mm": round(float(line_distance), 6),
                "candidate_axis_interval_overlap": round(float(interval_overlap), 6),
                "region_span_mm": round(float(region_span), 6),
                "region_line_residual_mm": round(float(region_residual), 6),
                "owned_atom_count": len(region_atoms),
            }
        )
    same_pair_rows.sort(
        key=lambda row: (
            _same_pair_alias_rank(row),
            float(row["line_distance_mm"]),
            float(row["centroid_distance_mm"]),
            -float(row["atom_iou"]),
            str(row["bend_id"]),
        )
    )
    status = "unique_recovered_support"
    if same_pair_rows:
        strongest = same_pair_rows[0]
        if _is_same_pair_duplicate(strongest):
            status = "duplicate_of_existing_owned_region"
        elif _is_same_pair_line_alias(strongest):
            status = "same_pair_line_alias"
        elif _is_offset_parallel_same_pair_candidate(strongest):
            status = "offset_parallel_same_pair_candidate"
        elif _is_same_pair_ambiguous(strongest):
            status = "ambiguous_near_existing_owned_region"
    return {
        "status": status,
        "same_pair_owned_regions": same_pair_rows[:8],
    }


def _line_to_line_distance(point_a: np.ndarray, axis_a: np.ndarray, point_b: np.ndarray, axis_b: np.ndarray) -> float:
    cross = np.cross(axis_a, axis_b)
    norm = float(np.linalg.norm(cross))
    delta = point_b - point_a
    if norm <= 1e-9:
        return float(np.linalg.norm(delta - float(np.dot(delta, axis_a)) * axis_a))
    return abs(float(np.dot(delta, cross / norm)))


def _interval_overlap_fraction(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    left_span = max(0.0, lhs[1] - lhs[0])
    right_span = max(0.0, rhs[1] - rhs[0])
    if left_span <= 1e-9 or right_span <= 1e-9:
        return 0.0
    overlap = max(0.0, min(lhs[1], rhs[1]) - max(lhs[0], rhs[0]))
    return overlap / max(1e-9, min(left_span, right_span))


def _is_same_pair_duplicate(row: Mapping[str, Any]) -> bool:
    return _float_or(row.get("atom_iou"), 0.0) > 0.12


def _is_same_pair_line_alias(row: Mapping[str, Any]) -> bool:
    return (
        _float_or(row.get("line_distance_mm"), 999.0) <= 4.0
        and _float_or(row.get("candidate_axis_interval_overlap"), 0.0) >= 0.40
        and _float_or(row.get("axis_delta_deg"), 999.0) <= 25.0
    )


def _is_same_pair_ambiguous(row: Mapping[str, Any]) -> bool:
    return (
        _float_or(row.get("centroid_distance_mm"), 999.0) < 18.0
        or (
            _float_or(row.get("line_distance_mm"), 999.0) <= 8.0
            and _float_or(row.get("candidate_axis_interval_overlap"), 0.0) >= 0.25
            and _float_or(row.get("axis_delta_deg"), 999.0) <= 35.0
        )
    )


def _is_offset_parallel_same_pair_candidate(row: Mapping[str, Any]) -> bool:
    return (
        _float_or(row.get("atom_iou"), 1.0) <= 0.02
        and _float_or(row.get("centroid_distance_mm"), 0.0) >= 12.0
        and _float_or(row.get("line_distance_mm"), 0.0) >= 10.0
        and _float_or(row.get("candidate_axis_interval_overlap"), 0.0) >= 0.40
        and _float_or(row.get("axis_delta_deg"), 999.0) <= 25.0
    )


def _float_or(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _same_pair_alias_rank(row: Mapping[str, Any]) -> int:
    if _is_same_pair_duplicate(row):
        return 0
    if _is_same_pair_line_alias(row):
        return 1
    if _is_offset_parallel_same_pair_candidate(row):
        return 3
    if _is_same_pair_ambiguous(row):
        return 2
    return 4


def _recovered_contact_birth_candidates(
    *,
    contact_recovery_payload: Optional[Mapping[str, Any]],
    interior_gaps_payload: Optional[Mapping[str, Any]],
    decomposition: Mapping[str, Any],
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    locked_pair: Tuple[str, str],
    allow_recovered_contact_counting: bool,
) -> List[Dict[str, Any]]:
    if not contact_recovery_payload:
        return []
    source_lookup = _interior_source_lookup(interior_gaps_payload)
    candidates: List[Dict[str, Any]] = []
    for index, row in enumerate(contact_recovery_payload.get("validated_pairs") or (), start=1):
        source_id = str(row.get("source_id") or "")
        source = source_lookup.get(source_id)
        pair = tuple(sorted(str(value) for value in row.get("flange_pair") or ()))
        if not source or len(pair) != 2 or pair[0] not in flanges or pair[1] not in flanges:
            continue
        atom_ids = [str(value) for value in source.get("atom_ids") or () if str(value) in atoms]
        if len(atom_ids) < 4:
            continue
        line_candidate = _line_candidate(
            candidate_id=f"RCB{index}",
            pair=pair,
            atom_ids=atom_ids,
            atoms=atoms,
            flanges=flanges,
            adjacency=adjacency,
            atom_labels=atom_labels,
            label_types=label_types,
            locked_pair=locked_pair,
            preserve_support_components=True,
        )
        duplicate_diagnostics = _owned_region_duplicate_diagnostics(
            candidate=line_candidate,
            decomposition=decomposition,
            atoms=atoms,
        )
        line_candidate["source_kind"] = "recovered_interior_contact_birth"
        line_candidate["interior_source_id"] = source_id
        line_candidate["contact_recovery_status"] = row.get("status")
        line_candidate["contact_recovery_score"] = row.get("score")
        line_candidate["contact_recovery_metrics"] = {
            "axis_delta_deg": row.get("axis_delta_deg"),
            "mean_normal_arc_cost": row.get("mean_normal_arc_cost"),
            "nearby_counts": row.get("nearby_counts"),
            "side_balance": row.get("side_balance"),
            "median_line_distance_mm": row.get("median_line_distance_mm"),
            "reason_codes": row.get("reason_codes"),
        }
        line_candidate["duplicate_diagnostics"] = duplicate_diagnostics
        line_candidate["admissible_without_recovered_contact_gate"] = bool(line_candidate.get("admissible"))
        line_candidate["admissible"] = bool(
            allow_recovered_contact_counting
            and line_candidate.get("admissible")
            and row.get("status") == "recovered_contact_validated"
            and duplicate_diagnostics.get("status") in {"unique_recovered_support", "offset_parallel_same_pair_candidate"}
        )
        candidates.append(line_candidate)
    candidates.sort(
        key=lambda item: (
            item.get("duplicate_diagnostics", {}).get("status") not in {"unique_recovered_support", "offset_parallel_same_pair_candidate"},
            -float(item.get("contact_recovery_score") or 0.0),
            item.get("candidate_id"),
        )
    )
    return candidates


def _suppress_raw_family_covered_ridge_candidates(
    *,
    candidates: Sequence[Dict[str, Any]],
    recovered_contact_candidates: Sequence[Mapping[str, Any]],
    decomposition: Mapping[str, Any],
    atoms: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    valid_recovered_candidates = [
        candidate
        for candidate in recovered_contact_candidates
        if candidate.get("admissible")
        and (candidate.get("duplicate_diagnostics") or {}).get("status")
        in {"unique_recovered_support", "offset_parallel_same_pair_candidate"}
    ]
    recovered_by_pair = {
        tuple(sorted(str(value) for value in candidate.get("flange_pair") or ())): candidate
        for candidate in valid_recovered_candidates
    }
    best_recovered = max(
        valid_recovered_candidates,
        key=lambda candidate: (float(candidate.get("candidate_score") or 0.0), str(candidate.get("candidate_id"))),
        default=None,
    )
    if not valid_recovered_candidates:
        return [dict(candidate) for candidate in candidates]
    next_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        pair = tuple(sorted(str(value) for value in row.get("flange_pair") or ()))
        recovered = recovered_by_pair.get(pair)
        if (
            recovered
            and row.get("source_kind") == "ridge_pair_support"
            and not row.get("locked_accepted")
            and row.get("admissible")
        ):
            duplicate_diagnostics = _owned_region_duplicate_diagnostics(
                candidate=row,
                decomposition=decomposition,
                atoms=atoms,
            )
            if duplicate_diagnostics.get("status") in {"duplicate_of_existing_owned_region", "same_pair_line_alias"}:
                row["admissible"] = False
                row["raw_family_cover_diagnostics"] = duplicate_diagnostics
                row["suppressed_by_recovered_contact_candidate"] = recovered.get("candidate_id")
                row["reason_codes"] = sorted(
                    set([*(str(value) for value in row.get("reason_codes") or ()), "raw_family_covered_when_recovered_contact_exists"])
                )
        elif (
            best_recovered
            and row.get("source_kind") == "ridge_pair_support"
            and not row.get("locked_accepted")
            and row.get("admissible")
            and float(row.get("candidate_score") or 0.0) <= float(best_recovered.get("candidate_score") or 0.0) + 0.75
        ):
            # Some bad scans produce generic ridge candidates on a raw edge family but
            # with a different flange pair. Keep this suppression comparable-score
            # gated so a genuinely strong direct transition can still win.
            duplicate_diagnostics = _owned_region_duplicate_diagnostics(
                candidate=row,
                decomposition=decomposition,
                atoms=atoms,
                same_pair_only=False,
            )
            if duplicate_diagnostics.get("status") in {"duplicate_of_existing_owned_region", "same_pair_line_alias"}:
                row["admissible"] = False
                row["raw_family_cover_diagnostics"] = duplicate_diagnostics
                row["suppressed_by_recovered_contact_candidate"] = best_recovered.get("candidate_id")
                row["reason_codes"] = sorted(
                    set([*(str(value) for value in row.get("reason_codes") or ()), "raw_family_covered_when_recovered_contact_exists"])
                )
        next_candidates.append(row)
    return next_candidates


def _candidate_overlap(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> float:
    left = {str(value) for value in lhs.get("atom_ids") or ()}
    right = {str(value) for value in rhs.get("atom_ids") or ()}
    return len(left & right) / max(1, len(left | right))


def _subset_score(subset: Sequence[Mapping[str, Any]], *, label_cost: float = 1.15, overlap_cost: float = 2.2) -> Dict[str, Any]:
    score = sum(float(item.get("candidate_score") or 0.0) for item in subset)
    penalty = label_cost * len([item for item in subset if not item.get("locked_accepted")])
    overlap_penalty = 0.0
    conflicts: List[Dict[str, Any]] = []
    for left, right in combinations(subset, 2):
        overlap = _candidate_overlap(left, right)
        if overlap <= 0.10:
            continue
        cost = overlap_cost * overlap
        overlap_penalty += cost
        conflicts.append(
            {
                "lhs": left.get("candidate_id"),
                "rhs": right.get("candidate_id"),
                "atom_iou": round(float(overlap), 6),
                "cost": round(float(cost), 6),
            }
        )
    total = score - penalty - overlap_penalty
    return {
        "candidate_ids": [item.get("candidate_id") for item in subset],
        "flange_pairs": [item.get("flange_pair") for item in subset],
        "new_bend_count": len([item for item in subset if not item.get("locked_accepted")]),
        "subset_score": round(float(total), 6),
        "raw_candidate_score": round(float(score), 6),
        "label_penalty": round(float(penalty), 6),
        "overlap_penalty": round(float(overlap_penalty), 6),
        "conflicts": conflicts,
    }


def solve_junction_bend_arrangement(
    *,
    decomposition: Mapping[str, Any],
    feature_labels: Mapping[str, Any],
    local_ridge_payload: Mapping[str, Any],
    interior_gaps_payload: Optional[Mapping[str, Any]] = None,
    contact_recovery_payload: Optional[Mapping[str, Any]] = None,
    flange_ids: Sequence[str] = ("F1", "F11", "F14", "F17"),
    target_new_bends: Optional[int] = None,
    allow_interior_birth_counting: bool = False,
    allow_recovered_contact_counting: bool = False,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    flanges = _flange_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in ((decomposition.get("atom_graph") or {}).get("adjacency") or {}).items()}
    manual = _manual_sets(decomposition, feature_labels)
    accepted_regions = manual["accepted_regions"]
    locked_region = next(iter(accepted_regions.values())) if accepted_regions else {}
    locked_pair = tuple(sorted(str(value) for value in locked_region.get("incident_flange_ids") or ("F11", "F14")))
    ridge = (local_ridge_payload.get("ridges") or [{}])[0]
    ridge_atom_ids = [str(value) for value in ridge.get("atom_ids") or () if str(value) in atoms]
    score_by_atom = {str(row.get("atom_id")): row for row in local_ridge_payload.get("atom_scores") or ()}

    candidate_pairs = [
        tuple(sorted(pair))
        for pair in combinations([flange_id for flange_id in flange_ids if flange_id in flanges], 2)
    ]
    candidates: List[Dict[str, Any]] = []
    for index, pair in enumerate(candidate_pairs, start=1):
        if pair == locked_pair:
            support = [str(value) for value in locked_region.get("owned_atom_ids") or () if str(value) in atoms]
        else:
            support = _support_for_pair(
                pair=pair,
                ridge_atom_ids=ridge_atom_ids,
                atoms=atoms,
                flanges=flanges,
                score_by_atom=score_by_atom,
            )
        if not support:
            continue
        candidates.append(
            _line_candidate(
                candidate_id=f"JBL{index}",
                pair=pair,
                atom_ids=support,
                atoms=atoms,
                flanges=flanges,
                adjacency=adjacency,
                atom_labels=atom_labels,
                label_types=label_types,
                locked_pair=locked_pair,
            )
        )
    interior_candidates = _interior_birth_candidates(
        interior_gaps_payload=interior_gaps_payload,
        candidate_pairs=candidate_pairs,
        atoms=atoms,
        flanges=flanges,
        adjacency=adjacency,
        atom_labels=atom_labels,
        label_types=label_types,
        locked_pair=locked_pair,
        allow_interior_birth_counting=allow_interior_birth_counting,
    )
    recovered_contact_candidates = _recovered_contact_birth_candidates(
        contact_recovery_payload=contact_recovery_payload,
        interior_gaps_payload=interior_gaps_payload,
        decomposition=decomposition,
        atoms=atoms,
        flanges=flanges,
        adjacency=adjacency,
        atom_labels=atom_labels,
        label_types=label_types,
        locked_pair=locked_pair,
        allow_recovered_contact_counting=allow_recovered_contact_counting,
    )
    candidates = _suppress_raw_family_covered_ridge_candidates(
        candidates=candidates,
        recovered_contact_candidates=recovered_contact_candidates,
        decomposition=decomposition,
        atoms=atoms,
    )
    candidates.extend(interior_candidates)
    candidates.extend(recovered_contact_candidates)
    candidates.sort(key=lambda item: (not item["locked_accepted"], -float(item["candidate_score"]), item["candidate_id"]))
    locked = [item for item in candidates if item.get("locked_accepted")]
    nonlocked = [item for item in candidates if not item.get("locked_accepted") and item.get("admissible")]

    subsets: List[Dict[str, Any]] = []
    locked_base = locked[:1]
    max_size = min(3, len(nonlocked))
    if target_new_bends is not None:
        max_size = min(max_size, max(0, int(target_new_bends)))
        min_size = max_size
    else:
        min_size = 0
    for size in range(min_size, max_size + 1):
        for selected in combinations(nonlocked, size):
            subset = [*locked_base, *selected]
            row = _subset_score(subset)
            row["hypothesis"] = f"H{len(locked_base) + size}_locked_{len(locked_base)}_plus_{size}_new"
            subsets.append(row)
    subsets.sort(key=lambda item: (-float(item["subset_score"]), int(item["new_bend_count"]), item["candidate_ids"]))
    best = subsets[0] if subsets else None
    selected_atoms = {str(atom_id) for candidate_id in ((best or {}).get("candidate_ids") or ()) for candidate in candidates if candidate.get("candidate_id") == candidate_id for atom_id in candidate.get("atom_ids") or ()}
    junction_atom_ids = sorted(set(ridge_atom_ids) - selected_atoms)
    status = "no_arrangement_candidate"
    if best:
        expected_new = int(target_new_bends) if target_new_bends is not None else 2
        actual_new = int(best.get("new_bend_count") or 0)
        if actual_new == expected_new and not best.get("conflicts"):
            status = "two_bend_arrangement_candidate"
            if expected_new == 1:
                status = "single_bend_arrangement_candidate"
        elif actual_new == expected_new:
            status = "two_bend_arrangement_conflicted"
            if expected_new == 1:
                status = "single_bend_arrangement_conflicted"
        elif actual_new < expected_new:
            status = "underfit_arrangement"
        else:
            status = "overfit_arrangement"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only junction-aware bend-line arrangement over a connected ridge complex.",
        "locked_accepted_pair": list(locked_pair),
        "input_flange_ids": [flange_id for flange_id in flange_ids if flange_id in flanges],
        "target_new_bends": target_new_bends,
        "allow_interior_birth_counting": allow_interior_birth_counting,
        "allow_recovered_contact_counting": allow_recovered_contact_counting,
        "ridge_atom_count": len(ridge_atom_ids),
        "interior_birth_candidate_count": len(interior_candidates),
        "interior_birth_status_counts": {
            status: len([candidate for candidate in interior_candidates if candidate.get("interior_incidence_status") == status])
            for status in sorted({str(candidate.get("interior_incidence_status")) for candidate in interior_candidates})
        },
        "top_interior_birth_candidates": interior_candidates[:10],
        "recovered_contact_birth_candidate_count": len(recovered_contact_candidates),
        "recovered_contact_duplicate_status_counts": {
            status: len(
                [
                    candidate
                    for candidate in recovered_contact_candidates
                    if candidate.get("duplicate_diagnostics", {}).get("status") == status
                ]
            )
            for status in sorted(
                {
                    str(candidate.get("duplicate_diagnostics", {}).get("status"))
                    for candidate in recovered_contact_candidates
                }
            )
        },
        "top_recovered_contact_birth_candidates": recovered_contact_candidates[:10],
        "candidate_count": len(candidates),
        "admissible_nonlocked_candidate_count": len(nonlocked),
        "junction_transition_atom_ids": junction_atom_ids,
        "junction_transition_atom_count": len(junction_atom_ids),
        "status": status,
        "best_hypothesis": best,
        "candidates": candidates,
        "hypotheses": subsets[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a diagnostic junction-aware bend-line arrangement for F1 local ridge support.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--feature-labels-json", required=True, type=Path)
    parser.add_argument("--local-ridge-json", required=True, type=Path)
    parser.add_argument("--interior-gaps-json", type=Path, default=None)
    parser.add_argument("--contact-recovery-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--flange-ids", nargs="+", default=["F1", "F11", "F14", "F17"])
    parser.add_argument("--target-new-bends", type=int, default=None)
    parser.add_argument("--allow-interior-birth-counting", action="store_true")
    parser.add_argument("--allow-recovered-contact-counting", action="store_true")
    args = parser.parse_args()

    report = solve_junction_bend_arrangement(
        decomposition=_load_json(args.decomposition_json),
        feature_labels=_load_json(args.feature_labels_json),
        local_ridge_payload=_load_json(args.local_ridge_json),
        interior_gaps_payload=_load_json(args.interior_gaps_json) if args.interior_gaps_json else None,
        contact_recovery_payload=_load_json(args.contact_recovery_json) if args.contact_recovery_json else None,
        flange_ids=args.flange_ids,
        target_new_bends=args.target_new_bends,
        allow_interior_birth_counting=bool(args.allow_interior_birth_counting),
        allow_recovered_contact_counting=bool(args.allow_recovered_contact_counting),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "interior_birth_candidate_count": report["interior_birth_candidate_count"],
                "interior_birth_status_counts": report["interior_birth_status_counts"],
                "recovered_contact_birth_candidate_count": report["recovered_contact_birth_candidate_count"],
                "recovered_contact_duplicate_status_counts": report["recovered_contact_duplicate_status_counts"],
                "best_hypothesis": report.get("best_hypothesis", {}).get("hypothesis"),
                "best_candidate_ids": report.get("best_hypothesis", {}).get("candidate_ids"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
