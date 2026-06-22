#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
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


def _axis_angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    left = _unit(lhs)
    right = _unit(rhs)
    dot = abs(float(np.dot(left, right)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _normal_angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    left = _unit(lhs)
    right = _unit(rhs)
    dot = float(np.dot(left, right))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _plane_pair_line(
    flange_a: Mapping[str, Any],
    flange_b: Mapping[str, Any],
    ref: Sequence[Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    normal_a = _unit(flange_a.get("normal") or (0, 0, 0))
    normal_b = _unit(flange_b.get("normal") or (0, 0, 0))
    axis = np.cross(normal_a, normal_b)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return None
    axis = axis / norm
    matrix = np.vstack([normal_a, normal_b])
    offset = np.asarray([float(flange_a.get("d") or 0.0), float(flange_b.get("d") or 0.0)], dtype=np.float64)
    ref_vec = _vec(ref)
    residual = matrix @ ref_vec + offset
    point = ref_vec - matrix.T @ np.linalg.pinv(matrix @ matrix.T) @ residual
    return point, axis


def _distance_to_line(point: Sequence[Any], anchor: Sequence[Any], axis: Sequence[Any]) -> float:
    point_vec = _vec(point)
    anchor_vec = _vec(anchor)
    axis_vec = _unit(axis)
    delta = point_vec - anchor_vec
    projected = anchor_vec + float(np.dot(delta, axis_vec)) * axis_vec
    return float(np.linalg.norm(point_vec - projected))


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
    labels = {str(key): str(value) for key, value in (assignment.get("atom_labels") or {}).items()}
    label_types = {str(key): str(value) for key, value in (assignment.get("label_types") or {}).items()}
    return labels, label_types


def _edge_key(lhs: str, rhs: str) -> str:
    return "::".join(sorted((str(lhs), str(rhs))))


def _owned_region_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(region.get("bend_id")): region
        for region in decomposition.get("owned_bend_regions") or ()
        if region.get("bend_id")
    }


def _manual_region_sets(
    *,
    decomposition: Mapping[str, Any],
    feature_labels: Optional[Mapping[str, Any]],
) -> Dict[str, set[str]]:
    regions = _owned_region_lookup(decomposition)
    accepted: set[str] = set()
    rejected: set[str] = set()
    if not feature_labels:
        return {"accepted": accepted, "rejected": rejected}
    for row in feature_labels.get("region_labels") or ():
        bend_id = str(row.get("bend_id"))
        region = regions.get(bend_id)
        if not region:
            continue
        atoms = {str(value) for value in region.get("owned_atom_ids") or ()}
        if row.get("human_feature_label") == "conventional_bend":
            accepted.update(atoms)
        elif row.get("human_feature_label") in {"fragment_or_duplicate", "false_positive", "non_bend"}:
            rejected.update(atoms)
    return {"accepted": accepted, "rejected": rejected}


def _connected_components(atom_ids: Iterable[str], adjacency: Mapping[str, Sequence[str]]) -> List[List[str]]:
    remaining = set(str(value) for value in atom_ids)
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()) or ():
                neighbor_id = str(neighbor)
                if neighbor_id not in remaining:
                    continue
                remaining.remove(neighbor_id)
                stack.append(neighbor_id)
                component.append(neighbor_id)
        components.append(sorted(component))
    return components


def _touching_flanges(
    *,
    atom_id: str,
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
) -> set[str]:
    values: set[str] = set()
    for neighbor in adjacency.get(atom_id, ()) or ():
        label = atom_labels.get(str(neighbor))
        if label and label_types.get(label) == "flange":
            values.add(label)
    return values


def _normal_arc_cost(atom_normal: Sequence[Any], flange_a: Mapping[str, Any], flange_b: Mapping[str, Any]) -> float:
    theta = _normal_angle_deg(flange_a.get("normal") or (1, 0, 0), flange_b.get("normal") or (1, 0, 0))
    lhs = _normal_angle_deg(atom_normal, flange_a.get("normal") or (1, 0, 0))
    rhs = _normal_angle_deg(atom_normal, flange_b.get("normal") or (1, 0, 0))
    return abs((lhs + rhs) - theta) / max(theta, 1.0)


def _component_summary(
    *,
    component: Sequence[str],
    pair: Tuple[str, str],
    line_anchor: np.ndarray,
    line_axis: np.ndarray,
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    adjacency: Mapping[str, Sequence[str]],
    edge_weights: Mapping[str, Any],
    local_spacing: float,
) -> Dict[str, Any]:
    points = [_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in component if atom_id in atoms]
    distances = [_distance_to_line(point, line_anchor, line_axis) for point in points]
    projections = [float(np.dot(point, line_axis)) for point in points]
    span = max(projections) - min(projections) if projections else 0.0
    median_line = float(median(distances)) if distances else 999.0
    p90_line = float(np.percentile(distances, 90)) if distances else 999.0
    weights = [float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in component]
    support_mass = float(sum(weights))
    curvature = [
        min(1.0, float((atoms.get(atom_id) or {}).get("curvature_proxy") or 0.0) * 20.0)
        for atom_id in component
    ]
    bendness = float(sum(curvature) / max(1, len(curvature)))
    normal_costs = [
        _normal_arc_cost((atoms.get(atom_id) or {}).get("normal") or (1, 0, 0), flanges[pair[0]], flanges[pair[1]])
        for atom_id in component
        if atom_id in atoms
    ]
    median_normal_arc = float(median(normal_costs)) if normal_costs else 999.0
    touch_counts = {pair[0]: 0, pair[1]: 0}
    touch_weight = {pair[0]: 0.0, pair[1]: 0.0}
    labels_in_component: Dict[str, int] = {}
    for atom_id in component:
        label = atom_labels.get(atom_id, "unassigned")
        labels_in_component[label] = labels_in_component.get(label, 0) + 1
        for neighbor in adjacency.get(atom_id, ()) or ():
            neighbor_id = str(neighbor)
            neighbor_label = atom_labels.get(neighbor_id)
            if neighbor_label not in pair:
                continue
            touch_counts[neighbor_label] += 1
            touch_weight[neighbor_label] += float(edge_weights.get(_edge_key(atom_id, neighbor_id), 1.0))
    present_contacts = sum(1 for value in touch_counts.values() if value > 0)
    line_score = max(0.0, 1.0 - median_line / max(8.0, 5.0 * local_spacing))
    p90_score = max(0.0, 1.0 - p90_line / max(14.0, 8.0 * local_spacing))
    span_score = min(1.0, span / max(24.0, 10.0 * local_spacing))
    atom_score = min(1.0, math.log1p(len(component)) / math.log(18.0))
    mass_score = min(1.0, support_mass / 1200.0)
    contact_score = present_contacts / 2.0
    normal_score = max(0.0, 1.0 - median_normal_arc / 0.45)
    proposal_score = (
        0.20 * line_score
        + 0.12 * p90_score
        + 0.16 * normal_score
        + 0.14 * contact_score
        + 0.12 * bendness
        + 0.10 * atom_score
        + 0.08 * mass_score
        + 0.08 * span_score
    )
    reasons: List[str] = []
    if len(component) < 4:
        reasons.append("support_atom_count_below_floor")
    if span < max(8.0, 4.0 * local_spacing):
        reasons.append("span_below_floor")
    if median_line > max(10.0, 6.0 * local_spacing):
        reasons.append("weak_interface_line_fit")
    if median_normal_arc > 0.55:
        reasons.append("weak_normal_arc_fit")
    if present_contacts < 2:
        reasons.append("missing_two_sided_flange_contact")
    if proposal_score < 0.52:
        reasons.append("proposal_score_below_floor")
    if not reasons:
        reasons.append("residual_interface_support_candidate")
    centroid = np.mean(points, axis=0).tolist() if points else [0.0, 0.0, 0.0]
    return {
        "candidate_id": "",
        "flange_pair": list(pair),
        "atom_ids": list(component),
        "atom_count": len(component),
        "support_mass": round(float(support_mass), 6),
        "support_centroid": [round(float(value), 6) for value in centroid],
        "span_mm": round(float(span), 6),
        "median_line_distance_mm": round(float(median_line), 6),
        "p90_line_distance_mm": round(float(p90_line), 6),
        "median_normal_arc_cost": round(float(median_normal_arc), 6),
        "support_bendness": round(float(bendness), 6),
        "touch_counts": touch_counts,
        "touch_weight": {key: round(float(value), 6) for key, value in touch_weight.items()},
        "label_mix": dict(sorted(labels_in_component.items())),
        "component_scores": {
            "line_score": round(float(line_score), 6),
            "p90_score": round(float(p90_score), 6),
            "normal_score": round(float(normal_score), 6),
            "contact_score": round(float(contact_score), 6),
            "bendness": round(float(bendness), 6),
            "atom_score": round(float(atom_score), 6),
            "mass_score": round(float(mass_score), 6),
            "span_score": round(float(span_score), 6),
        },
        "proposal_score": round(float(proposal_score), 6),
        "admissible": reasons == ["residual_interface_support_candidate"],
        "reason_codes": reasons,
    }


def _candidate_conflicts(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for index, left in enumerate(candidates):
        left_atoms = {str(value) for value in left.get("atom_ids") or ()}
        left_centroid = _vec(left.get("support_centroid") or (0, 0, 0))
        for right in candidates[index + 1 :]:
            right_atoms = {str(value) for value in right.get("atom_ids") or ()}
            union = left_atoms | right_atoms
            atom_iou = len(left_atoms & right_atoms) / max(1, len(union))
            centroid_distance = float(np.linalg.norm(left_centroid - _vec(right.get("support_centroid") or (0, 0, 0))))
            same_pair = tuple(left.get("flange_pair") or ()) == tuple(right.get("flange_pair") or ())
            reason_codes: List[str] = []
            if atom_iou >= 0.20:
                reason_codes.append("atom_overlap")
            if centroid_distance <= 12.0:
                reason_codes.append("near_centroid")
            if same_pair:
                reason_codes.append("same_flange_pair")
            if not reason_codes:
                continue
            conflicts.append(
                {
                    "lhs": left.get("candidate_id"),
                    "rhs": right.get("candidate_id"),
                    "atom_iou": round(float(atom_iou), 6),
                    "centroid_distance_mm": round(float(centroid_distance), 6),
                    "same_flange_pair": same_pair,
                    "reason_codes": reason_codes,
                }
            )
    return conflicts


def _conflict_components(candidates: Sequence[Mapping[str, Any]], conflicts: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    ids = [str(item.get("candidate_id")) for item in candidates]
    parent = {candidate_id: candidate_id for candidate_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(lhs: str, rhs: str) -> None:
        left = find(lhs)
        right = find(rhs)
        if left != right:
            parent[right] = left

    known = set(ids)
    for conflict in conflicts:
        lhs = str(conflict.get("lhs"))
        rhs = str(conflict.get("rhs"))
        if lhs in known and rhs in known:
            union(lhs, rhs)
    groups: Dict[str, List[str]] = {}
    for candidate_id in ids:
        groups.setdefault(find(candidate_id), []).append(candidate_id)
    return [sorted(values) for values in groups.values()]


def analyze_residual_interface_support_candidates(
    *,
    decomposition: Mapping[str, Any],
    feature_labels: Optional[Mapping[str, Any]] = None,
    feature_label_summary: Optional[Mapping[str, Any]] = None,
    max_pairs: int = 80,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    flanges = _flange_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    atom_graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (atom_graph.get("adjacency") or {}).items()}
    edge_weights = {str(key): value for key, value in (atom_graph.get("edge_weights") or {}).items()}
    local_spacing = float(atom_graph.get("local_spacing_mm") or 1.0)
    manual_sets = _manual_region_sets(decomposition=decomposition, feature_labels=feature_labels)
    accepted_atoms = manual_sets["accepted"]
    rejected_atoms = manual_sets["rejected"]
    capacity = dict((feature_label_summary or {}).get("label_capacity") or {})
    recovery_deficit = max(
        int(capacity.get("valid_conventional_region_deficit") or 0),
        int(capacity.get("valid_counted_feature_deficit") or 0),
        int(capacity.get("owned_region_capacity_deficit") or 0),
    )

    pair_rows: List[Tuple[float, Tuple[str, str], np.ndarray, np.ndarray]] = []
    flange_ids = sorted(flanges)
    all_points = [_vec(atom.get("centroid") or (0, 0, 0)) for atom in atoms.values()]
    global_centroid = np.mean(np.asarray(all_points, dtype=np.float64), axis=0) if all_points else np.zeros(3)
    for index, left_id in enumerate(flange_ids):
        for right_id in flange_ids[index + 1 :]:
            left = flanges[left_id]
            right = flanges[right_id]
            angle = _normal_angle_deg(left.get("normal") or (1, 0, 0), right.get("normal") or (1, 0, 0))
            if angle < 22.0 or angle > 168.0:
                continue
            line = _plane_pair_line(left, right, global_centroid)
            if line is None:
                continue
            line_anchor, line_axis = line
            centroid_distance = _distance_to_line(global_centroid, line_anchor, line_axis)
            pair_rows.append((centroid_distance, (left_id, right_id), line_anchor, line_axis))
    pair_rows.sort(key=lambda item: (item[0], item[1]))

    candidates: List[Dict[str, Any]] = []
    for _, pair, line_anchor, line_axis in pair_rows[: max(1, int(max_pairs))]:
        pair_atoms: List[str] = []
        for atom_id, atom in atoms.items():
            if atom_id in accepted_atoms:
                continue
            label = atom_labels.get(atom_id, "unassigned")
            label_type = label_types.get(label, "unassigned")
            if label_type == "bend" and atom_id not in rejected_atoms:
                continue
            point = atom.get("centroid") or (0, 0, 0)
            line_distance = _distance_to_line(point, line_anchor, line_axis)
            if line_distance > max(16.0, 8.0 * local_spacing):
                continue
            normal_arc = _normal_arc_cost(atom.get("normal") or (1, 0, 0), flanges[pair[0]], flanges[pair[1]])
            touching = _touching_flanges(atom_id=atom_id, adjacency=adjacency, atom_labels=atom_labels, label_types=label_types)
            curvature_signal = min(1.0, float(atom.get("curvature_proxy") or 0.0) * 20.0)
            touches_pair = bool(touching & set(pair))
            eligible_label = label_type in {"residual", "outlier", "flange", "unassigned"} or atom_id in rejected_atoms
            if not eligible_label:
                continue
            if normal_arc <= 0.70 and (touches_pair or curvature_signal >= 0.25 or label_type == "residual"):
                pair_atoms.append(atom_id)
        for component in _connected_components(pair_atoms, adjacency):
            summary = _component_summary(
                component=component,
                pair=pair,
                line_anchor=line_anchor,
                line_axis=line_axis,
                atoms=atoms,
                flanges=flanges,
                atom_labels=atom_labels,
                label_types=label_types,
                adjacency=adjacency,
                edge_weights=edge_weights,
                local_spacing=local_spacing,
            )
            if summary["atom_count"] < 3:
                continue
            candidates.append(summary)

    candidates.sort(
        key=lambda item: (
            not bool(item.get("admissible")),
            -float(item.get("proposal_score") or 0.0),
            -int(item.get("atom_count") or 0),
            item.get("flange_pair") or [],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"RIC{index}"

    admissible = [item for item in candidates if item.get("admissible")]
    conflicts = _candidate_conflicts(admissible)
    conflict_components = _conflict_components(admissible, conflicts)
    status = "no_recovery_deficit" if recovery_deficit <= 0 else "no_residual_interface_candidate"
    ambiguity_limit = max(recovery_deficit + 2, recovery_deficit * 3)
    if recovery_deficit > 0 and len(admissible) > ambiguity_limit:
        status = "ambiguous_overcomplete_candidate_pool"
    elif recovery_deficit > 0 and len(admissible) > recovery_deficit and conflicts:
        status = "ambiguous_overlapping_candidate_pool"
    elif recovery_deficit > 0 and len(admissible) >= recovery_deficit:
        status = "candidate_pool_covers_deficit"
    elif recovery_deficit > 0 and admissible:
        status = "partial_candidate_pool"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only residual/flange-interface support proposal search. It does not change F1 accepted counts.",
        "recovery_deficit": recovery_deficit,
        "accepted_manual_atom_count": len(accepted_atoms),
        "rejected_manual_atom_count": len(rejected_atoms),
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "ambiguity_limit": ambiguity_limit,
        "admissible_conflict_count": len(conflicts),
        "admissible_conflict_components": conflict_components,
        "status": status,
        "top_candidate_ids": [item["candidate_id"] for item in candidates[:8]],
        "top_admissible_candidate_ids": [item["candidate_id"] for item in admissible[:8]],
        "admissible_conflicts": conflicts[:100],
        "candidates": candidates[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find diagnostic residual/flange-interface support candidates for missing F1 bend regions.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--feature-labels-json", type=Path, default=None)
    parser.add_argument("--feature-label-summary-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=80)
    args = parser.parse_args()

    report = analyze_residual_interface_support_candidates(
        decomposition=_load_json(args.decomposition_json),
        feature_labels=_load_json(args.feature_labels_json) if args.feature_labels_json else None,
        feature_label_summary=_load_json(args.feature_label_summary_json) if args.feature_label_summary_json else None,
        max_pairs=args.max_pairs,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "recovery_deficit": report["recovery_deficit"],
                "candidate_count": report["candidate_count"],
                "admissible_candidate_count": report["admissible_candidate_count"],
                "top_admissible_candidate_ids": report["top_admissible_candidate_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
