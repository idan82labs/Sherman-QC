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


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any], *, signless: bool = True) -> float:
    dot = float(np.dot(_unit(lhs), _unit(rhs)))
    if signless:
        dot = abs(dot)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _plane_pair_line(flange_a: Mapping[str, Any], flange_b: Mapping[str, Any], ref: Sequence[Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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


def _normal_arc_cost(normal: Sequence[Any], flange_a: Mapping[str, Any], flange_b: Mapping[str, Any]) -> float:
    theta = _angle_deg(flange_a.get("normal") or (1, 0, 0), flange_b.get("normal") or (1, 0, 0), signless=False)
    lhs = _angle_deg(normal, flange_a.get("normal") or (1, 0, 0), signless=False)
    rhs = _angle_deg(normal, flange_b.get("normal") or (1, 0, 0), signless=False)
    return abs((lhs + rhs) - theta) / max(abs(theta), 1.0)


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


def _touching_counts(atom_ids: Sequence[str], pair: Tuple[str, str], adjacency: Mapping[str, Sequence[str]], atom_labels: Mapping[str, str], label_types: Mapping[str, str]) -> Dict[str, int]:
    counts = {pair[0]: 0, pair[1]: 0}
    for atom_id in atom_ids:
        for neighbor in adjacency.get(atom_id, ()) or ():
            label = atom_labels.get(str(neighbor))
            if label in counts and label_types.get(label) == "flange":
                counts[label] += 1
    return counts


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
    return centroid, axis, span, float(np.median(distances))


def analyze_uncovered_bend_corridors(
    *,
    decomposition: Mapping[str, Any],
    exclude_owned_bend_atoms: bool = False,
    max_pairs: int = 80,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    flanges = _flange_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (graph.get("adjacency") or {}).items()}
    local_spacing = max(float(graph.get("local_spacing_mm") or 1.0), 1.0)
    all_points = np.asarray([_vec(atom.get("centroid") or (0, 0, 0)) for atom in atoms.values()], dtype=np.float64)
    global_centroid = np.mean(all_points, axis=0) if len(all_points) else np.zeros(3)

    owned_bend_atoms = {
        str(atom_id)
        for region in decomposition.get("owned_bend_regions") or ()
        for atom_id in region.get("owned_atom_ids") or ()
    }

    pair_rows: List[Tuple[float, Tuple[str, str], np.ndarray, np.ndarray]] = []
    flange_ids = sorted(flanges)
    for index, left_id in enumerate(flange_ids):
        for right_id in flange_ids[index + 1 :]:
            left = flanges[left_id]
            right = flanges[right_id]
            angle = _angle_deg(left.get("normal") or (1, 0, 0), right.get("normal") or (1, 0, 0), signless=False)
            if angle < 18.0 or angle > 170.0:
                continue
            line = _plane_pair_line(left, right, global_centroid)
            if line is None:
                continue
            anchor, axis = line
            pair_rows.append((_distance_to_line(global_centroid, anchor, axis), (left_id, right_id), anchor, axis))
    pair_rows.sort(key=lambda item: (item[0], item[1]))

    candidates: List[Dict[str, Any]] = []
    for _, pair, anchor, axis in pair_rows[: max(1, int(max_pairs))]:
        pair_atoms: List[str] = []
        for atom_id, atom in atoms.items():
            if exclude_owned_bend_atoms and atom_id in owned_bend_atoms:
                continue
            label = atom_labels.get(atom_id)
            label_type = label_types.get(label, "unassigned")
            if label_type == "outlier":
                continue
            point = atom.get("centroid") or (0, 0, 0)
            line_distance = _distance_to_line(point, anchor, axis)
            if line_distance > max(18.0, 9.0 * local_spacing):
                continue
            normal_arc = _normal_arc_cost(atom.get("normal") or (1, 0, 0), flanges[pair[0]], flanges[pair[1]])
            curvature_signal = min(1.0, float(atom.get("curvature_proxy") or 0.0) * 20.0)
            if normal_arc <= 0.78 and (curvature_signal >= 0.16 or label_type in {"residual", "flange", "bend"}):
                pair_atoms.append(atom_id)
        for component in _connected_components(pair_atoms, adjacency):
            if len(component) < 4:
                continue
            points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in component], dtype=np.float64)
            centroid, line_axis, span, residual = _pca_line(points)
            expected_axis = _unit(np.cross(_unit(flanges[pair[0]].get("normal") or (1, 0, 0)), _unit(flanges[pair[1]].get("normal") or (0, 1, 0))))
            axis_delta = _angle_deg(line_axis, expected_axis)
            normal_costs = [_normal_arc_cost(atoms[atom_id].get("normal") or (1, 0, 0), flanges[pair[0]], flanges[pair[1]]) for atom_id in component]
            touching = _touching_counts(component, pair, adjacency, atom_labels, label_types)
            present_contacts = sum(1 for value in touching.values() if value > 0)
            owned_overlap = len(set(component) & owned_bend_atoms) / max(1, len(component))
            support_mass = sum(float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in component)
            score = (
                min(1.0, len(component) / 28.0)
                + min(1.0, span / 35.0)
                + max(0.0, 1.0 - residual / 7.0)
                + max(0.0, 1.0 - axis_delta / 35.0)
                + max(0.0, 1.0 - float(median(normal_costs)) / 0.50)
                + present_contacts / 2.0
                + max(0.0, 1.0 - owned_overlap)
            )
            reasons: List[str] = []
            if span < 14.0:
                reasons.append("span_below_floor")
            if residual > 8.0:
                reasons.append("weak_line_fit")
            if axis_delta > 45.0:
                reasons.append("axis_mismatch")
            if present_contacts < 1:
                reasons.append("no_flange_contact")
            if owned_overlap > 0.65:
                reasons.append("mostly_existing_owned_bend_support")
            if score < 4.05:
                reasons.append("score_below_floor")
            if not reasons:
                reasons.append("uncovered_bend_corridor_candidate")
            projections = (points - centroid) @ line_axis
            endpoint_min = centroid + line_axis * float(np.min(projections))
            endpoint_max = centroid + line_axis * float(np.max(projections))
            candidates.append(
                {
                    "candidate_id": "",
                    "flange_pair": list(pair),
                    "atom_ids": sorted(component),
                    "atom_count": len(component),
                    "support_mass": round(float(support_mass), 6),
                    "centroid": [round(float(value), 6) for value in centroid.tolist()],
                    "axis_direction": [round(float(value), 6) for value in line_axis.tolist()],
                    "expected_axis": [round(float(value), 6) for value in expected_axis.tolist()],
                    "endpoints": [
                        [round(float(value), 6) for value in endpoint_min.tolist()],
                        [round(float(value), 6) for value in endpoint_max.tolist()],
                    ],
                    "span_mm": round(float(span), 6),
                    "median_line_residual_mm": round(float(residual), 6),
                    "axis_delta_deg": round(float(axis_delta), 6),
                    "median_normal_arc_cost": round(float(median(normal_costs)), 6),
                    "touch_counts": touching,
                    "owned_bend_atom_overlap": round(float(owned_overlap), 6),
                    "candidate_score": round(float(score), 6),
                    "admissible": reasons == ["uncovered_bend_corridor_candidate"],
                    "reason_codes": reasons,
                }
            )
    candidates.sort(
        key=lambda row: (
            not bool(row.get("admissible")),
            -float(row.get("candidate_score") or 0.0),
            float(row.get("owned_bend_atom_overlap") or 0.0),
            -int(row.get("atom_count") or 0),
            row.get("flange_pair") or [],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"UBC{index}"
    admissible = [row for row in candidates if row.get("admissible")]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only scan-wide search for uncovered bend corridors, including possible missed interior bends.",
        "exclude_owned_bend_atoms": exclude_owned_bend_atoms,
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "top_admissible_candidate_ids": [row["candidate_id"] for row in admissible[:8]],
        "status": "candidate_pool_has_uncovered_corridors" if admissible else "no_uncovered_corridor_candidate",
        "candidates": candidates[:120],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find scan-wide uncovered bend corridor candidates for missed interior bends.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--exclude-owned-bend-atoms", action="store_true")
    args = parser.parse_args()
    report = analyze_uncovered_bend_corridors(
        decomposition=_load_json(args.decomposition_json),
        exclude_owned_bend_atoms=bool(args.exclude_owned_bend_atoms),
        max_pairs=int(args.max_pairs),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
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
