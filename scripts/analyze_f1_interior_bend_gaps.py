#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    dot = abs(float(np.dot(_unit(lhs), _unit(rhs))))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
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


def _pca_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, List[List[float]]]:
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        return centroid, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), 0.0, 0.0, [centroid.tolist(), centroid.tolist()]
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = _unit(vh[0])
    projections = (points - centroid) @ axis
    closest = centroid + np.outer(projections, axis)
    distances = np.linalg.norm(points - closest, axis=1)
    endpoint_min = centroid + axis * float(np.min(projections))
    endpoint_max = centroid + axis * float(np.max(projections))
    return (
        centroid,
        axis,
        float(np.max(projections) - np.min(projections)),
        float(np.median(distances)),
        [endpoint_min.tolist(), endpoint_max.tolist()],
    )


def _merge_fragment_hypotheses(candidates: Sequence[Mapping[str, Any]], atoms: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    admissible = [candidate for candidate in candidates if candidate.get("admissible")]
    if not admissible:
        return []
    remaining = list(admissible)
    groups: List[List[Mapping[str, Any]]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            keep: List[Mapping[str, Any]] = []
            for candidate in remaining:
                if any(_should_merge_fragments(candidate, member) for member in group):
                    group.append(candidate)
                    changed = True
                else:
                    keep.append(candidate)
            remaining = keep
        groups.append(group)

    hypotheses: List[Dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        atom_ids = sorted({str(atom_id) for candidate in group for atom_id in candidate.get("atom_ids") or ()})
        points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids if atom_id in atoms], dtype=np.float64)
        if len(points) == 0:
            continue
        centroid, axis, span, residual, endpoints = _pca_line(points)
        source_ids = [str(candidate.get("candidate_id")) for candidate in group]
        score_values = [float(candidate.get("mean_interior_gap_score") or 0.0) for candidate in group]
        hypotheses.append(
            {
                "hypothesis_id": f"IBH{index}",
                "source_candidate_ids": source_ids,
                "atom_ids": atom_ids,
                "atom_count": len(atom_ids),
                "centroid": [round(float(value), 6) for value in centroid.tolist()],
                "axis_direction": [round(float(value), 6) for value in axis.tolist()],
                "endpoints": [[round(float(value), 6) for value in endpoint] for endpoint in endpoints],
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "mean_interior_gap_score": round(float(sum(score_values) / max(1, len(score_values))), 6),
                "status": "merged_fragment_hypothesis" if len(group) > 1 else "single_fragment_hypothesis",
                "reason_codes": ["interior_gap_fragments_merged"] if len(group) > 1 else ["single_interior_gap_candidate"],
            }
        )
    hypotheses.sort(key=lambda row: (-int(row.get("atom_count") or 0), row.get("hypothesis_id")))
    for index, hypothesis in enumerate(hypotheses, start=1):
        hypothesis["hypothesis_id"] = f"IBH{index}"
    return hypotheses


def _should_merge_fragments(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    lhs_axis = _unit(lhs.get("axis_direction") or (1, 0, 0))
    rhs_axis = _unit(rhs.get("axis_direction") or (1, 0, 0))
    axis_delta = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(lhs_axis, rhs_axis)))))))
    lhs_centroid = _vec(lhs.get("centroid") or (0, 0, 0))
    rhs_centroid = _vec(rhs.get("centroid") or (0, 0, 0))
    centroid_gap = float(np.linalg.norm(lhs_centroid - rhs_centroid))
    span_scale = max(float(lhs.get("span_mm") or 0.0), float(rhs.get("span_mm") or 0.0), 1.0)
    return axis_delta <= 30.0 and centroid_gap <= max(1.15 * span_scale, 28.0)


def analyze_interior_bend_gaps(
    *,
    decomposition: Mapping[str, Any],
    score_threshold: float = 0.58,
    interior_percentile_low: float = 20.0,
    interior_percentile_high: float = 80.0,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (graph.get("adjacency") or {}).items()}
    owned_bend_atoms = {
        str(atom_id)
        for region in decomposition.get("owned_bend_regions") or ()
        for atom_id in region.get("owned_atom_ids") or ()
    }
    points = np.asarray([_vec(atom.get("centroid") or (0, 0, 0)) for atom in atoms.values()], dtype=np.float64)
    if len(points) == 0:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scan_path": decomposition.get("scan_path"),
            "part_id": decomposition.get("part_id"),
            "purpose": "Diagnostic-only interior crease/ridge search for missed middle bends.",
            "score_threshold": score_threshold,
            "interior_percentile_low": interior_percentile_low,
            "interior_percentile_high": interior_percentile_high,
            "interior_bounds": {"low": [], "high": []},
            "selected_atom_count": 0,
            "candidate_count": 0,
            "admissible_candidate_count": 0,
            "merged_hypothesis_count": 0,
            "status": "no_atoms",
            "top_candidate_ids": [],
            "top_admissible_candidate_ids": [],
            "top_merged_hypothesis_ids": [],
            "merged_hypotheses": [],
            "candidates": [],
            "atom_scores": [],
        }
    low = np.percentile(points, interior_percentile_low, axis=0)
    high = np.percentile(points, interior_percentile_high, axis=0)

    score_rows: Dict[str, Dict[str, Any]] = {}
    selected: List[str] = []
    for atom_id, atom in atoms.items():
        point = _vec(atom.get("centroid") or (0, 0, 0))
        interior = bool(np.all(point >= low) and np.all(point <= high))
        if not interior or atom_id in owned_bend_atoms:
            continue
        label_type = label_types.get(atom_labels.get(atom_id), "unassigned")
        if label_type == "outlier":
            continue
        neighbors = [str(neighbor) for neighbor in adjacency.get(atom_id, ()) or () if str(neighbor) in atoms]
        if not neighbors:
            continue
        normal_angles = [
            _axis_angle_deg(atom.get("normal") or (1, 0, 0), atoms[neighbor].get("normal") or (1, 0, 0))
            for neighbor in neighbors
        ]
        contrast = min(1.0, float(np.percentile(normal_angles, 80)) / 70.0)
        curvature = min(1.0, float(atom.get("curvature_proxy") or 0.0) * 20.0)
        score = 0.55 * curvature + 0.45 * contrast
        row = {
            "atom_id": atom_id,
            "interior": interior,
            "label_type": label_type,
            "curvature_signal": round(float(curvature), 6),
            "normal_contrast": round(float(contrast), 6),
            "interior_gap_score": round(float(score), 6),
        }
        score_rows[atom_id] = row
        if score >= score_threshold:
            selected.append(atom_id)

    candidates: List[Dict[str, Any]] = []
    for index, component in enumerate(_connected_components(selected, adjacency), start=1):
        component_points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in component], dtype=np.float64)
        centroid, axis, span, residual, endpoints = _pca_line(component_points)
        values = [score_rows[atom_id]["interior_gap_score"] for atom_id in component if atom_id in score_rows]
        label_mix: Dict[str, int] = {}
        for atom_id in component:
            label_type = label_types.get(atom_labels.get(atom_id), "unassigned")
            label_mix[label_type] = label_mix.get(label_type, 0) + 1
        reasons: List[str] = []
        if len(component) < 4:
            reasons.append("support_atom_count_below_floor")
        if span < 10.0:
            reasons.append("span_below_floor")
        if residual > 6.0:
            reasons.append("weak_line_fit")
        if sum(values) / max(1, len(values)) < score_threshold:
            reasons.append("score_below_floor")
        if not reasons:
            reasons.append("interior_bend_gap_candidate")
        candidates.append(
            {
                "candidate_id": f"IBG{index}",
                "atom_ids": component,
                "atom_count": len(component),
                "centroid": [round(float(value), 6) for value in centroid.tolist()],
                "axis_direction": [round(float(value), 6) for value in axis.tolist()],
                "endpoints": [[round(float(value), 6) for value in endpoint] for endpoint in endpoints],
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "mean_interior_gap_score": round(float(sum(values) / max(1, len(values))), 6),
                "label_mix": dict(sorted(label_mix.items())),
                "admissible": reasons == ["interior_bend_gap_candidate"],
                "reason_codes": reasons,
            }
        )
    candidates.sort(
        key=lambda row: (
            not bool(row.get("admissible")),
            -float(row.get("mean_interior_gap_score") or 0.0),
            -int(row.get("atom_count") or 0),
            row.get("candidate_id"),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"IBG{index}"
    admissible = [candidate for candidate in candidates if candidate.get("admissible")]
    merged_hypotheses = _merge_fragment_hypotheses(candidates, atoms)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only interior crease/ridge search for missed middle bends.",
        "score_threshold": score_threshold,
        "interior_percentile_low": interior_percentile_low,
        "interior_percentile_high": interior_percentile_high,
        "interior_bounds": {
            "low": [round(float(value), 6) for value in low.tolist()],
            "high": [round(float(value), 6) for value in high.tolist()],
        },
        "selected_atom_count": len(selected),
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "merged_hypothesis_count": len(merged_hypotheses),
        "status": "interior_gap_candidates_found" if admissible else "no_interior_gap_candidate",
        "top_candidate_ids": [candidate["candidate_id"] for candidate in candidates[:8]],
        "top_admissible_candidate_ids": [candidate["candidate_id"] for candidate in admissible[:8]],
        "top_merged_hypothesis_ids": [hypothesis["hypothesis_id"] for hypothesis in merged_hypotheses[:8]],
        "merged_hypotheses": merged_hypotheses,
        "candidates": candidates,
        "atom_scores": [score_rows[atom_id] for atom_id in sorted(score_rows)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find interior crease/ridge candidates for missed F1 middle bends.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.58)
    parser.add_argument("--interior-percentile-low", type=float, default=20.0)
    parser.add_argument("--interior-percentile-high", type=float, default=80.0)
    args = parser.parse_args()
    report = analyze_interior_bend_gaps(
        decomposition=_load_json(args.decomposition_json),
        score_threshold=float(args.score_threshold),
        interior_percentile_low=float(args.interior_percentile_low),
        interior_percentile_high=float(args.interior_percentile_high),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "selected_atom_count": report["selected_atom_count"],
                "candidate_count": report["candidate_count"],
                "admissible_candidate_count": report["admissible_candidate_count"],
                "merged_hypothesis_count": report["merged_hypothesis_count"],
                "top_admissible_candidate_ids": report["top_admissible_candidate_ids"],
                "top_merged_hypothesis_ids": report["top_merged_hypothesis_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
