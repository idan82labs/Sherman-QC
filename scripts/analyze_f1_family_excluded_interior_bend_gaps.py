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


def _raw_family_lines(decomposition: Mapping[str, Any], atoms: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for region in decomposition.get("owned_bend_regions") or ():
        atom_ids = [str(atom_id) for atom_id in region.get("owned_atom_ids") or () if str(atom_id) in atoms]
        if len(atom_ids) < 2:
            continue
        points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
        centroid, axis, span, residual, endpoints = _pca_line(points)
        lines.append(
            {
                "bend_id": str(region.get("bend_id")),
                "incident_flange_ids": [str(value) for value in region.get("incident_flange_ids") or ()],
                "centroid": centroid,
                "axis_direction": axis,
                "span_mm": span,
                "median_line_residual_mm": residual,
                "endpoints": endpoints,
                "atom_ids": atom_ids,
            }
        )
    return lines


def _raw_line_covering_atom(
    point: np.ndarray,
    raw_lines: Sequence[Mapping[str, Any]],
    *,
    tube_radius_mm: float,
    span_pad_mm: float,
) -> Mapping[str, Any] | None:
    for line in raw_lines:
        centroid_value = line.get("centroid")
        axis_value = line.get("axis_direction")
        centroid = _vec(centroid_value if centroid_value is not None else (0, 0, 0))
        axis = _unit(axis_value if axis_value is not None else (1, 0, 0))
        delta = point - centroid
        projection = float(np.dot(delta, axis))
        span = float(line.get("span_mm") or 0.0)
        if abs(projection) > 0.5 * span + span_pad_mm:
            continue
        closest = centroid + projection * axis
        line_distance = float(np.linalg.norm(point - closest))
        if line_distance <= tube_radius_mm:
            return {
                "bend_id": line.get("bend_id"),
                "line_distance_mm": round(line_distance, 6),
                "axis_projection_mm": round(projection, 6),
            }
    return None


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
        scores = [float(candidate.get("mean_interior_gap_score") or 0.0) for candidate in group]
        source_ids = [str(candidate.get("candidate_id")) for candidate in group]
        hypotheses.append(
            {
                "hypothesis_id": f"FEIH{index}",
                "source_candidate_ids": source_ids,
                "atom_ids": atom_ids,
                "atom_count": len(atom_ids),
                "centroid": [round(float(value), 6) for value in centroid.tolist()],
                "axis_direction": [round(float(value), 6) for value in axis.tolist()],
                "endpoints": [[round(float(value), 6) for value in endpoint] for endpoint in endpoints],
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "mean_interior_gap_score": round(float(sum(scores) / max(1, len(scores))), 6),
                "status": "merged_family_excluded_hypothesis" if len(group) > 1 else "single_family_excluded_hypothesis",
                "reason_codes": ["family_excluded_fragments_merged"] if len(group) > 1 else ["single_family_excluded_candidate"],
            }
        )
    hypotheses.sort(key=lambda row: (-int(row.get("atom_count") or 0), row.get("hypothesis_id")))
    for index, hypothesis in enumerate(hypotheses, start=1):
        hypothesis["hypothesis_id"] = f"FEIH{index}"
    return hypotheses


def _should_merge_fragments(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    axis_delta = _axis_angle_deg(lhs.get("axis_direction") or (1, 0, 0), rhs.get("axis_direction") or (1, 0, 0))
    centroid_gap = float(np.linalg.norm(_vec(lhs.get("centroid") or (0, 0, 0)) - _vec(rhs.get("centroid") or (0, 0, 0))))
    span_scale = max(float(lhs.get("span_mm") or 0.0), float(rhs.get("span_mm") or 0.0), 1.0)
    return axis_delta <= 25.0 and centroid_gap <= max(1.05 * span_scale, 24.0)


def analyze_family_excluded_interior_bend_gaps(
    *,
    decomposition: Mapping[str, Any],
    score_threshold: float = 0.58,
    interior_percentile_low: float = 20.0,
    interior_percentile_high: float = 80.0,
    raw_family_tube_radius_mm: float | None = None,
    raw_family_span_pad_mm: float | None = None,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (graph.get("adjacency") or {}).items()}
    local_spacing = max(float(graph.get("local_spacing_mm") or 1.0), 1.0)
    tube_radius = float(raw_family_tube_radius_mm) if raw_family_tube_radius_mm is not None else max(8.0, 4.0 * local_spacing)
    span_pad = float(raw_family_span_pad_mm) if raw_family_span_pad_mm is not None else max(18.0, 8.0 * local_spacing)
    raw_lines = _raw_family_lines(decomposition, atoms)
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
            "purpose": "Diagnostic-only interior ridge search after excluding raw owned bend-line family tubes.",
            "status": "no_atoms",
            "candidate_count": 0,
            "admissible_candidate_count": 0,
            "selected_atom_count": 0,
            "raw_family_excluded_atom_count": 0,
            "candidates": [],
            "merged_hypotheses": [],
            "atom_scores": [],
        }
    low = np.percentile(points, interior_percentile_low, axis=0)
    high = np.percentile(points, interior_percentile_high, axis=0)
    selected: List[str] = []
    score_rows: Dict[str, Dict[str, Any]] = {}
    excluded_by_raw_family = 0
    for atom_id, atom in atoms.items():
        point = _vec(atom.get("centroid") or (0, 0, 0))
        interior = bool(np.all(point >= low) and np.all(point <= high))
        if not interior or atom_id in owned_bend_atoms:
            continue
        raw_cover = _raw_line_covering_atom(point, raw_lines, tube_radius_mm=tube_radius, span_pad_mm=span_pad)
        if raw_cover:
            excluded_by_raw_family += 1
            score_rows[atom_id] = {
                "atom_id": atom_id,
                "interior": interior,
                "excluded_by_raw_family": True,
                "raw_family_cover": raw_cover,
            }
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
            "excluded_by_raw_family": False,
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
        values = [score_rows[atom_id]["interior_gap_score"] for atom_id in component if "interior_gap_score" in score_rows.get(atom_id, {})]
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
            reasons.append("family_excluded_interior_bend_gap_candidate")
        candidates.append(
            {
                "candidate_id": f"FEIG{index}",
                "atom_ids": component,
                "atom_count": len(component),
                "centroid": [round(float(value), 6) for value in centroid.tolist()],
                "axis_direction": [round(float(value), 6) for value in axis.tolist()],
                "endpoints": [[round(float(value), 6) for value in endpoint] for endpoint in endpoints],
                "span_mm": round(float(span), 6),
                "median_line_residual_mm": round(float(residual), 6),
                "mean_interior_gap_score": round(float(sum(values) / max(1, len(values))), 6),
                "label_mix": dict(sorted(label_mix.items())),
                "admissible": reasons == ["family_excluded_interior_bend_gap_candidate"],
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
        candidate["candidate_id"] = f"FEIG{index}"
    admissible = [candidate for candidate in candidates if candidate.get("admissible")]
    merged_hypotheses = _merge_fragment_hypotheses(candidates, atoms)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only interior ridge search after excluding raw owned bend-line family tubes.",
        "score_threshold": score_threshold,
        "interior_percentile_low": interior_percentile_low,
        "interior_percentile_high": interior_percentile_high,
        "raw_family_tube_radius_mm": round(float(tube_radius), 6),
        "raw_family_span_pad_mm": round(float(span_pad), 6),
        "raw_family_line_count": len(raw_lines),
        "interior_bounds": {
            "low": [round(float(value), 6) for value in low.tolist()],
            "high": [round(float(value), 6) for value in high.tolist()],
        },
        "selected_atom_count": len(selected),
        "raw_family_excluded_atom_count": excluded_by_raw_family,
        "candidate_count": len(candidates),
        "admissible_candidate_count": len(admissible),
        "merged_hypothesis_count": len(merged_hypotheses),
        "status": "family_excluded_interior_candidates_found" if admissible else "no_family_excluded_interior_candidate",
        "top_candidate_ids": [candidate["candidate_id"] for candidate in candidates[:8]],
        "top_admissible_candidate_ids": [candidate["candidate_id"] for candidate in admissible[:8]],
        "top_merged_hypothesis_ids": [hypothesis["hypothesis_id"] for hypothesis in merged_hypotheses[:8]],
        "merged_hypotheses": merged_hypotheses,
        "candidates": candidates,
        "atom_scores": [score_rows[atom_id] for atom_id in sorted(score_rows)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Find interior bend-gap candidates after excluding raw owned bend-family tubes.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.58)
    parser.add_argument("--interior-percentile-low", type=float, default=20.0)
    parser.add_argument("--interior-percentile-high", type=float, default=80.0)
    parser.add_argument("--raw-family-tube-radius-mm", type=float)
    parser.add_argument("--raw-family-span-pad-mm", type=float)
    args = parser.parse_args()
    report = analyze_family_excluded_interior_bend_gaps(
        decomposition=_load_json(args.decomposition_json),
        score_threshold=float(args.score_threshold),
        interior_percentile_low=float(args.interior_percentile_low),
        interior_percentile_high=float(args.interior_percentile_high),
        raw_family_tube_radius_mm=args.raw_family_tube_radius_mm,
        raw_family_span_pad_mm=args.raw_family_span_pad_mm,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "selected_atom_count": report["selected_atom_count"],
                "raw_family_excluded_atom_count": report["raw_family_excluded_atom_count"],
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
