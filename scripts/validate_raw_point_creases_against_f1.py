#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


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


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    dot = abs(float(np.dot(_unit(lhs), _unit(rhs))))
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


def _distance_to_line(points: np.ndarray, anchor: Sequence[Any], axis: Sequence[Any]) -> np.ndarray:
    anchor_vec = _vec(anchor)
    axis_vec = _unit(axis)
    deltas = points - anchor_vec
    projected = anchor_vec + np.outer(deltas @ axis_vec, axis_vec)
    return np.linalg.norm(points - projected, axis=1)


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


def _flange_points(flange: Mapping[str, Any], atoms: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    points = [
        _vec((atoms.get(str(atom_id)) or {}).get("centroid") or (0.0, 0.0, 0.0))
        for atom_id in flange.get("atom_ids") or ()
        if str(atom_id) in atoms
    ]
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)


def _pair_proximity(points: np.ndarray, flange_points: np.ndarray) -> float:
    if len(points) == 0 or len(flange_points) == 0:
        return 999.0
    tree = cKDTree(flange_points)
    distances, _ = tree.query(points, k=1)
    return float(np.percentile(distances, 20))


def _candidate_points(candidate: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(candidate.get("sampled_points") or (), dtype=np.float64)


def _score_candidate_pair(
    *,
    candidate: Mapping[str, Any],
    pair: Tuple[str, str],
    flanges: Mapping[str, Mapping[str, Any]],
    atoms: Mapping[str, Mapping[str, Any]],
    local_spacing_mm: float,
) -> Optional[Dict[str, Any]]:
    points = _candidate_points(candidate)
    if len(points) == 0:
        return None
    left = flanges.get(pair[0])
    right = flanges.get(pair[1])
    if not left or not right:
        return None
    angle = _angle_deg(left.get("normal") or (1, 0, 0), right.get("normal") or (0, 1, 0))
    if angle < 18.0 or angle > 170.0:
        return None
    centroid = np.mean(points, axis=0)
    pair_line = _plane_pair_line(left, right, centroid)
    if pair_line is None:
        return None
    line_anchor, pair_axis = pair_line
    candidate_axis = _unit(candidate.get("axis_direction") or (1, 0, 0))
    axis_delta = _angle_deg(candidate_axis, pair_axis)
    line_distances = _distance_to_line(points, line_anchor, pair_axis)
    median_line_distance = float(np.median(line_distances))
    p80_line_distance = float(np.percentile(line_distances, 80))
    left_proximity = _pair_proximity(points, _flange_points(left, atoms))
    right_proximity = _pair_proximity(points, _flange_points(right, atoms))
    two_sided_proximity = max(left_proximity, right_proximity)
    contact_floor = max(22.0, 8.0 * local_spacing_mm)
    corridor_floor = max(24.0, 9.0 * local_spacing_mm)
    axis_score = max(0.0, 1.0 - axis_delta / 35.0)
    line_score = max(0.0, 1.0 - median_line_distance / corridor_floor)
    p80_line_score = max(0.0, 1.0 - p80_line_distance / (corridor_floor * 1.5))
    contact_score = max(0.0, 1.0 - two_sided_proximity / contact_floor)
    angle_score = min(1.0, angle / 90.0)
    score = (
        1.15 * axis_score
        + 1.00 * line_score
        + 0.55 * p80_line_score
        + 1.05 * contact_score
        + 0.25 * angle_score
    )
    reasons: List[str] = []
    if axis_delta > 38.0:
        reasons.append("axis_mismatch")
    if median_line_distance > corridor_floor:
        reasons.append("corridor_far_from_candidate")
    if p80_line_distance > corridor_floor * 1.65:
        reasons.append("candidate_not_inside_corridor")
    if two_sided_proximity > contact_floor:
        reasons.append("weak_two_sided_flange_support")
    if left_proximity > contact_floor:
        reasons.append("weak_left_flange_support")
    if right_proximity > contact_floor:
        reasons.append("weak_right_flange_support")
    return {
        "flange_pair": list(pair),
        "score": round(float(score), 6),
        "axis_delta_deg": round(float(axis_delta), 6),
        "pair_angle_deg": round(float(angle), 6),
        "median_line_distance_mm": round(float(median_line_distance), 6),
        "p80_line_distance_mm": round(float(p80_line_distance), 6),
        "left_flange_proximity_mm": round(float(left_proximity), 6),
        "right_flange_proximity_mm": round(float(right_proximity), 6),
        "two_sided_proximity_mm": round(float(two_sided_proximity), 6),
        "reason_codes": reasons,
        "promotion_safe": not reasons and score >= 2.65,
    }


def validate_raw_point_creases_against_f1(
    *,
    decomposition: Mapping[str, Any],
    raw_creases: Mapping[str, Any],
    max_pairs_per_candidate: int = 8,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    flanges = _flange_lookup(decomposition)
    local_spacing_mm = max(float((decomposition.get("atom_graph") or {}).get("local_spacing_mm") or 1.0), 1.0)
    flange_ids = sorted(flanges)
    pair_ids = [(left, right) for index, left in enumerate(flange_ids) for right in flange_ids[index + 1 :]]
    rows: List[Dict[str, Any]] = []
    for candidate in raw_creases.get("candidates") or ():
        if not candidate.get("admissible"):
            continue
        pair_scores = [
            score
            for pair in pair_ids
            if (score := _score_candidate_pair(candidate=candidate, pair=pair, flanges=flanges, atoms=atoms, local_spacing_mm=local_spacing_mm))
        ]
        pair_scores.sort(key=lambda row: (-float(row["score"]), row["flange_pair"]))
        best = pair_scores[0] if pair_scores else None
        reasons: List[str] = []
        if not best:
            reasons.append("no_valid_flange_pair")
        elif not best.get("promotion_safe"):
            reasons.extend(str(value) for value in best.get("reason_codes") or ())
            if float(best.get("score") or 0.0) < 2.65:
                reasons.append("flange_pair_score_below_floor")
        if float(candidate.get("thickness_mm") or 0.0) > 14.0:
            reasons.append("raw_candidate_too_thick_for_bridge")
        if float(candidate.get("linearity") or 0.0) < 0.82:
            reasons.append("raw_candidate_not_line_like_enough")
        bridge_safe = best is not None and not reasons
        rows.append(
            {
                "raw_candidate_id": candidate.get("candidate_id"),
                "point_count": candidate.get("point_count"),
                "span_mm": candidate.get("span_mm"),
                "thickness_mm": candidate.get("thickness_mm"),
                "linearity": candidate.get("linearity"),
                "mean_score": candidate.get("mean_score"),
                "best_flange_pair": best.get("flange_pair") if best else [],
                "best_pair_score": best.get("score") if best else 0.0,
                "bridge_safe": bool(bridge_safe),
                "reason_codes": sorted(set(reasons)),
                "top_pair_scores": pair_scores[: max(1, int(max_pairs_per_candidate))],
                "sampled_points": candidate.get("sampled_points") or [],
                "axis_direction": candidate.get("axis_direction"),
            }
        )
    rows.sort(key=lambda row: (not bool(row["bridge_safe"]), -float(row["best_pair_score"]), str(row["raw_candidate_id"])))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "raw_crease_source": raw_creases.get("scan_path"),
        "purpose": "Diagnostic-only bridge validation from raw-point crease candidates to F1 flange-pair support.",
        "local_spacing_mm": local_spacing_mm,
        "raw_admissible_candidate_count": sum(1 for candidate in raw_creases.get("candidates") or () if candidate.get("admissible")),
        "bridge_candidate_count": len(rows),
        "bridge_safe_candidate_count": sum(1 for row in rows if row["bridge_safe"]),
        "status": "bridge_candidates_found" if any(row["bridge_safe"] for row in rows) else "no_bridge_safe_candidate",
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw-point crease candidates against F1 flange-pair/contact geometry.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--raw-creases-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-pairs-per-candidate", type=int, default=8)
    args = parser.parse_args()
    report = validate_raw_point_creases_against_f1(
        decomposition=_load_json(args.decomposition_json),
        raw_creases=_load_json(args.raw_creases_json),
        max_pairs_per_candidate=int(args.max_pairs_per_candidate),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "bridge_candidate_count": report["bridge_candidate_count"],
                "bridge_safe_candidate_count": report["bridge_safe_candidate_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
