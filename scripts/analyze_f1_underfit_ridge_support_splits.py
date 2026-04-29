#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
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


def _pca_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        return centroid, np.asarray([1.0, 0.0, 0.0]), 0.0, 0.0, np.zeros(len(points))
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = _unit(vh[0])
    projections = (points - centroid) @ axis
    closest = centroid + np.outer(projections, axis)
    distances = np.linalg.norm(points - closest, axis=1)
    return centroid, axis, float(np.max(projections) - np.min(projections)), float(np.median(distances)), projections


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


def _touching_flanges(
    *,
    atom_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for atom_id in atom_ids:
        for neighbor in adjacency.get(atom_id, ()) or ():
            label = atom_labels.get(str(neighbor))
            if label and label_types.get(label) == "flange":
                counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _score_segment(
    *,
    segment_id: str,
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    score_by_atom: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, Sequence[str]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    local_spacing: float,
) -> Dict[str, Any]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis, span, residual, _ = _pca_line(points)
    ridge_scores = [float((score_by_atom.get(atom_id) or {}).get("ridge_score") or 0.0) for atom_id in atom_ids]
    mean_ridge = float(sum(ridge_scores) / max(1, len(ridge_scores)))
    touches = _touching_flanges(
        atom_ids=atom_ids,
        adjacency=adjacency,
        atom_labels=atom_labels,
        label_types=label_types,
    )
    top_touch_count = len([value for value in touches.values() if value > 0])
    score = (
        min(1.0, len(atom_ids) / 8.0)
        + min(1.0, span / max(12.0, 6.0 * local_spacing))
        + max(0.0, 1.0 - residual / max(6.0, 3.0 * local_spacing))
        + max(0.0, min(1.0, (mean_ridge - 0.50) / 0.25))
        + min(1.0, top_touch_count / 2.0)
    )
    reasons: List[str] = []
    if len(atom_ids) < 4:
        reasons.append("segment_atom_count_below_floor")
    if span < max(8.0, 4.0 * local_spacing):
        reasons.append("segment_span_below_floor")
    if residual > max(7.0, 3.5 * local_spacing):
        reasons.append("segment_line_fit_weak")
    if mean_ridge < 0.58:
        reasons.append("segment_ridge_score_below_floor")
    if top_touch_count < 1:
        reasons.append("segment_has_no_flange_contact")
    if not reasons:
        reasons.append("candidate_support_segment")
    return {
        "segment_id": segment_id,
        "atom_ids": sorted(atom_ids),
        "atom_count": len(atom_ids),
        "support_centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "span_mm": round(float(span), 6),
        "median_line_residual_mm": round(float(residual), 6),
        "mean_ridge_score": round(float(mean_ridge), 6),
        "touching_flange_counts": touches,
        "segment_score": round(float(score), 6),
        "admissible": reasons == ["candidate_support_segment"],
        "reason_codes": reasons,
    }


def _split_by_projection_gaps(
    *,
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    gap_threshold: float,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    _, axis, span, residual, projections = _pca_line(points)
    order = np.argsort(projections)
    sorted_ids = [atom_ids[int(index)] for index in order]
    sorted_projections = [float(projections[int(index)]) for index in order]
    gaps = [
        {
            "after_atom_id": sorted_ids[index],
            "before_atom_id": sorted_ids[index + 1],
            "gap_mm": round(float(sorted_projections[index + 1] - sorted_projections[index]), 6),
        }
        for index in range(len(sorted_ids) - 1)
    ]
    split_indices = [
        index + 1
        for index, row in enumerate(gaps)
        if float(row["gap_mm"]) >= gap_threshold
    ]
    segments: List[List[str]] = []
    start = 0
    for split_index in split_indices:
        segments.append(sorted(sorted_ids[start:split_index]))
        start = split_index
    segments.append(sorted(sorted_ids[start:]))
    diagnostics = {
        "ridge_span_mm": round(float(span), 6),
        "ridge_median_line_residual_mm": round(float(residual), 6),
        "ridge_axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "gap_threshold_mm": round(float(gap_threshold), 6),
        "largest_projection_gaps": sorted(gaps, key=lambda row: -float(row["gap_mm"]))[:12],
        "split_gap_count": len(split_indices),
    }
    return segments, diagnostics


def analyze_underfit_ridge_support_splits(
    *,
    decomposition: Mapping[str, Any],
    local_ridge_payload: Mapping[str, Any],
    feature_label_summary: Optional[Mapping[str, Any]] = None,
    min_gap_mm: Optional[float] = None,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    atom_graph = decomposition.get("atom_graph") or {}
    adjacency = {str(key): [str(value) for value in values or ()] for key, values in (atom_graph.get("adjacency") or {}).items()}
    local_spacing = float(atom_graph.get("local_spacing_mm") or 1.0)
    score_by_atom = {str(row.get("atom_id")): row for row in local_ridge_payload.get("atom_scores") or ()}
    ridges = [row for row in local_ridge_payload.get("ridges") or () if row.get("atom_ids")]
    capacity = dict((feature_label_summary or {}).get("label_capacity") or {})
    recovery_deficit = max(
        int(capacity.get("valid_conventional_region_deficit") or 0),
        int(capacity.get("valid_counted_feature_deficit") or 0),
        int(capacity.get("owned_region_capacity_deficit") or 0),
    )
    if not ridges:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scan_path": decomposition.get("scan_path"),
            "part_id": decomposition.get("part_id"),
            "status": "no_ridge_to_split",
            "recovery_deficit": recovery_deficit,
            "candidate_segment_count": 0,
            "admissible_segment_count": 0,
            "segments": [],
        }
    ridge = max(ridges, key=lambda row: len(row.get("atom_ids") or ()))
    ridge_atom_ids = [str(value) for value in ridge.get("atom_ids") or () if str(value) in atoms]
    gap_threshold = float(min_gap_mm) if min_gap_mm is not None else max(6.0, 5.0 * local_spacing)
    raw_segments, split_diagnostics = _split_by_projection_gaps(
        atom_ids=ridge_atom_ids,
        atoms=atoms,
        gap_threshold=gap_threshold,
    )
    segments = [
        _score_segment(
            segment_id=f"URS{index}",
            atom_ids=segment,
            atoms=atoms,
            score_by_atom=score_by_atom,
            adjacency=adjacency,
            atom_labels=atom_labels,
            label_types=label_types,
            local_spacing=local_spacing,
        )
        for index, segment in enumerate(raw_segments, start=1)
        if segment
    ]
    segments.sort(key=lambda row: (not row.get("admissible"), -float(row.get("segment_score") or 0.0), row.get("segment_id")))
    admissible = [row for row in segments if row.get("admissible")]
    status = "no_recovery_deficit"
    if recovery_deficit > 0:
        status = "underfit_not_splittable_single_ridge_family"
        if len(admissible) >= recovery_deficit:
            status = "split_segments_cover_recovery_deficit"
        elif len(admissible) > 1:
            status = "partial_split_segments_found"
        elif split_diagnostics.get("split_gap_count", 0) > 0:
            status = "projection_gaps_found_but_segments_do_not_cover_deficit"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only severe-underfit support split test over the strongest local ridge family.",
        "recovery_deficit": recovery_deficit,
        "source_ridge_id": ridge.get("ridge_id"),
        "source_ridge_atom_count": len(ridge_atom_ids),
        "candidate_segment_count": len(segments),
        "admissible_segment_count": len(admissible),
        "status": status,
        "split_diagnostics": split_diagnostics,
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether a broad F1 ridge support can be split into multiple underfit bend supports.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--local-ridge-json", required=True, type=Path)
    parser.add_argument("--feature-label-summary-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-gap-mm", type=float, default=None)
    args = parser.parse_args()

    report = analyze_underfit_ridge_support_splits(
        decomposition=_load_json(args.decomposition_json) or {},
        local_ridge_payload=_load_json(args.local_ridge_json) or {},
        feature_label_summary=_load_json(args.feature_label_summary_json),
        min_gap_mm=args.min_gap_mm,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "recovery_deficit": report["recovery_deficit"],
                "candidate_segment_count": report["candidate_segment_count"],
                "admissible_segment_count": report["admissible_segment_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
