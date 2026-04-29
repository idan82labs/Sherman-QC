#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


def _angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    dot = abs(float(np.dot(_unit(lhs), _unit(rhs))))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in (decomposition.get("atom_graph") or {}).get("atoms") or ()
        if atom.get("atom_id")
    }


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


def _line_row(
    *,
    line_id: str,
    source_kind: str,
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    flange_pair: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    valid_atom_ids = [str(atom_id) for atom_id in atom_ids if str(atom_id) in atoms]
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in valid_atom_ids], dtype=np.float64)
    centroid, axis, span, residual, endpoints = _pca_line(points)
    row = {
        "line_id": line_id,
        "source_kind": source_kind,
        "flange_pair": list(tuple(sorted(str(value) for value in flange_pair))),
        "atom_ids": valid_atom_ids,
        "atom_count": len(valid_atom_ids),
        "centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
        "span_mm": round(float(span), 6),
        "median_line_residual_mm": round(float(residual), 6),
        "endpoints": [[round(float(value), 6) for value in endpoint] for endpoint in endpoints],
    }
    if extra:
        row.update(extra)
    return row


def _line_to_line_distance(point_a: np.ndarray, axis_a: np.ndarray, point_b: np.ndarray, axis_b: np.ndarray) -> float:
    cross = np.cross(axis_a, axis_b)
    norm = float(np.linalg.norm(cross))
    delta = point_b - point_a
    if norm <= 1e-9:
        return float(np.linalg.norm(delta - float(np.dot(delta, axis_a)) * axis_a))
    return abs(float(np.dot(delta, cross / norm)))


def _interval_on_axis(line: Mapping[str, Any], axis: np.ndarray, origin: np.ndarray) -> Tuple[float, float]:
    endpoints = np.asarray(line.get("endpoints") or (), dtype=np.float64)
    if endpoints.shape != (2, 3):
        centroid = _vec(line.get("centroid") or (0, 0, 0))
        value = float(np.dot(centroid - origin, axis))
        return value, value
    projections = (endpoints - origin) @ axis
    return float(np.min(projections)), float(np.max(projections))


def _interval_overlap_fraction(lhs: Tuple[float, float], rhs: Tuple[float, float]) -> float:
    lhs_span = max(0.0, lhs[1] - lhs[0])
    rhs_span = max(0.0, rhs[1] - rhs[0])
    if lhs_span <= 1e-9 or rhs_span <= 1e-9:
        return 0.0
    overlap = max(0.0, min(lhs[1], rhs[1]) - max(lhs[0], rhs[0]))
    return overlap / max(1e-9, min(lhs_span, rhs_span))


def _relation(candidate: Mapping[str, Any], raw: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_axis = _unit(candidate.get("axis_direction") or (1, 0, 0))
    raw_axis = _unit(raw.get("axis_direction") or (1, 0, 0))
    candidate_centroid = _vec(candidate.get("centroid") or (0, 0, 0))
    raw_centroid = _vec(raw.get("centroid") or (0, 0, 0))
    axis_delta = _angle_deg(candidate_axis, raw_axis)
    line_distance = _line_to_line_distance(candidate_centroid, candidate_axis, raw_centroid, raw_axis)
    centroid_distance = float(np.linalg.norm(candidate_centroid - raw_centroid))
    candidate_interval = _interval_on_axis(candidate, candidate_axis, candidate_centroid)
    raw_interval = _interval_on_axis(raw, candidate_axis, candidate_centroid)
    span_overlap = _interval_overlap_fraction(candidate_interval, raw_interval)
    candidate_atoms = {str(value) for value in candidate.get("atom_ids") or ()}
    raw_atoms = {str(value) for value in raw.get("atom_ids") or ()}
    atom_iou = len(candidate_atoms & raw_atoms) / max(1, len(candidate_atoms | raw_atoms))
    alias = atom_iou > 0.12 or (line_distance <= 4.0 and span_overlap >= 0.40 and axis_delta <= 25.0)
    separable = atom_iou == 0.0 and (span_overlap <= 0.10 or line_distance >= 8.0 or axis_delta >= 35.0)
    relation = "alias" if alias else "separable" if separable else "ambiguous"
    return {
        "candidate_id": candidate.get("line_id"),
        "raw_bend_id": raw.get("line_id"),
        "relation": relation,
        "atom_iou": round(float(atom_iou), 6),
        "centroid_distance_mm": round(float(centroid_distance), 6),
        "axis_delta_deg": round(float(axis_delta), 6),
        "line_distance_mm": round(float(line_distance), 6),
        "span_overlap": round(float(span_overlap), 6),
    }


def _raw_lines_by_pair(decomposition: Mapping[str, Any], atoms: Mapping[str, Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for region in decomposition.get("owned_bend_regions") or ():
        pair = tuple(sorted(str(value) for value in region.get("incident_flange_ids") or ()))
        if len(pair) != 2:
            continue
        by_pair.setdefault(pair, []).append(
            _line_row(
                line_id=str(region.get("bend_id")),
                source_kind="raw_owned_region",
                atom_ids=[str(value) for value in region.get("owned_atom_ids") or ()],
                atoms=atoms,
                flange_pair=pair,
            )
        )
    return by_pair


def _source_rows(interior_gaps: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for candidate in interior_gaps.get("candidates") or ():
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = candidate
    for hypothesis in interior_gaps.get("merged_hypotheses") or ():
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        if hypothesis_id:
            rows[hypothesis_id] = hypothesis
    return rows


def _candidate_lines(
    *,
    interior_gaps: Mapping[str, Any],
    contact_recovery: Mapping[str, Any],
    atoms: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sources = _source_rows(interior_gaps)
    rows: List[Dict[str, Any]] = []
    for contact_row in contact_recovery.get("candidates") or ():
        source_id = str(contact_row.get("source_id") or "")
        source = sources.get(source_id)
        best_pair = contact_row.get("best_pair") or {}
        pair = tuple(sorted(str(value) for value in best_pair.get("flange_pair") or ()))
        if source is None or len(pair) != 2:
            continue
        rows.append(
            _line_row(
                line_id=source_id,
                source_kind=str(source.get("source_kind") or ("merged_interior_hypothesis" if source_id.startswith("IBH") else "interior_gap_fragment")),
                atom_ids=[str(value) for value in source.get("atom_ids") or ()],
                atoms=atoms,
                flange_pair=pair,
                extra={
                    "contact_status": best_pair.get("status"),
                    "contact_score": best_pair.get("score"),
                    "contact_reason_codes": best_pair.get("reason_codes") or [],
                    "axis_delta_deg": best_pair.get("axis_delta_deg"),
                    "mean_normal_arc_cost": best_pair.get("mean_normal_arc_cost"),
                    "nearby_counts": best_pair.get("nearby_counts") or {},
                    "side_balance": best_pair.get("side_balance"),
                },
            )
        )
    return rows


def _evaluate_candidate(candidate: Mapping[str, Any], raw_lines: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    relations = [_relation(candidate, raw) for raw in raw_lines]
    relation_order = {"alias": 0, "ambiguous": 1, "separable": 2}
    relations.sort(
        key=lambda row: (
            relation_order.get(str(row.get("relation")), 9),
            float(row.get("line_distance_mm") or 999.0),
            -float(row.get("span_overlap") or 0.0),
            str(row.get("raw_bend_id")),
        )
    )
    relation_counts: Dict[str, int] = {}
    for row in relations:
        key = str(row.get("relation"))
        relation_counts[key] = relation_counts.get(key, 0) + 1
    blockers: List[str] = []
    if candidate.get("contact_status") != "recovered_contact_validated":
        blockers.append("contact_not_validated")
    if int(candidate.get("atom_count") or 0) < 4:
        blockers.append("support_atom_count_below_floor")
    if any(row.get("relation") == "alias" for row in relations):
        blockers.append("same_pair_alias_to_raw_family")
    elif any(row.get("relation") == "ambiguous" for row in relations):
        blockers.append("same_pair_ambiguous_with_raw_family")
    if not raw_lines:
        blockers.append("no_raw_same_pair_family_for_comparison")
    status = "separable_missing_bend_candidate" if not blockers else "blocked_by_" + "_and_".join(blockers[:2])
    return {
        "candidate_id": candidate.get("line_id"),
        "flange_pair": candidate.get("flange_pair"),
        "source_kind": candidate.get("source_kind"),
        "atom_count": candidate.get("atom_count"),
        "centroid": candidate.get("centroid"),
        "axis_direction": candidate.get("axis_direction"),
        "span_mm": candidate.get("span_mm"),
        "median_line_residual_mm": candidate.get("median_line_residual_mm"),
        "contact_status": candidate.get("contact_status"),
        "contact_score": candidate.get("contact_score"),
        "contact_reason_codes": candidate.get("contact_reason_codes") or [],
        "family_exclusion_status": status,
        "family_exclusion_passed": not blockers,
        "blockers": blockers,
        "same_pair_raw_region_count": len(raw_lines),
        "same_pair_relation_counts": dict(sorted(relation_counts.items())),
        "nearest_same_pair_relations": relations[:6],
    }


def analyze_missing_bend_family_exclusion(
    *,
    decomposition: Mapping[str, Any],
    interior_gaps: Mapping[str, Any],
    contact_recovery: Mapping[str, Any],
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    raw_by_pair = _raw_lines_by_pair(decomposition, atoms)
    candidates = _candidate_lines(interior_gaps=interior_gaps, contact_recovery=contact_recovery, atoms=atoms)
    evaluations = [
        _evaluate_candidate(candidate, raw_by_pair.get(tuple(candidate.get("flange_pair") or ()), ()))
        for candidate in candidates
    ]
    evaluations.sort(
        key=lambda row: (
            not bool(row.get("family_exclusion_passed")),
            -float(row.get("contact_score") or 0.0),
            -int(row.get("atom_count") or 0),
            str(row.get("candidate_id")),
        )
    )
    passed = [row for row in evaluations if row.get("family_exclusion_passed")]
    blocked_counts: Dict[str, int] = {}
    for row in evaluations:
        for blocker in row.get("blockers") or ():
            blocked_counts[str(blocker)] = blocked_counts.get(str(blocker), 0) + 1
    status = "separable_missing_bend_candidates_found" if passed else "no_separable_missing_bend_candidate"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only missing-bend gate that excludes candidates already covered by raw same-pair line families.",
        "status": status,
        "candidate_count": len(evaluations),
        "family_exclusion_pass_count": len(passed),
        "blocked_candidate_count": len(evaluations) - len(passed),
        "blocked_reason_counts": dict(sorted(blocked_counts.items())),
        "candidate_ids_passing_family_exclusion": [str(row.get("candidate_id")) for row in passed],
        "candidates": evaluations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate missed-bend candidates against already-covered same-pair raw families.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--interior-gaps-json", required=True, type=Path)
    parser.add_argument("--contact-recovery-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_missing_bend_family_exclusion(
        decomposition=_load_json(args.decomposition_json),
        interior_gaps=_load_json(args.interior_gaps_json),
        contact_recovery=_load_json(args.contact_recovery_json),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "family_exclusion_pass_count": report["family_exclusion_pass_count"],
                "blocked_reason_counts": report["blocked_reason_counts"],
                "candidate_ids_passing_family_exclusion": report["candidate_ids_passing_family_exclusion"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
