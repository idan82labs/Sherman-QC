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


def _line_row(
    *,
    line_id: str,
    source_kind: str,
    atom_ids: Sequence[str],
    atoms: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    valid_atom_ids = [str(value) for value in atom_ids if str(value) in atoms]
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in valid_atom_ids], dtype=np.float64)
    centroid, axis, span, residual, endpoints = _pca_line(points)
    row = {
        "line_id": line_id,
        "source_kind": source_kind,
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


def _candidate_interval_on_axis(line: Mapping[str, Any], axis: np.ndarray, origin: np.ndarray) -> Tuple[float, float]:
    endpoints = np.asarray(line.get("endpoints") or (), dtype=np.float64)
    if endpoints.shape != (2, 3):
        centroid = _vec(line.get("centroid") or (0, 0, 0))
        return (float(np.dot(centroid - origin, axis)), float(np.dot(centroid - origin, axis)))
    projections = (endpoints - origin) @ axis
    return float(np.min(projections)), float(np.max(projections))


def _pairwise_relation(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> Dict[str, Any]:
    lhs_axis = _unit(lhs.get("axis_direction") or (1, 0, 0))
    rhs_axis = _unit(rhs.get("axis_direction") or (1, 0, 0))
    lhs_centroid = _vec(lhs.get("centroid") or (0, 0, 0))
    rhs_centroid = _vec(rhs.get("centroid") or (0, 0, 0))
    axis_delta = _angle_deg(lhs_axis, rhs_axis)
    line_distance = _line_to_line_distance(lhs_centroid, lhs_axis, rhs_centroid, rhs_axis)
    centroid_distance = float(np.linalg.norm(lhs_centroid - rhs_centroid))
    lhs_interval = _candidate_interval_on_axis(lhs, lhs_axis, lhs_centroid)
    rhs_interval = _candidate_interval_on_axis(rhs, lhs_axis, lhs_centroid)
    span_overlap = _interval_overlap_fraction(lhs_interval, rhs_interval)
    lhs_atoms = {str(value) for value in lhs.get("atom_ids") or ()}
    rhs_atoms = {str(value) for value in rhs.get("atom_ids") or ()}
    atom_iou = len(lhs_atoms & rhs_atoms) / max(1, len(lhs_atoms | rhs_atoms))
    alias = atom_iou > 0.12 or (line_distance <= 4.0 and span_overlap >= 0.40 and axis_delta <= 25.0)
    separable = atom_iou == 0.0 and (span_overlap <= 0.10 or line_distance >= 8.0 or axis_delta >= 35.0)
    return {
        "lhs": lhs.get("line_id"),
        "rhs": rhs.get("line_id"),
        "atom_iou": round(float(atom_iou), 6),
        "centroid_distance_mm": round(float(centroid_distance), 6),
        "axis_delta_deg": round(float(axis_delta), 6),
        "line_distance_mm": round(float(line_distance), 6),
        "span_overlap": round(float(span_overlap), 6),
        "relation": "alias" if alias else "separable" if separable else "ambiguous",
    }


def _line_components(lines: Sequence[Mapping[str, Any]], relations: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    adjacency: Dict[str, set[str]] = {str(line.get("line_id")): set() for line in lines}
    for relation in relations:
        if relation.get("relation") != "alias":
            continue
        lhs = str(relation.get("lhs"))
        rhs = str(relation.get("rhs"))
        adjacency.setdefault(lhs, set()).add(rhs)
        adjacency.setdefault(rhs, set()).add(lhs)
    remaining = set(adjacency)
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values))
    return components


def _is_recovered_contact_line(line_id: str) -> bool:
    return str(line_id).startswith("RCB")


def _is_interior_line(line_id: str) -> bool:
    return str(line_id).startswith(("IBG", "IBH"))


def _interior_rows_for_pair(
    *,
    interior_gaps: Mapping[str, Any] | None,
    contact_recovery: Mapping[str, Any] | None,
    pair_key: Tuple[str, str],
    atoms: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not interior_gaps:
        return []
    contact_pair_by_source: Dict[str, Mapping[str, Any]] = {}
    for row in (contact_recovery or {}).get("candidates") or ():
        source_id = str(row.get("source_id") or "")
        best_pair = row.get("best_pair") or {}
        candidate_pair = tuple(sorted(str(value) for value in best_pair.get("flange_pair") or ()))
        if source_id and candidate_pair == pair_key:
            contact_pair_by_source[source_id] = best_pair
    rows: List[Dict[str, Any]] = []
    for candidate in interior_gaps.get("candidates") or ():
        source_id = str(candidate.get("candidate_id") or "")
        if source_id not in contact_pair_by_source:
            continue
        contact = contact_pair_by_source[source_id]
        rows.append(
            _line_row(
                line_id=source_id,
                source_kind="interior_gap_fragment",
                atom_ids=[str(value) for value in candidate.get("atom_ids") or ()],
                atoms=atoms,
                extra={
                    "incident_flange_ids": list(pair_key),
                    "contact_status": contact.get("status"),
                    "contact_score": contact.get("score"),
                    "contact_reason_codes": contact.get("reason_codes") or [],
                },
            )
        )
    for hypothesis in interior_gaps.get("merged_hypotheses") or ():
        source_id = str(hypothesis.get("hypothesis_id") or "")
        if source_id not in contact_pair_by_source:
            continue
        contact = contact_pair_by_source[source_id]
        rows.append(
            _line_row(
                line_id=source_id,
                source_kind="merged_interior_hypothesis",
                atom_ids=[str(value) for value in hypothesis.get("atom_ids") or ()],
                atoms=atoms,
                extra={
                    "incident_flange_ids": list(pair_key),
                    "contact_status": contact.get("status"),
                    "contact_score": contact.get("score"),
                    "contact_reason_codes": contact.get("reason_codes") or [],
                },
            )
        )
    return rows


def analyze_same_pair_family_consolidation(
    *,
    decomposition: Mapping[str, Any],
    arrangement: Mapping[str, Any],
    flange_pair: Tuple[str, str],
    interior_gaps: Mapping[str, Any] | None = None,
    contact_recovery: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    pair_key = tuple(sorted(str(value) for value in flange_pair))
    lines: List[Dict[str, Any]] = []
    for region in decomposition.get("owned_bend_regions") or ():
        region_pair = tuple(sorted(str(value) for value in region.get("incident_flange_ids") or ()))
        if region_pair != pair_key:
            continue
        lines.append(
            _line_row(
                line_id=str(region.get("bend_id")),
                source_kind="raw_owned_region",
                atom_ids=[str(value) for value in region.get("owned_atom_ids") or ()],
                atoms=atoms,
                extra={"incident_flange_ids": list(region_pair)},
            )
        )
    for candidate in arrangement.get("top_recovered_contact_birth_candidates") or ():
        candidate_pair = tuple(sorted(str(value) for value in candidate.get("flange_pair") or ()))
        if candidate_pair != pair_key:
            continue
        lines.append(
            _line_row(
                line_id=str(candidate.get("candidate_id")),
                source_kind="recovered_contact_birth",
                atom_ids=[str(value) for value in candidate.get("atom_ids") or ()],
                atoms=atoms,
                extra={
                    "incident_flange_ids": list(candidate_pair),
                    "interior_source_id": candidate.get("interior_source_id"),
                    "contact_recovery_score": candidate.get("contact_recovery_score"),
                    "duplicate_status": (candidate.get("duplicate_diagnostics") or {}).get("status"),
                },
            )
        )
    lines.extend(
        _interior_rows_for_pair(
            interior_gaps=interior_gaps,
            contact_recovery=contact_recovery,
            pair_key=pair_key,
            atoms=atoms,
        )
    )
    relations: List[Dict[str, Any]] = []
    for index, lhs in enumerate(lines):
        for rhs in lines[index + 1 :]:
            relations.append(_pairwise_relation(lhs, rhs))
    components = _line_components(lines, relations)
    relation_counts: Dict[str, int] = {}
    for relation in relations:
        key = str(relation.get("relation"))
        relation_counts[key] = relation_counts.get(key, 0) + 1
    status = "same_pair_family_empty"
    if lines:
        if any(any(_is_recovered_contact_line(line_id) for line_id in component) and len(component) > 1 for component in components):
            status = "recovered_contact_aliases_existing_family"
        elif any(any(_is_recovered_contact_line(line_id) for line_id in component) and len(component) == 1 for component in components):
            status = "recovered_contact_separate_family"
        elif any(any(_is_interior_line(line_id) for line_id in component) and len(component) == 1 for component in components):
            status = "interior_candidate_separate_family"
        else:
            status = "raw_same_pair_family_only"
    interior_separate_components = [
        component
        for component in components
        if any(_is_interior_line(line_id) for line_id in component) and not any(not _is_interior_line(line_id) for line_id in component)
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only same-pair line family consolidation over raw and recovered-contact supports.",
        "flange_pair": list(pair_key),
        "line_count": len(lines),
        "relation_counts": dict(sorted(relation_counts.items())),
        "component_count": len(components),
        "components": components,
        "interior_separate_component_count": len(interior_separate_components),
        "interior_separate_components": interior_separate_components,
        "status": status,
        "lines": lines,
        "relations": relations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze same-pair line family consolidation for recovered-contact candidates.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--arrangement-json", required=True, type=Path)
    parser.add_argument("--interior-gaps-json", type=Path)
    parser.add_argument("--contact-recovery-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--flange-pair", nargs=2, required=True)
    args = parser.parse_args()
    report = analyze_same_pair_family_consolidation(
        decomposition=_load_json(args.decomposition_json),
        arrangement=_load_json(args.arrangement_json),
        interior_gaps=_load_json(args.interior_gaps_json) if args.interior_gaps_json else None,
        contact_recovery=_load_json(args.contact_recovery_json) if args.contact_recovery_json else None,
        flange_pair=tuple(args.flange_pair),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "line_count": report["line_count"],
                "component_count": report["component_count"],
                "relation_counts": report["relation_counts"],
                "components": report["components"],
                "interior_separate_components": report["interior_separate_components"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
