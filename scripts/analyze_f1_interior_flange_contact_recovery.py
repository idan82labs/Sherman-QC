#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from itertools import combinations
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


def _normal_arc_cost(normal: Sequence[Any], flange_a: Mapping[str, Any], flange_b: Mapping[str, Any]) -> float:
    theta = _angle_deg(flange_a.get("normal") or (1, 0, 0), flange_b.get("normal") or (1, 0, 0), signless=False)
    lhs = _angle_deg(normal, flange_a.get("normal") or (1, 0, 0), signless=False)
    rhs = _angle_deg(normal, flange_b.get("normal") or (1, 0, 0), signless=False)
    return abs((lhs + rhs) - theta) / max(abs(theta), 1.0)


def _nearby_flange_atoms(
    *,
    candidate: Mapping[str, Any],
    atoms: Mapping[str, Mapping[str, Any]],
    atom_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    line_radius_mm: float,
    span_pad_mm: float,
) -> Dict[str, List[Dict[str, float]]]:
    centroid = _vec(candidate.get("centroid") or (0.0, 0.0, 0.0))
    axis = _unit(candidate.get("axis_direction") or (1.0, 0.0, 0.0))
    span = max(float(candidate.get("span_mm") or 0.0), 1.0)
    by_flange: Dict[str, List[Dict[str, float]]] = {}
    for atom_id, atom in atoms.items():
        label = atom_labels.get(atom_id)
        if label_types.get(label) != "flange":
            continue
        point = _vec(atom.get("centroid") or (0.0, 0.0, 0.0))
        projection = float(np.dot(point - centroid, axis))
        if abs(projection) > span * 0.75 + span_pad_mm:
            continue
        closest = centroid + projection * axis
        line_distance = float(np.linalg.norm(point - closest))
        if line_distance > line_radius_mm:
            continue
        by_flange.setdefault(str(label), []).append(
            {
                "atom_id": atom_id,
                "line_distance_mm": line_distance,
                "axis_projection_mm": projection,
            }
        )
    return by_flange


def _score_pair(
    *,
    candidate: Mapping[str, Any],
    pair: Tuple[str, str],
    nearby: Mapping[str, Sequence[Mapping[str, float]]],
    atoms: Mapping[str, Mapping[str, Any]],
    flanges: Mapping[str, Mapping[str, Any]],
    min_side_atoms: int,
) -> Dict[str, Any]:
    flange_a = flanges[pair[0]]
    flange_b = flanges[pair[1]]
    expected_axis = _unit(np.cross(_unit(flange_a.get("normal") or (1, 0, 0)), _unit(flange_b.get("normal") or (0, 1, 0))))
    axis_delta = _angle_deg(candidate.get("axis_direction") or (1, 0, 0), expected_axis)
    atom_ids = [str(value) for value in candidate.get("atom_ids") or () if str(value) in atoms]
    normal_costs = [_normal_arc_cost(atoms[atom_id].get("normal") or (1, 0, 0), flange_a, flange_b) for atom_id in atom_ids]
    mean_normal_arc = float(sum(normal_costs) / max(1, len(normal_costs)))
    side_a = list(nearby.get(pair[0]) or ())
    side_b = list(nearby.get(pair[1]) or ())
    count_a = len(side_a)
    count_b = len(side_b)
    side_balance = min(count_a, count_b) / max(1, max(count_a, count_b))
    median_distance_a = float(np.median([row["line_distance_mm"] for row in side_a])) if side_a else None
    median_distance_b = float(np.median([row["line_distance_mm"] for row in side_b])) if side_b else None
    reasons: List[str] = []
    if count_a < min_side_atoms:
        reasons.append(f"{pair[0]}_nearby_flange_atoms_below_floor")
    if count_b < min_side_atoms:
        reasons.append(f"{pair[1]}_nearby_flange_atoms_below_floor")
    if axis_delta > 25.0:
        reasons.append("axis_mismatch")
    if mean_normal_arc > 0.75:
        reasons.append("weak_normal_arc")
    if side_balance < 0.12:
        reasons.append("weak_two_sided_balance")
    status = "recovered_contact_validated" if not reasons else "recovered_contact_rejected"
    score = (
        max(0.0, 1.0 - axis_delta / 35.0)
        + max(0.0, 1.0 - mean_normal_arc / 0.75)
        + min(1.0, count_a / 40.0)
        + min(1.0, count_b / 40.0)
        + side_balance
    )
    return {
        "flange_pair": list(pair),
        "status": status,
        "score": round(float(score), 6),
        "axis_delta_deg": round(float(axis_delta), 6),
        "mean_normal_arc_cost": round(float(mean_normal_arc), 6),
        "nearby_counts": {pair[0]: count_a, pair[1]: count_b},
        "side_balance": round(float(side_balance), 6),
        "median_line_distance_mm": {
            pair[0]: round(float(median_distance_a), 6) if median_distance_a is not None else None,
            pair[1]: round(float(median_distance_b), 6) if median_distance_b is not None else None,
        },
        "reason_codes": reasons or ["recovered_two_sided_contact"],
    }


def _candidate_rows(interior_gaps: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in interior_gaps.get("candidates") or ():
        if not candidate.get("admissible"):
            continue
        row = dict(candidate)
        row["source_kind"] = "interior_fragment"
        row["source_id"] = candidate.get("candidate_id")
        rows.append(row)
    for hypothesis in interior_gaps.get("merged_hypotheses") or ():
        row = dict(hypothesis)
        row["source_kind"] = "merged_interior_hypothesis"
        row["source_id"] = hypothesis.get("hypothesis_id")
        rows.append(row)
    return rows


def analyze_interior_flange_contact_recovery(
    *,
    decomposition: Mapping[str, Any],
    interior_gaps: Mapping[str, Any],
    line_radius_mm: float = 24.0,
    span_pad_mm: float = 15.0,
    min_side_atoms: int = 3,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    flanges = _flange_lookup(decomposition)
    atom_labels, label_types = _assignment(decomposition)
    candidates = _candidate_rows(interior_gaps)
    candidate_reports: List[Dict[str, Any]] = []
    validated_pairs: List[Dict[str, Any]] = []
    for candidate in candidates:
        nearby = _nearby_flange_atoms(
            candidate=candidate,
            atoms=atoms,
            atom_labels=atom_labels,
            label_types=label_types,
            line_radius_mm=line_radius_mm,
            span_pad_mm=span_pad_mm,
        )
        nearby_summary = {
            flange_id: {
                "atom_count": len(rows),
                "median_line_distance_mm": round(float(np.median([row["line_distance_mm"] for row in rows])), 6),
                "min_line_distance_mm": round(float(min(row["line_distance_mm"] for row in rows)), 6),
            }
            for flange_id, rows in sorted(nearby.items(), key=lambda item: (-len(item[1]), item[0]))
        }
        pair_reports = [
            _score_pair(
                candidate=candidate,
                pair=tuple(sorted(pair)),
                nearby=nearby,
                atoms=atoms,
                flanges=flanges,
                min_side_atoms=min_side_atoms,
            )
            for pair in combinations([flange_id for flange_id in nearby if flange_id in flanges], 2)
        ]
        pair_reports.sort(key=lambda row: (row["status"] != "recovered_contact_validated", -float(row["score"]), row["flange_pair"]))
        for pair_report in pair_reports:
            if pair_report["status"] == "recovered_contact_validated":
                validated_pairs.append(
                    {
                        "source_id": candidate.get("source_id"),
                        "source_kind": candidate.get("source_kind"),
                        **pair_report,
                    }
                )
        candidate_reports.append(
            {
                "source_id": candidate.get("source_id"),
                "source_kind": candidate.get("source_kind"),
                "atom_count": candidate.get("atom_count"),
                "span_mm": candidate.get("span_mm"),
                "median_line_residual_mm": candidate.get("median_line_residual_mm"),
                "nearby_flange_summary": nearby_summary,
                "pair_reports": pair_reports[:12],
                "best_pair": pair_reports[0] if pair_reports else None,
            }
        )
    status = "fragment_contact_recovered" if validated_pairs else "no_validated_interior_contact"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only local flange/contact recovery around interior missed-bend support.",
        "line_radius_mm": line_radius_mm,
        "span_pad_mm": span_pad_mm,
        "min_side_atoms": min_side_atoms,
        "candidate_count": len(candidate_reports),
        "validated_pair_count": len(validated_pairs),
        "status": status,
        "validated_pairs": sorted(validated_pairs, key=lambda row: (-float(row["score"]), row["source_id"], row["flange_pair"])),
        "candidates": candidate_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover local flange contacts around interior missed-bend candidates.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--interior-gaps-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--line-radius-mm", type=float, default=24.0)
    parser.add_argument("--span-pad-mm", type=float, default=15.0)
    parser.add_argument("--min-side-atoms", type=int, default=3)
    args = parser.parse_args()
    report = analyze_interior_flange_contact_recovery(
        decomposition=_load_json(args.decomposition_json),
        interior_gaps=_load_json(args.interior_gaps_json),
        line_radius_mm=float(args.line_radius_mm),
        span_pad_mm=float(args.span_pad_mm),
        min_side_atoms=int(args.min_side_atoms),
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "validated_pair_count": report["validated_pair_count"],
                "validated_pairs": [
                    {
                        "source_id": row["source_id"],
                        "flange_pair": row["flange_pair"],
                        "score": row["score"],
                    }
                    for row in report["validated_pairs"][:8]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
