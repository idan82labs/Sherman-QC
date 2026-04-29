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
) -> Dict[str, Any]:
    points = np.asarray([_vec(atoms[atom_id].get("centroid") or (0, 0, 0)) for atom_id in atom_ids], dtype=np.float64)
    centroid, axis, span, residual = _pca_line(points)
    flange_a = flanges[pair[0]]
    flange_b = flanges[pair[1]]
    expected_axis = _unit(np.cross(_unit(flange_a.get("normal") or (1, 0, 0)), _unit(flange_b.get("normal") or (0, 1, 0))))
    axis_delta = _angle_deg(axis, expected_axis)
    touch_counts = _touching_counts(
        atom_ids,
        pair,
        adjacency=adjacency,
        atom_labels=atom_labels,
        label_types=label_types,
    )
    present_contacts = sum(1 for value in touch_counts.values() if value > 0)
    normal_costs = [_normal_arc_cost(atoms[atom_id].get("normal") or (1, 0, 0), flange_a, flange_b) for atom_id in atom_ids]
    mean_normal_cost = float(sum(normal_costs) / max(1, len(normal_costs)))
    support_mass = float(sum(float((atoms.get(atom_id) or {}).get("weight") or 0.0) for atom_id in atom_ids))
    is_locked = pair == locked_pair
    is_chord = pair == tuple(sorted((locked_pair[0], "F1"))) or pair == tuple(sorted(("F1", "F17")))
    # Diagnostic score: higher is better, not calibrated as final energy.
    score = (
        min(1.0, len(atom_ids) / 35.0)
        + min(1.0, span / 20.0)
        + max(0.0, 1.0 - residual / 6.0)
        + max(0.0, 1.0 - axis_delta / 35.0)
        + max(0.0, 1.0 - mean_normal_cost / 0.45)
        + present_contacts / 2.0
    )
    penalty = 0.0
    reasons: List[str] = []
    if len(atom_ids) < 6:
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
        "flange_pair": list(pair),
        "atom_ids": sorted(atom_ids),
        "atom_count": len(atom_ids),
        "support_mass": round(float(support_mass), 6),
        "centroid": [round(float(value), 6) for value in centroid.tolist()],
        "axis_direction": [round(float(value), 6) for value in axis.tolist()],
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
        "admissible": score - penalty >= 3.25 and span >= 10.0 and len(atom_ids) >= 6,
    }


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
    flange_ids: Sequence[str] = ("F1", "F11", "F14", "F17"),
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
    candidates.sort(key=lambda item: (not item["locked_accepted"], -float(item["candidate_score"]), item["candidate_id"]))
    locked = [item for item in candidates if item.get("locked_accepted")]
    nonlocked = [item for item in candidates if not item.get("locked_accepted") and item.get("admissible")]

    subsets: List[Dict[str, Any]] = []
    locked_base = locked[:1]
    max_size = min(3, len(nonlocked))
    for size in range(0, max_size + 1):
        for selected in combinations(nonlocked, size):
            subset = [*locked_base, *selected]
            row = _subset_score(subset)
            row["hypothesis"] = f"H{size + 1}_locked_plus_{size}_new"
            subsets.append(row)
    subsets.sort(key=lambda item: (-float(item["subset_score"]), int(item["new_bend_count"]), item["candidate_ids"]))
    best = subsets[0] if subsets else None
    status = "no_arrangement_candidate"
    if best:
        if int(best.get("new_bend_count") or 0) == 2 and not best.get("conflicts"):
            status = "two_bend_arrangement_candidate"
        elif int(best.get("new_bend_count") or 0) == 2:
            status = "two_bend_arrangement_conflicted"
        elif int(best.get("new_bend_count") or 0) < 2:
            status = "underfit_arrangement"
        else:
            status = "overfit_arrangement"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "purpose": "Diagnostic-only junction-aware bend-line arrangement over the 49024000 connected ridge complex.",
        "locked_accepted_pair": list(locked_pair),
        "input_flange_ids": [flange_id for flange_id in flange_ids if flange_id in flanges],
        "ridge_atom_count": len(ridge_atom_ids),
        "candidate_count": len(candidates),
        "admissible_nonlocked_candidate_count": len(nonlocked),
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
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--flange-ids", nargs="+", default=["F1", "F11", "F14", "F17"])
    args = parser.parse_args()

    report = solve_junction_bend_arrangement(
        decomposition=_load_json(args.decomposition_json),
        feature_labels=_load_json(args.feature_labels_json),
        local_ridge_payload=_load_json(args.local_ridge_json),
        flange_ids=args.flange_ids,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "best_hypothesis": report.get("best_hypothesis", {}).get("hypothesis"),
                "best_candidate_ids": report.get("best_hypothesis", {}).get("candidate_ids"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
