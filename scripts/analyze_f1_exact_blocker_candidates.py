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


def _distance(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    return float(np.linalg.norm(_vec(lhs) - _vec(rhs)))


def _axis_angle_deg(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    left = _unit(lhs)
    right = _unit(rhs)
    # Bend axes are signless.
    dot = abs(float(np.dot(left, right)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _pair(region: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(str(value) for value in region.get("incident_flange_ids") or ()))


def _pair_from_values(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(sorted(str(value) for value in values if str(value)))


def _region_atoms(region: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in region.get("owned_atom_ids") or ()}


def _atom_lookup(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(atom["atom_id"]): atom for atom in (payload.get("atom_graph") or {}).get("atoms") or ()}


def _support_bendness(region: Mapping[str, Any], atoms: Mapping[str, Mapping[str, Any]]) -> float:
    atom_ids = list(_region_atoms(region))
    if not atom_ids:
        return 0.0
    values = [
        min(1.0, float((atoms.get(atom_id) or {}).get("curvature_proxy") or 0.0) * 20.0)
        for atom_id in atom_ids
    ]
    return float(sum(values) / max(1, len(values)))


def _best_baseline_match(
    region: Mapping[str, Any],
    baseline_regions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    atoms = _region_atoms(region)
    best: Optional[Dict[str, Any]] = None
    for baseline in baseline_regions:
        baseline_atoms = _region_atoms(baseline)
        intersection = len(atoms & baseline_atoms)
        union = len(atoms | baseline_atoms)
        iou = float(intersection / union) if union else 0.0
        centroid_distance = _distance(region.get("support_centroid") or (0, 0, 0), baseline.get("support_centroid") or (0, 0, 0))
        axis_delta = _axis_angle_deg(region.get("axis_direction") or (1, 0, 0), baseline.get("axis_direction") or (1, 0, 0))
        candidate = {
            "baseline_bend_id": baseline.get("bend_id"),
            "baseline_incident_flange_ids": list(_pair(baseline)),
            "atom_intersection": intersection,
            "atom_iou": round(iou, 6),
            "centroid_distance_mm": round(centroid_distance, 6),
            "axis_delta_deg": round(axis_delta, 6),
            "same_incident_pair": _pair(region) == _pair(baseline),
        }
        key = (
            float(candidate["atom_iou"]),
            -float(candidate["centroid_distance_mm"]),
            bool(candidate["same_incident_pair"]),
        )
        if best is None:
            best = candidate
            best_key = key
            continue
        if key > best_key:
            best = candidate
            best_key = key
    return best or {
        "baseline_bend_id": None,
        "baseline_incident_flange_ids": [],
        "atom_intersection": 0,
        "atom_iou": 0.0,
        "centroid_distance_mm": None,
        "axis_delta_deg": None,
        "same_incident_pair": False,
    }


def _nearest_baseline_distance(
    region: Mapping[str, Any],
    baseline_regions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not baseline_regions:
        return {"baseline_bend_id": None, "centroid_distance_mm": None}
    nearest = min(
        baseline_regions,
        key=lambda baseline: _distance(region.get("support_centroid") or (0, 0, 0), baseline.get("support_centroid") or (0, 0, 0)),
    )
    return {
        "baseline_bend_id": nearest.get("bend_id"),
        "centroid_distance_mm": round(
            _distance(region.get("support_centroid") or (0, 0, 0), nearest.get("support_centroid") or (0, 0, 0)),
            6,
        ),
    }


def _score_candidate(
    *,
    region: Mapping[str, Any],
    baseline_regions: Sequence[Mapping[str, Any]],
    atoms: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    atom_count = len(_region_atoms(region))
    support_mass = float(region.get("support_mass") or 0.0)
    best_match = _best_baseline_match(region, baseline_regions)
    nearest = _nearest_baseline_distance(region, baseline_regions)
    bendness = _support_bendness(region, atoms)
    angle = float(region.get("angle_deg") or 0.0)
    angle_score = max(0.0, 1.0 - abs(angle - 90.0) / 25.0)
    overlap_score = max(0.0, 1.0 - float(best_match.get("atom_iou") or 0.0))
    support_score = min(1.0, math.log1p(atom_count) / math.log(25.0))
    mass_score = min(1.0, support_mass / 2500.0)
    nearest_distance = float(nearest.get("centroid_distance_mm") or 0.0)
    separation_score = min(1.0, nearest_distance / 35.0)
    duplicate_penalty = 0.0
    if best_match.get("same_incident_pair"):
        duplicate_penalty += 0.25
    if float(best_match.get("atom_iou") or 0.0) >= 0.35:
        duplicate_penalty += 0.45
    if nearest_distance < 8.0:
        duplicate_penalty += 0.25
    score = (
        0.28 * overlap_score
        + 0.22 * support_score
        + 0.16 * mass_score
        + 0.14 * separation_score
        + 0.12 * angle_score
        + 0.08 * bendness
        - duplicate_penalty
    )
    reasons: List[str] = []
    if float(best_match.get("atom_iou") or 0.0) >= 0.35:
        reasons.append("likely_duplicate_overlap")
    if best_match.get("same_incident_pair"):
        reasons.append("same_incident_pair_as_baseline")
    if nearest_distance < 8.0:
        reasons.append("centroid_near_existing_bend")
    if atom_count < 5:
        reasons.append("tiny_support")
    if abs(angle - 90.0) > 18.0:
        reasons.append("angle_far_from_near90_family")
    if not reasons:
        reasons.append("non_duplicate_candidate")
    return {
        "bend_id": region.get("bend_id"),
        "incident_flange_ids": list(_pair(region)),
        "atom_count": atom_count,
        "support_mass": round(support_mass, 6),
        "angle_deg": round(angle, 6),
        "support_centroid": region.get("support_centroid"),
        "span_endpoints": region.get("span_endpoints"),
        "axis_direction": region.get("axis_direction"),
        "best_baseline_match": best_match,
        "nearest_baseline": nearest,
        "support_bendness": round(bendness, 6),
        "component_scores": {
            "overlap_score": round(overlap_score, 6),
            "support_score": round(support_score, 6),
            "mass_score": round(mass_score, 6),
            "separation_score": round(separation_score, 6),
            "angle_score": round(angle_score, 6),
            "duplicate_penalty": round(duplicate_penalty, 6),
        },
        "candidate_score": round(float(score), 6),
        "reason_codes": reasons,
    }


def build_candidate_report(
    *,
    baseline_payload: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
    baseline_label: str,
    overcomplete_label: str,
) -> Dict[str, Any]:
    baseline_regions = list(baseline_payload.get("owned_bend_regions") or ())
    overcomplete_regions = list(overcomplete_payload.get("owned_bend_regions") or ())
    atoms = _atom_lookup(overcomplete_payload)
    candidates = [
        _score_candidate(region=region, baseline_regions=baseline_regions, atoms=atoms)
        for region in overcomplete_regions
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["candidate_score"]),
            float((item["best_baseline_match"] or {}).get("atom_iou") or 0.0),
            -int(item["atom_count"]),
            str(item["bend_id"]),
        )
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {
            "label": baseline_label,
            "exact_bend_count": baseline_payload.get("exact_bend_count"),
            "bend_count_range": baseline_payload.get("bend_count_range"),
            "candidate_solution_counts": baseline_payload.get("candidate_solution_counts"),
            "candidate_solution_energies": baseline_payload.get("candidate_solution_energies"),
            "owned_bend_region_count": len(baseline_regions),
        },
        "overcomplete": {
            "label": overcomplete_label,
            "exact_bend_count": overcomplete_payload.get("exact_bend_count"),
            "bend_count_range": overcomplete_payload.get("bend_count_range"),
            "candidate_solution_counts": overcomplete_payload.get("candidate_solution_counts"),
            "candidate_solution_energies": overcomplete_payload.get("candidate_solution_energies"),
            "owned_bend_region_count": len(overcomplete_regions),
        },
        "interpretation": {
            "purpose": "Diagnostic-only ranking of overcomplete subspan owned regions against the stable baseline. No candidate is promoted automatically.",
            "score_direction": "higher means more plausible as an extra non-duplicate bend support",
            "promotion_status": "diagnostic_only",
        },
        "ranked_candidates": candidates,
    }


def _span_interval(region: Mapping[str, Any], axis: Sequence[Any]) -> Optional[Tuple[float, float]]:
    endpoints = region.get("span_endpoints") or ()
    if len(endpoints) < 2:
        return None
    direction = _unit(axis)
    values = [float(np.dot(_vec(point), direction)) for point in endpoints[:2]]
    return (min(values), max(values))


def _interval_overlap(lhs: Optional[Tuple[float, float]], rhs: Optional[Tuple[float, float]]) -> float:
    if lhs is None or rhs is None:
        return 0.0
    left = max(lhs[0], rhs[0])
    right = min(lhs[1], rhs[1])
    overlap = max(0.0, right - left)
    lhs_len = max(0.0, lhs[1] - lhs[0])
    rhs_len = max(0.0, rhs[1] - rhs[0])
    denom = min(lhs_len, rhs_len)
    if denom <= 1e-9:
        return 0.0
    return float(overlap / denom)


def _candidate_conflict(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> Dict[str, Any]:
    same_pair = tuple(lhs.get("incident_flange_ids") or ()) == tuple(rhs.get("incident_flange_ids") or ())
    centroid_distance = _distance(lhs.get("support_centroid") or (0, 0, 0), rhs.get("support_centroid") or (0, 0, 0))
    axis_delta = _axis_angle_deg(lhs.get("axis_direction") or (1, 0, 0), rhs.get("axis_direction") or (1, 0, 0))
    overlap = _interval_overlap(
        _span_interval(lhs, lhs.get("axis_direction") or (1, 0, 0)),
        _span_interval(rhs, rhs.get("axis_direction") or (1, 0, 0)),
    )
    atom_iou = 0.0
    lhs_atoms = set(str(value) for value in lhs.get("owned_atom_ids") or ())
    rhs_atoms = set(str(value) for value in rhs.get("owned_atom_ids") or ())
    if lhs_atoms or rhs_atoms:
        atom_iou = len(lhs_atoms & rhs_atoms) / max(1, len(lhs_atoms | rhs_atoms))
    conflict_score = 0.0
    reasons: List[str] = []
    if same_pair:
        conflict_score += 0.35
        reasons.append("same_incident_pair")
    if centroid_distance < 18.0:
        conflict_score += 0.30
        reasons.append("near_centroid")
    if axis_delta < 12.0:
        conflict_score += 0.15
        reasons.append("parallel_axis")
    if overlap >= 0.35:
        conflict_score += 0.25
        reasons.append("span_overlap")
    if atom_iou >= 0.15:
        conflict_score += 0.45
        reasons.append("atom_overlap")
    return {
        "lhs": lhs.get("bend_id"),
        "rhs": rhs.get("bend_id"),
        "conflict_score": round(min(1.0, conflict_score), 6),
        "same_incident_pair": same_pair,
        "centroid_distance_mm": round(float(centroid_distance), 6),
        "axis_delta_deg": round(float(axis_delta), 6),
        "span_overlap": round(float(overlap), 6),
        "atom_iou": round(float(atom_iou), 6),
        "reason_codes": reasons,
    }


def _selection_reject_reasons(candidate: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    score = float(candidate.get("candidate_score") or 0.0)
    atom_count = int(candidate.get("atom_count") or 0)
    candidate_reasons = set(str(value) for value in candidate.get("reason_codes") or ())
    nearest = dict(candidate.get("nearest_baseline") or {})
    nearest_distance = float(nearest.get("centroid_distance_mm") or 0.0)
    if score < 0.72:
        reasons.append("selection_score_below_floor")
    if atom_count < 5:
        reasons.append("support_atom_count_below_floor")
    if "likely_duplicate_overlap" in candidate_reasons:
        reasons.append("overlaps_existing_baseline_support")
    if "centroid_near_existing_bend" in candidate_reasons and nearest_distance < 8.0:
        reasons.append("too_close_to_existing_baseline_bend")
    if "angle_far_from_near90_family" in candidate_reasons:
        reasons.append("angle_outside_repeated_near90_family")
    return reasons


def build_local_selection_report(
    *,
    baseline_payload: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    score_gap_floor: float = 0.08,
    top_plausible_limit: int = 3,
) -> Dict[str, Any]:
    """Evaluate whether one overcomplete candidate is safe enough for exact +1 promotion.

    This is intentionally conservative. It does not mutate the decomposition and it does
    not claim exact correctness. It only identifies whether a formal single-birth move
    has a uniquely dominant candidate worth implementing next.
    """
    overcomplete_by_id = {str(region.get("bend_id")): region for region in overcomplete_payload.get("owned_bend_regions") or ()}
    scored: List[Dict[str, Any]] = []
    for candidate in candidate_report.get("ranked_candidates") or ():
        bend_id = str(candidate.get("bend_id"))
        reject_reasons = _selection_reject_reasons(candidate)
        local_score = float(candidate.get("candidate_score") or 0.0)
        scored.append(
            {
                "bend_id": bend_id,
                "local_selection_score": round(local_score, 6),
                "candidate_score": candidate.get("candidate_score"),
                "atom_count": candidate.get("atom_count"),
                "incident_flange_ids": candidate.get("incident_flange_ids"),
                "support_centroid": candidate.get("support_centroid"),
                "nearest_baseline": candidate.get("nearest_baseline"),
                "best_baseline_match": candidate.get("best_baseline_match"),
                "candidate_reason_codes": candidate.get("reason_codes") or [],
                "selection_reject_reasons": reject_reasons,
                "selection_admissible": not reject_reasons,
            }
        )
    admissible = [item for item in scored if item["selection_admissible"]]
    admissible.sort(key=lambda item: (-float(item["local_selection_score"]), str(item["bend_id"])))

    conflicts: List[Dict[str, Any]] = []
    for index, left in enumerate(admissible):
        left_region = overcomplete_by_id.get(str(left["bend_id"]))
        if left_region is None:
            continue
        for right in admissible[index + 1 :]:
            right_region = overcomplete_by_id.get(str(right["bend_id"]))
            if right_region is None:
                continue
            conflict = _candidate_conflict(left_region, right_region)
            if float(conflict["conflict_score"]) >= 0.45:
                conflicts.append(conflict)

    top = admissible[0] if admissible else None
    runner_up = admissible[1] if len(admissible) > 1 else None
    score_gap = None
    if top is not None and runner_up is not None:
        score_gap = round(float(top["local_selection_score"]) - float(runner_up["local_selection_score"]), 6)
    close_competitors = [
        item
        for item in admissible[1:]
        if top is not None and float(top["local_selection_score"]) - float(item["local_selection_score"]) < score_gap_floor
    ]
    status = "no_admissible_candidate"
    hold_reasons: List[str] = []
    if top is not None:
        status = "single_candidate_promotable"
        if score_gap is not None and score_gap < score_gap_floor:
            status = "hold_exact_promotion"
            hold_reasons.append("top_candidate_not_unique_enough")
        if len(admissible) > top_plausible_limit:
            status = "hold_exact_promotion"
            hold_reasons.append("too_many_plausible_candidates")
        if conflicts:
            status = "hold_exact_promotion"
            hold_reasons.append("plausible_candidates_conflict_on_corridors")

    baseline_count = int(candidate_report.get("baseline", {}).get("owned_bend_region_count") or len(baseline_payload.get("owned_bend_regions") or ()))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic-only one-birth local selection against the stable baseline. No exact count is promoted automatically.",
        "baseline_owned_bend_count": baseline_count,
        "target_one_birth_count": baseline_count + 1,
        "selection_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "score_gap_floor": score_gap_floor,
        "top_plausible_limit": top_plausible_limit,
        "recommended_candidate": top,
        "runner_up_candidate": runner_up,
        "top_score_gap": score_gap,
        "close_competitors": close_competitors,
        "admissible_candidate_count": len(admissible),
        "rejected_candidate_count": len(scored) - len(admissible),
        "admissible_candidates": admissible,
        "rejected_candidates": [item for item in scored if not item["selection_admissible"]],
        "candidate_conflicts": conflicts,
        "interpretation": {
            "single_candidate_promotable": "A unique candidate cleared duplicate/support checks and separated from the runner-up by the configured score gap.",
            "hold_exact_promotion": "At least one plausible candidate exists, but ambiguity/conflict is still too high for exact promotion.",
            "no_admissible_candidate": "The overcomplete pool did not contain a candidate that cleared the local selection gates.",
        },
    }


def _conflict_lookup(conflicts: Sequence[Mapping[str, Any]]) -> Dict[frozenset[str], Mapping[str, Any]]:
    lookup: Dict[frozenset[str], Mapping[str, Any]] = {}
    for conflict in conflicts:
        lhs = str(conflict.get("lhs"))
        rhs = str(conflict.get("rhs"))
        if lhs and rhs:
            lookup[frozenset((lhs, rhs))] = conflict
    return lookup


def _candidate_subset_score(
    subset: Sequence[Mapping[str, Any]],
    conflict_by_pair: Mapping[frozenset[str], Mapping[str, Any]],
    *,
    label_cost: float,
    conflict_weight: float,
) -> Dict[str, Any]:
    bend_ids = [str(item.get("bend_id")) for item in subset]
    raw_score = float(sum(float(item.get("local_selection_score") or 0.0) for item in subset))
    label_penalty = float(label_cost * len(subset))
    conflict_penalty = 0.0
    subset_conflicts: List[Mapping[str, Any]] = []
    for lhs, rhs in combinations(bend_ids, 2):
        conflict = conflict_by_pair.get(frozenset((lhs, rhs)))
        if conflict is None:
            continue
        conflict_score = float(conflict.get("conflict_score") or 0.0)
        conflict_penalty += conflict_weight * conflict_score
        subset_conflicts.append(conflict)
    subset_score = raw_score - label_penalty - conflict_penalty
    return {
        "bend_ids": bend_ids,
        "added_count": len(bend_ids),
        "subset_score": round(subset_score, 6),
        "raw_candidate_score": round(raw_score, 6),
        "label_penalty": round(label_penalty, 6),
        "conflict_penalty": round(conflict_penalty, 6),
        "conflicts": list(subset_conflicts),
    }


def build_conflict_set_report(
    *,
    selection_report: Mapping[str, Any],
    max_subset_size: int = 3,
    label_cost: float = 0.42,
    conflict_weight: float = 0.85,
    dominance_gap: float = 0.08,
) -> Dict[str, Any]:
    """Enumerate small local birth subsets from the admissible candidate set.

    This is a diagnostic stand-in for the eventual local MAP repair move. It answers
    whether a target +1 object is structurally dominant once label costs and candidate
    conflicts are included.
    """
    candidates = list(selection_report.get("admissible_candidates") or ())
    conflict_by_pair = _conflict_lookup(selection_report.get("candidate_conflicts") or ())
    subsets: List[Dict[str, Any]] = []
    max_size = min(max(1, int(max_subset_size)), len(candidates))
    for size in range(1, max_size + 1):
        for subset in combinations(candidates, size):
            subsets.append(
                _candidate_subset_score(
                    subset,
                    conflict_by_pair,
                    label_cost=label_cost,
                    conflict_weight=conflict_weight,
                )
            )
    subsets.sort(key=lambda item: (-float(item["subset_score"]), int(item["added_count"]), tuple(item["bend_ids"])))
    best_by_added_count: Dict[str, Dict[str, Any]] = {}
    top_by_added_count: Dict[str, List[Dict[str, Any]]] = {}
    for size in range(1, max_size + 1):
        size_subsets = [item for item in subsets if int(item["added_count"]) == size]
        key = f"+{size}"
        if size_subsets:
            best_by_added_count[key] = size_subsets[0]
            top_by_added_count[key] = size_subsets[:5]

    one_birth = best_by_added_count.get("+1")
    one_birth_runner_up: Optional[Dict[str, Any]] = None
    one_birth_candidates = top_by_added_count.get("+1") or []
    if len(one_birth_candidates) > 1:
        one_birth_runner_up = one_birth_candidates[1]

    status = "no_admissible_candidate"
    hold_reasons: List[str] = []
    if one_birth is not None:
        status = "single_birth_structurally_dominant"
        if one_birth_runner_up is not None:
            gap = float(one_birth["subset_score"]) - float(one_birth_runner_up["subset_score"])
            if gap < dominance_gap:
                status = "hold_exact_promotion"
                hold_reasons.append("one_birth_subset_not_dominant")
        if len(candidates) > 3:
            status = "hold_exact_promotion"
            hold_reasons.append("too_many_admissible_birth_candidates")
        if selection_report.get("candidate_conflicts"):
            status = "hold_exact_promotion"
            hold_reasons.append("conflict_graph_non_empty")
        best_two = best_by_added_count.get("+2")
        if best_two is not None and float(best_two["subset_score"]) >= float(one_birth["subset_score"]) - dominance_gap:
            status = "hold_exact_promotion"
            hold_reasons.append("multi_birth_subset_competes_with_one_birth")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic-only conflict-set enumeration over admissible local birth candidates.",
        "selection_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "max_subset_size": max_subset_size,
        "label_cost": label_cost,
        "conflict_weight": conflict_weight,
        "dominance_gap": dominance_gap,
        "admissible_candidate_count": len(candidates),
        "subset_count": len(subsets),
        "best_by_added_count": best_by_added_count,
        "top_by_added_count": top_by_added_count,
        "top_subsets": subsets[:10],
        "interpretation": {
            "single_birth_structurally_dominant": "The best +1 subset is separated from alternatives and no local conflict/multi-birth ambiguity blocks exact promotion.",
            "hold_exact_promotion": "The local candidate set still contains competing/conflicting explanations; exact promotion remains blocked.",
            "no_admissible_candidate": "No local birth candidate cleared prior selection gates.",
        },
    }


def build_conflict_sensitivity_report(
    *,
    selection_report: Mapping[str, Any],
    label_costs: Sequence[float] = (0.30, 0.42, 0.55, 0.70, 0.85, 1.00),
    conflict_weights: Sequence[float] = (0.50, 0.85, 1.20, 1.60),
    max_subset_size: int = 3,
    dominance_gap: float = 0.08,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    dominant_runs: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    for label_cost in label_costs:
        for conflict_weight in conflict_weights:
            report = build_conflict_set_report(
                selection_report=selection_report,
                max_subset_size=max_subset_size,
                label_cost=float(label_cost),
                conflict_weight=float(conflict_weight),
                dominance_gap=dominance_gap,
            )
            best_by_count = report.get("best_by_added_count") or {}
            best_plus_one = best_by_count.get("+1") or {}
            best_plus_two = best_by_count.get("+2") or {}
            best_plus_three = best_by_count.get("+3") or {}
            run = {
                "label_cost": round(float(label_cost), 6),
                "conflict_weight": round(float(conflict_weight), 6),
                "selection_status": report.get("selection_status"),
                "hold_reasons": report.get("hold_reasons") or [],
                "best_plus_one": {
                    "bend_ids": best_plus_one.get("bend_ids"),
                    "subset_score": best_plus_one.get("subset_score"),
                },
                "best_plus_two": {
                    "bend_ids": best_plus_two.get("bend_ids"),
                    "subset_score": best_plus_two.get("subset_score"),
                },
                "best_plus_three": {
                    "bend_ids": best_plus_three.get("bend_ids"),
                    "subset_score": best_plus_three.get("subset_score"),
                },
            }
            runs.append(run)
            status = str(report.get("selection_status"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "single_birth_structurally_dominant":
                dominant_runs.append(run)

    dominant_candidate_counts: Dict[str, int] = {}
    for run in dominant_runs:
        bend_ids = run.get("best_plus_one", {}).get("bend_ids") or []
        if bend_ids:
            bend_id = str(bend_ids[0])
            dominant_candidate_counts[bend_id] = dominant_candidate_counts.get(bend_id, 0) + 1

    if not dominant_runs:
        interpretation_status = "no_reasonable_sweep_promotes_exact_12"
    elif len(dominant_candidate_counts) == 1:
        interpretation_status = "stable_single_candidate_region_exists"
    else:
        interpretation_status = "dominant_candidate_changes_across_sweep"

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Sensitivity sweep for the diagnostic conflict-set solver. This tests whether exact +1 dominance is a stable property or only a tuning artifact.",
        "interpretation_status": interpretation_status,
        "status_counts": status_counts,
        "dominant_candidate_counts": dominant_candidate_counts,
        "dominant_run_count": len(dominant_runs),
        "total_run_count": len(runs),
        "label_costs": [round(float(value), 6) for value in label_costs],
        "conflict_weights": [round(float(value), 6) for value in conflict_weights],
        "max_subset_size": max_subset_size,
        "dominance_gap": dominance_gap,
        "dominant_runs": dominant_runs,
        "runs": runs,
    }


def _corridor_components(
    candidates: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    *,
    min_conflict_score: float = 0.45,
) -> List[List[str]]:
    ids = [str(item.get("bend_id")) for item in candidates]
    parent = {bend_id: bend_id for bend_id in ids}

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
        if float(conflict.get("conflict_score") or 0.0) < min_conflict_score:
            continue
        lhs = str(conflict.get("lhs"))
        rhs = str(conflict.get("rhs"))
        if lhs in known and rhs in known:
            union(lhs, rhs)

    groups: Dict[str, List[str]] = {}
    for bend_id in ids:
        groups.setdefault(find(bend_id), []).append(bend_id)
    return [sorted(values) for values in groups.values()]


def build_corridor_coverage_report(
    *,
    selection_report: Mapping[str, Any],
    min_gap_score: float = 0.72,
    dominance_gap: float = 0.08,
    min_conflict_score: float = 0.45,
) -> Dict[str, Any]:
    """Group local candidates into provisional physical corridor families.

    This is the first diagnostic object-likelihood layer. It asks whether the
    overcomplete pool contains exactly one under-owned corridor, or several competing
    corridor families/fragments.
    """
    candidates = list(selection_report.get("admissible_candidates") or ())
    by_id = {str(item.get("bend_id")): item for item in candidates}
    conflicts = list(selection_report.get("candidate_conflicts") or ())
    conflict_by_pair = _conflict_lookup(conflicts)
    components: List[Dict[str, Any]] = []
    for group_ids in _corridor_components(candidates, conflicts, min_conflict_score=min_conflict_score):
        members = sorted((by_id[bend_id] for bend_id in group_ids), key=lambda item: (-float(item.get("local_selection_score") or 0.0), str(item.get("bend_id"))))
        if not members:
            continue
        group_conflicts = []
        for lhs, rhs in combinations([str(item.get("bend_id")) for item in members], 2):
            conflict = conflict_by_pair.get(frozenset((lhs, rhs)))
            if conflict is not None:
                group_conflicts.append(conflict)
        best = members[0]
        runner_up = members[1] if len(members) > 1 else None
        best_score = float(best.get("local_selection_score") or 0.0)
        runner_up_score = float(runner_up.get("local_selection_score") or 0.0) if runner_up is not None else None
        score_gap = None if runner_up_score is None else round(best_score - runner_up_score, 6)
        member_scores = [float(item.get("local_selection_score") or 0.0) for item in members]
        nearest_distances = [
            float((item.get("nearest_baseline") or {}).get("centroid_distance_mm") or 0.0)
            for item in members
        ]
        reason_codes: List[str] = []
        if best_score < min_gap_score:
            reason_codes.append("weak_corridor_gap_score")
        if len(members) > 1 and (score_gap is None or score_gap < dominance_gap):
            reason_codes.append("corridor_fragment_choice_ambiguous")
        if len(group_conflicts) > 0:
            reason_codes.append("internal_corridor_conflicts")
        if len(members) > 2:
            reason_codes.append("many_fragments_on_corridor")
        if not reason_codes:
            reason_codes.append("corridor_gap_candidate")
        components.append(
            {
                "corridor_id": f"C{len(components) + 1}",
                "member_bend_ids": [str(item.get("bend_id")) for item in members],
                "member_count": len(members),
                "best_candidate": {
                    "bend_id": best.get("bend_id"),
                    "score": round(best_score, 6),
                    "incident_flange_ids": best.get("incident_flange_ids"),
                    "atom_count": best.get("atom_count"),
                    "nearest_baseline": best.get("nearest_baseline"),
                },
                "runner_up_candidate": None
                if runner_up is None
                else {
                    "bend_id": runner_up.get("bend_id"),
                    "score": round(float(runner_up.get("local_selection_score") or 0.0), 6),
                    "incident_flange_ids": runner_up.get("incident_flange_ids"),
                    "atom_count": runner_up.get("atom_count"),
                },
                "best_score_gap": score_gap,
                "score_sum": round(float(sum(member_scores)), 6),
                "score_mean": round(float(sum(member_scores) / max(1, len(member_scores))), 6),
                "min_nearest_baseline_distance_mm": round(float(min(nearest_distances)) if nearest_distances else 0.0, 6),
                "internal_conflict_count": len(group_conflicts),
                "internal_conflicts": group_conflicts,
                "reason_codes": reason_codes,
                "coverage_admissible": reason_codes == ["corridor_gap_candidate"],
            }
        )
    components.sort(key=lambda item: (-float((item.get("best_candidate") or {}).get("score") or 0.0), -int(item.get("member_count") or 0), str(item.get("corridor_id"))))
    for index, component in enumerate(components, start=1):
        component["corridor_id"] = f"C{index}"

    admissible_corridors = [item for item in components if item.get("coverage_admissible")]
    strong_corridors = [
        item
        for item in components
        if float((item.get("best_candidate") or {}).get("score") or 0.0) >= min_gap_score
    ]
    status = "no_corridor_gap"
    hold_reasons: List[str] = []
    if len(admissible_corridors) == 1 and len(strong_corridors) == 1:
        status = "single_corridor_gap"
    elif len(strong_corridors) > 1:
        status = "hold_exact_promotion"
        hold_reasons.append("multiple_strong_corridor_gaps")
    elif strong_corridors:
        status = "hold_exact_promotion"
        hold_reasons.append("strong_corridor_gap_is_fragment_ambiguous")
    if any("many_fragments_on_corridor" in item.get("reason_codes", []) for item in components):
        hold_reasons.append("corridor_fragmentation_unresolved")
    if any("internal_corridor_conflicts" in item.get("reason_codes", []) for item in components):
        hold_reasons.append("internal_corridor_conflicts")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic-only corridor coverage grouping over admissible local birth candidates.",
        "coverage_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "min_gap_score": min_gap_score,
        "dominance_gap": dominance_gap,
        "min_conflict_score": min_conflict_score,
        "corridor_count": len(components),
        "strong_corridor_count": len(strong_corridors),
        "admissible_corridor_count": len(admissible_corridors),
        "corridors": components,
        "interpretation": {
            "single_corridor_gap": "Exactly one strong, internally unambiguous under-owned corridor was found.",
            "hold_exact_promotion": "The corridor-level evidence is still ambiguous or indicates multiple possible gaps.",
            "no_corridor_gap": "No strong corridor-level gap was found.",
        },
    }


def _write_corridor_coverage_markdown(
    *,
    path: Path,
    coverage_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Corridor Coverage Probe",
        "",
        f"Generated: `{coverage_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{coverage_report.get('coverage_status')}`",
        f"- Hold reasons: `{coverage_report.get('hold_reasons')}`",
        f"- Corridors: `{coverage_report.get('corridor_count')}`",
        f"- Strong corridors: `{coverage_report.get('strong_corridor_count')}`",
        f"- Admissible corridors: `{coverage_report.get('admissible_corridor_count')}`",
        "",
        "## Corridors",
        "",
        "| Corridor | Members | Best | Best score | Gap | Reasons |",
        "|---|---|---|---:|---:|---|",
    ]
    for corridor in coverage_report.get("corridors") or ():
        best = corridor.get("best_candidate") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{corridor.get('corridor_id')}`",
                    f"`{corridor.get('member_bend_ids')}`",
                    f"`{best.get('bend_id')}`",
                    f"`{best.get('score')}`",
                    f"`{corridor.get('best_score_gap')}`",
                    f"`{corridor.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This is a corridor-level diagnostic, not promotion logic. A stable exact promotion should require one strong unowned corridor and no competing strong corridor families.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assignment(payload: Mapping[str, Any]) -> Tuple[Mapping[str, str], Mapping[str, str]]:
    assignment = payload.get("assignment") or {}
    atom_labels = {str(key): str(value) for key, value in (assignment.get("atom_labels") or {}).items()}
    label_types = {str(key): str(value) for key, value in (assignment.get("label_types") or {}).items()}
    return atom_labels, label_types


def _quotient_contacts(payload: Mapping[str, Any]) -> Dict[str, Any]:
    atom_labels, label_types = _assignment(payload)
    adjacency = (payload.get("atom_graph") or {}).get("adjacency") or {}
    label_edge_weights: Dict[Tuple[str, str], int] = {}
    seen_edges: set[Tuple[str, str]] = set()
    for atom_id, neighbors in adjacency.items():
        atom = str(atom_id)
        left_label = atom_labels.get(atom)
        if left_label is None:
            continue
        for neighbor_id in neighbors or ():
            neighbor = str(neighbor_id)
            edge = tuple(sorted((atom, neighbor)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            right_label = atom_labels.get(neighbor)
            if right_label is None or right_label == left_label:
                continue
            label_edge = tuple(sorted((left_label, right_label)))
            label_edge_weights[label_edge] = label_edge_weights.get(label_edge, 0) + 1

    direct_flange_contacts: Dict[Tuple[str, str], int] = {}
    bend_flange_contacts: Dict[str, Dict[str, int]] = {}
    for (left, right), weight in label_edge_weights.items():
        left_type = label_types.get(left)
        right_type = label_types.get(right)
        if left_type == "flange" and right_type == "flange":
            pair = _pair_from_values((left, right))
            direct_flange_contacts[pair] = direct_flange_contacts.get(pair, 0) + weight
        elif left_type == "bend" and right_type == "flange":
            bend_flange_contacts.setdefault(left, {})[right] = bend_flange_contacts.setdefault(left, {}).get(right, 0) + weight
        elif left_type == "flange" and right_type == "bend":
            bend_flange_contacts.setdefault(right, {})[left] = bend_flange_contacts.setdefault(right, {}).get(left, 0) + weight

    bend_contact_pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for bend_label, contacts in bend_flange_contacts.items():
        ranked = sorted(contacts.items(), key=lambda item: (-int(item[1]), item[0]))
        if len(ranked) < 2:
            continue
        pair = _pair_from_values((ranked[0][0], ranked[1][0]))
        bend_contact_pairs.setdefault(pair, []).append(
            {
                "bend_label": bend_label,
                "flange_contacts": {key: int(value) for key, value in ranked},
                "top_pair": list(pair),
                "top_pair_contact_weight": int(ranked[0][1] + ranked[1][1]),
            }
        )
    return {
        "direct_flange_contacts": direct_flange_contacts,
        "bend_contact_pairs": bend_contact_pairs,
        "bend_flange_contacts": bend_flange_contacts,
    }


def _flange_lookup(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(flange.get("flange_id")): flange for flange in payload.get("flange_hypotheses") or ()}


def _plane_pair_line(
    flange_a: Mapping[str, Any],
    flange_b: Mapping[str, Any],
    ref: Sequence[Any],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    normal_a = _unit(flange_a.get("normal") or (0, 0, 0))
    normal_b = _unit(flange_b.get("normal") or (0, 0, 0))
    axis = np.cross(normal_a, normal_b)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
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


def _pca_line(points: Sequence[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return None
    matrix = np.asarray(points, dtype=np.float64)
    centroid = np.mean(matrix, axis=0)
    centered = matrix - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(vh) == 0:
        return None
    return centroid, _unit(vh[0])


def _score_uncovered_corridor_geometry(
    *,
    audit: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    best = audit.get("best_candidate") or {}
    bend_id = str(best.get("bend_id"))
    region_lookup = {str(region.get("bend_id")): region for region in overcomplete_payload.get("owned_bend_regions") or ()}
    region = region_lookup.get(bend_id) or {}
    pair = _pair_from_values(audit.get("candidate_pair") or best.get("incident_flange_ids") or ())
    flanges = _flange_lookup(overcomplete_payload)
    atoms = _atom_lookup(overcomplete_payload)
    atom_ids = list(_region_atoms(region))
    support_points = [
        _vec((atoms.get(atom_id) or {}).get("centroid") or (0, 0, 0))
        for atom_id in atom_ids
        if atom_id in atoms
    ]
    support_centroid = region.get("support_centroid") or best.get("support_centroid") or (0, 0, 0)
    local_spacing = float((overcomplete_payload.get("atom_graph") or {}).get("local_spacing_mm") or 1.0)
    support_bendness = _support_bendness(region, atoms)
    line_fit_score = 0.0
    median_line_distance = None
    p90_line_distance = None
    axis_delta_deg = None
    angle_delta_deg = None
    span_length_mm = None
    local_line_fit_score = 0.0
    local_line_axis_score = 0.0
    local_line_median_distance = None
    local_line_p90_distance = None
    local_line_span_mm = None
    geometry_reasons: List[str] = []

    if len(pair) == 2 and pair[0] in flanges and pair[1] in flanges and support_points:
        line = _plane_pair_line(flanges[pair[0]], flanges[pair[1]], support_centroid)
        if line is not None:
            anchor, pair_axis = line
            distances = [_distance_to_line(point, anchor, pair_axis) for point in support_points]
            median_line_distance = float(np.median(distances))
            p90_line_distance = float(np.percentile(distances, 90))
            sigma_line = max(4.0 * local_spacing, 6.0)
            line_fit_score = max(0.0, 1.0 - median_line_distance / sigma_line)
            axis_delta_deg = _axis_angle_deg(region.get("axis_direction") or best.get("axis_direction") or (1, 0, 0), pair_axis)
            pair_angle = _axis_angle_deg(flanges[pair[0]].get("normal") or (1, 0, 0), flanges[pair[1]].get("normal") or (1, 0, 0))
            angle_delta_deg = abs(float(region.get("angle_deg") or best.get("angle_deg") or 0.0) - pair_angle)
            projections = [float(np.dot(point, pair_axis)) for point in support_points]
            span_length_mm = max(projections) - min(projections) if projections else None
        else:
            geometry_reasons.append("parallel_or_invalid_flange_pair")
    else:
        geometry_reasons.append("missing_flange_or_support_geometry")

    local_line = _pca_line(support_points)
    if local_line is not None:
        local_anchor, local_axis = local_line
        local_distances = [_distance_to_line(point, local_anchor, local_axis) for point in support_points]
        local_line_median_distance = float(np.median(local_distances))
        local_line_p90_distance = float(np.percentile(local_distances, 90))
        sigma_local = max(3.0 * local_spacing, 5.0)
        local_line_fit_score = max(0.0, 1.0 - local_line_median_distance / sigma_local)
        local_line_axis_score = max(
            0.0,
            1.0
            - _axis_angle_deg(region.get("axis_direction") or best.get("axis_direction") or (1, 0, 0), local_axis)
            / 25.0,
        )
        local_projections = [float(np.dot(point, local_axis)) for point in support_points]
        local_line_span_mm = max(local_projections) - min(local_projections) if local_projections else None
    else:
        geometry_reasons.append("insufficient_support_for_local_corridor_fit")

    axis_score = max(0.0, 1.0 - (float(axis_delta_deg) if axis_delta_deg is not None else 90.0) / 30.0)
    angle_score = max(0.0, 1.0 - (float(angle_delta_deg) if angle_delta_deg is not None else 90.0) / 35.0)
    atom_score = min(1.0, math.log1p(len(atom_ids)) / math.log(25.0))
    mass_score = min(1.0, float(region.get("support_mass") or best.get("support_mass") or 0.0) / 2500.0)
    separation_score = min(1.0, float((best.get("nearest_baseline") or {}).get("centroid_distance_mm") or 0.0) / 45.0)
    span_score = min(1.0, float(span_length_mm or 0.0) / max(20.0, 6.0 * local_spacing))
    corridor_score = (
        0.14 * line_fit_score
        + 0.14 * local_line_fit_score
        + 0.12 * axis_score
        + 0.10 * local_line_axis_score
        + 0.12 * angle_score
        + 0.14 * atom_score
        + 0.10 * mass_score
        + 0.12 * separation_score
        + 0.10 * support_bendness
        + 0.10 * span_score
    )
    if line_fit_score < 0.35:
        geometry_reasons.append("weak_interface_line_fit")
    if local_line_fit_score < 0.50:
        geometry_reasons.append("weak_local_corridor_fit")
    if local_line_axis_score < 0.35:
        geometry_reasons.append("weak_local_corridor_axis_alignment")
    if axis_score < 0.35:
        geometry_reasons.append("weak_pair_axis_alignment")
    if atom_score < 0.45:
        geometry_reasons.append("small_support")
    if not geometry_reasons:
        geometry_reasons.append("physically_plausible_uncovered_corridor")
    return {
        "corridor_id": audit.get("corridor_id"),
        "bend_id": bend_id,
        "candidate_pair": list(pair),
        "corridor_geometry_score": round(float(corridor_score), 6),
        "component_scores": {
            "line_fit_score": round(float(line_fit_score), 6),
            "local_line_fit_score": round(float(local_line_fit_score), 6),
            "axis_score": round(float(axis_score), 6),
            "local_line_axis_score": round(float(local_line_axis_score), 6),
            "angle_score": round(float(angle_score), 6),
            "atom_score": round(float(atom_score), 6),
            "mass_score": round(float(mass_score), 6),
            "separation_score": round(float(separation_score), 6),
            "support_bendness": round(float(support_bendness), 6),
            "span_score": round(float(span_score), 6),
        },
        "median_line_distance_mm": None if median_line_distance is None else round(float(median_line_distance), 6),
        "p90_line_distance_mm": None if p90_line_distance is None else round(float(p90_line_distance), 6),
        "local_line_median_distance_mm": None if local_line_median_distance is None else round(float(local_line_median_distance), 6),
        "local_line_p90_distance_mm": None if local_line_p90_distance is None else round(float(local_line_p90_distance), 6),
        "axis_delta_deg": None if axis_delta_deg is None else round(float(axis_delta_deg), 6),
        "angle_delta_deg": None if angle_delta_deg is None else round(float(angle_delta_deg), 6),
        "span_length_mm": None if span_length_mm is None else round(float(span_length_mm), 6),
        "local_line_span_mm": None if local_line_span_mm is None else round(float(local_line_span_mm), 6),
        "atom_count": len(atom_ids),
        "support_mass": round(float(region.get("support_mass") or best.get("support_mass") or 0.0), 6),
        "nearest_baseline": best.get("nearest_baseline"),
        "reason_codes": geometry_reasons,
    }


def _point_plane_distance(point: Sequence[Any], flange: Mapping[str, Any]) -> float:
    normal = _unit(flange.get("normal") or (0, 0, 0))
    return abs(float(np.dot(normal, _vec(point)) + float(flange.get("d") or 0.0)))


def _score_candidate_flange_proximity(
    *,
    audit: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    best = audit.get("best_candidate") or {}
    bend_id = str(best.get("bend_id"))
    pair = _pair_from_values(audit.get("candidate_pair") or best.get("incident_flange_ids") or ())
    regions = {str(region.get("bend_id")): region for region in overcomplete_payload.get("owned_bend_regions") or ()}
    region = regions.get(bend_id) or {}
    atom_lookup = _atom_lookup(overcomplete_payload)
    flanges = _flange_lookup(overcomplete_payload)
    atom_ids = sorted(_region_atoms(region))
    points = [
        (atom_lookup.get(atom_id) or {}).get("centroid") or (0, 0, 0)
        for atom_id in atom_ids
        if atom_id in atom_lookup
    ]
    local_spacing = float((overcomplete_payload.get("atom_graph") or {}).get("local_spacing_mm") or 1.0)
    near_threshold = max(4.0 * local_spacing, 8.0)
    flange_stats: List[Dict[str, Any]] = []
    for flange_id, flange in flanges.items():
        if not points:
            continue
        distances = [_point_plane_distance(point, flange) for point in points]
        near_fraction = sum(1 for value in distances if value <= near_threshold) / max(1, len(distances))
        flange_stats.append(
            {
                "flange_id": flange_id,
                "median_distance_mm": round(float(np.median(distances)), 6),
                "p90_distance_mm": round(float(np.percentile(distances, 90)), 6),
                "near_fraction": round(float(near_fraction), 6),
                "designated": flange_id in pair,
            }
        )
    flange_stats.sort(key=lambda item: (float(item["median_distance_mm"]), -float(item["near_fraction"]), item["flange_id"]))
    designated_stats = [item for item in flange_stats if item["flange_id"] in pair]
    other_stats = [item for item in flange_stats if item["flange_id"] not in pair]
    designated_near = sum(float(item["near_fraction"]) for item in designated_stats)
    designated_median = float(np.mean([float(item["median_distance_mm"]) for item in designated_stats])) if designated_stats else 999.0
    best_other_near = max((float(item["near_fraction"]) for item in other_stats), default=0.0)
    best_other_median = min((float(item["median_distance_mm"]) for item in other_stats), default=999.0)
    designated_proximity_score = min(1.0, designated_near / max(1, len(pair)))
    median_margin = max(0.0, min(1.0, (best_other_median - designated_median + near_threshold) / (2.0 * near_threshold)))
    leakage_score = max(0.0, 1.0 - best_other_near)
    proximity_score = 0.50 * designated_proximity_score + 0.30 * median_margin + 0.20 * leakage_score
    reasons: List[str] = []
    if len(designated_stats) < 2:
        reasons.append("missing_designated_flange_geometry")
    if designated_proximity_score < 0.35:
        reasons.append("weak_designated_flange_proximity")
    if best_other_near > designated_proximity_score + 0.20:
        reasons.append("closer_to_non_designated_flange")
    if leakage_score < 0.50:
        reasons.append("high_non_designated_flange_proximity")
    if not reasons:
        reasons.append("clean_designated_flange_proximity")
    return {
        "corridor_id": audit.get("corridor_id"),
        "bend_id": bend_id,
        "candidate_pair": list(pair),
        "flange_proximity_score": round(float(proximity_score), 6),
        "near_threshold_mm": round(float(near_threshold), 6),
        "designated_proximity_score": round(float(designated_proximity_score), 6),
        "designated_mean_median_distance_mm": round(float(designated_median), 6),
        "best_other_near_fraction": round(float(best_other_near), 6),
        "best_other_median_distance_mm": round(float(best_other_median), 6),
        "component_scores": {
            "designated_proximity_score": round(float(designated_proximity_score), 6),
            "median_margin": round(float(median_margin), 6),
            "leakage_score": round(float(leakage_score), 6),
        },
        "top_flange_distances": flange_stats[:8],
        "reason_codes": reasons,
    }


def build_flange_proximity_incidence_report(
    *,
    overcomplete_payload: Mapping[str, Any],
    quotient_audit_report: Mapping[str, Any],
    dominance_gap: float = 0.10,
) -> Dict[str, Any]:
    rows = [
        _score_candidate_flange_proximity(audit=audit, overcomplete_payload=overcomplete_payload)
        for audit in quotient_audit_report.get("corridor_audits") or ()
        if audit.get("uncovered_exact_pair")
    ]
    rows.sort(key=lambda item: (-float(item.get("flange_proximity_score") or 0.0), str(item.get("corridor_id"))))
    top = rows[0] if rows else None
    runner_up = rows[1] if len(rows) > 1 else None
    score_gap = None
    status = "no_uncovered_corridor"
    hold_reasons: List[str] = []
    if top is not None:
        status = "single_flange_proximity_candidate"
        if runner_up is not None:
            score_gap = round(float(top["flange_proximity_score"]) - float(runner_up["flange_proximity_score"]), 6)
            if score_gap < dominance_gap:
                status = "hold_exact_promotion"
                hold_reasons.append("flange_proximity_not_dominant")
        if len(rows) > 1:
            hold_reasons.append("multiple_uncovered_corridors")
        if top.get("reason_codes") != ["clean_designated_flange_proximity"]:
            status = "hold_exact_promotion"
            hold_reasons.append("top_flange_proximity_not_clean")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic distance-based flange incidence for exact-uncovered candidate corridors.",
        "proximity_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "dominance_gap": dominance_gap,
        "candidate_count": len(rows),
        "recommended_proximity_candidate": top,
        "runner_up_proximity_candidate": runner_up,
        "top_score_gap": score_gap,
        "candidates": rows,
    }


def build_flange_family_report(
    *,
    payload: Mapping[str, Any],
    normal_threshold_deg: float = 4.0,
    plane_threshold_mm: float = 12.0,
) -> Dict[str, Any]:
    flanges = _flange_lookup(payload)
    ids = sorted(flanges)
    parent = {flange_id: flange_id for flange_id in ids}

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

    for left, right in combinations(ids, 2):
        left_flange = flanges[left]
        right_flange = flanges[right]
        normal_delta = _axis_angle_deg(left_flange.get("normal") or (1, 0, 0), right_flange.get("normal") or (1, 0, 0))
        plane_delta = abs(float(left_flange.get("d") or 0.0) - float(right_flange.get("d") or 0.0))
        if normal_delta <= normal_threshold_deg and plane_delta <= plane_threshold_mm:
            union(left, right)

    grouped: Dict[str, List[str]] = {}
    for flange_id in ids:
        grouped.setdefault(find(flange_id), []).append(flange_id)

    families: List[Dict[str, Any]] = []
    flange_to_family: Dict[str, str] = {}
    for index, members in enumerate(sorted(grouped.values(), key=lambda values: (values[0], len(values))), start=1):
        family_id = f"FF{index}"
        normals = np.asarray([_unit(flanges[member].get("normal") or (1, 0, 0)) for member in members], dtype=np.float64)
        mean_normal = _unit(np.mean(normals, axis=0))
        mean_d = float(np.mean([float(flanges[member].get("d") or 0.0) for member in members]))
        centroids = [
            flanges[member].get("centroid")
            for member in members
            if flanges[member].get("centroid") is not None
        ]
        centroid = np.mean(np.asarray([_vec(value) for value in centroids], dtype=np.float64), axis=0).tolist() if centroids else None
        for member in members:
            flange_to_family[member] = family_id
        families.append(
            {
                "family_id": family_id,
                "flange_ids": members,
                "normal": [round(float(value), 6) for value in mean_normal.tolist()],
                "d": round(mean_d, 6),
                "centroid": None if centroid is None else [round(float(value), 6) for value in centroid],
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic flange orientation/plane family grouping. This does not replace connected flange component IDs for counting.",
        "normal_threshold_deg": normal_threshold_deg,
        "plane_threshold_mm": plane_threshold_mm,
        "flange_count": len(ids),
        "family_count": len(families),
        "flange_to_family": flange_to_family,
        "families": families,
    }


def _family_pair(pair: Sequence[Any], flange_to_family: Mapping[str, str]) -> Tuple[str, ...]:
    values = [flange_to_family.get(str(flange_id), str(flange_id)) for flange_id in pair]
    return _pair_from_values(values)


def build_flange_family_incidence_report(
    *,
    baseline_payload: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
    quotient_audit_report: Mapping[str, Any],
    family_report: Mapping[str, Any],
) -> Dict[str, Any]:
    flange_to_family = {str(key): str(value) for key, value in (family_report.get("flange_to_family") or {}).items()}
    baseline_family_pairs: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for region in baseline_payload.get("owned_bend_regions") or ():
        raw_pair = _pair(region)
        family_pair = _family_pair(raw_pair, flange_to_family)
        if len(family_pair) == 2:
            baseline_family_pairs.setdefault(family_pair, []).append(
                {
                    "bend_id": region.get("bend_id"),
                    "raw_pair": list(raw_pair),
                    "atom_count": len(_region_atoms(region)),
                    "support_mass": region.get("support_mass"),
                }
            )

    proximity_rows = build_flange_proximity_incidence_report(
        overcomplete_payload=overcomplete_payload,
        quotient_audit_report=quotient_audit_report,
    ).get("candidates") or []
    proximity_by_bend = {str(row.get("bend_id")): row for row in proximity_rows}
    rows: List[Dict[str, Any]] = []
    for audit in quotient_audit_report.get("corridor_audits") or ():
        if not audit.get("uncovered_exact_pair"):
            continue
        best = audit.get("best_candidate") or {}
        bend_id = str(best.get("bend_id"))
        raw_pair = _pair_from_values(audit.get("candidate_pair") or best.get("incident_flange_ids") or ())
        family_pair = _family_pair(raw_pair, flange_to_family)
        family_matches = baseline_family_pairs.get(family_pair, [])
        proximity = proximity_by_bend.get(bend_id) or {}
        top_distances = proximity.get("top_flange_distances") or []
        designated_families = set(family_pair)
        other_family_near = 0.0
        same_family_near = 0.0
        for item in top_distances:
            family_id = flange_to_family.get(str(item.get("flange_id")), str(item.get("flange_id")))
            near_fraction = float(item.get("near_fraction") or 0.0)
            if family_id in designated_families:
                same_family_near = max(same_family_near, near_fraction)
            else:
                other_family_near = max(other_family_near, near_fraction)
        family_leakage = max(0.0, other_family_near - same_family_near)
        reasons: List[str] = []
        if family_matches:
            reasons.append("family_pair_already_covered_by_baseline")
        else:
            reasons.append("family_pair_uncovered")
        if family_leakage > 0.20:
            reasons.append("nearer_to_other_flange_family")
        if len(set(raw_pair)) == 2 and len(set(family_pair)) < 2:
            reasons.append("raw_pair_collapses_to_one_family")
        rows.append(
            {
                "corridor_id": audit.get("corridor_id"),
                "bend_id": bend_id,
                "raw_pair": list(raw_pair),
                "family_pair": list(family_pair),
                "baseline_family_matches": family_matches,
                "family_pair_uncovered": not family_matches and len(family_pair) == 2,
                "same_family_near_fraction": round(float(same_family_near), 6),
                "other_family_near_fraction": round(float(other_family_near), 6),
                "family_leakage": round(float(family_leakage), 6),
                "raw_proximity_score": proximity.get("flange_proximity_score"),
                "reason_codes": reasons,
            }
        )
    uncovered = [row for row in rows if row.get("family_pair_uncovered")]
    clean = [row for row in uncovered if "nearer_to_other_flange_family" not in row.get("reason_codes", [])]
    status = "hold_exact_promotion"
    hold_reasons: List[str] = []
    if len(clean) == 1 and len(uncovered) == 1:
        status = "single_uncovered_flange_family_pair"
    elif len(uncovered) > 1:
        hold_reasons.append("multiple_uncovered_family_pairs")
    elif len(uncovered) == 1:
        hold_reasons.append("single_uncovered_family_pair_not_clean")
    else:
        hold_reasons.append("no_uncovered_family_pair")
    if any("family_pair_already_covered_by_baseline" in row.get("reason_codes", []) for row in rows):
        hold_reasons.append("some_raw_uncovered_pairs_are_family_covered")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic incidence audit after grouping raw flange IDs into orientation/plane families.",
        "family_incidence_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "candidate_count": len(rows),
        "uncovered_family_pair_count": len(uncovered),
        "clean_uncovered_family_pair_count": len(clean),
        "candidates": rows,
    }


def build_family_residual_repair_promotion_evidence(
    *,
    baseline_payload: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
    family_incidence_report: Mapping[str, Any],
    ownership_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Emit runner-ready exact +1 evidence when one residual family gap is isolated.

    This is stricter than a plain candidate score and intentionally narrower than the
    full exact-blocker diagnostics. It is meant for cases like y-sherman where raw
    flange IDs create multiple uncovered-looking pairs, but flange-family incidence
    collapses all but one candidate into already-covered structure.
    """
    baseline_count = int(
        baseline_payload.get("exact_bend_count")
        or len(baseline_payload.get("owned_bend_regions") or ())
    )
    promoted_count = baseline_count + 1
    blockers: List[str] = []
    reasons: List[str] = []

    if family_incidence_report.get("family_incidence_status") != "single_uncovered_flange_family_pair":
        blockers.append("family_incidence_not_single_uncovered_pair")
    else:
        reasons.append("single_uncovered_flange_family_pair")

    family_candidates = list(family_incidence_report.get("candidates") or ())
    uncovered = [row for row in family_candidates if row.get("family_pair_uncovered")]
    covered_competitors = [
        row
        for row in family_candidates
        if not row.get("family_pair_uncovered")
        and "family_pair_already_covered_by_baseline" in set(row.get("reason_codes") or ())
    ]
    if len(uncovered) != 1:
        blockers.append("uncovered_family_pair_count_not_one")
        selected_family_candidate: Dict[str, Any] = {}
    else:
        selected_family_candidate = dict(uncovered[0])
        reasons.append("one_clean_uncovered_family_pair")
    if len(family_candidates) > 1 and len(covered_competitors) != len(family_candidates) - len(uncovered):
        blockers.append("raw_competitors_not_all_family_covered")
    else:
        reasons.append("raw_competitors_family_covered")

    selected_bend_id = str(selected_family_candidate.get("bend_id") or "")
    ownership_candidates = {
        str(row.get("bend_id")): row
        for row in ownership_report.get("candidates") or ()
    }
    ownership = dict(ownership_candidates.get(selected_bend_id) or {})
    baseline_fractions = (
        (ownership.get("baseline_label_distribution") or {}).get("type_fractions") or {}
    )
    residual_fraction = float(baseline_fractions.get("residual", 0.0)) + float(baseline_fractions.get("outlier", 0.0))
    if not ownership:
        blockers.append("missing_boundary_residual_ownership_candidate")
    elif residual_fraction < 0.75:
        blockers.append("candidate_baseline_support_not_residual_owned")
    else:
        reasons.append("candidate_owns_baseline_residual")

    overcomplete_regions = {
        str(region.get("bend_id")): dict(region)
        for region in overcomplete_payload.get("owned_bend_regions") or ()
    }
    supplemental_region = dict(overcomplete_regions.get(selected_bend_id) or {})
    if not supplemental_region:
        blockers.append("missing_overcomplete_owned_region")
    else:
        supplemental_region["bend_id"] = f"OB{promoted_count}"
        supplemental_region["admissibility_reasons"] = [
            "supplemental_from_family_residual_repair_evidence"
        ]
        supplemental_region["debug_confidence"] = min(
            0.95,
            max(0.50, float(supplemental_region.get("debug_confidence") or 0.75) * 0.75),
        )
        reasons.append("supplemental_owned_region_available")

    evidence = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_key": overcomplete_payload.get("part_id") or baseline_payload.get("part_id"),
        "family_residual_repair_eligible": not blockers,
        "promoted_exact_count": promoted_count,
        "baseline_exact_count": baseline_count,
        "candidate_bend_id": selected_bend_id or None,
        "candidate_corridor_id": selected_family_candidate.get("corridor_id"),
        "evidence_status": "eligible" if not blockers else "blocked",
        "reason_codes": sorted(set(reasons)),
        "blockers": sorted(set(blockers)),
        "checks": {
            "single_uncovered_flange_family_pair": family_incidence_report.get("family_incidence_status") == "single_uncovered_flange_family_pair",
            "candidate_owns_baseline_residual": bool(ownership and residual_fraction >= 0.75),
            "raw_competitors_family_covered": len(family_candidates) <= 1
            or len(covered_competitors) == len(family_candidates) - len(uncovered),
            "baseline_residual_fraction": round(float(residual_fraction), 6),
        },
        "selected_family_candidate": selected_family_candidate or None,
        "selected_ownership_candidate": ownership or None,
        "render_semantics": "supplemental region is diagnostic/offline; raw F1 decomposition is preserved separately",
    }
    if supplemental_region:
        evidence["supplemental_owned_bend_regions"] = [supplemental_region]
    return evidence


def build_uncovered_corridor_comparison_report(
    *,
    overcomplete_payload: Mapping[str, Any],
    quotient_audit_report: Mapping[str, Any],
    dominance_gap: float = 0.10,
) -> Dict[str, Any]:
    uncovered_audits = [
        audit
        for audit in quotient_audit_report.get("corridor_audits") or ()
        if audit.get("uncovered_exact_pair")
    ]
    scored = [
        _score_uncovered_corridor_geometry(audit=audit, overcomplete_payload=overcomplete_payload)
        for audit in uncovered_audits
    ]
    scored.sort(key=lambda item: (-float(item.get("corridor_geometry_score") or 0.0), str(item.get("corridor_id"))))
    top = scored[0] if scored else None
    runner_up = scored[1] if len(scored) > 1 else None
    score_gap = None
    status = "no_uncovered_corridor"
    hold_reasons: List[str] = []
    if top is not None:
        status = "single_uncovered_corridor_dominant"
        if runner_up is not None:
            score_gap = round(float(top["corridor_geometry_score"]) - float(runner_up["corridor_geometry_score"]), 6)
            if score_gap < dominance_gap:
                status = "hold_exact_promotion"
                hold_reasons.append("uncovered_corridor_comparison_not_dominant")
        if len(scored) > 1:
            hold_reasons.append("multiple_uncovered_corridors")
        if "physically_plausible_uncovered_corridor" not in set(top.get("reason_codes") or ()):
            status = "hold_exact_promotion"
            hold_reasons.append("top_uncovered_corridor_geometry_weak")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic geometry comparison among exact-uncovered quotient corridors.",
        "comparison_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "dominance_gap": dominance_gap,
        "uncovered_corridor_count": len(scored),
        "recommended_uncovered_corridor": top,
        "runner_up_uncovered_corridor": runner_up,
        "top_score_gap": score_gap,
        "uncovered_corridors": scored,
        "interpretation": {
            "single_uncovered_corridor_dominant": "One exact-uncovered corridor has clearly stronger support/interface geometry than alternatives.",
            "hold_exact_promotion": "Exact-uncovered corridor geometry is still ambiguous or weak.",
            "no_uncovered_corridor": "No exact-uncovered corridor was found by the quotient audit.",
        },
    }


def _label_distribution(atom_ids: Sequence[str], atom_labels: Mapping[str, str], label_types: Mapping[str, str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    for atom_id in atom_ids:
        label = atom_labels.get(atom_id, "unassigned")
        labels[label] = labels.get(label, 0) + 1
        label_type = label_types.get(label, label if label in {"residual", "outlier"} else "unassigned")
        counts[label_type] = counts.get(label_type, 0) + 1
    total = max(1, len(atom_ids))
    return {
        "total_atoms": len(atom_ids),
        "type_counts": dict(sorted(counts.items())),
        "label_counts": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
        "type_fractions": {key: round(value / total, 6) for key, value in sorted(counts.items())},
    }


def _component_boundary_contacts(
    *,
    atom_ids: Sequence[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    atom_set = set(atom_ids)
    atom_labels, label_types = _assignment(payload)
    adjacency = (payload.get("atom_graph") or {}).get("adjacency") or {}
    contact_counts: Dict[str, int] = {}
    contact_type_counts: Dict[str, int] = {}
    edge_count = 0
    for atom_id in atom_set:
        for neighbor in adjacency.get(atom_id, ()) or ():
            neighbor_id = str(neighbor)
            if neighbor_id in atom_set:
                continue
            label = atom_labels.get(neighbor_id, "unassigned")
            label_type = label_types.get(label, label if label in {"residual", "outlier"} else "unassigned")
            contact_counts[label] = contact_counts.get(label, 0) + 1
            contact_type_counts[label_type] = contact_type_counts.get(label_type, 0) + 1
            edge_count += 1
    return {
        "boundary_edge_count": edge_count,
        "contact_label_counts": dict(sorted(contact_counts.items(), key=lambda item: (-item[1], item[0]))),
        "contact_type_counts": dict(sorted(contact_type_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def build_boundary_residual_ownership_report(
    *,
    baseline_payload: Mapping[str, Any],
    overcomplete_payload: Mapping[str, Any],
    quotient_audit_report: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_atom_labels, baseline_label_types = _assignment(baseline_payload)
    overcomplete_atom_labels, overcomplete_label_types = _assignment(overcomplete_payload)
    overcomplete_regions = {str(region.get("bend_id")): region for region in overcomplete_payload.get("owned_bend_regions") or ()}
    rows: List[Dict[str, Any]] = []
    for audit in quotient_audit_report.get("corridor_audits") or ():
        if not audit.get("uncovered_exact_pair"):
            continue
        best = audit.get("best_candidate") or {}
        bend_id = str(best.get("bend_id"))
        region = overcomplete_regions.get(bend_id) or {}
        atom_ids = sorted(_region_atoms(region))
        pair = _pair_from_values(audit.get("candidate_pair") or best.get("incident_flange_ids") or ())
        baseline_distribution = _label_distribution(atom_ids, baseline_atom_labels, baseline_label_types)
        overcomplete_distribution = _label_distribution(atom_ids, overcomplete_atom_labels, overcomplete_label_types)
        baseline_contacts = _component_boundary_contacts(atom_ids=atom_ids, payload=baseline_payload)
        overcomplete_contacts = _component_boundary_contacts(atom_ids=atom_ids, payload=overcomplete_payload)
        contact_labels = overcomplete_contacts.get("contact_label_counts") or {}
        designated_contact = sum(int(contact_labels.get(flange_id, 0)) for flange_id in pair)
        flange_contact = int((overcomplete_contacts.get("contact_type_counts") or {}).get("flange", 0))
        leakage_contact = max(0, flange_contact - designated_contact)
        total_boundary = max(1, int(overcomplete_contacts.get("boundary_edge_count") or 0))
        baseline_fractions = baseline_distribution.get("type_fractions") or {}
        residual_or_outlier_fraction = float(baseline_fractions.get("residual", 0.0)) + float(baseline_fractions.get("outlier", 0.0))
        baseline_bend_fraction = float(baseline_fractions.get("bend", 0.0))
        designated_contact_ratio = designated_contact / total_boundary
        leakage_ratio = leakage_contact / total_boundary
        boundary_score = min(1.0, designated_contact_ratio * 2.0) * max(0.0, 1.0 - leakage_ratio)
        residual_score = max(0.0, residual_or_outlier_fraction - 0.5 * baseline_bend_fraction)
        support_score = min(1.0, math.log1p(len(atom_ids)) / math.log(25.0))
        ownership_score = 0.42 * residual_score + 0.34 * boundary_score + 0.24 * support_score
        reasons: List[str] = []
        if residual_or_outlier_fraction < 0.35:
            reasons.append("baseline_support_not_mostly_residual")
        if baseline_bend_fraction > 0.20:
            reasons.append("candidate_steals_existing_baseline_bend_atoms")
        if designated_contact == 0:
            reasons.append("missing_designated_flange_boundary_contact")
        if leakage_ratio > 0.35:
            reasons.append("boundary_leaks_to_other_flanges")
        if boundary_score < 0.20:
            reasons.append("weak_two_flange_boundary_topology")
        if not reasons:
            reasons.append("residual_support_with_clean_boundary")
        rows.append(
            {
                "corridor_id": audit.get("corridor_id"),
                "bend_id": bend_id,
                "candidate_pair": list(pair),
                "ownership_topology_score": round(float(ownership_score), 6),
                "baseline_label_distribution": baseline_distribution,
                "overcomplete_label_distribution": overcomplete_distribution,
                "baseline_boundary_contacts": baseline_contacts,
                "overcomplete_boundary_contacts": overcomplete_contacts,
                "designated_contact_edges": int(designated_contact),
                "designated_contact_ratio": round(float(designated_contact_ratio), 6),
                "leakage_contact_edges": int(leakage_contact),
                "leakage_ratio": round(float(leakage_ratio), 6),
                "component_scores": {
                    "residual_score": round(float(residual_score), 6),
                    "boundary_score": round(float(boundary_score), 6),
                    "support_score": round(float(support_score), 6),
                },
                "reason_codes": reasons,
            }
        )
    rows.sort(key=lambda item: (-float(item.get("ownership_topology_score") or 0.0), str(item.get("corridor_id"))))
    status = "no_uncovered_corridor"
    hold_reasons: List[str] = []
    if rows:
        status = "single_boundary_residual_candidate"
        if len(rows) > 1:
            status = "hold_exact_promotion"
            hold_reasons.append("multiple_uncovered_corridors_need_boundary_resolution")
        if rows[0].get("reason_codes") != ["residual_support_with_clean_boundary"]:
            status = "hold_exact_promotion"
            hold_reasons.append("top_boundary_residual_candidate_not_clean")
        if len(rows) > 1:
            gap = float(rows[0]["ownership_topology_score"]) - float(rows[1]["ownership_topology_score"])
            if gap < 0.10:
                hold_reasons.append("boundary_residual_scores_not_dominant")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic baseline residual ownership and boundary-contact topology for exact-uncovered candidate corridors.",
        "ownership_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "candidate_count": len(rows),
        "candidates": rows,
    }


def build_quotient_corridor_audit_report(
    *,
    baseline_payload: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_region_pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for region in baseline_payload.get("owned_bend_regions") or ():
        pair = _pair(region)
        if len(pair) == 2:
            baseline_region_pairs.setdefault(pair, []).append(
                {
                    "bend_id": region.get("bend_id"),
                    "atom_count": len(_region_atoms(region)),
                    "support_mass": region.get("support_mass"),
                    "angle_deg": region.get("angle_deg"),
                }
            )
    contacts = _quotient_contacts(baseline_payload)
    direct_flange_contacts = contacts["direct_flange_contacts"]
    bend_contact_pairs = contacts["bend_contact_pairs"]

    audits: List[Dict[str, Any]] = []
    for corridor in coverage_report.get("corridors") or ():
        best = corridor.get("best_candidate") or {}
        pair = _pair_from_values(best.get("incident_flange_ids") or ())
        exact_region_matches = baseline_region_pairs.get(pair, [])
        quotient_bend_matches = bend_contact_pairs.get(pair, [])
        direct_contact_weight = int(direct_flange_contacts.get(pair, 0))
        shared_region_matches = []
        for baseline_pair, regions in baseline_region_pairs.items():
            shared = sorted(set(pair) & set(baseline_pair))
            if shared and baseline_pair != pair:
                shared_region_matches.append(
                    {
                        "baseline_pair": list(baseline_pair),
                        "shared_flanges": shared,
                        "regions": regions,
                    }
                )
        reason_codes: List[str] = []
        if exact_region_matches:
            reason_codes.append("exact_pair_already_has_owned_region")
        if quotient_bend_matches:
            reason_codes.append("quotient_graph_has_bend_on_pair")
        if direct_contact_weight > 0:
            reason_codes.append("baseline_has_direct_flange_contact_without_explicit_bend")
        if not reason_codes:
            reason_codes.append("no_exact_baseline_pair_coverage")
        if shared_region_matches and not exact_region_matches:
            reason_codes.append("only_shared_flange_related_coverage")
        audits.append(
            {
                "corridor_id": corridor.get("corridor_id"),
                "candidate_pair": list(pair),
                "best_candidate": best,
                "corridor_reason_codes": corridor.get("reason_codes") or [],
                "baseline_exact_region_matches": exact_region_matches,
                "baseline_quotient_bend_matches": quotient_bend_matches,
                "baseline_direct_flange_contact_weight": direct_contact_weight,
                "baseline_shared_flange_region_matches": shared_region_matches[:6],
                "audit_reason_codes": reason_codes,
                "uncovered_exact_pair": "no_exact_baseline_pair_coverage" in reason_codes,
            }
        )
    uncovered = [item for item in audits if item.get("uncovered_exact_pair")]
    exact_covered = [item for item in audits if not item.get("uncovered_exact_pair")]
    status = "hold_exact_promotion"
    hold_reasons: List[str] = []
    if len(uncovered) == 1 and coverage_report.get("coverage_status") == "single_corridor_gap":
        status = "single_uncovered_corridor"
    elif len(uncovered) == 1:
        hold_reasons.append("one_uncovered_pair_but_corridor_coverage_ambiguous")
    elif len(uncovered) > 1:
        hold_reasons.append("multiple_uncovered_exact_pairs")
    else:
        hold_reasons.append("all_candidate_pairs_have_exact_baseline_coverage")
    if coverage_report.get("coverage_status") != "single_corridor_gap":
        hold_reasons.append("corridor_coverage_not_single_gap")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Diagnostic quotient-graph audit comparing candidate corridors against baseline flange-bend ownership.",
        "audit_status": status,
        "hold_reasons": sorted(set(hold_reasons)),
        "baseline_owned_pair_count": len(baseline_region_pairs),
        "baseline_quotient_pair_count": len(bend_contact_pairs),
        "candidate_corridor_count": len(audits),
        "uncovered_exact_pair_count": len(uncovered),
        "exact_covered_corridor_count": len(exact_covered),
        "baseline_owned_pairs": {",".join(pair): regions for pair, regions in sorted(baseline_region_pairs.items())},
        "corridor_audits": audits,
    }


def _write_quotient_corridor_audit_markdown(
    *,
    path: Path,
    audit_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Quotient Corridor Audit",
        "",
        f"Generated: `{audit_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{audit_report.get('audit_status')}`",
        f"- Hold reasons: `{audit_report.get('hold_reasons')}`",
        f"- Candidate corridors: `{audit_report.get('candidate_corridor_count')}`",
        f"- Uncovered exact pairs: `{audit_report.get('uncovered_exact_pair_count')}`",
        f"- Exact-covered corridors: `{audit_report.get('exact_covered_corridor_count')}`",
        "",
        "## Candidate Corridor Audit",
        "",
        "| Corridor | Pair | Best | Uncovered exact pair | Direct flange contact | Reasons |",
        "|---|---|---|---|---:|---|",
    ]
    for item in audit_report.get("corridor_audits") or ():
        best = item.get("best_candidate") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('corridor_id')}`",
                    f"`{item.get('candidate_pair')}`",
                    f"`{best.get('bend_id')}`",
                    f"`{item.get('uncovered_exact_pair')}`",
                    f"`{item.get('baseline_direct_flange_contact_weight')}`",
                    f"`{item.get('audit_reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This is exact-pair quotient evidence only. Because flange IDs can fragment, an uncovered exact pair is a signal for review, not sufficient exact-count proof.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_uncovered_corridor_comparison_markdown(
    *,
    path: Path,
    comparison_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = comparison_report.get("recommended_uncovered_corridor") or {}
    runner_up = comparison_report.get("runner_up_uncovered_corridor") or {}
    lines = [
        "# F1 Exact-Blocker Uncovered Corridor Comparison",
        "",
        f"Generated: `{comparison_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{comparison_report.get('comparison_status')}`",
        f"- Hold reasons: `{comparison_report.get('hold_reasons')}`",
        f"- Uncovered corridors: `{comparison_report.get('uncovered_corridor_count')}`",
        f"- Top score gap: `{comparison_report.get('top_score_gap')}`",
        "",
        "## Top Candidate",
        "",
        f"- Bend: `{top.get('bend_id')}`",
        f"- Pair: `{top.get('candidate_pair')}`",
        f"- Score: `{top.get('corridor_geometry_score')}`",
        f"- Reasons: `{top.get('reason_codes')}`",
        "",
        "## Runner-Up",
        "",
        f"- Bend: `{runner_up.get('bend_id')}`",
        f"- Pair: `{runner_up.get('candidate_pair')}`",
        f"- Score: `{runner_up.get('corridor_geometry_score')}`",
        f"- Reasons: `{runner_up.get('reason_codes')}`",
        "",
        "## Uncovered Corridors",
        "",
        "| Rank | Corridor | Bend | Pair | Score | Pair line | Local line | Axis | Local axis | Angle | Atoms | Reasons |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, item in enumerate(comparison_report.get("uncovered_corridors") or (), start=1):
        scores = item.get("component_scores") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{item.get('corridor_id')}`",
                    f"`{item.get('bend_id')}`",
                    f"`{item.get('candidate_pair')}`",
                    f"`{item.get('corridor_geometry_score')}`",
                    f"`{scores.get('line_fit_score')}`",
                    f"`{scores.get('local_line_fit_score')}`",
                    f"`{scores.get('axis_score')}`",
                    f"`{scores.get('local_line_axis_score')}`",
                    f"`{scores.get('angle_score')}`",
                    f"`{item.get('atom_count')}`",
                    f"`{item.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This comparison is diagnostic only. A dominant uncovered corridor should still be verified visually or folded into a formal local MAP repair objective before exact-count promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_boundary_residual_ownership_markdown(
    *,
    path: Path,
    ownership_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Boundary / Residual Ownership Probe",
        "",
        f"Generated: `{ownership_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{ownership_report.get('ownership_status')}`",
        f"- Hold reasons: `{ownership_report.get('hold_reasons')}`",
        f"- Candidates: `{ownership_report.get('candidate_count')}`",
        "",
        "## Candidates",
        "",
        "| Rank | Corridor | Bend | Pair | Score | Baseline fractions | Designated contact | Leakage | Reasons |",
        "|---:|---|---|---|---:|---|---:|---:|---|",
    ]
    for index, item in enumerate(ownership_report.get("candidates") or (), start=1):
        baseline_fractions = (item.get("baseline_label_distribution") or {}).get("type_fractions") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{item.get('corridor_id')}`",
                    f"`{item.get('bend_id')}`",
                    f"`{item.get('candidate_pair')}`",
                    f"`{item.get('ownership_topology_score')}`",
                    f"`{baseline_fractions}`",
                    f"`{item.get('designated_contact_ratio')}`",
                    f"`{item.get('leakage_ratio')}`",
                    f"`{item.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "A true missing bend should usually own atoms that were residual/outlier in the baseline and should have clean boundary contact to its selected flange pair.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_flange_proximity_incidence_markdown(
    *,
    path: Path,
    proximity_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = proximity_report.get("recommended_proximity_candidate") or {}
    runner_up = proximity_report.get("runner_up_proximity_candidate") or {}
    lines = [
        "# F1 Exact-Blocker Flange Proximity Incidence Probe",
        "",
        f"Generated: `{proximity_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{proximity_report.get('proximity_status')}`",
        f"- Hold reasons: `{proximity_report.get('hold_reasons')}`",
        f"- Candidates: `{proximity_report.get('candidate_count')}`",
        f"- Top score gap: `{proximity_report.get('top_score_gap')}`",
        "",
        "## Top Candidate",
        "",
        f"- Bend: `{top.get('bend_id')}`",
        f"- Pair: `{top.get('candidate_pair')}`",
        f"- Score: `{top.get('flange_proximity_score')}`",
        f"- Reasons: `{top.get('reason_codes')}`",
        "",
        "## Runner-Up",
        "",
        f"- Bend: `{runner_up.get('bend_id')}`",
        f"- Pair: `{runner_up.get('candidate_pair')}`",
        f"- Score: `{runner_up.get('flange_proximity_score')}`",
        f"- Reasons: `{runner_up.get('reason_codes')}`",
        "",
        "## Candidates",
        "",
        "| Rank | Corridor | Bend | Pair | Score | Designated near | Designated median | Other near | Other median | Reasons |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, item in enumerate(proximity_report.get("candidates") or (), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{item.get('corridor_id')}`",
                    f"`{item.get('bend_id')}`",
                    f"`{item.get('candidate_pair')}`",
                    f"`{item.get('flange_proximity_score')}`",
                    f"`{item.get('designated_proximity_score')}`",
                    f"`{item.get('designated_mean_median_distance_mm')}`",
                    f"`{item.get('best_other_near_fraction')}`",
                    f"`{item.get('best_other_median_distance_mm')}`",
                    f"`{item.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This probe supplements adjacency boundary contact with distance-to-flange incidence. It is diagnostic-only until folded into the local MAP repair objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_flange_family_incidence_markdown(
    *,
    path: Path,
    family_report: Mapping[str, Any],
    incidence_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Flange Family Incidence Probe",
        "",
        f"Generated: `{incidence_report.get('generated_at')}`",
        "",
        "## Family Grouping",
        "",
        f"- Flanges: `{family_report.get('flange_count')}`",
        f"- Families: `{family_report.get('family_count')}`",
        f"- Normal threshold: `{family_report.get('normal_threshold_deg')}`",
        f"- Plane threshold: `{family_report.get('plane_threshold_mm')}`",
        "",
        "| Family | Flanges | d | Normal |",
        "|---|---|---:|---|",
    ]
    for family in family_report.get("families") or ():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{family.get('family_id')}`",
                    f"`{family.get('flange_ids')}`",
                    f"`{family.get('d')}`",
                    f"`{family.get('normal')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Incidence Decision",
            "",
            f"- Status: `{incidence_report.get('family_incidence_status')}`",
            f"- Hold reasons: `{incidence_report.get('hold_reasons')}`",
            f"- Candidates: `{incidence_report.get('candidate_count')}`",
            f"- Uncovered family pairs: `{incidence_report.get('uncovered_family_pair_count')}`",
            "",
            "| Corridor | Bend | Raw pair | Family pair | Family uncovered | Family leakage | Reasons |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for row in incidence_report.get("candidates") or ():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('corridor_id')}`",
                    f"`{row.get('bend_id')}`",
                    f"`{row.get('raw_pair')}`",
                    f"`{row.get('family_pair')}`",
                    f"`{row.get('family_pair_uncovered')}`",
                    f"`{row.get('family_leakage')}`",
                    f"`{row.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This is diagnostic-only. It tests whether raw uncovered flange pairs are actually aliases of already-covered physical flange families.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_conflict_sensitivity_markdown(
    *,
    path: Path,
    sensitivity_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Conflict Sensitivity Sweep",
        "",
        f"Generated: `{sensitivity_report.get('generated_at')}`",
        "",
        "## Summary",
        "",
        f"- Interpretation: `{sensitivity_report.get('interpretation_status')}`",
        f"- Dominant runs: `{sensitivity_report.get('dominant_run_count')}` / `{sensitivity_report.get('total_run_count')}`",
        f"- Status counts: `{sensitivity_report.get('status_counts')}`",
        f"- Dominant candidate counts: `{sensitivity_report.get('dominant_candidate_counts')}`",
        f"- Label costs: `{sensitivity_report.get('label_costs')}`",
        f"- Conflict weights: `{sensitivity_report.get('conflict_weights')}`",
        "",
        "## Runs",
        "",
        "| Label cost | Conflict weight | Status | +1 | +1 score | +2 | +2 score | +3 | +3 score |",
        "|---:|---:|---|---|---:|---|---:|---|---:|",
    ]
    for run in sensitivity_report.get("runs") or ():
        plus_one = run.get("best_plus_one") or {}
        plus_two = run.get("best_plus_two") or {}
        plus_three = run.get("best_plus_three") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run.get('label_cost')}`",
                    f"`{run.get('conflict_weight')}`",
                    f"`{run.get('selection_status')}`",
                    f"`{plus_one.get('bend_ids')}`",
                    f"`{plus_one.get('subset_score')}`",
                    f"`{plus_two.get('bend_ids')}`",
                    f"`{plus_two.get('subset_score')}`",
                    f"`{plus_three.get('bend_ids')}`",
                    f"`{plus_three.get('subset_score')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "If exact +1 dominance appears only under extreme label costs, the missing piece is a better object likelihood or corridor coverage model, not parameter tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_conflict_set_markdown(
    *,
    path: Path,
    conflict_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Conflict-Set Solver Probe",
        "",
        f"Generated: `{conflict_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{conflict_report.get('selection_status')}`",
        f"- Hold reasons: `{conflict_report.get('hold_reasons')}`",
        f"- Admissible candidates: `{conflict_report.get('admissible_candidate_count')}`",
        f"- Enumerated subsets: `{conflict_report.get('subset_count')}`",
        f"- Label cost: `{conflict_report.get('label_cost')}`",
        f"- Conflict weight: `{conflict_report.get('conflict_weight')}`",
        "",
        "## Best Subset By Added Count",
        "",
        "| Added | Bends | Score | Raw | Label penalty | Conflict penalty |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for added, subset in (conflict_report.get("best_by_added_count") or {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{added}`",
                    f"`{subset.get('bend_ids')}`",
                    f"`{subset.get('subset_score')}`",
                    f"`{subset.get('raw_candidate_score')}`",
                    f"`{subset.get('label_penalty')}`",
                    f"`{subset.get('conflict_penalty')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Subsets", "", "| Rank | Added | Bends | Score | Conflicts |", "|---:|---:|---|---:|---|"])
    for index, subset in enumerate(conflict_report.get("top_subsets") or (), start=1):
        conflict_ids = [
            f"{conflict.get('lhs')}~{conflict.get('rhs')}:{conflict.get('conflict_score')}"
            for conflict in subset.get("conflicts") or ()
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`+{subset.get('added_count')}`",
                    f"`{subset.get('bend_ids')}`",
                    f"`{subset.get('subset_score')}`",
                    f"`{conflict_ids}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This is still diagnostic-only. `hold_exact_promotion` means the next production solver change needs a real local MAP repair objective, not a direct count promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_local_selection_markdown(
    *,
    path: Path,
    selection_report: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended = dict(selection_report.get("recommended_candidate") or {})
    runner_up = dict(selection_report.get("runner_up_candidate") or {})
    lines = [
        "# F1 Exact-Blocker Local Selection Probe",
        "",
        f"Generated: `{selection_report.get('generated_at')}`",
        "",
        "## Decision",
        "",
        f"- Status: `{selection_report.get('selection_status')}`",
        f"- Target one-birth count: `{selection_report.get('target_one_birth_count')}`",
        f"- Hold reasons: `{selection_report.get('hold_reasons')}`",
        f"- Admissible candidates: `{selection_report.get('admissible_candidate_count')}`",
        f"- Rejected candidates: `{selection_report.get('rejected_candidate_count')}`",
        "",
        "## Recommended Candidate",
        "",
        f"- Bend: `{recommended.get('bend_id')}`",
        f"- Score: `{recommended.get('local_selection_score')}`",
        f"- Pair: `{recommended.get('incident_flange_ids')}`",
        f"- Atoms: `{recommended.get('atom_count')}`",
        f"- Nearest baseline: `{recommended.get('nearest_baseline')}`",
        f"- Candidate reasons: `{recommended.get('candidate_reason_codes')}`",
        "",
        "## Runner-Up",
        "",
        f"- Bend: `{runner_up.get('bend_id')}`",
        f"- Score: `{runner_up.get('local_selection_score')}`",
        f"- Gap: `{selection_report.get('top_score_gap')}`",
        "",
        "## Admissible Candidates",
        "",
        "| Rank | Bend | Score | Atoms | Pair | Nearest baseline | Reasons |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for index, item in enumerate(selection_report.get("admissible_candidates") or (), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{item.get('bend_id')}`",
                    f"`{item.get('local_selection_score')}`",
                    f"`{item.get('atom_count')}`",
                    f"`{item.get('incident_flange_ids')}`",
                    f"`{item.get('nearest_baseline')}`",
                    f"`{item.get('candidate_reason_codes')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Candidate Conflicts", ""])
    conflicts = list(selection_report.get("candidate_conflicts") or ())
    if conflicts:
        lines.extend(["| A | B | Score | Distance | Reasons |", "|---|---|---:|---:|---|"])
        for conflict in conflicts:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{conflict.get('lhs')}`",
                        f"`{conflict.get('rhs')}`",
                        f"`{conflict.get('conflict_score')}`",
                        f"`{conflict.get('centroid_distance_mm')}`",
                        f"`{conflict.get('reason_codes')}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No high-conflict admissible candidate pairs were found.")
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This probe is diagnostic only. `hold_exact_promotion` means the next solver change should implement a formal conflict-aware local add/split objective before F1 can promote an exact 12 for this scan.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_or_absolute(path: Optional[Path], markdown_dir: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(markdown_dir))
    except ValueError:
        return str(path)


def _write_markdown(
    *,
    path: Path,
    report: Mapping[str, Any],
    image_root: Optional[Path],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F1 Exact-Blocker Candidate Ranking",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "## Baseline vs Overcomplete",
        "",
        f"- Baseline `{report['baseline']['label']}`: range `{report['baseline']['bend_count_range']}`, owned `{report['baseline']['owned_bend_region_count']}`, energies `{report['baseline']['candidate_solution_energies']}`",
        f"- Overcomplete `{report['overcomplete']['label']}`: range `{report['overcomplete']['bend_count_range']}`, owned `{report['overcomplete']['owned_bend_region_count']}`, energies `{report['overcomplete']['candidate_solution_energies']}`",
        "",
        "## Ranked Extra-Candidate Regions",
        "",
        "| Rank | Bend | Score | Atoms | Pair | Angle | Best baseline match | IoU | Nearest distance | Reasons |",
        "|---:|---|---:|---:|---|---:|---|---:|---:|---|",
    ]
    for index, candidate in enumerate(report.get("ranked_candidates") or (), start=1):
        match = dict(candidate.get("best_baseline_match") or {})
        nearest = dict(candidate.get("nearest_baseline") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{candidate.get('bend_id')}`",
                    f"`{candidate.get('candidate_score')}`",
                    f"`{candidate.get('atom_count')}`",
                    f"`{candidate.get('incident_flange_ids')}`",
                    f"`{candidate.get('angle_deg')}`",
                    f"`{match.get('baseline_bend_id')}`",
                    f"`{match.get('atom_iou')}`",
                    f"`{nearest.get('centroid_distance_mm')}`",
                    f"`{candidate.get('reason_codes')}`",
                ]
            )
            + " |"
        )
    if image_root is not None:
        lines.extend(["", "## Top Candidate Focus Images", ""])
        for candidate in list(report.get("ranked_candidates") or ())[:top_n]:
            bend_id = str(candidate.get("bend_id"))
            focus = image_root / "subspan_17" / "render" / f"patch_graph_owned_region_{bend_id}.png"
            if focus.exists():
                lines.extend([f"### `{bend_id}`", "", f"![{bend_id}]({_relative_or_absolute(focus, path.parent)})", ""])
    lines.extend(
        [
            "",
            "## Use",
            "",
            "This is diagnostic evidence only. A high-scoring candidate should be reviewed as a possible missing bend, but exact-count promotion still requires a formal local split/birth acceptance rule.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(
    *,
    output_path: Path,
    report: Mapping[str, Any],
    image_root: Path,
    top_n: int,
) -> Optional[str]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None
    candidates = list(report.get("ranked_candidates") or ())[:top_n]
    cols = 2
    rows = max(1, math.ceil((len(candidates) + 2) / cols))
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(rows, cols)
    image_items: List[Tuple[str, Path]] = [
        ("baseline overview", image_root / "baseline_11" / "render" / "patch_graph_owned_region_overview.png"),
        ("subspan overview", image_root / "subspan_17" / "render" / "patch_graph_owned_region_overview.png"),
    ]
    for candidate in candidates:
        bend_id = str(candidate.get("bend_id"))
        score = candidate.get("candidate_score")
        image_items.append(
            (
                f"{bend_id} score={score}",
                image_root / "subspan_17" / "render" / f"patch_graph_owned_region_{bend_id}.png",
            )
        )
    for index, (title, image_path) in enumerate(image_items[: rows * cols]):
        ax = fig.add_subplot(grid[index // cols, index % cols])
        ax.axis("off")
        if image_path.exists():
            ax.imshow(mpimg.imread(str(image_path)))
            ax.set_title(title, fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"missing\\n{image_path.name}", ha="center", va="center")
    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank overcomplete F1 subspan candidates against a stable baseline decomposition.")
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--overcomplete-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--overcomplete-label", default="overcomplete")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    baseline_payload = _load_json(Path(args.baseline_json).expanduser().resolve())
    overcomplete_payload = _load_json(Path(args.overcomplete_json).expanduser().resolve())
    report = build_candidate_report(
        baseline_payload=baseline_payload,
        overcomplete_payload=overcomplete_payload,
        baseline_label=args.baseline_label,
        overcomplete_label=args.overcomplete_label,
    )
    selection_report = build_local_selection_report(
        baseline_payload=baseline_payload,
        overcomplete_payload=overcomplete_payload,
        candidate_report=report,
    )
    conflict_report = build_conflict_set_report(selection_report=selection_report)
    sensitivity_report = build_conflict_sensitivity_report(selection_report=selection_report)
    coverage_report = build_corridor_coverage_report(selection_report=selection_report)
    quotient_audit_report = build_quotient_corridor_audit_report(
        baseline_payload=baseline_payload,
        coverage_report=coverage_report,
    )
    uncovered_comparison_report = build_uncovered_corridor_comparison_report(
        overcomplete_payload=overcomplete_payload,
        quotient_audit_report=quotient_audit_report,
    )
    ownership_report = build_boundary_residual_ownership_report(
        baseline_payload=baseline_payload,
        overcomplete_payload=overcomplete_payload,
        quotient_audit_report=quotient_audit_report,
    )
    proximity_report = build_flange_proximity_incidence_report(
        overcomplete_payload=overcomplete_payload,
        quotient_audit_report=quotient_audit_report,
    )
    family_report = build_flange_family_report(payload=overcomplete_payload)
    family_incidence_report = build_flange_family_incidence_report(
        baseline_payload=baseline_payload,
        overcomplete_payload=overcomplete_payload,
        quotient_audit_report=quotient_audit_report,
        family_report=family_report,
    )
    promotion_evidence = build_family_residual_repair_promotion_evidence(
        baseline_payload=baseline_payload,
        overcomplete_payload=overcomplete_payload,
        family_incidence_report=family_incidence_report,
        ownership_report=ownership_report,
    )
    _write_json(output_dir / "exact_blocker_candidate_ranking.json", report)
    _write_json(output_dir / "exact_blocker_local_selection.json", selection_report)
    _write_json(output_dir / "exact_blocker_conflict_set_solver.json", conflict_report)
    _write_json(output_dir / "exact_blocker_conflict_sensitivity.json", sensitivity_report)
    _write_json(output_dir / "exact_blocker_corridor_coverage.json", coverage_report)
    _write_json(output_dir / "exact_blocker_quotient_corridor_audit.json", quotient_audit_report)
    _write_json(output_dir / "exact_blocker_uncovered_corridor_comparison.json", uncovered_comparison_report)
    _write_json(output_dir / "exact_blocker_boundary_residual_ownership.json", ownership_report)
    _write_json(output_dir / "exact_blocker_flange_proximity_incidence.json", proximity_report)
    _write_json(output_dir / "exact_blocker_flange_families.json", family_report)
    _write_json(output_dir / "exact_blocker_flange_family_incidence.json", family_incidence_report)
    _write_json(output_dir / "exact_blocker_family_residual_repair_promotion_evidence.json", promotion_evidence)
    _write_markdown(
        path=output_dir / "EXACT_BLOCKER_CANDIDATE_RANKING.md",
        report=report,
        image_root=image_root,
        top_n=max(1, int(args.top_n)),
    )
    _write_local_selection_markdown(
        path=output_dir / "EXACT_BLOCKER_LOCAL_SELECTION.md",
        selection_report=selection_report,
    )
    _write_conflict_set_markdown(
        path=output_dir / "EXACT_BLOCKER_CONFLICT_SET_SOLVER.md",
        conflict_report=conflict_report,
    )
    _write_conflict_sensitivity_markdown(
        path=output_dir / "EXACT_BLOCKER_CONFLICT_SENSITIVITY.md",
        sensitivity_report=sensitivity_report,
    )
    _write_corridor_coverage_markdown(
        path=output_dir / "EXACT_BLOCKER_CORRIDOR_COVERAGE.md",
        coverage_report=coverage_report,
    )
    _write_quotient_corridor_audit_markdown(
        path=output_dir / "EXACT_BLOCKER_QUOTIENT_CORRIDOR_AUDIT.md",
        audit_report=quotient_audit_report,
    )
    _write_uncovered_corridor_comparison_markdown(
        path=output_dir / "EXACT_BLOCKER_UNCOVERED_CORRIDOR_COMPARISON.md",
        comparison_report=uncovered_comparison_report,
    )
    _write_boundary_residual_ownership_markdown(
        path=output_dir / "EXACT_BLOCKER_BOUNDARY_RESIDUAL_OWNERSHIP.md",
        ownership_report=ownership_report,
    )
    _write_flange_proximity_incidence_markdown(
        path=output_dir / "EXACT_BLOCKER_FLANGE_PROXIMITY_INCIDENCE.md",
        proximity_report=proximity_report,
    )
    _write_flange_family_incidence_markdown(
        path=output_dir / "EXACT_BLOCKER_FLANGE_FAMILY_INCIDENCE.md",
        family_report=family_report,
        incidence_report=family_incidence_report,
    )
    contact_sheet = None
    if image_root is not None:
        contact_sheet = _write_contact_sheet(
            output_path=output_dir / "exact_blocker_candidate_contact_sheet_1080p.png",
            report=report,
            image_root=image_root,
            top_n=max(1, int(args.top_n)),
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "json": str(output_dir / "exact_blocker_candidate_ranking.json"),
                "markdown": str(output_dir / "EXACT_BLOCKER_CANDIDATE_RANKING.md"),
                "selection_json": str(output_dir / "exact_blocker_local_selection.json"),
                "selection_markdown": str(output_dir / "EXACT_BLOCKER_LOCAL_SELECTION.md"),
                "conflict_set_json": str(output_dir / "exact_blocker_conflict_set_solver.json"),
                "conflict_set_markdown": str(output_dir / "EXACT_BLOCKER_CONFLICT_SET_SOLVER.md"),
                "conflict_sensitivity_json": str(output_dir / "exact_blocker_conflict_sensitivity.json"),
                "conflict_sensitivity_markdown": str(output_dir / "EXACT_BLOCKER_CONFLICT_SENSITIVITY.md"),
                "corridor_coverage_json": str(output_dir / "exact_blocker_corridor_coverage.json"),
                "corridor_coverage_markdown": str(output_dir / "EXACT_BLOCKER_CORRIDOR_COVERAGE.md"),
                "quotient_corridor_audit_json": str(output_dir / "exact_blocker_quotient_corridor_audit.json"),
                "quotient_corridor_audit_markdown": str(output_dir / "EXACT_BLOCKER_QUOTIENT_CORRIDOR_AUDIT.md"),
                "uncovered_corridor_comparison_json": str(output_dir / "exact_blocker_uncovered_corridor_comparison.json"),
                "uncovered_corridor_comparison_markdown": str(output_dir / "EXACT_BLOCKER_UNCOVERED_CORRIDOR_COMPARISON.md"),
                "boundary_residual_ownership_json": str(output_dir / "exact_blocker_boundary_residual_ownership.json"),
                "boundary_residual_ownership_markdown": str(output_dir / "EXACT_BLOCKER_BOUNDARY_RESIDUAL_OWNERSHIP.md"),
                "flange_proximity_incidence_json": str(output_dir / "exact_blocker_flange_proximity_incidence.json"),
                "flange_proximity_incidence_markdown": str(output_dir / "EXACT_BLOCKER_FLANGE_PROXIMITY_INCIDENCE.md"),
                "flange_families_json": str(output_dir / "exact_blocker_flange_families.json"),
                "flange_family_incidence_json": str(output_dir / "exact_blocker_flange_family_incidence.json"),
                "flange_family_incidence_markdown": str(output_dir / "EXACT_BLOCKER_FLANGE_FAMILY_INCIDENCE.md"),
                "family_residual_repair_promotion_evidence_json": str(output_dir / "exact_blocker_family_residual_repair_promotion_evidence.json"),
                "contact_sheet": contact_sheet,
                "selection_status": selection_report.get("selection_status"),
                "conflict_set_status": conflict_report.get("selection_status"),
                "conflict_sensitivity_status": sensitivity_report.get("interpretation_status"),
                "corridor_coverage_status": coverage_report.get("coverage_status"),
                "quotient_corridor_audit_status": quotient_audit_report.get("audit_status"),
                "uncovered_corridor_comparison_status": uncovered_comparison_report.get("comparison_status"),
                "boundary_residual_ownership_status": ownership_report.get("ownership_status"),
                "flange_proximity_incidence_status": proximity_report.get("proximity_status"),
                "flange_family_incidence_status": family_incidence_report.get("family_incidence_status"),
                "family_residual_repair_eligible": promotion_evidence.get("family_residual_repair_eligible"),
                "family_residual_repair_blockers": promotion_evidence.get("blockers"),
                "recommended_candidate": (selection_report.get("recommended_candidate") or {}).get("bend_id"),
                "hold_reasons": selection_report.get("hold_reasons"),
                "conflict_set_hold_reasons": conflict_report.get("hold_reasons"),
                "top_candidates": [
                    {
                        "bend_id": item.get("bend_id"),
                        "score": item.get("candidate_score"),
                        "atom_count": item.get("atom_count"),
                        "reasons": item.get("reason_codes"),
                    }
                    for item in (report.get("ranked_candidates") or [])[: max(1, int(args.top_n))]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
