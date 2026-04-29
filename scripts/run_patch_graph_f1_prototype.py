#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from bend_inspection_pipeline import load_bend_runtime_config
from patch_graph_latent_decomposition import (
    DecompositionResult,
    OwnedBendRegion,
    build_owned_region_marker_admissibility,
    render_decomposition_artifacts,
    run_patch_graph_latent_decomposition,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_range(raw_range: Any, exact_count: Optional[int]) -> List[int]:
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        try:
            low = int(raw_range[0])
            high = int(raw_range[1])
            return [min(low, high), max(low, high)]
        except Exception:
            pass
    if exact_count is not None:
        return [int(exact_count), int(exact_count)]
    return [0, 0]


def _range_contains(expected: Optional[int], count_range: Sequence[int]) -> Optional[bool]:
    if expected is None or len(count_range) < 2:
        return None
    return bool(int(count_range[0]) <= int(expected) <= int(count_range[1]))


def _range_width(count_range: Sequence[int]) -> Optional[int]:
    if len(count_range) < 2:
        return None
    return int(count_range[1]) - int(count_range[0])


def _exact_if_singleton(exact_count: Optional[int], count_range: Sequence[int]) -> Optional[int]:
    if exact_count is None or len(count_range) < 2:
        return None
    exact = int(exact_count)
    return exact if int(count_range[0]) == exact and int(count_range[1]) == exact else None


def _compare(control_range: Sequence[int], candidate_range: Sequence[int], expected_count: Optional[int]) -> Dict[str, Any]:
    control_width = _range_width(control_range)
    candidate_width = _range_width(candidate_range)
    candidate_contains_truth = _range_contains(expected_count, candidate_range)
    return {
        "control_range": list(control_range),
        "candidate_range": list(candidate_range),
        "control_width": control_width,
        "candidate_width": candidate_width,
        "tightened": bool(
            control_width is not None
            and candidate_width is not None
            and candidate_width < control_width
            and candidate_contains_truth
        ),
        "broadened": bool(control_width is not None and candidate_width is not None and candidate_width > control_width),
    }


def _routed_candidate_decision(
    *,
    route_payload: Dict[str, Any],
    raw_exact: int,
    raw_range: Sequence[int],
    control_exact: Optional[int],
    control_range: Sequence[int],
    promotion_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_range = list(raw_range)
    control_range = list(control_range)
    if not route_payload.get("applied"):
        return {
            "exact_bend_count": _exact_if_singleton(raw_exact, raw_range),
            "bend_count_range": raw_range,
            "candidate_source": "raw_f1",
            "guard_reason": None,
        }
    if route_payload.get("solver_profile") == "observe_only":
        return {
            "exact_bend_count": control_exact,
            "bend_count_range": control_range,
            "candidate_source": "control_guard",
            "guard_reason": "route_observe_only",
        }
    if (
        route_payload.get("solver_profile") == "control_exact_tiebreak"
        and control_exact is not None
        and len(raw_range) >= 2
        and int(raw_range[1]) == int(control_exact)
    ):
        return {
            "exact_bend_count": control_exact,
            "bend_count_range": control_range,
            "candidate_source": "control_aligned_f1_tiebreak" if raw_exact == int(control_exact) else "control_guard",
            "guard_reason": None if raw_exact == int(control_exact) else "f1_range_outside_narrow_control",
        }
    if route_payload.get("route_id") == "partial_sparse_seed_support_birth" and len(raw_range) >= 2:
        upper = int(raw_range[1])
        if len(control_range) >= 2:
            upper = max(upper, int(control_range[1]))
        envelope = [int(raw_range[0]), upper + 1]
        return {
            "exact_bend_count": None,
            "bend_count_range": envelope,
            "candidate_source": "partial_sparse_seed_f1_envelope",
            "guard_reason": "scan_limited_sparse_seed_blocks_exact_promotion",
        }
    if route_payload.get("scan_family") == "partial_scan_limited" or route_payload.get("route_id") == "partial_scan_limited":
        if len(raw_range) >= 2 and len(control_range) >= 2:
            raw_low, raw_high = int(raw_range[0]), int(raw_range[1])
            control_low, control_high = int(control_range[0]), int(control_range[1])
            control_width = control_high - control_low
            if raw_high < control_low:
                envelope = [raw_low, control_low]
            elif raw_low > control_high and control_width > 0:
                envelope = [control_low, control_high]
            else:
                lower = control_low
                upper = control_high
                if raw_low < control_low:
                    lower = raw_low
                elif control_width >= 4 and raw_low > control_low:
                    lower = raw_low
                if raw_high > control_high and control_width == 0:
                    upper = raw_high
                elif raw_low == raw_high and control_width >= 8 and control_low <= raw_low <= control_high:
                    upper = min(control_high, raw_high + 3)
                envelope = [lower, upper]
        else:
            envelope = raw_range or control_range
        return {
            "exact_bend_count": None,
            "bend_count_range": envelope,
            "candidate_source": "partial_observed_control_f1_envelope",
            "guard_reason": "scan_limited_blocks_full_count_promotion",
        }
    if control_exact is not None and raw_exact != int(control_exact):
        return {
            "exact_bend_count": control_exact,
            "bend_count_range": control_range,
            "candidate_source": "control_guard",
            "guard_reason": "protected_control_exact",
        }
    if len(raw_range) >= 2 and len(control_range) >= 2:
        ranges_overlap = int(raw_range[0]) <= int(control_range[1]) and int(control_range[0]) <= int(raw_range[1])
        if not ranges_overlap:
            return {
                "exact_bend_count": control_exact,
                "bend_count_range": control_range,
                "candidate_source": "control_guard",
                "guard_reason": "f1_range_disjoint_from_control",
            }
        raw_subset_of_control = int(control_range[0]) <= int(raw_range[0]) <= int(raw_range[1]) <= int(control_range[1])
        raw_width = _range_width(raw_range)
        control_width = _range_width(control_range)
        if (
            raw_width == 0
            and control_width is not None
            and control_width > 0
            and control_exact is None
            and int(control_range[0]) <= int(raw_range[0]) < int(control_range[1])
        ):
            if _promotion_evidence_accepts_raw_singleton(
                promotion_evidence=promotion_evidence,
                raw_exact=raw_exact,
                raw_range=raw_range,
                control_range=control_range,
            ):
                return {
                    "exact_bend_count": int(raw_range[0]),
                    "bend_count_range": [int(raw_range[0]), int(raw_range[0])],
                    "candidate_source": "count_sweep_promoted_f1",
                    "guard_reason": "count_sweep_upper_tail_rejected",
                }
            family_residual_promoted_count = _family_residual_repair_promoted_count(
                promotion_evidence=promotion_evidence,
                raw_exact=raw_exact,
                raw_range=raw_range,
                control_range=control_range,
            )
            if family_residual_promoted_count is not None:
                return {
                    "exact_bend_count": family_residual_promoted_count,
                    "bend_count_range": [family_residual_promoted_count, family_residual_promoted_count],
                    "candidate_source": "family_residual_repair_promoted_f1",
                    "guard_reason": "single_uncovered_flange_family_residual_candidate",
                }
            if control_width > 2 and int(raw_range[0]) > int(control_range[0]):
                return {
                    "exact_bend_count": control_exact,
                    "bend_count_range": control_range,
                    "candidate_source": "control_guard",
                    "guard_reason": "upper_tail_singleton_on_wide_control",
                }
            return {
                "exact_bend_count": None,
                "bend_count_range": [int(raw_range[0]), int(control_range[1])],
                "candidate_source": "control_constrained_f1",
                "guard_reason": "singleton_below_control_upper",
            }
        if (
            raw_width == 0
            and control_width is not None
            and control_width > 0
            and control_exact is None
            and int(raw_range[0]) < int(control_range[1])
        ):
            return {
                "exact_bend_count": control_exact,
                "bend_count_range": control_range,
                "candidate_source": "control_guard",
                "guard_reason": "singleton_below_control_upper",
            }
        intersection = [max(int(raw_range[0]), int(control_range[0])), min(int(raw_range[1]), int(control_range[1]))]
        intersection_width = intersection[1] - intersection[0]
        if not raw_subset_of_control and control_width is not None and control_width <= 2:
            if intersection_width >= 0 and intersection_width < control_width:
                return {
                    "exact_bend_count": intersection[0] if intersection_width == 0 else None,
                    "bend_count_range": intersection,
                    "candidate_source": "control_intersection_f1",
                    "guard_reason": "f1_intersection_with_narrow_control",
                }
            return {
                "exact_bend_count": control_exact,
                "bend_count_range": control_range,
                "candidate_source": "control_guard",
                "guard_reason": "f1_range_outside_narrow_control",
            }
        if (
            route_payload.get("route_id") == "small_tight_bend_recovery"
            and raw_subset_of_control
            and control_width is not None
            and control_width > 2
            and int(raw_range[0]) > int(control_range[0])
        ):
            return {
                "exact_bend_count": control_exact,
                "bend_count_range": control_range,
                "candidate_source": "control_guard",
                "guard_reason": "upper_tail_f1_on_wide_control",
            }
        if (
            raw_subset_of_control
            and control_width is not None
            and control_width > 2
            and int(raw_range[0]) > int(control_range[0])
        ):
            return {
                "exact_bend_count": control_exact,
                "bend_count_range": control_range,
                "candidate_source": "control_guard",
                "guard_reason": "upper_tail_f1_on_wide_control",
            }
        if raw_subset_of_control and raw_width is not None and control_width is not None and raw_width <= control_width:
            return {
                "exact_bend_count": _exact_if_singleton(raw_exact, raw_range),
                "bend_count_range": raw_range,
                "candidate_source": "raw_f1",
                "guard_reason": None,
            }
    return {
        "exact_bend_count": _exact_if_singleton(raw_exact, raw_range),
        "bend_count_range": raw_range,
        "candidate_source": "raw_f1",
        "guard_reason": None,
    }


def _promotion_evidence_accepts_raw_singleton(
    *,
    promotion_evidence: Optional[Dict[str, Any]],
    raw_exact: Optional[int],
    raw_range: Sequence[int],
    control_range: Sequence[int],
) -> bool:
    if not promotion_evidence or not promotion_evidence.get("exact_promotion_eligible"):
        return False
    if raw_exact is None or len(raw_range) < 2 or len(control_range) < 2:
        return False
    exact = int(raw_exact)
    if [int(raw_range[0]), int(raw_range[1])] != [exact, exact]:
        return False
    if not (int(control_range[0]) <= exact <= int(control_range[1])):
        return False
    return exact == int(promotion_evidence.get("promoted_exact_count"))


def _family_residual_repair_promoted_count(
    *,
    promotion_evidence: Optional[Dict[str, Any]],
    raw_exact: Optional[int],
    raw_range: Sequence[int],
    control_range: Sequence[int],
) -> Optional[int]:
    if not promotion_evidence or not promotion_evidence.get("family_residual_repair_eligible"):
        return None
    if raw_exact is None or len(raw_range) < 2 or len(control_range) < 2:
        return None
    exact = int(raw_exact)
    if [int(raw_range[0]), int(raw_range[1])] != [exact, exact]:
        return None
    promoted = promotion_evidence.get("promoted_exact_count")
    if promoted is None:
        return None
    promoted_count = int(promoted)
    if promoted_count != exact + 1:
        return None
    if promoted_count != int(control_range[1]):
        return None
    if not (int(control_range[0]) <= exact < promoted_count):
        return None

    checks = dict(promotion_evidence.get("checks") or {})
    required_checks = (
        "single_uncovered_flange_family_pair",
        "candidate_owns_baseline_residual",
        "raw_competitors_family_covered",
    )
    if not all(bool(checks.get(name)) for name in required_checks):
        return None
    residual_fraction = float(checks.get("baseline_residual_fraction", 0.0))
    if residual_fraction < 0.75:
        return None
    return promoted_count


def _count_sweep_promotion_evidence(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    variants = list(summary_payload.get("variants") or ())
    baseline = next((dict(item) for item in variants if item.get("variant_id") == "baseline"), None)
    blockers: List[str] = []
    reasons: List[str] = []
    if not baseline:
        blockers.append("missing_baseline")
        return {"exact_promotion_eligible": False, "blockers": blockers, "reasons": reasons}
    raw_range = list(baseline.get("raw_bend_count_range") or ())
    raw_exact = baseline.get("raw_exact_bend_count")
    if raw_exact is None or len(raw_range) < 2 or int(raw_range[0]) != int(raw_range[1]) or int(raw_range[0]) != int(raw_exact):
        blockers.append("baseline_not_exact_singleton")
    else:
        reasons.append("baseline_exact_singleton")
    if not baseline.get("promotion_candidate"):
        blockers.append("baseline_not_promotion_candidate")
    else:
        reasons.append("baseline_promotion_candidate")
    baseline_energy = baseline.get("best_energy")
    if baseline_energy is None:
        blockers.append("missing_baseline_energy")
    upper_variants = [
        dict(item)
        for item in variants
        if item.get("variant_id") != "baseline"
        and item.get("raw_map_bend_count") is not None
        and raw_exact is not None
        and int(item.get("raw_map_bend_count")) > int(raw_exact)
    ]
    if not upper_variants:
        blockers.append("missing_upper_tail_variants")
    else:
        reasons.append("upper_tail_variants_present")
    deltas = [
        float(item.get("energy_delta_vs_baseline"))
        for item in upper_variants
        if item.get("energy_delta_vs_baseline") is not None
    ]
    if not deltas:
        blockers.append("missing_upper_tail_energy_deltas")
    elif min(deltas) < 100.0:
        blockers.append("upper_tail_energy_gap_too_small")
    else:
        reasons.append("upper_tail_energy_gap_large")
    immediate_next = [
        item for item in upper_variants
        if raw_exact is not None and int(item.get("raw_map_bend_count")) == int(raw_exact) + 1
    ]
    if not immediate_next:
        blockers.append("missing_immediate_next_count_probe")
    else:
        reasons.append("immediate_next_count_probe_present")
    uncapped = next((dict(item) for item in variants if item.get("variant_id") == "uncapped"), None)
    if not uncapped:
        blockers.append("missing_uncapped_probe")
    elif uncapped.get("energy_delta_vs_baseline") is None or float(uncapped.get("energy_delta_vs_baseline")) < 500.0:
        blockers.append("uncapped_energy_gap_too_small")
    else:
        reasons.append("uncapped_probe_strongly_worse")
    return {
        "exact_promotion_eligible": not blockers,
        "promoted_exact_count": None if raw_exact is None else int(raw_exact),
        "part_key": summary_payload.get("part_key"),
        "min_upper_tail_energy_delta": None if not deltas else round(min(deltas), 6),
        "blockers": blockers,
        "reasons": reasons,
    }


def _load_count_sweep_promotion_evidence(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = _load_json(path)
        part_key = str(payload.get("part_key") or "")
        if not part_key:
            continue
        evidence[part_key] = _count_sweep_promotion_evidence(payload)
        evidence[part_key]["source_path"] = str(path)
    return evidence


def _load_promotion_evidence(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = _load_json(path)
        payloads = payload.get("evidence") if isinstance(payload.get("evidence"), list) else None
        if payloads is None:
            payloads = payload.get("results") if isinstance(payload.get("results"), list) else None
        if payloads is None:
            payloads = [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            part_key = str(item.get("part_key") or payload.get("part_key") or "")
            if not part_key:
                continue
            merged = dict(item)
            merged.setdefault("source_path", str(path))
            evidence[part_key] = merged
    return evidence


def _owned_bend_region_from_payload(payload: Dict[str, Any]) -> OwnedBendRegion:
    span_endpoints = payload.get("span_endpoints") or ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return OwnedBendRegion(
        bend_id=str(payload.get("bend_id") or "OB_PROMOTED"),
        bend_class=str(payload.get("bend_class") or "transition_only"),
        incident_flange_ids=tuple(str(value) for value in payload.get("incident_flange_ids") or ()),
        canonical_bend_key=str(payload.get("canonical_bend_key") or payload.get("bend_id") or "promoted_region"),
        owned_atom_ids=tuple(str(value) for value in payload.get("owned_atom_ids") or ()),
        anchor=tuple(float(value) for value in payload.get("anchor") or (0.0, 0.0, 0.0)),
        axis_direction=tuple(float(value) for value in payload.get("axis_direction") or (1.0, 0.0, 0.0)),
        support_centroid=tuple(float(value) for value in payload.get("support_centroid") or payload.get("anchor") or (0.0, 0.0, 0.0)),
        span_endpoints=(
            tuple(float(value) for value in span_endpoints[0]),
            tuple(float(value) for value in span_endpoints[1]),
        ),
        angle_deg=float(payload.get("angle_deg") or 0.0),
        radius_mm=None if payload.get("radius_mm") is None else float(payload.get("radius_mm")),
        visibility_state=str(payload.get("visibility_state") or "internal"),
        support_mass=float(payload.get("support_mass") or 0.0),
        admissible=bool(payload.get("admissible", True)),
        admissibility_reasons=tuple(str(value) for value in payload.get("admissibility_reasons") or ()),
        post_bend_class=str(payload.get("post_bend_class") or payload.get("bend_class") or "transition_only"),
        debug_confidence=float(payload.get("debug_confidence", 1.0)),
    )


def _next_owned_bend_id(existing_regions: Sequence[OwnedBendRegion]) -> str:
    max_index = 0
    for region in existing_regions:
        bend_id = str(region.bend_id)
        if bend_id.startswith("OB") and bend_id[2:].isdigit():
            max_index = max(max_index, int(bend_id[2:]))
    return f"OB{max_index + 1}"


def _supplemental_owned_bend_regions_from_evidence(
    *,
    promotion_evidence: Optional[Dict[str, Any]],
    existing_regions: Sequence[OwnedBendRegion],
) -> Tuple[OwnedBendRegion, ...]:
    if not promotion_evidence:
        return tuple()
    payloads = promotion_evidence.get("supplemental_owned_bend_regions")
    if payloads is None and promotion_evidence.get("supplemental_owned_bend_region") is not None:
        payloads = [promotion_evidence.get("supplemental_owned_bend_region")]
    if not isinstance(payloads, list):
        return tuple()

    existing_keys = {str(region.canonical_bend_key) for region in existing_regions}
    existing_ids = {str(region.bend_id) for region in existing_regions}
    supplemental: List[OwnedBendRegion] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        region = _owned_bend_region_from_payload(payload)
        if str(region.canonical_bend_key) in existing_keys:
            continue
        if str(region.bend_id) in existing_ids:
            region = replace(region, bend_id=_next_owned_bend_id([*existing_regions, *supplemental]))
        supplemental.append(region)
        existing_keys.add(str(region.canonical_bend_key))
        existing_ids.add(str(region.bend_id))
    return tuple(supplemental)


def _result_with_supplemental_owned_regions(
    *,
    result: DecompositionResult,
    supplemental_regions: Sequence[OwnedBendRegion],
) -> DecompositionResult:
    if not supplemental_regions:
        return result
    owned_regions = tuple([*result.owned_bend_regions, *supplemental_regions])
    admissible_count = sum(1 for region in owned_regions if region.admissible)
    return replace(
        result,
        owned_bend_regions=owned_regions,
        exact_bend_count=admissible_count,
        bend_count_range=(admissible_count, admissible_count),
        candidate_solution_counts=tuple([*result.candidate_solution_counts, admissible_count]),
        repair_diagnostics={
            **dict(result.repair_diagnostics or {}),
            "supplemental_owned_bend_region_count": len(supplemental_regions),
            "supplemental_owned_bend_region_source": "promotion_evidence",
        },
    )


def _raw_f1_promotion_diagnostic(
    *,
    route_payload: Dict[str, Any],
    raw_exact: Optional[int],
    raw_range: Sequence[int],
    control_range: Sequence[int],
    solution_energies: Sequence[float],
    solution_counts: Sequence[int],
    owned_bend_region_count: int,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_range = list(raw_range)
    control_range = list(control_range)
    reasons: List[str] = []
    blockers: List[str] = []
    raw_width = _range_width(raw_range)
    control_width = _range_width(control_range)
    energy_values = [float(value) for value in solution_energies]
    sorted_energies = sorted(energy_values)
    energy_gap = None
    if len(sorted_energies) >= 2:
        energy_gap = float(sorted_energies[1] - sorted_energies[0])
    elif len(sorted_energies) == 1:
        energy_gap = float("inf")
    singleton = bool(raw_exact is not None and raw_width == 0 and len(raw_range) >= 2 and int(raw_range[0]) == int(raw_range[1]))
    if singleton:
        reasons.append("raw_f1_singleton")
    else:
        blockers.append("raw_f1_not_singleton")
    within_control = bool(
        singleton
        and len(control_range) >= 2
        and int(control_range[0]) <= int(raw_range[0]) <= int(control_range[1])
    )
    if within_control:
        reasons.append("raw_singleton_within_control")
    else:
        blockers.append("raw_singleton_outside_control")
    count_stable = bool(solution_counts and len({int(value) for value in solution_counts}) == 1)
    if count_stable:
        reasons.append("nearby_decompositions_same_count")
    else:
        blockers.append("nearby_decompositions_change_count")
    has_large_energy_gap = bool(energy_gap is not None and energy_gap >= 50.0)
    if has_large_energy_gap:
        reasons.append("large_energy_gap")
    else:
        blockers.append("energy_gap_too_small")
    if singleton and owned_bend_region_count == int(raw_range[0]):
        reasons.append("owned_region_count_matches_singleton")
    else:
        blockers.append("owned_region_count_mismatch")

    route_id = str(route_payload.get("route_id") or "")
    if dict(solver_options or {}).get("upper_tail_probe"):
        blockers.append("diagnostic_upper_tail_probe_run")
    allowed_exact_promotion_routes = {"small_tight_bend_recovery"}
    if route_id in allowed_exact_promotion_routes:
        reasons.append("route_allows_upper_tail_review")
    else:
        blockers.append("route_not_allowed_for_exact_upper_tail_promotion")

    stable_singleton_for_review = bool(
        route_payload.get("applied")
        and singleton
        and within_control
        and count_stable
        and owned_bend_region_count == int(raw_range[0])
        and route_id in allowed_exact_promotion_routes
        and control_width is not None
        and control_width >= 3
        and not dict(solver_options or {}).get("upper_tail_probe")
    )
    if stable_singleton_for_review:
        reasons.append("stable_singleton_manual_review_candidate")
    candidate = stable_singleton_for_review
    if candidate:
        recommendation = "manual_review_exact_promotion_candidate"
    else:
        recommendation = "do_not_promote_exact"
    return {
        "candidate": candidate,
        "recommendation": recommendation,
        "route_id": route_id,
        "raw_singleton": singleton,
        "raw_within_control": within_control,
        "nearby_count_stable": count_stable,
        "energy_gap_to_next": None if energy_gap is None or energy_gap == float("inf") else round(float(energy_gap), 6),
        "large_energy_gap": has_large_energy_gap,
        "promotion_safety": "manual_review_only"
        if candidate and not has_large_energy_gap
        else "manual_review_with_large_energy_gap"
        if candidate
        else "not_candidate",
        "owned_region_count_matches_singleton": bool(singleton and owned_bend_region_count == int(raw_range[0])),
        "upper_tail_probe": bool(dict(solver_options or {}).get("upper_tail_probe")),
        "reasons": reasons,
        "blockers": blockers,
    }


def _entry_result_payload(
    *,
    entry: Dict[str, Any],
    result: DecompositionResult,
    render_info: Optional[Dict[str, Any]],
    promotion_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expectation = dict(entry.get("expectation") or {})
    expected_bend_count = expectation.get("expected_bend_count")
    candidate_range = list(result.bend_count_range)
    control_range = list(result.current_zero_shot_count_range)
    raw_map_count = int(result.exact_bend_count)
    raw_exact = _exact_if_singleton(raw_map_count, candidate_range)
    control_exact = result.current_zero_shot_exact_count
    route_payload = dict((result.repair_diagnostics or {}).get("scan_family_route") or {})
    raw_f1_candidate = {
        "map_bend_count": raw_map_count,
        "exact_bend_count": raw_exact,
        "bend_count_range": candidate_range,
        "owned_bend_region_count": len(result.owned_bend_regions),
        "solution_energies": list(result.candidate_solution_energies),
        "solution_counts": list(getattr(result, "candidate_solution_counts", (raw_map_count,))),
    }
    promotion_diagnostic = _raw_f1_promotion_diagnostic(
        route_payload=route_payload,
        raw_exact=raw_exact,
        raw_range=candidate_range,
        control_range=control_range,
        solution_energies=result.candidate_solution_energies,
        solution_counts=getattr(result, "candidate_solution_counts", (raw_map_count,)),
        owned_bend_region_count=len(result.owned_bend_regions),
        solver_options=dict((result.repair_diagnostics or {}).get("solver_options") or {}),
    )
    decision = _routed_candidate_decision(
        route_payload=route_payload,
        raw_exact=raw_map_count,
        raw_range=candidate_range,
        control_exact=control_exact,
        control_range=control_range,
        promotion_evidence=promotion_evidence,
    )
    if (
        decision["candidate_source"] == "control_aligned_f1_tiebreak"
        and decision["exact_bend_count"] is not None
        and len(result.owned_bend_regions) == int(decision["exact_bend_count"])
    ):
        raw_f1_candidate["exact_bend_count"] = int(decision["exact_bend_count"])
        raw_f1_candidate["bend_count_range"] = [int(decision["exact_bend_count"]), int(decision["exact_bend_count"])]
        promotion_diagnostic["reasons"] = list(promotion_diagnostic.get("reasons") or []) + ["control_exact_tiebreak_selected_owned_regions"]
        promotion_diagnostic["blockers"] = [
            blocker
            for blocker in list(promotion_diagnostic.get("blockers") or [])
            if blocker not in {"raw_f1_not_singleton", "raw_singleton_outside_control", "owned_region_count_mismatch"}
        ]
    candidate_exact = decision["exact_bend_count"]
    candidate_range = decision["bend_count_range"]
    return {
        "part_key": entry.get("part_key"),
        "scan_path": entry.get("scan_path"),
        "expectation": expectation,
        "control": {
            "exact_bend_count": control_exact,
            "bend_count_range": control_range,
            "profile_family": result.current_zero_shot_result.get("profile_family"),
        },
        "candidate": {
            "exact_bend_count": candidate_exact,
            "bend_count_range": candidate_range,
            "candidate_source": decision["candidate_source"],
            "guard_reason": decision["guard_reason"],
            "owned_bend_region_count": len(result.owned_bend_regions),
            "solution_energies": list(result.candidate_solution_energies),
        },
        "raw_f1_candidate": raw_f1_candidate,
        "raw_f1_promotion_diagnostic": promotion_diagnostic,
        "promotion_evidence": promotion_evidence or None,
        "count_sweep_promotion_evidence": promotion_evidence or None,
        "scan_family_route": route_payload or None,
        "metrics": {
                "control_exact_match": None if expected_bend_count is None or control_exact is None else bool(int(control_exact) == int(expected_bend_count)),
                "candidate_exact_match": None if expected_bend_count is None or candidate_exact is None else bool(int(candidate_exact) == int(expected_bend_count)),
                "control_range_contains_truth": _range_contains(expected_bend_count, control_range),
                "candidate_range_contains_truth": _range_contains(expected_bend_count, candidate_range),
                "range_delta": _compare(control_range, candidate_range, expected_bend_count),
            },
        "artifacts": render_info or {},
    }


def _owned_region_marker_admissibility(result: DecompositionResult) -> Dict[str, Any]:
    return build_owned_region_marker_admissibility(result.owned_bend_regions, atom_graph=result.atom_graph)


def _attach_candidate_render_semantics(
    payload: Dict[str, Any],
    *,
    render_info: Optional[Dict[str, Any]],
    rendered_owned_region_count: int,
    marker_admissibility: Optional[Dict[str, Any]] = None,
) -> None:
    candidate = dict(payload.get("candidate") or {})
    candidate_source = str(candidate.get("candidate_source") or "")
    candidate_exact = candidate.get("exact_bend_count")
    candidate_range = list(candidate.get("bend_count_range") or ())
    singleton_exact = (
        candidate_exact is not None
        and len(candidate_range) >= 2
        and int(candidate_range[0]) == int(candidate_range[1]) == int(candidate_exact)
    )
    raw_marker_sources = {
        "raw_f1",
        "count_sweep_promoted_f1",
        "family_residual_repair_promoted_f1",
        "control_aligned_f1_tiebreak",
    }
    visual_blockers: List[str] = []
    marker_payload = dict(marker_admissibility or {})
    marker_owned_count = int(marker_payload.get("marker_admissible_owned_region_count", rendered_owned_region_count))
    render_suppressed_count = int(
        marker_payload.get(
            "marker_suppressed_owned_region_count",
            0 if not render_info else int(render_info.get("render_suppressed_owned_region_count", 0)),
        )
    )
    render_suppressed_ids = list(
        marker_payload.get(
            "marker_suppressed_region_ids",
            [] if not render_info else list(render_info.get("render_suppressed_region_ids") or []),
        )
    )
    render_suppression_details = list(
        marker_payload.get(
            "marker_suppression_details",
            [] if not render_info else list(render_info.get("render_suppression_details") or []),
        )
    )
    render_admissible_exact = marker_owned_count
    raw_markers_match_accepted = bool(singleton_exact and marker_owned_count == int(candidate_exact))
    if render_info is None:
        visual_blockers.append("render_not_generated")
    if singleton_exact and marker_owned_count != int(candidate_exact):
        visual_blockers.append("render_marker_count_mismatch")
    if render_suppressed_count > 0:
        visual_blockers.append("owned_region_markers_suppressed")
    if candidate_source not in raw_marker_sources:
        visual_blockers.append("candidate_source_not_raw_owned_region_renderable")
    marker_blockers: List[str] = []
    marker_applicable = bool(singleton_exact and candidate_source in raw_marker_sources)
    if marker_applicable:
        if marker_owned_count != int(candidate_exact):
            marker_blockers.append("marker_count_mismatch")
        if render_suppressed_count > 0:
            marker_blockers.append("owned_region_markers_suppressed")
    if raw_markers_match_accepted and candidate_source in raw_marker_sources:
        candidate["accepted_render_semantics"] = "owned_regions_match_accepted_exact"
        candidate["accepted_render_marker_count"] = int(marker_owned_count)
        candidate["accepted_render_overview_path"] = None if not render_info else render_info.get("overview_path")
        candidate["visual_acceptance_status"] = "render_verified"
    else:
        candidate["accepted_render_semantics"] = "scan_context_only_raw_f1_debug_not_final"
        candidate["accepted_render_marker_count"] = 0
        candidate["accepted_render_overview_path"] = None if not render_info else render_info.get("scan_context_path")
        candidate["visual_acceptance_status"] = "blocked"
    candidate["visual_acceptance_blockers"] = sorted(set(visual_blockers))
    candidate["render_suppressed_region_ids"] = render_suppressed_ids
    candidate["marker_suppression_details"] = render_suppression_details
    candidate["render_admissible_exact_bend_count"] = render_admissible_exact
    candidate["render_admissible_bend_count_range"] = (
        None if render_admissible_exact is None else [render_admissible_exact, render_admissible_exact]
    )
    candidate["render_admissible_delta_from_candidate_exact"] = (
        None if render_admissible_exact is None or candidate_exact is None else render_admissible_exact - int(candidate_exact)
    )
    candidate["marker_acceptance_status"] = (
        "not_applicable"
        if not marker_applicable
        else "blocked"
        if marker_blockers
        else "marker_verified"
    )
    candidate["marker_acceptance_blockers"] = sorted(set(marker_blockers))

    marker_guard_applied = False
    if marker_applicable and marker_blockers:
        control_range = list((payload.get("control") or {}).get("bend_count_range") or ())
        if len(control_range) >= 2:
            guarded_range = [int(control_range[0]), int(control_range[1])]
            candidate["pre_marker_guard_candidate"] = {
                "exact_bend_count": candidate_exact,
                "bend_count_range": candidate_range,
                "candidate_source": candidate_source,
                "guard_reason": candidate.get("guard_reason"),
            }
            candidate["exact_bend_count"] = None
            candidate["bend_count_range"] = guarded_range
            candidate["candidate_source"] = "marker_blocked_control_guard"
            candidate["guard_reason"] = "raw_f1_marker_acceptance_blocked"
            marker_guard_applied = True

            expected_bend_count = (payload.get("expectation") or {}).get("expected_bend_count")
            metrics = dict(payload.get("metrics") or {})
            metrics["candidate_exact_match"] = None
            metrics["candidate_range_contains_truth"] = _range_contains(expected_bend_count, guarded_range)
            metrics["range_delta"] = _compare(control_range, guarded_range, expected_bend_count)
            payload["metrics"] = metrics
    candidate["marker_guard_applied"] = marker_guard_applied
    payload["candidate"] = candidate

    promotion_diagnostic = dict(payload.get("raw_f1_promotion_diagnostic") or {})
    if visual_blockers:
        blockers = list(promotion_diagnostic.get("blockers") or [])
        blockers.extend(f"visual_acceptance:{blocker}" for blocker in sorted(set(visual_blockers)))
        promotion_diagnostic["blockers"] = sorted(set(blockers))
        promotion_diagnostic["candidate"] = False
        promotion_diagnostic["recommendation"] = "do_not_promote_exact"
        promotion_diagnostic["visual_acceptance_status"] = "blocked"
    else:
        promotion_diagnostic["visual_acceptance_status"] = "render_verified"
    promotion_diagnostic["visual_acceptance_blockers"] = sorted(set(visual_blockers))
    promotion_diagnostic["marker_acceptance_status"] = candidate["marker_acceptance_status"]
    promotion_diagnostic["marker_acceptance_blockers"] = candidate["marker_acceptance_blockers"]
    if marker_guard_applied:
        blockers = list(promotion_diagnostic.get("blockers") or [])
        blockers.append("marker_acceptance_guard_applied")
        promotion_diagnostic["blockers"] = sorted(set(blockers))
    payload["raw_f1_promotion_diagnostic"] = promotion_diagnostic


def _aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    exact_evaluated = 0
    exact_correct = 0
    raw_exact_evaluated = 0
    raw_exact_correct = 0
    render_admissible_exact_evaluated = 0
    render_admissible_exact_correct = 0
    render_admissible_contains = 0
    render_admissible_contains_evaluated = 0
    tightened = 0
    broadened = 0
    raw_tightened = 0
    raw_broadened = 0
    control_contains = 0
    candidate_contains = 0
    raw_contains = 0
    contains_evaluated = 0
    candidate_source_counts: Dict[str, int] = {}
    guard_reason_counts: Dict[str, int] = {}
    by_route: Dict[str, Dict[str, Any]] = {}
    promotion_candidate_parts: List[str] = []
    render_blocked_parts: List[str] = []
    marker_blocked_parts: List[str] = []
    for row in results:
        metrics = dict(row.get("metrics") or {})
        expected_count = (row.get("expectation") or {}).get("expected_bend_count")
        control_range = list((row.get("control") or {}).get("bend_count_range") or [0, 0])
        candidate_payload = dict(row.get("candidate") or {})
        raw_payload = dict(row.get("raw_f1_candidate") or {})
        raw_range = list(raw_payload.get("bend_count_range") or [0, 0])
        raw_exact = raw_payload.get("exact_bend_count")
        render_admissible_exact = candidate_payload.get("render_admissible_exact_bend_count")
        render_admissible_range = candidate_payload.get("render_admissible_bend_count_range")
        candidate_source = str(candidate_payload.get("candidate_source") or "unknown")
        guard_reason = candidate_payload.get("guard_reason")
        promotion_diagnostic = dict(row.get("raw_f1_promotion_diagnostic") or {})
        if promotion_diagnostic.get("candidate"):
            promotion_candidate_parts.append(str(row.get("part_key") or ""))
        if str(candidate_payload.get("visual_acceptance_status") or "") == "blocked":
            render_blocked_parts.append(str(row.get("part_key") or ""))
        if str(candidate_payload.get("marker_acceptance_status") or "") == "blocked":
            marker_blocked_parts.append(str(row.get("part_key") or ""))
        candidate_source_counts[candidate_source] = candidate_source_counts.get(candidate_source, 0) + 1
        if guard_reason:
            key = str(guard_reason)
            guard_reason_counts[key] = guard_reason_counts.get(key, 0) + 1
        route_payload = dict(row.get("scan_family_route") or {})
        route_id = str(route_payload.get("route_id") or "unrouted")
        route_bucket = by_route.setdefault(
            route_id,
            {
                "scans_processed": 0,
                "candidate_exact_evaluated": 0,
                "candidate_exact_correct": 0,
                "raw_f1_exact_evaluated": 0,
                "raw_f1_exact_correct": 0,
                "tightened_case_count": 0,
                "broadened_case_count": 0,
                "raw_f1_tightened_case_count": 0,
                "raw_f1_broadened_case_count": 0,
                "guarded_case_count": 0,
            },
        )
        route_bucket["scans_processed"] += 1
        if candidate_source == "control_guard":
            route_bucket["guarded_case_count"] += 1
        if metrics.get("candidate_exact_match") is not None:
            exact_evaluated += 1
            exact_correct += int(bool(metrics.get("candidate_exact_match")))
            route_bucket["candidate_exact_evaluated"] += 1
            route_bucket["candidate_exact_correct"] += int(bool(metrics.get("candidate_exact_match")))
        if expected_count is not None and raw_exact is not None:
            raw_exact_evaluated += 1
            raw_match = int(raw_exact) == int(expected_count)
            raw_exact_correct += int(raw_match)
            route_bucket["raw_f1_exact_evaluated"] += 1
            route_bucket["raw_f1_exact_correct"] += int(raw_match)
        if expected_count is not None and render_admissible_exact is not None:
            render_admissible_exact_evaluated += 1
            render_admissible_exact_correct += int(int(render_admissible_exact) == int(expected_count))
        if expected_count is not None and isinstance(render_admissible_range, list) and len(render_admissible_range) >= 2:
            render_admissible_contains_evaluated += 1
            render_admissible_contains += int(bool(_range_contains(expected_count, render_admissible_range)))
        if metrics.get("control_range_contains_truth") is not None:
            contains_evaluated += 1
            control_contains += int(bool(metrics.get("control_range_contains_truth")))
            candidate_contains += int(bool(metrics.get("candidate_range_contains_truth")))
            raw_contains += int(bool(_range_contains(expected_count, raw_range)))
        delta = dict(metrics.get("range_delta") or {})
        raw_delta = _compare(control_range, raw_range, expected_count)
        tightened += int(bool(delta.get("tightened")))
        broadened += int(bool(delta.get("broadened")))
        raw_tightened += int(bool(raw_delta.get("tightened")))
        raw_broadened += int(bool(raw_delta.get("broadened")))
        route_bucket["tightened_case_count"] += int(bool(delta.get("tightened")))
        route_bucket["broadened_case_count"] += int(bool(delta.get("broadened")))
        route_bucket["raw_f1_tightened_case_count"] += int(bool(raw_delta.get("tightened")))
        route_bucket["raw_f1_broadened_case_count"] += int(bool(raw_delta.get("broadened")))
    for route_bucket in by_route.values():
        evaluated = int(route_bucket["candidate_exact_evaluated"])
        route_bucket["candidate_exact_accuracy"] = (
            round(float(route_bucket["candidate_exact_correct"]) / float(evaluated), 3)
            if evaluated
            else None
        )
        raw_evaluated = int(route_bucket["raw_f1_exact_evaluated"])
        route_bucket["raw_f1_exact_accuracy"] = (
            round(float(route_bucket["raw_f1_exact_correct"]) / float(raw_evaluated), 3)
            if raw_evaluated
            else None
        )
    return {
        "scans_processed": len(results),
        "exact_count_evaluated": exact_evaluated,
        "candidate_exact_accuracy": round(exact_correct / exact_evaluated, 3) if exact_evaluated else None,
        "raw_f1_exact_count_evaluated": raw_exact_evaluated,
        "raw_f1_exact_accuracy": round(raw_exact_correct / raw_exact_evaluated, 3) if raw_exact_evaluated else None,
        "render_admissible_exact_count_evaluated": render_admissible_exact_evaluated,
        "render_admissible_exact_accuracy": round(render_admissible_exact_correct / render_admissible_exact_evaluated, 3)
        if render_admissible_exact_evaluated
        else None,
        "control_range_contains_truth_rate": round(control_contains / contains_evaluated, 3) if contains_evaluated else None,
        "candidate_range_contains_truth_rate": round(candidate_contains / contains_evaluated, 3) if contains_evaluated else None,
        "raw_f1_range_contains_truth_rate": round(raw_contains / contains_evaluated, 3) if contains_evaluated else None,
        "render_admissible_range_contains_truth_rate": round(
            render_admissible_contains / render_admissible_contains_evaluated,
            3,
        )
        if render_admissible_contains_evaluated
        else None,
        "tightened_case_count": tightened,
        "broadened_case_count": broadened,
        "raw_f1_tightened_case_count": raw_tightened,
        "raw_f1_broadened_case_count": raw_broadened,
        "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        "guard_reason_counts": dict(sorted(guard_reason_counts.items())),
        "raw_f1_manual_promotion_candidate_count": len([value for value in promotion_candidate_parts if value]),
        "raw_f1_manual_promotion_candidate_parts": sorted(value for value in promotion_candidate_parts if value),
        "render_blocked_candidate_count": len([value for value in render_blocked_parts if value]),
        "render_blocked_candidate_parts": sorted(value for value in render_blocked_parts if value),
        "marker_blocked_candidate_count": len([value for value in marker_blocked_parts if value]),
        "marker_blocked_candidate_parts": sorted(value for value in marker_blocked_parts if value),
        "by_route_id": dict(sorted(by_route.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the F1 patch-graph latent decomposition prototype on a focused hard-scan slice.")
    parser.add_argument("--slice-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-render", action="store_true", help="Skip static render artifact generation for faster benchmark iteration.")
    parser.add_argument("--part-key", action="append", default=[], help="Limit execution to one or more part keys from the slice.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected entries from the slice.")
    parser.add_argument("--cache-dir", default=None, help="Optional cache root for fixed offline research inputs.")
    parser.add_argument(
        "--upper-tail-probe",
        action="store_true",
        help="Diagnostic-only mode: relax the small-tight active-bend cap to test whether higher-count F1 alternatives become viable.",
    )
    parser.add_argument(
        "--birth-rescue-max-active-bends",
        type=int,
        default=None,
        help="Diagnostic-only override for the maximum active bend count used by interface-birth rescue.",
    )
    parser.add_argument(
        "--enable-interface-subspan-births",
        action="store_true",
        help="Diagnostic-only mode: enable overcomplete interface subspan birth hypotheses.",
    )
    parser.add_argument(
        "--prioritize-residual-birth-rescue",
        action="store_true",
        help="Diagnostic-only mode: when birth rescue is capped, prioritize birth supports currently owned by residual.",
    )
    parser.add_argument(
        "--allow-corridor-backed-contact-recovery",
        action="store_true",
        help="Diagnostic-only mode: keep one-sided/imbalanced bend support when axis/span/corridor/contact purity are strong.",
    )
    parser.add_argument(
        "--count-sweep-summary-json",
        action="append",
        default=[],
        help="Optional count-sweep summary JSON used as offline exact-promotion evidence for the matching part key.",
    )
    parser.add_argument(
        "--promotion-evidence-json",
        action="append",
        default=[],
        help="Optional explicit offline promotion evidence JSON keyed by part_key.",
    )
    parser.add_argument(
        "--route-mode",
        choices=("off", "observe", "apply"),
        default="observe",
        help="Offline scan-family router mode. observe writes route diagnostics without changing solver behavior; apply passes route solver options into F1.",
    )
    args = parser.parse_args()

    slice_payload = _load_json(Path(args.slice_json).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = load_bend_runtime_config(None)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else (output_dir / "_cache")
    solver_option_overrides = None
    if args.upper_tail_probe:
        solver_option_overrides = {
            "limit_birth_rescue_to_control_upper": False,
            "birth_rescue_max_active_bends": 0,
            "upper_tail_probe": True,
        }
    if args.birth_rescue_max_active_bends is not None:
        solver_option_overrides = dict(solver_option_overrides or {})
        solver_option_overrides["limit_birth_rescue_to_control_upper"] = False
        solver_option_overrides["birth_rescue_max_active_bends"] = max(0, int(args.birth_rescue_max_active_bends))
        solver_option_overrides["upper_tail_probe"] = True
    if args.enable_interface_subspan_births:
        solver_option_overrides = dict(solver_option_overrides or {})
        solver_option_overrides["enable_interface_subspan_births"] = True
        solver_option_overrides["limit_birth_rescue_to_control_upper"] = False
        solver_option_overrides["upper_tail_probe"] = True
    if args.prioritize_residual_birth_rescue:
        solver_option_overrides = dict(solver_option_overrides or {})
        solver_option_overrides["prioritize_residual_birth_rescue"] = True
        solver_option_overrides["upper_tail_probe"] = True
    if args.allow_corridor_backed_contact_recovery:
        solver_option_overrides = dict(solver_option_overrides or {})
        solver_option_overrides["allow_corridor_backed_contact_recovery"] = True
        solver_option_overrides["upper_tail_probe"] = True
    promotion_evidence = _load_count_sweep_promotion_evidence(args.count_sweep_summary_json or ())
    promotion_evidence.update(_load_promotion_evidence(args.promotion_evidence_json or ()))
    results: List[Dict[str, Any]] = []
    entries = list(slice_payload.get("entries") or [])
    selected_part_keys = {str(value) for value in (args.part_key or []) if str(value)}
    if selected_part_keys:
        entries = [entry for entry in entries if str(entry.get("part_key") or "") in selected_part_keys]
    if args.limit is not None:
        entries = entries[: max(0, int(args.limit))]

    for index, entry in enumerate(entries, start=1):
        part_key = str(entry.get("part_key") or "")
        scan_path = str(entry["scan_path"])
        print(f"[{index}/{len(entries)}] START {part_key}: {scan_path}", flush=True)
        scan_dir = output_dir / part_key
        scan_dir.mkdir(parents=True, exist_ok=True)
        result = run_patch_graph_latent_decomposition(
            scan_path=scan_path,
            runtime_config=runtime_config,
            part_id=part_key,
            cache_dir=cache_root / part_key,
            route_mode=args.route_mode,
            solver_option_overrides=solver_option_overrides,
        )
        _write_json(scan_dir / "decomposition_result.json", result.to_dict())
        part_promotion_evidence = promotion_evidence.get(part_key)
        supplemental_regions: Tuple[OwnedBendRegion, ...] = tuple()
        if (
            _family_residual_repair_promoted_count(
                promotion_evidence=part_promotion_evidence,
                raw_exact=int(result.exact_bend_count),
                raw_range=result.bend_count_range,
                control_range=result.current_zero_shot_count_range,
            )
            is not None
        ):
            supplemental_regions = _supplemental_owned_bend_regions_from_evidence(
                promotion_evidence=part_promotion_evidence,
                existing_regions=result.owned_bend_regions,
            )
        render_result = _result_with_supplemental_owned_regions(
            result=result,
            supplemental_regions=supplemental_regions,
        )
        if supplemental_regions:
            _write_json(scan_dir / "decomposition_result_with_supplemental_regions.json", render_result.to_dict())
        marker_admissibility = _owned_region_marker_admissibility(render_result)
        render_info = None if args.no_render else render_decomposition_artifacts(result=render_result, output_dir=scan_dir / "render")
        payload = _entry_result_payload(
            entry=entry,
            result=result,
            render_info=render_info,
            promotion_evidence=part_promotion_evidence,
        )
        payload["candidate"]["render_owned_bend_region_count"] = (
            int(marker_admissibility["marker_admissible_owned_region_count"])
            if render_info is None
            else int(render_info.get("render_owned_region_count", len(render_result.owned_bend_regions)))
        )
        payload["candidate"]["render_suppressed_owned_region_count"] = int(
            marker_admissibility["marker_suppressed_owned_region_count"]
            if render_info is None
            else int(render_info.get("render_suppressed_owned_region_count", 0))
        )
        payload["candidate"]["supplemental_owned_bend_region_count"] = len(supplemental_regions)
        _attach_candidate_render_semantics(
            payload,
            render_info=render_info,
            rendered_owned_region_count=int(payload["candidate"]["render_owned_bend_region_count"]),
            marker_admissibility=marker_admissibility,
        )
        _write_json(scan_dir / "summary.json", payload)
        results.append(payload)
        print(
            f"[{index}/{len(entries)}] DONE {part_key}: exact={payload['candidate']['exact_bend_count']} range={payload['candidate']['bend_count_range']}",
            flush=True,
        )

    aggregate = _aggregate(results)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slice_json": str(Path(args.slice_json).resolve()),
        "output_dir": str(output_dir),
        "aggregate": aggregate,
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
