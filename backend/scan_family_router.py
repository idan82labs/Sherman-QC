from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ScanFormationFeatures:
    scan_state_hint: str
    point_count: int
    local_spacing_mm: float
    bbox_mm: Tuple[float, float, float]
    aspect_ratios: Tuple[float, float, float]
    dominant_angle_family_deg: Optional[float]
    dominant_angle_family_mass: float
    duplicate_line_family_rate: float
    single_scale_count: int
    multiscale_count: int
    union_candidate_count: int
    zero_shot_profile_family: str
    zero_shot_exact_count: Optional[int]
    zero_shot_count_range: Tuple[int, int]
    zero_shot_confidence: float
    zero_shot_abstained: bool
    atom_count: int
    flange_hypothesis_count: int
    flange_orientation_family_count: int
    bend_hypothesis_count: int
    avg_candidate_attachment_pairs: float
    weak_incident_bend_fraction: float
    multi_attachment_bend_fraction: float
    density_class: str
    repeated_near90_strength: float
    candidate_fragmentation: float
    flange_fragmentation: float
    seed_incidence_risk: float
    small_tight_bend_signal: float
    count_range_width: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_state_hint": self.scan_state_hint,
            "point_count": int(self.point_count),
            "local_spacing_mm": round(float(self.local_spacing_mm), 4),
            "bbox_mm": [round(float(value), 4) for value in self.bbox_mm],
            "aspect_ratios": [round(float(value), 4) for value in self.aspect_ratios],
            "dominant_angle_family_deg": None
            if self.dominant_angle_family_deg is None
            else round(float(self.dominant_angle_family_deg), 4),
            "dominant_angle_family_mass": round(float(self.dominant_angle_family_mass), 4),
            "duplicate_line_family_rate": round(float(self.duplicate_line_family_rate), 4),
            "single_scale_count": int(self.single_scale_count),
            "multiscale_count": int(self.multiscale_count),
            "union_candidate_count": int(self.union_candidate_count),
            "zero_shot_profile_family": self.zero_shot_profile_family,
            "zero_shot_exact_count": self.zero_shot_exact_count,
            "zero_shot_count_range": list(self.zero_shot_count_range),
            "zero_shot_confidence": round(float(self.zero_shot_confidence), 4),
            "zero_shot_abstained": bool(self.zero_shot_abstained),
            "atom_count": int(self.atom_count),
            "flange_hypothesis_count": int(self.flange_hypothesis_count),
            "flange_orientation_family_count": int(self.flange_orientation_family_count),
            "bend_hypothesis_count": int(self.bend_hypothesis_count),
            "avg_candidate_attachment_pairs": round(float(self.avg_candidate_attachment_pairs), 4),
            "weak_incident_bend_fraction": round(float(self.weak_incident_bend_fraction), 4),
            "multi_attachment_bend_fraction": round(float(self.multi_attachment_bend_fraction), 4),
            "density_class": self.density_class,
            "repeated_near90_strength": round(float(self.repeated_near90_strength), 4),
            "candidate_fragmentation": round(float(self.candidate_fragmentation), 4),
            "flange_fragmentation": round(float(self.flange_fragmentation), 4),
            "seed_incidence_risk": round(float(self.seed_incidence_risk), 4),
            "small_tight_bend_signal": round(float(self.small_tight_bend_signal), 4),
            "count_range_width": int(self.count_range_width),
        }


@dataclass(frozen=True)
class ScanFamilyRoute:
    route_id: str
    scan_family: str
    solver_profile: str
    enabled_moves: Tuple[str, ...]
    preprocess_recipe: Optional[str]
    reason_codes: Tuple[str, ...]
    confidence: float
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "scan_family": self.scan_family,
            "solver_profile": self.solver_profile,
            "enabled_moves": list(self.enabled_moves),
            "preprocess_recipe": self.preprocess_recipe,
            "reason_codes": list(self.reason_codes),
            "confidence": round(float(self.confidence), 4),
            "feature_snapshot": dict(self.feature_snapshot),
        }


def build_scan_formation_features(
    *,
    current_zero_shot_result: Mapping[str, Any],
    atom_graph: Any = None,
    flange_hypotheses: Sequence[Any] = (),
    bend_hypotheses: Sequence[Any] = (),
    f1_diagnostics: Optional[Mapping[str, Any]] = None,
) -> ScanFormationFeatures:
    del f1_diagnostics  # Reserved for follow-up routing from prior F1 runs.
    candidate_summary = dict(current_zero_shot_result.get("candidate_generator_summary") or {})
    point_count = _coerce_int(candidate_summary.get("point_count"), 0)
    local_spacing_mm = _coerce_float(candidate_summary.get("local_spacing_mm"), 0.0)
    bbox_mm = _coerce_float_tuple(candidate_summary.get("bbox_mm"), length=3)
    aspect_ratios = _coerce_float_tuple(candidate_summary.get("aspect_ratios"), length=3)
    dominant_angle = _dominant_angle(current_zero_shot_result, candidate_summary)
    dominant_mass = _dominant_angle_mass(current_zero_shot_result, candidate_summary, dominant_angle)
    duplicate_rate = _coerce_float(candidate_summary.get("duplicate_line_family_rate"), 0.0)
    single_count = _coerce_int(candidate_summary.get("single_scale_count"), 0)
    multiscale_count = _coerce_int(candidate_summary.get("multiscale_count"), 0)
    union_count = _coerce_int(candidate_summary.get("union_candidate_count"), single_count + multiscale_count)
    profile_family = str(current_zero_shot_result.get("profile_family") or "unknown")
    exact_count = _coerce_optional_int(current_zero_shot_result.get("estimated_bend_count"))
    count_range = _normalize_count_range(current_zero_shot_result.get("estimated_bend_count_range"), exact_count)
    confidence = _coerce_float(current_zero_shot_result.get("hypothesis_confidence"), 0.0)
    scan_state_hint = str(candidate_summary.get("scan_state_hint") or "").strip().lower()

    atom_count = len(tuple(getattr(atom_graph, "atoms", ()) or ()))
    flange_count = len(tuple(flange_hypotheses or ()))
    flange_orientation_count = _flange_orientation_family_count(flange_hypotheses)
    bend_count = len(tuple(bend_hypotheses or ()))
    attachment_counts = [
        len(_candidate_attachment_pairs_for_payload(bend))
        for bend in tuple(bend_hypotheses or ())
    ]
    weak_incident_count = sum(
        1
        for bend in tuple(bend_hypotheses or ())
        if len(tuple(getattr(bend, "seed_incident_flange_ids", ()) or getattr(bend, "incident_flange_ids", ()) or ())) < 2
    )
    multi_attachment_count = sum(1 for value in attachment_counts if value > 1)
    reference_count = max(count_range[1], exact_count or 0, 1)
    density_class = _density_class(point_count, local_spacing_mm)
    repeated_strength = _repeated_near90_strength(profile_family, dominant_angle, dominant_mass)
    candidate_fragmentation = float(bend_count) / float(reference_count)
    flange_fragmentation = float(flange_count) / float(max(reference_count + 1, 1))
    weak_fraction = float(weak_incident_count) / float(max(bend_count, 1))
    multi_fraction = float(multi_attachment_count) / float(max(bend_count, 1))
    seed_risk = min(1.0, weak_fraction * 0.65 + multi_fraction * 0.45 + max(0.0, candidate_fragmentation - 1.7) * 0.18)
    small_tight_signal = max(0.0, float(multiscale_count - single_count)) / float(max(union_count, 1))

    return ScanFormationFeatures(
        scan_state_hint=scan_state_hint,
        point_count=point_count,
        local_spacing_mm=local_spacing_mm,
        bbox_mm=bbox_mm,
        aspect_ratios=aspect_ratios,
        dominant_angle_family_deg=dominant_angle,
        dominant_angle_family_mass=dominant_mass,
        duplicate_line_family_rate=duplicate_rate,
        single_scale_count=single_count,
        multiscale_count=multiscale_count,
        union_candidate_count=union_count,
        zero_shot_profile_family=profile_family,
        zero_shot_exact_count=exact_count,
        zero_shot_count_range=count_range,
        zero_shot_confidence=confidence,
        zero_shot_abstained=bool(current_zero_shot_result.get("abstained")),
        atom_count=atom_count,
        flange_hypothesis_count=flange_count,
        flange_orientation_family_count=flange_orientation_count,
        bend_hypothesis_count=bend_count,
        avg_candidate_attachment_pairs=float(sum(attachment_counts)) / float(max(len(attachment_counts), 1)),
        weak_incident_bend_fraction=weak_fraction,
        multi_attachment_bend_fraction=multi_fraction,
        density_class=density_class,
        repeated_near90_strength=repeated_strength,
        candidate_fragmentation=candidate_fragmentation,
        flange_fragmentation=flange_fragmentation,
        seed_incidence_risk=seed_risk,
        small_tight_bend_signal=small_tight_signal,
        count_range_width=max(0, int(count_range[1] - count_range[0])),
    )


def route_scan_family(features: ScanFormationFeatures) -> ScanFamilyRoute:
    reasons = []

    if features.scan_state_hint == "partial" or features.zero_shot_profile_family == "scan_limited_partial":
        reasons.append("partial_or_scan_limited")
        if (
            features.zero_shot_count_range[1] <= 2
            and features.union_candidate_count <= 2
            and features.flange_hypothesis_count >= 6
            and features.repeated_near90_strength >= 0.75
        ):
            reasons.extend(["sparse_seed_candidate_pool", "many_flange_components"])
            return _route(
                route_id="partial_sparse_seed_support_birth",
                scan_family="partial_scan_limited",
                solver_profile="partial_sparse_seed_support_birth",
                enabled_moves=(
                    "observed_bend_semantics",
                    "full_geometric_attachment",
                    "interface_activation",
                    "interface_subspan_birth",
                    "no_contact_birth_rescue",
                    "control_f1_envelope",
                ),
                reasons=reasons,
                confidence=0.7,
                features=features,
            )
        return _route(
            route_id="partial_scan_limited",
            scan_family="partial_scan_limited",
            solver_profile="partial_observed_recovery",
            enabled_moves=(
                "observed_bend_semantics",
                "full_geometric_attachment",
                "interface_activation",
                "duplicate_collapse",
                "repair_before_prune",
                "control_f1_envelope",
            ),
            reasons=reasons,
            confidence=0.88,
            features=features,
        )

    if _zero_shot_is_stable(features):
        reasons.extend(["zero_shot_exact_stable", "low_route_risk"])
        return _route(
            route_id="zero_shot_control_stable",
            scan_family="zero_shot_control_stable",
            solver_profile="control_exact_tiebreak",
            enabled_moves=("observe_route", "control_exact_near_energy_tiebreak"),
            reasons=reasons,
            confidence=0.84,
            features=features,
        )

    if _incidence_is_unavailable(features):
        reasons.extend(["flange_incidence_unavailable", "degenerate_flange_orientation_pool"])
        return _route(
            route_id="incidence_unavailable_control",
            scan_family="incidence_unavailable_control",
            solver_profile="observe_only",
            enabled_moves=("observe_route", "control_guard"),
            reasons=reasons,
            confidence=0.82,
            features=features,
        )

    if features.repeated_near90_strength >= 0.72 and (
        features.seed_incidence_risk >= 0.42
        or features.candidate_fragmentation >= 1.9
        or features.flange_fragmentation >= 1.15
    ):
        reasons.extend(["repeated_near90", "seed_or_fragmentation_risk"])
        return _route(
            route_id="fragmented_repeat90_interface_birth",
            scan_family="fragmented_repeat90_interface_birth",
            solver_profile="interface_birth_enabled",
            enabled_moves=("full_geometric_attachment", "interface_activation", "duplicate_collapse", "repair_before_prune"),
            reasons=reasons,
            confidence=0.78,
            features=features,
        )

    if features.repeated_near90_strength >= 0.65:
        reasons.append("repeated_near90")
        if features.duplicate_line_family_rate >= 0.25:
            reasons.append("duplicate_line_pressure")
        return _route(
            route_id="dense_repeated_near90",
            scan_family="dense_repeated_near90",
            solver_profile="full_geometric_attachment",
            enabled_moves=("full_geometric_attachment", "duplicate_collapse"),
            reasons=reasons,
            confidence=0.72,
            features=features,
        )

    if features.small_tight_bend_signal >= 0.22:
        reasons.append("multiscale_small_bend_gap")
        return _route(
            route_id="small_tight_bend_recovery",
            scan_family="small_tight_bend_recovery",
            solver_profile="interface_birth_enabled",
            enabled_moves=("support_creation", "full_geometric_attachment", "conservative_count_gate"),
            reasons=reasons,
            confidence=0.7,
            features=features,
        )

    if features.zero_shot_profile_family == "rolled_or_mixed_transition" or (
        features.count_range_width >= 2 and features.duplicate_line_family_rate >= 0.35
    ):
        reasons.append("mixed_or_rolled_ambiguous")
        return _route(
            route_id="ambiguous_mixed_transition",
            scan_family="ambiguous_mixed_transition",
            solver_profile="mixed_transition_recovery",
            enabled_moves=("mixed_single_contact_recovery", "interface_activation", "conservative_count_gate"),
            reasons=reasons,
            confidence=0.67,
            features=features,
        )

    reasons.append("default_conservative")
    return _route(
        route_id="conservative_attachment",
        scan_family="conservative_attachment",
        solver_profile="conservative_attachment",
        enabled_moves=("conservative_attachment", "duplicate_collapse"),
        reasons=reasons,
        confidence=0.62,
        features=features,
    )


def solver_options_for_route(route: Optional[ScanFamilyRoute], route_mode: str) -> Dict[str, Any]:
    mode = str(route_mode or "off").strip().lower()
    if mode not in {"off", "observe", "apply"}:
        mode = "off"
    if route is None or mode != "apply":
        return {
            "route_mode": mode,
            "solver_profile": "default",
            "enabled_moves": ("current_default",),
            "repair_iterations": 4,
            "allow_interface_activation": True,
            "enable_interface_birth": False,
            "repair_before_prune": False,
            "enable_birth_support_rescue": False,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": False,
            "allow_hypothesis_support_contacts": False,
            "allow_cross_axis_attachment_match": False,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": False,
        }
    if route.solver_profile == "observe_only":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 2,
            "allow_interface_activation": False,
            "enable_interface_birth": False,
            "repair_before_prune": False,
            "enable_birth_support_rescue": False,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": False,
            "allow_hypothesis_support_contacts": False,
            "allow_cross_axis_attachment_match": False,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": False,
        }
    if route.solver_profile == "control_exact_tiebreak":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 2,
            "allow_interface_activation": False,
            "enable_interface_birth": False,
            "repair_before_prune": False,
            "enable_birth_support_rescue": False,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": False,
            "allow_hypothesis_support_contacts": False,
            "allow_cross_axis_attachment_match": False,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": False,
            "enable_control_exact_solution_tiebreak": True,
            "control_exact_solution_tiebreak_energy_gap": 3.0,
        }
    if route.solver_profile == "conservative_attachment":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 3,
            "allow_interface_activation": False,
            "enable_interface_birth": False,
            "repair_before_prune": False,
            "enable_birth_support_rescue": False,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": False,
            "allow_hypothesis_support_contacts": False,
            "allow_cross_axis_attachment_match": False,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": False,
        }
    if route.solver_profile == "mixed_transition_recovery":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 4,
            "allow_interface_activation": True,
            "enable_interface_birth": True,
            "repair_before_prune": True,
            "enable_birth_support_rescue": True,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": True,
            "allow_geometry_backed_single_contact": True,
            "allow_hypothesis_support_contacts": True,
            "allow_cross_axis_attachment_match": True,
            "allow_mixed_transition_single_contact": True,
            "enable_selected_pair_duplicate_prune": True,
            "selected_pair_duplicate_prune_min_active_bends": 0,
            "selected_pair_duplicate_prune_alias_atom_max": 15,
            "enable_final_duplicate_cluster_cleanup": True,
            "enable_final_duplicate_cluster_scoring": True,
            "final_duplicate_cluster_alias_atom_max": 15,
            "final_duplicate_cluster_same_pair_penalty": 140.0,
            "final_duplicate_cluster_cross_pair_penalty": 95.0,
        }
    if route.solver_profile == "partial_observed_recovery":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 6,
            "allow_interface_activation": True,
            "enable_interface_birth": True,
            "repair_before_prune": True,
            "enable_birth_support_rescue": True,
            "enable_interface_subspan_births": False,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": True,
            "allow_hypothesis_support_contacts": True,
            "allow_cross_axis_attachment_match": True,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": True,
            "partial_observed_semantics": True,
        }
    if route.solver_profile == "partial_sparse_seed_support_birth":
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 6,
            "allow_interface_activation": True,
            "enable_interface_birth": True,
            "repair_before_prune": True,
            "enable_birth_support_rescue": True,
            "enable_interface_subspan_births": True,
            "limit_birth_rescue_to_control_upper": False,
            "allow_geometry_backed_single_contact": True,
            "allow_hypothesis_support_contacts": True,
            "allow_cross_axis_attachment_match": True,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": True,
            "allow_no_contact_birth_rescue": True,
            "partial_observed_semantics": True,
            "partial_sparse_seed_upper_pad": 1,
        }
    if route.solver_profile == "interface_birth_enabled":
        feature_snapshot = dict(route.feature_snapshot or {})
        control_range = tuple(feature_snapshot.get("zero_shot_count_range") or ())
        control_upper = int(control_range[1]) if len(control_range) >= 2 else 0
        high_count_fragmented_repeat90 = bool(
            route.route_id == "fragmented_repeat90_interface_birth"
            and (
                control_upper >= 10
                or int(feature_snapshot.get("union_candidate_count") or 0) >= 18
            )
        )
        return {
            "route_mode": mode,
            "solver_profile": route.solver_profile,
            "enabled_moves": route.enabled_moves,
            "repair_iterations": 6,
            "allow_interface_activation": True,
            "enable_interface_birth": True,
            "repair_before_prune": True,
            "enable_birth_support_rescue": True,
            "enable_interface_subspan_births": bool(route.route_id == "small_tight_bend_recovery" or high_count_fragmented_repeat90),
            "limit_birth_rescue_to_control_upper": bool(route.route_id == "small_tight_bend_recovery" or high_count_fragmented_repeat90),
            "limit_subspan_births_to_control_low_plus_one": bool(route.route_id == "small_tight_bend_recovery"),
            "prioritize_residual_birth_rescue": high_count_fragmented_repeat90,
            "allow_geometry_backed_single_contact": True,
            "allow_hypothesis_support_contacts": True,
            "allow_cross_axis_attachment_match": True,
            "allow_mixed_transition_single_contact": False,
            "enable_selected_pair_duplicate_prune": True,
        }
    return {
        "route_mode": mode,
        "solver_profile": route.solver_profile,
        "enabled_moves": route.enabled_moves,
        "repair_iterations": 4,
        "allow_interface_activation": True,
        "enable_interface_birth": False,
        "repair_before_prune": False,
        "enable_birth_support_rescue": False,
        "enable_interface_subspan_births": False,
        "limit_birth_rescue_to_control_upper": False,
        "allow_geometry_backed_single_contact": False,
        "allow_hypothesis_support_contacts": False,
        "allow_cross_axis_attachment_match": False,
        "allow_mixed_transition_single_contact": False,
        "enable_selected_pair_duplicate_prune": False,
    }


def route_from_payload(payload: Mapping[str, Any]) -> Optional[ScanFamilyRoute]:
    if not payload:
        return None
    return ScanFamilyRoute(
        route_id=str(payload.get("route_id") or ""),
        scan_family=str(payload.get("scan_family") or ""),
        solver_profile=str(payload.get("solver_profile") or "default"),
        enabled_moves=tuple(str(value) for value in payload.get("enabled_moves") or ()),
        preprocess_recipe=payload.get("preprocess_recipe"),
        reason_codes=tuple(str(value) for value in payload.get("reason_codes") or ()),
        confidence=_coerce_float(payload.get("confidence"), 0.0),
        feature_snapshot=dict(payload.get("feature_snapshot") or {}),
    )


def _route(
    *,
    route_id: str,
    scan_family: str,
    solver_profile: str,
    enabled_moves: Tuple[str, ...],
    reasons: Sequence[str],
    confidence: float,
    features: ScanFormationFeatures,
    preprocess_recipe: Optional[str] = None,
) -> ScanFamilyRoute:
    return ScanFamilyRoute(
        route_id=route_id,
        scan_family=scan_family,
        solver_profile=solver_profile,
        enabled_moves=tuple(enabled_moves),
        preprocess_recipe=preprocess_recipe,
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        confidence=confidence,
        feature_snapshot=features.to_dict(),
    )


def _zero_shot_is_stable(features: ScanFormationFeatures) -> bool:
    if (
        features.zero_shot_exact_count is not None
        and features.count_range_width <= 1
        and features.zero_shot_confidence >= 0.62
        and features.zero_shot_count_range[0] <= features.zero_shot_exact_count <= features.zero_shot_count_range[1]
    ):
        return True
    return bool(
        features.zero_shot_exact_count is not None
        and features.count_range_width == 0
        and features.zero_shot_confidence >= 0.65
        and features.duplicate_line_family_rate < 0.25
    )


def _incidence_is_unavailable(features: ScanFormationFeatures) -> bool:
    if features.bend_hypothesis_count <= 0:
        return False
    if features.weak_incident_bend_fraction < 0.80:
        return False
    if features.avg_candidate_attachment_pairs >= 0.50:
        return False
    return bool(features.flange_hypothesis_count < 2 or features.flange_orientation_family_count < 2)


def _flange_orientation_family_count(flange_hypotheses: Sequence[Any]) -> int:
    families = []
    for flange in tuple(flange_hypotheses or ()):
        normal = getattr(flange, "normal", None)
        if normal is None:
            continue
        try:
            values = tuple(float(value) for value in tuple(normal))
        except Exception:
            continue
        if len(values) != 3:
            continue
        norm = sum(value * value for value in values) ** 0.5
        if norm <= 1e-9:
            continue
        unit = tuple(value / norm for value in values)
        duplicate = False
        for family in families:
            dot = abs(sum(unit[index] * family[index] for index in range(3)))
            if dot >= 0.9659:  # within roughly 15 degrees, ignoring sign.
                duplicate = True
                break
        if not duplicate:
            families.append(unit)
    return len(families)


def _dominant_angle(current_zero_shot_result: Mapping[str, Any], candidate_summary: Mapping[str, Any]) -> Optional[float]:
    raw = candidate_summary.get("dominant_angle_family_deg")
    if raw is None:
        families = current_zero_shot_result.get("dominant_angle_families_deg") or ()
        raw = families[0] if families else None
    try:
        return None if raw is None else float(raw)
    except Exception:
        return None


def _dominant_angle_mass(
    current_zero_shot_result: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    dominant_angle: Optional[float],
) -> float:
    raw = candidate_summary.get("dominant_angle_family_mass")
    if raw is not None:
        return _coerce_float(raw, 0.0)
    if dominant_angle is None:
        return 0.0
    family_count = len(current_zero_shot_result.get("dominant_angle_families_deg") or ())
    if family_count <= 1:
        return 0.82
    return 0.55


def _repeated_near90_strength(profile_family: str, dominant_angle: Optional[float], dominant_mass: float) -> float:
    near90 = dominant_angle is not None and 78.0 <= abs(float(dominant_angle)) <= 102.0
    profile_bonus = 0.25 if profile_family in {"repeated_90_sheetmetal", "orthogonal_dense_sheetmetal"} else 0.0
    if not near90:
        return profile_bonus
    return min(1.0, float(dominant_mass) + profile_bonus)


def _density_class(point_count: int, local_spacing_mm: float) -> str:
    if point_count >= 650000 or (local_spacing_mm > 0.0 and local_spacing_mm <= 0.75):
        return "dense"
    if point_count >= 300000 or (local_spacing_mm > 0.0 and local_spacing_mm <= 1.5):
        return "medium"
    return "sparse"


def _candidate_attachment_pairs_for_payload(bend: Any) -> Tuple[Tuple[str, str], ...]:
    pairs = []
    for raw in (
        *(getattr(bend, "candidate_incident_flange_pairs", ()) or ()),
        getattr(bend, "seed_incident_flange_ids", ()) or (),
        getattr(bend, "canonical_flange_pair", ()) or (),
        getattr(bend, "incident_flange_ids", ()) or (),
    ):
        pair = tuple(sorted(str(value) for value in tuple(raw or ()) if str(value)))
        if len(pair) == 2:
            pairs.append((pair[0], pair[1]))
    return tuple(dict.fromkeys(pairs))


def _normalize_count_range(raw_range: Any, exact_count: Optional[int]) -> Tuple[int, int]:
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        try:
            low = int(raw_range[0])
            high = int(raw_range[1])
            return (min(low, high), max(low, high))
        except Exception:
            pass
    if exact_count is not None:
        return (int(exact_count), int(exact_count))
    return (0, 0)


def _coerce_float_tuple(raw: Any, *, length: int) -> Tuple[float, ...]:
    values = []
    if isinstance(raw, (list, tuple)):
        for item in list(raw)[:length]:
            values.append(_coerce_float(item, 0.0))
    while len(values) < length:
        values.append(0.0)
    return tuple(values)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
