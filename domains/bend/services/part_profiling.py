from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from backend.feature_detection.bend_detector import BendSpecification


SEMANTIC_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".brep", ".brp", ".x_t", ".x_b"}
MESH_NOMINAL_EXTENSIONS = {".stl", ".obj", ".ply", ".pcd", ".off"}


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _coerce_nonnegative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _coerce_float_list(values: Optional[Sequence[Any]]) -> Tuple[float, ...]:
    out = []
    for value in values or ():
        try:
            out.append(float(value))
        except Exception:
            continue
    return tuple(out)


def _normalize_string_list(values: Optional[Sequence[Any]], *, upper: bool = False) -> Tuple[str, ...]:
    normalized = []
    for value in values or ():
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text.upper() if upper else text)
    return tuple(sorted(dict.fromkeys(normalized)))


def _normalize_angle_overrides(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for key, value in (raw or {}).items():
        bend_id = str(key).strip()
        if not bend_id or value is None:
            continue
        try:
            overrides[bend_id] = float(value)
        except Exception:
            continue
    return dict(sorted(overrides.items()))


def _cluster_angle_families(angles: Sequence[float], threshold_deg: float = 10.0) -> Tuple[float, ...]:
    values = sorted(float(value) for value in angles)
    if not values:
        return ()
    clusters = [[values[0]]]
    for angle in values[1:]:
        current = clusters[-1]
        current_mean = sum(current) / len(current)
        if abs(angle - current_mean) <= threshold_deg:
            current.append(angle)
        else:
            clusters.append([angle])
    return tuple(round(sum(cluster) / len(cluster), 2) for cluster in clusters)


def normalize_nominal_source_class(
    *,
    cad_path: Optional[str] = None,
    cad_source_label: Optional[str] = None,
) -> str:
    ext = Path(str(cad_path or "")).suffix.lower()
    source_label = str(cad_source_label or "").strip().lower()
    if ext in SEMANTIC_CAD_EXTENSIONS:
        return "semantic_nominal_cad"
    if ext in MESH_NOMINAL_EXTENSIONS:
        return "mesh_nominal"
    if "semantic" in source_label or source_label in {"cadex", "brep", "step", "iges"}:
        return "semantic_nominal_cad"
    if any(token in source_label for token in ("mesh", "triangulated", "vertices")):
        return "mesh_nominal"
    if "scan" in source_label:
        return "scan_only"
    return "mesh_nominal" if cad_path else "scan_only"


def normalize_spec_schedule_type(spec_schedule_type: Optional[str]) -> str:
    value = str(spec_schedule_type or "").strip().lower()
    return value or "unknown"


def normalize_feature_policy(
    feature_policy: Optional[Mapping[str, Any]],
    *,
    metadata_truth: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    feature_policy = dict(feature_policy or {})
    metadata_truth = dict(metadata_truth or {})

    part_family = str(feature_policy.get("part_family") or metadata_truth.get("part_family") or "").strip()
    if part_family:
        normalized["part_family"] = part_family

    countable_source = str(feature_policy.get("countable_source") or "").strip().lower()
    if countable_source:
        normalized["countable_source"] = countable_source

    rolled_feature_type = str(feature_policy.get("rolled_feature_type") or "ROLLED_SECTION").strip().upper()
    normalized["rolled_feature_type"] = rolled_feature_type or "ROLLED_SECTION"

    normalized["non_counting_feature_forms"] = list(
        _normalize_string_list(feature_policy.get("non_counting_feature_forms"), upper=True)
    )
    normalized["countable_feature_forms"] = list(
        _normalize_string_list(feature_policy.get("countable_feature_forms"), upper=True)
    )
    normalized["non_counting_bend_ids"] = list(
        _normalize_string_list(feature_policy.get("non_counting_bend_ids"))
    )
    normalized["countable_bend_ids"] = list(
        _normalize_string_list(feature_policy.get("countable_bend_ids"))
    )

    merged_angle_overrides = _normalize_angle_overrides(feature_policy.get("countable_bend_angle_overrides"))
    if not merged_angle_overrides:
        merged_angle_overrides = _normalize_angle_overrides(
            metadata_truth.get("countable_bend_target_angles_deg")
        )
    normalized["countable_bend_angle_overrides"] = merged_angle_overrides
    return normalized


def _resolve_expected_count(
    *,
    countable_bend_count: int,
    expected_bend_override: Optional[int],
    metadata_truth: Optional[Mapping[str, Any]],
    spec_bend_angles_hint: Sequence[float],
    spec_schedule_type: str,
) -> Tuple[int, Tuple[int, int], str]:
    override = _coerce_positive_int(expected_bend_override)
    if override is not None:
        return override, (override, override), "expected_bend_override"

    metadata_count = _coerce_positive_int((metadata_truth or {}).get("expected_total_bends"))
    if metadata_count is not None:
        return metadata_count, (metadata_count, metadata_count), "metadata_truth.expected_total_bends"

    hint_count = len(spec_bend_angles_hint)
    if hint_count > 0:
        if spec_schedule_type == "partial_angle_schedule":
            lower = hint_count
            upper = max(hint_count, countable_bend_count)
            return upper, (lower, upper), "spec_schedule_type.partial_angle_schedule"
        return hint_count, (hint_count, hint_count), f"spec_schedule_type.{spec_schedule_type}"

    return countable_bend_count, (countable_bend_count, countable_bend_count), "countable_bend_set"


def _resolve_dominant_angle_families(
    *,
    cad_bends: Sequence[BendSpecification],
    spec_bend_angles_hint: Sequence[float],
    metadata_truth: Optional[Mapping[str, Any]],
) -> Tuple[Tuple[float, ...], str]:
    metadata_angles = tuple(
        _normalize_angle_overrides((metadata_truth or {}).get("countable_bend_target_angles_deg")).values()
    )
    if metadata_angles:
        return _cluster_angle_families(metadata_angles), "metadata_truth.countable_bend_target_angles_deg"
    if spec_bend_angles_hint:
        return _cluster_angle_families(spec_bend_angles_hint), "spec_bend_angles_hint"
    cad_angles = [float(spec.target_angle) for spec in cad_bends if getattr(spec, "countable_in_regression", True)]
    if not cad_angles:
        cad_angles = [float(spec.target_angle) for spec in cad_bends]
    return _cluster_angle_families(cad_angles), "cad_bend_targets"


def _resolve_rolled_allowed(
    *,
    cad_bends: Sequence[BendSpecification],
    countable_feature_policy: Mapping[str, Any],
    metadata_truth: Optional[Mapping[str, Any]],
) -> bool:
    countable_forms = {str(value).upper() for value in countable_feature_policy.get("countable_feature_forms", [])}
    non_counting_forms = {
        str(value).upper() for value in countable_feature_policy.get("non_counting_feature_forms", [])
    }
    if "ROLLED" in countable_forms:
        return True
    if "ROLLED" in non_counting_forms:
        return False
    if any(str(getattr(spec, "bend_form", "")).upper() == "ROLLED" for spec in cad_bends):
        return True
    part_family = str((metadata_truth or {}).get("part_family") or "").strip().lower()
    return "rolled" in part_family


def _resolve_authority(source_class: str) -> Tuple[str, float]:
    if source_class == "semantic_nominal_cad":
        return "authoritative", 1.0
    if source_class == "mesh_nominal":
        return "geometric_prior", 0.7
    return "inferred_only", 0.25


def _default_topology_graph(cad_bends: Sequence[BendSpecification]) -> Dict[str, Tuple[str, ...]]:
    return {str(spec.bend_id): () for spec in cad_bends}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return _json_safe_mapping(value)
    return value


def _json_safe_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in mapping.items()}


def _coerce_scan_payload(zero_shot_result: Any) -> Dict[str, Any]:
    if zero_shot_result is None:
        raise ValueError("zero_shot_result is required")
    if isinstance(zero_shot_result, Mapping):
        return dict(zero_shot_result)
    to_dict = getattr(zero_shot_result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("zero_shot_result must be a mapping or expose to_dict()")


def _normalize_count_range(
    raw_range: Optional[Sequence[Any]],
    *,
    estimated_bend_count: Optional[int],
) -> Tuple[int, int]:
    if raw_range and len(raw_range) >= 2:
        low = _coerce_nonnegative_int(raw_range[0])
        high = _coerce_nonnegative_int(raw_range[1])
        if low is not None and high is not None:
            return (min(low, high), max(low, high))
    if estimated_bend_count is not None:
        return (estimated_bend_count, estimated_bend_count)
    return (0, 0)


def _normalize_line_family_groups(raw_groups: Optional[Sequence[Any]]) -> Tuple[Tuple[str, ...], ...]:
    normalized = []
    for raw_group in raw_groups or ():
        if not isinstance(raw_group, Sequence) or isinstance(raw_group, (str, bytes)):
            continue
        group = tuple(
            str(item).strip()
            for item in raw_group
            if str(item).strip()
        )
        if group:
            normalized.append(group)
    return tuple(normalized)


@dataclass(frozen=True)
class CadStructuralTarget:
    nominal_bend_set: Tuple[str, ...]
    nominal_topology_graph: Dict[str, Tuple[str, ...]]
    dominant_angle_families_deg: Tuple[float, ...]
    expected_bend_count: int
    expected_bend_count_range: Tuple[int, int]
    countable_feature_policy: Dict[str, Any]
    rolled_allowed: bool
    tolerance_context: Dict[str, Any]
    authority_level: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nominal_bend_set": list(self.nominal_bend_set),
            "nominal_topology_graph": {
                key: list(value) for key, value in self.nominal_topology_graph.items()
            },
            "dominant_angle_families_deg": list(self.dominant_angle_families_deg),
            "expected_bend_count": self.expected_bend_count,
            "expected_bend_count_range": list(self.expected_bend_count_range),
            "countable_feature_policy": _json_safe_mapping(self.countable_feature_policy),
            "rolled_allowed": self.rolled_allowed,
            "tolerance_context": _json_safe_mapping(self.tolerance_context),
            "authority_level": self.authority_level,
            "provenance": _json_safe_mapping(self.provenance),
        }


@dataclass(frozen=True)
class ScanStructuralHypothesis:
    profile_family: str
    estimated_bend_count: Optional[int]
    estimated_bend_count_range: Tuple[int, int]
    dominant_angle_families_deg: Tuple[float, ...]
    line_family_groups: Tuple[Tuple[str, ...], ...]
    hypothesis_confidence: float
    abstained: bool
    abstain_reasons: Tuple[str, ...]
    support_breakdown: Dict[str, Any]
    bend_support_strips: Tuple[Dict[str, Any], ...] = ()
    renderable_bend_objects: Tuple[Dict[str, Any], ...] = ()
    strip_count_confidence: Optional[float] = None
    strip_duplicate_merge_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_family": self.profile_family,
            "estimated_bend_count": self.estimated_bend_count,
            "estimated_bend_count_range": list(self.estimated_bend_count_range),
            "dominant_angle_families_deg": list(self.dominant_angle_families_deg),
            "line_family_groups": [list(group) for group in self.line_family_groups],
            "bend_support_strips": [_json_safe_mapping(strip) for strip in self.bend_support_strips],
            "renderable_bend_objects": [_json_safe_mapping(item) for item in self.renderable_bend_objects],
            "strip_count_confidence": round(float(self.strip_count_confidence), 4)
            if self.strip_count_confidence is not None else None,
            "strip_duplicate_merge_count": int(self.strip_duplicate_merge_count),
            "hypothesis_confidence": round(float(self.hypothesis_confidence), 4),
            "abstained": self.abstained,
            "abstain_reasons": list(self.abstain_reasons),
            "support_breakdown": _json_safe_mapping(self.support_breakdown),
        }


@dataclass(frozen=True)
class PartProfile:
    profile_mode: str
    nominal_source_class: str
    profile_confidence: float
    provenance: Dict[str, Any]
    cad_target: Optional[CadStructuralTarget] = None
    scan_hypothesis: Optional[ScanStructuralHypothesis] = None

    def __post_init__(self) -> None:
        has_cad = self.cad_target is not None
        has_scan = self.scan_hypothesis is not None
        if has_cad == has_scan:
            raise ValueError("PartProfile must carry exactly one of cad_target or scan_hypothesis")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_mode": self.profile_mode,
            "nominal_source_class": self.nominal_source_class,
            "profile_confidence": round(float(self.profile_confidence), 4),
            "provenance": _json_safe_mapping(self.provenance),
            "cad_target": self.cad_target.to_dict() if self.cad_target else None,
            "scan_hypothesis": self.scan_hypothesis.to_dict() if self.scan_hypothesis else None,
        }


def build_scan_structural_hypothesis(
    *,
    zero_shot_result: Any,
) -> ScanStructuralHypothesis:
    payload = _coerce_scan_payload(zero_shot_result)
    estimated_bend_count = _coerce_nonnegative_int(payload.get("estimated_bend_count"))
    support_breakdown = _json_safe_mapping(dict(payload.get("support_breakdown") or {}))
    line_family_groups = _normalize_line_family_groups(support_breakdown.get("line_family_groups"))
    strip_duplicate_merge_count = _coerce_nonnegative_int(
        payload.get("strip_duplicate_merge_count")
    )
    if strip_duplicate_merge_count is None:
        strip_duplicate_merge_count = _coerce_nonnegative_int(
            support_breakdown.get("strip_duplicate_merge_count")
        ) or 0

    strip_count_confidence_raw = payload.get("strip_count_confidence")
    if strip_count_confidence_raw is None:
        strip_count_confidence_raw = payload.get("hypothesis_confidence")
    try:
        strip_count_confidence = (
            None if strip_count_confidence_raw is None else float(strip_count_confidence_raw)
        )
    except Exception:
        strip_count_confidence = None

    try:
        hypothesis_confidence = float(payload.get("hypothesis_confidence") or 0.0)
    except Exception:
        hypothesis_confidence = 0.0

    return ScanStructuralHypothesis(
        profile_family=str(payload.get("profile_family") or "unknown"),
        estimated_bend_count=estimated_bend_count,
        estimated_bend_count_range=_normalize_count_range(
            payload.get("estimated_bend_count_range"),
            estimated_bend_count=estimated_bend_count,
        ),
        dominant_angle_families_deg=_coerce_float_list(payload.get("dominant_angle_families_deg")),
        line_family_groups=line_family_groups,
        hypothesis_confidence=hypothesis_confidence,
        abstained=bool(payload.get("abstained")),
        abstain_reasons=_normalize_string_list(payload.get("abstain_reasons")),
        support_breakdown=support_breakdown,
        bend_support_strips=tuple(
            _json_safe_mapping(item)
            for item in (payload.get("bend_support_strips") or ())
            if isinstance(item, Mapping)
        ),
        renderable_bend_objects=tuple(
            _json_safe_mapping(item)
            for item in (payload.get("renderable_bend_objects") or ())
            if isinstance(item, Mapping)
        ),
        strip_count_confidence=strip_count_confidence,
        strip_duplicate_merge_count=strip_duplicate_merge_count,
    )


def build_scan_part_profile(
    *,
    zero_shot_result: Any,
    scan_path: Optional[str] = None,
    part_id: Optional[str] = None,
) -> PartProfile:
    payload = _coerce_scan_payload(zero_shot_result)
    scan_hypothesis = build_scan_structural_hypothesis(zero_shot_result=payload)
    resolved_scan_path = str(scan_path or payload.get("scan_path") or "")
    resolved_part_id = str(part_id or payload.get("part_id") or "").strip()
    provenance = {
        "nominal_source_class": "scan_only",
        "source": "zero_shot_bend_profile",
        "scan_path": resolved_scan_path,
        "part_id": resolved_part_id,
        "profile_family": scan_hypothesis.profile_family,
        "abstained": scan_hypothesis.abstained,
    }
    return PartProfile(
        profile_mode="scan_zero_shot",
        nominal_source_class="scan_only",
        profile_confidence=scan_hypothesis.hypothesis_confidence,
        provenance=provenance,
        scan_hypothesis=scan_hypothesis,
    )


def build_cad_part_profile(
    *,
    cad_bends: Sequence[BendSpecification],
    cad_path: Optional[str] = None,
    cad_source_label: Optional[str] = None,
    expected_bend_override: Optional[int] = None,
    spec_bend_angles_hint: Optional[Sequence[Any]] = None,
    spec_schedule_type: Optional[str] = None,
    feature_policy: Optional[Mapping[str, Any]] = None,
    metadata_truth: Optional[Mapping[str, Any]] = None,
    nominal_topology_graph: Optional[Mapping[str, Sequence[str]]] = None,
) -> PartProfile:
    normalized_schedule = normalize_spec_schedule_type(spec_schedule_type)
    normalized_spec_angles = _coerce_float_list(spec_bend_angles_hint)
    normalized_metadata = dict(metadata_truth or {})
    normalized_policy = normalize_feature_policy(feature_policy, metadata_truth=normalized_metadata)

    nominal_source_class = normalize_nominal_source_class(
        cad_path=cad_path,
        cad_source_label=cad_source_label,
    )
    authority_level, profile_confidence = _resolve_authority(nominal_source_class)

    countable_specs = [spec for spec in cad_bends if getattr(spec, "countable_in_regression", True)]
    countable_bend_ids = tuple(str(spec.bend_id) for spec in countable_specs) or tuple(
        str(spec.bend_id) for spec in cad_bends
    )
    expected_bend_count, expected_bend_count_range, expected_count_source = _resolve_expected_count(
        countable_bend_count=len(countable_bend_ids),
        expected_bend_override=expected_bend_override,
        metadata_truth=normalized_metadata,
        spec_bend_angles_hint=normalized_spec_angles,
        spec_schedule_type=normalized_schedule,
    )
    dominant_angle_families_deg, angle_source = _resolve_dominant_angle_families(
        cad_bends=cad_bends,
        spec_bend_angles_hint=normalized_spec_angles,
        metadata_truth=normalized_metadata,
    )
    rolled_allowed = _resolve_rolled_allowed(
        cad_bends=cad_bends,
        countable_feature_policy=normalized_policy,
        metadata_truth=normalized_metadata,
    )
    topology_graph = {
        str(key): tuple(str(item) for item in value)
        for key, value in (nominal_topology_graph or _default_topology_graph(cad_bends)).items()
    }

    provenance = {
        "nominal_source_class": nominal_source_class,
        "cad_path": str(cad_path or ""),
        "cad_source_label": str(cad_source_label or ""),
        "expected_bend_count_source": expected_count_source,
        "angle_family_source": angle_source,
        "spec_schedule_type": normalized_schedule,
        "feature_policy_sources": {
            "input_policy": bool(feature_policy),
            "metadata_truth": bool(normalized_metadata),
        },
    }
    tolerance_context = {
        "angle_tolerances_deg": sorted(
            dict.fromkeys(round(float(spec.tolerance_angle), 4) for spec in cad_bends)
        ),
        "radius_tolerances_mm": sorted(
            dict.fromkeys(round(float(spec.tolerance_radius), 4) for spec in cad_bends)
        ),
    }
    cad_target = CadStructuralTarget(
        nominal_bend_set=countable_bend_ids,
        nominal_topology_graph=topology_graph,
        dominant_angle_families_deg=dominant_angle_families_deg,
        expected_bend_count=expected_bend_count,
        expected_bend_count_range=expected_bend_count_range,
        countable_feature_policy=normalized_policy,
        rolled_allowed=rolled_allowed,
        tolerance_context=tolerance_context,
        authority_level=authority_level,
        provenance=provenance,
    )
    return PartProfile(
        profile_mode="cad_nominal",
        nominal_source_class=nominal_source_class,
        profile_confidence=profile_confidence,
        provenance=provenance,
        cad_target=cad_target,
    )
