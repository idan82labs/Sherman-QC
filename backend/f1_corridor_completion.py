"""Deterministic line-corridor completion for scan-only F1 bend evidence.

The patch-graph solver deliberately proposes an overcomplete interface-birth
pool.  This module reduces that pool geometrically: robust zero-shot strips are
first deduplicated as finite 3-D line segments, then interface births are
clustered by their infinite supporting line and split only at observed gaps.
Birth components already covered by a strip are suppressed.

There is intentionally no target bend count in this API.  The number of output
corridors is a consequence of the observed line offsets and finite-span gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CorridorEvidence:
    evidence_id: str
    source_kind: str
    anchor: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    span_mm: float
    cross_width_mm: float
    confidence: float
    support_ids: tuple[str, ...]
    atom_ids: tuple[str, ...]
    incident_flange_pairs: tuple[tuple[str, str], ...]

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        axis = _canonical_axis(self.axis_direction)
        anchor = np.asarray(self.anchor, dtype=np.float64)
        half_span = 0.5 * max(0.0, float(self.span_mm))
        return anchor - axis * half_span, anchor + axis * half_span


def corridor_completion_applicability(
    *,
    route_payload: Mapping[str, Any] | None,
    strip_hypothesis_count: int,
    interface_birth_hypothesis_count: int,
) -> dict[str, Any]:
    """Gate the repair to high-fragmentation full repeated-near-90 scans.

    Partial scans and stable zero-shot routes must not inherit an interface-
    completion move merely because an overcomplete diagnostic pool exists.
    """

    route = dict(route_payload or {})
    snapshot = dict(route.get("feature_snapshot") or {})
    reasons: list[str] = []
    if str(route.get("route_id") or "") != "fragmented_repeat90_interface_birth":
        reasons.append("route_not_fragmented_repeat90_interface_birth")
    if str(snapshot.get("scan_state_hint") or "").lower() == "partial":
        reasons.append("partial_scan_not_eligible")
    if float(snapshot.get("repeated_near90_strength") or 0.0) < 0.75:
        reasons.append("repeated_near90_strength_too_low")
    if float(snapshot.get("candidate_fragmentation") or 0.0) < 1.40:
        reasons.append("candidate_fragmentation_too_low")
    if int(strip_hypothesis_count) < 3:
        reasons.append("insufficient_strip_corridors")
    if int(interface_birth_hypothesis_count) < 4:
        reasons.append("insufficient_interface_birth_pool")
    return {"applicable": not reasons, "reason_codes": reasons}


def complete_scan_supported_corridors(
    *,
    strip_hypotheses: Sequence[Any],
    interface_birth_hypotheses: Sequence[Any],
    local_spacing_mm: float,
    strip_min_confidence: float = 0.60,
    axis_tolerance_deg: float = 5.0,
) -> dict[str, Any]:
    """Return scan-supported strip corridors plus uncovered birth components.

    Hypotheses may be dataclasses or dictionaries.  Ground-truth counts and
    labels are not accepted by design.
    """

    h = max(float(local_spacing_mm), 1.0)
    robust_strips = [
        item
        for item in (_to_evidence(value) for value in strip_hypotheses)
        if item.confidence >= strip_min_confidence
        and item.span_mm >= max(12.0, 5.0 * h)
        and len(item.atom_ids) >= 3
    ]
    robust_births = [
        item
        for item in (_to_evidence(value) for value in interface_birth_hypotheses)
        if item.confidence >= 0.50
        and item.span_mm >= max(4.0, 2.0 * h)
        and len(item.atom_ids) >= 3
    ]

    # Plane-pair intersections from adjacent atom fragments can drift by about
    # four voxels even when they describe the same physical interface.  Finite
    # support gaps are allowed one additional voxel because atomization removes
    # boundary samples on both sides of a gap.
    offset_tolerance = max(8.0, 4.0 * h)
    gap_tolerance = max(10.0, 5.0 * h)
    strip_components = _finite_corridor_components(
        robust_strips,
        axis_tolerance_deg=axis_tolerance_deg,
        offset_tolerance_mm=offset_tolerance,
        gap_tolerance_mm=gap_tolerance,
    )
    selected_strips = [
        _merge_component(component, source_kind="zero_shot_strip_corridor", ordinal=index + 1)
        for index, component in enumerate(strip_components)
    ]

    birth_line_families = _infinite_line_families(
        robust_births,
        axis_tolerance_deg=axis_tolerance_deg,
        offset_tolerance_mm=offset_tolerance,
    )
    uncovered_births: list[CorridorEvidence] = []
    covered_births: list[dict[str, Any]] = []
    rejected_births: list[dict[str, Any]] = []
    next_ordinal = 1
    for family in birth_line_families:
        for component in _split_family_at_gaps(family, gap_tolerance_mm=gap_tolerance):
            merged = _merge_component(
                component,
                source_kind="uncovered_interface_corridor",
                ordinal=next_ordinal,
            )
            next_ordinal += 1
            covering_strip = _covering_corridor(
                merged,
                selected_strips,
                axis_tolerance_deg=axis_tolerance_deg,
                offset_tolerance_mm=offset_tolerance,
                gap_tolerance_mm=gap_tolerance,
            )
            if covering_strip is not None:
                covered_births.append(
                    {
                        "birth_corridor_id": merged.evidence_id,
                        "covered_by": covering_strip.evidence_id,
                        "source_ids": list(merged.support_ids),
                    }
                )
                continue
            # A single tiny subspan is not enough to create a new physical
            # object.  Two independent proposals, or one long/dense proposal,
            # are required.  This is evidence gating, not count forcing.
            independently_supported = len(merged.support_ids) >= 2
            strong_single = len(merged.atom_ids) >= 10 and merged.span_mm >= max(16.0, 6.0 * h)
            if not independently_supported and not strong_single:
                rejected_births.append(
                    {
                        "birth_corridor_id": merged.evidence_id,
                        "reason_codes": ["insufficient_independent_birth_support"],
                        "source_ids": list(merged.support_ids),
                    }
                )
                continue
            uncovered_births.append(merged)

    selected = sorted(
        [*selected_strips, *uncovered_births],
        key=lambda item: (item.source_kind, item.evidence_id),
    )
    return {
        "algorithm": "scan_corridor_completion_v1",
        "local_spacing_mm": round(h, 6),
        "parameters": {
            "axis_tolerance_deg": float(axis_tolerance_deg),
            "offset_tolerance_mm": round(offset_tolerance, 6),
            "gap_tolerance_mm": round(gap_tolerance, 6),
            "strip_min_confidence": float(strip_min_confidence),
        },
        "input_counts": {
            "robust_strips": len(robust_strips),
            "robust_interface_births": len(robust_births),
        },
        "selected_strip_count": len(selected_strips),
        "selected_uncovered_birth_count": len(uncovered_births),
        "selected_count": len(selected),
        "selected": [_evidence_to_dict(item) for item in selected],
        "covered_birth_components": covered_births,
        "rejected_birth_components": rejected_births,
    }


def _to_evidence(value: Any) -> CorridorEvidence:
    def field(name: str, default: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    pairs = field("candidate_incident_flange_pairs", ()) or ()
    if not pairs:
        incident = tuple(str(item) for item in (field("incident_flange_ids", ()) or ()))
        pairs = (incident[:2],) if len(incident) >= 2 else ()
    canonical_pairs = sorted(
        {
            tuple(sorted((str(pair[0]), str(pair[1]))))
            for pair in pairs
            if isinstance(pair, Sequence) and len(pair) >= 2 and str(pair[0]) != str(pair[1])
        }
    )
    evidence_id = str(field("bend_id", field("evidence_id", "unknown")))
    source_ids = field("support_ids", ()) or (evidence_id,)
    return CorridorEvidence(
        evidence_id=evidence_id,
        source_kind=str(field("source_kind", "unknown")),
        anchor=tuple(float(item) for item in field("anchor", (0.0, 0.0, 0.0))),
        axis_direction=tuple(float(item) for item in field("axis_direction", (1.0, 0.0, 0.0))),
        span_mm=float(field("span_mm", 0.0)),
        cross_width_mm=float(field("cross_width_mm", 0.0)),
        confidence=float(field("confidence", 0.0)),
        support_ids=tuple(sorted(str(item) for item in source_ids)),
        atom_ids=tuple(sorted(str(item) for item in (field("atom_ids", ()) or ()))),
        incident_flange_pairs=tuple(canonical_pairs),
    )


def _canonical_axis(direction: Iterable[float]) -> np.ndarray:
    axis = np.asarray(tuple(direction), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    axis /= norm
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    return axis


def _axis_angle_deg(left: CorridorEvidence, right: CorridorEvidence) -> float:
    dot = float(np.clip(abs(np.dot(_canonical_axis(left.axis_direction), _canonical_axis(right.axis_direction))), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def _parallel_line_offset_mm(left: CorridorEvidence, right: CorridorEvidence) -> float:
    axis = _canonical_axis(np.asarray(left.axis_direction) + np.asarray(right.axis_direction))
    delta = np.asarray(left.anchor, dtype=np.float64) - np.asarray(right.anchor, dtype=np.float64)
    return float(np.linalg.norm(delta - axis * float(np.dot(delta, axis))))


def _projection_interval(evidence: CorridorEvidence, axis: np.ndarray) -> tuple[float, float]:
    start, end = evidence.endpoints
    values = (float(np.dot(start, axis)), float(np.dot(end, axis)))
    return min(values), max(values)


def _interval_gap(left: tuple[float, float], right: tuple[float, float]) -> float:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0.0


def _infinite_line_families(
    evidence: Sequence[CorridorEvidence],
    *,
    axis_tolerance_deg: float,
    offset_tolerance_mm: float,
) -> list[list[CorridorEvidence]]:
    return _union_components(
        evidence,
        linked=lambda left, right: (
            _axis_angle_deg(left, right) <= axis_tolerance_deg
            and _parallel_line_offset_mm(left, right) <= offset_tolerance_mm
        ),
    )


def _finite_corridor_components(
    evidence: Sequence[CorridorEvidence],
    *,
    axis_tolerance_deg: float,
    offset_tolerance_mm: float,
    gap_tolerance_mm: float,
) -> list[list[CorridorEvidence]]:
    return _union_components(
        evidence,
        linked=lambda left, right: (
            _axis_angle_deg(left, right) <= axis_tolerance_deg
            and _parallel_line_offset_mm(left, right) <= offset_tolerance_mm
            and _interval_gap(
                _projection_interval(left, _canonical_axis(left.axis_direction)),
                _projection_interval(right, _canonical_axis(left.axis_direction)),
            )
            <= gap_tolerance_mm
        ),
    )


def _split_family_at_gaps(
    family: Sequence[CorridorEvidence],
    *,
    gap_tolerance_mm: float,
) -> list[list[CorridorEvidence]]:
    if not family:
        return []
    axis = _mean_axis(family)
    return _union_components(
        family,
        linked=lambda left, right: (
            _interval_gap(_projection_interval(left, axis), _projection_interval(right, axis)) <= gap_tolerance_mm
        ),
    )


def _union_components(
    evidence: Sequence[CorridorEvidence],
    *,
    linked: Any,
) -> list[list[CorridorEvidence]]:
    ordered = sorted(evidence, key=lambda item: item.evidence_id)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            if linked(left, ordered[right_index]):
                left_root, right_root = find(left_index), find(right_index)
                if left_root != right_root:
                    parent[right_root] = left_root
    groups: dict[int, list[CorridorEvidence]] = {}
    for index, item in enumerate(ordered):
        groups.setdefault(find(index), []).append(item)
    return sorted(groups.values(), key=lambda group: tuple(item.evidence_id for item in group))


def _mean_axis(component: Sequence[CorridorEvidence]) -> np.ndarray:
    reference = _canonical_axis(component[0].axis_direction)
    vectors = []
    for item in component:
        axis = _canonical_axis(item.axis_direction)
        if float(np.dot(reference, axis)) < 0.0:
            axis *= -1.0
        vectors.append(axis * max(1.0, item.span_mm))
    return _canonical_axis(np.sum(vectors, axis=0))


def _merge_component(
    component: Sequence[CorridorEvidence],
    *,
    source_kind: str,
    ordinal: int,
) -> CorridorEvidence:
    axis = _mean_axis(component)
    weights = np.asarray([max(1.0, item.span_mm) for item in component], dtype=np.float64)
    anchors = np.asarray([item.anchor for item in component], dtype=np.float64)
    mean_anchor = np.average(anchors, axis=0, weights=weights)
    intervals = [_projection_interval(item, axis) for item in component]
    low = min(item[0] for item in intervals)
    high = max(item[1] for item in intervals)
    along = 0.5 * (low + high)
    anchor = mean_anchor + axis * (along - float(np.dot(mean_anchor, axis)))
    support_ids = sorted({support for item in component for support in item.support_ids})
    atom_ids = sorted({atom_id for item in component for atom_id in item.atom_ids})
    pairs = sorted({pair for item in component for pair in item.incident_flange_pairs})
    prefix = "ZS" if source_kind == "zero_shot_strip_corridor" else "IFC"
    return CorridorEvidence(
        evidence_id=f"{prefix}{ordinal}",
        source_kind=source_kind,
        anchor=tuple(float(value) for value in anchor),
        axis_direction=tuple(float(value) for value in axis),
        span_mm=float(max(0.0, high - low)),
        cross_width_mm=float(max(item.cross_width_mm for item in component)),
        confidence=float(max(item.confidence for item in component)),
        support_ids=tuple(support_ids),
        atom_ids=tuple(atom_ids),
        incident_flange_pairs=tuple(pairs),
    )


def _covering_corridor(
    candidate: CorridorEvidence,
    strips: Sequence[CorridorEvidence],
    *,
    axis_tolerance_deg: float,
    offset_tolerance_mm: float,
    gap_tolerance_mm: float,
) -> CorridorEvidence | None:
    candidate_axis = _canonical_axis(candidate.axis_direction)
    candidate_interval = _projection_interval(candidate, candidate_axis)
    matches = []
    for strip in strips:
        if _axis_angle_deg(candidate, strip) > axis_tolerance_deg:
            continue
        offset = _parallel_line_offset_mm(candidate, strip)
        if offset > offset_tolerance_mm:
            continue
        gap = _interval_gap(candidate_interval, _projection_interval(strip, candidate_axis))
        if gap <= gap_tolerance_mm:
            matches.append((offset, gap, strip.evidence_id, strip))
    return min(matches, default=(0.0, 0.0, "", None))[3] if matches else None


def _evidence_to_dict(item: CorridorEvidence) -> dict[str, Any]:
    start, end = item.endpoints
    return {
        "bend_id": item.evidence_id,
        "source_kind": item.source_kind,
        "anchor": [round(float(value), 8) for value in item.anchor],
        "axis_direction": [round(float(value), 8) for value in item.axis_direction],
        "span_mm": round(float(item.span_mm), 6),
        "cross_width_mm": round(float(item.cross_width_mm), 6),
        "confidence": round(float(item.confidence), 6),
        "support_ids": list(item.support_ids),
        "atom_ids": list(item.atom_ids),
        "incident_flange_pairs": [list(pair) for pair in item.incident_flange_pairs],
        "span_endpoints": [
            [round(float(value), 8) for value in start],
            [round(float(value), 8) for value in end],
        ],
    }
