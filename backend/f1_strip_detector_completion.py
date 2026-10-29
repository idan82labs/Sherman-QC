"""Deterministic scan-only completion of supported F1 bend-line corridors.

The zero-shot strip detector is deliberately conservative.  On partial scans it
can return a short fragment of a physical bend or omit a bend whose local
cross-section is nevertheless strong in the conventional detector.  This
module reconciles the two hypothesis sources geometrically; it does not inspect
part identifiers, expected counts, CAD, or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class StripDetectorCompletion:
    """Selected bend hypotheses and auditable source-reconciliation details."""

    selected_bend_ids: Tuple[str, ...]
    strip_bend_ids: Tuple[str, ...]
    detector_bend_ids: Tuple[str, ...]
    fragment_promotions: Tuple[Tuple[str, str], ...]
    rejected_duplicate_ids: Tuple[str, ...]


def select_strip_detector_completion(
    bend_hypotheses: Sequence[Any],
    *,
    local_spacing_mm: float,
    min_strip_anchors: int = 3,
) -> StripDetectorCompletion:
    """Select robust strip corridors plus non-redundant detector complements.

    The gate requiring several independent strip anchors prevents this from
    becoming a generic fallback to the conventional detector.  Within each
    collinear corridor, strip evidence wins unless a conventional detector has
    at least 1.75x its supported span, which is treated as deterministic strip
    fragment completion.
    """

    h = max(float(local_spacing_mm or 1.0), 1.0)
    strips = sorted(
        (item for item in bend_hypotheses if _robust_strip(item, h)),
        key=_candidate_sort_key,
    )
    if len(strips) < max(1, int(min_strip_anchors)):
        return StripDetectorCompletion((), (), (), (), ())
    detectors = sorted(
        (item for item in bend_hypotheses if _robust_detector(item, h)),
        key=_candidate_sort_key,
    )
    candidates = [*strips, *detectors]
    clusters = _corridor_clusters(candidates, h)

    selected: list[Any] = []
    promotions: list[Tuple[str, str]] = []
    rejected: list[str] = []
    for cluster in clusters:
        cluster_strips = [item for item in cluster if _source(item) == "zero_shot_strip_seed"]
        cluster_detectors = [item for item in cluster if _source(item) == "detector_bend_seed"]
        best_strip = min(cluster_strips, key=_candidate_sort_key) if cluster_strips else None
        best_detector = min(cluster_detectors, key=_candidate_sort_key) if cluster_detectors else None
        winner = best_strip or best_detector
        if best_strip is not None and best_detector is not None:
            strip_span = _number(best_strip, "span_mm")
            detector_span = _number(best_detector, "span_mm")
            if detector_span >= 1.75 * max(strip_span, h):
                winner = best_detector
                promotions.append((_identifier(best_strip), _identifier(best_detector)))
        if winner is None:
            continue
        selected.append(winner)
        rejected.extend(
            _identifier(item)
            for item in cluster
            if _identifier(item) != _identifier(winner)
        )

    selected.sort(key=lambda item: (_source(item) != "zero_shot_strip_seed", _identifier(item)))
    strip_ids = tuple(_identifier(item) for item in selected if _source(item) == "zero_shot_strip_seed")
    detector_ids = tuple(_identifier(item) for item in selected if _source(item) == "detector_bend_seed")
    return StripDetectorCompletion(
        selected_bend_ids=tuple(_identifier(item) for item in selected),
        strip_bend_ids=strip_ids,
        detector_bend_ids=detector_ids,
        fragment_promotions=tuple(sorted(promotions)),
        rejected_duplicate_ids=tuple(sorted(set(rejected))),
    )


def _corridor_clusters(candidates: Sequence[Any], h: float) -> list[list[Any]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _same_supported_corridor(candidates[left], candidates[right], h):
                union(left, right)
    grouped: dict[int, list[Any]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return [
        sorted(group, key=_candidate_sort_key)
        for _root, group in sorted(grouped.items(), key=lambda item: min(_identifier(x) for x in item[1]))
    ]


def _same_supported_corridor(left: Any, right: Any, h: float) -> bool:
    left_axis = _unit(_vector(left, "axis_direction"))
    right_axis = _unit(_vector(right, "axis_direction"))
    if left_axis is None or right_axis is None:
        return False
    alignment = min(1.0, max(-1.0, abs(_dot(left_axis, right_axis))))
    if math.degrees(math.acos(alignment)) > 8.0:
        return False
    left_anchor = _vector(left, "anchor")
    right_anchor = _vector(right, "anchor")
    delta = _sub(right_anchor, left_anchor)
    axial_offset = _dot(delta, left_axis)
    perpendicular = _norm(_sub(delta, _scale(left_axis, axial_offset)))
    corridor_width = max(
        2.5 * h,
        1.25 * max(_number(left, "cross_width_mm"), _number(right, "cross_width_mm")),
    )
    if perpendicular > corridor_width:
        return False
    interval_gap = max(
        0.0,
        abs(axial_offset) - 0.5 * (_number(left, "span_mm") + _number(right, "span_mm")),
    )
    return interval_gap <= max(3.0 * h, corridor_width)


def _robust_strip(item: Any, h: float) -> bool:
    angle = _number(item, "angle_deg")
    return (
        _source(item) == "zero_shot_strip_seed"
        and _number(item, "confidence") >= 0.60
        and 15.0 <= angle <= 165.0
        and _number(item, "span_mm") >= max(12.0, 5.0 * h)
        and _atom_count(item) >= 3
        and str(_value(item, "visibility_state") or "") != "boundary_truncated"
    )


def _robust_detector(item: Any, h: float) -> bool:
    angle = _number(item, "angle_deg")
    return (
        _source(item) == "detector_bend_seed"
        and _number(item, "confidence") >= 0.80
        and 45.0 <= angle <= 135.0
        and _number(item, "span_mm") >= max(18.0, 5.0 * h)
        and _atom_count(item) >= 5
        and str(_value(item, "visibility_state") or "") == "internal"
    )


def _candidate_sort_key(item: Any) -> tuple[float, float, str]:
    return (-_number(item, "span_mm"), -_number(item, "confidence"), _identifier(item))


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _identifier(item: Any) -> str:
    return str(_value(item, "bend_id") or "")


def _source(item: Any) -> str:
    return str(_value(item, "source_kind") or "")


def _number(item: Any, name: str) -> float:
    try:
        return float(_value(item, name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _atom_count(item: Any) -> int:
    atom_ids = _value(item, "atom_ids") or ()
    return len(tuple(atom_ids))


def _vector(item: Any, name: str) -> Vector3:
    raw = tuple(_value(item, name) or ())
    if len(raw) != 3:
        return (0.0, 0.0, 0.0)
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _unit(vector: Vector3) -> Vector3 | None:
    length = _norm(vector)
    if length <= 1e-12:
        return None
    return _scale(vector, 1.0 / length)


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(value * factor for value in vector)  # type: ignore[return-value]


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))
