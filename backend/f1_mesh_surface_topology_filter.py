"""Deterministic, identity-free filtering for scan-mesh bend candidates.

The filter deliberately consumes only candidate geometry and surface topology.  It
does not accept a part identifier, expected bend count, CAD data, or labels.  It is
kept in the research lane until frozen-corpus and visual promotion gates pass.
"""

from __future__ import annotations

from math import acos, degrees, sqrt
from typing import Any, Mapping, Sequence


DEFAULT_CONFIG: dict[str, float] = {
    "max_surface_gap_mm": 6.0,
    "min_transition_score": 0.60,
    "max_direct_bend_angle_deg": 110.0,
    "min_supported_surface_area_mm2": 500.0,
    "min_area_span_ratio": 10.0,
    "sparse_base_max_count": 1.0,
    "short_max_span_mm": 40.0,
    "short_max_surface_gap_mm": 2.0,
    "short_min_area_span_ratio": 8.0,
    "short_pair_min_angle_deg": 20.0,
    "high_angle_sparse_base_max_count": 3.0,
    "high_angle_min_deg": 120.0,
    "high_angle_min_transition_score": 0.70,
    "high_angle_min_span_mm": 50.0,
    "duplicate_max_angle_deg": 8.0,
    "duplicate_max_offset_mm": 4.5,
    "duplicate_min_overlap_fraction": 0.45,
}


def filter_mesh_surface_topology(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Select physical bend axes from raw scan-mesh adjacency candidates.

    The returned candidates preserve input geometry and receive an auditable
    ``mesh_topology_filter`` annotation.  Input ordering is the final stable
    tiebreak, so repeated runs are byte-for-byte deterministic after JSON dump.
    """

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update({key: float(value) for key, value in config.items()})

    mesh = [
        (index, dict(candidate))
        for index, candidate in enumerate(candidates)
        if candidate.get("candidate_source") == "scan_mesh_adjacency"
    ]
    surface_areas = _surface_areas([candidate for _, candidate in mesh])
    dominant_surface = _dominant_surface(surface_areas)

    base: list[tuple[int, dict[str, Any], str]] = []
    rejected: dict[str, list[str]] = {}
    for index, candidate in mesh:
        reasons = _base_reasons(candidate, dominant_surface, cfg)
        if reasons:
            base.append((index, candidate, reasons[0]))
        else:
            rejected[_candidate_id(candidate, index)] = _rejection_reasons(
                candidate, dominant_surface, cfg
            )

    additions: list[tuple[int, dict[str, Any], str]] = []
    short_eligible: list[tuple[int, dict[str, Any]]] = []
    if len(base) <= int(cfg["sparse_base_max_count"]):
        short_eligible = [
            (index, candidate)
            for index, candidate in mesh
            if not any(index == selected_index for selected_index, _, _ in base)
            and _is_short_supported(candidate, cfg)
        ]
        if (
            len(short_eligible) == 2
            and _line_angle_deg(short_eligible[0][1], short_eligible[1][1])
            >= cfg["short_pair_min_angle_deg"]
        ):
            additions.extend(
                (index, candidate, "sparse_nonparallel_short_pair")
                for index, candidate in short_eligible
            )

    high_angle_eligible: list[tuple[int, dict[str, Any]]] = []
    if len(base) <= int(cfg["high_angle_sparse_base_max_count"]):
        high_angle_eligible = [
            (index, candidate)
            for index, candidate in mesh
            if not any(index == selected_index for selected_index, _, _ in base)
            and _is_high_angle_supported(candidate, cfg)
        ]
        if len(high_angle_eligible) >= 2:
            additions.extend(
                (index, candidate, "sparse_paired_high_angle_transition")
                for index, candidate in high_angle_eligible
            )

    selected: list[tuple[int, dict[str, Any], str]] = list(base)
    duplicate_ids: list[str] = []
    for item in additions:
        if any(_lines_are_duplicates(item[1], current[1], cfg) for current in selected):
            duplicate_ids.append(_candidate_id(item[1], item[0]))
            continue
        selected.append(item)

    selected.sort(key=lambda item: item[0])
    selected_payloads = []
    for index, candidate, reason in selected:
        candidate["mesh_topology_filter"] = {
            "selected": True,
            "reason_code": reason,
            "dominant_surface_id": dominant_surface,
        }
        selected_payloads.append(candidate)

    return {
        "selected": selected_payloads,
        "diagnostics": {
            "algorithm": "f1_mesh_surface_topology_filter_v1",
            "input_mesh_candidate_count": len(mesh),
            "selected_count": len(selected_payloads),
            "dominant_surface_id": dominant_surface,
            "surface_areas_mm2": {
                str(key): round(value, 6) for key, value in sorted(surface_areas.items())
            },
            "base_selected_ids": [
                _candidate_id(candidate, index) for index, candidate, _ in base
            ],
            "short_eligible_ids": [
                _candidate_id(candidate, index) for index, candidate in short_eligible
            ],
            "high_angle_eligible_ids": [
                _candidate_id(candidate, index) for index, candidate in high_angle_eligible
            ],
            "duplicate_supplement_ids": duplicate_ids,
            "rejected": rejected,
            "config": cfg,
            "inference_contract": {
                "uses_part_identifier": False,
                "uses_expected_count": False,
                "uses_manual_truth": False,
                "uses_cad": False,
            },
        },
    }


def route_mesh_topology_profile(
    candidate_payload: Mapping[str, Any],
    topology_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose an offline completion profile from geometry-only diagnostics.

    A six-surface closed adjacency skeleton in which all six mesh candidates
    survive the direct-transition gates is the observed signature for formed
    prismatic parts whose four internal bends are absent from mesh adjacency.
    The completion itself still has to independently support every added axis.
    """

    mesh_count = sum(
        1
        for candidate in candidate_payload.get("candidates") or ()
        if candidate.get("candidate_source") == "scan_mesh_adjacency"
    )
    diagnostics = dict(topology_result.get("diagnostics") or {})
    selected_count = len(topology_result.get("selected") or ())
    base_count = len(diagnostics.get("base_selected_ids") or ())
    mesh_diagnostics = dict(
        (candidate_payload.get("generator_diagnostics") or {}).get("scan_mesh_adjacency")
        or {}
    )
    major_surface_count = int(mesh_diagnostics.get("major_surface_count") or 0)
    formed_prismatic = (
        mesh_count == 6
        and selected_count == 6
        and base_count == 6
        and major_surface_count == 6
    )
    return {
        "route_id": (
            "mesh_closed6_formed_prismatic_completion"
            if formed_prismatic
            else "mesh_surface_topology_filter"
        ),
        "enable_prismatic_plane_lattice_completion": formed_prismatic,
        "enable_interior_formed_transition_completion": formed_prismatic,
        "reason_codes": (
            [
                "six_mesh_adjacencies",
                "six_major_surfaces",
                "all_mesh_adjacencies_directly_supported",
                "completion_requires_independent_scan_support",
            ]
            if formed_prismatic
            else ["no_formed_prismatic_signature"]
        ),
        "feature_snapshot": {
            "mesh_candidate_count": mesh_count,
            "major_surface_count": major_surface_count,
            "base_selected_count": base_count,
            "selected_count": selected_count,
        },
        "inference_contract": {
            "uses_part_identifier": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
        },
    }


def _surface_areas(candidates: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    areas: dict[int, float] = {}
    for candidate in candidates:
        for side in (1, 2):
            surface_id = _as_int(candidate.get(f"surface{side}_id"))
            area = _as_float(candidate.get(f"surface{side}_area"))
            if surface_id is not None and area > areas.get(surface_id, 0.0):
                areas[surface_id] = area
    return areas


def _dominant_surface(surface_areas: Mapping[int, float]) -> int | None:
    if not surface_areas:
        return None
    return min(surface_areas, key=lambda key: (-surface_areas[key], key))


def _base_reasons(
    candidate: Mapping[str, Any], dominant_surface: int | None, cfg: Mapping[str, float]
) -> list[str]:
    span = _span(candidate)
    min_area = _min_surface_area(candidate)
    if (
        span <= 0.0
        or _as_float(candidate.get("min_surface_gap"), 1e9) > cfg["max_surface_gap_mm"]
        or _as_float(candidate.get("bend_angle"), 180.0) > cfg["max_direct_bend_angle_deg"]
        or _as_float(candidate.get("two_sided_transition_score")) < cfg["min_transition_score"]
        or min_area < cfg["min_supported_surface_area_mm2"]
    ):
        return []
    if min_area / span >= cfg["min_area_span_ratio"]:
        return ["area_span_supported_direct_transition"]
    if dominant_surface is not None and dominant_surface in (
        _as_int(candidate.get("surface1_id")),
        _as_int(candidate.get("surface2_id")),
    ):
        return ["dominant_face_direct_transition"]
    return []


def _rejection_reasons(
    candidate: Mapping[str, Any], dominant_surface: int | None, cfg: Mapping[str, float]
) -> list[str]:
    reasons: list[str] = []
    span = _span(candidate)
    min_area = _min_surface_area(candidate)
    if span <= 0.0:
        reasons.append("invalid_span")
    if _as_float(candidate.get("min_surface_gap"), 1e9) > cfg["max_surface_gap_mm"]:
        reasons.append("surface_gap")
    if _as_float(candidate.get("bend_angle"), 180.0) > cfg["max_direct_bend_angle_deg"]:
        reasons.append("non_direct_angle")
    if _as_float(candidate.get("two_sided_transition_score")) < cfg["min_transition_score"]:
        reasons.append("weak_transition")
    if min_area < cfg["min_supported_surface_area_mm2"]:
        reasons.append("small_surface")
    if span > 0.0 and min_area / span < cfg["min_area_span_ratio"]:
        incident = dominant_surface is not None and dominant_surface in (
            _as_int(candidate.get("surface1_id")),
            _as_int(candidate.get("surface2_id")),
        )
        if not incident:
            reasons.append("weak_area_span_without_dominant_face")
    return reasons or ["not_selected"]


def _is_short_supported(candidate: Mapping[str, Any], cfg: Mapping[str, float]) -> bool:
    span = _span(candidate)
    return (
        0.0 < span <= cfg["short_max_span_mm"]
        and _as_float(candidate.get("min_surface_gap"), 1e9)
        <= cfg["short_max_surface_gap_mm"]
        and _min_surface_area(candidate) / span >= cfg["short_min_area_span_ratio"]
    )


def _is_high_angle_supported(candidate: Mapping[str, Any], cfg: Mapping[str, float]) -> bool:
    return (
        _as_float(candidate.get("min_surface_gap"), 1e9) <= cfg["max_surface_gap_mm"]
        and _as_float(candidate.get("bend_angle")) >= cfg["high_angle_min_deg"]
        and _as_float(candidate.get("two_sided_transition_score"))
        >= cfg["high_angle_min_transition_score"]
        and _span(candidate) >= cfg["high_angle_min_span_mm"]
    )


def _lines_are_duplicates(
    left: Mapping[str, Any], right: Mapping[str, Any], cfg: Mapping[str, float]
) -> bool:
    if _line_angle_deg(left, right) > cfg["duplicate_max_angle_deg"]:
        return False
    p0, p1 = _line(left)
    q0, q1 = _line(right)
    direction = _unit(_sub(p1, p0))
    if direction is None:
        return False
    midpoint_delta = _sub(_midpoint(q0, q1), _midpoint(p0, p1))
    axial = _scale(direction, _dot(midpoint_delta, direction))
    offset = _norm(_sub(midpoint_delta, axial))
    if offset > cfg["duplicate_max_offset_mm"]:
        return False
    p_projection = sorted((_dot(p0, direction), _dot(p1, direction)))
    q_projection = sorted((_dot(q0, direction), _dot(q1, direction)))
    overlap = max(0.0, min(p_projection[1], q_projection[1]) - max(p_projection[0], q_projection[0]))
    shortest = min(p_projection[1] - p_projection[0], q_projection[1] - q_projection[0])
    return shortest > 0.0 and overlap / shortest >= cfg["duplicate_min_overlap_fraction"]


def _line_angle_deg(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    lp0, lp1 = _line(left)
    rp0, rp1 = _line(right)
    left_unit = _unit(_sub(lp1, lp0))
    right_unit = _unit(_sub(rp1, rp0))
    if left_unit is None or right_unit is None:
        return 90.0
    cosine = max(-1.0, min(1.0, abs(_dot(left_unit, right_unit))))
    return degrees(acos(cosine))


def _line(candidate: Mapping[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return _point(candidate.get("line_start")), _point(candidate.get("line_end"))


def _span(candidate: Mapping[str, Any]) -> float:
    start, end = _line(candidate)
    return _norm(_sub(end, start))


def _min_surface_area(candidate: Mapping[str, Any]) -> float:
    return min(
        _as_float(candidate.get("surface1_area")),
        _as_float(candidate.get("surface2_area")),
    )


def _candidate_id(candidate: Mapping[str, Any], fallback: int) -> str:
    return str(candidate.get("candidate_id") or f"mesh_candidate_{fallback:04d}")


def _point(value: Any) -> tuple[float, float, float]:
    values = list(value or ())
    if len(values) != 3:
        return (0.0, 0.0, 0.0)
    return tuple(_as_float(item) for item in values)  # type: ignore[return-value]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(value: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (value[0] * factor, value[1] * factor, value[2] * factor)


def _midpoint(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0, (left[2] + right[2]) / 2.0)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(value: tuple[float, float, float]) -> float:
    return sqrt(_dot(value, value))


def _unit(value: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = _norm(value)
    return None if norm <= 1e-9 else _scale(value, 1.0 / norm)
