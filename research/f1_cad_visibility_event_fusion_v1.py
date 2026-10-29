"""Source-only visibility filtering and late ownership for CAD bend events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VisibilityGates:
    minimum_support_fraction: float = 0.50
    minimum_longest_support_run_fraction: float = 0.50
    minimum_supported_span_mm: float = 12.0
    maximum_median_distance_mm: float = 2.0
    minimum_normal_turn_deg: float = 25.0
    minimum_side_fraction: float = 0.25
    duplicate_axis_angle_deg_max: float = 5.0
    duplicate_transverse_offset_mm_max: float = 4.0
    duplicate_shorter_span_overlap_min: float = 0.60


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def source_visible(record: Mapping[str, Any], gates: VisibilityGates) -> tuple[bool, list[str]]:
    support = record.get("scan_support") or {}
    turn = record.get("normal_turn_verification") or {}
    checks = {
        "PARENT_ACCEPTED": record.get("accepted") is True,
        "CONTINUOUS_SUPPORT": _number(support, "support_fraction") >= gates.minimum_support_fraction,
        "LONGEST_RUN": _number(support, "longest_support_run_fraction") >= gates.minimum_longest_support_run_fraction,
        "FINITE_SPAN": _number(support, "supported_span_mm") >= gates.minimum_supported_span_mm,
        "PROXIMITY": _number(support, "median_distance_mm") <= gates.maximum_median_distance_mm,
        "NORMAL_TURN": _number(turn, "normal_turn_deg") >= gates.minimum_normal_turn_deg,
        "TWO_SIDED_BALANCE": _number(turn, "minimum_side_fraction") >= gates.minimum_side_fraction,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def _line(row: Mapping[str, Any]) -> np.ndarray:
    endpoints = row.get("span_endpoints")
    if endpoints is None:
        start = row.get("detected_line_start", row.get("line_start"))
        end = row.get("detected_line_end", row.get("line_end"))
        endpoints = [start, end]
    line = np.asarray(endpoints, dtype=np.float64)
    if line.shape != (2, 3) or not np.all(np.isfinite(line)):
        raise ValueError("candidate must provide finite 2x3 endpoints")
    if float(np.linalg.norm(line[1] - line[0])) <= 1e-9:
        raise ValueError("candidate line must have positive length")
    return line


def _canonical_direction(line: np.ndarray) -> np.ndarray:
    direction = line[1] - line[0]
    direction = direction / np.linalg.norm(direction)
    dominant = int(np.argmax(np.abs(direction)))
    return -direction if direction[dominant] < 0.0 else direction


def pair_geometry(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    a = _line(left)
    b = _line(right)
    da = _canonical_direction(a)
    db = _canonical_direction(b)
    axis_angle = math.degrees(math.acos(float(np.clip(abs(np.dot(da, db)), 0.0, 1.0))))
    direction = da if np.dot(da, db) >= 0.0 else -da
    origins = np.vstack((a, b))
    anchor = np.mean(origins, axis=0)
    a_projection = (a - anchor) @ direction
    b_projection = (b - anchor) @ direction
    a_center = np.mean(a, axis=0)
    b_center = np.mean(b, axis=0)
    delta = b_center - a_center
    transverse = float(np.linalg.norm(delta - np.dot(delta, direction) * direction))
    overlap = max(0.0, min(float(np.max(a_projection)), float(np.max(b_projection))) - max(float(np.min(a_projection)), float(np.min(b_projection))))
    shorter = min(float(np.ptp(a_projection)), float(np.ptp(b_projection)))
    return {
        "axis_angle_deg": axis_angle,
        "transverse_offset_mm": transverse,
        "shorter_span_overlap": overlap / shorter if shorter > 1e-9 else 0.0,
    }


def duplicate(left: Mapping[str, Any], right: Mapping[str, Any], gates: VisibilityGates) -> tuple[bool, dict[str, float]]:
    geometry = pair_geometry(left, right)
    result = bool(
        geometry["axis_angle_deg"] <= gates.duplicate_axis_angle_deg_max
        and geometry["transverse_offset_mm"] <= gates.duplicate_transverse_offset_mm_max
        and geometry["shorter_span_overlap"] >= gates.duplicate_shorter_span_overlap_min
    )
    return result, geometry


def filter_cad_events(parent: Mapping[str, Any], gates: VisibilityGates) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = {str(row["bend_id"]): row for row in parent.get("predictions") or []}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for record in parent.get("records") or []:
        event_id = str(record.get("event_id") or "")
        visible, failed = source_visible(record, gates)
        prediction = predictions.get(event_id)
        if prediction is None:
            visible = False
            failed = [*failed, "NO_PARENT_PREDICTION"]
        audit.append({
            "event_id": event_id,
            "decision": "VISIBLE_CAD_EVENT" if visible else "NULL_UNOBSERVED_OR_WEAK",
            "failed_gates": failed,
        })
        if visible and prediction is not None:
            line = _line(prediction)
            accepted.append({
                **dict(prediction),
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "detected_line_start": line[0].tolist(),
                "detected_line_end": line[1].tolist(),
                "assignment_source": "CAD_VISIBILITY_VERIFIED_EVENT",
                "visibility_state": "OBSERVED_CONTINUOUS",
            })
    return accepted, audit


def late_fuse(
    scan_candidates: Sequence[Mapping[str, Any]],
    cad_candidates: Sequence[Mapping[str, Any]],
    gates: VisibilityGates,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fused: list[dict[str, Any]] = []
    ownership: list[dict[str, Any]] = []
    for ordinal, row in enumerate(scan_candidates, 1):
        line = _line(row)
        fused.append({
            "bend_id": f"SCAN_{row.get('bend_id', ordinal)}",
            "line_start": line[0].tolist(),
            "line_end": line[1].tolist(),
            "detected_line_start": line[0].tolist(),
            "detected_line_end": line[1].tolist(),
            "assignment_source": "SCAN_ONLY_A0_FIRST_OWNER",
            "visibility_state": "SCAN_OBSERVED",
        })
    for row in cad_candidates:
        owner = None
        owner_geometry = None
        for existing in fused:
            is_duplicate, geometry = duplicate(row, existing, gates)
            if is_duplicate:
                owner = existing["bend_id"]
                owner_geometry = geometry
                break
        if owner is not None:
            ownership.append({
                "cad_event_id": row["bend_id"],
                "decision": "NULL_DUPLICATE_SCAN_OR_PRIOR_EVENT",
                "owner_id": owner,
                "pair_geometry": owner_geometry,
            })
            continue
        added = dict(row)
        added["bend_id"] = "CAD_" + str(row["bend_id"])
        added["assignment_source"] = "CAD_VISIBILITY_VERIFIED_EVENT"
        added["visibility_state"] = "OBSERVED_CONTINUOUS"
        fused.append(added)
        ownership.append({
            "cad_event_id": row["bend_id"],
            "decision": "CAD_EVENT_ADMITTED",
            "owner_id": added["bend_id"],
        })
    return fused, ownership


def contract_payload() -> dict[str, Any]:
    return {
        "schema": "CAD_VISIBILITY_EVENT_CONTRACTION_LATE_FUSION_V1",
        "gates": asdict(VisibilityGates()),
        "scan_candidates_first": True,
        "explicit_null": True,
        "gold_or_expected_counts_used": False,
    }
