"""One-to-one contextual assignment between finite CAD and scan bend lines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from research.f1_cad_visibility_event_fusion_v1 import pair_geometry


@dataclass(frozen=True)
class AssignmentConfig:
    axis_angle_deg_max: float = 10.0
    transverse_offset_mm_max: float = 12.0
    shorter_span_overlap_min: float = 0.25
    parallel_context_radius_mm: float = 30.0
    parallel_context_angle_deg_max: float = 10.0
    parallel_context_overlap_min: float = 0.25
    junction_endpoint_radius_mm: float = 20.0
    junction_angle_deg_min: float = 20.0
    null_cost: float = 2.25
    parallel_degree_weight: float = 0.20
    junction_degree_weight: float = 0.10


def _line(row: Mapping[str, Any]) -> np.ndarray:
    endpoints = row.get("span_endpoints")
    if endpoints is None:
        endpoints = [
            row.get("detected_line_start", row.get("line_start")),
            row.get("detected_line_end", row.get("line_end")),
        ]
    value = np.asarray(endpoints, dtype=np.float64)
    if value.shape != (2, 3) or not np.all(np.isfinite(value)):
        raise ValueError("finite 2x3 line required")
    return value


def _endpoint_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = _line(left)
    b = _line(right)
    return float(min(np.linalg.norm(a[i] - b[j]) for i in range(2) for j in range(2)))


def graph_signatures(rows: Sequence[Mapping[str, Any]], config: AssignmentConfig) -> list[dict[str, int]]:
    result = []
    for index, row in enumerate(rows):
        parallel = 0
        junction = 0
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            geometry = pair_geometry(row, other)
            if (
                geometry["axis_angle_deg"] <= config.parallel_context_angle_deg_max
                and geometry["transverse_offset_mm"] <= config.parallel_context_radius_mm
                and geometry["shorter_span_overlap"] >= config.parallel_context_overlap_min
            ):
                parallel += 1
            if (
                geometry["axis_angle_deg"] >= config.junction_angle_deg_min
                and _endpoint_distance(row, other) <= config.junction_endpoint_radius_mm
            ):
                junction += 1
        result.append({"parallel_degree": parallel, "junction_degree": junction})
    return result


def _eligible(geometry: Mapping[str, float], config: AssignmentConfig) -> bool:
    return bool(
        geometry["axis_angle_deg"] <= config.axis_angle_deg_max
        and geometry["transverse_offset_mm"] <= config.transverse_offset_mm_max
        and geometry["shorter_span_overlap"] >= config.shorter_span_overlap_min
    )


def _cost(
    geometry: Mapping[str, float],
    cad_signature: Mapping[str, int],
    scan_signature: Mapping[str, int],
    config: AssignmentConfig,
) -> float:
    return float(
        geometry["axis_angle_deg"] / config.axis_angle_deg_max
        + geometry["transverse_offset_mm"] / config.transverse_offset_mm_max
        + (1.0 - geometry["shorter_span_overlap"])
        + config.parallel_degree_weight * abs(cad_signature["parallel_degree"] - scan_signature["parallel_degree"])
        + config.junction_degree_weight * abs(cad_signature["junction_degree"] - scan_signature["junction_degree"])
    )


def assign(
    cad_rows: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
    config: AssignmentConfig,
) -> dict[str, Any]:
    cad_signatures = graph_signatures(cad_rows, config)
    scan_signatures = graph_signatures(scan_rows, config)
    pair_records: dict[tuple[int, int], dict[str, Any]] = {}
    if not cad_rows:
        return {"matches": [], "unmatched_cad": [], "cad_signatures": [], "scan_signatures": scan_signatures, "pair_records": []}
    width = len(scan_rows) + len(cad_rows)
    matrix = np.full((len(cad_rows), width), 1e6, dtype=np.float64)
    for cad_index, cad in enumerate(cad_rows):
        for scan_index, scan in enumerate(scan_rows):
            geometry = pair_geometry(cad, scan)
            eligible = _eligible(geometry, config)
            value = _cost(geometry, cad_signatures[cad_index], scan_signatures[scan_index], config) if eligible else 1e6
            matrix[cad_index, scan_index] = value
            pair_records[(cad_index, scan_index)] = {
                "cad_index": cad_index,
                "scan_index": scan_index,
                "eligible": eligible,
                "cost": None if not eligible else value,
                "geometry": geometry,
            }
        matrix[cad_index, len(scan_rows) + cad_index] = config.null_cost
    row_indices, column_indices = linear_sum_assignment(matrix)
    matches = []
    matched_cad = set()
    for cad_index, column in zip(row_indices.tolist(), column_indices.tolist()):
        if column < len(scan_rows) and matrix[cad_index, column] < config.null_cost:
            record = pair_records[(cad_index, column)]
            matches.append(record)
            matched_cad.add(cad_index)
    return {
        "matches": matches,
        "unmatched_cad": [index for index in range(len(cad_rows)) if index not in matched_cad],
        "cad_signatures": cad_signatures,
        "scan_signatures": scan_signatures,
        "pair_records": [pair_records[key] for key in sorted(pair_records)],
    }


def fuse(
    cad_rows: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
    config: AssignmentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment = assign(cad_rows, scan_rows, config)
    fused = [dict(row) for row in scan_rows]
    match_by_cad = {row["cad_index"]: row for row in assignment["matches"]}
    audit = []
    for cad_index, cad in enumerate(cad_rows):
        if cad_index in match_by_cad:
            match = match_by_cad[cad_index]
            audit.append({
                "cad_event_id": cad["bend_id"],
                "decision": "SCAN_OWNS_ASSIGNED_EVENT",
                "scan_owner_id": scan_rows[match["scan_index"]]["bend_id"],
                "assignment_cost": match["cost"],
                "pair_geometry": match["geometry"],
            })
            continue
        eligible_conflicts = [
            row for row in assignment["pair_records"]
            if row["cad_index"] == cad_index and row["eligible"]
        ]
        if eligible_conflicts:
            audit.append({
                "cad_event_id": cad["bend_id"],
                "decision": "NULL_ONE_TO_ONE_CONFLICT",
                "eligible_scan_indices": [row["scan_index"] for row in eligible_conflicts],
            })
            continue
        added = dict(cad)
        added["bend_id"] = "CAD_" + str(cad["bend_id"])
        added["assignment_source"] = "UNMATCHED_VISIBLE_CAD_EVENT"
        fused.append(added)
        audit.append({"cad_event_id": cad["bend_id"], "decision": "UNMATCHED_CAD_EVENT_ADMITTED"})
    return fused, {**assignment, "decisions": audit}


def contract_payload() -> dict[str, Any]:
    return {
        "schema": "JOINT_FINITE_SPAN_CONTEXTUAL_CAD_SCAN_ASSIGNMENT_V1",
        "assignment": asdict(AssignmentConfig()),
        "hard_one_to_one": True,
        "explicit_null": True,
        "gold_or_expected_counts_used": False,
    }
