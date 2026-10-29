"""Canonical compact candidate/event intermediate representation for F1 research.

The IR is deliberately geometry- and provenance-only.  It cannot contain raw
points, source point indices, corpus identity, filenames, expected counts,
truth, or predictions from another lane.  Adapters accept the historical line,
span-endpoint, and anchor/axis/span shapes but always emit one byte-stable form.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "F1_CANDIDATE_IR_V1"
STAGE_GRAPH_SCHEMA = "F1_STAGE_GRAPH_V1"
UNITS = "mm"
COORDINATE_FRAME = "RAW_SCAN"
QUANTUM = Decimal("0.000001")
MAX_FEATURES = 64
MAX_SOURCE_IDS = 32
MAX_TEXT_LENGTH = 128
PROHIBITED_KEYS = frozenset({
    "points", "source_points", "raw_points", "point_indices", "source_point_indices",
    "vertices", "triangles", "mesh", "point_cloud", "filename", "file_name", "path",
    "part_id", "part_key", "part_name", "family", "family_id", "expected_count",
    "expected_counts", "gold", "truth", "label", "labels", "manual_bends",
    "prediction", "predictions", "cad_path", "scan_path",
})
SUPPORT_STATUSES = frozenset({"SUPPORTED", "INSUFFICIENT", "UNKNOWN"})
PROHIBITED_COMPACT_KEYS = frozenset({
    "points", "sourcepoints", "rawpoints", "pointindices", "sourcepointindices",
    "vertices", "triangles", "mesh", "pointcloud", "filename", "filepath", "path",
    "partid", "partkey", "partname", "family", "familyid", "expectedcount",
    "expectedcounts", "expectedbendcount", "targetcount", "gold", "truth", "label",
    "labels", "manualbends", "prediction", "predictions", "cadpath", "scanpath",
})


class CandidateIRValidationError(ValueError):
    pass


def _reject_prohibited(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            compact = "".join(char for char in folded if char.isalnum())
            if folded in PROHIBITED_KEYS or compact in PROHIBITED_COMPACT_KEYS:
                raise CandidateIRValidationError(f"{path}.{key} is prohibited")
            _reject_prohibited(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_prohibited(child, f"{path}[{index}]")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateIRValidationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateIRValidationError(f"{path} must be a finite number")
    return result


def _quantize(value: float) -> float:
    result = float(Decimal(str(value)).quantize(QUANTUM, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def _point(value: Any, path: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise CandidateIRValidationError(f"{path} must be a three-number array")
    return [_quantize(_finite(item, f"{path}[{index}]")) for index, item in enumerate(value)]


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_TEXT_LENGTH:
        raise CandidateIRValidationError(f"{path} must be a non-empty string <= {MAX_TEXT_LENGTH} characters")
    return value.strip()


def _canonical_segment(start_raw: Any, end_raw: Any) -> tuple[list[float], list[float], list[float], list[float], float]:
    start = _point(start_raw, "line_start_mm"); end = _point(end_raw, "line_end_mm")
    delta = [end[index] - start[index] for index in range(3)]
    span = math.sqrt(sum(item * item for item in delta))
    if not math.isfinite(span) or span <= 0.0:
        raise CandidateIRValidationError("candidate line must have positive finite span")
    direction = [item / span for item in delta]
    dominant_magnitude = max(abs(item) for item in direction)
    dominant = next(index for index, item in enumerate(direction) if abs(item) == dominant_magnitude)
    if direction[dominant] < 0.0:
        start, end = end, start
        direction = [-item for item in direction]
    midpoint = [_quantize((start[index] + end[index]) / 2.0) for index in range(3)]
    span = _quantize(math.sqrt(sum((end[index] - start[index]) ** 2 for index in range(3))))
    direction = [_quantize(item) for item in direction]
    return start, end, midpoint, direction, span


def _extract_segment(row: Mapping[str, Any]) -> tuple[Any, Any, str]:
    shapes: list[tuple[Any, Any, str]] = []
    if row.get("line_start_mm") is not None or row.get("line_end_mm") is not None:
        if row.get("line_start_mm") is None or row.get("line_end_mm") is None:
            raise CandidateIRValidationError("line_start_mm and line_end_mm must be present together")
        shapes.append((row["line_start_mm"], row["line_end_mm"], "LINE_MM"))
    if row.get("line_start") is not None or row.get("line_end") is not None:
        if row.get("line_start") is None or row.get("line_end") is None:
            raise CandidateIRValidationError("line_start and line_end must be present together")
        shapes.append((row["line_start"], row["line_end"], "LINE"))
    if row.get("render_line_start") is not None or row.get("render_line_end") is not None:
        if row.get("render_line_start") is None or row.get("render_line_end") is None:
            raise CandidateIRValidationError("render_line_start and render_line_end must be present together")
        shapes.append((row["render_line_start"], row["render_line_end"], "RENDER_LINE"))
    if row.get("span_endpoints") is not None:
        endpoints = row["span_endpoints"]
        if not isinstance(endpoints, Sequence) or len(endpoints) != 2:
            raise CandidateIRValidationError("span_endpoints must contain exactly two endpoints")
        shapes.append((endpoints[0], endpoints[1], "SPAN_ENDPOINTS"))
    axis_keys = ("anchor", "axis_direction", "span_mm")
    if any(row.get(key) is not None for key in axis_keys):
        if not all(row.get(key) is not None for key in axis_keys):
            raise CandidateIRValidationError("anchor, axis_direction, and span_mm must be present together")
        anchor = _point(row["anchor"], "anchor")
        axis = _point(row["axis_direction"], "axis_direction")
        norm = math.sqrt(sum(item * item for item in axis))
        span = _finite(row["span_mm"], "span_mm")
        if norm <= 0.0 or span <= 0.0:
            raise CandidateIRValidationError("axis and span must be positive")
        unit = [item / norm for item in axis]
        shapes.append((
            [anchor[index] - 0.5 * span * unit[index] for index in range(3)],
            [anchor[index] + 0.5 * span * unit[index] for index in range(3)],
            "ANCHOR_AXIS_SPAN",
        ))
    if not shapes:
        raise CandidateIRValidationError("candidate has no complete line representation")
    canonical = [_canonical_segment(start, end)[:2] for start, end, _kind in shapes]
    if any(item != canonical[0] for item in canonical[1:]):
        raise CandidateIRValidationError("candidate line representations disagree after canonicalization")
    return shapes[0]


def _features(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > MAX_FEATURES:
        raise CandidateIRValidationError(f"features must be an object with at most {MAX_FEATURES} scalar entries")
    result: dict[str, float] = {}
    for raw_key in sorted(value):
        key = _text(raw_key, "features key")
        compact = "".join(char for char in key.casefold() if char.isalnum())
        if key.casefold() in PROHIBITED_KEYS or compact in PROHIBITED_COMPACT_KEYS:
            raise CandidateIRValidationError(f"features.{key} is prohibited")
        if key in result:
            raise CandidateIRValidationError(f"features.{key} duplicates a normalized feature key")
        result[key] = _quantize(_finite(value[raw_key], f"features.{key}"))
    return result


def canonicalize_candidate(row: Any, *, source_route: str | None = None,
                           source_stage: str | None = None) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise CandidateIRValidationError("candidate must be an object")
    _reject_prohibited(row)
    start_raw, end_raw, _adapter = _extract_segment(row)
    start, end, midpoint, direction, span = _canonical_segment(start_raw, end_raw)
    route = _text(source_route or row.get("source_route") or row.get("route"), "source_route")
    stage = _text(source_stage or row.get("source_stage") or row.get("stage"), "source_stage")
    source_values = row.get("source_ids")
    if source_values is None:
        source_values = [value for value in (row.get("source_hypothesis_id"), row.get("bend_id"), row.get("candidate_id")) if value]
    if not isinstance(source_values, Sequence) or isinstance(source_values, (str, bytes, bytearray)) or not source_values or len(source_values) > MAX_SOURCE_IDS:
        raise CandidateIRValidationError(f"source_ids must be a non-empty array of at most {MAX_SOURCE_IDS} strings")
    source_ids = sorted({_text(value, "source_ids[]") for value in source_values})
    support = str(row.get("support_status") or "UNKNOWN")
    if support not in SUPPORT_STATUSES:
        raise CandidateIRValidationError("invalid support_status")
    features = _features(row.get("features"))
    identity_payload = {
        "line_start_mm": start, "line_end_mm": end, "source_route": route,
        "source_stage": stage, "source_ids": source_ids,
    }
    candidate_id = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    return {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "units": UNITS,
        "coordinate_frame": COORDINATE_FRAME,
        "line_start_mm": start,
        "line_end_mm": end,
        "midpoint_mm": midpoint,
        "axis_direction": direction,
        "span_mm": span,
        "render_admissible": span >= 8.0,
        "source_route": route,
        "source_stage": stage,
        "source_ids": source_ids,
        "support_status": support,
        "features": features,
    }


def canonicalize_candidate_set(rows: Any, *, source_route: str | None = None,
                               source_stage: str | None = None) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise CandidateIRValidationError("candidate set must be an array")
    candidates = [canonicalize_candidate(row, source_route=source_route, source_stage=source_stage) for row in rows]
    candidates.sort(key=lambda row: row["candidate_id"])
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise CandidateIRValidationError("candidate set contains duplicate canonical identities")
    return {
        "schema": "F1_CANDIDATE_SET_V1",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def validate_stage_graph(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != STAGE_GRAPH_SCHEMA:
        raise CandidateIRValidationError(f"schema must be {STAGE_GRAPH_SCHEMA}")
    stages_raw = payload.get("stages")
    if not isinstance(stages_raw, Sequence) or isinstance(stages_raw, (str, bytes, bytearray)) or not stages_raw:
        raise CandidateIRValidationError("stages must be a non-empty array")
    stages: dict[str, dict[str, Any]] = {}
    for raw in stages_raw:
        if not isinstance(raw, Mapping):
            raise CandidateIRValidationError("each stage must be an object")
        stage_id = _text(raw.get("stage_id"), "stage_id")
        if stage_id in stages:
            raise CandidateIRValidationError("stage_id values must be unique")
        implementation_sha256 = str(raw.get("implementation_sha256") or "")
        if len(implementation_sha256) != 64 or any(item not in "0123456789abcdef" for item in implementation_sha256):
            raise CandidateIRValidationError("implementation_sha256 must be lowercase hex64")
        parents = raw.get("parent_stage_ids") or []
        if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes, bytearray)):
            raise CandidateIRValidationError("parent_stage_ids must be an array")
        stages[stage_id] = {
            "stage_id": stage_id,
            "implementation_sha256": implementation_sha256,
            "parent_stage_ids": sorted({_text(item, "parent_stage_ids[]") for item in parents}),
            "input_schema": _text(raw.get("input_schema"), "input_schema"),
            "output_schema": _text(raw.get("output_schema"), "output_schema"),
        }
    for stage in stages.values():
        missing = [item for item in stage["parent_stage_ids"] if item not in stages]
        if missing or stage["stage_id"] in stage["parent_stage_ids"]:
            raise CandidateIRValidationError("stage graph has a missing or self parent")
    visiting: set[str] = set(); visited: set[str] = set()
    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise CandidateIRValidationError("stage graph contains a cycle")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for parent in stages[stage_id]["parent_stage_ids"]:
            visit(parent)
        visiting.remove(stage_id); visited.add(stage_id)
    for stage_id in sorted(stages):
        visit(stage_id)
    return {"schema": STAGE_GRAPH_SCHEMA, "stages": [stages[key] for key in sorted(stages)]}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
