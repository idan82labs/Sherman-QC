"""Deterministic CIBEC_V1 validation and serialization.

This module is evaluation infrastructure only.  It accepts no predictions,
expected counts, filenames, family identity, or source-point arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Mapping


QUANTUM = Decimal("0.000001")
MINIMUM_SPAN_MM = 8.0
HEX64 = frozenset("0123456789abcdef")
PRIMARY_SCHEMA = "SCAN_BEND_EVENTS_V1"
CAD_SCHEMA = "CAD_EVENT_CORRESPONDENCE_V1"
CASE_STATUSES = frozenset({"RESOLVED", "ONTOLOGY_UNRESOLVED", "NONLINE_EVENT_CONFLICT"})
VISIBILITIES = frozenset({"FULL", "PARTIAL"})
OVERLAY_STATUSES = frozenset({"TRUSTED", "ABSTAIN"})
MAPPING_STATUSES = frozenset({"RESOLVED", "CAD_MAPPING_UNRESOLVED"})
RELATIONS = frozenset({"ONE_TO_ONE", "MANY_TO_ONE", "SCAN_ONLY_EVENT"})
CHARTER_FIELDS = (
    "COUNTING_UNIT", "CANONICAL_LINE", "INNER_OUTER_RULE", "COMPOUND_RADIUS_RULE",
    "RETURN_LIP_RULE", "OPEN_HEM_RULE", "CLOSED_HEM_RULE", "BEAD_RULE",
    "ROLLED_EDGE_RULE", "INTERRUPTION_RULE", "PARTIAL_VISIBILITY_RULE",
    "NONLINE_EVENT_RULE", "ENDPOINT_RULE",
)
PROHIBITED_KEYS = frozenset({
    "points", "source_points", "raw_points", "thumbnail", "thumbnails", "prediction",
    "predictions", "score", "scores", "expected_count", "expected_counts", "filename",
    "family", "family_id", "part_id", "part_name", "step_face_ids",
})


class CibecValidationError(ValueError):
    """Raised when a CIBEC_V1 artifact violates the frozen contract."""


def _reject_prohibited(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in PROHIBITED_KEYS:
                raise CibecValidationError(f"{path}.{key} is prohibited")
            _reject_prohibited(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_prohibited(child, f"{path}[{index}]")


def _hex64(value: Any, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise CibecValidationError(f"{path} must be 64 lowercase hexadecimal characters")
    return value


def _case_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 32 or any(c not in HEX64 for c in value):
        raise CibecValidationError("case_id must be a random 128-bit lowercase hexadecimal identifier")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CibecValidationError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CibecValidationError(f"{path} must be a finite number")
    return number


def _quantize(value: float) -> float:
    quantized = Decimal(str(value)).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)
    result = float(quantized)
    return 0.0 if result == 0.0 else result


def _point(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise CibecValidationError(f"{path} must be a three-number array")
    return [_quantize(_number(item, f"{path}[{index}]")) for index, item in enumerate(value)]


def _canonical_line(start_raw: Any, end_raw: Any) -> tuple[list[float], list[float], float, list[float]]:
    start = _point(start_raw, "line_start_mm")
    end = _point(end_raw, "line_end_mm")
    delta = [end[i] - start[i] for i in range(3)]
    length = math.sqrt(sum(item * item for item in delta))
    # Accept the exact decimal 8.000000 mm boundary despite binary64
    # cancellation, but never admit a one-quantum (1e-6 mm) undercut.
    boundary_tolerance = max(1e-12, MINIMUM_SPAN_MM * 1e-13)
    if not math.isfinite(length) or length < MINIMUM_SPAN_MM - boundary_tolerance:
        raise CibecValidationError(f"event line length must be at least {MINIMUM_SPAN_MM:.1f} mm")
    direction = [item / length for item in delta]
    maximum = max(abs(item) for item in direction)
    dominant = next(index for index, item in enumerate(direction) if abs(item) == maximum)
    if direction[dominant] < 0.0:
        start, end = end, start
        direction = [-item for item in direction]
    length = _quantize(math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3))))
    direction = [_quantize(item) for item in direction]
    return start, end, length, direction


def validate_charter(payload: Any, *, require_signed: bool = True) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != "CIBEC_V1_CHARTER":
        raise CibecValidationError("charter schema must be CIBEC_V1_CHARTER")
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict) or tuple(decisions) != CHARTER_FIELDS:
        raise CibecValidationError("charter decisions must contain the frozen ordered field set")
    for field in CHARTER_FIELDS:
        value = decisions[field]
        if not isinstance(value, str) or not value.strip() or value == "UNDECIDED":
            raise CibecValidationError(f"charter decision {field} is not frozen")
    signatures = payload.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 2:
        raise CibecValidationError("charter requires exactly two authority signatures")
    roles = [item.get("role") if isinstance(item, dict) else None for item in signatures]
    if roles != ["MANUFACTURING_AUTHORITY", "COUNTERSIGNING_AUTHORITY"]:
        raise CibecValidationError("charter signatures have invalid ordered roles")
    for index, signature in enumerate(signatures):
        if not isinstance(signature.get("name"), str) or not signature["name"].strip():
            raise CibecValidationError(f"signature {index} name is missing")
        _hex64(signature.get("signed_payload_sha256"), f"signatures[{index}].signed_payload_sha256")
    if require_signed and payload.get("status") != "SIGNED_AND_FROZEN":
        raise CibecValidationError("charter is not SIGNED_AND_FROZEN")
    return json.loads(json.dumps(payload, allow_nan=False))


def canonicalize_scan_truth(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CibecValidationError("scan truth must be a JSON object")
    _reject_prohibited(payload)
    if payload.get("schema") != PRIMARY_SCHEMA:
        raise CibecValidationError(f"schema must be {PRIMARY_SCHEMA}")
    if payload.get("units") != "mm" or payload.get("coordinate_frame") != "RAW_SCAN":
        raise CibecValidationError("scan truth must use millimetres in RAW_SCAN")
    _hex64(payload.get("ontology_sha256"), "ontology_sha256")
    _hex64(payload.get("renderer_sha256"), "renderer_sha256")
    _hex64(payload.get("scan_sha256"), "scan_sha256")
    _case_id(payload.get("case_id"))
    status = payload.get("case_status")
    if status not in CASE_STATUSES:
        raise CibecValidationError("invalid case_status")
    events_raw = payload.get("events")
    if not isinstance(events_raw, list):
        raise CibecValidationError("events must be an array")
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(events_raw):
        if not isinstance(raw, dict):
            raise CibecValidationError(f"events[{index}] must be an object")
        start, end, length, direction = _canonical_line(raw.get("line_start_mm"), raw.get("line_end_mm"))
        class_code = raw.get("class_code")
        rule_id = raw.get("adjudication_rule_id")
        if not isinstance(class_code, str) or not class_code.strip():
            raise CibecValidationError(f"events[{index}].class_code is required")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise CibecValidationError(f"events[{index}].adjudication_rule_id is required")
        if raw.get("visibility") not in VISIBILITIES or raw.get("render_admissible") is not True:
            raise CibecValidationError(f"events[{index}] visibility/render admissibility is invalid")
        views = raw.get("verified_view_ids")
        if not isinstance(views, list) or not views or any(item not in {f"V{i}" for i in range(1, 7)} for item in views):
            raise CibecValidationError(f"events[{index}].verified_view_ids is invalid")
        midpoint = [_quantize((start[i] + end[i]) / 2.0) for i in range(3)]
        events.append({
            "class_code": class_code.strip(), "line_start_mm": start, "line_end_mm": end,
            "length_mm": length, "visibility": raw["visibility"], "render_admissible": True,
            "verified_view_ids": sorted(set(views), key=lambda item: int(item[1:])),
            "adjudication_rule_id": rule_id.strip(), "_sort": (*midpoint, *direction, length, class_code.strip()),
        })
    events.sort(key=lambda item: item["_sort"])
    for ordinal, event in enumerate(events, 1):
        event.pop("_sort")
        event["event_id"] = f"E{ordinal:04d}"
        event_order = ("event_id", "class_code", "line_start_mm", "line_end_mm", "length_mm", "visibility", "render_admissible", "verified_view_ids", "adjudication_rule_id")
        ordered = {key: event[key] for key in event_order}
        event.clear(); event.update(ordered)
    order = ("schema", "ontology_sha256", "renderer_sha256", "case_id", "scan_sha256", "units", "coordinate_frame", "events", "case_status")
    return {key: (events if key == "events" else payload[key]) for key in order}


def validate_cad_correspondence(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CibecValidationError("CAD correspondence must be a JSON object")
    _reject_prohibited(payload)
    if payload.get("schema") != CAD_SCHEMA:
        raise CibecValidationError(f"schema must be {CAD_SCHEMA}")
    for key in ("primary_truth_sha256", "step_sha256", "overlay_transform_sha256"):
        _hex64(payload.get(key), key)
    status = payload.get("overlay_status")
    if status not in OVERLAY_STATUSES:
        raise CibecValidationError("invalid overlay_status")
    events = payload.get("events")
    cad_only = payload.get("cad_only_unobserved")
    if not isinstance(events, list) or not isinstance(cad_only, list):
        raise CibecValidationError("events and cad_only_unobserved must be arrays")
    if status == "ABSTAIN":
        if events or cad_only or payload.get("reason") != "OVERLAY_UNAVAILABLE_OR_UNTRUSTED":
            raise CibecValidationError("ABSTAIN correspondence must be empty with the frozen reason")
    owned: set[int] = set()
    normalized_events = []
    for index, raw in enumerate(events):
        faces = raw.get("step_face_indices")
        if not isinstance(faces, list) or not faces or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in faces):
            raise CibecValidationError(f"events[{index}].step_face_indices is invalid")
        if len(set(faces)) != len(faces) or owned.intersection(faces):
            raise CibecValidationError("duplicate STEP-face ownership")
        owned.update(faces)
        if raw.get("relation") not in RELATIONS or raw.get("mapping_status") not in MAPPING_STATUSES:
            raise CibecValidationError(f"events[{index}] relation/mapping_status is invalid")
        primitives = raw.get("v9_primitive_ids")
        if not isinstance(primitives, list) or any(not isinstance(v, str) or not v for v in primitives):
            raise CibecValidationError(f"events[{index}].v9_primitive_ids is invalid")
        normalized_events.append({
            "event_id": str(raw.get("event_id")), "step_face_indices": sorted(faces),
            "v9_primitive_ids": sorted(set(primitives)), "relation": raw["relation"],
            "mapping_status": raw["mapping_status"],
        })
    normalized_events.sort(key=lambda item: item["event_id"])
    normalized_cad_only = []
    for index, raw in enumerate(cad_only):
        faces = raw.get("step_face_indices")
        if not isinstance(faces, list) or not faces or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in faces):
            raise CibecValidationError(f"cad_only_unobserved[{index}] faces are invalid")
        if len(set(faces)) != len(faces) or owned.intersection(faces):
            raise CibecValidationError("duplicate STEP-face ownership")
        owned.update(faces)
        reason = raw.get("reason_code")
        if not isinstance(reason, str) or not reason.strip():
            raise CibecValidationError(f"cad_only_unobserved[{index}].reason_code is required")
        normalized_cad_only.append({"step_face_indices": sorted(faces), "reason_code": reason.strip()})
    normalized_cad_only.sort(key=lambda item: (min(item["step_face_indices"]), max(item["step_face_indices"]), item["step_face_indices"]))
    result = {key: payload[key] for key in ("schema", "primary_truth_sha256", "step_sha256", "overlay_transform_sha256", "overlay_status")}
    if status == "ABSTAIN": result["reason"] = payload["reason"]
    result["events"] = normalized_events
    result["cad_only_unobserved"] = normalized_cad_only
    return result


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
