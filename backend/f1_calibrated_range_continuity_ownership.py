"""V29 calibrated-range continuity ownership (CRCO_V1).

This research-only module consumes bounded geometric-core candidate records and
calibrated raw range frames.  It never creates or refits geometry: it may only
rescue an ownership-ambiguous core candidate or veto a boundary-risk legacy
candidate, before the one ordinary NMS pass.

The implementation is deliberately single-threaded and fail-closed.  Invalid
range samples are UNKNOWN, structural caps raise, and all identity/order keys
are content-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Iterable, Sequence

import numpy as np


PROFILE = "V29_CALIBRATED_RANGE_CONTINUITY_OWNERSHIP_V1"
ABBREVIATION = "CRCO_V1"
SCHEMA_VERSION = 1

STATION_COUNT = 24
FLANK_MULTIPLIERS = (2, 3, 4, 5, 6, 7, 8)
MIN_MATCHES_PER_FLANK = 5
DECISION_STATION_COUNT = 16
DECISION_RUN_LENGTH = 8
MIN_VIEW_SEPARATION_DEG = 12.0
MATCH_DISTANCE_S = 1.5
DEPTH_SEPARATION_S = 4.0
PATH_GAP_S = 2.0
BOUNDARY_DISTANCE_S = 8.0
BOUNDARY_STATION_COUNT = 8
COORDINATE_QUANTIZATION_DIVISOR = 1024

KEYFRAME_COUNT = 4
MAX_CANDIDATE_RECORDS = 4096
MAX_QUERIED_PIXELS = 262_144
MAX_INCREMENTAL_BYTES = 512 * 1024 * 1024
MAX_CACHE_BYTES = 16 * 1024 * 1024

NMS_ANGLE_DEG = 7.0
NMS_DISTANCE_MM = 6.0
NMS_OVERLAP = 0.60

BOUNDARY_REJECTION_REASON = "BOUNDARY_OWNERSHIP_AMBIGUOUS"
ACTION_NONE = "UNCHANGED"
ACTION_RESCUE = "RESCUE"
ACTION_VETO = "VETO"
STATE_CONTINUOUS = "C"
STATE_JUMP = "J"
STATE_OCCLUDED = "O"
STATE_UNKNOWN = "U"

CANDIDATE_SCHEMA_FIELDS = (
    "line_start",
    "line_end",
    "support_interval",
    "flank_tangents",
    "geometric_core_pass_bits",
    "boundary_risk",
    "rejection_reason",
    "selected_legacy",
    "legacy_rank_bytes",
    "span",
    "quality_summaries",
)

RANGE_FRAME_SCHEMA_FIELDS = (
    "schema_version",
    "projection_model",
    "width",
    "height",
    "depth_quantum_mm",
    "depth_q",
    "valid",
    "calibration",
    "sensor_to_scan",
)

PROHIBITED_FRAME_FIELDS = frozenset(
    {
        "rgb",
        "reflectance",
        "intensity",
        "timestamp",
        "filename",
        "scanner_serial_number",
        "operator_id",
        "part_id",
        "acquisition_comments",
    }
)


class CRCOContractError(RuntimeError):
    """A fail-closed V29 structural or schema violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _reject_negative_zero(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CRCOContractError("NONFINITE_VALUE", label)
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise CRCOContractError("SIGNED_ZERO_VALUE", label)
    return result


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise CRCOContractError("INVALID_ARRAY", label)
    if np.any((result == 0.0) & np.signbit(result)):
        raise CRCOContractError("SIGNED_ZERO_VALUE", label)
    return result


def _unit(value: Any, label: str) -> np.ndarray:
    result = _array(value, (3,), label)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise CRCOContractError("ZERO_VECTOR", label)
    return result / norm


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_hex(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _round_half_away(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _quantized_tuple(point: np.ndarray, quantum: float) -> tuple[int, int, int]:
    return tuple(_round_half_away(float(component) / quantum) for component in point)  # type: ignore[return-value]


def _pack_int_tuple(values: Iterable[int]) -> bytes:
    values = tuple(int(value) for value in values)
    return struct.pack("<" + "q" * len(values), *values)


def _proper_rigid_pose(value: Any, label: str) -> np.ndarray:
    pose = _array(value, (4, 4), label)
    if not np.array_equal(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise CRCOContractError("INVALID_RIGID_POSE", label)
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10, rtol=0.0):
        raise CRCOContractError("INVALID_RIGID_POSE", label)
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-10):
        raise CRCOContractError("IMPROPER_RIGID_POSE", label)
    return pose


@dataclass(frozen=True)
class RangeFrame:
    width: int
    height: int
    depth_quantum_mm: float
    depth_q: np.ndarray
    valid: np.ndarray
    pixel_pitch_mm: float
    cx: float
    cy: float
    sensor_to_scan: np.ndarray
    content_hash: str
    calibration_hash: str
    frame_key: str

    @property
    def rotation(self) -> np.ndarray:
        return self.sensor_to_scan[:3, :3]

    @property
    def origin(self) -> np.ndarray:
        return self.sensor_to_scan[:3, 3]

    @property
    def optical_axis(self) -> np.ndarray:
        return self.rotation[:, 2]

    def scan_to_camera(self, point: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (point - self.origin)

    def project(self, point: np.ndarray) -> tuple[int, int, float] | None:
        camera = self.scan_to_camera(point)
        u = _round_half_away(float(camera[0]) / self.pixel_pitch_mm + self.cx)
        v = _round_half_away(float(camera[1]) / self.pixel_pitch_mm + self.cy)
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return u, v, float(camera[2])

    def backproject(self, u: int, v: int) -> np.ndarray:
        depth = float(self.depth_q[v, u]) * self.depth_quantum_mm
        camera = np.asarray(
            [
                (float(u) - self.cx) * self.pixel_pitch_mm,
                (float(v) - self.cy) * self.pixel_pitch_mm,
                depth,
            ],
            dtype=np.float64,
        )
        return self.rotation @ camera + self.origin


def _frame_content_bytes(raw: dict[str, Any]) -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    if set(raw).intersection(PROHIBITED_FRAME_FIELDS):
        raise CRCOContractError("PROHIBITED_RANGE_FIELD")
    missing = [name for name in RANGE_FRAME_SCHEMA_FIELDS if name not in raw]
    if missing:
        raise CRCOContractError("RANGE_SCHEMA_MISSING", ",".join(missing))
    if int(raw["schema_version"]) != SCHEMA_VERSION:
        raise CRCOContractError("RANGE_SCHEMA_VERSION")
    if str(raw["projection_model"]) != "orthographic_range_lattice":
        raise CRCOContractError("UNSUPPORTED_RANGE_PROJECTION")
    width = int(raw["width"])
    height = int(raw["height"])
    if width <= 0 or height <= 0:
        raise CRCOContractError("RANGE_FRAME_SHAPE")
    depth = np.asarray(raw["depth_q"], dtype=np.int64)
    valid = np.asarray(raw["valid"], dtype=np.bool_)
    if depth.shape != (height, width) or valid.shape != (height, width):
        raise CRCOContractError("RANGE_FRAME_SHAPE")
    if np.any(depth[valid] <= 0):
        raise CRCOContractError("RANGE_VALID_DEPTH_NONPOSITIVE")
    int32 = np.iinfo(np.int32)
    if np.any(depth < int32.min) or np.any(depth > int32.max):
        raise CRCOContractError("RANGE_DEPTH_INTEGER_OVERFLOW")
    depth_i32 = depth.astype("<i4", copy=False)
    validity = np.packbits(valid.reshape(-1), bitorder="little").tobytes()
    depth_bytes = depth_i32.tobytes(order="C")
    header = _canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "projection_model": "orthographic_range_lattice",
            "width": width,
            "height": height,
            "depth_quantum_mm": _reject_negative_zero(raw["depth_quantum_mm"], "depth_quantum_mm"),
        }
    )
    return header + b"\0" + validity + b"\0" + depth_bytes, validity, depth_i32, valid


def parse_range_frames(raw_frames: Sequence[dict[str, Any]]) -> list[RangeFrame]:
    if len(raw_frames) < KEYFRAME_COUNT:
        raise CRCOContractError("FEWER_THAN_FOUR_VALID_RANGE_FRAMES")
    provisional: list[dict[str, Any]] = []
    incremental_bytes = 0
    for raw in raw_frames:
        frame_bytes, _validity, depth_q, valid = _frame_content_bytes(raw)
        incremental_bytes += int(depth_q.nbytes + valid.nbytes)
        if incremental_bytes > MAX_INCREMENTAL_BYTES:
            raise CRCOContractError("INCREMENTAL_MEMORY_CAP_EXCEEDED")
        calibration = dict(raw["calibration"])
        if set(calibration) != {"pixel_pitch_mm", "cx", "cy"}:
            raise CRCOContractError("INVALID_RANGE_CALIBRATION_SCHEMA")
        pixel_pitch = _reject_negative_zero(calibration["pixel_pitch_mm"], "pixel_pitch_mm")
        cx = _reject_negative_zero(calibration["cx"], "cx")
        cy = _reject_negative_zero(calibration["cy"], "cy")
        if pixel_pitch <= 0.0:
            raise CRCOContractError("INVALID_RANGE_CALIBRATION")
        pose = _proper_rigid_pose(raw["sensor_to_scan"], "sensor_to_scan")
        calibration_bytes = _canonical_json_bytes(
            {
                "pixel_pitch_mm": pixel_pitch,
                "cx": cx,
                "cy": cy,
            }
        )
        provisional.append(
            {
                "raw": raw,
                "depth_q": depth_q,
                "valid": valid,
                "pose": pose,
                "pixel_pitch": pixel_pitch,
                "cx": cx,
                "cy": cy,
                "content_hash": _sha256_hex(frame_bytes),
                "calibration_hash": _sha256_hex(calibration_bytes),
            }
        )

    # A global-rigid-invariant signature uses only inter-camera distances and
    # optical-axis dot products, keyed by other frames' content/calibration.
    for current in provisional:
        invariants = []
        for other in provisional:
            distance_q = _round_half_away(
                float(np.linalg.norm(current["pose"][:3, 3] - other["pose"][:3, 3]))
                * 1_000_000.0
            )
            dot_q = _round_half_away(
                float(np.dot(current["pose"][:3, 2], other["pose"][:3, 2]))
                * 1_000_000_000.0
            )
            invariants.append(
                (
                    str(other["content_hash"]),
                    str(other["calibration_hash"]),
                    distance_q,
                    dot_q,
                )
            )
        signature = _canonical_json_bytes(sorted(invariants))
        current["frame_key"] = _sha256_hex(
            bytes.fromhex(current["content_hash"]),
            bytes.fromhex(current["calibration_hash"]),
            signature,
        )

    unique: dict[str, dict[str, Any]] = {}
    for item in sorted(provisional, key=lambda row: str(row["frame_key"])):
        unique.setdefault(str(item["frame_key"]), item)
    if len(unique) < KEYFRAME_COUNT:
        raise CRCOContractError("FEWER_THAN_FOUR_VALID_RANGE_FRAMES")

    frames = []
    for item in unique.values():
        raw = item["raw"]
        frames.append(
            RangeFrame(
                width=int(raw["width"]),
                height=int(raw["height"]),
                depth_quantum_mm=_reject_negative_zero(raw["depth_quantum_mm"], "depth_quantum_mm"),
                depth_q=item["depth_q"],
                valid=item["valid"],
                pixel_pitch_mm=float(item["pixel_pitch"]),
                cx=float(item["cx"]),
                cy=float(item["cy"]),
                sensor_to_scan=item["pose"],
                content_hash=str(item["content_hash"]),
                calibration_hash=str(item["calibration_hash"]),
                frame_key=str(item["frame_key"]),
            )
        )
    return frames


def _axis_separation(first: RangeFrame, second: RangeFrame) -> float:
    dot = float(np.clip(np.dot(first.optical_axis, second.optical_axis), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def select_keyframes(frames: Sequence[RangeFrame]) -> list[RangeFrame]:
    ordered = sorted(frames, key=lambda frame: frame.frame_key)
    pairs: list[tuple[float, tuple[str, str], int, int]] = []
    for first in range(len(ordered)):
        for second in range(first):
            keys = tuple(sorted((ordered[first].frame_key, ordered[second].frame_key)))
            pairs.append((-_axis_separation(ordered[first], ordered[second]), keys, second, first))
    if not pairs:
        raise CRCOContractError("FEWER_THAN_FOUR_VALID_RANGE_FRAMES")
    _negative, _keys, first_index, second_index = sorted(pairs)[0]
    selected = [ordered[first_index], ordered[second_index]]
    selected_keys = {frame.frame_key for frame in selected}
    remaining = [frame for frame in ordered if frame.frame_key not in selected_keys]
    while len(selected) < KEYFRAME_COUNT:
        if not remaining:
            raise CRCOContractError("FEWER_THAN_FOUR_VALID_RANGE_FRAMES")
        ranked = []
        for frame in remaining:
            minimum = min(_axis_separation(frame, prior) for prior in selected)
            ranked.append((-minimum, frame.frame_key, frame))
        chosen = sorted(ranked, key=lambda item: (item[0], item[1]))[0][2]
        selected.append(chosen)
        remaining = [frame for frame in remaining if frame.frame_key != chosen.frame_key]
    return selected


@dataclass(frozen=True)
class CandidateRecord:
    line_start: np.ndarray
    line_end: np.ndarray
    support_interval: tuple[float, float]
    flank_tangents: tuple[np.ndarray, np.ndarray]
    geometric_core_pass_bits: tuple[tuple[str, bool], ...]
    boundary_risk: bool
    rejection_reason: str
    selected_legacy: bool
    legacy_rank_bytes: str
    span: float
    quality_summaries: tuple[tuple[str, float], ...]
    candidate_hash: str

    def as_row(self) -> dict[str, Any]:
        return {
            "line_start": self.line_start.tolist(),
            "line_end": self.line_end.tolist(),
            "span_endpoints": [self.line_start.tolist(), self.line_end.tolist()],
            "support_interval": list(self.support_interval),
            "flank_tangents": [value.tolist() for value in self.flank_tangents],
            "geometric_core_pass_bits": dict(self.geometric_core_pass_bits),
            "boundary_risk": self.boundary_risk,
            "rejection_reason": self.rejection_reason,
            "selected_legacy": self.selected_legacy,
            "legacy_rank_bytes": self.legacy_rank_bytes,
            "span": self.span,
            "span_mm": self.span,
            "quality_summaries": dict(self.quality_summaries),
            "candidate_hash": self.candidate_hash,
            "candidate_id": self.candidate_hash,
        }


def _canonicalize_line_and_flanks(
    start: np.ndarray, end: np.ndarray, flanks: Sequence[np.ndarray], quantum: float
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], bytes]:
    start_q = _quantized_tuple(start, quantum)
    end_q = _quantized_tuple(end, quantum)
    if start_q == end_q:
        raise CRCOContractError("ZERO_SPAN_CANDIDATE")
    if end_q < start_q:
        start, end = end, start
        start_q, end_q = end_q, start_q
    canonical_flanks = []
    for value in flanks:
        direction = _unit(value, "flank_tangent")
        q = _quantized_tuple(direction, 1.0 / COORDINATE_QUANTIZATION_DIVISOR)
        negated = tuple(-item for item in q)
        if negated < q:
            direction = -direction
            q = negated
        canonical_flanks.append((q, direction))
    canonical_flanks.sort(key=lambda item: item[0])
    packed = _pack_int_tuple((*start_q, *end_q, *canonical_flanks[0][0], *canonical_flanks[1][0]))
    return start, end, (canonical_flanks[0][1], canonical_flanks[1][1]), packed


def parse_candidates(raw_candidates: Sequence[dict[str, Any]], s: float) -> list[CandidateRecord]:
    if len(raw_candidates) > MAX_CANDIDATE_RECORDS:
        raise CRCOContractError("CANDIDATE_CAP_EXCEEDED")
    quantum = s / COORDINATE_QUANTIZATION_DIVISOR
    candidates: list[CandidateRecord] = []
    for raw in raw_candidates:
        missing = [name for name in CANDIDATE_SCHEMA_FIELDS if name not in raw]
        if missing:
            raise CRCOContractError("CANDIDATE_SCHEMA_MISSING", ",".join(missing))
        start = _array(raw["line_start"], (3,), "line_start")
        end = _array(raw["line_end"], (3,), "line_end")
        raw_flanks = raw["flank_tangents"]
        if not isinstance(raw_flanks, list) or len(raw_flanks) != 2:
            raise CRCOContractError("CANDIDATE_FLANK_COUNT")
        start, end, flanks, geometry_bytes = _canonicalize_line_and_flanks(
            start, end, [_array(value, (3,), "flank_tangent") for value in raw_flanks], quantum
        )
        support = tuple(
            _reject_negative_zero(value, "support_interval")
            for value in raw["support_interval"]
        )
        if len(support) != 2 or support[1] < support[0]:
            raise CRCOContractError("INVALID_SUPPORT_INTERVAL")
        pass_bits_raw = raw["geometric_core_pass_bits"]
        if not isinstance(pass_bits_raw, dict) or not pass_bits_raw:
            raise CRCOContractError("INVALID_GEOMETRIC_CORE_BITS")
        pass_bits = tuple(sorted((str(key), bool(value)) for key, value in pass_bits_raw.items()))
        quality_raw = raw["quality_summaries"]
        if not isinstance(quality_raw, dict):
            raise CRCOContractError("INVALID_QUALITY_SUMMARIES")
        quality = tuple(
            sorted(
                (str(key), _reject_negative_zero(value, f"quality:{key}"))
                for key, value in quality_raw.items()
            )
        )
        span = _reject_negative_zero(raw["span"], "span")
        if span <= 0.0:
            raise CRCOContractError("ZERO_SPAN_CANDIDATE")
        certificate_bytes = _canonical_json_bytes(
            {
                "geometric_core_pass_bits": pass_bits,
                "support_interval": support,
            }
        )
        candidate_hash = _sha256_hex(geometry_bytes, certificate_bytes)
        candidates.append(
            CandidateRecord(
                line_start=start,
                line_end=end,
                support_interval=(support[0], support[1]),
                flank_tangents=flanks,
                geometric_core_pass_bits=pass_bits,
                boundary_risk=bool(raw["boundary_risk"]),
                rejection_reason=str(raw["rejection_reason"]),
                selected_legacy=bool(raw["selected_legacy"]),
                legacy_rank_bytes=str(raw["legacy_rank_bytes"]),
                span=span,
                quality_summaries=quality,
                candidate_hash=candidate_hash,
            )
        )
    hashes = [candidate.candidate_hash for candidate in candidates]
    if len(hashes) != len(set(hashes)):
        raise CRCOContractError("DUPLICATE_GEOMETRIC_CORE_HASH")
    return sorted(candidates, key=lambda candidate: candidate.candidate_hash)


class PixelLedger:
    def __init__(self) -> None:
        self._keys: set[tuple[str, int, int]] = set()

    def record(self, frame: RangeFrame, u: int, v: int) -> None:
        self._keys.add((frame.frame_key, int(u), int(v)))
        if len(self._keys) > MAX_QUERIED_PIXELS:
            raise CRCOContractError("QUERY_PIXEL_CAP_EXCEEDED")

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"frame_key": frame_key, "u": u, "v": v}
            for frame_key, u, v in sorted(self._keys)
        ]


def _sample_state(
    frame: RangeFrame,
    point: np.ndarray,
    s: float,
    ledger: PixelLedger,
) -> tuple[str, dict[str, Any]]:
    projected = frame.project(point)
    if projected is None:
        return STATE_UNKNOWN, {"pixel": None, "valid": False, "state": STATE_UNKNOWN}
    u, v, expected_depth = projected
    ledger.record(frame, u, v)
    if not bool(frame.valid[v, u]):
        return STATE_UNKNOWN, {"pixel": [u, v], "valid": False, "state": STATE_UNKNOWN}
    observed = frame.backproject(u, v)
    observed_depth = float(frame.depth_q[v, u]) * frame.depth_quantum_mm
    quantum = s / COORDINATE_QUANTIZATION_DIVISOR
    point_q = np.asarray(_quantized_tuple(point, quantum), dtype=np.int64)
    observed_q = np.asarray(_quantized_tuple(observed, quantum), dtype=np.int64)
    distance_q = float(np.linalg.norm(observed_q - point_q))
    depth_delta_q = _round_half_away((observed_depth - expected_depth) / quantum)
    if distance_q <= MATCH_DISTANCE_S * COORDINATE_QUANTIZATION_DIVISOR:
        state = "MATCH"
    elif depth_delta_q >= DEPTH_SEPARATION_S * COORDINATE_QUANTIZATION_DIVISOR:
        state = "BEHIND"
    elif depth_delta_q <= -DEPTH_SEPARATION_S * COORDINATE_QUANTIZATION_DIVISOR:
        state = "IN_FRONT"
    else:
        state = STATE_UNKNOWN
    return state, {
        "pixel": [u, v],
        "valid": True,
        "state": state,
        "expected_depth_q": _round_half_away(expected_depth / quantum),
        "observed_depth_q": _round_half_away(observed_depth / quantum),
        "observed_point_q": observed_q.tolist(),
    }


def _four_connected_path(
    frame: RangeFrame,
    first: tuple[int, int],
    second: tuple[int, int],
    s: float,
    ledger: PixelLedger,
) -> tuple[bool, list[list[int]]]:
    min_u, max_u = sorted((first[0], second[0]))
    min_v, max_v = sorted((first[1], second[1]))
    if (max_u - min_u + 1) * (max_v - min_v + 1) > MAX_QUERIED_PIXELS:
        raise CRCOContractError("QUERY_PIXEL_CAP_EXCEEDED")
    start = (first[0], first[1])
    goal = (second[0], second[1])
    queue = [start]
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    if not bool(frame.valid[start[1], start[0]]) or not bool(frame.valid[goal[1], goal[0]]):
        ledger.record(frame, start[0], start[1])
        ledger.record(frame, goal[0], goal[1])
        return False, []
    while queue:
        current = queue.pop(0)
        ledger.record(frame, current[0], current[1])
        if current == goal:
            path = []
            cursor: tuple[int, int] | None = current
            while cursor is not None:
                path.append([cursor[0], cursor[1]])
                cursor = parents[cursor]
            return True, list(reversed(path))
        current_point = frame.backproject(current[0], current[1])
        neighbors = (
            (current[0] - 1, current[1]),
            (current[0], current[1] - 1),
            (current[0], current[1] + 1),
            (current[0] + 1, current[1]),
        )
        for neighbor in neighbors:
            u, v = neighbor
            if not (min_u <= u <= max_u and min_v <= v <= max_v):
                continue
            if neighbor in parents:
                continue
            ledger.record(frame, u, v)
            if not bool(frame.valid[v, u]):
                continue
            neighbor_point = frame.backproject(u, v)
            if float(np.linalg.norm(neighbor_point - current_point)) > PATH_GAP_S * s:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    return False, []


def _classify_frame_station(
    frame: RangeFrame,
    station: np.ndarray,
    flanks: tuple[np.ndarray, np.ndarray],
    s: float,
    ledger: PixelLedger,
) -> tuple[str, dict[str, Any]]:
    flank_details: list[list[dict[str, Any]]] = []
    flank_states: list[list[str]] = []
    inner_pixels: list[tuple[int, int] | None] = []
    for flank in flanks:
        details = []
        states = []
        for multiplier in FLANK_MULTIPLIERS:
            point = station + float(multiplier) * s * flank
            state, detail = _sample_state(frame, point, s, ledger)
            details.append(detail)
            states.append(state)
        flank_details.append(details)
        flank_states.append(states)
        pixel = details[0]["pixel"]
        inner_pixels.append(None if pixel is None else (int(pixel[0]), int(pixel[1])))
    center_state, center_detail = _sample_state(frame, station, s, ledger)

    path_exists = False
    path: list[list[int]] = []
    if inner_pixels[0] is not None and inner_pixels[1] is not None:
        path_exists, path = _four_connected_path(
            frame, inner_pixels[0], inner_pixels[1], s, ledger
        )
    match_counts = [states.count("MATCH") for states in flank_states]
    behind_counts = [states.count("BEHIND") for states in flank_states]
    in_front_counts = [states.count("IN_FRONT") for states in flank_states]
    continuous = (
        min(match_counts) >= MIN_MATCHES_PER_FLANK
        and center_state != "IN_FRONT"
        and path_exists
    )
    jump = (
        (
            match_counts[0] >= MIN_MATCHES_PER_FLANK
            and behind_counts[1] >= MIN_MATCHES_PER_FLANK
        )
        or (
            match_counts[1] >= MIN_MATCHES_PER_FLANK
            and behind_counts[0] >= MIN_MATCHES_PER_FLANK
        )
    ) and not path_exists and center_state != "IN_FRONT"
    occluded = max(in_front_counts) >= MIN_MATCHES_PER_FLANK
    if continuous:
        state = STATE_CONTINUOUS
    elif jump:
        state = STATE_JUMP
    elif occluded:
        state = STATE_OCCLUDED
    else:
        state = STATE_UNKNOWN
    return state, {
        "frame_key": frame.frame_key,
        "state": state,
        "flanks": flank_details,
        "center": center_detail,
        "path_exists": path_exists,
        "path": path,
        "match_counts": match_counts,
        "behind_counts": behind_counts,
        "in_front_counts": in_front_counts,
    }


def _has_separated_pair(
    target: str, frame_states: Sequence[tuple[RangeFrame, str]]
) -> bool:
    matching = [frame for frame, state in frame_states if state == target]
    return any(
        _axis_separation(first, second) >= MIN_VIEW_SEPARATION_DEG
        for index, first in enumerate(matching)
        for second in matching[:index]
    )


def _cross_view_state(frame_states: Sequence[tuple[RangeFrame, str]]) -> str:
    observed = {state for _frame, state in frame_states if state != STATE_UNKNOWN}
    if len(observed) > 1:
        return STATE_UNKNOWN
    if observed == {STATE_CONTINUOUS} and _has_separated_pair(STATE_CONTINUOUS, frame_states):
        return STATE_CONTINUOUS
    if observed == {STATE_JUMP} and _has_separated_pair(STATE_JUMP, frame_states):
        return STATE_JUMP
    if observed == {STATE_OCCLUDED} and _has_separated_pair(STATE_OCCLUDED, frame_states):
        return STATE_OCCLUDED
    return STATE_UNKNOWN


def _longest_run(states: Sequence[str], target: str) -> int:
    best = current = 0
    for state in states:
        if state == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _eligible_rescue(candidate: CandidateRecord) -> bool:
    return (
        not candidate.selected_legacy
        and all(value for _name, value in candidate.geometric_core_pass_bits)
        and candidate.rejection_reason == BOUNDARY_REJECTION_REASON
    )


def _eligible_veto(candidate: CandidateRecord) -> bool:
    return candidate.selected_legacy and candidate.boundary_risk


def adjudicate_crco_v1(
    candidates: Sequence[CandidateRecord],
    keyframes: Sequence[RangeFrame],
    s: float,
) -> tuple[list[dict[str, Any]], PixelLedger]:
    ledger = PixelLedger()
    decisions = []
    for candidate in sorted(candidates, key=lambda value: value.candidate_hash):
        stations = [
            candidate.line_start
            + ((float(index) + 0.5) / float(STATION_COUNT))
            * (candidate.line_end - candidate.line_start)
            for index in range(STATION_COUNT)
        ]
        station_rows = []
        cross_states = []
        for station_index, station in enumerate(stations):
            per_frame = []
            compact = []
            for frame in keyframes:
                state, detail = _classify_frame_station(
                    frame, station, candidate.flank_tangents, s, ledger
                )
                compact.append((frame, state))
                per_frame.append(detail)
            cross = _cross_view_state(compact)
            cross_states.append(cross)
            station_rows.append(
                {
                    "station_index": station_index,
                    "station_q": list(
                        _quantized_tuple(
                            station, s / COORDINATE_QUANTIZATION_DIVISOR
                        )
                    ),
                    "per_frame": per_frame,
                    "cross_view_state": cross,
                }
            )
        continuous_count = cross_states.count(STATE_CONTINUOUS)
        jump_count = cross_states.count(STATE_JUMP)
        unknown_count = cross_states.count(STATE_UNKNOWN)
        continuous_run = _longest_run(cross_states, STATE_CONTINUOUS)
        jump_run = _longest_run(cross_states, STATE_JUMP)
        action = ACTION_NONE
        if (
            _eligible_rescue(candidate)
            and continuous_count >= DECISION_STATION_COUNT
            and jump_count == 0
            and continuous_run >= DECISION_RUN_LENGTH
        ):
            action = ACTION_RESCUE
        elif (
            _eligible_veto(candidate)
            and jump_count >= DECISION_STATION_COUNT
            and continuous_count == 0
            and jump_run >= DECISION_RUN_LENGTH
        ):
            action = ACTION_VETO
        decisions.append(
            {
                "candidate": candidate.as_row(),
                "action": action,
                "continuous_station_count": continuous_count,
                "jump_station_count": jump_count,
                "occluded_station_count": cross_states.count(STATE_OCCLUDED),
                "unknown_station_count": unknown_count,
                "longest_continuous_run": continuous_run,
                "longest_jump_run": jump_run,
                "cross_view_states": cross_states,
                "stations": station_rows,
            }
        )
    return decisions, ledger


def deterministic_total_sort(decisions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy = [
        decision
        for decision in decisions
        if bool(decision["candidate"]["selected_legacy"])
        and decision["action"] != ACTION_VETO
    ]
    legacy.sort(
        key=lambda decision: (
            str(decision["candidate"]["legacy_rank_bytes"]),
            str(decision["candidate"]["candidate_hash"]),
        )
    )
    rescues = [decision for decision in decisions if decision["action"] == ACTION_RESCUE]
    rescues.sort(
        key=lambda decision: (
            -int(decision["continuous_station_count"]),
            -int(decision["longest_continuous_run"]),
            int(decision["unknown_station_count"]),
            -float(decision["candidate"]["span"]),
            str(decision["candidate"]["candidate_hash"]),
        )
    )
    return legacy + rescues


def _line_metrics(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float, float]:
    first_line = np.asarray([first["line_start"], first["line_end"]], dtype=np.float64)
    second_line = np.asarray([second["line_start"], second["line_end"]], dtype=np.float64)
    first_delta = first_line[1] - first_line[0]
    second_delta = second_line[1] - second_line[0]
    first_length = float(np.linalg.norm(first_delta))
    second_length = float(np.linalg.norm(second_delta))
    first_axis = first_delta / first_length
    second_axis = second_delta / second_length
    angle = math.degrees(
        math.acos(float(np.clip(abs(np.dot(first_axis, second_axis)), -1.0, 1.0)))
    )
    first_mid = first_line.mean(axis=0)
    second_mid = second_line.mean(axis=0)
    delta = second_mid - first_mid
    first_offset = float(np.linalg.norm(delta - first_axis * np.dot(delta, first_axis)))
    second_offset = float(np.linalg.norm(delta - second_axis * np.dot(delta, second_axis)))
    distance = max(first_offset, second_offset)
    center_distance = abs(float(np.dot(delta, first_axis)))
    overlap = max(0.0, (first_length + second_length) / 2.0 - center_distance)
    overlap_fraction = overlap / min(first_length, second_length)
    return angle, distance, overlap_fraction


def ordinary_nms(ordered: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for decision in ordered:
        candidate = decision["candidate"]
        duplicate = False
        for prior in accepted:
            angle, distance, overlap = _line_metrics(candidate, prior["candidate"])
            if (
                angle <= NMS_ANGLE_DEG
                and distance <= NMS_DISTANCE_MM
                and overlap >= NMS_OVERLAP
            ):
                duplicate = True
                break
        if not duplicate:
            accepted.append(decision)
    return accepted


def build_result(
    accepted: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    keyframes: Sequence[RangeFrame],
    ledger: PixelLedger,
    s: float,
) -> dict[str, Any]:
    final_lines = []
    for decision in accepted:
        candidate = decision["candidate"]
        final_lines.append(
            {
                "candidate_hash": candidate["candidate_hash"],
                "line_start": candidate["line_start"],
                "line_end": candidate["line_end"],
                "source_lane": "v29_rescue"
                if decision["action"] == ACTION_RESCUE
                else "legacy",
                "v29_action": decision["action"],
            }
        )
    final_lines.sort(key=lambda row: str(row["candidate_hash"]))
    cache = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "frame_hashes": [frame.content_hash for frame in keyframes],
        "calibration_hashes": [frame.calibration_hash for frame in keyframes],
        "selected_keyframe_hashes": [frame.frame_key for frame in keyframes],
        "candidate_state_counts": [
            {
                "candidate_hash": decision["candidate"]["candidate_hash"],
                "action": decision["action"],
                "continuous_station_count": decision["continuous_station_count"],
                "jump_station_count": decision["jump_station_count"],
                "occluded_station_count": decision["occluded_station_count"],
                "unknown_station_count": decision["unknown_station_count"],
            }
            for decision in sorted(
                decisions, key=lambda item: str(item["candidate"]["candidate_hash"])
            )
        ],
        "final_lines": final_lines,
    }
    cache_bytes = _canonical_json_bytes(cache)
    if len(cache_bytes) > MAX_CACHE_BYTES:
        raise CRCOContractError("CACHE_CAP_EXCEEDED")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sherman_qc_f1_v29_crco_research_result",
        "profile": PROFILE,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "s_mm": s,
        "selected_keyframe_hashes": [frame.frame_key for frame in keyframes],
        "queried_pixels": ledger.rows(),
        "queried_pixel_count": len(ledger.rows()),
        "decisions": list(decisions),
        "final_lines": final_lines,
        "cache": cache,
        "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "cache_bytes": len(cache_bytes),
        "ordinary_nms_after_v29": True,
        "candidate_added_after_nms": False,
        "legacy_wins_duplicate_collision": True,
    }


def select_unified_v29(
    raw_candidates: Sequence[dict[str, Any]],
    raw_frames: Sequence[dict[str, Any]],
    s: float,
) -> dict[str, Any]:
    """Run the bounded V29 ownership lane and the one final ordinary NMS."""

    scale = _reject_negative_zero(s, "s")
    if scale <= 0.0:
        raise CRCOContractError("INVALID_SCALE")
    candidates = parse_candidates(raw_candidates, scale)
    frames = parse_range_frames(raw_frames)
    keyframes = select_keyframes(frames)
    decisions, ledger = adjudicate_crco_v1(candidates, keyframes, scale)
    ordered = deterministic_total_sort(decisions)
    accepted = ordinary_nms(ordered)
    return build_result(accepted, decisions, keyframes, ledger, scale)


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    return _canonical_json_bytes(result) + b"\n"
