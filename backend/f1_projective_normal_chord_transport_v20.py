"""Projective normal chord transport (PNCT_V1), research-only.

This isolated module implements the source-free V20 primitive authorized by
the 2026-08-14 GPT-5.6 Sol Pro ruling.  It never reads labels, identities,
expected counts, CAD, or V16 decisions.  Native mesh geometry exists only
transiently; the persistent artifact is a 64-byte header plus one fixed
192-byte record per eligible candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE = "PROJECTIVE_NORMAL_CHORD_TRANSPORT_V1"
SCAN_HEADER_BYTES = 64
RECORD_BYTES = 192
STATION_COUNT = 11
BIN_COUNT = 12
MAX_SEGMENTS_PER_STATION = 8192
MAX_PARTNERS_PER_ORIGIN = 32
MAX_TRANSIENT_BYTES = 2 * 1024 * 1024

FLAG_RETAIN = 1 << 0
FAIL_INVALID = 1 << 1
FAIL_SEGMENT_OVERFLOW = 1 << 2
FAIL_PARTNER_OVERFLOW = 1 << 3
FAIL_NO_UNIQUE_ARC = 1 << 4
FAIL_NO_THICKNESS = 1 << 5
FAIL_STATION_COVERAGE = 1 << 6
FAIL_TAIL_COVERAGE = 1 << 7
FAIL_THICKNESS_DISPERSION = 1 << 8
FAIL_AMBIGUITY = 1 << 9
FAIL_COMPLEMENT = 1 << 10
FAIL_MIDPOINT_TRANSPORT = 1 << 11
FAIL_AXIAL_COHERENCE = 1 << 12
FAIL_NONFINITE = 1 << 13

COS20 = math.cos(math.radians(20.0))
SIN20 = math.sin(math.radians(20.0))


class PnctStructuralError(RuntimeError):
    """A frozen PNCT resource or structure gate was exceeded."""


def _unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected one finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("expected a nonzero vector")
    vector = vector / length
    vector[vector == 0.0] = 0.0
    return vector


def _endpoints(row: Mapping[str, Any]) -> np.ndarray:
    value = row.get("span_endpoints")
    if value is None and row.get("line_start") is not None and row.get("line_end") is not None:
        value = (row["line_start"], row["line_end"])
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("candidate has invalid endpoints")
    return result


def _hash128(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()[:16]


def _candidate_key(row: Mapping[str, Any]) -> bytes:
    raw = row.get("candidate_key")
    if isinstance(raw, str):
        try:
            value = bytes.fromhex(raw)
            if len(value) >= 16:
                return value[:16]
        except ValueError:
            pass
    return _hash128(str(row.get("candidate_id") or "").encode("utf8"))


def _q_length(value: float | None, scale: float) -> int:
    if value is None or not math.isfinite(value):
        return 0xFFFF
    return min(0xFFFE, max(0, int(np.rint(4096.0 * value / scale))))


def _q_fraction(value: float | None) -> int:
    if value is None or not math.isfinite(value):
        return 0xFFFF
    return min(0xFFFE, max(0, int(np.rint(65535.0 * value))))


def _q_u8(value: float | None) -> int:
    if value is None or not math.isfinite(value):
        return 0xFF
    return min(0xFE, max(0, int(np.rint(255.0 * value))))


def _canonical_sequence(values: Sequence[int]) -> tuple[int, ...]:
    forward = tuple(int(value) for value in values)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _weighted_median(rows: Sequence[tuple[float, int, bytes]]) -> float | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (row[0], row[2]))
    total = sum(max(0, int(row[1])) for row in ordered)
    if total <= 0:
        return None
    cumulative = 0
    for value, weight, _key in ordered:
        cumulative += max(0, int(weight))
        if 2 * cumulative >= total:
            return float(value)
    return float(ordered[-1][0])


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return math.inf
    index = max(0, min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1))
    return ordered[index]


def _projective_angle(vector: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    angle = math.atan2(float(np.dot(vector, v)), float(np.dot(vector, u))) % math.pi
    return 0.0 if abs(angle - math.pi) <= 1e-14 else angle


@dataclass(frozen=True)
class Frame:
    origin: np.ndarray
    axis: np.ndarray
    u: np.ndarray
    v: np.ndarray
    length: float
    scale: float
    seed_angle: float

    def local_point(self, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, dtype=np.float64) - self.origin
        raw = np.asarray((np.dot(delta, self.axis), np.dot(delta, self.u), np.dot(delta, self.v)))
        quantum = math.ldexp(self.scale, -36)
        result = np.rint(raw / quantum) * quantum
        result[result == 0.0] = 0.0
        return result


@dataclass(frozen=True)
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    midpoint: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float
    key: bytes
    angle: float


@dataclass(frozen=True)
class Chord:
    distance: float
    midpoint: np.ndarray
    weight: int
    key: bytes
    ambiguous: bool


@dataclass
class BinEvidence:
    raw_weight: int = 0
    valid_weight: int = 0
    ambiguous_weight: int = 0
    valid_count: int = 0
    chords: list[Chord] = field(default_factory=list)
    owned: bool = False
    thickness: float | None = None
    midpoint: np.ndarray | None = None

    @property
    def coverage(self) -> float:
        return self.valid_weight / self.raw_weight if self.raw_weight > 0 else 0.0


@dataclass
class StationEvidence:
    index: int
    coordinate: float
    segments: tuple[Segment, ...]
    bins: tuple[BinEvidence, ...]
    provisional: bool = False
    final: bool = False
    path_length: float = 0.0
    displacement: float = 0.0
    max_step: float = 0.0
    thickness: float | None = None
    reason: str = ""
    midpoint_path: tuple[np.ndarray, ...] = ()


@dataclass
class ArcEvidence:
    start: float
    length: float
    stations: tuple[StationEvidence, ...]
    complement_valid_weight: int
    selected_valid_weight: int
    tau: float | None = None
    pass_arc: bool = False
    failure_mask: int = 0
    axial_rms: float = math.inf
    thickness_p90: float = math.inf
    ambiguity_fraction: float = 1.0


@dataclass
class PnctRecord:
    candidate_id: str
    candidate_key: bytes
    flags: int
    failure_mask: int
    scale: float
    selected_arc: ArcEvidence | None
    arcs: tuple[ArcEvidence, ...]
    structural_counters: tuple[int, ...]
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def retain(self) -> bool:
        return bool(self.flags & FLAG_RETAIN)

    def encode(self) -> bytes:
        result = bytearray(RECORD_BYTES)
        result[0:4] = b"PNC1"
        struct.pack_into("<HH", result, 4, 1, self.flags & 0xFFFF)
        result[8:24] = self.candidate_key[:16].ljust(16, b"\0")
        struct.pack_into("<Q", result, 24, self.failure_mask & 0xFFFFFFFFFFFFFFFF)
        arc = self.selected_arc
        passing = [station for station in arc.stations if station.final] if arc else []
        metrics = (
            _q_length(arc.tau if arc else None, self.scale),
            _q_fraction(arc.thickness_p90 if arc and math.isfinite(arc.thickness_p90) else None),
            _q_fraction(arc.axial_rms / arc.tau if arc and arc.tau and math.isfinite(arc.axial_rms) else None),
            _q_fraction(arc.ambiguity_fraction if arc else None),
            _q_fraction((arc.complement_valid_weight / arc.selected_valid_weight) if arc and arc.selected_valid_weight else None),
            len(passing),
            sum(sum(bin_.owned for bin_ in station.bins[1:11]) for station in passing),
            0,
        )
        struct.pack_into("<8H", result, 32, *(int(value) & 0xFFFF for value in metrics))
        if arc:
            cover = _canonical_sequence([_q_u8(sum(station.bins[b].coverage for station in arc.stations) / STATION_COUNT) for b in range(BIN_COUNT)])
            owned = _canonical_sequence([sum(station.bins[b].owned for station in arc.stations) for b in range(BIN_COUNT)])
            thick = _canonical_sequence([
                _q_length(_weighted_median([(chord.distance, chord.weight, chord.key) for station in arc.stations for chord in station.bins[b].chords]), self.scale)
                for b in range(BIN_COUNT)
            ])
            station_thick = [_q_length(station.thickness, self.scale) for station in arc.stations]
            station_flags = [int(station.provisional) | (int(station.final) << 1) for station in arc.stations]
            station_cover = [_q_u8(sum(bin_.valid_weight for bin_ in station.bins) / max(1, sum(bin_.raw_weight for bin_ in station.bins))) for station in arc.stations]
            complement = self.arcs[1] if len(self.arcs) == 2 and self.arcs[0] is arc else self.arcs[0] if len(self.arcs) == 2 else None
            complement_cover = _canonical_sequence([_q_u8(sum(station.bins[b].coverage for station in complement.stations) / STATION_COUNT) if complement else 0xFF for b in range(BIN_COUNT)])
        else:
            cover, owned, thick = (0xFF,) * 12, (0,) * 12, (0xFFFF,) * 12
            station_thick, station_flags, station_cover = [0xFFFF] * 11, [0] * 11, [0xFF] * 11
            complement_cover = (0xFF,) * 12
        struct.pack_into("<12B", result, 48, *cover)
        struct.pack_into("<12B", result, 60, *owned)
        struct.pack_into("<12H", result, 72, *thick)
        struct.pack_into("<11H", result, 96, *station_thick)
        struct.pack_into("<11B", result, 118, *station_flags)
        struct.pack_into("<11B", result, 129, *station_cover)
        struct.pack_into("<12B", result, 140, *complement_cover)
        counters = tuple(int(value) & 0xFFFFFFFF for value in self.structural_counters[:8]) + (0,) * max(0, 8 - len(self.structural_counters))
        struct.pack_into("<8I", result, 152, *counters[:8])
        assert len(result) == RECORD_BYTES
        return bytes(result)

    def summary(self) -> dict[str, Any]:
        arc = self.selected_arc
        return {
            "candidate_id": self.candidate_id,
            "retain": self.retain,
            "flags": self.flags,
            "failure_mask": self.failure_mask,
            "tau": arc.tau if arc else None,
            "passing_stations": sum(station.final for station in arc.stations) if arc else 0,
            "arc_passes": [item.pass_arc for item in self.arcs],
            "arc_failure_masks": [item.failure_mask for item in self.arcs],
            "structural_counters": list(self.structural_counters),
            "record_bytes": len(self.encode()),
            "debug": self.debug,
        }


def encode_scan_header(records: Sequence[PnctRecord]) -> bytes:
    body = b"".join(record.encode() for record in records)
    result = bytearray(SCAN_HEADER_BYTES)
    result[:8] = b"PNCTV1\0\0"
    result[8:24] = _hash128(PROFILE.encode("ascii"))
    struct.pack_into("<II", result, 24, len(records), RECORD_BYTES)
    result[32:64] = hashlib.sha256(body).digest()
    return bytes(result)


def encode_sidecar(records: Sequence[PnctRecord]) -> bytes:
    ordered = sorted(records, key=lambda record: record.candidate_key)
    return encode_scan_header(ordered) + b"".join(record.encode() for record in ordered)


def _build_frame(row: Mapping[str, Any], scale: float, seed_normals: tuple[Sequence[float], Sequence[float]]) -> Frame:
    endpoints = _endpoints(row)
    axis = _unit(endpoints[1] - endpoints[0])
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    origin = np.mean(endpoints, axis=0)
    projected = []
    for normal in seed_normals:
        vector = _unit(normal)
        vector = vector - float(np.dot(vector, axis)) * axis
        projected.append(_unit(vector))
    u = projected[0]
    v = _unit(np.cross(axis, u))
    angle = _projective_angle(projected[1], u, v)
    if angle <= 1e-12:
        raise ValueError("degenerate projected seed normals")
    return Frame(origin, axis, u, v, length, float(scale), angle)


def _triangle_plane_segment(local: np.ndarray, coordinate: float, epsilon: float) -> tuple[np.ndarray, np.ndarray] | None:
    distances = local[:, 0] - coordinate
    distances[np.abs(distances) <= epsilon] = 0.0
    if np.all(distances == 0.0):
        return None
    points: list[np.ndarray] = []
    for i in range(3):
        j = (i + 1) % 3
        di, dj = float(distances[i]), float(distances[j])
        if di == 0.0:
            points.append(local[i].copy())
        if di * dj < 0.0:
            fraction = di / (di - dj)
            points.append(local[i] + fraction * (local[j] - local[i]))
    unique: dict[tuple[int, int], np.ndarray] = {}
    for point in points:
        key = (int(np.rint(point[1] / epsilon)), int(np.rint(point[2] / epsilon)))
        unique.setdefault(key, point)
    values = list(unique.values())
    if len(values) != 2:
        return None
    return values[0], values[1]


def _cross_section(vertices: np.ndarray, triangles: np.ndarray, frame: Frame, coordinate: float) -> tuple[Segment, ...]:
    epsilon = math.ldexp(frame.scale, -36)
    endpoint_quantum = frame.scale / 4096.0
    result: dict[bytes, Segment] = {}
    for face in np.asarray(triangles, dtype=np.int64):
        local = np.asarray([frame.local_point(vertices[index]) for index in face])
        if np.min(local[:, 0]) > coordinate or np.max(local[:, 0]) < coordinate:
            continue
        segment = _triangle_plane_segment(local, coordinate, epsilon)
        if segment is None:
            continue
        p0, p1 = segment
        midpoint = (p0 + p1) / 2.0
        if math.hypot(float(midpoint[1]), float(midpoint[2])) > 36.0 * frame.scale:
            continue
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 0.10 * frame.scale:
            continue
        tangent = _unit(delta)
        normal = _unit(np.cross(np.asarray((1.0, 0.0, 0.0)), tangent))
        angle = math.atan2(float(normal[2]), float(normal[1])) % math.pi
        keys = sorted((tuple(int(np.rint(value / endpoint_quantum)) for value in p0[1:]), tuple(int(np.rint(value / endpoint_quantum)) for value in p1[1:])))
        key = _hash128(repr(keys).encode("ascii"))
        result.setdefault(key, Segment(p0, p1, midpoint, tangent, normal, length, key, angle))
        if len(result) > MAX_SEGMENTS_PER_STATION:
            raise PnctStructuralError("SEGMENT_OVERFLOW")
    return tuple(sorted(result.values(), key=lambda item: item.key))


def _arc_position(angle: float, start: float, length: float) -> float | None:
    delta = (angle - start) % math.pi
    if delta > length + 1e-14:
        return None
    return min(length, max(0.0, delta))


def _bin_segments(segments: Sequence[Segment], start: float, length: float) -> tuple[list[Segment], ...]:
    bins: list[list[Segment]] = [[] for _ in range(BIN_COUNT)]
    for segment in segments:
        position = _arc_position(segment.angle, start, length)
        if position is None:
            continue
        raw = BIN_COUNT * position / length
        index = min(BIN_COUNT - 1, int(math.floor(raw)))
        if raw > 0.0 and abs(raw - round(raw)) <= 1e-14:
            index = max(0, int(round(raw)) - 1)
        bins[index].append(segment)
    return tuple(bins)


def _bin_evidence(segments: Sequence[Segment], frame: Frame, representative_angle: float) -> BinEvidence:
    evidence = BinEvidence()
    t_b = np.asarray((0.0, -math.sin(representative_angle), math.cos(representative_angle)))
    for origin in segments:
        weight = int(np.rint(4096.0 * origin.length / frame.scale))
        evidence.raw_weight += weight
        partners: list[tuple[float, bytes, Segment]] = []
        for candidate in segments:
            if candidate.key == origin.key:
                continue
            delta = candidate.midpoint - origin.midpoint
            tangential = abs(float(np.dot(delta, t_b)))
            if tangential <= 2.0 * frame.scale:
                partners.append((tangential, candidate.key, candidate))
        partners.sort(key=lambda row: (row[0], row[1]))
        if len(partners) > MAX_PARTNERS_PER_ORIGIN:
            raise PnctStructuralError("PARTNER_DENSITY_OVERFLOW")
        valid: list[tuple[float, bytes, Segment]] = []
        for _tangential, key, partner in partners:
            delta = partner.midpoint - origin.midpoint
            distance = float(np.linalg.norm(delta))
            if not 1.5 * frame.scale <= distance <= 16.0 * frame.scale:
                continue
            if abs(float(np.dot(origin.tangent, partner.tangent))) < COS20:
                continue
            if abs(float(np.dot(delta, origin.tangent))) > SIN20 * distance:
                continue
            if abs(float(np.dot(delta, partner.tangent))) > SIN20 * distance:
                continue
            if abs(float(np.dot(delta, t_b))) > min(2.0 * frame.scale, SIN20 * distance):
                continue
            valid.append((distance, key, partner))
        if not valid:
            continue
        valid.sort(key=lambda row: (row[0], row[1]))
        distance, partner_key, partner = valid[0]
        ambiguous = len(valid) > 1 and valid[1][0] <= 1.25 * distance
        chord_key = min(origin.key + partner_key, partner_key + origin.key)
        chord = Chord(distance, (origin.midpoint + partner.midpoint) / 2.0, weight, chord_key, ambiguous)
        evidence.chords.append(chord)
        evidence.valid_count += 1
        evidence.valid_weight += weight
        if ambiguous:
            evidence.ambiguous_weight += weight
    evidence.owned = (
        evidence.raw_weight >= int(np.rint(4096.0 * 0.75))
        and evidence.valid_count >= 2
        and evidence.valid_weight >= math.ceil(0.55 * evidence.raw_weight)
        and evidence.ambiguous_weight <= math.floor(0.20 * evidence.valid_weight)
    )
    if evidence.chords:
        evidence.thickness = _weighted_median([(row.distance, row.weight, row.key) for row in evidence.chords])
        components = []
        for index in range(3):
            value = _weighted_median([(float(row.midpoint[index]), row.weight, row.key) for row in evidence.chords])
            components.append(0.0 if value is None else value)
        evidence.midpoint = np.asarray(components, dtype=np.float64)
    return evidence


def _provisional_station(index: int, coordinate: float, segments: tuple[Segment, ...], frame: Frame, start: float, length: float) -> StationEvidence:
    assigned = _bin_segments(segments, start, length)
    bins = tuple(_bin_evidence(rows, frame, start + (b + 0.5) * length / BIN_COUNT) for b, rows in enumerate(assigned))
    station = StationEvidence(index, coordinate, segments, bins)
    interior = [bins[index].owned for index in range(1, 11)]
    coverage = sum(bin_.valid_weight for bin_ in bins) / max(1, sum(bin_.raw_weight for bin_ in bins))
    station.provisional = bins[0].owned and bins[11].owned and sum(interior) >= 7 and not any(not interior[i] and not interior[i + 1] for i in range(9)) and coverage >= 0.60
    if not station.provisional:
        station.reason = "coverage"
        return station
    path: list[np.ndarray] = []
    for b, bin_ in enumerate(bins):
        if bin_.owned and bin_.midpoint is not None:
            path.append(bin_.midpoint)
        elif 0 < b < 11 and bins[b - 1].owned and bins[b + 1].owned and bins[b - 1].midpoint is not None and bins[b + 1].midpoint is not None:
            path.append((bins[b - 1].midpoint + bins[b + 1].midpoint) / 2.0)
        else:
            station.reason = "noninterpolable_gap"
            station.provisional = False
            return station
    station.midpoint_path = tuple(path)
    steps = [float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1)]
    station.path_length = math.fsum(steps)
    station.displacement = float(np.linalg.norm(path[-1] - path[0]))
    station.max_step = max(steps, default=0.0)
    station.thickness = _weighted_median([(chord.distance, chord.weight, chord.key) for bin_ in bins for chord in bin_.chords])
    return station


def _evaluate_arc(segments_by_station: Sequence[tuple[Segment, ...]], frame: Frame, start: float, length: float, complement_valid_weight: int = 0) -> ArcEvidence:
    stations = tuple(_provisional_station(index, (index - 5) * frame.length / 12.0, segments, frame, start, length) for index, segments in enumerate(segments_by_station))
    all_chords = [(chord.distance, chord.weight, chord.key) for station in stations if station.provisional for bin_ in station.bins for chord in bin_.chords]
    tau = _weighted_median(all_chords)
    selected_valid = sum(chord.weight for station in stations for bin_ in station.bins for chord in bin_.chords)
    evidence = ArcEvidence(start, length, stations, complement_valid_weight, selected_valid, tau=tau)
    if tau is None or not 1.5 * frame.scale <= tau <= 16.0 * frame.scale:
        evidence.failure_mask |= FAIL_NO_THICKNESS
        return evidence
    for station in stations:
        if not station.provisional:
            continue
        station.final = station.path_length >= tau and station.displacement >= 0.65 * tau and station.max_step <= 0.40 * station.path_length
        if not station.final:
            station.reason = "midpoint_collapse"
    passing = [station for station in stations if station.final]
    if len(passing) < 8:
        evidence.failure_mask |= FAIL_STATION_COVERAGE
    if any(sum(station.bins[b].owned for station in stations) < 9 for b in (0, 11)):
        evidence.failure_mask |= FAIL_TAIL_COVERAGE
    tail_thickness = [_weighted_median([(chord.distance, chord.weight, chord.key) for station in stations for chord in station.bins[b].chords]) for b in (0, 11)]
    if any(value is None or abs(value - tau) > 0.20 * tau for value in tail_thickness):
        evidence.failure_mask |= FAIL_THICKNESS_DISPERSION
    deviations = [abs(chord.distance - tau) / tau for station in stations for bin_ in station.bins for chord in bin_.chords]
    station_deviations = [abs(station.thickness - tau) / tau for station in passing if station.thickness is not None]
    evidence.thickness_p90 = _nearest_rank(deviations, 0.90)
    if evidence.thickness_p90 > 0.20 or _nearest_rank(station_deviations, 0.90) > 0.15:
        evidence.failure_mask |= FAIL_THICKNESS_DISPERSION
    valid_weight = sum(bin_.valid_weight for station in stations for bin_ in station.bins)
    ambiguous_weight = sum(bin_.ambiguous_weight for station in stations for bin_ in station.bins)
    evidence.ambiguity_fraction = ambiguous_weight / valid_weight if valid_weight else 1.0
    if evidence.ambiguity_fraction > 0.15:
        evidence.failure_mask |= FAIL_AMBIGUITY
    if selected_valid and complement_valid_weight / selected_valid > 0.15:
        evidence.failure_mask |= FAIL_COMPLEMENT
    if not passing or float(np.median([station.path_length / tau for station in passing])) < 1.25:
        evidence.failure_mask |= FAIL_MIDPOINT_TRANSPORT
    if passing:
        profiles = np.asarray([[point[1:] for point in station.midpoint_path] for station in passing])
        median = np.median(profiles, axis=0)
        evidence.axial_rms = float(np.sqrt(np.mean((profiles - median) ** 2)))
        if evidence.axial_rms > 0.35 * tau:
            evidence.failure_mask |= FAIL_AXIAL_COHERENCE
    else:
        evidence.failure_mask |= FAIL_AXIAL_COHERENCE
    evidence.pass_arc = evidence.failure_mask == 0
    return evidence


def analyze_candidate(*, vertices: np.ndarray, triangles: np.ndarray, row: Mapping[str, Any], scale: float, seed_normals: tuple[Sequence[float], Sequence[float]]) -> PnctRecord:
    candidate_id = str(row.get("candidate_id") or "")
    key = _candidate_key(row)
    counters = [0] * 8
    try:
        source = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(triangles, dtype=np.int64)
        if source.ndim != 2 or source.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not np.isfinite(source).all() or scale <= 0.0:
            raise ValueError("invalid mesh")
        frame = _build_frame(row, scale, seed_normals)
        if frame.length < 12.0 * frame.scale:
            raise ValueError("insufficient support")
        segments_by_station = tuple(_cross_section(source, faces, frame, (index - 5) * frame.length / 12.0) for index in range(STATION_COUNT))
        counters[0] = len(source)
        counters[1] = len(faces)
        counters[2] = sum(len(rows) for rows in segments_by_station)
        lengths = (frame.seed_angle, math.pi - frame.seed_angle)
        starts = (0.0, frame.seed_angle)
        raw_arcs: list[ArcEvidence] = []
        for start, length in zip(starts, lengths):
            degrees = math.degrees(length)
            if not 25.0 <= degrees <= 155.0:
                raw_arcs.append(ArcEvidence(start, length, (), 0, 0, failure_mask=FAIL_INVALID))
                continue
            raw_arcs.append(_evaluate_arc(segments_by_station, frame, start, length))
        if len(raw_arcs) == 2:
            # The two arcs share their seed-normal boundary lines.  Complement
            # support is therefore the support strictly outside the selected
            # interval (rotation-interior bins 1..10), not the common tails.
            # Counting the shared tail bins twice would make every physical
            # positive with the required planar tails fail by construction.
            first_weight = sum(
                chord.weight for station in raw_arcs[0].stations
                for bin_ in station.bins[1:11] for chord in bin_.chords
            )
            second_weight = sum(
                chord.weight for station in raw_arcs[1].stations
                for bin_ in station.bins[1:11] for chord in bin_.chords
            )
            raw_arcs[0] = _evaluate_arc(segments_by_station, frame, starts[0], lengths[0], second_weight)
            raw_arcs[1] = _evaluate_arc(segments_by_station, frame, starts[1], lengths[1], first_weight)
        arcs = tuple(raw_arcs)
        passing = [arc for arc in arcs if arc.pass_arc]
        selected = passing[0] if len(passing) == 1 else None
        failure = 0 if selected else FAIL_NO_UNIQUE_ARC
        if not selected:
            for arc in arcs:
                failure |= arc.failure_mask
        counters[3] = sum(station.provisional for arc in arcs for station in arc.stations)
        counters[4] = sum(station.final for arc in arcs for station in arc.stations)
        counters[5] = sum(bin_.valid_count for arc in arcs for station in arc.stations for bin_ in station.bins)
        counters[6] = sum(bin_.ambiguous_weight for arc in arcs for station in arc.stations for bin_ in station.bins)
        counters[7] = sum(bin_.raw_weight for arc in arcs for station in arc.stations for bin_ in station.bins)
        return PnctRecord(candidate_id, key, FLAG_RETAIN if selected else 0, failure, scale, selected, arcs, tuple(counters), debug={"station_segment_counts": [len(rows) for rows in segments_by_station]})
    except PnctStructuralError as exc:
        bit = FAIL_SEGMENT_OVERFLOW if "SEGMENT" in str(exc) else FAIL_PARTNER_OVERFLOW
        return PnctRecord(candidate_id, key, 0, bit, scale, None, (), tuple(counters), debug={"structural_error": str(exc)})
    except (ValueError, FloatingPointError) as exc:
        return PnctRecord(candidate_id, key, 0, FAIL_INVALID, scale, None, (), tuple(counters), debug={"error": str(exc)})
