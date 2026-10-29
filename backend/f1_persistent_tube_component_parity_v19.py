"""Persistent tube component parity (PTCP_V1), research-only.

This module implements the single post-PSRO source-only architecture frozen by
the 2026-08-14 GPT-5.6 Sol Pro ruling.  It is deliberately isolated from the
runtime detector and from V17/V18 decision fields.  Native geometry is held
only transiently; the sole persistent artifact is a 64-byte scan header plus
one fixed 224-byte record per eligible candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROFILE = "PERSISTENT_TUBE_COMPONENT_PARITY_V1"
SCAN_HEADER_BYTES = 64
RECORD_BYTES = 224

MAX_FAN_TRIANGLES = 65_536
MAX_INITIAL_COMPONENTS = 16_384
MAX_PROXIMITY_EDGES = 262_144
MAX_TRANSIENT_BYTES = 256 * 1024 * 1024

STATUS_RETAIN = 1 << 0
FAIL_INVALID_CANDIDATE = 1 << 1
FAIL_INVALID_PARENTS = 1 << 2
FAIL_TUBE_TOO_SMALL = 1 << 3
FAIL_NO_UNIQUE_INTERVAL = 1 << 4
FAIL_DIRECT_DEATH = 1 << 5
FAIL_DEATH_SCALE = 1 << 6
FAIL_PERSISTENCE = 1 << 7
FAIL_THIRD_SHEET = 1 << 8
FAIL_OFFSET_COVERAGE = 1 << 9
FAIL_NONFINITE = 1 << 10
FAIL_STRUCTURAL_CAP = 1 << 11
FAIL_DUPLICATE_NMS = 1 << 12

COS_PORT = math.cos(math.pi / 8.0)
COS_NORMAL = math.cos(math.radians(25.0))
COS_OFFSET = math.cos(math.radians(35.0))


class PtcpStructuralCapError(RuntimeError):
    """A frozen PTCP per-candidate resource bound was exceeded."""


def _unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected one finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("expected a nonzero vector")
    result = vector / length
    result[result == 0.0] = 0.0
    return result


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


def _quantized_tuple(value: np.ndarray, scale: float) -> tuple[int, ...]:
    return tuple(int(np.rint(float(item) / scale)) for item in np.asarray(value).ravel())


def _canonical_triangle_key(vertices: np.ndarray, scale: float) -> bytes:
    rows = sorted(_quantized_tuple(row, scale) for row in np.asarray(vertices))
    return _hash128(repr(rows).encode("ascii"))


def _aabb_distance(left_min: np.ndarray, left_max: np.ndarray, right_min: np.ndarray, right_max: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(left_min - right_max, right_min - left_max))
    return float(np.linalg.norm(delta))


def _point_triangle_closest(point: np.ndarray, tri: np.ndarray) -> tuple[float, np.ndarray]:
    # Ericson, Real-Time Collision Detection, closest point on triangle.
    a, b, c = tri
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(point - a)), a
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(point - b)), b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        return float(np.linalg.norm(point - q)), q
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(point - c)), c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        return float(np.linalg.norm(point - q)), q
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * (c - b)
        return float(np.linalg.norm(point - q)), q
    denominator = va + vb + vc
    if denominator == 0.0:
        candidates = [(float(np.linalg.norm(point - vertex)), vertex) for vertex in tri]
        return min(candidates, key=lambda item: item[0])
    v, w = vb / denominator, vc / denominator
    q = a + ab * v + ac * w
    return float(np.linalg.norm(point - q)), q


def _segment_segment_closest(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(np.dot(d1, d1)), float(np.dot(d2, d2)), float(np.dot(d2, r))
    eps = 1e-30
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2)), p1, p2
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0)) if denominator != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    left, right = p1 + s * d1, p2 + t * d2
    return float(np.linalg.norm(left - right)), left, right


def _triangle_triangle_closest(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for vertex in left:
        distance, closest = _point_triangle_closest(vertex, right)
        item = (distance, vertex, closest)
        if best is None or distance < best[0]:
            best = item
    for vertex in right:
        distance, closest = _point_triangle_closest(vertex, left)
        item = (distance, closest, vertex)
        if best is None or distance < best[0]:
            best = item
    for i in range(3):
        for j in range(3):
            item = _segment_segment_closest(left[i], left[(i + 1) % 3], right[j], right[(j + 1) % 3])
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    return best


def _clip_polygon(polygon: list[np.ndarray], normal: np.ndarray, bound: float, epsilon: float) -> list[np.ndarray]:
    if not polygon:
        return []
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(np.dot(normal, previous) - bound)
    previous_inside = previous_value <= epsilon
    for current in polygon:
        current_value = float(np.dot(normal, current) - bound)
        current_inside = current_value <= epsilon
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            t = previous_value / denominator if denominator != 0.0 else 0.0
            output.append(previous + t * (current - previous))
        if current_inside:
            output.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return output


def _canonical_polygon(polygon: list[np.ndarray], quantum: float) -> list[np.ndarray]:
    if len(polygon) < 3:
        return []
    # Remove adjacent duplicate clipping vertices.
    cleaned: list[np.ndarray] = []
    for point in polygon:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > quantum / 8.0:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= quantum / 8.0:
        cleaned.pop()
    if len(cleaned) < 3:
        return []
    keys = [_quantized_tuple(point, quantum) for point in cleaned]
    start = min(range(len(cleaned)), key=lambda index: keys[index])
    forward = cleaned[start:] + cleaned[:start]
    reverse = [forward[0]] + list(reversed(forward[1:]))
    forward_key = [_quantized_tuple(point, quantum) for point in forward]
    reverse_key = [_quantized_tuple(point, quantum) for point in reverse]
    return forward if forward_key <= reverse_key else reverse


@dataclass(frozen=True)
class CandidateFrame:
    origin: np.ndarray
    direction: np.ndarray
    transverse_a: np.ndarray
    transverse_b: np.ndarray
    basis_v: np.ndarray
    length: float
    scale: float
    radius: float
    arm_angle_deg: float

    def local(self, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, dtype=np.float64) - self.origin
        raw = np.asarray((np.dot(delta, self.direction), np.dot(delta, self.transverse_a), np.dot(delta, self.basis_v)))
        # All construction arithmetic is performed in one canonical,
        # dimensionless candidate frame.  Rounding at 2^-40 of one native
        # spacing removes only coordinate-transform ULP noise (roughly eight
        # orders finer than the frozen s/1024 filtration quantum), while
        # keeping exact conservative nextafter/ceil distance quantization.
        quantum = math.ldexp(self.scale, -40)
        result = np.rint(raw / quantum) * quantum
        result[result == 0.0] = 0.0
        return result

    def local_vector(self, vector: np.ndarray) -> np.ndarray:
        raw = np.asarray((np.dot(vector, self.direction), np.dot(vector, self.transverse_a), np.dot(vector, self.basis_v)))
        quantum = math.ldexp(1.0, -40)
        result = np.rint(raw / quantum) * quantum
        result[result == 0.0] = 0.0
        return result

    @property
    def arm_a_local(self) -> np.ndarray:
        return self.local_vector(self.transverse_a)

    @property
    def arm_b_local(self) -> np.ndarray:
        return self.local_vector(self.transverse_b)


@dataclass
class FanTriangle:
    vertices: np.ndarray
    area: float
    centroid: np.ndarray
    normal: np.ndarray
    native_key: bytes
    patch_key: bytes
    axial_bin: int
    port_a: bool
    port_b: bool
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass
class Component:
    key: bytes
    fan_ids: tuple[int, ...]
    area: float
    port_a_area: np.ndarray
    port_b_area: np.ndarray
    axial_min: float
    axial_max: float
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass(frozen=True)
class ProximityEdge:
    q: int
    distance: float
    left: int
    right: int
    left_point: np.ndarray
    right_point: np.ndarray
    left_normal: np.ndarray
    right_normal: np.ndarray
    left_fan_key: bytes
    right_fan_key: bytes


@dataclass
class _BvhNode:
    minimum: np.ndarray
    maximum: np.ndarray
    fan_ids: tuple[int, ...] = ()
    left: "_BvhNode | None" = None
    right: "_BvhNode | None" = None

    @property
    def leaf(self) -> bool:
        return self.left is None and self.right is None


class _UnionFind:
    def __init__(self, components: Sequence[Component]):
        count = len(components)
        self.parent = list(range(count))
        self.members = [set((index,)) for index in range(count)]
        self.port_a = [component.port_a_area.copy() for component in components]
        self.port_b = [component.port_b_area.copy() for component in components]
        self.area = [component.area for component in components]
        self.key = [component.key for component in components]

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> int:
        left, right = self.find(left), self.find(right)
        if left == right:
            return left
        # Canonical component key, never input/component encounter order.
        if self.key[right] < self.key[left]:
            left, right = right, left
        self.parent[right] = left
        self.members[left].update(self.members[right])
        self.port_a[left] = np.asarray([math.fsum((float(self.port_a[left][i]), float(self.port_a[right][i]))) for i in range(8)])
        self.port_b[left] = np.asarray([math.fsum((float(self.port_b[left][i]), float(self.port_b[right][i]))) for i in range(8)])
        self.area[left] = math.fsum((self.area[left], self.area[right]))
        self.key[left] = min(self.key[left], self.key[right])
        return left

    def roots(self) -> list[int]:
        return sorted({self.find(index) for index in range(len(self.parent))}, key=lambda root: self.key[root])


@dataclass
class IntervalEvidence:
    birth_q: int
    death_q: int
    birth_roots: tuple[frozenset[int], frozenset[int]]
    predeath_roots: tuple[frozenset[int], frozenset[int]]
    third_sheet_ok: bool
    direct_edges: tuple[ProximityEdge, ...]
    axial_mask: int = 0
    a_cross_mask: int = 0
    b_cross_mask: int = 0
    offset_valid_mask: int = 0
    offset_valid_fraction: float = 0.0
    port_exclusivity_fraction: float = 0.0
    qualifying: bool = False


@dataclass
class PtcpRecord:
    candidate_id: str
    canonical_key: bytes
    source_rank: int
    status_bits: int
    native_triangle_count: int = 0
    fan_count: int = 0
    component_count: int = 0
    edge_count: int = 0
    birth_q: int = 0
    death_q: int = 0
    root_ids: tuple[int, int] = (0, 0)
    scale: float = 0.0
    radius: float = 0.0
    length: float = 0.0
    birth_gap: float = 0.0
    death_gap: float = 0.0
    persistence: float = 0.0
    death_radius_ratio: float = 0.0
    arm_angle_deg: float = 0.0
    offset_valid_fraction: float = 0.0
    port_exclusivity_fraction: float = 0.0
    masks: tuple[int, ...] = (0,) * 8
    component_summary: tuple[float, ...] = (0.0,) * 8
    death_patch_ids: tuple[int, ...] = (0,) * 4
    topology_hash: bytes = bytes(16)
    construction_hash: bytes = field(default_factory=lambda: _hash128(PROFILE.encode("ascii")))
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def retain(self) -> bool:
        return bool(self.status_bits & STATUS_RETAIN)

    def encode(self) -> bytes:
        result = bytearray(RECORD_BYTES)
        candidate_digest = hashlib.sha256(self.candidate_id.encode("utf8")).digest()[:8]
        result[0:8] = candidate_digest
        result[8:24] = self.canonical_key[:16].ljust(16, b"\0")
        struct.pack_into("<II", result, 24, self.source_rank & 0xFFFFFFFF, self.status_bits & 0xFFFFFFFF)
        struct.pack_into(
            "<8I", result, 32,
            self.native_triangle_count, self.fan_count, self.component_count, self.edge_count,
            self.birth_q, self.death_q, self.root_ids[0], self.root_ids[1],
        )
        normalizer = self.scale if self.scale > 0.0 else 1.0
        scalar = (
            1.0 if self.scale > 0.0 else 0.0,
            self.radius / normalizer,
            self.length / normalizer,
            self.birth_gap / normalizer,
            self.death_gap / normalizer,
            self.persistence,
            self.death_radius_ratio,
            self.arm_angle_deg,
            self.offset_valid_fraction,
            self.port_exclusivity_fraction,
        )
        struct.pack_into("<10f", result, 64, *scalar)
        struct.pack_into("<8H", result, 104, *(int(value) & 0xFFFF for value in self.masks[:8]))
        summary = tuple(float(value) for value in self.component_summary[:8]) + (0.0,) * max(0, 8 - len(self.component_summary))
        struct.pack_into("<8f", result, 120, *summary[:8])
        patch_ids = tuple(int(value) & 0xFFFFFFFF for value in self.death_patch_ids[:4]) + (0,) * max(0, 4 - len(self.death_patch_ids))
        struct.pack_into("<4I", result, 152, *patch_ids[:4])
        result[168:184] = self.topology_hash[:16].ljust(16, b"\0")
        result[184:200] = self.construction_hash[:16].ljust(16, b"\0")
        # 200:224 is frozen zero-filled reserved space.
        assert len(result) == RECORD_BYTES
        return bytes(result)

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "retain": self.retain,
            "status_bits": self.status_bits,
            "native_triangle_count": self.native_triangle_count,
            "clipped_fan_count": self.fan_count,
            "initial_component_count": self.component_count,
            "proximity_edge_count": self.edge_count,
            "birth_q": self.birth_q,
            "death_q": self.death_q,
            "birth_gap": self.birth_gap,
            "death_gap": self.death_gap,
            "persistence": self.persistence,
            "death_radius_ratio": self.death_radius_ratio,
            "masks": list(self.masks),
            "topology_hash": self.topology_hash.hex(),
            "record_bytes": len(self.encode()),
            "debug": self.debug,
        }


def encode_scan_header(records: Sequence[PtcpRecord]) -> bytes:
    result = bytearray(SCAN_HEADER_BYTES)
    result[:8] = b"PTCPV1\0\0"
    result[8:24] = _hash128(PROFILE.encode("ascii"))
    struct.pack_into("<II", result, 24, len(records), RECORD_BYTES)
    digest = hashlib.sha256(b"".join(record.encode() for record in records)).digest()
    result[32:64] = digest
    return bytes(result)


def encode_sidecar(records: Sequence[PtcpRecord]) -> bytes:
    return encode_scan_header(records) + b"".join(record.encode() for record in records)


def _build_frame(
    row: Mapping[str, Any],
    scale: float,
    parent_normals: tuple[np.ndarray, np.ndarray],
    parent_centroids: tuple[np.ndarray, np.ndarray],
) -> CandidateFrame:
    endpoints = _endpoints(row)
    direction = _unit(endpoints[1] - endpoints[0])
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    origin = np.mean(endpoints, axis=0)
    arms: list[np.ndarray] = []
    for normal, centroid in zip(parent_normals, parent_centroids):
        arm = _unit(np.cross(_unit(normal), direction))
        if float(np.dot(arm, np.asarray(centroid) - origin)) < 0.0:
            arm = -arm
        arms.append(arm)
    basis_v = _unit(np.cross(direction, arms[0]))
    radius = min(0.30 * length, 80.0 * scale)
    arm_angle = math.degrees(math.acos(float(np.clip(np.dot(arms[0], arms[1]), -1.0, 1.0))))
    return CandidateFrame(origin, direction, arms[0], arms[1], basis_v, length, scale, radius, arm_angle)


def _port(point: np.ndarray, arm: np.ndarray, radius: float) -> bool:
    transverse = np.asarray((point[1], point[2]), dtype=np.float64)
    arm_transverse = np.asarray((arm[1], arm[2]), dtype=np.float64)
    radial = float(np.linalg.norm(transverse))
    if radial < 5.0 * radius / 8.0 or radial > radius or radial <= 0.0:
        return False
    return float(np.dot(arm_transverse, transverse)) / radial >= COS_PORT


def _axial_bin(x: float, length: float) -> int:
    position = (x + 0.40 * length) / (0.80 * length)
    return max(0, min(7, int(math.floor(position * 8.0))))


def _clip_fans(vertices: np.ndarray, triangles: np.ndarray, frame: CandidateFrame) -> list[FanTriangle]:
    quantum = frame.scale / 1024.0
    epsilon = quantum / 16.0
    apothem = frame.radius * math.cos(math.pi / 16.0)
    halfspaces: list[tuple[np.ndarray, float]] = [
        (np.asarray((1.0, 0.0, 0.0)), 0.40 * frame.length),
        (np.asarray((-1.0, 0.0, 0.0)), 0.40 * frame.length),
    ]
    for index in range(16):
        angle = index * math.pi / 8.0
        halfspaces.append((np.asarray((0.0, math.cos(angle), math.sin(angle))), apothem))
    fans: list[FanTriangle] = []
    arm_a, arm_b = frame.arm_a_local, frame.arm_b_local
    source = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    for face in faces:
        world = source[face]
        polygon = [frame.local(point) for point in world]
        for normal, bound in halfspaces:
            polygon = _clip_polygon(polygon, normal, bound, epsilon)
            if len(polygon) < 3:
                break
        polygon = _canonical_polygon(polygon, quantum)
        if len(polygon) < 3:
            continue
        native_key = _canonical_triangle_key(np.asarray([frame.local(point) for point in world]), quantum)
        anchor = polygon[0]
        for index in range(1, len(polygon) - 1):
            fan_vertices = np.asarray((anchor, polygon[index], polygon[index + 1]), dtype=np.float64)
            cross = np.cross(fan_vertices[1] - fan_vertices[0], fan_vertices[2] - fan_vertices[0])
            area = 0.5 * float(np.linalg.norm(cross))
            if not math.isfinite(area) or area <= epsilon * epsilon:
                continue
            normal = _unit(cross)
            centroid = np.mean(fan_vertices, axis=0)
            patch_key = _canonical_triangle_key(fan_vertices / frame.scale, 1.0 / 1024.0)
            fans.append(FanTriangle(
                vertices=fan_vertices,
                area=area,
                centroid=centroid,
                normal=normal,
                native_key=native_key,
                patch_key=patch_key,
                axial_bin=_axial_bin(float(centroid[0]), frame.length),
                port_a=_port(centroid, arm_a, frame.radius),
                port_b=_port(centroid, arm_b, frame.radius),
                minimum=np.min(fan_vertices, axis=0),
                maximum=np.max(fan_vertices, axis=0),
            ))
            if len(fans) > MAX_FAN_TRIANGLES:
                raise PtcpStructuralCapError("clipped fan triangle cap exceeded")
    fans.sort(key=lambda fan: fan.patch_key)
    return fans


def _initial_components(fans: Sequence[FanTriangle], frame: CandidateFrame) -> list[Component]:
    count = len(fans)
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if fans[right].patch_key < fans[left].patch_key:
            left, right = right, left
        parent[right] = left

    native_groups: dict[bytes, int] = {}
    vertex_groups: dict[tuple[int, ...], int] = {}
    zero_quantum = frame.scale / (1024.0 * 1024.0)
    for fan_id, fan in enumerate(fans):
        previous = native_groups.setdefault(fan.native_key, fan_id)
        union(fan_id, previous)
        for vertex in fan.vertices:
            key = _quantized_tuple(vertex, zero_quantum)
            previous = vertex_groups.setdefault(key, fan_id)
            union(fan_id, previous)
    groups: dict[int, list[int]] = {}
    for fan_id in range(count):
        groups.setdefault(find(fan_id), []).append(fan_id)
    if len(groups) > MAX_INITIAL_COMPONENTS:
        raise PtcpStructuralCapError("initial component cap exceeded")
    components: list[Component] = []
    for fan_ids in groups.values():
        fan_ids = sorted(fan_ids, key=lambda index: fans[index].patch_key)
        area = math.fsum(fans[index].area for index in fan_ids)
        port_a_area = np.zeros(8, dtype=np.float64)
        port_b_area = np.zeros(8, dtype=np.float64)
        for index in fan_ids:
            fan = fans[index]
            if fan.port_a:
                port_a_area[fan.axial_bin] += fan.area
            if fan.port_b:
                port_b_area[fan.axial_bin] += fan.area
        key = _hash128(b"".join(fans[index].patch_key for index in fan_ids))
        components.append(Component(
            key=key,
            fan_ids=tuple(fan_ids),
            area=area,
            port_a_area=port_a_area,
            port_b_area=port_b_area,
            axial_min=min(float(fans[index].minimum[0]) for index in fan_ids),
            axial_max=max(float(fans[index].maximum[0]) for index in fan_ids),
            minimum=np.min(np.vstack([fans[index].minimum for index in fan_ids]), axis=0),
            maximum=np.max(np.vstack([fans[index].maximum for index in fan_ids]), axis=0),
        ))
    components.sort(key=lambda component: component.key)
    return components


def _build_bvh(fans: Sequence[FanTriangle], fan_ids: Sequence[int]) -> _BvhNode:
    ids = tuple(fan_ids)
    minimum = np.min(np.vstack([fans[index].minimum for index in ids]), axis=0)
    maximum = np.max(np.vstack([fans[index].maximum for index in ids]), axis=0)
    if len(ids) <= 8:
        return _BvhNode(minimum, maximum, ids)
    centers = np.vstack([fans[index].centroid for index in ids])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = sorted(ids, key=lambda index: (float(fans[index].centroid[axis]), fans[index].patch_key))
    middle = len(ordered) // 2
    return _BvhNode(minimum, maximum, left=_build_bvh(fans, ordered[:middle]), right=_build_bvh(fans, ordered[middle:]))


def _component_closest(
    fans: Sequence[FanTriangle], left: _BvhNode, right: _BvhNode, maximum: float,
) -> tuple[float, np.ndarray, np.ndarray, int, int] | None:
    counter = 0
    heap: list[tuple[float, int, _BvhNode, _BvhNode]] = []
    heapq.heappush(heap, (_aabb_distance(left.minimum, left.maximum, right.minimum, right.maximum), counter, left, right))
    best = maximum
    result: tuple[float, np.ndarray, np.ndarray, int, int] | None = None
    while heap:
        bound, _order, node_a, node_b = heapq.heappop(heap)
        if bound > best:
            break
        if node_a.leaf and node_b.leaf:
            for fan_a in node_a.fan_ids:
                for fan_b in node_b.fan_ids:
                    distance, point_a, point_b = _triangle_triangle_closest(fans[fan_a].vertices, fans[fan_b].vertices)
                    candidate_key = (distance, fans[fan_a].patch_key, fans[fan_b].patch_key)
                    current_key = (result[0], fans[result[3]].patch_key, fans[result[4]].patch_key) if result is not None else None
                    if distance <= maximum and (current_key is None or candidate_key < current_key):
                        best = distance
                        result = (distance, point_a, point_b, fan_a, fan_b)
            continue
        children_a = (node_a,) if node_a.leaf else (node_a.left, node_a.right)
        children_b = (node_b,) if node_b.leaf else (node_b.left, node_b.right)
        for child_a in children_a:
            for child_b in children_b:
                assert child_a is not None and child_b is not None
                child_bound = _aabb_distance(child_a.minimum, child_a.maximum, child_b.minimum, child_b.maximum)
                if child_bound <= best:
                    counter += 1
                    heapq.heappush(heap, (child_bound, counter, child_a, child_b))
    return result


def _component_pairs(components: Sequence[Component], maximum: float) -> list[tuple[int, int]]:
    if maximum <= 0.0:
        return []
    cell = maximum
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, component in enumerate(components):
        lower = np.floor((component.minimum - maximum) / cell).astype(np.int64)
        upper = np.floor((component.maximum + maximum) / cell).astype(np.int64)
        for x in range(int(lower[0]), int(upper[0]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                for z in range(int(lower[2]), int(upper[2]) + 1):
                    buckets.setdefault((x, y, z), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        unique = sorted(set(members))
        for position, left in enumerate(unique):
            for right in unique[position + 1:]:
                if _aabb_distance(components[left].minimum, components[left].maximum, components[right].minimum, components[right].maximum) <= maximum:
                    pairs.add((left, right))
    return sorted(pairs, key=lambda pair: (components[pair[0]].key, components[pair[1]].key))


def _proximity_edges(fans: Sequence[FanTriangle], components: Sequence[Component], frame: CandidateFrame) -> list[ProximityEdge]:
    maximum = frame.radius / 4.0
    delta = frame.scale / 1024.0
    bvhs = [_build_bvh(fans, component.fan_ids) for component in components]
    edges: list[ProximityEdge] = []
    for left, right in _component_pairs(components, maximum):
        closest = _component_closest(fans, bvhs[left], bvhs[right], maximum)
        if closest is None:
            continue
        distance, point_a, point_b, fan_a, fan_b = closest
        if distance <= 0.0:
            continue
        q = int(math.ceil(math.nextafter(distance, math.inf) / delta))
        if q * delta > maximum:
            continue
        edges.append(ProximityEdge(
            q=q, distance=distance, left=left, right=right,
            left_point=point_a, right_point=point_b,
            left_normal=fans[fan_a].normal, right_normal=fans[fan_b].normal,
            left_fan_key=fans[fan_a].patch_key, right_fan_key=fans[fan_b].patch_key,
        ))
        if len(edges) > MAX_PROXIMITY_EDGES:
            raise PtcpStructuralCapError("proximity edge cap exceeded")
    edges.sort(key=lambda edge: (edge.q, components[edge.left].key, components[edge.right].key))
    return edges


def _mask(area: np.ndarray, minimum: float) -> int:
    result = 0
    for index, value in enumerate(area):
        if float(value) >= minimum:
            result |= 1 << index
    return result


def _span(mask: int) -> int:
    if mask == 0:
        return 0
    low = (mask & -mask).bit_length() - 1
    high = mask.bit_length() - 1
    return high - low + 1


def _spanning(uf: _UnionFind, root: int, scale: float) -> bool:
    mask_a = _mask(uf.port_a[root], 2.0 * scale * scale)
    mask_b = _mask(uf.port_b[root], 2.0 * scale * scale)
    return mask_a.bit_count() >= 4 and _span(mask_a) >= 6 and mask_b.bit_count() >= 4 and _span(mask_b) >= 6


def _third_sheet_ok(uf: _UnionFind, chosen: tuple[int, int]) -> tuple[bool, float]:
    roots = uf.roots()
    total_a = math.fsum(float(np.sum(uf.port_a[root])) for root in roots)
    total_b = math.fsum(float(np.sum(uf.port_b[root])) for root in roots)
    worst = 0.0
    for root in roots:
        if root in chosen:
            continue
        fraction_a = float(np.sum(uf.port_a[root])) / total_a if total_a > 0.0 else 0.0
        fraction_b = float(np.sum(uf.port_b[root])) / total_b if total_b > 0.0 else 0.0
        worst = max(worst, min(fraction_a, fraction_b))
        if fraction_a >= 0.10 and fraction_b >= 0.10:
            return False, max(0.0, 1.0 - worst)
    return True, max(0.0, 1.0 - worst)


def _edge_masks(evidence: IntervalEvidence, edges: Sequence[ProximityEdge], frame: CandidateFrame) -> None:
    left_members, right_members = evidence.predeath_roots
    death_gap = evidence.death_q * frame.scale / 1024.0
    valid = 0
    considered = 0
    axial = a_cross = b_cross = offset = 0
    arm_a, arm_b = frame.arm_a_local, frame.arm_b_local
    for edge in edges:
        crosses = (edge.left in left_members and edge.right in right_members) or (edge.right in left_members and edge.left in right_members)
        if not crosses or edge.distance > 1.25 * death_gap:
            continue
        considered += 1
        vector = edge.right_point - edge.left_point
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            continue
        unit = vector / length
        if (
            abs(float(np.dot(edge.left_normal, edge.right_normal))) < COS_NORMAL
            or min(abs(float(np.dot(edge.left_normal, unit))), abs(float(np.dot(edge.right_normal, unit)))) < COS_OFFSET
        ):
            continue
        valid += 1
        midpoint = 0.5 * (edge.left_point + edge.right_point)
        bit = 1 << _axial_bin(float(midpoint[0]), frame.length)
        axial |= bit
        offset |= bit
        if _port(midpoint, arm_a, frame.radius):
            a_cross |= bit
        if _port(midpoint, arm_b, frame.radius):
            b_cross |= bit
    evidence.axial_mask = axial
    evidence.a_cross_mask = a_cross
    evidence.b_cross_mask = b_cross
    evidence.offset_valid_mask = offset
    evidence.offset_valid_fraction = valid / considered if considered else 0.0


def _filtration(
    components: Sequence[Component], edges: Sequence[ProximityEdge], frame: CandidateFrame,
) -> tuple[list[IntervalEvidence], bytes]:
    uf = _UnionFind(components)
    intervals: list[IntervalEvidence] = []
    current: dict[str, Any] | None = None

    def maybe_start(q: int) -> dict[str, Any] | None:
        spanning = [root for root in uf.roots() if _spanning(uf, root, frame.scale)]
        if len(spanning) != 2:
            return None
        spanning = sorted(spanning, key=lambda root: uf.key[root])
        third_ok, exclusivity = _third_sheet_ok(uf, (spanning[0], spanning[1]))
        return {
            "birth_q": q,
            "roots": tuple(spanning),
            "birth_members": tuple(frozenset(uf.members[root]) for root in spanning),
            "third_ok": third_ok,
            "exclusivity": exclusivity,
        }

    current = maybe_start(0)
    cursor = 0
    while cursor < len(edges):
        q = edges[cursor].q
        end = cursor
        while end < len(edges) and edges[end].q == q:
            end += 1
        batch = edges[cursor:end]
        pre_roots = [(uf.find(edge.left), uf.find(edge.right)) for edge in batch]
        death_edges: list[ProximityEdge] = []
        if current is not None:
            left_root, right_root = (uf.find(current["roots"][0]), uf.find(current["roots"][1]))
            for edge, (edge_left, edge_right) in zip(batch, pre_roots):
                if {edge_left, edge_right} == {left_root, right_root}:
                    death_edges.append(edge)
            predeath = (frozenset(uf.members[left_root]), frozenset(uf.members[right_root]))
        else:
            predeath = (frozenset(), frozenset())
        for edge in batch:
            uf.union(edge.left, edge.right)
        if current is not None and death_edges:
            evidence = IntervalEvidence(
                birth_q=current["birth_q"], death_q=q,
                birth_roots=current["birth_members"], predeath_roots=predeath,
                third_sheet_ok=current["third_ok"], direct_edges=tuple(death_edges),
                port_exclusivity_fraction=current["exclusivity"],
            )
            _edge_masks(evidence, edges, frame)
            birth_gap, death_gap = evidence.birth_q * frame.scale / 1024.0, q * frame.scale / 1024.0
            persistence = 1.0 - birth_gap / death_gap if death_gap > 0.0 else 0.0
            evidence.qualifying = bool(
                3.0 * frame.scale <= death_gap <= frame.radius / 5.0
                and birth_gap <= death_gap / 4.0
                and death_gap - birth_gap >= 2.0 * frame.scale
                and persistence >= 0.75
                and evidence.third_sheet_ok
                and evidence.axial_mask.bit_count() >= 5
                and evidence.a_cross_mask.bit_count() >= 3
                and evidence.b_cross_mask.bit_count() >= 3
            )
            intervals.append(evidence)
            current = None
        elif current is not None:
            old_left, old_right = current["roots"]
            new_left, new_right = uf.find(old_left), uf.find(old_right)
            spanning = [root for root in uf.roots() if _spanning(uf, root, frame.scale)]
            if new_left == new_right or len(spanning) != 2 or set(spanning) != {new_left, new_right}:
                current = None
            else:
                current["roots"] = (new_left, new_right)
        if current is None:
            current = maybe_start(q)
        cursor = end
    topology_payload = b"".join(
        struct.pack("<I", edge.q) + components[edge.left].key + components[edge.right].key
        for edge in edges
    )
    return intervals, _hash128(topology_payload)


def analyze_candidate(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    row: Mapping[str, Any],
    scale: float,
    parent_normals: tuple[np.ndarray, np.ndarray],
    parent_centroids: tuple[np.ndarray, np.ndarray],
    source_rank: int = 0,
) -> PtcpRecord:
    candidate_id = str(row.get("candidate_id") or row.get("bend_id") or "")
    status = 0
    try:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("invalid scale")
        frame = _build_frame(row, scale, parent_normals, parent_centroids)
    except (TypeError, ValueError, FloatingPointError):
        return PtcpRecord(candidate_id, bytes(16), source_rank, FAIL_INVALID_CANDIDATE)
    endpoints = _endpoints(row)
    canonical_key = _hash128(repr(_quantized_tuple(np.vstack([frame.local(point) for point in endpoints]) / scale, 1.0 / 1024.0)).encode("ascii"))
    record = PtcpRecord(
        candidate_id=candidate_id, canonical_key=canonical_key, source_rank=source_rank,
        status_bits=0, native_triangle_count=len(triangles), scale=scale,
        radius=frame.radius, length=frame.length, arm_angle_deg=frame.arm_angle_deg,
    )
    if frame.radius < 20.0 * scale:
        record.status_bits = FAIL_TUBE_TOO_SMALL
        return record
    try:
        fans = _clip_fans(np.asarray(vertices), np.asarray(triangles), frame)
        components = _initial_components(fans, frame)
        edges = _proximity_edges(fans, components, frame)
        estimated = (
            sum(fan.vertices.nbytes + 256 for fan in fans)
            + len(components) * 512
            + len(edges) * 320
        )
        if estimated > MAX_TRANSIENT_BYTES:
            raise PtcpStructuralCapError("transient allocation bound exceeded")
        intervals, topology_hash = _filtration(components, edges, frame)
    except PtcpStructuralCapError as error:
        record.status_bits = FAIL_STRUCTURAL_CAP
        record.debug = {"error": str(error)}
        return record
    except (FloatingPointError, OverflowError, ValueError) as error:
        record.status_bits = FAIL_NONFINITE
        record.debug = {"error": str(error)}
        return record
    record.fan_count = len(fans)
    record.component_count = len(components)
    record.edge_count = len(edges)
    record.topology_hash = topology_hash
    qualifying = [interval for interval in intervals if interval.qualifying]
    if len(qualifying) != 1:
        record.status_bits = FAIL_NO_UNIQUE_INTERVAL
        record.debug = {
            "interval_count": len(intervals),
            "qualifying_interval_count": len(qualifying),
            "intervals": [
                {
                    "birth_q": item.birth_q, "death_q": item.death_q,
                    "third_sheet_ok": item.third_sheet_ok,
                    "axial_mask": item.axial_mask,
                    "a_cross_mask": item.a_cross_mask,
                    "b_cross_mask": item.b_cross_mask,
                    "qualifying": item.qualifying,
                }
                for item in intervals
            ],
        }
        return record
    evidence = qualifying[0]
    birth_gap = evidence.birth_q * scale / 1024.0
    death_gap = evidence.death_q * scale / 1024.0
    persistence = 1.0 - birth_gap / death_gap
    record.status_bits = STATUS_RETAIN
    record.birth_q, record.death_q = evidence.birth_q, evidence.death_q
    record.birth_gap, record.death_gap = birth_gap, death_gap
    record.persistence = persistence
    record.death_radius_ratio = death_gap / frame.radius
    record.offset_valid_fraction = evidence.offset_valid_fraction
    record.port_exclusivity_fraction = evidence.port_exclusivity_fraction
    left_members, right_members = evidence.predeath_roots
    record.root_ids = (
        int.from_bytes(_hash128(b"".join(components[index].key for index in sorted(left_members)))[:4], "little"),
        int.from_bytes(_hash128(b"".join(components[index].key for index in sorted(right_members)))[:4], "little"),
    )
    left_area = math.fsum(components[index].area for index in left_members)
    right_area = math.fsum(components[index].area for index in right_members)
    left_a = math.fsum(float(np.sum(components[index].port_a_area)) for index in left_members)
    left_b = math.fsum(float(np.sum(components[index].port_b_area)) for index in left_members)
    right_a = math.fsum(float(np.sum(components[index].port_a_area)) for index in right_members)
    right_b = math.fsum(float(np.sum(components[index].port_b_area)) for index in right_members)
    normalizer_area = scale * scale
    record.component_summary = (
        left_area / normalizer_area, right_area / normalizer_area,
        left_a / normalizer_area, left_b / normalizer_area,
        right_a / normalizer_area, right_b / normalizer_area,
        min((components[index].axial_min for index in left_members), default=0.0) / scale,
        max((components[index].axial_max for index in right_members), default=0.0) / scale,
    )
    record.masks = (
        evidence.axial_mask, evidence.a_cross_mask, evidence.b_cross_mask,
        evidence.offset_valid_mask, 0, 0, 0, 0,
    )
    death_keys: list[int] = []
    for edge in evidence.direct_edges[:2]:
        death_keys.extend((int.from_bytes(edge.left_fan_key[:4], "little"), int.from_bytes(edge.right_fan_key[:4], "little")))
    record.death_patch_ids = tuple(death_keys[:4])
    record.debug = {
        "birth_root_component_ids": [sorted(left_members), sorted(right_members)],
        "axial_mask": evidence.axial_mask,
        "a_cross_mask": evidence.a_cross_mask,
        "b_cross_mask": evidence.b_cross_mask,
        "offset_valid_mask": evidence.offset_valid_mask,
        "direct_death_edge_count": len(evidence.direct_edges),
    }
    return record


def analyze_context_candidate(context: Any, row: Mapping[str, Any], *, source_rank: int = 0) -> PtcpRecord:
    try:
        owners = (int(row["surface1_id"]), int(row["surface2_id"]))
        surfaces = (context.surfaces[owners[0]], context.surfaces[owners[1]])
    except (KeyError, TypeError, ValueError):
        candidate_id = str(row.get("candidate_id") or "")
        return PtcpRecord(candidate_id, bytes(16), source_rank, FAIL_INVALID_PARENTS)
    scale = float(row.get("mesh_spacing") or row.get("native_mesh_spacing") or context.mesh_scale)
    return analyze_candidate(
        vertices=context.vertices,
        triangles=context.triangles,
        row=row,
        scale=scale,
        parent_normals=(np.asarray(surfaces[0].normal), np.asarray(surfaces[1].normal)),
        parent_centroids=(np.asarray(surfaces[0].centroid), np.asarray(surfaces[1].centroid)),
        source_rank=source_rank,
    )


def rank_key(record: PtcpRecord) -> tuple[Any, ...]:
    masks = record.masks
    return (
        -record.persistence,
        -int(masks[0]).bit_count(),
        -(int(masks[1]).bit_count() + int(masks[2]).bit_count()),
        record.death_radius_ratio,
        record.source_rank,
        record.canonical_key,
    )


def retained_in_rank_order(records: Iterable[PtcpRecord]) -> list[PtcpRecord]:
    return sorted((record for record in records if record.retain), key=rank_key)
