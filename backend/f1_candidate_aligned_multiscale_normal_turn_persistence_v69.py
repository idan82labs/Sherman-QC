"""V69 candidate-aligned multiscale normal-turn persistence (CAMNTP_V1).

Research-only, source-only, candidate-local and deletion-only.  The mechanism
intersects the canonical triangle mesh with seventeen planes normal to each
thresholded finite segment.  A candidate survives only when a localized,
bilaterally-supported accumulated normal turn persists across three fixed
relative scales and along a sufficiently long run of stations.

This module intentionally does not inspect candidate scores, ranks, routes,
part identity, filenames, labels, CAD, expected counts, or other candidates.
It returns surviving input objects unchanged and records only compact integer
summaries.  It is inserted before the caller's one ordinary NMS invocation.
"""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ARCHITECTURE = "V69_CANDIDATE_ALIGNED_MULTISCALE_NORMAL_TURN_PERSISTENCE_V1"
SHORT_NAME = "CAMNTP_V1"
SCHEMA = "sherman_qc_f1_camntp_v1_decision_record"
FIXTURE_TOKEN = "V69_CAMNTP_V1_FROZEN_SOURCE_FREE_FIXTURES"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests/test_f1_candidate_aligned_multiscale_normal_turn_persistence_v69.py"

MAX_VERTICES = 200_000
MAX_TRIANGLES = 350_000
MAX_CANDIDATES = 128
STATION_COUNT = 17
SCALE_COUNT = 15
MAX_INTERSECTING_TRIANGLES_PER_STATION = 20_000
MAX_SEGMENTS_PER_STATION = 20_000
MAX_COMPONENTS_PER_STATION = 512
MAX_SEGMENTS_PER_CANDIDATE = 120_000
MAX_DISCONTINUITIES_PER_CANDIDATE = 240_000
MAX_EXACT_TESTS_PER_SCAN = 8_000_000
MAX_SEGMENTS_PER_SCAN = 2_000_000
MAX_DISCONTINUITIES_PER_SCAN = 4_000_000
MAX_AUDIT_BYTES = 256 * 1024
MAX_ADDED_SECONDS = 120.0
MAX_ADDED_RSS_BYTES = 4 * 1024**3
MAX_TOTAL_RSS_BYTES = 16 * 1024**3

ANGLE_MIN = math.radians(8.0)
ANGLE_SPREAD_ABS = math.radians(4.0)
SHELF_DRIFT_MAX = math.radians(12.0)
PROJECTED_NORMAL_MIN = math.cos(math.radians(35.0))
SCALE_RATIOS = tuple(2.0 ** (j / 2.0 - 9.0) for j in range(SCALE_COUNT))


def _ge(value: float, threshold: float) -> bool:
    """Inclusive geometric comparator with a bounded filtered-float margin."""

    return float(value) >= float(threshold) - 64.0 * math.ulp(float(threshold))


def _le(value: float, threshold: float) -> bool:
    """Inclusive geometric comparator with a bounded filtered-float margin."""

    return float(value) <= float(threshold) + 64.0 * math.ulp(float(threshold))


class CAMNTPInputError(RuntimeError):
    """Malformed input requiring an atomic scan-level abort."""


class CAMNTPResourceAbort(RuntimeError):
    """Frozen operational cap exceeded; no V69 mutation may be committed."""


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.floating, float)):
        raise CAMNTPInputError("FLOAT_IN_DETERMINISTIC_AUDIT")
    raise CAMNTPInputError("UNSUPPORTED_AUDIT_VALUE")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _native(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _q_angle(value: float) -> int:
    return int(round(math.degrees(float(value)) * 1_000_000.0))


def _q_ppm(value: float) -> int:
    if not math.isfinite(value):
        raise CAMNTPInputError("NONFINITE_DERIVED_SCALAR")
    return int(round(float(value) * 1_000_000.0))


def _q_ppb(value: float) -> int:
    if not math.isfinite(value):
        raise CAMNTPInputError("NONFINITE_DERIVED_SCALAR")
    return int(round(float(value) * 1_000_000_000.0))


def _clean_coordinates(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise CAMNTPInputError("NONFINITE_SOURCE_COORDINATE")
    result[result == 0.0] = 0.0
    return result


def _coordinate_bytes(point: np.ndarray) -> bytes:
    return _clean_coordinates(np.asarray(point, dtype=np.float64)).astype(">f8", copy=False).tobytes()


def _coordinate_hash(point: np.ndarray) -> str:
    return hashlib.sha256(_coordinate_bytes(point)).hexdigest()


def _candidate_opaque_hash(row: Mapping[str, Any], start: np.ndarray, end: np.ndarray) -> str:
    supplied = row.get("route_hash") or row.get("candidate_opaque_hash") or row.get("bend_id")
    if supplied is not None:
        return hashlib.sha256(str(supplied).encode("utf-8")).hexdigest()
    endpoints = sorted((_coordinate_bytes(start), _coordinate_bytes(end)))
    return hashlib.sha256(endpoints[0] + endpoints[1]).hexdigest()


@dataclass(frozen=True)
class Candidate:
    opaque_hash: str
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    original_ordinal: int
    original: Mapping[str, Any]


def _candidate(row: Mapping[str, Any], ordinal: int) -> Candidate:
    try:
        start = _clean_coordinates(np.asarray(row["line_start"], dtype=np.float64))
        end = _clean_coordinates(np.asarray(row["line_end"], dtype=np.float64))
    except (KeyError, TypeError, ValueError) as error:
        raise CAMNTPInputError("ABORT_MALFORMED_CANDIDATE_STREAM") from error
    if start.shape != (3,) or end.shape != (3,):
        raise CAMNTPInputError("ABORT_MALFORMED_CANDIDATE_STREAM")
    # Canonical endpoint order makes station masks invariant to endpoint reversal.
    if _coordinate_bytes(end) < _coordinate_bytes(start):
        start, end = end.copy(), start.copy()
    length = float(np.linalg.norm(end - start))
    if not math.isfinite(length) or length <= 0.0:
        raise CAMNTPInputError("ABORT_MALFORMED_CANDIDATE_STREAM")
    direction = (end - start) / length
    return Candidate(_candidate_opaque_hash(row, start, end), start, end, direction, length, ordinal, row)


@dataclass
class Budget:
    exact_tests: int = 0
    segments: int = 0
    discontinuities: int = 0
    bvh_node_visits: int = 0
    candidate_segments: int = 0
    candidate_discontinuities: int = 0

    def begin_candidate(self) -> None:
        self.candidate_segments = 0
        self.candidate_discontinuities = 0

    def add_tests(self, count: int) -> None:
        self.exact_tests += int(count)
        if self.exact_tests > MAX_EXACT_TESTS_PER_SCAN:
            raise CAMNTPResourceAbort("EXACT_TRIANGLE_PLANE_TEST_CAP")

    def add_segments(self, count: int) -> None:
        count = int(count)
        self.segments += count
        self.candidate_segments += count
        if self.segments > MAX_SEGMENTS_PER_SCAN:
            raise CAMNTPResourceAbort("SCAN_SEGMENT_CAP")
        if self.candidate_segments > MAX_SEGMENTS_PER_CANDIDATE:
            raise CAMNTPResourceAbort("CANDIDATE_SEGMENT_CAP")

    def add_discontinuities(self, count: int) -> None:
        count = int(count)
        self.discontinuities += count
        self.candidate_discontinuities += count
        if self.discontinuities > MAX_DISCONTINUITIES_PER_SCAN:
            raise CAMNTPResourceAbort("SCAN_DISCONTINUITY_CAP")
        if self.candidate_discontinuities > MAX_DISCONTINUITIES_PER_CANDIDATE:
            raise CAMNTPResourceAbort("CANDIDATE_DISCONTINUITY_CAP")


@dataclass(frozen=True)
class _BVHNode:
    low: np.ndarray
    high: np.ndarray
    left: int
    right: int
    indices: tuple[int, ...]


class AABBBVH:
    """Deterministic AABB hierarchy used only as a conservative accelerator."""

    def __init__(self, lows: np.ndarray, highs: np.ndarray, *, leaf_size: int = 32):
        self.lows = np.asarray(lows, dtype=np.float64)
        self.highs = np.asarray(highs, dtype=np.float64)
        self.centers = 0.5 * (self.lows + self.highs)
        self.leaf_size = int(leaf_size)
        self.nodes: list[_BVHNode] = []
        self.root = self._build(tuple(range(len(self.lows)))) if len(self.lows) else -1

    def _build(self, indices: tuple[int, ...]) -> int:
        array = np.asarray(indices, dtype=np.int64)
        low = np.min(self.lows[array], axis=0)
        high = np.max(self.highs[array], axis=0)
        node_id = len(self.nodes)
        self.nodes.append(_BVHNode(low, high, -1, -1, ()))
        if len(indices) <= self.leaf_size:
            self.nodes[node_id] = _BVHNode(low, high, -1, -1, tuple(sorted(indices)))
            return node_id
        span = np.ptp(self.centers[array], axis=0)
        axis = int(np.argmax(span))
        ordered = tuple(sorted(indices, key=lambda index: (float(self.centers[index, axis]), index)))
        middle = len(ordered) // 2
        left = self._build(ordered[:middle])
        right = self._build(ordered[middle:])
        self.nodes[node_id] = _BVHNode(low, high, left, right, ())
        return node_id

    @staticmethod
    def _box_sphere(low: np.ndarray, high: np.ndarray, center: np.ndarray, radius: float) -> bool:
        closest = np.minimum(np.maximum(center, low), high)
        return float(np.dot(closest - center, closest - center)) <= radius * radius

    @staticmethod
    def _box_plane(low: np.ndarray, high: np.ndarray, point: np.ndarray, normal: np.ndarray) -> bool:
        center = 0.5 * (low + high)
        extent = 0.5 * (high - low)
        radius = float(np.dot(np.abs(normal), extent))
        distance = abs(float(np.dot(normal, center - point)))
        return distance <= radius

    def query_plane_ball(self, point: np.ndarray, normal: np.ndarray, radius: float, budget: Budget) -> list[int]:
        if self.root < 0:
            return []
        output: list[int] = []
        stack = [self.root]
        while stack:
            node_id = stack.pop()
            budget.bvh_node_visits += 1
            node = self.nodes[node_id]
            if not self._box_sphere(node.low, node.high, point, radius):
                continue
            if not self._box_plane(node.low, node.high, point, normal):
                continue
            if node.indices:
                output.extend(node.indices)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return sorted(set(output))

    def nearest_segment_distance(self, point: np.ndarray, starts: np.ndarray, ends: np.ndarray, budget: Budget) -> float:
        if self.root < 0:
            return math.inf

        def box_distance2(node: _BVHNode) -> float:
            delta = np.maximum(np.maximum(node.low - point, point - node.high), 0.0)
            return float(np.dot(delta, delta))

        queue: list[tuple[float, int]] = [(box_distance2(self.nodes[self.root]), self.root)]
        best2 = math.inf
        while queue:
            lower, node_id = heapq.heappop(queue)
            if lower > best2:
                break
            budget.bvh_node_visits += 1
            node = self.nodes[node_id]
            if node.indices:
                for index in node.indices:
                    delta = ends[index] - starts[index]
                    denominator = float(np.dot(delta, delta))
                    fraction = 0.0 if denominator <= 0.0 else float(np.clip(np.dot(point - starts[index], delta) / denominator, 0.0, 1.0))
                    closest = starts[index] + fraction * delta
                    best2 = min(best2, float(np.dot(point - closest, point - closest)))
            else:
                for child in (node.left, node.right):
                    child_lower = box_distance2(self.nodes[child])
                    if child_lower <= best2:
                        heapq.heappush(queue, (child_lower, child))
        return math.sqrt(best2)


@dataclass(frozen=True)
class ComponentTopology:
    component_id: int
    face_ids: tuple[int, ...]
    valid_orientation: bool
    boundary_edges: tuple[tuple[int, int], ...]


class CanonicalMesh:
    """Exact-coordinate welded, duplicate-free, consistently oriented mesh."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray | None):
        raw = _clean_coordinates(np.asarray(vertices, dtype=np.float64))
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise CAMNTPInputError("INVALID_SOURCE_VERTICES")
        if len(raw) > MAX_VERTICES:
            raise CAMNTPResourceAbort("SOURCE_VERTEX_CAP")
        if faces is None or len(np.asarray(faces)) == 0:
            raise CAMNTPResourceAbort("ABSTAIN_NO_MESH")
        raw_faces = np.asarray(faces, dtype=np.int64)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise CAMNTPInputError("INVALID_SOURCE_FACES")
        if len(raw_faces) > MAX_TRIANGLES:
            raise CAMNTPResourceAbort("SOURCE_TRIANGLE_CAP")
        if not np.all((raw_faces >= 0) & (raw_faces < len(raw))):
            raise CAMNTPInputError("INVALID_SOURCE_FACES")

        unique, inverse = np.unique(raw, axis=0, return_inverse=True)
        remapped = inverse[raw_faces]
        geometric: set[tuple[int, int, int]] = set()
        for face in remapped:
            key = tuple(sorted(int(value) for value in face))
            if len(set(key)) != 3 or key in geometric:
                continue
            a, b, c = (unique[index] for index in key)
            if np.array_equal(np.cross(b - a, c - a), np.zeros(3, dtype=np.float64)):
                continue
            geometric.add(key)
        ordered_faces = sorted(geometric)
        self.vertices = np.asarray(unique, dtype=np.float64)
        self.faces = np.asarray(ordered_faces, dtype=np.int64)
        self.vertex_hashes = tuple(_coordinate_hash(point) for point in self.vertices)
        self.face_keys = tuple(hashlib.sha256("".join(self.vertex_hashes[index] for index in face).encode("ascii")).hexdigest() for face in self.faces)

        incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for face_id, face in enumerate(self.faces):
            ids = [int(value) for value in face]
            for first, second in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                key = (min(first, second), max(first, second))
                direction = 1 if (first, second) == key else -1
                incidence.setdefault(key, []).append((face_id, direction))
        self.edge_incidence = incidence
        self.boundary_edge_count = sum(len(rows) == 1 for rows in incidence.values())
        self.nonmanifold_edge_count = sum(len(rows) > 2 for rows in incidence.values())
        global_nonmanifold = sorted(edge for edge, rows in incidence.items() if len(rows) > 2)
        global_nm_starts = np.asarray([unique[first] for first, _ in global_nonmanifold], dtype=np.float64).reshape((-1, 3))
        global_nm_ends = np.asarray([unique[second] for _, second in global_nonmanifold], dtype=np.float64).reshape((-1, 3))
        self.global_nonmanifold_geometry = (
            global_nm_starts,
            global_nm_ends,
            AABBBVH(np.minimum(global_nm_starts, global_nm_ends), np.maximum(global_nm_starts, global_nm_ends), leaf_size=16),
        )

        adjacency: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(self.faces))}
        for rows in incidence.values():
            if len(rows) != 2:
                continue
            (first, first_dir), (second, second_dir) = rows
            relation = -first_dir * second_dir
            adjacency[first].append((second, relation))
            adjacency[second].append((first, relation))
        signs: dict[int, int] = {}
        raw_components: list[tuple[list[int], bool]] = []
        for seed in range(len(self.faces)):
            if seed in signs:
                continue
            signs[seed] = 1
            queue = deque([seed])
            members: list[int] = []
            valid = True
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor, relation in sorted(adjacency[current]):
                    proposed = signs[current] * relation
                    if neighbor not in signs:
                        signs[neighbor] = proposed
                        queue.append(neighbor)
                    elif signs[neighbor] != proposed:
                        valid = False
            raw_components.append((sorted(members), valid))
        raw_components.sort(key=lambda row: tuple(self.face_keys[index] for index in row[0]))
        self.face_component = np.full(len(self.faces), -1, dtype=np.int64)
        self.face_orientation = np.ones(len(self.faces), dtype=np.int8)
        components: list[ComponentTopology] = []
        for component_id, (members, valid) in enumerate(raw_components):
            for face_id in members:
                self.face_component[face_id] = component_id
                self.face_orientation[face_id] = signs[face_id]
            boundary = []
            member_set = set(members)
            for edge, rows in incidence.items():
                if len(rows) != 2 and any(face_id in member_set for face_id, _direction in rows):
                    boundary.append(edge)
            components.append(ComponentTopology(component_id, tuple(members), valid, tuple(sorted(boundary))))
        self.components = tuple(components)
        normals = []
        for face_id, face in enumerate(self.faces):
            a, b, c = self.vertices[face]
            vector = np.cross(b - a, c - a) * int(self.face_orientation[face_id])
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0 or not math.isfinite(norm):
                raise CAMNTPInputError("INTERNAL_TOPOLOGY_INCONSISTENCY")
            normals.append(vector / norm)
        self.face_normals = np.asarray(normals, dtype=np.float64)
        triangle_points = self.vertices[self.faces]
        self.triangle_bvh = AABBBVH(np.min(triangle_points, axis=1), np.max(triangle_points, axis=1))
        self.boundary_geometry: dict[int, tuple[np.ndarray, np.ndarray, AABBBVH]] = {}
        self.nonmanifold_geometry: dict[int, tuple[np.ndarray, np.ndarray, AABBBVH]] = {}
        for component in self.components:
            starts = np.asarray([self.vertices[first] for first, _ in component.boundary_edges], dtype=np.float64).reshape((-1, 3))
            ends = np.asarray([self.vertices[second] for _, second in component.boundary_edges], dtype=np.float64).reshape((-1, 3))
            lows = np.minimum(starts, ends)
            highs = np.maximum(starts, ends)
            self.boundary_geometry[component.component_id] = (starts, ends, AABBBVH(lows, highs, leaf_size=16))
            nonmanifold = [edge for edge in component.boundary_edges if len(incidence[edge]) > 2]
            nm_starts = np.asarray([self.vertices[first] for first, _ in nonmanifold], dtype=np.float64).reshape((-1, 3))
            nm_ends = np.asarray([self.vertices[second] for _, second in nonmanifold], dtype=np.float64).reshape((-1, 3))
            self.nonmanifold_geometry[component.component_id] = (
                nm_starts,
                nm_ends,
                AABBBVH(np.minimum(nm_starts, nm_ends), np.maximum(nm_starts, nm_ends), leaf_size=16),
            )
        mesh_payload = {
            "vertices": sorted(self.vertex_hashes),
            "faces": sorted(self.face_keys),
        }
        self.opaque_hash = hashlib.sha256(canonical_bytes(mesh_payload)).hexdigest()

    def boundary_distance(self, component_id: int, point: np.ndarray, budget: Budget) -> float:
        starts, ends, bvh = self.boundary_geometry[component_id]
        return bvh.nearest_segment_distance(point, starts, ends, budget)

    def nonmanifold_distance(self, component_id: int, point: np.ndarray, budget: Budget) -> float:
        starts, ends, bvh = self.nonmanifold_geometry[component_id]
        return bvh.nearest_segment_distance(point, starts, ends, budget)

    def global_nonmanifold_distance(self, point: np.ndarray, budget: Budget) -> float:
        starts, ends, bvh = self.global_nonmanifold_geometry
        return bvh.nearest_segment_distance(point, starts, ends, budget)


@dataclass(frozen=True)
class SectionSegment:
    first_node: str
    second_node: str
    first_point: np.ndarray
    second_point: np.ndarray
    normal: np.ndarray
    face_id: int
    component_id: int


@dataclass(frozen=True)
class SectionComponent:
    node_ids: tuple[str, ...]
    points: tuple[np.ndarray, ...]
    normals: tuple[np.ndarray, ...]
    segment_face_ids: tuple[int, ...]
    component_id: int
    closed: bool
    arc: tuple[float, ...]
    length: float


def _exact_dot_sign(normal: np.ndarray, delta: np.ndarray) -> int:
    products = np.asarray(normal, dtype=np.longdouble) * np.asarray(delta, dtype=np.longdouble)
    value = np.sum(products, dtype=np.longdouble)
    bound = np.finfo(np.longdouble).eps * 32.0 * np.sum(np.abs(products), dtype=np.longdouble)
    if abs(value) > bound:
        return 1 if value > 0 else -1
    exact = sum((Fraction.from_float(float(normal[index])) * Fraction.from_float(float(delta[index])) for index in range(3)), Fraction(0))
    return 1 if exact > 0 else (-1 if exact < 0 else 0)


def _crossing_node(mesh: CanonicalMesh, first: int, second: int, d_first: float, d_second: float) -> tuple[str, np.ndarray]:
    if second < first:
        first, second, d_first, d_second = second, first, d_second, d_first
    fraction = d_first / (d_first - d_second)
    point = mesh.vertices[first] + fraction * (mesh.vertices[second] - mesh.vertices[first])
    return f"E{first}:{second}", point


def _triangle_plane_segment(mesh: CanonicalMesh, face_id: int, point: np.ndarray, normal: np.ndarray) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]] | str | None:
    face = [int(value) for value in mesh.faces[face_id]]
    deltas = [mesh.vertices[index] - point for index in face]
    signs = [_exact_dot_sign(normal, delta) for delta in deltas]
    if signs.count(0) >= 2:
        return "COPLANAR_PATCH"
    nodes: list[tuple[str, np.ndarray]] = []
    if signs.count(0) == 1:
        zero = signs.index(0)
        others = [index for index in range(3) if index != zero]
        if signs[others[0]] * signs[others[1]] < 0:
            vertex = face[zero]
            nodes.append((f"V{vertex}", mesh.vertices[vertex].copy()))
            first, second = face[others[0]], face[others[1]]
            d_first = float(np.dot(normal, mesh.vertices[first] - point))
            d_second = float(np.dot(normal, mesh.vertices[second] - point))
            nodes.append(_crossing_node(mesh, first, second, d_first, d_second))
    else:
        for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
            if signs[first_index] * signs[second_index] < 0:
                first, second = face[first_index], face[second_index]
                d_first = float(np.dot(normal, mesh.vertices[first] - point))
                d_second = float(np.dot(normal, mesh.vertices[second] - point))
                nodes.append(_crossing_node(mesh, first, second, d_first, d_second))
    unique = {node_id: (node_id, node_point) for node_id, node_point in nodes}
    if len(unique) != 2:
        return None
    ordered = sorted(unique.values(), key=lambda item: item[0])
    return ordered[0], ordered[1]


def _ordered_section_component(segments: Sequence[SectionSegment]) -> SectionComponent | None:
    if not segments:
        return None
    component_ids = {segment.component_id for segment in segments}
    if len(component_ids) != 1:
        return None
    adjacency: dict[str, list[str]] = {}
    lookup: dict[tuple[str, str], SectionSegment] = {}
    points: dict[str, np.ndarray] = {}
    for segment in segments:
        key = tuple(sorted((segment.first_node, segment.second_node)))
        if key in lookup:
            return None
        lookup[key] = segment
        adjacency.setdefault(segment.first_node, []).append(segment.second_node)
        adjacency.setdefault(segment.second_node, []).append(segment.first_node)
        points.setdefault(segment.first_node, segment.first_point)
        points.setdefault(segment.second_node, segment.second_point)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    endpoints = sorted(node for node, neighbors in adjacency.items() if len(neighbors) == 1)
    closed = not endpoints and all(len(neighbors) == 2 for neighbors in adjacency.values())
    if not closed and len(endpoints) != 2:
        return None
    start = min(adjacency) if closed else endpoints[0]

    def walk(first_neighbor: str | None) -> tuple[str, ...]:
        result = [start]
        previous: str | None = None
        current = start
        while True:
            options = sorted(node for node in adjacency[current] if node != previous)
            if previous is None and first_neighbor is not None:
                options = [first_neighbor]
            if not options:
                break
            selected = options[0]
            if closed and selected == start:
                break
            if selected in result:
                return ()
            result.append(selected)
            previous, current = current, selected
        return tuple(result)

    if closed:
        paths = [walk(neighbor) for neighbor in sorted(adjacency[start])]
        ordered_nodes = min(path for path in paths if path)
    else:
        ordered_nodes = walk(None)
    if len(ordered_nodes) != len(adjacency):
        return None
    segment_pairs = [
        (ordered_nodes[index], ordered_nodes[(index + 1) % len(ordered_nodes)])
        for index in range(len(ordered_nodes) if closed else len(ordered_nodes) - 1)
    ]
    normals: list[np.ndarray] = []
    face_ids: list[int] = []
    lengths: list[float] = []
    ordered_points = tuple(points[node] for node in ordered_nodes)
    for first, second in segment_pairs:
        segment = lookup.get(tuple(sorted((first, second))))
        if segment is None:
            return None
        # Segment reversal changes the curve orientation, not the inherited face normal.
        normals.append(segment.normal)
        face_ids.append(segment.face_id)
        length = float(np.linalg.norm(points[second] - points[first]))
        if not math.isfinite(length) or length <= 0.0:
            return None
        lengths.append(length)
    arc = [0.0]
    for length in lengths:
        arc.append(arc[-1] + length)
    return SectionComponent(
        ordered_nodes,
        ordered_points,
        tuple(normals),
        tuple(face_ids),
        next(iter(component_ids)),
        closed,
        tuple(arc),
        float(arc[-1]),
    )


def _section_components(
    mesh: CanonicalMesh,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    radius: float,
    budget: Budget,
) -> list[SectionComponent]:
    face_ids = mesh.triangle_bvh.query_plane_ball(plane_point, plane_normal, radius, budget)
    budget.add_tests(len(face_ids))
    if len(face_ids) > MAX_INTERSECTING_TRIANGLES_PER_STATION:
        raise CAMNTPResourceAbort("INTERSECTING_TRIANGLE_CAP")
    segments: list[SectionSegment] = []
    invalid_nodes: set[str] = set()
    for face_id in face_ids:
        component_id = int(mesh.face_component[face_id])
        component = mesh.components[component_id]
        if not component.valid_orientation:
            continue
        crossing = _triangle_plane_segment(mesh, face_id, plane_point, plane_normal)
        if crossing == "COPLANAR_PATCH":
            for vertex in mesh.faces[face_id]:
                invalid_nodes.add(f"V{int(vertex)}")
            continue
        if crossing is None:
            continue
        (first_node, first_point), (second_node, second_point) = crossing
        delta = second_point - first_point
        denominator = float(np.dot(delta, delta))
        if denominator <= 0.0:
            continue
        fraction = float(np.clip(np.dot(plane_point - first_point, delta) / denominator, 0.0, 1.0))
        closest = first_point + fraction * delta
        if float(np.linalg.norm(closest - plane_point)) > radius:
            continue
        segments.append(SectionSegment(
            first_node,
            second_node,
            first_point,
            second_point,
            mesh.face_normals[face_id],
            face_id,
            component_id,
        ))
    if len(segments) > MAX_SEGMENTS_PER_STATION:
        raise CAMNTPResourceAbort("STATION_SEGMENT_CAP")
    budget.add_segments(len(segments))
    if not segments:
        return []
    node_segments: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        node_segments.setdefault(segment.first_node, []).append(index)
        node_segments.setdefault(segment.second_node, []).append(index)
    visited: set[int] = set()
    output: list[SectionComponent] = []
    for seed in range(len(segments)):
        if seed in visited:
            continue
        stack = [seed]
        members: list[int] = []
        touches_invalid = False
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            segment = segments[current]
            for node in (segment.first_node, segment.second_node):
                if node in invalid_nodes or len(node_segments[node]) > 2:
                    touches_invalid = True
                stack.extend(node_segments[node])
        if touches_invalid:
            continue
        ordered = _ordered_section_component([segments[index] for index in sorted(members)])
        if ordered is not None:
            output.append(ordered)
    if len(output) > MAX_COMPONENTS_PER_STATION:
        raise CAMNTPResourceAbort("STATION_COMPONENT_CAP")
    output.sort(key=lambda component: (component.component_id, component.node_ids))
    return output


@dataclass(frozen=True)
class Attachment:
    distance: float
    arc: float
    point: np.ndarray


def _closest_attachment(component: SectionComponent, point: np.ndarray, tolerance: float) -> tuple[Attachment | None, bool]:
    segment_count = len(component.normals)
    proposals: list[Attachment] = []
    for index in range(segment_count):
        first = component.points[index]
        second = component.points[(index + 1) % len(component.points)]
        delta = second - first
        denominator = float(np.dot(delta, delta))
        if denominator <= 0.0:
            continue
        fraction = float(np.clip(np.dot(point - first, delta) / denominator, 0.0, 1.0))
        closest = first + fraction * delta
        proposals.append(Attachment(
            float(np.linalg.norm(point - closest)),
            component.arc[index] + fraction * math.sqrt(denominator),
            closest,
        ))
    if not proposals:
        return None, False
    proposals.sort(key=lambda row: (row.distance, row.arc))
    best = proposals[0]
    ambiguous = any(
        abs(other.distance - best.distance) <= tolerance
        and float(np.linalg.norm(other.point - best.point)) > tolerance
        for other in proposals[1:]
    )
    return best, ambiguous


@dataclass(frozen=True)
class ArcPiece:
    start: float
    end: float
    normal: np.ndarray
    segment_index: int


def _arc_pieces(component: SectionComponent, attachment_arc: float, low: float, high: float) -> list[ArcPiece] | None:
    if high <= low:
        return []
    absolute_low = attachment_arc + low
    absolute_high = attachment_arc + high
    if component.closed:
        if high - low > component.length:
            return None
        shifts = range(
            int(math.floor(absolute_low / component.length)) - 1,
            int(math.floor(absolute_high / component.length)) + 2,
        )
    else:
        if absolute_low < -1e-12 or absolute_high > component.length + 1e-12:
            return None
        shifts = (0,)
    pieces: list[ArcPiece] = []
    for segment_index, normal in enumerate(component.normals):
        base_low = component.arc[segment_index]
        base_high = component.arc[segment_index + 1]
        for shift in shifts:
            current_low = base_low + shift * component.length
            current_high = base_high + shift * component.length
            overlap_low = max(absolute_low, current_low)
            overlap_high = min(absolute_high, current_high)
            if overlap_high > overlap_low:
                pieces.append(ArcPiece(
                    overlap_low - attachment_arc,
                    overlap_high - attachment_arc,
                    normal,
                    segment_index,
                ))
    pieces.sort(key=lambda piece: (piece.start, piece.end, piece.segment_index))
    covered = sum(piece.end - piece.start for piece in pieces)
    if abs(covered - (high - low)) > max(1e-11, 1e-9 * (high - low)):
        return None
    return pieces


def _projected_normal(normal: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    projected = normal - float(np.dot(normal, direction)) * direction
    norm = float(np.linalg.norm(projected))
    if norm < PROJECTED_NORMAL_MIN:
        return None
    return projected / norm


@dataclass(frozen=True)
class IntervalSummary:
    coverage: float
    resultant: float
    mean: np.ndarray | None


def _interval_summary(component: SectionComponent, attachment_arc: float, low: float, high: float, direction: np.ndarray) -> IntervalSummary | None:
    pieces = _arc_pieces(component, attachment_arc, low, high)
    if pieces is None:
        return None
    total = high - low
    valid = 0.0
    vector = np.zeros(3, dtype=np.float64)
    for piece in pieces:
        length = piece.end - piece.start
        projected = _projected_normal(piece.normal, direction)
        if projected is None:
            continue
        valid += length
        vector += length * projected
    coverage = valid / total if total > 0.0 else 0.0
    magnitude = float(np.linalg.norm(vector))
    resultant = magnitude / valid if valid > 0.0 else 0.0
    mean = vector / magnitude if magnitude > 0.0 else None
    return IntervalSummary(coverage, resultant, mean)


@dataclass(frozen=True)
class ScaleResult:
    passed: bool
    reason: str
    theta: float | None = None
    coherence: float | None = None
    concentration: float | None = None
    boundary_ratio: float | None = None
    center_ratio: float | None = None
    minus_mean: np.ndarray | None = None
    plus_mean: np.ndarray | None = None


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))


def _turn_discontinuities(
    component: SectionComponent,
    attachment_arc: float,
    low: float,
    high: float,
    direction: np.ndarray,
) -> list[tuple[float, float]]:
    pieces = _arc_pieces(component, attachment_arc, low, high)
    if pieces is None or len(pieces) < 2:
        return []
    output: list[tuple[float, float]] = []
    for first, second in zip(pieces, pieces[1:]):
        if abs(first.end - second.start) > 1e-10:
            continue
        first_normal = _projected_normal(first.normal, direction)
        second_normal = _projected_normal(second.normal, direction)
        if first_normal is None or second_normal is None:
            continue
        delta = math.atan2(
            float(np.dot(direction, np.cross(first_normal, second_normal))),
            float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0)),
        )
        output.append((0.5 * (first.end + second.start), delta))
    return output


def evaluate_scale(
    mesh: CanonicalMesh,
    component: SectionComponent,
    attachment: Attachment,
    candidate: Candidate,
    plane_point: np.ndarray,
    radius: float,
    budget: Budget,
) -> ScaleResult:
    j_summary = _interval_summary(component, attachment.arc, -1.5 * radius, 1.5 * radius, candidate.direction)
    minus = _interval_summary(component, attachment.arc, -1.25 * radius, -0.75 * radius, candidate.direction)
    plus = _interval_summary(component, attachment.arc, 0.75 * radius, 1.25 * radius, candidate.direction)
    if j_summary is None or minus is None or plus is None:
        return ScaleResult(False, "NO_BILATERAL_ARC")
    if not _ge(min(j_summary.coverage, minus.coverage, plus.coverage), 0.95):
        return ScaleResult(False, "TANGENT_COVERAGE")
    if not _ge(minus.resultant, 0.965) or not _ge(plus.resultant, 0.965) or minus.mean is None or plus.mean is None:
        return ScaleResult(False, "SHELF_INCOHERENT")
    theta = _angle(minus.mean, plus.mean)
    # Filtered geometric arithmetic can undershoot an analytic equality by a
    # handful of ulps; the tolerance is many orders below the 1e-6-degree
    # comparator fixture and preserves its pass/fail separation.
    if not _ge(theta, ANGLE_MIN):
        return ScaleResult(False, "TURN_TOO_SMALL", theta=theta)
    turns = _turn_discontinuities(component, attachment.arc, -1.25 * radius, 1.25 * radius, candidate.direction)
    budget.add_discontinuities(len(turns))
    if not turns:
        return ScaleResult(False, "TURN_TOO_SMALL", theta=theta)
    variation = sum(abs(delta) for _position, delta in turns)
    accumulated = abs(sum(delta for _position, delta in turns))
    central = sum(abs(delta) for position, delta in turns if -0.5 * radius <= position <= 0.5 * radius)
    if variation <= 0.0 or not _le(variation, 1.25 * math.pi):
        return ScaleResult(False, "TURN_VARIATION", theta=theta)
    coherence = accumulated / variation
    if not _ge(coherence, 0.70):
        return ScaleResult(False, "TURN_INCOHERENT", theta=theta, coherence=coherence)
    concentration = central / variation
    if not _ge(concentration, 0.70):
        return ScaleResult(False, "TURN_DIFFUSE", theta=theta, coherence=coherence, concentration=concentration)
    if not _le(abs(accumulated - theta), max(ANGLE_SPREAD_ABS, 0.20 * theta)):
        return ScaleResult(False, "ENDPOINT_TURN_MISMATCH", theta=theta, coherence=coherence, concentration=concentration)
    barycenter = sum(abs(delta) * position for position, delta in turns) / variation
    if not _le(abs(barycenter), 0.20 * radius):
        return ScaleResult(False, "TURN_OFF_CENTER", theta=theta, coherence=coherence, concentration=concentration)
    center_distance = float(np.linalg.norm(attachment.point - plane_point))
    center_ratio = center_distance / radius
    if not _le(center_distance, min(candidate.length / 128.0, radius / 4.0)):
        return ScaleResult(False, "CANDIDATE_OFF_CENTER", theta=theta, coherence=coherence, concentration=concentration, center_ratio=center_ratio)
    boundary_distance = mesh.boundary_distance(component.component_id, attachment.point, budget)
    boundary_ratio = boundary_distance / radius
    if not _ge(boundary_distance, 1.25 * radius):
        return ScaleResult(False, "BOUNDARY_NEAR", theta=theta, coherence=coherence, concentration=concentration, boundary_ratio=boundary_ratio, center_ratio=center_ratio)
    return ScaleResult(
        True,
        "PASS",
        theta,
        coherence,
        concentration,
        boundary_ratio,
        center_ratio,
        minus.mean,
        plus.mean,
    )


@dataclass(frozen=True)
class Plateau:
    start_index: int
    representative_radius: float
    representative_angle: float
    minimum_coherence: float
    minimum_concentration: float
    minimum_boundary_ratio: float
    maximum_center_ratio: float


def _shelf_pair_drift(first: ScaleResult, second: ScaleResult) -> float:
    assert first.minus_mean is not None and first.plus_mean is not None
    assert second.minus_mean is not None and second.plus_mean is not None
    aligned = max(_angle(first.minus_mean, second.minus_mean), _angle(first.plus_mean, second.plus_mean))
    swapped = max(_angle(first.minus_mean, second.plus_mean), _angle(first.plus_mean, second.minus_mean))
    return min(aligned, swapped)


def _first_plateau(results: Sequence[ScaleResult], candidate_length: float) -> Plateau | None:
    for start in range(SCALE_COUNT - 2):
        window = results[start : start + 3]
        if not all(result.passed for result in window):
            continue
        angles = [float(result.theta) for result in window if result.theta is not None]
        median_angle = float(np.median(np.asarray(angles, dtype=np.float64)))
        if not _le(max(angles) - min(angles), max(ANGLE_SPREAD_ABS, 0.20 * median_angle)):
            continue
        drift = max(_shelf_pair_drift(window[first], window[second]) for first in range(3) for second in range(first + 1, 3))
        if not _le(drift, SHELF_DRIFT_MAX):
            continue
        return Plateau(
            start,
            candidate_length * SCALE_RATIOS[start + 1],
            median_angle,
            min(float(result.coherence) for result in window if result.coherence is not None),
            min(float(result.concentration) for result in window if result.concentration is not None),
            min(float(result.boundary_ratio) for result in window if result.boundary_ratio is not None),
            max(float(result.center_ratio) for result in window if result.center_ratio is not None),
        )
    return None


@dataclass(frozen=True)
class StationResult:
    station_index: int
    passed: bool
    reason: str
    component_id: int | None = None
    attachment_point: np.ndarray | None = None
    offset: np.ndarray | None = None
    plateau: Plateau | None = None
    scale_results: tuple[ScaleResult, ...] = ()
    section_component: SectionComponent | None = None
    plane_point: np.ndarray | None = None


def _station_point(candidate: Candidate, index: int) -> np.ndarray:
    fraction = (index + 1) / 18.0
    return (
        candidate.start
        + fraction * (candidate.end - candidate.start)
        + (index - 8) * (2.0 ** -42) * candidate.length * candidate.direction
    )


def evaluate_station(mesh: CanonicalMesh, candidate: Candidate, station_index: int, budget: Budget) -> StationResult:
    plane_point = _station_point(candidate, station_index)
    if mesh.global_nonmanifold_distance(plane_point, budget) <= (2.0 ** -40) * candidate.length:
        return StationResult(station_index, False, "NONMANIFOLD_SUPPORT", plane_point=plane_point)
    components = _section_components(mesh, plane_point, candidate.direction, 3.0 * candidate.length / 8.0, budget)
    proposals: list[tuple[float, int, Attachment, SectionComponent, Plateau, tuple[ScaleResult, ...]]] = []
    initial_reasons: list[tuple[float, str]] = []
    for component in components:
        attachment, ambiguous_closest = _closest_attachment(component, plane_point, (2.0 ** -40) * candidate.length)
        if attachment is None:
            continue
        if ambiguous_closest:
            initial_reasons.append((attachment.distance, "AMBIGUOUS_COMPONENT"))
            continue
        if attachment.distance > candidate.length / 64.0:
            initial_reasons.append((attachment.distance, "NO_COMPONENT_SUPPORT"))
            continue
        if mesh.nonmanifold_distance(component.component_id, attachment.point, budget) <= (2.0 ** -40) * candidate.length:
            initial_reasons.append((attachment.distance, "NONMANIFOLD_SUPPORT"))
            continue
        scale_results = tuple(
            evaluate_scale(
                mesh,
                component,
                attachment,
                candidate,
                plane_point,
                candidate.length * ratio,
                budget,
            )
            for ratio in SCALE_RATIOS
        )
        plateau = _first_plateau(scale_results, candidate.length)
        if plateau is None:
            reasons = [result.reason for result in scale_results if not result.passed]
            dominant = max(set(reasons), key=lambda reason: (reasons.count(reason), reason)) if reasons else "NO_SCALE_PLATEAU"
            initial_reasons.append((attachment.distance, dominant))
            continue
        proposals.append((attachment.distance, component.component_id, attachment, component, plateau, scale_results))
    proposals.sort(key=lambda row: (row[0], row[1], row[3].node_ids))
    if len(proposals) > 1 and proposals[1][0] - proposals[0][0] <= (2.0 ** -40) * candidate.length:
        return StationResult(station_index, False, "AMBIGUOUS_COMPONENT", plane_point=plane_point)
    if not proposals:
        if not components:
            reason = "NO_SECTION_SUPPORT"
        elif initial_reasons:
            reason = min(initial_reasons, key=lambda item: (item[0], item[1]))[1]
        else:
            reason = "NO_SCALE_PLATEAU"
        return StationResult(station_index, False, reason, plane_point=plane_point)
    _distance, component_id, attachment, component, plateau, scale_results = proposals[0]
    return StationResult(
        station_index,
        True,
        "PASS",
        component_id,
        attachment.point,
        attachment.point - plane_point,
        plateau,
        scale_results,
        component,
        plane_point,
    )


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    decision: str
    terminal_reason: str
    stations: tuple[StationResult, ...]
    station_pass_mask: int
    linked_run_mask: int
    station_pass_count: int
    longest_linked_run: int


def _link(first: StationResult, second: StationResult, candidate_length: float) -> bool:
    if not first.passed or not second.passed:
        return False
    assert first.plateau is not None and second.plateau is not None
    assert first.offset is not None and second.offset is not None
    if first.component_id != second.component_id:
        return False
    limit = 0.50 * min(first.plateau.representative_radius, second.plateau.representative_radius) + candidate_length / 1024.0
    if not _le(float(np.linalg.norm(second.offset - first.offset)), limit):
        return False
    ratio = max(
        first.plateau.representative_radius / second.plateau.representative_radius,
        second.plateau.representative_radius / first.plateau.representative_radius,
    )
    return _le(ratio, 4.0)


def evaluate_candidate(mesh: CanonicalMesh, candidate: Candidate, budget: Budget, started: float) -> CandidateEvaluation:
    budget.begin_candidate()
    stations: list[StationResult] = []
    for index in range(STATION_COUNT):
        if time.perf_counter() - started > MAX_ADDED_SECONDS:
            raise CAMNTPResourceAbort("V69_WALL_TIME_CAP")
        stations.append(evaluate_station(mesh, candidate, index, budget))
    pass_mask = sum((1 << result.station_index) for result in stations if result.passed)
    pass_count = int(pass_mask.bit_count())
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, result in enumerate(stations):
        if result.passed and (index == 0 or (stations[index - 1].passed and _link(stations[index - 1], result, candidate.length))):
            if run_start is None:
                run_start = index
        else:
            if run_start is not None:
                runs.append((run_start, index - 1))
            run_start = index if result.passed else None
    if run_start is not None:
        runs.append((run_start, STATION_COUNT - 1))
    longest = max((end - start + 1 for start, end in runs), default=0)
    longest_rows = [(start, end) for start, end in runs if end - start + 1 == longest]
    linked_mask = 0
    if longest_rows:
        start, end = min(longest_rows)
        linked_mask = sum(1 << index for index in range(start, end + 1))
    if pass_count < 9:
        decision, reason = "VETO", "LONGITUDINAL_COUNT"
    elif longest < 7:
        decision, reason = "VETO", "LONGITUDINAL_RUN"
    else:
        decision, reason = "KEEP", "KEEP"
    return CandidateEvaluation(candidate, decision, reason, tuple(stations), pass_mask, linked_mask, pass_count, longest)


def _histogram(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _candidate_record(evaluation: CandidateEvaluation) -> dict[str, Any]:
    passing = [station for station in evaluation.stations if station.passed and station.plateau is not None]
    plateau_histogram = [0] * 13
    for station in passing:
        plateau_histogram[station.plateau.start_index] += 1
    angles = [station.plateau.representative_angle for station in passing]
    scales = [station.plateau.representative_radius / evaluation.candidate.length for station in passing]
    coherences = [station.plateau.minimum_coherence for station in passing]
    concentrations = [station.plateau.minimum_concentration for station in passing]
    boundaries = [station.plateau.minimum_boundary_ratio for station in passing]
    centers = [station.plateau.maximum_center_ratio for station in passing]
    return {
        "candidate_opaque_hash": evaluation.candidate.opaque_hash,
        "decision": evaluation.decision,
        "station_pass_mask_17": evaluation.station_pass_mask,
        "linked_run_mask_17": evaluation.linked_run_mask,
        "station_pass_count": evaluation.station_pass_count,
        "longest_linked_run": evaluation.longest_linked_run,
        "selected_plateau_start_histogram_13": plateau_histogram,
        "median_turn_microdegrees": _q_angle(float(np.median(angles))) if angles else 0,
        "median_scale_over_L_ppb": _q_ppb(float(np.median(scales))) if scales else 0,
        "minimum_coherence_ppm": _q_ppm(min(coherences)) if coherences else 0,
        "minimum_concentration_ppm": _q_ppm(min(concentrations)) if concentrations else 0,
        "minimum_boundary_ratio_ppm": _q_ppm(min(boundaries)) if boundaries else 0,
        "maximum_center_offset_ratio_ppm": _q_ppm(max(centers)) if centers else 0,
        "station_failure_histogram": _histogram(station.reason for station in evaluation.stations if not station.passed),
        "terminal_reason": evaluation.terminal_reason,
    }


def frozen_configuration() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "candidate_count_cap": MAX_CANDIDATES,
        "triangle_cap": MAX_TRIANGLES,
        "vertex_cap": MAX_VERTICES,
        "station_count": STATION_COUNT,
        "scale_ratios_ppb": [_q_ppb(value) for value in SCALE_RATIOS],
        "turn_min_microdegrees": _q_angle(ANGLE_MIN),
        "projected_normal_min_ppm": _q_ppm(PROJECTED_NORMAL_MIN),
        "coverage_min_ppm": 950_000,
        "shelf_resultant_min_ppm": 965_000,
        "coherence_min_ppm": 700_000,
        "concentration_min_ppm": 700_000,
        "station_pass_min": 9,
        "linked_run_min": 7,
        "max_audit_bytes": MAX_AUDIT_BYTES,
        "max_added_seconds_millis": int(MAX_ADDED_SECONDS * 1000),
    }


CONSTANT_SHA256 = hashlib.sha256(canonical_bytes(frozen_configuration())).hexdigest()
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() if FIXTURE_PATH.exists() else hashlib.sha256(FIXTURE_TOKEN.encode("ascii")).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most BSD-derived Python builds report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _abort_result(
    baseline: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    status: str,
    reason: str,
    started_wall: float,
    started_cpu: float,
    started_rss: int,
) -> dict[str, Any]:
    baseline["camntp_v1_decision_record"] = {
        "schema": SCHEMA,
        "architecture": ARCHITECTURE,
        "constant_sha256": CONSTANT_SHA256,
        "implementation_sha256": _implementation_sha256(),
        "fixture_sha256": FIXTURE_SHA256,
        "preprocess_status": status,
        "scan_decision_status": reason,
        "candidate_count": len(rows),
        "candidate_records": [],
    }
    baseline["camntp_v1_operational_envelope"] = {
        "wall_time_microseconds": int(round((time.perf_counter() - started_wall) * 1_000_000)),
        "cpu_time_microseconds": int(round((time.process_time() - started_cpu) * 1_000_000)),
        "peak_rss_bytes": _rss_bytes(),
        "additional_peak_rss_bytes": max(0, _rss_bytes() - started_rss),
        "job_status": status,
        "terminal_reason": reason,
    }
    baseline["predictions"] = [deepcopy(dict(row)) for row in rows]
    baseline["candidate_count"] = len(rows)
    baseline["camntp_v1_modified_output_committed"] = False
    baseline["source_only"] = True
    baseline["research_only"] = True
    return baseline


def apply_camntp_v1(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    *,
    ordinary_nms: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    """Evaluate all candidates independently and atomically commit deletions.

    ``ordinary_nms`` is optional for source-free and shadow runs.  When passed,
    it is invoked exactly once on the surviving canonical upstream stream.
    """

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_rss = _rss_bytes()
    baseline = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
    rows = [deepcopy(dict(row)) for row in (source.get("predictions") or ())] if isinstance(source, Mapping) else [deepcopy(dict(row)) for row in source]
    if len(rows) > MAX_CANDIDATES:
        return _abort_result(baseline, rows, "RESOURCE_ABORT", "CANDIDATE_CAP", started_wall, started_cpu, started_rss), ()
    if vertices is None or faces is None or len(np.asarray(faces)) == 0:
        return _abort_result(baseline, rows, "ABSTAIN_NO_MESH", "ABSTAIN_NO_MESH", started_wall, started_cpu, started_rss), ()
    try:
        candidates = tuple(_candidate(row, ordinal) for ordinal, row in enumerate(rows))
        if len({candidate.opaque_hash for candidate in candidates}) != len(candidates):
            raise CAMNTPInputError("DUPLICATE_CANDIDATE_OPAQUE_HASH")
        mesh = CanonicalMesh(np.asarray(vertices), np.asarray(faces))
        budget = Budget()
        evaluations = tuple(
            evaluate_candidate(mesh, candidate, budget, started_wall)
            for candidate in sorted(candidates, key=lambda item: item.opaque_hash)
        )
        evaluation_by_ordinal = {evaluation.candidate.original_ordinal: evaluation for evaluation in evaluations}
        survivors = [rows[index] for index in range(len(rows)) if evaluation_by_ordinal[index].decision == "KEEP"]
        output_rows: Any = survivors
        ordinary_nms_audit: Any = None
        if ordinary_nms is not None:
            nms_result = ordinary_nms(survivors)
            if isinstance(nms_result, tuple) and len(nms_result) == 2:
                output_rows, ordinary_nms_audit = nms_result
            else:
                output_rows = nms_result
        candidate_hashes = sorted(candidate.opaque_hash for candidate in candidates)
        record = {
            "schema": SCHEMA,
            "architecture": ARCHITECTURE,
            "constant_sha256": CONSTANT_SHA256,
            "implementation_sha256": _implementation_sha256(),
            "fixture_sha256": FIXTURE_SHA256,
            "source_mesh_opaque_hash": mesh.opaque_hash,
            "candidate_stream_opaque_hash": hashlib.sha256(canonical_bytes(candidate_hashes)).hexdigest(),
            "preprocess_status": "OK",
            "triangle_count": len(mesh.faces),
            "manifold_component_count": sum(component.valid_orientation for component in mesh.components),
            "boundary_edge_count": mesh.boundary_edge_count,
            "nonmanifold_edge_count": mesh.nonmanifold_edge_count,
            "candidate_count": len(candidates),
            "scan_decision_status": "OK",
            "candidate_records": [_candidate_record(evaluation) for evaluation in evaluations],
        }
        record["decision_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
        encoded = canonical_bytes(record)
        if len(encoded) > MAX_AUDIT_BYTES:
            raise CAMNTPResourceAbort("AUDIT_CAP")
        wall = time.perf_counter() - started_wall
        additional_rss = max(0, _rss_bytes() - started_rss)
        if wall > MAX_ADDED_SECONDS:
            raise CAMNTPResourceAbort("V69_WALL_TIME_CAP")
        if additional_rss >= MAX_ADDED_RSS_BYTES or _rss_bytes() >= MAX_TOTAL_RSS_BYTES:
            raise CAMNTPResourceAbort("RSS_CAP")
        baseline.update({
            "schema_version": 1,
            "kind": "sherman_qc_f1_v69_camntp_result",
            "profile": ARCHITECTURE,
            "predictions": [deepcopy(dict(row)) for row in output_rows],
            "candidate_count": len(output_rows),
            "camntp_v1_decision_record": record,
            "camntp_v1_operational_envelope": {
                "wall_time_microseconds": int(round(wall * 1_000_000)),
                "cpu_time_microseconds": int(round((time.process_time() - started_cpu) * 1_000_000)),
                "peak_rss_bytes": _rss_bytes(),
                "additional_peak_rss_bytes": additional_rss,
                "bvh_node_visits": budget.bvh_node_visits,
                "exact_tests": budget.exact_tests,
                "segment_count": budget.segments,
                "discontinuity_count": budget.discontinuities,
                "audit_bytes": len(encoded),
                "ordinary_nms_call_count": 1 if ordinary_nms is not None else 0,
                "ordinary_nms_audit_present": ordinary_nms_audit is not None,
                "job_status": "OK",
            },
            "camntp_v1_modified_output_committed": True,
            "source_only": True,
            "research_only": True,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "filename_identity_used": False,
            "candidate_score_used": False,
            "route_provenance_used": False,
        })
        return baseline, evaluations
    except (CAMNTPInputError, CAMNTPResourceAbort, MemoryError, OverflowError, FloatingPointError) as error:
        status = "RESOURCE_ABORT" if isinstance(error, (CAMNTPResourceAbort, MemoryError)) else "INPUT_OR_NUMERIC_ABORT"
        return _abort_result(baseline, rows, status, str(error), started_wall, started_cpu, started_rss), ()


__all__ = [
    "ARCHITECTURE",
    "CAMNTPInputError",
    "CAMNTPResourceAbort",
    "CandidateEvaluation",
    "CanonicalMesh",
    "CONSTANT_SHA256",
    "FIXTURE_SHA256",
    "SCALE_RATIOS",
    "ScaleResult",
    "StationResult",
    "apply_camntp_v1",
    "canonical_bytes",
    "evaluate_scale",
    "frozen_configuration",
]
