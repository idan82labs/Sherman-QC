"""V66 transverse-face-turn ownership certificate (TFTOC_V1).

Research-only, source-only and suppression-only.  TFTOC consumes the frozen
V52 pre-NMS list, measures exact transverse sections of the source triangle
mesh, removes only non-representatives of all-pairs-positive ownership
cliques, and leaves the surviving candidate objects byte-for-byte unchanged.

The source-free fixtures exercise mathematical and metamorphic invariants.
They are deliberately not evidence of physical-corpus similarity or F1.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import resource
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE = "V66_TRANSVERSE_FACE_TURN_OWNERSHIP_CERTIFICATE_V1"
SHORT_NAME = "TFTOC_V1"
LOCK_TOKEN = "V66_TFTOC_V1_MECHANISM_LOCKED"
FIXTURE_TOKEN = "V66_TFTOC_FIXTURES_LOCKED"

MAX_SOURCE_VERTICES = 220_000
MAX_SOURCE_TRIANGLES = 440_000
MAX_CANDIDATES = 128
MAX_ELIGIBLE_PAIRS = 512
MAX_PAIR_STATIONS = 4_608
MAX_QUERIED_FACES_PER_STATION = 8_192
MAX_SECTION_NODES_PER_STATION = 4_096
MAX_SECTION_COMPONENTS_PER_STATION = 64
MAX_FACE_PLANE_TESTS = 8_000_000
MAX_SECTION_SEGMENTS = 1_000_000
MAX_AUDIT_BYTES = 2 * 1024 * 1024
WALL_SECONDS = 120.0

SAME_EVENT = 1
SPLIT_EVENT = 2
SPLIT_COMPONENT = 3
INDETERMINATE = 0

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
UNKNOWN = "UNKNOWN"

PLANE_EPS = 2.0 ** -30
ANGLE_FLOOR = math.radians(0.05)
ACTIVE_TURN = math.radians(8.0)
EVENT_MIN = math.radians(20.0)
EVENT_MAX = math.radians(145.0)
TAIL_MAX = math.radians(6.0)


class TFTOCInputError(RuntimeError):
    pass


class TFTOCResourceFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    route_hash: str
    better_key: tuple[Any, ...]
    original: Mapping[str, Any]


@dataclass(frozen=True)
class MeshEdge:
    first: int
    second: int
    length: float
    key: str


@dataclass(frozen=True)
class PairGeometry:
    first: Candidate
    second: Candidate
    axis: np.ndarray
    overlap_min: float
    overlap_max: float
    overlap: float
    h: float
    edge_population: int


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    center: float
    theta: float
    coherence: float
    width: float


@dataclass(frozen=True)
class Attachment:
    component: int
    arclength: float
    distance: float


@dataclass(frozen=True)
class Ownership:
    component: int
    event: int
    distance: float
    event_data: Event


@dataclass(frozen=True)
class StationResult:
    state: int
    reason: str
    first_distance: float | None = None
    second_distance: float | None = None
    event_theta: float | None = None
    event_width: float | None = None
    event_coherence: float | None = None


@dataclass(frozen=True)
class PairResult:
    first: int
    second: int
    route_hashes: tuple[str, str]
    scale: float
    stations: tuple[StationResult, ...]
    pair_class: str
    same_count: int
    split_count: int
    audit: Mapping[str, Any]


@dataclass
class _Budget:
    face_plane_tests: int = 0
    section_segments: int = 0

    def faces(self, count: int) -> None:
        self.face_plane_tests += int(count)
        if self.face_plane_tests > MAX_FACE_PLANE_TESTS:
            raise TFTOCResourceFailure("FACE_PLANE_TEST_CAP")

    def segments(self, count: int) -> None:
        self.section_segments += int(count)
        if self.section_segments > MAX_SECTION_SEGMENTS:
            raise TFTOCResourceFailure("SECTION_SEGMENT_CAP")


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise TFTOCInputError("NONFINITE_AUDIT_VALUE")
        return number.hex()
    if value is None or isinstance(value, str):
        return value
    return str(value)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _q(value: float, scale: float = 1_000_000.0) -> int:
    return int(round(float(value) * scale))


def _coordinate_bytes(point: np.ndarray) -> bytes:
    clean = np.asarray(point, dtype=np.float64).copy()
    clean[clean == 0.0] = 0.0
    return clean.astype(">f8", copy=False).tobytes()


def _coordinate_hash(point: np.ndarray) -> str:
    return hashlib.sha256(_coordinate_bytes(point)).hexdigest()


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise TFTOCInputError("INVALID_DIRECTION")
    result /= norm
    for value in result:
        if value > 0.0:
            break
        if value < 0.0:
            result = -result
            break
    result[result == 0.0] = 0.0
    return result


def _candidate(row: Mapping[str, Any], ordinal: int) -> Candidate:
    try:
        start = np.asarray(row["line_start"], dtype=np.float64)
        end = np.asarray(row["line_end"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise TFTOCInputError("INVALID_CANDIDATE") from error
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise TFTOCInputError("INVALID_CANDIDATE")
    length = float(np.linalg.norm(end - start))
    if not math.isfinite(length) or length <= 0.0:
        raise TFTOCInputError("INVALID_CANDIDATE")
    direction = _canonical_direction(end - start)
    if float(np.dot(end - start, direction)) < 0.0:
        start, end = end.copy(), start.copy()
    geometry_hash = hashlib.sha256(_coordinate_bytes(start) + _coordinate_bytes(end)).hexdigest()[:32]
    route_hash = str(row.get("route_hash") or row.get("bend_id") or geometry_hash)
    raw_key = row.get("better_first_key", row.get("total_order_key"))
    if raw_key is None:
        # Frozen V52 rows expose their immutable selector score and bend id.
        # Higher score is better; the stable identifier resolves exact ties.
        raw_key = (-float(row.get("selector_score") or 0.0), str(row.get("bend_id") or route_hash))
    if isinstance(raw_key, (str, int, float, bool)) or raw_key is None:
        better_key = (_native(raw_key),)
    else:
        better_key = tuple(_native(raw_key))
    return Candidate(ordinal, start, end, direction, length, route_hash, better_key, row)


def _point_on_axis(line: Candidate, axis: np.ndarray, coordinate: float) -> np.ndarray:
    denominator = float(np.dot(line.direction, axis))
    if abs(denominator) <= 1e-12:
        raise TFTOCInputError("PAIR_AXIS_DEGENERATE")
    return line.start + ((coordinate - float(np.dot(line.start, axis))) / denominator) * line.direction


class SourceMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray | None):
        raw = np.asarray(vertices, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise TFTOCInputError("INVALID_SOURCE_VERTICES")
        if len(raw) > MAX_SOURCE_VERTICES:
            raise TFTOCResourceFailure("SOURCE_VERTEX_CAP")
        if faces is None or len(np.asarray(faces)) == 0:
            raise TFTOCResourceFailure("NO_TRIANGLE_ADJACENCY")
        raw_faces = np.asarray(faces, dtype=np.int64)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise TFTOCInputError("INVALID_SOURCE_FACES")
        if len(raw_faces) > MAX_SOURCE_TRIANGLES:
            raise TFTOCResourceFailure("SOURCE_TRIANGLE_CAP")
        if not np.all((raw_faces >= 0) & (raw_faces < len(raw))):
            raise TFTOCInputError("INVALID_SOURCE_FACES")

        # Coordinate canonicalization makes every later decision independent of
        # input vertex order.  Duplicate-coordinate faces are discarded only if
        # they collapse geometrically.
        unique, inverse = np.unique(raw, axis=0, return_inverse=True)
        remapped = inverse[raw_faces]
        valid = (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
        remapped = remapped[valid]
        face_records: list[tuple[tuple[str, str, str], tuple[int, int, int]]] = []
        hashes = [_coordinate_hash(point) for point in unique]
        for face in remapped:
            ids = tuple(int(item) for item in face)
            key = tuple(sorted(hashes[item] for item in ids))
            face_records.append((key, ids))
        face_records.sort(key=lambda item: (item[0], item[1]))
        self.vertices = unique
        self.faces = np.asarray([item[1] for item in face_records], dtype=np.int64)
        self.vertex_hashes = hashes

        edge_map: dict[tuple[int, int], MeshEdge] = {}
        for face in self.faces:
            for first, second in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
                key_ids = (min(first, second), max(first, second))
                if key_ids in edge_map:
                    continue
                length = float(np.linalg.norm(self.vertices[first] - self.vertices[second]))
                if length <= 0.0 or not math.isfinite(length):
                    continue
                endpoint_hashes = sorted((hashes[first], hashes[second]))
                key = hashlib.sha256((endpoint_hashes[0] + endpoint_hashes[1]).encode("ascii")).hexdigest()
                edge_map[key_ids] = MeshEdge(key_ids[0], key_ids[1], length, key)
        self.edges = sorted(edge_map.values(), key=lambda edge: (edge.length, edge.key))
        topology_payload = {
            "vertices": sorted(hashes),
            "faces": sorted(item[0] for item in face_records),
        }
        self.topology_hash = hashlib.sha256(canonical_bytes(topology_payload)).hexdigest()


def _pair_axis(first: Candidate, second: Candidate) -> np.ndarray:
    aligned = second.direction if float(np.dot(first.direction, second.direction)) >= 0.0 else -second.direction
    return _canonical_direction(first.direction + aligned)


def _pair_scale(mesh: SourceMesh, first: Candidate, second: Candidate, axis: np.ndarray) -> tuple[float, int] | None:
    minimum = min(first.length, second.length)
    first_interval = sorted((float(np.dot(first.start, axis)), float(np.dot(first.end, axis))))
    second_interval = sorted((float(np.dot(second.start, axis)), float(np.dot(second.end, axis))))
    common_lo = max(first_interval[0], second_interval[0]) - 0.05 * minimum
    common_hi = min(first_interval[1], second_interval[1]) + 0.05 * minimum
    if common_hi <= common_lo:
        return None
    selected: list[MeshEdge] = []
    for edge in mesh.edges:
        if edge.length < (2.0 ** -20) * minimum or edge.length > 0.05 * minimum:
            continue
        midpoint = 0.5 * (mesh.vertices[edge.first] + mesh.vertices[edge.second])
        coordinate = float(np.dot(midpoint, axis))
        if coordinate < common_lo or coordinate > common_hi:
            continue
        distances = []
        for line in (first, second):
            anchor = _point_on_axis(line, axis, coordinate)
            residual = midpoint - anchor
            distances.append(float(np.linalg.norm(residual - float(np.dot(residual, axis)) * axis)))
        if min(distances) <= 0.15 * minimum:
            selected.append(edge)
    selected.sort(key=lambda edge: (edge.length, edge.key))
    if len(selected) < 256:
        return None
    h = float(selected[(len(selected) - 1) // 2].length)
    if h <= 0.0 or not math.isfinite(h) or not (40.0 <= minimum / h <= 400.0):
        return None
    return h, len(selected)


def _pair_geometry(mesh: SourceMesh, first: Candidate, second: Candidate) -> PairGeometry | None:
    if abs(float(np.dot(first.direction, second.direction))) < 0.9986295347545738:
        return None
    if max(first.length, second.length) / min(first.length, second.length) > 1.5:
        return None
    axis = _pair_axis(first, second)
    first_interval = sorted((float(np.dot(first.start, axis)), float(np.dot(first.end, axis))))
    second_interval = sorted((float(np.dot(second.start, axis)), float(np.dot(second.end, axis))))
    lo, hi = max(first_interval[0], second_interval[0]), min(first_interval[1], second_interval[1])
    overlap = hi - lo
    if overlap < 0.75 * min(first.length, second.length):
        return None
    scale = _pair_scale(mesh, first, second, axis)
    if scale is None:
        return None
    h, population = scale
    endpoint_separation = []
    projector = np.eye(3) - np.outer(axis, axis)
    for coordinate in (lo, hi):
        endpoint_separation.append(float(np.linalg.norm(projector @ (_point_on_axis(first, axis, coordinate) - _point_on_axis(second, axis, coordinate)))))
    if max(endpoint_separation) > 8.0 * h:
        return None
    return PairGeometry(first, second, axis, lo, hi, overlap, h, population)


def _edge_key(mesh: SourceMesh, first: int, second: int) -> str:
    hashes = sorted((mesh.vertex_hashes[first], mesh.vertex_hashes[second]))
    return hashlib.sha256((hashes[0] + hashes[1]).encode("ascii")).hexdigest()


def _crossing_point(mesh: SourceMesh, first: int, second: int, d_first: float, d_second: float) -> np.ndarray:
    # The endpoint ordering is coordinate-canonical so reordered meshes perform
    # the same arithmetic in the same order.
    if mesh.vertex_hashes[second] < mesh.vertex_hashes[first]:
        first, second, d_first, d_second = second, first, d_second, d_first
    fraction = d_first / (d_first - d_second)
    return mesh.vertices[first] + fraction * (mesh.vertices[second] - mesh.vertices[first])


@dataclass(frozen=True)
class _SectionEdge:
    first: str
    second: str
    normal: np.ndarray
    face_key: str


@dataclass(frozen=True)
class _Component:
    nodes: tuple[str, ...]
    points: tuple[np.ndarray, ...]
    normals: tuple[np.ndarray, ...]
    closed: bool
    arc: tuple[float, ...]
    length: float


def _ordered_component(
    node_ids: Sequence[str],
    node_points: Mapping[str, np.ndarray],
    edge_lookup: Mapping[tuple[str, str], _SectionEdge],
) -> _Component | None:
    adjacency = {node: [] for node in node_ids}
    for first, second in edge_lookup:
        if first in adjacency and second in adjacency:
            adjacency[first].append(second)
            adjacency[second].append(first)
    if any(len(adjacency[node]) > 2 for node in node_ids):
        return None
    endpoints = sorted(node for node in node_ids if len(adjacency[node]) == 1)
    closed = not endpoints and all(len(adjacency[node]) == 2 for node in node_ids)
    if not closed and len(endpoints) != 2:
        return None
    start = min(node_ids) if closed else endpoints[0]

    def walk(next_node: str | None) -> tuple[str, ...]:
        ordered = [start]
        previous = None
        current = start
        while True:
            options = sorted(node for node in adjacency[current] if node != previous)
            if previous is None and next_node is not None:
                options = [next_node]
            if not options:
                break
            chosen = options[0]
            if closed and chosen == start:
                break
            if chosen in ordered:
                return tuple()
            ordered.append(chosen)
            previous, current = current, chosen
        return tuple(ordered)

    if closed:
        choices = sorted(adjacency[start])
        paths = [walk(choice) for choice in choices]
        ordered = min(path for path in paths if path)
    else:
        ordered = walk(None)
    if len(ordered) != len(node_ids):
        return None

    points = tuple(node_points[node] for node in ordered)
    segment_pairs = [(ordered[index], ordered[(index + 1) % len(ordered)]) for index in range(len(ordered) if closed else len(ordered) - 1)]
    normals: list[np.ndarray] = []
    lengths: list[float] = []
    for first, second in segment_pairs:
        edge = edge_lookup.get(tuple(sorted((first, second))))
        if edge is None:
            return None
        normals.append(edge.normal)
        lengths.append(float(np.linalg.norm(node_points[second] - node_points[first])))
    arc = [0.0]
    for length in lengths:
        arc.append(arc[-1] + length)
    return _Component(ordered, points, tuple(normals), closed, tuple(arc), float(arc[-1]))


def _section_components(
    mesh: SourceMesh,
    pair: PairGeometry,
    coordinate: float,
    anchor_center: np.ndarray,
    budget: _Budget,
) -> tuple[list[_Component] | None, str]:
    h = pair.h
    # ``coordinate`` is the global axial level u·x.  Subtracting the pair
    # origin here as well would translate the section plane a second time.
    signed = (mesh.vertices @ pair.axis - coordinate) / h
    candidate_faces: list[tuple[np.ndarray, np.ndarray]] = []
    budget.faces(len(mesh.faces))
    for face in mesh.faces:
        ids = [int(item) for item in face]
        distances = signed[ids]
        if not (np.min(distances) < 0.0 < np.max(distances)):
            continue
        # A plane degeneracy is fatal only for a triangle in the retained 16h
        # station neighbourhood.
        crossings = []
        for first, second in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            d_first, d_second = float(signed[first]), float(signed[second])
            if d_first * d_second < 0.0:
                crossings.append((first, second, d_first, d_second))
        if len(crossings) != 2:
            if np.any(np.abs(distances) <= PLANE_EPS):
                return None, "PLANE_DEGENERACY"
            continue
        points = [_crossing_point(mesh, *crossing) for crossing in crossings]
        segment = points[1] - points[0]
        length2 = float(np.dot(segment, segment))
        if length2 <= 0.0:
            continue
        fraction = float(np.clip(np.dot(anchor_center - points[0], segment) / length2, 0.0, 1.0))
        closest = points[0] + fraction * segment
        if float(np.linalg.norm(closest - anchor_center)) <= 16.0 * h:
            candidate_faces.append((face, distances))
    if len(candidate_faces) > MAX_QUERIED_FACES_PER_STATION:
        raise TFTOCResourceFailure("QUERIED_FACE_CAP")

    node_points: dict[str, np.ndarray] = {}
    graph_edges: dict[tuple[str, str], _SectionEdge] = {}
    adjacency: dict[str, set[str]] = {}
    for face, distances in candidate_faces:
        ids = [int(item) for item in face]
        if np.any(np.abs(distances) <= PLANE_EPS):
            return None, "PLANE_DEGENERACY"
        cross = np.cross(mesh.vertices[ids[1]] - mesh.vertices[ids[0]], mesh.vertices[ids[2]] - mesh.vertices[ids[0]])
        norm = float(np.linalg.norm(cross))
        if norm <= 0.0:
            return None, "DEGENERATE_FACE"
        normal = cross / norm
        crossings = []
        for first, second in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            d_first, d_second = float(signed[first]), float(signed[second])
            if d_first * d_second < 0.0:
                key = _edge_key(mesh, first, second)
                point = _crossing_point(mesh, first, second, d_first, d_second)
                crossings.append((key, point))
        if len(crossings) != 2:
            return None, "INVALID_CROSSING_COUNT"
        (first_key, first_point), (second_key, second_point) = crossings
        node_points.setdefault(first_key, first_point)
        node_points.setdefault(second_key, second_point)
        edge_key = tuple(sorted((first_key, second_key)))
        if edge_key in graph_edges:
            return None, "DUPLICATED_SECTION_EDGE"
        face_key = hashlib.sha256("".join(sorted(mesh.vertex_hashes[item] for item in ids)).encode("ascii")).hexdigest()
        graph_edges[edge_key] = _SectionEdge(edge_key[0], edge_key[1], normal, face_key)
        adjacency.setdefault(first_key, set()).add(second_key)
        adjacency.setdefault(second_key, set()).add(first_key)
    budget.segments(len(graph_edges))
    if len(node_points) > MAX_SECTION_NODES_PER_STATION:
        raise TFTOCResourceFailure("SECTION_NODE_CAP")
    if not graph_edges:
        return [], "NO_SECTION_SUPPORT"

    components_nodes: list[list[str]] = []
    visited: set[str] = set()
    for seed in sorted(node_points):
        if seed in visited:
            continue
        stack, component = [seed], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(sorted(adjacency.get(node, ()), reverse=True))
        components_nodes.append(sorted(component))
    if len(components_nodes) > MAX_SECTION_COMPONENTS_PER_STATION:
        raise TFTOCResourceFailure("SECTION_COMPONENT_CAP")
    result: list[_Component] = []
    for nodes in components_nodes:
        component = _ordered_component(nodes, node_points, graph_edges)
        if component is not None:
            result.append(component)
    result.sort(key=lambda component: component.nodes)
    return result, "OK"


def _closest_attachment(point: np.ndarray, components: Sequence[_Component], h: float) -> tuple[Attachment | None, str]:
    rows: list[Attachment] = []
    for component_index, component in enumerate(components):
        segment_count = len(component.points) if component.closed else len(component.points) - 1
        best: tuple[float, float] | None = None
        for index in range(segment_count):
            first = component.points[index]
            second = component.points[(index + 1) % len(component.points)]
            delta = second - first
            length2 = float(np.dot(delta, delta))
            if length2 <= 0.0:
                continue
            fraction = float(np.clip(np.dot(point - first, delta) / length2, 0.0, 1.0))
            closest = first + fraction * delta
            distance = float(np.linalg.norm(point - closest))
            arc = component.arc[index] + fraction * math.sqrt(length2)
            proposal = (distance, arc)
            if best is None or proposal < best:
                best = proposal
        if best is not None:
            rows.append(Attachment(component_index, best[1], best[0]))
    rows.sort(key=lambda row: (row.distance, components[row.component].nodes, row.arclength))
    if not rows or rows[0].distance > 1.0 * h:
        return None, "NO_COMPONENT_SUPPORT"
    if len(rows) > 1 and rows[1].distance - rows[0].distance < 0.5 * h:
        return None, "AMBIGUOUS_COMPONENT"
    return rows[0], "OK"


def _component_turns(component: _Component, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    segment_count = len(component.normals)
    canonical_normals: list[np.ndarray] = []
    for index in range(segment_count):
        first = component.points[index]
        second = component.points[(index + 1) % len(component.points)]
        tangent = second - first
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            return np.empty(0), np.empty(0), "ZERO_SECTION_SEGMENT"
        tangent /= length
        raw = component.normals[index]
        g = float(np.dot(raw, np.cross(axis, tangent)))
        if abs(g) < 0.5:
            return np.empty(0), np.empty(0), "FACE_ALIGNMENT"
        canonical_normals.append(raw if g > 0.0 else -raw)
    turn_count = segment_count if component.closed else segment_count - 1
    locations, deltas = [], []
    for index in range(turn_count):
        first = canonical_normals[index]
        second = canonical_normals[(index + 1) % segment_count]
        cross = np.cross(first, second)
        alpha = math.atan2(float(np.linalg.norm(cross)), float(np.clip(np.dot(first, second), -1.0, 1.0)))
        location = component.arc[index + 1] if index + 1 < len(component.arc) else component.length
        if alpha < ANGLE_FLOOR:
            delta = 0.0
        else:
            cross_norm = float(np.linalg.norm(cross))
            alignment = abs(float(np.dot(axis, cross))) / max(cross_norm, 1e-300)
            if alignment < 0.5:
                return np.empty(0), np.empty(0), "TURN_AXIS_ALIGNMENT"
            delta = math.copysign(alpha, float(np.dot(axis, cross)))
        locations.append(location % component.length if component.closed else location)
        deltas.append(delta)
    return np.asarray(locations, dtype=np.float64), np.asarray(deltas, dtype=np.float64), "OK"


def _events(component: _Component, axis: np.ndarray, h: float) -> tuple[list[Event], str]:
    locations, deltas, reason = _component_turns(component, axis)
    if reason != "OK" or not len(locations):
        return [], reason
    # The locked definition operates on turn locations.  Collinear face
    # transitions whose delta was floored to zero are not turn locations and
    # must not widen a sharp or circular event merely because of triangulation.
    turning = np.abs(deltas) > 0.0
    locations, deltas = locations[turning], deltas[turning]
    if not len(locations):
        return [], "NO_ACTIVE_TURN"
    if component.closed:
        # Rotate the cycle after the largest arclength gap.  This makes ordinary
        # interval logic valid unless an event straddles the cut; trying every
        # largest-gap tie and choosing the canonical sequence handles that case.
        ordered = np.argsort(locations, kind="mergesort")
        locations, deltas = locations[ordered], deltas[ordered]
        gaps = np.diff(np.concatenate((locations, [locations[0] + component.length])))
        cut = int(np.argmax(gaps))
        order = np.concatenate((np.arange(cut + 1, len(locations)), np.arange(0, cut + 1)))
        locations = locations[order]
        locations[locations <= locations[0]] += component.length
        deltas = deltas[order]
    active = []
    for location in locations:
        mask = np.abs(locations - location) <= 1.0 * h
        signed = float(np.sum(deltas[mask]))
        absolute = float(np.sum(np.abs(deltas[mask])))
        active.append(abs(signed) >= ACTIVE_TURN and absolute > 0.0 and abs(signed) / absolute >= 0.80)
    active_ids = np.flatnonzero(active)
    if not len(active_ids):
        return [], "NO_ACTIVE_TURN"
    seeds: list[list[int]] = []
    for index in active_ids:
        if not seeds or locations[index] - locations[seeds[-1][-1]] > 1.25 * h:
            seeds.append([int(index)])
        else:
            seeds[-1].append(int(index))
    intervals: list[tuple[int, int]] = []
    for seed in seeds:
        left_value = locations[seed[0]] - 1.0 * h
        right_value = locations[seed[-1]] + 1.0 * h
        indices = np.flatnonzero((locations >= left_value) & (locations <= right_value))
        intervals.append((int(indices[0]), int(indices[-1])))
    merged: list[tuple[int, int]] = []
    for interval in intervals:
        if merged and locations[interval[0]] - locations[merged[-1][1]] <= 0.5 * h:
            merged[-1] = (merged[-1][0], interval[1])
        else:
            merged.append(interval)

    events: list[Event] = []
    for left, right in merged:
        event_delta = deltas[left : right + 1]
        signed = float(np.sum(event_delta))
        theta = abs(signed)
        total_variation = float(np.sum(np.abs(event_delta)))
        coherence = theta / total_variation if total_variation > 0.0 else 0.0
        width = float(locations[right] - locations[left])
        if not (EVENT_MIN <= theta <= EVENT_MAX and coherence >= 0.85 and width <= 6.0 * h):
            continue
        event_start, event_end = float(locations[left]), float(locations[right])
        tail_left_start, tail_right_end = event_start - 2.0 * h, event_end + 2.0 * h
        if not component.closed and (tail_left_start < 0.0 or tail_right_end > component.length):
            continue
        left_tail = (locations >= tail_left_start) & (locations < event_start)
        right_tail = (locations > event_end) & (locations <= tail_right_end)
        if float(np.sum(np.abs(deltas[left_tail]))) > TAIL_MAX or float(np.sum(np.abs(deltas[right_tail]))) > TAIL_MAX:
            continue
        # Every segment in the event and tails must be at most 1.25 h.
        segment_lengths = np.diff(np.asarray(component.arc))
        segment_positions = 0.5 * (np.asarray(component.arc[:-1]) + np.asarray(component.arc[1:]))
        if np.any(segment_lengths[(segment_positions >= max(0.0, tail_left_start % component.length if component.closed else tail_left_start)) & (segment_positions <= min(component.length, tail_right_end % component.length if component.closed and tail_right_end > component.length else tail_right_end))] > 1.25 * h):
            continue
        weights = np.abs(event_delta)
        center = float(np.sum(weights * locations[left : right + 1]) / np.sum(weights))
        events.append(Event(event_start % component.length, event_end % component.length, center % component.length, theta, coherence, width))
    events.sort(key=lambda event: (event.center, event.start, event.end, event.theta))
    return events, "OK" if events else "NO_OWNABLE_EVENT"


def _arc_distance(component: _Component, first: float, second: float) -> float:
    distance = abs(first - second)
    return min(distance, component.length - distance) if component.closed else distance


def _ownership(attachment: Attachment, components: Sequence[_Component], events_by_component: Sequence[Sequence[Event]], h: float) -> tuple[Ownership | None, str]:
    component = components[attachment.component]
    rows = [(_arc_distance(component, attachment.arclength, event.center), index, event) for index, event in enumerate(events_by_component[attachment.component])]
    rows.sort(key=lambda row: (row[0], row[2].center, row[1]))
    if not rows or rows[0][0] > 4.5 * h:
        return None, "NO_EVENT_OWNERSHIP"
    if len(rows) > 1 and rows[1][0] - rows[0][0] < 1.0 * h:
        return None, "AMBIGUOUS_EVENT"
    return Ownership(attachment.component, rows[0][1], rows[0][0], rows[0][2]), "OK"


def _evaluate_station(mesh: SourceMesh, pair: PairGeometry, station: int, budget: _Budget) -> StationResult:
    coordinate = pair.overlap_min + station * pair.overlap / 10.0
    first_anchor = _point_on_axis(pair.first, pair.axis, coordinate)
    second_anchor = _point_on_axis(pair.second, pair.axis, coordinate)
    center = 0.5 * (first_anchor + second_anchor)
    components, reason = _section_components(mesh, pair, coordinate, center, budget)
    if components is None or not components:
        return StationResult(INDETERMINATE, reason)
    first_attachment, reason_first = _closest_attachment(first_anchor, components, pair.h)
    second_attachment, reason_second = _closest_attachment(second_anchor, components, pair.h)
    if first_attachment is None or second_attachment is None:
        return StationResult(INDETERMINATE, reason_first if first_attachment is None else reason_second)
    events_by_component = []
    for component in components:
        component_events, _component_reason = _events(component, pair.axis, pair.h)
        events_by_component.append(component_events)
    first_owner, first_reason = _ownership(first_attachment, components, events_by_component, pair.h)
    second_owner, second_reason = _ownership(second_attachment, components, events_by_component, pair.h)
    if first_owner is None or second_owner is None:
        return StationResult(INDETERMINATE, first_reason if first_owner is None else second_reason)
    if first_owner.component != second_owner.component:
        return StationResult(SPLIT_COMPONENT, "SPLIT_COMPONENT", first_owner.distance / pair.h, second_owner.distance / pair.h)
    if first_owner.event != second_owner.event:
        return StationResult(SPLIT_EVENT, "SPLIT_EVENT", first_owner.distance / pair.h, second_owner.distance / pair.h)
    event = first_owner.event_data
    return StationResult(
        SAME_EVENT,
        "SAME_EVENT",
        first_owner.distance / pair.h,
        second_owner.distance / pair.h,
        math.degrees(event.theta),
        event.width / pair.h,
        event.coherence,
    )


def _station_audit(result: StationResult) -> Mapping[str, Any]:
    return {
        "state": result.state,
        "reason": result.reason,
        "first_anchor_distance_over_h_e6": _q(result.first_distance or 0.0),
        "second_anchor_distance_over_h_e6": _q(result.second_distance or 0.0),
        "event_angle_microdegrees": _q(result.event_theta or 0.0),
        "event_width_over_h_e6": _q(result.event_width or 0.0),
        "event_coherence_e6": _q(result.event_coherence or 0.0),
    }


def _evaluate_pair(mesh: SourceMesh, pair: PairGeometry, budget: _Budget) -> PairResult:
    stations = tuple(_evaluate_station(mesh, pair, station, budget) for station in range(1, 10))
    same = sum(result.state == SAME_EVENT for result in stations)
    split = sum(result.state in (SPLIT_EVENT, SPLIT_COMPONENT) for result in stations)
    pair_class = POSITIVE if same >= 7 and split == 0 else (NEGATIVE if split >= 1 else UNKNOWN)
    forward = [_station_audit(result) for result in stations]
    reverse = list(reversed(forward))
    station_vector = min(forward, reverse, key=canonical_bytes)
    hashes = tuple(sorted((pair.first.route_hash, pair.second.route_hash)))
    audit = {
        "route_hashes": list(hashes),
        "h_e6": _q(pair.h),
        "edge_population": pair.edge_population,
        "stations": station_vector,
        "same": same,
        "split": split,
        "indeterminate": 9 - same - split,
        "pair_class": pair_class,
    }
    return PairResult(pair.first.ordinal, pair.second.ordinal, hashes, pair.h, stations, pair_class, same, split, audit)


def _candidate_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes([dict(row) for row in rows])).hexdigest()


def _noop(baseline: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], reason: str) -> tuple[dict[str, Any], tuple[PairResult, ...]]:
    result = deepcopy(dict(baseline))
    result["predictions"] = [deepcopy(row) for row in rows]
    audit = {
        "profile": PROFILE,
        "lock_token": LOCK_TOKEN,
        "fixture_token": FIXTURE_TOKEN,
        "status": "RESOURCE_ATOMIC_NOOP" if reason not in ("NO_TRIANGLE_ADJACENCY",) else reason,
        "reason": reason,
        "incoming_candidate_count": len(rows),
        "candidate_list_digest": _candidate_digest(rows),
        "suppressed_hashes": [],
    }
    result["tftoc_v1_audit"] = _native(audit)
    result["tftoc_v1_audit_sha256"] = hashlib.sha256(canonical_bytes(audit)).hexdigest()
    return result, tuple()


def apply_tftoc_v1(
    baseline: Mapping[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray | None,
) -> tuple[dict[str, Any], tuple[PairResult, ...]]:
    """Apply TFTOC once to the frozen V52 pre-NMS candidate list."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rows = list(baseline.get("predictions", ()))
    if len(rows) > MAX_CANDIDATES:
        return _noop(baseline, rows, "CANDIDATE_CAP")
    try:
        candidates = [_candidate(row, ordinal) for ordinal, row in enumerate(rows)]
        mesh = SourceMesh(vertices, faces)
        pairs: list[PairGeometry] = []
        for first_index in range(len(candidates)):
            for second_index in range(first_index + 1, len(candidates)):
                pair = _pair_geometry(mesh, candidates[first_index], candidates[second_index])
                if pair is not None:
                    pairs.append(pair)
        pairs.sort(key=lambda pair: tuple(sorted((pair.first.route_hash, pair.second.route_hash))))
        if len(pairs) > MAX_ELIGIBLE_PAIRS:
            raise TFTOCResourceFailure("ELIGIBLE_PAIR_CAP")
        if 9 * len(pairs) > MAX_PAIR_STATIONS:
            raise TFTOCResourceFailure("PAIR_STATION_CAP")
        budget = _Budget()
        pair_results: list[PairResult] = []
        for pair in pairs:
            if time.perf_counter() - started_wall > WALL_SECONDS:
                raise TFTOCResourceFailure("WALL_CAP")
            pair_results.append(_evaluate_pair(mesh, pair, budget))

        positive = {(min(result.first, result.second), max(result.first, result.second)): result for result in pair_results if result.pair_class == POSITIVE}
        graph = {candidate.ordinal: set() for candidate in candidates}
        for first, second in positive:
            graph[first].add(second)
            graph[second].add(first)
        visited: set[int] = set()
        cliques: list[list[int]] = []
        for seed in sorted(graph):
            if seed in visited or not graph[seed]:
                continue
            stack, component = [seed], []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(sorted(graph[node], reverse=True))
            component.sort()
            complete = all((min(first, second), max(first, second)) in positive for index, first in enumerate(component) for second in component[index + 1 :])
            if complete and len(component) >= 2:
                cliques.append(component)

        suppressed: set[int] = set()
        clique_records = []
        for clique in cliques:
            ranks = []
            for ordinal in clique:
                candidate = candidates[ordinal]
                same_assignments = 0
                distances = []
                for other in clique:
                    if other == ordinal:
                        continue
                    result = positive[(min(ordinal, other), max(ordinal, other))]
                    for station in result.stations:
                        if station.state != SAME_EVENT:
                            continue
                        same_assignments += 1
                        distance = station.first_distance if result.first == ordinal else station.second_distance
                        if distance is not None:
                            distances.append(distance)
                centeredness = float(np.median(np.asarray(distances, dtype=np.float64))) if distances else math.inf
                rank = (-same_assignments, _q(centeredness), candidate.better_key, -_q(candidate.length), candidate.route_hash)
                ranks.append((rank, ordinal))
            ranks.sort(key=lambda row: row[0])
            representative = ranks[0][1]
            removed = sorted(ordinal for ordinal in clique if ordinal != representative)
            suppressed.update(removed)
            clique_records.append({
                "members": sorted(candidates[ordinal].route_hash for ordinal in clique),
                "representative": candidates[representative].route_hash,
                "representative_key": _native(ranks[0][0]),
                "suppressed": sorted(candidates[ordinal].route_hash for ordinal in removed),
            })
        survivors = [deepcopy(row) for ordinal, row in enumerate(rows) if ordinal not in suppressed]
        mutation = {
            "input": _candidate_digest(rows),
            "output": _candidate_digest(survivors),
            "suppressed": sorted(candidates[ordinal].route_hash for ordinal in suppressed),
        }
        audit = {
            "profile": PROFILE,
            "lock_token": LOCK_TOKEN,
            "fixture_token": FIXTURE_TOKEN,
            "configuration_sha256": CONFIGURATION_SHA256,
            "status": "OK",
            "mesh_topology_sha256": mesh.topology_hash,
            "candidate_list_digest": _candidate_digest(rows),
            "incoming_candidate_count": len(rows),
            "eligible_pair_count": len(pairs),
            "pair_records": [result.audit for result in pair_results],
            "cliques": clique_records,
            "suppressed_hashes": mutation["suppressed"],
            "pre_nms_output_digest": mutation["output"],
            "mutation_digest": hashlib.sha256(canonical_bytes(mutation)).hexdigest(),
            "face_plane_tests": budget.face_plane_tests,
            "emitted_section_segments": budget.section_segments,
        }
        if len(canonical_bytes(audit)) > MAX_AUDIT_BYTES:
            raise TFTOCResourceFailure("AUDIT_CAP")
        result = deepcopy(dict(baseline))
        result["predictions"] = survivors
        result["tftoc_v1_audit"] = _native(audit)
        result["tftoc_v1_audit_sha256"] = hashlib.sha256(canonical_bytes(audit)).hexdigest()
        result["tftoc_v1_runtime_unhashed"] = {
            "wall_time_us": _q(time.perf_counter() - started_wall),
            "cpu_time_us": _q(time.process_time() - started_cpu),
            "peak_rss_native": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        return result, tuple(pair_results)
    except (TFTOCResourceFailure, MemoryError) as error:
        reason = "MEMORY_CAP" if isinstance(error, MemoryError) else str(error)
        return _noop(baseline, rows, reason)


def frozen_configuration() -> Mapping[str, Any]:
    return {
        "profile": PROFILE,
        "short_name": SHORT_NAME,
        "lock_token": LOCK_TOKEN,
        "fixture_token": FIXTURE_TOKEN,
        "pair": {
            "parallel_dot_min": 0.9986295347545738,
            "length_ratio_max": 1.5,
            "overlap_fraction_min": 0.75,
            "endpoint_separation_over_h_max": 8.0,
        },
        "section": {
            "stations": [index / 10.0 for index in range(1, 10)],
            "plane_eps": PLANE_EPS,
            "component_attach_over_h": 1.0,
            "component_margin_over_h": 0.5,
            "face_alignment_min": 0.5,
            "turn_axis_alignment_min": 0.5,
        },
        "event": {
            "active_turn_deg": 8.0,
            "active_coherence": 0.80,
            "theta_deg": [20.0, 145.0],
            "coherence": 0.85,
            "width_over_h": 6.0,
            "tail_over_h": 2.0,
            "tail_turn_deg": 6.0,
            "ownership_over_h": 4.5,
            "ownership_margin_over_h": 1.0,
        },
        "positive": {"same_at_least": 7, "split_exactly": 0},
        "mutation": "all_pairs_positive_clique_suppression_only_before_one_ordinary_nms",
        "caps": {
            "vertices": MAX_SOURCE_VERTICES,
            "triangles": MAX_SOURCE_TRIANGLES,
            "candidates": MAX_CANDIDATES,
            "eligible_pairs": MAX_ELIGIBLE_PAIRS,
            "pair_stations": MAX_PAIR_STATIONS,
            "queried_faces_per_station": MAX_QUERIED_FACES_PER_STATION,
            "section_nodes_per_station": MAX_SECTION_NODES_PER_STATION,
            "section_components_per_station": MAX_SECTION_COMPONENTS_PER_STATION,
            "face_plane_tests": MAX_FACE_PLANE_TESTS,
            "section_segments": MAX_SECTION_SEGMENTS,
            "wall_seconds": WALL_SECONDS,
        },
    }


CONFIGURATION_SHA256 = hashlib.sha256(canonical_bytes(frozen_configuration())).hexdigest()
