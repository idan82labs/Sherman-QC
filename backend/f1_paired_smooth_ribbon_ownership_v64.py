"""V64 paired smooth-ribbon ownership certificate (PSROC_V1).

This research-only module implements the frozen post-V63 deletion certificate.
It evaluates only disjoint mutual pairs from an unchanged V52 line set.  A pair
is removed only after repeated source-mesh sections positively certify one
connected, diffusely turning ribbon.  Missing or ambiguous source evidence can
never delete a line.

The public audit is intentionally compact: source vertices, triangles,
section polylines, and normal samples remain transient and are never returned.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROFILE = "V64_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE_V1"
SHORT_NAME = "PSROC_V1"
LOCK_TOKEN = "V64_PSROC_MECHANISM_LOCKED"
DELETION_REASON = "DELETED_PAIRED_SMOOTH_RIBBON_OWNERSHIP_CERTIFICATE"

MAXIMUM_SOURCE_VERTICES = 200_000
MAXIMUM_MUTUAL_PAIRS = 32
MAXIMUM_QUERIED_TRIANGLES_PER_STATION = 50_000
MAXIMUM_SECTION_CONSTRUCTIONS = 224
WALL_TIME_SECONDS = 90.0

STATION_FRACTIONS = tuple(index / 8.0 for index in range(1, 8))
MAXIMUM_DIRECTION_DIFFERENCE_DEG = 7.5
MINIMUM_LINE_LENGTH_OVER_H = 24.0
MINIMUM_OVERLAP_OVER_H = 20.0
MINIMUM_OVERLAP_FRACTION = 0.65
MINIMUM_SEPARATION_OVER_H = 10.0
MAXIMUM_SEPARATION_OVER_OVERLAP = 0.35
MAXIMUM_SEPARATION_ROBUST_CV = 0.15
MAXIMUM_TRANSVERSE_DIRECTION_DEVIATION_DEG = 12.0

SECTION_PLANE_OFFSET_OVER_D = 2.0**-40
COPLANAR_EPSILON_OVER_D = 2.0**-48
SECTION_SLAB_OVER_H = 3.0
SECTION_MARGIN_OVER_H = 4.0
ATTACHMENT_TOLERANCE_OVER_H = 2.0
MINIMUM_ARC_OVER_H = 10.0
MAXIMUM_ARC_OVER_SEPARATION = 1.75
FLANK_FRACTION = 0.25
PROFILE_SAMPLE_COUNT = 61
NORMAL_RADIUS_OVER_H = 1.5
MINIMUM_PROJECTED_NORMAL_MAGNITUDE = 0.75

MINIMUM_ROTATION_DEG = 18.0
MAXIMUM_ROTATION_DEG = 135.0
MAXIMUM_OPPOSITE_ROTATION_FRACTION = 0.08
MAXIMUM_SINGLE_INCREMENT_DEG = 6.0
MAXIMUM_SINGLE_INCREMENT_FRACTION = 0.18
MINIMUM_10_90_SUPPORT_FRACTION = 0.55
MINIMUM_25_75_SUPPORT_FRACTION = 0.30
MAXIMUM_SIX_INTERVAL_MASS_FRACTION = 0.40
MAXIMUM_JUMP_DEG = 6.0
MAXIMUM_JUMP_FRACTION = 0.16

MINIMUM_POSITIVE_STATIONS = 5
MAXIMUM_ROTATION_RATIO = 1.50
MAXIMUM_ARC_RATIO = 1.25
MAXIMUM_PHASE_PROFILE_DEVIATION = 0.18


class PSROCInputError(RuntimeError):
    """Frozen input contract is malformed."""


class PSROCResourceFailure(RuntimeError):
    """A scan-level resource failure; the unchanged V52 result must survive."""


@dataclass(frozen=True)
class Line:
    ordinal: int
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    original: Mapping[str, Any]


@dataclass(frozen=True)
class PairGeometry:
    first: Line
    second: Line
    direction: np.ndarray
    overlap_min: float
    overlap_max: float
    overlap_length: float
    station_first: tuple[np.ndarray, ...]
    station_second: tuple[np.ndarray, ...]
    separations: tuple[float, ...]
    transverse: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _BVHNode:
    lower: np.ndarray
    upper: np.ndarray
    left: int
    right: int
    faces: np.ndarray | None


@dataclass(frozen=True)
class StationArtifact:
    station_index: int
    state: str
    reason: str
    section_points: np.ndarray | None
    section_arclength: np.ndarray | None
    profile_points: np.ndarray | None
    phases: np.ndarray | None
    cumulative_positive: np.ndarray | None
    a_point: np.ndarray | None
    b_point: np.ndarray | None
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None


@dataclass(frozen=True)
class PairArtifact:
    first_ordinal: int
    second_ordinal: int
    certified: bool
    stations: tuple[StationArtifact, ...]


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if len(array) == 0:
        raise PSROCInputError("EMPTY_NEAREST_RANK")
    rank = max(1, int(math.ceil(float(probability) * len(array))))
    return float(array[rank - 1])


def _float_hex(value: float) -> str:
    native = float(value)
    if not math.isfinite(native):
        raise PSROCInputError("NONFINITE_AUDIT_VALUE")
    return native.hex()


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return _float_hex(float(value))
    if value is None or isinstance(value, str):
        return value
    raise PSROCInputError("UNSUPPORTED_AUDIT_VALUE")


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + suffix).encode("utf-8")


def frozen_configuration() -> dict[str, Any]:
    return {
        "profile": PROFILE,
        "short_name": SHORT_NAME,
        "lock_token": LOCK_TOKEN,
        "scope": {"deletion_only": True, "simultaneous_pair_deletion": True, "second_pass": False, "second_nms": False},
        "caps": {
            "source_vertices": MAXIMUM_SOURCE_VERTICES,
            "mutual_pairs": MAXIMUM_MUTUAL_PAIRS,
            "queried_triangles_per_station": MAXIMUM_QUERIED_TRIANGLES_PER_STATION,
            "section_constructions": MAXIMUM_SECTION_CONSTRUCTIONS,
            "wall_time_seconds": WALL_TIME_SECONDS,
        },
        "eligibility": {
            "station_fractions": STATION_FRACTIONS,
            "maximum_direction_difference_deg": MAXIMUM_DIRECTION_DIFFERENCE_DEG,
            "minimum_line_length_over_h": MINIMUM_LINE_LENGTH_OVER_H,
            "minimum_overlap_over_h": MINIMUM_OVERLAP_OVER_H,
            "minimum_overlap_fraction": MINIMUM_OVERLAP_FRACTION,
            "minimum_separation_over_h": MINIMUM_SEPARATION_OVER_H,
            "maximum_separation_over_overlap": MAXIMUM_SEPARATION_OVER_OVERLAP,
            "maximum_separation_robust_cv": MAXIMUM_SEPARATION_ROBUST_CV,
            "maximum_transverse_direction_deviation_deg": MAXIMUM_TRANSVERSE_DIRECTION_DEVIATION_DEG,
        },
        "certificate": {
            "minimum_rotation_deg": MINIMUM_ROTATION_DEG,
            "maximum_rotation_deg": MAXIMUM_ROTATION_DEG,
            "maximum_opposite_rotation_fraction": MAXIMUM_OPPOSITE_ROTATION_FRACTION,
            "maximum_single_increment_deg": MAXIMUM_SINGLE_INCREMENT_DEG,
            "maximum_single_increment_fraction": MAXIMUM_SINGLE_INCREMENT_FRACTION,
            "minimum_10_90_support_fraction": MINIMUM_10_90_SUPPORT_FRACTION,
            "minimum_25_75_support_fraction": MINIMUM_25_75_SUPPORT_FRACTION,
            "maximum_six_interval_mass_fraction": MAXIMUM_SIX_INTERVAL_MASS_FRACTION,
            "maximum_jump_deg": MAXIMUM_JUMP_DEG,
            "maximum_jump_fraction": MAXIMUM_JUMP_FRACTION,
            "minimum_positive_stations": MINIMUM_POSITIVE_STATIONS,
            "maximum_rotation_ratio": MAXIMUM_ROTATION_RATIO,
            "maximum_arc_ratio": MAXIMUM_ARC_RATIO,
            "maximum_phase_profile_deviation": MAXIMUM_PHASE_PROFILE_DEVIATION,
        },
    }


CONFIGURATION_SHA256 = hashlib.sha256(canonical_bytes(frozen_configuration())).hexdigest()


def _line(row: Mapping[str, Any], ordinal: int) -> Line:
    try:
        start = np.asarray(row["line_start"], dtype=np.float64)
        end = np.asarray(row["line_end"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise PSROCInputError("INVALID_V52_LINE") from error
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise PSROCInputError("INVALID_V52_LINE")
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 0.0:
        raise PSROCInputError("INVALID_V52_LINE")
    return Line(int(ordinal), start, end, delta / length, length, row)


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PSROCInputError("INVALID_DIRECTION")
    result = result / norm
    for value in result:
        if value > 0.0:
            break
        if value < 0.0:
            result = -result
            break
    result[result == 0.0] = 0.0
    return result


def _point_at_axis_coordinate(line: Line, direction: np.ndarray, coordinate: float) -> np.ndarray:
    denominator = float(np.dot(line.direction, direction))
    if abs(denominator) <= 1e-12:
        raise PSROCInputError("PAIR_AXIS_DEGENERATE")
    distance = (coordinate - float(np.dot(line.start, direction))) / denominator
    return line.start + distance * line.direction


def _acute_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(float(np.dot(first, second)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _pair_geometry(first: Line, second: Line, h: float) -> PairGeometry | None:
    if first.length < MINIMUM_LINE_LENGTH_OVER_H * h or second.length < MINIMUM_LINE_LENGTH_OVER_H * h:
        return None
    if _acute_angle_deg(first.direction, second.direction) > MAXIMUM_DIRECTION_DIFFERENCE_DEG:
        return None
    second_direction = second.direction if float(np.dot(first.direction, second.direction)) >= 0.0 else -second.direction
    direction = _canonical_direction(first.direction + second_direction)
    first_projection = np.dot(np.stack((first.start, first.end)), direction)
    second_projection = np.dot(np.stack((second.start, second.end)), direction)
    overlap_min = max(float(np.min(first_projection)), float(np.min(second_projection)))
    overlap_max = min(float(np.max(first_projection)), float(np.max(second_projection)))
    overlap_length = overlap_max - overlap_min
    if overlap_length < max(MINIMUM_OVERLAP_OVER_H * h, MINIMUM_OVERLAP_FRACTION * min(first.length, second.length)):
        return None
    station_first: list[np.ndarray] = []
    station_second: list[np.ndarray] = []
    separations: list[float] = []
    transverse: list[np.ndarray] = []
    for fraction in STATION_FRACTIONS:
        coordinate = overlap_min + fraction * overlap_length
        a = _point_at_axis_coordinate(first, direction, coordinate)
        b = _point_at_axis_coordinate(second, direction, coordinate)
        delta = b - a
        delta = delta - float(np.dot(delta, direction)) * direction
        width = float(np.linalg.norm(delta))
        if not math.isfinite(width) or width <= 0.0:
            return None
        station_first.append(a)
        station_second.append(b)
        separations.append(width)
        transverse.append(delta / width)
    width = float(np.median(np.asarray(separations, dtype=np.float64)))
    if not (MINIMUM_SEPARATION_OVER_H * h <= width <= MAXIMUM_SEPARATION_OVER_OVERLAP * overlap_length):
        return None
    mad = float(np.median(np.abs(np.asarray(separations) - width)))
    if mad / width > MAXIMUM_SEPARATION_ROBUST_CV:
        return None
    components = np.asarray(transverse, dtype=np.float64)
    median_direction = _canonical_direction(np.median(components, axis=0))
    oriented = []
    for vector in components:
        oriented.append(vector if float(np.dot(vector, median_direction)) >= 0.0 else -vector)
    for vector in oriented:
        angle = math.degrees(math.acos(float(np.clip(np.dot(vector, median_direction), -1.0, 1.0))))
        if angle > MAXIMUM_TRANSVERSE_DIRECTION_DEVIATION_DEG:
            return None
    return PairGeometry(
        first=first,
        second=second,
        direction=direction,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        overlap_length=overlap_length,
        station_first=tuple(station_first),
        station_second=tuple(station_second),
        separations=tuple(separations),
        transverse=tuple(oriented),
    )


class MeshIndex:
    """Deterministically oriented source mesh with a bounded BVH."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        raw_triangles = np.asarray(triangles, dtype=np.int64)
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1:] != (3,)
            or len(self.vertices) == 0
            or len(self.vertices) > MAXIMUM_SOURCE_VERTICES
            or not np.isfinite(self.vertices).all()
            or raw_triangles.ndim != 2
            or raw_triangles.shape[1:] != (3,)
            or len(raw_triangles) == 0
            or np.any(raw_triangles < 0)
            or np.any(raw_triangles >= len(self.vertices))
        ):
            raise PSROCInputError("INVALID_SOURCE_MESH")
        self.bbox_diagonal = float(np.linalg.norm(np.ptp(self.vertices, axis=0)))
        if not math.isfinite(self.bbox_diagonal) or self.bbox_diagonal <= 0.0:
            raise PSROCInputError("INVALID_SOURCE_SCALE")

        all_edges = np.sort(
            np.concatenate(
                (raw_triangles[:, (0, 1)], raw_triangles[:, (1, 2)], raw_triangles[:, (2, 0)]),
                axis=0,
            ),
            axis=1,
        )
        unique_edges = np.unique(all_edges, axis=0)
        edge_lengths = np.linalg.norm(self.vertices[unique_edges[:, 1]] - self.vertices[unique_edges[:, 0]], axis=1)
        edge_lengths = np.sort(edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)])
        if len(edge_lengths) < 1024:
            raise PSROCInputError("INSUFFICIENT_VALID_EDGES")
        trim = int(math.ceil(0.05 * len(edge_lengths)))
        trimmed = edge_lengths[trim : len(edge_lengths) - trim]
        if len(trimmed) == 0:
            raise PSROCInputError("INSUFFICIENT_VALID_EDGES")
        self.h = _nearest_rank(trimmed, 0.5)
        if self.h < (2.0**-40) * self.bbox_diagonal:
            raise PSROCInputError("INVALID_SOURCE_SCALE")

        edge_to_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for face_id, face in enumerate(raw_triangles):
            for local, (left, right) in enumerate(((0, 1), (1, 2), (2, 0))):
                u, v = int(face[left]), int(face[right])
                edge_to_faces[(min(u, v), max(u, v))].append((int(face_id), u, v))
        adjacency: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for incident in edge_to_faces.values():
            if len(incident) != 2:
                continue
            (f0, u0, v0), (f1, u1, v1) = incident
            adjacency[f0].append((f1, u0, v0, 1 if (u0 == u1 and v0 == v1) else 0))
            adjacency[f1].append((f0, u1, v1, 1 if (u0 == u1 and v0 == v1) else 0))

        flips: dict[int, bool] = {}
        components = np.full(len(raw_triangles), -1, dtype=np.int64)
        unusable_components: set[int] = set()
        component_id = 0
        for root in range(len(raw_triangles)):
            if root in flips:
                continue
            flips[root] = False
            components[root] = component_id
            queue = deque([root])
            while queue:
                face = queue.popleft()
                for neighbor, _u, _v, same_direction in sorted(adjacency.get(face, ()), key=lambda item: item[0]):
                    required = bool(flips[face]) ^ bool(same_direction)
                    if neighbor not in flips:
                        flips[neighbor] = required
                        components[neighbor] = component_id
                        queue.append(neighbor)
                    elif flips[neighbor] != required:
                        unusable_components.add(component_id)
            component_id += 1
        oriented = raw_triangles.copy()
        for face_id, flip in flips.items():
            if flip:
                oriented[face_id, (1, 2)] = oriented[face_id, (2, 1)]
        self.triangles = oriented
        self.face_components = components
        self.unusable_components = frozenset(unusable_components)
        p0 = self.vertices[oriented[:, 0]]
        p1 = self.vertices[oriented[:, 1]]
        p2 = self.vertices[oriented[:, 2]]
        raw_normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(raw_normals, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise PSROCInputError("DEGENERATE_SOURCE_FACE")
        self.face_normals = raw_normals / lengths[:, None]

        oriented_edges = np.sort(
            np.stack((oriented[:, (0, 1)], oriented[:, (1, 2)], oriented[:, (2, 0)]), axis=1),
            axis=2,
        )
        unique_oriented, inverse = np.unique(oriented_edges.reshape(-1, 2), axis=0, return_inverse=True)
        self.edge_ordinals = inverse.reshape(len(oriented), 3)
        self.unique_edges = unique_oriented

        lower = np.minimum(np.minimum(p0, p1), p2)
        upper = np.maximum(np.maximum(p0, p1), p2)
        centroids = (p0 + p1 + p2) / 3.0
        nodes: list[_BVHNode] = []

        def build(face_ids: np.ndarray) -> int:
            node_lower = np.min(lower[face_ids], axis=0)
            node_upper = np.max(upper[face_ids], axis=0)
            index = len(nodes)
            nodes.append(_BVHNode(node_lower, node_upper, -1, -1, None))
            if len(face_ids) <= 16:
                nodes[index] = _BVHNode(node_lower, node_upper, -1, -1, np.sort(face_ids))
                return index
            span = node_upper - node_lower
            axis = min((axis for axis in range(3) if span[axis] == np.max(span)), default=0)
            order = np.lexsort((face_ids, centroids[face_ids, axis]))
            ordered = face_ids[order]
            midpoint = len(ordered) // 2
            left_node = build(ordered[:midpoint])
            right_node = build(ordered[midpoint:])
            nodes[index] = _BVHNode(node_lower, node_upper, left_node, right_node, None)
            return index

        self.root = build(np.arange(len(oriented), dtype=np.int64))
        self.nodes = tuple(nodes)

    def query_prism(self, midpoint: np.ndarray, t: np.ndarray, c: np.ndarray, b: np.ndarray, width: float) -> np.ndarray:
        limits = (SECTION_SLAB_OVER_H * self.h, width + SECTION_MARGIN_OVER_H * self.h, width + SECTION_MARGIN_OVER_H * self.h)
        axes = (t, c, b)
        result: list[int] = []
        stack = [self.root]
        while stack:
            node_id = stack.pop()
            node = self.nodes[node_id]
            center = 0.5 * (node.lower + node.upper)
            half = 0.5 * (node.upper - node.lower)
            outside = False
            for axis, limit in zip(axes, limits):
                distance = abs(float(np.dot(center - midpoint, axis)))
                radius = float(np.dot(np.abs(axis), half))
                if distance > radius + limit:
                    outside = True
                    break
            if outside:
                continue
            if node.faces is not None:
                result.extend(int(value) for value in node.faces)
                if len(result) > MAXIMUM_QUERIED_TRIANGLES_PER_STATION:
                    raise PSROCResourceFailure("TRIANGLE_QUERY_CAP")
            else:
                stack.append(node.right)
                stack.append(node.left)
        return np.asarray(sorted(result), dtype=np.int64)


def _segment_projection(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float, np.ndarray]:
    delta = second - first
    denominator = float(np.dot(delta, delta))
    fraction = 0.0 if denominator <= 0.0 else float(np.clip(np.dot(point - first, delta) / denominator, 0.0, 1.0))
    nearest = first + fraction * delta
    return float(np.linalg.norm(point - nearest)), fraction, nearest


def _sample_polyline(points: np.ndarray, cumulative: np.ndarray, position: float) -> np.ndarray:
    position = float(np.clip(position, 0.0, float(cumulative[-1])))
    index = int(np.searchsorted(cumulative, position, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    length = float(cumulative[index + 1] - cumulative[index])
    fraction = 0.0 if length <= 0.0 else (position - float(cumulative[index])) / length
    return points[index] + fraction * (points[index + 1] - points[index])


def _section_component(
    mesh: MeshIndex,
    pair: PairGeometry,
    station_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    a = pair.station_first[station_index]
    b_line = pair.station_second[station_index]
    width = pair.separations[station_index]
    t = pair.direction
    c = pair.transverse[station_index]
    b_axis = np.cross(t, c)
    b_axis = b_axis / np.linalg.norm(b_axis)
    midpoint = 0.5 * (a + b_line)
    plane_point = midpoint + SECTION_PLANE_OFFSET_OVER_D * mesh.bbox_diagonal * t
    epsilon = COPLANAR_EPSILON_OVER_D * mesh.bbox_diagonal
    face_ids = mesh.query_prism(midpoint, t, c, b_axis, width)
    if len(face_ids) == 0:
        raise PSROCInputError("MISSING_NO_SECTION")

    node_points: dict[tuple[str, int], np.ndarray] = {}
    segments: list[tuple[tuple[str, int], tuple[str, int], int]] = []
    for face_id in face_ids:
        vertex_ids = mesh.triangles[face_id]
        points = mesh.vertices[vertex_ids]
        signed = np.dot(points - plane_point, t)
        if np.max(np.abs(signed)) <= epsilon or int(np.count_nonzero(np.abs(signed) <= epsilon)) >= 2:
            raise PSROCInputError("MISSING_COPLANAR_SECTION")
        signed[np.abs(signed) <= epsilon] = 0.0
        emitted: dict[tuple[str, int], np.ndarray] = {}
        for local_vertex in np.flatnonzero(signed == 0.0):
            emitted[("v", int(vertex_ids[local_vertex]))] = points[local_vertex]
        for local_edge, (left, right) in enumerate(((0, 1), (1, 2), (2, 0))):
            dl, dr = float(signed[left]), float(signed[right])
            if dl * dr < 0.0:
                fraction = dl / (dl - dr)
                emitted[("e", int(mesh.edge_ordinals[face_id, local_edge]))] = points[left] + fraction * (points[right] - points[left])
        if len(emitted) <= 1:
            continue
        if len(emitted) != 2:
            raise PSROCInputError("MISSING_UNRESOLVED_SECTION")
        keys = sorted(emitted, key=lambda key: (0 if key[0] == "v" else 1, key[1]))
        for key in keys:
            if key in node_points and not np.allclose(node_points[key], emitted[key], rtol=0.0, atol=epsilon):
                raise PSROCInputError("MISSING_INCONSISTENT_SECTION")
            node_points[key] = emitted[key]
        segments.append((keys[0], keys[1], int(face_id)))
    if not segments:
        raise PSROCInputError("MISSING_NO_SECTION")

    node_to_segments: dict[tuple[str, int], list[int]] = defaultdict(list)
    for segment_id, (left, right, _face) in enumerate(segments):
        node_to_segments[left].append(segment_id)
        node_to_segments[right].append(segment_id)
    remaining = set(range(len(segments)))
    candidates = []
    while remaining:
        seed = min(remaining)
        component: set[int] = set()
        queue = deque([seed])
        while queue:
            segment_id = queue.popleft()
            if segment_id in component:
                continue
            component.add(segment_id)
            for key in segments[segment_id][:2]:
                queue.extend(sorted(node_to_segments[key]))
        remaining.difference_update(component)
        nodes = {key for segment_id in component for key in segments[segment_id][:2]}
        degrees = {key: sum(1 for segment_id in node_to_segments[key] if segment_id in component) for key in nodes}
        endpoints = sorted((key for key, degree in degrees.items() if degree == 1), key=lambda key: (key[0], key[1]))
        if any(degree > 2 or degree <= 0 for degree in degrees.values()) or len(endpoints) != 2:
            continue
        component_ids = {int(mesh.face_components[segments[segment_id][2]]) for segment_id in component}
        if len(component_ids) != 1 or next(iter(component_ids)) in mesh.unusable_components:
            continue
        ordered_keys = [endpoints[0]]
        ordered_segments: list[int] = []
        current = endpoints[0]
        previous = None
        while current != endpoints[1]:
            options = sorted(
                segment_id for segment_id in node_to_segments[current]
                if segment_id in component and segment_id != previous
            )
            if not options:
                break
            segment_id = options[0]
            left, right, _face = segments[segment_id]
            next_key = right if left == current else left
            ordered_segments.append(segment_id)
            ordered_keys.append(next_key)
            previous = segment_id
            current = next_key
        if set(ordered_segments) != component:
            continue
        points = np.asarray([node_points[key] for key in ordered_keys], dtype=np.float64)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 0.0) or not np.isfinite(lengths).all():
            continue
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        projections = []
        for target in (a, b_line):
            best = None
            for index in range(len(points) - 1):
                distance, fraction, nearest = _segment_projection(target, points[index], points[index + 1])
                option = (distance, index, fraction, nearest)
                if best is None or option[:3] < best[:3]:
                    best = option
            assert best is not None
            distance, index, fraction, nearest = best
            position = float(cumulative[index] + fraction * lengths[index])
            projections.append((distance, position, nearest))
        if max(projections[0][0], projections[1][0]) > ATTACHMENT_TOLERANCE_OVER_H * mesh.h:
            continue
        first_position, second_position = projections[0][1], projections[1][1]
        g = abs(second_position - first_position)
        if not (MINIMUM_ARC_OVER_H * mesh.h <= g <= MAXIMUM_ARC_OVER_SEPARATION * width):
            continue
        lower_position = min(first_position, second_position)
        upper_position = max(first_position, second_position)
        if lower_position < FLANK_FRACTION * g or float(cumulative[-1]) - upper_position < FLANK_FRACTION * g:
            continue
        minimum_face = min(segments[segment_id][2] for segment_id in component)
        score = (projections[0][0] ** 2 + projections[1][0] ** 2, max(projections[0][0], projections[1][0]), minimum_face)
        candidates.append((score, points, cumulative, ordered_segments, segments, first_position, second_position, g))
    if not candidates:
        raise PSROCInputError("MISSING_NO_COMMON_OPEN_COMPONENT")
    _, points, cumulative, ordered_segments, segments, first_position, second_position, g = min(candidates, key=lambda row: row[0])
    return points, cumulative, np.asarray(ordered_segments, dtype=np.int64), np.asarray([segments[s][2] for s in ordered_segments], dtype=np.int64), plane_point, first_position, second_position, g


def _percentile_location(mass: np.ndarray, fraction: float) -> float:
    total = float(np.sum(mass))
    target = fraction * total
    cumulative = 0.0
    for index, value in enumerate(mass):
        next_value = cumulative + float(value)
        if next_value >= target:
            part = 0.0 if value <= 0.0 else (target - cumulative) / float(value)
            return float(index + part)
        cumulative = next_value
    return float(len(mass))


def _station(
    mesh: MeshIndex,
    pair: PairGeometry,
    station_index: int,
) -> tuple[dict[str, Any], StationArtifact]:
    try:
        points, cumulative, ordered_segments, face_ids, plane_point, a_position, b_position, g = _section_component(mesh, pair, station_index)
    except PSROCInputError as error:
        reason = str(error)
        if not reason.startswith("MISSING_"):
            raise
        record = {"state": "MISSING", "reason": reason, "station_index": station_index}
        artifact = StationArtifact(station_index, "MISSING", reason, None, None, None, None, None, pair.station_first[station_index], pair.station_second[station_index], None, pair.direction)
        return record, artifact

    lower = min(a_position, b_position) - FLANK_FRACTION * g
    upper = max(a_position, b_position) + FLANK_FRACTION * g
    profile_positions = np.linspace(lower, upper, PROFILE_SAMPLE_COUNT, dtype=np.float64)
    profile_points = np.asarray([_sample_polyline(points, cumulative, value) for value in profile_positions])
    segment_lengths = np.diff(cumulative)
    normals = []
    for position in profile_positions:
        along_distance = np.maximum(
            np.maximum(cumulative[:-1] - position, 0.0),
            position - cumulative[1:],
        )
        weights = np.maximum(
            0.0,
            1.0 - along_distance / (NORMAL_RADIUS_OVER_H * mesh.h),
        ) * segment_lengths
        valid = weights > 0.0
        if not np.any(valid):
            reason = "MISSING_NORMAL_SUPPORT"
            record = {"state": "MISSING", "reason": reason, "station_index": station_index}
            artifact = StationArtifact(station_index, "MISSING", reason, points, cumulative, profile_points, None, None, pair.station_first[station_index], pair.station_second[station_index], plane_point, pair.direction)
            return record, artifact
        normal = np.sum(mesh.face_normals[face_ids[valid]] * weights[valid, None], axis=0) / float(np.sum(weights[valid]))
        projected = normal - float(np.dot(normal, pair.direction)) * pair.direction
        magnitude = float(np.linalg.norm(projected))
        if not math.isfinite(magnitude) or magnitude < MINIMUM_PROJECTED_NORMAL_MAGNITUDE:
            reason = "MISSING_NORMAL_PROJECTION"
            record = {"state": "MISSING", "reason": reason, "station_index": station_index}
            artifact = StationArtifact(station_index, "MISSING", reason, points, cumulative, profile_points, None, None, pair.station_first[station_index], pair.station_second[station_index], plane_point, pair.direction)
            return record, artifact
        normals.append(projected / magnitude)
    normals_array = np.asarray(normals, dtype=np.float64)
    c = pair.transverse[station_index]
    b_axis = np.cross(pair.direction, c)
    phases = np.unwrap(np.arctan2(normals_array @ b_axis, normals_array @ c))
    inner_delta = np.diff(phases[10:51])
    signed = float(np.sum(inner_delta))
    rotation = abs(math.degrees(signed))
    sigma = 1.0 if signed > 0.0 else -1.0 if signed < 0.0 else 0.0
    positive = np.maximum(0.0, sigma * np.degrees(inner_delta))
    opposite = np.maximum(0.0, -sigma * np.degrees(inner_delta))
    cumulative_positive = np.concatenate(([0.0], np.cumsum(positive)))
    reasons = []
    if not (MINIMUM_ROTATION_DEG <= rotation <= MAXIMUM_ROTATION_DEG):
        reasons.append("ROTATION_RANGE")
    if float(np.sum(opposite)) > MAXIMUM_OPPOSITE_ROTATION_FRACTION * rotation:
        reasons.append("OPPOSITE_ROTATION")
    if len(inner_delta) and float(np.max(np.abs(np.degrees(inner_delta)))) > max(MAXIMUM_SINGLE_INCREMENT_DEG, MAXIMUM_SINGLE_INCREMENT_FRACTION * rotation):
        reasons.append("SINGLE_INCREMENT")
    if float(np.sum(positive)) <= 0.0:
        reasons.append("NO_POSITIVE_ROTATION")
        locations = [0.0, 0.0, 0.0, 0.0]
    else:
        locations = [_percentile_location(positive, value) for value in (0.10, 0.25, 0.75, 0.90)]
        if (locations[3] - locations[0]) / 40.0 < MINIMUM_10_90_SUPPORT_FRACTION:
            reasons.append("NARROW_10_90_SUPPORT")
        if (locations[2] - locations[1]) / 40.0 < MINIMUM_25_75_SUPPORT_FRACTION:
            reasons.append("NARROW_25_75_SUPPORT")
        largest_six = max((float(np.sum(positive[index : index + 6])) for index in range(35)), default=0.0)
        if largest_six > MAXIMUM_SIX_INTERVAL_MASS_FRACTION * float(np.sum(positive)):
            reasons.append("CONCENTRATED_SIX_INTERVAL_MASS")
    phases_deg = np.degrees(phases)
    jump_a = abs(float(np.median(phases_deg[[7, 8, 9]]) - np.median(phases_deg[[11, 12, 13]])))
    jump_b = abs(float(np.median(phases_deg[[47, 48, 49]]) - np.median(phases_deg[[51, 52, 53]])))
    jump_limit = max(MAXIMUM_JUMP_DEG, MAXIMUM_JUMP_FRACTION * rotation)
    if jump_a > jump_limit:
        reasons.append("CANDIDATE_A_JUMP")
    if jump_b > jump_limit:
        reasons.append("CANDIDATE_B_JUMP")
    state = "POSITIVE" if not reasons else "CONTRADICTORY"
    reason = "PASS" if not reasons else reasons[0]
    record = {
        "state": state,
        "reason": reason,
        "station_index": station_index,
        "rotation_deg": rotation,
        "rotation_sign": int(sigma),
        "arc_length": g,
        "jump_a_deg": jump_a,
        "jump_b_deg": jump_b,
        "locations": locations,
        "phase_profile": [float(np.interp(value, np.arange(41, dtype=np.float64), cumulative_positive) / cumulative_positive[-1]) if cumulative_positive[-1] > 0.0 else 0.0 for value in np.linspace(0.0, 40.0, 9)],
    }
    artifact = StationArtifact(station_index, state, reason, points, cumulative, profile_points, phases, cumulative_positive, pair.station_first[station_index], pair.station_second[station_index], plane_point, pair.direction)
    return record, artifact


def evaluate_pair(mesh: MeshIndex, pair: PairGeometry) -> tuple[dict[str, Any], PairArtifact]:
    stations = []
    artifacts = []
    for index in range(7):
        record, artifact = _station(mesh, pair, index)
        stations.append(record)
        artifacts.append(artifact)
    positives = [row for row in stations if row["state"] == "POSITIVE"]
    contradictory = [row for row in stations if row["state"] == "CONTRADICTORY"]
    reasons = []
    if len(positives) < MINIMUM_POSITIVE_STATIONS:
        reasons.append("INSUFFICIENT_POSITIVE_STATIONS")
    if contradictory:
        reasons.append("CONTRADICTORY_STATION")
    if positives:
        if len({int(row["rotation_sign"]) for row in positives}) != 1:
            reasons.append("ROTATION_SIGN_INCONSISTENT")
        rotations = [float(row["rotation_deg"]) for row in positives]
        arcs = [float(row["arc_length"]) for row in positives]
        if max(rotations) / min(rotations) > MAXIMUM_ROTATION_RATIO:
            reasons.append("ROTATION_INCONSISTENT")
        if max(arcs) / min(arcs) > MAXIMUM_ARC_RATIO:
            reasons.append("ARC_INCONSISTENT")
        profiles = np.asarray([row["phase_profile"] for row in positives], dtype=np.float64)
        median_profile = np.asarray([_nearest_rank(profiles[:, column], 0.5) for column in range(profiles.shape[1])])
        if np.any(np.max(np.abs(profiles - median_profile[None, :]), axis=1) > MAXIMUM_PHASE_PROFILE_DEVIATION):
            reasons.append("PHASE_PROFILE_INCONSISTENT")
    certified = not reasons
    record = {
        "first_ordinal": pair.first.ordinal,
        "second_ordinal": pair.second.ordinal,
        "certified": certified,
        "reason": DELETION_REASON if certified else reasons[0],
        "positive_station_count": len(positives),
        "contradictory_station_count": len(contradictory),
        "stations": stations,
    }
    return record, PairArtifact(pair.first.ordinal, pair.second.ordinal, certified, tuple(artifacts))


def _mutual_pairs(lines: Sequence[Line], h: float) -> list[PairGeometry]:
    geometry: dict[tuple[int, int], PairGeometry] = {}
    partners: dict[int, list[tuple[tuple[float, float, int], int]]] = defaultdict(list)
    for first_index in range(len(lines)):
        for second_index in range(first_index + 1, len(lines)):
            pair = _pair_geometry(lines[first_index], lines[second_index], h)
            if pair is None:
                continue
            key = (pair.first.ordinal, pair.second.ordinal)
            geometry[key] = pair
            width = float(np.median(np.asarray(pair.separations)))
            first_rank = (pair.overlap_length / width, -pair.overlap_length / min(pair.first.length, pair.second.length), pair.second.ordinal)
            second_rank = (pair.overlap_length / width, -pair.overlap_length / min(pair.first.length, pair.second.length), pair.first.ordinal)
            partners[pair.first.ordinal].append((first_rank, pair.second.ordinal))
            partners[pair.second.ordinal].append((second_rank, pair.first.ordinal))
    choices = {ordinal: min(options)[1] for ordinal, options in partners.items() if options}
    result = []
    for (first, second), pair in sorted(geometry.items()):
        if choices.get(first) == second and choices.get(second) == first:
            result.append(pair)
    if len(result) > MAXIMUM_MUTUAL_PAIRS:
        raise PSROCResourceFailure("MUTUAL_PAIR_CAP")
    return result


def apply_psroc_v1(
    baseline: Mapping[str, Any],
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    started: float | None = None,
) -> tuple[dict[str, Any], tuple[PairArtifact, ...]]:
    """Apply the locked deletion-only certificate to one unchanged V52 output."""

    started = time.perf_counter() if started is None else float(started)
    rows = list(baseline.get("predictions", ()))
    lines = [_line(row, ordinal) for ordinal, row in enumerate(rows)]
    try:
        mesh = MeshIndex(vertices, triangles)
        pairs = _mutual_pairs(lines, mesh.h)
        if len(pairs) * len(STATION_FRACTIONS) > MAXIMUM_SECTION_CONSTRUCTIONS:
            raise PSROCResourceFailure("SECTION_CONSTRUCTION_CAP")
        pair_records = []
        artifacts = []
        deleted: set[int] = set()
        for pair in pairs:
            if time.perf_counter() - started >= WALL_TIME_SECONDS:
                raise PSROCResourceFailure("WATCHDOG")
            record, artifact = evaluate_pair(mesh, pair)
            pair_records.append(record)
            artifacts.append(artifact)
            if record["certified"]:
                deleted.update((pair.first.ordinal, pair.second.ordinal))
        survivors = [deepcopy(row) for ordinal, row in enumerate(rows) if ordinal not in deleted]
        audit = {
            "profile": PROFILE,
            "lock_token": LOCK_TOKEN,
            "configuration_sha256": CONFIGURATION_SHA256,
            "status": "OK",
            "h": mesh.h,
            "bbox_diagonal": mesh.bbox_diagonal,
            "mutual_pair_count": len(pairs),
            "deleted_ordinals": sorted(deleted),
            "deletion_reason": DELETION_REASON,
            "second_pass_used": False,
            "second_nms_used": False,
            "pair_records": pair_records,
        }
        result = deepcopy(dict(baseline))
        result["predictions"] = survivors
        result["psroc_v1_audit"] = _native(audit)
        result["psroc_v1_audit_sha256"] = hashlib.sha256(canonical_bytes(audit)).hexdigest()
        return result, tuple(artifacts)
    except (PSROCResourceFailure, MemoryError) as error:
        result = deepcopy(dict(baseline))
        result["predictions"] = [deepcopy(row) for row in rows]
        audit = {
            "profile": PROFILE,
            "lock_token": LOCK_TOKEN,
            "configuration_sha256": CONFIGURATION_SHA256,
            "status": "RESOURCE_FAILURE_UNCHANGED",
            "reason": str(error) if not isinstance(error, MemoryError) else "ALLOCATION_FAILURE",
            "deleted_ordinals": [],
        }
        result["psroc_v1_audit"] = _native(audit)
        result["psroc_v1_audit_sha256"] = hashlib.sha256(canonical_bytes(audit)).hexdigest()
        return result, tuple()
