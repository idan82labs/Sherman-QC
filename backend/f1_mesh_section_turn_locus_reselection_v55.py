"""Frozen V55 mesh-section turn-locus reselection.

This module implements the one-shot, source-only MSTLR intervention over the
immutable V52 predictions and frozen candidate pool.  It measures the
transverse location of tangent rotation in deterministic source-mesh sections;
it never imports V53/V54 state and never creates or refits line geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import open3d as o3d


PROFILE = "V55_MESH_SECTION_TURN_LOCUS_RESELECTION_V1"
SHORT_NAME = "MSTLR_V1"
PRECOMMIT_TOKEN = "V55_MSTLR_MECHANISM_LOCKED"
TERMINAL_FAILURE_TOKEN = "V55_MSTLR_TERMINAL_RETIRE"

STATION_FRACTIONS = tuple(0.10 * value for value in range(1, 10))
SECTION_EPSILON_FACTOR = 2.0**-40
PROFILE_POINT_COUNT = 65
MINIMUM_SECTION_SEGMENTS_PER_SIDE = 16
MINIMUM_FLANK_TURN_DEG = 10.0
MAXIMUM_FLANK_TURN_DEG = 170.0
MINIMUM_SIGNED_TURN_COHERENCE = 0.85
MINIMUM_CENTERED_TURN_MASS = 0.45
MAXIMUM_OFFSET_AXIS_TURN_MASS = 0.10
MAXIMUM_OFFSET_AXIS_ABSOLUTE_TURN_DEG = 3.0
MAXIMUM_TURN_WIDTH_OVER_R = 0.55
MAXIMUM_TURN_LOCUS_DIRECTION_ERROR_DEG = 2.5
MINIMUM_ADDITION_USABLE_STATIONS = 7
MINIMUM_ADDITION_CENTERED_STATIONS = 7
MINIMUM_ADDITION_CENTERED_RUN = 5
MINIMUM_PRUNING_OFFSET_STATIONS = 8
MINIMUM_PRUNING_OFFSET_RUN = 7
MINIMUM_OFFSET_DIRECTION_CONCENTRATION = 0.95
MAXIMUM_OFFSET_Q90_Q10 = 1.50
CONFLICT_DIRECTION_ANGLE_DEG = 3.0
MINIMUM_CONFLICT_OVERLAP = 0.50
BVH_LEAF_SIZE = 16
MAXIMUM_INTERSECTING_TRIANGLES = 8_192
MAXIMUM_CANDIDATES = 2_048
MAXIMUM_PUBLIC_BYTES = 2 * 1024 * 1024
RSS_LIMIT_BYTES = int(15.5 * 1024**3)
WALL_LIMIT_SECONDS = 285.0


class V55ScanAbstention(RuntimeError):
    """A scan-level failure that terminally retires the sealed V55 lane."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


class _StationUnusable(RuntimeError):
    """A local candidate/station abstention.  It never terminates a scan."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


@dataclass(frozen=True)
class Scale:
    bbox_diagonal: float
    mesh_spacing: float

    def values(self, line_length: float) -> dict[str, float]:
        b = self.bbox_diagonal
        h = self.mesh_spacing
        minimum_line_span = max(24.0 * h, 0.10 * b)
        radius = min(0.12 * b, max(12.0 * h, 0.08 * line_length))
        return {
            "minimum_line_span": minimum_line_span,
            "radius": radius,
            "profile_half_span": 0.70 * radius,
            "section_epsilon": SECTION_EPSILON_FACTOR * b,
            "attachment_tolerance": max(3.0 * h, 0.04 * radius),
            "flank_rms_limit": max(1.5 * h, 0.02 * radius),
            "center_tolerance": max(3.0 * h, 0.08 * radius),
            "center_q90_limit": max(4.0 * h, 0.12 * radius),
            "definite_offset_threshold": max(6.0 * h, 0.20 * radius),
            "maximum_offset": 0.75 * radius,
            "turn_locus_rms_limit": max(3.0 * h, 0.05 * radius),
            "conflict_separation": max(4.0 * h, 0.015 * b),
        }


@dataclass(frozen=True)
class _BVHNode:
    lower: np.ndarray
    upper: np.ndarray
    left: int
    right: int
    faces: np.ndarray | None


@dataclass(frozen=True)
class MeshSectionIndex:
    vertices: np.ndarray
    triangles: np.ndarray
    face_edge_ordinals: np.ndarray
    nodes: tuple[_BVHNode, ...]
    root: int
    scale: Scale

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
        *,
        scale_override: tuple[float, float] | None = None,
    ) -> "MeshSectionIndex":
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or len(vertices) == 0
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or len(triangles) == 0
            or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))
        ):
            raise V55ScanAbstention("ABSTAIN_INVALID_SOURCE_MESH")
        p0 = vertices[triangles[:, 0]]
        p1 = vertices[triangles[:, 1]]
        p2 = vertices[triangles[:, 2]]
        area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        if not np.isfinite(area2).all() or np.any(area2 <= 0.0):
            raise V55ScanAbstention("ABSTAIN_INVALID_SOURCE_MESH")

        raw_edges = np.stack(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            ),
            axis=1,
        )
        raw_edges = np.sort(raw_edges, axis=2)
        unique_edges, inverse = np.unique(
            raw_edges.reshape(-1, 2), axis=0, return_inverse=True
        )
        edge_lengths = np.linalg.norm(
            vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]], axis=1
        )
        eligible = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
        if len(eligible) < 100:
            raise V55ScanAbstention("ABSTAIN_INVALID_MESH_SPACING")
        lower_median_index = (len(eligible) - 1) // 2
        mesh_spacing = float(np.partition(eligible, lower_median_index)[lower_median_index])
        bbox_span = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        bbox_diagonal = float(np.linalg.norm(bbox_span))
        if scale_override is not None:
            bbox_diagonal, mesh_spacing = map(float, scale_override)
        if (
            not math.isfinite(bbox_diagonal)
            or bbox_diagonal <= 0.0
            or not math.isfinite(mesh_spacing)
            or mesh_spacing <= 0.0
            or not (1e-6 <= mesh_spacing / bbox_diagonal <= 0.05)
        ):
            raise V55ScanAbstention("ABSTAIN_INVALID_GLOBAL_SCALE")

        face_lower = np.minimum(np.minimum(p0, p1), p2)
        face_upper = np.maximum(np.maximum(p0, p1), p2)
        centroids = (p0 + p1 + p2) / 3.0
        built: list[_BVHNode] = []

        def build(face_ids: np.ndarray) -> int:
            lower = np.min(face_lower[face_ids], axis=0)
            upper = np.max(face_upper[face_ids], axis=0)
            node_index = len(built)
            built.append(_BVHNode(lower, upper, -1, -1, None))
            if len(face_ids) <= BVH_LEAF_SIZE:
                ordered = np.sort(face_ids.astype(np.int64, copy=False))
                built[node_index] = _BVHNode(lower, upper, -1, -1, ordered)
                return node_index
            span = upper - lower
            axis = int(np.argmax(span))
            order = np.lexsort((face_ids, centroids[face_ids, axis]))
            ordered = face_ids[order]
            midpoint = len(ordered) // 2
            left = build(ordered[:midpoint])
            right = build(ordered[midpoint:])
            built[node_index] = _BVHNode(lower, upper, left, right, None)
            return node_index

        root = build(np.arange(len(triangles), dtype=np.int64))
        return cls(
            vertices=vertices,
            triangles=triangles,
            face_edge_ordinals=inverse.reshape(len(triangles), 3),
            nodes=tuple(built),
            root=root,
            scale=Scale(bbox_diagonal=bbox_diagonal, mesh_spacing=mesh_spacing),
        )

    @classmethod
    def from_path(cls, path: Path) -> "MeshSectionIndex":
        mesh = o3d.io.read_triangle_mesh(str(path))
        return cls.from_arrays(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

    def query_faces(self, point: np.ndarray, normal: np.ndarray, radius: float, epsilon: float) -> np.ndarray:
        result: list[int] = []
        radius2 = radius * radius
        stack = [self.root]
        while stack:
            index = stack.pop()
            node = self.nodes[index]
            center = 0.5 * (node.lower + node.upper)
            half = 0.5 * (node.upper - node.lower)
            plane_distance = float(np.dot(center - point, normal))
            plane_radius = float(np.dot(np.abs(normal), half))
            if abs(plane_distance) > plane_radius + epsilon:
                continue
            nearest = np.maximum(np.maximum(node.lower - point, 0.0), point - node.upper)
            if float(np.dot(nearest, nearest)) > radius2:
                continue
            if node.faces is not None:
                result.extend(int(value) for value in node.faces)
                if len(result) > MAXIMUM_INTERSECTING_TRIANGLES:
                    raise _StationUnusable("LOCAL_TRIANGLE_CAP")
            else:
                # Push right first so the left child is visited first.
                stack.append(node.right)
                stack.append(node.left)
        return np.asarray(sorted(result), dtype=np.int64)


@dataclass(frozen=True)
class LineRecord:
    ordinal: int
    start: np.ndarray
    end: np.ndarray
    canonical_start: np.ndarray
    canonical_end: np.ndarray
    direction: np.ndarray
    length: float
    geometry_bytes: bytes
    geometry_sha256: str
    original: Mapping[str, Any]


@dataclass(frozen=True)
class StationMeasurement:
    station_index: int
    state: str
    reason: str
    q: np.ndarray
    profile: np.ndarray | None
    profile_s: np.ndarray | None
    turn_center: np.ndarray | None
    rho: float | None
    total_turn_deg: float | None
    flank_turn_deg: float | None
    coherence: float | None
    turn_width: float | None
    turn_width_interval: tuple[float, float] | None
    centered_mass_fraction: float | None
    axis_absolute_turn_deg: float | None
    left_flank_line: tuple[np.ndarray, np.ndarray] | None
    right_flank_line: tuple[np.ndarray, np.ndarray] | None


@dataclass
class CandidateMeasurement:
    line: LineRecord
    parameters: dict[str, float]
    stations: list[StationMeasurement]
    usable_mask: int
    centered_mask: int
    offset_mask: int
    valid_turn_count: int
    centered_count: int
    offset_count: int
    longest_centered_run: int
    longest_offset_run: int
    median_rho: float | None
    q10_rho: float | None
    q90_rho: float | None
    median_turn_deg: float | None
    centered_locus_angle_deg: float | None
    centered_locus_rms: float | None
    offset_locus_angle_deg: float | None
    offset_locus_rms: float | None
    offset_direction_concentration: float | None
    local_reason: str | None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_limits(started: float) -> None:
    if time.perf_counter() - started >= WALL_LIMIT_SECONDS:
        raise V55ScanAbstention("ABSTAIN_WALL_TIME")
    if _rss_bytes() >= RSS_LIMIT_BYTES:
        raise V55ScanAbstention("ABSTAIN_ACTUAL_RSS")


def _native(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise V55ScanAbstention("ABSTAIN_CANONICAL_SERIALIZER")
        return 0.0 if result == 0.0 else result
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise V55ScanAbstention("ABSTAIN_CANONICAL_SERIALIZER")


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    try:
        return (
            json.dumps(
                _native(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + suffix
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise V55ScanAbstention("ABSTAIN_CANONICAL_SERIALIZER") from error


def _extract_raw_endpoints(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "line_start" in row and "line_end" in row:
        raw_start, raw_end = row["line_start"], row["line_end"]
    elif "span_endpoints" in row and len(row["span_endpoints"]) == 2:
        raw_start, raw_end = row["span_endpoints"]
    else:
        raise V55ScanAbstention("ABSTAIN_MALFORMED_FROZEN_LINE")
    start = np.asarray(raw_start, dtype=np.float64)
    end = np.asarray(raw_end, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise V55ScanAbstention("ABSTAIN_MALFORMED_FROZEN_LINE")
    return start, end


def canonical_line(row: Mapping[str, Any], ordinal: int) -> LineRecord:
    start, end = _extract_raw_endpoints(row)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 0.0:
        raise V55ScanAbstention("ABSTAIN_MALFORMED_FROZEN_LINE")
    direction = delta / length
    pivot = int(np.argmax(np.abs(direction)))
    canonical_start = start.copy()
    canonical_end = end.copy()
    if direction[pivot] < 0.0:
        canonical_start, canonical_end = canonical_end, canonical_start
        direction = -direction
    canonical_start[canonical_start == 0.0] = 0.0
    canonical_end[canonical_end == 0.0] = 0.0
    direction[direction == 0.0] = 0.0
    payload = struct.pack(">6d", *(canonical_start.tolist() + canonical_end.tolist()))
    return LineRecord(
        ordinal=int(ordinal),
        start=start,
        end=end,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        direction=direction,
        length=length,
        geometry_bytes=payload,
        geometry_sha256=hashlib.sha256(payload).hexdigest(),
        original=row,
    )


def _segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float, np.ndarray]:
    delta = second - first
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        return float(np.linalg.norm(point - first)), 0.0, first.copy()
    fraction = float(np.clip(np.dot(point - first, delta) / denominator, 0.0, 1.0))
    nearest = first + fraction * delta
    return float(np.linalg.norm(point - nearest)), fraction, nearest


def _key_sort(key: tuple[str, int]) -> tuple[int, int]:
    return (0 if key[0] == "v" else 1, int(key[1]))


def _sample_polyline(points: np.ndarray, cumulative: np.ndarray, position: float, closed: bool) -> np.ndarray:
    total = float(cumulative[-1])
    if closed:
        position %= total
    else:
        position = float(np.clip(position, 0.0, total))
    index = int(np.searchsorted(cumulative, position, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    fraction = 0.0 if span <= 0.0 else (position - float(cumulative[index])) / span
    return points[index] + fraction * (points[index + 1] - points[index])


def _section_profile(
    index: MeshSectionIndex,
    q: np.ndarray,
    direction: np.ndarray,
    parameters: Mapping[str, float],
    *,
    seed_point: np.ndarray | None = None,
    output_spacing: float | None = None,
    maximum_output_points: int = PROFILE_POINT_COUNT,
    allow_partial_symmetric_extent: bool = False,
    minimum_section_segments_per_side: int = MINIMUM_SECTION_SEGMENTS_PER_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    radius = float(parameters["radius"])
    width = float(parameters["profile_half_span"])
    epsilon = float(parameters["section_epsilon"])
    face_ids = index.query_faces(q, direction, radius, epsilon)
    if len(face_ids) == 0:
        raise _StationUnusable("NO_SECTION_COMPONENT")

    node_points: dict[tuple[str, int], np.ndarray] = {}
    segments: list[tuple[tuple[str, int], tuple[str, int], int]] = []
    for face_id in face_ids:
        vertex_ids = index.triangles[face_id]
        points = index.vertices[vertex_ids]
        signed = np.dot(points - q, direction)
        signed[np.abs(signed) <= epsilon] = 0.0
        zero = signed == 0.0
        if bool(np.all(zero)) or int(np.count_nonzero(zero)) >= 2:
            continue
        emitted: dict[tuple[str, int], np.ndarray] = {}
        for local_vertex in np.flatnonzero(zero):
            key = ("v", int(vertex_ids[local_vertex]))
            emitted[key] = points[local_vertex]
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        for local_edge, (left, right) in enumerate(edge_pairs):
            dl = float(signed[left])
            dr = float(signed[right])
            if dl * dr < 0.0:
                fraction = dl / (dl - dr)
                key = ("e", int(index.face_edge_ordinals[face_id, local_edge]))
                emitted[key] = points[left] + fraction * (points[right] - points[left])
        if len(emitted) <= 1:
            # A single exactly-on-plane vertex is a resolved tangential touch,
            # not a section segment.  It contributes no topology.
            continue
        if len(emitted) != 2:
            raise _StationUnusable("UNRESOLVED_LOCAL_FACE")
        keys = sorted(emitted, key=_key_sort)
        if keys[0] == keys[1]:
            raise _StationUnusable("UNRESOLVED_LOCAL_FACE")
        for key in keys:
            point = emitted[key]
            existing = node_points.get(key)
            if existing is not None and not np.allclose(existing, point, rtol=0.0, atol=epsilon):
                raise _StationUnusable("INCONSISTENT_INTERSECTION_NODE")
            node_points[key] = point
        segments.append((keys[0], keys[1], int(face_id)))
    if not segments:
        raise _StationUnusable("NO_SECTION_SEGMENTS")

    seed_target = q if seed_point is None else np.asarray(seed_point, dtype=np.float64)
    if seed_target.shape != (3,) or not np.isfinite(seed_target).all():
        raise _StationUnusable("INVALID_SECTION_SEED")
    nearest_options = []
    for segment_index, (first_key, second_key, face_id) in enumerate(segments):
        distance, _, _ = _segment_distance(
            seed_target, node_points[first_key], node_points[second_key]
        )
        nearest_options.append((distance, face_id, segment_index))
    _, _, seed_segment = min(nearest_options)

    node_to_segments: dict[tuple[str, int], list[int]] = defaultdict(list)
    for segment_index, (first_key, second_key, _face_id) in enumerate(segments):
        node_to_segments[first_key].append(segment_index)
        node_to_segments[second_key].append(segment_index)
    component_segments: set[int] = set()
    queue = deque([seed_segment])
    while queue:
        segment_index = queue.popleft()
        if segment_index in component_segments:
            continue
        component_segments.add(segment_index)
        first_key, second_key, _ = segments[segment_index]
        for key in (first_key, second_key):
            queue.extend(sorted(node_to_segments[key]))
    component_nodes = {
        key
        for segment_index in component_segments
        for key in segments[segment_index][:2]
    }
    degrees = {
        key: sum(1 for segment_index in node_to_segments[key] if segment_index in component_segments)
        for key in component_nodes
    }
    if any(value > 2 or value <= 0 for value in degrees.values()):
        raise _StationUnusable("BRANCHED_SECTION_COMPONENT")
    endpoints = sorted((key for key, value in degrees.items() if value == 1), key=_key_sort)
    closed = len(endpoints) == 0
    if not closed and len(endpoints) != 2:
        raise _StationUnusable("OPEN_SECTION_TOPOLOGY")
    start_key = min(component_nodes, key=_key_sort) if closed else endpoints[0]
    ordered_keys = [start_key]
    used_segments: list[int] = []
    current = start_key
    previous_segment: int | None = None
    while True:
        options = sorted(
            (
                segment_index
                for segment_index in node_to_segments[current]
                if segment_index in component_segments and segment_index != previous_segment
            ),
            key=lambda segment_index: (
                _key_sort(
                    segments[segment_index][1]
                    if segments[segment_index][0] == current
                    else segments[segment_index][0]
                ),
                segments[segment_index][2],
            ),
        )
        if not options:
            break
        segment_index = options[0]
        if segment_index in used_segments:
            break
        first_key, second_key, _ = segments[segment_index]
        next_key = second_key if first_key == current else first_key
        used_segments.append(segment_index)
        if closed and next_key == start_key:
            ordered_keys.append(next_key)
            break
        ordered_keys.append(next_key)
        previous_segment = segment_index
        current = next_key
    if set(used_segments) != component_segments:
        raise _StationUnusable("SECTION_TRAVERSAL_INCOMPLETE")
    ordered_points = np.asarray([node_points[key] for key in ordered_keys], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0) or not np.isfinite(segment_lengths).all():
        raise _StationUnusable("DEGENERATE_SECTION_SEGMENT")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    nearest = None
    for segment_index in range(len(ordered_points) - 1):
        distance, fraction, _ = _segment_distance(q, ordered_points[segment_index], ordered_points[segment_index + 1])
        option = (distance, segment_index, fraction)
        if nearest is None or option < nearest:
            nearest = option
    assert nearest is not None
    nearest_distance, nearest_segment, nearest_fraction = nearest
    if nearest_distance > float(parameters["attachment_tolerance"]):
        raise _StationUnusable("AXIS_NOT_ATTACHED")
    nearest_position = float(cumulative[nearest_segment] + nearest_fraction * segment_lengths[nearest_segment])
    effective_width = width
    if closed:
        if total < 2.0 * width:
            raise _StationUnusable("INSUFFICIENT_TWO_SIDED_COVERAGE")
    elif nearest_position < width or total - nearest_position < width:
        if not allow_partial_symmetric_extent:
            raise _StationUnusable("INSUFFICIENT_TWO_SIDED_COVERAGE")
        effective_width = min(width, nearest_position, total - nearest_position)
        if effective_width <= 0.0:
            raise _StationUnusable("INSUFFICIENT_TWO_SIDED_COVERAGE")

    step = 2.0 * width / (PROFILE_POINT_COUNT - 1)
    segment_positions = 0.5 * (cumulative[:-1] + cumulative[1:])
    if closed:
        signed_positions = ((segment_positions - nearest_position + 0.5 * total) % total) - 0.5 * total
    else:
        signed_positions = segment_positions - nearest_position
    if int(
        np.count_nonzero(
            (signed_positions < 0.0) & (signed_positions >= -effective_width)
        )
    ) < int(minimum_section_segments_per_side):
        raise _StationUnusable("INSUFFICIENT_LEFT_SEGMENTS")
    if int(
        np.count_nonzero(
            (signed_positions > 0.0) & (signed_positions <= effective_width)
        )
    ) < int(minimum_section_segments_per_side):
        raise _StationUnusable("INSUFFICIENT_RIGHT_SEGMENTS")

    if output_spacing is None:
        profile_width = effective_width
        output_count = PROFILE_POINT_COUNT
    else:
        spacing = float(output_spacing)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise _StationUnusable("INVALID_PROFILE_SPACING")
        half_steps = int(math.floor(effective_width / spacing + 1e-12))
        output_count = 2 * half_steps + 1
        if output_count < 3 or output_count > int(maximum_output_points):
            raise _StationUnusable("PROFILE_OUTPUT_CAP")
        profile_width = half_steps * spacing
    negative_endpoint = _sample_polyline(
        ordered_points, cumulative, nearest_position - profile_width, closed
    )
    positive_endpoint = _sample_polyline(
        ordered_points, cumulative, nearest_position + profile_width, closed
    )
    orientation = 1.0 if tuple(positive_endpoint.tolist()) > tuple(negative_endpoint.tolist()) else -1.0
    profile_s = np.linspace(-profile_width, profile_width, output_count, dtype=np.float64)
    profile = np.asarray(
        [
            _sample_polyline(ordered_points, cumulative, nearest_position + orientation * value, closed)
            for value in profile_s
        ],
        dtype=np.float64,
    )
    return profile_s, profile


def _principal_direction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    top = float(values[-1])
    tolerance = max(abs(top), 1.0) * 1e-12
    tied = np.flatnonzero(top - values <= tolerance)
    basis = vectors[:, tied]
    candidates: list[np.ndarray] = []
    for axis in np.eye(3):
        projected = basis @ (basis.T @ axis)
        norm = float(np.linalg.norm(projected))
        if norm > 1e-15:
            candidate = projected / norm
            pivot = int(np.argmax(np.abs(candidate)))
            if candidate[pivot] < 0.0:
                candidate = -candidate
            candidate[candidate == 0.0] = 0.0
            candidates.append(candidate)
    if not candidates:
        raise _StationUnusable("TLS_DEGENERATE")
    direction = min(candidates, key=lambda value: tuple(value.tolist()))
    projection = centered @ direction
    residual = centered - projection[:, None] * direction
    rms = float(math.sqrt(float(np.mean(np.sum(residual * residual, axis=1)))))
    return center, direction, rms


def _directed_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def _shortest_turn_interval(mass: np.ndarray, midpoint_s: np.ndarray, step: float) -> tuple[float, float] | None:
    target = 0.80 * float(np.sum(mass))
    best: tuple[int, int] | None = None
    total = 0.0
    left = 0
    for right, value in enumerate(mass):
        total += float(value)
        while left <= right and total - float(mass[left]) >= target:
            total -= float(mass[left])
            left += 1
        if total >= target:
            option = (right - left + 1, left)
            if best is None or option < best:
                best = option
    if best is None:
        return None
    count, start = best
    return (
        float(midpoint_s[start] - 0.5 * step),
        float(midpoint_s[start + count - 1] + 0.5 * step),
    )


def _measure_station(
    index: MeshSectionIndex,
    line: LineRecord,
    station_index: int,
    parameters: Mapping[str, float],
) -> StationMeasurement:
    q = line.canonical_start + STATION_FRACTIONS[station_index] * (
        line.canonical_end - line.canonical_start
    )
    try:
        profile_s, profile = _section_profile(index, q, line.direction, parameters)
        left_mask = (profile_s / float(parameters["profile_half_span"]) >= -1.0) & (
            profile_s / float(parameters["profile_half_span"]) <= -0.55
        )
        right_mask = (profile_s / float(parameters["profile_half_span"]) >= 0.55) & (
            profile_s / float(parameters["profile_half_span"]) <= 1.0
        )
        left_center, left_direction, left_rms = _principal_direction(profile[left_mask])
        right_center, right_direction, right_rms = _principal_direction(profile[right_mask])
        arclength_direction = profile[-1] - profile[0]
        if float(np.dot(left_direction, arclength_direction)) < 0.0:
            left_direction = -left_direction
        if float(np.dot(right_direction, arclength_direction)) < 0.0:
            right_direction = -right_direction
        flank_turn_deg = _directed_angle_degrees(left_direction, right_direction)

        raw_tangents = np.asarray(
            [
                (profile[k + 2] - profile[k - 2])
                / np.linalg.norm(profile[k + 2] - profile[k - 2])
                for k in range(2, 63)
            ],
            dtype=np.float64,
        )
        smoothed = []
        for k in range(4, 61):
            raw_index = k - 2
            vector = np.sum(raw_tangents[raw_index - 2 : raw_index + 3], axis=0)
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0 or not math.isfinite(norm):
                raise _StationUnusable("TANGENT_DEGENERATE")
            smoothed.append(vector / norm)
        smoothed_tangents = np.asarray(smoothed, dtype=np.float64)
        deltas = np.asarray(
            [
                math.atan2(
                    float(np.dot(line.direction, np.cross(smoothed_tangents[k], smoothed_tangents[k + 1]))),
                    float(np.dot(smoothed_tangents[k], smoothed_tangents[k + 1])),
                )
                for k in range(len(smoothed_tangents) - 1)
            ],
            dtype=np.float64,
        )
        absolute_turn = float(np.sum(np.abs(deltas)))
        signed_turn = float(np.sum(deltas))
        total_turn = abs(signed_turn)
        coherence = 0.0 if absolute_turn == 0.0 else total_turn / absolute_turn
        total_turn_deg = math.degrees(total_turn)
        if (
            left_rms > float(parameters["flank_rms_limit"])
            or right_rms > float(parameters["flank_rms_limit"])
            or not (MINIMUM_FLANK_TURN_DEG <= flank_turn_deg <= MAXIMUM_FLANK_TURN_DEG)
            or total_turn_deg < MINIMUM_FLANK_TURN_DEG
            or coherence < MINIMUM_SIGNED_TURN_COHERENCE
            or abs(total_turn_deg - flank_turn_deg) > max(6.0, 0.25 * flank_turn_deg)
        ):
            return StationMeasurement(
                station_index, "UNUSABLE", "NON_LOCALIZED_TURN", q, profile, profile_s,
                None, None, total_turn_deg, flank_turn_deg, coherence, None, None, None, None,
                (left_center, left_direction), (right_center, right_direction),
            )
        sign = 1.0 if signed_turn >= 0.0 else -1.0
        mass = np.maximum(0.0, sign * deltas)
        mass_total = float(np.sum(mass))
        if mass_total <= 0.0:
            raise _StationUnusable("ZERO_POSITIVE_TURN_MASS")
        # delta k spans profile samples k+4 to k+5.
        midpoint_indices = np.arange(4, 60, dtype=np.float64) + 0.5
        step = float(profile_s[1] - profile_s[0])
        midpoint_s = profile_s[0] + midpoint_indices * step
        midpoint_points = 0.5 * (profile[4:60] + profile[5:61])
        turn_center = np.sum(mass[:, None] * midpoint_points, axis=0) / mass_total
        rho = float(np.linalg.norm(turn_center - q))
        turn_interval = _shortest_turn_interval(mass, midpoint_s, step)
        turn_width = math.inf if turn_interval is None else turn_interval[1] - turn_interval[0]
        if turn_width > MAXIMUM_TURN_WIDTH_OVER_R * float(parameters["radius"]):
            return StationMeasurement(
                station_index, "UNUSABLE", "DISTRIBUTED_TURN", q, profile, profile_s,
                turn_center, rho, total_turn_deg, flank_turn_deg, coherence, turn_width,
                turn_interval, None, None, (left_center, left_direction), (right_center, right_direction),
            )
        center_mask = np.abs(midpoint_s) <= float(parameters["center_tolerance"])
        centered_fraction = float(np.sum(mass[center_mask]) / mass_total)
        axis_absolute_turn_deg = math.degrees(float(np.sum(np.abs(deltas[center_mask]))))
        if rho <= float(parameters["center_tolerance"]) and centered_fraction >= MINIMUM_CENTERED_TURN_MASS:
            state = "CENTERED"
        elif (
            float(parameters["definite_offset_threshold"]) <= rho <= float(parameters["maximum_offset"])
            and centered_fraction <= MAXIMUM_OFFSET_AXIS_TURN_MASS
            and axis_absolute_turn_deg <= MAXIMUM_OFFSET_AXIS_ABSOLUTE_TURN_DEG
        ):
            state = "OFFSET_FLAT_AXIS"
        else:
            state = "VALID_OTHER"
        return StationMeasurement(
            station_index, state, state, q, profile, profile_s, turn_center, rho,
            total_turn_deg, flank_turn_deg, coherence, turn_width, turn_interval, centered_fraction,
            axis_absolute_turn_deg, (left_center, left_direction), (right_center, right_direction),
        )
    except _StationUnusable as error:
        return StationMeasurement(
            station_index, "UNUSABLE", error.reason, q, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
        )


def _longest_run(mask: int) -> int:
    best = current = 0
    for index in range(9):
        if mask & (1 << index):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _lower_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[(len(ordered) - 1) // 2]


def _nearest_rank(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def _line_angle(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))))


def _fit_locus(stations: Sequence[StationMeasurement], direction: np.ndarray) -> tuple[float | None, float | None]:
    points = np.asarray([station.turn_center for station in stations if station.turn_center is not None])
    if len(points) < 2:
        return None, None
    try:
        _center, locus_direction, rms = _principal_direction(points)
    except _StationUnusable:
        return None, None
    if float(np.dot(locus_direction, direction)) < 0.0:
        locus_direction = -locus_direction
    return _line_angle(locus_direction, direction), rms


def measure_candidate(index: MeshSectionIndex, line: LineRecord) -> CandidateMeasurement:
    parameters = index.scale.values(line.length)
    if line.length < parameters["minimum_line_span"]:
        return CandidateMeasurement(
            line, parameters, [], 0, 0, 0, 0, 0, 0, 0, 0,
            None, None, None, None, None, None, None, None, None,
            "LINE_SHORTER_THAN_MINIMUM",
        )
    if parameters["radius"] < 10.0 * index.scale.mesh_spacing:
        return CandidateMeasurement(
            line, parameters, [], 0, 0, 0, 0, 0, 0, 0, 0,
            None, None, None, None, None, None, None, None, None,
            "INSUFFICIENT_SCALE_SEPARATION",
        )
    stations = [_measure_station(index, line, station, parameters) for station in range(9)]
    usable_mask = sum(1 << row.station_index for row in stations if row.state != "UNUSABLE")
    centered_mask = sum(1 << row.station_index for row in stations if row.state == "CENTERED")
    offset_mask = sum(1 << row.station_index for row in stations if row.state == "OFFSET_FLAT_AXIS")
    valid = [row for row in stations if row.state != "UNUSABLE"]
    centered = [row for row in stations if row.state == "CENTERED"]
    offset = [row for row in stations if row.state == "OFFSET_FLAT_AXIS"]
    relevant = [row for row in stations if row.rho is not None]
    rhos = [float(row.rho) for row in relevant]
    turns = [float(row.flank_turn_deg) for row in valid if row.flank_turn_deg is not None]
    centered_angle, centered_rms = _fit_locus(centered, line.direction)
    offset_angle, offset_rms = _fit_locus(offset, line.direction)
    concentration = None
    if offset:
        directions = []
        for row in offset:
            assert row.turn_center is not None
            vector = row.turn_center - row.q
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                directions.append(vector / norm)
        if directions:
            concentration = float(np.linalg.norm(np.mean(np.asarray(directions), axis=0)))
    return CandidateMeasurement(
        line=line,
        parameters=parameters,
        stations=stations,
        usable_mask=usable_mask,
        centered_mask=centered_mask,
        offset_mask=offset_mask,
        valid_turn_count=len(valid),
        centered_count=len(centered),
        offset_count=len(offset),
        longest_centered_run=_longest_run(centered_mask),
        longest_offset_run=_longest_run(offset_mask),
        median_rho=_lower_median(rhos),
        q10_rho=_nearest_rank(rhos, 0.10),
        q90_rho=_nearest_rank(rhos, 0.90),
        median_turn_deg=_lower_median(turns),
        centered_locus_angle_deg=centered_angle,
        centered_locus_rms=centered_rms,
        offset_locus_angle_deg=offset_angle,
        offset_locus_rms=offset_rms,
        offset_direction_concentration=concentration,
        local_reason=None,
    )


def _prune_decision(measurement: CandidateMeasurement) -> tuple[bool, str]:
    if measurement.local_reason:
        return False, "KEEP_DEFAULT_" + measurement.local_reason
    p = measurement.parameters
    rho_values = [float(row.rho) for row in measurement.stations if row.state == "OFFSET_FLAT_AXIS" and row.rho is not None]
    q10 = _nearest_rank(rho_values, 0.10)
    q90 = _nearest_rank(rho_values, 0.90)
    median = _lower_median(rho_values)
    checks = (
        (measurement.offset_count >= MINIMUM_PRUNING_OFFSET_STATIONS, "KEEP_DEFAULT_OFFSET_COUNT"),
        (measurement.longest_offset_run >= MINIMUM_PRUNING_OFFSET_RUN, "KEEP_DEFAULT_OFFSET_RUN"),
        (measurement.centered_count == 0, "KEEP_DEFAULT_CENTERED_PRESENT"),
        (measurement.offset_locus_angle_deg is not None and measurement.offset_locus_angle_deg <= MAXIMUM_TURN_LOCUS_DIRECTION_ERROR_DEG, "KEEP_DEFAULT_LOCUS_ANGLE"),
        (measurement.offset_locus_rms is not None and measurement.offset_locus_rms <= p["turn_locus_rms_limit"], "KEEP_DEFAULT_LOCUS_RMS"),
        (measurement.offset_direction_concentration is not None and measurement.offset_direction_concentration >= MINIMUM_OFFSET_DIRECTION_CONCENTRATION, "KEEP_DEFAULT_OFFSET_DIRECTION"),
        (q10 is not None and q10 > 0.0 and q90 is not None and q90 / q10 <= MAXIMUM_OFFSET_Q90_Q10, "KEEP_DEFAULT_OFFSET_SPREAD"),
        (median is not None and median >= p["definite_offset_threshold"], "KEEP_DEFAULT_OFFSET_MEDIAN"),
        (q90 is not None and q90 <= p["maximum_offset"], "KEEP_DEFAULT_OFFSET_LIMIT"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "PRUNE_OFFSET_FLAT_AXIS_CERTIFICATE"


def _addition_decision(measurement: CandidateMeasurement) -> tuple[bool, str]:
    if measurement.local_reason:
        return False, "REJECT_" + measurement.local_reason
    centered_rhos = [float(row.rho) for row in measurement.stations if row.state == "CENTERED" and row.rho is not None]
    median = _lower_median(centered_rhos)
    q90 = _nearest_rank(centered_rhos, 0.90)
    p = measurement.parameters
    checks = (
        (measurement.valid_turn_count >= MINIMUM_ADDITION_USABLE_STATIONS, "REJECT_USABLE_COUNT"),
        (measurement.centered_count >= MINIMUM_ADDITION_CENTERED_STATIONS, "REJECT_CENTERED_COUNT"),
        (measurement.longest_centered_run >= MINIMUM_ADDITION_CENTERED_RUN, "REJECT_CENTERED_RUN"),
        (measurement.centered_locus_angle_deg is not None and measurement.centered_locus_angle_deg <= MAXIMUM_TURN_LOCUS_DIRECTION_ERROR_DEG, "REJECT_LOCUS_ANGLE"),
        (measurement.centered_locus_rms is not None and measurement.centered_locus_rms <= p["turn_locus_rms_limit"], "REJECT_LOCUS_RMS"),
        (median is not None and median <= p["center_tolerance"], "REJECT_CENTER_MEDIAN"),
        (q90 is not None and q90 <= p["center_q90_limit"], "REJECT_CENTER_Q90"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "PROPOSED"


def _measurement_summary(measurement: CandidateMeasurement, decision: str) -> dict[str, Any]:
    radius = float(measurement.parameters["radius"])
    def normalized(value: float | None) -> float | None:
        return None if value is None else round(float(value) / radius, 12)
    return {
        "ordinal": measurement.line.ordinal,
        "geometry_sha256": measurement.line.geometry_sha256,
        "decision": decision,
        "usable_station_mask": measurement.usable_mask,
        "centered_station_mask": measurement.centered_mask,
        "offset_station_mask": measurement.offset_mask,
        "valid_turn_station_count": measurement.valid_turn_count,
        "centered_station_count": measurement.centered_count,
        "offset_station_count": measurement.offset_count,
        "longest_centered_run": measurement.longest_centered_run,
        "longest_offset_run": measurement.longest_offset_run,
        "median_offset_over_r": normalized(measurement.median_rho),
        "q90_offset_over_r": normalized(measurement.q90_rho),
        "median_turn_angle_deg": None if measurement.median_turn_deg is None else round(measurement.median_turn_deg, 12),
        "centered_locus_angle_deg": None if measurement.centered_locus_angle_deg is None else round(measurement.centered_locus_angle_deg, 12),
        "centered_locus_rms_over_r": normalized(measurement.centered_locus_rms),
        "offset_locus_angle_deg": None if measurement.offset_locus_angle_deg is None else round(measurement.offset_locus_angle_deg, 12),
        "offset_locus_rms_over_r": normalized(measurement.offset_locus_rms),
        "offset_direction_concentration": None if measurement.offset_direction_concentration is None else round(measurement.offset_direction_concentration, 12),
    }


def _lines_conflict(first: LineRecord, second: LineRecord, separation_limit: float) -> bool:
    if _line_angle(first.direction, second.direction) > CONFLICT_DIRECTION_ANGLE_DEG:
        return False
    cross = np.cross(first.direction, second.direction)
    cross_norm = float(np.linalg.norm(cross))
    delta = second.canonical_start - first.canonical_start
    if cross_norm > 1e-12:
        separation = abs(float(np.dot(delta, cross / cross_norm)))
    else:
        separation = float(np.linalg.norm(delta - np.dot(delta, first.direction) * first.direction))
    if separation > separation_limit:
        return False
    mean = first.direction + second.direction
    mean /= np.linalg.norm(mean)
    first_interval = sorted((float(np.dot(first.canonical_start, mean)), float(np.dot(first.canonical_end, mean))))
    second_interval = sorted((float(np.dot(second.canonical_start, mean)), float(np.dot(second.canonical_end, mean))))
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    shorter = min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0])
    return shorter > 0.0 and overlap >= MINIMUM_CONFLICT_OVERLAP * shorter


def _serialize_addition(line: LineRecord) -> dict[str, Any]:
    digest = line.geometry_sha256
    return {
        "bend_id": f"TRV55-{line.ordinal:04d}-{digest[:12]}",
        "line_start": line.start.tolist(),
        "line_end": line.end.tolist(),
        "selection_route": "mesh_section_turn_locus_reselection_v55",
        "source_hypothesis_id": f"frozen_pool_ordinal::{line.ordinal}",
        "source_kind": "mesh_section_turn_locus",
    }


def _constants_payload() -> dict[str, Any]:
    return {
        "station_fractions": STATION_FRACTIONS,
        "profile_point_count": PROFILE_POINT_COUNT,
        "minimum_section_segments_per_side": MINIMUM_SECTION_SEGMENTS_PER_SIDE,
        "minimum_flank_turn_deg": MINIMUM_FLANK_TURN_DEG,
        "maximum_flank_turn_deg": MAXIMUM_FLANK_TURN_DEG,
        "minimum_signed_turn_coherence": MINIMUM_SIGNED_TURN_COHERENCE,
        "minimum_centered_turn_mass": MINIMUM_CENTERED_TURN_MASS,
        "maximum_offset_axis_turn_mass": MAXIMUM_OFFSET_AXIS_TURN_MASS,
        "maximum_offset_axis_absolute_turn_deg": MAXIMUM_OFFSET_AXIS_ABSOLUTE_TURN_DEG,
        "maximum_turn_width_over_r": MAXIMUM_TURN_WIDTH_OVER_R,
        "maximum_turn_locus_direction_error_deg": MAXIMUM_TURN_LOCUS_DIRECTION_ERROR_DEG,
        "addition_usable_centered": [MINIMUM_ADDITION_USABLE_STATIONS, MINIMUM_ADDITION_CENTERED_STATIONS],
        "addition_longest_centered_run": MINIMUM_ADDITION_CENTERED_RUN,
        "pruning_offset_stations": MINIMUM_PRUNING_OFFSET_STATIONS,
        "pruning_longest_offset_run": MINIMUM_PRUNING_OFFSET_RUN,
        "minimum_offset_direction_concentration": MINIMUM_OFFSET_DIRECTION_CONCENTRATION,
        "maximum_offset_q90_q10": MAXIMUM_OFFSET_Q90_Q10,
        "conflict_direction_angle_deg": CONFLICT_DIRECTION_ANGLE_DEG,
        "minimum_conflict_overlap": MINIMUM_CONFLICT_OVERLAP,
        "bvh_leaf_size": BVH_LEAF_SIZE,
        "maximum_intersecting_triangles": MAXIMUM_INTERSECTING_TRIANGLES,
        "maximum_candidates": MAXIMUM_CANDIDATES,
        "maximum_public_bytes": MAXIMUM_PUBLIC_BYTES,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
    }


CONSTANTS_SHA256 = hashlib.sha256(canonical_bytes(_constants_payload())).hexdigest()


def _compose_from_index(
    v52_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    index: MeshSectionIndex,
    *,
    integrity: Mapping[str, str] | None = None,
    started: float | None = None,
) -> tuple[dict[str, Any], dict[str, CandidateMeasurement], dict[int, CandidateMeasurement]]:
    started = time.perf_counter() if started is None else started
    if v52_result.get("source_only") is not True or v52_result.get("research_only") is not True:
        raise V55ScanAbstention("ABSTAIN_INVALID_V52_CONTRACT")
    raw_candidates = list(pool.get("candidates") or ())
    if len(raw_candidates) > MAXIMUM_CANDIDATES:
        raise V55ScanAbstention("ABSTAIN_CANDIDATE_CAP")
    v52_predictions = list(v52_result.get("predictions") or ())
    v52_lines = [canonical_line(row, ordinal) for ordinal, row in enumerate(v52_predictions)]
    pool_lines = [canonical_line(row, ordinal) for ordinal, row in enumerate(raw_candidates)]
    original_geometry = {line.geometry_bytes for line in v52_lines}

    retained_predictions: list[dict[str, Any]] = []
    retained_lines: list[LineRecord] = []
    v52_summaries: list[dict[str, Any]] = []
    v52_measurements: dict[str, CandidateMeasurement] = {}
    prune_count = 0
    reason_counts: Counter[str] = Counter()
    for line, prediction in zip(v52_lines, v52_predictions):
        measurement = measure_candidate(index, line)
        prune, reason = _prune_decision(measurement)
        v52_measurements[line.geometry_sha256] = measurement
        reason_counts[reason] += 1
        v52_summaries.append(_measurement_summary(measurement, reason))
        if prune:
            prune_count += 1
        else:
            retained_predictions.append(deepcopy(dict(prediction)))
            retained_lines.append(line)
        _check_limits(started)

    proposals: list[tuple[tuple[Any, ...], LineRecord, CandidateMeasurement]] = []
    pool_measurements: dict[int, CandidateMeasurement] = {}
    for line in pool_lines:
        if line.geometry_bytes in original_geometry:
            reason_counts["REJECT_ORIGINAL_V52_GEOMETRY_DUPLICATE"] += 1
            continue
        measurement = measure_candidate(index, line)
        pool_measurements[line.ordinal] = measurement
        proposed, reason = _addition_decision(measurement)
        reason_counts[reason] += 1
        if proposed:
            radius = measurement.parameters["radius"]
            q90 = _nearest_rank(
                [float(row.rho) for row in measurement.stations if row.state == "CENTERED" and row.rho is not None],
                0.90,
            )
            key = (
                -measurement.centered_count,
                -measurement.longest_centered_run,
                float(q90) / radius if q90 is not None else math.inf,
                float(measurement.centered_locus_rms) / radius if measurement.centered_locus_rms is not None else math.inf,
                -line.length / index.scale.bbox_diagonal,
                line.ordinal,
            )
            proposals.append((key, line, measurement))
        _check_limits(started)
    proposals.sort(key=lambda row: row[0])

    accepted_summaries: list[dict[str, Any]] = []
    accepted_lines: list[LineRecord] = []
    final_predictions = list(retained_predictions)
    for _key, line, measurement in proposals:
        separation_limit = measurement.parameters["conflict_separation"]
        if any(_lines_conflict(line, incumbent, separation_limit) for incumbent in [*retained_lines, *accepted_lines]):
            reason_counts["REJECT_CONFLICT"] += 1
            continue
        accepted_lines.append(line)
        final_predictions.append(_serialize_addition(line))
        accepted_summaries.append(_measurement_summary(measurement, "ACCEPTED_ADDITION"))
    audit = {
        "precommit_token": PRECOMMIT_TOKEN,
        "short_name": SHORT_NAME,
        "status": "COMMITTED",
        "constants_sha256": CONSTANTS_SHA256,
        "integrity_sha256": dict(integrity or {}),
        "bbox_diagonal": round(index.scale.bbox_diagonal, 12),
        "mesh_spacing": round(index.scale.mesh_spacing, 12),
        "original_v52_count": len(v52_predictions),
        "retained_v52_count": len(retained_predictions),
        "pruned_v52_count": prune_count,
        "frozen_pool_candidate_count": len(pool_lines),
        "accepted_addition_count": len(accepted_lines),
        "v52_line_summaries": v52_summaries,
        "accepted_addition_summaries": accepted_summaries,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
    }
    result = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_mesh_section_turn_locus_reselection_result",
        "profile": PROFILE,
        "source_only": True,
        "research_only": True,
        "expected_count_used": False,
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
        "v52_artifact_mutated": False,
        "candidate_count": len(final_predictions),
        "predictions": final_predictions,
        "mesh_section_turn_locus_reselection_audit": audit,
    }
    result["v55_result_sha256"] = hashlib.sha256(
        canonical_bytes({"profile": PROFILE, "predictions": final_predictions, "audit": audit})
    ).hexdigest()
    encoded = canonical_bytes(result, newline=True)
    if len(encoded) > MAXIMUM_PUBLIC_BYTES:
        raise V55ScanAbstention("ABSTAIN_OUTPUT_SCHEMA_OVERFLOW")
    return result, v52_measurements, pool_measurements


def compose_v55(
    v52_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    scan_path: Path,
    *,
    integrity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        index = MeshSectionIndex.from_path(Path(scan_path))
    except V55ScanAbstention:
        raise
    except Exception as error:
        raise V55ScanAbstention("ABSTAIN_SECTION_INDEX_CONSTRUCTION") from error
    result, _v52, _pool = _compose_from_index(
        v52_result, pool, index, integrity=integrity, started=started
    )
    _check_limits(started)
    return result


def _profile_points(
    turn_events: Sequence[tuple[float, float]],
    *,
    minimum_s: float = -28.0,
    maximum_s: float = 28.0,
    step: float = 0.4,
    distributed: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = list(np.arange(minimum_s, maximum_s + 0.5 * step, step))
    values.extend(float(position) for position, _angle in turn_events)
    if distributed is not None:
        values.extend((distributed[0], distributed[1]))
    s_values = np.asarray(sorted(set(round(value, 12) for value in values)), dtype=np.float64)
    angles = np.zeros_like(s_values)
    for index, value in enumerate(s_values):
        angle = 0.0
        for position, delta in turn_events:
            if value > position:
                angle += math.radians(delta)
        if distributed is not None:
            start, end, delta = distributed
            if value >= end:
                angle += math.radians(delta)
            elif value > start:
                angle += math.radians(delta) * (value - start) / (end - start)
        angles[index] = angle
    yz = np.zeros((len(s_values), 2), dtype=np.float64)
    for index in range(1, len(s_values)):
        ds = float(s_values[index] - s_values[index - 1])
        midpoint_angle = 0.5 * (angles[index] + angles[index - 1])
        yz[index] = yz[index - 1] + ds * np.asarray([math.cos(midpoint_angle), math.sin(midpoint_angle)])
    origin = int(np.argmin(np.abs(s_values)))
    yz -= yz[origin]
    return s_values, yz


def _extruded_fixture_mesh(s_values: np.ndarray, yz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Offset the longitudinal mesh rows so the fixed station planes exercise
    # ordinary edge crossings rather than coinciding with complete mesh edges.
    x_values = np.linspace(-96.25, 95.75, 97, dtype=np.float64)
    vertices = np.asarray(
        [[x, yz[index, 0], yz[index, 1]] for x in x_values for index in range(len(s_values))],
        dtype=np.float64,
    )
    faces = []
    width = len(s_values)
    for x_index in range(len(x_values) - 1):
        for profile_index in range(width - 1):
            first = x_index * width + profile_index
            second = first + width
            faces.extend(((first, second, second + 1), (first, second + 1, first + 1)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _fixture_line(s_values: np.ndarray, yz: np.ndarray, parameter: float, ordinal: int = 0) -> LineRecord:
    y = float(np.interp(parameter, s_values, yz[:, 0]))
    z = float(np.interp(parameter, s_values, yz[:, 1]))
    return canonical_line(
        {"line_start": [-60.0, y, z], "line_end": [60.0, y, z]}, ordinal
    )


def run_preflight_fixtures() -> dict[str, Any]:
    """Run the ten frozen source-free fixture assertions exactly once."""

    rows: list[dict[str, Any]] = []

    def build(
        turn_events: Sequence[tuple[float, float]],
        *,
        distributed: tuple[float, float, float] | None = None,
        minimum_s: float = -28.0,
        maximum_s: float = 28.0,
    ) -> tuple[MeshSectionIndex, np.ndarray, np.ndarray]:
        s_values, yz = _profile_points(
            turn_events,
            distributed=distributed,
            minimum_s=minimum_s,
            maximum_s=maximum_s,
        )
        vertices, faces = _extruded_fixture_mesh(s_values, yz)
        return MeshSectionIndex.from_arrays(vertices, faces, scale_override=(200.0, 0.5)), s_values, yz

    def record(name: str, passed: bool, facts: Mapping[str, Any]) -> None:
        rows.append({"fixture": name, "passed": bool(passed), "facts": dict(facts)})
        if not passed:
            raise V55ScanAbstention("PREFLIGHT_" + name)

    index, s, yz = build(((0.0, 90.0),))
    sharp = measure_candidate(index, _fixture_line(s, yz, 0.0))
    record("PF01_CENTERED_SHARP90", sharp.centered_count == 9 and not _prune_decision(sharp)[0], {"centered": sharp.centered_count})

    index, s, yz = build((), distributed=(-2.0, 2.0, 60.0))
    rounded = measure_candidate(index, _fixture_line(s, yz, 0.0))
    record("PF02_CENTERED_ROUND60", rounded.centered_count >= 8 and not _prune_decision(rounded)[0], {"centered": rounded.centered_count})

    index, s, yz = build(())
    plane = measure_candidate(index, _fixture_line(s, yz, 0.0))
    record("PF03_SINGLE_PLANE", not _addition_decision(plane)[0] and not _prune_decision(plane)[0], {"centered": plane.centered_count, "offset": plane.offset_count})

    offset = 0.40 * 9.6
    index, s, yz = build(((offset, 90.0),))
    incumbent = measure_candidate(index, _fixture_line(s, yz, 0.0))
    true_fold = measure_candidate(index, _fixture_line(s, yz, offset, 1))
    record("PF04_OFFSET_FOLD_0P40R", _prune_decision(incumbent)[0] and _addition_decision(true_fold)[0], {"offset": incumbent.offset_count, "centered": true_fold.centered_count})

    index, s, yz = build(((-offset, 60.0), (offset, -60.0)))
    midpoint = measure_candidate(index, _fixture_line(s, yz, 0.0))
    left = measure_candidate(index, _fixture_line(s, yz, -offset, 1))
    right = measure_candidate(index, _fixture_line(s, yz, offset, 2))
    record("PF05_DOUBLE_FOLD_PM0P40R", not _addition_decision(midpoint)[0] and not _prune_decision(midpoint)[0] and _addition_decision(left)[0] and _addition_decision(right)[0], {"left": left.centered_count, "right": right.centered_count})

    index, s, yz = build((), distributed=(-8.0, 8.0, 90.0))
    cylinder = measure_candidate(index, _fixture_line(s, yz, 0.0))
    record("PF06_DISTRIBUTED_CYLINDER", not _addition_decision(cylinder)[0], {"reason": _addition_decision(cylinder)[1]})

    index, s, yz = build(((0.0, 90.0),), minimum_s=-2.0, maximum_s=28.0)
    boundary = measure_candidate(index, _fixture_line(s, yz, 0.0))
    record("PF07_OPEN_BOUNDARY", not _addition_decision(boundary)[0] and not _prune_decision(boundary)[0], {"usable": boundary.valid_turn_count})

    index, s, yz = build(((0.0, 90.0),))
    true_line = _fixture_line(s, yz, 0.0, 1)
    y_first = float(np.interp(-8.0, s, yz[:, 0])); z_first = float(np.interp(-8.0, s, yz[:, 1]))
    y_second = float(np.interp(8.0, s, yz[:, 0])); z_second = float(np.interp(8.0, s, yz[:, 1]))
    diagonal = canonical_line({"line_start": [-60.0, y_first, z_first], "line_end": [60.0, y_second, z_second]}, 0)
    diagonal_measurement = measure_candidate(index, diagonal)
    true_measurement = measure_candidate(index, true_line)
    record("PF08_DIAGONAL_CROSS_PANEL", not _addition_decision(diagonal_measurement)[0] and _addition_decision(true_measurement)[0], {"diagonal_centered": diagonal_measurement.centered_count, "true_centered": true_measurement.centered_count})

    index, s, yz = build(((-8.0, 60.0), (8.0, -60.0)))
    first = measure_candidate(index, _fixture_line(s, yz, -8.0, 0))
    second = measure_candidate(index, _fixture_line(s, yz, 8.0, 1))
    first_line = first.line; second_line = second.line
    record("PF09_TWO_PARALLEL_FOLDS_0P08B", _addition_decision(first)[0] and _addition_decision(second)[0] and not _lines_conflict(first_line, second_line, first.parameters["conflict_separation"]), {"first": first.centered_count, "second": second.centered_count})

    native = {"b": np.bool_(True), "i": np.int64(7), "f": np.float64(1.25), "z": -0.0}
    ordinary = {"b": True, "i": 7, "f": 1.25, "z": 0.0}
    serializer_pass = canonical_bytes(native) == canonical_bytes(ordinary)
    nonfinite_failed = False
    try:
        canonical_bytes({"bad": math.inf})
    except V55ScanAbstention:
        nonfinite_failed = True
    record("PF10_CANONICAL_SERIALIZER_REPLAY", serializer_pass and nonfinite_failed, {"native_equal": serializer_pass, "nonfinite_failed": nonfinite_failed})

    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_source_free_preflight",
        "precommit_token": PRECOMMIT_TOKEN,
        "constants_sha256": CONSTANTS_SHA256,
        "fixture_scale": {"bbox_diagonal": 200.0, "mesh_spacing": 0.5, "line_span": 120.0},
        "fixtures": rows,
        "passed": all(row["passed"] for row in rows) and len(rows) == 10,
    }


__all__ = [
    "CONSTANTS_SHA256",
    "MAXIMUM_PUBLIC_BYTES",
    "PRECOMMIT_TOKEN",
    "PROFILE",
    "RSS_LIMIT_BYTES",
    "SHORT_NAME",
    "TERMINAL_FAILURE_TOKEN",
    "V55ScanAbstention",
    "WALL_LIMIT_SECONDS",
    "MeshSectionIndex",
    "canonical_bytes",
    "canonical_line",
    "compose_v55",
    "measure_candidate",
    "run_preflight_fixtures",
]
