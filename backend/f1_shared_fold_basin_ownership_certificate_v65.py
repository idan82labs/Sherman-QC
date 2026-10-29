"""V65 shared-fold-basin ownership certificate (SFBOC_V1).

Research-only, source-only, deletion-only consolidator.  It is inserted before
the unchanged ordinary final NMS.  Missing or ambiguous geometry is never
negative evidence and therefore can never delete a candidate.

The implementation follows the 2026-08-15 locked professor ruling.  Large
transient objects (source points, graph nodes, normals and profiles) never
enter the returned audit.  Runtime metrics are deliberately outside the
canonical audit hash so primary/replay hashes remain reproducible.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import resource
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


PROFILE = "V65_SHARED_FOLD_BASIN_OWNERSHIP_CERTIFICATE_V1"
SHORT_NAME = "SFBOC_V1"
LOCK_TOKEN = "V65_SFBOC_V1_MECHANISM_LOCKED"

MAX_SOURCE_POINTS = 200_000
MAX_CANDIDATES = 64
MAX_ELIGIBLE_PAIRS = 128
STATION_COUNT = 9
MAX_STATION_NODES = 4_096
MAX_STATION_EDGES = 32_768
MAX_CLUSTER_SIZE = 8
MAX_AUDIT_BYTES = 512 * 1024
WALL_SECONDS = 90.0

SAME_BASIN = 1
SPLIT_BASIN = 2
INDETERMINATE = 0


class SFBOCInputError(RuntimeError):
    pass


class SFBOCResourceFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Line:
    ordinal: int
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    hash128: str
    original: Mapping[str, Any]


@dataclass(frozen=True)
class Pair:
    first: Line
    second: Line
    direction: np.ndarray
    transverse: np.ndarray
    overlap_min: float
    overlap_max: float
    overlap: float
    overlap_fraction: float
    separation: float
    h: float
    theta_deg: float


@dataclass(frozen=True)
class StationResult:
    state: int
    reason: str
    theta_deg: float | None = None
    total_turn_deg: float | None = None
    alignment: float | None = None
    dominant_ratio: float | None = None
    core_residual: float | None = None
    centroid: np.ndarray | None = None


@dataclass(frozen=True)
class PairResult:
    first: int
    second: int
    states: tuple[int, ...]
    positive: bool
    quality: tuple[int, int, int]
    centroids: tuple[np.ndarray, ...]
    audit: Mapping[str, Any]


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not math.isfinite(v):
            raise SFBOCInputError("NONFINITE_AUDIT_VALUE")
        return v.hex()
    if value is None or isinstance(value, str):
        return value
    raise SFBOCInputError("UNSUPPORTED_AUDIT_VALUE")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _q(value: float, scale: float = 1_000_000.0) -> int:
    return int(round(float(value) * scale))


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SFBOCInputError("INVALID_DIRECTION")
    result /= norm
    for value in result:
        if value > 0.0:
            break
        if value < 0.0:
            result = -result
            break
    result[result == 0.0] = 0.0
    return result


def _hash_line(start: np.ndarray, end: np.ndarray) -> str:
    payload = np.concatenate((start, end)).astype(">f8", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()[:32]


def _line(row: Mapping[str, Any], ordinal: int) -> Line:
    try:
        start = np.asarray(row["line_start"], dtype=np.float64)
        end = np.asarray(row["line_end"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise SFBOCInputError("INVALID_LINE") from error
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise SFBOCInputError("INVALID_LINE")
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0.0 or not math.isfinite(length):
        raise SFBOCInputError("INVALID_LINE")
    direction = _canonical_direction(delta)
    if float(np.dot(delta, direction)) < 0.0:
        start, end = end.copy(), start.copy()
    return Line(ordinal, start, end, direction, length, _hash_line(start, end), row)


def _jacobi_eigh3(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-order 12-sweep symmetric 3x3 Jacobi eigensolver."""
    a = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    v = np.eye(3, dtype=np.float64)
    for _ in range(12):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = float(a[p, q])
            if apq == 0.0:
                continue
            tau = (float(a[q, q]) - float(a[p, p])) / (2.0 * apq)
            t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c
            for k in range(3):
                if k in (p, q):
                    continue
                akp, akq = float(a[k, p]), float(a[k, q])
                a[k, p] = a[p, k] = c * akp - s * akq
                a[k, q] = a[q, k] = s * akp + c * akq
            app, aqq = float(a[p, p]), float(a[q, q])
            a[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
            a[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
            a[p, q] = a[q, p] = 0.0
            vp, vq = v[:, p].copy(), v[:, q].copy()
            v[:, p], v[:, q] = c * vp - s * vq, s * vp + c * vq
    order = sorted(range(3), key=lambda i: (float(a[i, i]), tuple(float(x) for x in v[:, i])))
    values = np.asarray([a[i, i] for i in order], dtype=np.float64)
    vectors = np.column_stack([_canonical_direction(v[:, i]) for i in order])
    return values, vectors


def _principal_projector(projectors: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray | None, float]:
    total = float(np.sum(weights))
    if total <= 0.0:
        return None, 0.0
    matrix = np.einsum("i,ijk->jk", weights, projectors) / total
    values, vectors = _jacobi_eigh3(matrix)
    return vectors[:, 2], float(values[2])


class SourceGeometry:
    def __init__(self, points: np.ndarray, triangles: np.ndarray | None = None):
        raw = np.asarray(points, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise SFBOCInputError("INVALID_SOURCE_POINTS")
        raw = raw[np.isfinite(raw).all(axis=1)]
        if len(raw) > MAX_SOURCE_POINTS:
            raise SFBOCResourceFailure("SOURCE_CAP_NOOP")
        unique, inverse = np.unique(raw, axis=0, return_inverse=True)
        if len(unique) < 256:
            raise SFBOCResourceFailure("SOURCE_TOO_SMALL_NOOP")
        self.points = unique
        self.tree = cKDTree(unique, compact_nodes=True, balanced_tree=True)
        distances, _ = self.tree.query(unique, k=2, workers=1)
        self.nearest = np.asarray(distances[:, 1], dtype=np.float64)
        self.h0 = float(np.median(self.nearest))
        if self.h0 <= 0.0 or not math.isfinite(self.h0):
            raise SFBOCResourceFailure("INVALID_H0_NOOP")
        self.mesh = triangles is not None and len(np.asarray(triangles)) > 0
        self.edges: np.ndarray | None = None
        self.projectors = np.zeros((len(unique), 3, 3), dtype=np.float64)
        self.normal_valid = np.zeros(len(unique), dtype=bool)
        if self.mesh:
            self._mesh_geometry(raw, inverse, np.asarray(triangles, dtype=np.int64))
        else:
            self._point_geometry()

    def _mesh_geometry(self, raw: np.ndarray, inverse: np.ndarray, triangles: np.ndarray) -> None:
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise SFBOCInputError("INVALID_TRIANGLES")
        good = np.all((triangles >= 0) & (triangles < len(raw)), axis=1)
        remapped = inverse[triangles[good]]
        remapped = remapped[(remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])]
        accum = np.zeros_like(self.projectors)
        weights = np.zeros(len(self.points), dtype=np.float64)
        edge_set: set[tuple[int, int]] = set()
        for face in remapped:
            a, b, c = (int(v) for v in face)
            cross = np.cross(self.points[b] - self.points[a], self.points[c] - self.points[a])
            twice_area = float(np.linalg.norm(cross))
            if twice_area <= 0.0:
                continue
            normal = cross / twice_area
            projector = np.outer(normal, normal)
            area = 0.5 * twice_area
            for vertex in (a, b, c):
                accum[vertex] += area * projector
                weights[vertex] += area
            for first, second in ((a, b), (b, c), (c, a)):
                edge_set.add((min(first, second), max(first, second)))
        valid = weights > 0.0
        self.projectors[valid] = accum[valid] / weights[valid, None, None]
        self.normal_valid = valid
        self.edges = np.asarray(sorted(edge_set), dtype=np.int64)

    def _point_geometry(self) -> None:
        _distances, neighbours = self.tree.query(self.points, k=24, workers=1)
        for index, row in enumerate(np.asarray(neighbours)):
            local = self.points[row]
            center = np.mean(local, axis=0)
            delta = local - center
            covariance = delta.T @ delta / float(len(local))
            values, vectors = _jacobi_eigh3(covariance)
            if values[1] > 0.0 and values[0] / values[1] <= 0.25:
                normal = vectors[:, 0]
                self.projectors[index] = np.outer(normal, normal)
                self.normal_valid[index] = True


def _point_on_axis(line: Line, direction: np.ndarray, coordinate: float) -> np.ndarray:
    denominator = float(np.dot(line.direction, direction))
    if abs(denominator) <= 1e-12:
        raise SFBOCInputError("PAIR_AXIS_DEGENERATE")
    return line.start + ((coordinate - float(np.dot(line.start, direction))) / denominator) * line.direction


def _ordinary_nms_default(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    # Frozen V52 predicate, used only as an eligibility veto.
    a, b = _line(first, 0), _line(second, 1)
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(a.direction, b.direction)), -1.0, 1.0))))
    u = _canonical_direction(a.direction + math.copysign(1.0, float(np.dot(a.direction, b.direction))) * b.direction)
    lo = max(min(float(np.dot(a.start, u)), float(np.dot(a.end, u))), min(float(np.dot(b.start, u)), float(np.dot(b.end, u))))
    hi = min(max(float(np.dot(a.start, u)), float(np.dot(a.end, u))), max(float(np.dot(b.start, u)), float(np.dot(b.end, u))))
    overlap = max(0.0, hi - lo) / min(a.length, b.length)
    middle = 0.5 * (lo + hi)
    distance = float(np.linalg.norm((np.eye(3) - np.outer(u, u)) @ (_point_on_axis(b, u, middle) - _point_on_axis(a, u, middle))))
    return angle <= 7.0 and distance <= 6.0 and overlap >= 0.50


def _pair_geometry(source: SourceGeometry, first: Line, second: Line, nms: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]) -> Pair | None:
    dot = float(np.dot(first.direction, second.direction))
    theta = math.degrees(math.acos(float(np.clip(abs(dot), -1.0, 1.0))))
    if theta > 2.0:
        return None
    direction = _canonical_direction(first.direction + math.copysign(1.0, dot) * second.direction)
    first_interval = sorted((float(np.dot(first.start, direction)), float(np.dot(first.end, direction))))
    second_interval = sorted((float(np.dot(second.start, direction)), float(np.dot(second.end, direction))))
    lo, hi = max(first_interval[0], second_interval[0]), min(first_interval[1], second_interval[1])
    overlap = hi - lo
    if overlap <= 0.0:
        return None
    fraction = overlap / min(first.length, second.length)
    if fraction < 0.8:
        return None
    middle = 0.5 * (lo + hi)
    first_point, second_point = _point_on_axis(first, direction, middle), _point_on_axis(second, direction, middle)
    residual = (np.eye(3) - np.outer(direction, direction)) @ (second_point - first_point)
    separation = float(np.linalg.norm(residual))
    if separation <= 0.0:
        return None
    transverse = residual / separation
    # Pair-local source corridor for h.
    center = 0.5 * (first_point + second_point)
    candidates = source.tree.query_ball_point(center, r=math.sqrt((0.5 * overlap + 4.0 * source.h0) ** 2 + (20.0 * source.h0) ** 2))
    ids = np.asarray(sorted(candidates), dtype=np.int64)
    if len(ids):
        delta = source.points[ids] - center
        axial = delta @ direction
        transverse_distance = np.linalg.norm(delta - axial[:, None] * direction, axis=1)
        ids = ids[(np.abs(axial) <= 0.5 * overlap + 4.0 * source.h0) & (transverse_distance <= 20.0 * source.h0)]
    if len(ids) < 256:
        return None
    h = float(np.clip(np.median(source.nearest[ids]), 0.5 * source.h0, 2.0 * source.h0))
    if overlap < 40.0 * h or not (0.5 * h <= separation < 10.0 * h):
        return None
    if nms(first.original, second.original):
        return None
    return Pair(first, second, direction, transverse, lo, hi, overlap, fraction, separation, h, theta)


def _station_graph(source: SourceGeometry, ids: np.ndarray, h: float) -> list[list[tuple[int, float]]]:
    local = {int(global_id): local_id for local_id, global_id in enumerate(ids)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in ids]
    edge_count = 0
    if source.mesh:
        assert source.edges is not None
        for ga, gb in source.edges:
            a, b = local.get(int(ga)), local.get(int(gb))
            if a is None or b is None:
                continue
            distance = float(np.linalg.norm(source.points[int(ga)] - source.points[int(gb)]))
            if distance <= 4.0 * h:
                adjacency[a].append((b, distance))
                adjacency[b].append((a, distance))
                edge_count += 1
    else:
        local_points = source.points[ids]
        tree = cKDTree(local_points, compact_nodes=True, balanced_tree=True)
        _dist, near = tree.query(local_points, k=min(9, len(local_points)), workers=1)
        neighbour_sets = [set(int(v) for v in np.atleast_1d(row)[1:]) for row in near]
        for a in range(len(ids)):
            for b in sorted(neighbour_sets[a]):
                if b <= a or a not in neighbour_sets[b]:
                    continue
                delta = local_points[b] - local_points[a]
                distance = float(np.linalg.norm(delta))
                if distance > 3.0 * h or distance <= 0.0:
                    continue
                normals = []
                for global_id in (int(ids[a]), int(ids[b])):
                    values, vectors = _jacobi_eigh3(source.projectors[global_id])
                    normals.append(vectors[:, 2])
                tangent_error = min(abs(float(np.dot(delta, normals[0]))), abs(float(np.dot(delta, normals[1]))))
                if tangent_error <= 0.75 * distance:
                    adjacency[a].append((b, distance))
                    adjacency[b].append((a, distance))
                    edge_count += 1
    if edge_count > MAX_STATION_EDGES:
        raise SFBOCResourceFailure("STATION_EDGE_CAP")
    for row in adjacency:
        row.sort(key=lambda item: item[0])
    return adjacency


def _dijkstra(points: np.ndarray, adjacency: list[list[tuple[int, float]]], left: Sequence[int], right: set[int], direction: np.ndarray) -> list[int] | None:
    distances = np.full(len(points), np.inf, dtype=np.float64)
    predecessor = np.full(len(points), -1, dtype=np.int64)
    queue: list[tuple[float, tuple[float, float, float], int]] = []
    for seed in sorted(left, key=lambda i: tuple(float(v) for v in points[i])):
        distances[seed] = 0.0
        heapq.heappush(queue, (0.0, tuple(float(v) for v in points[seed]), seed))
    target = None
    while queue:
        distance, _key, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node in right:
            target = node
            break
        for neighbour, length in adjacency[node]:
            delta = points[neighbour] - points[node]
            cost = length * (1.0 + 4.0 * (float(np.dot(delta, direction)) ** 2) / (length * length))
            proposal = distance + cost
            if proposal < distances[neighbour] or (proposal == distances[neighbour] and node < predecessor[neighbour]):
                distances[neighbour] = proposal
                predecessor[neighbour] = node
                heapq.heappush(queue, (proposal, tuple(float(v) for v in points[neighbour]), neighbour))
    if target is None:
        return None
    path = []
    while target >= 0:
        path.append(int(target))
        target = int(predecessor[target])
    path.reverse()
    return path


def _smooth3(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return values.copy()
    result = np.empty_like(values)
    result[0] = (2.0 * values[0] + values[1]) / 3.0
    result[-1] = (values[-2] + 2.0 * values[-1]) / 3.0
    if len(values) > 2:
        result[1:-1] = (values[:-2] + 2.0 * values[1:-1] + values[2:]) / 4.0
    return result


def _close_one(active: np.ndarray) -> np.ndarray:
    result = active.copy()
    for i in range(1, len(active) - 1):
        if not active[i] and active[i - 1] and active[i + 1]:
            result[i] = True
    return result


def _components(active: np.ndarray) -> list[tuple[int, int]]:
    result = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(active) - 1):
            end = i if value and i == len(active) - 1 else i - 1
            result.append((start, end))
            start = None
    return result


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(v) for v in values)
    return ordered[max(0, int(math.ceil(probability * len(ordered))) - 1)]


def _evaluate_station(source: SourceGeometry, pair: Pair, station_index: int) -> StationResult:
    h = pair.h
    coordinate = pair.overlap_min + station_index * pair.overlap / 10.0
    pi = _point_on_axis(pair.first, pair.direction, coordinate)
    pj = _point_on_axis(pair.second, pair.direction, coordinate)
    center = 0.5 * (pi + pj)
    nearby = source.tree.query_ball_point(center, r=math.sqrt((24.0 * h) ** 2 + (3.0 * h) ** 2))
    ids = np.asarray(sorted(nearby), dtype=np.int64)
    if len(ids):
        delta = source.points[ids] - center
        axial = delta @ pair.direction
        radial = np.linalg.norm(delta - axial[:, None] * pair.direction, axis=1)
        ids = ids[(np.abs(axial) <= 3.0 * h) & (radial <= 24.0 * h)]
    if len(ids) > MAX_STATION_NODES:
        return StationResult(INDETERMINATE, "NODE_CAP")
    if len(ids) < 24:
        return StationResult(INDETERMINATE, "MISSING_SUPPORT")
    points = source.points[ids]
    t = (points - center) @ pair.transverse
    left = np.flatnonzero((-0.5 * pair.separation - 10.0 * h <= t) & (t <= -0.5 * pair.separation - 6.0 * h))
    right_array = np.flatnonzero((0.5 * pair.separation + 6.0 * h <= t) & (t <= 0.5 * pair.separation + 10.0 * h))
    if len(left) < 12 or len(right_array) < 12:
        return StationResult(INDETERMINATE, "MISSING_TAIL_SEEDS")
    try:
        adjacency = _station_graph(source, ids, h)
    except SFBOCResourceFailure:
        return StationResult(INDETERMINATE, "EDGE_CAP")
    path_local = _dijkstra(points, adjacency, left.tolist(), set(int(v) for v in right_array), pair.direction)
    if path_local is None or len(path_local) < 3:
        return StationResult(INDETERMINATE, "GRAPH_FAILURE")
    path_global = ids[np.asarray(path_local, dtype=np.int64)]
    path_points = source.points[path_global]
    segments = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segments)))
    length = float(arc[-1])
    if length > 48.0 * h:
        return StationResult(INDETERMINATE, "PATH_TOO_LONG")
    backtrack = float(np.sum(np.maximum(0.0, -np.diff((path_points - center) @ pair.transverse))))
    if backtrack > 2.0 * h:
        return StationResult(INDETERMINATE, "PATH_BACKTRACK")
    line_path_distances = (float(np.min(np.linalg.norm(path_points - pi, axis=1))), float(np.min(np.linalg.norm(path_points - pj, axis=1))))
    if max(line_path_distances) > 3.0 * h:
        return StationResult(INDETERMINATE, "LINE_OFF_PATH")
    samples = np.minimum(np.arange(int(math.ceil(length / h)) + 1, dtype=np.float64) * h, length)
    normals = []
    confidences = []
    for z in samples:
        weights = np.maximum(0.0, 1.0 - np.abs(arc - z) / (1.5 * h))
        valid = source.normal_valid[path_global]
        normal, confidence = _principal_projector(source.projectors[path_global[valid]], weights[valid])
        if normal is None or confidence < 0.85:
            return StationResult(INDETERMINATE, "AMBIGUOUS_NORMAL")
        if normals and float(np.dot(normals[-1], normal)) < 0.0:
            normal = -normal
        normals.append(normal)
        confidences.append(confidence)
    normals_array = np.asarray(normals)
    delta_turn = np.degrees(np.arccos(np.clip(np.sum(normals_array[:-1] * normals_array[1:], axis=1), -1.0, 1.0)))
    total_turn = float(np.sum(delta_turn))
    if len(delta_turn) < 2 or total_turn <= 0.0:
        return StationResult(INDETERMINATE, "WEAK_TURN")
    left_ids = np.flatnonzero(samples <= 4.0 * h)
    right_ids = np.flatnonzero(samples >= length - 4.0 * h)
    left_normal, left_conf = _principal_projector(source.projectors[path_global], np.maximum(0.0, 1.0 - np.minimum(np.abs(arc[:, None] - samples[left_ids][None, :]), axis=1) / (1.5 * h)))
    right_normal, right_conf = _principal_projector(source.projectors[path_global], np.maximum(0.0, 1.0 - np.minimum(np.abs(arc[:, None] - samples[right_ids][None, :]), axis=1) / (1.5 * h)))
    if left_normal is None or right_normal is None or left_conf < 0.95 or right_conf < 0.95:
        return StationResult(INDETERMINATE, "TAIL_AMBIGUITY")
    tail_left_turn = float(np.sum(delta_turn[samples[:-1] < 4.0 * h]))
    tail_right_turn = float(np.sum(delta_turn[samples[1:] > length - 4.0 * h]))
    theta = math.degrees(math.acos(float(np.clip(abs(np.dot(left_normal, right_normal)), -1.0, 1.0))))
    if tail_left_turn > 4.0 or tail_right_turn > 4.0 or theta < 12.0:
        return StationResult(INDETERMINATE, "TAIL_CERTIFICATE")
    excessive_return = total_turn >= 1.5 * theta + 10.0
    axes = np.cross(normals_array[:-1], normals_array[1:])
    active_axes = delta_turn >= 0.35
    if not np.any(active_axes):
        return StationResult(INDETERMINATE, "NO_AXIS_MASS")
    axes_norm = np.linalg.norm(axes, axis=1)
    aligned = active_axes & (axes_norm > 0.0) & (np.abs((axes @ pair.direction) / np.maximum(axes_norm, 1e-300)) >= math.cos(math.radians(20.0)))
    alignment = float(np.sum(delta_turn[aligned]) / np.sum(delta_turn[active_axes]))
    if alignment < 0.85:
        return StationResult(INDETERMINATE, "AXIS_ALIGNMENT")
    g1, g2 = _smooth3(delta_turn), _smooth3(_smooth3(delta_turn))
    dominant = []
    split_components: list[tuple[int, int]] = []
    dominant_ratio = 0.0
    for scale_index, profile in enumerate((g1, g2)):
        threshold = max(0.35, 0.025 * total_turn)
        components = _components(_close_one(profile >= threshold))
        masses = [float(np.sum(delta_turn[a : b + 1])) for a, b in components]
        if not components:
            return StationResult(INDETERMINATE, "NO_COMPONENT")
        order = sorted(range(len(components)), key=lambda i: (-masses[i], components[i]))
        best = order[0]
        ratio = masses[best] / total_turn
        other = (sum(masses) - masses[best]) / total_turn
        if ratio < 0.75 or other > 0.10:
            if scale_index == 0:
                split_components = [components[i] for i, mass in enumerate(masses) if mass >= 0.20 * total_turn]
            dominant.append(None)
        else:
            dominant.append(components[best])
            dominant_ratio = max(dominant_ratio, ratio)
        for level in (0.50, 0.75):
            if len(_components(_close_one(profile >= level * float(np.max(profile))))) != 1:
                if scale_index == 0:
                    split_components = [components[i] for i, mass in enumerate(masses) if mass >= 0.20 * total_turn]
                dominant[-1] = None
    zi = float(arc[int(np.argmin(np.linalg.norm(path_points - pi, axis=1)))])
    zj = float(arc[int(np.argmin(np.linalg.norm(path_points - pj, axis=1)))])
    if len(split_components) >= 2:
        separated = []
        for component in split_components:
            separated.append((samples[component[0]], samples[min(component[1] + 1, len(samples) - 1)]))
        for a in range(len(separated)):
            for b in range(a + 1, len(separated)):
                first_interval, second_interval = separated[a], separated[b]
                gap = second_interval[0] - first_interval[1]
                if gap >= 2.0 * h:
                    in_first = first_interval[0] - 2.0 * h <= zi <= first_interval[1] + 2.0 * h
                    in_second = second_interval[0] - 2.0 * h <= zj <= second_interval[1] + 2.0 * h
                    reverse = first_interval[0] - 2.0 * h <= zj <= first_interval[1] + 2.0 * h and second_interval[0] - 2.0 * h <= zi <= second_interval[1] + 2.0 * h
                    if (in_first and in_second) or reverse:
                        return StationResult(SPLIT_BASIN, "TWO_LOBE_SPLIT", theta, total_turn, alignment, dominant_ratio)
    if excessive_return:
        return StationResult(SPLIT_BASIN, "RETURN_EXCESS", theta, total_turn, alignment, dominant_ratio)
    if total_turn > 1.25 * theta + 6.0 or dominant[0] is None or dominant[1] is None:
        return StationResult(INDETERMINATE, "PERSISTENCE")
    fine, coarse = dominant  # type: ignore[misc]
    fine_interval = (samples[fine[0]], samples[min(fine[1] + 1, len(samples) - 1)])
    coarse_interval = (samples[coarse[0]], samples[min(coarse[1] + 1, len(samples) - 1)])
    intersection = max(0.0, min(fine_interval[1], coarse_interval[1]) - max(fine_interval[0], coarse_interval[0]))
    union = max(fine_interval[1], coarse_interval[1]) - min(fine_interval[0], coarse_interval[0])
    if union <= 0.0 or intersection / union < 0.60:
        return StationResult(INDETERMINATE, "NONPERSISTENT")
    for interval in (fine_interval, coarse_interval):
        if not (interval[0] - 5.0 * h <= zi <= interval[1] + 5.0 * h and interval[0] - 5.0 * h <= zj <= interval[1] + 5.0 * h):
            return StationResult(INDETERMINATE, "LINES_OUTSIDE_BASIN")
    if fine_interval[0] < 4.0 * h or length - fine_interval[1] < 4.0 * h:
        return StationResult(INDETERMINATE, "INSUFFICIENT_FLANK")
    indices = np.arange(fine[0], fine[1] + 1)
    centers = 0.5 * (samples[indices] + samples[np.minimum(indices + 1, len(samples) - 1)])
    weights = g1[indices]
    centroid_arc = float(np.sum(weights * centers) / np.sum(weights))
    segment_index = min(len(path_points) - 2, int(np.searchsorted(arc, centroid_arc, side="right") - 1))
    fraction = (centroid_arc - arc[segment_index]) / max(segments[segment_index], 1e-300)
    centroid = path_points[segment_index] + fraction * (path_points[segment_index + 1] - path_points[segment_index])
    core_residual = 0.5 * (float(np.linalg.norm(centroid - pi)) + float(np.linalg.norm(centroid - pj))) / h
    return StationResult(SAME_BASIN, "POSITIVE", theta, total_turn, alignment, dominant_ratio, core_residual, centroid)


def _median_or_zero(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _evaluate_pair(source: SourceGeometry, pair: Pair) -> PairResult:
    stations = tuple(_evaluate_station(source, pair, index) for index in range(1, 10))
    same = sum(item.state == SAME_BASIN for item in stations)
    split = sum(item.state == SPLIT_BASIN for item in stations)
    positive = same >= 7 and split == 0
    core = [item.core_residual for item in stations if item.state == SAME_BASIN and item.core_residual is not None]
    median_core = _median_or_zero(core)
    quality = (same, -_q(median_core), -_q(pair.separation / pair.h))
    def med(field: str) -> int:
        values = [getattr(item, field) for item in stations if getattr(item, field) is not None]
        return _q(_median_or_zero(values))
    audit = {
        "candidate_hashes": [pair.first.hash128, pair.second.hash128],
        "theta_microdegrees": _q(pair.theta_deg),
        "overlap_fraction_e6": _q(pair.overlap_fraction),
        "separation_over_h_e6": _q(pair.separation / pair.h),
        "station_states": [item.state for item in stations],
        "station_reasons": [item.reason for item in stations],
        "same": same,
        "split": split,
        "indeterminate": 9 - same - split,
        "median_theta_microdegrees": med("theta_deg"),
        "median_total_turn_microdegrees": med("total_turn_deg"),
        "median_alignment_e6": med("alignment"),
        "median_dominant_ratio_e6": med("dominant_ratio"),
        "median_core_residual_e6": _q(median_core),
        "positive_edge": positive,
        "reason": "POSITIVE_7_OF_9" if positive else ("SPLIT_VETO" if split else "INSUFFICIENT_POSITIVE"),
    }
    centroids = tuple(item.centroid for item in stations if item.state == SAME_BASIN and item.centroid is not None)
    return PairResult(pair.first.ordinal, pair.second.ordinal, tuple(item.state for item in stations), positive, quality, centroids, audit)


def _clusters(lines: Sequence[Line], pair_results: Sequence[PairResult]) -> tuple[list[list[int]], list[Mapping[str, Any]]]:
    positive = {(min(p.first, p.second), max(p.first, p.second)): p for p in pair_results if p.positive}
    # Oversized positive connected components are wholly frozen.
    connected = {line.ordinal: set() for line in lines}
    for a, b in positive:
        connected[a].add(b); connected[b].add(a)
    oversized: set[int] = set()
    visited: set[int] = set()
    for node in connected:
        if node in visited:
            continue
        stack, component = [node], []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current); component.append(current); stack.extend(sorted(connected[current], reverse=True))
        if len(component) > MAX_CLUSTER_SIZE:
            oversized.update(component)
    clusters = [[line.ordinal] for line in lines]
    merge_records = []
    while True:
        options = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                merged = sorted(clusters[i] + clusters[j])
                if len(merged) > MAX_CLUSTER_SIZE or any(v in oversized for v in merged):
                    continue
                cross = [positive.get((min(a, b), max(a, b))) for a in clusters[i] for b in clusters[j]]
                if any(item is None for item in cross):
                    continue
                quality = min(item.quality for item in cross if item is not None)
                hashes = tuple(sorted(lines[v].hash128 for v in merged))
                options.append((quality, hashes, i, j))
        if not options:
            break
        best_quality = max(item[0] for item in options)
        chosen = min((item for item in options if item[0] == best_quality), key=lambda item: item[1])
        _quality, hashes, i, j = chosen
        merged = sorted(clusters[i] + clusters[j])
        merge_records.append({"left": [lines[v].hash128 for v in clusters[i]], "right": [lines[v].hash128 for v in clusters[j]], "quality": list(best_quality), "merged": list(hashes)})
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in (i, j)] + [merged]
        clusters.sort(key=lambda cluster: tuple(lines[v].hash128 for v in cluster))
    return clusters, merge_records


def _segment_residual(point: np.ndarray, line: Line, expansion: float) -> tuple[bool, float]:
    direction = line.direction
    coordinate = float(np.dot(point - line.start, direction))
    covered = -expansion <= coordinate <= line.length + expansion
    projection = line.start + coordinate * direction
    return covered, float(np.linalg.norm(point - projection))


def _noop(baseline: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], reason: str, h0: float = 0.0) -> tuple[dict[str, Any], tuple[PairResult, ...]]:
    result = deepcopy(dict(baseline)); result["predictions"] = [deepcopy(row) for row in rows]
    audit = {"profile": PROFILE, "lock_token": LOCK_TOKEN, "status": reason, "h0_e6": _q(h0), "incoming_candidate_count": len(rows), "suppressed_hashes": []}
    result["sfboc_v1_audit"] = _native(audit)
    result["sfboc_v1_audit_sha256"] = hashlib.sha256(canonical_bytes(audit)).hexdigest()
    return result, tuple()


def apply_sfboc_v1(
    baseline: Mapping[str, Any],
    points: np.ndarray,
    triangles: np.ndarray | None = None,
    *,
    ordinary_nms_pair: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], tuple[PairResult, ...]]:
    """Apply the locked pre-NMS ownership consolidator to accepted lines."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    rows = list(baseline.get("predictions", ()))
    if len(rows) > MAX_CANDIDATES:
        return _noop(baseline, rows, "CANDIDATE_CAP_NOOP")
    try:
        lines = [_line(row, ordinal) for ordinal, row in enumerate(rows)]
        source = SourceGeometry(points, triangles)
        nms = ordinary_nms_pair or _ordinary_nms_default
        pairs = []
        preliminary = 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                preliminary += 1
                pair = _pair_geometry(source, lines[i], lines[j], nms)
                if pair is not None:
                    pairs.append(pair)
        pairs.sort(key=lambda pair: (pair.first.hash128, pair.second.hash128))
        if len(pairs) > MAX_ELIGIBLE_PAIRS:
            return _noop(baseline, rows, "PAIR_CAP_NOOP", source.h0)
        pair_results = []
        for pair in pairs:
            if time.perf_counter() - started_wall >= WALL_SECONDS:
                raise SFBOCResourceFailure("WALL_CAP_NOOP")
            pair_results.append(_evaluate_pair(source, pair))
        clusters, merge_records = _clusters(lines, pair_results)
        pair_lookup = {(min(p.first, p.second), max(p.first, p.second)): p for p in pair_results}
        suppressed: set[int] = set()
        cluster_records = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            cores = []
            for a_index, a in enumerate(cluster):
                for b in cluster[a_index + 1:]:
                    result = pair_lookup.get((min(a, b), max(a, b)))
                    if result is not None and result.positive:
                        cores.extend(result.centroids)
            if not cores:
                continue
            ranks = []
            for ordinal in cluster:
                line = lines[ordinal]
                covered, residuals = [], []
                for core in cores:
                    inside, residual = _segment_residual(core, line, 2.0 * source.h0)
                    covered.append(inside); residuals.append(residual / source.h0)
                rank = (-sum(covered), _q(_median_or_zero(residuals)), _q(_nearest_rank(residuals, 0.90)), -_q(line.length), line.hash128)
                ranks.append((rank, ordinal))
            ranks.sort()
            representative = ranks[0][1]
            removed = [ordinal for ordinal in cluster if ordinal != representative]
            suppressed.update(removed)
            cluster_records.append({"members": [lines[v].hash128 for v in cluster], "representative": lines[representative].hash128, "representative_rank": list(ranks[0][0]), "suppressed": [lines[v].hash128 for v in removed]})
        survivors = [deepcopy(row) for ordinal, row in enumerate(rows) if ordinal not in suppressed]
        semantic = {"profile": PROFILE, "predictions": survivors}
        audit = {
            "profile": PROFILE,
            "lock_token": LOCK_TOKEN,
            "status": "OK",
            "source_count": int(len(np.asarray(points))),
            "unique_source_count": len(source.points),
            "h0_e6": _q(source.h0),
            "incoming_candidate_count": len(rows),
            "preliminary_pair_count": preliminary,
            "eligible_pair_count": len(pairs),
            "pair_records": [item.audit for item in pair_results],
            "complete_link_merges": merge_records,
            "cluster_records": cluster_records,
            "suppressed_hashes": [lines[v].hash128 for v in sorted(suppressed)],
            "semantic_sha256": hashlib.sha256(canonical_bytes(semantic)).hexdigest(),
        }
        audit_bytes = canonical_bytes(audit)
        if len(audit_bytes) > MAX_AUDIT_BYTES:
            return _noop(baseline, rows, "AUDIT_CAP_NOOP", source.h0)
        result = deepcopy(dict(baseline)); result["predictions"] = survivors
        result["sfboc_v1_audit"] = _native(audit)
        result["sfboc_v1_audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
        # Useful resource evidence, explicitly unhashed/non-semantic.
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["sfboc_v1_runtime_unhashed"] = {
            "wall_time_us": _q(time.perf_counter() - started_wall),
            "cpu_time_us": _q(time.process_time() - started_cpu),
            "peak_rss_native": peak,
        }
        return result, tuple(pair_results)
    except (SFBOCResourceFailure, MemoryError) as error:
        return _noop(baseline, rows, str(error) if not isinstance(error, MemoryError) else "MEMORY_CAP_NOOP")


def frozen_configuration() -> Mapping[str, Any]:
    return {
        "profile": PROFILE,
        "short_name": SHORT_NAME,
        "lock_token": LOCK_TOKEN,
        "caps": {"source": MAX_SOURCE_POINTS, "candidates": MAX_CANDIDATES, "pairs": MAX_ELIGIBLE_PAIRS, "station_nodes": MAX_STATION_NODES, "station_edges": MAX_STATION_EDGES, "cluster": MAX_CLUSTER_SIZE, "audit_bytes": MAX_AUDIT_BYTES, "wall_seconds": WALL_SECONDS},
        "stations": [index / 10.0 for index in range(1, 10)],
        "positive_pair": {"same_at_least": 7, "split_exactly": 0},
        "mutation": "suppress_nonrepresentatives_only_before_ordinary_nms",
    }


CONFIGURATION_SHA256 = hashlib.sha256(canonical_bytes(frozen_configuration())).hexdigest()
