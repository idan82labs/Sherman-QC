"""V30 axial multipath fold certificate (AMFC_V1).

Research-only, source-only rescue sidecar for the frozen V16 selector.  AMFC
never creates a candidate and never vetoes or mutates a V16-accepted record.
It may only certify an already-existing rejected candidate, append certified
records after the untouched legacy sequence, and feed that one sequence to the
single frozen ordinary NMS pass.

The module intentionally has no file I/O and no access to identity, labels,
CAD, range frames, evaluator state, or production/runtime code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import struct
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


PROFILE = "V30_AXIAL_MULTIPATH_FOLD_CERTIFICATE_V1"
PRIMITIVE = "CANDIDATE_CONDITIONED_TANGENT_GRAPH_NODE_DISJOINT_PATHS_V1"
SCHEMA_VERSION = 1

NORMAL_NEIGHBORS = 16
NORMAL_RADIUS_S = 3.0
NORMAL_MINIMUM_NEIGHBORS = 8
NORMAL_JACOBI_SWEEPS = 12
SURFACE_VARIATION_MAXIMUM = 0.12

STATION_COUNT = 9
STATION_HALF_WIDTH_S = 6.0
STATION_RADIUS_S = 32.0
STATION_END_MARGIN_S = 8.0
MAX_STATION_NODES = 8_192
STATION_VOXEL_S = 0.75
RECIPROCAL_NEIGHBORS = 12
MAX_STATION_EDGES = 98_304

TAIL_MIN_S = 4.0
TAIL_MAX_S = 12.0
TAIL_PLANE_DISTANCE_S = 1.25
TAIL_NORMAL_DEGREES = 25.0
TAIL_MINIMUM_NODES = 24
TAIL_MINIMUM_AXIAL_SPAN_S = 8.0
TAIL_MINIMUM_OUTWARD_SPAN_S = 4.0

EDGE_MINIMUM_S = 0.2
EDGE_MAXIMUM_S = 2.75
EDGE_MAXIMUM_AXIAL_S = 2.25
EDGE_TANGENT_RATIO_MAXIMUM = 0.30
EDGE_NORMAL_DEGREES_MAXIMUM = 32.0

MAX_PATHS = 8
PATH_MINIMUM_EDGES = 5
PATH_MAXIMUM_EDGES = 64
PATH_MAXIMUM_LENGTH_S = 40.0
PATH_MAXIMUM_CHORD_RATIO = 1.65
PATH_ENDPOINT_NORMAL_DEGREES = 15.0
PATH_MINIMUM_INTERNAL_NODES = 3
PATH_MAXIMUM_AXIAL_EXCURSION_S = 8.0
PATH_TURN_ERROR_DEGREES = 12.0
PATH_MONOTONICITY_MINIMUM = 0.72

MINIMUM_TARGET_TURN_DEGREES = 15.0
MAXIMUM_TARGET_TURN_DEGREES = 165.0
MINIMUM_VALID_PATHS_PER_STATION = 4
MINIMUM_PATH_MIDPOINT_SPAN_S = 4.0
MINIMUM_TAIL_VALID_STATIONS = 7
MINIMUM_VALID_STATIONS = 6
MINIMUM_CONSECUTIVE_VALID_STATIONS = 4
MINIMUM_VALID_PATHS_TOTAL = 24
MEDIAN_TURN_ERROR_DEGREES = 8.0

MAX_ELIGIBLE_REJECTED_CANDIDATES = 256
MAX_SOURCE_POINTS = 200_000
MAX_SOURCE_TRIANGLES = 400_000
MAX_WORKERS = 16
MAX_CANDIDATE_RECORD_BYTES = 512
MAX_SUMMARY_BYTES = 132 * 1024
COORDINATE_QUANTIZATION_DIVISOR = 1024
TURN_QUANTIZATION_MICRODEGREES = 1_000_000

NMS_ANGLE_DEGREES = 7.0
NMS_DISTANCE_MM = 6.0
NMS_MINIMUM_OVERLAP = 0.60

ACTION_LEGACY = "LEGACY"
ACTION_RESCUE = "AMFC_PASS"
ACTION_NONE = "NO_RESCUE"

REASON_FRAME_INVALID = "FRAME_INVALID"
REASON_SPAN_SHORT = "SPAN_SHORT"
REASON_TAIL_MISSING = "TAIL_MISSING"
REASON_GRAPH_DISCONNECTED = "GRAPH_DISCONNECTED"
REASON_PATH_BOTTLENECK = "PATH_BOTTLENECK"
REASON_TURN_TOO_SMALL = "TURN_TOO_SMALL"
REASON_TURN_MISMATCH = "TURN_MISMATCH"
REASON_TURN_NONMONOTONE = "TURN_NONMONOTONE"
REASON_AXIAL_COVERAGE = "AXIAL_COVERAGE"
REASON_RESOURCE_CAP = "RESOURCE_CAP"

SOURCE_SCHEMA_FIELDS = ("coordinates", "triangles")
CANDIDATE_SCHEMA_FIELDS = (
    "candidate_key",
    "v16_rank",
    "selected_legacy",
    "line_center",
    "line_axis",
    "plane_normals",
    "plane_offsets",
    "parent_support_centroids",
    "support_interval",
    "line_start",
    "line_end",
)

PROHIBITED_FIELD_TOKENS = frozenset(
    {
        "filename",
        "directory",
        "part_number",
        "part_id",
        "family",
        "customer",
        "identity",
        "content_hash",
        "arrival_order",
        "prospective_selection_key",
        "expected_count",
        "good_scan",
        "bad_scan",
        "deviation_status",
        "label",
        "gold",
        "manual_selection",
        "adjudication",
        "benchmark",
        "cad",
        "step",
        "drawing",
        "mbd",
        "revit",
        "nominal",
        "tolerance",
        "acquisition_pose",
        "intrinsics",
        "range_frame",
        "visibility_ray",
        "v29",
        "color",
        "intensity",
        "texture",
        "reflectivity",
        "production",
        "runtime",
    }
)


class AMFCContractError(RuntimeError):
    """Fail-closed V30 schema, numerical, solver, or resource violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    serializable = {key: value for key, value in result.items() if key != "_ephemeral"}
    return _canonical_json_bytes(serializable) + b"\n"


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AMFCContractError("NONFINITE_VALUE", label)
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise AMFCContractError("SIGNED_ZERO_VALUE", label)
    return result


def _array(value: Any, shape: tuple[int, ...], label: str, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise AMFCContractError("INVALID_ARRAY_SHAPE", label)
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all():
            raise AMFCContractError("NONFINITE_ARRAY", label)
        if np.any((result == 0.0) & np.signbit(result)):
            raise AMFCContractError("SIGNED_ZERO_VALUE", label)
    return result


def _unit(value: Any, label: str) -> tuple[np.ndarray, float]:
    raw = _array(value, (3,), label)
    norm = float(np.linalg.norm(raw))
    if norm <= 0.0:
        raise AMFCContractError("ZERO_VECTOR", label)
    return raw / norm, norm


def _quantized_tuple(point: np.ndarray, quantum: float) -> tuple[int, int, int]:
    return tuple(_round_half_away(float(component) / quantum) for component in point)  # type: ignore[return-value]


def _float_bits_tuple(point: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(point, dtype="<f8").view("<u8")
    return int(values[0]), int(values[1]), int(values[2])


def _node_key(point: np.ndarray, quantum: float) -> tuple[int, ...]:
    return _quantized_tuple(point, quantum) + _float_bits_tuple(point)


def _candidate_id(candidate_key: str) -> str:
    return hashlib.sha256(("V30_AMFC_V1" + candidate_key).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceGeometry:
    points: np.ndarray
    triangles: np.ndarray
    node_keys: tuple[tuple[int, ...], ...]
    mesh_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Candidate:
    raw: dict[str, Any]
    candidate_key: str
    candidate_id: str
    v16_rank: int
    selected_legacy: bool
    center: np.ndarray
    axis: np.ndarray
    normals: tuple[np.ndarray, np.ndarray]
    offsets: tuple[float, float]
    centroids: tuple[np.ndarray, np.ndarray]
    tails: tuple[np.ndarray, np.ndarray]
    support_interval: tuple[float, float]
    line_start: np.ndarray
    line_end: np.ndarray
    target_turns: tuple[float, ...]
    frame_valid: bool
    span_valid: bool


def _contains_prohibited_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            for token in PROHIBITED_FIELD_TOKENS:
                if token in lowered:
                    return str(key)
            found = _contains_prohibited_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_prohibited_key(nested)
            if found is not None:
                return found
    return None


def parse_source(raw: dict[str, Any], s: float) -> SourceGeometry:
    if set(raw) != set(SOURCE_SCHEMA_FIELDS):
        raise AMFCContractError("SOURCE_SCHEMA_MISMATCH", repr(sorted(raw)))
    prohibited = _contains_prohibited_key(raw)
    if prohibited is not None:
        raise AMFCContractError("PROHIBITED_INPUT_FIELD", prohibited)
    points = np.asarray(raw["coordinates"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise AMFCContractError("INVALID_SOURCE_COORDINATES")
    if len(points) == 0 or len(points) > MAX_SOURCE_POINTS:
        raise AMFCContractError(REASON_RESOURCE_CAP, "source_point_count")
    if np.any((points == 0.0) & np.signbit(points)):
        raise AMFCContractError("SIGNED_ZERO_VALUE", "source_coordinates")

    triangles = np.asarray(raw["triangles"], dtype=np.int64)
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise AMFCContractError("INVALID_TRIANGLE_ARRAY")
    if len(triangles) > MAX_SOURCE_TRIANGLES:
        raise AMFCContractError(REASON_RESOURCE_CAP, "triangle_count")
    if len(triangles) and (int(triangles.min()) < 0 or int(triangles.max()) >= len(points)):
        raise AMFCContractError("TRIANGLE_INDEX_RANGE")

    quantum = s / COORDINATE_QUANTIZATION_DIVISOR
    quantized = np.asarray(np.rint(points / quantum), dtype=np.int64)
    bits = points.astype("<f8", copy=False).view("<u8")
    order = np.lexsort((bits[:, 2], bits[:, 1], bits[:, 0], quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    points = points[order]
    quantized = quantized[order]
    bits = bits[order]
    if len(points) > 1:
        duplicate = np.all(quantized[1:] == quantized[:-1], axis=1) & np.all(bits[1:] == bits[:-1], axis=1)
        if bool(np.any(duplicate)):
            raise AMFCContractError("DUPLICATE_SOURCE_POINT")
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    if len(triangles):
        triangles = inverse[triangles]

    edges: set[tuple[int, int]] = set()
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for first, second in ((a, b), (b, c), (c, a)):
            if first != second:
                edges.add((min(first, second), max(first, second)))
    keys = tuple(tuple(int(v) for v in (*quantized[index], *bits[index])) for index in range(len(points)))
    return SourceGeometry(points=points, triangles=triangles, node_keys=keys, mesh_edges=tuple(sorted(edges)))


def _canonical_axis(axis: np.ndarray, support: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float]]:
    for value in axis:
        if abs(float(value)) > 1.0e-12:
            if float(value) < 0.0:
                return -axis, (-support[1], -support[0])
            break
    return axis, support


def parse_candidates(raw_candidates: Sequence[dict[str, Any]], s: float) -> list[Candidate]:
    parsed: list[Candidate] = []
    seen: set[str] = set()
    for index, raw_value in enumerate(raw_candidates):
        raw = dict(raw_value)
        if set(raw) != set(CANDIDATE_SCHEMA_FIELDS):
            raise AMFCContractError("CANDIDATE_SCHEMA_MISMATCH", str(index))
        prohibited = _contains_prohibited_key(raw)
        if prohibited is not None:
            raise AMFCContractError("PROHIBITED_INPUT_FIELD", prohibited)
        key = str(raw["candidate_key"])
        if not key or key in seen:
            raise AMFCContractError("CANDIDATE_KEY_INVALID", key)
        seen.add(key)
        rank = int(raw["v16_rank"])
        if isinstance(raw["selected_legacy"], bool) is False:
            raise AMFCContractError("LEGACY_BIT_INVALID", key)
        selected = bool(raw["selected_legacy"])
        center = _array(raw["line_center"], (3,), f"{key}.line_center")
        axis_raw = _array(raw["line_axis"], (3,), f"{key}.line_axis")
        axis_norm = float(np.linalg.norm(axis_raw))
        support_raw = _array(raw["support_interval"], (2,), f"{key}.support_interval")
        support = (float(support_raw[0]), float(support_raw[1]))
        axis = axis_raw / axis_norm if axis_norm > 0.0 else axis_raw
        axis, support = _canonical_axis(axis, support)
        normals_raw = _array(raw["plane_normals"], (2, 3), f"{key}.plane_normals")
        offsets_raw = _array(raw["plane_offsets"], (2,), f"{key}.plane_offsets")
        centroids_raw = _array(raw["parent_support_centroids"], (2, 3), f"{key}.parent_support_centroids")
        line_start = _array(raw["line_start"], (3,), f"{key}.line_start")
        line_end = _array(raw["line_end"], (3,), f"{key}.line_end")

        normals: list[np.ndarray] = []
        offsets: list[float] = []
        frame_valid = bool(axis_norm > 0.0 and abs(axis_norm - 1.0) <= 1.0e-4)
        tails: list[np.ndarray] = []
        for plane_index in range(2):
            normal_norm = float(np.linalg.norm(normals_raw[plane_index]))
            if normal_norm <= 0.0:
                normal = normals_raw[plane_index]
                offset = float(offsets_raw[plane_index])
                frame_valid = False
            else:
                normal = normals_raw[plane_index] / normal_norm
                offset = float(offsets_raw[plane_index]) / normal_norm
                frame_valid = frame_valid and abs(normal_norm - 1.0) <= 1.0e-4
                frame_valid = frame_valid and abs(float(np.dot(axis, normal))) <= math.sin(math.radians(3.0))
            normals.append(normal)
            offsets.append(offset)
            cross = np.cross(normal, axis)
            cross_norm = float(np.linalg.norm(cross))
            if cross_norm <= 0.0:
                tails.append(np.zeros(3, dtype=np.float64))
                frame_valid = False
            else:
                tail = cross / cross_norm
                if float(np.dot(tail, centroids_raw[plane_index] - center)) < 0.0:
                    tail = -tail
                tails.append(tail)

        alpha = math.acos(float(np.clip(abs(np.dot(normals[0], normals[1])), 0.0, 1.0))) if frame_valid else 0.0
        target_turns = tuple(
            value
            for value in sorted({alpha, math.pi - alpha})
            if math.radians(MINIMUM_TARGET_TURN_DEGREES) <= value <= math.radians(MAXIMUM_TARGET_TURN_DEGREES)
        )
        span_valid = support[1] > support[0] and support[1] - support[0] >= 64.0 * s
        line_delta = line_end - line_start
        line_length = float(np.linalg.norm(line_delta))
        frame_valid = frame_valid and line_length > 0.0
        if line_length > 0.0:
            frame_valid = frame_valid and abs(float(np.dot(line_delta / line_length, axis))) >= math.cos(math.radians(1.0))
        parsed.append(
            Candidate(
                raw=raw,
                candidate_key=key,
                candidate_id=_candidate_id(key),
                v16_rank=rank,
                selected_legacy=selected,
                center=center,
                axis=axis,
                normals=(normals[0], normals[1]),
                offsets=(offsets[0], offsets[1]),
                centroids=(centroids_raw[0], centroids_raw[1]),
                tails=(tails[0], tails[1]),
                support_interval=support,
                line_start=line_start,
                line_end=line_end,
                target_turns=target_turns,
                frame_valid=frame_valid,
                span_valid=span_valid,
            )
        )
    rejected_count = sum(not candidate.selected_legacy for candidate in parsed)
    if rejected_count > MAX_ELIGIBLE_REJECTED_CANDIDATES:
        raise AMFCContractError(REASON_RESOURCE_CAP, "eligible_rejected_candidate_count")
    return parsed


def _jacobi_smallest_eigenvectors(covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.asarray(covariances, dtype=np.float64).copy()
    count = len(matrices)
    vectors = np.broadcast_to(np.eye(3, dtype=np.float64), (count, 3, 3)).copy()
    for _ in range(NORMAL_JACOBI_SWEEPS):
        for p, q, r in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            app = matrices[:, p, p].copy()
            aqq = matrices[:, q, q].copy()
            apq = matrices[:, p, q].copy()
            active = np.abs(apq) > 1.0e-30
            tau = np.zeros(count, dtype=np.float64)
            tau[active] = (aqq[active] - app[active]) / (2.0 * apq[active])
            sign = np.where(tau >= 0.0, 1.0, -1.0)
            tangent = np.zeros(count, dtype=np.float64)
            tangent[active] = sign[active] / (np.abs(tau[active]) + np.sqrt(1.0 + tau[active] * tau[active]))
            cosine = 1.0 / np.sqrt(1.0 + tangent * tangent)
            sine = tangent * cosine
            matrices[:, p, p] = app - tangent * apq
            matrices[:, q, q] = aqq + tangent * apq
            matrices[:, p, q] = 0.0
            matrices[:, q, p] = 0.0
            arp = matrices[:, r, p].copy()
            arq = matrices[:, r, q].copy()
            new_rp = cosine * arp - sine * arq
            new_rq = sine * arp + cosine * arq
            matrices[:, r, p] = new_rp
            matrices[:, p, r] = new_rp
            matrices[:, r, q] = new_rq
            matrices[:, q, r] = new_rq
            vp = vectors[:, :, p].copy()
            vq = vectors[:, :, q].copy()
            vectors[:, :, p] = cosine[:, None] * vp - sine[:, None] * vq
            vectors[:, :, q] = sine[:, None] * vp + cosine[:, None] * vq
    eigenvalues = np.stack((matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2]), axis=1)
    smallest = np.argmin(eigenvalues, axis=1)
    normals = vectors[np.arange(count), :, smallest]
    dominant = np.argmax(np.abs(normals), axis=1)
    signs = np.sign(normals[np.arange(count), dominant])
    signs[signs == 0.0] = 1.0
    normals *= signs[:, None]
    return normals, eigenvalues


def compute_point_normals(points: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    k = min(NORMAL_NEIGHBORS + 1, len(points))
    distances, indices = tree.query(points, k=k, distance_upper_bound=NORMAL_RADIUS_S * s, workers=1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    valid_neighbors = np.isfinite(neighbor_distances) & (neighbor_indices < len(points))
    counts = valid_neighbors.sum(axis=1)
    normals = np.zeros_like(points)
    surface_like = np.zeros(len(points), dtype=np.bool_)
    batch_size = 50_000
    for begin in range(0, len(points), batch_size):
        end = min(len(points), begin + batch_size)
        mask = valid_neighbors[begin:end]
        local_counts = counts[begin:end]
        safe = np.where(mask, neighbor_indices[begin:end], 0)
        gathered = points[safe]
        weighted = gathered * mask[:, :, None]
        denominator = np.maximum(local_counts, 1)[:, None]
        means = weighted.sum(axis=1) / denominator
        centered = (gathered - means[:, None, :]) * mask[:, :, None]
        covariance = np.einsum("bni,bnj->bij", centered, centered, optimize=False) / denominator[:, :, None]
        local_normals, eigenvalues = _jacobi_smallest_eigenvectors(covariance)
        trace = eigenvalues.sum(axis=1)
        minimum = eigenvalues.min(axis=1)
        valid = (local_counts >= NORMAL_MINIMUM_NEIGHBORS) & (trace > 0.0) & (minimum / np.maximum(trace, 1.0e-30) <= SURFACE_VARIATION_MAXIMUM)
        normals[begin:end] = local_normals
        surface_like[begin:end] = valid
    return normals, surface_like


@dataclass
class _ResidualEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    original_capacity: int
    meta: tuple[str, int, int] | None


def _add_residual_edge(
    graph: list[list[_ResidualEdge]],
    first: int,
    second: int,
    capacity: int,
    cost: int,
    meta: tuple[str, int, int] | None,
) -> None:
    forward = _ResidualEdge(second, len(graph[second]), capacity, cost, capacity, meta)
    reverse = _ResidualEdge(first, len(graph[first]), 0, -cost, 0, None)
    graph[first].append(forward)
    graph[second].append(reverse)


def _minimum_cost_disjoint_paths(
    node_count: int,
    graph_edges: Sequence[tuple[int, int, int]],
    tail0: set[int],
    tail1: set[int],
    node_keys: Sequence[tuple[int, ...]],
) -> list[list[int]]:
    source = 2 * node_count
    sink = source + 1
    network: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]
    network_keys: list[tuple[Any, ...]] = []
    for key in node_keys:
        network_keys.extend(((0,) + tuple(key) + (0,), (0,) + tuple(key) + (1,)))
    network_keys.extend(((-1,), (1,)))
    for node in range(node_count):
        _add_residual_edge(network, 2 * node, 2 * node + 1, 1, 0, ("split", node, node))
    for first, second, cost in sorted(graph_edges, key=lambda row: (node_keys[row[0]], node_keys[row[1]], row[2])):
        _add_residual_edge(network, 2 * first + 1, 2 * second, 1, int(cost), ("graph", first, second))
        _add_residual_edge(network, 2 * second + 1, 2 * first, 1, int(cost), ("graph", second, first))
    for node in sorted(tail0, key=lambda value: node_keys[value]):
        _add_residual_edge(network, source, 2 * node, 1, 0, ("source", node, node))
    for node in sorted(tail1, key=lambda value: node_keys[value]):
        _add_residual_edge(network, 2 * node + 1, sink, 1, 0, ("sink", node, node))

    potentials = [0] * len(network)
    for _ in range(MAX_PATHS):
        infinity = 10**30
        distances = [infinity] * len(network)
        predecessors: list[tuple[int, int] | None] = [None] * len(network)
        predecessor_keys: list[tuple[Any, ...] | None] = [None] * len(network)
        distances[source] = 0
        queue: list[tuple[int, tuple[Any, ...], int]] = [(0, network_keys[source], source)]
        while queue:
            distance, _key, first = heapq.heappop(queue)
            if distance != distances[first]:
                continue
            for edge_index, edge in sorted(enumerate(network[first]), key=lambda item: (network_keys[item[1].to], item[0])):
                if edge.capacity <= 0:
                    continue
                reduced = edge.cost + potentials[first] - potentials[edge.to]
                candidate_distance = distance + reduced
                candidate_predecessor_key = (network_keys[first], edge_index)
                if candidate_distance < distances[edge.to] or (
                    candidate_distance == distances[edge.to]
                    and (predecessor_keys[edge.to] is None or candidate_predecessor_key < predecessor_keys[edge.to])
                ):
                    distances[edge.to] = candidate_distance
                    predecessors[edge.to] = (first, edge_index)
                    predecessor_keys[edge.to] = candidate_predecessor_key
                    heapq.heappush(queue, (candidate_distance, network_keys[edge.to], edge.to))
        if predecessors[sink] is None:
            break
        for node, distance in enumerate(distances):
            if distance < infinity:
                potentials[node] += distance
        cursor = sink
        while cursor != source:
            previous, edge_index = predecessors[cursor]  # type: ignore[misc]
            edge = network[previous][edge_index]
            edge.capacity -= 1
            network[cursor][edge.reverse].capacity += 1
            cursor = previous

    positive: dict[int, list[tuple[int, int, _ResidualEdge]]] = {}
    for first, edges in enumerate(network):
        for edge_index, edge in enumerate(edges):
            if edge.original_capacity > 0 and edge.capacity < edge.original_capacity:
                positive.setdefault(first, []).append((edge.to, edge_index, edge))
    for rows in positive.values():
        rows.sort(key=lambda row: (network_keys[row[0]], row[1]))

    consumed: set[tuple[int, int]] = set()
    paths: list[list[int]] = []

    def find_path(first: int, visited: set[int]) -> list[tuple[int, int, _ResidualEdge]] | None:
        if first == sink:
            return []
        visited.add(first)
        for second, edge_index, edge in positive.get(first, []):
            token = (first, edge_index)
            if token in consumed or second in visited:
                continue
            remainder = find_path(second, visited)
            if remainder is not None:
                return [(first, edge_index, edge)] + remainder
        visited.remove(first)
        return None

    for _ in range(MAX_PATHS):
        network_path = find_path(source, set())
        if network_path is None:
            break
        source_nodes: list[int] = []
        for first, edge_index, edge in network_path:
            consumed.add((first, edge_index))
            if edge.meta is None:
                continue
            kind, start_node, end_node = edge.meta
            if kind == "source":
                source_nodes.append(start_node)
            elif kind == "graph":
                if not source_nodes:
                    source_nodes.append(start_node)
                source_nodes.append(end_node)
        if source_nodes:
            paths.append(source_nodes)
    return paths


def _longest_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _station_positions(candidate: Candidate, s: float) -> tuple[float, ...]:
    lower, upper = candidate.support_interval
    usable = upper - lower - 2.0 * STATION_END_MARGIN_S * s
    return tuple(lower + STATION_END_MARGIN_S * s + index * usable / 8.0 for index in range(STATION_COUNT))


def _reduce_station_nodes(
    global_indices: np.ndarray,
    points: np.ndarray,
    candidate: Candidate,
    s: float,
) -> np.ndarray:
    if len(global_indices) <= MAX_STATION_NODES:
        return global_indices
    relative = points[global_indices] - candidate.center
    local = np.column_stack(
        (
            relative @ candidate.axis,
            relative @ candidate.tails[0],
            relative @ candidate.normals[0],
        )
    )
    voxels = np.floor(local / (STATION_VOXEL_S * s)).astype(np.int64)
    keys = [(_node_key(local[index], s / COORDINATE_QUANTIZATION_DIVISOR), index) for index in range(len(local))]
    retained: dict[tuple[int, int, int], tuple[tuple[int, ...], int]] = {}
    for index, voxel in enumerate(voxels):
        voxel_key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        record = keys[index]
        if voxel_key not in retained or record[0] < retained[voxel_key][0]:
            retained[voxel_key] = record
    selected = np.asarray([global_indices[value[1]] for value in sorted(retained.values())], dtype=np.int64)
    if len(selected) > MAX_STATION_NODES:
        raise AMFCContractError(REASON_RESOURCE_CAP, "station_node_cap")
    return selected


def _tail_seed_set(
    points: np.ndarray,
    normals: np.ndarray,
    axial: np.ndarray,
    candidate: Candidate,
    parent: int,
    s: float,
) -> tuple[set[int], bool, np.ndarray]:
    normal = candidate.normals[parent]
    offset = candidate.offsets[parent]
    tail = candidate.tails[parent]
    projection = candidate.center + (offset - float(np.dot(normal, candidate.center))) * normal
    outward = (points - projection) @ tail
    distance = np.abs(points @ normal - offset)
    alignment = np.abs(normals @ normal)
    mask = (
        (outward >= TAIL_MIN_S * s)
        & (outward <= TAIL_MAX_S * s)
        & (distance <= TAIL_PLANE_DISTANCE_S * s)
        & (alignment >= math.cos(math.radians(TAIL_NORMAL_DEGREES)))
    )
    indices = np.flatnonzero(mask)
    valid = False
    if len(indices) >= TAIL_MINIMUM_NODES:
        axial_span = float(np.ptp(axial[indices]))
        outward_span = float(np.ptp(outward[indices]))
        valid = axial_span >= TAIL_MINIMUM_AXIAL_SPAN_S * s and outward_span >= TAIL_MINIMUM_OUTWARD_SPAN_S * s
    return set(int(value) for value in indices), valid, outward


def _station_graph(
    local_global_indices: np.ndarray,
    source: SourceGeometry,
    normals: np.ndarray,
    candidate: Candidate,
    s: float,
) -> tuple[list[tuple[int, int, int]], list[tuple[np.ndarray, np.ndarray]]]:
    points = source.points[local_global_indices]
    local_count = len(points)
    if local_count == 0:
        return [], []
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    k = min(RECIPROCAL_NEIGHBORS + 1, local_count)
    _distances, neighbors = tree.query(points, k=k, workers=1)
    if k == 1:
        neighbors = neighbors[:, None]
    neighbor_sets: list[set[int]] = []
    for index in range(local_count):
        neighbor_sets.append(set(int(value) for value in np.atleast_1d(neighbors[index])[1:] if int(value) < local_count))
    pairs: set[tuple[int, int]] = set()
    for first, values in enumerate(neighbor_sets):
        for second in values:
            if first in neighbor_sets[second] and first != second:
                pairs.add((min(first, second), max(first, second)))
    global_to_local = {int(value): index for index, value in enumerate(local_global_indices)}
    for first_global, second_global in source.mesh_edges:
        if first_global in global_to_local and second_global in global_to_local:
            first = global_to_local[first_global]
            second = global_to_local[second_global]
            if first != second:
                pairs.add((min(first, second), max(first, second)))

    retained: list[tuple[int, int, int]] = []
    visual_edges: list[tuple[np.ndarray, np.ndarray]] = []
    local_normals = normals[local_global_indices]
    for first, second in sorted(pairs):
        delta = points[second] - points[first]
        length = float(np.linalg.norm(delta))
        if length < EDGE_MINIMUM_S * s or length > EDGE_MAXIMUM_S * s:
            continue
        if abs(float(np.dot(delta, candidate.axis))) > EDGE_MAXIMUM_AXIAL_S * s:
            continue
        tangent_ratio = max(
            abs(float(np.dot(delta, local_normals[first]))),
            abs(float(np.dot(delta, local_normals[second]))),
        ) / length
        if tangent_ratio > EDGE_TANGENT_RATIO_MAXIMUM:
            continue
        beta = math.acos(float(np.clip(abs(np.dot(local_normals[first], local_normals[second])), 0.0, 1.0)))
        if beta > math.radians(EDGE_NORMAL_DEGREES_MAXIMUM):
            continue
        cost = 1 + _round_half_away(4096.0 * length / s + 1024.0 * tangent_ratio + 512.0 * beta / math.pi)
        retained.append((first, second, cost))
        visual_edges.append((points[first].copy(), points[second].copy()))
        if len(retained) > MAX_STATION_EDGES:
            raise AMFCContractError(REASON_RESOURCE_CAP, "station_edge_cap")
    return retained, visual_edges


def _validate_path(
    path: Sequence[int],
    points: np.ndarray,
    normals: np.ndarray,
    axial: np.ndarray,
    candidate: Candidate,
    tail0: set[int],
    tail1: set[int],
    s: float,
) -> dict[str, Any]:
    edge_count = len(path) - 1
    if edge_count < PATH_MINIMUM_EDGES or edge_count > PATH_MAXIMUM_EDGES:
        return {"valid": False, "reason": REASON_PATH_BOTTLENECK}
    coordinates = points[np.asarray(path, dtype=np.int64)]
    segments = np.diff(coordinates, axis=0)
    length = float(np.linalg.norm(segments, axis=1).sum())
    chord = float(np.linalg.norm(coordinates[-1] - coordinates[0]))
    if length > PATH_MAXIMUM_LENGTH_S * s or length / max(chord, s) > PATH_MAXIMUM_CHORD_RATIO:
        return {"valid": False, "reason": REASON_PATH_BOTTLENECK}
    path_normals = normals[np.asarray(path, dtype=np.int64)].copy()
    if abs(float(np.dot(path_normals[0], candidate.normals[0]))) < math.cos(math.radians(PATH_ENDPOINT_NORMAL_DEGREES)):
        return {"valid": False, "reason": REASON_TURN_MISMATCH}
    if abs(float(np.dot(path_normals[-1], candidate.normals[1]))) < math.cos(math.radians(PATH_ENDPOINT_NORMAL_DEGREES)):
        return {"valid": False, "reason": REASON_TURN_MISMATCH}
    internal = sum(int(value not in tail0 and value not in tail1) for value in path[1:-1])
    if internal < PATH_MINIMUM_INTERNAL_NODES:
        return {"valid": False, "reason": REASON_PATH_BOTTLENECK}
    path_axial = axial[np.asarray(path, dtype=np.int64)]
    if float(np.ptp(path_axial)) > PATH_MAXIMUM_AXIAL_EXCURSION_S * s:
        return {"valid": False, "reason": REASON_AXIAL_COVERAGE}
    if float(np.dot(path_normals[0], candidate.normals[0])) < 0.0:
        path_normals[0] *= -1.0
    for index in range(1, len(path_normals)):
        if float(np.dot(path_normals[index], path_normals[index - 1])) < 0.0:
            path_normals[index] *= -1.0
    turns = []
    for first, second in zip(path_normals[:-1], path_normals[1:]):
        turns.append(math.atan2(float(np.dot(candidate.axis, np.cross(first, second))), float(np.dot(first, second))))
    total = abs(float(sum(turns)))
    variation = float(sum(abs(value) for value in turns))
    monotonicity = total / max(variation, 1.0e-9)
    if total < math.radians(MINIMUM_TARGET_TURN_DEGREES):
        return {"valid": False, "reason": REASON_TURN_TOO_SMALL, "turn": total, "variation": variation, "monotonicity": monotonicity}
    error = min(abs(total - target) for target in candidate.target_turns)
    if error > math.radians(PATH_TURN_ERROR_DEGREES):
        return {"valid": False, "reason": REASON_TURN_MISMATCH, "turn": total, "variation": variation, "monotonicity": monotonicity, "error": error}
    if monotonicity < PATH_MONOTONICITY_MINIMUM:
        return {"valid": False, "reason": REASON_TURN_NONMONOTONE, "turn": total, "variation": variation, "monotonicity": monotonicity, "error": error}
    return {
        "valid": True,
        "reason": None,
        "turn": total,
        "variation": variation,
        "monotonicity": monotonicity,
        "error": error,
        "midpoint_axial": 0.5 * float(path_axial[0] + path_axial[-1]),
        "coordinates": coordinates,
    }


def _analyze_station(
    station: float,
    source: SourceGeometry,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    candidate: Candidate,
    s: float,
) -> dict[str, Any]:
    relative = source.points - candidate.center
    axial_all = relative @ candidate.axis
    radial_vectors = relative - axial_all[:, None] * candidate.axis[None, :]
    radial = np.linalg.norm(radial_vectors, axis=1)
    selection = normal_valid & (np.abs(axial_all - station) <= STATION_HALF_WIDTH_S * s) & (radial <= STATION_RADIUS_S * s)
    global_indices = np.flatnonzero(selection)
    global_indices = _reduce_station_nodes(global_indices, source.points, candidate, s)
    points = source.points[global_indices]
    local_normals = normals[global_indices]
    axial = (points - candidate.center) @ candidate.axis
    tail0, tail0_valid, _out0 = _tail_seed_set(points, local_normals, axial, candidate, 0, s)
    tail1, tail1_valid, _out1 = _tail_seed_set(points, local_normals, axial, candidate, 1, s)
    visual: dict[str, Any] = {
        "position": station,
        "tail0": points[sorted(tail0)].copy() if tail0 else np.empty((0, 3)),
        "tail1": points[sorted(tail1)].copy() if tail1 else np.empty((0, 3)),
        "graph_edges": [],
        "accepted_paths": [],
    }
    if not tail0_valid or not tail1_valid:
        return {"valid": False, "reason": REASON_TAIL_MISSING, "tail_valid": False, "valid_paths": 0, "path_errors": [], "visual": visual}
    graph_edges, visual_edges = _station_graph(global_indices, source, normals, candidate, s)
    visual["graph_edges"] = visual_edges
    if not graph_edges:
        return {"valid": False, "reason": REASON_GRAPH_DISCONNECTED, "tail_valid": True, "valid_paths": 0, "path_errors": [], "visual": visual}
    local_keys = tuple(_node_key(point - candidate.center, s / COORDINATE_QUANTIZATION_DIVISOR) for point in points)
    paths = _minimum_cost_disjoint_paths(len(points), graph_edges, tail0, tail1, local_keys)
    if not paths:
        return {"valid": False, "reason": REASON_GRAPH_DISCONNECTED, "tail_valid": True, "valid_paths": 0, "path_errors": [], "visual": visual}
    outcomes = [_validate_path(path, points, local_normals, axial, candidate, tail0, tail1, s) for path in paths]
    accepted = [outcome for outcome in outcomes if outcome["valid"]]
    visual["accepted_paths"] = [outcome["coordinates"] for outcome in accepted]
    if len(accepted) < MINIMUM_VALID_PATHS_PER_STATION:
        reasons = [str(outcome["reason"]) for outcome in outcomes if not outcome["valid"]]
        if REASON_TURN_NONMONOTONE in reasons:
            reason = REASON_TURN_NONMONOTONE
        elif REASON_TURN_TOO_SMALL in reasons:
            reason = REASON_TURN_TOO_SMALL
        elif REASON_TURN_MISMATCH in reasons:
            reason = REASON_TURN_MISMATCH
        else:
            reason = REASON_PATH_BOTTLENECK
        return {"valid": False, "reason": reason, "tail_valid": True, "valid_paths": len(accepted), "path_errors": [float(value["error"]) for value in accepted], "visual": visual}
    midpoint_span = float(np.ptp([float(value["midpoint_axial"]) for value in accepted]))
    if midpoint_span < MINIMUM_PATH_MIDPOINT_SPAN_S * s:
        return {"valid": False, "reason": REASON_AXIAL_COVERAGE, "tail_valid": True, "valid_paths": len(accepted), "path_errors": [float(value["error"]) for value in accepted], "visual": visual}
    return {"valid": True, "reason": None, "tail_valid": True, "valid_paths": len(accepted), "path_errors": [float(value["error"]) for value in accepted], "visual": visual}


def _candidate_decision(
    candidate: Candidate,
    source: SourceGeometry,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate.selected_legacy:
        decision = {
            "candidate_id": candidate.candidate_id,
            "candidate_key": candidate.candidate_key,
            "v16_rank": candidate.v16_rank,
            "selected_legacy": True,
            "action": ACTION_LEGACY,
            "reason": None,
            "valid_station_count": 0,
            "valid_path_count": 0,
            "median_turn_error_q": 0,
        }
        return decision, {"candidate": candidate, "stations": []}
    if not candidate.frame_valid:
        reason = REASON_FRAME_INVALID
        station_rows: list[dict[str, Any]] = []
    elif not candidate.span_valid:
        reason = REASON_SPAN_SHORT
        station_rows = []
    elif not candidate.target_turns:
        reason = REASON_TURN_TOO_SMALL
        station_rows = []
    else:
        station_rows = [_analyze_station(position, source, normals, normal_valid, candidate, s) for position in _station_positions(candidate, s)]
        tail_valid_count = sum(bool(row["tail_valid"]) for row in station_rows)
        valid_flags = [bool(row["valid"]) for row in station_rows]
        valid_station_count = sum(valid_flags)
        valid_path_count = sum(int(row["valid_paths"]) for row in station_rows)
        errors = [float(error) for row in station_rows for error in row["path_errors"]]
        median_error = float(np.median(errors)) if errors else math.inf
        passed = (
            tail_valid_count >= MINIMUM_TAIL_VALID_STATIONS
            and valid_station_count >= MINIMUM_VALID_STATIONS
            and _longest_true_run(valid_flags) >= MINIMUM_CONSECUTIVE_VALID_STATIONS
            and valid_path_count >= MINIMUM_VALID_PATHS_TOTAL
            and median_error <= math.radians(MEDIAN_TURN_ERROR_DEGREES)
        )
        if passed:
            reason = None
        elif tail_valid_count < MINIMUM_TAIL_VALID_STATIONS:
            reason = REASON_TAIL_MISSING
        elif valid_station_count > 0 or valid_path_count > 0:
            nonvalid_reasons = [str(row["reason"]) for row in station_rows if not row["valid"]]
            if REASON_TURN_NONMONOTONE in nonvalid_reasons and valid_station_count == 0:
                reason = REASON_TURN_NONMONOTONE
            elif REASON_TURN_TOO_SMALL in nonvalid_reasons and valid_station_count == 0:
                reason = REASON_TURN_TOO_SMALL
            elif REASON_TURN_MISMATCH in nonvalid_reasons and valid_station_count == 0:
                reason = REASON_TURN_MISMATCH
            else:
                reason = REASON_AXIAL_COVERAGE
        else:
            station_reasons = [str(row["reason"]) for row in station_rows]
            if REASON_TURN_NONMONOTONE in station_reasons:
                reason = REASON_TURN_NONMONOTONE
            elif REASON_TURN_TOO_SMALL in station_reasons:
                reason = REASON_TURN_TOO_SMALL
            elif REASON_TURN_MISMATCH in station_reasons:
                reason = REASON_TURN_MISMATCH
            elif REASON_PATH_BOTTLENECK in station_reasons:
                reason = REASON_PATH_BOTTLENECK
            else:
                reason = REASON_GRAPH_DISCONNECTED

    valid_station_count = sum(bool(row.get("valid")) for row in station_rows)
    valid_path_count = sum(int(row.get("valid_paths", 0)) for row in station_rows)
    errors = [float(error) for row in station_rows for error in row.get("path_errors", [])]
    median_error = float(np.median(errors)) if errors else math.pi
    decision = {
        "candidate_id": candidate.candidate_id,
        "candidate_key": candidate.candidate_key,
        "v16_rank": candidate.v16_rank,
        "selected_legacy": False,
        "action": ACTION_RESCUE if reason is None else ACTION_NONE,
        "reason": reason,
        "valid_station_count": valid_station_count,
        "valid_path_count": valid_path_count,
        "median_turn_error_q": _round_half_away(math.degrees(median_error) * TURN_QUANTIZATION_MICRODEGREES),
    }
    if len(_canonical_json_bytes(decision)) > MAX_CANDIDATE_RECORD_BYTES:
        raise AMFCContractError(REASON_RESOURCE_CAP, "candidate_record_bytes")
    return decision, {"candidate": candidate, "stations": [row.get("visual") for row in station_rows]}


def deterministic_total_sort(
    candidates: Sequence[Candidate], decisions: Sequence[dict[str, Any]]
) -> list[tuple[Candidate, dict[str, Any]]]:
    by_key = {candidate.candidate_key: candidate for candidate in candidates}
    legacy = [decision for decision in decisions if decision["action"] == ACTION_LEGACY]
    legacy.sort(key=lambda value: (int(value["v16_rank"]), str(value["candidate_key"])))
    rescues = [decision for decision in decisions if decision["action"] == ACTION_RESCUE]
    rescues.sort(
        key=lambda value: (
            -int(value["valid_station_count"]),
            -int(value["valid_path_count"]),
            int(value["median_turn_error_q"]),
            int(value["v16_rank"]),
            str(value["candidate_key"]),
        )
    )
    return [(by_key[str(value["candidate_key"])], value) for value in legacy + rescues]


def _line_metrics(first: Candidate, second: Candidate) -> tuple[float, float, float]:
    first_delta = first.line_end - first.line_start
    second_delta = second.line_end - second.line_start
    first_length = float(np.linalg.norm(first_delta))
    second_length = float(np.linalg.norm(second_delta))
    first_axis = first_delta / first_length
    second_axis = second_delta / second_length
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))))
    first_mid = 0.5 * (first.line_start + first.line_end)
    second_mid = 0.5 * (second.line_start + second.line_end)
    delta = second_mid - first_mid
    distance = max(
        float(np.linalg.norm(delta - first_axis * np.dot(delta, first_axis))),
        float(np.linalg.norm(delta - second_axis * np.dot(delta, second_axis))),
    )
    center_distance = abs(float(np.dot(delta, first_axis)))
    overlap = max(0.0, (first_length + second_length) / 2.0 - center_distance) / min(first_length, second_length)
    return angle, distance, overlap


def ordinary_nms(
    ordered: Sequence[tuple[Candidate, dict[str, Any]]], s: float
) -> list[tuple[Candidate, dict[str, Any]]]:
    accepted: list[tuple[Candidate, dict[str, Any]]] = []
    for candidate, decision in ordered:
        duplicate = False
        for prior, _prior_decision in accepted:
            angle, distance_mm, overlap = _line_metrics(candidate, prior)
            if angle <= NMS_ANGLE_DEGREES and distance_mm <= NMS_DISTANCE_MM and overlap >= NMS_MINIMUM_OVERLAP:
                duplicate = True
                break
        if not duplicate:
            accepted.append((candidate, decision))
    return accepted


def build_result(
    candidates: Sequence[Candidate],
    decisions: Sequence[dict[str, Any]],
    accepted: Sequence[tuple[Candidate, dict[str, Any]]],
    visual_rows: Sequence[dict[str, Any]],
    s: float,
    worker_count: int,
) -> dict[str, Any]:
    legacy_inputs = [candidate.raw for candidate in candidates if candidate.selected_legacy]
    final_lines = [
        {
            "candidate_id": candidate.candidate_id,
            "candidate_key": candidate.candidate_key,
            "line_start": candidate.line_start.tolist(),
            "line_end": candidate.line_end.tolist(),
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v30_amfc_rescue",
            "v30_action": decision["action"],
        }
        for candidate, decision in accepted
    ]
    cache = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "candidate_decisions": sorted(decisions, key=lambda value: str(value["candidate_id"])),
        "final_lines": final_lines,
    }
    cache_bytes = _canonical_json_bytes(cache)
    if len(cache_bytes) > MAX_SUMMARY_BYTES:
        raise AMFCContractError(REASON_RESOURCE_CAP, "summary_cache_bytes")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sherman_qc_f1_v30_amfc_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "research_only": True,
        "source_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "labels_used": False,
        "cad_used": False,
        "v29_used": False,
        "s_mm": s,
        "decisions": list(decisions),
        "legacy_input_records": legacy_inputs,
        "final_lines": final_lines,
        "rescue_count": sum(value["action"] == ACTION_RESCUE for value in decisions),
        "cache": cache,
        "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "cache_bytes": len(cache_bytes),
        "ordinary_nms_call_count": 1,
        "candidate_added_after_nms": False,
        "legacy_wins_duplicate_collision": True,
        "_ephemeral": {"candidates": list(visual_rows)},
    }


def select_unified_v30(
    raw_source: dict[str, Any],
    raw_candidates: Sequence[dict[str, Any]],
    s: float,
    *,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Run frozen V16 records through AMFC and one ordinary NMS call."""

    scale = _finite_float(s, "s")
    if scale <= 0.0:
        raise AMFCContractError("INVALID_SCALE")
    workers = int(worker_count)
    if workers < 1 or workers > MAX_WORKERS:
        raise AMFCContractError("INVALID_WORKER_COUNT")
    source = parse_source(raw_source, scale)
    candidates = parse_candidates(raw_candidates, scale)
    normals, normal_valid = compute_point_normals(source.points, scale)

    def analyze(candidate: Candidate) -> tuple[dict[str, Any], dict[str, Any]]:
        return _candidate_decision(candidate, source, normals, normal_valid, scale)

    if workers == 1:
        analyzed = [analyze(candidate) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v30-amfc") as executor:
            analyzed = list(executor.map(analyze, candidates))
    decisions = [value[0] for value in analyzed]
    visuals = [value[1] for value in analyzed]
    ordered = deterministic_total_sort(candidates, decisions)
    accepted = ordinary_nms(ordered, scale)
    return build_result(candidates, decisions, accepted, visuals, scale, workers)
