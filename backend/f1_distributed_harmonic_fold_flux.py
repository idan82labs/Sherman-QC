"""V31 distributed harmonic fold flux certificate (DHFF_V1).

Research-only, source-only rescue sidecar for the frozen V16 selector.  The
module consumes only scan-surface samples and already-rejected V16 candidates.
It neither creates candidates nor changes V16 geometry, scores, or incumbents.

The causal primitive is an electrical-current field on an observed-surface
graph.  Eleven non-overlapping axial stations are solved independently; a
candidate is rescued only when the current-carrying normals form one broad,
monotone turn at the frozen candidate angle.

There is deliberately no file I/O, label/CAD access, path search, path
enumeration, or architecture-specific NMS in this module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
import struct
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


PROFILE = "V31_DISTRIBUTED_HARMONIC_FOLD_FLUX_V1"
PRIMITIVE = "DIRICHLET_ELECTRICAL_CURRENT_NORMAL_PROFILE_V1"
SCHEMA_VERSION = 1

STATION_COUNT = 11
STATION_OFFSETS = tuple(-0.40 + 0.08 * j for j in range(STATION_COUNT))
STATION_HALF_WIDTH_S = 0.75
STATION_RADIUS_S = 24.0
STATION_VOXEL_S = 0.5
MAX_STATION_NODES = 2_048
MAX_STATION_EDGES = 24_576
MUTUAL_NEIGHBORS = 12

TAIL_MIN_S = 5.0
TAIL_MAX_S = 13.0
TAIL_BINS = ((5.0, 7.0), (7.0, 9.0), (9.0, 11.0), (11.0, 13.0))
TAIL_PLANE_RESIDUAL_S = 0.75
TAIL_NORMAL_DEGREES = 10.0
TAIL_MINIMUM_PER_BIN = 2
TAIL_MINIMUM_TOTAL = 12
TAIL_COMPONENT_FRACTION = 0.75

EDGE_MAXIMUM_S = 2.25
EDGE_NORMAL_DEGREES = 35.0
EDGE_NORMAL_TRAVEL_MAXIMUM = 0.60
ORIENTATION_CONFLICT_DEGREES = 10.0

PCG_MAXIMUM_ITERATIONS = 128
PCG_RELATIVE_INFINITY_RESIDUAL = 1.0e-10
MINIMUM_EFFECTIVE_PARTICIPATION = 24.0
MAXIMUM_EDGE_CURRENT_FRACTION = 0.12
POTENTIAL_BAND_HALF_WIDTH = 1.0 / 16.0
POTENTIAL_BAND_MINIMUM_NODES = 3
POTENTIAL_BAND_MINIMUM_MASS = 0.03
ANGULAR_BIN_MINIMUM_NODES = 2
ANGULAR_BIN_MINIMUM_MASS = 0.015
OFF_ARC_MARGIN_DEGREES = 10.0
MAXIMUM_OFF_ARC_MASS = 0.08
MAXIMUM_BACKTRACK_DEGREES = 5.0
MAXIMUM_ANGULAR_VARIATION_DEGREES = 12.0

MINIMUM_LENGTH_S = 24.0
MINIMUM_TURN_DEGREES = 40.0
MAXIMUM_TURN_DEGREES = 140.0
MAXIMUM_AXIS_NORMAL_DEGREES = 12.0
MINIMUM_PASSING_STATIONS = 8
MINIMUM_CONSECUTIVE_PASSING = 6
MAX_ELIGIBLE_CANDIDATES = 128
MAX_SOURCE_SAMPLES = 200_000
MAX_WORKERS = 32

CACHE_HEADER_BYTES = 256
CACHE_CANDIDATE_BYTES = 192
MAX_CACHE_BYTES = 24_832
Q18 = 262_144
Q24 = 16_777_216

NMS_ANGLE_DEGREES = 7.0
NMS_DISTANCE_MM = 6.0
NMS_MINIMUM_OVERLAP = 0.60

ACTION_LEGACY = "LEGACY"
ACTION_RESCUE = "DHFF_PASS"
ACTION_NONE = "NO_RESCUE"

REASONS = (
    "STATION_PASS",
    "TURN_ELIGIBILITY",
    "FRAME_INVALID",
    "CANDIDATE_WORK_CAP",
    "STATION_NODE_CAP",
    "STATION_EDGE_CAP",
    "TAIL_A_BREADTH",
    "TAIL_B_BREADTH",
    "ORIENTATION_CONFLICT",
    "NO_MATERIAL_CONNECTION",
    "SOLVE_FAILURE",
    "LOW_CURRENT_PARTICIPATION",
    "POTENTIAL_COVERAGE",
    "ANGULAR_COVERAGE",
    "OFF_ARC_CURRENT",
    "NORMAL_BACKTRACK",
    "AXIAL_STATION_COVERAGE",
    "ORDINARY_NMS_SUPPRESSED",
    "RESOURCE_CAP",
)
REASON_CODE = {name: index for index, name in enumerate(REASONS)}

PROHIBITED_FIELD_TOKENS = frozenset(
    {
        "filename", "directory", "part_number", "part_id", "family", "customer",
        "identity", "content_hash", "arrival_order", "expected_count", "good_scan",
        "bad_scan", "deviation_status", "label", "gold", "manual_selection",
        "adjudication", "benchmark", "cad", "step", "drawing", "nominal",
        "tolerance", "acquisition_pose", "intrinsics", "range_frame", "visibility_ray",
        "color", "intensity", "texture", "reflectivity", "production", "runtime",
    }
)


class DHFFContractError(RuntimeError):
    """Fail-closed V31 schema, numerical, solver, or resource violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    return _canonical_json_bytes({key: value for key, value in result.items() if key != "_ephemeral"}) + b"\n"


def _round_half_away(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _q24(value: float) -> float:
    return _round_half_away(float(value) * Q24) / Q24


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DHFFContractError("NONFINITE_VALUE", label)
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise DHFFContractError("SIGNED_ZERO_VALUE", label)
    return result


def _array(value: Any, shape: tuple[int, ...], label: str, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise DHFFContractError("INVALID_ARRAY_SHAPE", label)
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise DHFFContractError("NONFINITE_ARRAY", label)
    return result


def _unit(value: Any, label: str) -> np.ndarray:
    result = _array(value, (3,), label)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise DHFFContractError("ZERO_VECTOR", label)
    return result / norm


def _contains_prohibited_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in PROHIBITED_FIELD_TOKENS):
                return str(key)
            found = _contains_prohibited_key(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _contains_prohibited_key(nested)
            if found:
                return found
    return None


def _angle_degrees(first: np.ndarray, second: np.ndarray, *, unoriented: bool = False) -> float:
    dot = float(np.dot(first, second))
    if unoriented:
        dot = abs(dot)
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def _stable_digest(position: np.ndarray, normal: np.ndarray, parent: int) -> bytes:
    quantized = np.concatenate((np.rint(position * Q18), np.rint(normal * Q18))).astype("<i8", copy=False)
    return hashlib.sha256(quantized.tobytes() + struct.pack("<I", int(parent))).digest()


@dataclass(frozen=True)
class SourceGeometry:
    positions: np.ndarray
    normals: np.ndarray
    parents: np.ndarray
    stable_keys: np.ndarray
    mesh_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ParentPlane:
    parent_id: int
    normal: np.ndarray
    offset: float
    centroid: np.ndarray


@dataclass(frozen=True)
class Candidate:
    raw: dict[str, Any]
    candidate_id: int
    frozen_order: int
    selected_legacy: bool
    rejected_before_nms: bool
    render_admissible: bool
    center: np.ndarray
    axis: np.ndarray
    length: float
    scale: float
    alpha_degrees: float
    parents: tuple[ParentPlane, ParentPlane]
    line_start: np.ndarray
    line_end: np.ndarray
    eligible: bool


def parse_source(raw: dict[str, Any]) -> SourceGeometry:
    prohibited = _contains_prohibited_key(raw)
    if prohibited:
        raise DHFFContractError("PROHIBITED_SOURCE_FIELD", prohibited)
    positions = np.asarray(raw.get("positions", []), dtype=np.float64)
    normals = np.asarray(raw.get("source_normals", []), dtype=np.float64)
    parents = np.asarray(raw.get("parent_fragment_ids", []), dtype=np.uint32)
    stable_keys = np.asarray(raw.get("stable_source_keys", []), dtype=np.uint64)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise DHFFContractError("INVALID_SOURCE_POSITIONS")
    count = len(positions)
    if count > MAX_SOURCE_SAMPLES:
        raise DHFFContractError("RESOURCE_CAP", "source_samples")
    if normals.shape != (count, 3) or parents.shape != (count,) or stable_keys.shape != (count,):
        raise DHFFContractError("INVALID_SOURCE_SCHEMA")
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise DHFFContractError("NONFINITE_SOURCE")
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    positions = positions[valid]
    normals = normals[valid] / lengths[valid, None]
    parents = parents[valid]
    stable_keys = stable_keys[valid]
    original_to_valid = np.full(count, -1, dtype=np.int64)
    original_to_valid[np.flatnonzero(valid)] = np.arange(int(valid.sum()))
    edge_set: set[tuple[int, int]] = set()
    for value in raw.get("mesh_adjacencies", []):
        pair = np.asarray(value, dtype=np.int64)
        if pair.shape != (2,) or np.any(pair < 0) or np.any(pair >= count):
            raise DHFFContractError("INVALID_MESH_ADJACENCY")
        a, b = int(original_to_valid[pair[0]]), int(original_to_valid[pair[1]])
        if a >= 0 and b >= 0 and a != b:
            edge_set.add((min(a, b), max(a, b)))
    return SourceGeometry(positions, normals, parents, stable_keys, tuple(sorted(edge_set)))


def _parse_plane(raw: dict[str, Any], label: str) -> ParentPlane:
    return ParentPlane(
        int(raw["parent_id"]),
        _unit(raw["unit_normal_xyz"], f"{label}.normal"),
        _finite(raw["offset"], f"{label}.offset"),
        _array(raw["centroid_xyz"], (3,), f"{label}.centroid"),
    )


def parse_candidates(raw_candidates: Sequence[dict[str, Any]]) -> list[Candidate]:
    if len(raw_candidates) > MAX_ELIGIBLE_CANDIDATES:
        raise DHFFContractError("CANDIDATE_WORK_CAP")
    candidates: list[Candidate] = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_candidates):
        prohibited = _contains_prohibited_key(raw)
        if prohibited:
            raise DHFFContractError("PROHIBITED_CANDIDATE_FIELD", prohibited)
        candidate_id = int(raw["candidate_id"])
        if candidate_id < 0 or candidate_id >= 1 << 64 or candidate_id in seen:
            raise DHFFContractError("INVALID_CANDIDATE_ID", str(candidate_id))
        seen.add(candidate_id)
        order = int(raw["frozen_v16_order"])
        center = _array(raw["center_xyz"], (3,), f"candidate[{index}].center")
        axis = _unit(raw["unit_axis_xyz"], f"candidate[{index}].axis")
        length = _finite(raw["segment_length_L"], f"candidate[{index}].length")
        scale = _finite(raw["local_scale_s"], f"candidate[{index}].scale")
        alpha = _finite(raw["signed_turn_alpha"], f"candidate[{index}].alpha")
        plane_a = _parse_plane(raw["parent_A"], f"candidate[{index}].parent_A")
        plane_b = _parse_plane(raw["parent_B"], f"candidate[{index}].parent_B")
        line_start = _array(raw["render_line_start"], (3,), f"candidate[{index}].line_start")
        line_end = _array(raw["render_line_end"], (3,), f"candidate[{index}].line_end")
        selected = bool(raw.get("selected_legacy", False))
        rejected = bool(raw.get("rejected_before_final_nms", not selected))
        render_ok = bool(raw.get("render_admissible", True))
        axis_limit = math.sin(math.radians(MAXIMUM_AXIS_NORMAL_DEGREES))
        eligible = bool(
            not selected
            and rejected
            and render_ok
            and scale > 0.0
            and length >= MINIMUM_LENGTH_S * scale
            and MINIMUM_TURN_DEGREES <= abs(alpha) <= MAXIMUM_TURN_DEGREES
            and abs(float(np.dot(axis, plane_a.normal))) <= axis_limit
            and abs(float(np.dot(axis, plane_b.normal))) <= axis_limit
            and plane_a.parent_id != plane_b.parent_id
        )
        candidates.append(Candidate(raw, candidate_id, order, selected, rejected, render_ok, center, axis, length, scale, alpha, (plane_a, plane_b), line_start, line_end, eligible))
    candidates.sort(key=lambda value: (value.frozen_order, value.candidate_id))
    if sum(candidate.eligible for candidate in candidates) > MAX_ELIGIBLE_CANDIDATES:
        raise DHFFContractError("CANDIDATE_WORK_CAP")
    return candidates


def _station_center(candidate: Candidate, station: int) -> np.ndarray:
    return candidate.center + STATION_OFFSETS[station] * candidate.length * candidate.axis


def _project_to_plane(point: np.ndarray, plane: ParentPlane) -> np.ndarray:
    return point - (float(np.dot(plane.normal, point)) - plane.offset) * plane.normal


def _tail_direction(candidate: Candidate, plane: ParentPlane, station_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = _project_to_plane(station_center, plane)
    direction = np.cross(candidate.axis, plane.normal)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise DHFFContractError("FRAME_INVALID")
    direction /= norm
    if float(np.dot(plane.centroid - projected, direction)) < 0.0:
        direction = -direction
    return projected, direction


def _local_subset(candidate: Candidate, source: SourceGeometry, station: int) -> tuple[np.ndarray, np.ndarray]:
    center = _station_center(candidate, station)
    rel_world = (source.positions - center) / candidate.scale
    bisector = candidate.parents[0].normal + candidate.parents[1].normal
    bisector = bisector - float(np.dot(bisector, candidate.axis)) * candidate.axis
    bisector_norm = float(np.linalg.norm(bisector))
    if bisector_norm <= 1.0e-12:
        raise DHFFContractError("FRAME_INVALID")
    bisector /= bisector_norm
    third = np.cross(candidate.axis, bisector)
    third /= np.linalg.norm(third)
    rel = np.column_stack((rel_world @ candidate.axis, rel_world @ bisector, rel_world @ third))
    q = np.asarray([[_round_half_away(float(v) * Q18) for v in row] for row in rel], dtype=np.int64)
    axial = np.abs(q[:, 0] / Q18)
    radial = np.linalg.norm(q[:, 1:] / Q18, axis=1)
    selected = np.flatnonzero((axial <= STATION_HALF_WIDTH_S) & (radial <= STATION_RADIUS_S))
    winners: dict[tuple[int, int, int], tuple[float, bytes, int]] = {}
    for global_index in selected:
        voxel = tuple(math.floor(float(value) / STATION_VOXEL_S) for value in rel[global_index])
        voxel_center = (np.asarray(voxel, dtype=np.float64) + 0.5) * STATION_VOXEL_S
        distance2 = float(np.dot(rel[global_index] - voxel_center, rel[global_index] - voxel_center))
        digest = _stable_digest(rel[global_index], source.normals[global_index], int(source.parents[global_index]))
        key = (_q24(distance2), digest, int(source.stable_keys[global_index]))
        previous = winners.get(voxel)
        if previous is None or key < previous:
            winners[voxel] = (key[0], key[1], int(global_index))
    indices = np.asarray(sorted((value[2] for value in winners.values()), key=lambda i: (int(source.stable_keys[i]), _stable_digest(rel[i], source.normals[i], int(source.parents[i])))), dtype=np.int64)
    if len(indices) > MAX_STATION_NODES:
        raise DHFFContractError("STATION_NODE_CAP")
    return center, indices


def _tail_seeds(candidate: Candidate, source: SourceGeometry, indices: np.ndarray, station_center: np.ndarray, which: int) -> tuple[np.ndarray, dict[str, Any]]:
    plane = candidate.parents[which]
    projected, direction = _tail_direction(candidate, plane, station_center)
    points = source.positions[indices]
    x = ((points - projected) @ direction) / candidate.scale
    residual = np.abs(points @ plane.normal - plane.offset) / candidate.scale
    normal_angle = np.asarray([_angle_degrees(source.normals[index], plane.normal, unoriented=True) for index in indices])
    mask = (source.parents[indices] == plane.parent_id) & (x >= TAIL_MIN_S) & (x <= TAIL_MAX_S) & (residual <= TAIL_PLANE_RESIDUAL_S) & (normal_angle <= TAIL_NORMAL_DEGREES)
    local = np.flatnonzero(mask)
    counts = [int(np.count_nonzero(mask & (x >= lo) & (x < hi if hi < TAIL_MAX_S else x <= hi))) for lo, hi in TAIL_BINS]
    valid = len(local) >= TAIL_MINIMUM_TOTAL and min(counts, default=0) >= TAIL_MINIMUM_PER_BIN
    return local, {"valid": bool(valid), "count": int(len(local)), "bin_counts": counts, "x": x.tolist()}


def _edge_allowed(delta: np.ndarray, normal_i: np.ndarray, normal_j: np.ndarray, scale: float) -> tuple[bool, float]:
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0 or distance > EDGE_MAXIMUM_S * scale:
        return False, distance
    direction = delta / distance
    if _angle_degrees(normal_i, normal_j, unoriented=True) > EDGE_NORMAL_DEGREES:
        return False, distance
    if max(abs(float(np.dot(direction, normal_i))), abs(float(np.dot(direction, normal_j)))) > EDGE_NORMAL_TRAVEL_MAXIMUM:
        return False, distance
    return True, distance


def _surface_graph(candidate: Candidate, source: SourceGeometry, indices: np.ndarray) -> tuple[list[tuple[int, int, float]], list[list[int]]]:
    points = source.positions[indices]
    global_to_local = {int(value): pos for pos, value in enumerate(indices)}
    proposed: set[tuple[int, int]] = set()
    for ga, gb in source.mesh_edges:
        if ga in global_to_local and gb in global_to_local:
            a, b = global_to_local[ga], global_to_local[gb]
            proposed.add((min(a, b), max(a, b)))
    if len(points) > 1:
        k = min(MUTUAL_NEIGHBORS + 1, len(points))
        _distance, neighbors = cKDTree(points).query(points, k=k, workers=1)
        if k == 1:
            neighbors = neighbors[:, None]
        neighbor_sets = [set(int(v) for v in np.atleast_1d(row) if int(v) != i) for i, row in enumerate(neighbors)]
        for i, values in enumerate(neighbor_sets):
            for j in values:
                if i in neighbor_sets[j]:
                    proposed.add((min(i, j), max(i, j)))
    base: list[tuple[int, int, float]] = []
    degree_kernel = np.zeros(len(points), dtype=np.float64)
    for i, j in sorted(proposed):
        allowed, distance = _edge_allowed(points[j] - points[i], source.normals[indices[i]], source.normals[indices[j]], candidate.scale)
        if not allowed:
            continue
        b = (1.0 - (distance / (EDGE_MAXIMUM_S * candidate.scale)) ** 2) ** 2
        base.append((i, j, b))
        degree_kernel[i] += b
        degree_kernel[j] += b
    edges: list[tuple[int, int, float]] = []
    adjacency: list[list[int]] = [[] for _ in range(len(points))]
    for i, j, b in base:
        if degree_kernel[i] <= 0.0 or degree_kernel[j] <= 0.0:
            continue
        weight = _q24(b / math.sqrt(degree_kernel[i] * degree_kernel[j]))
        if weight <= 0.0:
            continue
        edges.append((i, j, weight))
        adjacency[i].append(j)
        adjacency[j].append(i)
    if len(edges) > MAX_STATION_EDGES:
        raise DHFFContractError("STATION_EDGE_CAP")
    for row in adjacency:
        row.sort()
    return edges, adjacency


def _components(adjacency: Sequence[Sequence[int]]) -> list[list[int]]:
    seen = np.zeros(len(adjacency), dtype=bool)
    values: list[list[int]] = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reversed(adjacency[node]):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        values.append(sorted(component))
    return values


def _orient_normals(source: SourceGeometry, indices: np.ndarray, adjacency: Sequence[Sequence[int]], components: Sequence[Sequence[int]], seeds_a: set[int]) -> np.ndarray:
    oriented = source.normals[indices].copy()
    for component in components:
        roots = sorted(seeds_a.intersection(component), key=lambda i: (int(source.stable_keys[indices[i]]), i))
        if not roots:
            continue
        root = roots[0]
        visited = {root}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if neighbor not in component or neighbor in visited:
                    continue
                if float(np.dot(oriented[node], oriented[neighbor])) < 0.0:
                    oriented[neighbor] = -oriented[neighbor]
                visited.add(neighbor)
                queue.append(neighbor)
    return oriented


def _pcg(node_count: int, edges: Sequence[tuple[int, int, float]], boundary_a: set[int], boundary_b: set[int], active: set[int]) -> np.ndarray | None:
    phi = np.zeros(node_count, dtype=np.float64)
    phi[list(boundary_b)] = 1.0
    interior = sorted(active - boundary_a - boundary_b)
    if not interior:
        return None
    mapping = {node: row for row, node in enumerate(interior)}
    diagonal = np.zeros(len(interior), dtype=np.float64)
    rhs = np.zeros(len(interior), dtype=np.float64)
    rows: list[list[tuple[int, float]]] = [[] for _ in interior]
    for i, j, weight in edges:
        if i not in active or j not in active:
            continue
        for node, other in ((i, j), (j, i)):
            if node not in mapping:
                continue
            row = mapping[node]
            diagonal[row] += weight
            if other in mapping:
                rows[row].append((mapping[other], -weight))
            elif other in boundary_b:
                rhs[row] += weight
    if np.any(diagonal <= 0.0):
        return None

    def multiply(vector: np.ndarray) -> np.ndarray:
        result = diagonal * vector
        for row, entries in enumerate(rows):
            for column, value in entries:
                result[row] += value * vector[column]
        return result

    x = np.zeros(len(interior), dtype=np.float64)
    residual = rhs - multiply(x)
    rhs_scale = max(1.0, float(np.max(np.abs(rhs))))
    if float(np.max(np.abs(residual))) <= PCG_RELATIVE_INFINITY_RESIDUAL * rhs_scale:
        phi[interior] = x
        return phi
    z = residual / diagonal
    direction = z.copy()
    rz = float(np.dot(residual, z))
    for _iteration in range(PCG_MAXIMUM_ITERATIONS):
        product = multiply(direction)
        denominator = float(np.dot(direction, product))
        if not math.isfinite(denominator) or denominator <= 0.0:
            return None
        step = rz / denominator
        x += step * direction
        residual -= step * product
        if float(np.max(np.abs(residual))) <= PCG_RELATIVE_INFINITY_RESIDUAL * rhs_scale:
            phi[interior] = x
            return phi
        z = residual / diagonal
        next_rz = float(np.dot(residual, z))
        if not math.isfinite(next_rz) or rz == 0.0:
            return None
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    return None


def _weighted_median(values: np.ndarray, weights: np.ndarray, stable: np.ndarray) -> float:
    order = np.lexsort((stable, values))
    threshold = 0.5 * float(np.sum(weights))
    cumulative = 0.0
    for index in order:
        cumulative += float(weights[index])
        if cumulative >= threshold:
            return float(values[index])
    return float(values[order[-1]])


def _signed_angles(normals: np.ndarray, reference: np.ndarray, axis: np.ndarray, alpha: float) -> np.ndarray:
    projected_reference = reference - float(np.dot(reference, axis)) * axis
    projected_reference /= np.linalg.norm(projected_reference)
    result = []
    sign = 1.0 if alpha >= 0.0 else -1.0
    for normal in normals:
        projected = normal - float(np.dot(normal, axis)) * axis
        norm = float(np.linalg.norm(projected))
        if norm <= 1.0e-12:
            result.append(float("nan"))
            continue
        projected /= norm
        angle = math.degrees(math.atan2(float(np.dot(axis, np.cross(projected_reference, projected))), float(np.dot(projected_reference, projected))))
        result.append(sign * angle)
    return np.asarray(result, dtype=np.float64)


def _station_failure(station: int, reason: str, center: np.ndarray, **extra: Any) -> dict[str, Any]:
    value = {"station": station, "pass": False, "reason": reason, "center": center.tolist(), "mask_bit": 0}
    value.update(extra)
    return value


def _analyze_station(candidate: Candidate, source: SourceGeometry, station: int) -> dict[str, Any]:
    center = _station_center(candidate, station)
    try:
        center, indices = _local_subset(candidate, source, station)
    except DHFFContractError as exc:
        return _station_failure(station, exc.code, center)
    seeds_a_local, tail_a = _tail_seeds(candidate, source, indices, center, 0)
    if not tail_a["valid"]:
        return _station_failure(station, "TAIL_A_BREADTH", center, node_indices=indices.tolist(), tail_a=tail_a)
    seeds_b_local, tail_b = _tail_seeds(candidate, source, indices, center, 1)
    if not tail_b["valid"]:
        return _station_failure(station, "TAIL_B_BREADTH", center, node_indices=indices.tolist(), tail_a=tail_a, tail_b=tail_b)
    try:
        edges, adjacency = _surface_graph(candidate, source, indices)
    except DHFFContractError as exc:
        return _station_failure(station, exc.code, center, node_indices=indices.tolist(), tail_a=tail_a, tail_b=tail_b)
    seeds_a = set(int(v) for v in seeds_a_local)
    seeds_b = set(int(v) for v in seeds_b_local)
    all_components = _components(adjacency)
    connecting = [component for component in all_components if seeds_a.intersection(component) and seeds_b.intersection(component)]
    connecting_nodes = set().union(*map(set, connecting)) if connecting else set()
    participating_a = seeds_a.intersection(connecting_nodes)
    participating_b = seeds_b.intersection(connecting_nodes)
    if not connecting or len(participating_a) < math.ceil(TAIL_COMPONENT_FRACTION * len(seeds_a)) or len(participating_b) < math.ceil(TAIL_COMPONENT_FRACTION * len(seeds_b)):
        return _station_failure(station, "NO_MATERIAL_CONNECTION", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), tail_a=tail_a, tail_b=tail_b)
    oriented = _orient_normals(source, indices, adjacency, connecting, seeds_a)
    expected_b = candidate.parents[1].normal
    b_angles = [_angle_degrees(oriented[index], expected_b) for index in sorted(participating_b)]
    if not b_angles or float(np.median(b_angles)) > ORIENTATION_CONFLICT_DEGREES:
        return _station_failure(station, "ORIENTATION_CONFLICT", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b))
    active = set().union(*map(set, connecting))
    phi = _pcg(len(indices), edges, participating_a, participating_b, active)
    if phi is None or not np.isfinite(phi[list(active)]).all():
        return _station_failure(station, "SOLVE_FAILURE", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b))
    currents = []
    participation = np.zeros(len(indices), dtype=np.float64)
    for i, j, weight in edges:
        if i not in active or j not in active:
            continue
        current = weight * abs(float(phi[i] - phi[j]))
        currents.append((i, j, current))
        participation[i] += 0.5 * current
        participation[j] += 0.5 * current
    total_mass = float(np.sum(participation))
    total_edge_current = float(sum(value[2] for value in currents))
    effective = total_mass * total_mass / max(float(np.dot(participation, participation)), 1.0e-300)
    bottleneck = max((value[2] for value in currents), default=0.0) / max(total_edge_current, 1.0e-300)
    if _q24(effective) < MINIMUM_EFFECTIVE_PARTICIPATION or _q24(bottleneck) > MAXIMUM_EDGE_CURRENT_FRACTION:
        return _station_failure(station, "LOW_CURRENT_PARTICIPATION", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), phi=phi.tolist(), currents=[list(v) for v in currents], effective=_q24(effective), bottleneck=_q24(bottleneck))
    active_nodes = participation > 0.0
    band_indices: list[np.ndarray] = []
    for k in range(1, 8):
        mask = active_nodes & (np.abs(phi - k / 8.0) <= POTENTIAL_BAND_HALF_WIDTH)
        band = np.flatnonzero(mask)
        if len(band) < POTENTIAL_BAND_MINIMUM_NODES or _q24(float(np.sum(participation[band])) / total_mass) < POTENTIAL_BAND_MINIMUM_MASS:
            return _station_failure(station, "POTENTIAL_COVERAGE", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), phi=phi.tolist(), currents=[list(v) for v in currents])
        band_indices.append(band)
    alpha = abs(candidate.alpha_degrees)
    theta = _signed_angles(oriented, candidate.parents[0].normal, candidate.axis, candidate.alpha_degrees)
    for m in range(1, 8):
        mask = active_nodes & np.isfinite(theta) & (np.abs(theta - m * alpha / 8.0) <= alpha / 16.0)
        angular = np.flatnonzero(mask)
        if len(angular) < ANGULAR_BIN_MINIMUM_NODES or _q24(float(np.sum(participation[angular])) / total_mass) < ANGULAR_BIN_MINIMUM_MASS:
            return _station_failure(station, "ANGULAR_COVERAGE", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), phi=phi.tolist(), currents=[list(v) for v in currents], theta=theta.tolist())
    off_arc_mask = active_nodes & ((theta < -OFF_ARC_MARGIN_DEGREES) | (theta > alpha + OFF_ARC_MARGIN_DEGREES) | ~np.isfinite(theta))
    off_arc = float(np.sum(participation[off_arc_mask])) / total_mass
    if _q24(off_arc) > MAXIMUM_OFF_ARC_MASS:
        return _station_failure(station, "OFF_ARC_CURRENT", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), phi=phi.tolist(), currents=[list(v) for v in currents], theta=theta.tolist(), off_arc_fraction=_q24(off_arc))
    medians = [0.0]
    stable = source.stable_keys[indices]
    for band in band_indices:
        medians.append(_weighted_median(theta[band], participation[band], stable[band]))
    medians.append(alpha)
    backtrack = sum(max(0.0, medians[k] - medians[k + 1]) for k in range(8))
    variation = sum(abs(medians[k + 1] - medians[k]) for k in range(8)) - alpha
    if _q24(backtrack) > MAXIMUM_BACKTRACK_DEGREES or _q24(variation) > MAXIMUM_ANGULAR_VARIATION_DEGREES:
        return _station_failure(station, "NORMAL_BACKTRACK", center, node_indices=indices.tolist(), edges=[list(v) for v in edges], seeds_a=sorted(seeds_a), seeds_b=sorted(seeds_b), phi=phi.tolist(), currents=[list(v) for v in currents], theta=theta.tolist(), medians=medians, backtrack_degrees=_q24(backtrack), variation_degrees=_q24(variation), off_arc_fraction=_q24(off_arc))
    return {"station": station, "pass": True, "reason": "STATION_PASS", "center": center.tolist(), "mask_bit": 1, "node_indices": indices.tolist(), "edges": [list(v) for v in edges], "seeds_a": sorted(seeds_a), "seeds_b": sorted(seeds_b), "phi": phi.tolist(), "currents": [list(v) for v in currents], "theta": theta.tolist(), "medians": medians, "effective_participation": _q24(effective), "maximum_edge_current_fraction": _q24(bottleneck), "off_arc_fraction": _q24(off_arc), "backtrack_degrees": _q24(backtrack), "variation_degrees": _q24(variation)}


def _longest_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _candidate_decision(candidate: Candidate, source: SourceGeometry) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate.selected_legacy:
        decision = {"candidate_id": candidate.candidate_id, "frozen_v16_order": candidate.frozen_order, "action": ACTION_LEGACY, "reason": None, "station_mask": 0, "station_reasons": [], "passing_stations": 0, "longest_run": 0, "median_normal_backtrack": 0.0, "median_off_arc_fraction": 0.0, "profile_hash": "0" * 64}
        return decision, {"candidate_id": candidate.candidate_id, "stations": []}
    if not candidate.eligible:
        reason = "TURN_ELIGIBILITY"
        decision = {"candidate_id": candidate.candidate_id, "frozen_v16_order": candidate.frozen_order, "action": ACTION_NONE, "reason": reason, "station_mask": 0, "station_reasons": [reason] * STATION_COUNT, "passing_stations": 0, "longest_run": 0, "median_normal_backtrack": 0.0, "median_off_arc_fraction": 0.0, "profile_hash": hashlib.sha256(reason.encode()).hexdigest()}
        return decision, {"candidate_id": candidate.candidate_id, "stations": []}
    stations = [_analyze_station(candidate, source, station) for station in range(STATION_COUNT)]
    passes = [bool(value["pass"]) for value in stations]
    passing_count = sum(passes)
    longest = _longest_true_run(passes)
    certified = passing_count >= MINIMUM_PASSING_STATIONS and longest >= MINIMUM_CONSECUTIVE_PASSING and any(passes[0:2]) and any(passes[9:11])
    failing_reasons = [str(value["reason"]) for value in stations if not value["pass"]]
    if certified:
        reason = None
    elif failing_reasons and len(set(failing_reasons)) == 1 and passing_count == 0:
        reason = failing_reasons[0]
    else:
        reason = "AXIAL_STATION_COVERAGE"
    backtracks = [float(value.get("backtrack_degrees", 0.0)) for value in stations if value["pass"]]
    off_arcs = [float(value.get("off_arc_fraction", 0.0)) for value in stations if value["pass"]]
    summary = {"mask": passes, "reasons": [value["reason"] for value in stations], "backtrack": backtracks, "off_arc": off_arcs}
    decision = {
        "candidate_id": candidate.candidate_id,
        "frozen_v16_order": candidate.frozen_order,
        "action": ACTION_RESCUE if certified else ACTION_NONE,
        "reason": reason,
        "station_mask": sum((1 << index) for index, value in enumerate(passes) if value),
        "station_reasons": [str(value["reason"]) for value in stations],
        "passing_stations": passing_count,
        "longest_run": longest,
        "median_normal_backtrack": _q24(float(np.median(backtracks))) if backtracks else 0.0,
        "median_off_arc_fraction": _q24(float(np.median(off_arcs))) if off_arcs else 0.0,
        "profile_hash": hashlib.sha256(_canonical_json_bytes(summary)).hexdigest(),
    }
    return decision, {"candidate_id": candidate.candidate_id, "stations": stations}


def deterministic_total_sort(candidates: Sequence[Candidate], decisions: Sequence[dict[str, Any]]) -> list[tuple[Candidate, dict[str, Any]]]:
    rows = list(zip(candidates, decisions))
    legacy = sorted((row for row in rows if row[1]["action"] == ACTION_LEGACY), key=lambda row: (row[0].frozen_order, row[0].candidate_id))
    rescues = sorted(
        (row for row in rows if row[1]["action"] == ACTION_RESCUE),
        key=lambda row: (-int(row[1]["passing_stations"]), -int(row[1]["longest_run"]), float(row[1]["median_normal_backtrack"]), float(row[1]["median_off_arc_fraction"]), row[0].frozen_order, row[0].candidate_id),
    )
    return legacy + rescues


def _line_metrics(first: Candidate, second: Candidate) -> tuple[float, float, float]:
    axis_a = first.line_end - first.line_start
    axis_b = second.line_end - second.line_start
    length_a, length_b = float(np.linalg.norm(axis_a)), float(np.linalg.norm(axis_b))
    if length_a <= 0.0 or length_b <= 0.0:
        return 180.0, math.inf, 0.0
    axis_a /= length_a
    axis_b /= length_b
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))))
    center_a = 0.5 * (first.line_start + first.line_end)
    center_b = 0.5 * (second.line_start + second.line_end)
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta - np.dot(delta, axis_a) * axis_a))
    along = abs(float(np.dot(delta, axis_a)))
    overlap = max(0.0, 0.5 * (length_a + length_b) - along) / min(length_a, length_b)
    return angle, distance, overlap


def ordinary_nms(ordered: Sequence[tuple[Candidate, dict[str, Any]]]) -> tuple[list[tuple[Candidate, dict[str, Any]]], set[int]]:
    accepted: list[tuple[Candidate, dict[str, Any]]] = []
    suppressed: set[int] = set()
    for candidate, decision in ordered:
        duplicate = False
        for prior, _prior_decision in accepted:
            angle, distance, overlap = _line_metrics(candidate, prior)
            if angle <= NMS_ANGLE_DEGREES and distance <= NMS_DISTANCE_MM and overlap >= NMS_MINIMUM_OVERLAP:
                duplicate = True
                break
        if duplicate:
            suppressed.add(candidate.candidate_id)
        else:
            accepted.append((candidate, decision))
    return accepted, suppressed


def build_fixed_cache(decisions: Sequence[dict[str, Any]]) -> bytes:
    header = bytearray(CACHE_HEADER_BYTES)
    profile_digest = hashlib.sha256(PROFILE.encode("ascii")).digest()
    header[:8] = b"DHFFV1\0\0"
    struct.pack_into("<IIII", header, 8, SCHEMA_VERSION, len(decisions), CACHE_CANDIDATE_BYTES, MAX_CACHE_BYTES)
    header[32:64] = profile_digest
    records = bytearray()
    for decision in sorted(decisions, key=lambda value: (int(value["frozen_v16_order"]), int(value["candidate_id"]))):
        record = bytearray(CACHE_CANDIDATE_BYTES)
        struct.pack_into("<QIIH", record, 0, int(decision["candidate_id"]), int(decision["frozen_v16_order"]), 1 if decision["action"] == ACTION_RESCUE else 0, int(decision["station_mask"]))
        reasons = list(decision["station_reasons"])
        for index in range(STATION_COUNT):
            record[22 + index] = REASON_CODE.get(reasons[index] if index < len(reasons) else "AXIAL_STATION_COVERAGE", 255)
        struct.pack_into("<IIii", record, 40, int(decision["passing_stations"]), int(decision["longest_run"]), _round_half_away(float(decision["median_normal_backtrack"]) * Q24), _round_half_away(float(decision["median_off_arc_fraction"]) * Q24))
        record[64:96] = bytes.fromhex(str(decision["profile_hash"]))
        records.extend(record)
    cache = bytes(header + records)
    if len(cache) > MAX_CACHE_BYTES:
        raise DHFFContractError("RESOURCE_CAP", "fixed_cache")
    return cache


def select_unified_v31(raw_source: dict[str, Any], raw_candidates: Sequence[dict[str, Any]], *, worker_count: int = 1) -> dict[str, Any]:
    workers = int(worker_count)
    physical = os.cpu_count() or 1
    if workers < 1 or workers > min(MAX_WORKERS, physical):
        raise DHFFContractError("INVALID_WORKER_COUNT")
    source = parse_source(raw_source)
    candidates = parse_candidates(raw_candidates)

    def analyze(candidate: Candidate) -> tuple[dict[str, Any], dict[str, Any]]:
        return _candidate_decision(candidate, source)

    if workers == 1:
        analyzed = [analyze(candidate) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v31-dhff") as executor:
            analyzed = list(executor.map(analyze, candidates))
    decisions = [value[0] for value in analyzed]
    visuals = [value[1] for value in analyzed]
    ordered = deterministic_total_sort(candidates, decisions)
    accepted, suppressed = ordinary_nms(ordered)
    for decision in decisions:
        if int(decision["candidate_id"]) in suppressed:
            decision["nms_reason"] = "ORDINARY_NMS_SUPPRESSED"
    final_lines = [
        {
            "candidate_id": candidate.candidate_id,
            "line_start": candidate.line_start.tolist(),
            "line_end": candidate.line_end.tolist(),
            "source_lane": "legacy" if decision["action"] == ACTION_LEGACY else "v31_dhff_rescue",
            "v31_action": decision["action"],
        }
        for candidate, decision in accepted
    ]
    fixed_cache = build_fixed_cache(decisions)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sherman_qc_f1_v31_dhff_research_result",
        "profile": PROFILE,
        "primitive": PRIMITIVE,
        "research_only": True,
        "source_only": True,
        "rescue_only": True,
        "expected_count_used": False,
        "part_identity_used": False,
        "filename_identity_used": False,
        "labels_used": False,
        "cad_used": False,
        "decisions": decisions,
        "final_lines": final_lines,
        "pre_nms_rescue_count": sum(value["action"] == ACTION_RESCUE for value in decisions),
        "rescue_count": sum(line["source_lane"] == "v31_dhff_rescue" for line in final_lines),
        "ordinary_nms_call_count": 1,
        "candidate_added_after_nms": False,
        "cache_sha256": hashlib.sha256(fixed_cache).hexdigest(),
        "cache_bytes": len(fixed_cache),
        "_ephemeral": {"candidate_visuals": visuals, "fixed_cache": fixed_cache, "source": source, "candidates": candidates, "nms_suppressed": sorted(suppressed)},
    }
