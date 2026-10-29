"""V70 boundary-rejecting multiscale variation track composition (BRMVT_V1).

Research-only, source-only replacement composition over the frozen V16
thresholded pre-NMS stream.  The implementation follows the locked 2026-08-16
GPT-5.6 Pro ruling: multiscale covariance variation, local ridge contrast,
boundary-rejecting hysteresis, partial candidate support tracks, geometry-only
duplicate ownership, then one unchanged ordinary NMS supplied by the caller.

No score, rank, route, identity, expected count, CAD, label, visual decision,
or V52 result is inspected.  Candidate endpoints and directions are emitted
unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


ARCHITECTURE = "V70_BOUNDARY_REJECTING_MULTISCALE_VARIATION_TRACK_COMPOSITION_V1"
SHORT_NAME = "BRMVT_V1"
SCHEMA = "sherman_qc_f1_brmvt_v1_audit"
RULING_PATH = Path(__file__).resolve().parents[1] / "docs/research/2026-08-16_gpt56_pro_post_v69_real_morphology_ruling_verbatim.md"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests/test_f1_boundary_rejecting_multiscale_variation_track_composition_v70.py"

K_ALL = (6, 12, 24, 64, 160, 256, 512, 1024, 1280)
K_VARIATION = (24, 64, 160, 256, 512, 1024)
MAX_POINTS = 200_000
MAX_TRIANGLES = 350_000
MAX_CANDIDATES = 128
MAX_TIE_EXPANSION = 64
MAX_NEIGHBOR_REFERENCES = 268_800_000
MAX_GRAPH_EDGES = 8_000_000
MAX_FEATURE_NODES_PER_CANDIDATE = 8_192
MAX_ATOMS_PER_BIN = 16
MAX_ATOMS_PER_CANDIDATE = 1_024
MAX_CANDIDATE_NODE_VISITS = 1_048_576
MAX_DAG_EDGE_EXAMINATIONS = 25_000_000
MAX_AUDIT_BYTES = 512 * 1024
MAX_RSS_BYTES = 16 * 1024**3
MAX_WALL_SECONDS = 300.0

U_LOW = 0.70
U_HIGH = 0.95
A_LOW = 0.01
A_HIGH = 0.10
BOUNDARY_START = 5.0 * math.pi / 9.0
BOUNDARY_SPAN = 4.0 * math.pi / 9.0
BOUNDARY_PENALTY = 0.70
WEAK_THRESHOLD = 0.20
STRONG_THRESHOLD = 0.55
HYSTERESIS_DISTANCE = 8.0
TRACK_STRENGTH = 0.35


class BRMVTInputError(RuntimeError):
    """Invalid input or a deterministic numeric failure."""


class BRMVTResourceAbort(RuntimeError):
    """Frozen operational cap exceeded; no candidate mutation may commit."""


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_time(started: float) -> None:
    if time.perf_counter() - started > MAX_WALL_SECONDS:
        raise BRMVTResourceAbort("WALL_TIME_CAP")
    if _rss_bytes() >= MAX_RSS_BYTES:
        raise BRMVTResourceAbort("RSS_CAP")


def _clean_float_array(value: Any, *, shape_tail: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if shape_tail is not None and (array.ndim < len(shape_tail) or tuple(array.shape[-len(shape_tail):]) != shape_tail):
        raise BRMVTInputError("MALFORMED_ARRAY")
    if not np.isfinite(array).all():
        raise BRMVTInputError("NONFINITE_DATUM")
    array[array == 0.0] = 0.0
    return array


def _hex(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise BRMVTInputError("NONFINITE_DERIVED_SCALAR")
    return value.hex()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_hash(row: Mapping[str, Any], start: np.ndarray, end: np.ndarray) -> str:
    supplied = row.get("route_hash") or row.get("candidate_opaque_hash") or row.get("opaque_candidate_hash") or row.get("bend_id")
    if supplied is not None:
        return _sha(str(supplied).encode("utf-8"))
    parts = sorted((start.astype(">f8", copy=False).tobytes(), end.astype(">f8", copy=False).tobytes()))
    return _sha(parts[0] + parts[1])


@dataclass(frozen=True)
class Candidate:
    opaque_hash: str
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    length: float
    ordinal: int
    original: Mapping[str, Any]


def _candidate(row: Mapping[str, Any], ordinal: int) -> Candidate:
    try:
        start = _clean_float_array(row["line_start"])
        end = _clean_float_array(row["line_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise BRMVTInputError("MALFORMED_CANDIDATE") from error
    if start.shape != (3,) or end.shape != (3,):
        raise BRMVTInputError("MALFORMED_CANDIDATE")
    # Internal station/bin orientation is canonical.  The original row remains
    # untouched and is the only object ever emitted, so this provides endpoint
    # reversal invariance without modifying candidate geometry.
    if end.astype(">f8", copy=False).tobytes() < start.astype(">f8", copy=False).tobytes():
        start, end = end.copy(), start.copy()
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length == 0.0:
        raise BRMVTInputError("ZERO_LENGTH_CANDIDATE")
    return Candidate(_candidate_hash(row, start, end), start, end, delta / length, length, ordinal, row)


@dataclass
class CanonicalSource:
    points: np.ndarray
    faces: np.ndarray
    mesh_edges: np.ndarray
    mesh_boundary: np.ndarray
    source_sha256: str


def _canonical_source(vertices: Any, faces: Any | None) -> CanonicalSource:
    raw = _clean_float_array(vertices, shape_tail=(3,))
    if raw.ndim != 2:
        raise BRMVTInputError("MALFORMED_SOURCE_POINTS")
    if len(raw) > MAX_POINTS:
        raise BRMVTResourceAbort("POINT_CAP")
    points, inverse = np.unique(raw, axis=0, return_inverse=True)
    if not 2048 <= len(points) <= MAX_POINTS:
        raise BRMVTInputError("POINT_COUNT_OUT_OF_RANGE")

    if faces is None:
        tri = np.empty((0, 3), dtype=np.int64)
    else:
        face_array = np.asarray(faces)
        if face_array.size == 0:
            tri = np.empty((0, 3), dtype=np.int64)
        else:
            if face_array.ndim != 2 or face_array.shape[1] != 3:
                raise BRMVTInputError("MALFORMED_TRIANGLES")
            if not np.issubdtype(face_array.dtype, np.integer):
                if not np.isfinite(face_array).all() or not np.equal(face_array, np.floor(face_array)).all():
                    raise BRMVTInputError("INVALID_TRIANGLE_INDEX")
            face_array = face_array.astype(np.int64, copy=False)
            if len(face_array) > MAX_TRIANGLES or np.any(face_array < 0) or np.any(face_array >= len(raw)):
                raise BRMVTInputError("INVALID_TRIANGLE_INDEX")
            tri = inverse[face_array]
            tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])]
            tri = np.sort(tri, axis=1)
            tri = np.unique(tri, axis=0)

    if len(tri):
        edges = np.vstack((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (0, 2)]))
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        mesh_boundary = np.zeros(len(points), dtype=bool)
        boundary_vertices = np.unique(unique_edges[counts != 2])
        mesh_boundary[boundary_vertices] = True
        adjacency = [set() for _ in range(len(points))]
        for first, second in unique_edges:
            adjacency[int(first)].add(int(second))
            adjacency[int(second)].add(int(first))
        one_hop = set(int(value) for value in boundary_vertices)
        two_hop = set(one_hop)
        for index in one_hop:
            two_hop.update(adjacency[index])
        for index in list(two_hop):
            two_hop.update(adjacency[index])
        mesh_boundary[list(two_hop)] = True
        mesh_edges = unique_edges.astype(np.int32)
    else:
        mesh_edges = np.empty((0, 2), dtype=np.int32)
        mesh_boundary = np.zeros(len(points), dtype=bool)
    payload = points.astype(">f8", copy=False).tobytes() + tri.astype(">i8", copy=False).tobytes()
    return CanonicalSource(points, tri.astype(np.int32), mesh_edges, mesh_boundary, _sha(payload))


@dataclass
class Neighborhoods:
    indices: list[np.ndarray]
    h: np.ndarray
    h0: float
    reference_count: int


def _neighborhoods(points: np.ndarray, started: float) -> Neighborhoods:
    count = len(points)
    maximum = min(1280, count - 1)
    if maximum < 6:
        raise BRMVTInputError("POINT_COUNT_OUT_OF_RANGE")
    tree = cKDTree(points, balanced_tree=True, compact_nodes=True)
    output: list[np.ndarray] = []
    reference_count = 0
    query_k = min(count, maximum + 1)
    for begin in range(0, count, 1024):
        _check_time(started)
        end = min(count, begin + 1024)
        distances, indices = tree.query(points[begin:end], k=query_k, workers=1)
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for local, point_index in enumerate(range(begin, end)):
            pairs = [(float(distance), int(index)) for distance, index in zip(distances[local], indices[local]) if int(index) != point_index]
            pairs.sort(key=lambda item: (item[0], item[1]))
            if len(pairs) < maximum:
                raise BRMVTInputError("NEIGHBOR_QUERY_FAILURE")
            boundary_distance = pairs[maximum - 1][0]
            tied = tree.query_ball_point(points[point_index], np.nextafter(boundary_distance, math.inf))
            tied = [int(index) for index in tied if int(index) != point_index]
            tied.sort(key=lambda index: (float(np.dot(points[index] - points[point_index], points[index] - points[point_index])), index))
            selected = [index for index in tied if float(np.linalg.norm(points[index] - points[point_index])) <= np.nextafter(boundary_distance, math.inf)]
            if len(selected) > maximum + MAX_TIE_EXPANSION:
                raise BRMVTResourceAbort("NEIGHBOR_TIE_CAP_ABORT")
            array = np.asarray(selected, dtype=np.int32)
            output.append(array)
            reference_count += len(array)
            if reference_count > MAX_NEIGHBOR_REFERENCES:
                raise BRMVTResourceAbort("NEIGHBOR_REFERENCE_CAP")
    h = np.empty(count, dtype=np.float64)
    for index, neighbors in enumerate(output):
        first_six = neighbors[:6]
        h[index] = float(np.median(np.linalg.norm(points[first_six] - points[index], axis=1)))
    h0 = float(np.median(h))
    if not math.isfinite(h0) or h0 <= 0.0:
        raise BRMVTInputError("NONPOSITIVE_SAMPLING_SCALE")
    return Neighborhoods(output, h, h0, reference_count)


def _covariance(points: np.ndarray) -> np.ndarray:
    mean = np.mean(points, axis=0, dtype=np.float64)
    centered = points - mean
    # Canonical point order makes this sum independent of input ordering.
    return centered.T @ centered / float(len(points))


@dataclass
class RidgeField:
    sigma: np.ndarray
    u: np.ndarray
    v: np.ndarray
    persistence: np.ndarray
    contrast: np.ndarray
    saliency: np.ndarray
    boundary: np.ndarray
    weight: np.ndarray
    strong: np.ndarray
    weak: np.ndarray
    feature: np.ndarray
    graph_edges: np.ndarray


def _clip01(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def local_percentile(value: float, peers: np.ndarray) -> float:
    return (float(np.count_nonzero(peers < value)) + 0.5 * float(np.count_nonzero(peers == value))) / float(len(peers))


def _ridge_field(source: CanonicalSource, neighborhoods: Neighborhoods, started: float) -> RidgeField:
    points = source.points
    count = len(points)
    sigma = np.zeros((count, len(K_VARIATION)), dtype=np.float64)
    eigen24 = np.zeros((count, 3), dtype=np.float64)
    vectors24 = np.zeros((count, 3, 3), dtype=np.float64)
    for point_index in range(count):
        if point_index % 512 == 0:
            _check_time(started)
        neighbors = neighborhoods.indices[point_index]
        for scale_index, cardinality in enumerate(K_VARIATION):
            selected = neighbors[: min(cardinality, len(neighbors))]
            covariance = _covariance(points[selected])
            values, vectors = np.linalg.eigh(covariance)
            trace = float(np.trace(covariance))
            tolerance = math.ldexp(trace, -40)
            if float(np.min(values)) < -tolerance:
                raise BRMVTInputError("EIGENVALUE_ABORT")
            values = np.maximum(values, 0.0)
            sigma[point_index, scale_index] = 0.0 if trace == 0.0 else float(values[0] / trace)
            if cardinality == 24:
                eigen24[point_index] = values
                vectors24[point_index] = vectors

    u = np.zeros_like(sigma)
    v = np.zeros_like(sigma)
    for point_index in range(count):
        peers_index = neighborhoods.indices[point_index]
        for scale_index in range(len(K_VARIATION)):
            peers = sigma[peers_index, scale_index]
            rank = local_percentile(float(sigma[point_index, scale_index]), peers)
            u[point_index, scale_index] = float(_clip01((rank - U_LOW) / (U_HIGH - U_LOW)))
            median = float(np.median(peers))
            contrast = max(0.0, float(sigma[point_index, scale_index] - median) / max(median, math.ldexp(1.0, -20)))
            v[point_index, scale_index] = float(_clip01((contrast - A_LOW) / (A_HIGH - A_LOW)))
    ordered = np.sort(u, axis=1)
    persistence = np.sqrt(ordered[:, -1] * ordered[:, -2])
    contrast = np.sqrt(np.max(v, axis=1))
    saliency = persistence * contrast

    angular = np.zeros(count, dtype=np.float64)
    for point_index in range(count):
        values = eigen24[point_index]
        if values[1] == 0.0:
            angular[point_index] = 1.0
            continue
        if values[0] / values[1] > 0.35:
            continue
        basis = vectors24[point_index][:, 1:3]
        selected = neighborhoods.indices[point_index][:24]
        projected = (points[selected] - points[point_index]) @ basis
        angles = np.sort(np.arctan2(projected[:, 1], projected[:, 0]))
        gaps = np.diff(np.concatenate((angles, [angles[0] + 2.0 * math.pi])))
        largest = float(np.max(gaps))
        angular[point_index] = float(_clip01((largest - BOUNDARY_START) / BOUNDARY_SPAN))
    boundary = np.maximum(angular, source.mesh_boundary.astype(np.float64))
    weight = saliency * (1.0 - BOUNDARY_PENALTY * boundary)
    strong = weight >= STRONG_THRESHOLD
    weak = weight >= WEAK_THRESHOLD

    edge_set: set[tuple[int, int]] = set()
    for first, neighbors in enumerate(neighborhoods.indices):
        for second in neighbors[:12]:
            a, b = sorted((first, int(second)))
            if a != b:
                edge_set.add((a, b))
    edge_set.update((int(first), int(second)) for first, second in source.mesh_edges)
    if len(edge_set) > MAX_GRAPH_EDGES:
        raise BRMVTResourceAbort("SOURCE_GRAPH_EDGE_CAP")
    valid_edges = []
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for first, second in sorted(edge_set):
        distance = float(np.linalg.norm(points[first] - points[second]))
        if 0.0 < distance <= 3.0 * max(neighborhoods.h[first], neighborhoods.h[second]):
            cost = distance / max(neighborhoods.h[first], neighborhoods.h[second])
            adjacency[first].append((second, cost))
            adjacency[second].append((first, cost))
            valid_edges.append((first, second))
    distance_to_strong = np.full(count, math.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for index in np.flatnonzero(strong):
        distance_to_strong[int(index)] = 0.0
        heapq.heappush(heap, (0.0, int(index)))
    while heap:
        distance, index = heapq.heappop(heap)
        if distance != distance_to_strong[index] or distance > HYSTERESIS_DISTANCE:
            continue
        for neighbor, cost in adjacency[index]:
            if not weak[neighbor]:
                continue
            proposal = distance + cost
            if proposal <= HYSTERESIS_DISTANCE and proposal < distance_to_strong[neighbor]:
                distance_to_strong[neighbor] = proposal
                heapq.heappush(heap, (proposal, neighbor))
    feature = weak & (distance_to_strong <= HYSTERESIS_DISTANCE)
    return RidgeField(sigma, u, v, persistence, contrast, saliency, boundary, weight, strong, weak, feature, np.asarray(valid_edges, dtype=np.int32))


@dataclass(frozen=True)
class Atom:
    bin_index: int
    centroid: np.ndarray
    rho: float
    alpha: float
    beta: float
    node_indices: tuple[int, ...]
    canonical_hash: str


@dataclass(frozen=True)
class PathState:
    atoms: tuple[int, ...]
    occupied: int
    beta_sum: float
    rho_over_h_sum: float


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    hc: float
    bin_count: int
    atoms: list[Atom]
    selected_path: tuple[int, ...]
    occupied_length: float
    mean_beta: float
    admissibility: float
    mean_rho_over_h: float
    support_nodes: set[int]
    strong_count: int
    weak_bridge_count: int
    boundary_q50: float
    boundary_q90: float
    decision: str
    terminal_reason: str
    duplicate_owner: str | None = None


def _orthogonal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def _components(coords: np.ndarray, radius: float) -> list[np.ndarray]:
    if len(coords) == 0:
        return []
    tree = cKDTree(coords)
    neighbors = tree.query_ball_tree(tree, radius)
    unseen = set(range(len(coords)))
    output = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in sorted(neighbors[index], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        output.append(np.asarray(sorted(component), dtype=np.int32))
    return output


def _better_state(first: PathState, second: PathState | None) -> bool:
    if second is None:
        return True
    first_key = (first.beta_sum / first.occupied, first.occupied, -first.rho_over_h_sum / first.occupied, tuple(-value for value in first.atoms))
    second_key = (second.beta_sum / second.occupied, second.occupied, -second.rho_over_h_sum / second.occupied, tuple(-value for value in second.atoms))
    return first_key > second_key


def _evaluate_candidate(candidate: Candidate, source: CanonicalSource, neighborhoods: Neighborhoods, ridge: RidgeField, started: float) -> CandidateEvaluation:
    points = source.points
    tree = cKDTree(points)
    positions = candidate.start + ((np.arange(17, dtype=np.float64) + 0.5) / 17.0)[:, None] * (candidate.end - candidate.start)
    _distances, station_neighbors = tree.query(positions, k=8, workers=1)
    station_neighbors = np.asarray(station_neighbors, dtype=np.int64)
    hc_hat = float(np.median(np.median(neighborhoods.h[station_neighbors], axis=1)))
    hc = float(np.clip(hc_hat, 0.5 * neighborhoods.h0, 2.0 * neighborhoods.h0))
    along = (points - candidate.start) @ candidate.direction
    perpendicular = points - candidate.start - along[:, None] * candidate.direction
    distance = np.linalg.norm(perpendicular, axis=1)
    tube = np.flatnonzero(ridge.feature & (along >= -3.0 * hc) & (along <= candidate.length + 3.0 * hc) & (distance <= 6.0 * hc))
    if len(tube) > MAX_FEATURE_NODES_PER_CANDIDATE:
        raise BRMVTResourceAbort("CANDIDATE_FEATURE_NODE_CAP")
    bin_count = min(64, max(4, int(math.ceil(candidate.length / (2.0 * hc)))))
    delta = candidate.length / bin_count
    first_axis, second_axis = _orthogonal_basis(candidate.direction)
    atoms: list[Atom] = []
    for bin_index in range(bin_count):
        local = tube[(along[tube] >= bin_index * delta) & (along[tube] < (bin_index + 1) * delta if bin_index + 1 < bin_count else along[tube] <= candidate.length)]
        if len(local) == 0:
            continue
        coords = np.column_stack((perpendicular[local] @ first_axis, perpendicular[local] @ second_axis))
        components = _components(coords, 2.5 * hc)
        if len(components) > MAX_ATOMS_PER_BIN:
            raise BRMVTResourceAbort("ATOMS_PER_BIN_CAP")
        for component in components:
            nodes = local[component]
            weights = ridge.weight[nodes]
            if float(np.sum(weights)) > 0.0:
                centroid = np.average(points[nodes], axis=0, weights=weights)
            else:
                centroid = np.mean(points[nodes], axis=0)
            rho = float(np.linalg.norm(centroid - candidate.start - np.dot(centroid - candidate.start, candidate.direction) * candidate.direction))
            strongest = np.sort(weights)[-min(5, len(weights)):]
            alpha = float(np.median(strongest))
            beta = alpha / (1.0 + (rho / (4.0 * hc)) ** 2)
            atom_hash = _sha(canonical_bytes({"bin": bin_index, "nodes": [int(value) for value in sorted(nodes.tolist())]}))
            atoms.append(Atom(bin_index, centroid, rho, alpha, beta, tuple(int(value) for value in sorted(nodes.tolist())), atom_hash))
    atoms.sort(key=lambda value: (value.bin_index, value.canonical_hash))
    if len(atoms) > MAX_ATOMS_PER_CANDIDATE:
        raise BRMVTResourceAbort("ATOMS_PER_CANDIDATE_CAP")
    if not atoms:
        return CandidateEvaluation(candidate, hc, bin_count, atoms, (), 0.0, 0.0, 0.0, math.inf, set(), 0, 0, 0.0, 0.0, "DROP", "NO_FEATURE_NODE")

    incoming: list[list[int]] = [[] for _ in atoms]
    dag_edges = 0
    for right, atom in enumerate(atoms):
        for left in range(right):
            previous = atoms[left]
            gap = atom.bin_index - previous.bin_index
            if gap < 1 or gap > 3:
                continue
            dag_edges += 1
            if dag_edges > MAX_DAG_EDGE_EXAMINATIONS:
                raise BRMVTResourceAbort("DAG_EDGE_EXAMINATION_CAP")
            first_perp = previous.centroid - candidate.start - np.dot(previous.centroid - candidate.start, candidate.direction) * candidate.direction
            second_perp = atom.centroid - candidate.start - np.dot(atom.centroid - candidate.start, candidate.direction) * candidate.direction
            if float(np.linalg.norm(first_perp - second_perp)) <= 3.0 * hc:
                incoming[right].append(left)
    states: list[dict[int, PathState]] = [dict() for _ in atoms]
    visits = 0
    for index, atom in enumerate(atoms):
        initial = PathState((index,), 1, atom.beta, atom.rho / hc)
        states[index][1] = initial
        for previous in incoming[index]:
            for occupied, state in sorted(states[previous].items()):
                visits += 1
                if visits > MAX_CANDIDATE_NODE_VISITS:
                    raise BRMVTResourceAbort("CANDIDATE_NODE_VISIT_CAP")
                proposal = PathState(state.atoms + (index,), occupied + 1, state.beta_sum + atom.beta, state.rho_over_h_sum + atom.rho / hc)
                if _better_state(proposal, states[index].get(occupied + 1)):
                    states[index][occupied + 1] = proposal
    threshold_length = min(candidate.length, max(8.0 * hc, 0.30 * candidate.length))
    best: tuple[float, float, float, float, str, PathState] | None = None
    for mapping in states:
        for state in mapping.values():
            occupied_length = state.occupied * delta
            mean_beta = state.beta_sum / state.occupied
            admissibility = min(occupied_length / threshold_length, mean_beta / TRACK_STRENGTH)
            path_hash = _sha(canonical_bytes([atoms[index].canonical_hash for index in state.atoms]))
            key = (admissibility, mean_beta, occupied_length, -state.rho_over_h_sum / state.occupied, "".join(chr(255 - ord(char)) for char in path_hash), state)
            if best is None or key[:5] > best[:5]:
                best = key
    assert best is not None
    state = best[5]
    occupied_length = state.occupied * delta
    mean_beta = state.beta_sum / state.occupied
    admissibility = min(occupied_length / threshold_length, mean_beta / TRACK_STRENGTH)
    selected_atoms = [atoms[index] for index in state.atoms]
    support_nodes: set[int] = set()
    for atom in selected_atoms:
        eligible_bins = {atom.bin_index - 1, atom.bin_index, atom.bin_index + 1}
        for node in tube:
            node_bin = min(bin_count - 1, max(0, int(math.floor(np.clip(along[node], 0.0, np.nextafter(candidate.length, 0.0)) / delta))))
            if node_bin not in eligible_bins:
                continue
            node_perp = points[node] - candidate.start - np.dot(points[node] - candidate.start, candidate.direction) * candidate.direction
            atom_perp = atom.centroid - candidate.start - np.dot(atom.centroid - candidate.start, candidate.direction) * candidate.direction
            if float(np.linalg.norm(node_perp - atom_perp)) <= 2.0 * hc:
                support_nodes.add(int(node))
    tube_boundary = ridge.boundary[tube]
    decision = "KEEP" if admissibility >= 1.0 else "DROP"
    reason = "KEEP" if decision == "KEEP" else "NO_ADMISSIBLE_TRACK"
    return CandidateEvaluation(
        candidate, hc, bin_count, atoms, state.atoms, occupied_length, mean_beta,
        admissibility, state.rho_over_h_sum / state.occupied, support_nodes,
        int(np.count_nonzero(ridge.strong[tube])), int(np.count_nonzero(ridge.feature[tube] & ~ridge.strong[tube])),
        float(np.quantile(tube_boundary, 0.5)) if len(tube_boundary) else 0.0,
        float(np.quantile(tube_boundary, 0.9)) if len(tube_boundary) else 0.0,
        decision, reason,
    )


def _interval(evaluation: CandidateEvaluation) -> tuple[float, float]:
    if not evaluation.selected_path:
        return 0.0, 0.0
    bins = [evaluation.atoms[index].bin_index for index in evaluation.selected_path]
    delta = evaluation.candidate.length / evaluation.bin_count
    return min(bins) * delta, (max(bins) + 1) * delta


def _duplicate(first: CandidateEvaluation, second: CandidateEvaluation, ridge: RidgeField) -> bool:
    angle = math.acos(float(np.clip(abs(np.dot(first.candidate.direction, second.candidate.direction)), -1.0, 1.0)))
    if angle > math.pi / 18.0:
        return False
    first_interval, second_interval = _interval(first), _interval(second)
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    shorter = min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0])
    ratio = 0.0 if shorter <= 0.0 else overlap / shorter
    if ratio < 0.60:
        return False
    union = first.support_nodes | second.support_nodes
    intersection = first.support_nodes & second.support_nodes
    denominator = float(np.sum(ridge.weight[list(union)])) if union else 0.0
    jaccard = 0.0 if denominator == 0.0 else float(np.sum(ridge.weight[list(intersection)])) / denominator
    first_centroids = np.asarray([first.atoms[index].centroid for index in first.selected_path])
    second_centroids = np.asarray([second.atoms[index].centroid for index in second.selected_path])
    if len(first_centroids) and len(second_centroids):
        first_dist = np.median(cKDTree(second_centroids).query(first_centroids, k=1, workers=1)[0])
        second_dist = np.median(cKDTree(first_centroids).query(second_centroids, k=1, workers=1)[0])
        track_distance = max(float(first_dist), float(second_dist))
    else:
        track_distance = math.inf
    return jaccard >= 0.45 or track_distance <= 1.25 * math.sqrt(first.hc * second.hc)


def _apply_duplicate_ownership(evaluations: list[CandidateEvaluation], ridge: RidgeField) -> None:
    kept = [index for index, value in enumerate(evaluations) if value.decision == "KEEP"]
    parent = {index: index for index in kept}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(kept):
        for second in kept[position + 1:]:
            if _duplicate(evaluations[first], evaluations[second], ridge):
                a, b = root(first), root(second)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in kept:
        groups[root(index)].append(index)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        owner = min(indices, key=lambda index: (
            -evaluations[index].admissibility,
            -evaluations[index].mean_beta,
            -evaluations[index].occupied_length,
            evaluations[index].mean_rho_over_h,
            evaluations[index].candidate.opaque_hash,
        ))
        owner_hash = evaluations[owner].candidate.opaque_hash
        for index in indices:
            if index == owner:
                continue
            evaluations[index].decision = "DROP"
            evaluations[index].terminal_reason = "DUPLICATE_SUPPRESS"
            evaluations[index].duplicate_owner = owner_hash


def frozen_configuration() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "k_all": list(K_ALL),
        "k_variation": list(K_VARIATION),
        "u_low_hex": _hex(U_LOW), "u_high_hex": _hex(U_HIGH),
        "a_low_hex": _hex(A_LOW), "a_high_hex": _hex(A_HIGH),
        "boundary_penalty_hex": _hex(BOUNDARY_PENALTY),
        "weak_hex": _hex(WEAK_THRESHOLD), "strong_hex": _hex(STRONG_THRESHOLD),
        "hysteresis_distance_hex": _hex(HYSTERESIS_DISTANCE),
        "track_strength_hex": _hex(TRACK_STRENGTH),
        "caps": {
            "points": MAX_POINTS, "triangles": MAX_TRIANGLES, "candidates": MAX_CANDIDATES,
            "neighbor_references": MAX_NEIGHBOR_REFERENCES, "graph_edges": MAX_GRAPH_EDGES,
            "feature_nodes_per_candidate": MAX_FEATURE_NODES_PER_CANDIDATE,
            "atoms_per_bin": MAX_ATOMS_PER_BIN, "atoms_per_candidate": MAX_ATOMS_PER_CANDIDATE,
            "candidate_node_visits": MAX_CANDIDATE_NODE_VISITS,
            "dag_edge_examinations": MAX_DAG_EDGE_EXAMINATIONS,
        },
    }


CONSTANT_SHA256 = _sha(canonical_bytes(frozen_configuration()))


def _implementation_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def _fixture_sha256() -> str:
    return _sha(FIXTURE_PATH.read_bytes()) if FIXTURE_PATH.exists() else _sha(b"V70_BRMVT_V1_FIXTURES_PENDING")


def _record(evaluation: CandidateEvaluation) -> dict[str, Any]:
    path_hash = _sha(canonical_bytes([evaluation.atoms[index].canonical_hash for index in evaluation.selected_path]))
    return {
        "candidate_hash": evaluation.candidate.opaque_hash,
        "length_over_hc_hex": _hex(evaluation.candidate.length / evaluation.hc),
        "hc_over_h0_hex": None,
        "tube_feature_node_count": int(sum(len(atom.node_indices) for atom in evaluation.atoms)),
        "bin_count": evaluation.bin_count,
        "atom_count": len(evaluation.atoms),
        "strong_node_count": evaluation.strong_count,
        "weak_bridge_node_count": evaluation.weak_bridge_count,
        "boundary_confidence_q50_hex": _hex(evaluation.boundary_q50),
        "boundary_confidence_q90_hex": _hex(evaluation.boundary_q90),
        "selected_occupied_length_over_L_hex": _hex(evaluation.occupied_length / evaluation.candidate.length),
        "selected_mean_beta_hex": _hex(evaluation.mean_beta),
        "admissibility_A_hex": _hex(evaluation.admissibility),
        "support_path_sha256": path_hash,
        "decision": evaluation.decision,
        "duplicate_owner_hash_or_null": evaluation.duplicate_owner,
        "terminal_reason": evaluation.terminal_reason,
    }


def _abort(baseline: dict[str, Any], rows: list[dict[str, Any]], status: str, reason: str, started: float) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    baseline.update({
        "schema_version": 1, "kind": "sherman_qc_f1_v70_brmvt_result", "profile": ARCHITECTURE,
        "predictions": rows, "candidate_count": len(rows), "source_only": True, "research_only": True,
        "brmvt_v1_audit": {"schema": SCHEMA, "status": status, "terminal_reason": "SCAN_ABORT", "abort_reason": reason},
        "brmvt_v1_operational_envelope": {"job_status": status, "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": _rss_bytes()},
        "brmvt_v1_modified_output_committed": False,
    })
    return baseline, ()


def apply_brmvt_v1(
    source_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    *,
    ordinary_nms: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
) -> tuple[dict[str, Any], tuple[CandidateEvaluation, ...]]:
    """Atomically apply BRMVT support/ownership, then one caller NMS."""

    started = time.perf_counter()
    baseline = deepcopy(dict(source_payload)) if isinstance(source_payload, Mapping) else {}
    source_rows = (source_payload.get("predictions") or ()) if isinstance(source_payload, Mapping) else source_payload
    rows = [deepcopy(dict(row)) for row in source_rows]
    if len(rows) > MAX_CANDIDATES:
        return _abort(baseline, rows, "RESOURCE_ABORT", "CANDIDATE_CAP", started)
    if vertices is None:
        return _abort(baseline, rows, "INPUT_ABORT", "MISSING_SOURCE_POINTS", started)
    try:
        candidates = [_candidate(row, ordinal) for ordinal, row in enumerate(rows)]
        if len({value.opaque_hash for value in candidates}) != len(candidates):
            raise BRMVTInputError("DUPLICATE_CANDIDATE_HASH")
        canonical = _canonical_source(vertices, faces)
        neighborhoods = _neighborhoods(canonical.points, started)
        ridge = _ridge_field(canonical, neighborhoods, started)
        evaluations = [_evaluate_candidate(candidate, canonical, neighborhoods, ridge, started) for candidate in sorted(candidates, key=lambda value: value.opaque_hash)]
        _apply_duplicate_ownership(evaluations, ridge)
        by_ordinal = {value.candidate.ordinal: value for value in evaluations}
        survivors = [rows[index] for index in range(len(rows)) if by_ordinal[index].decision == "KEEP"]
        output_rows: Any = survivors
        nms_audit = None
        if ordinary_nms is not None:
            nms_result = ordinary_nms(survivors)
            if isinstance(nms_result, tuple) and len(nms_result) == 2:
                output_rows, nms_audit = nms_result
            else:
                output_rows = nms_result
        records = []
        for evaluation in sorted(evaluations, key=lambda value: value.candidate.opaque_hash):
            record = _record(evaluation)
            record["hc_over_h0_hex"] = _hex(evaluation.hc / neighborhoods.h0)
            records.append(record)
        audit = {
            "schema": SCHEMA,
            "constants_sha256": CONSTANT_SHA256,
            "implementation_sha256": _implementation_sha256(),
            "fixture_sha256": _fixture_sha256(),
            "source_geometry_sha256": canonical.source_sha256,
            "input_counts": {"points": len(canonical.points), "triangles": len(canonical.faces), "candidates": len(candidates)},
            "global": {
                "point_count": len(canonical.points), "triangle_count": len(canonical.faces), "candidate_count": len(candidates),
                "h0_hex": _hex(neighborhoods.h0), "neighbor_reference_count": neighborhoods.reference_count,
                "source_graph_edge_count": len(ridge.graph_edges),
            },
            "candidates": records,
            "output_candidate_hashes": sorted(by_ordinal[index].candidate.opaque_hash for index in range(len(rows)) if by_ordinal[index].decision == "KEEP"),
        }
        audit["canonical_output_sha256"] = _sha(canonical_bytes(audit))
        encoded = canonical_bytes(audit)
        if len(encoded) > MAX_AUDIT_BYTES:
            raise BRMVTResourceAbort("AUDIT_CAP")
        _check_time(started)
        baseline.update({
            "schema_version": 1, "kind": "sherman_qc_f1_v70_brmvt_result", "profile": ARCHITECTURE,
            "predictions": [deepcopy(dict(row)) for row in output_rows], "candidate_count": len(output_rows),
            "brmvt_v1_audit": audit,
            "brmvt_v1_operational_envelope": {
                "job_status": "OK", "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": _rss_bytes(),
                "ordinary_nms_call_count": 1 if ordinary_nms is not None else 0,
                "ordinary_nms_audit_present": nms_audit is not None,
            },
            "brmvt_v1_modified_output_committed": True,
            "source_only": True, "research_only": True, "expected_count_used": False,
            "labels_used": False, "cad_used": False, "part_identity_used": False,
            "filename_identity_used": False, "candidate_score_used": False,
            "route_provenance_used": False, "v52_result_used": False,
        })
        return baseline, tuple(evaluations)
    except (BRMVTInputError, BRMVTResourceAbort, MemoryError, OverflowError, FloatingPointError) as error:
        status = "RESOURCE_ABORT" if isinstance(error, (BRMVTResourceAbort, MemoryError)) else "INPUT_OR_NUMERIC_ABORT"
        return _abort(baseline, rows, status, str(error), started)


__all__ = [
    "ARCHITECTURE", "SHORT_NAME", "SCHEMA", "BRMVTInputError", "BRMVTResourceAbort",
    "CandidateEvaluation", "CONSTANT_SHA256", "apply_brmvt_v1", "canonical_bytes",
    "frozen_configuration", "local_percentile",
]
