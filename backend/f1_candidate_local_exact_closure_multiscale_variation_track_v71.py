"""V71 candidate-local exact-closure multiscale variation track (CLEC_MVT_V1).

The module never calls the V70 executable or runner.  It attests the frozen
V70 implementation and reuses its exact mathematical primitives by direct
function identity, while replacing the global field/hysteresis domains with
candidate-local dependency closures and conservative domain-exit abstention.
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

import backend.f1_boundary_rejecting_multiscale_variation_track_composition_v70 as parent


ARCHITECTURE = "V71_CANDIDATE_LOCAL_EXACT_CLOSURE_MULTISCALE_VARIATION_TRACK_V1"
SHORT_NAME = "CLEC_MVT_V1"
SCHEMA = "sherman_qc_f1_v71_clec_mvt_v1_audit"
V70_PARENT_LOCK_SHA256 = "f22e1a3c77d58b8683ea5411c422abd07103082f39d2e9d39835b4e24229e7b4"
V70_IMPLEMENTATION_SHA256 = "92f5cb62ec1bb250254d06aac644985668d304c8623ac42f044872de2d162e81"
RULING_PATH = Path(__file__).resolve().parents[1] / "docs/research/2026-08-16_gpt56_pro_post_v70_sparse_field_ruling_verbatim.md"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests/test_f1_candidate_local_exact_closure_multiscale_variation_track_v71.py"

MAX_POINTS = 200_000
MAX_TRIANGLES = 350_000
MAX_CANDIDATES = 128
MAX_Q0 = 32_768
MAX_Q1 = 65_536
MAX_DOMAIN_MEMBERSHIP = 262_144
MAX_GRAPH_EDGE_VISITS = 4_194_304
MAX_AUDIT_BYTES = 8 * 1024**2
MAX_SCRATCH_BYTES = 4 * 1024**3
MAX_RSS_BYTES = 16 * 1024**3
FIELD_WALL_SECONDS = 180.0
SCAN_WALL_SECONDS = 240.0


class CLECInputError(RuntimeError):
    pass


class CLECResourceAbort(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_deadline(started: float, cap: float, reason: str) -> None:
    if time.perf_counter() - started > cap:
        raise CLECResourceAbort(reason)
    if _rss_bytes() >= MAX_RSS_BYTES:
        raise CLECResourceAbort("PROCESS_RSS_CAP")


def _ids_hash(values: Sequence[int] | np.ndarray) -> str:
    array = np.asarray(sorted(int(value) for value in values), dtype=">i8")
    return _sha(array.tobytes())


def _attest_parent() -> None:
    observed = _sha(Path(parent.__file__).read_bytes())
    if observed != V70_IMPLEMENTATION_SHA256:
        raise CLECInputError("V70_PARENT_HASH_MISMATCH")


@dataclass
class SourceGraph:
    source: parent.CanonicalSource
    tree: cKDTree
    h: np.ndarray
    h0: float
    offsets: np.ndarray
    neighbors: np.ndarray
    lengths: np.ndarray
    edge_count: int


def _build_source_graph(vertices: np.ndarray, faces: np.ndarray | None, started: float) -> SourceGraph:
    source = parent._canonical_source(vertices, faces)
    if len(source.points) < 1281:
        raise CLECInputError("FEWER_THAN_1280_USABLE_NEIGHBORS")
    tree = cKDTree(source.points, balanced_tree=True, compact_nodes=True)
    count = len(source.points)
    distances = np.empty((count, 13), dtype=np.float64)
    indices = np.empty((count, 13), dtype=np.int32)
    for begin in range(0, count, 4096):
        _check_deadline(started, SCAN_WALL_SECONDS, "SCAN_WALL_CAP")
        end = min(count, begin + 4096)
        current_dist, current_idx = tree.query(source.points[begin:end], k=13, workers=1)
        distances[begin:end] = current_dist
        indices[begin:end] = current_idx
    h = np.median(distances[:, 1:7], axis=1)
    h0 = float(np.median(h))
    if not math.isfinite(h0) or h0 <= 0.0:
        raise CLECInputError("NONPOSITIVE_SOURCE_SCALE")
    edge_codes = []
    for first in range(count):
        for second in indices[first, 1:13]:
            a, b = sorted((first, int(second)))
            if a != b:
                edge_codes.append(a * count + b)
    if len(source.mesh_edges):
        edge_codes.extend(int(min(a, b)) * count + int(max(a, b)) for a, b in source.mesh_edges)
    codes = np.unique(np.asarray(edge_codes, dtype=np.int64))
    first = codes // count
    second = codes % count
    edge_lengths = np.linalg.norm(source.points[first] - source.points[second], axis=1)
    valid = (edge_lengths > 0.0) & (edge_lengths <= 3.0 * np.maximum(h[first], h[second]))
    first, second, edge_lengths = first[valid], second[valid], edge_lengths[valid]
    degree = np.bincount(np.concatenate((first, second)), minlength=count)
    offsets = np.concatenate(([0], np.cumsum(degree, dtype=np.int64)))
    adjacency = np.empty(int(offsets[-1]), dtype=np.int32)
    weights = np.empty(int(offsets[-1]), dtype=np.float64)
    cursor = offsets[:-1].copy()
    for a, b, length in zip(first, second, edge_lengths):
        position = cursor[a]; adjacency[position] = b; weights[position] = length; cursor[a] += 1
        position = cursor[b]; adjacency[position] = a; weights[position] = length; cursor[b] += 1
    for vertex in range(count):
        low, high = int(offsets[vertex]), int(offsets[vertex + 1])
        order = np.argsort(adjacency[low:high], kind="stable")
        adjacency[low:high] = adjacency[low:high][order]
        weights[low:high] = weights[low:high][order]
    return SourceGraph(source, tree, h, h0, offsets, adjacency, weights, len(first))


def _adjacent(graph: SourceGraph, vertex: int) -> tuple[np.ndarray, np.ndarray]:
    low, high = int(graph.offsets[vertex]), int(graph.offsets[vertex + 1])
    return graph.neighbors[low:high], graph.lengths[low:high]


@dataclass
class CandidateDomain:
    candidate: parent.Candidate
    local_scale: float
    hc: float
    core: np.ndarray
    interior: np.ndarray
    outer: np.ndarray
    core_hash: str
    interior_hash: str
    outer_hash: str
    degenerate: bool = False


def _candidate_hc(candidate: parent.Candidate, graph: SourceGraph) -> float:
    positions = candidate.start + ((np.arange(17, dtype=np.float64) + 0.5) / 17.0)[:, None] * (candidate.end - candidate.start)
    _distance, neighbors = graph.tree.query(positions, k=8, workers=1)
    value = float(np.median(np.median(graph.h[np.asarray(neighbors, dtype=np.int64)], axis=1)))
    return float(np.clip(value, 0.5 * graph.h0, 2.0 * graph.h0))


def _track_core(candidate: parent.Candidate, graph: SourceGraph, hc: float) -> np.ndarray:
    midpoint = 0.5 * (candidate.start + candidate.end)
    conservative_radius = math.hypot(0.5 * candidate.length + 3.0 * hc, 6.0 * hc)
    possible = np.asarray(graph.tree.query_ball_point(midpoint, np.nextafter(conservative_radius, math.inf)), dtype=np.int64)
    points = graph.source.points[possible]
    along = (points - candidate.start) @ candidate.direction
    perpendicular = points - candidate.start - along[:, None] * candidate.direction
    eligible = (along >= -3.0 * hc) & (along <= candidate.length + 3.0 * hc) & (np.linalg.norm(perpendicular, axis=1) <= 6.0 * hc)
    return np.asarray(sorted(possible[eligible].tolist()), dtype=np.int32)


def _graph_domain(graph: SourceGraph, seeds: np.ndarray, radius: float) -> tuple[np.ndarray, int]:
    if len(seeds) == 0:
        return np.empty(0, dtype=np.int32), 0
    distance: dict[int, float] = {int(seed): 0.0 for seed in seeds}
    heap = [(0.0, int(seed)) for seed in seeds]
    heapq.heapify(heap)
    visits = 0
    while heap:
        current, vertex = heapq.heappop(heap)
        if current != distance.get(vertex) or current > radius:
            continue
        adjacent, lengths = _adjacent(graph, vertex)
        for neighbor, length in zip(adjacent, lengths):
            visits += 1
            if visits > MAX_GRAPH_EDGE_VISITS:
                raise CLECResourceAbort("GRAPH_EDGE_VISIT_CAP")
            proposal = current + float(length)
            key = int(neighbor)
            if proposal <= radius and proposal < distance.get(key, math.inf):
                distance[key] = proposal
                heapq.heappush(heap, (proposal, key))
    return np.asarray(sorted(distance), dtype=np.int32), visits


def _domain(candidate: parent.Candidate, graph: SourceGraph) -> tuple[CandidateDomain, int]:
    station = candidate.start + (np.arange(9, dtype=np.float64) / 8.0)[:, None] * (candidate.end - candidate.start)
    distances, _indices = graph.tree.query(station, k=64, workers=1)
    local_scale = float(np.sort(np.asarray(distances)[:, 63])[4])
    hc = _candidate_hc(candidate, graph)
    core = _track_core(candidate, graph, hc)
    if len(core) == 0 or not math.isfinite(local_scale) or local_scale <= 0.0:
        empty = np.empty(0, dtype=np.int32)
        return CandidateDomain(candidate, local_scale, hc, core, empty, empty, _ids_hash(core), _ids_hash(empty), _ids_hash(empty), True), 0
    interior, first_visits = _graph_domain(graph, core, local_scale)
    outer, second_visits = _graph_domain(graph, core, 2.0 * local_scale)
    return CandidateDomain(candidate, local_scale, hc, core, interior, outer, _ids_hash(core), _ids_hash(interior), _ids_hash(outer)), first_visits + second_visits


@dataclass
class SparseField:
    q0: np.ndarray
    q1: np.ndarray
    q0_hash: str
    q1_hash: str
    sigma: dict[int, np.ndarray]
    weight: np.ndarray
    strong: np.ndarray
    weak: np.ndarray
    boundary: np.ndarray
    neighbor_references: int


def _query_exact(tree: cKDTree, points: np.ndarray, vertex_ids: np.ndarray, k: int) -> np.ndarray:
    if len(points) <= k:
        raise CLECInputError("NEIGHBOR_QUERY_FAILURE")
    # Request one candidate beyond the cutoff.  If that candidate is tied with
    # the kth neighbour, enumerate the complete exact-distance shell before
    # applying the canonical-ID tiebreak.  This avoids inheriting cKDTree's
    # unspecified choice among an equal-distance shell.
    _dist, indices = tree.query(points[vertex_ids], k=min(len(points), k + 2), workers=1)
    output = np.empty((len(vertex_ids), k), dtype=np.int32)
    for row, vertex in enumerate(vertex_ids):
        values = [int(value) for value in indices[row] if int(value) != int(vertex)]
        if len(values) < k:
            raise CLECInputError("NEIGHBOR_QUERY_FAILURE")
        keyed = sorted(
            ((float(np.dot(points[value] - points[vertex], points[value] - points[vertex])), value) for value in values),
            key=lambda item: (item[0], item[1]),
        )
        if len(keyed) > k and keyed[k - 1][0] == keyed[k][0]:
            radius = math.nextafter(math.sqrt(keyed[k - 1][0]), math.inf)
            shell = [int(value) for value in tree.query_ball_point(points[vertex], radius) if int(value) != int(vertex)]
            keyed = sorted(
                ((float(np.dot(points[value] - points[vertex], points[value] - points[vertex])), value) for value in shell),
                key=lambda item: (item[0], item[1]),
            )
        output[row] = np.asarray([value for _distance, value in keyed[:k]], dtype=np.int32)
    return output


def _sv_batch(points: np.ndarray, vertex_ids: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    output = np.zeros((len(vertex_ids), len(parent.K_VARIATION)), dtype=np.float64)
    for scale_index, cardinality in enumerate(parent.K_VARIATION):
        coordinates = points[neighbors[:, :cardinality]]
        mean = np.mean(coordinates, axis=1, dtype=np.float64)
        centered = coordinates - mean[:, None, :]
        covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / float(cardinality)
        values = np.linalg.eigvalsh(covariance)
        trace = np.trace(covariance, axis1=1, axis2=2)
        tolerance = np.ldexp(trace, -40)
        if np.any(values[:, 0] < -tolerance):
            raise CLECInputError("EIGENVALUE_ABORT")
        values = np.maximum(values, 0.0)
        output[:, scale_index] = np.divide(values[:, 0], trace, out=np.zeros_like(trace), where=trace != 0.0)
    return output


def _build_sparse_field(graph: SourceGraph, domains: Sequence[CandidateDomain], started: float) -> SparseField:
    field_started = time.perf_counter()
    q0 = np.asarray(sorted({int(value) for domain in domains for value in domain.outer}), dtype=np.int32)
    if len(q0) > MAX_Q0:
        raise CLECResourceAbort("Q0_CAP")
    q0_neighbors: dict[int, np.ndarray] = {}
    q1_set = set(int(value) for value in q0)
    neighbor_references = 0
    for begin in range(0, len(q0), 256):
        _check_deadline(field_started, FIELD_WALL_SECONDS, "FIELD_WALL_CAP")
        block = q0[begin:begin + 256]
        neighbors = _query_exact(graph.tree, graph.source.points, block, 1280)
        for vertex, row in zip(block, neighbors):
            q0_neighbors[int(vertex)] = row.copy()
            q1_set.update(int(value) for value in row)
        neighbor_references += neighbors.size
        if len(q1_set) > MAX_Q1:
            raise CLECResourceAbort("Q1_CAP")
    q1 = np.asarray(sorted(q1_set), dtype=np.int32)
    sigma: dict[int, np.ndarray] = {}
    q0_set = set(int(value) for value in q0)
    for begin in range(0, len(q1), 256):
        _check_deadline(field_started, FIELD_WALL_SECONDS, "FIELD_WALL_CAP")
        block = q1[begin:begin + 256]
        neighbor_rows = np.empty((len(block), 1024), dtype=np.int32)
        missing_positions = []
        missing_ids = []
        for position, vertex in enumerate(block):
            cached = q0_neighbors.get(int(vertex))
            if cached is None:
                missing_positions.append(position); missing_ids.append(int(vertex))
            else:
                neighbor_rows[position] = cached[:1024]
        if missing_ids:
            queried = _query_exact(graph.tree, graph.source.points, np.asarray(missing_ids, dtype=np.int32), 1024)
            neighbor_references += queried.size
            for position, row in zip(missing_positions, queried):
                neighbor_rows[position] = row
        values = _sv_batch(graph.source.points, block, neighbor_rows)
        for vertex, row in zip(block, values):
            sigma[int(vertex)] = row.copy()
    q0_position = {int(vertex): index for index, vertex in enumerate(q0)}
    weight = np.zeros(len(q0), dtype=np.float64)
    strong = np.zeros(len(q0), dtype=bool)
    weak = np.zeros(len(q0), dtype=bool)
    boundary = np.zeros(len(q0), dtype=np.float64)
    for position, vertex in enumerate(q0):
        own = sigma[int(vertex)]
        peers_index = q0_neighbors[int(vertex)]
        peers = np.asarray([sigma[int(value)] for value in peers_index], dtype=np.float64)
        u = np.zeros(6, dtype=np.float64)
        v = np.zeros(6, dtype=np.float64)
        for scale in range(6):
            rank = parent.local_percentile(float(own[scale]), peers[:, scale])
            u[scale] = np.clip((rank - parent.U_LOW) / (parent.U_HIGH - parent.U_LOW), 0.0, 1.0)
            median = float(np.median(peers[:, scale]))
            contrast = max(0.0, float(own[scale] - median) / max(median, math.ldexp(1.0, -20)))
            v[scale] = np.clip((contrast - parent.A_LOW) / (parent.A_HIGH - parent.A_LOW), 0.0, 1.0)
        ordered = np.sort(u)
        saliency = math.sqrt(float(ordered[-1] * ordered[-2])) * math.sqrt(float(np.max(v)))
        neighbors24 = peers_index[:24]
        covariance = parent._covariance(graph.source.points[neighbors24])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        angular = 0.0
        if eigenvalues[1] == 0.0:
            angular = 1.0
        elif eigenvalues[0] / eigenvalues[1] <= 0.35:
            basis = eigenvectors[:, 1:3]
            projected = (graph.source.points[neighbors24] - graph.source.points[int(vertex)]) @ basis
            angles = np.sort(np.arctan2(projected[:, 1], projected[:, 0]))
            gap = float(np.max(np.diff(np.concatenate((angles, [angles[0] + 2.0 * math.pi])))))
            angular = float(np.clip((gap - parent.BOUNDARY_START) / parent.BOUNDARY_SPAN, 0.0, 1.0))
        boundary[position] = max(angular, float(graph.source.mesh_boundary[int(vertex)]))
        weight[position] = saliency * (1.0 - parent.BOUNDARY_PENALTY * boundary[position])
        strong[position] = weight[position] >= parent.STRONG_THRESHOLD
        weak[position] = weight[position] >= parent.WEAK_THRESHOLD
    _check_deadline(field_started, FIELD_WALL_SECONDS, "FIELD_WALL_CAP")
    _check_deadline(started, SCAN_WALL_SECONDS, "SCAN_WALL_CAP")
    return SparseField(q0, q1, _ids_hash(q0), _ids_hash(q1), sigma, weight, strong, weak, boundary, neighbor_references)


@dataclass
class V71Evaluation:
    domain: CandidateDomain
    parent_evaluation: parent.CandidateEvaluation | None
    decision: str
    reason: str
    strong_count: int
    weak_count: int
    active_component_count: int
    unresolved_component_count: int
    owner_hash: str | None = None


def _candidate_field(graph: SourceGraph, sparse: SparseField, domain: CandidateDomain) -> tuple[parent.RidgeField, bool, int, int, int, int]:
    count = len(graph.source.points)
    q0_position = {int(vertex): index for index, vertex in enumerate(sparse.q0)}
    strong_global = np.zeros(count, dtype=bool)
    weak_global = np.zeros(count, dtype=bool)
    weight_global = np.zeros(count, dtype=np.float64)
    boundary_global = np.zeros(count, dtype=np.float64)
    for vertex in domain.outer:
        position = q0_position[int(vertex)]
        strong_global[int(vertex)] = sparse.strong[position]
        weak_global[int(vertex)] = sparse.weak[position]
        weight_global[int(vertex)] = sparse.weight[position]
        boundary_global[int(vertex)] = sparse.boundary[position]
    outer_set = set(int(value) for value in domain.outer)
    interior_set = set(int(value) for value in domain.interior)
    core_set = set(int(value) for value in domain.core)
    eligible = set(int(value) for value in domain.outer if strong_global[int(value)] or weak_global[int(value)])
    components = []
    unseen = set(eligible)
    while unseen:
        seed = min(unseen); unseen.remove(seed); stack = [seed]; component = []
        while stack:
            vertex = stack.pop(); component.append(vertex)
            adjacent, _lengths = _adjacent(graph, vertex)
            for neighbor in adjacent:
                value = int(neighbor)
                if value in unseen:
                    unseen.remove(value); stack.append(value)
        components.append(component)
    feature = np.zeros(count, dtype=bool)
    unresolved_intersects_core = False
    active_count = 0
    unresolved_count = 0
    for component in components:
        active = any(strong_global[value] for value in component)
        if active:
            active_count += 1
            # Unchanged V70 normalized-distance hysteresis inside the candidate domain.
            distances = {value: 0.0 for value in component if strong_global[value]}
            heap = [(0.0, value) for value in distances]; heapq.heapify(heap)
            component_set = set(component)
            while heap:
                current, vertex = heapq.heappop(heap)
                if current != distances.get(vertex) or current > parent.HYSTERESIS_DISTANCE:
                    continue
                feature[vertex] = True
                adjacent, lengths = _adjacent(graph, vertex)
                for neighbor, length in zip(adjacent, lengths):
                    value = int(neighbor)
                    if value not in component_set:
                        continue
                    proposal = current + float(length) / max(graph.h[vertex], graph.h[value])
                    if proposal <= parent.HYSTERESIS_DISTANCE and proposal < distances.get(value, math.inf):
                        distances[value] = proposal; heapq.heappush(heap, (proposal, value))
        else:
            touches_shell = any(value not in interior_set for value in component)
            has_outside_neighbor = False
            for vertex in component:
                adjacent, _lengths = _adjacent(graph, vertex)
                if any(int(value) not in outer_set for value in adjacent):
                    has_outside_neighbor = True; break
            unresolved = touches_shell or has_outside_neighbor
            if unresolved:
                unresolved_count += 1
                if any(value in core_set for value in component):
                    unresolved_intersects_core = True
    ridge = parent.RidgeField(
        sigma=np.empty((0, 6)), u=np.empty((0, 6)), v=np.empty((0, 6)),
        persistence=np.empty(0), contrast=np.empty(0), saliency=np.empty(0),
        boundary=boundary_global, weight=weight_global, strong=strong_global,
        weak=weak_global, feature=feature, graph_edges=np.empty((0, 2), dtype=np.int32),
    )
    return ridge, unresolved_intersects_core, int(np.count_nonzero(strong_global[list(outer_set)])), int(np.count_nonzero(weak_global[list(outer_set)])), active_count, unresolved_count


def _evaluate(domain: CandidateDomain, graph: SourceGraph, sparse: SparseField, started: float) -> V71Evaluation:
    if domain.degenerate:
        return V71Evaluation(domain, None, "ABSTAIN_DEGENERATE_LOCAL_DOMAIN", "DEGENERATE_LOCAL_DOMAIN", 0, 0, 0, 0)
    ridge, unresolved, strong_count, weak_count, active_count, unresolved_count = _candidate_field(graph, sparse, domain)
    fake_neighborhoods = parent.Neighborhoods([], graph.h, graph.h0, 0)
    evaluation = parent._evaluate_candidate(domain.candidate, graph.source, fake_neighborhoods, ridge, started)
    if evaluation.decision == "KEEP":
        decision, reason = "KEEP_TRACK", "UNCHANGED_TRACK_PASSES"
    elif unresolved:
        decision, reason = "KEEP_ABSTAIN_DOMAIN_EXIT", "UNRESOLVED_COMPONENT_INTERSECTS_CORE"
    else:
        decision, reason = "VETO_COMPLETE_NO_TRACK", "HYSTERESIS_CLOSED_NO_TRACK"
    return V71Evaluation(domain, evaluation, decision, reason, strong_count, weak_count, active_count, unresolved_count)


def frozen_configuration() -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "v70_parent_lock_sha256": V70_PARENT_LOCK_SHA256,
        "v70_implementation_sha256": V70_IMPLEMENTATION_SHA256,
        "parent_constants_sha256": parent.CONSTANT_SHA256,
        "changed_numeric_constants": [], "changed_field_formulas": [], "changed_comparators": [],
        "changed_track_predicates": [], "changed_ownership_predicates": [], "changed_nms_rules": [],
        "caps": {"q0": MAX_Q0, "q1": MAX_Q1, "membership": MAX_DOMAIN_MEMBERSHIP,
                 "graph_edge_visits": MAX_GRAPH_EDGE_VISITS, "field_wall_seconds": FIELD_WALL_SECONDS,
                 "scan_wall_seconds": SCAN_WALL_SECONDS},
    }


CONSTANT_SHA256 = _sha(canonical_bytes(frozen_configuration()))


def _implementation_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def _abort(baseline: dict[str, Any], rows: list[dict[str, Any]], reason: str, started: float) -> tuple[dict[str, Any], tuple[V71Evaluation, ...]]:
    baseline.update({
        "schema_version": 1, "kind": "sherman_qc_f1_v71_clec_mvt_result", "profile": ARCHITECTURE,
        "predictions": rows, "candidate_count": len(rows), "source_only": True, "research_only": True,
        "clec_mvt_v1_audit": {"architecture": "V71_CLEC_MVT_V1", "status": "SCAN_ATOMIC_ABORT", "abort_reason": reason,
                              "modified_output_count": 0, "ordinary_nms_invocation_count": 0},
        "clec_mvt_v1_operational_envelope": {"wall_seconds": time.perf_counter() - started, "peak_rss_bytes": _rss_bytes()},
        "clec_mvt_v1_modified_output_committed": False,
    })
    return baseline, ()


def apply_clec_mvt_v1(
    source_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    *, ordinary_nms: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
) -> tuple[dict[str, Any], tuple[V71Evaluation, ...]]:
    started = time.perf_counter()
    baseline = deepcopy(dict(source_payload)) if isinstance(source_payload, Mapping) else {}
    source_rows = (source_payload.get("predictions") or ()) if isinstance(source_payload, Mapping) else source_payload
    rows = [deepcopy(dict(row)) for row in source_rows]
    try:
        _attest_parent()
        if vertices is None or len(rows) > MAX_CANDIDATES:
            raise CLECInputError("MISSING_SOURCE_OR_CANDIDATE_CAP")
        # Whitelist serialization physically excludes scores/routes/identity.
        candidates = [parent._candidate({"candidate_opaque_hash": _sha(str(row.get("route_hash") or row.get("bend_id")).encode()),
                                         "line_start": row["line_start"], "line_end": row["line_end"]}, ordinal)
                      for ordinal, row in enumerate(rows)]
        if len({candidate.opaque_hash for candidate in candidates}) != len(candidates):
            raise CLECInputError("DUPLICATE_CANDIDATE_HASH")
        graph = _build_source_graph(np.asarray(vertices), None if faces is None else np.asarray(faces), started)
        domains = []
        graph_visits = 0
        for candidate in sorted(candidates, key=lambda value: value.opaque_hash):
            domain, visits = _domain(candidate, graph); domains.append(domain); graph_visits += visits
            if graph_visits > MAX_GRAPH_EDGE_VISITS:
                raise CLECResourceAbort("GRAPH_EDGE_VISIT_CAP")
        membership_sum = sum(len(domain.outer) for domain in domains)
        if membership_sum > MAX_DOMAIN_MEMBERSHIP:
            raise CLECResourceAbort("DOMAIN_MEMBERSHIP_CAP")
        sparse = _build_sparse_field(graph, domains, started)
        evaluations = [_evaluate(domain, graph, sparse, started) for domain in domains]
        # Exact V70 duplicate ownership applies to all retained candidates.
        parent_evaluations = []
        for value in evaluations:
            if value.parent_evaluation is not None and value.decision in {"KEEP_TRACK", "KEEP_ABSTAIN_DOMAIN_EXIT", "ABSTAIN_DEGENERATE_LOCAL_DOMAIN"}:
                value.parent_evaluation.decision = "KEEP"; value.parent_evaluation.terminal_reason = "KEEP"
                parent_evaluations.append(value.parent_evaluation)
        if parent_evaluations:
            # Sparse union field weights are sufficient because selected support lies in Q0.
            count = len(graph.source.points); weights = np.zeros(count); qpos = {int(v): i for i, v in enumerate(sparse.q0)}
            for vertex, position in qpos.items(): weights[vertex] = sparse.weight[position]
            ridge = parent.RidgeField(np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),np.empty(0),np.empty(0),np.empty(0),np.zeros(count),weights,np.zeros(count,bool),np.zeros(count,bool),np.zeros(count,bool),np.empty((0,2),np.int32))
            parent._apply_duplicate_ownership(parent_evaluations, ridge)
            by_hash = {value.candidate.opaque_hash: value for value in parent_evaluations}
            for value in evaluations:
                linked = by_hash.get(value.domain.candidate.opaque_hash)
                if linked is not None and linked.terminal_reason == "DUPLICATE_SUPPRESS":
                    value.decision = "SUPPRESS_DUPLICATE"; value.reason = "GEOMETRY_DUPLICATE_OWNER"; value.owner_hash = linked.duplicate_owner
        kept_hashes = {value.domain.candidate.opaque_hash for value in evaluations if value.decision in {"KEEP_TRACK","KEEP_ABSTAIN_DOMAIN_EXIT","ABSTAIN_DEGENERATE_LOCAL_DOMAIN"}}
        survivors = [row for row, candidate in zip(rows, candidates) if candidate.opaque_hash in kept_hashes]
        output_rows: Any = survivors
        if ordinary_nms is not None:
            result = ordinary_nms(survivors); output_rows = result[0] if isinstance(result, tuple) else result
        records = []
        for value in evaluations:
            track = value.parent_evaluation
            records.append({
                "candidate_hash": value.domain.candidate.opaque_hash,
                "local_scale_r": float(value.domain.local_scale).hex(),
                "track_core_count": len(value.domain.core), "interior_halo_count": len(value.domain.interior), "outer_domain_count": len(value.domain.outer),
                "core_sha256": value.domain.core_hash, "interior_sha256": value.domain.interior_hash, "outer_sha256": value.domain.outer_hash,
                "strong_count": value.strong_count, "weak_count": value.weak_count,
                "active_component_count": value.active_component_count, "externally_unresolved_component_count": value.unresolved_component_count,
                "track_summary": {} if track is None else {"admissibility_A_hex": float(track.admissibility).hex(), "occupied_length_hex": float(track.occupied_length).hex(), "mean_beta_hex": float(track.mean_beta).hex()},
                "decision": value.decision, "reason": value.reason, "owner_candidate_hash": value.owner_hash,
            })
        audit = {
            "architecture": "V71_CLEC_MVT_V1", "source_sha256": graph.source.source_sha256,
            "candidate_stream_sha256": _sha(canonical_bytes(sorted(candidate.opaque_hash for candidate in candidates))),
            "v70_parent_lock_sha256": V70_PARENT_LOCK_SHA256, "v71_lock_sha256": CONSTANT_SHA256,
            "vertex_count": len(graph.source.points), "triangle_count": len(graph.source.faces), "candidate_count": len(candidates),
            "q0_count": len(sparse.q0), "q1_count": len(sparse.q1), "q0_sha256": sparse.q0_hash, "q1_sha256": sparse.q1_hash,
            "domain_membership_sum": membership_sum, "graph_edge_visit_count": graph_visits,
            "neighbor_reference_count": sparse.neighbor_references, "status": "COMPLETE", "abort_reason": None,
            "candidates": records,
        }
        audit["output_sha256"] = _sha(canonical_bytes({"audit": audit, "kept": sorted(kept_hashes)}))
        if len(canonical_bytes(audit)) > MAX_AUDIT_BYTES:
            raise CLECResourceAbort("AUDIT_CAP")
        _check_deadline(started, SCAN_WALL_SECONDS, "SCAN_WALL_CAP")
        baseline.update({
            "schema_version": 1, "kind": "sherman_qc_f1_v71_clec_mvt_result", "profile": ARCHITECTURE,
            "predictions": [deepcopy(dict(row)) for row in output_rows], "candidate_count": len(output_rows),
            "clec_mvt_v1_audit": audit,
            "clec_mvt_v1_operational_envelope": {"wall_seconds": time.perf_counter()-started, "peak_rss_bytes": _rss_bytes(), "ordinary_nms_invocation_count": 1 if ordinary_nms is not None else 0},
            "clec_mvt_v1_modified_output_committed": True, "source_only": True, "research_only": True,
            "expected_count_used": False, "labels_used": False, "cad_used": False, "part_identity_used": False,
            "filename_identity_used": False, "candidate_score_used": False, "route_provenance_used": False,
        })
        return baseline, tuple(evaluations)
    except (CLECInputError, CLECResourceAbort, parent.BRMVTInputError, parent.BRMVTResourceAbort, MemoryError, OverflowError, FloatingPointError, KeyError, ValueError) as error:
        return _abort(baseline, rows, str(error), started)


__all__ = ["ARCHITECTURE", "SHORT_NAME", "CONSTANT_SHA256", "apply_clec_mvt_v1", "frozen_configuration"]
