"""Deterministic source-only GCDF-V1 scientific core.

The module never accepts target, gold, CAD, labels, benchmark results, or prior
candidate artifacts.  Scientific outputs contain identifiers, hashes, scalar
censuses, and compact axes only; full source geometry and paths are ephemeral.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from research.f1_ciip_v1 import partition, primary_parser
from .config import ATTRITION_TOKENS, CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json


class GCDFStop(RuntimeError):
    def __init__(self, token: str, message: str) -> None:
        super().__init__(message)
        self.token = token


@dataclass(slots=True)
class Sample:
    xyz: np.ndarray
    point_ids: np.ndarray
    multiplicities: np.ndarray
    source_ordinals: np.ndarray
    ordered_source_ordinal_sha256: str
    sampled_coordinate_sha256: str
    sampler_executable_sha256: str
    sampler_configuration_sha256: str


@dataclass(slots=True)
class CanonicalNodes:
    integer_xyz: np.ndarray
    metric_xyz: np.ndarray
    multiplicities: np.ndarray
    minimum_source_ordinals: np.ndarray
    q: float
    origin: np.ndarray
    duplicate_count: int
    table_sha256: str


@dataclass(slots=True)
class SurfaceGraph:
    neighbors32: np.ndarray
    neighbor_sq32: np.ndarray
    neighbor_w32: np.ndarray
    r8: np.ndarray
    reliable: np.ndarray
    normals: np.ndarray
    eigenvalues_desc: np.ndarray
    edges_left: np.ndarray
    edges_right: np.ndarray
    edge_weights: np.ndarray
    component_labels: np.ndarray
    active_components: np.ndarray
    active_mask: np.ndarray
    component_medians: dict[int, int]
    gate_census: dict[str, int]
    canonical_node_sha256: str
    edge_set_sha256: str
    component_table_sha256: str
    semantic_sha256: str


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_arrays(tag: bytes, arrays: Iterable[np.ndarray | bytes]) -> str:
    digest = hashlib.sha256(tag)
    for value in arrays:
        raw = value if isinstance(value, bytes) else np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def ppm(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise ValueError("ppm domain violation")
    return (1_000_000 * int(a)) // int(b)


def lower_median(values: Sequence[int] | np.ndarray) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.int64), kind="stable")
    if len(ordered) == 0:
        raise ValueError("median of empty sequence")
    return int(ordered[(len(ordered) - 1) // 2])


def ceil_sqrt_q16(squared_integer_distance: int) -> int:
    if squared_integer_distance <= 0:
        raise ValueError("edge distance must be positive")
    n = int(squared_integer_distance) << 32
    root = math.isqrt(n)
    return root if root * root == n else root + 1


def _q16_weights(squared: np.ndarray) -> np.ndarray:
    flat = np.asarray(squared, dtype=np.uint64).reshape(-1)
    result = np.fromiter((ceil_sqrt_q16(int(v)) for v in flat), dtype=np.uint64, count=len(flat))
    return result.reshape(np.shape(squared))


def _sampler_configuration() -> dict[str, Any]:
    return {
        "sampler": "F1_CIIP_V1_CANONICAL_SAMPLE",
        "sample_cap": CONFIG.sample_count,
        "unique_preimage": "EXACT_BINARY64_XYZ",
        "rank": "SHA256_DOMAIN_TAGGED_POINT_ID",
        "source_ordinal": "MINIMUM_RAW_VERTEX_ORDINAL_PER_EXACT_XYZ",
    }


def load_bound_sample(raw_source: Path, *, permute_input: bool = False) -> Sample:
    if sha_file(raw_source) != CONFIG.raw_source_sha256:
        raise GCDFStop("INVALID_GCDF_V1_SOURCE_OR_LOCK_BINDING_STOP", "raw source SHA-256 mismatch")
    descriptor = os.open(raw_source, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    xyz_raw = np.asarray(parsed.xyz, dtype="<f8")
    if not np.all(np.isfinite(xyz_raw)):
        raise GCDFStop("INVALID_GCDF_V1_SOURCE_OR_LOCK_BINDING_STOP", "nonfinite source coordinate")
    ordinals_raw = np.arange(len(xyz_raw), dtype=np.uint32)
    if permute_input:
        # Deterministic Fisher-Yates permutation. Point IDs and stable node IDs
        # remain coordinate/source derived and are not restored to producer order.
        seed = hashlib.sha256((CONFIG.attempt_id + "|PERMUTED_ROLE").encode("ascii")).digest()
        state = int.from_bytes(seed[:8], "little") or 1
        order = np.arange(len(xyz_raw), dtype=np.uint32)
        for index in range(len(order) - 1, 0, -1):
            state ^= (state << 13) & ((1 << 64) - 1)
            state ^= state >> 7
            state ^= (state << 17) & ((1 << 64) - 1)
            pick = state % (index + 1)
            order[index], order[pick] = order[pick], order[index]
        xyz_work = xyz_raw[order]
        ordinal_work = ordinals_raw[order]
    else:
        xyz_work = xyz_raw
        ordinal_work = ordinals_raw
    packed = np.ascontiguousarray(xyz_work).view("V24").reshape(len(xyz_work))
    _unique, first, inverse, counts = np.unique(packed, return_index=True, return_inverse=True, return_counts=True)
    unique_xyz = np.ascontiguousarray(xyz_work[first], dtype="<f8")
    minimum_ordinals = np.full(len(first), np.iinfo(np.uint32).max, dtype=np.uint32)
    np.minimum.at(minimum_ordinals, inverse, ordinal_work)
    sampled = partition.canonical_sample(unique_xyz, np.asarray(counts, dtype=np.uint32), sample_cap=CONFIG.sample_count)
    if len(sampled.xyz) != CONFIG.sample_count:
        raise GCDFStop("INVALID_GCDF_V1_SOURCE_OR_LOCK_BINDING_STOP", "canonical sample count mismatch")
    packed_unique = np.ascontiguousarray(unique_xyz).view("V24").reshape(len(unique_xyz))
    lookup = {bytes(row): int(minimum_ordinals[i]) for i, row in enumerate(packed_unique)}
    sampled_packed = np.ascontiguousarray(sampled.xyz).view(np.uint8).reshape(len(sampled.xyz), 24)
    sampled_ordinals = np.fromiter((lookup[bytes(row)] for row in sampled_packed), dtype=np.uint32, count=len(sampled_packed))
    ordinal_sha = _sha_arrays(b"GCDF_V1_ORDERED_SOURCE_ORDINALS\0", (sampled_ordinals.astype("<u4"),))
    coordinate_sha = _sha_arrays(b"GCDF_V1_SAMPLED_BINARY64_XYZ\0", (np.asarray(sampled.xyz, dtype="<f8"),))
    return Sample(
        xyz=np.asarray(sampled.xyz, dtype=np.float64),
        point_ids=np.asarray(sampled.point_ids, dtype=np.uint8),
        multiplicities=np.asarray(sampled.multiplicities, dtype=np.uint32),
        source_ordinals=sampled_ordinals,
        ordered_source_ordinal_sha256=ordinal_sha,
        sampled_coordinate_sha256=coordinate_sha,
        sampler_executable_sha256=sha_file(Path(partition.__file__).resolve()),
        sampler_configuration_sha256=sha_json(_sampler_configuration()),
    )


def canonicalize_sample(sample: Sample) -> CanonicalNodes:
    xyz = np.asarray(sample.xyz, dtype=np.float64)
    minimum = np.min(xyz, axis=0)
    maximum = np.max(xyz, axis=0)
    extent = float(np.max(maximum - minimum))
    if not math.isfinite(extent) or extent <= 0.0:
        raise GCDFStop("INVALID_GCDF_V1_CANONICALIZATION_STOP", "zero/nonfinite source range")
    q = extent / float(1 << CONFIG.quantization_bits)
    integer = np.rint((xyz - minimum) / q).astype("<u4")
    packed = np.ascontiguousarray(integer).view("V12").reshape(len(integer))
    _unique, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
    count = len(first)
    if count < CONFIG.minimum_unique_nodes:
        raise GCDFStop("INVALID_GCDF_V1_CANONICALIZATION_STOP", f"only {count} unique quantized nodes")
    multiplicities = np.zeros(count, dtype=np.uint64)
    np.add.at(multiplicities, inverse, sample.multiplicities.astype(np.uint64))
    minimum_ordinals = np.full(count, np.iinfo(np.uint32).max, dtype=np.uint32)
    np.minimum.at(minimum_ordinals, inverse, sample.source_ordinals)
    records = np.empty(count, dtype=[("x", "<u4"), ("y", "<u4"), ("z", "<u4"), ("o", "<u4")])
    records["x"] = integer[first, 0]
    records["y"] = integer[first, 1]
    records["z"] = integer[first, 2]
    records["o"] = minimum_ordinals
    order = np.argsort(records, order=("x", "y", "z", "o"), kind="stable")
    ordered_integer = np.ascontiguousarray(integer[first][order], dtype="<u4")
    ordered_ordinals = np.ascontiguousarray(minimum_ordinals[order], dtype="<u4")
    ordered_multiplicities = np.ascontiguousarray(multiplicities[order], dtype="<u8")
    metric_xyz = minimum + q * ordered_integer.astype(np.float64)
    table_sha = _sha_arrays(
        b"GCDF_V1_CANONICAL_NODE_TABLE\0",
        (ordered_integer, ordered_ordinals, ordered_multiplicities),
    )
    return CanonicalNodes(
        integer_xyz=ordered_integer,
        metric_xyz=metric_xyz,
        multiplicities=ordered_multiplicities,
        minimum_source_ordinals=ordered_ordinals,
        q=q,
        origin=minimum,
        duplicate_count=len(integer) - count,
        table_sha256=table_sha,
    )


def exact_neighbors_kdtree(nodes: CanonicalNodes, *, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    xyz = nodes.integer_xyz.astype(np.float64)
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    distances, indices = tree.query(xyz, k=k + 2, workers=1)
    n = len(xyz)
    result = np.empty((n, k), dtype=np.uint32)
    result_sq = np.empty((n, k), dtype=np.uint64)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        for row in range(start, end):
            candidates = np.asarray(indices[row], dtype=np.int64)
            candidates = candidates[candidates != row]
            delta = nodes.integer_xyz[candidates].astype(np.int64) - nodes.integer_xyz[row].astype(np.int64)
            sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
            order = np.lexsort((candidates, sq))
            candidates, sq = candidates[order], sq[order]
            boundary = int(sq[k - 1])
            if len(sq) > k and int(sq[k]) == boundary:
                radius = np.nextafter(math.sqrt(boundary), math.inf)
                tied = np.asarray(tree.query_ball_point(xyz[row], radius), dtype=np.int64)
                tied = tied[tied != row]
                delta = nodes.integer_xyz[tied].astype(np.int64) - nodes.integer_xyz[row].astype(np.int64)
                tied_sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
                tied_order = np.lexsort((tied, tied_sq))
                candidates, sq = tied[tied_order], tied_sq[tied_order]
            result[row] = candidates[:k]
            result_sq[row] = sq[:k]
    return result, result_sq


def exact_neighbors_grid_verified(nodes: CanonicalNodes, *, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Independent validator ordering through a uniform-grid candidate proof.

    A tree supplies only a conservative finite search radius. Candidates are
    independently enumerated from a uniform grid and sorted by the locked
    integer metric and node ID. The cell-boundary proof fails closed if the
    conservative radius cannot cover the kth result.
    """
    xyz_i = nodes.integer_xyz.astype(np.int64)
    xyz_f = xyz_i.astype(np.float64)
    tree = cKDTree(xyz_f, compact_nodes=False, balanced_tree=False, copy_data=False)
    radius, _seed = tree.query(xyz_f, k=k + 1, workers=1)
    kth = np.nextafter(np.asarray(radius[:, -1], dtype=np.float64), math.inf)
    # Grid width is the upper median of per-node kth radii, at least one.
    cell = max(1, int(np.ceil(float(np.median(kth)))))
    keys = xyz_i // cell
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node_id, key in enumerate(keys):
        buckets[(int(key[0]), int(key[1]), int(key[2]))].append(node_id)
    n = len(xyz_i)
    result = np.empty((n, k), dtype=np.uint32)
    result_sq = np.empty((n, k), dtype=np.uint64)
    for row in range(n):
        shell = int(math.ceil(kth[row] / cell)) + 1
        base = keys[row]
        candidates: list[int] = []
        for dx in range(-shell, shell + 1):
            for dy in range(-shell, shell + 1):
                for dz in range(-shell, shell + 1):
                    candidates.extend(buckets.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)), ()))
        candidate_array = np.asarray(candidates, dtype=np.int64)
        candidate_array = candidate_array[candidate_array != row]
        delta = xyz_i[candidate_array] - xyz_i[row]
        sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
        order = np.lexsort((candidate_array, sq))
        if len(order) < k:
            raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "grid validator found fewer than k candidates")
        chosen = order[:k]
        result[row] = candidate_array[chosen]
        result_sq[row] = sq[chosen]
    return result, result_sq


def _temporary_tangents(nodes: CanonicalNodes, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = nodes.integer_xyz[neighbors].astype(np.float64) - nodes.integer_xyz[:, None, :].astype(np.float64)
    centered = offsets - np.mean(offsets, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=False)
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    abs_normal = np.abs(normals)
    largest_axis = np.argmax(abs_normal, axis=1)
    sign = np.take_along_axis(normals, largest_axis[:, None], axis=1)[:, 0] < 0.0
    normals[sign] *= -1.0
    descending = values[:, ::-1]
    numerator = np.maximum(descending[:, 1] - descending[:, 2], 0.0)
    gap = np.zeros(len(nodes.integer_xyz), dtype=np.int64)
    positive = descending[:, 0] > 0.0
    gap[positive] = np.floor(1_000_000.0 * numerator[positive] / descending[positive, 0]).astype(np.int64)
    reliable = gap >= CONFIG.reliable_gap_ppm
    return normals, descending, reliable


def _directed_code(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    return src.astype(np.uint64) * np.uint64(n) + dst.astype(np.uint64)


def build_surface_graph(nodes: CanonicalNodes, *, validator_backend: bool = False) -> SurfaceGraph:
    n = len(nodes.integer_xyz)
    neighbors, neighbor_sq = (
        exact_neighbors_grid_verified(nodes) if validator_backend else exact_neighbors_kdtree(nodes)
    )
    neighbor_w = _q16_weights(neighbor_sq)
    r8 = neighbor_w[:, CONFIG.density_neighbor_rank - 1]
    normals, eigenvalues_desc, reliable = _temporary_tangents(nodes, neighbors)
    first24 = neighbors[:, : CONFIG.graph_neighbors]
    first24_w = neighbor_w[:, : CONFIG.graph_neighbors]
    src_all = np.repeat(np.arange(n, dtype=np.uint32), CONFIG.graph_neighbors)
    dst_all = first24.reshape(-1)
    directed_codes = _directed_code(src_all, dst_all, n)
    directed_order = np.argsort(directed_codes, kind="stable")
    sorted_codes = directed_codes[directed_order]
    sorted_weights = first24_w.reshape(-1)[directed_order]
    reverse_codes = _directed_code(dst_all, src_all, n)
    pos = np.searchsorted(sorted_codes, reverse_codes)
    mutual = (pos < len(sorted_codes)) & (sorted_codes[np.minimum(pos, len(sorted_codes) - 1)] == reverse_codes)
    left = src_all[mutual]
    right = dst_all[mutual]
    weight = first24_w.reshape(-1)[mutual]
    unique_half = left < right
    left, right, weight = left[unique_half], right[unique_half], weight[unique_half]
    gate = {token: 0 for token in ATTRITION_TOKENS}
    gate["DUPLICATE_QUANTIZATION"] = nodes.duplicate_count
    gate["EDGE_NONMUTUAL"] = int(len(src_all) - np.count_nonzero(mutual))
    radius_ok = weight <= CONFIG.edge_radius_multiple * np.maximum(r8[left], r8[right])
    gate["EDGE_RADIUS_HIGH"] = int(np.count_nonzero(~radius_ok))
    left, right, weight = left[radius_ok], right[radius_ok], weight[radius_ok]
    # Exact short-triangle witness lookup through the canonical directed table.
    redundancy = np.zeros(len(left), dtype=np.uint8)
    for start in range(0, len(left), 100_000):
        end = min(len(left), start + 100_000)
        le = left[start:end]
        ri = right[start:end]
        candidates = first24[le]
        codes = ri[:, None].astype(np.uint64) * np.uint64(n) + candidates.astype(np.uint64)
        locations = np.searchsorted(sorted_codes, codes)
        clipped = np.minimum(locations, len(sorted_codes) - 1)
        present = sorted_codes[clipped] == codes
        witness_w = sorted_weights[clipped]
        inequality = 2 * (first24_w[le].astype(np.uint64) + witness_w.astype(np.uint64)) <= 5 * weight[start:end, None].astype(np.uint64)
        redundancy[start:end] = np.sum(present & inequality, axis=1).clip(0, 255).astype(np.uint8)
    both_reliable = reliable[left] & reliable[right]
    required = np.where(both_reliable, 1, 2)
    redundant_ok = redundancy >= required
    gate["EDGE_REDUNDANCY_LOW"] = int(np.count_nonzero(~redundant_ok))
    left, right, weight, both_reliable = (
        left[redundant_ok], right[redundant_ok], weight[redundant_ok], both_reliable[redundant_ok]
    )
    tangent_ok = np.ones(len(left), dtype=bool)
    if np.any(both_reliable):
        idx = np.flatnonzero(both_reliable)
        delta = nodes.integer_xyz[right[idx]].astype(np.float64) - nodes.integer_xyz[left[idx]].astype(np.float64)
        delta /= np.linalg.norm(delta, axis=1)[:, None]
        tau = np.maximum(
            np.abs(np.einsum("ij,ij->i", normals[left[idx]], delta)),
            np.abs(np.einsum("ij,ij->i", normals[right[idx]], delta)),
        )
        score = np.rint((1 << 24) * tau).astype(np.int64)
        boundary = (score >= CONFIG.tangent_admit_q24 + 1) & (score <= CONFIG.tangent_boundary_high_q24)
        high = score >= CONFIG.tangent_boundary_high_q24 + 1
        gate["EDGE_TANGENCY_NUMERIC_BOUNDARY"] = int(np.count_nonzero(boundary))
        gate["EDGE_TANGENCY_HIGH"] = int(np.count_nonzero(high))
        tangent_ok[idx] = score <= CONFIG.tangent_admit_q24
    left, right, weight = left[tangent_ok], right[tangent_ok], weight[tangent_ok]
    order = np.lexsort((right, left))
    left = np.ascontiguousarray(left[order], dtype=np.uint32)
    right = np.ascontiguousarray(right[order], dtype=np.uint32)
    weight = np.ascontiguousarray(weight[order], dtype=np.uint64)
    matrix = sparse.csr_matrix(
        (np.concatenate([weight, weight]).astype(np.float64),
         (np.concatenate([left, right]), np.concatenate([right, left]))),
        shape=(n, n),
    )
    component_count, labels = csgraph.connected_components(matrix, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=component_count)
    active_components = np.flatnonzero(sizes >= CONFIG.active_component_minimum_nodes).astype(np.uint32)
    active_mask = np.isin(labels, active_components)
    active_mass = int(np.sum(nodes.multiplicities[active_mask], dtype=np.uint64))
    total_mass = int(np.sum(nodes.multiplicities, dtype=np.uint64))
    active_mass_ppm = ppm(active_mass, total_mass)
    active_edge = active_mask[left] & active_mask[right]
    active_node_count = int(np.count_nonzero(active_mask))
    mean_degree_ppm = 0 if active_node_count == 0 else (2_000_000 * int(np.count_nonzero(active_edge))) // active_node_count
    reliable_eligible = int(np.count_nonzero(both_reliable)) + gate["EDGE_TANGENCY_NUMERIC_BOUNDARY"] + gate["EDGE_TANGENCY_HIGH"]
    boundary_ppm = 0 if reliable_eligible == 0 else ppm(gate["EDGE_TANGENCY_NUMERIC_BOUNDARY"], reliable_eligible)
    if (
        active_mass_ppm < CONFIG.active_mass_minimum_ppm
        or len(active_components) > CONFIG.active_component_maximum
        or mean_degree_ppm < CONFIG.active_mean_degree_minimum * 1_000_000
        or boundary_ppm > CONFIG.tangency_boundary_maximum_ppm
    ):
        raise GCDFStop(
            "INVALID_GCDF_V1_SURFACE_GRAPH_STOP",
            f"graph wall failed mass={active_mass_ppm} components={len(active_components)} mean_degree_ppm={mean_degree_ppm} boundary_ppm={boundary_ppm}",
        )
    component_medians: dict[int, int] = {}
    for component in active_components:
        mask = (labels[left] == component) & (labels[right] == component)
        component_medians[int(component)] = lower_median(weight[mask])
    edge_sha = _sha_arrays(b"GCDF_V1_EDGE_SET\0", (left, right, weight))
    component_table = np.stack([np.arange(n, dtype=np.uint32), labels.astype(np.uint32), active_mask.astype(np.uint32)], axis=1)
    component_sha = _sha_arrays(b"GCDF_V1_COMPONENT_TABLE\0", (component_table,))
    semantic = sha_json({
        "canonical_node_sha256": nodes.table_sha256,
        "edge_set_sha256": edge_sha,
        "component_table_sha256": component_sha,
        "active_components": active_components.tolist(),
        "component_medians": component_medians,
        "gate_census": gate,
    })
    return SurfaceGraph(
        neighbors32=neighbors, neighbor_sq32=neighbor_sq, neighbor_w32=neighbor_w,
        r8=r8, reliable=reliable, normals=normals, eigenvalues_desc=eigenvalues_desc,
        edges_left=left, edges_right=right, edge_weights=weight,
        component_labels=labels.astype(np.uint32), active_components=active_components,
        active_mask=active_mask, component_medians=component_medians, gate_census=gate,
        canonical_node_sha256=nodes.table_sha256, edge_set_sha256=edge_sha,
        component_table_sha256=component_sha, semantic_sha256=semantic,
    )


def graph_csr(graph: SurfaceGraph, node_count: int) -> sparse.csr_matrix:
    left, right, weight = graph.edges_left, graph.edges_right, graph.edge_weights
    matrix = sparse.csr_matrix(
        (np.concatenate([weight, weight]).astype(np.float64),
         (np.concatenate([left, right]), np.concatenate([right, left]))),
        shape=(node_count, node_count),
    )
    matrix.sort_indices()
    return matrix


def select_landmarks(nodes: CanonicalNodes, graph: SurfaceGraph) -> tuple[np.ndarray, dict[int, list[int]]]:
    components = [int(v) for v in graph.active_components]
    if not components:
        raise GCDFStop("INVALID_GCDF_V1_SURFACE_GRAPH_STOP", "no active graph components")
    if 4 * len(components) > CONFIG.landmark_count:
        raise GCDFStop("INVALID_GCDF_V1_SURFACE_GRAPH_STOP", "too many active components for landmark floor")
    mass = {c: int(np.sum(nodes.multiplicities[graph.component_labels == c], dtype=np.uint64)) for c in components}
    remaining = CONFIG.landmark_count - 4 * len(components)
    total = sum(mass.values())
    base = {c: 4 + (remaining * mass[c]) // total for c in components}
    used = sum(base.values())
    remainder_order = sorted(components, key=lambda c: (-(remaining * mass[c]) % total, c))
    # The negative modulo expression above is not the Hamilton remainder. Keep
    # an explicit exact ordering for auditability.
    remainder_order = sorted(components, key=lambda c: (-(remaining * mass[c] - ((remaining * mass[c]) // total) * total), c))
    for c in remainder_order[: CONFIG.landmark_count - used]:
        base[c] += 1
    per_component: dict[int, list[int]] = {}
    all_landmarks: list[int] = []
    xyz = nodes.integer_xyz.astype(np.int64)
    for component in components:
        members = np.flatnonzero(graph.component_labels == component)
        first = int(members[0])
        chosen = [first]
        delta = xyz[members] - xyz[first]
        minimum_sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
        for _ in range(1, base[component]):
            maximum = np.max(minimum_sq)
            candidates = members[minimum_sq == maximum]
            pick = int(np.min(candidates))
            chosen.append(pick)
            delta = xyz[members] - xyz[pick]
            sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
            minimum_sq = np.minimum(minimum_sq, sq)
        per_component[component] = chosen
        all_landmarks.extend(chosen)
    result = np.asarray(sorted(all_landmarks), dtype=np.uint32)
    if len(result) != CONFIG.landmark_count:
        raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "landmark allocation count mismatch")
    return result, per_component


def _ambient_weight(nodes: CanonicalNodes, left: int, right: int) -> int:
    delta = nodes.integer_xyz[left].astype(np.int64) - nodes.integer_xyz[right].astype(np.int64)
    squared = int(np.dot(delta, delta))
    return ceil_sqrt_q16(squared)


def _canonical_path(matrix: sparse.csr_matrix, distances: np.ndarray, source: int, target: int) -> tuple[np.ndarray, np.ndarray]:
    current = int(target)
    reverse = [current]
    edge_reverse: list[int] = []
    while current != source:
        start, end = matrix.indptr[current], matrix.indptr[current + 1]
        neighbors = matrix.indices[start:end]
        weights = matrix.data[start:end].astype(np.int64)
        equality = distances[neighbors] + weights == distances[current]
        if not np.any(equality):
            raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "canonical predecessor not found")
        predecessor = int(np.min(neighbors[equality]))
        location = int(np.flatnonzero(neighbors == predecessor)[0])
        edge_reverse.append(int(weights[location]))
        reverse.append(predecessor)
        current = predecessor
        if len(reverse) > matrix.shape[0]:
            raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "predecessor cycle")
    return np.asarray(reverse[::-1], dtype=np.uint32), np.asarray(edge_reverse[::-1], dtype=np.uint64)


def _nearest_path_vertex(cumulative: np.ndarray, target: int, path_nodes: np.ndarray) -> int:
    position = int(np.searchsorted(cumulative, target))
    choices = []
    if position < len(cumulative):
        choices.append(position)
    if position > 0:
        choices.append(position - 1)
    return min(choices, key=lambda i: (abs(int(cumulative[i]) - target), int(path_nodes[i])))


def _profile_for_path(
    nodes: CanonicalNodes,
    path_nodes: np.ndarray,
    edge_weights: np.ndarray,
    component_median: int,
) -> dict[str, Any]:
    cumulative = np.concatenate([np.asarray([0], dtype=np.uint64), np.cumsum(edge_weights, dtype=np.uint64)])
    total = int(cumulative[-1])
    half = CONFIG.local_half_window_multiple * component_median
    stride = CONFIG.profile_stride_multiple * component_median
    if total < 2 * half:
        return {"evaluable": False, "samples": []}
    centers = range(half, total - half + 1, stride)
    samples: list[dict[str, int]] = []
    for center in centers:
        li = _nearest_path_vertex(cumulative, center - half, path_nodes)
        ri = _nearest_path_vertex(cumulative, center + half, path_nodes)
        if li >= ri:
            continue
        left_len = int(cumulative[ri] - cumulative[li])
        realized_left = center - int(cumulative[li])
        realized_right = int(cumulative[ri]) - center
        low = CONFIG.local_side_low_multiple * component_median
        high = CONFIG.local_side_high_multiple * component_median
        if not (low <= realized_left <= high and low <= realized_right <= high):
            continue
        chord = _ambient_weight(nodes, int(path_nodes[li]), int(path_nodes[ri]))
        if chord > left_len:
            raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "local metric invariant violated")
        samples.append({
            "center_s": center, "center_node": int(path_nodes[_nearest_path_vertex(cumulative, center, path_nodes)]),
            "left_node": int(path_nodes[li]), "right_node": int(path_nodes[ri]),
            "path_length": left_len, "chord": chord, "deficit_ppm": ppm(left_len - chord, left_len),
        })
    if not samples:
        return {"evaluable": False, "samples": []}
    peak_index = min(
        range(len(samples)),
        key=lambda i: (-samples[i]["deficit_ppm"], samples[i]["center_s"], samples[i]["center_node"]),
    )
    activation = [max(row["deficit_ppm"] - CONFIG.concentration_floor_ppm, 0) for row in samples]
    doubled_weights: list[int] = []
    for i, row in enumerate(samples):
        if len(samples) == 1:
            doubled_weights.append(1)
        elif i == 0:
            doubled_weights.append(samples[1]["center_s"] - row["center_s"])
        elif i == len(samples) - 1:
            doubled_weights.append(row["center_s"] - samples[i - 1]["center_s"])
        else:
            doubled_weights.append(samples[i + 1]["center_s"] - samples[i - 1]["center_s"])
    all_mass = sum(a * w for a, w in zip(activation, doubled_weights, strict=True))
    peak_s = samples[peak_index]["center_s"]
    peak_mass = sum(
        a * w for row, a, w in zip(samples, activation, doubled_weights, strict=True)
        if abs(row["center_s"] - peak_s) <= 2 * half
    )
    concentration = 0 if all_mass == 0 else ppm(peak_mass, all_mass)
    return {
        "evaluable": True,
        "samples": samples,
        "peak_index": peak_index,
        "peak": samples[peak_index],
        "all_activation_mass": all_mass,
        "peak_activation_mass": peak_mass,
        "concentration_ppm": concentration,
    }


def evaluate_landmark_paths(nodes: CanonicalNodes, graph: SurfaceGraph, landmarks_by_component: Mapping[int, Sequence[int]]) -> dict[str, Any]:
    matrix = graph_csr(graph, len(nodes.integer_xyz))
    census = Counter({token: 0 for token in ATTRITION_TOKENS})
    votes: list[dict[str, Any]] = []
    visual_global_paths: list[dict[str, Any]] = []
    controls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_pairs: list[tuple[int, int]] = []
    for component in sorted(landmarks_by_component):
        landmarks = sorted(int(v) for v in landmarks_by_component[component])
        median = graph.component_medians[component]
        for source_index, source in enumerate(landmarks):
            distances = csgraph.dijkstra(matrix, directed=False, indices=source, return_predecessors=False)
            for target in landmarks[source_index + 1 :]:
                distance_float = distances[target]
                if not np.isfinite(distance_float):
                    raise GCDFStop("INVALID_GCDF_V1_SURFACE_GRAPH_STOP", "landmarks disconnected within component")
                distance = int(distance_float)
                pair_id = hashlib.sha256(f"{source}:{target}".encode("ascii")).hexdigest()
                if distance < CONFIG.pair_scale_low_multiple * median:
                    census["PAIR_SCALE_LOW"] += 1
                    controls["PAIR_SCALE_LOW"].append({"id": pair_id, "margin": distance - CONFIG.pair_scale_low_multiple * median, "endpoints": [source, target]})
                    continue
                if distance > CONFIG.pair_scale_high_multiple * median:
                    census["PAIR_SCALE_HIGH"] += 1
                    controls["PAIR_SCALE_HIGH"].append({"id": pair_id, "margin": CONFIG.pair_scale_high_multiple * median - distance, "endpoints": [source, target]})
                    continue
                eligible_pairs.append((source, target))
                path_nodes, path_edges = _canonical_path(matrix, distances, source, target)
                chord = _ambient_weight(nodes, source, target)
                if distance < chord:
                    raise GCDFStop("INVALID_GCDF_V1_ROLE_CONSENSUS_STOP", "global metric invariant violated")
                global_ppm = ppm(distance - chord, distance)
                base = {
                    "id": pair_id, "component": component, "endpoints": [source, target],
                    "distance": distance, "chord": chord, "global_deficit_ppm": global_ppm,
                    "path_sha256": _sha_arrays(b"GCDF_V1_PATH\0", (path_nodes, path_edges)),
                    "visual_path_nodes": path_nodes.tolist(),
                }
                if global_ppm < CONFIG.global_deficit_minimum_ppm:
                    census["GLOBAL_DEFICIT_LOW"] += 1
                    controls["GLOBAL_DEFICIT_LOW"].append({**base, "margin": global_ppm - CONFIG.global_deficit_minimum_ppm})
                    continue
                visual_global_paths.append({"id": pair_id, "nodes": path_nodes.tolist(), "endpoints": [source, target], "global_deficit_ppm": global_ppm})
                profile = _profile_for_path(nodes, path_nodes, path_edges, median)
                if not profile["evaluable"]:
                    census["LOCAL_PROFILE_NOT_EVALUABLE"] += 1
                    controls["LOCAL_PROFILE_NOT_EVALUABLE"].append({**base, "margin": -1})
                    continue
                peak = profile["peak"]
                profile_summary = [{"s": r["center_s"], "d": r["deficit_ppm"]} for r in profile["samples"]]
                enriched = {**base, "profile": profile_summary, "peak": peak, "concentration_ppm": profile["concentration_ppm"]}
                if peak["deficit_ppm"] < CONFIG.local_peak_minimum_ppm:
                    census["LOCAL_PEAK_LOW"] += 1
                    controls["LOCAL_PEAK_LOW"].append({**enriched, "margin": peak["deficit_ppm"] - CONFIG.local_peak_minimum_ppm})
                    continue
                if profile["concentration_ppm"] < CONFIG.peak_concentration_minimum_ppm:
                    census["PEAK_CONCENTRATION_LOW"] += 1
                    controls["PEAK_CONCENTRATION_LOW"].append({**enriched, "margin": profile["concentration_ppm"] - CONFIG.peak_concentration_minimum_ppm})
                    continue
                census["PASS"] += 1
                votes.append({
                    **enriched,
                    "visual_path_nodes": path_nodes.tolist(),
                    "peak_node": peak["center_node"],
                    "left_node": peak["left_node"],
                    "right_node": peak["right_node"],
                    "lower_landmark_id": min(source, target),
                    "upper_landmark_id": max(source, target),
                })
    if len(eligible_pairs) < CONFIG.pair_coverage_minimum:
        decision = "GCDF_V1_PAIR_COVERAGE_ARCHITECTURE_STOP"
    else:
        fraction = ppm(len(votes), len(eligible_pairs))
        if (
            len(votes) < CONFIG.accepted_path_minimum
            or fraction < CONFIG.accepted_fraction_minimum_ppm
            or fraction > CONFIG.accepted_fraction_maximum_ppm
        ):
            decision = "GCDF_V1_PATH_SELECTIVITY_ARCHITECTURE_STOP"
        else:
            decision = "GCDF_V1_PATH_FIELD_SURVIVES"
    for key in list(controls):
        controls[key] = sorted(controls[key], key=lambda row: (abs(int(row.get("margin", 0))), str(row["id"])))[:12]
    vote_table = [
        {k: row[k] for k in (
            "id", "component", "endpoints", "distance", "chord", "global_deficit_ppm",
            "path_sha256", "peak", "profile", "concentration_ppm", "peak_node", "left_node",
            "right_node", "lower_landmark_id", "upper_landmark_id",
        )}
        for row in votes
    ]
    return {
        "decision": decision,
        "census": dict(census),
        "scale_eligible_pair_count": len(eligible_pairs),
        "accepted_vote_count": len(votes),
        "accepted_fraction_ppm": 0 if not eligible_pairs else ppm(len(votes), len(eligible_pairs)),
        "eligible_pair_sha256": sha_json(eligible_pairs),
        "path_gate_census_sha256": sha_json(dict(census)),
        "vote_table_sha256": sha_json(vote_table),
        "votes": votes,
        "visual_global_paths": visual_global_paths,
        "controls": dict(controls),
    }


def _truncated_distances(matrix: sparse.csr_matrix, source: int, limit: int) -> dict[int, int]:
    distances = {source: 0}
    queue = [(0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node) or distance > limit:
            continue
        start, end = matrix.indptr[node], matrix.indptr[node + 1]
        for neighbor, weight in zip(matrix.indices[start:end], matrix.data[start:end], strict=True):
            candidate = distance + int(weight)
            neighbor_i = int(neighbor)
            if candidate <= limit and candidate < distances.get(neighbor_i, limit + 1):
                distances[neighbor_i] = candidate
                heapq.heappush(queue, (candidate, neighbor_i))
    return distances


def _union_find_components(node_count: int, edges: Sequence[tuple[int, int, int]]) -> list[list[int]]:
    parent = list(range(node_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a

    for left, right, _weight in edges:
        union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for node in range(node_count):
        groups[find(node)].append(node)
    return [groups[key] for key in sorted(groups)]


def _mst_for_component(members: Sequence[int], edges: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    member_set = set(members)
    local = {member: index for index, member in enumerate(sorted(members))}
    parent = list(range(len(local)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    result: list[tuple[int, int, int]] = []
    for left, right, weight in sorted(
        (row for row in edges if row[0] in member_set and row[1] in member_set),
        key=lambda row: (row[2], min(row[0], row[1]), max(row[0], row[1])),
    ):
        a, b = find(local[left]), find(local[right])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        parent[b] = a
        result.append((left, right, weight))
        if len(result) == len(members) - 1:
            break
    return result


def _tree_diameter(members: Sequence[int], mst: Sequence[tuple[int, int, int]]) -> tuple[int, int, int, list[int]]:
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for left, right, weight in mst:
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    best: tuple[int, tuple[int, int], list[int]] | None = None
    for source in sorted(members):
        stack = [(source, -1, 0, [source])]
        while stack:
            node, parent, distance, path = stack.pop()
            pair = (min(source, node), max(source, node))
            candidate = (distance, pair, path if source <= node else list(reversed(path)))
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
            for neighbor, weight in adjacency[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, distance + weight, path + [neighbor]))
    assert best is not None
    return best[1][0], best[1][1], best[0], best[2]


def build_vote_candidates(nodes: CanonicalNodes, graph: SurfaceGraph, path_result: Mapping[str, Any]) -> dict[str, Any]:
    if path_result["decision"] != "GCDF_V1_PATH_FIELD_SURVIVES":
        return {
            "decision": path_result["decision"], "census": {}, "vote_link_edges": [],
            "components": [], "candidates": [], "vote_component_sha256": sha_json([]),
            "candidate_axis_sha256": sha_json([]),
        }
    votes = list(path_result["votes"])
    peak_nodes = sorted({int(row["peak_node"]) for row in votes})
    location = {node: index for index, node in enumerate(peak_nodes)}
    matrix = graph_csr(graph, len(nodes.integer_xyz))
    potential: set[tuple[int, int]] = set()
    peak_xyz = nodes.integer_xyz[peak_nodes].astype(np.float64)
    # Ambient distance is a necessary lower bound on graph distance.
    maximum_median = max(graph.component_medians.values())
    ambient_radius = (CONFIG.vote_link_radius_multiple * maximum_median) / float(1 << CONFIG.metric_fraction_bits)
    for left, right in cKDTree(peak_xyz).query_pairs(ambient_radius, output_type="set"):
        if graph.component_labels[peak_nodes[left]] == graph.component_labels[peak_nodes[right]]:
            potential.add((int(left), int(right)))
    by_left: dict[int, list[int]] = defaultdict(list)
    for left, right in potential:
        by_left[left].append(right)
    edges: list[tuple[int, int, int]] = []
    degree = np.zeros(len(peak_nodes), dtype=np.int64)
    for left in sorted(by_left):
        component = int(graph.component_labels[peak_nodes[left]])
        limit = CONFIG.vote_link_radius_multiple * graph.component_medians[component]
        distances = _truncated_distances(matrix, peak_nodes[left], limit)
        for right in sorted(by_left[left]):
            distance = distances.get(peak_nodes[right])
            if distance is not None:
                edges.append((left, right, distance))
                degree[left] += 1
                degree[right] += 1
    if np.any(degree > CONFIG.vote_link_edge_cap):
        return {
            "decision": "GCDF_V1_PATH_SELECTIVITY_ARCHITECTURE_STOP",
            "census": {"VOTE_LINK_EDGE_CAP_EXCEEDED": int(np.count_nonzero(degree > CONFIG.vote_link_edge_cap))},
            "vote_link_edges": edges, "components": [], "candidates": [],
            "vote_component_sha256": sha_json([]), "candidate_axis_sha256": sha_json([]),
        }
    census = Counter()
    census["VOTE_LINK_ISOLATED"] = int(np.count_nonzero(degree == 0))
    groups = _union_find_components(len(peak_nodes), edges)
    vote_by_location: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        vote_by_location[location[int(vote["peak_node"])]].append(vote)
    component_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for members in groups:
        component_votes = [vote for member in members for vote in vote_by_location[member]]
        endpoints = [endpoint for vote in component_votes for endpoint in vote["endpoints"]]
        endpoint_counts = Counter(endpoints)
        gate = "PASS"
        margin = 0
        if len(component_votes) < CONFIG.vote_minimum:
            gate, margin = "VOTE_COUNT_LOW", len(component_votes) - CONFIG.vote_minimum
        elif len(members) < CONFIG.unique_peak_minimum:
            gate, margin = "UNIQUE_PEAK_COUNT_LOW", len(members) - CONFIG.unique_peak_minimum
        elif len(endpoint_counts) < CONFIG.endpoint_diversity_minimum:
            gate, margin = "ENDPOINT_DIVERSITY_LOW", len(endpoint_counts) - CONFIG.endpoint_diversity_minimum
        elif max(endpoint_counts.values()) * 1_000_000 > CONFIG.endpoint_dominance_maximum_ppm * len(component_votes):
            gate = "ENDPOINT_DOMINANCE_HIGH"
            margin = CONFIG.endpoint_dominance_maximum_ppm - ppm(max(endpoint_counts.values()), len(component_votes))
        mst = _mst_for_component(members, edges)
        if len(members) == 1:
            endpoint_a = endpoint_b = members[0]
            diameter, spine = 0, [members[0]]
        elif len(mst) != len(members) - 1:
            endpoint_a = endpoint_b = members[0]
            diameter, spine = 0, [members[0]]
        else:
            endpoint_a, endpoint_b, diameter, spine = _tree_diameter(members, mst)
        component_id = hashlib.sha256(
            b"GCDF_V1_VOTE_COMPONENT\0" + b"".join(int(peak_nodes[m]).to_bytes(4, "little") for m in sorted(members))
        ).hexdigest()
        component = int(graph.component_labels[peak_nodes[members[0]]])
        median = graph.component_medians[component]
        mst_total = sum(row[2] for row in mst)
        if gate == "PASS" and diameter < CONFIG.spine_extent_minimum_multiple * median:
            gate, margin = "SPINE_EXTENT_LOW", diameter - CONFIG.spine_extent_minimum_multiple * median
        branch_ppm = 0 if diameter == 0 else ppm(max(mst_total - diameter, 0), diameter)
        if gate == "PASS" and branch_ppm > CONFIG.spine_branching_maximum_ppm:
            gate, margin = "SPINE_BRANCHING_HIGH", CONFIG.spine_branching_maximum_ppm - branch_ppm
        endpoint_nodes = (peak_nodes[endpoint_a], peak_nodes[endpoint_b])
        chord = 0 if endpoint_nodes[0] == endpoint_nodes[1] else _ambient_weight(nodes, *endpoint_nodes)
        chord_ppm = 0 if diameter == 0 else ppm(min(chord, diameter), diameter)
        if gate == "PASS" and chord_ppm < CONFIG.spine_chord_ratio_minimum_ppm:
            gate, margin = "SPINE_CHORD_RATIO_LOW", chord_ppm - CONFIG.spine_chord_ratio_minimum_ppm
        census[gate] += 1
        row = {
            "component_id": component_id, "member_locations": members, "peak_nodes": [peak_nodes[m] for m in members],
            "vote_ids": [vote["id"] for vote in component_votes], "vote_count": len(component_votes),
            "unique_peak_count": len(members), "endpoint_diversity": len(endpoint_counts),
            "endpoint_dominance_ppm": 0 if not component_votes else ppm(max(endpoint_counts.values()), len(component_votes)),
            "mst": mst, "diameter": diameter, "spine_locations": spine,
            "spine_peak_nodes": [peak_nodes[m] for m in spine], "mst_total": mst_total,
            "branch_ppm": branch_ppm, "spine_chord": chord, "spine_chord_ppm": chord_ppm,
            "gate": gate, "margin": margin,
        }
        component_rows.append(row)
        if gate == "PASS":
            anchor = nodes.metric_xyz[endpoint_nodes[0]]
            vector = nodes.metric_xyz[endpoint_nodes[1]] - anchor
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                continue
            direction = vector / norm
            peak_points = nodes.metric_xyz[row["peak_nodes"]]
            axial = (peak_points - anchor) @ direction
            radial = peak_points - (anchor + axial[:, None] * direction)
            support_radius = float(np.median(np.linalg.norm(radial, axis=1)))
            candidates.append({
                **row, "candidate_id": component_id,
                "axis_anchor": anchor.tolist(), "axis_direction": direction.tolist(),
                "axial_minimum": float(np.min(axial)), "axial_maximum": float(np.max(axial)),
                "support_radius": support_radius, "component": component,
                "component_median": median,
            })
    if len(candidates) == 0:
        decision = "GCDF_V1_NO_COHERENT_EXCESS_SPINE_ARCHITECTURE_STOP"
    elif len(candidates) > CONFIG.candidate_maximum:
        decision = "GCDF_V1_NONSELECTIVE_CANDIDATE_FIELD_ARCHITECTURE_STOP"
    else:
        decision = "GCDF_V1_GEOMETRIC_SPINES_SURVIVE"
    compact = [
        {k: row[k] for k in (
            "candidate_id", "axis_anchor", "axis_direction", "axial_minimum", "axial_maximum",
            "support_radius", "vote_count", "unique_peak_count", "endpoint_diversity",
            "diameter", "branch_ppm", "spine_chord_ppm",
        )}
        for row in candidates
    ]
    return {
        "decision": decision, "census": dict(census), "vote_link_edges": edges,
        "components": component_rows, "candidates": candidates,
        "vote_component_sha256": sha_json(component_rows),
        "candidate_axis_sha256": sha_json(compact),
    }


def _wrapped_delta(left: int, right: int) -> int:
    period = int(round(2.0 * math.pi * 1_000_000))
    value = (left - right) % period
    if value > period // 2:
        value -= period
    return value


def _fit_circular_l1(angles: Sequence[int]) -> int:
    if not angles:
        raise ValueError("empty angle fit")
    return min(sorted(set(int(v) for v in angles)), key=lambda candidate: (sum(abs(_wrapped_delta(v, candidate)) for v in angles), candidate))


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle) + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))


def evaluate_causality(nodes: CanonicalNodes, path_result: Mapping[str, Any], candidate_result: Mapping[str, Any]) -> dict[str, Any]:
    if candidate_result["decision"] != "GCDF_V1_GEOMETRIC_SPINES_SURVIVE":
        return {"decision": candidate_result["decision"], "candidate_results": [], "causal_fold_sha256": sha_json([])}
    vote_lookup = {row["id"]: row for row in path_result["votes"]}
    results: list[dict[str, Any]] = []
    passing: list[str] = []
    for candidate in candidate_result["candidates"]:
        anchor = np.asarray(candidate["axis_anchor"], dtype=np.float64)
        axis = np.asarray(candidate["axis_direction"], dtype=np.float64)
        median = int(candidate["component_median"])
        rows: list[dict[str, Any]] = []
        for vote_id in candidate["vote_ids"]:
            vote = vote_lookup[vote_id]
            left_id, right_id = int(vote["left_node"]), int(vote["right_node"])
            left, right = nodes.metric_xyz[left_id].copy(), nodes.metric_xyz[right_id].copy()
            displacement = right - left
            transverse = displacement - axis * float(np.dot(displacement, axis))
            coordinate = int(np.argmax(np.abs(transverse)))
            if transverse[coordinate] < 0.0:
                left_id, right_id = right_id, left_id
                left, right = right, left
            left_rel, right_rel = left - anchor, right - anchor
            left_radial = left_rel - axis * float(np.dot(left_rel, axis))
            right_radial = right_rel - axis * float(np.dot(right_rel, axis))
            radial_q16 = int(round(min(np.linalg.norm(left_radial), np.linalg.norm(right_radial)) / nodes.q * (1 << 16)))
            if radial_q16 < CONFIG.causal_radial_minimum_multiple * median:
                continue
            theta = math.atan2(float(np.dot(axis, np.cross(right_radial, -left_radial))), float(np.dot(right_radial, -left_radial)))
            theta_u = int(round(theta * 1_000_000.0))
            fold_digest = hashlib.sha256(
                candidate["candidate_id"].encode("ascii")
                + int(vote["lower_landmark_id"]).to_bytes(4, "little")
                + int(vote["upper_landmark_id"]).to_bytes(4, "little")
            ).digest()
            rows.append({
                "vote_id": vote_id, "fold": int(fold_digest[0] & 1), "theta_urad": theta_u,
                "left": left.tolist(), "right": right.tolist(), "left_radial": left_radial.tolist(),
                "right_radial": right_radial.tolist(), "path_length": int(vote["peak"]["path_length"]),
                "chord": int(vote["peak"]["chord"]), "left_node": left_id, "right_node": right_id,
            })
        directions = []
        candidate_pass = True
        candidate_gate = "PASS"
        for train_fold, test_fold in ((0, 1), (1, 0)):
            train = [row for row in rows if row["fold"] == train_fold]
            test = [row for row in rows if row["fold"] == test_fold]
            if len(train) < CONFIG.causal_fold_minimum or len(test) < CONFIG.causal_fold_minimum:
                candidate_pass = False
                candidate_gate = "CAUSAL_SPLIT_LOW"
                directions.append({"train_fold": train_fold, "test_fold": test_fold, "train_count": len(train), "test_count": len(test), "gate": "CAUSAL_SPLIT_LOW"})
                continue
            fitted = _fit_circular_l1([row["theta_urad"] for row in train])
            reductions: list[int] = []
            wrong: list[int] = []
            for row in test:
                left = np.asarray(row["left"], dtype=np.float64)
                right = np.asarray(row["right"], dtype=np.float64)
                rel = right - anchor
                axial = axis * float(np.dot(rel, axis))
                radial = rel - axial
                rotated = anchor + axial + _rodrigues(radial, axis, fitted / 1_000_000.0)
                wrong_rotated = anchor + axial + _rodrigues(radial, axis, -fitted / 1_000_000.0)
                original = int(row["chord"])
                path_length = int(row["path_length"])
                post = int(round(np.linalg.norm(rotated - left) / nodes.q * (1 << 16)))
                wrong_post = int(round(np.linalg.norm(wrong_rotated - left) / nodes.q * (1 << 16)))
                denominator = path_length - original
                reductions.append(0 if denominator <= 0 else ppm(min(max(post - original, 0), denominator), denominator))
                wrong.append(0 if denominator <= 0 else ppm(min(max(wrong_post - original, 0), denominator), denominator))
            median_reduction = lower_median(reductions)
            vote_pass_count = sum(value >= CONFIG.causal_vote_minimum_ppm for value in reductions)
            wrong_median = lower_median(wrong)
            direction_pass = (
                median_reduction >= CONFIG.causal_median_minimum_ppm
                and 3 * vote_pass_count >= 2 * len(reductions)
                and wrong_median <= CONFIG.causal_wrong_sign_maximum_ppm
            )
            if not direction_pass:
                candidate_pass = False
                candidate_gate = "WRONG_SIGN_CONTROL_HIGH" if wrong_median > CONFIG.causal_wrong_sign_maximum_ppm else "CAUSAL_REDUCTION_LOW"
            directions.append({
                "train_fold": train_fold, "test_fold": test_fold, "train_count": len(train), "test_count": len(test),
                "fitted_angle_urad": fitted, "reductions_ppm": sorted(reductions),
                "wrong_sign_reductions_ppm": sorted(wrong), "median_reduction_ppm": median_reduction,
                "vote_pass_count": vote_pass_count, "wrong_sign_median_ppm": wrong_median,
                "gate": "PASS" if direction_pass else candidate_gate,
            })
        if candidate_pass:
            passing.append(candidate["candidate_id"])
        results.append({
            "candidate_id": candidate["candidate_id"], "causally_evaluable_vote_count": len(rows),
            "directions": directions, "gate": "PASS" if candidate_pass else candidate_gate,
        })
    decision = "GCDF_V1_CAUSAL_PREFLIGHT_SURVIVES" if passing else "GCDF_V1_HINGE_COUNTERFACTUAL_ARCHITECTURE_STOP"
    return {
        "decision": decision, "passing_candidate_ids": passing,
        "candidate_results": results, "causal_fold_sha256": sha_json(results),
    }


def semantic_bundle(
    *, sample: Sample, nodes: CanonicalNodes, graph: SurfaceGraph, landmarks: np.ndarray,
    path_result: Mapping[str, Any], candidate_result: Mapping[str, Any], causal_result: Mapping[str, Any], role: str,
) -> dict[str, Any]:
    decision = str(causal_result["decision"])
    compact_candidates = [
        {k: row[k] for k in (
            "candidate_id", "axis_anchor", "axis_direction", "axial_minimum", "axial_maximum",
            "support_radius", "vote_count", "diameter", "branch_ppm", "spine_chord_ppm",
        )}
        for row in candidate_result.get("candidates", [])
    ]
    value = {
        "schema_version": 1, "role": role, "attempt_id": CONFIG.attempt_id,
        "configuration_sha256": CONFIG_SHA256, "source_sha256": CONFIG.raw_source_sha256,
        "point_count": CONFIG.sample_count, "unique_node_count": len(nodes.integer_xyz),
        "ordered_source_ordinal_sha256": sample.ordered_source_ordinal_sha256,
        "sampled_coordinate_sha256": sample.sampled_coordinate_sha256,
        "sampler_executable_sha256": sample.sampler_executable_sha256,
        "sampler_configuration_sha256": sample.sampler_configuration_sha256,
        "canonical_unique_node_table_sha256": nodes.table_sha256,
        "graph_edge_set_sha256": graph.edge_set_sha256,
        "graph_component_table_sha256": graph.component_table_sha256,
        "landmark_ids": landmarks.tolist(), "landmark_sha256": sha_json(landmarks.tolist()),
        "graph_gate_census": graph.gate_census,
        "path_gate_census": path_result["census"],
        "scale_eligible_pair_count": path_result["scale_eligible_pair_count"],
        "accepted_vote_count": path_result["accepted_vote_count"],
        "accepted_fraction_ppm": path_result["accepted_fraction_ppm"],
        "eligible_pair_sha256": path_result["eligible_pair_sha256"],
        "path_gate_census_sha256": path_result["path_gate_census_sha256"],
        "vote_table_sha256": path_result["vote_table_sha256"],
        "vote_component_sha256": candidate_result["vote_component_sha256"],
        "candidate_axis_sha256": candidate_result["candidate_axis_sha256"],
        "causal_fold_sha256": causal_result["causal_fold_sha256"],
        "compact_candidates": compact_candidates,
        "causal_candidate_results": causal_result.get("candidate_results", []),
        "decision": decision,
        "target_capability_granted": False, "target_capability_opened": False,
        "target_process_launched": False, "source_points_serialized": False,
    }
    normalized = dict(value)
    normalized.pop("role")
    value["semantic_sha256"] = sha_json(normalized)
    return value
