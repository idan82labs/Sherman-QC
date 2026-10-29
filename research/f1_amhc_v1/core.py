"""Deterministic producer for Antipodal Mid-surface Hinge Current V1.

The durable semantic result contains only aggregates, capped components, and
integer-set hashes.  Source coordinates and full intermediate tables exist only
in process memory and in the temporary renderer scratch file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from research.f1_ciip_v1 import partition, primary_parser
from .config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json


class AMHCInvalid(RuntimeError):
    pass


@dataclass(slots=True)
class CanonicalSource:
    xyz: np.ndarray
    point_ids: np.ndarray
    raw_source_sha256: str
    raw_source_bytes: int
    sampled_coordinate_sha256: str
    canonical_table_sha256: str


@dataclass(slots=True)
class NormalField:
    h: float
    normals: np.ndarray
    eigenvalues: np.ndarray
    neighbor_counts: np.ndarray
    eligible: np.ndarray


@dataclass(slots=True)
class PairField:
    h: float
    tau: float
    pilot_distances: np.ndarray
    pilot_mode_fraction: float
    pair_points: np.ndarray
    midpoints: np.ndarray
    directors: np.ndarray
    thicknesses: np.ndarray
    g: float


@dataclass(slots=True)
class GraphField:
    edges: np.ndarray
    lengths: np.ndarray
    fan_interior: np.ndarray
    core: np.ndarray
    theta: np.ndarray
    raw_active: np.ndarray
    core_edges: np.ndarray
    core_lengths: np.ndarray
    core_theta: np.ndarray
    active_core: np.ndarray


@dataclass(slots=True)
class Component:
    component_id: int
    edge_indices: np.ndarray
    midpoint_ids: np.ndarray
    edge_count: int
    diameter: float
    linear_fraction: float
    retained: bool


@dataclass(slots=True)
class Evaluation:
    source: CanonicalSource
    normals: NormalField
    pairs: PairField | None
    graph: GraphField | None
    components: list[Component]
    decision: str
    semantic_payload: dict[str, Any]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_arrays(tag: bytes, arrays: Iterable[np.ndarray | bytes]) -> str:
    digest = hashlib.sha256(tag)
    for value in arrays:
        raw = value if isinstance(value, bytes) else np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _permuted_raw(xyz: np.ndarray, source_sha: str) -> np.ndarray:
    rows = np.ascontiguousarray(xyz, dtype="<f8").view(np.uint8).reshape(len(xyz), 24)
    prefix = b"AMHC_V1_ORDER_PERMUTED\0" + bytes.fromhex(source_sha)
    keys = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(prefix + bytes(row)).digest()[:8], "little")
            for row in rows
        ),
        dtype=np.uint64,
        count=len(rows),
    )
    row_view = rows.view("V24").reshape(len(rows))
    order = np.lexsort((row_view, keys))
    return np.ascontiguousarray(xyz[order], dtype="<f8")


def _content_sample(xyz_raw: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(xyz_raw, dtype="<f8").view("V24").reshape(len(xyz_raw))
    _unique, first, _inverse, counts = np.unique(
        packed, return_index=True, return_inverse=True, return_counts=True
    )
    unique_xyz = np.ascontiguousarray(xyz_raw[first], dtype="<f8")
    sampled = partition.canonical_sample(
        unique_xyz, np.asarray(counts, dtype=np.uint32), sample_cap=CONFIG.sample_count
    )
    if len(sampled.xyz) != CONFIG.sample_count:
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: canonical sample is not 200000")
    return np.asarray(sampled.xyz, dtype=np.float64)


def canonicalize_xyz(
    xyz: np.ndarray,
    *,
    raw_source_sha256: str = "SOURCE_FREE_FIXTURE",
    raw_source_bytes: int = 0,
) -> CanonicalSource:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 25:
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: invalid XYZ shape")
    if not np.all(np.isfinite(points)):
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: nonfinite XYZ")
    bits = np.ascontiguousarray(points, dtype="<f8").view(np.uint64).reshape(-1, 3)
    order = np.lexsort(
        (bits[:, 2], bits[:, 1], bits[:, 0], points[:, 2], points[:, 1], points[:, 0])
    )
    ordered = np.ascontiguousarray(points[order], dtype="<f8")
    if len(ordered) > 1:
        keep = np.ones(len(ordered), dtype=bool)
        keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        ordered = ordered[keep]
    if raw_source_sha256 != "SOURCE_FREE_FIXTURE" and len(ordered) != CONFIG.sample_count:
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: canonical cardinality changed")
    return CanonicalSource(
        xyz=ordered,
        point_ids=np.arange(len(ordered), dtype=np.uint32),
        raw_source_sha256=raw_source_sha256,
        raw_source_bytes=raw_source_bytes,
        sampled_coordinate_sha256=sha_arrays(
            b"AMHC_V1_SAMPLED_BINARY64_XYZ\0", (np.asarray(points, dtype="<f8"),)
        ),
        canonical_table_sha256=sha_arrays(
            b"AMHC_V1_CANONICAL_BINARY64_XYZ\0", (ordered,)
        ),
    )


def load_bound_source(path: Path, *, permute_input: bool = False) -> CanonicalSource:
    observed_sha = sha_file(path)
    observed_bytes = path.stat().st_size
    if observed_sha != CONFIG.raw_source_sha256 or observed_bytes != CONFIG.raw_source_bytes:
        raise AMHCInvalid("INVALID_AMHC_V1_GOVERNANCE_STOP: source binding mismatch")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    raw = np.asarray(parsed.xyz, dtype="<f8")
    if not np.all(np.isfinite(raw)):
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: raw nonfinite coordinate")
    if permute_input:
        raw = _permuted_raw(raw, observed_sha)
    sampled = _content_sample(raw)
    return canonicalize_xyz(
        sampled, raw_source_sha256=observed_sha, raw_source_bytes=observed_bytes
    )


def estimate_normals(source: CanonicalSource) -> NormalField:
    xyz = source.xyz
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    k8 = min(9, len(xyz))
    distances8, _ = tree.query(xyz, k=k8, workers=1)
    distances8 = np.asarray(distances8, dtype=np.float64)
    if distances8.ndim == 1:
        distances8 = distances8[:, None]
    rank = min(8, distances8.shape[1] - 1)
    h = float(np.median(distances8[:, rank]))
    if not math.isfinite(h) or h <= 0:
        raise AMHCInvalid("AMHC_V1_SOURCE_CONTRACT_STOP: invalid h")
    k = min(CONFIG.normal_neighbor_count + 1, len(xyz))
    distances, indices = tree.query(xyz, k=k, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    distances, indices = distances[:, 1:], indices[:, 1:]
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float64)
    eigenvalues = np.full((n, 3), np.inf, dtype=np.float64)
    counts = np.sum(distances <= CONFIG.normal_radius_h * h, axis=1).astype(np.uint8)
    for start in range(0, n, CONFIG.block_size):
        stop = min(n, start + CONFIG.block_size)
        ids = indices[start:stop]
        ds = distances[start:stop]
        mask = ds <= CONFIG.normal_radius_h * h
        weights = np.exp(-((ds / (2.0 * h)) ** 2)) * mask
        total = np.sum(weights, axis=1)
        total_safe = np.where(total > 0, total, 1.0)
        neighbors = xyz[ids]
        centers = np.einsum("bk,bkj->bj", weights, neighbors) / total_safe[:, None]
        delta = neighbors - centers[:, None, :]
        covariance = np.einsum(
            "bk,bki,bkj->bij", weights, delta, delta, optimize=True
        ) / total_safe[:, None, None]
        vals, vecs = np.linalg.eigh(covariance)
        eigenvalues[start:stop] = vals
        normals[start:stop] = vecs[:, :, 0]
    eligible = (
        (counts >= CONFIG.normal_minimum_neighbors)
        & np.isfinite(eigenvalues[:, 1])
        & (eigenvalues[:, 1] > 0)
        & (eigenvalues[:, 0] / eigenvalues[:, 1] <= CONFIG.normal_lambda_ratio_maximum)
    )
    return NormalField(h, normals, eigenvalues, counts, eligible)


def _angular_accept(
    anchor_normals: np.ndarray,
    candidate_normals: np.ndarray,
    vectors: np.ndarray,
    *,
    h: float,
    axial_reference: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.linalg.norm(vectors, axis=-1)
    norm_safe = np.where(norm > 0, norm, 1.0)
    unit = vectors / norm_safe[..., None]
    axial = np.abs(np.sum(vectors * anchor_normals, axis=-1))
    lateral = np.sqrt(np.maximum(0.0, norm * norm - axial * axial))
    accept = (
        (axial >= CONFIG.axial_low_h * h)
        & (axial <= CONFIG.axial_high_h * h)
        & (lateral <= CONFIG.lateral_maximum_h * h)
        & (
            np.abs(np.sum(anchor_normals * candidate_normals, axis=-1))
            >= math.cos(math.radians(CONFIG.normal_pair_maximum_degrees))
        )
        & (
            np.abs(np.sum(anchor_normals * unit, axis=-1))
            >= math.cos(math.radians(CONFIG.anchor_axis_maximum_degrees))
        )
        & (
            np.abs(np.sum(candidate_normals * unit, axis=-1))
            >= math.cos(math.radians(CONFIG.peer_axis_maximum_degrees))
        )
    )
    if axial_reference is not None:
        accept &= (
            np.abs(np.log2(np.maximum(axial, np.finfo(float).tiny) / axial_reference))
            <= CONFIG.thickness_log2_tolerance
        )
    return accept, axial, lateral


def discover_thickness(
    source: CanonicalSource, normals: NormalField
) -> tuple[float, np.ndarray, float]:
    eligible_ids = np.flatnonzero(normals.eligible).astype(np.uint32)
    if len(eligible_ids) == 0:
        return math.nan, np.empty(0, dtype=np.float64), 0.0
    xyz_bytes = np.ascontiguousarray(source.xyz[eligible_ids], dtype="<f8").view(np.uint8)
    keys = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(bytes(row)).digest()[:8], "little")
            for row in xyz_bytes.reshape(len(eligible_ids), 24)
        ),
        dtype=np.uint64,
        count=len(eligible_ids),
    )
    anchors = eligible_ids[np.lexsort((eligible_ids, keys))[: CONFIG.thickness_anchor_cap]]
    js = np.arange(CONFIG.pilot_j_minimum, CONFIG.pilot_j_maximum + 1)
    t_values = normals.h * np.exp2(js / CONFIG.pilot_octave_denominator)
    tree = cKDTree(source.xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    hits: list[float] = []
    for start in range(0, len(anchors), 128):
        block = anchors[start : start + 128]
        p = source.xyz[block]
        n = normals.normals[block]
        positions = (
            p[:, None, None, :]
            + np.asarray((1.0, -1.0))[None, :, None, None]
            * t_values[None, None, :, None]
            * n[:, None, None, :]
        )
        shape = positions.shape[:-1]
        query_distance, candidate = tree.query(
            positions.reshape(-1, 3), k=CONFIG.query_neighbor_count, workers=1
        )
        query_distance = np.asarray(query_distance).reshape(*shape, CONFIG.query_neighbor_count)
        candidate = np.asarray(candidate, dtype=np.int64).reshape(
            *shape, CONFIG.query_neighbor_count
        )
        anchor_p = p[:, None, None, None, :]
        anchor_n = n[:, None, None, None, :]
        vectors = source.xyz[candidate] - anchor_p
        candidate_n = normals.normals[candidate]
        accept, axial, lateral = _angular_accept(
            np.broadcast_to(anchor_n, vectors.shape), candidate_n, vectors, h=normals.h
        )
        accept &= query_distance <= CONFIG.query_accept_radius_h * normals.h
        accept &= normals.eligible[candidate]
        for local in range(len(block)):
            for sign in range(2):
                locations = np.argwhere(accept[local, sign])
                if len(locations) == 0:
                    continue
                ranked = sorted(
                    (
                        (
                            float(lateral[local, sign, j, k]),
                            abs(float(axial[local, sign, j, k]) - float(t_values[j])),
                            int(candidate[local, sign, j, k]),
                            float(axial[local, sign, j, k]),
                        )
                        for j, k in locations
                    ),
                    key=lambda row: (row[0], row[1], row[2]),
                )
                hits.append(ranked[0][3])
    distances = np.asarray(hits, dtype=np.float64)
    if len(distances) == 0:
        return math.nan, distances, 0.0
    bins = np.floor(
        np.log2(distances / normals.h) * CONFIG.pilot_octave_denominator
    ).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    maximum = int(np.max(counts))
    winning = int(np.min(unique[counts == maximum]))
    mode_values = distances[np.abs(bins - winning) <= 1]
    tau = float(np.median(mode_values))
    fraction = float(
        np.mean(
            np.abs(np.log2(distances / tau))
            <= CONFIG.thickness_log2_tolerance
        )
    )
    return tau, distances, fraction


def reciprocal_pairs(
    source: CanonicalSource, normals: NormalField, tau: float
) -> PairField:
    n_points = len(source.xyz)
    tree = cKDTree(source.xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    chosen = np.full(n_points, -1, dtype=np.int64)
    eligible_ids = np.flatnonzero(normals.eligible)
    for start in range(0, len(eligible_ids), CONFIG.block_size):
        block = eligible_ids[start : start + CONFIG.block_size]
        p = source.xyz[block]
        n = normals.normals[block]
        positions = p[:, None, :] + np.asarray((1.0, -1.0))[None, :, None] * tau * n[:, None, :]
        query_distance, candidate = tree.query(
            positions.reshape(-1, 3), k=CONFIG.query_neighbor_count, workers=1
        )
        query_distance = np.asarray(query_distance).reshape(
            len(block), 2, CONFIG.query_neighbor_count
        )
        candidate = np.asarray(candidate, dtype=np.int64).reshape(
            len(block), 2, CONFIG.query_neighbor_count
        )
        vectors = source.xyz[candidate] - p[:, None, None, :]
        anchor_n = np.broadcast_to(n[:, None, None, :], vectors.shape)
        candidate_n = normals.normals[candidate]
        accept, axial, lateral = _angular_accept(
            anchor_n, candidate_n, vectors, h=normals.h, axial_reference=tau
        )
        accept &= query_distance <= CONFIG.query_accept_radius_h * normals.h
        accept &= normals.eligible[candidate]
        accept &= candidate != block[:, None, None]
        normal_dot = np.abs(np.sum(anchor_n * candidate_n, axis=-1))
        score = (
            (lateral / normals.h) ** 2
            + 16.0 * np.log2(np.maximum(axial, np.finfo(float).tiny) / tau) ** 2
            + 4.0 * (1.0 - normal_dot) ** 2
        )
        score = np.where(accept, score, np.inf).reshape(len(block), -1)
        candidates_flat = candidate.reshape(len(block), -1)
        minimum = np.min(score, axis=1)
        tied = score == minimum[:, None]
        best = np.min(
            np.where(tied, candidates_flat, n_points), axis=1
        )
        valid = np.isfinite(minimum) & (best < n_points)
        chosen[block[valid]] = best[valid]
    ids = np.arange(n_points, dtype=np.int64)
    mutual = (chosen >= 0) & (chosen < n_points)
    safe = np.where(mutual, chosen, 0)
    mutual &= chosen[safe] == ids
    left = ids[mutual & (ids < chosen)]
    right = chosen[left]
    pair_points = np.column_stack((left, right)).astype(np.uint32)
    p = source.xyz[left]
    q = source.xyz[right]
    vectors = q - p
    thicknesses = np.linalg.norm(vectors, axis=1)
    midpoints = (p + q) * 0.5
    directors = vectors / np.maximum(thicknesses[:, None], np.finfo(float).tiny)
    if len(midpoints) >= 7:
        mid_tree = cKDTree(midpoints, compact_nodes=True, balanced_tree=True, copy_data=False)
        d6, _ = mid_tree.query(midpoints, k=7, workers=1)
        g = float(np.median(np.asarray(d6)[:, 6]))
    else:
        g = math.inf
    return PairField(
        h=normals.h,
        tau=tau,
        pilot_distances=np.empty(0, dtype=np.float64),
        pilot_mode_fraction=0.0,
        pair_points=pair_points,
        midpoints=midpoints,
        directors=directors,
        thicknesses=thicknesses,
        g=g,
    )


def _adjacency(vertex_count: int, edges: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(vertex_count)]
    for left, right in np.asarray(edges, dtype=np.int64):
        result[int(left)].append(int(right))
        result[int(right)].append(int(left))
    for row in result:
        row.sort()
    return result


def _two_hop(adjacency: list[list[int]], root: int) -> list[int]:
    values = {root}
    first = adjacency[root]
    values.update(first)
    for neighbor in first:
        values.update(adjacency[neighbor])
    values.discard(root)
    return sorted(values)


def _robust_active(theta: np.ndarray, lengths: np.ndarray, g: float) -> np.ndarray:
    if len(theta) == 0:
        return np.zeros(0, dtype=bool)
    scaled = lengths / g
    labels = np.full(len(theta), -1, dtype=np.int8)
    labels[(scaled >= 0.5) & (scaled < 1.0)] = 0
    labels[(scaled >= 1.0) & (scaled < 1.75)] = 1
    labels[(scaled >= 1.75) & (scaled <= 2.5 + 1e-12)] = 2
    counts = np.asarray([np.sum(labels == value) for value in range(3)], dtype=np.int64)
    merged = labels.copy()
    for value in range(3):
        if 0 < counts[value] < 256:
            candidates = [other for other in range(3) if other != value and counts[other] > 0]
            if candidates:
                target = min(candidates, key=lambda other: (abs(other - value), other))
                merged[labels == value] = target
    active = np.zeros(len(theta), dtype=bool)
    minimum = math.radians(CONFIG.active_angle_minimum_degrees)
    floor = math.radians(CONFIG.active_sigma_floor_degrees)
    for label in sorted(set(int(value) for value in merged if value >= 0)):
        mask = merged == label
        values = theta[mask]
        median = float(np.median(values))
        sigma = 1.4826 * float(np.median(np.abs(values - median)))
        threshold = max(minimum, median + CONFIG.active_robust_sigma_multiplier * max(sigma, floor))
        active[mask] = values >= threshold
    return active


def build_graph(pairs: PairField) -> GraphField:
    mid = pairs.midpoints
    directors = pairs.directors
    count = len(mid)
    if count < 2 or not math.isfinite(pairs.g) or pairs.g <= 0:
        empty_edges = np.empty((0, 2), dtype=np.uint32)
        return GraphField(
            empty_edges, np.empty(0), np.zeros(count, bool), np.zeros(count, bool),
            np.empty(0), np.empty(0, bool), empty_edges, np.empty(0), np.empty(0),
            np.empty(0, bool)
        )
    tree = cKDTree(mid, compact_nodes=True, balanced_tree=True, copy_data=False)
    k = min(CONFIG.graph_neighbor_count + 1, count)
    _distance, neighbor = tree.query(mid, k=k, workers=1)
    neighbor = np.asarray(neighbor, dtype=np.int64)
    if neighbor.ndim == 1:
        neighbor = neighbor[:, None]
    neighbor = neighbor[:, 1:]
    edge_rows: list[tuple[int, int]] = []
    for left in range(count):
        for right in neighbor[left]:
            right_i = int(right)
            if left < right_i and np.any(neighbor[right_i] == left):
                edge_rows.append((left, right_i))
    edges = np.asarray(edge_rows, dtype=np.uint32).reshape(-1, 2)
    if len(edges):
        vectors = mid[edges[:, 1]] - mid[edges[:, 0]]
        lengths = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(lengths[:, None], np.finfo(float).tiny)
        keep = (
            (lengths >= CONFIG.graph_length_low_g * pairs.g)
            & (lengths <= CONFIG.graph_length_high_g * pairs.g)
            & (lengths <= CONFIG.graph_length_maximum_h * pairs.h)
            & (
                np.abs(np.log2(pairs.thicknesses[edges[:, 0]] / pairs.thicknesses[edges[:, 1]]))
                <= CONFIG.thickness_log2_tolerance
            )
            & (
                np.maximum(
                    np.abs(np.sum(directors[edges[:, 0]] * unit, axis=1)),
                    np.abs(np.sum(directors[edges[:, 1]] * unit, axis=1)),
                )
                <= CONFIG.graph_director_edge_dot_maximum
            )
        )
        long = keep & (lengths > CONFIG.graph_long_edge_g * pairs.g)
        if np.any(long):
            geometric_mid = (mid[edges[long, 0]] + mid[edges[long, 1]]) * 0.5
            fill_distance, _ = tree.query(geometric_mid, k=1, workers=1)
            keep[np.flatnonzero(long)] &= (
                np.asarray(fill_distance) <= CONFIG.graph_midpoint_fill_g * pairs.g
            )
        edges, lengths = edges[keep], lengths[keep]
    else:
        lengths = np.empty(0, dtype=np.float64)
    adjacency = _adjacency(count, edges)
    fan = np.zeros(count, dtype=bool)
    for root in range(count):
        nodes = _two_hop(adjacency, root)
        if len(nodes) < CONFIG.fan_minimum_directions:
            continue
        director = directors[root]
        reference = np.eye(3)[int(np.argmin(np.abs(director)))]
        e1 = np.cross(director, reference)
        norm = float(np.linalg.norm(e1))
        if norm <= 0:
            continue
        e1 /= norm
        e2 = np.cross(director, e1)
        vectors = mid[nodes] - mid[root]
        vectors -= np.outer(vectors @ director, director)
        vector_norm = np.linalg.norm(vectors, axis=1)
        vectors = vectors[vector_norm > 0]
        if len(vectors) < CONFIG.fan_minimum_directions:
            continue
        angles = np.mod(np.arctan2(vectors @ e2, vectors @ e1), 2 * math.pi)
        sectors = np.floor(angles / (2 * math.pi) * CONFIG.fan_sector_count).astype(int)
        occupied = len(np.unique(np.clip(sectors, 0, CONFIG.fan_sector_count - 1)))
        ordered = np.sort(angles)
        gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2 * math.pi)))
        fan[root] = (
            occupied >= CONFIG.fan_minimum_occupied_sectors
            and float(np.max(gaps)) <= math.radians(CONFIG.fan_maximum_gap_degrees)
        )
    core = np.zeros(count, dtype=bool)
    for root in np.flatnonzero(fan):
        neighborhood = _two_hop(adjacency, int(root))
        core[root] = bool(neighborhood) and bool(np.all(fan[neighborhood]))
    if len(edges):
        left_d = directors[edges[:, 0]]
        right_d = directors[edges[:, 1]].copy()
        sign = np.where(np.sum(left_d * right_d, axis=1) < 0, -1.0, 1.0)
        right_d *= sign[:, None]
        theta = np.arctan2(
            np.linalg.norm(np.cross(left_d, right_d), axis=1),
            np.sum(left_d * right_d, axis=1),
        )
        raw_active = _robust_active(theta, lengths, pairs.g)
        core_edge_mask = core[edges[:, 0]] & core[edges[:, 1]]
        core_edges = edges[core_edge_mask]
        core_lengths = lengths[core_edge_mask]
        core_theta = theta[core_edge_mask]
        active_core = _robust_active(core_theta, core_lengths, pairs.g)
    else:
        theta = np.empty(0)
        raw_active = np.empty(0, bool)
        core_edges = np.empty((0, 2), dtype=np.uint32)
        core_lengths = np.empty(0)
        core_theta = np.empty(0)
        active_core = np.empty(0, bool)
    return GraphField(
        edges, lengths, fan, core, theta, raw_active, core_edges, core_lengths,
        core_theta, active_core
    )


def _farthest(
    adjacency: dict[int, list[tuple[int, float]]], start: int
) -> tuple[int, float]:
    distance = {start: 0.0}
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        value, node = heapq.heappop(queue)
        if value != distance[node]:
            continue
        for neighbor, weight in adjacency.get(node, []):
            candidate = value + weight
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    maximum = max(distance.values(), default=0.0)
    node = min(key for key, value in distance.items() if value == maximum)
    return node, maximum


def components_from_graph(
    pairs: PairField, graph: GraphField
) -> list[Component]:
    active_indices = np.flatnonzero(graph.active_core)
    if len(active_indices) == 0:
        return []
    active_edges = graph.core_edges[active_indices]
    active_lengths = graph.core_lengths[active_indices]
    edge_by_vertex: dict[int, list[int]] = {}
    for local, (left, right) in enumerate(active_edges):
        edge_by_vertex.setdefault(int(left), []).append(local)
        edge_by_vertex.setdefault(int(right), []).append(local)
    unseen = set(range(len(active_edges)))
    groups: list[np.ndarray] = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        group: set[int] = set()
        while stack:
            edge_id = stack.pop()
            if edge_id not in unseen:
                continue
            unseen.remove(edge_id)
            group.add(edge_id)
            for vertex in active_edges[edge_id]:
                stack.extend(edge_by_vertex[int(vertex)])
        groups.append(np.asarray(sorted(group), dtype=np.int64))
    results: list[Component] = []
    centers = (pairs.midpoints[active_edges[:, 0]] + pairs.midpoints[active_edges[:, 1]]) * 0.5
    center_tree = cKDTree(centers, compact_nodes=True, balanced_tree=True, copy_data=False)
    linear = np.zeros(len(active_edges), dtype=bool)
    for index, center in enumerate(centers):
        ids = center_tree.query_ball_point(center, CONFIG.line_support_radius_g * pairs.g)
        if len(ids) < 3:
            continue
        values = centers[np.asarray(ids, dtype=np.int64)]
        covariance = np.cov(values, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        ratio = float(eigenvalues[2] / (eigenvalues[1] + (0.05 * pairs.g) ** 2))
        linear[index] = ratio >= CONFIG.line_support_ratio_minimum
    for component_id, local_ids in enumerate(groups):
        component_edges = active_edges[local_ids]
        component_lengths = active_lengths[local_ids]
        midpoint_ids = np.unique(component_edges).astype(np.uint32)
        weighted: dict[int, list[tuple[int, float]]] = {}
        for (left, right), length in zip(component_edges, component_lengths):
            weighted.setdefault(int(left), []).append((int(right), float(length)))
            weighted.setdefault(int(right), []).append((int(left), float(length)))
        start = int(np.min(midpoint_ids))
        endpoint, _ = _farthest(weighted, start)
        _other, diameter = _farthest(weighted, endpoint)
        fraction = float(np.mean(linear[local_ids]))
        retained = (
            len(component_edges) >= CONFIG.component_minimum_edges
            and diameter >= CONFIG.component_minimum_diameter_g * pairs.g
            and fraction >= CONFIG.line_support_fraction_minimum
        )
        results.append(
            Component(
                component_id=component_id,
                edge_indices=active_indices[local_ids].astype(np.uint32),
                midpoint_ids=midpoint_ids,
                edge_count=len(component_edges),
                diameter=diameter,
                linear_fraction=fraction,
                retained=retained,
            )
        )
    return results


def _empty_pair(tau: float, pilot: np.ndarray, fraction: float) -> PairField:
    return PairField(
        h=math.nan,
        tau=tau,
        pilot_distances=pilot,
        pilot_mode_fraction=fraction,
        pair_points=np.empty((0, 2), dtype=np.uint32),
        midpoints=np.empty((0, 3), dtype=np.float64),
        directors=np.empty((0, 3), dtype=np.float64),
        thicknesses=np.empty(0, dtype=np.float64),
        g=math.inf,
    )


def _semantic(
    source: CanonicalSource,
    normals: NormalField,
    pairs: PairField | None,
    graph: GraphField | None,
    components: Sequence[Component],
    decision: str,
) -> dict[str, Any]:
    pair_count = 0 if pairs is None else len(pairs.pair_points)
    core_count = 0 if graph is None else int(np.sum(graph.core))
    core_edge_count = 0 if graph is None else len(graph.core_edges)
    active_count = 0 if graph is None else int(np.sum(graph.active_core))
    retained = [row for row in components if row.retained]
    pair_hash = sha_arrays(
        b"AMHC_V1_RECIPROCAL_PAIR_IDS\0",
        (np.empty((0, 2), dtype=np.uint32) if pairs is None else pairs.pair_points,),
    )
    core_hash = sha_arrays(
        b"AMHC_V1_CORE_MIDPOINT_IDS\0",
        (np.empty(0, dtype=np.uint32) if graph is None else np.flatnonzero(graph.core).astype(np.uint32),),
    )
    active_hash = sha_arrays(
        b"AMHC_V1_ACTIVE_CORE_EDGE_IDS\0",
        (np.empty(0, dtype=np.uint32) if graph is None else np.flatnonzero(graph.active_core).astype(np.uint32),),
    )
    component_membership = [
        [int(row.component_id), [int(value) for value in row.midpoint_ids]]
        for row in retained[: CONFIG.component_cap]
    ]
    component_hash = sha_json(component_membership)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": CONFIG.attempt_id,
        "architecture_token": CONFIG.architecture_token,
        "authorization_token": CONFIG.authorization_token,
        "configuration_sha256": CONFIG_SHA256,
        "source": {
            "raw_source_sha256": source.raw_source_sha256,
            "raw_source_bytes": source.raw_source_bytes,
            "canonical_point_count": len(source.xyz),
            "sampled_coordinate_sha256": source.sampled_coordinate_sha256,
            "canonical_table_sha256": source.canonical_table_sha256,
        },
        "scales": {
            "h": normals.h,
            "tau": None if pairs is None or not math.isfinite(pairs.tau) else pairs.tau,
            "g": None if pairs is None or not math.isfinite(pairs.g) else pairs.g,
        },
        "counts": {
            "normal_eligible": int(np.sum(normals.eligible)),
            "pilot_hits": 0 if pairs is None else len(pairs.pilot_distances),
            "reciprocal_pairs": pair_count,
            "participating_source_points": pair_count * 2,
            "fan_interior": 0 if graph is None else int(np.sum(graph.fan_interior)),
            "core_midpoints": core_count,
            "graph_edges": 0 if graph is None else len(graph.edges),
            "core_edges": core_edge_count,
            "active_core_edges": active_count,
            "components": len(components),
            "retained_components": len(retained),
        },
        "fractions": {
            "pilot_mode": 0.0 if pairs is None else pairs.pilot_mode_fraction,
            "pair_participation": (
                0.0
                if int(np.sum(normals.eligible)) == 0
                else (2.0 * pair_count / int(np.sum(normals.eligible)))
            ),
            "core_coverage": 0.0 if pair_count == 0 else core_count / pair_count,
            "active_core_edges": 0.0 if core_edge_count == 0 else active_count / core_edge_count,
        },
        "hashes": {
            "reciprocal_pairs": pair_hash,
            "core_midpoints": core_hash,
            "active_core_edges": active_hash,
            "component_membership": component_hash,
        },
        "component_table": [
            {
                "component_id": int(row.component_id),
                "edge_count": int(row.edge_count),
                "midpoint_count": len(row.midpoint_ids),
                "diameter_over_g": (
                    None if pairs is None or not math.isfinite(pairs.g) or pairs.g <= 0
                    else row.diameter / pairs.g
                ),
                "linear_fraction": row.linear_fraction,
                "retained": row.retained,
                "membership_sha256": sha_arrays(
                    b"AMHC_V1_COMPONENT_MEMBERSHIP\0", (row.midpoint_ids,)
                ),
            }
            for row in components[: CONFIG.component_cap]
        ],
        "decision": decision,
        "source_point_serialized": False,
        "target_gold_cad_benchmark_opened": False,
        "runtime_main_production_opened": False,
    }
    payload["semantic_sha256"] = sha_json(payload)
    return payload


def evaluate(source: CanonicalSource) -> Evaluation:
    normals = estimate_normals(source)
    tau, pilot, mode_fraction = discover_thickness(source, normals)
    if (
        len(pilot) < CONFIG.pilot_minimum_hits
        or not math.isfinite(tau)
        or mode_fraction < CONFIG.pilot_mode_fraction_minimum
    ):
        pairs = _empty_pair(tau, pilot, mode_fraction)
        decision = "AMHC_V1_PAIRABILITY_ARCHITECTURE_STOP"
        semantic = _semantic(source, normals, pairs, None, [], decision)
        return Evaluation(source, normals, pairs, None, [], decision, semantic)
    pairs = reciprocal_pairs(source, normals, tau)
    pairs.pilot_distances = pilot
    pairs.pilot_mode_fraction = mode_fraction
    participation = 2.0 * len(pairs.pair_points) / max(1, int(np.sum(normals.eligible)))
    if (
        len(pairs.pair_points) < CONFIG.reciprocal_minimum_pairs
        or participation < CONFIG.reciprocal_participation_minimum
        or not math.isfinite(pairs.g)
        or pairs.g > CONFIG.midpoint_sixth_neighbor_maximum_h * normals.h
    ):
        decision = "AMHC_V1_PAIRABILITY_ARCHITECTURE_STOP"
        semantic = _semantic(source, normals, pairs, None, [], decision)
        return Evaluation(source, normals, pairs, None, [], decision, semantic)
    graph = build_graph(pairs)
    core_count = int(np.sum(graph.core))
    core_fraction = core_count / max(1, len(pairs.pair_points))
    if (
        core_count < CONFIG.core_minimum_midpoints
        or core_fraction < CONFIG.core_coverage_minimum
    ):
        decision = "AMHC_V1_CLOSED_FAN_CORE_ARCHITECTURE_STOP"
        semantic = _semantic(source, normals, pairs, graph, [], decision)
        return Evaluation(source, normals, pairs, graph, [], decision, semantic)
    components = components_from_graph(pairs, graph)
    active_fraction = float(np.mean(graph.active_core)) if len(graph.active_core) else 0.0
    retained = [row for row in components if row.retained]
    if active_fraction > CONFIG.active_fraction_maximum:
        decision = "AMHC_V1_NONSELECTIVE_HINGE_CURRENT_ARCHITECTURE_STOP"
    elif not retained:
        decision = "AMHC_V1_NO_LINE_SUPPORTED_HINGE_ARCHITECTURE_STOP"
    else:
        decision = "AMHC_V1_SOURCE_ONLY_MACHINE_PREFLIGHT_PASS"
    semantic = _semantic(source, normals, pairs, graph, components, decision)
    return Evaluation(source, normals, pairs, graph, components, decision, semantic)


def normalized_semantic(bundle: dict[str, Any]) -> dict[str, Any]:
    excluded = {"role", "wall_seconds", "peak_rss_bytes"}
    return {key: value for key, value in bundle.items() if key not in excluded}


def visual_scratch_arrays(evaluation: Evaluation) -> dict[str, np.ndarray]:
    source = evaluation.source
    pairs = evaluation.pairs
    graph = evaluation.graph
    if pairs is None:
        midpoints = np.empty((0, 3), dtype=np.float64)
    else:
        midpoints = pairs.midpoints
    if graph is None or pairs is None:
        raw_centers = np.empty((0, 3), dtype=np.float64)
        rejected_centers = np.empty((0, 3), dtype=np.float64)
        retained_centers = np.empty((0, 3), dtype=np.float64)
    else:
        raw_edges = graph.edges[graph.raw_active]
        raw_centers = (
            (pairs.midpoints[raw_edges[:, 0]] + pairs.midpoints[raw_edges[:, 1]]) * 0.5
            if len(raw_edges) else np.empty((0, 3), dtype=np.float64)
        )
        rejected_mask = graph.raw_active & ~(
            graph.core[graph.edges[:, 0]] & graph.core[graph.edges[:, 1]]
        )
        rejected_edges = graph.edges[rejected_mask]
        rejected_centers = (
            (pairs.midpoints[rejected_edges[:, 0]] + pairs.midpoints[rejected_edges[:, 1]]) * 0.5
            if len(rejected_edges) else np.empty((0, 3), dtype=np.float64)
        )
        retained_ids = np.concatenate(
            [row.edge_indices for row in evaluation.components if row.retained]
        ) if any(row.retained for row in evaluation.components) else np.empty(0, dtype=np.uint32)
        retained_edges = graph.core_edges[retained_ids] if len(retained_ids) else np.empty((0, 2), dtype=np.uint32)
        retained_centers = (
            (pairs.midpoints[retained_edges[:, 0]] + pairs.midpoints[retained_edges[:, 1]]) * 0.5
            if len(retained_edges) else np.empty((0, 3), dtype=np.float64)
        )
    return {
        "source_xyz": np.ascontiguousarray(source.xyz, dtype="<f8"),
        "midpoints": np.ascontiguousarray(midpoints, dtype="<f8"),
        "raw_high_turn_centers": np.ascontiguousarray(raw_centers, dtype="<f8"),
        "rejected_noncore_centers": np.ascontiguousarray(rejected_centers, dtype="<f8"),
        "retained_centers": np.ascontiguousarray(retained_centers, dtype="<f8"),
        "pilot_distances": np.ascontiguousarray(
            np.empty(0) if pairs is None else pairs.pilot_distances, dtype="<f8"
        ),
        "normal_eligible": np.ascontiguousarray(evaluation.normals.eligible, dtype=np.uint8),
        "pair_point_ids": np.ascontiguousarray(
            np.empty((0, 2), dtype=np.uint32) if pairs is None else pairs.pair_points,
            dtype="<u4",
        ),
        "core_mask": np.ascontiguousarray(
            np.empty(0, dtype=np.uint8) if graph is None else graph.core,
            dtype=np.uint8,
        ),
    }
