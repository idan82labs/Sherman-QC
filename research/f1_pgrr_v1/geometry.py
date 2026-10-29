"""Deterministic target-blind PGRR candidate construction.

This producer module accepts only the raw source and immutable configuration.
It deliberately has no target-ledger, case, filename-feature, gold, CAD, or
benchmark input.  Scientific evidence contains hashes and aggregates only;
packed membership bitsets are written exclusively to ephemeral internal
scratch for the post-commit handoff process.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from research.f1_ciip_v1 import partition, primary_parser
from .config import CONFIG, CONFIG_SHA256, PGRRConfig, canonical_json_bytes, sha_json


SOURCE_SHA256 = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
V4_CONTAINER_SHA256 = "7065b3b77a15435d8654d047d9afb50e954e8997375d94e9a3e1606141477817"


class PGRRStop(RuntimeError):
    def __init__(self, token: str, message: str) -> None:
        super().__init__(message)
        self.token = token


@dataclass(slots=True)
class Sample:
    xyz: np.ndarray
    point_ids: np.ndarray
    multiplicities: np.ndarray
    canonical_point_order_sha256: str
    unique_record_count: int


@dataclass(slots=True)
class NormalField:
    neighbors: np.ndarray
    neighbor_distances: np.ndarray
    normal24: np.ndarray
    normal48: np.ndarray
    mean_normal: np.ndarray
    stable: np.ndarray
    conditioned24: np.ndarray
    conditioned48: np.ndarray
    eig24: np.ndarray
    eig48: np.ndarray
    density_scale: float
    knn_commitment_sha256: str
    normal_commitment_sha256: str


@dataclass(slots=True)
class Ribbon:
    pole_id: int
    component_min_point_id: str
    member_indices: np.ndarray
    membership_sha256: str
    axis: np.ndarray
    theta_start: float
    theta_span: float
    bin_count: int
    bin_indices: np.ndarray
    cut_successor_point_id: str
    maximum_gap: float
    q05: float
    q95: float
    occupancy_mask: int
    column_coverage: tuple[int, ...]
    ruling_coherent_bin_count: int
    ruling_axis_maximum_angle: float
    ruling_minimum_elongation: float
    bounding_box_min: np.ndarray
    bounding_box_max: np.ndarray

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.pole_id, self.component_min_point_id, self.membership_sha256)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_chunks(tag: bytes, arrays: Iterable[np.ndarray | bytes]) -> str:
    digest = hashlib.sha256(tag)
    for value in arrays:
        raw = value if isinstance(value, bytes) else np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_v4_sample(raw_source: Path, *, reverse_input: bool = False) -> Sample:
    if sha_file(raw_source) != SOURCE_SHA256:
        raise PGRRStop("PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "raw source hash mismatch")
    descriptor = os.open(raw_source, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    if parsed.diagnostics.raw_ply_sha256 != SOURCE_SHA256:
        raise PGRRStop("PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "parser source binding mismatch")
    xyz = parsed.xyz[::-1] if reverse_input else parsed.xyz
    packed = np.ascontiguousarray(xyz).view("V24").reshape(len(xyz))
    _unique, first, counts = np.unique(packed, return_index=True, return_counts=True)
    unique_xyz = np.ascontiguousarray(xyz[first], dtype="<f8")
    sampled = partition.canonical_sample(
        unique_xyz,
        np.ascontiguousarray(counts, dtype=np.uint32),
        sample_cap=CONFIG.sample_cap,
    )
    if len(sampled.xyz) != CONFIG.sample_cap:
        raise PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP",
            "V4 canonical sample is not exactly 200,000 points",
        )
    order_hash = _sha_chunks(
        b"PGRR_V1_V4_CANONICAL_POINT_ORDER\0",
        (np.asarray(sampled.point_ids, dtype=np.uint8),),
    )
    return Sample(
        xyz=np.asarray(sampled.xyz, dtype=np.float64),
        point_ids=np.asarray(sampled.point_ids, dtype=np.uint8),
        multiplicities=np.asarray(sampled.multiplicities, dtype=np.uint32),
        canonical_point_order_sha256=order_hash,
        unique_record_count=int(sampled.unique_record_count),
    )


def _exact_neighbor_rows(
    xyz: np.ndarray, point_ids: np.ndarray, *, query_count: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xyz)
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    distances, indices = tree.query(xyz, k=query_count, workers=1)
    neighbors = np.empty((n, 48), dtype=np.uint32)
    neighbor_distances = np.empty((n, 48), dtype=np.float64)
    pid_void = point_ids.view("V32").reshape(n)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        for local, row_id in enumerate(range(start, end)):
            candidates = np.asarray(indices[row_id], dtype=np.int64)
            candidates = candidates[candidates != row_id]
            delta = xyz[candidates] - xyz[row_id]
            squared = np.einsum("ij,ij->i", delta, delta, optimize=False)
            candidate_ids = pid_void[candidates]
            order = np.lexsort((candidate_ids, squared))
            candidates = candidates[order]
            squared = squared[order]
            if len(candidates) < 48:
                raise PGRRStop(
                    "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "insufficient neighbours"
                )
            boundary = squared[47]
            if len(squared) > 48 and squared[48] == boundary:
                tied = np.asarray(
                    tree.query_ball_point(
                        xyz[row_id], np.nextafter(math.sqrt(boundary), math.inf)
                    ),
                    dtype=np.int64,
                )
                tied = tied[tied != row_id]
                delta = xyz[tied] - xyz[row_id]
                tied_squared = np.einsum("ij,ij->i", delta, delta, optimize=False)
                tied_order = np.lexsort((pid_void[tied], tied_squared))
                candidates = tied[tied_order]
                squared = tied_squared[tied_order]
            neighbors[row_id] = candidates[:48]
            neighbor_distances[row_id] = np.sqrt(squared[:48])
    return neighbors, neighbor_distances


def _analytic_eigh_smallest(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed analytic symmetric 3x3 eigensolver for producer measurements."""
    a = np.asarray(covariance, dtype=np.float64)
    q = np.trace(a, axis1=1, axis2=2) / 3.0
    centered = a.copy()
    centered[:, 0, 0] -= q
    centered[:, 1, 1] -= q
    centered[:, 2, 2] -= q
    p2 = np.sum(centered * centered, axis=(1, 2)) / 6.0
    p = np.sqrt(np.maximum(p2, 0.0))
    eig = np.empty((len(a), 3), dtype=np.float64)
    nondegenerate = p > 0.0
    eig[~nondegenerate] = q[~nondegenerate, None]
    if np.any(nondegenerate):
        b = centered[nondegenerate] / p[nondegenerate, None, None]
        det = (
            b[:, 0, 0] * (b[:, 1, 1] * b[:, 2, 2] - b[:, 1, 2] * b[:, 2, 1])
            - b[:, 0, 1] * (b[:, 1, 0] * b[:, 2, 2] - b[:, 1, 2] * b[:, 2, 0])
            + b[:, 0, 2] * (b[:, 1, 0] * b[:, 2, 1] - b[:, 1, 1] * b[:, 2, 0])
        )
        phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
        high = q[nondegenerate] + 2.0 * p[nondegenerate] * np.cos(phi)
        low = q[nondegenerate] + 2.0 * p[nondegenerate] * np.cos(phi + 2.0 * math.pi / 3.0)
        middle = 3.0 * q[nondegenerate] - low - high
        eig[nondegenerate] = np.stack((low, middle, high), axis=1)
    eig.sort(axis=1)
    shifted = a - eig[:, 0, None, None] * np.eye(3, dtype=np.float64)[None, :, :]
    c01 = np.cross(shifted[:, 0], shifted[:, 1])
    c02 = np.cross(shifted[:, 0], shifted[:, 2])
    c12 = np.cross(shifted[:, 1], shifted[:, 2])
    candidates = np.stack((c01, c02, c12), axis=1)
    norms2 = np.sum(candidates * candidates, axis=2)
    choose = np.argmax(norms2, axis=1)
    vector = candidates[np.arange(len(a)), choose]
    norm = np.linalg.norm(vector, axis=1)
    fallback = norm == 0.0
    if np.any(fallback):
        vector[fallback] = np.array([1.0, 0.0, 0.0])
        norm[fallback] = 1.0
    vector /= norm[:, None]
    return eig, vector


def _covariance_field(
    xyz: np.ndarray, neighbors: np.ndarray, neighbor_count: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xyz)
    eigenvalues = np.empty((n, 3), dtype=np.float64)
    normals = np.empty((n, 3), dtype=np.float64)
    take = neighbor_count - 1
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        base = xyz[start:end]
        total = base.copy()
        outer = base[:, :, None] * base[:, None, :]
        for offset in range(take):
            values = xyz[neighbors[start:end, offset]]
            total += values
            outer += values[:, :, None] * values[:, None, :]
        mean = total / float(neighbor_count)
        covariance = outer / float(neighbor_count) - mean[:, :, None] * mean[:, None, :]
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
        values, vectors = _analytic_eigh_smallest(covariance)
        eigenvalues[start:end] = values
        normals[start:end] = vectors
    return eigenvalues, normals


def build_normal_field(sample: Sample) -> NormalField:
    neighbors, distances = _exact_neighbor_rows(sample.xyz, sample.point_ids)
    eig24, normal24 = _covariance_field(sample.xyz, neighbors, CONFIG.normal_k_small)
    eig48, normal48 = _covariance_field(sample.xyz, neighbors, CONFIG.normal_k_large)
    conditioned24 = (eig24[:, 2] > 0.0) & (
        eig24[:, 1] >= CONFIG.normal_condition_ratio * eig24[:, 2]
    )
    conditioned48 = (eig48[:, 2] > 0.0) & (
        eig48[:, 1] >= CONFIG.normal_condition_ratio * eig48[:, 2]
    )
    dot = np.einsum("ij,ij->i", normal24, normal48, optimize=False)
    agreement = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    stable = conditioned24 & conditioned48 & (agreement <= CONFIG.normal_scale_agreement_radians)
    flip = dot < 0.0
    aligned48 = normal48.copy()
    aligned48[flip] *= -1.0
    mean = normal24 + aligned48
    mean_norm = np.linalg.norm(mean, axis=1)
    stable &= mean_norm > 0.0
    mean[mean_norm > 0.0] /= mean_norm[mean_norm > 0.0, None]
    density = np.sort(distances[:, CONFIG.density_neighbor_rank - 1], kind="stable")
    density_scale = float(density[(len(density) - 1) // 2])
    if not math.isfinite(density_scale) or density_scale <= 0.0:
        raise PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "source-density scale invalid"
        )
    return NormalField(
        neighbors=neighbors,
        neighbor_distances=distances,
        normal24=normal24,
        normal48=normal48,
        mean_normal=mean,
        stable=stable,
        conditioned24=conditioned24,
        conditioned48=conditioned48,
        eig24=eig24,
        eig48=eig48,
        density_scale=density_scale,
        knn_commitment_sha256=_sha_chunks(
            b"PGRR_V1_KNN\0", (neighbors, distances.astype("<f8"))
        ),
        normal_commitment_sha256=_sha_chunks(
            b"PGRR_V1_NORMAL_FIELD\0",
            (
                normal24.astype("<f8"),
                normal48.astype("<f8"),
                stable.astype(np.uint8),
            ),
        ),
    )


_ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def projective_poles() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    points = [np.asarray(row, dtype=np.float64) / np.linalg.norm(row) for row in vertices]
    faces = list(_ICO_FACES)
    for _level in range(CONFIG.icosphere_subdivision_level):
        midpoint: dict[tuple[int, int], int] = {}

        def mid(left: int, right: int) -> int:
            key = (min(left, right), max(left, right))
            if key not in midpoint:
                value = points[left] + points[right]
                value /= np.linalg.norm(value)
                midpoint[key] = len(points)
                points.append(value)
            return midpoint[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined
    if len(points) != 642:
        raise ValueError("fixed level-3 icosphere does not contain 642 vertices")
    canonical = []
    for raw in points:
        value = np.asarray(raw, dtype=np.float64)
        for coordinate in value:
            if coordinate != 0.0:
                if coordinate < 0.0:
                    value = -value
                break
        canonical.append(value)
    ordered = sorted(canonical, key=lambda row: tuple(float(value).hex() for value in row))
    poles: list[np.ndarray] = []
    for value in ordered:
        if not poles or not np.any(np.abs(np.asarray(poles) @ value) >= 1.0 - 8.0 * np.finfo(float).eps):
            poles.append(value)
    if len(poles) != CONFIG.projective_pole_count:
        raise ValueError(f"projective pole count mismatch: {len(poles)}")
    return np.ascontiguousarray(poles, dtype=np.float64)


def pole_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(3, dtype=np.float64)
    dots = np.abs(basis @ axis)
    choice = int(np.argmin(dots))
    u = basis[choice] - float(np.dot(axis, basis[choice])) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _projective_theta(normal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    u, v = pole_basis(axis)
    return np.mod(np.arctan2(normal @ v, normal @ u), math.pi)


def _canonical_interval(
    theta: np.ndarray, member_indices: np.ndarray, point_ids: np.ndarray
) -> tuple[np.ndarray, float, float, str, float]:
    order = np.lexsort((point_ids[member_indices].view("V32").reshape(len(member_indices)), theta))
    sorted_theta = theta[order]
    sorted_members = member_indices[order]
    gaps = np.empty(len(sorted_theta), dtype=np.float64)
    gaps[:-1] = sorted_theta[1:] - sorted_theta[:-1]
    gaps[-1] = sorted_theta[0] + math.pi - sorted_theta[-1]
    maximum = float(np.max(gaps))
    candidates = np.flatnonzero(gaps == maximum)
    successor_positions = (candidates + 1) % len(sorted_theta)
    successor_ids = point_ids[sorted_members[successor_positions]].view("V32").reshape(len(candidates))
    chosen = int(candidates[int(np.argsort(successor_ids, kind="stable")[0])])
    successor = (chosen + 1) % len(sorted_theta)
    start = float(sorted_theta[successor])
    unwrapped = np.mod(theta - start, math.pi)
    span = float(math.pi - maximum)
    successor_id = bytes(point_ids[sorted_members[successor]]).hex()
    return unwrapped, start, span, successor_id, maximum


def _mutual_graph(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(neighbors)
    source = np.repeat(np.arange(n, dtype=np.uint64), CONFIG.mutual_neighbor_count)
    target = neighbors[:, : CONFIG.mutual_neighbor_count].reshape(-1).astype(np.uint64)
    directed = source * np.uint64(n) + target
    order = np.argsort(directed, kind="stable")
    directed_sorted = directed[order]
    reverse = target * np.uint64(n) + source
    positions = np.searchsorted(directed_sorted, reverse)
    mutual = (positions < len(directed_sorted)) & (directed_sorted[np.minimum(positions, len(directed_sorted) - 1)] == reverse)
    keep = mutual & (source < target)
    left = source[keep].astype(np.uint32)
    right = target[keep].astype(np.uint32)
    if len(left) > CONFIG.mutual_edge_maximum:
        raise PGRRStop(
            "PGRR_V1_RESOURCE_CONTRACT_STOP", "mutual-12NN edge cap exceeded"
        )
    edge_order = np.lexsort((right, left))
    return left[edge_order], right[edge_order]


def _component_labels_union_find(
    eligible: np.ndarray, left: np.ndarray, right: np.ndarray, edge_mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    vertices = np.flatnonzero(eligible).astype(np.int64)
    if len(vertices) == 0:
        return []
    parent = np.arange(len(eligible), dtype=np.int32)
    rank = np.zeros(len(eligible), dtype=np.uint8)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root

    retained_indices = np.flatnonzero(edge_mask)
    for edge_index in retained_indices:
        a, b = int(left[edge_index]), int(right[edge_index])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb] or (rank[ra] == rank[rb] and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in vertices:
        groups[find(int(vertex))].append(int(vertex))
    edge_groups: dict[int, list[int]] = defaultdict(list)
    for edge_index in retained_indices:
        edge_groups[find(int(left[edge_index]))].append(int(edge_index))
    return [
        (np.asarray(members, dtype=np.uint32), np.asarray(edge_groups[root], dtype=np.uint32))
        for root, members in sorted(groups.items(), key=lambda item: min(item[1]))
    ]


def _membership_hash(member_indices: np.ndarray, point_ids: np.ndarray) -> str:
    ids = np.ascontiguousarray(point_ids[member_indices])
    order = np.argsort(ids.view("V32").reshape(len(ids)), kind="stable")
    return _sha_chunks(b"PGRR_V1_MEMBERSHIP\0", (ids[order],))


def _component_id(member_indices: np.ndarray, point_ids: np.ndarray) -> str:
    ids = point_ids[member_indices].view("V32").reshape(len(member_indices))
    local = int(np.argsort(ids, kind="stable")[0])
    return bytes(point_ids[member_indices[local]]).hex()


def _evaluate_component(
    *,
    pole_id: int,
    axis: np.ndarray,
    members: np.ndarray,
    component_edge_indices: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    theta_all: np.ndarray,
    sample: Sample,
    field: NormalField,
) -> tuple[Ribbon | None, str, dict[str, Any]]:
    if len(members) < CONFIG.component_minimum_points:
        return None, "MEMBER_COUNT", {"member_count": len(members)}
    theta = theta_all[members]
    unwrapped, start, span, successor_id, maximum_gap = _canonical_interval(
        theta, members, sample.point_ids
    )
    if span < CONFIG.component_minimum_span_radians:
        return None, "ARC_SPAN_LOW", {"member_count": len(members), "span": span}
    if span > CONFIG.component_maximum_span_radians:
        return None, "ARC_SPAN_HIGH", {"member_count": len(members), "span": span}
    bin_count = int(math.ceil(span / CONFIG.nominal_angular_bin_width_radians))
    if not 4 <= bin_count <= CONFIG.angular_bin_maximum:
        return None, "ANGULAR_BIN_COUNT", {"member_count": len(members), "span": span}
    width = span / float(bin_count)
    local_bins = np.minimum(np.floor(unwrapped / width).astype(np.int32), bin_count - 1)
    counts = np.bincount(local_bins, minlength=bin_count)
    if np.any(counts < CONFIG.minimum_points_per_angular_bin):
        return None, "ANGULAR_BIN_SUPPORT", {"member_count": len(members), "span": span}
    if maximum_gap > CONFIG.edge_maximum_angular_gap_radians:
        return None, "ANGULAR_DISCONNECTION", {"member_count": len(members), "span": span}
    bin_by_vertex = np.full(len(sample.xyz), -1, dtype=np.int16)
    bin_by_vertex[members] = local_bins
    boundary_edges = np.zeros(bin_count - 1, dtype=np.int32)
    for edge_index in component_edge_indices:
        a, b = int(left[edge_index]), int(right[edge_index])
        ba, bb = int(bin_by_vertex[a]), int(bin_by_vertex[b])
        if abs(ba - bb) == 1:
            boundary_edges[min(ba, bb)] += 1
    if np.any(boundary_edges == 0):
        return None, "ANGULAR_BOUNDARY_EDGE", {"member_count": len(members), "span": span}
    z = sample.xyz[members] @ axis
    sorted_z = np.sort(z, kind="stable")
    q05 = float(sorted_z[int(math.floor(0.05 * (len(sorted_z) - 1)))])
    q95 = float(sorted_z[int(math.floor(0.95 * (len(sorted_z) - 1)))])
    if q95 - q05 < CONFIG.longitudinal_scale_multiple * field.density_scale:
        return None, "LONGITUDINAL_SPAN", {"member_count": len(members), "span": span}
    within = (z >= q05) & (z <= q95)
    column_width = (q95 - q05) / CONFIG.longitudinal_column_count
    columns = np.minimum(
        np.floor((z[within] - q05) / column_width).astype(np.int32),
        CONFIG.longitudinal_column_count - 1,
    )
    cell_counts = np.zeros((bin_count, CONFIG.longitudinal_column_count), dtype=np.int32)
    np.add.at(cell_counts, (local_bins[within], columns), 1)
    occupied = cell_counts >= 2
    minimum_bins = int(math.ceil(3.0 * bin_count / 4.0))
    column_coverage = tuple(int(value) for value in np.sum(occupied, axis=0))
    if int(np.sum(np.sum(occupied, axis=0) >= minimum_bins)) < CONFIG.minimum_longitudinal_columns:
        return None, "LONGITUDINAL_OCCUPANCY", {"member_count": len(members), "span": span}
    if np.any(np.sum(occupied, axis=1) < CONFIG.minimum_bin_column_occupancy):
        return None, "ANGULAR_LONGITUDINAL_COVERAGE", {"member_count": len(members), "span": span}
    coherent = 0
    maximum_angle = 0.0
    minimum_elongation = math.inf
    for angular_bin in range(bin_count):
        selected = members[(local_bins == angular_bin) & within]
        values = sample.xyz[selected]
        centroid = np.sum(values, axis=0) / float(len(values))
        delta = values - centroid
        covariance = (delta.T @ delta) / float(len(values))
        eigenvalues, _small = _analytic_eigh_smallest(covariance[None, :, :])
        shifted = covariance - eigenvalues[0, 2] * np.eye(3)
        candidates = np.stack(
            (
                np.cross(shifted[0], shifted[1]),
                np.cross(shifted[0], shifted[2]),
                np.cross(shifted[1], shifted[2]),
            )
        )
        norms = np.linalg.norm(candidates, axis=1)
        principal = candidates[int(np.argmax(norms))]
        if np.max(norms) > 0.0:
            principal /= np.linalg.norm(principal)
        angle = math.acos(float(np.clip(abs(np.dot(principal, axis)), 0.0, 1.0)))
        elongation = float(eigenvalues[0, 2] / max(eigenvalues[0, 1], np.finfo(float).tiny))
        maximum_angle = max(maximum_angle, angle)
        minimum_elongation = min(minimum_elongation, elongation)
        if (
            angle <= CONFIG.ruling_axis_radians
            and eigenvalues[0, 2] > 0.0
            and eigenvalues[0, 2] >= CONFIG.ruling_elongation_ratio * eigenvalues[0, 1]
        ):
            coherent += 1
    if coherent < minimum_bins:
        return None, "RULING_COHERENCE", {"member_count": len(members), "span": span}
    membership_hash = _membership_hash(members, sample.point_ids)
    occupancy_mask = 0
    for angular_bin in range(bin_count):
        for column in range(CONFIG.longitudinal_column_count):
            if occupied[angular_bin, column]:
                occupancy_mask |= 1 << (angular_bin * CONFIG.longitudinal_column_count + column)
    points = sample.xyz[members]
    return Ribbon(
        pole_id=pole_id,
        component_min_point_id=_component_id(members, sample.point_ids),
        member_indices=np.asarray(members, dtype=np.uint32),
        membership_sha256=membership_hash,
        axis=np.asarray(axis, dtype=np.float64),
        theta_start=start,
        theta_span=span,
        bin_count=bin_count,
        bin_indices=local_bins,
        cut_successor_point_id=successor_id,
        maximum_gap=maximum_gap,
        q05=q05,
        q95=q95,
        occupancy_mask=occupancy_mask,
        column_coverage=column_coverage,
        ruling_coherent_bin_count=coherent,
        ruling_axis_maximum_angle=maximum_angle,
        ruling_minimum_elongation=minimum_elongation,
        bounding_box_min=np.min(points, axis=0),
        bounding_box_max=np.max(points, axis=0),
    ), "PASS", {"member_count": len(members), "span": span}


def _bitset(indices: np.ndarray, point_count: int) -> np.ndarray:
    bits = np.zeros((point_count + 7) // 8, dtype=np.uint8)
    np.bitwise_or.at(bits, indices.astype(np.int64) // 8, 1 << (indices.astype(np.int64) % 8))
    return bits


def _deduplicate(ribbons: Sequence[Ribbon], point_count: int) -> tuple[list[Ribbon], list[list[int]]]:
    count = len(ribbons)
    parent = np.arange(count, dtype=np.int32)
    bitsets = [_bitset(row.member_indices, point_count) for row in ribbons]

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    for left_index in range(count):
        left = ribbons[left_index]
        for right_index in range(left_index + 1, count):
            right = ribbons[right_index]
            angle = math.acos(float(np.clip(abs(np.dot(left.axis, right.axis)), 0.0, 1.0)))
            if angle > CONFIG.duplicate_axis_radians:
                continue
            intersection = int(np.unpackbits(np.bitwise_and(bitsets[left_index], bitsets[right_index])).sum())
            if intersection == 0:
                continue
            union = len(left.member_indices) + len(right.member_indices) - intersection
            if intersection * CONFIG.duplicate_jaccard_denominator < union * CONFIG.duplicate_jaccard_numerator:
                continue
            lroot, rroot = find(left_index), find(right_index)
            if lroot != rroot:
                if rroot < lroot:
                    lroot, rroot = rroot, lroot
                parent[rroot] = lroot
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        groups[find(index)].append(index)
    clusters = sorted((sorted(indices) for indices in groups.values()), key=lambda values: min(ribbons[i].key for i in values))
    representatives = [ribbons[min(indices, key=lambda i: ribbons[i].key)] for indices in clusters]
    return representatives, clusters


def build_candidates(sample: Sample, field: NormalField) -> dict[str, Any]:
    poles = projective_poles()
    pole_hash = _sha_chunks(b"PGRR_V1_POLES\0", (poles.astype("<f8"),))
    left, right = _mutual_graph(field.neighbors)
    pole_rows: list[dict[str, Any]] = []
    failure_members: dict[tuple[int, str], list[str]] = defaultdict(list)
    raw_ribbons: list[Ribbon] = []
    control_subjects: dict[str, dict[str, Any]] = {}
    viable_poles = 0
    band_limit = math.sin(CONFIG.pole_band_radians)
    for pole_id, axis in enumerate(poles):
        eligible = field.stable & (
            np.abs(field.normal24 @ axis) <= band_limit
        ) & (
            np.abs(field.normal48 @ axis) <= band_limit
        )
        eligible_indices = np.flatnonzero(eligible)
        theta_all = np.full(len(sample.xyz), np.nan, dtype=np.float64)
        if len(eligible_indices):
            theta_all[eligible_indices] = _projective_theta(field.mean_normal[eligible_indices], axis)
            _unwrapped, _start, spread, _successor, _gap = _canonical_interval(
                theta_all[eligible_indices], eligible_indices, sample.point_ids
            )
        else:
            spread = 0.0
        potential = len(eligible_indices) >= CONFIG.pole_minimum_eligible_points and spread >= CONFIG.pole_minimum_angular_span_radians
        induced = eligible[left] & eligible[right]
        if potential:
            viable_poles += 1
            delta = np.abs(theta_all[left] - theta_all[right])
            projective_delta = np.minimum(delta, math.pi - delta)
            retained = induced & (projective_delta <= CONFIG.edge_maximum_angular_gap_radians)
            components = _component_labels_union_find(eligible, left, right, retained)
            raw_before = len(raw_ribbons)
            for members, component_edges in components:
                ribbon, code, diagnostic = _evaluate_component(
                    pole_id=pole_id,
                    axis=axis,
                    members=members,
                    component_edge_indices=component_edges,
                    left=left,
                    right=right,
                    theta_all=theta_all,
                    sample=sample,
                    field=field,
                )
                component_id = _component_id(members, sample.point_ids)
                if ribbon is not None:
                    raw_ribbons.append(ribbon)
                    if len(raw_ribbons) > CONFIG.raw_ribbon_maximum:
                        raise PGRRStop(
                            "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP",
                            "513th raw ribbon encountered",
                        )
                else:
                    failure_members[(pole_id, code)].append(component_id)
                    control_subjects.setdefault(code, {
                        "pole_id": pole_id,
                        "component_min_point_id": component_id,
                        "member_count": int(diagnostic.get("member_count", len(members))),
                        "theta_span": float(diagnostic.get("span", 0.0)),
                        "member_indices": np.asarray(members, dtype=np.uint32),
                    })
            component_count = len(components)
            raw_pass_count = len(raw_ribbons) - raw_before
            induced_count = int(np.count_nonzero(retained))
        else:
            component_count = 0
            raw_pass_count = 0
            induced_count = int(np.count_nonzero(induced))
        pole_rows.append({
            "pole_id": pole_id,
            "axis": [float(value) for value in axis],
            "eligible_count": int(len(eligible_indices)),
            "angular_spread": float(spread),
            "potentially_viable": bool(potential),
            "induced_edge_count": induced_count,
            "component_count": component_count,
            "raw_pass_count": raw_pass_count,
            "eligible_set_sha256": _membership_hash(eligible_indices.astype(np.uint32), sample.point_ids) if len(eligible_indices) else sha_json([]),
        })
    if viable_poles == 0:
        decision = "PGRR_V1_NO_GAUSS_ARC_MASS_ARCHITECTURE_STOP"
        deduplicated: list[Ribbon] = []
        clusters: list[list[int]] = []
    elif not raw_ribbons:
        decision = "PGRR_V1_NO_RULED_PREIMAGE_ARCHITECTURE_STOP"
        deduplicated = []
        clusters = []
    else:
        deduplicated, clusters = _deduplicate(raw_ribbons, len(sample.xyz))
        if len(deduplicated) > CONFIG.deduplicated_ribbon_maximum:
            decision = "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP"
        else:
            decision = "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF"
    failure_rows = []
    for (pole_id, code), identifiers in sorted(failure_members.items()):
        failure_rows.append({
            "pole_id": pole_id,
            "fail_code": code,
            "count": len(identifiers),
            "component_ids_sha256": sha_json(sorted(identifiers)),
        })
    raw_rows = []
    for row in sorted(raw_ribbons, key=lambda value: value.key):
        raw_rows.append(_ribbon_evidence(row))
    representative_index = {id(row): index for index, row in enumerate(deduplicated)}
    dedup_rows = []
    for candidate_index, (row, cluster) in enumerate(zip(deduplicated, clusters, strict=True)):
        evidence = _ribbon_evidence(row)
        evidence.update({
            "ribbon_id": f"PGRR_{candidate_index:04d}_{row.membership_sha256[:16]}",
            "duplicate_cluster_size": len(cluster),
            "duplicate_cluster_sha256": sha_json([raw_ribbons[index].membership_sha256 for index in cluster]),
        })
        dedup_rows.append(evidence)
    return {
        "schema_version": 1,
        "kind": "pgrr_v1_target_blind_candidate_commitment",
        "configuration_sha256": CONFIG_SHA256,
        "source_sha256": SOURCE_SHA256,
        "v4_container_sha256": V4_CONTAINER_SHA256,
        "canonical_point_order_sha256": sample.canonical_point_order_sha256,
        "point_count": len(sample.xyz),
        "unique_record_count": sample.unique_record_count,
        "density_scale": field.density_scale,
        "pole_table_sha256": pole_hash,
        "normal_field_commitment_sha256": field.normal_commitment_sha256,
        "stable_point_commitment_sha256": _membership_hash(
            np.flatnonzero(field.stable).astype(np.uint32), sample.point_ids
        ),
        "knn_commitment_sha256": field.knn_commitment_sha256,
        "stable_point_count": int(np.count_nonzero(field.stable)),
        "conditioned24_count": int(np.count_nonzero(field.conditioned24)),
        "conditioned48_count": int(np.count_nonzero(field.conditioned48)),
        "mutual_edge_count": len(left),
        "coarse_possible_pole_count": viable_poles,
        "raw_ribbon_count": len(raw_ribbons),
        "deduplicated_ribbon_count": len(deduplicated),
        "target_access_before_candidate_freeze": False,
        "source_points_serialized": False,
        "pole_census": pole_rows,
        "component_fail_census": failure_rows,
        "raw_ribbons": raw_rows,
        "deduplicated_ribbons": dedup_rows,
        "decision": decision,
        "_internal": {
            "sample": sample,
            "field": field,
            "left": left,
            "right": right,
            "raw_ribbons": raw_ribbons,
            "deduplicated": deduplicated,
            "control_subjects": control_subjects,
        },
    }


def _ribbon_evidence(row: Ribbon) -> dict[str, Any]:
    return {
        "pole_id": row.pole_id,
        "component_min_point_id": row.component_min_point_id,
        "membership_sha256": row.membership_sha256,
        "member_count": len(row.member_indices),
        "axis": [float(value) for value in row.axis],
        "theta_start": row.theta_start,
        "theta_span": row.theta_span,
        "bin_count": row.bin_count,
        "cut_successor_point_id": row.cut_successor_point_id,
        "maximum_gap": row.maximum_gap,
        "q05": row.q05,
        "q95": row.q95,
        "occupancy_mask_hex": hex(row.occupancy_mask),
        "column_coverage": list(row.column_coverage),
        "ruling_coherent_bin_count": row.ruling_coherent_bin_count,
        "ruling_axis_maximum_angle": row.ruling_axis_maximum_angle,
        "ruling_minimum_elongation": row.ruling_minimum_elongation,
        "bounding_box_min": [float(value) for value in row.bounding_box_min],
        "bounding_box_max": [float(value) for value in row.bounding_box_max],
    }


def public_commitment(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "_internal"}


def write_internal_state(
    path: Path, result: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    internal = result["_internal"]
    sample: Sample = internal["sample"]
    field: NormalField = internal["field"]
    ribbons: Sequence[Ribbon] = internal["deduplicated"]
    byte_count = (len(sample.xyz) + 7) // 8
    membership = np.zeros((len(ribbons), byte_count), dtype=np.uint8)
    bin_bits = np.zeros((len(ribbons), CONFIG.angular_bin_maximum, byte_count), dtype=np.uint8)
    bin_counts = np.zeros((len(ribbons), CONFIG.angular_bin_maximum), dtype=np.uint32)
    for row_index, ribbon in enumerate(ribbons):
        membership[row_index] = _bitset(ribbon.member_indices, len(sample.xyz))
        for angular_bin in range(ribbon.bin_count):
            indices = ribbon.member_indices[ribbon.bin_indices == angular_bin]
            bin_bits[row_index, angular_bin] = _bitset(indices, len(sample.xyz))
            bin_counts[row_index, angular_bin] = len(indices)
    control_codes = sorted(internal["control_subjects"])
    control_membership = np.zeros((len(control_codes), byte_count), dtype=np.uint8)
    control_metadata = []
    for index, code in enumerate(control_codes):
        row = internal["control_subjects"][code]
        control_membership[index] = _bitset(row["member_indices"], len(sample.xyz))
        control_metadata.append({key: value for key, value in row.items() if key != "member_indices"})
    metadata = {
        "schema_version": 1,
        "role": role,
        "point_count": len(sample.xyz),
        "candidate_rows": public_commitment(result)["deduplicated_ribbons"],
        "control_rows": control_metadata,
        "scratch_contains_coordinates": False,
        "scratch_contains_point_ids": False,
        "scratch_contains_point_index_arrays": False,
        "scratch_contains_internal_packed_membership_bitsets": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(
                stream,
                metadata=np.frombuffer(canonical_json_bytes(metadata), dtype=np.uint8),
                membership=membership,
                bin_bits=bin_bits,
                bin_counts=bin_counts,
                control_membership=control_membership,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha_file(path),
        "candidate_count": len(ribbons),
    }


def construct_and_commit(
    *, raw_source: Path, commitment_path: Path, scratch_path: Path, role: str,
    reverse_input: bool = False,
) -> dict[str, Any]:
    sample = load_v4_sample(raw_source, reverse_input=reverse_input)
    field = build_normal_field(sample)
    result = build_candidates(sample, field)
    public = public_commitment(result)
    public["role"] = role
    public["commitment_sha256"] = sha_json(public)
    _atomic_write(commitment_path, canonical_json_bytes(public) + b"\n")
    state = write_internal_state(scratch_path, result, role=role)
    return {"commitment": public, "scratch": state}
