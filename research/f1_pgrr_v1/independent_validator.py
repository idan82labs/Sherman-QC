"""Independent PGRR semantic validator.

This module does not import or call :mod:`research.f1_pgrr_v1.geometry`.
It uses scikit-learn's KDTree, NumPy/LAPACK symmetric eigenanalysis, and
SciPy sparse connected components instead of the producer's cKDTree,
analytic eigensolver, and union-find.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.neighbors import KDTree

from research.f1_ciip_v1 import primary_parser
from research.f1_ciip_v1.identifiers import point_id_sha256, sample_rank_sha256
from .config import CONFIG, CONFIG_SHA256, sha_json


SOURCE_SHA256 = "050cba324c35e32979b9ee626a1e73e64f8e39e4d7797c04a48eb8a9ff52bb97"
V4_CONTAINER_SHA256 = "7065b3b77a15435d8654d047d9afb50e954e8997375d94e9a3e1606141477817"


@dataclass(slots=True)
class IPointSet:
    xyz: np.ndarray
    point_ids: np.ndarray
    unique_record_count: int
    order_hash: str


@dataclass(slots=True)
class IField:
    neighbors: np.ndarray
    distances: np.ndarray
    normal24: np.ndarray
    normal48: np.ndarray
    mean: np.ndarray
    stable: np.ndarray
    conditioned24: np.ndarray
    conditioned48: np.ndarray
    density_scale: float


@dataclass(slots=True)
class IRibbon:
    pole_id: int
    component_min_point_id: str
    members: np.ndarray
    membership_sha256: str
    axis: np.ndarray
    theta_start: float
    theta_span: float
    bins: np.ndarray
    bin_count: int
    successor_id: str
    maximum_gap: float
    q05: float
    q95: float
    occupancy_mask: int
    column_coverage: tuple[int, ...]
    coherent_count: int
    maximum_axis_angle: float
    minimum_elongation: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.pole_id, self.component_min_point_id, self.membership_sha256)


def _sha_chunks(tag: bytes, *values: np.ndarray) -> str:
    digest = hashlib.sha256(tag)
    for value in values:
        raw = np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_points(path: Path) -> IPointSet:
    if _sha_file(path) != SOURCE_SHA256:
        raise ValueError("independent validator source hash mismatch")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    xyz = np.asarray(parsed.xyz, dtype="<f8", order="C").copy()
    xyz[xyz == 0.0] = 0.0
    packed = xyz.view("V24").reshape(len(xyz))
    _values, first, _counts = np.unique(packed, return_index=True, return_counts=True)
    unique = np.ascontiguousarray(xyz[first], dtype="<f8")
    point_ids = np.empty((len(unique), 32), dtype=np.uint8)
    ranks = np.empty_like(point_ids)
    raw = unique.view(np.uint8).reshape(len(unique), 24)
    for index in range(len(unique)):
        pid = point_id_sha256(bytes(raw[index]))
        point_ids[index] = np.frombuffer(pid, dtype=np.uint8)
        ranks[index] = np.frombuffer(sample_rank_sha256(pid), dtype=np.uint8)
    point_order = np.argsort(point_ids.view("V32").reshape(len(unique)), kind="stable")
    ordered = point_ids[point_order]
    for position in np.flatnonzero(np.all(ordered[1:] == ordered[:-1], axis=1)):
        if not np.array_equal(raw[point_order[position]], raw[point_order[position + 1]]):
            raise ValueError("independent point-ID collision")
    combined = np.empty((len(unique), 64), dtype=np.uint8)
    combined[:, :32] = ranks
    combined[:, 32:] = point_ids
    order = np.argsort(combined.view("V64").reshape(len(unique)), kind="stable")
    take = order[: CONFIG.sample_cap]
    sampled_ids = np.ascontiguousarray(point_ids[take])
    return IPointSet(
        xyz=np.ascontiguousarray(unique[take]),
        point_ids=sampled_ids,
        unique_record_count=len(unique),
        order_hash=_sha_chunks(b"PGRR_V1_V4_CANONICAL_POINT_ORDER\0", sampled_ids),
    )


def _neighbors(points: IPointSet) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(points.xyz, leaf_size=40, metric="euclidean")
    distances, indices = tree.query(points.xyz, k=50, return_distance=True, sort_results=False)
    n = len(points.xyz)
    output = np.empty((n, 48), dtype=np.uint32)
    output_distances = np.empty((n, 48), dtype=np.float64)
    pid = points.point_ids.view("V32").reshape(n)
    for row in range(n):
        candidate = np.asarray(indices[row], dtype=np.int64)
        candidate = candidate[candidate != row]
        delta = points.xyz[candidate] - points.xyz[row]
        squared = np.sum(delta * delta, axis=1)
        order = np.lexsort((pid[candidate], squared))
        candidate, squared = candidate[order], squared[order]
        boundary = squared[47]
        if len(squared) > 48 and squared[48] == boundary:
            tied = np.asarray(
                tree.query_radius(
                    points.xyz[row : row + 1],
                    r=np.nextafter(math.sqrt(boundary), math.inf),
                )[0],
                dtype=np.int64,
            )
            tied = tied[tied != row]
            delta = points.xyz[tied] - points.xyz[row]
            tied_squared = np.sum(delta * delta, axis=1)
            order = np.lexsort((pid[tied], tied_squared))
            candidate, squared = tied[order], tied_squared[order]
        output[row] = candidate[:48]
        output_distances[row] = np.sqrt(squared[:48])
    return output, output_distances


def _covariance(points: np.ndarray, neighbors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    eigenvalues = np.empty((n, 3), dtype=np.float64)
    normals = np.empty((n, 3), dtype=np.float64)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        rows = np.concatenate(
            (np.arange(start, end, dtype=np.int64)[:, None], neighbors[start:end, : k - 1]),
            axis=1,
        )
        values = points[rows]
        centroid = np.sum(values, axis=1) / float(k)
        delta = values - centroid[:, None, :]
        covariance = np.einsum("nki,nkj->nij", delta, delta, optimize=False) / float(k)
        eig, vectors = np.linalg.eigh(covariance)
        eigenvalues[start:end] = eig
        normals[start:end] = vectors[:, :, 0]
    return eigenvalues, normals


def build_field(points: IPointSet) -> IField:
    neighbors, distances = _neighbors(points)
    eig24, n24 = _covariance(points.xyz, neighbors, CONFIG.normal_k_small)
    eig48, n48 = _covariance(points.xyz, neighbors, CONFIG.normal_k_large)
    c24 = (eig24[:, 2] > 0.0) & (eig24[:, 1] >= CONFIG.normal_condition_ratio * eig24[:, 2])
    c48 = (eig48[:, 2] > 0.0) & (eig48[:, 1] >= CONFIG.normal_condition_ratio * eig48[:, 2])
    dot = np.sum(n24 * n48, axis=1)
    agreement = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    stable = c24 & c48 & (agreement <= CONFIG.normal_scale_agreement_radians)
    aligned = n48.copy()
    aligned[dot < 0.0] *= -1.0
    mean = n24 + aligned
    norm = np.linalg.norm(mean, axis=1)
    stable &= norm > 0.0
    mean[norm > 0.0] /= norm[norm > 0.0, None]
    d12 = np.sort(distances[:, CONFIG.density_neighbor_rank - 1], kind="stable")
    return IField(
        neighbors=neighbors,
        distances=distances,
        normal24=n24,
        normal48=n48,
        mean=mean,
        stable=stable,
        conditioned24=c24,
        conditioned48=c48,
        density_scale=float(d12[(len(d12) - 1) // 2]),
    )


_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def poles() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    vertices = [np.asarray(row, dtype=np.float64) / np.linalg.norm(row) for row in base]
    faces = list(_FACES)
    for _ in range(3):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                value = vertices[a] + vertices[b]
                value /= np.linalg.norm(value)
                cache[key] = len(vertices)
                vertices.append(value)
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = new_faces
    canonical = []
    for value in vertices:
        value = value.copy()
        for coordinate in value:
            if coordinate != 0.0:
                if coordinate < 0.0:
                    value *= -1.0
                break
        canonical.append(value)
    result: list[np.ndarray] = []
    for value in sorted(canonical, key=lambda row: tuple(float(v).hex() for v in row)):
        if not result or not np.any(np.abs(np.asarray(result) @ value) >= 1.0 - 8 * np.finfo(float).eps):
            result.append(value)
    if len(result) != 321:
        raise ValueError("independent pole cardinality mismatch")
    return np.asarray(result)


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(3)
    choice = int(np.argmin(np.abs(eye @ axis)))
    u = eye[choice] - np.dot(axis, eye[choice]) * axis
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _theta(normals: np.ndarray, axis: np.ndarray) -> np.ndarray:
    u, v = _basis(axis)
    return np.mod(np.arctan2(normals @ v, normals @ u), math.pi)


def _membership_hash(indices: np.ndarray, point_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(point_ids[indices])
    order = np.argsort(values.view("V32").reshape(len(values)), kind="stable")
    return _sha_chunks(b"PGRR_V1_MEMBERSHIP\0", values[order])


def _min_id(indices: np.ndarray, ids: np.ndarray) -> str:
    local = min(
        range(len(indices)),
        key=lambda offset: bytes(ids[indices[offset]]),
    )
    return bytes(ids[indices[local]]).hex()


def _interval(theta: np.ndarray, members: np.ndarray, ids: np.ndarray):
    order = np.lexsort((ids[members].view("V32").reshape(len(members)), theta))
    sorted_theta, sorted_members = theta[order], members[order]
    gap = np.empty(len(theta))
    gap[:-1] = sorted_theta[1:] - sorted_theta[:-1]
    gap[-1] = sorted_theta[0] + math.pi - sorted_theta[-1]
    maximum = float(np.max(gap))
    candidates = np.flatnonzero(gap == maximum)
    successors = (candidates + 1) % len(theta)
    successor_ids = ids[sorted_members[successors]].view("V32").reshape(len(candidates))
    chosen = int(candidates[int(np.argsort(successor_ids, kind="stable")[0])])
    successor = (chosen + 1) % len(theta)
    start = float(sorted_theta[successor])
    return np.mod(theta - start, math.pi), start, math.pi - maximum, bytes(ids[sorted_members[successor]]).hex(), maximum


def _mutual(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(neighbors)
    source = np.repeat(np.arange(n, dtype=np.uint64), 12)
    target = neighbors[:, :12].reshape(-1).astype(np.uint64)
    directed = source * np.uint64(n) + target
    ordered = np.sort(directed, kind="stable")
    reverse = target * np.uint64(n) + source
    position = np.searchsorted(ordered, reverse)
    mutual = (position < len(ordered)) & (ordered[np.minimum(position, len(ordered) - 1)] == reverse)
    keep = mutual & (source < target)
    left, right = source[keep].astype(np.uint32), target[keep].astype(np.uint32)
    order = np.lexsort((right, left))
    return left[order], right[order]


def _components(eligible: np.ndarray, left: np.ndarray, right: np.ndarray, retained: np.ndarray):
    vertices = np.flatnonzero(eligible)
    if len(vertices) == 0:
        return []
    selected_left, selected_right = left[retained], right[retained]
    data = np.ones(2 * len(selected_left), dtype=np.uint8)
    rows = np.concatenate((selected_left, selected_right))
    cols = np.concatenate((selected_right, selected_left))
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(len(eligible), len(eligible)))
    _count, labels = csgraph.connected_components(graph, directed=False, return_labels=True)
    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in vertices:
        groups[int(labels[vertex])].append(int(vertex))
    edge_indices = np.flatnonzero(retained)
    result = []
    for _label, members in sorted(groups.items(), key=lambda item: min(item[1])):
        member_array = np.asarray(members, dtype=np.uint32)
        mask = np.isin(selected_left, member_array, assume_unique=False)
        result.append((member_array, edge_indices[mask]))
    return result


def _evaluate(
    pole_id: int, axis: np.ndarray, members: np.ndarray, edge_indices: np.ndarray,
    left: np.ndarray, right: np.ndarray, theta_all: np.ndarray, points: IPointSet, field: IField,
) -> tuple[IRibbon | None, str]:
    if len(members) < 64:
        return None, "MEMBER_COUNT"
    unwrapped, start, span, successor, maximum_gap = _interval(theta_all[members], members, points.point_ids)
    if span < CONFIG.component_minimum_span_radians:
        return None, "ARC_SPAN_LOW"
    if span > CONFIG.component_maximum_span_radians:
        return None, "ARC_SPAN_HIGH"
    count = int(math.ceil(span / CONFIG.nominal_angular_bin_width_radians))
    if not 4 <= count <= 33:
        return None, "ANGULAR_BIN_COUNT"
    width = span / count
    bins = np.minimum(np.floor(unwrapped / width).astype(np.int32), count - 1)
    if np.any(np.bincount(bins, minlength=count) < 4):
        return None, "ANGULAR_BIN_SUPPORT"
    if maximum_gap > CONFIG.edge_maximum_angular_gap_radians:
        return None, "ANGULAR_DISCONNECTION"
    bin_by_vertex = np.full(len(points.xyz), -1, dtype=np.int16)
    bin_by_vertex[members] = bins
    boundary = np.zeros(count - 1, dtype=np.int32)
    for edge_index in edge_indices:
        a, b = int(left[edge_index]), int(right[edge_index])
        ba, bb = int(bin_by_vertex[a]), int(bin_by_vertex[b])
        if abs(ba - bb) == 1:
            boundary[min(ba, bb)] += 1
    if np.any(boundary == 0):
        return None, "ANGULAR_BOUNDARY_EDGE"
    z = points.xyz[members] @ axis
    ordered_z = np.sort(z, kind="stable")
    q05 = float(ordered_z[int(math.floor(0.05 * (len(z) - 1)))])
    q95 = float(ordered_z[int(math.floor(0.95 * (len(z) - 1)))])
    if q95 - q05 < 8.0 * field.density_scale:
        return None, "LONGITUDINAL_SPAN"
    within = (z >= q05) & (z <= q95)
    column_width = (q95 - q05) / 8.0
    columns = np.minimum(np.floor((z[within] - q05) / column_width).astype(np.int32), 7)
    cells = np.zeros((count, 8), dtype=np.int32)
    np.add.at(cells, (bins[within], columns), 1)
    occupied = cells >= 2
    minimum = int(math.ceil(3.0 * count / 4.0))
    coverage = tuple(int(value) for value in np.sum(occupied, axis=0))
    if np.count_nonzero(np.sum(occupied, axis=0) >= minimum) < 6:
        return None, "LONGITUDINAL_OCCUPANCY"
    if np.any(np.sum(occupied, axis=1) < 4):
        return None, "ANGULAR_LONGITUDINAL_COVERAGE"
    coherent = 0
    maximum_angle = 0.0
    minimum_elongation = math.inf
    for angular_bin in range(count):
        values = points.xyz[members[(bins == angular_bin) & within]]
        center = np.sum(values, axis=0) / len(values)
        delta = values - center
        covariance = delta.T @ delta / len(values)
        eig, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, 2]
        angle = math.acos(float(np.clip(abs(np.dot(principal, axis)), 0.0, 1.0)))
        elongation = float(eig[2] / max(eig[1], np.finfo(float).tiny))
        maximum_angle = max(maximum_angle, angle)
        minimum_elongation = min(minimum_elongation, elongation)
        if angle <= CONFIG.ruling_axis_radians and eig[2] > 0 and eig[2] >= 4.0 * eig[1]:
            coherent += 1
    if coherent < minimum:
        return None, "RULING_COHERENCE"
    occupancy_mask = 0
    for row in range(count):
        for column in range(8):
            if occupied[row, column]:
                occupancy_mask |= 1 << (8 * row + column)
    selected_points = points.xyz[members]
    return IRibbon(
        pole_id=pole_id,
        component_min_point_id=_min_id(members, points.point_ids),
        members=members,
        membership_sha256=_membership_hash(members, points.point_ids),
        axis=axis,
        theta_start=start,
        theta_span=span,
        bins=bins,
        bin_count=count,
        successor_id=successor,
        maximum_gap=maximum_gap,
        q05=q05,
        q95=q95,
        occupancy_mask=occupancy_mask,
        column_coverage=coverage,
        coherent_count=coherent,
        maximum_axis_angle=maximum_angle,
        minimum_elongation=minimum_elongation,
        bbox_min=np.min(selected_points, axis=0),
        bbox_max=np.max(selected_points, axis=0),
    ), "PASS"


def _deduplicate(rows: Sequence[IRibbon]) -> tuple[list[IRibbon], list[list[int]]]:
    parent = np.arange(len(rows), dtype=np.int32)
    member_sets = [set(map(int, row.members)) for row in rows]

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if math.acos(float(np.clip(abs(np.dot(rows[left].axis, rows[right].axis)), 0.0, 1.0))) > CONFIG.duplicate_axis_radians:
                continue
            intersection = len(member_sets[left] & member_sets[right])
            union = len(member_sets[left] | member_sets[right])
            if intersection * 100 < union * 80:
                continue
            a, b = find(left), find(right)
            if a != b:
                if b < a:
                    a, b = b, a
                parent[b] = a
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    clusters = sorted((sorted(v) for v in groups.values()), key=lambda v: min(rows[i].key for i in v))
    return [rows[min(v, key=lambda i: rows[i].key)] for v in clusters], clusters


def _semantic_ribbon(row: IRibbon, ribbon_id: str | None = None) -> dict[str, Any]:
    value = {
        "pole_id": row.pole_id,
        "component_min_point_id": row.component_min_point_id,
        "membership_sha256": row.membership_sha256,
        "member_count": len(row.members),
        "bin_count": row.bin_count,
    }
    if ribbon_id is not None:
        value["ribbon_id"] = ribbon_id
    return value


def build_semantic_bundle(raw_source: Path) -> dict[str, Any]:
    points = load_points(raw_source)
    field = build_field(points)
    pole_table = poles()
    pole_hash = _sha_chunks(b"PGRR_V1_POLES\0", pole_table.astype("<f8"))
    left, right = _mutual(field.neighbors)
    viable = 0
    raw: list[IRibbon] = []
    band = math.sin(CONFIG.pole_band_radians)
    for pole_id, axis in enumerate(pole_table):
        eligible = field.stable & (np.abs(field.normal24 @ axis) <= band) & (np.abs(field.normal48 @ axis) <= band)
        indices = np.flatnonzero(eligible)
        theta_all = np.full(len(points.xyz), np.nan)
        if len(indices):
            theta_all[indices] = _theta(field.mean[indices], axis)
            _u, _s, spread, _sid, _gap = _interval(theta_all[indices], indices, points.point_ids)
        else:
            spread = 0.0
        if len(indices) < 64 or spread < CONFIG.pole_minimum_angular_span_radians:
            continue
        viable += 1
        delta = np.abs(theta_all[left] - theta_all[right])
        retained = eligible[left] & eligible[right] & (np.minimum(delta, math.pi - delta) <= CONFIG.edge_maximum_angular_gap_radians)
        for members, edge_indices in _components(eligible, left, right, retained):
            row, _code = _evaluate(pole_id, axis, members, edge_indices, left, right, theta_all, points, field)
            if row is not None:
                raw.append(row)
                if len(raw) > 512:
                    break
        if len(raw) > 512:
            break
    if viable == 0:
        decision = "PGRR_V1_NO_GAUSS_ARC_MASS_ARCHITECTURE_STOP"
        deduplicated, clusters = [], []
    elif not raw:
        decision = "PGRR_V1_NO_RULED_PREIMAGE_ARCHITECTURE_STOP"
        deduplicated, clusters = [], []
    elif len(raw) > 512:
        decision = "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP"
        deduplicated, clusters = [], []
    else:
        deduplicated, clusters = _deduplicate(raw)
        decision = (
            "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP"
            if len(deduplicated) > 128
            else "PGRR_V1_CANDIDATES_FROZEN_PENDING_HANDOFF"
        )
    raw_semantic = [_semantic_ribbon(row) for row in sorted(raw, key=lambda value: value.key)]
    dedup_semantic = []
    for index, row in enumerate(deduplicated):
        dedup_semantic.append(_semantic_ribbon(row, f"PGRR_{index:04d}_{row.membership_sha256[:16]}"))
    return {
        "schema_version": 1,
        "implementation": "INDEPENDENT_SKLEARN_KDTREE_NUMPY_EIGH_SCIPY_CSGRAPH",
        "configuration_sha256": CONFIG_SHA256,
        "source_sha256": SOURCE_SHA256,
        "v4_container_sha256": V4_CONTAINER_SHA256,
        "point_count": len(points.xyz),
        "unique_record_count": points.unique_record_count,
        "canonical_point_order_sha256": points.order_hash,
        "stable_point_commitment_sha256": _membership_hash(np.flatnonzero(field.stable).astype(np.uint32), points.point_ids),
        "pole_table_sha256": pole_hash,
        "coarse_possible_pole_count": viable,
        "raw_ribbon_count": len(raw),
        "deduplicated_ribbon_count": len(deduplicated),
        "raw_ribbons_semantic": raw_semantic,
        "deduplicated_ribbons_semantic": dedup_semantic,
        "decision": decision,
        "target_access_before_candidate_freeze": False,
        "producer_geometry_imported": False,
    }


def validate_against_producer(
    producer: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    producer_raw = [
        {key: row[key] for key in ("pole_id", "component_min_point_id", "membership_sha256", "member_count", "bin_count")}
        for row in producer["raw_ribbons"]
    ]
    producer_dedup = [
        {key: row[key] for key in ("pole_id", "component_min_point_id", "membership_sha256", "member_count", "bin_count", "ribbon_id")}
        for row in producer["deduplicated_ribbons"]
    ]
    checks = {
        "configuration_equal": producer["configuration_sha256"] == independent["configuration_sha256"],
        "source_equal": producer["source_sha256"] == independent["source_sha256"],
        "point_count_equal": producer["point_count"] == independent["point_count"],
        "canonical_point_order_equal": producer["canonical_point_order_sha256"] == independent["canonical_point_order_sha256"],
        "stable_point_commitment_equal": producer["stable_point_commitment_sha256"] == independent["stable_point_commitment_sha256"],
        "pole_table_equal": producer["pole_table_sha256"] == independent["pole_table_sha256"],
        "coarse_pole_count_equal": producer["coarse_possible_pole_count"] == independent["coarse_possible_pole_count"],
        "raw_ribbon_semantics_equal": producer_raw == independent["raw_ribbons_semantic"],
        "deduplicated_ribbon_semantics_equal": producer_dedup == independent["deduplicated_ribbons_semantic"],
        "candidate_decision_equal": producer["decision"] == independent["decision"],
    }
    return {
        "checks": checks,
        "semantic_consensus": all(checks.values()),
        "producer_semantic_sha256": sha_json({"raw": producer_raw, "dedup": producer_dedup}),
        "validator_semantic_sha256": sha_json({
            "raw": independent["raw_ribbons_semantic"],
            "dedup": independent["deduplicated_ribbons_semantic"],
        }),
    }
