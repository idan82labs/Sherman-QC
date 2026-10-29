"""Production computation for Concentric Focal Axis Collapse V1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from research.f1_ciip_v1 import partition, primary_parser
from .config import CONFIG, CONFIG_SHA256, sha_json


class CFACInvalid(RuntimeError):
    pass


@dataclass(slots=True)
class CanonicalSource:
    xyz: np.ndarray
    point_ids: np.ndarray
    raw_source_sha256: str
    raw_source_bytes: int
    canonical_sha256: str


@dataclass(slots=True)
class JetField:
    s: float
    normals: np.ndarray
    curvatures: np.ndarray
    weak_directions: np.ndarray
    radii: np.ndarray
    focal_points: np.ndarray
    base_quality: np.ndarray
    focal_vote: np.ndarray
    vertical_rms_s: np.ndarray
    condition_number: np.ndarray
    occupied_sectors: np.ndarray
    d48_s: np.ndarray


@dataclass(slots=True)
class AxisCandidate:
    candidate_id: int
    focal_vote_ids: np.ndarray
    source_point_ids: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    axial_low: float
    axial_high: float
    span: float
    line_fraction: float
    transverse_rms: float
    collapse_density: float
    radii: tuple[float, float] | None
    radial_modes: np.ndarray
    arc_widths_degrees: tuple[float, float] | None
    arc_endpoints_radians: tuple[tuple[float, float], tuple[float, float]] | None
    continuation_occupancies: tuple[float, float, float, float] | None
    continuation_offsets_degrees: tuple[float, float, float, float] | None
    retained: bool
    rejection: str | None


@dataclass(slots=True)
class Evaluation:
    source: CanonicalSource
    jets: JetField
    dense_vote_mask: np.ndarray
    focal_edges: np.ndarray
    candidates: list[AxisCandidate]
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
    prefix = b"CFAC_V1_ORDER_PERMUTED\0" + bytes.fromhex(source_sha)
    keys = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(prefix + bytes(row)).digest()[:8], "little")
            for row in rows
        ),
        dtype=np.uint64,
        count=len(rows),
    )
    order = np.lexsort((rows.view("V24").reshape(len(rows)), keys))
    return np.ascontiguousarray(xyz[order], dtype="<f8")


def _content_sample(xyz_raw: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(xyz_raw, dtype="<f8").view("V24").reshape(len(xyz_raw))
    _unique, first, _inverse, counts = np.unique(
        packed, return_index=True, return_inverse=True, return_counts=True
    )
    sampled = partition.canonical_sample(
        np.ascontiguousarray(xyz_raw[first], dtype="<f8"),
        np.asarray(counts, dtype=np.uint32),
        sample_cap=CONFIG.sample_count,
    )
    if len(sampled.xyz) != CONFIG.sample_count:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: sample is not 200000")
    return np.asarray(sampled.xyz, dtype=np.float64)


def canonicalize_xyz(
    xyz: np.ndarray,
    *,
    raw_source_sha256: str = "SOURCE_FREE_FIXTURE",
    raw_source_bytes: int = 0,
) -> CanonicalSource:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: invalid XYZ")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: no finite XYZ")
    bits = np.ascontiguousarray(points, dtype="<f8").view(np.uint64).reshape(-1, 3)
    order = np.lexsort(
        (bits[:, 2], bits[:, 1], bits[:, 0], points[:, 2], points[:, 1], points[:, 0])
    )
    ordered = np.ascontiguousarray(points[order], dtype="<f8")
    keep = np.ones(len(ordered), dtype=bool)
    if len(ordered) > 1:
        keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    ordered = ordered[keep]
    if raw_source_sha256 != "SOURCE_FREE_FIXTURE" and len(ordered) < CONFIG.minimum_unique_points:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: fewer than 50000 unique points")
    return CanonicalSource(
        xyz=ordered,
        point_ids=np.arange(len(ordered), dtype=np.uint32),
        raw_source_sha256=raw_source_sha256,
        raw_source_bytes=raw_source_bytes,
        canonical_sha256=sha_arrays(b"CFAC_V1_CANONICAL_XYZ\0", (ordered,)),
    )


def load_bound_source(path: Path, *, permute_input: bool = False) -> CanonicalSource:
    observed_sha = sha_file(path)
    observed_bytes = path.stat().st_size
    if observed_sha != CONFIG.raw_source_sha256 or observed_bytes != CONFIG.raw_source_bytes:
        raise CFACInvalid("INVALID_CFAC_V1_GOVERNANCE_STOP: source binding")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        parsed = primary_parser.parse_ply_fd(descriptor)
    finally:
        os.close(descriptor)
    raw = np.asarray(parsed.xyz, dtype="<f8")
    if permute_input:
        raw = _permuted_raw(raw, observed_sha)
    sampled = _content_sample(raw)
    return canonicalize_xyz(
        sampled, raw_source_sha256=observed_sha, raw_source_bytes=observed_bytes
    )


def _deterministic_knn(tree: cKDTree, xyz: np.ndarray, k_nonself: int) -> tuple[np.ndarray, np.ndarray]:
    proposal = min(len(xyz), k_nonself + 9)
    distance, indices = tree.query(xyz, k=proposal, workers=1)
    distance = np.asarray(distance, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distance.ndim == 1:
        distance, indices = distance[:, None], indices[:, None]
    out_distance = np.empty((len(xyz), k_nonself), dtype=np.float64)
    out_indices = np.empty((len(xyz), k_nonself), dtype=np.int64)
    for start in range(0, len(xyz), CONFIG.block_size):
        stop = min(len(xyz), start + CONFIG.block_size)
        for local, point_id in enumerate(range(start, stop)):
            ds = distance[point_id]
            ids = indices[point_id]
            valid = ids != point_id
            ds, ids = ds[valid], ids[valid]
            ranked = np.lexsort((ids, ds))
            out_distance[point_id] = ds[ranked[:k_nonself]]
            out_indices[point_id] = ids[ranked[:k_nonself]]
    return out_distance, out_indices


def estimate_jets(source: CanonicalSource) -> JetField:
    xyz = source.xyz
    if len(xyz) <= CONFIG.jet_neighbor_count:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: insufficient points for jet")
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    d4, _ = tree.query(xyz, k=CONFIG.scale_neighbor_rank + 1, workers=1)
    s = float(np.median(np.asarray(d4)[:, CONFIG.scale_neighbor_rank]))
    if not math.isfinite(s) or s <= 0:
        raise CFACInvalid("CFAC_V1_SOURCE_CONTRACT_STOP: invalid s")
    distances, indices = _deterministic_knn(tree, xyz, CONFIG.jet_neighbor_count)
    count = len(xyz)
    normals = np.zeros((count, 3), dtype=np.float64)
    curvatures = np.full((count, 2), np.nan, dtype=np.float64)
    weak_directions = np.zeros((count, 3), dtype=np.float64)
    vertical_rms = np.full(count, np.inf)
    condition = np.full(count, np.inf)
    sectors = np.zeros(count, dtype=np.uint8)
    d48_s = distances[:, -1] / s
    base = np.zeros(count, dtype=bool)
    focal = np.zeros(count, dtype=bool)
    focal_points = np.full((count, 3), np.nan)
    radii = np.full(count, np.nan)
    for start in range(0, count, CONFIG.block_size):
        stop = min(count, start + CONFIG.block_size)
        for point_id in range(start, stop):
            ids = indices[point_id]
            delta = xyz[ids] - xyz[point_id]
            dmax = float(distances[point_id, -1])
            if dmax <= 0:
                continue
            weights = np.square(np.maximum(0.0, 1.0 - np.square(distances[point_id] / dmax)))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                continue
            centroid = np.sum(weights[:, None] * delta, axis=0) / weight_sum
            centered = delta - centroid
            covariance = np.einsum(
                "i,ij,ik->jk", weights, centered, centered, optimize=True
            ) / weight_sum
            values, vectors = np.linalg.eigh(covariance)
            e1 = vectors[:, 2]
            e2 = vectors[:, 1]
            n0 = vectors[:, 0]
            x = delta @ e1
            y = delta @ e2
            z = delta @ n0
            angles = np.mod(np.arctan2(y, x), 2 * math.pi)
            sector_ids = np.floor(
                angles / (2 * math.pi) * CONFIG.tangent_sector_count
            ).astype(int)
            sectors[point_id] = len(np.unique(np.clip(sector_ids, 0, 11)))
            X, Y, Z = x / s, y / s, z / s
            design = np.column_stack(
                (
                    np.ones(len(X)),
                    X,
                    Y,
                    0.5 * X * X,
                    X * Y,
                    0.5 * Y * Y,
                )
            )
            root_w = np.sqrt(weights)
            weighted_design = design * root_w[:, None]
            weighted_z = Z * root_w
            normal_matrix = weighted_design.T @ weighted_design
            condition[point_id] = float(np.linalg.cond(normal_matrix))
            try:
                beta, _residuals, rank, _singular = np.linalg.lstsq(
                    weighted_design, weighted_z, rcond=None
                )
            except np.linalg.LinAlgError:
                continue
            if rank < 6:
                continue
            residual = Z - design @ beta
            vertical_rms[point_id] = math.sqrt(
                float(np.sum(weights * residual * residual) / weight_sum)
            )
            bx, by = float(beta[1]), float(beta[2])
            gradient_norm = math.hypot(bx, by)
            q = math.sqrt(1.0 + bx * bx + by * by)
            first = np.asarray(((1 + bx * bx, bx * by), (bx * by, 1 + by * by)))
            second = np.asarray(
                ((float(beta[3]), float(beta[4])), (float(beta[4]), float(beta[5])))
            ) / (q * s)
            try:
                first_values, first_vectors = np.linalg.eigh(first)
                inverse_root = (
                    first_vectors
                    @ np.diag(1.0 / np.sqrt(first_values))
                    @ first_vectors.T
                )
                symmetric_shape = inverse_root @ second @ inverse_root
                k_values, k_vectors = np.linalg.eigh(symmetric_shape)
            except np.linalg.LinAlgError:
                continue
            parameter_vectors = inverse_root @ k_vectors
            dominant_index = int(np.argmax(np.abs(k_values)))
            weak_index = 1 - dominant_index
            kd = float(k_values[dominant_index])
            kw = float(k_values[weak_index])
            local_normal = (-bx * e1 - by * e2 + n0) / q
            weak_param = parameter_vectors[:, weak_index]
            weak = (
                weak_param[0] * e1
                + weak_param[1] * e2
                + (bx * weak_param[0] + by * weak_param[1]) * n0
            )
            weak_norm = float(np.linalg.norm(weak))
            if weak_norm <= 0 or not math.isfinite(kd) or not math.isfinite(kw):
                continue
            weak /= weak_norm
            normals[point_id] = local_normal
            curvatures[point_id] = (kd, kw)
            weak_directions[point_id] = weak
            base[point_id] = (
                sectors[point_id] >= CONFIG.tangent_minimum_sectors
                and d48_s[point_id] <= CONFIG.jet_d48_maximum_s
                and vertical_rms[point_id] <= CONFIG.vertical_rms_maximum_s
                and condition[point_id] <= CONFIG.normal_matrix_condition_maximum
                and gradient_norm <= CONFIG.gradient_norm_maximum
            )
            if not base[point_id] or kd == 0:
                continue
            rho = 1.0 / abs(kd)
            anisotropy = abs(kw) / abs(kd)
            if (
                CONFIG.radius_minimum_s * s <= rho <= CONFIG.radius_maximum_s * s
                and anisotropy <= CONFIG.anisotropy_maximum
            ):
                focal[point_id] = True
                radii[point_id] = rho
                focal_points[point_id] = xyz[point_id] + local_normal / kd
    return JetField(
        s=s,
        normals=normals,
        curvatures=curvatures,
        weak_directions=weak_directions,
        radii=radii,
        focal_points=focal_points,
        base_quality=base,
        focal_vote=focal,
        vertical_rms_s=vertical_rms,
        condition_number=condition,
        occupied_sectors=sectors,
        d48_s=d48_s,
    )


def _axis_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    origin = np.mean(points, axis=0)
    centered = points - origin
    covariance = centered.T @ centered / max(1, len(points))
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, 2]
    # Projective canonical sign.
    for coordinate in direction:
        if abs(coordinate) > 1e-15:
            if coordinate < 0:
                direction = -direction
            break
    axial = centered @ direction
    perpendicular = centered - np.outer(axial, direction)
    line_fraction = float(values[2] / max(np.sum(values), np.finfo(float).tiny))
    transverse_rms = math.sqrt(float(np.mean(np.sum(perpendicular * perpendicular, axis=1))))
    return origin, direction, axial, line_fraction, transverse_rms


def _components(vertex_count: int, edges: np.ndarray, active_vertices: np.ndarray) -> list[np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    unseen = set(int(value) for value in np.flatnonzero(active_vertices))
    results: list[np.ndarray] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        group: list[int] = []
        while stack:
            node = stack.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            group.append(node)
            stack.extend(adjacency[node])
        results.append(np.asarray(sorted(group), dtype=np.uint32))
    return results


def focal_graph(
    source: CanonicalSource, jets: JetField
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vote_source_ids = np.flatnonzero(jets.focal_vote).astype(np.uint32)
    focal = jets.focal_points[vote_source_ids]
    directions = jets.weak_directions[vote_source_ids]
    if len(focal) == 0:
        return vote_source_ids, np.zeros(0, bool), np.empty((0, 2), np.uint32), focal
    if len(focal) > min(
        int(CONFIG.focal_vote_fraction_maximum * len(source.xyz)),
        CONFIG.focal_vote_absolute_maximum,
    ):
        return vote_source_ids, np.zeros(len(focal), bool), np.empty((0, 2), np.uint32), focal
    tree = cKDTree(focal, compact_nodes=True, balanced_tree=True, copy_data=False)
    k = min(6, len(focal))
    d6, _ = tree.query(focal, k=k, workers=1)
    d6 = np.asarray(d6)
    if d6.ndim == 1:
        d6 = d6[:, None]
    dense = np.zeros(len(focal), dtype=bool)
    for index in range(len(focal)):
        if d6[index, -1] > CONFIG.focal_dense_sixth_maximum_s * jets.s:
            continue
        ids = tree.query_ball_point(
            focal[index], CONFIG.focal_coherence_radius_s * jets.s
        )
        matrix = directions[ids].T @ directions[ids]
        coherence = float(np.linalg.eigvalsh(matrix)[2] / max(np.trace(matrix), np.finfo(float).tiny))
        dense[index] = coherence >= CONFIG.focal_coherence_minimum
    rows: set[tuple[int, int]] = set()
    cos_axis = math.cos(math.radians(CONFIG.graph_axis_disagreement_maximum_degrees))
    for start in range(0, len(focal), CONFIG.block_size):
        block = np.arange(start, min(len(focal), start + CONFIG.block_size), dtype=np.int64)
        block = block[dense[block]]
        if len(block) == 0:
            continue
        targets = (
            focal[block, None, :]
            + np.asarray(CONFIG.graph_target_offsets_s)[None, :, None]
            * jets.s
            * directions[block, None, :]
        )
        miss, peer = tree.query(targets.reshape(-1, 3), k=1, workers=1)
        miss = np.asarray(miss).reshape(len(block), -1)
        peer = np.asarray(peer, dtype=np.int64).reshape(len(block), -1)
        for local, left in enumerate(block):
            for target_index, right in enumerate(peer[local]):
                right_i = int(right)
                if right_i == int(left):
                    continue
                displacement = focal[right_i] - focal[left]
                perpendicular = displacement - directions[left] * np.dot(
                    displacement, directions[left]
                )
                if (
                    miss[local, target_index]
                    <= CONFIG.graph_target_miss_maximum_s * jets.s
                    and abs(np.dot(directions[left], directions[right_i])) >= cos_axis
                    and np.linalg.norm(perpendicular)
                    <= CONFIG.graph_perpendicular_maximum_s * jets.s
                ):
                    rows.add(tuple(sorted((int(left), right_i))))
    edges = np.asarray(sorted(rows), dtype=np.uint32).reshape(-1, 2)
    if len(edges) > CONFIG.graph_edge_cap:
        raise CFACInvalid("INVALID_CFAC_V1_RESOURCE_STOP: graph edge cap")
    return vote_source_ids, dense, edges, focal


def _l1_error(sorted_values: np.ndarray, prefix: np.ndarray, low: int, high: int) -> float:
    if high <= low:
        return 0.0
    middle = (low + high - 1) // 2
    median = sorted_values[middle]
    left = median * (middle - low) - (prefix[middle] - prefix[low])
    right = (prefix[high] - prefix[middle + 1]) - median * (high - middle - 1)
    return float(left + right)


def _optimal_radial_split(radii: np.ndarray) -> tuple[int, float, float, float]:
    ordered = np.sort(np.asarray(radii, dtype=np.float64))
    count = len(ordered)
    prefix = np.concatenate(([0.0], np.cumsum(ordered)))
    one_error = _l1_error(ordered, prefix, 0, count)
    lower = max(CONFIG.radial_split_minimum_points, math.ceil(CONFIG.radial_split_minimum_fraction * count))
    upper = min(count - CONFIG.radial_split_minimum_points, math.floor((1.0 - CONFIG.radial_split_minimum_fraction) * count))
    if lower > upper:
        return -1, 0.0, math.inf, one_error
    best = (math.inf, -1)
    for split in range(lower, upper + 1):
        error = _l1_error(ordered, prefix, 0, split) + _l1_error(ordered, prefix, split, count)
        if (error, split) < best:
            best = (error, split)
    gain = 0.0 if one_error <= 0 else 1.0 - best[0] / one_error
    return best[1], gain, best[0], one_error


def _circular_difference(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


def _arc_interval(angles: np.ndarray) -> tuple[float, float, float]:
    ordered = np.sort(np.mod(angles, 2 * math.pi))
    if len(ordered) < 2:
        return 0.0, 0.0, 0.0
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2 * math.pi)))
    gap_index = int(np.argmax(gaps))
    start = float(ordered[(gap_index + 1) % len(ordered)])
    end = float(ordered[gap_index])
    width = (end - start) % (2 * math.pi)
    return start, end, width


def _continuation_gate(
    source: CanonicalSource,
    jets: JetField,
    candidate: AxisCandidate,
    mode_radius: float,
    endpoint_angle: float,
    endpoint_sign: float,
) -> tuple[float, float]:
    e1, e2 = _axis_frame(candidate.direction)
    axis_positions = np.linspace(candidate.axial_low, candidate.axial_high, CONFIG.continuation_axial_sites)
    raw_tree = cKDTree(source.xyz, compact_nodes=True, balanced_tree=True, copy_data=False)
    offsets = np.arange(
        CONFIG.continuation_offset_minimum_degrees,
        CONFIG.continuation_offset_maximum_degrees + 0.5 * CONFIG.continuation_offset_step_degrees,
        CONFIG.continuation_offset_step_degrees,
    )
    rows: list[tuple[float, float]] = []
    cos_normal = math.cos(math.radians(CONFIG.continuation_normal_degrees))
    for offset in offsets:
        angle = endpoint_angle + math.radians(float(offset))
        radial = math.cos(angle) * e1 + math.sin(angle) * e2
        tangent = endpoint_sign * np.cross(candidate.direction, radial)
        sites = []
        for axial in axis_positions:
            base = candidate.origin + axial * candidate.direction + mode_radius * radial
            for distance_s in CONFIG.continuation_tangent_distances_s:
                sites.append(base + distance_s * jets.s * tangent)
        sites_array = np.asarray(sites)
        miss, point_ids = raw_tree.query(sites_array, k=1, workers=1)
        point_ids = np.asarray(point_ids, dtype=np.int64)
        planar = (
            jets.s * np.max(np.abs(jets.curvatures[point_ids]), axis=1)
            <= CONFIG.continuation_planar_curvature_maximum
        )
        normal_ok = (
            np.abs(jets.normals[point_ids] @ radial) >= cos_normal
        )
        occupied = (
            (np.asarray(miss) <= CONFIG.continuation_site_miss_maximum_s * jets.s)
            & jets.base_quality[point_ids]
            & planar
            & normal_ok
        )
        rows.append((float(np.mean(occupied)), float(offset)))
    best = min(rows, key=lambda row: (-row[0], abs(row[1]), row[1]))
    return best


def _evaluate_candidate(
    candidate_id: int,
    component_vote_ids: np.ndarray,
    vote_source_ids: np.ndarray,
    focal: np.ndarray,
    jets: JetField,
    source: CanonicalSource,
) -> AxisCandidate | None:
    points = focal[component_vote_ids]
    origin, direction, axial, line_fraction, transverse_rms = _line_fit(points)
    low, high = np.percentile(axial, (5, 95))
    span = float(high - low)
    median_radius = float(np.median(jets.radii[vote_source_ids[component_vote_ids]]))
    collapse_density = len(points) / max(span / jets.s, np.finfo(float).tiny)
    if not (
        len(points) >= CONFIG.provisional_minimum_votes
        and span >= max(
            CONFIG.provisional_span_minimum_s * jets.s,
            CONFIG.provisional_span_radius_multiplier * median_radius,
        )
        and line_fraction >= CONFIG.provisional_line_fraction_minimum
        and transverse_rms <= CONFIG.provisional_transverse_rms_maximum_s * jets.s
        and collapse_density >= CONFIG.provisional_collapse_density_minimum
    ):
        return None
    all_vote_ids = np.arange(len(focal), dtype=np.uint32)
    all_centered = focal - origin
    all_axial = all_centered @ direction
    all_perpendicular = all_centered - np.outer(all_axial, direction)
    axis_ok = (
        np.abs(jets.weak_directions[vote_source_ids] @ direction)
        >= math.cos(math.radians(CONFIG.reassociation_axis_degrees))
    )
    reassociate = (
        (np.linalg.norm(all_perpendicular, axis=1) <= CONFIG.reassociation_distance_s * jets.s)
        & axis_ok
        & (all_axial >= np.percentile(axial, 1) - CONFIG.reassociation_axial_expansion_s * jets.s)
        & (all_axial <= np.percentile(axial, 99) + CONFIG.reassociation_axial_expansion_s * jets.s)
    )
    associated_vote_ids = all_vote_ids[reassociate]
    associated_source_ids = vote_source_ids[associated_vote_ids]
    origin, direction, associated_axial, line_fraction, transverse_rms = _line_fit(
        focal[associated_vote_ids]
    )
    low, high = np.percentile(associated_axial, (1, 99))
    span = float(high - low)
    candidate = AxisCandidate(
        candidate_id=candidate_id,
        focal_vote_ids=associated_vote_ids,
        source_point_ids=associated_source_ids,
        origin=origin,
        direction=direction,
        axial_low=float(low),
        axial_high=float(high),
        span=span,
        line_fraction=line_fraction,
        transverse_rms=transverse_rms,
        collapse_density=len(associated_vote_ids) / max(span / jets.s, np.finfo(float).tiny),
        radii=None,
        radial_modes=np.empty(0, dtype=np.uint8),
        arc_widths_degrees=None,
        arc_endpoints_radians=None,
        continuation_occupancies=None,
        continuation_offsets_degrees=None,
        retained=False,
        rejection=None,
    )
    raw_points = source.xyz[associated_source_ids]
    centered = raw_points - origin
    axial_source = centered @ direction
    perpendicular = centered - np.outer(axial_source, direction)
    radii = np.linalg.norm(perpendicular, axis=1)
    split, gain, _two_error, _one_error = _optimal_radial_split(radii)
    if split < 0 or gain < CONFIG.radial_two_mode_gain_minimum:
        candidate.rejection = "TWIN_RADIUS_GAIN"
        return candidate
    order = np.argsort(radii, kind="stable")
    modes = np.zeros(len(radii), dtype=np.uint8)
    modes[order[split:]] = 1
    medians = tuple(float(np.median(radii[modes == mode])) for mode in (0, 1))
    separation = medians[1] - medians[0]
    if separation < CONFIG.radial_separation_minimum_s * jets.s:
        candidate.rejection = "TWIN_RADIUS_SEPARATION"
        return candidate
    for mode in (0, 1):
        values = radii[modes == mode]
        mad = float(np.median(np.abs(values - medians[mode])))
        if mad > max(
            CONFIG.radial_mad_floor_s * jets.s,
            CONFIG.radial_mad_radius_fraction * medians[mode],
        ):
            candidate.rejection = "TWIN_RADIUS_MAD"
            return candidate
    edges = np.linspace(float(np.min(axial_source)), float(np.max(axial_source)), CONFIG.axial_bin_count + 1)
    dual_bins = 0
    bin_gaps: list[float] = []
    for index in range(CONFIG.axial_bin_count):
        include = (
            (axial_source >= edges[index])
            & (axial_source <= edges[index + 1] if index == CONFIG.axial_bin_count - 1 else axial_source < edges[index + 1])
        )
        left = radii[include & (modes == 0)]
        right = radii[include & (modes == 1)]
        if len(left) >= CONFIG.axial_minimum_mode_points and len(right) >= CONFIG.axial_minimum_mode_points:
            dual_bins += 1
            bin_gaps.append(float(np.median(right) - np.median(left)))
    if dual_bins < CONFIG.axial_minimum_dual_bins:
        candidate.rejection = "AXIAL_DUAL_RADIUS_SUPPORT"
        return candidate
    gap_mad = float(np.median(np.abs(np.asarray(bin_gaps) - np.median(bin_gaps))))
    if gap_mad > CONFIG.axial_gap_mad_fraction * separation:
        candidate.rejection = "AXIAL_THICKNESS_CONSISTENCY"
        return candidate
    e1, e2 = _axis_frame(direction)
    angles = np.mod(np.arctan2(perpendicular @ e2, perpendicular @ e1), 2 * math.pi)
    intervals = tuple(_arc_interval(angles[modes == mode]) for mode in (0, 1))
    widths = tuple(math.degrees(row[2]) for row in intervals)
    if any(
        width < CONFIG.arc_width_minimum_degrees or width > CONFIG.arc_width_maximum_degrees
        for width in widths
    ):
        candidate.rejection = "OCCUPIED_ARC_WIDTH"
        return candidate
    if abs(widths[0] - widths[1]) > CONFIG.arc_width_agreement_degrees:
        candidate.rejection = "OCCUPIED_ARC_WIDTH_AGREEMENT"
        return candidate
    direct = max(
        _circular_difference(intervals[0][0], intervals[1][0]),
        _circular_difference(intervals[0][1], intervals[1][1]),
    )
    swapped = max(
        _circular_difference(intervals[0][0], intervals[1][1]),
        _circular_difference(intervals[0][1], intervals[1][0]),
    )
    if min(direct, swapped) > math.radians(CONFIG.arc_endpoint_agreement_degrees):
        candidate.rejection = "OCCUPIED_ARC_ENDPOINT_AGREEMENT"
        return candidate
    if swapped < direct:
        intervals = (intervals[0], (intervals[1][1], intervals[1][0], intervals[1][2]))
    candidate.radii = medians
    candidate.radial_modes = modes
    candidate.arc_widths_degrees = widths
    candidate.arc_endpoints_radians = (
        (intervals[0][0], intervals[0][1]),
        (intervals[1][0], intervals[1][1]),
    )
    continuation: list[tuple[float, float]] = []
    for mode in (0, 1):
        continuation.append(
            _continuation_gate(source, jets, candidate, medians[mode], intervals[mode][0], -1.0)
        )
        continuation.append(
            _continuation_gate(source, jets, candidate, medians[mode], intervals[mode][1], 1.0)
        )
    occupancies = tuple(row[0] for row in continuation)
    offsets = tuple(row[1] for row in continuation)
    candidate.continuation_occupancies = occupancies
    candidate.continuation_offsets_degrees = offsets
    if any(value < CONFIG.continuation_occupancy_minimum for value in occupancies):
        candidate.rejection = "TANGENT_CONTINUATION_OCCUPANCY"
        return candidate
    start_difference = _circular_difference(
        intervals[0][0] + math.radians(offsets[0]),
        intervals[1][0] + math.radians(offsets[2]),
    )
    end_difference = _circular_difference(
        intervals[0][1] + math.radians(offsets[1]),
        intervals[1][1] + math.radians(offsets[3]),
    )
    if max(start_difference, end_difference) > math.radians(
        CONFIG.continuation_endpoint_agreement_degrees
    ):
        candidate.rejection = "TANGENT_CONTINUATION_ENDPOINT_AGREEMENT"
        return candidate
    candidate.retained = True
    return candidate


def evaluate(source: CanonicalSource) -> Evaluation:
    jets = estimate_jets(source)
    vote_ids, dense, edges, focal = focal_graph(source, jets)
    components = _components(len(focal), edges, dense)
    if len(components) > CONFIG.provisional_component_cap:
        raise CFACInvalid("INVALID_CFAC_V1_RESOURCE_STOP: provisional component cap")
    candidates: list[AxisCandidate] = []
    for component_id, component in enumerate(components):
        candidate = _evaluate_candidate(
            component_id, component, vote_ids, focal, jets, source
        )
        if candidate is not None:
            candidates.append(candidate)
    if len([row for row in candidates if row.retained]) > CONFIG.final_component_cap:
        raise CFACInvalid("INVALID_CFAC_V1_RESOURCE_STOP: final component cap")
    decision = (
        "CFAC_V1_C006_FUND"
        if any(row.retained for row in candidates)
        else "CFAC_V1_C006_ARCHITECTURE_STOP"
    )
    semantic = semantic_payload(source, jets, dense, edges, candidates, decision)
    return Evaluation(source, jets, dense, edges, candidates, decision, semantic)


def semantic_payload(
    source: CanonicalSource,
    jets: JetField,
    dense: np.ndarray,
    edges: np.ndarray,
    candidates: list[AxisCandidate],
    decision: str,
) -> dict[str, Any]:
    retained = [row for row in candidates if row.retained]
    candidate_rows = [
        {
            "candidate_id": row.candidate_id,
            "focal_vote_count": len(row.focal_vote_ids),
            "source_support_count": len(row.source_point_ids),
            "origin": [float(value) for value in row.origin],
            "direction": [float(value) for value in row.direction],
            "span_over_s": row.span / jets.s,
            "line_fraction": row.line_fraction,
            "transverse_rms_over_s": row.transverse_rms / jets.s,
            "collapse_density": row.collapse_density,
            "radii_over_s": None if row.radii is None else [value / jets.s for value in row.radii],
            "arc_widths_degrees": row.arc_widths_degrees,
            "continuation_occupancies": row.continuation_occupancies,
            "continuation_offsets_degrees": row.continuation_offsets_degrees,
            "retained": row.retained,
            "rejection": row.rejection,
            "source_support_sha256": sha_arrays(
                b"CFAC_V1_CANDIDATE_SOURCE_SUPPORT\0", (row.source_point_ids,)
            ),
        }
        for row in candidates
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "architecture_token": CONFIG.architecture_token,
        "authorization_token": CONFIG.authorization_token,
        "configuration_sha256": CONFIG_SHA256,
        "source": {
            "raw_source_sha256": source.raw_source_sha256,
            "raw_source_bytes": source.raw_source_bytes,
            "point_count": len(source.xyz),
            "canonical_sha256": source.canonical_sha256,
        },
        "scale_s": jets.s,
        "counts": {
            "base_quality_points": int(np.sum(jets.base_quality)),
            "focal_votes": int(np.sum(jets.focal_vote)),
            "dense_focal_votes": int(np.sum(dense)),
            "focal_graph_edges": len(edges),
            "provisional_axes": len(candidates),
            "final_axes": len(retained),
        },
        "hashes": {
            "focal_vote_source_ids": sha_arrays(
                b"CFAC_V1_FOCAL_VOTE_SOURCE_IDS\0",
                (np.flatnonzero(jets.focal_vote).astype(np.uint32),),
            ),
            "dense_focal_vote_ids": sha_arrays(
                b"CFAC_V1_DENSE_FOCAL_VOTE_IDS\0",
                (np.flatnonzero(dense).astype(np.uint32),),
            ),
            "focal_edges": sha_arrays(b"CFAC_V1_FOCAL_EDGES\0", (edges,)),
            "candidate_table": sha_json(candidate_rows),
        },
        "candidate_table": candidate_rows,
        "decision": decision,
        "source_point_serialized": False,
        "target_gold_cad_benchmark_opened": False,
        "runtime_main_production_opened": False,
    }
    payload["semantic_sha256"] = sha_json(payload)
    return payload
