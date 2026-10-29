"""Frozen raw-coordinate extrusion/profile computation for LEPG-V1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .config import CONFIG, CONFIG_SHA256, sha_json


class LEPGInvalid(RuntimeError):
    pass


@dataclass(slots=True)
class CanonicalSource:
    xyz: np.ndarray
    canonical_sha256: str


@dataclass(slots=True)
class TranslationPair:
    first: int
    second: int
    direction: np.ndarray
    length: float
    unmatched: int
    median_residual: float
    key: str


@dataclass(slots=True)
class DirectionMode:
    direction: np.ndarray
    pair_ids: np.ndarray
    unique_endpoints: int
    median_residual: float


@dataclass(slots=True)
class CrossSection:
    direction: np.ndarray
    basis_b: np.ndarray
    basis_c: np.ndarray
    u: np.ndarray
    xz: np.ndarray
    bin_ids: np.ndarray
    bin_centers: dict[int, float]
    raw_cells_by_bin: dict[int, set[tuple[int, int]]]
    profile_cells: np.ndarray
    profile_cell_ids: np.ndarray
    profile_weights: np.ndarray


@dataclass(slots=True)
class ProfileModel:
    center: np.ndarray
    radii: tuple[float, float]
    shell_ids: tuple[np.ndarray, np.ndarray]
    radial_rms: float
    center_votes: int
    alpha: float | None = None
    beta: float | None = None
    arc_span_degrees: float | None = None
    continuation_coverages: tuple[float, float, float, float] | None = None
    passing_bins: tuple[int, ...] = ()
    minimum_per_bin_arc_coverage: float = 0.0
    minimum_per_bin_ray_coverage: float = 0.0


@dataclass(slots=True)
class AxisResult:
    direction: np.ndarray
    point: np.ndarray
    axial_interval: tuple[float, float]
    radii: tuple[float, float]
    arc_endpoints: tuple[float, float]
    arc_span_degrees: float
    continuation_coverages: tuple[float, float, float, float]
    passing_bin_count: int
    radial_rms: float
    center_votes: int


@dataclass(slots=True)
class Evaluation:
    source: CanonicalSource
    s: float
    centroid: np.ndarray
    accepted_pairs: list[TranslationPair]
    directions: list[DirectionMode]
    cross_sections: list[CrossSection]
    dual_models: list[ProfileModel]
    open_models: list[ProfileModel]
    tangent_models: list[ProfileModel]
    axes: list[AxisResult]
    semantic_payload: dict[str, Any]


def _sha_array(tag: bytes, array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(tag + len(raw).to_bytes(8, "little") + raw).hexdigest()


def canonicalize_xyz(xyz: np.ndarray) -> CanonicalSource:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise LEPGInvalid("LEPG_V1_INVALID_XYZ")
    points = points[np.all(np.isfinite(points), axis=1)].copy()
    points[points == 0.0] = 0.0
    if len(points) > CONFIG.max_source_points:
        raise LEPGInvalid("LEPG_V1_SOURCE_POINT_CAP")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = np.ascontiguousarray(points[order], dtype="<f8")
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    ordered = ordered[keep]
    if len(ordered) < CONFIG.min_source_points:
        raise LEPGInvalid("LEPG_V1_TOO_FEW_POINTS")
    return CanonicalSource(ordered, _sha_array(b"LEPG_V1_CANONICAL_XYZ\0", ordered))


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64).copy()
    value /= np.linalg.norm(value)
    first = np.flatnonzero(np.abs(value) > 1e-12)
    if len(first) and value[first[0]] < 0:
        value *= -1.0
    return value


def _projective_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def _scale_and_signatures(source: CanonicalSource) -> tuple[float, np.ndarray, np.ndarray, cKDTree]:
    tree = cKDTree(source.xyz, compact_nodes=True, balanced_tree=True)
    distances, neighbours = tree.query(source.xyz, k=CONFIG.knn_count, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    neighbours = np.asarray(neighbours, dtype=np.int64)
    s = float(np.median(distances[:, 1]))
    if not math.isfinite(s) or s <= 0:
        raise LEPGInvalid("LEPG_V1_INVALID_SCALE")
    quantum = CONFIG.signature_quantum_s * s
    signature = np.rint(distances[:, np.asarray(CONFIG.signature_ranks)] / quantum).astype(np.int64)
    return s, signature, neighbours[:, 1 : CONFIG.patch_neighbour_count + 1], tree


def translation_pairs(source: CanonicalSource, s: float, signature: np.ndarray, patch_neighbours: np.ndarray, tree: cKDTree) -> list[TranslationPair]:
    xyz = source.xyz
    centroid = np.mean(xyz, axis=0)
    anchor_rows: list[tuple[bytes, bytes, int]] = []
    for point_id, row in enumerate(signature):
        radial = int(round(np.linalg.norm(xyz[point_id] - centroid) / (CONFIG.signature_quantum_s * s)))
        digest = hashlib.sha256(struct.pack(">7q", radial, *row.tolist())).digest()
        tie = np.ascontiguousarray(xyz[point_id], dtype=">f8").tobytes()
        anchor_rows.append((digest, tie, point_id))
    anchor_rows.sort()
    anchors = np.asarray([row[2] for row in anchor_rows[: CONFIG.anchor_limit]], dtype=np.int64)
    signature_tree = cKDTree(signature.astype(np.float64), compact_nodes=True, balanced_tree=True)
    R95 = float(np.quantile(np.linalg.norm(xyz - centroid, axis=1), 0.95))
    maximum = min(CONFIG.pair_distance_max_s * s, 0.5 * (2.0 * R95))
    minimum = CONFIG.pair_distance_min_s * s
    accepted: dict[tuple[int, int], TranslationPair] = {}
    for anchor_id in anchors:
        distances, candidates = signature_tree.query(
            signature[anchor_id].astype(np.float64),
            k=min(len(xyz), CONFIG.signature_query_neighbours + 1),
            workers=1,
        )
        ranked: list[tuple[float, bytes, int]] = []
        for signature_distance, candidate in zip(np.ravel(distances), np.ravel(candidates)):
            candidate = int(candidate)
            if candidate == anchor_id:
                continue
            displacement = xyz[candidate] - xyz[anchor_id]
            length = float(np.linalg.norm(displacement))
            if minimum <= length <= maximum:
                ranked.append((float(signature_distance), np.ascontiguousarray(xyz[candidate], dtype=">f8").tobytes(), candidate))
        ranked.sort()
        for _sig_distance, _tie, candidate in ranked[: CONFIG.signature_candidates_retained][: CONFIG.pairs_evaluated_per_anchor]:
            first, second = sorted((int(anchor_id), candidate))
            pair_key = (first, second)
            if pair_key in accepted:
                continue
            translation = xyz[second] - xyz[first]
            forward_distance, _ = tree.query(xyz[patch_neighbours[first]] + translation, k=1, workers=1)
            reverse_distance, _ = tree.query(xyz[patch_neighbours[second]] - translation, k=1, workers=1)
            forward_distance = np.asarray(forward_distance)
            reverse_distance = np.asarray(reverse_distance)
            forward_matches = int(np.count_nonzero(forward_distance <= CONFIG.patch_match_tolerance_s * s))
            reverse_matches = int(np.count_nonzero(reverse_distance <= CONFIG.patch_match_tolerance_s * s))
            if forward_matches < CONFIG.patch_required_matches_each_direction or reverse_matches < CONFIG.patch_required_matches_each_direction:
                continue
            residuals = np.concatenate((forward_distance, reverse_distance))
            key_bytes = np.ascontiguousarray(xyz[[first, second]], dtype="<f8").tobytes()
            accepted[pair_key] = TranslationPair(
                first,
                second,
                _canonical_direction(translation),
                float(np.linalg.norm(translation)),
                24 - forward_matches - reverse_matches,
                float(np.median(residuals)),
                hashlib.sha256(key_bytes).hexdigest(),
            )
            if len(accepted) >= CONFIG.accepted_pair_hard_cap:
                raise LEPGInvalid("LEPG_V1_ACCEPTED_PAIR_HARD_CAP")
    return sorted(accepted.values(), key=lambda value: (value.unmatched, value.median_residual, -value.length, value.key))


def direction_modes(pairs: list[TranslationPair]) -> list[DirectionMode]:
    if not pairs:
        return []
    directions = np.asarray([pair.direction for pair in pairs])
    cosine = math.cos(math.radians(CONFIG.direction_mean_shift_half_angle_degrees))
    seeds: list[np.ndarray] = []
    for pair in pairs[: CONFIG.direction_seed_limit]:
        direction = pair.direction.copy()
        for _ in range(CONFIG.direction_mean_shift_iterations):
            inliers = np.abs(directions @ direction) >= cosine
            if not np.any(inliers):
                break
            matrix = directions[inliers].T @ directions[inliers]
            direction = _canonical_direction(np.linalg.eigh(matrix)[1][:, -1])
        seeds.append(direction)
    modes: list[DirectionMode] = []
    for direction in seeds:
        inliers = np.flatnonzero(np.abs(directions @ direction) >= cosine)
        endpoints = np.unique(np.asarray([(pairs[i].first, pairs[i].second) for i in inliers], dtype=np.int64))
        if len(inliers) < CONFIG.direction_min_pairs or len(endpoints) < CONFIG.direction_min_unique_endpoints:
            continue
        mode = DirectionMode(
            direction,
            inliers,
            len(endpoints),
            float(np.median([pairs[i].median_residual for i in inliers])),
        )
        if any(_projective_angle(mode.direction, existing.direction) <= CONFIG.direction_merge_angle_degrees for existing in modes):
            continue
        modes.append(mode)
    modes.sort(key=lambda value: (-len(value.pair_ids), -value.unique_endpoints, value.median_residual, tuple(value.direction)))
    return modes[: CONFIG.direction_candidate_limit]


def _perpendicular_basis(xyz: np.ndarray, centroid: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projector = np.eye(3) - np.outer(direction, direction)
    covariance = (xyz - centroid).T @ (xyz - centroid) / len(xyz)
    projected = projector @ covariance @ projector
    values, vectors = np.linalg.eigh(projected)
    if abs(values[-1] - values[-2]) <= 1e-8 * max(abs(values[-1]), 1.0):
        axes = np.eye(3)
        basis_axis = min(axes, key=lambda value: abs(float(np.dot(value, direction))))
        b = projector @ basis_axis
    else:
        b = vectors[:, -1]
    b = _canonical_direction(b)
    c = np.cross(direction, b)
    c /= np.linalg.norm(c)
    return b, c


def cross_section(source: CanonicalSource, s: float, direction: np.ndarray) -> CrossSection | None:
    xyz = source.xyz
    centroid = np.mean(xyz, axis=0)
    b, c = _perpendicular_basis(xyz, centroid, direction)
    delta = xyz - centroid
    u = delta @ direction
    xz = np.column_stack((delta @ b, delta @ c))
    width = CONFIG.axial_bin_width_s * s
    bin_ids = np.floor((u - float(np.min(u))) / width).astype(np.int64)
    cell_size = CONFIG.profile_cell_size_s * s
    raw_cells_by_bin: dict[int, set[tuple[int, int]]] = {}
    bin_centers: dict[int, float] = {}
    cell_bins: dict[tuple[int, int], set[int]] = {}
    for bin_id in np.unique(bin_ids):
        ids = np.flatnonzero(bin_ids == bin_id)
        if len(ids) < CONFIG.eligible_axial_bin_min_points:
            continue
        bin_id = int(bin_id)
        bin_centers[bin_id] = float(np.mean(u[ids]))
        raw = set((int(math.floor(x / cell_size)), int(math.floor(z / cell_size))) for x, z in xz[ids])
        raw_cells_by_bin[bin_id] = raw
        dilated = set()
        for cell in raw:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    dilated.add((cell[0] + dx, cell[1] + dz))
        for cell in dilated:
            cell_bins.setdefault(cell, set()).add(bin_id)
    survivors: list[tuple[tuple[int, int], set[int]]] = []
    for cell, bins in cell_bins.items():
        if len(bins) < CONFIG.consensus_cell_min_axial_bins:
            continue
        centers = [bin_centers[value] for value in bins]
        if max(centers) - min(centers) >= CONFIG.consensus_cell_min_center_span_s * s:
            survivors.append((cell, bins))
    if len(survivors) > CONFIG.consensus_profile_hard_cap:
        raise LEPGInvalid("LEPG_V1_PROFILE_CELL_HARD_CAP")
    if len(survivors) < CONFIG.consensus_profile_min_cells:
        return None
    survivors.sort(key=lambda value: value[0])
    cell_ids = np.asarray([value[0] for value in survivors], dtype=np.int64)
    cells = (cell_ids.astype(np.float64) + 0.5) * cell_size
    weights = np.asarray([min(len(value[1]), 8) for value in survivors], dtype=np.float64)
    return CrossSection(direction, b, c, u, xz, bin_ids, bin_centers, raw_cells_by_bin, cells, cell_ids, weights)


def _circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    ab, ac = b - a, c - a
    cross = float(ab[0] * ac[1] - ab[1] * ac[0])
    sine = abs(cross) / max(float(np.linalg.norm(ab) * np.linalg.norm(ac)), 1e-15)
    if sine < CONFIG.triplet_min_sine:
        return None
    matrix = 2.0 * np.asarray((b - a, c - a))
    rhs = np.asarray((np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)))
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    return center, float(np.linalg.norm(a - center))


def dual_circle_models(section: CrossSection, s: float, D95: float) -> list[ProfileModel]:
    points, weights, cell_ids = section.profile_cells, section.profile_weights, section.profile_cell_ids
    hashes = [hashlib.sha256(struct.pack(">2q", int(cell[0]), int(cell[1]))).digest() for cell in cell_ids]
    anchors = sorted(range(len(points)), key=lambda i: (-weights[i], hashes[i]))[: CONFIG.profile_anchor_limit]
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    votes: dict[tuple[int, int], list[np.ndarray]] = {}
    for anchor in anchors:
        nearby = tree.query_ball_point(points[anchor], CONFIG.profile_neighbour_distance_max_s * s, workers=1)
        nearby = [int(i) for i in nearby if i != anchor and np.linalg.norm(points[i] - points[anchor]) >= CONFIG.profile_neighbour_distance_min_s * s]
        nearby.sort(key=lambda i: (float(np.linalg.norm(points[i] - points[anchor])), hashes[i]))
        nearby = nearby[: CONFIG.profile_neighbour_limit]
        triplets = []
        for left in range(len(nearby)):
            for right in range(left + 1, len(nearby)):
                ids = sorted((anchor, nearby[left], nearby[right]))
                key = hashlib.sha256(b"".join(struct.pack(">2q", *cell_ids[i].tolist()) for i in ids)).digest()
                triplets.append((key, ids))
        triplets.sort(key=lambda value: value[0])
        for _key, ids in triplets[: CONFIG.triplets_per_anchor]:
            circle = _circumcircle(points[ids[0]], points[ids[1]], points[ids[2]])
            if circle is None:
                continue
            center, radius = circle
            if not CONFIG.circle_radius_min_s * s <= radius <= min(CONFIG.circle_radius_max_s * s, 0.40 * D95):
                continue
            center_cell = tuple(np.floor(center / (CONFIG.center_vote_cell_size_s * s)).astype(int).tolist())
            votes.setdefault(center_cell, []).append(center)
    candidates = [(len(values), cell, np.mean(values, axis=0)) for cell, values in votes.items() if len(values) >= CONFIG.center_vote_min]
    candidates.sort(key=lambda value: (-value[0], value[1]))
    models: list[ProfileModel] = []
    for vote_count, _cell, center_seed in candidates[: CONFIG.center_candidate_limit]:
        radii = np.linalg.norm(points - center_seed, axis=1)
        radial_bins = np.rint(radii / (CONFIG.radial_histogram_bin_s * s)).astype(int)
        histogram: dict[int, float] = {}
        for radial_bin, weight in zip(radial_bins, weights):
            histogram[int(radial_bin)] = histogram.get(int(radial_bin), 0.0) + float(weight)
        peaks = sorted(histogram, key=lambda value: (-histogram[value], value))[: CONFIG.radial_peak_limit]
        for inner_bin in peaks:
            for outer_bin in peaks:
                if inner_bin >= outer_bin:
                    continue
                current_radii = np.asarray((inner_bin, outer_bin), dtype=float) * CONFIG.radial_histogram_bin_s * s
                if not CONFIG.skin_gap_min_s * s <= current_radii[1] - current_radii[0] <= CONFIG.skin_gap_max_s * s:
                    continue
                center = center_seed.copy()
                assignments: tuple[np.ndarray, np.ndarray] | None = None
                for _ in range(CONFIG.common_center_refit_iterations):
                    distances = np.linalg.norm(points - center, axis=1)
                    shell = np.argmin(np.abs(distances[:, None] - current_radii[None, :]), axis=1)
                    residual = np.min(np.abs(distances[:, None] - current_radii[None, :]), axis=1)
                    ids0 = np.flatnonzero((shell == 0) & (residual <= CONFIG.shell_assignment_tolerance_s * s))
                    ids1 = np.flatnonzero((shell == 1) & (residual <= CONFIG.shell_assignment_tolerance_s * s))
                    if len(ids0) < CONFIG.shell_min_profile_cells or len(ids1) < CONFIG.shell_min_profile_cells:
                        assignments = None
                        break
                    ids = np.concatenate((ids0, ids1))
                    design = np.column_stack((points[ids, 0], points[ids, 1], np.isin(ids, ids0).astype(float), np.isin(ids, ids1).astype(float)))
                    target = -np.sum(points[ids] ** 2, axis=1)
                    root_weight = np.sqrt(weights[ids])
                    beta, *_ = np.linalg.lstsq(design * root_weight[:, None], target * root_weight, rcond=None)
                    center = -0.5 * beta[:2]
                    squared = np.sum(center * center) - beta[2:]
                    if np.any(squared <= 0):
                        assignments = None
                        break
                    current_radii = np.sqrt(squared)
                    if current_radii[0] > current_radii[1]:
                        current_radii = current_radii[::-1]
                    assignments = (ids0, ids1)
                if assignments is None:
                    continue
                ids = np.concatenate(assignments)
                shell_radius = np.concatenate((np.full(len(assignments[0]), current_radii[0]), np.full(len(assignments[1]), current_radii[1])))
                residual = np.linalg.norm(points[ids] - center, axis=1) - shell_radius
                rms = math.sqrt(float(np.average(residual * residual, weights=weights[ids])))
                if rms <= CONFIG.radial_rms_max_s * s:
                    models.append(ProfileModel(center, (float(current_radii[0]), float(current_radii[1])), assignments, rms, vote_count))
    models.sort(key=lambda model: (model.radial_rms, -model.center_votes, tuple(model.center), model.radii))
    return models


def _angle_delta_degrees(a: float, b: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def _shortest_interval(angles: np.ndarray, weights: np.ndarray, mass_fraction: float) -> tuple[float, float, float]:
    order = np.argsort(angles)
    a = np.mod(angles[order], 2.0 * math.pi)
    w = weights[order]
    a2 = np.concatenate((a, a + 2.0 * math.pi))
    w2 = np.concatenate((w, w))
    target = mass_fraction * float(np.sum(w))
    best = (math.inf, 0.0, 0.0)
    right = 0
    mass = 0.0
    for left in range(len(a)):
        while right < left + len(a) and mass < target:
            mass += w2[right]
            right += 1
        if mass >= target and right > left:
            span = a2[right - 1] - a2[left]
            if span < best[0]:
                best = (span, a2[left], a2[right - 1])
        mass -= w2[left]
    return best[1], best[2], best[0]


def open_sector_models(section: CrossSection, models: list[ProfileModel], s: float) -> list[ProfileModel]:
    points, weights = section.profile_cells, section.profile_weights
    output: list[ProfileModel] = []
    for model in models:
        intervals = []
        valid = True
        for ids in model.shell_ids:
            angles = np.arctan2(points[ids, 1] - model.center[1], points[ids, 0] - model.center[0])
            intervals.append(_shortest_interval(angles, weights[ids], CONFIG.shortest_interval_mass_fraction))
        for _alpha, _beta, span in intervals:
            degrees = math.degrees(span)
            if not CONFIG.arc_span_min_degrees <= degrees <= CONFIG.arc_span_max_degrees:
                valid = False
        if not valid:
            continue
        if _angle_delta_degrees(intervals[0][0], intervals[1][0]) > CONFIG.shell_endpoint_disagreement_max_degrees or _angle_delta_degrees(intervals[0][1], intervals[1][1]) > CONFIG.shell_endpoint_disagreement_max_degrees:
            continue
        alpha = 0.5 * (intervals[0][0] + intervals[1][0])
        beta = 0.5 * (intervals[0][1] + intervals[1][1])
        span = beta - alpha
        for ids in model.shell_ids:
            angles = np.mod(np.arctan2(points[ids, 1] - model.center[1], points[ids, 0] - model.center[0]) - alpha, 2.0 * math.pi)
            inside = angles <= span
            outside_mass = 1.0 - float(np.sum(weights[ids][inside]) / np.sum(weights[ids]))
            if outside_mass > CONFIG.arc_outside_mass_fraction_max:
                valid = False
                break
            bin_count = max(1, int(math.ceil(math.degrees(span) / CONFIG.angle_bin_degrees)))
            bins = np.floor(np.degrees(angles[inside]) / CONFIG.angle_bin_degrees).astype(int)
            coverage = len(np.unique(np.clip(bins, 0, bin_count - 1))) / bin_count
            if coverage < CONFIG.arc_angular_bin_coverage_min:
                valid = False
                break
        if valid:
            model.alpha, model.beta = alpha, beta
            model.arc_span_degrees = math.degrees(span)
            output.append(model)
    return output


def _classify_profile_points(points: np.ndarray, center: np.ndarray, radius: float, alpha: float, beta: float, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radial = np.linalg.norm(points - center, axis=1)
    angles = np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]) - alpha, 2.0 * math.pi)
    span = beta - alpha
    arc = (np.abs(radial - radius) <= CONFIG.shell_assignment_tolerance_s * s) & (angles <= span)
    tangent = lambda phi: np.asarray((-math.sin(phi), math.cos(phi)))
    radial_vec = lambda phi: np.asarray((math.cos(phi), math.sin(phi)))
    start_origin = center + radius * radial_vec(alpha)
    end_origin = center + radius * radial_vec(beta)
    start_direction = -tangent(alpha)
    end_direction = tangent(beta)
    start_delta = points - start_origin
    end_delta = points - end_origin
    start_along = start_delta @ start_direction
    end_along = end_delta @ end_direction
    start_perp = np.abs(start_delta[:, 0] * start_direction[1] - start_delta[:, 1] * start_direction[0])
    end_perp = np.abs(end_delta[:, 0] * end_direction[1] - end_delta[:, 1] * end_direction[0])
    required = max(CONFIG.ray_required_length_min_s * s, min(CONFIG.ray_required_length_max_s * s, CONFIG.ray_required_length_inner_fraction * radius))
    start = (start_along >= 0) & (start_along <= required) & (start_perp <= CONFIG.ray_line_tolerance_s * s)
    end = (end_along >= 0) & (end_along <= required) & (end_perp <= CONFIG.ray_line_tolerance_s * s)
    return arc, start, end, start_along, end_along


def tangent_models(section: CrossSection, models: list[ProfileModel], s: float) -> list[ProfileModel]:
    output: list[ProfileModel] = []
    points = section.profile_cells
    for model in models:
        assert model.alpha is not None and model.beta is not None
        coverages: list[float] = []
        passed = True
        for radius in model.radii:
            arc, start, end, start_along, end_along = _classify_profile_points(points, model.center, radius, model.alpha, model.beta, s)
            required = max(CONFIG.ray_required_length_min_s * s, min(CONFIG.ray_required_length_max_s * s, CONFIG.ray_required_length_inner_fraction * model.radii[0]))
            expected = max(1, int(math.ceil((required - CONFIG.ray_start_distance_s * s) / (CONFIG.ray_bin_length_s * s))))
            local_coverages = []
            for mask, along in ((start, start_along), (end, end_along)):
                gated = mask & (along >= CONFIG.ray_start_distance_s * s)
                bins = np.floor((along[gated] - CONFIG.ray_start_distance_s * s) / (CONFIG.ray_bin_length_s * s)).astype(int)
                coverage = len(np.unique(np.clip(bins, 0, expected - 1))) / expected
                local_coverages.append(coverage)
                if coverage < CONFIG.ray_bin_coverage_min:
                    passed = False
            coverages.extend(local_coverages)
            assigned = arc | start | end
            assigned_cells = set(map(tuple, section.profile_cell_ids[assigned].tolist()))
            if assigned_cells:
                seen: set[tuple[int, int]] = set()
                largest: set[tuple[int, int]] = set()
                for seed in assigned_cells:
                    if seed in seen:
                        continue
                    component = {seed}
                    seen.add(seed)
                    stack = [seed]
                    while stack:
                        cell = stack.pop()
                        for dx in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                neighbor = (cell[0] + dx, cell[1] + dz)
                                if neighbor in assigned_cells and neighbor not in seen:
                                    seen.add(neighbor)
                                    component.add(neighbor)
                                    stack.append(neighbor)
                    if len(component) > len(largest):
                        largest = component
                if len(largest) / len(assigned_cells) < CONFIG.shell_contour_component_fraction_min:
                    passed = False
            else:
                passed = False
        if passed:
            model.continuation_coverages = tuple(float(value) for value in coverages)
            output.append(model)
    return output


def verify_axial_bins(section: CrossSection, model: ProfileModel, s: float) -> bool:
    assert model.alpha is not None and model.beta is not None
    passing: list[int] = []
    minimum_arc = 1.0
    minimum_ray = 1.0
    cell_size = CONFIG.profile_cell_size_s * s
    for bin_id, cell_set in section.raw_cells_by_bin.items():
        cells = (np.asarray(sorted(cell_set), dtype=np.float64) + 0.5) * cell_size
        shell_arc_coverages = []
        ray_coverages = []
        for radius in model.radii:
            arc, start, end, start_along, end_along = _classify_profile_points(cells, model.center, radius, model.alpha, model.beta, s)
            arc_bins_expected = max(1, int(math.ceil((model.arc_span_degrees or 0.0) / CONFIG.angle_bin_degrees)))
            angles = np.mod(np.degrees(np.arctan2(cells[:, 1] - model.center[1], cells[:, 0] - model.center[0]) - model.alpha), 360.0)
            arc_bins = np.floor(angles[arc] / CONFIG.angle_bin_degrees).astype(int)
            shell_arc_coverages.append(len(np.unique(np.clip(arc_bins, 0, arc_bins_expected - 1))) / arc_bins_expected)
            required = max(CONFIG.ray_required_length_min_s * s, min(CONFIG.ray_required_length_max_s * s, CONFIG.ray_required_length_inner_fraction * model.radii[0]))
            expected = max(1, int(math.ceil((required - CONFIG.ray_start_distance_s * s) / (CONFIG.ray_bin_length_s * s))))
            for mask, along in ((start, start_along), (end, end_along)):
                gated = mask & (along >= CONFIG.ray_start_distance_s * s)
                bins = np.floor((along[gated] - CONFIG.ray_start_distance_s * s) / (CONFIG.ray_bin_length_s * s)).astype(int)
                ray_coverages.append(len(np.unique(np.clip(bins, 0, expected - 1))) / expected)
        minimum_arc = min(minimum_arc, *shell_arc_coverages)
        minimum_ray = min(minimum_ray, *ray_coverages)
        if min(shell_arc_coverages) >= CONFIG.per_bin_arc_coverage_min and min(ray_coverages) >= CONFIG.per_bin_ray_coverage_min:
            passing.append(bin_id)
    if len(passing) < CONFIG.passing_axial_bins_min:
        return False
    centers = [section.bin_centers[value] for value in passing]
    span = max(centers) - min(centers)
    between = [value for value, center in section.bin_centers.items() if min(centers) <= center <= max(centers)]
    if span < CONFIG.passing_axial_center_span_s * s or len(passing) / max(len(between), 1) < CONFIG.passing_bin_fraction_min:
        return False
    model.passing_bins = tuple(sorted(passing))
    model.minimum_per_bin_arc_coverage = minimum_arc
    model.minimum_per_bin_ray_coverage = minimum_ray
    return True


def _line_distance(a: AxisResult, b: AxisResult) -> float:
    cross = np.cross(a.direction, b.direction)
    norm = np.linalg.norm(cross)
    delta = b.point - a.point
    if norm <= 1e-9:
        return float(np.linalg.norm(delta - a.direction * np.dot(delta, a.direction)))
    return abs(float(np.dot(delta, cross))) / float(norm)


def evaluate(source: CanonicalSource) -> Evaluation:
    s, signature, patch_neighbours, tree = _scale_and_signatures(source)
    centroid = np.mean(source.xyz, axis=0)
    R95 = float(np.quantile(np.linalg.norm(source.xyz - centroid, axis=1), 0.95))
    D95 = 2.0 * R95
    pairs = translation_pairs(source, s, signature, patch_neighbours, tree)
    directions = direction_modes(pairs)
    sections: list[CrossSection] = []
    dual: list[ProfileModel] = []
    opened: list[ProfileModel] = []
    tangents: list[ProfileModel] = []
    axes: list[AxisResult] = []
    for mode in directions:
        section = cross_section(source, s, mode.direction)
        if section is None:
            continue
        sections.append(section)
        models = dual_circle_models(section, s, D95)
        dual.extend(models)
        open_models = open_sector_models(section, models, s)
        opened.extend(open_models)
        tangent = tangent_models(section, open_models, s)
        tangents.extend(tangent)
        for model in tangent:
            if not verify_axial_bins(section, model, s):
                continue
            passing_centers = [section.bin_centers[value] for value in model.passing_bins]
            u_median = float(np.median(passing_centers))
            point = centroid + model.center[0] * section.basis_b + model.center[1] * section.basis_c + u_median * mode.direction
            point -= mode.direction * float(np.dot(point, mode.direction))
            axis = AxisResult(
                mode.direction.copy(),
                point,
                (min(passing_centers) - 0.5 * CONFIG.axial_bin_width_s * s, max(passing_centers) + 0.5 * CONFIG.axial_bin_width_s * s),
                model.radii,
                (float(model.alpha), float(model.beta)),
                float(model.arc_span_degrees),
                model.continuation_coverages or (0.0, 0.0, 0.0, 0.0),
                len(model.passing_bins),
                model.radial_rms,
                model.center_votes,
            )
            duplicate = None
            for existing in axes:
                if (
                    _projective_angle(axis.direction, existing.direction) <= CONFIG.duplicate_direction_degrees
                    and _line_distance(axis, existing) <= CONFIG.duplicate_line_distance_s * s
                    and max(abs(axis.radii[0] - existing.radii[0]), abs(axis.radii[1] - existing.radii[1])) <= CONFIG.duplicate_radius_difference_s * s
                    and max(_angle_delta_degrees(axis.arc_endpoints[0], existing.arc_endpoints[0]), _angle_delta_degrees(axis.arc_endpoints[1], existing.arc_endpoints[1])) <= CONFIG.duplicate_endpoint_difference_degrees
                ):
                    duplicate = existing
                    break
            if duplicate is None:
                axes.append(axis)
            else:
                key_new = (-axis.passing_bin_count, -model.minimum_per_bin_arc_coverage, -model.minimum_per_bin_ray_coverage, axis.radial_rms, -axis.center_votes, tuple(axis.direction), tuple(axis.point))
                old_model = next(value for value in tangents if value.radii == duplicate.radii and value.radial_rms == duplicate.radial_rms)
                key_old = (-duplicate.passing_bin_count, -old_model.minimum_per_bin_arc_coverage, -old_model.minimum_per_bin_ray_coverage, duplicate.radial_rms, -duplicate.center_votes, tuple(duplicate.direction), tuple(duplicate.point))
                if key_new < key_old:
                    axes[axes.index(duplicate)] = axis
    if len(axes) > CONFIG.final_axis_hard_cap:
        raise LEPGInvalid("LEPG_V1_FINAL_AXIS_HARD_CAP")
    axes.sort(key=lambda axis: (tuple(np.round(axis.direction, 12)), tuple(np.round(axis.point, 12))))
    counts = {
        "source_points": len(source.xyz),
        "accepted_translation_pairs": len(pairs),
        "translation_direction_candidates": len(directions),
        "translation_consensus_profiles": len(sections),
        "consensus_profile_cells_max": max((len(value.profile_cells) for value in sections), default=0),
        "dual_circle_models": len(dual),
        "open_sector_models": len(opened),
        "four_tangent_models": len(tangents),
        "final_axes": len(axes),
    }
    axis_table = [{
        "direction": [round(float(value), 9) for value in axis.direction],
        "point": [round(float(value), 9) for value in axis.point],
        "axial_interval": [round(float(value), 9) for value in axis.axial_interval],
        "radii": [round(float(value), 9) for value in axis.radii],
        "arc_span_degrees": round(axis.arc_span_degrees, 9),
        "continuation_coverages": [round(float(value), 9) for value in axis.continuation_coverages],
        "passing_bin_count": axis.passing_bin_count,
        "radial_rms_over_s": round(axis.radial_rms / s, 9),
    } for axis in axes]
    payload: dict[str, Any] = {
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "source_canonical_sha256": source.canonical_sha256,
        "spacing": round(s, 12),
        "counts": counts,
        "direction_table": [{"direction": [round(float(v), 9) for v in mode.direction], "pair_support": len(mode.pair_ids), "unique_endpoints": mode.unique_endpoints, "median_patch_residual_over_s": round(mode.median_residual / s, 9)} for mode in directions],
        "axis_table": axis_table,
    }
    payload["semantic_sha256"] = sha_json(payload)
    return Evaluation(source, s, centroid, pairs, directions, sections, dual, opened, tangents, axes, payload)
