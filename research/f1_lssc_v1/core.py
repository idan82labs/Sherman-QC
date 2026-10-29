"""Frozen source-only implementation of Local Screw Slippage Coherence V1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .config import CONFIG, CONFIG_SHA256, sha_json


class LSSCInvalid(RuntimeError):
    pass


class LSSCResourceStop(RuntimeError):
    pass


@dataclass(slots=True)
class CanonicalSource:
    xyz: np.ndarray
    canonical_sha256: str


@dataclass(slots=True)
class SurfaceField:
    s: float
    normals: np.ndarray
    eligible: np.ndarray
    chart: csr_matrix
    chart_labels: np.ndarray
    anchor_ids: np.ndarray
    anchor_digests: tuple[str, ...]


@dataclass(slots=True)
class PatchSignature:
    valid: bool
    point_count: int
    radius: float
    center: np.ndarray | None
    scale: float | None
    eigenvalues: np.ndarray | None
    eigenvectors: np.ndarray | None
    planar: bool
    planar_normal: np.ndarray | None


@dataclass(slots=True)
class ScrewCandidate:
    anchor_slot: int
    anchor_id: int
    anchor_digest: str
    radius_index: int
    radius: float
    axis_point: np.ndarray
    axis_direction: np.ndarray
    foot: np.ndarray
    translation_leak: float
    rotation_fraction: float


@dataclass(slots=True)
class LocalizedVote:
    candidate: ScrewCandidate
    positive_anchor_slot: int
    negative_anchor_slot: int
    positive_normal: np.ndarray
    negative_normal: np.ndarray
    dihedral_degrees: float
    tail_cross_axis_error_degrees: float


@dataclass(slots=True)
class AxisResult:
    line_point: np.ndarray
    direction: np.ndarray
    vote_slots: np.ndarray
    vote_count: int
    axial_span: float
    rms_residual: float
    p90_residual: float
    occupied_bin_fraction: float
    maximum_empty_run: float
    line_likeness: float


@dataclass(slots=True)
class Evaluation:
    source: CanonicalSource
    field: SurfaceField
    patches: list[list[PatchSignature]]
    raw_candidates: list[ScrewCandidate]
    localized_votes: list[LocalizedVote]
    axes: list[AxisResult]
    semantic_payload: dict[str, Any]


def _sha_arrays(tag: bytes, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(tag)
    for value in arrays:
        raw = np.ascontiguousarray(value).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def canonicalize_xyz(xyz: np.ndarray) -> CanonicalSource:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise LSSCInvalid("LSSC_V1_SOURCE_CONTRACT_STOP: invalid XYZ")
    points = points[np.all(np.isfinite(points), axis=1)].copy()
    points[points == 0.0] = 0.0
    if not CONFIG.input_min_points <= len(points) <= CONFIG.input_max_points:
        raise LSSCInvalid("LSSC_V1_SOURCE_CONTRACT_STOP: point cardinality")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = np.ascontiguousarray(points[order], dtype="<f8")
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    ordered = ordered[keep]
    if len(ordered) < CONFIG.input_min_points:
        raise LSSCInvalid("LSSC_V1_SOURCE_CONTRACT_STOP: too few distinct points")
    return CanonicalSource(
        xyz=ordered,
        canonical_sha256=_sha_arrays(b"LSSC_V1_CANONICAL_XYZ\0", ordered),
    )


def _query_knn(tree: cKDTree, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    distance, index = tree.query(points, k=min(len(points), k + 1), workers=1)
    distance = np.asarray(distance, dtype=np.float64)
    index = np.asarray(index, dtype=np.int64)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    return distance[:, 1 : k + 1], index[:, 1 : k + 1]


def build_surface_field(source: CanonicalSource) -> SurfaceField:
    xyz = source.xyz
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True)
    distances, neighbors = _query_knn(tree, xyz, CONFIG.normal_neighbor_count)
    s = float(np.median(distances[:, CONFIG.spacing_neighbor_rank - 1]))
    if not math.isfinite(s) or s <= 0:
        raise LSSCInvalid("LSSC_V1_SOURCE_CONTRACT_STOP: invalid spacing")
    normals = np.zeros_like(xyz)
    eligible = np.zeros(len(xyz), dtype=bool)
    for point_id in range(len(xyz)):
        within = distances[point_id] <= CONFIG.normal_radius_max_s * s
        ids = neighbors[point_id, within]
        if len(ids) < CONFIG.normal_min_neighbors:
            continue
        local = xyz[ids]
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / len(centered)
        values, vectors = np.linalg.eigh(covariance)
        if values[2] <= 0 or values[1] <= 0:
            continue
        if (
            values[1] / max(values[2], 1e-15) >= CONFIG.normal_surface_ratio_min
            and values[0] / max(values[1], 1e-15) <= CONFIG.normal_thickness_ratio_max
        ):
            eligible[point_id] = True
            normals[point_id] = vectors[:, 0]
    eligible_ids = np.flatnonzero(eligible)
    if len(eligible_ids) < CONFIG.anchor_floor:
        raise LSSCInvalid("LSSC_V1_SOURCE_CONTRACT_STOP: insufficient eligible surface")
    eligible_xyz = xyz[eligible_ids]
    eligible_tree = cKDTree(eligible_xyz, compact_nodes=True, balanced_tree=True)
    chart_distance, chart_local = _query_knn(
        eligible_tree, eligible_xyz, CONFIG.chart_neighbor_count
    )
    rows: list[int] = []
    cols: list[int] = []
    for local_id, global_id in enumerate(eligible_ids):
        proposals = chart_local[local_id]
        proposal_ids = eligible_ids[proposals]
        dots = np.abs(normals[proposal_ids] @ normals[global_id])
        passed = (
            (chart_distance[local_id] <= CONFIG.chart_edge_length_max_s * s)
            & (dots >= CONFIG.chart_normal_abs_dot_min)
        )
        rows.extend([int(global_id)] * int(np.count_nonzero(passed)))
        cols.extend(proposal_ids[passed].astype(int).tolist())
    row_array = np.asarray(rows + cols, dtype=np.int64)
    col_array = np.asarray(cols + rows, dtype=np.int64)
    chart = coo_matrix(
        (np.ones(len(row_array), dtype=np.uint8), (row_array, col_array)),
        shape=(len(xyz), len(xyz)),
    ).tocsr()
    chart.data[:] = 1
    chart.eliminate_zeros()
    _component_count, chart_labels = connected_components(chart, directed=False)
    centroid = np.mean(eligible_xyz, axis=0)
    anchor_rows: list[tuple[bytes, int]] = []
    for global_id in eligible_ids:
        numeric = np.concatenate(
            (
                (xyz[global_id] - centroid) / s,
                distances[global_id, :8] / s,
            )
        )
        numeric = np.rint(numeric / 1e-9) * 1e-9
        key = b"LSSC_V1_ANCHOR" + struct.pack("<11d", *numeric.tolist())
        anchor_rows.append((hashlib.sha256(key).digest(), int(global_id)))
    anchor_rows.sort(key=lambda item: (item[0], item[1]))
    anchor_count = min(
        len(eligible_ids),
        CONFIG.anchor_cap,
        max(CONFIG.anchor_floor, math.ceil(len(eligible_ids) / CONFIG.anchor_divisor)),
    )
    chosen = anchor_rows[:anchor_count]
    return SurfaceField(
        s=s,
        normals=normals,
        eligible=eligible,
        chart=chart,
        chart_labels=np.asarray(chart_labels, dtype=np.int32),
        anchor_ids=np.asarray([item[1] for item in chosen], dtype=np.int64),
        anchor_digests=tuple(item[0].hex() for item in chosen),
    )


def _induced_patch(
    center_id: int,
    candidate_ids: list[int],
    chart: csr_matrix,
) -> np.ndarray:
    allowed = set(int(value) for value in candidate_ids)
    if center_id not in allowed:
        return np.empty(0, dtype=np.int64)
    seen = {center_id}
    stack = [center_id]
    while stack:
        point_id = stack.pop()
        begin, end = chart.indptr[point_id], chart.indptr[point_id + 1]
        for neighbor in chart.indices[begin:end]:
            value = int(neighbor)
            if value in allowed and value not in seen:
                seen.add(value)
                stack.append(value)
    return np.asarray(sorted(seen), dtype=np.int64)


def compute_patches(source: CanonicalSource, field: SurfaceField) -> list[list[PatchSignature]]:
    xyz = source.xyz
    tree = cKDTree(xyz, compact_nodes=True, balanced_tree=True)
    output: list[list[PatchSignature]] = []
    memberships = 0
    for anchor_id in field.anchor_ids:
        anchor_rows: list[PatchSignature] = []
        for radius_factor in CONFIG.patch_radii_over_s:
            radius = radius_factor * field.s
            candidates = tree.query_ball_point(xyz[anchor_id], radius, workers=1)
            candidates = [
                int(value)
                for value in candidates
                if field.eligible[value]
                and field.chart_labels[value] == field.chart_labels[anchor_id]
            ]
            ids = _induced_patch(int(anchor_id), candidates, field.chart)
            memberships += len(ids)
            if memberships > CONFIG.total_patch_membership_max:
                raise LSSCResourceStop("LSSC_V1_RESOURCE_STOP: patch memberships")
            if len(ids) > CONFIG.patch_max_points:
                raise LSSCResourceStop("LSSC_V1_RESOURCE_STOP: patch cardinality")
            if len(ids) < CONFIG.patch_min_points:
                anchor_rows.append(PatchSignature(False, len(ids), radius, None, None, None, None, False, None))
                continue
            points = xyz[ids]
            normals = field.normals[ids]
            center = np.mean(points, axis=0)
            scale = float(np.mean(np.linalg.norm(points - center, axis=1)))
            if scale < 0.5 * field.s:
                anchor_rows.append(PatchSignature(False, len(ids), radius, center, scale, None, None, False, None))
                continue
            normalized = (points - center) / scale
            b = np.column_stack((np.cross(normalized, normals), normals))
            matrix = b.T @ b / len(b)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            if eigenvalues[5] < CONFIG.eigen_floor:
                anchor_rows.append(PatchSignature(False, len(ids), radius, center, scale, eigenvalues, eigenvectors, False, None))
                continue
            planar = False
            planar_normal: np.ndarray | None = None
            ratios = eigenvalues / eigenvalues[5]
            if ratios[2] <= CONFIG.planar_lambda3_max and ratios[3] >= CONFIG.planar_lambda4_min:
                u3 = eigenvectors[:, :3]
                r3, t3 = u3[:3], u3[3:]
                ur, sr, _vhr = np.linalg.svd(r3, full_matrices=True)
                st = np.linalg.svd(t3, compute_uv=False)
                if (
                    sr[1] / max(sr[0], 1e-12) <= CONFIG.planar_rotation_rank_ratio_max
                    and st[1] >= CONFIG.planar_translation_second_singular_min
                ):
                    planar = True
                    planar_normal = ur[:, 0]
            anchor_rows.append(
                PatchSignature(True, len(ids), radius, center, scale, eigenvalues, eigenvectors, planar, planar_normal)
            )
        output.append(anchor_rows)
    return output


def extract_screw_candidates(
    source: CanonicalSource,
    field: SurfaceField,
    patches: list[list[PatchSignature]],
) -> list[ScrewCandidate]:
    candidates: list[ScrewCandidate] = []
    for anchor_slot, anchor_id in enumerate(field.anchor_ids):
        q = source.xyz[anchor_id]
        for radius_index, patch in enumerate(patches[anchor_slot]):
            if not patch.valid or patch.eigenvalues is None or patch.eigenvectors is None:
                continue
            ratios = patch.eigenvalues / patch.eigenvalues[5]
            if not (
                ratios[1] <= CONFIG.cylindrical_lambda2_max
                and ratios[2] >= CONFIG.cylindrical_lambda3_min
                and ratios[3] >= CONFIG.cylindrical_lambda4_min
            ):
                continue
            u2 = patch.eigenvectors[:, :2]
            r2 = u2[:3]
            t2 = u2[3:]
            values, vectors = np.linalg.eigh(r2.T @ r2)
            z_t = vectors[:, 0]
            m_t = u2 @ z_t
            r_t, t_t = m_t[:3], m_t[3:]
            translation_norm = float(np.linalg.norm(t_t))
            total_norm = float(np.linalg.norm(m_t))
            if translation_norm <= 1e-9 or total_norm <= 0:
                continue
            translation_leak = float(np.linalg.norm(r_t) / total_norm)
            if translation_leak > CONFIG.translation_leak_max:
                continue
            z_r = np.asarray((-z_t[1], z_t[0]))
            m_r = u2 @ z_r
            r, t = m_r[:3], m_r[3:]
            r_norm = float(np.linalg.norm(r))
            if r_norm <= 1e-9:
                continue
            d_t = t_t / translation_norm
            d_r = r / r_norm
            if abs(float(np.dot(d_t, d_r))) < CONFIG.translation_rotation_axis_abs_dot_min:
                continue
            t0 = t - np.dot(t, d_t) * d_t
            rotation_fraction = r_norm / math.sqrt(r_norm * r_norm + float(np.dot(t0, t0)))
            if rotation_fraction < CONFIG.rotation_fraction_min:
                continue
            assert patch.center is not None and patch.scale is not None
            axis_normalized = np.cross(r, t0) / float(np.dot(r, r))
            axis_point = patch.center + patch.scale * axis_normalized
            foot = axis_point + d_r * float(np.dot(d_r, q - axis_point))
            if np.linalg.norm(q - foot) > CONFIG.axis_foot_distance_rho_max * patch.radius:
                continue
            direction = d_r.copy()
            first = np.flatnonzero(np.abs(direction) > 1e-12)
            if len(first) and direction[first[0]] < 0:
                direction *= -1.0
            candidates.append(
                ScrewCandidate(
                    anchor_slot=anchor_slot,
                    anchor_id=int(anchor_id),
                    anchor_digest=field.anchor_digests[anchor_slot],
                    radius_index=radius_index,
                    radius=patch.radius,
                    axis_point=axis_point,
                    axis_direction=direction,
                    foot=foot,
                    translation_leak=translation_leak,
                    rotation_fraction=rotation_fraction,
                )
            )
            break
    return candidates


def localize_votes(
    source: CanonicalSource,
    field: SurfaceField,
    patches: list[list[PatchSignature]],
    candidates: list[ScrewCandidate],
) -> list[LocalizedVote]:
    anchor_xyz = source.xyz[field.anchor_ids]
    anchor_tree = cKDTree(anchor_xyz, compact_nodes=True, balanced_tree=True)
    votes: list[LocalizedVote] = []
    for candidate in candidates:
        q = source.xyz[candidate.anchor_id]
        radial = q - candidate.foot
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < CONFIG.radial_min_s * field.s:
            continue
        g = np.cross(candidate.axis_direction, radial)
        g_norm = float(np.linalg.norm(g))
        if g_norm <= 1e-12:
            continue
        g /= g_norm
        nearby = anchor_tree.query_ball_point(q, CONFIG.tail_distance_rho_max * candidate.radius, workers=1)
        sides: list[list[tuple[float, str, int, np.ndarray]]] = [[], []]
        for slot in nearby:
            slot = int(slot)
            if slot == candidate.anchor_slot:
                continue
            if field.chart_labels[field.anchor_ids[slot]] != field.chart_labels[candidate.anchor_id]:
                continue
            patch = patches[slot][candidate.radius_index]
            if not patch.planar or patch.planar_normal is None:
                continue
            delta = anchor_xyz[slot] - q
            distance = float(np.linalg.norm(delta))
            if not CONFIG.tail_distance_rho_min * candidate.radius <= distance <= CONFIG.tail_distance_rho_max * candidate.radius:
                continue
            if abs(float(np.dot(delta, candidate.axis_direction))) / distance > CONFIG.tail_axial_fraction_max:
                continue
            transverse = float(np.dot(delta, g))
            if abs(transverse) < CONFIG.tail_transverse_rho_min * candidate.radius:
                continue
            if abs(transverse) / distance < CONFIG.tail_transverse_fraction_min:
                continue
            side = 0 if transverse > 0 else 1
            sides[side].append((distance, field.anchor_digests[slot], slot, patch.planar_normal))
        for side in sides:
            side.sort(key=lambda value: (value[0], value[1]))
            del side[CONFIG.tail_candidates_per_side :]
        best: tuple[tuple[Any, ...], LocalizedVote] | None = None
        for positive in sides[0]:
            for negative in sides[1]:
                pslot, nslot = positive[2], negative[2]
                pnormal, nnormal = positive[3], negative[3]
                axial_separation = abs(float(np.dot(anchor_xyz[pslot] - anchor_xyz[nslot], candidate.axis_direction)))
                if axial_separation > CONFIG.tail_pair_axial_rho_max * candidate.radius:
                    continue
                if abs(float(np.dot(pnormal, candidate.axis_direction))) > CONFIG.tail_normal_axis_abs_dot_max:
                    continue
                if abs(float(np.dot(nnormal, candidate.axis_direction))) > CONFIG.tail_normal_axis_abs_dot_max:
                    continue
                cosine = float(np.clip(abs(np.dot(pnormal, nnormal)), 0.0, 1.0))
                dihedral = math.degrees(math.acos(cosine))
                if not CONFIG.tail_dihedral_min_degrees <= dihedral <= CONFIG.tail_dihedral_max_degrees:
                    continue
                cross = np.cross(pnormal, nnormal)
                cross_norm = float(np.linalg.norm(cross))
                if cross_norm <= 1e-9:
                    continue
                agreement = abs(float(np.dot(cross / cross_norm, candidate.axis_direction)))
                if agreement < CONFIG.tail_cross_axis_abs_dot_min:
                    continue
                cross_error = math.degrees(math.acos(np.clip(agreement, 0.0, 1.0)))
                vote = LocalizedVote(candidate, pslot, nslot, pnormal, nnormal, dihedral, cross_error)
                key = (positive[0] + negative[0], axial_separation, positive[1], negative[1])
                if best is None or key < best[0]:
                    best = (key, vote)
        if best is not None:
            votes.append(best[1])
    return votes


def _axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def _line_separation(a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray) -> float:
    cross = np.cross(da, db)
    norm = float(np.linalg.norm(cross))
    delta = b - a
    if norm <= 1e-8:
        return float(np.linalg.norm(delta - da * np.dot(delta, da)))
    return abs(float(np.dot(delta, cross))) / norm


def _fit_component(votes: list[LocalizedVote], slots: np.ndarray, s: float) -> AxisResult | None:
    selected = [votes[int(index)] for index in slots]
    directions = np.asarray([vote.candidate.axis_direction for vote in selected])
    scatter = directions.T @ directions
    values, vectors = np.linalg.eigh(scatter)
    direction = vectors[:, -1]
    first = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(first) and direction[first[0]] < 0:
        direction *= -1.0
    points = np.asarray([vote.candidate.axis_point for vote in selected])
    fit_matrix = np.outer(direction, direction)
    rhs = np.zeros(3)
    for point, local_direction in zip(points, directions):
        projector = np.eye(3) - np.outer(local_direction, local_direction)
        fit_matrix += projector
        rhs += projector @ point
    try:
        line_point = np.linalg.solve(fit_matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    line_point -= direction * float(np.dot(line_point, direction))
    feet = np.asarray([vote.candidate.foot for vote in selected])
    axial = (feet - line_point) @ direction
    low, high = np.quantile(axial, (0.05, 0.95))
    span = float(high - low)
    residual = np.linalg.norm((feet - line_point) - np.outer(axial, direction), axis=1)
    deviations = np.asarray([_axis_angle(value, direction) for value in directions])
    dihedrals = np.asarray([vote.dihedral_degrees for vote in selected])
    cross_errors = np.asarray([vote.tail_cross_axis_error_degrees for vote in selected])
    radii = np.asarray([vote.candidate.radius for vote in selected])
    bin_count = max(1, int(math.ceil(span / (CONFIG.axial_bin_width_s * s))))
    bin_ids = np.floor((axial - low) / max(CONFIG.axial_bin_width_s * s, 1e-12)).astype(int)
    occupied = np.zeros(bin_count, dtype=bool)
    valid_bins = (bin_ids >= 0) & (bin_ids < bin_count)
    occupied[bin_ids[valid_bins]] = True
    empty_run = 0
    current = 0
    for value in occupied:
        current = 0 if value else current + 1
        empty_run = max(empty_run, current)
    covariance = np.cov(feet, rowvar=False, bias=True) if len(feet) > 1 else np.zeros((3, 3))
    q = np.linalg.eigvalsh(covariance)[::-1]
    line_likeness = float(q[0] / max(q[1], s * s))
    gates = (
        len(slots) >= CONFIG.component_min_votes
        and span >= CONFIG.component_span_min_s * s
        and float(np.sqrt(np.mean(residual * residual))) <= CONFIG.component_rms_max_s * s
        and float(np.quantile(residual, 0.90)) <= CONFIG.component_p90_max_s * s
        and float(np.median(deviations)) <= CONFIG.component_median_axis_deviation_max_degrees
        and float(np.quantile(deviations, 0.90)) <= CONFIG.component_p90_axis_deviation_max_degrees
        and float(np.quantile(dihedrals, 0.75) - np.quantile(dihedrals, 0.25)) <= CONFIG.component_dihedral_iqr_max_degrees
        and float(np.quantile(cross_errors, 0.90)) <= CONFIG.component_tail_cross_p90_max_degrees
        and float(np.quantile(radii, 0.90) / max(np.quantile(radii, 0.10), 1e-12)) <= CONFIG.component_radius_ratio_max
        and float(np.mean(occupied)) >= CONFIG.occupied_bin_fraction_min
        and empty_run * CONFIG.axial_bin_width_s * s <= CONFIG.maximum_empty_run_s * s
        and line_likeness >= CONFIG.line_likeness_min
    )
    if not gates:
        return None
    return AxisResult(
        line_point=line_point,
        direction=direction,
        vote_slots=slots,
        vote_count=len(slots),
        axial_span=span,
        rms_residual=float(np.sqrt(np.mean(residual * residual))),
        p90_residual=float(np.quantile(residual, 0.90)),
        occupied_bin_fraction=float(np.mean(occupied)),
        maximum_empty_run=empty_run * CONFIG.axial_bin_width_s * s,
        line_likeness=line_likeness,
    )


def build_axes(votes: list[LocalizedVote], s: float) -> list[AxisResult]:
    if not votes:
        return []
    feet = np.asarray([vote.candidate.foot for vote in votes])
    tree = cKDTree(feet, compact_nodes=True, balanced_tree=True)
    edges: set[tuple[int, int]] = set()
    for index, vote in enumerate(votes):
        proposals: list[tuple[float, float, str, int]] = []
        for neighbor in tree.query_ball_point(feet[index], CONFIG.vote_neighbor_distance_s * s, workers=1):
            neighbor = int(neighbor)
            if neighbor == index:
                continue
            other = votes[neighbor]
            angle = _axis_angle(vote.candidate.axis_direction, other.candidate.axis_direction)
            if angle > CONFIG.vote_axis_angle_max_degrees:
                continue
            separation = _line_separation(
                vote.candidate.axis_point,
                vote.candidate.axis_direction,
                other.candidate.axis_point,
                other.candidate.axis_direction,
            )
            if separation > CONFIG.vote_line_separation_max_s * s:
                continue
            radius_ratio = max(vote.candidate.radius, other.candidate.radius) / min(vote.candidate.radius, other.candidate.radius)
            if radius_ratio > CONFIG.vote_radius_ratio_max:
                continue
            if abs(vote.dihedral_degrees - other.dihedral_degrees) > CONFIG.vote_dihedral_difference_max_degrees:
                continue
            delta = feet[neighbor] - feet[index]
            distance = float(np.linalg.norm(delta))
            if distance > CONFIG.vote_axial_test_min_s * s:
                mean_direction = vote.candidate.axis_direction.copy()
                other_direction = other.candidate.axis_direction
                if np.dot(mean_direction, other_direction) < 0:
                    other_direction = -other_direction
                mean_direction += other_direction
                mean_direction /= np.linalg.norm(mean_direction)
                if abs(float(np.dot(delta, mean_direction))) / distance < CONFIG.vote_axial_fraction_min:
                    continue
            proposals.append((distance, angle, other.candidate.anchor_digest, neighbor))
        proposals.sort()
        for _distance, _angle, _digest, neighbor in proposals[: CONFIG.vote_neighbor_cap]:
            edges.add((min(index, neighbor), max(index, neighbor)))
    if not edges:
        return []
    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    graph = coo_matrix(
        (np.ones(2 * len(edge_array), dtype=np.uint8),
         (np.concatenate((edge_array[:, 0], edge_array[:, 1])), np.concatenate((edge_array[:, 1], edge_array[:, 0])))),
        shape=(len(votes), len(votes)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    axes: list[AxisResult] = []
    for component in range(count):
        slots = np.flatnonzero(labels == component)
        fitted = _fit_component(votes, slots, s)
        if fitted is not None:
            axes.append(fitted)
    axes.sort(key=lambda axis: tuple(np.round(np.concatenate((axis.direction, axis.line_point)), 12)))
    return axes


def evaluate(source: CanonicalSource) -> Evaluation:
    field = build_surface_field(source)
    patches = compute_patches(source, field)
    raw = extract_screw_candidates(source, field, patches)
    localized = localize_votes(source, field, patches, raw)
    axes = build_axes(localized, field.s)
    valid_patch_anchors = sum(any(patch.valid for patch in row) for row in patches)
    stage_counts = {
        "source_points": len(source.xyz),
        "eligible_points": int(np.count_nonzero(field.eligible)),
        "anchors": len(field.anchor_ids),
        "anchors_with_valid_patch": valid_patch_anchors,
        "raw_two_motion_candidates": len(raw),
        "bilateral_tail_pairs": len(localized),
        "final_axes": len(axes),
    }
    axis_table = [
        {
            "direction": [round(float(v), 9) for v in axis.direction],
            "line_point": [round(float(v), 9) for v in axis.line_point],
            "vote_count": axis.vote_count,
            "axial_span_over_s": round(axis.axial_span / field.s, 9),
            "rms_over_s": round(axis.rms_residual / field.s, 9),
            "p90_over_s": round(axis.p90_residual / field.s, 9),
            "occupied_bin_fraction": round(axis.occupied_bin_fraction, 9),
            "maximum_empty_run_over_s": round(axis.maximum_empty_run / field.s, 9),
            "line_likeness": round(axis.line_likeness, 9),
        }
        for axis in axes
    ]
    payload: dict[str, Any] = {
        "architecture_token": CONFIG.architecture_token,
        "configuration_sha256": CONFIG_SHA256,
        "source_canonical_sha256": source.canonical_sha256,
        "spacing": round(field.s, 12),
        "counts": stage_counts,
        "raw_two_motion_fraction_of_valid_patch_anchors": round(len(raw) / max(valid_patch_anchors, 1), 12),
        "axis_table": axis_table,
    }
    payload["semantic_sha256"] = sha_json(payload)
    return Evaluation(source, field, patches, raw, localized, axes, payload)
