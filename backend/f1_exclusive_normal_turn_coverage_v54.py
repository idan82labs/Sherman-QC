"""Frozen V54 exclusive normal-turn coverage selector.

V54 is an append-only, source-only intervention over the exact frozen V52
candidate population.  It reconstructs only scan-local support membership
from frozen support records and never regenerates candidate geometry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

from backend.cad_bend_extractor import CADBendExtractor
from backend.f1_cylindrical_consensus_recovery_v52 import _represented
from research.f1_multifamily_selector_v2.model import endpoints


PROFILE = "F1_V54_EXCLUSIVE_NORMAL_TURN_COVERAGE_V1"
PRECOMMIT_TOKEN = "V54_ENTC_MECHANISM_LOCKED"
TERMINAL_FAILURE_TOKEN = "V54_ENTC_TERMINAL_RETIRE"
AXIAL_BIN_COUNT = 24
MAXIMUM_CANDIDATE_COUNT = 4096
MAXIMUM_CANDIDATE_EDGE_INCIDENCES = 8_000_000
MAXIMUM_ADDITIONS = 16
MAXIMUM_PUBLIC_BYTES = 256 * 1024
PREDICTED_MEMORY_LIMIT_BYTES = 12 * 1024**3
RSS_LIMIT_BYTES = 16 * 1024**3
WALL_LIMIT_SECONDS = 300.0


class V54ScanAbstention(RuntimeError):
    """A scan-level V54 failure; the enclosing one-shot run must retire."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


@dataclass(frozen=True)
class MeshTurnEvidence:
    vertices: np.ndarray
    triangles: np.ndarray
    edge_vertices: np.ndarray
    edge_directions: np.ndarray
    edge_midpoints: np.ndarray
    edge_h: np.ndarray
    edge_boundary_near: np.ndarray
    spacing_mm: float
    adjacency_offsets: np.ndarray
    adjacency_vertices: np.ndarray
    eligible_edge_count: int
    structural_zero_mass_edge_count: int
    excess_turn_floor: float


@dataclass
class CandidateEvidence:
    digest: str
    start: np.ndarray
    end: np.ndarray
    axis: np.ndarray
    length_mm: float
    support_vertices: np.ndarray
    edge_ids: np.ndarray
    h: np.ndarray
    w: np.ndarray
    t: np.ndarray
    bins: np.ndarray
    bin_h: np.ndarray
    bin_w: np.ndarray
    effective_edge_count: float
    boundary_contamination: float
    coherent_mask: int
    total_w: float
    static_reason: str | None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_limits(started: float) -> None:
    if time.perf_counter() - started >= WALL_LIMIT_SECONDS:
        raise V54ScanAbstention("ABSTAIN_WALL_TIME")
    if _rss_bytes() >= RSS_LIMIT_BYTES:
        raise V54ScanAbstention("ABSTAIN_ACTUAL_RSS")


def _neumaier(values: Sequence[float] | np.ndarray) -> float:
    total = 0.0
    correction = 0.0
    for raw in values:
        value = float(raw)
        candidate = total + value
        if abs(total) >= abs(value):
            correction += (total - candidate) + value
        else:
            correction += (value - candidate) + total
        total = candidate
    return total + correction


def _stable_median(values: np.ndarray, identifiers: np.ndarray) -> float:
    if not len(values):
        raise V54ScanAbstention("ABSTAIN_EMPTY_MEDIAN")
    order = np.lexsort((identifiers, values))
    ordered = values[order]
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))


def _principal_dyad_vectors(
    xx: np.ndarray,
    xy: np.ndarray,
    xz: np.ndarray,
    yy: np.ndarray,
    yz: np.ndarray,
    zz: np.ndarray,
) -> np.ndarray:
    """Principal eigenvectors of symmetric 3x3 dyads without BLAS/LAPACK."""

    q = (xx + yy + zz) / 3.0
    p2 = (
        (xx - q) ** 2
        + (yy - q) ** 2
        + (zz - q) ** 2
        + 2.0 * (xy * xy + xz * xz + yz * yz)
    )
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))
    safe_p = np.where(p > 1e-30, p, 1.0)
    bxx = (xx - q) / safe_p
    byy = (yy - q) / safe_p
    bzz = (zz - q) / safe_p
    bxy = xy / safe_p
    bxz = xz / safe_p
    byz = yz / safe_p
    determinant = (
        bxx * (byy * bzz - byz * byz)
        - bxy * (bxy * bzz - byz * bxz)
        + bxz * (bxy * byz - byy * bxz)
    )
    phi = np.arccos(np.clip(0.5 * determinant, -1.0, 1.0)) / 3.0
    largest = q + 2.0 * p * np.cos(phi)

    r0 = np.column_stack((xx - largest, xy, xz))
    r1 = np.column_stack((xy, yy - largest, yz))
    r2 = np.column_stack((xz, yz, zz - largest))
    crosses = np.stack(
        (np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)), axis=1
    )
    norms2 = np.einsum("nij,nij->ni", crosses, crosses)
    choice = np.argmax(norms2, axis=1)
    vectors = crosses[np.arange(len(crosses)), choice]
    norms = np.sqrt(np.maximum(norms2[np.arange(len(crosses)), choice], 0.0))

    fallback = norms <= 1e-24
    if np.any(fallback):
        diagonal = np.column_stack((xx, yy, zz))
        axes = np.argmax(diagonal, axis=1)
        vectors[fallback] = 0.0
        vectors[np.flatnonzero(fallback), axes[fallback]] = 1.0
        norms[fallback] = 1.0
    vectors /= norms[:, None]
    first = np.where(
        np.abs(vectors[:, 0]) > 1e-15,
        vectors[:, 0],
        np.where(np.abs(vectors[:, 1]) > 1e-15, vectors[:, 1], vectors[:, 2]),
    )
    vectors[first < 0.0] *= -1.0
    if not np.isfinite(vectors).all():
        raise V54ScanAbstention("ABSTAIN_NONFINITE_VERTEX_NORMAL")
    return vectors


def _csr_neighbors(vertex_count: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources = np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64, copy=False)
    targets = np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64, copy=False)
    order = np.lexsort((targets, sources))
    sources = sources[order]
    targets = targets[order]
    counts = np.bincount(sources, minlength=vertex_count)
    offsets = np.empty(vertex_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, targets


def _build_mesh_turn_evidence(scan_path: Path, started: float) -> tuple[o3d.geometry.TriangleMesh, MeshTurnEvidence]:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or not len(triangles)
    ):
        raise V54ScanAbstention("ABSTAIN_NO_VALID_TRIANGULAR_MESH")
    in_range = np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)
    safe_triangles = triangles[in_range]
    if not len(safe_triangles):
        raise V54ScanAbstention("ABSTAIN_NO_VALID_TRIANGULAR_MESH")
    p0 = vertices[safe_triangles[:, 0]]
    p1 = vertices[safe_triangles[:, 1]]
    p2 = vertices[safe_triangles[:, 2]]
    crosses = np.cross(p1 - p0, p2 - p0)
    double_area = np.sqrt(np.einsum("ij,ij->i", crosses, crosses))
    valid_local = np.isfinite(double_area) & (double_area > 1e-15)
    valid_face_count = int(np.count_nonzero(valid_local))
    invalid_face_count = len(triangles) - valid_face_count
    if invalid_face_count / len(triangles) > 0.01:
        raise V54ScanAbstention("ABSTAIN_DEGENERATE_FACE_FRACTION")
    face_triangles = safe_triangles[valid_local]
    face_crosses = crosses[valid_local]
    face_double_area = double_area[valid_local]
    face_normals = face_crosses / face_double_area[:, None]
    face_area = 0.5 * face_double_area

    dyad = np.column_stack(
        (
            face_area * face_normals[:, 0] * face_normals[:, 0],
            face_area * face_normals[:, 0] * face_normals[:, 1],
            face_area * face_normals[:, 0] * face_normals[:, 2],
            face_area * face_normals[:, 1] * face_normals[:, 1],
            face_area * face_normals[:, 1] * face_normals[:, 2],
            face_area * face_normals[:, 2] * face_normals[:, 2],
        )
    )
    components = [np.zeros(len(vertices), dtype=np.float64) for _ in range(6)]
    # Face order is the source order. np.add.at preserves that deterministic
    # order and does not dispatch to a BLAS eigensolver.
    for corner in range(3):
        ids = face_triangles[:, corner]
        for index in range(6):
            np.add.at(components[index], ids, dyad[:, index])
    normals = _principal_dyad_vectors(*components)

    raw_edges = np.concatenate(
        (
            face_triangles[:, (0, 1)],
            face_triangles[:, (1, 2)],
            face_triangles[:, (2, 0)],
        ),
        axis=0,
    )
    raw_edges.sort(axis=1)
    order = np.lexsort((raw_edges[:, 1], raw_edges[:, 0]))
    ordered_edges = raw_edges[order]
    new_group = np.ones(len(ordered_edges), dtype=bool)
    new_group[1:] = np.any(ordered_edges[1:] != ordered_edges[:-1], axis=1)
    starts = np.flatnonzero(new_group)
    merged_edges = ordered_edges[starts]
    incident_counts = np.diff(np.append(starts, len(ordered_edges)))
    if not len(merged_edges):
        raise V54ScanAbstention("ABSTAIN_NO_VALID_TRIANGULAR_MESH")
    vectors = vertices[merged_edges[:, 1]] - vertices[merged_edges[:, 0]]
    lengths = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    structural = (incident_counts == 2) & np.isfinite(lengths) & (lengths > 0.0)
    structural_ids = np.flatnonzero(structural)
    spacing = _stable_median(lengths[structural], structural_ids)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise V54ScanAbstention("ABSTAIN_INVALID_EDGE_SPACING")

    boundary = np.zeros(len(vertices), dtype=bool)
    boundary_vertices = np.unique(merged_edges[incident_counts != 2].reshape(-1))
    boundary[boundary_vertices] = True
    for _ in range(2):
        expanded = boundary.copy()
        np.logical_or.at(expanded, merged_edges[:, 0], boundary[merged_edges[:, 1]])
        np.logical_or.at(expanded, merged_edges[:, 1], boundary[merged_edges[:, 0]])
        boundary = expanded

    eligible = structural & (lengths >= 0.25 * spacing) & (lengths <= 4.0 * spacing)
    edge_ids = np.flatnonzero(eligible)
    if not len(edge_ids):
        raise V54ScanAbstention("ABSTAIN_NO_ELIGIBLE_NORMAL_TURN_EDGES")
    edges = merged_edges[eligible]
    edge_lengths = lengths[eligible]
    edge_vectors = vectors[eligible]
    directions = edge_vectors / edge_lengths[:, None]
    dots = np.einsum("ij,ij->i", normals[edges[:, 0]], normals[edges[:, 1]])
    delta = np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    kappa = delta / edge_lengths
    median_kappa = _stable_median(kappa, edge_ids)
    absolute = np.abs(kappa - median_kappa)
    mad = _stable_median(absolute, edge_ids)
    kappa0 = median_kappa + 3.0 * 1.4826 * mad
    h = edge_lengths * np.maximum(0.0, delta - kappa0 * edge_lengths)
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    near = boundary[edges[:, 0]] | boundary[edges[:, 1]]
    offsets, adjacency = _csr_neighbors(len(vertices), merged_edges)
    predicted = (
        vertices.nbytes
        + triangles.nbytes
        + merged_edges.nbytes
        + adjacency.nbytes
        + offsets.nbytes
        + directions.nbytes
        + midpoints.nbytes
        + h.nbytes
        + 512 * 1024**2
    )
    if predicted > PREDICTED_MEMORY_LIMIT_BYTES:
        raise V54ScanAbstention("ABSTAIN_PREDICTED_MEMORY")
    _check_limits(started)
    return mesh, MeshTurnEvidence(
        vertices=vertices,
        triangles=triangles,
        edge_vertices=edges,
        edge_directions=directions,
        edge_midpoints=midpoints,
        edge_h=h,
        edge_boundary_near=near,
        spacing_mm=float(spacing),
        adjacency_offsets=offsets,
        adjacency_vertices=adjacency,
        eligible_edge_count=int(len(edges)),
        structural_zero_mass_edge_count=int(np.count_nonzero(h == 0.0)),
        excess_turn_floor=float(kappa0),
    )


def _canonical_candidate(row: Mapping[str, Any]) -> tuple[bytes, str, np.ndarray, np.ndarray, np.ndarray, float]:
    if row.get("span_endpoints") is not None:
        line = np.asarray(row.get("span_endpoints"), dtype=np.float64)
    elif row.get("line_start") is not None and row.get("line_end") is not None:
        line = np.asarray((row.get("line_start"), row.get("line_end")), dtype=np.float64)
    else:
        line = np.asarray(endpoints(dict(row)), dtype=np.float64)
    if line.shape != (2, 3) or not np.isfinite(line).all():
        raise ValueError("invalid candidate geometry")
    delta = line[1] - line[0]
    length = float(np.sqrt(np.dot(delta, delta)))
    if length <= 1e-12:
        raise ValueError("zero candidate span")
    axis = delta / length
    significant = np.flatnonzero(np.abs(axis) > 1e-15)
    if not len(significant):
        raise ValueError("indeterminate candidate axis")
    if axis[int(significant[0])] < 0.0:
        line = line[::-1].copy()
        axis = -axis
    line[line == 0.0] = 0.0
    payload = struct.pack("<6d", *(float(value) for value in line.reshape(-1)))
    digest = hashlib.sha256(payload).hexdigest()
    return payload, digest, line[0], line[1], axis, length


class _SupportResolver:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        evidence: MeshTurnEvidence,
        pool: Mapping[str, Any],
        research_inputs: Mapping[str, Any],
    ) -> None:
        self.mesh = mesh
        self.evidence = evidence
        self.pool = pool
        self.research_inputs = research_inputs
        self.by_id = {str(row.get("candidate_id") or ""): row for row in pool.get("candidates") or ()}
        atom_graph = dict(research_inputs.get("atom_graph") or {})
        self.atom_size = float(atom_graph.get("voxel_size_mm") or 0.0)
        self.atom_keys = {
            str(row.get("atom_id") or ""): tuple(int(v) for v in row.get("voxel_key") or ())
            for row in atom_graph.get("atoms") or ()
        }
        self.strip_atoms: dict[str, tuple[str, ...]] = {}
        current = dict(research_inputs.get("current_zero_shot_result") or {})
        for row in current.get("bend_support_strips") or ():
            hint = dict(row.get("axis_line_hint") or {})
            provenance = dict(row.get("merge_provenance") or {})
            atoms = sorted(
                {
                    str(value)
                    for value in (
                        list(hint.get("source_atom_ids") or ())
                        + list(provenance.get("merged_atom_ids") or ())
                    )
                    if str(value)
                }
            )
            self.strip_atoms[str(row.get("strip_id") or "")] = tuple(atoms)
        self.plane_by_id = {
            str(row.get("plane_id") or ""): row for row in pool.get("plane_table") or ()
        }
        self.parent_keys = {
            str(row.get("parent_atom_id") or ""): tuple(int(v) for v in row.get("voxel_key") or ())
            for row in pool.get("support_parent_table") or ()
        }
        self.parent_size = float(pool.get("transition_parent_voxel_size_mm") or 0.0)
        self._atom_vertex_buckets: dict[tuple[int, int, int], np.ndarray] | None = None
        self._parent_vertex_buckets: dict[tuple[int, int, int], np.ndarray] | None = None
        self._mesh_support: dict[str, np.ndarray] | None = None

    def _buckets(self, size: float) -> dict[tuple[int, int, int], np.ndarray]:
        if not math.isfinite(size) or size <= 0.0:
            return {}
        keys = np.floor(self.evidence.vertices / size).astype(np.int64)
        order = np.lexsort((np.arange(len(keys), dtype=np.int64), keys[:, 2], keys[:, 1], keys[:, 0]))
        ordered = keys[order]
        starts = np.ones(len(order), dtype=bool)
        starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
        positions = np.flatnonzero(starts)
        result: dict[tuple[int, int, int], np.ndarray] = {}
        for index, begin in enumerate(positions):
            end = positions[index + 1] if index + 1 < len(positions) else len(order)
            result[tuple(int(v) for v in ordered[begin])] = np.sort(order[begin:end])
        return result

    def _from_atom_ids(self, atom_ids: Sequence[str]) -> np.ndarray | None:
        if self._atom_vertex_buckets is None:
            self._atom_vertex_buckets = self._buckets(self.atom_size)
        groups = []
        for atom_id in sorted(set(str(value) for value in atom_ids)):
            key = self.atom_keys.get(atom_id)
            values = None if key is None else self._atom_vertex_buckets.get(key)
            if values is None or not len(values):
                return None
            groups.append(values)
        return np.unique(np.concatenate(groups)) if groups else None

    def _from_parent_ids(self, parent_ids: Sequence[str]) -> np.ndarray | None:
        if self._parent_vertex_buckets is None:
            self._parent_vertex_buckets = self._buckets(self.parent_size)
        groups = []
        for parent_id in sorted(set(str(value) for value in parent_ids)):
            key = self.parent_keys.get(parent_id)
            values = None if key is None else self._parent_vertex_buckets.get(key)
            if values is None or not len(values):
                return None
            groups.append(values)
        return np.unique(np.concatenate(groups)) if groups else None

    def _mesh_banks(self) -> dict[str, np.ndarray]:
        if self._mesh_support is not None:
            return self._mesh_support
        extraction = CADBendExtractor().extract_from_mesh(self.mesh)
        surfaces = {int(row.surface_id): row for row in extraction.surfaces}
        bends = {f"scan_mesh_adjacency::SMA{int(row.bend_id)}": row for row in extraction.bends}
        result: dict[str, np.ndarray] = {}
        for candidate_id, row in sorted(self.by_id.items()):
            if str(row.get("candidate_source") or "") != "scan_mesh_adjacency":
                continue
            bend = bends.get(candidate_id)
            if bend is None:
                continue
            owners = (int(row.get("surface1_id") or -1), int(row.get("surface2_id") or -1))
            if owners != (int(bend.surface1_id), int(bend.surface2_id)):
                continue
            groups = []
            for owner in owners:
                surface = surfaces.get(owner)
                if surface is None:
                    groups = []
                    break
                face_ids = np.asarray(surface.face_indices, dtype=np.int64)
                if np.any((face_ids < 0) | (face_ids >= len(self.evidence.triangles))):
                    groups = []
                    break
                groups.append(np.unique(self.evidence.triangles[face_ids].reshape(-1)))
            if groups:
                result[candidate_id] = np.unique(np.concatenate(groups))
        self._mesh_support = result
        return result

    def resolve(self, row: Mapping[str, Any]) -> np.ndarray | None:
        direct = tuple(str(value) for value in row.get("support_atom_ids") or ())
        if direct:
            if bool(row.get("support_atom_ids_truncated")):
                return None
            return self._from_atom_ids(direct)
        source = str(row.get("candidate_source") or "")
        if source == "control_support_strip":
            return self._from_atom_ids(self.strip_atoms.get(str(row.get("source_candidate_id") or ""), ()))
        if source in {"cylindrical_standard", "cylindrical_wide"}:
            atoms: list[str] = []
            for seed in sorted(set(str(value) for value in row.get("seed_ids") or ())):
                seed_id = seed if "::" in seed else "bend_hypothesis::" + seed
                seed_row = self.by_id.get(seed_id)
                if seed_row is None or bool(seed_row.get("support_atom_ids_truncated")):
                    return None
                atoms.extend(str(value) for value in seed_row.get("support_atom_ids") or ())
            return self._from_atom_ids(atoms)
        if source == "plane_lattice":
            parents: list[str] = []
            plane_ids = tuple(str(value) for value in row.get("incident_plane_ids") or ())
            if len(plane_ids) != 2 or plane_ids[0] == plane_ids[1]:
                return None
            for plane_id in plane_ids:
                plane = self.plane_by_id.get(plane_id)
                if plane is None:
                    return None
                parents.extend(str(value) for value in plane.get("support_parent_ids") or ())
            return self._from_parent_ids(parents)
        if source == "scan_mesh_adjacency":
            return self._mesh_banks().get(str(row.get("candidate_id") or ""))
        return None


def _one_ring(evidence: MeshTurnEvidence, support: np.ndarray) -> np.ndarray:
    groups = [support]
    for vertex in support:
        begin = int(evidence.adjacency_offsets[int(vertex)])
        end = int(evidence.adjacency_offsets[int(vertex) + 1])
        groups.append(evidence.adjacency_vertices[begin:end])
    return np.unique(np.concatenate(groups))


def _candidate_evidence(
    digest: str,
    start: np.ndarray,
    end: np.ndarray,
    axis: np.ndarray,
    length_mm: float,
    support: np.ndarray,
    evidence: MeshTurnEvidence,
) -> CandidateEvidence:
    static_reason: str | None = None
    if len(support) < 64:
        static_reason = "SUPPORT_VERTEX_COUNT"
    if length_mm < 24.0 * evidence.spacing_mm and static_reason is None:
        static_reason = "MINIMUM_CANDIDATE_SPAN"
    region = _one_ring(evidence, support)
    in_support = np.zeros(len(evidence.vertices), dtype=bool)
    in_region = np.zeros(len(evidence.vertices), dtype=bool)
    in_support[support] = True
    in_region[region] = True
    edges = evidence.edge_vertices
    mask = (
        in_region[edges[:, 0]]
        & in_region[edges[:, 1]]
        & (in_support[edges[:, 0]] | in_support[edges[:, 1]])
        & (evidence.edge_h > 0.0)
    )
    ids = np.flatnonzero(mask)
    if len(ids):
        t_all = (evidence.edge_midpoints[ids] - start) @ axis
        within = (t_all >= 0.0) & (t_all <= length_mm)
        ids = ids[within]
        t = t_all[within]
    else:
        t = np.zeros(0, dtype=np.float64)
    if not len(ids):
        if static_reason is None:
            static_reason = "EMPTY_EDGE_SET"
        h = w = bins = np.zeros(0, dtype=np.float64)
        bins = bins.astype(np.int64)
        bin_h = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
        bin_w = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
        effective = 0.0
        boundary = 0.0
        coherent_mask = 0
        total_w = 0.0
    else:
        h = evidence.edge_h[ids]
        directions = evidence.edge_directions[ids]
        align = directions @ axis
        w = h * np.maximum(0.0, 1.0 - align * align)
        bins = np.floor((t / length_mm) * AXIAL_BIN_COUNT).astype(np.int64)
        bins[t == length_mm] = AXIAL_BIN_COUNT - 1
        bins = np.clip(bins, 0, AXIAL_BIN_COUNT - 1)
        bin_h = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
        bin_w = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
        bin_h2 = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
        for index in range(AXIAL_BIN_COUNT):
            local = np.flatnonzero(bins == index)
            bin_h[index] = _neumaier(h[local])
            bin_w[index] = _neumaier(w[local])
            bin_h2[index] = _neumaier(h[local] * h[local])
        total_h = _neumaier(h)
        total_h2 = _neumaier(h * h)
        effective = total_h * total_h / total_h2 if total_h2 > 0.0 else 0.0
        total_w = _neumaier(w)
        boundary_w = _neumaier(w[evidence.edge_boundary_near[ids]])
        boundary = boundary_w / total_w if total_w > 0.0 else math.inf
        coherent_mask = 0
        for index in range(AXIAL_BIN_COUNT):
            effective_bin = (
                bin_h[index] * bin_h[index] / bin_h2[index]
                if bin_h2[index] > 0.0
                else 0.0
            )
            ratio = bin_w[index] / bin_h[index] if bin_h[index] > 0.0 else 0.0
            if bin_h[index] > 0.0 and effective_bin >= 3.0 and ratio >= 0.75:
                coherent_mask |= 1 << index
        if effective < 48.0 and static_reason is None:
            static_reason = "EFFECTIVE_EDGE_COUNT"
        if boundary > 0.25 and static_reason is None:
            static_reason = "BOUNDARY_CONTAMINATION"
        if coherent_mask.bit_count() < 16 and static_reason is None:
            static_reason = "COHERENT_BIN_COUNT"
        if total_w <= 0.0 and static_reason is None:
            static_reason = "ZERO_COMPATIBLE_TURN_MASS"
    return CandidateEvidence(
        digest=digest,
        start=start,
        end=end,
        axis=axis,
        length_mm=float(length_mm),
        support_vertices=support,
        edge_ids=ids,
        h=h,
        w=w,
        t=t,
        bins=bins,
        bin_h=bin_h,
        bin_w=bin_w,
        effective_edge_count=float(effective),
        boundary_contamination=float(boundary),
        coherent_mask=int(coherent_mask),
        total_w=float(total_w),
        static_reason=static_reason,
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, edge_ids: np.ndarray, fraction: float) -> float:
    order = np.lexsort((edge_ids, values))
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = _neumaier(ordered_weights)
    target = fraction * total
    cumulative = 0.0
    correction = 0.0
    for value, raw_weight in zip(ordered_values, ordered_weights):
        weight = float(raw_weight)
        candidate = cumulative + weight
        if abs(cumulative) >= abs(weight):
            correction += (cumulative - candidate) + weight
        else:
            correction += (weight - candidate) + cumulative
        cumulative = candidate
        if cumulative + correction >= target:
            return float(value)
    return float(ordered_values[-1])


def _dynamic_metrics(candidate: CandidateEvidence, ownership: np.ndarray) -> dict[str, Any]:
    if candidate.static_reason is not None:
        return {"admissible": False, "reason": candidate.static_reason}
    gain = np.maximum(0.0, candidate.w - ownership[candidate.edge_ids])
    delta = _neumaier(gain)
    rho = delta / candidate.total_w if candidate.total_w > 0.0 else 0.0
    bin_g = np.zeros(AXIAL_BIN_COUNT, dtype=np.float64)
    novel_mask = 0
    for index in range(AXIAL_BIN_COUNT):
        local = np.flatnonzero(candidate.bins == index)
        bin_g[index] = _neumaier(gain[local])
        if (
            candidate.coherent_mask & (1 << index)
            and candidate.bin_w[index] > 0.0
            and bin_g[index] / candidate.bin_w[index] >= 0.60
        ):
            novel_mask |= 1 << index
    novel_count = novel_mask.bit_count()
    one_hole = max(
        sum(bool(novel_mask & (1 << index)) for index in range(start, start + 10))
        for start in range(AXIAL_BIN_COUNT - 9)
    )
    positive = gain > 0.0
    if np.any(positive):
        low = _weighted_percentile(
            candidate.t[positive], gain[positive], candidate.edge_ids[positive], 0.05
        )
        high = _weighted_percentile(
            candidate.t[positive], gain[positive], candidate.edge_ids[positive], 0.95
        )
        span_fraction = max(0.0, high - low) / candidate.length_mm
    else:
        span_fraction = 0.0
    reason = None
    if novel_count < 12:
        reason = "NOVEL_BIN_COUNT"
    elif one_hole < 9:
        reason = "NOVEL_CONTIGUITY"
    elif rho < 0.55:
        reason = "MARGINAL_OWNERSHIP"
    elif span_fraction < 0.75:
        reason = "MARGINAL_SPAN"
    values = (delta, rho, span_fraction, *bin_g)
    if not all(math.isfinite(float(value)) for value in values):
        raise V54ScanAbstention("ABSTAIN_NONFINITE_SELECTION_TUPLE")
    return {
        "admissible": reason is None,
        "reason": reason,
        "gain": gain,
        "delta": float(delta),
        "rho": float(rho),
        "bin_g": bin_g,
        "novel_mask": int(novel_mask),
        "novel_count": int(novel_count),
        "one_hole": int(one_hole),
        "span_fraction": float(span_fraction),
    }


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 12) for value in values]


def _finalize(
    v52_result: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(v52_result))
    result.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v54_exclusive_normal_turn_coverage_result",
            "profile": PROFILE,
            "v52_retained": True,
            "candidate_count": len(predictions),
            "predictions": predictions,
            "exclusive_normal_turn_coverage_audit": audit,
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
        }
    )
    compact = json.dumps(
        {"profile": PROFILE, "predictions": predictions, "audit": audit},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result["v54_result_sha256"] = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) + 1 > MAXIMUM_PUBLIC_BYTES:
        raise V54ScanAbstention("ABSTAIN_PUBLIC_JSON_SIZE")
    return result


def compose_v54(
    v52_result: Mapping[str, Any],
    pool: Mapping[str, Any],
    research_inputs: Mapping[str, Any],
    scan_path: Path,
) -> dict[str, Any]:
    """Execute the single frozen ENTC selection for one scan."""

    started = time.perf_counter()
    if v52_result.get("source_only") is not True or v52_result.get("research_only") is not True:
        raise ValueError("V54 requires a source-only research V52 input")
    if int(research_inputs.get("source_point_count") or -1) <= 0:
        raise V54ScanAbstention("ABSTAIN_SOURCE_SUPPORT_RECORD_MISMATCH")
    mesh, evidence = _build_mesh_turn_evidence(scan_path, started)
    if int(research_inputs.get("source_point_count") or -1) != len(evidence.vertices):
        raise V54ScanAbstention("ABSTAIN_SOURCE_SUPPORT_RECORD_MISMATCH")
    resolver = _SupportResolver(mesh, evidence, pool, research_inputs)
    raw_candidates = list(pool.get("candidates") or ())
    groups: dict[bytes, dict[str, Any]] = {}
    candidate_id_to_payload: dict[str, bytes] = {}
    local_exclusions: list[dict[str, str]] = []
    for row in raw_candidates:
        candidate_id = str(row.get("candidate_id") or "")
        try:
            payload, digest, start, end, axis, length = _canonical_candidate(row)
        except Exception:
            local_exclusions.append({"candidate_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(), "reason": "INVALID_GEOMETRY"})
            continue
        candidate_id_to_payload[candidate_id] = payload
        support = resolver.resolve(row)
        group = groups.setdefault(
            payload,
            {
                "digest": digest,
                "start": start,
                "end": end,
                "axis": axis,
                "length": length,
                "supports": [],
                "candidate_ids": [],
            },
        )
        group["candidate_ids"].append(candidate_id)
        if support is not None and len(support):
            group["supports"].append(support)
    if len(groups) > MAXIMUM_CANDIDATE_COUNT:
        raise V54ScanAbstention("ABSTAIN_CANDIDATE_CAP")

    selected_payloads: set[bytes] = set()
    for prediction in v52_result.get("predictions") or ():
        source_id = str(prediction.get("source_hypothesis_id") or "")
        payload = candidate_id_to_payload.get(source_id)
        if payload is None or not groups.get(payload, {}).get("supports"):
            raise V54ScanAbstention("ABSTAIN_V52_PROVENANCE_MAPPING")
        selected_payloads.add(payload)

    models: dict[str, CandidateEvidence] = {}
    payload_to_digest: dict[bytes, str] = {}
    incidences = 0
    for payload in sorted(groups, key=lambda value: hashlib.sha256(value).digest()):
        group = groups[payload]
        digest = str(group["digest"])
        payload_to_digest[payload] = digest
        if not group["supports"]:
            local_exclusions.append({"candidate_sha256": digest, "reason": "UNRESOLVED_SOURCE_BANK"})
            continue
        support = np.unique(np.concatenate(group["supports"]))
        model = _candidate_evidence(
            digest,
            group["start"],
            group["end"],
            group["axis"],
            group["length"],
            support,
            evidence,
        )
        models[digest] = model
        incidences += len(model.edge_ids)
        if incidences > MAXIMUM_CANDIDATE_EDGE_INCIDENCES:
            raise V54ScanAbstention("ABSTAIN_CANDIDATE_EDGE_INCIDENCE_CAP")
        _check_limits(started)

    ownership = np.zeros(len(evidence.edge_h), dtype=np.float64)
    selected_digests = {payload_to_digest[payload] for payload in selected_payloads}
    for digest in sorted(selected_digests):
        model = models.get(digest)
        if model is None:
            raise V54ScanAbstention("ABSTAIN_V52_PROVENANCE_MAPPING")
        ownership[model.edge_ids] = np.maximum(ownership[model.edge_ids], model.w)

    prefix = [deepcopy(row) for row in v52_result.get("predictions") or ()]
    predictions = [deepcopy(row) for row in prefix]
    additions: list[dict[str, Any]] = []
    dynamic_final: dict[str, dict[str, Any]] = {}
    while True:
        admissible: list[tuple[tuple[float, int, int, int, float, float], str, dict[str, Any]]] = []
        for digest in sorted(models):
            if digest in selected_digests:
                continue
            model = models[digest]
            metrics = _dynamic_metrics(model, ownership)
            dynamic_final[digest] = metrics
            if not metrics["admissible"]:
                continue
            proposed = {"line_start": model.start.tolist(), "line_end": model.end.tolist()}
            if _represented(proposed, predictions):
                metrics["admissible"] = False
                metrics["reason"] = "FROZEN_V52_NMS_CONFLICT"
                continue
            key = (
                float(metrics["delta"]),
                int(metrics["novel_count"]),
                int(model.coherent_mask.bit_count()),
                int(metrics["one_hole"]),
                float(metrics["rho"]),
                float(model.length_mm / evidence.spacing_mm),
            )
            admissible.append((key, digest, metrics))
        if not admissible:
            break
        admissible.sort(key=lambda row: tuple(-value for value in row[0]) + (row[1],))
        if len(additions) >= MAXIMUM_ADDITIONS:
            raise V54ScanAbstention("ABSTAIN_RUNAWAY_MODEL_COUNT")
        _key, digest, metrics = admissible[0]
        model = models[digest]
        prediction = {
            "bend_id": "TRV54-" + digest[:20],
            "line_start": model.start.tolist(),
            "line_end": model.end.tolist(),
            "selection_route": "exclusive_normal_turn_coverage_v54",
            "source_hypothesis_id": digest,
            "source_kind": "exclusive_normal_turn_coverage",
        }
        predictions.append(prediction)
        ownership[model.edge_ids] = np.maximum(ownership[model.edge_ids], model.w)
        selected_digests.add(digest)
        additions.append(
            {
                "candidate_sha256": digest,
                "line_geometry": {
                    "line_start": model.start.tolist(),
                    "line_end": model.end.tolist(),
                },
                "support_vertex_count": int(len(model.support_vertices)),
                "effective_edge_count": round(model.effective_edge_count, 12),
                "boundary_contamination": round(model.boundary_contamination, 12),
                "coherent_bin_mask": int(model.coherent_mask),
                "novel_bin_mask": int(metrics["novel_mask"]),
                "marginal_coverage": round(float(metrics["delta"]), 12),
                "marginal_ownership_fraction": round(float(metrics["rho"]), 12),
                "marginal_span_fraction": round(float(metrics["span_fraction"]), 12),
                "bin_turn_mass": _round_list(model.bin_h),
                "bin_compatible_mass": _round_list(model.bin_w),
                "bin_marginal_mass": _round_list(metrics["bin_g"]),
            }
        )
        _check_limits(started)

    decisions = list(local_exclusions)
    for digest in sorted(models):
        model = models[digest]
        if digest in {row["candidate_sha256"] for row in additions}:
            reason = "SELECTED"
        elif digest in selected_digests:
            reason = "INITIAL_V52"
        elif model.static_reason is not None:
            reason = model.static_reason
        else:
            reason = str(dynamic_final.get(digest, {}).get("reason") or "NOT_SELECTED")
        decisions.append({"candidate_sha256": digest, "reason": reason})
    decisions.sort(key=lambda row: (row["candidate_sha256"], row["reason"]))
    decision_bytes = json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    bin_summary = [
        {
            "candidate_sha256": digest,
            "coherent_bin_mask": model.coherent_mask,
            "bin_h": _round_list(model.bin_h),
            "bin_w": _round_list(model.bin_w),
        }
        for digest, model in sorted(models.items())
    ]
    bin_bytes = json.dumps(bin_summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    reason_counts: dict[str, int] = {}
    for row in decisions:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    audit = {
        "status": "COMMITTED",
        "precommit_token": PRECOMMIT_TOKEN,
        "frozen_v52_prefix_count": len(prefix),
        "raw_candidate_count": len(raw_candidates),
        "exact_merged_candidate_count": len(groups),
        "resolved_candidate_count": len(models),
        "candidate_edge_incidence_count": int(incidences),
        "eligible_mesh_edge_count": evidence.eligible_edge_count,
        "zero_turn_mass_edge_count": evidence.structural_zero_mass_edge_count,
        "median_structural_edge_length_mm": round(evidence.spacing_mm, 12),
        "excess_turn_floor": round(evidence.excess_turn_floor, 12),
        "candidate_decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "candidate_bin_summary_sha256": hashlib.sha256(bin_bytes).hexdigest(),
        "candidate_reason_counts": dict(sorted(reason_counts.items())),
        "recovered_candidate_count": len(additions),
        "recovered_candidates": additions,
        "source_identifier_serialization": False,
    }
    if predictions[: len(prefix)] != prefix:
        raise V54ScanAbstention("ABSTAIN_V52_PREFIX_MUTATION")
    _check_limits(started)
    return _finalize(v52_result, predictions, audit)
