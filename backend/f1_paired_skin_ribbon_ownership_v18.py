"""Bounded source-only paired-skin ribbon ownership probe (PSRO_V1).

This research module implements the single V18 primitive authorized in the
2026-08-14 post-V17 ruling.  It deliberately does not import or inspect V17.
The native mesh is held transiently; only fixed-size 224-byte probe records
may be serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import open3d as o3d

from backend.cad_bend_extractor import CADBendExtractor, MeshSurface


PROFILE = "PAIRED_SKIN_RIBBON_OWNERSHIP_V1"
STATIONS = (1.0 / 16.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 15.0 / 16.0)
MANDATORY_MASK = (1 << 0) | (1 << 2) | (1 << 4)
MIN_STATIONS = 4
MAX_NORMAL_OPPOSITION_DOT = -math.cos(math.radians(15.0))
MAX_RECIPROCAL_RATIO = 1.20
MAX_SIDE_THICKNESS_RATIO = 1.35
MAX_CROSS_SIDE_THICKNESS_RATIO = 1.25
MAX_THICKNESS_WIDTH_FRACTION = 0.25
MAX_MATE_DIRECTION_ERROR_DEG = 5.0
MIN_MATE_OVERLAP = 0.80
MAX_MATE_DISTANCE_THICKNESS_RATIO = 2.5
MAX_MATE_TURN_ERROR_DEG = 10.0
PROBE_RECORD_BYTES = 224
SCAN_HEADER_BYTES = 64


REJECT_INVALID_CANDIDATE = 1 << 0
REJECT_INVALID_OWNERS = 1 << 1
REJECT_INVALID_WIDTH = 1 << 2
REJECT_STATION_COVERAGE = 1 << 3
REJECT_NONUNIQUE_OPPOSITE = 1 << 4
REJECT_RECIPROCAL = 1 << 5
REJECT_THICKNESS = 1 << 6
REJECT_OWNER_DEGENERACY = 1 << 7
REJECT_NO_CONJUGATE = 1 << 8
REJECT_MATE_INCUMBENT = 1 << 9


def _canonical_unit(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected one finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ValueError("expected a nonzero vector")
    result = vector / length
    order = sorted(range(3), key=lambda index: (-abs(float(result[index])), index))
    if result[order[0]] < 0.0:
        result = -result
    result[result == 0.0] = 0.0
    return result


def _endpoints(row: Mapping[str, Any]) -> np.ndarray:
    value = row.get("span_endpoints")
    if value is None and row.get("line_start") is not None and row.get("line_end") is not None:
        value = (row["line_start"], row["line_end"])
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("candidate has invalid endpoints")
    return result


def _nearest_rank(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        return 0.0
    index = max(0, min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1))
    return float(ordered[index])


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if not count:
        return 0.0
    middle = count // 2
    return ordered[middle] if count % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.degrees(math.acos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))))


def _interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    denominator = min(left[1] - left[0], right[1] - right[0])
    return intersection / denominator if denominator > 0.0 else 0.0


def _line_distance(a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray) -> float:
    cross = np.cross(da, db)
    length = float(np.linalg.norm(cross))
    delta = b - a
    if length <= 1e-12:
        return float(np.linalg.norm(delta - float(np.dot(delta, da)) * da))
    return abs(float(np.dot(delta, cross / length)))


def _mesh_scale(vertices: np.ndarray, triangles: np.ndarray) -> float:
    edges = np.vstack((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    return _nearest_rank(lengths, 0.50)


@dataclass(frozen=True)
class TransitionParent:
    parent_id: int
    owner_a: int
    owner_b: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    direction: tuple[float, float, float]
    turn_angle_deg: float


@dataclass
class SideProbe:
    owner: int
    opposite_owner: int = -1
    mask: int = 0
    thickness: tuple[float, ...] = (0.0,) * 5
    width: float = 0.0
    reciprocal_ok: bool = False
    hit_faces: tuple[int, ...] = (-1,) * 5
    origin_faces: tuple[int, ...] = (-1,) * 5

    @property
    def valid_values(self) -> tuple[float, ...]:
        return tuple(value for index, value in enumerate(self.thickness) if self.mask & (1 << index))

    @property
    def median(self) -> float:
        return _median(self.valid_values)

    @property
    def ratio(self) -> float:
        values = self.valid_values
        return max(values) / min(values) if values and min(values) > 0.0 else math.inf


@dataclass
class ProbeRecord:
    candidate_id: str
    near_owners: tuple[int, int]
    sides: tuple[SideProbe, SideProbe]
    mate_parent_id: int = -1
    mate_direction_error_deg: float = 0.0
    mate_overlap: float = 0.0
    mate_line_distance: float = 0.0
    mate_turn_error_deg: float = 0.0
    candidate_parent_id: int = -1
    reject_mask: int = 0
    rank: tuple[Any, ...] = ()
    ribbon_key: tuple[int, ...] = ()

    @property
    def retain(self) -> bool:
        return self.reject_mask == 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "near_owner_ids": list(self.near_owners),
            "opposite_owner_ids": [side.opposite_owner for side in self.sides],
            "station_masks": [side.mask for side in self.sides],
            "thickness_values": [list(side.thickness) for side in self.sides],
            "side_widths": [side.width for side in self.sides],
            "reciprocal_flags": [side.reciprocal_ok for side in self.sides],
            "mate_parent_id": self.mate_parent_id,
            "mate_direction_error_deg": self.mate_direction_error_deg,
            "mate_overlap": self.mate_overlap,
            "mate_line_distance": self.mate_line_distance,
            "mate_turn_error_deg": self.mate_turn_error_deg,
            "ribbon_key": list(self.ribbon_key),
            "reject_mask": self.reject_mask,
            "retain": self.retain,
        }

    def encode(self) -> bytes:
        digest = int.from_bytes(hashlib.sha256(self.candidate_id.encode()).digest()[:8], "little")
        side_a, side_b = self.sides
        values = tuple(float(value) if math.isfinite(float(value)) else 0.0 for value in (*side_a.thickness, *side_b.thickness))
        ribbon = tuple(int(value) & 0xFFFFFFFF for value in self.ribbon_key[:6])
        ribbon += (0,) * (6 - len(ribbon))
        packed = struct.pack(
            "<Q5I2B10d2d2B4d6IQ",
            digest,
            int(self.near_owners[0]) & 0xFFFFFFFF,
            int(self.near_owners[1]) & 0xFFFFFFFF,
            int(side_a.opposite_owner) & 0xFFFFFFFF,
            int(side_b.opposite_owner) & 0xFFFFFFFF,
            int(self.mate_parent_id) & 0xFFFFFFFF,
            int(side_a.mask),
            int(side_b.mask),
            *values,
            float(side_a.width),
            float(side_b.width),
            int(side_a.reciprocal_ok),
            int(side_b.reciprocal_ok),
            float(self.mate_direction_error_deg),
            float(self.mate_overlap),
            float(self.mate_line_distance),
            float(self.mate_turn_error_deg),
            *ribbon,
            int(self.reject_mask),
        )
        if len(packed) > PROBE_RECORD_BYTES:
            raise AssertionError("PSRO record exceeded 224 bytes")
        return packed + bytes(PROBE_RECORD_BYTES - len(packed))


class NativeMeshProbeContext:
    """One transient mesh, one ray scene, and deterministic raw face owners."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        mesh.compute_triangle_normals()
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(mesh.triangles, dtype=np.int32)
        self.face_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
        self.face_centers = self.vertices[self.triangles].mean(axis=1)
        extractor = CADBendExtractor()
        self.extractor = extractor
        self.face_areas = extractor._compute_face_areas(self.vertices, self.triangles)
        self.total_area = float(np.sum(self.face_areas))
        self.edge_to_faces, self.face_adjacency = extractor._build_adjacency(self.triangles)
        flat_edges: set[tuple[int, int]] = set()
        for edge, faces in self.edge_to_faces.items():
            if len(faces) != 2:
                continue
            dot = np.clip(float(np.dot(self.face_normals[faces[0]], self.face_normals[faces[1]])), -1.0, 1.0)
            if math.degrees(math.acos(abs(dot))) < extractor.coplanar_threshold:
                flat_edges.add(edge)
        surfaces = extractor._region_growing(
            len(self.triangles), self.face_normals, self.face_adjacency, flat_edges,
            self.face_areas, self.face_centers, self.total_area,
        )
        minimum_area = self.total_area * extractor.min_area_pct / 100.0
        self.surfaces = {surface.surface_id: surface for surface in surfaces if surface.area >= minimum_area}
        self.face_owner = np.full(len(self.triangles), -1, dtype=np.int32)
        for owner, surface in self.surfaces.items():
            self.face_owner[surface.face_indices] = int(owner)
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)
        self.mesh_scale = _mesh_scale(self.vertices, self.triangles)
        self.one_ring = [set(self.face_adjacency.get(index, ())) | {index} for index in range(len(self.triangles))]
        # The bounded transition-parent identifier graph is the exact set of
        # shared mesh edges between raw plane owners.  Do not reuse the legacy
        # proximity/PCA bend extent here: its line can be displaced from the
        # actual owner boundary even on a perfect finite-thickness ribbon.
        edge_groups: dict[tuple[int, int], set[int]] = {}
        for edge, faces in self.edge_to_faces.items():
            if len(faces) != 2:
                continue
            owner_pair = tuple(sorted((int(self.face_owner[faces[0]]), int(self.face_owner[faces[1]]))))
            if owner_pair[0] < 0 or owner_pair[0] == owner_pair[1]:
                continue
            edge_groups.setdefault(owner_pair, set()).update(int(value) for value in edge)
        parents: list[TransitionParent] = []
        for parent_id, (owner_pair, vertex_ids) in enumerate(sorted(edge_groups.items()), start=1):
            points = self.vertices[sorted(vertex_ids)]
            center = np.mean(points, axis=0)
            covariance = (points - center).T @ (points - center)
            _values, vectors = np.linalg.eigh(covariance)
            direction = _canonical_unit(vectors[:, -1])
            projection = points @ direction
            start = center + (float(np.min(projection)) - float(np.dot(center, direction))) * direction
            end = center + (float(np.max(projection)) - float(np.dot(center, direction))) * direction
            normal_a = _canonical_unit(self.surfaces[owner_pair[0]].normal)
            normal_b = _canonical_unit(self.surfaces[owner_pair[1]].normal)
            parents.append(TransitionParent(
                parent_id=parent_id,
                owner_a=owner_pair[0],
                owner_b=owner_pair[1],
                start=tuple(float(value) for value in start),
                end=tuple(float(value) for value in end),
                direction=tuple(float(value) for value in direction),
                turn_angle_deg=_angle_deg(normal_a, normal_b),
            ))
        self.transition_parents = tuple(parents)

    def _intersections(self, origin: np.ndarray, direction: np.ndarray) -> list[tuple[float, int]]:
        ray = np.asarray([[*origin, *direction]], dtype=np.float32)
        result = self.scene.list_intersections(o3d.core.Tensor(ray))
        distances = np.asarray(result["t_hit"].numpy(), dtype=np.float64)
        faces = np.asarray(result["primitive_ids"].numpy(), dtype=np.uint32).astype(np.int64)
        ordered = sorted(zip(distances.tolist(), faces.tolist()), key=lambda item: (item[0], item[1]))
        return [(float(distance), int(face)) for distance, face in ordered if math.isfinite(distance) and distance > 0.0]

    def first_valid_hit(self, origin_face: int, direction: np.ndarray) -> tuple[float, int, int] | None:
        center = self.face_centers[origin_face]
        epsilon = max(0.5 * self.mesh_scale, 8.0 * math.ulp(max(1.0, float(np.linalg.norm(center)))))
        for distance, face in self._intersections(center + epsilon * direction, direction):
            if face in self.one_ring[origin_face]:
                continue
            owner = int(self.face_owner[face])
            if owner < 0:
                continue
            if float(np.dot(self.face_normals[origin_face], self.face_normals[face])) > MAX_NORMAL_OPPOSITION_DOT:
                continue
            # Thickness is measured from the selected origin-face centroid;
            # the epsilon offset is solely a self-hit guard for the ray query.
            return distance + epsilon, face, owner
        return None


def _side_probe(
    context: NativeMeshProbeContext,
    owner: int,
    endpoints: np.ndarray,
    direction: np.ndarray,
    interval: tuple[float, float],
) -> SideProbe:
    surface = context.surfaces.get(owner)
    if surface is None:
        return SideProbe(owner=owner)
    faces = np.asarray(surface.face_indices, dtype=np.int64)
    centers = context.face_centers[faces]
    projection = centers @ direction
    anchor = endpoints[0]
    relative = centers - anchor
    radial = relative - (relative @ direction)[:, None] * direction
    distances = np.linalg.norm(radial, axis=1)
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    width = _nearest_rank(positive, 0.90)
    span = interval[1] - interval[0]
    thickness = [0.0] * 5
    origin_faces = [-1] * 5
    hit_faces = [-1] * 5
    owners: list[int] = []
    mask = 0
    reciprocal_ok = False
    for station_index, q in enumerate(STATIONS):
        coordinate = interval[0] + q * span
        residual = np.abs(projection - coordinate)
        eligible = np.flatnonzero(residual <= span / 32.0)
        if not len(eligible):
            continue
        key = sorted(
            eligible.tolist(),
            key=lambda index: (float(residual[index]), -float(distances[index]), int(faces[index])),
        )[0]
        origin_face = int(faces[key])
        candidates = []
        normal = context.face_normals[origin_face]
        for direction_sign, ray_direction in enumerate((-normal, normal)):
            hit = context.first_valid_hit(origin_face, ray_direction)
            if hit is not None:
                candidates.append((direction_sign, ray_direction, hit))
        if len(candidates) != 1:
            continue
        _sign, inward, hit = candidates[0]
        hit_distance, hit_face, hit_owner = hit
        mask |= 1 << station_index
        thickness[station_index] = float(hit_distance)
        origin_faces[station_index] = origin_face
        hit_faces[station_index] = hit_face
        owners.append(hit_owner)
        if station_index == 2:
            returned = context.first_valid_hit(hit_face, -inward)
            reciprocal_ok = bool(
                returned is not None
                and returned[2] == owner
                and min(hit_distance, returned[0]) > 0.0
                and max(hit_distance, returned[0]) / min(hit_distance, returned[0]) <= MAX_RECIPROCAL_RATIO
            )
    opposite = owners[0] if owners and all(value == owners[0] for value in owners) else -1
    return SideProbe(
        owner=owner, opposite_owner=opposite, mask=mask,
        thickness=tuple(thickness), width=width, reciprocal_ok=reciprocal_ok,
        hit_faces=tuple(hit_faces), origin_faces=tuple(origin_faces),
    )


def probe_candidate(
    context: NativeMeshProbeContext,
    row: Mapping[str, Any],
    *,
    incumbent_parent_pairs: Iterable[tuple[int, int]] = (),
) -> ProbeRecord:
    candidate_id = str(row.get("candidate_id") or row.get("bend_id") or "")
    reject = 0
    try:
        endpoints = _endpoints(row)
        direction = _canonical_unit(endpoints[1] - endpoints[0])
    except ValueError:
        empty = SideProbe(-1)
        return ProbeRecord(candidate_id, (-1, -1), (empty, empty), reject_mask=REJECT_INVALID_CANDIDATE)
    projected = endpoints @ direction
    interval = (float(min(projected)), float(max(projected)))
    if interval[1] <= interval[0]:
        reject |= REJECT_INVALID_CANDIDATE
    try:
        owners = (int(row["surface1_id"]), int(row["surface2_id"]))
    except (KeyError, TypeError, ValueError):
        owners = (-1, -1)
    if owners[0] == owners[1] or any(owner not in context.surfaces for owner in owners):
        reject |= REJECT_INVALID_OWNERS
    sides = tuple(_side_probe(context, owner, endpoints, direction, interval) for owner in owners)
    for side in sides:
        if not math.isfinite(side.width) or side.width <= 0.0:
            reject |= REJECT_INVALID_WIDTH
        if side.mask & MANDATORY_MASK != MANDATORY_MASK or side.mask.bit_count() < MIN_STATIONS:
            reject |= REJECT_STATION_COVERAGE
        if side.opposite_owner < 0:
            reject |= REJECT_NONUNIQUE_OPPOSITE
        if not side.reciprocal_ok:
            reject |= REJECT_RECIPROCAL
        if (
            not side.valid_values
            or side.ratio > MAX_SIDE_THICKNESS_RATIO
            or side.median > MAX_THICKNESS_WIDTH_FRACTION * side.width
        ):
            reject |= REJECT_THICKNESS
    medians = [side.median for side in sides]
    if min(medians, default=0.0) <= 0.0 or max(medians) / min(medians) > MAX_CROSS_SIDE_THICKNESS_RATIO:
        reject |= REJECT_THICKNESS
    opposite = (sides[0].opposite_owner, sides[1].opposite_owner)
    if len(set((*owners, *opposite))) != 4 or opposite[0] == owners[1] or opposite[1] == owners[0]:
        reject |= REJECT_OWNER_DEGENERACY

    theta = _angle_deg(
        _canonical_unit(context.surfaces[owners[0]].normal) if owners[0] in context.surfaces else direction,
        _canonical_unit(context.surfaces[owners[1]].normal) if owners[1] in context.surfaces else direction,
    )
    span = interval[1] - interval[0]
    mandatory = tuple(interval[0] + q * span for q in (STATIONS[0], STATIONS[2], STATIONS[4]))
    denominator = max(medians, default=0.0)
    def parent_options(target_pair: tuple[int, int], *, distance_limit: float | None) -> list[tuple[Any, ...]]:
        options: list[tuple[Any, ...]] = []
        for parent in context.transition_parents:
            if (parent.owner_a, parent.owner_b) != target_pair:
                continue
            parent_direction = np.asarray(parent.direction, dtype=np.float64)
            parent_endpoints = np.asarray((parent.start, parent.end), dtype=np.float64)
            parent_projection = parent_endpoints @ direction
            parent_interval = (float(min(parent_projection)), float(max(parent_projection)))
            direction_error = _angle_deg(direction, parent_direction)
            overlap = _interval_overlap(interval, parent_interval)
            line_distance = _line_distance(
                np.mean(endpoints, axis=0), direction, np.mean(parent_endpoints, axis=0), parent_direction
            )
            turn_error = abs(theta - parent.turn_angle_deg)
            expanded = (
                parent_interval[0] - 2.0 * context.mesh_scale,
                parent_interval[1] + 2.0 * context.mesh_scale,
            )
            if (
                direction_error <= MAX_MATE_DIRECTION_ERROR_DEG
                and overlap >= MIN_MATE_OVERLAP
                and all(expanded[0] <= value <= expanded[1] for value in mandatory)
                and (distance_limit is None or line_distance <= distance_limit)
                and turn_error <= MAX_MATE_TURN_ERROR_DEG
            ):
                normalized_distance = line_distance / denominator if denominator > 0.0 else math.inf
                options.append(
                    (direction_error, -overlap, normalized_distance, turn_error, parent.parent_id, parent)
                )
        return options

    candidate_options = parent_options(tuple(sorted(owners)), distance_limit=2.0 * context.mesh_scale)
    if candidate_options:
        candidate_parent = min(candidate_options)[-1]
    else:
        candidate_parent = None
        reject |= REJECT_INVALID_CANDIDATE

    mate_options = []
    target_pair = tuple(sorted(opposite))
    for parent in context.transition_parents:
        if (parent.owner_a, parent.owner_b) != target_pair:
            continue
        mate_direction = np.asarray(parent.direction, dtype=np.float64)
        mate_endpoints = np.asarray((parent.start, parent.end), dtype=np.float64)
        mate_projection = mate_endpoints @ direction
        mate_interval = (float(min(mate_projection)), float(max(mate_projection)))
        direction_error = _angle_deg(direction, mate_direction)
        overlap = _interval_overlap(interval, mate_interval)
        line_distance = _line_distance(
            np.mean(endpoints, axis=0), direction, np.mean(mate_endpoints, axis=0), mate_direction
        )
        turn_error = abs(theta - parent.turn_angle_deg)
        expanded = (mate_interval[0] - 2.0 * context.mesh_scale, mate_interval[1] + 2.0 * context.mesh_scale)
        if (
            direction_error <= MAX_MATE_DIRECTION_ERROR_DEG
            and overlap >= MIN_MATE_OVERLAP
            and all(expanded[0] <= value <= expanded[1] for value in mandatory)
            and denominator > 0.0
            and line_distance <= MAX_MATE_DISTANCE_THICKNESS_RATIO * denominator
            and turn_error <= MAX_MATE_TURN_ERROR_DEG
        ):
            mate_options.append((direction_error, -overlap, line_distance / denominator, turn_error, parent.parent_id, parent))
    if not mate_options:
        reject |= REJECT_NO_CONJUGATE
        selected_parent = None
    else:
        selected_parent = min(mate_options)[-1]
    incumbent_pairs = {tuple(sorted(pair)) for pair in incumbent_parent_pairs}
    if selected_parent and (selected_parent.owner_a, selected_parent.owner_b) in incumbent_pairs:
        reject |= REJECT_MATE_INCUMBENT

    record = ProbeRecord(candidate_id, owners, sides, reject_mask=reject)
    if candidate_parent is not None:
        record.candidate_parent_id = int(candidate_parent.parent_id)
    if selected_parent:
        chosen = min(mate_options)
        record.mate_direction_error_deg = float(chosen[0])
        record.mate_overlap = float(-chosen[1])
        record.mate_line_distance = float(chosen[2] * denominator)
        record.mate_turn_error_deg = float(chosen[3])
        record.mate_parent_id = int(selected_parent.parent_id)
        pair_a = tuple(sorted((owners[0], opposite[0])))
        pair_b = tuple(sorted((owners[1], opposite[1])))
        parent_pair = tuple(sorted((record.candidate_parent_id, record.mate_parent_id)))
        record.ribbon_key = (*min(pair_a, pair_b), *max(pair_a, pair_b), *parent_pair)
        record.rank = (
            -min(sides[0].mask.bit_count(), sides[1].mask.bit_count()),
            -record.mate_overlap,
            max(sides[0].ratio, sides[1].ratio),
            max(medians) / min(medians) if min(medians) > 0.0 else math.inf,
            record.mate_direction_error_deg,
            record.mate_turn_error_deg,
            record.mate_line_distance / denominator if denominator > 0.0 else math.inf,
            -span,
            candidate_id,
        )
    record.encode()
    return record


def collapse_ribbon_duplicates(records: Sequence[ProbeRecord]) -> list[ProbeRecord]:
    retained: dict[tuple[int, ...], ProbeRecord] = {}
    for record in records:
        if not record.retain:
            continue
        prior = retained.get(record.ribbon_key)
        if prior is None or record.rank < prior.rank:
            retained[record.ribbon_key] = record
    return sorted(retained.values(), key=lambda record: record.rank)
