#!/usr/bin/env python3
"""Shared deterministic geometry primitives for EXP_CAD_0029.

This module contains no evaluator, role, label, gold, expected-count, or
quality-name interface.  Its CAD functions are used by the scan-free preflight.
The source runner may later import the frozen primitives after audit approval.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        value = int(value)
        root = value
        while root != int(self.parent[root]):
            root = int(self.parent[root])
        while value != root:
            next_value = int(self.parent[value])
            self.parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def canonical_tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must be Nx3")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(normals)) or np.any(lengths <= 1e-12):
        raise ValueError("invalid normals")
    unit = normals / lengths[:, None]
    axes = np.eye(3, dtype=np.float64)
    choices = np.argmin(np.abs(unit), axis=1)
    reference = axes[choices]
    first = np.cross(unit, reference)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(unit, first)
    second /= np.linalg.norm(second, axis=1)[:, None]
    return first, second


def radical_inverse_base2(index: int) -> float:
    value = int(index)
    result = 0.0
    fraction = 0.5
    while value:
        if value & 1:
            result += fraction
        fraction *= 0.5
        value >>= 1
    return result


def deterministic_equal_area_candidates(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    target_area_mm2: float,
) -> dict[str, np.ndarray]:
    """Return a deterministic face-owned Hammersley micro-atlas."""

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    triangle_normals = np.asarray(triangle_normals, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or triangle_normals.shape != (len(triangles), 3)
        or target_area_mm2 <= 0.0
    ):
        raise ValueError("invalid mesh candidate input")
    point_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    parent_rows: list[np.ndarray] = []
    ordinal_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    for parent, face in enumerate(triangles):
        triangle = vertices[face]
        area = 0.5 * float(
            np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
        if not np.isfinite(area) or area <= 1e-12:
            raise ValueError("mesh contains a degenerate triangle")
        count = max(1, int(math.ceil(area / target_area_mm2)))
        if count == 1:
            barycentric = np.asarray([[1.0 / 3.0] * 3], dtype=np.float64)
        else:
            indices = np.arange(count, dtype=np.int64)
            u = (indices.astype(np.float64) + 0.5) / float(count)
            v = np.asarray(
                [radical_inverse_base2(int(index) + 1) for index in indices],
                dtype=np.float64,
            )
            root = np.sqrt(u)
            barycentric = np.column_stack(
                (1.0 - root, root * (1.0 - v), root * v)
            )
        point_rows.append(barycentric @ triangle)
        normal_rows.append(np.repeat(triangle_normals[parent][None, :], count, axis=0))
        parent_rows.append(np.full(count, parent, dtype=np.int64))
        ordinal_rows.append(np.arange(count, dtype=np.int64))
        weight_rows.append(np.full(count, area / count, dtype=np.float64))
    return {
        "points": np.vstack(point_rows),
        "normals": np.vstack(normal_rows),
        "parent_faces": np.concatenate(parent_rows),
        "ordinals": np.concatenate(ordinal_rows),
        "area_weights_mm2": np.concatenate(weight_rows),
    }


def welded_mesh_topology(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    *,
    weld_tolerance_mm: float,
    smooth_angle_deg: float,
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    triangle_normals = np.asarray(triangle_normals, dtype=np.float64)
    if weld_tolerance_mm <= 0.0:
        raise ValueError("weld tolerance must be positive")
    weld_union = UnionFind(len(vertices))
    weld_pairs = cKDTree(vertices).query_pairs(
        float(weld_tolerance_mm), output_type="ndarray"
    )
    if len(weld_pairs):
        weld_pairs = np.asarray(weld_pairs, dtype=np.int64)
        weld_pairs = weld_pairs[np.lexsort((weld_pairs[:, 1], weld_pairs[:, 0]))]
        for first, second in weld_pairs:
            if (
                float(np.linalg.norm(vertices[int(first)] - vertices[int(second)]))
                <= float(weld_tolerance_mm)
            ):
                weld_union.union(int(first), int(second))
    weld_groups: dict[int, list[int]] = defaultdict(list)
    for vertex_id in range(len(vertices)):
        weld_groups[weld_union.find(vertex_id)].append(vertex_id)
    canonical_groups = []
    for members in weld_groups.values():
        representative = min(
            members,
            key=lambda index: (
                float(vertices[index, 0]),
                float(vertices[index, 1]),
                float(vertices[index, 2]),
                int(index),
            ),
        )
        canonical_groups.append((representative, sorted(members)))
    canonical_groups.sort(
        key=lambda row: (
            float(vertices[row[0], 0]),
            float(vertices[row[0], 1]),
            float(vertices[row[0], 2]),
            int(row[0]),
        )
    )
    inverse = np.empty(len(vertices), dtype=np.int64)
    welded_vertices = np.empty((len(canonical_groups), 3), dtype=np.float64)
    for welded_id, (representative, members) in enumerate(canonical_groups):
        welded_vertices[welded_id] = vertices[representative]
        inverse[np.asarray(members, dtype=np.int64)] = welded_id
    welded_triangles = inverse[triangles]
    rebuilt_cross = np.cross(
        welded_vertices[welded_triangles[:, 1]] - welded_vertices[welded_triangles[:, 0]],
        welded_vertices[welded_triangles[:, 2]] - welded_vertices[welded_triangles[:, 0]],
    )
    rebuilt_lengths = np.linalg.norm(rebuilt_cross, axis=1)
    if np.any(rebuilt_lengths <= 1e-12):
        raise ValueError("welding collapsed or degenerated a triangle")
    triangle_normals = rebuilt_cross / rebuilt_lengths[:, None]
    edge_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(welded_triangles):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(triangle[first]), int(triangle[second]))))
            if edge[0] == edge[1]:
                raise ValueError("welding collapsed a triangle edge")
            edge_owners[edge].append(face_index)
    face_union = UnionFind(len(triangles))
    smooth_union = UnionFind(len(triangles))
    cosine = math.cos(math.radians(float(smooth_angle_deg)))
    for owners in edge_owners.values():
        for owner_index, first in enumerate(owners):
            for second in owners[owner_index + 1 :]:
                face_union.union(first, second)
                if (
                    len(owners) == 2
                    and float(np.dot(triangle_normals[first], triangle_normals[second]))
                    >= cosine
                ):
                    smooth_union.union(first, second)

    def canonical_labels(union: UnionFind) -> np.ndarray:
        roots = np.asarray([union.find(index) for index in range(len(triangles))])
        members: dict[int, list[int]] = defaultdict(list)
        for index, root in enumerate(roots):
            members[int(root)].append(index)
        ordered = sorted(members.values(), key=lambda row: min(row))
        mapping = {
            face: label for label, row in enumerate(ordered) for face in row
        }
        return np.asarray([mapping[index] for index in range(len(triangles))], dtype=np.int64)

    mesh_component_labels = canonical_labels(face_union)
    smooth_region_labels = canonical_labels(smooth_union)
    ordered_edges = np.asarray(sorted(edge_owners), dtype=np.int64)
    boundary_by_region: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge in sorted(edge_owners):
        owners = edge_owners[edge]
        owner_regions = sorted(
            set(int(smooth_region_labels[owner]) for owner in owners)
        )
        for region in owner_regions:
            # An edge is a boundary of this exact welded five-degree region if
            # it does not have two manifold owners in that same region.
            same_region_count = sum(
                int(smooth_region_labels[owner]) == region for owner in owners
            )
            if len(owners) != 2 or same_region_count != 2:
                boundary_by_region[region].append(edge)
    return {
        "welded_vertex_count": int(len(canonical_groups)),
        "welded_vertices": welded_vertices,
        "original_to_welded": inverse.astype(np.int64),
        "welded_triangles": welded_triangles.astype(np.int64),
        "welded_triangle_normals": triangle_normals,
        "ordered_welded_edges": ordered_edges,
        "edge_owners": {edge: tuple(owners) for edge, owners in edge_owners.items()},
        "boundary_edges_by_smooth_region": {
            int(region): np.asarray(sorted(set(edges)), dtype=np.int64).reshape(-1, 2)
            for region, edges in boundary_by_region.items()
        },
        "edge_count": int(len(edge_owners)),
        "boundary_edge_count": int(sum(len(owners) == 1 for owners in edge_owners.values())),
        "nonmanifold_edge_count": int(sum(len(owners) > 2 for owners in edge_owners.values())),
        "mesh_component_labels": mesh_component_labels,
        "smooth_region_labels": smooth_region_labels,
    }


def _raycast_probe_pass(
    scene: o3d.t.geometry.RaycastingScene,
    probes: np.ndarray,
    expected_normals: np.ndarray,
    *,
    maximum_distance_mm: float,
    minimum_normal_dot: float,
    expected_smooth_regions: np.ndarray | None = None,
    smooth_region_by_face: np.ndarray | None = None,
    chunk_size: int = 250000,
) -> np.ndarray:
    probes = np.asarray(probes, dtype=np.float64)
    expected_normals = np.asarray(expected_normals, dtype=np.float64)
    result = np.zeros(len(probes), dtype=bool)
    for start in range(0, len(probes), chunk_size):
        stop = min(len(probes), start + chunk_size)
        answer = scene.compute_closest_points(
            o3d.core.Tensor(probes[start:stop].astype(np.float32))
        )
        closest = answer["points"].numpy().astype(np.float64)
        normals = answer["primitive_normals"].numpy().astype(np.float64)
        distance = np.linalg.norm(probes[start:stop] - closest, axis=1)
        dot = np.sum(normals * expected_normals[start:stop], axis=1)
        result[start:stop] = (distance <= maximum_distance_mm) & (
            dot >= minimum_normal_dot
        )
        if expected_smooth_regions is not None:
            primitive_ids = answer["primitive_ids"].numpy().astype(np.int64)
            if smooth_region_by_face is None:
                raise ValueError("smooth-region lookup is required")
            valid_primitive = primitive_ids < len(smooth_region_by_face)
            observed_regions = np.full(len(primitive_ids), -1, dtype=np.int64)
            observed_regions[valid_primitive] = np.asarray(
                smooth_region_by_face, dtype=np.int64
            )[primitive_ids[valid_primitive]]
            result[start:stop] &= observed_regions == np.asarray(
                expected_smooth_regions[start:stop], dtype=np.int64
            )
    return result


def cad_interior_mask(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    radii_mm: Sequence[float],
    azimuth_count: int,
    maximum_distance_mm: float,
    maximum_normal_angle_deg: float,
    parent_faces: np.ndarray | None = None,
    smooth_region_by_face: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    first, second = canonical_tangent_basis(normals)
    probes = []
    expected = []
    for radius in radii_mm:
        for ordinal in range(int(azimuth_count)):
            angle = 2.0 * math.pi * ordinal / float(azimuth_count)
            direction = math.cos(angle) * first + math.sin(angle) * second
            probes.append(points + float(radius) * direction)
            expected.append(normals)
    probe_array = np.stack(probes, axis=1)
    expected_array = np.stack(expected, axis=1)
    flat_pass = _raycast_probe_pass(
        scene,
        probe_array.reshape(-1, 3),
        expected_array.reshape(-1, 3),
        maximum_distance_mm=maximum_distance_mm,
        minimum_normal_dot=math.cos(math.radians(maximum_normal_angle_deg)),
        expected_smooth_regions=(
            None
            if parent_faces is None or smooth_region_by_face is None
            else np.repeat(
                np.asarray(smooth_region_by_face, dtype=np.int64)[
                    np.asarray(parent_faces, dtype=np.int64)
                ],
                probe_array.shape[1],
            )
        ),
        smooth_region_by_face=smooth_region_by_face,
    )
    return np.all(flat_pass.reshape(len(points), -1), axis=1)


def smooth_region_boundary_clearance_mm(
    points: np.ndarray,
    parent_faces: np.ndarray,
    smooth_region_by_face: np.ndarray,
    welded_vertices: np.ndarray,
    boundary_edges_by_smooth_region: Mapping[int, np.ndarray],
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Conservative lower bound on intrinsic distance to every region edge.

    Ambient point-to-segment distance never exceeds mesh-geodesic distance.
    Requiring this exact lower bound to be at least the clearance therefore
    proves (rather than estimates) the requested mesh-geodesic clearance while
    also rejecting nearby slits, holes, and folded-back boundaries.
    """

    points = np.asarray(points, dtype=np.float64)
    parents = np.asarray(parent_faces, dtype=np.int64)
    face_regions = np.asarray(smooth_region_by_face, dtype=np.int64)
    vertices = np.asarray(welded_vertices, dtype=np.float64)
    output = np.full(len(points), math.inf, dtype=np.float64)
    point_regions = face_regions[parents]
    for region in sorted(set(int(value) for value in point_regions)):
        point_ids = np.flatnonzero(point_regions == region)
        edges = np.asarray(
            boundary_edges_by_smooth_region.get(region, np.empty((0, 2))),
            dtype=np.int64,
        ).reshape(-1, 2)
        if not len(edges):
            continue
        starts = vertices[edges[:, 0]]
        vectors = vertices[edges[:, 1]] - starts
        lengths_squared = np.sum(vectors * vectors, axis=1)
        if np.any(lengths_squared <= 1e-24):
            raise ValueError("collapsed welded boundary edge")
        for begin in range(0, len(point_ids), int(chunk_size)):
            ids = point_ids[begin : begin + int(chunk_size)]
            delta = points[ids, None, :] - starts[None, :, :]
            fraction = np.sum(delta * vectors[None, :, :], axis=2) / lengths_squared[None, :]
            fraction = np.clip(fraction, 0.0, 1.0)
            closest_delta = delta - fraction[:, :, None] * vectors[None, :, :]
            output[ids] = np.sqrt(np.min(np.sum(closest_delta * closest_delta, axis=2), axis=1))
    return output


def canonical_candidate_order(candidates: Mapping[str, np.ndarray]) -> np.ndarray:
    points = np.round(np.asarray(candidates["points"], dtype=np.float64), 12)
    normals = np.round(np.asarray(candidates["normals"], dtype=np.float64), 12)
    parents = np.asarray(candidates["parent_faces"], dtype=np.int64)
    ordinals = np.asarray(candidates["ordinals"], dtype=np.int64)
    keys = (
        ordinals,
        parents,
        normals[:, 2],
        normals[:, 1],
        normals[:, 0],
        points[:, 2],
        points[:, 1],
        points[:, 0],
    )
    return np.lexsort(keys)


def deterministic_poisson_nodes(
    candidates: Mapping[str, np.ndarray],
    candidate_mask: np.ndarray,
    smooth_region_by_face: np.ndarray,
    *,
    minimum_spacing_mm: float,
) -> np.ndarray:
    points = np.asarray(candidates["points"], dtype=np.float64)
    parents = np.asarray(candidates["parent_faces"], dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    regions = np.asarray(smooth_region_by_face, dtype=np.int64)[parents]
    order = canonical_candidate_order(candidates)
    accepted: list[int] = []
    buckets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    inverse = 1.0 / float(minimum_spacing_mm)
    spacing_squared = float(minimum_spacing_mm) ** 2
    for candidate_index in order:
        candidate_index = int(candidate_index)
        if not candidate_mask[candidate_index]:
            continue
        point = points[candidate_index]
        cell = np.floor(point * inverse).astype(np.int64)
        region = int(regions[candidate_index])
        blocked = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for prior_index in buckets.get(
                        (region, int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)),
                        (),
                    ):
                        if float(np.sum((point - points[prior_index]) ** 2)) < spacing_squared:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue
        accepted.append(candidate_index)
        buckets[(region, int(cell[0]), int(cell[1]), int(cell[2]))].append(candidate_index)
    return np.asarray(accepted, dtype=np.int64)


def topology_qualified_graph_edges(
    node_points: np.ndarray,
    node_parent_faces: np.ndarray,
    node_smooth_regions: np.ndarray,
    topology: Mapping[str, Any],
    *,
    maximum_length_mm: float,
    maximum_geodesic_to_euclidean_ratio: float,
    atlas_edge_sample_spacing_mm: float,
    dijkstra_batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edges using a deterministic, region-confined surface atlas.

    The finite atlas contains every accepted node and welded-edge samples no
    farther than ``atlas_edge_sample_spacing_mm`` apart.  Within each triangle,
    atlas points separated by at most the search limit receive their exact
    in-triangle chord.  Dijkstra distance is therefore a valid surface path
    upper bound: passing its length and ratio gates proves the corresponding
    shortest mesh-geodesic gates.  No raycast probe participates in selection.
    """

    points = np.asarray(node_points, dtype=np.float64)
    parents = np.asarray(node_parent_faces, dtype=np.int64)
    regions = np.asarray(node_smooth_regions, dtype=np.int64)
    welded_vertices = np.asarray(topology["welded_vertices"], dtype=np.float64)
    welded_triangles = np.asarray(topology["welded_triangles"], dtype=np.int64)
    face_regions = np.asarray(topology["smooth_region_labels"], dtype=np.int64)
    if atlas_edge_sample_spacing_mm <= 0.0:
        raise ValueError("atlas sample spacing must be positive")
    euclidean_pairs = cKDTree(points).query_pairs(
        float(maximum_length_mm), output_type="ndarray"
    )
    if not len(euclidean_pairs):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    same_region = regions[euclidean_pairs[:, 0]] == regions[euclidean_pairs[:, 1]]
    euclidean_pairs = np.asarray(euclidean_pairs[same_region], dtype=np.int64)
    if not len(euclidean_pairs):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    accepted_pairs: list[tuple[int, int]] = []
    accepted_distances: list[float] = []
    search_limit = float(maximum_length_mm)
    chord_limit = search_limit * float(maximum_geodesic_to_euclidean_ratio)

    for region in sorted(set(int(value) for value in regions)):
        region_node_ids = np.flatnonzero(regions == region)
        region_pair_mask = regions[euclidean_pairs[:, 0]] == region
        region_pairs = euclidean_pairs[region_pair_mask]
        if not len(region_pairs):
            continue
        region_faces = np.flatnonzero(face_regions == region)
        face_keys: dict[int, set[tuple[Any, ...]]] = {
            int(face): set() for face in region_faces
        }
        coordinates: dict[tuple[Any, ...], np.ndarray] = {}
        for face in region_faces:
            face = int(face)
            triangle = welded_triangles[face]
            for first, second in ((0, 1), (1, 2), (2, 0)):
                a, b = sorted((int(triangle[first]), int(triangle[second])))
                length = float(np.linalg.norm(welded_vertices[b] - welded_vertices[a]))
                divisions = max(1, int(math.ceil(length / atlas_edge_sample_spacing_mm)))
                for ordinal in range(divisions + 1):
                    if ordinal == 0:
                        key: tuple[Any, ...] = ("v", a)
                    elif ordinal == divisions:
                        key = ("v", b)
                    else:
                        key = ("e", a, b, ordinal, divisions)
                    if key not in coordinates:
                        fraction = float(ordinal) / float(divisions)
                        coordinates[key] = (
                            (1.0 - fraction) * welded_vertices[a]
                            + fraction * welded_vertices[b]
                        )
                    face_keys[face].add(key)
        nodes_by_face: dict[int, list[int]] = defaultdict(list)
        for node_id in region_node_ids:
            nodes_by_face[int(parents[node_id])].append(int(node_id))
            key = ("n", int(node_id))
            coordinates[key] = points[node_id]
            face_keys[int(parents[node_id])].add(key)

        ordered_keys = sorted(coordinates)
        key_to_id = {key: index for index, key in enumerate(ordered_keys)}
        atlas_points = np.asarray([coordinates[key] for key in ordered_keys], dtype=np.float64)
        edge_weights: dict[tuple[int, int], float] = {}
        for face in sorted(face_keys):
            ids = np.asarray(
                sorted(key_to_id[key] for key in face_keys[face]), dtype=np.int64
            )
            if len(ids) < 2:
                continue
            local_pairs = cKDTree(atlas_points[ids]).query_pairs(
                chord_limit, output_type="ndarray"
            )
            for first_local, second_local in local_pairs:
                first_id, second_id = sorted(
                    (int(ids[int(first_local)]), int(ids[int(second_local)]))
                )
                length = float(
                    np.linalg.norm(atlas_points[first_id] - atlas_points[second_id])
                )
                if length <= 1e-15:
                    continue
                prior = edge_weights.get((first_id, second_id))
                if prior is None or length < prior:
                    edge_weights[(first_id, second_id)] = length
        if not edge_weights:
            continue
        edge_rows = np.asarray(sorted(edge_weights), dtype=np.int64)
        weights = np.asarray([edge_weights[tuple(row)] for row in edge_rows], dtype=np.float64)
        row = np.r_[edge_rows[:, 0], edge_rows[:, 1]]
        column = np.r_[edge_rows[:, 1], edge_rows[:, 0]]
        values = np.r_[weights, weights]
        graph = sparse.csr_matrix(
            (values, (row, column)), shape=(len(ordered_keys), len(ordered_keys))
        )
        node_atlas_ids = {
            int(node_id): key_to_id[("n", int(node_id))]
            for node_id in region_node_ids
        }
        pairs_by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pair_index, (first, second) in enumerate(region_pairs):
            pairs_by_source[int(first)].append((int(pair_index), int(second)))
        source_ids = sorted(pairs_by_source)
        region_results = np.full(len(region_pairs), math.inf, dtype=np.float64)
        for begin in range(0, len(source_ids), int(dijkstra_batch_size)):
            batch = source_ids[begin : begin + int(dijkstra_batch_size)]
            distances = dijkstra(
                graph,
                directed=False,
                indices=np.asarray([node_atlas_ids[node] for node in batch]),
                limit=chord_limit,
            )
            if distances.ndim == 1:
                distances = distances[None, :]
            for batch_row, source in enumerate(batch):
                for pair_index, target in pairs_by_source[source]:
                    region_results[pair_index] = float(
                        distances[batch_row, node_atlas_ids[target]]
                    )
        euclidean = np.linalg.norm(
            points[region_pairs[:, 0]] - points[region_pairs[:, 1]], axis=1
        )
        passes = (
            np.isfinite(region_results)
            & (region_results <= search_limit)
            & (
                region_results
                <= float(maximum_geodesic_to_euclidean_ratio) * euclidean
            )
        )
        for pair, distance in zip(
            region_pairs[passes], region_results[passes], strict=True
        ):
            accepted_pairs.append((int(pair[0]), int(pair[1])))
            accepted_distances.append(float(distance))
    if not accepted_pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    order = np.lexsort(
        (
            np.asarray([row[1] for row in accepted_pairs]),
            np.asarray([row[0] for row in accepted_pairs]),
        )
    )
    return (
        np.asarray(accepted_pairs, dtype=np.int64)[order],
        np.asarray(accepted_distances, dtype=np.float64)[order],
    )


def assign_node_area_weights(
    candidates: Mapping[str, np.ndarray],
    interior_mask: np.ndarray,
    node_candidate_indices: np.ndarray,
    smooth_region_by_face: np.ndarray,
    *,
    maximum_distance_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(candidates["points"], dtype=np.float64)
    parents = np.asarray(candidates["parent_faces"], dtype=np.int64)
    weights = np.asarray(candidates["area_weights_mm2"], dtype=np.float64)
    regions = np.asarray(smooth_region_by_face, dtype=np.int64)[parents]
    node_candidate_indices = np.asarray(node_candidate_indices, dtype=np.int64)
    node_regions = regions[node_candidate_indices]
    output = np.zeros(len(node_candidate_indices), dtype=np.float64)
    assigned_candidate_mask = np.zeros(len(points), dtype=bool)
    for region in sorted(set(int(value) for value in node_regions)):
        node_local = np.flatnonzero(node_regions == region)
        candidate_local = np.flatnonzero(interior_mask & (regions == region))
        if not len(candidate_local) or not len(node_local):
            continue
        tree = cKDTree(points[node_candidate_indices[node_local]])
        ownership: dict[int, list[float]] = defaultdict(list)
        for candidate_index in candidate_local:
            nearby_local = tree.query_ball_point(
                points[int(candidate_index)], r=float(maximum_distance_mm)
            )
            if not nearby_local:
                continue
            canonical_node_ids = sorted(int(node_local[int(index)]) for index in nearby_local)
            squared = np.asarray(
                [
                    float(
                        np.sum(
                            (
                                points[int(candidate_index)]
                                - points[int(node_candidate_indices[node_id])]
                            )
                            ** 2
                        )
                    )
                    for node_id in canonical_node_ids
                ],
                dtype=np.float64,
            )
            within = squared <= float(maximum_distance_mm) ** 2
            canonical_node_ids = [
                node_id
                for node_id, keep in zip(canonical_node_ids, within, strict=True)
                if bool(keep)
            ]
            squared = squared[within]
            if not len(canonical_node_ids):
                continue
            minimum = float(np.min(squared))
            # Exact float64 distance ties are owned by the lowest canonical ID.
            tied = [
                node_id
                for node_id, value in zip(canonical_node_ids, squared, strict=True)
                if float(value) == minimum
            ]
            owner = int(min(tied))
            ownership[owner].append(float(weights[int(candidate_index)]))
            assigned_candidate_mask[int(candidate_index)] = True
        for owner in sorted(ownership):
            output[owner] = math.fsum(ownership[owner])
    return output, assigned_candidate_mask


def graph_component_labels(node_count: int, edges: np.ndarray) -> np.ndarray:
    union = UnionFind(int(node_count))
    for first, second in np.asarray(edges, dtype=np.int64):
        union.union(int(first), int(second))
    roots = np.asarray([union.find(index) for index in range(node_count)], dtype=np.int64)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        groups[int(root)].append(index)
    ordered = sorted(groups.values(), key=lambda row: min(row))
    mapping = {node: label for label, row in enumerate(ordered) for node in row}
    return np.asarray([mapping[index] for index in range(node_count)], dtype=np.int64)


def canonical_node_record_bytes(
    point: np.ndarray, normal: np.ndarray, parent_face: int, ordinal: int
) -> bytes:
    values = [float(value) for value in np.round(np.r_[point, normal], 12)]
    return struct.pack("<6dqq", *values, int(parent_face), int(ordinal))


def spatial_pca_fold_ids(
    points: np.ndarray,
    area_weights: np.ndarray,
    *,
    fold_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Contiguous area-quantile spatial bands on a canonical weighted PCA axis."""

    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    total = math.fsum(float(value) for value in weights)
    centroid = np.asarray(
        [
            math.fsum(float(weights[i] * points[i, axis]) for i in range(len(points)))
            / total
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    relative = points - centroid
    covariance = np.zeros((3, 3), dtype=np.float64)
    for first in range(3):
        for second in range(3):
            covariance[first, second] = math.fsum(
                float(weights[i] * relative[i, first] * relative[i, second])
                for i in range(len(points))
            ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis = eigenvectors[:, order[0]]
    sign_index = int(np.argmax(np.abs(axis)))
    if axis[sign_index] < 0.0:
        axis = -axis
    projection = relative @ axis
    node_ids = np.arange(len(points), dtype=np.int64)
    spatial_order = np.lexsort((node_ids, projection))
    folds = np.empty(len(points), dtype=np.int8)
    cumulative = 0.0
    for node_id in spatial_order:
        midpoint_fraction = (
            cumulative + 0.5 * float(weights[int(node_id)])
        ) / total
        folds[int(node_id)] = min(
            int(fold_count) - 1,
            int(math.floor(int(fold_count) * midpoint_fraction)),
        )
        cumulative += float(weights[int(node_id)])
    eigengap_ratio = (
        float(eigenvalues[0] / eigenvalues[1])
        if eigenvalues[1] > 0.0
        else math.inf
    )
    return folds, {
        "area_centroid_mm": centroid.tolist(),
        "pca_axis": axis.tolist(),
        "eigenvalues_mm2": eigenvalues.tolist(),
        "leading_to_second_eigenvalue_ratio": eigengap_ratio,
    }


def weighted_node_graph(
    node_count: int, edges: np.ndarray, edge_weights_mm: np.ndarray
) -> sparse.csr_matrix:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(edge_weights_mm, dtype=np.float64)
    if len(edges) != len(weights) or np.any(weights <= 0.0):
        raise ValueError("invalid weighted node graph")
    if not len(edges):
        return sparse.csr_matrix((int(node_count), int(node_count)), dtype=np.float64)
    return sparse.csr_matrix(
        (
            np.r_[weights, weights],
            (
                np.r_[edges[:, 0], edges[:, 1]],
                np.r_[edges[:, 1], edges[:, 0]],
            ),
        ),
        shape=(int(node_count), int(node_count)),
    )


def fold_training_masks_with_halo(
    points: np.ndarray,
    smooth_regions: np.ndarray,
    fold_ids: np.ndarray,
    graph: sparse.csr_matrix,
    *,
    fold_count: int,
    halo_mm: float,
) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    regions = np.asarray(smooth_regions, dtype=np.int64)
    folds = np.asarray(fold_ids, dtype=np.int8)
    masks = []
    for held_fold in range(int(fold_count)):
        held = np.flatnonzero(folds == held_fold)
        training = folds != held_fold
        if not len(held):
            masks.append(np.zeros(len(points), dtype=bool))
            continue
        distances = dijkstra(
            graph, directed=False, indices=held, min_only=True, limit=float(halo_mm)
        )
        graph_clear = (~np.isfinite(distances)) | (distances > float(halo_mm))
        # Ambient clearance is a conservative certificate that the true surface
        # geodesic halo also exceeds six millimetres.  Only same-region held
        # nodes participate; disconnected regions cannot leak through space.
        ambient_clear = np.ones(len(points), dtype=bool)
        for region in sorted(set(int(value) for value in regions)):
            held_region = held[regions[held] == region]
            query_region = np.flatnonzero(regions == region)
            if len(held_region):
                nearest, _ = cKDTree(points[held_region]).query(
                    points[query_region], k=1, workers=1
                )
                ambient_clear[query_region] = nearest > float(halo_mm)
        masks.append(training & graph_clear & ambient_clear)
    return masks


def component_two_sweep_geodesic_spans(
    component_labels: np.ndarray,
    graph: sparse.csr_matrix,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray(component_labels, dtype=np.int64)
    spans = np.zeros(int(np.max(labels)) + 1 if len(labels) else 0, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for component in range(len(spans)):
        members = np.flatnonzero(labels == component)
        anchor = int(np.min(members))
        first_dist = dijkstra(graph, directed=False, indices=anchor)
        finite_members = members[np.isfinite(first_dist[members])]
        maximum = float(np.max(first_dist[finite_members]))
        first_endpoint = int(np.min(finite_members[first_dist[finite_members] == maximum]))
        second_dist = dijkstra(graph, directed=False, indices=first_endpoint)
        maximum = float(np.max(second_dist[finite_members]))
        second_endpoint = int(np.min(finite_members[second_dist[finite_members] == maximum]))
        spans[component] = maximum
        records.append(
            {
                "component_id": component,
                "anchor_node_id": anchor,
                "first_endpoint_node_id": first_endpoint,
                "second_endpoint_node_id": second_endpoint,
                "two_sweep_span_mm": maximum,
            }
        )
    return spans, records


def rigid_design_matrix(
    points: np.ndarray,
    normals: np.ndarray,
    area_weights: np.ndarray,
    *,
    fixed_area_centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    if len(points) != len(normals) or len(points) != len(weights) or np.sum(weights) <= 0:
        raise ValueError("invalid rigid design input")
    if fixed_area_centroid is None:
        centroid = np.asarray(
            [
                math.fsum(float(weights[i] * points[i, axis]) for i in range(len(points)))
                / math.fsum(float(value) for value in weights)
                for axis in range(3)
            ],
            dtype=np.float64,
        )
    else:
        centroid = np.asarray(fixed_area_centroid, dtype=np.float64)
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError("invalid frozen area centroid")
    relative = points - centroid
    first, second = canonical_tangent_basis(normals)
    rows = np.zeros((3 * len(points), 6), dtype=np.float64)
    rows[0::3, :3] = normals
    rows[0::3, 3:] = np.cross(relative, normals)
    rows[1::3, 3:] = 3.0 * np.cross(first, normals)
    rows[2::3, 3:] = 3.0 * np.cross(second, normals)
    row_weights = np.repeat(np.sqrt(weights), 3)
    weighted = rows * row_weights[:, None]
    column_norms = np.linalg.norm(weighted, axis=0)
    if np.any(column_norms <= 1e-12):
        return rows, centroid, np.asarray([], dtype=np.float64)
    singular = np.linalg.svd(weighted / column_norms[None, :], compute_uv=False)
    return rows, centroid, singular


def design_rank_condition(singular_values: np.ndarray) -> tuple[int, float]:
    singular = np.asarray(singular_values, dtype=np.float64)
    if len(singular) != 6 or singular[0] <= 0.0:
        return 0, math.inf
    tolerance = singular[0] * 1e-10
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == 6 else math.inf
    return rank, condition


def permutation_invariant_voxel_aggregate(
    points: np.ndarray, normals: np.ndarray, voxel_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate without acquisition-order dependence.

    Rows are canonically sorted by voxel, coordinates, and normals.  Every sum
    uses :func:`math.fsum`, so an arbitrary input permutation produces the same
    output bytes.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(normals))
        or voxel_mm <= 0.0
    ):
        raise ValueError("invalid voxel aggregation input")
    keys = np.floor(points / float(voxel_mm)).astype(np.int64)
    order = np.lexsort(
        (
            normals[:, 2],
            normals[:, 1],
            normals[:, 0],
            points[:, 2],
            points[:, 1],
            points[:, 0],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )
    keys, points, normals = keys[order], points[order], normals[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)), len(keys)]
    output_points = []
    output_normals = []
    output_keys = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        count = int(stop - start)
        point = np.asarray(
            [math.fsum(float(value) for value in points[start:stop, axis]) / count for axis in range(3)],
            dtype=np.float64,
        )
        normal = np.asarray(
            [math.fsum(float(value) for value in normals[start:stop, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            continue
        output_points.append(point)
        output_normals.append(normal / length)
        output_keys.append(keys[start].copy())
    return (
        np.asarray(output_points, dtype=np.float64),
        np.asarray(output_normals, dtype=np.float64),
        np.asarray(output_keys, dtype=np.int64),
    )
