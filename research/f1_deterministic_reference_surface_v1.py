"""Deterministic area-stratified sampling of a triangulated CAD reference surface."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


_IRRATIONAL_FRACTION = 0.6180339887498949


def _validate_mesh(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise ValueError("vertices must be nonempty Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("triangles must be nonempty Mx3")
    if np.min(faces) < 0 or np.max(faces) >= len(verts):
        raise ValueError("triangle index outside vertex array")
    return verts, faces


def triangle_sample_counts(vertices: np.ndarray, triangles: np.ndarray, target_count: int) -> np.ndarray:
    verts, faces = _validate_mesh(vertices, triangles)
    if target_count <= 0:
        raise ValueError("target count must be positive")
    tri = verts[faces]
    twice_area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(twice_area_vectors, axis=1)
    positive = areas > 1e-15
    if not np.any(positive):
        raise ValueError("mesh has no positive-area triangle")
    weights = areas / np.sum(areas)
    ideal = weights * int(target_count)
    counts = np.floor(ideal).astype(np.int64)
    remaining = int(target_count - np.sum(counts))
    if remaining:
        fractional = ideal - counts
        order = np.lexsort((np.arange(len(faces), dtype=np.int64), -fractional))
        counts[order[:remaining]] += 1
    if int(np.sum(counts)) != int(target_count):
        raise AssertionError("deterministic triangle allocation failed")
    return counts


def sample_surface(vertices: np.ndarray, triangles: np.ndarray, target_count: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = _validate_mesh(vertices, triangles)
    counts = triangle_sample_counts(verts, faces, target_count)
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 1e-15
    normals[valid] /= normal_lengths[valid, None]
    points_out: list[np.ndarray] = []
    normals_out: list[np.ndarray] = []
    for index, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        if count <= 0:
            continue
        sequence = np.arange(count, dtype=np.float64) + 0.5
        u = sequence / count
        v = np.mod(sequence * _IRRATIONAL_FRACTION, 1.0)
        root = np.sqrt(u)
        barycentric = np.column_stack((1.0 - root, root * (1.0 - v), root * v))
        points_out.append(barycentric @ tri[index])
        normals_out.append(np.repeat(normals[index][None, :], count, axis=0))
    points = np.ascontiguousarray(np.vstack(points_out), dtype=np.float64)
    point_normals = np.ascontiguousarray(np.vstack(normals_out), dtype=np.float64)
    if len(points) != target_count or len(point_normals) != target_count:
        raise AssertionError("deterministic sampler returned wrong point count")
    return points, point_normals


def sorted_reference_points(points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError("fallback reference points must be nonempty Nx3")
    order = np.lexsort((source[:, 2], source[:, 1], source[:, 0]))
    return np.ascontiguousarray(source[order], dtype=np.float64)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def contract_payload() -> dict[str, Any]:
    return {
        "schema": "DETERMINISTIC_AREA_STRATIFIED_REFERENCE_SURFACE_V1",
        "triangle_allocation": "AREA_FLOOR_LARGEST_REMAINDER_TRIANGLE_INDEX_TIEBREAK",
        "within_triangle_sequence": "FIXED_SQRT_BARYCENTRIC_IRRATIONAL_SECOND_COORDINATE",
        "random_numbers_used": False,
    }

