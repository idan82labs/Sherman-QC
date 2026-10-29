from __future__ import annotations

import numpy as np

from research.f1_deterministic_reference_surface_v1 import (
    array_sha256,
    sample_surface,
    sorted_reference_points,
    triangle_sample_counts,
)


def _square() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return vertices, triangles


def test_area_allocation_is_exact_and_tie_broken_by_triangle_index() -> None:
    vertices, triangles = _square()
    assert triangle_sample_counts(vertices, triangles, 5).tolist() == [3, 2]


def test_surface_sampling_replays_byte_exactly() -> None:
    vertices, triangles = _square()
    first_points, first_normals = sample_surface(vertices, triangles, 101)
    second_points, second_normals = sample_surface(vertices, triangles, 101)
    assert first_points.tobytes() == second_points.tobytes()
    assert first_normals.tobytes() == second_normals.tobytes()
    assert array_sha256(first_points) == array_sha256(second_points)
    assert len(first_points) == 101
    assert np.allclose(first_normals, [0.0, 0.0, 1.0])


def test_point_cloud_fallback_has_canonical_xyz_order() -> None:
    points = np.array([[1, 0, 0], [0, 2, 0], [0, 1, 3], [0, 1, 2]], dtype=float)
    ordered = sorted_reference_points(points)
    assert ordered.tolist() == [[0, 1, 2], [0, 1, 3], [0, 2, 0], [1, 0, 0]]

