from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scripts.apply_f1_step_topology_targeted_verifier import _two_surface_normal_turn


def _grid_two_faces() -> np.ndarray:
    x = np.linspace(0.0, 80.0, 41)
    width = np.linspace(1.0, 9.0, 9)
    face_a = np.array([[xx, yy, 0.0] for xx in x for yy in width])
    face_b = np.array([[xx, 0.0, zz] for xx in x for zz in width])
    return np.vstack([face_a, face_b])


def test_two_perpendicular_faces_pass_normal_turn_verification():
    points = _grid_two_faces()
    result = _two_surface_normal_turn(
        points, cKDTree(points), [0.0, 0.0, 0.0], [80.0, 0.0, 0.0]
    )
    assert result["passed"] is True
    assert result["normal_turn_deg"] >= 70.0


def test_single_flat_face_fails_normal_turn_verification():
    x = np.linspace(0.0, 80.0, 41)
    width = np.linspace(-9.0, 9.0, 19)
    points = np.array([[xx, yy, 0.0] for xx in x for yy in width])
    result = _two_surface_normal_turn(
        points, cKDTree(points), [0.0, 0.0, 0.0], [80.0, 0.0, 0.0]
    )
    assert result["passed"] is False
