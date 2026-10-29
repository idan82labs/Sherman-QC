from __future__ import annotations

import numpy as np

from backend.f1_calibrated_multiview_turn_consensus_v58 import (
    _angle_deg,
    _canonical_direction,
    _line_metrics,
    _nearest_rank,
    _representative,
)


def _line(start, end):
    return {"line_start": list(start), "line_end": list(end)}


def test_projective_angle_and_canonical_direction() -> None:
    assert _angle_deg(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])) == 0.0
    assert np.array_equal(_canonical_direction(np.array([-2.0, 0.0, 0.0])), np.array([2.0, 0.0, 0.0]))


def test_robust_bank_accepts_concentrated_normals_and_rejects_scatter() -> None:
    concentrated = [np.array([0.0, np.sin(np.deg2rad(value)), np.cos(np.deg2rad(value))]) for value in (-3, -2, 0, 2, 3, 4)]
    assert _representative(concentrated) is not None
    scattered = [np.array([0.0, np.sin(np.deg2rad(value)), np.cos(np.deg2rad(value))]) for value in (-35, -20, 0, 20, 35, 50)]
    assert _representative(scattered) is None


def test_line_metrics_are_projective_and_measure_overlap() -> None:
    angle, distance, overlap = _line_metrics(_line((0, 0, 0), (10, 0, 0)), _line((9, 1, 0), (-1, 1, 0)))
    assert angle == 0.0
    assert distance == 1.0
    assert overlap == 0.9


def test_nearest_rank_is_not_interpolated() -> None:
    assert _nearest_rank([1.0, 2.0, 3.0, 100.0], 0.75) == 3.0
