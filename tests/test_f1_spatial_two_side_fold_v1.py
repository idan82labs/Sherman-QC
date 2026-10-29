from __future__ import annotations

import numpy as np

from research.f1_spatial_two_side_fold_v1 import spatial_fold_evidence


def _grid(values):
    return np.asarray(values, dtype=np.float64)


def test_flat_plane_crossing_fails_spatial_turn():
    points = _grid([[x, y, 0.0] for x in np.linspace(0, 100, 51) for y in np.linspace(-10, 10, 21)])
    result = spatial_fold_evidence(points, [0, 0, 0], [100, 0, 0])
    assert result["state"] == "EVALUATED_FAIL"
    assert result["passed"] is False


def test_two_planar_flanks_pass_spatial_turn():
    angle = np.deg2rad(25.0)
    points = []
    for x in np.linspace(0, 100, 51):
        for y in np.linspace(-10, -2, 10):
            points.append([x, y, np.tan(angle) * -y])
        for y in np.linspace(2, 10, 10):
            points.append([x, y, np.tan(angle) * y])
    result = spatial_fold_evidence(_grid(points), [0, 0, 0], [100, 0, 0])
    assert result["state"] == "EVALUATED_PASS"
    assert result["turn_deg"] >= 40.0


def test_missing_evidence_is_unresolved_not_rejection():
    points = _grid([[x, 2, 0] for x in np.linspace(0, 10, 20)])
    result = spatial_fold_evidence(points, [0, 0, 0], [10, 0, 0])
    assert result["state"] == "UNRESOLVED"
    assert result["passed"] is None
