import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0018_native_carrier_support.py"
SPEC = importlib.util.spec_from_file_location("exp18", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_full_finite_carrier_support():
    event = {"event_id": "E1", "line_start_mm": [0, 0, 0], "line_end_mm": [0, 0, 10], "radii_mm": [1, 3]}
    points = []
    for z in np.linspace(0.1, 9.9, 80):
        points.extend(([3, 0, z], [0, 3, z], [-3, 0, z]))
    metric, mask = MODULE.evaluate_carrier_support(np.asarray(points), event, thickness_mm=2, min_points_per_bin=2)
    assert mask.all()
    assert metric["carrier_coverage_fraction"] == pytest.approx(1.0)
    assert metric["median_carrier_radial_residual_mm"] == pytest.approx(0.0)


def test_distant_plane_does_not_fake_carrier():
    event = {"event_id": "E1", "line_start_mm": [0, 0, 0], "line_end_mm": [0, 0, 10], "radii_mm": [1, 3]}
    points = np.asarray([[20, y, z] for y in range(-5, 6) for z in range(11)], dtype=float)
    metric, mask = MODULE.evaluate_carrier_support(points, event, thickness_mm=2)
    assert not mask.any()
    assert metric["carrier_coverage_fraction"] == 0.0
