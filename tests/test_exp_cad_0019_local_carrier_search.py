import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0019_local_carrier_search.py"
SPEC = importlib.util.spec_from_file_location("exp19", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_recovers_shifted_cylindrical_carrier():
    event = {"event_id": "E1", "line_start_mm": [0, 0, 0], "line_end_mm": [0, 0, 20], "radii_mm": [1, 3]}
    points = []
    center = np.asarray([4.0, -2.0, 0.0])
    for z in np.linspace(0.1, 19.9, 120):
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            points.append(center + [3 * np.cos(angle), 3 * np.sin(angle), z])
    result, shifted, _mask = MODULE.search_local_carrier(np.asarray(points), event, thickness_mm=1)
    assert result["best_coverage_fraction"] == 1.0
    assert np.linalg.norm(np.asarray(result["best_shift_vector_mm"]) - center) <= 0.5
    assert np.asarray(shifted["line_start_mm"])[:2].tolist() == result["best_shift_vector_mm"][:2]
