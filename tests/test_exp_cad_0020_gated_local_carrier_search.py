import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0020_gated_local_carrier_search.py"
SPEC = importlib.util.spec_from_file_location("exp20", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_saturated_nominal_support_forbids_search():
    event = {"event_id": "E1", "line_start_mm": [0, 0, 0], "line_end_mm": [0, 0, 10], "radii_mm": [1, 3]}
    points = []
    for z in np.linspace(0.1, 9.9, 100):
        points.extend(([3, 0, z], [0, 3, z], [-3, 0, z]))
    result, shifted, _mask = MODULE.gated_search_local_carrier(np.asarray(points), event, thickness_mm=2)
    assert result["search_status"] == "NOMINAL_SUPPORT_SATURATED_SEARCH_FORBIDDEN"
    assert result["best_shift_norm_mm"] == 0.0
    assert shifted == event
