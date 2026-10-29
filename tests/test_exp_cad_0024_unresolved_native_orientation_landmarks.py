import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0024_unresolved_native_orientation_landmarks.py"
SPEC = importlib.util.spec_from_file_location("exp24", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _row(candidate_id, transform, minimum, mean, p90):
    return {
        "candidate_id": candidate_id,
        "absolute_transform": np.asarray(transform).tolist(),
        "orientation_landmark_metrics": {
            "minimum_table_coverage": minimum,
            "mean_equal_table_coverage": mean,
            "combined_p90_distance_mm": p90,
        },
    }


def test_near_duplicate_candidates_form_one_pose_cluster():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.001
    third = np.eye(4)
    third[0, 3] = 1.0
    clusters = MODULE.cluster_candidates([
        _row("A", first, 1, 1, 0.2),
        _row("B", second, 1, 1, 0.2),
        _row("C", third, 0.5, 0.5, 2.0),
    ])
    assert len(clusters) == 2
    assert clusters[0]["members"] == ["A", "B"]


def test_orientation_gate_requires_complete_best_and_margin():
    best = {"representative": _row("A", np.eye(4), 1.0, 1.0, 0.5), "members": ["A"]}
    runner = {"representative": _row("B", np.eye(4), 0.6, 0.6, 3.0), "members": ["B"]}
    passed, reasons = MODULE.orientation_gate([best, runner])
    assert passed and not reasons
    close_runner = {"representative": _row("C", np.eye(4), 0.9, 0.85, 1.0), "members": ["C"]}
    passed, reasons = MODULE.orientation_gate([best, close_runner])
    assert not passed and "RUNNER_UP_MARGIN_GATE" in reasons
