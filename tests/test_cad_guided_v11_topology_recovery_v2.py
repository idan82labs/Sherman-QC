import numpy as np

from backend.cad_guided_v11_topology_recovery_v2 import select_pose_v2


def _candidate(index=0, p95=1.0, coverage=0.8):
    return {
        "rotation": np.eye(3).tolist(),
        "translation": [1.0, 2.0, 3.0],
        "conditional_context_support": {
            "finite_distance_metrics": {"p95_upper": p95, "rms_upper": p95 / 2.0},
            "geometric_scan_coverage_lower": coverage,
        },
        "index": index,
    }


def test_admitted_context_keeps_exact_context_identity():
    answer = select_pose_v2({
        "measurement_context_indices": [1],
        "candidates": [_candidate(0, p95=1.9), _candidate(1, p95=0.2)],
    })
    assert answer["status"] == "AVAILABLE"
    assert answer["pose_source"] == "RETAINED_V11_ACCEPTED_CONTEXT"
    assert answer["registration_context_accepted"] is True
    assert answer["candidate_index"] == 1
    assert np.allclose(answer["rotation"], np.eye(3))


def test_rejected_context_requires_global_provisional_gate_and_is_deterministic():
    answer = select_pose_v2({
        "measurement_context_indices": [],
        "candidates": [
            _candidate(0, p95=2.1, coverage=0.99),
            _candidate(1, p95=1.5, coverage=0.56),
            _candidate(2, p95=1.5, coverage=0.80),
        ],
    })
    assert answer["status"] == "AVAILABLE"
    assert answer["pose_source"] == "RETAINED_V11_REJECTED_CONTEXT_GATED_PROVISIONAL"
    assert answer["registration_context_accepted"] is False
    assert answer["candidate_index"] == 2
    assert answer["quality"]["p95_upper_mm"] <= 2.0
    assert answer["quality"]["coverage_lower"] >= 0.55


def test_no_finite_or_gated_candidate_is_an_abstention():
    answer = select_pose_v2({
        "measurement_context_indices": [],
        "candidates": [_candidate(0, p95=2.01, coverage=0.99), {"index": 1}],
    })
    assert answer["status"] == "UNAVAILABLE"
    assert answer["reason"] == "NO_GATED_PROVISIONAL_V11_POSE"

