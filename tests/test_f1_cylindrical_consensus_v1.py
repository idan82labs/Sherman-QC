from __future__ import annotations

from copy import deepcopy

from scripts.select_f1_cylindrical_consensus_v1 import select


def _raw(seed: str, score: float, ordinal: int) -> dict:
    return {
        "candidate_id": f"cylindrical_standard::{seed}_{ordinal}",
        "candidate_source": "cylindrical_standard",
        "seed_ids": [seed],
        "two_sided_transition_score": score,
    }


def _line(
    seed: str,
    ordinal: int,
    *,
    consensus: int,
    normal_angle: float,
    transition: float,
    balance: float = 0.96,
    span: float = 40.0,
) -> dict:
    return {
        "candidate_id": f"cylindrical_standard::{seed}_{ordinal}",
        "representative_source": "cylindrical_standard",
        "seed_ids": [seed],
        "span_endpoints": [[0.0, float(ordinal), 0.0], [span, float(ordinal), 0.0]],
        "span_mm": span,
        "radius_mm": 2.0,
        "consensus_count": consensus,
        "radius_consensus_count": 4,
        "score": float(consensus * 10),
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "p90_line_distance_mm": 1.0,
        "two_sided_transition_score": transition,
        "two_sided_normal_angle_deg": normal_angle,
        "two_sided_support_balance": balance,
    }


def test_selection_is_input_order_invariant_and_count_free() -> None:
    payload = {
        "candidates": [
            _raw("ORTH", 0.90, 1),
            _raw("ORTH", 0.50, 2),
            _raw("ORTH", 0.31, 3),
            _raw("WEAK", 0.87, 1),
            _raw("WEAK", 0.86, 2),
        ],
        "line_objects": [
            _line("ORTH", 1, consensus=20, normal_angle=82.0, transition=0.85),
            _line("ORTH", 2, consensus=30, normal_angle=84.0, transition=0.86),
            _line("SHALLOW", 1, consensus=22, normal_angle=18.0, transition=0.22, span=24.0),
            _line("SHALLOW", 2, consensus=25, normal_angle=18.0, transition=0.10, span=24.0),
            _line("FALSE", 1, consensus=100, normal_angle=20.0, transition=0.30, balance=0.70, span=24.0),
        ],
    }
    first = select(payload)
    reversed_payload = deepcopy(payload)
    reversed_payload["candidates"].reverse()
    reversed_payload["line_objects"].reverse()
    second = select(reversed_payload)

    assert first == second
    assert first["candidate_count"] == 2
    assert {row["selection_route"] for row in first["predictions"]} == {
        "stable_near_orthogonal_group",
        "balanced_shallow_partial_cylinder",
    }
    assert first["inference_contract"]["expected_count_exposed"] is False
    assert first["inference_contract"]["oracle_feedback_exposed"] is False

