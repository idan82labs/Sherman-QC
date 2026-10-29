import copy

from scripts.select_f1_full_mesh_adjacency_hysteresis_v17 import rescue_rejected_adjacency
from scripts.validate_f1_mesh_route_v17 import independent_eligible, independent_rescue


def _row(candidate_id: str, y: float, **updates: float | str) -> dict:
    row = {
        "candidate_id": candidate_id,
        "representative_source": "scan_mesh_adjacency",
        "span_mm": 40.0,
        "span_endpoints": [[-20.0, y, 0.0], [20.0, y, 0.0]],
        "finite_support_fraction": 0.95,
        "finite_support_longest_run_fraction": 0.95,
        "endpoint_support_fraction": 0.8,
        "p90_line_distance_mm": 2.0,
        "two_sided_transition_score": 0.4,
        "two_sided_normal_angle_deg": 60.0,
        "two_sided_support_balance": 0.7,
        "two_sided_min_support_fraction": 0.3,
        "two_sided_normal_coherence": 0.6,
    }
    row.update(updates)
    return row


def test_independent_validator_matches_implementation_and_order_permutation() -> None:
    rows = [
        _row("incumbent", 50.0, representative_source="plane_lattice"),
        _row("B", 0.0),
        _row("A", 10.0),
        _row("weak", 20.0, two_sided_normal_angle_deg=44.0),
    ]
    expected_indices, expected_ids = independent_rescue(rows, [0])
    actual_indices, actual_rows = rescue_rejected_adjacency(rows, [0])
    assert actual_indices == expected_indices
    assert [row["candidate_id"] for row in actual_rows] == expected_ids == ["A", "B"]
    assert independent_eligible(rows[1])
    assert not independent_eligible(rows[3])

    permuted = [copy.deepcopy(rows[index]) for index in (2, 3, 0, 1)]
    expected_permuted, ids_permuted = independent_rescue(permuted, [2])
    actual_permuted, rows_permuted = rescue_rejected_adjacency(permuted, [2])
    assert [permuted[index]["candidate_id"] for index in expected_permuted] == [
        "incumbent",
        "A",
        "B",
    ]
    assert [permuted[index]["candidate_id"] for index in actual_permuted] == [
        "incumbent",
        "A",
        "B",
    ]
    assert [row["candidate_id"] for row in rows_permuted] == ids_permuted == ["A", "B"]
