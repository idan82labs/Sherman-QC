import copy
import math

import pytest

from scripts.select_f1_full_mesh_adjacency_hysteresis_v17 import (
    FROZEN_V11_DUPLICATE_PREDICATE_SHA256,
    assert_frozen_duplicate_predicate,
    axial_localization_ratio,
    eligible_rejected_adjacency,
    frozen_duplicate_predicate_sha256,
    rescue_rejected_adjacency,
)


def _row(candidate_id: str, y: float = 0.0, **updates: float | str) -> dict:
    row = {
        "candidate_id": candidate_id,
        "representative_source": "scan_mesh_adjacency",
        "span_mm": 30.0,
        "span_endpoints": [[-15.0, y, 0.0], [15.0, y, 0.0]],
        "finite_support_fraction": 0.90,
        "finite_support_longest_run_fraction": 0.90,
        "endpoint_support_fraction": 2.0 / 3.0,
        "p90_line_distance_mm": 3.0,
        "two_sided_transition_score": 0.25,
        "two_sided_normal_angle_deg": 45.0,
        "two_sided_support_balance": 0.50,
        "two_sided_min_support_fraction": 0.20,
        "two_sided_normal_coherence": 0.50,
    }
    row.update(updates)
    return row


def test_frozen_v11_duplicate_predicate_hash_is_unchanged() -> None:
    assert frozen_duplicate_predicate_sha256() == FROZEN_V11_DUPLICATE_PREDICATE_SHA256
    assert_frozen_duplicate_predicate()


def test_eligibility_accepts_every_inclusive_boundary() -> None:
    row = _row("boundary")
    assert eligible_rejected_adjacency(row)
    assert axial_localization_ratio(row) == 10.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representative_source", "plane_lattice"),
        ("finite_support_fraction", 0.899999999999),
        ("finite_support_longest_run_fraction", 0.899999999999),
        ("endpoint_support_fraction", (2.0 / 3.0) - 1e-12),
        ("p90_line_distance_mm", 0.0),
        ("p90_line_distance_mm", 3.000000000001),
        ("span_mm", 29.999999999),
        ("two_sided_transition_score", 0.249999999999),
        ("two_sided_normal_angle_deg", 44.999999999),
        ("two_sided_support_balance", 0.499999999999),
        ("two_sided_min_support_fraction", 0.199999999999),
        ("two_sided_normal_coherence", 0.499999999999),
    ],
)
def test_eligibility_rejects_each_failed_gate(field: str, value: float | str) -> None:
    assert not eligible_rejected_adjacency(_row("failed", **{field: value}))


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf, -1.0])
def test_invalid_p90_is_rejected(value: float | None) -> None:
    row = _row("bad-p90")
    row["p90_line_distance_mm"] = value
    assert axial_localization_ratio(row) is None
    assert not eligible_rejected_adjacency(row)


def test_incumbent_wins_duplicate_conflict_and_rescue_is_append_only() -> None:
    incumbent = _row("incumbent", y=0.0)
    incumbent["representative_source"] = "cylindrical_standard"
    duplicate = _row("duplicate", y=1.0, span_mm=60.0, p90_line_distance_mm=1.0)
    independent = _row("independent", y=8.0, span_mm=50.0, p90_line_distance_mm=1.0)
    retained, rescues = rescue_rejected_adjacency(
        [incumbent, duplicate, independent], [0]
    )
    assert retained == [0, 2]
    assert [item["candidate_id"] for item in rescues] == ["independent"]


def test_ranking_is_deterministic_and_candidate_id_is_final_tie_break() -> None:
    rows = [_row("incumbent", y=50.0)]
    rows[0]["representative_source"] = "plane_lattice"
    rows += [_row("B", y=0.0), _row("A", y=10.0)]
    retained, rescues = rescue_rejected_adjacency(rows, [0])
    assert retained == [0, 2, 1]
    assert [item["candidate_id"] for item in rescues] == ["A", "B"]

    permuted = [copy.deepcopy(rows[index]) for index in (2, 0, 1)]
    retained_permuted, rescues_permuted = rescue_rejected_adjacency(permuted, [1])
    assert [permuted[index]["candidate_id"] for index in retained_permuted] == [
        "incumbent",
        "A",
        "B",
    ]
    assert [item["candidate_id"] for item in rescues_permuted] == ["A", "B"]
