from __future__ import annotations

from copy import deepcopy

import pytest

from backend.f1_post_v51_precision_ownership_v52_development import (
    PROFILE,
    compose_v52_development,
)


def _prediction(
    token: str,
    kind: str,
    *,
    length: float,
    score: float | None = 0.8,
    y: float = 0.0,
) -> dict:
    return {
        "bend_id": "P-" + token,
        "line_start": [0.0, y, 0.0],
        "line_end": [length, y, 0.0],
        "selection_route": "synthetic_source_only_test",
        "selector_score": score,
        "source_hypothesis_id": token,
        "source_kind": kind,
    }


def _candidate(
    token: str,
    *,
    normal: float = 0.40,
    p90: float = 1.0,
    gap: float = 0.1,
    transition: float = 0.90,
) -> dict:
    return {
        "candidate_id": token,
        "normal_second_eigenvalue": normal,
        "p90_line_distance_mm": p90,
        "min_surface_gap": gap,
        "two_sided_transition_score": transition,
    }


def _source(predictions: list[dict]) -> dict:
    return {
        "predictions": predictions,
        "v51_result_sha256": "frozen-v51-test-sha",
        "labels_used": False,
        "cad_used": False,
        "part_identity_used": False,
    }


@pytest.mark.parametrize(
    ("prediction", "candidate"),
    [
        (
            _prediction("plane", "plane_lattice", length=60.0),
            _candidate("plane", normal=0.39, transition=0.90),
        ),
        (
            _prediction("scan", "scan_mesh_adjacency", length=80.0),
            _candidate("scan", normal=0.33, p90=1.1, gap=0.12, transition=0.80),
        ),
        (
            _prediction("compact", "cylindrical_standard", length=19.0, score=0.4),
            _candidate("compact", normal=0.29, p90=1.8, transition=0.83),
        ),
        (
            _prediction("long", "cylindrical_standard", length=190.0, score=1.2),
            _candidate("long", normal=0.44, p90=1.1, transition=0.96),
        ),
        (
            _prediction("sidecar", "v41_fragment_tolerant_parent_bridge_sidecar", length=95.0, score=None),
            _candidate("sidecar", normal=0.33, p90=1.7, transition=0.73),
        ),
        (
            _prediction("consensus", "source_consensus_midpoint", length=220.0, score=0.9),
            None,
        ),
    ],
)
def test_accepts_each_supported_high_quality_family(prediction, candidate):
    pool = {"candidates": [] if candidate is None else [candidate]}
    result = compose_v52_development(_source([prediction]), pool)
    assert [row["bend_id"] for row in result["predictions"]] == [prediction["bend_id"]]
    assert result["precision_ownership_decisions"][0]["final_accepted"] is True


@pytest.mark.parametrize(
    ("prediction", "candidate"),
    [
        (
            _prediction("weak-plane", "plane_lattice", length=60.0),
            _candidate("weak-plane", normal=0.30, transition=0.90),
        ),
        (
            _prediction("edge-scan", "scan_mesh_adjacency", length=80.0),
            _candidate("edge-scan", normal=0.33, p90=1.1, gap=1.0, transition=0.80),
        ),
        (
            _prediction("wide-compact", "cylindrical_standard", length=19.0, score=0.4),
            _candidate("wide-compact", normal=0.39, p90=1.0, transition=0.90),
        ),
        (
            _prediction("weak-long", "cylindrical_standard", length=190.0, score=1.2),
            _candidate("weak-long", normal=0.44, p90=1.1, transition=0.90),
        ),
        (
            _prediction("paired", "paired_return_completion", length=220.0, score=2.0),
            _candidate("paired", normal=0.5, p90=0.5, transition=1.0),
        ),
        (
            _prediction("strict", "strict_blind_owned", length=60.0, score=2.0),
            _candidate("strict", normal=0.5, p90=0.5, transition=1.0),
        ),
        (
            _prediction("missing", "plane_lattice", length=60.0),
            None,
        ),
    ],
)
def test_fails_closed_on_low_quality_unsupported_or_missing_evidence(prediction, candidate):
    pool = {"candidates": [] if candidate is None else [candidate]}
    result = compose_v52_development(_source([prediction]), pool)
    assert result["predictions"] == []
    assert result["precision_ownership_decisions"][0]["final_accepted"] is False


def test_higher_score_owns_a_near_duplicate():
    low = _prediction("low", "plane_lattice", length=60.0, score=0.7, y=0.5)
    high = _prediction("high", "plane_lattice", length=60.0, score=0.9, y=0.0)
    pool = {
        "candidates": [
            _candidate("low", normal=0.39, transition=0.90),
            _candidate("high", normal=0.39, transition=0.90),
        ]
    }
    result = compose_v52_development(_source([low, high]), pool)
    assert [row["bend_id"] for row in result["predictions"]] == ["P-high"]
    assert result["duplicate_reconciliation"][0]["owner_prediction_id"] == "P-high"
    decisions = {row["prediction_id"]: row for row in result["precision_ownership_decisions"]}
    assert decisions["P-low"]["reason"] == "near_duplicate_owned_by_higher_ranked_prediction"


def test_pool_permutation_is_exact_and_inputs_are_not_mutated():
    predictions = [
        _prediction("plane", "plane_lattice", length=60.0),
        _prediction("scan", "scan_mesh_adjacency", length=80.0),
    ]
    candidates = [
        _candidate("plane", normal=0.39, transition=0.90),
        _candidate("scan", normal=0.33, p90=1.1, gap=0.12, transition=0.80),
    ]
    source = _source(predictions)
    pool = {"candidates": candidates}
    source_before = deepcopy(source)
    pool_before = deepcopy(pool)
    first = compose_v52_development(source, pool)
    second = compose_v52_development(source, {"candidates": list(reversed(candidates))})
    assert first == second
    assert source == source_before
    assert pool == pool_before
    assert first["profile"] == PROFILE
    assert first["labels_used"] is False
    assert first["development_tuning_labels_used"] is True
    assert first["promotion_eligible"] is False


@pytest.mark.parametrize("flag", ["labels_used", "cad_used", "part_identity_used"])
def test_rejects_non_source_only_v51_input(flag):
    source = _source([])
    source[flag] = True
    with pytest.raises(ValueError):
        compose_v52_development(source, {"candidates": []})

