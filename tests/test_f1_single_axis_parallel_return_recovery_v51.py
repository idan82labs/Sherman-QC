from __future__ import annotations

from copy import deepcopy

from backend.f1_single_axis_parallel_return_recovery_v51 import compose_v51


def _prediction(candidate_id, y=0.0):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": [0.0, y, 0.0],
        "line_end": [100.0, y, 0.0],
        "selection_route": "test",
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": "cylindrical_standard",
    }


def _candidate(candidate_id, y, score, length=90.0):
    return {
        "candidate_id": candidate_id,
        "span_endpoints": [[5.0, y, 0.0], [5.0 + length, y, 0.0]],
        "representative_source": "cylindrical_standard",
        "score": score,
    }


def test_recovers_highest_support_distinct_parallel_family():
    source = {"predictions": [_prediction("cylindrical_standard::CB1_1")]}
    pool = {
        "scan_bbox_mm": [100.0, 20.0, 10.0],
        "line_objects": [
            _candidate("cylindrical_standard::CB1_2", 8.0, 100.0),
            _candidate("cylindrical_standard::CB2_1", 8.0, 20.0),
            _candidate("cylindrical_standard::CB2_2", 8.0, 50.0),
        ],
    }
    result = compose_v51(source, pool)
    assert len(result["predictions"]) == 2
    assert result["single_axis_parallel_return_recovery_audit"][
        "selected_candidate_id"
    ] == "cylindrical_standard::CB2_2"


def test_does_not_recover_when_more_than_one_prediction_survives():
    source = {
        "predictions": [
            _prediction("cylindrical_standard::CB1_1"),
            _prediction("cylindrical_standard::CB2_1", 10.0),
        ]
    }
    pool = {
        "scan_bbox_mm": [100.0, 20.0, 10.0],
        "line_objects": [_candidate("cylindrical_standard::CB3_1", 20.0, 100.0)],
    }
    assert len(compose_v51(source, pool)["predictions"]) == 2


def test_rejects_same_family_and_too_close_candidates():
    source = {"predictions": [_prediction("cylindrical_standard::CB1_1")]}
    pool = {
        "scan_bbox_mm": [100.0, 20.0, 10.0],
        "line_objects": [
            _candidate("cylindrical_standard::CB1_2", 10.0, 100.0),
            _candidate("cylindrical_standard::CB2_1", 4.0, 100.0),
        ],
    }
    assert len(compose_v51(source, pool)["predictions"]) == 1


def test_pool_permutation_is_exact():
    source = {"predictions": [_prediction("cylindrical_standard::CB1_1")]}
    candidates = [
        _candidate("cylindrical_standard::CB2_1", 8.0, 20.0),
        _candidate("cylindrical_standard::CB2_2", 8.0, 50.0),
    ]
    first = compose_v51(source, {"scan_bbox_mm": [100, 20, 10], "line_objects": candidates})
    second = compose_v51(
        source,
        {"scan_bbox_mm": [100, 20, 10], "line_objects": list(reversed(deepcopy(candidates)))},
    )
    assert first["predictions"] == second["predictions"]
    assert first["single_axis_parallel_return_recovery_audit"] == second[
        "single_axis_parallel_return_recovery_audit"
    ]
    assert first["v51_result_sha256"] == second["v51_result_sha256"]
