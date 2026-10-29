from __future__ import annotations

from copy import deepcopy

from backend.f1_route_independent_elongated_lane_v47 import compose_v47


def _prediction(candidate_id, start, end, route):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": list(start),
        "line_end": list(end),
        "selection_route": route,
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": "test",
    }


def _source(predictions, route):
    return {"route": route, "predictions": predictions}


def test_small_pool_route_suppresses_microfragment_and_stitches_lane():
    route = "bounded_mesh_small_pool_recall_consensus_repair"
    predictions = [
        _prediction("micro", (210, 18, 0), (220, 18, 0), route),
        _prediction("lane-a-left", (0, 0, 0), (214, 0, 0), route),
        _prediction("lane-b", (0, 34, 0), (536, 34, 0), route),
        _prediction("lane-a-right", (325, 5.9, 0), (538, 5.9, 0), route),
    ]
    result = compose_v47(_source(predictions, route), [538.0, 10.0, 38.0])
    audit = result["route_independent_elongated_lane_audit"]
    assert audit["gate_active"] is True
    assert audit["suppressed_microfragment_candidate_ids"] == ["micro"]
    assert len(audit["stitch_changes"]) == 1
    assert len(result["predictions"]) == 2


def test_full_route_correct_two_axis_result_is_passthrough():
    route = "full_resolution_mesh_v11"
    predictions = [
        _prediction("long", (0, 0, 0), (190, 0, 0), route),
        _prediction("root", (0, 0, 0), (0, 45, 0), route),
    ]
    result = compose_v47(_source(predictions, route), [202.0, 90.0, 57.0])
    assert result["route_independent_elongated_lane_audit"]["gate_active"] is False
    assert len(result["predictions"]) == 2


def test_input_permutation_is_exact():
    route = "bounded_mesh_small_pool_recall_consensus_repair"
    predictions = [
        _prediction("micro", (210, 18, 0), (220, 18, 0), route),
        _prediction("lane-a-left", (0, 0, 0), (214, 0, 0), route),
        _prediction("lane-b", (0, 34, 0), (536, 34, 0), route),
        _prediction("lane-a-right", (325, 5.9, 0), (538, 5.9, 0), route),
    ]
    first = compose_v47(_source(predictions, route), [538.0, 10.0, 38.0])
    second = compose_v47(
        _source(list(reversed(deepcopy(predictions))), route), [538.0, 10.0, 38.0]
    )
    assert first["predictions"] == second["predictions"]
    assert first["route_independent_elongated_lane_audit"] == second["route_independent_elongated_lane_audit"]
    assert first["v47_result_sha256"] == second["v47_result_sha256"]
