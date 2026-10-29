from __future__ import annotations

from copy import deepcopy

from backend.f1_elongated_lane_consolidation_v46 import compose_v46


ROUTE = "bounded_mesh_complex_v15_paired_plane_fold_consensus_repair"


def _prediction(candidate_id, start, end):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": list(start),
        "line_end": list(end),
        "selection_route": ROUTE,
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": "test",
    }


def _source(predictions, route=ROUTE):
    return {"route": route, "predictions": predictions}


def test_elongated_gate_suppresses_terminal_artifacts_and_stitches_lane():
    predictions = [
        _prediction("terminal-left", (0, 0, 0), (0, 9, 0)),
        _prediction("lane-a-left", (0, 0, 0), (103, 0, 0)),
        _prediction("terminal-right", (418, 0, 0), (418, 9, 0)),
        _prediction("lane-b", (230, 38, 0), (409, 38, 0)),
        _prediction("lane-a-right", (220, 3, 0), (296, 3, 0)),
    ]
    result = compose_v46(_source(predictions), [418.0, 81.0, 20.0])
    audit = result["elongated_lane_consolidation_audit"]
    assert audit["gate_active"] is True
    assert audit["suppressed_terminal_candidate_ids"] == [
        "terminal-left", "terminal-right"
    ]
    assert len(audit["stitch_changes"]) == 1
    assert len(result["predictions"]) == 2
    assert {row["source_kind"] for row in result["predictions"]} == {
        "elongated_lane_consolidation", "test"
    }


def test_nonelongated_scan_is_semantic_passthrough():
    predictions = [
        _prediction("short", (0, 0, 0), (0, 9, 0)),
        _prediction("a", (0, 0, 0), (100, 0, 0)),
        _prediction("b", (120, 0, 0), (220, 0, 0)),
        _prediction("c", (0, 30, 0), (220, 30, 0)),
    ]
    result = compose_v46(_source(predictions), [260.0, 156.0, 30.0])
    assert result["elongated_lane_consolidation_audit"]["gate_active"] is False
    assert {row["source_hypothesis_id"] for row in result["predictions"]} == {
        "short", "a", "b", "c"
    }


def test_requires_three_substantial_parallel_members():
    predictions = [
        _prediction("short", (0, 0, 0), (0, 9, 0)),
        _prediction("a", (0, 0, 0), (100, 0, 0)),
        _prediction("b", (230, 30, 0), (409, 30, 0)),
    ]
    result = compose_v46(_source(predictions), [418.0, 81.0, 20.0])
    assert result["elongated_lane_consolidation_audit"]["gate_active"] is False
    assert len(result["predictions"]) == 3


def test_input_permutation_is_byte_exact_in_result_payload():
    predictions = [
        _prediction("terminal-left", (0, 0, 0), (0, 9, 0)),
        _prediction("lane-a-left", (0, 0, 0), (103, 0, 0)),
        _prediction("terminal-right", (418, 0, 0), (418, 9, 0)),
        _prediction("lane-b", (230, 38, 0), (409, 38, 0)),
        _prediction("lane-a-right", (220, 3, 0), (296, 3, 0)),
    ]
    first = compose_v46(_source(predictions), [418.0, 81.0, 20.0])
    second = compose_v46(_source(list(reversed(deepcopy(predictions)))), [418.0, 81.0, 20.0])
    assert first["predictions"] == second["predictions"]
    assert first["elongated_lane_consolidation_audit"] == second["elongated_lane_consolidation_audit"]
    assert first["v46_result_sha256"] == second["v46_result_sha256"]
