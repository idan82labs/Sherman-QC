from __future__ import annotations

from copy import deepcopy

from backend.f1_nested_partial_duplicate_nms_v48 import compose_v48


def _prediction(candidate_id, start, end):
    return {
        "bend_id": "legacy-" + candidate_id,
        "line_start": list(start),
        "line_end": list(end),
        "selection_route": "test",
        "selector_score": 0.5,
        "source_hypothesis_id": candidate_id,
        "source_kind": "test",
    }


def test_suppresses_only_nested_partial_duplicate():
    source = {
        "predictions": [
            _prediction("full-left", (0, 0, 0), (90, 0, 0)),
            _prediction("partial-left", (40, 7.5, 0), (84, 7.5, 0)),
            _prediction("full-right", (0, 138, 0), (97, 138, 0)),
        ]
    }
    result = compose_v48(source)
    assert len(result["predictions"]) == 2
    assert result["nested_partial_duplicate_nms_changes"][0]["suppressed_candidate_id"] == "partial-left"


def test_preserves_nearby_full_length_physical_axes():
    source = {
        "predictions": [
            _prediction("a", (0, 0, 0), (90, 0, 0)),
            _prediction("b", (0, 7.5, 0), (88, 7.5, 0)),
        ]
    }
    result = compose_v48(source)
    assert len(result["predictions"]) == 2
    assert result["nested_partial_duplicate_nms_changes"] == []


def test_input_permutation_is_exact():
    predictions = [
        _prediction("full-left", (0, 0, 0), (90, 0, 0)),
        _prediction("partial-left", (40, 7.5, 0), (84, 7.5, 0)),
        _prediction("full-right", (0, 138, 0), (97, 138, 0)),
    ]
    first = compose_v48({"predictions": predictions})
    second = compose_v48({"predictions": list(reversed(deepcopy(predictions)))})
    assert first["predictions"] == second["predictions"]
    assert first["nested_partial_duplicate_nms_changes"] == second["nested_partial_duplicate_nms_changes"]
    assert first["v48_result_sha256"] == second["v48_result_sha256"]
