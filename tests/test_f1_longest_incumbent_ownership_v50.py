from __future__ import annotations

from copy import deepcopy

from backend.f1_longest_incumbent_ownership_v50 import compose_v50


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


def test_duplicate_ownership_retains_longest_incumbent():
    source = {
        "route": "test",
        "predictions": [
            _prediction("short", (88, 1, 0), (100, 1, 0)),
            _prediction("long", (0, 0, 0), (100, 0, 0)),
        ],
    }
    result = compose_v50(source, [], 100.0)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["source_hypothesis_id"] == "long"
    assert result["post_completion_nms_suppressed_candidate_ids"] == ["short"]


def test_preserves_distinct_incumbents():
    source = {
        "route": "test",
        "predictions": [
            _prediction("a", (0, 0, 0), (100, 0, 0)),
            _prediction("b", (0, 20, 0), (100, 20, 0)),
        ],
    }
    assert len(compose_v50(source, [], 100.0)["predictions"]) == 2


def test_input_permutation_is_exact():
    predictions = [
        _prediction("short", (88, 1, 0), (100, 1, 0)),
        _prediction("long", (0, 0, 0), (100, 0, 0)),
        _prediction("other", (0, 20, 0), (100, 20, 0)),
    ]
    first = compose_v50({"route": "test", "predictions": predictions}, [], 100.0)
    second = compose_v50(
        {"route": "test", "predictions": list(reversed(deepcopy(predictions)))},
        [],
        100.0,
    )
    assert first["predictions"] == second["predictions"]
    assert first["post_completion_nms_suppressed_candidate_ids"] == second[
        "post_completion_nms_suppressed_candidate_ids"
    ]
    assert first["v50_result_sha256"] == second["v50_result_sha256"]
