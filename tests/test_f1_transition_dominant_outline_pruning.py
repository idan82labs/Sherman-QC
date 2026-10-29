from backend.f1_transition_dominant_outline_pruning import (
    prune_transition_dominant_outline_fan,
)


def _candidate(candidate_id: str, score: float):
    return {"candidate_id": candidate_id, "two_sided_transition_score": score}


def test_prunes_outline_fan_with_one_material_transition():
    result = prune_transition_dominant_outline_fan(
        [
            _candidate("physical", 0.64),
            _candidate("outline-a", 0.28),
            _candidate("outline-b", 0.25),
            _candidate("outline-c", 0.24),
        ]
    )

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == ["physical"]
    assert len(result["pruned"]) == 3


def test_abstains_when_second_transition_is_material():
    candidates = [
        _candidate("bend-a", 0.82),
        _candidate("bend-b", 0.55),
        _candidate("outline-a", 0.20),
        _candidate("outline-b", 0.10),
    ]

    result = prune_transition_dominant_outline_fan(candidates)

    assert result["activated"] is False
    assert result["selected"] == candidates


def test_abstains_on_small_candidate_sets():
    candidates = [_candidate("bend", 0.90), _candidate("outline", 0.10)]

    result = prune_transition_dominant_outline_fan(candidates)

    assert result["activated"] is False
    assert result["selected"] == candidates
