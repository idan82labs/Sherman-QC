from __future__ import annotations

from copy import deepcopy

import pytest

from backend.f1_contained_parallel_fragment_pruning import (
    prune_contained_parallel_fragments,
)


def _line(candidate_id: str, start: tuple[float, float, float], end: tuple[float, float, float]):
    return {
        "candidate_id": candidate_id,
        "line_start": list(start),
        "line_end": list(end),
        "span_endpoints": [list(start), list(end)],
        "span_mm": sum((b - a) ** 2 for a, b in zip(start, end)) ** 0.5,
    }


def test_prunes_only_a_short_fully_contained_parallel_fragment() -> None:
    long = _line("long", (0.0, 0.0, 0.0), (50.0, 0.0, 0.0))
    fragment = _line("fragment", (20.0, 6.5, 0.0), (29.0, 6.5, 0.0))

    result = prune_contained_parallel_fragments([long, fragment])

    assert [item["candidate_id"] for item in result["selected"]] == ["long"]
    assert result["pruned_candidate_ids"] == ["fragment"]
    assert result["conflicts"][0]["short_span_overlap"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fragment",
    [
        _line("too_near", (20.0, 4.9, 0.0), (29.0, 4.9, 0.0)),
        _line("too_far", (20.0, 8.1, 0.0), (29.0, 8.1, 0.0)),
        _line("not_contained", (46.0, 6.5, 0.0), (55.0, 6.5, 0.0)),
        _line("not_short", (15.0, 6.5, 0.0), (26.0, 6.5, 0.0)),
        _line("not_parallel", (20.0, 6.5, 0.0), (29.0, 7.5, 0.0)),
    ],
)
def test_preserves_lines_outside_the_strict_conflict_contract(fragment) -> None:
    long = _line("long", (0.0, 0.0, 0.0), (50.0, 0.0, 0.0))

    result = prune_contained_parallel_fragments([fragment, long])

    assert [item["candidate_id"] for item in result["selected"]] == [fragment["candidate_id"], "long"]
    assert result["pruned_candidate_ids"] == []


def test_is_deterministic_order_preserving_and_does_not_mutate_inputs() -> None:
    selected = [
        _line("keep_first", (0.0, 20.0, 0.0), (12.0, 20.0, 0.0)),
        _line("long", (0.0, 0.0, 0.0), (50.0, 0.0, 0.0)),
        _line("fragment", (20.0, 6.5, 0.0), (29.0, 6.5, 0.0)),
    ]
    before = deepcopy(selected)

    first = prune_contained_parallel_fragments(selected)
    second = prune_contained_parallel_fragments(selected)

    assert first == second
    assert selected == before
    assert [item["candidate_id"] for item in first["selected"]] == ["keep_first", "long"]


def test_fails_closed_for_missing_or_degenerate_geometry() -> None:
    malformed = {"candidate_id": "malformed", "span_mm": "not-a-number"}
    degenerate = _line("degenerate", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    long = _line("long", (0.0, 0.0, 0.0), (50.0, 0.0, 0.0))

    result = prune_contained_parallel_fragments([malformed, degenerate, long])

    assert result["selected"] == [malformed, degenerate, long]
    assert result["pruned_candidate_ids"] == []
