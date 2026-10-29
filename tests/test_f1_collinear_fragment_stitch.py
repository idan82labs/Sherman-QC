from __future__ import annotations

from backend.f1_collinear_fragment_stitch import stitch_collinear_selected_fragments


def _row(candidate_id: str, start, end):
    return {
        "candidate_id": candidate_id,
        "span_endpoints": [list(start), list(end)],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": float(end[0] - start[0]),
    }


def test_stitches_end_to_end_fragments_into_one_axis():
    rows = [
        _row("a", (0, 0, 0), (40, 0, 0)),
        _row("b", (38, 1, 0), (80, 1, 0)),
        _row("c", (79, 2, 0), (120, 2, 0)),
    ]
    result = stitch_collinear_selected_fragments(rows)
    assert len(result["selected"]) == 1
    assert result["selected"][0]["span_mm"] > 119.0
    assert result["changes"][0]["member_count"] == 3


def test_does_not_merge_overlapping_alternative_lanes_without_span_gain():
    rows = [
        _row("a", (0, 0, 0), (100, 0, 0)),
        _row("b", (2, 2, 0), (98, 2, 0)),
        _row("c", (1, 4, 0), (99, 4, 0)),
    ]
    result = stitch_collinear_selected_fragments(rows)
    assert result["selected"] == rows
    assert result["changes"] == []


def test_does_not_merge_distinct_parallel_bends():
    rows = [
        _row("a", (0, 0, 0), (100, 0, 0)),
        _row("b", (0, 20, 0), (100, 20, 0)),
    ]
    result = stitch_collinear_selected_fragments(rows)
    assert result["selected"] == rows
    assert result["changes"] == []
