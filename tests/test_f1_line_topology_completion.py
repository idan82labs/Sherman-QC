from __future__ import annotations

from backend.f1_line_topology_completion import (
    complete_parallel_endpoint_connectors,
    merge_collinear_fragments,
)


def test_fragment_merge_and_parallel_connectors_close_rectangle():
    fragments = [
        {"bend_id": "L1a", "span_endpoints": [[0, 0, 0], [4, 0, 0]]},
        {"bend_id": "L1b", "span_endpoints": [[4.5, 0, 0], [10, 0, 0]]},
        {"bend_id": "L2a", "span_endpoints": [[0, 20, 0], [6, 20, 0]]},
        {"bend_id": "L2b", "span_endpoints": [[6.5, 20, 0], [10, 20, 0]]},
    ]
    merged = merge_collinear_fragments(fragments, gap_tolerance_mm=1.0)
    connectors = complete_parallel_endpoint_connectors(merged)
    assert len(merged) == 2
    assert sorted(round(item["span_mm"], 6) for item in merged) == [10.0, 10.0]
    assert len(connectors) == 2
    assert all(round(item["span_mm"], 6) == 20.0 for item in connectors)


def test_connectors_reject_unrelated_length_mismatch():
    lines = [
        {"bend_id": "long", "span_endpoints": [[0, 0, 0], [100, 0, 0]]},
        {"bend_id": "short", "span_endpoints": [[0, 20, 0], [20, 20, 0]]},
    ]
    assert complete_parallel_endpoint_connectors(lines) == []
