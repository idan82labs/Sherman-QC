from __future__ import annotations

from scripts.select_f1_topology_finite_span_completion_v8 import _eligible, select


def _row(candidate_id: str, y: float, *, source_count: int, finite: float, longest: float) -> dict:
    sources = ["strict_blind_owned"] + [f"source-{index}" for index in range(max(source_count - 1, 0))]
    return {
        "candidate_id": candidate_id,
        "representative_source": "cylindrical_standard",
        "cluster_sources": sources,
        "cluster_source_count": source_count,
        "cluster_member_count": source_count,
        "finite_support_fraction": finite,
        "finite_support_longest_run_fraction": longest,
        "two_sided_transition_score": 0.8,
        "case_percentile_score": 0.9,
        "span_mm": 40.0,
        "span_endpoints": [[0.0, y, 0.0], [40.0, y, 0.0]],
    }


def test_eligible_requires_three_sources_and_complete_finite_run() -> None:
    assert _eligible(_row("good", 0.0, source_count=3, finite=0.98, longest=0.98))
    assert not _eligible(_row("two-sources", 0.0, source_count=2, finite=1.0, longest=1.0))
    assert not _eligible(_row("broken", 0.0, source_count=3, finite=1.0, longest=0.60))
    without_independent = _row("no-independent", 0.0, source_count=3, finite=1.0, longest=1.0)
    without_independent["cluster_sources"] = ["a", "b", "c"]
    assert not _eligible(without_independent)
    merged = _row("merged", 0.0, source_count=3, finite=1.0, longest=1.0)
    merged["representative_source"] = "fragment_merge"
    assert not _eligible(merged)


def test_select_adds_distinct_finite_span_and_suppresses_alias() -> None:
    rows = [
        _row("first", 0.0, source_count=3, finite=1.0, longest=1.0),
        _row("alias", 1.0, source_count=3, finite=1.0, longest=1.0),
        _row("distinct", 8.0, source_count=3, finite=1.0, longest=1.0),
    ]
    selected, additions = select(rows, bbox_span=100.0, bounded_mesh_route=True)
    assert len(selected) == 2
    assert 2 in selected
    assert len(set(selected) & {0, 1}) == 1
    assert additions == set(selected)


def test_select_fails_closed_for_non_mesh_pool() -> None:
    rows = [_row("candidate", 0.0, source_count=3, finite=1.0, longest=1.0)]
    selected, additions = select(
        rows,
        bbox_span=100.0,
        bounded_mesh_route=False,
        core_scores=__import__("numpy").zeros(1),
        adjacency_scores=__import__("numpy").zeros(1),
    )
    assert selected == []
    assert additions == set()
