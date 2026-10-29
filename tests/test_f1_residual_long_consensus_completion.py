from __future__ import annotations

from backend.f1_residual_long_consensus_completion import complete_residual_long_consensus


def _line(candidate_id: str, *, y: float, span: float = 200.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
    }


def _selected(count: int = 9) -> list[dict]:
    return [_line(f"S{index}", y=float(index * 20)) for index in range(count)]


def _candidate(y: float = 190.0) -> dict:
    return {
        **_line("RESIDUAL", y=y),
        "representative_source": "cylindrical_standard",
        "cluster_source_count": 2,
        "cluster_max_two_sided_transition_score": 0.91,
        "case_percentile_score": 1.0,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
    }


def test_completes_separated_maximum_rank_long_consensus() -> None:
    result = complete_residual_long_consensus(_selected(), [_candidate()])

    assert result["activated"] is True
    assert result["added"][0]["candidate_source"] == "residual_long_consensus_completion"


def test_rejects_near_duplicate_residual() -> None:
    assert complete_residual_long_consensus(_selected(), [_candidate(y=164.5)])["activated"] is False


def test_requires_mature_bend_object_and_maximum_rank() -> None:
    candidate = _candidate()
    candidate["case_percentile_score"] = 0.99

    assert complete_residual_long_consensus(_selected(8), [_candidate()])["activated"] is False
    assert complete_residual_long_consensus(_selected(), [candidate])["activated"] is False

