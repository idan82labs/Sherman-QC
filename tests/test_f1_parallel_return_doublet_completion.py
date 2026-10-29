from __future__ import annotations

from backend.f1_parallel_return_doublet_completion import complete_parallel_return_doublet


def _line(
    candidate_id: str,
    *,
    y: float,
    span: float,
    bend_angle: float,
    confidence: float = 0.9,
    transition: float = 0.85,
    probability: float | None = None,
) -> dict:
    item = {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "bend_angle": bend_angle,
        "confidence": confidence,
        "two_sided_transition_score": transition,
    }
    if probability is not None:
        item["selection_probability"] = probability
    return item


def test_recovers_strong_complementary_parallel_return() -> None:
    seed = _line("SMA1", y=0.0, span=100.0, bend_angle=136.0, probability=0.91)
    return_line = _line("SMA2", y=1.4, span=109.0, bend_angle=93.0)

    result = complete_parallel_return_doublet([seed], [seed, return_line])

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == ["SMA1", "SMA2"]
    assert result["added"][0]["candidate_source"] == "parallel_return_doublet_completion"


def test_rejects_equal_span_duplicate_outline() -> None:
    seed = _line("SMA1", y=0.0, span=100.0, bend_angle=136.0, probability=0.91)
    outline = _line("SMA2", y=1.4, span=101.0, bend_angle=93.0)

    result = complete_parallel_return_doublet([seed], [seed, outline])

    assert result["activated"] is False
    assert [item["candidate_id"] for item in result["selected"]] == ["SMA1"]


def test_rejects_weak_seed_or_weak_candidate() -> None:
    weak_seed = _line("SMA1", y=0.0, span=100.0, bend_angle=136.0, probability=0.70)
    weak_return = _line(
        "SMA2", y=1.4, span=109.0, bend_angle=93.0, confidence=0.75
    )

    assert complete_parallel_return_doublet([weak_seed], [weak_seed, weak_return])["activated"] is False

