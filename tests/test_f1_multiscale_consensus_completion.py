from __future__ import annotations

from backend.f1_multiscale_consensus_completion import complete_multiscale_consensus


def _line(candidate_id: str, *, y: float, span: float, transition: float, sources: int = 4) -> dict:
    return {
        "candidate_id": candidate_id,
        "representative_source": "cylindrical_standard",
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "cluster_source_count": sources,
        "cluster_max_two_sided_transition_score": transition,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
        "case_percentile_score": 0.8,
    }


def _selected() -> list[dict]:
    return [
        {
            **_line(f"S{index}", y=float(index * 20), span=60.0, transition=0.9),
            "candidate_source": "scan_mesh_adjacency",
        }
        for index in range(5)
    ]


def test_completes_long_and_tiny_four_source_consensus() -> None:
    long_line = _line("LONG", y=120.0, span=150.0, transition=0.96)
    tiny_line = _line("TINY", y=140.0, span=14.0, transition=0.64)

    result = complete_multiscale_consensus(_selected(), [long_line, tiny_line])

    assert result["activated"] is True
    assert {item["completion_mode"] for item in result["added"]} == {
        "long_four_source_consensus",
        "tiny_four_source_consensus",
    }


def test_rejects_completion_before_object_is_established() -> None:
    long_line = _line("LONG", y=120.0, span=150.0, transition=0.96)

    result = complete_multiscale_consensus(_selected()[:4], [long_line])

    assert result["activated"] is False
    assert result["reason"] == "fewer_than_five_selected_bends"


def test_rejects_medium_scale_and_weak_consensus() -> None:
    medium = _line("MEDIUM", y=120.0, span=60.0, transition=0.99)
    weak = _line("WEAK", y=140.0, span=14.0, transition=0.64, sources=3)

    assert complete_multiscale_consensus(_selected(), [medium, weak])["activated"] is False

