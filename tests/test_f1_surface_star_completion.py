from __future__ import annotations

from backend.f1_surface_star_completion import complete_strong_surface_star_leaf


def _line(
    candidate_id: str,
    left: int,
    right: int,
    *,
    y: float,
    confidence: float = 0.8,
    transition: float = 0.9,
    gap: float = 2.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": left,
        "surface2_id": right,
        "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
        "confidence": confidence,
        "two_sided_transition_score": transition,
        "min_surface_gap": gap,
    }


def _star() -> list[dict]:
    return [
        _line("S1", 10, 20, y=0.0),
        _line("S2", 10, 30, y=15.0),
        _line("S3", 10, 40, y=30.0),
    ]


def test_completes_strong_new_leaf_on_established_surface_star() -> None:
    selected = _star()
    candidate = _line("S4", 10, 50, y=45.0)

    result = complete_strong_surface_star_leaf(selected, selected + [candidate])

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == ["S1", "S2", "S3", "S4"]
    assert result["added"][0]["candidate_source"] == "strong_surface_star_leaf_completion"


def test_rejects_duplicate_outline_even_when_it_adds_a_surface() -> None:
    selected = _star()
    duplicate = _line("S4", 10, 50, y=2.0)

    result = complete_strong_surface_star_leaf(selected, selected + [duplicate])

    assert result["activated"] is False


def test_rejects_leaf_without_established_hub_or_strong_evidence() -> None:
    selected = _star()[:2]
    weak = _line("S4", 10, 50, y=45.0, transition=0.79)

    result = complete_strong_surface_star_leaf(selected, selected + [weak])

    assert result["activated"] is False

