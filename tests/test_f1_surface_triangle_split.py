from __future__ import annotations

from backend.f1_surface_triangle_split import split_supported_surface_triangle


def _edge(
    candidate_id: str,
    left: int,
    right: int,
    *,
    y: float,
    span: float = 150.0,
    confidence: float,
    transition: float,
    gap: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": left,
        "surface2_id": right,
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "confidence": confidence,
        "two_sided_transition_score": transition,
        "min_surface_gap": gap,
    }


def test_splits_supported_thin_surface_triangle() -> None:
    center = _edge("CENTER", 1, 2, y=0.0, confidence=0.93, transition=0.61, gap=0.1)
    side_a = _edge("SIDE_A", 1, 3, y=6.0, confidence=0.82, transition=0.39, gap=11.0)
    side_b = _edge("SIDE_B", 2, 3, y=11.0, confidence=0.99, transition=0.12, gap=0.3)

    result = split_supported_surface_triangle([center], [center, side_a, side_b])

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == ["SIDE_A", "SIDE_B"]
    assert len(result["removed"]) == 1


def test_rejects_triangle_with_mismatched_side_span() -> None:
    center = _edge("CENTER", 1, 2, y=0.0, confidence=0.93, transition=0.61, gap=0.1)
    side_a = _edge("SIDE_A", 1, 3, y=6.0, confidence=0.82, transition=0.39, gap=11.0)
    side_b = _edge(
        "SIDE_B", 2, 3, y=11.0, span=100.0, confidence=0.99, transition=0.12, gap=0.3
    )

    assert split_supported_surface_triangle([center], [center, side_a, side_b])["activated"] is False


def test_rejects_triangle_whose_sides_are_stronger_than_center() -> None:
    center = _edge("CENTER", 1, 2, y=0.0, confidence=0.93, transition=0.61, gap=0.1)
    side_a = _edge("SIDE_A", 1, 3, y=6.0, confidence=0.82, transition=0.70, gap=11.0)
    side_b = _edge("SIDE_B", 2, 3, y=11.0, confidence=0.99, transition=0.12, gap=0.3)

    assert split_supported_surface_triangle([center], [center, side_a, side_b])["activated"] is False

