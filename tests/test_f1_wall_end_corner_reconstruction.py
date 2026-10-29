from __future__ import annotations

from backend.f1_wall_end_corner_reconstruction import (
    _plane_signature,
    has_wall_end_corner_signature,
    reconstruct_wall_end_corners,
)


def _edge(candidate_id: str, left: str, right: str, span: float, transition: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "plane_lattice",
        "incident_plane_ids": [left, right],
        "span_mm": span,
        "span_endpoints": [[0.0, 0.0, 0.0], [span, 0.0, 0.0]],
        "two_sided_transition_score": transition,
    }


def _signature() -> list[dict]:
    return [
        _edge("BASE_A", "H", "A", 180.0, 0.75),
        _edge("BASE_B", "H", "B", 175.0, 0.70),
        _edge("FREE_A", "H", "X", 170.0, 0.40),
        _edge("FREE_B", "H", "Y", 170.0, 0.40),
        _edge("RETURN_A", "A", "C", 120.0, 0.65),
        _edge("RETURN_B", "B", "C", 120.0, 0.48),
    ]


def test_recognizes_unambiguous_wall_end_plane_graph() -> None:
    assert has_wall_end_corner_signature(_signature()) is True


def test_plane_signature_orders_base_lines_deterministically() -> None:
    forward = _plane_signature(_signature())
    reverse = _plane_signature(list(reversed(_signature())))

    assert forward is not None
    assert reverse is not None
    assert [item["candidate_id"] for item in forward["base_lines"]] == [
        "BASE_A",
        "BASE_B",
    ]
    assert [item["candidate_id"] for item in reverse["base_lines"]] == [
        "BASE_A",
        "BASE_B",
    ]


def test_rejects_ambiguous_return_strength() -> None:
    candidates = _signature()
    candidates[-1]["two_sided_transition_score"] = 0.60

    assert has_wall_end_corner_signature(candidates) is False


def test_requires_points_after_plane_signature_matches() -> None:
    result = reconstruct_wall_end_corners([], _signature(), None)

    assert result["activated"] is False
    assert result["reason"] == "scan_points_unavailable"
