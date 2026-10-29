from __future__ import annotations

from backend.f1_terminal_cycle_completion import complete_terminal_surface_cycles


def _edge(
    candidate_id: str,
    left: int,
    right: int,
    *,
    y: float,
    span: float,
    confidence: float = 0.99,
    gap: float = 0.2,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": left,
        "surface2_id": right,
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "confidence": confidence,
        "min_surface_gap": gap,
    }


def test_completes_short_edge_opposite_selected_long_edge_in_cycle() -> None:
    central = _edge("LONG", 1, 2, y=0.0, span=150.0)
    target = _edge("SHORT", 3, 4, y=30.0, span=25.0)
    candidates = [
        central,
        target,
        _edge("LEFT", 1, 3, y=60.0, span=20.0),
        _edge("RIGHT", 2, 4, y=90.0, span=20.0),
    ]

    result = complete_terminal_surface_cycles([central], candidates)

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == ["LONG", "SHORT"]
    assert result["added"][0]["candidate_source"] == "terminal_surface_cycle_completion"


def test_rejects_open_cycle() -> None:
    central = _edge("LONG", 1, 2, y=0.0, span=150.0)
    target = _edge("SHORT", 3, 4, y=30.0, span=25.0)
    candidates = [central, target, _edge("LEFT", 1, 3, y=60.0, span=20.0)]

    assert complete_terminal_surface_cycles([central], candidates)["activated"] is False


def test_rejects_weak_or_nonterminal_opposite_edge() -> None:
    central = _edge("LONG", 1, 2, y=0.0, span=150.0)
    weak = _edge("SHORT", 3, 4, y=30.0, span=25.0, confidence=0.90)
    candidates = [
        central,
        weak,
        _edge("LEFT", 1, 3, y=60.0, span=20.0),
        _edge("RIGHT", 2, 4, y=90.0, span=20.0),
    ]

    assert complete_terminal_surface_cycles([central], candidates)["activated"] is False


def test_completes_moderate_edge_when_entire_other_cycle_is_selected() -> None:
    central = _edge("LONG", 1, 2, y=0.0, span=150.0)
    left = _edge("LEFT", 1, 3, y=60.0, span=20.0)
    right = _edge("RIGHT", 2, 4, y=90.0, span=20.0)
    target = _edge("SHORT", 3, 4, y=30.0, span=25.0, confidence=0.65, gap=5.5)

    result = complete_terminal_surface_cycles(
        [central, left, right],
        [central, left, right, target],
    )

    assert result["activated"] is True
    assert result["evidence"][0]["evidence_mode"] == "fully_selected_cycle"


def test_rejects_moderate_edge_when_connector_is_not_selected() -> None:
    central = _edge("LONG", 1, 2, y=0.0, span=150.0)
    left = _edge("LEFT", 1, 3, y=60.0, span=20.0)
    right = _edge("RIGHT", 2, 4, y=90.0, span=20.0)
    target = _edge("SHORT", 3, 4, y=30.0, span=25.0, confidence=0.65, gap=5.5)

    result = complete_terminal_surface_cycles(
        [central, left],
        [central, left, right, target],
    )

    assert result["activated"] is False
