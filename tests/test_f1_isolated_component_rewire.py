from __future__ import annotations

from backend.f1_isolated_component_rewire import rewire_isolated_short_component


def _edge(candidate_id: str, left: int, right: int, *, x: float, span: float, angle: float = 95.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "scan_mesh_adjacency",
        "surface1_id": left,
        "surface2_id": right,
        "span_endpoints": [[x, 0.0, 0.0], [x, span, 0.0]],
        "span_mm": span,
        "bend_angle": angle,
        "confidence": 0.8,
    }


def test_rewires_short_island_into_main_component() -> None:
    main = [
        _edge("M1", 1, 2, x=0.0, span=80.0),
        _edge("M2", 2, 3, x=20.0, span=80.0),
    ]
    island = _edge("BAD", 10, 11, x=50.0, span=20.0)
    replacement = _edge("GOOD", 1, 11, x=70.0, span=18.0)

    result = rewire_isolated_short_component(main + [island], main + [island, replacement])

    assert result["activated"] is True
    assert {item["candidate_id"] for item in result["selected"]} == {"M1", "M2", "GOOD"}


def test_preserves_island_without_candidate_joining_main_component() -> None:
    main = [
        _edge("M1", 1, 2, x=0.0, span=80.0),
        _edge("M2", 2, 3, x=20.0, span=80.0),
    ]
    island = _edge("BAD", 10, 11, x=50.0, span=20.0)
    replacement = _edge("NOPE", 11, 12, x=70.0, span=18.0)

    assert rewire_isolated_short_component(
        main + [island], main + [island, replacement]
    )["activated"] is False


def test_preserves_nonisolated_short_edge() -> None:
    selected = [
        _edge("M1", 1, 2, x=0.0, span=80.0),
        _edge("M2", 2, 3, x=20.0, span=80.0),
        _edge("SHORT", 3, 4, x=40.0, span=20.0),
    ]
    replacement = _edge("NOPE", 1, 4, x=70.0, span=18.0)

    assert rewire_isolated_short_component(selected, selected + [replacement])["activated"] is False

