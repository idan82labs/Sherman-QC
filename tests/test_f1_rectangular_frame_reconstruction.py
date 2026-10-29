from __future__ import annotations

from backend.f1_rectangular_frame_reconstruction import reconstruct_rectangular_frame


def _line(candidate_id: str, start: list[float], end: list[float]) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_source": "strict_blind_owned",
        "span_endpoints": [start, end],
    }


def _frame() -> list[dict]:
    return [
        _line("L1", [-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]),
        _line("L2", [-100.0, 30.0, 0.0], [100.0, 30.0, 0.0]),
        _line("L3", [-100.0, 300.0, 0.0], [100.0, 300.0, 0.0]),
        _line("L4", [-60.0, 330.0, 0.0], [60.0, 330.0, 0.0]),
        _line("C1", [-100.0, 0.0, 0.0], [-100.0, 0.0, 20.0]),
        _line("C2", [-100.0, 300.0, 0.0], [-100.0, 300.0, 20.0]),
        _line("C3", [100.0, 0.0, 0.0], [100.0, 0.0, 20.0]),
        _line("C4", [100.0, 300.0, 0.0], [100.0, 300.0, 20.0]),
    ]


def test_reconstructs_four_rails_and_four_connectors() -> None:
    result = reconstruct_rectangular_frame([], _frame())

    assert result["activated"] is True
    assert len(result["selected"]) == 8
    assert sum(
        item["candidate_source"] == "rectangular_frame_connector_reconstruction"
        for item in result["selected"]
    ) == 4
    extended = next(item for item in result["selected"] if item["candidate_id"] == "L4")
    assert round(extended["span_mm"], 6) == 200.0


def test_rejects_incomplete_owned_fragment_set() -> None:
    assert reconstruct_rectangular_frame([], _frame()[:-1])["activated"] is False


def test_rejects_nonorthogonal_corner_fragments() -> None:
    candidates = _frame()
    for item in candidates[4:]:
        start = item["span_endpoints"][0]
        item["span_endpoints"][1] = [start[0] + 20.0, start[1], start[2]]

    assert reconstruct_rectangular_frame([], candidates)["activated"] is False
