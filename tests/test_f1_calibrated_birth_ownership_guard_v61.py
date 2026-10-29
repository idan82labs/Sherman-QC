import copy

from backend.f1_calibrated_birth_ownership_guard_v61 import compose_v61


def _line(bend_id: str, y: float, route: str) -> dict:
    return {"bend_id": bend_id, "line_start": [0.0, y, 0.0], "line_end": [10.0, y, 0.0], "selection_route": route}


def test_v61_suppresses_near_duplicate_and_preserves_incumbents() -> None:
    incumbent = _line("I1", 0.0, "v52")
    v52 = {"predictions": [incumbent], "v52_result_sha256": "v52"}
    v60 = {"predictions": [incumbent, _line("B1", 0.09, "calibrated_per_view_segment_consensus_v60")], "v60_result_sha256": "v60"}
    frozen = copy.deepcopy(v52["predictions"])
    result = compose_v61(v52, v60, B=10.0)
    assert result["predictions"] == frozen
    assert result["audit"]["ownership_suppressed_count"] == 1


def test_v61_admits_separate_birth() -> None:
    incumbent = _line("I1", 0.0, "v52")
    birth = _line("B1", 0.11, "calibrated_per_view_segment_consensus_v60")
    result = compose_v61({"predictions": [incumbent]}, {"predictions": [incumbent, birth]}, B=10.0)
    assert len(result["predictions"]) == 2
    assert result["predictions"][1]["selection_route"] == "calibrated_birth_ownership_guard_v61"
