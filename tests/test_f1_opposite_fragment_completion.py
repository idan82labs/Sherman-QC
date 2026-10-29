from __future__ import annotations

from backend.f1_opposite_fragment_completion import complete_opposite_fragment_pair


def _line(candidate_id: str, *, y: float, span: float, transition: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "two_sided_transition_score": transition,
        "finite_support_fraction": 1.0,
        "endpoint_support_fraction": 1.0,
    }


def test_completes_strong_opposite_fragment() -> None:
    seed = _line("SEED", y=0.0, span=70.0, transition=0.97)
    seed["candidate_source"] = "overextended_mesh_fragment_replacement"
    weaker = _line("WEAKER", y=175.0, span=72.0, transition=0.88)
    stronger = _line("STRONGER", y=178.0, span=78.0, transition=0.92)

    result = complete_opposite_fragment_pair([seed], [weaker, stronger], [])

    assert result["activated"] is True
    assert result["added"][0]["candidate_id"] == "STRONGER"


def test_rejects_nearby_or_nonparallel_fragment() -> None:
    seed = _line("SEED", y=0.0, span=70.0, transition=0.97)
    seed["candidate_source"] = "overextended_mesh_fragment_replacement"
    nearby = _line("NEAR", y=50.0, span=75.0, transition=0.95)
    rotated = _line("ROTATED", y=175.0, span=75.0, transition=0.95)
    rotated["span_endpoints"] = [[0.0, 140.0, 0.0], [0.0, 210.0, 0.0]]

    assert complete_opposite_fragment_pair([seed], [nearby, rotated], [])["activated"] is False


def test_requires_a_cropped_mesh_fragment_seed() -> None:
    seed = _line("SEED", y=0.0, span=70.0, transition=0.97)
    seed["candidate_source"] = "scan_mesh_adjacency"
    opposite = _line("OPPOSITE", y=175.0, span=75.0, transition=0.95)

    assert complete_opposite_fragment_pair([seed], [opposite], [])["activated"] is False

