from __future__ import annotations

from backend.f1_formed_end_junction_reconstruction import reconstruct_formed_end_junction


def test_rejects_incomplete_formed_end_signature() -> None:
    result = reconstruct_formed_end_junction([], [], [])

    assert result["activated"] is False
    assert result["reason"] == "formed_end_source_signature_failed"


def test_rejects_twelve_owned_lines_without_three_strip_seeds() -> None:
    owned = [
        {
            "candidate_id": f"OB{index}",
            "candidate_source": "strict_blind_owned",
            "span_endpoints": [[0.0, float(index), 0.0], [100.0, float(index), 0.0]],
        }
        for index in range(12)
    ]

    assert reconstruct_formed_end_junction([], owned, [])["activated"] is False

