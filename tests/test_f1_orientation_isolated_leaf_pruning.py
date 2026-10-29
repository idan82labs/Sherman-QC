from __future__ import annotations

from backend.f1_orientation_isolated_leaf_pruning import prune_orientation_isolated_weak_leaf


def _line(candidate_id: str, leaf: int, *, angle_axis: str, endpoint_support: float = 0.6) -> dict:
    endpoint = [30.0, 0.0, 0.0] if angle_axis == "x" else [0.0, 30.0, 0.0]
    return {
        "candidate_id": candidate_id,
        "surface1_id": 1,
        "surface2_id": leaf,
        "span_endpoints": [[0.0, 0.0, 0.0], endpoint],
        "span_mm": 30.0,
        "two_sided_transition_score": 0.55,
        "finite_support_fraction": 0.92,
        "endpoint_support_fraction": endpoint_support,
    }


def test_prunes_only_weak_directionally_isolated_four_spoke_leaf() -> None:
    selected = [
        _line("A", 2, angle_axis="x", endpoint_support=0.9),
        _line("B", 3, angle_axis="x", endpoint_support=0.9),
        _line("C", 4, angle_axis="x", endpoint_support=0.9),
        _line("OUTLINE", 5, angle_axis="y"),
    ]

    result = prune_orientation_isolated_weak_leaf(selected)

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["pruned"]] == ["OUTLINE"]


def test_preserves_isolated_leaf_with_complete_endpoint_support() -> None:
    selected = [
        _line("A", 2, angle_axis="x", endpoint_support=0.9),
        _line("B", 3, angle_axis="x", endpoint_support=0.9),
        _line("C", 4, angle_axis="x", endpoint_support=0.9),
        _line("REAL", 5, angle_axis="y", endpoint_support=0.9),
    ]

    assert prune_orientation_isolated_weak_leaf(selected)["activated"] is False


def test_preserves_leaf_on_larger_surface_star() -> None:
    selected = [
        _line("A", 2, angle_axis="x", endpoint_support=0.9),
        _line("B", 3, angle_axis="x", endpoint_support=0.9),
        _line("C", 4, angle_axis="x", endpoint_support=0.9),
        _line("D", 5, angle_axis="x", endpoint_support=0.9),
        _line("REAL", 6, angle_axis="y"),
    ]

    assert prune_orientation_isolated_weak_leaf(selected)["activated"] is False

