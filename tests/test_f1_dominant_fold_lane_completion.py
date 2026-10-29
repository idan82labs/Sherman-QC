from backend.f1_dominant_fold_lane_completion import complete_dominant_fold_lane


def _line(identifier, start, end, *, sources=3, members=3, normal_turn=70.0):
    return {
        "bend_id": identifier,
        "candidate_id": identifier,
        "span_endpoints": [start, end],
        "cluster_source_count": sources,
        "cluster_member_count": members,
        "cluster_max_finite_support_fraction": 1.0,
        "cluster_max_two_sided_normal_angle_deg": normal_turn,
        "cluster_mean_median_line_distance_mm": 1.0,
    }


def test_dominant_fold_lane_selects_three_cores_and_best_terminal_per_side():
    lines = [
        _line("C1", [0, 0, 0], [80, 0, 0], sources=6, members=8),
        _line("C2", [0, 5, 0], [80, 5, 0], sources=5, members=6),
        _line("C3", [0, 10, 0], [80, 10, 0], sources=4, members=5),
        _line("N_TRUE", [-12, 0, 0], [-2, 0, 0], sources=2, normal_turn=82),
        _line("N_ALIAS", [-11, 2, 0], [-1, 2, 0], sources=4, normal_turn=45),
        _line("P_TRUE", [82, 10, 0], [92, 10, 0], sources=2, normal_turn=84),
        _line("P_ALIAS", [81, 7, 0], [91, 7, 0], sources=4, normal_turn=50),
    ]

    result = complete_dominant_fold_lane(lines)

    assert result["eligible"] is True
    assert set(result["core_ids"]) == {"C1", "C2", "C3"}
    assert set(result["terminal_ids"]) == {"N_TRUE", "P_TRUE"}
    assert len(result["selected"]) == 5


def test_dominant_fold_lane_fails_closed_without_both_terminal_sides():
    lines = [
        _line("C1", [0, 0, 0], [80, 0, 0], sources=6, members=8),
        _line("C2", [0, 5, 0], [80, 5, 0], sources=5, members=6),
        _line("C3", [0, 10, 0], [80, 10, 0], sources=4, members=5),
        _line("N", [-12, 0, 0], [-2, 0, 0], sources=2),
    ]

    result = complete_dominant_fold_lane(lines)

    assert result["eligible"] is False
    assert result["selected"] == []
