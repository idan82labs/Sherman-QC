from __future__ import annotations

from backend.f1_paired_return_completion import complete_paired_return_axis


def _row(candidate_id, y, z, x0, x1, *, good=True):
    return {
        "candidate_id": candidate_id,
        "representative_source": "cylindrical_standard",
        "span_endpoints": [[x0, y, z], [x1, y, z]],
        "axis_direction": [1.0, 0.0, 0.0],
        "span_mm": x1 - x0,
        "finite_support_fraction": 1.0 if good else 0.2,
        "finite_support_longest_run_fraction": 1.0 if good else 0.2,
        "endpoint_support_fraction": 1.0 if good else 0.2,
        "two_sided_normal_angle_deg": 85.0,
        "two_sided_transition_score": 0.8,
    }


def test_recovers_stitched_counterpart_of_close_return_pair():
    selected = [
        _row("left_base", -190, 0, 0, 300),
        _row("left_return", -190, 16, 0, 280),
        _row("right_base", 190, 0, 0, 500),
    ]
    rows = [
        _row("f1", 190, -16, 0, 170),
        _row("f2", 191, -16, 180, 350),
        _row("f3", 192, -16, 340, 500),
    ]
    result = complete_paired_return_axis(selected, rows, 500.0)
    assert len(result["selected"]) == 4
    assert result["changes"][0]["repair_basis"] == "paired_return_profile_stitched_counterpart"


def test_requires_three_lane_return_configuration():
    selected = [_row("a", 0, 0, 0, 100), _row("b", 20, 0, 0, 100)]
    result = complete_paired_return_axis(selected, [], 500.0)
    assert result["selected"] == selected
    assert result["changes"] == []
