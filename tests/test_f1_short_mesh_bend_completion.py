from backend.f1_short_mesh_bend_completion import complete_short_scan_mesh_bends


def _line(
    bend_id,
    start,
    end,
    *,
    sources=("scan_mesh_adjacency",),
    turn=70.0,
    support=0.95,
):
    return {
        "bend_id": bend_id,
        "candidate_id": f"scan_mesh_adjacency::{bend_id}",
        "span_endpoints": [start, end],
        "cluster_sources": list(sources),
        "cluster_max_two_sided_normal_angle_deg": turn,
        "cluster_max_finite_support_fraction": support,
    }


def test_adds_only_supported_short_mesh_axis_and_deduplicates():
    selected = [_line("kept", [0, 0, 0], [10, 0, 0])]
    candidates = [
        _line("duplicate", [0, 1, 0], [10, 1, 0]),
        _line("short", [20, 0, 0], [32, 0, 0]),
        _line("long", [40, 0, 0], [70, 0, 0]),
        _line("weak_turn", [80, 0, 0], [90, 0, 0], turn=59.9),
        _line("weak_support", [100, 0, 0], [110, 0, 0], support=0.79),
        _line("mixed_source", [120, 0, 0], [130, 0, 0], sources=("scan_mesh_adjacency", "other")),
    ]
    result = complete_short_scan_mesh_bends(
        selected,
        candidates,
        maximum_span_mm=16.0,
        minimum_normal_turn_deg=60.0,
        minimum_finite_support_fraction=0.8,
    )
    assert result["added_ids"] == ["short"]
    assert [item["bend_id"] for item in result["selected"]] == ["kept", "short"]
    assert result["selected"][1]["selection_reason"] == "short_scan_mesh_bend_completion"


def test_nonpositive_span_cap_disables_completion():
    selected = [_line("kept", [0, 0, 0], [10, 0, 0])]
    result = complete_short_scan_mesh_bends(
        selected,
        [_line("candidate", [20, 0, 0], [30, 0, 0])],
        maximum_span_mm=0.0,
        minimum_normal_turn_deg=60.0,
        minimum_finite_support_fraction=0.8,
    )
    assert result == {"enabled": False, "added_ids": [], "selected": selected}


def test_output_is_deterministic_under_input_order():
    left = _line("left", [20, 0, 0], [30, 0, 0], turn=75.0)
    right = _line("right", [40, 0, 0], [50, 0, 0], turn=80.0)
    kwargs = {
        "maximum_span_mm": 16.0,
        "minimum_normal_turn_deg": 60.0,
        "minimum_finite_support_fraction": 0.8,
    }
    forward = complete_short_scan_mesh_bends([], [left, right], **kwargs)
    reverse = complete_short_scan_mesh_bends([], [right, left], **kwargs)
    assert forward == reverse
