from backend.f1_patch_graph_endpoint_frame_completion import complete_patch_graph_endpoint_frame
from backend.f1_rolled_local_bend_completion import complete_rolled_local_bends
from backend.f1_rolled_long_strip_pruning import prune_rolled_long_strips


def _line(identifier, y, span=100, **extra):
    return {"bend_id": identifier, "span_endpoints": [[0, y, 0], [span, y, 0]], **extra}


def _vertical(identifier, x, span=150, **extra):
    return {"bend_id": identifier, "span_endpoints": [[x, 0, 0], [x, span, 0]], **extra}


def test_completes_unique_two_rail_endpoint_frame():
    selected = [_line("OB1", 0, span=100), _line("OB2", 100, span=95), _line("OB3", 50, span=30)]
    connectors = [
        _vertical("LO1", 0, representative_source="endpoint_connector", cluster_max_finite_support_longest_run_fraction=1, cluster_max_endpoint_support_fraction=1),
        _vertical("LO2", 100, representative_source="endpoint_connector", cluster_max_finite_support_longest_run_fraction=1, cluster_max_endpoint_support_fraction=1),
    ]
    result = complete_patch_graph_endpoint_frame(selected, {"route": "patch_graph"}, connectors)
    assert len(result["promoted"]) == 2
    assert [item["bend_id"] for item in result["pruned"]] == ["OB3"]


def test_frame_completion_fails_closed_with_more_than_two_dominant_rails():
    selected = [_line("OB1", 0), _line("OB2", 50), _line("OB3", 100)]
    assert complete_patch_graph_endpoint_frame(selected, {"route": "patch_graph"}, [])["promoted"] == []


def test_prunes_rolled_strip_longer_than_ten_support_widths():
    selected = [_line("S1", 0, span=101), _line("S2", 20, span=80)]
    strips = [
        {"strip_id": "S1", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 10},
        {"strip_id": "S2", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 10},
    ]
    result = prune_rolled_long_strips(selected, {"profile_family": "rolled_or_mixed_transition"}, strips)
    assert [item["bend_id"] for item in result["pruned"]] == ["S1"]


def test_completes_two_nms_separated_local_rolled_lanes():
    selected = [_line("S1", 0, span=20)]
    strips = [{"strip_id": "S1", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 5}]
    common = dict(
        cluster_source_count=2,
        cluster_max_finite_support_longest_run_fraction=1,
        cluster_max_endpoint_support_fraction=1,
        cluster_max_two_sided_transition_score=.7,
        cluster_max_two_sided_normal_coherence=.8,
        cluster_mean_median_line_distance_mm=1,
    )
    candidates = [_line("LO1", 20, span=25, **common), _line("LO2", 23, span=25, **common), _line("LO3", 35, span=20, **common)]
    result = complete_rolled_local_bends(selected, {"profile_family": "rolled_or_mixed_transition"}, strips, candidates)
    assert len(result["promoted"]) == 2
