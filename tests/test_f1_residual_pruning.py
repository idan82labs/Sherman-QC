from backend.f1_boundary_truncated_residual_pruning import prune_boundary_truncated_residuals
from backend.f1_rolled_parallel_doublet_pruning import prune_rolled_parallel_doublet


def _line(identifier, y, span=100, **extra):
    return {"bend_id": identifier, "span_endpoints": [[0, y, 0], [span, y, 0]], **extra}


def test_prunes_only_short_weak_boundary_truncated_patch_graph_objects():
    selected = [
        _line("OB1", 0, span=200, visibility_state="internal", support_mass=20_000),
        _line("OB2", 50, span=180, visibility_state="internal", support_mass=18_000),
        _line("OB3", 100, span=20, visibility_state="boundary_truncated", support_mass=1_000),
        _line("OB4", 150, span=80, visibility_state="boundary_truncated", support_mass=1_000),
    ]
    result = prune_boundary_truncated_residuals(selected, {"route": "patch_graph"})
    assert [item["bend_id"] for item in result["pruned"]] == ["OB3"]
    assert [item["bend_id"] for item in result["selected"]] == ["OB1", "OB2", "OB4"]


def test_boundary_pruning_fails_closed_off_patch_graph_route():
    selected = [_line("S1", 0, visibility_state="boundary_truncated", support_mass=1)]
    assert prune_boundary_truncated_residuals(selected, {"route": "zero_shot"})["pruned"] == []


def test_prunes_unique_rolled_parallel_support_width_doublet():
    selected = [_line("S1", 0), _line("S2", 9), _line("S3", 40)]
    strips = [
        {"strip_id": "S1", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 8},
        {"strip_id": "S2", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 8},
        {"strip_id": "S3", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 8},
    ]
    result = prune_rolled_parallel_doublet(selected, {"profile_family": "rolled_or_mixed_transition"}, strips)
    assert [item["bend_id"] for item in result["pruned"]] == ["S1", "S2"]
    assert [item["bend_id"] for item in result["selected"]] == ["S3"]


def test_rolled_doublet_pruning_fails_closed_when_pair_is_not_unique():
    selected = [_line("S1", 0), _line("S2", 9), _line("S3", 18)]
    strips = [
        {"strip_id": item["bend_id"], "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 8}
        for item in selected
    ]
    assert prune_rolled_parallel_doublet(selected, {"profile_family": "rolled_or_mixed_transition"}, strips)["pruned"] == []
