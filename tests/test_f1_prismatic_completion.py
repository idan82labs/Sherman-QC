from backend.f1_partial_lattice_completion import complete_partial_lattice
from backend.f1_prismatic_junction_completion import complete_prismatic_junction
from backend.f1_prismatic_transition_pruning import prune_weak_prismatic_transitions


def _line(identifier, y, span=100, **extra):
    return {"bend_id": identifier, "span_endpoints": [[0, y, 0], [span, y, 0]], **extra}


def test_prunes_prismatic_strip_with_weak_matched_transition():
    selected = [_line("S1", 0)]
    strips = [{"strip_id": "S1", "strip_family": "prismatic_plane_lattice_long"}]
    pool = [_line("LO1", 0, cluster_max_two_sided_transition_score=.5)]
    result = prune_weak_prismatic_transitions(selected, {"profile_family": "prismatic_plane_lattice_completion"}, strips, pool)
    assert [item["bend_id"] for item in result["pruned"]] == ["S1"]


def test_completes_two_endpoint_connected_prismatic_junction_lanes():
    selected = [_line("S1", 0, span=100), {"bend_id": "S2", "span_endpoints": [[100, 0, 0], [100, 80, 0]]}, _line("IBC1", 30, span=20, candidate_source="interface_birth_consensus")]
    strips = [
        {"strip_id": "S1", "strip_family": "prismatic_plane_lattice_long"},
        {"strip_id": "S2", "strip_family": "prismatic_plane_lattice_long"},
    ]
    common = dict(cluster_sources=["cylindrical_standard", "cylindrical_wide"], cluster_source_count=2, cluster_max_finite_support_longest_run_fraction=1, cluster_max_endpoint_support_fraction=1, cluster_max_two_sided_transition_score=.7, tight_to_wide_normal_second_ratio=.1, cluster_max_two_sided_normal_coherence=.8, cluster_mean_median_line_distance_mm=1)
    candidates = [
        {"bend_id": "LO1", "span_endpoints": [[50, 80, 0], [100, 80, 0]], **common},
        {"bend_id": "LO2", "span_endpoints": [[100, 0, 0], [140, 0, 0]], **common},
    ]
    result = complete_prismatic_junction(selected, {"profile_family": "prismatic_plane_lattice_completion"}, strips, candidates)
    assert len(result["promoted"]) == 2


def test_partial_lattice_fails_closed_without_unique_strip_pair():
    result = complete_partial_lattice([], {"route": "patch_graph", "profile_family": "prismatic_plane_lattice_completion"}, [], [])
    assert result["promoted"] == []
