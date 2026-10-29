from backend.f1_parallel_width_lattice_completion import complete_parallel_width_lattice


def _line(identifier, y, span=240, **extra):
    return {"bend_id": identifier, "span_endpoints": [[0, y, 0], [span, y, 0]], **extra}


def test_promotes_unique_candidate_at_measured_width():
    selected = [_line("S1", 0), _line("S2", 50)]
    strips = [
        {"strip_id": "S1", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
        {"strip_id": "S2", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
    ]
    candidate = _line(
        "LO1", 12,
        cluster_source_count=2,
        cluster_sources=["cylindrical_standard", "cylindrical_wide"],
        cluster_max_finite_support_longest_run_fraction=1.0,
        cluster_max_endpoint_support_fraction=1.0,
        cluster_max_two_sided_transition_score=0.8,
        cluster_max_two_sided_normal_coherence=0.9,
        cluster_mean_median_line_distance_mm=1.0,
    )
    result = complete_parallel_width_lattice(selected, strips, [candidate])
    assert len(result["promoted"]) == 1
    assert result["promoted"][0]["source_strip_id"] == "S1"


def test_fails_closed_for_nonprismatic_or_bad_width_ratio():
    selected = [_line("S1", 0), _line("S2", 50)]
    strips = [
        {"strip_id": "S1", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 12.0},
        {"strip_id": "S2", "strip_family": "discrete_fold_strip", "cross_fold_width_mm": 12.0},
    ]
    assert complete_parallel_width_lattice(selected, strips, [])["promoted"] == []


def test_fails_closed_when_selected_topology_contains_non_strip_object():
    selected = [_line("S1", 0), _line("S2", 50), _line("IBC1", 100)]
    strips = [
        {"strip_id": "S1", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
        {"strip_id": "S2", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
    ]
    result = complete_parallel_width_lattice(selected, strips, [])
    assert result["promoted"] == []
    assert "selected_topology_contains_non_strip_objects" in result["activation_reasons"]


def test_fails_closed_for_short_aspect_prismatic_lattice():
    selected = [_line("S1", 0, span=120), _line("S2", 50, span=240)]
    strips = [
        {"strip_id": "S1", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
        {"strip_id": "S2", "strip_family": "prismatic_plane_lattice_long", "cross_fold_width_mm": 12.0},
    ]
    result = complete_parallel_width_lattice(selected, strips, [])
    assert result["promoted"] == []
    assert "selected_strip_aspect_ratio_below_18" in result["activation_reasons"]
