from backend.f1_mesh_consensus_snap import snap_selected_mesh_consensus


def _line(candidate_id, y, *, sources=1, members=1):
    return {
        "bend_id": candidate_id,
        "candidate_id": candidate_id,
        "span_endpoints": [[0, y, 0], [20, y, 0]],
        "cluster_source_count": sources,
        "cluster_member_count": members,
        "cluster_mean_finite_support_fraction": 0.95,
        "cluster_mean_median_line_distance_mm": 1.0,
    }


def test_snaps_mesh_alias_to_multi_source_consensus():
    original = _line("scan_mesh_adjacency::SMA1", 0.0)
    consensus = _line("scan_mesh_adjacency::SMA2", 1.0, sources=3, members=4)
    result = snap_selected_mesh_consensus([original], [consensus])
    assert [item["candidate_id"] for item in result["selected"]] == [
        "scan_mesh_adjacency::SMA2"
    ]
    assert result["replacements"] == [
        {
            "original_candidate_id": "scan_mesh_adjacency::SMA1",
            "replacement_candidate_id": "scan_mesh_adjacency::SMA2",
        }
    ]


def test_does_not_snap_to_nonmesh_or_low_consensus_alias():
    original = _line("scan_mesh_adjacency::SMA1", 0.0)
    nonmesh = _line("cylindrical_standard::C1", 0.5, sources=5, members=5)
    weak = _line("scan_mesh_adjacency::SMA2", 1.0, sources=2, members=5)
    result = snap_selected_mesh_consensus([original], [nonmesh, weak])
    assert result["selected"] == [original]
    assert result["replacements"] == []


def test_does_not_replace_cross_source_selector_choice_with_mesh_alias():
    original = _line("cylindrical_standard::CB5", 0.0, sources=2, members=2)
    original["candidate_source"] = "cross_source_line_object"
    consensus = _line("scan_mesh_adjacency::SMA1", 1.0, sources=6, members=6)

    result = snap_selected_mesh_consensus([original], [consensus])

    assert result["selected"] == [original]
    assert result["replacements"] == []


def test_snap_is_deterministic_under_line_object_order():
    original = _line("scan_mesh_adjacency::SMA1", 0.0)
    left = _line("scan_mesh_adjacency::SMA2", 1.0, sources=3, members=4)
    right = _line("scan_mesh_adjacency::SMA3", 1.5, sources=3, members=5)
    forward = snap_selected_mesh_consensus([original], [left, right])
    reverse = snap_selected_mesh_consensus([original], [right, left])
    assert forward == reverse
