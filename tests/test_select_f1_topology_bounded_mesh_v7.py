from scripts.select_f1_topology_bounded_mesh_v7 import select_bounded_mesh


def _line(candidate_id, y, *, span=300.0, transition=0.6, source="scan_mesh_adjacency"):
    return {
        "candidate_id": candidate_id,
        "representative_source": source,
        "cluster_sources": [source],
        "span_endpoints": [[-span / 2.0, y, 0.0], [span / 2.0, y, 0.0]],
        "span_mm": span,
        "two_sided_transition_score": transition,
    }


def test_bounded_mesh_coverage_suppresses_only_long_parallel_aliases():
    rows = [
        _line("preferred", 0.0, transition=0.8),
        _line("long-alias", 9.0, transition=0.5),
        _line("other-physical-axis", 44.0, transition=0.7),
        _line("short-neighbor-a", 2.0, span=30.0, transition=0.6),
        _line("short-neighbor-b", 4.0, span=30.0, transition=0.5),
    ]
    selected = select_bounded_mesh(rows, 330.0)
    assert selected == [0, 2, 3, 4]


def test_bounded_mesh_coverage_keeps_strong_plane_candidate_without_mesh_cluster():
    rows = [
        _line("strong-plane", 0.0, source="plane_lattice", transition=0.7),
        _line("weak-plane", 40.0, source="plane_lattice", transition=0.49),
    ]
    assert select_bounded_mesh(rows, 330.0) == [0]
