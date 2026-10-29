from scripts.select_f1_mesh_route_v12 import repair_bounded_cylindrical_representatives


def _row(candidate_id: str, source: str, sources: list[str], y: float, transition: float = 0.9) -> dict:
    return {
        "candidate_id": candidate_id,
        "representative_source": source,
        "cluster_sources": sources,
        "cluster_source_count": len(sources),
        "two_sided_transition_score": transition,
        "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
    }


def test_repairs_cylindrical_only_axis_to_nearby_mesh_consensus() -> None:
    rows = [
        _row("cyl", "cylindrical_standard", ["cylindrical_standard", "cylindrical_wide"], 0.0),
        _row("mesh", "scan_mesh_adjacency", ["cylindrical_standard", "cylindrical_wide", "scan_mesh_adjacency"], 4.0),
    ]
    selected, changes = repair_bounded_cylindrical_representatives(rows, [0])
    assert selected[0]["span_endpoints"] == [[0.0, 2.0, 0.0], [100.0, 2.0, 0.0]]
    assert selected[0]["representative_source"] == "source_consensus_midpoint"
    assert changes[0]["from_candidate_id"] == "cyl"
    assert changes[0]["toward_candidate_id"] == "mesh"


def test_does_not_repair_non_cylindrical_selection() -> None:
    rows = [
        _row("plane", "plane_lattice", ["plane_lattice", "cylindrical_standard"], 0.0),
        _row("mesh", "scan_mesh_adjacency", ["cylindrical_standard", "cylindrical_wide", "scan_mesh_adjacency"], 2.0),
    ]
    repaired, changes = repair_bounded_cylindrical_representatives(rows, [0])
    assert repaired == [rows[0]]
    assert changes == []


def test_requires_strong_nearby_multisource_mesh_evidence() -> None:
    selected = _row("cyl", "cylindrical_standard", ["cylindrical_standard", "cylindrical_wide"], 0.0)
    weak = _row("weak", "scan_mesh_adjacency", ["cylindrical_standard", "cylindrical_wide", "scan_mesh_adjacency"], 2.0, 0.84)
    far = _row("far", "scan_mesh_adjacency", ["cylindrical_standard", "cylindrical_wide", "scan_mesh_adjacency"], 5.1)
    repaired, changes = repair_bounded_cylindrical_representatives([selected, weak, far], [0])
    assert repaired == [selected]
    assert changes == []
