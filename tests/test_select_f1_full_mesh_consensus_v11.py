from scripts.select_f1_full_mesh_consensus_v11 import prune_short_cylindrical_fragments


def _row(candidate_id: str, span: float, sources: list[str]) -> dict:
    return {"candidate_id": candidate_id, "span_mm": span, "cluster_sources": sources}


def test_prunes_only_short_cylindrical_alias_fragment() -> None:
    rows = [
        _row("short", 29.0, ["cylindrical_standard", "cylindrical_wide"]),
        _row("a", 71.0, ["cylindrical_standard", "cylindrical_wide"]),
        _row("b", 75.0, ["plane_lattice", "cylindrical_standard"]),
        _row("c", 79.0, ["scan_mesh_adjacency"]),
        _row("d", 96.0, ["plane_lattice", "scan_mesh_adjacency"]),
    ]
    selected, changes = prune_short_cylindrical_fragments(rows, list(range(5)))
    assert selected == [1, 2, 3, 4]
    assert [row["candidate_id"] for row in changes] == ["short"]


def test_preserves_short_line_with_independent_non_cylindrical_support() -> None:
    rows = [
        _row("short-consensus", 20.0, ["cylindrical_standard", "plane_lattice"]),
        _row("a", 60.0, ["scan_mesh_adjacency"]),
        _row("b", 70.0, ["plane_lattice"]),
    ]
    selected, changes = prune_short_cylindrical_fragments(rows, [0, 1, 2])
    assert selected == [0, 1, 2]
    assert changes == []


def test_preserves_cylindrical_line_at_boundary() -> None:
    rows = [
        _row("boundary", 45.0, ["cylindrical_standard", "cylindrical_wide"]),
        _row("a", 100.0, ["plane_lattice"]),
        _row("b", 100.0, ["scan_mesh_adjacency"]),
    ]
    selected, changes = prune_short_cylindrical_fragments(rows, [0, 1, 2])
    assert selected == [0, 1, 2]
    assert changes == []
