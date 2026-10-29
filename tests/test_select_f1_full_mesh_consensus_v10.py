from scripts.select_f1_full_mesh_consensus_v10 import repair_full_mesh_representatives


def _row(candidate_id: str, y: float, angle: float, *, source: str, source_count: int) -> dict:
    import math

    return {
        "candidate_id": candidate_id,
        "representative_source": source,
        "cluster_source_count": source_count,
        "span_endpoints": [
            [0.0, y, 0.0],
            [40.0 * math.cos(math.radians(angle)), y, 40.0 * math.sin(math.radians(angle))],
        ],
    }


def test_replaces_singleton_mesh_axis_with_nearby_consensus() -> None:
    rows = [
        _row("mesh", 0.0, 10.0, source="scan_mesh_adjacency", source_count=1),
        _row("consensus", 1.0, 0.0, source="cylindrical_standard", source_count=5),
    ]
    selected, changes = repair_full_mesh_representatives(rows, [0])
    assert selected == [1]
    assert changes == [{"from_candidate_id": "mesh", "to_candidate_id": "consensus"}]


def test_fails_closed_when_consensus_is_not_nearby() -> None:
    rows = [
        _row("mesh", 0.0, 0.0, source="scan_mesh_adjacency", source_count=1),
        _row("far", 8.0, 0.0, source="cylindrical_standard", source_count=5),
    ]
    selected, changes = repair_full_mesh_representatives(rows, [0])
    assert selected == [0]
    assert changes == []


def test_never_replaces_non_mesh_or_existing_consensus() -> None:
    rows = [
        _row("cylinder", 0.0, 0.0, source="cylindrical_standard", source_count=1),
        _row("mesh-consensus", 0.0, 0.0, source="scan_mesh_adjacency", source_count=3),
        _row("other", 1.0, 0.0, source="cylindrical_standard", source_count=5),
    ]
    selected, changes = repair_full_mesh_representatives(rows, [0, 1])
    assert selected == [0, 1]
    assert changes == []
