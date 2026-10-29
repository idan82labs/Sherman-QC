from backend.f1_cylindrical_consensus_recovery_v52 import compose_v52


def _base() -> dict:
    return {"predictions": [], "source_only": True, "research_only": True}


def _candidate(candidate_id: str, y: float, span: float = 80.0, source_count: int = 3) -> dict:
    return {
        "candidate_id": candidate_id,
        "bend_id": candidate_id,
        "span_endpoints": [[-span / 2, y, 0], [span / 2, y, 0]],
        "representative_source": "cylindrical_standard",
        "cluster_sources": ["cylindrical_standard", "cylindrical_wide", "scan_mesh_adjacency"],
        "cluster_source_count": source_count,
        "support_count": 50,
        "two_sided_transition_score": 0.9,
    }


def test_recovers_distinct_long_consensus_axis() -> None:
    result = compose_v52(_base(), {"scan_bbox_mm": [100, 30, 20], "line_objects": [_candidate("C1", 4)]})
    assert result["candidate_count"] == 1
    assert result["cylindrical_consensus_recovery_audit"]["recovered_candidate_count"] == 1


def test_rejects_short_or_weak_consensus_and_deduplicates_incumbent() -> None:
    incumbent = {"bend_id": "P1", "line_start": [-40, 4, 0], "line_end": [40, 4, 0]}
    result = compose_v52(
        {"predictions": [incumbent], "source_only": True, "research_only": True},
        {
            "scan_bbox_mm": [100, 30, 20],
            "line_objects": [
                _candidate("DUP", 4),
                _candidate("SHORT", 12, span=20),
                _candidate("WEAK", 20, source_count=2),
            ],
        },
    )
    assert result["candidate_count"] == 1
    assert result["cylindrical_consensus_recovery_audit"]["recovered_candidate_count"] == 0
