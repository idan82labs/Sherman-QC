from backend.f1_seed_completed_outline_pruning import (
    prune_seed_completed_outline_residuals,
)


def _candidate(candidate_id, reason=None):
    return {"candidate_id": candidate_id, "selection_reason": reason}


def test_keeps_unique_consensus_core_and_two_completed_terminals():
    candidates = [
        _candidate("core", "mesh_cross_source_consensus_snap"),
        _candidate("outline-a"),
        _candidate("outline-b"),
        _candidate("terminal-a", "scan_supported_short_seed_completion"),
        _candidate("terminal-b", "scan_supported_short_seed_completion"),
    ]

    result = prune_seed_completed_outline_residuals(candidates)

    assert result["activated"] is True
    assert [item["candidate_id"] for item in result["selected"]] == [
        "core",
        "terminal-a",
        "terminal-b",
    ]
    assert [item["candidate_id"] for item in result["pruned"]] == [
        "outline-a",
        "outline-b",
    ]


def test_abstains_without_exact_completion_signature():
    candidates = [
        _candidate("core", "mesh_cross_source_consensus_snap"),
        _candidate("outline"),
        _candidate("terminal", "scan_supported_short_seed_completion"),
    ]

    result = prune_seed_completed_outline_residuals(candidates)

    assert result["activated"] is False
    assert result["selected"] == candidates
