"""Prune raw outline residuals after an explicit short-seed topology completes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def prune_seed_completed_outline_residuals(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the consensus core and two supported terminals when uniquely present."""

    output = [dict(item) for item in selected]
    consensus = [
        item
        for item in output
        if str(item.get("selection_reason") or "") == "mesh_cross_source_consensus_snap"
    ]
    terminals = [
        item
        for item in output
        if str(item.get("selection_reason") or "") == "scan_supported_short_seed_completion"
    ]
    residuals = [item for item in output if item not in consensus and item not in terminals]
    if len(consensus) != 1 or len(terminals) != 2 or len(residuals) != 2:
        return {
            "selected": output,
            "pruned": [],
            "activated": False,
            "reason": "unique_consensus_two_terminal_contract_not_met",
        }
    pruned = []
    for item in residuals:
        removed = dict(item)
        removed["pruning_reason"] = "raw_outline_residual_after_consensus_two_terminal_completion"
        pruned.append(removed)
    return {
        "selected": [*consensus, *terminals],
        "pruned": pruned,
        "activated": True,
        "reason": "consensus_core_with_two_supported_terminal_seeds",
    }
