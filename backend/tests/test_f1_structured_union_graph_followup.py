from __future__ import annotations

import json
from pathlib import Path

from backend import f1_structured_union_graph as union


ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "tests/fixtures/f1_structured_union/compact_incidence_pool.json"


def test_followup_graph_serializes_supported_and_residual_only_paths() -> None:
    graph = union.build_structured_union_graph(
        json.loads(POOL.read_text(encoding="utf-8"))
    )

    supported = [path for path in graph["latent_paths"] if path["incidence_complete"]]
    unsupported = [path for path in graph["latent_paths"] if not path["incidence_complete"]]
    assert supported and unsupported
    assert graph["panel_boundary_arcs"]
    assert graph["hard_duplicate_pairs"]
    assert all(
        path["label_eligibility"]
        == union.build_engineering_label_eligibility(path)
        for path in graph["latent_paths"]
    )
    assert all(
        eligibility == (label == union.RESIDUAL_NONCOUNTED)
        for path in unsupported
        for label, eligibility in path["label_eligibility"].items()
    )
    assert all(
        path["semantic_label"] == union.RESIDUAL_NONCOUNTED
        for path in graph["latent_paths"]
    )
    assert graph["invariants"]["topology_factors_mutate_labels"] is False

