from __future__ import annotations

from scripts.build_f1_retained_stage_graph_v1 import build


def test_retained_stage_graphs_bind_every_stage_and_preserve_lane_separation() -> None:
    payload = build()
    assert payload["status"] == "RESEARCH_ONLY_NOT_PROMOTION"
    scan = payload["scan_only"]
    cad = payload["cad_assisted"]
    assert len(scan["graph"]["stages"]) == len(scan["bindings"]) == 8
    assert len(cad["graph"]["stages"]) == len(cad["bindings"]) == 5
    assert set(scan["bindings"]).isdisjoint(cad["bindings"])
    assert all(row["sha256"] and len(row["sha256"]) == 64 for row in scan["bindings"].values())
    assert all(stage["output_schema"] == "F1_CANDIDATE_SET_V1" for stage in scan["graph"]["stages"] + cad["graph"]["stages"])
