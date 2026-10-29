from __future__ import annotations

import json
from pathlib import Path

from scripts.build_exp_cad_0010_binding_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_actual_binding_is_complete_truth_denying_and_exact() -> None:
    pack = ROOT / "output/research/cad-line-ontology-authority-pack-v1"
    payload = build_manifest(
        slice_path=ROOT / "data/bend_corpus/evaluation_slices/cad_guided_gold13_physical_20260822.json",
        parent_summary_path=ROOT / "output/research/f1-cad-neutral-surface-v2-20260825/summary.json",
        parent_cases_root=ROOT / "output/research/f1-cad-neutral-surface-v2-20260825/cases",
        ontology_path=pack / "AI_CONSENSUS_ADJUDICATED_PROPOSAL_V2.json",
        acceptance_paths=[
            pack / "AI_CONSENSUS_REVIEW_A_V2_ACCEPTANCE.json",
            pack / "AI_CONSENSUS_REVIEW_B_V2_ACCEPTANCE.json",
            pack / "AI_CONSENSUS_REVIEW_C_V2_ACCEPTANCE.json",
        ],
    )
    assert payload["case_count"] == 13
    assert payload["ontology_summary"] == {
        "event_count": 63,
        "qc_bend_count": 60,
        "not_qc_bend_count": 3,
        "physical_event_group_count": 53,
    }
    assert payload["gold_or_expected_counts_present"] is False
    encoded = json.dumps(payload, sort_keys=True).lower()
    assert "expected_bend_count" not in encoded
    assert "part_key" not in encoded
    assert all(row["ontology_coverage"].startswith("EXACT") for row in payload["records"])
    assert sum(len(row["event_bindings"]) for row in payload["records"]) == 63


def test_c005_and_empty_parent_cases_are_bound_explicitly() -> None:
    pack = ROOT / "output/research/cad-line-ontology-authority-pack-v1"
    payload = build_manifest(
        slice_path=ROOT / "data/bend_corpus/evaluation_slices/cad_guided_gold13_physical_20260822.json",
        parent_summary_path=ROOT / "output/research/f1-cad-neutral-surface-v2-20260825/summary.json",
        parent_cases_root=ROOT / "output/research/f1-cad-neutral-surface-v2-20260825/cases",
        ontology_path=pack / "AI_CONSENSUS_ADJUDICATED_PROPOSAL_V2.json",
        acceptance_paths=[
            pack / "AI_CONSENSUS_REVIEW_A_V2_ACCEPTANCE.json",
            pack / "AI_CONSENSUS_REVIEW_B_V2_ACCEPTANCE.json",
            pack / "AI_CONSENSUS_REVIEW_C_V2_ACCEPTANCE.json",
        ],
    )
    by_parent = {row["parent_case_id"]: row for row in payload["records"]}
    assert {row["status"] for row in by_parent["C005"]["event_bindings"]} == {"NOT_QC_BEND"}
    assert by_parent["C006"]["accepted_parent_event_ids"] == []
    assert by_parent["C011"]["accepted_parent_event_ids"] == []
    assert by_parent["C006"]["event_bindings"] == []
    assert by_parent["C011"]["event_bindings"] == []
