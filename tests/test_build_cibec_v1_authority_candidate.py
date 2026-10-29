from __future__ import annotations

import json
from pathlib import Path

from research.cibec_v1.contract import CHARTER_FIELDS
from scripts.build_cibec_v1_authority_candidate import DECISIONS, candidate_outputs


def test_candidate_charter_has_exact_decision_set_without_signatures() -> None:
    assert tuple(DECISIONS) == CHARTER_FIELDS
    assert all(value and "UNDECIDED" not in value for value in DECISIONS.values())


def test_candidate_outputs_cover_exact_source_free_fixture_set() -> None:
    outputs = candidate_outputs()
    assert tuple(outputs) == tuple(f"S{index:02d}" for index in range(1, 15))
    assert all(row["case_status"] == "RESOLVED" for row in outputs.values())
    assert all(row["events"] for row in outputs.values())


def test_candidate_key_preserves_required_falsification_cases() -> None:
    outputs = candidate_outputs()
    assert len(outputs["S01"]["events"]) == 1
    assert len(outputs["S03"]["events"]) == 1
    assert len(outputs["S05"]["events"]) == 4
    assert len(outputs["S06"]["events"]) == 2
    assert outputs["S06"]["events"][0]["line_end_mm"][0] < outputs["S06"]["events"][1]["line_start_mm"][0]
    assert outputs["S13"]["events"][0]["visibility"] == "PARTIAL"
    assert outputs["S13"]["events"][0]["line_start_mm"][0] == 17.0
    assert outputs["S13"]["events"][0]["line_end_mm"][0] == 58.0
    assert outputs["S14"]["cad_overlay_status"] == "ABSTAIN"


def test_built_candidate_pack_is_explicitly_non_authoritative() -> None:
    target = Path("output/research/cibec-v1-authority-candidate-v2")
    if not target.exists():
        return
    manifest = json.loads((target / "manifest.json").read_text())
    charter = json.loads((target / "CIBEC_V1_CHARTER_CANDIDATE.json").read_text())
    assert manifest["physical_assets_opened"] == 0
    assert manifest["prediction_assets_opened"] == 0
    assert manifest["qualification_truth"] is False
    assert manifest["authority_status"] == "UNSIGNED"
    assert charter["status"] == "PROPOSED_NON_AUTHORITATIVE"
    assert charter["signatures"] == []


def test_candidate_v2_is_presentation_only_and_preserves_v1_review_payload() -> None:
    first = Path("output/research/cibec-v1-authority-candidate-v1")
    second = Path("output/research/cibec-v1-authority-candidate-v2")
    if not first.exists() or not second.exists():
        return
    for name in ("CIBEC_V1_CHARTER_CANDIDATE.json", "S01_S14_EXPECTATIONS_CANDIDATE.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest["review_payload_sha256"] == second_manifest["review_payload_sha256"]
