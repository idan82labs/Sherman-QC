from __future__ import annotations

import json
import ast
from pathlib import Path

import pytest

from apps.api.services.bend_readiness_service import load_bend_inspection_readiness
from domains.bend.services.promotion_readiness import project_program_status


def test_missing_or_invalid_status_fails_closed(tmp_path):
    missing = load_bend_inspection_readiness(tmp_path / "missing.json")
    assert missing["status"] == "UNAVAILABLE_FAIL_CLOSED"
    assert missing["production_use_allowed"] is False
    assert missing["operating_mode"] == "RESEARCH_ONLY"


def test_unpromoted_status_is_explicitly_research_only(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "production_promotion_authorized": False,
        "physical_generalization_progress": False,
        "active_blockers": ["NEEDS_UNTOUCHED_CORPUS"],
        "generalization_progress_indicator": {"cad_assisted": "GPI-C"},
        "lane_status": {"cad_assisted": {"grade": "C"}},
        "contract_soak": {"status": "PASS"},
    }))
    result = load_bend_inspection_readiness(path)
    assert result["status"] == "PROMOTION_BLOCKED"
    assert result["active_blockers"] == ["NEEDS_UNTOUCHED_CORPUS"]
    assert result["production_use_allowed"] is False
    assert len(result["program_status_sha256"]) == 64


def test_hash_shaped_declarations_never_open_production(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "production_promotion_authorized": True,
        "physical_generalization_progress": True,
        "promotion_evidence": {
            "status": "PASS",
            "candidate_lock_sha256": "a" * 64,
            "qualification_report_sha256": "b" * 64,
            "dual_review_truth_sha256": "c" * 64,
        },
    }))
    result = load_bend_inspection_readiness(path)
    assert result["status"] == "PROMOTION_EVIDENCE_INVALID_FAIL_CLOSED"
    assert result["production_use_allowed"] is False
    assert result["evidence_verification"] == "NOT_PERFORMED"
    assert result["autonomous_disposition_status"] == "NOT_AUTHORIZED"


def test_bare_promotion_boolean_fails_closed(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2",
        "production_promotion_authorized": True,
        "physical_generalization_progress": True,
    }))
    result = load_bend_inspection_readiness(path)
    assert result["status"] == "PROMOTION_EVIDENCE_INVALID_FAIL_CLOSED"
    assert result["production_use_allowed"] is False


def test_readiness_route_precedes_dynamic_job_route():
    # Source-only routing assertion: importing the legacy route closure or
    # calling the default readiness loader would touch protected real state.
    source = Path(__file__).resolve().parents[1] / "apps/api/routes/bend_inspection.py"
    tree = ast.parse(source.read_text())
    strings = [(node.lineno, node.value) for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    positions = {value: line for line, value in strings}
    assert positions["/api/bend-inspection/readiness"] < positions["/api/bend-inspection/{job_id}"]


@pytest.mark.parametrize("field,value", [
    ("active_blockers", "BLOCKED"), ("active_blockers", [None]),
    ("active_blockers", [""]), ("active_blockers", {}),
    ("lane_status", []), ("contract_soak", "PASS"),
    ("generalization_progress_indicator", 1),
    ("physical_generalization_progress", "true"),
    ("production_promotion_authorized", 1),
    ("supervised_shadow_authorized", "true"),
    ("lane_status", {"score": float("nan")}),
    ("lane_status", {"score": float("inf")}),
])
def test_malformed_nested_metadata_fails_closed(field, value):
    result = project_program_status({"kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2", field: value})
    assert result["status"] == "UNAVAILABLE_FAIL_CLOSED"
    assert result["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("blockers", [[], ["OPEN_PHYSICAL_REVIEW"], ["CANDIDATE_MISMATCH"]])
@pytest.mark.parametrize("claim", ["production_promotion_authorized", "supervised_shadow_authorized"])
def test_self_declared_verified_or_shadow_release_has_no_authority(blockers, claim):
    result = project_program_status({
        "kind": "SHERMAN_QC_F1_PROGRAM_STATUS_V2", claim: True,
        "active_blockers": blockers, "physical_generalization_progress": True,
        "promotion_evidence": {"status": "PASS", "verified": True,
                               "candidate_lock_sha256": "a" * 64,
                               "qualification_report_sha256": "b" * 64,
                               "dual_review_truth_sha256": "c" * 64},
    })
    assert result["production_use_allowed"] is False
    assert result["supervised_shadow_authorized"] is False
    assert result["physical_generalization_progress_verified"] is False
    assert all(blocker in result["active_blockers"] for blocker in blockers)


@pytest.mark.parametrize("body", [b"\xff", b"[]", b"{", b"{}", b"x" * 1_048_577])
def test_invalid_bytes_fail_closed_without_path_disclosure(tmp_path, body):
    path = tmp_path / "synthetic-status.json"
    path.write_bytes(body)
    result = load_bend_inspection_readiness(path)
    assert result["status"] == "UNAVAILABLE_FAIL_CLOSED"
    assert str(path) not in json.dumps(result)
