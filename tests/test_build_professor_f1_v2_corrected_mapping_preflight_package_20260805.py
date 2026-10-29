from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_professor_f1_v2_corrected_mapping_preflight_package_20260805 import (
    DECISION,
    EVIDENCE_KIND,
    PREFLIGHT_KIND,
    PROFILE,
    build_package,
)
from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (
    MAPPING_PREFLIGHT_CRITERIA,
)


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _file(path: Path) -> Path:
    path.write_text("evidence", encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> dict:
    criteria = {key: True for key in MAPPING_PREFLIGHT_CRITERIA}
    preflight = {
        "schema_version": 1, "kind": PREFLIGHT_KIND, "profile": PROFILE,
        "decision": DECISION, "ordinary_batch_authorized": False,
        "gold_open_authorized": False, "next_professor_packet_authorized": True,
        "mapping_preflight_criteria": criteria, "target_row_count": 4_788,
        "target_component_image_present_count": 4_772,
        "target_component_image_absent_outside_tube_count": 16,
        "frozen_target_projection": [{}] * 4_788,
        "frozen_target_baseline_relation_table": [{}] * 4_788,
        "frozen_target_outcome_table": [{}] * 4_788,
        "all_pair_membership_row_count": 16_988,
        "all_pair_membership_table": [{}] * 16_988,
        "threshold_hash_table": [{}] * 37,
        "relation_table_sha256": "1" * 64,
        "outcome_table_sha256": "2" * 64,
        "frozen_target_projection_sha256": "3" * 64,
        "all_pair_membership_table_sha256": "4" * 64,
        "threshold_hash_table_sha256": "5" * 64,
    }
    preflight_path = _json(tmp_path / "preflight.json", preflight)
    evidence = {
        "schema_version": 2, "kind": EVIDENCE_KIND, "profile": PROFILE,
        "decision": DECISION, "ordinary_batch_authorized": False,
        "gold_open_authorized": False, "next_professor_packet_authorized": True,
        "mapping_preflight_criteria": criteria,
        "mapping_preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        **{field: preflight[field] for field in (
            "relation_table_sha256", "outcome_table_sha256",
            "frozen_target_projection_sha256", "all_pair_membership_table_sha256",
            "threshold_hash_table_sha256",
        )},
    }
    return {
        "mapping_preflight_path": preflight_path,
        "compact_evidence_path": _json(tmp_path / "evidence.json", evidence),
        "invalid_response_path": _file(tmp_path / "invalid.md"),
        "amended_ruling_path": _file(tmp_path / "amended.md"),
        "v2_core_path": _file(tmp_path / "core.py"),
        "runner_path": _file(tmp_path / "runner.py"),
        "validator_path": _file(tmp_path / "validator.py"),
        "validator_test_path": _file(tmp_path / "validator_test.py"),
        "complete_edge_baseline_manifest_path": _file(tmp_path / "baseline.json"),
    }


def test_corrected_packet_is_exactly_ten_individual_files(tmp_path: Path) -> None:
    result = build_package(output_dir=tmp_path / "package", **_inputs(tmp_path))
    assert result["individual_file_count"] == 10
    prompt = (tmp_path / "package/00_GPT_PRO_PROMPT.md").read_text()
    assert "complete 16,988-row all-pair" in prompt
    assert "prior mapping packet was correctly stopped" in prompt
    assert "five required tamper regressions" in prompt


def test_corrected_packet_rejects_compact_v1(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    evidence = json.loads(inputs["compact_evidence_path"].read_text())
    evidence["schema_version"] = 1
    evidence["kind"] = "f1_v2_cross_parent_rung_mapping_preflight_compact_evidence_v1"
    _json(inputs["compact_evidence_path"], evidence)
    with pytest.raises(ValueError, match="compact-v2"):
        build_package(output_dir=tmp_path / "package", **inputs)


def test_corrected_packet_rejects_incomplete_pair_population(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    preflight = json.loads(inputs["mapping_preflight_path"].read_text())
    preflight["all_pair_membership_table"] = preflight["all_pair_membership_table"][:-1]
    _json(inputs["mapping_preflight_path"], preflight)
    evidence = json.loads(inputs["compact_evidence_path"].read_text())
    evidence["mapping_preflight_sha256"] = hashlib.sha256(
        inputs["mapping_preflight_path"].read_bytes()
    ).hexdigest()
    _json(inputs["compact_evidence_path"], evidence)
    with pytest.raises(ValueError, match="16,988"):
        build_package(output_dir=tmp_path / "package", **inputs)
