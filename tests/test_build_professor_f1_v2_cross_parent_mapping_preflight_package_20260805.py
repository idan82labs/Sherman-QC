from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_professor_f1_v2_cross_parent_mapping_preflight_package_20260805 import (
    DECISION,
    EVIDENCE_KIND,
    PREFLIGHT_KIND,
    PROFILE,
    build_package,
)
from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (
    MAPPING_PREFLIGHT_CRITERIA,
)


def _json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _file(path: Path, body: bytes = b"evidence") -> Path:
    path.write_bytes(body)
    return path


def _inputs(tmp_path: Path) -> dict:
    preflight = {
        "schema_version": 1, "kind": PREFLIGHT_KIND, "profile": PROFILE,
        "decision": DECISION, "ordinary_batch_authorized": False,
        "gold_open_authorized": False, "next_professor_packet_authorized": True,
        "target_row_count": 4_788,
        "target_component_image_present_count": 4_000,
        "target_component_image_absent_outside_tube_count": 788,
        "relation_table_sha256": "1" * 64, "outcome_table_sha256": "2" * 64,
        "mapping_preflight_criteria": {
            criterion: True for criterion in MAPPING_PREFLIGHT_CRITERIA
        },
        "frozen_target_baseline_relation_table": [{} for _ in range(4_788)],
        "frozen_target_outcome_table": [{} for _ in range(4_788)],
    }
    preflight_path = _json(tmp_path / "preflight.json", preflight)
    preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    evidence = {
        "schema_version": 1, "kind": EVIDENCE_KIND, "profile": PROFILE,
        "decision": DECISION, "ordinary_batch_authorized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "compact_bundle_authorized": False, "gold_open_authorized": False,
        "next_professor_packet_authorized": True,
        "mapping_preflight_sha256": preflight_sha, "target_row_count": 4_788,
        "target_component_image_present_count": 4_000,
        "target_component_image_absent_outside_tube_count": 788,
        "relation_table_sha256": "1" * 64, "outcome_table_sha256": "2" * 64,
        "mapping_preflight_criteria": {
            criterion: True for criterion in MAPPING_PREFLIGHT_CRITERIA
        },
    }
    return {
        "mapping_preflight_path": preflight_path,
        "compact_evidence_path": _json(tmp_path / "evidence.json", evidence),
        "ruling_path": _file(tmp_path / "ruling.md"),
        "v2_core_path": _file(tmp_path / "core.py"),
        "runner_path": _file(tmp_path / "runner.py"),
        "validator_path": _file(tmp_path / "validator.py"),
        "validator_test_path": _file(tmp_path / "validator_test.py"),
        "complete_edge_baseline_manifest_path": _file(tmp_path / "baseline.json"),
        "historical_anchor_pdf": _file(tmp_path / "anchor.pdf", b"%PDF"),
    }


def test_package_is_exactly_ten_individual_files_and_mapping_only(tmp_path: Path) -> None:
    result = build_package(output_dir=tmp_path / "package", **_inputs(tmp_path))
    assert result["individual_file_count"] == 10
    assert result["ordinary_batch_authorized"] is False
    assert result["gold_open_authorized"] is False
    prompt = (tmp_path / "package/00_GPT_PRO_PROMPT.md").read_text()
    assert "full untruncated 4,788-row relation" in prompt
    assert "No cross-parent rung was built" in prompt
    assert "AUTHORIZE_ORDINARY_V2_BATCH_SPECIFICATION" in prompt
    assert "fresh conversation" in prompt
    names = {row["name"] for row in result["files"]}
    assert "02_AUTHORITATIVE_MAPPING_PREFLIGHT.json" in names
    assert "03_COMPACT_VALIDATOR_EVIDENCE.json" in names
    assert all("REPLAY" not in name and "SYNTHETIC" not in name for name in names)


def test_package_rejects_external_sha_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    evidence = json.loads(inputs["compact_evidence_path"].read_text())
    evidence["mapping_preflight_sha256"] = "0" * 64
    _json(inputs["compact_evidence_path"], evidence)
    with pytest.raises(ValueError, match="external SHA"):
        build_package(output_dir=tmp_path / "package", **inputs)


def test_package_rejects_v1_or_authorized_batch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    preflight = json.loads(inputs["mapping_preflight_path"].read_text())
    preflight["profile"] = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1"
    preflight["ordinary_batch_authorized"] = True
    _json(inputs["mapping_preflight_path"], preflight)
    with pytest.raises(ValueError, match="amended V2"):
        build_package(output_dir=tmp_path / "package", **inputs)


def test_package_rejects_noncanonical_twenty_criteria(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    preflight = json.loads(inputs["mapping_preflight_path"].read_text())
    preflight["mapping_preflight_criteria"] = {
        f"wrong_{index:02d}": True for index in range(20)
    }
    _json(inputs["mapping_preflight_path"], preflight)
    evidence = json.loads(inputs["compact_evidence_path"].read_text())
    evidence["mapping_preflight_sha256"] = hashlib.sha256(
        inputs["mapping_preflight_path"].read_bytes()
    ).hexdigest()
    _json(inputs["compact_evidence_path"], evidence)
    with pytest.raises(ValueError, match="exact all-true"):
        build_package(output_dir=tmp_path / "package", **inputs)
