from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_f1_cad_grouping_qualification_v1.py"


def _mapping() -> dict:
    return {
        "schema": "CAD_EVENT_CORRESPONDENCE_V1",
        "primary_truth_sha256": "a" * 64,
        "step_sha256": "b" * 64,
        "overlay_transform_sha256": "c" * 64,
        "overlay_status": "TRUSTED",
        "events": [{
            "event_id": "E0001",
            "step_face_indices": [1],
            "v9_primitive_ids": ["P1", "P2"],
            "relation": "MANY_TO_ONE",
            "mapping_status": "RESOLVED",
        }],
        "cad_only_unobserved": [],
    }


def _payload(*, replay: bool = True) -> dict:
    mapping = _mapping()
    cases = []
    for index in range(12):
        cases.append({
            "case_id": f"C{index + 1:02d}",
            "mapper_a": mapping,
            "mapper_b": json.loads(json.dumps(mapping)),
            "native_v9_primitive_ids": ["P1", "P2"],
            "mapper_a_replay_identical": replay if index == 0 else True,
            "mapper_b_replay_identical": True,
            "load_export_latency_seconds": [1.0, 1.0, 1.0, 1.0],
            "peak_rss_bytes": 100_000_000,
            "jobs_attempted": 4,
            "jobs_succeeded": 4,
        })
    abstain = {
        "schema": "CAD_EVENT_CORRESPONDENCE_V1",
        "primary_truth_sha256": "a" * 64,
        "step_sha256": "b" * 64,
        "overlay_transform_sha256": "c" * 64,
        "overlay_status": "ABSTAIN",
        "reason": "OVERLAY_UNAVAILABLE_OR_UNTRUSTED",
        "events": [],
        "cad_only_unobserved": [],
    }
    cases.append({
        "case_id": "C13",
        "mapper_a": abstain,
        "mapper_b": json.loads(json.dumps(abstain)),
        "mapper_a_replay_identical": True,
        "mapper_b_replay_identical": True,
        "load_export_latency_seconds": [1.0, 1.0, 1.0, 1.0],
        "peak_rss_bytes": 100_000_000,
        "jobs_attempted": 4,
        "jobs_succeeded": 4,
        "registration_abstention_excluded": True,
    })
    return {
        "schema": "CIBEC_V1_CAD_GROUPING_QUALIFICATION_INPUT",
        "hash_manifest_verified": True,
        "leakage_audit_prohibited_field_or_asset_count": 0,
        "cases": cases,
    }


def _run(tmp_path: Path, payload: dict) -> tuple[subprocess.CompletedProcess[str], dict]:
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_cli_writes_pass_and_returns_zero(tmp_path: Path) -> None:
    completed, result = _run(tmp_path, _payload())
    assert completed.returncode == 0
    assert result["status"] == "PASS"


def test_cli_writes_fail_and_returns_two(tmp_path: Path) -> None:
    completed, result = _run(tmp_path, _payload(replay=False))
    assert completed.returncode == 2
    assert result["status"] == "FAIL"
    assert "deterministic_replay" in result["failed_gates"]
