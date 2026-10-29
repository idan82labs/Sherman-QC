from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_f1_v51_today_scan_only_development import (
    MANIFEST_KIND,
    NAMESPACE,
    run_cohort,
    validate_scan_manifest,
)


def _write_input(tmp_path: Path, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        scan = tmp_path / f"synthetic-{index}.ply"
        scan.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
        rows.append(
            {
                "scan_path": str(scan),
                "scan_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "kind": MANIFEST_KIND, "cases": rows}),
        encoding="utf-8",
    )
    return manifest


def test_manifest_derives_opaque_ids_for_arbitrary_count(tmp_path: Path) -> None:
    manifest = _write_input(tmp_path, count=3)
    cases = validate_scan_manifest(manifest)
    assert [case["case_id"] for case in cases] == ["T001", "T002", "T003"]
    assert all(set(case) == {"case_id", "scan_path", "scan_sha256"} for case in cases)


def test_manifest_accepts_legacy_scan_geometry_stl(tmp_path: Path) -> None:
    scan = tmp_path / "legacy-scan.stl"
    scan.write_text("solid scan\nendsolid scan\n", encoding="ascii")
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": MANIFEST_KIND,
                "cases": [{"scan_path": str(scan), "scan_sha256": hashlib.sha256(scan.read_bytes()).hexdigest()}],
            }
        ),
        encoding="utf-8",
    )
    assert validate_scan_manifest(manifest)[0]["scan_path"] == str(scan.resolve())


@pytest.mark.parametrize("forbidden", ["labels", "sender_count", "cad_path", "drawing_path", "part_id"])
def test_manifest_rejects_any_side_information(tmp_path: Path, forbidden: str) -> None:
    manifest = _write_input(tmp_path, count=1)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0][forbidden] = "forbidden"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only scan_path and scan_sha256"):
        validate_scan_manifest(manifest)


def test_mock_primary_replay_and_local_namespace(tmp_path: Path) -> None:
    manifest = _write_input(tmp_path, count=2)
    v9 = tmp_path / "v9.json"
    v4 = tmp_path / "v4.json"
    v9.write_text("{}\n", encoding="utf-8")
    v4.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    calls = []

    def mock_runner(case: dict, root: Path, pass_name: str, _v9: Path, _v4: Path) -> dict:
        calls.append((case["case_id"], pass_name, root))
        return {
            "case_id": case["case_id"],
            "pass": pass_name,
            "scan_sha256": case["scan_sha256"],
            "semantic_result_sha256": hashlib.sha256(case["case_id"].encode()).hexdigest(),
        }

    result = run_cohort(manifest, v9, v4, output, case_runner=mock_runner)
    assert calls == [
        ("T001", "primary", output),
        ("T002", "primary", output),
        ("T001", "replay", output),
        ("T002", "replay", output),
    ]
    assert result["namespace"] == NAMESPACE
    assert result["case_count"] == 2
    assert result["semantic_replay_byte_identity"] == {"T001": True, "T002": True}
    assert result["truth_opened"] is False
    assert result["cad_used"] is False
    assert result["drawings_used"] is False
    assert result["sender_counts_used"] is False
    assert (output / "development_execution_manifest.json").is_file()
    assert not (output / "invocation_events.jsonl").exists()


def test_mock_replay_difference_is_terminal(tmp_path: Path) -> None:
    manifest = _write_input(tmp_path, count=1)
    v9 = tmp_path / "v9.json"
    v4 = tmp_path / "v4.json"
    v9.write_text("{}", encoding="utf-8")
    v4.write_text("{}", encoding="utf-8")

    def mock_runner(case: dict, _root: Path, pass_name: str, _v9: Path, _v4: Path) -> dict:
        return {
            "case_id": case["case_id"],
            "pass": pass_name,
            "semantic_result_sha256": "a" * 64 if pass_name == "primary" else "b" * 64,
        }

    with pytest.raises(RuntimeError, match="semantic result bytes differ"):
        run_cohort(manifest, v9, v4, tmp_path / "output", case_runner=mock_runner)


def test_fail_closed_case_abstention_continues_and_replays(tmp_path: Path) -> None:
    manifest = _write_input(tmp_path, count=2)
    v9 = tmp_path / "v9.json"
    v4 = tmp_path / "v4.json"
    v9.write_text("{}", encoding="utf-8")
    v4.write_text("{}", encoding="utf-8")

    def mock_runner(case: dict, root: Path, pass_name: str, _v9: Path, _v4: Path) -> dict:
        if case["case_id"] == "T001":
            raise ValueError("bounded pool failed")
        semantic = root / pass_name / case["case_id"] / "semantic_result.json"
        semantic.parent.mkdir(parents=True, exist_ok=True)
        semantic.write_text('{"predictions":[],"source_only":true}\n', encoding="utf-8")
        return {
            "case_id": case["case_id"],
            "pass": pass_name,
            "semantic_result_sha256": hashlib.sha256(semantic.read_bytes()).hexdigest(),
        }

    result = run_cohort(
        manifest,
        v9,
        v4,
        tmp_path / "output",
        case_runner=mock_runner,
        record_case_failures_as_abstentions=True,
        max_workers=2,
    )
    assert result["primary_failure_abstention_count"] == 1
    assert result["replay_failure_abstention_count"] == 1
    assert all(result["semantic_replay_byte_identity"].values())
    failed = json.loads((tmp_path / "output/primary/T001/semantic_result.json").read_text())
    assert failed["pipeline_status"] == "FAILED_ABSTAIN"
    assert failed["predictions"] == []


def test_prediction_renderer_direct_entry_from_foreign_cwd(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "scripts/render_f1_v51_new_corpus_prediction_only.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
