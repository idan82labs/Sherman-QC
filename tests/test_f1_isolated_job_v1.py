from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from research.f1_isolated_job_v1 import IsolatedJobError, JobLimits, _parse_time_peak_rss, run_isolated_job, validate_receipt


HASH = "a" * 64


def _run(tmp_path: Path, command: list[str], *, timeout: float = 5.0, rss: int = 512 * 1024**2, log: int = 16 * 1024**2, artifacts=None):
    return run_isolated_job(
        job_id="1" * 32, lane="SCAN_ONLY", command=command,
        receipt_path=tmp_path / "receipt.json", log_directory=tmp_path / "logs",
        input_hashes={"scan": HASH}, detector_sha256=HASH, config_sha256=HASH,
        ontology_sha256=HASH, artifacts=artifacts,
        limits=JobLimits(timeout_seconds=timeout, rss_limit_bytes=rss, log_limit_bytes=log, poll_interval_seconds=0.02, termination_grace_seconds=0.1),
    )


def test_success_receipt_contains_hashes_not_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "semantic.json"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok'); print('done')"]
    code, receipt = _run(tmp_path, command, artifacts={"semantic": artifact})
    assert code == 0 and receipt["status"] == "COMPLETE"
    assert receipt["artifacts"][0]["sha256"]
    serialized = json.dumps(receipt)
    assert str(tmp_path) not in serialized
    assert "done" not in serialized
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    assert validate_receipt(receipt) == receipt


def test_nonzero_and_missing_artifact_fail_closed(tmp_path: Path) -> None:
    code, receipt = _run(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    assert code == 2 and receipt["status"] == "FAILED" and receipt["reason"] == "NONZERO_EXIT"
    other = tmp_path / "other"
    code, receipt = _run(other, [sys.executable, "-c", "pass"], artifacts={"required": other / "missing"})
    assert code == 2 and receipt["reason"] == "REQUIRED_ARTIFACT_MISSING"


def test_declared_abstention_is_successful_but_not_complete(tmp_path: Path) -> None:
    code, receipt = _run(tmp_path, [sys.executable, "-c", "raise SystemExit(3)"])
    assert code == 0
    assert receipt["status"] == "ABSTAIN"
    assert receipt["reason"] == "DECLARED_ABSTENTION"
    assert receipt["return_code"] == 3
    assert validate_receipt(receipt) == receipt


def test_abstention_cannot_hide_a_missing_required_artifact(tmp_path: Path) -> None:
    code, receipt = _run(
        tmp_path,
        [sys.executable, "-c", "raise SystemExit(3)"],
        artifacts={"abstention_evidence": tmp_path / "missing.json"},
    )
    assert code == 2
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "REQUIRED_ARTIFACT_MISSING"


def test_timeout_kills_entire_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    source = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    code, receipt = _run(tmp_path, [sys.executable, "-c", source], timeout=0.35)
    assert code == 2 and receipt["status"] == "TIMEOUT"
    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        probe = subprocess.run(["ps", "-p", str(child_pid), "-o", "pid="], capture_output=True, text=True)
        if not probe.stdout.strip():
            break
        time.sleep(0.05)
    assert not subprocess.run(["ps", "-p", str(child_pid), "-o", "pid="], capture_output=True, text=True).stdout.strip()


def test_clean_parent_exit_with_background_child_fails_and_drains_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "orphan.pid"
    source = (
        "import subprocess,sys,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid))"
    )
    code, receipt = _run(tmp_path, [sys.executable, "-c", source])
    assert code == 2
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "ORPHAN_PROCESS_GROUP"
    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not subprocess.run(["ps", "-p", str(child_pid), "-o", "pid="], capture_output=True, text=True).stdout.strip():
            break
        time.sleep(0.05)
    assert not subprocess.run(["ps", "-p", str(child_pid), "-o", "pid="], capture_output=True, text=True).stdout.strip()


def test_resource_limit_is_external_and_fail_closed(tmp_path: Path) -> None:
    code, receipt = _run(tmp_path, [sys.executable, "-c", "import time; x=bytearray(8_000_000); time.sleep(2)"], rss=1)
    assert code == 2 and receipt["status"] == "RESOURCE_STOP"
    assert receipt["peak_process_group_rss_bytes"] >= 1


def test_log_flood_is_bounded_and_receipted(tmp_path: Path) -> None:
    code, receipt = _run(tmp_path, [sys.executable, "-c", "import sys; sys.stdout.write('x'*2_000_000)"], log=1_000_000)
    assert code == 2
    assert receipt["status"] == "RESOURCE_STOP"
    assert receipt["reason"] in {"LOG_SIZE_LIMIT", "LOG_SIZE_LIMIT_POST_EXIT"}


def test_short_lived_memory_burst_is_caught_by_post_exit_accounting(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("post-exit /usr/bin/time -l accounting is macOS-specific")
    code, receipt = _run(tmp_path, [sys.executable, "-c", "x=bytearray(100_000_000)"], rss=50_000_000)
    assert code == 2
    assert receipt["status"] == "RESOURCE_STOP"
    assert receipt["reason"] == "PROCESS_GROUP_RSS_LIMIT_POST_EXIT"
    assert receipt["peak_process_group_rss_bytes"] >= 100_000_000


def test_time_peak_parser_rejects_malformed_and_reads_bytes(tmp_path: Path) -> None:
    path = tmp_path / "resource.log"
    path.write_text("  216399872  maximum resident set size\n")
    assert _parse_time_peak_rss(path) == 216399872
    path.write_text("bad maximum resident set size\n")
    assert _parse_time_peak_rss(path) == 0


def test_post_exit_wrapper_overhead_is_accounted_as_timeout(tmp_path: Path) -> None:
    code, receipt = _run(tmp_path, [sys.executable, "-c", "pass"], timeout=0.005)
    assert code == 2
    assert receipt["status"] == "TIMEOUT"
    assert receipt["reason"] in {"WALL_TIME_LIMIT", "WALL_TIME_LIMIT_POST_EXIT"}


def test_tampered_complete_receipts_are_rejected(tmp_path: Path) -> None:
    _code, receipt = _run(tmp_path, [sys.executable, "-c", "pass"])
    receipt["return_code"] = 7
    with pytest.raises(IsolatedJobError, match="COMPLETE"):
        validate_receipt(receipt)
    receipt["return_code"] = 0
    receipt["input_hashes"]["part_id"] = HASH
    with pytest.raises(IsolatedJobError, match="prohibited"):
        validate_receipt(receipt)


@pytest.mark.parametrize("kwargs", [
    {"job_id": "bad"},
    {"lane": "MIXED"},
    {"command": []},
])
def test_invalid_contract_is_rejected(tmp_path: Path, kwargs: dict[str, object]) -> None:
    values = dict(job_id="1" * 32, lane="SCAN_ONLY", command=[sys.executable, "-c", "pass"])
    values.update(kwargs)
    with pytest.raises(IsolatedJobError):
        run_isolated_job(
            **values, receipt_path=tmp_path / "receipt", log_directory=tmp_path / "logs",
            input_hashes={"scan": HASH}, detector_sha256=HASH, config_sha256=HASH,
            ontology_sha256=HASH,
        )


def test_input_roles_enforce_lane_and_feedback_isolation(tmp_path: Path) -> None:
    base = dict(
        job_id="1" * 32, command=[sys.executable, "-c", "pass"],
        receipt_path=tmp_path / "receipt", log_directory=tmp_path / "logs",
        detector_sha256=HASH, config_sha256=HASH, ontology_sha256=HASH,
    )
    for lane, inputs in (
        ("SCAN_ONLY", {}),
        ("SCAN_ONLY", {"cad_step": HASH}),
        ("CAD_ASSISTED", {"expected_count": HASH}),
        ("CAD_ASSISTED", {"expectedcount": HASH}),
    ):
        with pytest.raises(IsolatedJobError):
            run_isolated_job(**base, lane=lane, input_hashes=inputs)
    code, receipt = run_isolated_job(**base, lane="CAD_ASSISTED", input_hashes={"scan": HASH, "cad_step": HASH})
    assert code == 0 and receipt["status"] == "COMPLETE"
    code, receipt = run_isolated_job(**base, lane="SCAN_ONLY", input_hashes={"source_partition": HASH})
    assert code == 0 and receipt["status"] == "COMPLETE"


def test_callers_cannot_override_single_thread_environment(tmp_path: Path) -> None:
    code, receipt = run_isolated_job(
        job_id="1" * 32, lane="SCAN_ONLY",
        command=[sys.executable, "-c", "import os; print(os.environ['OMP_NUM_THREADS'])"],
        receipt_path=tmp_path / "receipt", log_directory=tmp_path / "logs",
        input_hashes={"scan": HASH}, detector_sha256=HASH, config_sha256=HASH,
        ontology_sha256=HASH, environment={"OMP_NUM_THREADS": "99"},
    )
    assert code == 0 and receipt["status"] == "COMPLETE"
    assert (tmp_path / "logs" / f"{'1' * 32}.stdout.log").read_text().strip() == "1"
