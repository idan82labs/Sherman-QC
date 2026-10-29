"""Failure transport/admission tests with no FreeCAD subprocess or CAD access."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("manufacturing_synthetic_runner",
    Path(__file__).parents[1] / "scripts/run_vendor_manufacturing_synthetic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def frozen():
    return {"schema": "SHERMAN_MANUFACTURING_SYNTHETIC_FREEZE_V1", "limits": runner.LIMITS,
            "source_sha256": {"fake": "a" * 64}, "runtime_identity": runner.RUNTIME_IDENTITY}


def test_popen_failure_is_terminal_and_not_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_sources", lambda: {"fake": "a" * 64})
    def unavailable(*args, **kwargs):
        raise OSError("authored missing executable")
    monkeypatch.setattr(runner.subprocess, "Popen", unavailable)
    path = tmp_path / "campaign.json"
    assert runner.execute(path, frozen()) == 1
    ledger = json.loads(path.read_text())
    assert ledger["status"] == "FAILED" and ledger["error_type"] == "OSError"
    assert ledger["finished_at"] and ledger["returncode"] is None
    with pytest.raises(FileExistsError):
        runner.execute(path, frozen())


def test_pin_mismatch_does_not_reserve_or_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_sources", lambda: {"fake": "b" * 64})
    path = tmp_path / "campaign.json"
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        runner.execute(path, frozen())
    assert not path.exists()


def test_missing_runtime_identity_does_not_create_empty_record(tmp_path):
    data = frozen(); del data["runtime_identity"]
    path = tmp_path / "campaign.json"
    with pytest.raises(ValueError, match="SCHEMA_OR_LIMITS"):
        runner.execute(path, data)
    assert not path.exists()


def test_nonfinite_result_json_refused():
    with pytest.raises(ValueError, match="NONFINITE_JSON"):
        json.loads('{"area": NaN}', parse_constant=runner._reject_json_constant)


def test_dead_leader_still_has_group_killed(monkeypatch):
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, signal: killed.append(pid))
    monkeypatch.setattr(runner, "_group_rows", lambda pid: [])
    class Process:
        pid = 123
        returncode = 0
        def wait(self, timeout):
            return 0
    runner._stop_group(Process())
    assert killed == [123]


@pytest.mark.parametrize("raw", ['{"area":1e999}', '{"area":1,"area":2}'])
def test_overflow_or_duplicate_keys_are_not_strict_records(raw):
    with pytest.raises(ValueError):
        runner._load_json(raw)


@pytest.mark.parametrize("result", [{}, {"status": "SYNTHETIC_SUITE_FAILED"},
    {"status": "SYNTHETIC_SUITE_PASSED", "passed": 0, "cases": []}])
def test_empty_or_failed_suite_cannot_be_complete(result):
    with pytest.raises(ValueError, match="NOT_COMPLETE"):
        runner._validate_suite_result(result)


def test_exit_between_samples_still_obeys_final_wall_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_sources", lambda: {"fake": "a" * 64})
    monkeypatch.setattr(runner, "_group_rows", lambda pid: [])
    monkeypatch.setattr(runner, "_stop_group", lambda process: None)
    ticks = iter([0, runner.MAX_SECONDS + 1])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    class Process:
        pid, returncode = 123, 0
        def __init__(self, arguments, **kwargs):
            result = {"status": "SYNTHETIC_SUITE_PASSED", "passed": 1, "cases": ["fake"],
                "failed": 0, "failed_cases": [], "physical_inputs_read": False,
                "manufacturing_release_evidence": False, "freecad_version":
                ["1", "1", "3", "", "", "", "", runner.RUNTIME_IDENTITY["freecad_commit"]],
                "upstream_commit": runner.RUNTIME_IDENTITY["sheetmetal_commit"]}
            Path(arguments[-1]).write_text(json.dumps(result))
        def poll(self):
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    path = tmp_path / "late-campaign.json"
    assert runner.execute(path, frozen()) == 1
    ledger = json.loads(path.read_text())
    assert ledger["status"] == "FAILED" and ledger["stop_reason"] == "WALL_BUDGET_EXCEEDED"
    assert ledger["result"]["status"] == "SYNTHETIC_SUITE_PASSED"
