"""Synthetic harness admission/failure checks without native calls/CADs."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("file_synthetic_runner", Path(__file__).parents[1] / "scripts/run_vendor_manufacturing_file_synthetic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_generation_launch_failure_terminal_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "pins", lambda: {"fake": "a"*64})
    def fail(*args): raise OSError("authored no process")
    monkeypatch.setattr(runner.job, "_supervise_command", fail)
    record, folder = tmp_path / "record.json", tmp_path / "generated"
    result = runner.generate(runner.freeze_generation(), folder, record)
    assert result["status"] == "FAILED" and result["error_type"] == "OSError"
    with pytest.raises(FileExistsError):
        runner.generate(runner.freeze_generation(), folder, record)


def test_generation_pin_mismatch_no_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "pins", lambda: {"fake": "a"*64})
    frozen = copy.deepcopy(runner.freeze_generation())
    frozen["limits"]["wall_seconds"] = 121
    path = tmp_path / "record.json"
    with pytest.raises(ValueError, match="FREEZE_MISMATCH"):
        runner.generate(frozen, tmp_path / "generated", path)
    assert not path.exists()


def test_missing_case_cannot_become_smaller_campaign(tmp_path):
    (tmp_path / "generated-cases.json").write_text(json.dumps([{"case_id": runner.EXPECTED_IDS[0]}]))
    with pytest.raises(ValueError, match="COMPLETE_FROZEN"):
        runner.load_cases(tmp_path)


def test_path_traversal_refused_before_source_read(tmp_path):
    cases = [{"case_id": name, "step_filename": "../source.step", "source_sha256": "a"*64,
              "profile": {}, "expected_bend_count": 1, "expected_area_mm2": 1,
              "expected_status": "DRAFT_COMPUTED", "expected_reason_prefix": None} for name in runner.EXPECTED_IDS]
    (tmp_path / "generated-cases.json").write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="ID_OR_PATH"):
        runner.load_cases(tmp_path)


def test_retains_all_cases_after_first_runtime_exception(tmp_path, monkeypatch):
    names = runner.EXPECTED_IDS
    jobs = [{"case": {"case_id": name, "step_filename": name+".step", "source_sha256": "a"*64,
                      "profile": {}, "expected_status": "DRAFT_COMPUTED", "expected_bend_count": 1}, "freeze": {}}
            for name in names]
    frozen = {"jobs": jobs, "limits": {"total_wall_seconds": 510}, "source_sha256": {"fake": "a"*64},
              "area_oracle_error_limit_mm2": 1e-6}
    monkeypatch.setattr(runner, "freeze_evaluation", lambda *args: frozen)
    monkeypatch.setattr(runner, "pins", lambda: frozen["source_sha256"])
    monkeypatch.setattr(runner.job, "_read_bounded", lambda *args: b"not a CAD")
    def fail(*args): raise RuntimeError("authored worker failure")
    monkeypatch.setattr(runner.job, "run_file_job", fail)
    result = runner.evaluate(frozen, tmp_path, tmp_path / "record.json", tmp_path / "generated-record.json")
    assert result["status"] == "FAILED" and len(result["cases"]) == 4
    assert [row["case_id"] for row in result["cases"]] == list(names)
    assert all(not row["passed"] and row["reason"] == "authored worker failure" for row in result["cases"])


def test_terminal_failed_generation_cannot_authorize_evaluation(tmp_path):
    record = tmp_path / "generation.json"
    record.write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(ValueError, match="GENERATION_RECEIPT"):
        runner.generation_provenance(record, [])


def test_unrelated_unit_refusal_does_not_pass_negative_oracle(tmp_path, monkeypatch):
    jobs = [{"case": {"case_id": name, "step_filename": name+".step", "source_sha256": "a"*64,
                      "profile": {}, "expected_status": "UNAVAILABLE" if index == 3 else "DRAFT_COMPUTED",
                      "expected_reason_prefix": "NATIVE_UNFOLD_REFUSED:" if index == 3 else None,
                      "expected_bend_count": 1, "expected_area_mm2": None if index == 3 else 10}, "freeze": {}}
            for index, name in enumerate(runner.EXPECTED_IDS)]
    frozen = {"jobs": jobs, "limits": {"total_wall_seconds": 510}, "source_sha256": {"fake": "a"*64},
              "area_oracle_error_limit_mm2": 1e-6}
    monkeypatch.setattr(runner, "freeze_evaluation", lambda *args: frozen)
    monkeypatch.setattr(runner, "pins", lambda: frozen["source_sha256"])
    monkeypatch.setattr(runner.job, "_read_bounded", lambda *args: b"not a CAD")
    # Isolate this evaluator decision from the stronger real full-result oracle.
    monkeypatch.setattr(runner.importlib.util, "module_from_spec", lambda spec: SimpleNamespace(assert_file_result=lambda *args: None))
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda *args: None)))
    count = [0]
    def result(raw, digest, profile, freeze, target):
        target.write_text("{}")
        index = count[0]; count[0] += 1
        if index == 3:
            return {"status": "UNAVAILABLE", "result": {"draft": None, "reason": "UNKNOWN_STEP_LENGTH_UNIT"}}
        return {"status": "COMPLETE_DRAFT", "result": {"draft": {"flat_bounds": {"area_mm2": 10}, "bend_rows": [{}]}}}
    monkeypatch.setattr(runner.job, "run_file_job", result)
    receipt = runner.evaluate(frozen, tmp_path, tmp_path / "bad-negative.json", tmp_path / "generation.json")
    assert receipt["status"] == "FAILED"
    assert [case["passed"] for case in receipt["cases"]] == [True, True, True, False]
