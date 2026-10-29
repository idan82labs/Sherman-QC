"""Source-only tests of the one-shot fresh-upload five-part evaluator.

No physical CAD, historical attempt, admission journal or native process is
opened. All route work uses synthetic bytes and mocked native/flat outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import evaluate_vendor_cad_upload_unfold_v1 as campaign


def _bend(label: str):
    return SimpleNamespace(bend_id=label, target_angle=90., target_radius=2.,
        tolerance_angle=1., tolerance_radius=None,
        bend_line_start=(0., 0., 0.), bend_line_end=(10., 0., 0.),
        to_dict=lambda: {"bend_id": label})


def test_five_bindings_are_exact_source_declared_and_no_fixture_candidate_read():
    import ast
    source = (campaign.ROOT / "scripts/build_vendor_unfold_demo.py").read_text()
    tree = ast.parse(source)
    declared = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                            for target in node.targets))
    assert tuple(ast.literal_eval(declared)) == campaign.CASE_SOURCES
    assert len(campaign.CASE_SOURCES) == 5
    text = Path(campaign.__file__).read_text()
    assert "fixture_path" not in text
    assert "vendor-corpus" not in text


def test_freeze_reserves_all_five_without_raw_access_and_pre_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_source_pins", lambda: {"synthetic.py": "a" * 64})
    def archive(root, pins):
        (root / "SOURCE.tar").write_bytes(b"synthetic source archive")
        manifest = campaign._json({"members": list(pins)})
        campaign._write_x(root / "SOURCE_MANIFEST.json", manifest)
        return {"source_archive_sha256": campaign._digest(b"synthetic source archive"),
                "source_manifest_sha256": campaign._digest(manifest)}
    monkeypatch.setattr(campaign, "_archive_sources", archive)
    root = tmp_path / "fresh-five-attempt"
    frozen = campaign.freeze(root)
    assert frozen["state"] == "FROZEN_AWAITING_INDEPENDENT_PRE"
    assert [(root / f"case-{index:02d}-{part}" / "RESERVED.json").exists()
            for index, (part, _, _) in enumerate(campaign.CASE_SOURCES, 1)] == [True] * 5
    with pytest.raises(FileExistsError):
        campaign.freeze(root)
    frozen2, frozen_sha = campaign._validate_frozen(root)
    assert frozen2 == frozen
    with pytest.raises(FileNotFoundError):
        campaign._validate_pre(root, frozen, frozen_sha)
    approved = {"profile": campaign.PRE_PROFILE,
        "decision": "GREEN_FOR_FROZEN_FIVE_PART_UPLOAD_UNFOLD_RUN",
        "blocking_findings": [], "frozen_sha256": frozen_sha,
        "source_manifest_sha256": frozen["source_manifest_sha256"],
        "source_archive_sha256": frozen["source_archive_sha256"],
        "verified_case_sources": campaign._bindings(), "heldouts_excluded": True,
        "isolated_route_checked": True, "whole_case_budget_checked": True,
        "independent_reviewer": "synthetic-independent-test"}
    campaign._write_x(root / "PRE_RUN_AUDIT.json", campaign._json(approved))
    assert campaign._validate_pre(root, frozen, frozen_sha) == approved
    approved["verified_case_sources"] = approved["verified_case_sources"][:-1]
    (root / "PRE_RUN_AUDIT.json").write_bytes(campaign._json(approved))
    with pytest.raises(ValueError, match="INDEPENDENT_PRE_NOT_GREEN"):
        campaign._validate_pre(root, frozen, frozen_sha)


def test_source_archive_members_are_byte_bound_without_any_cad(tmp_path):
    source = tmp_path / "synthetic-source.py"
    source.write_bytes(b"synthetic = True\n")
    pins = {str(source): campaign._digest(source.read_bytes())}
    archive = campaign._archive_sources(tmp_path, pins)
    assert archive["source_archive_sha256"] == campaign._digest((tmp_path / "SOURCE.tar").read_bytes())
    manifest = json.loads((tmp_path / "SOURCE_MANIFEST.json").read_text())
    assert len(manifest["members"]) == 1
    with tarfile.open(tmp_path / "SOURCE.tar", "r") as stream:
        member = manifest["members"][0]
        payload = stream.extractfile(member["member"]).read()
    assert payload == source.read_bytes()
    assert member["sha256"] == campaign._digest(payload)


def test_real_route_ast_publishes_only_its_native_rows_then_real_get_dispatches_flat(tmp_path, monkeypatch):
    from backend import vendor_native_step_upload_v1 as native
    from backend import vendor_unfold_worker_v3 as flat_worker
    from backend.vendor_unfold_contract_v3 import unavailable
    raw = b"synthetic STEP bytes only"
    expected = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(campaign, "CASE_SOURCES", (("SYNTHETIC", "operator.step", expected),))
    folded = b"ply\nformat ascii 1.0\nend_header\n"
    native_calls, flat_calls = [], []

    def preprocess(path, source_hash, angle):
        native_calls.append((path, source_hash, angle))
        return [_bend("B1")], folded, {"status": "AVAILABLE",
            "source_sha256": source_hash,
            "folded_mesh_sha256": hashlib.sha256(folded).hexdigest()}

    def preview(path, source_hash, candidates):
        flat_calls.append((path, source_hash, candidates))
        return unavailable("SYNTHETIC_FLAT_REFUSAL", source_hash)

    monkeypatch.setattr(native, "preprocess_vendor_step", preprocess)
    monkeypatch.setattr(flat_worker, "build_unfold_preview_bounded", preview)
    monkeypatch.setattr(flat_worker, "validate_preview", lambda value, *_: value)
    case = tmp_path / "case-01-SYNTHETIC"
    case.mkdir()
    outcome = campaign._actual_upload_result(case, raw, "SYNTHETIC", expected)
    assert outcome["status"] == "UNAVAILABLE"
    assert outcome["reason"] == "SYNTHETIC_FLAT_REFUSAL"
    assert len(native_calls) == len(flat_calls) == 1
    assert native_calls[0][1:] == (expected, 1.0)
    assert flat_calls[0][1] == expected
    candidate = flat_calls[0][2][0]
    assert candidate["label"] == "B1" and candidate["feature_id"] != "B1"
    setup = json.loads((case / "VENDOR_SETUP_RESPONSE.json").read_text())
    assert setup["part"]["cad_file_hash"] == expected
    assert setup["candidate_generation"] == outcome["candidate_generation"]
    assert setup["candidates"] == flat_calls[0][2]
    assert setup["unfold_preview"]["reason"] == "SYNTHETIC_FLAT_REFUSAL"
    asset = json.loads((case / "FOLDED_ASSET.json").read_text())
    assert (case / asset["relative_path"]).read_bytes() == folded
    assert asset["sha256"] == hashlib.sha256(folded).hexdigest()


def test_bad_native_result_cannot_publish_or_start_flat(tmp_path, monkeypatch):
    from backend import vendor_native_step_upload_v1 as native
    from backend import vendor_unfold_worker_v3 as flat_worker
    raw = b"synthetic STEP refusal"
    expected = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(campaign, "CASE_SOURCES", (("SYNTHETIC", "operator.step", expected),))
    monkeypatch.setattr(native, "preprocess_vendor_step", lambda *_: (
        [], None, {"status": "UNAVAILABLE", "reason": "NO_NATIVE_BENDS"}))
    def forbidden(*_args):
        raise AssertionError("Flat worker must not run after refused upload")
    monkeypatch.setattr(flat_worker, "build_unfold_preview_bounded", forbidden)
    case = tmp_path / "case-01-SYNTHETIC"
    case.mkdir()
    outcome = campaign._actual_upload_result(case, raw, "SYNTHETIC", expected)
    assert outcome["status"] == "UNAVAILABLE"
    assert "NO_NATIVE_BENDS" in outcome["reason"]
    assert not (case / "CANDIDATES.json").exists()
    assert not (case / "VENDOR_SETUP_RESPONSE.json").exists()


def test_direct_case_entry_refuses_without_parent_launch_before_raw_read(tmp_path, monkeypatch):
    expected = "a" * 64
    monkeypatch.setattr(campaign, "CASE_SOURCES", (("SYNTHETIC", "operator.step", expected),))
    monkeypatch.setattr(campaign, "_validate_frozen_started", lambda root: ({}, "f" * 64))
    monkeypatch.setattr(campaign, "_validate_pre", lambda *args: {})
    monkeypatch.setenv("SHERMAN_VENDOR_FIVE_CASE_NONCE", "nonce")
    case = tmp_path / "case-01-SYNTHETIC"
    case.mkdir()
    monkeypatch.setattr(campaign, "_read_bounded", lambda path, limit: (
        (_ for _ in ()).throw(AssertionError("Raw CAD must not be opened"))
        if Path(path).name == "operator.step" else Path(path).read_bytes()))
    with pytest.raises(FileNotFoundError):
        campaign._case_worker(tmp_path, 0, "nonce")


def test_forged_env_and_mismatched_parent_launch_refuse_before_raw(tmp_path, monkeypatch):
    expected = "a" * 64
    monkeypatch.setattr(campaign, "CASE_SOURCES", (("SYNTHETIC", "operator.step", expected),))
    monkeypatch.setattr(campaign, "_validate_frozen_started", lambda root: ({}, "f" * 64))
    monkeypatch.setattr(campaign, "_validate_pre", lambda *args: {})
    case = tmp_path / "case-01-SYNTHETIC"
    case.mkdir()
    (tmp_path / "EXECUTION_STARTED.json").write_bytes(campaign._json({
        "profile": campaign.PROFILE, "frozen_sha256": "f" * 64,
        "started_unix_ns": 1}))
    (case / "CASE_LAUNCH.json").write_bytes(campaign._json({
        "profile": campaign.PROFILE, "case_index": 0, "part": "SYNTHETIC",
        "frozen_sha256": "f" * 64, "nonce_sha256": campaign._digest(b"real-nonce"),
        "parent_pid": os.getppid() + 1}))
    monkeypatch.setattr(campaign, "_read_bounded", lambda path, limit: (
        (_ for _ in ()).throw(AssertionError("Raw CAD must not be opened"))
        if Path(path).name == "operator.step" else Path(path).read_bytes()))
    monkeypatch.setenv("SHERMAN_VENDOR_FIVE_CASE_NONCE", "forged-nonce")
    with pytest.raises(ValueError, match="PARENT_OWNED_CASE_LAUNCH_REQUIRED"):
        campaign._case_worker(tmp_path, 0, "forged-nonce")
    monkeypatch.setenv("SHERMAN_VENDOR_FIVE_CASE_NONCE", "real-nonce")
    with pytest.raises(ValueError, match="PARENT_OWNED_CASE_LAUNCH_REQUIRED"):
        campaign._case_worker(tmp_path, 0, "real-nonce")
    assert not (case / "CASE_CLAIM.json").exists()


def test_direct_script_case_worker_reaches_frozen_gate_without_backend_import_error(tmp_path):
    """Production Popen uses direct script path, not `python -m scripts...`."""
    (tmp_path / "FROZEN.json").write_bytes(b"{}")
    env = dict(os.environ)
    env.update({"SHERMAN_VENDOR_FIVE_CASE_NONCE": "synthetic-nonce",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""})
    completed = subprocess.run([sys.executable, str(campaign.__file__),
        "--case-worker", str(tmp_path), "0", "synthetic-nonce"],
        cwd=campaign.ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert completed.returncode != 0
    assert "FROZEN_SOURCE_OR_CONFIG_MISMATCH" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert not (tmp_path / "UPLOAD_SOURCE.step").exists()


def test_process_tree_and_inner_journal_are_both_required(tmp_path, monkeypatch):
    rows = {100: (1, 100, 200), 101: (100, 101, 300),
            102: (101, 102, 400), 200: (1, 200, 500)}
    assert campaign._tree(rows, 100) == {100, 101, 102}
    from backend import vendor_unfold_worker_v3 as shared
    journal = tmp_path / "admission.json"
    monkeypatch.setattr(shared, "ADMISSION_LOCK_PATH", str(journal))
    assert campaign._admission_quiescent(allow_absent=True) == (True, "NO_NATIVE_ADMISSION_YET")
    assert campaign._admission_quiescent()[0] is False
    journal.write_text('{"state":"RUNNING","pgid":102}')
    assert campaign._admission_quiescent()[0] is False
    journal.write_text('{"state":"QUIESCENT","pgid":null}')
    assert campaign._admission_quiescent() == (True, "QUIESCENT")


def test_denied_ps_is_typed_monitor_failure(monkeypatch):
    def denied(*_args, **_kwargs):
        raise PermissionError("synthetic denied ps")
    monkeypatch.setattr(campaign.subprocess, "run", denied)
    with pytest.raises(RuntimeError, match="PROCESS_TREE_MONITOR_UNAVAILABLE"):
        campaign._ps_rows()


def test_outer_case_refuses_available_child_when_inner_journal_is_not_quiescent(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    part = campaign.CASE_SOURCES[0][0]
    case = root / f"case-01-{part}"
    case.mkdir()
    (root / "FROZEN.json").write_bytes(b"synthetic frozen bytes")
    (case / "CHILD_OUTCOME.json").write_bytes(campaign._json({
        "status": "AVAILABLE", "reason": None}))
    class Child:
        pid = 1234567
        returncode = 0
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(campaign, "_ps_rows", lambda: {os.getpid(): (os.getppid(), os.getpgrp(), 100)})
    monkeypatch.setattr(campaign, "_kill_seen_groups", lambda groups: True)
    checks = iter([(True, "QUIESCENT"), (False, "RUNNING")])
    monkeypatch.setattr(campaign, "_admission_quiescent", lambda **kwargs: next(checks))
    row = campaign._run_case(root, 0, campaign.time.monotonic() + 330)
    assert row["status"] == "UNAVAILABLE"
    assert row["reason"] == "INNER_NATIVE_PROCESS_CLEANUP_UNVERIFIED"
    assert row["native_admission_quiescent"] is False
    assert (case / "CASE_LAUNCH.json").exists()


def test_outer_case_monitor_failure_kills_seen_group_and_cannot_publish_success(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    part = campaign.CASE_SOURCES[0][0]
    case = root / f"case-01-{part}"
    case.mkdir()
    (root / "FROZEN.json").write_bytes(b"synthetic frozen bytes")
    class Child:
        pid = 2345678
        returncode = -9
        def poll(self):
            return None
        def wait(self, timeout=None):
            return -9
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: Child())
    ps_calls = []
    def ps():
        ps_calls.append(1)
        if len(ps_calls) == 1:
            return {os.getpid(): (os.getppid(), os.getpgrp(), 100)}
        raise RuntimeError("PROCESS_TREE_MONITOR_LOST_CHILD")
    monkeypatch.setattr(campaign, "_ps_rows", ps)
    killed = []
    monkeypatch.setattr(campaign, "_kill_seen_groups", lambda groups: killed.append(groups.copy()) or True)
    monkeypatch.setattr(campaign, "_admission_quiescent", lambda **kwargs: (True, "QUIESCENT"))
    row = campaign._run_case(root, 0, campaign.time.monotonic() + 330)
    assert row["status"] == "UNAVAILABLE" and row["monitor_failed"] is True
    assert row["reason"] == "PROCESS_TREE_MONITOR_LOST_CHILD"
    assert killed == [{Child.pid}]


def test_whole_case_rss_includes_inner_different_process_group(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    part = campaign.CASE_SOURCES[0][0]
    (root / f"case-01-{part}").mkdir()
    (root / "FROZEN.json").write_bytes(b"synthetic frozen bytes")
    class Child:
        pid = 3456789
        returncode = -9
        def poll(self):
            return None
        def wait(self, timeout=None):
            return -9
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: Child())
    calls = []
    def ps():
        calls.append(1)
        if len(calls) == 1:
            return {os.getpid(): (os.getppid(), os.getpgrp(), 100)}
        return {Child.pid: (os.getpid(), Child.pid, campaign.RSS_KIB - 100),
                Child.pid + 1: (Child.pid, Child.pid + 1, 200)}
    monkeypatch.setattr(campaign, "_ps_rows", ps)
    killed = []
    monkeypatch.setattr(campaign, "_kill_seen_groups", lambda groups: killed.append(groups.copy()) or True)
    monkeypatch.setattr(campaign, "_admission_quiescent", lambda **kwargs: (True, "QUIESCENT"))
    row = campaign._run_case(root, 0, campaign.time.monotonic() + 330)
    assert row["status"] == "UNAVAILABLE"
    assert row["reason"] == "WHOLE_CASE_RSS_BUDGET_EXCEEDED"
    assert row["sampled_peak_process_tree_rss_kib"] == campaign.RSS_KIB + 100
    assert killed == [{Child.pid, Child.pid + 1}]


def test_full_case_reservation_rechecked_after_preflight_before_spawn(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    part = campaign.CASE_SOURCES[0][0]
    (root / f"case-01-{part}").mkdir()
    (root / "FROZEN.json").write_bytes(b"synthetic frozen bytes")
    start = campaign.time.monotonic()
    import time
    def slow_preflight():
        time.sleep(.015)
        return {os.getpid(): (os.getppid(), os.getpgrp(), 100)}
    monkeypatch.setattr(campaign, "_ps_rows", slow_preflight)
    monkeypatch.setattr(campaign, "_admission_quiescent", lambda **kwargs: (True, "QUIESCENT"))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: (
        (_ for _ in ()).throw(AssertionError("Expired case must not spawn"))))
    row = campaign._run_case(root, 0, start + campaign.WALL_SECONDS + .01)
    assert row["status"] == "UNAVAILABLE"
    assert row["reason"] == "WHOLE_CASE_RESERVATION_EXPIRED_BEFORE_SPAWN"
    assert row["parent_observed_exit"] is None


def test_execute_cli_reports_actual_failed_state_not_hardcoded_complete(tmp_path, monkeypatch, capsys):
    def failed(root):
        (root / "CAMPAIGN_TERMINAL.json").write_bytes(campaign._json({
            "state": "FAILED_SAFETY_GATE_PENDING_POST"}))
        return [{"status": "NOT_EXECUTED"}] * 5
    monkeypatch.setattr(campaign, "execute", failed)
    monkeypatch.setattr(campaign.sys, "argv", ["runner", "--execute", "--attempt-root", str(tmp_path)])
    campaign.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"available_count": 0, "state": "FAILED_SAFETY_GATE_PENDING_POST",
                       "terminal_rows": 5}


def test_failed_monitor_halts_later_cases_but_keeps_five_terminal_rows(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    for index, (part, _, _) in enumerate(campaign.CASE_SOURCES, 1):
        (root / f"case-{index:02d}-{part}").mkdir()
    monkeypatch.setattr(campaign, "_validate_frozen", lambda _: ({}, "f" * 64))
    monkeypatch.setattr(campaign, "_validate_pre", lambda *_: {})
    called = []
    def run(_root, index, _deadline):
        called.append(index)
        return {"part": campaign.CASE_SOURCES[index][0], "status": "UNAVAILABLE",
                "reason": "PROCESS_TREE_MONITOR_LOST_CHILD", "monitor_failed": True,
                "process_tree_quiescent": False, "native_admission_quiescent": False}
    monkeypatch.setattr(campaign, "_run_case", run)
    rows = campaign.execute(root)
    assert called == [0] and len(rows) == 5
    assert [row["status"] for row in rows] == ["UNAVAILABLE"] + ["NOT_EXECUTED"] * 4
    assert all((root / f"case-{index:02d}-{part}" / "TERMINAL.json").exists()
               for index, (part, _, _) in enumerate(campaign.CASE_SOURCES, 1))
    assert json.loads((root / "CAMPAIGN_TERMINAL.json").read_text())["state"] == "FAILED_SAFETY_GATE_PENDING_POST"
    with pytest.raises(FileExistsError):
        campaign.execute(root)
