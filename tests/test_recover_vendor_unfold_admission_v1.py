"""Mocked recovery only: no live admission journal, /bin/ps or native work.

Temporary files contain authored protocols/PRE/receipts, not old artifacts.
Every recovery test replaces the live journal, monitor and source-pin reader.
"""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recover_vendor_unfold_admission_v1 as recovery


def raw_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Journal(io.BytesIO):
    def fileno(self):
        return 987654

    def close(self):
        self.closed_by_recovery = True


@pytest.fixture
def setup(monkeypatch, tmp_path):
    pins = {recovery.SOURCE_PATHS[0]: "a" * 64, recovery.SOURCE_PATHS[1]: "b" * 64,
        recovery.SOURCE_PATHS[2]: recovery.SHARED_WORKER_SHA256,
        recovery.SOURCE_PATHS[3]: recovery.COMMON_CONTRACT_SHA256}
    source_reads, opens, probes, writes, flocks = [], [], [], [], []
    monkeypatch.setattr(recovery, "_source_pins", lambda: source_reads.append(True) or dict(pins))
    journal = Journal(raw_json(recovery.JOURNAL_STATE))
    assert hashlib.sha256(journal.getvalue()).hexdigest() == recovery.JOURNAL_SHA256
    monkeypatch.setattr(recovery, "_open_journal", lambda path: opens.append(path) or journal)
    monkeypatch.setattr(recovery, "_journal_identity", lambda *_: None)
    monkeypatch.setattr(recovery, "_verify_owned_record", lambda *_: None)
    monkeypatch.setattr(recovery, "_group_rows", lambda pgid, started: probes.append(pgid) or [])
    monkeypatch.setattr(recovery.fcntl, "flock", lambda fd, mode: flocks.append((fd, mode)))
    def write(lock, state):
        writes.append(dict(state)); lock.seek(0); lock.write(raw_json(state)); lock.truncate()
    monkeypatch.setattr(recovery.shared, "_record", write)
    clock = [0.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recovery.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    protocol_path, receipt_path, pre_path = (tmp_path / name for name in ("protocol.json", "receipt.jsonl", "pre.json"))
    protocol = recovery.freeze(str(protocol_path), receipt_path=str(receipt_path), author="implementation-author",
        owned_record_sha256=recovery.OWNED_RECORD_SHA256)
    pre = {"schema": recovery.PRE_SCHEMA, "disposition": "GREEN", "author": "implementation-author",
        "reviewer": "independent-reviewer", "blocking_findings": [],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_bundle_sha256": hashlib.sha256(raw_json(pins)).hexdigest(), "journal_sha256": recovery.JOURNAL_SHA256}
    pre_path.write_bytes(raw_json(pre))
    def execute():
        result = recovery.recover(str(protocol_path), str(pre_path), receipt_path=str(receipt_path))
        records = [json.loads(line) for line in receipt_path.read_bytes().splitlines()]
        assert len(records) == 2 and records[0]["event"] == "START" and records[1] == result
        assert not recovery.shared._slot.locked(), "terminal must follow shared-slot release"
        return result
    return SimpleNamespace(pins=pins, source_reads=source_reads, opens=opens, probes=probes, writes=writes,
        flocks=flocks, journal=journal, clock=clock, protocol=protocol, pre=pre, protocol_path=protocol_path,
        pre_path=pre_path, receipt_path=receipt_path, execute=execute)


def test_fixed_scope_success_three_empty_observations_and_cleanup(setup):
    result = setup.execute()
    assert result["status"] == "COMPLETE" and result["cleanup_complete"] is True
    assert setup.probes == [23559, 23559, 23559]
    assert setup.writes == [{"state": "QUIESCENT", "pgid": None}]
    assert result["before_sha256"] == recovery.JOURNAL_SHA256
    assert result["after_sha256"] == hashlib.sha256(raw_json({"state": "QUIESCENT", "pgid": None})).hexdigest()
    assert result["journal_mutated"] is True and result["action_time_utc"]
    assert result["observations"][1]["elapsed_seconds"] - result["observations"][0]["finished_elapsed_seconds"] >= .1
    assert result["native_activity_extent"] == "UNOBSERVED"
    assert result["manufacturing_ready"] is False and result["release_allowed"] is False
    assert setup.flocks[-1][1] == recovery.fcntl.LOCK_UN


@pytest.mark.parametrize("phase", [0, 1])
def test_live_group_never_mutates_journal(monkeypatch, setup, phase):
    observations = iter([[]] * phase + [[(23559, 99)]])
    monkeypatch.setattr(recovery, "_group_rows", lambda *_: next(observations))
    result = setup.execute()
    assert result["reason"] == "OWNED_PROCESS_GROUP_NOT_EMPTY"
    assert result["journal_mutated"] is False and not setup.writes
    assert result["before_sha256"] == result["after_sha256"] == recovery.JOURNAL_SHA256


def test_group_appearing_after_write_is_requarantined(monkeypatch, setup):
    observations = iter([[], [], [(23560, 10)]])
    monkeypatch.setattr(recovery, "_group_rows", lambda *_: next(observations))
    result = setup.execute()
    assert result["status"] == "FAILED" and result["quarantine_restored"] is True
    assert setup.writes == [{"state": "QUIESCENT", "pgid": None}, recovery.JOURNAL_STATE]
    assert result["after_sha256"] == recovery.JOURNAL_SHA256


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_ps_permission_denial_is_retained_without_quiescent_on_early_failure(monkeypatch, setup, phase):
    calls = [0]
    def probe(*_):
        call = calls[0]; calls[0] += 1
        if call == phase:
            raise PermissionError("authored /bin/ps denial")
        return []
    monkeypatch.setattr(recovery, "_group_rows", probe)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["exception_type"] == "PermissionError"
    assert result["observations"][-1]["status"] == "ERROR"
    assert result["after_sha256"] == recovery.JOURNAL_SHA256
    assert setup.writes == ([] if phase < 2 else [{"state": "QUIESCENT", "pgid": None}, recovery.JOURNAL_STATE])


def test_changed_initial_journal_refuses_before_monitor(setup):
    setup.journal.seek(0); setup.journal.write(b'{"pgid":1,"state":"QUARANTINED"}'); setup.journal.truncate()
    result = setup.execute()
    assert result["reason"] == "JOURNAL_HASH_CHANGED" and not setup.probes and not setup.writes


def test_exact_bytes_changed_under_os_lock_refuses(monkeypatch, setup):
    calls = [0]
    def probe(*_):
        calls[0] += 1
        if calls[0] == 2:
            setup.journal.seek(0); setup.journal.write(b" "); setup.journal.truncate()
        return []
    monkeypatch.setattr(recovery, "_group_rows", probe)
    result = setup.execute()
    assert result["reason"] == "JOURNAL_BYTES_CHANGED_UNDER_LOCK" and not setup.writes


@pytest.mark.parametrize("which", ["pre", "before_write", "after_write"])
def test_changed_sources_never_admit_success(monkeypatch, setup, which):
    calls = [0]
    # Freeze already completed; recovery calls before admission, before write,
    # and after write. A changed last pin causes rollback under the same lock.
    at = {"pre": 1, "before_write": 2, "after_write": 3}[which]
    def pins():
        calls[0] += 1
        return {**setup.pins, recovery.SOURCE_PATHS[0]: "c" * 64} if calls[0] == at else dict(setup.pins)
    monkeypatch.setattr(recovery, "_source_pins", pins)
    result = setup.execute()
    assert result["status"] == "FAILED" and "SOURCE_CHANGED" in result["reason"]
    assert setup.writes == ([{"state": "QUIESCENT", "pgid": None}, recovery.JOURNAL_STATE] if which == "after_write" else [])


@pytest.mark.parametrize("key,value", [("reviewer", "implementation-author"), ("reviewer", ""),
    ("author", "another-author"), ("disposition", "AMBER"), ("blocking_findings", ["blocker"]),
    ("blocking_findings", False), ("protocol_sha256", "0" * 64),
    ("source_bundle_sha256", "0" * 64), ("journal_sha256", "0" * 64), ("schema", "OTHER")])
def test_independent_exact_pre_required_before_journal_access(setup, key, value):
    setup.pre[key] = value; setup.pre_path.write_bytes(raw_json(setup.pre))
    result = setup.execute()
    assert result["status"] == "FAILED" and not setup.opens and not setup.probes and not setup.writes
    assert result["journal_mutated"] is False


def test_pre_cannot_contain_extra_keys(setup):
    setup.pre["extra"] = True; setup.pre_path.write_bytes(raw_json(setup.pre))
    assert setup.execute()["reason"] == "INVALID_PRE_SCHEMA"
    assert not setup.opens


@pytest.mark.parametrize("key,value", [("pgid", 123), ("pgid", True), ("journal_path", "/tmp/other.lock"),
    ("journal_sha256", "a" * 64), ("journal_state", {"pgid": 23559, "state": "RUNNING"}),
    ("owned_generation_record", "/tmp/generic-failure.jsonl"),
    ("owned_generation_record_sha256", "a" * 64), ("owned_generation_protocol", "/tmp/generic.json"),
    ("declared_reason", "generic unlock"), ("receipt_path", "/tmp/other-receipt"),
    ("limits", {**recovery.LIMITS, "aggregate_seconds": 20})])
def test_protocol_cannot_generalize_recovery_scope(setup, key, value):
    setup.protocol[key] = value; setup.protocol_path.write_bytes(raw_json(setup.protocol))
    assert setup.execute()["status"] == "FAILED"
    assert not setup.opens and not setup.probes and not setup.writes


def test_shared_slot_busy_refuses_before_journal_access(setup):
    recovery.shared._slot.acquire()
    try:
        result = recovery.recover(str(setup.protocol_path), str(setup.pre_path), receipt_path=str(setup.receipt_path))
    finally:
        recovery.shared._slot.release()
    assert result["reason"] == "SHARED_SLOT_BUSY" and not setup.opens and not setup.writes


def test_shared_os_lock_busy_refuses_before_journal_bytes(monkeypatch, setup):
    def busy(*_):
        raise BlockingIOError("authored contention")
    monkeypatch.setattr(recovery.fcntl, "flock", busy)
    assert setup.execute()["reason"] == "SHARED_OS_LOCK_BUSY"
    assert not setup.probes and not setup.writes


@pytest.mark.parametrize("reason", ["JOURNAL_NOT_CURRENT_UID_REGULAR", "JOURNAL_PATH_REPLACED"])
def test_foreign_or_replaced_journal_abstains(monkeypatch, setup, reason):
    def identity(*_):
        raise recovery.RecoveryError(reason)
    monkeypatch.setattr(recovery, "_journal_identity", identity)
    assert setup.execute()["reason"] == reason
    assert not setup.probes and not setup.writes


def test_symlink_denial_never_mutates(monkeypatch, setup):
    def symlink(*_):
        raise OSError("authored O_NOFOLLOW denial")
    monkeypatch.setattr(recovery, "_open_journal", symlink)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["journal_mutated"] is False and not setup.writes


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_monitor_time_expiry_refuses_or_restores(monkeypatch, setup, phase):
    calls = [0]
    def probe(*_):
        call = calls[0]; calls[0] += 1
        if call == phase:
            setup.clock[0] = 2.1
        return []
    monkeypatch.setattr(recovery, "_group_rows", probe)
    result = setup.execute()
    assert result["reason"] == "RECOVERY_TIME_BUDGET_EXCEEDED" and result["status"] == "FAILED"
    assert result["after_sha256"] == recovery.JOURNAL_SHA256
    assert setup.writes == ([] if phase < 2 else [{"state": "QUIESCENT", "pgid": None}, recovery.JOURNAL_STATE])


def test_late_os_lock_cleanup_marks_failed_without_unguarded_rewrite(monkeypatch, setup):
    def flock(_fd, mode):
        if mode == recovery.fcntl.LOCK_UN:
            setup.clock[0] = 2.1
    monkeypatch.setattr(recovery.fcntl, "flock", flock)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["reason"] == "RECOVERY_TIME_BUDGET_EXCEEDED"
    assert result["cleanup_complete"] is True and result["journal_mutated"] is True
    assert setup.writes == [{"state": "QUIESCENT", "pgid": None}], "no rollback outside the held flock"
    assert result["after_sha256"] == hashlib.sha256(raw_json({"state": "QUIESCENT", "pgid": None})).hexdigest()


def test_last_clock_crossing_before_terminal_overrides_complete(monkeypatch, setup):
    calls = [0]
    original = recovery._deadline
    def deadline(started):
        remaining = original(started)
        # The last ordinary check is after all admission cleanup. Its own
        # successful sample must not authorize a later over-cap terminal.
        if getattr(setup.journal, "closed_by_recovery", False):
            calls[0] += 1
            setup.clock[0] = 2.1
        return remaining
    monkeypatch.setattr(recovery, "_deadline", deadline)
    result = setup.execute()
    assert calls[0] == 1 and result["status"] == "FAILED" and result["time_budget_exceeded"] is True
    assert setup.writes == [{"state": "QUIESCENT", "pgid": None}]


@pytest.mark.parametrize("phase", [0, 2])
def test_keyboard_interrupt_is_retained_and_rolls_back_only_under_lock(monkeypatch, setup, phase):
    calls = [0]
    def probe(*_):
        call = calls[0]; calls[0] += 1
        if call == phase:
            raise KeyboardInterrupt()
        return []
    monkeypatch.setattr(recovery, "_group_rows", probe)
    held = [False]
    def flock(_fd, mode):
        held[0] = mode != recovery.fcntl.LOCK_UN
    monkeypatch.setattr(recovery.fcntl, "flock", flock)
    original_record = recovery.shared._record
    def record(lock, state):
        assert held[0], "all writes, including rollback, require the same OS lock"
        return original_record(lock, state)
    monkeypatch.setattr(recovery.shared, "_record", record)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["exception_type"] == "KeyboardInterrupt"
    assert result["cleanup_complete"] is True and not held[0]
    assert setup.writes == ([] if phase == 0 else [{"state": "QUIESCENT", "pgid": None}, recovery.JOURNAL_STATE])


def test_partial_quiescent_write_failure_restores_under_lock(monkeypatch, setup):
    original_record = recovery.shared._record
    def record(lock, state):
        if state["state"] == "QUIESCENT":
            lock.seek(0); lock.write(b'{"pgid":null,'); lock.truncate()
            raise OSError("authored partial write")
        return original_record(lock, state)
    monkeypatch.setattr(recovery.shared, "_record", record)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["journal_mutated"] is True
    assert result["quarantine_restored"] is True and result["after_sha256"] == recovery.JOURNAL_SHA256


def test_rollback_failure_is_explicit_and_never_completed(monkeypatch, setup):
    original_record = recovery.shared._record
    def record(lock, state):
        if state["state"] == "QUARANTINED":
            raise PermissionError("authored rollback denial")
        return original_record(lock, state)
    monkeypatch.setattr(recovery.shared, "_record", record)
    observations = iter([[], [], [(23559, 1)]])
    monkeypatch.setattr(recovery, "_group_rows", lambda *_: next(observations))
    result = setup.execute()
    assert result["status"] == "FAILED" and result["quarantine_restored"] is False
    assert result["rollback_exception_type"] == "PermissionError" and result["cleanup_complete"] is True


def test_cleanup_failure_is_terminal_after_slot_release(monkeypatch, setup):
    def flock(_fd, mode):
        if mode == recovery.fcntl.LOCK_UN:
            raise OSError("authored unlock failure")
    monkeypatch.setattr(recovery.fcntl, "flock", flock)
    result = setup.execute()
    assert result["status"] == "FAILED" and result["reason"] == "RECOVERY_ADMISSION_CLEANUP_FAILED"
    assert result["cleanup_complete"] is False and result["cleanup_errors"] == ["OS_LOCK_RELEASE_FAILED"]


def test_existing_receipt_is_preserved_before_any_access(setup):
    setup.receipt_path.write_bytes(b"existing terminal\n")
    with pytest.raises(FileExistsError):
        setup.execute()
    assert setup.receipt_path.read_bytes() == b"existing terminal\n" and not setup.opens and not setup.probes


def test_freeze_reads_only_mocked_named_sources_and_is_exclusive(monkeypatch, tmp_path):
    pins = {name: "a" * 64 for name in recovery.SOURCE_PATHS}
    pins[recovery.SOURCE_PATHS[2]] = recovery.SHARED_WORKER_SHA256
    pins[recovery.SOURCE_PATHS[3]] = recovery.COMMON_CONTRACT_SHA256
    monkeypatch.setattr(recovery, "_source_pins", lambda: pins)
    def forbidden(*_):
        raise AssertionError("freeze must not read journal/old artifacts")
    for name in ("_read_capped", "_open_journal", "_verify_owned_record", "_group_rows"):
        monkeypatch.setattr(recovery, name, forbidden)
    path = tmp_path / "freeze.json"
    value = recovery.freeze(str(path), receipt_path=str(tmp_path / "receipt.jsonl"), author="author",
        owned_record_sha256=recovery.OWNED_RECORD_SHA256)
    assert value["pgid"] == 23559 and value["journal_state"] == recovery.JOURNAL_STATE
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        recovery.freeze(str(path), receipt_path=str(tmp_path / "receipt.jsonl"), author="author",
            owned_record_sha256=recovery.OWNED_RECORD_SHA256)
    assert path.read_bytes() == before


def owned_events():
    freeze = {"record_path": "/private" + recovery.OWNED_RECORD, "directory": "/private" + recovery.OWNED_DIRECTORY,
        "runner_path": recovery.OWNED_RUNNER, "uninterpreted_frozen_key": "retained"}
    return [{"event": "START", "freeze": freeze, "review": {}, "started_at": "authored-time"},
        {"event": "TERMINAL", "status": "FAILED", "exception_type": "PermissionError",
         "reason": "[Errno 1] Operation not permitted: '/bin/ps'", "cases": [], "freeze": dict(freeze),
         "physical_inputs_read": False, "manufacturing_ready": False, "elapsed_seconds": .008,
         "finished_at": "authored-time"}]


def test_owned_record_parser_accepts_only_declared_failure_not_native_extent():
    recovery._owned_events(b"\n".join(raw_json(row) for row in owned_events()))


@pytest.mark.parametrize("key,value", [("event", "START"), ("status", "COMPLETE"),
    ("exception_type", "TimeoutError"), ("reason", "different failure"), ("cases", [{}]),
    ("cases", False), ("physical_inputs_read", True), ("manufacturing_ready", True)])
def test_owned_record_wrong_terminal_refuses(key, value):
    events = owned_events(); events[1][key] = value
    with pytest.raises(recovery.RecoveryError):
        recovery._owned_events(b"\n".join(raw_json(row) for row in events))


def test_owned_record_duplicate_or_mismatched_freeze_refuses():
    events = owned_events(); events.append(events[1])
    with pytest.raises(recovery.RecoveryError):
        recovery._owned_events(b"\n".join(raw_json(row) for row in events))
    events = owned_events(); events[1]["freeze"]["directory"] = "/tmp/not-r3"
    with pytest.raises(recovery.RecoveryError):
        recovery._owned_events(b"\n".join(raw_json(row) for row in events))


def test_owned_record_raw_sha_checked_before_parser(monkeypatch):
    monkeypatch.setattr(recovery, "_read_capped", lambda *_: b"authored mismatch")
    with pytest.raises(recovery.RecoveryError, match="OWNED_RECORD_HASH_MISMATCH"):
        recovery._verify_owned_record({"owned_generation_record": recovery.OWNED_RECORD,
            "owned_generation_record_sha256": recovery.OWNED_RECORD_SHA256})


def test_bounded_ps_parser_has_same_zombie_policy_and_fixed_scope(monkeypatch):
    monkeypatch.setattr(recovery.time, "monotonic", lambda: 0)
    def ps(command, **kwargs):
        assert command == ["/bin/ps", "-axo", "pid=,pgid=,rss=,stat="]
        assert kwargs["timeout"] == .5 and kwargs["check"] is True
        return SimpleNamespace(stdout=b"23559 23559 100 Z+\n23560 23559 200 S\n1 1 20 S\n", stderr=b"")
    monkeypatch.setattr(recovery.subprocess, "run", ps)
    assert recovery._group_rows(23559, 0) == [(23560, 200)]


@pytest.mark.parametrize("raw", [b"bad row", b"1 23559 -1 S\n", b"1 False 0 S\n", b"1 23559 0\n", b"\xff"])
def test_bounded_ps_parser_malformed_output_never_claims_empty(monkeypatch, raw):
    monkeypatch.setattr(recovery.time, "monotonic", lambda: 0)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=raw, stderr=b""))
    with pytest.raises((recovery.RecoveryError, UnicodeError)):
        recovery._group_rows(23559, 0)


def test_bounded_probe_remaining_timeout_and_oversized_output(monkeypatch):
    monkeypatch.setattr(recovery.time, "monotonic", lambda: 1.8)
    def ps(*_args, **kwargs):
        assert 0 < kwargs["timeout"] <= .21
        return SimpleNamespace(stdout=b"", stderr=b"x" * (recovery.LIMITS["max_monitor_bytes"] + 1))
    monkeypatch.setattr(recovery.subprocess, "run", ps)
    with pytest.raises(recovery.RecoveryError, match="MONITOR_OUTPUT_BUDGET_EXCEEDED"):
        recovery._group_rows(23559, 0)


def test_journal_open_never_creates_or_follows_symlink(monkeypatch):
    def open_mock(path, flags):
        assert path == recovery.JOURNAL_PATH
        assert flags == recovery.os.O_RDWR | recovery.os.O_NOFOLLOW
        raise PermissionError("authored-only")
    monkeypatch.setattr(recovery.os, "open", open_mock)
    with pytest.raises(PermissionError):
        recovery._open_journal(recovery.JOURNAL_PATH)


def test_journal_identity_checks_actual_owner_regular_file_and_inode(monkeypatch):
    info = SimpleNamespace(st_mode=0o100600, st_uid=recovery.os.getuid(), st_dev=1, st_ino=5)
    monkeypatch.setattr(recovery.os, "fstat", lambda _: info)
    monkeypatch.setattr(recovery.os, "stat", lambda *_args, **_kwargs: info)
    recovery._journal_identity(Journal(), recovery.JOURNAL_PATH)
    info.st_uid += 1
    with pytest.raises(recovery.RecoveryError, match="JOURNAL_NOT_CURRENT_UID_REGULAR"):
        recovery._journal_identity(Journal(), recovery.JOURNAL_PATH)
