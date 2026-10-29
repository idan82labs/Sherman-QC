"""One independently PRE-gated recovery of the owned r3 admission journal.

This is NOT a generic unlock, retry, native worker, kill or journal deletion.
Freeze does not read the journal or old generation artifacts. Recovery is fixed
to the declared r3 failure, three bounded PGID observations and exact journal
bytes. Identity strings are declarations, not cryptographic reviewer identity.
No journal, artifact, source hashing or process probe occurs at module import.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from backend import vendor_unfold_worker_v2 as shared
from backend.vendor_unfold_contract_v2 import strict_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "SHERMAN_VENDOR_UNFOLD_ADMISSION_RECOVERY_V1"
PRE_SCHEMA = "SHERMAN_VENDOR_UNFOLD_ADMISSION_RECOVERY_PRE_V1"
PGID = 23559
JOURNAL_PATH = "/tmp/sherman-vendor-unfold-v2.lock"
JOURNAL_SHA256 = "6c1408694298656c487b4bf9e40ba8464da5ef3e7541e86eb61504e31a305a59"
JOURNAL_STATE = {"pgid": PGID, "state": "QUARANTINED"}
OWNED_RECORD = "/tmp/sherman-unfold-runtime-v3-generation-run-r3-20260929.jsonl"
OWNED_RECORD_SHA256 = "4fe560b55d97b270add366018238907ed516648e7a8db60fe038aa0c6d5d83a1"
OWNED_PROTOCOL = "/tmp/sherman-unfold-runtime-v3-generation-freeze-r3-20260929.json"
OWNED_DIRECTORY = "/tmp/sherman-unfold-runtime-v3-synthetic-r3-20260929"
OWNED_RUNNER = str(ROOT / "scripts/run_vendor_unfold_runtime_smoke_v2.py")
REASON = "OWNED_R3_GENERATION_PS_PERMISSION_FAILURE_WITH_QUARANTINED_ADMISSION"
SHARED_WORKER_SHA256 = "713b84499a1049c166000eea8e23e84385c1b1840234157ea578abc7874c9886"
COMMON_CONTRACT_SHA256 = "cbcd792106750a866b6fe960e6ff8046cd4c68a842b40c8acc01f551615be76f"
SOURCE_PATHS = (
    "scripts/recover_vendor_unfold_admission_v1.py",
    "tests/test_recover_vendor_unfold_admission_v1.py",
    "backend/vendor_unfold_worker_v2.py",
    "backend/vendor_unfold_contract_v2.py",
)
LIMITS = {"aggregate_seconds": 2.0, "empty_check_separation_seconds": 0.1,
    "probe_timeout_seconds": 0.5, "max_record_bytes": 2_000_000,
    "max_monitor_bytes": 2_000_000, "max_source_bytes": 100_000,
    "max_protocol_bytes": 100_000, "max_receipt_bytes": 100_000}
PROTOCOL_KEYS = {"schema", "author", "journal_path", "journal_sha256", "journal_state", "pgid",
    "owned_generation_record", "owned_generation_record_sha256", "owned_generation_protocol",
    "declared_reason", "receipt_path", "source_pins", "limits"}
PRE_KEYS = {"schema", "disposition", "author", "reviewer", "blocking_findings",
    "protocol_sha256", "source_bundle_sha256", "journal_sha256"}


class RecoveryError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise RecoveryError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value):
    return type(value) is str and 0 < len(value) <= 200 and bool(value.strip())


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _tmp_alias(value, expected):
    return type(value) is str and value in {expected, "/private" + expected}


def _absolute(value):
    return type(value) is str and 0 < len(value) <= 1024 and Path(value).is_absolute()


def _regular_owned(info):
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()


def _read_capped(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        _require(_regular_owned(os.fstat(stream.fileno())), "FILE_NOT_CURRENT_UID_REGULAR")
        raw = stream.read(maximum + 1)
    _require(0 < len(raw) <= maximum, "FILE_BUDGET_EXCEEDED")
    return raw


def _source_pins():
    pins = {name: _sha(_read_capped(ROOT / name, LIMITS["max_source_bytes"])) for name in SOURCE_PATHS}
    _require(pins[SOURCE_PATHS[2]] == SHARED_WORKER_SHA256
             and pins[SOURCE_PATHS[3]] == COMMON_CONTRACT_SHA256, "SHARED_SOURCE_NOT_PINNED")
    return pins


def _exclusive(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")


def _append(stream, value):
    raw = _json(value) + b"\n"
    _require(stream.tell() + len(raw) <= LIMITS["max_receipt_bytes"], "RECEIPT_BUDGET_EXCEEDED")
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def _validate_protocol(value, receipt_path):
    _require(type(value) is dict and set(value) == PROTOCOL_KEYS, "INVALID_PROTOCOL_SCHEMA")
    _require(value["schema"] == SCHEMA and _identity(value["author"]), "INVALID_PROTOCOL_IDENTITY")
    _require(_tmp_alias(value["journal_path"], JOURNAL_PATH)
             and value["journal_sha256"] == JOURNAL_SHA256
             and type(value["pgid"]) is int and value["pgid"] == PGID
             and type(value["journal_state"]) is dict and set(value["journal_state"]) == {"pgid", "state"}
             and type(value["journal_state"]["pgid"]) is int and value["journal_state"] == JOURNAL_STATE,
             "NOT_OWNED_QUARANTINED_JOURNAL")
    _require(_tmp_alias(value["owned_generation_record"], OWNED_RECORD)
             and value["owned_generation_record_sha256"] == OWNED_RECORD_SHA256
             and _tmp_alias(value["owned_generation_protocol"], OWNED_PROTOCOL)
             and value["declared_reason"] == REASON, "NOT_OWNED_R3_FAILURE")
    _require(_absolute(value["receipt_path"]) and value["receipt_path"] == str(receipt_path), "RECEIPT_PATH_MISMATCH")
    _require(type(value["limits"]) is dict and _json(value["limits"]) == _json(LIMITS), "CHANGED_RECOVERY_LIMITS")
    pins = value["source_pins"]
    _require(type(pins) is dict and set(pins) == set(SOURCE_PATHS)
             and all(_digest(v) for v in pins.values()), "INVALID_SOURCE_PINS")
    _require(pins[SOURCE_PATHS[2]] == SHARED_WORKER_SHA256
             and pins[SOURCE_PATHS[3]] == COMMON_CONTRACT_SHA256, "SHARED_SOURCE_NOT_PINNED")


def _validate_pre(pre, protocol, raw):
    _require(type(pre) is dict and set(pre) == PRE_KEYS, "INVALID_PRE_SCHEMA")
    _require(pre["schema"] == PRE_SCHEMA and pre["disposition"] == "GREEN"
             and type(pre["blocking_findings"]) is list and pre["blocking_findings"] == [], "PRE_NOT_GREEN")
    _require(_identity(pre["author"]) and _identity(pre["reviewer"])
             and pre["author"] == protocol["author"] and pre["author"] != pre["reviewer"], "PRE_NOT_INDEPENDENT")
    _require(pre["protocol_sha256"] == _sha(raw)
             and pre["source_bundle_sha256"] == _sha(_json(protocol["source_pins"]))
             and pre["journal_sha256"] == JOURNAL_SHA256, "PRE_BINDING_MISMATCH")


def _owned_events(raw):
    lines = raw.splitlines()
    _require(len(lines) == 2, "INVALID_OWNED_RECORD_EVENTS")
    start, terminal = (strict_json(line) for line in lines)
    _require(type(start) is dict and start.get("event") == "START"
             and type(terminal) is dict and terminal.get("event") == "TERMINAL"
             and terminal.get("status") == "FAILED" and terminal.get("exception_type") == "PermissionError"
             and type(terminal.get("reason")) is str and "'/bin/ps'" in terminal["reason"]
             and type(terminal.get("cases")) is list and terminal["cases"] == []
             and terminal.get("physical_inputs_read") is False and terminal.get("manufacturing_ready") is False,
             "NOT_OWNED_PS_FAILURE_TERMINAL")
    freeze = start.get("freeze")
    _require(type(freeze) is dict and terminal.get("freeze") == freeze
             and _tmp_alias(freeze.get("record_path"), OWNED_RECORD)
             and _tmp_alias(freeze.get("directory"), OWNED_DIRECTORY)
             and freeze.get("runner_path") == OWNED_RUNNER, "OWNED_RECORD_FREEZE_MISMATCH")


def _verify_owned_record(protocol):
    raw = _read_capped(protocol["owned_generation_record"], LIMITS["max_record_bytes"])
    _require(_sha(raw) == protocol["owned_generation_record_sha256"], "OWNED_RECORD_HASH_MISMATCH")
    _owned_events(raw)


def freeze(protocol_path, *, receipt_path, author, owned_record_sha256):
    """Create a NEW protocol from fixed scope; read only the four named sources."""
    _require(_absolute(str(protocol_path)) and _absolute(str(receipt_path)) and _identity(author), "INVALID_FREEZE_ARGUMENTS")
    _require(owned_record_sha256 == OWNED_RECORD_SHA256, "NOT_OWNED_R3_FAILURE")
    _require(str(protocol_path) != str(receipt_path), "PROTOCOL_RECEIPT_PATH_COLLISION")
    value = {"schema": SCHEMA, "author": author, "journal_path": JOURNAL_PATH,
        "journal_sha256": JOURNAL_SHA256, "journal_state": dict(JOURNAL_STATE), "pgid": PGID,
        "owned_generation_record": OWNED_RECORD, "owned_generation_record_sha256": owned_record_sha256,
        "owned_generation_protocol": OWNED_PROTOCOL, "declared_reason": REASON,
        "receipt_path": str(receipt_path), "source_pins": _source_pins(), "limits": dict(LIMITS)}
    _validate_protocol(value, receipt_path)
    with _exclusive(protocol_path) as stream:
        stream.write(_json(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    return value


def _deadline(started):
    remaining = LIMITS["aggregate_seconds"] - (time.monotonic() - started)
    _require(remaining > 0, "RECOVERY_TIME_BUDGET_EXCEEDED")
    return remaining


def _group_rows(pgid, started):
    """Same non-zombie PGID policy as V2, with a remaining-budget timeout."""
    timeout = min(LIMITS["probe_timeout_seconds"], _deadline(started))
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss=,stat="],
        capture_output=True, timeout=timeout, check=True)
    _deadline(started)
    _require(type(result.stdout) is bytes and type(result.stderr) is bytes
             and len(result.stdout) + len(result.stderr) <= LIMITS["max_monitor_bytes"], "MONITOR_OUTPUT_BUDGET_EXCEEDED")
    rows = []
    for line in result.stdout.decode("ascii", errors="strict").splitlines():
        values = line.split()
        _require(len(values) == 4, "INVALID_MONITOR_ROW")
        _require(all(value.isdigit() for value in values[:3]) and values[3], "INVALID_MONITOR_ROW")
        pid, group, rss = map(int, values[:3])
        _require(pid > 0 and group > 0 and rss >= 0, "INVALID_MONITOR_ROW")
        if group == pgid and not values[3].startswith("Z"):
            rows.append((pid, rss))
    _require(len(rows) <= 1024, "MONITOR_GROUP_ROW_BUDGET_EXCEEDED")
    return rows


def _open_journal(path):
    return os.fdopen(os.open(path, os.O_RDWR | os.O_NOFOLLOW), "r+b")


def _journal_bytes(lock):
    lock.seek(0)
    raw = lock.read(1025)
    _require(0 < len(raw) <= 1024, "INVALID_JOURNAL_BYTES")
    return raw


def _journal_identity(lock, path):
    info = os.fstat(lock.fileno())
    _require(_regular_owned(info), "JOURNAL_NOT_CURRENT_UID_REGULAR")
    current = os.stat(path, follow_symlinks=False)
    _require(stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino), "JOURNAL_PATH_REPLACED")


def _utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _admission(path, summary, started):
    _deadline(started)
    _require(shared._slot.acquire(blocking=False), "SHARED_SLOT_BUSY")
    lock, held, cleanup_errors = None, False, []
    try:
        _deadline(started)
        lock = _open_journal(path)
        _journal_identity(lock, path)
        _deadline(started)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryError("SHARED_OS_LOCK_BUSY") from exc
        held = True
        yield lock
    finally:
        if held:
            try:
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BaseException:
                cleanup_errors.append("OS_LOCK_RELEASE_FAILED")
        if lock is not None:
            try:
                lock.close()
            except BaseException:
                cleanup_errors.append("JOURNAL_CLOSE_FAILED")
        try:
            shared._slot.release()
        except BaseException:
            cleanup_errors.append("SLOT_RELEASE_FAILED")
        summary["cleanup_complete"] = not cleanup_errors
        if cleanup_errors:
            summary["cleanup_errors"] = cleanup_errors
            raise RecoveryError("RECOVERY_ADMISSION_CLEANUP_FAILED")


def recover(protocol_path, pre_path, *, receipt_path):
    """Run only when independently authorized; always retain a new receipt.

    PRE/source/owned-record verification precedes the 2-second mutation clock.
    Every terminal record follows slot/OS-lock release. Failure after an
    attempted write re-quarantines under the still-held lock, never kills.
    """
    _require(all(_absolute(str(path)) for path in (protocol_path, pre_path, receipt_path)), "INVALID_RECOVERY_PATH")
    summary = {"event": "TERMINAL", "status": "FAILED", "reason": "UNSTARTED",
        "declared_reason": REASON, "pgid": PGID, "journal_path": JOURNAL_PATH,
        "before_sha256": None, "after_sha256": None, "journal_mutated": False,
        "observations": [], "action_time_utc": None, "native_activity_extent": "UNOBSERVED",
        "manufacturing_ready": False, "release_allowed": False, "elapsed_seconds": None,
        "protocol_sha256": None, "pre_sha256": None, "source_bundle_sha256": None,
        "cleanup_complete": False, "quarantine_restored": None}
    started = None
    # Exclusive creation must precede ALL possible journal mutations. Existing
    # receipts are never reopened, overwritten, deleted or retried in place.
    with _exclusive(receipt_path) as receipt:
        _append(receipt, {"event": "START", "schema": SCHEMA, "started_at": _utc(),
            "protocol_path": str(protocol_path), "pre_path": str(pre_path), "receipt_path": str(receipt_path)})
        try:
            raw = _read_capped(protocol_path, LIMITS["max_protocol_bytes"])
            protocol = strict_json(raw)
            _validate_protocol(protocol, receipt_path)
            pre_raw = _read_capped(pre_path, LIMITS["max_protocol_bytes"])
            _validate_pre(strict_json(pre_raw), protocol, raw)
            _require(_source_pins() == protocol["source_pins"], "SOURCE_CHANGED_BEFORE_RECOVERY")
            _verify_owned_record(protocol)
            summary.update({"protocol_sha256": _sha(raw), "pre_sha256": _sha(pre_raw),
                "source_bundle_sha256": _sha(_json(protocol["source_pins"]))})
            _require(shared.ADMISSION_LOCK_PATH == JOURNAL_PATH, "SHARED_ADMISSION_PATH_CHANGED")
            started = time.monotonic()
            with _admission(protocol["journal_path"], summary, started) as lock:
                try:
                    before = _journal_bytes(lock)
                    summary["before_sha256"] = _sha(before)
                    _require(summary["before_sha256"] == JOURNAL_SHA256, "JOURNAL_HASH_CHANGED")
                    state = strict_json(before)
                    _require(type(state) is dict and set(state) == {"pgid", "state"}
                             and type(state["pgid"]) is int and state == JOURNAL_STATE, "JOURNAL_STATE_CHANGED")

                    def observe():
                        observation = {"pgid": PGID, "ordinal": len(summary["observations"]) + 1,
                            "elapsed_seconds": time.monotonic() - started}
                        try:
                            rows = _group_rows(PGID, started)
                        except BaseException as exc:
                            observation.update({"status": "ERROR", "exception_type": type(exc).__name__})
                            summary["observations"].append(observation)
                            raise
                        observation.update({"status": "OBSERVED", "empty": rows == [], "rows": [list(row) for row in rows],
                            "finished_elapsed_seconds": time.monotonic() - started})
                        summary["observations"].append(observation)
                        _require(rows == [], "OWNED_PROCESS_GROUP_NOT_EMPTY")
                        _deadline(started)

                    observe()
                    _require(_deadline(started) > LIMITS["empty_check_separation_seconds"], "RECOVERY_TIME_BUDGET_EXCEEDED")
                    time.sleep(LIMITS["empty_check_separation_seconds"])
                    _deadline(started)
                    observe()
                    _require(_source_pins() == protocol["source_pins"], "SOURCE_CHANGED_BEFORE_WRITE")
                    _deadline(started)
                    _journal_identity(lock, protocol["journal_path"])
                    _require(_journal_bytes(lock) == before, "JOURNAL_BYTES_CHANGED_UNDER_LOCK")
                    _deadline(started)
                    summary["journal_mutated"] = True
                    summary["action_time_utc"] = _utc()
                    shared._record(lock, {"state": "QUIESCENT", "pgid": None})
                    _deadline(started)
                    observe()
                    after = _journal_bytes(lock)
                    summary["after_sha256"] = _sha(after)
                    _require(after == _json({"state": "QUIESCENT", "pgid": None}), "QUIESCENT_WRITE_NOT_RETAINED")
                    _require(_source_pins() == protocol["source_pins"], "SOURCE_CHANGED_AFTER_WRITE")
                    _journal_identity(lock, protocol["journal_path"])
                    _require(_journal_bytes(lock) == after, "QUIESCENT_BYTES_CHANGED_UNDER_LOCK")
                    _deadline(started)
                    summary.update({"status": "COMPLETE", "reason": "OWNED_R3_ADMISSION_RECOVERED"})
                except BaseException:
                    if summary["journal_mutated"]:
                        try:
                            shared._record(lock, dict(JOURNAL_STATE))
                            summary["after_sha256"] = _sha(_journal_bytes(lock))
                            _journal_identity(lock, protocol["journal_path"])
                            summary["quarantine_restored"] = summary["after_sha256"] == JOURNAL_SHA256
                        except BaseException as rollback_error:
                            summary["quarantine_restored"] = False
                            summary["rollback_exception_type"] = type(rollback_error).__name__
                    else:
                        try:
                            summary["after_sha256"] = _sha(_journal_bytes(lock))
                        except BaseException:
                            summary["after_sha256"] = None
                    raise
            _deadline(started)
        except BaseException as exc:
            summary.update({"status": "FAILED", "reason": str(exc) if isinstance(exc, RecoveryError) else "RECOVERY_OPERATION_FAILED",
                "exception_type": type(exc).__name__})
        finally:
            if started is not None:
                summary["elapsed_seconds"] = time.monotonic() - started
                if summary["elapsed_seconds"] >= LIMITS["aggregate_seconds"]:
                    summary["time_budget_exceeded"] = True
                    if summary["status"] == "COMPLETE":
                        summary.update({"status": "FAILED", "reason": "RECOVERY_TIME_BUDGET_EXCEEDED",
                            "exception_type": "RecoveryError"})
            # Do not reopen/rewrite the journal here: OS lock and slot have
            # already been released and a newer admitted owner may be active.
            summary["finished_at"] = _utc()
            _append(receipt, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("--protocol", required=True)
    create.add_argument("--receipt", required=True)
    create.add_argument("--author", required=True)
    create.add_argument("--owned-record-sha256", required=True)
    execute = commands.add_parser("recover")
    execute.add_argument("--protocol", required=True)
    execute.add_argument("--pre", required=True)
    execute.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.protocol, receipt_path=args.receipt, author=args.author, owned_record_sha256=args.owned_record_sha256)
        return 0
    result = recover(args.protocol, args.pre, receipt_path=args.receipt)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
