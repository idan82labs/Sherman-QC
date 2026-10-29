"""Independent tiny-filesystem contracts; no runner, native or scientific inputs.

Authoring is not execution. Prospective gate: 60wall/45CPU/1GiB/1MiB. All filesystem
actions below stay inside pytest-owned, canonically resolved tmp_path. Faults and
path changes are explicit controlled injections, not hostile-code/atomic-snapshot
qualification. The caller still must stop writers and inventory its whole tree.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1]/"scripts/gold11_terminal_artifacts_v4.py"
TERMINALS = ("SCORING_INPUT.json", "COMPLETED.json")


@pytest.fixture
def kernel(monkeypatch):
    spec = importlib.util.spec_from_file_location("_invented_terminal_artifacts_v4", SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owned(tmp_path):
    # macOS /var is a symlink; valid cases use the canonical /private/... path.
    return tmp_path.resolve()


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _spec(blob):
    return dict(max_bytes=len(blob), sha256=_sha(blob), bytes=len(blob))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _ensure(kernel, directory, name, blob, make=None, validator=None):
    return kernel.ensure_terminal_record_v4(directory, name,
        (lambda: blob) if make is None else make,
        (lambda value: value == blob) if validator is None else validator,
        max_bytes=len(blob))


def _inventory(kernel, directory, blobs, **overrides):
    options = dict(max_entries=len(blobs), max_total_bytes=sum(map(len, blobs.values())))
    options.update(overrides)
    return kernel.inventory_directory_v4(directory, {name: _spec(b) for name, b in blobs.items()}, **options)


def _complete(row):
    assert row["status"] == "COMPLETE" and row["complete"] is True and row["reason"] is None
    assert row["close_error_types"] == [] and row["directory_path_binding_rechecked"] is True


def _unavailable(row):
    assert row["status"] == "UNAVAILABLE" and row["complete"] is False and row["reason"]
    assert "injected private text" not in json.dumps(row)


def _failure(kernel, operation, code=None):
    with pytest.raises(kernel.ArtifactFailureV4) as caught:
        operation()
    if code is not None:
        assert caught.value.code == code
    _unavailable(caught.value.evidence)
    assert str(caught.value) == caught.value.code
    return caught.value.evidence


def _snapshot(directory):
    # Reading may alter atime; content/inode/link/name/write timestamps may not.
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    return {p.name: (tuple(getattr(p.lstat(), f) for f in fields), p.read_bytes())
            for p in directory.iterdir()}


def _never(*args, **kwargs):
    raise AssertionError("unexpected factory/write/content read")


@pytest.mark.parametrize("blob", (b"", b"{}", b"x"*(65536+7)))
def test_true_new_publication_has_one_regular_target_no_stage_and_exact_hash(kernel, owned, blob):
    row = kernel.publish_bytes_exclusive_v4(owned, "record", blob, max_bytes=len(blob))
    _complete(row)
    assert sorted(p.name for p in owned.iterdir()) == ["record"]
    assert (owned/"record").read_bytes() == blob
    assert row["sha256"] == _sha(blob) and row["bytes"] == row["written_bytes"] == len(blob)
    assert row["identity"]["inode"] == (owned/"record").stat().st_ino
    assert row["identity"]["links"] == 1 and row["written_bytes_exact"] is True
    assert row["target_linked"] and row["successful_stage_removed"]
    assert row["target_directory_synced"] and row["final_directory_synced"]
    assert row["write_calls"] == (len(blob)+65535)//65536
    _complete(_inventory(kernel, owned, {"record": blob}))


def test_existing_terminal_never_calls_factory_or_any_write_and_preserves_inode_names(kernel, owned, monkeypatch):
    blob = b'{"status":"RETAINED"}'
    (owned/"record").write_bytes(blob)
    before, directory_before, validations = _snapshot(owned), owned.stat(), []
    def validate(value):
        validations.append(value)
        return value == blob
    with monkeypatch.context() as patch:
        for name in ("write", "link", "unlink", "fsync"):
            patch.setattr(kernel.os, name, _never)
        patch.setattr(kernel.secrets, "token_hex", _never)
        row = _ensure(kernel, owned, "record", blob, _never, validate)
    _complete(row)
    assert validations == [blob] and row["disposition"] == "PREEXISTING_VALID_NO_WRITE"
    assert row["publication_attempted"] is False and row["wrote_new_bytes"] is False
    assert row["read_bytes"] == len(blob) and _snapshot(owned) == before
    assert owned.stat().st_mtime_ns == directory_before.st_mtime_ns
    assert owned.stat().st_ctime_ns == directory_before.st_ctime_ns


def test_twenty_two_existing_case_directories_repeat_fill_never_creates_forty_four_extras(kernel, owned):
    records, before = {}, {}
    for i in range(22):
        directory = owned/f"invented-{i:02d}"; directory.mkdir()
        records[directory] = {name: _encoded(dict(case=i, kind=name, status="ORIGINAL_TERMINAL")) for name in TERMINALS}
        for name, blob in records[directory].items(): (directory/name).write_bytes(blob)
        before[directory] = _snapshot(directory)
    for _ in range(2):
        for directory, blobs in records.items():
            for name, blob in blobs.items():
                row = _ensure(kernel, directory, name, blob, _never)
                _complete(row)
                assert row["disposition"] == "PREEXISTING_VALID_NO_WRITE"
            inventory = _inventory(kernel, directory, blobs)
            _complete(inventory)
            assert inventory["initial_names"] == inventory["final_names"] == sorted(TERMINALS)
            assert len(inventory["entries"]) == 2 and inventory["unexpected_names"] == []
            assert _snapshot(directory) == before[directory]
    expected = {f"invented-{i:02d}" for i in range(22)}
    expected |= {f"invented-{i:02d}/{name}" for i in range(22) for name in TERMINALS}
    assert {str(p.relative_to(owned)) for p in owned.rglob("*")} == expected
    assert sum(p.is_file() for p in owned.rglob("*")) == 44


def test_mixed_existing_and_missing_terminal_records_publish_only_missing_once(kernel, owned):
    originals, made, records = {}, [], {}
    for i in range(4):
        d = owned/f"case-{i}"; d.mkdir()
        records[d] = {name: _encoded(dict(case=i, kind=name)) for name in TERMINALS}
        for j, (name, blob) in enumerate(records[d].items()):
            if (i+j)%2 == 0:
                (d/name).write_bytes(blob); originals[d/name] = (d/name).stat().st_ino
    for d, blobs in records.items():
        for name, blob in blobs.items():
            def make(d=d, name=name, blob=blob):
                made.append((d, name)); return blob
            row = _ensure(kernel, d, name, blob, make)
            _complete(row)
            assert row["wrote_new_bytes"] is ((d/name) not in originals)
        _complete(_inventory(kernel, d, blobs))
        for name, blob in blobs.items(): _complete(_ensure(kernel, d, name, blob, _never))
    assert len(made) == 4 and len(set(made)) == 4
    assert all(p.stat().st_ino == inode for p, inode in originals.items())
    assert sum(p.is_file() for p in owned.rglob("*")) == 8


def test_link_time_collision_preserves_distinct_same_byte_partial_and_vetoes_inventory(kernel, owned, monkeypatch):
    blob = b"same retained bytes"; real_link = kernel.os.link
    def collision(source, target, **kwargs):
        (owned/target).write_bytes(blob)  # Simulated independently published winner.
        return real_link(source, target, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "link", collision)
        row = _failure(kernel, lambda: _ensure(kernel, owned, "record", blob), "LINK_TARGET_COLLISION_AFTER_STAGING")
    partial = owned/row["partial_name"]
    assert partial.read_bytes() == (owned/"record").read_bytes() == blob
    assert partial.stat().st_ino != (owned/"record").stat().st_ino
    assert partial.stat().st_nlink == (owned/"record").stat().st_nlink == 1
    assert row["publication_attempted"] is True and "wrote_new_bytes" not in row
    assert row["partial_created"] and row["written_bytes"] == len(blob) and not row["target_linked"]
    inventory = _inventory(kernel, owned, {"record": blob}, max_entries=2)
    _unavailable(inventory)
    assert inventory["unexpected_names"] == [partial.name]
    assert inventory["entries"][partial.name]["status"] == "UNDECLARED_REGULAR_FILE_METADATA_ONLY"
    assert inventory["entries"][partial.name]["sha256"] is None
    assert inventory["read_bytes"] == inventory["charged_read_bytes"] == len(blob)
    assert sorted(p.name for p in owned.iterdir()) == sorted(["record", partial.name])


@pytest.mark.parametrize("existing", (False, True))
@pytest.mark.parametrize("answer", (False, None, 1, "yes"))
def test_validator_requires_singleton_true_without_repair_or_factory_on_existing(kernel, owned, existing, answer):
    blob = b"valid-sized"; calls = []
    if existing: (owned/"record").write_bytes(blob)
    def make(): calls.append(1); return blob
    row = _failure(kernel, lambda: _ensure(kernel, owned, "record", blob, make, lambda _: answer),
        "EXISTING_RECORD_VALIDATION_FAILED" if existing else "NEW_RECORD_VALIDATION_FAILED")
    assert calls == ([] if existing else [1])
    assert sorted(p.name for p in owned.iterdir()) == (["record"] if existing else [])
    assert row["publication_attempted"] is False


@pytest.mark.parametrize("mode", ("factory_exception", "validator_exception", "bytearray", "oversize"))
def test_factory_and_validator_failures_are_typed_and_create_no_artifacts(kernel, owned, mode):
    def make():
        if mode == "factory_exception": raise RuntimeError("injected private text")
        return bytearray(b"ok") if mode == "bytearray" else b"long" if mode == "oversize" else b"ok"
    def validate(_):
        if mode == "validator_exception": raise ValueError("injected private text")
        return True
    _failure(kernel, lambda: kernel.ensure_terminal_record_v4(owned, "record", make, validate, max_bytes=2))
    assert list(owned.iterdir()) == []


def test_target_created_by_factory_is_not_overwritten_or_reclassified_preexisting(kernel, owned):
    def make():
        (owned/"record").write_bytes(b"winner"); return b"candidate"
    row = _failure(kernel, lambda: kernel.ensure_terminal_record_v4(owned, "record", make, lambda _: True,
        max_bytes=20), "TARGET_ALREADY_EXISTS_NO_WRITE")
    assert row["publication_attempted"] and row["partial_created"] is False
    assert (owned/"record").read_bytes() == b"winner" and [p.name for p in owned.iterdir()] == ["record"]


@pytest.mark.parametrize("which,phase,names,linked,removed", (
    (1, "FSYNC_STAGE", 1, False, False), (2, "FSYNC_LINKED_DIRECTORY", 2, True, False),
    (3, "FSYNC_FINAL_DIRECTORY", 1, True, True)))
def test_every_late_fsync_failure_is_unavailable_and_retains_actual_publication_state(kernel, owned, monkeypatch,
        which, phase, names, linked, removed):
    real, calls = kernel.os.fsync, []
    def fail(fd):
        calls.append(fd)
        if len(calls) == which: raise OSError("injected private text")
        return real(fd)
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "fsync", fail)
        row = _failure(kernel, lambda: kernel.publish_bytes_exclusive_v4(owned, "record", b"abc", max_bytes=3),
            "FILESYSTEM_OR_CALLBACK_EXCEPTION")
    assert row["phase"] == phase and row["written_bytes"] == 3 and row["written_bytes_exact"]
    assert row["target_linked"] is linked and row["successful_stage_removed"] is removed
    assert len(list(owned.iterdir())) == names
    assert (owned/"record").exists() is linked
    assert (owned/row["partial_name"]).exists() is (not removed)


def test_unlink_failure_preserves_both_hardlink_names_and_never_reports_complete(kernel, owned, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(PermissionError("injected private text")))
        row = _failure(kernel, lambda: kernel.publish_bytes_exclusive_v4(owned, "record", b"abc", max_bytes=3))
    partial = owned/row["partial_name"]
    assert row["target_linked"] and row["target_directory_synced"] and not row["successful_stage_removed"]
    assert partial.stat().st_ino == (owned/"record").stat().st_ino and partial.stat().st_nlink == 2
    assert len(list(owned.iterdir())) == 2


@pytest.mark.parametrize("mode", ("short", "write_then_raise", "invalid_bool", "invalid_large"))
def test_failed_stage_write_keeps_partial_and_honest_observed_byte_ledger(kernel, owned, monkeypatch, mode):
    real, calls = kernel.os.write, []
    def write(fd, block):
        calls.append(1)
        if mode == "short": return real(fd, block[:1])
        if mode == "write_then_raise":
            real(fd, block[:1]); raise OSError("injected private text")
        return True if mode == "invalid_bool" else len(block)+1
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "write", write)
        row = _failure(kernel, lambda: kernel.publish_bytes_exclusive_v4(owned, "record", b"abc", max_bytes=3))
    assert calls == [1] and row["write_calls"] == 1 and row["target_linked"] is False
    assert len(list(owned.iterdir())) == 1 and not (owned/"record").exists()
    assert (owned/row["partial_name"]).stat().st_size == (1 if mode in ("short", "write_then_raise") else 0)
    assert row["written_bytes"] == (1 if mode == "short" else 0)
    assert row["written_bytes_exact"] is (mode == "short")


@pytest.mark.parametrize("where", ("stage", "existing_file", "directory", "recheck"))
def test_close_failure_vetoes_success_without_deleting_retained_files(kernel, owned, monkeypatch, where):
    if where != "stage": (owned/"record").write_bytes(b"abc")
    real_open, real_close, real_fstat = kernel._open_directory, kernel.os.close, kernel.os.fstat
    opened, failed = [], []
    def open_directory(*args):
        fd = real_open(*args); opened.append(fd); return fd
    def close(fd):
        is_regular = stat.S_ISREG(real_fstat(fd).st_mode)
        selected = is_regular if where in ("stage", "existing_file") else (
            len(opened) == 2 and fd == opened[1] if where == "recheck" else
            len(opened) == 2 and fd == opened[0])
        result = real_close(fd)
        if selected and not failed:
            failed.append(fd); raise OSError("injected private text")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_open_directory", open_directory)
        patch.setattr(kernel.os, "close", close)
        row = _failure(kernel, lambda: _ensure(kernel, owned, "record", b"abc"),
            {"stage": "STAGING_CLOSE_FAILED", "existing_file": "FILE_CLOSE_FAILED",
             "directory": "DIRECTORY_CLOSE_FAILED", "recheck": "DIRECTORY_RECHECK_CLOSE_FAILED"}[where])
    assert len(failed) == 1 and row["close_error_types"] == ["OSError"]
    assert len(list(owned.iterdir())) == 1
    assert next(owned.iterdir()).read_bytes() == b"abc"


def test_baseexception_propagates_after_owned_descriptor_closure(kernel, owned, monkeypatch):
    real, captured = kernel._open_directory, []
    def open_directory(*args):
        fd = real(*args); captured.append(fd); return fd
    def stop(): raise KeyboardInterrupt()
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_open_directory", open_directory)
        with pytest.raises(KeyboardInterrupt): _ensure(kernel, owned, "record", b"abc", stop)
    assert list(owned.iterdir()) == [] and len(captured) == 1
    with pytest.raises(OSError): os.fstat(captured[0])


@pytest.mark.parametrize("mode", ("short_read", "grows_during_read", "changed_validator"))
def test_existing_record_instability_never_calls_factory_or_becomes_valid(kernel, owned, monkeypatch, mode):
    target = owned/"record"; target.write_bytes(b"abc"); real_read = kernel.os.read
    def read(fd, count):
        if mode == "short_read": return b""
        block = real_read(fd, count)
        with target.open("ab") as stream: stream.write(b"!")
        return block
    def validate(_):
        target.write_bytes(b"changed"); return True
    with monkeypatch.context() as patch:
        if mode != "changed_validator": patch.setattr(kernel.os, "read", read)
        row = _failure(kernel, lambda: _ensure(kernel, owned, "record", b"abc", _never,
            validate if mode == "changed_validator" else None),
            {"short_read": "SHORT_OR_INVALID_FILE_READ", "grows_during_read": "FILE_CHANGED_DURING_READ",
             "changed_validator": "EXISTING_RECORD_CHANGED_DURING_VALIDATION"}[mode])
    assert row["publication_attempted"] is False and len(list(owned.iterdir())) == 1


@pytest.mark.parametrize("kind", ("symlink", "directory", "fifo"))
def test_existing_nonregular_target_is_never_read_or_replaced(kernel, owned, monkeypatch, kind):
    target = owned/"record"
    if kind == "symlink":
        outside = owned/"elsewhere"; outside.write_bytes(b"untouched"); target.symlink_to(outside)
    elif kind == "directory": target.mkdir()
    else: os.mkfifo(target)
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "read", _never)
        _failure(kernel, lambda: _ensure(kernel, owned, "record", b"abc", _never), "REGULAR_NON_SYMLINK_FILE_REQUIRED")
    assert target.lstat()
    if kind == "symlink": assert outside.read_bytes() == b"untouched"


@pytest.mark.parametrize("where", ("final", "ancestor"))
def test_every_directory_component_is_opened_without_following_symlinks(kernel, owned, where):
    real = owned/"real"; real.mkdir(); nested = real/"nested"; nested.mkdir()
    alias = owned/"alias"; alias.symlink_to(real, target_is_directory=True)
    directory = alias if where == "final" else alias/"nested"
    _failure(kernel, lambda: _ensure(kernel, directory, "record", b"abc", _never))
    row = kernel.inventory_directory_v4(directory, {}, max_entries=0, max_total_bytes=0)
    _unavailable(row)
    assert not (real/"record").exists() and not (nested/"record").exists()


@pytest.mark.parametrize("replacement", ("directory", "symlink"))
def test_pinned_operations_do_not_escape_and_path_recheck_vetoes_rebinding(kernel, owned, replacement):
    target = owned/"target"; target.mkdir(); parked = owned/"parked"; foreign = owned/"foreign"; foreign.mkdir()
    def make():
        target.rename(parked)
        if replacement == "directory": target.mkdir()
        else: target.symlink_to(foreign, target_is_directory=True)
        return b"abc"
    row = _failure(kernel, lambda: _ensure(kernel, target, "record", b"abc", make))
    assert row["reason"] == ("DIRECTORY_PATH_BINDING_CHANGED" if replacement == "directory" else "FILESYSTEM_OR_CALLBACK_EXCEPTION")
    assert (parked/"record").read_bytes() == b"abc" and not (target/"record").exists()
    assert not (foreign/"record").exists() and row["target_linked"]
    assert row.get("directory_path_binding_rechecked") is not True


@pytest.mark.parametrize("directory", ("relative", "", "/a//b", "/a/../b", "/a/./b", "/a\x00b"))
def test_invalid_directory_spelling_refuses_before_filesystem_open(kernel, monkeypatch, directory):
    code = "DIRECTORY_COMPONENT_POLICY" if directory in ("/a//b", "/a/../b", "/a/./b") else "ABSOLUTE_OWNED_DIRECTORY_REQUIRED"
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "open", _never)
        _failure(kernel, lambda: _ensure(kernel, directory, "record", b"abc", _never), code)


@pytest.mark.parametrize("name", ("", ".", "..", "../escape", "a/b", "a\x00b", "a"*221, "é"*111, 1))
def test_invalid_publication_name_never_invokes_factory_or_writes(kernel, owned, name):
    code = "ARTIFACT_NAME_BYTE_LIMIT" if name in ("a"*221, "é"*111) else "INVALID_ARTIFACT_NAME"
    _failure(kernel, lambda: _ensure(kernel, owned, name, b"abc", _never), code)
    assert list(owned.iterdir()) == []


@pytest.mark.parametrize("value", (True, -1, 1.0, 128*1024**2+1))
def test_invalid_publication_cap_is_refused_before_factory(kernel, owned, value):
    _failure(kernel, lambda: kernel.ensure_terminal_record_v4(owned, "record", _never, _never, max_bytes=value),
        "INVALID_FILE_BYTE_LIMIT")
    assert list(owned.iterdir()) == []


@pytest.mark.parametrize("delta", (-1, 0, 1))
def test_publication_byte_limit_exact_boundary(kernel, owned, delta):
    operation = lambda: kernel.publish_bytes_exclusive_v4(owned, "record", b"abc", max_bytes=3+delta)
    if delta < 0:
        _failure(kernel, operation, "PUBLICATION_BYTES_OR_LIMIT"); assert list(owned.iterdir()) == []
    else: _complete(operation())


@pytest.mark.parametrize("delta", (-1, 0, 1))
def test_complete_inventory_entry_limit_boundary(kernel, owned, delta):
    blobs = {"a": b"abc", "b": b"defg"}
    for name, blob in blobs.items(): (owned/name).write_bytes(blob)
    row = _inventory(kernel, owned, blobs, max_entries=2+delta)
    if delta < 0:
        _unavailable(row); assert row["reason"] == "EXPECTED_ENTRY_SET_OR_LIMIT" and row["read_bytes"] == 0
    else:
        _complete(row)
        assert row["enumerated_entries"] == 4 and row["name_inventory_complete"] and row["directory_stable"]


@pytest.mark.parametrize("delta", (-1, 0, 1))
def test_complete_inventory_aggregate_byte_limit_boundary(kernel, owned, delta):
    blobs = {"a": b"abc", "b": b"defg"}
    for name, blob in blobs.items(): (owned/name).write_bytes(blob)
    row = _inventory(kernel, owned, blobs, max_total_bytes=7+delta)
    if delta < 0:
        _unavailable(row)
        assert row["entries"]["b"]["status"] == "TOTAL_BYTE_LIMIT_NO_READ"
        assert row["charged_read_bytes"] == row["read_bytes"] == 3
    else:
        _complete(row); assert row["charged_read_bytes"] == row["read_bytes"] == 7


@pytest.mark.parametrize("mode", ("hash", "length", "file_cap", "unbound_observation"))
def test_inventory_hash_length_and_admission_are_not_silently_repaired(kernel, owned, mode):
    blob = b"abc"; (owned/"record").write_bytes(blob); expected = {"record": _spec(blob)}
    if mode == "hash": expected["record"]["sha256"] = "0"*64
    elif mode == "length": expected["record"]["bytes"] = 2
    elif mode == "file_cap": expected["record"].update(max_bytes=2, bytes=None)
    else: expected["record"].update(sha256=None, bytes=None)
    row = kernel.inventory_directory_v4(owned, expected, max_entries=1, max_total_bytes=3)
    if mode == "unbound_observation":
        _complete(row); assert row["entries"]["record"]["sha256"] == _sha(blob)
        # The caller elected an observation, not historical authentication.
        assert expected["record"]["sha256"] is None
    else:
        _unavailable(row)
        assert row["entries"]["record"]["status"] == {"hash": "EXPECTED_HASH_MISMATCH",
            "length": "EXPECTED_SIZE_MISMATCH_NO_READ", "file_cap": "FILE_BYTE_LIMIT_NO_READ"}[mode]
        assert row["read_bytes"] == (3 if mode == "hash" else 0)
    assert (owned/"record").read_bytes() == blob


@pytest.mark.parametrize("relation", ("distinct_inode", "hardlink"))
def test_same_bytes_never_deduplicate_names_or_aggregate_read_charge(kernel, owned, relation):
    blob = b"same"; (owned/"a").write_bytes(blob)
    if relation == "hardlink": os.link(owned/"a", owned/"b")
    else: (owned/"b").write_bytes(blob)
    row = _inventory(kernel, owned, {"a": blob, "b": blob}); _complete(row)
    assert row["initial_names"] == row["final_names"] == ["a", "b"]
    assert row["charged_read_bytes"] == row["read_bytes"] == 2*len(blob)
    assert (row["entries"]["a"]["identity"]["inode"] == row["entries"]["b"]["identity"]["inode"]) is (relation == "hardlink")


@pytest.mark.parametrize("name", ("extra", ".record.partial-invented"))
def test_unexpected_same_byte_file_is_metadata_only_and_vetoes_whole_inventory(kernel, owned, monkeypatch, name):
    blob = b"same"; (owned/"record").write_bytes(blob); (owned/name).write_bytes(blob)
    real, read_names = kernel._read_file, []
    def read(fd, requested, *a, **kw):
        read_names.append(requested); assert requested == "record"
        return real(fd, requested, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_read_file", read)
        row = _inventory(kernel, owned, {"record": blob}, max_entries=2)
    _unavailable(row)
    assert read_names == ["record"] and row["unexpected_names"] == [name]
    assert row["entries"][name]["sha256"] is None and row["entries"][name]["identity"]["bytes"] == len(blob)
    assert row["entries"][name]["identity"]["inode"] != row["entries"]["record"]["identity"]["inode"]
    assert set(row["entries"]) == {"record", name} and len(list(owned.iterdir())) == 2


def test_explicitly_declared_dotfile_is_bound_but_missing_names_still_veto(kernel, owned):
    (owned/".declared").write_bytes(b"x")
    _complete(_inventory(kernel, owned, {".declared": b"x"}))
    row = _inventory(kernel, owned, {".declared": b"x", "missing": b"y"})
    _unavailable(row); assert row["missing_names"] == ["missing"] and row["read_bytes"] == 1


@pytest.mark.parametrize("kind", ("symlink", "directory", "fifo"))
def test_inventory_nonregular_entries_are_metadata_only_and_never_followed(kernel, owned, monkeypatch, kind):
    target = owned/"record"
    if kind == "symlink": target.symlink_to(owned/"deliberately-absent")
    elif kind == "directory": target.mkdir(); (target/"never-recursed").write_bytes(b"x")
    else: os.mkfifo(target)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_read_file", _never)
        row = _inventory(kernel, owned, {"record": b"abc"})
    _unavailable(row)
    assert row["entries"]["record"]["status"] == "NON_REGULAR_OR_SYMLINK_NO_READ"
    assert row["read_bytes"] == row["charged_read_bytes"] == 0


def test_directory_entry_overflow_retains_bounded_initial_names_without_reading(kernel, owned):
    for name in ("a", "b", "c"): (owned/name).write_bytes(b"x")
    row = kernel.inventory_directory_v4(owned, {}, max_entries=2, max_total_bytes=0)
    _unavailable(row)
    assert row["reason"] == "DIRECTORY_ENTRY_LIMIT" and row["enumerated_entries"] == 3
    assert len(row["entries"]) == len(row["initial_names"]) == 2
    assert row["first_unrecorded_name"] in {"a", "b", "c"}-set(row["entries"])
    assert row["name_inventory_complete"] is False and row["read_bytes"] == 0
    assert len(list(owned.iterdir())) == 3


def test_empty_directory_at_zero_entry_and_byte_caps_is_complete(kernel, owned):
    row = _inventory(kernel, owned, {}); _complete(row)
    assert row["entries"] == {} and row["enumerated_entries"] == row["read_bytes"] == 0


@pytest.mark.parametrize("field,value", (("max_entries", True), ("max_entries", -1), ("max_entries", 1025),
    ("max_total_bytes", 1.), ("max_total_bytes", -1), ("max_total_bytes", 8*1024**3+1)))
def test_invalid_inventory_caps_refuse_before_any_directory_read(kernel, owned, monkeypatch, field, value):
    options = dict(max_entries=0, max_total_bytes=0); options[field] = value
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_open_directory", _never)
        row = kernel.inventory_directory_v4(owned, {}, **options)
    _unavailable(row); assert row["entries"] == {} and row["read_bytes"] == 0
    assert row["reason"] == ("INVALID_ENTRY_LIMIT" if field == "max_entries" else "INVALID_TOTAL_BYTE_LIMIT")


@pytest.mark.parametrize("mode", ("table_type", "row_type", "extra_key", "missing_key", "hash_upper", "hash_length",
    "bool_size", "oversize", "bool_limit", "bad_name"))
def test_inventory_expectation_schema_is_exact_and_not_repaired(kernel, owned, mode):
    expected = {"record": _spec(b"abc")}
    if mode == "table_type": expected = []
    elif mode == "row_type": expected["record"] = []
    elif mode == "extra_key": expected["record"]["extra"] = 1
    elif mode == "missing_key": del expected["record"]["sha256"]
    elif mode == "hash_upper": expected["record"]["sha256"] = "A"*64
    elif mode == "hash_length": expected["record"]["sha256"] = "a"*63
    elif mode == "bool_size": expected["record"]["bytes"] = True
    elif mode == "oversize": expected["record"]["bytes"] = 4
    elif mode == "bool_limit": expected["record"]["max_bytes"] = True
    else: expected = {"../escape": _spec(b"abc")}
    row = kernel.inventory_directory_v4(owned, expected, max_entries=1, max_total_bytes=3)
    _unavailable(row); assert row["entries"] == {} and row["read_bytes"] == 0


def test_late_declared_read_failure_retains_all_names_and_charged_bytes(kernel, owned, monkeypatch):
    blobs = {"a": b"aa", "b": b"bbb"}
    for name, blob in blobs.items(): (owned/name).write_bytes(blob)
    real = kernel._read_file
    def read(fd, name, *a, **kw):
        if name == "b": raise OSError("injected private text")
        return real(fd, name, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_read_file", read)
        row = _inventory(kernel, owned, blobs)
    _unavailable(row)
    assert set(row["entries"]) == {"a", "b"} and row["name_inventory_complete"]
    assert row["charged_read_bytes"] == 5 and row["read_bytes"] == 2
    assert row["entries"]["a"]["status"] == "REGULAR_FILE_BYTES_BOUND"
    assert row["entries"]["b"]["status"] == "ENTRY_STAT_OR_READ_EXCEPTION"


@pytest.mark.parametrize("mode", ("adds_name", "changes_file", "replaces_directory"))
def test_late_inventory_change_cannot_publish_a_complete_inventory(kernel, owned, monkeypatch, mode):
    directory = owned/"records"; directory.mkdir(); (directory/"a").write_bytes(b"aa")
    real, changed = kernel._scan, []
    def scan(fd, cap, evidence, *, first):
        if not first and not changed:
            changed.append(1)
            if mode == "adds_name": (directory/"late").write_bytes(b"x")
            elif mode == "changes_file": (directory/"a").write_bytes(b"changed")
            else:
                directory.rename(owned/"parked"); directory.mkdir()
        return real(fd, cap, evidence, first=first)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_scan", scan)
        row = _inventory(kernel, directory, {"a": b"aa"}, max_entries=2)
    _unavailable(row)
    codes = {e["code"] for e in row["errors"]}
    if mode == "adds_name": assert "DIRECTORY_NAMES_CHANGED" in codes and "late" in row["final_names"]
    elif mode == "changes_file": assert "ENTRY_CHANGED_BEFORE_INVENTORY_END" in codes
    else: assert row["reason"] == "DIRECTORY_PATH_BINDING_CHANGED"


@pytest.mark.parametrize("pass_number", (1, 2))
@pytest.mark.parametrize("failure", ("iterate", "close"))
def test_enumerator_failure_keeps_partial_name_pass_and_never_returns_complete(kernel, owned, monkeypatch, pass_number, failure):
    (owned/"a").write_bytes(b"x"); real, calls = kernel.os.scandir, []
    class BrokenEntries:
        def __enter__(self): return self
        def __exit__(self, *args):
            if failure == "close": raise OSError("injected private text")
            return False
        def __iter__(self):
            yield SimpleNamespace(name="a")
            if failure == "iterate": raise OSError("injected private text")
    def scandir(fd):
        calls.append(fd)
        return BrokenEntries() if len(calls) == pass_number else real(fd)
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "scandir", scandir)
        row = _inventory(kernel, owned, {"a": b"x"})
    _unavailable(row)
    assert row["initial_names"] == ["a"] and "a" in row["entries"]
    if pass_number == 2: assert row["final_names"] == ["a"] and row["read_bytes"] == 1
    else: assert not row["name_inventory_complete"] and row["read_bytes"] == 0


def test_empty_new_terminal_distinguishes_publication_from_zero_content_bytes(kernel, owned):
    row = _ensure(kernel, owned, "empty", b""); _complete(row)
    assert row["publication_attempted"] is True and row["wrote_new_bytes"] is False
    assert row["target_linked"] is True and row["written_bytes"] == row["bytes"] == 0
    assert [p.name for p in owned.iterdir()] == ["empty"]


@pytest.mark.parametrize("where", ("scan_descriptor", "declared_file"))
def test_inventory_descriptor_close_failure_is_whole_failure_not_empty_success(kernel, owned, monkeypatch, where):
    blobs = {"a": b"aa", "b": b"bbb"}
    for name, blob in blobs.items(): (owned/name).write_bytes(blob)
    real_scan, real_close, real_stat = kernel.os.scandir, kernel.os.close, kernel.os.fstat
    scan_fds, failures = [], []
    def scandir(fd):
        scan_fds.append(fd); return real_scan(fd)
    def close(fd):
        selected = fd in scan_fds if where == "scan_descriptor" else stat.S_ISREG(real_stat(fd).st_mode)
        answer = real_close(fd)
        if selected and not failures:
            failures.append(fd); raise OSError("injected private text")
        return answer
    with monkeypatch.context() as patch:
        patch.setattr(kernel.os, "scandir", scandir)
        patch.setattr(kernel.os, "close", close)
        row = _inventory(kernel, owned, blobs)
    _unavailable(row)
    assert len(failures) == 1 and row["close_error_types"] == ["OSError"]
    assert set(row["entries"]) == {"a", "b"} and len(list(owned.iterdir())) == 2
    if where == "scan_descriptor":
        assert row["reason"] == "DIRECTORY_SCAN_CLOSE_FAILED" and row["read_bytes"] == 0
        assert set(row["initial_names"]) == {"a", "b"}
    else:
        assert row["read_bytes"] == row["charged_read_bytes"] == 5
        assert row["entries"]["a"]["status"] == "FILE_CLOSE_FAILED"
        assert {e["code"] for e in row["errors"]} == {"FILE_CLOSE_FAILED", "DESCRIPTOR_CLOSE_FAILURE"}


def test_changed_file_between_inventory_admission_and_stable_read_is_not_rebound(kernel, owned, monkeypatch):
    (owned/"record").write_bytes(b"old"); real = kernel._read_file
    def read(fd, name, *args, **kwargs):
        before = (owned/name).stat()
        (owned/name).write_bytes(b"new")
        os.utime(owned/name, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000_000))
        return real(fd, name, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_read_file", read)
        row = _inventory(kernel, owned, {"record": b"old"})
    _unavailable(row)
    assert row["entries"]["record"]["status"] == "FILE_CHANGED_AFTER_ADMISSION"
    assert row["read_bytes"] == row["charged_read_bytes"] == 3
    assert row["entries"]["record"]["sha256"] is None and (owned/"record").read_bytes() == b"new"


def test_source_import_closure_is_stdlib_only_without_runner_or_import_time_io():
    tree = ast.parse(SOURCE.read_text())
    allowed = {"hashlib", "os", "secrets", "stat", "sys"}
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports == allowed and not any(isinstance(n, ast.ImportFrom) for n in ast.walk(tree))
    # Only definitions, docstring and literal constants at module level.
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.Expr))
        if isinstance(node, ast.Assign): assert not any(isinstance(n, ast.Call) for n in ast.walk(node.value))
        if isinstance(node, ast.Expr): assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
