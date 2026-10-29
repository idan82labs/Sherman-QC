"""C02 hotfolder: SMB-safe readiness, archive/quarantine, dedupe, recovery.

Every scan is a tiny synthetic file in tmp_path that only borrows the real
ZEISS filename shapes (``23553000_B_FULL_SCAN 1.ply``, ``38172000_01_SCAN_2
1.ply``, ``BEFORE BENDING  11979003_C.ply``, ``(1)`` copies). No source
assets are read.
"""
from __future__ import annotations

import errno
import json
import os
import struct
import time
from pathlib import Path

import pytest

from apps.production_line.ingest import hotfolder as hotfolder_mod
from apps.production_line.ingest.hotfolder import HotfolderConfig, HotfolderConfigError, HotfolderPoller
from apps.production_line.ingest.ports import ActiveRecipeRef, ResolveOutcome
from apps.production_line.ingest.sku_resolver import FilenameRecipeResolver
from apps.production_line.store.line_store import LineStore

RECIPE_SHA = {sku: (str(i) * 64)[:64] for i, sku in enumerate(["23553000", "38172000", "11979003", "28870000"], start=1)}
COMPILED_SHA = "e" * 64


def ply(n_vertices: int = 4, seed: int = 0, faces: bool = True) -> bytes:
    head = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if faces:
        head += ["element face 1", "property list uchar int vertex_indices"]
    head.append("end_header")
    body = b"".join(struct.pack("<fff", i + seed, 2.0 * i, -1.0) for i in range(n_vertices))
    if faces:
        body += struct.pack("<Biii", 3, 0, 1, 2)
    return ("\n".join(head) + "\n").encode() + body


class FakeCatalog:
    def __init__(self) -> None:
        self.refs = {
            "23553000": ActiveRecipeRef("23553000", "23553000@v1", 1, RECIPE_SHA["23553000"], "B", COMPILED_SHA, "COMPILED"),
            "38172000": ActiveRecipeRef("38172000", "38172000@v2", 2, RECIPE_SHA["38172000"], "01", COMPILED_SHA, "COMPILED"),
            "11979003": ActiveRecipeRef("11979003", "11979003@v1", 1, RECIPE_SHA["11979003"], "C", COMPILED_SHA, "COMPILED"),
            "28870000": ActiveRecipeRef("28870000", "28870000@v1", 1, RECIPE_SHA["28870000"], "A", None, "BLOCKED"),
        }
        self.registered = {"40500000"}

    def list_active_recipes(self):
        return list(self.refs.values())

    def list_registered_skus(self):
        return set(self.registered) | set(self.refs)


class StaticResolver:
    def __init__(self, outcome: ResolveOutcome | Exception) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    def resolve(self, name: str) -> ResolveOutcome:
        self.calls.append(name)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture()
def env(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    root = tmp_path / "data_root"
    root.mkdir()
    store = LineStore(root / "db" / "line.sqlite3")
    return inbox, root, store


def make_poller(env, resolver=None, **cfg) -> HotfolderPoller:
    inbox, root, store = env
    defaults = dict(poll_interval_s=0.05, stable_polls=3, min_age_s=0.0, min_size_bytes=0, truncated_grace_s=0.0)
    defaults.update(cfg)
    config = HotfolderConfig(station_id="cell-1", inbox=inbox, data_root=root, **defaults)
    return HotfolderPoller(config, store, resolver or FilenameRecipeResolver(FakeCatalog()))


def poll(poller: HotfolderPoller, times: int, sleep: float = 0.0) -> list:
    out = []
    for _ in range(times):
        out.extend(poller.poll_once())
        if sleep:
            time.sleep(sleep)
    return out


def kinds(store: LineStore) -> list[str]:
    return [e["kind"] for e in store.events_after(0, limit=1000)]


# ---------------------------------------------------------------- readiness
def test_chunked_binary_ply_ingested_once_after_three_stable_polls(env):
    inbox, root, store = env
    poller = make_poller(env)
    data = ply(200)
    target = inbox / "23553000_B_FULL_SCAN 1.ply"
    chunks = [data[: len(data) // 3], data[len(data) // 3 : 2 * len(data) // 3], data[2 * len(data) // 3 :]]
    with open(target, "wb") as fh:
        for chunk in chunks:
            fh.write(chunk)
            fh.flush()
            assert poller.poll_once() == []  # the signature keeps changing: never ready
            time.sleep(0.05)
    # The poll after the last write saw the final signature first; the file is
    # ingested on the 3rd consecutive poll that sees it unchanged, not earlier.
    outcomes = []
    stable_polls = 0
    for _ in range(10):
        stable_polls += 1
        got = poller.poll_once()
        if got:
            outcomes.extend(got)
            break
        time.sleep(0.05)
    assert stable_polls == 3
    assert [o.action for o in outcomes] == ["enqueued"]
    assert poll(poller, 5, 0.01) == []  # exactly once
    jobs = store.list_jobs()
    assert len(jobs) == 1 and jobs[0]["sku"] == "23553000" and jobs[0]["compiled_part_sha256"] == COMPILED_SHA
    sha = sha256_file_bytes(data)
    assert jobs[0]["dedupe_key"] == f"inspect:{sha}:{RECIPE_SHA['23553000']}"
    archived = list((root / "archive" / "scans").rglob("*.ply"))
    assert [p.name for p in archived] == [f"{sha}.ply"] and archived[0].read_bytes() == data
    assert not target.exists()  # moved
    assert kinds(store) == ["SCAN_RECEIVED", "JOB_QUEUED"]


def sha256_file_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_temp_names_are_never_ingested(env):
    inbox, _root, store = env
    poller = make_poller(env)
    for name in (".x.ply", "a.ply.part", "a.tmp", "b.ply.partial", "c.ply~", "d.ply.crdownload", "e.ply.sherman-partial",
                 "notes.txt", "mesh.obj"):
        (inbox / name).write_bytes(ply())
    assert poll(poller, 6) == []
    assert store.list_scans() == [] and store.list_jobs() == []


def test_temp_then_rename_is_ingested(env):
    inbox, _root, store = env
    poller = make_poller(env)
    tmp = inbox / "38172000_01_SCAN_2 1.ply.part"
    tmp.write_bytes(ply(8))
    assert poll(poller, 5) == []
    os.replace(tmp, inbox / "38172000_01_SCAN_2 1.ply")
    out = poll(poller, 5)
    assert [o.action for o in out] == ["enqueued"]
    assert store.list_jobs()[0]["recipe_id"] == "38172000@v2"


def test_sidecar_required_waits_for_done(env):
    inbox, _root, store = env
    poller = make_poller(env, sidecar_mode="required")
    (inbox / "a.ply").write_bytes(ply())
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=3))
    assert poll(poller, 6) == []
    (inbox / "23553000_B_FULL_SCAN 1.ply.done").write_text("")
    out = poll(poller, 2)
    assert [(o.action, o.original_name) for o in out] == [("enqueued", "23553000_B_FULL_SCAN 1.ply")]
    assert not (inbox / "23553000_B_FULL_SCAN 1.ply.done").exists()  # consumed with the move
    (inbox / "a.ply.done").write_text("")
    out = poll(poller, 2)
    assert [(o.action, o.reason) for o in out] == [("quarantined", "UNMATCHED_SKU")]
    assert len(store.list_jobs()) == 1


def test_sidecar_json_declared_stage_and_mismatch(env):
    inbox, _root, store = env
    poller = make_poller(env, sidecar_mode="required")
    data = ply(seed=11)
    good = inbox / "23553000_B_SCAN_1 1.ply"
    good.write_bytes(data)
    (inbox / "23553000_B_SCAN_1 1.ply.done").write_text(json.dumps({"sha256": sha256_file_bytes(data), "declared_stage": 1}))
    bad = inbox / "23553000_B_SCAN_2 1.ply"
    bad.write_bytes(ply(seed=12))
    (inbox / "23553000_B_SCAN_2 1.ply.done").write_text(json.dumps({"sha256": "0" * 64}))
    junk = inbox / "23553000_B_FULL_SCAN 1.ply"
    junk.write_bytes(ply(seed=13))
    (inbox / "23553000_B_FULL_SCAN 1.ply.done").write_text("{not json")
    out = {o.original_name: o for o in poll(poller, 4)}
    assert out[good.name].action == "enqueued"
    assert out[bad.name].reason == "SIDECAR_INVALID" and out[junk.name].reason == "SIDECAR_INVALID"
    job = store.get_job(out[good.name].job_id)
    assert job["declared_stage"] == {"stage_index": 1, "source": "SIDECAR"}
    for entry in store.list_quarantine():
        assert entry["resolvable"] is False  # content problem: re-export, never assign


def test_small_and_fresh_files_wait(env):
    inbox, _root, store = env
    poller = make_poller(env, min_size_bytes=10_000)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply())
    assert poll(poller, 6) == []
    poller = make_poller(env, min_age_s=3600)
    assert poll(poller, 6) == []
    assert store.list_scans() == []


# ---------------------------------------------------------------- outcomes
def test_truncated_ply_quarantined_with_reason_json(env):
    inbox, root, store = env
    poller = make_poller(env)
    data = ply(50)[:-30]
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    out = poll(poller, 4)
    assert [(o.action, o.reason) for o in out] == [("quarantined", "PLY_TRUNCATED")]
    sha = sha256_file_bytes(data)
    entry = store.list_quarantine()[0]
    assert entry["quarantine_reason"] == "PLY_TRUNCATED" and entry["resolvable"] is False
    scan = entry["scan_file"]
    assert scan["stored_rel_path"].startswith("quarantine/") and scan["format"] is None
    assert scan["parse"]["sku"] == "23553000"  # the name still parsed, for the operator
    qfile = root / scan["stored_rel_path"]
    assert qfile.read_bytes() == data
    reason = json.loads((qfile.parent / f"{sha}.reason.json").read_text())
    assert reason["reason"] == "PLY_TRUNCATED" and reason["original_name"] == "23553000_B_FULL_SCAN 1.ply"
    assert store.list_jobs() == []
    assert kinds(store) == ["SCAN_RECEIVED", "SCAN_QUARANTINED"]


def test_duplicate_bytes_under_new_name(env):
    inbox, _root, store = env
    poller = make_poller(env)
    data = ply(seed=5)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    poll(poller, 4)
    (inbox / "23553000_B_FULL_SCAN 1 (1).ply").write_bytes(data)
    out = poll(poller, 4)
    assert [(o.action, o.reason) for o in out] == [("duplicate", "SCAN_DUPLICATE")]
    assert out[0].job_id == store.list_jobs()[0]["job_id"]
    assert len(store.list_jobs()) == 1
    assert kinds(store).count("SCAN_DUPLICATE") == 1
    assert not (inbox / "23553000_B_FULL_SCAN 1 (1).ply").exists()


def test_real_filename_shapes_and_quarantine_reasons(env):
    from contracts.production.line_job import LineJobV1, ScanFileV1

    inbox, root, store = env
    poller = make_poller(env)
    names = {
        "23553000_B_FULL_SCAN 1.ply": ("enqueued", None),
        "23553000_B_SCAN_1 1 (1).ply": ("enqueued", None),
        "38172000_01_SCAN_2 1.ply": ("enqueued", None),
        "BEFORE BENDING  11979003_C.ply": ("enqueued", None),
        "11979003_SN_0042.ply": ("enqueued", None),  # SN labels a serial, never a revision
        "23553000_C_FULL_SCAN 1.ply": ("quarantined", "REVISION_MISMATCH"),
        "99999999_A_FULL_SCAN 1.ply": ("quarantined", "UNMATCHED_SKU"),
        "23553000_B 38172000_01.ply": ("quarantined", "AMBIGUOUS_SKU"),
        "___.ply": ("quarantined", "UNPARSEABLE_FILENAME"),
        "28870000_A_FULL_SCAN 1.ply": ("quarantined", "PART_NOT_READY"),  # compile BLOCKED
        "40500000_SCAN1.ply": ("quarantined", "PART_NOT_READY"),  # registered, no recipe
    }
    for i, name in enumerate(names):
        (inbox / name).write_bytes(ply(seed=100 + i))
    got = {o.original_name: (o.action, o.reason) for o in poll(poller, 4)}
    assert got == names
    for scan in store.list_scans(limit=100):
        model = ScanFileV1.model_validate(scan)
        assert (root / model.stored_rel_path).is_file()
        if model.ingest_state == "quarantined":
            assert list((root / "quarantine").rglob(f"{model.sha256}.reason.json")), model.original_name
    for job in store.list_jobs():
        LineJobV1.model_validate(job)
    by_name = {s["original_name"]: s for s in store.list_scans(limit=100)}
    assert by_name["BEFORE BENDING  11979003_C.ply"]["parse"]["scan_label"] == "BEFORE_BENDING"
    assert by_name["23553000_C_FULL_SCAN 1.ply"]["parse"]["revision_status"] == "MISMATCH"
    assert by_name["23553000_B_SCAN_1 1 (1).ply"]["parse"]["copy_index"] == 1
    assert by_name["11979003_SN_0042.ply"]["parse"]["serial"] == "0042"
    assert by_name["11979003_SN_0042.ply"]["parse"]["revision"] is None
    assert {e["quarantine_reason"] for e in store.list_quarantine() if e["resolvable"]} == {
        "REVISION_MISMATCH",
        "UNMATCHED_SKU",
        "AMBIGUOUS_SKU",
        "UNPARSEABLE_FILENAME",
        "PART_NOT_READY",
    }


def test_unmatched_quarantine_then_assignment(env):
    inbox, root, store = env
    poller = make_poller(env)
    (inbox / "99999999_A.ply").write_bytes(ply())
    out = poll(poller, 4)
    assert out[0].reason == "UNMATCHED_SKU"
    entry = store.list_quarantine()[0]
    sha = entry["scan_file"]["sha256"]
    assert entry["scan_file"]["stored_rel_path"].startswith("archive/scans/")  # stays archived
    reason_files = list((root / "quarantine").rglob(f"{sha}.reason.json"))
    assert len(reason_files) == 1 and json.loads(reason_files[0].read_text())["reason"] == "UNMATCHED_SKU"
    job_id = store.resolve_quarantine(
        entry["scan_file"]["scan_file_id"],
        {
            "recipe_id": "23553000@v1",
            "recipe_version": 1,
            "recipe_sha256": RECIPE_SHA["23553000"],
            "sku": "23553000",
            "compiled_part_sha256": COMPILED_SHA,
        },
        {"kind": "user", "id": "admin", "role": "admin"},
    )
    assert store.get_job(job_id)["scan_sha256"] == sha


def test_file_dropped_while_stopped_is_ingested_by_new_instance(env):
    inbox, _root, store = env
    first = make_poller(env)
    poll(first, 2)
    del first
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=9))
    second = make_poller(env)
    second.startup_rescan()
    out = poll(second, 4)
    assert [o.action for o in out] == ["enqueued"]


def test_copy_leave_keeps_source_and_never_reingests(env):
    inbox, _root, store = env
    poller = make_poller(env, source_policy="copy_leave")
    src = inbox / "23553000_B_FULL_SCAN 1.ply"
    src.write_bytes(ply(seed=21))
    out = poll(poller, 8)
    assert [o.action for o in out] == ["enqueued"]
    assert src.exists()
    restarted = make_poller(env, source_policy="copy_leave")
    assert poll(restarted, 8) == []
    assert kinds(store).count("SCAN_DUPLICATE") == 0 and len(store.list_scans()) == 1
    assert not list((env[1] / "archive").rglob("*.sherman-partial"))


def test_cross_device_move_copies_verifies_and_unlinks(env, monkeypatch):
    inbox, root, store = env
    real_replace = os.replace
    calls = []

    def fake_replace(src, dst):
        calls.append((str(src), str(dst)))
        if str(src).startswith(str(inbox)):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(hotfolder_mod.os, "replace", fake_replace)
    poller = make_poller(env)
    src = inbox / "23553000_B_FULL_SCAN 1.ply"
    data = ply(seed=31)
    src.write_bytes(data)
    out = poll(poller, 4)
    assert [o.action for o in out] == ["enqueued"]
    assert not src.exists()
    archived = list((root / "archive").rglob("*.ply"))
    assert archived[0].read_bytes() == data
    assert any(c[0].endswith(".sherman-partial") for c in calls)  # temp copy, then rename
    assert not list(root.rglob("*.sherman-partial"))


def test_startup_rescan_removes_partials(env):
    _inbox, root, _store = env
    leftover = root / "archive" / "scans" / "2026" / "10" / ("f" * 64 + ".ply.sherman-partial")
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"half")
    make_poller(env).startup_rescan()
    assert not leftover.exists()


def test_crash_after_register_before_move_is_completed(env):
    """A row in 'received' whose bytes are still in the inbox: the next poll finishes it."""
    inbox, root, store = env
    data = ply(seed=41)
    sha = sha256_file_bytes(data)
    store.register_scan(
        sha256=sha,
        original_name="23553000_B_FULL_SCAN 1.ply",
        size_bytes=len(data),
        station_id="cell-1",
        stored_rel_path=f"archive/scans/2026/10/{sha}.ply",
        parse={"status": "OK", "sku": "23553000"},
        match={"status": "UNPARSEABLE", "reason": "RESOLUTION_PENDING"},
        format="PLY_BINARY_LE",
    )
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    out = poll(make_poller(env), 4)
    assert [o.action for o in out] == ["enqueued"]
    assert (root / f"archive/scans/2026/10/{sha}.ply").read_bytes() == data
    assert "SCAN_DUPLICATE" not in kinds(store)


def test_same_bytes_under_another_sku_name_never_take_over_a_pending_scan(env):
    """A scan left in 'received' (resolver failed) must not be bound to the
    recipe that a second drop of the same bytes under another name resolves to."""
    inbox, _root, store = env
    real = FilenameRecipeResolver(FakeCatalog())

    class FailsForFirstName:
        def resolve(self, name: str) -> ResolveOutcome:
            if name.startswith("23553000"):
                raise RuntimeError("recipe store offline")
            return real.resolve(name)

    poller = make_poller(env, resolver=FailsForFirstName())
    data = ply(seed=77)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    assert [o.action for o in poll(poller, 4)] == ["pending"]
    (inbox / "38172000_01_SCAN_2 1.ply").write_bytes(data)
    out = poll(poller, 4)
    assert [(o.action, o.reason, o.original_name) for o in out] == [
        ("duplicate", "SCAN_DUPLICATE", "38172000_01_SCAN_2 1.ply")
    ]
    assert store.list_jobs() == []  # never inspected against 38172000's recipe
    scan = store.list_scans()[0]
    assert scan["ingest_state"] == "received" and scan["original_name"] == "23553000_B_FULL_SCAN 1.ply"
    assert scan["parse"]["sku"] != "38172000" and scan["match"]["reason"].startswith("RESOLVER_ERROR")
    dup = [e for e in store.events_after(0, limit=100) if e["kind"] == "SCAN_DUPLICATE"]
    assert len(dup) == 1 and dup[0]["payload"]["original_name"] == "38172000_01_SCAN_2 1.ply"
    assert dup[0]["payload"]["existing_state"] == "received"
    assert not (inbox / "38172000_01_SCAN_2 1.ply").exists()  # disposed like any duplicate
    # healed: the pending row is resolved from its OWN name
    out = make_poller(env).startup_rescan()
    assert [o.action for o in out] == ["enqueued"]
    jobs = store.list_jobs()
    assert [j["sku"] for j in jobs] == ["23553000"]


def test_future_mtime_from_a_skewed_writer_clock_does_not_stall(env, caplog):
    """The ZEISS PC's clock may run ahead of the server: an mtime in the future
    must not block min_age_s forever (the observed stable time counts)."""
    inbox, _root, store = env
    poller = make_poller(env, min_age_s=0.2)
    path = inbox / "23553000_B_FULL_SCAN 1.ply"
    path.write_bytes(ply(seed=88))
    future = time.time() + 600
    os.utime(path, (future, future))
    with caplog.at_level("WARNING", logger=hotfolder_mod.__name__):
        out = poll(poller, 8, sleep=0.05)
    assert [o.action for o in out] == ["enqueued"]
    assert len(store.list_jobs()) == 1
    skew = [r for r in caplog.records if "in the future" in r.getMessage()]
    assert len(skew) == 1  # warned once per file, not every poll


def test_fresh_file_still_waits_for_min_age(env):
    """The skew fix must not weaken min_age_s for a file seen only briefly."""
    inbox, _root, store = env
    poller = make_poller(env, min_age_s=30.0)
    path = inbox / "23553000_B_FULL_SCAN 1.ply"
    path.write_bytes(ply(seed=89))
    future = time.time() + 600
    os.utime(path, (future, future))
    assert poll(poller, 6, sleep=0.02) == []
    assert store.list_scans() == []


def test_resolver_failure_leaves_scan_pending_then_resumes(env):
    inbox, _root, store = env
    broken = StaticResolver(RuntimeError("recipe store offline"))
    poller = make_poller(env, resolver=broken)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=51))
    out = poll(poller, 4)
    assert [o.action for o in out] == ["pending"]
    scan = store.list_scans()[0]
    assert scan["ingest_state"] == "received" and scan["match"]["reason"].startswith("RESOLVER_ERROR")
    healed = make_poller(env)  # real resolver
    out = healed.startup_rescan()
    assert [o.action for o in out] == ["enqueued"]
    assert store.list_scans()[0]["ingest_state"] == "enqueued"


def test_per_file_exception_does_not_stop_the_loop(env, monkeypatch):
    inbox, _root, store = env
    real_probe = hotfolder_mod.probe

    def flaky(path, **kw):
        if "BOOM" in str(path):
            raise RuntimeError("disk hiccup")
        return real_probe(path, **kw)

    monkeypatch.setattr(hotfolder_mod, "probe", flaky)
    poller = make_poller(env)
    (inbox / "BOOM 23553000_B.ply").write_bytes(ply(seed=61))
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=62))
    out = poll(poller, 20)
    assert sorted(o.action for o in out) == ["enqueued", "error", "error", "error"]  # gives up after 3 tries
    assert (inbox / "BOOM 23553000_B.ply").exists()  # never lost


def test_symlinks_are_ignored(env, tmp_path):
    inbox, _root, store = env
    outside = tmp_path / "outside.ply"
    outside.write_bytes(ply())
    os.symlink(outside, inbox / "23553000_B_FULL_SCAN 1.ply")
    assert poll(make_poller(env), 5) == []
    assert outside.exists() and store.list_scans() == []


def test_stl_ingest(env):
    inbox, root, store = env
    tri = struct.pack("<3f", 0, 0, 1) + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0) + b"\x00\x00"
    data = b"binary".ljust(80, b" ") + struct.pack("<I", 1) + tri
    (inbox / "38172000_01_SCAN_2 1.stl").write_bytes(data)
    out = poll(make_poller(env), 4)
    assert [o.action for o in out] == ["enqueued"]
    scan = store.list_scans()[0]
    assert scan["format"] == "STL_BINARY" and scan["stored_rel_path"].endswith(".stl")


# ---------------------------------------------------------------- config
def test_config_validation(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    ok = HotfolderConfig(station_id="cell-1", inbox=inbox, data_root=root, repo_root=tmp_path / "repo")
    ok.validate()
    bad = [
        dict(station_id="bad id"),
        dict(inbox=inbox.relative_to(tmp_path)),
        dict(sidecar_mode="sometimes"),
        dict(source_policy="delete"),
        dict(extensions=(".obj",)),
        dict(stable_polls=0),
        dict(inbox=root / "archive"),
        dict(data_root=inbox / "root"),
    ]
    (root / "archive").mkdir()
    (inbox / "root").mkdir()
    for change in bad:
        cfg = HotfolderConfig(**{**ok.__dict__, **change})
        with pytest.raises(HotfolderConfigError):
            cfg.validate()
    repo = tmp_path / "repo"
    (repo / "data" / "line").mkdir(parents=True)
    with pytest.raises(HotfolderConfigError):
        HotfolderConfig(station_id="c", inbox=inbox, data_root=repo / "data" / "line", repo_root=repo).validate()
    (repo / "uploads" / "inbox").mkdir(parents=True)
    with pytest.raises(HotfolderConfigError):
        HotfolderConfig(station_id="c", inbox=repo / "uploads" / "inbox", data_root=root, repo_root=repo).validate()


def test_resolver_regex_must_name_sku():
    from domains.production.recipe.sku import SkuError

    with pytest.raises(SkuError):
        FilenameRecipeResolver(FakeCatalog(), pattern=r"(?P<part>\d+)")
    resolver = FilenameRecipeResolver(FakeCatalog(), pattern=r"(?P<sku>\d{8})_(?P<revision>[A-Z])_.*")
    assert resolver.resolve("23553000_B_FULL_SCAN 1.ply").status == "MATCHED"
    assert resolver.resolve("23553000_C_FULL_SCAN 1.ply").status == "REVISION_MISMATCH"


def test_resolver_cache_and_invalidate():
    catalog = FakeCatalog()
    now = [0.0]
    resolver = FilenameRecipeResolver(catalog, cache_ttl_s=10, monotonic=lambda: now[0])
    assert resolver.resolve("40500000_SCAN1.ply").status == "PART_NOT_READY"
    catalog.refs["40500000"] = ActiveRecipeRef("40500000", "40500000@v1", 1, "9" * 64, None, COMPILED_SHA, "COMPILED")
    assert resolver.resolve("40500000_SCAN1.ply").status == "PART_NOT_READY"  # cached
    resolver.invalidate()
    out = resolver.resolve("40500000_SCAN1.ply")
    assert out.status == "MATCHED" and out.compiled_part_sha256 == COMPILED_SHA
    now[0] = 11.0
    del catalog.refs["40500000"]
    catalog.registered.discard("40500000")
    assert resolver.resolve("40500000_SCAN1.ply").status == "UNMATCHED"


def test_hotfolder_module_does_not_import_c01():
    import subprocess
    import sys

    code = (
        "import sys, apps.production_line.ingest.hotfolder; "
        "bad=[m for m in sys.modules if m.startswith(('contracts.production','domains.production','numpy','pydantic'))]; "
        "assert not bad, bad"
    )
    env = {**os.environ, "PYTHONPATH": ".:backend"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)


# ---------------------------------------------------------------- review fixes (fix stage)
def test_truncated_file_waits_out_the_grace_then_completes(env):
    inbox, root, store = env
    poller = make_poller(env, truncated_grace_s=0.6)
    data = ply(80)
    target = inbox / "23553000_B_FULL_SCAN 1.ply"
    target.write_bytes(data[:-40])  # the writer paused mid-file (SMB hiccup)
    assert poll(poller, 6, 0.02) == []  # stable, but inside the grace: not consumed
    assert target.exists() and store.list_scans() == [] and not (root / "quarantine").exists()
    with open(target, "ab") as fh:
        fh.write(data[-40:])  # the writer resumes inside the grace
    out = poll(poller, 6, 0.02)
    assert [o.action for o in out] == ["enqueued"]
    assert store.list_scans()[0]["sha256"] == sha256_file_bytes(data)  # the complete bytes, not the partial
    assert "SCAN_QUARANTINED" not in kinds(store)


def test_truncated_file_is_quarantined_after_the_grace(env):
    inbox, root, store = env
    poller = make_poller(env, truncated_grace_s=0.4)
    data = ply(50)[:-30]
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    assert poll(poller, 4, 0.02) == []
    time.sleep(0.5)
    out = poll(poller, 2)
    assert [(o.action, o.reason) for o in out] == [("quarantined", "PLY_TRUNCATED")]


def test_crash_after_quarantined_register_resumes_without_duplicate(env):
    """Crash after register_scan(quarantined) and before the move: the restart
    places the bytes, writes the missing reason.json and emits no SCAN_DUPLICATE."""
    inbox, root, store = env
    data = ply(50)[:-30]
    sha = sha256_file_bytes(data)
    rel = f"quarantine/2026-10-17/{sha}.ply"
    store.register_scan(
        sha256=sha, original_name="23553000_B_FULL_SCAN 1.ply", size_bytes=len(data), station_id="cell-1",
        stored_rel_path=rel, parse={"status": "OK", "sku": "23553000"}, match={"status": "UNPARSEABLE"},
        ingest_state="quarantined", quarantine_reason="PLY_TRUNCATED",
        quarantine_detail={"original_name": "23553000_B_FULL_SCAN 1.ply", "detail": "cut"},
    )  # fmt: skip
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(data)
    out = poll(make_poller(env), 4)
    assert [(o.action, o.reason) for o in out] == [("quarantined", "PLY_TRUNCATED")]
    assert (root / rel).read_bytes() == data and not (inbox / "23553000_B_FULL_SCAN 1.ply").exists()
    reason = json.loads((root / "quarantine" / "2026-10-17" / f"{sha}.reason.json").read_text())
    assert reason["reason"] == "PLY_TRUNCATED" and reason["scan_file_id"] == store.list_scans()[0]["scan_file_id"]
    assert "SCAN_DUPLICATE" not in kinds(store)


def test_copy_leave_crash_before_source_record_is_a_resume(env):
    """copy_leave crash between the copy and record_ingest_source: no spurious duplicate."""
    inbox, root, store = env
    data = ply(seed=71)
    sha = sha256_file_bytes(data)
    src = inbox / "23553000_B_FULL_SCAN 1.ply"
    src.write_bytes(data)
    past = time.time() - 60
    os.utime(src, (past, past))  # written (and stable) before the crashed ingest registered it
    rel = f"archive/scans/2026/10/{sha}.ply"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(data)
    store.register_scan(
        sha256=sha, original_name=src.name, size_bytes=len(data), station_id="cell-1", stored_rel_path=rel,
        parse={"status": "OK", "sku": "23553000"}, match={"status": "UNPARSEABLE", "reason": "RESOLUTION_PENDING"},
        format="PLY_BINARY_LE",
    )  # fmt: skip
    poller = make_poller(env, source_policy="copy_leave")
    out = poller.startup_rescan() + poll(poller, 6)
    assert [o.action for o in out] == ["enqueued", "enqueued"]  # pending row resumed, then the source recorded
    assert len(store.list_jobs()) == 1 and "SCAN_DUPLICATE" not in kinds(store)
    st = os.stat(src)
    assert store.has_ingest_source("cell-1", str(src), st.st_size, st.st_mtime_ns)
    assert poll(make_poller(env, source_policy="copy_leave"), 6) == []


def test_same_name_redrop_after_processing_is_still_a_duplicate(env):
    inbox, _root, store = env
    data = ply(seed=72)
    name = "23553000_B_FULL_SCAN 1.ply"
    (inbox / name).write_bytes(data)
    poller = make_poller(env)
    assert [o.action for o in poll(poller, 4)] == ["enqueued"]
    time.sleep(1.1)  # the re-drop is modified after the first ingest registered it
    (inbox / name).write_bytes(data)
    out = poll(poller, 4)
    assert [(o.action, o.reason) for o in out] == [("duplicate", "SCAN_DUPLICATE")]
    assert kinds(store).count("SCAN_DUPLICATE") == 1


def test_resolution_that_never_applies_is_quarantined(env, monkeypatch):
    inbox, root, store = env
    poller = make_poller(env)

    def broken(*_a, **_kw):
        raise RuntimeError("catalog recipe_id does not belong to its SKU")

    monkeypatch.setattr(store, "match_and_enqueue", broken)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=81))
    out = poll(poller, 4)
    assert [o.action for o in out] == ["error"]
    assert store.list_scans()[0]["ingest_state"] == "received"
    assert poller._resume_pending(force=True) == [] and poller._resume_pending(force=True) == []
    out = poller._resume_pending(force=True)  # the 3rd failed apply quarantines it
    assert [(o.action, o.reason, o.detail) for o in out] == [("quarantined", "PART_NOT_READY", "RESOLVE_APPLY_FAILED")]
    entry = store.list_quarantine()[0]
    assert entry["quarantine_reason"] == "PART_NOT_READY" and entry["resolvable"] is True
    assert "RESOLVE_APPLY_FAILED" in entry["quarantine_detail"]["detail"]
    sha = entry["scan_file"]["sha256"]
    assert list((root / "quarantine").rglob(f"{sha}.reason.json"))
    assert poller._resume_pending(force=True) == []  # no more retries


def test_file_changed_while_hashing_is_not_ingested(env, monkeypatch):
    inbox, _root, store = env
    poller = make_poller(env)
    target = inbox / "23553000_B_FULL_SCAN 1.ply"
    original = ply(seed=91)
    target.write_bytes(original)
    real = hotfolder_mod.sha256_file
    grown = []

    def growing(path):
        if Path(path).name == target.name and not grown:
            grown.append(True)
            with open(target, "ab") as fh:
                fh.write(b"\x00" * 16)  # the writer appends while we hash
        return real(path)

    monkeypatch.setattr(hotfolder_mod, "sha256_file", growing)
    assert poll(poller, 4) == []  # first sighting + 3 stable polls: ready, hashed, changed underneath
    assert grown and store.list_scans() == []
    (watch,) = poller._watch.values()
    assert watch.stable == 0  # must settle again from scratch
    out = poll(poller, 4)
    assert [o.action for o in out] == ["enqueued"]
    assert store.list_scans()[0]["size_bytes"] == len(original) + 16  # the settled bytes, hashed once


def test_unsupported_and_empty_content_quarantined_not_resolvable(env):
    """Content, not the extension, decides: an OFF mesh named .ply and a
    zero-vertex PLY are quarantined with the probe reason, and an operator
    cannot assign them a recipe (they would only fail in the engine)."""
    inbox, root, store = env
    poller = make_poller(env)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(b"OFF\n3 1 0\n" + b"0 0 0\n" * 3 + b"3 0 1 2\n")
    (inbox / "38172000_01_SCAN_2 1.ply").write_bytes(ply(n_vertices=0, faces=False))
    got = sorted((o.original_name, o.action, o.reason) for o in poll(poller, 4))
    assert got == [
        ("23553000_B_FULL_SCAN 1.ply", "quarantined", "UNSUPPORTED_FORMAT"),
        ("38172000_01_SCAN_2 1.ply", "quarantined", "EMPTY_SCAN"),
    ]
    assert store.list_jobs() == []
    entries = store.list_quarantine()
    assert {e["quarantine_reason"] for e in entries} == {"UNSUPPORTED_FORMAT", "EMPTY_SCAN"}
    assert not any(e["resolvable"] for e in entries)
