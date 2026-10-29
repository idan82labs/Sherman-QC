"""C02 hotfolder edge cases added with the station part picker / hotfolder E2E
(2026-10-28): undersized files are quarantined (EMPTY_SCAN) instead of waiting
forever, a failing archive (permission denied, disk full) is retried after a
back-off and never loses the file, orphan ``.done`` sidecars are removed,
foreign files are left untouched and counted, and Hebrew / space filenames
ingest unchanged. Synthetic tiny PLYs in tmp_path only (helpers from
``test_prod_hotfolder.py``); no source assets are read.
"""
from __future__ import annotations

import errno
import os
import time

import pytest

from apps.production_line.ingest import hotfolder as hotfolder_mod
from apps.production_line.ingest.hotfolder import HotfolderConfigError
from tests.test_prod_hotfolder import env, kinds, make_poller, ply, poll  # noqa: F401  (env is a fixture)


def test_zero_byte_and_undersized_files_are_quarantined_after_the_grace(env):  # noqa: F811
    inbox, root, store = env
    poller = make_poller(env, min_size_bytes=1024, undersize_grace_s=0.3)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(b"")
    (inbox / "23553000_B_SCAN_1 1.ply").write_bytes(ply(4))  # a few bytes: below min_size_bytes
    assert poll(poller, 5) == []  # inside the grace: a writer may still be starting
    assert store.list_scans() == []
    time.sleep(0.35)
    out = sorted((o.original_name, o.action, o.reason) for o in poll(poller, 3))
    assert out == [
        ("23553000_B_FULL_SCAN 1.ply", "quarantined", "EMPTY_SCAN"),
        ("23553000_B_SCAN_1 1.ply", "quarantined", "EMPTY_SCAN"),
    ]
    assert store.list_jobs() == [] and not any(e["resolvable"] for e in store.list_quarantine())
    assert not any(inbox.iterdir())  # moved to quarantine/, nothing left behind
    rows = store.list_scans()
    assert all(r["stored_rel_path"].startswith("quarantine/") for r in rows)
    assert all((root / r["stored_rel_path"]).is_file() for r in rows)
    assert "below min_size_bytes" in (store.list_quarantine()[0]["quarantine_detail"]["detail"])


def test_growing_file_is_never_taken_for_undersized(env):  # noqa: F811
    inbox, _root, store = env
    poller = make_poller(env, min_size_bytes=1024, undersize_grace_s=0.2)
    target = inbox / "23553000_B_FULL_SCAN 1.ply"
    data = ply(400, seed=5)
    with open(target, "wb") as fh:
        for i in range(0, len(data), 300):
            fh.write(data[i:i + 300])
            fh.flush()
            assert poller.poll_once() == []
            time.sleep(0.08)
    out = poll(poller, 5)
    assert [(o.action, o.reason) for o in out] == [("enqueued", None)]
    assert store.list_scans()[0]["size_bytes"] == len(data)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_permission_denied_archive_is_retried_and_never_loses_the_file(env):  # noqa: F811
    inbox, root, store = env
    poller = make_poller(env, failed_retry_s=0.3)
    # The archive month directory exists but is read-only (a full / locked share).
    now = poller._clock()
    month = root / "archive" / "scans" / f"{now:%Y}" / f"{now:%m}"
    month.mkdir(parents=True)
    month.chmod(0o555)
    try:
        src = inbox / "23553000_B_FULL_SCAN 1.ply"
        src.write_bytes(ply(seed=71))
        out = poll(poller, 12)
        assert [o.action for o in out] == ["error", "error", "error"]  # three tries, then a back-off
        assert src.exists()  # never lost
        assert poll(poller, 5) == []  # backing off
        month.chmod(0o755)  # the operator frees the share
        time.sleep(0.35)
        out = poll(poller, 6)
        assert [o.action for o in out] == ["enqueued"]
    finally:
        month.chmod(0o755)
    assert not src.exists()
    assert len(store.list_scans()) == 1 and len(store.list_jobs()) == 1  # exactly once
    assert kinds(store).count("SCAN_RECEIVED") == 1


def test_disk_full_while_copying_is_retried(env, monkeypatch):  # noqa: F811
    inbox, root, store = env
    poller = make_poller(env, failed_retry_s=0.2, source_policy="copy_leave")
    real_copy = hotfolder_mod.shutil.copyfileobj
    state = {"full": True}

    def copy(src, dst, length=0):
        if state["full"]:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copy(src, dst, length)

    monkeypatch.setattr(hotfolder_mod.shutil, "copyfileobj", copy)
    src = inbox / "23553000_B_FULL_SCAN 1.ply"
    src.write_bytes(ply(seed=72))
    out = poll(poller, 12)
    assert [o.action for o in out] == ["error", "error", "error"]
    assert not list(root.rglob("*.sherman-partial"))  # no half copy left behind
    state["full"] = False
    time.sleep(0.25)
    out = poll(poller, 6)
    assert [o.action for o in out] == ["enqueued"]
    assert src.exists()  # copy_leave: the source stays, recorded as ingested
    assert poll(poller, 5) == [] and len(store.list_jobs()) == 1


def test_orphan_sidecar_removed_after_the_grace_and_foreign_files_untouched(env):  # noqa: F811
    inbox, _root, store = env
    poller = make_poller(env, truncated_grace_s=0.2)
    (inbox / "23553000_B_FULL_SCAN 1.ply").write_bytes(ply(seed=81))
    out = poll(poller, 4)
    assert [o.action for o in out] == ["enqueued"]
    # The scanner's marker arrives after the scan was already taken.
    late = inbox / "23553000_B_FULL_SCAN 1.ply.done"
    late.write_text("")
    notes = inbox / "10839000_01 notes.txt"
    notes.write_text("operator notes")
    poll(poller, 3)
    assert late.exists()  # inside the grace
    time.sleep(0.25)
    poll(poller, 2)
    assert not late.exists()
    assert notes.read_text() == "operator notes"  # never touched
    assert poller.ignored_files == 1
    assert len(store.list_scans()) == 1


def test_hebrew_and_space_filenames_ingest_unchanged(env):  # noqa: F811
    inbox, _root, store = env
    poller = make_poller(env)
    good = "23553000_B סריקה מלאה 1.ply"
    unknown = "סריקה ללא מקט 1.ply"
    (inbox / good).write_bytes(ply(seed=91))
    (inbox / unknown).write_bytes(ply(seed=92))
    out = {o.original_name: o for o in poll(poller, 4)}
    assert out[good].action == "enqueued"
    assert out[unknown].action == "quarantined" and out[unknown].reason in ("UNMATCHED_SKU", "UNPARSEABLE_FILENAME")
    names = {r["original_name"] for r in store.list_scans()}
    assert names == {good, unknown}


def test_new_config_values_are_validated(env):  # noqa: F811
    for bad in ({"undersize_grace_s": -1}, {"failed_retry_s": -1}):
        with pytest.raises(HotfolderConfigError):
            make_poller(env, **bad)
