"""C12 ops: backup/restore round trip and retention (tmp_path only; no real data)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sqlite3
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.production_line.ops import backup as bk
from apps.production_line.ops import retention as rt
from apps.production_line.settings import LinePaths, ensure_layout
from apps.production_line.store.line_store import LineStore, iso_utc

REPO = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    """sha256 of every regular file; SQLite's transient -wal/-shm companions (created by any
    reader of a WAL database) are not data files and are left out."""
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink() and not p.name.endswith(("-wal", "-shm"))}


def _dump(db: Path) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return [line for line in conn.iterdump()]
    finally:
        conn.close()


def _mkdb(path: Path, rows: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t(v) VALUES (?)", [(f"row-{i}",) for i in range(rows)])
    conn.commit()
    conn.close()


def _register(store: LineStore, sha: str, rel: str, name: str = "SYN-1_A_FULL_SCAN 1.ply") -> str:
    match = {"status": "MATCHED", "recipe_id": "SYN-1@v1", "recipe_version": 1, "recipe_sha256": SHA_A}
    row, _ = store.register_scan(sha256=sha, original_name=name, size_bytes=10, station_id="cell-1",
                                 stored_rel_path=rel, parse={"status": "OK", "sku": "SYN-1"}, match=match,
                                 format="PLY_BINARY_LE")
    job_id, _ = store.match_and_enqueue(row["scan_file_id"], match=match, sku="SYN-1", compiled_part_sha256="c" * 64)
    return job_id


@pytest.fixture()
def install(tmp_path):
    """A small but realistic installation: line store (WAL), C05/C04 DBs, recipes, config, app copy DBs."""
    root = tmp_path / "data_root"
    paths = ensure_layout(LinePaths.for_root(root))
    store = LineStore(paths.db_line)
    _register(store, "1" * 64, "archive/scans/2026/10/" + "1" * 64 + ".ply")
    store.append_event("DEVICE_TOKEN_ISSUED", station_id="cell-1", payload={"token_id": "dt_0123456789abcdef"})
    _mkdb(paths.db_recipes, 25)
    _mkdb(paths.db_security, 3)
    (root / "recipes" / "cad" / f"{SHA_A}.step").write_bytes(b"ISO-10303-21;\n" + os.urandom(2048))
    comp = root / "recipes" / "compiled" / SHA_A
    (comp / "stages").mkdir(parents=True)
    (comp / "compiled_part.json").write_text('{"status": "COMPILED"}\n')
    (comp / "stages" / "S0.npz").write_bytes(os.urandom(4096))
    (root / "config" / "f1_program_status.json").write_text('{"status": "fail_closed"}\n')
    (root / "secrets" / "stream_tokens.key").write_bytes(os.urandom(32))
    app = tmp_path / "app"
    (app / "data").mkdir(parents=True)
    _mkdb(app / "data" / "users.db", 2)
    _mkdb(app / "data" / "qc_jobs.db", 7)
    return root, app, store


# ---------------------------------------------------------------------------- backup


def test_backup_round_trip_restores_identical_dbs_and_files(install, tmp_path):
    root, app, store = install
    # A write still in the WAL (not checkpointed) must be captured by the online backup.
    store.append_event("DEVICE_TOKEN_REVOKED", station_id="cell-1", payload={"token_id": "dt_0123456789abcdef"})
    out = tmp_path / "backups"
    summary = bk.create_backup(root, out, app_root=app)
    archive = Path(summary["backup"])
    assert archive.is_file() and archive.parent == out
    assert oct(archive.stat().st_mode & 0o777) == "0o600"
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    assert sorted(summary["databases"]) == ["app/data/qc_jobs.db", "app/data/users.db", "db/line.sqlite3",
                                            "db/recipes.sqlite3", "db/security.sqlite3"]
    assert "secrets.tar.gz" not in summary["trees"]  # secrets are opt-in

    assert bk.verify_backup(archive)["status"] == "verified"

    new_root, new_app = tmp_path / "restored_root", tmp_path / "restored_app"
    res = bk.restore_backup(archive, new_root, app_root=new_app)
    assert res["status"] == "restored"
    for name in ("line.sqlite3", "recipes.sqlite3", "security.sqlite3"):
        restored = new_root / "db" / name
        conn = sqlite3.connect(restored)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        conn.close()
        assert _dump(restored) == _dump(root / "db" / name), name
    for name in ("users.db", "qc_jobs.db"):
        assert _dump(new_app / "data" / name) == _dump(app / "data" / name)
    # the line DB goes back to WAL, as the store expects
    conn = sqlite3.connect(new_root / "db" / "line.sqlite3")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    kinds = [r[0] for r in conn.execute("SELECT kind FROM events ORDER BY id")]
    conn.close()
    assert "DEVICE_TOKEN_REVOKED" in kinds
    assert _tree_hashes(new_root / "recipes") == _tree_hashes(root / "recipes")
    assert _tree_hashes(new_root / "config") == _tree_hashes(root / "config")
    assert not (new_root / "secrets" / "stream_tokens.key").exists()
    for d in LinePaths.for_root(new_root).layout_dirs():
        assert oct(d.stat().st_mode & 0o777) == "0o700", d
    # the restored store opens and reads back the job
    assert LineStore(new_root / "db" / "line.sqlite3").counts()


def test_backup_include_secrets_and_job_results(install, tmp_path):
    root, app, _ = install
    ws = root / "jobs" / ("line_" + "0" * 32)
    (ws / "result").mkdir(parents=True)
    (ws / "job.json").write_text("{}")
    (ws / "result" / "shopfloor_result.json").write_text('{"x": 1}')
    (ws / "result" / "report.pdf").write_bytes(b"%PDF-1.7 fake")
    (ws / "result" / "viewer_scan.ply").write_bytes(os.urandom(1000))  # heavy: never in a backup
    summary = bk.create_backup(root, tmp_path / "b", app_root=app, include_secrets=True, include_job_results=True)
    assert summary["trees"]["secrets.tar.gz"] == 1
    assert summary["trees"]["job_results.tar.gz"] == 3
    new_root = tmp_path / "r"
    bk.restore_backup(Path(summary["backup"]), new_root, app_root=tmp_path / "ra")
    assert _sha(new_root / "secrets" / "stream_tokens.key") == _sha(root / "secrets" / "stream_tokens.key")
    assert (new_root / "jobs" / ws.name / "result" / "report.pdf").read_bytes() == b"%PDF-1.7 fake"
    assert not (new_root / "jobs" / ws.name / "result" / "viewer_scan.ply").exists()


def test_restore_refuses_non_empty_targets(install, tmp_path):
    root, app, _ = install
    archive = Path(bk.create_backup(root, tmp_path / "b", app_root=app)["backup"])
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    with pytest.raises(bk.BackupError) as e:
        bk.restore_backup(archive, busy, app_root=tmp_path / "fresh_app")
    assert e.value.code == "TARGET_NOT_EMPTY"
    assert (busy / "keep.txt").read_text() == "x"
    with pytest.raises(bk.BackupError) as e:
        bk.restore_backup(archive, tmp_path / "fresh_root", app_root=app)  # app/data holds DBs
    assert e.value.code == "TARGET_NOT_EMPTY"
    assert not (tmp_path / "fresh_root").exists()
    with pytest.raises(bk.BackupError) as e:
        bk.restore_backup(archive, tmp_path / "fresh_root2")  # app DBs in the backup need --app-root
    assert e.value.code == "TARGET_INVALID"


def _rewrite_member(archive: Path, name: str, mutate) -> Path:
    """Copy ``archive`` with member ``name`` replaced by ``mutate(bytes)``; manifest unchanged."""
    out = archive.with_name("tampered.tar.gz")
    with tarfile.open(archive, "r:gz") as src, tarfile.open(out, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name == name:
                data = mutate(data)
            info = tarfile.TarInfo(m.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    return out


def test_tampered_backup_is_refused_before_anything_is_written(install, tmp_path):
    root, app, _ = install
    archive = Path(bk.create_backup(root, tmp_path / "b", app_root=app)["backup"])

    def flip(data: bytes) -> bytes:
        return data[:-1] + bytes([data[-1] ^ 0xFF])

    bad = _rewrite_member(archive, "db/recipes.sqlite3", flip)
    with pytest.raises(bk.BackupError) as e:
        bk.verify_backup(bad)
    assert e.value.code == "HASH_MISMATCH"
    target = tmp_path / "never"
    with pytest.raises(bk.BackupError):
        bk.restore_backup(bad, target, app_root=tmp_path / "never_app")
    assert not target.exists() or not any(target.iterdir())

    # a recipe file altered inside recipes.tar.gz with the outer hash fixed up is still caught by the index
    def alter_tree(data: bytes) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t, tarfile.open(fileobj=buf, mode="w:gz") as w:
            for m in t.getmembers():
                body = t.extractfile(m).read()
                if m.name.endswith(".step"):
                    body = body + b"tampered"
                info = tarfile.TarInfo(m.name)
                info.size = len(body)
                w.addfile(info, io.BytesIO(body))
        return buf.getvalue()

    with tarfile.open(archive, "r:gz") as t:
        manifest = json.loads(t.extractfile("manifest.json").read())
        new_tree = alter_tree(t.extractfile("recipes.tar.gz").read())
    for e_ in manifest["entries"]:
        if e_["name"] == "recipes.tar.gz":
            e_["sha256"] = hashlib.sha256(new_tree).hexdigest()
    forged = tmp_path / "forged.tar.gz"
    with tarfile.open(archive, "r:gz") as src, tarfile.open(forged, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name == "manifest.json":
                data = json.dumps(manifest).encode()
            elif m.name == "recipes.tar.gz":
                data = new_tree
            info = tarfile.TarInfo(m.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    with pytest.raises(bk.BackupError) as e:
        bk.verify_backup(forged)
    assert e.value.code == "HASH_MISMATCH"
    assert bk.main(["verify", "--backup", str(forged)]) == 2


def test_restore_into_root_owned_parent_stages_elsewhere(install, tmp_path):
    """Runbook flow: the parent of the data root and of app/data is not writable by the
    restoring user; the targets are pre-created empty and owned by that user."""
    root, app, _ = install
    archive = Path(bk.create_backup(root, tmp_path / "b", app_root=app)["backup"])
    locked = tmp_path / "var_lib"
    locked.mkdir()
    new_root = locked / "sherman-qc"
    app_copy = tmp_path / "opt_app"
    app_copy.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    # without a pre-created target the restore is refused cleanly, not an internal error
    os.chmod(locked, 0o555)
    try:
        with pytest.raises(bk.BackupError) as e:
            bk.restore_backup(archive, new_root, app_root=app_copy, staging_dir=staging)
        assert e.value.code == "TARGET_INVALID" and "pre-create" in e.value.message
        os.chmod(locked, 0o755)
        new_root.mkdir(mode=0o700)
        (app_copy / "data").mkdir(mode=0o700)
        os.chmod(locked, 0o555)
        os.chmod(app_copy, 0o555)
        res = bk.restore_backup(archive, new_root, app_root=app_copy, staging_dir=staging)
        assert res["status"] == "restored"
        assert _dump(new_root / "db" / "line.sqlite3") == _dump(root / "db" / "line.sqlite3")
        assert _dump(app_copy / "data" / "users.db") == _dump(app / "data" / "users.db")
        assert sorted(p.name for p in locked.iterdir()) == ["sherman-qc"]  # nothing staged beside it
        assert list(staging.iterdir()) == []  # the private staging dir is removed
    finally:
        os.chmod(locked, 0o755)
        os.chmod(app_copy, 0o755)
    # the CLI path defaults to the system temp dir and reports the same refusal as exit 2
    assert bk.main(["restore", "--backup", str(archive), "--data-root", str(tmp_path / "x" / "y"),
                    "--staging-dir", str(tmp_path / "no-such-dir")]) == 2


def _forge(archive: Path, out: Path, tree_name: str, extra: dict[str, bytes]) -> Path:
    """A self-consistent archive whose ``tree_name`` gains ``extra`` files (hashes fixed up)."""
    with tarfile.open(archive, "r:gz") as t:
        manifest = json.loads(t.extractfile("manifest.json").read())
        tree_bytes = t.extractfile(tree_name).read()
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tree_bytes), mode="r:gz") as src, tarfile.open(fileobj=buf, mode="w:gz") as w:
        for m in src.getmembers():
            w.addfile(m, io.BytesIO(src.extractfile(m).read()))
        for rel, body in extra.items():
            info = tarfile.TarInfo(rel)
            info.size = len(body)
            w.addfile(info, io.BytesIO(body))
    new_tree = buf.getvalue()
    for e_ in manifest["entries"]:
        if e_["name"] == tree_name:
            e_["sha256"] = hashlib.sha256(new_tree).hexdigest()
            e_["files"] += [{"path": rel, "sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}
                            for rel, b in extra.items()]
    with tarfile.open(archive, "r:gz") as src, tarfile.open(out, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name == "manifest.json":
                data = json.dumps(manifest).encode()
            elif m.name == tree_name:
                data = new_tree
            info = tarfile.TarInfo(m.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    return out


def test_restore_enforces_tree_roots_and_unique_destinations(install, tmp_path):
    root, app, _ = install
    archive = Path(bk.create_backup(root, tmp_path / "b", app_root=app)["backup"])
    # a recipes member that would overwrite the restored line DB, and one that escapes into secrets/
    dup = f"recipes/cad/{SHA_A}.step"  # already in the tree: a second copy would silently win
    for i, rel in enumerate(("db/line.sqlite3", "secrets/stream_tokens.key", "config/f1_program_status.json", dup,
                             dup.upper().replace("RECIPES/", "recipes/"))):
        forged = _forge(archive, tmp_path / f"forged-{i}.tar.gz", "recipes.tar.gz", {rel: b"evil"})
        with pytest.raises(bk.BackupError) as e:
            bk.verify_backup(forged)
        assert e.value.code == "BACKUP_INVALID", rel
        target = tmp_path / f"t{i}"
        with pytest.raises(bk.BackupError):
            bk.restore_backup(forged, target, app_root=tmp_path / f"ta{i}")
        assert not target.exists()
    # a well-formed extra file inside its own root is still accepted (the check is not over-broad)
    ok = _forge(archive, tmp_path / "ok.tar.gz", "recipes.tar.gz", {"recipes/cad/extra.step": b"fine"})
    assert bk.verify_backup(ok)["status"] == "verified"


def test_expect_sha256_pins_the_archive(install, tmp_path):
    root, app, _ = install
    summary = bk.create_backup(root, tmp_path / "b", app_root=app)
    archive = Path(summary["backup"])
    assert bk.verify_backup(archive, expect_sha256=summary["sha256"])["sha256_checked"] is True
    forged = _forge(archive, tmp_path / "forged.tar.gz", "recipes.tar.gz", {"recipes/cad/x.step": b"x"})
    assert bk.verify_backup(forged)["status"] == "verified"  # internally consistent ...
    with pytest.raises(bk.BackupError) as e:  # ... but not the recorded archive
        bk.verify_backup(forged, expect_sha256=summary["sha256"])
    assert e.value.code == "HASH_MISMATCH"
    with pytest.raises(bk.BackupError) as e:
        bk.restore_backup(forged, tmp_path / "r", app_root=tmp_path / "ra", expect_sha256=summary["sha256"])
    assert e.value.code == "HASH_MISMATCH" and not (tmp_path / "r").exists()
    assert bk.main(["verify", "--backup", str(archive), "--expect-sha256", "z" * 64]) == 2


def test_backup_refuses_out_inside_repo_or_data_root(install, tmp_path):
    root, app, _ = install
    inside_repo = REPO / "output" / "c12-backup-test-never-created"
    with pytest.raises(bk.BackupError) as e:
        bk.create_backup(root, inside_repo, app_root=app)
    assert e.value.code == "OUT_FORBIDDEN"
    assert not inside_repo.exists()
    with pytest.raises(bk.BackupError) as e:
        bk.create_backup(root, root / "db" / "bk", app_root=app)
    assert e.value.code == "OUT_FORBIDDEN"
    with pytest.raises(bk.BackupError) as e:
        bk.create_backup(tmp_path / "not_a_root", tmp_path / "b")
    assert e.value.code == "DATA_ROOT_INVALID"


def test_backup_skips_symlinks_in_trees(install, tmp_path):
    root, app, _ = install
    secret_elsewhere = tmp_path / "elsewhere.txt"
    secret_elsewhere.write_text("must not be archived")
    os.symlink(secret_elsewhere, root / "recipes" / "cad" / "link.step")
    summary = bk.create_backup(root, tmp_path / "b", app_root=app)
    assert summary["skipped_symlinks"] == 1
    with tarfile.open(summary["backup"], "r:gz") as t:
        blob = t.extractfile("recipes.tar.gz").read()
    assert b"must not be archived" not in gzip.decompress(blob)


# ---------------------------------------------------------------------------- retention

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
HEAVY = ("input/scan.ply", "engine/registered_scan.ply", "engine/deviations.f32", "engine/segment_labels.u16",
         "engine/display/stage_mesh.ply", "compiled/stages/S1.npz", "result/viewer_scan.ply",
         "result/viewer_deviations.f32", "result/stage_mesh.ply", "result/final_mesh.ply", "result/overview.png")
KEPT = ("job.json", "recipe.json", "input/context.json", "compiled/compiled_part.json", "engine/result.json",
        "engine/bend_geometry.json", "engine/engine.log", "result/shopfloor_result.json", "result/bend_geometry.json",
        "result/report.pdf", "result/postprocess.log", "logs/attempt-1/engine_step.log")


def _workspace(root: Path, job_id: str) -> Path:
    ws = root / "jobs" / job_id
    for rel in HEAVY + KEPT:
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(256))
    return ws


def _set(db: Path, sql: str, args: tuple) -> None:
    conn = sqlite3.connect(db)
    conn.execute(sql, args)
    conn.commit()
    conn.close()


@pytest.fixture()
def line(tmp_path):
    root = tmp_path / "root"
    paths = ensure_layout(LinePaths.for_root(root))
    store = LineStore(paths.db_line)
    db = paths.db_line
    old = iso_utc(NOW - timedelta(days=100))
    recent = iso_utc(NOW - timedelta(days=2))
    jobs = {}
    for name, sha, finished, status in (
        ("old_done", "1" * 64, old, "completed"),
        ("old_failed", "2" * 64, old, "failed"),
        ("recent_done", "3" * 64, recent, "completed"),
        ("old_running", "4" * 64, None, "running"),
    ):
        rel = f"archive/scans/2026/07/{sha}.ply"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(os.urandom(512))
        job_id = _register(store, sha, rel, name=f"{name}.ply")
        _set(db, "UPDATE jobs SET execution_status=?, finished_at=?, created_at=? WHERE job_id=?",
             (status, finished, old, job_id))
        _set(db, "UPDATE scan_files SET received_at=? WHERE sha256=?", (old, sha))
        jobs[name] = (job_id, root / rel, _workspace(root, job_id))
    # an old quarantined (still assignable) scan stays
    qsha = "5" * 64
    qrel = f"archive/scans/2026/07/{qsha}.ply"
    (root / qrel).write_bytes(os.urandom(64))
    store.register_scan(sha256=qsha, original_name="UNKNOWN_SKU.ply", size_bytes=64, station_id="cell-1",
                        stored_rel_path=qrel, parse={"status": "UNMATCHED"}, match={"status": "UNMATCHED"},
                        ingest_state="quarantined", quarantine_reason="UNMATCHED_SKU")
    _set(db, "UPDATE scan_files SET received_at=? WHERE sha256=?", (old, qsha))
    # events: two old, the rest recent
    conn = sqlite3.connect(db)
    conn.execute("UPDATE events SET ts=?", (recent,))
    ids = [r[0] for r in conn.execute("SELECT id FROM events ORDER BY id LIMIT 2")]
    conn.executemany("UPDATE events SET ts=? WHERE id=?", [(old, i) for i in ids])
    conn.commit()
    conn.close()
    return root, store, jobs, root / qrel


def test_retention_dry_run_deletes_nothing(line):
    root, store, jobs, _ = line
    before = _tree_hashes(root)
    events_before = store.last_event_id()
    plan = rt.plan_retention(root, 30, log_days=30, now=NOW)
    summary = plan.summary()
    assert summary["prune"]["workspace_heavy"]["files"] == 2 * len(HEAVY)  # old_done + old_failed
    assert summary["prune"]["archived_scan"]["files"] == 2
    assert summary["events"]["delete"] == 2
    assert _tree_hashes(root) == before
    assert rt.main(["--data-root", str(root), "--days", "30", "--log-days", "30"]) == 0
    assert _tree_hashes(root) == before
    conn = sqlite3.connect(root / "db" / "line.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == events_before
    conn.close()


def test_retention_apply_keeps_results_and_reports(line):
    root, store, jobs, qscan = line
    plan = rt.plan_retention(root, 30, log_days=30, now=NOW)
    applied = rt.apply_plan(plan)
    assert applied["errors"] == []
    assert applied["deleted_files"] == 2 * len(HEAVY) + 2
    assert applied["events_deleted"] == 2
    for name in ("old_done", "old_failed"):
        _, scan, ws = jobs[name]
        assert not scan.exists()
        for rel in HEAVY:
            assert not (ws / rel).exists(), rel
        for rel in KEPT:
            assert (ws / rel).is_file(), rel
        assert not (ws / "engine" / "display").exists()  # emptied pruned subtree is removed
    for name in ("recent_done", "old_running"):
        _, scan, ws = jobs[name]
        assert scan.exists(), name
        for rel in HEAVY + KEPT:
            assert (ws / rel).is_file(), (name, rel)
    assert qscan.exists()  # quarantined scans stay assignable
    # the store still works and the event ids keep growing
    last = store.last_event_id()
    store.append_event("DEVICE_TOKEN_ISSUED", station_id="cell-1", payload={"token_id": "dt_fedcba9876543210"})
    assert store.last_event_id() > last


def test_retention_never_follows_symlinks(line, tmp_path):
    root, _, jobs, _ = line
    outside = tmp_path / "outside.ply"
    outside.write_bytes(b"keep me")
    ws = jobs["old_done"][2]
    (ws / "result" / "overview.png").unlink()
    os.symlink(outside, ws / "result" / "overview.png")
    plan = rt.plan_retention(root, 30, log_days=30, now=NOW)
    assert any("overview.png" in s for s in plan.skipped_symlinks)
    rt.apply_plan(plan)
    assert outside.read_bytes() == b"keep me"
    assert (ws / "result" / "overview.png").is_symlink()


def test_retention_skips_events_on_unknown_schema(line):
    root, store, _, _ = line
    _set(root / "db" / "line.sqlite3", "UPDATE schema_meta SET version=? WHERE component='line_store'", (99,))
    plan = rt.plan_retention(root, 30, log_days=30, now=NOW)
    assert plan.events_status == "EVENTS_SCHEMA_UNKNOWN"
    applied = rt.apply_plan(plan)
    assert applied["events_deleted"] == 0
    conn = sqlite3.connect(root / "db" / "line.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM events WHERE ts < ?", (plan.log_cutoff,)).fetchone()[0] == 2
    conn.close()


def test_retention_refuses_bad_roots(tmp_path):
    with pytest.raises(rt.RetentionError) as e:
        rt.plan_retention(tmp_path / "empty", 30, log_days=30)
    assert e.value.code == "NOT_A_DATA_ROOT"
    with pytest.raises(rt.RetentionError) as e:
        rt.plan_retention(REPO / "output", 30, log_days=30)
    assert e.value.code == "DATA_ROOT_INVALID"
    with pytest.raises(rt.RetentionError) as e:
        rt.plan_retention(tmp_path, 0, log_days=30)
    assert e.value.code == "VALUE_INVALID"
    assert rt.main(["--data-root", str(tmp_path / "empty"), "--days", "30"]) == 2


def test_heavy_patterns_never_name_kept_files():
    for pattern in rt.HEAVY_WORKSPACE_PATTERNS:
        assert not rt._is_kept(pattern), pattern
    for rel in KEPT:
        assert rt._is_kept(rel), rel


def test_prune_backups_keeps_newest_and_touches_nothing_else(tmp_path):
    out = tmp_path / "b"
    out.mkdir()
    names = [f"sherman-qc-backup-202610{d:02d}T023000Z.tar.gz" for d in (1, 2, 3, 4)]
    for n in names:
        (out / n).write_bytes(b"x")
    (out / "notes.txt").write_text("keep")
    removed = bk.prune_backups(out, 2)
    assert removed == names[:2]
    assert sorted(p.name for p in out.iterdir()) == sorted(names[2:] + ["notes.txt"])
    with pytest.raises(bk.BackupError):
        bk.prune_backups(out, 0)
