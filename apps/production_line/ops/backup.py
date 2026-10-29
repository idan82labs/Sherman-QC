"""Backup and restore of a shop-floor line installation.

What is backed up (``create``):

=========================  =====================================================
``db/<name>.sqlite3``      every ``<data_root>/db/*.sqlite3`` (line, recipes,
                           security). Copied with the SQLite *online backup API*,
                           so a running line is captured consistently (WAL
                           included) without stopping it.
``app/data/qc_jobs.db``    the app copy's legacy job DB (``--app-root``).
``app/data/users.db``      the app copy's users DB (holds the bootstrap admin).
``recipes.tar.gz``         ``<data_root>/recipes/**``: frozen CAD, drawings,
                           flats and compiled parts (content addressed).
``config.tar.gz``          ``<data_root>/config/**`` (program status file etc.).
``secrets.tar.gz``         ``<data_root>/secrets/**``, only with
                           ``--include-secrets`` (stream-token keys; device
                           tokens are stored hashed in security.sqlite3 and do
                           not need them).
``job_results.tar.gz``     with ``--include-job-results``: each job's
                           ``job.json``, ``recipe.json``, ``input/context.json``,
                           ``result/*.json`` and ``result/report.pdf``.
``manifest.json``          schema ``sherman_line_backup_v1``: every member's
                           sha256 and size, each tree's per-file sha256 index,
                           each DB's ``PRAGMA integrity_check`` and source
                           journal mode, and what was missing.
=========================  =====================================================

Scan archives (``archive/scans``) and job workspaces are *not* backed up by
default. They are large, and :mod:`.retention` prunes them anyway. Back up the
data-root volume itself if raw scans must survive a disk loss.

The backup is one file, ``<out>/sherman-qc-backup-<UTC>.tar.gz`` (0600, in a
0700 directory). Symlinks inside the data root are never followed or archived;
they are listed in the manifest as ``skipped_symlinks``. ``--out`` may not be
inside the repository or inside ``db/``, ``recipes/``, ``config/`` or ``secrets/``.

``restore --backup FILE --data-root R --app-root A [--staging-dir S] [--expect-sha256 H]``:

- It refuses unless ``R`` is absent or an empty directory, and ``A/data`` is absent
  or empty (restore into a fresh install, before its first start). On an
  installed host the service user cannot create ``R`` or ``A/data`` itself
  (their parents are root-owned), so the runbook pre-creates both, empty and
  owned by the service user; a target that cannot be created is refused with
  ``TARGET_INVALID``, never an internal error.
- The backup is extracted and verified in a private (0700) staging directory:
  ``--staging-dir`` or, by default, the system temp directory. It is never
  staged next to ``R``, whose parent the restoring user usually cannot write.
  Files are then moved into place (``shutil.move`` copies across filesystems).
- Every member is verified against the manifest sha256 before anything is placed.
- Member names are a closed set: ``db/<name>.sqlite3``, ``app/data/qc_jobs.db``,
  ``app/data/users.db`` and the tree archives ``recipes``, ``config``,
  ``secrets``, ``job_results``. Every tree file path must lie under that tree's
  declared root (``recipes/...`` for ``recipes.tar.gz`` and so on), and no two
  entries may restore to the same destination, so a tree member can never
  overwrite a restored DB or land in another subtree.
- Tree archives are extracted with the tarfile ``data`` filter, and each file
  is re-hashed against the tree index.
- Every restored DB must pass ``PRAGMA integrity_check`` (``ok``), and its
  original journal mode (WAL for the line DBs) is restored.
- Only then are files moved into place, and the data-root layout is created
  with 0700 directories.

``verify --backup FILE [--expect-sha256 H]`` runs the same checks without restoring.

Trust model. The manifest hashes are self-referential: they prove that an
archive is internally consistent (no corruption, no partial edit), not who
made it. Someone who can write the backup directory can forge a consistent
archive. ``create`` prints the archive's sha256; record it off the machine
(the backup log, a ticket, the off-box copy's catalogue). ``verify`` and
``restore`` with ``--expect-sha256`` then refuse any other file
(``HASH_MISMATCH``). Without it, a restore trusts whoever could write the
backup directory, which is root and the service user only (0700).

``create --keep N`` (N >= 1) then deletes all but the newest N
``sherman-qc-backup-*.tar.gz`` files in ``--out`` (the daily timer uses it); it
never touches any other file.

Exit codes: 0 ok, 2 refused or invalid input (including a hash or integrity
mismatch), 3 internal error. Output is one JSON line.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import sqlite3
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

__all__ = [
    "BACKUP_SCHEMA",
    "BACKUP_TOOL_VERSION",
    "BackupError",
    "create_backup",
    "verify_backup",
    "restore_backup",
    "prune_backups",
    "main",
]

BACKUP_SCHEMA = "sherman_line_backup_v1"
BACKUP_TOOL_VERSION = "c12-backup-1.1.0"
APP_DB_NAMES: tuple[str, ...] = ("qc_jobs.db", "users.db")
#: Data-root subtrees archived as tar.gz members (name -> relative dir), in order.
TREES: tuple[tuple[str, str], ...] = (("recipes", "recipes"), ("config", "config"))
#: Every tree archive a backup may hold -> the data-root subtree its files must lie under.
TREE_ROOTS: dict[str, str] = {"recipes.tar.gz": "recipes", "config.tar.gz": "config",
                              "secrets.tar.gz": "secrets", "job_results.tar.gz": "jobs"}
_PROTECTED_OUT_SUBDIRS: tuple[str, ...] = ("db", "recipes", "config", "secrets", "jobs", "archive")
_CHUNK = 1024 * 1024


class BackupError(Exception):
    """A refused backup/restore. ``code`` is machine readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _real(p: Path) -> Path:
    return Path(os.path.realpath(os.fspath(p)))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _custody_reason(path: Path) -> Optional[str]:
    from apps.production_line.settings import forbidden_location_reason

    return forbidden_location_reason(path, _repo_root())


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def _sqlite_uri(path: Path, mode: str) -> str:
    from apps.production_line.health import sqlite_uri

    return sqlite_uri(path, mode)


def _online_backup(src_path: Path, dst_path: Path) -> dict[str, Any]:
    """Copy ``src_path`` into a new self-contained file with the SQLite backup API.
    Returns ``{journal_mode_source, integrity}``."""
    try:
        src = sqlite3.connect(_sqlite_uri(src_path, "ro"), uri=True, timeout=10)
        mode = str(src.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    except sqlite3.Error:
        # A WAL database opened read-only needs its -shm; fall back to rw (never creates: mode=rw).
        src = sqlite3.connect(_sqlite_uri(src_path, "rw"), uri=True, timeout=10)
        mode = str(src.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    try:
        src.execute("PRAGMA busy_timeout=10000")
        dst = sqlite3.connect(str(dst_path))
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")  # the copy carries no -wal/-shm companions
            integrity = str(dst.execute("PRAGMA integrity_check").fetchone()[0])
        finally:
            dst.close()
    finally:
        src.close()
    return {"journal_mode_source": mode, "integrity": integrity}


def _integrity(path: Path) -> str:
    conn = sqlite3.connect(_sqlite_uri(path, "ro"), uri=True)
    try:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        conn.close()


def _set_journal_mode(path: Path, mode: str) -> None:
    if mode not in ("wal", "delete"):
        return
    conn = sqlite3.connect(_sqlite_uri(path, "rw"), uri=True)
    try:
        conn.execute(f"PRAGMA journal_mode={mode.upper()}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------


@dataclass
class _TreeScan:
    files: list[tuple[str, Path]]  # (posix relpath under the tree's parent, absolute path)
    skipped_symlinks: list[str]


def _scan_tree(root: Path, base: Path, select: Optional[Iterable[str]] = None) -> _TreeScan:
    """Regular files under ``root`` (no symlinks followed), as paths relative to ``base``."""
    files: list[tuple[str, Path]] = []
    skipped: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return _TreeScan(files, skipped)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        for name in list(dirnames):
            if (d / name).is_symlink():
                skipped.append((d / name).relative_to(base).as_posix())
                dirnames.remove(name)
        dirnames.sort()
        for name in sorted(filenames):
            p = d / name
            rel = p.relative_to(base).as_posix()
            if p.is_symlink():
                skipped.append(rel)
                continue
            if not p.is_file():
                continue
            files.append((rel, p))
    if select is not None:
        keep = set(select)
        files = [f for f in files if f[0] in keep]
    return _TreeScan(sorted(files), sorted(skipped))


def _tar_tree(scan: _TreeScan, dest: Path) -> list[dict[str, Any]]:
    """Write ``scan`` as a deterministic tar.gz; returns the per-file index."""
    index: list[dict[str, Any]] = []
    with tarfile.open(dest, "w:gz", compresslevel=6) as tar:
        for rel, path in scan.files:
            data = path.read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
            index.append({"path": rel, "sha256": _sha256_bytes(data), "size_bytes": len(data)})
    return index


def _job_result_files(data_root: Path) -> list[str]:
    out: list[str] = []
    jobs = data_root / "jobs"
    if not jobs.is_dir():
        return out
    for ws in sorted(jobs.iterdir()):
        if ws.is_symlink() or not ws.is_dir():
            continue
        for rel in ("job.json", "recipe.json", "input/context.json", "result/report.pdf"):
            if (ws / rel).is_file() and not (ws / rel).is_symlink():
                out.append(f"jobs/{ws.name}/{rel}")
        res = ws / "result"
        if res.is_dir() and not res.is_symlink():
            for f in sorted(res.glob("*.json")):
                if f.is_file() and not f.is_symlink():
                    out.append(f"jobs/{ws.name}/result/{f.name}")
    return out


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def _check_out_dir(out: Path, data_root: Path) -> None:
    if not out.is_absolute():
        raise BackupError("OUT_INVALID", "--out must be an absolute path")
    reason = _custody_reason(out)
    if reason:
        raise BackupError("OUT_FORBIDDEN", f"--out {reason}")
    real_out = _real(out)
    real_root = _real(data_root)
    for sub in _PROTECTED_OUT_SUBDIRS:
        if _inside(real_out, real_root / sub):
            raise BackupError("OUT_FORBIDDEN", f"--out may not be inside the data root's {sub}/")


def create_backup(
    data_root: Path,
    out_dir: Path,
    *,
    app_root: Optional[Path] = None,
    include_secrets: bool = False,
    include_job_results: bool = False,
) -> dict[str, Any]:
    """Create one backup file under ``out_dir`` (module docstring). Returns the summary."""
    data_root = Path(data_root)
    if not data_root.is_absolute() or not (data_root / "db").is_dir():
        raise BackupError("DATA_ROOT_INVALID", "--data-root must be an absolute line data root (with db/)")
    reason = _custody_reason(data_root)
    if reason:
        raise BackupError("DATA_ROOT_INVALID", f"--data-root {reason}")
    _check_out_dir(Path(out_dir), data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    stamp = _utc_stamp()
    final = out_dir / f"sherman-qc-backup-{stamp}.tar.gz"
    if final.exists():
        raise BackupError("OUT_EXISTS", f"{final.name} already exists")
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".sherman-backup-", dir=out_dir) as tmp_s:
        tmp = Path(tmp_s)
        os.chmod(tmp, 0o700)
        members: list[tuple[str, Path]] = []

        db_sources = sorted(p for p in (data_root / "db").glob("*.sqlite3") if p.is_file() and not p.is_symlink())
        app_dbs: list[tuple[str, Path]] = []
        if app_root is not None:
            for name in APP_DB_NAMES:
                p = Path(app_root) / "data" / name
                if p.is_file() and not p.is_symlink():
                    app_dbs.append((f"app/data/{name}", p))
                else:
                    missing.append(f"app/data/{name}")
        for arcname, src in [(f"db/{p.name}", p) for p in db_sources] + app_dbs:
            staged = tmp / arcname.replace("/", "__")
            info = _online_backup(src, staged)
            if info["integrity"] != "ok":
                raise BackupError("DB_INTEGRITY", f"{arcname}: integrity_check returned {info['integrity']!r}")
            members.append((arcname, staged))
            entries.append({"name": arcname, "kind": "sqlite", "sha256": _sha256_file(staged),
                            "size_bytes": staged.stat().st_size, **info})
        if not db_sources:
            missing.append("db/*.sqlite3")

        trees = list(TREES) + ([("secrets", "secrets")] if include_secrets else [])
        for name, rel in trees:
            scan = _scan_tree(data_root / rel, data_root)
            skipped.extend(scan.skipped_symlinks)
            if not (data_root / rel).is_dir():
                missing.append(f"{rel}/")
            staged = tmp / f"{name}.tar.gz"
            index = _tar_tree(scan, staged)
            members.append((f"{name}.tar.gz", staged))
            entries.append({"name": f"{name}.tar.gz", "kind": "tree", "root": rel, "files": index,
                            "sha256": _sha256_file(staged), "size_bytes": staged.stat().st_size})
        if include_job_results:
            rels = _job_result_files(data_root)
            scan = _TreeScan([(r, data_root / r) for r in rels], [])
            staged = tmp / "job_results.tar.gz"
            index = _tar_tree(scan, staged)
            members.append(("job_results.tar.gz", staged))
            entries.append({"name": "job_results.tar.gz", "kind": "tree", "root": "jobs", "files": index,
                            "sha256": _sha256_file(staged), "size_bytes": staged.stat().st_size})

        manifest = {
            "schema": BACKUP_SCHEMA,
            "tool_version": BACKUP_TOOL_VERSION,
            "created_at": _utc_iso(),
            "host": socket.gethostname(),
            "data_root": str(data_root),
            "app_root": None if app_root is None else str(app_root),
            "options": {"include_secrets": include_secrets, "include_job_results": include_job_results},
            "entries": entries,
            "missing": sorted(missing),
            "skipped_symlinks": sorted(skipped),
        }
        mbytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
        partial = out_dir / f".{final.name}.partial"
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz", compresslevel=6) as tar:
                info = tarfile.TarInfo("manifest.json")
                info.size = len(mbytes)
                info.mode = 0o600
                info.mtime = 0
                tar.addfile(info, io.BytesIO(mbytes))
                for arcname, path in members:
                    tar.add(str(path), arcname=arcname, recursive=False,
                            filter=lambda ti: _reset_tarinfo(ti))
            with open(partial, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(partial, final)
        except BaseException:
            try:
                os.unlink(partial)
            except FileNotFoundError:
                pass
            raise
    os.chmod(final, 0o600)
    return {
        "status": "created",
        "backup": str(final),
        "sha256": _sha256_file(final),
        "databases": [e["name"] for e in entries if e["kind"] == "sqlite"],
        "trees": {e["name"]: len(e["files"]) for e in entries if e["kind"] == "tree"},
        "missing": sorted(missing),
        "skipped_symlinks": len(skipped),
    }


def prune_backups(out_dir: Path, keep: int) -> list[str]:
    """Delete all but the newest ``keep`` backup files in ``out_dir``; returns the removed names."""
    if keep < 1:
        raise BackupError("VALUE_INVALID", "--keep must be >= 1")
    files = sorted(p for p in Path(out_dir).glob("sherman-qc-backup-*.tar.gz") if p.is_file() and not p.is_symlink())
    removed = []
    for p in files[:-keep]:  # names sort by their UTC stamp
        p.unlink()
        removed.append(p.name)
    return removed


def _reset_tarinfo(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.mode = 0o600
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0
    return ti


# ---------------------------------------------------------------------------
# verify / restore
# ---------------------------------------------------------------------------


def _safe_member_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name


def _sqlite_entry_ok(name: str) -> bool:
    p = PurePosixPath(name)
    if len(p.parts) == 2 and p.parts[0] == "db":
        return p.suffix == ".sqlite3" and len(p.name) > len(".sqlite3")
    return len(p.parts) == 3 and p.parts[:2] == ("app", "data") and p.name in APP_DB_NAMES


def _check_manifest_layout(manifest: dict[str, Any]) -> None:
    """Refuse a manifest whose entries could restore outside their declared place."""
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BackupError("BACKUP_INVALID", "manifest has no entries")
    names: set[str] = set()
    destinations: dict[str, str] = {}

    def claim(dest: str, owner: str) -> None:
        key = dest.casefold()  # case-insensitive filesystems (Windows, default macOS)
        if key in destinations:
            raise BackupError("BACKUP_INVALID", f"{owner} and {destinations[key]} both restore to {dest!r}")
        destinations[key] = owner

    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise BackupError("BACKUP_INVALID", "manifest entry without a name")
        name = entry["name"]
        if name in names or not _safe_member_name(name):
            raise BackupError("BACKUP_INVALID", f"duplicate or unsafe entry name {name!r}")
        names.add(name)
        kind = entry.get("kind")
        if kind == "sqlite":
            if not _sqlite_entry_ok(name):
                raise BackupError("BACKUP_INVALID", f"unexpected database entry {name!r}")
            claim(name, name)
        elif kind == "tree":
            root = TREE_ROOTS.get(name)
            if root is None or entry.get("root") != root:
                raise BackupError("BACKUP_INVALID", f"tree {name!r} must declare root {root!r}")
            files = entry.get("files")
            if not isinstance(files, list):
                raise BackupError("BACKUP_INVALID", f"tree {name!r} has no file index")
            for f in files:
                rel = f.get("path") if isinstance(f, dict) else None
                if not isinstance(rel, str) or not _safe_member_name(rel) or not rel.startswith(root + "/"):
                    raise BackupError("BACKUP_INVALID", f"{name}: file {rel!r} is outside its root {root}/")
                claim(rel, f"{name}:{rel}")
        else:
            raise BackupError("BACKUP_INVALID", f"unknown entry kind {kind!r}")


def _check_expected_sha(backup: Path, expect_sha256: Optional[str]) -> None:
    if expect_sha256 is None:
        return
    want = expect_sha256.strip().lower()
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        raise BackupError("VALUE_INVALID", "--expect-sha256 must be 64 hex characters")
    if not Path(backup).is_file():
        raise BackupError("BACKUP_MISSING", f"{backup} is not a file")
    if _sha256_file(Path(backup)) != want:
        raise BackupError("HASH_MISMATCH", "the backup file's sha256 differs from the recorded value")


def _stage_and_verify(backup: Path, staging: Path) -> dict[str, Any]:
    """Extract ``backup`` into ``staging`` and verify everything; returns the manifest."""
    if not backup.is_file():
        raise BackupError("BACKUP_MISSING", f"{backup} is not a file")
    try:
        tar = tarfile.open(backup, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise BackupError("BACKUP_INVALID", f"not a readable tar.gz ({exc})") from None
    with tar:
        try:
            mf = tar.extractfile("manifest.json")
        except KeyError:
            raise BackupError("BACKUP_INVALID", "manifest.json missing") from None
        if mf is None:
            raise BackupError("BACKUP_INVALID", "manifest.json is not a file")
        try:
            manifest = json.loads(mf.read().decode("utf-8"))
        except ValueError as exc:
            raise BackupError("BACKUP_INVALID", f"manifest.json unreadable ({exc})") from None
        if manifest.get("schema") != BACKUP_SCHEMA:
            raise BackupError("BACKUP_INVALID", f"unsupported backup schema {manifest.get('schema')!r}")
        _check_manifest_layout(manifest)
        declared = {e["name"]: e for e in manifest.get("entries", [])}
        seen: set[str] = set()
        for member in tar.getmembers():
            if member.name == "manifest.json":
                continue
            if not member.isfile() or not _safe_member_name(member.name):
                raise BackupError("BACKUP_INVALID", f"unexpected member {member.name!r}")
            entry = declared.get(member.name)
            if entry is None:
                raise BackupError("BACKUP_INVALID", f"member {member.name!r} is not in the manifest")
            fh = tar.extractfile(member)
            assert fh is not None
            dest = staging / "members" / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with open(dest, "wb") as out:
                for chunk in iter(lambda: fh.read(_CHUNK), b""):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != entry["sha256"]:
                raise BackupError("HASH_MISMATCH", f"{member.name}: sha256 differs from the manifest")
            seen.add(member.name)
        absent = sorted(set(declared) - seen)
        if absent:
            raise BackupError("BACKUP_INVALID", f"manifest entries missing from the archive: {absent}")
    for entry in manifest["entries"]:
        path = staging / "members" / entry["name"]
        if entry["kind"] == "sqlite":
            result = _integrity(path)
            if result != "ok":
                raise BackupError("DB_INTEGRITY", f"{entry['name']}: integrity_check returned {result!r}")
        elif entry["kind"] == "tree":
            tree_dir = staging / "trees" / entry["name"]
            tree_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "r:gz") as t:
                names = [m.name for m in t.getmembers()]
                for m in t.getmembers():
                    if not m.isfile() or not _safe_member_name(m.name):
                        raise BackupError("BACKUP_INVALID", f"{entry['name']}: unexpected member {m.name!r}")
                t.extractall(tree_dir, filter="data")
            index = {f["path"]: f for f in entry["files"]}
            if sorted(names) != sorted(index):
                raise BackupError("BACKUP_INVALID", f"{entry['name']}: file list differs from the manifest index")
            for rel, f in index.items():
                if _sha256_file(tree_dir / rel) != f["sha256"]:
                    raise BackupError("HASH_MISMATCH", f"{entry['name']}:{rel}: sha256 differs from the index")
        else:
            raise BackupError("BACKUP_INVALID", f"unknown entry kind {entry['kind']!r}")
    return manifest


def verify_backup(backup: Path, *, expect_sha256: Optional[str] = None,
                  staging_dir: Optional[Path] = None) -> dict[str, Any]:
    """Verify every hash and DB integrity of ``backup`` without restoring."""
    _check_expected_sha(Path(backup), expect_sha256)
    with tempfile.TemporaryDirectory(prefix=".sherman-verify-", dir=staging_dir) as tmp:
        manifest = _stage_and_verify(Path(backup), Path(tmp))
    return {"status": "verified", "backup": str(backup), "created_at": manifest.get("created_at"),
            "entries": len(manifest["entries"]), "sha256_checked": expect_sha256 is not None}


def _empty_or_absent(path: Path) -> bool:
    if path.is_symlink():
        return False
    if not path.exists():
        return True
    return path.is_dir() and not any(path.iterdir())


def _ensure_target_dir(path: Path, what: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupError(
            "TARGET_INVALID",
            f"cannot create {what} {path} ({exc.strerror}); pre-create it empty and owned by the restoring user",
        ) from None
    if not os.access(path, os.W_OK | os.X_OK):
        raise BackupError("TARGET_INVALID", f"{what} {path} is not writable by the restoring user")


def restore_backup(backup: Path, data_root: Path, *, app_root: Optional[Path] = None,
                   staging_dir: Optional[Path] = None, expect_sha256: Optional[str] = None) -> dict[str, Any]:
    """Restore ``backup`` into an empty data root (and an app copy without data/).

    ``staging_dir`` (default: the system temp directory) holds the private 0700
    extraction directory; it must have room for the uncompressed backup."""
    data_root = Path(data_root)
    if not data_root.is_absolute():
        raise BackupError("TARGET_INVALID", "--data-root must be absolute")
    reason = _custody_reason(data_root)
    if reason:
        raise BackupError("TARGET_INVALID", f"--data-root {reason}")
    if not _empty_or_absent(data_root):
        raise BackupError("TARGET_NOT_EMPTY", "--data-root must be absent or an empty directory")
    app_data = Path(app_root) / "data" if app_root is not None else None
    if app_data is not None and not _empty_or_absent(app_data):
        raise BackupError("TARGET_NOT_EMPTY", "--app-root data/ must be absent or empty (restore before first start)")
    if staging_dir is not None and not Path(staging_dir).is_dir():
        raise BackupError("VALUE_INVALID", f"--staging-dir {staging_dir} is not a directory")
    _check_expected_sha(Path(backup), expect_sha256)
    with tempfile.TemporaryDirectory(prefix=".sherman-restore-", dir=staging_dir) as tmp_s:
        tmp = Path(tmp_s)
        os.chmod(tmp, 0o700)
        manifest = _stage_and_verify(Path(backup), tmp)
        app_entries = [e for e in manifest["entries"] if e["name"].startswith("app/")]
        if app_entries and app_data is None:
            raise BackupError("TARGET_INVALID", "the backup holds app DBs; pass --app-root")
        from apps.production_line.settings import LinePaths, ensure_layout

        _ensure_target_dir(data_root, "--data-root")
        if app_entries and app_data is not None:
            _ensure_target_dir(app_data, "--app-root data/")

        ensure_layout(LinePaths.for_root(data_root))
        restored: list[str] = []
        for entry in manifest["entries"]:
            src = tmp / "members" / entry["name"]
            if entry["kind"] == "sqlite":
                if entry["name"].startswith("db/"):
                    dest = data_root / entry["name"]
                else:
                    assert app_data is not None
                    dest = app_data / PurePosixPath(entry["name"]).name
                shutil.move(str(src), str(dest))
                os.chmod(dest, 0o600)
                _set_journal_mode(dest, entry.get("journal_mode_source", "delete"))
                if _integrity(dest) != "ok":
                    raise BackupError("DB_INTEGRITY", f"{entry['name']}: integrity_check failed after restore")
                restored.append(entry["name"])
            else:
                tree_dir = tmp / "trees" / entry["name"]
                for f in entry["files"]:
                    dest = data_root / f["path"]
                    dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    shutil.move(str(tree_dir / f["path"]), str(dest))
                    os.chmod(dest, 0o600)
                restored.append(entry["name"])
        for d in LinePaths.for_root(data_root).layout_dirs():
            os.chmod(d, 0o700)
    return {"status": "restored", "data_root": str(data_root), "app_root": None if app_root is None else str(app_root),
            "restored": restored, "backup_created_at": manifest.get("created_at")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _abs(p: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(p)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.ops.backup", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create", help="write one backup file")
    c.add_argument("--data-root", required=True)
    c.add_argument("--app-root")
    c.add_argument("--out", required=True)
    c.add_argument("--include-secrets", action="store_true")
    c.add_argument("--include-job-results", action="store_true")
    c.add_argument("--keep", type=int, default=None, help="then keep only the newest N backups in --out")
    v = sub.add_parser("verify", help="verify a backup's hashes and DB integrity")
    v.add_argument("--backup", required=True)
    v.add_argument("--expect-sha256", help="the archive sha256 recorded off the machine at create time")
    v.add_argument("--staging-dir", help="private extraction parent (default: the system temp dir)")
    r = sub.add_parser("restore", help="restore into an empty data root")
    r.add_argument("--backup", required=True)
    r.add_argument("--data-root", required=True)
    r.add_argument("--app-root")
    r.add_argument("--expect-sha256", help="the archive sha256 recorded off the machine at create time")
    r.add_argument("--staging-dir", help="private extraction parent (default: the system temp dir)")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            doc = create_backup(_abs(args.data_root), _abs(args.out),
                                app_root=_abs(args.app_root) if args.app_root else None,
                                include_secrets=args.include_secrets, include_job_results=args.include_job_results)
            if args.keep is not None:
                doc["pruned"] = prune_backups(_abs(args.out), args.keep)
        elif args.command == "verify":
            doc = verify_backup(_abs(args.backup), expect_sha256=args.expect_sha256,
                                staging_dir=_abs(args.staging_dir) if args.staging_dir else None)
        else:
            doc = restore_backup(_abs(args.backup), _abs(args.data_root),
                                 app_root=_abs(args.app_root) if args.app_root else None,
                                 staging_dir=_abs(args.staging_dir) if args.staging_dir else None,
                                 expect_sha256=args.expect_sha256)
    except BackupError as exc:
        sys.stdout.write(json.dumps({"status": "refused", "code": exc.code, "message": exc.message}) + "\n")
        return 2
    except Exception as exc:  # pragma: no cover - reported, never a traceback-only exit
        sys.stdout.write(json.dumps({"status": "error", "code": "INTERNAL_ERROR",
                                     "message": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        return 3
    sys.stdout.write(json.dumps(doc, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
