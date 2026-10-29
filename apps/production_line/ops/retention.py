"""Retention: prune old scans, heavy job artifacts and old line events (dry run first).

``python -m apps.production_line.ops.retention --data-root R --days N [--log-days M] [--apply]``

Without ``--apply`` nothing is deleted. The tool prints the plan (counts and
bytes per category) as one JSON document. The age of a job is its
``finished_at`` in the line DB, never a file mtime, and only **terminal** jobs
(completed, failed, cancelled) finished more than N days ago are touched.

Pruned, per old terminal job workspace ``jobs/<job_id>/``:

- the staged scan ``input/scan.ply|.stl`` and the compiled-part copy's
  ``compiled/stages/**`` (sample clouds and stage meshes);
- the E3 heavy outputs: ``engine/registered_scan.ply``,
  ``engine/deviations.f32``, ``engine/segment_labels.u16`` and
  ``engine/display/**``, plus the legacy names ``engine/aligned_scan.ply``,
  ``engine/deviations.npy`` and ``engine/*overlay*.png``;
- the viewer assets and renders: ``result/viewer_scan.ply``,
  ``result/viewer_deviations.f32``, ``result/stage_mesh.ply``,
  ``result/final_mesh.ply`` and ``result/overview.png``.

Always kept:

- ``job.json``, ``recipe.json`` and ``input/context.json``;
- every ``*.json`` (including ``result/*.json``, ``engine/result.json`` and
  ``bend_geometry.json``), every ``*.log`` and ``result/report.pdf``.

:data:`KEEP_SUFFIXES` is checked again before every unlink. Once the viewer
assets are gone, the station shows an old job without its 3D view, while the
PDF and the result stay available.

Archived scans ``archive/scans/YYYY/MM/<sha>.<ext>`` are pruned only when all of
these hold:

- the scan row is ``enqueued``, ``duplicate`` or ``dismissed``;
- it was received more than N days ago;
- every job that references the scan is terminal and finished more than N days
  ago.

Quarantined scans (still assignable) and scans in ``received`` are kept. A
pruned scan can no longer be retried; a retry then fails closed with
SCAN_MISSING.

Events older than the log retention (``--log-days``, default
``SHERMAN_LINE_LOG_RETENTION_DAYS``, else 30) are deleted from
``db/line.sqlite3``. This is one ``DELETE FROM events WHERE ts < ?`` in a
``BEGIN IMMEDIATE`` transaction. It only runs when the store's ``schema_meta``
says ``line_store`` v1, the schema this tool was written against; any other
version skips event pruning with EVENTS_SCHEMA_UNKNOWN. Event ids are
AUTOINCREMENT, so ids are never reused and SSE cursors stay monotonic.

Safety:

- Symlinks are never followed or deleted (they are reported).
- Every path is resolved and must stay inside the data root.
- The data root must pass the line's custody check (outside the repo).
- Only regular files are removed. Directories are removed only when they become
  empty and are one of the pruned subtrees.

Exit codes: 0 ok (dry run or applied), 2 refused or invalid input, 3 some
deletions failed (listed in ``errors``).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "RETENTION_TOOL_VERSION",
    "HEAVY_WORKSPACE_PATTERNS",
    "HEAVY_WORKSPACE_TREES",
    "KEEP_SUFFIXES",
    "KEEP_NAMES",
    "RetentionError",
    "RetentionPlan",
    "plan_retention",
    "apply_plan",
    "main",
]

RETENTION_TOOL_VERSION = "c12-retention-1.0.0"
TERMINAL = ("completed", "failed", "cancelled")
PRUNABLE_SCAN_STATES = ("enqueued", "duplicate", "dismissed")
#: Workspace-relative glob patterns (one path segment each) of heavy files.
HEAVY_WORKSPACE_PATTERNS: tuple[str, ...] = (
    "input/scan.ply",
    "input/scan.stl",
    "engine/registered_scan.ply",
    "engine/deviations.f32",
    "engine/segment_labels.u16",
    "engine/aligned_scan.ply",
    "engine/deviations.npy",
    "engine/*overlay*.png",
    "result/viewer_scan.ply",
    "result/viewer_deviations.f32",
    "result/stage_mesh.ply",
    "result/final_mesh.ply",
    "result/overview.png",
)
#: Workspace-relative subtrees whose regular files are all heavy.
HEAVY_WORKSPACE_TREES: tuple[str, ...] = ("engine/display", "compiled/stages")
KEEP_SUFFIXES: tuple[str, ...] = (".json", ".log", ".pdf")
KEEP_NAMES: tuple[str, ...] = ("job.json", "recipe.json", "context.json", "report.pdf")
_SCHEMA_COMPONENT = "line_store"
_KNOWN_SCHEMA_VERSIONS = (1,)


class RetentionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _iso(dt: datetime) -> str:
    """The store's fixed-width timestamp format (string order == time order)."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _real(p: Path) -> Path:
    return Path(os.path.realpath(os.fspath(p)))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_kept(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name in KEEP_NAMES or name.lower().endswith(KEEP_SUFFIXES)


@dataclass
class _Item:
    category: str
    path: Path
    size: int
    detail: str = ""


@dataclass
class RetentionPlan:
    data_root: Path
    days: int
    log_days: int
    cutoff: str
    log_cutoff: str
    items: list[_Item] = field(default_factory=list)
    empty_dirs: list[Path] = field(default_factory=list)
    events_to_delete: int = 0
    events_status: str = "OK"
    kept: dict[str, int] = field(default_factory=dict)
    skipped_symlinks: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    jobs_considered: int = 0

    def keep(self, reason: str) -> None:
        self.kept[reason] = self.kept.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        cats: dict[str, dict[str, int]] = {}
        for it in self.items:
            c = cats.setdefault(it.category, {"files": 0, "bytes": 0})
            c["files"] += 1
            c["bytes"] += it.size
        return {
            "tool_version": RETENTION_TOOL_VERSION,
            "data_root": str(self.data_root),
            "days": self.days,
            "log_days": self.log_days,
            "cutoff": self.cutoff,
            "log_cutoff": self.log_cutoff,
            "jobs_considered": self.jobs_considered,
            "prune": dict(sorted(cats.items())),
            "prune_total": {"files": len(self.items), "bytes": sum(i.size for i in self.items)},
            "events": {"delete": self.events_to_delete, "status": self.events_status},
            "kept": dict(sorted(self.kept.items())),
            "skipped_symlinks": sorted(self.skipped_symlinks)[:200],
            "orphan_workspaces": sorted(self.orphans)[:200],
        }


def _connect_ro(db: Path) -> sqlite3.Connection:
    from apps.production_line.health import sqlite_uri

    try:
        conn = sqlite3.connect(sqlite_uri(db, "ro"), uri=True, timeout=10)
        conn.execute("SELECT 1 FROM jobs LIMIT 1")
    except sqlite3.Error:
        conn = sqlite3.connect(sqlite_uri(db, "rw"), uri=True, timeout=10)  # WAL without a -shm yet
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _schema_version(conn: sqlite3.Connection) -> Optional[int]:
    try:
        row = conn.execute("SELECT version FROM schema_meta WHERE component=?", (_SCHEMA_COMPONENT,)).fetchone()
    except sqlite3.Error:
        return None
    return int(row["version"]) if row else None


def _add_file(plan: RetentionPlan, root_real: Path, path: Path, category: str, rel: str) -> None:
    if path.is_symlink():
        plan.skipped_symlinks.append(str(path.relative_to(plan.data_root)))
        return
    if not path.is_file():
        return
    if not _inside(_real(path), root_real):
        plan.skipped_symlinks.append(str(path.relative_to(plan.data_root)))
        return
    if _is_kept(rel):  # pragma: no cover - the patterns never name kept files; defence in depth
        plan.keep("keep_rule")
        return
    plan.items.append(_Item(category, path, path.stat().st_size, rel))


def _workspace_items(plan: RetentionPlan, root_real: Path, ws: Path) -> None:
    for pattern in HEAVY_WORKSPACE_PATTERNS:
        parent_rel, _, name_pat = pattern.rpartition("/")
        parent = ws / parent_rel
        if parent.is_symlink() or not parent.is_dir():
            continue
        for p in sorted(parent.iterdir()):
            if fnmatch.fnmatchcase(p.name, name_pat):
                _add_file(plan, root_real, p, "workspace_heavy", f"{parent_rel}/{p.name}")
    for tree in HEAVY_WORKSPACE_TREES:
        top = ws / tree
        if top.is_symlink() or not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top, topdown=False, followlinks=False):
            d = Path(dirpath)
            for name in dirnames:
                if (d / name).is_symlink():
                    plan.skipped_symlinks.append(str((d / name).relative_to(plan.data_root)))
            for name in sorted(filenames):
                p = d / name
                _add_file(plan, root_real, p, "workspace_heavy", str(p.relative_to(ws)))
            plan.empty_dirs.append(d)


def plan_retention(data_root: Path, days: int, *, log_days: int, now: Optional[datetime] = None) -> RetentionPlan:
    """Compute what would be pruned (module docstring); touches nothing."""
    from apps.production_line.settings import forbidden_location_reason

    data_root = Path(data_root)
    if not data_root.is_absolute():
        raise RetentionError("DATA_ROOT_INVALID", "--data-root must be absolute")
    if days < 1 or log_days < 1:
        raise RetentionError("VALUE_INVALID", "--days and --log-days must be >= 1")
    reason = forbidden_location_reason(data_root, Path(__file__).resolve().parents[3])
    if reason:
        raise RetentionError("DATA_ROOT_INVALID", f"--data-root {reason}")
    db = data_root / "db" / "line.sqlite3"
    if not db.is_file():
        raise RetentionError("NOT_A_DATA_ROOT", "db/line.sqlite3 not found under --data-root")
    now = now or datetime.now(timezone.utc)
    cutoff = _iso(now - timedelta(days=days))
    log_cutoff = _iso(now - timedelta(days=log_days))
    plan = RetentionPlan(data_root=data_root, days=days, log_days=log_days, cutoff=cutoff, log_cutoff=log_cutoff)
    root_real = _real(data_root)
    conn = _connect_ro(db)
    try:
        version = _schema_version(conn)
        jobs = conn.execute(
            "SELECT job_id, execution_status, finished_at, scan_file_id, scan_sha256, workspace_rel FROM jobs"
        ).fetchall()
        scans = conn.execute(
            "SELECT scan_file_id, sha256, stored_rel_path, ingest_state, received_at FROM scan_files"
        ).fetchall()
        if version in _KNOWN_SCHEMA_VERSIONS:
            plan.events_to_delete = int(conn.execute("SELECT COUNT(*) FROM events WHERE ts < ?", (log_cutoff,)).fetchone()[0])
        else:
            plan.events_status = "EVENTS_SCHEMA_UNKNOWN"
    finally:
        conn.close()

    known_ws: set[str] = set()
    jobs_by_scan: dict[str, list[sqlite3.Row]] = {}
    for j in jobs:
        known_ws.add(j["job_id"])
        if j["scan_sha256"]:
            jobs_by_scan.setdefault(j["scan_sha256"], []).append(j)
        old_terminal = j["execution_status"] in TERMINAL and j["finished_at"] is not None and j["finished_at"] < cutoff
        if not old_terminal:
            plan.keep("job_recent_or_active")
            continue
        plan.jobs_considered += 1
        ws = data_root / "jobs" / j["job_id"]
        if ws.is_symlink():
            plan.skipped_symlinks.append(f"jobs/{j['job_id']}")
            continue
        if ws.is_dir() and _inside(_real(ws), root_real):
            _workspace_items(plan, root_real, ws)

    jobs_dir = data_root / "jobs"
    if jobs_dir.is_dir():
        for ws in jobs_dir.iterdir():
            if ws.name not in known_ws and not ws.name.startswith("."):
                plan.orphans.append(ws.name)

    for s in scans:
        rel = s["stored_rel_path"] or ""
        if not rel.startswith("archive/scans/"):
            plan.keep("scan_not_archived")
            continue
        if s["ingest_state"] not in PRUNABLE_SCAN_STATES:
            plan.keep(f"scan_{s['ingest_state']}")
            continue
        if not s["received_at"] or s["received_at"] >= cutoff:
            plan.keep("scan_recent")
            continue
        refs = jobs_by_scan.get(s["sha256"], [])
        if any(r["execution_status"] not in TERMINAL or not r["finished_at"] or r["finished_at"] >= cutoff for r in refs):
            plan.keep("scan_job_recent_or_active")
            continue
        _add_file(plan, root_real, data_root / rel, "archived_scan", rel)
    return plan


def apply_plan(plan: RetentionPlan) -> dict[str, Any]:
    """Delete what ``plan`` lists; returns ``{deleted_files, deleted_bytes, events_deleted, errors}``."""
    root_real = _real(plan.data_root)
    deleted = 0
    freed = 0
    errors: list[str] = []
    for it in plan.items:
        try:
            st = os.lstat(it.path)
            if not (st.st_mode & 0o170000 == 0o100000):  # regular file only, never a link
                errors.append(f"{it.detail}: no longer a regular file")
                continue
            if not _inside(_real(it.path), root_real) or _is_kept(it.detail):
                errors.append(f"{it.detail}: refused by the safety check")
                continue
            os.unlink(it.path)
            deleted += 1
            freed += st.st_size
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"{it.detail}: {exc.strerror}")
    for d in plan.empty_dirs:
        try:
            if d.is_dir() and not d.is_symlink() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    events_deleted = 0
    if plan.events_status == "OK" and plan.events_to_delete:
        from apps.production_line.health import sqlite_uri

        conn = sqlite3.connect(sqlite_uri(plan.data_root / "db" / "line.sqlite3", "rw"), uri=True, timeout=10,
                               isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("BEGIN IMMEDIATE")
            try:
                if _schema_version_raw(conn) not in _KNOWN_SCHEMA_VERSIONS:
                    raise RetentionError("EVENTS_SCHEMA_UNKNOWN", "line_store schema changed during the run")
                cur = conn.execute("DELETE FROM events WHERE ts < ?", (plan.log_cutoff,))
                events_deleted = int(cur.rowcount or 0)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()
    return {"deleted_files": deleted, "deleted_bytes": freed, "events_deleted": events_deleted, "errors": errors}


def _schema_version_raw(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute("SELECT version FROM schema_meta WHERE component=?", (_SCHEMA_COMPONENT,)).fetchone()
    return int(row[0]) if row else None


def _default_log_days() -> int:
    raw = os.environ.get("SHERMAN_LINE_LOG_RETENTION_DAYS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) >= 1 else 30


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.ops.retention",
                                     description="Prune old scans, heavy job artifacts and old events (dry run by default).")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--days", type=int, required=True, help="prune terminal jobs/scans older than this")
    parser.add_argument("--log-days", type=int, default=None, help="event retention (default SHERMAN_LINE_LOG_RETENTION_DAYS or 30)")
    parser.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = parser.parse_args(argv)
    log_days = args.log_days if args.log_days is not None else _default_log_days()
    try:
        plan = plan_retention(Path(os.path.abspath(os.path.expanduser(args.data_root))), args.days, log_days=log_days)
        doc = {"mode": "apply" if args.apply else "dry_run", **plan.summary()}
        if args.apply:
            doc["applied"] = apply_plan(plan)
    except RetentionError as exc:
        sys.stdout.write(json.dumps({"status": "refused", "code": exc.code, "message": exc.message}) + "\n")
        return 2
    except Exception as exc:  # pragma: no cover
        sys.stdout.write(json.dumps({"status": "error", "code": "INTERNAL_ERROR",
                                     "message": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        return 3
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 3 if args.apply and doc["applied"]["errors"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
