"""SMB-safe polling hotfolder: accept a ZEISS scan only once it is fully written.

No filesystem events are used (SMB/CIFS mounts deliver them unreliably); the
inbox is polled. A file is *ready* when all of these hold:

- its ``(size, mtime_ns)`` signature is unchanged for ``stable_polls``
  consecutive polls,
- it is at least ``min_age_s`` old and its size at least ``min_size_bytes``.
  The age is the larger of (server clock - mtime) and the time this poller has
  seen the current signature unchanged, so a ZEISS PC whose clock runs ahead
  of the server (mtime in the future on the SMB share) cannot stall ingest; a
  future mtime beyond ``_CLOCK_SKEW_WARN_S`` is logged once per file,
- with ``sidecar_mode='required'``, ``<file>.done`` exists,
- a read-open succeeds.

Temporary names are never ingested: dotfiles, ``*.tmp``, ``*.part``,
``*.partial``, ``*~``, ``*.crdownload``, ``*.done`` and our own
``*.sherman-partial``. A writer that streams to a temp name and renames it
when done is therefore picked up after the rename. Symlinks are never
followed.

Ingest of a ready file:

1. ``ply_probe.probe`` (format, completeness, counts, bbox) and an optional
   ``.done`` sidecar (empty, or JSON ``{sha256?, size_bytes?, declared_stage?}``;
   anything else, or a sha/size that disagrees, is SIDECAR_INVALID).
   A content problem (truncated, invalid, unsupported, bad sidecar) is not
   quarantined until the file's signature has been unchanged for
   ``truncated_grace_s`` (60 s): a writer that pauses (an SMB hiccup) must
   not have its partial file consumed. Until then the file stays in the
   inbox, untouched and not re-probed.
2. Streaming sha256 in 1 MiB chunks; the signature is re-checked afterwards,
   and a file that changed while hashing is left for the next polls.
3. ``resolver.resolve(original name)`` for the operator-facing parse/match.
4. ``store.register_scan`` (state ``received``, or ``quarantined`` for a
   content problem) **before** the bytes are moved, so a crash can never lose
   a file: a restart finds the row and completes the ingest. A drop whose
   bytes are already registered is a *resume* (no SCAN_DUPLICATE) when it is
   the same station, name and size and either its stored copy is missing or
   the file was last modified before the row was registered; otherwise it is
   a genuine duplicate. The same bytes under *another* name (or station) are
   always a duplicate, even while the first row is still ``received``: the new
   name's resolution is never applied to that row, which is later resolved
   from its own name. Either way a quarantined scan's ``reason.json`` is
   rewritten when missing.
5. The bytes go to ``archive/scans/YYYY/MM/<sha256><ext>`` (valid) or
   ``quarantine/<YYYY-MM-DD>/<sha256><ext>`` plus ``<sha256>.reason.json``
   (content problem). Same device: ``os.replace``. Cross device or
   ``copy_leave``: copy to ``<dest>.sherman-partial``, fsync, re-hash,
   ``os.replace``; then unlink the source (``move``) or remember
   ``(path, size, mtime_ns, sha)`` in the store (``copy_leave``).
6. MATCHED -> ``store.match_and_enqueue`` (inspect job, priority 100,
   2 attempts, dedupe ``inspect:<scan_sha>:<recipe_sha>``). Anything else ->
   ``store.quarantine_scan`` with UNMATCHED_SKU, AMBIGUOUS_SKU,
   UNPARSEABLE_FILENAME, REVISION_MISMATCH or PART_NOT_READY plus
   ``quarantine/<date>/<sha256>.reason.json``; the scan stays archived so an
   admin can assign it.

Identical bytes dropped again give SCAN_DUPLICATE and no job. A per-file
exception is logged and never stops the loop; after 3 failed attempts the file
is left alone for ``failed_retry_s`` (60 s) and then retried, so a transient
archive problem (disk full, permission denied) heals by itself once fixed and
the file is never lost (it stays in the inbox until it is placed).

A file that stays below ``min_size_bytes`` (a zero-byte or truncated-to-nothing
export) is quarantined as EMPTY_SCAN once its signature has been unchanged for
``undersize_grace_s`` (120 s), so the operator sees it instead of it waiting in
the inbox forever. A ``.done`` sidecar whose scan is gone (already ingested, or
never written) is removed after ``truncated_grace_s``. Files with another
extension are never touched; they are logged once and counted in the poller
heartbeat (``ignored_files``). Scans left in ``received`` (the
resolver failed, or a crash) are re-resolved on later polls; a scan whose
resolution cannot be applied (``match_and_enqueue`` keeps raising) is
quarantined after 3 failures as PART_NOT_READY with detail
RESOLVE_APPLY_FAILED, so the operator sees it and can assign it.

Only stdlib imports; the resolver is injected (C11 wires
``sku_resolver.FilenameRecipeResolver``).
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.production_line.ingest.ply_probe import ProbeResult, probe
from apps.production_line.ingest.ports import ResolveOutcome, ScanRecipeResolver, minimal_parse
from apps.production_line.store.line_store import LineStore, iso_utc, parse_iso_utc, utc_now
from apps.production_line.store.vocabulary import (
    MAX_ATTEMPTS_BY_KIND,
    PRIORITY_BY_KIND,
    RESOLVE_STATUS_TO_QUARANTINE_REASON,
)

__all__ = [
    "HotfolderConfig",
    "HotfolderConfigError",
    "HotfolderPoller",
    "IngestOutcome",
    "IGNORED_SUFFIXES",
    "PARTIAL_SUFFIX",
    "is_ignored_name",
    "sha256_file",
]

PARTIAL_SUFFIX = ".sherman-partial"
IGNORED_SUFFIXES: tuple[str, ...] = (".tmp", ".part", ".partial", "~", ".crdownload", ".done", PARTIAL_SUFFIX)
_FORBIDDEN_REPO_DIRS: tuple[str, ...] = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")
_STATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SIDECAR_MAX_BYTES = 64 * 1024
_SIDECAR_KEYS = frozenset({"sha256", "size_bytes", "declared_stage"})
_MAX_INGEST_FAILURES = 3
_HASH_CHUNK = 1024 * 1024
_CLOCK_SKEW_WARN_S = 5.0
_CONTENT_GRACE_DETAIL = "CONTENT_GRACE: waiting for truncated_grace_s before quarantining"


class HotfolderConfigError(ValueError):
    pass


def is_ignored_name(name: str) -> bool:
    """True for names a writer uses while a file is incomplete (and our own temp files)."""
    lower = name.lower()
    return name.startswith(".") or any(lower.endswith(s) for s in IGNORED_SUFFIXES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class HotfolderConfig:
    """One station's inbox. ``data_root`` is the line data root
    (``SHERMAN_LINE_DATA_ROOT``); archive and quarantine live under it."""

    station_id: str
    inbox: Path
    data_root: Path
    poll_interval_s: float = 1.0
    stable_polls: int = 3
    min_age_s: float = 2.0
    min_size_bytes: int = 1024
    extensions: tuple[str, ...] = (".ply", ".stl")
    sidecar_mode: str = "off"  # 'off' | 'required'
    source_policy: str = "move"  # 'move' | 'copy_leave'
    recursive: bool = False
    unstable_timeout_s: float = 1800.0
    truncated_grace_s: float = 60.0
    undersize_grace_s: float = 120.0
    failed_retry_s: float = 60.0
    repo_root: Path | None = None

    @property
    def archive_root(self) -> Path:
        return self.data_root / "archive"

    @property
    def quarantine_root(self) -> Path:
        return self.data_root / "quarantine"

    def validate(self) -> "HotfolderConfig":
        """Raise ``HotfolderConfigError`` for an unsafe or inconsistent config."""
        if not _STATION_RE.match(self.station_id or ""):
            raise HotfolderConfigError("station_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
        for name in ("inbox", "data_root"):
            value = getattr(self, name)
            if not Path(value).is_absolute():
                raise HotfolderConfigError(f"{name} must be an absolute path")
        if self.sidecar_mode not in ("off", "required"):
            raise HotfolderConfigError("sidecar_mode must be 'off' or 'required'")
        if self.source_policy not in ("move", "copy_leave"):
            raise HotfolderConfigError("source_policy must be 'move' or 'copy_leave'")
        if self.stable_polls < 1 or self.poll_interval_s <= 0 or self.min_age_s < 0 or self.min_size_bytes < 0:
            raise HotfolderConfigError("stable_polls >= 1, poll_interval_s > 0, min_age_s >= 0, min_size_bytes >= 0")
        if self.truncated_grace_s < 0 or self.undersize_grace_s < 0 or self.failed_retry_s < 0:
            raise HotfolderConfigError("truncated_grace_s, undersize_grace_s and failed_retry_s must be >= 0")
        exts = tuple(e.lower() for e in self.extensions)
        if not exts or any(e not in (".ply", ".stl") for e in exts):
            raise HotfolderConfigError("extensions must be a non-empty subset of ('.ply', '.stl')")
        inbox = Path(self.inbox).resolve()
        root = Path(self.data_root).resolve()
        if not inbox.is_dir():
            raise HotfolderConfigError(f"inbox {inbox} is not a directory")
        for managed in (root / "archive", root / "quarantine", root / "jobs"):
            if _is_within(inbox, managed) or _is_within(managed, inbox):
                raise HotfolderConfigError(f"inbox and {managed} must not contain each other")
        if _is_within(root, inbox):
            raise HotfolderConfigError("the data root must not be inside the inbox")
        repo = (self.repo_root or _default_repo_root()).resolve()
        for path, label in ((inbox, "inbox"), (root, "data_root")):
            for forbidden in _FORBIDDEN_REPO_DIRS:
                if _is_within(path, repo / forbidden):
                    raise HotfolderConfigError(f"{label} must not be under the repo's {forbidden}/ directory")
        return self


@dataclass(frozen=True)
class IngestOutcome:
    """What happened to one inbox file (or one pending scan) during a poll.

    ``action``: enqueued | quarantined | duplicate | pending | error.
    """

    action: str
    original_name: str
    source_path: str | None = None
    scan_file_id: str | None = None
    job_id: str | None = None
    reason: str | None = None
    detail: str | None = None


@dataclass
class _Watch:
    """Poll state of one inbox path; recreated whenever its signature changes,
    so ``first_seen`` is when the current signature was first seen."""

    size: int
    mtime_ns: int
    stable: int = 0
    first_seen: float = 0.0
    content_wait_until: float = 0.0
    failures: int = 0
    gave_up: bool = False
    retry_at: float = 0.0
    warned_unstable: bool = False
    warned_skew: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class _SidecarError(Exception):
    pass


class HotfolderPoller:
    """Polls one station inbox (module docstring). Not thread-safe; run one per station."""

    def __init__(
        self,
        config: HotfolderConfig,
        store: LineStore,
        resolver: ScanRecipeResolver,
        clock: Callable[[], datetime] = utc_now,
        logger: logging.Logger | None = None,
        *,
        instance_id: str | None = None,
    ) -> None:
        self.config = config.validate()
        self.store = store
        self.resolver = resolver
        self._clock = clock
        self._log = logger or logging.getLogger(__name__)
        self._inbox = Path(config.inbox).resolve()
        self._root = Path(config.data_root).resolve()
        self._exts = tuple(e.lower() for e in config.extensions)
        self._watch: dict[str, _Watch] = {}
        self._instance_id = instance_id or f"hotfolder-{uuid.uuid4().hex[:12]}"
        self._last_pending_check = 0.0
        self._apply_failures: dict[str, int] = {}
        self._orphan_sidecars: dict[str, tuple[int, int, float]] = {}
        self._ignored_logged: set[str] = set()
        self.ignored_files = 0

    # ------------------------------------------------------------------ lifecycle
    def startup_rescan(self) -> list[IngestOutcome]:
        """Remove leftover ``*.sherman-partial`` files, forget poll state, resume pending scans."""
        for base in (self._root / "archive", self._root / "quarantine"):
            if not base.is_dir():
                continue
            for dirpath, _dirs, files in os.walk(base, followlinks=False):
                for name in files:
                    if name.endswith(PARTIAL_SUFFIX):
                        leftover = Path(dirpath) / name
                        try:
                            leftover.unlink()
                            self._log.warning("removed leftover partial copy %s", leftover)
                        except OSError as exc:
                            self._log.error("cannot remove leftover partial copy %s: %s", leftover, exc)
        self._watch.clear()
        return self._resume_pending(force=True)

    def run_forever(self, stop_event: threading.Event) -> None:
        self.startup_rescan()
        while not stop_event.is_set():
            try:
                self.poll_once()
                self.store.put_heartbeat(
                    f"hotfolder:{self.config.station_id}",
                    self._instance_id,
                    {"inbox": str(self._inbox), "watching": len(self._watch), "ignored_files": self.ignored_files},
                )
            except Exception:  # pragma: no cover - defensive: the loop must survive
                self._log.exception("hotfolder poll failed for station %s", self.config.station_id)
            stop_event.wait(self.config.poll_interval_s)

    # ------------------------------------------------------------------ polling
    def _candidates(self) -> list[Path]:
        out: list[Path] = []
        sidecars: list[Path] = []
        ignored = 0
        stack = [self._inbox]
        while stack:
            directory = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError as exc:
                self._log.error("cannot list %s: %s", directory, exc)
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
                if is_ignored_name(entry.name):
                    if entry.name.lower().endswith(".done") and not entry.name.startswith("."):
                        sidecars.append(Path(entry.path))
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if self.config.recursive:
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if Path(entry.name).suffix.lower() not in self._exts:
                    ignored += 1
                    if entry.path not in self._ignored_logged and len(self._ignored_logged) < 1000:
                        self._ignored_logged.add(entry.path)
                        self._log.warning("%s is not a scan file (%s); left untouched", entry.path, ", ".join(self._exts))
                    continue
                out.append(Path(entry.path))
        self.ignored_files = ignored
        self._sweep_orphan_sidecars(sidecars)
        return sorted(out)

    def _sweep_orphan_sidecars(self, sidecars: list[Path]) -> None:
        """Remove a ``<scan>.done`` whose scan file is gone once it has been unchanged
        for ``truncated_grace_s`` (its scan was ingested before the marker arrived)."""
        now_mono = time.monotonic()
        seen: set[str] = set()
        for sidecar in sidecars:
            key = str(sidecar)
            if os.path.lexists(key[: -len(".done")]):
                continue
            try:
                st = os.lstat(sidecar)
            except OSError:
                continue
            seen.add(key)
            prev = self._orphan_sidecars.get(key)
            if prev is None or prev[:2] != (st.st_size, st.st_mtime_ns):
                self._orphan_sidecars[key] = (st.st_size, st.st_mtime_ns, now_mono)
                continue
            if now_mono - prev[2] < self.config.truncated_grace_s or self.config.source_policy != "move":
                continue
            try:
                sidecar.unlink()
                self._log.info("removed orphan sidecar %s (its scan is not in the inbox)", sidecar)
            except OSError as exc:
                self._log.error("cannot remove orphan sidecar %s: %s", sidecar, exc)
            self._orphan_sidecars.pop(key, None)
        for key in list(self._orphan_sidecars):
            if key not in seen:
                self._orphan_sidecars.pop(key, None)

    def poll_once(self) -> list[IngestOutcome]:
        outcomes = self._resume_pending()
        now_mono = time.monotonic()
        seen: set[str] = set()
        for path in self._candidates():
            key = str(path)
            seen.add(key)
            try:
                st = os.lstat(path)
            except OSError:
                continue  # vanished between listing and stat
            watch = self._watch.get(key)
            if watch is None or (watch.size, watch.mtime_ns) != (st.st_size, st.st_mtime_ns):
                self._watch[key] = _Watch(size=st.st_size, mtime_ns=st.st_mtime_ns, first_seen=now_mono)
                continue
            watch.stable += 1
            if watch.gave_up:
                if now_mono < watch.retry_at:
                    continue
                self._log.info("retrying %s after %.0f s", path, self.config.failed_retry_s)
                watch.gave_up = False
                watch.failures = 0
            if self.config.source_policy == "copy_leave" and self.store.has_ingest_source(
                self.config.station_id, key, st.st_size, st.st_mtime_ns
            ):
                continue
            if not self._ready(path, st, watch, now_mono):
                continue
            if now_mono < watch.content_wait_until:
                continue  # a content problem inside its grace period: leave the file alone
            try:
                outcome = self._ingest(
                    path, st, stable_for_s=now_mono - watch.first_seen, undersize=st.st_size < self.config.min_size_bytes
                )
            except Exception as exc:  # per-file failures never stop the loop
                watch.failures += 1
                watch.stable = 0
                self._log.exception("ingest of %s failed (attempt %d)", path, watch.failures)
                if watch.failures >= _MAX_INGEST_FAILURES:
                    watch.gave_up = True
                    watch.retry_at = now_mono + self.config.failed_retry_s
                    self._log.error(
                        "giving up on %s for %.0f s (or until it changes); it stays in the inbox",
                        path, self.config.failed_retry_s,
                    )  # fmt: skip
                outcomes.append(
                    IngestOutcome(action="error", original_name=path.name, source_path=key, detail=str(exc)[:500])
                )
                continue
            if outcome is None:
                watch.stable = 0  # changed while hashing; wait for it to settle again
                continue
            if outcome.action == "pending" and outcome.detail == _CONTENT_GRACE_DETAIL:
                # content problem inside truncated_grace_s: not registered, not moved
                watch.content_wait_until = watch.first_seen + self.config.truncated_grace_s
                self._log.info(
                    "%s looks incomplete (%s); waiting up to %.0f s for the writer before quarantining",
                    path, outcome.reason, self.config.truncated_grace_s,
                )  # fmt: skip
                continue
            outcomes.append(outcome)
            if self.config.source_policy == "move":
                self._watch.pop(key, None)
        for key in list(self._watch):
            if key not in seen:
                self._watch.pop(key, None)
        return outcomes

    def _ready(self, path: Path, st: os.stat_result, watch: _Watch, now_mono: float) -> bool:
        cfg = self.config
        if watch.stable < cfg.stable_polls:
            return False
        if st.st_size < cfg.min_size_bytes:
            if now_mono - watch.first_seen < cfg.undersize_grace_s:
                return False
            if not watch.warned_unstable:
                watch.warned_unstable = True
                self._log.warning(
                    "%s stayed below min_size_bytes (%d bytes) for %.0f s; quarantining it as EMPTY_SCAN",
                    path, st.st_size, cfg.undersize_grace_s,
                )  # fmt: skip
        wall_age = self._clock().timestamp() - st.st_mtime_ns / 1e9
        if wall_age < -_CLOCK_SKEW_WARN_S and not watch.warned_skew:
            watch.warned_skew = True
            self._log.warning(
                "%s has an mtime %.0f s in the future (writer clock ahead of this server?);"
                " using the observed stable time for min_age_s",
                path, -wall_age,
            )  # fmt: skip
        age = max(wall_age, now_mono - watch.first_seen)
        if age < cfg.min_age_s:
            return False
        if cfg.sidecar_mode == "required" and not os.path.isfile(str(path) + ".done"):
            return False
        try:
            with open(path, "rb") as fh:
                fh.read(1)
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------ ingest
    def _read_sidecar(self, path: Path) -> dict[str, Any] | None:
        sidecar = Path(str(path) + ".done")
        if not sidecar.is_file() or sidecar.is_symlink():
            return None
        if sidecar.stat().st_size > _SIDECAR_MAX_BYTES:
            raise _SidecarError("sidecar larger than 64 KiB")
        text = sidecar.read_bytes().decode("utf-8", errors="strict").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _SidecarError(f"sidecar is not JSON: {exc.msg}") from None
        if not isinstance(data, dict) or set(data) - _SIDECAR_KEYS:
            raise _SidecarError(f"sidecar must be an object with keys among {sorted(_SIDECAR_KEYS)}")
        if "sha256" in data and not (isinstance(data["sha256"], str) and re.fullmatch(r"[0-9a-fA-F]{64}", data["sha256"])):
            raise _SidecarError("sidecar sha256 must be 64 hex characters")
        if "size_bytes" in data and not (isinstance(data["size_bytes"], int) and data["size_bytes"] >= 0):
            raise _SidecarError("sidecar size_bytes must be a non-negative integer")
        if "declared_stage" in data and not (
            isinstance(data["declared_stage"], int) and not isinstance(data["declared_stage"], bool) and data["declared_stage"] >= 0
        ):
            raise _SidecarError("sidecar declared_stage must be a non-negative integer")
        return data

    def _resolve(self, original_name: str) -> tuple[ResolveOutcome | None, str | None]:
        try:
            outcome = self.resolver.resolve(original_name)
        except Exception as exc:
            self._log.exception("resolver failed for %r", original_name)
            return None, f"RESOLVER_ERROR: {exc}"[:200]
        if not isinstance(outcome, ResolveOutcome):
            return None, "RESOLVER_ERROR: resolver returned a non-ResolveOutcome"
        return outcome, None

    def _ingest(
        self, path: Path, st: os.stat_result, *, stable_for_s: float = float("inf"), undersize: bool = False
    ) -> IngestOutcome | None:
        cfg = self.config
        original_name = path.name
        content_reason: str | None = None
        content_detail: dict[str, Any] = {}
        if undersize:
            # _ready lets an undersized file through only after undersize_grace_s unchanged.
            content_reason = "EMPTY_SCAN"
            content_detail = {"detail": f"file has {st.st_size} bytes, below min_size_bytes ({cfg.min_size_bytes})"}
        sidecar: dict[str, Any] | None = None
        if content_reason is None and (cfg.sidecar_mode == "required" or os.path.isfile(str(path) + ".done")):
            try:
                sidecar = self._read_sidecar(path)
            except (_SidecarError, UnicodeDecodeError, OSError) as exc:
                content_reason, content_detail = "SIDECAR_INVALID", {"detail": str(exc)[:300]}

        result: ProbeResult = probe(path)
        if (content_reason is not None or not result.ok) and stable_for_s < cfg.truncated_grace_s:
            return IngestOutcome(
                action="pending",
                original_name=original_name,
                source_path=str(path),
                reason=content_reason or result.reason,
                detail=_CONTENT_GRACE_DETAIL,
            )
        sha = sha256_file(path)
        try:
            after = os.lstat(path)
        except OSError:
            return None
        if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
            self._log.info("%s changed while hashing; waiting for it to settle", path)
            return None
        if content_reason is None and not result.ok:
            content_reason = result.reason or "READ_ERROR"
            content_detail = {"detail": result.detail, "probe": result.to_dict()}
        if content_reason is None and sidecar:
            if "sha256" in sidecar and sidecar["sha256"].lower() != sha:
                content_reason, content_detail = "SIDECAR_INVALID", {"detail": "sidecar sha256 differs from the file"}
            elif "size_bytes" in sidecar and sidecar["size_bytes"] != st.st_size:
                content_reason, content_detail = "SIDECAR_INVALID", {"detail": "sidecar size_bytes differs from the file"}
        declared = None
        if content_reason is None and sidecar and "declared_stage" in sidecar:
            declared = {"stage_index": int(sidecar["declared_stage"]), "source": "SIDECAR"}

        outcome, resolver_error = self._resolve(original_name)
        if outcome is not None:
            parse, match = outcome.parse_dict(), outcome.match_dict()
        else:
            parse = minimal_parse("UNPARSEABLE", reason="RESOLUTION_PENDING")
            match = {"status": "UNPARSEABLE", "reason": resolver_error}

        existing = self.store.get_scan_by_sha(sha)
        if existing is not None and self._is_resume(existing, original_name, st):
            return self._resume(existing, path, st, sha, outcome, resolver_error)

        now = self._clock()
        ext = result.extension or Path(original_name).suffix.lower() or ".bin"
        if content_reason is None:
            rel = f"archive/scans/{now:%Y}/{now:%m}/{sha}{ext}"
        else:
            rel = f"quarantine/{now:%Y-%m-%d}/{sha}{ext}"
        row, created = self.store.register_scan(
            sha256=sha,
            original_name=original_name[:255],
            size_bytes=st.st_size,
            station_id=cfg.station_id,
            stored_rel_path=rel,
            parse=parse,
            match=match,
            format=result.fmt if result.ok else None,
            vertex_count=result.vertex_count if result.ok else None,
            face_count=result.face_count if result.ok else None,
            bbox_mm=result.bbox_mm if result.ok else None,
            declared_stage=declared,
            ingest_state="quarantined" if content_reason else "received",
            quarantine_reason=content_reason,
            quarantine_detail=({"original_name": original_name, **content_detail} if content_reason else None),
        )
        dest = self._root / row["stored_rel_path"]
        self._place(path, dest, sha, st)
        if content_reason is not None and created:
            self._write_reason(dest.parent, sha, row["scan_file_id"], original_name, content_reason, content_detail)
        elif not created:
            self._ensure_reason(row)
        self._dispose_source(path, st, sha)

        if not created and self.store.is_duplicate_drop(
            row, original_name=original_name[:255], station_id=cfg.station_id
        ):
            # Already processed, or the same bytes under another name/station: never
            # apply this drop's resolution to the row (a still-pending row is resolved
            # by _resume_pending from its own original_name).
            return IngestOutcome(
                action="duplicate",
                original_name=original_name,
                source_path=str(path),
                scan_file_id=row["scan_file_id"],
                job_id=(self.store.get_scan_meta(row["scan_file_id"]) or {}).get("job_id"),
                reason="SCAN_DUPLICATE",
            )
        if row["ingest_state"] == "quarantined":
            return IngestOutcome(
                action="quarantined",
                original_name=original_name,
                source_path=str(path),
                scan_file_id=row["scan_file_id"],
                reason=content_reason,
                detail=str(content_detail.get("detail") or "")[:500] or None,
            )
        return self._apply_outcome(row, outcome, resolver_error, source_path=str(path))

    def _is_resume(self, row: Mapping[str, Any], original_name: str, st: os.stat_result) -> bool:
        """True when this drop is the continuation of the ingest that registered
        ``row`` (a crash between register and move/record), not a new drop."""
        if (
            row["station_id"] != self.config.station_id
            or row["original_name"] != original_name[:255]
            or int(row["size_bytes"]) != st.st_size
        ):
            return False
        if not (self._root / row["stored_rel_path"]).is_file():
            return True  # the bytes never reached their stored location
        try:
            registered = parse_iso_utc(str(row["received_at"])).timestamp()
        except (TypeError, ValueError):
            return False
        return st.st_mtime_ns / 1e9 <= registered + 1.0

    def _resume(
        self,
        row: Mapping[str, Any],
        path: Path,
        st: os.stat_result,
        sha: str,
        outcome: ResolveOutcome | None,
        resolver_error: str | None,
    ) -> IngestOutcome:
        """Finish an interrupted ingest of ``row``: place, reason file, dispose."""
        self._log.warning("resuming the interrupted ingest of %s (%s)", path, row["scan_file_id"])
        self._place(path, self._root / row["stored_rel_path"], sha, st)
        self._ensure_reason(row)
        self._dispose_source(path, st, sha)
        scan_id = row["scan_file_id"]
        state = row["ingest_state"]
        if state == "received":
            return self._apply_outcome(row, outcome, resolver_error, source_path=str(path))
        meta = self.store.get_scan_meta(scan_id) or {}
        if state == "enqueued":
            return IngestOutcome(
                action="enqueued",
                original_name=row["original_name"],
                source_path=str(path),
                scan_file_id=scan_id,
                job_id=meta.get("job_id"),
            )
        return IngestOutcome(
            action="quarantined",
            original_name=row["original_name"],
            source_path=str(path),
            scan_file_id=scan_id,
            reason=meta.get("quarantine_reason"),
        )

    def _ensure_reason(self, row: Mapping[str, Any]) -> None:
        """Write a quarantined scan's ``<sha>.reason.json`` when it is missing."""
        if row["ingest_state"] != "quarantined":
            return
        meta = self.store.get_scan_meta(row["scan_file_id"]) or {}
        reason = meta.get("quarantine_reason")
        if not reason:
            return
        rel = str(row["stored_rel_path"])
        if rel.startswith("quarantine/"):
            directory = (self._root / rel).parent
        else:
            day = str(meta.get("quarantined_at") or "")[:10] or f"{self._clock():%Y-%m-%d}"
            directory = self._root / "quarantine" / day
        if (directory / f"{row['sha256']}.reason.json").is_file():
            return
        detail = meta.get("quarantine_detail") or {}
        self._write_reason(
            directory,
            row["sha256"],
            row["scan_file_id"],
            row["original_name"],
            reason,
            detail if isinstance(detail, Mapping) else {"detail": str(detail)},
            stored_rel_path=None if rel.startswith("quarantine/") else rel,
        )

    def _apply_outcome(
        self,
        row: Mapping[str, Any],
        outcome: ResolveOutcome | None,
        resolver_error: str | None,
        *,
        source_path: str | None,
    ) -> IngestOutcome:
        scan_id = row["scan_file_id"]
        name = row["original_name"]
        if outcome is None:
            return IngestOutcome(
                action="pending", original_name=name, source_path=source_path, scan_file_id=scan_id, reason=resolver_error
            )
        if outcome.status == "MATCHED":
            job_id, _created = self.store.match_and_enqueue(
                scan_id,
                match=outcome.match_dict(),
                parse=outcome.parse_dict(),
                sku=outcome.sku,  # type: ignore[arg-type]
                compiled_part_sha256=outcome.compiled_part_sha256,  # type: ignore[arg-type]
                priority=PRIORITY_BY_KIND["inspect"],
                max_attempts=MAX_ATTEMPTS_BY_KIND["inspect"],
            )
            return IngestOutcome(
                action="enqueued", original_name=name, source_path=source_path, scan_file_id=scan_id, job_id=job_id
            )
        reason = RESOLVE_STATUS_TO_QUARANTINE_REASON[outcome.status]
        detail = {
            "original_name": name,
            "resolve_status": outcome.status,
            "resolve_reason": outcome.reason,
            "sku": outcome.sku,
        }
        self.store.quarantine_scan(scan_id, reason, detail, match=outcome.match_dict(), parse=outcome.parse_dict())
        qdir = self._root / "quarantine" / f"{self._clock():%Y-%m-%d}"
        self._write_reason(qdir, row["sha256"], scan_id, name, reason, detail, stored_rel_path=row["stored_rel_path"])
        return IngestOutcome(
            action="quarantined",
            original_name=name,
            source_path=source_path,
            scan_file_id=scan_id,
            reason=reason,
            detail=outcome.reason,
        )

    def _resume_pending(self, *, force: bool = False) -> list[IngestOutcome]:
        """Re-resolve this station's scans stuck in ``received`` (resolver failure or crash)."""
        now_mono = time.monotonic()
        if not force and now_mono - self._last_pending_check < max(5.0, self.config.poll_interval_s):
            return []
        self._last_pending_check = now_mono
        outcomes: list[IngestOutcome] = []
        for row in self.store.list_scans(station_id=self.config.station_id, ingest_state="received", limit=50):
            if not (self._root / row["stored_rel_path"]).is_file():
                continue  # bytes still in the inbox (crash before placement): the inbox path finishes it
            scan_id = row["scan_file_id"]
            try:
                outcome, err = self._resolve(row["original_name"])
                outcomes.append(self._apply_outcome(row, outcome, err, source_path=None))
                self._apply_failures.pop(scan_id, None)
            except Exception as exc:
                failures = self._apply_failures.get(scan_id, 0) + 1
                self._apply_failures[scan_id] = failures
                self._log.exception("resuming pending scan %s failed (attempt %d)", scan_id, failures)
                if failures >= _MAX_INGEST_FAILURES:
                    quarantined = self._quarantine_apply_failure(row, exc)
                    if quarantined is not None:
                        outcomes.append(quarantined)
        return [o for o in outcomes if o.action != "pending"]

    def _quarantine_apply_failure(self, row: Mapping[str, Any], exc: Exception) -> IngestOutcome | None:
        """Park a scan whose resolution keeps failing to apply where an operator sees it."""
        scan_id = row["scan_file_id"]
        detail = {
            "original_name": row["original_name"],
            "resolve_status": "PART_NOT_READY",
            "resolve_reason": "RESOLVE_APPLY_FAILED",
            "detail": f"RESOLVE_APPLY_FAILED: {type(exc).__name__}: {exc}"[:300],
        }
        try:
            self.store.quarantine_scan(scan_id, "PART_NOT_READY", detail)
            qdir = self._root / "quarantine" / f"{self._clock():%Y-%m-%d}"
            self._write_reason(
                qdir, row["sha256"], scan_id, row["original_name"], "PART_NOT_READY", detail,
                stored_rel_path=row["stored_rel_path"],
            )  # fmt: skip
        except Exception:
            self._log.exception("cannot quarantine scan %s after repeated apply failures", scan_id)
            return None
        self._apply_failures.pop(scan_id, None)
        self._log.error("scan %s quarantined: its resolution could not be applied (%s)", scan_id, exc)
        return IngestOutcome(
            action="quarantined",
            original_name=row["original_name"],
            scan_file_id=scan_id,
            reason="PART_NOT_READY",
            detail="RESOLVE_APPLY_FAILED",
        )

    # ------------------------------------------------------------------ file moves
    def _place(self, source: Path, dest: Path, sha: str, st: os.stat_result) -> None:
        """Put the source bytes at ``dest`` (idempotent when ``dest`` already holds them)."""
        if dest.is_file() and not dest.is_symlink():
            if dest.stat().st_size == st.st_size and sha256_file(dest) == sha:
                return
            raise RuntimeError(f"{dest} exists with different content; refusing to overwrite")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.config.source_policy == "move":
            try:
                if os.stat(dest.parent).st_dev == st.st_dev:
                    os.replace(source, dest)
                    return
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
        partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
        try:
            with open(source, "rb") as src, open(partial, "wb") as out:
                shutil.copyfileobj(src, out, _HASH_CHUNK)
                out.flush()
                os.fsync(out.fileno())
            if sha256_file(partial) != sha:
                raise RuntimeError(f"copy of {source} does not hash to {sha}")
            os.replace(partial, dest)
            _fsync_dir(dest.parent)
        finally:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:  # pragma: no cover
                    pass

    def _dispose_source(self, source: Path, st: os.stat_result, sha: str) -> None:
        if self.config.source_policy == "copy_leave":
            self.store.record_ingest_source(self.config.station_id, str(source), st.st_size, st.st_mtime_ns, sha)
            return
        for path in (source, Path(str(source) + ".done")):
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
            except FileNotFoundError:
                pass

    def _write_reason(
        self,
        directory: Path,
        sha: str,
        scan_file_id: str,
        original_name: str,
        reason: str,
        detail: Mapping[str, Any],
        *,
        stored_rel_path: str | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        doc = {
            "scan_file_id": scan_file_id,
            "sha256": sha,
            "station_id": self.config.station_id,
            "original_name": original_name,
            "reason": reason,
            "detail": dict(detail),
            "stored_rel_path": stored_rel_path,
            "at": iso_utc(self._clock()),
        }
        target = directory / f"{sha}.reason.json"
        tmp = target.with_name(target.name + PARTIAL_SUFFIX)
        tmp.write_text(json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, target)


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":  # pragma: no cover
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)
