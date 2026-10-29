"""Supervised job runner: one leased job at a time, every step a subprocess.

Each job runs from the pinned engine bundle directory (``engine_root``, the
subprocess cwd) with ``PYTHONPATH=<engine_root>:<engine_root>/backend``, the
thread caps and a whitelisted environment (secrets such as the TypeSafe key
never reach a step). There is no legacy worker on this path (plan §10
amendment 1).

Inspect job (ENGINE_IO §5):

1. Lease (JOB_STARTED); reset the workspace ``<data_root>/jobs/<job_id>/``.
2. Verify everything before spawning anything (stage ``preparing``):

   - ``RecipeProvider.load`` (else RECIPE_MISSING);
   - the recipe bytes validate as ``PartRecipeV1`` (hashes recomputed) and
     name the job's recipe id and sha (else RECIPE_SHA_MISMATCH);
   - the CAD exists (RECIPE_CAD_MISSING) and hashes to ``recipe.cad.sha256``
     (RECIPE_CAD_HASH_MISMATCH);
   - ``compiled_part.json`` exists (COMPILED_PART_MISSING), validates, equals
     the job's ``compiled_part_sha256`` and agrees with the recipe
     (COMPILED_PART_HASH_MISMATCH), and is COMPILED (COMPILE_BLOCKED); its
     directory is copied to ``compiled/`` and every stage file's sha256 is
     checked (COMPILED_PART_HASH_MISMATCH);
   - the archived scan exists (SCAN_MISSING) and its staged copy
     ``input/scan.<ext>`` hashes to the job's ``scan_sha256``
     (SCAN_HASH_MISMATCH).

   Then ``recipe.json`` (canonical bytes), ``input/context.json``
   (``ResultContextV1``) and ``job.json`` (``LineJobV1``) are written.
3. Engine step (stage ``engine``, progress 5-80)::

       <python> -m domains.production.engine.inspect --compiled <ws>/compiled/compiled_part.json
           --recipe <ws>/recipe.json --scan <ws>/input/scan.<ext> --out <ws>/engine
           --scan-sha256 S --compiled-sha256 C --recipe-sha256 R
           --time-budget-s 170 [--declared-stage K --declared-stage-source SRC]
           [--engine-bundle-id ID] [--config CFG]

   Blindness (plan §10 amendment 4, ENGINE_IO §4): no argument carries the
   filename, serial or any other job identity. ``input/context.json`` is never
   passed to E3; E3 writes ``result.context`` empty, and the runner rejects a
   non-empty one (ENGINE_OUTPUT_INVALID ``ENGINE_CONTEXT_NOT_EMPTY``).

   Hard stop 180 s (SIGTERM, then SIGKILL after ``kill_grace_s`` = 3 s)
   and 32 GiB RSS (ENGINE_IO-A5; SIGKILL at once) (``ENGINE_HARD_TIMEOUT_S``,
   ``ENGINE_RSS_LIMIT_BYTES``). Mapping: timeout -> WORKER_TIMEOUT; RSS ->
   WORKER_RSS_LIMIT; cancel -> cancelled; exit 2 -> the ``engine_error.json``
   code (one of the E3 input codes, else WORKER_ERROR); exit 3, or another
   non-zero exit after an ERROR line -> WORKER_ERROR; any other non-zero exit
   (signal, traceback) -> WORKER_CRASH; exit 0 without RESULT or ``result.json`` ->
   WORKER_NO_RESULT. ``engine/result.json`` must validate as
   ``ShopfloorInspectionResultV1``, pass
   ``consistency.check_result_against_compiled``, name the job's scan sha,
   carry an empty context and agree with the RESULT line, else
   ENGINE_OUTPUT_INVALID.
3b. Zero-shot overlay (optional; ``RunnerConfig.zeroshot_module``, set by the wiring when the
   engine bundle carries the pinned lane ``domains/production/zeroshot``). The zs step runs in
   PARALLEL with E3, under its own supervisor limits (``zeroshot_timeout_s`` 90 s,
   ``zeroshot_rss_limit_bytes`` = the engine's 32 GiB, so the two steps together stay under 64 GiB)::

       <python> -m domains.production.zeroshot.lane_cli --scan <ws>/input/scan.<ext>
           --compiled <ws>/compiled/compiled_part.json --out <ws>/zeroshot/zs_lines.json

   Like E3 it is blind to job identity (only the staged ``scan.<ext>`` and the CompiledPart). Its
   group is recorded in ``logs/active_step_zs.json`` and reaped like the main step. If E3 fails, is
   cancelled or loses the lease, the zs step is stopped first. Once E3's result validated, the
   runner waits for the zs step and runs the merge step (stage ``postprocess``, 60 s)::

       <python> -m domains.production.zeroshot.overlay_cli --result <ws>/engine/result.json
           --compiled <ws>/compiled/compiled_part.json --zs <ws>/zeroshot/zs_lines.json
           --out <ws>/zeroshot/zeroshot_overlay_v1.json

   Containment: any zs or merge outcome other than success (crash, timeout, RSS, pin mismatch, bad
   output) means **no overlay** and nothing else: the job, E3's result, the summary and the
   verdict are unchanged. ``<ws>/zeroshot/overlay_status.json`` records what happened. The overlay
   (``sherman_zeroshot_overlay_v1``) can only lower a pass-like verdict to REVIEW; it is never part
   of ``shopfloor_result.json`` (served separately, ``GET /jobs/{id}/zeroshot-overlay``).
4. Render step, C06 (stage ``postprocess``, progress 80-99)::

       <python> -m domains.production.reporting.render_cli --job-dir <ws>
           --result <ws>/engine/result.json --out <ws>/result [--engine-bundle-id ID]

   Exit 0: ``result/shopfloor_result.json`` was written (viewer assets and
   PDF next to it). Exit 2 (invalid input) or 3 (internal): C06 writes
   ``result/postprocess_error.json {code, message}``. Mapping: timeout ->
   POSTPROCESS_TIMEOUT; anything else -> POSTPROCESS_ERROR with that
   message. The rendered result must validate, stay consistent, and keep the
   engine's verdict: only ``viewer_assets``, ``artifacts``, appended
   ``warnings`` and ``context`` may differ from ``engine/result.json`` (else
   POSTPROCESS_ERROR ``RESULT_ALTERED``). ``context`` must be exactly
   ``input/context.json`` (C06 fills it, ENGINE_IO §5). If C06 left it empty,
   the runner fills it ("C06 or the runner"), appends the warning
   ``CONTEXT_FILLED_WARNING`` (the rendered PDF has no job identity) and
   rewrites ``result/shopfloor_result.json`` as canonical JSON, changing
   nothing else.
   Any other context is ``RESULT_ALTERED``.
5. ``complete_job`` with ``result_summary`` (``JobResultSummaryV1``) and
   JOB_COMPLETED.

Recipe-compile job (``kind='recipe_compile'``): CAD (and flat) checks, then
``python -m domains.production.engine.compile --recipe --cad [--flat] --out
<ws>/compile_out [--config]`` (120 s, 32 GiB). COMPILED (exit 0) or BLOCKED
(exit 2 with ``compiled_part.json``) is handed to the ``CompiledPartSink``
and announced with RECIPE_COMPILED / RECIPE_COMPILE_BLOCKED; any failure
emits RECIPE_COMPILE_FAILED. Each of these events is written in the same
transaction as the job's final state (``complete_job`` / ``fail_job``
``extra_event``), so a lost lease never announces an outcome.

Durability: a lease keeper thread renews the lease and writes the ``runner``
heartbeat; if a renewal fails (the lease was recovered elsewhere) the running
step is killed and nothing more is written for that job. Startup calls
``recover_expired_leases``; with a stable ``instance_id`` (C11: e.g.
``runner``) it also reclaims at once the leases of its own earlier processes
that are no longer alive, so a killed-and-restarted line service resumes its
job without waiting for the lease TTL. A lease owner is
``<instance_id>@<host>:<pid>`` (``lease_owner_for``); only owners on *this*
host are probed, so two hosts sharing one store with the same
``instance_id`` never reclaim each other's live leases (those wait for the
TTL). Exactly one job runs per
runner process.

Orphaned steps: every step runs with the supervisor's parent-death sentinel
(it dies with the runner process), and its group is recorded in
``<ws>/logs/active_step.json`` (``{pid, pgid, host, tag, job_id, attempt,
owner, stage, cmd_sha256, spawned_at}``) for as long as it runs. Before a job
(re)starts, and when a dead predecessor's lease is reclaimed, a surviving
recorded group is killed (``supervisor.kill_recorded_group``: only when a
live member still carries the job id) and waited for, so two engines never
write the same ``engine/`` and never run at once. A group that cannot be
stopped fails the job with WORKER_INTERRUPTED instead of starting a second
engine. Step logs are kept per attempt: ``logs/attempt-<n>/<step>_step.log``.

Contract modules (pydantic) are imported lazily inside the job methods, so
importing this module stays cheap.
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
import shutil
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.production_line.ingest.hotfolder import sha256_file
from apps.production_line.ingest.ports import CompiledPartSink, RecipeProvider, ResolvedRecipe
from apps.production_line.runner.supervisor import SupervisedResult, kill_recorded_group, run_supervised
from apps.production_line.runner.worker_protocol import error_message, parse_protocol_line, sanitize_message
from apps.production_line.store.line_store import LineStore, utc_now
from apps.production_line.store.vocabulary import (
    ENGINE_HARD_TIMEOUT_S,
    ENGINE_INPUT_ERROR_CODES,
    ENGINE_RSS_LIMIT_BYTES,
)

__all__ = [
    "RunnerConfig",
    "JobRunner",
    "JobFailure",
    "default_thread_count",
    "build_step_env",
    "THREAD_ENV_VARS",
    "ENV_PASSTHROUGH",
    "lease_owner_for",
    "lease_owner_prefix",
    "CONTEXT_FILLED_WARNING",
]

#: Appended to ``result.warnings`` when the runner (not C06) filled ``result.context``:
#: the PDF and overview rendered before that carry no job id or serial.
CONTEXT_FILLED_WARNING = "CONTEXT_FILLED_BY_RUNNER: the rendered report was produced without job identity"

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
#: The only parent variables a step inherits (plus ``RunnerConfig.extra_env``).
ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "USERPROFILE",
    "SHERMAN_REPORT_FONT_PATH",
)
_ENGINE_PROGRESS = (5.0, 80.0)
_RENDER_PROGRESS = (80.0, 99.0)
_COMPILE_PROGRESS = (5.0, 95.0)
_CANCEL_POLL_S = 0.5
_TAIL_LINES = 20
#: Workspace-relative record of the step group currently running for a job.
ACTIVE_STEP_REL = Path("logs") / "active_step.json"
#: Record of the zero-shot step, which runs alongside the engine step.
ZS_ACTIVE_STEP_REL = Path("logs") / "active_step_zs.json"
#: Workspace-relative directory of the zs step and overlay outputs.
ZEROSHOT_DIR_REL = Path("zeroshot")


def default_thread_count() -> str:
    """Thread cap per step: ``BEND_WORKER_THREADS`` when set, else 1 on macOS,
    else ``max(4, min(16, cpu))`` (same rule as the legacy launcher)."""
    configured = os.environ.get("BEND_WORKER_THREADS", "").strip()
    if configured.isdigit() and int(configured) > 0:
        return str(int(configured))
    if sys.platform == "darwin":
        return "1"
    return str(max(4, min(16, os.cpu_count() or 8)))


@dataclass(frozen=True)
class RunnerConfig:
    """Runner settings. ``engine_root`` is the engine bundle directory (cwd of
    every step); ``data_root`` is ``SHERMAN_LINE_DATA_ROOT``."""

    engine_root: Path
    python_exe: Path
    data_root: Path
    jobs_root: Path | None = None
    inspect_module: str = "domains.production.engine.inspect"
    render_module: str = "domains.production.reporting.render_cli"
    compile_module: str = "domains.production.engine.compile"
    engine_timeout_s: float = float(ENGINE_HARD_TIMEOUT_S)
    engine_rss_limit_bytes: int = ENGINE_RSS_LIMIT_BYTES
    engine_time_budget_s: float = 170.0
    render_timeout_s: float = 120.0
    render_rss_limit_bytes: int = ENGINE_RSS_LIMIT_BYTES
    compile_timeout_s: float = 120.0
    compile_rss_limit_bytes: int = ENGINE_RSS_LIMIT_BYTES
    lease_ttl_s: float = 60.0
    heartbeat_s: float = 15.0
    supervisor_poll_s: float = 0.5
    idle_poll_s: float = 1.0
    kill_grace_s: float = 3.0
    thread_env: Mapping[str, str] | None = None
    engine_config_path: Path | None = None
    compile_config_path: Path | None = None
    engine_bundle_id: str | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)
    #: Zero-shot overlay step (3b); None disables it.
    zeroshot_module: str | None = None
    overlay_module: str = "domains.production.zeroshot.overlay_cli"
    zeroshot_timeout_s: float = 90.0
    zeroshot_rss_limit_bytes: int = ENGINE_RSS_LIMIT_BYTES
    overlay_timeout_s: float = 60.0

    @property
    def resolved_jobs_root(self) -> Path:
        return Path(self.jobs_root) if self.jobs_root is not None else Path(self.data_root) / "jobs"


def build_step_env(config: RunnerConfig) -> dict[str, str]:
    """Whitelisted environment for a step (see module docstring)."""
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
    root = str(Path(config.engine_root))
    env["PYTHONPATH"] = os.pathsep.join([root, str(Path(root) / "backend")])
    threads = default_thread_count()
    caps = dict(config.thread_env) if config.thread_env is not None else {k: threads for k in THREAD_ENV_VARS}
    for key in THREAD_ENV_VARS:
        env[key] = str(caps.get(key, threads))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env.update({str(k): str(v) for k, v in config.extra_env.items()})
    return env


class JobFailure(Exception):
    """A job-level failure with a contract FAILURE_CODE."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = sanitize_message(message)


class _Cancelled(Exception):
    pass


class _LeaseLost(Exception):
    pass


@dataclass
class _StepCapture:
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    tail: list[str] = field(default_factory=list)


@dataclass
class _ZsRun:
    """The zero-shot step running alongside the engine step (3b)."""

    abort: threading.Event
    started: float
    thread: threading.Thread | None = None
    result: SupervisedResult | None = None
    error: str | None = None


@dataclass
class _JobContext:
    job: dict[str, Any]
    owner: str
    started: float
    lease_lost: threading.Event
    peak_rss: int = 0
    log_dir: Path | None = None
    _last_cancel_poll: float = 0.0
    _cancel_seen: bool = False

    @property
    def job_id(self) -> str:
        return str(self.job["job_id"])


def lease_owner_prefix(instance_id: str, host: str | None = None) -> str:
    """``<instance_id>@<host>:``, shared by every process of ``instance_id`` on ``host``
    (default: this host). The host is always part of a lease owner so that a
    pid is never interpreted on a host that did not issue it."""
    return f"{instance_id}@{host if host is not None else socket.gethostname()}:"


def lease_owner_for(instance_id: str, pid: int | None = None, host: str | None = None) -> str:
    """The lease owner ``<instance_id>@<host>:<pid>`` of a runner process (default: this one)."""
    return lease_owner_prefix(instance_id, host) + str(os.getpid() if pid is None else int(pid))


class _LeaseKeeper(threading.Thread):
    def __init__(self, store: LineStore, ctx: _JobContext, ttl_s: float, interval_s: float, instance_id: str,
                 log: logging.Logger) -> None:
        super().__init__(name=f"lease-{ctx.job_id}", daemon=True)
        self._store = store
        self._ctx = ctx
        self._ttl = ttl_s
        self._interval = interval_s
        self._instance_id = instance_id
        self._halt = threading.Event()
        self._log = log

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            try:
                if not self._store.renew_lease(self._ctx.job_id, self._ctx.owner, self._ttl):
                    self._log.error("lease lost for job %s; stopping it", self._ctx.job_id)
                    self._ctx.lease_lost.set()
                    return
                self._store.put_heartbeat(
                    "runner", self._instance_id, {"job_id": self._ctx.job_id, "kind": self._ctx.job["kind"]}
                )
            except Exception:  # pragma: no cover - transient DB errors: retry next tick
                self._log.exception("lease renewal failed for job %s", self._ctx.job_id)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


class JobRunner:
    """Leases and runs line jobs (module docstring). One job at a time."""

    def __init__(
        self,
        config: RunnerConfig,
        store: LineStore,
        recipe_provider: RecipeProvider,
        clock: Callable[[], datetime] = utc_now,
        logger: logging.Logger | None = None,
        instance_id: str | None = None,
        *,
        compiled_sink: CompiledPartSink | None = None,
        recover_on_start: bool = True,
    ) -> None:
        if config.engine_timeout_s > ENGINE_HARD_TIMEOUT_S or config.engine_rss_limit_bytes > ENGINE_RSS_LIMIT_BYTES:
            raise ValueError("engine limits may be tightened but never exceed 180 s / 32 GiB")
        if config.zeroshot_timeout_s > ENGINE_HARD_TIMEOUT_S or config.zeroshot_rss_limit_bytes > ENGINE_RSS_LIMIT_BYTES:
            raise ValueError("zero-shot limits may never exceed the engine's 180 s / 32 GiB")
        self.config = config
        self.store = store
        self.provider = recipe_provider
        self.compiled_sink = compiled_sink
        self._clock = clock
        self._log = logger or logging.getLogger(__name__)
        self.instance_id = instance_id or f"runner-{uuid.uuid4().hex[:12]}"
        self._busy = threading.Lock()
        self._jobs_root = config.resolved_jobs_root
        self._data_root = Path(config.data_root)
        if recover_on_start:
            self.recover()

    # ------------------------------------------------------------------ lifecycle
    def _dead_predecessor_owners(self) -> list[str]:
        """Lease owners ``<instance_id>@<this host>:<pid>`` of this runner's earlier,
        now dead, processes on this host (only meaningful with a stable
        ``instance_id``). An owner naming another host, or none, is never
        probed: a pid is only meaningful on the host that issued it."""
        if os.name != "posix":  # pragma: no cover - no cheap liveness probe
            return []
        prefix = lease_owner_prefix(self.instance_id)
        dead: list[str] = []
        for job in self.store.list_jobs(statuses=["running"], limit=500):
            owner = job.get("lease_owner") or ""
            pid_text = owner[len(prefix):] if owner.startswith(prefix) else ""
            if not pid_text.isdigit() or int(pid_text) == os.getpid():
                continue
            try:
                os.kill(int(pid_text), 0)
            except ProcessLookupError:
                dead.append(owner)
            except PermissionError:
                continue  # alive (another user's process)
        return dead

    def recover(self) -> list[tuple[str, str]]:
        """Recover expired leases, and at once those of this runner's dead predecessors."""
        dead = self._dead_predecessor_owners()
        if dead:
            self._log.warning("reclaiming leases of dead runner processes: %s", ", ".join(dead))
            for job_id in self.store.expire_leases_of(dead):
                try:
                    self._reap_recorded_step(job_id)
                except JobFailure as exc:  # retried (and enforced) when the job is leased again
                    self._log.error("job %s: %s", job_id, exc)
                except Exception:
                    self._log.exception("job %s: reaping its orphaned step failed", job_id)
        recovered = self.store.recover_expired_leases()
        for job_id, action in recovered:
            self._log.warning("recovered job %s after an expired lease: %s", job_id, action)
        return recovered

    def run_forever(self, stop_event: threading.Event) -> None:
        last_recover = time.monotonic()
        last_beat = 0.0
        while not stop_event.is_set():
            now = time.monotonic()
            # Transient store errors (database locked, disk I/O) must never end the loop.
            if now - last_beat >= self.config.heartbeat_s:
                try:
                    self.store.put_heartbeat("runner", self.instance_id, {"job_id": None})
                except Exception:
                    self._log.exception("runner heartbeat failed")
                last_beat = now
            if now - last_recover >= self.config.lease_ttl_s:
                try:
                    self.recover()
                except Exception:
                    self._log.exception("lease recovery failed")
                last_recover = now
            try:
                ran = self.run_once()
            except Exception:  # pragma: no cover - run_once already contains job errors
                self._log.exception("runner loop error")
                ran = None
            if ran is None:
                stop_event.wait(self.config.idle_poll_s)

    def run_once(self) -> str | None:
        """Lease and run one job to a terminal state; return its id (None when idle)."""
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("a JobRunner runs exactly one job at a time")
        try:
            owner = lease_owner_for(self.instance_id)
            job = self.store.lease_next(owner, self.config.lease_ttl_s, self.config.engine_bundle_id)
            if job is None:
                return None
            ctx = _JobContext(job=job, owner=owner, started=time.monotonic(), lease_lost=threading.Event())
            interval = max(0.05, min(self.config.heartbeat_s, self.config.lease_ttl_s / 3.0))
            keeper = _LeaseKeeper(self.store, ctx, self.config.lease_ttl_s, interval, self.instance_id, self._log)
            keeper.start()
            try:
                self._run_job(ctx)
            finally:
                keeper.stop()
            return ctx.job_id
        finally:
            self._busy.release()

    def _duration(self, ctx: _JobContext) -> float:
        return round(time.monotonic() - ctx.started, 3)

    def _run_job(self, ctx: _JobContext) -> None:
        kind = ctx.job["kind"]
        try:
            self._reap_recorded_step(ctx.job_id)
            if kind == "inspect":
                self._run_inspect(ctx)
            elif kind == "recipe_compile":
                self._run_compile(ctx)
            else:
                raise JobFailure("INTERNAL_ERROR", f"unknown job kind {kind!r}")
        except _LeaseLost:
            self._log.error("job %s abandoned: lease lost", ctx.job_id)
        except _Cancelled:
            if not self.store.mark_cancelled(ctx.job_id, ctx.owner, ctx.peak_rss or None, self._duration(ctx)):
                self._log.error("job %s: cancel not recorded (lease lost)", ctx.job_id)
        except JobFailure as failure:
            self._fail(ctx, failure.code, failure.message)
        except Exception as exc:
            self._log.exception("job %s: internal error", ctx.job_id)
            self._fail(ctx, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")

    def _fail(self, ctx: _JobContext, code: str, message: str) -> None:
        extra = None
        if ctx.job["kind"] == "recipe_compile":
            extra = (
                "RECIPE_COMPILE_FAILED",
                {"recipe_id": ctx.job["recipe_id"], "recipe_sha256": ctx.job["recipe_sha256"], "code": code},
            )
        if not self.store.fail_job(
            ctx.job_id, ctx.owner, code, message, ctx.peak_rss or None, self._duration(ctx), extra_event=extra
        ):
            self._log.error("job %s: failure %s not recorded (lease lost)", ctx.job_id, code)

    # ------------------------------------------------------------------ orphaned steps
    def _active_step_path(self, job_id: str, rel: Path = ACTIVE_STEP_REL) -> Path:
        return self._jobs_root / job_id / rel

    def _reap_recorded_step(self, job_id: str) -> None:
        """Kill the step groups (main and zero-shot) an earlier attempt left running (module docstring)."""
        for rel in (ACTIVE_STEP_REL, ZS_ACTIVE_STEP_REL):
            self._reap_one(job_id, rel)

    def _reap_one(self, job_id: str, rel: Path) -> None:
        record_path = self._active_step_path(job_id, rel)
        if record_path.is_symlink():
            record_path.unlink()
            return
        if not record_path.is_file():
            return
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            pgid = int(record["pgid"])
            tag = str(record["tag"])
            host = str(record.get("host") or "")
        except (OSError, ValueError, KeyError, TypeError):
            self._log.error("job %s: unreadable %s; removing it", job_id, rel)
            record_path.unlink(missing_ok=True)
            return
        if host and host != socket.gethostname():
            self._log.warning("job %s: recorded step ran on host %s; cannot check it from here", job_id, host)
            record_path.unlink(missing_ok=True)
            return
        status = kill_recorded_group(pgid, tag, grace_s=self.config.kill_grace_s)
        if status == "alive":
            raise JobFailure(
                "WORKER_INTERRUPTED", f"the previous attempt's step (process group {pgid}) could not be stopped"
            )
        if status == "killed":
            self._log.warning("job %s: killed the orphaned step of an earlier attempt (pgid %s)", job_id, pgid)
        record_path.unlink(missing_ok=True)

    def _record_step(self, ctx: _JobContext, stage: str, cmd: list[str],
                     rel: Path = ACTIVE_STEP_REL) -> Callable[[int, int], None]:
        record_path = self._active_step_path(ctx.job_id, rel)

        def on_spawn(pid: int, pgid: int) -> None:
            record = {
                "pid": pid,
                "pgid": pgid,
                "host": socket.gethostname(),
                "tag": ctx.job_id,
                "job_id": ctx.job_id,
                "attempt": int(ctx.job.get("attempts") or 0),
                "owner": ctx.owner,
                "stage": stage,
                "cmd_sha256": hashlib.sha256("\0".join(cmd).encode("utf-8")).hexdigest(),
                "spawned_at": utc_now().isoformat(),
            }
            record_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_atomic(record_path, json.dumps(record, sort_keys=True).encode("utf-8"))

        return on_spawn

    # ------------------------------------------------------------------ helpers
    def _check_live(self, ctx: _JobContext) -> None:
        if ctx.lease_lost.is_set():
            raise _LeaseLost()
        if self.store.is_cancel_requested(ctx.job_id):
            raise _Cancelled()

    def _cancel_check(self, ctx: _JobContext) -> Callable[[], bool]:
        def check() -> bool:
            if ctx.lease_lost.is_set():
                return True
            now = time.monotonic()
            if now - ctx._last_cancel_poll >= _CANCEL_POLL_S:
                ctx._last_cancel_poll = now
                ctx._cancel_seen = self.store.is_cancel_requested(ctx.job_id)
            return ctx._cancel_seen

        return check

    def _progress(self, ctx: _JobContext, stage: str, pct: float, message: str) -> None:
        if not self.store.update_progress(ctx.job_id, ctx.owner, stage, pct, message):
            ctx.lease_lost.set()

    def _on_line(self, ctx: _JobContext, stage: str, span: tuple[float, float], capture: _StepCapture) -> Callable[[str], None]:
        lo, hi = span

        def handle(line: str) -> None:
            tag, payload = parse_protocol_line(line)
            if tag is None:
                if line.strip():
                    capture.tail.append(sanitize_message(line)[:300])
                    del capture.tail[:-_TAIL_LINES]
                return
            assert payload is not None
            if tag == "PROGRESS":
                try:
                    p = float(payload.get("progress", 0.0))
                except (TypeError, ValueError):
                    p = 0.0
                p = max(0.0, min(100.0, p))
                msg = sanitize_message(payload.get("message") or payload.get("stage") or stage)[:200]
                self._progress(ctx, stage, lo + (hi - lo) * p / 100.0, msg)
            elif tag == "RESULT":
                capture.result = payload
            elif tag == "ERROR":
                capture.error = payload

        return handle

    def _run_step(
        self,
        ctx: _JobContext,
        cmd: list[str],
        *,
        stage: str,
        span: tuple[float, float],
        timeout_s: float,
        rss_limit_bytes: int,
        log_path: Path,
    ) -> tuple[SupervisedResult, _StepCapture]:
        self._check_live(ctx)
        capture = _StepCapture()
        self._progress(ctx, stage, span[0], f"starting {stage}")
        result = run_supervised(
            cmd,
            env=build_step_env(self.config),
            cwd=self.config.engine_root,
            timeout_s=timeout_s,
            rss_limit_bytes=rss_limit_bytes,
            poll_s=self.config.supervisor_poll_s,
            cancel_check=self._cancel_check(ctx),
            on_line=self._on_line(ctx, stage, span, capture),
            log_path=log_path,
            kill_grace_s=self.config.kill_grace_s,
            on_spawn=self._record_step(ctx, stage, cmd),
            identity_tag=ctx.job_id,
        )
        # run_supervised returns only once the whole group is gone.
        self._active_step_path(ctx.job_id).unlink(missing_ok=True)
        ctx.peak_rss = max(ctx.peak_rss, result.peak_rss_bytes)
        if ctx.lease_lost.is_set():
            raise _LeaseLost()
        if result.cancelled:
            raise _Cancelled()
        return result, capture

    @staticmethod
    def _crash_message(result: SupervisedResult, capture: _StepCapture, what: str) -> str:
        tail = " | ".join(capture.tail[-3:])
        return f"{what} exited with code {result.exit_code}" + (f"; last output: {tail}" if tail else "")

    @staticmethod
    def _read_error_file(path: Path) -> tuple[str | None, str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, ""
        if not isinstance(data, dict):
            return None, ""
        return error_message(data)

    def _reset_workspace(self, ctx: _JobContext, ws: Any) -> None:
        """Clear the previous attempt's outputs; keep ``logs/`` (one dir per attempt)."""
        root = Path(ws.root)
        if root.is_symlink():
            raise JobFailure("INTERNAL_ERROR", "job workspace is a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self._reap_recorded_step(ctx.job_id)  # never clear engine/ under a live step
        for child in (ws.compiled_dir, ws.input_dir, ws.engine_dir, ws.result_dir, root / "compile_out",
                      root / ZEROSHOT_DIR_REL):
            child = Path(child)
            if child.is_symlink():
                child.unlink()
            elif child.exists():
                shutil.rmtree(child)
        Path(ws.input_dir).mkdir()
        Path(ws.result_dir).mkdir()
        logs = root / "logs"
        if logs.is_symlink():
            logs.unlink()
        logs.mkdir(exist_ok=True)
        attempt_dir = logs / f"attempt-{int(ctx.job.get('attempts') or 0)}"
        if attempt_dir.is_symlink():
            attempt_dir.unlink()
        elif attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir()
        ctx.log_dir = attempt_dir

    def _log_path(self, ctx: _JobContext, ws: Any, step: str) -> Path:
        base = ctx.log_dir if ctx.log_dir is not None else Path(ws.root) / "logs"
        return base / f"{step}_step.log"

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _load_recipe(self, ctx: _JobContext) -> tuple[ResolvedRecipe, Any]:
        from pydantic import ValidationError

        from contracts.production.part_recipe import PartRecipeV1

        job = ctx.job
        pinned = job.get("compiled_part_sha256") if job.get("kind") == "inspect" else None
        try:
            if pinned and _accepts_keyword(self.provider.load, "compiled_part_sha256"):
                # C05: the job runs against the compile it was enqueued with, also when a later
                # recompute superseded it (kept under recipes/compiled_history/).
                resolved = self.provider.load(job["recipe_id"], compiled_part_sha256=pinned)
            else:
                resolved = self.provider.load(job["recipe_id"])
        except KeyError:
            raise JobFailure("RECIPE_MISSING", f"recipe {job['recipe_id']} is not available") from None
        try:
            recipe = PartRecipeV1.model_validate_json(resolved.recipe_json_bytes)
        except ValidationError as exc:
            raise JobFailure("RECIPE_SHA_MISMATCH", f"recipe bytes do not validate: {_first_error(exc)}") from None
        if recipe.recipe_sha256 != job["recipe_sha256"] or recipe.recipe_id != job["recipe_id"]:
            raise JobFailure(
                "RECIPE_SHA_MISMATCH",
                f"recipe {recipe.recipe_id} sha {recipe.recipe_sha256[:12]} != job {job['recipe_id']} sha "
                f"{job['recipe_sha256'][:12]}",
            )
        cad = Path(resolved.cad_path)
        if not cad.is_absolute() or not cad.is_file():
            raise JobFailure("RECIPE_CAD_MISSING", f"recipe CAD not found at {cad}")
        if sha256_file(cad) != recipe.cad.sha256:
            raise JobFailure("RECIPE_CAD_HASH_MISMATCH", "recipe CAD bytes do not hash to recipe.cad.sha256")
        return resolved, recipe

    def _context_dict(self, job: Mapping[str, Any], scan: Mapping[str, Any], recipe: Any = None) -> dict[str, Any]:
        parse = scan.get("parse") or {}
        # Studio (CAD) bend labels for display; the recipe's own labels are B<sequence_index>.
        labels = {b.bend_uid: b.cad_label for b in recipe.bends} if recipe is not None else {}
        return {
            "bend_display_labels": labels or None,
            "job_id": job["job_id"],
            "station_id": job.get("station_id"),
            "scan_file_name": scan.get("original_name"),
            "serial": parse.get("serial"),
            "scan_timestamp": parse.get("scan_timestamp"),
            "received_at": scan.get("received_at"),
        }

    # ------------------------------------------------------------------ zero-shot overlay (3b)
    def _start_zeroshot(self, ctx: _JobContext, ws: Any, staged: Path) -> _ZsRun | None:
        """Start the zs step in a thread next to the engine step (module docstring, 3b)."""
        if not self.config.zeroshot_module:
            return None
        out_dir = Path(ws.root) / ZEROSHOT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.config.python_exe), "-m", self.config.zeroshot_module,
            "--scan", str(staged),
            "--compiled", str(Path(ws.compiled_dir) / "compiled_part.json"),
            "--out", str(out_dir / "zs_lines.json"),
        ]  # fmt: skip
        run = _ZsRun(abort=threading.Event(), started=time.monotonic())
        log_path = self._log_path(ctx, ws, "zeroshot")

        def target() -> None:
            try:
                run.result = run_supervised(
                    cmd,
                    env=build_step_env(self.config),
                    cwd=self.config.engine_root,
                    timeout_s=self.config.zeroshot_timeout_s,
                    rss_limit_bytes=self.config.zeroshot_rss_limit_bytes,
                    poll_s=self.config.supervisor_poll_s,
                    cancel_check=lambda: run.abort.is_set() or ctx.lease_lost.is_set(),
                    log_path=log_path,
                    kill_grace_s=self.config.kill_grace_s,
                    on_spawn=self._record_step(ctx, "zeroshot", cmd, ZS_ACTIVE_STEP_REL),
                    identity_tag=ctx.job_id,
                )
            except Exception as exc:  # contained: no overlay
                run.error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                self._active_step_path(ctx.job_id, ZS_ACTIVE_STEP_REL).unlink(missing_ok=True)

        run.thread = threading.Thread(target=target, name=f"zs-{ctx.job_id}", daemon=True)
        run.thread.start()
        return run

    def _join_zeroshot(self, run: _ZsRun) -> None:
        if run.thread is None:
            return
        run.thread.join(timeout=self.config.zeroshot_timeout_s + self.config.kill_grace_s + 15.0)
        if run.thread.is_alive():  # pragma: no cover - the supervisor always returns after its hard stop
            run.abort.set()
            run.thread.join(timeout=self.config.kill_grace_s + 15.0)

    def _stop_zeroshot(self, run: _ZsRun | None) -> None:
        """Stop the zs step (the engine step failed, was cancelled or lost the lease)."""
        if run is None:
            return
        run.abort.set()
        try:
            self._join_zeroshot(run)
        except Exception:  # pragma: no cover
            self._log.exception("stopping the zero-shot step failed")

    def _finish_zeroshot(self, ctx: _JobContext, ws: Any, run: _ZsRun | None, result_path: Path) -> None:
        """Wait for the zs step, then merge the overlay. Never fails the job (only cancel / lease loss propagate)."""
        if run is None:
            return
        out_dir = Path(ws.root) / ZEROSHOT_DIR_REL
        overlay_path = out_dir / "zeroshot_overlay_v1.json"
        status: dict[str, Any] = {"zeroshot_module": self.config.zeroshot_module}
        t0 = time.monotonic()
        self._join_zeroshot(run)
        status["wait_after_engine_s"] = round(time.monotonic() - t0, 3)
        res = run.result
        if res is not None:
            ctx.peak_rss = max(ctx.peak_rss, res.peak_rss_bytes)
            status["zs"] = {"exit_code": res.exit_code, "timed_out": res.timed_out, "rss_exceeded": res.rss_exceeded,
                            "cancelled": res.cancelled, "peak_rss_bytes": res.peak_rss_bytes,
                            "wall_s": round(float(getattr(res, "duration_s", 0.0) or 0.0), 3)}
        if run.error:
            status["zs_error"] = run.error
        self._check_live(ctx)
        zs_ok = res is not None and res.exit_code == 0 and not res.timed_out and not res.rss_exceeded
        if zs_ok and (out_dir / "zs_lines.json").is_file():
            cmd = [
                str(self.config.python_exe), "-m", self.config.overlay_module,
                "--result", str(result_path),
                "--compiled", str(Path(ws.compiled_dir) / "compiled_part.json"),
                "--zs", str(out_dir / "zs_lines.json"),
                "--out", str(overlay_path),
            ]  # fmt: skip
            try:
                ores, _cap = self._run_step(
                    ctx, cmd, stage="postprocess", span=(_RENDER_PROGRESS[0], _RENDER_PROGRESS[0]),
                    timeout_s=self.config.overlay_timeout_s, rss_limit_bytes=self.config.zeroshot_rss_limit_bytes,
                    log_path=self._log_path(ctx, ws, "overlay"),
                )
                status["overlay"] = {"exit_code": ores.exit_code, "timed_out": ores.timed_out,
                                     "rss_exceeded": ores.rss_exceeded}
                if ores.exit_code != 0 or ores.timed_out or ores.rss_exceeded:
                    overlay_path.unlink(missing_ok=True)
            except (_Cancelled, _LeaseLost):
                raise
            except Exception as exc:  # contained: no overlay
                overlay_path.unlink(missing_ok=True)
                status["overlay_error"] = f"{type(exc).__name__}: {exc}"[:300]
        else:
            overlay_path.unlink(missing_ok=True)
        status["overlay_written"] = overlay_path.is_file()
        if not status["overlay_written"]:
            self._log.warning("job %s: no zero-shot overlay (%s)", ctx.job_id, json.dumps(status, sort_keys=True)[:400])
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            self._write_atomic(out_dir / "overlay_status.json", json.dumps(status, sort_keys=True, indent=1).encode("utf-8"))
        except OSError:  # pragma: no cover
            self._log.exception("job %s: could not write overlay_status.json", ctx.job_id)

    # ------------------------------------------------------------------ inspect
    def _run_inspect(self, ctx: _JobContext) -> None:
        from pydantic import ValidationError

        from contracts.production.canonical import canonical_json_bytes
        from contracts.production.compiled_part import CompiledPartV1, check_compiled_against_recipe
        from contracts.production.consistency import check_result_against_compiled
        from contracts.production.inspection_result import ResultContextV1, ShopfloorInspectionResultV1
        from contracts.production.layout import workspace_paths
        from contracts.production.line_job import LineJobV1

        job = ctx.job
        ws = workspace_paths(self._jobs_root, ctx.job_id)
        self._progress(ctx, "preparing", 1, "verifying recipe, compiled part and scan")
        resolved, recipe = self._load_recipe(ctx)

        # --- compiled part
        if resolved.compiled_dir is None or not (Path(resolved.compiled_dir) / "compiled_part.json").is_file():
            raise JobFailure("COMPILED_PART_MISSING", f"no compiled part for recipe {recipe.recipe_id}")
        src_compiled = Path(resolved.compiled_dir)
        try:
            compiled = CompiledPartV1.model_validate_json((src_compiled / "compiled_part.json").read_bytes())
        except ValidationError as exc:
            raise JobFailure("COMPILED_PART_HASH_MISMATCH", f"compiled_part.json does not validate: {_first_error(exc)}") from None
        if compiled.compiled_part_sha256 != job["compiled_part_sha256"]:
            raise JobFailure(
                "COMPILED_PART_HASH_MISMATCH",
                f"compiled part {compiled.compiled_part_sha256[:12]} != job {str(job['compiled_part_sha256'])[:12]}",
            )
        if compiled.status != "COMPILED":
            raise JobFailure("COMPILE_BLOCKED", "compile is BLOCKED: " + ", ".join(compiled.blocker_codes))
        problems = check_compiled_against_recipe(compiled, recipe)
        if problems:
            raise JobFailure("COMPILED_PART_HASH_MISMATCH", "compiled part disagrees with recipe: " + ", ".join(problems))

        # --- scan
        scan = self.store.get_scan_file(job["scan_file_id"])
        if scan is None:
            raise JobFailure("SCAN_MISSING", f"scan {job['scan_file_id']} is not registered")
        archived = self._data_root / scan["stored_rel_path"]
        if not archived.is_file() or archived.is_symlink():
            raise JobFailure("SCAN_MISSING", f"archived scan not found at {scan['stored_rel_path']}")

        # --- materialise the workspace
        self._check_live(ctx)
        self._reset_workspace(ctx, ws)
        _copy_tree_no_symlinks(src_compiled, Path(ws.compiled_dir))
        for stage in compiled.stages:
            for ref in (stage.sample_cloud, stage.display_mesh):
                target = Path(ws.compiled_dir) / ref.rel_path
                if not target.is_file() or sha256_file(target) != ref.sha256:
                    raise JobFailure("COMPILED_PART_HASH_MISMATCH", f"stage file {ref.rel_path} missing or altered")
        ext = ".stl" if str(scan.get("format") or "").startswith("STL") else ".ply"
        staged = Path(ws.scan(ext))
        shutil.copyfile(archived, staged)
        if sha256_file(staged) != job["scan_sha256"]:
            raise JobFailure("SCAN_HASH_MISMATCH", "archived scan bytes do not hash to the job's scan_sha256")
        context = ResultContextV1.model_validate(self._context_dict(job, scan, recipe))
        self._write_atomic(Path(ws.recipe), canonical_json_bytes(recipe))
        self._write_atomic(Path(ws.context), canonical_json_bytes(context))
        job_now = self.store.get_job(ctx.job_id) or job
        self._write_atomic(Path(ws.job), canonical_json_bytes(LineJobV1.model_validate(job_now)))

        # --- engine step (E3)
        # The zero-shot step (3b) runs alongside E3; it is stopped first if E3 does not yield a valid result.
        zs_run = self._start_zeroshot(ctx, ws, staged)
        try:
            cmd = [
                str(self.config.python_exe), "-m", self.config.inspect_module,
                "--compiled", str(Path(ws.compiled_dir) / "compiled_part.json"),
                "--recipe", str(ws.recipe),
                "--scan", str(staged),
                "--out", str(ws.engine_dir),
                "--scan-sha256", str(job["scan_sha256"]),
                "--compiled-sha256", str(job["compiled_part_sha256"]),
                "--recipe-sha256", str(job["recipe_sha256"]),
                "--time-budget-s", f"{min(self.config.engine_time_budget_s, self.config.engine_timeout_s - 5):g}",
            ]  # fmt: skip
            declared = job.get("declared_stage")
            if declared:
                cmd += ["--declared-stage", str(int(declared["stage_index"])), "--declared-stage-source", str(declared["source"])]
            if self.config.engine_bundle_id:
                cmd += ["--engine-bundle-id", self.config.engine_bundle_id]
            if self.config.engine_config_path is not None:
                cmd += ["--config", str(self.config.engine_config_path)]
            res, cap = self._run_step(
                ctx,
                cmd,
                stage="engine",
                span=_ENGINE_PROGRESS,
                timeout_s=self.config.engine_timeout_s,
                rss_limit_bytes=self.config.engine_rss_limit_bytes,
                log_path=self._log_path(ctx, ws, "engine"),
            )
            if res.timed_out:
                raise JobFailure("WORKER_TIMEOUT", f"engine exceeded the {self.config.engine_timeout_s:g} s hard stop")
            if res.rss_exceeded:
                raise JobFailure(
                    "WORKER_RSS_LIMIT",
                    f"engine RSS {res.peak_rss_bytes} bytes exceeded the {self.config.engine_rss_limit_bytes} byte ceiling",
                )
            if res.exit_code == 2:
                code, msg = self._read_error_file(Path(ws.engine_dir) / "engine_error.json")
                if code is None:
                    code, msg = error_message(cap.error)
                mapped = code if code in ENGINE_INPUT_ERROR_CODES else "WORKER_ERROR"
                raise JobFailure(mapped, msg or f"engine rejected its input ({code or 'no code'})")
            if res.exit_code != 0:
                # ENGINE_IO §4: exit 3 is E3's internal error; any other non-zero exit
                # is a crash unless the engine reported an ERROR record first.
                if cap.error is not None:
                    code, msg = error_message(cap.error)
                    raise JobFailure("WORKER_ERROR", f"{code + ': ' if code else ''}{msg}" or "engine reported an error")
                if res.exit_code == 3:
                    raise JobFailure("WORKER_ERROR", self._crash_message(res, cap, "engine (internal error)"))
                raise JobFailure("WORKER_CRASH", self._crash_message(res, cap, "engine"))
            if cap.result is None:
                raise JobFailure("WORKER_NO_RESULT", "engine exited 0 without a RESULT line")
            result_path = Path(ws.engine_dir) / "result.json"
            if not result_path.is_file():
                raise JobFailure("WORKER_NO_RESULT", "engine exited 0 without engine/result.json")
            try:
                result = ShopfloorInspectionResultV1.model_validate_json(result_path.read_bytes())
            except (ValidationError, ValueError) as exc:
                detail = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
                raise JobFailure("ENGINE_OUTPUT_INVALID", f"result.json does not validate: {detail}") from None
            problems = check_result_against_compiled(result, compiled, recipe)
            if problems:
                raise JobFailure("ENGINE_OUTPUT_INVALID", "result disagrees with compiled part/recipe: " + ", ".join(problems))
            if result.scan.sha256 != job["scan_sha256"]:
                raise JobFailure("ENGINE_OUTPUT_INVALID", "result.scan.sha256 is not the job's scan")
            if result.context != ResultContextV1():
                # E3 is blind to job identity (ENGINE_IO §4): it has nowhere to learn it from.
                raise JobFailure(
                    "ENGINE_OUTPUT_INVALID", "ENGINE_CONTEXT_NOT_EMPTY: the engine wrote job identity into result.context"
                )
            claimed = cap.result.get("summary_state")
            if claimed is not None and claimed != result.summary.state:
                raise JobFailure("ENGINE_OUTPUT_INVALID", f"RESULT line says {claimed} but result.json says {result.summary.state}")
        except BaseException:
            self._stop_zeroshot(zs_run)
            raise
        self._finish_zeroshot(ctx, ws, zs_run, result_path)

        # --- render step (C06)
        cmd = [
            str(self.config.python_exe), "-m", self.config.render_module,
            "--job-dir", str(ws.root),
            "--result", str(result_path),
            "--out", str(ws.result_dir),
        ]  # fmt: skip
        if self.config.engine_bundle_id:
            cmd += ["--engine-bundle-id", self.config.engine_bundle_id]
        res, cap = self._run_step(
            ctx,
            cmd,
            stage="postprocess",
            span=_RENDER_PROGRESS,
            timeout_s=self.config.render_timeout_s,
            rss_limit_bytes=self.config.render_rss_limit_bytes,
            log_path=self._log_path(ctx, ws, "render"),
        )
        if res.timed_out:
            raise JobFailure("POSTPROCESS_TIMEOUT", f"render exceeded {self.config.render_timeout_s:g} s")
        if res.rss_exceeded:
            raise JobFailure("POSTPROCESS_ERROR", f"render RSS exceeded {self.config.render_rss_limit_bytes} bytes")
        if res.exit_code != 0:
            code, msg = self._read_error_file(Path(ws.result_dir) / "postprocess_error.json")
            if code is None and not msg and cap.error is not None:
                code, msg = error_message(cap.error)
            text = f"{code + ': ' if code else ''}{msg}" if (code or msg) else self._crash_message(res, cap, "render")
            raise JobFailure("POSTPROCESS_ERROR", text)
        final_path = Path(ws.result_dir) / "shopfloor_result.json"
        if not final_path.is_file():
            raise JobFailure("POSTPROCESS_ERROR", "RESULT_MISSING: render exited 0 without result/shopfloor_result.json")
        try:
            final = ShopfloorInspectionResultV1.model_validate_json(final_path.read_bytes())
        except (ValidationError, ValueError) as exc:
            detail = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
            raise JobFailure("POSTPROCESS_ERROR", f"RESULT_INVALID: {detail}") from None
        problems = check_result_against_compiled(final, compiled, recipe)
        if problems:
            raise JobFailure("POSTPROCESS_ERROR", "RESULT_INVALID: " + ", ".join(problems))
        if not _same_verdict(result, final):
            raise JobFailure("POSTPROCESS_ERROR", "RESULT_ALTERED: the rendered result changed engine measurements")
        if final.context != context:
            if final.context != ResultContextV1():
                raise JobFailure("POSTPROCESS_ERROR", "RESULT_ALTERED: result.context is not input/context.json")
            # ENGINE_IO §5: "C06 (or the runner)" fills the job identity after the engine.
            self._log.warning("job %s: render left result.context empty; the runner fills it", ctx.job_id)
            # Surfaced in the result too: the PDF/overview C06 rendered carry no job identity.
            warnings = list(final.warnings)
            if CONTEXT_FILLED_WARNING not in warnings:
                warnings.append(CONTEXT_FILLED_WARNING)
            final = ShopfloorInspectionResultV1.model_validate(
                {**final.model_dump(mode="json"), "context": context.model_dump(mode="json"), "warnings": warnings}
            )
            self._write_atomic(final_path, canonical_json_bytes(final))

        summary = {
            "inspection_validity": final.inspection_validity.state,
            "summary_state": final.summary.state,
            "bends_total": final.summary.bends_total,
            "required_fail": final.summary.required_fail,
            "required_review": final.summary.required_review,
            "required_unknown": final.summary.required_unknown,
            "stage_status": final.stage_decision.status,
        }
        if not self.store.complete_job(ctx.job_id, ctx.owner, summary, ctx.peak_rss or None, self._duration(ctx)):
            self._log.error("job %s finished but its lease was lost; result not recorded", ctx.job_id)

    # ------------------------------------------------------------------ recipe compile
    def _run_compile(self, ctx: _JobContext) -> None:
        from pydantic import ValidationError

        from contracts.production.canonical import canonical_json_bytes
        from contracts.production.compiled_part import CompiledPartV1, check_compiled_against_recipe
        from contracts.production.layout import workspace_paths
        from contracts.production.line_job import LineJobV1

        if self.compiled_sink is None:
            raise JobFailure("INTERNAL_ERROR", "no CompiledPartSink configured for recipe_compile jobs")
        ws = workspace_paths(self._jobs_root, ctx.job_id)
        self._progress(ctx, "preparing", 1, "verifying recipe and CAD")
        resolved, recipe = self._load_recipe(ctx)
        flat: Path | None = None
        if recipe.flat is not None:
            flat = Path(resolved.flat_path) if resolved.flat_path is not None else None
            if flat is None or not flat.is_file():
                raise JobFailure("RECIPE_CAD_MISSING", "recipe flat (DXF/STEP) not found")
            if sha256_file(flat) != recipe.flat.sha256:
                raise JobFailure("RECIPE_CAD_HASH_MISMATCH", "recipe flat bytes do not hash to recipe.flat.sha256")
        self._check_live(ctx)
        self._reset_workspace(ctx, ws)
        self._write_atomic(Path(ws.recipe), canonical_json_bytes(recipe))
        job_now = self.store.get_job(ctx.job_id) or ctx.job
        self._write_atomic(Path(ws.job), canonical_json_bytes(LineJobV1.model_validate(job_now)))
        out_dir = Path(ws.root) / "compile_out"
        cmd = [
            str(self.config.python_exe), "-m", self.config.compile_module,
            "--recipe", str(ws.recipe),
            "--cad", str(resolved.cad_path),
            "--out", str(out_dir),
        ]  # fmt: skip
        if flat is not None:
            cmd += ["--flat", str(flat)]
        if self.config.compile_config_path is not None:
            cmd += ["--config", str(self.config.compile_config_path)]
        res, cap = self._run_step(
            ctx,
            cmd,
            stage="engine",
            span=_COMPILE_PROGRESS,
            timeout_s=self.config.compile_timeout_s,
            rss_limit_bytes=self.config.compile_rss_limit_bytes,
            log_path=self._log_path(ctx, ws, "compile"),
        )
        if res.timed_out:
            raise JobFailure("WORKER_TIMEOUT", f"compile exceeded {self.config.compile_timeout_s:g} s")
        if res.rss_exceeded:
            raise JobFailure("WORKER_RSS_LIMIT", f"compile RSS exceeded {self.config.compile_rss_limit_bytes} bytes")
        compiled_json = out_dir / "compiled_part.json"
        if res.exit_code == 2 and not compiled_json.is_file():
            code, msg = self._read_error_file(out_dir / "compile_error.json")
            mapped = {"CAD_HASH_MISMATCH": "RECIPE_CAD_HASH_MISMATCH", "RECIPE_INVALID": "RECIPE_SHA_MISMATCH"}.get(
                code or "", "WORKER_ERROR"
            )
            raise JobFailure(mapped, f"{code or 'INVALID_INPUT'}: {msg}")
        if res.exit_code not in (0, 2):
            if cap.error is not None:
                code, msg = error_message(cap.error)
                raise JobFailure("WORKER_ERROR", f"{code + ': ' if code else ''}{msg}")
            raise JobFailure("WORKER_CRASH", self._crash_message(res, cap, "compile"))
        if not compiled_json.is_file():
            raise JobFailure("WORKER_NO_RESULT", "compile exited 0 without compiled_part.json")
        try:
            compiled = CompiledPartV1.model_validate_json(compiled_json.read_bytes())
        except (ValidationError, ValueError) as exc:
            detail = _first_error(exc) if isinstance(exc, ValidationError) else str(exc)
            raise JobFailure("ENGINE_OUTPUT_INVALID", f"compiled_part.json does not validate: {detail}") from None
        expected_status = "COMPILED" if res.exit_code == 0 else "BLOCKED"
        if compiled.status != expected_status:
            raise JobFailure("ENGINE_OUTPUT_INVALID", f"compile exit {res.exit_code} but status {compiled.status}")
        problems = check_compiled_against_recipe(compiled, recipe)
        if problems:
            raise JobFailure("ENGINE_OUTPUT_INVALID", "compiled part disagrees with recipe: " + ", ".join(problems))
        for stage in compiled.stages:
            for ref in (stage.sample_cloud, stage.display_mesh):
                target = out_dir / ref.rel_path
                if not target.is_file() or sha256_file(target) != ref.sha256:
                    raise JobFailure("ENGINE_OUTPUT_INVALID", f"stage file {ref.rel_path} missing or not matching its sha256")
        self._check_live(ctx)
        self.compiled_sink.store_compiled(
            recipe.recipe_sha256,
            out_dir,
            status=compiled.status,
            compiled_part_sha256=compiled.compiled_part_sha256,
            blocker_codes=tuple(compiled.blocker_codes),
        )
        event = (
            "RECIPE_COMPILED" if compiled.status == "COMPILED" else "RECIPE_COMPILE_BLOCKED",
            {
                "recipe_id": recipe.recipe_id,
                "recipe_sha256": recipe.recipe_sha256,
                "compiled_part_sha256": compiled.compiled_part_sha256,
                "blocker_codes": list(compiled.blocker_codes),
            },
        )
        if not self.store.complete_job(
            ctx.job_id, ctx.owner, None, ctx.peak_rss or None, self._duration(ctx), extra_event=event
        ):
            self._log.error("compile job %s finished but its lease was lost", ctx.job_id)


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """True when ``fn`` takes a keyword argument ``name`` (or ``**kwargs``)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) or p.kind == p.VAR_KEYWORD
               for p in params)


def _first_error(exc: Any) -> str:
    try:
        errors = exc.errors()
    except Exception:
        return str(exc)[:300]
    if not errors:
        return str(exc)[:300]
    first = errors[0]
    loc = ".".join(str(x) for x in first.get("loc", ()))
    return f"{loc}: {first.get('msg', '')}"[:300] + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else "")


def _copy_tree_no_symlinks(src: Path, dst: Path) -> None:
    """Copy a compile output directory, refusing symlinks anywhere inside it."""
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        base = Path(dirpath)
        for name in dirnames + filenames:
            if (base / name).is_symlink():
                raise JobFailure("COMPILED_PART_HASH_MISMATCH", f"compiled part contains a symlink: {name}")
    shutil.copytree(src, dst, symlinks=False)


def _same_verdict(engine_result: Any, rendered: Any) -> bool:
    """True when ``rendered`` differs from ``engine_result`` only in viewer
    assets, artifacts, appended warnings and the (separately checked) context."""
    drop = {"viewer_assets", "artifacts", "warnings", "context"}
    a = engine_result.model_dump(mode="json")
    b = rendered.model_dump(mode="json")
    if {k: v for k, v in a.items() if k not in drop} != {k: v for k, v in b.items() if k not in drop}:
        return False
    return set(a.get("warnings", ())) <= set(b.get("warnings", ()))
