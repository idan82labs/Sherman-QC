"""Line service of the Sherman QC production line (C11): hotfolders + job runner.

Usage (from an app copy, ``cwd=<copy>``, ``PYTHONPATH=<copy>:<copy>/backend``)::

    python -m apps.production_line.service_main [--build-bundle] [--run-for SECONDS]

Startup (every refusal prints ``[service] <CODE>: ...`` on stderr and exits 2):

1. :func:`~apps.production_line.settings.load_settings`; a disabled
   (``LINE_DISABLED``) or invalid configuration exits 2 with its code.
2. :func:`~apps.production_line.settings.ensure_layout` and C04 JSON logging.
3. The engine bundle at ``settings.engine_bundle_dir``:

   - ``--build-bundle``: build it from the app root with the C04 builder
     (``--profile engine``), atomically replacing an existing bundle, then verify;
   - otherwise it must exist (``ENGINE_BUNDLE_MISSING``) and verify byte for
     byte (``ENGINE_BUNDLE_INVALID``, the first mismatches are printed).

   The runner executes every step from that bundle (cwd and ``PYTHONPATH``), so
   the dev repo's mutable research state is never on the line path.
4. Every station inbox must be an existing directory (``STATION_INBOX_MISSING``):
   a missing SMB mount is never silently replaced by an empty local folder.
5. :func:`~apps.production_line.wiring.build_runtime` with ``workers=True``: one
   :class:`~apps.production_line.runner.runner.JobRunner` (instance id
   ``runner@<host>``; its constructor recovers expired leases and reclaims at
   once the leases of its own dead predecessor) and one hotfolder poller per
   station, each on its own thread, plus a ``service`` heartbeat.

SIGTERM / SIGINT set the stop event. Pollers stop after their current poll and
the runner after its current job; the process waits at most
``SHUTDOWN_GRACE_S`` (30 s) for them. A job still running then is left to lease
recovery: its step dies with this process (the C02 parent-death sentinel) and
the restarted service re-queues it (``JOB_RECOVERED``), so a kill mid-job never
loses the scan. ``--run-for`` stops the service after that many seconds (tests).

Exit codes: 0 clean stop (every worker thread finished within the grace);
2 startup refusal; 3 the grace elapsed with a worker still busy (the job it
held is left to lease recovery), so a supervisor can tell the two apart.

The service never forks while a step runs (C02 sentinel requirement): it uses
threads only.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

__all__ = ["EXIT_GRACE_ELAPSED", "SHUTDOWN_GRACE_S", "SERVICE_HEARTBEAT_S", "ServiceError", "prepare_engine_bundle", "run_service", "main"]

SHUTDOWN_GRACE_S = 30.0
#: Exit code when the shutdown grace elapsed with a worker still running.
EXIT_GRACE_ELAPSED = 3
SERVICE_HEARTBEAT_S = 15.0
_LOG = logging.getLogger("apps.production_line.service")


class ServiceError(Exception):
    """A startup refusal with a stable code (printed, exit 2)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def prepare_engine_bundle(settings: Any, *, build: bool) -> str:
    """Build (``build=True``) and/or verify the engine bundle; return its bundle id.

    Raises :class:`ServiceError` ``ENGINE_BUNDLE_BUILD_FAILED``,
    ``ENGINE_BUNDLE_MISSING`` or ``ENGINE_BUNDLE_INVALID``.
    """
    from apps.production_line.engine_bundle import BundleError, build_bundle, read_manifest, verify

    dest = Path(settings.engine_bundle_dir)
    if build:
        try:
            dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            result = build_bundle(Path(settings.repo_root), dest, profile="engine", force=True)
        except BundleError as exc:
            raise ServiceError("ENGINE_BUNDLE_BUILD_FAILED", str(exc)) from None
        for warning in getattr(result, "warnings", ()) or ():
            _LOG.warning("engine bundle: %s", warning)
    if not (dest / "bundle_manifest.json").is_file():
        raise ServiceError("ENGINE_BUNDLE_MISSING", f"no engine bundle at {dest}; start with --build-bundle")
    problems = verify(dest)
    if problems:
        raise ServiceError("ENGINE_BUNDLE_INVALID", "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))
    return str(read_manifest(dest).get("bundle_id") or "")


def _check_inboxes(settings: Any) -> None:
    for st in settings.stations:
        if not Path(st.inbox).is_dir():
            raise ServiceError("STATION_INBOX_MISSING", f"station {st.station_id}: inbox {st.inbox} is not a directory")


def _thread(name: str, target: Any, *args: Any) -> threading.Thread:
    def run() -> None:
        try:
            target(*args)
        except Exception:  # pragma: no cover - the loops contain their own errors
            _LOG.exception("%s stopped with an error", name)

    t = threading.Thread(target=run, name=name, daemon=True)
    t.start()
    return t


def run_service(runtime: Any, stop: threading.Event, *, heartbeat_s: float = SERVICE_HEARTBEAT_S,
                grace_s: float = SHUTDOWN_GRACE_S, bundle_id: str | None = None) -> bool:
    """Run the runtime's runner and pollers until ``stop`` is set; return True when
    every worker thread stopped within ``grace_s``."""
    threads = [_thread(f"hotfolder-{p.config.station_id}", p.run_forever, stop) for p in runtime.pollers]
    if runtime.runner is not None:
        threads.append(_thread("runner", runtime.runner.run_forever, stop))
    instance = f"service@{os.getpid()}"
    while not stop.is_set():
        try:
            runtime.line_store.put_heartbeat(
                "service", instance,
                {"pid": os.getpid(), "stations": [p.config.station_id for p in runtime.pollers],
                 "engine_bundle_id": bundle_id},
            )
        except Exception:
            _LOG.exception("service heartbeat failed")
        stop.wait(heartbeat_s)
    deadline = time.monotonic() + grace_s
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in threads if t.is_alive()]
    if alive:
        _LOG.warning("shutdown grace of %.0f s elapsed with %s still running; leases recover on restart",
                     grace_s, ", ".join(alive))
    return not alive


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apps.production_line.service_main",
                                description="Sherman QC line service: station hotfolders + supervised job runner")
    p.add_argument("--build-bundle", action="store_true",
                   help="build the engine bundle from the app root into SHERMAN_LINE_ENGINE_BUNDLE_DIR first")
    p.add_argument("--run-for", type=float, default=None, help="stop after this many seconds (tests)")
    p.add_argument("--log-level", default="INFO")
    return p


def _err(text: str) -> None:
    print(f"[service] {text}", file=sys.stderr, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from apps.production_line.settings import SettingsError, ensure_layout, load_settings

    try:
        settings = load_settings()
    except SettingsError as exc:
        _err(f"{exc.code}: {exc.message}")
        return 2
    try:
        ensure_layout(settings.paths)
        from apps.production_line.telemetry import DEFAULT_LOGGERS, configure_logging

        configure_logging(settings.paths, args.log_level, settings.log_retention_days,
                          logger_names=tuple(DEFAULT_LOGGERS))
        bundle_id = prepare_engine_bundle(settings, build=args.build_bundle)
        _check_inboxes(settings)
    except ServiceError as exc:
        _err(f"{exc.code}: {exc.message}")
        return 2

    from apps.production_line.wiring import build_runtime, service_catalog_port

    runtime = build_runtime(settings, workers=True, catalog=service_catalog_port(settings))
    stop = threading.Event()

    def on_signal(signum: int, _frame: Any) -> None:
        _LOG.info("signal %s: stopping (grace %.0f s)", signum, SHUTDOWN_GRACE_S)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if args.run_for is not None:
        timer = threading.Timer(max(0.0, args.run_for), stop.set)
        timer.daemon = True
        timer.start()
    _LOG.info("line service started: bundle %s, stations %s", bundle_id,
              ",".join(st.station_id for st in settings.stations) or "-")
    print(f"[service] READY bundle={bundle_id} stations={','.join(settings.station_ids)}", file=sys.stderr, flush=True)
    clean = run_service(runtime, stop, bundle_id=bundle_id)
    runtime.close()
    if clean:
        _LOG.info("line service stopped (clean)")
        return 0
    _LOG.warning("line service stopped: grace elapsed with a worker still busy; its job is left to lease recovery")
    _err("GRACE_ELAPSED: a worker was still busy after the shutdown grace; its job is left to lease recovery")
    return EXIT_GRACE_ELAPSED


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
