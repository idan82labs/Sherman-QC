"""Readiness probes for the production line (``GET /api/line/health/ready``).

Each probe returns a :class:`ProbeResult` ``(name, ok, detail)`` and never raises;
:func:`readiness` runs a list of probes (results or zero-argument callables) and
returns ``(all_ok, results)``.  The unauthenticated HTTP body is deliberately
minimal (:func:`readiness_body`: ``{ready, checks: {name: bool}}``) so that paths and
versions are only visible on the authenticated ``/status`` route.

Probes:

- :func:`sqlite_writable` – opens an *existing* DB read-write (never creates it),
  takes the write lock and performs a rolled-back schema write.
- :func:`disk_free` – free bytes on the data-root filesystem ≥ ``min_bytes`` (10 GiB).
- :func:`heartbeat_fresh` – a component's last heartbeat is younger than ``max_age_s``.
- :func:`python_probe` – the engine interpreter can ``import numpy, scipy`` within
  ``timeout_s`` (with thread caps set to 1).
- :func:`bundle_present` – the pinned engine bundle has a readable manifest, and
  optionally verifies byte-for-byte (:func:`apps.production_line.engine_bundle.verify`).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

__all__ = [
    "ProbeResult",
    "MIN_FREE_BYTES",
    "sqlite_uri",
    "sqlite_writable",
    "disk_free",
    "heartbeat_fresh",
    "python_probe",
    "bundle_present",
    "readiness",
    "readiness_body",
]

MIN_FREE_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str = ""


def sqlite_uri(path: Path, mode: str) -> str:
    """A ``file:`` URI for ``path`` with every reserved character percent-encoded,
    so ``#``, ``?`` or ``%`` in a directory name can never truncate the path or drop
    the ``mode=`` parameter (which would let SQLite create a stray database)."""
    if mode not in ("ro", "rw", "rwc"):
        raise ValueError(f"unsupported sqlite mode {mode!r}")
    absolute = os.path.abspath(os.fspath(path))
    return "file:" + urllib.parse.quote(absolute, safe="/") + f"?mode={mode}"


def sqlite_writable(db_path: Path, name: Optional[str] = None) -> ProbeResult:
    label = name or f"sqlite:{Path(db_path).name}"
    p = Path(db_path)
    if not p.is_file():
        return ProbeResult(label, False, "database file missing")
    if not os.access(p, os.R_OK | os.W_OK):
        return ProbeResult(label, False, "not writable: file permissions")
    try:
        conn = sqlite3.connect(sqlite_uri(p, "rw"), uri=True, timeout=2, isolation_level=None)
        try:
            # SQLite silently opens a write-protected file read-only, and BEGIN
            # IMMEDIATE alone writes nothing, so attempt a real (rolled back) write.
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("CREATE TABLE IF NOT EXISTS _sherman_health_probe (x INTEGER)")
            finally:
                conn.execute("ROLLBACK")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return ProbeResult(label, False, f"not writable: {type(exc).__name__}: {exc}")
    return ProbeResult(label, True, "writable")


def disk_free(path: Path, min_bytes: int = MIN_FREE_BYTES, name: str = "disk_free") -> ProbeResult:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return ProbeResult(name, False, f"cannot stat filesystem: {type(exc).__name__}")
    ok = usage.free >= min_bytes
    return ProbeResult(name, ok, f"free={usage.free} min={min_bytes}")


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


Heartbeats = Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]


def heartbeat_fresh(
    heartbeats: Heartbeats,
    component: str,
    max_age_s: float,
    *,
    now: Optional[datetime] = None,
) -> ProbeResult:
    """``heartbeats`` is ``{component: ts}`` or rows ``{component, ts, ...}`` (the C02
    heartbeats table); ``ts`` may be ISO text, epoch seconds or a datetime."""
    name = f"heartbeat:{component}"
    ts: Any = None
    if isinstance(heartbeats, Mapping):
        ts = heartbeats.get(component)
        if isinstance(ts, Mapping):
            ts = ts.get("ts")
    else:
        for row in heartbeats:
            if row.get("component") == component:
                ts = row.get("ts")
                break
    when = _to_datetime(ts)
    if when is None:
        return ProbeResult(name, False, "no heartbeat")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age = (current - when).total_seconds()
    if age < -5:
        return ProbeResult(name, False, f"heartbeat is in the future ({age:.0f}s)")
    return ProbeResult(name, age <= max_age_s, f"age_s={max(age, 0.0):.1f} max={max_age_s}")


def python_probe(
    python_exe: Path,
    timeout_s: float = 20,
    *,
    code: str = "import numpy, scipy",
    name: str = "engine_python",
) -> ProbeResult:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "PYTHONSTARTUP"))}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [str(python_exe), "-I", "-c", code],
            capture_output=True,
            timeout=timeout_s,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(name, False, f"timed out after {timeout_s}s")
    except OSError as exc:
        return ProbeResult(name, False, f"cannot start: {type(exc).__name__}")
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        return ProbeResult(name, False, f"exit {proc.returncode}: {tail[0][:200]}")
    return ProbeResult(name, True, "imports ok")


def bundle_present(bundle_dir: Path, *, verify: bool = False, name: str = "engine_bundle") -> ProbeResult:
    manifest_path = Path(bundle_dir) / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_id = str(manifest["bundle_id"])
    except FileNotFoundError:
        return ProbeResult(name, False, "bundle manifest missing")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ProbeResult(name, False, f"bundle manifest unreadable: {type(exc).__name__}")
    if verify:
        from apps.production_line.engine_bundle import verify as verify_bundle

        problems = verify_bundle(Path(bundle_dir))
        if problems:
            return ProbeResult(name, False, f"bundle {bundle_id}: {len(problems)} mismatch(es), first: {problems[0]}")
    return ProbeResult(name, True, f"bundle_id={bundle_id}")


Probe = Union[ProbeResult, Callable[[], ProbeResult]]


def readiness(probes: Iterable[Probe]) -> tuple[bool, list[ProbeResult]]:
    """Run probes; a probe that raises becomes a failed result (fail closed)."""
    results: list[ProbeResult] = []
    for i, probe in enumerate(probes):
        if isinstance(probe, ProbeResult):
            results.append(probe)
            continue
        try:
            res = probe()
        except Exception as exc:
            res = ProbeResult(getattr(probe, "__name__", f"probe_{i}"), False, f"raised {type(exc).__name__}")
        if not isinstance(res, ProbeResult):
            res = ProbeResult(getattr(probe, "__name__", f"probe_{i}"), False, "probe returned no result")
        results.append(res)
    ok = bool(results) and all(r.ok for r in results)
    return ok, results


def readiness_body(ok: bool, results: Iterable[ProbeResult]) -> dict[str, Any]:
    """The minimal unauthenticated body ``{ready, checks: {name: bool}}``."""
    return {"ready": bool(ok), "checks": {r.name: bool(r.ok) for r in results}}
