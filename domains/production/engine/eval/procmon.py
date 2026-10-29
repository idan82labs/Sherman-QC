"""Run one subprocess under wall-time and RSS limits and measure both (POSIX).

The tiered runner drives E1 and E3 through their command lines, serially, one
heavy process at a time (plan §7). For each run this module records:

- ``wall_s``: monotonic wall time from spawn to exit;
- ``peak_rss_bytes``: the larger of (a) the peak of the process group's summed
  RSS, sampled with ``ps -A -o pgid=,rss=`` (the same probe the C02 supervisor
  uses, so threads and helper children count), and (b) the leader's exact
  ``ru_maxrss`` from ``os.wait4`` (catches spikes between samples);
- the stdout protocol records (``PROGRESS|RESULT|ERROR\\t{json}``) and the exit
  status. Stderr goes to a log file.

Limits: past ``timeout_s`` the group gets SIGTERM, then SIGKILL after
``kill_grace_s`` (``timed_out``); a sampled RSS above ``rss_limit_bytes`` is
killed at once (``rss_exceeded``). The child runs in its own session/process
group so the kill reaches every descendant. No ``psutil`` dependency.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProcessOutcome", "run_monitored", "group_rss_bytes", "maxrss_to_bytes"]


@dataclass(frozen=True)
class ProcessOutcome:
    argv: tuple[str, ...]
    returncode: int | None
    #: Terminating signal number when the process was killed by a signal.
    signal: int | None
    wall_s: float
    peak_rss_bytes: int | None
    sampled_peak_rss_bytes: int | None
    wait4_maxrss_bytes: int | None
    timed_out: bool
    rss_exceeded: bool
    records: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    stdout_tail: str = ""
    samples: int = 0

    def last(self, kind: str) -> Any:
        for k, payload in reversed(self.records):
            if k == kind:
                return payload
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv), "returncode": self.returncode, "signal": self.signal,
            "wall_s": round(self.wall_s, 3), "peak_rss_bytes": self.peak_rss_bytes,
            "sampled_peak_rss_bytes": self.sampled_peak_rss_bytes, "wait4_maxrss_bytes": self.wait4_maxrss_bytes,
            "timed_out": self.timed_out, "rss_exceeded": self.rss_exceeded, "samples": self.samples,
            "result_record": self.last("RESULT"), "error_record": self.last("ERROR"),
        }


def maxrss_to_bytes(ru_maxrss: int) -> int:
    """``ru_maxrss`` is bytes on macOS and KiB on Linux."""
    return int(ru_maxrss) if sys.platform == "darwin" else int(ru_maxrss) * 1024


def group_rss_bytes(pgid: int) -> int | None:
    """Summed RSS of every process in ``pgid`` (``ps`` probe), or None when ``ps`` fails."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=5.0,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    found = False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            g, rss = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if g == pgid:
            total += rss * 1024
            found = True
    return total if found else 0


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_monitored(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float,
    rss_limit_bytes: int,
    kill_grace_s: float = 3.0,
    sample_interval_s: float = 0.25,
    stderr_path: str | os.PathLike | None = None,
    stdout_path: str | os.PathLike | None = None,
) -> ProcessOutcome:
    """Spawn ``argv`` and wait for it under the limits (module docstring)."""
    if os.name != "posix":  # pragma: no cover - the runner is used on macOS/Linux only
        raise RuntimeError("run_monitored needs a POSIX host")
    err_fh = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL  # noqa: SIM115
    out_fh = open(stdout_path, "w", encoding="utf-8") if stdout_path else None  # noqa: SIM115
    records: list[tuple[str, Any]] = []
    tail: list[str] = []
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(argv), cwd=None if cwd is None else str(cwd), env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err_fh, start_new_session=True,
        )
    except BaseException:
        if stderr_path:
            err_fh.close()  # type: ignore[union-attr]
        if out_fh:
            out_fh.close()
        raise
    pgid = proc.pid  # start_new_session: the child leads its own group

    def reader() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if out_fh:
                out_fh.write(line + "\n")
            tail.append(line)
            del tail[:-50]
            kind, sep, payload = line.partition("\t")
            if sep and kind in ("PROGRESS", "RESULT", "ERROR"):
                try:
                    records.append((kind, json.loads(payload)))
                except ValueError:
                    records.append((kind, {"unparsed": payload[:500]}))
        proc.stdout.close()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    peak_sampled: int | None = None
    samples = 0
    timed_out = rss_exceeded = False
    status = None
    rusage = None
    next_sample = 0.0
    term_at: float | None = None
    try:
        while True:
            pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid == proc.pid:
                status, rusage = st, ru
                break
            now = time.monotonic()
            if now >= next_sample:
                rss = group_rss_bytes(pgid)
                samples += 1
                next_sample = now + sample_interval_s
                if rss is not None:
                    peak_sampled = rss if peak_sampled is None else max(peak_sampled, rss)
                    if rss > rss_limit_bytes and not rss_exceeded and not timed_out:
                        rss_exceeded = True
                        _signal_group(pgid, signal.SIGKILL)
            if not timed_out and not rss_exceeded and now - t0 > timeout_s:
                timed_out = True
                term_at = now
                _signal_group(pgid, signal.SIGTERM)
            if term_at is not None and now - term_at > kill_grace_s:
                _signal_group(pgid, signal.SIGKILL)
                term_at = None
            time.sleep(0.02)
    finally:
        if status is None:  # interrupted: never leave the group running
            _signal_group(pgid, signal.SIGKILL)
            _, status, rusage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        _signal_group(pgid, signal.SIGKILL)  # stray grandchildren of a finished leader
        th.join(timeout=5.0)
        if stderr_path:
            err_fh.close()  # type: ignore[union-attr]
        if out_fh:
            out_fh.close()
    wall = time.monotonic() - t0
    code = proc.returncode
    sig = -code if code is not None and code < 0 else None
    w4 = maxrss_to_bytes(rusage.ru_maxrss) if rusage is not None else None
    peaks = [v for v in (peak_sampled, w4) if v is not None]
    return ProcessOutcome(
        argv=tuple(argv), returncode=None if sig is not None else code, signal=sig, wall_s=wall,
        peak_rss_bytes=max(peaks) if peaks else None, sampled_peak_rss_bytes=peak_sampled, wait4_maxrss_bytes=w4,
        timed_out=timed_out, rss_exceeded=rss_exceeded, records=tuple(records), stdout_tail="\n".join(tail[-20:]),
        samples=samples,
    )
