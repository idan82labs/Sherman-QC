"""Run one step as a supervised subprocess: wall-clock timeout, RSS ceiling,
cancellation, process-group kill, capped log.

POSIX:

- ``Popen(process_group=0)``: the child leads a new process group (in the
  supervisor's session), so every grandchild it spawns can be killed together,
  and a terminal's Ctrl-C (sent to the foreground group) never reaches it.
- Parent-death sentinel: right after the spawn a tiny ``python -I -S``
  process joins the step's group, holding the read end of a pipe whose write
  end only the supervisor holds. If the supervising process dies for any
  reason (SIGKILL, a crash, a service stop whose ``finally`` never runs) the
  sentinel reads EOF and kills the whole group: SIGTERM, up to
  ``SENTINEL_GRACE_S`` for the leader to exit, then SIGKILL (itself
  included). On a normal finish the supervisor writes ``q`` and reaps it
  first. So a step can never outlive its supervisor unsupervised (no 180 s
  hard stop, no RSS ceiling). The sentinel's argv carries ``identity_tag``.
- ``on_spawn(pid, pgid)`` lets the caller record the group durably (the
  runner writes ``logs/active_step.json``); :func:`kill_recorded_group`
  kills such a recorded group after a restart, but only when a live member's
  command line contains the recorded tag (a recycled pgid is never killed).
- stdout and stderr are merged and read by a thread, line by line (buffered
  reader; each line is delivered as soon as it arrives). Each line goes to
  ``on_line`` (exceptions there are logged, never fatal) and to ``log_path``,
  capped at ``log_cap_bytes`` with one truncation note.
- RSS is the sum over ``ps -A -o pgid=,rss=`` rows whose pgid is the child's
  (KiB), sampled every ``poll_s``. The peak is reported.
- Kill: on a timeout or cancel, SIGTERM to the group, up to ``kill_grace_s``
  of grace, then SIGKILL, then reap. On an RSS breach the group gets SIGKILL
  at once (a fast allocator must not keep growing through a grace period).
  Members left behind by a child that exited normally are killed the same
  way, so no step leaves orphans.

Windows: ``CREATE_NEW_PROCESS_GROUP``; CTRL_BREAK then ``kill()``. RSS uses
``psutil`` when importable, else the guard is ``unavailable`` (logged once).

``rss_limit_bytes`` above 32 GiB is rejected (the per-step cap of ENGINE_IO-A5).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

__all__ = [
    "MAX_RSS_LIMIT_BYTES",
    "SENTINEL_GRACE_S",
    "SupervisedResult",
    "run_supervised",
    "process_group_rss_bytes",
    "group_members",
    "kill_recorded_group",
]

MAX_RSS_LIMIT_BYTES = 32 * 1024**3
#: Grace the parent-death sentinel gives the group leader after SIGTERM.
SENTINEL_GRACE_S = 2.0
_IS_WINDOWS = os.name == "nt"
_LOG = logging.getLogger(__name__)

# Runs as ``python -I -S -c _SENTINEL <pgid> <grace_s> <tag>`` inside the
# step's process group. stdin is a pipe from the supervisor: ``q`` means exit
# quietly; EOF means the supervisor is gone, so the group is killed.
_SENTINEL = r"""
import os, signal, sys, time
pgid = int(sys.argv[1]); grace = float(sys.argv[2])
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
try:
    data = os.read(0, 1)
except OSError:
    data = b""
if data == b"q":
    sys.exit(0)
try:
    os.killpg(pgid, signal.SIGTERM)
except OSError:
    sys.exit(0)
end = time.monotonic() + grace
while time.monotonic() < end:
    try:
        os.kill(pgid, 0)
    except ProcessLookupError:
        break
    except OSError:
        pass
    time.sleep(0.05)
os.killpg(pgid, signal.SIGKILL)
"""


@dataclass(frozen=True)
class SupervisedResult:
    """Outcome of :func:`run_supervised`.

    ``exit_code`` is the child's return code (negative = killed by that
    signal on POSIX). At most one of ``timed_out`` / ``rss_exceeded`` /
    ``cancelled`` is true: the first limit that fired. ``rss_guard`` is
    ``ps``, ``psutil`` or ``unavailable``.
    """

    exit_code: int | None
    timed_out: bool
    rss_exceeded: bool
    cancelled: bool
    peak_rss_bytes: int
    duration_s: float
    rss_guard: str
    log_truncated: bool = False

    @property
    def killed(self) -> bool:
        return self.timed_out or self.rss_exceeded or self.cancelled


def _ps_group_rss(pgid: int) -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pgid=,rss="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    total_kib = 0
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            if int(parts[0]) == pgid:
                total_kib += int(parts[1])
        except ValueError:
            continue
    return total_kib * 1024


def _psutil_tree_rss(pid: int) -> int | None:  # pragma: no cover - Windows only
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return None
    total = 0
    for proc in procs:
        try:
            total += int(proc.memory_info().rss)
        except psutil.Error:
            continue
    return total


def process_group_rss_bytes(pgid: int) -> int | None:
    """Total RSS of a POSIX process group, or None when ``ps`` is unavailable."""
    return _ps_group_rss(pgid)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - a member we cannot signal still exists
        return True
    return True


def _kill_group_posix(proc: subprocess.Popen[bytes], pgid: int, grace_s: float) -> None:
    """SIGTERM, ``grace_s`` of grace, SIGKILL; ``grace_s <= 0`` sends SIGKILL at once."""
    if grace_s > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if proc.poll() is not None and not _group_alive(pgid):
                break
            time.sleep(0.05)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL cannot be ignored
        _LOG.error("child %s did not exit after SIGKILL", proc.pid)
    # Reparented grandchildren are reaped by init/launchd; wait briefly for the group to vanish.
    end = time.monotonic() + 5.0
    while _group_alive(pgid) and time.monotonic() < end:
        time.sleep(0.05)


def group_members(pgid: int) -> list[tuple[int, str]] | None:
    """``[(pid, command line)]`` of the live (non-zombie) members of a POSIX
    process group (``ps -A -ww -o pid=,pgid=,stat=,command=``), or None when
    ``ps`` fails."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-ww", "-o", "pid=,pgid=,stat=,command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    members: list[tuple[int, str]] = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, group = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if group == pgid and not parts[2].startswith("Z"):
            members.append((pid, parts[3] if len(parts) > 3 else ""))
    return members


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass  # gone, or only zombies left (macOS answers EPERM for those)


def _members_gone(pgid: int, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while True:
        members = group_members(pgid)
        if members is not None and not members:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.1)


def kill_recorded_group(pgid: int, tag: str, *, grace_s: float = 3.0, wait_s: float = 10.0) -> str:
    """Kill a step group recorded before a restart (``on_spawn``), if it is still ours.

    Liveness comes from ``ps`` (zombies do not count). Returns ``gone`` (no
    live member), ``killed`` (it was alive, a member's command line carried
    ``tag``, and it is now gone), ``foreign`` (live members, none carrying
    ``tag``: the pgid was recycled, nothing is signalled), ``alive`` (still
    running after SIGKILL and ``wait_s``, or ``ps`` failed so nothing could
    be checked) or ``unsupported`` (Windows: no process groups to look up).
    """
    if _IS_WINDOWS:  # pragma: no cover
        return "unsupported"
    if pgid <= 1 or not tag:
        raise ValueError("a recorded pgid > 1 and a non-empty tag are required")
    if pgid == os.getpgrp():
        return "foreign"  # never signal our own group
    if not _group_alive(pgid):
        return "gone"
    members = group_members(pgid)
    if members is None:
        _LOG.error("cannot list process group %s; not killing it blind", pgid)
        return "alive"
    if not members:
        return "gone"
    if not any(tag in cmd for _pid, cmd in members):
        _LOG.warning("process group %s no longer carries tag %s (pgid reused); left alone", pgid, tag)
        return "foreign"
    _LOG.warning("killing orphaned step group %s (%d member(s), tag %s)", pgid, len(members), tag)
    _signal_group(pgid, signal.SIGTERM)
    if not _members_gone(pgid, max(0.0, grace_s)):
        _signal_group(pgid, signal.SIGKILL)
    return "killed" if _members_gone(pgid, wait_s) else "alive"


def _start_sentinel(pgid: int, tag: str | None) -> subprocess.Popen[bytes] | None:
    """Spawn the parent-death sentinel into group ``pgid`` (None when that fails)."""
    exe = sys.executable
    if not exe:
        _LOG.warning("no sys.executable: the parent-death sentinel is not available")
        return None
    try:
        return subprocess.Popen(
            [exe, "-I", "-S", "-c", _SENTINEL, str(pgid), f"{SENTINEL_GRACE_S:g}", tag or "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": os.environ.get("PATH", "")},
            cwd="/",
            process_group=pgid,
        )
    except Exception as exc:  # the step already exited (group gone) or spawn failed
        _LOG.debug("parent-death sentinel not started for group %s: %s", pgid, exc)
        return None


def _stop_sentinel(sentinel: subprocess.Popen[bytes] | None) -> None:
    if sentinel is None:
        return
    try:
        assert sentinel.stdin is not None
        sentinel.stdin.write(b"q")
        sentinel.stdin.close()
    except (OSError, ValueError):
        pass  # already gone (killed with its group)
    try:
        sentinel.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - it exits on "q" at once
        sentinel.kill()
        sentinel.wait(timeout=5)


def _kill_windows(proc: subprocess.Popen[bytes], grace_s: float) -> None:  # pragma: no cover - Windows only
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


class _LogSink:
    def __init__(self, path: Path | None, cap_bytes: int) -> None:
        self._fh: IO[bytes] | None = None
        self._cap = int(cap_bytes)
        self._written = 0
        self.truncated = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "ab")

    def write(self, raw: bytes) -> None:
        if self._fh is None or self.truncated:
            return
        if self._written + len(raw) > self._cap:
            self._fh.write(b"\n[supervisor] log truncated at cap\n")
            self.truncated = True
            return
        self._fh.write(raw)
        self._written += len(raw)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()


def run_supervised(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    env: Mapping[str, str],
    cwd: str | os.PathLike[str],
    timeout_s: float,
    rss_limit_bytes: int,
    poll_s: float = 1.0,
    cancel_check: Callable[[], bool] | None = None,
    on_line: Callable[[str], Any] | None = None,
    log_path: str | os.PathLike[str] | None = None,
    log_cap_bytes: int = 20 * 1024**2,
    kill_grace_s: float = 10.0,
    on_spawn: Callable[[int, int], Any] | None = None,
    identity_tag: str | None = None,
) -> SupervisedResult:
    """Run ``cmd`` under supervision (module docstring) and return the outcome.

    ``on_spawn(pid, pgid)`` is called once the step and its sentinel run
    (exceptions are logged, never fatal); ``identity_tag`` goes into the
    sentinel's argv so :func:`kill_recorded_group` can recognise the group.
    """
    if rss_limit_bytes <= 0 or rss_limit_bytes > MAX_RSS_LIMIT_BYTES:
        raise ValueError("rss_limit_bytes must be in (0, 32 GiB]")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if poll_s <= 0:
        raise ValueError("poll_s must be > 0")
    sink = _LogSink(Path(log_path) if log_path is not None else None, log_cap_bytes)
    started = time.monotonic()
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": dict(env),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if _IS_WINDOWS:  # pragma: no cover
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["process_group"] = 0
    try:
        proc = subprocess.Popen([str(c) for c in cmd], **kwargs)
    except BaseException:
        sink.close()
        raise
    pgid = proc.pid  # group leader: pgid == pid
    sentinel = None if _IS_WINDOWS else _start_sentinel(pgid, identity_tag)
    if on_spawn is not None:
        try:
            on_spawn(proc.pid, pgid)
        except Exception:
            _LOG.exception("on_spawn callback failed")

    def _reader() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            sink.write(raw)
            if on_line is not None:
                try:
                    on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
                except Exception:
                    _LOG.exception("on_line callback failed")
        proc.stdout.close()

    reader = threading.Thread(target=_reader, name=f"supervised-{proc.pid}-stdout", daemon=True)
    reader.start()

    if _IS_WINDOWS:  # pragma: no cover
        guard = "psutil"
        try:
            import psutil  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            guard = "unavailable"
            _LOG.warning("psutil is not installed: the RSS ceiling is not enforced on Windows")
    else:
        guard = "ps"

    peak = 0
    timed_out = rss_exceeded = cancelled = False
    warned_ps = False
    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_s:
                timed_out = True
                break
            if guard != "unavailable":
                rss = _psutil_tree_rss(proc.pid) if _IS_WINDOWS else _ps_group_rss(pgid)
                if rss is None:
                    if not warned_ps:
                        _LOG.warning("RSS sampling failed for pid %s; ceiling not enforced this sample", proc.pid)
                        warned_ps = True
                else:
                    peak = max(peak, rss)
                    if rss > rss_limit_bytes:
                        rss_exceeded = True
                        break
            if cancel_check is not None:
                try:
                    if cancel_check():
                        cancelled = True
                        break
                except Exception:
                    _LOG.exception("cancel_check failed; continuing")
            if proc.poll() is not None:
                break
            time.sleep(min(poll_s, max(0.01, timeout_s - (time.monotonic() - started))))
    finally:
        if _IS_WINDOWS:  # pragma: no cover
            if proc.poll() is None:
                _kill_windows(proc, 0.0 if rss_exceeded else kill_grace_s)
        else:
            _stop_sentinel(sentinel)  # first, so only real step members keep the group alive
            if proc.poll() is None or _group_alive(pgid):
                _kill_group_posix(proc, pgid, 0.0 if rss_exceeded else kill_grace_s)
        try:
            exit_code: int | None = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            exit_code = None
        reader.join(timeout=10)
        sink.close()
    return SupervisedResult(
        exit_code=exit_code,
        timed_out=timed_out,
        rss_exceeded=rss_exceeded,
        cancelled=cancelled,
        peak_rss_bytes=int(peak),
        duration_s=round(time.monotonic() - started, 3),
        rss_guard=guard,
        log_truncated=sink.truncated,
    )
