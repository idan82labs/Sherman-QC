"""C02 supervisor: timeout, RSS ceiling, cancel, process-group kill, protocol lines.

Fake steps are short Python scripts written into tmp_path. POSIX-only cases
are skipped on Windows.
"""
from __future__ import annotations

import os
import sys
import textwrap
import threading
import time

import pytest

from apps.production_line.runner.supervisor import run_supervised
from apps.production_line.runner.worker_protocol import error_message, format_protocol_line, parse_protocol_line, sanitize_message

posix_only = pytest.mark.skipif(os.name != "posix", reason="process groups and ps are POSIX")
ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}


def script(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def run(tmp_path, path, **kw):
    lines: list[str] = []
    defaults = dict(env=ENV, cwd=tmp_path, timeout_s=30, rss_limit_bytes=2 * 1024**3, poll_s=0.1, on_line=lines.append)
    defaults.update(kw)
    return run_supervised([sys.executable, str(path)], **defaults), lines


# ---------------------------------------------------------------- protocol
def test_protocol_parsing():
    assert parse_protocol_line('PROGRESS\t{"stage": "x", "progress": 5}') == ("PROGRESS", {"stage": "x", "progress": 5})
    assert parse_protocol_line("RESULT\t{}\n") == ("RESULT", {})
    assert parse_protocol_line("ERROR\tnot json") == (None, None)
    assert parse_protocol_line("ERROR\t[1, 2]") == (None, None)  # payload must be an object
    assert parse_protocol_line("HELLO\t{}") == (None, None)
    assert parse_protocol_line("free text") == (None, None)
    assert error_message({"code": "SCAN_MISSING", "message": "gone"}) == ("SCAN_MISSING", "gone")
    assert error_message({"error": "\x1b[31mred\x1b[0m   text\n"}) == (None, "red text")
    assert sanitize_message("a" * 5000) == "a" * 2000
    assert format_protocol_line("RESULT", {"a": 1}) == 'RESULT\t{"a": 1}'
    with pytest.raises(ValueError):
        format_protocol_line("NOPE", {})


# ---------------------------------------------------------------- outcomes
def test_success_and_lines_and_log(tmp_path):
    p = script(
        tmp_path,
        "ok.py",
        """
        import json, sys
        print("PROGRESS\\t" + json.dumps({"stage": "load", "progress": 50}), flush=True)
        print("some free text")
        print("warning on stderr", file=sys.stderr)
        print("RESULT\\t" + json.dumps({"summary_state": "INCOMPLETE"}), flush=True)
        """,
    )
    log = tmp_path / "logs" / "step.log"
    res, lines = run(tmp_path, p, log_path=log)
    assert res.exit_code == 0 and not res.killed and res.rss_guard in ("ps", "psutil", "unavailable")
    tags = [parse_protocol_line(line)[0] for line in lines]
    assert tags.count("PROGRESS") == 1 and tags.count("RESULT") == 1 and len(lines) == 4
    assert parse_protocol_line(lines[-1]) == ("RESULT", {"summary_state": "INCOMPLETE"})
    assert "warning on stderr" in log.read_text()  # stderr merged into the log


def test_error_line_and_nonzero_exit(tmp_path):
    p = script(
        tmp_path,
        "err.py",
        """
        import json, sys
        print("ERROR\\t" + json.dumps({"code": "SCAN_HASH_MISMATCH", "message": "bytes differ"}), flush=True)
        sys.exit(2)
        """,
    )
    res, lines = run(tmp_path, p)
    assert res.exit_code == 2 and not res.killed
    tag, payload = parse_protocol_line(lines[0])
    assert tag == "ERROR" and error_message(payload) == ("SCAN_HASH_MISMATCH", "bytes differ")


def test_crash_is_reported_as_exit_code(tmp_path):
    p = script(tmp_path, "crash.py", "raise SystemExit(-11 % 256)\n")
    res, _ = run(tmp_path, p)
    assert res.exit_code == 245 and not res.timed_out
    p = script(tmp_path, "boom.py", "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    res, _ = run(tmp_path, p)
    assert res.exit_code is not None and res.exit_code != 0


@posix_only
def test_timeout_kills_whole_group_including_grandchild(tmp_path):
    marker = tmp_path / "ids.txt"
    p = script(
        tmp_path,
        "sleepy.py",
        f"""
        import os, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        open({str(marker)!r}, "w").write(f"{{os.getpgid(0)}} {{child.pid}}")
        time.sleep(600)
        """,
    )
    started = time.monotonic()
    res, _ = run(tmp_path, p, timeout_s=1.5)
    elapsed = time.monotonic() - started
    assert res.timed_out and not res.rss_exceeded and not res.cancelled
    assert elapsed < 1.5 + 12  # acceptance bound (timeout + grace + reap)
    assert elapsed < 1.5 + 5  # SIGTERM reached the whole group: no wait for the SIGKILL grace
    pgid, grandchild = (int(x) for x in marker.read_text().split())
    assert pgid != os.getpgid(0)  # the step led its own process group
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@posix_only
def test_sigterm_ignoring_group_gets_sigkill(tmp_path):
    marker = tmp_path / "stubborn.pid"
    p = script(
        tmp_path,
        "stubborn.py",
        f"""
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
        child = subprocess.Popen([sys.executable, "-c", code])
        open({str(marker)!r}, "w").write(str(child.pid))
        print("ready", flush=True)
        time.sleep(600)
        """,
    )
    started = time.monotonic()
    res, _ = run(tmp_path, p, timeout_s=1.5, kill_grace_s=1.0)
    assert res.timed_out and res.exit_code == -9
    assert time.monotonic() - started < 1.5 + 1.0 + 6
    with pytest.raises(ProcessLookupError):  # the SIGTERM-ignoring grandchild got SIGKILL too
        os.kill(int(marker.read_text()), 0)


@posix_only
def test_rss_ceiling(tmp_path):
    p = script(
        tmp_path,
        "hog.py",
        """
        import time
        block = b"\\x01" * (300 * 1024 * 1024)
        print("allocated", flush=True)
        time.sleep(60)
        """,
    )
    limit = 100 * 1024 * 1024
    res, _ = run(tmp_path, p, rss_limit_bytes=limit, timeout_s=60, poll_s=0.2)
    assert res.rss_exceeded and not res.timed_out
    assert res.peak_rss_bytes > limit and res.rss_guard == "ps"
    assert res.duration_s < 30


def test_cancel(tmp_path):
    p = script(tmp_path, "wait.py", "import time\nprint('started', flush=True)\ntime.sleep(600)\n")
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    started = time.monotonic()
    res, _ = run(tmp_path, p, cancel_check=flag.is_set, timeout_s=60, poll_s=0.1)
    assert res.cancelled and not res.timed_out and time.monotonic() - started < 3.0


def test_cancel_check_exception_is_not_fatal(tmp_path):
    p = script(tmp_path, "quick.py", "import time\ntime.sleep(0.5)\n")

    def broken() -> bool:
        raise RuntimeError("store hiccup")

    res, _ = run(tmp_path, p, cancel_check=broken)
    assert res.exit_code == 0 and not res.cancelled


def test_on_line_exception_is_not_fatal(tmp_path):
    p = script(tmp_path, "talk.py", "for i in range(3): print(i, flush=True)\n")

    def bad(_line: str) -> None:
        raise ValueError("callback bug")

    res, _ = run(tmp_path, p, on_line=bad)
    assert res.exit_code == 0


def test_log_cap(tmp_path):
    p = script(tmp_path, "chatty.py", "for i in range(5000): print('x' * 100)\n")
    log = tmp_path / "chatty.log"
    res, lines = run(tmp_path, p, log_path=log, log_cap_bytes=10_000)
    assert res.exit_code == 0 and res.log_truncated
    assert log.stat().st_size < 10_200 and "log truncated" in log.read_text()
    assert len(lines) == 5000  # the callback still sees every line


def test_rejects_unsafe_limits(tmp_path):
    p = script(tmp_path, "noop.py", "pass\n")
    with pytest.raises(ValueError):
        run(tmp_path, p, rss_limit_bytes=33 * 1024**3)
    with pytest.raises(ValueError):
        run(tmp_path, p, timeout_s=0)


@posix_only
def test_orphans_of_a_finished_step_are_killed(tmp_path):
    marker = tmp_path / "orphan.pid"
    p = script(
        tmp_path,
        "leaver.py",
        f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open({str(marker)!r}, "w").write(str(child.pid))
        """,
    )
    res, _ = run(tmp_path, p)
    assert res.exit_code == 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(int(marker.read_text()), 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the orphaned grandchild survived the step")


# ---------------------------------------------------------------- review fixes (fix stage)
DRIVER = r"""
import json, sys, threading, time
from apps.production_line.runner.supervisor import run_supervised
step, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
def record(pid, pgid):
    open(out, "w").write(json.dumps({"pid": pid, "pgid": pgid}))
def go():
    run_supervised([sys.executable, step], env={"PATH": ""}, cwd="/", timeout_s=600,
                   rss_limit_bytes=2 * 1024**3, poll_s=0.1, on_spawn=record, identity_tag="tag-xyz")
if mode == "thread_exit":  # a service that runs the step in a daemon thread and exits without cleanup
    threading.Thread(target=go, daemon=True).start()
    while True:
        try:
            open(out).read()
            break
        except OSError:
            time.sleep(0.02)
    time.sleep(0.3)
    sys.exit(0)
go()
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _group_gone_within(pgid: int, seconds: float) -> bool:
    """No live (non-zombie) member within ``seconds`` (ps decides: macOS answers
    killpg(pgid, 0) with EPERM for zombies awaiting their reaper)."""
    from apps.production_line.runner.supervisor import group_members

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if group_members(pgid) == []:
            return True
        time.sleep(0.1)
    return False


@posix_only
@pytest.mark.parametrize("how", ["sigkill", "thread_exit"])
def test_step_group_dies_with_its_supervisor(tmp_path, how):
    import json
    import signal
    import subprocess

    from apps.production_line.runner.supervisor import SENTINEL_GRACE_S

    grand = tmp_path / "grand.pid"
    step = script(
        tmp_path,
        "deep.py",
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        open({str(grand)!r}, "w").write(str(child.pid))
        time.sleep(600)
        """,
    )
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    record = tmp_path / "spawn.json"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, str(driver), str(step), str(record), how], env=env)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (record.exists() and grand.exists()):
            time.sleep(0.05)
        spawn = json.loads(record.read_text())
        pgid, grandchild = spawn["pgid"], int(grand.read_text())
        assert spawn["pid"] == pgid and _alive(grandchild)
        if how == "sigkill":
            proc.send_signal(signal.SIGKILL)
        proc.wait(10)
        assert _group_gone_within(pgid, SENTINEL_GRACE_S + 6), "the step group outlived its supervisor"
        assert not _alive(grandchild)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            os.killpg(spawn["pgid"], signal.SIGKILL)  # noqa: F821 - cleanup when an assertion fired early
        except (ProcessLookupError, NameError, UnboundLocalError):
            pass


@posix_only
def test_sentinel_joins_group_with_tag_and_leaves_no_trace(tmp_path):
    from apps.production_line.runner.supervisor import group_members

    seen: dict = {}
    p = script(tmp_path, "nap.py", "import time\nprint('up', flush=True)\ntime.sleep(1.0)\n")

    def on_spawn(pid: int, pgid: int) -> None:
        seen["pid"], seen["pgid"] = pid, pgid
        seen["members"] = group_members(pgid)

    res, _ = run(tmp_path, p, on_spawn=on_spawn, identity_tag="line_" + "a" * 32)
    assert res.exit_code == 0 and seen["pid"] == seen["pgid"]
    cmds = [cmd for _pid, cmd in seen["members"]]
    assert any("line_" + "a" * 32 in c for c in cmds)  # sentinel carries the identity tag
    assert _group_gone_within(seen["pgid"], 2)  # sentinel reaped with the step: nothing left over


@posix_only
def test_fast_finish_without_sentinel_overhead(tmp_path):
    p = script(tmp_path, "noop2.py", "pass\n")
    started = time.monotonic()
    res, _ = run(tmp_path, p, kill_grace_s=10)
    assert res.exit_code == 0 and not res.killed
    assert time.monotonic() - started < 5  # stopping the sentinel never waits for the kill grace


def test_large_output_drains_fast(tmp_path):
    p = script(tmp_path, "flood.py", "import sys\nsys.stdout.write(('x' * 99 + '\\n') * 100_000)\n")  # 10 MB
    count = [0]
    started = time.monotonic()
    res, _ = run(tmp_path, p, on_line=lambda _l: count.__setitem__(0, count[0] + 1), log_path=tmp_path / "flood.log")
    elapsed = time.monotonic() - started
    assert res.exit_code == 0 and count[0] == 100_000
    assert elapsed < 2.0, f"10 MB of output took {elapsed:.2f} s to drain"


@posix_only
def test_rss_breach_is_sigkilled_without_grace(tmp_path):
    p = script(
        tmp_path,
        "stubborn_hog.py",
        """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        block = b"\\x01" * (300 * 1024 * 1024)
        print("allocated", flush=True)
        time.sleep(60)
        """,
    )
    res, _ = run(tmp_path, p, rss_limit_bytes=100 * 1024 * 1024, timeout_s=60, poll_s=0.2, kill_grace_s=20)
    assert res.rss_exceeded and res.exit_code == -9
    assert res.duration_s < 10  # no 20 s SIGTERM grace for a memory breach


@posix_only
def test_kill_recorded_group_checks_identity(tmp_path):
    import signal
    import subprocess

    from apps.production_line.runner.supervisor import kill_recorded_group

    tag = "line_" + "b" * 32
    ours = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", tag], process_group=0)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "unrelated"], process_group=0)
    try:
        assert kill_recorded_group(other.pid, tag) == "foreign"
        assert other.poll() is None  # a recycled pgid is never signalled
        assert kill_recorded_group(ours.pid, tag, grace_s=1) in ("killed", "gone")
        ours.wait(5)
        assert ours.returncode is not None
        assert kill_recorded_group(ours.pid, tag) == "gone"
        with pytest.raises(ValueError):
            kill_recorded_group(ours.pid, "")
    finally:
        for proc in (ours, other):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(5)
