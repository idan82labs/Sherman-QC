"""Single bounded source-only V4 custody qualification, never a Gold runner.

Reuses the separately reviewed V3 supervisor mechanism in an additive harness.
Only the named AST-extracted invented-file tests may run. No physical paths.
"""
import hashlib
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time


WALL_SECONDS = 60
WORK_SECONDS = 55
RSS_BYTES = 1024**3
OUTPUT_BYTES = 1024**2


def _remaining(deadline, local_cap):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("NATIVE_TEST_ABSOLUTE_DEADLINE")
    return min(local_cap, value)


def _group_sample(pgid, deadline):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="],
        capture_output=True, text=True, timeout=_remaining(deadline, 2))
    rows = [line.split() for line in result.stdout.splitlines()]
    if result.returncode != 0 or not rows or any(
        len(row) != 3 or not all(value.isdigit() for value in row) for row in rows):
        raise RuntimeError("NATIVE_TEST_PROCESS_TABLE_INVALID")
    if len({int(row[0]) for row in rows}) != len(rows):
        raise RuntimeError("NATIVE_TEST_PROCESS_TABLE_DUPLICATE_PID")
    return [(int(pid), int(rss)*1024) for pid, group, rss in rows if int(group) == pgid]


def supervise_native_context_test_v3(command, *, cwd, env, retained):
    """Run exactly one worker; preserve cache and output even on failed custody."""
    retained = Path(retained)
    cache = tempfile.mkdtemp(prefix="native-context-v3-tests-")
    child_env = dict(env, HOME=cache, TMPDIR=cache, XDG_CACHE_HOME=cache)
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    work_deadline = started + WORK_SECONDS
    child = selector = None
    stopped = primary_error = cleanup_error = None
    child_code = None
    peak = forwarded = 0
    empty_confirmed = False
    observed_group_pids = set()
    phase_times = {}
    output_path = retained / "OUTPUT.txt"
    with output_path.open("xb") as output:
        try:
            child = subprocess.Popen(command, cwd=cwd, env=child_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() >= work_deadline:
                    stopped = "WORK_WALL_LIMIT"
                    break
                for key, _ in selector.select(timeout=_remaining(work_deadline, .02)):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max(0, OUTPUT_BYTES-forwarded)
                    output.write(block[:remaining])
                    output.flush()
                    forwarded += len(block)
                group = _group_sample(child.pid, work_deadline)
                observed_group_pids.update(pid for pid, _ in group)
                peak = max(peak, sum(rss for _, rss in group))
                child_code = child.poll()
                if child_code is None and child.pid not in {pid for pid, _ in group}:
                    raise RuntimeError("NATIVE_TEST_ACTIVE_CHILD_MISSING")
                if any(pid != child.pid for pid, _ in group):
                    stopped = "UNEXPECTED_NATIVE_TEST_CHILD_PROCESS"
                elif peak > RSS_BYTES:
                    stopped = "SAMPLED_RSS_LIMIT"
                elif forwarded > OUTPUT_BYTES:
                    stopped = "OUTPUT_LIMIT"
                elif time.monotonic() >= work_deadline:
                    stopped = "WORK_WALL_LIMIT"
                if stopped or child_code is not None and not selector.get_map():
                    break
        except Exception as error:
            stopped = stopped or ("WORK_WALL_LIMIT" if time.monotonic() >= work_deadline
                                  else "NATIVE_TEST_SUPERVISION_FAILURE")
            primary_error = type(error).__name__ + ":" + str(error)[:256]
        finally:
            phase_times["work_stopped_seconds"] = time.monotonic()-started
            if child is not None:
                try:
                    # Signal only the group belonging to the child created here.
                    # This is bounded lifecycle custody, not protection against
                    # arbitrary external process-identity interference.
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child_code = child.wait(timeout=_remaining(deadline, 5))
                    phase_times["reaped_seconds"] = time.monotonic()-started
                    group = _group_sample(child.pid, deadline)
                    if group and all(pid in observed_group_pids for pid, _ in group):
                        os.killpg(child.pid, signal.SIGKILL)
                        while group:
                            time.sleep(_remaining(deadline, .02))
                            group = _group_sample(child.pid, deadline)
                    if group:
                        # A surviving group is an unexpected contract failure;
                        # retain its observed identities, never claim cleanup.
                        cleanup_error = "NATIVE_TEST_GROUP_NOT_EMPTY:" + ",".join(str(pid) for pid, _ in group)[:256]
                    else:
                        empty_confirmed = True
                    phase_times["group_checked_seconds"] = time.monotonic()-started
                except Exception as error:
                    cleanup_error = type(error).__name__ + ":" + str(error)[:256]
            if selector is not None:
                selector.close()
            if child is not None and child.stdout is not None:
                child.stdout.close()
    elapsed = time.monotonic()-started
    if elapsed >= WALL_SECONDS:
        stopped = stopped or "TOTAL_WALL_LIMIT"
    if not empty_confirmed:
        stopped = stopped or "NATIVE_TEST_GROUP_CUSTODY_UNCONFIRMED"
    phase_times["terminal_decision_seconds"] = elapsed
    return dict(exit_code=child_code, parent_wall_seconds=elapsed,
        sampled_group_peak_rss_bytes=peak, budget_stop=stopped,
        primary_error=primary_error, cleanup_error=cleanup_error,
        process_group_confirmed_empty=empty_confirmed,
        observed_group_pids=sorted(observed_group_pids),
        requires_group_attention=not empty_confirmed and child is not None,
        private_cache_retained_for_custody=cache,
        phase_times=phase_times, output_bytes_observed=forwarded,
        output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        limits=dict(wall_seconds=WALL_SECONDS, work_wall_seconds=WORK_SECONDS,
            worker_cpu_seconds=45, sampled_group_rss_bytes=RSS_BYTES,
            output_bytes=OUTPUT_BYTES),
        enforcement_scope="SAMPLED_WORK_AND_ABSOLUTE_CLEANUP_DEADLINES_NOT_HARD_REAL_TIME_OS_IO")


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/check_gold11_mesh_v4_custody.py",
    "scripts/run_gold11_mesh_development_v4.py",
    "tests/test_gold11_mesh_v4_terminal_custody.py",
    "scripts/gold11_terminal_artifacts_v4.py",
    "scripts/run_gold11_mesh_development_v3.py",
    "docs/research/2026-09-10_gold11_v4_custody_gate_protocol.md",
)


def _sources():
    result = {}
    for name in SOURCE_FILES:
        blob = (ROOT/name).read_bytes()
        if len(blob) > 2*1024**2:
            raise ValueError("SOURCE_FILE_BUDGET")
        result[name] = hashlib.sha256(blob).hexdigest()
    return result


def main():
    import argparse
    import json
    import resource
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
        sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
        import pytest
        code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
            "-q", "-s", str(ROOT/"tests/test_gold11_mesh_v4_terminal_custody.py")])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps(dict(worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
                              worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)
        return int(code)
    before = _sources()
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    base.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="gold11_v4_custody-", dir=base))
    with (retained/"STARTED.json").open("x") as output:
        json.dump(dict(status="STARTED_SYNTHETIC_SOURCE_GATE_NO_OVERWRITE",
            source_sha256_before=before, expected_test_outcomes=22,
            limits=dict(wall_seconds=60, work_seconds=55, worker_cpu_seconds=45,
                        sampled_group_rss_bytes=1024**3, output_bytes=1024**2)), output, sort_keys=True)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    receipt = supervise_native_context_test_v3(
        [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()), "--worker"],
        cwd=ROOT, env=env, retained=retained)
    receipt.update(source_sha256_after=_sources(), retained_attempt_directory=str(retained),
                   evidence_scope="INVENTED_FILES_FAKE_TIMER_NOT_FULL_RUNNER_QUALIFICATION")
    receipt["source_before_after_consistent"] = receipt["source_sha256_after"] == before
    with (retained/"COMPLETED.json").open("x") as output:
        json.dump(receipt, output, sort_keys=True, allow_nan=False)
    print((retained/"OUTPUT.txt").read_text(), end="")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if (receipt["exit_code"] == 0 and not receipt["budget_stop"]
        and receipt["process_group_confirmed_empty"] and not receipt["cleanup_error"]
        and receipt["source_before_after_consistent"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
