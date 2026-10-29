"""Prospective eight-case macOS owned-child qualification; exact PRE required.

No physical inputs, pytest, application initialization or unrelated PID targets.
All spawned payloads self-expire; the qualified candidate alone signals freshly
authenticated owned generations. This is not hard native-I/O containment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).absolute().parents[1]
SELF = "scripts/check_owned_process_supervision_v2_host.py"
SUT = "scripts/gold11_owned_process_supervision_v2.py"
POST = "docs/research/2026-09-12_owned_process_supervision_v2_source_post_run_audit.md"
PROTOCOL = "docs/research/2026-09-12_owned_process_supervision_v2_host_protocol.md"
PRE = "docs/research/2026-09-12_owned_process_supervision_v2_host_pre_run_audit.md"
SOURCES = (SELF, SUT, POST, PROTOCOL)
PYTHON = ROOT/".venv/bin/python"
SUT_SHA = "649815cad6dc10d70b34cf9ae079fe77146e371ee12d1bcc4b7ea64e340492e5"
PRE_MARKER = "OWNED_PROCESS_SUPERVISION_V2_HOST_PRE_JSON="
ADMIN_BYTES, LOG_BYTES = 256*1024, 16*1024
RSS_BYTES, PARENT_CPU, PAYLOAD_CPU = 512*1024**2, 10, 2
WORK_WALL, PAYLOAD_ALARM = 55., 4
CASES = (
    ("outer_eof", "eof", True, 2., 0, None),
    ("outer_nonzero", "nonzero", True, 2., 7, None),
    ("outer_deadline", "deadline", True, .3, -9, "WALL_LIMIT"),
    ("outer_exec", "exec_before", True, 2., 0, None),
    ("outer_waited_descendant", "waited_parent", True, 2., 0, None),
    ("outer_surviving_descendant", "exiting_parent", True, 2., 0,
     "OWNED_PROCESSES_REMAIN_AFTER_DIRECT_CHILD_EXIT"),
    ("inner_eof", "eof", False, 2., 0, None),
    ("inner_deadline", "deadline", False, .3, -9, "WALL_LIMIT"),
)
LIMITS = dict(parent_cpu_seconds=PARENT_CPU, payload_cpu_seconds=PAYLOAD_CPU,
    payload_self_alarm_seconds=PAYLOAD_ALARM, campaign_between_case_work_seconds=WORK_WALL,
    case_work_seconds=[row[3] for row in CASES], failed_case_cleanup_seconds=5,
    sampled_case_rss_bytes=RSS_BYTES, log_bytes_per_case=LOG_BYTES,
    admin_bytes_per_file=ADMIN_BYTES, expected_cases=8, maximum_live_python_processes=3,
    maximum_payload_lifetimes=10,
    wall_scope="BETWEEN_CASE_CHECKS_PLUS_BOUNDED_SUPERVISOR_NOT_HARD_SYNCHRONOUS_IO")


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read(path, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, "INVALID_OWNED_FILE")
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, cap-size+1))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            require(size <= cap, "READ_CAP")
        require(identity(before) == identity(os.fstat(fd)) == identity(Path(path).lstat())
            and size == before.st_size, "FILE_CHANGED")
        return b"".join(chunks)
    finally:
        os.close(fd)


def binding(path, cap=ADMIN_BYTES):
    blob = read(path, cap)
    return dict(bytes=len(blob), sha256=sha(blob))


def write(path, value):
    blob = encoded(value)
    require(0 < len(blob) <= ADMIN_BYTES, "PUBLICATION_CAP")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            require(output.write(blob) == len(blob), "SHORT_PUBLICATION_WRITE")
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(fd)
    return dict(bytes=len(blob), sha256=sha(blob))


def source_bindings():
    try:
        (ROOT/"__init__.py").lstat()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("ROOT_INITIALIZER_MUST_BE_ABSENT")
    result = {name: binding(ROOT/name) for name in SOURCES}
    require(result[SUT]["sha256"] == SUT_SHA, "FROZEN_SUPERVISOR_CHANGED")
    return result


def runtime():
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
        "REQUIRES_I_S_B")
    require(sys.version_info[:2] == (3, 11) and sys.platform == "darwin", "REQUIRES_HOST_PYTHON311")
    exe = Path(sys.executable).resolve(strict=True)
    require(exe == PYTHON.resolve(strict=True), "WRONG_INTERPRETER")
    return dict(python_version=sys.version, executable=str(exe),
        executable_stat=list(identity(exe.stat())), implementation=sys.implementation.name,
        platform=sys.platform, machine=os.uname().machine, kernel_release=os.uname().release,
        kernel_version=os.uname().version,
        scope="EXECUTABLE_METADATA_AND_KERNEL_IDENTITY_NOT_TRANSITIVE_BINARY_CLOSURE")


def admit(pre_sha, sources, environment):
    require(type(pre_sha) is str and len(pre_sha) == 64
        and all(c in "0123456789abcdef" for c in pre_sha), "EXACT_PRE_SHA_REQUIRED")
    blob = read(ROOT/PRE, ADMIN_BYTES)
    require(sha(blob) == pre_sha, "PRE_BYTES_CHANGED")
    markers = [line[len(PRE_MARKER):] for line in blob.decode().splitlines() if line.startswith(PRE_MARKER)]
    require(len(markers) == 1 and len(markers[0]) <= 4096, "UNIQUE_PRE_MARKER_REQUIRED")
    expected = dict(schema="OWNED_PROCESS_SUPERVISION_V2_HOST_PRE", decision="GREEN",
        scope="OWNED_PROCESS_HOST_ONLY", expected_cases=8,
        source_map_sha256=sha(encoded({name: item["sha256"] for name, item in sources.items()})),
        runtime_binding_sha256=sha(encoded(environment)))
    marker = json.loads(markers[0])
    require(type(marker) is dict and marker == expected and type(marker["expected_cases"]) is int,
        "PRE_SCOPE_OR_BINDING_MISMATCH")
    return dict(path=PRE, bytes=len(blob), sha256=pre_sha, declaration=marker)


def payload(name):
    """Only synthetic self-limited children of the approved host invocation."""
    resource.setrlimit(resource.RLIMIT_CPU, (PAYLOAD_CPU, PAYLOAD_CPU))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(98))
    signal.alarm(PAYLOAD_ALARM)
    require(name in {"eof", "nonzero", "deadline", "exec_before", "exec_after",
        "waited_parent", "exiting_parent", "descendant"}, "UNKNOWN_PAYLOAD")
    if name != "exec_after":
        require(len(sys.stdin.buffer.read(16385)) <= 16384, "PAYLOAD_STDIN_CAP")
    print("HOST_PAYLOAD_JSON="+encoded(dict(payload=name, pid=os.getpid(), ppid=os.getppid())).decode(), flush=True)
    if name in {"eof", "nonzero"}:
        time.sleep(.15)  # Allow real RSS observation; absence still fails the gate.
    if name == "nonzero":
        return 7
    if name in {"deadline", "descendant"}:
        time.sleep(3. if name == "deadline" else 1.5)
    elif name == "exec_before":
        time.sleep(.2)
        os.execve(str(PYTHON), [str(PYTHON), "-I", "-S", "-B", "-u", str(ROOT/SELF),
            "--payload", "exec_after"], dict(os.environ))
    elif name == "exec_after":
        time.sleep(.2)
    elif name in {"waited_parent", "exiting_parent"}:
        child = subprocess.Popen([str(PYTHON), "-I", "-S", "-B", "-u", str(ROOT/SELF),
            "--payload", "descendant"], stdin=subprocess.DEVNULL)
        if name == "waited_parent":
            require(child.wait(timeout=2.) == 0, "DESCENDANT_EXIT_MISMATCH")
        else:
            time.sleep(.2)
            os._exit(0)  # Deliberately leave the same-group self-expiring descendant.
    return 0


def case_gate(case, receipt):
    name, _, outer, _, expected_exit, expected_stop = case
    if type(receipt) is not dict:
        return False
    stop_matches = (receipt.get("budget_stop") == expected_stop
        and receipt.get("supervisor_failure_reason") is None)
    if name in {"outer_deadline", "inner_deadline"}:
        stop_matches = stop_matches or (receipt.get("budget_stop") == "SUPERVISOR_FAILURE:TimeoutError"
            and receipt.get("supervisor_failure_reason") == "NATIVE_PROCESS_SAMPLE_DEADLINE")
    tree_peak, parent_peak = receipt.get("sampled_tree_peak_rss_bytes"), receipt.get("sampled_parent_peak_rss_bytes")
    ok = (type(receipt) is dict and receipt.get("exit_code") == expected_exit
        and stop_matches and receipt.get("direct_child_reaped") is True
        and receipt.get("requires_process_attention") is False
        and receipt.get("cleanup_errors") == [] and receipt.get("control_close_errors") == []
        and receipt.get("observed_foreign_group_escape_detected") is False
        and receipt.get("observed_identity_unknown_pid_count_peak") == 0
        and receipt.get("log_storage_bytes_uncertain") is False
        and receipt.get("owned_group_empty_confirmed") is (True if outer else None)
        and receipt.get("sampled_budget_peak_rss_bytes", RSS_BYTES+1) <= RSS_BYTES
        and type(tree_peak) is int and type(parent_peak) is int
        and tree_peak > 0 and parent_peak > 0
        and receipt.get("child_tree_rss_sample_count", 0) > 0
        and tree_peak + parent_peak <= RSS_BYTES)
    if name in {"outer_deadline", "inner_deadline", "outer_surviving_descendant"}:
        ok = ok and receipt.get("native_process_work", {}).get("signal_calls", 0) >= 1
    if name == "outer_exec":
        ok = ok and receipt.get("identity_counters", {}).get("exec_version_updates", 0) >= 1
    if name in {"outer_waited_descendant", "outer_surviving_descendant"}:
        ok = ok and receipt.get("identity_counters", {}).get("tracked_peak", 0) >= 2
    return bool(ok)


def execute(pre_sha):
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (PARENT_CPU, PARENT_CPU))
    sources, environment = source_bindings(), runtime()
    pre = admit(pre_sha, sources, environment)
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    require(stat.S_ISDIR(base.lstat().st_mode), "OWNED_RETAINED_PARENT_REQUIRED")
    retained = Path(tempfile.mkdtemp(prefix="owned_process_supervision_v2_host-", dir=base))
    cache = Path(tempfile.mkdtemp(prefix="sherman-process-host-v2-")).resolve(strict=True)
    inventory = {"STARTED.json": write(retained/"STARTED.json", dict(schema="OWNED_PROCESS_HOST_V2_STARTED",
        cases=CASES, limits=LIMITS, sources=sources, runtime=environment, pre=pre,
        private_cache_metadata_only=str(cache)))}
    module = types.ModuleType("_owned_process_host_v2_candidate")
    results, halt, quiescent, load_error, publication_errors = [], None, True, None, []
    try:
        candidate = read(ROOT/SUT, ADMIN_BYTES)
        require(sha(candidate) == sources[SUT]["sha256"], "CANDIDATE_CHANGED_BEFORE_LOAD")
        exec(compile(candidate, str(ROOT/SUT), "exec"), module.__dict__)
    except Exception as failure:
        load_error = type(failure).__name__+":"+str(failure)[:256]
        halt = "CANDIDATE_LOAD_FAILED_BEFORE_ANY_CASE_INVOCATION"
    for index, case in enumerate(CASES):
        name, role, outer, case_wall, _, _ = case
        receipt, error, log_binding, invoked = None, None, None, False
        if halt is None and time.monotonic()-started >= WORK_WALL:
            halt = "CAMPAIGN_BETWEEN_CASE_WALL_LIMIT"
        if halt is None:
            try:
                require(source_bindings() == sources and runtime() == environment
                    and admit(pre_sha, sources, environment) == pre, "CASE_PREFLIGHT_CHANGED")
                invoked = True
                receipt = module.supervise_process([str(PYTHON), "-I", "-S", "-B", "-u",
                    str(ROOT/SELF), "--payload", role], dict(case=name), cache=cache,
                    log_path=retained/f"CASE_{index:02d}_OUTPUT.txt", wall_seconds=case_wall,
                    rss_limit=RSS_BYTES, log_limit=LOG_BYTES, new_session=outer,
                    parent_rss_limit=RSS_BYTES)
                quiescent = (receipt.get("direct_child_reaped") is True
                    and receipt.get("requires_process_attention") is False
                    and (not outer or receipt.get("owned_group_empty_confirmed") is True))
                if quiescent and receipt.get("log_created"):
                    log_binding = binding(retained/f"CASE_{index:02d}_OUTPUT.txt", LOG_BYTES)
                    require(log_binding["bytes"] == receipt.get("stored_log_bytes"), "CASE_LOG_SIZE_MISMATCH")
                if not quiescent:
                    halt = "OWNED_WRITERS_UNCONFIRMED_NO_MORE_CASES"
            except Exception as failure:
                error = type(failure).__name__+":"+str(failure)[:256]
                halt = "CASE_INVOCATION_OR_CUSTODY_FAILURE"
                if invoked and receipt is None:
                    quiescent = False
        result = dict(case=name, invocation_started=invoked, receipt=receipt, error=error,
            unavailable_reason=halt if receipt is None else None,
            passed=receipt is not None and error is None and log_binding is not None and case_gate(case, receipt),
            output_binding=log_binding)
        results.append(result)
        try:
            inventory[f"CASE_{index:02d}_RESULT.json"] = write(retained/f"CASE_{index:02d}_RESULT.json", result)
        except Exception as failure:
            publication_errors.append(f"CASE_{index:02d}:"+type(failure).__name__+":"+str(failure)[:160])
            halt = "CASE_PUBLICATION_FAILURE_NO_MORE_INVOCATIONS"
        if log_binding is not None:
            inventory[f"CASE_{index:02d}_OUTPUT.txt"] = log_binding
    postflight, after_sources, after_runtime, after_pre = [], None, None, None
    if quiescent:
        for label, function in (("sources", source_bindings), ("runtime", runtime),
                ("pre", lambda: admit(pre_sha, sources, environment))):
            try:
                value = function()
                if label == "sources": after_sources = value
                elif label == "runtime": after_runtime = value
                else: after_pre = value
            except Exception as failure:
                postflight.append(label+":"+type(failure).__name__+":"+str(failure)[:160])
    else:
        postflight.append("WRITERS_UNCONFIRMED_NO_POSTFLIGHT_CONTENT_READ")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime+usage.ru_stime
    gate = (len(results) == 8 and all(row["passed"] for row in results) and quiescent
        and not postflight and not publication_errors and after_sources == sources and after_runtime == environment
        and after_pre == pre and cpu <= PARENT_CPU and time.monotonic()-started <= WORK_WALL)
    completed = dict(schema="OWNED_PROCESS_HOST_V2_COMPLETED", outcome_gate=gate,
        status="PASS_PENDING_PUBLICATION_AND_INDEPENDENT_POST" if gate else "FAIL",
        expected_cases=8, passed=sum(row["passed"] for row in results), results=results,
        parent_cpu_seconds=cpu, parent_peak_rss_bytes=usage.ru_maxrss,
        wall_seconds=time.monotonic()-started, all_owned_writers_confirmed_quiescent=quiescent,
        halt_reason=halt, candidate_load_error=load_error, publication_errors=list(publication_errors),
        postflight_errors=postflight, sources_after=after_sources,
        runtime_after=after_runtime, pre_after=after_pre, limits=LIMITS)
    try:
        inventory["COMPLETED.json"] = write(retained/"COMPLETED.json", completed)
    except Exception as failure:
        publication_errors.append("COMPLETED:"+type(failure).__name__+":"+str(failure)[:160])
    publication_error = None
    try:
        require(quiescent, "UNCONFIRMED_WRITERS_NO_FINAL_INVENTORY")
        require(not publication_errors, "ADMINISTRATIVE_PUBLICATION_FAILURE")
        with os.scandir(retained) as iterator:
            names = []
            for item in iterator:
                require(len(names) < 19, "FINAL_NAMESPACE_CAP")
                names.append(item.name)
        require(set(names) == set(inventory), "FINAL_NAMESPACE_MISMATCH")
        require(all(binding(retained/name, LOG_BYTES if name.endswith("OUTPUT.txt") else ADMIN_BYTES) == value
            for name, value in inventory.items()), "FINAL_CONTENT_MISMATCH")
    except Exception as failure:
        publication_error = type(failure).__name__+":"+str(failure)[:256]
    final = gate and publication_error is None
    print(encoded(dict(status="PASS_PENDING_INDEPENDENT_POST" if final else "FAIL",
        outcome_gate=gate, final_publication_gate=publication_error is None,
        publication_error=publication_error, publication_errors=publication_errors,
        retained_attempt_directory=str(retained),
        final_inventory=inventory if publication_error is None else None,
        parent_through_inventory_wall_seconds=time.monotonic()-started)).decode(), flush=True)
    return 0 if final else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--payload", help=argparse.SUPPRESS)
    parser.add_argument("--pre-sha256")
    arguments = parser.parse_args()
    return payload(arguments.payload) if arguments.payload else execute(arguments.pre_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
