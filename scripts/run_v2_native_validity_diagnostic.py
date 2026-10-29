"""Source-freeze/one-shot supervisor for one consumed CAD validity observation.

No application/native import or input access at module import. --freeze reads
named sources and runtime metadata only. --execute needs a separately authored
exact-bundle audit. The worker CLI is a private one-use parent-bound capability,
not permission to run a diagnostic. No scan, Gold, alternate path or retry mode.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUN = ROOT / "docs/research/v2_native_validity_diagnostic_attempt"
CAD_SHA = "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"
CAD_BYTES = 260402
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
SUPPORT_SHA = "100b9dc7dfcb5dfe6a3f4bdd77b03e375ff878278efdf9524d0582299a856837"
DIAGNOSTIC_SHA = "e2767721c5b6c1bdf02b77cb42d4c6c2cbc68402b6e63407afecfb92d8cfca91"
PROTOCOL_SHA = "129c63facdecaaaee2a5ed0ebfb93dfd6020c3e427759abf5d3dfdbdb1e2fd45"
AUDIT_DECISION = "GREEN_FOR_FROZEN_SINGLE_CAD_NATIVE_VALIDITY_DIAGNOSTIC"
SCHEMA = "SINGLE_CAD_NATIVE_VALIDITY_SUPERVISOR_V1"
LIMITS = dict(total_wall_seconds=120, terminalization_reserve_seconds=10,
              worker_cpu_seconds=90, sampled_complete_tree_rss_bytes=1024**3,
              combined_worker_output_bytes=1024**2, result_bytes=768*1024,
              parent_io_seconds=5, input_bytes=CAD_BYTES, dispatch_bytes=4096)
SUPERVISION_CONTRACT = dict(
    scope="ONE_EXACT_CONSUMED_CAD_NO_SCAN_GOLD_OR_INFERENCE",
    input="ONE_NOFOLLOW_REGULAR_DESCRIPTOR_SNAPSHOT_AFTER_EXCLUSIVE_STARTED",
    wall="120S_FROM_EXECUTE_ENTRY_INCLUDING_PREPARATION_OUTPUT_AND_TERMINALIZATION",
    work_wall="110S_MAX_WITH_10S_RESERVED_INSIDE_TOTAL_FOR_KILL_AND_TERMINAL_RECEIPT",
    process_boundary="OUTER_WATCHDOG_NEW_SESSION_COORDINATOR_NATIVE_WORKER_INHERITS_GROUP",
    memory="SAMPLED_ENTIRE_COORDINATOR_GROUP_PLUS_OUTER_AND_MONITOR_DESCENDANTS_1GIB",
    output="ONE_LIVE_COMBINED_COORDINATOR_AND_NATIVE_STDOUT_STDERR_STREAM_1MIB",
    cleanup="COORDINATOR_CLEANUP_INSIDE_KILLABLE_GROUP_NO_OUTER_RECURSIVE_CLEANUP",
    failures="TERMINAL_NO_RETRY_NO_HEALING_NO_CAP_CHANGE",
    native_exact_method="OCCT_OPTION_NOT_CERTIFIED_EXACT_GEOMETRY",
    authority="INDEPENDENT_EXACT_BUNDLE_AUDIT_AND_EXCLUSIVE_STARTED_REQUIRED",
    limitations="PYTHON_IO_GUARD_NOT_OS_SANDBOX_RUNTIME_METADATA_NOT_BINARY_ATTESTATION",
    production_use_allowed=False)
SOURCES = {
    "driver.py": "scripts/run_v2_native_validity_diagnostic.py",
    "diagnostic/v2_native_validity_diagnostic.py": "scripts/v2_native_validity_diagnostic.py",
    "review.py": "scripts/review_gold11_inputs_v1.py",
    "protocol.md": "docs/research/2026-09-09_v2_native_validity_diagnostic_protocol.md",
    "support.py": "docs/research/gold11_mesh_development_v1_attempt/source/scripts/run_gold11_mesh_development_v1.py",
    "tests/supervisor.py": "tests/test_v2_native_validity_supervisor.py",
    "tests/diagnostic.py": "tests/test_v2_native_validity_diagnostic.py",
}
FIXED_HASHES = {"support.py": SUPPORT_SHA, "review.py": REVIEW_SHA,
                "diagnostic/v2_native_validity_diagnostic.py": DIAGNOSTIC_SHA,
                "protocol.md": PROTOCOL_SHA}
# Only these generic functions from the exact reviewed V1 bytes are compiled.
# No V1 source assignments, main, case/coordinator/scorer/dispatch functions run.
SUPPORT_FUNCTIONS = (
    "digest", "encoded", "unique_object", "decode", "deadline", "_bounded_read",
    "bounded_read", "exclusive_json", "runtime_metadata", "contained",
    "runtime_roots", "allowed_native_library", "worker_audit_hook",
    "install_namespace", "initialize_worker", "process_table", "descendants",
    "clean_environment", "memory_budget", "cache_identity")
WORKER_SOURCES = ("driver.py", "support.py", "diagnostic/v2_native_validity_diagnostic.py")
HARD_END = None
ACTIVE_GROUP = None


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def decode(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    def reject(_):
        raise ValueError("NONFINITE_JSON")
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=reject)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def same(left, right):
    return encoded(left) == encoded(right)


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def kill_group(pgid):
    """Only the new session created by this outer watchdog; never wait here."""
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def hard_exit(*_):
    # No exception handler, repeated wait, tempfile finalizer or blocked receipt
    # path can extend the total deadline. STARTED remains durably spent.
    try:
        kill_group(ACTIVE_GROUP)
    finally:
        os._exit(124)


def restore_alarm(previous, handler, elapsed):
    remaining_previous = max(1e-6, previous[0]-elapsed) if previous[0] else None
    if HARD_END is not None:
        left = HARD_END-time.monotonic()
        if left <= 0:
            hard_exit()
        if remaining_previous is None or left <= remaining_previous:
            signal.signal(signal.SIGALRM, hard_exit)
            signal.setitimer(signal.ITIMER_REAL, left)
            return
    signal.signal(signal.SIGALRM, handler)
    if remaining_previous is not None or previous[1]:
        signal.setitimer(signal.ITIMER_REAL, remaining_previous or 0, previous[1])


@contextmanager
def hard_deadline(end):
    global HARD_END
    began = time.monotonic()
    previous = signal.getitimer(signal.ITIMER_REAL)
    handler, prior_end = signal.getsignal(signal.SIGALRM), HARD_END
    HARD_END = min(end, began+previous[0]) if previous[0] else end
    signal.signal(signal.SIGALRM, hard_exit)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, HARD_END-began))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        HARD_END = prior_end
        restore_alarm(previous, handler, time.monotonic()-began)


@contextmanager
def deadline(seconds):
    """Soft I/O deadlines always leave the independent total alarm armed."""
    require(seconds > 0, "WALL_BUDGET_EXHAUSTED")
    def expired(*_):
        if HARD_END is not None:
            left = HARD_END-time.monotonic()
            if left <= 0:
                hard_exit()
            signal.signal(signal.SIGALRM, hard_exit)
            signal.setitimer(signal.ITIMER_REAL, left)
        raise TimeoutError("PARENT_WALL_LIMIT")
    started = time.monotonic()
    previous = signal.getitimer(signal.ITIMER_REAL)
    handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous[0]) if previous[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        restore_alarm(previous, handler, time.monotonic()-started)


def read_file(path, cap, expected=None, observation=None, *, allow_empty=False):
    """One regular no-follow descriptor, bounded bytes and before/after identity."""
    with deadline(LIMITS["parent_io_seconds"]):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and (0 if allow_empty else 1) <= before.st_size <= cap,
                    "NONREGULAR_OR_BYTE_BUDGET")
            chunks, total = [], 0
            while True:
                block = os.read(fd, min(1024**2, cap+1-total))
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                require(total <= cap, "BYTE_BUDGET")
            after = os.fstat(fd)
            require(all(getattr(before, k) == getattr(after, k)
                        for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
                    "SOURCE_CHANGED_DURING_READ")
            blob = b"".join(chunks)
            require(expected is None or digest(blob) == expected, "DECLARED_HASH_MISMATCH")
            if observation is not None:
                observation.update(sha256=digest(blob), bytes=len(blob), device=before.st_dev,
                    inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns,
                    regular_descriptor=True, descriptor_metadata_stable=True)
            return blob
        finally:
            os.close(fd)


def read_json(path, cap=256*1024):
    return decode(read_file(path, cap))


def exclusive_json(path, value):
    with deadline(LIMITS["parent_io_seconds"]):
        with Path(path).open("xb") as stream:
            stream.write(encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


def generic_support(blob):
    require(digest(blob) == SUPPORT_SHA, "GENERIC_SUPPORT_SOURCE_MISMATCH")
    tree = ast.parse(blob)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(all(name in functions for name in SUPPORT_FUNCTIONS), "GENERIC_SUPPORT_CLOSURE")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    # The pinned source's top-level imports are standard library only.
    selected = ast.Module(body=imports+[functions[name] for name in SUPPORT_FUNCTIONS], type_ignores=[])
    module = ModuleType("_verified_v1_generic_support_only")
    module.__dict__.update(ROOT=ROOT, LIMITS={"parent_read_seconds": LIMITS["parent_io_seconds"]})
    exec(compile(selected, "<hash-verified-v1-generic-functions>", "exec"), module.__dict__)
    return module


def approved_input(review_blob):
    """Parse only the hash-pinned review constant; never resolve/stat its paths."""
    require(digest(review_blob) == REVIEW_SHA, "REVIEW_SOURCE_BINDING_MISMATCH")
    nodes = [n for n in ast.parse(review_blob).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "CASES" for t in n.targets)]
    require(len(nodes) == 1, "REVIEW_CASE_CONSTANT_SCHEMA")
    call = nodes[0].value
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
            and call.func.attr == "loads" and len(call.args) == 1 and not call.keywords,
            "REVIEW_CASE_CONSTANT_SCHEMA")
    cases = decode(ast.literal_eval(call.args[0]))
    require(type(cases) is list and len(cases) == 11, "REVIEW_COHORT_SCHEMA")
    selected = [c for c in cases if c.get("case_token") == "CG191DEC7F81"]
    require(len(selected) == 1 and selected[0].get("part_key") == "49001000"
            and selected[0].get("cad_sha256") == CAD_SHA, "APPROVED_SINGLE_CAD_BINDING")
    path = selected[0].get("cad_path")
    require(type(path) is str and "\0" not in path and PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts, "APPROVED_CAD_PATH_SCHEMA")
    return dict(cad_path=path, cad_sha256=CAD_SHA, cad_bytes=CAD_BYTES,
                provenance_case="CG191DEC7F81", provenance_family="49001000",
                review_source_sha256=REVIEW_SHA)


def freeze():
    blobs = {alias: read_file(ROOT/name, 2*1024**2, FIXED_HASHES.get(alias))
             for alias, name in SOURCES.items()}
    helper = generic_support(blobs["support.py"])
    bundle = dict(schema=SCHEMA, limits=LIMITS, supervision_contract=SUPERVISION_CONTRACT,
        approved_input=approved_input(blobs["review.py"]), runtime_metadata=helper.runtime_metadata(),
        sources={alias: dict(path=SOURCES[alias], sha256=digest(blob)) for alias, blob in blobs.items()})
    RUN.mkdir(exist_ok=False)
    for alias, blob in blobs.items():
        destination = RUN/"source"/alias
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(blob)
    exclusive_json(RUN/"FROZEN.json", bundle)
    print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", bundle_sha256=digest(encoded(bundle)))), flush=True)


def verify_bundle():
    bundle = read_json(RUN/"FROZEN.json")
    require(bundle.get("schema") == SCHEMA and same(bundle.get("limits"), LIMITS)
            and same(bundle.get("supervision_contract"), SUPERVISION_CONTRACT), "FROZEN_CONTRACT_MISMATCH")
    records = bundle.get("sources")
    require(type(records) is dict and set(records) == set(SOURCES), "FROZEN_SOURCE_DENOMINATOR")
    blobs = {}
    for alias, name in SOURCES.items():
        record = records[alias]
        require(type(record) is dict and set(record) == {"path", "sha256"}
                and record["path"] == name and is_hash(record["sha256"]), "FROZEN_SOURCE_SCHEMA")
        require(alias not in FIXED_HASHES or record["sha256"] == FIXED_HASHES[alias], "FIXED_SOURCE_CHANGED")
        read_file(ROOT/name, 2*1024**2, record["sha256"])
        blobs[alias] = read_file(RUN/"source"/alias, 2*1024**2, record["sha256"])
    helper = generic_support(blobs["support.py"])
    require(same(bundle.get("runtime_metadata"), helper.runtime_metadata()), "RUNTIME_METADATA_CHANGED")
    # Manifest paths never authorize input access; this value is rederived from
    # the immutable, hash-pinned review SOURCE before every admitted execution.
    binding = approved_input(blobs["review.py"])
    require(same(bundle.get("approved_input"), binding), "FROZEN_APPROVED_INPUT_CHANGED")
    return bundle, binding, helper


def verify_audit(bundle):
    audit = read_json(RUN/"PRE_RUN_AUDIT.json")
    require(audit.get("decision") == AUDIT_DECISION
            and audit.get("bundle_sha256") == digest(encoded(bundle))
            and type(audit.get("reviewer")) is str and bool(audit["reviewer"].strip())
            and audit.get("blocking_findings") == [], "EXACT_INDEPENDENT_AUDIT_REQUIRED")
    return audit


def issue_dispatch(bundle, audit, started, cache, helper):
    context = dict(schema=SCHEMA, parent_pid=os.getpid(), bundle_sha256=digest(encoded(bundle)),
        audit_sha256=digest(encoded(audit)), started_sha256=digest(encoded(started)), limits=LIMITS,
        source_sha256={alias: row["sha256"] for alias, row in bundle["sources"].items()},
        runtime_metadata=bundle["runtime_metadata"], cad_sha256=CAD_SHA, cad_bytes=CAD_BYTES)
    exclusive_json(RUN/"VERIFIED_CONTEXT.json", context)
    token = secrets.token_hex(32)
    dispatch = dict(schema=SCHEMA, role="SINGLE_CAD_VALIDITY_WORKER", parent_pid=os.getpid(),
        context_sha256=digest(encoded(context)), token_sha256=digest(token.encode()),
        cache=str(cache), cache_identity=helper.cache_identity(cache))
    exclusive_json(RUN/"DISPATCH.json", dispatch)
    return dict(token=token)


def issue_coordinator(bundle, audit, started, cache, helper):
    token = secrets.token_hex(32)
    record = dict(schema=SCHEMA, role="ONE_USE_COORDINATOR", parent_pid=os.getpid(),
        bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
        started_sha256=digest(encoded(started)), token_sha256=digest(token.encode()),
        cache=str(cache), cache_identity=helper.cache_identity(cache))
    exclusive_json(RUN/"COORDINATOR_DISPATCH.json", record)
    return dict(token=token)


def authorize_coordinator(request):
    require(type(request) is dict and set(request) == {"token"} and is_hash(request["token"]),
            "ONE_USE_COORDINATOR_REQUIRED")
    bundle, binding, helper = verify_bundle()
    audit = verify_audit(bundle)
    started = read_json(RUN/"STARTED.json", 4096)
    record = read_json(RUN/"COORDINATOR_DISPATCH.json", 4096)
    require(record.get("schema") == SCHEMA and record.get("role") == "ONE_USE_COORDINATOR"
            and record.get("parent_pid") == os.getppid() and os.getpgrp() == os.getpid()
            and record.get("bundle_sha256") == started.get("bundle_sha256") == digest(encoded(bundle))
            and record.get("audit_sha256") == started.get("audit_sha256") == digest(encoded(audit))
            and record.get("started_sha256") == digest(encoded(started))
            and started.get("status") == "STARTED_NO_RETRY" and same(started.get("limits"), LIMITS)
            and secrets.compare_digest(record.get("token_sha256", ""), digest(request["token"].encode())),
            "COORDINATOR_PARENT_OR_CONTROL_BINDING")
    require(Path(__file__).resolve() == (RUN/"source/driver.py").resolve(), "FROZEN_COORDINATOR_REQUIRED")
    require(not (RUN/"COMPLETED.json").exists(), "ATTEMPT_ALREADY_TERMINAL")
    exclusive_json(RUN/"COORDINATOR_CLAIMED.json", dict(coordinator_pid=os.getpid(),
        parent_pid=os.getppid(), dispatch_sha256=digest(encoded(record)), no_retry=True))
    cache = Path(record["cache"])
    require(helper.cache_identity(cache) == record["cache_identity"]
            and Path.cwd() == cache and os.environ.get("TMPDIR") == str(cache), "COORDINATOR_CACHE_CHANGED")
    began = started.get("began_monotonic")
    require(type(began) in (int, float) and began <= time.monotonic()
            and started.get("work_stop_monotonic") == began + LIMITS["total_wall_seconds"]-LIMITS["terminalization_reserve_seconds"],
            "COORDINATOR_ABSOLUTE_WALL_BINDING")
    remaining(started["work_stop_monotonic"])
    return bundle, binding, helper, audit, started, cache, record["cache_identity"]


def authorize_worker(request):
    require(type(request) is dict and set(request) == {"token"} and is_hash(request["token"]),
            "ONE_USE_PARENT_DISPATCH_REQUIRED")
    dispatch = read_json(RUN/"DISPATCH.json", 4096)
    context = read_json(RUN/"VERIFIED_CONTEXT.json", 32768)
    started = read_json(RUN/"STARTED.json", 4096)
    audit = read_json(RUN/"PRE_RUN_AUDIT.json")
    coordinator = read_json(RUN/"COORDINATOR_CLAIMED.json", 4096)
    coordinator_dispatch = read_json(RUN/"COORDINATOR_DISPATCH.json", 4096)
    require(dispatch.get("schema") == context.get("schema") == SCHEMA
            and dispatch.get("role") == "SINGLE_CAD_VALIDITY_WORKER"
            and dispatch.get("parent_pid") == context.get("parent_pid") == os.getppid()
            and os.getpgrp() == os.getppid()
            and coordinator.get("coordinator_pid") == os.getppid()
            and coordinator.get("parent_pid") == coordinator_dispatch.get("parent_pid")
            and coordinator.get("dispatch_sha256") == digest(encoded(coordinator_dispatch))
            and coordinator.get("no_retry") is True
            and dispatch.get("context_sha256") == digest(encoded(context))
            and secrets.compare_digest(dispatch.get("token_sha256", ""), digest(request["token"].encode()))
            and context.get("started_sha256") == digest(encoded(started))
            and context.get("bundle_sha256") == started.get("bundle_sha256") == audit.get("bundle_sha256")
            and context.get("audit_sha256") == started.get("audit_sha256") == digest(encoded(audit))
            and audit.get("decision") == AUDIT_DECISION and audit.get("blocking_findings") == []
            and type(audit.get("reviewer")) is str and bool(audit["reviewer"].strip())
            and started.get("status") == "STARTED_NO_RETRY"
            and same(started.get("limits"), LIMITS) and same(context.get("limits"), LIMITS)
            and context.get("cad_sha256") == CAD_SHA and context.get("cad_bytes") == CAD_BYTES,
            "WORKER_PARENT_OR_CONTROL_BINDING")
    require(not (RUN/"COMPLETED.json").exists(), "ATTEMPT_ALREADY_TERMINAL")
    # Burn before frozen payload/runtime/input access. A later failure cannot
    # return to this capability or create another worker in the same attempt.
    exclusive_json(RUN/"CLAIMED.json", dict(worker_pid=os.getpid(), parent_pid=os.getppid(),
        dispatch_sha256=digest(encoded(dispatch)), no_retry=True))
    require(Path(__file__).resolve() == (RUN/"source/driver.py").resolve(), "FROZEN_WORKER_REQUIRED")
    hashes = context.get("source_sha256")
    require(type(hashes) is dict and set(hashes) == set(SOURCES) and all(map(is_hash, hashes.values())),
            "WORKER_SOURCE_DENOMINATOR")
    blobs = {alias: read_file(RUN/"source"/alias, 2*1024**2, hashes[alias]) for alias in WORKER_SOURCES}
    require(hashes["support.py"] == SUPPORT_SHA
            and hashes["diagnostic/v2_native_validity_diagnostic.py"] == DIAGNOSTIC_SHA,
            "WORKER_FIXED_SOURCE_CHANGED")
    helper = generic_support(blobs["support.py"])
    require(same(context.get("runtime_metadata"), helper.runtime_metadata()), "WORKER_RUNTIME_CHANGED")
    cache = Path(dispatch["cache"])
    require(helper.cache_identity(cache) == dispatch.get("cache_identity")
            and Path.cwd() == cache and os.environ.get("TMPDIR") == str(cache), "WORKER_CACHE_CHANGED")
    return cache, helper


def worker():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["worker_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(LIMITS["dispatch_bytes"]+1))
    cache, helper = authorize_worker(request)
    helper.initialize_worker(RUN/"source", cache, "diagnostic", LIMITS["worker_cpu_seconds"])
    from diagnostic.v2_native_validity_diagnostic import diagnose_v2_native_validity
    blob = helper.bounded_read(cache/"cad.step", CAD_BYTES, CAD_SHA)
    require(len(blob) == CAD_BYTES, "PRIVATE_CAD_SIZE_MISMATCH")
    result = diagnose_v2_native_validity(blob, allow_ocp=True)
    result_blob = encoded(result)
    require(len(result_blob) <= LIMITS["result_bytes"], "RESULT_BYTE_BUDGET")
    with (cache/"RESULT.json").open("xb") as stream:
        stream.write(result_blob)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(status="SINGLE_CAD_WORKER_TERMINAL", result_sha256=digest(result_blob),
        worker_pid=os.getpid(),
        worker_cpu_seconds=usage.ru_utime+usage.ru_stime, worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)


def remaining(stop):
    seconds = stop-time.monotonic()
    require(seconds > 0, "TOTAL_WALL_EXHAUSTED")
    return seconds


def memory_check(helper):
    row = helper.process_table().get(os.getpid())
    require(row is not None and row["rss"] <= LIMITS["sampled_complete_tree_rss_bytes"]
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= LIMITS["sampled_complete_tree_rss_bytes"],
            "SUPERVISOR_MEMORY_UNAVAILABLE_OR_OVER_LIMIT")


def validate_result(result):
    require(type(result) is dict and result.get("schema") == "V2_NATIVE_VALIDITY_DIAGNOSTIC"
            and result.get("status") in ("DIAGNOSTIC_COMPLETE", "PARTIAL_DIAGNOSTIC", "UNAVAILABLE")
            and result.get("expected_cad_sha256") == CAD_SHA and result.get("expected_cad_bytes") == CAD_BYTES
            and result.get("production_use_allowed") is False and result.get("observed_bend_count") is None
            and result.get("predictions_not_computed") is True
            and type(result.get("native_analyzer_count")) is int and 0 <= result["native_analyzer_count"] <= 1
            and type(result.get("step_import_count")) is int and result["step_import_count"] == 1,
            "WORKER_REPORT_CONTRACT")
    require(type(result.get("evidence_complete")) is bool
            and result["evidence_complete"] == (result["status"] == "DIAGNOSTIC_COMPLETE")
            and (result.get("native_valid") is None or type(result["native_valid"]) is bool),
            "WORKER_REPORT_COMPLETENESS")


def verify_worker_claim(worker_end, expected_parent=None):
    claim = read_json(RUN/"CLAIMED.json", 4096)
    dispatch = read_json(RUN/"DISPATCH.json", 4096)
    require(type(worker_end.get("worker_pid")) is int and worker_end["worker_pid"] > 0
            and claim.get("worker_pid") == worker_end["worker_pid"]
            and claim.get("parent_pid") == dispatch.get("parent_pid") == (os.getpid() if expected_parent is None else expected_parent)
            and claim.get("dispatch_sha256") == digest(encoded(dispatch))
            and claim.get("no_retry") is True, "WORKER_CLAIM_RESULT_BINDING")
    return digest(encoded(claim))


def watchdog_sample(helper, coordinator_pid):
    table = helper.process_table()
    require(os.getpid() in table, "OUTER_RSS_MISSING")
    group = {pid for pid, row in table.items() if row["pgid"] == coordinator_pid}
    # Include orphaned members of the dedicated group AND the outer watchdog's
    # own monitor subprocesses. No double counting of the coordinator tree.
    measured = group | helper.descendants(table, os.getpid()) | {os.getpid()}
    return group, sum(table[pid]["rss"] for pid in measured), table[os.getpid()]["rss"]


def watch_coordinator(command, request, *, cache, helper, work_stop, end):
    """Independent outer watchdog; all cleanup is inside the child group.

    No 110s signal exception is delegated to/cooperatively handled by the
    coordinator. This process owns elapsed-time checks and group termination.
    The caller's hard deadline remains armed across every exceptional path.
    """
    global ACTIVE_GROUP
    began, child, selector = time.monotonic(), None, selectors.DefaultSelector()
    receipt = dict(exit_code=None, budget_stop=None, coordinator_pid=None,
        sampled_budget_peak_rss_bytes=0, sampled_parent_peak_rss_bytes=0,
        rss_budget_scope="ENTIRE_COORDINATOR_GROUP_PLUS_OUTER_AND_MONITOR_DESCENDANTS",
        observed_output_bytes=0, retained_output_bytes=0, group_gone=False)
    try:
        request_blob = encoded(request)
        require(len(request_blob) <= LIMITS["dispatch_bytes"], "DISPATCH_BYTE_BUDGET")
        with (RUN/"OUTPUT.txt").open("xb") as log:
            child = subprocess.Popen(command, cwd=cache, env=helper.clean_environment(cache),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True)
            ACTIVE_GROUP = child.pid
            receipt["coordinator_pid"] = child.pid
            child.stdin.write(request_blob)
            child.stdin.close()
            selector.register(child.stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() >= work_stop:
                    receipt["budget_stop"] = "TOTAL_WORK_WALL_LIMIT"
                    break
                for key, _ in selector.select(timeout=min(.05, max(0., work_stop-time.monotonic()))):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        keep = max(0, LIMITS["combined_worker_output_bytes"]-receipt["retained_output_bytes"])
                        log.write(block[:keep])
                        log.flush()
                        receipt["retained_output_bytes"] += len(block[:keep])
                        receipt["observed_output_bytes"] += len(block)
                if receipt["observed_output_bytes"] > LIMITS["combined_worker_output_bytes"]:
                    receipt["budget_stop"] = "COMBINED_OUTPUT_BYTE_LIMIT"
                    break
                done = child.poll() is not None
                group, rss, parent_rss = watchdog_sample(helper, child.pid)
                if not done:
                    require(child.pid in group or child.poll() is not None, "ACTIVE_COORDINATOR_RSS_MISSING")
                receipt["sampled_budget_peak_rss_bytes"] = max(receipt["sampled_budget_peak_rss_bytes"], rss)
                receipt["sampled_parent_peak_rss_bytes"] = max(receipt["sampled_parent_peak_rss_bytes"], parent_rss)
                if rss > LIMITS["sampled_complete_tree_rss_bytes"]:
                    receipt["budget_stop"] = "COMPLETE_GROUP_PLUS_OUTER_RSS_LIMIT"
                    break
                if done and group-{child.pid}:
                    receipt["budget_stop"] = "COORDINATOR_EXIT_WITH_LIVE_GROUP"
                    break
                if done and not selector.get_map():
                    receipt["group_gone"] = not (group-{child.pid})
                    break
    except Exception as error:
        receipt["budget_stop"] = "OUTER_WATCHDOG_FAILURE:"+type(error).__name__
    finally:
        # Never use the inherited helper's repeated wait(timeout=5) cleanup.
        # Kill is nonblocking; reaping/confirmation has ONE absolute deadline.
        if child is not None:
            if receipt["budget_stop"] is not None or child.poll() is None or not receipt["group_gone"]:
                kill_group(child.pid)
            reap_stop = min(end-1., time.monotonic()+5.)
            while not receipt["group_gone"] and time.monotonic() < reap_stop:
                done = child.poll() is not None
                try:
                    group, _, _ = watchdog_sample(helper, child.pid)
                    receipt["group_gone"] = done and not (group-{child.pid})
                except Exception:
                    break
                if not receipt["group_gone"]:
                    time.sleep(min(.02, max(0., reap_stop-time.monotonic())))
            receipt["exit_code"] = child.poll()
            if not receipt["group_gone"]:
                kill_group(child.pid)
            if child.stdout is not None:
                child.stdout.close()
        else:
            receipt["group_gone"] = True
        selector.close()
        receipt["requires_group_attention"] = not receipt["group_gone"]
        receipt["wall_seconds"] = time.monotonic()-began
        if receipt["group_gone"]:
            ACTIVE_GROUP = None
    return receipt


def coordinator(request):
    bundle, binding, helper, audit, started, cache, identity = authorize_coordinator(request)
    stop, child = started["work_stop_monotonic"], None
    status, phase, error_type, reason = "TERMINAL_FAILURE", "APPROVED_DESCRIPTOR_SNAPSHOT", None, None
    input_snapshot, result_hash, claim_hash, diagnostic_status = {}, None, None, None
    source_consistent, cleanup_complete = False, False
    try:
        with deadline(remaining(stop)):
            # One original descriptor only, AFTER outer STARTED and coordinator
            # one-use claim. Binding comes from fixed review source, not CLI.
            blob = read_file(binding["cad_path"], CAD_BYTES, CAD_SHA, observation=input_snapshot)
            require(len(blob) == CAD_BYTES, "APPROVED_CAD_SIZE_MISMATCH")
            with (cache/"cad.step").open("xb") as stream:
                stream.write(blob)
            del blob
            dispatch = issue_dispatch(bundle, audit, started, cache, helper)
            phase = "GUARDED_NATIVE_WORKER"
            command = [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
                       str(RUN/"source/driver.py"), "--worker"]
            # Inherit the coordinator's dedicated group AND stdout pipe. The
            # independent outer watchdog sees every native log byte live.
            child = subprocess.Popen(command, cwd=cache, env=helper.clean_environment(cache),
                stdin=subprocess.PIPE, stdout=None, stderr=subprocess.STDOUT, start_new_session=False)
            child.stdin.write(encoded(dispatch))
            child.stdin.close()
            require(child.wait(timeout=remaining(stop)) == 0, "NATIVE_WORKER_FAILED")
            phase = "RESULT_CUSTODY"
            result_blob = read_file(cache/"RESULT.json", LIMITS["result_bytes"])
            result = decode(result_blob)
            validate_result(result)
            diagnostic_status, result_hash = result["status"], digest(result_blob)
            claim_hash = verify_worker_claim(dict(worker_pid=child.pid))
            with (RUN/"RESULT.json").open("xb") as stream:
                stream.write(result_blob)
            verified, _, _ = verify_bundle()
            require(same(verified, bundle), "POST_RUN_BUNDLE_CHANGED")
            source_consistent = True
            remaining(stop)
            status, phase = "COMPLETE_DEVELOPMENT_DIAGNOSTIC", "COMPLETE"
    except Exception as error:
        error_type = type(error).__name__
        reason = str(error)[:160] if isinstance(error, (ValueError, TimeoutError)) else error_type
    finally:
        # This entire block, including a slow wait or filesystem cleanup, is
        # INSIDE the independently supervised process group, never the watchdog.
        try:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=1.)
            require(helper.cache_identity(cache) == identity, "CLEANUP_CACHE_IDENTITY_CHANGED")
            os.chdir(RUN)
            shutil.rmtree(cache)
            cleanup_complete = True
        except Exception as error:
            status = "TERMINAL_FAILURE"
            error_type, reason = type(error).__name__, "COORDINATOR_CLEANUP_FAILED"
        exclusive_json(RUN/"COORDINATOR_COMPLETED.json", dict(status=status, phase=phase,
            coordinator_pid=os.getpid(), worker_pid=child.pid if child is not None else None,
            worker_exit_code=child.returncode if child is not None else None,
            error_type=error_type, reason=reason, input_snapshot=input_snapshot,
            result_sha256=result_hash, worker_claim_sha256=claim_hash,
            diagnostic_status=diagnostic_status, source_before_after_consistent=source_consistent,
            cache_cleanup_complete=cleanup_complete, no_resume=True, production_use_allowed=False))
    print(json.dumps(dict(status="COORDINATOR_TERMINAL", outcome=status)), flush=True)


def worker_terminal_from_output(blob):
    candidates = []
    for line in blob.splitlines():
        try:
            row = decode(line)
        except (ValueError, UnicodeError):
            continue
        if type(row) is dict and row.get("status") == "SINGLE_CAD_WORKER_TERMINAL":
            candidates.append(row)
    require(len(candidates) == 1, "EXACT_ONE_NATIVE_WORKER_LOG_TERMINAL_REQUIRED")
    return candidates[0]


def execute():
    began = time.monotonic()
    end = began+LIMITS["total_wall_seconds"]
    work_stop = end-LIMITS["terminalization_reserve_seconds"]
    claimed, bundle, receipt, cache = False, None, None, None
    result_hash, output_hash, claim_hash, source_consistent, report = None, None, None, False, None
    status, phase, error_type, reason = "TERMINAL_FAILURE", "PREFLIGHT", None, None
    with hard_deadline(end):
        try:
            bundle, _, helper = verify_bundle()
            audit = verify_audit(bundle)
            remaining(work_stop)
            started = dict(status="STARTED_NO_RETRY", bundle_sha256=digest(encoded(bundle)),
                audit_sha256=digest(encoded(audit)), limits=LIMITS, no_resume=True,
                began_monotonic=began, work_stop_monotonic=work_stop)
            with (RUN/"STARTED.json").open("xb") as stream:
                claimed = True
                stream.write(encoded(started)+b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            memory_check(helper)
            # No TemporaryDirectory finalizer in the outer watchdog. All
            # recursive cleanup belongs to the independently killable child.
            cache = Path(tempfile.mkdtemp(prefix="v2-native-validity-coordinate-")).resolve()
            request = issue_coordinator(bundle, audit, started, cache, helper)
            phase = "INDEPENDENT_COORDINATOR_WATCHDOG"
            command = [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
                       str(RUN/"source/driver.py"), "--coordinator"]
            # Interrupt outer log/monitor I/O at the work boundary too. The
            # repaired soft handler rearms hard120 before raising, so bounded
            # group kill/reap cannot disable the independent total deadline.
            with deadline(remaining(work_stop)):
                receipt = watch_coordinator(command, request, cache=cache, helper=helper, work_stop=work_stop, end=end)
            remaining(work_stop)
            require(receipt["exit_code"] == 0 and receipt["budget_stop"] is None and receipt["group_gone"],
                    "COORDINATOR_FAILED_OR_BUDGET_STOP")
            phase = "OUTER_RESULT_CUSTODY"
            with deadline(remaining(work_stop)):
                report = read_json(RUN/"COORDINATOR_COMPLETED.json", 32768)
                coordinator_claim = read_json(RUN/"COORDINATOR_CLAIMED.json", 4096)
                coordinator_dispatch = read_json(RUN/"COORDINATOR_DISPATCH.json", 4096)
                require(report.get("coordinator_pid") == coordinator_claim.get("coordinator_pid") == receipt["coordinator_pid"]
                        and coordinator_claim.get("parent_pid") == os.getpid()
                        and coordinator_claim.get("dispatch_sha256") == digest(encoded(coordinator_dispatch))
                        and coordinator_claim.get("no_retry") is True, "COORDINATOR_RECEIPT_BINDING")
                require(report.get("status") == "COMPLETE_DEVELOPMENT_DIAGNOSTIC"
                        and report.get("cache_cleanup_complete") is True and report.get("worker_exit_code") == 0,
                        "COORDINATOR_DIAGNOSTIC_UNAVAILABLE")
                result_blob = read_file(RUN/"RESULT.json", LIMITS["result_bytes"])
                validate_result(decode(result_blob))
                output_blob = read_file(RUN/"OUTPUT.txt", LIMITS["combined_worker_output_bytes"], allow_empty=True)
                worker_end = worker_terminal_from_output(output_blob)
                result_hash, output_hash = digest(result_blob), digest(output_blob)
                require(worker_end.get("result_sha256") == result_hash == report.get("result_sha256")
                        and worker_end.get("worker_pid") == report.get("worker_pid"), "WORKER_RESULT_LOG_BINDING")
                claim_hash = verify_worker_claim(worker_end, expected_parent=receipt["coordinator_pid"])
                require(claim_hash == report.get("worker_claim_sha256"), "COORDINATOR_WORKER_CLAIM_BINDING")
                verified, _, _ = verify_bundle()
                require(same(verified, bundle), "POST_RUN_BUNDLE_CHANGED")
                source_consistent = True
            remaining(work_stop)
            status, phase = "COMPLETE_DEVELOPMENT_DIAGNOSTIC", "COMPLETE"
        except Exception as error:
            if not claimed:
                raise
            status, error_type = "TERMINAL_FAILURE", type(error).__name__
            reason = str(error)[:160] if isinstance(error, (ValueError, TimeoutError)) else error_type
        finally:
            if claimed:
                # The hard alarm remains armed through every caught soft alarm,
                # source check, output read and receipt fsync. At120s it sends
                # SIGKILL to the known group and _exit(124), never more cleanup.
                kill_group(ACTIVE_GROUP)
                if output_hash is None and (RUN/"OUTPUT.txt").exists():
                    try:
                        with deadline(min(1., remaining(end)-1.)):
                            output_hash = digest(read_file(RUN/"OUTPUT.txt", LIMITS["combined_worker_output_bytes"], allow_empty=True))
                    except Exception:
                        pass
                if not source_consistent:
                    try:
                        with deadline(min(3., remaining(end)-1.)):
                            verified, _, _ = verify_bundle()
                            source_consistent = same(verified, bundle)
                    except Exception:
                        pass
                exclusive_json(RUN/"COMPLETED.json", dict(status=status, phase=phase,
                    error_type=error_type, reason=reason, limits=LIMITS,
                    bundle_sha256=digest(encoded(bundle)), source_before_after_consistent=source_consistent,
                    coordinator_receipt=receipt, coordinator_report=report,
                    result_sha256=result_hash, output_sha256=output_hash, worker_claim_sha256=claim_hash,
                    private_scratch_path=str(cache) if cache is not None else None,
                    scratch_cleanup="COORDINATOR_CONFIRMED" if report and report.get("cache_cleanup_complete") else "NOT_CONFIRMED_NO_OUTER_RECURSIVE_CLEANUP",
                    parent_wall_seconds=time.monotonic()-began, no_resume=True,
                    production_use_allowed=False, conformance="INDETERMINATE"))
        print(json.dumps(dict(status=status, attempt=str(RUN))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        worker()
    elif args.coordinator:
        coordinator(decode(sys.stdin.buffer.read(LIMITS["dispatch_bytes"]+1)))
    else:
        execute()


if __name__ == "__main__":
    main()
