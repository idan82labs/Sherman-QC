"""Exclusive, frozen same-input mesh-workflow development comparison.

Freeze is source-only. Execution requires an independent audit of that exact
bundle. Every case/arm is attempted once under fixed limits; no resume/retry CLI.
Gold is opened only by the parent after all inference attempts are terminal.
This is consumed development, not historical replay or production qualification.
"""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata as import_metadata
import json
import os
from pathlib import Path
import resource
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import ModuleType


ROOT = Path("/Users/idantal/ShermanQC/FullProject")
RUN = ROOT / "docs/research/gold11_mesh_development_v1_attempt"
PROTOCOL = "docs/research/2026-09-09_gold11_mesh_development_v1_protocol.md"
REVIEW = "scripts/review_gold11_inputs_v1.py"
REVIEW_SHA = "41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
CATALOG_SHA = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_SHA = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
SOURCES = (
    "scripts/run_gold11_mesh_development_v1.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py",
    "backend/cad_guided_mesh_offline_v1.py", "backend/cad_guided_mesh_input_v1.py",
    "backend/cad_guided_global_registration_v1.py", "backend/cad_guided_cad_sampling_v1.py",
    "backend/cad_guided_native_features_v1.py", "backend/cad_guided_native_features.py",
    "backend/cad_guided_local_measurement_v1.py", "backend/cad_guided_measurement.py",
    "backend/cad_guided_instances.py", REVIEW, PROTOCOL,
)
LIMITS = dict(case_wall_seconds=180, case_cpu_seconds=150,
              sampled_worker_tree_rss_bytes=8*1024**3,
              case_log_bytes=1024**2, case_result_bytes=64*1024**2,
              total_wall_seconds=4200, max_scan_bytes=64*1024**2,
              max_cad_bytes=8*1024**2, scorer_wall_seconds=60,
              scorer_cpu_seconds=50, scorer_rss_bytes=2*1024**3,
              scorer_request_bytes=192*1024**2, scorer_result_bytes=32*1024**2,
              compact_case_bytes=8*1024**2, coordinator_rss_bytes=4*1024**3,
              global_tree_rss_bytes=12*1024**3, parent_read_seconds=20,
              coordinator_log_bytes=2*1024**2, terminalization_reserve_seconds=30)
CONFIG = dict(
    workflow=dict(max_registration_scan_rows=50000, cad_sample_count=32768,
                  cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
                  unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED"),
    registration=dict(coarse_points=2048, refine_points=8192, coarse_iterations=30,
                      refine_iterations=30, trim_fraction=.85, huber_delta_mm=.25,
                      coarse_distance_fraction=.15, max_correspondence_distance_mm=2.,
                      max_inlier_plane_p95_mm=.5, min_scan_coverage=.75,
                      min_cad_coverage=.25, min_observability_ratio=.005,
                      pose_class_distance_mm=.1, ambiguity_plane_rms_margin_mm=.05,
                      ambiguity_coverage_margin=.03, max_rotation_step_radians=.15,
                      max_translation_step_mm=5.),
    measurement=dict(flange_window_mm=[2.5, 12.5], search_margin_mm=3.,
                     max_fit_residual_mm=.05, plane_exclusion_mm=.075,
                     max_tangency_error_mm=.1, fit_model="free_circle"),
    support=dict(max_axial_gap_mm=4., max_angular_gap_deg=5., min_arc_sweep_deg=20.,
                 min_local_points=3),
)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def decode(blob):
    def reject(_):
        raise ValueError("NONFINITE_JSON")
    return json.loads(blob, object_pairs_hook=unique_object, parse_constant=reject)


@contextmanager
def deadline(seconds):
    """Bound parent-side regular-file preparation as well as native workers."""
    def expired(_signum, _frame):
        raise TimeoutError("PARENT_IO_WALL_LIMIT")
    started = time.monotonic()
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    effective = min(seconds, previous_timer[0]) if previous_timer[0] > 0 else seconds
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, effective)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] or previous_timer[1]:
            remaining = max(1e-6, previous_timer[0]-(time.monotonic()-started)) if previous_timer[0] else 0
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def unavailable(reason, phase="SUPERVISOR"):
    return dict(status="UNAVAILABLE", phase=phase, reasons=[reason], predictions=[],
                observed_bend_count=None, production_use_allowed=False)


SCORING_FIELDS = ("status", "phase", "predictions", "observed_bend_count",
                  "production_use_allowed", "conformance", "uncertainty_status",
                  "prediction_frame", "units_status", "reference_frame_compatibility",
                  "reasons", "case_id", "family_id")


def compact_result(payload):
    # Preserve optional identity conflicts and every prediction record: the
    # scorer rejects the whole malformed case, never a filtered valid prefix.
    if type(payload) is not dict or type(payload.get("status")) is not str:
        return unavailable("MALFORMED_WORKFLOW_RESULT")
    result = {key: payload[key] for key in SCORING_FIELDS if key in payload}
    if len(encoded(result)) > LIMITS["compact_case_bytes"]:
        return unavailable("SCORING_PROJECTION_BYTE_BUDGET")
    return result


def _bounded_read(path, cap, expected=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= cap:
            raise ValueError("NONREGULAR_OR_BYTE_BUDGET")
        chunks, total = [], 0
        while True:
            block = os.read(fd, min(1024**2, cap+1-total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > cap:
                raise ValueError("BYTE_BUDGET")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError("SOURCE_CHANGED_DURING_READ")
        blob = b"".join(chunks)
        if expected is not None and digest(blob) != expected:
            raise ValueError("DECLARED_HASH_MISMATCH")
        return blob
    finally:
        os.close(fd)


def bounded_read(path, cap, expected=None):
    with deadline(LIMITS["parent_read_seconds"]):
        return _bounded_read(path, cap, expected)


def exclusive_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value)+b"\n")


def approved_cases_from_review(blob):
    if digest(blob) != REVIEW_SHA:
        raise ValueError("REVIEW_SOURCE_BINDING_MISMATCH")
    assignments = [node for node in ast.parse(blob).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "CASES"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("CASE_CONSTANT_SCHEMA")
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
            and call.func.attr == "loads" and len(call.args) == 1 and not call.keywords):
        raise ValueError("CASE_CONSTANT_SCHEMA")
    cases = decode(ast.literal_eval(call.args[0]))
    if len(cases) != 11 or len({case["scan_sha256"] for case in cases}) != 11:
        raise ValueError("CASE_SCOPE_MISMATCH")
    if len({case["cad_sha256"] for case in cases}) != 6:
        raise ValueError("CAD_SCOPE_MISMATCH")
    return cases


def runtime_metadata():
    versions = {}
    for name in ("numpy", "scipy", "cadquery-ocp"):
        distributions = list(import_metadata.Distribution.discover(name=name, path=[runtime_roots()[0]]))
        if len(distributions) != 1:
            raise ValueError("RUNTIME_DISTRIBUTION_METADATA_AMBIGUOUS:" + name)
        versions[name] = distributions[0].version
    return dict(python_executable=str(Path(sys.executable).resolve()), python_version=sys.version,
                platform=sys.platform, machine=os.uname().machine,
                distributions=versions, attestation="VERSION_METADATA_NOT_BINARY_ATTESTATION")


def freeze():
    # No input/reference paths are touched by this operation.
    blobs = {name: bounded_read(ROOT/name, 2*1024**2) for name in SOURCES}
    cases = approved_cases_from_review(blobs[REVIEW])
    bundle = dict(schema="GOLD11_MESH_DEVELOPMENT_V1", config=CONFIG, limits=LIMITS,
                  cases=cases, source_sha256={name: digest(blob) for name, blob in blobs.items()},
                  catalog_sha256=CATALOG_SHA, selected_reference_sha256=SELECTED_SHA,
                  runtime_metadata=runtime_metadata(),
                  arms=["r0", "v1"], comparator="NEW_COMMON_WRAPPER_EXTRACTOR_CONTROL",
                  historical_comparability="UNESTABLISHED", production_use_allowed=False)
    RUN.mkdir(exist_ok=False)
    snapshot = RUN/"source"
    for name, blob in blobs.items():
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(blob)
    exclusive_json(RUN/"FROZEN.json", bundle)
    print(json.dumps(dict(status="FROZEN_NOT_EXECUTED", bundle_sha256=digest(encoded(bundle)),
                          attempt=str(RUN))), flush=True)


def verify_bundle():
    bundle = decode(bounded_read(RUN/"FROZEN.json", 256*1024))
    if bundle["config"] != CONFIG or bundle["limits"] != LIMITS or bundle["arms"] != ["r0", "v1"]:
        raise ValueError("FROZEN_PROTOCOL_CONSTANTS_CHANGED")
    if bundle.get("runtime_metadata") != runtime_metadata():
        raise ValueError("FROZEN_RUNTIME_METADATA_CHANGED")
    if set(bundle["source_sha256"]) != set(SOURCES):
        raise ValueError("SOURCE_DENOMINATOR_CHANGED")
    for name, sha in bundle["source_sha256"].items():
        bounded_read(ROOT/name, 2*1024**2, sha)
        bounded_read(RUN/"source"/name, 2*1024**2, sha)
    cases = approved_cases_from_review(bounded_read(RUN/"source"/REVIEW, 2*1024**2))
    if (bundle["cases"] != cases or bundle["catalog_sha256"] != CATALOG_SHA
            or bundle["selected_reference_sha256"] != SELECTED_SHA):
        raise ValueError("FROZEN_INPUT_OR_REFERENCE_SCOPE_CHANGED")
    return bundle


def contained(path, roots):
    return any(path == root or path.startswith(root+os.sep) for root in roots)


def runtime_roots():
    return (str((ROOT/".venv/lib/python3.11/site-packages").resolve()),
            str((Path(sys.base_prefix)/"lib/python3.11").resolve()))


def allowed_native_library(value, roots):
    # Importing ctypes itself initializes pythonapi with PyDLL(None).
    if value is None:
        return True
    if not isinstance(value, (str, bytes)):
        return False
    value = os.fsdecode(value)
    if not os.path.isabs(value):
        return False
    path = os.path.realpath(value)
    system = {"/usr/lib/libSystem.B.dylib", "/usr/lib/libc.dylib"}
    return path in system or (contained(path, roots) and path.endswith((".so", ".dylib")))


def worker_audit_hook(snapshot, cache, package="backend"):
    runtime = runtime_roots()
    source = str((snapshot/package).resolve())
    cache = str(cache.resolve())
    allowed_roots, writable = (source, cache, *runtime), (cache,)

    def descriptor_path(fd):
        if sys.platform != "darwin":
            raise PermissionError("WORKER_DESCRIPTOR_PLATFORM_UNSUPPORTED")
        # Darwin F_GETPATH=50, PATH_MAX=1024. Needed for Python's fd-safe
        # TemporaryDirectory/rmtree cleanup; never admit a descriptor blindly.
        value = fcntl.fcntl(fd, 50, bytes(1024)).split(b"\0", 1)[0]
        if not value or not os.path.isabs(value):
            raise PermissionError("WORKER_DESCRIPTOR_PATH_UNAVAILABLE")
        return os.path.realpath(os.fsdecode(value))

    def checked_path(value, roots, dir_fd=None):
        if isinstance(value, int):
            path = descriptor_path(value)
        else:
            value = os.fsdecode(value)
            if dir_fd not in (None, -1) and not os.path.isabs(value):
                value = os.path.join(descriptor_path(dir_fd), value)
            path = os.path.realpath(value)
        if not contained(path, roots):
            raise PermissionError("WORKER_PATH_OUTSIDE_FROZEN_SCOPE")
        return path

    def audit(event, args):
        if event.startswith("socket.") or event in (
                "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.exec",
                "os.posix_spawn", "os.posix_spawnp", "os.setsid", "os.setpgid"):
            raise PermissionError("WORKER_EXTERNAL_ACTION_DENIED")
        if event == "ctypes.dlopen":
            if not allowed_native_library(args[0], runtime):
                raise PermissionError("WORKER_NATIVE_LIBRARY_OUTSIDE_RUNTIME")
        if event == "open" and not isinstance(args[0], int):
            mode, flags = args[1], args[2]
            writes = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                      or isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            path = os.path.realpath(os.fsdecode(args[0]))
            if not writes and path in ("/dev/null", "/dev/urandom"):
                return
            checked_path(args[0], writable if writes else allowed_roots)
        if event in ("os.listdir", "os.scandir"):
            checked_path(args[0], allowed_roots)
        if event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"):
            position = {"os.remove": 1, "os.unlink": 1, "os.rmdir": 1,
                        "os.mkdir": 2, "os.chmod": 2, "os.utime": 3, "os.truncate": 99}[event]
            checked_path(args[0], writable, args[position] if len(args) > position else None)
        if event in ("os.rename", "os.link"):
            checked_path(args[0], writable, args[2] if len(args) > 2 else None)
            checked_path(args[1], writable, args[3] if len(args) > 3 else None)
        if event == "os.symlink":
            # No symlink creation is part of input/STEP/cache preparation.
            raise PermissionError("WORKER_SYMLINK_DENIED")
    return audit


def install_namespace(name, directory):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory.resolve())]
    sys.modules[name] = module


def initialize_worker(snapshot, cache, package, cpu_seconds):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds,)*2)
    sys.path.insert(0, runtime_roots()[0])
    install_namespace(package, snapshot/package)
    # This guards Python-level accidental I/O, not native OS syscalls or hostile
    # extension code. The frozen native import/reader source is reviewed too.
    sys.addaudithook(worker_audit_hook(snapshot, cache, package))


_COORDINATOR_CONTEXT = None
LEAF_SCHEMA = "ONE_USE_PARENT_BOUND_LEAF_V1"
RUNNER_SOURCE = "scripts/run_gold11_mesh_development_v1.py"


def is_hex(value, length):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def leaf_sources(role):
    if role == "INFERENCE":
        return (RUNNER_SOURCE, *(name for name in SOURCES if name.startswith("backend/")))
    if role == "SCORER":
        return (RUNNER_SOURCE, "scripts/gold11_mesh_scoring_v1.py",
                "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py")
    raise ValueError("UNKNOWN_LEAF_ROLE")


def activate_coordinator(bundle, audit, expected, start):
    """Publish only neutral proof after the full coordinator gate succeeds.

    Leaves never open FROZEN/review source: those include case identities. The
    actual parent is the coordinator that verified them, not a caller-selected
    source of authority. Exclusive context creation also makes dispatch one-use.
    This is a fail-closed workflow gate, not protection from a hostile local user
    rewriting controls or replacing the Python executable.
    """
    global _COORDINATOR_CONTEXT
    context = dict(schema=LEAF_SCHEMA, coordinator_pid=os.getpid(), outer_pid=os.getppid(),
        bundle_sha256=digest(encoded(bundle)), audit_sha256=digest(encoded(audit)),
        started_sha256=digest(encoded(start)), coordinator_dispatch_sha256=digest(encoded(expected)),
        source_sha256=bundle["source_sha256"], runtime_metadata=bundle["runtime_metadata"])
    exclusive_json(RUN/"COORDINATOR_VERIFIED.json", context)
    (RUN/"leaf_dispatch").mkdir(exist_ok=False)
    _COORDINATOR_CONTEXT = context


def cache_identity(cache):
    cache = Path(cache)
    info = cache.lstat()
    if (not cache.is_absolute() or cache != cache.resolve() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError("LEAF_CACHE_NOT_PRIVATE_OWNED_DIRECTORY")
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid)


def issue_leaf_dispatch(role, cache, **binding):
    context = _COORDINATOR_CONTEXT
    if (type(context) is not dict or context.get("coordinator_pid") != os.getpid()
            or context.get("outer_pid") != os.getppid() or os.getpgrp() != os.getpid()):
        raise ValueError("VERIFIED_COORDINATOR_REQUIRED_TO_ISSUE_LEAF")
    validate_leaf_binding(role, binding)
    # This neutral digest is unique per scan/CAD/arm, not a case or family ID.
    # Exclusive issuance prevents retries even when a worker never starts.
    dispatch_id = digest(encoded(dict(role=role, binding=binding)))
    token = secrets.token_hex(32)
    record = dict(schema=LEAF_SCHEMA, role=role, binding=binding, cache=str(cache),
        cache_identity=cache_identity(cache), coordinator_pid=os.getpid(),
        context_sha256=digest(encoded(context)), token_sha256=digest(token.encode()))
    exclusive_json(RUN/"leaf_dispatch"/(dispatch_id+".json"), record)
    return dict(dispatch_id=dispatch_id, token=token)


def validate_leaf_binding(role, binding):
    expected = {"arm", "scan_sha256", "cad_sha256"} if role == "INFERENCE" else {"request_sha256"}
    if role not in ("INFERENCE", "SCORER") or type(binding) is not dict or set(binding) != expected:
        raise ValueError("LEAF_BINDING_SCHEMA")
    if role == "INFERENCE" and binding["arm"] not in ("r0", "v1"):
        raise ValueError("LEAF_ARM_SCHEMA")
    if any(not is_hex(value, 64) for key, value in binding.items() if key != "arm"):
        raise ValueError("LEAF_HASH_SCHEMA")


def authorize_leaf(request, role):
    # Reject the former arbitrary snapshot/cache/arm API before any file read.
    if (type(request) is not dict or set(request) != {"dispatch_id", "token"}
            or not is_hex(request["dispatch_id"], 64) or not is_hex(request["token"], 64)):
        raise ValueError("COORDINATOR_LEAF_DISPATCH_REQUIRED")
    paths = leaf_sources(role)
    record = decode(bounded_read(RUN/"leaf_dispatch"/(request["dispatch_id"]+".json"), 16384))
    context = decode(bounded_read(RUN/"COORDINATOR_VERIFIED.json", 32768))
    expected = decode(bounded_read(RUN/"COORDINATOR_DISPATCH.json", 4096))
    start = decode(bounded_read(RUN/"STARTED.json", 4096))
    if (record.get("schema") != LEAF_SCHEMA or context.get("schema") != LEAF_SCHEMA
            or record.get("role") != role or record.get("coordinator_pid") != os.getppid()
            or context.get("coordinator_pid") != os.getppid() or os.getpgrp() != os.getppid()
            or context.get("outer_pid") != expected.get("parent_pid")
            or record.get("context_sha256") != digest(encoded(context))
            or context.get("coordinator_dispatch_sha256") != digest(encoded(expected))
            or context.get("started_sha256") != digest(encoded(start))
            or start.get("status") != "STARTED_NO_RETRY" or start.get("limits") != LIMITS
            or context.get("bundle_sha256") != start.get("bundle_sha256")
            or context.get("audit_sha256") != start.get("audit_sha256")
            or not is_hex(start.get("bundle_sha256"), 64) or not is_hex(start.get("audit_sha256"), 64)
            or not secrets.compare_digest(record.get("token_sha256", ""), digest(request["token"].encode()))):
        raise ValueError("LEAF_DISPATCH_PARENT_OR_CONTROL_BINDING")
    binding = record.get("binding")
    validate_leaf_binding(role, binding)
    if request["dispatch_id"] != digest(encoded(dict(role=role, binding=binding))):
        raise ValueError("LEAF_DISPATCH_IDENTITY_BINDING")
    # Terminal runs cannot authorize a later process even if PIDs are reused.
    for name in ("COMPLETED.json", "COORDINATOR_COMPLETED.json", "SCORING_UNAVAILABLE.json"):
        if (RUN/name).exists():
            raise ValueError("LEAF_RUN_ALREADY_TERMINAL")
    if role == "SCORER":
        terminal = decode(bounded_read(RUN/"INFERENCE_TERMINAL.json", 32768))
        if (terminal.get("case_count"), terminal.get("arm_count"), terminal.get("terminal_record_count")) != (11, 2, 22):
            raise ValueError("SCORER_REQUIRES_ALL_22_TERMINALS")
    # Burn capability before source, runtime or input access; failed validation
    # after this point is terminal and never retried with the same dispatch.
    exclusive_json(RUN/"leaf_dispatch"/(request["dispatch_id"]+".CLAIMED.json"),
        dict(worker_pid=os.getpid(), coordinator_pid=os.getppid(), role=role,
             dispatch_sha256=digest(encoded(record)), no_retry=True))
    snapshot, cache = RUN/"source", Path(record["cache"])
    if (cache_identity(cache) != record.get("cache_identity") or Path.cwd() != cache
            or os.environ.get("TMPDIR") != str(cache)):
        raise ValueError("LEAF_CACHE_BINDING_CHANGED")
    if Path(__file__).resolve() != (snapshot/RUNNER_SOURCE).resolve():
        raise ValueError("LEAF_MUST_EXECUTE_FROZEN_RUNNER")
    sources = context.get("source_sha256")
    if type(sources) is not dict or set(sources) != set(SOURCES):
        raise ValueError("LEAF_FROZEN_SOURCE_DENOMINATOR")
    for name in paths:
        if not is_hex(sources[name], 64):
            raise ValueError("LEAF_FROZEN_SOURCE_HASH_SCHEMA")
        bounded_read(snapshot/name, 2*1024**2, sources[name])
    if context.get("runtime_metadata") != runtime_metadata():
        raise ValueError("LEAF_FROZEN_RUNTIME_METADATA_CHANGED")
    return snapshot, cache, binding


def worker():
    # Parent sends no Gold, case/family identity or original source path.
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["case_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "INFERENCE")
    initialize_worker(snapshot, cache, "backend", LIMITS["case_cpu_seconds"])
    from backend.cad_guided_mesh_offline_v1 import MeshWorkflowPolicyV1, inspect_cad_guided_mesh_v1
    from backend.cad_guided_global_registration_v1 import WholePartRegistrationPolicyV1
    from backend.cad_guided_native_features import NativeGuidedPolicy
    from backend.cad_guided_instances import SupportPolicy
    scan = bounded_read(cache/"scan.ply", LIMITS["max_scan_bytes"], request["scan_sha256"])
    cad = bounded_read(cache/"cad.step", LIMITS["max_cad_bytes"], request["cad_sha256"])
    result = inspect_cad_guided_mesh_v1(scan_bytes=scan, cad_bytes=cad,
        workflow_policy=MeshWorkflowPolicyV1(**CONFIG["workflow"], native_adapter_version=request["arm"]),
        registration_policy=WholePartRegistrationPolicyV1(**CONFIG["registration"]),
        measurement_policy=NativeGuidedPolicy(**CONFIG["measurement"]),
        instance_policy=SupportPolicy(**CONFIG["support"]), allow_ocp=True)
    blob = encoded(result)
    if len(blob) > LIMITS["case_result_bytes"]:
        raise ValueError("RESULT_BYTE_BUDGET")
    with (cache/"RESULT.json").open("xb") as stream:
        stream.write(blob)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(status="INFERENCE_TERMINAL", result_sha256=digest(blob),
                          worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
                          worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)


def process_table():
    ps = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
                        capture_output=True, text=True, timeout=1)
    if ps.returncode != 0 or not ps.stdout.strip():
        raise RuntimeError("RSS_MONITOR_UNAVAILABLE")
    table = {}
    for line in ps.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or not all(value.isdigit() for value in fields):
            raise RuntimeError("RSS_MONITOR_MALFORMED")
        pid, ppid, pgid, kib = map(int, fields)
        if pid in table:
            raise RuntimeError("RSS_MONITOR_DUPLICATE_PID")
        table[pid] = dict(ppid=ppid, pgid=pgid, rss=kib*1024)
    return table


def descendants(table, root_pid):
    found = {root_pid} if root_pid in table else set()
    while True:
        new = {pid for pid, row in table.items() if row["ppid"] in found} - found
        if not new:
            return found
        found.update(new)


def clean_environment(cache):
    return dict(PATH="/usr/bin:/bin", HOME=str(cache), TMPDIR=str(cache),
                XDG_CACHE_HOME=str(cache), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def runner_command(mode):
    return [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
            str(RUN/"source/scripts/run_gold11_mesh_development_v1.py"), mode]


def memory_budget(tree_rss, parent_rss, rss_limit, *, include_parent=False, parent_rss_limit=None):
    if any(type(value) is not int or value < 0 for value in (tree_rss, parent_rss, rss_limit)):
        raise ValueError("INVALID_RSS_SAMPLE")
    measured = tree_rss + parent_rss if include_parent else tree_rss
    if measured > rss_limit:
        return measured, "GLOBAL_RSS_INCLUDING_SUPERVISOR_LIMIT" if include_parent else "TREE_RSS_LIMIT"
    if parent_rss_limit is not None and parent_rss > parent_rss_limit:
        return measured, "COORDINATOR_RSS_LIMIT"
    return measured, None


def supervise_process(command, request, *, cache, log_path, wall_seconds, rss_limit,
                      log_limit, new_session=False, parent_rss_limit=None):
    """One child attempt; monitor failures are terminal, never RSS=0 success.

    Only the outer coordinator starts a new group. Its guarded workers inherit
    that group, allowing atomic whole-run termination even between samples.
    Per-worker supervision measures/kills that worker's descendant tree. The
    global new-session budget includes this outer supervisor's own RSS too.
    """
    started, child, selector = time.monotonic(), None, selectors.DefaultSelector()
    receipt = dict(exit_code=None, budget_stop=None, sampled_tree_peak_rss_bytes=0,
                   sampled_parent_peak_rss_bytes=0, sampled_budget_peak_rss_bytes=0,
                   rss_budget_scope="CHILD_TREE_PLUS_OUTER_SUPERVISOR" if new_session else "CHILD_TREE",
                   forwarded_log_bytes=0)
    known_pids = set()

    def kill():
        if child is None:
            return
        if new_session:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            for pid in sorted(known_pids | {child.pid}, reverse=True):
                try:
                    # Do not signal a reused PID that belongs to another task.
                    if os.getpgid(pid) == os.getpgrp():
                        os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        child.wait(timeout=5)

    try:
        request_blob = encoded(request)
        if len(request_blob) > 16384:
            raise ValueError("WORKER_DISPATCH_BYTE_BUDGET")
        child = subprocess.Popen(command, cwd=cache, env=clean_environment(cache), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=new_session)
        child.stdin.write(request_blob)
        child.stdin.close()
        selector.register(child.stdout, selectors.EVENT_READ)
        with log_path.open("xb") as log:
            while True:
                for key, _ in selector.select(timeout=.05):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        keep = max(0, log_limit-receipt["forwarded_log_bytes"])
                        log.write(block[:keep])
                        log.flush()
                        receipt["forwarded_log_bytes"] += len(block)
                if receipt["forwarded_log_bytes"] > log_limit:
                    receipt["budget_stop"] = "LOG_BYTE_LIMIT"
                if time.monotonic()-started > wall_seconds:
                    receipt["budget_stop"] = "WALL_LIMIT"
                done = child.poll() is not None
                if not done or new_session:
                    try:
                        table = process_table()
                        tree = descendants(table, child.pid)
                        # Child may exit between poll and ps; do not turn a
                        # legitimate completed process into a monitor failure.
                        if not tree and child.poll() is None:
                            raise RuntimeError("ACTIVE_WORKER_RSS_MISSING")
                        known_pids = tree
                        rss = sum(table[pid]["rss"] for pid in tree)
                        parent_rss = table.get(os.getpid(), {}).get("rss")
                        if parent_rss is None:
                            raise RuntimeError("PARENT_RSS_MISSING")
                        receipt["sampled_tree_peak_rss_bytes"] = max(receipt["sampled_tree_peak_rss_bytes"], rss)
                        receipt["sampled_parent_peak_rss_bytes"] = max(receipt["sampled_parent_peak_rss_bytes"], parent_rss)
                        measured, stop = memory_budget(rss, parent_rss, rss_limit,
                            include_parent=new_session, parent_rss_limit=parent_rss_limit)
                        receipt["sampled_budget_peak_rss_bytes"] = max(receipt["sampled_budget_peak_rss_bytes"], measured)
                        if stop is not None:
                            receipt["budget_stop"] = stop
                    except Exception as error:
                        if new_session or child.poll() is None:
                            receipt["budget_stop"] = "RSS_MONITOR_FAILURE:" + type(error).__name__
                if receipt["budget_stop"]:
                    kill()
                    break
                if done and not selector.get_map():
                    break
    except Exception as error:
        receipt["budget_stop"] = "SUPERVISOR_FAILURE:" + type(error).__name__
        kill()
    finally:
        selector.close()
        if child is not None:
            if child.poll() is None:
                kill()
            receipt["exit_code"] = child.returncode
            if child.stdout is not None:
                child.stdout.close()
        receipt["wall_seconds"] = time.monotonic()-started
    return receipt


def coordinator_memory_check():
    table = process_table()
    rss = table.get(os.getpid(), {}).get("rss")
    if (rss is None or rss > LIMITS["coordinator_rss_bytes"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > LIMITS["coordinator_rss_bytes"]):
        raise RuntimeError("COORDINATOR_MEMORY_UNAVAILABLE_OR_OVER_LIMIT")


def supervise_case(case, arm, destination, remaining_wall):
    started = time.monotonic()
    outcome = unavailable("INFERENCE_NOT_STARTED")
    receipt = dict(status="TERMINAL", budget_stop=None, source_binding_valid=False,
                   inference_started=False)
    try:
        destination.mkdir(parents=True, exist_ok=False)
        exclusive_json(destination/"STARTED.json", dict(case_id=case["case_token"], arm=arm,
                       limits=LIMITS, no_resume=True))
        coordinator_memory_check()
        if remaining_wall < LIMITS["case_wall_seconds"]:
            raise ValueError("TOTAL_WALL_RESERVE_UNAVAILABLE")
        with deadline(min(LIMITS["parent_read_seconds"], remaining_wall)):
            scan = _bounded_read(case["scan_path"], LIMITS["max_scan_bytes"], case["scan_sha256"])
            cad = _bounded_read(case["cad_path"], LIMITS["max_cad_bytes"], case["cad_sha256"])
        receipt["source_binding_valid"] = True
        with tempfile.TemporaryDirectory(prefix="gold11-mesh-v1-") as temporary:
            cache = Path(temporary).resolve()
            with deadline(LIMITS["parent_read_seconds"]):
                for name, blob in (("scan.ply", scan), ("cad.step", cad)):
                    with (cache/name).open("xb") as stream:
                        stream.write(blob)
            del scan, cad
            receipt["inference_started"] = True
            request = issue_leaf_dispatch("INFERENCE", cache, arm=arm,
                           scan_sha256=case["scan_sha256"], cad_sha256=case["cad_sha256"])
            receipt["leaf_dispatch_id"] = request["dispatch_id"]
            worker_receipt = supervise_process(runner_command("--worker"), request, cache=cache,
                log_path=destination/"OUTPUT.txt", wall_seconds=max(.01, LIMITS["case_wall_seconds"]-(time.monotonic()-started)),
                rss_limit=LIMITS["sampled_worker_tree_rss_bytes"], log_limit=LIMITS["case_log_bytes"],
                parent_rss_limit=LIMITS["coordinator_rss_bytes"])
            receipt.update(worker_receipt)
            if receipt["exit_code"] != 0 or receipt["budget_stop"]:
                raise ValueError("INFERENCE_WORKER_FAILED")
            result_blob = bounded_read(cache/"RESULT.json", LIMITS["case_result_bytes"])
            receipt["result_sha256"] = digest(result_blob)
            with deadline(LIMITS["parent_read_seconds"]):
                with (destination/"RESULT.json").open("xb") as stream:
                    stream.write(result_blob)
                outcome = compact_result(decode(result_blob))
            del result_blob
            coordinator_memory_check()
    except Exception as error:
        outcome = unavailable(str(error)[:160] if isinstance(error, (ValueError, TimeoutError))
                              else "SUPERVISOR_EXCEPTION:"+type(error).__name__)
        receipt["failure_type"] = type(error).__name__
    receipt["parent_wall_seconds"] = time.monotonic()-started
    # Storage failures propagate to the coordinator/outer terminal-fill path;
    # they must not claim a durable outcome that could not actually be written.
    destination.mkdir(parents=True, exist_ok=True)
    exclusive_json(destination/"SCORING_INPUT.json", outcome)
    exclusive_json(destination/"COMPLETED.json", receipt)
    print(json.dumps(dict(case_id=case["case_token"], arm=arm, status=outcome.get("status", "MALFORMED"),
                          phase=outcome.get("phase"), wall_seconds=receipt["parent_wall_seconds"],
                          budget_stop=receipt["budget_stop"])), flush=True)
    return outcome, receipt


def selected_references(cases):
    blob = bounded_read(CATALOG, 8*1024**2, CATALOG_SHA)
    payload = decode(blob)
    by_sha = {case["scan_sha256"]: case for case in cases}
    selected = {}
    for row in payload["cases"]:
        case = by_sha.get(row.get("scan_sha256"))
        if case is None:
            continue
        key = case["case_token"]
        if (key in selected or row.get("case_id") != case["reference_id"]
                or row.get("scan_path") != case["scan_path"]
                or str(row.get("part_id")) != case["part_key"]):
            raise ValueError("REFERENCE_IDENTITY_OR_DUPLICATE")
        selected[key] = row
    if set(selected) != {case["case_token"] for case in cases}:
        raise ValueError("REFERENCE_CASE_DENOMINATOR")
    if sum(len(row["bends"]) for row in selected.values()) != 71 or digest(encoded(selected)) != SELECTED_SHA:
        raise ValueError("REFERENCE_SNAPSHOT_MISMATCH")
    return selected


def verify_pre_run_audit(bundle):
    audit = decode(bounded_read(RUN/"PRE_RUN_AUDIT.json", 256*1024))
    if not (audit.get("decision") == "GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN"
            and audit.get("bundle_sha256") == digest(encoded(bundle))
            and isinstance(audit.get("reviewer"), str) and audit["reviewer"].strip()
            and audit.get("blocking_findings") == []):
        raise ValueError("EXACT_INDEPENDENT_PRE_RUN_AUDIT_REQUIRED")
    return audit


def terminalize_case_records(bundle, reason):
    """Fill missing terminal records only; never rerun/replace an existing arm."""
    records = []
    for case in bundle["cases"]:
        for arm in bundle["arms"]:
            destination = RUN/"cases"/case["case_token"]/arm
            row = dict(case_id=case["case_token"], arm=arm, durable=False)
            try:
                destination.mkdir(parents=True, exist_ok=True)
                for name, payload in (
                    ("SCORING_INPUT.json", unavailable(reason, "GLOBAL_TERMINALIZATION")),
                    ("COMPLETED.json", dict(status="TERMINAL", reason=reason, no_resume=True,
                         inference_started="UNKNOWN_IF_STARTED_WITHOUT_RECEIPT"))):
                    try:
                        exclusive_json(destination/name, payload)
                    except FileExistsError:
                        # Check an existing regular bounded record, do not trust
                        # a symlink/partial write as a durable terminal record.
                        prior = decode(bounded_read(destination/name, LIMITS["compact_case_bytes"]))
                        if type(prior) is not dict or type(prior.get("status")) is not str:
                            raise ValueError("MALFORMED_EXISTING_TERMINAL_RECORD")
                        if name == "COMPLETED.json" and prior["status"] != "TERMINAL":
                            raise ValueError("NONTERMINAL_EXISTING_CASE_RECEIPT")
                row["durable"] = True
            except Exception as error:
                row["storage_error"] = type(error).__name__
            records.append(row)
    return records


def scorer_worker():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["scorer_cpu_seconds"],)*2)
    request = decode(sys.stdin.buffer.read(16384))
    snapshot, cache, request = authorize_leaf(request, "SCORER")
    initialize_worker(snapshot, cache, "scripts", LIMITS["scorer_cpu_seconds"])
    payload = decode(bounded_read(cache/"SCORING_REQUEST.json", LIMITS["scorer_request_bytes"],
                                 request["request_sha256"]))
    from scripts.gold11_mesh_scoring_v1 import score_gold11_mesh_workflows_v1
    result = score_gold11_mesh_workflows_v1(**payload)
    blob = encoded(result)
    if len(blob) > LIMITS["scorer_result_bytes"]:
        raise ValueError("SCORER_RESULT_BYTE_BUDGET")
    with (cache/"SCORER_RESULT.json").open("xb") as stream:
        stream.write(blob)
    print(json.dumps(dict(status="SCORER_TERMINAL", result_sha256=digest(blob))), flush=True)


def score_after_terminal(bundle, outcomes):
    # Only this post-terminal coordinator function can open the fixed catalog.
    verify_bundle()
    selected = selected_references(bundle["cases"])
    request = dict(reference_records_by_case={key: row["bends"] for key, row in selected.items()},
                   control_results_by_case=outcomes["r0"], challenger_results_by_case=outcomes["v1"],
                   case_family_by_id={case["case_token"]: case["part_key"] for case in bundle["cases"]})
    blob = encoded(request)
    if len(blob) > LIMITS["scorer_request_bytes"]:
        raise ValueError("SCORER_REQUEST_BYTE_BUDGET")
    with tempfile.TemporaryDirectory(prefix="gold11-score-v1-") as temporary:
        cache = Path(temporary).resolve()
        with deadline(LIMITS["parent_read_seconds"]):
            with (cache/"SCORING_REQUEST.json").open("xb") as stream:
                stream.write(blob)
        dispatch = issue_leaf_dispatch("SCORER", cache, request_sha256=digest(blob))
        receipt = supervise_process(runner_command("--score-worker"), dispatch,
            cache=cache, log_path=RUN/"SCORER_OUTPUT.txt", wall_seconds=LIMITS["scorer_wall_seconds"],
            rss_limit=LIMITS["scorer_rss_bytes"], log_limit=LIMITS["case_log_bytes"],
            parent_rss_limit=LIMITS["coordinator_rss_bytes"])
        exclusive_json(RUN/"SCORER_RECEIPT.json", receipt)
        if receipt["exit_code"] != 0 or receipt["budget_stop"]:
            raise ValueError("SCORER_FAILED_OR_BUDGET_STOP")
        report = decode(bounded_read(cache/"SCORER_RESULT.json", LIMITS["scorer_result_bytes"]))
    # Bind the source/runtime AFTER scorer execution, not only before import.
    verify_bundle()
    bounded_read(CATALOG, 8*1024**2, CATALOG_SHA)
    report.update(bundle_sha256=digest(encoded(bundle)), reference_snapshot_sha256=SELECTED_SHA,
                  historical_score_beaten_claim_allowed=False, source_before_after_consistent=True,
                  case_declared_coordinate_frames={key: row.get("coordinate_frame")
                                                   for key, row in selected.items()},
                  declared_frame_interpretation="PROVISIONAL_COMMON_RAW_FRAME_NOT_VERIFIED")
    exclusive_json(RUN/"SCORED.json", report)


def internal_coordinator():
    dispatch = decode(sys.stdin.buffer.read(4096))
    bundle = verify_bundle()
    audit = verify_pre_run_audit(bundle)
    expected = decode(bounded_read(RUN/"COORDINATOR_DISPATCH.json", 4096))
    start = decode(bounded_read(RUN/"STARTED.json", 4096))
    if (type(dispatch.get("token")) is not str or len(dispatch["token"]) != 64
            or digest(dispatch["token"].encode()) != expected.get("token_sha256")
            or dispatch.get("parent_pid") != os.getppid() or expected.get("parent_pid") != os.getppid()
            or os.getpgrp() != os.getpid() or start.get("bundle_sha256") != digest(encoded(bundle))
            or start.get("audit_sha256") != digest(encoded(audit))
            or start.get("status") != "STARTED_NO_RETRY" or start.get("limits") != LIMITS):
        raise ValueError("INTERNAL_COORDINATOR_DISPATCH_REQUIRED")
    activate_coordinator(bundle, audit, expected, start)
    started = time.monotonic()
    outcomes = {arm: {} for arm in bundle["arms"]}
    receipts = {arm: {} for arm in bundle["arms"]}
    try:
        for case in bundle["cases"]:
            for arm in bundle["arms"]:
                outcome, receipt = supervise_case(case, arm, RUN/"cases"/case["case_token"]/arm,
                    LIMITS["total_wall_seconds"]-(time.monotonic()-started)-LIMITS["terminalization_reserve_seconds"])
                outcomes[arm][case["case_token"]] = outcome
                receipts[arm][case["case_token"]] = receipt
                coordinator_memory_check()
        exclusive_json(RUN/"INFERENCE_TERMINAL.json", dict(case_count=11, arm_count=2,
            terminal_record_count=sum(map(len, receipts.values())), outcomes_sha256=digest(encoded(outcomes)),
            result_bindings={arm: {case: row.get("result_sha256") for case, row in rows.items()}
                             for arm, rows in receipts.items()}, parent_wall_seconds=time.monotonic()-started))
        score_after_terminal(bundle, outcomes)
        status = "COMPLETE_DEVELOPMENT_ONLY"
    except Exception as error:
        terminal = terminalize_case_records(bundle, "COORDINATOR_ABORT_NO_RETRY")
        exclusive_json(RUN/"SCORING_UNAVAILABLE.json", dict(error_type=type(error).__name__,
                       reason=str(error)[:160], no_inference_retry=True, terminal_case_records=terminal))
        status = "TERMINAL_SCORING_UNAVAILABLE"
    exclusive_json(RUN/"COORDINATOR_COMPLETED.json", dict(status=status,
                   parent_wall_seconds=time.monotonic()-started, no_resume=True,
                   historical_comparability="UNESTABLISHED", production_use_allowed=False))
    print(json.dumps(dict(status=status, attempt=str(RUN))), flush=True)


def execute():
    # Exclusive STARTED plus independent exact-bundle review remains necessary.
    # The internal CLI mode cannot substitute for this authorization/one-shot.
    bundle = verify_bundle()
    audit = verify_pre_run_audit(bundle)
    exclusive_json(RUN/"STARTED.json", dict(bundle_sha256=digest(encoded(bundle)),
                   audit_sha256=digest(encoded(audit)), status="STARTED_NO_RETRY", limits=LIMITS))
    started, token = time.monotonic(), secrets.token_hex(32)
    exclusive_json(RUN/"COORDINATOR_DISPATCH.json", dict(parent_pid=os.getpid(), token_sha256=digest(token.encode())))
    with tempfile.TemporaryDirectory(prefix="gold11-coordinate-v1-") as temporary:
        receipt = supervise_process(runner_command("--internal-coordinator"),
            dict(token=token, parent_pid=os.getpid()), cache=Path(temporary).resolve(),
            log_path=RUN/"COORDINATOR_OUTPUT.txt",
            wall_seconds=LIMITS["total_wall_seconds"]-LIMITS["terminalization_reserve_seconds"],
            rss_limit=LIMITS["global_tree_rss_bytes"], log_limit=LIMITS["coordinator_log_bytes"],
            new_session=True)
    with deadline(LIMITS["terminalization_reserve_seconds"]):
        terminal = terminalize_case_records(bundle, "GLOBAL_ABORT_OR_MISSING_TERMINAL_RECORD")
        durable = len(terminal) == 22 and all(row["durable"] for row in terminal)
        if receipt["exit_code"] == 0 and receipt["budget_stop"] is None:
            try:
                status = decode(bounded_read(RUN/"COORDINATOR_COMPLETED.json", 4096))["status"]
            except Exception:
                status = "TERMINAL_COORDINATOR_RECEIPT_UNAVAILABLE"
        else:
            status = "TERMINAL_GLOBAL_BUDGET_OR_COORDINATOR_FAILURE"
        if not durable:
            status = "TERMINAL_STORAGE_FAILURE"
        exclusive_json(RUN/"COMPLETED.json", dict(status=status, coordinator_receipt=receipt,
            terminal_case_records=terminal, parent_wall_seconds=time.monotonic()-started,
            no_resume=True, production_use_allowed=False))
    print(json.dumps(dict(status=status, attempt=str(RUN), durable_terminal_case_records=sum(row["durable"] for row in terminal))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--score-worker", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--internal-coordinator", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        worker()
    elif args.score_worker:
        scorer_worker()
    elif args.internal_coordinator:
        internal_coordinator()
    else:
        execute()


if __name__ == "__main__":
    main()
