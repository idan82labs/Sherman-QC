"""Frozen invented-only V5 source candidate; exact independent PRE required.

Self-contained successor of the reviewed V4 custody supervisor mechanism.
The old supervisor is a byte-bound preservation oracle only, never imported.
No runner CLI, physical inputs, native calls or test-selection overrides.
Exact PRE/archive and concrete runner/test pins must precede source release.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FOR_SOURCE_GATE = True
RUNNER_FINAL_SHA256 = "71d55915db6e6fcb8ae7f8c658de7d7e0c4b93b553eb78413f3914b037e4e481"
TEST_FINAL_SHA256 = "dc988d8a1fc8526ce8f82ae9d904e9e945dd8d8f3c73700e470aeeae397d18ca"
EXPECTED_OUTCOMES = 85
CPU_SECONDS, WORK_SECONDS, WALL_SECONDS = 120, 175, 180
RSS_BYTES, OUTPUT_BYTES = 2*1024**3, 1024**2
SOURCE_BYTES, CONTROL_BYTES = 2*1024**2, 256*1024
TEST = "tests/test_gold11_mesh_development_runner_v5.py"
RUNNER = "scripts/run_gold11_mesh_development_v5.py"
PROTOCOL = "docs/research/2026-09-11_gold11_mesh_runner_v5_source_protocol.md"
OLD_SUPERVISOR = "scripts/check_gold11_mesh_v4_custody.py"
OLD_SUPERVISOR_SHA256 = "579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b"
TIMING_KEYS = ("case_wall_seconds", "case_cpu_seconds", "total_wall_seconds",
    "scorer_wall_seconds", "scorer_cpu_seconds", "parent_read_seconds",
    "terminalization_reserve_seconds")
SOURCE_FILES = (
    "scripts/run_gold11_mesh_development_v5.py", "scripts/gold11_mesh_scoring_v1.py",
    "scripts/gold11_owned_process_supervision_v1.py", "scripts/gold11_terminal_artifacts_v4.py",
    "scripts/evaluate_f1_against_manual_gold.py", "scripts/f1_physical_evaluation_contract_v2.py",
    "backend/cad_guided_mesh_offline_v1.py", "backend/cad_guided_mesh_input_v1.py",
    "backend/cad_guided_global_registration_v1.py", "backend/cad_guided_cad_sampling_v1.py",
    "backend/cad_guided_native_features_v1.py", "backend/cad_guided_native_features.py",
    "backend/cad_guided_local_measurement_v1.py", "backend/cad_guided_measurement.py",
    "backend/cad_guided_instances.py", "backend/cad_guided_mesh_offline_v2.py",
    "backend/cad_guided_mesh_offline_v8.py", "backend/cad_guided_cad_surface_v2.py",
    "backend/cad_guided_global_registration_v2.py", "backend/cad_guided_triangle_surface_v2.py",
    "backend/cad_guided_mesh_topology_v4.py", "backend/cad_guided_plane_initialization_v4.py",
    "backend/cad_guided_plane_initialization_v8.py", "backend/cad_guided_mesh_plane_proposals_v4.py",
    "backend/cad_guided_plane_hypotheses_v4.py", "backend/cad_guided_noisy_mesh_planes_v5.py",
    "backend/cad_guided_noisy_mesh_planes_v7.py", "backend/cad_guided_plane_search_v6.py",
    "backend/cad_guided_report_envelope_v6.py", "backend/cad_guided_mesh_offline_v10.py",
    "backend/cad_guided_plane_initialization_v10.py", "backend/cad_guided_global_registration_v3.py",
    "backend/cad_guided_triangle_surface_v3.py", "backend/cad_guided_triangle_distance_bounds_v3.py",
    "backend/cad_guided_face_box_screen_v1.py", "backend/cad_guided_mesh_offline_v11.py",
    "backend/cad_guided_plane_initialization_v11.py", "backend/cad_guided_two_plane_context_v1.py",
    "backend/cad_guided_two_plane_schedule_v1.py", "scripts/run_gold11_mesh_development_v3.py",
    "scripts/run_gold11_mesh_development_v2.py", "tests/test_gold11_mesh_development_runner_v5.py",
    "scripts/run_gold11_mesh_development_v4.py", "scripts/gold11_contact_scoring_v2.py",
    "scripts/gold11_schedule_projection_v1.py", "backend/cad_guided_composite_surface_roles_v1.py",
    "backend/cad_guided_context_measurement_v1.py", "backend/cad_guided_context_surface_roles_v1.py",
    "backend/cad_guided_exact_support_cover_v1.py", "backend/cad_guided_mesh_offline_v12.py",
    "backend/cad_guided_mesh_offline_v13.py", "backend/cad_guided_plane_initialization_v12.py",
    "backend/cad_guided_scan_plane_preparation_v1.py", "backend/cad_guided_seeded_plane_proposals_v1.py",
    "backend/cad_guided_stage_policies_v1.py", "backend/cad_guided_support_cover_measurement_v2.py",
    "backend/cad_guided_two_plane_schedule_v2.py", "tests/__init__.py",
    "backend/cad_guided_two_plane_context_v2.py", "scripts/review_gold11_inputs_v1.py",
    "docs/research/2026-09-11_gold11_mesh_development_v5_protocol.md",
    "tests/test_gold11_contact_scoring_v2.py", "tests/test_gold11_schedule_projection_v1.py",
    "scripts/check_gold11_mesh_runner_v5.py",
    "docs/research/2026-09-11_gold11_mesh_runner_v5_source_protocol.md",
    "scripts/check_gold11_mesh_v4_custody.py",
)


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_frozen():
    # First operation in both parent/worker entry. No environment bypass.
    if FROZEN_FOR_SOURCE_GATE is not True:
        raise ValueError("V5_RUNNER_SOURCE_GATE_NOT_FROZEN")
    if not _hex(RUNNER_FINAL_SHA256) or not _hex(TEST_FINAL_SHA256):
        raise ValueError("V5_RUNNER_FINAL_SOURCE_AND_TEST_PINS_REQUIRED")
    if sys.platform != "darwin":
        raise ValueError("SOURCE_GATE_RUSAGE_BYTE_UNITS_REQUIRE_REVIEWED_DARWIN_RUNTIME")
    constants = (EXPECTED_OUTCOMES, CPU_SECONDS, WORK_SECONDS, WALL_SECONDS, RSS_BYTES, OUTPUT_BYTES)
    if any(type(value) is not int for value in constants) or constants != (85, 120, 175, 180, 2*1024**3, 1024**2):
        raise ValueError("V5_RUNNER_SOURCE_GATE_LITERAL_LIMITS_CHANGED")


def _remaining(deadline, local_cap):
    value = deadline-time.monotonic()
    if value <= 0:
        raise TimeoutError("SOURCE_GATE_ABSOLUTE_DEADLINE")
    return min(local_cap, value)


def _no_root_initializer():
    if os.path.lexists(ROOT/"__init__.py"):
        raise ValueError("UNDECLARED_ENCLOSING_REPOSITORY_INITIALIZER")


def _directory_identity(path, deadline):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in Path(path).parts[1:]:
            _remaining(deadline, 1)
            following = os.open(part, flags, dir_fd=fd)
            old, fd = fd, following
            os.close(old)
        info = os.fstat(fd)
        return info.st_dev, info.st_ino
    finally:
        os.close(fd)


def _read_file(path, cap, deadline, *, allow_empty=False):
    """Bounded regular file read via no-follow directory components.

    Rechecks the same parent entry and descriptor identity. This is quiescent
    source custody, not an atomic adversarial namespace snapshot or hard I/O timer.
    """
    path = Path(path)
    if not path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise ValueError("ABSOLUTE_FIXED_PATH_REQUIRED")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory, fd = None, None
    try:
        _remaining(deadline, 1)
        directory = os.open("/", flags)
        for part in path.parts[1:-1]:
            _remaining(deadline, 1)
            following = os.open(part, flags, dir_fd=directory)
            old, directory = directory, following
            os.close(old)
        _remaining(deadline, 1)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not int(not allow_empty) <= before.st_size <= cap:
            raise ValueError("REGULAR_FILE_TYPE_OR_BYTE_BUDGET")
        blob = bytearray()
        while True:
            _remaining(deadline, 1)
            block = os.read(fd, min(65536, cap-len(blob)+1))
            if not block:
                break
            blob.extend(block)
            if len(blob) > cap:
                raise ValueError("FILE_BYTE_BUDGET")
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        if (identity(before) != identity(after) or len(blob) != before.st_size
                or identity(os.stat(path.name, dir_fd=directory, follow_symlinks=False)) != identity(after)):
            raise ValueError("FILE_CHANGED_DURING_READ")
        return bytes(blob)
    finally:
        if fd is not None:
            os.close(fd)
        if directory is not None:
            os.close(directory)


def _sources(deadline, *, keep_preflight=False):
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or len(SOURCE_FILES) != 66:
        raise ValueError("FIXED_66_SOURCE_DENOMINATOR_REQUIRED")
    hashes, blobs, errors, unattempted = {}, {}, {}, []
    for index, name in enumerate(SOURCE_FILES):
        try:
            blob = _read_file(ROOT/name, SOURCE_BYTES, deadline)
            hashes[name] = hashlib.sha256(blob).hexdigest()
            if keep_preflight and name in (RUNNER, TEST, "tests/__init__.py", OLD_SUPERVISOR):
                blobs[name] = blob
        except TimeoutError:
            errors[name] = "TimeoutError"
            unattempted = list(SOURCE_FILES[index+1:])
            break
        except Exception as error:
            errors[name] = type(error).__name__
    return dict(hashes=hashes, errors=errors, unattempted=unattempted), blobs


def _preflight(blobs, snapshot):
    if snapshot["errors"] or snapshot["unattempted"] or set(snapshot["hashes"]) != set(SOURCE_FILES):
        raise ValueError("COMPLETE_SOURCE_PREFLIGHT_REQUIRED")
    if (snapshot["hashes"][RUNNER] != RUNNER_FINAL_SHA256
            or snapshot["hashes"][TEST] != TEST_FINAL_SHA256
            or snapshot["hashes"][OLD_SUPERVISOR] != OLD_SUPERVISOR_SHA256):
        raise ValueError("FINAL_RUNNER_TEST_OR_SUPERVISOR_SOURCE_PIN_CHANGED")
    # No import/compile-exec of any runner or preservation oracle here.
    def assignments(blob):
        return {target.id: row.value for row in ast.parse(blob).body if isinstance(row, ast.Assign)
            for target in row.targets if isinstance(target, ast.Name)}
    runner, test = assignments(blobs[RUNNER]), assignments(blobs[TEST])
    if ast.literal_eval(runner["RELEASE_AUTHORIZED"]) is not True:
        raise ValueError("QUALIFY_FINAL_RUNNER_DECLARATION_NOT_TEMPORARY_FALSE_BIT")
    if ast.literal_eval(test["RUNNER_SHA"]) != RUNNER_FINAL_SHA256:
        raise ValueError("TEST_MUST_PIN_FINAL_CONCRETE_RUNNER")
    limits = runner["LIMITS"]
    if not isinstance(limits, ast.Call) or not isinstance(limits.func, ast.Name) or limits.func.id != "dict":
        raise ValueError("RUNNER_LITERAL_LIMIT_DECLARATION")
    values = {row.arg: ast.literal_eval(row.value) for row in limits.keywords if row.arg in TIMING_KEYS}
    if set(values) != set(TIMING_KEYS) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("RUNNER_NATIVE_DERIVED_CONCRETE_TIMINGS_REQUIRED")
    if (type(values["case_cpu_seconds"]) is not int or type(values["scorer_cpu_seconds"]) is not int
            or values["terminalization_reserve_seconds"] >= values["total_wall_seconds"]):
        raise ValueError("RUNNER_CPU_OR_RESERVE_DECLARATION")
    declared = runner["SOURCES"]
    if not isinstance(declared, ast.Tuple):
        raise ValueError("RUNNER_LITERAL_SOURCE_DECLARATION")
    names = [ast.literal_eval(runner[row.id]) if isinstance(row, ast.Name) else ast.literal_eval(row)
        for row in declared.elts]
    extra = {"tests/test_gold11_contact_scoring_v2.py", "tests/test_gold11_schedule_projection_v1.py",
        "scripts/check_gold11_mesh_runner_v5.py", PROTOCOL, OLD_SUPERVISOR}
    if len(names) != len(set(names)) or len(names) != 61 or set(SOURCE_FILES) != set(names) | extra:
        raise ValueError("RUNNER_PLUS_TEST_AND_GATE_CLOSURE_MISMATCH")
    if any(not isinstance(row, ast.Expr) or not isinstance(row.value, ast.Constant)
            or type(row.value.value) is not str for row in ast.parse(blobs["tests/__init__.py"]).body):
        raise ValueError("TEST_PACKAGE_INITIALIZER_MUST_REMAIN_INERT")


def _group_sample(pgid, deadline):
    sampled = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="], capture_output=True,
        text=True, timeout=_remaining(deadline, 2))
    rows = [line.split() for line in sampled.stdout.splitlines()]
    if sampled.returncode != 0 or not rows or any(len(row) != 3 or not all(v.isdigit() for v in row) for row in rows):
        raise RuntimeError("SOURCE_GATE_PROCESS_TABLE_INVALID")
    if len({int(row[0]) for row in rows}) != len(rows):
        raise RuntimeError("SOURCE_GATE_PROCESS_TABLE_DUPLICATE_PID")
    return [(int(pid), int(rss)*1024) for pid, group, rss in rows if int(group) == pgid]


def supervise_tests(command, *, env, retained, started):
    """Copied V4 lifecycle mechanism with this gate's explicit limits/ledger.

    Popen, pipe/log I/O and OS cleanup are synchronous. Deadlines bound waits and
    polling, not uninterruptible calls; this is not a hard real-time guarantee.
    Only the owned process group is signaled; no escaped-group discovery claim.
    """
    deadline, work_deadline = started+WALL_SECONDS, started+WORK_SECONDS
    cache = str(Path(tempfile.mkdtemp(prefix="gold11-runner-v5-source-private-")).resolve(strict=True))
    child_env = dict(env, HOME=cache, TMPDIR=cache, XDG_CACHE_HOME=cache)
    child = selector = None
    stopped = primary_error = None
    cleanup_errors, observed = [], set()
    code, peak, combined_peak, sample_count, forwarded, stored = None, 0, 0, 0, 0, 0
    stored_exact, empty = True, False
    phase_times = {}
    output = None
    try:
        _remaining(work_deadline, 1)
        output = (retained/"OUTPUT.txt").open("xb")
        child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
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
                forwarded += len(block)  # observed even if a subsequent write/flush fails
                selected = block[:max(0, OUTPUT_BYTES-stored)]
                try:
                    count = output.write(selected)
                    if type(count) is not int or not 0 <= count <= len(selected):
                        stored_exact = False
                        raise OSError("OUTPUT_WRITE_COUNT_UNAVAILABLE")
                    stored += count
                    if count != len(selected):
                        stored_exact = False
                        raise OSError("OUTPUT_SHORT_WRITE")
                    output.flush()
                except Exception:
                    stored_exact = False
                    raise
            group = _group_sample(child.pid, work_deadline)
            observed.update(pid for pid, _ in group)
            if group:
                sample_count += 1
                peak = max(peak, sum(rss for _, rss in group))
                combined_peak = max(combined_peak,
                    sum(rss for _, rss in group)+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            code = child.poll()
            if code is None and child.pid not in {pid for pid, _ in group}:
                raise RuntimeError("SOURCE_GATE_ACTIVE_CHILD_MISSING")
            if any(pid != child.pid for pid, _ in group):
                stopped = "UNEXPECTED_SOURCE_TEST_CHILD_PROCESS"
            elif combined_peak > RSS_BYTES:
                stopped = "COMBINED_GROUP_SAMPLE_AND_PARENT_HIGHWATER_RSS_LIMIT"
            elif forwarded > OUTPUT_BYTES:
                stopped = "OUTPUT_LIMIT"
            elif time.monotonic() >= work_deadline:
                stopped = "WORK_WALL_LIMIT"
            if stopped or code is not None and not selector.get_map():
                break
    except Exception as error:
        stopped = stopped or ("WORK_WALL_LIMIT" if time.monotonic() >= work_deadline else "SUPERVISION_FAILURE")
        primary_error = type(error).__name__+":"+str(error)[:256]
    finally:
        phase_times["work_stopped_seconds"] = time.monotonic()-started
        # Cleanup precedes selector/stream close so ordinary close failures
        # cannot bypass owned-child reaping. Every close error vetoes release.
        if child is not None:
            try:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=_remaining(deadline, 5))
                phase_times["reaped_seconds"] = time.monotonic()-started
                group = _group_sample(child.pid, deadline)
                if group and all(pid in observed for pid, _ in group):
                    os.killpg(child.pid, signal.SIGKILL)
                    while group:
                        time.sleep(_remaining(deadline, .02))
                        group = _group_sample(child.pid, deadline)
                if group:
                    cleanup_errors.append("OWNED_GROUP_NOT_EMPTY:"+",".join(str(pid) for pid, _ in group)[:256])
                else:
                    empty = True
                phase_times["group_checked_seconds"] = time.monotonic()-started
            except Exception as error:
                cleanup_errors.append(type(error).__name__+":"+str(error)[:256])
        for label, stream in (("selector", selector), ("stdout", child.stdout if child is not None else None), ("output", output)):
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    cleanup_errors.append(label+":"+type(error).__name__)
                    if label == "output":
                        stored_exact = False
    elapsed = time.monotonic()-started
    if elapsed >= WALL_SECONDS:
        stopped = stopped or "TOTAL_WALL_LIMIT"
    if not empty:
        stopped = stopped or "SOURCE_GROUP_CUSTODY_UNCONFIRMED"
    return dict(exit_code=code, parent_wall_seconds=elapsed, budget_stop=stopped,
        primary_error=primary_error, cleanup_errors=cleanup_errors,
        child_pid=child.pid if child is not None else None, process_group_confirmed_empty=empty,
        observed_group_pids=sorted(observed), requires_group_attention=not empty and child is not None,
        sampled_group_peak_rss_bytes=peak if sample_count else None, group_rss_sample_count=sample_count,
        group_sample_plus_parent_highwater_peak_bytes=combined_peak if sample_count else None,
        rss_sample_availability="OBSERVED" if sample_count else "UNOBSERVED_NOT_ZERO",
        output_bytes_observed=forwarded, output_bytes_write_returns=stored, output_stored_bytes_exact=stored_exact,
        private_cache_retained_for_custody=cache, phase_times=phase_times,
        enforcement_scope="MONITORED_DEADLINES_AND_SAMPLED_GROUP_RSS_NOT_HARD_OS_IO_OR_ESCAPED_GROUP_SANDBOX")


class _Outcomes:
    def __init__(self):
        self.collected = self.passed = self.failed = self.skipped = self.errors = self.deselected = 0
        self.xfailed = self.xpassed = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors += 1

    def pytest_runtest_logreport(self, report):
        if getattr(report, "wasxfail", None) is not None:
            self.xfailed += int(report.skipped)
            self.xpassed += int(report.when == "call" and report.passed)
        if report.skipped:
            self.skipped += 1
        elif report.when == "call":
            self.passed += int(report.passed)
            self.failed += int(report.failed)
        elif report.failed:
            self.errors += 1


def _worker():
    _require_frozen()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    started = time.monotonic()
    _no_root_initializer()
    snapshot, blobs = _sources(started+WORK_SECONDS, keep_preflight=True)
    _preflight(blobs, snapshot)
    cache = Path(os.environ["TMPDIR"])
    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise ValueError("OWNED_PRIVATE_CACHE_REQUIRED")
    sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
    import pytest
    outcomes = _Outcomes()
    code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider", "-q", "-s",
        "--basetemp", str(cache/"pytest-invented"), str(ROOT/TEST)], plugins=[outcomes])
    _no_root_initializer()
    native_imported = any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(kind="V5_RUNNER_SOURCE_WORKER_RECEIPT", **vars(outcomes), pytest_exit_code=int(code),
        worker_cpu_seconds=usage.ru_utime+usage.ru_stime, worker_peak_rss_bytes=usage.ru_maxrss,
        native_ocp_imported=native_imported, repository_initializer_absent=True)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if (int(code) == 0 and outcomes.collected == outcomes.passed == EXPECTED_OUTCOMES
        and not (outcomes.failed or outcomes.skipped or outcomes.errors or outcomes.deselected
            or outcomes.xfailed or outcomes.xpassed or native_imported)) else 1


def _publish(path, value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n"
    if len(blob) > CONTROL_BYTES:
        raise ValueError("SOURCE_GATE_CONTROL_BYTE_BUDGET")
    with path.open("xb") as stream:
        if stream.write(blob) != len(blob):
            raise OSError("SOURCE_GATE_CONTROL_SHORT_WRITE")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def _attempt_inventory(retained, deadline):
    """Final three-file inventory, after stopped/empty worker; no unknown reads."""
    expected = {"STARTED.json": CONTROL_BYTES, "OUTPUT.txt": OUTPUT_BYTES, "COMPLETED.json": CONTROL_BYTES}
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open("/", flags)
    try:
        for part in retained.parts[1:]:
            _remaining(deadline, 1)
            following = os.open(part, flags, dir_fd=directory)
            old, directory = directory, following
            os.close(old)
        names = []
        scan = os.open(".", flags, dir_fd=directory)
        try:
            with os.scandir(scan) as entries:
                for entry in entries:
                    _remaining(deadline, 1)
                    names.append(entry.name)
                    if len(names) > 3:
                        raise ValueError("UNEXPECTED_FOURTH_SOURCE_GATE_ARTIFACT")
        finally:
            os.close(scan)
        if set(names) != set(expected) or len(names) != 3:
            raise ValueError("EXACT_THREE_SOURCE_GATE_ARTIFACTS_REQUIRED")
        result = {}
        for name, cap in expected.items():
            blob = _read_file(retained/name, cap, deadline, allow_empty=name == "OUTPUT.txt")
            result[name] = dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        return result
    finally:
        os.close(directory)


def main():
    _require_frozen()
    if sys.argv[1:] == ["--worker"]:
        return _worker()
    if sys.argv[1:]:
        raise ValueError("NO_TEST_SUBSET_BUDGET_OR_POLICY_OVERRIDES")
    started = time.monotonic()
    deadline = started+WALL_SECONDS
    _no_root_initializer()
    before, blobs = _sources(started+WORK_SECONDS, keep_preflight=True)
    _preflight(blobs, before)
    base = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    _directory_identity(base.parent, deadline)
    base.mkdir(exist_ok=True)
    base_identity = _directory_identity(base, deadline)
    retained = Path(tempfile.mkdtemp(prefix="gold11_mesh_runner_v5_source-", dir=base))
    if _directory_identity(base, deadline) != base_identity:
        raise ValueError("SOURCE_GATE_ATTEMPT_PARENT_REBOUND")
    _directory_identity(retained, deadline)
    limits = dict(worker_cpu_seconds=CPU_SECONDS, work_wall_seconds=WORK_SECONDS,
        parent_wall_seconds=WALL_SECONDS, sampled_group_rss_bytes=RSS_BYTES, output_bytes=OUTPUT_BYTES)
    started_file = _publish(retained/"STARTED.json", dict(status="STARTED_INVENTED_SOURCE_GATE_NO_RETRY",
        source_sha256_before=before["hashes"], expected_test_outcomes=EXPECTED_OUTCOMES,
        repository_initializer_absent_before=True, source_file_count=66, limits=limits,
        scope="INVENTED_119_FILE_CAMPAIGN_AND_CONSTRUCTORS_NO_PHYSICAL_OR_NATIVE_EXECUTION",
        attempted_directory_files=["STARTED.json", "OUTPUT.txt", "COMPLETED.json"],
        private_cache_scope="INVENTED_TEST_FILES_ONLY_SEPARATELY_RETAINED_NOT_SCIENTIFIC_ARTIFACTS"))
    print(json.dumps(dict(status="RUNNING_NOT_EVIDENCE_COMPLETE", attempt=str(retained))), flush=True)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    receipt = supervise_tests([str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u",
        str(Path(__file__).resolve()), "--worker"], env=env, retained=retained, started=started)
    receipt["limits"] = limits
    receipt["worker_receipt"] = None
    receipt["output_sha256"] = None
    try:
        if receipt["process_group_confirmed_empty"] is not True:
            raise ValueError("NO_OUTPUT_READ_BEFORE_GROUP_EMPTY")
        output = _read_file(retained/"OUTPUT.txt", OUTPUT_BYTES, deadline, allow_empty=True)
        receipt["output_sha256"] = hashlib.sha256(output).hexdigest()
        receipt["output_retained_bytes"] = len(output)
        rows = []
        for line in output.splitlines():
            if line.startswith(b"{"):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if type(row) is dict and row.get("kind") == "V5_RUNNER_SOURCE_WORKER_RECEIPT":
                    rows.append(row)
        if len(rows) != 1:
            raise ValueError("EXACT_ONE_COMPLETE_WORKER_RECEIPT_REQUIRED")
        row = rows[0]
        receipt["worker_receipt"] = row
        if (any(type(row.get(key)) is not int or row[key] != expected for key, expected in (
                ("collected", 85), ("passed", 85), ("failed", 0), ("skipped", 0), ("errors", 0),
                ("deselected", 0), ("xfailed", 0), ("xpassed", 0), ("pytest_exit_code", 0)))
                or row.get("native_ocp_imported") is not False or row.get("repository_initializer_absent") is not True
                or type(row.get("worker_cpu_seconds")) not in (int, float)
                or not math.isfinite(row["worker_cpu_seconds"]) or not 0 <= row["worker_cpu_seconds"] <= CPU_SECONDS
                or type(row.get("worker_peak_rss_bytes")) is not int or not 0 < row["worker_peak_rss_bytes"] <= RSS_BYTES):
            raise ValueError("WORKER_OUTCOME_OR_RESOURCE_GATE")
        if not receipt["output_stored_bytes_exact"] or len(output) != receipt["output_bytes_write_returns"]:
            raise ValueError("OUTPUT_WRITE_RETENTION_NOT_EXACT")
    except Exception as error:
        receipt["output_or_worker_receipt_error"] = type(error).__name__+":"+str(error)[:160]
    after, _ = _sources(deadline)
    receipt.update(source_sha256_after=after["hashes"], source_after_errors=after["errors"],
        source_after_unattempted=after["unattempted"], source_before_after_consistent=after == before,
        repository_initializer_absent_after=not os.path.lexists(ROOT/"__init__.py"),
        reaped_children_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parent_wall_seconds=time.monotonic()-started, retained_attempt_directory=str(retained),
        evidence_scope="SOURCE_MOCK_CONTRACTS_NOT_NATIVE_PHYSICAL_F1_OR_MAXIMUM_TRANSPORT_CAPACITY")
    passed = (receipt["exit_code"] == 0 and not receipt["budget_stop"] and not receipt["primary_error"]
        and not receipt["cleanup_errors"] and receipt["process_group_confirmed_empty"]
        and not receipt.get("output_or_worker_receipt_error") and receipt["source_before_after_consistent"]
        and receipt["repository_initializer_absent_after"] and receipt["parent_wall_seconds"] < WALL_SECONDS
        and receipt["parent_peak_rss_bytes"] <= RSS_BYTES
        and receipt["reaped_children_peak_rss_bytes"]+receipt["parent_peak_rss_bytes"] <= RSS_BYTES)
    receipt["status"] = "PASS_PENDING_INDEPENDENT_POST" if passed else "TERMINAL_SOURCE_GATE_FAILURE"
    completed_file = _publish(retained/"COMPLETED.json", receipt)
    final_inventory, final_inventory_error = None, None
    try:
        if receipt["process_group_confirmed_empty"] is not True:
            raise ValueError("NO_ARTIFACT_READ_BEFORE_GROUP_EMPTY")
        final_inventory = _attempt_inventory(retained, deadline)
        if (final_inventory["STARTED.json"] != started_file or final_inventory["COMPLETED.json"] != completed_file
                or final_inventory["OUTPUT.txt"]["sha256"] != receipt["output_sha256"]):
            raise ValueError("FINAL_SOURCE_GATE_ARTIFACT_BINDING_CHANGED")
    except Exception as error:
        final_inventory_error = type(error).__name__+":"+str(error)[:160]
    final_within_wall = time.monotonic()-started < WALL_SECONDS
    print(json.dumps(dict(status=receipt["status"], attempt=str(retained), outcome_gate=passed,
        final_publication_within_wall=final_within_wall, output_sha256=receipt["output_sha256"],
        final_three_file_inventory=final_inventory, final_inventory_error=final_inventory_error)), flush=True)
    return 0 if passed and final_within_wall and not final_inventory_error and time.monotonic()-started < WALL_SECONDS else 1


if __name__ == "__main__":
    raise SystemExit(main())
