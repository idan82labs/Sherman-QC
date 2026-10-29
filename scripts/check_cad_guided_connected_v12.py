"""Bounded invented-array qualification for the additive connected V12 path.

No native/physical dataset or default engine entrypoint. Exact source/fixture
freeze and independent PRE precede each new named suite attempt.
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
            worker_cpu_seconds=CPU_SECONDS, sampled_group_rss_bytes=RSS_BYTES,
            output_bytes=OUTPUT_BYTES),
        enforcement_scope="SAMPLED_WORK_AND_ABSOLUTE_CLEANUP_DEADLINES_NOT_HARD_REAL_TIME_OS_IO")


import argparse
import json
import resource
import stat
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
CPU_SECONDS = 45
SOURCE_FILES = [
    "scripts/check_cad_guided_connected_v12.py",
    "docs/research/2026-09-10_connected_v12_gate_protocol.md",
    "docs/research/2026-09-10_connected_v12_fixture_design.md",
    "scripts/run_gold11_mesh_development_v3.py",
    "backend/cad_guided_stage_policies_v1.py",
    "backend/cad_guided_context_measurement_v1.py",
    "backend/cad_guided_context_surface_roles_v1.py",
    "backend/cad_guided_plane_initialization_v12.py",
    "backend/cad_guided_mesh_offline_v12.py",
    "backend/cad_guided_plane_initialization_v11.py",
    "backend/cad_guided_plane_initialization_v10.py",
    "backend/cad_guided_plane_initialization_v9.py",
    "backend/cad_guided_plane_initialization_v8.py",
    "backend/cad_guided_plane_initialization_v4.py",
    "backend/cad_guided_mesh_offline_v11.py",
    "backend/cad_guided_mesh_offline_v8.py",
    "backend/cad_guided_mesh_offline_v2.py",
    "backend/cad_guided_mesh_offline_v1.py",
    "backend/cad_guided_global_registration_v3.py",
    "backend/cad_guided_global_registration_v2.py",
    "backend/cad_guided_global_registration_v1.py",
    "backend/cad_guided_triangle_surface_v3.py",
    "backend/cad_guided_triangle_surface_v2.py",
    "backend/cad_guided_triangle_distance_bounds_v3.py",
    "backend/cad_guided_face_box_screen_v1.py",
    "backend/cad_guided_two_plane_context_v1.py",
    "backend/cad_guided_two_plane_schedule_v1.py",
    "backend/cad_guided_noisy_mesh_planes_v7.py",
    "backend/cad_guided_noisy_mesh_planes_v5.py",
    "backend/cad_guided_plane_search_v6.py",
    "backend/cad_guided_plane_search_v5.py",
    "backend/cad_guided_report_envelope_v6.py",
    "backend/cad_guided_mesh_plane_proposals_v4.py",
    "backend/cad_guided_plane_hypotheses_v4.py",
    "backend/cad_guided_mesh_topology_v4.py",
    "backend/cad_guided_mesh_input_v1.py",
    "backend/cad_guided_native_features.py",
    "backend/cad_guided_native_features_v1.py",
    "backend/cad_guided_measurement.py",
    "backend/cad_guided_instances.py",
    "backend/cad_guided_local_measurement_v1.py",
    "backend/cad_guided_cad_surface_v2.py",
    "backend/cad_guided_cad_sampling_v1.py",
    "tests/test_cad_guided_context_measurement_v1.py",
    "tests/test_cad_guided_context_surface_roles_v1.py",
    "tests/test_cad_guided_connected_v12.py",
    "tests/test_cad_guided_plane_initialization_v11.py",
    "tests/test_cad_guided_plane_initialization_v10.py",
    "tests/test_cad_guided_plane_initialization_v8.py"
]
TESTS = (
    "tests/test_cad_guided_context_measurement_v1.py",
    "tests/test_cad_guided_context_surface_roles_v1.py",
    "tests/test_cad_guided_connected_v12.py",
)
CASE_IDS = ("P-clean", "P-noisy", "U-R", "U-A", "L-R", "L-A")
ARMS = ("C0", "C1", "C2")


def artifact_inventory(directory, suite):
    """Read only expected owned regular artifacts; unknown names veto, no read."""
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("INVALID_ARTIFACT_DIRECTORY_TYPE")
    specs = {} if suite == "source" else {
        f"ARM_{kind}_{index:02d}_{arm}.json": (4096 if kind == "START" else 16*1024**2)
        for index in range(6) for arm in ARMS for kind in ("START", "RESULT")}
    result = dict(complete=False, entries={}, errors=[], case_arm_dispositions=[], read_bytes=0)
    names = []
    with os.scandir(directory) as iterator:
        for entry in iterator:
            if len(names) >= len(specs)+1:
                result["errors"].append("ARTIFACT_NAME_BUDGET")
                break
            names.append(entry.name)
    if any(name not in specs for name in names):
        result["errors"].append("UNDECLARED_ARTIFACT_NAME_NO_CONTENT_READ")
    for name in sorted(names):
        if name not in specs:
            continue
        path, maximum = directory/name, specs[name]
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
                raise ValueError("INVALID_ARTIFACT_TYPE_OR_SIZE")
            digest, size = hashlib.sha256(), 0
            while True:
                block = os.read(fd, min(65536, maximum-size+1))
                if not block:
                    break
                size += len(block)
                result["read_bytes"] += len(block)
                if size > maximum or result["read_bytes"] > 288*1024**2+18*4096:
                    raise ValueError("ARTIFACT_READ_BUDGET")
                digest.update(block)
            after = os.fstat(fd)
            def identity(info):
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            if identity(before) != identity(after) or size != before.st_size or identity(path.lstat()) != identity(after):
                raise ValueError("ARTIFACT_CHANGED_DURING_READ")
            result["entries"][name] = dict(bytes=size, sha256=digest.hexdigest())
        except Exception as error:
            result["errors"].append(name+":"+type(error).__name__)
        finally:
            if fd is not None:
                os.close(fd)
    if suite == "connected":
        for index, case_id in enumerate(CASE_IDS):
            for arm in ARMS:
                start = f"ARM_START_{index:02d}_{arm}.json" in result["entries"]
                end = f"ARM_RESULT_{index:02d}_{arm}.json" in result["entries"]
                status = ("COMPLETE_ARTIFACT_PENDING_AUDIT" if start and end else
                    "WORKER_TERMINATED_OR_PUBLICATION_FAILED_STAGE_UNKNOWN" if start else
                    "RESULT_WITHOUT_START_INVALID" if end else "NOT_ATTEMPTED_OR_START_PUBLICATION_FAILED")
                result["case_arm_dispositions"].append(dict(case_index=index, case_id=case_id, arm=arm, status=status))
        if len(result["entries"]) != 36:
            result["errors"].append("INCOMPLETE_18_ARM_ARTIFACT_DENOMINATOR")
    result["complete"] = not result["errors"] and set(result["entries"]) == set(specs)
    if len(json.dumps(result, sort_keys=True).encode()) > 65536:
        raise ValueError("ARTIFACT_INVENTORY_METADATA_BUDGET")
    return result


def source_bindings():
    result = {}
    for name in SOURCE_FILES:
        path = ROOT/name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 2*1024**2:
            raise ValueError("INVALID_SOURCE_FILE")
        blob = path.read_bytes()
        if len(blob) > 2*1024**2:
            raise ValueError("SOURCE_BYTE_BUDGET")
        result[name] = hashlib.sha256(blob).hexdigest()
    return result


def unavailable_inventory(suite, reason):
    """Preserve the full denominator when even artifact inspection is unavailable."""
    return dict(complete=False, entries={}, errors=[reason], read_bytes=0,
        case_arm_dispositions=[] if suite == "source" else [
            dict(case_index=index, case_id=case_id, arm=arm,
                 status="ARTIFACT_CUSTODY_UNAVAILABLE_STAGE_UNKNOWN")
            for index, case_id in enumerate(CASE_IDS) for arm in ARMS])


def main():
    global WALL_SECONDS, WORK_SECONDS, CPU_SECONDS, RSS_BYTES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("source", "connected"), required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.suite == "connected":
        WALL_SECONDS, WORK_SECONDS, CPU_SECONDS, RSS_BYTES = 300, 295, 240, 4*1024**3
    if args.worker:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
        sys.path.insert(0, str(ROOT))
        package = types.ModuleType("backend")
        package.__path__ = [str(ROOT/"backend")]
        sys.modules["backend"] = package
        if args.suite == "connected":
            os.environ["SHERMAN_RUN_CONNECTED_V12"] = "1"
        import pytest
        selected = TESTS if args.suite == "source" else (
            "tests/test_cad_guided_connected_v12.py::test_connected_six_case_three_arm_packet",)
        code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
            "-q", "-s", *[str(ROOT/name) for name in selected]])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps(dict(worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
                              worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)
        return int(code)
    before = source_bindings()
    directory = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    directory.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="connected_v12_"+args.suite+"-", dir=directory))
    artifact_directory = retained/"ARTIFACTS"
    if args.suite == "source":
        artifact_directory.mkdir()
    frozen = dict(status="STARTED_SYNTHETIC_CHECK_NO_OVERWRITE", suite=args.suite,
        source_sha256_before=before, limits=dict(wall_seconds=WALL_SECONDS,
        work_seconds=WORK_SECONDS, cpu_seconds=CPU_SECONDS, sampled_group_rss_bytes=RSS_BYTES,
        output_bytes=OUTPUT_BYTES), artifact_directory=str(artifact_directory),
        scientific_scope="INVENTED_ARRAYS_NO_NATIVE_OR_PHYSICAL_INPUTS")
    with (retained/"STARTED.json").open("x") as output:
        json.dump(frozen, output, sort_keys=True)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        SHERMAN_CONNECTED_V12_ARTIFACT_DIRECTORY=str(artifact_directory))
    try:
        receipt = supervise_native_context_test_v3(
            [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()),
             "--worker", "--suite", args.suite], cwd=ROOT, env=env, retained=retained)
    except Exception as error:
        receipt = dict(exit_code=None, budget_stop="SUPERVISION_RECEIPT_UNAVAILABLE",
            primary_error=type(error).__name__+":"+str(error)[:256],
            cleanup_error="GROUP_CUSTODY_UNAVAILABLE", process_group_confirmed_empty=False,
            requires_group_attention=True)
    try:
        after = source_bindings()
    except Exception as error:
        after = None
        receipt["source_after_error"] = type(error).__name__+":"+str(error)[:256]
    receipt.update(source_sha256_after=after, retained_attempt_directory=str(retained),
        suite=args.suite, artifact_directory=str(artifact_directory),
        evidence_scope="INVENTED_ARRAY_CONTRACTS_NOT_NATIVE_OR_GOLD_QUALIFICATION")
    receipt["source_before_after_consistent"] = receipt["source_sha256_after"] == before
    if receipt["process_group_confirmed_empty"]:
        try:
            inventory = artifact_inventory(artifact_directory, args.suite)
        except Exception as error:
            inventory = unavailable_inventory(args.suite,
                "ARTIFACT_INVENTORY_FAILED:"+type(error).__name__+":"+str(error)[:256])
    else:
        inventory = unavailable_inventory(args.suite,
            "WRITER_QUIESCENCE_UNCONFIRMED_NO_ARTIFACT_READS")
    try:
        with (retained/"ARTIFACT_INVENTORY.json").open("x") as output:
            json.dump(inventory, output, sort_keys=True, allow_nan=False)
    except Exception as error:
        inventory["complete"] = False
        receipt["inventory_publication_error"] = type(error).__name__+":"+str(error)[:256]
        receipt["unpublished_inventory"] = inventory
    receipt["artifact_inventory_complete"] = inventory["complete"]
    with (retained/"COMPLETED.json").open("x") as output:
        json.dump(receipt, output, sort_keys=True, allow_nan=False)
    if receipt["process_group_confirmed_empty"] and (retained/"OUTPUT.txt").is_file():
        print((retained/"OUTPUT.txt").read_text(), end="")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if (receipt["exit_code"] == 0 and not receipt["budget_stop"]
        and receipt["process_group_confirmed_empty"] and not receipt["cleanup_error"]
        and receipt["source_before_after_consistent"] and inventory["complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
