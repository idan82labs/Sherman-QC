"""Frozen context-corrected depth-scheduler array packet; exact PRE required.

Concrete limits, prerequisite independent POSTs, exact PRE and archive must
precede a new execution. No native/physical inputs or old attempt continuation.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
WALL_SECONDS, WORK_SECONDS, CPU_SECONDS = 6600, 6595, 6000
RSS_BYTES, OUTPUT_BYTES = 4*1024**3, 1024**2
TEST = "tests/test_cad_guided_two_plane_depth_connected_v2.py"
PREDECESSOR = "scripts/check_cad_guided_two_plane_depth_connected_v1.py"
PREDECESSOR_SHA256 = "7b093ef70606d3037d0e5b09ce4b40bd639f698f61e51eb9c29142c5d0d61c36"
ADDITIONAL_SOURCE_FILES = (
    "scripts/check_cad_guided_two_plane_depth_connected_v2.py",
    "tests/test_cad_guided_two_plane_depth_connected_v2.py",
    "docs/research/2026-09-11_two_plane_depth_v2_connected_protocol.md",
    "backend/cad_guided_two_plane_context_v2.py",
    "docs/research/2026-09-11_two_plane_depth_v1_connected_post_run_audit.md",
    "docs/research/2026-09-11_two_plane_context_v2_source_post_run_audit.md",
)
STAGES = (
    *[("PLANAR", index, name, arm, 8*1024**2)
      for index, name in enumerate(("P-clean", "P-noisy")) for arm in ("V11", "V12")],
    *[(kind, index, name, arm, maximum)
      for index, name in enumerate(("U-R", "U-A", "L-R", "L-A"))
      for kind, arm, maximum in (("REGISTRATION", "SHARED", 64*1024**2),
                                ("MEASUREMENT", "V1", 32*1024**2),
                                ("MEASUREMENT", "V2", 32*1024**2))],
)
MAX_ARTIFACT_BYTES = 544*1024**2+16*4096


def _require_execution_limits():
    if not (all(type(value) is int and value > 0 for value in (WALL_SECONDS, WORK_SECONDS, CPU_SECONDS))
            and CPU_SECONDS <= WORK_SECONDS <= WALL_SECONDS-5):
        raise ValueError("DEPTH_PACKET_LIMITS_UNSET_PREREQUISITE_POST_AND_EXACT_PRE_REQUIRED")


def _require_no_repository_initializer():
    # Pytest imports tests/__init__.py (in SOURCE_FILES). An enclosing package
    # initializer could introduce a further import; any object here refuses.
    if os.path.lexists(ROOT/"__init__.py"):
        raise ValueError("UNDECLARED_ENCLOSING_REPOSITORY_INITIALIZER")


def _supervisor():
    path = ROOT/PREDECESSOR
    with path.open("rb") as stream:
        blob = stream.read(512*1024+1)
    if len(blob) > 512*1024 or hashlib.sha256(blob).hexdigest() != PREDECESSOR_SHA256:
        raise ValueError("EXACT_PREDECESSOR_SUPERVISOR_SOURCE_REQUIRED")
    spec = importlib.util.spec_from_file_location("depth_packet_predecessor_definitions", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(path), "exec"), module.__dict__)
    supervisor = module._supervisor()  # Definitions only; never predecessor main.
    supervisor.SOURCE_FILES = list(dict.fromkeys([*supervisor.SOURCE_FILES, *ADDITIONAL_SOURCE_FILES]))
    supervisor.WALL_SECONDS, supervisor.WORK_SECONDS, supervisor.CPU_SECONDS = WALL_SECONDS, WORK_SECONDS, CPU_SECONDS
    supervisor.RSS_BYTES, supervisor.OUTPUT_BYTES = RSS_BYTES, OUTPUT_BYTES
    return supervisor


def _unknown_inventory(reason):
    return dict(complete=False, entries={}, errors=[reason], read_bytes=0, stage_dispositions=[
        dict(kind=kind, case_index=index, case_id=name, arm=arm, status="CUSTODY_UNAVAILABLE_STAGE_UNKNOWN")
        for kind, index, name, arm, _ in STAGES])


def artifact_inventory(directory):
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("INVALID_CONNECTED_ARTIFACT_DIRECTORY")
    specs = {f"{kind}_{phase}_{index:02d}_{arm}.json": 4096 if phase == "START" else maximum
        for kind, index, _, arm, maximum in STAGES for phase in ("START", "RESULT")}
    result = dict(complete=False, entries={}, errors=[], read_bytes=0, stage_dispositions=[])
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
        path, maximum, fd = directory/name, specs[name], None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
                raise ValueError("INVALID_ARTIFACT_TYPE_OR_SIZE")
            size, digest = 0, hashlib.sha256()
            while True:
                block = os.read(fd, min(65536, maximum-size+1))
                if not block:
                    break
                size += len(block)
                result["read_bytes"] += len(block)
                if size > maximum or result["read_bytes"] > MAX_ARTIFACT_BYTES:
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
    for kind, index, name, arm, _ in STAGES:
        start = f"{kind}_START_{index:02d}_{arm}.json" in result["entries"]
        end = f"{kind}_RESULT_{index:02d}_{arm}.json" in result["entries"]
        disposition = ("COMPLETE_ARTIFACT_PENDING_AUDIT" if start and end else
            "WORKER_TERMINATED_OR_PUBLICATION_FAILED_STAGE_UNKNOWN" if start else
            "RESULT_WITHOUT_START_INVALID" if end else "NOT_ATTEMPTED_OR_START_PUBLICATION_FAILED")
        result["stage_dispositions"].append(dict(kind=kind, case_index=index, case_id=name, arm=arm,
            status=disposition))
    if set(result["entries"]) != set(specs):
        result["errors"].append("INCOMPLETE_16_STAGE_32_ARTIFACT_DENOMINATOR")
    result["complete"] = not result["errors"]
    if len(json.dumps(result, sort_keys=True).encode()) > 65536:
        raise ValueError("ARTIFACT_INVENTORY_METADATA_BUDGET")
    return result


def main():
    _require_execution_limits()  # Before --worker, imports, source/attempt IO or dispatch; no environment bypass.
    _require_no_repository_initializer()
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
        sys.path.insert(0, str(ROOT))
        package = types.ModuleType("backend")
        package.__path__ = [str(ROOT/"backend")]
        sys.modules["backend"] = package
        os.environ["SHERMAN_RUN_TWO_PLANE_DEPTH_CONNECTED_V2"] = "1"
        import pytest
        code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
            "-q", "-s", str(ROOT/TEST)])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps(dict(worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)
        return int(code)
    if sys.argv[1:]:
        raise ValueError("NO_CASE_OR_BUDGET_OVERRIDES")
    supervisor = _supervisor()
    before = supervisor.source_bindings()
    directory = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    directory.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="two_plane_depth_v2_connected-", dir=directory))
    artifacts = retained/"ARTIFACTS"
    limits = dict(wall_seconds=WALL_SECONDS, work_seconds=WORK_SECONDS, cpu_seconds=CPU_SECONDS,
        sampled_group_rss_bytes=RSS_BYTES, output_bytes=OUTPUT_BYTES, artifact_bytes=MAX_ARTIFACT_BYTES)
    with (retained/"STARTED.json").open("x") as output:
        json.dump(dict(status="STARTED_SYNTHETIC_CHECK_NO_OVERWRITE", source_sha256_before=before,
            repository_initializer_absent_before=True,
            limits=limits, expected_pytest_outcomes=7, stage_denominator=16,
            measurement_denominator=8, planar_denominator=4, shared_registration_denominator=4,
            scientific_scope="DEPTH_SCHEDULER_V2_CONTEXT_V2_SAME_COMPOSITE_ARRAYS_NOT_NATIVE_OR_PHYSICAL"),
            output, sort_keys=True, allow_nan=False)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        SHERMAN_TWO_PLANE_DEPTH_CONNECTED_V2_ARTIFACT_DIRECTORY=str(artifacts))
    print(json.dumps(dict(started_attempt_directory=str(retained), status="RUNNING_NOT_EVIDENCE_COMPLETE")), flush=True)
    try:
        receipt = supervisor.supervise_native_context_test_v3(
            [str(ROOT/".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, env=env, retained=retained)
    except Exception as error:
        receipt = dict(exit_code=None, budget_stop="SUPERVISION_RECEIPT_UNAVAILABLE",
            primary_error=type(error).__name__+":"+str(error)[:256], cleanup_error="GROUP_CUSTODY_UNAVAILABLE",
            process_group_confirmed_empty=False, requires_group_attention=True)
    try:
        _require_no_repository_initializer()
        after = supervisor.source_bindings()
        receipt["repository_initializer_absent_after"] = True
    except Exception as error:
        after = None
        receipt["repository_initializer_absent_after"] = not os.path.lexists(ROOT/"__init__.py")
        receipt["source_after_error"] = type(error).__name__+":"+str(error)[:256]
    receipt.update(source_sha256_after=after, source_before_after_consistent=after == before,
        retained_attempt_directory=str(retained), evidence_scope="DEPTH_SCHEDULER_INVENTED_ARRAYS_RETAINED_COMPOSITE_CONTROL_ONLY")
    if receipt["process_group_confirmed_empty"]:
        try:
            inventory = artifact_inventory(artifacts)
        except Exception as error:
            inventory = _unknown_inventory("ARTIFACT_INVENTORY_FAILED:"+type(error).__name__)
    else:
        inventory = _unknown_inventory("WRITER_QUIESCENCE_UNCONFIRMED_NO_ARTIFACT_READS")
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
    if receipt["process_group_confirmed_empty"]:
        print((retained/"OUTPUT.txt").read_text(), end="")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if (receipt["exit_code"] == 0 and not receipt["budget_stop"] and not receipt["cleanup_error"]
        and receipt["process_group_confirmed_empty"] and receipt["source_before_after_consistent"]
        and inventory["complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
