"""Prospective single-attempt composite native-byte gate; exact PRE required.

DRAFT: native time limits are deliberately unset pending connected evidence.
No physical input, old suite or previous terminal attempt is executed here.
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
WALL_SECONDS = WORK_SECONDS = CPU_SECONDS = None
RSS_BYTES, OUTPUT_BYTES = 4*1024**3, 1024**2
TEST = "tests/test_cad_guided_composite_plane_native_v1.py"
ADDITIONAL_SOURCE_FILES = (
    "scripts/check_cad_guided_composite_plane_native_v1.py",
    TEST,
    "docs/research/2026-09-10_composite_plane_v1_native_protocol.md",
    "tests/test_cad_guided_connected_v12_native_bytes.py",
    "tests/test_cad_guided_mesh_connected_native_v1.py",
    "tests/test_cad_guided_support_cover_native_v13.py",
    "docs/research/2026-09-10_composite_plane_v1_connected_post_run_audit.md",
)
ARTIFACT_SPECS = {"CASE_START.json": 4096, "CAD.step": 8*1024**2,
    "SCAN.ply": 1024**2, "CASE_RESULT.json": 32*1024**2}
MAX_ARTIFACT_BYTES = 41*1024**2+4096


def _supervisor():
    spec = importlib.util.spec_from_file_location("source_cover_supervisor_definitions",
        ROOT/"scripts/check_cad_guided_composite_plane_connected_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supervisor = module._supervisor()
    supervisor.SOURCE_FILES += list(ADDITIONAL_SOURCE_FILES)
    supervisor.WALL_SECONDS, supervisor.WORK_SECONDS, supervisor.CPU_SECONDS = WALL_SECONDS, WORK_SECONDS, CPU_SECONDS
    supervisor.RSS_BYTES, supervisor.OUTPUT_BYTES = RSS_BYTES, OUTPUT_BYTES
    return supervisor


def _unknown_inventory(reason):
    return dict(complete=False, entries={}, errors=[reason], read_bytes=0,
        case_disposition="CUSTODY_UNAVAILABLE_CASE_UNKNOWN")


def artifact_inventory(directory):
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("INVALID_CONNECTED_ARTIFACT_DIRECTORY")
    specs = ARTIFACT_SPECS
    result = dict(complete=False, entries={}, errors=[], read_bytes=0, case_disposition=None)
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
    names = set(result["entries"])
    result["case_disposition"] = ("COMPLETE_ARTIFACTS_PENDING_AUDIT" if names == set(specs)
        else "INCOMPLETE_CASE_STAGE_UNKNOWN" if "CASE_START.json" in names
        else "NOT_ATTEMPTED_OR_START_PUBLICATION_FAILED")
    if set(result["entries"]) != set(specs):
        result["errors"].append("INCOMPLETE_ONE_NATIVE_CASE_FOUR_ARTIFACT_DENOMINATOR")
    result["complete"] = not result["errors"]
    if len(json.dumps(result, sort_keys=True).encode()) > 65536:
        raise ValueError("ARTIFACT_INVENTORY_METADATA_BUDGET")
    return result


def main():
    if not all(type(value) is int and value > 0 for value in (WALL_SECONDS, WORK_SECONDS, CPU_SECONDS)):
        raise ValueError("NATIVE_TIME_LIMITS_PENDING_CONNECTED_EVIDENCE_AND_EXACT_PRE")
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
        sys.path.insert(0, str(ROOT))
        package = types.ModuleType("backend")
        package.__path__ = [str(ROOT/"backend")]
        sys.modules["backend"] = package
        os.environ["SHERMAN_RUN_COMPOSITE_PLANE_NATIVE_V1"] = "1"
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
    retained = Path(tempfile.mkdtemp(prefix="composite_plane_v1_native-", dir=directory))
    artifacts = retained/"ARTIFACTS"  # The reviewed fixture creates it exclusively.
    limits = dict(wall_seconds=WALL_SECONDS, work_seconds=WORK_SECONDS, cpu_seconds=CPU_SECONDS,
        sampled_group_rss_bytes=RSS_BYTES, output_bytes=OUTPUT_BYTES, artifact_bytes=MAX_ARTIFACT_BYTES)
    with (retained/"STARTED.json").open("x") as output:
        json.dump(dict(status="STARTED_SYNTHETIC_CHECK_NO_OVERWRITE", source_sha256_before=before,
            limits=limits, native_case_denominator=1, prospective_test_outcomes=28,
            scientific_scope="ONE_INVENTED_NATIVE_STEP_PLY_WORKFLOW_NOT_PHYSICAL"),
            output, sort_keys=True, allow_nan=False)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        SHERMAN_COMPOSITE_PLANE_NATIVE_V1_ARTIFACT_DIRECTORY=str(artifacts))
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
        after = supervisor.source_bindings()
    except Exception as error:
        after = None
        receipt["source_after_error"] = type(error).__name__+":"+str(error)[:256]
    receipt.update(source_sha256_after=after, source_before_after_consistent=after == before,
        retained_attempt_directory=str(retained), evidence_scope="ONE_INVENTED_NATIVE_STEP_PLY_WORKFLOW_NOT_PHYSICAL")
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
