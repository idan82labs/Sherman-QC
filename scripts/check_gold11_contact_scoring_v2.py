"""Prospective pure contact-scorer gate; exact independent PRE is required.

The reviewed prospective denominator is fixed before exact independent PRE.
No inference, native geometry, physical cohort or prior attempt is executed.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTCOMES = 123
WALL_SECONDS, WORK_SECONDS, CPU_SECONDS = 60, 55, 45
RSS_BYTES, OUTPUT_BYTES = 1024**3, 1024**2
SUPERVISOR = "scripts/check_gold11_mesh_v4_custody.py"
SUPERVISOR_SHA256 = "579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b"
TEST = "tests/test_gold11_contact_scoring_v2.py"
SOURCE_FILES = (
    "scripts/check_gold11_contact_scoring_v2.py", SUPERVISOR,
    "scripts/gold11_contact_scoring_v2.py", TEST,
    "scripts/gold11_mesh_scoring_v1.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    "scripts/f1_physical_evaluation_contract_v2.py",
    "docs/research/2026-09-10_gold11_contact_scoring_v2_source_protocol.md",
)


def _read_source(relative):
    path = ROOT / relative
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= 2*1024**2:
            raise ValueError("SOURCE_TYPE_OR_BYTE_BUDGET")
        blob = bytearray()
        while True:
            block = os.read(fd, min(65536, 2*1024**2-len(blob)+1))
            if not block:
                break
            blob.extend(block)
            if len(blob) > 2*1024**2:
                raise ValueError("SOURCE_BYTE_BUDGET")
        after = os.fstat(fd)
        def identity(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        if identity(before) != identity(after) or identity(path.lstat()) != identity(after) or len(blob) != before.st_size:
            raise ValueError("SOURCE_CHANGED_DURING_READ")
        return bytes(blob)
    finally:
        os.close(fd)


def _sources():
    return {name: hashlib.sha256(_read_source(name)).hexdigest() for name in SOURCE_FILES}


def _supervisor():
    blob = _read_source(SUPERVISOR)
    if hashlib.sha256(blob).hexdigest() != SUPERVISOR_SHA256:
        raise ValueError("EXACT_SUPERVISOR_SOURCE_REQUIRED")
    module = types.ModuleType("_contact_v2_frozen_supervisor_definitions")
    module.__file__ = str(ROOT / SUPERVISOR)
    exec(compile(blob, module.__file__, "exec"), module.__dict__)
    # Only the reviewed supervision function runs; its old main/source selection
    # and old custody tests are not called.
    return module


def main():
    if type(EXPECTED_OUTCOMES) is not int or EXPECTED_OUTCOMES <= 0:
        raise ValueError("CONTACT_GATE_DENOMINATOR_PENDING_FINAL_REVIEW_AND_EXACT_PRE")
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT / ".venv/lib/python3.11/site-packages"))
        import pytest
        code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
            "-q", "-s", str(ROOT / TEST)])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps(dict(worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)
        return int(code)
    if sys.argv[1:]:
        raise ValueError("NO_TEST_CASE_OR_BUDGET_OVERRIDES")
    before = _sources()
    supervisor = _supervisor()
    if (supervisor.WALL_SECONDS, supervisor.WORK_SECONDS, supervisor.RSS_BYTES, supervisor.OUTPUT_BYTES) != (
            WALL_SECONDS, WORK_SECONDS, RSS_BYTES, OUTPUT_BYTES):
        raise ValueError("SUPERVISOR_LIMIT_MISMATCH")
    base = ROOT / "docs/research/cad_engine_v1_synthetic_checks"
    base.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="gold11_contact_v2_source-", dir=base))
    with (retained / "STARTED.json").open("x") as output:
        json.dump(dict(status="STARTED_SYNTHETIC_SOURCE_GATE_NO_OVERWRITE",
            source_sha256_before=before, expected_test_outcomes=EXPECTED_OUTCOMES,
            limits=dict(wall_seconds=WALL_SECONDS, work_seconds=WORK_SECONDS,
                worker_cpu_seconds=CPU_SECONDS, sampled_group_rss_bytes=RSS_BYTES,
                output_bytes=OUTPUT_BYTES)), output, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(started_attempt_directory=str(retained), status="RUNNING_NOT_EVIDENCE_COMPLETE")), flush=True)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    try:
        receipt = supervisor.supervise_native_context_test_v3(
            [str(ROOT / ".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, env=env, retained=retained)
    except Exception as error:
        receipt = dict(exit_code=None, budget_stop="SUPERVISION_RECEIPT_UNAVAILABLE",
            primary_error=type(error).__name__, cleanup_error="GROUP_CUSTODY_UNAVAILABLE",
            process_group_confirmed_empty=False, requires_group_attention=True)
    try:
        after = _sources()
    except Exception as error:
        after = None
        receipt["source_after_error"] = type(error).__name__
    receipt.update(source_sha256_after=after, source_before_after_consistent=after == before,
        retained_attempt_directory=str(retained),
        evidence_scope="INVENTED_JSON_AND_LINES_NOT_NATIVE_PHYSICAL_OR_RUNNER_QUALIFICATION")
    with (retained / "COMPLETED.json").open("x") as output:
        json.dump(receipt, output, sort_keys=True, allow_nan=False)
    if receipt["process_group_confirmed_empty"]:
        print((retained / "OUTPUT.txt").read_text(), end="")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if (receipt["exit_code"] == 0 and not receipt["budget_stop"]
        and not receipt["cleanup_error"] and receipt["process_group_confirmed_empty"]
        and receipt["source_before_after_consistent"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
