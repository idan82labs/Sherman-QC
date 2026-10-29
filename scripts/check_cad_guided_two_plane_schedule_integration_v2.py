"""Prospective controlled-integration source gate; exact independent PRE required."""
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
EXPECTED_OUTCOMES = 69
WALL_SECONDS, WORK_SECONDS, CPU_SECONDS = 60, 55, 45
RSS_BYTES, OUTPUT_BYTES = 1024**3, 1024**2
SUPERVISOR = "scripts/check_gold11_mesh_v4_custody.py"
SUPERVISOR_SHA256 = "579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b"
TEST = "tests/test_cad_guided_two_plane_schedule_integration_v2.py"
SOURCE_FILES = (
    "scripts/check_cad_guided_two_plane_schedule_integration_v2.py",
    "scripts/check_gold11_mesh_v4_custody.py",
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
    "backend/cad_guided_seeded_plane_proposals_v1.py",
    "backend/cad_guided_scan_plane_preparation_v1.py",
    "backend/cad_guided_composite_surface_roles_v1.py",
    "backend/cad_guided_mesh_offline_v13.py",
    "backend/cad_guided_exact_support_cover_v1.py",
    "backend/cad_guided_support_cover_measurement_v2.py",
    "backend/cad_guided_two_plane_schedule_v2.py",
    "tests/test_cad_guided_two_plane_schedule_integration_v2.py",
    "tests/test_cad_guided_mesh_offline_v13.py",
    "tests/__init__.py",
    "docs/research/2026-09-10_two_plane_schedule_v2_integration_source_protocol.md",
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


def main():
    if type(EXPECTED_OUTCOMES) is not int or EXPECTED_OUTCOMES <= 0:
        raise ValueError("SCHEDULE_V2_GATE_PENDING_DENOMINATOR_REVIEW_AND_EXACT_PRE")
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT / ".venv/lib/python3.11/site-packages"))
        package = types.ModuleType("backend")
        package.__path__ = [str(ROOT / "backend")]
        sys.modules["backend"] = package
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
    blob = _read_source(SUPERVISOR)
    if hashlib.sha256(blob).hexdigest() != SUPERVISOR_SHA256:
        raise ValueError("EXACT_SUPERVISOR_SOURCE_REQUIRED")
    supervisor = types.ModuleType("_schedule_v2_integration_supervisor_definitions")
    supervisor.__file__ = str(ROOT / SUPERVISOR)
    exec(compile(blob, supervisor.__file__, "exec"), supervisor.__dict__)
    if (supervisor.WALL_SECONDS, supervisor.WORK_SECONDS, supervisor.RSS_BYTES, supervisor.OUTPUT_BYTES) != (
            WALL_SECONDS, WORK_SECONDS, RSS_BYTES, OUTPUT_BYTES):
        raise ValueError("SUPERVISOR_LIMIT_MISMATCH")
    base = ROOT / "docs/research/cad_engine_v1_synthetic_checks"
    base.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="two_plane_schedule_v2_integration-", dir=base))
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
        evidence_scope="INVENTED_TINY_ARRAY_AND_MOCK_WORKFLOW_BINDING_NOT_NATIVE_OR_PHYSICAL_QUALIFICATION")
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
