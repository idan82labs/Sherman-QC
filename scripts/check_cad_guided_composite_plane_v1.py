"""Single-attempt explicit composite integration source gate, not a connected run.

Requires an exact independent PRE before execution. Reuses the unchanged V12
supervisor; does not execute its main or rerun either terminal V12 suite.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
WALL_SECONDS, WORK_SECONDS, CPU_SECONDS = 60, 55, 45
RSS_BYTES, OUTPUT_BYTES = 1024**3, 1024**2
TESTS = (
    "tests/test_cad_guided_scan_plane_preparation_v1.py",
    "tests/test_cad_guided_composite_surface_roles_v1.py",
    "tests/test_cad_guided_composite_plane_integration_v1.py",
    "tests/test_cad_guided_mesh_offline_v13.py",
)
ADDITIONAL_SOURCE_FILES = (
    "scripts/check_cad_guided_composite_plane_v1.py",
    "docs/research/2026-09-10_composite_plane_v1_source_protocol.md",
    "docs/research/2026-09-10_seeded_plane_proposal_successor.md",
    "backend/cad_guided_seeded_plane_proposals_v1.py",
    "backend/cad_guided_scan_plane_preparation_v1.py",
    "backend/cad_guided_composite_surface_roles_v1.py",
    "backend/cad_guided_mesh_offline_v13.py",
    "backend/cad_guided_exact_support_cover_v1.py",
    "backend/cad_guided_support_cover_measurement_v2.py",
    *TESTS,
)


def _supervisor():
    path = ROOT/"scripts/check_cad_guided_connected_v12.py"
    spec = importlib.util.spec_from_file_location("frozen_v12_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_FILES = [*module.SOURCE_FILES, *ADDITIONAL_SOURCE_FILES]
    module.WALL_SECONDS, module.WORK_SECONDS, module.CPU_SECONDS = WALL_SECONDS, WORK_SECONDS, CPU_SECONDS
    module.RSS_BYTES, module.OUTPUT_BYTES = RSS_BYTES, OUTPUT_BYTES
    return module


def main():
    if sys.argv[1:] == ["--worker"]:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT/".venv/lib/python3.11/site-packages"))
        sys.path.insert(0, str(ROOT))
        package = types.ModuleType("backend")
        package.__path__ = [str(ROOT/"backend")]
        sys.modules["backend"] = package
        import pytest
        code = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
            "-q", "-s", *[str(ROOT/name) for name in TESTS]])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps(dict(worker_cpu_seconds=usage.ru_utime+usage.ru_stime,
            worker_peak_rss_bytes=usage.ru_maxrss)), flush=True)
        return int(code)
    if sys.argv[1:]:
        raise ValueError("NO_SUITE_OR_BUDGET_OVERRIDES")
    supervisor = _supervisor()
    before = supervisor.source_bindings()
    directory = ROOT/"docs/research/cad_engine_v1_synthetic_checks"
    directory.mkdir(exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix="composite_plane_v1_source-", dir=directory))
    artifacts = retained/"ARTIFACTS"
    artifacts.mkdir()
    limits = dict(wall_seconds=WALL_SECONDS, work_seconds=WORK_SECONDS,
        cpu_seconds=CPU_SECONDS, sampled_group_rss_bytes=RSS_BYTES, output_bytes=OUTPUT_BYTES)
    with (retained/"STARTED.json").open("x") as output:
        json.dump(dict(status="STARTED_SYNTHETIC_CHECK_NO_OVERWRITE", source_sha256_before=before,
            limits=limits, scientific_scope="INVENTED_COMPOSITE_INTEGRATION_AND_MOCK_NATIVE_ADAPTER_NOT_CONNECTED_NATIVE_OR_GOLD"),
            output, sort_keys=True, allow_nan=False)
    env = dict(PATH="/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
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
        retained_attempt_directory=str(retained), evidence_scope="INVENTED_COMPOSITE_INTEGRATION_AND_ADAPTER_CONTRACTS_ONLY")
    if receipt["process_group_confirmed_empty"]:
        try:
            inventory = supervisor.artifact_inventory(artifacts, "source")
        except Exception as error:
            inventory = supervisor.unavailable_inventory("source", "ARTIFACT_INVENTORY_FAILED:"+type(error).__name__)
    else:
        inventory = supervisor.unavailable_inventory("source", "WRITER_QUIESCENCE_UNCONFIRMED_NO_ARTIFACT_READS")
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
    return 0 if (receipt["exit_code"] == 0 and not receipt["budget_stop"]
        and not receipt["cleanup_error"] and receipt["process_group_confirmed_empty"]
        and receipt["source_before_after_consistent"] and inventory["complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

