#!/usr/bin/env python3
"""Shared immutable execution contract for EXP_CAD_0035.

This module is source-free.  It never opens either physical scan.  Physical
scan rows in an EXP35 lock are path-bound to preregistered expected digests;
all other static inputs are path/size/SHA-256 bound and revalidated before a
source or renderer child can be launched.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import sysconfig
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP_CAD_0035"
ARCHITECTURE = (
    "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_PHYSICAL_SEQUENTIAL_MEMBER_INTAKE"
)
PREREGISTRATION_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0035_sequential_member_intake_preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "462274a567cd0ed84b041b8e0576f93ceb1ce6b4df9abe75e63c15ec1f268cb2"
)
LOCK_PATH = "docs/research/2026-08-30_exp_cad_0035_physical_execution_lock.json"
PRELOCK_APPROVAL_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0035_prelock_physical_readiness_approval.json"
)
ENVIRONMENT_ATTESTATION_PATH = (
    "docs/research/2026-08-30_exp_cad_0035_physical_environment_attestation.json"
)
DETACHED_APPROVAL_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0035_detached_postlock_physical_execution_approval.json"
)

LOCK_SCHEMA = "EXP_CAD_0035_PHYSICAL_EXECUTION_LOCK_V1"
LOCK_STATUS = "IMMUTABLE_LOCKED_AWAITING_DETACHED_APPROVAL"
PRELOCK_APPROVAL_SCHEMA = "EXP_CAD_0035_PRELOCK_PHYSICAL_READINESS_APPROVAL_V1"
DETACHED_APPROVAL_SCHEMA = (
    "EXP_CAD_0035_DETACHED_POSTLOCK_PHYSICAL_EXECUTION_APPROVAL_V1"
)
ENVIRONMENT_ATTESTATION_SCHEMA = "EXP_CAD_0035_PHYSICAL_ENVIRONMENT_ATTESTATION_V1"
TEST_ATTESTATION_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_TEST_ATTESTATION_V1"
AUDIT_PACKET_SCHEMA = "EXP_CAD_0035_SOURCE_FREE_AUDIT_PACKET_V1"

FORBIDDEN_ACCESS_COUNT_KEYS = frozenset(
    {
        "physical_scan_hash_call_count",
        "physical_scan_point_cloud_load_call_count",
        "physical_scan_byte_open_operation_count",
        "evaluator_role_label_gold_quality_expected_count_access_count",
    }
)

SOURCE_FREE_REGRESSION_PATHS = (
    "tests/test_exp_cad_0031_source_interface.py",
    "tests/test_exp_cad_0032_cfpdg_v5_equivalence.py",
    "tests/test_exp_cad_0033_qualification_protocol.py",
    "tests/test_exp_cad_0034_execution_contract.py",
    "tests/test_exp_cad_0034_source_runner.py",
    "tests/test_exp_cad_0034_orchestration.py",
    "tests/test_exp_cad_0034_process_monitor.py",
    "tests/test_exp_cad_0035_execution_contract.py",
    "tests/test_exp_cad_0035_process_monitor.py",
    "tests/test_exp_cad_0035_source_runner.py",
    "tests/test_exp_cad_0035_orchestration.py",
    "tests/test_exp_cad_0035_source_free_memory_qualification.py",
)
SOURCE_FREE_COLLECT_COMMAND = (
    ".venv/bin/python",
    "-B",
    "-m",
    "pytest",
    "--collect-only",
    "-q",
    "-p",
    "no:cacheprovider",
    *SOURCE_FREE_REGRESSION_PATHS,
)
SOURCE_FREE_TEST_COMMAND = (
    ".venv/bin/python",
    "-B",
    "tests/test_exp_cad_0035_orchestration.py",
)
# Ordered node IDs emitted by the exact collect command after the final
# source-free hardening pass.  The digest is over strict canonical JSON bytes.
EXPECTED_SOURCE_FREE_TEST_COUNT = 349
EXPECTED_SOURCE_FREE_TEST_IDS_SHA256 = (
    "39a1a9c33e7a48289a6c038310d05c08a78bac6c882806d2f889ae7e7135255c"
)

ENVIRONMENT_FINGERPRINT_KEYS = frozenset(
    {
        "python_executable_invocation_path",
        "python_executable_realpath",
        "python_executable_sha256",
        "python_version",
        "python_implementation",
        "python_cache_tag",
        "python_soabi",
        "python_extension_suffix",
        "python_abi_flags",
        "python_compiler",
        "python_byteorder",
        "python_pointer_bits",
        "platform_system",
        "platform_release",
        "platform_machine",
        "platform_platform",
        "numpy_version",
        "scipy_version",
        "open3d_version",
        "numpy_module_origin",
        "scipy_module_origin",
        "open3d_module_origin",
    }
)

RESOURCE_LIMITS = {
    "source_wall_seconds_maximum": 180,
    "source_peak_process_tree_rss_bytes_maximum": 4294967296,
    "renderer_wall_seconds_maximum": 60,
    "renderer_peak_process_tree_rss_bytes_maximum": 4294967296,
}

ORCHESTRATOR_PATH_ARGUMENTS = (
    "preregistration",
    "lock",
    "detached-approval",
    "source-runner",
    "process-monitor",
    "orchestrator",
    "renderer",
    "parent-preregistration",
    "exp31-preregistration",
    "v5-preregistration",
    "cad-preflight",
    "nodes",
    "edges",
    "edge-geodesic-mm",
    "mesh",
    "scan-first",
    "scan-second",
    "transform-first",
    "transform-second",
    "attempt-root",
    "primary-source-root",
    "replay-source-root",
    "primary-measurement-root",
    "replay-measurement-root",
    "primary-renderer-root",
    "replay-renderer-root",
    "launch-claim",
    "result",
)

SOURCE_ARGUMENT_ORDER = (
    "mode",
    "preregistration",
    "lock",
    "lock-sha256",
    "detached-approval",
    "detached-approval-sha256",
    "launch-claim",
    "launch-claim-sha256",
    "source-launch-capability",
    "source-launch-child-claim",
    "parent-preregistration",
    "exp31-preregistration",
    "v5-preregistration",
    "cad-preflight",
    "nodes",
    "edges",
    "edge-geodesic-mm",
    "mesh",
    "scan-first",
    "scan-second",
    "transform-first",
    "transform-second",
    "output-root",
    "measurement-root",
    "scan-accounting-journal",
    "lifecycle-measurement",
)

RENDERER_BOOTSTRAP = (
    "import importlib.util,json,sys;"
    "from pathlib import Path;"
    "p=Path(sys.argv[1]).resolve(strict=True);"
    "s=importlib.util.spec_from_file_location('exp35_bound_exp30_renderer',p);"
    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
    "x=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));"
    "m.render_native_visuals(x,Path(sys.argv[3]))"
)


def command_templates(environment: Mapping[str, str]) -> dict[str, Any]:
    invocation = environment.get("python_executable_invocation_path")
    if not isinstance(invocation, str) or not invocation:
        raise ValueError("EXP35 environment lacks Python invocation path")
    return {
        "orchestrator": {
            "python_executable_invocation_path": invocation,
            "script": "scripts/orchestrate_exp_cad_0035_cfpdg_v5_sequential_physical.py",
            "required_path_flags_in_order": list(ORCHESTRATOR_PATH_ARGUMENTS),
            "required_digest_flags_in_order": [
                "lock-sha256",
                "detached-approval-sha256",
            ],
            "preparation_or_retry_mode_permitted": False,
        },
        "source": {
            "python_executable_invocation_path": invocation,
            "script": "scripts/run_exp_cad_0035_cfpdg_source.py",
            "required_flags_in_order": list(SOURCE_ARGUMENT_ORDER),
            "mode_values": ["PRIMARY", "REPLAY"],
            "lifecycle_measurement_required": True,
        },
        "process_monitor": {
            "module": "scripts/exp_cad_0035_process_monitor.py",
            "entry_point": "run_measured_command",
            "source_wall_limit_seconds": 180,
            "renderer_wall_limit_seconds": 60,
            "rss_limit_bytes": 4294967296,
            "sampling_interval_seconds": 0.05,
        },
        "renderer": {
            "python_executable_invocation_path": invocation,
            "argv_prefix": ["-c", RENDERER_BOOTSTRAP],
            "script": "scripts/render_exp_cad_0030_cfpdg_native.py",
            "input": "{source_root}/SOURCE_VISUAL_PAYLOAD.json",
            "output": "{renderer_root}",
        },
    }


# Every key is a regular, non-scan file.  The two PLY scans are deliberately
# absent: they live under lock["physical_scans"] and may not be opened or
# hashed by this module or by the lock builder.
QUALIFICATION_ARTIFACT_PATHS = {
    "fixture_manifest": (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures/"
        "EXP_CAD_0035_SOURCE_FREE_FIXTURE_MANIFEST.json"
    ),
    "fixture_boundary_N009": (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures/"
        "boundary-equivalence/N009.ply"
    ),
    "fixture_boundary_N010": (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures/"
        "boundary-equivalence/N010.ply"
    ),
    "fixture_maximal_N009": (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures/"
        "maximal-retention/N009.ply"
    ),
    "fixture_maximal_N010": (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-fixtures/"
        "maximal-retention/N010.ply"
    ),
}
for _trial_ordinal, _trial_name in (
    (1, "legacy-batch"),
    (2, "sequential"),
    (3, "legacy-batch"),
    (4, "sequential"),
    (5, "legacy-batch"),
    (6, "sequential"),
):
    _prefix = f"qualification_trial_{_trial_ordinal:02d}_{_trial_name.replace('-', '_')}"
    _root = (
        "output/research/autoresearch-loop/exp-cad-0035-source-free-qualification/"
        f"trial-{_trial_ordinal:02d}-{_trial_name}"
    )
    QUALIFICATION_ARTIFACT_PATHS[f"{_prefix}_result"] = (
        f"{_root}/SOURCE_FREE_MEMORY_QUALIFICATION_RESULT.json"
    )
    QUALIFICATION_ARTIFACT_PATHS[f"{_prefix}_measurement"] = (
        f"{_root}/EXTERNAL_PROCESS_MEASUREMENT.json"
    )
QUALIFICATION_ARTIFACT_PATHS["qualification_aggregate"] = (
    "output/research/autoresearch-loop/exp-cad-0035-source-free-qualification/"
    "EXP_CAD_0035_SOURCE_FREE_MEMORY_QUALIFICATION_AGGREGATE.json"
)
QUALIFICATION_ARTIFACT_KEYS = frozenset(QUALIFICATION_ARTIFACT_PATHS)

CANONICAL_FRESH_BOUND_PATHS = {
    "exp35_contract": "scripts/exp_cad_0035_execution_contract.py",
    "exp35_lock_builder": "scripts/build_exp_cad_0035_physical_lock.py",
    "exp35_source_runner": "scripts/run_exp_cad_0035_cfpdg_source.py",
    "exp35_orchestrator": "scripts/orchestrate_exp_cad_0035_cfpdg_v5_sequential_physical.py",
    "exp35_monitor": "scripts/exp_cad_0035_process_monitor.py",
    "exp35_fixture_generator": "scripts/generate_exp_cad_0035_source_free_fixtures.py",
    "exp35_memory_worker": "scripts/run_exp_cad_0035_source_free_memory_qualification.py",
    "exp35_contract_tests": "tests/test_exp_cad_0035_execution_contract.py",
    "exp35_source_tests": "tests/test_exp_cad_0035_source_runner.py",
    "exp35_orchestrator_tests": "tests/test_exp_cad_0035_orchestration.py",
    "exp35_monitor_tests": "tests/test_exp_cad_0035_process_monitor.py",
    "exp35_memory_tests": "tests/test_exp_cad_0035_source_free_memory_qualification.py",
    "exp31_regression_tests": "tests/test_exp_cad_0031_source_interface.py",
    "exp32_regression_tests": "tests/test_exp_cad_0032_cfpdg_v5_equivalence.py",
    "exp33_regression_tests": "tests/test_exp_cad_0033_qualification_protocol.py",
    "exp34_contract_tests": "tests/test_exp_cad_0034_execution_contract.py",
    "exp34_source_tests": "tests/test_exp_cad_0034_source_runner.py",
    "exp34_orchestrator_tests": "tests/test_exp_cad_0034_orchestration.py",
    "exp34_monitor_tests": "tests/test_exp_cad_0034_process_monitor.py",
    "source_free_test_attestation": (
        "docs/research/2026-08-30_exp_cad_0035_source_free_test_attestation.json"
    ),
    "environment_attestation": ENVIRONMENT_ATTESTATION_PATH,
    "source_free_audit_packet": (
        "docs/research/2026-08-30_exp_cad_0035_source_free_audit_packet.json"
    ),
    "prelock_readiness_approval": PRELOCK_APPROVAL_PATH,
    **QUALIFICATION_ARTIFACT_PATHS,
}


REQUIRED_BOUND_FILES = frozenset(
    {
        "exp35_preregistration",
        "exp34_preregistration",
        "exp34_terminal_result",
        "exp34_terminal_report",
        "exp34_terminal_orchestration",
        "exp34_process_measurement",
        "exp34_scan_accounting_journal",
        "exp34_launch_claim",
        "exp34_contract",
        "exp34_lock_builder",
        "exp34_monitor",
        "exp34_source_runner",
        "exp34_orchestrator",
        "exp34_test_attestation",
        "exp34_environment_attestation",
        "exp34_audit_packet",
        "exp34_prelock_approval",
        "exp34_lock",
        "exp34_detached_approval",
        "parent_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "exp31_filter_module",
        "exp31_source_runner_lineage",
        "exp32_v5_engine",
        "exp32_source_free_lineage_runner",
        "frozen_parent_core",
        "frozen_parent_science",
        "frozen_parent_runner",
        "renderer",
        "source_manifest",
        "source_cad",
        "mesh_report",
        "accepted_mesh",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "transform_N009",
        "transform_N010",
        "exp35_contract",
        "exp35_lock_builder",
        "exp35_source_runner",
        "exp35_orchestrator",
        "exp35_monitor",
        "exp35_fixture_generator",
        "exp35_memory_worker",
        "exp35_contract_tests",
        "exp35_source_tests",
        "exp35_orchestrator_tests",
        "exp35_monitor_tests",
        "exp35_memory_tests",
        "exp31_regression_tests",
        "exp32_regression_tests",
        "exp33_regression_tests",
        "exp34_contract_tests",
        "exp34_source_tests",
        "exp34_orchestrator_tests",
        "exp34_monitor_tests",
        "source_free_test_attestation",
        "environment_attestation",
        "source_free_audit_packet",
        "prelock_readiness_approval",
    }
    | QUALIFICATION_ARTIFACT_KEYS
)

TEST_ATTESTED_FILE_KEYS = frozenset(
    {
        "exp35_preregistration",
        "exp35_contract",
        "exp35_lock_builder",
        "exp35_source_runner",
        "exp35_orchestrator",
        "exp35_monitor",
        "exp35_fixture_generator",
        "exp35_memory_worker",
        "exp35_contract_tests",
        "exp35_source_tests",
        "exp35_orchestrator_tests",
        "exp35_monitor_tests",
        "exp35_memory_tests",
        "exp31_regression_tests",
        "exp32_regression_tests",
        "exp33_regression_tests",
        "exp34_contract_tests",
        "exp34_source_tests",
        "exp34_orchestrator_tests",
        "exp34_monitor_tests",
        "exp31_filter_module",
        "exp32_v5_engine",
        "frozen_parent_core",
        "frozen_parent_science",
        "frozen_parent_runner",
        "renderer",
        "environment_attestation",
    }
    | QUALIFICATION_ARTIFACT_KEYS
)

AUDITED_FILE_KEYS = frozenset(
    set(REQUIRED_BOUND_FILES)
    - {
        "source_free_audit_packet",
        "prelock_readiness_approval",
    }
)


AUTHORIZED_OUTPUTS = {
    "result_namespace": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1"
    ),
    "primary_source": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/source"
    ),
    "replay_source": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/source"
    ),
    "primary_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/measurement"
    ),
    "replay_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/measurement"
    ),
    "primary_renderer": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/renderer"
    ),
    "replay_renderer": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/renderer"
    ),
    "primary_scan_accounting_journal": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/measurement/"
        "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    ),
    "replay_scan_accounting_journal": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/measurement/"
        "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    ),
    "primary_lifecycle_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/measurement/"
        "SOURCE_LIFECYCLE_MEASUREMENT.json"
    ),
    "replay_lifecycle_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/measurement/"
        "SOURCE_LIFECYCLE_MEASUREMENT.json"
    ),
    "primary_source_launch_capability": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/measurement/"
        "SOURCE_LAUNCH_CAPABILITY.json"
    ),
    "replay_source_launch_capability": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/measurement/"
        "SOURCE_LAUNCH_CAPABILITY.json"
    ),
    "primary_source_child_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/primary/measurement/"
        "SOURCE_CHILD.claim.json"
    ),
    "replay_source_child_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/replay/measurement/"
        "SOURCE_CHILD.claim.json"
    ),
    "manual_native_review": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/"
        "EXP_CAD_0035_MANUAL_NATIVE_REVIEW.json"
    ),
    "orchestration_result": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/"
        "EXP_CAD_0035_ORCHESTRATION_RESULT.json"
    ),
    "global_result": "output/research/autoresearch-loop/EXP_CAD_0035_RESULT.json",
    "launch_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0035-cfpdg-v5-sequential-intake-attempt1/"
        "EXP_CAD_0035_LAUNCH.claim.json"
    ),
}


SOURCE_MODE_OUTPUT_KEYS = {
    "PRIMARY": {
        "source_root": "primary_source",
        "measurement_root": "primary_measurement",
        "scan_accounting_journal": "primary_scan_accounting_journal",
        "lifecycle_measurement": "primary_lifecycle_measurement",
        "launch_claim": "launch_claim",
        "source_launch_capability": "primary_source_launch_capability",
        "source_child_claim": "primary_source_child_claim",
    },
    "REPLAY": {
        "source_root": "replay_source",
        "measurement_root": "replay_measurement",
        "scan_accounting_journal": "replay_scan_accounting_journal",
        "lifecycle_measurement": "replay_lifecycle_measurement",
        "launch_claim": "launch_claim",
        "source_launch_capability": "replay_source_launch_capability",
        "source_child_claim": "replay_source_child_claim",
    },
}

ORCHESTRATOR_CONSUMED_PATH_KEYS = frozenset(
    {
        "exp35_source_runner",
        "exp35_monitor",
        "exp35_orchestrator",
        "renderer",
        "parent_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "accepted_mesh",
        "scan_N009",
        "scan_N010",
        "transform_N009",
        "transform_N010",
    }
)
SOURCE_CONSUMED_PATH_KEYS = frozenset(
    {
        "exp35_source_runner",
        "parent_preregistration",
        "exp31_preregistration",
        "exp32_preregistration",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "accepted_mesh",
        "scan_N009",
        "scan_N010",
        "transform_N009",
        "transform_N010",
        "exp31_filter_module",
        "exp32_v5_engine",
        "frozen_parent_core",
        "frozen_parent_science",
        "frozen_parent_runner",
    }
)
ORCHESTRATOR_OUTPUT_PATH_KEYS = frozenset(AUTHORIZED_OUTPUTS)
SOURCE_OUTPUT_PATH_KEYS = frozenset(SOURCE_MODE_OUTPUT_KEYS["PRIMARY"])


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def null_canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): null_canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [null_canonicalize(item) for item in value]
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and hasattr(value, "tolist"):
        return null_canonicalize(value.tolist())
    if module == "numpy" and hasattr(value, "item"):
        return null_canonicalize(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        null_canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _null_canonicalize_for_stream(value: Any) -> Any:
    """Canonicalize in place where possible, avoiding a second payload tree.

    The physical visual payload can be hundreds of megabytes.  A recursive
    copy would overlap the already-materialized scientific object and can
    violate the frozen 4-GiB process cap.  The caller publishes and then
    discards these objects, so replacing non-finite scalar leaves in mutable
    containers is byte-equivalent and safe.
    """

    if isinstance(value, dict):
        for key in list(value):
            item = value.pop(key)
            value[str(key)] = _null_canonicalize_for_stream(item)
        return value
    if isinstance(value, list):
        for index in range(len(value)):
            value[index] = _null_canonicalize_for_stream(value[index])
        return value
    if isinstance(value, tuple):
        return [_null_canonicalize_for_stream(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _null_canonicalize_for_stream(item)
            for key, item in value.items()
        }
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and hasattr(value, "tolist"):
        return _null_canonicalize_for_stream(value.tolist())
    if module == "numpy" and hasattr(value, "item"):
        return _null_canonicalize_for_stream(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json_write(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for text in encoder.iterencode(_null_canonicalize_for_stream(value)):
                block = text.encode("utf-8")
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest.hexdigest()


def _normalized_repo_relative(value: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("path must be a nonempty POSIX repo-relative string")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"path is not normalized repo-relative: {value}")
    root = ROOT.resolve(strict=True)
    lexical = root / relative
    probe = root
    for part in relative.parts:
        probe = probe / part
        if probe.exists() and probe.is_symlink():
            raise ValueError(f"path traverses symlink: {value}")
    resolved = lexical.resolve(strict=must_exist)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes repository: {value}")
    return resolved


def _lexical_scan_path_only(value: str) -> str:
    """Validate a frozen scan path without any filesystem metadata access."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("scan path must be a nonempty POSIX repo-relative string")
    relative = Path(value)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValueError(f"scan path is not normalized repo-relative: {value}")
    if relative.as_posix() != value:
        raise ValueError(f"scan path has an alternate spelling: {value}")
    return value


def repo_relative(path: Path, *, must_exist: bool) -> str:
    resolved = Path(path).resolve(strict=must_exist)
    root = ROOT.resolve(strict=True)
    relative = resolved.relative_to(root).as_posix()
    if _normalized_repo_relative(relative, must_exist=must_exist) != resolved:
        raise ValueError("path does not have one canonical repository spelling")
    return relative


def file_binding(path: Path) -> dict[str, Any]:
    relative = repo_relative(path, must_exist=True)
    resolved = _normalized_repo_relative(relative, must_exist=True)
    if not resolved.is_file():
        raise ValueError(f"bound path is not a regular file: {relative}")
    return {
        "path": relative,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_path(resolved),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def runtime_environment_fingerprint() -> dict[str, str]:
    """Recompute the attested runtime without importing scientific packages."""

    invocation = Path(sys.executable)
    realpath = invocation.resolve(strict=True)
    soabi = sysconfig.get_config_var("SOABI")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(soabi, str) or not soabi:
        raise ValueError("runtime Python SOABI is unavailable")
    if not isinstance(extension_suffix, str) or not extension_suffix:
        raise ValueError("runtime Python extension suffix is unavailable")

    def origin(name: str) -> str:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"runtime module is unavailable: {name}")
        return str(Path(spec.origin).resolve(strict=True))

    return {
        "python_executable_invocation_path": str(invocation),
        "python_executable_realpath": str(realpath),
        "python_executable_sha256": sha256_path(realpath),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "python_soabi": soabi,
        "python_extension_suffix": extension_suffix,
        "python_abi_flags": str(getattr(sys, "abiflags", "")),
        "python_compiler": platform.python_compiler(),
        "python_byteorder": sys.byteorder,
        "python_pointer_bits": str(struct.calcsize("P") * 8),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "open3d_version": importlib.metadata.version("open3d"),
        "numpy_module_origin": origin("numpy"),
        "scipy_module_origin": origin("scipy"),
        "open3d_module_origin": origin("open3d"),
    }


def _expected_preregistered_files(prereg: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    lineage = prereg["exp34_terminal_lineage"]
    source = prereg["immutable_source_and_cad"]
    dependencies = prereg["immutable_science_and_interface"]
    return {
        "exp35_preregistration": (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        "exp34_preregistration": (
            lineage["preregistration"]["path"],
            lineage["preregistration"]["sha256"],
        ),
        "exp34_terminal_result": (
            lineage["terminal_result"]["path"],
            lineage["terminal_result"]["sha256"],
        ),
        "exp34_terminal_report": (
            lineage["terminal_report"]["path"],
            lineage["terminal_report"]["sha256"],
        ),
        "exp34_terminal_orchestration": (
            lineage["orchestration_result"]["path"],
            lineage["orchestration_result"]["sha256"],
        ),
        "exp34_process_measurement": (
            lineage["process_measurement"]["path"],
            lineage["process_measurement"]["sha256"],
        ),
        "exp34_scan_accounting_journal": (
            lineage["scan_accounting_journal"]["path"],
            lineage["scan_accounting_journal"]["sha256"],
        ),
        "exp34_launch_claim": (
            lineage["launch_claim"]["path"],
            lineage["launch_claim"]["sha256"],
        ),
        "exp34_contract": tuple(lineage["exp34_execution_contract"].values()),
        "exp34_lock_builder": tuple(lineage["exp34_lock_builder"].values()),
        "exp34_monitor": tuple(lineage["exp34_process_monitor"].values()),
        "exp34_source_runner": (
            lineage["exp34_batch_source_runner"]["path"],
            lineage["exp34_batch_source_runner"]["sha256"],
        ),
        "exp34_orchestrator": tuple(lineage["exp34_orchestrator"].values()),
        "exp34_test_attestation": tuple(
            lineage["exp34_source_free_test_attestation"].values()
        ),
        "exp34_environment_attestation": tuple(
            lineage["exp34_environment_attestation"].values()
        ),
        "exp34_audit_packet": tuple(
            lineage["exp34_source_free_audit_packet"].values()
        ),
        "exp34_prelock_approval": tuple(
            lineage["exp34_prelock_readiness_approval"].values()
        ),
        "exp34_lock": tuple(lineage["exp34_immutable_lock"].values()),
        "exp34_detached_approval": tuple(lineage["exp34_detached_approval"].values()),
        "parent_preregistration": tuple(
            dependencies["frozen_parent_preregistration"].values()
        ),
        "exp31_preregistration": tuple(dependencies["exp31_preregistration"].values()),
        "exp32_preregistration": tuple(dependencies["exp32_preregistration"].values()),
        "exp31_filter_module": (
            dependencies["exp31_filter_module"]["path"],
            dependencies["exp31_filter_module"]["sha256"],
        ),
        "exp31_source_runner_lineage": (
            dependencies["exp31_v4_source_runner_lineage"]["path"],
            dependencies["exp31_v4_source_runner_lineage"]["sha256"],
        ),
        "exp32_v5_engine": (
            dependencies["qualified_v5_engine"]["path"],
            dependencies["qualified_v5_engine"]["sha256"],
        ),
        "exp32_source_free_lineage_runner": (
            dependencies["forbidden_lineage_only_v5_source_free_wrapper"]["path"],
            dependencies["forbidden_lineage_only_v5_source_free_wrapper"]["sha256"],
        ),
        "frozen_parent_core": tuple(dependencies["frozen_parent_core"].values()),
        "frozen_parent_science": tuple(dependencies["frozen_parent_science"].values()),
        "frozen_parent_runner": tuple(dependencies["frozen_parent_runner"].values()),
        "renderer": tuple(dependencies["exact_parent_renderer"].values()),
        "source_manifest": (
            source["authenticated_source_manifest"]["path"],
            source["authenticated_source_manifest"]["sha256"],
        ),
        "source_cad": (
            source["source_cad"]["path"],
            source["source_cad"]["sha256"],
        ),
        "mesh_report": tuple(source["mesh_report"].values()),
        "accepted_mesh": tuple(source["accepted_mesh"].values()),
        "cad_preflight": tuple(source["cad_preflight"].values()),
        "cad_nodes": (source["cad_nodes"]["path"], source["cad_nodes"]["sha256"]),
        "cad_edges": (source["cad_edges"]["path"], source["cad_edges"]["sha256"]),
        "cad_edge_geodesic_mm": (
            source["cad_edge_geodesic_mm"]["path"],
            source["cad_edge_geodesic_mm"]["sha256"],
        ),
        "transform_N009": (
            source["frozen_transforms"]["N009"]["path"],
            source["frozen_transforms"]["N009"]["sha256"],
        ),
        "transform_N010": (
            source["frozen_transforms"]["N010"]["path"],
            source["frozen_transforms"]["N010"]["sha256"],
        ),
    }


def _validate_preregistered_files(
    prereg: Mapping[str, Any], bound: Mapping[str, Mapping[str, Any]]
) -> None:
    expected = _expected_preregistered_files(prereg)
    for key, (path, digest) in expected.items():
        row = bound.get(key)
        if row is None or row.get("path") != path or row.get("sha256") != digest:
            raise ValueError(f"lock row {key} does not match EXP35 preregistration")
    for key, path in sorted(CANONICAL_FRESH_BOUND_PATHS.items()):
        row = bound.get(key)
        if row is None or row.get("path") != path:
            raise ValueError(f"lock row {key} is not at its canonical EXP35 path")


def _require_exact_zero_forbidden_counts(
    value: Any, *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(FORBIDDEN_ACCESS_COUNT_KEYS):
        raise ValueError(f"{label} forbidden-access counter inventory is not exact")
    for key in sorted(FORBIDDEN_ACCESS_COUNT_KEYS):
        observed = value[key]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed != 0:
            raise ValueError(f"{label} has nonzero or malformed {key}")
    return value


def _validate_exact_source_free_result(value: Any, *, label: str) -> Mapping[str, Any]:
    expected_keys = {
        "collected_count",
        "passed_count",
        "failed_count",
        "error_count",
        "skipped_count",
        "returncode",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} result inventory is not exact")
    for key in sorted(expected_keys):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise ValueError(f"{label} result {key} is not an exact integer")
    if EXPECTED_SOURCE_FREE_TEST_COUNT <= 0 or not EXPECTED_SOURCE_FREE_TEST_IDS_SHA256:
        raise ValueError("EXP35 expected source-free test identity has not been frozen")
    if (
        value["collected_count"] != EXPECTED_SOURCE_FREE_TEST_COUNT
        or value["passed_count"] != EXPECTED_SOURCE_FREE_TEST_COUNT
        or any(
            value[key] != 0
            for key in ("failed_count", "error_count", "skipped_count", "returncode")
        )
    ):
        raise ValueError(f"{label} result is not the exact frozen all-pass result")
    return value


def _validate_exact_source_free_commands_and_ids(
    row: Mapping[str, Any], *, label: str
) -> None:
    if row.get("collect_command") != list(SOURCE_FREE_COLLECT_COMMAND):
        raise ValueError(f"{label} collection command is not exact")
    if row.get("test_command") != list(SOURCE_FREE_TEST_COMMAND):
        raise ValueError(f"{label} test command is not exact")
    test_ids = row.get("collected_test_ids")
    if (
        not isinstance(test_ids, list)
        or len(test_ids) != EXPECTED_SOURCE_FREE_TEST_COUNT
        or any(not isinstance(item, str) or not item for item in test_ids)
        or len(set(test_ids)) != len(test_ids)
    ):
        raise ValueError(f"{label} collected test IDs are not exact")
    if hashlib.sha256(canonical_bytes(test_ids)).hexdigest() != (
        EXPECTED_SOURCE_FREE_TEST_IDS_SHA256
    ):
        raise ValueError(f"{label} collected test ID digest differs from the freeze")


def _validate_environment_attestation(row: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "experiment_id",
        "status",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "physical_scan_or_evaluator_input_used",
        "generalization_evidence",
        "environment",
    }
    if set(row) != expected_keys:
        raise ValueError("EXP35 environment attestation field inventory is not exact")
    if row.get("schema") != ENVIRONMENT_ATTESTATION_SCHEMA:
        raise ValueError("wrong EXP35 environment attestation schema")
    if row.get("experiment_id") != EXPERIMENT_ID or row.get("status") != "FROZEN":
        raise ValueError("wrong EXP35 environment attestation identity/status")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("environment attestation cannot authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("environment attestation cannot authorize evaluator access")
    if row.get("physical_scan_or_evaluator_input_used") is not False:
        raise ValueError("environment attestation reports forbidden input use")
    if row.get("generalization_evidence") is not False:
        raise ValueError("environment attestation cannot claim generalization evidence")
    environment = row.get("environment")
    parent_environment = parent.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(ENVIRONMENT_FINGERPRINT_KEYS)
        or not isinstance(parent_environment, Mapping)
        or any(environment.get(key) != value for key, value in parent_environment.items())
    ):
        raise ValueError("EXP35 runtime environment differs from frozen EXP34 environment")
    if environment != runtime_environment_fingerprint():
        raise ValueError("EXP35 environment attestation differs from the live ABI/platform")


def _validate_test_attestation(
    row: Mapping[str, Any], *, bound: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    expected_keys = {
        "schema",
        "experiment_id",
        "status",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "generalization_evidence",
        "result",
        "forbidden_access_counts",
        "collected_test_ids",
        "tested_file_sha256",
        "collect_command",
        "test_command",
    }
    if set(row) != expected_keys:
        raise ValueError("EXP35 source-free test attestation field inventory is not exact")
    if row.get("schema") != TEST_ATTESTATION_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP35 source-free test attestation identity")
    if row.get("status") != "PASS_SOURCE_FREE_TESTS":
        raise ValueError("EXP35 source-free tests are not PASS")
    if (
        row.get("physical_source_execution_authorized") is not False
        or row.get("evaluator_input_authorized") is not False
        or row.get("generalization_evidence") is not False
    ):
        raise ValueError("source-free test attestation carries forbidden authority")
    _validate_exact_source_free_result(
        row.get("result"), label="source-free test attestation"
    )
    _require_exact_zero_forbidden_counts(
        row.get("forbidden_access_counts"), label="source-free test attestation"
    )
    _validate_exact_source_free_commands_and_ids(
        row, label="source-free test attestation"
    )
    tested = row.get("tested_file_sha256")
    if not isinstance(tested, Mapping):
        raise ValueError("source-free tested-file hash map is absent")
    if bound is not None:
        if set(tested) != set(TEST_ATTESTED_FILE_KEYS):
            raise ValueError("source-free tested-file hash key set is not exact")
        for key in sorted(TEST_ATTESTED_FILE_KEYS):
            if tested[key] != bound[key]["sha256"]:
                raise ValueError(f"source-free tested-file hash mismatch for {key}")


def _validate_audit_packet(
    row: Mapping[str, Any], *, bound: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    expected_keys = {
        "schema",
        "experiment_id",
        "status",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "physical_scan_or_evaluator_input_used",
        "generalization_evidence",
        "source_free_regression",
        "forbidden_access_counts",
        "collect_command",
        "test_command",
        "collected_test_ids",
        "audit_findings",
        "audited_file_sha256",
    }
    if set(row) != expected_keys:
        raise ValueError("EXP35 source-free audit packet field inventory is not exact")
    if row.get("schema") != AUDIT_PACKET_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP35 source-free audit packet identity")
    if row.get("status") != "PASS_SOURCE_FREE_AUDIT_EXECUTION_NOT_AUTHORIZED":
        raise ValueError("EXP35 source-free audit did not pass")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("source-free audit packet cannot authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("source-free audit packet cannot authorize evaluator access")
    if row.get("physical_scan_or_evaluator_input_used") is not False:
        raise ValueError("source-free audit packet reports forbidden access")
    if row.get("generalization_evidence") is not False:
        raise ValueError("source-free audit packet cannot claim generalization evidence")
    _require_exact_zero_forbidden_counts(
        row.get("forbidden_access_counts"), label="source-free audit packet"
    )
    _validate_exact_source_free_result(
        row.get("source_free_regression"), label="source-free audit packet"
    )
    _validate_exact_source_free_commands_and_ids(
        row, label="source-free audit packet"
    )
    findings = row.get("audit_findings")
    if (
        not isinstance(findings, list)
        or not findings
        or any(not isinstance(item, str) or not item for item in findings)
    ):
        raise ValueError("source-free audit findings are absent or malformed")
    if bound is not None:
        audited = row.get("audited_file_sha256")
        if not isinstance(audited, Mapping) or set(audited) != set(AUDITED_FILE_KEYS):
            raise ValueError("source-free audit packet hash key set is not exact")
        for key in sorted(AUDITED_FILE_KEYS):
            if audited[key] != bound[key]["sha256"]:
                raise ValueError(f"source-free audit packet hash mismatch for {key}")


def _validate_prelock_approval(
    row: Mapping[str, Any], *, bound: Mapping[str, Mapping[str, Any]]
) -> None:
    expected_fields = {
        "schema",
        "experiment_id",
        "status",
        "immutable_lock_materialization_authorized",
        "physical_source_execution_authorized",
        "physical_orchestrator_invocation_authorized",
        "evaluator_input_authorized",
        "physical_scan_or_evaluator_input_used",
        "detached_postlock_physical_execution_approval_still_required",
        "generalization_evidence",
        "source_free_test_result",
        "forbidden_access_counts",
        "collect_command",
        "test_command",
        "collected_test_ids",
        "approved_bound_file_sha256",
    }
    if set(row) != expected_fields:
        raise ValueError("EXP35 prelock approval field inventory is not exact")
    if row.get("schema") != PRELOCK_APPROVAL_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP35 prelock approval identity")
    if row.get("status") != "APPROVED_TO_MATERIALIZE_IMMUTABLE_PHYSICAL_LOCK":
        raise ValueError("EXP35 prelock approval status is not exact")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("prelock approval cannot authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("prelock approval cannot authorize evaluator access")
    if (
        row.get("physical_orchestrator_invocation_authorized") is not False
        or row.get("immutable_lock_materialization_authorized") is not True
        or row.get("detached_postlock_physical_execution_approval_still_required")
        is not True
        or row.get("physical_scan_or_evaluator_input_used") is not False
        or row.get("generalization_evidence") is not False
    ):
        raise ValueError("EXP35 prelock approval authority boundary is not exact")
    _require_exact_zero_forbidden_counts(
        row.get("forbidden_access_counts"), label="EXP35 prelock approval"
    )
    _validate_exact_source_free_result(
        row.get("source_free_test_result"), label="EXP35 prelock approval"
    )
    _validate_exact_source_free_commands_and_ids(
        row, label="EXP35 prelock approval"
    )
    test_attestation = _load_json(
        _normalized_repo_relative(
            bound["source_free_test_attestation"]["path"], must_exist=True
        )
    )
    for field in (
        "result",
        "collect_command",
        "test_command",
        "collected_test_ids",
    ):
        expected = (
            row["source_free_test_result"] if field == "result" else row[field]
        )
        if expected != test_attestation.get(field):
            raise ValueError(f"EXP35 prelock approval test evidence mismatch: {field}")
    approved = row.get("approved_bound_file_sha256")
    expected_keys = set(REQUIRED_BOUND_FILES) - {"prelock_readiness_approval"}
    if not isinstance(approved, Mapping) or set(approved) != expected_keys:
        raise ValueError("prelock approval hash-key set is not exact")
    for key in sorted(expected_keys):
        if approved[key] != bound[key]["sha256"]:
            raise ValueError(f"prelock approval hash mismatch for {key}")


def _verify_bound_files(
    bound: Mapping[str, Mapping[str, Any]], *, reread_payloads: bool
) -> None:
    if set(bound) != set(REQUIRED_BOUND_FILES):
        raise ValueError("EXP35 lock bound-file key set is not exact")
    for key, row in sorted(bound.items()):
        if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"malformed lock binding row: {key}")
        path = _normalized_repo_relative(str(row["path"]), must_exist=True)
        if not path.is_file() or int(row["size_bytes"]) != int(path.stat().st_size):
            raise ValueError(f"bound file size mismatch: {key}")
        if sha256_path(path) != row["sha256"]:
            raise ValueError(f"bound file hash mismatch: {key}")
    prereg = _load_json(_normalized_repo_relative(PREREGISTRATION_PATH, must_exist=True))
    _validate_preregistered_files(prereg, bound)
    if reread_payloads:
        _validate_environment_attestation(
            _load_json(_normalized_repo_relative(bound["environment_attestation"]["path"], must_exist=True)),
            _load_json(_normalized_repo_relative(bound["exp34_environment_attestation"]["path"], must_exist=True)),
        )
        _validate_test_attestation(
            _load_json(_normalized_repo_relative(bound["source_free_test_attestation"]["path"], must_exist=True)),
            bound=bound,
        )
        _validate_audit_packet(
            _load_json(_normalized_repo_relative(bound["source_free_audit_packet"]["path"], must_exist=True)),
            bound=bound,
        )
        _validate_prelock_approval(
            _load_json(_normalized_repo_relative(bound["prelock_readiness_approval"]["path"], must_exist=True)),
            bound=bound,
        )


def _physical_scan_rows(prereg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for token in ("N009", "N010"):
        source = prereg["immutable_source_and_cad"]["physical_scans"][token]
        _lexical_scan_path_only(source["path"])
        rows[f"scan_{token}"] = {
            "opaque_token": token,
            "path": source["path"],
            "expected_sha256": source["expected_sha256"],
            "expected_raw_point_row_count": int(source["expected_raw_point_row_count"]),
            "bytes_opened_or_hashed_during_lock_construction": False,
        }
    return rows


def _authorized_output_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, relative in AUTHORIZED_OUTPUTS.items():
        _normalized_repo_relative(relative, must_exist=False)
        rows[key] = relative
    roots = [
        Path(rows[key])
        for key in (
            "primary_source",
            "replay_source",
            "primary_measurement",
            "replay_measurement",
            "primary_renderer",
            "replay_renderer",
        )
    ]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first in second.parents or second in first.parents or first == second:
                raise ValueError("EXP35 compared/noncompared roots overlap")
    return rows


def build_lock_payload(*, bindings: Mapping[str, Path], lock_path: Path) -> dict[str, Any]:
    if set(bindings) != set(REQUIRED_BOUND_FILES):
        missing = sorted(REQUIRED_BOUND_FILES - set(bindings))
        extra = sorted(set(bindings) - REQUIRED_BOUND_FILES)
        raise ValueError(f"EXP35 lock keys differ; missing={missing}, extra={extra}")
    if repo_relative(lock_path, must_exist=False) != LOCK_PATH:
        raise ValueError("EXP35 immutable lock path is not canonical")
    bound = {key: file_binding(path) for key, path in sorted(bindings.items())}
    if bound["exp35_preregistration"]["sha256"] != PREREGISTRATION_SHA256:
        raise ValueError("EXP35 preregistration hash is not frozen")
    _verify_bound_files(bound, reread_payloads=True)
    prereg = _load_json(_normalized_repo_relative(PREREGISTRATION_PATH, must_exist=True))
    environment_row = _load_json(
        _normalized_repo_relative(bound["environment_attestation"]["path"], must_exist=True)
    )
    environment = environment_row.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("EXP35 environment attestation lacks an environment object")
    return {
        "schema": LOCK_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": LOCK_STATUS,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authorized_orchestrator_invocation_count": 1,
        "maximum_primary_source_child_count": 1,
        "maximum_replay_source_child_count": 1,
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "bound_files": bound,
        "physical_scans": _physical_scan_rows(prereg),
        "authorized_outputs": _authorized_output_rows(),
        "detached_approval_path": DETACHED_APPROVAL_PATH,
        "resource_limits": RESOURCE_LIMITS,
        "environment": dict(environment),
        "command_templates": command_templates(environment),
    }


def _exact_cli_path(path: Path, relative: str, *, must_exist: bool) -> Path:
    root = ROOT.resolve(strict=True)
    expected = root / relative
    # CLI paths have exactly one accepted spelling: canonical absolute path.
    if str(path) != expected.as_posix():
        raise ValueError(f"CLI path is not the canonical absolute locked path: {path}")
    probe = root
    for part in Path(relative).parts:
        probe = probe / part
        if probe.exists() and probe.is_symlink():
            raise ValueError(f"CLI path traverses a symlink: {path}")
    resolved = Path(path).resolve(strict=must_exist)
    if resolved != expected.resolve(strict=must_exist):
        raise ValueError(f"CLI path does not resolve to locked path: {path}")
    return resolved


def verify_lock(lock_path: Path, lock_sha256: str) -> Mapping[str, Any]:
    _exact_cli_path(lock_path, LOCK_PATH, must_exist=True)
    if sha256_path(lock_path) != lock_sha256:
        raise ValueError("EXP35 lock SHA-256 mismatch")
    lock = _load_json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA or lock.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP35 lock identity")
    if lock.get("status") != LOCK_STATUS:
        raise ValueError("wrong EXP35 lock status")
    if lock.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("lock preregistration binding mismatch")
    if lock.get("physical_source_execution_authorized") is not False:
        raise ValueError("immutable lock may not self-authorize physical execution")
    if lock.get("evaluator_input_authorized") is not False:
        raise ValueError("immutable lock may not authorize evaluator access")
    expected_keys = {
        "schema",
        "experiment_id",
        "status",
        "preregistration_sha256",
        "authorized_orchestrator_invocation_count",
        "maximum_primary_source_child_count",
        "maximum_replay_source_child_count",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "bound_files",
        "physical_scans",
        "authorized_outputs",
        "detached_approval_path",
        "resource_limits",
        "environment",
        "command_templates",
    }
    if set(lock) != expected_keys:
        raise ValueError("EXP35 immutable lock field inventory is not exact")
    if (
        lock.get("authorized_orchestrator_invocation_count") != 1
        or lock.get("maximum_primary_source_child_count") != 1
        or lock.get("maximum_replay_source_child_count") != 1
        or lock.get("detached_approval_path") != DETACHED_APPROVAL_PATH
        or lock.get("resource_limits") != RESOURCE_LIMITS
    ):
        raise ValueError("EXP35 immutable lock authority/resource values drifted")
    _verify_bound_files(lock.get("bound_files", {}), reread_payloads=True)
    environment_row = _load_json(
        _normalized_repo_relative(
            lock["bound_files"]["environment_attestation"]["path"],
            must_exist=True,
        )
    )
    if (
        lock.get("environment") != environment_row.get("environment")
        or lock.get("command_templates") != command_templates(lock["environment"])
    ):
        raise ValueError("EXP35 immutable environment/command templates drifted")
    prereg = _load_json(_normalized_repo_relative(PREREGISTRATION_PATH, must_exist=True))
    if lock.get("physical_scans") != _physical_scan_rows(prereg):
        raise ValueError("physical scan path/expected-hash rows differ from preregistration")
    if lock.get("authorized_outputs") != _authorized_output_rows():
        raise ValueError("authorized output topology differs from preregistration")
    return lock


def _verify_detached_approval(
    path: Path,
    digest: str,
    *,
    lock_path: Path,
    lock_sha256: str,
    lock: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact_cli_path(path, DETACHED_APPROVAL_PATH, must_exist=True)
    if sha256_path(path) != digest:
        raise ValueError("detached EXP35 approval SHA-256 mismatch")
    row = _load_json(path)
    expected_fields = {
        "schema",
        "experiment_id",
        "architecture",
        "status",
        "preregistration_path",
        "preregistration_sha256",
        "lock_path",
        "lock_sha256",
        "prelock_approval_path",
        "prelock_approval_sha256",
        "source_free_audit_packet_path",
        "source_free_audit_packet_sha256",
        "test_attestation_path",
        "test_attestation_sha256",
        "environment_attestation_path",
        "environment_attestation_sha256",
        "approved_bound_file_sha256",
        "physical_source_execution_authorized",
        "evaluator_input_authorized",
        "authorized_orchestrator_invocations",
        "maximum_primary_source_children",
        "maximum_conditional_replay_source_children",
        "replay_only_after_primary_success",
        "retry_rescue_or_cap_extension_permitted",
        "manual_native_review_required_when_visual_payload_exists",
        "manual_review_may_override_failure",
        "evaluator_roles_labels_gold_expected_counts_or_quality_access_permitted",
        "generalization_evidence",
        "accuracy_precision_recall_or_f1_authorized",
        "quality_rank_authorized",
        "bend_line_f1_authorized",
        "production_promotion_authorized",
    }
    if set(row) != expected_fields:
        raise ValueError("detached EXP35 approval field inventory is not exact")
    if row.get("schema") != DETACHED_APPROVAL_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong detached EXP35 approval identity")
    if row.get("architecture") != ARCHITECTURE:
        raise ValueError("detached EXP35 approval architecture is not exact")
    if row.get("status") != "APPROVED_FOR_ONE_EXP35_ORCHESTRATOR_INVOCATION":
        raise ValueError("detached EXP35 approval decision is not exact")
    if row.get("physical_source_execution_authorized") is not True:
        raise ValueError("detached EXP35 approval does not authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("detached EXP35 approval authorizes evaluator access")
    if type(row.get("authorized_orchestrator_invocations")) is not int or row.get(
        "authorized_orchestrator_invocations"
    ) != 1:
        raise ValueError("detached approval invocation budget is not one")
    if type(row.get("maximum_primary_source_children")) is not int or row.get(
        "maximum_primary_source_children"
    ) != 1:
        raise ValueError("detached approval primary budget is not one")
    if type(row.get("maximum_conditional_replay_source_children")) is not int or row.get(
        "maximum_conditional_replay_source_children"
    ) != 1:
        raise ValueError("detached approval replay budget is not one")
    preregistration_path = _normalized_repo_relative(
        PREREGISTRATION_PATH, must_exist=True
    )
    if (
        row.get("preregistration_path") != str(preregistration_path)
        or row.get("preregistration_sha256") != PREREGISTRATION_SHA256
    ):
        raise ValueError("detached approval preregistration binding mismatch")
    if row.get("lock_path") != str(lock_path):
        raise ValueError("detached approval lock path mismatch")
    if row.get("lock_sha256") != lock_sha256:
        raise ValueError("detached approval lock SHA-256 mismatch")
    bound = lock.get("bound_files")
    if not isinstance(bound, Mapping) or set(bound) != set(REQUIRED_BOUND_FILES):
        raise ValueError("detached approval cannot bind a malformed immutable lock")
    binding_fields = {
        "prelock_approval": (
            "prelock_readiness_approval",
            "prelock_approval_path",
            "prelock_approval_sha256",
        ),
        "source_free_audit_packet": (
            "source_free_audit_packet",
            "source_free_audit_packet_path",
            "source_free_audit_packet_sha256",
        ),
        "test_attestation": (
            "source_free_test_attestation",
            "test_attestation_path",
            "test_attestation_sha256",
        ),
        "environment_attestation": (
            "environment_attestation",
            "environment_attestation_path",
            "environment_attestation_sha256",
        ),
    }
    for label, (bound_key, path_field, sha_field) in binding_fields.items():
        bound_path = _normalized_repo_relative(
            str(bound[bound_key]["path"]), must_exist=True
        )
        if (
            row.get(path_field) != str(bound_path)
            or row.get(sha_field) != bound[bound_key]["sha256"]
        ):
            raise ValueError(f"detached approval {label} binding mismatch")
    approved = row.get("approved_bound_file_sha256")
    if (
        not isinstance(approved, Mapping)
        or set(approved) != set(REQUIRED_BOUND_FILES)
        or any(approved[key] != bound[key]["sha256"] for key in REQUIRED_BOUND_FILES)
    ):
        raise ValueError("detached approval bound-file hash inventory is not exact")
    if (
        row.get("replay_only_after_primary_success") is not True
        or row.get("retry_rescue_or_cap_extension_permitted") is not False
        or row.get("manual_native_review_required_when_visual_payload_exists")
        is not True
        or row.get("manual_review_may_override_failure") is not False
    ):
        raise ValueError("detached EXP35 approval retry/review policy is not exact")
    for field in (
        "evaluator_roles_labels_gold_expected_counts_or_quality_access_permitted",
        "generalization_evidence",
        "accuracy_precision_recall_or_f1_authorized",
        "quality_rank_authorized",
        "bend_line_f1_authorized",
        "production_promotion_authorized",
    ):
        if row.get(field) is not False:
            raise ValueError(f"detached EXP35 approval grants forbidden {field}")
    return row


def verify_execution_authority(
    *,
    preregistration_path: Path,
    lock_path: Path,
    lock_sha256: str,
    detached_approval_path: Path,
    detached_approval_sha256: str,
    consumed_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    """Verify combined lock+detached authority without opening scan bytes."""

    if not isinstance(consumed_paths, Mapping) or not isinstance(output_paths, Mapping):
        raise ValueError("EXP35 authority path maps must be mappings")
    consumed_keys = frozenset(consumed_paths)
    output_keys = frozenset(output_paths)
    source_shape = (
        consumed_keys == SOURCE_CONSUMED_PATH_KEYS
        and output_keys == SOURCE_OUTPUT_PATH_KEYS
    )
    orchestrator_shape = (
        consumed_keys == ORCHESTRATOR_CONSUMED_PATH_KEYS
        and output_keys == ORCHESTRATOR_OUTPUT_PATH_KEYS
    )
    if not source_shape and not orchestrator_shape:
        raise ValueError(
            "EXP35 authority consumed/output key inventories do not match "
            "the exact source or orchestrator shape"
        )
    _exact_cli_path(preregistration_path, PREREGISTRATION_PATH, must_exist=True)
    if sha256_path(preregistration_path) != PREREGISTRATION_SHA256:
        raise ValueError("EXP35 preregistration SHA-256 mismatch")
    lock = verify_lock(lock_path, lock_sha256)
    _verify_detached_approval(
        detached_approval_path,
        detached_approval_sha256,
        lock_path=lock_path,
        lock_sha256=lock_sha256,
        lock=lock,
    )
    bound = lock["bound_files"]
    environment_row = _load_json(
        _normalized_repo_relative(bound["environment_attestation"]["path"], must_exist=True)
    )
    if environment_row.get("environment") != runtime_environment_fingerprint():
        raise ValueError("live EXP35 runtime differs from immutable environment attestation")
    scans = lock["physical_scans"]
    for key, path in sorted(consumed_paths.items()):
        if key in scans:
            # Exact path only.  Deliberately no stat/open/hash of a scan here.
            _exact_cli_path(path, scans[key]["path"], must_exist=True)
        elif key in bound:
            resolved = _exact_cli_path(path, bound[key]["path"], must_exist=True)
            if int(resolved.stat().st_size) != int(bound[key]["size_bytes"]):
                raise ValueError(f"consumed static file size mismatch: {key}")
            if sha256_path(resolved) != bound[key]["sha256"]:
                raise ValueError(f"consumed static file hash mismatch: {key}")
        else:
            raise ValueError(f"consumed path key is not lock-bound: {key}")
    authorized = lock["authorized_outputs"]
    if source_shape:
        mode = infer_source_mode(output_paths, lock)
        for generic_key, locked_key in SOURCE_MODE_OUTPUT_KEYS[mode].items():
            _exact_cli_path(output_paths[generic_key], authorized[locked_key], must_exist=False)
    else:
        for key, path in sorted(output_paths.items()):
            _exact_cli_path(path, authorized[key], must_exist=False)
    return lock


def infer_source_mode(output_paths: Mapping[str, Path], lock: Mapping[str, Any]) -> str:
    authorized = lock["authorized_outputs"]
    for mode, mapping in SOURCE_MODE_OUTPUT_KEYS.items():
        if all(
            key in output_paths
            and str(output_paths[key]) == (ROOT / authorized[locked_key]).as_posix()
            for key, locked_key in mapping.items()
        ):
            return mode
    raise ValueError("source output paths do not match PRIMARY or REPLAY lock topology")
