#!/usr/bin/env python3
"""Shared immutable execution contract for EXP_CAD_0034.

This module is source-free.  It never opens either physical scan.  Physical
scan rows in an EXP34 lock are path-bound to preregistered expected digests;
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
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "EXP_CAD_0034"
PREREGISTRATION_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "323c7cf72884f33e6358471f80b100c5d0e00ec0ca4b795cff5aa6e9afea5058"
)
LOCK_PATH = "docs/research/2026-08-30_exp_cad_0034_physical_execution_lock.json"
PRELOCK_APPROVAL_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0034_prelock_physical_readiness_approval.json"
)
ENVIRONMENT_ATTESTATION_PATH = (
    "docs/research/2026-08-30_exp_cad_0034_physical_environment_attestation.json"
)
DETACHED_APPROVAL_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0034_detached_postlock_physical_execution_approval.json"
)

LOCK_SCHEMA = "EXP_CAD_0034_PHYSICAL_EXECUTION_LOCK_V1"
LOCK_STATUS = "IMMUTABLE_LOCKED_AWAITING_DETACHED_APPROVAL"
PRELOCK_APPROVAL_SCHEMA = "EXP_CAD_0034_PRELOCK_PHYSICAL_READINESS_APPROVAL_V1"
DETACHED_APPROVAL_SCHEMA = (
    "EXP_CAD_0034_DETACHED_POSTLOCK_PHYSICAL_EXECUTION_APPROVAL_V1"
)
ENVIRONMENT_ATTESTATION_SCHEMA = "EXP_CAD_0034_PHYSICAL_ENVIRONMENT_ATTESTATION_V1"
TEST_ATTESTATION_SCHEMA = "EXP_CAD_0034_SOURCE_FREE_TEST_ATTESTATION_V1"
AUDIT_PACKET_SCHEMA = "EXP_CAD_0034_SOURCE_FREE_AUDIT_PACKET_V1"


# Every key is a regular, non-scan file.  The two PLY scans are deliberately
# absent: they live under lock["physical_scans"] and may not be opened or
# hashed by this module or by the lock builder.
REQUIRED_BOUND_FILES = frozenset(
    {
        "exp34_preregistration",
        "exp30_terminal_result",
        "parent_preregistration",
        "exp31_terminal_result",
        "exp31_terminal_report",
        "exp31_terminal_orchestration",
        "exp31_preregistration",
        "exp33_result",
        "exp33_preregistration",
        "exp33_environment_attestation",
        "exp33_lock",
        "exp33_detached_approval",
        "exp33_outer_result",
        "exp33_nested_result",
        "exp33_manual_review",
        "exp32_preregistration",
        "exp31_filter_module",
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
        "exp34_contract",
        "exp34_lock_builder",
        "exp34_source_runner",
        "exp34_orchestrator",
        "exp34_monitor",
        "exp34_contract_tests",
        "exp34_source_tests",
        "exp34_orchestrator_tests",
        "exp34_monitor_tests",
        "exp31_regression_tests",
        "exp32_regression_tests",
        "exp33_regression_tests",
        "source_free_test_attestation",
        "environment_attestation",
        "source_free_audit_packet",
        "prelock_readiness_approval",
    }
)

TEST_ATTESTED_FILE_KEYS = frozenset(
    {
        "exp34_preregistration",
        "exp34_contract",
        "exp34_lock_builder",
        "exp34_source_runner",
        "exp34_orchestrator",
        "exp34_monitor",
        "exp34_contract_tests",
        "exp34_source_tests",
        "exp34_orchestrator_tests",
        "exp34_monitor_tests",
        "exp31_regression_tests",
        "exp32_regression_tests",
        "exp33_regression_tests",
        "exp31_filter_module",
        "exp32_v5_engine",
        "frozen_parent_core",
        "frozen_parent_science",
        "frozen_parent_runner",
        "renderer",
    }
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
        "exp-cad-0034-cfpdg-v5-physical-attempt1"
    ),
    "primary_source": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/source"
    ),
    "replay_source": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/source"
    ),
    "primary_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/measurement"
    ),
    "replay_measurement": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/measurement"
    ),
    "primary_renderer": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/renderer"
    ),
    "replay_renderer": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/renderer"
    ),
    "primary_scan_accounting_journal": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/measurement/"
        "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    ),
    "replay_scan_accounting_journal": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/measurement/"
        "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    ),
    "primary_source_launch_capability": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/measurement/"
        "SOURCE_LAUNCH_CAPABILITY.json"
    ),
    "replay_source_launch_capability": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/measurement/"
        "SOURCE_LAUNCH_CAPABILITY.json"
    ),
    "primary_source_child_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/primary/measurement/"
        "SOURCE_CHILD.claim.json"
    ),
    "replay_source_child_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/replay/measurement/"
        "SOURCE_CHILD.claim.json"
    ),
    "manual_native_review": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/"
        "EXP_CAD_0034_MANUAL_NATIVE_VISUAL_REVIEW.json"
    ),
    "orchestration_result": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/"
        "EXP_CAD_0034_ORCHESTRATION_RESULT.json"
    ),
    "global_result": "output/research/autoresearch-loop/EXP_CAD_0034_RESULT.json",
    "launch_claim": (
        "output/research/autoresearch-loop/"
        "exp-cad-0034-cfpdg-v5-physical-attempt1/"
        "EXP_CAD_0034_LAUNCH.claim.json"
    ),
}


SOURCE_MODE_OUTPUT_KEYS = {
    "PRIMARY": {
        "source_root": "primary_source",
        "measurement_root": "primary_measurement",
        "scan_accounting_journal": "primary_scan_accounting_journal",
        "launch_claim": "launch_claim",
        "source_launch_capability": "primary_source_launch_capability",
        "source_child_claim": "primary_source_child_claim",
    },
    "REPLAY": {
        "source_root": "replay_source",
        "measurement_root": "replay_measurement",
        "scan_accounting_journal": "replay_scan_accounting_journal",
        "launch_claim": "launch_claim",
        "source_launch_capability": "replay_source_launch_capability",
        "source_child_claim": "replay_source_child_claim",
    },
}


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
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "open3d_version": importlib.metadata.version("open3d"),
        "numpy_module_origin": origin("numpy"),
        "scipy_module_origin": origin("scipy"),
        "open3d_module_origin": origin("open3d"),
    }


def _expected_preregistered_files(prereg: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    lineage = prereg["lineage"]
    source = prereg["immutable_source_lineage"]
    cad = prereg["immutable_cad_artifacts"]
    dependencies = prereg["hash_bound_execution_dependencies"]
    return {
        "exp34_preregistration": (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        "exp30_terminal_result": tuple(lineage["exp30_terminal_result"].values()),
        "parent_preregistration": tuple(lineage["exp30_scientific_preregistration"].values()),
        "exp31_terminal_result": tuple(lineage["exp31_terminal_result"].values()),
        "exp31_terminal_report": tuple(lineage["exp31_terminal_report"].values()),
        "exp31_terminal_orchestration": tuple(lineage["exp31_terminal_orchestration"].values()),
        "exp31_preregistration": tuple(lineage["exp31_source_interface_preregistration"].values()),
        "exp33_result": (
            lineage["exp33_source_free_qualification_result"]["path"],
            lineage["exp33_source_free_qualification_result"]["sha256"],
        ),
        "exp33_preregistration": tuple(lineage["exp33_source_free_preregistration"].values()),
        "exp33_environment_attestation": tuple(lineage["exp33_source_free_environment_attestation"].values()),
        "exp33_lock": tuple(lineage["exp33_immutable_qualification_lock"].values()),
        "exp33_detached_approval": tuple(lineage["exp33_detached_postlock_approval"].values()),
        "exp33_outer_result": tuple(lineage["exp33_outer_qualification_artifact"].values()),
        "exp33_nested_result": tuple(lineage["exp33_nested_candidate_artifact"].values()),
        "exp33_manual_review": tuple(lineage["exp33_manual_native_visual_review"].values()),
        "exp32_preregistration": tuple(
            dependencies["exp32_v5_execution_equivalent_preregistration"].values()
        ),
        "exp31_filter_module": (
            dependencies["exact_exp31_filter_module"]["path"],
            dependencies["exact_exp31_filter_module"]["sha256"],
        ),
        "exp32_v5_engine": (
            dependencies["qualified_v5_engine"]["path"],
            dependencies["qualified_v5_engine"]["sha256"],
        ),
        "exp32_source_free_lineage_runner": (
            dependencies["qualified_v5_source_free_call_surface_lineage_only"]["path"],
            dependencies["qualified_v5_source_free_call_surface_lineage_only"]["sha256"],
        ),
        "frozen_parent_core": tuple(dependencies["frozen_parent_core_module"].values()),
        "frozen_parent_science": tuple(dependencies["frozen_parent_science_module"].values()),
        "frozen_parent_runner": tuple(dependencies["frozen_parent_execute_source"].values()),
        "renderer": tuple(dependencies["exact_parent_renderer"].values()),
        "source_manifest": (
            source["authenticated_source_manifest"]["path"],
            source["authenticated_source_manifest"]["sha256"],
        ),
        "source_cad": (
            source["source_cad"]["path"],
            source["source_cad"]["expected_sha256"],
        ),
        "mesh_report": tuple(cad["mesh_report"].values()),
        "accepted_mesh": tuple(cad["accepted_mesh"].values()),
        "cad_preflight": tuple(cad["cad_only_preflight"].values()),
        "cad_nodes": (cad["cad_graph_nodes"]["path"], cad["cad_graph_nodes"]["sha256"]),
        "cad_edges": (cad["cad_graph_edges"]["path"], cad["cad_graph_edges"]["sha256"]),
        "cad_edge_geodesic_mm": (
            cad["cad_graph_edge_geodesic_mm"]["path"],
            cad["cad_graph_edge_geodesic_mm"]["sha256"],
        ),
        "transform_N009": (
            source["frozen_transforms"]["N009"]["path"],
            source["frozen_transforms"]["N009"]["expected_semantic_file_sha256"],
        ),
        "transform_N010": (
            source["frozen_transforms"]["N010"]["path"],
            source["frozen_transforms"]["N010"]["expected_semantic_file_sha256"],
        ),
    }


def _validate_preregistered_files(
    prereg: Mapping[str, Any], bound: Mapping[str, Mapping[str, Any]]
) -> None:
    expected = _expected_preregistered_files(prereg)
    for key, (path, digest) in expected.items():
        row = bound.get(key)
        if row is None or row.get("path") != path or row.get("sha256") != digest:
            raise ValueError(f"lock row {key} does not match EXP34 preregistration")


def _validate_environment_attestation(row: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    if row.get("schema") != ENVIRONMENT_ATTESTATION_SCHEMA:
        raise ValueError("wrong EXP34 environment attestation schema")
    if row.get("experiment_id") != EXPERIMENT_ID or row.get("status") != "FROZEN":
        raise ValueError("wrong EXP34 environment attestation identity/status")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("environment attestation cannot authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("environment attestation cannot authorize evaluator access")
    if row.get("physical_scan_or_evaluator_input_used") is not False:
        raise ValueError("environment attestation reports forbidden input use")
    if row.get("environment") != parent.get("environment"):
        raise ValueError("EXP34 runtime environment differs from frozen EXP33 environment")


def _validate_test_attestation(
    row: Mapping[str, Any], *, bound: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    if row.get("schema") != TEST_ATTESTATION_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP34 source-free test attestation identity")
    if row.get("status") != "PASS_SOURCE_FREE_TESTS":
        raise ValueError("EXP34 source-free tests are not PASS")
    result = row.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("source-free test result is absent")
    if int(result.get("collected_count", 0)) <= 0:
        raise ValueError("source-free test collection is empty")
    if int(result.get("passed_count", -1)) != int(result["collected_count"]):
        raise ValueError("not every collected source-free test passed")
    for key in ("failed_count", "error_count", "skipped_count", "returncode"):
        if int(result.get(key, -1)) != 0:
            raise ValueError(f"source-free test attestation has nonzero {key}")
    counters = row.get("forbidden_access_counts")
    if not isinstance(counters, Mapping) or any(int(value) != 0 for value in counters.values()):
        raise ValueError("source-free test attestation has forbidden input access")
    if not isinstance(row.get("collected_test_ids"), list):
        raise ValueError("source-free test IDs are absent")
    if len(row["collected_test_ids"]) != int(result["collected_count"]):
        raise ValueError("source-free test ID count mismatch")
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
    if row.get("schema") != AUDIT_PACKET_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP34 source-free audit packet identity")
    if row.get("status") != "PASS_SOURCE_FREE_AUDIT_EXECUTION_NOT_AUTHORIZED":
        raise ValueError("EXP34 source-free audit did not pass")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("source-free audit packet cannot authorize physical execution")
    if row.get("physical_scan_or_evaluator_input_used") is not False:
        raise ValueError("source-free audit packet reports forbidden access")
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
    if row.get("schema") != PRELOCK_APPROVAL_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP34 prelock approval identity")
    if row.get("status") != "APPROVED_TO_MATERIALIZE_IMMUTABLE_PHYSICAL_LOCK":
        raise ValueError("EXP34 prelock approval status is not exact")
    if row.get("physical_source_execution_authorized") is not False:
        raise ValueError("prelock approval cannot authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("prelock approval cannot authorize evaluator access")
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
        raise ValueError("EXP34 lock bound-file key set is not exact")
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
            _load_json(_normalized_repo_relative(bound["exp33_environment_attestation"]["path"], must_exist=True)),
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
        source = prereg["immutable_source_lineage"]["physical_scans"][token]
        _normalized_repo_relative(source["path"], must_exist=False)
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
                raise ValueError("EXP34 compared/noncompared roots overlap")
    return rows


def build_lock_payload(*, bindings: Mapping[str, Path], lock_path: Path) -> dict[str, Any]:
    if set(bindings) != set(REQUIRED_BOUND_FILES):
        missing = sorted(REQUIRED_BOUND_FILES - set(bindings))
        extra = sorted(set(bindings) - REQUIRED_BOUND_FILES)
        raise ValueError(f"EXP34 lock keys differ; missing={missing}, extra={extra}")
    if repo_relative(lock_path, must_exist=False) != LOCK_PATH:
        raise ValueError("EXP34 immutable lock path is not canonical")
    bound = {key: file_binding(path) for key, path in sorted(bindings.items())}
    if bound["exp34_preregistration"]["sha256"] != PREREGISTRATION_SHA256:
        raise ValueError("EXP34 preregistration hash is not frozen")
    _verify_bound_files(bound, reread_payloads=True)
    prereg = _load_json(_normalized_repo_relative(PREREGISTRATION_PATH, must_exist=True))
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
        "resource_limits": {
            "source_wall_seconds_maximum": 180,
            "source_peak_process_tree_rss_bytes_maximum": 4294967296,
            "renderer_wall_seconds_maximum": 60,
            "renderer_peak_process_tree_rss_bytes_maximum": 4294967296,
        },
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
        raise ValueError("EXP34 lock SHA-256 mismatch")
    lock = _load_json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA or lock.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong EXP34 lock identity")
    if lock.get("status") != LOCK_STATUS:
        raise ValueError("wrong EXP34 lock status")
    if lock.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("lock preregistration binding mismatch")
    if lock.get("physical_source_execution_authorized") is not False:
        raise ValueError("immutable lock may not self-authorize physical execution")
    if lock.get("evaluator_input_authorized") is not False:
        raise ValueError("immutable lock may not authorize evaluator access")
    _verify_bound_files(lock.get("bound_files", {}), reread_payloads=True)
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
) -> Mapping[str, Any]:
    _exact_cli_path(path, DETACHED_APPROVAL_PATH, must_exist=True)
    if sha256_path(path) != digest:
        raise ValueError("detached EXP34 approval SHA-256 mismatch")
    row = _load_json(path)
    if row.get("schema") != DETACHED_APPROVAL_SCHEMA or row.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("wrong detached EXP34 approval identity")
    if row.get("status") != "AUTHORIZE_ONE_EXP34_PHYSICAL_PRIMARY_AND_CONDITIONAL_REPLAY":
        raise ValueError("detached EXP34 approval decision is not exact")
    if row.get("physical_source_execution_authorized") is not True:
        raise ValueError("detached EXP34 approval does not authorize physical execution")
    if row.get("evaluator_input_authorized") is not False:
        raise ValueError("detached EXP34 approval authorizes evaluator access")
    if row.get("authorized_orchestrator_invocations") != 1:
        raise ValueError("detached approval invocation budget is not one")
    if row.get("maximum_primary_source_children") != 1:
        raise ValueError("detached approval primary budget is not one")
    if row.get("maximum_conditional_replay_source_children") != 1:
        raise ValueError("detached approval replay budget is not one")
    if row.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("detached approval preregistration binding mismatch")
    if row.get("lock_path") != str(lock_path):
        raise ValueError("detached approval lock path mismatch")
    if row.get("lock_sha256") != lock_sha256:
        raise ValueError("detached approval lock SHA-256 mismatch")
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

    _exact_cli_path(preregistration_path, PREREGISTRATION_PATH, must_exist=True)
    if sha256_path(preregistration_path) != PREREGISTRATION_SHA256:
        raise ValueError("EXP34 preregistration SHA-256 mismatch")
    lock = verify_lock(lock_path, lock_sha256)
    _verify_detached_approval(
        detached_approval_path,
        detached_approval_sha256,
        lock_path=lock_path,
        lock_sha256=lock_sha256,
    )
    bound = lock["bound_files"]
    environment_row = _load_json(
        _normalized_repo_relative(bound["environment_attestation"]["path"], must_exist=True)
    )
    if environment_row.get("environment") != runtime_environment_fingerprint():
        raise ValueError("live EXP34 runtime differs from immutable environment attestation")
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
    if set(output_paths) == set(SOURCE_MODE_OUTPUT_KEYS["PRIMARY"]):
        mode = infer_source_mode(output_paths, lock)
        for generic_key, locked_key in SOURCE_MODE_OUTPUT_KEYS[mode].items():
            _exact_cli_path(output_paths[generic_key], authorized[locked_key], must_exist=False)
    else:
        for key, path in sorted(output_paths.items()):
            if key not in authorized:
                raise ValueError(f"output path key is not lock-authorized: {key}")
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
