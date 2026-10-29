#!/usr/bin/env python3
"""Immutable source-free qualification contract for EXP_CAD_0033.

This module has no physical-source CLI and performs no benchmark execution.
It owns strict JSON publication, immutable-lock validation, and the exact-path
checks that keep the EXP33 worker and harness from becoming confused deputies.
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
LOCK_SCHEMA = "EXP_CAD_0033_SOURCE_FREE_QUALIFICATION_LOCK_V1"
LOCK_STATUS = "PASS_SOURCE_FREE_LOCK_EXECUTION_AUTHORIZED"
EXPERIMENT_ID = "EXP_CAD_0033"
PREREGISTRATION_PATH = (
    "docs/research/"
    "2026-08-30_exp_cad_0033_cfpdg_v5_source_free_qualification_preregistration.json"
)
SOURCE_FREE_TEST_COMMAND = [
    ".venv/bin/python",
    "-m",
    "pytest",
    "-q",
    "tests/test_exp_cad_0032_cfpdg_v5_equivalence.py",
    "tests/test_exp_cad_0032_performance_harness.py",
    "tests/test_exp_cad_0033_qualification_protocol.py",
]
# Updated only while the source-free test file is still under implementation;
# the final lock binds this module, the test source, and the zero-failure row.
SOURCE_FREE_TEST_COLLECTED_COUNT = 63

REQUIRED_BOUND_FILES = frozenset(
    {
        "exp33_preregistration",
        "terminal_exp32_result",
        "terminal_exp32_report",
        "candidate_preregistration",
        "parent_preregistration",
        "cad_preflight",
        "cad_nodes",
        "cad_edges",
        "cad_edge_geodesic_mm",
        "cad_mesh",
        "frozen_science",
        "frozen_core",
        "frozen_parent_runner",
        "frozen_source_interface",
        "frozen_v4_runner",
        "exp31_preregistration",
        "candidate_engine",
        "candidate_source_free_runner",
        "candidate_worker",
        "candidate_harness",
        "candidate_engine_tests",
        "candidate_harness_tests",
        "renderer",
        "benchmark_manifest",
        "first_points",
        "first_normals",
        "second_points",
        "second_normals",
        "exp33_contract",
        "exp33_lock_builder",
        "exp33_worker",
        "exp33_harness",
        "exp33_tests",
        "source_free_test_attestation",
        "environment_attestation",
        "prelock_source_free_readiness_approval",
    }
)

WORKER_STATIC_ARGUMENT_BINDINGS = {
    "source": {
        "benchmark_manifest": "benchmark_manifest",
        "v5_preregistration": "candidate_preregistration",
        "parent_preregistration": "parent_preregistration",
        "cad_preflight": "cad_preflight",
        "source_free_runner": "candidate_source_free_runner",
    },
    "reference": {
        "benchmark_manifest": "benchmark_manifest",
        "parent_preregistration": "parent_preregistration",
        "cad_preflight": "cad_preflight",
        "frozen_parent_runner": "frozen_parent_runner",
    },
    "render": {"renderer": "renderer"},
}

HARNESS_STATIC_ARGUMENT_BINDINGS = {
    "cad_nodes": "cad_nodes",
    "cad_edges": "cad_edges",
    "edge_geodesic_mm": "cad_edge_geodesic_mm",
    "cad_mesh": "cad_mesh",
    "v5_preregistration": "candidate_preregistration",
    "parent_preregistration": "parent_preregistration",
    "cad_preflight": "cad_preflight",
    "source_free_runner": "candidate_source_free_runner",
    "frozen_parent_runner": "frozen_parent_runner",
    "renderer": "renderer",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def null_canonicalize(value: Any) -> Any:
    """Match frozen-parent publication: every non-finite scalar becomes null."""

    if isinstance(value, Mapping):
        return {str(key): null_canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [null_canonicalize(item) for item in value]
    value_module = type(value).__module__.split(".", 1)[0]
    if value_module == "numpy" and hasattr(value, "tolist"):
        return null_canonicalize(value.tolist())
    if value_module == "numpy" and hasattr(value, "item"):
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
            for text in encoder.iterencode(null_canonicalize(value)):
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
        raise ValueError("lock path must be a nonempty POSIX repo-relative string")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"lock path is not normalized repo-relative: {value}")
    lexical = ROOT / relative
    resolved = lexical.resolve(strict=must_exist)
    root = ROOT.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"lock path escapes repository: {value}")
    # Reject aliases through a symlink in any existing component.
    probe = root
    for part in relative.parts:
        probe = probe / part
        if probe.exists() and probe.is_symlink():
            raise ValueError(f"lock path traverses symlink: {value}")
    return resolved


def repo_relative(path: Path, *, must_exist: bool) -> str:
    resolved = Path(path).resolve(strict=must_exist)
    relative = resolved.relative_to(ROOT.resolve(strict=True)).as_posix()
    if _normalized_repo_relative(relative, must_exist=must_exist) != resolved:
        raise ValueError("path does not have one canonical repository spelling")
    return relative


def file_binding(path: Path) -> dict[str, Any]:
    path = Path(path)
    relative = repo_relative(path, must_exist=True)
    resolved = _normalized_repo_relative(relative, must_exist=True)
    if not resolved.is_file():
        raise ValueError(f"bound path is not a regular file: {relative}")
    return {
        "path": relative,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_path(resolved),
    }


def build_lock_payload(
    *,
    bindings: Mapping[str, Path],
    envelope_root: Path,
    result_root: Path,
    lock_path: Path,
) -> dict[str, Any]:
    if set(bindings) != set(REQUIRED_BOUND_FILES):
        missing = sorted(REQUIRED_BOUND_FILES - set(bindings))
        extra = sorted(set(bindings) - REQUIRED_BOUND_FILES)
        raise ValueError(f"EXP33 lock binding keys differ; missing={missing}, extra={extra}")
    bound = {key: file_binding(path) for key, path in sorted(bindings.items())}
    if bound["exp33_preregistration"]["path"] != PREREGISTRATION_PATH:
        raise ValueError("EXP33 preregistration path is not canonical")
    _enforce_preregistered_bindings(bound)
    _enforce_attestations(bound)
    envelope_relative = repo_relative(envelope_root, must_exist=True)
    result_relative = repo_relative(result_root, must_exist=False)
    lock_relative = repo_relative(lock_path, must_exist=False)
    expected_manifest = (Path(envelope_relative) / "BENCHMARK_MANIFEST.json").as_posix()
    if bound["benchmark_manifest"]["path"] != expected_manifest:
        raise ValueError("benchmark manifest is not the locked envelope manifest")
    return {
        "schema": LOCK_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": LOCK_STATUS,
        "qualification_invocation_budget": 1,
        "physical_source_execution_authorized": False,
        "evaluator_input_authorized": False,
        "authorized_paths": {
            "envelope_root": envelope_relative,
            "result_root": result_relative,
            "lock_path": lock_relative,
        },
        "bound_files": bound,
        "worker_static_argument_bindings": WORKER_STATIC_ARGUMENT_BINDINGS,
        "harness_static_argument_bindings": HARNESS_STATIC_ARGUMENT_BINDINGS,
        "candidate_payload_identity_must_remain_nested": True,
    }


def _enforce_preregistered_bindings(bound: Mapping[str, Mapping[str, Any]]) -> None:
    prereg = json.loads(
        _normalized_repo_relative(PREREGISTRATION_PATH, must_exist=True).read_text(
            encoding="utf-8"
        )
    )
    candidate_keys = {
        "v5_engine": "candidate_engine",
        "source_free_runner": "candidate_source_free_runner",
        "candidate_worker": "candidate_worker",
        "candidate_harness": "candidate_harness",
        "candidate_engine_tests": "candidate_engine_tests",
        "candidate_harness_tests": "candidate_harness_tests",
    }
    frozen_candidates = prereg.get("candidate_dependencies_at_freeze", {})
    for prereg_key, lock_key in candidate_keys.items():
        frozen = frozen_candidates.get(prereg_key)
        if not isinstance(frozen, Mapping) or (
            bound[lock_key]["path"] != frozen.get("path")
            or bound[lock_key]["sha256"] != frozen.get("sha256")
        ):
            raise ValueError(f"candidate binding differs from preregistration: {lock_key}")
    terminal = prereg.get("terminal_parent", {})
    for lock_key, path_key, hash_key in (
        ("terminal_exp32_result", "result_path", "result_sha256"),
        ("terminal_exp32_report", "report_path", "report_sha256"),
    ):
        if (
            bound[lock_key]["path"] != terminal.get(path_key)
            or bound[lock_key]["sha256"] != terminal.get(hash_key)
        ):
            raise ValueError(f"terminal binding differs from preregistration: {lock_key}")


def _enforce_attestations(bound: Mapping[str, Mapping[str, Any]]) -> None:
    requirements = {
        "source_free_test_attestation": (
            "EXP_CAD_0033_SOURCE_FREE_TEST_ATTESTATION_V1",
            "PASS_ZERO_FAILURES",
        ),
        "environment_attestation": (
            "EXP_CAD_0033_SOURCE_FREE_ENVIRONMENT_ATTESTATION_V1",
            "FROZEN",
        ),
        "prelock_source_free_readiness_approval": (
            "EXP_CAD_0033_PRELOCK_SOURCE_FREE_READINESS_APPROVAL_V1",
            "APPROVED_TO_MATERIALIZE_IMMUTABLE_LOCK",
        ),
    }
    for key, (schema, status) in requirements.items():
        path = _normalized_repo_relative(bound[key]["path"], must_exist=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != schema
            or payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("status") != status
            or payload.get("qualification_executed") is not False
            or payload.get("physical_source_execution_authorized") is not False
        ):
            raise ValueError(f"EXP33 attestation is not execution-authorizing: {key}")
    environment = json.loads(
        _normalized_repo_relative(
            bound["environment_attestation"]["path"], must_exist=True
        ).read_text(encoding="utf-8")
    )
    required_environment = {
        "python_executable_realpath",
        "python_executable_invocation_path",
        "python_executable_sha256",
        "python_version",
        "numpy_version",
        "scipy_version",
        "open3d_version",
        "numpy_module_origin",
        "scipy_module_origin",
        "open3d_module_origin",
    }
    if not required_environment <= set(environment.get("environment", {})):
        raise ValueError("EXP33 environment attestation is incomplete")
    tests = json.loads(
        _normalized_repo_relative(
            bound["source_free_test_attestation"]["path"], must_exist=True
        ).read_text(encoding="utf-8")
    )
    expected_test_hashes = {
        key: bound[key]["sha256"]
        for key in (
            "candidate_engine_tests",
            "candidate_harness_tests",
            "exp33_tests",
        )
    }
    code_under_test_keys = (
        "exp33_preregistration",
        "candidate_preregistration",
        "parent_preregistration",
        "exp31_preregistration",
        "frozen_core",
        "frozen_science",
        "frozen_parent_runner",
        "frozen_source_interface",
        "frozen_v4_runner",
        "candidate_engine",
        "candidate_source_free_runner",
        "candidate_worker",
        "candidate_harness",
        "exp33_contract",
        "exp33_lock_builder",
        "exp33_worker",
        "exp33_harness",
    )
    expected_code_hashes = {key: bound[key]["sha256"] for key in code_under_test_keys}
    if (
        tests.get("returncode") != 0
        or tests.get("test_command") != SOURCE_FREE_TEST_COMMAND
        or tests.get("collected_count") != SOURCE_FREE_TEST_COLLECTED_COUNT
        or tests.get("passed_count") != SOURCE_FREE_TEST_COLLECTED_COUNT
        or tests.get("failed_count") != 0
        or tests.get("error_count") != 0
        or tests.get("skipped_count") != 0
        or tests.get("tested_file_sha256") != expected_test_hashes
        or tests.get("code_under_test_sha256") != expected_code_hashes
    ):
        raise ValueError("EXP33 source-free test attestation is not a zero-failure run")
    approval = json.loads(
        _normalized_repo_relative(
            bound["prelock_source_free_readiness_approval"]["path"], must_exist=True
        ).read_text(encoding="utf-8")
    )
    expected_sha_map = {
        key: row["sha256"]
        for key, row in sorted(bound.items())
        if key != "prelock_source_free_readiness_approval"
    }
    if approval.get("approved_bound_file_sha256") != expected_sha_map:
        raise ValueError("prelock approval does not name every exact dependency SHA")


def verify_current_environment(lock: Mapping[str, Any]) -> Mapping[str, str]:
    path = _normalized_repo_relative(
        lock["bound_files"]["environment_attestation"]["path"], must_exist=True
    )
    attested = json.loads(path.read_text(encoding="utf-8"))["environment"]

    def origin(module_name: str) -> str:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ValueError(f"cannot resolve module origin: {module_name}")
        return str(Path(spec.origin).resolve(strict=True))

    invocation = Path(sys.executable)
    if not invocation.is_absolute():
        raise ValueError("Python invocation path must be absolute")
    executable = invocation.resolve(strict=True)
    observed = {
        "python_executable_invocation_path": str(invocation),
        "python_executable_realpath": str(executable),
        "python_executable_sha256": sha256_path(executable),
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "open3d_version": importlib.metadata.version("open3d"),
        "numpy_module_origin": origin("numpy"),
        "scipy_module_origin": origin("scipy"),
        "open3d_module_origin": origin("open3d"),
    }
    if attested != observed:
        raise ValueError("current interpreter/scientific environment differs from lock")
    return observed


def _validate_authorized_paths_metadata(
    paths: Mapping[str, Any], *, require_result_root_exists: bool
) -> None:
    relative_paths: dict[str, Path] = {}
    for key in ("envelope_root", "result_root", "lock_path"):
        value = paths.get(key)
        relative = Path(value) if isinstance(value, str) else Path()
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError(f"EXP33 authorized path is not canonical repo-relative: {key}")
        relative_paths[key] = relative
    envelope = relative_paths["envelope_root"]
    result = relative_paths["result_root"]
    if envelope == result or envelope in result.parents or result in envelope.parents:
        raise ValueError("EXP33 envelope and result roots must be disjoint")
    if require_result_root_exists:
        result_absolute = _normalized_repo_relative(
            paths["result_root"], must_exist=True
        )
        if not result_absolute.is_dir():
            raise ValueError("EXP33 worker result root is not an existing directory")


def verify_lock_bootstrap(
    path: Path,
    expected_sha256: str,
    *,
    require_result_root_exists: bool = False,
) -> Mapping[str, Any]:
    """Validate lock identity and metadata without opening any bound input."""

    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError("EXP33 qualification lock path must be canonical absolute")
    resolved_path = supplied.resolve(strict=True)
    if str(supplied) != str(resolved_path):
        raise ValueError("EXP33 qualification lock path uses an alias")
    if sha256_path(resolved_path) != expected_sha256:
        raise ValueError("EXP33 qualification lock packet hash mismatch")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("EXP33 qualification lock root must be an object")
    if (
        payload.get("schema") != LOCK_SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != LOCK_STATUS
        or payload.get("qualification_invocation_budget") != 1
        or payload.get("physical_source_execution_authorized") is not False
        or payload.get("evaluator_input_authorized") is not False
        or payload.get("candidate_payload_identity_must_remain_nested") is not True
    ):
        raise ValueError("EXP33 qualification lock identity/status mismatch")
    if payload.get("worker_static_argument_bindings") != WORKER_STATIC_ARGUMENT_BINDINGS:
        raise ValueError("EXP33 worker static binding map mismatch")
    if payload.get("harness_static_argument_bindings") != HARNESS_STATIC_ARGUMENT_BINDINGS:
        raise ValueError("EXP33 harness static binding map mismatch")
    paths = payload.get("authorized_paths")
    if not isinstance(paths, Mapping) or set(paths) != {
        "envelope_root", "result_root", "lock_path"
    }:
        raise ValueError("EXP33 authorized path roots malformed")
    _validate_authorized_paths_metadata(
        paths, require_result_root_exists=require_result_root_exists
    )
    # Check the lock's own canonical location before trusting any file row.
    locked_path = _normalized_repo_relative(paths["lock_path"], must_exist=True)
    if str(supplied) != str(locked_path):
        raise ValueError("EXP33 qualification lock path differs from locked canonical path")
    files = payload.get("bound_files")
    if not isinstance(files, Mapping) or set(files) != set(REQUIRED_BOUND_FILES):
        raise ValueError("EXP33 qualification lock bound-file key mismatch")
    for key, row in files.items():
        if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"EXP33 malformed binding: {key}")
        value = row["path"]
        relative = Path(value) if isinstance(value, str) else Path()
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError(f"EXP33 malformed binding metadata: {key}")
    return payload


def verify_lock(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    payload = verify_lock_bootstrap(path, expected_sha256)
    files = payload["bound_files"]
    paths = payload["authorized_paths"]
    for key, row in files.items():
        bound = _normalized_repo_relative(row["path"], must_exist=True)
        if not bound.is_file():
            raise ValueError(f"EXP33 binding is not a regular file: {key}")
        if int(bound.stat().st_size) != row["size_bytes"] or sha256_path(bound) != row["sha256"]:
            raise ValueError(f"EXP33 qualification lock file mismatch: {key}")
    _enforce_preregistered_bindings(files)
    _enforce_attestations(files)
    envelope = _normalized_repo_relative(paths["envelope_root"], must_exist=True)
    result = _normalized_repo_relative(paths["result_root"], must_exist=False)
    if files["benchmark_manifest"]["path"] != (
        Path(paths["envelope_root"]) / "BENCHMARK_MANIFEST.json"
    ).as_posix():
        raise ValueError("EXP33 locked manifest/envelope relationship mismatch")
    if envelope == result or envelope in result.parents or result in envelope.parents:
        raise ValueError("EXP33 envelope and result roots must be disjoint")
    _verify_locked_envelope_files(payload)
    return payload


def _verify_locked_envelope_files(lock: Mapping[str, Any]) -> None:
    files = lock["bound_files"]
    manifest_path = _normalized_repo_relative(
        files["benchmark_manifest"]["path"], must_exist=True
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise ValueError("locked benchmark manifest lacks file bindings")
    for manifest_key, lock_key in (
        ("cad_nodes", "cad_nodes"),
        ("cad_edges", "cad_edges"),
        ("cad_edge_geodesic_mm", "cad_edge_geodesic_mm"),
        ("cad_mesh", "cad_mesh"),
        ("first_points", "first_points"),
        ("first_normals", "first_normals"),
        ("second_points", "second_points"),
        ("second_normals", "second_normals"),
    ):
        row = manifest_files.get(manifest_key)
        locked = files[lock_key]
        if not isinstance(row, Mapping):
            raise ValueError(f"locked manifest lacks {manifest_key}")
        actual = (manifest_path.parent / row.get("relative_path", "")).resolve(strict=True)
        expected = _normalized_repo_relative(locked["path"], must_exist=True)
        if (
            actual != expected
            or row.get("sha256") != locked["sha256"]
            or sha256_path(actual) != locked["sha256"]
        ):
            raise ValueError(f"locked source-free envelope file mismatch: {manifest_key}")


def _exact_static_argument(
    supplied: Path, binding_key: str, lock: Mapping[str, Any]
) -> None:
    row = lock["bound_files"][binding_key]
    expected = _normalized_repo_relative(row["path"], must_exist=True)
    raw = Path(supplied)
    if not raw.is_absolute() or str(raw) != str(expected):
        raise ValueError(f"static CLI path mismatch for binding {binding_key}")
    actual = raw.resolve(strict=True)
    if actual != expected:
        raise ValueError(f"static CLI path alias for binding {binding_key}")
    if sha256_path(actual) != row["sha256"]:
        raise ValueError(f"static CLI hash mismatch for binding {binding_key}")


def _beneath(path: Path, root: Path, *, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"{label} must use a canonical absolute path")
    resolved = raw.resolve(strict=False)
    if str(raw) != str(resolved):
        raise ValueError(f"{label} must not use a lexical or symlink alias")
    locked_root = Path(root).resolve(strict=False)
    if resolved == locked_root or locked_root not in resolved.parents:
        raise ValueError(f"{label} must resolve strictly beneath locked result root")
    return resolved


def verify_self_binding(path: Path, binding_key: str, lock: Mapping[str, Any]) -> None:
    _exact_static_argument(Path(path), binding_key, lock)


def verify_detached_postlock_approval(
    path: Path,
    expected_sha256: str,
    *,
    qualification_lock: Path,
    qualification_lock_sha256: str,
) -> Mapping[str, Any]:
    raw = Path(path)
    if not raw.is_absolute() or str(raw) != str(raw.resolve(strict=True)):
        raise ValueError("detached approval must use its canonical absolute path")
    if sha256_path(raw) != expected_sha256:
        raise ValueError("detached approval hash mismatch")
    payload = json.loads(raw.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "EXP_CAD_0033_DETACHED_POSTLOCK_APPROVAL_V1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "AUTHORIZE_ONE_SOURCE_FREE_QUALIFICATION"
        or payload.get("approval_path") != str(raw)
        or payload.get("qualification_lock_path") != str(qualification_lock)
        or payload.get("qualification_lock_sha256") != qualification_lock_sha256
        or payload.get("authorized_attempts") != 1
        or payload.get("qualification_executed") is not False
        or payload.get("physical_source_execution_authorized") is not False
        or payload.get("evaluator_input_authorized") is not False
    ):
        raise ValueError("detached post-lock approval identity/authority mismatch")
    return payload


def _relative_to_locked_result(path: Path, result_root: Path, *, label: str) -> str:
    resolved = _beneath(path, result_root, label=label)
    return resolved.relative_to(result_root).as_posix()


def _verify_worker_dynamic_schedule(mode: str, args: Any, result_root: Path) -> None:
    output_relative = _relative_to_locked_result(
        args.output_root, result_root, label="worker output root"
    )
    parts = Path(output_relative).parts
    full_source = (
        len(parts) == 3
        and parts[0] == "full"
        and parts[1] in {"run_1", "run_2", "run_3"}
        and parts[2] == "source"
    )
    full_render = (
        len(parts) == 3
        and parts[0] == "full"
        and parts[1] in {"run_1", "run_2", "run_3"}
        and parts[2] == "renderer"
    )
    half_source = parts == ("half", "run_1", "source")
    differential = (
        len(parts) == 4
        and parts[0] == "reference_differential"
        and parts[1] in {"frozen_parent", "v5"}
        and parts[2] in {"run_1", "run_2", "run_3"}
        and parts[3] == "output"
    )
    allowed = {
        "source": full_source or half_source or (differential and parts[1] == "v5"),
        "reference": differential and parts[1] == "frozen_parent",
        "render": full_render,
    }[mode]
    if not allowed:
        raise ValueError("worker output path is outside the exact EXP33 schedule")
    if mode == "source":
        expected = (
            result_root / parts[0] / parts[1] / "source_measurement" / "ENGINE_MONITORING_METRICS.json"
            if parts[0] in {"full", "half"}
            else result_root / parts[0] / parts[1] / parts[2] / "measurement" / "ENGINE_MONITORING_METRICS.json"
        )
        if str(Path(args.monitoring_output)) != str(expected):
            raise ValueError("source monitoring path differs from exact EXP33 schedule")
    if mode == "render":
        expected_visual = result_root / "full" / parts[1] / "source" / "VISUAL_PAYLOAD.json"
        if str(Path(args.visual_payload)) != str(expected_visual):
            raise ValueError("renderer visual path differs from exact EXP33 schedule")


def verify_worker_cli(mode: str, args: Any, lock: Mapping[str, Any]) -> None:
    mapping = WORKER_STATIC_ARGUMENT_BINDINGS.get(mode)
    if mapping is None:
        raise ValueError(f"unexpected EXP33 worker mode: {mode}")
    for argument, binding_key in mapping.items():
        _exact_static_argument(getattr(args, argument), binding_key, lock)
    result_root = _normalized_repo_relative(
        lock["authorized_paths"]["result_root"], must_exist=False
    )
    _verify_worker_dynamic_schedule(mode, args, result_root)
    if mode == "source":
        _beneath(args.monitoring_output, result_root, label="monitoring output")
    if mode == "render":
        _beneath(args.visual_payload, result_root, label="visual payload")


def verify_harness_cli(args: Any, lock: Mapping[str, Any]) -> None:
    for argument, binding_key in HARNESS_STATIC_ARGUMENT_BINDINGS.items():
        _exact_static_argument(getattr(args, argument), binding_key, lock)
    expected_envelope = _normalized_repo_relative(
        lock["authorized_paths"]["envelope_root"], must_exist=True
    )
    expected_result = _normalized_repo_relative(
        lock["authorized_paths"]["result_root"], must_exist=False
    )
    if Path(args.envelope_root).resolve(strict=True) != expected_envelope:
        raise ValueError("EXP33 envelope root differs from immutable lock")
    if (
        not Path(args.envelope_root).is_absolute()
        or str(Path(args.envelope_root)) != str(expected_envelope)
    ):
        raise ValueError("EXP33 envelope root must use its canonical absolute spelling")
    if (
        not Path(args.result_root).is_absolute()
        or str(Path(args.result_root)) != str(expected_result)
        or Path(args.result_root).resolve(strict=False) != expected_result
    ):
        raise ValueError("EXP33 result root differs from immutable lock")
    manifest = expected_envelope / "BENCHMARK_MANIFEST.json"
    _exact_static_argument(manifest, "benchmark_manifest", lock)
