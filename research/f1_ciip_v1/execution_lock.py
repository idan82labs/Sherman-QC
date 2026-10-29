"""Immutable pre-execution lock for the one user-authorized CIIP-V1 smoke.

This module performs no coordinate parsing or geometry.  It binds the frozen
paper packet, content-selected raw source, exact code, dependency/toolchain
environment, deterministic permutation, and resource/output contract before a
worker may receive an already-open source descriptor.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping

from .config import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PROOF_SHA256,
    EXPECTED_SCHEMA_SHA256,
    FrozenV7,
    canonical_json_bytes,
    select_smoke_source_by_content,
)


LOCK_SCHEMA_VERSION = 1
ATTEMPT_ID = "CORRECTION_V4"
AUTHORIZATION_RECEIPT = "USER_FRESH_INDEPENDENT_SOURCE_SCHEMA_CORRECTED_GOLD_FREE_SMOKE_CORRECTION_V4_AUTHORIZATION_20260809"
AUTHORIZATION_SCOPE = (
    "EXECUTE_EXACTLY_ONE_FRESH_INDEPENDENT_SOURCE_SCHEMA_CORRECTED_CONTENT_SELECTED_GOLD_FREE_SMOKE;"
    "PRESERVE_PRIOR_V1_AND_CORRECTION_V2_AND_CORRECTION_V3_CONSUMED_CLAIMS_LOCKS_AND_TERMINAL_STOPS;"
    "NO_LABELS_GOLD_CAD_EXPECTED_COUNTS_TUNING_BENCHMARK_RUNTIME_MAIN_PRODUCTION_"
    "REPLAY_SYNTHETIC_RESCUE;NO_RERUN_OR_REPLACEMENT_AFTER_CORRECTION_V4_FAILURE"
)
PERMUTATION_RULE = "REVERSE_DECLARED_VERTEX_RECORD_SEQUENCE_BEFORE_CANONICALIZATION_V1"
COMPILE_FLAGS = (
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-frounding-math",
    "-fno-associative-math",
    "-fno-unsafe-math-optimizations",
    "-fno-finite-math-only",
    "-Werror",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
)
ROLE_ORDER = ("BASE_A", "BASE_B", "INPUT_ORDER_PERMUTED", "INDEPENDENT_VALIDATOR")
LOCK_RELATIVE = Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_PREEXECUTION_LOCK_CORRECTION_V4_20260809.json")
CLAIM_RELATIVE = Path("docs/research/CIIP_V1_GOLD_FREE_SMOKE_EXECUTION_CLAIM_CORRECTION_V4_20260809.json")
NATIVE_BUILD_RELATIVE = Path("output/research/ciip-digital-occupied-shell-h0-v1/native-lock-correction-v4")
OUTPUT_RELATIVE = Path("output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v4")
SCRATCH_RELATIVE = Path("tmp/ciip-digital-occupied-shell-h0-v1-smoke-correction-v4")
CODE_LOCK_TAG = b"CIIP_CODE_LOCK_V1\0"
DEPENDENCY_LOCK_TAG = b"CIIP_DEPENDENCY_LOCK_V1\0"
EXECUTION_LOCK_TAG = b"CIIP_EXECUTION_LOCK_V1\0"


class ExecutionLockError(RuntimeError):
    """Raised when the pre-execution contract cannot be locked exactly."""


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_open_regular_source(path: Path) -> tuple[str, int]:
    """Hash a regular non-symlink source through an O_NOFOLLOW descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ExecutionLockError("selected source cannot be opened fail-closed") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ExecutionLockError("selected source is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        if size != info.st_size:
            raise ExecutionLockError("source size changed while hashing")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _run_exact(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="strict",
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise ExecutionLockError(f"toolchain probe failed: {argv[0]}") from exc
    return completed.stdout.rstrip("\n")


def dependency_record(clang: Path) -> dict[str, Any]:
    clang = clang.resolve(strict=True)
    clang_sha, clang_size = sha256_file(clang)
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExecutionLockError(f"required dependency is absent: {name}") from exc
    ps_path = Path("/bin/ps").resolve(strict=True)
    ps_sha, ps_size = sha256_file(ps_path)
    return {
        "clang_path_sha256": hashlib.sha256(os.fsencode(str(clang))).hexdigest(),
        "clang_binary_sha256": clang_sha,
        "clang_binary_bytes": clang_size,
        "clang_version_stdout": _run_exact([str(clang), "--version"]),
        "compile_flags": list(COMPILE_FLAGS),
        "executable_format": "MACH_O_ARM64",
        "fenv_rounding": "FE_TONEAREST_REQUIRED_AT_WORKER_START",
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "packages": packages,
        "platform": platform.platform(),
        "ps_binary_bytes": ps_size,
        "ps_binary_sha256": ps_sha,
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve(strict=True))[0],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "roles_sequential": True,
        "thread_environment": {
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }


def _relative_files(root: Path, paths: Iterable[Path]) -> tuple[Path, ...]:
    normalized: set[Path] = set()
    for path in paths:
        if path.is_symlink():
            raise ExecutionLockError("code-lock input must not be a symlink")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise ExecutionLockError("code-lock source escaped repository root") from exc
        if not resolved.is_file():
            raise ExecutionLockError("code-lock input must be a regular non-symlink file")
        normalized.add(relative)
    if not normalized:
        raise ExecutionLockError("code-lock source set is empty")
    return tuple(sorted(normalized, key=lambda item: item.as_posix().encode("utf-8")))


def code_record(root: Path, paths: Iterable[Path]) -> tuple[dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256(CODE_LOCK_TAG)
    for relative in _relative_files(root, paths):
        raw = (root / relative).read_bytes()
        encoded = relative.as_posix().encode("utf-8")
        file_sha = hashlib.sha256(raw).digest()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(file_sha)
        rows.append({"bytes": len(raw), "path": relative.as_posix(), "sha256": file_sha.hex()})
    return {"files": rows, "tag": CODE_LOCK_TAG[:-1].decode("ascii")}, digest.hexdigest()


def tagged_json_digest(tag: bytes, value: Mapping[str, Any]) -> str:
    return hashlib.sha256(tag + canonical_json_bytes(value)).hexdigest()


def prepare_execution_lock(
    *,
    root: Path,
    frozen: FrozenV7,
    clang: Path,
    code_paths: Iterable[Path],
    native_binaries: Mapping[str, Path],
    native_binary_manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a complete immutable lock object; this function does not write it."""

    root = root.resolve(strict=True)
    output_path = root / OUTPUT_RELATIVE
    scratch_path = root / SCRATCH_RELATIVE
    if output_path.exists() or scratch_path.exists():
        raise ExecutionLockError("reserved smoke output or scratch destination already exists")

    selected = select_smoke_source_by_content(frozen)
    raw_path = Path(str(selected["raw_path"]))
    raw_sha, raw_size = hash_open_regular_source(raw_path)
    if raw_sha != selected["raw_ply_sha256"] or raw_size != selected["raw_byte_count"]:
        raise ExecutionLockError("content-selected raw source disagrees with frozen manifest")

    dependency = dependency_record(clang)
    dependency_sha = tagged_json_digest(DEPENDENCY_LOCK_TAG, dependency)
    code, code_sha = code_record(root, code_paths)

    supplied_manifests = dict(native_binary_manifests or {})
    if set(supplied_manifests) - set(native_binaries):
        raise ExecutionLockError("native manifest supplied without matching binary")
    binaries: dict[str, Any] = {}
    for role, path in sorted(native_binaries.items()):
        if role not in ROLE_ORDER:
            raise ExecutionLockError(f"unknown native role binding: {role}")
        resolved_binary = path.resolve(strict=True)
        try:
            relative_binary = resolved_binary.relative_to(root)
        except ValueError as exc:
            raise ExecutionLockError("native binary escaped repository root") from exc
        if relative_binary.parent != NATIVE_BUILD_RELATIVE:
            raise ExecutionLockError("native binary escaped correction-v4 native namespace")
        binary_sha, binary_size = sha256_file(resolved_binary)
        binary_row: dict[str, Any] = {
            "bytes": binary_size,
            "path": relative_binary.as_posix(),
            "sha256": binary_sha,
        }
        if role in supplied_manifests:
            binary_row["build_manifest"] = dict(supplied_manifests[role])
        binaries[role] = binary_row

    body: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "authorization": {"receipt": AUTHORIZATION_RECEIPT, "scope": AUTHORIZATION_SCOPE},
        "frozen": {
            "configuration_sha256": EXPECTED_CONFIG_SHA256,
            "resource_proof_sha256": EXPECTED_PROOF_SHA256,
            "schema_sha256": EXPECTED_SCHEMA_SHA256,
        },
        "source": {
            "raw_byte_count": raw_size,
            "raw_ply_sha256": raw_sha,
            "ply_header_byte_count": selected["ply_header_byte_count"],
            "ply_header_sha256": selected["ply_header_sha256"],
            "ply_format": selected["ply_format"],
            "parsed_vertex_count": selected["parsed_vertex_count"],
            "finite_coordinate_count": selected["finite_coordinate_count"],
            "exact_duplicate_count": selected["exact_duplicate_count"],
            "ignored_property_name_count": selected["ignored_property_name_count"],
            "ignored_property_name_digest": selected["ignored_property_name_digest"],
            "canonical_coordinate_multiset_digest": selected["canonical_coordinate_multiset_digest"],
            "selection_rule": "UNIQUE_LEXICOGRAPHIC_MINIMUM_RAW_PLY_SHA256_OVER_EXACT_37_ROWS",
        },
        "execution": {
            "input_order_permutation": PERMUTATION_RULE,
            "output_relative": OUTPUT_RELATIVE.as_posix(),
            "scratch_relative": SCRATCH_RELATIVE.as_posix(),
            "role_order": list(ROLE_ORDER),
            "worker_count": 1,
            "wall_seconds_per_role": 300,
            "cpu_seconds_per_role": 300,
            "rss_bytes": 12 * 1024**3,
            "scratch_bytes": 256 * 1024**2,
            "serialized_bytes_per_role": 64 * 1024**2,
            "watchdog_poll_milliseconds": 10,
            "failure_policy": "TERMINAL_NO_RERUN_NO_REPLACEMENT_NO_PARAMETER_CHANGE",
        },
        "dependency": dependency,
        "dependency_lock_sha256": dependency_sha,
        "code": code,
        "code_sha256": code_sha,
        "native_binaries": binaries,
    }
    body["execution_lock_sha256"] = tagged_json_digest(EXECUTION_LOCK_TAG, body)
    return body


def validate_execution_lock(lock: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version", "authorization", "frozen", "source", "execution", "dependency",
        "dependency_lock_sha256", "code", "code_sha256", "native_binaries",
        "execution_lock_sha256",
    }
    if set(lock) != expected_top:
        raise ExecutionLockError("execution-lock top-level shape mismatch")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ExecutionLockError("execution-lock schema mismatch")
    body = dict(lock)
    asserted = body.pop("execution_lock_sha256", None)
    if asserted != tagged_json_digest(EXECUTION_LOCK_TAG, body):
        raise ExecutionLockError("execution-lock self-hash mismatch")
    if body.get("authorization") != {"receipt": AUTHORIZATION_RECEIPT, "scope": AUTHORIZATION_SCOPE}:
        raise ExecutionLockError("execution authorization receipt mismatch")
    if body.get("frozen") != {
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "resource_proof_sha256": EXPECTED_PROOF_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
    }:
        raise ExecutionLockError("frozen packet binding mismatch")
    source = body.get("source")
    expected_source_keys = {
        "raw_byte_count", "raw_ply_sha256", "ply_header_byte_count", "ply_header_sha256",
        "ply_format", "parsed_vertex_count", "finite_coordinate_count", "exact_duplicate_count",
        "ignored_property_name_count", "ignored_property_name_digest",
        "canonical_coordinate_multiset_digest", "selection_rule",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ExecutionLockError("source-lock shape mismatch")
    for key in ("raw_ply_sha256", "ply_header_sha256", "ignored_property_name_digest", "canonical_coordinate_multiset_digest"):
        value = source[key]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ExecutionLockError(f"source-lock digest mismatch: {key}")
    for key in ("raw_byte_count", "ply_header_byte_count", "parsed_vertex_count", "finite_coordinate_count", "exact_duplicate_count", "ignored_property_name_count"):
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExecutionLockError(f"source-lock integer mismatch: {key}")
    if not (
        0 < source["ply_header_byte_count"] <= source["raw_byte_count"] <= 64_382_075
        and 0 < source["parsed_vertex_count"] <= 1_140_033
        and source["finite_coordinate_count"] == source["parsed_vertex_count"]
        and source["exact_duplicate_count"] <= source["finite_coordinate_count"]
        and source["ply_format"] in {"ascii", "binary_little_endian"}
    ):
        raise ExecutionLockError("source-lock census/bound mismatch")
    if source["selection_rule"] != "UNIQUE_LEXICOGRAPHIC_MINIMUM_RAW_PLY_SHA256_OVER_EXACT_37_ROWS":
        raise ExecutionLockError("source selection rule mismatch")
    execution = body.get("execution")
    expected_execution = {
        "input_order_permutation": PERMUTATION_RULE,
        "output_relative": OUTPUT_RELATIVE.as_posix(),
        "scratch_relative": SCRATCH_RELATIVE.as_posix(),
        "role_order": list(ROLE_ORDER),
        "worker_count": 1,
        "wall_seconds_per_role": 300,
        "cpu_seconds_per_role": 300,
        "rss_bytes": 12 * 1024**3,
        "scratch_bytes": 256 * 1024**2,
        "serialized_bytes_per_role": 64 * 1024**2,
        "watchdog_poll_milliseconds": 10,
        "failure_policy": "TERMINAL_NO_RERUN_NO_REPLACEMENT_NO_PARAMETER_CHANGE",
    }
    if execution != expected_execution:
        raise ExecutionLockError("execution contract mismatch")
    if body.get("dependency_lock_sha256") != tagged_json_digest(
        DEPENDENCY_LOCK_TAG, body.get("dependency", {})
    ):
        raise ExecutionLockError("dependency-lock digest mismatch")
    dependency = body.get("dependency")
    expected_dependency_keys = {
        "clang_path_sha256", "clang_binary_sha256", "clang_binary_bytes",
        "clang_version_stdout", "compile_flags", "executable_format", "fenv_rounding",
        "machine", "macos_version", "packages", "platform", "ps_binary_bytes",
        "ps_binary_sha256", "python_executable_sha256",
        "python_implementation", "python_version", "roles_sequential", "thread_environment",
    }
    if not isinstance(dependency, dict) or set(dependency) != expected_dependency_keys:
        raise ExecutionLockError("dependency-lock shape mismatch")
    if dependency["compile_flags"] != list(COMPILE_FLAGS) or dependency["roles_sequential"] is not True:
        raise ExecutionLockError("dependency compile/thread contract mismatch")
    expected_threads = {
        "BLIS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    }
    if dependency["thread_environment"] != expected_threads:
        raise ExecutionLockError("dependency thread environment mismatch")
    for key in (
        "clang_path_sha256", "clang_binary_sha256", "ps_binary_sha256",
        "python_executable_sha256",
    ):
        value = dependency[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ExecutionLockError(f"dependency digest invalid: {key}")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ExecutionLockError(f"dependency digest invalid: {key}") from exc
    code = body.get("code")
    if not isinstance(code, dict) or set(code) != {"files", "tag"} or code["tag"] != "CIIP_CODE_LOCK_V1":
        raise ExecutionLockError("code-lock shape mismatch")
    rows = code.get("files")
    if not isinstance(rows, list) or not rows:
        raise ExecutionLockError("code-lock file set is empty")
    digest = hashlib.sha256(CODE_LOCK_TAG)
    previous: bytes | None = None
    source_hash: str | None = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256"}:
            raise ExecutionLockError("code-lock row shape mismatch")
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.as_posix() != row["path"]:
            raise ExecutionLockError("code-lock path is not safe canonical relative")
        encoded = row["path"].encode("utf-8")
        if previous is not None and encoded <= previous:
            raise ExecutionLockError("code-lock rows are not strict canonical order")
        previous = encoded
        size, file_sha = row["bytes"], row["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ExecutionLockError("code-lock byte count invalid")
        if not isinstance(file_sha, str) or len(file_sha) != 64:
            raise ExecutionLockError("code-lock file digest invalid")
        try:
            file_sha_bytes = bytes.fromhex(file_sha)
        except ValueError as exc:
            raise ExecutionLockError("code-lock file digest invalid") from exc
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(file_sha_bytes)
        if row["path"] == "research/f1_ciip_v1/native/semantic_worker.cpp":
            source_hash = file_sha
    if body.get("code_sha256") != digest.hexdigest():
        raise ExecutionLockError("code-lock aggregate digest mismatch")
    binaries = body.get("native_binaries")
    if not isinstance(binaries, dict) or set(binaries) != set(ROLE_ORDER[:3]):
        raise ExecutionLockError("native role binding set mismatch")
    manifests: list[dict[str, Any]] = []
    for role in ROLE_ORDER[:3]:
        row = binaries[role]
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256", "build_manifest"}:
            raise ExecutionLockError("native binary row shape mismatch")
        binary_path = Path(row["path"])
        if binary_path.is_absolute() or ".." in binary_path.parts or binary_path.as_posix() != row["path"]:
            raise ExecutionLockError("native binary path is not safe canonical relative")
        if binary_path.parent != NATIVE_BUILD_RELATIVE:
            raise ExecutionLockError("native binary path is outside correction-v4 namespace")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ExecutionLockError("native binary byte count invalid")
        if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
            raise ExecutionLockError("native binary digest invalid")
        try:
            bytes.fromhex(row["sha256"])
        except ValueError as exc:
            raise ExecutionLockError("native binary digest invalid") from exc
        manifest = row["build_manifest"]
        expected_manifest_keys = {
            "compiler_path", "compiler_version", "compiler_sha256", "flags",
            "source_sha256", "binary_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
            raise ExecutionLockError("native build manifest shape mismatch")
        if row["sha256"] != manifest["binary_sha256"] or manifest["source_sha256"] != source_hash:
            raise ExecutionLockError("native source/binary cross-binding mismatch")
        if manifest["compiler_sha256"] != dependency["clang_binary_sha256"]:
            raise ExecutionLockError("native compiler cross-binding mismatch")
        if manifest["compiler_version"] != dependency["clang_version_stdout"]:
            raise ExecutionLockError("native compiler version cross-binding mismatch")
        if manifest["flags"][: len(COMPILE_FLAGS)] != list(COMPILE_FLAGS):
            raise ExecutionLockError("native compile flags cross-binding mismatch")
        manifests.append(manifest)
    if not all(manifest == manifests[0] for manifest in manifests[1:]):
        raise ExecutionLockError("semantic roles do not bind the same native build")


def write_new_execution_lock(path: Path, lock: Mapping[str, Any]) -> None:
    validate_execution_lock(lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags, 0o444)
    except OSError as exc:
        raise ExecutionLockError("execution lock already exists or cannot be created") from exc
    raw = canonical_json_bytes(dict(lock)) + b"\n"
    try:
        written = 0
        while written < len(raw):
            written += os.write(fd, raw[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_execution_lock(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        lock = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionLockError("execution lock is not canonical ASCII JSON") from exc
    if not isinstance(lock, dict) or canonical_json_bytes(lock) + b"\n" != raw:
        raise ExecutionLockError("execution lock is not byte-exact canonical JSON")
    validate_execution_lock(lock)
    return lock


__all__ = [
    "ATTEMPT_ID",
    "AUTHORIZATION_RECEIPT",
    "AUTHORIZATION_SCOPE",
    "CLAIM_RELATIVE",
    "COMPILE_FLAGS",
    "ExecutionLockError",
    "LOCK_RELATIVE",
    "NATIVE_BUILD_RELATIVE",
    "OUTPUT_RELATIVE",
    "PERMUTATION_RULE",
    "ROLE_ORDER",
    "SCRATCH_RELATIVE",
    "code_record",
    "dependency_record",
    "hash_open_regular_source",
    "load_execution_lock",
    "prepare_execution_lock",
    "validate_execution_lock",
    "write_new_execution_lock",
]
