#!/usr/bin/env python3
"""Guarded Candidate14 full-pool adapter preflight parent.

Repeatable modes authenticate only exact sources or fixed value-free controls.
The one-shot mode requires an exact external authorization and creates a
durable no-replace anchor before opening the TRAIN18 manifest.  The parent
reads and retains all 36 pool leaves, launches one fresh exact Candidate14
worker, streams a complete descriptor prelude followed by the 36 blobs, and
samples combined parent/worker Darwin memory at 10 ms.  Only one aggregate
result plus a value-free ledger and lock can be published.
"""

from __future__ import annotations

import builtins
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time
import types
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_C14_VERIFIED_ROOT_VALUE = globals().get("_C14_VERIFIED_ROOT")
ROOT = (
    Path(_C14_VERIFIED_ROOT_VALUE)
    if type(_C14_VERIFIED_ROOT_VALUE) is str
    and _C14_VERIFIED_ROOT_VALUE == "/Users/idantal/ShermanQC/FullProject"
    else Path(__file__).resolve().parents[1]
)
RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"
RUNNER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"
WORKER_SOURCE = ROOT / "scripts/run_f1_train18_candidate14_pool_adapter_worker_v1.py"
WORKER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate14_pool_adapter_worker_v1.py"
RESOURCE_SOURCE = ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py"
RESOURCE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate14_darwin_resource_capability_v1.py"
C13_CORE_SOURCE = ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py"
C13_CORE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py"
C9_ADAPTER_SOURCE = ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C9_ADAPTER_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C6_HYPOTHESIS_SOURCE = ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
BUILDER_SOURCE = ROOT / "scripts/build_f1_scan_only_candidate_pool.py"
CLUSTER_SOURCE = ROOT / "backend/f1_line_object_clustering.py"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
C13_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate13_terminal_disposition_v1.json"
CUSTODY_ROOT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
CACHE_ATTESTATION = CUSTODY_ROOT / "cache/train_v51_cache_manifest_attestation.json"
CACHE_MANIFEST = CUSTODY_ROOT / "cache/train_v51_cache_manifest.json"
CUSTODY_EVIDENCE_PATHS = MappingProxyType(
    {
        "access_evidence": CUSTODY_ROOT / "cache/access_evidence.json",
        "builder": ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py",
        "staged_validator_pass": CUSTODY_ROOT / "independent_validation_report.json",
        "validator": ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py",
    }
)
ALLOWED_CACHE_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"

C14_QUALIFICATION = ROOT / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_qualification_v1.json"
C14_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_authorization_v1.json"
C14_ANCHOR = ROOT / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_consumption_v1.json"
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate14-full-pool-adapter-preflight-v1"

SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 14
ENGINEERING_ATTEMPT_ID = "CANDIDATE14_FULL_POOL_ADAPTER_PREFLIGHT_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RECORD_COUNT = 36
PAIR_COUNT = 18
POOL_FILE_COUNT = 36
MAX_POOL_LEAF_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_PARENT_ENTRIES = 4096
MAX_WORKER_SPAWN_FD_INVENTORY = 256
MAX_BOUND_SOURCE_OR_CONTROL_BYTES = 4 * 1024 * 1024
MAX_AUTHORIZATION_BYTES = 64 * 1024
MAX_QUALIFICATION_BYTES = 64 * 1024
MAX_RUNNER_TEST_SOURCE_BYTES = 2 * 1024 * 1024
MEMORY_CEILING_BYTES = 8 * 1024 * 1024 * 1024
WALL_TIMEOUT_SECONDS = 300.0
COMBINED_CPU_LIMIT_SECONDS = 240.0
POLL_INTERVAL_SECONDS = 0.010
MAX_SAMPLING_GAP_SECONDS = 0.050
WORKER_REAP_GRACE_SECONDS = 5.0
WORKER_REAP_POLL_SECONDS = 0.010
CHILD_CPU_SOFT_LIMIT_SECONDS = 240
CHILD_CPU_HARD_LIMIT_SECONDS = 241

VERIFIED_EXEC_BOOTSTRAP_CODE = r"""import hashlib
import os
import stat
import sys
import types
from types import MappingProxyType

_ROOT = "/Users/idantal/ShermanQC/FullProject"
_EXPECTED = {
    "parent": _ROOT + "/scripts/run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py",
    "worker": _ROOT + "/scripts/run_f1_train18_candidate14_pool_adapter_worker_v1.py",
}


def _identity(value):
    return int(value.st_dev), int(value.st_ino)


def _read_verified(path_text, expected_sha, expected_size):
    if type(path_text) is not str or path_text != _EXPECTED.get(_ROLE):
        raise PermissionError("Candidate14 bootstrap path mismatch")
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or type(expected_size) is not int
        or not 0 < expected_size <= 4 * 1024 * 1024
    ):
        raise ValueError("Candidate14 bootstrap binding mismatch")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate14 bootstrap requires nofollow directories")
    relative_parts = os.path.relpath(path_text, _ROOT).split(os.path.sep)
    if relative_parts[:-1] != ["scripts"] or len(relative_parts) != 2:
        raise PermissionError("Candidate14 bootstrap relative path mismatch")
    walk_fd = os.open(os.path.sep, os.O_RDONLY | nofollow | directory)
    parent_fd = None
    descriptor = None
    failure = None
    data = None
    try:
        for component in _ROOT.split(os.path.sep)[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | nofollow | directory,
                dir_fd=walk_fd,
            )
            opened_directory = os.fstat(next_fd)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(next_fd)
                raise PermissionError("Candidate14 bootstrap directory mismatch")
            os.close(walk_fd)
            walk_fd = next_fd
        root_identity = _identity(os.fstat(walk_fd))
        for component in relative_parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | nofollow | directory,
                dir_fd=walk_fd,
            )
            opened_directory = os.fstat(next_fd)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(next_fd)
                raise PermissionError("Candidate14 bootstrap directory mismatch")
            os.close(walk_fd)
            walk_fd = next_fd
        parent_fd = walk_fd
        walk_fd = -1
        parent_opened = os.fstat(parent_fd)
        leaf_name = relative_parts[-1]
        before = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or int(before.st_size) != expected_size
        ):
            raise PermissionError("Candidate14 bootstrap leaf mismatch")
        descriptor = os.open(
            leaf_name, os.O_RDONLY | nofollow, dir_fd=parent_fd
        )
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != _identity(before)
            or int(opened.st_size) != expected_size
        ):
            raise PermissionError("Candidate14 bootstrap open mismatch")
        chunks = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise PermissionError("Candidate14 bootstrap source truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise PermissionError("Candidate14 bootstrap source grew")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible = os.stat(
            leaf_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or _identity(after) != identity
            or _identity(visible) != identity
            or int(after.st_size) != expected_size
            or _identity(os.fstat(parent_fd)) != _identity(parent_opened)
        ):
            raise PermissionError("Candidate14 bootstrap source changed")
    except BaseException as error:
        failure = error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                if failure is None:
                    failure = error
        for directory_fd in (parent_fd, walk_fd if walk_fd >= 0 else None):
            if directory_fd is None:
                continue
            try:
                os.close(directory_fd)
            except BaseException as error:
                if failure is None:
                    failure = error
    if failure is not None:
        raise failure
    if data is None or hashlib.sha256(data).hexdigest() != expected_sha:
        raise PermissionError("Candidate14 bootstrap source hash mismatch")
    return data, root_identity


def _launch():
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError("Candidate14 bootstrap requires isolated startup")
    arguments = list(sys.argv[1:])
    if len(arguments) < 5 or not all(type(value) is str for value in arguments):
        raise ValueError("Candidate14 bootstrap argument mismatch")
    global _ROLE
    _ROLE, path_text, expected_sha, size_text, *application_arguments = arguments
    if _ROLE not in _EXPECTED or not size_text.isdecimal():
        raise ValueError("Candidate14 bootstrap role or size mismatch")
    expected_size = int(size_text)
    data, root_identity = _read_verified(path_text, expected_sha, expected_size)
    private_name = "_candidate14_preverified_" + _ROLE + "_" + expected_sha
    if private_name in sys.modules:
        raise PermissionError("Candidate14 bootstrap module already exists")
    outer = sys.modules.get("__main__")
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate14 bootstrap outer module mismatch")
    outer_namespace = object.__getattribute__(outer, "__dict__")
    if outer_namespace.get("_C14_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate14 bootstrap is not reentrant")
    token = object()
    binding = MappingProxyType(
        {"module_name": private_name, "sha256": expected_sha, "size_bytes": expected_size}
    )
    prefix = "_C14_VERIFIED_RUNNER_" if _ROLE == "parent" else "_C14_VERIFIED_WORKER_"
    module = types.ModuleType(private_name)
    module.__file__ = path_text
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__[prefix + "BINDING"] = binding
    module.__dict__[prefix + "LAUNCH_TOKEN"] = token
    module.__dict__[prefix + "OUTER_MODULE"] = outer
    module.__dict__["_C14_VERIFIED_ROOT"] = _ROOT
    module.__dict__["_C14_VERIFIED_ROOT_IDENTITY"] = root_identity
    outer_namespace["_C14_OUTER_LAUNCH_TOKEN"] = token
    sys.modules[private_name] = module
    try:
        exec(compile(data, path_text, "exec", dont_inherit=True), module.__dict__)
        private_main = object.__getattribute__(module, "__dict__").get("main")
        if (
            type(private_main) is not types.FunctionType
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != path_text
        ):
            raise PermissionError("Candidate14 bootstrap entry point mismatch")
        return int(private_main(application_arguments))
    finally:
        sys.modules.pop(private_name, None)
        if outer_namespace.get("_C14_OUTER_LAUNCH_TOKEN") is token:
            del outer_namespace["_C14_OUTER_LAUNCH_TOKEN"]


try:
    _ROLE = sys.argv[1] if len(sys.argv) > 1 else ""
    _CODE = _launch()
except BaseException:
    if _ROLE == "worker":
        os.write(2, b"C14_WORKER_FAIL_CLOSED\n")
    else:
        os.write(1, b'{"check_code":"C14_RUNNER_REJECTED_OR_FAILED","diagnostic_complete":false,"holdout_label_evaluator_metric_fit_full28_opened_or_started":false,"status":"FAIL_CLOSED"}\n')
    _CODE = 1
raise SystemExit(_CODE)
"""

QUALIFICATION_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_qualification_v1"
AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_authorization_v1"
ANCHOR_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_consumption_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_terminal_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_read_ledger_v1"
LOCK_KIND = "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_lock_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"

DESCRIPTOR_MAGIC = b"C14POOLBATCHV1!!"
DESCRIPTOR_VERSION = 1
DESCRIPTOR_RECORD = struct.Struct(">IBQ32s")
BLOB_LENGTH = struct.Struct(">Q")
BATCH_COMMITMENT_DOMAIN = b"C14_POOL_BATCH_DESCRIPTOR_COMMITMENT_V1\x00"

FIXED_SOURCE_BINDINGS = MappingProxyType(
    {
        "builder": (BUILDER_SOURCE, "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2", 57613),
        "c13_core": (C13_CORE_SOURCE, "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b", 65344),
        "c13_core_tests": (C13_CORE_TESTS_SOURCE, "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15", 43387),
        "c14_resource": (RESOURCE_SOURCE, "c82fb204b61f99ce7fb7eab8b5ed15f9003be893c004dac37f347bbdbfe4c079", 15629),
        "c14_resource_tests": (RESOURCE_TESTS_SOURCE, "ac8843ed2fa3d83fc291c5f2ecb0c16fded3320d4484195658551366db8caa51", 12642),
        "c14_worker": (WORKER_SOURCE, "58e68be857aa81651fff10d423a43b4d7fe745f45dafe718a34dd68ec49bc137", 50687),
        "c14_worker_tests": (WORKER_TESTS_SOURCE, "6842932c572f448b1f6a087dee44965cdf5a935ba9991787ba808aea951d515a", 20844),
        "c6_hypothesis": (C6_HYPOTHESIS_SOURCE, "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252", 53837),
        "c9_adapter": (C9_ADAPTER_SOURCE, "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096", 22354),
        "c9_adapter_tests": (C9_ADAPTER_TESTS_SOURCE, "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623", 16598),
        "clusterer": (CLUSTER_SOURCE, "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c", 6817),
    }
)

FIXED_CONTROLS = MappingProxyType(
    {
        "SEALED_SPLIT": (SPLIT, "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185", 10706),
        "ORIGINAL_PREREGISTRATION": (ORIGINAL_PREREG, "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a", 2651),
        "TRAIN_ALLOWLIST": (ALLOWLIST, "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0", 2428),
        "CANDIDATE13_TERMINAL_DISPOSITION": (C13_DISPOSITION, "a0b2b04c9a3101e2a7228541c1003e34872ccb29c6c1c28318200060cfcf4abd", 3865),
        "TRAIN_CACHE_ATTESTATION": (CACHE_ATTESTATION, "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71", 1834),
    }
)

CACHE_MANIFEST_SHA256 = "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022"
CACHE_MANIFEST_SIZE_BYTES = 19601
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
V51_SCAN_ONLY_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

_IMPORT_HOOK_SEAL = (builtins.__import__, tuple(sys.meta_path), tuple(sys.path_hooks))
_CONTEXT_SEAL = object()
_QUALIFICATION_SEAL = object()
_AUTHORIZATION_SEAL = object()
_CONSUMPTION_SEAL = object()
_BASELINE_SEAL = object()
_ISSUED_CONTEXTS: dict[int, tuple[Any, object]] = {}
_ISSUED_QUALIFICATIONS: dict[int, tuple[Any, object, tuple[int, int, int, int], bytes, str]] = {}
_ISSUED_AUTHORIZATIONS: dict[int, tuple[Any, object]] = {}
_ISSUED_CONSUMPTIONS: dict[int, tuple[Any, object]] = {}
_ISSUED_BASELINES: dict[int, tuple[Any, object, tuple[float, int, int, float]]] = {}
_UNREAPED_PROCESS_QUARANTINE: list[Any] = []

_C14_VERIFIED_RUNNER_BINDING = globals().get("_C14_VERIFIED_RUNNER_BINDING")
_C14_VERIFIED_RUNNER_LAUNCH_TOKEN = globals().get("_C14_VERIFIED_RUNNER_LAUNCH_TOKEN")
_C14_VERIFIED_RUNNER_OUTER_MODULE = globals().get("_C14_VERIFIED_RUNNER_OUTER_MODULE")
_C14_VERIFIED_ROOT_IDENTITY_VALUE = globals().get("_C14_VERIFIED_ROOT_IDENTITY")
_VERIFIED_ROOT_DIRECTORY_IDENTITY: tuple[int, int] | None = (
    _C14_VERIFIED_ROOT_IDENTITY_VALUE
    if type(_C14_VERIFIED_ROOT_IDENTITY_VALUE) is tuple
    and len(_C14_VERIFIED_ROOT_IDENTITY_VALUE) == 2
    and all(
        type(value) is int and value >= 0
        for value in _C14_VERIFIED_ROOT_IDENTITY_VALUE
    )
    else None
)


class RunnerFailure(RuntimeError):
    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


class _OwnedWorkerPipe:
    __slots__ = ("_descriptor", "closed")

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor
        self.closed = False

    def fileno(self) -> int:
        if self.closed:
            raise ValueError("Candidate14 worker pipe is closed")
        return self._descriptor

    def close(self) -> None:
        if self.closed:
            return
        os.close(self._descriptor)
        self.closed = True
        self._descriptor = -1


@dataclass
class _SpawnedWorker:
    pid: int | None = None
    stdin: _OwnedWorkerPipe | None = None
    stdout: _OwnedWorkerPipe | None = None
    stderr: _OwnedWorkerPipe | None = None
    returncode: int | None = None
    child_stdin_descriptor: int = -1
    child_stdout_descriptor: int = -1
    child_stderr_descriptor: int = -1
    spawn_pipe_cleanup_failed: bool = False


class AnchorCreationFailure(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("Candidate14 runner hashes exact bytes")
    return hashlib.sha256(value).hexdigest()


def _verified_exec_bootstrap_commitment_sha256() -> str:
    return sha256_bytes(VERIFIED_EXEC_BOOTSTRAP_CODE.encode("utf-8"))


def _require_sha(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _verified_exec_command(
    role: str,
    source: Path,
    source_sha256: str,
    source_size_bytes: int,
    application_arguments: Sequence[str],
) -> list[str]:
    if role not in {"parent", "worker"}:
        raise ValueError("Candidate14 verified-exec role mismatch")
    expected_source = RUNNER_SOURCE if role == "parent" else WORKER_SOURCE
    candidate = Path(source)
    if candidate != expected_source:
        raise ValueError("Candidate14 verified-exec source path mismatch")
    sha = _require_sha(source_sha256, name=f"{role} source SHA")
    size = _strict_int(source_size_bytes, name=f"{role} source size")
    if not 0 < size <= 4 * 1024 * 1024 or not all(
        type(value) is str for value in application_arguments
    ):
        raise ValueError("Candidate14 verified-exec arguments mismatch")
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        VERIFIED_EXEC_BOOTSTRAP_CODE,
        role,
        str(candidate),
        sha,
        str(size),
        *application_arguments,
    ]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{name} is invalid JSON") from None
    if type(value) is not dict or canonical_bytes(value) != data:
        raise ValueError(f"{name} is not one canonical JSON object")
    return value


def _identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _lexical(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} contains parent traversal")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    root = Path(os.path.abspath(os.fspath(ROOT)))
    if os.path.commonpath((os.fspath(root), os.fspath(candidate))) != os.fspath(root):
        raise ValueError(f"{role} leaves the repository")
    return candidate


def _reject_symlinks(path: Path, *, role: str, require_exists: bool = True) -> Path:
    candidate = _lexical(path, role=role)
    root = _lexical(ROOT, role="repository")
    chain: list[Path] = []
    current = candidate
    while current != root:
        chain.append(current)
        current = current.parent
    chain.append(root)
    missing = False
    for item in reversed(chain):
        try:
            metadata = os.lstat(item)
        except FileNotFoundError:
            missing = True
            continue
        if missing:
            raise ValueError(f"{role} has a child below a missing ancestor")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path or ancestor is a symlink")
        if item != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor is not a directory")
    if require_exists and missing:
        raise FileNotFoundError(f"{role} must exist")
    return candidate


def _open_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _lexical(path, role=role)
    root = _lexical(ROOT, role="repository")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate14 requires O_NOFOLLOW/O_DIRECTORY")
    relative = candidate.relative_to(root)
    descriptor = os.open(os.path.sep, os.O_RDONLY | nofollow | directory)
    try:
        for component in root.parts[1:]:
            opened_descriptor = os.open(
                component,
                os.O_RDONLY | nofollow | directory,
                dir_fd=descriptor,
            )
            opened = os.fstat(opened_descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(opened_descriptor)
                raise RuntimeError(f"{role} component is not a directory")
            os.close(descriptor)
            descriptor = opened_descriptor
        root_identity = _identity(os.fstat(descriptor))
        if _C14_VERIFIED_RUNNER_BINDING is not None:
            if _VERIFIED_ROOT_DIRECTORY_IDENTITY != root_identity:
                raise RuntimeError("Candidate14 verified repository root changed")
        for component in relative.parts:
            opened_descriptor = os.open(
                component,
                os.O_RDONLY | nofollow | directory,
                dir_fd=descriptor,
            )
            opened = os.fstat(opened_descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(opened_descriptor)
                raise RuntimeError(f"{role} component is not a directory")
            os.close(descriptor)
            descriptor = opened_descriptor
        metadata = os.fstat(descriptor)
        return descriptor, _identity(metadata)
    except BaseException:
        os.close(descriptor)
        raise


def _stat_leaf(path: Path, *, role: str) -> os.stat_result:
    candidate = _lexical(path, role=role)
    if candidate == _lexical(ROOT, role="repository"):
        raise ValueError(f"{role} must be a repository leaf")
    parent_fd, parent_identity = _open_directory(
        candidate.parent, role=f"{role} parent"
    )
    failure: BaseException | None = None
    metadata: os.stat_result | None = None
    try:
        metadata = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError(f"{role} parent changed during stat")
    except BaseException as error:
        failure = error
    finally:
        try:
            os.close(parent_fd)
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    assert metadata is not None
    return metadata


def _read_regular_file_once(
    path: Path,
    *,
    role: str,
    expected_size: int,
    after_raw_fd_open: Any = None,
    after_verified_inode: Any = None,
    before_first_read: Any = None,
    after_exact_bytes_read: Any = None,
) -> tuple[bytes, tuple[int, int]]:
    if after_exact_bytes_read is not None and not callable(after_exact_bytes_read):
        raise TypeError(f"{role} exact-read callback is not callable")
    bounded_size = _strict_int(expected_size, name=f"{role} expected size")
    if bounded_size < 0:
        raise ValueError(f"{role} expected size is negative")
    candidate = _lexical(path, role=role)
    parent_fd, parent_identity = _open_directory(candidate.parent, role=f"{role} parent")
    before = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or int(before.st_size) != bounded_size
    ):
        raise ValueError(f"{role} must be one regular non-hard-linked file")
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        if after_raw_fd_open is not None:
            after_raw_fd_open()
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != _identity(before)
            or int(opened.st_size) != bounded_size
        ):
            raise ValueError(f"{role} changed during leaf open")
        if after_verified_inode is not None:
            after_verified_inode(identity)
        if before_first_read is not None:
            before_first_read(identity)
        chunks: list[bytes] = []
        remaining = bounded_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{role} ended before its exact size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise ValueError(f"{role} grew beyond its exact size")
        data = b"".join(chunks)
        if after_exact_bytes_read is not None:
            after_exact_bytes_read()
        after = os.fstat(descriptor)
        visible = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 or _identity(after) != identity or _identity(visible) != identity or int(after.st_size) != bounded_size or len(data) != bounded_size or _identity(os.fstat(parent_fd)) != parent_identity:
            raise ValueError(f"{role} changed during read")
        return data, identity
    finally:
        close_failure: BaseException | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                close_failure = error
        try:
            os.close(parent_fd)
        except BaseException as error:
            if close_failure is None:
                close_failure = error
        if close_failure is not None:
            raise close_failure


def _read_exact(path: Path, *, role: str, sha256: str, size_bytes: int, **callbacks: Any) -> tuple[bytes, dict[str, Any]]:
    if not 0 <= _strict_int(size_bytes, name=f"{role} size") <= MAX_BOUND_SOURCE_OR_CONTROL_BYTES:
        raise ValueError(f"{role} exceeds the bounded source/control size")
    data, identity = _read_regular_file_once(
        path, role=role, expected_size=size_bytes, **callbacks
    )
    if len(data) != _strict_int(size_bytes, name=f"{role} size") or sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA"):
        raise ValueError(f"{role} exact binding mismatch")
    return data, {"device_id": identity[0], "inode": identity[1], "path": str(path.relative_to(ROOT)), "purpose": role, "sha256": sha256, "size_bytes": size_bytes}


def _verified_runner_execution_binding() -> Mapping[str, Any]:
    binding = _C14_VERIFIED_RUNNER_BINDING
    if type(binding) is not MappingProxyType or set(binding) != {"module_name", "sha256", "size_bytes"} or binding["module_name"] != __name__:
        raise PermissionError("Candidate14 runner is not privately verified")
    _require_sha(binding["sha256"], name="runner SHA")
    if _strict_int(binding["size_bytes"], name="runner size") <= 0:
        raise PermissionError("Candidate14 runner size mismatch")
    return binding


def _validate_isolated_startup() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError(
            "Candidate14 parent requires isolated no-site Python startup"
        )


def _reject_preloaded(paths: Sequence[Path]) -> None:
    if builtins.__import__ is not _IMPORT_HOOK_SEAL[0] or tuple(sys.meta_path) != _IMPORT_HOOK_SEAL[1] or tuple(sys.path_hooks) != _IMPORT_HOOK_SEAL[2]:
        raise RuntimeError("Candidate14 rejects changed import hooks")
    targets = {path.resolve(strict=True) for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in paths}
    current = sys.modules.get(__name__)
    outer = _C14_VERIFIED_RUNNER_OUTER_MODULE
    token = _C14_VERIFIED_RUNNER_LAUNCH_TOKEN
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        if module is outer:
            if type(module) is types.ModuleType and object.__getattribute__(module, "__dict__").get("_C14_OUTER_LAUNCH_TOKEN") is token:
                continue
            raise RuntimeError("Candidate14 outer launch token mismatch")
        if not isinstance(module, types.ModuleType):
            continue
        if type(module) is not types.ModuleType:
            raise RuntimeError(f"Candidate14 rejects preloaded module subclass: {name}")
        loaded = object.__getattribute__(module, "__dict__").get("__file__")
        if type(loaded) is not str or not loaded:
            continue
        try:
            candidate = Path(loaded).resolve(strict=True)
            identity = _identity(os.stat(candidate, follow_symlinks=True))
        except OSError:
            continue
        if candidate in targets or identity in identities:
            raise RuntimeError(f"Candidate14 rejects preloaded source alias: {name}")


def _exec_verified(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate14 private module is already loaded")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    module.__loader__ = None
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class BootstrapContext:
    resource_core: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_entries: tuple[Mapping[str, Any], ...]
    source_commitment_sha256: str
    _resource_functions: tuple[Any, ...]
    _sentinel: object
    _token: object


def bootstrap_verified_sources(*, runner_sha256: str, runner_size_bytes: int, runner_tests_sha256: str, runner_tests_size_bytes: int) -> BootstrapContext:
    execution = _verified_runner_execution_binding()
    if execution["sha256"] != _require_sha(runner_sha256, name="runner SHA") or execution["size_bytes"] != _strict_int(runner_size_bytes, name="runner size"):
        raise PermissionError("Candidate14 executing runner binding mismatch")
    specs = {**FIXED_SOURCE_BINDINGS, "runner": (RUNNER_SOURCE, runner_sha256, runner_size_bytes), "runner_tests": (RUNNER_TESTS_SOURCE, _require_sha(runner_tests_sha256, name="runner tests SHA"), _strict_int(runner_tests_size_bytes, name="runner tests size"))}
    _reject_preloaded(tuple(value[0] for value in specs.values()))
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[Mapping[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha256, size_bytes = specs[name]
        data, entry = _read_exact(path, role=f"CANDIDATE14_SOURCE_{name.upper()}", sha256=sha256, size_bytes=size_bytes)
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate14 source inventory contains an inode alias")
        identities.add(identity)
        source_bytes[name] = data
        entries.append(MappingProxyType(entry))
        bindings[name] = {"path": entry["path"], "sha256": sha256, "size_bytes": size_bytes}
    resource_core = _exec_verified("_candidate14_parent_verified_resource_core", RESOURCE_SOURCE, source_bytes["c14_resource"], package="backend")
    if resource_core.ATTEMPT_ORDINAL != 14 or resource_core.ENGINEERING_ATTEMPT_ID != "CANDIDATE14_DARWIN_RESOURCE_CAPABILITY_V1" or resource_core.POLL_INTERVAL_SECONDS != POLL_INTERVAL_SECONDS or resource_core.CHILD_CPU_SOFT_LIMIT_SECONDS != CHILD_CPU_SOFT_LIMIT_SECONDS or resource_core.CHILD_CPU_HARD_LIMIT_SECONDS != CHILD_CPU_HARD_LIMIT_SECONDS or resource_core.MEMORY_ENFORCEMENT_IS_HARD_RLIMIT_AS is not False or resource_core.SAMPLING_OVERSHOOT_BETWEEN_POLLS_POSSIBLE is not True:
        raise ValueError("Candidate14 resource capability identity mismatch")
    immutable_bindings = MappingProxyType({name: MappingProxyType(dict(value)) for name, value in bindings.items()})
    commitment = sha256_bytes(canonical_bytes({name: dict(value) for name, value in sorted(immutable_bindings.items())}))
    token = object()
    context = BootstrapContext(resource_core=resource_core, source_bindings=immutable_bindings, source_entries=tuple(entries), source_commitment_sha256=commitment, _resource_functions=(resource_core.DarwinRusageSampler, resource_core.SampledDarwinCeiling, resource_core.run_repeatable_source_only_capability_probe), _sentinel=_CONTEXT_SEAL, _token=token)
    _ISSUED_CONTEXTS[id(context)] = (context, token)
    return context


def _validate_context(context: BootstrapContext) -> None:
    issued = _ISSUED_CONTEXTS.get(id(context))
    if type(context) is not BootstrapContext or context._sentinel is not _CONTEXT_SEAL or issued is None or issued[0] is not context or issued[1] is not context._token or context._resource_functions != (context.resource_core.DarwinRusageSampler, context.resource_core.SampledDarwinCeiling, context.resource_core.run_repeatable_source_only_capability_probe) or context.source_commitment_sha256 != sha256_bytes(canonical_bytes({name: dict(value) for name, value in sorted(context.source_bindings.items())})):
        raise PermissionError("Candidate14 bootstrap context changed")


def source_check_result(context: BootstrapContext) -> dict[str, Any]:
    _validate_context(context)
    return {"check_code": "C14_EXACT_SOURCE_BYTES_VERIFIED", "engineering_attempt_id": ENGINEERING_ATTEMPT_ID, "manifest_pool_semantic_label_evaluator_metric_fit_full28_holdout_opened": False, "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID, "source_inventory_commitment_sha256": context.source_commitment_sha256, "status": "SOURCE_BYTES_VERIFIED_ONLY"}


FORBIDDEN_INPUT_BASENAMES = frozenset(
    {
        "complete_label_manifest.json",
        "labels.json",
        "label_projection.json",
        "train_label_projection.json",
        "evaluator_output.json",
        "metrics.json",
        "semantic_result.json",
    }
)
FORBIDDEN_PURPOSE_TOKENS = frozenset(
    {"LABEL", "GOLD", "TRUTH", "SEMANTIC", "EVALUATOR", "METRIC", "FIT", "MODEL", "HOLDOUT", "FULL28"}
)


def _inside_cache_root(path: Path) -> bool:
    candidate = _lexical(path, role="TRAIN pool")
    root = _lexical(ALLOWED_CACHE_ROOT, role="TRAIN cache root")
    return candidate == root or root in candidate.parents


def _static_read_paths() -> frozenset[Path]:
    return frozenset(
        {
            *(_lexical(path, role="fixed control") for path, _sha, _size in FIXED_CONTROLS.values()),
            *(_lexical(path, role="custody evidence") for path in CUSTODY_EVIDENCE_PATHS.values()),
            _lexical(C14_AUTHORIZATION, role="authorization"),
            _lexical(C14_QUALIFICATION, role="qualification"),
            _lexical(CACHE_MANIFEST, role="manifest"),
        }
    )


@dataclass(frozen=True)
class ArtifactBinding:
    path: Path
    purpose: str
    sha256: str
    size_bytes: int
    identity: tuple[int, int]


@dataclass(frozen=True)
class ArtifactInventory:
    bindings: Mapping[Path, ArtifactBinding]
    commitment_sha256: str
    _reader: Any
    _nonce: object


def _expected_pool_purposes() -> list[str]:
    return [
        f"CANDIDATE14_{pass_name.upper()}_{index:02d}_POOL"
        for index in range(PAIR_COUNT)
        for pass_name in ("primary", "replay")
    ]


@dataclass
class GuardedReader:
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[str] = field(default_factory=list)
    raw_fd_opened_purposes: list[str] = field(default_factory=list)
    verified_inode_purposes: list[str] = field(default_factory=list)
    read_started_purposes: list[str] = field(default_factory=list)
    allowed: dict[Path, dict[str, Any]] = field(default_factory=dict)
    identities: dict[tuple[int, int], Path] = field(default_factory=dict)
    consumed_paths: set[Path] = field(default_factory=set)
    raw_fd_opened_paths: set[Path] = field(default_factory=set)
    verified_inode_paths: set[Path] = field(default_factory=set)
    read_started_paths: set[Path] = field(default_factory=set)
    artifact_inventory: ArtifactInventory | None = None

    def __post_init__(self) -> None:
        purposes = [str(entry["purpose"]) for entry in self.entries]
        for entry in self.entries:
            path = _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
            identity = int(entry["device_id"]), int(entry["inode"])
            if identity in self.identities and self.identities[identity] != path:
                raise ValueError("Candidate14 bootstrap ledger inode alias")
            self.identities[identity] = path
        if not self.raw_fd_opened_purposes:
            self.raw_fd_opened_purposes.extend(purposes)
        if not self.verified_inode_purposes:
            self.verified_inode_purposes.extend(purposes)
        if not self.read_started_purposes:
            self.read_started_purposes.extend(purposes)

    @classmethod
    def from_context(cls, context: BootstrapContext) -> "GuardedReader":
        _validate_context(context)
        return cls(entries=[dict(entry) for entry in context.source_entries])

    def _deny(self, purpose: str, reason: str) -> None:
        self.denied_attempts.append(f"{purpose}:{reason}")
        raise PermissionError("Candidate14 guarded read denied")

    def install_artifact_inventory(
        self, specifications: tuple[tuple[Path, str, str], ...]
    ) -> ArtifactInventory:
        if self.artifact_inventory is not None:
            self._deny("TRAIN_POOL_INVENTORY", "DUPLICATE_INVENTORY")
        if (
            type(specifications) is not tuple
            or len(specifications) != POOL_FILE_COUNT
            or [item[1] for item in specifications] != _expected_pool_purposes()
            or any(type(item) is not tuple or len(item) != 3 for item in specifications)
        ):
            raise ValueError("Candidate14 requires the exact ordered 36-pool inventory")
        bindings: dict[Path, ArtifactBinding] = {}
        identities = set(self.identities)
        projection: list[dict[str, Any]] = []
        for path, purpose, expected_sha in specifications:
            candidate = _lexical(path, role=purpose)
            if candidate.name in FORBIDDEN_INPUT_BASENAMES or any(token in purpose.upper() for token in FORBIDDEN_PURPOSE_TOKENS):
                self._deny(purpose, "FORBIDDEN_SURFACE")
            if not _inside_cache_root(candidate) or candidate in bindings:
                raise ValueError("Candidate14 pool inventory path mismatch")
            metadata = _stat_leaf(candidate, role=purpose)
            identity = _identity(metadata)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or not 0 <= int(metadata.st_size) <= MAX_POOL_LEAF_BYTES or identity in identities:
                raise ValueError("Candidate14 pool inventory metadata mismatch")
            identities.add(identity)
            binding = ArtifactBinding(candidate, purpose, _require_sha(expected_sha, name=f"{purpose} SHA"), int(metadata.st_size), identity)
            bindings[candidate] = binding
            projection.append({"device_id": identity[0], "inode": identity[1], "path": str(candidate.relative_to(ROOT)), "purpose": purpose, "sha256": binding.sha256, "size_bytes": binding.size_bytes})
        nonce = object()
        inventory = ArtifactInventory(MappingProxyType(dict(bindings)), sha256_bytes(canonical_bytes({"domain": "CANDIDATE14_EXACT_36_POOL_INVENTORY_V1", "files": projection})), self, nonce)
        self.artifact_inventory = inventory
        return inventory

    def authorize(self, path: Path, *, purpose: str, sha256: str, size_bytes: int, artifact_inventory: ArtifactInventory | None = None) -> None:
        candidate = _lexical(path, role=purpose)
        if candidate.name in FORBIDDEN_INPUT_BASENAMES or any(token in purpose.upper() for token in FORBIDDEN_PURPOSE_TOKENS):
            self._deny(purpose, "FORBIDDEN_SURFACE")
        expected_sha = _require_sha(sha256, name=f"{purpose} SHA")
        expected_size = _strict_int(size_bytes, name=f"{purpose} size")
        if expected_size < 0:
            raise ValueError("Candidate14 read size is invalid")
        if _inside_cache_root(candidate):
            inventory = self.artifact_inventory
            if artifact_inventory is None or artifact_inventory is not inventory or inventory is None or inventory._reader is not self:
                self._deny(purpose, "POOL_CAPABILITY_REQUIRED")
            binding = inventory.bindings.get(candidate)
            if binding is None or (binding.purpose, binding.sha256, binding.size_bytes) != (purpose, expected_sha, expected_size):
                self._deny(purpose, "POOL_OUTSIDE_INVENTORY")
        elif candidate not in _static_read_paths():
            self._deny(purpose, "PATH_OUTSIDE_ALLOWLIST")
        metadata = _stat_leaf(candidate, role=purpose)
        identity = _identity(metadata)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or int(metadata.st_size) != expected_size:
            raise ValueError("Candidate14 authorized metadata mismatch")
        if _inside_cache_root(candidate) and self.artifact_inventory is not None and self.artifact_inventory.bindings[candidate].identity != identity:
            raise ValueError("Candidate14 pool inode changed after inventory")
        if identity in self.identities and self.identities[identity] != candidate:
            raise ValueError("Candidate14 authorized inode alias")
        if candidate in self.consumed_paths or candidate in self.allowed:
            self._deny(purpose, "DUPLICATE_READ")
        self.allowed[candidate] = {"identity": identity, "purpose": purpose, "sha256": expected_sha, "size_bytes": expected_size}
        self.identities[identity] = candidate

    def read_bytes(
        self, path: Path, *, purpose: str, after_exact_bytes_read: Any = None
    ) -> bytes:
        candidate = _lexical(path, role=purpose)
        binding = self.allowed.get(candidate)
        if binding is None or binding["purpose"] != purpose or candidate in self.consumed_paths or candidate in self.raw_fd_opened_paths:
            self._deny(purpose, "NOT_EXACTLY_AUTHORIZED")

        def raw() -> None:
            self.raw_fd_opened_paths.add(candidate)
            self.raw_fd_opened_purposes.append(purpose)

        def inode(identity: tuple[int, int]) -> None:
            if identity != binding["identity"]:
                raise ValueError("Candidate14 read inode mismatch")
            self.verified_inode_paths.add(candidate)
            self.verified_inode_purposes.append(purpose)

        def start(identity: tuple[int, int]) -> None:
            if identity != binding["identity"] or candidate not in self.verified_inode_paths:
                raise ValueError("Candidate14 read-start mismatch")
            self.read_started_paths.add(candidate)
            self.read_started_purposes.append(purpose)

        data, identity = _read_regular_file_once(candidate, role=purpose, expected_size=int(binding["size_bytes"]), after_raw_fd_open=raw, after_verified_inode=inode, before_first_read=start, after_exact_bytes_read=after_exact_bytes_read)
        if identity != binding["identity"] or len(data) != binding["size_bytes"] or sha256_bytes(data) != binding["sha256"]:
            raise ValueError("Candidate14 exact read binding mismatch")
        self.entries.append({"device_id": identity[0], "inode": identity[1], "path": str(candidate.relative_to(ROOT)), "purpose": purpose, "sha256": binding["sha256"], "size_bytes": len(data)})
        self.consumed_paths.add(candidate)
        self.allowed.pop(candidate, None)
        return data


def _read_bound(reader: GuardedReader, *, path: Path, purpose: str, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    reader.authorize(path, purpose=purpose, sha256=sha256, size_bytes=size_bytes)
    data = reader.read_bytes(path, purpose=purpose)
    return data, _decode(data, name=purpose)


def _entry_commitment(entries: Sequence[Mapping[str, Any]], *, domain: str) -> str:
    projection = [{"device_id": _strict_int(entry["device_id"], name="device"), "inode": _strict_int(entry["inode"], name="inode"), "path": str(entry["path"]), "purpose": str(entry["purpose"]), "sha256": _require_sha(entry["sha256"], name="read SHA"), "size_bytes": _strict_int(entry["size_bytes"], name="read size")} for entry in entries]
    return sha256_bytes(canonical_bytes({"domain": domain, "reads": projection}))


def _purpose_commitment(domain: str, purposes: Sequence[str]) -> str:
    return sha256_bytes(canonical_bytes({"domain": domain, "purposes": list(purposes)}))


def _binding_from_attestation(value: Any, *, role: str, expected_path: Path, typed: bool = False, expected_kind: str | None = None) -> tuple[Path, str, int]:
    keys = {"path", "sha256", "size_bytes"} | ({"kind"} if typed else set())
    if type(value) is not dict or set(value) != keys or (typed and value.get("kind") != expected_kind):
        raise RunnerFailure("SOURCE_CONTROL", "ATTESTATION_BINDING_SCHEMA")
    relative = Path(str(value["path"]))
    path = _lexical(ROOT / relative, role=role)
    if relative.is_absolute() or ".." in relative.parts or path != _lexical(expected_path, role=f"expected {role}"):
        raise RunnerFailure("SOURCE_CONTROL", "ATTESTATION_BINDING_PATH")
    size = _strict_int(value["size_bytes"], name=f"{role} size")
    if size <= 0 or size > MAX_BOUND_SOURCE_OR_CONTROL_BYTES:
        raise RunnerFailure("SOURCE_CONTROL", "ATTESTATION_BINDING_SIZE")
    return path, _require_sha(value["sha256"], name=f"{role} SHA"), size


def _expected_control_purposes() -> list[str]:
    return [*FIXED_CONTROLS, "CACHE_CUSTODY_ACCESS_EVIDENCE", "CACHE_CUSTODY_BUILDER", "CACHE_CUSTODY_STAGED_VALIDATOR_PASS", "CACHE_CUSTODY_VALIDATOR"]


@dataclass(frozen=True)
class ControlQualification:
    reader: GuardedReader
    train_ids: tuple[str, ...]
    scan_by_case: Mapping[str, str]
    train_commitment_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    control_binding_commitment_sha256: str
    control_read_inventory_commitment_sha256: str
    control_raw_fd_opened_commitment_sha256: str
    control_verified_inode_commitment_sha256: str
    control_read_started_commitment_sha256: str
    resource_capability_result_commitment_sha256: str
    worker_source_result_commitment_sha256: str
    worker_source_inventory_commitment_sha256: str
    _context_token: object
    _sentinel: object
    _token: object


def _control_entries(reader: GuardedReader) -> list[dict[str, Any]]:
    purposes = set(_expected_control_purposes())
    return [entry for entry in reader.entries if str(entry["purpose"]) in purposes]


def _qualification_issuance_projection(
    qualification: ControlQualification,
    *,
    entry_length: int,
    raw_length: int,
    verified_length: int,
    started_length: int,
) -> bytes:
    if (
        type(qualification) is not ControlQualification
        or type(entry_length) is not int
        or type(raw_length) is not int
        or type(verified_length) is not int
        or type(started_length) is not int
        or min(entry_length, raw_length, verified_length, started_length) < 0
        or any(type(value) is not str for value in qualification.train_ids)
        or type(qualification.train_ids) is not tuple
        or type(qualification.scan_by_case) is not MappingProxyType
    ):
        raise PermissionError("Candidate14 qualification issuance schema mismatch")
    reader = qualification.reader
    entries = reader.entries[:entry_length]
    raw = reader.raw_fd_opened_purposes[:raw_length]
    verified = reader.verified_inode_purposes[:verified_length]
    started = reader.read_started_purposes[:started_length]
    if (
        len(entries) != entry_length
        or len(raw) != raw_length
        or len(verified) != verified_length
        or len(started) != started_length
        or any(type(value) is not str for value in raw + verified + started)
    ):
        raise PermissionError("Candidate14 qualification issuance prefix changed")
    entry_projection = [
        {
            "device_id": _strict_int(entry["device_id"], name="qualification device"),
            "inode": _strict_int(entry["inode"], name="qualification inode"),
            "path": str(entry["path"]),
            "purpose": str(entry["purpose"]),
            "sha256": _require_sha(entry["sha256"], name="qualification read SHA"),
            "size_bytes": _strict_int(entry["size_bytes"], name="qualification read size"),
        }
        for entry in entries
    ]
    prefix_paths = {
        _lexical(ROOT / entry["path"], role=entry["purpose"])
        for entry in entry_projection
    }
    source_prefix_paths = {
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in entries
        if str(entry["purpose"]).startswith("CANDIDATE14_SOURCE_")
    }
    non_source_prefix_paths = prefix_paths - source_prefix_paths
    if reader.allowed or not non_source_prefix_paths <= reader.consumed_paths:
        raise PermissionError("Candidate14 qualification has a latent read capability")
    consumed_projection = []
    for entry in entry_projection:
        path = _lexical(ROOT / entry["path"], role=entry["purpose"])
        if path in source_prefix_paths:
            continue
        identity = entry["device_id"], entry["inode"]
        if reader.identities.get(identity) != path:
            raise PermissionError("Candidate14 qualification consumed identity changed")
        consumed_projection.append(dict(entry))
    return canonical_bytes(
        {
            "consumed_prefix": consumed_projection,
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "control_raw_fd_opened_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
            "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
            "control_read_started_commitment_sha256": qualification.control_read_started_commitment_sha256,
            "control_verified_inode_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
            "domain": "CANDIDATE14_CONTROL_QUALIFICATION_ISSUANCE_V1",
            "entries": entry_projection,
            "manifest": {
                "path": str(qualification.manifest_path.relative_to(ROOT)),
                "sha256": qualification.manifest_sha256,
                "size_bytes": qualification.manifest_size_bytes,
            },
            "raw_fd_opened_purposes": raw,
            "read_started_purposes": started,
            "resource_capability_result_commitment_sha256": qualification.resource_capability_result_commitment_sha256,
            "scan_by_case": {name: qualification.scan_by_case[name] for name in sorted(qualification.scan_by_case)},
            "train_commitment_sha256": qualification.train_commitment_sha256,
            "train_ids": list(qualification.train_ids),
            "verified_inode_purposes": verified,
            "worker_source_inventory_commitment_sha256": qualification.worker_source_inventory_commitment_sha256,
            "worker_source_result_commitment_sha256": qualification.worker_source_result_commitment_sha256,
        }
    )


def _run_worker_source_check() -> dict[str, Any]:
    command = _verified_exec_command(
        "worker",
        WORKER_SOURCE,
        FIXED_SOURCE_BINDINGS["c14_worker"][1],
        FIXED_SOURCE_BINDINGS["c14_worker"][2],
        [
            "--worker-sha256",
            FIXED_SOURCE_BINDINGS["c14_worker"][1],
            "--worker-size-bytes",
            str(FIXED_SOURCE_BINDINGS["c14_worker"][2]),
            "--worker-tests-sha256",
            FIXED_SOURCE_BINDINGS["c14_worker_tests"][1],
            "--worker-tests-size-bytes",
            str(FIXED_SOURCE_BINDINGS["c14_worker_tests"][2]),
            "--source-check-only",
        ],
    )
    completed = subprocess.run(command, input=b"", capture_output=True, check=False, timeout=60, close_fds=True)
    if completed.returncode != 0 or completed.stderr != b"":
        raise RunnerFailure("SOURCE_CONTROL", "WORKER_SOURCE_CHECK_FAILED")
    value = _decode(completed.stdout, name="worker source check")
    if value.get("status") != "SOURCE_BYTES_VERIFIED_ONLY" or value.get("check_code") != "C14_WORKER_EXACT_SOURCE_BYTES_VERIFIED" or value.get("manifest_pool_semantic_label_evaluator_metric_fit_full28_holdout_opened") is not False:
        raise RunnerFailure("SOURCE_CONTROL", "WORKER_SOURCE_CHECK_CONTRACT")
    _require_sha(value.get("source_inventory_commitment_sha256"), name="worker source commitment")
    return value


def _validate_candidate13_terminal_disposition(disposition: Any) -> None:
    if (
        type(disposition) is not dict
        or disposition.get("kind")
        != "sherman_qc_f1_aug30_train18_candidate13_terminal_disposition_v1"
        or disposition.get("schema_version") != 1
        or type(disposition.get("schema_version")) is not int
        or disposition.get("engineering_attempt_id")
        != "CANDIDATE13_POOL_ADAPTER_STAGE_LOCALIZER_V1"
        or disposition.get("candidate14_engineering_successor_authorized") is not True
        or disposition.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
    ):
        raise RunnerFailure("SOURCE_CONTROL", "C13_DISPOSITION_MISMATCH")


def qualify_source_controls(context: BootstrapContext, *, reader: GuardedReader | None = None, authority_prefix: bool = False) -> ControlQualification:
    _validate_context(context)
    reader = GuardedReader.from_context(context) if reader is None else reader
    expected_prefix = [str(entry["purpose"]) for entry in context.source_entries]
    if authority_prefix:
        expected_prefix.extend(["CANDIDATE14_EXTERNAL_AUTHORIZATION", "CANDIDATE14_QUALIFICATION_RECORD"])
    if [str(entry["purpose"]) for entry in reader.entries] != expected_prefix:
        raise RunnerFailure("SOURCE_CONTROL", "READ_PREFIX_MISMATCH")
    controls: dict[str, dict[str, Any]] = {}
    for purpose, (path, sha, size) in FIXED_CONTROLS.items():
        _data, controls[purpose] = _read_bound(reader, path=path, purpose=purpose, sha256=sha, size_bytes=size)
    disposition = controls["CANDIDATE13_TERMINAL_DISPOSITION"]
    _validate_candidate13_terminal_disposition(disposition)
    split = controls["SEALED_SPLIT"]
    prereg = controls["ORIGINAL_PREREGISTRATION"]
    allowlist = controls["TRAIN_ALLOWLIST"]
    cases = allowlist.get("cases")
    if split.get("kind") != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1" or split.get("label_or_outcome_fields_present") is not False or prereg.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN" or allowlist.get("kind") != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1" or allowlist.get("reject_any_case_not_listed") is not True or type(cases) is not list or len(cases) != PAIR_COUNT or any(type(row) is not dict for row in cases):
        raise RunnerFailure("SOURCE_CONTROL", "SPLIT_ALLOWLIST_PREREG_MISMATCH")
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    if len(set(train_ids)) != PAIR_COUNT or any(not value for value in train_ids) or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids:
        raise RunnerFailure("SOURCE_CONTROL", "TRAIN_ORDER_MISMATCH")
    scan_by_case = {str(row["case_id"]): _require_sha(row.get("scan_sha256"), name="scan SHA") for row in cases}
    train_commitment = sha256_bytes(canonical_bytes([{"case_id": case_id, "scan_sha256": scan_by_case[case_id]} for case_id in train_ids]))
    attestation = controls["TRAIN_CACHE_ATTESTATION"]
    if attestation.get("kind") != "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1" or attestation.get("independent_validation_passed") is not True or attestation.get("case_count") != PAIR_COUNT or attestation.get("record_count") != RECORD_COUNT or attestation.get("holdout_entry_count") != 0 or attestation.get("train_exact_match") is not True or attestation.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1] or attestation.get("train_allowlist_sha256") != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1] or attestation.get("train_set_commitment_sha256") != train_commitment:
        raise RunnerFailure("SOURCE_CONTROL", "CACHE_ATTESTATION_MISMATCH")
    for prefix in ("access_evidence", "builder", "staged_validator_pass", "validator"):
        typed = prefix == "staged_validator_pass"
        path, sha, size = _binding_from_attestation(attestation.get(f"{prefix}_binding"), role=f"CACHE_CUSTODY_{prefix.upper()}", expected_path=CUSTODY_EVIDENCE_PATHS[prefix], typed=typed, expected_kind="sherman_qc_f1_aug30_train_only_staged_validator_pass_v1" if typed else None)
        reader.authorize(path, purpose=f"CACHE_CUSTODY_{prefix.upper()}", sha256=sha, size_bytes=size)
        reader.read_bytes(path, purpose=f"CACHE_CUSTODY_{prefix.upper()}")
    if (attestation.get("staged_validator_pass_binding") or {}).get("sha256") != CUSTODY_PASS_SHA256:
        raise RunnerFailure("SOURCE_CONTROL", "CUSTODY_PASS_MISMATCH")
    manifest_path, manifest_sha, manifest_size = _binding_from_attestation(attestation.get("manifest_binding"), role="TRAIN_CACHE_MANIFEST", expected_path=CACHE_MANIFEST, typed=True, expected_kind=CACHE_MANIFEST_KIND)
    if (manifest_sha, manifest_size) != (CACHE_MANIFEST_SHA256, CACHE_MANIFEST_SIZE_BYTES):
        raise RunnerFailure("SOURCE_CONTROL", "MANIFEST_BINDING_MISMATCH")
    capability = context.resource_core.run_repeatable_source_only_capability_probe()
    if capability.get("status") != "PASS_SOURCE_ONLY_NONAUTHORIZING" or capability.get("memory_enforcement_kind") != "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING" or capability.get("poll_interval_seconds") != "0.010" or capability.get("sampling_overshoot_between_polls_possible") is not True or capability.get("memory_enforcement_is_hard_rlimit_as") is not False:
        raise RunnerFailure("SOURCE_CONTROL", "RESOURCE_CAPABILITY_MISMATCH")
    worker_source = _run_worker_source_check()
    expected_tail = _expected_control_purposes()
    full_expected = expected_prefix + expected_tail
    if [str(entry["purpose"]) for entry in reader.entries[len(expected_prefix):]] != expected_tail or reader.raw_fd_opened_purposes != full_expected or reader.verified_inode_purposes != full_expected or reader.read_started_purposes != full_expected or reader.denied_attempts:
        raise RunnerFailure("SOURCE_CONTROL", "CONTROL_INVENTORY_MISMATCH")
    entries = _control_entries(reader)
    purposes = [str(entry["purpose"]) for entry in entries]
    control_inventory = _entry_commitment(entries, domain="CANDIDATE14_CONTROL_READ_INVENTORY_V1")
    control_binding = sha256_bytes(canonical_bytes({"domain": "CANDIDATE14_REPEATABLE_CONTROLS_V1", "read_inventory_commitment_sha256": control_inventory, "fixed_controls": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha, "size_bytes": size} for name, (path, sha, size) in sorted(FIXED_CONTROLS.items())}}))
    token = object()
    qualification = ControlQualification(reader, train_ids, MappingProxyType(scan_by_case), train_commitment, manifest_path, manifest_sha, manifest_size, control_binding, control_inventory, _purpose_commitment("CANDIDATE14_CONTROL_RAW_FD_OPENED_V1", purposes), _purpose_commitment("CANDIDATE14_CONTROL_VERIFIED_INODE_V1", purposes), _purpose_commitment("CANDIDATE14_CONTROL_READ_STARTED_V1", purposes), sha256_bytes(canonical_bytes(capability)), sha256_bytes(canonical_bytes(worker_source)), str(worker_source["source_inventory_commitment_sha256"]), context._token, _QUALIFICATION_SEAL, token)
    lengths = (
        len(reader.entries),
        len(reader.raw_fd_opened_purposes),
        len(reader.verified_inode_purposes),
        len(reader.read_started_purposes),
    )
    issuance_projection = _qualification_issuance_projection(
        qualification,
        entry_length=lengths[0],
        raw_length=lengths[1],
        verified_length=lengths[2],
        started_length=lengths[3],
    )
    _ISSUED_QUALIFICATIONS[id(qualification)] = (
        qualification,
        token,
        lengths,
        issuance_projection,
        sha256_bytes(issuance_projection),
    )
    return qualification


def _validate_qualification(context: BootstrapContext, qualification: ControlQualification) -> None:
    issued = _ISSUED_QUALIFICATIONS.get(id(qualification))
    if type(qualification) is not ControlQualification or qualification._sentinel is not _QUALIFICATION_SEAL or qualification._context_token is not context._token or issued is None or len(issued) != 5 or issued[0] is not qualification or issued[1] is not qualification._token or type(issued[2]) is not tuple or len(issued[2]) != 4 or any(type(value) is not int for value in issued[2]) or type(issued[3]) is not bytes or type(issued[4]) is not str or qualification.manifest_path != _lexical(CACHE_MANIFEST, role="manifest") or qualification.manifest_sha256 != CACHE_MANIFEST_SHA256 or qualification.manifest_size_bytes != CACHE_MANIFEST_SIZE_BYTES or qualification.train_commitment_sha256 != sha256_bytes(canonical_bytes([{"case_id": case_id, "scan_sha256": qualification.scan_by_case[case_id]} for case_id in qualification.train_ids])):
        raise PermissionError("Candidate14 qualification changed")
    current = _qualification_issuance_projection(
        qualification,
        entry_length=issued[2][0],
        raw_length=issued[2][1],
        verified_length=issued[2][2],
        started_length=issued[2][3],
    )
    if current != issued[3] or sha256_bytes(current) != issued[4]:
        raise PermissionError("Candidate14 qualification issuance commitment changed")


def _namespace_commitment(domain: str, path: Path) -> str:
    return sha256_bytes(canonical_bytes({"domain": domain, "namespace": str(_lexical(path, role="namespace").relative_to(ROOT))}))


def _manifest_contract_commitment(path: Path, sha256: str, size_bytes: int) -> str:
    return sha256_bytes(canonical_bytes({"domain": "CANDIDATE14_MANIFEST_CONTRACT_V1", "kind": CACHE_MANIFEST_KIND, "path": str(path.relative_to(ROOT)), "sha256": sha256, "size_bytes": size_bytes}))


def build_qualification_record(context: BootstrapContext, qualification: ControlQualification) -> bytes:
    _validate_context(context)
    _validate_qualification(context, qualification)
    return canonical_bytes(
        {
            "adapter_or_hypothesis_modified": False,
            "cache_manifest_opened": False,
            "check_code": "ALL_C14_SOURCE_CONTROL_CHECKS_PASSED",
            "combined_cpu_limit_seconds": "240.000",
            "consumption_anchor_namespace_commitment_sha256": _namespace_commitment(
                "CANDIDATE14_ANCHOR_NAMESPACE_V1", C14_ANCHOR
            ),
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "control_raw_fd_opened_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
            "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
            "control_read_started_commitment_sha256": qualification.control_read_started_commitment_sha256,
            "control_verified_inode_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
            "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
            "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
            "kind": QUALIFICATION_KIND,
            "manifest_binding_commitment_sha256": _manifest_contract_commitment(
                qualification.manifest_path,
                qualification.manifest_sha256,
                qualification.manifest_size_bytes,
            ),
            "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
            "memory_enforcement_is_hard_rlimit_as": False,
            "memory_enforcement_kind": "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING",
            "output_namespace_commitment_sha256": _namespace_commitment(
                "CANDIDATE14_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT
            ),
            "poll_interval_seconds": "0.010",
            "repeatable": True,
            "resource_capability_result_commitment_sha256": qualification.resource_capability_result_commitment_sha256,
            "sampling_gap_terminal": True,
            "sampling_max_gap_seconds": "0.050",
            "sampling_overshoot_between_polls_possible": True,
            "schema_version": 1,
            "scientific_hypothesis_consumed": False,
            "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
            "source_inventory_commitment_sha256": context.source_commitment_sha256,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
            "terminal_publication_reserved_outside_compute_resource_envelope": True,
            "terminal_publication_still_anchor_integrity_checked_when_receipt_available": True,
            "verified_exec_bootstrap_commitment_sha256": _verified_exec_bootstrap_commitment_sha256(),
            "worker_source_inventory_commitment_sha256": qualification.worker_source_inventory_commitment_sha256,
            "worker_source_result_commitment_sha256": qualification.worker_source_result_commitment_sha256,
            "wall_timeout_seconds": "300.000",
            "worker_cpu_hard_limit_seconds": CHILD_CPU_HARD_LIMIT_SECONDS,
            "worker_cpu_soft_limit_seconds": CHILD_CPU_SOFT_LIMIT_SECONDS,
        }
    )


def qualification_result(context: BootstrapContext) -> dict[str, Any]:
    try:
        qualification = qualify_source_controls(context)
        record = build_qualification_record(context, qualification)
        return {"cache_manifest_opened": False, "check_code": "ALL_C14_SOURCE_CONTROL_CHECKS_PASSED", "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256, "holdout_label_evaluator_metric_fit_full28_opened_or_started": False, "qualification_record_commitment_sha256": sha256_bytes(record), "resource_capability_passed": True, "source_inventory_commitment_sha256": context.source_commitment_sha256, "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE", "worker_source_check_passed": True}
    except BaseException:
        return {"cache_manifest_opened": False, "check_code": "C14_SOURCE_CONTROL_CHECK_FAILED", "control_binding_commitment_sha256": None, "holdout_label_evaluator_metric_fit_full28_opened_or_started": False, "qualification_record_commitment_sha256": None, "resource_capability_passed": False, "source_inventory_commitment_sha256": context.source_commitment_sha256, "status": "SOURCE_CONTROL_REJECTED", "worker_source_check_passed": False}


@dataclass(frozen=True)
class FreshOutputSeal:
    parent_path: Path
    parent_identity: tuple[int, int]
    _token: object


def _output_residue_prefixes() -> tuple[str, str]:
    return f".{OUTPUT_ROOT.name}.staging-", f".{OUTPUT_ROOT.name}.quarantine-"


def prevalidate_fresh_output(*, integrity_check: Any = None) -> FreshOutputSeal:
    if integrity_check is not None and not callable(integrity_check):
        raise TypeError("Candidate14 output integrity check is not callable")
    if integrity_check is not None:
        integrity_check()
    output = _lexical(OUTPUT_ROOT, role="output namespace")
    parent_fd, identity = _open_directory(output.parent, role="output parent")
    try:
        prefixes = _output_residue_prefixes()
        entry_count = 0
        with os.scandir(parent_fd) as entries:
            for entry in entries:
                if integrity_check is not None:
                    integrity_check()
                entry_count += 1
                if entry_count > MAX_OUTPUT_PARENT_ENTRIES:
                    raise OSError(
                        errno.E2BIG,
                        "Candidate14 output parent inventory exceeds its bound",
                    )
                name = entry.name
                if type(name) is not str:
                    raise TypeError("Candidate14 output parent entry name is invalid")
                if name == output.name or name.startswith(prefixes):
                    raise FileExistsError(
                        "Candidate14 output namespace or residue exists"
                    )
        if integrity_check is not None:
            integrity_check()
    finally:
        os.close(parent_fd)
    return FreshOutputSeal(output.parent, identity, object())


def _validate_fresh_output(
    seal: FreshOutputSeal, *, integrity_check: Any = None
) -> None:
    if type(seal) is not FreshOutputSeal:
        raise PermissionError("Candidate14 output freshness seal mismatch")
    current = prevalidate_fresh_output(integrity_check=integrity_check)
    if current.parent_path != seal.parent_path or current.parent_identity != seal.parent_identity:
        raise PermissionError("Candidate14 output parent changed")


def _binding(path: Path, sha256: str, size_bytes: int, *, kind: str | None = None) -> dict[str, Any]:
    size = _strict_int(size_bytes, name="binding size")
    if size <= 0:
        raise ValueError("Candidate14 binding size must be positive")
    value: dict[str, Any] = {"path": str(_lexical(path, role="binding").relative_to(ROOT)), "sha256": _require_sha(sha256, name="binding SHA"), "size_bytes": size}
    if kind is not None:
        value["kind"] = kind
    return value


def build_authorization_contract(context: BootstrapContext, qualification: ControlQualification, *, qualification_sha256: str, qualification_size_bytes: int) -> dict[str, Any]:
    _validate_context(context)
    _validate_qualification(context, qualification)
    return {
        "adapter_or_hypothesis_change_authorized": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "cache_manifest_read_authorized_once": True,
        "continuation_authorized": False,
        "combined_cpu_limit_seconds": "240.000",
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "consumption_anchor_namespace_commitment_sha256": _namespace_commitment("CANDIDATE14_ANCHOR_NAMESPACE_V1", C14_ANCHOR),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "exact_pool_inventory_read_authorized_once": True,
        "holdout_label_evaluator_metric_fit_full28_authorized": False,
        "kind": AUTHORIZATION_KIND,
        "manifest_binding_commitment_sha256": _manifest_contract_commitment(qualification.manifest_path, qualification.manifest_sha256, qualification.manifest_size_bytes),
        "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
        "memory_enforcement_is_hard_rlimit_as": False,
        "memory_enforcement_kind": "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING",
        "output_namespace_commitment_sha256": _namespace_commitment("CANDIDATE14_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT),
        "poll_interval_seconds": "0.010",
        "sampling_gap_terminal": True,
        "sampling_max_gap_seconds": "0.050",
        "qualification_binding": _binding(C14_QUALIFICATION, qualification_sha256, qualification_size_bytes, kind=QUALIFICATION_KIND),
        "resource_capability_result_commitment_sha256": qualification.resource_capability_result_commitment_sha256,
        "retry_authorized": False,
        "sampling_overshoot_between_polls_possible": True,
        "schema_version": SCHEMA_VERSION,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "terminal_publication_reserved_outside_compute_resource_envelope": True,
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available": True,
        "verified_exec_bootstrap_commitment_sha256": _verified_exec_bootstrap_commitment_sha256(),
        "worker_source_inventory_commitment_sha256": qualification.worker_source_inventory_commitment_sha256,
        "worker_source_result_commitment_sha256": qualification.worker_source_result_commitment_sha256,
        "wall_timeout_seconds": "300.000",
        "worker_cpu_hard_limit_seconds": CHILD_CPU_HARD_LIMIT_SECONDS,
        "worker_cpu_soft_limit_seconds": CHILD_CPU_SOFT_LIMIT_SECONDS,
    }


@dataclass(frozen=True)
class AuthorizationSeal:
    reader: GuardedReader
    qualification: ControlQualification
    authorization_bytes: bytes
    authorization_sha256: str
    authorization_size_bytes: int
    qualification_bytes: bytes
    qualification_sha256: str
    qualification_size_bytes: int
    output_freshness: FreshOutputSeal
    anchor_parent_identity: tuple[int, int]
    issuance_commitment_sha256: str
    _context_token: object
    _sentinel: object
    _token: object


def _authority_entries(reader: GuardedReader, context: BootstrapContext) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    offset = len(context.source_entries)
    if len(reader.entries) < offset + 2:
        raise PermissionError("Candidate14 authority entries are absent")
    return reader.entries[offset], reader.entries[offset + 1]


def _authorization_issuance_commitment(context: BootstrapContext, seal: AuthorizationSeal) -> str:
    auth_entry, qualification_entry = _authority_entries(seal.reader, context)
    return sha256_bytes(canonical_bytes({"authorization_bytes_sha256": sha256_bytes(seal.authorization_bytes), "authorization_entry": dict(auth_entry), "domain": "CANDIDATE14_AUTHORIZATION_ISSUANCE_V1", "qualification_bytes_sha256": sha256_bytes(seal.qualification_bytes), "qualification_entry": dict(qualification_entry), "source_inventory_commitment_sha256": context.source_commitment_sha256}))


def authorize_preflight_once(context: BootstrapContext, *, authorization_sha256: str, authorization_size_bytes: int, output_freshness: FreshOutputSeal | None = None) -> AuthorizationSeal:
    _validate_context(context)
    authorization_size = _strict_int(
        authorization_size_bytes, name="authorization size"
    )
    if not 0 < authorization_size <= MAX_AUTHORIZATION_BYTES:
        raise PermissionError("Candidate14 authorization exceeds its size bound")
    freshness = prevalidate_fresh_output() if output_freshness is None else output_freshness
    _validate_fresh_output(freshness)
    reader = GuardedReader.from_context(context)
    auth_bytes, auth = _read_bound(reader, path=C14_AUTHORIZATION, purpose="CANDIDATE14_EXTERNAL_AUTHORIZATION", sha256=_require_sha(authorization_sha256, name="authorization SHA"), size_bytes=authorization_size)
    qualification_binding = auth.get("qualification_binding")
    qualification_path, qualification_sha, qualification_size = _binding_from_attestation(qualification_binding, role="CANDIDATE14_QUALIFICATION_RECORD", expected_path=C14_QUALIFICATION, typed=True, expected_kind=QUALIFICATION_KIND)
    if qualification_size > MAX_QUALIFICATION_BYTES:
        raise PermissionError("Candidate14 qualification exceeds its size bound")
    qualification_bytes, _qualification_value = _read_bound(reader, path=qualification_path, purpose="CANDIDATE14_QUALIFICATION_RECORD", sha256=qualification_sha, size_bytes=qualification_size)
    qualification = qualify_source_controls(context, reader=reader, authority_prefix=True)
    expected_qualification = build_qualification_record(context, qualification)
    if qualification_bytes != expected_qualification:
        raise PermissionError("Candidate14 qualification record mismatch")
    expected_auth = build_authorization_contract(context, qualification, qualification_sha256=qualification_sha, qualification_size_bytes=qualification_size)
    if auth != expected_auth:
        raise PermissionError("Candidate14 external authorization contract mismatch")
    anchor_parent_fd, anchor_parent_identity = _open_directory(C14_ANCHOR.parent, role="anchor parent")
    os.close(anchor_parent_fd)
    token = object()
    seal = AuthorizationSeal(reader, qualification, auth_bytes, authorization_sha256, authorization_size, qualification_bytes, qualification_sha, qualification_size, freshness, anchor_parent_identity, "", context._token, _AUTHORIZATION_SEAL, token)
    object.__setattr__(seal, "issuance_commitment_sha256", _authorization_issuance_commitment(context, seal))
    _ISSUED_AUTHORIZATIONS[id(seal)] = (seal, token)
    return seal


def _validate_authorization(context: BootstrapContext, authorization: AuthorizationSeal) -> None:
    issued = _ISSUED_AUTHORIZATIONS.get(id(authorization))
    if type(authorization) is not AuthorizationSeal or authorization._sentinel is not _AUTHORIZATION_SEAL or authorization._context_token is not context._token or issued is None or issued[0] is not authorization or issued[1] is not authorization._token:
        raise PermissionError("Candidate14 authorization seal mismatch")
    _validate_qualification(context, authorization.qualification)
    auth = _decode(authorization.authorization_bytes, name="authorization")
    expected = build_authorization_contract(context, authorization.qualification, qualification_sha256=authorization.qualification_sha256, qualification_size_bytes=authorization.qualification_size_bytes)
    if auth != expected or sha256_bytes(authorization.authorization_bytes) != authorization.authorization_sha256 or len(authorization.authorization_bytes) != authorization.authorization_size_bytes or sha256_bytes(authorization.qualification_bytes) != authorization.qualification_sha256 or len(authorization.qualification_bytes) != authorization.qualification_size_bytes or authorization.qualification_bytes != build_qualification_record(context, authorization.qualification) or authorization.issuance_commitment_sha256 != _authorization_issuance_commitment(context, authorization):
        raise PermissionError("Candidate14 authorization bytes changed")
    _validate_fresh_output(authorization.output_freshness)


def _validate_preanchor_reader(context: BootstrapContext, authorization: AuthorizationSeal) -> None:
    reader = authorization.reader
    expected = [str(entry["purpose"]) for entry in context.source_entries] + ["CANDIDATE14_EXTERNAL_AUTHORIZATION", "CANDIDATE14_QUALIFICATION_RECORD"] + _expected_control_purposes()
    observed = [str(entry["purpose"]) for entry in reader.entries]
    non_source_entries = reader.entries[len(context.source_entries):]
    consumed_paths = {
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in non_source_entries
    }
    entry_paths = [
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in reader.entries
    ]
    entry_identities = [
        (int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries
    ]
    manifest = _lexical(CACHE_MANIFEST, role="pre-anchor manifest")
    active_paths = (
        set(reader.allowed)
        | reader.consumed_paths
        | reader.raw_fd_opened_paths
        | reader.verified_inode_paths
        | reader.read_started_paths
    )
    if (
        authorization.reader is not authorization.qualification.reader
        or observed != expected
        or reader.raw_fd_opened_purposes != expected
        or reader.verified_inode_purposes != expected
        or reader.read_started_purposes != expected
        or reader.denied_attempts
        or reader.artifact_inventory is not None
        or reader.allowed
        or reader.consumed_paths != consumed_paths
        or reader.raw_fd_opened_paths != consumed_paths
        or reader.verified_inode_paths != consumed_paths
        or reader.read_started_paths != consumed_paths
        or len(entry_paths) != len(set(entry_paths))
        or len(entry_identities) != len(set(entry_identities))
        or len(reader.identities) != len(entry_identities)
        or any(
            reader.identities.get(identity) != path
            for identity, path in zip(entry_identities, entry_paths)
        )
        or manifest in active_paths
        or any(_inside_cache_root(path) for path in active_paths)
        or any(
            purpose == "TRAIN_CACHE_MANIFEST" or purpose.endswith("_POOL")
            for purpose in observed
        )
    ):
        raise PermissionError("Candidate14 pre-anchor read state mismatch")


@dataclass
class AnchorProgress:
    raw_entry_created_or_partial: bool = False


@dataclass(frozen=True)
class ConsumptionSeal:
    path: Path
    sha256: str
    size_bytes: int
    identity: tuple[int, int]
    parent_identity: tuple[int, int]
    authorization_sha256: str
    qualification_sha256: str
    _context_token: object
    _sentinel: object
    _token: object


def _anchor_payload(context: BootstrapContext, authorization: AuthorizationSeal) -> bytes:
    return canonical_bytes({"attempt_ordinal": ATTEMPT_ORDINAL, "authorization_binding": _binding(C14_AUTHORIZATION, authorization.authorization_sha256, authorization.authorization_size_bytes, kind=AUTHORIZATION_KIND), "combined_cpu_limit_seconds": "240.000", "continuation_authorized": False, "engineering_attempt_id": ENGINEERING_ATTEMPT_ID, "kind": ANCHOR_KIND, "output_namespace_commitment_sha256": _namespace_commitment("CANDIDATE14_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT), "qualification_binding": _binding(C14_QUALIFICATION, authorization.qualification_sha256, authorization.qualification_size_bytes, kind=QUALIFICATION_KIND), "retry_authorized": False, "schema_version": 1, "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID, "source_inventory_commitment_sha256": context.source_commitment_sha256, "verified_exec_bootstrap_commitment_sha256": _verified_exec_bootstrap_commitment_sha256(), "wall_timeout_seconds": "300.000", "worker_cpu_hard_limit_seconds": CHILD_CPU_HARD_LIMIT_SECONDS, "worker_cpu_soft_limit_seconds": CHILD_CPU_SOFT_LIMIT_SECONDS})


def consume_authorization(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    *,
    resource_check: Any,
    progress: AnchorProgress | None = None,
) -> ConsumptionSeal:
    _validate_context(context)
    _validate_authorization(context, authorization)
    _validate_preanchor_reader(context, authorization)
    if type(resource_check) is not types.FunctionType:
        raise TypeError("Candidate14 anchor resource check must be an exact function")
    progress = AnchorProgress() if progress is None else progress
    payload = _anchor_payload(context, authorization)
    parent_fd, parent_identity = _open_directory(C14_ANCHOR.parent, role="anchor parent")
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    failure: BaseException | None = None
    try:
        if parent_identity != authorization.anchor_parent_identity:
            raise PermissionError("Candidate14 anchor parent changed")
        resource_check()
        descriptor = os.open(C14_ANCHOR.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400, dir_fd=parent_fd)
        progress.raw_entry_created_or_partial = True
        os.fsync(parent_fd)
        identity = _identity(os.fstat(descriptor))
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "anchor write failed")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        visible = os.stat(C14_ANCHOR.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o400 or _identity(opened) != identity or _identity(visible) != identity or stat.S_IMODE(visible.st_mode) != 0o400 or int(opened.st_size) != len(payload):
            raise OSError(errno.EIO, "anchor verification failed")
        os.fsync(parent_fd)
    except BaseException as error:
        failure = error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            os.close(parent_fd)
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is None:
        try:
            visible_bytes, visible_identity = _read_regular_file_once(
                C14_ANCHOR,
                role="consumption anchor post-create",
                expected_size=len(payload),
            )
            visible_metadata = _stat_leaf(
                C14_ANCHOR, role="consumption anchor post-create"
            )
            parent_check_fd, current_parent_identity = _open_directory(
                C14_ANCHOR.parent, role="consumption anchor parent post-create"
            )
            os.close(parent_check_fd)
            if (
                identity is None
                or visible_identity != identity
                or visible_bytes != payload
                or stat.S_IMODE(visible_metadata.st_mode) != 0o400
                or current_parent_identity != parent_identity
            ):
                raise OSError(errno.EIO, "anchor exact post-create binding failed")
        except BaseException as error:
            failure = error
    if failure is not None:
        if progress.raw_entry_created_or_partial:
            raise AnchorCreationFailure(
                "Candidate14 anchor creation consumed authority"
            ) from None
        raise failure
    assert identity is not None
    token = object()
    seal = ConsumptionSeal(C14_ANCHOR, sha256_bytes(payload), len(payload), identity, parent_identity, authorization.authorization_sha256, authorization.qualification_sha256, context._token, _CONSUMPTION_SEAL, token)
    _ISSUED_CONSUMPTIONS[id(seal)] = (seal, token)
    revalidate_anchor(context, seal)
    return seal


def revalidate_anchor(context: BootstrapContext, seal: ConsumptionSeal) -> None:
    issued = _ISSUED_CONSUMPTIONS.get(id(seal))
    if type(seal) is not ConsumptionSeal or seal._sentinel is not _CONSUMPTION_SEAL or seal._context_token is not context._token or issued is None or issued[0] is not seal or issued[1] is not seal._token:
        raise PermissionError("Candidate14 consumption seal mismatch")
    data, identity = _read_regular_file_once(
        seal.path, role="consumption anchor", expected_size=seal.size_bytes
    )
    metadata = _stat_leaf(seal.path, role="consumption anchor metadata")
    parent_check_fd, current_parent_identity = _open_directory(
        seal.path.parent, role="consumption anchor parent"
    )
    os.close(parent_check_fd)
    if identity != seal.identity or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o400 or len(data) != seal.size_bytes or sha256_bytes(data) != seal.sha256 or current_parent_identity != seal.parent_identity:
        raise PermissionError("Candidate14 consumption anchor changed")


@dataclass(frozen=True)
class ManifestRecord:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    pool_path: Path
    pool_sha256: str
    manifest_record_binding_sha256: str


@dataclass(frozen=True)
class PoolVector:
    record: ManifestRecord
    file_binding_sha256: str
    size_bytes: int
    pool_bytes: bytes


@dataclass(frozen=True)
class WorkerOutcome:
    aggregate: Mapping[str, Any]
    aggregate_bytes: bytes
    descriptor_commitment_sha256: str
    pool_inventory_commitment_sha256: str
    child_cpu_seconds: float
    child_maxrss_bytes: int


@dataclass
class ReadProgress:
    manifest_bytes_read_completed: bool = False
    manifest_contract_validated: bool = False
    all_36_pool_bytes_read_completed: bool = False
    pool_inventory_validated: bool = False
    pool_inventory_commitment_sha256: str | None = None


def _manifest_file_binding(entry: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({"device_id": _strict_int(entry["device_id"], name="manifest device"), "domain": "CANDIDATE14_ATTESTED_MANIFEST_FILE_V1", "inode": _strict_int(entry["inode"], name="manifest inode"), "path": str(entry["path"]), "sha256": _require_sha(entry["sha256"], name="manifest SHA"), "size_bytes": _strict_int(entry["size_bytes"], name="manifest size")}))


def _validate_full_manifest(manifest: Mapping[str, Any], qualification: ControlQualification, *, manifest_binding_sha256: str) -> tuple[ManifestRecord, ...]:
    expected_keys = {"case_count", "holdout_entry_count", "kind", "record_count", "records", "schema_version", "source_only", "split_sha256", "train_allowlist_sha256", "train_set_commitment_sha256", "v51_scan_only_manifest_sha256"}
    if type(manifest) is not dict or set(manifest) != expected_keys or manifest.get("schema_version") != 1 or type(manifest.get("schema_version")) is not int or manifest.get("kind") != CACHE_MANIFEST_KIND or manifest.get("source_only") is not True or manifest.get("case_count") != PAIR_COUNT or type(manifest.get("case_count")) is not int or manifest.get("record_count") != RECORD_COUNT or type(manifest.get("record_count")) is not int or manifest.get("holdout_entry_count") != 0 or type(manifest.get("holdout_entry_count")) is not int or manifest.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1] or manifest.get("train_allowlist_sha256") != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1] or manifest.get("train_set_commitment_sha256") != qualification.train_commitment_sha256 or manifest.get("v51_scan_only_manifest_sha256") != V51_SCAN_ONLY_MANIFEST_SHA256:
        raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_HEADER_MISMATCH")
    raw_records = manifest.get("records")
    if type(raw_records) is not list or len(raw_records) != RECORD_COUNT:
        raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_RECORD_INVENTORY_MISMATCH")
    expected_pairs = [(case_id, pass_name) for case_id in qualification.train_ids for pass_name in ("primary", "replay")]
    record_keys = {"case_id", "pass", "pool_path", "pool_sha256", "runtime_case_id", "scan_sha256", "semantic_path", "semantic_sha256"}
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    paths: set[Path] = set()
    bindings: set[str] = set()
    primary_pool: dict[str, str] = {}
    primary_semantic: dict[str, str] = {}
    result: list[ManifestRecord] = []
    for zero_ordinal, (raw, expected_pair) in enumerate(zip(raw_records, expected_pairs)):
        if type(raw) is not dict or set(raw) != record_keys or (raw.get("case_id"), raw.get("pass")) != expected_pair:
            raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_ORDER_OR_SCHEMA_MISMATCH")
        case_id, pass_name = expected_pair
        runtime_id = raw.get("runtime_case_id")
        if type(runtime_id) is not str or len(runtime_id) != 4 or not runtime_id.startswith("T") or not runtime_id[1:].isdigit() or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id or case_by_runtime.setdefault(runtime_id, case_id) != case_id or raw.get("scan_sha256") != qualification.scan_by_case[case_id]:
            raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_IDENTITY_MISMATCH")
        pool_relative_raw = raw.get("pool_path")
        semantic_relative_raw = raw.get("semantic_path")
        if type(pool_relative_raw) is not str or type(semantic_relative_raw) is not str:
            raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_PATH_MISMATCH")
        pool_relative = Path(pool_relative_raw)
        semantic_relative = Path(semantic_relative_raw)
        pool_path = _lexical(ROOT / pool_relative, role="TRAIN pool")
        expected_pool = _lexical(ALLOWED_CACHE_ROOT / pass_name / runtime_id / "pool" / f"{runtime_id}.json", role="expected TRAIN pool")
        expected_semantic = (ALLOWED_CACHE_ROOT / pass_name / runtime_id / "semantic_result.json").relative_to(ROOT)
        if pool_relative.is_absolute() or semantic_relative.is_absolute() or ".." in pool_relative.parts or ".." in semantic_relative.parts or pool_path != expected_pool or not _inside_cache_root(pool_path) or pool_path in paths or semantic_relative != expected_semantic:
            raise RunnerFailure("CACHE_MANIFEST", "MANIFEST_PATH_MISMATCH")
        paths.add(pool_path)
        pool_sha = _require_sha(raw.get("pool_sha256"), name="pool SHA")
        semantic_sha = _require_sha(raw.get("semantic_sha256"), name="semantic SHA")
        if pass_name == "primary":
            primary_pool[case_id] = pool_sha
            primary_semantic[case_id] = semantic_sha
        elif primary_pool.get(case_id) != pool_sha or primary_semantic.get(case_id) != semantic_sha:
            raise RunnerFailure("CACHE_MANIFEST", "PRIMARY_REPLAY_HASH_MISMATCH")
        record_binding = sha256_bytes(canonical_bytes({"domain": "CANDIDATE14_ATTESTED_MANIFEST_RECORD_V1", "manifest_binding_sha256": manifest_binding_sha256, "ordinal": zero_ordinal + 1, "row": raw}))
        if record_binding in bindings:
            raise RunnerFailure("CACHE_MANIFEST", "DUPLICATE_RECORD_BINDING")
        bindings.add(record_binding)
        result.append(ManifestRecord(zero_ordinal + 1, zero_ordinal // 2, pass_name, pool_path, pool_sha, record_binding))
    if len(paths) != POOL_FILE_COUNT or len(bindings) != RECORD_COUNT:
        raise RunnerFailure("CACHE_MANIFEST", "EXACT_MANIFEST_INVENTORY_MISMATCH")
    return tuple(result)


def _artifact_file_binding(entry: Mapping[str, Any], *, record_binding: str) -> str:
    return sha256_bytes(canonical_bytes({"device_id": _strict_int(entry["device_id"], name="pool device"), "domain": "CANDIDATE14_ATTESTED_POOL_FILE_V1", "inode": _strict_int(entry["inode"], name="pool inode"), "manifest_record_binding_sha256": record_binding, "path": str(entry["path"]), "purpose": str(entry["purpose"]), "sha256": _require_sha(entry["sha256"], name="pool SHA"), "size_bytes": _strict_int(entry["size_bytes"], name="pool size")}))


def read_full_pool_inventory(context: BootstrapContext, authorization: AuthorizationSeal, consumption: ConsumptionSeal, *, progress: ReadProgress | None = None) -> tuple[tuple[PoolVector, ...], str]:
    _validate_context(context)
    _validate_authorization(context, authorization)
    _validate_qualification(context, authorization.qualification)
    reader = authorization.reader
    progress = ReadProgress() if progress is None else progress
    revalidate_anchor(context, consumption)
    reader.authorize(authorization.qualification.manifest_path, purpose="TRAIN_CACHE_MANIFEST", sha256=authorization.qualification.manifest_sha256, size_bytes=authorization.qualification.manifest_size_bytes)
    revalidate_anchor(context, consumption)
    manifest_bytes = reader.read_bytes(
        authorization.qualification.manifest_path,
        purpose="TRAIN_CACHE_MANIFEST",
        after_exact_bytes_read=lambda: setattr(
            progress, "manifest_bytes_read_completed", True
        ),
    )
    revalidate_anchor(context, consumption)
    manifest = _decode(manifest_bytes, name="TRAIN_CACHE_MANIFEST")
    manifest_binding = _manifest_file_binding(reader.entries[-1])
    records = _validate_full_manifest(manifest, authorization.qualification, manifest_binding_sha256=manifest_binding)
    progress.manifest_contract_validated = True
    specifications = tuple((record.pool_path, f"CANDIDATE14_{record.pass_name.upper()}_{record.pair_ordinal:02d}_POOL", record.pool_sha256) for record in records)
    inventory = reader.install_artifact_inventory(specifications)
    progress.pool_inventory_commitment_sha256 = inventory.commitment_sha256
    vectors: list[PoolVector] = []
    completed_pool_reads = 0

    def pool_bytes_read_completed() -> None:
        nonlocal completed_pool_reads
        completed_pool_reads += 1
        if completed_pool_reads == RECORD_COUNT:
            progress.all_36_pool_bytes_read_completed = True

    for record, (path, purpose, expected_sha) in zip(records, specifications):
        revalidate_anchor(context, consumption)
        reader.authorize(path, purpose=purpose, sha256=expected_sha, size_bytes=inventory.bindings[path].size_bytes, artifact_inventory=inventory)
        revalidate_anchor(context, consumption)
        pool_bytes = reader.read_bytes(
            path,
            purpose=purpose,
            after_exact_bytes_read=pool_bytes_read_completed,
        )
        revalidate_anchor(context, consumption)
        entry = reader.entries[-1]
        vectors.append(PoolVector(record, _artifact_file_binding(entry, record_binding=record.manifest_record_binding_sha256), len(pool_bytes), pool_bytes))
    if type(vectors) is not list or len(vectors) != RECORD_COUNT or any(vectors[index].pool_bytes != vectors[index + 1].pool_bytes for index in range(0, RECORD_COUNT, 2)):
        raise RunnerFailure("POOL_INVENTORY", "PRIMARY_REPLAY_POOL_BYTE_MISMATCH")
    expected_purposes = [str(entry["purpose"]) for entry in context.source_entries] + ["CANDIDATE14_EXTERNAL_AUTHORIZATION", "CANDIDATE14_QUALIFICATION_RECORD"] + _expected_control_purposes() + ["TRAIN_CACHE_MANIFEST"] + _expected_pool_purposes()
    observed = [str(entry["purpose"]) for entry in reader.entries]
    paths = [
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in reader.entries
    ]
    identities = [(int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries]
    non_source_paths = set(paths[len(context.source_entries):])
    if observed != expected_purposes or reader.raw_fd_opened_purposes != expected_purposes or reader.verified_inode_purposes != expected_purposes or reader.read_started_purposes != expected_purposes or reader.allowed or reader.denied_attempts or reader.consumed_paths != non_source_paths or reader.raw_fd_opened_paths != non_source_paths or reader.verified_inode_paths != non_source_paths or reader.read_started_paths != non_source_paths or len(paths) != len(set(paths)) or len(identities) != len(set(identities)) or len(reader.identities) != len(identities) or any(reader.identities.get(identity) != path for identity, path in zip(identities, paths)) or reader.artifact_inventory is not inventory:
        raise RunnerFailure("POOL_INVENTORY", "EXACT_READ_INVENTORY_MISMATCH")
    revalidate_anchor(context, consumption)
    progress.pool_inventory_validated = True
    return tuple(vectors), inventory.commitment_sha256


def _descriptor_record_bytes(vector: PoolVector) -> bytes:
    pass_code = 1 if vector.record.pass_name == "primary" else 2
    if vector.record.ordinal % 2 != (1 if pass_code == 1 else 0) or vector.size_bytes != len(vector.pool_bytes) or sha256_bytes(vector.pool_bytes) != vector.record.pool_sha256:
        raise RunnerFailure("WORKER_PROTOCOL", "DESCRIPTOR_VECTOR_MISMATCH")
    return DESCRIPTOR_RECORD.pack(vector.record.ordinal, pass_code, vector.size_bytes, bytes.fromhex(vector.record.pool_sha256))


def build_descriptor_prelude(vectors: tuple[PoolVector, ...]) -> tuple[bytes, str]:
    if type(vectors) is not tuple or len(vectors) != RECORD_COUNT or tuple(vector.record.ordinal for vector in vectors) != tuple(range(1, RECORD_COUNT + 1)):
        raise RunnerFailure("WORKER_PROTOCOL", "DESCRIPTOR_INVENTORY_MISMATCH")
    records = b"".join(_descriptor_record_bytes(vector) for vector in vectors)
    commitment = hashlib.sha256(BATCH_COMMITMENT_DOMAIN + records).hexdigest()
    return DESCRIPTOR_MAGIC + struct.pack(">II", DESCRIPTOR_VERSION, RECORD_COUNT) + records + bytes.fromhex(commitment), commitment


C13_RESULT_KEYS = frozenset({"adapter_or_hypothesis_semantics_modified_by_core", "artifact_discovery_or_open_performed_by_core", "builder_or_clusterer_invoked_by_core", "candidate11_actual_record_cause_resolved", "candidate9_internal_failure_stage_claimed", "check_code", "diagnostic_stage_precedence_is_candidate13_contract", "exact_c9_success_failure_equivalent_for_all", "first_failed_candidate13_diagnostic_stage", "fit_performed", "full28_opened", "holdout_opened", "input_or_projection_bytes_published", "input_origin_authenticated_by_core", "kind", "label_or_evaluator_opened", "metric_computed", "performance_tested", "real_or_control_input_opened_by_core", "rule_selected", "stage_contract_commitment_sha256", "stage_passed_all_bitset", "stage_reached_all_bitset", "status"})
C13_STAGE_CONTRACT = ("CONTEXT_INPUT_HASH_AND_JSON", "HEADER", "ROW_IDENTITY_ORDER_SNAPSHOT", "RECURSIVE_FORBIDDEN_FIELDS", "SOURCE_AUTHORITY_ALIAS_SPELLING", "CLUSTER_PROVENANCE_NAMESPACE", "GEOMETRY_ALIAS_EQUIVALENCE", "CANONICAL_PROJECTION_BIJECTION_ORDER", "UNCHANGED_C6_VALIDATION", "EXACT_C9_OUTCOME_AND_RECEIPT_EQUIVALENCE")


def _expected_reached_failed(passed: str) -> tuple[str, str]:
    if type(passed) is not str or len(passed) != len(C13_STAGE_CONTRACT):
        return "INVALID", "INVALID"
    diagnostic = passed[:-1]
    failed = next((index for index, bit in enumerate(diagnostic) if bit == "0"), None)
    if failed is None:
        return "1" * len(C13_STAGE_CONTRACT), "NONE" if passed[-1] == "1" else C13_STAGE_CONTRACT[-1]
    if diagnostic != "1" * failed + "0" * (len(diagnostic) - failed):
        return "INVALID", "INVALID"
    return "1" * (failed + 1) + "0" * (len(diagnostic) - failed - 1) + "1", C13_STAGE_CONTRACT[failed]


def validate_worker_aggregate(value: Any) -> None:
    if type(value) is not dict or set(value) != C13_RESULT_KEYS:
        raise RunnerFailure("WORKER_RESULT", "AGGREGATE_SCHEMA_MISMATCH")
    reached = value["stage_reached_all_bitset"]
    passed = value["stage_passed_all_bitset"]
    expected_reached, expected_failed = _expected_reached_failed(passed)
    false_keys = C13_RESULT_KEYS - {"check_code", "diagnostic_stage_precedence_is_candidate13_contract", "exact_c9_success_failure_equivalent_for_all", "first_failed_candidate13_diagnostic_stage", "kind", "stage_contract_commitment_sha256", "stage_passed_all_bitset", "stage_reached_all_bitset", "status"}
    if value["kind"] != "sherman_qc_f1_aug30_candidate13_pool_adapter_stage_result_v1" or any(value[key] is not False for key in false_keys) or value["diagnostic_stage_precedence_is_candidate13_contract"] is not True or type(value["exact_c9_success_failure_equivalent_for_all"]) is not bool or type(reached) is not str or type(passed) is not str or len(reached) != 10 or len(passed) != 10 or set(reached + passed) - {"0", "1"} or reached != expected_reached or value["first_failed_candidate13_diagnostic_stage"] != expected_failed or value["exact_c9_success_failure_equivalent_for_all"] is not (passed[-1] == "1") or value["stage_contract_commitment_sha256"] != sha256_bytes(canonical_bytes(list(C13_STAGE_CONTRACT))):
        raise RunnerFailure("WORKER_RESULT", "AGGREGATE_INVARIANT_MISMATCH")
    all_pass = passed == "1" * 10
    expected = ("C13_LOCALIZER_EXACT_C9_OUTCOME_MISMATCH", "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE") if not value["exact_c9_success_failure_equivalent_for_all"] else (("C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL", "PASS_ENGINEERING_ONLY_NONAUTHORIZING") if all_pass else ("C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION", "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING"))
    if (value["check_code"], value["status"]) != expected:
        raise RunnerFailure("WORKER_RESULT", "AGGREGATE_STATUS_MISMATCH")


class _DarwinCpuSampler:
    def __init__(self, resource_core: types.ModuleType) -> None:
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pid_rusage
            function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(resource_core.RUsageInfoV2)]
            function.restype = ctypes.c_int
            self._library = library
            self._function = function
            self._record_type = resource_core.RUsageInfoV2
        except BaseException:
            raise RunnerFailure(
                "RESOURCE_MONITOR", "CPU_SAMPLER_INITIALIZATION_FAILED"
            ) from None

    def sample(self, pid: int) -> tuple[int, int]:
        if type(pid) is not int or pid <= 0:
            raise RunnerFailure("RESOURCE_MONITOR", "CPU_PID_INVALID")
        try:
            value = self._record_type()
            ctypes.set_errno(0)
            if self._function(pid, 2, ctypes.byref(value)) != 0:
                raise RunnerFailure("RESOURCE_MONITOR", "CPU_SAMPLE_FAILED")
            start = int(value.ri_proc_start_abstime)
            cpu_nanoseconds = int(value.ri_user_time) + int(value.ri_system_time)
            if start <= 0 or cpu_nanoseconds < 0:
                raise RunnerFailure("RESOURCE_MONITOR", "CPU_SAMPLE_INVALID")
            return start, cpu_nanoseconds
        except RunnerFailure:
            raise
        except BaseException:
            raise RunnerFailure(
                "RESOURCE_MONITOR", "CPU_SAMPLE_DEPENDENCY_FAILED"
            ) from None


def _worker_command() -> list[str]:
    return _verified_exec_command(
        "worker",
        WORKER_SOURCE,
        FIXED_SOURCE_BINDINGS["c14_worker"][1],
        FIXED_SOURCE_BINDINGS["c14_worker"][2],
        [
            "--worker-sha256",
            FIXED_SOURCE_BINDINGS["c14_worker"][1],
            "--worker-size-bytes",
            str(FIXED_SOURCE_BINDINGS["c14_worker"][2]),
            "--worker-tests-sha256",
            FIXED_SOURCE_BINDINGS["c14_worker_tests"][1],
            "--worker-tests-size-bytes",
            str(FIXED_SOURCE_BINDINGS["c14_worker_tests"][2]),
            "--worker-mode",
        ],
    )


@dataclass(frozen=True)
class ResourceBaseline:
    wall_started_at: float
    parent_process_start_abstime: int
    parent_cpu_nanoseconds: int
    reaped_children_cpu_seconds: float
    _context_token: object
    _sentinel: object
    _token: object


def _reaped_children_usage() -> tuple[float, int]:
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        user = float(usage.ru_utime)
        system = float(usage.ru_stime)
        maxrss = int(usage.ru_maxrss)
    except BaseException:
        raise RunnerFailure(
            "RESOURCE_MONITOR", "REAPED_CHILD_USAGE_SAMPLE_FAILED"
        ) from None
    if (
        not (0.0 <= user < float("inf"))
        or not (0.0 <= system < float("inf"))
        or maxrss < 0
    ):
        raise RunnerFailure("RESOURCE_MONITOR", "REAPED_CHILD_USAGE_INVALID")
    return user + system, maxrss


def begin_authorized_resource_envelope(context: BootstrapContext) -> ResourceBaseline:
    _validate_context(context)
    sampler = _DarwinCpuSampler(context.resource_core)
    parent_start, parent_cpu = sampler.sample(os.getpid())
    children_cpu, _children_maxrss = _reaped_children_usage()
    wall = time.monotonic()
    if not isinstance(wall, float) or not (0.0 < wall < float("inf")):
        raise RunnerFailure("RESOURCE_MONITOR", "RESOURCE_BASELINE_CLOCK_INVALID")
    token = object()
    baseline = ResourceBaseline(
        wall,
        parent_start,
        parent_cpu,
        children_cpu,
        context._token,
        _BASELINE_SEAL,
        token,
    )
    _ISSUED_BASELINES[id(baseline)] = (
        baseline,
        token,
        (wall, parent_start, parent_cpu, children_cpu),
    )
    return baseline


def _validate_resource_baseline(
    context: BootstrapContext, baseline: ResourceBaseline
) -> None:
    issued = _ISSUED_BASELINES.get(id(baseline))
    if (
        type(baseline) is not ResourceBaseline
        or baseline._context_token is not context._token
        or baseline._sentinel is not _BASELINE_SEAL
        or issued is None
        or issued[0] is not baseline
        or issued[1] is not baseline._token
        or issued[2]
        != (
            baseline.wall_started_at,
            baseline.parent_process_start_abstime,
            baseline.parent_cpu_nanoseconds,
            baseline.reaped_children_cpu_seconds,
        )
        or type(baseline.wall_started_at) is not float
        or not (0.0 < baseline.wall_started_at < float("inf"))
        or type(baseline.parent_process_start_abstime) is not int
        or baseline.parent_process_start_abstime <= 0
        or type(baseline.parent_cpu_nanoseconds) is not int
        or baseline.parent_cpu_nanoseconds < 0
        or type(baseline.reaped_children_cpu_seconds) is not float
        or not (0.0 <= baseline.reaped_children_cpu_seconds < float("inf"))
        or time.monotonic() < baseline.wall_started_at
    ):
        raise RunnerFailure("RESOURCE_MONITOR", "RESOURCE_BASELINE_CHANGED")


def _validate_authorized_resource_envelope_inner(
    context: BootstrapContext,
    baseline: ResourceBaseline,
    *,
    required_reaped_child_cpu_seconds: float = 0.0,
    cumulative_child_maxrss_bytes: int = 0,
) -> None:
    _validate_resource_baseline(context, baseline)
    if (
        type(required_reaped_child_cpu_seconds) is not float
        or not (0.0 <= required_reaped_child_cpu_seconds < float("inf"))
        or type(cumulative_child_maxrss_bytes) is not int
        or cumulative_child_maxrss_bytes < 0
    ):
        raise RunnerFailure("RESOURCE_MONITOR", "CUMULATIVE_CHILD_RESOURCE_INVALID")
    if time.monotonic() - baseline.wall_started_at > WALL_TIMEOUT_SECONDS:
        raise RunnerFailure("RESOURCE_MONITOR", "WALL_TIMEOUT")
    cpu_sampler = _DarwinCpuSampler(context.resource_core)
    parent_start, parent_cpu = cpu_sampler.sample(os.getpid())
    children_cpu, children_maxrss = _reaped_children_usage()
    children_delta = children_cpu - baseline.reaped_children_cpu_seconds
    if (
        parent_start != baseline.parent_process_start_abstime
        or parent_cpu < baseline.parent_cpu_nanoseconds
        or children_delta < 0.0
        or children_delta + 1e-6 < required_reaped_child_cpu_seconds
        or (
            (parent_cpu - baseline.parent_cpu_nanoseconds) / 1_000_000_000.0
            + children_delta
        )
        > COMBINED_CPU_LIMIT_SECONDS
    ):
        raise RunnerFailure(
            "RESOURCE_MONITOR", "CUMULATIVE_COMBINED_CPU_LIMIT_EXCEEDED"
        )
    sample = context.resource_core.DarwinRusageSampler().sample_pid(os.getpid())
    parent_peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if (
        parent_peak_rss < 0
        or children_maxrss > MEMORY_CEILING_BYTES
        or parent_peak_rss + cumulative_child_maxrss_bytes
        > MEMORY_CEILING_BYTES
        or type(sample.resident_size_bytes) is not int
        or type(sample.phys_footprint_bytes) is not int
        or sample.resident_size_bytes < 0
        or sample.phys_footprint_bytes < 0
        or
        max(sample.resident_size_bytes, sample.phys_footprint_bytes)
        > MEMORY_CEILING_BYTES
    ):
        raise RunnerFailure(
            "RESOURCE_MONITOR", "SAMPLED_MEMORY_CEILING_EXCEEDED"
        )


def validate_authorized_resource_envelope(
    context: BootstrapContext,
    baseline: ResourceBaseline,
    *,
    required_reaped_child_cpu_seconds: float = 0.0,
    cumulative_child_maxrss_bytes: int = 0,
) -> None:
    try:
        _validate_authorized_resource_envelope_inner(
            context,
            baseline,
            required_reaped_child_cpu_seconds=required_reaped_child_cpu_seconds,
            cumulative_child_maxrss_bytes=cumulative_child_maxrss_bytes,
        )
    except RunnerFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "RESOURCE_MONITOR", "CUMULATIVE_RESOURCE_SAMPLE_FAILED"
        ) from None


def _first_failure(
    current: RunnerFailure | None, stage: str, code: str
) -> RunnerFailure:
    return current if current is not None else RunnerFailure(stage, code)


def _quarantine_unreaped_worker(process: _SpawnedWorker) -> None:
    if (
        type(process) is not _SpawnedWorker
        or type(process.pid) is not int
        or process.pid <= 0
        or process.returncode is not None
    ):
        raise RunnerFailure(
            "WORKER_EXECUTION", "WORKER_UNREAPED_QUARANTINE_STATE_INVALID"
        )
    _UNREAPED_PROCESS_QUARANTINE.append(process)


def _spawn_worker_wait4_only(
    command: list[str], process: _SpawnedWorker
) -> None:
    if (
        type(command) is not list
        or not command
        or any(type(value) is not str or not value for value in command)
        or not os.path.isabs(command[0])
        or not callable(getattr(os, "posix_spawn", None))
        or type(process) is not _SpawnedWorker
        or process.pid is not None
        or process.stdin is not None
        or process.stdout is not None
        or process.stderr is not None
        or process.returncode is not None
        or process.child_stdin_descriptor != -1
        or process.child_stdout_descriptor != -1
        or process.child_stderr_descriptor != -1
        or process.spawn_pipe_cleanup_failed is not False
    ):
        raise RunnerFailure("WORKER_EXECUTION", "WORKER_POSIX_SPAWN_CONTRACT_INVALID")
    for descriptor in (0, 1, 2):
        try:
            os.fstat(descriptor)
        except BaseException:
            raise RunnerFailure(
                "WORKER_EXECUTION", "WORKER_STANDARD_DESCRIPTOR_MISSING"
            ) from None
    inventory_count = 0
    try:
        with os.scandir("/dev/fd") as entries:
            for entry in entries:
                inventory_count += 1
                if inventory_count > MAX_WORKER_SPAWN_FD_INVENTORY:
                    raise RunnerFailure(
                        "WORKER_EXECUTION", "WORKER_FD_INVENTORY_EXCEEDS_BOUND"
                    )
                name = entry.name
                if type(name) is not str or not name.isascii() or not name.isdigit():
                    raise RunnerFailure(
                        "WORKER_EXECUTION", "WORKER_FD_INVENTORY_INVALID"
                    )
                descriptor = int(name)
                if descriptor <= 2:
                    continue
                try:
                    inheritable = os.get_inheritable(descriptor)
                except OSError as error:
                    if error.errno == errno.EBADF:
                        continue
                    raise
                if inheritable:
                    try:
                        os.set_inheritable(descriptor, False)
                        inheritable = os.get_inheritable(descriptor)
                    except OSError as error:
                        if error.errno == errno.EBADF:
                            continue
                        raise
                    if inheritable:
                        raise RunnerFailure(
                            "WORKER_EXECUTION",
                            "WORKER_INHERITABLE_DESCRIPTOR_SANITIZATION_FAILED",
                        )
    except RunnerFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "WORKER_EXECUTION", "WORKER_FD_INVENTORY_FAILED"
        ) from None
    opened: list[int] = []
    try:
        child_stdin, parent_stdin = os.pipe()
        opened.extend((child_stdin, parent_stdin))
        parent_stdout, child_stdout = os.pipe()
        opened.extend((parent_stdout, child_stdout))
        parent_stderr, child_stderr = os.pipe()
        opened.extend((parent_stderr, child_stderr))
        if (
            len(set(opened)) != 6
            or any(type(descriptor) is not int or descriptor <= 2 for descriptor in opened)
        ):
            raise RunnerFailure("WORKER_EXECUTION", "WORKER_PIPE_IDENTITY_INVALID")
        for descriptor in opened:
            os.set_inheritable(descriptor, False)
            if os.get_inheritable(descriptor):
                raise RunnerFailure(
                    "WORKER_EXECUTION", "WORKER_PIPE_CLOEXEC_SETUP_FAILED"
                )
        process.stdin = _OwnedWorkerPipe(parent_stdin)
        process.stdout = _OwnedWorkerPipe(parent_stdout)
        process.stderr = _OwnedWorkerPipe(parent_stderr)
        process.child_stdin_descriptor = child_stdin
        process.child_stdout_descriptor = child_stdout
        process.child_stderr_descriptor = child_stderr
        environment = dict(os.environ)
        if any(
            type(key) is not str
            or type(value) is not str
            or "=" in key
            or "\x00" in key
            or "\x00" in value
            for key, value in environment.items()
        ):
            raise RunnerFailure(
                "WORKER_EXECUTION", "WORKER_SPAWN_ENVIRONMENT_INVALID"
            )
        file_actions = [
            (os.POSIX_SPAWN_DUP2, child_stdin, 0),
            (os.POSIX_SPAWN_DUP2, child_stdout, 1),
            (os.POSIX_SPAWN_DUP2, child_stderr, 2),
        ]
        file_actions.extend(
            (os.POSIX_SPAWN_CLOSE, descriptor) for descriptor in opened
        )
        signal_defaults = [
            value
            for value in (
                getattr(signal, "SIGPIPE", None),
                getattr(signal, "SIGXFZ", None),
                getattr(signal, "SIGXFSZ", None),
            )
            if value is not None
        ]
    except BaseException:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except BaseException:
                pass
        process.stdin = None
        process.stdout = None
        process.stderr = None
        process.child_stdin_descriptor = -1
        process.child_stdout_descriptor = -1
        process.child_stderr_descriptor = -1
        raise

    try:
        pid = os.posix_spawn(
            command[0],
            command,
            environment,
            file_actions=file_actions,
            setsigdef=signal_defaults,
        )
    except BaseException:
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except BaseException:
                pass
        for descriptor in (
            process.child_stdin_descriptor,
            process.child_stdout_descriptor,
            process.child_stderr_descriptor,
        ):
            try:
                if descriptor >= 0:
                    os.close(descriptor)
            except BaseException:
                pass
        process.stdin = None
        process.stdout = None
        process.stderr = None
        process.child_stdin_descriptor = -1
        process.child_stdout_descriptor = -1
        process.child_stderr_descriptor = -1
        raise
    process.pid = pid
    if type(pid) is not int or pid <= 0:
        process.spawn_pipe_cleanup_failed = True
    for attribute in (
        "child_stdin_descriptor",
        "child_stdout_descriptor",
        "child_stderr_descriptor",
    ):
        descriptor = getattr(process, attribute)
        try:
            os.close(descriptor)
        except BaseException:
            process.spawn_pipe_cleanup_failed = True
        else:
            setattr(process, attribute, -1)


def run_worker_supervised(
    context: BootstrapContext,
    vectors: tuple[PoolVector, ...],
    *,
    pool_inventory_commitment_sha256: str,
    resource_baseline: ResourceBaseline,
) -> WorkerOutcome:
    _validate_context(context)
    validate_authorized_resource_envelope(context, resource_baseline)
    prelude, expected_descriptor_commitment = build_descriptor_prelude(vectors)
    sampling_started = time.monotonic()
    process = _SpawnedWorker()
    selector: selectors.BaseSelector | None = None
    stdin_fd: int | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    pieces: list[memoryview] = []
    piece_index = 0
    piece_offset = 0
    stdout = bytearray()
    stderr = bytearray()
    stdin_open = stdout_open = stderr_open = False
    failure: RunnerFailure | None = None
    reaped = False
    wait_status: int | None = None
    child_rusage: Any = None
    kill_sent = False
    last_sample_started = sampling_started
    sample_index = 0
    next_sample = sampling_started

    def observe_exit() -> bool:
        nonlocal reaped, wait_status, child_rusage
        if reaped:
            return True
        if (
            type(process) is not _SpawnedWorker
            or type(process.pid) is not int
            or process.pid <= 0
        ):
            raise RunnerFailure("WORKER_EXECUTION", "WORKER_PID_STATE_INVALID")
        waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if waited_pid == 0:
            return False
        if waited_pid != process.pid:
            raise RunnerFailure("WORKER_EXECUTION", "WORKER_REAP_IDENTITY_MISMATCH")
        reaped = True
        wait_status = status
        child_rusage = usage
        process.returncode = os.waitstatus_to_exitcode(status)
        return True

    def kill_once() -> None:
        nonlocal kill_sent, failure
        if process is None or reaped or kill_sent:
            return
        try:
            os.kill(process.pid, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            try:
                observe_exit()
            except BaseException:
                failure = _first_failure(
                    failure, "WORKER_EXECUTION", "WORKER_REAP_FAILED"
                )
        except BaseException:
            failure = _first_failure(
                failure, "WORKER_EXECUTION", "WORKER_KILL_FAILED"
            )

    try:
        try:
            _spawn_worker_wait4_only(_worker_command(), process)
        except BaseException:
            raise RunnerFailure(
                "WORKER_EXECUTION", "WORKER_SPAWN_FAILED"
            ) from None
        if process.spawn_pipe_cleanup_failed:
            raise RunnerFailure(
                "WORKER_EXECUTION", "WORKER_SPAWN_PIPE_CLOSE_FAILED"
            )
        stdin_fd, stdout_fd, stderr_fd = (
            process.stdin.fileno(),
            process.stdout.fileno(),
            process.stderr.fileno(),
        )
        for descriptor in (stdin_fd, stdout_fd, stderr_fd):
            os.set_blocking(descriptor, False)
        try:
            memory_monitor = context.resource_core.SampledDarwinCeiling(
                context.resource_core.DarwinRusageSampler(),
                parent_pid=os.getpid(),
                worker_pid=process.pid,
                ceiling_bytes=MEMORY_CEILING_BYTES,
            )
        except BaseException:
            raise RunnerFailure(
                "RESOURCE_MONITOR", "MEMORY_MONITOR_INITIALIZATION_FAILED"
            ) from None
        cpu_sampler = _DarwinCpuSampler(context.resource_core)
        parent_start, parent_cpu = cpu_sampler.sample(os.getpid())
        worker_start, _worker_cpu_at_setup = cpu_sampler.sample(process.pid)
        if (
            parent_start != resource_baseline.parent_process_start_abstime
            or parent_cpu < resource_baseline.parent_cpu_nanoseconds
        ):
            raise RunnerFailure("RESOURCE_MONITOR", "PARENT_RESOURCE_BASELINE_CHANGED")
        selector = selectors.DefaultSelector()
        selector.register(stdin_fd, selectors.EVENT_WRITE, "stdin")
        selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
        selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
        stdin_open = stdout_open = stderr_open = True
        pieces = [memoryview(prelude)]
        for vector in vectors:
            pieces.append(memoryview(BLOB_LENGTH.pack(vector.size_bytes)))
            pieces.append(memoryview(vector.pool_bytes))

        while True:
            now = time.monotonic()
            if (
                not reaped
                and failure is None
                and now - last_sample_started > MAX_SAMPLING_GAP_SECONDS
            ):
                failure = RunnerFailure(
                    "RESOURCE_MONITOR", "SAMPLING_GAP_EXCEEDED"
                )
            exited = observe_exit()
            if not exited and failure is None and now >= next_sample:
                sample_started = time.monotonic()
                if sample_started - last_sample_started > MAX_SAMPLING_GAP_SECONDS:
                    failure = RunnerFailure(
                        "RESOURCE_MONITOR", "SAMPLING_GAP_EXCEEDED"
                    )
                else:
                    last_sample_started = sample_started
                    sample_index += 1
                    next_sample = (
                        sampling_started
                        + sample_index * POLL_INTERVAL_SECONDS
                    )
                    if (
                        sample_started - resource_baseline.wall_started_at
                        > WALL_TIMEOUT_SECONDS
                    ):
                        failure = RunnerFailure(
                            "RESOURCE_MONITOR", "WALL_TIMEOUT"
                        )
                    else:
                        try:
                            memory_outcome = memory_monitor.sample_and_enforce()
                        except BaseException:
                            failure = RunnerFailure(
                                "RESOURCE_MONITOR",
                                "MEMORY_SAMPLE_OR_KILL_FAILED",
                            )
                        if failure is None and memory_outcome != "WITHIN_SAMPLED_CEILING":
                            kill_sent = True
                            failure = RunnerFailure(
                                "RESOURCE_MONITOR",
                                "SAMPLED_MEMORY_CEILING_EXCEEDED",
                            )
                        if failure is None:
                            current_parent_start, current_parent_cpu = cpu_sampler.sample(
                                os.getpid()
                            )
                            current_worker_start, current_worker_cpu = cpu_sampler.sample(
                                process.pid
                            )
                            reaped_children_cpu, _reaped_children_maxrss = (
                                _reaped_children_usage()
                            )
                            reaped_children_delta = (
                                reaped_children_cpu
                                - resource_baseline.reaped_children_cpu_seconds
                            )
                            if (
                                current_parent_start
                                != resource_baseline.parent_process_start_abstime
                                or current_worker_start != worker_start
                                or reaped_children_delta < 0.0
                            ):
                                failure = RunnerFailure(
                                    "RESOURCE_MONITOR", "PROCESS_IDENTITY_CHANGED"
                                )
                            elif (
                                current_parent_cpu
                                - resource_baseline.parent_cpu_nanoseconds
                            ) / 1_000_000_000.0 + reaped_children_delta + (
                                current_worker_cpu / 1_000_000_000.0
                            ) > COMBINED_CPU_LIMIT_SECONDS:
                                failure = RunnerFailure(
                                    "RESOURCE_MONITOR", "COMBINED_CPU_LIMIT_EXCEEDED"
                                )
            if failure is not None:
                kill_once()
                if stdin_open and selector is not None and stdin_fd is not None:
                    try:
                        selector.unregister(stdin_fd)
                    except BaseException:
                        pass
                    try:
                        process.stdin.close()
                    except BaseException:
                        pass
                    stdin_open = False
                break
            timeout = max(
                0.0,
                min(POLL_INTERVAL_SECONDS, next_sample - time.monotonic()),
            )
            assert selector is not None
            for key, _mask in selector.select(timeout):
                event_started = time.monotonic()
                if (
                    not reaped
                    and failure is None
                    and event_started - last_sample_started
                    > MAX_SAMPLING_GAP_SECONDS
                ):
                    failure = RunnerFailure(
                        "RESOURCE_MONITOR", "SAMPLING_GAP_EXCEEDED"
                    )
                    kill_once()
                if key.data == "stdin" and stdin_open and failure is None:
                    while (
                        piece_index < len(pieces)
                        and piece_offset == len(pieces[piece_index])
                    ):
                        piece_index += 1
                        piece_offset = 0
                    if piece_index >= len(pieces):
                        selector.unregister(stdin_fd)
                        process.stdin.close()
                        stdin_open = False
                        continue
                    piece = pieces[piece_index]
                    try:
                        written = os.write(
                            stdin_fd,
                            piece[piece_offset : piece_offset + 1024 * 1024],
                        )
                    except BlockingIOError:
                        written = 0
                    except BrokenPipeError:
                        failure = RunnerFailure(
                            "WORKER_EXECUTION", "WORKER_STDIN_BROKEN_PIPE"
                        )
                        written = 0
                    if written:
                        piece_offset += written
                        if piece_offset == len(piece):
                            piece_index += 1
                            piece_offset = 0
                elif key.data == "stdout" and stdout_open:
                    try:
                        chunk = os.read(stdout_fd, 65536)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        selector.unregister(stdout_fd)
                        stdout_open = False
                    elif chunk:
                        stdout.extend(chunk)
                        if len(stdout) > 65536:
                            failure = _first_failure(
                                failure, "WORKER_RESULT", "WORKER_STDOUT_OVERSIZE"
                            )
                            kill_once()
                elif key.data == "stderr" and stderr_open:
                    try:
                        chunk = os.read(stderr_fd, 4096)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        selector.unregister(stderr_fd)
                        stderr_open = False
                    elif chunk:
                        stderr.extend(chunk)
                        if len(stderr) > 4096:
                            failure = _first_failure(
                                failure,
                                "WORKER_EXECUTION",
                                "WORKER_STDERR_OVERSIZE",
                            )
                            kill_once()
            if reaped and not stdout_open and not stderr_open:
                break
    except RunnerFailure as error:
        failure = error if failure is None else failure
    except BaseException:
        failure = _first_failure(
            failure, "WORKER_EXECUTION", "WORKER_SUPERVISION_FAILED"
        )
    finally:
        if not reaped and process.pid is not None:
            kill_once()
            reap_deadline = time.monotonic() + WORKER_REAP_GRACE_SECONDS
            while not reaped and time.monotonic() < reap_deadline:
                try:
                    observe_exit()
                except BaseException:
                    failure = _first_failure(
                        failure, "WORKER_EXECUTION", "WORKER_REAP_FAILED"
                    )
                    break
                if not reaped:
                    kill_once()
                    remaining = reap_deadline - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(min(WORKER_REAP_POLL_SECONDS, remaining))
            if not reaped:
                failure = _first_failure(
                    failure, "WORKER_EXECUTION", "WORKER_REAP_TIMEOUT"
                )
                try:
                    _quarantine_unreaped_worker(process)
                except RunnerFailure as error:
                    failure = _first_failure(failure, error.stage, error.code)
        if selector is not None:
            try:
                selector.close()
            except BaseException:
                failure = _first_failure(
                    failure, "WORKER_EXECUTION", "SELECTOR_CLOSE_FAILED"
                )
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except BaseException:
                failure = _first_failure(
                    failure, "WORKER_EXECUTION", "WORKER_PIPE_CLOSE_FAILED"
                )
        for attribute in (
            "child_stdin_descriptor",
            "child_stdout_descriptor",
            "child_stderr_descriptor",
        ):
            descriptor = getattr(process, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except BaseException:
                failure = _first_failure(
                    failure,
                    "WORKER_EXECUTION",
                    "WORKER_AUXILIARY_PIPE_CLOSE_FAILED",
                )
            else:
                setattr(process, attribute, -1)

    try:
        final_wall_elapsed = time.monotonic() - resource_baseline.wall_started_at
        if not (0.0 <= final_wall_elapsed < float("inf")):
            failure = RunnerFailure(
                "RESOURCE_MONITOR", "FINAL_WALL_CLOCK_INVALID"
            )
        elif final_wall_elapsed > WALL_TIMEOUT_SECONDS:
            failure = RunnerFailure("RESOURCE_MONITOR", "WALL_TIMEOUT")
    except BaseException:
        failure = RunnerFailure("RESOURCE_MONITOR", "FINAL_WALL_GATE_FAILED")

    if child_rusage is None or wait_status is None:
        failure = _first_failure(
            failure, "WORKER_EXECUTION", "WORKER_FINAL_RUSAGE_MISSING"
        )
    else:
        try:
            cpu_sampler = _DarwinCpuSampler(context.resource_core)
            final_parent_start, final_parent_cpu = cpu_sampler.sample(os.getpid())
            child_user = float(child_rusage.ru_utime)
            child_system = float(child_rusage.ru_stime)
            child_maxrss = float(child_rusage.ru_maxrss)
            reaped_children_cpu, reaped_children_maxrss = (
                _reaped_children_usage()
            )
            reaped_children_delta = (
                reaped_children_cpu
                - resource_baseline.reaped_children_cpu_seconds
            )
            parent_peak_rss = float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            if (
                final_parent_cpu < resource_baseline.parent_cpu_nanoseconds
                or reaped_children_delta < 0.0
                or reaped_children_delta + 1e-6 < child_user + child_system
                or any(
                    not (0.0 <= value < float("inf"))
                    for value in (
                        child_user,
                        child_system,
                        child_maxrss,
                        parent_peak_rss,
                    )
                )
            ):
                raise RunnerFailure(
                    "RESOURCE_MONITOR", "FINAL_RESOURCE_RUSAGE_INVALID"
                )
            combined_cpu = (
                final_parent_cpu - resource_baseline.parent_cpu_nanoseconds
            ) / 1_000_000_000.0 + reaped_children_delta
            combined_maxrss = parent_peak_rss + child_maxrss
            if (
                final_parent_start
                != resource_baseline.parent_process_start_abstime
                or combined_cpu > COMBINED_CPU_LIMIT_SECONDS
            ):
                failure = RunnerFailure(
                    "RESOURCE_MONITOR", "FINAL_COMBINED_CPU_LIMIT_EXCEEDED"
                )
            if (
                combined_maxrss > MEMORY_CEILING_BYTES
                or reaped_children_maxrss > MEMORY_CEILING_BYTES
            ):
                failure = RunnerFailure(
                    "RESOURCE_MONITOR", "FINAL_COMBINED_MAXRSS_EXCEEDED"
                )
        except RunnerFailure as error:
            failure = error
        except BaseException:
            failure = RunnerFailure(
                "RESOURCE_MONITOR", "FINAL_RESOURCE_GATE_FAILED"
            )
    if failure is not None:
        raise failure
    return_code = os.waitstatus_to_exitcode(wait_status)
    if return_code != 0 or bytes(stderr) != b"" or piece_index != len(pieces) or stdin_open:
        raise RunnerFailure("WORKER_EXECUTION", "WORKER_FAILED_OR_INPUT_INCOMPLETE")

    envelope_bytes = bytes(stdout)
    envelope = _decode(envelope_bytes, name="worker result envelope")
    expected_envelope_keys = {
        "batch_commitment_sha256",
        "blob_inventory_complete",
        "c13_aggregate",
        "c13_aggregate_commitment_sha256",
        "descriptor_inventory_complete",
        "input_eof_verified",
        "kind",
        "schema_version",
    }
    aggregate = envelope.get("c13_aggregate")
    if (
        set(envelope) != expected_envelope_keys
        or envelope.get("kind")
        != "sherman_qc_f1_aug30_candidate14_worker_aggregate_envelope_v1"
        or envelope.get("schema_version") != 1
        or type(envelope.get("schema_version")) is not int
        or envelope.get("blob_inventory_complete") is not True
        or envelope.get("descriptor_inventory_complete") is not True
        or envelope.get("input_eof_verified") is not True
        or type(aggregate) is not dict
        or envelope.get("c13_aggregate_commitment_sha256")
        != sha256_bytes(canonical_bytes(aggregate))
    ):
        raise RunnerFailure("WORKER_RESULT", "WORKER_ENVELOPE_MISMATCH")
    if envelope.get("batch_commitment_sha256") != expected_descriptor_commitment:
        raise RunnerFailure(
            "WORKER_PROTOCOL", "WORKER_BATCH_COMMITMENT_MISMATCH"
        )
    validate_worker_aggregate(aggregate)
    aggregate_bytes = canonical_bytes(aggregate)
    return WorkerOutcome(
        MappingProxyType(dict(aggregate)),
        aggregate_bytes,
        str(envelope["batch_commitment_sha256"]),
        _require_sha(
            pool_inventory_commitment_sha256,
            name="pool inventory commitment",
        ),
        child_user + child_system,
        int(child_maxrss),
    )


def validate_post_worker_resource_envelope(
    context: BootstrapContext,
    baseline: ResourceBaseline,
    outcome: WorkerOutcome,
) -> None:
    if type(outcome) is not WorkerOutcome:
        raise RunnerFailure("RESOURCE_MONITOR", "WORKER_RESOURCE_RECEIPT_TYPE")
    validate_authorized_resource_envelope(
        context,
        baseline,
        required_reaped_child_cpu_seconds=outcome.child_cpu_seconds,
        cumulative_child_maxrss_bytes=outcome.child_maxrss_bytes,
    )


SUCCESS_RESULT_KEYS = frozenset(
    {
        "adapter_or_hypothesis_semantics_modified",
        "candidate13_aggregate_check_code",
        "candidate13_aggregate_status",
        "check_code",
        "descriptor_commitment_sha256",
        "diagnostic_complete",
        "engineering_attempt_id",
        "exact_c9_success_failure_equivalent_for_all",
        "holdout_label_evaluator_metric_fit_full28_opened_or_started",
        "input_or_projection_bytes_published",
        "kind",
        "performance_tested",
        "pool_inventory_commitment_sha256",
        "rule_selected",
        "scientific_hypothesis_id",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
        "worker_aggregate_commitment_sha256",
    }
)


def _success_result(outcome: WorkerOutcome) -> dict[str, Any]:
    aggregate = outcome.aggregate
    value = {
        "adapter_or_hypothesis_semantics_modified": False,
        "candidate13_aggregate_check_code": aggregate["check_code"],
        "candidate13_aggregate_status": aggregate["status"],
        "check_code": "C14_FULL_POOL_ADAPTER_AGGREGATE_COMPLETE",
        "descriptor_commitment_sha256": outcome.descriptor_commitment_sha256,
        "diagnostic_complete": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "exact_c9_success_failure_equivalent_for_all": aggregate["exact_c9_success_failure_equivalent_for_all"],
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "input_or_projection_bytes_published": False,
        "kind": RESULT_KIND,
        "performance_tested": False,
        "pool_inventory_commitment_sha256": outcome.pool_inventory_commitment_sha256,
        "rule_selected": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "stage_contract_commitment_sha256": aggregate["stage_contract_commitment_sha256"],
        "stage_passed_all_bitset": aggregate["stage_passed_all_bitset"],
        "stage_reached_all_bitset": aggregate["stage_reached_all_bitset"],
        "status": "PASS_ENGINEERING_PREFLIGHT_NONAUTHORIZING" if aggregate["stage_passed_all_bitset"] == "1" * 10 else "SOURCE_POOL_ADAPTER_INCOMPATIBLE_NONAUTHORIZING",
        "worker_aggregate_commitment_sha256": sha256_bytes(outcome.aggregate_bytes),
    }
    if set(value) != SUCCESS_RESULT_KEYS:
        raise RunnerFailure("PAYLOAD", "RESULT_SCHEMA_MISMATCH")
    return value


FAILURE_STAGES = frozenset(
    {
        "ANCHOR_CREATION",
        "ANCHOR_INTEGRITY",
        "CACHE_MANIFEST",
        "POOL_INVENTORY",
        "RESOURCE_MONITOR",
        "WORKER_EXECUTION",
        "WORKER_PROTOCOL",
        "WORKER_RESULT",
        "PAYLOAD",
        "UNEXPECTED_ENGINEERING_FAILURE",
    }
)


def _failure_terminal(
    *, stage: str, anchor_durable: bool, anchor_receipt_available: bool
) -> dict[str, Any]:
    bounded_stage = stage if stage in FAILURE_STAGES else "UNEXPECTED_ENGINEERING_FAILURE"
    return {
        "adapter_or_hypothesis_semantics_modified": False,
        "anchor_durable_entry_created_or_partial": bool(anchor_durable),
        "anchor_receipt_available": anchor_receipt_available,
        "check_code": "C14_ENGINEERING_PREFLIGHT_TERMINAL",
        "continuation_authorized": False,
        "diagnostic_complete": False,
        "engineering_failure_stage": bounded_stage,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "kind": TERMINAL_KIND,
        "performance_tested": False,
        "retry_authorized": False,
        "rule_selected": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "stage_passed_all_bitset": "0" * 10,
        "stage_reached_all_bitset": "0" * 10,
        "status": "TERMINAL_ENGINEERING_FAILURE",
    }


def _ledger(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    consumption: ConsumptionSeal | None,
    *,
    stage: str,
    diagnostic_complete: bool,
    manifest_bytes_read_completed: bool,
    manifest_contract_validated: bool,
    all_36_pool_bytes_read_completed: bool,
    pool_inventory_validated: bool,
    outcome: WorkerOutcome | None,
    anchor_durable_override: bool | None = None,
    pool_inventory_commitment_sha256: str | None = None,
    descriptor_commitment_sha256: str | None = None,
) -> dict[str, Any]:
    reader = authorization.reader
    publish_exact_read_commitments = pool_inventory_validated
    effective_pool_commitment = (
        outcome.pool_inventory_commitment_sha256
        if outcome is not None
        else pool_inventory_commitment_sha256
    )
    effective_descriptor_commitment = (
        outcome.descriptor_commitment_sha256
        if outcome is not None
        else descriptor_commitment_sha256
    )
    value = {
        "anchor_binding_sha256": None if consumption is None else consumption.sha256,
        "anchor_durable_entry_created_or_partial": (
            consumption is not None
            if anchor_durable_override is None
            else anchor_durable_override
        ),
        "anchor_receipt_available": consumption is not None,
        "authorization_sha256": authorization.authorization_sha256,
        "check_code": "C14_VALUE_FREE_READ_LEDGER",
        "combined_cpu_limit_seconds": "240.000",
        "descriptor_commitment_sha256": effective_descriptor_commitment,
        "descriptor_completed": effective_descriptor_commitment is not None,
        "diagnostic_complete": diagnostic_complete,
        "engineering_failure_stage": stage,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "kind": LEDGER_KIND,
        "all_36_pool_bytes_read_completed": all_36_pool_bytes_read_completed,
        "manifest_bytes_read_completed": manifest_bytes_read_completed,
        "manifest_contract_validated": manifest_contract_validated,
        "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
        "memory_enforcement_is_hard_rlimit_as": False,
        "memory_enforcement_kind": "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING",
        "pool_inventory_commitment_sha256": effective_pool_commitment,
        "pool_inventory_validated": pool_inventory_validated,
        "poll_interval_seconds": "0.010",
        "sampling_gap_terminal": True,
        "sampling_max_gap_seconds": "0.050",
        "qualification_sha256": authorization.qualification_sha256,
        "raw_fd_opened_commitment_sha256": _purpose_commitment("CANDIDATE14_FINAL_RAW_FD_OPENED_V1", reader.raw_fd_opened_purposes) if publish_exact_read_commitments else None,
        "read_inventory_commitment_sha256": _entry_commitment(reader.entries, domain="CANDIDATE14_FINAL_READ_INVENTORY_V1") if publish_exact_read_commitments else None,
        "read_started_commitment_sha256": _purpose_commitment("CANDIDATE14_FINAL_READ_STARTED_V1", reader.read_started_purposes) if publish_exact_read_commitments else None,
        "sampling_overshoot_between_polls_possible": True,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "COMPLETE_VALUE_FREE_LEDGER",
        "terminal_publication_reserved_outside_compute_resource_envelope": True,
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available": True,
        "verified_inode_commitment_sha256": _purpose_commitment("CANDIDATE14_FINAL_VERIFIED_INODE_V1", reader.verified_inode_purposes) if publish_exact_read_commitments else None,
        "worker_aggregate_commitment_sha256": None if outcome is None else sha256_bytes(outcome.aggregate_bytes),
        "wall_timeout_seconds": "300.000",
        "worker_cpu_hard_limit_seconds": CHILD_CPU_HARD_LIMIT_SECONDS,
        "worker_cpu_soft_limit_seconds": CHILD_CPU_SOFT_LIMIT_SECONDS,
    }
    return value


def _output_inventory_commitment(payloads: Mapping[str, bytes]) -> str:
    return sha256_bytes(canonical_bytes({"domain": "CANDIDATE14_PUBLIC_OUTPUT_INVENTORY_V1", "files": [{"filename": name, "sha256": sha256_bytes(data), "size_bytes": len(data)} for name, data in sorted(payloads.items())]}))


def _lock(context: BootstrapContext, authorization: AuthorizationSeal, consumption: ConsumptionSeal | None, payloads_without_lock: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "anchor_binding_sha256": None if consumption is None else consumption.sha256,
        "anchor_durable_entry_created_or_partial": consumption is not None,
        "anchor_receipt_available": consumption is not None,
        "authorization_binding": _binding(C14_AUTHORIZATION, authorization.authorization_sha256, authorization.authorization_size_bytes, kind=AUTHORIZATION_KIND),
        "check_code": "C14_PUBLICATION_LOCK",
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "kind": LOCK_KIND,
        "output_inventory_commitment_sha256": _output_inventory_commitment(payloads_without_lock),
        "output_namespace_commitment_sha256": _namespace_commitment("CANDIDATE14_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT),
        "payload_bindings": {name: {"sha256": sha256_bytes(data), "size_bytes": len(data)} for name, data in sorted(payloads_without_lock.items())},
        "qualification_binding": _binding(C14_QUALIFICATION, authorization.qualification_sha256, authorization.qualification_size_bytes, kind=QUALIFICATION_KIND),
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "LOCKED_NO_REPLACE",
    }


TERMINAL_KEYS = frozenset(
    {
        "adapter_or_hypothesis_semantics_modified",
        "anchor_durable_entry_created_or_partial",
        "anchor_receipt_available",
        "check_code",
        "continuation_authorized",
        "diagnostic_complete",
        "engineering_failure_stage",
        "engineering_attempt_id",
        "holdout_label_evaluator_metric_fit_full28_opened_or_started",
        "kind",
        "performance_tested",
        "retry_authorized",
        "rule_selected",
        "scientific_hypothesis_id",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)
LEDGER_KEYS = frozenset(
    {
        "all_36_pool_bytes_read_completed",
        "anchor_binding_sha256",
        "anchor_durable_entry_created_or_partial",
        "anchor_receipt_available",
        "authorization_sha256",
        "check_code",
        "combined_cpu_limit_seconds",
        "descriptor_commitment_sha256",
        "descriptor_completed",
        "diagnostic_complete",
        "engineering_failure_stage",
        "engineering_attempt_id",
        "holdout_label_evaluator_metric_fit_full28_opened_or_started",
        "kind",
        "manifest_bytes_read_completed",
        "manifest_contract_validated",
        "memory_ceiling_bytes",
        "memory_enforcement_is_hard_rlimit_as",
        "memory_enforcement_kind",
        "pool_inventory_commitment_sha256",
        "pool_inventory_validated",
        "poll_interval_seconds",
        "sampling_gap_terminal",
        "sampling_max_gap_seconds",
        "qualification_sha256",
        "raw_fd_opened_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_commitment_sha256",
        "sampling_overshoot_between_polls_possible",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "status",
        "terminal_publication_reserved_outside_compute_resource_envelope",
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available",
        "verified_inode_commitment_sha256",
        "worker_aggregate_commitment_sha256",
        "wall_timeout_seconds",
        "worker_cpu_hard_limit_seconds",
        "worker_cpu_soft_limit_seconds",
    }
)
LOCK_KEYS = frozenset(
    {
        "anchor_binding_sha256",
        "anchor_durable_entry_created_or_partial",
        "anchor_receipt_available",
        "authorization_binding",
        "check_code",
        "engineering_attempt_id",
        "kind",
        "output_inventory_commitment_sha256",
        "output_namespace_commitment_sha256",
        "payload_bindings",
        "qualification_binding",
        "schema_version",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "status",
    }
)


def _nullable_sha(value: Any) -> bool:
    return value is None or (
        type(value) is str
        and len(value) == 64
        and not any(character not in "0123456789abcdef" for character in value)
    )


def _validate_public_payloads(payloads: Mapping[str, bytes]) -> None:
    if type(payloads) is not dict:
        raise RunnerFailure("PAYLOAD", "PAYLOAD_MAPPING_TYPE")
    success = set(payloads) == {"aggregate_result.json", "read_ledger.json", "lock.json"}
    terminal = set(payloads) == {"terminal.json", "read_ledger.json", "lock.json"}
    if success is terminal:
        raise RunnerFailure("PAYLOAD", "PAYLOAD_INVENTORY_MISMATCH")
    if any(type(name) is not str or type(data) is not bytes for name, data in payloads.items()):
        raise RunnerFailure("PAYLOAD", "PAYLOAD_TYPE_MISMATCH")
    public = _decode(payloads["aggregate_result.json" if success else "terminal.json"], name="public payload")
    ledger = _decode(payloads["read_ledger.json"], name="read ledger")
    lock = _decode(payloads["lock.json"], name="lock")
    if set(ledger) != LEDGER_KEYS or set(lock) != LOCK_KEYS:
        raise RunnerFailure("PAYLOAD", "LEDGER_OR_LOCK_SCHEMA_MISMATCH")
    if success:
        if set(public) != SUCCESS_RESULT_KEYS or public.get("kind") != RESULT_KIND or public.get("diagnostic_complete") is not True or ledger.get("diagnostic_complete") is not True:
            raise RunnerFailure("PAYLOAD", "SUCCESS_PAYLOAD_INVARIANT")
    else:
        if set(public) != TERMINAL_KEYS or public.get("kind") != TERMINAL_KIND or public.get("diagnostic_complete") is not False or ledger.get("diagnostic_complete") is not False or public.get("engineering_failure_stage") not in FAILURE_STAGES:
            raise RunnerFailure("PAYLOAD", "TERMINAL_PAYLOAD_INVARIANT")
    if (
        ledger.get("kind") != LEDGER_KIND
        or ledger.get("check_code") != "C14_VALUE_FREE_READ_LEDGER"
        or ledger.get("status") != "COMPLETE_VALUE_FREE_LEDGER"
        or ledger.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or ledger.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or ledger.get("source_inventory_commitment_sha256")
        != lock.get("source_inventory_commitment_sha256")
        or ledger.get("holdout_label_evaluator_metric_fit_full28_opened_or_started")
        is not False
        or ledger.get("memory_ceiling_bytes") != MEMORY_CEILING_BYTES
        or type(ledger.get("memory_ceiling_bytes")) is not int
        or ledger.get("memory_enforcement_is_hard_rlimit_as") is not False
        or ledger.get("combined_cpu_limit_seconds") != "240.000"
        or ledger.get("memory_enforcement_kind")
        != "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING"
        or ledger.get("poll_interval_seconds") != "0.010"
        or ledger.get("sampling_gap_terminal") is not True
        or ledger.get("sampling_max_gap_seconds") != "0.050"
        or ledger.get("sampling_overshoot_between_polls_possible") is not True
        or ledger.get("terminal_publication_reserved_outside_compute_resource_envelope")
        is not True
        or ledger.get("terminal_publication_still_anchor_integrity_checked_when_receipt_available")
        is not True
        or ledger.get("wall_timeout_seconds") != "300.000"
        or ledger.get("worker_cpu_soft_limit_seconds")
        != CHILD_CPU_SOFT_LIMIT_SECONDS
        or type(ledger.get("worker_cpu_soft_limit_seconds")) is not int
        or ledger.get("worker_cpu_hard_limit_seconds")
        != CHILD_CPU_HARD_LIMIT_SECONDS
        or type(ledger.get("worker_cpu_hard_limit_seconds")) is not int
        or lock.get("kind") != LOCK_KIND
        or lock.get("check_code") != "C14_PUBLICATION_LOCK"
        or lock.get("status") != "LOCKED_NO_REPLACE"
        or lock.get("schema_version") != 1
        or type(lock.get("schema_version")) is not int
        or lock.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or lock.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or lock.get("output_namespace_commitment_sha256")
        != _namespace_commitment("CANDIDATE14_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT)
    ):
        raise RunnerFailure("PAYLOAD", "FIXED_PUBLIC_CONTRACT_MISMATCH")
    sha_fields = (
        "anchor_binding_sha256",
        "descriptor_commitment_sha256",
        "pool_inventory_commitment_sha256",
        "raw_fd_opened_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_commitment_sha256",
        "verified_inode_commitment_sha256",
        "worker_aggregate_commitment_sha256",
    )
    if any(not _nullable_sha(ledger.get(key)) for key in sha_fields) or any(
        type(ledger.get(key)) is not str or not _nullable_sha(ledger.get(key))
        for key in (
            "authorization_sha256",
            "qualification_sha256",
            "source_inventory_commitment_sha256",
        )
    ):
        raise RunnerFailure("PAYLOAD", "LEDGER_COMMITMENT_SCHEMA_MISMATCH")
    if (
        type(ledger.get("anchor_durable_entry_created_or_partial")) is not bool
        or type(ledger.get("anchor_receipt_available")) is not bool
        or ledger["anchor_receipt_available"]
        and not ledger["anchor_durable_entry_created_or_partial"]
        or (ledger["anchor_binding_sha256"] is not None)
        is not ledger["anchor_receipt_available"]
        or type(ledger.get("manifest_bytes_read_completed")) is not bool
        or type(ledger.get("manifest_contract_validated")) is not bool
        or type(ledger.get("all_36_pool_bytes_read_completed")) is not bool
        or type(ledger.get("pool_inventory_validated")) is not bool
        or type(ledger.get("descriptor_completed")) is not bool
        or ledger["manifest_contract_validated"]
        and not ledger["manifest_bytes_read_completed"]
        or ledger["all_36_pool_bytes_read_completed"]
        and not ledger["manifest_contract_validated"]
        or ledger["pool_inventory_validated"]
        and not ledger["all_36_pool_bytes_read_completed"]
        or ledger["pool_inventory_commitment_sha256"] is not None
        and not ledger["manifest_contract_validated"]
        or ledger["all_36_pool_bytes_read_completed"]
        and ledger["pool_inventory_commitment_sha256"] is None
        or ledger["pool_inventory_validated"]
        and ledger["pool_inventory_commitment_sha256"] is None
        or ledger["pool_inventory_validated"]
        and not all(
            type(ledger[key]) is str
            for key in (
                "raw_fd_opened_commitment_sha256",
                "read_inventory_commitment_sha256",
                "read_started_commitment_sha256",
                "verified_inode_commitment_sha256",
            )
        )
        or not ledger["pool_inventory_validated"]
        and any(
            ledger[key] is not None
            for key in (
                "raw_fd_opened_commitment_sha256",
                "read_inventory_commitment_sha256",
                "read_started_commitment_sha256",
                "verified_inode_commitment_sha256",
            )
        )
        or ledger["descriptor_completed"]
        and not ledger["pool_inventory_validated"]
        or (ledger["descriptor_commitment_sha256"] is not None)
        is not ledger["descriptor_completed"]
        or (ledger["worker_aggregate_commitment_sha256"] is not None)
        is not ledger["diagnostic_complete"]
    ):
        raise RunnerFailure("PAYLOAD", "LEDGER_STATE_INVARIANT")
    expected_authorization_binding = _binding(
        C14_AUTHORIZATION,
        ledger["authorization_sha256"],
        lock.get("authorization_binding", {}).get("size_bytes", -1)
        if type(lock.get("authorization_binding")) is dict
        else -1,
        kind=AUTHORIZATION_KIND,
    )
    expected_qualification_binding = _binding(
        C14_QUALIFICATION,
        ledger["qualification_sha256"],
        lock.get("qualification_binding", {}).get("size_bytes", -1)
        if type(lock.get("qualification_binding")) is dict
        else -1,
        kind=QUALIFICATION_KIND,
    )
    if (
        lock.get("authorization_binding") != expected_authorization_binding
        or lock.get("qualification_binding") != expected_qualification_binding
    ):
        raise RunnerFailure("PAYLOAD", "AUTHORITY_BINDING_MISMATCH")
    base_names = set(payloads) - {"lock.json"}
    expected_bindings = {
        name: {
            "sha256": sha256_bytes(payloads[name]),
            "size_bytes": len(payloads[name]),
        }
        for name in sorted(base_names)
    }
    base = {name: payloads[name] for name in base_names}
    if (
        lock.get("payload_bindings") != expected_bindings
        or lock.get("output_inventory_commitment_sha256")
        != _output_inventory_commitment(base)
        or lock.get("anchor_durable_entry_created_or_partial")
        is not ledger["anchor_durable_entry_created_or_partial"]
        or lock.get("anchor_receipt_available")
        is not ledger["anchor_receipt_available"]
        or lock.get("anchor_binding_sha256") != ledger["anchor_binding_sha256"]
    ):
        raise RunnerFailure("PAYLOAD", "LOCK_CROSS_BINDING_MISMATCH")
    if success:
        success_commitment_fields = (
            "anchor_binding_sha256",
            "descriptor_commitment_sha256",
            "pool_inventory_commitment_sha256",
            "worker_aggregate_commitment_sha256",
        )
        passed = public.get("stage_passed_all_bitset")
        if (
            type(passed) is not str
            or len(passed) != len(C13_STAGE_CONTRACT)
            or set(passed) - {"0", "1"}
        ):
            raise RunnerFailure("PAYLOAD", "SUCCESS_STAGE_SCHEMA_MISMATCH")
        expected_reached, expected_failed = _expected_reached_failed(passed)
        if expected_reached == "INVALID" or expected_failed == "INVALID":
            raise RunnerFailure("PAYLOAD", "SUCCESS_STAGE_SCHEMA_MISMATCH")
        exact_equivalence = passed[-1] == "1"
        all_pass = passed == "1" * len(C13_STAGE_CONTRACT)
        if not exact_equivalence:
            candidate13_check = "C13_LOCALIZER_EXACT_C9_OUTCOME_MISMATCH"
            candidate13_status = "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE"
        elif all_pass:
            candidate13_check = "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL"
            candidate13_status = "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
        else:
            candidate13_check = "C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION"
            candidate13_status = (
                "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING"
            )
        expected_public = {
            "adapter_or_hypothesis_semantics_modified": False,
            "candidate13_aggregate_check_code": candidate13_check,
            "candidate13_aggregate_status": candidate13_status,
            "check_code": "C14_FULL_POOL_ADAPTER_AGGREGATE_COMPLETE",
            "descriptor_commitment_sha256": ledger["descriptor_commitment_sha256"],
            "diagnostic_complete": True,
            "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
            "exact_c9_success_failure_equivalent_for_all": exact_equivalence,
            "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
            "input_or_projection_bytes_published": False,
            "kind": RESULT_KIND,
            "performance_tested": False,
            "pool_inventory_commitment_sha256": ledger[
                "pool_inventory_commitment_sha256"
            ],
            "rule_selected": False,
            "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
            "stage_contract_commitment_sha256": sha256_bytes(
                canonical_bytes(list(C13_STAGE_CONTRACT))
            ),
            "stage_passed_all_bitset": passed,
            "stage_reached_all_bitset": expected_reached,
            "status": (
                "PASS_ENGINEERING_PREFLIGHT_NONAUTHORIZING"
                if all_pass
                else "SOURCE_POOL_ADAPTER_INCOMPATIBLE_NONAUTHORIZING"
            ),
            "worker_aggregate_commitment_sha256": ledger[
                "worker_aggregate_commitment_sha256"
            ],
        }
        if (
            ledger.get("engineering_failure_stage") != "NONE"
            or ledger.get("anchor_durable_entry_created_or_partial") is not True
            or ledger.get("anchor_receipt_available") is not True
            or ledger.get("manifest_bytes_read_completed") is not True
            or ledger.get("manifest_contract_validated") is not True
            or ledger.get("all_36_pool_bytes_read_completed") is not True
            or ledger.get("pool_inventory_validated") is not True
            or ledger.get("descriptor_completed") is not True
            or any(
                type(ledger.get(key)) is not str
                or not _nullable_sha(ledger.get(key))
                for key in success_commitment_fields
            )
            or any(
                type(public.get(key)) is not str
                or not _nullable_sha(public.get(key))
                for key in (
                    "descriptor_commitment_sha256",
                    "pool_inventory_commitment_sha256",
                    "worker_aggregate_commitment_sha256",
                )
            )
            or public != expected_public
        ):
            raise RunnerFailure("PAYLOAD", "SUCCESS_CROSS_BINDING_MISMATCH")
    else:
        if (
            public
            != _failure_terminal(
                stage=ledger["engineering_failure_stage"],
                anchor_durable=ledger[
                    "anchor_durable_entry_created_or_partial"
                ],
                anchor_receipt_available=ledger["anchor_receipt_available"],
            )
        ):
            raise RunnerFailure("PAYLOAD", "TERMINAL_CROSS_BINDING_MISMATCH")
    forbidden_keys = {
        "case_id",
        "runtime_case_id",
        "ordinal",
        "record_count",
        "pool_path",
        "pool_sha256",
        "exception",
        "exception_text",
        "worker_pid",
        "measured_bytes",
        "elapsed_seconds",
    }
    def walk(value: Any) -> None:
        if type(value) is dict:
            if any(type(key) is not str or key in forbidden_keys for key in value):
                raise RunnerFailure("PAYLOAD", "SENSITIVE_PUBLIC_KEY")
            for child in value.values():
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
        elif type(value) not in {str, int, bool, type(None)}:
            raise RunnerFailure("PAYLOAD", "PUBLIC_VALUE_TYPE_MISMATCH")
    walk(public)
    walk(ledger)
    walk(lock)


def build_success_payloads(context: BootstrapContext, authorization: AuthorizationSeal, consumption: ConsumptionSeal, outcome: WorkerOutcome) -> dict[str, bytes]:
    result = canonical_bytes(_success_result(outcome))
    ledger = canonical_bytes(
        _ledger(
            context,
            authorization,
            consumption,
            stage="NONE",
            diagnostic_complete=True,
            manifest_bytes_read_completed=True,
            manifest_contract_validated=True,
            all_36_pool_bytes_read_completed=True,
            pool_inventory_validated=True,
            outcome=outcome,
        )
    )
    base = {"aggregate_result.json": result, "read_ledger.json": ledger}
    payloads = {**base, "lock.json": canonical_bytes(_lock(context, authorization, consumption, base))}
    _validate_public_payloads(payloads)
    return payloads


def build_failure_payloads(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    consumption: ConsumptionSeal | None,
    *,
    stage: str,
    anchor_durable: bool,
    manifest_bytes_read_completed: bool,
    manifest_contract_validated: bool,
    all_36_pool_bytes_read_completed: bool,
    pool_inventory_validated: bool,
    pool_inventory_commitment_sha256: str | None = None,
    descriptor_commitment_sha256: str | None = None,
) -> dict[str, bytes]:
    terminal = canonical_bytes(_failure_terminal(stage=stage, anchor_durable=anchor_durable, anchor_receipt_available=consumption is not None))
    ledger = canonical_bytes(
        _ledger(
            context,
            authorization,
            consumption,
            stage=stage
            if stage in FAILURE_STAGES
            else "UNEXPECTED_ENGINEERING_FAILURE",
            diagnostic_complete=False,
            manifest_bytes_read_completed=manifest_bytes_read_completed,
            manifest_contract_validated=manifest_contract_validated,
            all_36_pool_bytes_read_completed=all_36_pool_bytes_read_completed,
            pool_inventory_validated=pool_inventory_validated,
            outcome=None,
            anchor_durable_override=anchor_durable,
            pool_inventory_commitment_sha256=pool_inventory_commitment_sha256,
            descriptor_commitment_sha256=descriptor_commitment_sha256,
        )
    )
    base = {"terminal.json": terminal, "read_ledger.json": ledger}
    lock = _lock(context, authorization, consumption, base)
    lock["anchor_durable_entry_created_or_partial"] = anchor_durable
    lock["anchor_receipt_available"] = consumption is not None
    payloads = {**base, "lock.json": canonical_bytes(lock)}
    _validate_public_payloads(payloads)
    return payloads


def _rename_noreplace(source_name: str, destination_name: str, parent_fd: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameatx_np", None)
    if function is None:
        raise OSError(errno.ENOSYS, "renameatx_np unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(destination_name), 0x00000004) != 0:
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number))


@dataclass
class PublicationProgress:
    output_namespace_entry_created: bool = False
    output_verified_complete: bool = False
    cleanup_complete: bool = True
    output_inventory_commitment_sha256: str | None = None


def publish_atomically(
    payloads: Mapping[str, bytes],
    freshness: FreshOutputSeal,
    *,
    resource_check: Any = None,
    progress: PublicationProgress | None = None,
) -> str:
    progress = PublicationProgress() if progress is None else progress
    if type(progress) is not PublicationProgress:
        raise RunnerFailure("PAYLOAD", "PUBLICATION_PROGRESS_TYPE_MISMATCH")
    if resource_check is not None:
        if not callable(resource_check):
            raise RunnerFailure("PAYLOAD", "RESOURCE_CHECK_NOT_CALLABLE")
        resource_check()
    _validate_fresh_output(freshness, integrity_check=resource_check)
    _validate_public_payloads(payloads)
    if resource_check is not None:
        resource_check()
    expected_names = {"read_ledger.json", "lock.json"}
    if type(payloads) is not dict or set(payloads) not in (expected_names | {"aggregate_result.json"}, expected_names | {"terminal.json"}) or any(type(name) is not str or type(data) is not bytes or not name.endswith(".json") for name, data in payloads.items()):
        raise RunnerFailure("PAYLOAD", "PUBLICATION_PAYLOAD_SCHEMA_MISMATCH")
    pending_lock_name = ".candidate14-lock.pending"
    staged_payloads = {
        (pending_lock_name if name == "lock.json" else name): data
        for name, data in payloads.items()
    }
    output = _lexical(OUTPUT_ROOT, role="output")
    parent_fd, parent_identity = _open_directory(output.parent, role="output parent")
    stage_name = f".{output.name}.staging-{_output_inventory_commitment(payloads)[:24]}"
    stage_fd: int | None = None
    final_fd: int | None = None
    try:
        if parent_identity != freshness.parent_identity:
            raise PermissionError("Candidate14 output parent changed")
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        progress.cleanup_complete = False
        stage_fd = os.open(stage_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0), dir_fd=parent_fd)
        stage_identity = _identity(os.fstat(stage_fd))
        leaf_identities: dict[str, tuple[int, int]] = {}
        for name, data in sorted(staged_payloads.items()):
            if resource_check is not None:
                resource_check()
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400, dir_fd=stage_fd)
            try:
                offset = 0
                while offset < len(data):
                    if resource_check is not None:
                        resource_check()
                    written = os.write(descriptor, data[offset:])
                    if written <= 0:
                        raise OSError(errno.EIO, "publication write failed")
                    offset += written
                os.fsync(descriptor)
                if resource_check is not None:
                    resource_check()
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                if resource_check is not None:
                    resource_check()
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o400 or int(metadata.st_size) != len(data):
                    raise OSError(errno.EIO, "publication leaf verification failed")
                leaf_identities[name] = _identity(metadata)
            finally:
                leaf_active_failure = sys.exc_info()[0] is not None
                try:
                    os.close(descriptor)
                except BaseException:
                    progress.cleanup_complete = False
                    if not leaf_active_failure:
                        raise RunnerFailure(
                            "PAYLOAD", "PUBLICATION_STAGING_LEAF_CLOSE_FAILED"
                        ) from None
        os.fsync(stage_fd)
        os.fchmod(stage_fd, 0o700)
        os.fsync(stage_fd)
        if resource_check is not None:
            resource_check()
        stage_metadata = os.fstat(stage_fd)
        visible_directory = os.stat(
            stage_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(stage_metadata) != stage_identity
            or _identity(visible_directory) != stage_identity
            or not stat.S_ISDIR(stage_metadata.st_mode)
            or not stat.S_ISDIR(visible_directory.st_mode)
            or stat.S_IMODE(stage_metadata.st_mode) != 0o700
            or stat.S_IMODE(visible_directory.st_mode) != 0o700
            or stage_metadata.st_nlink <= 0
            or stage_metadata.st_nlink != visible_directory.st_nlink
        ):
            raise OSError(errno.EIO, "staged directory identity mismatch")
        for name, expected in sorted(staged_payloads.items()):
            if resource_check is not None:
                resource_check()
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=stage_fd)
            try:
                metadata = os.fstat(descriptor)
                chunks: list[bytes] = []
                remaining = len(expected)
                while remaining:
                    if resource_check is not None:
                        resource_check()
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        raise OSError(errno.EIO, "published leaf ended early")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if resource_check is not None:
                    resource_check()
                if os.read(descriptor, 1) != b"":
                    raise OSError(errno.EIO, "published leaf grew during readback")
                visible = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
                after = os.fstat(descriptor)
                if _identity(metadata) != leaf_identities[name] or _identity(after) != leaf_identities[name] or _identity(visible) != leaf_identities[name] or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o400 or int(metadata.st_size) != len(expected) or int(after.st_size) != len(expected) or b"".join(chunks) != expected:
                    raise OSError(errno.EIO, "staged leaf readback mismatch")
            finally:
                leaf_active_failure = sys.exc_info()[0] is not None
                try:
                    os.close(descriptor)
                except BaseException:
                    progress.cleanup_complete = False
                    if not leaf_active_failure:
                        raise RunnerFailure(
                            "PAYLOAD", "PUBLICATION_READBACK_LEAF_CLOSE_FAILED"
                        ) from None
            if resource_check is not None:
                resource_check()
        staged_names: set[str] = set()
        with os.scandir(stage_fd) as staged_entries:
            for staged_entry in staged_entries:
                if resource_check is not None:
                    resource_check()
                if len(staged_names) >= len(staged_payloads):
                    raise OSError(errno.E2BIG, "staged output inventory has extras")
                staged_name = staged_entry.name
                if type(staged_name) is not str or staged_name in staged_names:
                    raise OSError(errno.EIO, "staged output inventory is invalid")
                staged_names.add(staged_name)
        if staged_names != set(staged_payloads):
            raise OSError(errno.EIO, "staged output inventory mismatch")
        stage_after_inventory = os.fstat(stage_fd)
        if (
            _identity(stage_after_inventory) != stage_identity
            or not stat.S_ISDIR(stage_after_inventory.st_mode)
            or stat.S_IMODE(stage_after_inventory.st_mode) != 0o700
        ):
            raise OSError(errno.EIO, "staged directory changed during inventory")
        if _identity(os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)) != stage_identity or _identity(os.fstat(parent_fd)) != parent_identity:
            raise OSError(errno.EIO, "staged namespace changed")
        if resource_check is not None:
            resource_check()
        os.fsync(stage_fd)
        if resource_check is not None:
            resource_check()
        os.fsync(parent_fd)
        if resource_check is not None:
            resource_check()
        _rename_noreplace(stage_name, output.name, parent_fd)
        progress.output_namespace_entry_created = True
        final_fd = os.open(
            output.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        final_metadata = os.fstat(final_fd)
        visible_final = os.stat(
            output.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(final_metadata) != stage_identity
            or _identity(visible_final) != stage_identity
            or not stat.S_ISDIR(final_metadata.st_mode)
            or stat.S_IMODE(final_metadata.st_mode) != 0o700
        ):
            raise OSError(errno.EIO, "renamed staging directory identity mismatch")
        postrename_names: set[str] = set()
        with os.scandir(final_fd) as postrename_entries:
            for entry in postrename_entries:
                if len(postrename_names) >= len(staged_payloads):
                    raise OSError(errno.E2BIG, "renamed output inventory has extras")
                if type(entry.name) is not str or entry.name in postrename_names:
                    raise OSError(errno.EIO, "renamed output inventory is invalid")
                postrename_names.add(entry.name)
        if postrename_names != set(staged_payloads):
            raise OSError(errno.EIO, "renamed output inventory mismatch")
        for name, expected in sorted(staged_payloads.items()):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=final_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                remaining = len(expected)
                chunks: list[bytes] = []
                while remaining:
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        raise OSError(errno.EIO, "renamed output leaf ended early")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1) != b"":
                    raise OSError(errno.EIO, "renamed output leaf grew")
                visible = os.stat(name, dir_fd=final_fd, follow_symlinks=False)
                if (
                    _identity(metadata) != leaf_identities[name]
                    or _identity(visible) != leaf_identities[name]
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o400
                    or b"".join(chunks) != expected
                ):
                    raise OSError(errno.EIO, "renamed output leaf mismatch")
            finally:
                os.close(descriptor)
        final_before_commit = os.fstat(final_fd)
        visible_before_commit = os.stat(
            output.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(final_before_commit) != stage_identity
            or _identity(visible_before_commit) != stage_identity
            or _identity(os.fstat(parent_fd)) != parent_identity
            or stat.S_IMODE(final_before_commit.st_mode) != 0o700
        ):
            raise OSError(errno.EIO, "precommit output identity changed")
        final_precommit_names: set[str] = set()
        with os.scandir(final_fd) as final_precommit_entries:
            for entry in final_precommit_entries:
                if len(final_precommit_names) >= len(staged_payloads):
                    raise OSError(errno.E2BIG, "precommit output inventory has extras")
                if type(entry.name) is not str or entry.name in final_precommit_names:
                    raise OSError(errno.EIO, "precommit output inventory is invalid")
                final_precommit_names.add(entry.name)
        if final_precommit_names != set(staged_payloads):
            raise OSError(errno.EIO, "precommit output inventory changed")
        lexical_precommit_parent_fd, lexical_precommit_parent_identity = (
            _open_directory(output.parent, role="precommit lexical output parent")
        )
        try:
            if lexical_precommit_parent_identity != parent_identity:
                raise OSError(errno.EIO, "precommit lexical parent changed")
            lexical_precommit_output = os.stat(
                output.name,
                dir_fd=lexical_precommit_parent_fd,
                follow_symlinks=False,
            )
            if _identity(lexical_precommit_output) != stage_identity:
                raise OSError(errno.EIO, "precommit lexical output changed")
        finally:
            os.close(lexical_precommit_parent_fd)
        if resource_check is not None:
            resource_check()
        edge_names: set[str] = set()
        with os.scandir(final_fd) as edge_entries:
            for entry in edge_entries:
                if len(edge_names) >= len(staged_payloads):
                    raise OSError(errno.E2BIG, "commit-edge inventory has extras")
                if type(entry.name) is not str or entry.name in edge_names:
                    raise OSError(errno.EIO, "commit-edge inventory is invalid")
                edge_names.add(entry.name)
        if edge_names != set(staged_payloads):
            raise OSError(errno.EIO, "commit-edge inventory changed")
        edge_order = [
            *sorted(name for name in staged_payloads if name != pending_lock_name),
            pending_lock_name,
        ]
        for edge_name in edge_order:
            edge_expected = staged_payloads[edge_name]
            edge_fd = os.open(
                edge_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=final_fd,
            )
            edge_failure: BaseException | None = None
            try:
                edge_metadata = os.fstat(edge_fd)
                edge_remaining = len(edge_expected)
                edge_chunks: list[bytes] = []
                while edge_remaining:
                    chunk = os.read(edge_fd, min(65536, edge_remaining))
                    if not chunk:
                        raise OSError(errno.EIO, "commit-edge leaf ended early")
                    edge_chunks.append(chunk)
                    edge_remaining -= len(chunk)
                if os.read(edge_fd, 1) != b"":
                    raise OSError(errno.EIO, "commit-edge leaf grew")
                edge_visible = os.stat(
                    edge_name, dir_fd=final_fd, follow_symlinks=False
                )
                if (
                    _identity(edge_metadata) != leaf_identities[edge_name]
                    or _identity(edge_visible) != leaf_identities[edge_name]
                    or not stat.S_ISREG(edge_metadata.st_mode)
                    or edge_metadata.st_nlink != 1
                    or stat.S_IMODE(edge_metadata.st_mode) != 0o400
                    or int(edge_metadata.st_size) != len(edge_expected)
                    or b"".join(edge_chunks) != edge_expected
                ):
                    raise OSError(errno.EIO, "commit-edge leaf changed")
            except BaseException as error:
                edge_failure = error
            finally:
                try:
                    os.close(edge_fd)
                except BaseException as error:
                    if edge_failure is None:
                        edge_failure = error
            if edge_failure is not None:
                raise edge_failure
        _rename_noreplace(pending_lock_name, "lock.json", final_fd)
        lock_identity = leaf_identities.pop(pending_lock_name)
        leaf_identities["lock.json"] = lock_identity
        os.fchmod(final_fd, 0o500)
        final_metadata = os.fstat(final_fd)
        visible_final = os.stat(
            output.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(final_metadata) != stage_identity
            or _identity(visible_final) != stage_identity
            or stat.S_IMODE(final_metadata.st_mode) != 0o500
            or stat.S_IMODE(visible_final.st_mode) != 0o500
        ):
            raise OSError(errno.EIO, "committed output directory mismatch")
        lexical_parent_fd, lexical_parent_identity = _open_directory(
            output.parent, role="committed lexical output parent"
        )
        lexical_output_fd: int | None = None
        lexical_lock_fd: int | None = None
        lexical_failure: BaseException | None = None
        try:
            if lexical_parent_identity != parent_identity:
                raise OSError(errno.EIO, "committed lexical parent changed")
            lexical_output_fd = os.open(
                output.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0),
                dir_fd=lexical_parent_fd,
            )
            lexical_output_metadata = os.fstat(lexical_output_fd)
            if (
                _identity(lexical_output_metadata) != stage_identity
                or stat.S_IMODE(lexical_output_metadata.st_mode) != 0o500
            ):
                raise OSError(errno.EIO, "committed lexical output changed")
            lexical_lock_fd = os.open(
                "lock.json",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=lexical_output_fd,
            )
            lock_metadata = os.fstat(lexical_lock_fd)
            lock_expected = payloads["lock.json"]
            remaining = len(lock_expected)
            lock_chunks: list[bytes] = []
            while remaining:
                chunk = os.read(lexical_lock_fd, min(65536, remaining))
                if not chunk:
                    raise OSError(errno.EIO, "committed lexical lock ended early")
                lock_chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(lexical_lock_fd, 1) != b"":
                raise OSError(errno.EIO, "committed lexical lock grew")
            visible_lock = os.stat(
                "lock.json", dir_fd=lexical_output_fd, follow_symlinks=False
            )
            if (
                _identity(lock_metadata) != lock_identity
                or _identity(visible_lock) != lock_identity
                or not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_nlink != 1
                or stat.S_IMODE(lock_metadata.st_mode) != 0o400
                or b"".join(lock_chunks) != lock_expected
            ):
                raise OSError(errno.EIO, "committed lexical lock mismatch")
        except BaseException as error:
            lexical_failure = error
        finally:
            for descriptor in (
                lexical_lock_fd,
                lexical_output_fd,
                lexical_parent_fd,
            ):
                if descriptor is None:
                    continue
                try:
                    os.close(descriptor)
                except BaseException as error:
                    if lexical_failure is None:
                        lexical_failure = error
        if lexical_failure is not None:
            raise lexical_failure
        commitment = _output_inventory_commitment(payloads)
        progress.cleanup_complete = True
        progress.output_inventory_commitment_sha256 = commitment
        progress.output_verified_complete = True
        try:
            os.fsync(final_fd)
            os.fsync(parent_fd)
        except BaseException:
            progress.cleanup_complete = False
        return commitment
    finally:
        active_failure = sys.exc_info()[0] is not None
        close_failure = False
        if final_fd is not None:
            try:
                os.close(final_fd)
            except BaseException:
                close_failure = True
        if stage_fd is not None:
            try:
                os.close(stage_fd)
            except BaseException:
                close_failure = True
        try:
            os.close(parent_fd)
        except BaseException:
            close_failure = True
        if close_failure:
            progress.cleanup_complete = False
            if not active_failure and not progress.output_verified_complete:
                raise RunnerFailure("PAYLOAD", "PUBLICATION_CLOSE_FAILED")


def _public_run_summary(
    payloads: Mapping[str, bytes],
    output_commitment: str,
    publication_progress: PublicationProgress,
) -> dict[str, Any]:
    name = "aggregate_result.json" if "aggregate_result.json" in payloads else "terminal.json"
    value = _decode(payloads[name], name="published public payload")
    return {
        "check_code": value["check_code"],
        "diagnostic_complete": value["diagnostic_complete"],
        "engineering_failure_stage": value.get("engineering_failure_stage", "NONE"),
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "output_inventory_commitment_sha256": output_commitment,
        "publication_completed": True,
        "publication_cleanup_complete": publication_progress.cleanup_complete,
        "publication_namespace_entry_created": publication_progress.output_namespace_entry_created,
        "retry_authorized": False,
        "stage_passed_all_bitset": value.get("stage_passed_all_bitset", "0" * 10),
        "stage_reached_all_bitset": value.get("stage_reached_all_bitset", "0" * 10),
        "status": value["status"],
    }


def _publication_failure_summary(
    progress: PublicationProgress, *, stage: str = "PAYLOAD"
) -> dict[str, Any]:
    bounded_stage = stage if stage in FAILURE_STAGES else "PAYLOAD"
    return {
        "check_code": "C14_PUBLICATION_TERMINAL_AFTER_CONSUMPTION",
        "diagnostic_complete": False,
        "engineering_failure_stage": bounded_stage,
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "output_inventory_commitment_sha256": progress.output_inventory_commitment_sha256,
        "publication_cleanup_complete": progress.cleanup_complete,
        "publication_completed": progress.output_verified_complete,
        "publication_namespace_entry_created": progress.output_namespace_entry_created,
        "retry_authorized": False,
        "stage_passed_all_bitset": "0" * 10,
        "stage_reached_all_bitset": "0" * 10,
        "status": "TERMINAL_PUBLICATION_FAILURE_NO_RETRY",
    }


def run_one_shot(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
) -> dict[str, Any]:
    _validate_context(context)
    freshness = prevalidate_fresh_output()
    resource_baseline = begin_authorized_resource_envelope(context)
    authorization = authorize_preflight_once(
        context,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        output_freshness=freshness,
    )
    validate_authorized_resource_envelope(context, resource_baseline)
    progress = AnchorProgress()
    consumption: ConsumptionSeal | None = None
    read_progress = ReadProgress()
    pool_inventory_commitment: str | None = None
    descriptor_commitment: str | None = None
    outcome: WorkerOutcome | None = None
    anchor_integrity_failed = False
    try:
        consumption = consume_authorization(
            context,
            authorization,
            resource_check=lambda: validate_authorized_resource_envelope(
                context, resource_baseline
            ),
            progress=progress,
        )
    except AnchorCreationFailure:
        payloads = build_failure_payloads(
            context,
            authorization,
            None,
            stage="ANCHOR_CREATION",
            anchor_durable=True,
            manifest_bytes_read_completed=False,
            manifest_contract_validated=False,
            all_36_pool_bytes_read_completed=False,
            pool_inventory_validated=False,
        )
        try:
            validate_authorized_resource_envelope(context, resource_baseline)
        except BaseException:
            payloads = build_failure_payloads(
                context,
                authorization,
                None,
                stage="RESOURCE_MONITOR",
                anchor_durable=True,
                manifest_bytes_read_completed=False,
                manifest_contract_validated=False,
                all_36_pool_bytes_read_completed=False,
                pool_inventory_validated=False,
            )
        publication_progress = PublicationProgress()
        try:
            output_commitment = publish_atomically(
                payloads, freshness, progress=publication_progress
            )
        except RunnerFailure as error:
            return _publication_failure_summary(
                publication_progress, stage=error.stage
            )
        except BaseException:
            return _publication_failure_summary(publication_progress)
        return _public_run_summary(
            payloads, output_commitment, publication_progress
        )

    try:
        validate_authorized_resource_envelope(context, resource_baseline)
        vectors, pool_inventory_commitment = read_full_pool_inventory(
            context,
            authorization,
            consumption,
            progress=read_progress,
        )
        _prelude, descriptor_commitment = build_descriptor_prelude(vectors)
        validate_authorized_resource_envelope(context, resource_baseline)
        outcome = run_worker_supervised(
            context,
            vectors,
            pool_inventory_commitment_sha256=pool_inventory_commitment,
            resource_baseline=resource_baseline,
        )
        if outcome.descriptor_commitment_sha256 != descriptor_commitment:
            raise RunnerFailure(
                "WORKER_PROTOCOL", "WORKER_DESCRIPTOR_COMMITMENT_MISMATCH"
            )
        revalidate_anchor(context, consumption)
        validate_post_worker_resource_envelope(
            context, resource_baseline, outcome
        )
        # Retain the exact parent byte tuple until the worker is fully reaped.
        if len(vectors) != RECORD_COUNT or any(
            type(vector.pool_bytes) is not bytes for vector in vectors
        ):
            raise RunnerFailure("POOL_INVENTORY", "PARENT_BYTE_RETENTION_MISMATCH")
        payloads = build_success_payloads(
            context, authorization, consumption, outcome
        )
        validate_post_worker_resource_envelope(
            context, resource_baseline, outcome
        )
        revalidate_anchor(context, consumption)
    except RunnerFailure as error:
        failure_stage = error.stage
        failure_consumption: ConsumptionSeal | None = consumption
        try:
            revalidate_anchor(context, consumption)
        except BaseException:
            failure_stage = "ANCHOR_INTEGRITY"
            failure_consumption = None
            anchor_integrity_failed = True
        payloads = build_failure_payloads(
            context,
            authorization,
            failure_consumption,
            stage=failure_stage,
            anchor_durable=True,
            manifest_bytes_read_completed=read_progress.manifest_bytes_read_completed,
            manifest_contract_validated=read_progress.manifest_contract_validated,
            all_36_pool_bytes_read_completed=read_progress.all_36_pool_bytes_read_completed,
            pool_inventory_validated=read_progress.pool_inventory_validated,
            pool_inventory_commitment_sha256=(
                pool_inventory_commitment
                or read_progress.pool_inventory_commitment_sha256
            ),
            descriptor_commitment_sha256=descriptor_commitment,
        )
    except BaseException:
        failure_stage = "UNEXPECTED_ENGINEERING_FAILURE"
        failure_consumption = consumption
        try:
            revalidate_anchor(context, consumption)
        except BaseException:
            failure_stage = "ANCHOR_INTEGRITY"
            failure_consumption = None
            anchor_integrity_failed = True
        payloads = build_failure_payloads(
            context,
            authorization,
            failure_consumption,
            stage=failure_stage,
            anchor_durable=True,
            manifest_bytes_read_completed=read_progress.manifest_bytes_read_completed,
            manifest_contract_validated=read_progress.manifest_contract_validated,
            all_36_pool_bytes_read_completed=read_progress.all_36_pool_bytes_read_completed,
            pool_inventory_validated=read_progress.pool_inventory_validated,
            pool_inventory_commitment_sha256=(
                pool_inventory_commitment
                or read_progress.pool_inventory_commitment_sha256
            ),
            descriptor_commitment_sha256=descriptor_commitment,
        )
    try:
        if outcome is None:
            validate_authorized_resource_envelope(context, resource_baseline)
        else:
            validate_post_worker_resource_envelope(
                context, resource_baseline, outcome
            )
    except BaseException:
        payloads = build_failure_payloads(
            context,
            authorization,
            None if anchor_integrity_failed else consumption,
            stage="RESOURCE_MONITOR",
            anchor_durable=True,
            manifest_bytes_read_completed=read_progress.manifest_bytes_read_completed,
            manifest_contract_validated=read_progress.manifest_contract_validated,
            all_36_pool_bytes_read_completed=read_progress.all_36_pool_bytes_read_completed,
            pool_inventory_validated=read_progress.pool_inventory_validated,
            pool_inventory_commitment_sha256=(
                pool_inventory_commitment
                or read_progress.pool_inventory_commitment_sha256
            ),
            descriptor_commitment_sha256=descriptor_commitment,
        )
    try:
        revalidate_anchor(context, consumption)
    except BaseException:
        anchor_integrity_failed = True
        payloads = build_failure_payloads(
            context,
            authorization,
            None,
            stage="ANCHOR_INTEGRITY",
            anchor_durable=True,
            manifest_bytes_read_completed=read_progress.manifest_bytes_read_completed,
            manifest_contract_validated=read_progress.manifest_contract_validated,
            all_36_pool_bytes_read_completed=read_progress.all_36_pool_bytes_read_completed,
            pool_inventory_validated=read_progress.pool_inventory_validated,
            pool_inventory_commitment_sha256=(
                pool_inventory_commitment
                or read_progress.pool_inventory_commitment_sha256
            ),
            descriptor_commitment_sha256=descriptor_commitment,
        )
    is_success = "aggregate_result.json" in payloads
    if is_success:
        try:
            if outcome is None:
                raise RunnerFailure(
                    "RESOURCE_MONITOR", "WORKER_RESOURCE_RECEIPT_MISSING"
                )
            validate_post_worker_resource_envelope(
                context, resource_baseline, outcome
            )
        except RunnerFailure as error:
            payloads = build_failure_payloads(
                context,
                authorization,
                consumption,
                stage=error.stage,
                anchor_durable=True,
                manifest_bytes_read_completed=read_progress.manifest_bytes_read_completed,
                manifest_contract_validated=read_progress.manifest_contract_validated,
                all_36_pool_bytes_read_completed=read_progress.all_36_pool_bytes_read_completed,
                pool_inventory_validated=read_progress.pool_inventory_validated,
                pool_inventory_commitment_sha256=(
                    pool_inventory_commitment
                    or read_progress.pool_inventory_commitment_sha256
                ),
                descriptor_commitment_sha256=descriptor_commitment,
            )
            is_success = False
    def publication_integrity_check() -> None:
        if not anchor_integrity_failed and consumption is not None:
            try:
                revalidate_anchor(context, consumption)
            except BaseException:
                raise RunnerFailure(
                    "ANCHOR_INTEGRITY", "ANCHOR_CHANGED_DURING_PUBLICATION"
                ) from None
        if is_success:
            if outcome is None:
                raise RunnerFailure(
                    "RESOURCE_MONITOR", "WORKER_RESOURCE_RECEIPT_MISSING"
                )
            validate_post_worker_resource_envelope(
                context, resource_baseline, outcome
            )

    publication_progress = PublicationProgress()
    try:
        output_commitment = publish_atomically(
            payloads,
            freshness,
            resource_check=(
                publication_integrity_check
                if (is_success or (not anchor_integrity_failed and consumption is not None))
                else None
            ),
            progress=publication_progress,
        )
    except RunnerFailure as error:
        return _publication_failure_summary(
            publication_progress, stage=error.stage
        )
    except BaseException:
        return _publication_failure_summary(publication_progress)
    return _public_run_summary(payloads, output_commitment, publication_progress)


def _bootstrap_cli_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"Candidate14 bootstrap requires exactly one {flag}")
    value = arguments[positions[0] + 1]
    if type(value) is not str or value.startswith("--"):
        raise ValueError(f"Candidate14 bootstrap has an invalid {flag}")
    return value


def _dispatch_through_verified_runner(argv: Sequence[str] | None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not all(type(value) is str for value in arguments):
        raise TypeError("Candidate14 bootstrap arguments must be exact strings")
    expected_sha = _require_sha(
        _bootstrap_cli_value(arguments, "--runner-sha256"), name="runner SHA"
    )
    size_text = _bootstrap_cli_value(arguments, "--runner-size-bytes")
    if not size_text.isdecimal() or int(size_text) <= 0:
        raise ValueError("Candidate14 bootstrap runner size is invalid")
    expected_size = int(size_text)
    data, _identity_value = _read_regular_file_once(
        Path(__file__),
        role="Candidate14 executing runner bootstrap",
        expected_size=expected_size,
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise PermissionError("Candidate14 executing runner byte binding mismatch")
    private_name = f"_candidate14_verified_parent_{expected_sha}"
    if private_name in sys.modules:
        raise PermissionError("Candidate14 verified parent is already loaded")
    outer = sys.modules.get(__name__)
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate14 outer parent module is unavailable")
    namespace = object.__getattribute__(outer, "__dict__")
    if namespace.get("_C14_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate14 parent bootstrap is not reentrant")
    launch_token = object()
    binding = MappingProxyType(
        {
            "module_name": private_name,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    )
    module = types.ModuleType(private_name)
    module.__file__ = str(Path(__file__))
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_C14_VERIFIED_RUNNER_BINDING"] = binding
    module.__dict__["_C14_VERIFIED_RUNNER_LAUNCH_TOKEN"] = launch_token
    module.__dict__["_C14_VERIFIED_RUNNER_OUTER_MODULE"] = outer
    namespace["_C14_OUTER_LAUNCH_TOKEN"] = launch_token
    sys.modules[private_name] = module
    try:
        exec(
            compile(data, str(Path(__file__)), "exec", dont_inherit=True),
            module.__dict__,
        )
        private_main = object.__getattribute__(module, "__dict__").get("main")
        if (
            type(private_main) is not types.FunctionType
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != str(Path(__file__))
        ):
            raise PermissionError("Candidate14 private parent entry point mismatch")
        return int(private_main(arguments))
    finally:
        sys.modules.pop(private_name, None)
        if namespace.get("_C14_OUTER_LAUNCH_TOKEN") is launch_token:
            del namespace["_C14_OUTER_LAUNCH_TOKEN"]


@dataclass(frozen=True, slots=True)
class CliArguments:
    runner_sha256: str
    runner_size_bytes: int
    runner_tests_sha256: str
    runner_tests_size_bytes: int
    source_check_only: bool
    source_control_check_only: bool
    preflight_once: bool
    authorization_sha256: str | None
    authorization_size_bytes: int | None


def _parse_args(argv: Sequence[str]) -> CliArguments:
    if not all(type(value) is str for value in argv):
        raise ValueError("Candidate14 arguments must be exact strings")
    modes = [value for value in argv if value in {"--source-check-only", "--source-control-check-only", "--preflight-once"}]
    if len(modes) != 1:
        raise ValueError("Candidate14 requires exactly one explicit mode")
    allowed = {"--runner-sha256", "--runner-size-bytes", "--runner-tests-sha256", "--runner-tests-size-bytes", "--source-check-only", "--source-control-check-only", "--preflight-once", "--authorization-sha256", "--authorization-size-bytes"}
    if any(value.startswith("--") and value not in allowed for value in argv):
        raise ValueError("Candidate14 rejects unknown arguments")
    runner_sha = _require_sha(_bootstrap_cli_value(argv, "--runner-sha256"), name="runner SHA")
    tests_sha = _require_sha(_bootstrap_cli_value(argv, "--runner-tests-sha256"), name="runner tests SHA")
    size_text = _bootstrap_cli_value(argv, "--runner-size-bytes")
    tests_size_text = _bootstrap_cli_value(argv, "--runner-tests-size-bytes")
    if not size_text.isdecimal() or not tests_size_text.isdecimal() or int(size_text) <= 0 or int(tests_size_text) <= 0:
        raise ValueError("Candidate14 source sizes are invalid")
    if int(tests_size_text) > MAX_RUNNER_TEST_SOURCE_BYTES:
        raise ValueError("Candidate14 runner tests exceed their size bound")
    preflight = modes[0] == "--preflight-once"
    authorization_sha: str | None = None
    authorization_size: int | None = None
    if preflight:
        authorization_sha = _require_sha(_bootstrap_cli_value(argv, "--authorization-sha256"), name="authorization SHA")
        authorization_size_text = _bootstrap_cli_value(argv, "--authorization-size-bytes")
        if not authorization_size_text.isdecimal() or int(authorization_size_text) <= 0:
            raise ValueError("Candidate14 authorization size is invalid")
        authorization_size = int(authorization_size_text)
        if authorization_size > MAX_AUTHORIZATION_BYTES:
            raise ValueError("Candidate14 authorization exceeds its size bound")
        expected_length = 13
    else:
        if "--authorization-sha256" in argv or "--authorization-size-bytes" in argv:
            raise ValueError("Candidate14 authorization is valid only for one-shot mode")
        expected_length = 9
    if len(argv) != expected_length:
        raise ValueError("Candidate14 argument inventory mismatch")
    return CliArguments(runner_sha, int(size_text), tests_sha, int(tests_size_text), modes[0] == "--source-check-only", modes[0] == "--source-control-check-only", preflight, authorization_sha, authorization_size)


def main(argv: Sequence[str] | None = None) -> int:
    if _C14_VERIFIED_RUNNER_BINDING is None:
        value = {"check_code": "C14_RUNNER_REJECTED_OR_FAILED", "diagnostic_complete": False, "holdout_label_evaluator_metric_fit_full28_opened_or_started": False, "status": "FAIL_CLOSED"}
        os.write(1, canonical_bytes(value))
        return 1
    try:
        _validate_isolated_startup()
        _verified_runner_execution_binding()
        arguments = _parse_args(list(sys.argv[1:] if argv is None else argv))
        context = bootstrap_verified_sources(runner_sha256=arguments.runner_sha256, runner_size_bytes=arguments.runner_size_bytes, runner_tests_sha256=arguments.runner_tests_sha256, runner_tests_size_bytes=arguments.runner_tests_size_bytes)
        if arguments.source_check_only:
            value = source_check_result(context)
            exit_code = 0
        elif arguments.source_control_check_only:
            value = qualification_result(context)
            exit_code = 0 if value.get("status") == "SOURCE_CONTROL_QUALIFIED_REPEATABLE" else 1
        else:
            assert arguments.authorization_sha256 is not None and arguments.authorization_size_bytes is not None
            value = run_one_shot(context, authorization_sha256=arguments.authorization_sha256, authorization_size_bytes=arguments.authorization_size_bytes)
            exit_code = 0 if value.get("diagnostic_complete") is True else 1
    except BaseException:
        value = {"check_code": "C14_RUNNER_REJECTED_OR_FAILED", "diagnostic_complete": False, "holdout_label_evaluator_metric_fit_full28_opened_or_started": False, "status": "FAIL_CLOSED"}
        exit_code = 1
    os.write(1, canonical_bytes(value))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
