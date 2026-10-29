#!/usr/bin/env python3
"""Guarded Candidate13 TRAIN18 pool-adapter stage-localizer runner.

The repeatable modes authenticate only frozen source bytes or value-free
controls.  Only ``--preflight-once`` accepts an exact external authorization,
durably consumes it with a no-replace anchor before the first manifest open,
and then inventories and reads the exact attested 36 TRAIN pool leaves.  It
never authorizes or opens semantic-result leaves.  All 36 immutable pool byte
vectors are complete before the filesystem-blind Candidate13 context exists;
then exactly one opaque receipt is issued per leaf and exactly one aggregate
call consumes the exact built-in tuple of 36 receipts.  Published results are
aggregate engineering evidence only.  Labels,
evaluator outputs, metrics, fitting, models, full-28 inputs, holdout inputs, and
adapted-cache publication are outside this runner's capability.
"""

from __future__ import annotations

import argparse
import builtins
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import types
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
RUNNER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
C13_CORE_SOURCE = ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py"
C13_CORE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py"
C12_CORE_SOURCE = ROOT / "backend/f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py"
C12_CORE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py"
C11_CORE_SOURCE = ROOT / "backend/f1_train18_candidate11_full_replay_diagnostic_v1.py"
C11_RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py"
C9_ADAPTER_SOURCE = ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C9_ADAPTER_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C6_HYPOTHESIS_SOURCE = ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
BUILDER_SOURCE = ROOT / "scripts/build_f1_scan_only_candidate_pool.py"
CLUSTER_SOURCE = ROOT / "backend/f1_line_object_clustering.py"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
C11_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate11_terminal_disposition_v1.json"
C12_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate12_synthetic_contract_pass_disposition_v1.json"
CUSTODY_ROOT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
CACHE_ATTESTATION = CUSTODY_ROOT / "cache/train_v51_cache_manifest_attestation.json"
CACHE_MANIFEST = CUSTODY_ROOT / "cache/train_v51_cache_manifest.json"
CUSTODY_EVIDENCE_PATHS = {
    "access_evidence": CUSTODY_ROOT / "cache/access_evidence.json",
    "builder": ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py",
    "staged_validator_pass": CUSTODY_ROOT / "independent_validation_report.json",
    "validator": ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py",
}
ALLOWED_CACHE_ROOT = ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"

C13_QUALIFICATION = ROOT / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_qualification_v1.json"
C13_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_authorization_v1.json"
C13_ANCHOR = ROOT / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1.json"
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate13-pool-adapter-stage-localizer-v1"
FALLBACK_OUTPUT_ROOT = ROOT / (
    "output/research/"
    "f1-aug30-train18-candidate13-pool-adapter-stage-localizer-publication-fallback-v1"
)
ALLOWED_OUTPUT_ROOTS = (OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT)

SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 13
ENGINEERING_ATTEMPT_ID = "CANDIDATE13_POOL_ADAPTER_STAGE_LOCALIZER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RECORD_COUNT = 36
PAIR_COUNT = 18
POOL_FILE_COUNT = 36
WALL_TIMEOUT_SECONDS = 300
CPU_TIMER_SECONDS = 240
ADDRESS_SPACE_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
MAX_POOL_LEAF_BYTES = 64 * 1024 * 1024

QUALIFICATION_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_qualification_v1"
AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_authorization_v1"
ANCHOR_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_runner_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_runner_terminal_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_read_ledger_v1"
LOCK_KIND = "sherman_qc_f1_aug30_train18_candidate13_pool_adapter_stage_localizer_lock_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CUSTODY_VALIDATION_KIND = "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"

FIXED_SOURCE_BINDINGS = {
    "builder": (BUILDER_SOURCE, "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2", 57613),
    "c11_core": (C11_CORE_SOURCE, "42c5ded051dd3b57dd48e3039a433afe63a0467edc13d4bd3b65620a6a279359", 32967),
    "c11_runner": (C11_RUNNER_SOURCE, "d18c583e176778974a9c85ede90ab287f23f5606e1856634c9a6c783c5effdd2", 167000),
    "c12_core": (C12_CORE_SOURCE, "89712f14836e102ba8861b2068d464e98ccfdb670d3447201797373b9a2254f6", 42434),
    "c12_tests": (C12_CORE_TESTS_SOURCE, "a3cf4013384bb9ef0e3077bb00f3ce7028dc1d3e7d0333bb9ef3382ca0e41340", 16111),
    "c13_core": (C13_CORE_SOURCE, "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b", 65344),
    "c13_core_tests": (C13_CORE_TESTS_SOURCE, "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15", 43387),
    "c6_hypothesis": (C6_HYPOTHESIS_SOURCE, "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252", 53837),
    "c9_adapter": (C9_ADAPTER_SOURCE, "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096", 22354),
    "c9_adapter_tests": (C9_ADAPTER_TESTS_SOURCE, "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623", 16598),
    "clusterer": (CLUSTER_SOURCE, "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c", 6817),
}

FIXED_CONTROLS = {
    "SEALED_SPLIT": (SPLIT, "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185", 10706),
    "ORIGINAL_PREREGISTRATION": (ORIGINAL_PREREG, "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a", 2651),
    "TRAIN_ALLOWLIST": (ALLOWLIST, "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0", 2428),
    "CANDIDATE11_TERMINAL_DISPOSITION": (C11_DISPOSITION, "58bd4fd18b1f6f4259e051ac1360cf6b204b4c339f1ff6a93e20a0f7d3f05bab", 4150),
    "CANDIDATE12_SYNTHETIC_DISPOSITION": (C12_DISPOSITION, "64c51a9421c39a0b3bdb54e8f30701ae58324127ed4dbb828dde7a83add44f75", 3555),
    "TRAIN_CACHE_ATTESTATION": (CACHE_ATTESTATION, "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71", 1834),
}

CACHE_MANIFEST_SHA256 = "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022"
CACHE_MANIFEST_SIZE_BYTES = 19601
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
V51_SCAN_ONLY_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

FORBIDDEN_PRELOADED = frozenset(
    {
        "backend.f1_train18_candidate13_pool_adapter_stage_localizer_v1",
        "backend.f1_train18_candidate9_lossless_pool_alias_adapter_v1",
        "backend.f1_train18_candidate6_conservative_recall_recovery_v1",
        "scripts.build_f1_scan_only_candidate_pool",
        "backend.f1_line_object_clustering",
    }
)
FORBIDDEN_INPUT_BASENAMES = frozenset(
    {
        "complete_label_manifest.json",
        "development_execution_manifest.json",
        "bend_line_development_catalog_v2_20260830.json",
        "labels.json",
        "label_projection.json",
        "train_label_projection.json",
        "train_label_projection_attestation.json",
        "evaluator_output.json",
        "metrics.json",
        "semantic_result.json",
    }
)
FORBIDDEN_PURPOSE_TOKENS = frozenset(
    {"LABEL", "GOLD", "TRUTH", "EVALUATOR", "METRIC", "FIT", "MODEL", "HOLDOUT", "FULL28"}
)

_CONTEXT_SEAL = object()
_QUALIFICATION_SEAL = object()
_AUTHORIZATION_SEAL = object()
_CONSUMPTION_SEAL = object()
_INVENTORY_SEAL = object()
_ISSUED_CONTEXTS: dict[int, tuple[object, object]] = {}
_ISSUED_QUALIFICATIONS: dict[int, tuple[object, object]] = {}
_ISSUED_AUTHORIZATIONS: dict[int, tuple[object, object, str]] = {}
_ISSUED_CONSUMPTIONS: dict[int, tuple[object, object]] = {}
_IMPORT_HOOK_SEAL = (
    builtins.__import__,
    tuple(sys.meta_path),
    tuple(sys.path_hooks),
)

# A normal Python file executes before it can authenticate its own bytes.  The
# public entry point therefore uses a two-stage bootstrap: the initially loaded
# module reads and verifies the exact runner bytes supplied on the command line,
# compiles those bytes into a private module, and delegates to that module.  The
# injected objects below exist only in that private verified execution.  Direct
# imports/calls and post-import monkeypatches never receive them.
_C13_VERIFIED_RUNNER_BINDING = globals().get("_C13_VERIFIED_RUNNER_BINDING")
_C13_VERIFIED_RUNNER_LAUNCH_TOKEN = globals().get(
    "_C13_VERIFIED_RUNNER_LAUNCH_TOKEN"
)
_C13_VERIFIED_RUNNER_OUTER_MODULE = globals().get(
    "_C13_VERIFIED_RUNNER_OUTER_MODULE"
)
_C13_OUTER_LAUNCH_TOKEN = globals().get("_C13_OUTER_LAUNCH_TOKEN")


class RunnerFailure(RuntimeError):
    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


class AnchorCreationFailure(RuntimeError):
    pass


class WatchdogExpired(BaseException):
    pass


class CpuTimerExpired(BaseException):
    pass


class ResourceEnvelopeFailure(BaseException):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Candidate13 runner hashes immutable bytes only")
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _verified_runner_execution_binding() -> dict[str, Any]:
    binding = _C13_VERIFIED_RUNNER_BINDING
    module = sys.modules.get(__name__)
    outer_module = _C13_VERIFIED_RUNNER_OUTER_MODULE
    outer_namespace = (
        object.__getattribute__(outer_module, "__dict__")
        if type(outer_module) is types.ModuleType
        else {}
    )
    if (
        type(binding) is not types.MappingProxyType
        or _C13_VERIFIED_RUNNER_LAUNCH_TOKEN is None
        or type(outer_module) is not types.ModuleType
        or outer_namespace.get("_C13_OUTER_LAUNCH_TOKEN")
        is not _C13_VERIFIED_RUNNER_LAUNCH_TOKEN
        or not isinstance(module, types.ModuleType)
        or module is not sys.modules.get(str(binding.get("module_name", "")))
        or __name__ != binding.get("module_name")
        or __name__
        != f"_candidate13_verified_runner_{binding.get('sha256', '')}"
        or not isinstance(binding.get("sha256"), str)
        or len(str(binding.get("sha256"))) != 64
        or any(character not in "0123456789abcdef" for character in str(binding["sha256"]))
        or type(binding.get("size_bytes")) is not int
        or int(binding["size_bytes"]) <= 0
    ):
        raise PermissionError(
            "Candidate13 runner is not executing from verified private bytes"
        )
    return {
        "module_name": str(binding["module_name"]),
        "sha256": str(binding["sha256"]),
        "size_bytes": int(binding["size_bytes"]),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != data:
        raise ValueError(f"{name} must be one LF-canonical JSON object")
    return value


def _lexical(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} contains parent traversal")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    root = Path(os.path.abspath(os.fspath(ROOT)))
    if os.path.commonpath((os.fspath(root), os.fspath(candidate))) != os.fspath(root):
        raise ValueError(f"{role} leaves repository")
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


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlinks(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError("Candidate13 requires a regular O_NOFOLLOW/O_DIRECTORY filesystem")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory)
    opened = os.fstat(descriptor)
    visible = os.lstat(candidate)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _identity(opened) != _identity(before)
        or _identity(visible) != _identity(before)
    ):
        os.close(descriptor)
        raise RuntimeError(f"{role} changed during open")
    return descriptor, _identity(opened)


def _read_regular_file_once(
    path: Path,
    *,
    role: str,
    after_raw_fd_open: Any = None,
    after_verified_inode: Any = None,
    before_first_read: Any = None,
) -> tuple[bytes, tuple[int, int]]:
    candidate = _reject_symlinks(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{role} must be one non-hard-linked regular file")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        if _identity(os.lstat(candidate.parent)) != parent_identity:
            raise RuntimeError(f"{role} parent changed before leaf open")
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if after_raw_fd_open is not None:
            after_raw_fd_open()
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != _identity(before)
        ):
            raise ValueError(f"{role} changed during leaf open")
        if after_verified_inode is not None:
            after_verified_inode(identity)
        if before_first_read is not None:
            before_first_read(identity)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or _identity(after) != identity
            or _identity(visible) != identity
            or int(after.st_size) != len(data)
            or _identity(os.lstat(candidate.parent)) != parent_identity
        ):
            raise ValueError(f"{role} changed during read")
        return data, identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact_source(
    path: Path, *, role: str, sha256: str, size_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    opened = False
    verified = False
    started = False

    def raw() -> None:
        nonlocal opened
        opened = True

    def inode(_identity_value: tuple[int, int]) -> None:
        nonlocal verified
        if not opened:
            raise RuntimeError("Candidate13 source inode verified before fd open")
        verified = True

    def read_start(_identity_value: tuple[int, int]) -> None:
        nonlocal started
        if not verified:
            raise RuntimeError("Candidate13 source read started before inode verification")
        started = True

    data, identity = _read_regular_file_once(
        path,
        role=role,
        after_raw_fd_open=raw,
        after_verified_inode=inode,
        before_first_read=read_start,
    )
    expected_sha = _require_sha(sha256, name=f"{role} SHA")
    expected_size = _strict_int(size_bytes, name=f"{role} size")
    if (
        not opened
        or not verified
        or not started
        or expected_size <= 0
        or len(data) != expected_size
        or sha256_bytes(data) != expected_sha
    ):
        raise ValueError(f"{role} exact source byte binding mismatch")
    return data, {
        "device_id": identity[0],
        "inode": identity[1],
        "path": str(_lexical(path, role=role).relative_to(ROOT)),
        "purpose": role,
        "sha256": expected_sha,
        "size_bytes": len(data),
    }


def _reject_preloaded(paths: Sequence[Path]) -> None:
    if (
        builtins.__import__ is not _IMPORT_HOOK_SEAL[0]
        or tuple(sys.meta_path) != _IMPORT_HOOK_SEAL[1]
        or tuple(sys.path_hooks) != _IMPORT_HOOK_SEAL[2]
    ):
        raise RuntimeError("Candidate13 rejects changed import hooks")
    current = sys.modules.get(__name__)
    launch_token = _C13_VERIFIED_RUNNER_LAUNCH_TOKEN
    outer_module = _C13_VERIFIED_RUNNER_OUTER_MODULE
    targets = {path.resolve(strict=True) for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in paths}
    if any(name in sys.modules for name in FORBIDDEN_PRELOADED):
        raise RuntimeError("Candidate13 rejects a preloaded controlled module")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        if module is outer_module:
            namespace = object.__getattribute__(module, "__dict__")
            if namespace.get("_C13_OUTER_LAUNCH_TOKEN") is launch_token:
                continue
            raise RuntimeError("Candidate13 outer launch token changed")
        if not isinstance(module, types.ModuleType):
            # CPython itself installs non-module compatibility sentinels such
            # as ``typing.io``.  Do not interrogate arbitrary objects: they
            # cannot be a loaded module alias, and attribute access could have
            # attacker-controlled side effects.
            continue
        if type(module) is not types.ModuleType:
            # A ModuleType subclass can override attribute behavior.  Reject
            # it without reading any attribute or module dictionary; ordinary
            # CPython import machinery installs exact ModuleType instances.
            raise RuntimeError(
                f"Candidate13 rejects preloaded module subclass: {name}"
            )
        # Never invoke module-level __getattr__ while auditing preloads.  A
        # hostile alias could otherwise mutate the just-compiled private runner
        # as a side effect and spoof the outer-launch token.
        namespace = object.__getattribute__(module, "__dict__")
        loaded = namespace.get("__file__")
        if not isinstance(loaded, str) or not loaded:
            continue
        try:
            candidate = Path(loaded).resolve(strict=True)
            identity = _identity(os.stat(candidate, follow_symlinks=True))
        except OSError:
            continue
        if candidate in targets or identity in identities:
            raise RuntimeError(f"Candidate13 rejects preloaded source alias: {name}")


def _exec(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate13 private verified module is already loaded")
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


def _source_projection(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {name: dict(value) for name, value in sorted(bindings.items())}


def _function_snapshot(*modules: types.ModuleType) -> tuple[Any, ...]:
    snapshot: list[tuple[str, str, Any, Any]] = []
    for module in modules:
        for name, value in sorted(vars(module).items()):
            if isinstance(value, types.FunctionType):
                snapshot.append((module.__name__, name, value, value.__code__))
            elif isinstance(value, type) and getattr(value, "__module__", None) == module.__name__:
                for child_name, child in sorted(vars(value).items()):
                    if isinstance(child, types.FunctionType):
                        snapshot.append(
                            (f"{module.__name__}.{name}", child_name, child, child.__code__)
                        )
    return tuple(snapshot)


@dataclass(frozen=True)
class BootstrapContext:
    runner_module: types.ModuleType
    c13_core: types.ModuleType
    c9_adapter: types.ModuleType
    c6_hypothesis: types.ModuleType
    builder: types.ModuleType
    clusterer: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_entries: tuple[Mapping[str, Any], ...]
    source_commitment_sha256: str
    _controlled_callables: tuple[Any, ...]
    _function_snapshot: tuple[Any, ...]
    _constant_commitment_sha256: str
    _sentinel: object
    _issuance_token: object


def _c13_core_bindings() -> dict[str, dict[str, Any]]:
    source_names = (
        "builder",
        "c11_core",
        "c11_runner",
        "c12_core",
        "c12_tests",
        "c6_hypothesis",
        "c9_adapter",
        "c9_adapter_tests",
        "clusterer",
    )
    projection = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        for name, (path, sha256, size_bytes) in (
            (name, FIXED_SOURCE_BINDINGS[name]) for name in source_names
        )
    }
    for name, control_name in (
        ("c11_disposition", "CANDIDATE11_TERMINAL_DISPOSITION"),
        ("c12_disposition", "CANDIDATE12_SYNTHETIC_DISPOSITION"),
    ):
        path, sha256, size_bytes = FIXED_CONTROLS[control_name]
        projection[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return {name: projection[name] for name in sorted(projection)}


def _constant_commitment(context: BootstrapContext) -> str:
    runner_binding = _verified_runner_execution_binding()
    return sha256_bytes(
        canonical_bytes(
            {
                "executing_runner": runner_binding,
                "c13": {
                    "attempt_ordinal": context.c13_core.ATTEMPT_ORDINAL,
                    "engineering_attempt_id": context.c13_core.ENGINEERING_ATTEMPT_ID,
                    "result_kind": context.c13_core.RESULT_KIND,
                    "stage_contract": list(context.c13_core.STAGE_CONTRACT),
                    "result_keys": sorted(context.c13_core.RESULT_KEYS),
                    "fixed_bindings": _c13_core_bindings(),
                },
                "c9": {
                    "attempt_ordinal": context.c9_adapter.ATTEMPT_ORDINAL,
                    "scientific_hypothesis_id": context.c9_adapter.SCIENTIFIC_HYPOTHESIS_ID,
                    "representation_contract": dict(
                        context.c9_adapter.REPRESENTATION_CONTRACT
                    ),
                },
                "manifest": {
                    "path": str(
                        _lexical(CACHE_MANIFEST, role="manifest constant").relative_to(
                            ROOT
                        )
                    ),
                    "sha256": CACHE_MANIFEST_SHA256,
                    "size_bytes": CACHE_MANIFEST_SIZE_BYTES,
                },
                "ordered_allowed_output_namespaces": [
                    str(path.relative_to(ROOT)) for path in ALLOWED_OUTPUT_ROOTS
                ],
                "resource_envelope": {
                    "address_space_limit_bytes": ADDRESS_SPACE_LIMIT_BYTES,
                    "cpu_timer_seconds": CPU_TIMER_SECONDS,
                    "max_pool_leaf_bytes": MAX_POOL_LEAF_BYTES,
                    "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
                },
                "source_commitment_sha256": context.source_commitment_sha256,
            }
        )
    )


def _execute_builder_with_cluster_alias(
    *,
    builder_bytes: bytes,
    clusterer: types.ModuleType,
) -> types.ModuleType:
    canonical_name = "backend.f1_line_object_clustering"
    if canonical_name in sys.modules:
        raise RuntimeError("Candidate13 rejects a preloaded clusterer alias")
    sys.modules[canonical_name] = clusterer
    try:
        return _exec(
            "_candidate13_verified_builder",
            BUILDER_SOURCE,
            builder_bytes,
            package="scripts",
        )
    finally:
        if sys.modules.get(canonical_name) is clusterer:
            sys.modules.pop(canonical_name, None)


def bootstrap_verified_sources(
    *,
    runner_sha256: str,
    runner_size_bytes: int,
    runner_tests_sha256: str,
    runner_tests_size_bytes: int,
) -> BootstrapContext:
    runner_binding = _verified_runner_execution_binding()
    if (
        runner_binding["sha256"]
        != _require_sha(runner_sha256, name="Candidate13 runner SHA")
        or runner_binding["size_bytes"]
        != _strict_int(runner_size_bytes, name="Candidate13 runner size")
    ):
        raise PermissionError(
            "Candidate13 executing runner does not match the requested bytes"
        )
    specs = {
        **FIXED_SOURCE_BINDINGS,
        "runner": (
            RUNNER_SOURCE,
            _require_sha(runner_sha256, name="Candidate13 runner SHA"),
            _strict_int(runner_size_bytes, name="Candidate13 runner size"),
        ),
        "runner_tests": (
            RUNNER_TESTS_SOURCE,
            _require_sha(runner_tests_sha256, name="Candidate13 runner tests SHA"),
            _strict_int(runner_tests_size_bytes, name="Candidate13 runner tests size"),
        ),
    }
    if specs["runner"][2] <= 0 or specs["runner_tests"][2] <= 0:
        raise ValueError("Candidate13 source sizes must be positive")
    _reject_preloaded(tuple(path for path, _sha, _size in specs.values()))
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha256, size_bytes = specs[name]
        data, entry = _read_exact_source(
            path,
            role=f"CANDIDATE13_SOURCE_{name.upper()}",
            sha256=sha256,
            size_bytes=size_bytes,
        )
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate13 source inventory contains an inode alias")
        identities.add(identity)
        source_bytes[name] = data
        entries.append(entry)
        bindings[name] = {
            "path": entry["path"],
            "sha256": sha256,
            "size_bytes": size_bytes,
        }

    hypothesis = _exec(
        "_candidate13_verified_c6_hypothesis",
        C6_HYPOTHESIS_SOURCE,
        source_bytes["c6_hypothesis"],
        package="backend",
    )
    clusterer = _exec(
        "_candidate13_verified_clusterer",
        CLUSTER_SOURCE,
        source_bytes["clusterer"],
        package="backend",
    )
    builder = _execute_builder_with_cluster_alias(
        builder_bytes=source_bytes["builder"],
        clusterer=clusterer,
    )
    c9_adapter = _exec(
        "_candidate13_verified_c9_adapter",
        C9_ADAPTER_SOURCE,
        source_bytes["c9_adapter"],
        package="backend",
    )
    c13_core = _exec(
        "_candidate13_verified_pool_adapter_stage_localizer",
        C13_CORE_SOURCE,
        source_bytes["c13_core"],
        package="backend",
    )
    expected_core_bindings = _c13_core_bindings()
    if (
        c13_core.ATTEMPT_ORDINAL != ATTEMPT_ORDINAL
        or c13_core.ENGINEERING_ATTEMPT_ID != ENGINEERING_ATTEMPT_ID
        or c13_core.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or c13_core.FIXED_BINDINGS != expected_core_bindings
        or c9_adapter.ATTEMPT_ORDINAL != 9
        or hypothesis.ATTEMPT_ORDINAL != 6
    ):
        raise ValueError("Candidate13 controlled runtime identity mismatch")
    source_commitment = sha256_bytes(canonical_bytes(_source_projection(bindings)))
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in bindings.items()
        }
    )
    immutable_entries = tuple(
        types.MappingProxyType(dict(entry)) for entry in entries
    )
    issuance_token = object()
    runner_module = sys.modules.get(__name__)
    if type(runner_module) is not types.ModuleType:
        raise RuntimeError("Candidate13 verified runner module is unavailable")
    context = BootstrapContext(
        runner_module=runner_module,
        c13_core=c13_core,
        c9_adapter=c9_adapter,
        c6_hypothesis=hypothesis,
        builder=builder,
        clusterer=clusterer,
        source_bindings=immutable_bindings,
        source_entries=immutable_entries,
        source_commitment_sha256=source_commitment,
        _controlled_callables=(
            c13_core.make_verified_context,
            c13_core.localize_pool_adapter_contract,
            c13_core.aggregate_pool_adapter_receipts,
        ),
        _function_snapshot=_function_snapshot(
            runner_module,
            hypothesis,
            clusterer,
            builder,
            c9_adapter,
            c13_core,
        ),
        _constant_commitment_sha256="",
        _sentinel=_CONTEXT_SEAL,
        _issuance_token=issuance_token,
    )
    object.__setattr__(
        context, "_constant_commitment_sha256", _constant_commitment(context)
    )
    _ISSUED_CONTEXTS[id(context)] = (context, issuance_token)
    return context


def _validate_context(context: BootstrapContext) -> None:
    issued = _ISSUED_CONTEXTS.get(id(context))
    if (
        type(context) is not BootstrapContext
        or context._sentinel is not _CONTEXT_SEAL
        or issued is None
        or issued[0] is not context
        or issued[1] is not context._issuance_token
        or context.runner_module is not sys.modules.get(__name__)
        or _verified_runner_execution_binding()
        != {
            "module_name": context.runner_module.__name__,
            "sha256": str(context.source_bindings["runner"]["sha256"]),
            "size_bytes": int(context.source_bindings["runner"]["size_bytes"]),
        }
        or context.source_commitment_sha256
        != sha256_bytes(
            canonical_bytes(_source_projection(context.source_bindings))
        )
        or context._controlled_callables
        != (
            context.c13_core.make_verified_context,
            context.c13_core.localize_pool_adapter_contract,
            context.c13_core.aggregate_pool_adapter_receipts,
        )
        or context._function_snapshot
        != _function_snapshot(
            context.runner_module,
            context.c6_hypothesis,
            context.clusterer,
            context.builder,
            context.c9_adapter,
            context.c13_core,
        )
        or context._constant_commitment_sha256 != _constant_commitment(context)
    ):
        raise PermissionError("Candidate13 bootstrap context is not intact")


def source_check_result(context: BootstrapContext) -> dict[str, Any]:
    _validate_context(context)
    return {
        "artifact_or_control_input_opened": False,
        "check_code": "C13_EXACT_SOURCE_BYTES_VERIFIED",
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "scientific_hypothesis_consumed": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }

def _inside_cache_root(path: Path) -> bool:
    candidate = _lexical(path, role="TRAIN cache artifact")
    root = _lexical(ALLOWED_CACHE_ROOT, role="TRAIN cache root")
    return candidate == root or root in candidate.parents


def _static_read_paths() -> frozenset[Path]:
    return frozenset(
        {
            *(_lexical(path, role="fixed control") for path, _sha, _size in FIXED_CONTROLS.values()),
            *(_lexical(path, role="custody evidence") for path in CUSTODY_EVIDENCE_PATHS.values()),
            _lexical(C13_AUTHORIZATION, role="Candidate13 authorization"),
            _lexical(C13_QUALIFICATION, role="Candidate13 qualification"),
            _lexical(CACHE_MANIFEST, role="TRAIN cache manifest"),
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
    _sentinel: object
    _nonce: object


def _expected_pool_purposes() -> list[str]:
    return [
        f"CANDIDATE13_{pass_name.upper()}_{index:02d}_POOL"
        for index in range(PAIR_COUNT)
        for pass_name in ("primary", "replay")
    ]


@dataclass
class GuardedReader:
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)
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
        for entry in self.entries:
            path = _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
            identity = int(entry["device_id"]), int(entry["inode"])
            prior = self.identities.get(identity)
            if prior is not None and prior != path:
                raise ValueError("Candidate13 bootstrap ledger contains an inode alias")
            self.identities[identity] = path
        source_purposes = [str(entry["purpose"]) for entry in self.entries]
        if not self.raw_fd_opened_purposes:
            self.raw_fd_opened_purposes.extend(source_purposes)
        if not self.verified_inode_purposes:
            self.verified_inode_purposes.extend(source_purposes)
        if not self.read_started_purposes:
            self.read_started_purposes.extend(source_purposes)

    @classmethod
    def from_context(cls, context: BootstrapContext) -> "GuardedReader":
        _validate_context(context)
        return cls(entries=[dict(entry) for entry in context.source_entries])

    def _deny(self, *, purpose: str, reason: str) -> None:
        self.denied_attempts.append({"purpose": purpose, "reason": reason})
        raise PermissionError(f"Candidate13 read denied ({reason})")

    def install_artifact_inventory(
        self,
        specifications: Sequence[tuple[Path, str, str]],
    ) -> ArtifactInventory:
        if self.artifact_inventory is not None:
            self._deny(purpose="TRAIN_ARTIFACT_INVENTORY", reason="INVENTORY_ALREADY_INSTALLED")
        if (
            type(specifications) is not tuple
            or len(specifications) != POOL_FILE_COUNT
            or [item[1] for item in specifications] != _expected_pool_purposes()
            or any(type(item) is not tuple or len(item) != 3 for item in specifications)
        ):
            raise ValueError(
                "Candidate13 artifact inventory must be the exact ordered 36 pool leaves"
            )
        bindings: dict[Path, ArtifactBinding] = {}
        identities = set(self.identities)
        for path, purpose, expected_sha in specifications:
            candidate = _lexical(path, role=purpose)
            if candidate.name in FORBIDDEN_INPUT_BASENAMES:
                self._deny(purpose=purpose, reason="FORBIDDEN_BASENAME")
            if not _inside_cache_root(candidate) or candidate in bindings:
                raise ValueError("Candidate13 artifact inventory path is invalid or duplicated")
            if any(token in purpose.upper() for token in FORBIDDEN_PURPOSE_TOKENS):
                self._deny(purpose=purpose, reason="FORBIDDEN_PURPOSE")
            canonical = _reject_symlinks(candidate, role=purpose, require_exists=True)
            metadata = os.lstat(canonical)
            identity = _identity(metadata)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or int(metadata.st_size) > MAX_POOL_LEAF_BYTES
                or identity in identities
            ):
                raise ValueError("Candidate13 artifact metadata or inode inventory is invalid")
            identities.add(identity)
            bindings[candidate] = ArtifactBinding(
                path=candidate,
                purpose=purpose,
                sha256=_require_sha(expected_sha, name=f"{purpose} SHA"),
                size_bytes=int(metadata.st_size),
                identity=identity,
            )
        projection = [
            {
                "device_id": binding.identity[0],
                "inode": binding.identity[1],
                "path": str(path.relative_to(ROOT)),
                "purpose": binding.purpose,
                "sha256": binding.sha256,
                "size_bytes": binding.size_bytes,
            }
            for path, binding in sorted(bindings.items(), key=lambda item: item[1].purpose)
        ]
        nonce = object()
        inventory = ArtifactInventory(
            bindings=types.MappingProxyType(dict(bindings)),
            commitment_sha256=sha256_bytes(
                canonical_bytes(
                    {"domain": "CANDIDATE13_EXACT_36_POOL_INVENTORY_V1", "files": projection}
                )
            ),
            _reader=self,
            _sentinel=_INVENTORY_SEAL,
            _nonce=nonce,
        )
        self.artifact_inventory = inventory
        return inventory

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        sha256: str,
        size_bytes: int,
        artifact_inventory: ArtifactInventory | None = None,
    ) -> None:
        candidate = _lexical(path, role=purpose)
        upper = purpose.upper()
        lowered = tuple(part.lower() for part in candidate.parts)
        if candidate.name in FORBIDDEN_INPUT_BASENAMES:
            self._deny(purpose=purpose, reason="FORBIDDEN_BASENAME")
        if any(token in upper for token in FORBIDDEN_PURPOSE_TOKENS):
            self._deny(purpose=purpose, reason="FORBIDDEN_PURPOSE")
        if any(
            token in part
            for part in lowered
            for token in ("holdout", "full28", "full-28", "gold", "label", "truth")
        ):
            self._deny(purpose=purpose, reason="FORBIDDEN_PATH")
        expected_sha = _require_sha(sha256, name=f"{purpose} SHA")
        expected_size = _strict_int(size_bytes, name=f"{purpose} size")
        if expected_size < 0 or (
            expected_size == 0 and not _inside_cache_root(candidate)
        ):
            raise ValueError(f"{purpose} size binding is invalid")
        if _inside_cache_root(candidate):
            installed = self.artifact_inventory
            if (
                artifact_inventory is None
                or installed is None
                or artifact_inventory is not installed
                or artifact_inventory._reader is not self
                or artifact_inventory._sentinel is not _INVENTORY_SEAL
            ):
                self._deny(purpose=purpose, reason="EXACT_ARTIFACT_CAPABILITY_REQUIRED")
            inventory_binding = installed.bindings.get(candidate)
            if (
                inventory_binding is None
                or inventory_binding.purpose != purpose
                or inventory_binding.sha256 != expected_sha
                or inventory_binding.size_bytes != expected_size
            ):
                self._deny(purpose=purpose, reason="ARTIFACT_OUTSIDE_EXACT_INVENTORY")
        elif candidate not in _static_read_paths():
            self._deny(purpose=purpose, reason="PATH_OUTSIDE_STATIC_ALLOWLIST")
        canonical = _reject_symlinks(candidate, role=purpose, require_exists=True)
        metadata = os.lstat(canonical)
        identity = _identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != expected_size
        ):
            raise ValueError(f"{purpose} exact metadata binding mismatch")
        if _inside_cache_root(candidate):
            assert self.artifact_inventory is not None
            if self.artifact_inventory.bindings[candidate].identity != identity:
                raise ValueError(f"{purpose} inode changed after inventory attestation")
        prior_path = self.identities.get(identity)
        if prior_path is not None and prior_path != canonical:
            raise ValueError("Candidate13 input contains an inode alias")
        binding = {
            "identity": identity,
            "purpose": purpose,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
        prior = self.allowed.get(canonical)
        if prior is not None and prior != binding:
            raise ValueError("Candidate13 conflicting authorization")
        if canonical in self.consumed_paths:
            self._deny(purpose=purpose, reason="DUPLICATE_READ_FORBIDDEN")
        self.allowed[canonical] = binding
        self.identities[identity] = canonical

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        candidate = _lexical(path, role=purpose)
        binding = self.allowed.get(candidate)
        if binding is None or binding["purpose"] != purpose:
            self._deny(purpose=purpose, reason="NOT_EXACTLY_AUTHORIZED")
        if candidate in self.consumed_paths or candidate in self.raw_fd_opened_paths:
            self._deny(purpose=purpose, reason="DUPLICATE_READ_FORBIDDEN")

        def raw() -> None:
            self.raw_fd_opened_paths.add(candidate)
            self.raw_fd_opened_purposes.append(purpose)

        def inode(identity: tuple[int, int]) -> None:
            if identity != binding["identity"] or candidate not in self.raw_fd_opened_paths:
                raise ValueError(f"{purpose} inode evidence mismatch")
            self.verified_inode_paths.add(candidate)
            self.verified_inode_purposes.append(purpose)

        def read_start(identity: tuple[int, int]) -> None:
            if identity != binding["identity"] or candidate not in self.verified_inode_paths:
                raise ValueError(f"{purpose} read-start evidence mismatch")
            self.read_started_paths.add(candidate)
            self.read_started_purposes.append(purpose)

        data, identity = _read_regular_file_once(
            candidate,
            role=purpose,
            after_raw_fd_open=raw,
            after_verified_inode=inode,
            before_first_read=read_start,
        )
        if (
            identity != binding["identity"]
            or len(data) != binding["size_bytes"]
            or sha256_bytes(data) != binding["sha256"]
        ):
            raise ValueError(f"{purpose} exact byte binding mismatch")
        self.entries.append(
            {
                "device_id": identity[0],
                "inode": identity[1],
                "path": str(candidate.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": binding["sha256"],
                "size_bytes": len(data),
            }
        )
        self.consumed_paths.add(candidate)
        return data


def _read_bound(
    reader: GuardedReader,
    *,
    path: Path,
    purpose: str,
    sha256: str,
    size_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    reader.authorize(path, purpose=purpose, sha256=sha256, size_bytes=size_bytes)
    data = reader.read_bytes(path, purpose=purpose)
    return data, _decode(data, name=purpose)


def _binding(
    path: Path, sha256: str, size_bytes: int, *, kind: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(_lexical(path, role="binding").relative_to(ROOT)),
        "sha256": _require_sha(sha256, name="binding SHA"),
        "size_bytes": _strict_int(size_bytes, name="binding size"),
    }
    if kind is not None:
        value["kind"] = kind
    return value


def _entry_commitment(entries: Sequence[Mapping[str, Any]]) -> str:
    projection = [
        {
            "device_id": _strict_int(entry["device_id"], name="read device"),
            "inode": _strict_int(entry["inode"], name="read inode"),
            "path": str(entry["path"]),
            "purpose": str(entry["purpose"]),
            "sha256": _require_sha(entry["sha256"], name="read SHA"),
            "size_bytes": _strict_int(entry["size_bytes"], name="read size"),
        }
        for entry in entries
    ]
    return sha256_bytes(
        canonical_bytes(
            {"domain": "CANDIDATE13_EXACT_READ_INVENTORY_V1", "reads": projection}
        )
    )


def _purpose_commitment(domain: str, purposes: Sequence[str]) -> str:
    return sha256_bytes(canonical_bytes({"domain": domain, "purposes": list(purposes)}))


CACHE_ATTESTATION_KEYS = frozenset(
    {
        "access_evidence_binding",
        "blinded_complement_commitment_sha256",
        "builder_binding",
        "canonical_json",
        "case_count",
        "holdout_entry_count",
        "holdout_hash_count",
        "holdout_status_count",
        "independent_validation_passed",
        "kind",
        "manifest_binding",
        "optimizer_holdout_access_count",
        "record_count",
        "schema_version",
        "split_sha256",
        "staged_validator_pass_binding",
        "train_allowlist_sha256",
        "train_exact_match",
        "train_holdout_intersection_count",
        "train_set_commitment_sha256",
        "validator_binding",
    }
)


def _binding_from_attestation(
    value: Any,
    *,
    role: str,
    expected_path: Path,
    typed: bool,
    expected_kind: str | None = None,
) -> tuple[Path, str, int]:
    keys = {"path", "sha256", "size_bytes"}
    if typed:
        keys.add("kind")
    if not isinstance(value, dict) or set(value) != keys:
        raise RunnerFailure("SOURCE_CONTROL_QUALIFICATION", "BINDING_SCHEMA_MISMATCH")
    if typed and value.get("kind") != expected_kind:
        raise RunnerFailure("SOURCE_CONTROL_QUALIFICATION", "BINDING_KIND_MISMATCH")
    relative = Path(str(value.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise RunnerFailure("SOURCE_CONTROL_QUALIFICATION", "BINDING_PATH_INVALID")
    path = _lexical(ROOT / relative, role=role)
    if path != _lexical(expected_path, role=f"expected {role}"):
        raise RunnerFailure("SOURCE_CONTROL_QUALIFICATION", "BINDING_PATH_MISMATCH")
    size = _strict_int(value.get("size_bytes"), name=f"{role} size")
    if size <= 0:
        raise RunnerFailure("SOURCE_CONTROL_QUALIFICATION", "BINDING_SIZE_INVALID")
    return path, _require_sha(value.get("sha256"), name=f"{role} SHA"), size


def _expected_control_purposes() -> list[str]:
    return [
        *FIXED_CONTROLS,
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
    ]


@dataclass(frozen=True)
class ControlQualification:
    reader: GuardedReader
    train_ids: tuple[str, ...]
    scan_by_case: Mapping[str, str]
    train_commitment_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    candidate11_disposition_bytes: bytes
    candidate12_disposition_bytes: bytes
    control_binding_commitment_sha256: str
    control_read_inventory_commitment_sha256: str
    control_raw_fd_opened_commitment_sha256: str
    control_verified_inode_commitment_sha256: str
    control_read_started_commitment_sha256: str
    _context_token: object
    _sentinel: object
    _issuance_token: object


def _validate_predecessor_disposition(
    value: Mapping[str, Any],
    *,
    attempt_ordinal: int,
    expected_kind: str,
) -> None:
    safety_fields = (
        "label_or_gold_opened",
        "holdout_opened",
        "metric_or_evaluator_started",
        "model_or_fit_started",
        "scientific_hypothesis_performance_tested",
    )
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind") != expected_kind
        or type(value.get("attempt_ordinal")) is not int
        or value.get("attempt_ordinal") != attempt_ordinal
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or any(value.get(field) is not False for field in safety_fields if field in value)
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION",
            f"C{attempt_ordinal}_DISPOSITION_CONTRACT_MISMATCH",
        )

def _control_entries(reader: GuardedReader) -> list[dict[str, Any]]:
    expected = set(_expected_control_purposes())
    return [entry for entry in reader.entries if str(entry["purpose"]) in expected]


def _control_commitment(reader: GuardedReader) -> str:
    entries = _control_entries(reader)
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE13_REPEATABLE_SOURCE_CONTROLS_V1",
                "read_inventory_commitment_sha256": _entry_commitment(entries),
                "source_bindings": {
                    name: {
                        "path": str(_lexical(path, role=name).relative_to(ROOT)),
                        "sha256": sha,
                        "size_bytes": size,
                    }
                    for name, (path, sha, size) in sorted(FIXED_CONTROLS.items())
                },
            }
        )
    )


def qualify_source_controls(
    context: BootstrapContext,
    *,
    reader: GuardedReader | None = None,
    authority_prefix: bool = False,
) -> ControlQualification:
    _validate_context(context)
    if reader is None:
        reader = GuardedReader.from_context(context)
    source_purposes = [str(entry["purpose"]) for entry in context.source_entries]
    expected_prefix = list(source_purposes)
    if authority_prefix:
        expected_prefix.extend(
            ["CANDIDATE13_EXTERNAL_AUTHORIZATION", "CANDIDATE13_QUALIFICATION_RECORD"]
        )
    if [str(entry["purpose"]) for entry in reader.entries] != expected_prefix:
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "SOURCE_CONTROL_PREFIX_MISMATCH"
        )
    controls_bytes: dict[str, bytes] = {}
    controls: dict[str, dict[str, Any]] = {}
    for purpose, (path, sha, size) in FIXED_CONTROLS.items():
        data, value = _read_bound(
            reader, path=path, purpose=purpose, sha256=sha, size_bytes=size
        )
        controls_bytes[purpose] = data
        controls[purpose] = value
    _validate_predecessor_disposition(
        controls["CANDIDATE11_TERMINAL_DISPOSITION"],
        attempt_ordinal=11,
        expected_kind="sherman_qc_f1_aug30_train18_candidate11_terminal_disposition_v1",
    )
    _validate_predecessor_disposition(
        controls["CANDIDATE12_SYNTHETIC_DISPOSITION"],
        attempt_ordinal=12,
        expected_kind=(
            "sherman_qc_f1_aug30_train18_candidate12_"
            "synthetic_contract_pass_disposition_v1"
        ),
    )

    split = controls["SEALED_SPLIT"]
    preregistration = controls["ORIGINAL_PREREGISTRATION"]
    allowlist = controls["TRAIN_ALLOWLIST"]
    cases = allowlist.get("cases")
    if (
        split.get("kind")
        != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
        or split.get("label_or_outcome_fields_present") is not False
        or preregistration.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN"
        or allowlist.get("kind")
        != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
        or allowlist.get("reject_any_case_not_listed") is not True
        or not isinstance(cases, list)
        or len(cases) != PAIR_COUNT
        or any(not isinstance(row, dict) for row in cases)
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "SPLIT_ALLOWLIST_OR_PREREG_MISMATCH"
        )
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    if (
        len(set(train_ids)) != PAIR_COUNT
        or any(not case_id for case_id in train_ids)
        or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "TRAIN_MEMBERSHIP_ORDER_MISMATCH"
        )
    scan_by_case = {
        str(row["case_id"]): _require_sha(row.get("scan_sha256"), name="TRAIN scan SHA")
        for row in cases
    }
    train_commitment = sha256_bytes(
        canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    attestation = controls["TRAIN_CACHE_ATTESTATION"]
    if (
        set(attestation) != CACHE_ATTESTATION_KEYS
        or type(attestation.get("schema_version")) is not int
        or attestation.get("schema_version") != 1
        or attestation.get("kind") != CACHE_ATTESTATION_KIND
        or attestation.get("canonical_json") is not True
        or attestation.get("independent_validation_passed") is not True
        or type(attestation.get("case_count")) is not int
        or attestation.get("case_count") != PAIR_COUNT
        or type(attestation.get("record_count")) is not int
        or attestation.get("record_count") != RECORD_COUNT
        or any(
            type(attestation.get(key)) is not int or attestation.get(key) != 0
            for key in (
                "holdout_entry_count",
                "holdout_hash_count",
                "holdout_status_count",
                "optimizer_holdout_access_count",
                "train_holdout_intersection_count",
            )
        )
        or attestation.get("train_exact_match") is not True
        or attestation.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1]
        or attestation.get("train_allowlist_sha256")
        != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1]
        or attestation.get("train_set_commitment_sha256") != train_commitment
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "CACHE_ATTESTATION_BOUNDARY_MISMATCH"
        )
    _require_sha(
        attestation.get("blinded_complement_commitment_sha256"),
        name="blinded complement commitment",
    )
    for prefix in ("access_evidence", "builder", "staged_validator_pass", "validator"):
        typed = prefix == "staged_validator_pass"
        path, sha, size = _binding_from_attestation(
            attestation.get(f"{prefix}_binding"),
            role=f"CACHE_CUSTODY_{prefix.upper()}",
            expected_path=CUSTODY_EVIDENCE_PATHS[prefix],
            typed=typed,
            expected_kind=CUSTODY_VALIDATION_KIND if typed else None,
        )
        reader.authorize(
            path,
            purpose=f"CACHE_CUSTODY_{prefix.upper()}",
            sha256=sha,
            size_bytes=size,
        )
        reader.read_bytes(path, purpose=f"CACHE_CUSTODY_{prefix.upper()}")
    if (
        (attestation.get("staged_validator_pass_binding") or {}).get("sha256")
        != CUSTODY_PASS_SHA256
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "CUSTODY_PASS_SHA_MISMATCH"
        )
    manifest_path, manifest_sha, manifest_size = _binding_from_attestation(
        attestation.get("manifest_binding"),
        role="TRAIN_CACHE_MANIFEST",
        expected_path=CACHE_MANIFEST,
        typed=True,
        expected_kind=CACHE_MANIFEST_KIND,
    )
    if (
        manifest_sha != CACHE_MANIFEST_SHA256
        or manifest_size != CACHE_MANIFEST_SIZE_BYTES
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "CACHE_MANIFEST_BINDING_MISMATCH"
        )
    expected_tail = _expected_control_purposes()
    observed_tail = [
        str(entry["purpose"]) for entry in reader.entries[len(expected_prefix) :]
    ]
    full_expected = expected_prefix + expected_tail
    paths = [str(entry["path"]) for entry in reader.entries]
    identities = [
        (int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries
    ]
    if (
        observed_tail != expected_tail
        or reader.raw_fd_opened_purposes != full_expected
        or reader.verified_inode_purposes != full_expected
        or reader.read_started_purposes != full_expected
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or reader.denied_attempts
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "SOURCE_CONTROL_READ_INVENTORY_MISMATCH"
        )
    control_entries = _control_entries(reader)
    control_purposes = [str(entry["purpose"]) for entry in control_entries]
    issuance_token = object()
    qualification = ControlQualification(
        reader=reader,
        train_ids=train_ids,
        scan_by_case=types.MappingProxyType(dict(scan_by_case)),
        train_commitment_sha256=train_commitment,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest_size_bytes=manifest_size,
        candidate11_disposition_bytes=controls_bytes[
            "CANDIDATE11_TERMINAL_DISPOSITION"
        ],
        candidate12_disposition_bytes=controls_bytes[
            "CANDIDATE12_SYNTHETIC_DISPOSITION"
        ],
        control_binding_commitment_sha256=_control_commitment(reader),
        control_read_inventory_commitment_sha256=_entry_commitment(control_entries),
        control_raw_fd_opened_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_CONTROL_RAW_FD_OPENED_V1", control_purposes
        ),
        control_verified_inode_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_CONTROL_VERIFIED_INODES_V1", control_purposes
        ),
        control_read_started_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_CONTROL_READ_STARTED_V1", control_purposes
        ),
        _context_token=context._issuance_token,
        _sentinel=_QUALIFICATION_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_QUALIFICATIONS[id(qualification)] = (qualification, issuance_token)
    return qualification


def _validate_qualification(
    context: BootstrapContext, qualification: ControlQualification
) -> None:
    issued = _ISSUED_QUALIFICATIONS.get(id(qualification))
    control_entries = _control_entries(qualification.reader)
    control_purposes = [str(entry["purpose"]) for entry in control_entries]
    if (
        type(qualification) is not ControlQualification
        or qualification._sentinel is not _QUALIFICATION_SEAL
        or qualification._context_token is not context._issuance_token
        or issued is None
        or issued[0] is not qualification
        or issued[1] is not qualification._issuance_token
        or sha256_bytes(qualification.candidate11_disposition_bytes)
        != FIXED_CONTROLS["CANDIDATE11_TERMINAL_DISPOSITION"][1]
        or sha256_bytes(qualification.candidate12_disposition_bytes)
        != FIXED_CONTROLS["CANDIDATE12_SYNTHETIC_DISPOSITION"][1]
        or qualification.control_binding_commitment_sha256
        != _control_commitment(qualification.reader)
        or qualification.control_read_inventory_commitment_sha256
        != _entry_commitment(control_entries)
        or qualification.control_raw_fd_opened_commitment_sha256
        != _purpose_commitment("CANDIDATE13_CONTROL_RAW_FD_OPENED_V1", control_purposes)
        or qualification.control_verified_inode_commitment_sha256
        != _purpose_commitment("CANDIDATE13_CONTROL_VERIFIED_INODES_V1", control_purposes)
        or qualification.control_read_started_commitment_sha256
        != _purpose_commitment("CANDIDATE13_CONTROL_READ_STARTED_V1", control_purposes)
        or qualification.manifest_path != _lexical(CACHE_MANIFEST, role="manifest")
        or qualification.manifest_sha256 != CACHE_MANIFEST_SHA256
        or qualification.manifest_size_bytes != CACHE_MANIFEST_SIZE_BYTES
    ):
        raise PermissionError("Candidate13 source-control qualification is not intact")


def _validate_preanchor_reader_state(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
) -> None:
    reader = qualification.reader
    expected_purposes = [str(entry["purpose"]) for entry in context.source_entries]
    expected_purposes.extend(
        ["CANDIDATE13_EXTERNAL_AUTHORIZATION", "CANDIDATE13_QUALIFICATION_RECORD"]
    )
    expected_purposes.extend(_expected_control_purposes())
    observed_purposes = [str(entry["purpose"]) for entry in reader.entries]
    non_source_entries = reader.entries[len(context.source_entries) :]
    expected_consumed_paths = {
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in non_source_entries
    }
    if (
        qualification.reader is not authorization.reader
        or observed_purposes != expected_purposes
        or reader.raw_fd_opened_purposes != expected_purposes
        or reader.verified_inode_purposes != expected_purposes
        or reader.read_started_purposes != expected_purposes
        or reader.artifact_inventory is not None
        or reader.denied_attempts
        or set(reader.allowed) != expected_consumed_paths
        or reader.consumed_paths != expected_consumed_paths
        or _lexical(CACHE_MANIFEST, role="pre-anchor manifest") in reader.allowed
        or any(purpose == "TRAIN_CACHE_MANIFEST" for purpose in observed_purposes)
        or any(purpose.startswith("TRAIN_RECORD_") for purpose in observed_purposes)
        or any(
            purpose.startswith("CANDIDATE13_") and purpose.endswith("_POOL")
            for purpose in observed_purposes
        )
    ):
        raise PermissionError("Candidate13 pre-anchor reader state is not exact")


def _namespace_commitment(domain: str, path: Path) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": domain,
                "namespace": str(
                    _lexical(path, role="committed namespace").relative_to(ROOT)
                ),
            }
        )
    )


def _allowed_output_namespace_set_commitment() -> str:
    if (
        type(ALLOWED_OUTPUT_ROOTS) is not tuple
        or ALLOWED_OUTPUT_ROOTS != (OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT)
        or len(set(ALLOWED_OUTPUT_ROOTS)) != 2
    ):
        raise ValueError("Candidate13 allowed output namespace set changed")
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE13_ORDERED_ALLOWED_OUTPUT_NAMESPACES_V1",
                "namespaces": [
                    str(
                        _lexical(path, role="allowed output namespace").relative_to(
                            ROOT
                        )
                    )
                    for path in ALLOWED_OUTPUT_ROOTS
                ],
            }
        )
    )


def _validate_allowed_output_namespace_binding(
    value: Mapping[str, Any], *, role: str
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("allowed_output_namespace_set_commitment_sha256")
        != _allowed_output_namespace_set_commitment()
    ):
        raise PermissionError(
            f"Candidate13 {role} does not bind the ordered output namespaces"
        )


def _manifest_contract_commitment(path: Path, sha256: str, size_bytes: int) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE13_FIXED_MANIFEST_CONTRACT_V1",
                "kind": CACHE_MANIFEST_KIND,
                "path": str(_lexical(path, role="committed manifest").relative_to(ROOT)),
                "sha256": _require_sha(sha256, name="committed manifest SHA"),
                "size_bytes": _strict_int(size_bytes, name="committed manifest size"),
            }
        )
    )


def build_qualification_record(
    context: BootstrapContext, qualification: ControlQualification
) -> bytes:
    _validate_context(context)
    _validate_qualification(context, qualification)
    return canonical_bytes(
        {
            "adapted_cache_publication_authorized": False,
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED",
            "consumption_anchor_namespace_commitment_sha256": _namespace_commitment(
                "CANDIDATE13_CONSUMPTION_ANCHOR_NAMESPACE_V1", C13_ANCHOR
            ),
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "control_raw_fd_opened_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
            "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
            "control_read_started_commitment_sha256": qualification.control_read_started_commitment_sha256,
            "control_verified_inode_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
            "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "kind": QUALIFICATION_KIND,
            "manifest_binding_commitment_sha256": _manifest_contract_commitment(
                qualification.manifest_path,
                qualification.manifest_sha256,
                qualification.manifest_size_bytes,
            ),
            "allowed_output_namespace_set_commitment_sha256": (
                _allowed_output_namespace_set_commitment()
            ),
            "repeatable": True,
            "scientific_hypothesis_consumed": False,
            "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
            "source_inventory_commitment_sha256": context.source_commitment_sha256,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        }
    )


def qualification_result(context: BootstrapContext) -> dict[str, Any]:
    try:
        qualification = qualify_source_controls(context)
        record = build_qualification_record(context, qualification)
        return {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED",
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "failure_stage": "NONE",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "qualification_record_commitment_sha256": sha256_bytes(record),
            "repeatable": True,
            "source_inventory_commitment_sha256": context.source_commitment_sha256,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        }
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        return {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "C13_SOURCE_CONTROL_QUALIFICATION_FAILED",
            "failure_stage": "SOURCE_CONTROL_QUALIFICATION",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "repeatable": True,
            "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
        }


@dataclass(frozen=True)
class FreshOutputSeal:
    path: Path
    parent_identity: tuple[int, int]
    _sentinel: object


def _validate_output_namespace(path: Path | None = None) -> Path:
    candidate = _lexical(
        OUTPUT_ROOT if path is None else path,
        role="Candidate13 output namespace",
    )
    frozen = tuple(
        _lexical(value, role="frozen Candidate13 output namespace")
        for value in ALLOWED_OUTPUT_ROOTS
    )
    if (
        type(ALLOWED_OUTPUT_ROOTS) is not tuple
        or ALLOWED_OUTPUT_ROOTS != (OUTPUT_ROOT, FALLBACK_OUTPUT_ROOT)
        or len(frozen) != 2
        or len(set(frozen)) != 2
        or candidate not in frozen
        or candidate == candidate.parent
    ):
        raise ValueError("Candidate13 output namespace is not frozen")
    return candidate


def _assert_no_publication_residue(parent_fd: int, *, output_name: str) -> None:
    prefixes = (
        f".{output_name}.staging-",
        f".{output_name}.quarantine-",
    )
    if any(name.startswith(prefixes) for name in os.listdir(parent_fd)):
        raise FileExistsError("Candidate13 publication residue already exists")


def prevalidate_fresh_output(path: Path | None = None) -> FreshOutputSeal:
    candidate = _validate_output_namespace(path)
    _reject_symlinks(candidate.parent, role="Candidate13 output parent", require_exists=True)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate13 output parent"
    )
    try:
        _assert_no_publication_residue(parent_fd, output_name=candidate.name)
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate13 output namespace already exists")
    finally:
        os.close(parent_fd)
    return FreshOutputSeal(
        path=candidate, parent_identity=parent_identity, _sentinel=_CONTEXT_SEAL
    )


def revalidate_fresh_output(seal: FreshOutputSeal) -> None:
    if type(seal) is not FreshOutputSeal or seal._sentinel is not _CONTEXT_SEAL:
        raise PermissionError("Candidate13 output freshness seal is invalid")
    parent_fd, identity = _open_stable_directory(
        seal.path.parent, role="Candidate13 output freshness parent"
    )
    try:
        if identity != seal.parent_identity:
            raise RuntimeError("Candidate13 output parent identity changed")
        _assert_no_publication_residue(parent_fd, output_name=seal.path.name)
        try:
            os.stat(seal.path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError("Candidate13 output namespace is no longer fresh")
    finally:
        os.close(parent_fd)


def build_authorization_contract(
    context: BootstrapContext,
    *,
    qualification_record_sha256: str,
    control_binding_commitment_sha256: str,
) -> dict[str, Any]:
    _validate_context(context)
    value = {
        "adapted_cache_publication_authorized": False,
        "authorization_consumed_before_manifest_open": True,
        "consumption_anchor_namespace_commitment_sha256": _namespace_commitment(
            "CANDIDATE13_CONSUMPTION_ANCHOR_NAMESPACE_V1", C13_ANCHOR
        ),
        "control_binding_commitment_sha256": _require_sha(
            control_binding_commitment_sha256,
            name="Candidate13 authorization control commitment",
        ),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "full28_access_authorized": False,
        "holdout_label_evaluator_metric_fit_access_authorized": False,
        "kind": AUTHORIZATION_KIND,
        "manifest_binding_commitment_sha256": _manifest_contract_commitment(
            CACHE_MANIFEST,
            CACHE_MANIFEST_SHA256,
            CACHE_MANIFEST_SIZE_BYTES,
        ),
        "allowed_output_namespace_set_commitment_sha256": (
            _allowed_output_namespace_set_commitment()
        ),
        "qualification_record_commitment_sha256": _require_sha(
            qualification_record_sha256,
            name="Candidate13 authorization qualification commitment",
        ),
        "retry_after_anchor_authorized": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "AUTHORIZED_ONCE",
    }
    _validate_allowed_output_namespace_binding(value, role="authorization")
    _assert_aggregate_safe(value)
    return value


@dataclass(frozen=True)
class AuthorizationSeal:
    reader: GuardedReader
    authorization_bytes: bytes
    authorization_sha256: str
    authorization_size_bytes: int
    qualification_bytes: bytes
    qualification_sha256: str
    qualification_size_bytes: int
    control_binding_commitment_sha256: str
    output_freshness: FreshOutputSeal
    fallback_output_freshness: FreshOutputSeal
    authority_read_inventory_commitment_sha256: str
    authority_raw_fd_opened_commitment_sha256: str
    authority_verified_inode_commitment_sha256: str
    authority_read_started_commitment_sha256: str
    anchor_parent_identity: tuple[int, int]
    _context_token: object
    _sentinel: object
    _issuance_token: object


def _authorization_seal_commitment(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
) -> str:
    source_count = len(context.source_entries)
    authority_entries = authorization.reader.entries[
        source_count : source_count + 2
    ]
    return sha256_bytes(
        canonical_bytes(
            {
                "anchor_parent_identity": list(authorization.anchor_parent_identity),
                "authority_read_inventory_commitment_sha256": (
                    authorization.authority_read_inventory_commitment_sha256
                ),
                "authority_raw_fd_opened_commitment_sha256": (
                    authorization.authority_raw_fd_opened_commitment_sha256
                ),
                "authority_verified_inode_commitment_sha256": (
                    authorization.authority_verified_inode_commitment_sha256
                ),
                "authority_read_started_commitment_sha256": (
                    authorization.authority_read_started_commitment_sha256
                ),
                "authorization_bytes_sha256": sha256_bytes(
                    authorization.authorization_bytes
                ),
                "authorization_sha256": authorization.authorization_sha256,
                "authorization_size_bytes": authorization.authorization_size_bytes,
                "control_binding_commitment_sha256": (
                    authorization.control_binding_commitment_sha256
                ),
                "domain": "CANDIDATE13_ISSUED_AUTHORIZATION_SEAL_V1",
                "entries": [dict(entry) for entry in authority_entries],
                "output_parent_identity": list(
                    authorization.output_freshness.parent_identity
                ),
                "output_path": str(authorization.output_freshness.path.relative_to(ROOT)),
                "fallback_output_parent_identity": list(
                    authorization.fallback_output_freshness.parent_identity
                ),
                "fallback_output_path": str(
                    authorization.fallback_output_freshness.path.relative_to(ROOT)
                ),
                "qualification_bytes_sha256": sha256_bytes(
                    authorization.qualification_bytes
                ),
                "qualification_sha256": authorization.qualification_sha256,
                "qualification_size_bytes": authorization.qualification_size_bytes,
                "source_inventory_commitment_sha256": (
                    context.source_commitment_sha256
                ),
            }
        )
    )


def _qualification_static_contract(
    context: BootstrapContext,
    value: Mapping[str, Any],
    *,
    control_commitment: str,
) -> None:
    _validate_allowed_output_namespace_binding(value, role="qualification")
    expected_keys = {
        "adapted_cache_publication_authorized",
        "artifact_source_inventory_opened",
        "cache_manifest_opened",
        "check_code",
        "consumption_anchor_namespace_commitment_sha256",
        "control_binding_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_read_inventory_commitment_sha256",
        "control_read_started_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "engineering_attempt_id",
        "holdout_label_evaluator_metric_fit_opened_or_started",
        "kind",
        "manifest_binding_commitment_sha256",
        "allowed_output_namespace_set_commitment_sha256",
        "repeatable",
        "scientific_hypothesis_consumed",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "status",
    }
    if (
        set(value) != expected_keys
        or value.get("kind") != QUALIFICATION_KIND
        or value.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or value.get("check_code") != "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED"
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("repeatable") is not True
        or value.get("artifact_source_inventory_opened") is not False
        or value.get("cache_manifest_opened") is not False
        or value.get("scientific_hypothesis_consumed") is not False
        or value.get("adapted_cache_publication_authorized") is not False
        or value.get("holdout_label_evaluator_metric_fit_opened_or_started") is not False
        or value.get("consumption_anchor_namespace_commitment_sha256")
        != _namespace_commitment(
            "CANDIDATE13_CONSUMPTION_ANCHOR_NAMESPACE_V1", C13_ANCHOR
        )
        or value.get("allowed_output_namespace_set_commitment_sha256")
        != _allowed_output_namespace_set_commitment()
        or value.get("source_inventory_commitment_sha256")
        != context.source_commitment_sha256
        or value.get("control_binding_commitment_sha256") != control_commitment
        or value.get("manifest_binding_commitment_sha256")
        != _manifest_contract_commitment(
            CACHE_MANIFEST,
            CACHE_MANIFEST_SHA256,
            CACHE_MANIFEST_SIZE_BYTES,
        )
    ):
        raise PermissionError("Candidate13 qualification record contract mismatch")
    for key in (
        "consumption_anchor_namespace_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_read_inventory_commitment_sha256",
        "control_read_started_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "allowed_output_namespace_set_commitment_sha256",
    ):
        _require_sha(value.get(key), name=f"Candidate13 qualification {key}")
    _assert_aggregate_safe(value)


def authorize_preflight_once(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
    output_freshness: FreshOutputSeal,
    fallback_output_freshness: FreshOutputSeal,
) -> AuthorizationSeal:
    _validate_context(context)
    revalidate_fresh_output(output_freshness)
    revalidate_fresh_output(fallback_output_freshness)
    if (
        output_freshness.path != OUTPUT_ROOT
        or fallback_output_freshness.path != FALLBACK_OUTPUT_ROOT
        or output_freshness.path == fallback_output_freshness.path
    ):
        raise PermissionError("Candidate13 output freshness set mismatch")
    reader = GuardedReader.from_context(context)
    auth_bytes, auth = _read_bound(
        reader,
        path=C13_AUTHORIZATION,
        purpose="CANDIDATE13_EXTERNAL_AUTHORIZATION",
        sha256=_require_sha(authorization_sha256, name="Candidate13 authorization SHA"),
        size_bytes=_strict_int(
            authorization_size_bytes, name="Candidate13 authorization size"
        ),
    )
    qualification_path = _lexical(
        C13_QUALIFICATION, role="CANDIDATE13_QUALIFICATION_RECORD"
    )
    qualification_sha = _require_sha(
        auth.get("qualification_record_commitment_sha256"),
        name="Candidate13 qualification record commitment",
    )
    qualification_metadata = os.lstat(
        _reject_symlinks(
            qualification_path,
            role="CANDIDATE13_QUALIFICATION_RECORD",
            require_exists=True,
        )
    )
    qualification_size = int(qualification_metadata.st_size)
    if not stat.S_ISREG(qualification_metadata.st_mode) or qualification_size <= 0:
        raise PermissionError("Candidate13 qualification record is not a regular file")
    qualification_bytes, qualification_value = _read_bound(
        reader,
        path=qualification_path,
        purpose="CANDIDATE13_QUALIFICATION_RECORD",
        sha256=qualification_sha,
        size_bytes=qualification_size,
    )
    control_commitment = _require_sha(
        auth.get("control_binding_commitment_sha256"),
        name="Candidate13 authorization control commitment",
    )
    expected_auth = build_authorization_contract(
        context,
        qualification_record_sha256=qualification_sha,
        control_binding_commitment_sha256=control_commitment,
    )
    _validate_allowed_output_namespace_binding(auth, role="authorization")
    if auth != expected_auth or sha256_bytes(auth_bytes) != authorization_sha256:
        raise PermissionError("Candidate13 external authorization contract mismatch")
    _qualification_static_contract(
        context, qualification_value, control_commitment=control_commitment
    )
    source_count = len(context.source_entries)
    authority_entries = reader.entries[source_count:]
    authority_purposes = [str(entry["purpose"]) for entry in authority_entries]
    expected_purposes = [
        "CANDIDATE13_EXTERNAL_AUTHORIZATION",
        "CANDIDATE13_QUALIFICATION_RECORD",
    ]
    full_expected = [str(entry["purpose"]) for entry in context.source_entries] + expected_purposes
    if (
        authority_purposes != expected_purposes
        or reader.raw_fd_opened_purposes != full_expected
        or reader.verified_inode_purposes != full_expected
        or reader.read_started_purposes != full_expected
        or reader.denied_attempts
    ):
        raise PermissionError("Candidate13 authorization read inventory mismatch")
    anchor_parent_fd, anchor_parent_identity = _open_stable_directory(
        C13_ANCHOR.parent, role="Candidate13 anchor parent authorization pin"
    )
    os.close(anchor_parent_fd)
    issuance_token = object()
    seal = AuthorizationSeal(
        reader=reader,
        authorization_bytes=auth_bytes,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        qualification_bytes=qualification_bytes,
        qualification_sha256=qualification_sha,
        qualification_size_bytes=qualification_size,
        control_binding_commitment_sha256=control_commitment,
        output_freshness=output_freshness,
        fallback_output_freshness=fallback_output_freshness,
        authority_read_inventory_commitment_sha256=_entry_commitment(authority_entries),
        authority_raw_fd_opened_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_AUTHORITY_RAW_FD_OPENED_V1", authority_purposes
        ),
        authority_verified_inode_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_AUTHORITY_VERIFIED_INODES_V1", authority_purposes
        ),
        authority_read_started_commitment_sha256=_purpose_commitment(
            "CANDIDATE13_AUTHORITY_READ_STARTED_V1", authority_purposes
        ),
        anchor_parent_identity=anchor_parent_identity,
        _context_token=context._issuance_token,
        _sentinel=_AUTHORIZATION_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_AUTHORIZATIONS[id(seal)] = (
        seal,
        issuance_token,
        _authorization_seal_commitment(context, seal),
    )
    return seal


def _validate_authorization(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
) -> None:
    issued = _ISSUED_AUTHORIZATIONS.get(id(authorization))
    if (
        type(authorization) is not AuthorizationSeal
        or authorization._sentinel is not _AUTHORIZATION_SEAL
        or authorization._context_token is not context._issuance_token
        or issued is None
        or issued[0] is not authorization
        or issued[1] is not authorization._issuance_token
    ):
        raise PermissionError("Candidate13 external authorization seal is not intact")
    try:
        source_count = len(context.source_entries)
        authority_entries = authorization.reader.entries[
            source_count : source_count + 2
        ]
        purposes = [str(entry["purpose"]) for entry in authority_entries]
        expected_purposes = [
            "CANDIDATE13_EXTERNAL_AUTHORIZATION",
            "CANDIDATE13_QUALIFICATION_RECORD",
        ]
        expected_prefix = [
            str(entry["purpose"]) for entry in context.source_entries
        ] + expected_purposes
        authorization_value = _decode(
            authorization.authorization_bytes,
            name="Candidate13 issued authorization",
        )
        qualification_value = _decode(
            authorization.qualification_bytes,
            name="Candidate13 issued qualification",
        )
        expected_authorization = build_authorization_contract(
            context,
            qualification_record_sha256=authorization.qualification_sha256,
            control_binding_commitment_sha256=(
                authorization.control_binding_commitment_sha256
            ),
        )
        _qualification_static_contract(
            context,
            qualification_value,
            control_commitment=authorization.control_binding_commitment_sha256,
        )
        entry_contracts = (
            (
                authority_entries[0],
                C13_AUTHORIZATION,
                "CANDIDATE13_EXTERNAL_AUTHORIZATION",
                authorization.authorization_sha256,
                authorization.authorization_size_bytes,
            ),
            (
                authority_entries[1],
                C13_QUALIFICATION,
                "CANDIDATE13_QUALIFICATION_RECORD",
                authorization.qualification_sha256,
                authorization.qualification_size_bytes,
            ),
        )
        if (
            len(authority_entries) != 2
            or purposes != expected_purposes
            or authorization_value != expected_authorization
            or sha256_bytes(authorization.authorization_bytes)
            != authorization.authorization_sha256
            or len(authorization.authorization_bytes)
            != authorization.authorization_size_bytes
            or sha256_bytes(authorization.qualification_bytes)
            != authorization.qualification_sha256
            or len(authorization.qualification_bytes)
            != authorization.qualification_size_bytes
            or authorization.authority_read_inventory_commitment_sha256
            != _entry_commitment(authority_entries)
            or authorization.authority_raw_fd_opened_commitment_sha256
            != _purpose_commitment(
                "CANDIDATE13_AUTHORITY_RAW_FD_OPENED_V1", purposes
            )
            or authorization.authority_verified_inode_commitment_sha256
            != _purpose_commitment(
                "CANDIDATE13_AUTHORITY_VERIFIED_INODES_V1", purposes
            )
            or authorization.authority_read_started_commitment_sha256
            != _purpose_commitment(
                "CANDIDATE13_AUTHORITY_READ_STARTED_V1", purposes
            )
            or authorization.reader.raw_fd_opened_purposes[: len(expected_prefix)]
            != expected_prefix
            or authorization.reader.verified_inode_purposes[: len(expected_prefix)]
            != expected_prefix
            or authorization.reader.read_started_purposes[: len(expected_prefix)]
            != expected_prefix
            or issued[2] != _authorization_seal_commitment(context, authorization)
            or any(
                str(entry.get("path"))
                != str(_lexical(path, role=purpose).relative_to(ROOT))
                or entry.get("purpose") != purpose
                or entry.get("sha256") != sha256
                or entry.get("size_bytes") != size_bytes
                for entry, path, purpose, sha256, size_bytes in entry_contracts
            )
        ):
            raise ValueError("Candidate13 issued authorization changed")
    except Exception:
        raise PermissionError(
            "Candidate13 external authorization seal is not intact"
        ) from None
    parent_fd, parent_identity = _open_stable_directory(
        C13_ANCHOR.parent, role="Candidate13 anchor parent authorization revalidation"
    )
    os.close(parent_fd)
    if parent_identity != authorization.anchor_parent_identity:
        raise PermissionError("Candidate13 anchor parent changed after authorization")
    revalidate_fresh_output(authorization.output_freshness)
    revalidate_fresh_output(authorization.fallback_output_freshness)


@dataclass(frozen=True)
class ConsumptionSeal:
    path: Path
    sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    bytes_value: bytes
    _context_token: object
    _sentinel: object
    _issuance_token: object


@dataclass
class ConsumptionProgress:
    anchor_durable_entry_created_or_partial: bool = False
    consumption: ConsumptionSeal | None = None


def consume_authorization(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    progress: ConsumptionProgress | None = None,
) -> ConsumptionSeal:
    if progress is not None and type(progress) is not ConsumptionProgress:
        raise TypeError("Candidate13 consumption progress type mismatch")
    _validate_context(context)
    _validate_authorization(context, authorization)
    _validate_qualification(context, qualification)
    expected_qualification = build_qualification_record(context, qualification)
    if (
        qualification.reader is not authorization.reader
        or expected_qualification != authorization.qualification_bytes
        or qualification.control_binding_commitment_sha256
        != authorization.control_binding_commitment_sha256
    ):
        raise PermissionError("Candidate13 authority and qualification do not bind")
    _validate_preanchor_reader_state(context, authorization, qualification)
    anchor = _lexical(C13_ANCHOR, role="Candidate13 consumption anchor")
    frozen = _lexical(
        ROOT
        / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1.json",
        role="frozen Candidate13 consumption anchor",
    )
    if anchor != frozen:
        raise ValueError("Candidate13 consumption anchor namespace is not frozen")
    value = {
        "adapted_cache_publication_authorized": False,
        "authorization_commitment_sha256": authorization.authorization_sha256,
        "consumed_before_manifest_open": True,
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_authorized": False,
        "kind": ANCHOR_KIND,
        "manifest_opened_at_anchor_creation": False,
        "manifest_binding_commitment_sha256": _manifest_contract_commitment(
            CACHE_MANIFEST,
            CACHE_MANIFEST_SHA256,
            CACHE_MANIFEST_SIZE_BYTES,
        ),
        "allowed_output_namespace_set_commitment_sha256": (
            _allowed_output_namespace_set_commitment()
        ),
        "qualification_record_commitment_sha256": authorization.qualification_sha256,
        "retry_authorized": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "CONSUMED_BEFORE_MANIFEST_OPEN",
    }
    _validate_allowed_output_namespace_binding(value, role="consumption anchor")
    _assert_aggregate_safe(value)
    data = canonical_bytes(value)
    parent_fd, parent_identity = _open_stable_directory(
        anchor.parent, role="Candidate13 anchor parent"
    )
    if parent_identity != authorization.anchor_parent_identity:
        os.close(parent_fd)
        raise PermissionError("Candidate13 anchor parent changed before O_EXCL")
    descriptor: int | None = None
    file_identity: tuple[int, int] | None = None
    created = False
    try:
        descriptor = os.open(
            anchor.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        created = True
        if progress is not None:
            progress.anchor_durable_entry_created_or_partial = True
        os.fsync(parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate13 anchor write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        metadata = os.fstat(descriptor)
        file_identity = _identity(metadata)
        visible = os.stat(anchor.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != len(data)
            or _identity(visible) != file_identity
            or _identity(os.lstat(anchor.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate13 anchor publication mismatch")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise PermissionError("Candidate13 external authorization is already consumed") from error
    except (WatchdogExpired, CpuTimerExpired, ResourceEnvelopeFailure) as error:
        if created:
            raise AnchorCreationFailure(
                "Candidate13 anchor interrupted after durable consumption"
            ) from error
        raise
    except BaseException as error:
        if created:
            raise AnchorCreationFailure(
                "Candidate13 anchor failed after durable consumption"
            ) from error
        raise
    finally:
        close_error: BaseException | None = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                close_error = error
        try:
            os.close(parent_fd)
        except BaseException as error:
            if close_error is None:
                close_error = error
        if close_error is not None:
            if created:
                raise AnchorCreationFailure(
                    "Candidate13 anchor close failed after durable consumption"
                ) from close_error
            raise close_error
    if file_identity is None:
        raise AnchorCreationFailure("Candidate13 anchor lacks an owned identity")
    issuance_token = object()
    seal = ConsumptionSeal(
        path=anchor,
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        file_identity=file_identity,
        parent_identity=parent_identity,
        bytes_value=data,
        _context_token=context._issuance_token,
        _sentinel=_CONSUMPTION_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_CONSUMPTIONS[id(seal)] = (seal, issuance_token)
    try:
        revalidate_anchor(context, seal)
    except BaseException as error:
        raise AnchorCreationFailure(
            "Candidate13 anchor final validation failed after durable consumption"
        ) from error
    if progress is not None:
        progress.consumption = seal
    return seal


def revalidate_anchor(context: BootstrapContext, seal: ConsumptionSeal) -> None:
    issued = _ISSUED_CONSUMPTIONS.get(id(seal))
    if (
        type(seal) is not ConsumptionSeal
        or seal._sentinel is not _CONSUMPTION_SEAL
        or seal._context_token is not context._issuance_token
        or issued is None
        or issued[0] is not seal
        or issued[1] is not seal._issuance_token
        or sha256_bytes(seal.bytes_value) != seal.sha256
        or len(seal.bytes_value) != seal.size_bytes
    ):
        raise PermissionError("Candidate13 consumption anchor seal is invalid")
    parent_fd, parent_identity = _open_stable_directory(
        seal.path.parent, role="Candidate13 anchor parent revalidation"
    )
    descriptor: int | None = None
    try:
        if parent_identity != seal.parent_identity:
            raise RuntimeError("Candidate13 anchor parent changed")
        descriptor = os.open(
            seal.path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != seal.file_identity
            or int(opened.st_size) != seal.size_bytes
        ):
            raise RuntimeError("Candidate13 anchor metadata changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible = os.stat(seal.path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            data != seal.bytes_value
            or sha256_bytes(data) != seal.sha256
            or _identity(after) != identity
            or _identity(visible) != identity
            or _identity(os.lstat(seal.path.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate13 anchor bytes or visibility changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


@dataclass(frozen=True)
class ManifestRecord:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    pool_path: Path
    pool_sha256: str
    manifest_record_binding_sha256: str


@dataclass(frozen=True)
class PendingPoolVector:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    manifest_record_binding_sha256: str
    pool_file_binding_sha256: str
    pool_sha256: str
    pool_size_bytes: int
    pool_bytes: bytes


@dataclass(frozen=True)
class ExecutionOutcome:
    core_result: Mapping[str, Any]
    core_result_bytes: bytes
    manifest_binding_sha256: str
    artifact_inventory_commitment_sha256: str


@dataclass(frozen=True)
class PayloadPlan:
    consumption: ConsumptionSeal | None
    outcome: ExecutionOutcome | None
    failure_stage: str
    anchor_durable_entry_created_or_partial: bool
    resource_envelope_enforced: bool


def _manifest_file_binding(entry: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "device_id": _strict_int(entry["device_id"], name="manifest device"),
                "domain": "CANDIDATE13_ATTESTED_MANIFEST_FILE_V1",
                "inode": _strict_int(entry["inode"], name="manifest inode"),
                "path": str(entry["path"]),
                "sha256": _require_sha(entry["sha256"], name="manifest SHA"),
                "size_bytes": _strict_int(entry["size_bytes"], name="manifest size"),
            }
        )
    )


def _artifact_file_binding(
    entry: Mapping[str, Any], *, manifest_record_binding_sha256: str
) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "device_id": _strict_int(entry["device_id"], name="artifact device"),
                "domain": "CANDIDATE13_ATTESTED_POOL_FILE_V1",
                "inode": _strict_int(entry["inode"], name="artifact inode"),
                "manifest_record_binding_sha256": _require_sha(
                    manifest_record_binding_sha256,
                    name="manifest record binding",
                ),
                "path": str(entry["path"]),
                "purpose": str(entry["purpose"]),
                "sha256": _require_sha(entry["sha256"], name="artifact SHA"),
                "size_bytes": _strict_int(entry["size_bytes"], name="artifact size"),
            }
        )
    )


def _validate_full_manifest(
    manifest: Mapping[str, Any],
    qualification: ControlQualification,
    *,
    manifest_binding_sha256: str,
) -> tuple[ManifestRecord, ...]:
    expected_keys = {
        "case_count",
        "holdout_entry_count",
        "kind",
        "record_count",
        "records",
        "schema_version",
        "source_only",
        "split_sha256",
        "train_allowlist_sha256",
        "train_set_commitment_sha256",
        "v51_scan_only_manifest_sha256",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != CACHE_MANIFEST_KIND
        or manifest.get("source_only") is not True
        or type(manifest.get("case_count")) is not int
        or manifest.get("case_count") != PAIR_COUNT
        or type(manifest.get("record_count")) is not int
        or manifest.get("record_count") != RECORD_COUNT
        or type(manifest.get("holdout_entry_count")) is not int
        or manifest.get("holdout_entry_count") != 0
        or manifest.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1]
        or manifest.get("train_allowlist_sha256")
        != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1]
        or manifest.get("train_set_commitment_sha256")
        != qualification.train_commitment_sha256
        or manifest.get("v51_scan_only_manifest_sha256")
        != V51_SCAN_ONLY_MANIFEST_SHA256
    ):
        raise RunnerFailure(
            "CACHE_MANIFEST_VALIDATION", "CACHE_MANIFEST_HEADER_MISMATCH"
        )
    raw_records = manifest.get("records")
    if type(raw_records) is not list or len(raw_records) != RECORD_COUNT:
        raise RunnerFailure(
            "CACHE_MANIFEST_VALIDATION", "CACHE_MANIFEST_RECORD_COUNT_MISMATCH"
        )
    record_keys = {
        "case_id",
        "pass",
        "pool_path",
        "pool_sha256",
        "runtime_case_id",
        "scan_sha256",
        "semantic_path",
        "semantic_sha256",
    }
    expected_pairs = [
        (case_id, pass_name)
        for case_id in qualification.train_ids
        for pass_name in ("primary", "replay")
    ]
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    pool_paths: set[Path] = set()
    primary_hashes: dict[str, str] = {}
    primary_semantic_hashes: dict[str, str] = {}
    bindings: set[str] = set()
    records: list[ManifestRecord] = []
    for ordinal, (raw, expected_pair) in enumerate(zip(raw_records, expected_pairs)):
        if type(raw) is not dict or set(raw) != record_keys:
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_SCHEMA_MISMATCH"
            )
        case_id, pass_name = expected_pair
        if (raw.get("case_id"), raw.get("pass")) != expected_pair:
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_MEMBERSHIP_ORDER_MISMATCH"
            )
        runtime_id = raw.get("runtime_case_id")
        if (
            type(runtime_id) is not str
            or len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id
            or case_by_runtime.setdefault(runtime_id, case_id) != case_id
            or _require_sha(raw.get("scan_sha256"), name="manifest scan SHA")
            != qualification.scan_by_case[case_id]
        ):
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_IDENTITY_MISMATCH"
            )
        pool_relative_raw = raw.get("pool_path")
        semantic_relative_raw = raw.get("semantic_path")
        if (
            type(pool_relative_raw) is not str
            or type(semantic_relative_raw) is not str
        ):
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_PATH_MISMATCH"
            )
        pool_relative = Path(pool_relative_raw)
        semantic_relative = Path(semantic_relative_raw)
        if (
            pool_relative.is_absolute()
            or semantic_relative.is_absolute()
            or ".." in pool_relative.parts
            or ".." in semantic_relative.parts
        ):
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_PATH_MISMATCH"
            )
        pool_path = _lexical(ROOT / pool_relative, role="TRAIN pool")
        expected_pool = _lexical(
            ALLOWED_CACHE_ROOT
            / pass_name
            / runtime_id
            / "pool"
            / f"{runtime_id}.json",
            role="expected TRAIN pool",
        )
        expected_semantic_relative = (
            ALLOWED_CACHE_ROOT
            / pass_name
            / runtime_id
            / "semantic_result.json"
        ).relative_to(ROOT)
        if (
            not _inside_cache_root(pool_path)
            or pool_path != expected_pool
            or pool_path in pool_paths
            or semantic_relative != expected_semantic_relative
        ):
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_PATH_MISMATCH"
            )
        pool_paths.add(pool_path)
        pool_sha = _require_sha(raw.get("pool_sha256"), name="pool SHA")
        semantic_sha = _require_sha(
            raw.get("semantic_sha256"), name="semantic manifest SHA"
        )
        if pass_name == "primary":
            primary_hashes[case_id] = pool_sha
            primary_semantic_hashes[case_id] = semantic_sha
        elif (
            primary_hashes.get(case_id) != pool_sha
            or primary_semantic_hashes.get(case_id) != semantic_sha
        ):
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "PRIMARY_REPLAY_HASH_MISMATCH"
            )
        record_binding = sha256_bytes(
            canonical_bytes(
                {
                    "domain": "CANDIDATE13_ATTESTED_MANIFEST_RECORD_V1",
                    "manifest_binding_sha256": manifest_binding_sha256,
                    "ordinal": ordinal,
                    "row": raw,
                }
            )
        )
        if record_binding in bindings:
            raise RunnerFailure(
                "CACHE_MANIFEST_VALIDATION", "DUPLICATE_RECORD_BINDING"
            )
        bindings.add(record_binding)
        records.append(
            ManifestRecord(
                ordinal=ordinal,
                pair_ordinal=ordinal // 2,
                pass_name=pass_name,
                pool_path=pool_path,
                pool_sha256=pool_sha,
                manifest_record_binding_sha256=record_binding,
            )
        )
    if len(pool_paths) != POOL_FILE_COUNT or len(bindings) != RECORD_COUNT:
        raise RunnerFailure(
            "CACHE_MANIFEST_VALIDATION", "EXACT_MANIFEST_INVENTORY_MISMATCH"
        )
    return tuple(records)


def _expected_complete_purposes(
    context: BootstrapContext,
    records: Sequence[ManifestRecord],
) -> list[str]:
    expected = [str(entry["purpose"]) for entry in context.source_entries]
    expected.extend(
        ["CANDIDATE13_EXTERNAL_AUTHORIZATION", "CANDIDATE13_QUALIFICATION_RECORD"]
    )
    expected.extend(_expected_control_purposes())
    expected.append("TRAIN_CACHE_MANIFEST")
    expected.extend(
        f"CANDIDATE13_{record.pass_name.upper()}_{record.pair_ordinal:02d}_POOL"
        for record in records
    )
    return expected


def _validate_complete_inventory(
    context: BootstrapContext,
    qualification: ControlQualification,
    records: Sequence[ManifestRecord],
) -> None:
    reader = qualification.reader
    expected = _expected_complete_purposes(context, records)
    observed = [str(entry["purpose"]) for entry in reader.entries]
    paths = [str(entry["path"]) for entry in reader.entries]
    identities = [
        (int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries
    ]
    if (
        type(records) is not tuple
        or len(records) != RECORD_COUNT
        or observed != expected
        or reader.raw_fd_opened_purposes != expected
        or reader.verified_inode_purposes != expected
        or reader.read_started_purposes != expected
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or reader.denied_attempts
        or reader.artifact_inventory is None
        or len(reader.artifact_inventory.bindings) != POOL_FILE_COUNT
    ):
        raise RunnerFailure(
            "EXACT_READ_INVENTORY", "EXACT_POOL_READ_INVENTORY_MISMATCH"
        )

def read_localize_full_pool_inventory(
    context: BootstrapContext,
    qualification: ControlQualification,
    consumption: ConsumptionSeal,
    resource_envelope: ResourceEnvelopeSeal,
) -> ExecutionOutcome:
    _validate_context(context)
    _validate_qualification(context, qualification)
    _validate_resource_envelope(resource_envelope)
    reader = qualification.reader
    try:
        revalidate_anchor(context, consumption)
        reader.authorize(
            qualification.manifest_path,
            purpose="TRAIN_CACHE_MANIFEST",
            sha256=qualification.manifest_sha256,
            size_bytes=qualification.manifest_size_bytes,
        )
        revalidate_anchor(context, consumption)
        _validate_resource_envelope(resource_envelope)
        manifest_bytes = reader.read_bytes(
            qualification.manifest_path, purpose="TRAIN_CACHE_MANIFEST"
        )
        _validate_resource_envelope(resource_envelope)
        revalidate_anchor(context, consumption)
        manifest = _decode(manifest_bytes, name="TRAIN_CACHE_MANIFEST")
        manifest_entry = reader.entries[-1]
        manifest_binding = _manifest_file_binding(manifest_entry)
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "CACHE_MANIFEST_READ", "C13_CACHE_MANIFEST_READ_FAILED"
        ) from None
    try:
        records = _validate_full_manifest(
            manifest,
            qualification,
            manifest_binding_sha256=manifest_binding,
        )
    except RunnerFailure:
        raise
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "CACHE_MANIFEST_VALIDATION", "C13_CACHE_MANIFEST_VALIDATION_FAILED"
        ) from None
    try:
        revalidate_anchor(context, consumption)
        _validate_resource_envelope(resource_envelope)
        specifications = tuple(
            (
                record.pool_path,
                f"CANDIDATE13_{record.pass_name.upper()}_"
                f"{record.pair_ordinal:02d}_POOL",
                record.pool_sha256,
            )
            for record in records
        )
        artifact_inventory = reader.install_artifact_inventory(specifications)
        revalidate_anchor(context, consumption)
        pending_vectors: list[PendingPoolVector] = []
        for record in records:
            revalidate_anchor(context, consumption)
            _validate_resource_envelope(resource_envelope)
            purpose = (
                f"CANDIDATE13_{record.pass_name.upper()}_"
                f"{record.pair_ordinal:02d}_POOL"
            )
            pool_binding = artifact_inventory.bindings[record.pool_path]
            reader.authorize(
                record.pool_path,
                purpose=purpose,
                sha256=record.pool_sha256,
                size_bytes=pool_binding.size_bytes,
                artifact_inventory=artifact_inventory,
            )
            revalidate_anchor(context, consumption)
            pool_bytes = reader.read_bytes(record.pool_path, purpose=purpose)
            _validate_resource_envelope(resource_envelope)
            revalidate_anchor(context, consumption)
            pool_entry = reader.entries[-1]
            pending_vectors.append(
                PendingPoolVector(
                    ordinal=record.ordinal,
                    pair_ordinal=record.pair_ordinal,
                    pass_name=record.pass_name,
                    manifest_record_binding_sha256=(
                        record.manifest_record_binding_sha256
                    ),
                    pool_file_binding_sha256=_artifact_file_binding(
                        pool_entry,
                        manifest_record_binding_sha256=(
                            record.manifest_record_binding_sha256
                        ),
                    ),
                    pool_sha256=record.pool_sha256,
                    pool_size_bytes=len(pool_bytes),
                    pool_bytes=pool_bytes,
                )
            )
        revalidate_anchor(context, consumption)
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "POOL_INVENTORY_OR_READ", "C13_POOL_INVENTORY_OR_READ_FAILED"
        ) from None
    try:
        _validate_complete_inventory(context, qualification, records)
        _validate_resource_envelope(resource_envelope)
        if (
            type(pending_vectors) is not list
            or len(pending_vectors) != RECORD_COUNT
            or tuple(vector.ordinal for vector in pending_vectors)
            != tuple(range(RECORD_COUNT))
            or tuple((vector.pair_ordinal, vector.pass_name) for vector in pending_vectors)
            != tuple(
                (pair_ordinal, pass_name)
                for pair_ordinal in range(PAIR_COUNT)
                for pass_name in ("primary", "replay")
            )
            or any(
                type(vector.pool_bytes) is not bytes
                or len(vector.pool_bytes) != vector.pool_size_bytes
                or sha256_bytes(vector.pool_bytes) != vector.pool_sha256
                for vector in pending_vectors
            )
        ):
            raise ValueError("Candidate13 complete pool vector mismatch")
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "EXACT_READ_INVENTORY", "C13_EXACT_POOL_READ_INVENTORY_FAILED"
        ) from None
    try:
        # The complete manifest, exact 36 pool leaves, and raw-fd / verified-
        # inode / read-start vectors are sealed before the context is created.
        localizer_context = context.c13_core.make_verified_context(
            c9_adapter=context.c9_adapter,
            c6_hypothesis=context.c6_hypothesis,
            builder=context.builder,
            clusterer=context.clusterer,
            bindings=_c13_core_bindings(),
        )
        _validate_resource_envelope(resource_envelope)
        receipts_list: list[Any] = []
        for vector in pending_vectors:
            _validate_resource_envelope(resource_envelope)
            receipts_list.append(
                context.c13_core.localize_pool_adapter_contract(
                    localizer_context,
                    pool_bytes=vector.pool_bytes,
                    expected_pool_sha256=vector.pool_sha256,
                )
            )
            _validate_resource_envelope(resource_envelope)
        receipts = tuple(receipts_list)
        if (
            type(receipts) is not tuple
            or len(receipts) != RECORD_COUNT
            or len({id(receipt) for receipt in receipts}) != RECORD_COUNT
        ):
            raise RuntimeError("Candidate13 receipt issuance contract mismatch")
        core_result = context.c13_core.aggregate_pool_adapter_receipts(
            localizer_context,
            receipts,
        )
        _validate_resource_envelope(resource_envelope)
        _validate_core_aggregate(context, core_result)
        core_bytes = canonical_bytes(core_result)
        revalidate_anchor(context, consumption)
    except (WatchdogExpired, CpuTimerExpired):
        raise
    except ResourceEnvelopeFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "CORE_LOCALIZER_INVOCATION", "C13_CORE_LOCALIZER_INVOCATION_FAILED"
        ) from None
    return ExecutionOutcome(
        core_result=types.MappingProxyType(dict(core_result)),
        core_result_bytes=core_bytes,
        manifest_binding_sha256=manifest_binding,
        artifact_inventory_commitment_sha256=(
            artifact_inventory.commitment_sha256
        ),
    )


ENGINEERING_FAILURE_CODES = {
    "ANCHOR_CONSUMPTION": "C13_ANCHOR_CONSUMPTION_FAILED",
    "CACHE_MANIFEST_READ": "C13_CACHE_MANIFEST_READ_FAILED",
    "CACHE_MANIFEST_VALIDATION": "C13_CACHE_MANIFEST_VALIDATION_FAILED",
    "POOL_INVENTORY_OR_READ": "C13_POOL_INVENTORY_OR_READ_FAILED",
    "PAYLOAD_MATERIALIZATION": "C13_PAYLOAD_MATERIALIZATION_FAILED",
    "EXACT_READ_INVENTORY": "C13_EXACT_POOL_READ_INVENTORY_FAILED",
    "CORE_LOCALIZER_INVOCATION": "C13_CORE_LOCALIZER_INVOCATION_FAILED",
    "PREANCHOR_REVALIDATION": "C13_PREANCHOR_REVALIDATION_FAILED",
    "RESOURCE_ENVELOPE": "C13_RESOURCE_ENVELOPE_FAILED",
    "CPU_TIMER_TIMEOUT": "C13_CPU_TIMER_TIMEOUT",
    "WATCHDOG_TIMEOUT": "C13_WATCHDOG_TIMEOUT",
}


def _assert_aggregate_safe(value: Any) -> None:
    """Reject every non-scalar or record-bearing public artifact shape."""

    forbidden_key_fragments = (
        "case_id",
        "count",
        "elapsed",
        "exception",
        "identifier",
        "ordinal",
        "path",
        "per_record",
        "rss",
        "runtime_case",
        "size_bytes",
        "source_value",
        "timing",
    )
    if not isinstance(value, Mapping):
        raise ValueError("Candidate13 public artifact must be one flat mapping")
    for key, child in value.items():
        if type(key) is not str or any(
            fragment in key for fragment in forbidden_key_fragments
        ):
            raise ValueError("Candidate13 public artifact exposes a forbidden field")
        if type(child) not in (bool, str):
            raise ValueError("Candidate13 public artifact exposes a non-scalar value")


def _validate_core_aggregate(
    context: BootstrapContext, value: Mapping[str, Any]
) -> None:
    false_keys = (
        "adapter_or_hypothesis_semantics_modified_by_core",
        "artifact_discovery_or_open_performed_by_core",
        "builder_or_clusterer_invoked_by_core",
        "candidate11_actual_record_cause_resolved",
        "candidate9_internal_failure_stage_claimed",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_projection_bytes_published",
        "input_origin_authenticated_by_core",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "real_or_control_input_opened_by_core",
        "rule_selected",
    )
    reached = value.get("stage_reached_all_bitset")
    passed = value.get("stage_passed_all_bitset")
    if (
        type(value) is not dict
        or set(value) != set(context.c13_core.RESULT_KEYS)
        or value.get("kind") != context.c13_core.RESULT_KIND
        or any(value.get(key) is not False for key in false_keys)
        or value.get("diagnostic_stage_precedence_is_candidate13_contract")
        is not True
        or type(value.get("exact_c9_success_failure_equivalent_for_all"))
        is not bool
        or type(reached) is not str
        or type(passed) is not str
        or len(reached) != 10
        or len(passed) != 10
        or set(reached + passed) - {"0", "1"}
        or any(
            pass_bit == "1" and reach_bit != "1"
            for reach_bit, pass_bit in zip(reached, passed)
        )
        or reached[-1] != "1"
        or value.get("stage_contract_commitment_sha256")
        != sha256_bytes(canonical_bytes(list(context.c13_core.STAGE_CONTRACT)))
    ):
        raise ValueError("Candidate13 core aggregate schema mismatch")
    first_failure = next(
        (index for index, bit in enumerate(passed) if bit == "0"), None
    )
    expected_failed = (
        "NONE"
        if first_failure is None
        else context.c13_core.STAGE_CONTRACT[first_failure]
    )
    if first_failure is None:
        expected_reached = "1" * 10
    elif first_failure == 9:
        expected_reached = "1" * 10
    else:
        expected_reached = "1" * (first_failure + 1) + "0" * (
            8 - first_failure
        ) + "1"
    equivalent = passed[-1] == "1"
    all_pass = passed == "1" * 10
    if not equivalent:
        expected_status = (
            "C13_LOCALIZER_EXACT_C9_OUTCOME_MISMATCH",
            "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE",
        )
    elif all_pass:
        expected_status = (
            "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL",
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        )
    else:
        expected_status = (
            "C13_C9_OUTCOME_EQUIVALENT_DIAGNOSTIC_REJECTION",
            "SOURCE_POOL_ADAPTER_DIAGNOSTIC_INCOMPATIBLE_NONAUTHORIZING",
        )
    if (
        reached != expected_reached
        or value.get("first_failed_candidate13_diagnostic_stage")
        != expected_failed
        or value.get("exact_c9_success_failure_equivalent_for_all")
        is not equivalent
        or (value.get("check_code"), value.get("status")) != expected_status
    ):
        raise ValueError("Candidate13 core aggregate invariant mismatch")
    _assert_aggregate_safe(value)


C13_LEDGER_KEYS = frozenset(
    {
        "all_pool_reads_completed",
        "all_reads_exactly_authorized",
        "anchor_consumed_before_manifest",
        "anchor_durable_entry_created_or_partial",
        "anchor_receipt_available",
        "anchor_receipt_commitment_sha256",
        "authorization_commitment_sha256",
        "check_code",
        "control_binding_commitment_sha256",
        "core_aggregate_accepted_for_publication",
        "core_aggregate_commitment_sha256",
        "denied_read_attempt_observed",
        "engineering_attempt_id",
        "failure_detail_exposed",
        "failure_stage",
        "fit_or_model_performed",
        "holdout_opened",
        "kind",
        "labels_or_evaluator_opened",
        "manifest_binding_commitment_sha256",
        "manifest_inode_verified",
        "manifest_raw_fd_opened",
        "manifest_read_started",
        "metric_computed",
        "pool_inventory_commitment_sha256",
        "pool_inventory_installed",
        "promotion_authorized",
        "qualification_commitment_sha256",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
        "resource_envelope_enforced",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_inventory_commitment_sha256",
        "source_only",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
        "verified_inode_inventory_commitment_sha256",
    }
)

C13_LOCK_KEYS = frozenset(
    {
        "aggregate_commitment_sha256",
        "allowed_output_namespace_set_commitment_sha256",
        "anchor_commitment_sha256",
        "authorization_commitment_sha256",
        "check_code",
        "control_binding_commitment_sha256",
        "engineering_attempt_id",
        "failure_stage",
        "kind",
        "ledger_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "pool_inventory_commitment_sha256",
        "post_source_or_policy_mutation_invalidates_lock",
        "promotion_authorized",
        "qualification_commitment_sha256",
        "resource_envelope_enforced",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)

C13_TERMINAL_AGGREGATE_KEYS = frozenset(
    {
        "check_code",
        "diagnostic_complete",
        "engineering_attempt_id",
        "failure_detail_exposed",
        "failure_stage",
        "fit_or_model_performed",
        "holdout_opened",
        "kind",
        "labels_or_evaluator_opened",
        "metric_computed",
        "promotion_authorized",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_only",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)


def _unavailable_commitment(name: str, *, failure_stage: str) -> str:
    if failure_stage not in ENGINEERING_FAILURE_CODES:
        raise ValueError("Candidate13 unavailable commitment stage is invalid")
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE13_UNAVAILABLE_COMMITMENT_V1",
                "failure_stage": failure_stage,
                "name": name,
            }
        )
    )


def _fixed_bitset(value: Any, *, name: str) -> str:
    width = 10
    if type(value) is not str or len(value) != width or set(value) - {"0", "1"}:
        raise ValueError(f"Candidate13 {name} must be one exact 10-bit string")
    return value


def _terminal_aggregate_value(*, failure_stage: str) -> dict[str, Any]:
    if failure_stage not in ENGINEERING_FAILURE_CODES:
        raise ValueError("Candidate13 terminal failure stage mismatch")
    value = {
        "check_code": ENGINEERING_FAILURE_CODES[failure_stage],
        "diagnostic_complete": False,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "failure_detail_exposed": False,
        "failure_stage": failure_stage,
        "fit_or_model_performed": False,
        "holdout_opened": False,
        "kind": TERMINAL_KIND,
        "labels_or_evaluator_opened": False,
        "metric_computed": False,
        "promotion_authorized": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_only": True,
        "stage_passed_all_bitset": "0" * 10,
        "stage_reached_all_bitset": "0" * 10,
        "status": "TERMINAL_ENGINEERING_FAILURE",
    }
    if set(value) != C13_TERMINAL_AGGREGATE_KEYS:
        raise ValueError("Candidate13 terminal aggregate schema mismatch")
    _assert_aggregate_safe(value)
    return value


def _validate_published_aggregate(
    context: BootstrapContext, value: Mapping[str, Any]
) -> bool:
    if value.get("kind") == context.c13_core.RESULT_KIND:
        _validate_core_aggregate(context, value)
        return True
    failure_stage = value.get("failure_stage")
    if (
        type(value) is not dict
        or type(failure_stage) is not str
        or value != _terminal_aggregate_value(failure_stage=failure_stage)
    ):
        raise ValueError("Candidate13 published aggregate schema mismatch")
    return False


def _ledger_value(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    consumption: ConsumptionSeal | None,
    outcome: ExecutionOutcome | None,
    failure_stage: str,
    anchor_durable_entry_created_or_partial: bool,
    resource_envelope_enforced: bool,
) -> dict[str, Any]:
    reader = qualification.reader
    core_available = outcome is not None
    pool_purposes = [
        purpose
        for purpose in reader.read_started_purposes
        if purpose.startswith("CANDIDATE13_") and purpose.endswith("_POOL")
    ]
    all_pool_reads = (
        reader.artifact_inventory is not None
        and len(reader.artifact_inventory.bindings) == POOL_FILE_COUNT
        and pool_purposes == _expected_pool_purposes()
        and pool_purposes
        == [
            purpose
            for purpose in reader.raw_fd_opened_purposes
            if purpose.startswith("CANDIDATE13_") and purpose.endswith("_POOL")
        ]
        == [
            purpose
            for purpose in reader.verified_inode_purposes
            if purpose.startswith("CANDIDATE13_") and purpose.endswith("_POOL")
        ]
    )
    if core_available:
        reached = _fixed_bitset(
            outcome.core_result["stage_reached_all_bitset"], name="core reached"
        )
        passed = _fixed_bitset(
            outcome.core_result["stage_passed_all_bitset"], name="core passed"
        )
        core_commitment = sha256_bytes(outcome.core_result_bytes)
        pool_commitment = outcome.artifact_inventory_commitment_sha256
        manifest_commitment = outcome.manifest_binding_sha256
        stage = "NONE"
        check_code = str(outcome.core_result["check_code"])
        status = "ENGINEERING_AGGREGATE_COMPLETED_NONAUTHORIZING"
    else:
        reached = passed = "0" * 10
        core_commitment = _unavailable_commitment(
            "core_aggregate", failure_stage=failure_stage
        )
        pool_commitment = _unavailable_commitment(
            "pool_inventory", failure_stage=failure_stage
        )
        manifest_commitment = _unavailable_commitment(
            "manifest_binding", failure_stage=failure_stage
        )
        stage = failure_stage
        check_code = ENGINEERING_FAILURE_CODES[failure_stage]
        status = "TERMINAL_ENGINEERING_FAILURE"
    value = {
        "all_pool_reads_completed": all_pool_reads,
        "all_reads_exactly_authorized": not reader.denied_attempts,
        "anchor_consumed_before_manifest": (
            consumption is not None or anchor_durable_entry_created_or_partial
        ),
        "anchor_durable_entry_created_or_partial": (
            consumption is not None or anchor_durable_entry_created_or_partial
        ),
        "anchor_receipt_available": consumption is not None,
        "anchor_receipt_commitment_sha256": (
            consumption.sha256
            if consumption is not None
            else _unavailable_commitment(
                "anchor_receipt", failure_stage=failure_stage
            )
        ),
        "authorization_commitment_sha256": authorization.authorization_sha256,
        "check_code": check_code,
        "control_binding_commitment_sha256": (
            qualification.control_binding_commitment_sha256
        ),
        "core_aggregate_accepted_for_publication": core_available,
        "core_aggregate_commitment_sha256": core_commitment,
        "denied_read_attempt_observed": bool(reader.denied_attempts),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "failure_detail_exposed": False,
        "failure_stage": stage,
        "fit_or_model_performed": False,
        "holdout_opened": False,
        "kind": LEDGER_KIND,
        "labels_or_evaluator_opened": False,
        "manifest_binding_commitment_sha256": manifest_commitment,
        "manifest_inode_verified": (
            "TRAIN_CACHE_MANIFEST" in reader.verified_inode_purposes
        ),
        "manifest_raw_fd_opened": (
            "TRAIN_CACHE_MANIFEST" in reader.raw_fd_opened_purposes
        ),
        "manifest_read_started": (
            "TRAIN_CACHE_MANIFEST" in reader.read_started_purposes
        ),
        "metric_computed": False,
        "pool_inventory_commitment_sha256": pool_commitment,
        "pool_inventory_installed": reader.artifact_inventory is not None,
        "promotion_authorized": False,
        "qualification_commitment_sha256": authorization.qualification_sha256,
        "raw_fd_opened_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE13_ALL_RAW_FD_OPENED_V1",
            reader.raw_fd_opened_purposes,
        ),
        "read_inventory_commitment_sha256": _entry_commitment(reader.entries),
        "read_started_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE13_ALL_READ_STARTED_V1",
            reader.read_started_purposes,
        ),
        "resource_envelope_enforced": resource_envelope_enforced,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "source_only": True,
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": status,
        "verified_inode_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE13_ALL_VERIFIED_INODES_V1",
            reader.verified_inode_purposes,
        ),
    }
    if (
        set(value) != C13_LEDGER_KEYS
        or type(resource_envelope_enforced) is not bool
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or any(
            type(value[key]) is not bool
            for key in C13_LEDGER_KEYS
            if key
            not in {
                "anchor_receipt_commitment_sha256",
                "authorization_commitment_sha256",
                "check_code",
                "control_binding_commitment_sha256",
                "core_aggregate_commitment_sha256",
                "engineering_attempt_id",
                "failure_stage",
                "kind",
                "manifest_binding_commitment_sha256",
                "pool_inventory_commitment_sha256",
                "qualification_commitment_sha256",
                "raw_fd_opened_inventory_commitment_sha256",
                "read_inventory_commitment_sha256",
                "read_started_inventory_commitment_sha256",
                "scientific_hypothesis_id",
                "source_inventory_commitment_sha256",
                "stage_passed_all_bitset",
                "stage_reached_all_bitset",
                "status",
                "verified_inode_inventory_commitment_sha256",
            }
        )
    ):
        raise ValueError("Candidate13 ledger schema mismatch")
    for key in value:
        if key.endswith("_commitment_sha256"):
            _require_sha(value[key], name=f"Candidate13 ledger {key}")
    _assert_aggregate_safe(value)
    return value


def _lock_value(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    consumption: ConsumptionSeal | None,
    outcome: ExecutionOutcome | None,
    failure_stage: str,
    ledger_bytes: bytes,
    aggregate_bytes: bytes,
    resource_envelope_enforced: bool,
) -> dict[str, Any]:
    if outcome is not None:
        reached = _fixed_bitset(
            outcome.core_result["stage_reached_all_bitset"], name="lock reached"
        )
        passed = _fixed_bitset(
            outcome.core_result["stage_passed_all_bitset"], name="lock passed"
        )
        check_code = str(outcome.core_result["check_code"])
        status = "ENGINEERING_AGGREGATE_COMPLETED_NONAUTHORIZING"
        stage = "NONE"
        manifest_commitment = outcome.manifest_binding_sha256
        pool_commitment = outcome.artifact_inventory_commitment_sha256
    else:
        reached = passed = "0" * 10
        check_code = ENGINEERING_FAILURE_CODES[failure_stage]
        status = "TERMINAL_ENGINEERING_FAILURE"
        stage = failure_stage
        manifest_commitment = _unavailable_commitment(
            "manifest_binding", failure_stage=failure_stage
        )
        pool_commitment = _unavailable_commitment(
            "pool_inventory", failure_stage=failure_stage
        )
    value = {
        "aggregate_commitment_sha256": sha256_bytes(aggregate_bytes),
        "allowed_output_namespace_set_commitment_sha256": (
            _allowed_output_namespace_set_commitment()
        ),
        "anchor_commitment_sha256": (
            consumption.sha256
            if consumption is not None
            else _unavailable_commitment("anchor", failure_stage=failure_stage)
        ),
        "authorization_commitment_sha256": authorization.authorization_sha256,
        "check_code": check_code,
        "control_binding_commitment_sha256": (
            qualification.control_binding_commitment_sha256
        ),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "failure_stage": stage,
        "kind": LOCK_KIND,
        "ledger_commitment_sha256": sha256_bytes(ledger_bytes),
        "manifest_binding_commitment_sha256": manifest_commitment,
        "pool_inventory_commitment_sha256": pool_commitment,
        "post_source_or_policy_mutation_invalidates_lock": True,
        "promotion_authorized": False,
        "qualification_commitment_sha256": authorization.qualification_sha256,
        "resource_envelope_enforced": resource_envelope_enforced,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": status,
    }
    _validate_allowed_output_namespace_binding(value, role="publication lock")
    if (
        set(value) != C13_LOCK_KEYS
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
    ):
        raise ValueError("Candidate13 lock schema mismatch")
    for key in value:
        if key.endswith("_commitment_sha256"):
            _require_sha(value[key], name=f"Candidate13 lock {key}")
    _assert_aggregate_safe(value)
    return value


def _payloads(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    consumption: ConsumptionSeal | None,
    outcome: ExecutionOutcome | None,
    failure_stage: str,
    anchor_durable_entry_created_or_partial: bool = False,
    resource_envelope_enforced: bool,
) -> dict[str, bytes]:
    ledger = _ledger_value(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=outcome,
        failure_stage=failure_stage,
        anchor_durable_entry_created_or_partial=(
            anchor_durable_entry_created_or_partial
        ),
        resource_envelope_enforced=resource_envelope_enforced,
    )
    ledger_bytes = canonical_bytes(ledger)
    aggregate_bytes = (
        outcome.core_result_bytes
        if outcome is not None
        else canonical_bytes(
            _terminal_aggregate_value(failure_stage=failure_stage)
        )
    )
    lock = _lock_value(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=outcome,
        failure_stage=failure_stage,
        ledger_bytes=ledger_bytes,
        aggregate_bytes=aggregate_bytes,
        resource_envelope_enforced=resource_envelope_enforced,
    )
    return {
        "pool_adapter_stage_localizer_aggregate.json": aggregate_bytes,
        "pool_adapter_stage_localizer_lock.json": canonical_bytes(lock),
        "pool_adapter_stage_localizer_read_ledger.json": ledger_bytes,
    }


def _compute_payload_plan(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    resource_envelope: ResourceEnvelopeSeal,
    progress: ConsumptionProgress,
) -> PayloadPlan:
    if type(progress) is not ConsumptionProgress:
        raise TypeError("Candidate13 compute progress type mismatch")
    try:
        _validate_context(context)
        _validate_authorization(context, authorization)
        _validate_qualification(context, qualification)
        _validate_resource_envelope(resource_envelope)
        consumption = consume_authorization(
            context,
            authorization,
            qualification,
            progress=progress,
        )
        _validate_resource_envelope(resource_envelope)
    except WatchdogExpired:
        return PayloadPlan(
            consumption=progress.consumption,
            outcome=None,
            failure_stage="WATCHDOG_TIMEOUT",
            anchor_durable_entry_created_or_partial=(
                progress.anchor_durable_entry_created_or_partial
            ),
            resource_envelope_enforced=True,
        )
    except CpuTimerExpired:
        return PayloadPlan(
            consumption=progress.consumption,
            outcome=None,
            failure_stage="CPU_TIMER_TIMEOUT",
            anchor_durable_entry_created_or_partial=(
                progress.anchor_durable_entry_created_or_partial
            ),
            resource_envelope_enforced=True,
        )
    except ResourceEnvelopeFailure:
        return PayloadPlan(
            consumption=progress.consumption,
            outcome=None,
            failure_stage="RESOURCE_ENVELOPE",
            anchor_durable_entry_created_or_partial=(
                progress.anchor_durable_entry_created_or_partial
            ),
            resource_envelope_enforced=False,
        )
    except AnchorCreationFailure:
        return PayloadPlan(
            consumption=None,
            outcome=None,
            failure_stage="ANCHOR_CONSUMPTION",
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=True,
        )
    except BaseException:
        return PayloadPlan(
            consumption=progress.consumption,
            outcome=None,
            failure_stage="PREANCHOR_REVALIDATION",
            anchor_durable_entry_created_or_partial=(
                progress.anchor_durable_entry_created_or_partial
            ),
            resource_envelope_enforced=True,
        )
    try:
        outcome = read_localize_full_pool_inventory(
            context, qualification, consumption, resource_envelope
        )
    except WatchdogExpired:
        return PayloadPlan(
            consumption=consumption,
            outcome=None,
            failure_stage="WATCHDOG_TIMEOUT",
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=True,
        )
    except CpuTimerExpired:
        return PayloadPlan(
            consumption=consumption,
            outcome=None,
            failure_stage="CPU_TIMER_TIMEOUT",
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=True,
        )
    except ResourceEnvelopeFailure:
        return PayloadPlan(
            consumption=consumption,
            outcome=None,
            failure_stage="RESOURCE_ENVELOPE",
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=False,
        )
    except BaseException as error:
        stage = (
            error.stage
            if isinstance(error, RunnerFailure)
            and error.stage in ENGINEERING_FAILURE_CODES
            else "CORE_LOCALIZER_INVOCATION"
        )
        return PayloadPlan(
            consumption=consumption,
            outcome=None,
            failure_stage=stage,
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=True,
        )
    return PayloadPlan(
        consumption=consumption,
        outcome=outcome,
        failure_stage="CORE_LOCALIZER_INVOCATION",
        anchor_durable_entry_created_or_partial=True,
        resource_envelope_enforced=True,
    )


def _validate_payload_plan(plan: PayloadPlan) -> None:
    if (
        type(plan) is not PayloadPlan
        or type(plan.anchor_durable_entry_created_or_partial) is not bool
        or type(plan.resource_envelope_enforced) is not bool
        or (
            plan.outcome is None
            and plan.failure_stage not in ENGINEERING_FAILURE_CODES
        )
        or (
            plan.consumption is not None
            and plan.anchor_durable_entry_created_or_partial is not True
        )
    ):
        raise ValueError("Candidate13 payload plan is invalid")


def _payloads_from_plan(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    plan: PayloadPlan,
) -> dict[str, bytes]:
    _validate_payload_plan(plan)
    return _payloads(
        context,
        authorization,
        qualification,
        consumption=plan.consumption,
        outcome=plan.outcome,
        failure_stage=plan.failure_stage,
        anchor_durable_entry_created_or_partial=(
            plan.anchor_durable_entry_created_or_partial
        ),
        resource_envelope_enforced=plan.resource_envelope_enforced,
    )


def _materialize_payload_plan_under_compute_envelope(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    plan: PayloadPlan,
    resource_envelope: ResourceEnvelopeSeal,
) -> dict[str, bytes]:
    _validate_resource_envelope(resource_envelope)
    payloads = _payloads_from_plan(context, authorization, qualification, plan)
    _validate_resource_envelope(resource_envelope)
    return payloads


def _materialize_payload_plan_in_publication_phase(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    plan: PayloadPlan,
    publication_phase: PublicationPhaseSeal,
) -> dict[str, bytes]:
    _validate_publication_phase(publication_phase)
    payloads = _payloads_from_plan(context, authorization, qualification, plan)
    _validate_publication_phase(publication_phase)
    return payloads

def _renameat_noreplace(parent_fd: int, source: str, destination: str) -> None:
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        flag = 0x00000004
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        flag = 0x00000001
    else:
        raise RuntimeError("Candidate13 requires a no-replace directory rename")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if (
        function(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(destination),
            flag,
        )
        != 0
    ):
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, "Candidate13 publication destination exists")
        raise OSError(number, os.strerror(number))


def _verify_owned_directory(
    parent_fd: int,
    *,
    name: str,
    directory_identity: tuple[int, int],
    leaf_identities: Mapping[str, tuple[int, int]],
    payloads: Mapping[str, bytes],
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate13 requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(name, os.O_RDONLY | nofollow | directory, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(opened) != directory_identity
            or _identity(visible) != directory_identity
            or set(os.listdir(descriptor)) != set(payloads)
            or set(leaf_identities) != set(payloads)
        ):
            raise RuntimeError("Candidate13 owned publication inventory changed")
        for filename, expected in sorted(payloads.items()):
            leaf_fd = os.open(filename, os.O_RDONLY | nofollow, dir_fd=descriptor)
            try:
                before = os.fstat(leaf_fd)
                identity = _identity(before)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or identity != leaf_identities[filename]
                    or int(before.st_size) != len(expected)
                ):
                    raise RuntimeError("Candidate13 owned leaf metadata changed")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(leaf_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                after = os.fstat(leaf_fd)
                visible_leaf = os.stat(
                    filename, dir_fd=descriptor, follow_symlinks=False
                )
                if (
                    data != expected
                    or sha256_bytes(data) != sha256_bytes(expected)
                    or _identity(after) != identity
                    or _identity(visible_leaf) != identity
                    or int(after.st_size) != len(data)
                ):
                    raise RuntimeError("Candidate13 owned leaf bytes changed")
            finally:
                os.close(leaf_fd)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ResourceEnvelopeSeal:
    prior_address_space_limit: tuple[int, int]
    prior_alarm_handler: Any
    prior_prof_handler: Any
    prior_signal_mask: frozenset[int]
    applied_address_space_soft_limit: int
    installed_alarm_handler: Any
    installed_prof_handler: Any
    triggered: list[bool]
    _sentinel: object
    _issuance_token: object


_RESOURCE_ENVELOPE_SEAL = object()
_ISSUED_RESOURCE_ENVELOPES: dict[int, tuple[object, object]] = {}


@dataclass(frozen=True)
class PublicationPhaseSeal:
    prior_address_space_limit: tuple[int, int]
    prior_alarm_handler: Any
    prior_prof_handler: Any
    prior_signal_mask: frozenset[int]
    _sentinel: object
    _issuance_token: object


_PUBLICATION_PHASE_SEAL = object()
_ISSUED_PUBLICATION_PHASES: dict[int, tuple[object, object]] = {}


def _bounded_soft_limit(prior: tuple[int, int], fixed: int) -> int:
    soft, hard = prior
    infinity = resource.RLIM_INFINITY
    candidates = [fixed]
    if soft != infinity:
        candidates.append(int(soft))
    if hard != infinity:
        candidates.append(int(hard))
    value = min(candidates)
    if value <= 0:
        raise RuntimeError("Candidate13 inherited resource limit is unusable")
    return value


def _resource_signals() -> frozenset[int]:
    required = (
        "SIGALRM",
        "SIGPROF",
        "ITIMER_REAL",
        "ITIMER_PROF",
        "pthread_sigmask",
        "sigpending",
        "sigwait",
        "setitimer",
        "getitimer",
    )
    if any(not hasattr(signal, name) for name in required) or not hasattr(
        resource, "RLIMIT_AS"
    ):
        raise ResourceEnvelopeFailure(
            "Candidate13 cancelable resource primitives are unavailable"
        )
    return frozenset((int(signal.SIGALRM), int(signal.SIGPROF)))


def _current_signal_mask() -> frozenset[int]:
    return frozenset(signal.pthread_sigmask(signal.SIG_BLOCK, frozenset()))


def _drain_new_resource_signals(signals: frozenset[int]) -> None:
    pending = frozenset(int(value) for value in signal.sigpending()) & signals
    for value in sorted(pending):
        signal.sigwait({value})
    if frozenset(int(value) for value in signal.sigpending()) & signals:
        raise ResourceEnvelopeFailure(
            "Candidate13 resource signal drain did not converge"
        )


def _arm_resource_envelope() -> ResourceEnvelopeSeal:
    signals = _resource_signals()
    prior_mask = frozenset(signal.pthread_sigmask(signal.SIG_BLOCK, signals))
    prior_address: tuple[int, int] | None = None
    prior_handler: Any = None
    prior_prof_handler: Any = None
    address_soft: int | None = None
    triggered = [False, False]

    def expire(_signum: int, _frame: Any) -> None:
        triggered[0] = True
        raise WatchdogExpired("Candidate13 fixed watchdog expired")

    def cpu_expire(_signum: int, _frame: Any) -> None:
        triggered[1] = True
        raise CpuTimerExpired("Candidate13 fixed CPU timer expired")

    address_changed = False
    alarm_handler_changed = False
    prof_handler_changed = False
    real_timer_started = False
    prof_timer_started = False
    seal: ResourceEnvelopeSeal | None = None
    try:
        if prior_mask & signals:
            raise ResourceEnvelopeFailure(
                "Candidate13 resource signals were blocked before arming"
            )
        if frozenset(int(value) for value in signal.sigpending()) & signals:
            raise ResourceEnvelopeFailure(
                "Candidate13 resource signals were pending before arming"
            )
        prior_address = resource.getrlimit(resource.RLIMIT_AS)
        prior_handler = signal.getsignal(signal.SIGALRM)
        prior_prof_handler = signal.getsignal(signal.SIGPROF)
        if (
            signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
            or signal.getitimer(signal.ITIMER_PROF) != (0.0, 0.0)
        ):
            raise ResourceEnvelopeFailure(
                "Candidate13 requires unused process timers"
            )
        address_soft = _bounded_soft_limit(
            prior_address, ADDRESS_SPACE_LIMIT_BYTES
        )
        signal.signal(signal.SIGALRM, expire)
        alarm_handler_changed = True
        signal.signal(signal.SIGPROF, cpu_expire)
        prof_handler_changed = True
        resource.setrlimit(
            resource.RLIMIT_AS, (address_soft, prior_address[1])
        )
        address_changed = True
        signal.setitimer(signal.ITIMER_REAL, WALL_TIMEOUT_SECONDS)
        real_timer_started = True
        signal.setitimer(signal.ITIMER_PROF, CPU_TIMER_SECONDS)
        prof_timer_started = True
        token = object()
        seal = ResourceEnvelopeSeal(
            prior_address_space_limit=prior_address,
            prior_alarm_handler=prior_handler,
            prior_prof_handler=prior_prof_handler,
            prior_signal_mask=prior_mask,
            applied_address_space_soft_limit=address_soft,
            installed_alarm_handler=expire,
            installed_prof_handler=cpu_expire,
            triggered=triggered,
            _sentinel=_RESOURCE_ENVELOPE_SEAL,
            _issuance_token=token,
        )
        _ISSUED_RESOURCE_ENVELOPES[id(seal)] = (seal, token)
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        _validate_resource_envelope(seal)
        return seal
    except BaseException as error:
        # Cleanup begins by re-blocking both timers.  No second delivery can
        # interrupt cancellation, handler restoration, or limit restoration.
        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        cleanup_error: BaseException | None = None
        try:
            if real_timer_started:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            if prof_timer_started:
                signal.setitimer(signal.ITIMER_PROF, 0.0)
            if address_changed and prior_address is not None:
                resource.setrlimit(resource.RLIMIT_AS, prior_address)
            if prof_handler_changed:
                signal.signal(signal.SIGPROF, prior_prof_handler)
            if alarm_handler_changed:
                signal.signal(signal.SIGALRM, prior_handler)
            if seal is not None:
                _ISSUED_RESOURCE_ENVELOPES.pop(id(seal), None)
            if real_timer_started or prof_timer_started:
                _drain_new_resource_signals(signals)
        except BaseException as cleanup:
            cleanup_error = cleanup
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        if cleanup_error is not None:
            raise ResourceEnvelopeFailure(
                "Candidate13 resource envelope cleanup failed"
            ) from None
        if isinstance(error, (WatchdogExpired, CpuTimerExpired)):
            raise
        if isinstance(error, ResourceEnvelopeFailure):
            raise
        raise ResourceEnvelopeFailure(
            "Candidate13 resource envelope could not be armed"
        ) from None


def _validate_resource_envelope(seal: ResourceEnvelopeSeal) -> None:
    issued = _ISSUED_RESOURCE_ENVELOPES.get(id(seal))
    wall_remaining, wall_interval = signal.getitimer(signal.ITIMER_REAL)
    cpu_remaining, cpu_interval = signal.getitimer(signal.ITIMER_PROF)
    if (
        type(seal) is not ResourceEnvelopeSeal
        or seal._sentinel is not _RESOURCE_ENVELOPE_SEAL
        or issued is None
        or issued[0] is not seal
        or issued[1] is not seal._issuance_token
        or type(seal.prior_signal_mask) is not frozenset
        or type(seal.triggered) is not list
        or len(seal.triggered) != 2
        or any(type(value) is not bool for value in seal.triggered)
        or type(seal.applied_address_space_soft_limit) is not int
        or seal.applied_address_space_soft_limit <= 0
        or resource.getrlimit(resource.RLIMIT_AS)
        != (
            seal.applied_address_space_soft_limit,
            seal.prior_address_space_limit[1],
        )
        or _current_signal_mask() != seal.prior_signal_mask
        or signal.getsignal(signal.SIGALRM) is not seal.installed_alarm_handler
        or signal.getsignal(signal.SIGPROF) is not seal.installed_prof_handler
        or wall_interval != 0.0
        or cpu_interval != 0.0
        or (
            (wall_remaining != 0.0)
            if seal.triggered[0]
            else not (0.0 < wall_remaining <= float(WALL_TIMEOUT_SECONDS))
        )
        or (
            (cpu_remaining != 0.0)
            if seal.triggered[1]
            else not (0.0 < cpu_remaining <= float(CPU_TIMER_SECONDS))
        )
    ):
        raise ResourceEnvelopeFailure(
            "Candidate13 resource envelope is not intact"
        )


def _enter_reserved_publication_phase(
    seal: ResourceEnvelopeSeal | None,
) -> PublicationPhaseSeal:
    signals = _resource_signals()
    if seal is not None:
        issued = _ISSUED_RESOURCE_ENVELOPES.get(id(seal))
        if (
            type(seal) is not ResourceEnvelopeSeal
            or issued is None
            or issued[0] is not seal
            or issued[1] is not seal._issuance_token
        ):
            raise ResourceEnvelopeFailure(
                "Candidate13 resource envelope is not authentic"
            )
        prior_mask = seal.prior_signal_mask
        prior_address = seal.prior_address_space_limit
        prior_alarm = seal.prior_alarm_handler
        prior_prof = seal.prior_prof_handler
    else:
        prior_mask = _current_signal_mask()
        if prior_mask & signals:
            raise ResourceEnvelopeFailure(
                "Candidate13 publication signals were already blocked"
            )
        if frozenset(int(value) for value in signal.sigpending()) & signals:
            raise ResourceEnvelopeFailure(
                "Candidate13 publication signals were already pending"
            )
        if (
            signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
            or signal.getitimer(signal.ITIMER_PROF) != (0.0, 0.0)
        ):
            raise ResourceEnvelopeFailure(
                "Candidate13 publication inherited an active timer"
            )
        prior_address = resource.getrlimit(resource.RLIMIT_AS)
        prior_alarm = signal.getsignal(signal.SIGALRM)
        prior_prof = signal.getsignal(signal.SIGPROF)

    expected_blocked_mask = prior_mask | signals

    def ensure_blocked() -> None:
        # A timer handler may run immediately before the mask syscall or after
        # the kernel has applied it but before Python stores its return value.
        # Both states are safe to adopt: retry the former and continue from the
        # already-blocked authentic-envelope state in the latter.
        for _attempt in range(8):
            try:
                observed = frozenset(
                    signal.pthread_sigmask(signal.SIG_BLOCK, signals)
                )
            except (WatchdogExpired, CpuTimerExpired):
                try:
                    current = _current_signal_mask()
                except (WatchdogExpired, CpuTimerExpired):
                    continue
                if current == expected_blocked_mask:
                    return
                if current == prior_mask:
                    continue
                raise ResourceEnvelopeFailure(
                    "Candidate13 publication signal mask changed"
                ) from None
            current = _current_signal_mask()
            if (
                observed not in (prior_mask, expected_blocked_mask)
                or current != expected_blocked_mask
            ):
                raise ResourceEnvelopeFailure(
                    "Candidate13 publication signal mask changed"
                )
            return
        raise ResourceEnvelopeFailure(
            "Candidate13 publication signals could not be blocked"
        )

    phase: PublicationPhaseSeal | None = None
    try:
        ensure_blocked()
        for _attempt in range(16):
            try:
                # Every operation is idempotent.  If a handler was delivered
                # at an injected boundary, repeat the complete sequence while
                # both signals remain blocked.
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.setitimer(signal.ITIMER_PROF, 0.0)
                if seal is not None:
                    resource.setrlimit(resource.RLIMIT_AS, prior_address)
                    signal.signal(signal.SIGALRM, prior_alarm)
                    signal.signal(signal.SIGPROF, prior_prof)
                _drain_new_resource_signals(signals)
                break
            except (WatchdogExpired, CpuTimerExpired):
                ensure_blocked()
        else:
            raise ResourceEnvelopeFailure(
                "Candidate13 publication transition did not converge"
            )
        token = object()
        phase = PublicationPhaseSeal(
            prior_address_space_limit=prior_address,
            prior_alarm_handler=prior_alarm,
            prior_prof_handler=prior_prof,
            prior_signal_mask=prior_mask,
            _sentinel=_PUBLICATION_PHASE_SEAL,
            _issuance_token=token,
        )
        _ISSUED_PUBLICATION_PHASES[id(phase)] = (phase, token)
        _validate_publication_phase(phase)
        if seal is not None:
            _ISSUED_RESOURCE_ENVELOPES.pop(id(seal), None)
        return phase
    except BaseException:
        if phase is not None:
            _ISSUED_PUBLICATION_PHASES.pop(id(phase), None)
        if seal is None:
            # No anchor exists on this path.  Restore the entry mask so the
            # caller can report a recoverable pre-anchor failure rather than
            # inheriting a half-entered reserved phase.
            for timer in (signal.ITIMER_REAL, signal.ITIMER_PROF):
                try:
                    signal.setitimer(timer, 0.0)
                except BaseException:
                    pass
            try:
                _drain_new_resource_signals(signals)
            except BaseException:
                pass
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            except BaseException:
                pass
        # A consumed attempt keeps both signals blocked on transition failure;
        # the pre-anchor path above instead restores its entry mask.
        raise ResourceEnvelopeFailure(
            "Candidate13 could not reserve the publication phase"
        ) from None


def _validate_publication_phase(seal: PublicationPhaseSeal) -> None:
    issued = _ISSUED_PUBLICATION_PHASES.get(id(seal))
    if (
        type(seal) is not PublicationPhaseSeal
        or seal._sentinel is not _PUBLICATION_PHASE_SEAL
        or issued is None
        or issued[0] is not seal
        or issued[1] is not seal._issuance_token
        or _current_signal_mask()
        != seal.prior_signal_mask | _resource_signals()
        or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
        or signal.getitimer(signal.ITIMER_PROF) != (0.0, 0.0)
        or signal.getsignal(signal.SIGALRM) is not seal.prior_alarm_handler
        or signal.getsignal(signal.SIGPROF) is not seal.prior_prof_handler
        or resource.getrlimit(resource.RLIMIT_AS)
        != seal.prior_address_space_limit
    ):
        raise ResourceEnvelopeFailure(
            "Candidate13 reserved publication phase is not intact"
        )

def publish_atomically(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    freshness: FreshOutputSeal,
    publication_phase: PublicationPhaseSeal,
) -> None:
    _validate_publication_phase(publication_phase)
    expected = frozenset(
        {
            "pool_adapter_stage_localizer_aggregate.json",
            "pool_adapter_stage_localizer_lock.json",
            "pool_adapter_stage_localizer_read_ledger.json",
        }
    )
    if (
        frozenset(payloads) != expected
        or any(not isinstance(data, bytes) or not data for data in payloads.values())
    ):
        raise ValueError("Candidate13 publication inventory is invalid")
    revalidate_fresh_output(freshness)
    candidate = _validate_output_namespace(output)
    if candidate != freshness.path:
        raise ValueError("Candidate13 publication differs from freshness seal")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate13 publication parent"
    )
    if parent_identity != freshness.parent_identity:
        os.close(parent_fd)
        raise RuntimeError("Candidate13 publication parent identity changed")
    digest = sha256_bytes(
        canonical_bytes(
            {name: sha256_bytes(data) for name, data in sorted(payloads.items())}
        )
    )
    staging_name = f".{candidate.name}.staging-{digest[:24]}"
    staging_identity: tuple[int, int] | None = None
    leaf_identities: dict[str, tuple[int, int]] = {}
    staging_fd: int | None = None
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging_metadata = os.stat(
            staging_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(staging_metadata.st_mode):
            raise RuntimeError("Candidate13 staging leaf is not a directory")
        staging_identity = _identity(staging_metadata)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        if _identity(os.fstat(staging_fd)) != staging_identity:
            raise RuntimeError("Candidate13 staging directory identity changed")
        for filename, data in sorted(payloads.items()):
            _validate_publication_phase(publication_phase)
            descriptor = os.open(
                filename,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
                dir_fd=staging_fd,
            )
            try:
                created = os.fstat(descriptor)
                visible = os.stat(filename, dir_fd=staging_fd, follow_symlinks=False)
                identity = _identity(created)
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_nlink != 1
                    or _identity(visible) != identity
                    or identity in leaf_identities.values()
                ):
                    raise RuntimeError("Candidate13 staged leaf identity mismatch")
                leaf_identities[filename] = identity
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Candidate13 publication write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(staging_fd, 0o500)
        os.fsync(staging_fd)
        _verify_owned_directory(
            parent_fd,
            name=staging_name,
            directory_identity=staging_identity,
            leaf_identities=leaf_identities,
            payloads=payloads,
        )
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate13 publication destination raced into existence")
        _validate_publication_phase(publication_phase)
        _renameat_noreplace(parent_fd, staging_name, candidate.name)
        os.fsync(parent_fd)
        visible_output = os.stat(
            candidate.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(visible_output) != staging_identity
            or _identity(os.lstat(candidate.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate13 published directory identity changed")
        _verify_owned_directory(
            parent_fd,
            name=candidate.name,
            directory_identity=staging_identity,
            leaf_identities=leaf_identities,
            payloads=payloads,
        )
        _validate_publication_phase(publication_phase)
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        # Never reselect a staging or final pathname for cleanup.  After any
        # publication failure, the residue is deliberately left in place and
        # blocks all retry.  This avoids unlinking or moving a same-UID foreign
        # replacement introduced after an inode check.
        os.close(parent_fd)


def _publication_residue_present(parent_fd: int, *, output_name: str) -> bool:
    prefixes = (
        f".{output_name}.staging-",
        f".{output_name}.quarantine-",
    )
    return any(name.startswith(prefixes) for name in os.listdir(parent_fd))


def _published_namespace_exact(
    freshness: FreshOutputSeal,
    payloads: Mapping[str, bytes],
) -> bool:
    candidate = _validate_output_namespace(freshness.path)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate13 final inventory parent"
    )
    directory_fd: int | None = None
    try:
        if parent_identity != freshness.parent_identity:
            raise RuntimeError("Candidate13 final inventory parent changed")
        try:
            visible = os.stat(
                candidate.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(visible.st_mode):
            raise RuntimeError("Candidate13 final inventory is not a directory")
        directory_identity = _identity(visible)
        directory_fd = os.open(
            candidate.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(opened) != directory_identity
            or set(os.listdir(directory_fd)) != set(payloads)
        ):
            raise RuntimeError("Candidate13 final inventory changed")
        for filename, expected in sorted(payloads.items()):
            descriptor = os.open(
                filename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or int(before.st_size) != len(expected)
                ):
                    raise RuntimeError("Candidate13 final leaf metadata changed")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                visible_leaf = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    b"".join(chunks) != expected
                    or _identity(after) != _identity(before)
                    or _identity(visible_leaf) != _identity(before)
                    or after.st_nlink != 1
                ):
                    raise RuntimeError("Candidate13 final leaf changed")
            finally:
                os.close(descriptor)
        final_visible = os.stat(
            candidate.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(os.fstat(directory_fd)) != directory_identity
            or _identity(final_visible) != directory_identity
            or _identity(os.lstat(candidate.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate13 final directory identity changed")
        return True
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def _audit_allowed_output_inventory(
    primary: FreshOutputSeal,
    fallback: FreshOutputSeal,
    payloads: Mapping[str, bytes],
    *,
    fallback_used: bool,
) -> str:
    # Resolve both fixed namespaces read-only after publication.  Primary
    # residue is permitted only when fallback publication was required; no
    # residue or foreign state is ever removed.
    primary_exact = _published_namespace_exact(primary, payloads)
    fallback_exact = _published_namespace_exact(fallback, payloads)
    states: list[tuple[FreshOutputSeal, bool]] = [
        (primary, primary_exact),
        (fallback, fallback_exact),
    ]
    residues: list[bool] = []
    for freshness, _exact in states:
        parent_fd, identity = _open_stable_directory(
            freshness.path.parent, role="Candidate13 inventory residue parent"
        )
        try:
            if identity != freshness.parent_identity:
                raise RuntimeError("Candidate13 inventory residue parent changed")
            residues.append(
                _publication_residue_present(
                    parent_fd, output_name=freshness.path.name
                )
            )
        finally:
            os.close(parent_fd)
    if (
        type(fallback_used) is not bool
        or (not fallback_used and (not primary_exact or fallback_exact or any(residues)))
        or (
            fallback_used
            and (
                not fallback_exact
                or residues[1]
            )
        )
    ):
        raise RuntimeError("Candidate13 allowed output inventory mismatch")
    return sha256_bytes(
        canonical_bytes(
            {
                "allowed_output_namespace_set_commitment_sha256": (
                    _allowed_output_namespace_set_commitment()
                ),
                "domain": "CANDIDATE13_FINAL_ALLOWED_OUTPUT_INVENTORY_V1",
                "fallback_final_exact": fallback_exact,
                "fallback_residue_present": residues[1],
                "fallback_used": fallback_used,
                "payload_inventory_commitment_sha256": sha256_bytes(
                    canonical_bytes(
                        {
                            name: sha256_bytes(data)
                            for name, data in sorted(payloads.items())
                        }
                    )
                ),
                "primary_final_exact": primary_exact,
                "primary_residue_present": residues[0],
            }
        )
    )


def _publish_with_fallback(
    payloads: Mapping[str, bytes],
    *,
    primary_freshness: FreshOutputSeal,
    fallback_freshness: FreshOutputSeal,
    publication_phase: PublicationPhaseSeal,
) -> bool:
    try:
        publish_atomically(
            OUTPUT_ROOT,
            payloads,
            freshness=primary_freshness,
            publication_phase=publication_phase,
        )
    except BaseException:
        _validate_publication_phase(publication_phase)
        publish_atomically(
            FALLBACK_OUTPUT_ROOT,
            payloads,
            freshness=fallback_freshness,
            publication_phase=publication_phase,
        )
        return True
    return False


def run_one_shot(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
) -> dict[str, Any]:
    freshness = prevalidate_fresh_output(OUTPUT_ROOT)
    fallback_freshness = prevalidate_fresh_output(FALLBACK_OUTPUT_ROOT)
    authorization = authorize_preflight_once(
        context,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        output_freshness=freshness,
        fallback_output_freshness=fallback_freshness,
    )
    qualification = qualify_source_controls(
        context, reader=authorization.reader, authority_prefix=True
    )
    expected_qualification = build_qualification_record(context, qualification)
    if (
        expected_qualification != authorization.qualification_bytes
        or qualification.control_binding_commitment_sha256
        != authorization.control_binding_commitment_sha256
    ):
        raise PermissionError("Candidate13 external authority does not bind controls")

    try:
        envelope = _arm_resource_envelope()
        _validate_resource_envelope(envelope)
    except WatchdogExpired:
        publication_phase = _enter_reserved_publication_phase(None)
        plan = PayloadPlan(
            consumption=None,
            outcome=None,
            failure_stage="WATCHDOG_TIMEOUT",
            anchor_durable_entry_created_or_partial=False,
            resource_envelope_enforced=False,
        )
        payloads = _materialize_payload_plan_in_publication_phase(
            context,
            authorization,
            qualification,
            plan,
            publication_phase,
        )
    except CpuTimerExpired:
        publication_phase = _enter_reserved_publication_phase(None)
        plan = PayloadPlan(
            consumption=None,
            outcome=None,
            failure_stage="CPU_TIMER_TIMEOUT",
            anchor_durable_entry_created_or_partial=False,
            resource_envelope_enforced=False,
        )
        payloads = _materialize_payload_plan_in_publication_phase(
            context,
            authorization,
            qualification,
            plan,
            publication_phase,
        )
    except ResourceEnvelopeFailure:
        publication_phase = _enter_reserved_publication_phase(None)
        plan = PayloadPlan(
            consumption=None,
            outcome=None,
            failure_stage="RESOURCE_ENVELOPE",
            anchor_durable_entry_created_or_partial=False,
            resource_envelope_enforced=False,
        )
        payloads = _materialize_payload_plan_in_publication_phase(
            context,
            authorization,
            qualification,
            plan,
            publication_phase,
        )
    else:
        progress = ConsumptionProgress()
        payloads: dict[str, bytes] | None = None
        try:
            plan = _compute_payload_plan(
                context,
                authorization,
                qualification,
                resource_envelope=envelope,
                progress=progress,
            )
            if plan.failure_stage not in {
                "WATCHDOG_TIMEOUT",
                "CPU_TIMER_TIMEOUT",
                "RESOURCE_ENVELOPE",
            }:
                payloads = _materialize_payload_plan_under_compute_envelope(
                    context,
                    authorization,
                    qualification,
                    plan,
                    envelope,
                )
        except WatchdogExpired:
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="WATCHDOG_TIMEOUT",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=True,
            )
        except CpuTimerExpired:
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="CPU_TIMER_TIMEOUT",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=True,
            )
        except ResourceEnvelopeFailure:
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="RESOURCE_ENVELOPE",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=False,
            )
        except BaseException:
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="PAYLOAD_MATERIALIZATION",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=True,
            )
        try:
            publication_phase = _enter_reserved_publication_phase(envelope)
        except WatchdogExpired:
            publication_phase = _enter_reserved_publication_phase(envelope)
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="WATCHDOG_TIMEOUT",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=True,
            )
            payloads = None
        except CpuTimerExpired:
            publication_phase = _enter_reserved_publication_phase(envelope)
            plan = PayloadPlan(
                consumption=progress.consumption,
                outcome=None,
                failure_stage="CPU_TIMER_TIMEOUT",
                anchor_durable_entry_created_or_partial=(
                    progress.anchor_durable_entry_created_or_partial
                ),
                resource_envelope_enforced=True,
            )
            payloads = None
        if payloads is None:
            payloads = _materialize_payload_plan_in_publication_phase(
                context,
                authorization,
                qualification,
                plan,
                publication_phase,
            )
    fallback_used = _publish_with_fallback(
        payloads,
        primary_freshness=freshness,
        fallback_freshness=fallback_freshness,
        publication_phase=publication_phase,
    )
    output_commitment = _audit_allowed_output_inventory(
        freshness,
        fallback_freshness,
        payloads,
        fallback_used=fallback_used,
    )
    public = _decode(
        payloads["pool_adapter_stage_localizer_aggregate.json"],
        name="Candidate13 public aggregate",
    )
    diagnostic_complete = _validate_published_aggregate(context, public)
    result = {
        "check_code": public["check_code"],
        "diagnostic_complete": diagnostic_complete,
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "output_inventory_commitment_sha256": output_commitment,
        "publication_fallback_used": fallback_used,
        "stage_passed_all_bitset": public.get(
            "stage_passed_all_bitset", "0000000000"
        ),
        "stage_reached_all_bitset": public.get(
            "stage_reached_all_bitset", "0000000000"
        ),
        "status": public["status"],
    }
    _assert_aggregate_safe(result)
    return result

def _bootstrap_cli_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"Candidate13 bootstrap requires exactly one {flag}")
    position = positions[0]
    if position + 1 >= len(arguments):
        raise ValueError(f"Candidate13 bootstrap requires a value for {flag}")
    value = arguments[position + 1]
    if not isinstance(value, str) or value.startswith("--"):
        raise ValueError(f"Candidate13 bootstrap has an invalid value for {flag}")
    return value


def _dispatch_through_verified_runner(
    argv: Sequence[str] | None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not all(type(value) is str for value in arguments):
        raise TypeError("Candidate13 bootstrap arguments must be exact strings")
    expected_sha = _bootstrap_cli_value(arguments, "--runner-sha256")
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("Candidate13 bootstrap runner SHA is invalid")
    size_text = _bootstrap_cli_value(arguments, "--runner-size-bytes")
    if not size_text.isdecimal() or int(size_text) <= 0:
        raise ValueError("Candidate13 bootstrap runner size is invalid")
    expected_size = int(size_text)

    runner_path = Path(__file__)
    data, _identity_value = _read_regular_file_once(
        runner_path,
        role="Candidate13 executing runner bootstrap",
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise PermissionError("Candidate13 executing runner byte binding mismatch")

    private_name = f"_candidate13_verified_runner_{expected_sha}"
    if private_name in sys.modules:
        raise PermissionError("Candidate13 verified runner is already loaded")
    outer_module = sys.modules.get(__name__)
    if type(outer_module) is not types.ModuleType:
        raise RuntimeError("Candidate13 outer runner module is unavailable")
    outer_namespace = object.__getattribute__(outer_module, "__dict__")
    if outer_namespace.get("_C13_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate13 runner bootstrap is not reentrant")

    launch_token = object()
    binding = types.MappingProxyType(
        {
            "module_name": private_name,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    )
    module = types.ModuleType(private_name)
    module.__file__ = str(runner_path)
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_C13_VERIFIED_RUNNER_BINDING"] = binding
    module.__dict__["_C13_VERIFIED_RUNNER_LAUNCH_TOKEN"] = launch_token
    module.__dict__["_C13_VERIFIED_RUNNER_OUTER_MODULE"] = outer_module
    outer_namespace["_C13_OUTER_LAUNCH_TOKEN"] = launch_token
    sys.modules[private_name] = module
    try:
        exec(
            compile(data, str(runner_path), "exec", dont_inherit=True),
            module.__dict__,
        )
        private_main = getattr(module, "main", None)
        if (
            not isinstance(private_main, types.FunctionType)
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != str(runner_path)
        ):
            raise PermissionError("Candidate13 verified runner entry point mismatch")
        return int(private_main(arguments))
    finally:
        sys.modules.pop(private_name, None)
        if outer_namespace.get("_C13_OUTER_LAUNCH_TOKEN") is launch_token:
            del outer_namespace["_C13_OUTER_LAUNCH_TOKEN"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--runner-size-bytes", required=True, type=int)
    parser.add_argument("--runner-tests-sha256", required=True)
    parser.add_argument("--runner-tests-size-bytes", required=True, type=int)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--source-control-check-only", action="store_true")
    modes.add_argument("--preflight-once", action="store_true")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--authorization-size-bytes", type=int)
    args = parser.parse_args(argv)
    if args.preflight_once and (
        args.authorization_sha256 is None or args.authorization_size_bytes is None
    ):
        parser.error("--preflight-once requires exact authorization SHA and size")
    if not args.preflight_once and (
        args.authorization_sha256 is not None or args.authorization_size_bytes is not None
    ):
        parser.error("authorization binding is valid only with --preflight-once")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    if _C13_VERIFIED_RUNNER_BINDING is None:
        try:
            return _dispatch_through_verified_runner(argv)
        except BaseException:
            print(
                json.dumps(
                    {
                        "check_code": "C13_RUNNER_REJECTED_OR_FAILED",
                        "diagnostic_complete": False,
                        "holdout_label_evaluator_metric_fit_opened_or_started": False,
                        "status": "FAIL_CLOSED",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
    _verified_runner_execution_binding()
    args = _parse_args(argv)
    try:
        context = bootstrap_verified_sources(
            runner_sha256=args.runner_sha256,
            runner_size_bytes=args.runner_size_bytes,
            runner_tests_sha256=args.runner_tests_sha256,
            runner_tests_size_bytes=args.runner_tests_size_bytes,
        )
        _validate_context(context)
        if args.source_check_only:
            value = source_check_result(context)
            exit_code = 0
        elif args.source_control_check_only:
            value = qualification_result(context)
            exit_code = (
                0
                if value.get("status") == "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
                else 1
            )
        else:
            value = run_one_shot(
                context,
                authorization_sha256=str(args.authorization_sha256),
                authorization_size_bytes=int(args.authorization_size_bytes),
            )
            exit_code = 0 if value.get("diagnostic_complete") is True else 1
    except BaseException:
        value = {
            "check_code": "C13_RUNNER_REJECTED_OR_FAILED",
            "diagnostic_complete": False,
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "status": "FAIL_CLOSED",
        }
        exit_code = 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
