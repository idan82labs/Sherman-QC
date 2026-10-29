#!/usr/bin/env python3
"""Guarded Candidate11 full-TRAIN18 replay diagnostic runner.

The repeatable modes authenticate only frozen source bytes or value-free
controls.  Only ``--diagnostic-once`` accepts an exact external authorization,
durably consumes it with a no-replace anchor before the first manifest open,
and then reads the exact attested 36 TRAIN records / 72 semantic-and-pool files.
Every record is independently adapted through Candidate9 and diagnosed through
the unchanged Candidate8 D01--D05 wrapper by the filesystem-blind Candidate11
core.  Published results are aggregate engineering evidence only.  Labels,
evaluator outputs, metrics, fitting, models, full-28 inputs, holdout inputs, and
adapted-cache publication are outside this runner's capability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py"
RUNNER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py"
C11_CORE_SOURCE = ROOT / "backend/f1_train18_candidate11_full_replay_diagnostic_v1.py"
C11_CORE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate11_full_replay_diagnostic_v1.py"
C10_CORE_SOURCE = ROOT / "backend/f1_train18_candidate10_full_replay_adapter_preflight_v1.py"
C10_CORE_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate10_full_replay_adapter_preflight_v1.py"
C10_RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate10_full_replay_adapter_preflight_once_v1.py"
C10_RUNNER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate10_full_replay_adapter_preflight_once_v1.py"
C9_ADAPTER_SOURCE = ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C9_ADAPTER_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C8_RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
C8_RUNTIME_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
C8_ADAPTER_SOURCE = ROOT / "scripts/preflight_f1_train18_candidate8_diagnostic_adapter_v1.py"
C8_ADAPTER_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate8_diagnostic_adapter_v1.py"
C7_RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py"
C7_RUNTIME_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate7_conservative_recall_recovery_v1.py"
C6_HYPOTHESIS_SOURCE = ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
C6_HYPOTHESIS_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate6_conservative_recall_recovery_v1.py"
BUILDER_SOURCE = ROOT / "scripts/build_f1_scan_only_candidate_pool.py"
CLUSTER_SOURCE = ROOT / "backend/f1_line_object_clustering.py"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
C7_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate7_terminal_disposition_v1.json"
C9_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate9_terminal_disposition_v1.json"
C10_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate10_terminal_disposition_v1.json"
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

C11_QUALIFICATION = ROOT / "research/f1_aug30_train18_candidate11_full_replay_diagnostic_qualification_v1.json"
C11_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate11_full_replay_diagnostic_authorization_v1.json"
C11_ANCHOR = ROOT / "research/f1_aug30_train18_candidate11_full_replay_diagnostic_consumption_v1.json"
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate11-full-replay-diagnostic-v1"

SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 11
ENGINEERING_ATTEMPT_ID = "CANDIDATE11_FULL_TRAIN_REPLAY_DIAGNOSTIC_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RECORD_COUNT = 36
PAIR_COUNT = 18
SOURCE_FILE_COUNT = 72

QUALIFICATION_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_qualification_v1"
AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_authorization_v1"
ANCHOR_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_consumption_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_runner_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_runner_terminal_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_read_ledger_v1"
LOCK_KIND = "sherman_qc_f1_aug30_train18_candidate11_full_replay_diagnostic_lock_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CUSTODY_VALIDATION_KIND = "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"

FIXED_SOURCE_BINDINGS = {
    "builder": (BUILDER_SOURCE, "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2", 57613),
    "c10_core": (C10_CORE_SOURCE, "3ef5a1c26ba138ff3d085e89483da706288d683797977fa5ad8bc47b9a135e53", 33019),
    "c10_core_tests": (C10_CORE_TESTS_SOURCE, "a612a5a6bd243daa6536bb1139d454f7435431291553c344ef20f72d975d5796", 26994),
    "c10_runner": (C10_RUNNER_SOURCE, "dc47d8612b9dbdc01fe7d5432cae76b69b2e696061e488125e2c27bce77db8b5", 164874),
    "c10_runner_tests": (C10_RUNNER_TESTS_SOURCE, "c07c4192316ce8452310cb83b54119b0f20046fc7a6cebb5f985b7db5c53daa9", 69723),
    "c11_core": (C11_CORE_SOURCE, "42c5ded051dd3b57dd48e3039a433afe63a0467edc13d4bd3b65620a6a279359", 32967),
    "c11_core_tests": (C11_CORE_TESTS_SOURCE, "1419579dd40ad01f73e21bc9ca8defb0523ff104e2eb43ddc81fe29a324c420d", 26644),
    "c6_hypothesis": (C6_HYPOTHESIS_SOURCE, "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252", 53837),
    "c6_hypothesis_tests": (C6_HYPOTHESIS_TESTS_SOURCE, "98df5af442627ce71bf539bb6a801cb30e3ad5b3a10f4e680f55a07c6e4c452f", 26864),
    "c7_runtime": (C7_RUNTIME_SOURCE, "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d", 28098),
    "c7_runtime_tests": (C7_RUNTIME_TESTS_SOURCE, "fbfb3bb91f68fadcf415f2da2a585125e8a95b6b7ffedf3a68879684792c1ee8", 12364),
    "c8_adapter": (C8_ADAPTER_SOURCE, "b91a112356dba3ed4aa9e3fe51fe36ea12bf93d2dd9ab8900f2b3461509211be", 14807),
    "c8_adapter_tests": (C8_ADAPTER_TESTS_SOURCE, "20bb70082daf1e681dd2ce3a1367da59befce4eebabcfd7a75944c5246091b8b", 14197),
    "c8_runtime": (C8_RUNTIME_SOURCE, "d878eb9141f287435890ed3290a45f40dd0730130585a6846c38431e3342f9b8", 21360),
    "c8_runtime_tests": (C8_RUNTIME_TESTS_SOURCE, "85c8b684dcaf4acaccb9800ba7bb7747c3546b24baae7d3c0685169431f52cfb", 14652),
    "c9_adapter": (C9_ADAPTER_SOURCE, "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096", 22354),
    "c9_adapter_tests": (C9_ADAPTER_TESTS_SOURCE, "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623", 16598),
    "cluster": (CLUSTER_SOURCE, "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c", 6817),
}

FIXED_CONTROLS = {
    "SEALED_SPLIT": (SPLIT, "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185", 10706),
    "ORIGINAL_PREREGISTRATION": (ORIGINAL_PREREG, "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a", 2651),
    "TRAIN_ALLOWLIST": (ALLOWLIST, "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0", 2428),
    "CANDIDATE7_TERMINAL_DISPOSITION": (C7_DISPOSITION, "1a64d3148d018b224a3a44d9d36d2620f1d87f18063e32df14050210aa03f56b", 3666),
    "CANDIDATE9_TERMINAL_DISPOSITION": (C9_DISPOSITION, "66ccbadbe49d5563355d407f8c319e665a20dd937b3506c74402fc2c97165ea2", 4992),
    "CANDIDATE10_TERMINAL_DISPOSITION": (C10_DISPOSITION, "eb6240e3b28149446f2a971e522c9ce73d880facf28a4496e528629cff03818c", 6234),
    "TRAIN_CACHE_ATTESTATION": (CACHE_ATTESTATION, "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71", 1834),
}

CACHE_MANIFEST_SHA256 = "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022"
CACHE_MANIFEST_SIZE_BYTES = 19601
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
V51_SCAN_ONLY_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

FORBIDDEN_PRELOADED = frozenset(
    {
        "backend.f1_train18_candidate11_full_replay_diagnostic_v1",
        "backend.f1_train18_candidate9_lossless_pool_alias_adapter_v1",
        "backend.f1_train18_candidate8_diagnostic_recall_wrapper_v1",
        "scripts.preflight_f1_train18_candidate8_diagnostic_adapter_v1",
        "backend.f1_train18_candidate7_conservative_recall_recovery_v1",
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

# A normal Python file executes before it can authenticate its own bytes.  The
# public entry point therefore uses a two-stage bootstrap: the initially loaded
# module reads and verifies the exact runner bytes supplied on the command line,
# compiles those bytes into a private module, and delegates to that module.  The
# injected objects below exist only in that private verified execution.  Direct
# imports/calls and post-import monkeypatches never receive them.
_C11_VERIFIED_RUNNER_BINDING = globals().get("_C11_VERIFIED_RUNNER_BINDING")
_C11_VERIFIED_RUNNER_LAUNCH_TOKEN = globals().get(
    "_C11_VERIFIED_RUNNER_LAUNCH_TOKEN"
)
_C11_VERIFIED_RUNNER_OUTER_MODULE = globals().get(
    "_C11_VERIFIED_RUNNER_OUTER_MODULE"
)
_C11_OUTER_LAUNCH_TOKEN = globals().get("_C11_OUTER_LAUNCH_TOKEN")


class RunnerFailure(RuntimeError):
    def __init__(self, stage: str, code: str) -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code


class AnchorCreationFailure(RuntimeError):
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
        raise TypeError("Candidate11 runner hashes immutable bytes only")
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
    binding = _C11_VERIFIED_RUNNER_BINDING
    module = sys.modules.get(__name__)
    outer_module = _C11_VERIFIED_RUNNER_OUTER_MODULE
    outer_namespace = (
        object.__getattribute__(outer_module, "__dict__")
        if type(outer_module) is types.ModuleType
        else {}
    )
    if (
        type(binding) is not types.MappingProxyType
        or _C11_VERIFIED_RUNNER_LAUNCH_TOKEN is None
        or type(outer_module) is not types.ModuleType
        or outer_namespace.get("_C11_OUTER_LAUNCH_TOKEN")
        is not _C11_VERIFIED_RUNNER_LAUNCH_TOKEN
        or not isinstance(module, types.ModuleType)
        or module is not sys.modules.get(str(binding.get("module_name", "")))
        or __name__ != binding.get("module_name")
        or __name__
        != f"_candidate11_verified_runner_{binding.get('sha256', '')}"
        or not isinstance(binding.get("sha256"), str)
        or len(str(binding.get("sha256"))) != 64
        or any(character not in "0123456789abcdef" for character in str(binding["sha256"]))
        or type(binding.get("size_bytes")) is not int
        or int(binding["size_bytes"]) <= 0
    ):
        raise PermissionError(
            "Candidate11 runner is not executing from verified private bytes"
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
        raise RuntimeError("Candidate11 requires a regular O_NOFOLLOW/O_DIRECTORY filesystem")
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
            raise RuntimeError("Candidate11 source inode verified before fd open")
        verified = True

    def read_start(_identity_value: tuple[int, int]) -> None:
        nonlocal started
        if not verified:
            raise RuntimeError("Candidate11 source read started before inode verification")
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
    current = sys.modules.get(__name__)
    launch_token = _C11_VERIFIED_RUNNER_LAUNCH_TOKEN
    outer_module = _C11_VERIFIED_RUNNER_OUTER_MODULE
    targets = {path.resolve(strict=True) for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in paths}
    if any(name in sys.modules for name in FORBIDDEN_PRELOADED):
        raise RuntimeError("Candidate11 rejects a preloaded controlled module")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        if module is outer_module:
            namespace = object.__getattribute__(module, "__dict__")
            if namespace.get("_C11_OUTER_LAUNCH_TOKEN") is launch_token:
                continue
            raise RuntimeError("Candidate11 outer launch token changed")
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
                f"Candidate11 rejects preloaded module subclass: {name}"
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
            raise RuntimeError(f"Candidate11 rejects preloaded source alias: {name}")


def _exec(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate11 private verified module is already loaded")
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
    c11_core: types.ModuleType
    c9_adapter: types.ModuleType
    c9_adapter_context: Any
    c8_runtime: types.ModuleType
    c8_adapter: types.ModuleType
    c8_adapter_context: Any
    c7_runtime: types.ModuleType
    c6_hypothesis: types.ModuleType
    builder_facade: types.ModuleType
    cluster_facade: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_entries: tuple[Mapping[str, Any], ...]
    source_commitment_sha256: str
    _controlled_callables: tuple[Any, ...]
    _function_snapshot: tuple[Any, ...]
    _constant_commitment_sha256: str
    _sentinel: object
    _issuance_token: object


def _constant_commitment(context: BootstrapContext) -> str:
    runner_binding = _verified_runner_execution_binding()
    return sha256_bytes(
        canonical_bytes(
            {
                "executing_runner": runner_binding,
                "c11": {
                    "attempt_ordinal": context.c11_core.ATTEMPT_ORDINAL,
                    "engineering_attempt_id": context.c11_core.ENGINEERING_ATTEMPT_ID,
                    "pass_code": context.c11_core.PASS_CODE,
                    "record_count": context.c11_core.RECORD_COUNT,
                    "pair_count": context.c11_core.PAIR_COUNT,
                    "source_file_count": context.c11_core.SOURCE_FILE_COUNT,
                },
                "c8": {
                    "attempt_ordinal": context.c8_runtime.ATTEMPT_ORDINAL,
                    "pass_code": context.c8_runtime.PASS_CODE,
                    "phase_contract": list(context.c8_runtime.PHASE_CONTRACT),
                    "failure_codes": dict(sorted(context.c8_runtime.FAILURE_CODES.items())),
                    "result_keys": sorted(context.c8_runtime.RESULT_KEYS),
                },
                "c9": {
                    "attempt_ordinal": context.c9_adapter.ATTEMPT_ORDINAL,
                    "scientific_hypothesis_id": context.c9_adapter.SCIENTIFIC_HYPOTHESIS_ID,
                    "representation_contract": dict(context.c9_adapter.REPRESENTATION_CONTRACT),
                },
                "manifest": {
                    "path": str(_lexical(CACHE_MANIFEST, role="manifest constant").relative_to(ROOT)),
                    "sha256": CACHE_MANIFEST_SHA256,
                    "size_bytes": CACHE_MANIFEST_SIZE_BYTES,
                },
                "source_commitment_sha256": context.source_commitment_sha256,
            }
        )
    )


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
        != _require_sha(runner_sha256, name="Candidate11 runner SHA")
        or runner_binding["size_bytes"]
        != _strict_int(runner_size_bytes, name="Candidate11 runner size")
    ):
        raise PermissionError(
            "Candidate11 executing runner does not match the requested bytes"
        )
    specs = {
        **FIXED_SOURCE_BINDINGS,
        "runner": (
            RUNNER_SOURCE,
            _require_sha(runner_sha256, name="Candidate11 runner SHA"),
            _strict_int(runner_size_bytes, name="Candidate11 runner size"),
        ),
        "runner_tests": (
            RUNNER_TESTS_SOURCE,
            _require_sha(runner_tests_sha256, name="Candidate11 runner tests SHA"),
            _strict_int(runner_tests_size_bytes, name="Candidate11 runner tests size"),
        ),
    }
    if specs["runner"][2] <= 0 or specs["runner_tests"][2] <= 0:
        raise ValueError("Candidate11 source sizes must be positive")
    _reject_preloaded(tuple(path for path, _sha, _size in specs.values()))
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha, size = specs[name]
        data, entry = _read_exact_source(
            path,
            role=f"CANDIDATE11_SOURCE_{name.upper()}",
            sha256=sha,
            size_bytes=size,
        )
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate11 source inventory contains an inode alias")
        identities.add(identity)
        source_bytes[name] = data
        entries.append(entry)
        bindings[name] = {
            "path": entry["path"],
            "sha256": sha,
            "size_bytes": size,
        }

    hypothesis = _exec(
        "_candidate11_verified_c6_hypothesis",
        C6_HYPOTHESIS_SOURCE,
        source_bytes["c6_hypothesis"],
        package="backend",
    )
    candidate7 = _exec(
        "_candidate11_verified_c7_runtime",
        C7_RUNTIME_SOURCE,
        source_bytes["c7_runtime"],
        package="backend",
    )
    c8_runtime = _exec(
        "_candidate11_verified_c8_runtime",
        C8_RUNTIME_SOURCE,
        source_bytes["c8_runtime"],
        package="backend",
    )
    c8_adapter = _exec(
        "_candidate11_verified_c8_adapter",
        C8_ADAPTER_SOURCE,
        source_bytes["c8_adapter"],
        package="scripts",
    )
    c9_adapter = _exec(
        "_candidate11_verified_c9_adapter",
        C9_ADAPTER_SOURCE,
        source_bytes["c9_adapter"],
        package="backend",
    )
    c11_core = _exec(
        "_candidate11_verified_full_replay_core",
        C11_CORE_SOURCE,
        source_bytes["c11_core"],
        package="backend",
    )
    c8_runtime.verify_scientific_hypothesis_runtime(
        hypothesis, source_sha256=bindings["c6_hypothesis"]["sha256"]
    )
    c8_runtime.verify_candidate7_wrapper_runtime(
        candidate7,
        source_sha256=bindings["c7_runtime"]["sha256"],
        hypothesis_runtime=hypothesis,
    )
    c8_adapter_context = c8_adapter.AdapterContext(
        runtime=c8_runtime,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7,
        source_bindings={
            "adapter": dict(bindings["c8_adapter"]),
            "adapter_tests": dict(bindings["c8_adapter_tests"]),
            "hypothesis": dict(bindings["c6_hypothesis"]),
            "candidate7_runtime": dict(bindings["c7_runtime"]),
            "runtime": dict(bindings["c8_runtime"]),
            "runtime_tests": dict(bindings["c8_runtime_tests"]),
        },
    )
    builder_proxy = types.ModuleType("_candidate11_authenticated_builder_proxy")
    builder_proxy._tag = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("builder execution is outside Candidate11 runtime")
    )
    cluster_proxy = types.ModuleType("_candidate11_authenticated_cluster_proxy")
    cluster_proxy.cluster_line_candidates = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("cluster execution is outside Candidate11 runtime"))
    c9_adapter_context = c9_adapter.make_verified_context(
        hypothesis=hypothesis,
        builder=builder_proxy,
        clusterer=cluster_proxy,
        hypothesis_source_sha256=bindings["c6_hypothesis"]["sha256"],
        builder_source_sha256=bindings["builder"]["sha256"],
        cluster_source_sha256=bindings["cluster"]["sha256"],
    )
    if (
        c11_core.ATTEMPT_ORDINAL != ATTEMPT_ORDINAL
        or c11_core.ENGINEERING_ATTEMPT_ID != ENGINEERING_ATTEMPT_ID
        or c11_core.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or c11_core.RECORD_COUNT != RECORD_COUNT
        or c11_core.PAIR_COUNT != PAIR_COUNT
        or c11_core.SOURCE_FILE_COUNT != SOURCE_FILE_COUNT
        or c9_adapter.ATTEMPT_ORDINAL != 9
        or c8_runtime.ATTEMPT_ORDINAL != 8
    ):
        raise ValueError("Candidate11 controlled runtime identity mismatch")
    source_commitment = sha256_bytes(canonical_bytes(_source_projection(bindings)))
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in bindings.items()
        }
    )
    immutable_entries = tuple(types.MappingProxyType(dict(entry)) for entry in entries)
    issuance_token = object()
    runner_module = sys.modules.get(__name__)
    if not isinstance(runner_module, types.ModuleType):
        raise RuntimeError("Candidate11 verified runner module is unavailable")
    context = BootstrapContext(
        runner_module=runner_module,
        c11_core=c11_core,
        c9_adapter=c9_adapter,
        c9_adapter_context=c9_adapter_context,
        c8_runtime=c8_runtime,
        c8_adapter=c8_adapter,
        c8_adapter_context=c8_adapter_context,
        c7_runtime=candidate7,
        c6_hypothesis=hypothesis,
        builder_facade=builder_proxy,
        cluster_facade=cluster_proxy,
        source_bindings=immutable_bindings,
        source_entries=immutable_entries,
        source_commitment_sha256=source_commitment,
        _controlled_callables=(
            c11_core.make_verified_context,
            c11_core.diagnose_all_records,
            c11_core._validate_public_result,
            c9_adapter.adapt_pool_bytes,
            c8_adapter.diagnose_bound_pair,
            c8_runtime._validate_result,
        ),
        _function_snapshot=_function_snapshot(
            runner_module,
            hypothesis,
            candidate7,
            c8_runtime,
            c8_adapter,
            c9_adapter,
            c11_core,
        ),
        _constant_commitment_sha256="",
        _sentinel=_CONTEXT_SEAL,
        _issuance_token=issuance_token,
    )
    object.__setattr__(context, "_constant_commitment_sha256", _constant_commitment(context))
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
        != sha256_bytes(canonical_bytes(_source_projection(context.source_bindings)))
        or context._controlled_callables
        != (
            context.c11_core.make_verified_context,
            context.c11_core.diagnose_all_records,
            context.c11_core._validate_public_result,
            context.c9_adapter.adapt_pool_bytes,
            context.c8_adapter.diagnose_bound_pair,
            context.c8_runtime._validate_result,
        )
        or context._function_snapshot
        != _function_snapshot(
            context.runner_module,
            context.c6_hypothesis,
            context.c7_runtime,
            context.c8_runtime,
            context.c8_adapter,
            context.c9_adapter,
            context.c11_core,
        )
        or context._constant_commitment_sha256 != _constant_commitment(context)
    ):
        raise PermissionError("Candidate11 bootstrap context is not intact")


def source_check_result(context: BootstrapContext) -> dict[str, Any]:
    _validate_context(context)
    return {
        "artifact_or_control_input_opened": False,
        "check_code": "C11_EXACT_SOURCE_BYTES_VERIFIED",
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
            _lexical(C11_AUTHORIZATION, role="Candidate11 authorization"),
            _lexical(C11_QUALIFICATION, role="Candidate11 qualification"),
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
                raise ValueError("Candidate11 bootstrap ledger contains an inode alias")
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
        raise PermissionError(f"Candidate11 read denied ({reason})")

    def install_artifact_inventory(
        self,
        specifications: Sequence[tuple[Path, str, str]],
    ) -> ArtifactInventory:
        if self.artifact_inventory is not None:
            self._deny(purpose="TRAIN_ARTIFACT_INVENTORY", reason="INVENTORY_ALREADY_INSTALLED")
        if len(specifications) != SOURCE_FILE_COUNT:
            raise ValueError("Candidate11 artifact inventory must contain exactly 72 files")
        bindings: dict[Path, ArtifactBinding] = {}
        identities = set(self.identities)
        for path, purpose, expected_sha in specifications:
            candidate = _lexical(path, role=purpose)
            if not _inside_cache_root(candidate) or candidate in bindings:
                raise ValueError("Candidate11 artifact inventory path is invalid or duplicated")
            if any(token in purpose.upper() for token in FORBIDDEN_PURPOSE_TOKENS):
                self._deny(purpose=purpose, reason="FORBIDDEN_PURPOSE")
            canonical = _reject_symlinks(candidate, role=purpose, require_exists=True)
            metadata = os.lstat(canonical)
            identity = _identity(metadata)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or int(metadata.st_size) <= 0
                or identity in identities
            ):
                raise ValueError("Candidate11 artifact metadata or inode inventory is invalid")
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
                    {"domain": "CANDIDATE11_EXACT_72_FILE_INVENTORY_V1", "files": projection}
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
        if expected_size <= 0:
            raise ValueError(f"{purpose} size must be positive")
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
            raise ValueError("Candidate11 input contains an inode alias")
        binding = {
            "identity": identity,
            "purpose": purpose,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
        prior = self.allowed.get(canonical)
        if prior is not None and prior != binding:
            raise ValueError("Candidate11 conflicting authorization")
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
            {"domain": "CANDIDATE11_EXACT_READ_INVENTORY_V1", "reads": projection}
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
    candidate7_disposition_bytes: bytes
    candidate9_disposition_bytes: bytes
    candidate10_disposition_bytes: bytes
    control_binding_commitment_sha256: str
    control_read_inventory_commitment_sha256: str
    control_raw_fd_opened_commitment_sha256: str
    control_verified_inode_commitment_sha256: str
    control_read_started_commitment_sha256: str
    _context_token: object
    _sentinel: object
    _issuance_token: object


def _validate_candidate9_disposition(value: Mapping[str, Any]) -> None:
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind")
        != "sherman_qc_f1_aug30_train18_candidate9_terminal_disposition_v1"
        or type(value.get("attempt_ordinal")) is not int
        or value.get("attempt_ordinal") != 9
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("status") != "PASS_ENGINEERING_DIAGNOSTIC_NONAUTHORIZING"
        or value.get("check_code") != "C8_ALL_DIAGNOSTIC_PHASES_PASSED"
        or value.get("candidate10_engineering_successor_authorized") is not True
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "C9_DISPOSITION_CONTRACT_MISMATCH"
        )


def _validate_candidate10_disposition(value: Mapping[str, Any]) -> None:
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("kind")
        != "sherman_qc_f1_aug30_train18_candidate10_terminal_disposition_v1"
        or type(value.get("attempt_ordinal")) is not int
        or value.get("attempt_ordinal") != 10
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("status") != "REJECTED_ENGINEERING_CORE_PREFLIGHT"
        or value.get("check_code") != "C10_CORE_PREFLIGHT_FAILED"
        or value.get("candidate11_diagnostic_engineering_authorized") is not True
        or value.get("candidate11_fit_metric_or_promotion_authorized") is not False
        or value.get("label_or_gold_opened") is not False
        or value.get("holdout_opened") is not False
        or value.get("metric_or_evaluator_started") is not False
        or value.get("model_or_fit_started") is not False
        or value.get("scientific_hypothesis_performance_tested") is not False
    ):
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "C10_DISPOSITION_CONTRACT_MISMATCH"
        )


def _control_entries(reader: GuardedReader) -> list[dict[str, Any]]:
    expected = set(_expected_control_purposes())
    return [entry for entry in reader.entries if str(entry["purpose"]) in expected]


def _control_commitment(reader: GuardedReader) -> str:
    entries = _control_entries(reader)
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE11_REPEATABLE_SOURCE_CONTROLS_V1",
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
            ["CANDIDATE11_EXTERNAL_AUTHORIZATION", "CANDIDATE11_QUALIFICATION_RECORD"]
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
    try:
        context.c8_runtime._validate_candidate7_disposition(
            controls_bytes["CANDIDATE7_TERMINAL_DISPOSITION"],
            expected_sha256=FIXED_CONTROLS["CANDIDATE7_TERMINAL_DISPOSITION"][1],
        )
    except Exception as error:
        raise RunnerFailure(
            "SOURCE_CONTROL_QUALIFICATION", "C7_DISPOSITION_CONTRACT_MISMATCH"
        ) from error
    _validate_candidate9_disposition(controls["CANDIDATE9_TERMINAL_DISPOSITION"])
    _validate_candidate10_disposition(controls["CANDIDATE10_TERMINAL_DISPOSITION"])

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
        candidate7_disposition_bytes=controls_bytes[
            "CANDIDATE7_TERMINAL_DISPOSITION"
        ],
        candidate9_disposition_bytes=controls_bytes[
            "CANDIDATE9_TERMINAL_DISPOSITION"
        ],
        candidate10_disposition_bytes=controls_bytes[
            "CANDIDATE10_TERMINAL_DISPOSITION"
        ],
        control_binding_commitment_sha256=_control_commitment(reader),
        control_read_inventory_commitment_sha256=_entry_commitment(control_entries),
        control_raw_fd_opened_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_CONTROL_RAW_FD_OPENED_V1", control_purposes
        ),
        control_verified_inode_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_CONTROL_VERIFIED_INODES_V1", control_purposes
        ),
        control_read_started_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_CONTROL_READ_STARTED_V1", control_purposes
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
        or sha256_bytes(qualification.candidate7_disposition_bytes)
        != FIXED_CONTROLS["CANDIDATE7_TERMINAL_DISPOSITION"][1]
        or sha256_bytes(qualification.candidate9_disposition_bytes)
        != FIXED_CONTROLS["CANDIDATE9_TERMINAL_DISPOSITION"][1]
        or sha256_bytes(qualification.candidate10_disposition_bytes)
        != FIXED_CONTROLS["CANDIDATE10_TERMINAL_DISPOSITION"][1]
        or qualification.control_binding_commitment_sha256
        != _control_commitment(qualification.reader)
        or qualification.control_read_inventory_commitment_sha256
        != _entry_commitment(control_entries)
        or qualification.control_raw_fd_opened_commitment_sha256
        != _purpose_commitment("CANDIDATE11_CONTROL_RAW_FD_OPENED_V1", control_purposes)
        or qualification.control_verified_inode_commitment_sha256
        != _purpose_commitment("CANDIDATE11_CONTROL_VERIFIED_INODES_V1", control_purposes)
        or qualification.control_read_started_commitment_sha256
        != _purpose_commitment("CANDIDATE11_CONTROL_READ_STARTED_V1", control_purposes)
        or qualification.manifest_path != _lexical(CACHE_MANIFEST, role="manifest")
        or qualification.manifest_sha256 != CACHE_MANIFEST_SHA256
        or qualification.manifest_size_bytes != CACHE_MANIFEST_SIZE_BYTES
    ):
        raise PermissionError("Candidate11 source-control qualification is not intact")


def _validate_preanchor_reader_state(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
) -> None:
    reader = qualification.reader
    expected_purposes = [str(entry["purpose"]) for entry in context.source_entries]
    expected_purposes.extend(
        ["CANDIDATE11_EXTERNAL_AUTHORIZATION", "CANDIDATE11_QUALIFICATION_RECORD"]
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
    ):
        raise PermissionError("Candidate11 pre-anchor reader state is not exact")


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


def _manifest_contract_commitment(path: Path, sha256: str, size_bytes: int) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE11_FIXED_MANIFEST_CONTRACT_V1",
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
            "check_code": "ALL_C11_SOURCE_CONTROL_CHECKS_PASSED",
            "consumption_anchor_namespace_commitment_sha256": _namespace_commitment(
                "CANDIDATE11_CONSUMPTION_ANCHOR_NAMESPACE_V1", C11_ANCHOR
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
            "output_namespace_commitment_sha256": _namespace_commitment(
                "CANDIDATE11_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT
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
            "check_code": "ALL_C11_SOURCE_CONTROL_CHECKS_PASSED",
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "failure_stage": "NONE",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "qualification_record_commitment_sha256": sha256_bytes(record),
            "repeatable": True,
            "source_inventory_commitment_sha256": context.source_commitment_sha256,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        }
    except BaseException:
        return {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "C11_SOURCE_CONTROL_QUALIFICATION_FAILED",
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
        role="Candidate11 output namespace",
    )
    frozen = _lexical(
        ROOT
        / "output/research/f1-aug30-train18-candidate11-full-replay-diagnostic-v1",
        role="frozen Candidate11 output namespace",
    )
    if candidate != frozen or candidate == candidate.parent:
        raise ValueError("Candidate11 output namespace is not frozen")
    return candidate


def _assert_no_publication_residue(parent_fd: int, *, output_name: str) -> None:
    prefixes = (
        f".{output_name}.staging-",
        f".{output_name}.quarantine-",
    )
    if any(name.startswith(prefixes) for name in os.listdir(parent_fd)):
        raise FileExistsError("Candidate11 publication residue already exists")


def prevalidate_fresh_output(path: Path | None = None) -> FreshOutputSeal:
    candidate = _validate_output_namespace(path)
    _reject_symlinks(candidate.parent, role="Candidate11 output parent", require_exists=True)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate11 output parent"
    )
    try:
        _assert_no_publication_residue(parent_fd, output_name=candidate.name)
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate11 output namespace already exists")
    finally:
        os.close(parent_fd)
    return FreshOutputSeal(
        path=candidate, parent_identity=parent_identity, _sentinel=_CONTEXT_SEAL
    )


def revalidate_fresh_output(seal: FreshOutputSeal) -> None:
    if type(seal) is not FreshOutputSeal or seal._sentinel is not _CONTEXT_SEAL:
        raise PermissionError("Candidate11 output freshness seal is invalid")
    parent_fd, identity = _open_stable_directory(
        seal.path.parent, role="Candidate11 output freshness parent"
    )
    try:
        if identity != seal.parent_identity:
            raise RuntimeError("Candidate11 output parent identity changed")
        _assert_no_publication_residue(parent_fd, output_name=seal.path.name)
        try:
            os.stat(seal.path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError("Candidate11 output namespace is no longer fresh")
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
            "CANDIDATE11_CONSUMPTION_ANCHOR_NAMESPACE_V1", C11_ANCHOR
        ),
        "control_binding_commitment_sha256": _require_sha(
            control_binding_commitment_sha256,
            name="Candidate11 authorization control commitment",
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
        "output_namespace_commitment_sha256": _namespace_commitment(
            "CANDIDATE11_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT
        ),
        "qualification_record_commitment_sha256": _require_sha(
            qualification_record_sha256,
            name="Candidate11 authorization qualification commitment",
        ),
        "retry_after_anchor_authorized": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "AUTHORIZED_ONCE",
    }
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
                "domain": "CANDIDATE11_ISSUED_AUTHORIZATION_SEAL_V1",
                "entries": [dict(entry) for entry in authority_entries],
                "output_parent_identity": list(
                    authorization.output_freshness.parent_identity
                ),
                "output_path": str(authorization.output_freshness.path.relative_to(ROOT)),
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
        "output_namespace_commitment_sha256",
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
        or value.get("check_code") != "ALL_C11_SOURCE_CONTROL_CHECKS_PASSED"
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
            "CANDIDATE11_CONSUMPTION_ANCHOR_NAMESPACE_V1", C11_ANCHOR
        )
        or value.get("output_namespace_commitment_sha256")
        != _namespace_commitment("CANDIDATE11_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT)
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
        raise PermissionError("Candidate11 qualification record contract mismatch")
    for key in (
        "consumption_anchor_namespace_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_read_inventory_commitment_sha256",
        "control_read_started_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "output_namespace_commitment_sha256",
    ):
        _require_sha(value.get(key), name=f"Candidate11 qualification {key}")
    _assert_aggregate_safe(value)


def authorize_diagnostic_once(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
    output_freshness: FreshOutputSeal,
) -> AuthorizationSeal:
    _validate_context(context)
    revalidate_fresh_output(output_freshness)
    reader = GuardedReader.from_context(context)
    auth_bytes, auth = _read_bound(
        reader,
        path=C11_AUTHORIZATION,
        purpose="CANDIDATE11_EXTERNAL_AUTHORIZATION",
        sha256=_require_sha(authorization_sha256, name="Candidate11 authorization SHA"),
        size_bytes=_strict_int(
            authorization_size_bytes, name="Candidate11 authorization size"
        ),
    )
    qualification_path = _lexical(
        C11_QUALIFICATION, role="CANDIDATE11_QUALIFICATION_RECORD"
    )
    qualification_sha = _require_sha(
        auth.get("qualification_record_commitment_sha256"),
        name="Candidate11 qualification record commitment",
    )
    qualification_metadata = os.lstat(
        _reject_symlinks(
            qualification_path,
            role="CANDIDATE11_QUALIFICATION_RECORD",
            require_exists=True,
        )
    )
    qualification_size = int(qualification_metadata.st_size)
    if not stat.S_ISREG(qualification_metadata.st_mode) or qualification_size <= 0:
        raise PermissionError("Candidate11 qualification record is not a regular file")
    qualification_bytes, qualification_value = _read_bound(
        reader,
        path=qualification_path,
        purpose="CANDIDATE11_QUALIFICATION_RECORD",
        sha256=qualification_sha,
        size_bytes=qualification_size,
    )
    control_commitment = _require_sha(
        auth.get("control_binding_commitment_sha256"),
        name="Candidate11 authorization control commitment",
    )
    expected_auth = build_authorization_contract(
        context,
        qualification_record_sha256=qualification_sha,
        control_binding_commitment_sha256=control_commitment,
    )
    if auth != expected_auth or sha256_bytes(auth_bytes) != authorization_sha256:
        raise PermissionError("Candidate11 external authorization contract mismatch")
    _qualification_static_contract(
        context, qualification_value, control_commitment=control_commitment
    )
    source_count = len(context.source_entries)
    authority_entries = reader.entries[source_count:]
    authority_purposes = [str(entry["purpose"]) for entry in authority_entries]
    expected_purposes = [
        "CANDIDATE11_EXTERNAL_AUTHORIZATION",
        "CANDIDATE11_QUALIFICATION_RECORD",
    ]
    full_expected = [str(entry["purpose"]) for entry in context.source_entries] + expected_purposes
    if (
        authority_purposes != expected_purposes
        or reader.raw_fd_opened_purposes != full_expected
        or reader.verified_inode_purposes != full_expected
        or reader.read_started_purposes != full_expected
        or reader.denied_attempts
    ):
        raise PermissionError("Candidate11 authorization read inventory mismatch")
    anchor_parent_fd, anchor_parent_identity = _open_stable_directory(
        C11_ANCHOR.parent, role="Candidate11 anchor parent authorization pin"
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
        authority_read_inventory_commitment_sha256=_entry_commitment(authority_entries),
        authority_raw_fd_opened_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_AUTHORITY_RAW_FD_OPENED_V1", authority_purposes
        ),
        authority_verified_inode_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_AUTHORITY_VERIFIED_INODES_V1", authority_purposes
        ),
        authority_read_started_commitment_sha256=_purpose_commitment(
            "CANDIDATE11_AUTHORITY_READ_STARTED_V1", authority_purposes
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
        raise PermissionError("Candidate11 external authorization seal is not intact")
    try:
        source_count = len(context.source_entries)
        authority_entries = authorization.reader.entries[
            source_count : source_count + 2
        ]
        purposes = [str(entry["purpose"]) for entry in authority_entries]
        expected_purposes = [
            "CANDIDATE11_EXTERNAL_AUTHORIZATION",
            "CANDIDATE11_QUALIFICATION_RECORD",
        ]
        expected_prefix = [
            str(entry["purpose"]) for entry in context.source_entries
        ] + expected_purposes
        authorization_value = _decode(
            authorization.authorization_bytes,
            name="Candidate11 issued authorization",
        )
        qualification_value = _decode(
            authorization.qualification_bytes,
            name="Candidate11 issued qualification",
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
                C11_AUTHORIZATION,
                "CANDIDATE11_EXTERNAL_AUTHORIZATION",
                authorization.authorization_sha256,
                authorization.authorization_size_bytes,
            ),
            (
                authority_entries[1],
                C11_QUALIFICATION,
                "CANDIDATE11_QUALIFICATION_RECORD",
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
                "CANDIDATE11_AUTHORITY_RAW_FD_OPENED_V1", purposes
            )
            or authorization.authority_verified_inode_commitment_sha256
            != _purpose_commitment(
                "CANDIDATE11_AUTHORITY_VERIFIED_INODES_V1", purposes
            )
            or authorization.authority_read_started_commitment_sha256
            != _purpose_commitment(
                "CANDIDATE11_AUTHORITY_READ_STARTED_V1", purposes
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
            raise ValueError("Candidate11 issued authorization changed")
    except Exception:
        raise PermissionError(
            "Candidate11 external authorization seal is not intact"
        ) from None
    parent_fd, parent_identity = _open_stable_directory(
        C11_ANCHOR.parent, role="Candidate11 anchor parent authorization revalidation"
    )
    os.close(parent_fd)
    if parent_identity != authorization.anchor_parent_identity:
        raise PermissionError("Candidate11 anchor parent changed after authorization")
    revalidate_fresh_output(authorization.output_freshness)


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


def consume_authorization(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
) -> ConsumptionSeal:
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
        raise PermissionError("Candidate11 authority and qualification do not bind")
    _validate_preanchor_reader_state(context, authorization, qualification)
    anchor = _lexical(C11_ANCHOR, role="Candidate11 consumption anchor")
    frozen = _lexical(
        ROOT
        / "research/f1_aug30_train18_candidate11_full_replay_diagnostic_consumption_v1.json",
        role="frozen Candidate11 consumption anchor",
    )
    if anchor != frozen:
        raise ValueError("Candidate11 consumption anchor namespace is not frozen")
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
        "output_namespace_commitment_sha256": _namespace_commitment(
            "CANDIDATE11_OUTPUT_NAMESPACE_V1", OUTPUT_ROOT
        ),
        "qualification_record_commitment_sha256": authorization.qualification_sha256,
        "retry_authorized": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "CONSUMED_BEFORE_MANIFEST_OPEN",
    }
    _assert_aggregate_safe(value)
    data = canonical_bytes(value)
    parent_fd, parent_identity = _open_stable_directory(
        anchor.parent, role="Candidate11 anchor parent"
    )
    if parent_identity != authorization.anchor_parent_identity:
        os.close(parent_fd)
        raise PermissionError("Candidate11 anchor parent changed before O_EXCL")
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
        os.fsync(parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate11 anchor write made no progress")
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
            raise RuntimeError("Candidate11 anchor publication mismatch")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise PermissionError("Candidate11 external authorization is already consumed") from error
    except BaseException as error:
        if created:
            raise AnchorCreationFailure(
                "Candidate11 anchor failed after durable consumption"
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
                    "Candidate11 anchor close failed after durable consumption"
                ) from close_error
            raise close_error
    if file_identity is None:
        raise AnchorCreationFailure("Candidate11 anchor lacks an owned identity")
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
            "Candidate11 anchor final validation failed after durable consumption"
        ) from error
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
        raise PermissionError("Candidate11 consumption anchor seal is invalid")
    parent_fd, parent_identity = _open_stable_directory(
        seal.path.parent, role="Candidate11 anchor parent revalidation"
    )
    descriptor: int | None = None
    try:
        if parent_identity != seal.parent_identity:
            raise RuntimeError("Candidate11 anchor parent changed")
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
            raise RuntimeError("Candidate11 anchor metadata changed")
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
            raise RuntimeError("Candidate11 anchor bytes or visibility changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


@dataclass(frozen=True)
class ManifestRecord:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    semantic_path: Path
    semantic_sha256: str
    pool_path: Path
    pool_sha256: str
    manifest_record_binding_sha256: str


@dataclass(frozen=True)
class PendingCoreRecord:
    ordinal: int
    pair_ordinal: int
    pass_name: str
    manifest_record_binding_sha256: str
    semantic_file_binding_sha256: str
    pool_file_binding_sha256: str
    semantic_sha256: str
    pool_sha256: str
    semantic_size_bytes: int
    pool_size_bytes: int
    semantic_bytes: bytes
    pool_bytes: bytes


@dataclass(frozen=True)
class ExecutionOutcome:
    core_result: Mapping[str, Any]
    core_result_bytes: bytes
    manifest_binding_sha256: str
    artifact_inventory_commitment_sha256: str


def _manifest_file_binding(entry: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "device_id": _strict_int(entry["device_id"], name="manifest device"),
                "domain": "CANDIDATE11_ATTESTED_MANIFEST_FILE_V1",
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
                "domain": "CANDIDATE11_ATTESTED_ARTIFACT_FILE_V1",
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
        set(manifest) != expected_keys
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
        raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_MANIFEST_HEADER_MISMATCH")
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != RECORD_COUNT:
        raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_MANIFEST_RECORD_COUNT_MISMATCH")
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
    paths: set[Path] = set()
    primary_hashes: dict[str, tuple[str, str]] = {}
    bindings: set[str] = set()
    records: list[ManifestRecord] = []
    for ordinal, (raw, expected_pair) in enumerate(zip(raw_records, expected_pairs)):
        if not isinstance(raw, dict) or set(raw) != record_keys:
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_SCHEMA_MISMATCH")
        case_id, pass_name = expected_pair
        if (str(raw.get("case_id")), str(raw.get("pass"))) != expected_pair:
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_MEMBERSHIP_ORDER_MISMATCH")
        runtime_id = str(raw.get("runtime_case_id") or "")
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id
            or case_by_runtime.setdefault(runtime_id, case_id) != case_id
            or _require_sha(raw.get("scan_sha256"), name="manifest scan SHA")
            != qualification.scan_by_case[case_id]
        ):
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_IDENTITY_MISMATCH")
        semantic_relative = Path(str(raw.get("semantic_path") or ""))
        pool_relative = Path(str(raw.get("pool_path") or ""))
        if (
            semantic_relative.is_absolute()
            or pool_relative.is_absolute()
            or ".." in semantic_relative.parts
            or ".." in pool_relative.parts
        ):
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_PATH_MISMATCH")
        semantic_path = _lexical(ROOT / semantic_relative, role="TRAIN semantic")
        pool_path = _lexical(ROOT / pool_relative, role="TRAIN pool")
        expected_semantic = _lexical(
            ALLOWED_CACHE_ROOT / pass_name / runtime_id / "semantic_result.json",
            role="expected TRAIN semantic",
        )
        expected_pool = _lexical(
            ALLOWED_CACHE_ROOT / pass_name / runtime_id / "pool" / f"{runtime_id}.json",
            role="expected TRAIN pool",
        )
        if (
            not _inside_cache_root(semantic_path)
            or not _inside_cache_root(pool_path)
            or semantic_path != expected_semantic
            or pool_path != expected_pool
            or semantic_path in paths
            or pool_path in paths
        ):
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "CACHE_RECORD_PATH_MISMATCH")
        paths.update((semantic_path, pool_path))
        semantic_sha = _require_sha(raw.get("semantic_sha256"), name="semantic SHA")
        pool_sha = _require_sha(raw.get("pool_sha256"), name="pool SHA")
        if pass_name == "primary":
            primary_hashes[case_id] = semantic_sha, pool_sha
        elif primary_hashes.get(case_id) != (semantic_sha, pool_sha):
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "PRIMARY_REPLAY_HASH_MISMATCH")
        record_binding = sha256_bytes(
            canonical_bytes(
                {
                    "domain": "CANDIDATE11_ATTESTED_MANIFEST_RECORD_V1",
                    "manifest_binding_sha256": manifest_binding_sha256,
                    "ordinal": ordinal,
                    "row": raw,
                }
            )
        )
        if record_binding in bindings:
            raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "DUPLICATE_RECORD_BINDING")
        bindings.add(record_binding)
        records.append(
            ManifestRecord(
                ordinal=ordinal,
                pair_ordinal=ordinal // 2,
                pass_name=pass_name,
                semantic_path=semantic_path,
                semantic_sha256=semantic_sha,
                pool_path=pool_path,
                pool_sha256=pool_sha,
                manifest_record_binding_sha256=record_binding,
            )
        )
    if len(paths) != SOURCE_FILE_COUNT or len(bindings) != RECORD_COUNT:
        raise RunnerFailure("CACHE_MANIFEST_VALIDATION", "EXACT_MANIFEST_INVENTORY_MISMATCH")
    return tuple(records)


def _core_source_bindings(context: BootstrapContext) -> dict[str, str]:
    return {
        "builder": str(context.source_bindings["builder"]["sha256"]),
        "c10_core": str(context.source_bindings["c10_core"]["sha256"]),
        "c10_core_tests": str(context.source_bindings["c10_core_tests"]["sha256"]),
        "c10_runner": str(context.source_bindings["c10_runner"]["sha256"]),
        "c10_runner_tests": str(context.source_bindings["c10_runner_tests"]["sha256"]),
        "c6_hypothesis": str(context.source_bindings["c6_hypothesis"]["sha256"]),
        "c7_runtime": str(context.source_bindings["c7_runtime"]["sha256"]),
        "c8_adapter": str(context.source_bindings["c8_adapter"]["sha256"]),
        "c8_runtime": str(context.source_bindings["c8_runtime"]["sha256"]),
        "c9_adapter": str(context.source_bindings["c9_adapter"]["sha256"]),
        "c9_adapter_tests": str(context.source_bindings["c9_adapter_tests"]["sha256"]),
        "cluster": str(context.source_bindings["cluster"]["sha256"]),
    }


def _expected_complete_purposes(
    context: BootstrapContext,
    records: Sequence[ManifestRecord],
) -> list[str]:
    expected = [str(entry["purpose"]) for entry in context.source_entries]
    expected.extend(
        ["CANDIDATE11_EXTERNAL_AUTHORIZATION", "CANDIDATE11_QUALIFICATION_RECORD"]
    )
    expected.extend(_expected_control_purposes())
    expected.append("TRAIN_CACHE_MANIFEST")
    for record in records:
        prefix = f"TRAIN_RECORD_{record.ordinal:02d}_{record.pass_name.upper()}"
        expected.extend((f"{prefix}_SEMANTIC", f"{prefix}_POOL"))
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
        observed != expected
        or reader.raw_fd_opened_purposes != expected
        or reader.verified_inode_purposes != expected
        or reader.read_started_purposes != expected
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or reader.denied_attempts
        or reader.artifact_inventory is None
        or len(reader.artifact_inventory.bindings) != SOURCE_FILE_COUNT
    ):
        raise RunnerFailure("EXACT_READ_INVENTORY", "EXACT_READ_INVENTORY_MISMATCH")


def read_adapt_diagnose_full_inventory(
    context: BootstrapContext,
    qualification: ControlQualification,
    consumption: ConsumptionSeal,
) -> ExecutionOutcome:
    _validate_context(context)
    _validate_qualification(context, qualification)
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
        manifest_bytes = reader.read_bytes(
            qualification.manifest_path, purpose="TRAIN_CACHE_MANIFEST"
        )
        revalidate_anchor(context, consumption)
        manifest = _decode(manifest_bytes, name="TRAIN_CACHE_MANIFEST")
        manifest_entry = reader.entries[-1]
        manifest_binding = _manifest_file_binding(manifest_entry)
    except BaseException:
        raise RunnerFailure(
            "CACHE_MANIFEST_READ", "C11_CACHE_MANIFEST_READ_FAILED"
        ) from None
    try:
        records = _validate_full_manifest(
            manifest,
            qualification,
            manifest_binding_sha256=manifest_binding,
        )
    except RunnerFailure:
        raise
    except BaseException:
        raise RunnerFailure(
            "CACHE_MANIFEST_VALIDATION", "C11_CACHE_MANIFEST_VALIDATION_FAILED"
        ) from None
    try:
        revalidate_anchor(context, consumption)
        specifications: list[tuple[Path, str, str]] = []
        for record in records:
            prefix = f"TRAIN_RECORD_{record.ordinal:02d}_{record.pass_name.upper()}"
            specifications.extend(
                (
                    (record.semantic_path, f"{prefix}_SEMANTIC", record.semantic_sha256),
                    (record.pool_path, f"{prefix}_POOL", record.pool_sha256),
                )
            )
        artifact_inventory = reader.install_artifact_inventory(specifications)
        revalidate_anchor(context, consumption)
        pending_records: list[PendingCoreRecord] = []
        for record in records:
            revalidate_anchor(context, consumption)
            prefix = f"TRAIN_RECORD_{record.ordinal:02d}_{record.pass_name.upper()}"
            semantic_purpose = f"{prefix}_SEMANTIC"
            pool_purpose = f"{prefix}_POOL"
            semantic_binding = artifact_inventory.bindings[record.semantic_path]
            pool_binding = artifact_inventory.bindings[record.pool_path]
            reader.authorize(
                record.semantic_path,
                purpose=semantic_purpose,
                sha256=record.semantic_sha256,
                size_bytes=semantic_binding.size_bytes,
                artifact_inventory=artifact_inventory,
            )
            revalidate_anchor(context, consumption)
            semantic_bytes = reader.read_bytes(
                record.semantic_path, purpose=semantic_purpose
            )
            revalidate_anchor(context, consumption)
            semantic_entry = reader.entries[-1]
            reader.authorize(
                record.pool_path,
                purpose=pool_purpose,
                sha256=record.pool_sha256,
                size_bytes=pool_binding.size_bytes,
                artifact_inventory=artifact_inventory,
            )
            revalidate_anchor(context, consumption)
            pool_bytes = reader.read_bytes(record.pool_path, purpose=pool_purpose)
            revalidate_anchor(context, consumption)
            pool_entry = reader.entries[-1]
            pending_records.append(
                PendingCoreRecord(
                    ordinal=record.ordinal,
                    pair_ordinal=record.pair_ordinal,
                    pass_name=record.pass_name,
                    manifest_record_binding_sha256=record.manifest_record_binding_sha256,
                    semantic_file_binding_sha256=_artifact_file_binding(
                        semantic_entry,
                        manifest_record_binding_sha256=record.manifest_record_binding_sha256,
                    ),
                    pool_file_binding_sha256=_artifact_file_binding(
                        pool_entry,
                        manifest_record_binding_sha256=record.manifest_record_binding_sha256,
                    ),
                    semantic_sha256=record.semantic_sha256,
                    pool_sha256=record.pool_sha256,
                    semantic_size_bytes=len(semantic_bytes),
                    pool_size_bytes=len(pool_bytes),
                    semantic_bytes=semantic_bytes,
                    pool_bytes=pool_bytes,
                )
            )
        revalidate_anchor(context, consumption)
    except BaseException:
        raise RunnerFailure(
            "ARTIFACT_INVENTORY_OR_READ", "C11_ARTIFACT_INVENTORY_OR_READ_FAILED"
        ) from None
    try:
        _validate_complete_inventory(context, qualification, records)
    except BaseException:
        raise RunnerFailure(
            "EXACT_READ_INVENTORY", "C11_EXACT_READ_INVENTORY_FAILED"
        ) from None
    try:
        # No Candidate11-core callable is touched until the exact manifest plus
        # all 72 semantic/pool reads and their fd/inode/read-start vectors have
        # passed the complete-inventory gate above.
        core_records = tuple(
            context.c11_core.RecordBytes(
                ordinal=record.ordinal,
                pair_ordinal=record.pair_ordinal,
                pass_name=record.pass_name,
                manifest_record_binding_sha256=(
                    record.manifest_record_binding_sha256
                ),
                semantic_file_binding_sha256=record.semantic_file_binding_sha256,
                pool_file_binding_sha256=record.pool_file_binding_sha256,
                semantic_sha256=record.semantic_sha256,
                pool_sha256=record.pool_sha256,
                semantic_size_bytes=record.semantic_size_bytes,
                pool_size_bytes=record.pool_size_bytes,
                semantic_bytes=record.semantic_bytes,
                pool_bytes=record.pool_bytes,
            )
            for record in pending_records
        )
        c11_context = context.c11_core.make_verified_context(
            c9_adapter=context.c9_adapter,
            c9_adapter_context=context.c9_adapter_context,
            c8_adapter=context.c8_adapter,
            c8_adapter_context=context.c8_adapter_context,
            c8_runtime=context.c8_runtime,
            candidate7_disposition_bytes=qualification.candidate7_disposition_bytes,
            candidate10_disposition_bytes=qualification.candidate10_disposition_bytes,
            source_bindings=_core_source_bindings(context),
        )
        core_result = context.c11_core.diagnose_all_records(
            c11_context,
            core_records,
            manifest_binding_sha256=manifest_binding,
        )
        context.c11_core._validate_public_result(core_result)
        core_bytes = context.c11_core.canonical_bytes(core_result)
        revalidate_anchor(context, consumption)
    except BaseException:
        raise RunnerFailure(
            "CORE_DIAGNOSTIC_INVOCATION", "C11_CORE_DIAGNOSTIC_INVOCATION_FAILED"
        ) from None
    return ExecutionOutcome(
        core_result=types.MappingProxyType(dict(core_result)),
        core_result_bytes=core_bytes,
        manifest_binding_sha256=manifest_binding,
        artifact_inventory_commitment_sha256=artifact_inventory.commitment_sha256,
    )


ENGINEERING_FAILURE_CODES = {
    "ANCHOR_CONSUMPTION": "C11_ANCHOR_CONSUMPTION_FAILED",
    "CACHE_MANIFEST_READ": "C11_CACHE_MANIFEST_READ_FAILED",
    "CACHE_MANIFEST_VALIDATION": "C11_CACHE_MANIFEST_VALIDATION_FAILED",
    "ARTIFACT_INVENTORY_OR_READ": "C11_ARTIFACT_INVENTORY_OR_READ_FAILED",
    "EXACT_READ_INVENTORY": "C11_EXACT_READ_INVENTORY_FAILED",
    "CORE_DIAGNOSTIC_INVOCATION": "C11_CORE_DIAGNOSTIC_INVOCATION_FAILED",
}


def _assert_aggregate_safe(value: Any) -> None:
    """Reject every non-scalar or data-bearing public aggregate shape."""

    forbidden_key_fragments = (
        "case_id",
        "count",
        "exception",
        "identifier",
        "ordinal",
        "path",
        "runtime_case",
        "size_bytes",
        "source_value",
    )
    if not isinstance(value, Mapping):
        raise ValueError("Candidate11 public aggregate must be one flat mapping")
    for key, child in value.items():
        if not isinstance(key, str) or any(
            fragment in key for fragment in forbidden_key_fragments
        ):
            raise ValueError("Candidate11 aggregate exposes a forbidden field")
        if type(child) not in (bool, str):
            raise ValueError("Candidate11 aggregate exposes a non-scalar value")


# Candidate11 publication contract.  The Candidate10-compatible constructors
# above remain private migration scaffolding; this redefinition is the sole
# constructor used by the Candidate11 one-shot workflow.  It intentionally
# removes counts, paths, ordinals, identifiers, values, projection artifacts,
# and exception details from every published object.
C11_LEDGER_KEYS = frozenset(
    {
        "adapted_cache_published",
        "all_artifact_reads_completed",
        "all_reads_exactly_authorized",
        "anchor_consumed_before_manifest",
        "anchor_durable_entry_created_or_partial",
        "anchor_receipt_available",
        "anchor_receipt_commitment_sha256",
        "artifact_inventory_commitment_sha256",
        "artifact_inventory_installed",
        "authorization_commitment_sha256",
        "candidate_metric_computed",
        "check_code",
        "control_binding_commitment_sha256",
        "core_bitset_commitment_sha256",
        "core_result_available",
        "core_result_commitment_sha256",
        "denied_read_attempt_observed",
        "diagnostic_complete",
        "diagnostic_incompatibility_observed",
        "engineering_attempt_id",
        "evaluator_opened",
        "failure_detail_exposed",
        "failure_stage",
        "fit_or_model_performed",
        "holdout_opened",
        "kind",
        "labels_opened",
        "manifest_binding_commitment_sha256",
        "manifest_inode_verified",
        "manifest_raw_fd_opened",
        "manifest_read_started",
        "promotion_authorized",
        "qualification_commitment_sha256",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
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

C11_RESULT_KEYS = frozenset(
    {
        "adapted_cache_published",
        "all_records_compatible",
        "candidate_metric_computed",
        "check_code",
        "consumption_anchor_commitment_sha256",
        "core_bitset_commitment_sha256",
        "core_input_receipt_commitment_sha256",
        "core_result_commitment_sha256",
        "core_source_lineage_commitment_sha256",
        "core_stage_contract_commitment_sha256",
        "diagnostic_complete",
        "engineering_attempt_id",
        "evaluator_opened",
        "failure_detail_exposed",
        "fit_or_model_performed",
        "holdout_opened",
        "kind",
        "labels_opened",
        "manifest_binding_commitment_sha256",
        "primary_replay_identity",
        "promotion_authorized",
        "read_ledger_commitment_sha256",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_inventory_commitment_sha256",
        "source_only",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)

C11_TERMINAL_KEYS = frozenset(
    {
        "adapted_cache_published",
        "candidate_metric_computed",
        "check_code",
        "diagnostic_complete",
        "engineering_attempt_id",
        "evaluator_opened",
        "failure_detail_exposed",
        "failure_stage",
        "fit_or_model_performed",
        "holdout_opened",
        "kind",
        "labels_opened",
        "promotion_authorized",
        "read_ledger_commitment_sha256",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "source_only",
        "status",
    }
)

C11_LOCK_KEYS = frozenset(
    {
        "adapted_cache_published",
        "all_records_compatible",
        "authorization_commitment_sha256",
        "check_code",
        "consumption_anchor_commitment_sha256",
        "control_binding_commitment_sha256",
        "core_bitset_commitment_sha256",
        "engineering_attempt_id",
        "kind",
        "manifest_binding_commitment_sha256",
        "post_source_or_policy_mutation_invalidates_lock",
        "promotion_authorized",
        "qualification_commitment_sha256",
        "read_ledger_commitment_sha256",
        "result_commitment_sha256",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
)


def _unavailable_commitment(name: str, *, failure_stage: str) -> str:
    if failure_stage not in ENGINEERING_FAILURE_CODES:
        raise ValueError("Candidate11 unavailable commitment stage is invalid")
    return sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE11_UNAVAILABLE_COMMITMENT_V1",
                "failure_stage": failure_stage,
                "name": name,
            }
        )
    )


def _fixed_bitset(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 12 or set(value) - {"0", "1"}:
        raise ValueError(f"Candidate11 {name} must be one exact 12-bit string")
    return value


def _validate_c11_ledger(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != C11_LEDGER_KEYS:
        raise ValueError("Candidate11 ledger schema mismatch")
    boolean_keys = C11_LEDGER_KEYS - {
        "anchor_receipt_commitment_sha256",
        "artifact_inventory_commitment_sha256",
        "authorization_commitment_sha256",
        "check_code",
        "control_binding_commitment_sha256",
        "core_bitset_commitment_sha256",
        "core_result_commitment_sha256",
        "engineering_attempt_id",
        "failure_stage",
        "kind",
        "manifest_binding_commitment_sha256",
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
    failure_stage = value["failure_stage"]
    core_available = value["core_result_available"]
    if (
        value["kind"] != LEDGER_KIND
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or any(type(value[key]) is not bool for key in boolean_keys)
        or value["adapted_cache_published"] is not False
        or value["candidate_metric_computed"] is not False
        or value["evaluator_opened"] is not False
        or value["failure_detail_exposed"] is not False
        or value["fit_or_model_performed"] is not False
        or value["holdout_opened"] is not False
        or value["labels_opened"] is not False
        or value["promotion_authorized"] is not False
        or value["scientific_hypothesis_performance_tested"] is not False
        or value["source_only"] is not True
        or type(failure_stage) is not str
        or type(core_available) is not bool
    ):
        raise ValueError("Candidate11 ledger invariant mismatch")
    for key in C11_LEDGER_KEYS:
        if key.endswith("_commitment_sha256"):
            _require_sha(value[key], name=f"Candidate11 ledger {key}")
    reached = _fixed_bitset(value["stage_reached_all_bitset"], name="ledger reached")
    passed = _fixed_bitset(value["stage_passed_all_bitset"], name="ledger passed")
    if core_available:
        if (
            failure_stage != "NONE"
            or value["diagnostic_complete"] is not True
            or value["status"] != "DIAGNOSTIC_COMPLETED_NONAUTHORIZING"
            or value["check_code"] not in {
                "C11_ALL_RECORD_DIAGNOSTICS_REPLAY_COMPATIBLE",
                "C11_AGGREGATE_DIAGNOSTIC_INCOMPATIBLE",
            }
            or reached[-1] != "1"
            or passed[-1] != "1"
        ):
            raise ValueError("Candidate11 completed ledger mismatch")
    elif (
        failure_stage not in ENGINEERING_FAILURE_CODES
        or value["diagnostic_complete"] is not False
        or value["diagnostic_incompatibility_observed"] is not False
        or value["status"] != "TERMINAL_ENGINEERING_FAILURE"
        or value["check_code"] != ENGINEERING_FAILURE_CODES[failure_stage]
        or reached != "000000000000"
        or passed != "000000000000"
    ):
        raise ValueError("Candidate11 terminal ledger mismatch")
    _assert_aggregate_safe(dict(value))


def _c11_ledger_value(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    consumption: ConsumptionSeal | None,
    outcome: ExecutionOutcome | None,
    failure_stage: str,
    anchor_durable_entry_created_or_partial: bool,
) -> dict[str, Any]:
    reader = qualification.reader
    core_available = outcome is not None
    artifact_purposes = [
        purpose
        for purpose in reader.read_started_purposes
        if purpose.startswith("TRAIN_RECORD_")
    ]
    all_artifacts = (
        reader.artifact_inventory is not None
        and len(reader.artifact_inventory.bindings) == SOURCE_FILE_COUNT
        and len(artifact_purposes) == SOURCE_FILE_COUNT
        and artifact_purposes
        == [
            purpose
            for purpose in reader.raw_fd_opened_purposes
            if purpose.startswith("TRAIN_RECORD_")
        ]
        == [
            purpose
            for purpose in reader.verified_inode_purposes
            if purpose.startswith("TRAIN_RECORD_")
        ]
    )
    if core_available:
        core = outcome.core_result
        reached = _fixed_bitset(core["stage_reached_all_bitset"], name="core reached")
        passed = _fixed_bitset(core["stage_passed_all_bitset"], name="core passed")
        core_commitment = sha256_bytes(outcome.core_result_bytes)
        bitset_commitment = _require_sha(
            core["aggregate_reached_passed_bitsets_commitment_sha256"],
            name="Candidate11 core bitset commitment",
        )
        artifact_commitment = outcome.artifact_inventory_commitment_sha256
        manifest_commitment = outcome.manifest_binding_sha256
        stage = "NONE"
        check_code = str(core["check_code"])
    else:
        reached = passed = "000000000000"
        core_commitment = _unavailable_commitment(
            "core_result", failure_stage=failure_stage
        )
        bitset_commitment = _unavailable_commitment(
            "core_bitsets", failure_stage=failure_stage
        )
        artifact_commitment = _unavailable_commitment(
            "artifact_inventory", failure_stage=failure_stage
        )
        manifest_commitment = _unavailable_commitment(
            "manifest_binding", failure_stage=failure_stage
        )
        stage = failure_stage
        check_code = ENGINEERING_FAILURE_CODES[failure_stage]
    value = {
        "adapted_cache_published": False,
        "all_artifact_reads_completed": all_artifacts,
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
            else _unavailable_commitment("anchor_receipt", failure_stage=failure_stage)
        ),
        "artifact_inventory_commitment_sha256": artifact_commitment,
        "artifact_inventory_installed": reader.artifact_inventory is not None,
        "authorization_commitment_sha256": authorization.authorization_sha256,
        "candidate_metric_computed": False,
        "check_code": check_code,
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "core_bitset_commitment_sha256": bitset_commitment,
        "core_result_available": core_available,
        "core_result_commitment_sha256": core_commitment,
        "denied_read_attempt_observed": bool(reader.denied_attempts),
        "diagnostic_complete": core_available,
        "diagnostic_incompatibility_observed": (
            core_available and outcome.core_result["all_records_compatible"] is False
        ),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "failure_detail_exposed": False,
        "failure_stage": stage,
        "fit_or_model_performed": False,
        "holdout_opened": False,
        "kind": LEDGER_KIND,
        "labels_opened": False,
        "manifest_binding_commitment_sha256": manifest_commitment,
        "manifest_inode_verified": "TRAIN_CACHE_MANIFEST"
        in reader.verified_inode_purposes,
        "manifest_raw_fd_opened": "TRAIN_CACHE_MANIFEST"
        in reader.raw_fd_opened_purposes,
        "manifest_read_started": "TRAIN_CACHE_MANIFEST"
        in reader.read_started_purposes,
        "promotion_authorized": False,
        "qualification_commitment_sha256": authorization.qualification_sha256,
        "raw_fd_opened_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE11_ALL_RAW_FD_OPENED_V1", reader.raw_fd_opened_purposes
        ),
        "read_inventory_commitment_sha256": _entry_commitment(reader.entries),
        "read_started_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE11_ALL_READ_STARTED_V1", reader.read_started_purposes
        ),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "source_only": True,
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": reached,
        "status": (
            "DIAGNOSTIC_COMPLETED_NONAUTHORIZING"
            if core_available
            else "TERMINAL_ENGINEERING_FAILURE"
        ),
        "verified_inode_inventory_commitment_sha256": _purpose_commitment(
            "CANDIDATE11_ALL_VERIFIED_INODES_V1", reader.verified_inode_purposes
        ),
    }
    _validate_c11_ledger(value)
    return value


def _validate_c11_result(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != C11_RESULT_KEYS:
        raise ValueError("Candidate11 result schema mismatch")
    boolean_keys = C11_RESULT_KEYS - {
        "check_code",
        "consumption_anchor_commitment_sha256",
        "core_bitset_commitment_sha256",
        "core_input_receipt_commitment_sha256",
        "core_result_commitment_sha256",
        "core_source_lineage_commitment_sha256",
        "core_stage_contract_commitment_sha256",
        "engineering_attempt_id",
        "kind",
        "manifest_binding_commitment_sha256",
        "read_ledger_commitment_sha256",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
    }
    compatible = value["all_records_compatible"]
    reached = _fixed_bitset(value["stage_reached_all_bitset"], name="result reached")
    passed = _fixed_bitset(value["stage_passed_all_bitset"], name="result passed")
    if (
        value["kind"] != RESULT_KIND
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or any(type(value[key]) is not bool for key in boolean_keys)
        or value["diagnostic_complete"] is not True
        or value["adapted_cache_published"] is not False
        or value["candidate_metric_computed"] is not False
        or value["evaluator_opened"] is not False
        or value["failure_detail_exposed"] is not False
        or value["fit_or_model_performed"] is not False
        or value["holdout_opened"] is not False
        or value["labels_opened"] is not False
        or value["promotion_authorized"] is not False
        or value["scientific_hypothesis_performance_tested"] is not False
        or value["source_only"] is not True
        or value["check_code"]
        != (
            "C11_ALL_RECORD_DIAGNOSTICS_REPLAY_COMPATIBLE"
            if compatible
            else "C11_AGGREGATE_DIAGNOSTIC_INCOMPATIBLE"
        )
        or value["status"]
        != (
            "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
            if compatible
            else "SOURCE_REPLAY_INCOMPATIBLE_NONAUTHORIZING"
        )
        or compatible is (passed != "111111111111")
        or reached[-1] != "1"
        or passed[-1] != "1"
    ):
        raise ValueError("Candidate11 result invariant mismatch")
    for key in C11_RESULT_KEYS:
        if key.endswith("_commitment_sha256"):
            _require_sha(value[key], name=f"Candidate11 result {key}")
    _assert_aggregate_safe(dict(value))


def _c11_result_value(
    context: BootstrapContext,
    consumption: ConsumptionSeal,
    outcome: ExecutionOutcome,
    *,
    ledger_bytes: bytes,
) -> dict[str, Any]:
    core = outcome.core_result
    value = {
        "adapted_cache_published": False,
        "all_records_compatible": core["all_records_compatible"],
        "candidate_metric_computed": False,
        "check_code": core["check_code"],
        "consumption_anchor_commitment_sha256": consumption.sha256,
        "core_bitset_commitment_sha256": core[
            "aggregate_reached_passed_bitsets_commitment_sha256"
        ],
        "core_input_receipt_commitment_sha256": core[
            "input_receipt_vector_commitment_sha256"
        ],
        "core_result_commitment_sha256": sha256_bytes(outcome.core_result_bytes),
        "core_source_lineage_commitment_sha256": core[
            "source_lineage_commitment_sha256"
        ],
        "core_stage_contract_commitment_sha256": core["stage_contract_sha256"],
        "diagnostic_complete": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "failure_detail_exposed": False,
        "fit_or_model_performed": False,
        "holdout_opened": False,
        "kind": RESULT_KIND,
        "labels_opened": False,
        "manifest_binding_commitment_sha256": outcome.manifest_binding_sha256,
        "primary_replay_identity": core["primary_replay_identity"],
        "promotion_authorized": False,
        "read_ledger_commitment_sha256": sha256_bytes(ledger_bytes),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "source_only": True,
        "stage_passed_all_bitset": core["stage_passed_all_bitset"],
        "stage_reached_all_bitset": core["stage_reached_all_bitset"],
        "status": core["status"],
    }
    _validate_c11_result(value)
    return value


def _validate_c11_terminal(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != C11_TERMINAL_KEYS:
        raise ValueError("Candidate11 terminal schema mismatch")
    boolean_keys = C11_TERMINAL_KEYS - {
        "check_code",
        "engineering_attempt_id",
        "failure_stage",
        "kind",
        "read_ledger_commitment_sha256",
        "scientific_hypothesis_id",
        "status",
    }
    stage = value["failure_stage"]
    if (
        value["kind"] != TERMINAL_KIND
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or stage not in ENGINEERING_FAILURE_CODES
        or value["check_code"] != ENGINEERING_FAILURE_CODES[stage]
        or value["status"] != "TERMINAL_ENGINEERING_FAILURE"
        or any(type(value[key]) is not bool for key in boolean_keys)
        or value["diagnostic_complete"] is not False
        or value["adapted_cache_published"] is not False
        or value["candidate_metric_computed"] is not False
        or value["evaluator_opened"] is not False
        or value["failure_detail_exposed"] is not False
        or value["fit_or_model_performed"] is not False
        or value["holdout_opened"] is not False
        or value["labels_opened"] is not False
        or value["promotion_authorized"] is not False
        or value["scientific_hypothesis_performance_tested"] is not False
        or value["source_only"] is not True
    ):
        raise ValueError("Candidate11 terminal invariant mismatch")
    _require_sha(
        value["read_ledger_commitment_sha256"], name="Candidate11 terminal ledger"
    )
    _assert_aggregate_safe(dict(value))


def _c11_terminal_value(*, failure_stage: str, ledger_bytes: bytes) -> dict[str, Any]:
    value = {
        "adapted_cache_published": False,
        "candidate_metric_computed": False,
        "check_code": ENGINEERING_FAILURE_CODES[failure_stage],
        "diagnostic_complete": False,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "evaluator_opened": False,
        "failure_detail_exposed": False,
        "failure_stage": failure_stage,
        "fit_or_model_performed": False,
        "holdout_opened": False,
        "kind": TERMINAL_KIND,
        "labels_opened": False,
        "promotion_authorized": False,
        "read_ledger_commitment_sha256": sha256_bytes(ledger_bytes),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "source_only": True,
        "status": "TERMINAL_ENGINEERING_FAILURE",
    }
    _validate_c11_terminal(value)
    return value


def _validate_c11_lock(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != C11_LOCK_KEYS:
        raise ValueError("Candidate11 lock schema mismatch")
    compatible = value["all_records_compatible"]
    reached = _fixed_bitset(value["stage_reached_all_bitset"], name="lock reached")
    passed = _fixed_bitset(value["stage_passed_all_bitset"], name="lock passed")
    if (
        value["kind"] != LOCK_KIND
        or value["engineering_attempt_id"] != ENGINEERING_ATTEMPT_ID
        or value["scientific_hypothesis_id"] != SCIENTIFIC_HYPOTHESIS_ID
        or type(compatible) is not bool
        or value["adapted_cache_published"] is not False
        or value["promotion_authorized"] is not False
        or value["post_source_or_policy_mutation_invalidates_lock"] is not True
        or value["check_code"]
        != (
            "C11_ALL_RECORD_DIAGNOSTICS_REPLAY_COMPATIBLE"
            if compatible
            else "C11_AGGREGATE_DIAGNOSTIC_INCOMPATIBLE"
        )
        or value["status"] != "DIAGNOSTIC_COMPLETED_NONAUTHORIZING"
        or compatible is (passed != "111111111111")
        or reached[-1] != "1"
        or passed[-1] != "1"
    ):
        raise ValueError("Candidate11 lock invariant mismatch")
    for key in C11_LOCK_KEYS:
        if key.endswith("_commitment_sha256"):
            _require_sha(value[key], name=f"Candidate11 lock {key}")
    _assert_aggregate_safe(dict(value))


def _c11_lock_value(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    consumption: ConsumptionSeal,
    outcome: ExecutionOutcome,
    *,
    ledger_bytes: bytes,
    result_bytes: bytes,
) -> dict[str, Any]:
    core = outcome.core_result
    value = {
        "adapted_cache_published": False,
        "all_records_compatible": core["all_records_compatible"],
        "authorization_commitment_sha256": authorization.authorization_sha256,
        "check_code": core["check_code"],
        "consumption_anchor_commitment_sha256": consumption.sha256,
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "core_bitset_commitment_sha256": core[
            "aggregate_reached_passed_bitsets_commitment_sha256"
        ],
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "kind": LOCK_KIND,
        "manifest_binding_commitment_sha256": outcome.manifest_binding_sha256,
        "post_source_or_policy_mutation_invalidates_lock": True,
        "promotion_authorized": False,
        "qualification_commitment_sha256": authorization.qualification_sha256,
        "read_ledger_commitment_sha256": sha256_bytes(ledger_bytes),
        "result_commitment_sha256": sha256_bytes(result_bytes),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "stage_passed_all_bitset": core["stage_passed_all_bitset"],
        "stage_reached_all_bitset": core["stage_reached_all_bitset"],
        "status": "DIAGNOSTIC_COMPLETED_NONAUTHORIZING",
    }
    _validate_c11_lock(value)
    return value


def _c11_terminal_payloads(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    *,
    consumption: ConsumptionSeal | None,
    failure_stage: str,
    anchor_durable_entry_created_or_partial: bool = False,
) -> dict[str, bytes]:
    ledger = _c11_ledger_value(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=None,
        failure_stage=failure_stage,
        anchor_durable_entry_created_or_partial=anchor_durable_entry_created_or_partial,
    )
    ledger_bytes = canonical_bytes(ledger)
    terminal = _c11_terminal_value(
        failure_stage=failure_stage, ledger_bytes=ledger_bytes
    )
    return {
        "full_replay_diagnostic_read_ledger.json": ledger_bytes,
        "full_replay_diagnostic_terminal.json": canonical_bytes(terminal),
    }


def build_payloads(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
) -> dict[str, bytes]:
    _validate_context(context)
    _validate_authorization(context, authorization)
    _validate_qualification(context, qualification)
    consumption: ConsumptionSeal | None = None
    try:
        consumption = consume_authorization(context, authorization, qualification)
    except AnchorCreationFailure:
        return _c11_terminal_payloads(
            context,
            authorization,
            qualification,
            consumption=None,
            failure_stage="ANCHOR_CONSUMPTION",
            anchor_durable_entry_created_or_partial=True,
        )
    try:
        outcome = read_adapt_diagnose_full_inventory(
            context, qualification, consumption
        )
    except BaseException as error:
        stage = (
            error.stage
            if isinstance(error, RunnerFailure)
            and error.stage in ENGINEERING_FAILURE_CODES
            else "CORE_DIAGNOSTIC_INVOCATION"
        )
        return _c11_terminal_payloads(
            context,
            authorization,
            qualification,
            consumption=consumption,
            failure_stage=stage,
        )
    ledger = _c11_ledger_value(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=outcome,
        failure_stage="CORE_DIAGNOSTIC_INVOCATION",
        anchor_durable_entry_created_or_partial=False,
    )
    ledger_bytes = canonical_bytes(ledger)
    result = _c11_result_value(
        context, consumption, outcome, ledger_bytes=ledger_bytes
    )
    result_bytes = canonical_bytes(result)
    lock = _c11_lock_value(
        context,
        authorization,
        qualification,
        consumption,
        outcome,
        ledger_bytes=ledger_bytes,
        result_bytes=result_bytes,
    )
    return {
        "full_replay_diagnostic_lock.json": canonical_bytes(lock),
        "full_replay_diagnostic_read_ledger.json": ledger_bytes,
        "full_replay_diagnostic_result.json": result_bytes,
    }


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
        raise RuntimeError("Candidate11 requires a no-replace directory rename")
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
            raise FileExistsError(number, "Candidate11 publication destination exists")
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
        raise RuntimeError("Candidate11 requires O_NOFOLLOW/O_DIRECTORY")
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
            raise RuntimeError("Candidate11 owned publication inventory changed")
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
                    raise RuntimeError("Candidate11 owned leaf metadata changed")
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
                    raise RuntimeError("Candidate11 owned leaf bytes changed")
            finally:
                os.close(leaf_fd)
    finally:
        os.close(descriptor)


def publish_atomically(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    freshness: FreshOutputSeal,
) -> None:
    expected = (
        frozenset(
            {
                "full_replay_diagnostic_lock.json",
                "full_replay_diagnostic_read_ledger.json",
                "full_replay_diagnostic_result.json",
            }
        ),
        frozenset(
            {
                "full_replay_diagnostic_read_ledger.json",
                "full_replay_diagnostic_terminal.json",
            }
        ),
    )
    if (
        frozenset(payloads) not in expected
        or any(not isinstance(data, bytes) or not data for data in payloads.values())
    ):
        raise ValueError("Candidate11 publication inventory is invalid")
    revalidate_fresh_output(freshness)
    candidate = _validate_output_namespace(output)
    if candidate != freshness.path:
        raise ValueError("Candidate11 publication differs from freshness seal")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate11 publication parent"
    )
    if parent_identity != freshness.parent_identity:
        os.close(parent_fd)
        raise RuntimeError("Candidate11 publication parent identity changed")
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
            raise RuntimeError("Candidate11 staging leaf is not a directory")
        staging_identity = _identity(staging_metadata)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        if _identity(os.fstat(staging_fd)) != staging_identity:
            raise RuntimeError("Candidate11 staging directory identity changed")
        for filename, data in sorted(payloads.items()):
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
                    raise RuntimeError("Candidate11 staged leaf identity mismatch")
                leaf_identities[filename] = identity
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Candidate11 publication write made no progress")
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
            raise FileExistsError("Candidate11 publication destination raced into existence")
        _renameat_noreplace(parent_fd, staging_name, candidate.name)
        os.fsync(parent_fd)
        visible_output = os.stat(
            candidate.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            _identity(visible_output) != staging_identity
            or _identity(os.lstat(candidate.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate11 published directory identity changed")
        _verify_owned_directory(
            parent_fd,
            name=candidate.name,
            directory_identity=staging_identity,
            leaf_identities=leaf_identities,
            payloads=payloads,
        )
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        # Never reselect a staging or final pathname for cleanup.  After any
        # publication failure, the residue is deliberately left in place and
        # blocks all retry.  This avoids unlinking or moving a same-UID foreign
        # replacement introduced after an inode check.
        os.close(parent_fd)


def run_one_shot(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
) -> dict[str, Any]:
    freshness = prevalidate_fresh_output()
    authorization = authorize_diagnostic_once(
        context,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        output_freshness=freshness,
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
        raise PermissionError("Candidate11 external authority does not bind controls")
    payloads = build_payloads(context, authorization, qualification)
    publish_atomically(OUTPUT_ROOT, payloads, freshness=freshness)
    public_name = (
        "full_replay_diagnostic_result.json"
        if "full_replay_diagnostic_result.json" in payloads
        else "full_replay_diagnostic_terminal.json"
    )
    public = _decode(payloads[public_name], name="Candidate11 public result")
    return {
        "check_code": public["check_code"],
        "diagnostic_complete": public["diagnostic_complete"],
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "output_inventory_commitment_sha256": sha256_bytes(
            canonical_bytes(
                {name: sha256_bytes(data) for name, data in sorted(payloads.items())}
            )
        ),
        "stage_passed_all_bitset": public.get(
            "stage_passed_all_bitset", "000000000000"
        ),
        "stage_reached_all_bitset": public.get(
            "stage_reached_all_bitset", "000000000000"
        ),
        "status": public["status"],
    }


def _bootstrap_cli_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"Candidate11 bootstrap requires exactly one {flag}")
    position = positions[0]
    if position + 1 >= len(arguments):
        raise ValueError(f"Candidate11 bootstrap requires a value for {flag}")
    value = arguments[position + 1]
    if not isinstance(value, str) or value.startswith("--"):
        raise ValueError(f"Candidate11 bootstrap has an invalid value for {flag}")
    return value


def _dispatch_through_verified_runner(
    argv: Sequence[str] | None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not all(type(value) is str for value in arguments):
        raise TypeError("Candidate11 bootstrap arguments must be exact strings")
    expected_sha = _bootstrap_cli_value(arguments, "--runner-sha256")
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("Candidate11 bootstrap runner SHA is invalid")
    size_text = _bootstrap_cli_value(arguments, "--runner-size-bytes")
    if not size_text.isdecimal() or int(size_text) <= 0:
        raise ValueError("Candidate11 bootstrap runner size is invalid")
    expected_size = int(size_text)

    runner_path = Path(__file__)
    data, _identity_value = _read_regular_file_once(
        runner_path,
        role="Candidate11 executing runner bootstrap",
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise PermissionError("Candidate11 executing runner byte binding mismatch")

    private_name = f"_candidate11_verified_runner_{expected_sha}"
    if private_name in sys.modules:
        raise PermissionError("Candidate11 verified runner is already loaded")
    outer_module = sys.modules.get(__name__)
    if type(outer_module) is not types.ModuleType:
        raise RuntimeError("Candidate11 outer runner module is unavailable")
    outer_namespace = object.__getattribute__(outer_module, "__dict__")
    if outer_namespace.get("_C11_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate11 runner bootstrap is not reentrant")

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
    module.__dict__["_C11_VERIFIED_RUNNER_BINDING"] = binding
    module.__dict__["_C11_VERIFIED_RUNNER_LAUNCH_TOKEN"] = launch_token
    module.__dict__["_C11_VERIFIED_RUNNER_OUTER_MODULE"] = outer_module
    outer_namespace["_C11_OUTER_LAUNCH_TOKEN"] = launch_token
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
            raise PermissionError("Candidate11 verified runner entry point mismatch")
        return int(private_main(arguments))
    finally:
        sys.modules.pop(private_name, None)
        if outer_namespace.get("_C11_OUTER_LAUNCH_TOKEN") is launch_token:
            del outer_namespace["_C11_OUTER_LAUNCH_TOKEN"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--runner-size-bytes", required=True, type=int)
    parser.add_argument("--runner-tests-sha256", required=True)
    parser.add_argument("--runner-tests-size-bytes", required=True, type=int)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--source-control-check-only", action="store_true")
    modes.add_argument("--diagnostic-once", action="store_true")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--authorization-size-bytes", type=int)
    args = parser.parse_args(argv)
    if args.diagnostic_once and (
        args.authorization_sha256 is None or args.authorization_size_bytes is None
    ):
        parser.error("--diagnostic-once requires exact authorization SHA and size")
    if not args.diagnostic_once and (
        args.authorization_sha256 is not None or args.authorization_size_bytes is not None
    ):
        parser.error("authorization binding is valid only with --diagnostic-once")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    if _C11_VERIFIED_RUNNER_BINDING is None:
        try:
            return _dispatch_through_verified_runner(argv)
        except BaseException:
            print(
                json.dumps(
                    {
                        "check_code": "C11_RUNNER_REJECTED_OR_FAILED",
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
            "check_code": "C11_RUNNER_REJECTED_OR_FAILED",
            "diagnostic_complete": False,
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "status": "FAIL_CLOSED",
        }
        exit_code = 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
