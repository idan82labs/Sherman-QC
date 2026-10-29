#!/usr/bin/env python3
"""Guarded one-shot runner for Candidate8's value-free first-pair diagnostic.

The repeatable modes verify only frozen source bytes or value-free controls.
Only an exact external authorization may create the durable consumption anchor,
open the attested TRAIN cache manifest, and then read the one primary semantic /
pool pair already consumed by Candidate7.  Replay and every later pair remain
unopened.  Published artifacts contain only bounded diagnostic status plus a
value-free custody ledger; labels, evaluator outputs, metrics, fitting, the
full-28 manifest, and holdout artifacts are outside this program's capability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import types
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = Path(__file__).resolve()
RUNNER_TESTS_SOURCE = (
    ROOT / "tests/test_run_f1_train18_candidate8_first_primary_diagnostic_once_v1.py"
)
C8_RUNTIME_SOURCE = (
    ROOT / "backend/f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
)
C8_RUNTIME_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
)
C8_ADAPTER_SOURCE = (
    ROOT / "scripts/preflight_f1_train18_candidate8_diagnostic_adapter_v1.py"
)
C8_ADAPTER_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate8_diagnostic_adapter_v1.py"
)
C6_HYPOTHESIS_SOURCE = (
    ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
)
C7_RUNTIME_SOURCE = (
    ROOT / "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py"
)

C8_RUNTIME_SHA256 = "d878eb9141f287435890ed3290a45f40dd0730130585a6846c38431e3342f9b8"
C8_RUNTIME_SIZE_BYTES = 21360
C8_ADAPTER_SHA256 = "b91a112356dba3ed4aa9e3fe51fe36ea12bf93d2dd9ab8900f2b3461509211be"
C8_ADAPTER_SIZE_BYTES = 14807
C8_RUNTIME_TESTS_SHA256 = "85c8b684dcaf4acaccb9800ba7bb7747c3546b24baae7d3c0685169431f52cfb"
C8_RUNTIME_TESTS_SIZE_BYTES = 14652
C8_ADAPTER_TESTS_SHA256 = "20bb70082daf1e681dd2ce3a1367da59befce4eebabcfd7a75944c5246091b8b"
C8_ADAPTER_TESTS_SIZE_BYTES = 14197
C6_HYPOTHESIS_SHA256 = "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252"
C6_HYPOTHESIS_SIZE_BYTES = 53837
C7_RUNTIME_SHA256 = "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d"
C7_RUNTIME_SIZE_BYTES = 28098

C8_RUNTIME_MODULE = "backend.f1_train18_candidate8_diagnostic_recall_wrapper_v1"
C8_ADAPTER_MODULE = "scripts.preflight_f1_train18_candidate8_diagnostic_adapter_v1"
C6_HYPOTHESIS_MODULE = "backend.f1_train18_candidate6_conservative_recall_recovery_v1"
C7_RUNTIME_MODULE = "backend.f1_train18_candidate7_conservative_recall_recovery_v1"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
C7_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate7_terminal_disposition_v1.json"
C7_QUALIFICATION = ROOT / "research/f1_aug30_train18_candidate7_source_control_qualification_v1.json"
C7_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate7_source_inventory_authorization_v1.json"
C7_ANCHOR = ROOT / "research/f1_aug30_train18_candidate7_source_inventory_consumption_v1.json"
C7_OUTPUT_ROOT = (
    ROOT / "output/research/f1-aug30-train18-candidate7-semantic-pool-preflight-v1"
)
C7_TERMINAL = C7_OUTPUT_ROOT / "preflight_terminal.json"
C7_LEDGER = C7_OUTPUT_ROOT / "preflight_read_ledger.json"

CUSTODY_ROOT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
CACHE_ATTESTATION = CUSTODY_ROOT / "cache/train_v51_cache_manifest_attestation.json"
CACHE_MANIFEST = CUSTODY_ROOT / "cache/train_v51_cache_manifest.json"
CUSTODY_EVIDENCE_PATHS = {
    "access_evidence": CUSTODY_ROOT / "cache/access_evidence.json",
    "builder": ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py",
    "staged_validator_pass": CUSTODY_ROOT / "independent_validation_report.json",
    "validator": ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py",
}
ALLOWED_CACHE_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
)

C8_QUALIFICATION = (
    ROOT / "research/f1_aug30_train18_candidate8_first_primary_diagnostic_qualification_v1.json"
)
C8_AUTHORIZATION = (
    ROOT / "research/f1_aug30_train18_candidate8_first_primary_diagnostic_authorization_v1.json"
)
C8_CONSUMPTION_ANCHOR = (
    ROOT / "research/f1_aug30_train18_candidate8_first_primary_diagnostic_consumption_v1.json"
)
OUTPUT_ROOT = (
    ROOT / "output/research/f1-aug30-train18-candidate8-first-primary-diagnostic-v1"
)

FIXED_CONTROLS = {
    "SEALED_SPLIT": (SPLIT, "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185", 10706),
    "ORIGINAL_PREREGISTRATION": (ORIGINAL_PREREG, "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a", 2651),
    "TRAIN_ALLOWLIST": (ALLOWLIST, "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0", 2428),
    "CANDIDATE7_TERMINAL_DISPOSITION": (C7_DISPOSITION, "1a64d3148d018b224a3a44d9d36d2620f1d87f18063e32df14050210aa03f56b", 3666),
    "CANDIDATE7_PREFLIGHT_TERMINAL": (C7_TERMINAL, "9a9284f5f4d88854ca4162ecfc07f5ff022074fe4f7319b8fd7f0fa15be65295", 1032),
    "CANDIDATE7_PREFLIGHT_LEDGER": (C7_LEDGER, "50aca2acdf2f630ec492d61e4733b22e3d2cf1dd2ad33252a198bcff852e16d4", 6897),
    "CANDIDATE7_CONSUMPTION_ANCHOR": (C7_ANCHOR, "9a3d61eb80128ececcbfd2477bfd53d7a76519137f083eeae09d95ed9adaa793", 2076),
    "CANDIDATE7_SOURCE_AUTHORIZATION": (C7_AUTHORIZATION, "c4bf8b092f66025a786af28909cd98f8cf497c754a4b7b5e1fd49068123da210", 1975),
    "CANDIDATE7_CONTROL_QUALIFICATION": (C7_QUALIFICATION, "5aa45de26115812bbdbdfb6352f126e948f5c4e5829705f887d9563b743954e3", 1703),
    "TRAIN_CACHE_ATTESTATION": (CACHE_ATTESTATION, "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71", 1834),
}
CACHE_MANIFEST_SHA256 = "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022"
CACHE_MANIFEST_SIZE_BYTES = 19601
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
V51_SCAN_ONLY_MANIFEST_SHA256 = "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"

SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 8
ENGINEERING_ATTEMPT_ID = "CANDIDATE8_FIRST_PRIMARY_DIAGNOSTIC_RUNNER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
QUALIFICATION_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_qualification_v1"
AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_authorization_v1"
CONSUMPTION_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_consumption_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_terminal_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate8_first_primary_diagnostic_read_ledger_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CUSTODY_VALIDATION_KIND = "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"
RECORD_SELECTOR = "EXACT_C7_CONSUMED_FIRST_PRIMARY_BY_FROZEN_MANIFEST_ORDER"
ENGINEERING_FAILURE_CODES = {
    "ANCHOR_CONSUMPTION": frozenset(
        {"ANCHOR_CREATION_FAILED_AFTER_CONSUMPTION"}
    ),
    "CACHE_MANIFEST_VALIDATION": frozenset(
        {
            "C7_CONSUMED_FIRST_PAIR_BINDING_MISMATCH",
            "CACHE_MANIFEST_HEADER_MISMATCH",
            "CACHE_MANIFEST_RECORD_VECTOR_MISMATCH",
            "CACHE_MEMBERSHIP_ORDER_MISMATCH",
            "CACHE_RECORD_IDENTITY_MISMATCH",
            "CACHE_RECORD_PATH_MISMATCH",
            "CACHE_RECORD_SCHEMA_MISMATCH",
            "PRIMARY_REPLAY_HASH_MISMATCH",
        }
    ),
    "DIAGNOSTIC_EXECUTION": frozenset(
        {"UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE"}
    ),
    "DIAGNOSTIC_READ_INVENTORY": frozenset(
        {"EXACT_READ_INVENTORY_MISMATCH"}
    ),
    "DIAGNOSTIC_RESULT_VALIDATION": frozenset(
        {
            "DIAGNOSTIC_CODE_OUTSIDE_FROZEN_SET",
            "DIAGNOSTIC_RESULT_OUTSIDE_D01_D05_PASS",
            "DIAGNOSTIC_RESULT_SCHEMA_MISMATCH",
        }
    ),
    "FIRST_PRIMARY_BINDING": frozenset(
        {"C7_CONSUMED_FIRST_PAIR_INODE_MISMATCH"}
    ),
}

SOURCE_FIXED_BINDINGS = {
    "adapter": (C8_ADAPTER_SOURCE, C8_ADAPTER_SHA256, C8_ADAPTER_SIZE_BYTES),
    "adapter_tests": (C8_ADAPTER_TESTS_SOURCE, C8_ADAPTER_TESTS_SHA256, C8_ADAPTER_TESTS_SIZE_BYTES),
    "candidate7_runtime": (C7_RUNTIME_SOURCE, C7_RUNTIME_SHA256, C7_RUNTIME_SIZE_BYTES),
    "hypothesis": (C6_HYPOTHESIS_SOURCE, C6_HYPOTHESIS_SHA256, C6_HYPOTHESIS_SIZE_BYTES),
    "runtime": (C8_RUNTIME_SOURCE, C8_RUNTIME_SHA256, C8_RUNTIME_SIZE_BYTES),
    "runtime_tests": (C8_RUNTIME_TESTS_SOURCE, C8_RUNTIME_TESTS_SHA256, C8_RUNTIME_TESTS_SIZE_BYTES),
}
FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    C6_HYPOTHESIS_MODULE,
    C7_RUNTIME_MODULE,
    C8_RUNTIME_MODULE,
    C8_ADAPTER_MODULE,
    "numpy",
    "scipy",
    "sklearn",
    "scripts.evaluate_f1_against_manual_gold",
)
FORBIDDEN_INPUT_BASENAMES = frozenset(
    {
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
        "development_execution_manifest.json",
        "train_label_projection.json",
        "train_label_projection_attestation.json",
    }
)
FORBIDDEN_PURPOSE_TOKENS = frozenset(
    {"FULL28", "GOLD", "HOLDOUT", "LABEL", "PROJECTION", "TRUTH", "EVALUATOR"}
)
PASS_FILENAMES = frozenset({"diagnostic_read_ledger.json", "diagnostic_result.json"})
FAILURE_FILENAMES = frozenset({"diagnostic_read_ledger.json", "diagnostic_terminal.json"})
GENERIC_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})
TYPED_BINDING_KEYS = frozenset({"kind", "path", "sha256", "size_bytes"})
_HEX = frozenset("0123456789abcdef")
_FIRST_PRIMARY_BINDING_SENTINEL = object()


class CheckFailure(RuntimeError):
    def __init__(self, *, stage: str, code: str) -> None:
        super().__init__(f"{stage}:{code}")
        self.stage = stage
        self.code = code


class AnchorCreationFailure(RuntimeError):
    """Opaque marker that the current invocation created the durable anchor."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text) or text == "0" * 64:
        raise ValueError(f"{name} must be a resolved lowercase SHA-256")
    return text


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _decode_object(data: bytes, *, name: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_bytes(value) != data:
        raise ValueError(f"{name} must be one LF-canonical JSON object")
    return value


def _lexical_absolute(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} path contains parent traversal")
    return Path(os.path.abspath(os.fspath(candidate)))


def _lexical_repo_path(path: Path, *, role: str) -> Path:
    candidate = _lexical_absolute(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    if os.path.commonpath((os.fspath(root), os.fspath(candidate))) != os.fspath(root):
        raise ValueError(f"{role} leaves repository root")
    return candidate


def _reject_symlink_chain(path: Path, *, role: str, require_exists: bool = True) -> Path:
    candidate = _lexical_repo_path(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    chain: list[Path] = []
    current = candidate
    while current != root:
        chain.append(current)
        current = current.parent
    chain.append(root)
    for item in reversed(chain):
        try:
            metadata = os.lstat(item)
        except FileNotFoundError:
            if require_exists or item != candidate:
                raise
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path or ancestor is a symlink")
        if item != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor is not a directory")
    return candidate


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _assert_visible_directory_identity(path: Path, expected: tuple[int, int], *, role: str) -> None:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    metadata = os.lstat(candidate)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected:
        raise RuntimeError(f"{role} identity changed")


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate8 runner requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory)
    opened = os.fstat(descriptor)
    expected = _identity(before)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected:
        os.close(descriptor)
        raise RuntimeError(f"{role} changed during open")
    _assert_visible_directory_identity(candidate, expected, role=f"{role} visible")
    return descriptor, expected


def _read_regular_file_once(
    path: Path,
    *,
    role: str,
    after_raw_fd_open: Callable[[], None] | None = None,
    after_verified_inode: Callable[[tuple[int, int]], None] | None = None,
    before_first_read: Callable[[tuple[int, int]], None] | None = None,
) -> tuple[bytes, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{role} must be one non-hard-linked regular file")
    parent_fd, parent_identity = _open_stable_directory(candidate.parent, role=f"{role} parent")
    descriptor: int | None = None
    try:
        _assert_visible_directory_identity(candidate.parent, parent_identity, role=f"{role} parent before leaf")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate8 requires O_NOFOLLOW")
        descriptor = os.open(candidate.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        if after_raw_fd_open is not None:
            after_raw_fd_open()
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != _identity(before):
            raise ValueError(f"{role} changed during leaf open")
        if after_verified_inode is not None:
            after_verified_inode(identity)
        chunks: list[bytes] = []
        if before_first_read is not None:
            before_first_read(identity)
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
            or not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or _identity(after) != identity
            or _identity(visible) != identity
            or int(after.st_size) != len(data)
        ):
            raise ValueError(f"{role} changed during read")
        _assert_visible_directory_identity(candidate.parent, parent_identity, role=f"{role} parent after read")
        return data, identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact_source(path: Path, *, role: str, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    data, identity = _read_regular_file_once(path, role=role)
    if len(data) != size_bytes or sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA"):
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, {
        "device_id": identity[0],
        "inode": identity[1],
        "path": str(_lexical_repo_path(path, role=role).relative_to(ROOT)),
        "purpose": role,
        "sha256": sha256,
        "size_bytes": len(data),
    }


def _reject_preloaded_sources(paths: Sequence[Path]) -> None:
    current = sys.modules.get(__name__)
    identities: set[tuple[int, int]] = set()
    targets: set[Path] = set()
    for path in paths:
        candidate = _reject_symlink_chain(path, role="Candidate8 controlled source")
        identities.add(_identity(os.stat(candidate, follow_symlinks=True)))
        targets.add(candidate.resolve(strict=True))
    for name in FORBIDDEN_PRELOADED_MODULES:
        if name in sys.modules:
            raise RuntimeError(f"Candidate8 rejects preloaded module: {name}")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        loaded = getattr(module, "__file__", None)
        if not isinstance(loaded, str) or not loaded:
            continue
        path = _lexical_absolute(Path(loaded), role=f"preloaded module {name}")
        try:
            target = path.resolve(strict=True)
            identity = _identity(os.stat(path, follow_symlinks=True))
        except OSError:
            continue
        if target in targets or identity in identities:
            raise RuntimeError(f"Candidate8 rejects preloaded source alias: {name}")


def _exec_module(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError(f"Candidate8 verified module already loaded: {name}")
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
    runtime: types.ModuleType
    adapter: types.ModuleType
    hypothesis: types.ModuleType
    candidate7_runtime: types.ModuleType
    adapter_context: Any
    source_bindings: Mapping[str, Mapping[str, Any]]
    bootstrap_entries: tuple[dict[str, Any], ...]
    _context_nonce: object


def _expected_fixed_sha(name: str, supplied: str) -> str:
    _path, expected, _size = SOURCE_FIXED_BINDINGS[name]
    if _require_sha(supplied, name=f"{name} source SHA") != expected:
        raise ValueError(f"Candidate8 {name} source binding is not the cleared byte set")
    return expected


def bootstrap_verified_sources(
    *,
    runner_sha256: str,
    runner_size_bytes: int,
    runner_tests_sha256: str,
    runner_tests_size_bytes: int,
    runtime_sha256: str,
    runtime_tests_sha256: str,
    adapter_sha256: str,
    adapter_tests_sha256: str,
    hypothesis_sha256: str,
    candidate7_runtime_sha256: str,
) -> BootstrapContext:
    fixed_supplied = {
        "adapter": adapter_sha256,
        "adapter_tests": adapter_tests_sha256,
        "candidate7_runtime": candidate7_runtime_sha256,
        "hypothesis": hypothesis_sha256,
        "runtime": runtime_sha256,
        "runtime_tests": runtime_tests_sha256,
    }
    for name, supplied in fixed_supplied.items():
        _expected_fixed_sha(name, supplied)
    runner_binding = (
        RUNNER_SOURCE,
        _require_sha(runner_sha256, name="runner source SHA"),
        _strict_int(runner_size_bytes, name="runner source size"),
    )
    runner_tests_binding = (
        RUNNER_TESTS_SOURCE,
        _require_sha(runner_tests_sha256, name="runner tests SHA"),
        _strict_int(runner_tests_size_bytes, name="runner tests size"),
    )
    specs = {**SOURCE_FIXED_BINDINGS, "runner": runner_binding, "runner_tests": runner_tests_binding}
    if runner_binding[2] <= 0 or runner_tests_binding[2] <= 0:
        raise ValueError("Candidate8 source sizes must be positive")
    _reject_preloaded_sources(tuple(path for path, _sha, _size in specs.values()))
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha, size = specs[name]
        data, entry = _read_exact_source(path, role=f"CANDIDATE8_SOURCE_{name.upper()}", sha256=sha, size_bytes=size)
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate8 source inputs contain a hard-link alias")
        identities.add(identity)
        source_bytes[name] = data
        entries.append(entry)
        bindings[name] = {"path": entry["path"], "sha256": sha, "size_bytes": size}
    hypothesis = _exec_module("_candidate8_runner_verified_hypothesis", C6_HYPOTHESIS_SOURCE, source_bytes["hypothesis"], package="backend")
    candidate7_runtime = _exec_module("_candidate8_runner_verified_candidate7", C7_RUNTIME_SOURCE, source_bytes["candidate7_runtime"], package="backend")
    runtime = _exec_module("_candidate8_runner_verified_runtime", C8_RUNTIME_SOURCE, source_bytes["runtime"], package="backend")
    adapter = _exec_module("_candidate8_runner_verified_adapter", C8_ADAPTER_SOURCE, source_bytes["adapter"], package="scripts")
    runtime.verify_scientific_hypothesis_runtime(hypothesis, source_sha256=hypothesis_sha256)
    runtime.verify_candidate7_wrapper_runtime(candidate7_runtime, source_sha256=candidate7_runtime_sha256, hypothesis_runtime=hypothesis)
    if runtime.ATTEMPT_ORDINAL != 8 or runtime.ENGINEERING_ATTEMPT_ID != "CANDIDATE8_ENGINEERING_ATTEMPT_V1":
        raise ValueError("Candidate8 runtime identity mismatch")
    adapter_context = adapter.AdapterContext(
        runtime=runtime,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7_runtime,
        source_bindings={key: dict(value) for key, value in bindings.items() if key in SOURCE_FIXED_BINDINGS},
    )
    return BootstrapContext(
        runtime=runtime,
        adapter=adapter,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7_runtime,
        adapter_context=adapter_context,
        source_bindings=bindings,
        bootstrap_entries=tuple(entries),
        _context_nonce=object(),
    )


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
    first_primary_nonce: object | None = None

    def __post_init__(self) -> None:
        for entry in self.entries:
            identity = int(entry["device_id"]), int(entry["inode"])
            path = _lexical_repo_path(ROOT / str(entry["path"]), role=str(entry["purpose"]))
            prior = self.identities.get(identity)
            if prior is not None and prior != path:
                raise ValueError("Candidate8 bootstrap ledger has a hard-link alias")
            self.identities[identity] = path

    def _deny(self, path: Path, purpose: str, reason: str) -> None:
        self.denied_attempts.append({"purpose": purpose, "reason": reason})
        raise PermissionError(f"Candidate8 read denied ({reason})")

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        sha256: str,
        size_bytes: int,
        first_primary_binding: Any = None,
    ) -> None:
        candidate = _lexical_repo_path(path, role=purpose)
        upper = purpose.upper()
        lowered = tuple(part.lower() for part in candidate.parts)
        if candidate.name in FORBIDDEN_INPUT_BASENAMES:
            self._deny(candidate, purpose, "FORBIDDEN_BASENAME")
        if any(token in upper for token in FORBIDDEN_PURPOSE_TOKENS):
            self._deny(candidate, purpose, "FORBIDDEN_PURPOSE")
        if any("holdout" in part or "full28" in part or "full-28" in part for part in lowered):
            self._deny(candidate, purpose, "FORBIDDEN_PATH")
        if any(token in part for part in lowered for token in ("gold", "label", "projection", "truth")):
            self._deny(candidate, purpose, "FORBIDDEN_PATH")
        cache_root = _lexical_repo_path(ALLOWED_CACHE_ROOT, role="TRAIN cache root")
        if candidate == cache_root or cache_root in candidate.parents:
            valid_capability = (
                isinstance(first_primary_binding, FirstPrimaryBinding)
                and first_primary_binding._reader is self
                and first_primary_binding._qualification_nonce
                is self.first_primary_nonce
                and getattr(first_primary_binding, "_sentinel", None)
                is _FIRST_PRIMARY_BINDING_SENTINEL
            )
            valid_pair = valid_capability and (
                (
                    purpose == "TRAIN_FIRST_PRIMARY_V51_SEMANTIC"
                    and candidate == first_primary_binding.semantic_path
                    and sha256 == first_primary_binding.semantic_sha256
                    and size_bytes == first_primary_binding.semantic_size_bytes
                )
                or (
                    purpose == "TRAIN_FIRST_PRIMARY_V51_POOL"
                    and candidate == first_primary_binding.pool_path
                    and sha256 == first_primary_binding.pool_sha256
                    and size_bytes == first_primary_binding.pool_size_bytes
                )
            )
            if not valid_pair:
                self._deny(candidate, purpose, "FIRST_PRIMARY_CAPABILITY_REQUIRED")
        canonical = _reject_symlink_chain(candidate, role=purpose, require_exists=True)
        metadata = os.lstat(canonical)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{purpose} must be one non-hard-linked regular file")
        identity = _identity(metadata)
        prior = self.identities.get(identity)
        if prior is not None and prior != canonical:
            raise ValueError("Candidate8 input contains a hard-link alias")
        binding = {
            "identity": identity,
            "purpose": purpose,
            "sha256": _require_sha(sha256, name=f"{purpose} SHA"),
            "size_bytes": _strict_int(size_bytes, name=f"{purpose} size"),
        }
        if binding["size_bytes"] <= 0:
            raise ValueError(f"{purpose} size must be positive")
        prior_binding = self.allowed.get(canonical)
        if prior_binding is not None and prior_binding != binding:
            raise ValueError("Candidate8 conflicting authorization")
        self.allowed[canonical] = binding
        self.identities[identity] = canonical

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        candidate = _lexical_repo_path(path, role=purpose)
        binding = self.allowed.get(candidate)
        if binding is None or binding["purpose"] != purpose:
            self._deny(candidate, purpose, "NOT_EXACTLY_AUTHORIZED")
        if candidate in self.consumed_paths or candidate in self.raw_fd_opened_paths:
            self._deny(candidate, purpose, "DUPLICATE_READ_FORBIDDEN")

        def record_raw_fd_open() -> None:
            self.raw_fd_opened_paths.add(candidate)
            self.raw_fd_opened_purposes.append(purpose)

        def record_verified_inode(identity: tuple[int, int]) -> None:
            if identity != binding["identity"]:
                raise ValueError(f"Candidate8 {purpose} opened inode binding mismatch")
            if candidate not in self.raw_fd_opened_paths:
                raise ValueError(f"Candidate8 {purpose} lacks raw fd-open evidence")
            self.verified_inode_paths.add(candidate)
            self.verified_inode_purposes.append(purpose)

        def record_read_start(identity: tuple[int, int]) -> None:
            if (
                identity != binding["identity"]
                or candidate not in self.verified_inode_paths
            ):
                raise ValueError(f"Candidate8 {purpose} read-start binding mismatch")
            self.read_started_paths.add(candidate)
            self.read_started_purposes.append(purpose)

        data, identity = _read_regular_file_once(
            candidate,
            role=purpose,
            after_raw_fd_open=record_raw_fd_open,
            after_verified_inode=record_verified_inode,
            before_first_read=record_read_start,
        )
        if identity != binding["identity"] or len(data) != binding["size_bytes"] or sha256_bytes(data) != binding["sha256"]:
            raise ValueError(f"Candidate8 {purpose} exact byte binding mismatch")
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


def _read_control(reader: GuardedReader, *, purpose: str, path: Path, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    reader.authorize(path, purpose=purpose, sha256=sha256, size_bytes=size_bytes)
    data = reader.read_bytes(path, purpose=purpose)
    return data, _decode_object(data, name=purpose)


def _binding_from_attestation(
    value: Any,
    *,
    role: str,
    expected_path: Path,
    typed: bool,
    expected_kind: str | None = None,
) -> tuple[Path, str, int]:
    expected_keys = TYPED_BINDING_KEYS if typed else GENERIC_BINDING_KEYS
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="BINDING_ENVELOPE_SCHEMA_MISMATCH")
    if typed and (expected_kind is None or value.get("kind") != expected_kind):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="BINDING_KIND_MISMATCH")
    relative = Path(str(value.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="BINDING_PATH_INVALID")
    path = _lexical_repo_path(ROOT / relative, role=role)
    if path != _lexical_repo_path(expected_path, role=f"expected {role}"):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="BINDING_PATH_MISMATCH")
    size = _strict_int(value.get("size_bytes"), name=f"{role} size")
    if size <= 0:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="BINDING_SIZE_INVALID")
    return path, _require_sha(value.get("sha256"), name=f"{role} SHA"), size


def _validate_candidate7_terminal(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate7_preflight_terminal_v1"
        or value.get("attempt_ordinal") != 7
        or value.get("status") != "FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
        or value.get("continuation_authorized") is not False
        or value.get("candidate7_fit_authorized") is not False
        or value.get("candidate7_preregistration_authorized") is not False
        or value.get("label_attestation_or_projection_opened") is not False
        or value.get("holdout_score_opened") is not False
        or value.get("model_fit_performed") is not False
        or value.get("failure_stage") != "TRAIN_SOURCE_COMPATIBILITY"
        or value.get("check_code") != "TRAIN_SOURCE_COMPATIBILITY_UNEXPECTED_EXCEPTION"
        or value.get("preflight_read_ledger_sha256") != FIXED_CONTROLS["CANDIDATE7_PREFLIGHT_LEDGER"][1]
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_TERMINAL_CONTRACT_MISMATCH")
    anchor = value.get("source_inventory_consumption_anchor")
    if (
        not isinstance(anchor, dict)
        or set(anchor) != GENERIC_BINDING_KEYS
        or anchor.get("path") != str(C7_ANCHOR.relative_to(ROOT))
        or anchor.get("sha256") != FIXED_CONTROLS["CANDIDATE7_CONSUMPTION_ANCHOR"][1]
        or anchor.get("size_bytes") != FIXED_CONTROLS["CANDIDATE7_CONSUMPTION_ANCHOR"][2]
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_TERMINAL_ANCHOR_MISMATCH")


def _validate_candidate7_ledger(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {
        "assertions",
        "attempt_ordinal",
        "check_code",
        "denied_attempts",
        "engineering_attempt_id",
        "failure_stage",
        "kind",
        "reads",
        "schema_version",
        "scientific_hypothesis_id",
        "source_bindings",
        "status",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_preflight_read_ledger_v1"
        or value.get("attempt_ordinal") != 7
        or value.get("status") != "TERMINAL_FAILURE"
        or value.get("failure_stage") != "TRAIN_SOURCE_COMPATIBILITY"
        or value.get("check_code") != "TRAIN_SOURCE_COMPATIBILITY_UNEXPECTED_EXCEPTION"
        or value.get("denied_attempts") != []
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_LEDGER_CONTRACT_MISMATCH")
    assertions = value.get("assertions")
    if (
        not isinstance(assertions, dict)
        or assertions.get("all_reads_within_exact_authorized_inventory") is not True
        or assertions.get("cache_manifest_opened") is not True
        or assertions.get("exact_train_inventory_completed") is not False
        or assertions.get("full_28_catalog_or_manifest_opened") is not False
        or assertions.get("holdout_path_hash_status_or_score_opened") is not False
        or assertions.get("label_attestation_or_projection_opened") is not False
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_LEDGER_BOUNDARY_MISMATCH")
    reads = value.get("reads")
    expected_prefix = [
        "CANDIDATE7_SOURCE_HYPOTHESIS",
        "CANDIDATE7_SOURCE_PREFLIGHT",
        "CANDIDATE7_SOURCE_PREFLIGHT_TESTS",
        "CANDIDATE7_SOURCE_RUNTIME",
        "CANDIDATE7_SOURCE_RUNTIME_TESTS",
        "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
        "SEALED_SPLIT",
        "ORIGINAL_PREREGISTRATION",
        "TRAIN_ALLOWLIST",
        "CANDIDATE6_TERMINAL_DISPOSITION",
        "TRAIN_CACHE_ATTESTATION",
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_PRIMARY_V51_SEMANTIC",
        "TRAIN_PRIMARY_V51_POOL",
    ]
    entry_keys = {"case_id", "device_id", "inode", "path", "purpose", "sha256", "size_bytes"}
    if (
        not isinstance(reads, list)
        or len(reads) != len(expected_prefix)
        or [row.get("purpose") if isinstance(row, dict) else None for row in reads] != expected_prefix
        or any(not isinstance(row, dict) or set(row) != entry_keys for row in reads)
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_LEDGER_READ_VECTOR_MISMATCH")
    paths = [str(row["path"]) for row in reads]
    identities = [(row["device_id"], row["inode"]) for row in reads]
    for row in reads:
        _require_sha(row["sha256"], name="C7 ledger read SHA")
        if _strict_int(row["size_bytes"], name="C7 ledger read size") <= 0:
            raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_LEDGER_READ_BINDING_INVALID")
    semantic, pool = dict(reads[-2]), dict(reads[-1])
    if (
        len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or not isinstance(semantic.get("case_id"), str)
        or semantic.get("case_id") != pool.get("case_id")
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_LEDGER_FIRST_PAIR_MISMATCH")
    return semantic, pool


def _validate_candidate7_anchor(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate7_source_inventory_consumption_v1"
        or value.get("attempt_ordinal") != 7
        or value.get("status") != "CONSUMED_BEFORE_MANIFEST_OPEN"
        or value.get("consumed_before_manifest_open") is not True
        or value.get("manifest_opened_at_anchor_creation") is not False
        or value.get("retry_authorized") is not False
        or value.get("holdout_or_label_authorized") is not False
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_ANCHOR_CONTRACT_MISMATCH")


def _validate_candidate7_authorization(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate7_source_inventory_authorization_v1"
        or value.get("attempt_ordinal") != 7
        or value.get("status") != "AUTHORIZED_ONCE"
        or value.get("authorization_consumed_on_first_manifest_open") is not True
        or value.get("retry_after_manifest_open_authorized") is not False
        or value.get("label_or_metric_access_authorized") is not False
        or value.get("holdout_execution_authorized") is not False
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_AUTHORIZATION_CONTRACT_MISMATCH")


def _validate_candidate7_qualification(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate7_source_control_qualification_v1"
        or value.get("attempt_ordinal") != 7
        or value.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or value.get("check_code") != "ALL_SOURCE_CONTROL_CHECKS_PASSED"
        or value.get("repeatable") is not True
        or value.get("artifact_source_inventory_opened") is not False
        or value.get("cache_manifest_opened") is not False
        or value.get("holdout_or_label_opened") is not False
        or value.get("scientific_hypothesis_consumed") is not False
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_QUALIFICATION_CONTRACT_MISMATCH")


@dataclass(frozen=True)
class ControlQualification:
    reader: GuardedReader
    train_ids: tuple[str, ...]
    scan_by_case: Mapping[str, str]
    train_commitment: str
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    candidate7_disposition_bytes: bytes
    c7_semantic_evidence: Mapping[str, Any]
    c7_pool_evidence: Mapping[str, Any]
    control_binding_commitment_sha256: str
    _pair_capability_nonce: object


def _control_binding_commitment(reader: GuardedReader, *, source_prefix_length: int) -> str:
    projected = [
        {
            "path": entry["path"],
            "purpose": entry["purpose"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        for entry in reader.entries[source_prefix_length:]
        if not str(entry["purpose"]).startswith("CANDIDATE8_DIAGNOSTIC_")
    ]
    return sha256_bytes(canonical_bytes(projected))


def qualify_source_controls(
    context: BootstrapContext,
    *,
    reader: GuardedReader | None = None,
    prefix_has_c8_authority: bool = False,
) -> ControlQualification:
    if reader is None:
        reader = GuardedReader(entries=[dict(entry) for entry in context.bootstrap_entries])
    source_purposes = [str(entry["purpose"]) for entry in context.bootstrap_entries]
    expected_prefix = list(source_purposes)
    if prefix_has_c8_authority:
        expected_prefix.extend(
            [
                "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
                "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
            ]
        )
    observed_entries = reader.entries
    if (
        [str(entry["purpose"]) for entry in observed_entries] != expected_prefix
        or _entry_projection(observed_entries[: len(context.bootstrap_entries)])
        != _entry_projection(context.bootstrap_entries)
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="SOURCE_CONTROL_PREFIX_MISMATCH")
    controls_bytes: dict[str, bytes] = {}
    controls: dict[str, dict[str, Any]] = {}
    for purpose, (path, sha, size) in FIXED_CONTROLS.items():
        data, value = _read_control(reader, purpose=purpose, path=path, sha256=sha, size_bytes=size)
        controls_bytes[purpose] = data
        controls[purpose] = value
    try:
        context.runtime._validate_candidate7_disposition(
            controls_bytes["CANDIDATE7_TERMINAL_DISPOSITION"],
            expected_sha256=FIXED_CONTROLS["CANDIDATE7_TERMINAL_DISPOSITION"][1],
        )
    except (TypeError, ValueError) as error:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="C7_DISPOSITION_CONTRACT_MISMATCH") from error
    _validate_candidate7_terminal(controls["CANDIDATE7_PREFLIGHT_TERMINAL"])
    c7_semantic, c7_pool = _validate_candidate7_ledger(controls["CANDIDATE7_PREFLIGHT_LEDGER"])
    _validate_candidate7_anchor(controls["CANDIDATE7_CONSUMPTION_ANCHOR"])
    _validate_candidate7_authorization(controls["CANDIDATE7_SOURCE_AUTHORIZATION"])
    _validate_candidate7_qualification(controls["CANDIDATE7_CONTROL_QUALIFICATION"])

    split = controls["SEALED_SPLIT"]
    original = controls["ORIGINAL_PREREGISTRATION"]
    allowlist = controls["TRAIN_ALLOWLIST"]
    if (
        split.get("kind") != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
        or split.get("label_or_outcome_fields_present") is not False
        or original.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN"
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="SPLIT_OR_PREREG_MISMATCH")
    cases = allowlist.get("cases")
    if (
        allowlist.get("kind") != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
        or allowlist.get("reject_any_case_not_listed") is not True
        or not isinstance(cases, list)
        or len(cases) != 18
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="TRAIN_ALLOWLIST_MISMATCH")
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    if len(set(train_ids)) != 18 or any(not value for value in train_ids) or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="TRAIN_MEMBERSHIP_ORDER_MISMATCH")
    scan_by_case = {
        str(row["case_id"]): _require_sha(row.get("scan_sha256"), name="TRAIN scan SHA")
        for row in cases
    }
    train_commitment = sha256_bytes(
        canonical_bytes([{"case_id": case_id, "scan_sha256": scan_by_case[case_id]} for case_id in train_ids])
    )
    attestation = controls["TRAIN_CACHE_ATTESTATION"]
    if (
        set(attestation) != CACHE_ATTESTATION_KEYS
        or attestation.get("schema_version") != 1
        or attestation.get("kind") != CACHE_ATTESTATION_KIND
        or attestation.get("canonical_json") is not True
        or attestation.get("independent_validation_passed") is not True
        or attestation.get("case_count") != 18
        or attestation.get("record_count") != 36
        or attestation.get("holdout_entry_count") != 0
        or attestation.get("holdout_hash_count") != 0
        or attestation.get("holdout_status_count") != 0
        or attestation.get("optimizer_holdout_access_count") != 0
        or attestation.get("train_holdout_intersection_count") != 0
        or attestation.get("train_exact_match") is not True
        or attestation.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1]
        or attestation.get("train_allowlist_sha256") != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1]
        or attestation.get("train_set_commitment_sha256") != train_commitment
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="CACHE_ATTESTATION_BOUNDARY_MISMATCH")
    _require_sha(attestation.get("blinded_complement_commitment_sha256"), name="blinded complement commitment")
    for prefix in ("access_evidence", "builder", "staged_validator_pass", "validator"):
        typed = prefix == "staged_validator_pass"
        path, sha, size = _binding_from_attestation(
            attestation.get(f"{prefix}_binding"),
            role=f"CACHE_CUSTODY_{prefix.upper()}",
            expected_path=CUSTODY_EVIDENCE_PATHS[prefix],
            typed=typed,
            expected_kind=CUSTODY_VALIDATION_KIND if typed else None,
        )
        reader.authorize(path, purpose=f"CACHE_CUSTODY_{prefix.upper()}", sha256=sha, size_bytes=size)
        reader.read_bytes(path, purpose=f"CACHE_CUSTODY_{prefix.upper()}")
    if (attestation.get("staged_validator_pass_binding") or {}).get("sha256") != CUSTODY_PASS_SHA256:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="CUSTODY_PASS_SHA_MISMATCH")
    manifest_path, manifest_sha, manifest_size = _binding_from_attestation(
        attestation.get("manifest_binding"),
        role="TRAIN_CACHE_MANIFEST",
        expected_path=CACHE_MANIFEST,
        typed=True,
        expected_kind=CACHE_MANIFEST_KIND,
    )
    if manifest_sha != CACHE_MANIFEST_SHA256 or manifest_size != CACHE_MANIFEST_SIZE_BYTES:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="CACHE_MANIFEST_BINDING_MISMATCH")
    expected_tail = [*FIXED_CONTROLS, "CACHE_CUSTODY_ACCESS_EVIDENCE", "CACHE_CUSTODY_BUILDER", "CACHE_CUSTODY_STAGED_VALIDATOR_PASS", "CACHE_CUSTODY_VALIDATOR"]
    observed_tail = [str(entry["purpose"]) for entry in reader.entries[len(expected_prefix):]]
    opened_expected = [
        *(
            [
                "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
                "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
            ]
            if prefix_has_c8_authority
            else []
        ),
        *expected_tail,
    ]
    if (
        observed_tail != expected_tail
        or reader.raw_fd_opened_purposes != opened_expected
        or reader.verified_inode_purposes != opened_expected
        or reader.read_started_purposes != opened_expected
        or reader.denied_attempts
    ):
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="SOURCE_CONTROL_READ_INVENTORY_MISMATCH")
    pair_capability_nonce = object()
    reader.first_primary_nonce = pair_capability_nonce
    return ControlQualification(
        reader=reader,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=train_commitment,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest_size_bytes=manifest_size,
        candidate7_disposition_bytes=controls_bytes["CANDIDATE7_TERMINAL_DISPOSITION"],
        c7_semantic_evidence=c7_semantic,
        c7_pool_evidence=c7_pool,
        control_binding_commitment_sha256=_control_binding_commitment(reader, source_prefix_length=len(source_purposes)),
        _pair_capability_nonce=pair_capability_nonce,
    )


def qualification_result(context: BootstrapContext) -> dict[str, Any]:
    try:
        qualification = qualify_source_controls(context)
    except CheckFailure as error:
        return {
            "artifact_source_inventory_opened": False,
            "check_code": error.code,
            "failure_stage": error.stage,
            "holdout_or_label_opened": False,
            "repeatable": True,
            "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
        }
    except BaseException:
        return {
            "artifact_source_inventory_opened": False,
            "check_code": "UNEXPECTED_SOURCE_CONTROL_EXCEPTION",
            "failure_stage": "SOURCE_CONTROL_QUALIFICATION",
            "holdout_or_label_opened": False,
            "repeatable": True,
            "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
        }
    record_bytes = build_qualification_record(context, qualification)
    return {
        "artifact_source_inventory_opened": False,
        "cache_manifest_opened": False,
        "candidate7_consumed_first_pair_evidence_validated": True,
        "check_code": "ALL_C8_SOURCE_CONTROL_CHECKS_PASSED",
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "failure_stage": None,
        "holdout_or_label_opened": False,
        "repeatable": True,
        "scientific_hypothesis_consumed": False,
        "qualification_record_sha256": sha256_bytes(record_bytes),
        "qualification_record_size_bytes": len(record_bytes),
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }


def build_qualification_record(
    context: BootstrapContext, qualification: ControlQualification
) -> bytes:
    value = {
        "artifact_source_inventory_opened": False,
        "attempt_ordinal": 8,
        "cache_manifest_opened": False,
        "candidate7_consumed_first_pair_evidence_validated": True,
        "check_code": "ALL_C8_SOURCE_CONTROL_CHECKS_PASSED",
        "consumption_anchor_namespace": str(C8_CONSUMPTION_ANCHOR.relative_to(ROOT)),
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_or_label_opened": False,
        "kind": QUALIFICATION_KIND,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "repeatable": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": _expected_source_bindings(context),
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }
    _require_sha(
        qualification.control_binding_commitment_sha256,
        name="qualification control commitment",
    )
    return canonical_bytes(value)


_SEAL_SENTINEL = object()
_OUTPUT_SEAL_SENTINEL = object()


@dataclass(frozen=True)
class FreshOutputSeal:
    path: Path
    parent_identity: tuple[int, int]
    _sentinel: object


@dataclass(frozen=True)
class AuthorizationSeal:
    reader: GuardedReader
    authorization_sha256: str
    authorization_size_bytes: int
    qualification_sha256: str
    qualification_size_bytes: int
    control_binding_commitment_sha256: str
    output_freshness: FreshOutputSeal
    source_binding_commitment_sha256: str
    authority_prefix_commitment_sha256: str
    anchor_parent_identity: tuple[int, int]
    _context_nonce: object
    _sentinel: object


@dataclass(frozen=True)
class ConsumptionSeal:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    _sentinel: object


def _expected_source_bindings(context: BootstrapContext) -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in sorted(context.source_bindings.items())}


def _entry_projection(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "device_id": int(entry["device_id"]),
            "inode": int(entry["inode"]),
            "path": str(entry["path"]),
            "purpose": str(entry["purpose"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
        for entry in entries
    ]


def _entry_commitment(entries: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes(_entry_projection(entries)))


def _purpose_vector_commitment(purposes: Sequence[str]) -> str:
    return sha256_bytes(
        canonical_bytes(
            [
                {"ordinal": ordinal, "purpose": str(purpose)}
                for ordinal, purpose in enumerate(purposes)
            ]
        )
    )


def _validate_authorization_seal(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    *,
    require_qualified_controls: bool,
) -> None:
    if (
        not isinstance(authorization, AuthorizationSeal)
        or authorization._sentinel is not _SEAL_SENTINEL
        or authorization._context_nonce is not context._context_nonce
        or authorization.source_binding_commitment_sha256
        != _source_binding_commitment(context)
    ):
        raise PermissionError("Candidate8 authorization seal/context mismatch")
    reader = authorization.reader
    source_entries = [dict(entry) for entry in context.bootstrap_entries]
    observed = reader.entries
    if len(observed) < len(source_entries) + 2 or _entry_projection(observed[: len(source_entries)]) != _entry_projection(source_entries):
        raise PermissionError("Candidate8 authorization bootstrap prefix mismatch")
    auth_entry = observed[len(source_entries)]
    qualification_entry = observed[len(source_entries) + 1]
    expected_auth = {
        "path": str(C8_AUTHORIZATION.relative_to(ROOT)),
        "purpose": "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        "sha256": authorization.authorization_sha256,
        "size_bytes": authorization.authorization_size_bytes,
    }
    expected_qualification = {
        "path": str(C8_QUALIFICATION.relative_to(ROOT)),
        "purpose": "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
        "sha256": authorization.qualification_sha256,
        "size_bytes": authorization.qualification_size_bytes,
    }
    for entry, expected in ((auth_entry, expected_auth), (qualification_entry, expected_qualification)):
        if any(entry.get(key) != value for key, value in expected.items()):
            raise PermissionError("Candidate8 authorization entry binding mismatch")
    prefix = observed[: len(source_entries) + 2]
    if _entry_commitment(prefix) != authorization.authority_prefix_commitment_sha256:
        raise PermissionError("Candidate8 authorization prefix commitment mismatch")
    opened_expected = [
        "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
    ]
    if require_qualified_controls:
        expected_tail = [
            *FIXED_CONTROLS,
            "CACHE_CUSTODY_ACCESS_EVIDENCE",
            "CACHE_CUSTODY_BUILDER",
            "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
            "CACHE_CUSTODY_VALIDATOR",
        ]
        if [str(entry["purpose"]) for entry in observed[len(prefix) :]] != expected_tail:
            raise PermissionError("Candidate8 authorization control vector mismatch")
        if _control_binding_commitment(reader, source_prefix_length=len(source_entries)) != authorization.control_binding_commitment_sha256:
            raise PermissionError("Candidate8 authorization control commitment mismatch")
        paths = [str(entry["path"]) for entry in observed]
        identities = [(int(entry["device_id"]), int(entry["inode"])) for entry in observed]
        if len(paths) != len(set(paths)) or len(identities) != len(set(identities)) or reader.denied_attempts:
            raise PermissionError("Candidate8 authorization read vector is not unique")
        opened_expected.extend(expected_tail)
    if (
        reader.raw_fd_opened_purposes != opened_expected
        or reader.verified_inode_purposes != opened_expected
        or reader.read_started_purposes != opened_expected
    ):
        raise PermissionError("Candidate8 authorization open/read-start vector mismatch")
    parent_fd, parent_identity = _open_stable_directory(
        C8_CONSUMPTION_ANCHOR.parent,
        role="Candidate8 authorized anchor parent revalidation",
    )
    os.close(parent_fd)
    if parent_identity != authorization.anchor_parent_identity:
        raise PermissionError("Candidate8 authorized anchor parent identity changed")


def _binding(path: Path, sha256: str, size_bytes: int, *, kind: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    if kind is not None:
        value["kind"] = kind
    return value


def authorize_diagnostic_once(
    context: BootstrapContext,
    *,
    reader: GuardedReader,
    authorization_sha256: str,
    authorization_size_bytes: int,
    output_freshness: FreshOutputSeal,
) -> AuthorizationSeal:
    revalidate_fresh_output(output_freshness)
    source_purposes = [str(entry["purpose"]) for entry in context.bootstrap_entries]
    if [str(entry["purpose"]) for entry in reader.entries] != source_purposes:
        raise PermissionError("Candidate8 authorization requires exact source bootstrap")
    auth_path = _reject_symlink_chain(C8_AUTHORIZATION, role="Candidate8 diagnostic authorization", require_exists=True)
    reader.authorize(
        auth_path,
        purpose="CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        sha256=_require_sha(authorization_sha256, name="Candidate8 authorization SHA"),
        size_bytes=_strict_int(authorization_size_bytes, name="Candidate8 authorization size"),
    )
    auth_bytes = reader.read_bytes(auth_path, purpose="CANDIDATE8_DIAGNOSTIC_AUTHORIZATION")
    authorization = _decode_object(auth_bytes, name="CANDIDATE8_DIAGNOSTIC_AUTHORIZATION")
    expected_keys = {
        "attempt_ordinal",
        "authorization_consumed_before_manifest_open",
        "consumption_anchor_namespace",
        "continuation_authorized",
        "engineering_attempt_id",
        "full28_access_authorized",
        "holdout_access_authorized",
        "kind",
        "label_evaluator_metric_fit_access_authorized",
        "manifest_binding",
        "output_namespace",
        "qualification_record_binding",
        "record_selector",
        "retry_after_anchor_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "source_bindings",
        "status",
    }
    if (
        set(authorization) != expected_keys
        or authorization.get("schema_version") != 1
        or authorization.get("kind") != AUTHORIZATION_KIND
        or authorization.get("status") != "AUTHORIZED_ONCE"
        or authorization.get("attempt_ordinal") != 8
        or authorization.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or authorization.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or authorization.get("record_selector") != RECORD_SELECTOR
        or authorization.get("authorization_consumed_before_manifest_open") is not True
        or authorization.get("retry_after_anchor_authorized") is not False
        or authorization.get("continuation_authorized") is not False
        or authorization.get("label_evaluator_metric_fit_access_authorized") is not False
        or authorization.get("full28_access_authorized") is not False
        or authorization.get("holdout_access_authorized") is not False
        or authorization.get("output_namespace") != str(OUTPUT_ROOT.relative_to(ROOT))
        or authorization.get("consumption_anchor_namespace") != str(C8_CONSUMPTION_ANCHOR.relative_to(ROOT))
        or authorization.get("source_bindings") != _expected_source_bindings(context)
        or authorization.get("manifest_binding")
        != _binding(CACHE_MANIFEST, CACHE_MANIFEST_SHA256, CACHE_MANIFEST_SIZE_BYTES, kind=CACHE_MANIFEST_KIND)
    ):
        raise CheckFailure(stage="DIAGNOSTIC_AUTHORIZATION", code="AUTHORIZATION_CONTRACT_MISMATCH")
    qualification_path, qualification_sha, qualification_size = _binding_from_attestation(
        authorization.get("qualification_record_binding"),
        role="CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
        expected_path=C8_QUALIFICATION,
        typed=True,
        expected_kind=QUALIFICATION_KIND,
    )
    reader.authorize(
        qualification_path,
        purpose="CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
        sha256=qualification_sha,
        size_bytes=qualification_size,
    )
    qualification = _decode_object(
        reader.read_bytes(qualification_path, purpose="CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD"),
        name="CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
    )
    qualification_keys = {
        "artifact_source_inventory_opened",
        "attempt_ordinal",
        "cache_manifest_opened",
        "candidate7_consumed_first_pair_evidence_validated",
        "check_code",
        "consumption_anchor_namespace",
        "control_binding_commitment_sha256",
        "engineering_attempt_id",
        "holdout_or_label_opened",
        "kind",
        "output_namespace",
        "repeatable",
        "schema_version",
        "scientific_hypothesis_consumed",
        "scientific_hypothesis_id",
        "source_bindings",
        "status",
    }
    if (
        set(qualification) != qualification_keys
        or qualification.get("schema_version") != 1
        or qualification.get("kind") != QUALIFICATION_KIND
        or qualification.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or qualification.get("check_code") != "ALL_C8_SOURCE_CONTROL_CHECKS_PASSED"
        or qualification.get("attempt_ordinal") != 8
        or qualification.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or qualification.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or qualification.get("source_bindings") != _expected_source_bindings(context)
        or qualification.get("repeatable") is not True
        or qualification.get("candidate7_consumed_first_pair_evidence_validated") is not True
        or qualification.get("artifact_source_inventory_opened") is not False
        or qualification.get("cache_manifest_opened") is not False
        or qualification.get("holdout_or_label_opened") is not False
        or qualification.get("scientific_hypothesis_consumed") is not False
        or qualification.get("output_namespace") != str(OUTPUT_ROOT.relative_to(ROOT))
        or qualification.get("consumption_anchor_namespace") != str(C8_CONSUMPTION_ANCHOR.relative_to(ROOT))
    ):
        raise CheckFailure(stage="DIAGNOSTIC_AUTHORIZATION", code="QUALIFICATION_CONTRACT_MISMATCH")
    control_commitment = _require_sha(
        qualification.get("control_binding_commitment_sha256"),
        name="Candidate8 control binding commitment",
    )
    anchor_parent_fd, anchor_parent_identity = _open_stable_directory(
        C8_CONSUMPTION_ANCHOR.parent,
        role="Candidate8 anchor parent at authorization",
    )
    os.close(anchor_parent_fd)
    seal = AuthorizationSeal(
        reader=reader,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        qualification_sha256=qualification_sha,
        qualification_size_bytes=qualification_size,
        control_binding_commitment_sha256=control_commitment,
        output_freshness=output_freshness,
        source_binding_commitment_sha256=_source_binding_commitment(context),
        authority_prefix_commitment_sha256=_entry_commitment(reader.entries),
        anchor_parent_identity=anchor_parent_identity,
        _context_nonce=context._context_nonce,
        _sentinel=_SEAL_SENTINEL,
    )
    _validate_authorization_seal(
        context, seal, require_qualified_controls=False
    )
    return seal


def _source_binding_commitment(context: BootstrapContext) -> str:
    return sha256_bytes(canonical_bytes(_expected_source_bindings(context)))


def consume_diagnostic_authorization(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
) -> ConsumptionSeal:
    _validate_authorization_seal(
        context, authorization, require_qualified_controls=True
    )
    anchor = _lexical_repo_path(C8_CONSUMPTION_ANCHOR, role="Candidate8 consumption anchor")
    expected = _lexical_repo_path(
        ROOT / "research/f1_aug30_train18_candidate8_first_primary_diagnostic_consumption_v1.json",
        role="frozen Candidate8 consumption anchor",
    )
    if anchor != expected:
        raise ValueError("Candidate8 consumption anchor namespace is frozen")
    value = {
        "attempt_ordinal": 8,
        "authorization_binding": _binding(C8_AUTHORIZATION, authorization.authorization_sha256, authorization.authorization_size_bytes),
        "consumed_before_manifest_open": True,
        "control_binding_commitment_sha256": authorization.control_binding_commitment_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_authorized": False,
        "kind": CONSUMPTION_KIND,
        "manifest_binding": _binding(CACHE_MANIFEST, CACHE_MANIFEST_SHA256, CACHE_MANIFEST_SIZE_BYTES, kind=CACHE_MANIFEST_KIND),
        "manifest_opened_at_anchor_creation": False,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "qualification_record_binding": _binding(C8_QUALIFICATION, authorization.qualification_sha256, authorization.qualification_size_bytes, kind=QUALIFICATION_KIND),
        "record_selector": RECORD_SELECTOR,
        "retry_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": _source_binding_commitment(context),
        "status": "CONSUMED_BEFORE_MANIFEST_OPEN",
    }
    data = canonical_bytes(value)
    parent_fd, parent_identity = _open_stable_directory(anchor.parent, role="Candidate8 consumption anchor parent")
    if parent_identity != authorization.anchor_parent_identity:
        os.close(parent_fd)
        raise PermissionError("Candidate8 anchor parent changed before consumption")
    descriptor: int | None = None
    file_identity: tuple[int, int] | None = None
    created_by_this_invocation = False
    try:
        _assert_visible_directory_identity(anchor.parent, parent_identity, role="Candidate8 visible anchor parent")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate8 requires O_NOFOLLOW")
        descriptor = os.open(anchor.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o400, dir_fd=parent_fd)
        created_by_this_invocation = True
        os.fsync(parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate8 anchor write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        file_identity = _identity(metadata)
        visible = os.stat(anchor.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != len(data)
            or _identity(visible) != file_identity
        ):
            raise RuntimeError("Candidate8 anchor write mismatch")
        os.fsync(parent_fd)
        _assert_visible_directory_identity(anchor.parent, parent_identity, role="Candidate8 visible anchor parent after write")
        canonical = _reject_symlink_chain(anchor, role="Candidate8 visible anchor after write", require_exists=True)
        leaf = os.lstat(canonical)
        if not stat.S_ISREG(leaf.st_mode) or leaf.st_nlink != 1 or _identity(leaf) != file_identity or int(leaf.st_size) != len(data):
            raise RuntimeError("Candidate8 canonical anchor changed")
    except FileExistsError as error:
        raise PermissionError("Candidate8 diagnostic authorization is already consumed") from error
    except BaseException as error:
        if created_by_this_invocation:
            raise AnchorCreationFailure("Candidate8 anchor creation failed after durable consumption") from error
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
            if created_by_this_invocation:
                raise AnchorCreationFailure(
                    "Candidate8 anchor cleanup failed after durable consumption"
                ) from close_error
            raise close_error
    if file_identity is None:
        raise RuntimeError("Candidate8 anchor lacks captured identity")
    seal = ConsumptionSeal(
        path=anchor,
        data=data,
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        file_identity=file_identity,
        parent_identity=parent_identity,
        _sentinel=_SEAL_SENTINEL,
    )
    try:
        revalidate_consumption_anchor(seal)
    except BaseException as error:
        raise AnchorCreationFailure(
            "Candidate8 anchor final validation failed after durable consumption"
        ) from error
    return seal


def revalidate_consumption_anchor(seal: ConsumptionSeal) -> None:
    if not isinstance(seal, ConsumptionSeal) or seal._sentinel is not _SEAL_SENTINEL:
        raise PermissionError("Candidate8 consumption anchor seal is invalid")
    parent_fd, parent_identity = _open_stable_directory(seal.path.parent, role="Candidate8 anchor parent revalidation")
    descriptor: int | None = None
    try:
        if parent_identity != seal.parent_identity:
            raise RuntimeError("Candidate8 visible anchor parent changed")
        _assert_visible_directory_identity(seal.path.parent, seal.parent_identity, role="Candidate8 visible anchor parent before manifest")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate8 requires O_NOFOLLOW")
        descriptor = os.open(seal.path.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or _identity(metadata) != seal.file_identity or int(metadata.st_size) != seal.size_bytes:
            raise RuntimeError("Candidate8 anchor changed before manifest")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or _identity(after) != seal.file_identity
            or int(after.st_size) != len(data)
            or data != seal.data
            or sha256_bytes(data) != seal.sha256
        ):
            raise RuntimeError("Candidate8 anchor content changed before manifest")
        visible = os.stat(seal.path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(visible) != seal.file_identity:
            raise RuntimeError("Candidate8 visible anchor leaf changed")
        canonical = _reject_symlink_chain(seal.path, role="Candidate8 canonical anchor before manifest", require_exists=True)
        if _identity(os.lstat(canonical)) != seal.file_identity:
            raise RuntimeError("Candidate8 canonical anchor leaf changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


@dataclass(frozen=True)
class FirstPrimaryBinding:
    semantic_path: Path
    semantic_sha256: str
    semantic_size_bytes: int
    pool_path: Path
    pool_sha256: str
    pool_size_bytes: int
    commitment_sha256: str
    _reader: GuardedReader
    _qualification_nonce: object
    _sentinel: object


def _inside_cache_root(path: Path) -> bool:
    candidate = _lexical_repo_path(path, role="TRAIN cache artifact")
    root = _lexical_repo_path(ALLOWED_CACHE_ROOT, role="TRAIN cache root")
    return candidate == root or root in candidate.parents


def _validate_manifest_first_primary(
    manifest: Mapping[str, Any], qualification: ControlQualification
) -> FirstPrimaryBinding:
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
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != CACHE_MANIFEST_KIND
        or manifest.get("source_only") is not True
        or manifest.get("case_count") != 18
        or manifest.get("record_count") != 36
        or manifest.get("holdout_entry_count") != 0
        or manifest.get("split_sha256") != FIXED_CONTROLS["SEALED_SPLIT"][1]
        or manifest.get("train_allowlist_sha256") != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1]
        or manifest.get("train_set_commitment_sha256") != qualification.train_commitment
        or manifest.get("v51_scan_only_manifest_sha256") != V51_SCAN_ONLY_MANIFEST_SHA256
    ):
        raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MANIFEST_HEADER_MISMATCH")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MANIFEST_RECORD_VECTOR_MISMATCH")
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
    paths: set[str] = set()
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    primary_hashes: dict[str, tuple[str, str]] = {}
    for raw, expected_pair in zip(records, expected_pairs):
        if not isinstance(raw, dict) or set(raw) != record_keys:
            raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_RECORD_SCHEMA_MISMATCH")
        case_id, pass_name = expected_pair
        if (str(raw.get("case_id")), str(raw.get("pass"))) != expected_pair:
            raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MEMBERSHIP_ORDER_MISMATCH")
        runtime_id = str(raw.get("runtime_case_id") or "")
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id
            or case_by_runtime.setdefault(runtime_id, case_id) != case_id
            or _require_sha(raw.get("scan_sha256"), name="manifest scan SHA") != qualification.scan_by_case[case_id]
        ):
            raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_RECORD_IDENTITY_MISMATCH")
        semantic_relative = str(raw.get("semantic_path") or "")
        pool_relative = str(raw.get("pool_path") or "")
        for relative in (semantic_relative, pool_relative):
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts or relative in paths:
                raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="CACHE_RECORD_PATH_MISMATCH")
            paths.add(relative)
        semantic_sha = _require_sha(raw.get("semantic_sha256"), name="manifest semantic SHA")
        pool_sha = _require_sha(raw.get("pool_sha256"), name="manifest pool SHA")
        if pass_name == "primary":
            primary_hashes[case_id] = semantic_sha, pool_sha
        elif primary_hashes.get(case_id) != (semantic_sha, pool_sha):
            raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="PRIMARY_REPLAY_HASH_MISMATCH")
    first = records[0]
    runtime_id = str(first["runtime_case_id"])
    semantic_path = _lexical_repo_path(ROOT / str(first["semantic_path"]), role="first primary semantic")
    pool_path = _lexical_repo_path(ROOT / str(first["pool_path"]), role="first primary pool")
    expected_semantic = _lexical_repo_path(ALLOWED_CACHE_ROOT / "primary" / runtime_id / "semantic_result.json", role="expected first primary semantic")
    expected_pool = _lexical_repo_path(ALLOWED_CACHE_ROOT / "primary" / runtime_id / "pool" / f"{runtime_id}.json", role="expected first primary pool")
    c7_semantic = qualification.c7_semantic_evidence
    c7_pool = qualification.c7_pool_evidence
    if (
        not _inside_cache_root(semantic_path)
        or not _inside_cache_root(pool_path)
        or semantic_path != expected_semantic
        or pool_path != expected_pool
        or c7_semantic.get("case_id") != first["case_id"]
        or c7_pool.get("case_id") != first["case_id"]
        or c7_semantic.get("path") != first["semantic_path"]
        or c7_pool.get("path") != first["pool_path"]
        or c7_semantic.get("sha256") != first["semantic_sha256"]
        or c7_pool.get("sha256") != first["pool_sha256"]
        or c7_semantic.get("purpose") != "TRAIN_PRIMARY_V51_SEMANTIC"
        or c7_pool.get("purpose") != "TRAIN_PRIMARY_V51_POOL"
    ):
        raise CheckFailure(stage="CACHE_MANIFEST_VALIDATION", code="C7_CONSUMED_FIRST_PAIR_BINDING_MISMATCH")
    semantic_size = _strict_int(c7_semantic.get("size_bytes"), name="first semantic size")
    pool_size = _strict_int(c7_pool.get("size_bytes"), name="first pool size")
    projection = {
        "manifest_sha256": qualification.manifest_sha256,
        "pool_path": str(first["pool_path"]),
        "pool_sha256": str(first["pool_sha256"]),
        "pool_size_bytes": pool_size,
        "record_selector": RECORD_SELECTOR,
        "semantic_path": str(first["semantic_path"]),
        "semantic_sha256": str(first["semantic_sha256"]),
        "semantic_size_bytes": semantic_size,
    }
    return FirstPrimaryBinding(
        semantic_path=semantic_path,
        semantic_sha256=str(first["semantic_sha256"]),
        semantic_size_bytes=semantic_size,
        pool_path=pool_path,
        pool_sha256=str(first["pool_sha256"]),
        pool_size_bytes=pool_size,
        commitment_sha256=sha256_bytes(canonical_bytes(projection)),
        _reader=qualification.reader,
        _qualification_nonce=qualification._pair_capability_nonce,
        _sentinel=_FIRST_PRIMARY_BINDING_SENTINEL,
    )


def read_and_diagnose_first_primary(
    context: BootstrapContext,
    qualification: ControlQualification,
    consumption: ConsumptionSeal,
    *,
    after_diagnostic_call_returned: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], FirstPrimaryBinding]:
    revalidate_consumption_anchor(consumption)
    reader = qualification.reader
    reader.authorize(
        qualification.manifest_path,
        purpose="TRAIN_CACHE_MANIFEST",
        sha256=qualification.manifest_sha256,
        size_bytes=qualification.manifest_size_bytes,
    )
    manifest = _decode_object(
        reader.read_bytes(qualification.manifest_path, purpose="TRAIN_CACHE_MANIFEST"),
        name="TRAIN_CACHE_MANIFEST",
    )
    revalidate_consumption_anchor(consumption)
    first = _validate_manifest_first_primary(manifest, qualification)
    reader.authorize(
        first.semantic_path,
        purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        sha256=first.semantic_sha256,
        size_bytes=first.semantic_size_bytes,
        first_primary_binding=first,
    )
    reader.authorize(
        first.pool_path,
        purpose="TRAIN_FIRST_PRIMARY_V51_POOL",
        sha256=first.pool_sha256,
        size_bytes=first.pool_size_bytes,
        first_primary_binding=first,
    )
    if (
        reader.allowed[first.semantic_path]["identity"]
        != (
            int(qualification.c7_semantic_evidence["device_id"]),
            int(qualification.c7_semantic_evidence["inode"]),
        )
        or reader.allowed[first.pool_path]["identity"]
        != (
            int(qualification.c7_pool_evidence["device_id"]),
            int(qualification.c7_pool_evidence["inode"]),
        )
    ):
        raise CheckFailure(
            stage="FIRST_PRIMARY_BINDING",
            code="C7_CONSUMED_FIRST_PAIR_INODE_MISMATCH",
        )
    revalidate_consumption_anchor(consumption)
    semantic_bytes = reader.read_bytes(first.semantic_path, purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC")
    pool_bytes = reader.read_bytes(first.pool_path, purpose="TRAIN_FIRST_PRIMARY_V51_POOL")
    revalidate_consumption_anchor(consumption)
    result = context.adapter.diagnose_bound_pair(
        context.adapter_context,
        semantic_bytes=semantic_bytes,
        pool_bytes=pool_bytes,
        candidate7_disposition_bytes=qualification.candidate7_disposition_bytes,
        semantic_sha256=first.semantic_sha256,
        pool_sha256=first.pool_sha256,
        candidate7_disposition_sha256=FIXED_CONTROLS["CANDIDATE7_TERMINAL_DISPOSITION"][1],
    )
    if after_diagnostic_call_returned is not None:
        after_diagnostic_call_returned()
    try:
        context.runtime._validate_result(result)
    except BaseException as error:
        raise CheckFailure(
            stage="DIAGNOSTIC_RESULT_VALIDATION",
            code="DIAGNOSTIC_RESULT_SCHEMA_MISMATCH",
        ) from error
    if result.get("check_code") not in {context.runtime.PASS_CODE, *context.runtime.FAILURE_CODES.values()}:
        raise CheckFailure(stage="DIAGNOSTIC_RESULT_VALIDATION", code="DIAGNOSTIC_CODE_OUTSIDE_FROZEN_SET")
    return dict(result), first


def _validate_exact_read_inventory(context: BootstrapContext, qualification: ControlQualification) -> None:
    expected = [str(entry["purpose"]) for entry in context.bootstrap_entries]
    artifact_expected = [
        "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
        *FIXED_CONTROLS,
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ]
    expected.extend(artifact_expected)
    entries = qualification.reader.entries
    paths = [str(entry["path"]) for entry in entries]
    identities = [(int(entry["device_id"]), int(entry["inode"])) for entry in entries]
    if (
        [str(entry["purpose"]) for entry in entries] != expected
        or qualification.reader.raw_fd_opened_purposes != artifact_expected
        or qualification.reader.verified_inode_purposes != artifact_expected
        or qualification.reader.read_started_purposes != artifact_expected
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or qualification.reader.denied_attempts
    ):
        raise CheckFailure(stage="DIAGNOSTIC_READ_INVENTORY", code="EXACT_READ_INVENTORY_MISMATCH")


def _ledger_payload(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    consumption: ConsumptionSeal | None,
    *,
    diagnostic_check_code: str | None,
    diagnostic_failure_phase: str | None,
    engineering_failure_stage: str | None,
    engineering_check_code: str | None,
    first_primary_commitment_sha256: str | None,
    exact_read_inventory_validated: bool,
    diagnostic_call_returned: bool,
    diagnostic_result_accepted_for_publication: bool,
) -> dict[str, Any]:
    raw_opened_purposes = set(qualification.reader.raw_fd_opened_purposes)
    verified_inode_purposes = set(qualification.reader.verified_inode_purposes)
    read_started_purposes = set(qualification.reader.read_started_purposes)
    value = {
        "assertions": {
            "authorization_consumption_attempted": True,
            "consumption_anchor_sealed": consumption is not None,
            "denied_read_attempted": bool(qualification.reader.denied_attempts),
            "diagnostic_call_returned": diagnostic_call_returned,
            "diagnostic_result_accepted_for_publication": (
                diagnostic_result_accepted_for_publication
            ),
            "evaluator_metric_fit_opened_or_started": False,
            "exact_read_inventory_validated": exact_read_inventory_validated,
            "full28_opened": False,
            "holdout_opened": False,
            "labels_opened": False,
            "manifest_inode_verified": (
                "TRAIN_CACHE_MANIFEST" in verified_inode_purposes
            ),
            "manifest_opened": "TRAIN_CACHE_MANIFEST" in raw_opened_purposes,
            "manifest_read_started": "TRAIN_CACHE_MANIFEST" in read_started_purposes,
            "only_first_primary_pool_inode_verified": (
                "TRAIN_FIRST_PRIMARY_V51_POOL" in verified_inode_purposes
            ),
            "only_first_primary_pool_opened": (
                "TRAIN_FIRST_PRIMARY_V51_POOL" in raw_opened_purposes
            ),
            "only_first_primary_pool_read_started": (
                "TRAIN_FIRST_PRIMARY_V51_POOL" in read_started_purposes
            ),
            "only_first_primary_semantic_inode_verified": (
                "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in verified_inode_purposes
            ),
            "only_first_primary_semantic_opened": (
                "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in raw_opened_purposes
            ),
            "only_first_primary_semantic_read_started": (
                "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in read_started_purposes
            ),
            "replay_or_later_pair_opened": False,
        },
        "attempt_ordinal": 8,
        "authorization_binding_commitment_sha256": sha256_bytes(
            canonical_bytes(
                _binding(C8_AUTHORIZATION, authorization.authorization_sha256, authorization.authorization_size_bytes)
            )
        ),
        "check_code": diagnostic_check_code,
        "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
        "consumption_anchor_sha256": None if consumption is None else consumption.sha256,
        "diagnostic_failure_phase": diagnostic_failure_phase,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": engineering_check_code,
        "engineering_failure_stage": engineering_failure_stage,
        "raw_fd_opened_inventory_commitment_sha256": _purpose_vector_commitment(
            qualification.reader.raw_fd_opened_purposes
        ),
        "first_primary_binding_commitment_sha256": first_primary_commitment_sha256,
        "kind": LEDGER_KIND,
        "manifest_binding_commitment_sha256": sha256_bytes(
            canonical_bytes(_binding(CACHE_MANIFEST, CACHE_MANIFEST_SHA256, CACHE_MANIFEST_SIZE_BYTES, kind=CACHE_MANIFEST_KIND))
        ),
        "read_inventory_commitment_sha256": _entry_commitment(
            qualification.reader.entries
        ),
        "read_started_inventory_commitment_sha256": _purpose_vector_commitment(
            qualification.reader.read_started_purposes
        ),
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": _source_binding_commitment(context),
        "status": (
            "DIAGNOSTIC_RECORDED"
            if diagnostic_result_accepted_for_publication
            else "TERMINAL_ENGINEERING_FAILURE"
        ),
        "verified_inode_inventory_commitment_sha256": _purpose_vector_commitment(
            qualification.reader.verified_inode_purposes
        ),
    }
    _assert_value_free_ledger(value)
    return value


def _assert_value_free_ledger(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "assertions",
        "attempt_ordinal",
        "authorization_binding_commitment_sha256",
        "check_code",
        "control_binding_commitment_sha256",
        "consumption_anchor_sha256",
        "diagnostic_failure_phase",
        "engineering_attempt_id",
        "engineering_check_code",
        "engineering_failure_stage",
        "first_primary_binding_commitment_sha256",
        "kind",
        "manifest_binding_commitment_sha256",
        "read_inventory_commitment_sha256",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
        "schema_version",
        "scientific_hypothesis_id",
        "source_binding_commitment_sha256",
        "status",
        "verified_inode_inventory_commitment_sha256",
    }
    assertions = value.get("assertions")
    assertion_keys = {
        "authorization_consumption_attempted",
        "consumption_anchor_sealed",
        "denied_read_attempted",
        "diagnostic_call_returned",
        "diagnostic_result_accepted_for_publication",
        "evaluator_metric_fit_opened_or_started",
        "exact_read_inventory_validated",
        "full28_opened",
        "holdout_opened",
        "labels_opened",
        "manifest_inode_verified",
        "manifest_opened",
        "manifest_read_started",
        "only_first_primary_pool_inode_verified",
        "only_first_primary_pool_opened",
        "only_first_primary_pool_read_started",
        "only_first_primary_semantic_inode_verified",
        "only_first_primary_semantic_opened",
        "only_first_primary_semantic_read_started",
        "replay_or_later_pair_opened",
    }
    if (
        set(value) != expected_keys
        or not isinstance(assertions, dict)
        or set(assertions) != assertion_keys
        or any(not isinstance(item, bool) for item in assertions.values())
    ):
        raise ValueError("Candidate8 ledger schema mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != LEDGER_KIND
        or value.get("attempt_ordinal") != 8
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or assertions["authorization_consumption_attempted"] is not True
    ):
        raise ValueError("Candidate8 ledger identity contract mismatch")
    forbidden_fragments = (
        "case_id",
        "runtime_case",
        "semantic_path",
        "semantic_sha",
        "pool_path",
        "pool_sha",
        "prediction",
        "geometry",
        "exception",
        "feature_value",
        "source_kind",
    )
    encoded_keys = json.dumps(list(value), sort_keys=True)
    if any(fragment in encoded_keys for fragment in forbidden_fragments):
        raise ValueError("Candidate8 ledger exposes forbidden source detail")
    for key in (
        "authorization_binding_commitment_sha256",
        "control_binding_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "read_inventory_commitment_sha256",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
        "verified_inode_inventory_commitment_sha256",
        "source_binding_commitment_sha256",
    ):
        _require_sha(value.get(key), name=f"ledger {key}")
    first = value.get("first_primary_binding_commitment_sha256")
    if first is not None:
        _require_sha(first, name="first primary commitment")
    anchor = value.get("consumption_anchor_sha256")
    if anchor is not None:
        _require_sha(anchor, name="consumption anchor SHA")
    if assertions["consumption_anchor_sealed"] is not (anchor is not None):
        raise ValueError("Candidate8 ledger anchor declaration mismatch")
    if any(
        assertions[key]
        for key in (
            "evaluator_metric_fit_opened_or_started",
            "full28_opened",
            "holdout_opened",
            "labels_opened",
            "replay_or_later_pair_opened",
        )
    ):
        raise ValueError("Candidate8 ledger boundary declaration is unsafe")
    for opened_key, verified_key, started_key in (
        (
            "manifest_opened",
            "manifest_inode_verified",
            "manifest_read_started",
        ),
        (
            "only_first_primary_pool_opened",
            "only_first_primary_pool_inode_verified",
            "only_first_primary_pool_read_started",
        ),
        (
            "only_first_primary_semantic_opened",
            "only_first_primary_semantic_inode_verified",
            "only_first_primary_semantic_read_started",
        ),
    ):
        if assertions[verified_key] and not assertions[opened_key]:
            raise ValueError("Candidate8 inode verification lacks raw fd-open evidence")
        if assertions[started_key] and not assertions[verified_key]:
            raise ValueError("Candidate8 read-start declaration lacks verified inode")
    diagnostic_accepted = assertions["diagnostic_result_accepted_for_publication"]
    if diagnostic_accepted:
        expected_phase_by_code = {
            "C8_ALL_DIAGNOSTIC_PHASES_PASSED": None,
            "C8_SEMANTIC_VALIDATION_FAILED": "D01_SEMANTIC_VALIDATION",
            "C8_POOL_PARSE_SANITIZE_FAILED": "D02_POOL_PARSE_SANITIZE",
            "C8_OWNERSHIP_EXERCISE_FAILED": "D03_OWNERSHIP_EXERCISE",
            "C8_FIXED_RULE_EXERCISE_FAILED": "D04_FIXED_RULE_EXERCISE",
            "C8_WRAPPER_RESULT_VALIDATION_FAILED": "D05_WRAPPER_RESULT_VALIDATION",
        }
        if (
            not assertions["diagnostic_call_returned"]
            or not assertions["consumption_anchor_sealed"]
            or assertions["denied_read_attempted"]
            or not assertions["exact_read_inventory_validated"]
            or not assertions["manifest_opened"]
            or not assertions["manifest_inode_verified"]
            or not assertions["manifest_read_started"]
            or not assertions["only_first_primary_pool_opened"]
            or not assertions["only_first_primary_pool_inode_verified"]
            or not assertions["only_first_primary_pool_read_started"]
            or not assertions["only_first_primary_semantic_opened"]
            or not assertions["only_first_primary_semantic_inode_verified"]
            or not assertions["only_first_primary_semantic_read_started"]
            or first is None
            or value.get("status") != "DIAGNOSTIC_RECORDED"
            or value.get("engineering_failure_stage") is not None
            or value.get("engineering_check_code") is not None
            or value.get("check_code") not in expected_phase_by_code
            or value.get("diagnostic_failure_phase")
            != expected_phase_by_code.get(value.get("check_code"))
        ):
            raise ValueError("Candidate8 successful ledger inventory mismatch")
    elif (
        value.get("check_code") is not None
        or value.get("diagnostic_failure_phase") is not None
        or value.get("status") != "TERMINAL_ENGINEERING_FAILURE"
    ):
        raise ValueError("Candidate8 terminal ledger contract mismatch")
    else:
        stage = value.get("engineering_failure_stage")
        code = value.get("engineering_check_code")
        if (
            stage not in ENGINEERING_FAILURE_CODES
            or code not in ENGINEERING_FAILURE_CODES[stage]
        ):
            raise ValueError("Candidate8 terminal engineering code is outside allowlist")
        if assertions["exact_read_inventory_validated"] and (
            not assertions["diagnostic_call_returned"]
            or not assertions["consumption_anchor_sealed"]
            or assertions["denied_read_attempted"]
            or not assertions["manifest_opened"]
            or not assertions["manifest_inode_verified"]
            or not assertions["manifest_read_started"]
            or not assertions["only_first_primary_pool_opened"]
            or not assertions["only_first_primary_pool_inode_verified"]
            or not assertions["only_first_primary_pool_read_started"]
            or not assertions["only_first_primary_semantic_opened"]
            or not assertions["only_first_primary_semantic_inode_verified"]
            or not assertions["only_first_primary_semantic_read_started"]
            or first is None
        ):
            raise ValueError("Candidate8 terminal exact-inventory declaration mismatch")


PUBLIC_COMMON_KEYS = frozenset(
    {
        "attempt_ordinal",
        "candidate_metric_computed",
        "check_code",
        "continuation_authorized",
        "diagnostic_completed",
        "diagnostic_failure_phase",
        "diagnostic_read_ledger_sha256",
        "diagnostic_candidate_selection_exercise_is_not_model_selection",
        "engineering_attempt_id",
        "engineering_check_code",
        "engineering_failure_stage",
        "evaluator_opened",
        "fit_performed",
        "final_or_model_rule_selected",
        "full28_opened",
        "holdout_opened",
        "kind",
        "labels_opened",
        "promotion_authorized",
        "retry_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "status",
    }
)


def _public_payload(
    *,
    kind: str,
    status: str,
    check_code: str | None,
    failure_phase: str | None,
    diagnostic_completed: bool,
    ledger_sha256: str,
    engineering_failure_stage: str | None = None,
    engineering_check_code: str | None = None,
) -> dict[str, Any]:
    value = {
        "attempt_ordinal": 8,
        "candidate_metric_computed": False,
        "check_code": check_code,
        "continuation_authorized": False,
        "diagnostic_completed": diagnostic_completed,
        "diagnostic_failure_phase": failure_phase,
        "diagnostic_candidate_selection_exercise_is_not_model_selection": True,
        "diagnostic_read_ledger_sha256": ledger_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": engineering_check_code,
        "engineering_failure_stage": engineering_failure_stage,
        "evaluator_opened": False,
        "fit_performed": False,
        "final_or_model_rule_selected": False,
        "full28_opened": False,
        "holdout_opened": False,
        "kind": kind,
        "labels_opened": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "status": status,
    }
    _assert_public_payload(value)
    return value


def _assert_public_payload(value: Mapping[str, Any]) -> None:
    if set(value) != PUBLIC_COMMON_KEYS:
        raise ValueError("Candidate8 public payload schema mismatch")
    if not isinstance(value.get("diagnostic_completed"), bool):
        raise ValueError("Candidate8 diagnostic_completed must be strict bool")
    completed = value.get("diagnostic_completed") is True
    code = value.get("check_code")
    phase = value.get("diagnostic_failure_phase")
    allowed_codes = {
        "C8_ALL_DIAGNOSTIC_PHASES_PASSED": None,
        "C8_SEMANTIC_VALIDATION_FAILED": "D01_SEMANTIC_VALIDATION",
        "C8_POOL_PARSE_SANITIZE_FAILED": "D02_POOL_PARSE_SANITIZE",
        "C8_OWNERSHIP_EXERCISE_FAILED": "D03_OWNERSHIP_EXERCISE",
        "C8_FIXED_RULE_EXERCISE_FAILED": "D04_FIXED_RULE_EXERCISE",
        "C8_WRAPPER_RESULT_VALIDATION_FAILED": "D05_WRAPPER_RESULT_VALIDATION",
    }
    if completed:
        if (
            code not in allowed_codes
            or phase != allowed_codes[code]
            or value.get("engineering_failure_stage") is not None
            or value.get("engineering_check_code") is not None
            or value.get("kind") != RESULT_KIND
            or value.get("status") != "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"
        ):
            raise ValueError("Candidate8 public diagnostic code/phase mismatch")
    elif code is not None or phase is not None:
        raise ValueError("Candidate8 incomplete terminal must not expose a diagnostic code")
    else:
        stage = value.get("engineering_failure_stage")
        engineering_code = value.get("engineering_check_code")
        if (
            stage not in ENGINEERING_FAILURE_CODES
            or engineering_code not in ENGINEERING_FAILURE_CODES[stage]
            or value.get("kind") != TERMINAL_KIND
            or value.get("status") != "TERMINAL_ENGINEERING_FAILURE"
        ):
            raise ValueError("Candidate8 public engineering code is outside allowlist")
    if (
        value.get("schema_version") != 1
        or value.get("attempt_ordinal") != 8
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("continuation_authorized") is not False
        or value.get("retry_authorized") is not False
        or value.get("promotion_authorized") is not False
        or value.get("diagnostic_candidate_selection_exercise_is_not_model_selection")
        is not True
        or value.get("scientific_hypothesis_performance_tested") is not False
        or any(
            value.get(key) is not False
            for key in (
                "candidate_metric_computed",
                "evaluator_opened",
                "final_or_model_rule_selected",
                "fit_performed",
                "full28_opened",
                "holdout_opened",
                "labels_opened",
            )
        )
    ):
        raise ValueError("Candidate8 public safety declaration mismatch")
    _require_sha(value.get("diagnostic_read_ledger_sha256"), name="diagnostic ledger SHA")


def _bounded_engineering_failure(error: BaseException) -> tuple[str, str]:
    if (
        isinstance(error, CheckFailure)
        and error.stage in ENGINEERING_FAILURE_CODES
        and error.code in ENGINEERING_FAILURE_CODES[error.stage]
    ):
        return error.stage, error.code
    if isinstance(error, AnchorCreationFailure):
        return "ANCHOR_CONSUMPTION", "ANCHOR_CREATION_FAILED_AFTER_CONSUMPTION"
    return "DIAGNOSTIC_EXECUTION", "UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE"


def build_diagnostic_payloads(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    *,
    after_authorization_consumed: Callable[[Path], None] | None = None,
) -> dict[str, bytes]:
    if not isinstance(authorization, AuthorizationSeal) or authorization._sentinel is not _SEAL_SENTINEL:
        raise PermissionError("Candidate8 diagnostic lacks verified external authority")
    _validate_authorization_seal(
        context, authorization, require_qualified_controls=False
    )
    revalidate_fresh_output(authorization.output_freshness)
    qualification = qualify_source_controls(
        context,
        reader=authorization.reader,
        prefix_has_c8_authority=True,
    )
    if qualification.control_binding_commitment_sha256 != authorization.control_binding_commitment_sha256:
        raise CheckFailure(stage="SOURCE_CONTROL_QUALIFICATION", code="QUALIFICATION_CONTROL_COMMITMENT_MISMATCH")
    _validate_authorization_seal(
        context, authorization, require_qualified_controls=True
    )
    revalidate_fresh_output(authorization.output_freshness)
    consumption: ConsumptionSeal | None = None
    diagnostic: dict[str, Any] | None = None
    diagnostic_call_returned = False
    exact_read_inventory_validated = False
    first: FirstPrimaryBinding | None = None
    def mark_diagnostic_call_returned() -> None:
        nonlocal diagnostic_call_returned
        diagnostic_call_returned = True

    try:
        consumption = consume_diagnostic_authorization(context, authorization)
        if after_authorization_consumed is not None:
            after_authorization_consumed(consumption.path)
        diagnostic, first = read_and_diagnose_first_primary(
            context,
            qualification,
            consumption,
            after_diagnostic_call_returned=mark_diagnostic_call_returned,
        )
        diagnostic_call_returned = True
        allowed_codes = {
            context.runtime.PASS_CODE,
            *(context.runtime.FAILURE_CODES[phase] for phase in context.runtime.PHASE_CONTRACT),
        }
        expected_phase_by_code = {
            context.runtime.PASS_CODE: None,
            **{
                context.runtime.FAILURE_CODES[phase]: phase
                for phase in context.runtime.PHASE_CONTRACT
            },
        }
        if (
            diagnostic.get("check_code") not in allowed_codes
            or diagnostic.get("failure_phase")
            != expected_phase_by_code.get(diagnostic.get("check_code"))
        ):
            raise CheckFailure(stage="DIAGNOSTIC_RESULT_VALIDATION", code="DIAGNOSTIC_RESULT_OUTSIDE_D01_D05_PASS")
        _validate_exact_read_inventory(context, qualification)
        exact_read_inventory_validated = True
        ledger = _ledger_payload(
            context,
            authorization,
            qualification,
            consumption,
            diagnostic_check_code=str(diagnostic["check_code"]),
            diagnostic_failure_phase=diagnostic["failure_phase"],
            engineering_failure_stage=None,
            engineering_check_code=None,
            first_primary_commitment_sha256=first.commitment_sha256,
            exact_read_inventory_validated=True,
            diagnostic_call_returned=True,
            diagnostic_result_accepted_for_publication=True,
        )
        ledger_bytes = canonical_bytes(ledger)
        result = _public_payload(
            kind=RESULT_KIND,
            status="DIAGNOSTIC_COMPLETE_NONAUTHORIZING",
            check_code=str(diagnostic["check_code"]),
            failure_phase=diagnostic["failure_phase"],
            diagnostic_completed=True,
            ledger_sha256=sha256_bytes(ledger_bytes),
        )
        return {
            "diagnostic_read_ledger.json": ledger_bytes,
            "diagnostic_result.json": canonical_bytes(result),
        }
    except BaseException as error:
        if consumption is None and not isinstance(error, AnchorCreationFailure):
            raise
        engineering_stage, engineering_code = _bounded_engineering_failure(error)
        ledger = _ledger_payload(
            context,
            authorization,
            qualification,
            consumption,
            diagnostic_check_code=None,
            diagnostic_failure_phase=None,
            engineering_failure_stage=engineering_stage,
            engineering_check_code=engineering_code,
            first_primary_commitment_sha256=None if first is None else first.commitment_sha256,
            exact_read_inventory_validated=exact_read_inventory_validated,
            diagnostic_call_returned=diagnostic_call_returned,
            diagnostic_result_accepted_for_publication=False,
        )
        ledger_bytes = canonical_bytes(ledger)
        terminal = _public_payload(
            kind=TERMINAL_KIND,
            status="TERMINAL_ENGINEERING_FAILURE",
            check_code=None,
            failure_phase=None,
            diagnostic_completed=False,
            ledger_sha256=sha256_bytes(ledger_bytes),
            engineering_failure_stage=engineering_stage,
            engineering_check_code=engineering_code,
        )
        return {
            "diagnostic_read_ledger.json": ledger_bytes,
            "diagnostic_terminal.json": canonical_bytes(terminal),
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
        raise RuntimeError("Candidate8 requires no-replace rename")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), flag)
    if result != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, "Candidate8 destination exists")
        raise OSError(number, os.strerror(number))


def _staging_prefix(output: Path) -> str:
    return f".{output.name}.staging-"


def _quarantine_prefix(output: Path) -> str:
    return f".{output.name}.quarantine-"


def _sibling_names(parent_fd: int, *, prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix)))


def _validate_output(output: Path) -> Path:
    candidate = _lexical_repo_path(output, role="Candidate8 diagnostic output")
    expected = _lexical_repo_path(OUTPUT_ROOT, role="frozen Candidate8 diagnostic output")
    if candidate != expected:
        raise ValueError("Candidate8 output namespace is frozen")
    parent_fd, _identity_value = _open_stable_directory(candidate.parent, role="Candidate8 output parent")
    try:
        if _sibling_names(parent_fd, prefix=_staging_prefix(candidate)) or _sibling_names(parent_fd, prefix=_quarantine_prefix(candidate)):
            raise FileExistsError("Candidate8 rejects publication residue")
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate8 output must be fresh")
    finally:
        os.close(parent_fd)
    return candidate


def prevalidate_fresh_output(output: Path = OUTPUT_ROOT) -> FreshOutputSeal:
    candidate = _validate_output(output)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate8 output parent freshness seal"
    )
    os.close(parent_fd)
    return FreshOutputSeal(
        path=candidate,
        parent_identity=parent_identity,
        _sentinel=_OUTPUT_SEAL_SENTINEL,
    )


def revalidate_fresh_output(seal: FreshOutputSeal) -> None:
    if not isinstance(seal, FreshOutputSeal) or seal._sentinel is not _OUTPUT_SEAL_SENTINEL:
        raise PermissionError("Candidate8 output freshness seal is invalid")
    candidate = _validate_output(seal.path)
    if candidate != _lexical_repo_path(OUTPUT_ROOT, role="frozen Candidate8 output"):
        raise ValueError("Candidate8 output freshness namespace changed")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate8 output parent freshness revalidation"
    )
    os.close(parent_fd)
    if parent_identity != seal.parent_identity:
        raise RuntimeError("Candidate8 output parent changed after freshness seal")


def _open_owned_directory(parent_fd: int, name: str, expected_identity: tuple[int, int]) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate8 requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(name, os.O_RDONLY | nofollow | directory, dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected_identity:
        os.close(descriptor)
        raise RuntimeError("Candidate8 owned directory identity changed")
    return descriptor


def _write_payload(staging_fd: int, name: str, data: bytes) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate8 requires O_NOFOLLOW")
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600, dir_fd=staging_fd)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate8 staged write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or int(metadata.st_size) != len(data):
            raise RuntimeError("Candidate8 staged payload mismatch")
    finally:
        os.close(descriptor)


def _verify_payloads(parent_fd: int, name: str, expected_identity: tuple[int, int], payloads: Mapping[str, bytes]) -> None:
    directory_fd = _open_owned_directory(parent_fd, name, expected_identity)
    try:
        if set(os.listdir(directory_fd)) != set(payloads):
            raise RuntimeError("Candidate8 publication inventory changed")
        identities: set[tuple[int, int]] = set()
        for filename, expected in sorted(payloads.items()):
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is None:
                raise RuntimeError("Candidate8 requires O_NOFOLLOW")
            descriptor = os.open(filename, os.O_RDONLY | nofollow, dir_fd=directory_fd)
            try:
                before = os.fstat(descriptor)
                identity = _identity(before)
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                after = os.fstat(descriptor)
                visible = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or not stat.S_ISREG(after.st_mode)
                    or after.st_nlink != 1
                    or not stat.S_ISREG(visible.st_mode)
                    or visible.st_nlink != 1
                    or identity in identities
                    or _identity(after) != identity
                    or _identity(visible) != identity
                    or int(after.st_size) != len(data)
                    or data != expected
                ):
                    raise RuntimeError("Candidate8 publication payload changed")
                identities.add(identity)
            finally:
                os.close(descriptor)
    finally:
        os.close(directory_fd)


def _quarantine_owned(parent_fd: int, source_name: str, output: Path, expected_identity: tuple[int, int]) -> str:
    before = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(before) != expected_identity:
        raise RuntimeError("Candidate8 refuses to quarantine an unowned inode")
    destination = f"{_quarantine_prefix(output)}{secrets.token_hex(16)}"
    _renameat_noreplace(parent_fd, source_name, destination)
    after = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(after) != expected_identity:
        raise RuntimeError("Candidate8 quarantine identity mismatch")
    os.fsync(parent_fd)
    return destination


def publish_atomically(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    output_freshness: FreshOutputSeal,
    after_stage_write: Callable[[Path], None] | None = None,
    after_rename: Callable[[Path], None] | None = None,
) -> None:
    names = set(payloads)
    if names not in (PASS_FILENAMES, FAILURE_FILENAMES):
        raise ValueError("Candidate8 publication inventory mismatch")
    if any(Path(name).name != name or not name.endswith(".json") for name in names):
        raise ValueError("Candidate8 publication filenames must be flat JSON")
    if any(not isinstance(data, bytes) for data in payloads.values()):
        raise ValueError("Candidate8 publication payloads must be bytes")
    revalidate_fresh_output(output_freshness)
    output = _validate_output(output)
    if output != output_freshness.path:
        raise ValueError("Candidate8 publication output freshness path mismatch")
    parent_fd, parent_identity = _open_stable_directory(output.parent, role="Candidate8 output parent")
    if parent_identity != output_freshness.parent_identity:
        os.close(parent_fd)
        raise RuntimeError("Candidate8 publication output parent changed")
    staging_name = ""
    staging_identity: tuple[int, int] | None = None
    renamed = False
    try:
        _assert_visible_directory_identity(output.parent, parent_identity, role="Candidate8 visible output parent")
        if _sibling_names(parent_fd, prefix=_staging_prefix(output)) or _sibling_names(parent_fd, prefix=_quarantine_prefix(output)):
            raise FileExistsError("Candidate8 publication residue exists")
        staging_name = f"{_staging_prefix(output)}{secrets.token_hex(16)}"
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_identity = _identity(os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False))
        staging_fd = _open_owned_directory(parent_fd, staging_name, staging_identity)
        try:
            for name, data in sorted(payloads.items()):
                _write_payload(staging_fd, name, data)
            os.fsync(staging_fd)
            os.fchmod(staging_fd, 0o500)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        if after_stage_write is not None:
            after_stage_write(output.parent / staging_name)
        _verify_payloads(parent_fd, staging_name, staging_identity, payloads)
        if _identity(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError("Candidate8 pinned output parent changed")
        _assert_visible_directory_identity(output.parent, parent_identity, role="Candidate8 visible output parent before rename")
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate8 output raced into existence")
        _renameat_noreplace(parent_fd, staging_name, output.name)
        renamed = True
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate8 published directory identity mismatch")
        _verify_payloads(parent_fd, output.name, staging_identity, payloads)
        if after_rename is not None:
            after_rename(output)
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate8 published directory changed")
        _assert_visible_directory_identity(output.parent, parent_identity, role="Candidate8 visible output parent after rename")
        _verify_payloads(parent_fd, output.name, staging_identity, payloads)
        os.fsync(parent_fd)
    except BaseException:
        if staging_identity is not None:
            owned_name = output.name if renamed else staging_name
            try:
                metadata = os.stat(owned_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                metadata = None
            if metadata is not None and _identity(metadata) == staging_identity:
                _quarantine_owned(parent_fd, owned_name, output, staging_identity)
        raise
    finally:
        os.close(parent_fd)


def run_one_shot(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_freshness = prevalidate_fresh_output(output_root)
    output = output_freshness.path
    reader = GuardedReader(entries=[dict(entry) for entry in context.bootstrap_entries])
    authorization = authorize_diagnostic_once(
        context,
        reader=reader,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        output_freshness=output_freshness,
    )
    payloads = build_diagnostic_payloads(context, authorization)
    publish_atomically(
        output, payloads, output_freshness=output_freshness
    )
    public_name = "diagnostic_result.json" if "diagnostic_result.json" in payloads else "diagnostic_terminal.json"
    public = _decode_object(payloads[public_name], name="Candidate8 public result")
    ledger = _decode_object(
        payloads["diagnostic_read_ledger.json"], name="Candidate8 public ledger"
    )
    return {
        "check_code": public["check_code"],
        "consumption_anchor_sha256": ledger["consumption_anchor_sha256"],
        "diagnostic_failure_phase": public["diagnostic_failure_phase"],
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "output_artifact_sha256": {
            name: sha256_bytes(data) for name, data in sorted(payloads.items())
        },
        "output_inventory": sorted(payloads),
        "replay_or_later_pair_opened": False,
        "status": public["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--runner-size-bytes", required=True, type=int)
    parser.add_argument("--runner-tests-sha256", required=True)
    parser.add_argument("--runner-tests-size-bytes", required=True, type=int)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--runtime-tests-sha256", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--adapter-tests-sha256", required=True)
    parser.add_argument("--hypothesis-sha256", required=True)
    parser.add_argument("--candidate7-runtime-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--source-control-check-only", action="store_true")
    modes.add_argument("--diagnostic-once", action="store_true")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--authorization-size-bytes", type=int)
    args = parser.parse_args()
    if args.diagnostic_once and (
        args.authorization_sha256 is None or args.authorization_size_bytes is None
    ):
        parser.error("--diagnostic-once requires exact authorization SHA and size")
    if not args.diagnostic_once and (
        args.authorization_sha256 is not None or args.authorization_size_bytes is not None
    ):
        parser.error("authorization binding is valid only with --diagnostic-once")
    exit_code = 0
    try:
        context = bootstrap_verified_sources(
            runner_sha256=args.runner_sha256,
            runner_size_bytes=args.runner_size_bytes,
            runner_tests_sha256=args.runner_tests_sha256,
            runner_tests_size_bytes=args.runner_tests_size_bytes,
            runtime_sha256=args.runtime_sha256,
            runtime_tests_sha256=args.runtime_tests_sha256,
            adapter_sha256=args.adapter_sha256,
            adapter_tests_sha256=args.adapter_tests_sha256,
            hypothesis_sha256=args.hypothesis_sha256,
            candidate7_runtime_sha256=args.candidate7_runtime_sha256,
        )
        if args.source_check_only:
            value = {
                "artifact_or_control_input_opened": False,
                "holdout_or_label_opened": False,
                "source_inventory_complete": True,
                "status": "SOURCE_BYTES_VERIFIED_ONLY",
            }
        elif args.source_control_check_only:
            value = qualification_result(context)
            if value.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE":
                exit_code = 1
        else:
            value = run_one_shot(
                context,
                authorization_sha256=str(args.authorization_sha256),
                authorization_size_bytes=int(args.authorization_size_bytes),
            )
            if value.get("status") != "DIAGNOSTIC_COMPLETE_NONAUTHORIZING":
                exit_code = 1
    except BaseException:
        value = {
            "check_code": "C8_RUNNER_REJECTED_OR_FAILED",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "status": "FAIL_CLOSED",
        }
        print(json.dumps(value, indent=2, sort_keys=True))
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
