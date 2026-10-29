#!/usr/bin/env python3
"""Candidate7 repeatable control qualification and one-shot source preflight.

The source/control-only mode is explicitly repeatable and cannot open the cache
manifest, semantic artifacts, pools, labels, evaluator outputs, or holdout.
Only a separately authorized invocation without a qualification-only flag may
open the exact attested TRAIN18 manifest and its 36 semantic/pool pairs.
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
PREFLIGHT_SOURCE = Path(__file__).resolve()
RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py"
RUNTIME_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate7_conservative_recall_recovery_v1.py"
)
PREFLIGHT_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate7_semantic_pool_preflight_v1.py"
)
HYPOTHESIS_SOURCE = (
    ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
)
RUNTIME_MODULE = "backend.f1_train18_candidate7_conservative_recall_recovery_v1"
HYPOTHESIS_MODULE = "backend.f1_train18_candidate6_conservative_recall_recovery_v1"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
CANDIDATE6_DISPOSITION = (
    ROOT / "research/f1_aug30_train18_candidate6_terminal_disposition_v1.json"
)
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
OUTPUT_ROOT = (
    ROOT / "output/research/f1-aug30-train18-candidate7-semantic-pool-preflight-v1"
)
SOURCE_CONTROL_QUALIFICATION_RECORD = (
    ROOT / "research/f1_aug30_train18_candidate7_source_control_qualification_v1.json"
)
SOURCE_INVENTORY_AUTHORIZATION = (
    ROOT / "research/f1_aug30_train18_candidate7_source_inventory_authorization_v1.json"
)
SOURCE_INVENTORY_CONSUMPTION_ANCHOR = (
    ROOT / "research/f1_aug30_train18_candidate7_source_inventory_consumption_v1.json"
)

FIXED_CONTROLS = {
    "SEALED_SPLIT": (
        SPLIT,
        "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185",
        10706,
    ),
    "ORIGINAL_PREREGISTRATION": (
        ORIGINAL_PREREG,
        "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a",
        2651,
    ),
    "TRAIN_ALLOWLIST": (
        ALLOWLIST,
        "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0",
        2428,
    ),
    "CANDIDATE6_TERMINAL_DISPOSITION": (
        CANDIDATE6_DISPOSITION,
        "5849807fa82ab0755ab20b116774dccca68890ffa0cebb77e20a36060af88faf",
        2830,
    ),
    "TRAIN_CACHE_ATTESTATION": (
        CACHE_ATTESTATION,
        "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71",
        1834,
    ),
}
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
CUSTODY_LABEL_ATTESTATION_SHA256_NOT_OPENED = (
    "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
)
V51_SCAN_ONLY_MANIFEST_SHA256 = (
    "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"
)
CACHE_MANIFEST_SHA256 = "396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022"
CACHE_MANIFEST_SIZE_BYTES = 19601

REPORT_KIND = "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_preflight_report_v1"
LOCK_KIND = "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_preflight_lock_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate7_semantic_pool_preflight_read_ledger_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate7_preflight_terminal_v1"
CACHE_ATTESTATION_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"
CUSTODY_VALIDATION_KIND = "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"
QUALIFICATION_RECORD_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_source_control_qualification_v1"
)
SOURCE_INVENTORY_AUTHORIZATION_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_source_inventory_authorization_v1"
)
SOURCE_INVENTORY_CONSUMPTION_KIND = (
    "sherman_qc_f1_aug30_train18_candidate7_source_inventory_consumption_v1"
)

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
GENERIC_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})
TYPED_BINDING_KEYS = frozenset({"kind", "path", "sha256", "size_bytes"})

FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    RUNTIME_MODULE,
    HYPOTHESIS_MODULE,
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
    {"FULL28", "GOLD", "HOLDOUT", "LABEL", "PROJECTION", "TRUTH"}
)
FORBIDDEN_AGGREGATE_KEYS = frozenset(
    {
        "bend_id",
        "candidate_count",
        "feature_row_count",
        "feature_value",
        "line_object_count",
        "missing_numeric_feature_counts",
        "prediction_count",
        "prediction_row_count",
        "predictions",
        "rule_path_check_count",
        "selected_candidate_geometry_digest",
        "selector_score",
        "source_kind",
    }
)
PASS_FILENAMES = frozenset(
    {"preflight_lock.json", "preflight_read_ledger.json", "preflight_report.json"}
)
FAILURE_FILENAMES = frozenset(
    {"preflight_read_ledger.json", "preflight_terminal.json"}
)
_HEX = frozenset("0123456789abcdef")


class CheckFailure(RuntimeError):
    def __init__(self, *, stage: str, code: str) -> None:
        super().__init__(f"{stage}:{code}")
        self.stage = stage
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if text == "0" * 64:
        raise ValueError(f"{name} must not be unresolved")
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


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _decode_object(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
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
        raise ValueError(f"{role} path leaves repository root")
    return candidate


def _reject_symlink_chain(
    path: Path, *, role: str, require_exists: bool = True
) -> Path:
    candidate = _lexical_repo_path(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
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


def _assert_visible_directory_identity(
    path: Path, expected: tuple[int, int], *, role: str
) -> None:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    metadata = os.lstat(candidate)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected:
        raise RuntimeError(f"{role} identity changed")


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{role} is not a directory")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate7 requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory)
    opened = os.fstat(descriptor)
    expected = _identity(before)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected:
        os.close(descriptor)
        raise RuntimeError(f"{role} changed during open")
    _assert_visible_directory_identity(candidate, expected, role=f"{role} visible")
    return descriptor, expected


def _read_regular_file_once(path: Path, *, role: str) -> tuple[bytes, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{role} must be one non-hard-linked regular file")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        _assert_visible_directory_identity(
            candidate.parent, parent_identity, role=f"{role} parent before leaf"
        )
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or identity != _identity(before)
        ):
            raise ValueError(f"{role} changed during leaf open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible_leaf = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _identity(after) != identity
            or int(after.st_size) != len(data)
            or _identity(visible_leaf) != identity
        ):
            raise ValueError(f"{role} changed during read")
        _assert_visible_directory_identity(
            candidate.parent, parent_identity, role=f"{role} parent after read"
        )
        return data, identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact_source(
    path: Path, *, role: str, sha256: str
) -> tuple[bytes, dict[str, Any]]:
    data, identity = _read_regular_file_once(path, role=role)
    if sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA"):
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, {
        "case_id": None,
        "device_id": identity[0],
        "inode": identity[1],
        "path": str(_lexical_repo_path(path, role=role).relative_to(ROOT)),
        "purpose": role,
        "sha256": sha256,
        "size_bytes": len(data),
    }


def _reject_preloaded_sources(paths: Sequence[Path]) -> None:
    current = sys.modules.get(__name__)
    controlled: dict[tuple[int, int], Path] = {}
    resolved: set[Path] = set()
    for path in paths:
        candidate = _reject_symlink_chain(path, role="Candidate7 controlled source")
        metadata = os.stat(candidate, follow_symlinks=True)
        controlled[_identity(metadata)] = candidate
        resolved.add(candidate.resolve(strict=True))
    for name in FORBIDDEN_PRELOADED_MODULES:
        if name in sys.modules:
            raise RuntimeError(f"Candidate7 rejects preloaded module: {name}")
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
        if target in resolved or identity in controlled:
            raise RuntimeError(f"Candidate7 rejects preloaded source alias: {name}")


def _exec_verified_module(name: str, path: Path, data: bytes) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError(f"Candidate7 verified module already loaded: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "backend"
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
    hypothesis: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_bytes: Mapping[str, bytes]
    bootstrap_entries: tuple[dict[str, Any], ...]


def bootstrap_verified_sources(
    *,
    runtime_sha256: str,
    preflight_sha256: str,
    runtime_tests_sha256: str,
    preflight_tests_sha256: str,
    hypothesis_sha256: str,
) -> BootstrapContext:
    paths = (
        RUNTIME_SOURCE,
        PREFLIGHT_SOURCE,
        RUNTIME_TESTS_SOURCE,
        PREFLIGHT_TESTS_SOURCE,
        HYPOTHESIS_SOURCE,
    )
    _reject_preloaded_sources(paths)
    specs = {
        "hypothesis": (HYPOTHESIS_SOURCE, hypothesis_sha256),
        "preflight": (PREFLIGHT_SOURCE, preflight_sha256),
        "preflight_tests": (PREFLIGHT_TESTS_SOURCE, preflight_tests_sha256),
        "runtime": (RUNTIME_SOURCE, runtime_sha256),
        "runtime_tests": (RUNTIME_TESTS_SOURCE, runtime_tests_sha256),
    }
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha = specs[name]
        data, entry = _read_exact_source(
            path, role=f"CANDIDATE7_SOURCE_{name.upper()}", sha256=sha
        )
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate7 source inputs contain a hard-link alias")
        identities.add(identity)
        entries.append(entry)
        source_bytes[name] = data
        bindings[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha,
            "size_bytes": len(data),
        }
    hypothesis = _exec_verified_module(
        "_candidate7_verified_hypothesis", HYPOTHESIS_SOURCE, source_bytes["hypothesis"]
    )
    runtime = _exec_verified_module(
        "_candidate7_verified_runtime", RUNTIME_SOURCE, source_bytes["runtime"]
    )
    runtime.verify_scientific_hypothesis_runtime(
        hypothesis, source_sha256=hypothesis_sha256
    )
    if runtime.ATTEMPT_ORDINAL != 7 or runtime.ENGINEERING_ATTEMPT_ID != (
        "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
    ):
        raise ValueError("Candidate7 verified runtime identity mismatch")
    return BootstrapContext(
        runtime=runtime,
        hypothesis=hypothesis,
        source_bindings=bindings,
        source_bytes=source_bytes,
        bootstrap_entries=tuple(entries),
    )


@dataclass
class GuardedReader:
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)
    allowed: dict[Path, dict[str, Any]] = field(default_factory=dict)
    identities: dict[tuple[int, int], Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for entry in self.entries:
            identity = int(entry["device_id"]), int(entry["inode"])
            path = _lexical_repo_path(
                ROOT / str(entry["path"]), role=str(entry["purpose"])
            )
            prior = self.identities.get(identity)
            if prior is not None and prior != path:
                raise ValueError("Candidate7 bootstrap ledger has a hard-link alias")
            self.identities[identity] = path

    def _deny(self, path: Path, purpose: str, reason: str) -> None:
        self.denied_attempts.append(
            {"path": str(path), "purpose": purpose, "reason": reason}
        )
        raise PermissionError(f"Candidate7 read denied ({reason}): {path}")

    def authorize(
        self,
        path: Path,
        *,
        purpose: str,
        sha256: str,
        size_bytes: int,
        case_id: str | None = None,
    ) -> None:
        candidate = _lexical_repo_path(path, role=purpose)
        upper = purpose.upper()
        if candidate.name in FORBIDDEN_INPUT_BASENAMES:
            self._deny(candidate, purpose, "FORBIDDEN_BASENAME")
        if any(token in upper for token in FORBIDDEN_PURPOSE_TOKENS):
            self._deny(candidate, purpose, "FORBIDDEN_PURPOSE")
        lowered_parts = tuple(part.lower() for part in candidate.parts)
        if any("holdout" in part for part in lowered_parts):
            self._deny(candidate, purpose, "HOLDOUT_PATH_FORBIDDEN")
        if any("full28" in part or "full-28" in part for part in lowered_parts):
            self._deny(candidate, purpose, "FULL28_PATH_FORBIDDEN")
        if any(
            token in part
            for part in lowered_parts
            for token in ("gold", "label", "projection", "truth")
        ):
            self._deny(candidate, purpose, "LABEL_OR_TRUTH_PATH_FORBIDDEN")
        canonical = _reject_symlink_chain(candidate, role=purpose, require_exists=True)
        metadata = os.lstat(canonical)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{purpose} must be one non-hard-linked regular file")
        identity = _identity(metadata)
        prior = self.identities.get(identity)
        if prior is not None and prior != canonical:
            raise ValueError("Candidate7 input contains a hard-link alias")
        binding = {
            "case_id": case_id,
            "identity": identity,
            "purpose": purpose,
            "sha256": _require_sha(sha256, name=f"{purpose} SHA"),
            "size_bytes": _strict_int(size_bytes, name=f"{purpose} size"),
        }
        if binding["size_bytes"] <= 0:
            raise ValueError(f"{purpose} size must be positive")
        prior_binding = self.allowed.get(canonical)
        if prior_binding is not None and prior_binding != binding:
            raise ValueError("Candidate7 conflicting authorization")
        self.allowed[canonical] = binding
        self.identities[identity] = canonical

    def read_bytes(self, path: Path, *, purpose: str) -> bytes:
        candidate = _lexical_repo_path(path, role=purpose)
        binding = self.allowed.get(candidate)
        if binding is None or binding["purpose"] != purpose:
            self._deny(candidate, purpose, "NOT_EXACTLY_AUTHORIZED")
        data, identity = _read_regular_file_once(candidate, role=purpose)
        if (
            identity != binding["identity"]
            or len(data) != binding["size_bytes"]
            or sha256_bytes(data) != binding["sha256"]
        ):
            raise ValueError(f"Candidate7 {purpose} exact byte binding mismatch")
        self.entries.append(
            {
                "case_id": binding["case_id"],
                "device_id": identity[0],
                "inode": identity[1],
                "path": str(candidate.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": binding["sha256"],
                "size_bytes": len(data),
            }
        )
        return data


def _read_control(
    reader: GuardedReader,
    *,
    purpose: str,
    path: Path,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    reader.authorize(
        path,
        purpose=purpose,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    return _decode_object(reader.read_bytes(path, purpose=purpose), name=purpose)


def _binding_from_attestation(
    value: Any,
    *,
    role: str,
    expected_path: Path,
    typed: bool,
    expected_kind: str | None = None,
    failure_stage: str = "SOURCE_CONTROL_QUALIFICATION",
) -> tuple[Path, str, int]:
    expected_keys = TYPED_BINDING_KEYS if typed else GENERIC_BINDING_KEYS
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CheckFailure(
            stage=failure_stage,
            code=(
                "TYPED_BINDING_ENVELOPE_SCHEMA_MISMATCH"
                if typed
                else "GENERIC_BINDING_ENVELOPE_SCHEMA_MISMATCH"
            ),
        )
    if typed and (expected_kind is None or value["kind"] != expected_kind):
        raise CheckFailure(
            stage=failure_stage,
            code="TYPED_BINDING_KIND_MISMATCH",
        )
    relative = Path(str(value["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckFailure(
            stage=failure_stage,
            code="BINDING_PATH_NOT_REPOSITORY_RELATIVE",
        )
    path = _lexical_repo_path(ROOT / relative, role=role)
    if path != _lexical_repo_path(expected_path, role=f"expected {role}"):
        raise CheckFailure(
            stage=failure_stage,
            code="BINDING_PATH_MISMATCH",
        )
    return (
        path,
        _require_sha(value["sha256"], name=f"{role} SHA"),
        _strict_int(value["size_bytes"], name=f"{role} size"),
    )


def _validate_attestation_header(
    attestation: Mapping[str, Any], *, train_commitment: str
) -> dict[str, bool]:
    if set(attestation) != CACHE_ATTESTATION_KEYS:
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="CACHE_ATTESTATION_EXACT_KEY_SCHEMA_MISMATCH",
        )
    checks = {
        "allowlist_binding": attestation.get("train_allowlist_sha256")
        == FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "canonical_json": attestation.get("canonical_json") is True,
        "holdout_entries_absent": attestation.get("holdout_entry_count") == 0,
        "holdout_hashes_absent": attestation.get("holdout_hash_count") == 0,
        "holdout_status_absent": attestation.get("holdout_status_count") == 0,
        "independent_validation": attestation.get("independent_validation_passed")
        is True,
        "inventory": attestation.get("case_count") == 18
        and attestation.get("record_count") == 36,
        "kind": attestation.get("schema_version") == 1
        and attestation.get("kind") == CACHE_ATTESTATION_KIND,
        "optimizer_holdout_access_zero": attestation.get(
            "optimizer_holdout_access_count"
        )
        == 0,
        "split_binding": attestation.get("split_sha256")
        == FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_commitment": attestation.get("train_set_commitment_sha256")
        == train_commitment,
        "train_exact": attestation.get("train_exact_match") is True,
        "train_holdout_intersection_zero": attestation.get(
            "train_holdout_intersection_count"
        )
        == 0,
    }
    _require_sha(
        attestation.get("blinded_complement_commitment_sha256"),
        name="blinded complement commitment",
    )
    if not all(checks.values()):
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="CACHE_ATTESTATION_BOUNDARY_MISMATCH",
        )
    return checks


@dataclass(frozen=True)
class Qualification:
    reader: GuardedReader
    train_ids: tuple[str, ...]
    scan_by_case: Mapping[str, str]
    train_commitment: str
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    attestation_checks: Mapping[str, bool]


_AUTHORIZATION_SEAL_SENTINEL = object()


@dataclass(frozen=True)
class SourceInventoryAuthorizationSeal:
    reader: GuardedReader
    authorization_sha256: str
    qualification_record_sha256: str
    _sentinel: object


@dataclass(frozen=True)
class SourceInventoryConsumptionSeal:
    path: Path
    sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    _sentinel: object


def _expected_authorization_source_bindings(
    context: BootstrapContext,
) -> dict[str, dict[str, Any]]:
    value = {
        key: dict(binding) for key, binding in sorted(context.source_bindings.items())
    }
    value["candidate6_disposition"] = {
        "path": str(
            FIXED_CONTROLS["CANDIDATE6_TERMINAL_DISPOSITION"][0].relative_to(ROOT)
        ),
        "sha256": FIXED_CONTROLS["CANDIDATE6_TERMINAL_DISPOSITION"][1],
        "size_bytes": FIXED_CONTROLS["CANDIDATE6_TERMINAL_DISPOSITION"][2],
    }
    return value


def authorize_source_inventory_once(
    context: BootstrapContext,
    *,
    reader: GuardedReader,
    authorization_sha256: str,
) -> SourceInventoryAuthorizationSeal:
    """Validate external authority and its exact prior qualification record."""

    authorization_path = _lexical_repo_path(
        SOURCE_INVENTORY_AUTHORIZATION,
        role="Candidate7 source inventory authorization",
    )
    authorization_size = os.lstat(authorization_path).st_size
    reader.authorize(
        authorization_path,
        purpose="CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        sha256=_require_sha(
            authorization_sha256, name="source inventory authorization SHA"
        ),
        size_bytes=authorization_size,
    )
    authorization_bytes = reader.read_bytes(
        authorization_path,
        purpose="CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
    )
    authorization = _decode_object(
        authorization_bytes, name="CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION"
    )
    expected_keys = {
        "attempt_ordinal",
        "authorization_consumed_on_first_manifest_open",
        "engineering_attempt_id",
        "holdout_execution_authorized",
        "kind",
        "label_or_metric_access_authorized",
        "output_namespace",
        "qualification_record_binding",
        "retry_after_manifest_open_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "source_bindings",
        "status",
    }
    if (
        set(authorization) != expected_keys
        or authorization.get("schema_version") != 1
        or authorization.get("kind") != SOURCE_INVENTORY_AUTHORIZATION_KIND
        or authorization.get("status") != "AUTHORIZED_ONCE"
        or authorization.get("attempt_ordinal") != 7
        or authorization.get("engineering_attempt_id")
        != "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
        or authorization.get("scientific_hypothesis_id")
        != "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
        or authorization.get("authorization_consumed_on_first_manifest_open") is not True
        or authorization.get("retry_after_manifest_open_authorized") is not False
        or authorization.get("label_or_metric_access_authorized") is not False
        or authorization.get("holdout_execution_authorized") is not False
        or authorization.get("output_namespace")
        != str(OUTPUT_ROOT.relative_to(ROOT))
        or authorization.get("source_bindings")
        != _expected_authorization_source_bindings(context)
    ):
        raise CheckFailure(
            stage="SOURCE_INVENTORY_AUTHORIZATION",
            code="AUTHORIZATION_CONTRACT_MISMATCH",
        )
    qualification_binding = authorization.get("qualification_record_binding")
    qualification_path, qualification_sha, qualification_size = (
        _binding_from_attestation(
            qualification_binding,
            role="CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
            expected_path=SOURCE_CONTROL_QUALIFICATION_RECORD,
            typed=True,
            expected_kind=QUALIFICATION_RECORD_KIND,
            failure_stage="SOURCE_INVENTORY_AUTHORIZATION",
        )
    )
    reader.authorize(
        qualification_path,
        purpose="CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
        sha256=qualification_sha,
        size_bytes=qualification_size,
    )
    qualification_record = _decode_object(
        reader.read_bytes(
            qualification_path,
            purpose="CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
        ),
        name="CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
    )
    qualification_keys = {
        "artifact_source_inventory_opened",
        "attempt_ordinal",
        "cache_manifest_opened",
        "check_code",
        "engineering_attempt_id",
        "holdout_or_label_opened",
        "kind",
        "read_count",
        "repeatable",
        "schema_version",
        "scientific_hypothesis_consumed",
        "scientific_hypothesis_id",
        "source_bindings",
        "source_control_qualification_complete",
        "status",
    }
    if (
        set(qualification_record) != qualification_keys
        or qualification_record.get("schema_version") != 1
        or qualification_record.get("kind") != QUALIFICATION_RECORD_KIND
        or qualification_record.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or qualification_record.get("check_code")
        != "ALL_SOURCE_CONTROL_CHECKS_PASSED"
        or qualification_record.get("attempt_ordinal") != 7
        or qualification_record.get("engineering_attempt_id")
        != "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
        or qualification_record.get("scientific_hypothesis_id")
        != "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
        or qualification_record.get("read_count") != 14
        or qualification_record.get("repeatable") is not True
        or qualification_record.get("source_control_qualification_complete") is not True
        or qualification_record.get("artifact_source_inventory_opened") is not False
        or qualification_record.get("cache_manifest_opened") is not False
        or qualification_record.get("holdout_or_label_opened") is not False
        or qualification_record.get("scientific_hypothesis_consumed") is not False
        or qualification_record.get("source_bindings")
        != _expected_authorization_source_bindings(context)
    ):
        raise CheckFailure(
            stage="SOURCE_INVENTORY_AUTHORIZATION",
            code="QUALIFICATION_RECORD_CONTRACT_MISMATCH",
        )
    return SourceInventoryAuthorizationSeal(
        reader=reader,
        authorization_sha256=authorization_sha256,
        qualification_record_sha256=qualification_sha,
        _sentinel=_AUTHORIZATION_SEAL_SENTINEL,
    )


def consume_source_inventory_authorization(
    context: BootstrapContext,
    authorization: SourceInventoryAuthorizationSeal,
) -> SourceInventoryConsumptionSeal:
    """Durably consume one-shot authority before the first manifest open."""

    if (
        not isinstance(authorization, SourceInventoryAuthorizationSeal)
        or authorization._sentinel is not _AUTHORIZATION_SEAL_SENTINEL
    ):
        raise PermissionError("Candidate7 consumption requires a verified authority seal")
    reader = authorization.reader
    expected_purposes = [
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
    ]
    if [str(entry["purpose"]) for entry in reader.entries] != expected_purposes:
        raise PermissionError("Candidate7 consumption requires qualified controls")
    anchor = _lexical_repo_path(
        SOURCE_INVENTORY_CONSUMPTION_ANCHOR,
        role="Candidate7 source inventory consumption anchor",
    )
    expected_anchor = _lexical_repo_path(
        ROOT / "research/f1_aug30_train18_candidate7_source_inventory_consumption_v1.json",
        role="frozen Candidate7 consumption anchor",
    )
    if anchor != expected_anchor:
        raise ValueError("Candidate7 consumption anchor namespace is frozen")
    authorization_entry = reader.entries[5]
    qualification_entry = reader.entries[6]
    value = {
        "attempt_ordinal": 7,
        "authorization_binding": {
            "path": authorization_entry["path"],
            "sha256": authorization.authorization_sha256,
            "size_bytes": authorization_entry["size_bytes"],
        },
        "consumed_before_manifest_open": True,
        "engineering_attempt_id": "CANDIDATE7_ENGINEERING_ATTEMPT_V1",
        "holdout_or_label_authorized": False,
        "kind": SOURCE_INVENTORY_CONSUMPTION_KIND,
        "manifest_opened_at_anchor_creation": False,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "qualification_record_binding": {
            "path": qualification_entry["path"],
            "sha256": authorization.qualification_record_sha256,
            "size_bytes": qualification_entry["size_bytes"],
        },
        "retry_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
        "source_bindings": _expected_authorization_source_bindings(context),
        "status": "CONSUMED_BEFORE_MANIFEST_OPEN",
    }
    data = canonical_bytes(value)
    parent_fd, parent_identity = _open_stable_directory(
        anchor.parent, role="Candidate7 consumption anchor parent"
    )
    descriptor: int | None = None
    file_identity: tuple[int, int] | None = None
    try:
        _assert_visible_directory_identity(
            anchor.parent,
            parent_identity,
            role="Candidate7 visible consumption anchor parent",
        )
        descriptor = os.open(
            anchor.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        # Persist the no-replace directory entry before any subsequent work.
        # Even an empty/partial anchor permanently consumes this authorization.
        os.fsync(parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate7 consumption anchor write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        file_identity = _identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != len(data)
        ):
            raise RuntimeError("Candidate7 consumption anchor write mismatch")
        os.fchmod(descriptor, 0o400)
        visible = os.stat(anchor.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(visible) != file_identity:
            raise RuntimeError("Candidate7 consumption anchor identity changed")
        os.fsync(parent_fd)
        _assert_visible_directory_identity(
            anchor.parent,
            parent_identity,
            role="Candidate7 visible consumption anchor parent after write",
        )
        canonical_anchor = _reject_symlink_chain(
            anchor,
            role="Candidate7 visible consumption anchor after write",
            require_exists=True,
        )
        canonical_metadata = os.lstat(canonical_anchor)
        if (
            not stat.S_ISREG(canonical_metadata.st_mode)
            or canonical_metadata.st_nlink != 1
            or _identity(canonical_metadata) != file_identity
            or int(canonical_metadata.st_size) != len(data)
        ):
            raise RuntimeError("Candidate7 canonical consumption anchor changed")
    except FileExistsError as error:
        raise PermissionError(
            "Candidate7 source inventory authorization is already consumed"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if file_identity is None:
        raise RuntimeError("Candidate7 consumption anchor lacks a captured identity")
    seal = SourceInventoryConsumptionSeal(
        path=anchor,
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        file_identity=file_identity,
        parent_identity=parent_identity,
        _sentinel=_AUTHORIZATION_SEAL_SENTINEL,
    )
    revalidate_consumption_anchor(seal)
    return seal


def revalidate_consumption_anchor(seal: SourceInventoryConsumptionSeal) -> None:
    if (
        not isinstance(seal, SourceInventoryConsumptionSeal)
        or seal._sentinel is not _AUTHORIZATION_SEAL_SENTINEL
    ):
        raise PermissionError("Candidate7 consumption anchor seal is invalid")
    parent_fd, observed_parent_identity = _open_stable_directory(
        seal.path.parent, role="Candidate7 consumption anchor parent revalidation"
    )
    try:
        if observed_parent_identity != seal.parent_identity:
            raise RuntimeError("Candidate7 visible consumption parent identity changed")
        _assert_visible_directory_identity(
            seal.path.parent,
            seal.parent_identity,
            role="Candidate7 visible consumption parent immediately before manifest",
        )
        metadata = os.stat(
            seal.path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or _identity(metadata) != seal.file_identity
            or int(metadata.st_size) != seal.size_bytes
        ):
            raise RuntimeError("Candidate7 consumption anchor changed before manifest")
        canonical = _reject_symlink_chain(
            seal.path,
            role="Candidate7 canonical consumption anchor before manifest",
            require_exists=True,
        )
        visible = os.lstat(canonical)
        if _identity(visible) != seal.file_identity:
            raise RuntimeError("Candidate7 canonical consumption anchor identity changed")
    finally:
        os.close(parent_fd)


def qualify_source_controls(
    context: BootstrapContext, *, reader: GuardedReader | None = None
) -> Qualification:
    if reader is None:
        reader = GuardedReader(
            entries=[dict(entry) for entry in context.bootstrap_entries]
        )
    initial_purposes = [str(entry["purpose"]) for entry in reader.entries]
    controls = {
        purpose: _read_control(
            reader,
            purpose=purpose,
            path=path,
            sha256=sha,
            size_bytes=size,
        )
        for purpose, (path, sha, size) in FIXED_CONTROLS.items()
    }
    split = controls["SEALED_SPLIT"]
    original = controls["ORIGINAL_PREREGISTRATION"]
    allowlist = controls["TRAIN_ALLOWLIST"]
    try:
        context.runtime._validate_candidate6_disposition(
            canonical_bytes(controls["CANDIDATE6_TERMINAL_DISPOSITION"]),
            expected_sha256=FIXED_CONTROLS["CANDIDATE6_TERMINAL_DISPOSITION"][1],
        )
    except (TypeError, ValueError) as error:
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="CANDIDATE6_DISPOSITION_CONTRACT_MISMATCH",
        ) from error
    if (
        split.get("kind")
        != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
        or split.get("label_or_outcome_fields_present") is not False
        or original.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN"
    ):
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION", code="SPLIT_OR_PREREG_MISMATCH"
        )
    cases = allowlist.get("cases")
    if (
        allowlist.get("kind")
        != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
        or allowlist.get("reject_any_case_not_listed") is not True
        or not isinstance(cases, list)
        or len(cases) != 18
    ):
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION", code="TRAIN_ALLOWLIST_MISMATCH"
        )
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    if (
        len(set(train_ids)) != 18
        or any(not case_id for case_id in train_ids)
        or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids
    ):
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION", code="TRAIN_MEMBERSHIP_ORDER_MISMATCH"
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
    attestation_checks = _validate_attestation_header(
        attestation, train_commitment=train_commitment
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
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION", code="CUSTODY_PASS_SHA_MISMATCH"
        )
    manifest_path, manifest_sha, manifest_size = _binding_from_attestation(
        attestation.get("manifest_binding"),
        role="TRAIN_CACHE_MANIFEST",
        expected_path=CACHE_MANIFEST,
        typed=True,
        expected_kind=CACHE_MANIFEST_KIND,
    )
    if manifest_sha != CACHE_MANIFEST_SHA256 or manifest_size != CACHE_MANIFEST_SIZE_BYTES:
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="CACHE_MANIFEST_IMMUTABLE_BINDING_MISMATCH",
        )
    purposes = [str(entry["purpose"]) for entry in reader.entries]
    source_prefix = [
        "CANDIDATE7_SOURCE_HYPOTHESIS",
        "CANDIDATE7_SOURCE_PREFLIGHT",
        "CANDIDATE7_SOURCE_PREFLIGHT_TESTS",
        "CANDIDATE7_SOURCE_RUNTIME",
        "CANDIDATE7_SOURCE_RUNTIME_TESTS",
    ]
    observed_prefix = initial_purposes
    authorized_prefix = source_prefix + [
        "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
    ]
    if observed_prefix[:5] != source_prefix or observed_prefix not in (
        source_prefix,
        authorized_prefix,
    ):
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="SOURCE_CONTROL_PREFIX_INVENTORY_MISMATCH",
        )
    expected_purposes = observed_prefix + [
        "SEALED_SPLIT",
        "ORIGINAL_PREREGISTRATION",
        "TRAIN_ALLOWLIST",
        "CANDIDATE6_TERMINAL_DISPOSITION",
        "TRAIN_CACHE_ATTESTATION",
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
    ]
    if purposes != expected_purposes or reader.denied_attempts:
        raise CheckFailure(
            stage="SOURCE_CONTROL_QUALIFICATION",
            code="SOURCE_CONTROL_READ_INVENTORY_MISMATCH",
        )
    return Qualification(
        reader=reader,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=train_commitment,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest_size_bytes=manifest_size,
        attestation_checks=attestation_checks,
    )


def qualification_result(context: BootstrapContext) -> dict[str, Any]:
    reader = GuardedReader(entries=[dict(entry) for entry in context.bootstrap_entries])
    try:
        qualification = qualify_source_controls(context, reader=reader)
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
    return {
        "artifact_source_inventory_opened": False,
        "cache_manifest_opened": False,
        "check_code": "ALL_SOURCE_CONTROL_CHECKS_PASSED",
        "failure_stage": None,
        "holdout_or_label_opened": False,
        "repeatable": True,
        "scientific_hypothesis_consumed": False,
        "source_control_qualification_complete": True,
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        "verified_control_read_inventory_complete": len(qualification.reader.entries)
        == 14,
    }


def _inside_cache_root(path: Path) -> bool:
    candidate = _lexical_repo_path(path, role="TRAIN cache artifact")
    root = _lexical_repo_path(ALLOWED_CACHE_ROOT, role="TRAIN cache root")
    return candidate == root or root in candidate.parents


def _validate_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    train_ids: tuple[str, ...],
    scan_by_case: Mapping[str, str],
    train_commitment: str,
) -> dict[tuple[str, str], dict[str, Any]]:
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
        or manifest.get("train_allowlist_sha256")
        != FIXED_CONTROLS["TRAIN_ALLOWLIST"][1]
        or manifest.get("train_set_commitment_sha256") != train_commitment
        or manifest.get("v51_scan_only_manifest_sha256")
        != V51_SCAN_ONLY_MANIFEST_SHA256
    ):
        raise CheckFailure(
            stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MANIFEST_HEADER_MISMATCH"
        )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise CheckFailure(
            stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MANIFEST_RECORD_COUNT_MISMATCH"
        )
    expected_pairs = [
        (case_id, pass_name)
        for case_id in train_ids
        for pass_name in ("primary", "replay")
    ]
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
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    identities: dict[str, str] = {}
    reverse_identities: dict[str, str] = {}
    paths: set[Path] = set()
    for raw, pair in zip(records, expected_pairs):
        if not isinstance(raw, dict) or set(raw) != record_keys:
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="CACHE_RECORD_SCHEMA_MISMATCH"
            )
        if (str(raw["case_id"]), str(raw["pass"])) != pair:
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="CACHE_MEMBERSHIP_ORDER_MISMATCH"
            )
        case_id, pass_name = pair
        runtime_id = str(raw["runtime_case_id"])
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or identities.setdefault(case_id, runtime_id) != runtime_id
            or reverse_identities.setdefault(runtime_id, case_id) != case_id
        ):
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="RUNTIME_ID_BIJECTION_MISMATCH"
            )
        if _require_sha(raw["scan_sha256"], name="manifest scan SHA") != (
            scan_by_case[case_id]
        ):
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="MANIFEST_SCAN_BINDING_MISMATCH"
            )
        semantic_relative = Path(str(raw["semantic_path"]))
        pool_relative = Path(str(raw["pool_path"]))
        if (
            semantic_relative.is_absolute()
            or pool_relative.is_absolute()
            or ".." in semantic_relative.parts
            or ".." in pool_relative.parts
        ):
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="CACHE_PATH_NOT_REPOSITORY_RELATIVE"
            )
        semantic_path = _lexical_repo_path(
            ROOT / semantic_relative, role="TRAIN_V51_SEMANTIC"
        )
        pool_path = _lexical_repo_path(ROOT / pool_relative, role="TRAIN_V51_POOL")
        expected_semantic = _lexical_repo_path(
            ALLOWED_CACHE_ROOT / pass_name / runtime_id / "semantic_result.json",
            role="expected TRAIN semantic",
        )
        expected_pool = _lexical_repo_path(
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
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="CACHE_ARTIFACT_PATH_MISMATCH"
            )
        paths.update((semantic_path, pool_path))
        mapping[pair] = {
            "case_id": case_id,
            "pass": pass_name,
            "pool_path": pool_path,
            "pool_sha256": _require_sha(raw["pool_sha256"], name="pool SHA"),
            "runtime_case_id": runtime_id,
            "semantic_path": semantic_path,
            "semantic_sha256": _require_sha(
                raw["semantic_sha256"], name="semantic SHA"
            ),
        }
    for case_id in train_ids:
        primary = mapping[(case_id, "primary")]
        replay = mapping[(case_id, "replay")]
        if (
            primary["semantic_sha256"] != replay["semantic_sha256"]
            or primary["pool_sha256"] != replay["pool_sha256"]
        ):
            raise CheckFailure(
                stage="CACHE_MANIFEST_VALIDATION", code="PRIMARY_REPLAY_HASH_MISMATCH"
            )
    return mapping


def _assert_runtime_preflight_safe(result: Mapping[str, Any]) -> None:
    required_true = {
        "all_fixed_rule_forms_exercised",
        "candidate_source_rows_sanitized",
        "deterministic_geometry_source_tie_break_exercised",
        "fixed_allowed_source_kind_eligibility_exercised",
        "generic_pool_accounting_ignored",
        "missing_evidence_abstention_exercised",
        "ownership_geometry_exercised",
        "raw_candidates_ignored_for_selection",
        "recursive_forbidden_key_validation_exercised",
        "runtime_threshold_match_implies_conservative_conflict_for_bound_pairs",
        "scientific_hypothesis_exactly_preserved",
        "source_only",
        "v51_prediction_values_and_order_preserved",
    }
    if (
        result.get("schema_version") != 1
        or result.get("attempt_ordinal") != 7
        or result.get("engineering_attempt_id")
        != "CANDIDATE7_ENGINEERING_ATTEMPT_V1"
        or result.get("scientific_hypothesis_id")
        != "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
        or result.get("status") != "SOURCE_COMPATIBLE"
        or result.get("labels_used") is not False
        or result.get("candidate_metric_computed") is not False
        or result.get("candidate_rule_selected") is not False
        or result.get("model_fit_performed") is not False
        or result.get("holdout_score_opened") is not False
        or any(result.get(key) is not True for key in required_true)
        or FORBIDDEN_AGGREGATE_KEYS.intersection(result)
    ):
        raise CheckFailure(
            stage="TRAIN_SOURCE_COMPATIBILITY", code="RUNTIME_PREFLIGHT_RESULT_UNSAFE"
        )


def _read_train_sources(
    context: BootstrapContext,
    qualification: Qualification,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    reader = qualification.reader
    for case_id in qualification.train_ids:
        loaded: dict[str, tuple[bytes, bytes]] = {}
        for pass_name in ("primary", "replay"):
            record = records[(case_id, pass_name)]
            semantic_path = record["semantic_path"]
            pool_path = record["pool_path"]
            semantic_size = os.lstat(semantic_path).st_size
            pool_size = os.lstat(pool_path).st_size
            semantic_purpose = f"TRAIN_{pass_name.upper()}_V51_SEMANTIC"
            pool_purpose = f"TRAIN_{pass_name.upper()}_V51_POOL"
            reader.authorize(
                semantic_path,
                purpose=semantic_purpose,
                sha256=record["semantic_sha256"],
                size_bytes=semantic_size,
                case_id=case_id,
            )
            reader.authorize(
                pool_path,
                purpose=pool_purpose,
                sha256=record["pool_sha256"],
                size_bytes=pool_size,
                case_id=case_id,
            )
            semantic_bytes = reader.read_bytes(
                semantic_path, purpose=semantic_purpose
            )
            pool_bytes = reader.read_bytes(pool_path, purpose=pool_purpose)
            result = context.runtime.preflight_candidate7_source_bytes(
                semantic_bytes,
                pool_bytes,
                v51_semantic_sha256=record["semantic_sha256"],
                pool_sha256=record["pool_sha256"],
                hypothesis_runtime=context.hypothesis,
                hypothesis_source_sha256=context.source_bindings["hypothesis"]["sha256"],
            )
            _assert_runtime_preflight_safe(result)
            loaded[pass_name] = semantic_bytes, pool_bytes
        if loaded["primary"] != loaded["replay"]:
            raise CheckFailure(
                stage="TRAIN_SOURCE_COMPATIBILITY",
                code="PRIMARY_REPLAY_SOURCE_BYTES_MISMATCH",
            )


def _validate_success_inventory(qualification: Qualification) -> None:
    expected = [
        ("CANDIDATE7_SOURCE_HYPOTHESIS", None),
        ("CANDIDATE7_SOURCE_PREFLIGHT", None),
        ("CANDIDATE7_SOURCE_PREFLIGHT_TESTS", None),
        ("CANDIDATE7_SOURCE_RUNTIME", None),
        ("CANDIDATE7_SOURCE_RUNTIME_TESTS", None),
        ("CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION", None),
        ("CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD", None),
        ("SEALED_SPLIT", None),
        ("ORIGINAL_PREREGISTRATION", None),
        ("TRAIN_ALLOWLIST", None),
        ("CANDIDATE6_TERMINAL_DISPOSITION", None),
        ("TRAIN_CACHE_ATTESTATION", None),
        ("CACHE_CUSTODY_ACCESS_EVIDENCE", None),
        ("CACHE_CUSTODY_BUILDER", None),
        ("CACHE_CUSTODY_STAGED_VALIDATOR_PASS", None),
        ("CACHE_CUSTODY_VALIDATOR", None),
        ("TRAIN_CACHE_MANIFEST", None),
    ]
    for case_id in qualification.train_ids:
        for pass_name in ("PRIMARY", "REPLAY"):
            expected.extend(
                (
                    (f"TRAIN_{pass_name}_V51_SEMANTIC", case_id),
                    (f"TRAIN_{pass_name}_V51_POOL", case_id),
                )
            )
    observed = [
        (str(entry["purpose"]), entry["case_id"])
        for entry in qualification.reader.entries
    ]
    paths = [str(entry["path"]) for entry in qualification.reader.entries]
    identities = [
        (int(entry["device_id"]), int(entry["inode"]))
        for entry in qualification.reader.entries
    ]
    if (
        observed != expected
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or qualification.reader.denied_attempts
    ):
        raise CheckFailure(
            stage="TRAIN_SOURCE_COMPATIBILITY", code="EXACT_READ_INVENTORY_MISMATCH"
        )


def _assert_aggregate_safe(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_AGGREGATE_KEYS:
                raise ValueError(f"Candidate7 aggregate exposes forbidden key: {key}")
            _assert_aggregate_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_aggregate_safe(child)


def _ledger_payload(
    reader: GuardedReader,
    *,
    source_bindings: Mapping[str, Mapping[str, Any]],
    status: str,
    failure_stage: str | None,
    check_code: str,
    exact_train_inventory_completed: bool,
) -> dict[str, Any]:
    purposes = [str(entry["purpose"]) for entry in reader.entries]
    return {
        "assertions": {
            "all_reads_within_exact_authorized_inventory": True,
            "cache_manifest_opened": "TRAIN_CACHE_MANIFEST" in purposes,
            "candidate6_disposition_opened": (
                "CANDIDATE6_TERMINAL_DISPOSITION" in purposes
            ),
            "candidate6_terminal_or_ledger_opened": False,
            "exact_train_inventory_completed": exact_train_inventory_completed,
            "full_28_catalog_or_manifest_opened": False,
            "holdout_path_hash_status_or_score_opened": False,
            "label_attestation_or_projection_opened": False,
            "semantic_internal_scan_identity_used": False,
        },
        "attempt_ordinal": 7,
        "check_code": check_code,
        "denied_attempts": list(reader.denied_attempts),
        "engineering_attempt_id": "CANDIDATE7_ENGINEERING_ATTEMPT_V1",
        "failure_stage": failure_stage,
        "kind": LEDGER_KIND,
        "reads": list(reader.entries),
        "schema_version": 1,
        "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
        "source_bindings": {
            key: dict(binding) for key, binding in sorted(source_bindings.items())
        },
        "status": status,
    }


def build_preflight_payloads(
    context: BootstrapContext,
    *,
    authorization: SourceInventoryAuthorizationSeal,
    after_authorization_consumed: Callable[[Path], None] | None = None,
) -> dict[str, bytes]:
    stage = "SOURCE_CONTROL_QUALIFICATION"
    qualification: Qualification | None = None
    consumption: SourceInventoryConsumptionSeal | None = None
    if (
        not isinstance(authorization, SourceInventoryAuthorizationSeal)
        or authorization._sentinel is not _AUTHORIZATION_SEAL_SENTINEL
    ):
        raise PermissionError("Candidate7 source inventory lacks a verified authority seal")
    reader = authorization.reader
    required_prefix = [
        "CANDIDATE7_SOURCE_HYPOTHESIS",
        "CANDIDATE7_SOURCE_PREFLIGHT",
        "CANDIDATE7_SOURCE_PREFLIGHT_TESTS",
        "CANDIDATE7_SOURCE_RUNTIME",
        "CANDIDATE7_SOURCE_RUNTIME_TESTS",
        "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION",
        "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD",
    ]
    if [str(entry["purpose"]) for entry in reader.entries] != required_prefix:
        raise PermissionError("Candidate7 source inventory lacks exact external authority")
    if (
        reader.entries[-2]["sha256"] != authorization.authorization_sha256
        or reader.entries[-1]["sha256"]
        != authorization.qualification_record_sha256
    ):
        raise PermissionError("Candidate7 authority seal/read ledger mismatch")
    try:
        qualification = qualify_source_controls(context, reader=reader)
        stage = "SOURCE_INVENTORY_CONSUMPTION"
        consumption = consume_source_inventory_authorization(context, authorization)
        if after_authorization_consumed is not None:
            after_authorization_consumed(consumption.path)
        revalidate_consumption_anchor(consumption)
        stage = "CACHE_MANIFEST_VALIDATION"
        qualification.reader.authorize(
            qualification.manifest_path,
            purpose="TRAIN_CACHE_MANIFEST",
            sha256=qualification.manifest_sha256,
            size_bytes=qualification.manifest_size_bytes,
        )
        manifest = _decode_object(
            qualification.reader.read_bytes(
                qualification.manifest_path, purpose="TRAIN_CACHE_MANIFEST"
            ),
            name="TRAIN_CACHE_MANIFEST",
        )
        records = _validate_cache_manifest(
            manifest,
            train_ids=qualification.train_ids,
            scan_by_case=qualification.scan_by_case,
            train_commitment=qualification.train_commitment,
        )
        stage = "TRAIN_SOURCE_COMPATIBILITY"
        _read_train_sources(context, qualification, records)
        _validate_success_inventory(qualification)
    except BaseException as error:
        reader = qualification.reader if qualification is not None else reader
        if isinstance(error, CheckFailure):
            failure_stage = error.stage
            check_code = error.code
        else:
            failure_stage = stage
            check_code = f"{stage}_UNEXPECTED_EXCEPTION"
        ledger = _ledger_payload(
            reader,
            source_bindings=context.source_bindings,
            status="TERMINAL_FAILURE",
            failure_stage=failure_stage,
            check_code=check_code,
            exact_train_inventory_completed=False,
        )
        ledger_bytes = canonical_bytes(ledger)
        terminal = {
            "attempt_ordinal": 7,
            "candidate7_fit_authorized": False,
            "candidate7_preregistration_authorized": False,
            "check_code": check_code,
            "continuation_authorized": False,
            "engineering_attempt_id": "CANDIDATE7_ENGINEERING_ATTEMPT_V1",
            "failure_stage": failure_stage,
            "full_28_accessed": False,
            "holdout_score_opened": False,
            "kind": TERMINAL_KIND,
            "label_attestation_or_projection_opened": False,
            "model_fit_performed": False,
            "output_inventory": sorted(FAILURE_FILENAMES),
            "preflight_read_ledger_sha256": sha256_bytes(ledger_bytes),
            "schema_version": 1,
            "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
            "source_inventory_consumption_anchor": (
                None
                if consumption is None
                else {
                    "path": str(consumption.path.relative_to(ROOT)),
                    "sha256": consumption.sha256,
                    "size_bytes": consumption.size_bytes,
                }
            ),
            "status": "FAILED_ENGINEERING_SOURCE_COMPATIBILITY",
        }
        _assert_aggregate_safe(terminal)
        return {
            "preflight_read_ledger.json": ledger_bytes,
            "preflight_terminal.json": canonical_bytes(terminal),
        }
    if qualification is None:
        raise RuntimeError("Candidate7 success lacks qualification")
    if consumption is None:
        raise RuntimeError("Candidate7 success lacks durable authorization consumption")
    ledger = _ledger_payload(
        qualification.reader,
        source_bindings=context.source_bindings,
        status="PASS",
        failure_stage=None,
        check_code="ALL_SOURCE_COMPATIBILITY_CHECKS_PASSED",
        exact_train_inventory_completed=True,
    )
    ledger_bytes = canonical_bytes(ledger)
    ledger_sha = sha256_bytes(ledger_bytes)
    read_by_purpose = {
        str(entry["purpose"]): entry for entry in qualification.reader.entries
    }
    bindings = {
        "candidate1_terminal_sha256_not_opened": (
            "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
        ),
        "candidate2_terminal_sha256_not_opened": (
            "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515"
        ),
        "candidate3_terminal_sha256_not_opened": (
            "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116"
        ),
        "candidate4_terminal_sha256_not_opened": (
            "10d6bd14ad10dfd1ace464e551cb79f65abeb3c12bc0094fb2695f61a1aa2a50"
        ),
        "candidate5_terminal_sha256_not_opened": (
            "6b7d448b1c0b16c12cc32debb61c5255dc3ddb18956f4807b0c0090fbddc9049"
        ),
        "candidate6_terminal_disposition_sha256": FIXED_CONTROLS[
            "CANDIDATE6_TERMINAL_DISPOSITION"
        ][1],
        "candidate7_source_control_qualification_record_sha256": read_by_purpose[
            "CANDIDATE7_SOURCE_CONTROL_QUALIFICATION_RECORD"
        ]["sha256"],
        "candidate7_source_inventory_authorization_sha256": read_by_purpose[
            "CANDIDATE7_SOURCE_INVENTORY_AUTHORIZATION"
        ]["sha256"],
        "candidate7_source_inventory_consumption_anchor_sha256": consumption.sha256,
        "candidate6_preflight_ledger_sha256_not_opened": (
            "2cd0525372cdfc205e97b885a109fa061580f83d00c774e262edb4ba7e999f1a"
        ),
        "candidate6_preflight_terminal_sha256_not_opened": (
            "981a6711a8f5a3639caf33f3d43e5d66453cbd25d8ce325f53250fb0fff4848d"
        ),
        "custody_cache_attestation_sha256": FIXED_CONTROLS[
            "TRAIN_CACHE_ATTESTATION"
        ][1],
        "custody_label_attestation_sha256_not_opened": (
            CUSTODY_LABEL_ATTESTATION_SHA256_NOT_OPENED
        ),
        "custody_staged_validator_pass_sha256": CUSTODY_PASS_SHA256,
        "hypothesis_source_sha256": context.source_bindings["hypothesis"]["sha256"],
        "preflight_source_sha256": context.source_bindings["preflight"]["sha256"],
        "preflight_tests_sha256": context.source_bindings["preflight_tests"]["sha256"],
        "original_preregistration_sha256": FIXED_CONTROLS[
            "ORIGINAL_PREREGISTRATION"
        ][1],
        "runtime_source_sha256": context.source_bindings["runtime"]["sha256"],
        "runtime_tests_sha256": context.source_bindings["runtime_tests"]["sha256"],
        "sealed_split_sha256": FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
    }
    report = {
        "all_fixed_rule_forms_exercised": True,
        "all_primary_replay_pool_bytes_exact": True,
        "all_primary_replay_semantic_bytes_exact": True,
        "attempt_ordinal": 7,
        "attestation_checks": dict(qualification.attestation_checks),
        "bindings": bindings,
        "cache_manifest_opened": True,
        "candidate6_terminal_disposition_opened": True,
        "candidate6_terminal_or_ledger_opened": False,
        "candidate7_fit_authorized": False,
        "candidate7_preregistration_authorized": False,
        "case_count": 18,
        "check_code": "ALL_SOURCE_COMPATIBILITY_CHECKS_PASSED",
        "engineering_attempt_id": "CANDIDATE7_ENGINEERING_ATTEMPT_V1",
        "failure_stage": None,
        "full_28_manifest_path_resolved_or_opened": False,
        "holdout_path_resolved_or_opened": False,
        "holdout_score_opened": False,
        "kind": REPORT_KIND,
        "label_attestation_or_projection_path_resolved_or_opened": False,
        "model_fit_performed": False,
        "pool_and_semantic_schema_compatible": True,
        "preflight_read_ledger_sha256": ledger_sha,
        "record_count": 36,
        "schema_version": 1,
        "scientific_hypothesis_exactly_preserved": True,
        "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
        "source_inventory_authorization_consumed": True,
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        "v51_prediction_values_and_order_preserved": True,
    }
    _assert_aggregate_safe(report)
    report_bytes = canonical_bytes(report)
    lock = {
        "attempt_ordinal": 7,
        "bindings": bindings,
        "candidate7_fit_authorized": False,
        "candidate7_holdout_execution_authorized": False,
        "candidate7_preregistration_authorized": False,
        "engineering_attempt_id": "CANDIDATE7_ENGINEERING_ATTEMPT_V1",
        "holdout_score_opened": False,
        "kind": LOCK_KIND,
        "post_preflight_source_or_policy_mutation": "INVALIDATES_THIS_LOCK",
        "preflight_read_ledger_sha256": ledger_sha,
        "preflight_report_sha256": sha256_bytes(report_bytes),
        "schema_version": 1,
        "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
        "source_inventory_authorization_consumed": True,
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
    }
    _assert_aggregate_safe(lock)
    return {
        "preflight_lock.json": canonical_bytes(lock),
        "preflight_read_ledger.json": ledger_bytes,
        "preflight_report.json": report_bytes,
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
        raise RuntimeError("Candidate7 requires no-replace rename")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        flag,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, "Candidate7 destination exists")
        raise OSError(number, os.strerror(number))


def _staging_prefix(output: Path) -> str:
    return f".{output.name}.staging-"


def _quarantine_prefix(output: Path) -> str:
    return f".{output.name}.quarantine-"


def _sibling_names(parent_fd: int, *, prefix: str) -> tuple[str, ...]:
    return tuple(sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix)))


def _validate_output(output: Path) -> Path:
    candidate = _lexical_repo_path(output, role="Candidate7 preflight output")
    expected = _lexical_repo_path(OUTPUT_ROOT, role="frozen Candidate7 preflight output")
    if candidate != expected:
        raise ValueError("Candidate7 output namespace is frozen")
    parent_fd, _parent_identity = _open_stable_directory(
        candidate.parent, role="Candidate7 preflight output parent"
    )
    try:
        if _sibling_names(parent_fd, prefix=_staging_prefix(candidate)) or _sibling_names(
            parent_fd, prefix=_quarantine_prefix(candidate)
        ):
            raise FileExistsError("Candidate7 rejects publication residue")
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate7 output must be fresh")
    finally:
        os.close(parent_fd)
    return candidate


def _open_owned_directory(
    parent_fd: int, name: str, expected_identity: tuple[int, int]
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
        dir_fd=parent_fd,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected_identity:
        os.close(descriptor)
        raise RuntimeError("Candidate7 owned directory identity changed")
    return descriptor


def _write_payload(staging_fd: int, name: str, data: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=staging_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate7 staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or int(metadata.st_size) != len(data)
        ):
            raise RuntimeError("Candidate7 staged payload mismatch")
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _verify_payloads(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    payloads: Mapping[str, bytes],
) -> None:
    directory_fd = _open_owned_directory(parent_fd, name, expected_identity)
    try:
        if set(os.listdir(directory_fd)) != set(payloads):
            raise RuntimeError("Candidate7 publication inventory changed")
        identities: set[tuple[int, int]] = set()
        for filename, expected in sorted(payloads.items()):
            descriptor = os.open(
                filename,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
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
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or identity in identities
                    or _identity(after) != identity
                    or int(after.st_size) != len(data)
                    or data != expected
                ):
                    raise RuntimeError("Candidate7 publication payload changed")
                identities.add(identity)
            finally:
                os.close(descriptor)
    finally:
        os.close(directory_fd)


def _quarantine_owned(
    parent_fd: int,
    source_name: str,
    output: Path,
    expected_identity: tuple[int, int],
) -> str:
    before = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(before) != expected_identity:
        raise RuntimeError("Candidate7 refuses to quarantine an unowned inode")
    destination = f"{_quarantine_prefix(output)}{secrets.token_hex(16)}"
    _renameat_noreplace(parent_fd, source_name, destination)
    after = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(after) != expected_identity:
        raise RuntimeError("Candidate7 quarantine identity mismatch")
    os.fsync(parent_fd)
    return destination


def publish_atomically(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    after_stage_write: Callable[[Path], None] | None = None,
    after_rename: Callable[[Path], None] | None = None,
) -> None:
    names = set(payloads)
    if names not in (PASS_FILENAMES, FAILURE_FILENAMES):
        raise ValueError("Candidate7 publication inventory mismatch")
    if any(Path(name).name != name or not name.endswith(".json") for name in names):
        raise ValueError("Candidate7 publication filenames must be flat JSON")
    if any(not isinstance(data, bytes) for data in payloads.values()):
        raise ValueError("Candidate7 publication payloads must be bytes")
    output = _validate_output(output)
    parent_fd, parent_identity = _open_stable_directory(
        output.parent, role="Candidate7 preflight output parent"
    )
    staging_name = ""
    staging_identity: tuple[int, int] | None = None
    renamed = False
    try:
        _assert_visible_directory_identity(
            output.parent, parent_identity, role="Candidate7 visible output parent"
        )
        if _sibling_names(parent_fd, prefix=_staging_prefix(output)) or _sibling_names(
            parent_fd, prefix=_quarantine_prefix(output)
        ):
            raise FileExistsError("Candidate7 publication residue exists")
        staging_name = f"{_staging_prefix(output)}{secrets.token_hex(16)}"
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        metadata = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        staging_identity = _identity(metadata)
        staging_fd = _open_owned_directory(parent_fd, staging_name, staging_identity)
        try:
            for name, data in sorted(payloads.items()):
                _write_payload(staging_fd, name, data)
            os.fsync(staging_fd)
            os.fchmod(staging_fd, 0o500)
        finally:
            os.close(staging_fd)
        if after_stage_write is not None:
            after_stage_write(output.parent / staging_name)
        _verify_payloads(parent_fd, staging_name, staging_identity, payloads)
        if _identity(os.fstat(parent_fd)) != parent_identity:
            raise RuntimeError("Candidate7 pinned output parent changed")
        _assert_visible_directory_identity(
            output.parent, parent_identity, role="Candidate7 visible output parent"
        )
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate7 output raced into existence")
        _renameat_noreplace(parent_fd, staging_name, output.name)
        renamed = True
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate7 published directory identity mismatch")
        _verify_payloads(parent_fd, output.name, staging_identity, payloads)
        if after_rename is not None:
            after_rename(output)
        published = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != staging_identity:
            raise RuntimeError("Candidate7 published directory changed")
        _assert_visible_directory_identity(
            output.parent, parent_identity, role="Candidate7 visible output parent"
        )
        _verify_payloads(parent_fd, output.name, staging_identity, payloads)
        os.fsync(parent_fd)
    except BaseException:
        if staging_identity is not None:
            name = output.name if renamed else staging_name
            try:
                metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                metadata = None
            if metadata is not None and _identity(metadata) == staging_identity:
                _quarantine_owned(parent_fd, name, output, staging_identity)
        raise
    finally:
        os.close(parent_fd)


def run(
    context: BootstrapContext,
    *,
    authorization: SourceInventoryAuthorizationSeal,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output = _validate_output(output_root)
    payloads = build_preflight_payloads(context, authorization=authorization)
    publish_atomically(output, payloads)
    return {
        "case_count": 18,
        "check_code": (
            "ALL_SOURCE_COMPATIBILITY_CHECKS_PASSED"
            if set(payloads) == PASS_FILENAMES
            else json.loads(payloads["preflight_terminal.json"])["check_code"]
        ),
        "failure_stage": (
            None
            if set(payloads) == PASS_FILENAMES
            else json.loads(payloads["preflight_terminal.json"])["failure_stage"]
        ),
        "label_attestation_or_projection_opened": False,
        "output_inventory": sorted(payloads),
        "published_artifact_sha256": {
            name: sha256_bytes(data) for name, data in sorted(payloads.items())
        },
        "record_count": 36,
        "status": (
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING"
            if set(payloads) == PASS_FILENAMES
            else "FAILED_ENGINEERING_SOURCE_COMPATIBILITY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--runtime-tests-sha256", required=True)
    parser.add_argument("--preflight-tests-sha256", required=True)
    parser.add_argument("--hypothesis-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--source-control-check-only", action="store_true")
    modes.add_argument("--source-inventory-once", action="store_true")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args()
    if args.source_inventory_once and args.authorization_sha256 is None:
        parser.error("--source-inventory-once requires --authorization-sha256")
    if not args.source_inventory_once and args.authorization_sha256 is not None:
        parser.error("--authorization-sha256 is valid only with --source-inventory-once")
    context = bootstrap_verified_sources(
        runtime_sha256=args.runtime_sha256,
        preflight_sha256=args.preflight_sha256,
        runtime_tests_sha256=args.runtime_tests_sha256,
        preflight_tests_sha256=args.preflight_tests_sha256,
        hypothesis_sha256=args.hypothesis_sha256,
    )
    if args.source_check_only:
        value = {
            "artifact_input_opened": False,
            "label_or_holdout_opened": False,
            "source_binding_count": 5,
            "status": "SOURCE_BYTES_VERIFIED_ONLY",
        }
    elif args.source_control_check_only:
        value = qualification_result(context)
    else:
        reader = GuardedReader(
            entries=[dict(entry) for entry in context.bootstrap_entries]
        )
        try:
            authorization = authorize_source_inventory_once(
                context,
                reader=reader,
                authorization_sha256=str(args.authorization_sha256),
            )
        except CheckFailure as error:
            value = {
                "artifact_source_inventory_opened": False,
                "check_code": error.code,
                "failure_stage": error.stage,
                "holdout_or_label_opened": False,
                "status": "SOURCE_INVENTORY_AUTHORIZATION_FAILED",
            }
        except BaseException:
            value = {
                "artifact_source_inventory_opened": False,
                "check_code": "AUTHORIZATION_UNEXPECTED_EXCEPTION",
                "failure_stage": "SOURCE_INVENTORY_AUTHORIZATION",
                "holdout_or_label_opened": False,
                "status": "SOURCE_INVENTORY_AUTHORIZATION_FAILED",
            }
        else:
            value = run(context, authorization=authorization)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 2 if "FAILED" in str(value["status"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
