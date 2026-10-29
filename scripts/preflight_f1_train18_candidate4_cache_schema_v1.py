#!/usr/bin/env python3
"""Label-incapable TRAIN-cache compatibility preflight for Candidate4.

This process has no label-attestation or projection path.  It verifies exact
source bytes before dynamically loading the Candidate4 runtime, consumes only
the sealed 18-case / 36-record TRAIN cache manifest, and opens only V51 semantic
artifacts needed by the sidecar-burst guard.  Scan identity comes exclusively
from the sealed allowlist joined to the attested cache manifest; semantic-internal
identity is neither trusted nor compared.  Successful execution publishes only
an aggregate report, a complete read ledger, and a nonauthorizing lock.
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
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SOURCE = Path(__file__).resolve()
RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate4_sidecar_burst_guard_v1.py"
RUNTIME_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate4_sidecar_burst_guard_v1.py"
PREFLIGHT_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate4_cache_preflight_v1.py"
RUNTIME_MODULE = "backend.f1_train18_candidate4_sidecar_burst_guard_v1"

SPLIT = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json"
SPLIT_SHA256 = "77681a938f9784b26258281af65a8097fe1670885898bfa34fad4d54aa0ff185"
SPLIT_SIZE_BYTES = 10706
ORIGINAL_PREREG = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json"
ORIGINAL_PREREG_SHA256 = "f1c029593033ae50fcdface54e36c6847f4280aa76352cbed55b0eb10a27df0a"
ORIGINAL_PREREG_SIZE_BYTES = 2651
ALLOWLIST = ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json"
ALLOWLIST_SHA256 = "059e9e6acf934de5a9b3745251f2baea795ced0c2f0bbca57202dcae979bb2f0"
ALLOWLIST_SIZE_BYTES = 2428
CANDIDATE1_TERMINAL = ROOT / "research/f1_aug30_train18_candidate1_terminal_negative_v1.json"
CANDIDATE1_TERMINAL_SHA256 = "b470f8aa22d3157482fc0d52f34151b6dd3c9a30cda3fc08c1eebbdfbc0668c8"
CANDIDATE1_TERMINAL_SIZE_BYTES = 3989
CANDIDATE2_TERMINAL = ROOT / "research/f1_aug30_train18_candidate2_terminal_negative_v1.json"
CANDIDATE2_TERMINAL_SHA256 = "ecdbe871364044214fc46e5b5e59095cc9b38cefe9678292f1a8462dffdf3515"
CANDIDATE2_TERMINAL_SIZE_BYTES = 7001
CANDIDATE3_TERMINAL = ROOT / "research/f1_aug30_train18_candidate3_terminal_negative_v1.json"
CANDIDATE3_TERMINAL_SHA256 = "00449059f9e4cd989f3fbcb67e9fc8e7a499ad1dbca0a947deccd75d21019116"
CANDIDATE3_TERMINAL_SIZE_BYTES = 5518
CANDIDATE3_INCIDENT = ROOT / "research/f1_aug30_train18_candidate3_prefit_custody_incident_v1.json"
CANDIDATE3_INCIDENT_SHA256 = "05c044224918a9400f7c7da23f508ec3ca69c44c1b9f5c3700dcee1c8f4021f1"
CANDIDATE3_INCIDENT_SIZE_BYTES = 2277
CACHE_ATTESTATION = (
    ROOT
    / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json"
)
CACHE_ATTESTATION_SHA256 = "b5fbc8f5ac1d749108e8fcaa9c7111b2c74c0a8284cd4427b72ef2e2ff216e71"
CACHE_ATTESTATION_SIZE_BYTES = 1834
CUSTODY_LABEL_ATTESTATION_SHA256_NOT_OPENED = (
    "fe9313bbd09de945fc4da0c238a6b1f2769979e78cccc2abff29ca434dbf84ad"
)
CUSTODY_PASS_SHA256 = "8adf33fde808f502116b8db40ab8819e30aeae233784b2c7af8baa31c97e2433"
V51_SCAN_ONLY_MANIFEST_SHA256 = (
    "11daf93a1cff70ff68d0d6786ca8a96427a73df65aec51542ca30880c3f4df3c"
)

CACHE_CUSTODY_ROOT = ROOT / "output/research/f1-aug30-train-only-custody-v2"
CACHE_MANIFEST = CACHE_CUSTODY_ROOT / "cache/train_v51_cache_manifest.json"
CACHE_EVIDENCE_PATHS = {
    "builder": ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py",
    "validator": ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py",
    "staged_validator_pass": CACHE_CUSTODY_ROOT / "independent_validation_report.json",
    "access_evidence": CACHE_CUSTODY_ROOT / "cache/access_evidence.json",
}
CUSTODY_VALIDATION_KIND = (
    "sherman_qc_f1_aug30_train_only_staged_validator_pass_v1"
)

ALLOWED_CACHE_ROOT = (
    ROOT / "output/research/f1-v51-today-scan-only-development-20260830-r2"
)
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate4-cache-preflight-v1"

REPORT_KIND = "sherman_qc_f1_aug30_train18_candidate4_cache_preflight_report_v1"
LOCK_KIND = "sherman_qc_f1_aug30_train18_candidate4_cache_preflight_lock_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate4_cache_preflight_read_ledger_v1"
TRAIN_CACHE_ATTESTATION_KIND = (
    "sherman_qc_f1_aug30_train18_v51_cache_manifest_attestation_v1"
)
TRAIN_CACHE_MANIFEST_KIND = "sherman_qc_f1_aug30_train18_v51_cache_manifest_v1"

CACHE_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "canonical_json",
        "split_sha256",
        "train_allowlist_sha256",
        "train_set_commitment_sha256",
        "blinded_complement_commitment_sha256",
        "case_count",
        "record_count",
        "train_exact_match",
        "train_holdout_intersection_count",
        "holdout_entry_count",
        "holdout_status_count",
        "holdout_hash_count",
        "optimizer_holdout_access_count",
        "independent_validation_passed",
        "manifest_binding",
        "builder_binding",
        "validator_binding",
        "staged_validator_pass_binding",
        "access_evidence_binding",
    }
)

CACHE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "split_sha256",
        "train_allowlist_sha256",
        "train_set_commitment_sha256",
        "v51_scan_only_manifest_sha256",
        "source_only",
        "case_count",
        "record_count",
        "holdout_entry_count",
        "records",
    }
)

FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    RUNTIME_MODULE,
    "scripts.optimize_f1_train18_source_only_v1",
    "scripts.evaluate_f1_against_manual_gold",
    "scripts.optimize_f1_train18_candidate4_sidecar_burst_guard_v1",
)

FORBIDDEN_AUTHORIZED_BASENAMES = frozenset(
    {
        "bend_line_development_catalog_v2_20260830.json",
        "complete_label_manifest.json",
        "development_execution_manifest.json",
        "train_label_projection.json",
        "train_label_projection_attestation.json",
    }
)

FORBIDDEN_AGGREGATE_KEYS = frozenset(
    {
        "bend_id",
        "candidate_count",
        "deleted_sidecar_count",
        "observed_sidecar_multiplicity",
        "prediction_count",
        "predictions",
        "selector_score",
        "source_kind",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if text == "0" * 64:
        raise ValueError(f"{name} must not be an unresolved zero commitment")
    return text


def _read_verified_source_once(
    path: Path, *, expected_sha256: str, role: str
) -> tuple[bytes, dict[str, Any]]:
    data, identity = _read_regular_file_fd(path, role=role)
    if sha256_bytes(data) != expected_sha256:
        raise ValueError(f"{role} source hash mismatch")
    return data, {
        "case_id": None,
        "path": str(_lexical_repository_path(path, role=role).relative_to(ROOT)),
        "role": role,
        "sha256": expected_sha256,
        "size_bytes": len(data),
        "device_id": identity[0],
        "inode": identity[1],
    }


def _lexical_absolute(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} path must not contain parent traversal")
    return Path(os.path.abspath(os.fspath(candidate)))


def _lexical_repository_path(path: Path, *, role: str) -> Path:
    candidate = _lexical_absolute(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    try:
        inside = os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(root)
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"{role} path leaves repository root")
    return candidate


def _reject_symlink_chain(
    path: Path, *, role: str, require_exists: bool = False
) -> Path:
    candidate = _lexical_repository_path(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    current = candidate
    chain: list[Path] = []
    while current != root:
        chain.append(current)
        current = current.parent
    chain.append(root)
    missing_seen = False
    for current in reversed(chain):
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError(f"{role} path has an existing child below a missing ancestor")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path or ancestor must not be a symlink")
        if current != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor must be a directory")
    if require_exists and missing_seen:
        raise FileNotFoundError(f"{role} path must exist")
    return candidate


def _regular_file_identity(path: Path, *, role: str) -> tuple[Path, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    metadata = os.lstat(candidate)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular file")
    return candidate, (int(metadata.st_dev), int(metadata.st_ino))


def _read_regular_file_fd(
    path: Path,
    *,
    role: str,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[bytes, tuple[int, int]]:
    candidate, pre_identity = _regular_file_identity(path, role=role)
    if expected_identity is not None and pre_identity != expected_identity:
        raise ValueError(f"{role} file identity changed before open")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate4 preflight requires O_NOFOLLOW")
    parent_fd, _parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        pre_leaf = os.stat(
            candidate.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(pre_leaf.st_mode)
            or (int(pre_leaf.st_dev), int(pre_leaf.st_ino)) != pre_identity
        ):
            raise ValueError(f"{role} file identity changed before fd open")
        descriptor = os.open(
            candidate.name,
            os.O_RDONLY | nofollow,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        opened_identity = (int(opened.st_dev), int(opened.st_ino))
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{role} opened object is not a regular file")
        if opened_identity != pre_identity or (
            expected_identity is not None and opened_identity != expected_identity
        ):
            raise ValueError(f"{role} file identity changed during open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (int(after.st_dev), int(after.st_ino)) != opened_identity:
            raise ValueError(f"{role} file identity changed during read")
        data = b"".join(chunks)
        if len(data) != int(after.st_size):
            raise ValueError(f"{role} file size changed during read")
        post_leaf = os.stat(
            candidate.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(post_leaf.st_mode)
            or (int(post_leaf.st_dev), int(post_leaf.st_ino)) != opened_identity
        ):
            raise ValueError(f"{role} path identity changed after read")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return data, opened_identity


def bootstrap_verified_sources(
    *,
    runtime_sha256: str,
    preflight_sha256: str,
    runtime_tests_sha256: str,
    preflight_tests_sha256: str,
) -> tuple[types.ModuleType, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    for name in FORBIDDEN_PRELOADED_MODULES:
        if name in sys.modules:
            raise RuntimeError(f"Candidate4 preflight rejects preloaded module: {name}")
    expected = {
        RUNTIME_SOURCE: (
            _require_sha(runtime_sha256, name="runtime SHA"),
            "CANDIDATE4_RUNTIME_SOURCE",
        ),
        PREFLIGHT_SOURCE: (
            _require_sha(preflight_sha256, name="preflight SHA"),
            "CANDIDATE4_PREFLIGHT_SOURCE",
        ),
        RUNTIME_TESTS_SOURCE: (
            _require_sha(runtime_tests_sha256, name="runtime tests SHA"),
            "CANDIDATE4_RUNTIME_TESTS_SOURCE",
        ),
        PREFLIGHT_TESTS_SOURCE: (
            _require_sha(preflight_tests_sha256, name="preflight tests SHA"),
            "CANDIDATE4_PREFLIGHT_TESTS_SOURCE",
        ),
    }
    source_bytes: dict[Path, bytes] = {}
    entries: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    source_identities: dict[tuple[int, int], Path] = {}
    for path, (sha, role) in expected.items():
        data, entry = _read_verified_source_once(
            path, expected_sha256=sha, role=role
        )
        lexical_path = _lexical_repository_path(path, role=role)
        identity = (int(entry["device_id"]), int(entry["inode"]))
        aliased = source_identities.get(identity)
        if aliased is not None and aliased != lexical_path:
            raise ValueError("Candidate4 source files must not be hard-link aliases")
        source_identities[identity] = lexical_path
        source_bytes[lexical_path] = data
        entries.append(entry)
        bindings[role.lower()] = {
            "path": entry["path"],
            "sha256": sha,
            "size_bytes": len(data),
        }

    module = types.ModuleType(RUNTIME_MODULE)
    runtime_path = _lexical_repository_path(
        RUNTIME_SOURCE, role="CANDIDATE4_RUNTIME_SOURCE"
    )
    module.__file__ = str(runtime_path)
    module.__package__ = "backend"
    module.__loader__ = None
    code = compile(
        source_bytes[runtime_path],
        str(runtime_path),
        "exec",
        dont_inherit=True,
    )
    sys.modules[RUNTIME_MODULE] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(RUNTIME_MODULE, None)
        raise
    if _lexical_absolute(Path(str(module.__file__)), role="verified runtime module") != runtime_path:
        raise RuntimeError("Candidate4 verified runtime resolved from the wrong path")
    return module, bindings, entries


@dataclass
class GuardedReader:
    allowed: dict[Path, dict[str, Any]] = field(default_factory=dict)
    entries: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)
    file_identities: dict[tuple[int, int], Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for entry in self.entries:
            if "device_id" not in entry or "inode" not in entry:
                continue
            identity = (int(entry["device_id"]), int(entry["inode"]))
            path = _lexical_repository_path(
                ROOT / str(entry["path"]), role=str(entry.get("role") or "BOOTSTRAP")
            )
            prior = self.file_identities.get(identity)
            if prior is not None and prior != path:
                raise ValueError("Candidate4 bootstrap inputs contain a hard-link alias")
            self.file_identities[identity] = path

    def authorize(
        self,
        path: Path,
        *,
        role: str,
        sha256: str,
        size_bytes: int | None = None,
    ) -> None:
        candidate = _lexical_repository_path(path, role=role)
        if candidate.name in FORBIDDEN_AUTHORIZED_BASENAMES:
            self.denied_attempts.append(
                {"path": str(candidate), "reason": "FORBIDDEN_BASENAME", "role": role}
            )
            raise PermissionError(f"Candidate4 preflight forbids path: {candidate.name}")
        candidate, identity = _regular_file_identity(candidate, role=role)
        if size_bytes is not None and (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"{role} size binding must be a positive integer")
        binding = {
            "role": role,
            "sha256": _require_sha(sha256, name=f"{role} SHA"),
            "size_bytes": size_bytes,
            "device_id": identity[0],
            "inode": identity[1],
        }
        prior = self.allowed.get(candidate)
        if prior is not None and prior != binding:
            raise ValueError(f"conflicting Candidate4 authorization: {candidate}")
        aliased = self.file_identities.get(identity)
        if aliased is not None and aliased != candidate:
            raise ValueError(
                f"Candidate4 input hard-link alias rejected: {candidate} aliases {aliased}"
            )
        self.file_identities[identity] = candidate
        self.allowed[candidate] = binding

    def read_bytes(
        self, path: Path, *, role: str, case_id: str | None = None
    ) -> bytes:
        candidate = _lexical_repository_path(path, role=role)
        binding = self.allowed.get(candidate)
        if binding is None or binding["role"] != role:
            self.denied_attempts.append(
                {"path": str(candidate), "reason": "NOT_AUTHORIZED", "role": role}
            )
            raise PermissionError(f"Candidate4 preflight read denied: {candidate}")
        expected_identity = (int(binding["device_id"]), int(binding["inode"]))
        data, observed_identity = _read_regular_file_fd(
            candidate, role=role, expected_identity=expected_identity
        )
        if observed_identity != expected_identity:
            raise ValueError(f"Candidate4 preflight identity mismatch: {candidate}")
        if sha256_bytes(data) != binding["sha256"]:
            raise ValueError(f"Candidate4 preflight hash mismatch: {candidate}")
        if binding["size_bytes"] is not None and len(data) != binding["size_bytes"]:
            raise ValueError(f"Candidate4 preflight size mismatch: {candidate}")
        self.entries.append(
            {
                "case_id": case_id,
                "path": str(candidate.relative_to(ROOT)),
                "role": role,
                "sha256": binding["sha256"],
                "size_bytes": len(data),
                "device_id": observed_identity[0],
                "inode": observed_identity[1],
            }
        )
        return data


def _decode_object(data: bytes, *, role: str, canonical_required: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    if canonical_required and canonical_bytes(value) != data:
        raise ValueError(f"{role} must be LF-canonical JSON")
    return value


def _binding_path(
    binding: Mapping[str, Any],
    *,
    role: str,
    expected_path: Path,
    expected_kind: str | None = None,
) -> tuple[Path, str, int]:
    expected_keys = {"path", "sha256", "size_bytes"}
    if expected_kind is not None:
        expected_keys.add("kind")
    if set(binding) != expected_keys:
        raise ValueError(f"{role} binding exact-key schema mismatch")
    if expected_kind is not None and binding["kind"] != expected_kind:
        raise ValueError(f"{role} binding kind mismatch")
    relative = str(binding.get("path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"{role} path must be repository-relative")
    unresolved = ROOT / relative
    candidate = _reject_symlink_chain(unresolved, role=role, require_exists=True)
    expected = _lexical_repository_path(expected_path, role=f"{role} expected path")
    if candidate != expected:
        raise PermissionError(f"{role} path differs from the exact custody allowlist")
    size = binding.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{role} size binding is invalid")
    return candidate, _require_sha(binding.get("sha256"), name=f"{role} SHA"), size


def _validate_fixed_lineage(
    runtime: types.ModuleType, reader: GuardedReader
) -> tuple[tuple[str, ...], dict[str, str], str, dict[str, Any]]:
    fixed = (
        (SPLIT, SPLIT_SHA256, SPLIT_SIZE_BYTES, "SEALED_SPLIT"),
        (
            ORIGINAL_PREREG,
            ORIGINAL_PREREG_SHA256,
            ORIGINAL_PREREG_SIZE_BYTES,
            "ORIGINAL_PREREGISTRATION",
        ),
        (ALLOWLIST, ALLOWLIST_SHA256, ALLOWLIST_SIZE_BYTES, "TRAIN_ALLOWLIST"),
        (
            CANDIDATE1_TERMINAL,
            CANDIDATE1_TERMINAL_SHA256,
            CANDIDATE1_TERMINAL_SIZE_BYTES,
            "CANDIDATE1_TERMINAL",
        ),
        (
            CANDIDATE2_TERMINAL,
            CANDIDATE2_TERMINAL_SHA256,
            CANDIDATE2_TERMINAL_SIZE_BYTES,
            "CANDIDATE2_TERMINAL",
        ),
        (
            CANDIDATE3_TERMINAL,
            CANDIDATE3_TERMINAL_SHA256,
            CANDIDATE3_TERMINAL_SIZE_BYTES,
            "CANDIDATE3_TERMINAL",
        ),
        (
            CANDIDATE3_INCIDENT,
            CANDIDATE3_INCIDENT_SHA256,
            CANDIDATE3_INCIDENT_SIZE_BYTES,
            "CANDIDATE3_CUSTODY_INCIDENT",
        ),
        (
            CACHE_ATTESTATION,
            CACHE_ATTESTATION_SHA256,
            CACHE_ATTESTATION_SIZE_BYTES,
            "TRAIN_CACHE_ATTESTATION",
        ),
    )
    payloads: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for path, sha, size, role in fixed:
        reader.authorize(path, role=role, sha256=sha, size_bytes=size)
        raw[role] = reader.read_bytes(path, role=role)
        payloads[role] = _decode_object(raw[role], role=role)

    split = payloads["SEALED_SPLIT"]
    prereg = payloads["ORIGINAL_PREREGISTRATION"]
    allowlist = payloads["TRAIN_ALLOWLIST"]
    candidate1 = payloads["CANDIDATE1_TERMINAL"]
    candidate2 = payloads["CANDIDATE2_TERMINAL"]
    candidate3 = payloads["CANDIDATE3_TERMINAL"]
    incident = payloads["CANDIDATE3_CUSTODY_INCIDENT"]
    if (
        split.get("kind")
        != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_split_v1"
        or split.get("label_or_outcome_fields_present") is not False
    ):
        raise ValueError("sealed split exposes label or outcome fields")
    if (
        prereg.get("kind")
        != "sherman_qc_f1_aug30_retrospective_pseudo_unseen_prereg_v1"
        or prereg.get("status") != "SEALED_RETROSPECTIVE_PSEUDO_UNSEEN"
    ):
        raise ValueError("original preregistration status mismatch")
    if candidate1.get("candidate_id") != 1 or candidate1.get("status") != "REJECTED_TRAIN_GATE":
        raise ValueError("Candidate1 terminal mismatch")
    if (
        candidate2.get("attempt_ordinal") != 2
        or candidate2.get("status") != "REJECTED_PREFIT_SCHEMA_INCOMPATIBILITY"
        or candidate2.get("candidate2_continuation_authorized") is not False
        or candidate2.get("holdout_score_opened") is not False
    ):
        raise ValueError("Candidate2 terminal mismatch")
    runtime._validate_candidate3_terminal(
        raw["CANDIDATE3_TERMINAL"], expected_sha256=CANDIDATE3_TERMINAL_SHA256
    )
    if (
        incident.get("kind")
        != "sherman_qc_f1_aug30_train18_candidate3_prefit_custody_incident_v1"
        or incident.get("status") != "CONFIRMED_PREFIT_CUSTODY_BOUNDARY_VIOLATION"
        or incident.get("terminalization_required") is not True
        or incident.get("candidate_preregistration_created") is not False
        or incident.get("fit_started") is not False
        or incident.get("metric_computed") is not False
        or incident.get("model_created") is not False
        or incident.get("candidate_output_created") is not False
        or incident.get("holdout_access_count") != 0
        or incident.get("holdout_score_opened") is not False
    ):
        raise ValueError("Candidate3 custody incident mismatch")

    cases = allowlist.get("cases")
    if (
        allowlist.get("kind")
        != "sherman_qc_f1_aug30_train_only_scan_allowlist_v1"
        or allowlist.get("reject_any_case_not_listed") is not True
        or not isinstance(cases, list)
        or len(cases) != 18
    ):
        raise ValueError("TRAIN allowlist must contain exactly 18 cases")
    train_ids = tuple(str(row.get("case_id") or "") for row in cases)
    scan_by_case = {
        str(row.get("case_id") or ""): _require_sha(
            row.get("scan_sha256"), name="TRAIN scan SHA"
        )
        for row in cases
    }
    if (
        len(set(train_ids)) != 18
        or any(not case_id for case_id in train_ids)
        or tuple(str(value) for value in split.get("train_case_ids") or ()) != train_ids
        or allowlist.get("split_binding", {}).get("sha256") != SPLIT_SHA256
        or allowlist.get("prereg_binding", {}).get("sha256") != ORIGINAL_PREREG_SHA256
    ):
        raise ValueError("TRAIN split/allowlist binding mismatch")
    train_commitment = sha256_bytes(
        canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    return (
        train_ids,
        scan_by_case,
        train_commitment,
        payloads["TRAIN_CACHE_ATTESTATION"],
    )


def _validate_cache_attestation(
    attestation: Mapping[str, Any],
    *,
    train_commitment: str,
    reader: GuardedReader,
) -> tuple[Path, str, int, dict[str, bool]]:
    if set(attestation) != CACHE_ATTESTATION_KEYS:
        raise ValueError("TRAIN cache attestation exact-key schema mismatch")
    checks = {
        "schema_kind": (
            attestation.get("schema_version") == 1
            and attestation.get("kind") == TRAIN_CACHE_ATTESTATION_KIND
        ),
        "canonical_json": attestation.get("canonical_json") is True,
        "case_count": attestation.get("case_count") == 18,
        "record_count": attestation.get("record_count") == 36,
        "split_binding": attestation.get("split_sha256") == SPLIT_SHA256,
        "allowlist_binding": attestation.get("train_allowlist_sha256") == ALLOWLIST_SHA256,
        "train_commitment": attestation.get("train_set_commitment_sha256") == train_commitment,
        "train_exact": attestation.get("train_exact_match") is True,
        "train_holdout_intersection": attestation.get("train_holdout_intersection_count") == 0,
        "holdout_entries": attestation.get("holdout_entry_count") == 0,
        "holdout_hashes": attestation.get("holdout_hash_count") == 0,
        "holdout_status": attestation.get("holdout_status_count") == 0,
        "optimizer_holdout_access": attestation.get("optimizer_holdout_access_count") == 0,
        "independent_validation": attestation.get("independent_validation_passed") is True,
    }
    _require_sha(
        attestation.get("blinded_complement_commitment_sha256"),
        name="TRAIN cache blinded complement commitment",
    )
    if not all(checks.values()):
        raise ValueError("TRAIN cache attestation failed Candidate4 boundary checks")
    for prefix in ("builder", "validator", "staged_validator_pass", "access_evidence"):
        binding = attestation.get(f"{prefix}_binding")
        if not isinstance(binding, dict):
            raise ValueError(f"TRAIN cache attestation lacks {prefix} binding")
        expected_kind = (
            CUSTODY_VALIDATION_KIND if prefix == "staged_validator_pass" else None
        )
        path, sha, size = _binding_path(
            binding,
            role=f"CACHE_{prefix.upper()}",
            expected_path=CACHE_EVIDENCE_PATHS[prefix],
            expected_kind=expected_kind,
        )
        role = f"CACHE_CUSTODY_{prefix.upper()}"
        reader.authorize(path, role=role, sha256=sha, size_bytes=size)
        reader.read_bytes(path, role=role)
    staged = attestation.get("staged_validator_pass_binding") or {}
    if staged.get("sha256") != CUSTODY_PASS_SHA256:
        raise ValueError("TRAIN cache attestation custody PASS hash mismatch")
    manifest_binding = attestation.get("manifest_binding")
    if (
        not isinstance(manifest_binding, dict)
        or manifest_binding.get("kind") != TRAIN_CACHE_MANIFEST_KIND
    ):
        raise ValueError("TRAIN cache manifest binding is invalid")
    return (
        *_binding_path(
            manifest_binding,
            role="TRAIN_CACHE_MANIFEST",
            expected_path=CACHE_MANIFEST,
            expected_kind=TRAIN_CACHE_MANIFEST_KIND,
        ),
        checks,
    )


def _inside_cache_root(path: Path) -> bool:
    candidate = _lexical_repository_path(path, role="TRAIN cache artifact")
    cache_root = _lexical_repository_path(
        ALLOWED_CACHE_ROOT, role="TRAIN cache root"
    )
    return candidate == cache_root or cache_root in candidate.parents


def _validate_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    train_ids: tuple[str, ...],
    scan_by_case: Mapping[str, str],
    train_commitment: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if (
        set(manifest) != CACHE_MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != TRAIN_CACHE_MANIFEST_KIND
        or manifest.get("source_only") is not True
        or manifest.get("holdout_entry_count") != 0
        or manifest.get("split_sha256") != SPLIT_SHA256
        or manifest.get("train_allowlist_sha256") != ALLOWLIST_SHA256
        or manifest.get("train_set_commitment_sha256") != train_commitment
        or manifest.get("v51_scan_only_manifest_sha256")
        != V51_SCAN_ONLY_MANIFEST_SHA256
        or manifest.get("case_count") != 18
        or manifest.get("record_count") != 36
    ):
        raise ValueError("TRAIN cache manifest header is invalid")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 36:
        raise ValueError("TRAIN cache manifest must contain exactly 36 records")
    expected = [
        (case_id, pass_name)
        for case_id in train_ids
        for pass_name in ("primary", "replay")
    ]
    allowed_keys = {
        "case_id",
        "runtime_case_id",
        "pass",
        "scan_sha256",
        "semantic_path",
        "semantic_sha256",
        "pool_path",
        "pool_sha256",
    }
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    runtime_by_case: dict[str, str] = {}
    case_by_runtime: dict[str, str] = {}
    semantic_paths: set[Path] = set()
    pool_paths: set[Path] = set()
    for raw, pair in zip(records, expected):
        if not isinstance(raw, dict) or set(raw) != allowed_keys:
            raise ValueError("TRAIN cache record schema mismatch")
        if (str(raw["case_id"]), str(raw["pass"])) != pair:
            raise PermissionError("TRAIN cache record order or membership differs")
        case_id, pass_name = pair
        runtime_id = str(raw["runtime_case_id"])
        if (
            len(runtime_id) != 4
            or not runtime_id.startswith("T")
            or not runtime_id[1:].isdigit()
            or runtime_by_case.setdefault(case_id, runtime_id) != runtime_id
        ):
            raise ValueError("TRAIN primary/replay runtime mapping differs")
        if case_by_runtime.setdefault(runtime_id, case_id) != case_id:
            raise ValueError("TRAIN runtime mapping is not bijective")
        if _require_sha(raw["scan_sha256"], name="record scan SHA") != scan_by_case[case_id]:
            raise ValueError("manifest scan identity differs from TRAIN allowlist")
        semantic_relative = Path(str(raw["semantic_path"]))
        pool_relative = Path(str(raw["pool_path"]))
        semantic_path = _reject_symlink_chain(
            ROOT / semantic_relative, role="TRAIN_V51_SEMANTIC"
        )
        pool_path = _reject_symlink_chain(
            ROOT / pool_relative, role="TRAIN_SCAN_ONLY_POOL"
        )
        expected_semantic_path = (
            ALLOWED_CACHE_ROOT / pass_name / runtime_id / "semantic_result.json"
        )
        expected_pool_path = (
            ALLOWED_CACHE_ROOT
            / pass_name
            / runtime_id
            / "pool"
            / f"{runtime_id}.json"
        )
        expected_semantic_path = _lexical_repository_path(
            expected_semantic_path, role="expected TRAIN semantic path"
        )
        expected_pool_path = _lexical_repository_path(
            expected_pool_path, role="expected TRAIN pool path"
        )
        if (
            semantic_relative.is_absolute()
            or pool_relative.is_absolute()
            or not _inside_cache_root(semantic_path)
            or not _inside_cache_root(pool_path)
            or semantic_path != expected_semantic_path
            or pool_path != expected_pool_path
        ):
            raise ValueError("TRAIN cache record path contract mismatch")
        if semantic_path in semantic_paths or pool_path in pool_paths:
            raise ValueError("TRAIN cache record paths must be unique across all 36 records")
        semantic_paths.add(semantic_path)
        pool_paths.add(pool_path)
        mapping[pair] = {
            "case_id": case_id,
            "pass": pass_name,
            "runtime_case_id": runtime_id,
            "scan_sha256": scan_by_case[case_id],
            "semantic_path": semantic_path,
            "semantic_sha256": _require_sha(
                raw["semantic_sha256"], name="semantic SHA"
            ),
            "pool_path": pool_path,
            "pool_sha256": _require_sha(raw["pool_sha256"], name="pool SHA"),
        }
    for case_id in train_ids:
        primary = mapping[(case_id, "primary")]
        replay = mapping[(case_id, "replay")]
        if (
            primary["semantic_sha256"] != replay["semantic_sha256"]
            or primary["pool_sha256"] != replay["pool_sha256"]
        ):
            raise ValueError("TRAIN primary/replay cache hashes differ")
    return mapping


def _assert_aggregate_safe(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = set(value).intersection(FORBIDDEN_AGGREGATE_KEYS)
        if forbidden:
            raise ValueError(f"aggregate preflight output exposes forbidden keys: {forbidden}")
        for child in value.values():
            _assert_aggregate_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_aggregate_safe(child)


def _validate_output_root(path: Path) -> Path:
    candidate = _lexical_repository_path(path, role="Candidate4 preflight output")
    expected = _lexical_repository_path(
        OUTPUT_ROOT, role="frozen Candidate4 preflight output"
    )
    if candidate != expected:
        raise ValueError("Candidate4 preflight output root is frozen")
    parent = _reject_symlink_chain(
        candidate.parent,
        role="Candidate4 preflight output parent",
        require_exists=True,
    )
    parent_metadata = os.lstat(parent)
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("Candidate4 preflight output parent must be a directory")
    _reject_symlink_chain(candidate, role="Candidate4 preflight output")
    if os.path.lexists(candidate):
        raise FileExistsError("Candidate4 preflight output root must be fresh and absent")
    parent_fd, _parent_identity = _open_stable_directory(
        parent, role="Candidate4 preflight output parent"
    )
    try:
        _assert_no_staging_residue(parent_fd, candidate)
        _assert_no_prior_quarantine(parent_fd, candidate)
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "Candidate4 preflight output root raced into existence"
            )
    finally:
        os.close(parent_fd)
    return candidate


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role, require_exists=True)
    before = os.lstat(candidate)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{role} must be a directory")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise RuntimeError("Candidate4 preflight requires O_NOFOLLOW and O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory_flag)
    opened = os.fstat(descriptor)
    before_identity = (int(before.st_dev), int(before.st_ino))
    opened_identity = (int(opened.st_dev), int(opened.st_ino))
    if not stat.S_ISDIR(opened.st_mode) or opened_identity != before_identity:
        os.close(descriptor)
        raise ValueError(f"{role} identity changed during open")
    post = os.lstat(_reject_symlink_chain(candidate, role=role, require_exists=True))
    if (int(post.st_dev), int(post.st_ino)) != opened_identity:
        os.close(descriptor)
        raise ValueError(f"{role} identity changed after open")
    return descriptor, opened_identity


def _staging_prefix(output_root: Path) -> str:
    return f".{output_root.name}.staging-"


def _quarantine_prefix(output_root: Path) -> str:
    return f".{output_root.name}.quarantine-"


def _staging_names(parent_fd: int, output_root: Path) -> tuple[str, ...]:
    prefix = _staging_prefix(output_root)
    return tuple(sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix)))


def _quarantine_names(parent_fd: int, output_root: Path) -> tuple[str, ...]:
    prefix = _quarantine_prefix(output_root)
    return tuple(sorted(name for name in os.listdir(parent_fd) if name.startswith(prefix)))


def _assert_no_staging_residue(parent_fd: int, output_root: Path) -> None:
    names = _staging_names(parent_fd, output_root)
    if names:
        raise FileExistsError(
            f"Candidate4 preflight rejects stale staging siblings: {names}"
        )


def _assert_no_prior_quarantine(parent_fd: int, output_root: Path) -> None:
    names = _quarantine_names(parent_fd, output_root)
    if names:
        raise FileExistsError(
            f"Candidate4 preflight rejects prior publication quarantine: {names}"
        )


def _create_owned_staging(
    parent_fd: int, output_root: Path
) -> tuple[str, tuple[int, int]]:
    _assert_no_staging_residue(parent_fd, output_root)
    name = f"{_staging_prefix(output_root)}{secrets.token_hex(16)}"
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("Candidate4 staging object is not a directory")
    return name, (int(metadata.st_dev), int(metadata.st_ino))


def _open_owned_staging(
    parent_fd: int,
    staging_name: str,
    expected_identity: tuple[int, int],
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise RuntimeError("Candidate4 preflight requires O_NOFOLLOW and O_DIRECTORY")
    descriptor = os.open(
        staging_name,
        os.O_RDONLY | nofollow | directory_flag,
        dir_fd=parent_fd,
    )
    metadata = os.fstat(descriptor)
    observed = (int(metadata.st_dev), int(metadata.st_ino))
    if not stat.S_ISDIR(metadata.st_mode) or observed != expected_identity:
        os.close(descriptor)
        raise RuntimeError("Candidate4 staging directory identity changed")
    return descriptor


def _write_staging_file(staging_fd: int, name: str, data: bytes) -> None:
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Candidate4 preflight payload filename is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate4 preflight requires O_NOFOLLOW")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
        0o600,
        dir_fd=staging_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate4 staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_size) != len(data):
            raise RuntimeError("Candidate4 staged payload identity/size mismatch")
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _verify_owned_staging_payloads(
    parent_fd: int,
    staging_name: str,
    expected_identity: tuple[int, int],
    payloads: Mapping[str, bytes],
) -> None:
    staging_fd = _open_owned_staging(parent_fd, staging_name, expected_identity)
    try:
        if set(os.listdir(staging_fd)) != set(payloads):
            raise RuntimeError("Candidate4 staging payload inventory changed")
        identities: set[tuple[int, int]] = set()
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate4 preflight requires O_NOFOLLOW")
        for name, expected in sorted(payloads.items()):
            descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=staging_fd)
            try:
                metadata = os.fstat(descriptor)
                identity = (int(metadata.st_dev), int(metadata.st_ino))
                if not stat.S_ISREG(metadata.st_mode) or identity in identities:
                    raise RuntimeError(
                        "Candidate4 staged payload is non-regular or hard-linked"
                    )
                identities.add(identity)
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                observed = b"".join(chunks)
                after = os.fstat(descriptor)
                if (
                    (int(after.st_dev), int(after.st_ino)) != identity
                    or int(after.st_size) != len(observed)
                    or observed != expected
                ):
                    raise RuntimeError("Candidate4 staged payload bytes changed")
            finally:
                os.close(descriptor)
    finally:
        os.close(staging_fd)


def _renameat_directory_noreplace(
    parent_fd: int, source_name: str, destination_name: str
) -> None:
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, source, parent_fd, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(parent_fd, source, parent_fd, destination, 0x00000001)
    else:
        raise RuntimeError("Candidate4 preflight has no no-replace rename primitive")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error_number,
                "Candidate4 preflight output destination raced into existence",
                destination_name,
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _quarantine_directory(
    parent_fd: int,
    source_name: str,
    output_root: Path,
    *,
    expected_identity: tuple[int, int] | None,
) -> tuple[str, tuple[int, int]]:
    before = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    before_identity = (int(before.st_dev), int(before.st_ino))
    if not stat.S_ISDIR(before.st_mode):
        raise RuntimeError("Candidate4 refuses to quarantine a non-directory")
    if expected_identity is not None and before_identity != expected_identity:
        raise RuntimeError("Candidate4 refuses to quarantine an unexpected inode")
    quarantine_name = f"{_quarantine_prefix(output_root)}{secrets.token_hex(16)}"
    _renameat_directory_noreplace(parent_fd, source_name, quarantine_name)
    quarantined = os.stat(
        quarantine_name, dir_fd=parent_fd, follow_symlinks=False
    )
    quarantined_identity = (int(quarantined.st_dev), int(quarantined.st_ino))
    if not stat.S_ISDIR(quarantined.st_mode):
        raise RuntimeError("Candidate4 quarantine is not a directory")
    if expected_identity is not None and quarantined_identity != expected_identity:
        raise RuntimeError("Candidate4 quarantine captured an unexpected inode")
    try:
        os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("Candidate4 quarantine left the source name populated")
    os.fsync(parent_fd)
    return quarantine_name, quarantined_identity


def _publish_atomically(
    output_root: Path,
    payloads: Mapping[str, bytes],
    *,
    after_stage_write: Callable[[Path], None] | None = None,
) -> None:
    output_root = _validate_output_root(output_root)
    expected_payload_names = {
        "preflight_lock.json",
        "preflight_read_ledger.json",
        "preflight_report.json",
    }
    if set(payloads) != expected_payload_names or any(
        not isinstance(data, bytes) for data in payloads.values()
    ):
        raise ValueError("Candidate4 preflight publication payload schema mismatch")
    parent_fd, parent_identity = _open_stable_directory(
        output_root.parent, role="Candidate4 preflight output parent"
    )
    staging_name = ""
    staging_identity: tuple[int, int] | None = None
    rename_returned = False
    try:
        _assert_no_staging_residue(parent_fd, output_root)
        _assert_no_prior_quarantine(parent_fd, output_root)
        try:
            os.stat(output_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "Candidate4 preflight output raced into existence before staging"
            )
        staging_name, staging_identity = _create_owned_staging(parent_fd, output_root)
        staging_fd = _open_owned_staging(parent_fd, staging_name, staging_identity)
        try:
            for name, data in sorted(payloads.items()):
                _write_staging_file(staging_fd, name, data)
            os.fsync(staging_fd)
            os.fchmod(staging_fd, 0o500)
        finally:
            os.close(staging_fd)
        if after_stage_write is not None:
            after_stage_write(output_root.parent / staging_name)
        _verify_owned_staging_payloads(
            parent_fd, staging_name, staging_identity, payloads
        )
        parent_metadata = os.fstat(parent_fd)
        if (int(parent_metadata.st_dev), int(parent_metadata.st_ino)) != parent_identity:
            raise RuntimeError("Candidate4 output parent identity changed")
        visible_parent = _reject_symlink_chain(
            output_root.parent,
            role="Candidate4 preflight output parent before publish",
            require_exists=True,
        )
        visible_parent_metadata = os.lstat(visible_parent)
        if (
            not stat.S_ISDIR(visible_parent_metadata.st_mode)
            or (
                int(visible_parent_metadata.st_dev),
                int(visible_parent_metadata.st_ino),
            )
            != parent_identity
        ):
            raise RuntimeError("Candidate4 visible output parent identity changed")
        _renameat_directory_noreplace(
            parent_fd, staging_name, output_root.name
        )
        rename_returned = True
        published_metadata = os.stat(
            output_root.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(published_metadata.st_mode)
            or (int(published_metadata.st_dev), int(published_metadata.st_ino))
            != staging_identity
        ):
            raise RuntimeError("Candidate4 published directory identity mismatch")
        visible_output = _reject_symlink_chain(
            output_root,
            role="Candidate4 published output",
            require_exists=True,
        )
        visible_output_metadata = os.lstat(visible_output)
        if (
            not stat.S_ISDIR(visible_output_metadata.st_mode)
            or (
                int(visible_output_metadata.st_dev),
                int(visible_output_metadata.st_ino),
            )
            != staging_identity
        ):
            raise RuntimeError("Candidate4 visible output identity changed")
        _verify_owned_staging_payloads(
            parent_fd, output_root.name, staging_identity, payloads
        )
        os.fsync(parent_fd)
        _assert_no_staging_residue(parent_fd, output_root)
        _assert_no_prior_quarantine(parent_fd, output_root)
    except BaseException:
        output_metadata: os.stat_result | None
        try:
            output_metadata = os.stat(
                output_root.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            output_metadata = None
        if rename_returned and output_metadata is not None:
            _quarantine_directory(
                parent_fd,
                output_root.name,
                output_root,
                expected_identity=None,
            )
        elif (
            output_metadata is not None
            and staging_identity is not None
            and (
                int(output_metadata.st_dev),
                int(output_metadata.st_ino),
            )
            == staging_identity
        ):
            _quarantine_directory(
                parent_fd,
                output_root.name,
                output_root,
                expected_identity=staging_identity,
            )
        elif staging_name and staging_identity is not None:
            try:
                staging_metadata = os.stat(
                    staging_name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                staging_metadata = None
            if staging_metadata is not None:
                observed_staging_identity = (
                    int(staging_metadata.st_dev),
                    int(staging_metadata.st_ino),
                )
                if observed_staging_identity == staging_identity:
                    _quarantine_directory(
                        parent_fd,
                        staging_name,
                        output_root,
                        expected_identity=staging_identity,
                    )
        _assert_no_staging_residue(parent_fd, output_root)
        raise
    finally:
        os.close(parent_fd)


def build_preflight_payloads(
    runtime: types.ModuleType,
    *,
    source_bindings: Mapping[str, Mapping[str, Any]],
    bootstrap_entries: list[dict[str, Any]],
) -> dict[str, bytes]:
    reader = GuardedReader(entries=[dict(row) for row in bootstrap_entries])
    train_ids, scan_by_case, train_commitment, attestation = (
        _validate_fixed_lineage(runtime, reader)
    )
    manifest_path, manifest_sha, manifest_size, attestation_checks = (
        _validate_cache_attestation(
            attestation, train_commitment=train_commitment, reader=reader
        )
    )
    reader.authorize(
        manifest_path,
        role="TRAIN_CACHE_MANIFEST",
        sha256=manifest_sha,
        size_bytes=manifest_size,
    )
    manifest_bytes = reader.read_bytes(
        manifest_path, role="TRAIN_CACHE_MANIFEST"
    )
    manifest = _decode_object(manifest_bytes, role="TRAIN_CACHE_MANIFEST")
    records = _validate_cache_manifest(
        manifest,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=train_commitment,
    )
    for record in records.values():
        reader.authorize(
            record["semantic_path"],
            role="TRAIN_V51_SEMANTIC",
            sha256=record["semantic_sha256"],
        )

    exact_pair_count = 0
    for case_id in train_ids:
        primary_record = records[(case_id, "primary")]
        replay_record = records[(case_id, "replay")]
        primary_bytes = reader.read_bytes(
            primary_record["semantic_path"],
            role="TRAIN_V51_SEMANTIC",
            case_id=case_id,
        )
        replay_bytes = reader.read_bytes(
            replay_record["semantic_path"],
            role="TRAIN_V51_SEMANTIC",
            case_id=case_id,
        )
        if primary_bytes != replay_bytes:
            raise ValueError("Candidate4 semantic primary/replay bytes differ")
        primary_result = runtime.preflight_candidate4_semantic_bytes(
            primary_bytes,
            v51_semantic_sha256=primary_record["semantic_sha256"],
        )
        replay_result = runtime.preflight_candidate4_semantic_bytes(
            replay_bytes,
            v51_semantic_sha256=replay_record["semantic_sha256"],
        )
        if runtime.canonical_bytes(primary_result) != runtime.canonical_bytes(
            replay_result
        ):
            raise ValueError("Candidate4 runtime preflight differs on replay")
        if not all(
            primary_result.get(field) is True
            for field in (
                "all_frozen_rules_exercised",
                "all_retained_predictions_canonical_byte_identical",
                "flexible_geometry_schema_validated",
                "recursive_forbidden_field_validation_exercised",
                "manifest_scan_identity_required_externally",
            )
        ):
            raise ValueError("Candidate4 runtime preflight contract failed")
        exact_pair_count += 1

    authorized = [
        {"path": str(path), "role": binding["role"]}
        for path, binding in reader.allowed.items()
    ]
    expected_authorized_paths = {
        _lexical_repository_path(path, role="expected authorized Candidate4 input")
        for path in (
            SPLIT,
            ORIGINAL_PREREG,
            ALLOWLIST,
            CANDIDATE1_TERMINAL,
            CANDIDATE2_TERMINAL,
            CANDIDATE3_TERMINAL,
            CANDIDATE3_INCIDENT,
            CACHE_ATTESTATION,
            CACHE_MANIFEST,
            *CACHE_EVIDENCE_PATHS.values(),
        )
    }
    expected_authorized_paths.update(
        record["semantic_path"] for record in records.values()
    )
    forbidden_authorized = [
        row for row in authorized if Path(row["path"]).name in FORBIDDEN_AUTHORIZED_BASENAMES
    ]
    case_entries = [row for row in reader.entries if row.get("case_id") is not None]
    if (
        forbidden_authorized
        or reader.denied_attempts
        or set(reader.allowed) != expected_authorized_paths
        or len(reader.file_identities)
        != len(expected_authorized_paths) + len(bootstrap_entries)
        or len(case_entries) != 36
        or {row["case_id"] for row in case_entries} != set(train_ids)
        or exact_pair_count != 18
    ):
        raise PermissionError("Candidate4 preflight read capability is not exact TRAIN18")

    ledger = {
        "schema_version": 1,
        "kind": LEDGER_KIND,
        "attempt_ordinal": 4,
        "status": "PASS",
        "read_count": len(reader.entries),
        "reads": reader.entries,
        "denied_read_count": len(reader.denied_attempts),
        "denied_attempts": reader.denied_attempts,
        "assertions": {
            "exact_train_case_count": len({row["case_id"] for row in case_entries}) == 18,
            "exact_train_semantic_record_count": len(case_entries) == 36,
            "pool_artifacts_opened": False,
            "label_attestation_path_resolved_or_opened": False,
            "label_projection_path_resolved_or_opened": False,
            "full_28_manifest_path_resolved_or_opened": False,
            "holdout_path_resolved_or_opened": False,
            "holdout_score_opened": False,
            "semantic_internal_scan_identity_used": False,
            "manifest_allowlist_scan_identity_only": True,
            "all_authorized_input_file_identities_unique": True,
            "fd_nofollow_fstat_reads_used": True,
        },
    }
    ledger_bytes = canonical_bytes(ledger)
    ledger_sha = sha256_bytes(ledger_bytes)
    bindings = {
        "candidate1_terminal_sha256": CANDIDATE1_TERMINAL_SHA256,
        "candidate2_terminal_sha256": CANDIDATE2_TERMINAL_SHA256,
        "candidate3_terminal_sha256": CANDIDATE3_TERMINAL_SHA256,
        "candidate3_incident_sha256": CANDIDATE3_INCIDENT_SHA256,
        "split_sha256": SPLIT_SHA256,
        "train_allowlist_sha256": ALLOWLIST_SHA256,
        "original_preregistration_sha256": ORIGINAL_PREREG_SHA256,
        "custody_cache_attestation_sha256": CACHE_ATTESTATION_SHA256,
        "custody_cache_manifest_sha256": manifest_sha,
        "v51_scan_only_manifest_sha256": V51_SCAN_ONLY_MANIFEST_SHA256,
        "custody_label_attestation_sha256_not_opened": (
            CUSTODY_LABEL_ATTESTATION_SHA256_NOT_OPENED
        ),
        **{key: dict(value) for key, value in source_bindings.items()},
    }
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "profile": runtime.PROFILE,
        "attempt_ordinal": 4,
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        "case_count": 18,
        "record_count": 36,
        "semantic_file_read_count": 36,
        "primary_replay_pair_count": exact_pair_count,
        "all_primary_replay_bytes_exact": exact_pair_count == 18,
        "all_records_flexible_v51_schema_compatible": True,
        "all_frozen_rules_exercised": True,
        "all_retained_predictions_canonical_byte_identical": True,
        "recursive_forbidden_field_validation_exercised": True,
        "scan_identity_authority": "ATTESTED_MANIFEST_JOINED_TO_SEALED_TRAIN_ALLOWLIST_ONLY",
        "semantic_internal_scan_identity_used_or_trusted": False,
        "pool_artifacts_opened": False,
        "all_authorized_input_file_identities_unique": True,
        "fd_nofollow_fstat_reads_used": True,
        "atomic_noreplace_publication_required": True,
        "label_attestation_path_resolved_or_opened": False,
        "label_projection_path_resolved_or_opened": False,
        "full_28_manifest_path_resolved_or_opened": False,
        "holdout_path_resolved_or_opened": False,
        "holdout_score_opened": False,
        "candidate_metric_computed": False,
        "model_fit_performed": False,
        "deletion_rule_selected": False,
        "candidate4_preregistration_authorized": False,
        "candidate4_fit_authorized": False,
        "attestation_checks": attestation_checks,
        "read_ledger_sha256": ledger_sha,
        "bindings": bindings,
    }
    _assert_aggregate_safe(report)
    report_bytes = canonical_bytes(report)
    report_sha = sha256_bytes(report_bytes)
    lock = {
        "schema_version": 1,
        "kind": LOCK_KIND,
        "attempt_ordinal": 4,
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        "candidate4_preregistration_authorized": False,
        "candidate4_fit_authorized": False,
        "candidate4_holdout_execution_authorized": False,
        "holdout_score_opened": False,
        "post_preflight_source_or_policy_mutation": "INVALIDATES_THIS_LOCK",
        "preflight_report_sha256": report_sha,
        "preflight_read_ledger_sha256": ledger_sha,
        "bindings": bindings,
    }
    _assert_aggregate_safe(lock)
    return {
        "preflight_read_ledger.json": ledger_bytes,
        "preflight_report.json": report_bytes,
        "preflight_lock.json": canonical_bytes(lock),
    }


def run(
    *,
    runtime_sha256: str,
    preflight_sha256: str,
    runtime_tests_sha256: str,
    preflight_tests_sha256: str,
) -> dict[str, Any]:
    output = _validate_output_root(OUTPUT_ROOT)
    runtime, source_bindings, bootstrap_entries = bootstrap_verified_sources(
        runtime_sha256=runtime_sha256,
        preflight_sha256=preflight_sha256,
        runtime_tests_sha256=runtime_tests_sha256,
        preflight_tests_sha256=preflight_tests_sha256,
    )
    payloads = build_preflight_payloads(
        runtime,
        source_bindings=source_bindings,
        bootstrap_entries=bootstrap_entries,
    )
    _publish_atomically(output, payloads)
    return {
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        "case_count": 18,
        "record_count": 36,
        "label_projection_path_resolved_or_opened": False,
        "preflight_report_sha256": sha256_bytes(payloads["preflight_report.json"]),
        "preflight_read_ledger_sha256": sha256_bytes(
            payloads["preflight_read_ledger.json"]
        ),
        "preflight_lock_sha256": sha256_bytes(payloads["preflight_lock.json"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--runtime-tests-sha256", required=True)
    parser.add_argument("--preflight-tests-sha256", required=True)
    parser.add_argument("--source-check-only", action="store_true")
    args = parser.parse_args()
    runtime, source_bindings, bootstrap_entries = bootstrap_verified_sources(
        runtime_sha256=args.runtime_sha256,
        preflight_sha256=args.preflight_sha256,
        runtime_tests_sha256=args.runtime_tests_sha256,
        preflight_tests_sha256=args.preflight_tests_sha256,
    )
    if args.source_check_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_BYTES_VERIFIED_ONLY",
                    "profile": runtime.PROFILE,
                    "source_binding_count": len(source_bindings),
                    "cache_or_label_artifact_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = _validate_output_root(OUTPUT_ROOT)
    payloads = build_preflight_payloads(
        runtime,
        source_bindings=source_bindings,
        bootstrap_entries=bootstrap_entries,
    )
    _publish_atomically(output, payloads)
    print(
        json.dumps(
            {
                "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
                "case_count": 18,
                "record_count": 36,
                "label_projection_path_resolved_or_opened": False,
                "preflight_report_sha256": sha256_bytes(
                    payloads["preflight_report.json"]
                ),
                "preflight_read_ledger_sha256": sha256_bytes(
                    payloads["preflight_read_ledger.json"]
                ),
                "preflight_lock_sha256": sha256_bytes(
                    payloads["preflight_lock.json"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
