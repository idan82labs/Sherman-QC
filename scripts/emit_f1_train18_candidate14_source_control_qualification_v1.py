#!/usr/bin/env python3
"""Emit the exact Candidate14 repeatable qualification record to stdout only.

This launcher has no publication mode and performs no filesystem writes.  Its
only consuming mode delegates to the exact frozen Candidate14 runner's public
source bootstrap, repeatable source-control qualification, and canonical
qualification builder.  It rejects any manifest or pool capability/read state
and emits bytes only when the authorized canonical record SHA matches exactly.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EMITTER_SOURCE = (
    ROOT / "scripts/emit_f1_train18_candidate14_source_control_qualification_v1.py"
)
EMITTER_TESTS_SOURCE = (
    ROOT / "tests/test_emit_f1_train18_candidate14_source_control_qualification_v1.py"
)
RUNNER_SOURCE = (
    ROOT
    / "scripts/run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"
)
RUNNER_TESTS_SOURCE = (
    ROOT
    / "tests/test_run_f1_train18_candidate14_full_pool_adapter_preflight_once_v1.py"
)

RUNNER_SHA256 = "50eeb6959a18e60d55c4c1159a4a56ed879b4d74e1f4d09e147d60c8bb5e735c"
RUNNER_SIZE_BYTES = 216217
RUNNER_TESTS_SHA256 = "d6f62b6d6d301df8204887ce64a5778780981cf52ec9050e7c6a7651b2588695"
RUNNER_TESTS_SIZE_BYTES = 103948
EXPECTED_QUALIFICATION_SHA256 = (
    "667d0cc9527e86e21c3d6b537d3f7b06f29a5f33995239f07644853a20a5582a"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "bb8dc1ae00bac889702adf7902ac9d1a32ab28d17874f7751b84965203909392"
)
EXPECTED_CONTROL_BINDING_SHA256 = (
    "42ff3117cba8c9bd1f0030bf0fcb93947e20805145ccac4d45dd96e7f16db777"
)
EXPECTED_STATIC_QUALIFICATION_COMMITMENTS = types.MappingProxyType(
    {
        "consumption_anchor_namespace_commitment_sha256": (
            "ade77c5fce0496dbb3bd72f32b2c6e509106b1edc5705aaa1718570886b0f856"
        ),
        "control_raw_fd_opened_commitment_sha256": (
            "a853e58143aeadceb44564bdb93cad11c20a5cfc938ad7879964c529fe317d27"
        ),
        "control_read_started_commitment_sha256": (
            "8ebb26d6237d94cd31192df9b504698ff77ea540ad791a3138759c71c86bbd06"
        ),
        "control_verified_inode_commitment_sha256": (
            "fc0cfa35aabbec523eb465a41400cc660d17b2026d34e108f9e7a52ea34af68d"
        ),
        "manifest_binding_commitment_sha256": (
            "9f71e1829d11b19af0bb3648633cc7407176f49d0bdb702f4f116b6df1b20d18"
        ),
        "output_namespace_commitment_sha256": (
            "debc700938a083697c5bc3ef6039aea98cabaa9985c223f525d2fab065ff47f6"
        ),
        "verified_exec_bootstrap_commitment_sha256": (
            "005cf27767d196763666552d223cfb0bbb74cc2265033ef1c9696ef1cfb530df"
        ),
    }
)
EXPECTED_CONTROL_PURPOSES = (
    "SEALED_SPLIT",
    "ORIGINAL_PREREGISTRATION",
    "TRAIN_ALLOWLIST",
    "CANDIDATE13_TERMINAL_DISPOSITION",
    "TRAIN_CACHE_ATTESTATION",
    "CACHE_CUSTODY_ACCESS_EVIDENCE",
    "CACHE_CUSTODY_BUILDER",
    "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
    "CACHE_CUSTODY_VALIDATOR",
)

EXPECTED_CONTROL_PATHS = (
    ROOT / "research/f1_aug30_retrospective_pseudo_unseen_split_v1.json",
    ROOT / "research/f1_aug30_retrospective_pseudo_unseen_prereg_v1.json",
    ROOT / "research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json",
    ROOT / "research/f1_aug30_train18_candidate13_terminal_disposition_v1.json",
    ROOT
    / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest_attestation.json",
    ROOT
    / "output/research/f1-aug30-train-only-custody-v2/cache/access_evidence.json",
    ROOT / "scripts/build_f1_aug30_train_only_custody_v1.py",
    ROOT
    / "output/research/f1-aug30-train-only-custody-v2/independent_validation_report.json",
    ROOT / "scripts/audit_f1_aug30_train_only_custody_v1.py",
)

CONTROLLED_SOURCE_PATHS = (
    RUNNER_SOURCE,
    RUNNER_TESTS_SOURCE,
    ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py",
    ROOT / "tests/test_f1_train18_candidate14_darwin_resource_capability_v1.py",
    ROOT / "scripts/run_f1_train18_candidate14_pool_adapter_worker_v1.py",
    ROOT / "tests/test_run_f1_train18_candidate14_pool_adapter_worker_v1.py",
    ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
    ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
    ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
    ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
    ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
    ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
    ROOT / "backend/f1_line_object_clustering.py",
)

EXPECTED_SOURCE_SPECS = (
    (
        "builder",
        ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
        "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2",
        57613,
    ),
    (
        "c13_core",
        ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
        "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
        65344,
    ),
    (
        "c13_core_tests",
        ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
        "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
        43387,
    ),
    (
        "c14_resource",
        ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py",
        "c82fb204b61f99ce7fb7eab8b5ed15f9003be893c004dac37f347bbdbfe4c079",
        15629,
    ),
    (
        "c14_resource_tests",
        ROOT / "tests/test_f1_train18_candidate14_darwin_resource_capability_v1.py",
        "ac8843ed2fa3d83fc291c5f2ecb0c16fded3320d4484195658551366db8caa51",
        12642,
    ),
    (
        "c14_worker",
        ROOT / "scripts/run_f1_train18_candidate14_pool_adapter_worker_v1.py",
        "58e68be857aa81651fff10d423a43b4d7fe745f45dafe718a34dd68ec49bc137",
        50687,
    ),
    (
        "c14_worker_tests",
        ROOT / "tests/test_run_f1_train18_candidate14_pool_adapter_worker_v1.py",
        "6842932c572f448b1f6a087dee44965cdf5a935ba9991787ba808aea951d515a",
        20844,
    ),
    (
        "c6_hypothesis",
        ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
        "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
        53837,
    ),
    (
        "c9_adapter",
        ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
        22354,
    ),
    (
        "c9_adapter_tests",
        ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
        "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
        16598,
    ),
    (
        "clusterer",
        ROOT / "backend/f1_line_object_clustering.py",
        "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c",
        6817,
    ),
    ("runner", RUNNER_SOURCE, RUNNER_SHA256, RUNNER_SIZE_BYTES),
    (
        "runner_tests",
        RUNNER_TESTS_SOURCE,
        RUNNER_TESTS_SHA256,
        RUNNER_TESTS_SIZE_BYTES,
    ),
)

QUALIFICATION_KEYS = frozenset(
    {
        "adapter_or_hypothesis_modified",
        "cache_manifest_opened",
        "check_code",
        "combined_cpu_limit_seconds",
        "consumption_anchor_namespace_commitment_sha256",
        "control_binding_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_read_inventory_commitment_sha256",
        "control_read_started_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "engineering_attempt_id",
        "holdout_label_evaluator_metric_fit_full28_opened_or_started",
        "kind",
        "manifest_binding_commitment_sha256",
        "memory_ceiling_bytes",
        "memory_enforcement_is_hard_rlimit_as",
        "memory_enforcement_kind",
        "output_namespace_commitment_sha256",
        "poll_interval_seconds",
        "repeatable",
        "resource_capability_result_commitment_sha256",
        "sampling_gap_terminal",
        "sampling_max_gap_seconds",
        "sampling_overshoot_between_polls_possible",
        "schema_version",
        "scientific_hypothesis_consumed",
        "scientific_hypothesis_id",
        "source_inventory_commitment_sha256",
        "status",
        "terminal_publication_reserved_outside_compute_resource_envelope",
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available",
        "verified_exec_bootstrap_commitment_sha256",
        "wall_timeout_seconds",
        "worker_cpu_hard_limit_seconds",
        "worker_cpu_soft_limit_seconds",
        "worker_source_inventory_commitment_sha256",
        "worker_source_result_commitment_sha256",
    }
)

QUALIFICATION_COMMITMENT_KEYS = frozenset(
    {
        "consumption_anchor_namespace_commitment_sha256",
        "control_binding_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_read_inventory_commitment_sha256",
        "control_read_started_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "output_namespace_commitment_sha256",
        "resource_capability_result_commitment_sha256",
        "source_inventory_commitment_sha256",
        "verified_exec_bootstrap_commitment_sha256",
        "worker_source_inventory_commitment_sha256",
        "worker_source_result_commitment_sha256",
    }
)

PRIVATE_RESOURCE_MODULE_NAME = "_candidate14_parent_verified_resource_core"
MAX_BOUND_SOURCE_BYTES = 4 * 1024 * 1024

VERIFIED_EMITTER_BOOTSTRAP_CODE = r'''import hashlib
import os
import stat
import sys
import types
from types import MappingProxyType

_ROOT = "/Users/idantal/ShermanQC/FullProject"
_EXPECTED = _ROOT + "/scripts/emit_f1_train18_candidate14_source_control_qualification_v1.py"
_FAILURE = b'{"check_code":"C14_QUALIFICATION_EMIT_REJECTED","status":"FAIL_CLOSED"}\n'


def _identity(value):
    return int(value.st_dev), int(value.st_ino)


def _read_verified(path_text, expected_sha, expected_size):
    if type(path_text) is not str or path_text != _EXPECTED:
        raise PermissionError("Candidate14 emitter bootstrap path mismatch")
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or type(expected_size) is not int
        or not 0 < expected_size <= 4 * 1024 * 1024
    ):
        raise ValueError("Candidate14 emitter bootstrap binding mismatch")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate14 emitter bootstrap requires nofollow directories")
    relative_parts = os.path.relpath(path_text, _ROOT).split(os.path.sep)
    if relative_parts[:-1] != ["scripts"] or len(relative_parts) != 2:
        raise PermissionError("Candidate14 emitter bootstrap relative path mismatch")
    walk_fd = os.open(os.path.sep, os.O_RDONLY | nofollow | directory)
    parent_fd = None
    descriptor = None
    failure = None
    data = None
    try:
        for component in _ROOT.split(os.path.sep)[1:]:
            next_fd = os.open(
                component, os.O_RDONLY | nofollow | directory, dir_fd=walk_fd
            )
            opened_directory = os.fstat(next_fd)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(next_fd)
                raise PermissionError("Candidate14 emitter bootstrap directory mismatch")
            os.close(walk_fd)
            walk_fd = next_fd
        for component in relative_parts[:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | nofollow | directory, dir_fd=walk_fd
            )
            opened_directory = os.fstat(next_fd)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(next_fd)
                raise PermissionError("Candidate14 emitter bootstrap directory mismatch")
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
            raise PermissionError("Candidate14 emitter bootstrap leaf mismatch")
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
            raise PermissionError("Candidate14 emitter bootstrap open mismatch")
        chunks = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise PermissionError("Candidate14 emitter bootstrap source truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise PermissionError("Candidate14 emitter bootstrap source grew")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
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
            raise PermissionError("Candidate14 emitter bootstrap source changed")
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
        raise PermissionError("Candidate14 emitter bootstrap source hash mismatch")
    return data


def _write_all(descriptor, payload):
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Candidate14 emitter bootstrap write failed")
        view = view[written:]


def _launch():
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError("Candidate14 emitter bootstrap requires isolated startup")
    arguments = list(sys.argv[1:])
    if len(arguments) != 12 or not all(type(value) is str for value in arguments):
        raise ValueError("Candidate14 emitter bootstrap argument mismatch")
    path_text, expected_sha, size_text, *application_arguments = arguments
    if not size_text.isdecimal() or str(int(size_text)) != size_text:
        raise ValueError("Candidate14 emitter bootstrap size mismatch")
    expected_size = int(size_text)
    data = _read_verified(path_text, expected_sha, expected_size)
    private_name = "_candidate14_qualification_emitter_" + expected_sha
    if private_name in sys.modules:
        raise PermissionError("Candidate14 emitter bootstrap module already exists")
    outer = sys.modules.get("__main__")
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate14 emitter bootstrap outer module mismatch")
    outer_namespace = object.__getattribute__(outer, "__dict__")
    if outer_namespace.get("_EMITTER_OUTER_TOKEN") is not None:
        raise PermissionError("Candidate14 emitter bootstrap is not reentrant")
    token = object()
    module = types.ModuleType(private_name)
    module.__file__ = path_text
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_EMITTER_VERIFIED_BINDING"] = MappingProxyType(
        {
            "module_name": private_name,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    )
    module.__dict__["_EMITTER_LAUNCH_TOKEN"] = token
    module.__dict__["_EMITTER_OUTER_MODULE"] = outer
    outer_namespace["_EMITTER_OUTER_TOKEN"] = token
    sys.modules[private_name] = module
    output = None
    try:
        exec(compile(data, path_text, "exec", dont_inherit=True), module.__dict__)
        private_main = object.__getattribute__(module, "__dict__").get("_private_main")
        if (
            type(private_main) is not types.FunctionType
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != path_text
        ):
            raise PermissionError("Candidate14 emitter bootstrap entry mismatch")
        output = private_main(application_arguments)
        if type(output) is not bytes or not output.endswith(b"\n"):
            raise PermissionError("Candidate14 emitter bootstrap output mismatch")
    finally:
        if sys.modules.get(private_name) is module:
            sys.modules.pop(private_name)
        else:
            raise PermissionError("Candidate14 emitter private module changed")
        if outer_namespace.get("_EMITTER_OUTER_TOKEN") is token:
            del outer_namespace["_EMITTER_OUTER_TOKEN"]
        else:
            raise PermissionError("Candidate14 emitter outer token changed")
    _write_all(1, output)
    return 0


try:
    _CODE = _launch()
except BaseException:
    _write_all(2, _FAILURE)
    _CODE = 1
raise SystemExit(_CODE)
'''

_IMPORT_HOOK_SEAL = (
    builtins.__import__,
    tuple(sys.meta_path),
    tuple(sys.path_hooks),
)
_EMITTER_VERIFIED_BINDING = globals().get("_EMITTER_VERIFIED_BINDING")
_EMITTER_LAUNCH_TOKEN = globals().get("_EMITTER_LAUNCH_TOKEN")
_EMITTER_OUTER_MODULE = globals().get("_EMITTER_OUTER_MODULE")
_EMITTER_OUTER_TOKEN = globals().get("_EMITTER_OUTER_TOKEN")


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
    if type(data) is not bytes:
        raise TypeError("Candidate14 qualification emitter hashes exact bytes only")
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _strict_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _lexical(path: Path, *, role: str) -> Path:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ValueError(f"{role} contains parent traversal")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    root = Path(os.path.abspath(os.fspath(ROOT)))
    if os.path.commonpath((os.fspath(root), os.fspath(candidate))) != os.fspath(root):
        raise ValueError(f"{role} leaves repository")
    return candidate


def _reject_symlinks(path: Path, *, role: str) -> Path:
    candidate = _lexical(path, role=role)
    root = _lexical(ROOT, role="repository root")
    relative = candidate.relative_to(root)
    cursor = root
    if stat.S_ISLNK(os.lstat(cursor).st_mode):
        raise ValueError("Candidate14 qualification repository root is a symlink")
    for component in relative.parts:
        cursor = cursor / component
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} traverses a symlink")
    return candidate


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlinks(path, role=role)
    descriptor = os.open(
        candidate,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        opened = os.fstat(descriptor)
        visible = os.lstat(candidate)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(opened) != _identity(visible)
        ):
            raise ValueError(f"{role} directory identity changed")
        return descriptor, _identity(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_once(
    path: Path, *, role: str, expected_size: int
) -> tuple[bytes, tuple[int, int]]:
    bounded_size = _strict_positive_int(expected_size, name=f"{role} size")
    if bounded_size > MAX_BOUND_SOURCE_BYTES:
        raise ValueError(f"{role} exceeds the emitter source-size limit")
    candidate = _reject_symlinks(path, role=role)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        before = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or int(before.st_size) != bounded_size
        ):
            raise ValueError(f"{role} must be one regular single-link file")
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
            or int(opened.st_size) != bounded_size
        ):
            raise ValueError(f"{role} changed during open")
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
        after = os.fstat(descriptor)
        visible = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or _identity(after) != identity
            or _identity(visible) != identity
            or int(after.st_size) != bounded_size
            or len(data) != bounded_size
            or _identity(os.lstat(candidate.parent)) != parent_identity
        ):
            raise ValueError(f"{role} changed during read")
        return data, identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _read_exact(path: Path, *, role: str, sha256: str, size_bytes: int) -> bytes:
    bounded_size = _strict_positive_int(size_bytes, name=f"{role} size")
    data, _file_identity = _read_regular_file_once(
        path, role=role, expected_size=bounded_size
    )
    if (
        len(data) != bounded_size
        or sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA")
    ):
        raise PermissionError(f"{role} exact byte binding mismatch")
    return data


def _verified_emitter_binding() -> Mapping[str, Any]:
    binding = _EMITTER_VERIFIED_BINDING
    module = sys.modules.get(__name__)
    outer = _EMITTER_OUTER_MODULE
    namespace = (
        object.__getattribute__(outer, "__dict__")
        if type(outer) is types.ModuleType
        else {}
    )
    if (
        type(binding) is not types.MappingProxyType
        or set(binding) != {"module_name", "sha256", "size_bytes"}
        or _EMITTER_LAUNCH_TOKEN is None
        or type(outer) is not types.ModuleType
        or namespace.get("_EMITTER_OUTER_TOKEN") is not _EMITTER_LAUNCH_TOKEN
        or type(module) is not types.ModuleType
        or module is not sys.modules.get(str(binding.get("module_name", "")))
        or __name__ != binding.get("module_name")
        or type(binding.get("sha256")) is not str
        or len(str(binding.get("sha256"))) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(binding.get("sha256"))
        )
        or __name__
        != f"_candidate14_qualification_emitter_{binding.get('sha256', '')}"
        or type(binding.get("size_bytes")) is not int
        or int(binding.get("size_bytes")) <= 0
    ):
        raise PermissionError(
            "Candidate14 qualification emitter is not verified private code"
        )
    return binding


def _validate_import_hooks() -> None:
    if (
        builtins.__import__ is not _IMPORT_HOOK_SEAL[0]
        or tuple(sys.meta_path) != _IMPORT_HOOK_SEAL[1]
        or tuple(sys.path_hooks) != _IMPORT_HOOK_SEAL[2]
    ):
        raise RuntimeError("Candidate14 qualification emitter rejects import-hook drift")


def _reject_preloaded_sources(paths: Sequence[Path]) -> None:
    _validate_import_hooks()
    current = sys.modules.get(__name__)
    outer = _EMITTER_OUTER_MODULE
    targets = {_reject_symlinks(path, role="controlled source") for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in targets}
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current or module is outer:
            continue
        if not isinstance(module, types.ModuleType):
            continue
        if type(module) is not types.ModuleType:
            raise RuntimeError(
                f"Candidate14 qualification emitter rejects module subclass: {name}"
            )
        namespace = object.__getattribute__(module, "__dict__")
        loaded = namespace.get("__file__")
        if type(loaded) is not str or not loaded:
            continue
        try:
            candidate = Path(loaded).resolve(strict=True)
            identity = _identity(os.stat(candidate, follow_symlinks=True))
        except OSError:
            continue
        if candidate in targets or identity in identities:
            raise RuntimeError(
                f"Candidate14 qualification emitter rejects source alias: {name}"
            )


def _load_verified_runner() -> tuple[types.ModuleType, object]:
    _verified_emitter_binding()
    _reject_preloaded_sources(CONTROLLED_SOURCE_PATHS)
    runner_bytes = _read_exact(
        RUNNER_SOURCE,
        role="Candidate14 frozen runner",
        sha256=RUNNER_SHA256,
        size_bytes=RUNNER_SIZE_BYTES,
    )
    _read_exact(
        RUNNER_TESTS_SOURCE,
        role="Candidate14 frozen runner tests",
        sha256=RUNNER_TESTS_SHA256,
        size_bytes=RUNNER_TESTS_SIZE_BYTES,
    )
    name = f"_candidate14_verified_runner_{RUNNER_SHA256}"
    if name in sys.modules:
        raise PermissionError("Candidate14 frozen runner is already loaded")
    outer = sys.modules.get(__name__)
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate14 qualification emitter module is unavailable")
    namespace = object.__getattribute__(outer, "__dict__")
    if namespace.get("_C14_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate14 runner launch is not reentrant")
    root_fd, root_identity = _open_stable_directory(
        ROOT, role="Candidate14 verified repository root"
    )
    os.close(root_fd)
    token = object()
    module = types.ModuleType(name)
    module.__file__ = str(RUNNER_SOURCE)
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_C14_VERIFIED_RUNNER_BINDING"] = types.MappingProxyType(
        {
            "module_name": name,
            "sha256": RUNNER_SHA256,
            "size_bytes": RUNNER_SIZE_BYTES,
        }
    )
    module.__dict__["_C14_VERIFIED_RUNNER_LAUNCH_TOKEN"] = token
    module.__dict__["_C14_VERIFIED_RUNNER_OUTER_MODULE"] = outer
    module.__dict__["_C14_VERIFIED_ROOT"] = str(ROOT)
    module.__dict__["_C14_VERIFIED_ROOT_IDENTITY"] = root_identity
    namespace["_C14_OUTER_LAUNCH_TOKEN"] = token
    sys.modules[name] = module
    try:
        exec(
            compile(runner_bytes, str(RUNNER_SOURCE), "exec", dont_inherit=True),
            module.__dict__,
        )
        for function_name in (
            "bootstrap_verified_sources",
            "qualify_source_controls",
            "build_qualification_record",
        ):
            function = module.__dict__.get(function_name)
            if (
                type(function) is not types.FunctionType
                or function.__module__ != name
                or function.__code__.co_filename != str(RUNNER_SOURCE)
            ):
                raise PermissionError(
                    "Candidate14 qualification runner public function mismatch"
                )
        return module, token
    except BaseException:
        sys.modules.pop(name, None)
        if namespace.get("_C14_OUTER_LAUNCH_TOKEN") is token:
            namespace.pop("_C14_OUTER_LAUNCH_TOKEN", None)
        raise


def _release_verified_runner(
    module: types.ModuleType,
    token: object,
    resource_module: types.ModuleType | None,
) -> None:
    outer = sys.modules.get(__name__)
    namespace = (
        object.__getattribute__(outer, "__dict__")
        if type(outer) is types.ModuleType
        else {}
    )
    failure: BaseException | None = None
    try:
        if sys.modules.get(module.__name__) is not module:
            raise PermissionError("Candidate14 private runner identity changed")
        sys.modules.pop(module.__name__)
        if resource_module is not None:
            if (
                type(resource_module) is not types.ModuleType
                or resource_module.__name__ != PRIVATE_RESOURCE_MODULE_NAME
                or sys.modules.get(PRIVATE_RESOURCE_MODULE_NAME) is not resource_module
            ):
                raise PermissionError("Candidate14 private resource identity changed")
            sys.modules.pop(PRIVATE_RESOURCE_MODULE_NAME)
    except BaseException as error:
        failure = error
    finally:
        if namespace.get("_C14_OUTER_LAUNCH_TOKEN") is token:
            namespace.pop("_C14_OUTER_LAUNCH_TOKEN", None)
        elif failure is None:
            failure = PermissionError("Candidate14 runner launch token changed")
    if failure is not None:
        raise failure


def _clear_runner_issuance_state(
    runner: types.Module,
    context: Any,
    *,
    expected_qualifications: int | None,
) -> None:
    namespace = object.__getattribute__(runner, "__dict__")
    context_registry = namespace.get("_ISSUED_CONTEXTS")
    qualification_registry = namespace.get("_ISSUED_QUALIFICATIONS")
    if (
        type(context_registry) is not dict
        or len(context_registry) != 1
        or type(context_registry.get(id(context))) is not tuple
        or len(context_registry[id(context)]) != 2
        or context_registry[id(context)][0] is not context
        or context_registry[id(context)][1]
        is not object.__getattribute__(context, "_token")
        or type(qualification_registry) is not dict
        or (
            expected_qualifications is not None
            and len(qualification_registry) != expected_qualifications
        )
    ):
        raise PermissionError("Candidate14 runner issuance state is not exact")
    if expected_qualifications == 1:
        issued_qualification = next(iter(qualification_registry.values()))
        if (
            type(issued_qualification) is not tuple
            or len(issued_qualification) != 5
            or type(issued_qualification[0]) is not runner.ControlQualification
            or object.__getattribute__(issued_qualification[0], "_context_token")
            is not object.__getattribute__(context, "_token")
        ):
            raise PermissionError("Candidate14 qualification issuance state changed")
    for name in (
        "_ISSUED_AUTHORIZATIONS",
        "_ISSUED_CONSUMPTIONS",
        "_ISSUED_BASELINES",
    ):
        registry = namespace.get(name)
        if type(registry) is not dict or registry:
            raise PermissionError("Candidate14 emitter reached forbidden runner state")
    quarantine = namespace.get("_UNREAPED_PROCESS_QUARANTINE")
    if type(quarantine) is not list or quarantine:
        raise PermissionError("Candidate14 emitter left worker quarantine state")
    context_registry.clear()
    qualification_registry.clear()


def _path_within(path: Path, root: Path) -> bool:
    candidate = _lexical(path, role="qualified reader path")
    boundary = _lexical(root, role="forbidden cache root")
    return os.path.commonpath((os.fspath(candidate), os.fspath(boundary))) == os.fspath(
        boundary
    )


def _validate_source_context(runner: Any, context: Any) -> None:
    if type(context) is not runner.BootstrapContext:
        raise PermissionError("Candidate14 emitter source context mismatch")
    entries = context.source_entries
    if (
        type(entries) is not tuple
        or len(entries) != len(CONTROLLED_SOURCE_PATHS)
        or context.source_commitment_sha256 != EXPECTED_SOURCE_INVENTORY_SHA256
    ):
        raise PermissionError("Candidate14 emitter source context mismatch")
    paths: list[Path] = []
    purposes: list[str] = []
    identities: list[tuple[int, int]] = []
    expected_bindings: dict[str, dict[str, Any]] = {}
    for entry, (name, expected_path, expected_sha, expected_size) in zip(
        entries, EXPECTED_SOURCE_SPECS
    ):
        expected_purpose = f"CANDIDATE14_SOURCE_{name.upper()}"
        expected_relative = str(expected_path.relative_to(ROOT))
        if (
            type(entry) is not types.MappingProxyType
            or set(entry)
            != {"device_id", "inode", "path", "purpose", "sha256", "size_bytes"}
            or type(entry["device_id"]) is not int
            or type(entry["inode"]) is not int
            or type(entry["path"]) is not str
            or type(entry["purpose"]) is not str
            or entry["purpose"] != expected_purpose
            or entry["path"] != expected_relative
            or entry["sha256"] != expected_sha
            or entry["size_bytes"] != expected_size
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] <= 0
        ):
            raise PermissionError("Candidate14 emitter source entry mismatch")
        _require_sha(entry["sha256"], name="Candidate14 source entry SHA")
        paths.append(_lexical(ROOT / entry["path"], role=entry["purpose"]))
        purposes.append(entry["purpose"])
        identities.append((entry["device_id"], entry["inode"]))
        expected_bindings[name] = {
            "path": expected_relative,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    source_bindings = context.source_bindings
    if (
        tuple(path for _name, path, _sha, _size in EXPECTED_SOURCE_SPECS)
        != tuple(paths)
        or set(CONTROLLED_SOURCE_PATHS) != set(paths)
        or len(paths) != len(set(paths))
        or len(purposes) != len(set(purposes))
        or purposes != sorted(purposes)
        or len(identities) != len(set(identities))
        or type(source_bindings) is not types.MappingProxyType
        or tuple(source_bindings) != tuple(name for name, *_rest in EXPECTED_SOURCE_SPECS)
        or any(
            type(source_bindings[name]) is not types.MappingProxyType
            or dict(source_bindings[name]) != expected
            for name, expected in expected_bindings.items()
        )
        or context.source_commitment_sha256
        != sha256_bytes(
            canonical_bytes(
                {
                    name: expected_bindings[name]
                    for name in sorted(expected_bindings)
                }
            )
        )
        or type(context.resource_core) is not types.ModuleType
        or context.resource_core.__name__ != PRIVATE_RESOURCE_MODULE_NAME
        or context.resource_core.__file__
        != str(
            ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py"
        )
        or sys.modules.get(PRIVATE_RESOURCE_MODULE_NAME) is not context.resource_core
        or type(context._resource_functions) is not tuple
        or context._resource_functions
        != (
            context.resource_core.DarwinRusageSampler,
            context.resource_core.SampledDarwinCeiling,
            context.resource_core.run_repeatable_source_only_capability_probe,
        )
    ):
        raise PermissionError("Candidate14 emitter source inventory mismatch")


def _validate_post_qualification_boundary(
    runner: Any, context: Any, qualification: Any
) -> None:
    _validate_source_context(runner, context)
    if type(qualification) is not runner.ControlQualification:
        raise PermissionError("Candidate14 qualification type mismatch")
    reader = qualification.reader
    if type(reader) is not runner.GuardedReader:
        raise PermissionError("Candidate14 qualification reader type mismatch")
    manifest = _lexical(runner.CACHE_MANIFEST, role="Candidate14 manifest")
    cache_root = _lexical(runner.ALLOWED_CACHE_ROOT, role="Candidate14 pool root")
    source_count = len(context.source_entries)
    source_paths = [
        _lexical(ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in context.source_entries
    ]
    control_paths = [_lexical(path, role="fixed control") for path in EXPECTED_CONTROL_PATHS]
    purposes = [str(entry["purpose"]) for entry in reader.entries]
    expected_prefix = [str(entry["purpose"]) for entry in context.source_entries]
    expected_prefix.extend(EXPECTED_CONTROL_PURPOSES)
    entry_paths: list[Path] = []
    entry_identities: list[tuple[int, int]] = []
    for entry in reader.entries:
        if (
            type(entry) is not dict
            or set(entry)
            != {"device_id", "inode", "path", "purpose", "sha256", "size_bytes"}
            or type(entry["device_id"]) is not int
            or type(entry["inode"]) is not int
            or type(entry["path"]) is not str
            or type(entry["purpose"]) is not str
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] <= 0
        ):
            raise PermissionError("Candidate14 qualification read entry is invalid")
        _require_sha(entry["sha256"], name="Candidate14 qualification read SHA")
        entry_paths.append(_lexical(ROOT / entry["path"], role=entry["purpose"]))
        entry_identities.append((entry["device_id"], entry["inode"]))
    expected_paths = source_paths + control_paths
    control_path_set = set(control_paths)
    control_entry_projection = [dict(entry) for entry in reader.entries[source_count:]]
    expected_control_inventory_commitment = sha256_bytes(
        canonical_bytes(
            {
                "domain": "CANDIDATE14_CONTROL_READ_INVENTORY_V1",
                "reads": control_entry_projection,
            }
        )
    )
    if (
        reader.entries[:source_count]
        != [dict(entry) for entry in context.source_entries]
        or purposes != expected_prefix
        or len(reader.entries) != source_count + len(EXPECTED_CONTROL_PURPOSES)
        or [str(entry["path"]) for entry in reader.entries[source_count:]]
        != [str(path.relative_to(ROOT)) for path in EXPECTED_CONTROL_PATHS]
        or entry_paths != expected_paths
        or reader.raw_fd_opened_purposes != purposes
        or reader.verified_inode_purposes != purposes
        or reader.read_started_purposes != purposes
        or reader.artifact_inventory is not None
        or reader.denied_attempts
        or reader.allowed
        or reader.consumed_paths != control_path_set
        or reader.raw_fd_opened_paths != control_path_set
        or reader.verified_inode_paths != control_path_set
        or reader.read_started_paths != control_path_set
        or len(entry_paths) != len(set(entry_paths))
        or len(entry_identities) != len(set(entry_identities))
        or len(reader.identities) != len(entry_identities)
        or any(
            reader.identities.get(identity) != path
            for identity, path in zip(entry_identities, entry_paths)
        )
        or qualification.control_binding_commitment_sha256
        != EXPECTED_CONTROL_BINDING_SHA256
        or qualification.control_read_inventory_commitment_sha256
        != expected_control_inventory_commitment
        or qualification.control_raw_fd_opened_commitment_sha256
        != EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_raw_fd_opened_commitment_sha256"
        ]
        or qualification.control_verified_inode_commitment_sha256
        != EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_verified_inode_commitment_sha256"
        ]
        or qualification.control_read_started_commitment_sha256
        != EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_read_started_commitment_sha256"
        ]
        or qualification.manifest_path
        != _lexical(runner.CACHE_MANIFEST, role="Candidate14 manifest binding")
        or qualification.manifest_sha256 != runner.CACHE_MANIFEST_SHA256
        or qualification.manifest_size_bytes != runner.CACHE_MANIFEST_SIZE_BYTES
        or type(qualification.train_ids) is not tuple
        or len(qualification.train_ids) != runner.PAIR_COUNT
        or len(set(qualification.train_ids)) != runner.PAIR_COUNT
        or any(type(case_id) is not str or not case_id for case_id in qualification.train_ids)
        or type(qualification.scan_by_case) is not types.MappingProxyType
        or tuple(qualification.scan_by_case) != qualification.train_ids
        or any(
            _require_sha(qualification.scan_by_case[case_id], name="TRAIN scan SHA")
            != qualification.scan_by_case[case_id]
            for case_id in qualification.train_ids
        )
        or qualification.train_commitment_sha256
        != sha256_bytes(
            canonical_bytes(
                [
                    {
                        "case_id": case_id,
                        "scan_sha256": qualification.scan_by_case[case_id],
                    }
                    for case_id in qualification.train_ids
                ]
            )
        )
    ):
        raise PermissionError("Candidate14 qualification read vectors are not exact")
    for name in (
        "control_read_inventory_commitment_sha256",
        "control_raw_fd_opened_commitment_sha256",
        "control_verified_inode_commitment_sha256",
        "control_read_started_commitment_sha256",
        "resource_capability_result_commitment_sha256",
        "worker_source_result_commitment_sha256",
        "worker_source_inventory_commitment_sha256",
    ):
        try:
            _require_sha(getattr(qualification, name), name=name)
        except (AttributeError, TypeError, ValueError):
            raise PermissionError(
                "Candidate14 qualification commitment is invalid"
            ) from None
    observed_paths = set(entry_paths) | set(reader.allowed) | reader.consumed_paths
    forbidden_exact_paths = {
        manifest,
        _lexical(runner.C14_QUALIFICATION, role="Candidate14 qualification path"),
        _lexical(runner.C14_AUTHORIZATION, role="Candidate14 authorization path"),
        _lexical(runner.C14_ANCHOR, role="Candidate14 anchor path"),
    }
    if any(
        path in forbidden_exact_paths or _path_within(path, cache_root)
        for path in observed_paths
    ) or any(
        purpose == "TRAIN_CACHE_MANIFEST"
        or purpose.endswith("_POOL")
        or purpose in {
            "CANDIDATE14_EXTERNAL_AUTHORIZATION",
            "CANDIDATE14_QUALIFICATION_RECORD",
        }
        for purpose in (
            purposes
            + list(reader.raw_fd_opened_purposes)
            + list(reader.verified_inode_purposes)
            + list(reader.read_started_purposes)
        )
    ):
        raise PermissionError("Candidate14 qualification reached manifest or pool state")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Candidate14 qualification record has duplicate keys")
        result[key] = value
    return result


def _validate_qualification_record(
    record: bytes, *, context: Any = None, qualification: Any = None
) -> Mapping[str, Any]:
    try:
        value = json.loads(
            record.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise PermissionError("Candidate14 qualification record is invalid") from None
    if type(value) is not dict:
        raise PermissionError("Candidate14 qualification record does not match authority")
    for key in QUALIFICATION_COMMITMENT_KEYS:
        try:
            _require_sha(value.get(key), name=key)
        except (TypeError, ValueError):
            raise PermissionError(
                "Candidate14 qualification record does not match authority"
            ) from None
    if (
        set(value) != QUALIFICATION_KEYS
        or canonical_bytes(value) != record
        or sha256_bytes(record) != EXPECTED_QUALIFICATION_SHA256
        or value.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or value.get("check_code") != "ALL_C14_SOURCE_CONTROL_CHECKS_PASSED"
        or value.get("kind")
        != "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_qualification_v1"
        or value.get("schema_version") != 1
        or type(value.get("schema_version")) is not int
        or value.get("engineering_attempt_id")
        != "CANDIDATE14_FULL_POOL_ADAPTER_PREFLIGHT_V1"
        or value.get("scientific_hypothesis_id")
        != "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
        or value.get("source_inventory_commitment_sha256")
        != EXPECTED_SOURCE_INVENTORY_SHA256
        or value.get("control_binding_commitment_sha256")
        != EXPECTED_CONTROL_BINDING_SHA256
        or any(
            value.get(key) != expected
            for key, expected in EXPECTED_STATIC_QUALIFICATION_COMMITMENTS.items()
        )
        or value.get("adapter_or_hypothesis_modified") is not False
        or value.get("cache_manifest_opened") is not False
        or value.get("scientific_hypothesis_consumed") is not False
        or value.get("holdout_label_evaluator_metric_fit_full28_opened_or_started")
        is not False
        or value.get("repeatable") is not True
        or value.get("combined_cpu_limit_seconds") != "240.000"
        or value.get("wall_timeout_seconds") != "300.000"
        or value.get("worker_cpu_soft_limit_seconds") != 240
        or type(value.get("worker_cpu_soft_limit_seconds")) is not int
        or value.get("worker_cpu_hard_limit_seconds") != 241
        or type(value.get("worker_cpu_hard_limit_seconds")) is not int
        or value.get("memory_ceiling_bytes") != 8 * 1024 * 1024 * 1024
        or type(value.get("memory_ceiling_bytes")) is not int
        or value.get("memory_enforcement_kind")
        != "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING"
        or value.get("memory_enforcement_is_hard_rlimit_as") is not False
        or value.get("poll_interval_seconds") != "0.010"
        or value.get("sampling_max_gap_seconds") != "0.050"
        or value.get("sampling_gap_terminal") is not True
        or value.get("sampling_overshoot_between_polls_possible") is not True
        or value.get("terminal_publication_reserved_outside_compute_resource_envelope")
        is not True
        or value.get(
            "terminal_publication_still_anchor_integrity_checked_when_receipt_available"
        )
        is not True
    ):
        raise PermissionError("Candidate14 qualification record does not match authority")
    if context is not None and value["source_inventory_commitment_sha256"] != (
        context.source_commitment_sha256
    ):
        raise PermissionError("Candidate14 qualification source cross-binding mismatch")
    if qualification is not None:
        expected_dynamic = {
            "control_binding_commitment_sha256": qualification.control_binding_commitment_sha256,
            "control_raw_fd_opened_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
            "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
            "control_read_started_commitment_sha256": qualification.control_read_started_commitment_sha256,
            "control_verified_inode_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
            "resource_capability_result_commitment_sha256": qualification.resource_capability_result_commitment_sha256,
            "worker_source_inventory_commitment_sha256": qualification.worker_source_inventory_commitment_sha256,
            "worker_source_result_commitment_sha256": qualification.worker_source_result_commitment_sha256,
        }
        if any(value[key] != expected for key, expected in expected_dynamic.items()):
            raise PermissionError(
                "Candidate14 qualification dynamic cross-binding mismatch"
            )
    return types.MappingProxyType(dict(value))


def _runner_function_snapshot(runner: Any) -> tuple[tuple[Any, Any], ...]:
    return tuple(
        (runner.__dict__[name], runner.__dict__[name].__code__)
        for name in (
            "bootstrap_verified_sources",
            "qualify_source_controls",
            "build_qualification_record",
        )
    )


def _validate_runner_function_snapshot(
    runner: Any, functions: tuple[tuple[Any, Any], ...]
) -> None:
    if any(
        runner.__dict__[name] is not function
        or runner.__dict__[name].__code__ is not code
        for name, (function, code) in zip(
            (
                "bootstrap_verified_sources",
                "qualify_source_controls",
                "build_qualification_record",
            ),
            functions,
        )
    ):
        raise PermissionError("Candidate14 qualification runner changed during use")


def _build_qualification_bytes(
    runner: Any,
    context: Any,
    functions: tuple[tuple[Any, Any], ...],
) -> bytes:
    _validate_source_context(runner, context)
    qualification = runner.qualify_source_controls(context)
    _validate_post_qualification_boundary(runner, context, qualification)
    record = runner.build_qualification_record(context, qualification)
    _validate_post_qualification_boundary(runner, context, qualification)
    _validate_qualification_record(
        record, context=context, qualification=qualification
    )
    _validate_runner_function_snapshot(runner, functions)
    return record


def _source_check_value(runner: Any, context: Any) -> dict[str, Any]:
    _validate_source_context(runner, context)
    return {
        "artifact_or_control_input_opened": False,
        "check_code": "C14_QUALIFICATION_EMITTER_EXACT_SOURCES_VERIFIED",
        "qualification_emitted": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def _parse_args(argv: Sequence[str] | None = None) -> types.SimpleNamespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        len(arguments) != 9
        or not all(type(value) is str for value in arguments)
        or arguments[0] != "--emitter-sha256"
        or arguments[2] != "--emitter-size-bytes"
        or arguments[4] != "--emitter-tests-sha256"
        or arguments[6] != "--emitter-tests-size-bytes"
        or arguments[8]
        not in {"--source-check-only", "--emit-qualification-only"}
        or not arguments[3].isdecimal()
        or not arguments[7].isdecimal()
        or str(int(arguments[3])) != arguments[3]
        or str(int(arguments[7])) != arguments[7]
    ):
        raise ValueError("Candidate14 emitter argument inventory mismatch")
    return types.SimpleNamespace(
        emitter_sha256=_require_sha(arguments[1], name="emitter SHA"),
        emitter_size_bytes=_strict_positive_int(
            int(arguments[3]), name="emitter size"
        ),
        emitter_tests_sha256=_require_sha(
            arguments[5], name="emitter tests SHA"
        ),
        emitter_tests_size_bytes=_strict_positive_int(
            int(arguments[7]), name="emitter tests size"
        ),
        source_check_only=arguments[8] == "--source-check-only",
        emit_qualification_only=arguments[8] == "--emit-qualification-only",
    )


def _private_main(argv: Sequence[str] | None = None) -> bytes:
    binding = _verified_emitter_binding()
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError("Candidate14 emitter requires isolated startup")
    args = _parse_args(argv)
    if (
        binding["sha256"] != _require_sha(args.emitter_sha256, name="emitter SHA")
        or binding["size_bytes"]
        != _strict_positive_int(args.emitter_size_bytes, name="emitter size")
    ):
        raise PermissionError("Candidate14 emitter executing bytes changed")
    _read_exact(
        EMITTER_TESTS_SOURCE,
        role="Candidate14 emitter tests",
        sha256=_require_sha(args.emitter_tests_sha256, name="emitter tests SHA"),
        size_bytes=_strict_positive_int(
            args.emitter_tests_size_bytes, name="emitter tests size"
        ),
    )
    runner, token = _load_verified_runner()
    context: Any = None
    operation_succeeded = False
    emit_requested = args.emit_qualification_only
    try:
        functions = _runner_function_snapshot(runner)
        context = runner.bootstrap_verified_sources(
            runner_sha256=RUNNER_SHA256,
            runner_size_bytes=RUNNER_SIZE_BYTES,
            runner_tests_sha256=RUNNER_TESTS_SHA256,
            runner_tests_size_bytes=RUNNER_TESTS_SIZE_BYTES,
        )
        if args.source_check_only:
            output = canonical_bytes(_source_check_value(runner, context))
            _validate_runner_function_snapshot(runner, functions)
        else:
            output = _build_qualification_bytes(runner, context, functions)
        operation_succeeded = True
    finally:
        resource_module = (
            object.__getattribute__(context, "resource_core")
            if context is not None
            and type(object.__getattribute__(context, "resource_core"))
            is types.ModuleType
            else None
        )
        cleanup_failure: BaseException | None = None
        if context is not None:
            try:
                _clear_runner_issuance_state(
                    runner,
                    context,
                    expected_qualifications=(
                        1 if emit_requested else 0
                    )
                    if operation_succeeded
                    else None,
                )
            except BaseException as error:
                cleanup_failure = error
        try:
            _release_verified_runner(runner, token, resource_module)
        except BaseException as error:
            if cleanup_failure is None:
                cleanup_failure = error
        if cleanup_failure is not None:
            raise cleanup_failure
    return output


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    sys.stderr.write(
        '{"check_code":"C14_QUALIFICATION_EMIT_REJECTED","status":"FAIL_CLOSED"}\n'
    )
    sys.stderr.flush()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
