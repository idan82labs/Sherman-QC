#!/usr/bin/env python3
"""Fresh Candidate15 geometry-representation localizer worker.

The worker accepts one fixed binary stream on stdin.  It authenticates and
seals a complete 36-record descriptor prelude before creating exactly one
Candidate15 context, then reads and localizes one pool blob at a time.  Only
the single canonical aggregate result is written to stdout.  A future guarded
parent owns input custody, the durable consumption anchor, resource sampling,
the read ledger, and publication.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import struct
import sys
import types
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 15
ENGINEERING_ATTEMPT_ID = "CANDIDATE15_GEOMETRY_REPRESENTATION_WORKER_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
SOURCE_CHECK_KIND = "sherman_qc_f1_aug30_candidate15_worker_source_check_v1"
WORKER_RESULT_KIND = "sherman_qc_f1_aug30_candidate15_worker_aggregate_envelope_v1"

DESCRIPTOR_MAGIC = b"C15GEOMBATCHV1!!"
DESCRIPTOR_VERSION = 1
DESCRIPTOR_COUNT = 36
DESCRIPTOR_RECORD = struct.Struct(">IBQ32s")
BLOB_LENGTH = struct.Struct(">Q")
PRIMARY_PASS_CODE = 1
REPLAY_PASS_CODE = 2
MAX_POOL_LEAF_BYTES = 64 * 1024 * 1024
MAX_BOUND_SOURCE_BYTES = 4 * 1024 * 1024
MAX_WORKER_TEST_SOURCE_BYTES = 2 * 1024 * 1024
BATCH_COMMITMENT_DOMAIN = b"C15_GEOMETRY_BATCH_DESCRIPTOR_COMMITMENT_V1\x00"

CHILD_CPU_SOFT_LIMIT_SECONDS = 240
CHILD_CPU_HARD_LIMIT_SECONDS = 241

_C15_VERIFIED_ROOT_VALUE = globals().get("_C15_VERIFIED_ROOT")
ROOT = (
    Path(_C15_VERIFIED_ROOT_VALUE)
    if type(_C15_VERIFIED_ROOT_VALUE) is str
    and _C15_VERIFIED_ROOT_VALUE == "/Users/idantal/ShermanQC/FullProject"
    else Path(__file__).resolve().parents[1]
)
WORKER_SOURCE = ROOT / "scripts/run_f1_train18_candidate15_geometry_representation_worker_v1.py"
WORKER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate15_geometry_representation_worker_v1.py"

FIXED_SOURCE_BINDINGS = MappingProxyType(
    {
        "builder_source": (
            ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
            "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2",
            57613,
        ),
        "builder_tests": (
            ROOT / "tests/test_build_f1_scan_only_candidate_pool.py",
            "43172b7711ce7ec3f9e70845e9c61e5732999cd2f9140f6681a7ba6a98291a0e",
            21708,
        ),
        "c13_source": (
            ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
            "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
            65344,
        ),
        "c13_tests": (
            ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
            "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
            43387,
        ),
        "c14_terminal_disposition": (
            ROOT / "research/f1_aug30_train18_candidate14_terminal_disposition_v1.json",
            "15235350abdb2cc803bc5fb4abca7b836347a500fa9f24a25fb16ea30945b360",
            8008,
        ),
        "c14_resource": (
            ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py",
            "c82fb204b61f99ce7fb7eab8b5ed15f9003be893c004dac37f347bbdbfe4c079",
            15629,
        ),
        "c14_resource_tests": (
            ROOT / "tests/test_f1_train18_candidate14_darwin_resource_capability_v1.py",
            "ac8843ed2fa3d83fc291c5f2ecb0c16fded3320d4484195658551366db8caa51",
            12642,
        ),
        "c15_core": (
            ROOT / "backend/f1_train18_candidate15_geometry_representation_failure_localizer_v1.py",
            "cc65c7b70ac8072ce3adf8125c48f6d763cf57a45904c06ce42264a22df84d33",
            77095,
        ),
        "c15_core_tests": (
            ROOT / "tests/test_f1_train18_candidate15_geometry_representation_failure_localizer_v1.py",
            "b18a8da9edef86972a8f82c9c5cfa58512866d4b2b12d02150eda4c14b6c4c10",
            29681,
        ),
        "c6_source": (
            ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
            "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
            53837,
        ),
        "c6_tests": (
            ROOT / "tests/test_f1_train18_candidate6_conservative_recall_recovery_v1.py",
            "98df5af442627ce71bf539bb6a801cb30e3ad5b3a10f4e680f55a07c6e4c452f",
            26864,
        ),
        "c9_source": (
            ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
            "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
            22354,
        ),
        "c9_tests": (
            ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
            "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
            16598,
        ),
        "clusterer_source": (
            ROOT / "backend/f1_line_object_clustering.py",
            "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c",
            6817,
        ),
        "clusterer_tests": (
            ROOT / "tests/test_f1_line_object_clustering.py",
            "e47176604a465967c0e568c6388be6bec2fdaf5608818d7e44618a36e6c5b8d4",
            1252,
        ),
    }
)

FORBIDDEN_PRELOADED = frozenset(
    {
        "backend.f1_line_object_clustering",
        "backend.f1_train18_candidate6_conservative_recall_recovery_v1",
        "backend.f1_train18_candidate9_lossless_pool_alias_adapter_v1",
        "backend.f1_train18_candidate13_pool_adapter_stage_localizer_v1",
        "backend.f1_train18_candidate15_geometry_representation_failure_localizer_v1",
        "scripts.build_f1_scan_only_candidate_pool",
    }
)

C15_STAGE_CONTRACT = (
    "INPUT_HASH_AND_JSON",
    "C13_PREFIX_0_TO_5_REPRODUCTION",
    "SUPPORTED_GEOMETRY_PRESENT",
    "LINE_START_END_JOINT_OR_ABSENT",
    "C9_AUX_KEY_PRESENCE_REQUIRES_SPAN_WHEN_INCOMPLETE",
    "PRESENT_GEOMETRY_SHAPE_TYPE_FINITE",
    "CONDITIONED_DOMAIN_AND_NONDEGENERACY",
    "C9_PARTIAL_AUX_RECONSTRUCTION_SPAN_EQUIVALENCE",
    "C6_ORDERED_REFERENCE_REPRESENTATION_EQUIVALENCE",
    "EXACT_C13_AGGREGATE_STAGE6_PARITY",
    "EXACT_C9_FULL_OUTCOME_PARITY",
    "AGGREGATE_STAGE6_VIOLATION_COVERAGE",
)
C15_VIOLATION_CONTRACT = (
    "NO_SUPPORTED_GEOMETRY",
    "INCOMPLETE_LINE_ENDPOINT_PAIR",
    "PARTIAL_AUX_WITHOUT_SPAN",
    "MALFORMED_GEOMETRY_VALUE",
    "OUT_OF_DOMAIN_OR_DEGENERATE",
    "PARTIAL_AUX_SPAN_CONFLICT",
    "C6_ORDERED_REFERENCE_REPRESENTATION_CONFLICT",
    "UNCLASSIFIED_OR_PARITY_MISMATCH",
)
C15_RESULT_KIND = (
    "sherman_qc_f1_aug30_candidate15_geometry_representation_failure_localizer_result_v1"
)
C15_RESULT_KEYS = frozenset(
    {
        "adapter_or_hypothesis_semantics_modified_by_core",
        "aggregate_stage6_violation_coverage_for_cohort",
        "artifact_discovery_or_open_performed_by_core",
        "c13_aggregate_called_exactly_once_by_core",
        "c13_aggregate_stage6_parity_for_cohort",
        "c13_receipts_issued_internally_on_identical_inputs",
        "check_code",
        "exact36_ordered_receipts_required",
        "exact_c9_full_outcome_parity_for_cohort",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_geometry_values_published",
        "input_origin_authenticated_by_core",
        "kind",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "per_pool_c13_stage6_parity_claimed",
        "per_pool_violation_exhaustiveness_claimed",
        "public_result_is_aggregate_only",
        "real_or_control_input_opened_by_core",
        "rule_selected",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
        "violation_contract_commitment_sha256",
        "violation_or_bitset",
    }
)

VERIFIED_EXEC_BOOTSTRAP_CODE = r"""import hashlib
import os
import stat
import sys
import types
from types import MappingProxyType

_ROOT = "/Users/idantal/ShermanQC/FullProject"
_EXPECTED = _ROOT + "/scripts/run_f1_train18_candidate15_geometry_representation_worker_v1.py"


def _identity(value):
    return int(value.st_dev), int(value.st_ino)


def _read_verified(path_text, expected_sha, expected_size):
    if type(path_text) is not str or path_text != _EXPECTED:
        raise PermissionError("Candidate15 bootstrap path mismatch")
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or type(expected_size) is not int
        or not 0 < expected_size <= 4 * 1024 * 1024
    ):
        raise ValueError("Candidate15 bootstrap binding mismatch")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate15 bootstrap requires nofollow directories")
    relative_parts = os.path.relpath(path_text, _ROOT).split(os.path.sep)
    if relative_parts[:-1] != ["scripts"] or len(relative_parts) != 2:
        raise PermissionError("Candidate15 bootstrap relative path mismatch")
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
                raise PermissionError("Candidate15 bootstrap directory mismatch")
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
                raise PermissionError("Candidate15 bootstrap directory mismatch")
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
            raise PermissionError("Candidate15 bootstrap leaf mismatch")
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
            raise PermissionError("Candidate15 bootstrap open mismatch")
        chunks = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise PermissionError("Candidate15 bootstrap source truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise PermissionError("Candidate15 bootstrap source grew")
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
            raise PermissionError("Candidate15 bootstrap source changed")
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
        raise PermissionError("Candidate15 bootstrap source hash mismatch")
    return data, root_identity


def _launch():
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError("Candidate15 bootstrap requires isolated startup")
    arguments = list(sys.argv[1:])
    if len(arguments) < 4 or not all(type(value) is str for value in arguments):
        raise ValueError("Candidate15 bootstrap argument mismatch")
    path_text, expected_sha, size_text, *application_arguments = arguments
    if not size_text.isdecimal():
        raise ValueError("Candidate15 bootstrap source size mismatch")
    expected_size = int(size_text)
    data, root_identity = _read_verified(path_text, expected_sha, expected_size)
    private_name = "_candidate15_preverified_worker_" + expected_sha
    if private_name in sys.modules:
        raise PermissionError("Candidate15 bootstrap module already exists")
    outer = sys.modules.get("__main__")
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate15 bootstrap outer module mismatch")
    outer_namespace = object.__getattribute__(outer, "__dict__")
    if outer_namespace.get("_C15_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate15 bootstrap is not reentrant")
    token = object()
    binding = MappingProxyType(
        {"module_name": private_name, "sha256": expected_sha, "size_bytes": expected_size}
    )
    module = types.ModuleType(private_name)
    module.__file__ = path_text
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_C15_VERIFIED_WORKER_BINDING"] = binding
    module.__dict__["_C15_VERIFIED_WORKER_LAUNCH_TOKEN"] = token
    module.__dict__["_C15_VERIFIED_WORKER_OUTER_MODULE"] = outer
    module.__dict__["_C15_VERIFIED_ROOT"] = _ROOT
    module.__dict__["_C15_VERIFIED_ROOT_IDENTITY"] = root_identity
    outer_namespace["_C15_OUTER_LAUNCH_TOKEN"] = token
    sys.modules[private_name] = module
    try:
        exec(compile(data, path_text, "exec", dont_inherit=True), module.__dict__)
        private_main = object.__getattribute__(module, "__dict__").get("main")
        if (
            type(private_main) is not types.FunctionType
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != path_text
        ):
            raise PermissionError("Candidate15 bootstrap entry point mismatch")
        return int(private_main(application_arguments))
    finally:
        sys.modules.pop(private_name, None)
        if outer_namespace.get("_C15_OUTER_LAUNCH_TOKEN") is token:
            del outer_namespace["_C15_OUTER_LAUNCH_TOKEN"]


try:
    _CODE = _launch()
except BaseException:
    os.write(2, b"C15_WORKER_FAIL_CLOSED\n")
    _CODE = 1
raise SystemExit(_CODE)
"""

_IMPORT_HOOK_SEAL = (builtins.__import__, tuple(sys.meta_path), tuple(sys.path_hooks))
_CONTEXT_SEAL = object()
_ISSUED_CONTEXTS: dict[int, tuple[Any, object]] = {}

_C15_VERIFIED_WORKER_BINDING = globals().get("_C15_VERIFIED_WORKER_BINDING")
_C15_VERIFIED_WORKER_LAUNCH_TOKEN = globals().get(
    "_C15_VERIFIED_WORKER_LAUNCH_TOKEN"
)
_C15_VERIFIED_WORKER_OUTER_MODULE = globals().get(
    "_C15_VERIFIED_WORKER_OUTER_MODULE"
)
_C15_VERIFIED_ROOT_IDENTITY_VALUE = globals().get("_C15_VERIFIED_ROOT_IDENTITY")
_VERIFIED_ROOT_DIRECTORY_IDENTITY: tuple[int, int] | None = (
    _C15_VERIFIED_ROOT_IDENTITY_VALUE
    if type(_C15_VERIFIED_ROOT_IDENTITY_VALUE) is tuple
    and len(_C15_VERIFIED_ROOT_IDENTITY_VALUE) == 2
    and all(
        type(value) is int and value >= 0
        for value in _C15_VERIFIED_ROOT_IDENTITY_VALUE
    )
    else None
)


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


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("Candidate15 worker hashes exact bytes")
    return hashlib.sha256(value).hexdigest()


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
        raise ValueError(f"{name} must be one positive exact integer")
    return value


def verified_exec_command(application_arguments: Sequence[str]) -> list[str]:
    """Return the sole isolated, no-site command for a fresh worker process."""

    if type(application_arguments) not in (list, tuple) or not all(
        type(value) is str for value in application_arguments
    ):
        raise ValueError("Candidate15 verified-exec arguments must be exact strings")
    source_bytes, _identity_value = _read_regular_file_once(
        WORKER_SOURCE,
        role="Candidate15 verified-exec worker source",
        expected_size=WORKER_SOURCE.stat(follow_symlinks=False).st_size,
    )
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        VERIFIED_EXEC_BOOTSTRAP_CODE,
        str(WORKER_SOURCE),
        sha256_bytes(source_bytes),
        str(len(source_bytes)),
        *application_arguments,
    ]


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


def _reject_symlinks(path: Path, *, role: str) -> Path:
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
    if missing:
        raise FileNotFoundError(f"{role} must exist")
    return candidate


def _open_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _lexical(path, role=role)
    root = _lexical(ROOT, role="repository")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate15 worker requires nofollow directories")
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
        if _C15_VERIFIED_WORKER_BINDING is not None:
            if _VERIFIED_ROOT_DIRECTORY_IDENTITY != root_identity:
                raise RuntimeError("Candidate15 verified repository root changed")
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


def _read_regular_file_once(
    path: Path, *, role: str, expected_size: int
) -> tuple[bytes, tuple[int, int]]:
    bounded_size = _strict_positive_int(expected_size, name=f"{role} size")
    candidate = _lexical(path, role=role)
    parent_fd, parent_identity = _open_directory(
        candidate.parent, role=f"{role} parent"
    )
    before = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or int(before.st_size) != bounded_size
    ):
        raise ValueError(f"{role} must be one regular non-hard-linked file")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Candidate15 worker requires O_NOFOLLOW")
    descriptor: int | None = None
    try:
        parent_opened = os.fstat(parent_fd)
        if _identity(parent_opened) != parent_identity:
            raise RuntimeError(f"{role} parent changed during open")
        descriptor = os.open(candidate.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
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
            or _identity(os.fstat(parent_fd)) != parent_identity
        ):
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


def _read_exact_source(
    path: Path, *, role: str, sha256: str, size_bytes: int
) -> tuple[bytes, Mapping[str, Any]]:
    expected_sha = _require_sha(sha256, name=f"{role} SHA")
    expected_size = _strict_positive_int(size_bytes, name=f"{role} size")
    if expected_size > MAX_BOUND_SOURCE_BYTES:
        raise ValueError(f"{role} exceeds the source size bound")
    data, identity = _read_regular_file_once(
        path, role=role, expected_size=expected_size
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, MappingProxyType(
        {
            "device_id": identity[0],
            "inode": identity[1],
            "path": str(path.relative_to(ROOT)),
            "purpose": role,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    )


def _verified_worker_execution_binding() -> Mapping[str, Any]:
    binding = _C15_VERIFIED_WORKER_BINDING
    if type(binding) is not MappingProxyType or set(binding) != {
        "module_name",
        "sha256",
        "size_bytes",
    }:
        raise PermissionError("Candidate15 worker was not privately verified")
    if (
        binding["module_name"] != __name__
        or _require_sha(binding["sha256"], name="verified worker SHA")
        != binding["sha256"]
        or _strict_positive_int(
            binding["size_bytes"], name="verified worker size"
        )
        != binding["size_bytes"]
    ):
        raise PermissionError("Candidate15 private worker binding mismatch")
    return binding


def _validate_isolated_startup() -> None:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
    ):
        raise PermissionError(
            "Candidate15 worker requires isolated no-site Python startup"
        )


def _reject_preloaded(paths: Sequence[Path]) -> None:
    if (
        builtins.__import__ is not _IMPORT_HOOK_SEAL[0]
        or tuple(sys.meta_path) != _IMPORT_HOOK_SEAL[1]
        or tuple(sys.path_hooks) != _IMPORT_HOOK_SEAL[2]
    ):
        raise RuntimeError("Candidate15 worker rejects changed import hooks")
    current = sys.modules.get(__name__)
    outer = _C15_VERIFIED_WORKER_OUTER_MODULE
    launch_token = _C15_VERIFIED_WORKER_LAUNCH_TOKEN
    targets = {path.resolve(strict=True) for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in paths}
    if any(name in sys.modules for name in FORBIDDEN_PRELOADED):
        raise RuntimeError("Candidate15 worker rejects a preloaded controlled module")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        if module is outer:
            if type(module) is not types.ModuleType:
                raise RuntimeError("Candidate15 outer module type mismatch")
            namespace = object.__getattribute__(module, "__dict__")
            if namespace.get("_C15_OUTER_LAUNCH_TOKEN") is launch_token:
                continue
            raise RuntimeError("Candidate15 outer launch token changed")
        if not isinstance(module, types.ModuleType):
            continue
        if type(module) is not types.ModuleType:
            raise RuntimeError(
                f"Candidate15 worker rejects preloaded module subclass: {name}"
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
                f"Candidate15 worker rejects preloaded source alias: {name}"
            )


def _exec_verified(
    name: str, path: Path, data: bytes, *, package: str
) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate15 private module name is already loaded")
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


class _ContractArray:
    pass


def _contract_numpy_operation_unavailable(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("Candidate15 lineage-only numpy facade is not executable")


def _numpy_contract_facade() -> types.ModuleType:
    module = types.ModuleType("_candidate15_worker_numpy_contract_facade")
    for name in (
        "abs",
        "arccos",
        "arctan2",
        "argmax",
        "argmin",
        "asarray",
        "clip",
        "column_stack",
        "cos",
        "count_nonzero",
        "cross",
        "degrees",
        "diff",
        "dot",
        "einsum",
        "eye",
        "flatnonzero",
        "floor",
        "isfinite",
        "lexsort",
        "linspace",
        "max",
        "mean",
        "median",
        "mod",
        "ptp",
        "quantile",
        "radians",
        "rint",
        "sin",
        "sort",
        "sqrt",
        "sum",
        "unique",
        "zeros",
    ):
        setattr(module, name, _contract_numpy_operation_unavailable)
    module.float64 = float
    module.int64 = int
    module.ndarray = _ContractArray
    module.pi = math.pi
    module.r_ = object()
    linalg = types.ModuleType("_candidate15_worker_numpy_linalg_contract_facade")
    linalg.eigh = _contract_numpy_operation_unavailable
    linalg.eigvalsh = _contract_numpy_operation_unavailable
    linalg.norm = _contract_numpy_operation_unavailable
    module.linalg = linalg
    return module


def _exact_function_nodes(
    data: bytes,
    *,
    path: Path,
    names: tuple[str, ...],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    try:
        tree = ast.parse(data.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError):
        raise RuntimeError("Candidate15 lineage source parse failed") from None
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    if set(found) != set(names):
        raise RuntimeError("Candidate15 lineage witness function inventory mismatch")
    return tuple(found[name] for name in names)


def _exec_lineage_witness(
    name: str,
    path: Path,
    data: bytes,
    *,
    function_names: tuple[str, ...],
    namespace: Mapping[str, Any],
) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate15 lineage witness name is already loaded")
    functions = _exact_function_nodes(data, path=path, names=function_names)
    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    tree = ast.fix_missing_locations(
        ast.Module(body=[future, *functions], type_ignores=[])
    )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = "backend" if path.parent.name == "backend" else "scripts"
    module.__loader__ = None
    module.__dict__.update(namespace)
    module.__dict__["__candidate15_lineage_only__"] = True
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if any(
        type(module.__dict__.get(function_name)) is not types.FunctionType
        for function_name in function_names
    ):
        sys.modules.pop(name, None)
        raise RuntimeError("Candidate15 lineage witness execution mismatch")
    return module


def _function_snapshot(*modules: types.ModuleType) -> tuple[Any, ...]:
    values: list[tuple[Any, ...]] = []
    for module in modules:
        for name, value in sorted(vars(module).items()):
            if type(value) is types.FunctionType:
                values.append((module.__name__, name, value, value.__code__))
            elif type(value) is type and value.__module__ == module.__name__:
                for child_name, child in sorted(vars(value).items()):
                    if type(child) is types.FunctionType:
                        values.append(
                            (
                                f"{module.__name__}.{name}",
                                child_name,
                                child,
                                child.__code__,
                            )
                        )
    return tuple(values)


@dataclass(frozen=True)
class BootstrapContext:
    c15_core: types.ModuleType
    c13_core: types.ModuleType
    c9_adapter: types.ModuleType
    c6_hypothesis: types.ModuleType
    builder: types.ModuleType
    clusterer: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_commitment_sha256: str
    _function_snapshot: tuple[Any, ...]
    _controlled_callables: tuple[Any, ...]
    _sentinel: object
    _token: object


def _execute_lineage_witnesses(
    *, builder_data: bytes, clusterer_data: bytes
) -> tuple[types.ModuleType, types.ModuleType]:
    numpy_facade = _numpy_contract_facade()
    clusterer = _exec_lineage_witness(
        "_candidate15_worker_verified_clusterer",
        FIXED_SOURCE_BINDINGS["clusterer_source"][0],
        clusterer_data,
        function_names=("cluster_line_candidates",),
        namespace={"math": math, "np": numpy_facade},
    )
    builder = _exec_lineage_witness(
        "_candidate15_worker_verified_builder",
        FIXED_SOURCE_BINDINGS["builder_source"][0],
        builder_data,
        function_names=("_bounded_identifiers", "_tag"),
        namespace={
            "hashlib": hashlib,
            "json": json,
            "math": math,
            "np": numpy_facade,
            "SUPPORT_ATOM_ID_CAP_PER_CANDIDATE": 4096,
        },
    )
    return builder, clusterer


def bootstrap_verified_sources(
    *,
    worker_sha256: str,
    worker_size_bytes: int,
    worker_tests_sha256: str,
    worker_tests_size_bytes: int,
) -> BootstrapContext:
    execution = _verified_worker_execution_binding()
    if (
        execution["sha256"] != _require_sha(worker_sha256, name="worker SHA")
        or execution["size_bytes"]
        != _strict_positive_int(worker_size_bytes, name="worker size")
    ):
        raise PermissionError("Candidate15 executing worker byte binding mismatch")
    specs = {
        **FIXED_SOURCE_BINDINGS,
        "worker": (
            WORKER_SOURCE,
            _require_sha(worker_sha256, name="worker SHA"),
            _strict_positive_int(worker_size_bytes, name="worker size"),
        ),
        "worker_tests": (
            WORKER_TESTS_SOURCE,
            _require_sha(worker_tests_sha256, name="worker tests SHA"),
            _strict_positive_int(worker_tests_size_bytes, name="worker tests size"),
        ),
    }
    _reject_preloaded(tuple(value[0] for value in specs.values()))
    source_bytes: dict[str, bytes] = {}
    source_bindings: dict[str, dict[str, Any]] = {}
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha256, size_bytes = specs[name]
        data, entry = _read_exact_source(
            path,
            role=f"CANDIDATE15_WORKER_SOURCE_{name.upper()}",
            sha256=sha256,
            size_bytes=size_bytes,
        )
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate15 worker source inventory has an inode alias")
        identities.add(identity)
        source_bytes[name] = data
        source_bindings[name] = {
            "path": entry["path"],
            "sha256": sha256,
            "size_bytes": size_bytes,
        }

    c6 = _exec_verified(
        "_candidate15_worker_verified_c6",
        FIXED_SOURCE_BINDINGS["c6_source"][0],
        source_bytes["c6_source"],
        package="backend",
    )
    builder, clusterer = _execute_lineage_witnesses(
        builder_data=source_bytes["builder_source"],
        clusterer_data=source_bytes["clusterer_source"],
    )
    c9 = _exec_verified(
        "_candidate15_worker_verified_c9",
        FIXED_SOURCE_BINDINGS["c9_source"][0],
        source_bytes["c9_source"],
        package="backend",
    )
    c13 = _exec_verified(
        "_candidate15_worker_verified_c13",
        FIXED_SOURCE_BINDINGS["c13_source"][0],
        source_bytes["c13_source"],
        package="backend",
    )
    c15 = _exec_verified(
        "_candidate15_worker_verified_c15",
        FIXED_SOURCE_BINDINGS["c15_core"][0],
        source_bytes["c15_core"],
        package="backend",
    )
    c15_bindings = {
        name: dict(source_bindings[name]) for name in sorted(c15.FIXED_BINDINGS)
    }
    if (
        c6.ATTEMPT_ORDINAL != 6
        or c9.ATTEMPT_ORDINAL != 9
        or c13.ATTEMPT_ORDINAL != 13
        or c15.ATTEMPT_ORDINAL != 15
        or c13.ENGINEERING_ATTEMPT_ID
        != "CANDIDATE13_POOL_ADAPTER_STAGE_LOCALIZER_V1"
        or c13.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or c15.ENGINEERING_ATTEMPT_ID
        != "CANDIDATE15_GEOMETRY_REPRESENTATION_FAILURE_LOCALIZER_V1"
        or c15.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
        or getattr(builder, "__candidate15_lineage_only__", None) is not True
        or getattr(clusterer, "__candidate15_lineage_only__", None) is not True
        or tuple(c15.STAGE_CONTRACT) != C15_STAGE_CONTRACT
        or tuple(c15.VIOLATION_CONTRACT) != C15_VIOLATION_CONTRACT
        or set(c15.RESULT_KEYS) != C15_RESULT_KEYS
        or c15.FIXED_BINDINGS != c15_bindings
    ):
        raise ValueError("Candidate15 worker controlled runtime identity mismatch")
    immutable_bindings = MappingProxyType(
        {
            name: MappingProxyType(dict(value))
            for name, value in source_bindings.items()
        }
    )
    source_commitment = sha256_bytes(
        canonical_bytes(
            {
                name: dict(value)
                for name, value in sorted(immutable_bindings.items())
            }
        )
    )
    token = object()
    context = BootstrapContext(
        c15_core=c15,
        c13_core=c13,
        c9_adapter=c9,
        c6_hypothesis=c6,
        builder=builder,
        clusterer=clusterer,
        source_bindings=immutable_bindings,
        source_commitment_sha256=source_commitment,
        _function_snapshot=_function_snapshot(c6, clusterer, builder, c9, c13, c15),
        _controlled_callables=(
            c15.make_verified_context,
            c15.localize_geometry_representation_contract,
            c15.aggregate_geometry_representation_receipts,
        ),
        _sentinel=_CONTEXT_SEAL,
        _token=token,
    )
    _ISSUED_CONTEXTS[id(context)] = (context, token)
    return context


def _validate_bootstrap_context(context: BootstrapContext) -> None:
    issued = _ISSUED_CONTEXTS.get(id(context))
    if (
        type(context) is not BootstrapContext
        or context._sentinel is not _CONTEXT_SEAL
        or issued is None
        or issued[0] is not context
        or issued[1] is not context._token
        or context._controlled_callables
        != (
            context.c15_core.make_verified_context,
            context.c15_core.localize_geometry_representation_contract,
            context.c15_core.aggregate_geometry_representation_receipts,
        )
        or getattr(context.builder, "__candidate15_lineage_only__", None)
        is not True
        or getattr(context.clusterer, "__candidate15_lineage_only__", None)
        is not True
        or context._function_snapshot
        != _function_snapshot(
            context.c6_hypothesis,
            context.clusterer,
            context.builder,
            context.c9_adapter,
            context.c13_core,
            context.c15_core,
        )
        or context.source_commitment_sha256
        != sha256_bytes(
            canonical_bytes(
                {
                    name: dict(value)
                    for name, value in sorted(context.source_bindings.items())
                }
            )
        )
    ):
        raise PermissionError("Candidate15 worker bootstrap context changed")


@dataclass(frozen=True, slots=True)
class PoolDescriptor:
    ordinal: int
    pass_code: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DescriptorPrelude:
    records: tuple[PoolDescriptor, ...]
    batch_commitment_sha256: str


def _descriptor_record_bytes(record: PoolDescriptor) -> bytes:
    if (
        type(record) is not PoolDescriptor
        or type(record.ordinal) is not int
        or not 1 <= record.ordinal <= DESCRIPTOR_COUNT
        or type(record.pass_code) is not int
        or record.pass_code
        != (PRIMARY_PASS_CODE if record.ordinal % 2 else REPLAY_PASS_CODE)
        or type(record.size_bytes) is not int
        or not 0 <= record.size_bytes <= MAX_POOL_LEAF_BYTES
    ):
        raise ValueError("Candidate15 worker pool descriptor invariant mismatch")
    sha = _require_sha(record.sha256, name="pool descriptor SHA")
    return DESCRIPTOR_RECORD.pack(
        record.ordinal, record.pass_code, record.size_bytes, bytes.fromhex(sha)
    )


def batch_descriptor_commitment(records: tuple[PoolDescriptor, ...]) -> str:
    if type(records) is not tuple or len(records) != DESCRIPTOR_COUNT:
        raise ValueError("Candidate15 worker requires exactly 36 descriptors")
    if tuple(record.ordinal for record in records) != tuple(
        range(1, DESCRIPTOR_COUNT + 1)
    ):
        raise ValueError("Candidate15 worker descriptor order mismatch")
    material = b"".join(_descriptor_record_bytes(record) for record in records)
    return hashlib.sha256(BATCH_COMMITMENT_DOMAIN + material).hexdigest()


def encode_descriptor_prelude(records: tuple[PoolDescriptor, ...]) -> bytes:
    commitment = batch_descriptor_commitment(records)
    return b"".join(
        (
            DESCRIPTOR_MAGIC,
            struct.pack(">II", DESCRIPTOR_VERSION, DESCRIPTOR_COUNT),
            *(_descriptor_record_bytes(record) for record in records),
            bytes.fromhex(commitment),
        )
    )


def encode_blob_frame(value: bytes) -> bytes:
    if type(value) is not bytes or len(value) > MAX_POOL_LEAF_BYTES:
        raise ValueError("Candidate15 worker blob frame is invalid")
    return BLOB_LENGTH.pack(len(value)) + value


def _read_exact_fd(descriptor: int, size: int) -> bytes:
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("Candidate15 worker stream descriptor is invalid")
    if type(size) is not int or size < 0:
        raise ValueError("Candidate15 worker stream size is invalid")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise EOFError("Candidate15 worker stream ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_descriptor_prelude(descriptor: int = 0) -> DescriptorPrelude:
    magic = _read_exact_fd(descriptor, len(DESCRIPTOR_MAGIC))
    if magic != DESCRIPTOR_MAGIC:
        raise ValueError("Candidate15 worker descriptor magic mismatch")
    version, count = struct.unpack(">II", _read_exact_fd(descriptor, 8))
    if version != DESCRIPTOR_VERSION or count != DESCRIPTOR_COUNT:
        raise ValueError("Candidate15 worker descriptor header mismatch")
    records: list[PoolDescriptor] = []
    for expected_ordinal in range(1, DESCRIPTOR_COUNT + 1):
        ordinal, pass_code, size_bytes, sha = DESCRIPTOR_RECORD.unpack(
            _read_exact_fd(descriptor, DESCRIPTOR_RECORD.size)
        )
        record = PoolDescriptor(
            ordinal=ordinal,
            pass_code=pass_code,
            size_bytes=size_bytes,
            sha256=sha.hex(),
        )
        _descriptor_record_bytes(record)
        if ordinal != expected_ordinal:
            raise ValueError("Candidate15 worker descriptor ordinal mismatch")
        records.append(record)
    received_commitment = _read_exact_fd(descriptor, 32).hex()
    sealed_records = tuple(records)
    expected_commitment = batch_descriptor_commitment(sealed_records)
    if received_commitment != expected_commitment:
        raise ValueError("Candidate15 worker descriptor commitment mismatch")
    return DescriptorPrelude(
        records=sealed_records,
        batch_commitment_sha256=expected_commitment,
    )


def _install_cpu_backup() -> None:
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (CHILD_CPU_SOFT_LIMIT_SECONDS, CHILD_CPU_HARD_LIMIT_SECONDS),
    )
    if resource.getrlimit(resource.RLIMIT_CPU) != (
        CHILD_CPU_SOFT_LIMIT_SECONDS,
        CHILD_CPU_HARD_LIMIT_SECONDS,
    ):
        raise RuntimeError("Candidate15 worker CPU backup limit mismatch")


def _c15_bindings(context: BootstrapContext) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, binding in context.c15_core.FIXED_BINDINGS.items():
        if type(name) is not str or type(binding) is not dict:
            raise ValueError("Candidate15 worker core binding schema mismatch")
        supplied = context.source_bindings.get(name)
        if supplied is None or dict(supplied) != binding:
            raise ValueError("Candidate15 worker core source binding mismatch")
        result[name] = dict(supplied)
    return result


def validate_c15_aggregate(value: Any) -> None:
    if type(value) is not dict or set(value) != C15_RESULT_KEYS:
        raise ValueError("Candidate15 worker aggregate schema mismatch")
    reached = value["stage_reached_all_bitset"]
    passed = value["stage_passed_all_bitset"]
    violations = value["violation_or_bitset"]
    false_keys = {
        "adapter_or_hypothesis_semantics_modified_by_core",
        "artifact_discovery_or_open_performed_by_core",
        "fit_performed",
        "full28_opened",
        "holdout_opened",
        "input_or_geometry_values_published",
        "input_origin_authenticated_by_core",
        "label_or_evaluator_opened",
        "metric_computed",
        "performance_tested",
        "per_pool_c13_stage6_parity_claimed",
        "per_pool_violation_exhaustiveness_claimed",
        "real_or_control_input_opened_by_core",
        "rule_selected",
    }
    true_keys = {
        "c13_aggregate_called_exactly_once_by_core",
        "c13_receipts_issued_internally_on_identical_inputs",
        "exact36_ordered_receipts_required",
        "public_result_is_aggregate_only",
    }
    bool_keys = {
        "aggregate_stage6_violation_coverage_for_cohort",
        "c13_aggregate_stage6_parity_for_cohort",
        "exact_c9_full_outcome_parity_for_cohort",
    }
    string_keys = {
        "check_code",
        "kind",
        "stage_contract_commitment_sha256",
        "stage_passed_all_bitset",
        "stage_reached_all_bitset",
        "status",
        "violation_contract_commitment_sha256",
        "violation_or_bitset",
    }
    if (
        any(value[key] is not False for key in false_keys)
        or any(value[key] is not True for key in true_keys)
        or any(type(value[key]) is not bool for key in bool_keys)
        or any(type(value[key]) is not str for key in string_keys)
        or value["kind"] != C15_RESULT_KIND
        or len(reached) != len(C15_STAGE_CONTRACT)
        or len(passed) != len(C15_STAGE_CONTRACT)
        or len(violations) != len(C15_VIOLATION_CONTRACT)
        or set(reached + passed + violations) - {"0", "1"}
        or any(pass_bit == "1" and reached_bit != "1" for reached_bit, pass_bit in zip(reached, passed))
        or reached[-3:] != "111"
        or value["stage_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(C15_STAGE_CONTRACT)))
        or value["violation_contract_commitment_sha256"]
        != sha256_bytes(canonical_bytes(list(C15_VIOLATION_CONTRACT)))
        or value["c13_aggregate_stage6_parity_for_cohort"] is not (passed[9] == "1")
        or value["exact_c9_full_outcome_parity_for_cohort"] is not (passed[10] == "1")
        or value["aggregate_stage6_violation_coverage_for_cohort"] is not (passed[11] == "1")
        or (violations[7] == "1") is not (passed[9] == "0" or passed[10] == "0" or passed[11] == "0")
    ):
        raise ValueError("Candidate15 worker aggregate invariant mismatch")
    if passed[-3:] != "111":
        expected = (
            "C15_AGGREGATE_PARITY_OR_COVERAGE_MISMATCH",
            "FAILED_ENGINEERING_LOCALIZER_EQUIVALENCE",
        )
    elif passed[:9] == "1" * 9:
        expected = (
            "C15_GEOMETRY_REPRESENTATION_CONTRACT_CLEAR",
            "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
        )
    else:
        expected = (
            "C15_GEOMETRY_REPRESENTATION_FAILURE_LOCALIZED",
            "SOURCE_POOL_GEOMETRY_INCOMPATIBLE_NONAUTHORIZING",
        )
    if (value["check_code"], value["status"]) != expected:
        raise ValueError("Candidate15 worker aggregate status mismatch")


WORKER_RESULT_KEYS = frozenset(
    {
        "batch_commitment_sha256",
        "blob_inventory_complete",
        "c15_aggregate",
        "c15_aggregate_commitment_sha256",
        "descriptor_inventory_complete",
        "input_eof_verified",
        "kind",
        "schema_version",
    }
)


def validate_worker_result_envelope(value: Any) -> None:
    if type(value) is not dict or set(value) != WORKER_RESULT_KEYS:
        raise ValueError("Candidate15 worker result envelope schema mismatch")
    aggregate = value["c15_aggregate"]
    validate_c15_aggregate(aggregate)
    if (
        value["kind"] != WORKER_RESULT_KIND
        or value["schema_version"] != 1
        or type(value["schema_version"]) is not int
        or value["descriptor_inventory_complete"] is not True
        or value["blob_inventory_complete"] is not True
        or value["input_eof_verified"] is not True
        or _require_sha(
            value["batch_commitment_sha256"], name="worker batch commitment"
        )
        != value["batch_commitment_sha256"]
        or value["c15_aggregate_commitment_sha256"]
        != sha256_bytes(canonical_bytes(aggregate))
    ):
        raise ValueError("Candidate15 worker result envelope invariant mismatch")


def source_check_result(context: BootstrapContext) -> dict[str, Any]:
    _validate_bootstrap_context(context)
    return {
        "check_code": "C15_WORKER_EXACT_SOURCE_BYTES_VERIFIED",
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "fixed_source_and_disposition_lineage_files_opened": True,
        "input_custody_or_scientific_performance_tested": False,
        "kind": SOURCE_CHECK_KIND,
        "manifest_pool_semantic_label_evaluator_metric_fit_full28_holdout_opened": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def run_worker(context: BootstrapContext, *, input_descriptor: int = 0) -> dict[str, Any]:
    _validate_bootstrap_context(context)
    _install_cpu_backup()
    prelude = read_descriptor_prelude(input_descriptor)
    if type(prelude) is not DescriptorPrelude:
        raise PermissionError("Candidate15 worker descriptor seal mismatch")
    _validate_bootstrap_context(context)
    c15_context = context.c15_core.make_verified_context(
        c13_localizer=context.c13_core,
        c9_adapter=context.c9_adapter,
        c6_hypothesis=context.c6_hypothesis,
        builder=context.builder,
        clusterer=context.clusterer,
        bindings=_c15_bindings(context),
    )
    receipts: list[Any] = []
    for record in prelude.records:
        frame_size = BLOB_LENGTH.unpack(
            _read_exact_fd(input_descriptor, BLOB_LENGTH.size)
        )[0]
        if frame_size != record.size_bytes or frame_size > MAX_POOL_LEAF_BYTES:
            raise ValueError("Candidate15 worker blob frame size mismatch")
        pool_bytes = _read_exact_fd(input_descriptor, frame_size)
        if sha256_bytes(pool_bytes) != record.sha256:
            raise ValueError("Candidate15 worker pool blob SHA mismatch")
        receipt = context.c15_core.localize_geometry_representation_contract(
            c15_context,
            pool_bytes=pool_bytes,
            expected_pool_sha256=record.sha256,
        )
        receipts.append(receipt)
        del pool_bytes
    if os.read(input_descriptor, 1) != b"":
        raise ValueError("Candidate15 worker rejects trailing input")
    if len(receipts) != DESCRIPTOR_COUNT:
        raise RuntimeError("Candidate15 worker receipt inventory mismatch")
    aggregate = context.c15_core.aggregate_geometry_representation_receipts(
        c15_context, tuple(receipts)
    )
    validate_c15_aggregate(aggregate)
    _validate_bootstrap_context(context)
    result = {
        "batch_commitment_sha256": prelude.batch_commitment_sha256,
        "blob_inventory_complete": True,
        "c15_aggregate": aggregate,
        "c15_aggregate_commitment_sha256": sha256_bytes(canonical_bytes(aggregate)),
        "descriptor_inventory_complete": True,
        "input_eof_verified": True,
        "kind": WORKER_RESULT_KIND,
        "schema_version": 1,
    }
    validate_worker_result_envelope(result)
    return result


def _bootstrap_cli_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"Candidate15 worker requires exactly one {flag}")
    value = arguments[positions[0] + 1]
    if type(value) is not str or value.startswith("--"):
        raise ValueError(f"Candidate15 worker has an invalid {flag}")
    return value


def _dispatch_through_verified_worker(argv: Sequence[str] | None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not all(type(value) is str for value in arguments):
        raise TypeError("Candidate15 worker arguments must be exact strings")
    expected_sha = _require_sha(
        _bootstrap_cli_value(arguments, "--worker-sha256"), name="worker SHA"
    )
    size_text = _bootstrap_cli_value(arguments, "--worker-size-bytes")
    if not size_text.isdecimal() or int(size_text) <= 0:
        raise ValueError("Candidate15 worker size is invalid")
    expected_size = int(size_text)
    data, _identity_value = _read_regular_file_once(
        Path(__file__),
        role="Candidate15 executing worker bootstrap",
        expected_size=expected_size,
    )
    if len(data) != expected_size or sha256_bytes(data) != expected_sha:
        raise PermissionError("Candidate15 executing worker byte binding mismatch")
    private_name = f"_candidate15_verified_worker_{expected_sha}"
    if private_name in sys.modules:
        raise PermissionError("Candidate15 verified worker is already loaded")
    outer = sys.modules.get(__name__)
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate15 outer worker module is unavailable")
    namespace = object.__getattribute__(outer, "__dict__")
    if namespace.get("_C15_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate15 worker bootstrap is not reentrant")
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
    module.__dict__["_C15_VERIFIED_WORKER_BINDING"] = binding
    module.__dict__["_C15_VERIFIED_WORKER_LAUNCH_TOKEN"] = launch_token
    module.__dict__["_C15_VERIFIED_WORKER_OUTER_MODULE"] = outer
    namespace["_C15_OUTER_LAUNCH_TOKEN"] = launch_token
    sys.modules[private_name] = module
    try:
        exec(compile(data, str(Path(__file__)), "exec", dont_inherit=True), module.__dict__)
        private_main = object.__getattribute__(module, "__dict__").get("main")
        if (
            type(private_main) is not types.FunctionType
            or private_main.__module__ != private_name
            or private_main.__code__.co_filename != str(Path(__file__))
        ):
            raise PermissionError("Candidate15 private worker entry point mismatch")
        return int(private_main(arguments))
    finally:
        sys.modules.pop(private_name, None)
        if namespace.get("_C15_OUTER_LAUNCH_TOKEN") is launch_token:
            del namespace["_C15_OUTER_LAUNCH_TOKEN"]


@dataclass(frozen=True, slots=True)
class CliArguments:
    worker_sha256: str
    worker_size_bytes: int
    worker_tests_sha256: str
    worker_tests_size_bytes: int
    source_check_only: bool
    worker_mode: bool


def _parse_args(argv: Sequence[str]) -> CliArguments:
    if not all(type(value) is str for value in argv):
        raise ValueError("Candidate15 worker arguments must be exact strings")
    modes = [value for value in argv if value in {"--source-check-only", "--worker-mode"}]
    if len(modes) != 1:
        raise ValueError("Candidate15 worker requires exactly one explicit mode")
    allowed = {
        "--worker-sha256",
        "--worker-size-bytes",
        "--worker-tests-sha256",
        "--worker-tests-size-bytes",
        "--source-check-only",
        "--worker-mode",
    }
    if any(value.startswith("--") and value not in allowed for value in argv):
        raise ValueError("Candidate15 worker rejects unknown arguments")
    sha = _require_sha(_bootstrap_cli_value(argv, "--worker-sha256"), name="worker SHA")
    test_sha = _require_sha(
        _bootstrap_cli_value(argv, "--worker-tests-sha256"),
        name="worker tests SHA",
    )
    size_text = _bootstrap_cli_value(argv, "--worker-size-bytes")
    test_size_text = _bootstrap_cli_value(argv, "--worker-tests-size-bytes")
    if not size_text.isdecimal() or not test_size_text.isdecimal():
        raise ValueError("Candidate15 worker source sizes must be decimal")
    size = int(size_text)
    test_size = int(test_size_text)
    _strict_positive_int(size, name="worker size")
    _strict_positive_int(test_size, name="worker tests size")
    if size > MAX_BOUND_SOURCE_BYTES or test_size > MAX_WORKER_TEST_SOURCE_BYTES:
        raise ValueError("Candidate15 worker source size bound exceeded")
    expected_length = 9
    if len(argv) != expected_length:
        raise ValueError("Candidate15 worker argument inventory mismatch")
    return CliArguments(
        worker_sha256=sha,
        worker_size_bytes=size,
        worker_tests_sha256=test_sha,
        worker_tests_size_bytes=test_size,
        source_check_only=modes[0] == "--source-check-only",
        worker_mode=modes[0] == "--worker-mode",
    )


def main(argv: Sequence[str] | None = None) -> int:
    if _C15_VERIFIED_WORKER_BINDING is None:
        os.write(2, b"C15_WORKER_FAIL_CLOSED\n")
        return 1
    try:
        _validate_isolated_startup()
        _verified_worker_execution_binding()
        arguments = _parse_args(list(sys.argv[1:] if argv is None else argv))
        context = bootstrap_verified_sources(
            worker_sha256=arguments.worker_sha256,
            worker_size_bytes=arguments.worker_size_bytes,
            worker_tests_sha256=arguments.worker_tests_sha256,
            worker_tests_size_bytes=arguments.worker_tests_size_bytes,
        )
        if arguments.source_check_only:
            value = source_check_result(context)
        else:
            value = run_worker(context)
        os.write(1, canonical_bytes(value))
        return 0
    except BaseException:
        os.write(2, b"C15_WORKER_FAIL_CLOSED\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
