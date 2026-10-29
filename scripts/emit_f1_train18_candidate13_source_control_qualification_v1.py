#!/usr/bin/env python3
"""Emit the exact Candidate13 repeatable qualification record to stdout only.

This launcher has no publication mode and performs no filesystem writes.  Its
only consuming mode delegates to the exact frozen Candidate13 runner's public
source bootstrap, repeatable source-control qualification, and canonical
qualification builder.  It rejects any manifest or pool capability/read state
and emits bytes only when the authorized canonical record SHA matches exactly.
"""

from __future__ import annotations

import argparse
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
    ROOT / "scripts/emit_f1_train18_candidate13_source_control_qualification_v1.py"
)
EMITTER_TESTS_SOURCE = (
    ROOT / "tests/test_emit_f1_train18_candidate13_source_control_qualification_v1.py"
)
RUNNER_SOURCE = (
    ROOT
    / "scripts/run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
)
RUNNER_TESTS_SOURCE = (
    ROOT
    / "tests/test_run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
)

RUNNER_SHA256 = "b56c2de4b47a8e8eb77194d99d381fe385b8c4782330614ee0a5190eb61ca641"
RUNNER_SIZE_BYTES = 195650
RUNNER_TESTS_SHA256 = "7792aab139839d3967aa009444279d9f0601d9d1edd65f0265c27b910a5da703"
RUNNER_TESTS_SIZE_BYTES = 76268
EXPECTED_QUALIFICATION_SHA256 = (
    "f4a2e6e8f4bf8dbcf3b3151dc7c5fe7439a7c8502ffff6ea99c4417d605ac5d8"
)
EXPECTED_CONTROL_PURPOSES = (
    "SEALED_SPLIT",
    "ORIGINAL_PREREGISTRATION",
    "TRAIN_ALLOWLIST",
    "CANDIDATE11_TERMINAL_DISPOSITION",
    "CANDIDATE12_SYNTHETIC_DISPOSITION",
    "TRAIN_CACHE_ATTESTATION",
    "CACHE_CUSTODY_ACCESS_EVIDENCE",
    "CACHE_CUSTODY_BUILDER",
    "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
    "CACHE_CUSTODY_VALIDATOR",
)

CONTROLLED_SOURCE_PATHS = (
    RUNNER_SOURCE,
    RUNNER_TESTS_SOURCE,
    ROOT / "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
    ROOT / "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
    ROOT / "backend/f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py",
    ROOT / "tests/test_f1_train18_candidate12_synthetic_adapter_contract_localizer_v1.py",
    ROOT / "backend/f1_train18_candidate11_full_replay_diagnostic_v1.py",
    ROOT / "scripts/run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py",
    ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
    ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
    ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
    ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
    ROOT / "backend/f1_line_object_clustering.py",
)

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
        raise TypeError("Candidate13 qualification emitter hashes exact bytes only")
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
        raise ValueError("Candidate13 qualification repository root is a symlink")
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


def _read_regular_file_once(path: Path, *, role: str) -> tuple[bytes, tuple[int, int]]:
    candidate = _reject_symlinks(path, role=role)
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
        before = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
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
        ):
            raise ValueError(f"{role} changed during open")
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


def _read_exact(path: Path, *, role: str, sha256: str, size_bytes: int) -> bytes:
    data, _file_identity = _read_regular_file_once(path, role=role)
    if (
        len(data) != _strict_positive_int(size_bytes, name=f"{role} size")
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
        != f"_candidate13_qualification_emitter_{binding.get('sha256', '')}"
        or type(binding.get("size_bytes")) is not int
        or int(binding.get("size_bytes")) <= 0
    ):
        raise PermissionError(
            "Candidate13 qualification emitter is not verified private code"
        )
    return binding


def _validate_import_hooks() -> None:
    if (
        builtins.__import__ is not _IMPORT_HOOK_SEAL[0]
        or tuple(sys.meta_path) != _IMPORT_HOOK_SEAL[1]
        or tuple(sys.path_hooks) != _IMPORT_HOOK_SEAL[2]
    ):
        raise RuntimeError("Candidate13 qualification emitter rejects import-hook drift")


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
                f"Candidate13 qualification emitter rejects module subclass: {name}"
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
                f"Candidate13 qualification emitter rejects source alias: {name}"
            )


def _load_verified_runner() -> tuple[types.ModuleType, object]:
    _verified_emitter_binding()
    _reject_preloaded_sources(CONTROLLED_SOURCE_PATHS)
    runner_bytes = _read_exact(
        RUNNER_SOURCE,
        role="Candidate13 frozen runner",
        sha256=RUNNER_SHA256,
        size_bytes=RUNNER_SIZE_BYTES,
    )
    _read_exact(
        RUNNER_TESTS_SOURCE,
        role="Candidate13 frozen runner tests",
        sha256=RUNNER_TESTS_SHA256,
        size_bytes=RUNNER_TESTS_SIZE_BYTES,
    )
    name = f"_candidate13_verified_runner_{RUNNER_SHA256}"
    if name in sys.modules:
        raise PermissionError("Candidate13 frozen runner is already loaded")
    outer = sys.modules.get(__name__)
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate13 qualification emitter module is unavailable")
    namespace = object.__getattribute__(outer, "__dict__")
    if namespace.get("_C13_OUTER_LAUNCH_TOKEN") is not None:
        raise PermissionError("Candidate13 runner launch is not reentrant")
    token = object()
    module = types.ModuleType(name)
    module.__file__ = str(RUNNER_SOURCE)
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_C13_VERIFIED_RUNNER_BINDING"] = types.MappingProxyType(
        {
            "module_name": name,
            "sha256": RUNNER_SHA256,
            "size_bytes": RUNNER_SIZE_BYTES,
        }
    )
    module.__dict__["_C13_VERIFIED_RUNNER_LAUNCH_TOKEN"] = token
    module.__dict__["_C13_VERIFIED_RUNNER_OUTER_MODULE"] = outer
    namespace["_C13_OUTER_LAUNCH_TOKEN"] = token
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
                    "Candidate13 qualification runner public function mismatch"
                )
        return module, token
    except BaseException:
        sys.modules.pop(name, None)
        if namespace.get("_C13_OUTER_LAUNCH_TOKEN") is token:
            namespace.pop("_C13_OUTER_LAUNCH_TOKEN", None)
        raise


def _release_verified_runner(module: types.ModuleType, token: object) -> None:
    outer = sys.modules.get(__name__)
    namespace = (
        object.__getattribute__(outer, "__dict__")
        if type(outer) is types.ModuleType
        else {}
    )
    sys.modules.pop(module.__name__, None)
    if namespace.get("_C13_OUTER_LAUNCH_TOKEN") is token:
        namespace.pop("_C13_OUTER_LAUNCH_TOKEN", None)


def _path_within(path: Path, root: Path) -> bool:
    candidate = _lexical(path, role="qualified reader path")
    boundary = _lexical(root, role="forbidden cache root")
    return os.path.commonpath((os.fspath(candidate), os.fspath(boundary))) == os.fspath(
        boundary
    )


def _validate_post_qualification_boundary(runner: Any, context: Any, qualification: Any) -> None:
    reader = qualification.reader
    manifest = _lexical(runner.CACHE_MANIFEST, role="Candidate13 manifest")
    cache_root = _lexical(runner.ALLOWED_CACHE_ROOT, role="Candidate13 pool root")
    source_count = len(context.source_entries)
    purposes = [str(entry["purpose"]) for entry in reader.entries]
    expected_prefix = [str(entry["purpose"]) for entry in context.source_entries]
    expected_prefix.extend(EXPECTED_CONTROL_PURPOSES)
    if (
        purposes != expected_prefix
        or len(reader.entries) <= source_count
        or reader.raw_fd_opened_purposes != purposes
        or reader.verified_inode_purposes != purposes
        or reader.read_started_purposes != purposes
        or reader.artifact_inventory is not None
        or reader.denied_attempts
    ):
        raise PermissionError("Candidate13 qualification read vectors are not exact")
    observed_paths: list[Path] = []
    for entry in reader.entries:
        observed_paths.append(_lexical(ROOT / str(entry["path"]), role="read entry"))
    observed_paths.extend(_lexical(path, role="allowed path") for path in reader.allowed)
    observed_paths.extend(
        _lexical(path, role="consumed path") for path in reader.consumed_paths
    )
    if any(
        path == manifest or _path_within(path, cache_root)
        for path in observed_paths
    ) or any(
        purpose == "TRAIN_CACHE_MANIFEST"
        or (purpose.startswith("CANDIDATE13_") and purpose.endswith("_POOL"))
        for purpose in (
            purposes
            + list(reader.raw_fd_opened_purposes)
            + list(reader.verified_inode_purposes)
            + list(reader.read_started_purposes)
        )
    ):
        raise PermissionError("Candidate13 qualification reached manifest or pool state")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Candidate13 qualification record has duplicate keys")
        result[key] = value
    return result


def _validate_qualification_record(record: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            record.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise PermissionError("Candidate13 qualification record is invalid") from None
    if (
        type(value) is not dict
        or canonical_bytes(value) != record
        or sha256_bytes(record) != EXPECTED_QUALIFICATION_SHA256
        or value.get("status") != "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
        or value.get("check_code") != "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED"
        or value.get("artifact_source_inventory_opened") is not False
        or value.get("cache_manifest_opened") is not False
        or value.get("scientific_hypothesis_consumed") is not False
        or value.get("holdout_label_evaluator_metric_fit_opened_or_started") is not False
        or value.get("repeatable") is not True
    ):
        raise PermissionError("Candidate13 qualification record does not match authority")
    return types.MappingProxyType(dict(value))


def _build_qualification_bytes(runner: Any) -> bytes:
    functions = tuple(
        (
            runner.__dict__[name],
            runner.__dict__[name].__code__,
        )
        for name in (
            "bootstrap_verified_sources",
            "qualify_source_controls",
            "build_qualification_record",
        )
    )
    context = runner.bootstrap_verified_sources(
        runner_sha256=RUNNER_SHA256,
        runner_size_bytes=RUNNER_SIZE_BYTES,
        runner_tests_sha256=RUNNER_TESTS_SHA256,
        runner_tests_size_bytes=RUNNER_TESTS_SIZE_BYTES,
    )
    qualification = runner.qualify_source_controls(context)
    _validate_post_qualification_boundary(runner, context, qualification)
    record = runner.build_qualification_record(context, qualification)
    _validate_post_qualification_boundary(runner, context, qualification)
    _validate_qualification_record(record)
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
        raise PermissionError("Candidate13 qualification runner changed during use")
    return record


def _source_check_value(context: Any) -> dict[str, Any]:
    return {
        "artifact_or_control_input_opened": False,
        "check_code": "C13_QUALIFICATION_EMITTER_EXACT_SOURCES_VERIFIED",
        "qualification_emitted": False,
        "source_inventory_commitment_sha256": context.source_commitment_sha256,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--emitter-sha256", required=True)
    parser.add_argument("--emitter-size-bytes", required=True, type=int)
    parser.add_argument("--emitter-tests-sha256", required=True)
    parser.add_argument("--emitter-tests-size-bytes", required=True, type=int)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--emit-qualification-only", action="store_true")
    return parser.parse_args(argv)


def _private_main(argv: Sequence[str] | None = None) -> int:
    binding = _verified_emitter_binding()
    args = _parse_args(argv)
    if (
        binding["sha256"] != _require_sha(args.emitter_sha256, name="emitter SHA")
        or binding["size_bytes"]
        != _strict_positive_int(args.emitter_size_bytes, name="emitter size")
    ):
        raise PermissionError("Candidate13 emitter executing bytes changed")
    _read_exact(
        EMITTER_TESTS_SOURCE,
        role="Candidate13 emitter tests",
        sha256=_require_sha(args.emitter_tests_sha256, name="emitter tests SHA"),
        size_bytes=_strict_positive_int(
            args.emitter_tests_size_bytes, name="emitter tests size"
        ),
    )
    runner, token = _load_verified_runner()
    try:
        if args.source_check_only:
            context = runner.bootstrap_verified_sources(
                runner_sha256=RUNNER_SHA256,
                runner_size_bytes=RUNNER_SIZE_BYTES,
                runner_tests_sha256=RUNNER_TESTS_SHA256,
                runner_tests_size_bytes=RUNNER_TESTS_SIZE_BYTES,
            )
            output = canonical_bytes(_source_check_value(context))
        else:
            output = _build_qualification_bytes(runner)
    finally:
        _release_verified_runner(runner, token)
    written = sys.stdout.buffer.write(output)
    if written != len(output):
        raise OSError("Candidate13 qualification stdout write was incomplete")
    sys.stdout.buffer.flush()
    return 0


def _bootstrap_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"Candidate13 emitter requires exactly one {flag}")
    value = arguments[positions[0] + 1]
    if type(value) is not str or value.startswith("--"):
        raise ValueError(f"Candidate13 emitter has an invalid {flag}")
    return value


def _dispatch_verified_emitter(argv: Sequence[str] | None) -> int:
    _validate_import_hooks()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not all(type(value) is str for value in arguments):
        raise TypeError("Candidate13 emitter arguments must be exact strings")
    sha256 = _require_sha(
        _bootstrap_value(arguments, "--emitter-sha256"), name="emitter SHA"
    )
    size_text = _bootstrap_value(arguments, "--emitter-size-bytes")
    if not size_text.isdecimal():
        raise ValueError("Candidate13 emitter size is invalid")
    size_bytes = _strict_positive_int(int(size_text), name="emitter size")
    data = _read_exact(
        EMITTER_SOURCE,
        role="Candidate13 executing emitter",
        sha256=sha256,
        size_bytes=size_bytes,
    )
    outer = sys.modules.get(__name__)
    if type(outer) is not types.ModuleType:
        raise RuntimeError("Candidate13 outer emitter module is unavailable")
    namespace = object.__getattribute__(outer, "__dict__")
    if namespace.get("_EMITTER_OUTER_TOKEN") is not None:
        raise PermissionError("Candidate13 emitter bootstrap is not reentrant")
    name = f"_candidate13_qualification_emitter_{sha256}"
    if name in sys.modules:
        raise PermissionError("Candidate13 verified emitter is already loaded")
    token = object()
    module = types.ModuleType(name)
    module.__file__ = str(EMITTER_SOURCE)
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__dict__["_EMITTER_VERIFIED_BINDING"] = types.MappingProxyType(
        {"module_name": name, "sha256": sha256, "size_bytes": size_bytes}
    )
    module.__dict__["_EMITTER_LAUNCH_TOKEN"] = token
    module.__dict__["_EMITTER_OUTER_MODULE"] = outer
    namespace["_EMITTER_OUTER_TOKEN"] = token
    sys.modules[name] = module
    try:
        exec(
            compile(data, str(EMITTER_SOURCE), "exec", dont_inherit=True),
            module.__dict__,
        )
        entry = module.__dict__.get("_private_main")
        if (
            type(entry) is not types.FunctionType
            or entry.__module__ != name
            or entry.__code__.co_filename != str(EMITTER_SOURCE)
        ):
            raise PermissionError("Candidate13 emitter private entry mismatch")
        return int(entry(arguments))
    finally:
        sys.modules.pop(name, None)
        if namespace.get("_EMITTER_OUTER_TOKEN") is token:
            namespace.pop("_EMITTER_OUTER_TOKEN", None)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        if _EMITTER_VERIFIED_BINDING is None:
            return _dispatch_verified_emitter(argv)
        return _private_main(argv)
    except BaseException:
        sys.stderr.write(
            '{"check_code":"C13_QUALIFICATION_EMIT_REJECTED","status":"FAIL_CLOSED"}\n'
        )
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
