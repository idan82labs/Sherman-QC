#!/usr/bin/env python3
"""Verified, source-only adapter for Candidate8 value-free diagnostics.

This engineering adapter has no real cache, manifest, pool discovery, label,
evaluator, fit, publication, or holdout capability.  Its CLI can only verify
the exact six source files.  An eventual separately audited one-shot runner
may pass already-bound immutable semantic/pool/disposition bytes to
``diagnose_bound_pair`` without changing this adapter or the hypothesis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SOURCE = Path(__file__).resolve()
RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
RUNTIME_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
)
ADAPTER_TESTS_SOURCE = (
    ROOT / "tests/test_f1_train18_candidate8_diagnostic_adapter_v1.py"
)
HYPOTHESIS_SOURCE = (
    ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
)
CANDIDATE7_RUNTIME_SOURCE = (
    ROOT / "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py"
)
RUNTIME_MODULE = "backend.f1_train18_candidate8_diagnostic_recall_wrapper_v1"
HYPOTHESIS_MODULE = "backend.f1_train18_candidate6_conservative_recall_recovery_v1"
CANDIDATE7_RUNTIME_MODULE = (
    "backend.f1_train18_candidate7_conservative_recall_recovery_v1"
)

FORBIDDEN_PRELOADED_MODULES = (
    "backend",
    RUNTIME_MODULE,
    HYPOTHESIS_MODULE,
    CANDIDATE7_RUNTIME_MODULE,
    "numpy",
    "scipy",
    "sklearn",
    "scripts.evaluate_f1_against_manual_gold",
)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "bend_id",
        "candidate_count",
        "exception_class_name",
        "exception_message",
        "feature_value",
        "geometry",
        "line_object_count",
        "prediction_count",
        "predictions",
        "selector_score",
        "source_kind",
    }
)
_HEX = frozenset("0123456789abcdef")


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


def _reject_symlink_chain(path: Path, *, role: str) -> Path:
    candidate = _lexical_repo_path(path, role=role)
    root = _lexical_absolute(ROOT, role="repository root")
    chain: list[Path] = []
    current = candidate
    while current != root:
        chain.append(current)
        current = current.parent
    chain.append(root)
    for item in reversed(chain):
        metadata = os.lstat(item)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path or ancestor is a symlink")
        if item != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor is not a directory")
    return candidate


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _assert_visible_directory(
    path: Path, expected: tuple[int, int], *, role: str
) -> None:
    candidate = _reject_symlink_chain(path, role=role)
    metadata = os.lstat(candidate)
    if not stat.S_ISDIR(metadata.st_mode) or _identity(metadata) != expected:
        raise RuntimeError(f"{role} identity changed")


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlink_chain(path, role=role)
    before = os.lstat(candidate)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate8 adapter requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory)
    opened = os.fstat(descriptor)
    expected = _identity(before)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected:
        os.close(descriptor)
        raise RuntimeError(f"{role} changed during open")
    _assert_visible_directory(candidate, expected, role=f"{role} visible")
    return descriptor, expected


def _read_source_once(path: Path, *, role: str, sha256: str) -> tuple[bytes, dict[str, Any]]:
    candidate = _reject_symlink_chain(path, role=role)
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{role} must be one non-hard-linked regular file")
    parent_fd, parent_identity = _open_stable_directory(
        candidate.parent, role=f"{role} parent"
    )
    descriptor: int | None = None
    try:
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
            raise ValueError(f"{role} identity changed during open")
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
            _identity(after) != identity
            or _identity(visible) != identity
            or int(after.st_size) != len(data)
        ):
            raise ValueError(f"{role} changed during read")
        _assert_visible_directory(
            candidate.parent, parent_identity, role=f"{role} parent after read"
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if sha256_bytes(data) != _require_sha(sha256, name=f"{role} SHA"):
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, {
        "path": str(candidate.relative_to(ROOT)),
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


def _exec_module(name: str, path: Path, data: bytes) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError(f"Candidate8 verified module already loaded: {name}")
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
class AdapterContext:
    runtime: types.ModuleType
    hypothesis: types.ModuleType
    candidate7_runtime: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]


def bootstrap_verified_sources(
    *,
    runtime_sha256: str,
    adapter_sha256: str,
    runtime_tests_sha256: str,
    adapter_tests_sha256: str,
    hypothesis_sha256: str,
    candidate7_runtime_sha256: str,
) -> AdapterContext:
    specs = {
        "adapter": (ADAPTER_SOURCE, adapter_sha256),
        "adapter_tests": (ADAPTER_TESTS_SOURCE, adapter_tests_sha256),
        "candidate7_runtime": (
            CANDIDATE7_RUNTIME_SOURCE,
            candidate7_runtime_sha256,
        ),
        "hypothesis": (HYPOTHESIS_SOURCE, hypothesis_sha256),
        "runtime": (RUNTIME_SOURCE, runtime_sha256),
        "runtime_tests": (RUNTIME_TESTS_SOURCE, runtime_tests_sha256),
    }
    _reject_preloaded_sources(tuple(path for path, _sha in specs.values()))
    bindings: dict[str, dict[str, Any]] = {}
    source_bytes: dict[str, bytes] = {}
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha = specs[name]
        data, binding = _read_source_once(
            path, role=f"CANDIDATE8_SOURCE_{name.upper()}", sha256=sha
        )
        metadata = os.lstat(path)
        identity = _identity(metadata)
        if identity in identities:
            raise ValueError("Candidate8 source inputs contain a hard-link alias")
        identities.add(identity)
        bindings[name] = binding
        source_bytes[name] = data
    hypothesis = _exec_module(
        "_candidate8_verified_hypothesis", HYPOTHESIS_SOURCE, source_bytes["hypothesis"]
    )
    candidate7_runtime = _exec_module(
        "_candidate8_verified_candidate7_runtime",
        CANDIDATE7_RUNTIME_SOURCE,
        source_bytes["candidate7_runtime"],
    )
    runtime = _exec_module(
        "_candidate8_verified_runtime", RUNTIME_SOURCE, source_bytes["runtime"]
    )
    runtime.verify_scientific_hypothesis_runtime(
        hypothesis, source_sha256=hypothesis_sha256
    )
    runtime.verify_candidate7_wrapper_runtime(
        candidate7_runtime,
        source_sha256=candidate7_runtime_sha256,
        hypothesis_runtime=hypothesis,
    )
    if runtime.ATTEMPT_ORDINAL != 8:
        raise ValueError("Candidate8 verified runtime identity mismatch")
    return AdapterContext(
        runtime=runtime,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7_runtime,
        source_bindings=bindings,
    )


def _assert_value_free_result(result: Mapping[str, Any], runtime: types.ModuleType) -> None:
    runtime._validate_result(result)
    for key in result:
        if key.endswith("_exposed"):
            if result[key] is not False:
                raise ValueError("Candidate8 diagnostic exposure declaration is unsafe")
            continue
        if any(fragment in key for fragment in FORBIDDEN_RESULT_KEYS):
            raise ValueError("Candidate8 diagnostic result exposes a forbidden key")


def diagnose_bound_pair(
    context: AdapterContext,
    *,
    semantic_bytes: bytes,
    pool_bytes: bytes,
    candidate7_disposition_bytes: bytes,
    semantic_sha256: str,
    pool_sha256: str,
    candidate7_disposition_sha256: str,
) -> dict[str, Any]:
    if not all(
        isinstance(value, bytes)
        for value in (semantic_bytes, pool_bytes, candidate7_disposition_bytes)
    ):
        raise TypeError("Candidate8 adapter accepts immutable bytes only")
    result = context.runtime.diagnose_source_bytes(
        semantic_bytes,
        pool_bytes,
        candidate7_disposition_bytes,
        v51_semantic_sha256=_require_sha(semantic_sha256, name="semantic SHA"),
        pool_sha256=_require_sha(pool_sha256, name="pool SHA"),
        candidate7_disposition_sha256=_require_sha(
            candidate7_disposition_sha256, name="Candidate7 disposition SHA"
        ),
        hypothesis_runtime=context.hypothesis,
        hypothesis_source_sha256=context.source_bindings["hypothesis"]["sha256"],
        candidate7_runtime=context.candidate7_runtime,
        candidate7_runtime_source_sha256=context.source_bindings[
            "candidate7_runtime"
        ]["sha256"],
    )
    _assert_value_free_result(result, context.runtime)
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--runtime-tests-sha256", required=True)
    parser.add_argument("--adapter-tests-sha256", required=True)
    parser.add_argument("--hypothesis-sha256", required=True)
    parser.add_argument("--candidate7-runtime-sha256", required=True)
    parser.add_argument("--source-check-only", action="store_true", required=True)
    args = parser.parse_args()
    context = bootstrap_verified_sources(
        runtime_sha256=args.runtime_sha256,
        adapter_sha256=args.adapter_sha256,
        runtime_tests_sha256=args.runtime_tests_sha256,
        adapter_tests_sha256=args.adapter_tests_sha256,
        hypothesis_sha256=args.hypothesis_sha256,
        candidate7_runtime_sha256=args.candidate7_runtime_sha256,
    )
    print(
        json.dumps(
            {
                "artifact_discovery_or_open_capability_present": False,
                "artifact_input_opened": False,
                "cli_diagnostic_input_opened": False,
                "holdout_or_label_opened": False,
                "source_binding_count": len(context.source_bindings),
                "status": "SOURCE_BYTES_VERIFIED_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
