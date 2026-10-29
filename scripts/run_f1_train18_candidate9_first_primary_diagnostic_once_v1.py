"""Guarded Candidate9 first-primary representation diagnostic runner.

The runner has three explicit modes.  Source verification and repeatable
value-free control qualification never open the TRAIN cache manifest.  The
one-shot mode additionally requires an external authorization, durably consumes
it before the first manifest open, reads only the exact first-primary pair
already consumed by Candidate7/Candidate8, applies the lossless Candidate9 pool
projection, and invokes the unchanged Candidate8 D01--D05 diagnostic over the
adapted bytes.  It has no label, evaluator, metric, fit, model, full-28, replay,
later-pair, or holdout capability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
RUNNER_SOURCE = ROOT / "scripts/run_f1_train18_candidate9_first_primary_diagnostic_once_v1.py"
RUNNER_TESTS_SOURCE = ROOT / "tests/test_run_f1_train18_candidate9_first_primary_diagnostic_once_v1.py"
C9_ADAPTER_SOURCE = ROOT / "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C9_ADAPTER_TESTS_SOURCE = ROOT / "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py"
C8_SUPPORT_SOURCE = ROOT / "scripts/run_f1_train18_candidate8_first_primary_diagnostic_once_v1.py"
C8_RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate8_diagnostic_recall_wrapper_v1.py"
C8_ADAPTER_SOURCE = ROOT / "scripts/preflight_f1_train18_candidate8_diagnostic_adapter_v1.py"
C7_RUNTIME_SOURCE = ROOT / "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py"
C6_HYPOTHESIS_SOURCE = ROOT / "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py"
BUILDER_SOURCE = ROOT / "scripts/build_f1_scan_only_candidate_pool.py"
CLUSTER_SOURCE = ROOT / "backend/f1_line_object_clustering.py"

C8_DISPOSITION = ROOT / "research/f1_aug30_train18_candidate8_terminal_disposition_v1.json"
C8_RESULT = ROOT / "output/research/f1-aug30-train18-candidate8-first-primary-diagnostic-v1/diagnostic_result.json"
C8_LEDGER = ROOT / "output/research/f1-aug30-train18-candidate8-first-primary-diagnostic-v1/diagnostic_read_ledger.json"
C9_QUALIFICATION = ROOT / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_qualification_v1.json"
C9_AUTHORIZATION = ROOT / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_authorization_v1.json"
C9_ANCHOR = ROOT / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1.json"
OUTPUT_ROOT = ROOT / "output/research/f1-aug30-train18-candidate9-first-primary-diagnostic-v1"

ATTEMPT_ORDINAL = 9
ENGINEERING_ATTEMPT_ID = "CANDIDATE9_FIRST_PRIMARY_ADAPTER_DIAGNOSTIC_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
QUALIFICATION_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_qualification_v1"
AUTHORIZATION_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_authorization_v1"
ANCHOR_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1"
RESULT_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_result_v1"
TERMINAL_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_terminal_v1"
LEDGER_KIND = "sherman_qc_f1_aug30_train18_candidate9_first_primary_diagnostic_read_ledger_v1"
RECORD_SELECTOR = "EXACT_C7_C8_CONSUMED_FIRST_PRIMARY_BY_FROZEN_MANIFEST_ORDER"

FIXED_SOURCE_BINDINGS = {
    "builder": (BUILDER_SOURCE, "e270e44a06ad295c466152ed29b7313c687dce391ccfd75944d8bc203a83a3a2", 57613),
    "c6_hypothesis": (C6_HYPOTHESIS_SOURCE, "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252", 53837),
    "c7_runtime": (C7_RUNTIME_SOURCE, "a6bd7a23ead5bb07040a7bb1d02a8aab3588bf544b1ec1c9feaf534c9498f04d", 28098),
    "c8_adapter": (C8_ADAPTER_SOURCE, "b91a112356dba3ed4aa9e3fe51fe36ea12bf93d2dd9ab8900f2b3461509211be", 14807),
    "c8_runtime": (C8_RUNTIME_SOURCE, "d878eb9141f287435890ed3290a45f40dd0730130585a6846c38431e3342f9b8", 21360),
    "c8_support": (C8_SUPPORT_SOURCE, "941bdfbd5439a6a03594a90a604722444144fd0411b204b003f32a8e4a6772b0", 127350),
    "c9_adapter": (C9_ADAPTER_SOURCE, "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096", 22354),
    "c9_adapter_tests": (C9_ADAPTER_TESTS_SOURCE, "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623", 16598),
    "cluster": (CLUSTER_SOURCE, "1fbe2edeb24a90726b7936e40a32aae20cf587102a6df3f34f2ad71b10b2be5c", 6817),
}

C8_DISPOSITION_SHA256 = "e095c84cf616cda855f21843dfbddb69e6d6de38aa8cddde906a1a6aca87f331"
C8_DISPOSITION_SIZE_BYTES = 3930
C8_RESULT_SHA256 = "b804e09b07f5a44b2f80faa56c88d7f6ac8ee1ad8d57b67b8b9891c50f4e0ea1"
C8_RESULT_SIZE_BYTES = 990
C8_LEDGER_SHA256 = "5199e2a1eacad11ce8bad12a43057673c58529510657ef4e488ea98c2163b68e"
C8_LEDGER_SIZE_BYTES = 2262

FORBIDDEN_PRELOADED = frozenset(
    {
        "backend.f1_train18_candidate9_lossless_pool_alias_adapter_v1",
        "backend.f1_train18_candidate6_conservative_recall_recovery_v1",
        "backend.f1_train18_candidate7_conservative_recall_recovery_v1",
        "backend.f1_train18_candidate8_diagnostic_recall_wrapper_v1",
        "scripts.preflight_f1_train18_candidate8_diagnostic_adapter_v1",
        "scripts.run_f1_train18_candidate8_first_primary_diagnostic_once_v1",
        "scripts.build_f1_scan_only_candidate_pool",
        "backend.f1_line_object_clustering",
    }
)

_CONTEXT_SEAL = object()
_QUALIFICATION_SEAL = object()
_ISSUED_CONTEXTS: dict[int, tuple[object, object]] = {}
_ISSUED_QUALIFICATIONS: dict[int, tuple[object, object]] = {}
_ISSUED_AUTHORIZATIONS: dict[int, tuple[object, object]] = {}
_ISSUED_CONSUMPTIONS: dict[int, tuple[object, object]] = {}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


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


def _decode(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
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
    for item in reversed(chain):
        try:
            metadata = os.lstat(item)
        except FileNotFoundError:
            if require_exists or item != candidate:
                raise
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} path is a symlink")
        if item != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{role} ancestor is not a directory")
    return candidate


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _read_once(path: Path, *, role: str, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    candidate = _reject_symlinks(path, role=role)
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{role} must be one regular non-hard-linked file")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate9 requires O_NOFOLLOW/O_DIRECTORY")
    parent_fd = os.open(candidate.parent, os.O_RDONLY | nofollow | directory)
    descriptor: int | None = None
    try:
        parent_meta = os.fstat(parent_fd)
        visible_parent = os.lstat(candidate.parent)
        if not stat.S_ISDIR(parent_meta.st_mode) or _identity(parent_meta) != _identity(visible_parent):
            raise RuntimeError(f"{role} parent changed")
        descriptor = os.open(candidate.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != _identity(before):
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
        if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 or _identity(after) != identity or _identity(visible) != identity or int(after.st_size) != len(data):
            raise ValueError(f"{role} changed during read")
        if _identity(os.lstat(candidate.parent)) != _identity(parent_meta):
            raise RuntimeError(f"{role} parent changed after read")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    expected_sha = _require_sha(sha256, name=f"{role} SHA")
    if len(data) != size_bytes or sha256_bytes(data) != expected_sha:
        raise ValueError(f"{role} exact byte binding mismatch")
    return data, {
        "device_id": identity[0],
        "inode": identity[1],
        "path": str(candidate.relative_to(ROOT)),
        "purpose": role,
        "sha256": expected_sha,
        "size_bytes": len(data),
    }


def _reject_preloaded(paths: Sequence[Path]) -> None:
    current = sys.modules.get(__name__)
    targets = {path.resolve(strict=True) for path in paths}
    identities = {_identity(os.stat(path, follow_symlinks=True)) for path in paths}
    if any(name in sys.modules for name in FORBIDDEN_PRELOADED):
        raise RuntimeError("Candidate9 rejects a preloaded controlled module")
    for name, module in tuple(sys.modules.items()):
        if module is None or module is current:
            continue
        loaded = getattr(module, "__file__", None)
        if not isinstance(loaded, str) or not loaded:
            continue
        try:
            candidate = Path(loaded).resolve(strict=True)
            identity = _identity(os.stat(candidate))
        except OSError:
            continue
        if candidate in targets or identity in identities:
            raise RuntimeError(f"Candidate9 rejects preloaded source alias: {name}")


def _exec(name: str, path: Path, data: bytes, *, package: str) -> types.ModuleType:
    if name in sys.modules:
        raise RuntimeError("Candidate9 private verified module is already loaded")
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
    support: types.ModuleType
    c9_adapter: types.ModuleType
    c9_adapter_context: Any
    c8_adapter: types.ModuleType
    c8_adapter_context: Any
    c8_runtime: types.ModuleType
    c7_runtime: types.ModuleType
    c6_hypothesis: types.ModuleType
    builder_facade: types.ModuleType
    cluster_facade: types.ModuleType
    source_bindings: Mapping[str, Mapping[str, Any]]
    source_entries: tuple[dict[str, Any], ...]
    source_commitment_sha256: str
    _controlled_callables: tuple[Any, ...]
    _controlled_function_snapshot: tuple[Any, ...]
    _constant_commitment_sha256: str
    _nonce: object
    _sentinel: object
    _issuance_token: object


def _source_projection(bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: dict(value) for name, value in sorted(bindings.items())}


def _controlled_function_snapshot(*modules: types.ModuleType) -> tuple[Any, ...]:
    snapshot: list[tuple[str, str, Any, Any]] = []
    for module in modules:
        for name, value in sorted(vars(module).items()):
            if isinstance(value, types.FunctionType):
                snapshot.append((module.__name__, name, value, value.__code__))
            elif isinstance(value, type) and getattr(value, "__module__", None) == module.__name__:
                for child_name, child in sorted(vars(value).items()):
                    if isinstance(child, types.FunctionType):
                        snapshot.append(
                            (
                                f"{module.__name__}.{name}",
                                child_name,
                                child,
                                child.__code__,
                            )
                        )
    return tuple(snapshot)


def _controlled_constant_commitment(
    *,
    support: types.ModuleType,
    hypothesis: types.ModuleType,
    candidate7: types.ModuleType,
    c8_runtime: types.ModuleType,
    c9_adapter: types.ModuleType,
) -> str:
    fixed_controls = {
        name: {
            "path": str(_lexical(path, role=f"controlled constant {name}").relative_to(ROOT)),
            "sha256": sha,
            "size_bytes": size,
        }
        for name, (path, sha, size) in sorted(support.FIXED_CONTROLS.items())
    }
    value = {
        "candidate7_attempt_ordinal": candidate7.ATTEMPT_ORDINAL,
        "candidate7_scientific_hypothesis_id": candidate7.SCIENTIFIC_HYPOTHESIS_ID,
        "candidate8_failure_codes": dict(sorted(c8_runtime.FAILURE_CODES.items())),
        "candidate8_pass_code": c8_runtime.PASS_CODE,
        "candidate8_phase_contract": list(c8_runtime.PHASE_CONTRACT),
        "candidate9_adapter": {
            "authoritative_source_field": c9_adapter.AUTHORITATIVE_SOURCE_FIELD,
            "auxiliary_geometry_fields": list(c9_adapter.AUXILIARY_GEOMETRY_FIELDS),
            "candidate8_disposition_sha256": c9_adapter.CANDIDATE8_DISPOSITION_SHA256,
            "pool_algorithm": c9_adapter.POOL_ALGORITHM,
            "pool_schema_version": c9_adapter.POOL_SCHEMA_VERSION,
            "representation_contract": dict(c9_adapter.REPRESENTATION_CONTRACT),
            "stale_alias_field": c9_adapter.STALE_ALIAS_FIELD,
            "target_candidate_source": c9_adapter.TARGET_CANDIDATE_SOURCE,
            "working_point_budget": c9_adapter.WORKING_POINT_BUDGET,
        },
        "hypothesis": {
            "allowed_source_kinds": sorted(hypothesis.ALLOWED_SOURCE_KINDS),
            "attempt_ordinal": hypothesis.ATTEMPT_ORDINAL,
            "max_additions_per_case": hypothesis.MAX_ADDITIONS_PER_CASE,
            "ownership_transverse_offset_mm": hypothesis.OWNERSHIP_TRANSVERSE_OFFSET_MM,
            "pool_inference_contract": hypothesis.POOL_INFERENCE_CONTRACT,
        },
        "support": {
            "allowed_cache_root": str(
                _lexical(support.ALLOWED_CACHE_ROOT, role="controlled cache root").relative_to(ROOT)
            ),
            "cache_manifest_sha256": support.CACHE_MANIFEST_SHA256,
            "cache_manifest_size_bytes": support.CACHE_MANIFEST_SIZE_BYTES,
            "fixed_controls": fixed_controls,
        },
    }
    return sha256_bytes(canonical_bytes(value))


def bootstrap_verified_sources(*, runner_sha256: str, runner_size_bytes: int, runner_tests_sha256: str, runner_tests_size_bytes: int) -> BootstrapContext:
    specs = {
        **FIXED_SOURCE_BINDINGS,
        "runner": (RUNNER_SOURCE, _require_sha(runner_sha256, name="runner SHA"), _strict_int(runner_size_bytes, name="runner size")),
        "runner_tests": (RUNNER_TESTS_SOURCE, _require_sha(runner_tests_sha256, name="runner tests SHA"), _strict_int(runner_tests_size_bytes, name="runner tests size")),
    }
    if specs["runner"][2] <= 0 or specs["runner_tests"][2] <= 0:
        raise ValueError("Candidate9 source sizes must be positive")
    _reject_preloaded(tuple(path for path, _sha, _size in specs.values()))
    source_bytes: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for name in sorted(specs):
        path, sha, size = specs[name]
        data, entry = _read_once(path, role=f"CANDIDATE9_SOURCE_{name.upper()}", sha256=sha, size_bytes=size)
        identity = int(entry["device_id"]), int(entry["inode"])
        if identity in identities:
            raise ValueError("Candidate9 source inventory contains an inode alias")
        identities.add(identity)
        source_bytes[name] = data
        entries.append(entry)
        bindings[name] = {"path": entry["path"], "sha256": sha, "size_bytes": size}

    support = _exec("_candidate9_verified_c8_support", C8_SUPPORT_SOURCE, source_bytes["c8_support"], package="scripts")
    hypothesis = _exec("_candidate9_verified_c6_hypothesis", C6_HYPOTHESIS_SOURCE, source_bytes["c6_hypothesis"], package="backend")
    candidate7 = _exec("_candidate9_verified_c7_runtime", C7_RUNTIME_SOURCE, source_bytes["c7_runtime"], package="backend")
    c8_runtime = _exec("_candidate9_verified_c8_runtime", C8_RUNTIME_SOURCE, source_bytes["c8_runtime"], package="backend")
    c8_adapter = _exec("_candidate9_verified_c8_adapter", C8_ADAPTER_SOURCE, source_bytes["c8_adapter"], package="scripts")
    c9_adapter = _exec("_candidate9_verified_c9_adapter", C9_ADAPTER_SOURCE, source_bytes["c9_adapter"], package="backend")
    c8_runtime.verify_scientific_hypothesis_runtime(hypothesis, source_sha256=bindings["c6_hypothesis"]["sha256"])
    c8_runtime.verify_candidate7_wrapper_runtime(candidate7, source_sha256=bindings["c7_runtime"]["sha256"], hypothesis_runtime=hypothesis)
    c8_adapter_context = c8_adapter.AdapterContext(
        runtime=c8_runtime,
        hypothesis=hypothesis,
        candidate7_runtime=candidate7,
        source_bindings={
            "hypothesis": dict(bindings["c6_hypothesis"]),
            "candidate7_runtime": dict(bindings["c7_runtime"]),
        },
    )
    builder_proxy = types.ModuleType("_candidate9_authenticated_builder_proxy")
    builder_proxy._tag = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("builder execution is outside Candidate9 runtime"))
    cluster_proxy = types.ModuleType("_candidate9_authenticated_cluster_proxy")
    cluster_proxy.cluster_line_candidates = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cluster execution is outside Candidate9 runtime"))
    c9_adapter_context = c9_adapter.make_verified_context(
        hypothesis=hypothesis,
        builder=builder_proxy,
        clusterer=cluster_proxy,
        hypothesis_source_sha256=bindings["c6_hypothesis"]["sha256"],
        builder_source_sha256=bindings["builder"]["sha256"],
        cluster_source_sha256=bindings["cluster"]["sha256"],
    )
    if c9_adapter.ATTEMPT_ORDINAL != 9 or c9_adapter.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID:
        raise ValueError("Candidate9 adapter identity mismatch")
    source_commitment = sha256_bytes(canonical_bytes(_source_projection(bindings)))
    immutable_bindings = types.MappingProxyType(
        {
            name: types.MappingProxyType(dict(binding))
            for name, binding in bindings.items()
        }
    )
    immutable_entries = tuple(types.MappingProxyType(dict(entry)) for entry in entries)
    issuance_token = object()
    context = BootstrapContext(
        support=support,
        c9_adapter=c9_adapter,
        c9_adapter_context=c9_adapter_context,
        c8_adapter=c8_adapter,
        c8_adapter_context=c8_adapter_context,
        c8_runtime=c8_runtime,
        c7_runtime=candidate7,
        c6_hypothesis=hypothesis,
        builder_facade=builder_proxy,
        cluster_facade=cluster_proxy,
        source_bindings=immutable_bindings,
        source_entries=immutable_entries,
        source_commitment_sha256=source_commitment,
        _controlled_callables=(
            support.qualify_source_controls,
            support._validate_manifest_first_primary,
            support.GuardedReader,
            c8_adapter.diagnose_bound_pair,
            c8_runtime._validate_result,
            c9_adapter.adapt_pool_bytes,
        ),
        _controlled_function_snapshot=_controlled_function_snapshot(
            support,
            hypothesis,
            candidate7,
            c8_runtime,
            c8_adapter,
            c9_adapter,
        ),
        _constant_commitment_sha256=_controlled_constant_commitment(
            support=support,
            hypothesis=hypothesis,
            candidate7=candidate7,
            c8_runtime=c8_runtime,
            c9_adapter=c9_adapter,
        ),
        _nonce=object(),
        _sentinel=_CONTEXT_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_CONTEXTS[id(context)] = (context, issuance_token)
    return context


def _binding(path: Path, sha256: str, size_bytes: int, *, kind: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(_lexical(path, role="binding").relative_to(ROOT)),
        "sha256": _require_sha(sha256, name="binding SHA"),
        "size_bytes": _strict_int(size_bytes, name="binding size"),
    }
    if kind is not None:
        value["kind"] = kind
    return value


@dataclass(frozen=True)
class ControlQualification:
    base: Any
    reader: Any
    c8_disposition_bytes: bytes
    c8_disposition: Mapping[str, Any]
    c8_first_primary_commitment_sha256: str
    control_commitment_sha256: str
    base_contract_commitment_sha256: str
    control_read_inventory_commitment_sha256: str
    control_raw_fd_opened_commitment_sha256: str
    control_verified_inode_commitment_sha256: str
    control_read_started_commitment_sha256: str
    control_entry_prefix_length: int
    control_open_prefix_length: int
    _base_pair_capability_nonce: object
    _context_nonce: object
    _sentinel: object
    _issuance_token: object


def _read_bound(reader: Any, *, path: Path, purpose: str, sha256: str, size_bytes: int) -> tuple[bytes, dict[str, Any]]:
    reader.authorize(path, purpose=purpose, sha256=sha256, size_bytes=size_bytes)
    data = reader.read_bytes(path, purpose=purpose)
    return data, _decode(data, name=purpose)


def _base_qualification_commitment(base: Any) -> str:
    value = {
        "c7_pool_evidence": dict(base.c7_pool_evidence),
        "c7_semantic_evidence": dict(base.c7_semantic_evidence),
        "candidate7_disposition_sha256": sha256_bytes(base.candidate7_disposition_bytes),
        "control_binding_commitment_sha256": base.control_binding_commitment_sha256,
        "manifest_binding": _binding(
            base.manifest_path,
            base.manifest_sha256,
            base.manifest_size_bytes,
        ),
        "scan_by_case": dict(sorted(base.scan_by_case.items())),
        "train_commitment": base.train_commitment,
        "train_ids": list(base.train_ids),
    }
    return sha256_bytes(canonical_bytes(value))


def _support_context(context: BootstrapContext) -> Any:
    return context.support.BootstrapContext(
        runtime=context.c8_runtime,
        adapter=context.c8_adapter,
        hypothesis=context.c8_adapter_context.hypothesis,
        candidate7_runtime=context.c8_adapter_context.candidate7_runtime,
        adapter_context=context.c8_adapter_context,
        source_bindings=context.source_bindings,
        bootstrap_entries=context.source_entries,
        _context_nonce=context._nonce,
    )


def _validate_c8_disposition(value: Mapping[str, Any], context: BootstrapContext) -> None:
    expected_sources = {
        "adapter": {
            "path": "scripts/preflight_f1_train18_candidate8_diagnostic_adapter_v1.py",
            "sha256": FIXED_SOURCE_BINDINGS["c8_adapter"][1],
            "size_bytes": FIXED_SOURCE_BINDINGS["c8_adapter"][2],
        },
        "candidate7_runtime": {
            "path": "backend/f1_train18_candidate7_conservative_recall_recovery_v1.py",
            "sha256": FIXED_SOURCE_BINDINGS["c7_runtime"][1],
            "size_bytes": FIXED_SOURCE_BINDINGS["c7_runtime"][2],
        },
        "hypothesis": {
            "path": "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
            "sha256": FIXED_SOURCE_BINDINGS["c6_hypothesis"][1],
            "size_bytes": FIXED_SOURCE_BINDINGS["c6_hypothesis"][2],
        },
        "runner": {
            "path": "scripts/run_f1_train18_candidate8_first_primary_diagnostic_once_v1.py",
            "sha256": FIXED_SOURCE_BINDINGS["c8_support"][1],
            "size_bytes": FIXED_SOURCE_BINDINGS["c8_support"][2],
        },
        "runtime": {
            "path": "backend/f1_train18_candidate8_diagnostic_recall_wrapper_v1.py",
            "sha256": FIXED_SOURCE_BINDINGS["c8_runtime"][1],
            "size_bytes": FIXED_SOURCE_BINDINGS["c8_runtime"][2],
        },
    }
    sources = value.get("source_bindings")
    if not isinstance(sources, dict):
        raise ValueError("Candidate8 disposition source bindings are absent")
    for name, expected in expected_sources.items():
        if sources.get(name) != expected:
            raise ValueError("Candidate8 disposition source lineage mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "sherman_qc_f1_aug30_train18_candidate8_terminal_disposition_v1"
        or value.get("status") != "TERMINAL_D02_DIAGNOSTIC_NONAUTHORIZING"
        or value.get("attempt_ordinal") != 8
        or value.get("check_code") != "C8_POOL_PARSE_SANITIZE_FAILED"
        or value.get("diagnostic_failure_phase") != "D02_POOL_PARSE_SANITIZE"
        or value.get("first_primary_pair_only") is not True
        or value.get("replay_or_later_pair_opened") is not False
        or value.get("labels_opened") is not False
        or value.get("metric_computed") is not False
        or value.get("fit_performed") is not False
        or value.get("holdout_opened") is not False
        or value.get("retry_authorized") is not False
        or value.get("continuation_authorized") is not False
        or value.get("scientific_hypothesis_performance_tested") is not False
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("diagnostic_result_binding")
        != _binding(C8_RESULT, C8_RESULT_SHA256, C8_RESULT_SIZE_BYTES)
        or value.get("diagnostic_read_ledger_binding")
        != _binding(C8_LEDGER, C8_LEDGER_SHA256, C8_LEDGER_SIZE_BYTES)
    ):
        raise ValueError("Candidate8 disposition contract mismatch")
    if context.c9_adapter.CANDIDATE8_DISPOSITION_SHA256 != C8_DISPOSITION_SHA256:
        raise ValueError("Candidate9 adapter disposition binding mismatch")


def qualify_source_controls(context: BootstrapContext) -> ControlQualification:
    _validate_context_integrity(context)
    support_context = _support_context(context)
    base = context.support.qualify_source_controls(support_context)
    reader = base.reader
    disposition_bytes, disposition = _read_bound(
        reader,
        path=C8_DISPOSITION,
        purpose="CANDIDATE8_TERMINAL_DISPOSITION",
        sha256=C8_DISPOSITION_SHA256,
        size_bytes=C8_DISPOSITION_SIZE_BYTES,
    )
    result_bytes, result = _read_bound(
        reader,
        path=C8_RESULT,
        purpose="CANDIDATE8_DIAGNOSTIC_RESULT",
        sha256=C8_RESULT_SHA256,
        size_bytes=C8_RESULT_SIZE_BYTES,
    )
    ledger_bytes, ledger = _read_bound(
        reader,
        path=C8_LEDGER,
        purpose="CANDIDATE8_DIAGNOSTIC_LEDGER",
        sha256=C8_LEDGER_SHA256,
        size_bytes=C8_LEDGER_SIZE_BYTES,
    )
    _validate_c8_disposition(disposition, context)
    context.support._assert_public_payload(result)
    context.support._assert_value_free_ledger(ledger)
    assertions = ledger.get("assertions")
    c8_first_primary_commitment = _require_sha(
        ledger.get("first_primary_binding_commitment_sha256"),
        name="Candidate8 first-primary commitment",
    )
    if (
        result.get("check_code") != "C8_POOL_PARSE_SANITIZE_FAILED"
        or result.get("diagnostic_failure_phase") != "D02_POOL_PARSE_SANITIZE"
        or result.get("status") != "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"
        or not isinstance(assertions, dict)
        or assertions.get("manifest_opened") is not True
        or assertions.get("manifest_inode_verified") is not True
        or assertions.get("manifest_read_started") is not True
        or assertions.get("only_first_primary_semantic_opened") is not True
        or assertions.get("only_first_primary_semantic_inode_verified") is not True
        or assertions.get("only_first_primary_semantic_read_started") is not True
        or assertions.get("only_first_primary_pool_opened") is not True
        or assertions.get("only_first_primary_pool_inode_verified") is not True
        or assertions.get("only_first_primary_pool_read_started") is not True
        or assertions.get("replay_or_later_pair_opened") is not False
        or assertions.get("labels_opened") is not False
        or assertions.get("evaluator_metric_fit_opened_or_started") is not False
        or ledger.get("check_code") != "C8_POOL_PARSE_SANITIZE_FAILED"
        or ledger.get("diagnostic_failure_phase") != "D02_POOL_PARSE_SANITIZE"
        or result.get("diagnostic_read_ledger_sha256") != sha256_bytes(ledger_bytes)
    ):
        raise ValueError("Candidate8 consumed-first-pair evidence mismatch")
    expected_source_purposes = [str(entry["purpose"]) for entry in context.source_entries]
    expected_control_purposes = _expected_control_purposes(context)
    observed_purposes = [str(entry["purpose"]) for entry in reader.entries]
    observed_paths = [str(entry["path"]) for entry in reader.entries]
    observed_identities = [
        (int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries
    ]
    if (
        observed_purposes != expected_source_purposes + expected_control_purposes
        or reader.raw_fd_opened_purposes != expected_control_purposes
        or reader.verified_inode_purposes != expected_control_purposes
        or reader.read_started_purposes != expected_control_purposes
        or len(observed_paths) != len(set(observed_paths))
        or len(observed_identities) != len(set(observed_identities))
        or reader.denied_attempts
    ):
        raise PermissionError("Candidate9 exact source-control read inventory mismatch")
    controls_projection = {
        "base_control_commitment_sha256": base.control_binding_commitment_sha256,
        "candidate8_disposition": _binding(C8_DISPOSITION, C8_DISPOSITION_SHA256, C8_DISPOSITION_SIZE_BYTES),
        "candidate8_first_primary_binding_commitment_sha256": c8_first_primary_commitment,
        "control_raw_fd_opened_inventory_commitment_sha256": _purpose_vector_commitment(
            reader.raw_fd_opened_purposes
        ),
        "control_read_inventory_commitment_sha256": _entry_commitment(reader.entries),
        "control_read_started_inventory_commitment_sha256": _purpose_vector_commitment(
            reader.read_started_purposes
        ),
        "control_verified_inode_inventory_commitment_sha256": _purpose_vector_commitment(
            reader.verified_inode_purposes
        ),
        "candidate8_ledger": _binding(C8_LEDGER, C8_LEDGER_SHA256, C8_LEDGER_SIZE_BYTES),
        "candidate8_result": _binding(C8_RESULT, C8_RESULT_SHA256, C8_RESULT_SIZE_BYTES),
    }
    commitment = sha256_bytes(canonical_bytes(controls_projection))
    issuance_token = object()
    qualification = ControlQualification(
        base=base,
        reader=reader,
        c8_disposition_bytes=disposition_bytes,
        c8_disposition=disposition,
        c8_first_primary_commitment_sha256=c8_first_primary_commitment,
        control_commitment_sha256=commitment,
        base_contract_commitment_sha256=_base_qualification_commitment(base),
        control_read_inventory_commitment_sha256=_entry_commitment(reader.entries),
        control_raw_fd_opened_commitment_sha256=_purpose_vector_commitment(
            reader.raw_fd_opened_purposes
        ),
        control_verified_inode_commitment_sha256=_purpose_vector_commitment(
            reader.verified_inode_purposes
        ),
        control_read_started_commitment_sha256=_purpose_vector_commitment(
            reader.read_started_purposes
        ),
        control_entry_prefix_length=len(reader.entries),
        control_open_prefix_length=len(reader.raw_fd_opened_purposes),
        _base_pair_capability_nonce=base._pair_capability_nonce,
        _context_nonce=context._nonce,
        _sentinel=_QUALIFICATION_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_QUALIFICATIONS[id(qualification)] = (qualification, issuance_token)
    return qualification


def _validate_control_qualification(
    context: BootstrapContext, qualification: ControlQualification
) -> None:
    issued = _ISSUED_QUALIFICATIONS.get(id(qualification))
    if (
        type(qualification) is not ControlQualification
        or qualification._sentinel is not _QUALIFICATION_SEAL
        or qualification._context_nonce is not context._nonce
        or issued is None
        or issued[0] is not qualification
        or issued[1] is not qualification._issuance_token
        or qualification.base.reader is not qualification.reader
        or qualification.base._pair_capability_nonce
        is not qualification._base_pair_capability_nonce
        or qualification.reader.first_primary_nonce
        is not qualification._base_pair_capability_nonce
        or qualification.base.control_binding_commitment_sha256
        != _require_sha(
            qualification.base.control_binding_commitment_sha256,
            name="Candidate8 base control commitment",
        )
        or sha256_bytes(qualification.c8_disposition_bytes) != C8_DISPOSITION_SHA256
        or qualification.c8_first_primary_commitment_sha256
        != _require_sha(
            qualification.c8_first_primary_commitment_sha256,
            name="Candidate9 qualified first-primary commitment",
        )
        or qualification.base_contract_commitment_sha256
        != _base_qualification_commitment(qualification.base)
        or qualification.control_read_inventory_commitment_sha256
        != _entry_commitment(
            qualification.reader.entries[: qualification.control_entry_prefix_length]
        )
        or qualification.control_raw_fd_opened_commitment_sha256
        != _purpose_vector_commitment(
            qualification.reader.raw_fd_opened_purposes[
                : qualification.control_open_prefix_length
            ]
        )
        or qualification.control_verified_inode_commitment_sha256
        != _purpose_vector_commitment(
            qualification.reader.verified_inode_purposes[
                : qualification.control_open_prefix_length
            ]
        )
        or qualification.control_read_started_commitment_sha256
        != _purpose_vector_commitment(
            qualification.reader.read_started_purposes[
                : qualification.control_open_prefix_length
            ]
        )
        or qualification.control_entry_prefix_length <= 0
        or qualification.control_open_prefix_length <= 0
        or len(qualification.reader.entries) < qualification.control_entry_prefix_length
        or len(qualification.reader.raw_fd_opened_purposes)
        < qualification.control_open_prefix_length
        or len(qualification.reader.verified_inode_purposes)
        < qualification.control_open_prefix_length
        or len(qualification.reader.read_started_purposes)
        < qualification.control_open_prefix_length
    ):
        raise PermissionError("Candidate9 control qualification seal is invalid")
    controls_projection = {
        "base_control_commitment_sha256": qualification.base.control_binding_commitment_sha256,
        "candidate8_disposition": _binding(C8_DISPOSITION, C8_DISPOSITION_SHA256, C8_DISPOSITION_SIZE_BYTES),
        "candidate8_first_primary_binding_commitment_sha256": qualification.c8_first_primary_commitment_sha256,
        "control_raw_fd_opened_inventory_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
        "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
        "control_read_started_inventory_commitment_sha256": qualification.control_read_started_commitment_sha256,
        "control_verified_inode_inventory_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
        "candidate8_ledger": _binding(C8_LEDGER, C8_LEDGER_SHA256, C8_LEDGER_SIZE_BYTES),
        "candidate8_result": _binding(C8_RESULT, C8_RESULT_SHA256, C8_RESULT_SIZE_BYTES),
    }
    if qualification.control_commitment_sha256 != sha256_bytes(
        canonical_bytes(controls_projection)
    ):
        raise PermissionError("Candidate9 control qualification commitment changed")


def build_qualification_record(context: BootstrapContext, qualification: ControlQualification) -> bytes:
    _validate_context_integrity(context)
    _validate_control_qualification(context, qualification)
    value = {
        "artifact_source_inventory_opened": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "cache_manifest_opened": False,
        "candidate8_consumed_first_pair_evidence_validated": True,
        "candidate8_first_primary_binding_commitment_sha256": qualification.c8_first_primary_commitment_sha256,
        "check_code": "ALL_C9_SOURCE_CONTROL_CHECKS_PASSED",
        "consumption_anchor_namespace": str(C9_ANCHOR.relative_to(ROOT)),
        "control_binding_commitment_sha256": qualification.control_commitment_sha256,
        "control_raw_fd_opened_inventory_commitment_sha256": qualification.control_raw_fd_opened_commitment_sha256,
        "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
        "control_read_started_inventory_commitment_sha256": qualification.control_read_started_commitment_sha256,
        "control_verified_inode_inventory_commitment_sha256": qualification.control_verified_inode_commitment_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "kind": QUALIFICATION_KIND,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "repeatable": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": _source_projection(context.source_bindings),
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }
    return canonical_bytes(value)


def qualification_result(context: BootstrapContext) -> dict[str, Any]:
    try:
        qualification = qualify_source_controls(context)
        record = build_qualification_record(context, qualification)
        return {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "candidate8_consumed_first_pair_evidence_validated": True,
            "check_code": "ALL_C9_SOURCE_CONTROL_CHECKS_PASSED",
            "control_binding_commitment_sha256": qualification.control_commitment_sha256,
            "failure_stage": None,
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "qualification_record_sha256": sha256_bytes(record),
            "qualification_record_size_bytes": len(record),
            "repeatable": True,
            "scientific_hypothesis_consumed": False,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        }
    except BaseException:
        return {
            "artifact_source_inventory_opened": False,
            "check_code": "C9_SOURCE_CONTROL_QUALIFICATION_FAILED",
            "failure_stage": "SOURCE_CONTROL_QUALIFICATION",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "repeatable": True,
            "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
        }


_SEAL = object()


@dataclass(frozen=True)
class FreshOutputSeal:
    path: Path
    parent_identity: tuple[int, int]
    _sentinel: object


@dataclass(frozen=True)
class AuthorizationSeal:
    reader: Any
    authorization_sha256: str
    authorization_size_bytes: int
    qualification_sha256: str
    qualification_size_bytes: int
    control_commitment_sha256: str
    source_commitment_sha256: str
    authority_read_inventory_commitment_sha256: str
    authority_raw_fd_opened_commitment_sha256: str
    authority_verified_inode_commitment_sha256: str
    authority_read_started_commitment_sha256: str
    anchor_parent_identity: tuple[int, int]
    output_freshness: FreshOutputSeal
    _context_nonce: object
    _sentinel: object
    _issuance_token: object


@dataclass(frozen=True)
class ConsumptionSeal:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int
    file_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    _sentinel: object
    _issuance_token: object


def _validate_output_namespace(path: Path = OUTPUT_ROOT) -> Path:
    candidate = _lexical(path, role="Candidate9 output")
    if candidate != _lexical(OUTPUT_ROOT, role="frozen Candidate9 output"):
        raise ValueError("Candidate9 output namespace is frozen")
    parent_fd, _parent_identity = _open_stable_directory(candidate.parent, role="Candidate9 output parent")
    try:
        names = os.listdir(parent_fd)
        if candidate.name in names or any(
            name.startswith(f".{candidate.name}.staging-")
            or name.startswith(f".{candidate.name}.quarantine-")
            for name in names
        ):
            raise FileExistsError("Candidate9 output namespace is not fresh")
    finally:
        os.close(parent_fd)
    return candidate


def _open_stable_directory(path: Path, *, role: str) -> tuple[int, tuple[int, int]]:
    candidate = _reject_symlinks(path, role=role)
    before = os.lstat(candidate)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate9 requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(candidate, os.O_RDONLY | nofollow | directory)
    opened = os.fstat(descriptor)
    expected = _identity(before)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected or _identity(os.lstat(candidate)) != expected:
        os.close(descriptor)
        raise RuntimeError(f"{role} changed during open")
    return descriptor, expected


def prevalidate_fresh_output(path: Path = OUTPUT_ROOT) -> FreshOutputSeal:
    candidate = _validate_output_namespace(path)
    parent_fd, identity = _open_stable_directory(candidate.parent, role="Candidate9 output parent seal")
    os.close(parent_fd)
    return FreshOutputSeal(path=candidate, parent_identity=identity, _sentinel=_SEAL)


def revalidate_fresh_output(seal: FreshOutputSeal) -> None:
    if not isinstance(seal, FreshOutputSeal) or seal._sentinel is not _SEAL:
        raise PermissionError("Candidate9 output freshness seal is invalid")
    candidate = _validate_output_namespace(seal.path)
    parent_fd, identity = _open_stable_directory(candidate.parent, role="Candidate9 output parent revalidation")
    os.close(parent_fd)
    if identity != seal.parent_identity:
        raise RuntimeError("Candidate9 output parent changed")


def build_authorization_contract(
    context: BootstrapContext,
    *,
    qualification_sha256: str,
    qualification_size_bytes: int,
    control_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "authorization_consumed_before_manifest_open": True,
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "candidate8_disposition_binding": _binding(C8_DISPOSITION, C8_DISPOSITION_SHA256, C8_DISPOSITION_SIZE_BYTES),
        "consumption_anchor_namespace": str(C9_ANCHOR.relative_to(ROOT)),
        "continuation_authorized": False,
        "control_binding_commitment_sha256": _require_sha(control_commitment_sha256, name="control commitment"),
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "full28_access_authorized": False,
        "holdout_label_evaluator_metric_fit_access_authorized": False,
        "kind": AUTHORIZATION_KIND,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "qualification_record_binding": _binding(C9_QUALIFICATION, qualification_sha256, qualification_size_bytes, kind=QUALIFICATION_KIND),
        "record_selector": RECORD_SELECTOR,
        "replay_or_later_pair_access_authorized": False,
        "retry_after_anchor_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": _source_projection(context.source_bindings),
        "status": "AUTHORIZED_ONCE",
    }


def authorize_diagnostic_once(
    context: BootstrapContext,
    *,
    authorization_sha256: str,
    authorization_size_bytes: int,
    output_freshness: FreshOutputSeal,
) -> AuthorizationSeal:
    _validate_context_integrity(context)
    revalidate_fresh_output(output_freshness)
    reader = context.support.GuardedReader(entries=[dict(entry) for entry in context.source_entries])
    auth_bytes, auth = _read_bound(
        reader,
        path=C9_AUTHORIZATION,
        purpose="CANDIDATE9_DIAGNOSTIC_AUTHORIZATION",
        sha256=_require_sha(authorization_sha256, name="Candidate9 authorization SHA"),
        size_bytes=_strict_int(authorization_size_bytes, name="Candidate9 authorization size"),
    )
    qualification_binding = auth.get("qualification_record_binding")
    if not isinstance(qualification_binding, dict):
        raise ValueError("Candidate9 authorization qualification binding is absent")
    qualification_sha = _require_sha(qualification_binding.get("sha256"), name="qualification SHA")
    qualification_size = _strict_int(qualification_binding.get("size_bytes"), name="qualification size")
    qualification_bytes, qualification = _read_bound(
        reader,
        path=C9_QUALIFICATION,
        purpose="CANDIDATE9_DIAGNOSTIC_QUALIFICATION",
        sha256=qualification_sha,
        size_bytes=qualification_size,
    )
    control_commitment = _require_sha(
        qualification.get("control_binding_commitment_sha256"),
        name="qualification control commitment",
    )
    qualified_first_primary_commitment = _require_sha(
        qualification.get("candidate8_first_primary_binding_commitment_sha256"),
        name="qualification first-primary commitment",
    )
    qualified_control_raw_commitment = _require_sha(
        qualification.get("control_raw_fd_opened_inventory_commitment_sha256"),
        name="qualification control raw-open commitment",
    )
    qualified_control_read_commitment = _require_sha(
        qualification.get("control_read_inventory_commitment_sha256"),
        name="qualification control read commitment",
    )
    qualified_control_started_commitment = _require_sha(
        qualification.get("control_read_started_inventory_commitment_sha256"),
        name="qualification control read-start commitment",
    )
    qualified_control_verified_commitment = _require_sha(
        qualification.get("control_verified_inode_inventory_commitment_sha256"),
        name="qualification control verified-inode commitment",
    )
    expected_auth = build_authorization_contract(
        context,
        qualification_sha256=qualification_sha,
        qualification_size_bytes=qualification_size,
        control_commitment_sha256=control_commitment,
    )
    expected_qualification = {
        "artifact_source_inventory_opened": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "cache_manifest_opened": False,
        "candidate8_consumed_first_pair_evidence_validated": True,
        "candidate8_first_primary_binding_commitment_sha256": qualified_first_primary_commitment,
        "check_code": "ALL_C9_SOURCE_CONTROL_CHECKS_PASSED",
        "consumption_anchor_namespace": str(C9_ANCHOR.relative_to(ROOT)),
        "control_binding_commitment_sha256": control_commitment,
        "control_raw_fd_opened_inventory_commitment_sha256": qualified_control_raw_commitment,
        "control_read_inventory_commitment_sha256": qualified_control_read_commitment,
        "control_read_started_inventory_commitment_sha256": qualified_control_started_commitment,
        "control_verified_inode_inventory_commitment_sha256": qualified_control_verified_commitment,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "kind": QUALIFICATION_KIND,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "repeatable": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": _source_projection(context.source_bindings),
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }
    if auth != expected_auth or qualification != expected_qualification or sha256_bytes(auth_bytes) != authorization_sha256 or sha256_bytes(qualification_bytes) != qualification_sha:
        raise ValueError("Candidate9 authorization/qualification contract mismatch")
    expected_purposes = [
        "CANDIDATE9_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE9_DIAGNOSTIC_QUALIFICATION",
    ]
    if (
        reader.raw_fd_opened_purposes != expected_purposes
        or reader.verified_inode_purposes != expected_purposes
        or reader.read_started_purposes != expected_purposes
        or reader.denied_attempts
    ):
        raise PermissionError("Candidate9 authorization read vector mismatch")
    parent_fd, parent_identity = _open_stable_directory(C9_ANCHOR.parent, role="Candidate9 anchor parent authorization")
    os.close(parent_fd)
    issuance_token = object()
    seal = AuthorizationSeal(
        reader=reader,
        authorization_sha256=authorization_sha256,
        authorization_size_bytes=authorization_size_bytes,
        qualification_sha256=qualification_sha,
        qualification_size_bytes=qualification_size,
        control_commitment_sha256=control_commitment,
        source_commitment_sha256=context.source_commitment_sha256,
        authority_read_inventory_commitment_sha256=_entry_commitment(reader.entries),
        authority_raw_fd_opened_commitment_sha256=_purpose_vector_commitment(reader.raw_fd_opened_purposes),
        authority_verified_inode_commitment_sha256=_purpose_vector_commitment(reader.verified_inode_purposes),
        authority_read_started_commitment_sha256=_purpose_vector_commitment(reader.read_started_purposes),
        anchor_parent_identity=parent_identity,
        output_freshness=output_freshness,
        _context_nonce=context._nonce,
        _sentinel=_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_AUTHORIZATIONS[id(seal)] = (seal, issuance_token)
    return seal


class AnchorCreationFailure(RuntimeError):
    pass


def _validate_authorization_seal(context: BootstrapContext, seal: AuthorizationSeal) -> None:
    _validate_context_integrity(context)
    issued = _ISSUED_AUTHORIZATIONS.get(id(seal))
    if (
        type(seal) is not AuthorizationSeal
        or seal._sentinel is not _SEAL
        or issued is None
        or issued[0] is not seal
        or issued[1] is not seal._issuance_token
        or seal._context_nonce is not context._nonce
        or seal.source_commitment_sha256 != context.source_commitment_sha256
        or seal.output_freshness._sentinel is not _SEAL
        or seal.authority_read_inventory_commitment_sha256 != _entry_commitment(seal.reader.entries)
        or seal.authority_raw_fd_opened_commitment_sha256 != _purpose_vector_commitment(seal.reader.raw_fd_opened_purposes)
        or seal.authority_verified_inode_commitment_sha256 != _purpose_vector_commitment(seal.reader.verified_inode_purposes)
        or seal.authority_read_started_commitment_sha256 != _purpose_vector_commitment(seal.reader.read_started_purposes)
    ):
        raise PermissionError("Candidate9 authorization seal/context mismatch")
    expected = [str(entry["purpose"]) for entry in context.source_entries] + [
        "CANDIDATE9_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE9_DIAGNOSTIC_QUALIFICATION",
    ]
    expected_authority_purposes = expected[len(context.source_entries):]
    if (
        [str(entry["purpose"]) for entry in seal.reader.entries] != expected
        or _entry_commitment(seal.reader.entries[: len(context.source_entries)])
        != _entry_commitment(context.source_entries)
        or seal.reader.raw_fd_opened_purposes != expected_authority_purposes
        or seal.reader.verified_inode_purposes != expected_authority_purposes
        or seal.reader.read_started_purposes != expected_authority_purposes
        or seal.reader.denied_attempts
    ):
        raise PermissionError("Candidate9 authorization ledger prefix mismatch")
    parent_fd, identity = _open_stable_directory(C9_ANCHOR.parent, role="Candidate9 anchor parent seal revalidation")
    os.close(parent_fd)
    if identity != seal.anchor_parent_identity:
        raise PermissionError("Candidate9 anchor parent changed")
    revalidate_fresh_output(seal.output_freshness)


def _assert_canonical_anchor_visibility(
    *,
    path: Path,
    parent_identity: tuple[int, int],
    file_identity: tuple[int, int],
    expected_data: bytes,
) -> None:
    parent_fd, visible_parent_identity = _open_stable_directory(
        path.parent, role="Candidate9 canonical anchor parent"
    )
    descriptor: int | None = None
    try:
        if visible_parent_identity != parent_identity:
            raise RuntimeError("Candidate9 canonical anchor parent changed")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate9 requires O_NOFOLLOW")
        descriptor = os.open(path.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(opened) != file_identity
            or int(opened.st_size) != len(expected_data)
        ):
            raise RuntimeError("Candidate9 canonical anchor leaf changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        visible = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            data != expected_data
            or sha256_bytes(data) != sha256_bytes(expected_data)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or _identity(after) != file_identity
            or int(after.st_size) != len(data)
            or not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or _identity(visible) != file_identity
            or _identity(os.lstat(path.parent)) != parent_identity
        ):
            raise RuntimeError("Candidate9 canonical anchor visibility changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def consume_authorization(context: BootstrapContext, seal: AuthorizationSeal) -> ConsumptionSeal:
    _validate_authorization_seal(context, seal)
    anchor = _lexical(C9_ANCHOR, role="Candidate9 anchor")
    if anchor != _lexical(ROOT / "research/f1_aug30_train18_candidate9_first_primary_diagnostic_consumption_v1.json", role="frozen Candidate9 anchor"):
        raise ValueError("Candidate9 anchor namespace is frozen")
    value = {
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "authorization_binding": _binding(C9_AUTHORIZATION, seal.authorization_sha256, seal.authorization_size_bytes, kind=AUTHORIZATION_KIND),
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "consumed_before_manifest_open": True,
        "control_binding_commitment_sha256": seal.control_commitment_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "holdout_label_evaluator_metric_fit_authorized": False,
        "kind": ANCHOR_KIND,
        "manifest_opened_at_anchor_creation": False,
        "output_namespace": str(OUTPUT_ROOT.relative_to(ROOT)),
        "record_selector": RECORD_SELECTOR,
        "replay_or_later_pair_authorized": False,
        "retry_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": context.source_commitment_sha256,
        "status": "CONSUMED_BEFORE_MANIFEST_OPEN",
    }
    data = canonical_bytes(value)
    parent_fd, parent_identity = _open_stable_directory(anchor.parent, role="Candidate9 anchor parent")
    if parent_identity != seal.anchor_parent_identity:
        os.close(parent_fd)
        raise PermissionError("Candidate9 anchor parent changed before O_EXCL")
    descriptor: int | None = None
    created = False
    identity: tuple[int, int] | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("Candidate9 requires O_NOFOLLOW")
        descriptor = os.open(anchor.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o400, dir_fd=parent_fd)
        created = True
        os.fsync(parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Candidate9 anchor write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        identity = _identity(metadata)
        visible = os.stat(anchor.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or int(metadata.st_size) != len(data) or _identity(visible) != identity:
            raise RuntimeError("Candidate9 anchor write mismatch")
        os.fsync(parent_fd)
        _assert_canonical_anchor_visibility(
            path=anchor,
            parent_identity=parent_identity,
            file_identity=identity,
            expected_data=data,
        )
    except FileExistsError as error:
        raise PermissionError("Candidate9 authorization is already consumed") from error
    except BaseException as error:
        if created:
            raise AnchorCreationFailure("Candidate9 anchor failed after durable consumption") from error
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
                    "Candidate9 anchor cleanup failed after durable consumption"
                ) from close_error
            raise close_error
    if identity is None:
        raise AnchorCreationFailure("Candidate9 anchor lacks captured identity")
    issuance_token = object()
    consumption = ConsumptionSeal(
        path=anchor,
        data=data,
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        file_identity=identity,
        parent_identity=parent_identity,
        _sentinel=_SEAL,
        _issuance_token=issuance_token,
    )
    _ISSUED_CONSUMPTIONS[id(consumption)] = (consumption, issuance_token)
    try:
        revalidate_anchor(consumption)
    except BaseException as error:
        raise AnchorCreationFailure("Candidate9 anchor final validation failed") from error
    return consumption


def revalidate_anchor(seal: ConsumptionSeal) -> None:
    issued = _ISSUED_CONSUMPTIONS.get(id(seal))
    if (
        type(seal) is not ConsumptionSeal
        or seal._sentinel is not _SEAL
        or issued is None
        or issued[0] is not seal
        or issued[1] is not seal._issuance_token
    ):
        raise PermissionError("Candidate9 anchor seal is invalid")
    _assert_canonical_anchor_visibility(
        path=seal.path,
        parent_identity=seal.parent_identity,
        file_identity=seal.file_identity,
        expected_data=seal.data,
    )


@dataclass(frozen=True)
class DiagnosticOutcome:
    result: Mapping[str, Any]
    original_pool_sha256: str
    adapted_pool_sha256: str
    projection_byte_equivalent: bool
    first_primary_commitment_sha256: str


@dataclass
class DiagnosticProgress:
    diagnostic_call_returned: bool = False
    first_primary_commitment_sha256: str | None = None
    original_pool_sha256: str | None = None
    adapted_pool_sha256: str | None = None
    projection_byte_equivalent: bool = False


def _purpose_vector_commitment(purposes: Sequence[str]) -> str:
    return sha256_bytes(canonical_bytes([{"ordinal": index, "purpose": purpose} for index, purpose in enumerate(purposes)]))


def _entry_commitment(entries: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes([
        {
            "device_id": int(entry["device_id"]),
            "inode": int(entry["inode"]),
            "path": str(entry["path"]),
            "purpose": str(entry["purpose"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
        for entry in entries
    ]))


def _validate_context_integrity(context: BootstrapContext) -> None:
    issued = _ISSUED_CONTEXTS.get(id(context))
    if (
        type(context) is not BootstrapContext
        or context._sentinel is not _CONTEXT_SEAL
        or issued is None
        or issued[0] is not context
        or issued[1] is not context._issuance_token
        or context._nonce is None
    ):
        raise PermissionError("Candidate9 verified source context is invalid")
    expected_names = set(FIXED_SOURCE_BINDINGS) | {"runner", "runner_tests"}
    if set(context.source_bindings) != expected_names:
        raise PermissionError("Candidate9 verified source binding inventory changed")
    if context.source_commitment_sha256 != sha256_bytes(
        canonical_bytes(_source_projection(context.source_bindings))
    ):
        raise PermissionError("Candidate9 verified source commitment changed")
    entries_by_purpose = {
        str(entry.get("purpose")): entry for entry in context.source_entries
    }
    if len(entries_by_purpose) != len(expected_names) or len(context.source_entries) != len(expected_names):
        raise PermissionError("Candidate9 verified source entry inventory changed")
    identities: set[tuple[int, int]] = set()
    for name in sorted(expected_names):
        binding = context.source_bindings[name]
        purpose = f"CANDIDATE9_SOURCE_{name.upper()}"
        entry = entries_by_purpose.get(purpose)
        if (
            entry is None
            or {
                "path": str(entry.get("path")),
                "sha256": str(entry.get("sha256")),
                "size_bytes": entry.get("size_bytes"),
            }
            != dict(binding)
        ):
            raise PermissionError("Candidate9 verified source entry changed")
        identity = _strict_int(entry.get("device_id"), name="source device"), _strict_int(
            entry.get("inode"), name="source inode"
        )
        if identity in identities:
            raise PermissionError("Candidate9 verified sources acquired an inode alias")
        identities.add(identity)
    expected_callables = (
        context.support.qualify_source_controls,
        context.support._validate_manifest_first_primary,
        context.support.GuardedReader,
        context.c8_adapter.diagnose_bound_pair,
        context.c8_runtime._validate_result,
        context.c9_adapter.adapt_pool_bytes,
    )
    if context._controlled_callables != expected_callables:
        raise PermissionError("Candidate9 controlled runtime callable changed")
    modules = (
        context.support,
        context.c6_hypothesis,
        context.c7_runtime,
        context.c8_runtime,
        context.c8_adapter,
        context.c9_adapter,
    )
    if context._controlled_function_snapshot != _controlled_function_snapshot(*modules):
        raise PermissionError("Candidate9 controlled runtime function changed")
    if context._constant_commitment_sha256 != _controlled_constant_commitment(
        support=context.support,
        hypothesis=context.c6_hypothesis,
        candidate7=context.c7_runtime,
        c8_runtime=context.c8_runtime,
        c9_adapter=context.c9_adapter,
    ):
        raise PermissionError("Candidate9 controlled runtime constant changed")
    if (
        context.c8_adapter_context.runtime is not context.c8_runtime
        or context.c8_adapter_context.candidate7_runtime is not context.c7_runtime
        or context.c8_adapter_context.hypothesis is not context.c6_hypothesis
        or context.c9_adapter_context.hypothesis is not context.c6_hypothesis
        or context.c9_adapter_context.builder is not context.builder_facade
        or context.c9_adapter_context.clusterer is not context.cluster_facade
        or dict(context.c8_adapter_context.source_bindings)
        != {
            "candidate7_runtime": dict(context.source_bindings["c7_runtime"]),
            "hypothesis": dict(context.source_bindings["c6_hypothesis"]),
        }
        or context.c9_adapter.ATTEMPT_ORDINAL != ATTEMPT_ORDINAL
        or context.c9_adapter.SCIENTIFIC_HYPOTHESIS_ID != SCIENTIFIC_HYPOTHESIS_ID
    ):
        raise PermissionError("Candidate9 controlled runtime graph changed")
    context.c8_runtime.verify_scientific_hypothesis_runtime(
        context.c6_hypothesis,
        source_sha256=context.source_bindings["c6_hypothesis"]["sha256"],
    )
    context.c8_runtime.verify_candidate7_wrapper_runtime(
        context.c7_runtime,
        source_sha256=context.source_bindings["c7_runtime"]["sha256"],
        hypothesis_runtime=context.c6_hypothesis,
    )


def _expected_control_purposes(context: BootstrapContext) -> list[str]:
    return [
        *context.support.FIXED_CONTROLS,
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
        "CANDIDATE8_TERMINAL_DISPOSITION",
        "CANDIDATE8_DIAGNOSTIC_RESULT",
        "CANDIDATE8_DIAGNOSTIC_LEDGER",
    ]


def read_adapt_and_diagnose(
    context: BootstrapContext,
    qualification: ControlQualification,
    consumption: ConsumptionSeal,
    *,
    stage_callback: Any = None,
    progress: DiagnosticProgress | None = None,
) -> DiagnosticOutcome:
    _validate_context_integrity(context)
    _validate_control_qualification(context, qualification)
    if progress is None:
        progress = DiagnosticProgress()
    if stage_callback is not None:
        stage_callback("FIRST_PAIR_READ")
    reader = qualification.reader
    base = qualification.base
    revalidate_anchor(consumption)
    reader.authorize(
        base.manifest_path,
        purpose="TRAIN_CACHE_MANIFEST",
        sha256=base.manifest_sha256,
        size_bytes=base.manifest_size_bytes,
    )
    revalidate_anchor(consumption)
    manifest = _decode(
        reader.read_bytes(base.manifest_path, purpose="TRAIN_CACHE_MANIFEST"),
        name="TRAIN_CACHE_MANIFEST",
    )
    revalidate_anchor(consumption)
    first = context.support._validate_manifest_first_primary(manifest, base)
    if first.commitment_sha256 != qualification.c8_first_primary_commitment_sha256:
        raise RuntimeError("Candidate9 manifest pair differs from Candidate8 consumed pair")
    progress.first_primary_commitment_sha256 = first.commitment_sha256
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
        != (int(base.c7_semantic_evidence["device_id"]), int(base.c7_semantic_evidence["inode"]))
        or reader.allowed[first.pool_path]["identity"]
        != (int(base.c7_pool_evidence["device_id"]), int(base.c7_pool_evidence["inode"]))
    ):
        raise RuntimeError("Candidate9 first-pair inode binding mismatch")
    revalidate_anchor(consumption)
    semantic_bytes = reader.read_bytes(first.semantic_path, purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC")
    pool_bytes = reader.read_bytes(first.pool_path, purpose="TRAIN_FIRST_PRIMARY_V51_POOL")
    revalidate_anchor(consumption)
    if stage_callback is not None:
        stage_callback("REPRESENTATION_ADAPTER")
    adapted = context.c9_adapter.adapt_pool_bytes(
        context.c9_adapter_context,
        pool_bytes=pool_bytes,
        expected_pool_sha256=first.pool_sha256,
    )
    if adapted.original_pool_sha256 != first.pool_sha256 or adapted.projection_byte_equivalent is not True:
        raise RuntimeError("Candidate9 adapted-pool binding mismatch")
    if sha256_bytes(adapted.adapted_pool_bytes) != adapted.adapted_pool_sha256:
        raise RuntimeError("Candidate9 adapted-pool bytes do not match their receipt")
    progress.original_pool_sha256 = adapted.original_pool_sha256
    progress.adapted_pool_sha256 = adapted.adapted_pool_sha256
    progress.projection_byte_equivalent = True
    if stage_callback is not None:
        stage_callback("DIAGNOSTIC_RESULT")
    diagnostic = context.c8_adapter.diagnose_bound_pair(
        context.c8_adapter_context,
        semantic_bytes=semantic_bytes,
        pool_bytes=adapted.adapted_pool_bytes,
        candidate7_disposition_bytes=base.candidate7_disposition_bytes,
        semantic_sha256=first.semantic_sha256,
        pool_sha256=adapted.adapted_pool_sha256,
        candidate7_disposition_sha256=context.support.FIXED_CONTROLS["CANDIDATE7_TERMINAL_DISPOSITION"][1],
    )
    progress.diagnostic_call_returned = True
    context.c8_runtime._validate_result(diagnostic)
    expected_phase = {
        context.c8_runtime.PASS_CODE: None,
        **{
            context.c8_runtime.FAILURE_CODES[phase]: phase
            for phase in context.c8_runtime.PHASE_CONTRACT
        },
    }
    if diagnostic.get("check_code") not in expected_phase or diagnostic.get("failure_phase") != expected_phase[diagnostic.get("check_code")]:
        raise RuntimeError("Candidate9 diagnostic code/phase mismatch")
    revalidate_anchor(consumption)
    return DiagnosticOutcome(
        result=dict(diagnostic),
        original_pool_sha256=adapted.original_pool_sha256,
        adapted_pool_sha256=adapted.adapted_pool_sha256,
        projection_byte_equivalent=True,
        first_primary_commitment_sha256=first.commitment_sha256,
    )


def _validate_exact_inventory(
    context: BootstrapContext,
    qualification: ControlQualification,
    authorization: AuthorizationSeal,
) -> None:
    _validate_context_integrity(context)
    _validate_control_qualification(context, qualification)
    _validate_authorization_seal(context, authorization)
    source_purposes = [str(entry["purpose"]) for entry in context.source_entries]
    artifact_purposes = [
        *_expected_control_purposes(context),
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ]
    reader = qualification.reader
    observed = [str(entry["purpose"]) for entry in reader.entries]
    paths = [str(entry["path"]) for entry in reader.entries]
    identities = [(int(entry["device_id"]), int(entry["inode"])) for entry in reader.entries]
    if (
        observed != source_purposes + artifact_purposes
        or reader.raw_fd_opened_purposes != artifact_purposes
        or reader.verified_inode_purposes != artifact_purposes
        or reader.read_started_purposes != artifact_purposes
        or len(paths) != len(set(paths))
        or len(identities) != len(set(identities))
        or reader.denied_attempts
    ):
        raise RuntimeError("Candidate9 exact diagnostic read inventory mismatch")


ENGINEERING_FAILURES = {
    "ANCHOR_CONSUMPTION": "ANCHOR_FAILED_AFTER_CONSUMPTION",
    "FIRST_PAIR_READ": "FIRST_PAIR_READ_OR_BINDING_FAILED",
    "REPRESENTATION_ADAPTER": "LOSSLESS_PROJECTION_REJECTED",
    "DIAGNOSTIC_RESULT": "UNCHANGED_DIAGNOSTIC_REJECTED",
    "READ_INVENTORY": "EXACT_READ_INVENTORY_REJECTED",
    "UNEXPECTED": "UNEXPECTED_POST_ANCHOR_FAILURE",
}

DIAGNOSTIC_CODE_PHASES = {
    "C8_ALL_DIAGNOSTIC_PHASES_PASSED": None,
    "C8_SEMANTIC_VALIDATION_FAILED": "D01_SEMANTIC_VALIDATION",
    "C8_POOL_PARSE_SANITIZE_FAILED": "D02_POOL_PARSE_SANITIZE",
    "C8_OWNERSHIP_EXERCISE_FAILED": "D03_OWNERSHIP_EXERCISE",
    "C8_FIXED_RULE_EXERCISE_FAILED": "D04_FIXED_RULE_EXERCISE",
    "C8_WRAPPER_RESULT_VALIDATION_FAILED": "D05_WRAPPER_RESULT_VALIDATION",
}

LEDGER_KEYS = frozenset(
    {
        "adapted_pool_bytes_published",
        "adapted_pool_sha256",
        "assertions",
        "attempt_ordinal",
        "authorization_binding_commitment_sha256",
        "authorization_raw_fd_opened_inventory_commitment_sha256",
        "authorization_read_inventory_commitment_sha256",
        "authorization_read_started_inventory_commitment_sha256",
        "authorization_verified_inode_inventory_commitment_sha256",
        "builder_cluster_lineage_bytes_authenticated",
        "builder_cluster_source_execution_performed",
        "check_code",
        "consumption_anchor_sha256",
        "control_binding_commitment_sha256",
        "diagnostic_failure_phase",
        "engineering_attempt_id",
        "engineering_check_code",
        "engineering_failure_stage",
        "first_primary_binding_commitment_sha256",
        "kind",
        "original_pool_bytes_published",
        "original_pool_sha256",
        "projection_byte_equivalent",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
        "schema_version",
        "scientific_hypothesis_id",
        "source_binding_commitment_sha256",
        "status",
        "verified_inode_inventory_commitment_sha256",
    }
)

LEDGER_ASSERTION_KEYS = frozenset(
    {
        "authorization_consumption_attempted",
        "consumption_anchor_sealed",
        "denied_read_attempted",
        "diagnostic_call_returned",
        "diagnostic_result_accepted",
        "evaluator_metric_fit_opened_or_started",
        "exact_read_inventory_validated",
        "full28_opened",
        "holdout_or_label_opened",
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
)

PUBLIC_KEYS = frozenset(
    {
        "adapted_pool_bytes_published",
        "attempt_ordinal",
        "candidate_metric_computed",
        "builder_cluster_lineage_bytes_authenticated",
        "builder_cluster_source_execution_performed",
        "check_code",
        "continuation_authorized",
        "diagnostic_completed",
        "diagnostic_failure_phase",
        "diagnostic_read_ledger_sha256",
        "engineering_attempt_id",
        "engineering_check_code",
        "engineering_failure_stage",
        "evaluator_opened",
        "fit_or_model_performed",
        "full28_opened",
        "holdout_or_labels_opened",
        "kind",
        "original_pool_bytes_published",
        "promotion_authorized",
        "replay_or_later_pair_opened",
        "retry_authorized",
        "schema_version",
        "scientific_hypothesis_id",
        "scientific_hypothesis_performance_tested",
        "status",
    }
)


def _ledger(
    context: BootstrapContext,
    authorization: AuthorizationSeal,
    qualification: ControlQualification,
    consumption: ConsumptionSeal | None,
    *,
    outcome: DiagnosticOutcome | None,
    progress: DiagnosticProgress,
    engineering_failure_stage: str | None,
    engineering_check_code: str | None,
    exact_inventory: bool,
) -> dict[str, Any]:
    reader = qualification.reader
    raw = set(reader.raw_fd_opened_purposes)
    verified = set(reader.verified_inode_purposes)
    started = set(reader.read_started_purposes)
    accepted = outcome is not None
    value = {
        "adapted_pool_bytes_published": False,
        "adapted_pool_sha256": progress.adapted_pool_sha256,
        "assertions": {
            "authorization_consumption_attempted": True,
            "consumption_anchor_sealed": consumption is not None,
            "denied_read_attempted": bool(reader.denied_attempts),
            "diagnostic_call_returned": progress.diagnostic_call_returned,
            "diagnostic_result_accepted": accepted,
            "evaluator_metric_fit_opened_or_started": False,
            "exact_read_inventory_validated": exact_inventory,
            "full28_opened": False,
            "holdout_or_label_opened": False,
            "manifest_inode_verified": "TRAIN_CACHE_MANIFEST" in verified,
            "manifest_opened": "TRAIN_CACHE_MANIFEST" in raw,
            "manifest_read_started": "TRAIN_CACHE_MANIFEST" in started,
            "only_first_primary_pool_inode_verified": "TRAIN_FIRST_PRIMARY_V51_POOL" in verified,
            "only_first_primary_pool_opened": "TRAIN_FIRST_PRIMARY_V51_POOL" in raw,
            "only_first_primary_pool_read_started": "TRAIN_FIRST_PRIMARY_V51_POOL" in started,
            "only_first_primary_semantic_inode_verified": "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in verified,
            "only_first_primary_semantic_opened": "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in raw,
            "only_first_primary_semantic_read_started": "TRAIN_FIRST_PRIMARY_V51_SEMANTIC" in started,
            "replay_or_later_pair_opened": False,
        },
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "authorization_binding_commitment_sha256": sha256_bytes(canonical_bytes(_binding(C9_AUTHORIZATION, authorization.authorization_sha256, authorization.authorization_size_bytes))),
        "authorization_raw_fd_opened_inventory_commitment_sha256": authorization.authority_raw_fd_opened_commitment_sha256,
        "authorization_read_inventory_commitment_sha256": authorization.authority_read_inventory_commitment_sha256,
        "authorization_read_started_inventory_commitment_sha256": authorization.authority_read_started_commitment_sha256,
        "authorization_verified_inode_inventory_commitment_sha256": authorization.authority_verified_inode_commitment_sha256,
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "check_code": None if outcome is None else outcome.result["check_code"],
        "consumption_anchor_sha256": None if consumption is None else consumption.sha256,
        "control_binding_commitment_sha256": qualification.control_commitment_sha256,
        "diagnostic_failure_phase": None if outcome is None else outcome.result["failure_phase"],
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": engineering_check_code,
        "engineering_failure_stage": engineering_failure_stage,
        "first_primary_binding_commitment_sha256": progress.first_primary_commitment_sha256,
        "kind": LEDGER_KIND,
        "original_pool_bytes_published": False,
        "original_pool_sha256": progress.original_pool_sha256,
        "projection_byte_equivalent": progress.projection_byte_equivalent,
        "raw_fd_opened_inventory_commitment_sha256": _purpose_vector_commitment(reader.raw_fd_opened_purposes),
        "read_inventory_commitment_sha256": _entry_commitment(reader.entries),
        "read_started_inventory_commitment_sha256": _purpose_vector_commitment(reader.read_started_purposes),
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": context.source_commitment_sha256,
        "status": "DIAGNOSTIC_RECORDED" if accepted else "TERMINAL_ENGINEERING_FAILURE",
        "verified_inode_inventory_commitment_sha256": _purpose_vector_commitment(reader.verified_inode_purposes),
    }
    _validate_ledger(value)
    return value


def _validate_ledger(value: Mapping[str, Any]) -> None:
    assertions = value.get("assertions")
    if (
        set(value) != LEDGER_KEYS
        or not isinstance(assertions, dict)
        or set(assertions) != LEDGER_ASSERTION_KEYS
        or any(not isinstance(item, bool) for item in assertions.values())
    ):
        raise ValueError("Candidate9 ledger schema mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != LEDGER_KIND
        or value.get("attempt_ordinal") != 9
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("adapted_pool_bytes_published") is not False
        or value.get("original_pool_bytes_published") is not False
        or value.get("builder_cluster_lineage_bytes_authenticated") is not True
        or value.get("builder_cluster_source_execution_performed") is not False
        or assertions["authorization_consumption_attempted"] is not True
    ):
        raise ValueError("Candidate9 ledger identity mismatch")
    for key in (
        "authorization_binding_commitment_sha256",
        "authorization_raw_fd_opened_inventory_commitment_sha256",
        "authorization_read_inventory_commitment_sha256",
        "authorization_read_started_inventory_commitment_sha256",
        "authorization_verified_inode_inventory_commitment_sha256",
        "control_binding_commitment_sha256",
        "raw_fd_opened_inventory_commitment_sha256",
        "read_inventory_commitment_sha256",
        "read_started_inventory_commitment_sha256",
        "source_binding_commitment_sha256",
        "verified_inode_inventory_commitment_sha256",
    ):
        _require_sha(value.get(key), name=f"Candidate9 ledger {key}")
    for key in (
        "adapted_pool_sha256",
        "consumption_anchor_sha256",
        "first_primary_binding_commitment_sha256",
        "original_pool_sha256",
    ):
        if value.get(key) is not None:
            _require_sha(value.get(key), name=f"Candidate9 ledger {key}")
    if not isinstance(value.get("projection_byte_equivalent"), bool):
        raise ValueError("Candidate9 projection equivalence must be strict bool")
    for raw_key, verified_key, started_key in (
        ("manifest_opened", "manifest_inode_verified", "manifest_read_started"),
        ("only_first_primary_pool_opened", "only_first_primary_pool_inode_verified", "only_first_primary_pool_read_started"),
        ("only_first_primary_semantic_opened", "only_first_primary_semantic_inode_verified", "only_first_primary_semantic_read_started"),
    ):
        if assertions[verified_key] and not assertions[raw_key]:
            raise ValueError("Candidate9 inode verification lacks raw-open evidence")
        if assertions[started_key] and not assertions[verified_key]:
            raise ValueError("Candidate9 read start lacks inode verification")
    if any(assertions[key] for key in ("evaluator_metric_fit_opened_or_started", "full28_opened", "holdout_or_label_opened", "replay_or_later_pair_opened")):
        raise ValueError("Candidate9 boundary assertion is unsafe")
    if assertions["consumption_anchor_sealed"] != (value.get("consumption_anchor_sha256") is not None):
        raise ValueError("Candidate9 anchor sealing declaration mismatch")
    if assertions["diagnostic_result_accepted"] and not assertions["diagnostic_call_returned"]:
        raise ValueError("Candidate9 accepted a diagnostic that did not return")
    projection_present = value.get("projection_byte_equivalent") is True
    if projection_present != (
        value.get("original_pool_sha256") is not None
        and value.get("adapted_pool_sha256") is not None
    ):
        raise ValueError("Candidate9 projection binding tuple mismatch")
    if assertions["diagnostic_call_returned"] and not projection_present:
        raise ValueError("Candidate9 diagnostic returned without an adapted pool binding")
    accepted = assertions.get("diagnostic_result_accepted") is True
    if accepted:
        if (
            value.get("status") != "DIAGNOSTIC_RECORDED"
            or value.get("check_code") not in DIAGNOSTIC_CODE_PHASES
            or value.get("diagnostic_failure_phase")
            != DIAGNOSTIC_CODE_PHASES[value.get("check_code")]
            or value.get("engineering_failure_stage") is not None
            or value.get("engineering_check_code") is not None
            or not assertions.get("exact_read_inventory_validated")
            or assertions.get("denied_read_attempted")
            or not assertions.get("consumption_anchor_sealed")
            or not assertions.get("manifest_opened")
            or not assertions.get("manifest_inode_verified")
            or not assertions.get("manifest_read_started")
            or not assertions.get("only_first_primary_semantic_opened")
            or not assertions.get("only_first_primary_semantic_inode_verified")
            or not assertions.get("only_first_primary_semantic_read_started")
            or not assertions.get("only_first_primary_pool_opened")
            or not assertions.get("only_first_primary_pool_inode_verified")
            or not assertions.get("only_first_primary_pool_read_started")
            or not value.get("projection_byte_equivalent")
            or value.get("original_pool_sha256") is None
            or value.get("adapted_pool_sha256") is None
            or value.get("first_primary_binding_commitment_sha256") is None
        ):
            raise ValueError("Candidate9 accepted ledger mismatch")
    elif (
        value.get("status") != "TERMINAL_ENGINEERING_FAILURE"
        or value.get("check_code") is not None
        or value.get("diagnostic_failure_phase") is not None
        or value.get("engineering_failure_stage") not in ENGINEERING_FAILURES
        or value.get("engineering_check_code") != ENGINEERING_FAILURES[value.get("engineering_failure_stage")]
        or assertions.get("diagnostic_result_accepted")
        or assertions.get("exact_read_inventory_validated")
    ):
        raise ValueError("Candidate9 terminal ledger mismatch")


def _public_payload(*, outcome: DiagnosticOutcome | None, ledger_sha256: str, engineering_stage: str | None = None) -> dict[str, Any]:
    accepted = outcome is not None
    value = {
        "adapted_pool_bytes_published": False,
        "attempt_ordinal": ATTEMPT_ORDINAL,
        "candidate_metric_computed": False,
        "builder_cluster_lineage_bytes_authenticated": True,
        "builder_cluster_source_execution_performed": False,
        "check_code": None if outcome is None else outcome.result["check_code"],
        "continuation_authorized": False,
        "diagnostic_completed": accepted,
        "diagnostic_failure_phase": None if outcome is None else outcome.result["failure_phase"],
        "diagnostic_read_ledger_sha256": ledger_sha256,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": None if accepted else ENGINEERING_FAILURES[engineering_stage],
        "engineering_failure_stage": None if accepted else engineering_stage,
        "evaluator_opened": False,
        "fit_or_model_performed": False,
        "full28_opened": False,
        "holdout_or_labels_opened": False,
        "kind": RESULT_KIND if accepted else TERMINAL_KIND,
        "original_pool_bytes_published": False,
        "promotion_authorized": False,
        "replay_or_later_pair_opened": False,
        "retry_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "scientific_hypothesis_performance_tested": False,
        "status": "DIAGNOSTIC_COMPLETE_NONAUTHORIZING" if accepted else "TERMINAL_ENGINEERING_FAILURE",
    }
    _validate_public_payload(value)
    return value


def _validate_public_payload(value: Mapping[str, Any]) -> None:
    if set(value) != PUBLIC_KEYS or not isinstance(value.get("diagnostic_completed"), bool):
        raise ValueError("Candidate9 public payload schema mismatch")
    _require_sha(value.get("diagnostic_read_ledger_sha256"), name="Candidate9 ledger SHA")
    if (
        value.get("schema_version") != 1
        or value.get("attempt_ordinal") != ATTEMPT_ORDINAL
        or value.get("engineering_attempt_id") != ENGINEERING_ATTEMPT_ID
        or value.get("scientific_hypothesis_id") != SCIENTIFIC_HYPOTHESIS_ID
        or value.get("builder_cluster_lineage_bytes_authenticated") is not True
        or value.get("builder_cluster_source_execution_performed") is not False
        or any(
            value.get(key) is not False
            for key in (
                "adapted_pool_bytes_published",
                "candidate_metric_computed",
                "continuation_authorized",
                "evaluator_opened",
                "fit_or_model_performed",
                "full28_opened",
                "holdout_or_labels_opened",
                "original_pool_bytes_published",
                "promotion_authorized",
                "replay_or_later_pair_opened",
                "retry_authorized",
                "scientific_hypothesis_performance_tested",
            )
        )
    ):
        raise ValueError("Candidate9 public identity/boundary mismatch")
    if value.get("diagnostic_completed") is True:
        if (
            value.get("kind") != RESULT_KIND
            or value.get("status") != "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"
            or value.get("check_code") not in DIAGNOSTIC_CODE_PHASES
            or value.get("diagnostic_failure_phase")
            != DIAGNOSTIC_CODE_PHASES[value.get("check_code")]
            or value.get("engineering_failure_stage") is not None
            or value.get("engineering_check_code") is not None
        ):
            raise ValueError("Candidate9 public diagnostic code/phase mismatch")
    else:
        stage = value.get("engineering_failure_stage")
        if (
            value.get("kind") != TERMINAL_KIND
            or value.get("status") != "TERMINAL_ENGINEERING_FAILURE"
            or value.get("check_code") is not None
            or value.get("diagnostic_failure_phase") is not None
            or stage not in ENGINEERING_FAILURES
            or value.get("engineering_check_code") != ENGINEERING_FAILURES[stage]
        ):
            raise ValueError("Candidate9 public terminal code/stage mismatch")


def build_payloads(context: BootstrapContext, authorization: AuthorizationSeal) -> dict[str, bytes]:
    _validate_authorization_seal(context, authorization)
    qualification = qualify_source_controls(context)
    qualification_bytes = build_qualification_record(context, qualification)
    if (
        qualification.control_commitment_sha256 != authorization.control_commitment_sha256
        or sha256_bytes(qualification_bytes) != authorization.qualification_sha256
        or len(qualification_bytes) != authorization.qualification_size_bytes
    ):
        raise PermissionError("Candidate9 qualification changed after authorization")
    revalidate_fresh_output(authorization.output_freshness)
    consumption: ConsumptionSeal | None = None
    outcome: DiagnosticOutcome | None = None
    progress = DiagnosticProgress()
    stage = "UNEXPECTED"
    def set_stage(value: str) -> None:
        nonlocal stage
        stage = value
    try:
        consumption = consume_authorization(context, authorization)
        outcome = read_adapt_and_diagnose(
            context,
            qualification,
            consumption,
            stage_callback=set_stage,
            progress=progress,
        )
        stage = "READ_INVENTORY"
        _validate_exact_inventory(context, qualification, authorization)
        ledger_value = _ledger(context, authorization, qualification, consumption, outcome=outcome, progress=progress, engineering_failure_stage=None, engineering_check_code=None, exact_inventory=True)
        ledger_bytes = canonical_bytes(ledger_value)
        result = _public_payload(outcome=outcome, ledger_sha256=sha256_bytes(ledger_bytes))
        return {"diagnostic_read_ledger.json": ledger_bytes, "diagnostic_result.json": canonical_bytes(result)}
    except BaseException as error:
        if consumption is None and not isinstance(error, AnchorCreationFailure):
            raise
        if isinstance(error, AnchorCreationFailure):
            stage = "ANCHOR_CONSUMPTION"
        ledger_value = _ledger(context, authorization, qualification, consumption, outcome=None, progress=progress, engineering_failure_stage=stage, engineering_check_code=ENGINEERING_FAILURES[stage], exact_inventory=False)
        ledger_bytes = canonical_bytes(ledger_value)
        terminal = _public_payload(outcome=None, ledger_sha256=sha256_bytes(ledger_bytes), engineering_stage=stage)
        return {"diagnostic_read_ledger.json": ledger_bytes, "diagnostic_terminal.json": canonical_bytes(terminal)}


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
        raise RuntimeError("Candidate9 requires no-replace rename")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), flag) != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, "Candidate9 publication destination exists")
        raise OSError(number, os.strerror(number))


def _verify_owned_leaf_identities(
    parent_fd: int,
    *,
    directory_name: str,
    directory_identity: tuple[int, int],
    leaf_identities: Mapping[str, tuple[int, int]],
    require_exact_inventory: bool,
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate9 requires O_NOFOLLOW/O_DIRECTORY")
    descriptor = os.open(
        directory_name,
        os.O_RDONLY | nofollow | directory,
        dir_fd=parent_fd,
    )
    try:
        opened_directory = os.fstat(descriptor)
        visible_directory = os.stat(
            directory_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or _identity(opened_directory) != directory_identity
            or _identity(visible_directory) != directory_identity
        ):
            raise RuntimeError("Candidate9 owned directory identity changed")
        names = set(os.listdir(descriptor))
        expected_names = set(leaf_identities)
        if (
            (require_exact_inventory and names != expected_names)
            or (not require_exact_inventory and not names.issubset(expected_names))
        ):
            raise RuntimeError("Candidate9 owned leaf inventory changed")
        for filename in sorted(names):
            leaf_fd = os.open(filename, os.O_RDONLY | nofollow, dir_fd=descriptor)
            try:
                opened = os.fstat(leaf_fd)
                visible = os.stat(
                    filename, dir_fd=descriptor, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or not stat.S_ISREG(visible.st_mode)
                    or visible.st_nlink != 1
                    or _identity(opened) != leaf_identities[filename]
                    or _identity(visible) != leaf_identities[filename]
                ):
                    raise RuntimeError("Candidate9 owned leaf identity changed")
            finally:
                os.close(leaf_fd)
    finally:
        os.close(descriptor)


def _cleanup_owned_stage(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
    leaf_identities: Mapping[str, tuple[int, int]],
) -> None:
    try:
        visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(visible.st_mode) or _identity(visible) != identity:
        return
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if directory is None or nofollow is None:
        return
    descriptor = os.open(name, os.O_RDONLY | directory | nofollow, dir_fd=parent_fd)
    try:
        if _identity(os.fstat(descriptor)) != identity:
            return
        visible_names = os.listdir(descriptor)
        if not set(visible_names).issubset(set(leaf_identities)):
            return
        try:
            _verify_owned_leaf_identities(
                parent_fd,
                directory_name=name,
                directory_identity=identity,
                leaf_identities=leaf_identities,
                require_exact_inventory=False,
            )
        except (OSError, RuntimeError, KeyError):
            return
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        for filename in visible_names:
            leaf = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(leaf.st_mode)
                or leaf.st_nlink != 1
                or _identity(leaf) != leaf_identities[filename]
            ):
                return
            os.unlink(filename, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _identity(visible) == identity:
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _quarantine_owned_publication(
    parent_fd: int,
    *,
    output_name: str,
    output_identity: tuple[int, int],
    leaf_identities: Mapping[str, tuple[int, int]],
) -> None:
    try:
        visible = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(visible.st_mode) or _identity(visible) != output_identity:
        return
    try:
        _verify_owned_leaf_identities(
            parent_fd,
            directory_name=output_name,
            directory_identity=output_identity,
            leaf_identities=leaf_identities,
            require_exact_inventory=True,
        )
    except (OSError, RuntimeError, KeyError):
        return
    quarantine_name: str | None = None
    for _attempt in range(64):
        try:
            current = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(current.st_mode) or _identity(current) != output_identity:
            return
        try:
            _verify_owned_leaf_identities(
                parent_fd,
                directory_name=output_name,
                directory_identity=output_identity,
                leaf_identities=leaf_identities,
                require_exact_inventory=True,
            )
        except (OSError, RuntimeError, KeyError):
            return
        candidate = f".{output_name}.quarantine-{os.urandom(16).hex()}"
        try:
            _renameat_noreplace(parent_fd, output_name, candidate)
        except FileExistsError:
            continue
        quarantine_name = candidate
        break
    if quarantine_name is None:
        raise RuntimeError("Candidate9 could not reserve an owned quarantine name")
    moved = os.stat(quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(moved.st_mode) or _identity(moved) != output_identity:
        raise RuntimeError("Candidate9 quarantine identity mismatch")
    os.fsync(parent_fd)


def _verify_published_payloads(
    parent_fd: int,
    *,
    parent_path: Path,
    parent_identity: tuple[int, int],
    directory_name: str,
    directory_identity: tuple[int, int],
    payloads: Mapping[str, bytes],
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise RuntimeError("Candidate9 requires O_NOFOLLOW/O_DIRECTORY")
    if _identity(os.lstat(parent_path)) != parent_identity:
        raise RuntimeError("Candidate9 visible publication parent changed before verification")
    descriptor = os.open(
        directory_name,
        os.O_RDONLY | nofollow | directory,
        dir_fd=parent_fd,
    )
    try:
        opened_directory = os.fstat(descriptor)
        visible_directory = os.stat(
            directory_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or _identity(opened_directory) != directory_identity
            or _identity(visible_directory) != directory_identity
            or sorted(os.listdir(descriptor)) != sorted(payloads)
        ):
            raise RuntimeError("Candidate9 published directory changed")
        for filename, expected in sorted(payloads.items()):
            leaf_fd = os.open(filename, os.O_RDONLY | nofollow, dir_fd=descriptor)
            try:
                opened = os.fstat(leaf_fd)
                identity = _identity(opened)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or int(opened.st_size) != len(expected)
                ):
                    raise RuntimeError("Candidate9 published payload metadata changed")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(leaf_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                after = os.fstat(leaf_fd)
                visible = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
                if (
                    data != expected
                    or sha256_bytes(data) != sha256_bytes(expected)
                    or not stat.S_ISREG(after.st_mode)
                    or after.st_nlink != 1
                    or _identity(after) != identity
                    or int(after.st_size) != len(data)
                    or _identity(visible) != identity
                ):
                    raise RuntimeError("Candidate9 published payload changed")
            finally:
                os.close(leaf_fd)
        if (
            _identity(os.lstat(parent_path)) != parent_identity
            or _identity(
                os.stat(directory_name, dir_fd=parent_fd, follow_symlinks=False)
            )
            != directory_identity
        ):
            raise RuntimeError("Candidate9 publication visibility changed")
    finally:
        os.close(descriptor)


def publish_atomically(output: Path, payloads: Mapping[str, bytes], *, freshness: FreshOutputSeal) -> None:
    expected_inventories = (
        frozenset({"diagnostic_read_ledger.json", "diagnostic_result.json"}),
        frozenset({"diagnostic_read_ledger.json", "diagnostic_terminal.json"}),
    )
    if frozenset(payloads) not in expected_inventories or any(not isinstance(data, bytes) for data in payloads.values()):
        raise ValueError("Candidate9 publication inventory is invalid")
    revalidate_fresh_output(freshness)
    candidate = _lexical(output, role="Candidate9 output publication")
    if candidate != freshness.path:
        raise ValueError("Candidate9 publication path differs from freshness seal")
    parent_fd, parent_identity = _open_stable_directory(candidate.parent, role="Candidate9 publication parent")
    if parent_identity != freshness.parent_identity:
        os.close(parent_fd)
        raise RuntimeError("Candidate9 publication parent changed")
    digest = sha256_bytes(canonical_bytes({name: sha256_bytes(data) for name, data in sorted(payloads.items())}))
    staging_name = f".{candidate.name}.staging-{digest[:16]}"
    staging_identity: tuple[int, int] | None = None
    staging_leaf_identities: dict[str, tuple[int, int]] = {}
    staging_fd: int | None = None
    renamed = False
    verified = False
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging_meta = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        staging_identity = _identity(staging_meta)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory is None:
            raise RuntimeError("Candidate9 requires O_NOFOLLOW/O_DIRECTORY")
        staging_fd = os.open(staging_name, os.O_RDONLY | nofollow | directory, dir_fd=parent_fd)
        if _identity(os.fstat(staging_fd)) != staging_identity:
            raise RuntimeError("Candidate9 staging identity changed")
        for filename, data in sorted(payloads.items()):
            descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o400, dir_fd=staging_fd)
            try:
                created = os.fstat(descriptor)
                visible_created = os.stat(
                    filename, dir_fd=staging_fd, follow_symlinks=False
                )
                created_identity = _identity(created)
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_nlink != 1
                    or not stat.S_ISREG(visible_created.st_mode)
                    or visible_created.st_nlink != 1
                    or _identity(visible_created) != created_identity
                    or created_identity in staging_leaf_identities.values()
                ):
                    raise RuntimeError("Candidate9 staged leaf creation mismatch")
                staging_leaf_identities[filename] = created_identity
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Candidate9 publication write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(staging_fd, 0o500)
        os.fsync(staging_fd)
        _verify_published_payloads(
            parent_fd,
            parent_path=candidate.parent,
            parent_identity=parent_identity,
            directory_name=staging_name,
            directory_identity=staging_identity,
            payloads=payloads,
        )
        _verify_owned_leaf_identities(
            parent_fd,
            directory_name=staging_name,
            directory_identity=staging_identity,
            leaf_identities=staging_leaf_identities,
            require_exact_inventory=True,
        )
        if _identity(os.lstat(candidate.parent)) != parent_identity:
            raise RuntimeError("Candidate9 visible publication parent changed")
        try:
            os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Candidate9 publication destination raced into existence")
        _renameat_noreplace(parent_fd, staging_name, candidate.name)
        renamed = True
        os.fsync(parent_fd)
        published = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(published) != staging_identity or _identity(os.lstat(candidate.parent)) != parent_identity:
            raise RuntimeError("Candidate9 published directory identity changed")
        _verify_published_payloads(
            parent_fd,
            parent_path=candidate.parent,
            parent_identity=parent_identity,
            directory_name=candidate.name,
            directory_identity=staging_identity,
            payloads=payloads,
        )
        verified = True
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        if not renamed and staging_identity is not None:
            _cleanup_owned_stage(
                parent_fd,
                staging_name,
                staging_identity,
                staging_leaf_identities,
            )
        elif renamed and not verified and staging_identity is not None:
            _quarantine_owned_publication(
                parent_fd,
                output_name=candidate.name,
                output_identity=staging_identity,
                leaf_identities=staging_leaf_identities,
            )
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
    payloads = build_payloads(context, authorization)
    publish_atomically(OUTPUT_ROOT, payloads, freshness=freshness)
    public_name = "diagnostic_result.json" if "diagnostic_result.json" in payloads else "diagnostic_terminal.json"
    public = _decode(payloads[public_name], name="Candidate9 public result")
    return {
        "check_code": public["check_code"],
        "diagnostic_failure_phase": public["diagnostic_failure_phase"],
        "holdout_label_evaluator_metric_fit_opened_or_started": False,
        "output_artifact_sha256": {name: sha256_bytes(data) for name, data in sorted(payloads.items())},
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
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-check-only", action="store_true")
    modes.add_argument("--source-control-check-only", action="store_true")
    modes.add_argument("--diagnostic-once", action="store_true")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--authorization-size-bytes", type=int)
    args = parser.parse_args()
    if args.diagnostic_once and (args.authorization_sha256 is None or args.authorization_size_bytes is None):
        parser.error("--diagnostic-once requires exact authorization SHA and size")
    if not args.diagnostic_once and (args.authorization_sha256 is not None or args.authorization_size_bytes is not None):
        parser.error("authorization binding is valid only with --diagnostic-once")
    try:
        context = bootstrap_verified_sources(
            runner_sha256=args.runner_sha256,
            runner_size_bytes=args.runner_size_bytes,
            runner_tests_sha256=args.runner_tests_sha256,
            runner_tests_size_bytes=args.runner_tests_size_bytes,
        )
        _validate_context_integrity(context)
        if args.source_check_only:
            value = {
                "artifact_or_control_input_opened": False,
                "holdout_or_label_opened": False,
                "source_inventory_complete": True,
                "status": "SOURCE_BYTES_VERIFIED_ONLY",
            }
            exit_code = 0
        elif args.source_control_check_only:
            value = qualification_result(context)
            exit_code = 0 if value.get("status") == "SOURCE_CONTROL_QUALIFIED_REPEATABLE" else 1
        else:
            value = run_one_shot(
                context,
                authorization_sha256=str(args.authorization_sha256),
                authorization_size_bytes=int(args.authorization_size_bytes),
            )
            exit_code = 0 if value.get("status") == "DIAGNOSTIC_COMPLETE_NONAUTHORIZING" else 1
    except BaseException:
        value = {
            "check_code": "C9_RUNNER_REJECTED_OR_FAILED",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "status": "FAIL_CLOSED",
        }
        exit_code = 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
