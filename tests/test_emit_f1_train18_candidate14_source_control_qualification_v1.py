from __future__ import annotations

import ast
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
import types
from typing import Any

import pytest

from scripts import emit_f1_train18_candidate14_source_control_qualification_v1 as emitter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/emit_f1_train18_candidate14_source_control_qualification_v1.py"
TEST_SOURCE = ROOT / "tests/test_emit_f1_train18_candidate14_source_control_qualification_v1.py"


def _base_cli() -> list[str]:
    return [
        "--emitter-sha256",
        "0" * 64,
        "--emitter-size-bytes",
        "1",
        "--emitter-tests-sha256",
        "1" * 64,
        "--emitter-tests-size-bytes",
        "1",
    ]


def _record_value(qualification: Any = None) -> dict[str, Any]:
    dynamic = {
        "control_read_inventory_commitment_sha256": "1" * 64,
        "resource_capability_result_commitment_sha256": "2" * 64,
        "worker_source_inventory_commitment_sha256": "3" * 64,
        "worker_source_result_commitment_sha256": "4" * 64,
    }
    if qualification is not None:
        dynamic.update(
            {
                "control_read_inventory_commitment_sha256": qualification.control_read_inventory_commitment_sha256,
                "resource_capability_result_commitment_sha256": qualification.resource_capability_result_commitment_sha256,
                "worker_source_inventory_commitment_sha256": qualification.worker_source_inventory_commitment_sha256,
                "worker_source_result_commitment_sha256": qualification.worker_source_result_commitment_sha256,
            }
        )
    value: dict[str, Any] = {
        "adapter_or_hypothesis_modified": False,
        "cache_manifest_opened": False,
        "check_code": "ALL_C14_SOURCE_CONTROL_CHECKS_PASSED",
        "combined_cpu_limit_seconds": "240.000",
        "control_binding_commitment_sha256": emitter.EXPECTED_CONTROL_BINDING_SHA256,
        "engineering_attempt_id": "CANDIDATE14_FULL_POOL_ADAPTER_PREFLIGHT_V1",
        "holdout_label_evaluator_metric_fit_full28_opened_or_started": False,
        "kind": "sherman_qc_f1_aug30_train18_candidate14_full_pool_adapter_qualification_v1",
        "memory_ceiling_bytes": 8 * 1024 * 1024 * 1024,
        "memory_enforcement_is_hard_rlimit_as": False,
        "memory_enforcement_kind": "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING",
        "poll_interval_seconds": "0.010",
        "repeatable": True,
        "sampling_gap_terminal": True,
        "sampling_max_gap_seconds": "0.050",
        "sampling_overshoot_between_polls_possible": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1",
        "source_inventory_commitment_sha256": emitter.EXPECTED_SOURCE_INVENTORY_SHA256,
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        "terminal_publication_reserved_outside_compute_resource_envelope": True,
        "terminal_publication_still_anchor_integrity_checked_when_receipt_available": True,
        "wall_timeout_seconds": "300.000",
        "worker_cpu_hard_limit_seconds": 241,
        "worker_cpu_soft_limit_seconds": 240,
        **dict(emitter.EXPECTED_STATIC_QUALIFICATION_COMMITMENTS),
        **dynamic,
    }
    assert set(value) == emitter.QUALIFICATION_KEYS
    return value


def _record(qualification: Any = None) -> bytes:
    return emitter.canonical_bytes(_record_value(qualification))


def _verified_command(mode: str, *, source_sha256: str | None = None) -> list[str]:
    source_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    source_sha = source_sha256 or emitter.sha256_bytes(source_bytes)
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        emitter.VERIFIED_EMITTER_BOOTSTRAP_CODE,
        str(SOURCE),
        source_sha,
        str(len(source_bytes)),
        "--emitter-sha256",
        source_sha,
        "--emitter-size-bytes",
        str(len(source_bytes)),
        "--emitter-tests-sha256",
        emitter.sha256_bytes(test_bytes),
        "--emitter-tests-size-bytes",
        str(len(test_bytes)),
        mode,
    ]


def _clean_boundary_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any]:
    class BootstrapContext:
        pass

    class GuardedReader:
        pass

    class ControlQualification:
        pass

    resource = types.ModuleType(emitter.PRIVATE_RESOURCE_MODULE_NAME)
    resource.__file__ = str(
        ROOT / "backend/f1_train18_candidate14_darwin_resource_capability_v1.py"
    )

    class DarwinRusageSampler:
        pass

    class SampledDarwinCeiling:
        pass

    def capability_probe() -> dict[str, Any]:
        return {}

    resource.DarwinRusageSampler = DarwinRusageSampler
    resource.SampledDarwinCeiling = SampledDarwinCeiling
    resource.run_repeatable_source_only_capability_probe = capability_probe
    monkeypatch.setitem(sys.modules, emitter.PRIVATE_RESOURCE_MODULE_NAME, resource)

    source_entries: list[MappingProxyType] = []
    source_bindings: dict[str, MappingProxyType] = {}
    for index, (name, path, sha256, size_bytes) in enumerate(
        emitter.EXPECTED_SOURCE_SPECS, start=1
    ):
        relative = str(path.relative_to(ROOT))
        source_entries.append(
            MappingProxyType(
                {
                    "device_id": index,
                    "inode": index + 100,
                    "path": relative,
                    "purpose": f"CANDIDATE14_SOURCE_{name.upper()}",
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
            )
        )
        source_bindings[name] = MappingProxyType(
            {"path": relative, "sha256": sha256, "size_bytes": size_bytes}
        )
    context = BootstrapContext()
    context.resource_core = resource
    context.source_bindings = MappingProxyType(source_bindings)
    context.source_entries = tuple(source_entries)
    context.source_commitment_sha256 = emitter.sha256_bytes(
        emitter.canonical_bytes(
            {name: dict(value) for name, value in sorted(source_bindings.items())}
        )
    )
    assert context.source_commitment_sha256 == emitter.EXPECTED_SOURCE_INVENTORY_SHA256
    context._resource_functions = (
        resource.DarwinRusageSampler,
        resource.SampledDarwinCeiling,
        resource.run_repeatable_source_only_capability_probe,
    )
    context._token = object()

    control_entries: list[dict[str, Any]] = []
    for index, (purpose, path) in enumerate(
        zip(emitter.EXPECTED_CONTROL_PURPOSES, emitter.EXPECTED_CONTROL_PATHS),
        start=1,
    ):
        control_entries.append(
            {
                "device_id": index + 1000,
                "inode": index + 2000,
                "path": str(path.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": f"{index:064x}",
                "size_bytes": index,
            }
        )
    entries = [dict(entry) for entry in source_entries] + control_entries
    purposes = [str(entry["purpose"]) for entry in entries]
    control_paths = set(emitter.EXPECTED_CONTROL_PATHS)
    reader = GuardedReader()
    reader.entries = entries
    reader.raw_fd_opened_purposes = list(purposes)
    reader.verified_inode_purposes = list(purposes)
    reader.read_started_purposes = list(purposes)
    reader.artifact_inventory = None
    reader.denied_attempts = []
    reader.allowed = {}
    reader.consumed_paths = set(control_paths)
    reader.raw_fd_opened_paths = set(control_paths)
    reader.verified_inode_paths = set(control_paths)
    reader.read_started_paths = set(control_paths)
    reader.identities = {
        (int(entry["device_id"]), int(entry["inode"])): ROOT / str(entry["path"])
        for entry in entries
    }

    runner = SimpleNamespace(
        BootstrapContext=BootstrapContext,
        GuardedReader=GuardedReader,
        ControlQualification=ControlQualification,
        CACHE_MANIFEST=ROOT
        / "output/research/f1-aug30-train-only-custody-v2/cache/train_v51_cache_manifest.json",
        CACHE_MANIFEST_SHA256="396f4bd47050fe63ecbc582254efb5647e18fbc77289747936043ff5d656a022",
        CACHE_MANIFEST_SIZE_BYTES=19601,
        ALLOWED_CACHE_ROOT=ROOT
        / "output/research/f1-v51-today-scan-only-development-20260830-r2",
        C14_QUALIFICATION=ROOT
        / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_qualification_v1.json",
        C14_AUTHORIZATION=ROOT
        / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_authorization_v1.json",
        C14_ANCHOR=ROOT
        / "research/f1_aug30_train18_candidate14_full_pool_adapter_preflight_consumption_v1.json",
        PAIR_COUNT=18,
    )
    train_ids = tuple(f"T{index:03d}" for index in range(1, 19))
    scan_by_case = MappingProxyType(
        {case_id: f"{index + 100:064x}" for index, case_id in enumerate(train_ids)}
    )
    qualification = ControlQualification()
    qualification.reader = reader
    qualification.train_ids = train_ids
    qualification.scan_by_case = scan_by_case
    qualification.train_commitment_sha256 = emitter.sha256_bytes(
        emitter.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    qualification.manifest_path = runner.CACHE_MANIFEST
    qualification.manifest_sha256 = runner.CACHE_MANIFEST_SHA256
    qualification.manifest_size_bytes = runner.CACHE_MANIFEST_SIZE_BYTES
    qualification.control_binding_commitment_sha256 = (
        emitter.EXPECTED_CONTROL_BINDING_SHA256
    )
    qualification.control_read_inventory_commitment_sha256 = emitter.sha256_bytes(
        emitter.canonical_bytes(
            {
                "domain": "CANDIDATE14_CONTROL_READ_INVENTORY_V1",
                "reads": control_entries,
            }
        )
    )
    qualification.control_raw_fd_opened_commitment_sha256 = (
        emitter.EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_raw_fd_opened_commitment_sha256"
        ]
    )
    qualification.control_verified_inode_commitment_sha256 = (
        emitter.EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_verified_inode_commitment_sha256"
        ]
    )
    qualification.control_read_started_commitment_sha256 = (
        emitter.EXPECTED_STATIC_QUALIFICATION_COMMITMENTS[
            "control_read_started_commitment_sha256"
        ]
    )
    qualification.resource_capability_result_commitment_sha256 = "2" * 64
    qualification.worker_source_inventory_commitment_sha256 = "3" * 64
    qualification.worker_source_result_commitment_sha256 = "4" * 64
    qualification._context_token = context._token
    return runner, context, qualification


def test_top_level_is_stdlib_only_and_all_frozen_bindings_are_exact() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {"__future__", "builtins", "hashlib", "json", "os", "pathlib", "stat", "sys", "types", "typing"}
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed
    assert emitter.RUNNER_SHA256 == "50eeb6959a18e60d55c4c1159a4a56ed879b4d74e1f4d09e147d60c8bb5e735c"
    assert emitter.RUNNER_SIZE_BYTES == 216217
    assert emitter.RUNNER_TESTS_SHA256 == "d6f62b6d6d301df8204887ce64a5778780981cf52ec9050e7c6a7651b2588695"
    assert emitter.RUNNER_TESTS_SIZE_BYTES == 103948
    assert emitter.EXPECTED_QUALIFICATION_SHA256 == "667d0cc9527e86e21c3d6b537d3f7b06f29a5f33995239f07644853a20a5582a"
    assert emitter.EXPECTED_SOURCE_INVENTORY_SHA256 == "bb8dc1ae00bac889702adf7902ac9d1a32ab28d17874f7751b84965203909392"
    assert emitter.EXPECTED_CONTROL_BINDING_SHA256 == "42ff3117cba8c9bd1f0030bf0fcb93947e20805145ccac4d45dd96e7f16db777"
    assert tuple(name for name, *_ in emitter.EXPECTED_SOURCE_SPECS) == tuple(sorted(name for name, *_ in emitter.EXPECTED_SOURCE_SPECS))
    assert tuple(emitter.EXPECTED_CONTROL_PURPOSES) == (
        "SEALED_SPLIT", "ORIGINAL_PREREGISTRATION", "TRAIN_ALLOWLIST",
        "CANDIDATE13_TERMINAL_DISPOSITION", "TRAIN_CACHE_ATTESTATION",
        "CACHE_CUSTODY_ACCESS_EVIDENCE", "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS", "CACHE_CUSTODY_VALIDATOR",
    )


def test_emitter_has_exact_two_modes_no_default_or_file_write_surface() -> None:
    with pytest.raises(ValueError):
        emitter._parse_args(_base_cli())
    assert emitter._parse_args([*_base_cli(), "--source-check-only"]).source_check_only
    assert emitter._parse_args([*_base_cli(), "--emit-qualification-only"]).emit_qualification_only
    for invalid in (
        [*_base_cli(), "--source-check-only", "--source-check-only"],
        [*_base_cli(), "--source-check"],
        [*_base_cli(), "--source-check-only", "--emit-qualification-only"],
        [*_base_cli(), "--source-check-only", "EXTRA"],
        [*_base_cli()[:-1], "01", "--source-check-only"],
    ):
        with pytest.raises(ValueError):
            emitter._parse_args(invalid)
    source = SOURCE.read_text(encoding="utf-8")
    assert "O_WRONLY" not in source
    assert "O_CREAT" not in source
    assert "write_bytes(" not in source
    assert "open(" not in source.replace("os.open(", "")
    assert "isolated" in emitter.VERIFIED_EMITTER_BOOTSTRAP_CODE


def test_direct_private_and_direct_script_entry_are_deauthorized() -> None:
    with pytest.raises(PermissionError, match="verified private code"):
        emitter._private_main([*_base_cli(), "--source-check-only"])
    source_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(SOURCE),
         "--emitter-sha256", emitter.sha256_bytes(source_bytes),
         "--emitter-size-bytes", str(len(source_bytes)),
         "--emitter-tests-sha256", emitter.sha256_bytes(test_bytes),
         "--emitter-tests-size-bytes", str(len(test_bytes)),
         "--source-check-only"],
        cwd=ROOT, check=False, capture_output=True,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == b'{"check_code":"C14_QUALIFICATION_EMIT_REJECTED","status":"FAIL_CLOSED"}\n'


def test_preloaded_alias_and_import_hook_drift_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emitter, "_IMPORT_HOOK_SEAL", (builtins.__import__, tuple(sys.meta_path), tuple(sys.path_hooks)))
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, types.ModuleType) and type(module) is not types.ModuleType:
            monkeypatch.delitem(sys.modules, name, raising=False)
    alias = types.ModuleType("_candidate14_emitter_alias")
    alias.__file__ = str(emitter.RUNNER_SOURCE)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="source alias"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))
    monkeypatch.delitem(sys.modules, alias.__name__)
    class ModuleSubclass(types.ModuleType):
        pass
    subclass = ModuleSubclass("_candidate14_emitter_subclass")
    monkeypatch.setitem(sys.modules, subclass.__name__, subclass)
    with pytest.raises(RuntimeError, match="module subclass"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))
    monkeypatch.delitem(sys.modules, subclass.__name__)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, object()])
    with pytest.raises(RuntimeError, match="import-hook drift"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))


def test_clean_boundary_accepts_exact_sources_and_nine_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, context, qualification = _clean_boundary_fixture(monkeypatch)
    emitter._validate_post_qualification_boundary(runner, context, qualification)


@pytest.mark.parametrize("contamination", (
    "inventory", "allowed", "manifest_entry", "pool_purpose", "consumed_paths",
    "raw_paths", "verified_paths", "started_paths", "identity", "reordered",
    "denied", "control_binding", "train_commitment",
))
def test_boundary_rejects_manifest_pool_capability_and_vector_drift(monkeypatch: pytest.MonkeyPatch, contamination: str) -> None:
    runner, context, qualification = _clean_boundary_fixture(monkeypatch)
    reader = qualification.reader
    if contamination == "inventory":
        reader.artifact_inventory = object()
    elif contamination == "allowed":
        reader.allowed[emitter.EXPECTED_CONTROL_PATHS[0]] = {}
    elif contamination == "manifest_entry":
        reader.entries.append({"device_id": 9998, "inode": 9999, "path": str(runner.CACHE_MANIFEST.relative_to(ROOT)), "purpose": "TRAIN_CACHE_MANIFEST", "sha256": "a" * 64, "size_bytes": 1})
    elif contamination == "pool_purpose":
        reader.raw_fd_opened_purposes[-1] = "FOREIGN_POOL"
    elif contamination == "consumed_paths":
        reader.consumed_paths = set()
    elif contamination == "raw_paths":
        reader.raw_fd_opened_paths = set()
    elif contamination == "verified_paths":
        reader.verified_inode_paths = set()
    elif contamination == "started_paths":
        reader.read_started_paths = set()
    elif contamination == "identity":
        reader.identities.clear()
    elif contamination == "reordered":
        reader.entries[-1], reader.entries[-2] = reader.entries[-2], reader.entries[-1]
    elif contamination == "denied":
        reader.denied_attempts.append("DENIED")
    elif contamination == "control_binding":
        qualification.control_binding_commitment_sha256 = "0" * 64
    else:
        qualification.train_commitment_sha256 = "0" * 64
    with pytest.raises((PermissionError, ValueError)):
        emitter._validate_post_qualification_boundary(runner, context, qualification)


def test_source_context_rejects_wrong_commitment_or_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, context, _qualification = _clean_boundary_fixture(monkeypatch)
    context.source_commitment_sha256 = "0" * 64
    with pytest.raises(PermissionError):
        emitter._validate_source_context(runner, context)
    runner, context, _qualification = _clean_boundary_fixture(monkeypatch)
    changed = dict(context.source_entries[0])
    changed["sha256"] = "0" * 64
    context.source_entries = (MappingProxyType(changed), *context.source_entries[1:])
    with pytest.raises(PermissionError):
        emitter._validate_source_context(runner, context)


def test_record_validation_requires_exact_c14_schema_and_authorized_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record()
    monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record))
    value = emitter._validate_qualification_record(record)
    assert set(value) == emitter.QUALIFICATION_KEYS
    assert value["check_code"] == "ALL_C14_SOURCE_CONTROL_CHECKS_PASSED"
    for malformed in (record + b"\n", record[:-1], b'{"a":1,"a":1}\n'):
        monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(malformed))
        with pytest.raises(PermissionError):
            emitter._validate_qualification_record(malformed)


@pytest.mark.parametrize(("key", "replacement"), (
    ("check_code", "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED"),
    ("schema_version", True), ("cache_manifest_opened", True),
    ("memory_ceiling_bytes", True), ("source_inventory_commitment_sha256", "0" * 64),
    ("control_binding_commitment_sha256", "0" * 64),
    ("manifest_binding_commitment_sha256", "0" * 64),
))
def test_record_semantic_mutations_reject_even_when_hash_is_reauthorized(monkeypatch: pytest.MonkeyPatch, key: str, replacement: Any) -> None:
    value = _record_value()
    value[key] = replacement
    record = emitter.canonical_bytes(value)
    monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record))
    with pytest.raises(PermissionError):
        emitter._validate_qualification_record(record)


def test_record_extra_key_rejects_even_when_hash_is_reauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _record_value()
    value["extra"] = False
    record = emitter.canonical_bytes(value)
    monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record))
    with pytest.raises(PermissionError):
        emitter._validate_qualification_record(record)


def test_build_calls_only_public_sequence_and_rechecks_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, context, qualification = _clean_boundary_fixture(monkeypatch)
    record = _record(qualification)
    events: list[str] = []
    def bootstrap(**kwargs: Any) -> Any:
        assert kwargs == {"runner_sha256": emitter.RUNNER_SHA256, "runner_size_bytes": emitter.RUNNER_SIZE_BYTES, "runner_tests_sha256": emitter.RUNNER_TESTS_SHA256, "runner_tests_size_bytes": emitter.RUNNER_TESTS_SIZE_BYTES}
        events.append("bootstrap")
        return context
    def qualify(received: Any) -> Any:
        assert received is context
        events.append("qualify")
        return qualification
    def build(received_context: Any, received_qualification: Any) -> bytes:
        assert received_context is context and received_qualification is qualification
        events.append("build")
        return record
    runner.bootstrap_verified_sources = bootstrap
    runner.qualify_source_controls = qualify
    runner.build_qualification_record = build
    functions = emitter._runner_function_snapshot(runner)
    assert runner.bootstrap_verified_sources(runner_sha256=emitter.RUNNER_SHA256, runner_size_bytes=emitter.RUNNER_SIZE_BYTES, runner_tests_sha256=emitter.RUNNER_TESTS_SHA256, runner_tests_size_bytes=emitter.RUNNER_TESTS_SIZE_BYTES) is context
    monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record))
    checks = 0
    original_boundary = emitter._validate_post_qualification_boundary
    def counted_boundary(*args: Any) -> None:
        nonlocal checks
        checks += 1
        original_boundary(*args)
    monkeypatch.setattr(emitter, "_validate_post_qualification_boundary", counted_boundary)
    assert emitter._build_qualification_bytes(runner, context, functions) == record
    assert events == ["bootstrap", "qualify", "build"]
    assert checks == 2


def test_build_rejects_state_contamination_after_record_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, context, qualification = _clean_boundary_fixture(monkeypatch)
    record = _record(qualification)
    def bootstrap(**_kwargs: Any) -> Any:
        return context
    def qualify(_context: Any) -> Any:
        return qualification
    def build(_context: Any, _qualification: Any) -> bytes:
        qualification.reader.allowed[emitter.EXPECTED_CONTROL_PATHS[0]] = {}
        return record
    runner.bootstrap_verified_sources = bootstrap
    runner.qualify_source_controls = qualify
    runner.build_qualification_record = build
    functions = emitter._runner_function_snapshot(runner)
    monkeypatch.setattr(emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record))
    with pytest.raises(PermissionError):
        emitter._build_qualification_bytes(runner, context, functions)


def test_release_removes_exact_runner_resource_and_outer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = types.ModuleType("_candidate14_test_runner")
    resource = types.ModuleType(emitter.PRIVATE_RESOURCE_MODULE_NAME)
    token = object()
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setitem(sys.modules, resource.__name__, resource)
    emitter.__dict__["_C14_OUTER_LAUNCH_TOKEN"] = token
    try:
        emitter._release_verified_runner(runner, token, resource)
        assert runner.__name__ not in sys.modules
        assert resource.__name__ not in sys.modules
        assert "_C14_OUTER_LAUNCH_TOKEN" not in emitter.__dict__
    finally:
        emitter.__dict__.pop("_C14_OUTER_LAUNCH_TOKEN", None)


def test_exact_fresh_process_source_check_only() -> None:
    completed = subprocess.run(_verified_command("--source-check-only"), cwd=ROOT, check=False, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stderr == b""
    assert json.loads(completed.stdout) == {
        "artifact_or_control_input_opened": False,
        "check_code": "C14_QUALIFICATION_EMITTER_EXACT_SOURCES_VERIFIED",
        "qualification_emitted": False,
        "source_inventory_commitment_sha256": emitter.EXPECTED_SOURCE_INVENTORY_SHA256,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_external_bootstrap_ignores_pythonpath_sitecustomize(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    (tmp_path / "sitecustomize.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(_verified_command("--source-check-only"), cwd=ROOT, env=environment, check=False, capture_output=True)
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert not marker.exists()


@pytest.mark.parametrize("mode", ("--source-check-only", "--help"))
def test_wrong_self_hash_or_invalid_mode_fails_with_empty_stdout(mode: str) -> None:
    command = _verified_command(mode, source_sha256="0" * 64) if mode == "--source-check-only" else _verified_command(mode)
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True)
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == b'{"check_code":"C14_QUALIFICATION_EMIT_REJECTED","status":"FAIL_CLOSED"}\n'


def test_no_duplicate_literal_dict_keys() -> None:
    for path in (SOURCE, TEST_SOURCE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                literal_keys = [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                assert len(literal_keys) == len(set(literal_keys))
