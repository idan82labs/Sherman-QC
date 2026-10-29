from __future__ import annotations

import ast
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types
from typing import Any

import pytest

from scripts import emit_f1_train18_candidate13_source_control_qualification_v1 as emitter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "scripts/emit_f1_train18_candidate13_source_control_qualification_v1.py"
)
TEST_SOURCE = (
    ROOT / "tests/test_emit_f1_train18_candidate13_source_control_qualification_v1.py"
)


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


def _record() -> bytes:
    return emitter.canonical_bytes(
        {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "check_code": "ALL_C13_SOURCE_CONTROL_CHECKS_PASSED",
            "holdout_label_evaluator_metric_fit_opened_or_started": False,
            "repeatable": True,
            "scientific_hypothesis_consumed": False,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        }
    )


def _clean_boundary_fixture() -> tuple[Any, Any, Any]:
    source_path = ROOT / "scripts/source.py"
    source_entry = {
        "path": str(source_path.relative_to(ROOT)),
        "purpose": "SOURCE",
    }
    control_entries = [
        {
            "path": str(
                (ROOT / f"research/control-{index}.json").relative_to(ROOT)
            ),
            "purpose": purpose,
        }
        for index, purpose in enumerate(emitter.EXPECTED_CONTROL_PURPOSES)
    ]
    purposes = ["SOURCE", *emitter.EXPECTED_CONTROL_PURPOSES]
    paths = [
        source_path,
        *(ROOT / str(entry["path"]) for entry in control_entries),
    ]
    reader = SimpleNamespace(
        entries=[source_entry, *control_entries],
        raw_fd_opened_purposes=list(purposes),
        verified_inode_purposes=list(purposes),
        read_started_purposes=list(purposes),
        artifact_inventory=None,
        denied_attempts=[],
        allowed={path: {} for path in paths},
        consumed_paths=set(paths),
    )
    runner = SimpleNamespace(
        CACHE_MANIFEST=ROOT / "output/research/custody/cache/manifest.json",
        ALLOWED_CACHE_ROOT=ROOT / "output/research/pool-cache",
    )
    context = SimpleNamespace(source_entries=(source_entry,))
    qualification = SimpleNamespace(reader=reader)
    return runner, context, qualification


def test_top_level_is_stdlib_only_and_frozen_runner_binding_is_exact() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "builtins",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "stat",
        "sys",
        "types",
        "typing",
    }
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed
    assert emitter.RUNNER_SHA256 == (
        "b56c2de4b47a8e8eb77194d99d381fe385b8c4782330614ee0a5190eb61ca641"
    )
    assert emitter.RUNNER_SIZE_BYTES == 195650
    assert emitter.RUNNER_TESTS_SHA256 == (
        "7792aab139839d3967aa009444279d9f0601d9d1edd65f0265c27b910a5da703"
    )
    assert emitter.RUNNER_TESTS_SIZE_BYTES == 76268
    assert emitter.EXPECTED_QUALIFICATION_SHA256 == (
        "f4a2e6e8f4bf8dbcf3b3151dc7c5fe7439a7c8502ffff6ea99c4417d605ac5d8"
    )


def test_emitter_has_two_explicit_modes_no_default_or_output_path() -> None:
    with pytest.raises(SystemExit):
        emitter._parse_args(_base_cli())
    assert emitter._parse_args(
        [*_base_cli(), "--source-check-only"]
    ).source_check_only
    assert emitter._parse_args(
        [*_base_cli(), "--emit-qualification-only"]
    ).emit_qualification_only
    with pytest.raises(SystemExit):
        emitter._parse_args(
            [*_base_cli(), "--source-check-only", "--emit-qualification-only"]
        )
    source = SOURCE.read_text(encoding="utf-8")
    assert "O_WRONLY" not in source
    assert "O_CREAT" not in source
    assert "write_bytes(" not in source
    assert "publish" not in source.lower()


def test_direct_private_entry_rejects_unverified_import() -> None:
    with pytest.raises(PermissionError, match="verified private code"):
        emitter._private_main([*_base_cli(), "--source-check-only"])


def test_preloaded_alias_and_import_hook_drift_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        emitter,
        "_IMPORT_HOOK_SEAL",
        (builtins.__import__, tuple(sys.meta_path), tuple(sys.path_hooks)),
    )
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, types.ModuleType) and type(module) is not types.ModuleType:
            monkeypatch.delitem(sys.modules, name, raising=False)
    alias = types.ModuleType("_candidate13_emitter_alias")
    alias.__file__ = str(emitter.RUNNER_SOURCE)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="source alias"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))
    monkeypatch.delitem(sys.modules, alias.__name__)

    class ModuleSubclass(types.ModuleType):
        pass

    subclass = ModuleSubclass("_candidate13_emitter_subclass")
    monkeypatch.setitem(sys.modules, subclass.__name__, subclass)
    with pytest.raises(RuntimeError, match="module subclass"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))
    monkeypatch.delitem(sys.modules, subclass.__name__)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, object()])
    with pytest.raises(RuntimeError, match="import-hook drift"):
        emitter._reject_preloaded_sources((emitter.RUNNER_SOURCE,))
    with pytest.raises(RuntimeError, match="import-hook drift"):
        emitter._dispatch_verified_emitter([*_base_cli(), "--source-check-only"])


def test_clean_boundary_accepts_only_source_and_fixed_controls() -> None:
    runner, context, qualification = _clean_boundary_fixture()
    emitter._validate_post_qualification_boundary(runner, context, qualification)


@pytest.mark.parametrize(
    "contamination",
    (
        "inventory",
        "manifest_entry",
        "manifest_allowed",
        "pool_consumed",
        "pool_raw",
        "denied",
    ),
)
def test_boundary_rejects_every_manifest_pool_or_denied_state(
    contamination: str,
) -> None:
    runner, context, qualification = _clean_boundary_fixture()
    reader = qualification.reader
    if contamination == "inventory":
        reader.artifact_inventory = object()
    elif contamination == "manifest_entry":
        reader.entries.append(
            {
                "path": str(runner.CACHE_MANIFEST.relative_to(ROOT)),
                "purpose": "TRAIN_CACHE_MANIFEST",
            }
        )
        for vector in (
            reader.raw_fd_opened_purposes,
            reader.verified_inode_purposes,
            reader.read_started_purposes,
        ):
            vector.append("TRAIN_CACHE_MANIFEST")
    elif contamination == "manifest_allowed":
        reader.allowed[runner.CACHE_MANIFEST] = {}
    elif contamination == "pool_consumed":
        reader.consumed_paths.add(runner.ALLOWED_CACHE_ROOT / "pool.json")
    elif contamination == "pool_raw":
        reader.raw_fd_opened_purposes.append("CANDIDATE13_PRIMARY_00_POOL")
    else:
        reader.denied_attempts.append({"purpose": "DENIED"})
    with pytest.raises(PermissionError):
        emitter._validate_post_qualification_boundary(runner, context, qualification)


def test_record_validation_requires_exact_canonical_authorized_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    monkeypatch.setattr(
        emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record)
    )
    value = emitter._validate_qualification_record(record)
    assert value["status"] == "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
    with pytest.raises(PermissionError):
        emitter._validate_qualification_record(record + b"\n")
    changed = bytearray(record)
    changed[-2] = ord(" ")
    with pytest.raises(PermissionError):
        emitter._validate_qualification_record(bytes(changed))


def test_build_calls_only_public_sequence_and_rechecks_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, context, qualification = _clean_boundary_fixture()
    record = _record()
    events: list[str] = []

    def bootstrap(**_kwargs: Any) -> Any:
        events.append("bootstrap")
        return context

    def qualify(received: Any) -> Any:
        assert received is context
        events.append("qualify")
        return qualification

    def build(received_context: Any, received_qualification: Any) -> bytes:
        assert received_context is context
        assert received_qualification is qualification
        events.append("build")
        return record

    runner.bootstrap_verified_sources = bootstrap
    runner.qualify_source_controls = qualify
    runner.build_qualification_record = build
    monkeypatch.setattr(
        emitter, "EXPECTED_QUALIFICATION_SHA256", emitter.sha256_bytes(record)
    )
    assert emitter._build_qualification_bytes(runner) == record
    assert events == ["bootstrap", "qualify", "build"]


def test_exact_fresh_process_source_check_only() -> None:
    source_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--emitter-sha256",
            emitter.sha256_bytes(source_bytes),
            "--emitter-size-bytes",
            str(len(source_bytes)),
            "--emitter-tests-sha256",
            emitter.sha256_bytes(test_bytes),
            "--emitter-tests-size-bytes",
            str(len(test_bytes)),
            "--source-check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    value = json.loads(completed.stdout)
    assert value["status"] == "SOURCE_BYTES_VERIFIED_ONLY"
    assert value["artifact_or_control_input_opened"] is False
    assert completed.stderr == b""


def test_wrong_self_hash_fails_with_empty_stdout() -> None:
    test_bytes = TEST_SOURCE.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--emitter-sha256",
            "0" * 64,
            "--emitter-size-bytes",
            str(SOURCE.stat().st_size),
            "--emitter-tests-sha256",
            emitter.sha256_bytes(test_bytes),
            "--emitter-tests-size-bytes",
            str(len(test_bytes)),
            "--source-check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert b"C13_QUALIFICATION_EMIT_REJECTED" in completed.stderr


def test_help_is_fail_closed_with_valid_self_binding_and_empty_stdout() -> None:
    source_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--emitter-sha256",
            emitter.sha256_bytes(source_bytes),
            "--emitter-size-bytes",
            str(len(source_bytes)),
            "--emitter-tests-sha256",
            emitter.sha256_bytes(test_bytes),
            "--emitter-tests-size-bytes",
            str(len(test_bytes)),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert b"C13_QUALIFICATION_EMIT_REJECTED" in completed.stderr
