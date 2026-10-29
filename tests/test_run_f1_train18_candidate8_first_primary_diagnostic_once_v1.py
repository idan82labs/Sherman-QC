from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_f1_train18_candidate8_first_primary_diagnostic_once_v1 as runner


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _source_bindings() -> dict[str, tuple[str, int]]:
    paths = {
        "runner": runner.RUNNER_SOURCE,
        "runner_tests": runner.RUNNER_TESTS_SOURCE,
        "runtime": runner.C8_RUNTIME_SOURCE,
        "runtime_tests": runner.C8_RUNTIME_TESTS_SOURCE,
        "adapter": runner.C8_ADAPTER_SOURCE,
        "adapter_tests": runner.C8_ADAPTER_TESTS_SOURCE,
        "hypothesis": runner.C6_HYPOTHESIS_SOURCE,
        "candidate7_runtime": runner.C7_RUNTIME_SOURCE,
    }
    return {
        name: (runner.sha256_bytes(path.read_bytes()), len(path.read_bytes()))
        for name, path in paths.items()
    }


def _source_command() -> list[str]:
    values = _source_bindings()
    return [
        sys.executable,
        str(runner.RUNNER_SOURCE),
        "--runner-sha256",
        values["runner"][0],
        "--runner-size-bytes",
        str(values["runner"][1]),
        "--runner-tests-sha256",
        values["runner_tests"][0],
        "--runner-tests-size-bytes",
        str(values["runner_tests"][1]),
        "--runtime-sha256",
        values["runtime"][0],
        "--runtime-tests-sha256",
        values["runtime_tests"][0],
        "--adapter-sha256",
        values["adapter"][0],
        "--adapter-tests-sha256",
        values["adapter_tests"][0],
        "--hypothesis-sha256",
        values["hypothesis"][0],
        "--candidate7-runtime-sha256",
        values["candidate7_runtime"][0],
    ]


def _bootstrap_entry(path: Path, purpose: str) -> dict[str, object]:
    data = path.read_bytes()
    metadata = os.lstat(path)
    return {
        "device_id": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "path": str(path.relative_to(runner.ROOT)),
        "purpose": purpose,
        "sha256": _sha(data),
        "size_bytes": len(data),
    }


def _synthetic_manifest_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[runner.ControlQualification, dict[str, object], bytes, bytes]:
    root = tmp_path / "repo"
    root.mkdir()
    cache_root = root / "cache"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", cache_root)
    train_ids = tuple(f"C{index:02d}" for index in range(18))
    scan_by_case = {case: _sha(f"scan-{case}") for case in train_ids}
    train_commitment = runner.sha256_bytes(
        runner.canonical_bytes(
            [{"case_id": case, "scan_sha256": scan_by_case[case]} for case in train_ids]
        )
    )
    semantic_bytes = b'{"synthetic":"semantic"}\n'
    pool_bytes = b'{"synthetic":"pool"}\n'
    records: list[dict[str, object]] = []
    first_semantic_path: Path | None = None
    first_pool_path: Path | None = None
    for index, case in enumerate(train_ids, start=1):
        runtime_id = f"T{index:03d}"
        primary_semantic = cache_root / "primary" / runtime_id / "semantic_result.json"
        primary_pool = cache_root / "primary" / runtime_id / "pool" / f"{runtime_id}.json"
        replay_semantic = cache_root / "replay" / runtime_id / "semantic_result.json"
        replay_pool = cache_root / "replay" / runtime_id / "pool" / f"{runtime_id}.json"
        if index == 1:
            first_semantic_path = primary_semantic
            first_pool_path = primary_pool
            primary_semantic.parent.mkdir(parents=True)
            primary_pool.parent.mkdir(parents=True)
            primary_semantic.write_bytes(semantic_bytes)
            primary_pool.write_bytes(pool_bytes)
        for pass_name, semantic_path, pool_path in (
            ("primary", primary_semantic, primary_pool),
            ("replay", replay_semantic, replay_pool),
        ):
            records.append(
                {
                    "case_id": case,
                    "pass": pass_name,
                    "pool_path": str(pool_path.relative_to(root)),
                    "pool_sha256": _sha(pool_bytes if index == 1 else f"pool-{case}"),
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case],
                    "semantic_path": str(semantic_path.relative_to(root)),
                    "semantic_sha256": _sha(
                        semantic_bytes if index == 1 else f"semantic-{case}"
                    ),
                }
            )
    assert first_semantic_path is not None and first_pool_path is not None
    semantic_metadata = os.lstat(first_semantic_path)
    pool_metadata = os.lstat(first_pool_path)
    manifest = {
        "case_count": 18,
        "holdout_entry_count": 0,
        "kind": runner.CACHE_MANIFEST_KIND,
        "record_count": 36,
        "records": records,
        "schema_version": 1,
        "source_only": True,
        "split_sha256": runner.FIXED_CONTROLS["SEALED_SPLIT"][1],
        "train_allowlist_sha256": runner.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
        "train_set_commitment_sha256": train_commitment,
        "v51_scan_only_manifest_sha256": runner.V51_SCAN_ONLY_MANIFEST_SHA256,
    }
    manifest_path = root / "manifest.json"
    manifest_bytes = runner.canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    semantic_evidence = {
        "case_id": train_ids[0],
        "device_id": int(semantic_metadata.st_dev),
        "inode": int(semantic_metadata.st_ino),
        "path": records[0]["semantic_path"],
        "purpose": "TRAIN_PRIMARY_V51_SEMANTIC",
        "sha256": records[0]["semantic_sha256"],
        "size_bytes": len(semantic_bytes),
    }
    pool_evidence = {
        "case_id": train_ids[0],
        "device_id": int(pool_metadata.st_dev),
        "inode": int(pool_metadata.st_ino),
        "path": records[0]["pool_path"],
        "purpose": "TRAIN_PRIMARY_V51_POOL",
        "sha256": records[0]["pool_sha256"],
        "size_bytes": len(pool_bytes),
    }
    reader = runner.GuardedReader()
    nonce = object()
    reader.first_primary_nonce = nonce
    qualification = runner.ControlQualification(
        reader=reader,
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment=train_commitment,
        manifest_path=manifest_path,
        manifest_sha256=_sha(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
        candidate7_disposition_bytes=b"{}\n",
        c7_semantic_evidence=semantic_evidence,
        c7_pool_evidence=pool_evidence,
        control_binding_commitment_sha256=_sha("controls"),
        _pair_capability_nonce=nonce,
    )
    return qualification, manifest, semantic_bytes, pool_bytes


def test_runner_top_level_is_stdlib_only() -> None:
    tree = ast.parse(runner.RUNNER_SOURCE.read_text(encoding="utf-8"))
    forbidden = ("backend", "numpy", "scipy", "sklearn")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(forbidden) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not str(node.module or "").startswith(forbidden)


def test_source_check_only_verifies_eight_sources_and_no_artifacts() -> None:
    completed = subprocess.run(
        [*_source_command(), "--source-check-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "artifact_or_control_input_opened": False,
        "holdout_or_label_opened": False,
        "source_inventory_complete": True,
        "status": "SOURCE_BYTES_VERIFIED_ONLY",
    }


def test_no_mode_rejects_without_source_or_artifact_dispatch() -> None:
    completed = subprocess.run(
        _source_command(), cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 2
    assert "one of the arguments" in completed.stderr
    assert completed.stdout == ""


def test_diagnostic_mode_requires_authorization_sha_and_size() -> None:
    completed = subprocess.run(
        [*_source_command(), "--diagnostic-once"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "requires exact authorization SHA and size" in completed.stderr


def test_help_does_not_import_controlled_or_numeric_modules() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(runner.RUNNER_SOURCE), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for forbidden in (
        "f1_train18_candidate6_conservative_recall_recovery_v1",
        "f1_train18_candidate7_conservative_recall_recovery_v1",
        "f1_train18_candidate8_diagnostic_recall_wrapper_v1",
        "numpy",
        "scipy",
        "sklearn",
    ):
        assert forbidden not in completed.stderr


def test_source_control_dispatch_cannot_call_one_shot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bindings = _source_bindings()
    context = object()
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "qualification_result",
        lambda _context: {
            "artifact_source_inventory_opened": False,
            "cache_manifest_opened": False,
            "candidate7_consumed_first_pair_evidence_validated": True,
            "check_code": "ALL_C8_SOURCE_CONTROL_CHECKS_PASSED",
            "control_binding_commitment_sha256": "1" * 64,
            "failure_stage": None,
            "holdout_or_label_opened": False,
            "repeatable": True,
            "scientific_hypothesis_consumed": False,
            "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
        },
    )
    monkeypatch.setattr(
        runner,
        "run_one_shot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("one-shot dispatch reached")
        ),
    )
    command = [*_source_command()[2:], "--source-control-check-only"]
    monkeypatch.setattr(sys, "argv", [str(runner.RUNNER_SOURCE), *command])
    assert runner.main() == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "SOURCE_CONTROL_QUALIFIED_REPEATABLE"
    assert value["cache_manifest_opened"] is False


def test_failed_source_control_dispatch_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context = object()
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "qualification_result",
        lambda _context: {
            "artifact_source_inventory_opened": False,
            "check_code": "SYNTHETIC_CONTROL_FAILURE",
            "failure_stage": "SOURCE_CONTROL_QUALIFICATION",
            "holdout_or_label_opened": False,
            "repeatable": True,
            "status": "SOURCE_CONTROL_QUALIFICATION_FAILED",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(runner.RUNNER_SOURCE), *_source_command()[2:], "--source-control-check-only"],
    )
    assert runner.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_CONTROL_QUALIFICATION_FAILED"


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    (
        ("DIAGNOSTIC_COMPLETE_NONAUTHORIZING", 0),
        ("TERMINAL_ENGINEERING_FAILURE", 1),
    ),
)
def test_diagnostic_dispatch_exit_code_distinguishes_completed_from_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_exit: int,
) -> None:
    context = object()
    monkeypatch.setattr(runner, "bootstrap_verified_sources", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "run_one_shot",
        lambda *_args, **_kwargs: {
            "check_code": (
                "C8_SEMANTIC_VALIDATION_FAILED"
                if status == "DIAGNOSTIC_COMPLETE_NONAUTHORIZING"
                else None
            ),
            "status": status,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner.RUNNER_SOURCE),
            *_source_command()[2:],
            "--diagnostic-once",
            "--authorization-sha256",
            "9" * 64,
            "--authorization-size-bytes",
            "1",
        ],
    )
    assert runner.main() == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_canonical_qualification_builder_matches_authorization_schema() -> None:
    context = SimpleNamespace(
        source_bindings={
            "runner": {"path": "runner.py", "sha256": "1" * 64, "size_bytes": 1}
        }
    )
    qualification = SimpleNamespace(control_binding_commitment_sha256="2" * 64)
    value = json.loads(
        runner.build_qualification_record(context, qualification)  # type: ignore[arg-type]
    )
    assert value["kind"] == runner.QUALIFICATION_KIND
    assert value["source_bindings"] == context.source_bindings
    assert value["cache_manifest_opened"] is False
    assert value["scientific_hypothesis_consumed"] is False


def test_preloaded_symlink_alias_rejects_before_source_read(tmp_path: Path) -> None:
    bindings = _source_bindings()
    alias = tmp_path / "alias.py"
    alias.symlink_to(runner.C6_HYPOTHESIS_SOURCE)
    sentinel = tmp_path / "read"
    code = f"""
import importlib.util, pathlib, sys
alias = pathlib.Path({str(alias)!r})
spec = importlib.util.spec_from_file_location('arbitrary_alias', alias)
loaded = importlib.util.module_from_spec(spec)
sys.modules['arbitrary_alias'] = loaded
spec.loader.exec_module(loaded)
path = pathlib.Path({str(runner.RUNNER_SOURCE)!r})
spec2 = importlib.util.spec_from_file_location('isolated_c8_runner', path)
module = importlib.util.module_from_spec(spec2)
sys.modules['isolated_c8_runner'] = module
spec2.loader.exec_module(module)
module.FORBIDDEN_PRELOADED_MODULES = ('never.preloaded',)
def forbidden_read(*args, **kwargs):
    pathlib.Path({str(sentinel)!r}).write_text('READ')
    raise AssertionError('read before alias rejection')
module._read_exact_source = forbidden_read
try:
    module.bootstrap_verified_sources(
        runner_sha256={bindings['runner'][0]!r}, runner_size_bytes={bindings['runner'][1]},
        runner_tests_sha256={bindings['runner_tests'][0]!r}, runner_tests_size_bytes={bindings['runner_tests'][1]},
        runtime_sha256={bindings['runtime'][0]!r}, runtime_tests_sha256={bindings['runtime_tests'][0]!r},
        adapter_sha256={bindings['adapter'][0]!r}, adapter_tests_sha256={bindings['adapter_tests'][0]!r},
        hypothesis_sha256={bindings['hypothesis'][0]!r}, candidate7_runtime_sha256={bindings['candidate7_runtime'][0]!r},
    )
except RuntimeError as error:
    assert 'preloaded source alias' in str(error)
else:
    raise AssertionError('alias accepted')
assert not pathlib.Path({str(sentinel)!r}).exists()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_tampered_cleared_source_binding_fails_before_module_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _source_bindings()
    monkeypatch.setattr(
        runner,
        "_exec_module",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("executed")),
    )
    with pytest.raises(ValueError, match="cleared byte set"):
        runner.bootstrap_verified_sources(
            runner_sha256=bindings["runner"][0],
            runner_size_bytes=bindings["runner"][1],
            runner_tests_sha256=bindings["runner_tests"][0],
            runner_tests_size_bytes=bindings["runner_tests"][1],
            runtime_sha256="1" * 64,
            runtime_tests_sha256=bindings["runtime_tests"][0],
            adapter_sha256=bindings["adapter"][0],
            adapter_tests_sha256=bindings["adapter_tests"][0],
            hypothesis_sha256=bindings["hypothesis"][0],
            candidate7_runtime_sha256=bindings["candidate7_runtime"][0],
        )


def _authorization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, runner.GuardedReader, str, int]:
    root = tmp_path / "repo"
    research = root / "research"
    custody = root / "custody"
    output = root / "output" / "diagnostic"
    research.mkdir(parents=True)
    custody.mkdir()
    output.parent.mkdir()
    auth_path = research / "authorization.json"
    qualification_path = research / "qualification.json"
    manifest_path = custody / "manifest.json"
    anchor_path = research / "consumption.json"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "C8_AUTHORIZATION", auth_path)
    monkeypatch.setattr(runner, "C8_QUALIFICATION", qualification_path)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", manifest_path)
    monkeypatch.setattr(runner, "C8_CONSUMPTION_ANCHOR", anchor_path)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", root / "cache")
    source_path = root / "runner.py"
    source_path.write_bytes(b"synthetic runner\n")
    source_entry = _bootstrap_entry(source_path, "CANDIDATE8_SOURCE_RUNNER")
    source_bindings = {
        "runner": {
            "path": source_entry["path"],
            "sha256": source_entry["sha256"],
            "size_bytes": source_entry["size_bytes"],
        }
    }
    context = SimpleNamespace(
        source_bindings=source_bindings,
        bootstrap_entries=(source_entry,),
        _context_nonce=object(),
    )
    qualification = {
        "artifact_source_inventory_opened": False,
        "attempt_ordinal": 8,
        "cache_manifest_opened": False,
        "candidate7_consumed_first_pair_evidence_validated": True,
        "check_code": "ALL_C8_SOURCE_CONTROL_CHECKS_PASSED",
        "consumption_anchor_namespace": str(anchor_path.relative_to(root)),
        "control_binding_commitment_sha256": "2" * 64,
        "engineering_attempt_id": runner.ENGINEERING_ATTEMPT_ID,
        "holdout_or_label_opened": False,
        "kind": runner.QUALIFICATION_KIND,
        "output_namespace": str(output.relative_to(root)),
        "repeatable": True,
        "schema_version": 1,
        "scientific_hypothesis_consumed": False,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": source_bindings,
        "status": "SOURCE_CONTROL_QUALIFIED_REPEATABLE",
    }
    qualification_bytes = runner.canonical_bytes(qualification)
    qualification_path.write_bytes(qualification_bytes)
    authorization = {
        "attempt_ordinal": 8,
        "authorization_consumed_before_manifest_open": True,
        "consumption_anchor_namespace": str(anchor_path.relative_to(root)),
        "continuation_authorized": False,
        "engineering_attempt_id": runner.ENGINEERING_ATTEMPT_ID,
        "full28_access_authorized": False,
        "holdout_access_authorized": False,
        "kind": runner.AUTHORIZATION_KIND,
        "label_evaluator_metric_fit_access_authorized": False,
        "manifest_binding": runner._binding(
            manifest_path,
            runner.CACHE_MANIFEST_SHA256,
            runner.CACHE_MANIFEST_SIZE_BYTES,
            kind=runner.CACHE_MANIFEST_KIND,
        ),
        "output_namespace": str(output.relative_to(root)),
        "qualification_record_binding": runner._binding(
            qualification_path,
            _sha(qualification_bytes),
            len(qualification_bytes),
            kind=runner.QUALIFICATION_KIND,
        ),
        "record_selector": runner.RECORD_SELECTOR,
        "retry_after_anchor_authorized": False,
        "schema_version": 1,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "source_bindings": source_bindings,
        "status": "AUTHORIZED_ONCE",
    }
    authorization_bytes = runner.canonical_bytes(authorization)
    auth_path.write_bytes(authorization_bytes)
    return (
        context,
        runner.GuardedReader(entries=[dict(source_entry)]),
        _sha(authorization_bytes),
        len(authorization_bytes),
    )


def test_external_authorization_and_qualification_are_exactly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, reader, auth_sha, auth_size = _authorization_fixture(tmp_path, monkeypatch)
    seal = runner.authorize_diagnostic_once(
        context,  # type: ignore[arg-type]
        reader=reader,
        authorization_sha256=auth_sha,
        authorization_size_bytes=auth_size,
        output_freshness=runner.prevalidate_fresh_output(runner.OUTPUT_ROOT),
    )
    assert seal.control_binding_commitment_sha256 == "2" * 64
    assert [entry["purpose"] for entry in reader.entries] == [
        "CANDIDATE8_SOURCE_RUNNER",
        "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
    ]


def test_authorization_tamper_or_stale_qualification_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _reader, auth_sha, auth_size = _authorization_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        runner.authorize_diagnostic_once(
            context,  # type: ignore[arg-type]
            reader=runner.GuardedReader(
                entries=[dict(entry) for entry in context.bootstrap_entries]
            ),
            authorization_sha256="9" * 64,
            authorization_size_bytes=auth_size,
            output_freshness=runner.prevalidate_fresh_output(runner.OUTPUT_ROOT),
        )
    runner.C8_QUALIFICATION.write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        runner.authorize_diagnostic_once(
            context,  # type: ignore[arg-type]
            reader=runner.GuardedReader(
                entries=[dict(entry) for entry in context.bootstrap_entries]
            ),
            authorization_sha256=auth_sha,
            authorization_size_bytes=auth_size,
            output_freshness=runner.prevalidate_fresh_output(runner.OUTPUT_ROOT),
        )


def test_authorization_seal_rejects_context_substitution_empty_reader_and_entry_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, reader, auth_sha, auth_size = _authorization_fixture(tmp_path, monkeypatch)
    seal = runner.authorize_diagnostic_once(
        context,  # type: ignore[arg-type]
        reader=reader,
        authorization_sha256=auth_sha,
        authorization_size_bytes=auth_size,
        output_freshness=runner.prevalidate_fresh_output(runner.OUTPUT_ROOT),
    )
    substituted = SimpleNamespace(
        source_bindings=context.source_bindings,
        bootstrap_entries=context.bootstrap_entries,
        _context_nonce=object(),
    )
    with pytest.raises(PermissionError, match="context"):
        runner._validate_authorization_seal(
            substituted, seal, require_qualified_controls=False  # type: ignore[arg-type]
        )
    forged = runner.AuthorizationSeal(
        reader=runner.GuardedReader(),
        authorization_sha256=seal.authorization_sha256,
        authorization_size_bytes=seal.authorization_size_bytes,
        qualification_sha256=seal.qualification_sha256,
        qualification_size_bytes=seal.qualification_size_bytes,
        control_binding_commitment_sha256=seal.control_binding_commitment_sha256,
        output_freshness=seal.output_freshness,
        source_binding_commitment_sha256=seal.source_binding_commitment_sha256,
        authority_prefix_commitment_sha256=seal.authority_prefix_commitment_sha256,
        anchor_parent_identity=seal.anchor_parent_identity,
        _context_nonce=context._context_nonce,
        _sentinel=runner._SEAL_SENTINEL,
    )
    with pytest.raises(PermissionError, match="prefix"):
        runner._validate_authorization_seal(
            context, forged, require_qualified_controls=False  # type: ignore[arg-type]
        )
    seal.reader.entries[0]["sha256"] = "8" * 64
    with pytest.raises(PermissionError, match="binding|commitment|prefix"):
        runner._validate_authorization_seal(
            context, seal, require_qualified_controls=False  # type: ignore[arg-type]
        )


def test_anchor_parent_swap_after_authorization_is_rejected_before_o_excl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, reader, auth_sha, auth_size = _authorization_fixture(tmp_path, monkeypatch)
    seal = runner.authorize_diagnostic_once(
        context,  # type: ignore[arg-type]
        reader=reader,
        authorization_sha256=auth_sha,
        authorization_size_bytes=auth_size,
        output_freshness=runner.prevalidate_fresh_output(runner.OUTPUT_ROOT),
    )
    research = runner.C8_CONSUMPTION_ANCHOR.parent
    moved = research.parent / "moved-research-before-anchor"
    research.rename(moved)
    research.mkdir()
    with pytest.raises(PermissionError, match="anchor parent"):
        runner._validate_authorization_seal(
            context, seal, require_qualified_controls=False  # type: ignore[arg-type]
        )
    assert not runner.C8_CONSUMPTION_ANCHOR.exists()


def test_guarded_reader_denies_replay_holdout_labels_and_full28(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", root / "cache")
    reader = runner.GuardedReader()
    for relative, purpose in (
        ("cache/replay/T001.json", "TRAIN_REPLAY_POOL"),
        ("holdout/value.json", "CONTROL"),
        ("labels/value.json", "CONTROL"),
        ("full28/value.json", "CONTROL"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
        with pytest.raises(PermissionError):
            reader.authorize(path, purpose=purpose, sha256=_sha(b"{}\n"), size_bytes=3)


def test_manifest_validator_binds_exact_c7_consumed_first_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, manifest, semantic, pool = _synthetic_manifest_fixture(tmp_path, monkeypatch)
    first = runner._validate_manifest_first_primary(manifest, qualification)
    assert first.semantic_sha256 == _sha(semantic)
    assert first.pool_sha256 == _sha(pool)
    assert first.commitment_sha256 not in {first.semantic_sha256, first.pool_sha256}


def test_first_primary_capability_rejects_forgery_retarget_and_duplicate_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(tmp_path, monkeypatch)
    first = runner._validate_manifest_first_primary(manifest, qualification)
    replay = runner.ALLOWED_CACHE_ROOT / "replay" / "T001" / "semantic_result.json"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"{}\n")
    forged = SimpleNamespace(
        _sentinel=runner._FIRST_PRIMARY_BINDING_SENTINEL,
        _reader=qualification.reader,
        _qualification_nonce=qualification._pair_capability_nonce,
        semantic_path=replay,
        semantic_sha256=_sha(b"{}\n"),
        semantic_size_bytes=3,
        pool_path=replay,
        pool_sha256=_sha(b"{}\n"),
        pool_size_bytes=3,
    )
    with pytest.raises(PermissionError, match="CAPABILITY"):
        qualification.reader.authorize(
            replay,
            purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
            sha256=_sha(b"{}\n"),
            size_bytes=3,
            first_primary_binding=forged,
        )
    later = runner.ALLOWED_CACHE_ROOT / "primary" / "T002" / "semantic_result.json"
    later.parent.mkdir(parents=True)
    later.write_bytes(b"{}\n")
    with pytest.raises(PermissionError, match="CAPABILITY"):
        qualification.reader.authorize(
            later,
            purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
            sha256=_sha(b"{}\n"),
            size_bytes=3,
            first_primary_binding=first,
        )
    qualification.reader.authorize(
        first.semantic_path,
        purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        sha256=first.semantic_sha256,
        size_bytes=first.semantic_size_bytes,
        first_primary_binding=first,
    )
    qualification.reader.read_bytes(
        first.semantic_path, purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC"
    )
    with pytest.raises(PermissionError, match="DUPLICATE_READ"):
        qualification.reader.read_bytes(
            first.semantic_path, purpose="TRAIN_FIRST_PRIMARY_V51_SEMANTIC"
        )


def test_exact_read_inventory_rejects_any_extra_or_wrong_purpose() -> None:
    source_entry = {
        "device_id": 1,
        "inode": 1,
        "path": "source.py",
        "purpose": "CANDIDATE8_SOURCE_RUNNER",
        "sha256": "1" * 64,
        "size_bytes": 1,
    }
    purposes = [
        "CANDIDATE8_DIAGNOSTIC_AUTHORIZATION",
        "CANDIDATE8_DIAGNOSTIC_QUALIFICATION_RECORD",
        *runner.FIXED_CONTROLS,
        "CACHE_CUSTODY_ACCESS_EVIDENCE",
        "CACHE_CUSTODY_BUILDER",
        "CACHE_CUSTODY_STAGED_VALIDATOR_PASS",
        "CACHE_CUSTODY_VALIDATOR",
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ]
    entries = [source_entry]
    entries.extend(
        {
            "device_id": 1,
            "inode": index + 2,
            "path": f"entry-{index}.json",
            "purpose": purpose,
            "sha256": f"{(index % 9) + 1}" * 64,
            "size_bytes": 1,
        }
        for index, purpose in enumerate(purposes)
    )
    reader = runner.GuardedReader(
        entries=[dict(entry) for entry in entries],
        raw_fd_opened_purposes=list(purposes),
        verified_inode_purposes=list(purposes),
        read_started_purposes=list(purposes),
    )
    qualification = SimpleNamespace(reader=reader)
    context = SimpleNamespace(bootstrap_entries=(source_entry,))
    runner._validate_exact_read_inventory(context, qualification)  # type: ignore[arg-type]
    reader.entries.append(
        {
            "device_id": 1,
            "inode": 999,
            "path": "extra.json",
            "purpose": "TRAIN_FIRST_PRIMARY_V51_POOL",
            "sha256": "9" * 64,
            "size_bytes": 1,
        }
    )
    with pytest.raises(runner.CheckFailure, match="EXACT_READ_INVENTORY"):
        runner._validate_exact_read_inventory(context, qualification)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    ("wrong_pass", "wrong_c7_path", "wrong_c7_sha", "reordered", "replay_hash"),
)
def test_manifest_validator_fails_closed_on_first_pair_or_order_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(tmp_path, monkeypatch)
    records = manifest["records"]
    assert isinstance(records, list)
    if mutation == "wrong_pass":
        records[0]["pass"] = "replay"
    elif mutation == "wrong_c7_path":
        qualification.c7_semantic_evidence["path"] = "wrong.json"  # type: ignore[index]
    elif mutation == "wrong_c7_sha":
        qualification.c7_pool_evidence["sha256"] = "1" * 64  # type: ignore[index]
    elif mutation == "reordered":
        records[0], records[1] = records[1], records[0]
    elif mutation == "replay_hash":
        records[1]["pool_sha256"] = "2" * 64
    with pytest.raises(runner.CheckFailure):
        runner._validate_manifest_first_primary(manifest, qualification)


def test_read_and_diagnose_opens_only_manifest_and_first_primary_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(tmp_path, monkeypatch)
    manifest_bytes = runner.canonical_bytes(manifest)
    qualification.manifest_path.write_bytes(manifest_bytes)
    object.__setattr__(qualification, "manifest_sha256", _sha(manifest_bytes))
    object.__setattr__(qualification, "manifest_size_bytes", len(manifest_bytes))
    monkeypatch.setattr(runner, "revalidate_consumption_anchor", lambda _seal: None)
    result = {
        "check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
        "failure_phase": None,
    }
    runtime = SimpleNamespace(
        _validate_result=lambda _value: None,
        PASS_CODE=result["check_code"],
        FAILURE_CODES={},
    )
    adapter = SimpleNamespace(diagnose_bound_pair=lambda *_args, **_kwargs: dict(result))
    context = SimpleNamespace(adapter=adapter, adapter_context=object(), runtime=runtime)
    seal = SimpleNamespace()
    observed, _first = runner.read_and_diagnose_first_primary(  # type: ignore[arg-type]
        context, qualification, seal
    )
    assert observed == result
    assert [entry["purpose"] for entry in qualification.reader.entries] == [
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ]


def test_returned_malformed_diagnostic_is_bounded_as_result_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(
        tmp_path, monkeypatch
    )
    manifest_bytes = runner.canonical_bytes(manifest)
    qualification.manifest_path.write_bytes(manifest_bytes)
    object.__setattr__(qualification, "manifest_sha256", _sha(manifest_bytes))
    object.__setattr__(qualification, "manifest_size_bytes", len(manifest_bytes))
    monkeypatch.setattr(runner, "revalidate_consumption_anchor", lambda _seal: None)
    returned = False

    def mark_returned() -> None:
        nonlocal returned
        returned = True

    context = SimpleNamespace(
        adapter=SimpleNamespace(
            diagnose_bound_pair=lambda *_args, **_kwargs: {"malformed": True}
        ),
        adapter_context=object(),
        runtime=SimpleNamespace(
            _validate_result=lambda _value: (_ for _ in ()).throw(
                ValueError("hidden schema detail")
            ),
            PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            FAILURE_CODES={},
        ),
    )
    with pytest.raises(runner.CheckFailure) as captured:
        runner.read_and_diagnose_first_primary(  # type: ignore[arg-type]
            context,
            qualification,
            SimpleNamespace(),
            after_diagnostic_call_returned=mark_returned,
        )
    assert returned is True
    assert captured.value.stage == "DIAGNOSTIC_RESULT_VALIDATION"
    assert captured.value.code == "DIAGNOSTIC_RESULT_SCHEMA_MISMATCH"
    assert "hidden" not in str(captured.value)


@pytest.mark.parametrize("target", ("manifest", "semantic", "pool"))
@pytest.mark.parametrize("failure_mode", ("same_size_hash_mismatch", "truncated"))
def test_failed_bound_reads_record_open_and_read_start_before_full_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    failure_mode: str,
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(
        tmp_path, monkeypatch
    )
    first = runner._validate_manifest_first_primary(manifest, qualification)
    paths = {
        "manifest": qualification.manifest_path,
        "semantic": first.semantic_path,
        "pool": first.pool_path,
    }
    path = paths[target]
    original = path.read_bytes()
    if failure_mode == "same_size_hash_mismatch":
        mutated = bytes([original[0] ^ 1]) + original[1:]
        assert len(mutated) == len(original)
    else:
        mutated = original[:-1]
        assert 0 < len(mutated) < len(original)
    path.write_bytes(mutated)

    monkeypatch.setattr(runner, "revalidate_consumption_anchor", lambda _seal: None)
    adapter_called = False

    def diagnose(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal adapter_called
        adapter_called = True
        return {
            "check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            "failure_phase": None,
        }

    context = SimpleNamespace(
        adapter=SimpleNamespace(diagnose_bound_pair=diagnose),
        adapter_context=object(),
        runtime=SimpleNamespace(
            _validate_result=lambda _value: None,
            PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            FAILURE_CODES={},
        ),
    )
    with pytest.raises(ValueError, match="exact byte binding mismatch"):
        runner.read_and_diagnose_first_primary(  # type: ignore[arg-type]
            context, qualification, SimpleNamespace()
        )
    expected_vectors = {
        "manifest": ["TRAIN_CACHE_MANIFEST"],
        "semantic": [
            "TRAIN_CACHE_MANIFEST",
            "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        ],
        "pool": [
            "TRAIN_CACHE_MANIFEST",
            "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
            "TRAIN_FIRST_PRIMARY_V51_POOL",
        ],
    }
    expected_successful = {
        "manifest": [],
        "semantic": ["TRAIN_CACHE_MANIFEST"],
        "pool": [
            "TRAIN_CACHE_MANIFEST",
            "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        ],
    }
    assert qualification.reader.raw_fd_opened_purposes == expected_vectors[target]
    assert qualification.reader.verified_inode_purposes == expected_vectors[target]
    assert qualification.reader.read_started_purposes == expected_vectors[target]
    assert [entry["purpose"] for entry in qualification.reader.entries] == expected_successful[target]
    assert adapter_called is False


def test_verified_open_and_read_start_callbacks_precede_partial_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "input.json"
    path.write_bytes(b'{"value":true}\n')
    monkeypatch.setattr(runner, "ROOT", root)
    original_read = runner.os.read
    read_calls = 0
    events: list[str] = []

    def partial_then_fail(descriptor: int, size_bytes: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        if read_calls == 1:
            return original_read(descriptor, min(size_bytes, 1))
        raise OSError("synthetic partial read failure")

    monkeypatch.setattr(runner.os, "read", partial_then_fail)
    with pytest.raises(OSError, match="partial read failure"):
        runner._read_regular_file_once(
            path,
            role="SYNTHETIC_PARTIAL_INPUT",
            after_raw_fd_open=lambda: events.append("raw_fd_opened"),
            after_verified_inode=lambda _identity: events.append("inode_verified"),
            before_first_read=lambda _identity: events.append("read_started"),
        )
    assert events == ["raw_fd_opened", "inode_verified", "read_started"]
    assert read_calls == 2


def test_raw_fd_open_is_recorded_before_fstat_failure_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "bound.json"
    data = b'{"bound":true}\n'
    path.write_bytes(data)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", root / "cache")
    reader = runner.GuardedReader()
    reader.authorize(
        path,
        purpose="SYNTHETIC_BOUND_INPUT",
        sha256=_sha(data),
        size_bytes=len(data),
    )
    original_fstat = runner.os.fstat
    fstat_calls = 0

    def fail_leaf_fstat(descriptor: int):  # type: ignore[no-untyped-def]
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            raise OSError("synthetic fstat-after-open failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(runner.os, "fstat", fail_leaf_fstat)
    with pytest.raises(OSError, match="fstat-after-open"):
        reader.read_bytes(path, purpose="SYNTHETIC_BOUND_INPUT")
    monkeypatch.setattr(runner.os, "fstat", original_fstat)
    assert reader.raw_fd_opened_purposes == ["SYNTHETIC_BOUND_INPUT"]
    assert reader.verified_inode_purposes == []
    assert reader.read_started_purposes == []
    assert reader.entries == []
    with pytest.raises(PermissionError, match="DUPLICATE_READ_FORBIDDEN"):
        reader.read_bytes(path, purpose="SYNTHETIC_BOUND_INPUT")


@pytest.mark.parametrize(
    ("swap_call", "expected_purposes"),
    (
        (3, ["TRAIN_CACHE_MANIFEST"]),
        (
            4,
            [
                "TRAIN_CACHE_MANIFEST",
                "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
                "TRAIN_FIRST_PRIMARY_V51_POOL",
            ],
        ),
    ),
)
def test_anchor_parent_swap_after_manifest_is_rejected_before_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_call: int,
    expected_purposes: list[str],
) -> None:
    qualification, manifest, _semantic, _pool = _synthetic_manifest_fixture(
        tmp_path, monkeypatch
    )
    manifest_bytes = runner.canonical_bytes(manifest)
    qualification.manifest_path.write_bytes(manifest_bytes)
    object.__setattr__(qualification, "manifest_sha256", _sha(manifest_bytes))
    object.__setattr__(qualification, "manifest_size_bytes", len(manifest_bytes))

    research = runner.ROOT / "research"
    research.mkdir()
    anchor = research / "candidate8-anchor.json"
    anchor_bytes = b'{"consumed":true}\n'
    anchor.write_bytes(anchor_bytes)
    anchor.chmod(0o400)
    anchor_metadata = os.lstat(anchor)
    parent_metadata = os.lstat(research)
    seal = runner.ConsumptionSeal(
        path=anchor,
        data=anchor_bytes,
        sha256=_sha(anchor_bytes),
        size_bytes=len(anchor_bytes),
        file_identity=(int(anchor_metadata.st_dev), int(anchor_metadata.st_ino)),
        parent_identity=(int(parent_metadata.st_dev), int(parent_metadata.st_ino)),
        _sentinel=runner._SEAL_SENTINEL,
    )
    original_revalidate = runner.revalidate_consumption_anchor
    calls = 0
    moved = runner.ROOT / "moved-research"

    def swap_parent_then_revalidate(value: runner.ConsumptionSeal) -> None:
        nonlocal calls
        calls += 1
        if calls == swap_call:
            research.rename(moved)
            research.mkdir()
        original_revalidate(value)

    adapter_called = False

    def diagnose(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal adapter_called
        adapter_called = True
        return {
            "check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            "failure_phase": None,
        }

    context = SimpleNamespace(
        adapter=SimpleNamespace(diagnose_bound_pair=diagnose),
        adapter_context=object(),
        runtime=SimpleNamespace(
            _validate_result=lambda _value: None,
            PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            FAILURE_CODES={},
        ),
    )
    monkeypatch.setattr(
        runner, "revalidate_consumption_anchor", swap_parent_then_revalidate
    )
    with pytest.raises(RuntimeError, match="anchor parent changed"):
        runner.read_and_diagnose_first_primary(  # type: ignore[arg-type]
            context, qualification, seal
        )
    assert adapter_called is False
    assert [entry["purpose"] for entry in qualification.reader.entries] == expected_purposes


def test_public_payload_allows_only_d01_through_d05_or_pass() -> None:
    ledger_sha = "1" * 64
    pairs = {
        "C8_ALL_DIAGNOSTIC_PHASES_PASSED": None,
        "C8_SEMANTIC_VALIDATION_FAILED": "D01_SEMANTIC_VALIDATION",
        "C8_POOL_PARSE_SANITIZE_FAILED": "D02_POOL_PARSE_SANITIZE",
        "C8_OWNERSHIP_EXERCISE_FAILED": "D03_OWNERSHIP_EXERCISE",
        "C8_FIXED_RULE_EXERCISE_FAILED": "D04_FIXED_RULE_EXERCISE",
        "C8_WRAPPER_RESULT_VALIDATION_FAILED": "D05_WRAPPER_RESULT_VALIDATION",
    }
    for code, phase in pairs.items():
        value = runner._public_payload(
            kind=runner.RESULT_KIND,
            status="DIAGNOSTIC_COMPLETE_NONAUTHORIZING",
            check_code=code,
            failure_phase=phase,
            diagnostic_completed=True,
            ledger_sha256=ledger_sha,
        )
        assert value["check_code"] == code
        assert value["final_or_model_rule_selected"] is False
        assert (
            value["diagnostic_candidate_selection_exercise_is_not_model_selection"]
            is True
        )
        assert "candidate_selection_performed" not in value
        assert "case" not in json.dumps(value).lower()
    with pytest.raises(ValueError):
        runner._public_payload(
            kind=runner.RESULT_KIND,
            status="DIAGNOSTIC_COMPLETE_NONAUTHORIZING",
            check_code="C8_LINEAGE_VALIDATION_FAILED",
            failure_phase="LINEAGE",
            diagnostic_completed=True,
            ledger_sha256=ledger_sha,
        )


def test_engineering_terminal_exposes_no_exception_or_source_detail() -> None:
    value = runner._public_payload(
        kind=runner.TERMINAL_KIND,
        status="TERMINAL_ENGINEERING_FAILURE",
        check_code=None,
        failure_phase=None,
        diagnostic_completed=False,
        ledger_sha256="2" * 64,
        engineering_failure_stage="DIAGNOSTIC_EXECUTION",
        engineering_check_code="UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE",
    )
    encoded = json.dumps(value)
    for forbidden in ("exception", "semantic_sha", "pool_sha", "runtime_case", "bend_id"):
        assert forbidden not in encoded


def test_value_free_ledger_rejects_raw_pair_binding_keys() -> None:
    value = {
        "assertions": {
            "authorization_consumption_attempted": True,
            "consumption_anchor_sealed": False,
            "denied_read_attempted": False,
            "diagnostic_call_returned": False,
            "diagnostic_result_accepted_for_publication": False,
            "evaluator_metric_fit_opened_or_started": False,
            "exact_read_inventory_validated": False,
            "full28_opened": False,
            "holdout_opened": False,
            "labels_opened": False,
            "manifest_inode_verified": False,
            "manifest_opened": False,
            "manifest_read_started": False,
            "only_first_primary_pool_inode_verified": False,
            "only_first_primary_pool_opened": False,
            "only_first_primary_pool_read_started": False,
            "only_first_primary_semantic_inode_verified": False,
            "only_first_primary_semantic_opened": False,
            "only_first_primary_semantic_read_started": False,
            "replay_or_later_pair_opened": False,
        },
        "attempt_ordinal": 8,
        "authorization_binding_commitment_sha256": "1" * 64,
        "check_code": None,
        "control_binding_commitment_sha256": "2" * 64,
        "consumption_anchor_sha256": None,
        "diagnostic_failure_phase": None,
        "engineering_attempt_id": runner.ENGINEERING_ATTEMPT_ID,
        "engineering_check_code": "UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE",
        "engineering_failure_stage": "DIAGNOSTIC_EXECUTION",
        "first_primary_binding_commitment_sha256": None,
        "kind": runner.LEDGER_KIND,
        "manifest_binding_commitment_sha256": "3" * 64,
        "read_inventory_commitment_sha256": "5" * 64,
        "raw_fd_opened_inventory_commitment_sha256": "6" * 64,
        "read_started_inventory_commitment_sha256": "7" * 64,
        "schema_version": 1,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "source_binding_commitment_sha256": "4" * 64,
        "status": "TERMINAL_ENGINEERING_FAILURE",
        "verified_inode_inventory_commitment_sha256": "8" * 64,
    }
    runner._assert_value_free_ledger(value)
    value["semantic_sha256"] = "5" * 64
    with pytest.raises(ValueError):
        runner._assert_value_free_ledger(value)


def _anchor_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, runner.AuthorizationSeal, Path]:
    root = tmp_path / "repo"
    research = root / "research"
    output_parent = root / "output"
    custody = root / "custody"
    research.mkdir(parents=True)
    output_parent.mkdir()
    custody.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner,
        "C8_AUTHORIZATION",
        research / "f1_aug30_train18_candidate8_first_primary_diagnostic_authorization_v1.json",
    )
    monkeypatch.setattr(
        runner,
        "C8_QUALIFICATION",
        research / "f1_aug30_train18_candidate8_first_primary_diagnostic_qualification_v1.json",
    )
    monkeypatch.setattr(
        runner,
        "C8_CONSUMPTION_ANCHOR",
        research / "f1_aug30_train18_candidate8_first_primary_diagnostic_consumption_v1.json",
    )
    monkeypatch.setattr(runner, "CACHE_MANIFEST", custody / "manifest.json")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output_parent / "diagnostic")
    monkeypatch.setattr(
        runner, "_validate_authorization_seal", lambda *_args, **_kwargs: None
    )
    context = SimpleNamespace(
        source_bindings={"runner": {"path": "runner", "sha256": "1" * 64, "size_bytes": 1}},
        bootstrap_entries=(),
        _context_nonce=object(),
    )
    authorization = runner.AuthorizationSeal(
        reader=runner.GuardedReader(),
        authorization_sha256="2" * 64,
        authorization_size_bytes=100,
        qualification_sha256="3" * 64,
        qualification_size_bytes=101,
        control_binding_commitment_sha256="4" * 64,
        output_freshness=runner.FreshOutputSeal(
            path=runner.OUTPUT_ROOT,
            parent_identity=(1, 1),
            _sentinel=runner._OUTPUT_SEAL_SENTINEL,
        ),
        source_binding_commitment_sha256=runner._source_binding_commitment(context),
        authority_prefix_commitment_sha256=_sha(b"[]\n"),
        anchor_parent_identity=runner._identity(os.lstat(research)),
        _context_nonce=context._context_nonce,
        _sentinel=runner._SEAL_SENTINEL,
    )
    return context, authorization, research


def test_consumption_anchor_is_no_replace_and_content_revalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, authorization, _research = _anchor_fixture(tmp_path, monkeypatch)
    seal = runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]
    assert seal.path.exists()
    runner.revalidate_consumption_anchor(seal)
    with pytest.raises(PermissionError, match="already consumed"):
        runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]
    seal.path.chmod(0o600)
    seal.path.write_bytes(b"tampered\n")
    seal.path.chmod(0o400)
    with pytest.raises(RuntimeError, match="changed|content"):
        runner.revalidate_consumption_anchor(seal)


def test_partial_anchor_write_permanently_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, authorization, _research = _anchor_fixture(tmp_path, monkeypatch)
    original_write = runner.os.write
    invoked = False

    def crash_after_partial(descriptor: int, data) -> int:  # type: ignore[no-untyped-def]
        nonlocal invoked
        if not invoked:
            invoked = True
            original_write(descriptor, bytes(data[:1]))
            raise OSError("injected crash")
        return original_write(descriptor, data)

    monkeypatch.setattr(runner.os, "write", crash_after_partial)
    with pytest.raises(runner.AnchorCreationFailure):
        runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]
    monkeypatch.setattr(runner.os, "write", original_write)
    assert runner.C8_CONSUMPTION_ANCHOR.exists()
    with pytest.raises(PermissionError, match="already consumed"):
        runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]


def test_anchor_parent_swap_is_detected_before_manifest_and_restored_marker_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, authorization, research = _anchor_fixture(tmp_path, monkeypatch)
    moved = research.parent / "moved-research"
    original_assert = runner._assert_visible_directory_identity
    calls = 0

    def swap_after_write(path: Path, expected, *, role: str):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if "after write" in role:
            research.rename(moved)
            research.mkdir()
        return original_assert(path, expected, role=role)

    monkeypatch.setattr(runner, "_assert_visible_directory_identity", swap_after_write)
    with pytest.raises(runner.AnchorCreationFailure):
        runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]
    assert not runner.C8_CONSUMPTION_ANCHOR.exists()
    assert (moved / runner.C8_CONSUMPTION_ANCHOR.name).exists()
    research.rmdir()
    moved.rename(research)
    monkeypatch.setattr(runner, "_assert_visible_directory_identity", original_assert)
    with pytest.raises(PermissionError, match="already consumed"):
        runner.consume_diagnostic_authorization(context, authorization)  # type: ignore[arg-type]


def _payloads() -> dict[str, bytes]:
    return {
        "diagnostic_read_ledger.json": b'{"ledger":true}\n',
        "diagnostic_result.json": b'{"result":true}\n',
    }


def test_atomic_publication_is_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "output"
    parent.mkdir(parents=True)
    output = parent / "diagnostic"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    freshness = runner.prevalidate_fresh_output(output)
    runner.publish_atomically(output, _payloads(), output_freshness=freshness)
    assert set(path.name for path in output.iterdir()) == set(_payloads())
    with pytest.raises(FileExistsError):
        runner.publish_atomically(output, _payloads(), output_freshness=freshness)


def test_raced_foreign_output_is_untouched_and_never_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "output"
    parent.mkdir(parents=True)
    output = parent / "diagnostic"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    freshness = runner.prevalidate_fresh_output(output)

    def plant_foreign(_staging: Path) -> None:
        output.mkdir()
        (output / "foreign.txt").write_text("FOREIGN", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.publish_atomically(
            output,
            _payloads(),
            output_freshness=freshness,
            after_stage_write=plant_foreign,
        )
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "FOREIGN"
    quarantines = list(parent.glob(f"{runner._quarantine_prefix(output)}*"))
    assert len(quarantines) == 1


def test_postrename_foreign_replacement_is_not_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "output"
    parent.mkdir(parents=True)
    output = parent / "diagnostic"
    moved = parent / "moved-owned"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    freshness = runner.prevalidate_fresh_output(output)

    def replace_published(path: Path) -> None:
        path.rename(moved)
        path.mkdir()
        (path / "foreign.txt").write_text("FOREIGN", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        runner.publish_atomically(
            output,
            _payloads(),
            output_freshness=freshness,
            after_rename=replace_published,
        )
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "FOREIGN"
    assert moved.exists()


def test_output_parent_swap_before_publication_is_rejected_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    parent = root / "output"
    parent.mkdir(parents=True)
    output = parent / "diagnostic"
    moved = root / "moved-output"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT_ROOT", output)
    freshness = runner.prevalidate_fresh_output(output)
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(RuntimeError, match="parent changed"):
        runner.publish_atomically(
            output, _payloads(), output_freshness=freshness
        )
    assert list(parent.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_output_freshness_failure_precedes_authorization_and_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "prevalidate_fresh_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileExistsError("synthetic existing output")
        ),
    )
    monkeypatch.setattr(
        runner,
        "authorize_diagnostic_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization read after freshness failure")
        ),
    )
    with pytest.raises(FileExistsError, match="existing output"):
        runner.run_one_shot(
            SimpleNamespace(),  # type: ignore[arg-type]
            authorization_sha256="1" * 64,
            authorization_size_bytes=1,
        )


def _payload_build_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, runner.AuthorizationSeal, runner.ControlQualification, runner.FirstPrimaryBinding]:
    pair_purposes = [
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
        "TRAIN_FIRST_PRIMARY_V51_POOL",
    ]
    reader = runner.GuardedReader(
        entries=[
            {
                "device_id": 1,
                "inode": index,
                "path": f"synthetic-{index}.json",
                "purpose": purpose,
                "sha256": str(index) * 64,
                "size_bytes": 1,
            }
            for index, purpose in enumerate(
                (
                    "TRAIN_CACHE_MANIFEST",
                    "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
                    "TRAIN_FIRST_PRIMARY_V51_POOL",
                ),
                start=1,
            )
        ],
        raw_fd_opened_purposes=list(pair_purposes),
        verified_inode_purposes=list(pair_purposes),
        read_started_purposes=list(pair_purposes),
    )
    nonce = object()
    reader.first_primary_nonce = nonce
    qualification = runner.ControlQualification(
        reader=reader,
        train_ids=tuple(f"C{index:02d}" for index in range(18)),
        scan_by_case={},
        train_commitment="1" * 64,
        manifest_path=Path("manifest"),
        manifest_sha256="2" * 64,
        manifest_size_bytes=1,
        candidate7_disposition_bytes=b"{}\n",
        c7_semantic_evidence={},
        c7_pool_evidence={},
        control_binding_commitment_sha256="3" * 64,
        _pair_capability_nonce=nonce,
    )
    authorization = runner.AuthorizationSeal(
        reader=reader,
        authorization_sha256="4" * 64,
        authorization_size_bytes=1,
        qualification_sha256="5" * 64,
        qualification_size_bytes=1,
        control_binding_commitment_sha256="3" * 64,
        output_freshness=runner.FreshOutputSeal(
            path=runner.OUTPUT_ROOT,
            parent_identity=(1, 1),
            _sentinel=runner._OUTPUT_SEAL_SENTINEL,
        ),
        source_binding_commitment_sha256=_sha(b"{}\n"),
        authority_prefix_commitment_sha256=_sha(b"[]\n"),
        anchor_parent_identity=(1, 1),
        _context_nonce=object(),
        _sentinel=runner._SEAL_SENTINEL,
    )
    consumption = runner.ConsumptionSeal(
        path=Path("anchor"),
        data=b"{}\n",
        sha256=_sha(b"{}\n"),
        size_bytes=3,
        file_identity=(1, 2),
        parent_identity=(1, 1),
        _sentinel=runner._SEAL_SENTINEL,
    )
    first = runner.FirstPrimaryBinding(
        semantic_path=Path("semantic"),
        semantic_sha256="6" * 64,
        semantic_size_bytes=1,
        pool_path=Path("pool"),
        pool_sha256="7" * 64,
        pool_size_bytes=1,
        commitment_sha256="8" * 64,
        _reader=reader,
        _qualification_nonce=nonce,
        _sentinel=runner._FIRST_PRIMARY_BINDING_SENTINEL,
    )
    context = SimpleNamespace(
        source_bindings={},
        bootstrap_entries=(),
        _context_nonce=authorization._context_nonce,
        runtime=SimpleNamespace(
            PASS_CODE="C8_ALL_DIAGNOSTIC_PHASES_PASSED",
            PHASE_CONTRACT=(
                "D01_SEMANTIC_VALIDATION",
                "D02_POOL_PARSE_SANITIZE",
                "D03_OWNERSHIP_EXERCISE",
                "D04_FIXED_RULE_EXERCISE",
                "D05_WRAPPER_RESULT_VALIDATION",
            ),
            FAILURE_CODES={
                "D01_SEMANTIC_VALIDATION": "C8_SEMANTIC_VALIDATION_FAILED",
                "D02_POOL_PARSE_SANITIZE": "C8_POOL_PARSE_SANITIZE_FAILED",
                "D03_OWNERSHIP_EXERCISE": "C8_OWNERSHIP_EXERCISE_FAILED",
                "D04_FIXED_RULE_EXERCISE": "C8_FIXED_RULE_EXERCISE_FAILED",
                "D05_WRAPPER_RESULT_VALIDATION": "C8_WRAPPER_RESULT_VALIDATION_FAILED",
            },
        ),
    )
    monkeypatch.setattr(runner, "qualify_source_controls", lambda *_args, **_kwargs: qualification)
    monkeypatch.setattr(runner, "consume_diagnostic_authorization", lambda *_args, **_kwargs: consumption)
    monkeypatch.setattr(runner, "revalidate_fresh_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "_validate_authorization_seal", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runner, "_validate_exact_read_inventory", lambda *_args, **_kwargs: None)
    return context, authorization, qualification, first


def test_diagnostic_d03_result_publishes_fixed_result_and_value_free_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, _qualification, first = _payload_build_fixture(monkeypatch)
    monkeypatch.setattr(
        runner,
        "read_and_diagnose_first_primary",
        lambda *_args, **_kwargs: (
            {
                "check_code": "C8_OWNERSHIP_EXERCISE_FAILED",
                "failure_phase": "D03_OWNERSHIP_EXERCISE",
            },
            first,
        ),
    )
    payloads = runner.build_diagnostic_payloads(context, authorization)  # type: ignore[arg-type]
    assert set(payloads) == runner.PASS_FILENAMES
    result = json.loads(payloads["diagnostic_result.json"])
    ledger = json.loads(payloads["diagnostic_read_ledger.json"])
    assert result["check_code"] == "C8_OWNERSHIP_EXERCISE_FAILED"
    assert result["diagnostic_failure_phase"] == "D03_OWNERSHIP_EXERCISE"
    encoded = json.dumps({"result": result, "ledger": ledger})
    for forbidden in ("runtime_case", "semantic_sha256", "pool_sha256", "exception"):
        assert forbidden not in encoded


def test_terminal_ledger_uses_attempted_open_inventories_not_success_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, qualification, _first = _payload_build_fixture(monkeypatch)
    qualification.reader.entries.clear()
    qualification.reader.raw_fd_opened_purposes[:] = [
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
    ]
    qualification.reader.verified_inode_purposes[:] = [
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
    ]
    qualification.reader.read_started_purposes[:] = [
        "TRAIN_CACHE_MANIFEST",
        "TRAIN_FIRST_PRIMARY_V51_SEMANTIC",
    ]
    consumption = runner.ConsumptionSeal(
        path=Path("anchor"),
        data=b"{}\n",
        sha256=_sha(b"{}\n"),
        size_bytes=3,
        file_identity=(1, 2),
        parent_identity=(1, 1),
        _sentinel=runner._SEAL_SENTINEL,
    )
    ledger = runner._ledger_payload(
        context,  # type: ignore[arg-type]
        authorization,
        qualification,
        consumption,
        diagnostic_check_code=None,
        diagnostic_failure_phase=None,
        engineering_failure_stage="DIAGNOSTIC_EXECUTION",
        engineering_check_code="UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE",
        first_primary_commitment_sha256=None,
        exact_read_inventory_validated=False,
        diagnostic_call_returned=False,
        diagnostic_result_accepted_for_publication=False,
    )
    assertions = ledger["assertions"]
    assert assertions["manifest_opened"] is True
    assert assertions["manifest_inode_verified"] is True
    assert assertions["manifest_read_started"] is True
    assert assertions["only_first_primary_semantic_opened"] is True
    assert assertions["only_first_primary_semantic_inode_verified"] is True
    assert assertions["only_first_primary_semantic_read_started"] is True
    assert assertions["only_first_primary_pool_opened"] is False
    assert assertions["only_first_primary_pool_inode_verified"] is False
    assert assertions["only_first_primary_pool_read_started"] is False
    assert ledger["read_inventory_commitment_sha256"] == runner._entry_commitment([])
    assert ledger["raw_fd_opened_inventory_commitment_sha256"] != ledger[
        "read_inventory_commitment_sha256"
    ]


def test_terminal_ledger_retains_raw_open_when_inode_verification_never_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, qualification, _first = _payload_build_fixture(monkeypatch)
    qualification.reader.entries.clear()
    qualification.reader.raw_fd_opened_purposes[:] = ["TRAIN_CACHE_MANIFEST"]
    qualification.reader.verified_inode_purposes.clear()
    qualification.reader.read_started_purposes.clear()
    consumption = runner.ConsumptionSeal(
        path=Path("anchor"),
        data=b"{}\n",
        sha256=_sha(b"{}\n"),
        size_bytes=3,
        file_identity=(1, 2),
        parent_identity=(1, 1),
        _sentinel=runner._SEAL_SENTINEL,
    )
    ledger = runner._ledger_payload(
        context,  # type: ignore[arg-type]
        authorization,
        qualification,
        consumption,
        diagnostic_check_code=None,
        diagnostic_failure_phase=None,
        engineering_failure_stage="DIAGNOSTIC_EXECUTION",
        engineering_check_code="UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE",
        first_primary_commitment_sha256=None,
        exact_read_inventory_validated=False,
        diagnostic_call_returned=False,
        diagnostic_result_accepted_for_publication=False,
    )
    assertions = ledger["assertions"]
    assert assertions["manifest_opened"] is True
    assert assertions["manifest_inode_verified"] is False
    assert assertions["manifest_read_started"] is False
    assert (
        ledger["raw_fd_opened_inventory_commitment_sha256"]
        != ledger["verified_inode_inventory_commitment_sha256"]
    )


def test_post_anchor_unexpected_failure_publishes_terminal_only_without_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, _qualification, _first = _payload_build_fixture(monkeypatch)
    monkeypatch.setattr(
        runner,
        "read_and_diagnose_first_primary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LookupError("SECRET CASE T001 VALUE 0.75")
        ),
    )
    payloads = runner.build_diagnostic_payloads(context, authorization)  # type: ignore[arg-type]
    assert set(payloads) == runner.FAILURE_FILENAMES
    terminal = json.loads(payloads["diagnostic_terminal.json"])
    ledger = json.loads(payloads["diagnostic_read_ledger.json"])
    assert terminal["check_code"] is None
    assert terminal["diagnostic_failure_phase"] is None
    assert terminal["engineering_failure_stage"] == "DIAGNOSTIC_EXECUTION"
    assert (
        terminal["engineering_check_code"]
        == "UNEXPECTED_DIAGNOSTIC_EXECUTION_FAILURE"
    )
    assert ledger["engineering_failure_stage"] == terminal["engineering_failure_stage"]
    assert ledger["engineering_check_code"] == terminal["engineering_check_code"]
    assert ledger["assertions"]["diagnostic_call_returned"] is False
    assert (
        ledger["assertions"]["diagnostic_result_accepted_for_publication"]
        is False
    )
    encoded = b"".join(payloads.values()).decode("utf-8")
    assert "SECRET" not in encoded and "LookupError" not in encoded and "0.75" not in encoded


def test_created_anchor_final_validation_failure_terminalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, _qualification, _first = _payload_build_fixture(monkeypatch)
    monkeypatch.setattr(
        runner,
        "consume_diagnostic_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.AnchorCreationFailure("opaque")
        ),
    )
    payloads = runner.build_diagnostic_payloads(context, authorization)  # type: ignore[arg-type]
    assert set(payloads) == runner.FAILURE_FILENAMES
    ledger = json.loads(payloads["diagnostic_read_ledger.json"])
    assert ledger["consumption_anchor_sha256"] is None
    assert ledger["assertions"]["consumption_anchor_sealed"] is False
    assert ledger["engineering_failure_stage"] == "ANCHOR_CONSUMPTION"
    assert (
        ledger["engineering_check_code"]
        == "ANCHOR_CREATION_FAILED_AFTER_CONSUMPTION"
    )


def test_lineage_code_is_terminalized_not_published_as_diagnostic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, _qualification, first = _payload_build_fixture(monkeypatch)
    monkeypatch.setattr(
        runner,
        "read_and_diagnose_first_primary",
        lambda *_args, **_kwargs: (
            {"check_code": "C8_LINEAGE_VALIDATION_FAILED", "failure_phase": "LINEAGE"},
            first,
        ),
    )
    payloads = runner.build_diagnostic_payloads(context, authorization)  # type: ignore[arg-type]
    assert set(payloads) == runner.FAILURE_FILENAMES
    assert b"LINEAGE" not in payloads["diagnostic_terminal.json"]
    terminal = json.loads(payloads["diagnostic_terminal.json"])
    ledger = json.loads(payloads["diagnostic_read_ledger.json"])
    assert terminal["check_code"] is None
    assert terminal["diagnostic_failure_phase"] is None
    assert terminal["engineering_failure_stage"] == "DIAGNOSTIC_RESULT_VALIDATION"
    assert (
        terminal["engineering_check_code"]
        == "DIAGNOSTIC_RESULT_OUTSIDE_D01_D05_PASS"
    )
    assert ledger["assertions"]["diagnostic_call_returned"] is True
    assert (
        ledger["assertions"]["diagnostic_result_accepted_for_publication"]
        is False
    )


def test_mismatched_allowed_code_and_phase_pair_terminalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, _qualification, first = _payload_build_fixture(monkeypatch)
    monkeypatch.setattr(
        runner,
        "read_and_diagnose_first_primary",
        lambda *_args, **_kwargs: (
            {
                "check_code": "C8_SEMANTIC_VALIDATION_FAILED",
                "failure_phase": "D04_FIXED_RULE_EXERCISE",
            },
            first,
        ),
    )
    payloads = runner.build_diagnostic_payloads(context, authorization)  # type: ignore[arg-type]
    assert set(payloads) == runner.FAILURE_FILENAMES
    terminal = json.loads(payloads["diagnostic_terminal.json"])
    assert terminal["check_code"] is None
    assert terminal["diagnostic_failure_phase"] is None
    assert terminal["engineering_failure_stage"] == "DIAGNOSTIC_RESULT_VALIDATION"
    assert (
        terminal["engineering_check_code"]
        == "DIAGNOSTIC_RESULT_OUTSIDE_D01_D05_PASS"
    )
