from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types

import pytest

from scripts import run_f1_train18_candidate11_full_replay_diagnostic_once_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py"
TEST_SOURCE = ROOT / "tests/test_run_f1_train18_candidate11_full_replay_diagnostic_once_v1.py"
ZERO_SHA = "0" * 64


def _base_cli() -> list[str]:
    return [
        "--runner-sha256",
        ZERO_SHA,
        "--runner-size-bytes",
        "1",
        "--runner-tests-sha256",
        "1" * 64,
        "--runner-tests-size-bytes",
        "1",
    ]


def _patch_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    root = tmp_path / "repo"
    research = root / "research"
    output_parent = root / "output/research"
    allowed = root / "output/research/f1-v51-today-scan-only-development-20260830-r2"
    manifest = root / "output/research/cache/manifest.json"
    for path in (research, output_parent, allowed, manifest.parent):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", allowed)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", manifest)
    monkeypatch.setattr(
        runner,
        "OUTPUT_ROOT",
        output_parent
        / "f1-aug30-train18-candidate11-full-replay-diagnostic-v1",
    )
    monkeypatch.setattr(
        runner,
        "C11_ANCHOR",
        research
        / "f1_aug30_train18_candidate11_full_replay_diagnostic_consumption_v1.json",
    )
    monkeypatch.setattr(runner, "C11_AUTHORIZATION", research / "authorization.json")
    monkeypatch.setattr(runner, "C11_QUALIFICATION", research / "qualification.json")
    return {
        "root": root,
        "research": research,
        "output_parent": output_parent,
        "allowed": allowed,
        "manifest": manifest,
    }


def _patch_static_paths(
    monkeypatch: pytest.MonkeyPatch, tree: dict[str, Path]
) -> None:
    fixed = {
        name: (tree["research"] / f"control-{index}.json", sha, size)
        for index, (name, (_path, sha, size)) in enumerate(
            runner.FIXED_CONTROLS.items()
        )
    }
    monkeypatch.setattr(runner, "FIXED_CONTROLS", fixed)
    monkeypatch.setattr(
        runner,
        "CUSTODY_EVIDENCE_PATHS",
        {
            name: tree["research"] / f"custody-{name}.json"
            for name in (
                "access_evidence",
                "builder",
                "staged_validator_pass",
                "validator",
            )
        },
    )


def _qualification(context: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        manifest_path=runner.CACHE_MANIFEST,
        manifest_sha256=runner.CACHE_MANIFEST_SHA256,
        manifest_size_bytes=runner.CACHE_MANIFEST_SIZE_BYTES,
        control_binding_commitment_sha256="2" * 64,
        control_raw_fd_opened_commitment_sha256="3" * 64,
        control_read_inventory_commitment_sha256="4" * 64,
        control_read_started_commitment_sha256="5" * 64,
        control_verified_inode_commitment_sha256="6" * 64,
        _context_token=getattr(context, "_issuance_token", object()),
    )


def _core_result(*, compatible: bool = True) -> dict[str, object]:
    passed = "111111111111" if compatible else "111101111111"
    return {
        "aggregate_reached_passed_bitsets_commitment_sha256": "1" * 64,
        "all_records_compatible": compatible,
        "check_code": (
            "C11_ALL_RECORD_DIAGNOSTICS_REPLAY_COMPATIBLE"
            if compatible
            else "C11_AGGREGATE_DIAGNOSTIC_INCOMPATIBLE"
        ),
        "input_receipt_vector_commitment_sha256": "2" * 64,
        "primary_replay_identity": True,
        "source_lineage_commitment_sha256": "3" * 64,
        "stage_contract_sha256": "4" * 64,
        "stage_passed_all_bitset": passed,
        "stage_reached_all_bitset": "111111111111",
        "status": (
            "FULL_SOURCE_REPLAY_COMPATIBLE_NONAUTHORIZING"
            if compatible
            else "SOURCE_REPLAY_INCOMPATIBLE_NONAUTHORIZING"
        ),
    }


def _outcome(*, compatible: bool = True) -> runner.ExecutionOutcome:
    core = _core_result(compatible=compatible)
    return runner.ExecutionOutcome(
        core_result=core,
        core_result_bytes=runner.canonical_bytes(core),
        manifest_binding_sha256="5" * 64,
        artifact_inventory_commitment_sha256="6" * 64,
    )


def _publication_fixture(
    *, compatible: bool = True
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace, runner.ExecutionOutcome]:
    entries = [
        {
            "device_id": 1,
            "inode": index + 1,
            "path": f"internal/{index}.json",
            "purpose": f"TRAIN_RECORD_{index:02d}_FILE",
            "sha256": f"{index + 1:064x}",
            "size_bytes": index + 1,
        }
        for index in range(72)
    ]
    purposes = [str(entry["purpose"]) for entry in entries]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=purposes,
        verified_inode_purposes=purposes,
        read_started_purposes=purposes,
    )
    reader.artifact_inventory = SimpleNamespace(
        bindings={index: index for index in range(72)}
    )
    context = SimpleNamespace(source_commitment_sha256="7" * 64)
    authorization = SimpleNamespace(
        authorization_sha256="8" * 64,
        qualification_sha256="9" * 64,
    )
    qualification = SimpleNamespace(
        reader=reader,
        control_binding_commitment_sha256="a" * 64,
    )
    consumption = SimpleNamespace(sha256="b" * 64)
    return context, authorization, qualification, consumption, _outcome(
        compatible=compatible
    )


def _publication_payloads() -> dict[str, bytes]:
    return {
        "full_replay_diagnostic_lock.json": b'{"lock":true}\n',
        "full_replay_diagnostic_read_ledger.json": b'{"ledger":true}\n',
        "full_replay_diagnostic_result.json": b'{"result":true}\n',
    }


def test_top_level_imports_are_stdlib_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "errno",
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


def test_direct_bootstrap_rejects_unverified_executing_module() -> None:
    with pytest.raises(PermissionError, match="verified private bytes"):
        runner.bootstrap_verified_sources(
            runner_sha256="0" * 64,
            runner_size_bytes=1,
            runner_tests_sha256="1" * 64,
            runner_tests_size_bytes=1,
        )


def test_cli_private_exec_ignores_preimport_runner_monkeypatch() -> None:
    runner_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    arguments = [
        "--runner-sha256",
        runner.sha256_bytes(runner_bytes),
        "--runner-size-bytes",
        str(len(runner_bytes)),
        "--runner-tests-sha256",
        runner.sha256_bytes(test_bytes),
        "--runner-tests-size-bytes",
        str(len(test_bytes)),
        "--source-check-only",
    ]
    program = (
        "import json\n"
        "from scripts import run_f1_train18_candidate11_full_replay_diagnostic_once_v1 as r\n"
        "r.source_check_result = lambda _context: "
        "{'status':'UNAUTHENTICATED_EXECUTING_RUNNER'}\n"
        f"raise SystemExit(r.main({arguments!r}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["status"] == "SOURCE_BYTES_VERIFIED_ONLY"
    assert value["check_code"] == "C11_EXACT_SOURCE_BYTES_VERIFIED"
    assert "UNAUTHENTICATED_EXECUTING_RUNNER" not in completed.stdout


def test_preload_scan_never_calls_alias_getattr_or_accepts_token_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = object()
    outer = types.ModuleType("_candidate11_synthetic_outer")
    outer._C11_OUTER_LAUNCH_TOKEN = token
    alias = types.ModuleType("_candidate11_hostile_alias")
    alias.__file__ = str(runner.C11_CORE_SOURCE)
    side_effects: list[str] = []

    def hostile_getattr(name: str) -> object:
        side_effects.append(name)
        if name == "_C11_OUTER_LAUNCH_TOKEN":
            return token
        raise AttributeError(name)

    alias.__getattr__ = hostile_getattr
    monkeypatch.setattr(runner, "_C11_VERIFIED_RUNNER_LAUNCH_TOKEN", token)
    monkeypatch.setattr(runner, "_C11_VERIFIED_RUNNER_OUTER_MODULE", outer)
    for name in runner.FORBIDDEN_PRELOADED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="Candidate11 rejects"):
        runner._reject_preloaded([runner.C11_CORE_SOURCE])
    assert side_effects == []


def test_preload_scan_rejects_moduletype_subclass_alias_without_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileModule(types.ModuleType):
        pass

    alias = HostileModule("_candidate11_hostile_module_subclass")
    alias.__file__ = str(runner.C11_CORE_SOURCE)
    for name in runner.FORBIDDEN_PRELOADED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="preloaded module subclass"):
        runner._reject_preloaded([runner.C11_CORE_SOURCE])


def test_private_dispatch_rejects_outer_moduletype_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OuterSubclass(types.ModuleType):
        pass

    runner_bytes = SOURCE.read_bytes()
    test_bytes = TEST_SOURCE.read_bytes()
    arguments = [
        "--runner-sha256",
        runner.sha256_bytes(runner_bytes),
        "--runner-size-bytes",
        str(len(runner_bytes)),
        "--runner-tests-sha256",
        runner.sha256_bytes(test_bytes),
        "--runner-tests-size-bytes",
        str(len(test_bytes)),
        "--source-check-only",
    ]
    monkeypatch.setitem(
        sys.modules,
        runner.__name__,
        OuterSubclass(runner.__name__),
    )
    with pytest.raises(RuntimeError, match="outer runner module"):
        runner._dispatch_through_verified_runner(arguments)


def test_exact_lineage_bindings_and_fresh_namespaces() -> None:
    assert runner.ATTEMPT_ORDINAL == 11
    assert runner.FIXED_SOURCE_BINDINGS["c11_core"][1:] == (
        "42c5ded051dd3b57dd48e3039a433afe63a0467edc13d4bd3b65620a6a279359",
        32967,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c11_core_tests"][1:] == (
        "1419579dd40ad01f73e21bc9ca8defb0523ff104e2eb43ddc81fe29a324c420d",
        26644,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c10_runner"][1:] == (
        "dc47d8612b9dbdc01fe7d5432cae76b69b2e696061e488125e2c27bce77db8b5",
        164874,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c10_runner_tests"][1:] == (
        "c07c4192316ce8452310cb83b54119b0f20046fc7a6cebb5f985b7db5c53daa9",
        69723,
    )
    assert len(runner.FIXED_SOURCE_BINDINGS) == 18
    assert runner.FIXED_CONTROLS["CANDIDATE10_TERMINAL_DISPOSITION"][1:] == (
        "eb6240e3b28149446f2a971e522c9ce73d880facf28a4496e528629cff03818c",
        6234,
    )
    for path in (
        runner.C11_AUTHORIZATION,
        runner.C11_QUALIFICATION,
        runner.C11_ANCHOR,
        runner.OUTPUT_ROOT,
    ):
        assert "candidate11" in str(path).lower()


def test_cli_has_exactly_three_explicit_modes_and_no_default() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(_base_cli())
    assert runner._parse_args(
        [*_base_cli(), "--source-check-only"]
    ).source_check_only
    assert runner._parse_args(
        [*_base_cli(), "--source-control-check-only"]
    ).source_control_check_only
    parsed = runner._parse_args(
        [
            *_base_cli(),
            "--diagnostic-once",
            "--authorization-sha256",
            "2" * 64,
            "--authorization-size-bytes",
            "10",
        ]
    )
    assert parsed.diagnostic_once
    with pytest.raises(SystemExit):
        runner._parse_args([*_base_cli(), "--diagnostic-once"])
    with pytest.raises(SystemExit):
        runner._parse_args(
            [*_base_cli(), "--source-check-only", "--authorization-sha256", "2" * 64]
        )


def test_c8_adapter_context_is_constructed_from_exact_six_bindings() -> None:
    source = inspect.getsource(runner.bootstrap_verified_sources)
    for name in (
        '"adapter": dict(bindings["c8_adapter"])',
        '"adapter_tests": dict(bindings["c8_adapter_tests"])',
        '"candidate7_runtime": dict(bindings["c7_runtime"])',
        '"hypothesis": dict(bindings["c6_hypothesis"])',
        '"runtime": dict(bindings["c8_runtime"])',
        '"runtime_tests": dict(bindings["c8_runtime_tests"])',
    ):
        assert name in source
    binding_block = source.split("source_bindings={", 1)[1].split("},", 1)[0]
    assert binding_block.count(": dict(bindings[") == 6


def test_core_source_projection_is_exact_c10_c9_c8_c7_c6_lineage() -> None:
    expected = {
        "builder",
        "c10_core",
        "c10_core_tests",
        "c10_runner",
        "c10_runner_tests",
        "c6_hypothesis",
        "c7_runtime",
        "c8_adapter",
        "c8_runtime",
        "c9_adapter",
        "c9_adapter_tests",
        "cluster",
    }
    context = SimpleNamespace(
        source_bindings={
            name: {"sha256": f"{index + 1:064x}"}
            for index, name in enumerate(sorted(expected))
        }
    )
    assert set(runner._core_source_bindings(context)) == expected


def test_candidate9_and_candidate10_dispositions_are_exact() -> None:
    c9 = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate9_terminal_disposition_v1",
        "attempt_ordinal": 9,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "status": "PASS_ENGINEERING_DIAGNOSTIC_NONAUTHORIZING",
        "check_code": "C8_ALL_DIAGNOSTIC_PHASES_PASSED",
        "candidate10_engineering_successor_authorized": True,
    }
    runner._validate_candidate9_disposition(c9)
    c10 = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_aug30_train18_candidate10_terminal_disposition_v1",
        "attempt_ordinal": 10,
        "scientific_hypothesis_id": runner.SCIENTIFIC_HYPOTHESIS_ID,
        "status": "REJECTED_ENGINEERING_CORE_PREFLIGHT",
        "check_code": "C10_CORE_PREFLIGHT_FAILED",
        "candidate11_diagnostic_engineering_authorized": True,
        "candidate11_fit_metric_or_promotion_authorized": False,
        "label_or_gold_opened": False,
        "holdout_opened": False,
        "metric_or_evaluator_started": False,
        "model_or_fit_started": False,
        "scientific_hypothesis_performance_tested": False,
    }
    runner._validate_candidate10_disposition(c10)
    for key in (
        "candidate11_diagnostic_engineering_authorized",
        "label_or_gold_opened",
        "scientific_hypothesis_performance_tested",
    ):
        drifted = dict(c10)
        drifted[key] = not drifted[key]
        with pytest.raises(runner.RunnerFailure):
            runner._validate_candidate10_disposition(drifted)


def test_qualification_and_authorization_are_count_path_ordinal_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        source_commitment_sha256="1" * 64, _issuance_token=object()
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    qualification = _qualification(context)
    qualification_value = runner._decode(
        runner.build_qualification_record(context, qualification), name="qualification"
    )
    runner._qualification_static_contract(
        context, qualification_value, control_commitment="2" * 64
    )
    authorization = runner.build_authorization_contract(
        context,
        qualification_record_sha256=runner.sha256_bytes(
            runner.canonical_bytes(qualification_value)
        ),
        control_binding_commitment_sha256="2" * 64,
    )
    for value in (qualification_value, authorization):
        encoded = json.dumps(value, sort_keys=True)
        for forbidden in (
            "count",
            '"path"',
            "ordinal",
            "size_bytes",
            "source_bindings",
            '"namespace"',
        ):
            assert forbidden not in encoded
        runner._assert_aggregate_safe(value)


def test_qualification_contract_rejects_added_count_or_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        source_commitment_sha256="1" * 64, _issuance_token=object()
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    qualification = _qualification(context)
    value = runner._decode(
        runner.build_qualification_record(context, qualification), name="qualification"
    )
    for key, child in (("record_count", 36), ("path", "private")):
        drifted = dict(value)
        drifted[key] = child
        with pytest.raises(PermissionError):
            runner._qualification_static_contract(
                context, drifted, control_commitment="2" * 64
            )


def test_external_authority_reads_only_auth_and_qualification_synthetically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    _patch_static_paths(monkeypatch, tree)
    context = SimpleNamespace(
        source_entries=(),
        source_commitment_sha256="1" * 64,
        _issuance_token=object(),
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    qualification = _qualification(context)
    qualification_bytes = runner.build_qualification_record(context, qualification)
    runner.C11_QUALIFICATION.write_bytes(qualification_bytes)
    authorization_value = runner.build_authorization_contract(
        context,
        qualification_record_sha256=runner.sha256_bytes(qualification_bytes),
        control_binding_commitment_sha256="2" * 64,
    )
    authorization_bytes = runner.canonical_bytes(authorization_value)
    runner.C11_AUTHORIZATION.write_bytes(authorization_bytes)
    freshness = runner.prevalidate_fresh_output()
    seal = runner.authorize_diagnostic_once(
        context,
        authorization_sha256=runner.sha256_bytes(authorization_bytes),
        authorization_size_bytes=len(authorization_bytes),
        output_freshness=freshness,
    )
    assert [entry["purpose"] for entry in seal.reader.entries] == [
        "CANDIDATE11_EXTERNAL_AUTHORIZATION",
        "CANDIDATE11_QUALIFICATION_RECORD",
    ]
    assert seal.reader.raw_fd_opened_purposes == [
        "CANDIDATE11_EXTERNAL_AUTHORIZATION",
        "CANDIDATE11_QUALIFICATION_RECORD",
    ]
    assert not runner.C11_ANCHOR.exists()
    assert not runner.CACHE_MANIFEST.exists()


def test_consumption_anchor_is_count_path_ordinal_free_and_o_excl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    context = SimpleNamespace(
        source_commitment_sha256="1" * 64,
        _issuance_token=object(),
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(
        runner, "_validate_preanchor_reader_state", lambda *_args: None
    )
    monkeypatch.setattr(runner, "revalidate_anchor", lambda *_args: None)
    reader = runner.GuardedReader()
    qualification = _qualification(context)
    qualification.reader = reader
    qualification_bytes = runner.build_qualification_record(context, qualification)
    parent_fd, parent_identity = runner._open_stable_directory(
        tree["research"], role="synthetic anchor parent"
    )
    os.close(parent_fd)
    authorization = SimpleNamespace(
        reader=reader,
        authorization_sha256="7" * 64,
        authorization_size_bytes=1,
        qualification_sha256=runner.sha256_bytes(qualification_bytes),
        qualification_size_bytes=len(qualification_bytes),
        qualification_bytes=qualification_bytes,
        control_binding_commitment_sha256="2" * 64,
        anchor_parent_identity=parent_identity,
    )
    seal = runner.consume_authorization(context, authorization, qualification)
    value = runner._decode(seal.bytes_value, name="anchor")
    runner._assert_aggregate_safe(value)
    encoded = json.dumps(value, sort_keys=True)
    for forbidden in ("count", '"path"', "ordinal", "size_bytes", '"namespace"'):
        assert forbidden not in encoded
    with pytest.raises(PermissionError):
        runner.consume_authorization(context, authorization, qualification)


def test_preanchor_reader_rejects_manifest_or_inventory_contamination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    purposes = [
        "CANDIDATE11_EXTERNAL_AUTHORIZATION",
        "CANDIDATE11_QUALIFICATION_RECORD",
        *runner._expected_control_purposes(),
    ]
    entries = [
        {
            "device_id": 1,
            "inode": index + 1,
            "path": f"synthetic/{index}.json",
            "purpose": purpose,
            "sha256": f"{index + 1:064x}",
            "size_bytes": index + 1,
        }
        for index, purpose in enumerate(purposes)
    ]
    reader = runner.GuardedReader(
        entries=entries,
        raw_fd_opened_purposes=list(purposes),
        verified_inode_purposes=list(purposes),
        read_started_purposes=list(purposes),
    )
    consumed = {
        runner._lexical(runner.ROOT / str(entry["path"]), role=str(entry["purpose"]))
        for entry in entries
    }
    reader.allowed = {path: {} for path in consumed}
    reader.consumed_paths = set(consumed)
    context = SimpleNamespace(source_entries=())
    authorization = SimpleNamespace(reader=reader)
    qualification = SimpleNamespace(reader=reader)
    runner._validate_preanchor_reader_state(context, authorization, qualification)
    reader.artifact_inventory = object()
    with pytest.raises(PermissionError):
        runner._validate_preanchor_reader_state(
            context, authorization, qualification
        )


def _manifest_fixture(
    tree: dict[str, Path], *, write_files: bool
) -> tuple[dict[str, object], SimpleNamespace]:
    train_ids = tuple(f"case-{index:02d}" for index in range(18))
    scan_by_case = {
        case_id: runner.sha256_bytes(f"scan-{case_id}".encode())
        for case_id in train_ids
    }
    train_commitment = runner.sha256_bytes(
        runner.canonical_bytes(
            [
                {"case_id": case_id, "scan_sha256": scan_by_case[case_id]}
                for case_id in train_ids
            ]
        )
    )
    rows: list[dict[str, object]] = []
    for pair, case_id in enumerate(train_ids):
        runtime_id = f"T{pair:03d}"
        semantic = runner.canonical_bytes({"semantic": pair})
        pool = runner.canonical_bytes({"pool": pair})
        for pass_name in ("primary", "replay"):
            semantic_path = (
                tree["allowed"] / pass_name / runtime_id / "semantic_result.json"
            )
            pool_path = (
                tree["allowed"]
                / pass_name
                / runtime_id
                / "pool"
                / f"{runtime_id}.json"
            )
            if write_files:
                semantic_path.parent.mkdir(parents=True, exist_ok=True)
                pool_path.parent.mkdir(parents=True, exist_ok=True)
                semantic_path.write_bytes(semantic)
                pool_path.write_bytes(pool)
            rows.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(pool_path.relative_to(tree["root"])),
                    "pool_sha256": runner.sha256_bytes(pool),
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case_id],
                    "semantic_path": str(semantic_path.relative_to(tree["root"])),
                    "semantic_sha256": runner.sha256_bytes(semantic),
                }
            )
    qualification = SimpleNamespace(
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment_sha256=train_commitment,
    )
    return (
        {
            "case_count": 18,
            "holdout_entry_count": 0,
            "kind": runner.CACHE_MANIFEST_KIND,
            "record_count": 36,
            "records": rows,
            "schema_version": 1,
            "source_only": True,
            "split_sha256": runner.FIXED_CONTROLS["SEALED_SPLIT"][1],
            "train_allowlist_sha256": runner.FIXED_CONTROLS["TRAIN_ALLOWLIST"][1],
            "train_set_commitment_sha256": train_commitment,
            "v51_scan_only_manifest_sha256": runner.V51_SCAN_ONLY_MANIFEST_SHA256,
        },
        qualification,
    )


def test_manifest_requires_exact_36_rows_and_72_unique_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    manifest, qualification = _manifest_fixture(tree, write_files=False)
    records = runner._validate_full_manifest(
        manifest, qualification, manifest_binding_sha256="2" * 64
    )
    assert len(records) == 36
    assert len({record.manifest_record_binding_sha256 for record in records}) == 36
    assert len(
        {record.semantic_path for record in records}
        | {record.pool_path for record in records}
    ) == 72


@pytest.mark.parametrize("mutation", ["short", "reorder", "duplicate", "replay"])
def test_manifest_adversarial_drift_fails_closed(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    manifest, qualification = _manifest_fixture(tree, write_files=False)
    rows = manifest["records"]
    assert isinstance(rows, list)
    if mutation == "short":
        rows.pop()
    elif mutation == "reorder":
        rows[0], rows[2] = rows[2], rows[0]
    elif mutation == "duplicate":
        rows[2]["semantic_path"] = rows[0]["semantic_path"]
    else:
        rows[1]["pool_sha256"] = "3" * 64
    with pytest.raises(runner.RunnerFailure):
        runner._validate_full_manifest(
            manifest, qualification, manifest_binding_sha256="2" * 64
        )


class FakeCore:
    class RecordBytes(SimpleNamespace):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)

    def __init__(self) -> None:
        self.records: tuple[SimpleNamespace, ...] = ()
        self.context_kwargs: dict[str, object] = {}

    def make_verified_context(self, **kwargs: object) -> object:
        self.context_kwargs = kwargs
        return object()

    def diagnose_all_records(
        self,
        _context: object,
        records: tuple[SimpleNamespace, ...],
        *,
        manifest_binding_sha256: str,
    ) -> dict[str, object]:
        self.records = records
        assert len(records) == 36
        assert all(
            records[index].semantic_bytes == records[index + 1].semantic_bytes
            for index in range(0, 36, 2)
        )
        result = _core_result()
        result["manifest_binding_sha256"] = manifest_binding_sha256
        return result

    @staticmethod
    def _validate_public_result(_value: object) -> None:
        return None

    canonical_bytes = staticmethod(runner.canonical_bytes)


def test_synthetic_full_inventory_reads_all_bytes_before_one_core_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    _patch_static_paths(monkeypatch, tree)
    manifest, base = _manifest_fixture(tree, write_files=True)
    manifest_bytes = runner.canonical_bytes(manifest)
    tree["manifest"].write_bytes(manifest_bytes)
    monkeypatch.setattr(
        runner, "CACHE_MANIFEST_SHA256", runner.sha256_bytes(manifest_bytes)
    )
    monkeypatch.setattr(runner, "CACHE_MANIFEST_SIZE_BYTES", len(manifest_bytes))
    reader = runner.GuardedReader()
    qualification = SimpleNamespace(
        **vars(base),
        reader=reader,
        manifest_path=tree["manifest"],
        manifest_sha256=runner.sha256_bytes(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
        candidate7_disposition_bytes=b"synthetic-c7",
        candidate10_disposition_bytes=b"synthetic-c10",
    )
    fake_core = FakeCore()
    names = {
        "builder",
        "c10_core",
        "c10_core_tests",
        "c10_runner",
        "c10_runner_tests",
        "c6_hypothesis",
        "c7_runtime",
        "c8_adapter",
        "c8_runtime",
        "c9_adapter",
        "c9_adapter_tests",
        "cluster",
    }
    context = SimpleNamespace(
        c11_core=fake_core,
        c9_adapter=object(),
        c9_adapter_context=object(),
        c8_adapter=object(),
        c8_adapter_context=object(),
        c8_runtime=object(),
        source_bindings={
            name: {"sha256": f"{index + 1:064x}"}
            for index, name in enumerate(sorted(names))
        },
        source_entries=(),
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    complete_inventory_validated = False

    def validate_complete_inventory(*_args: object) -> None:
        nonlocal complete_inventory_validated
        assert len(reader.read_started_purposes) == 73
        assert reader.read_started_purposes[-1].endswith("_POOL")
        complete_inventory_validated = True

    monkeypatch.setattr(
        runner, "_validate_complete_inventory", validate_complete_inventory
    )
    record_creation_read_lengths: list[int] = []

    def record_factory(**kwargs: object) -> SimpleNamespace:
        assert complete_inventory_validated
        record_creation_read_lengths.append(len(reader.read_started_purposes))
        return SimpleNamespace(**kwargs)

    fake_core.RecordBytes = record_factory
    checks: list[int] = []
    monkeypatch.setattr(
        runner, "revalidate_anchor", lambda *_args: checks.append(len(checks))
    )
    outcome = runner.read_adapt_diagnose_full_inventory(
        context, qualification, SimpleNamespace()
    )
    assert len(fake_core.records) == 36
    assert len(
        {record.semantic_file_binding_sha256 for record in fake_core.records}
        | {record.pool_file_binding_sha256 for record in fake_core.records}
    ) == 72
    assert len(reader.read_started_purposes) == 73
    assert reader.read_started_purposes[-1].endswith("_POOL")
    assert record_creation_read_lengths == [73] * 36
    assert fake_core.context_kwargs["candidate7_disposition_bytes"] == b"synthetic-c7"
    assert fake_core.context_kwargs["candidate10_disposition_bytes"] == b"synthetic-c10"
    assert outcome.core_result["all_records_compatible"] is True
    assert checks


def test_guarded_reader_raw_inode_read_start_and_duplicate_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    _patch_static_paths(monkeypatch, tree)
    data = b'{"authorized":true}\n'
    runner.C11_QUALIFICATION.write_bytes(data)
    reader = runner.GuardedReader()
    reader.authorize(
        runner.C11_QUALIFICATION,
        purpose="CANDIDATE11_QUALIFICATION_RECORD",
        sha256=runner.sha256_bytes(data),
        size_bytes=len(data),
    )
    assert reader.read_bytes(
        runner.C11_QUALIFICATION, purpose="CANDIDATE11_QUALIFICATION_RECORD"
    ) == data
    assert reader.raw_fd_opened_purposes == ["CANDIDATE11_QUALIFICATION_RECORD"]
    assert reader.verified_inode_purposes == ["CANDIDATE11_QUALIFICATION_RECORD"]
    assert reader.read_started_purposes == ["CANDIDATE11_QUALIFICATION_RECORD"]
    with pytest.raises(PermissionError):
        reader.read_bytes(
            runner.C11_QUALIFICATION, purpose="CANDIDATE11_QUALIFICATION_RECORD"
        )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_regular_file_reader_rejects_aliases(
    kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tree = _patch_tree(monkeypatch, tmp_path)
    original = tree["research"] / "original.json"
    alias = tree["research"] / "alias.json"
    original.write_bytes(b"{}\n")
    if kind == "symlink":
        alias.symlink_to(original)
    else:
        os.link(original, alias)
    with pytest.raises(ValueError):
        runner._read_regular_file_once(alias, role="synthetic alias")


def test_public_result_preserves_compatible_and_incompatible_as_completed() -> None:
    for compatible in (True, False):
        context, _authorization, _qualification_value, consumption, outcome = (
            _publication_fixture(compatible=compatible)
        )
        ledger_bytes = b'{"ledger":true}\n'
        result = runner._c11_result_value(
            context, consumption, outcome, ledger_bytes=ledger_bytes
        )
        runner._validate_c11_result(result)
        assert result["diagnostic_complete"] is True
        assert result["all_records_compatible"] is compatible
        assert len(result["stage_reached_all_bitset"]) == 12
        assert len(result["stage_passed_all_bitset"]) == 12


def test_public_result_ledger_lock_are_count_path_ordinal_value_free() -> None:
    context, authorization, qualification, consumption, outcome = _publication_fixture()
    ledger = runner._c11_ledger_value(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=outcome,
        failure_stage="CORE_DIAGNOSTIC_INVOCATION",
        anchor_durable_entry_created_or_partial=False,
    )
    ledger_bytes = runner.canonical_bytes(ledger)
    result = runner._c11_result_value(
        context, consumption, outcome, ledger_bytes=ledger_bytes
    )
    result_bytes = runner.canonical_bytes(result)
    lock = runner._c11_lock_value(
        context,
        authorization,
        qualification,
        consumption,
        outcome,
        ledger_bytes=ledger_bytes,
        result_bytes=result_bytes,
    )
    for value in (ledger, result, lock):
        runner._assert_aggregate_safe(value)
        encoded = json.dumps(value, sort_keys=True)
        for forbidden in (
            "count",
            '"path"',
            "ordinal",
            "case_id",
            "runtime_case",
            "size_bytes",
            "exception",
            "semantic_bytes",
            "pool_bytes",
        ):
            assert forbidden not in encoded


@pytest.mark.parametrize(
    ("validator", "factory"),
    [
        (
            runner._validate_c11_result,
            lambda: runner._c11_result_value(
                _publication_fixture()[0],
                _publication_fixture()[3],
                _publication_fixture()[4],
                ledger_bytes=b'{"ledger":true}\n',
            ),
        ),
        (
            runner._validate_c11_lock,
            lambda: _valid_lock(),
        ),
    ],
)
def test_public_validators_reject_extra_or_malformed_fields(
    validator: object, factory: object
) -> None:
    value = factory()
    drifted = dict(value)
    drifted["record_count"] = 36
    with pytest.raises(ValueError):
        validator(drifted)
    drifted = dict(value)
    drifted["stage_passed_all_bitset"] = "1" * 11
    with pytest.raises(ValueError):
        validator(drifted)


def _valid_lock() -> dict[str, object]:
    context, authorization, qualification, consumption, outcome = _publication_fixture()
    ledger_bytes = b'{"ledger":true}\n'
    result_bytes = b'{"result":true}\n'
    return runner._c11_lock_value(
        context,
        authorization,
        qualification,
        consumption,
        outcome,
        ledger_bytes=ledger_bytes,
        result_bytes=result_bytes,
    )


def test_engineering_failure_terminal_and_ledger_are_fixed_and_non_disclosing() -> None:
    context, authorization, qualification, _consumption, _outcome_value = (
        _publication_fixture()
    )
    qualification.reader = runner.GuardedReader()
    payloads = runner._c11_terminal_payloads(
        context,
        authorization,
        qualification,
        consumption=None,
        failure_stage="ANCHOR_CONSUMPTION",
        anchor_durable_entry_created_or_partial=True,
    )
    assert set(payloads) == {
        "full_replay_diagnostic_read_ledger.json",
        "full_replay_diagnostic_terminal.json",
    }
    ledger = runner._decode(
        payloads["full_replay_diagnostic_read_ledger.json"],
        name="terminal ledger",
    )
    assert ledger["anchor_durable_entry_created_or_partial"] is True
    assert ledger["anchor_consumed_before_manifest"] is True
    assert ledger["anchor_receipt_available"] is False
    for data in payloads.values():
        value = runner._decode(data, name="terminal artifact")
        runner._assert_aggregate_safe(value)
        assert "private" not in json.dumps(value)


def test_build_payloads_treats_core_incompatibility_as_successful_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, qualification, consumption, outcome = _publication_fixture(
        compatible=False
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "consume_authorization", lambda *_args: consumption)
    monkeypatch.setattr(
        runner, "read_adapt_diagnose_full_inventory", lambda *_args: outcome
    )
    payloads = runner.build_payloads(context, authorization, qualification)
    assert set(payloads) == {
        "full_replay_diagnostic_lock.json",
        "full_replay_diagnostic_read_ledger.json",
        "full_replay_diagnostic_result.json",
    }
    result = runner._decode(
        payloads["full_replay_diagnostic_result.json"], name="result"
    )
    assert result["diagnostic_complete"] is True
    assert result["all_records_compatible"] is False
    assert result["status"] == "SOURCE_REPLAY_INCOMPATIBLE_NONAUTHORIZING"


def test_build_payloads_engineering_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, authorization, qualification, consumption, _outcome_value = (
        _publication_fixture()
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "consume_authorization", lambda *_args: consumption)

    def fail(*_args: object) -> object:
        raise runner.RunnerFailure(
            "CACHE_MANIFEST_READ", "C11_CACHE_MANIFEST_READ_FAILED"
        )

    monkeypatch.setattr(runner, "read_adapt_diagnose_full_inventory", fail)
    payloads = runner.build_payloads(context, authorization, qualification)
    assert set(payloads) == {
        "full_replay_diagnostic_read_ledger.json",
        "full_replay_diagnostic_terminal.json",
    }


@pytest.mark.parametrize("residue_kind", ["staging", "quarantine"])
def test_publication_residue_blocks_reuse_before_authorization(
    residue_kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    residue = runner.OUTPUT_ROOT.parent / (
        f".{runner.OUTPUT_ROOT.name}.{residue_kind}-foreign"
    )
    residue.mkdir()
    with pytest.raises(FileExistsError):
        runner.prevalidate_fresh_output()
    assert not runner.C11_ANCHOR.exists()


def test_atomic_publication_is_exact_and_no_clobber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    payloads = _publication_payloads()
    runner.publish_atomically(runner.OUTPUT_ROOT, payloads, freshness=freshness)
    assert {
        path.name: path.read_bytes() for path in runner.OUTPUT_ROOT.iterdir()
    } == payloads
    with pytest.raises(FileExistsError):
        runner.prevalidate_fresh_output()


def test_destination_race_is_never_clobbered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_tree(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    runner.OUTPUT_ROOT.mkdir()
    marker = runner.OUTPUT_ROOT / "foreign.txt"
    marker.write_text("foreign", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.publish_atomically(
            runner.OUTPUT_ROOT, _publication_payloads(), freshness=freshness
        )
    assert marker.read_text(encoding="utf-8") == "foreign"


def test_runner_contains_no_destructive_cleanup_or_adapted_cache_output() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "mutually_exclusive_group(required=True)" in text
    assert "os.O_EXCL" in text
    assert "before_first_read" in text
    assert "diagnose_all_records" in text
    assert "projection_receipt_attestation.json" not in text
    assert "os.unlink(" not in text
    assert "os.remove(" not in text
    assert "shutil.rmtree" not in text
    assert "_quarantine_owned_publication" not in text


def test_public_aggregate_guard_rejects_counts_paths_ordinals_and_non_scalars() -> None:
    for value in (
        {"record_count": True},
        {"path": "private"},
        {"ordinal": "private"},
        {"safe": ["not-flat"]},
        {"safe": b"bytes"},
    ):
        with pytest.raises(ValueError):
            runner._assert_aggregate_safe(value)


def test_no_real_mode_is_invoked_by_import_or_tests() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "run_one_shot"
        for call in calls
    )
    # Static path assertion only: the source-only suite must not even stat the
    # real one-shot anchor namespace.
    assert runner.C11_ANCHOR.parent == ROOT / "research"
