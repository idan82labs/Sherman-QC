from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import types
from typing import Any

import pytest

from scripts import (
    run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1 as runner,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts/run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
)
TEST_SOURCE = (
    ROOT
    / "tests/test_run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1.py"
)
ZERO_SHA = "0" * 64
STAGE_CONTRACT = (
    "CONTEXT_INPUT_HASH_AND_JSON",
    "HEADER",
    "ROW_IDENTITY_ORDER_SNAPSHOT",
    "RECURSIVE_FORBIDDEN_FIELDS",
    "SOURCE_AUTHORITY_ALIAS_SPELLING",
    "CLUSTER_PROVENANCE_NAMESPACE",
    "GEOMETRY_ALIAS_EQUIVALENCE",
    "CANONICAL_PROJECTION_BIJECTION_ORDER",
    "UNCHANGED_C6_VALIDATION",
    "EXACT_C9_OUTCOME_AND_RECEIPT_EQUIVALENCE",
)


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


def _patch_output_roots(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[Path, Path]:
    parent = root / "output/research"
    primary = (
        parent
        / "f1-aug30-train18-candidate13-pool-adapter-stage-localizer-v1"
    )
    fallback = (
        parent
        / "f1-aug30-train18-candidate13-pool-adapter-stage-localizer-publication-fallback-v1"
    )
    monkeypatch.setattr(runner, "OUTPUT_ROOT", primary)
    monkeypatch.setattr(runner, "FALLBACK_OUTPUT_ROOT", fallback)
    monkeypatch.setattr(runner, "ALLOWED_OUTPUT_ROOTS", (primary, fallback))
    return primary, fallback


def _core_result() -> dict[str, Any]:
    return {
        "adapter_or_hypothesis_semantics_modified_by_core": False,
        "artifact_discovery_or_open_performed_by_core": False,
        "builder_or_clusterer_invoked_by_core": False,
        "candidate11_actual_record_cause_resolved": False,
        "candidate9_internal_failure_stage_claimed": False,
        "check_code": "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL",
        "diagnostic_stage_precedence_is_candidate13_contract": True,
        "exact_c9_success_failure_equivalent_for_all": True,
        "first_failed_candidate13_diagnostic_stage": "NONE",
        "fit_performed": False,
        "full28_opened": False,
        "holdout_opened": False,
        "input_or_projection_bytes_published": False,
        "input_origin_authenticated_by_core": False,
        "kind": "sherman_qc_f1_aug30_candidate13_pool_adapter_stage_result_v1",
        "label_or_evaluator_opened": False,
        "metric_computed": False,
        "performance_tested": False,
        "real_or_control_input_opened_by_core": False,
        "rule_selected": False,
        "stage_contract_commitment_sha256": runner.sha256_bytes(
            runner.canonical_bytes(list(STAGE_CONTRACT))
        ),
        "stage_passed_all_bitset": "1" * 10,
        "stage_reached_all_bitset": "1" * 10,
        "status": "PASS_ENGINEERING_ONLY_NONAUTHORIZING",
    }


def test_top_level_is_stdlib_only_and_exact_c13_lineage_is_bound() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "builtins",
        "dataclasses",
        "errno",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "resource",
        "signal",
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
    assert runner.ATTEMPT_ORDINAL == 13
    assert runner.FIXED_SOURCE_BINDINGS["c13_core"][1:] == (
        "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
        65344,
    )
    assert runner.FIXED_SOURCE_BINDINGS["c13_core_tests"][1:] == (
        "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
        43387,
    )


def test_cli_has_three_explicit_modes_no_default_and_auth_only_for_once() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(_base_cli())
    assert runner._parse_args([*_base_cli(), "--source-check-only"]).source_check_only
    assert runner._parse_args(
        [*_base_cli(), "--source-control-check-only"]
    ).source_control_check_only
    parsed = runner._parse_args(
        [
            *_base_cli(),
            "--preflight-once",
            "--authorization-sha256",
            "2" * 64,
            "--authorization-size-bytes",
            "9",
        ]
    )
    assert parsed.preflight_once
    with pytest.raises(SystemExit):
        runner._parse_args([*_base_cli(), "--preflight-once"])
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                *_base_cli(),
                "--source-check-only",
                "--authorization-sha256",
                "2" * 64,
                "--authorization-size-bytes",
                "9",
            ]
        )


def test_direct_bootstrap_rejects_unverified_outer_execution() -> None:
    with pytest.raises(PermissionError, match="verified private bytes"):
        runner.bootstrap_verified_sources(
            runner_sha256="1" * 64,
            runner_size_bytes=1,
            runner_tests_sha256="2" * 64,
            runner_tests_size_bytes=1,
        )


def test_private_exec_ignores_outer_monkeypatch_in_source_check() -> None:
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
        "from scripts import "
        "run_f1_train18_candidate13_pool_adapter_stage_localizer_once_v1 as r\n"
        "r.source_check_result=lambda _context:{'status':'OUTER_PATCH'}\n"
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
    assert "C13_EXACT_SOURCE_BYTES_VERIFIED" in completed.stdout
    assert "OUTER_PATCH" not in completed.stdout


def test_preloaded_alias_subclass_and_import_hook_drift_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, types.ModuleType) and type(module) is not types.ModuleType:
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in runner.FORBIDDEN_PRELOADED:
        monkeypatch.delitem(sys.modules, name, raising=False)
    alias = types.ModuleType("_candidate13_hostile_alias")
    alias.__file__ = str(runner.C13_CORE_SOURCE)
    side_effects: list[str] = []
    alias.__getattr__ = lambda name: side_effects.append(name)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    with pytest.raises(RuntimeError, match="preloaded source alias"):
        runner._reject_preloaded([runner.C13_CORE_SOURCE])
    assert side_effects == []
    monkeypatch.delitem(sys.modules, alias.__name__)

    class HostileModule(types.ModuleType):
        pass

    subclass = HostileModule("_candidate13_hostile_subclass")
    monkeypatch.setitem(sys.modules, subclass.__name__, subclass)
    with pytest.raises(RuntimeError, match="preloaded module subclass"):
        runner._reject_preloaded([runner.C13_CORE_SOURCE])
    monkeypatch.delitem(sys.modules, subclass.__name__)
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, object()])
    with pytest.raises(RuntimeError, match="changed import hooks"):
        runner._reject_preloaded([runner.C13_CORE_SOURCE])


def test_all_36_pool_reads_finish_before_context_then_one_call_per_leaf_and_aggregate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    records = tuple(
        runner.ManifestRecord(
            ordinal=index,
            pair_ordinal=index // 2,
            pass_name=("primary" if index % 2 == 0 else "replay"),
            pool_path=tmp_path / f"pool-{index}.json",
            pool_sha256=runner.sha256_bytes(
                b"" if index == 0 else f"pool-{index}".encode()
            ),
            manifest_record_binding_sha256=f"{index + 1:064x}",
        )
        for index in range(36)
    )

    class FakeReader:
        def __init__(self) -> None:
            self.entries: list[dict[str, Any]] = []
            self.raw_fd_opened_purposes: list[str] = []
            self.verified_inode_purposes: list[str] = []
            self.read_started_purposes: list[str] = []
            self.denied_attempts: list[dict[str, Any]] = []
            self.artifact_inventory: Any = None

        def authorize(self, _path: Path, *, purpose: str, **_kwargs: Any) -> None:
            events.append(f"authorize:{purpose}")

        def read_bytes(self, path: Path, *, purpose: str) -> bytes:
            events.append(f"read:{purpose}")
            data = b"{}" if purpose == "TRAIN_CACHE_MANIFEST" else (
                b"" if path == records[0].pool_path else path.stem.encode()
            )
            self.entries.append(
                {
                    "device_id": 1,
                    "inode": len(self.entries) + 1,
                    "path": str(path),
                    "purpose": purpose,
                    "sha256": runner.sha256_bytes(data),
                    "size_bytes": len(data),
                }
            )
            self.raw_fd_opened_purposes.append(purpose)
            self.verified_inode_purposes.append(purpose)
            self.read_started_purposes.append(purpose)
            return data

        def install_artifact_inventory(self, specifications: Any) -> Any:
            assert type(specifications) is tuple and len(specifications) == 36
            bindings = {
                path: SimpleNamespace(size_bytes=(0 if path == records[0].pool_path else len(path.stem.encode())))
                for path, _purpose, _sha in specifications
            }
            self.artifact_inventory = SimpleNamespace(
                bindings=bindings,
                commitment_sha256="a" * 64,
            )
            events.append("inventory")
            return self.artifact_inventory

    class FakeCore:
        RESULT_KIND = _core_result()["kind"]
        RESULT_KEYS = frozenset(_core_result())
        STAGE_CONTRACT = STAGE_CONTRACT

        def make_verified_context(self, **_kwargs: Any) -> object:
            assert len([event for event in events if event.startswith("read:CANDIDATE13_")]) == 36
            events.append("make_context")
            return object()

        def localize_pool_adapter_contract(
            self,
            _context: object,
            *,
            pool_bytes: bytes,
            expected_pool_sha256: str,
        ) -> object:
            assert runner.sha256_bytes(pool_bytes) == expected_pool_sha256
            events.append("localize")
            return object()

        def aggregate_pool_adapter_receipts(
            self, _context: object, receipts: tuple[object, ...]
        ) -> dict[str, Any]:
            assert type(receipts) is tuple and len(receipts) == 36
            assert len({id(receipt) for receipt in receipts}) == 36
            events.append("aggregate")
            return _core_result()

    reader = FakeReader()
    context = SimpleNamespace(
        c13_core=FakeCore(),
        c9_adapter=object(),
        c6_hypothesis=object(),
        builder=object(),
        clusterer=object(),
    )
    qualification = SimpleNamespace(
        reader=reader,
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="2" * 64,
        manifest_size_bytes=2,
    )
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_resource_envelope", lambda *_args: None)
    monkeypatch.setattr(runner, "revalidate_anchor", lambda *_args: None)
    monkeypatch.setattr(runner, "_decode", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_manifest_file_binding", lambda *_args: "3" * 64)
    monkeypatch.setattr(runner, "_validate_full_manifest", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(runner, "_validate_complete_inventory", lambda *_args: None)
    monkeypatch.setattr(runner, "_c13_core_bindings", lambda: {})
    outcome = runner.read_localize_full_pool_inventory(
        context,
        qualification,
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert outcome.core_result["check_code"] == "C13_EXACT_C9_POOL_ADAPTER_ACCEPTED_FOR_ALL"
    assert events.count("make_context") == 1
    assert events.count("localize") == 36
    assert events.count("aggregate") == 1
    assert events.index("make_context") > max(
        index
        for index, event in enumerate(events)
        if event.startswith("read:CANDIDATE13_")
    )
    assert events.index("aggregate") > max(
        index for index, event in enumerate(events) if event == "localize"
    )


def _manifest_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], SimpleNamespace, Path]:
    root = tmp_path / "repo"
    allowed = (
        root
        / "output/research/f1-v51-today-scan-only-development-20260830-r2"
    )
    allowed.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", allowed)
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
    rows: list[dict[str, Any]] = []
    for pair, case_id in enumerate(train_ids):
        runtime_id = f"T{pair:03d}"
        pool_sha = runner.sha256_bytes(f"pool-{case_id}".encode())
        semantic_sha = runner.sha256_bytes(f"semantic-{case_id}".encode())
        for pass_name in ("primary", "replay"):
            rows.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "pool_path": str(
                        (
                            allowed
                            / pass_name
                            / runtime_id
                            / "pool"
                            / f"{runtime_id}.json"
                        ).relative_to(root)
                    ),
                    "pool_sha256": pool_sha,
                    "runtime_case_id": runtime_id,
                    "scan_sha256": scan_by_case[case_id],
                    "semantic_path": str(
                        (
                            allowed
                            / pass_name
                            / runtime_id
                            / "semantic_result.json"
                        ).relative_to(root)
                    ),
                    "semantic_sha256": semantic_sha,
                }
            )
    qualification = SimpleNamespace(
        train_ids=train_ids,
        scan_by_case=scan_by_case,
        train_commitment_sha256=train_commitment,
    )
    manifest = {
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
    }
    return manifest, qualification, allowed


def test_manifest_accepts_exact_ordered_36_pool_paths_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, qualification, _allowed = _manifest_fixture(monkeypatch, tmp_path)
    records = runner._validate_full_manifest(
        manifest,
        qualification,
        manifest_binding_sha256="1" * 64,
    )
    assert len(records) == 36
    assert len({record.pool_path for record in records}) == 36
    assert [record.pass_name for record in records[:4]] == [
        "primary",
        "replay",
        "primary",
        "replay",
    ]
    assert all(not hasattr(record, "semantic_path") for record in records)


@pytest.mark.parametrize(
    "mutation", ("short", "reorder", "duplicate_path", "replay_hash")
)
def test_manifest_short_reorder_duplicate_and_replay_drift_reject(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, qualification, _allowed = _manifest_fixture(monkeypatch, tmp_path)
    rows = manifest["records"]
    assert isinstance(rows, list)
    if mutation == "short":
        rows.pop()
    elif mutation == "reorder":
        rows[0], rows[2] = rows[2], rows[0]
    elif mutation == "duplicate_path":
        rows[2]["pool_path"] = rows[0]["pool_path"]
    else:
        rows[1]["pool_sha256"] = "f" * 64
    with pytest.raises(runner.RunnerFailure):
        runner._validate_full_manifest(
            manifest,
            qualification,
            manifest_binding_sha256="1" * 64,
        )


def test_semantic_leaf_can_never_be_authorized_or_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    allowed = root / "allowed"
    allowed.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", allowed)
    semantic = allowed / "semantic_result.json"
    semantic.write_bytes(b"")
    reader = runner.GuardedReader()
    with pytest.raises(PermissionError, match="FORBIDDEN_BASENAME"):
        reader.authorize(
            semantic,
            purpose="CANDIDATE13_PRIMARY_00_POOL",
            sha256=runner.sha256_bytes(b""),
            size_bytes=0,
        )
    specs = []
    for index, purpose in enumerate(runner._expected_pool_purposes()):
        path = semantic if index == 0 else allowed / f"pool-{index}.json"
        path.write_bytes(b"{}")
        specs.append((path, purpose, runner.sha256_bytes(path.read_bytes())))
    with pytest.raises(PermissionError, match="FORBIDDEN_BASENAME"):
        reader.install_artifact_inventory(tuple(specs))


def _pool_inventory_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    inode_alias: bool = False,
) -> tuple[runner.GuardedReader, tuple[tuple[Path, str, str], ...]]:
    root = tmp_path / "repo"
    allowed = root / "allowed"
    allowed.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "ALLOWED_CACHE_ROOT", allowed)
    paths = [allowed / f"pool-{index}.json" for index in range(36)]
    for index, path in enumerate(paths):
        if inode_alias and index == 1:
            os.link(paths[0], path)
        else:
            path.write_bytes(f"{index:02d}".encode())
    specifications = tuple(
        (
            path,
            purpose,
            runner.sha256_bytes(path.read_bytes()),
        )
        for path, purpose in zip(paths, runner._expected_pool_purposes())
    )
    return runner.GuardedReader(), specifications


def test_pool_inventory_requires_unique_paths_and_inodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reader, specifications = _pool_inventory_fixture(
        monkeypatch, tmp_path / "valid"
    )
    inventory = reader.install_artifact_inventory(specifications)
    assert len(inventory.bindings) == 36
    reader2, aliased = _pool_inventory_fixture(
        monkeypatch, tmp_path / "aliased", inode_alias=True
    )
    with pytest.raises(ValueError, match="metadata or inode"):
        reader2.install_artifact_inventory(aliased)


def test_guarded_pool_read_records_raw_inode_read_start_and_rejects_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reader, specifications = _pool_inventory_fixture(monkeypatch, tmp_path)
    inventory = reader.install_artifact_inventory(specifications)
    path, purpose, sha256 = specifications[0]
    binding = inventory.bindings[path]
    reader.authorize(
        path,
        purpose=purpose,
        sha256=sha256,
        size_bytes=binding.size_bytes,
        artifact_inventory=inventory,
    )
    assert reader.read_bytes(path, purpose=purpose) == b"00"
    assert reader.raw_fd_opened_purposes == [purpose]
    assert reader.verified_inode_purposes == [purpose]
    assert reader.read_started_purposes == [purpose]
    with pytest.raises(PermissionError, match="DUPLICATE_READ_FORBIDDEN"):
        reader.read_bytes(path, purpose=purpose)


def test_guarded_pool_read_rejects_same_size_hash_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reader, specifications = _pool_inventory_fixture(monkeypatch, tmp_path)
    inventory = reader.install_artifact_inventory(specifications)
    path, purpose, sha256 = specifications[0]
    path.write_bytes(b"XX")
    reader.authorize(
        path,
        purpose=purpose,
        sha256=sha256,
        size_bytes=2,
        artifact_inventory=inventory,
    )
    with pytest.raises(ValueError, match="exact byte binding"):
        reader.read_bytes(path, purpose=purpose)


def test_qualification_and_authorization_substitution_are_coupled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo"
    (root / "research").mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner,
        "C13_ANCHOR",
        root
        / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1.json",
    )
    _patch_output_roots(monkeypatch, root)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", root / "cache/manifest.json")
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    context = SimpleNamespace(source_commitment_sha256="1" * 64)
    qualification = SimpleNamespace(
        manifest_path=runner.CACHE_MANIFEST,
        manifest_sha256=runner.CACHE_MANIFEST_SHA256,
        manifest_size_bytes=runner.CACHE_MANIFEST_SIZE_BYTES,
        control_binding_commitment_sha256="2" * 64,
        control_raw_fd_opened_commitment_sha256="3" * 64,
        control_read_inventory_commitment_sha256="4" * 64,
        control_read_started_commitment_sha256="5" * 64,
        control_verified_inode_commitment_sha256="6" * 64,
    )
    qualification_value = runner._decode(
        runner.build_qualification_record(context, qualification),
        name="qualification",
    )
    runner._qualification_static_contract(
        context,
        qualification_value,
        control_commitment="2" * 64,
    )
    for key in (
        "control_binding_commitment_sha256",
        "source_inventory_commitment_sha256",
        "manifest_binding_commitment_sha256",
        "allowed_output_namespace_set_commitment_sha256",
    ):
        drifted = dict(qualification_value)
        drifted[key] = "f" * 64
        with pytest.raises(PermissionError):
            runner._qualification_static_contract(
                context,
                drifted,
                control_commitment="2" * 64,
            )
    authorization = runner.build_authorization_contract(
        context,
        qualification_record_sha256=runner.sha256_bytes(
            runner.canonical_bytes(qualification_value)
        ),
        control_binding_commitment_sha256="2" * 64,
    )
    assert authorization["control_binding_commitment_sha256"] == "2" * 64
    assert authorization["allowed_output_namespace_set_commitment_sha256"] == (
        runner._allowed_output_namespace_set_commitment()
    )
    with pytest.raises(PermissionError):
        runner._qualification_static_contract(
            context,
            qualification_value,
            control_commitment="f" * 64,
        )


def _preanchor_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, runner.GuardedReader]:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", root / "cache/manifest.json")
    purposes = [
        "CANDIDATE13_EXTERNAL_AUTHORIZATION",
        "CANDIDATE13_QUALIFICATION_RECORD",
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
    reader = runner.GuardedReader(entries=entries)
    consumed = {
        runner._lexical(
            root / str(entry["path"]), role=str(entry["purpose"])
        )
        for entry in entries
    }
    reader.allowed = {path: {} for path in consumed}
    reader.consumed_paths = set(consumed)
    context = SimpleNamespace(source_entries=())
    authorization = SimpleNamespace(reader=reader)
    qualification = SimpleNamespace(reader=reader)
    return context, authorization, qualification, reader


@pytest.mark.parametrize("contamination", ("inventory", "manifest", "pool"))
def test_preanchor_manifest_inventory_and_pool_contamination_reject(
    contamination: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, authorization, qualification, reader = _preanchor_fixture(
        monkeypatch, tmp_path
    )
    runner._validate_preanchor_reader_state(
        context, authorization, qualification
    )
    if contamination == "inventory":
        reader.artifact_inventory = object()
    else:
        purpose = (
            "TRAIN_CACHE_MANIFEST"
            if contamination == "manifest"
            else "CANDIDATE13_PRIMARY_00_POOL"
        )
        path = runner._lexical(
            runner.ROOT / f"synthetic/{contamination}.json",
            role=purpose,
        )
        entry = {
            "device_id": 2,
            "inode": 100,
            "path": str(path.relative_to(runner.ROOT)),
            "purpose": purpose,
            "sha256": "f" * 64,
            "size_bytes": 1,
        }
        reader.entries.append(entry)
        reader.raw_fd_opened_purposes.append(purpose)
        reader.verified_inode_purposes.append(purpose)
        reader.read_started_purposes.append(purpose)
        reader.allowed[path] = {}
        reader.consumed_paths.add(path)
    with pytest.raises(PermissionError, match="pre-anchor"):
        runner._validate_preanchor_reader_state(
            context, authorization, qualification
        )


def test_authentic_resource_seal_checks_live_caps_handler_and_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = object()
    alarm_handler = lambda _signum, _frame: None
    prof_handler = lambda _signum, _frame: None
    seal = runner.ResourceEnvelopeSeal(
        prior_address_space_limit=(10_000, 20_000),
        prior_alarm_handler=signal.SIG_DFL,
        prior_prof_handler=signal.SIG_IGN,
        prior_signal_mask=frozenset(),
        applied_address_space_soft_limit=runner.ADDRESS_SPACE_LIMIT_BYTES,
        installed_alarm_handler=alarm_handler,
        installed_prof_handler=prof_handler,
        triggered=[False, False],
        _sentinel=runner._RESOURCE_ENVELOPE_SEAL,
        _issuance_token=token,
    )
    runner._ISSUED_RESOURCE_ENVELOPES[id(seal)] = (seal, token)
    monkeypatch.setattr(
        runner.resource,
        "getrlimit",
        lambda _kind: (runner.ADDRESS_SPACE_LIMIT_BYTES, 20_000),
    )
    monkeypatch.setattr(
        runner.signal,
        "getsignal",
        lambda value: (
            alarm_handler if value == signal.SIGALRM else prof_handler
        ),
    )
    monkeypatch.setattr(runner.signal, "getitimer", lambda _timer: (30.0, 0.0))
    monkeypatch.setattr(runner, "_current_signal_mask", lambda: frozenset())
    runner._validate_resource_envelope(seal)
    monkeypatch.setattr(
        runner.signal,
        "getitimer",
        lambda timer: (
            (runner.CPU_TIMER_SECONDS + 1.0, 0.0)
            if timer == signal.ITIMER_PROF
            else (30.0, 0.0)
        ),
    )
    with pytest.raises(runner.ResourceEnvelopeFailure):
        runner._validate_resource_envelope(seal)
    monkeypatch.setattr(
        runner.signal,
        "getitimer",
        lambda timer: (
            (30.0, 1.0)
            if timer == signal.ITIMER_REAL
            else (30.0, 0.0)
        ),
    )
    with pytest.raises(runner.ResourceEnvelopeFailure):
        runner._validate_resource_envelope(seal)
    monkeypatch.setattr(
        runner.signal,
        "getitimer",
        lambda timer: (0.0, 0.0)
        if timer == signal.ITIMER_PROF
        else (30.0, 0.0),
    )
    with pytest.raises(runner.ResourceEnvelopeFailure):
        runner._validate_resource_envelope(seal)
    runner._ISSUED_RESOURCE_ENVELOPES.pop(id(seal), None)


def test_resource_contract_uses_cancelable_cpu_and_wall_timers() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "ITIMER_PROF" in source
    assert "SIGPROF" in source
    assert "ITIMER_REAL" in source
    assert "SIGALRM" in source
    assert "RLIMIT_CPU" not in source
    assert "SIGXCPU" not in source
    assert "CPU_LIMIT_SECONDS" not in source


def test_reserved_publication_phase_cancels_timers_drains_late_signal_and_keeps_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alarm = int(signal.SIGALRM)
    prof = int(signal.SIGPROF)
    signals = frozenset((alarm, prof))
    mask: set[int] = set()
    pending: set[int] = {alarm}
    timers = {
        signal.ITIMER_REAL: (12.0, 0.0),
        signal.ITIMER_PROF: (9.0, 0.0),
    }
    prior_alarm = object()
    prior_prof = object()
    installed_alarm = object()
    installed_prof = object()
    handlers = {
        signal.SIGALRM: installed_alarm,
        signal.SIGPROF: installed_prof,
    }
    address = [runner.ADDRESS_SPACE_LIMIT_BYTES, 20_000_000_000]

    def pthread_sigmask(how: int, values: Any) -> frozenset[int]:
        old = frozenset(mask)
        values_set = {int(value) for value in values}
        if how == signal.SIG_BLOCK:
            mask.update(values_set)
        elif how == signal.SIG_SETMASK:
            mask.clear()
            mask.update(values_set)
        else:
            raise AssertionError(how)
        return old

    def setitimer(which: int, seconds: float, _interval: float = 0.0) -> tuple[float, float]:
        old = timers[which]
        timers[which] = (float(seconds), 0.0)
        return old

    def set_handler(which: int, value: Any) -> Any:
        old = handlers[which]
        handlers[which] = value
        return old

    def set_limit(_kind: int, value: tuple[int, int]) -> None:
        address[:] = value

    monkeypatch.setattr(runner.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(runner.signal, "sigpending", lambda: set(pending))
    monkeypatch.setattr(
        runner.signal,
        "sigwait",
        lambda values: pending.remove(next(iter(values))),
    )
    monkeypatch.setattr(runner.signal, "setitimer", setitimer)
    monkeypatch.setattr(runner.signal, "getitimer", lambda which: timers[which])
    monkeypatch.setattr(runner.signal, "signal", set_handler)
    monkeypatch.setattr(runner.signal, "getsignal", lambda which: handlers[which])
    monkeypatch.setattr(
        runner.resource, "getrlimit", lambda _kind: tuple(address)
    )
    monkeypatch.setattr(runner.resource, "setrlimit", set_limit)
    token = object()
    envelope = runner.ResourceEnvelopeSeal(
        prior_address_space_limit=(30_000_000_000, 40_000_000_000),
        prior_alarm_handler=prior_alarm,
        prior_prof_handler=prior_prof,
        prior_signal_mask=frozenset(),
        applied_address_space_soft_limit=runner.ADDRESS_SPACE_LIMIT_BYTES,
        installed_alarm_handler=installed_alarm,
        installed_prof_handler=installed_prof,
        triggered=[False, False],
        _sentinel=runner._RESOURCE_ENVELOPE_SEAL,
        _issuance_token=token,
    )
    runner._ISSUED_RESOURCE_ENVELOPES[id(envelope)] = (envelope, token)
    phase = runner._enter_reserved_publication_phase(envelope)
    runner._validate_publication_phase(phase)
    assert timers[signal.ITIMER_REAL] == (0.0, 0.0)
    assert timers[signal.ITIMER_PROF] == (0.0, 0.0)
    assert pending == set()
    assert mask == set(signals)
    assert handlers == {
        signal.SIGALRM: prior_alarm,
        signal.SIGPROF: prior_prof,
    }
    assert tuple(address) == envelope.prior_address_space_limit


@pytest.mark.parametrize(
    "exception_type", (runner.WatchdogExpired, runner.CpuTimerExpired)
)
@pytest.mark.parametrize(
    "injection_point",
    (
        "before_block",
        "after_block",
        "after_cancel_real",
        "after_cancel_prof",
        "after_restore_as",
        "after_restore_alarm_handler",
        "after_restore_prof_handler",
        "during_drain",
    ),
)
def test_reserved_publication_transition_retries_every_timer_boundary(
    exception_type: type[BaseException],
    injection_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alarm = int(signal.SIGALRM)
    prof = int(signal.SIGPROF)
    resource_signals = frozenset((alarm, prof))
    mask: set[int] = set()
    timers = {
        signal.ITIMER_REAL: (12.0, 0.0),
        signal.ITIMER_PROF: (9.0, 0.0),
    }
    prior_alarm = object()
    prior_prof = object()
    installed_alarm = object()
    installed_prof = object()
    handlers = {
        signal.SIGALRM: installed_alarm,
        signal.SIGPROF: installed_prof,
    }
    prior_address = (30_000_000_000, 40_000_000_000)
    address = [runner.ADDRESS_SPACE_LIMIT_BYTES, prior_address[1]]
    injected = [False]
    envelope_holder: list[runner.ResourceEnvelopeSeal] = []

    def interrupt(point: str) -> None:
        if injection_point != point or injected[0]:
            return
        injected[0] = True
        envelope = envelope_holder[0]
        envelope.triggered[
            0 if exception_type is runner.WatchdogExpired else 1
        ] = True
        raise exception_type()

    def pthread_sigmask(how: int, values: Any) -> frozenset[int]:
        old = frozenset(mask)
        values_set = {int(value) for value in values}
        is_transition_block = (
            how == signal.SIG_BLOCK and values_set == set(resource_signals)
        )
        if is_transition_block:
            interrupt("before_block")
        if how == signal.SIG_BLOCK:
            mask.update(values_set)
        elif how == signal.SIG_SETMASK:
            mask.clear()
            mask.update(values_set)
        else:
            raise AssertionError(how)
        if is_transition_block:
            interrupt("after_block")
        return old

    def setitimer(
        which: int, seconds: float, _interval: float = 0.0
    ) -> tuple[float, float]:
        old = timers[which]
        timers[which] = (float(seconds), 0.0)
        if which == signal.ITIMER_REAL:
            interrupt("after_cancel_real")
        else:
            interrupt("after_cancel_prof")
        return old

    def set_limit(_kind: int, value: tuple[int, int]) -> None:
        address[:] = value
        interrupt("after_restore_as")

    def set_handler(which: int, value: Any) -> Any:
        old = handlers[which]
        handlers[which] = value
        interrupt(
            "after_restore_alarm_handler"
            if which == signal.SIGALRM
            else "after_restore_prof_handler"
        )
        return old

    def pending_values() -> set[int]:
        interrupt("during_drain")
        return set()

    monkeypatch.setattr(runner.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(runner.signal, "sigpending", pending_values)
    monkeypatch.setattr(runner.signal, "sigwait", lambda _values: None)
    monkeypatch.setattr(runner.signal, "setitimer", setitimer)
    monkeypatch.setattr(runner.signal, "getitimer", lambda which: timers[which])
    monkeypatch.setattr(runner.signal, "signal", set_handler)
    monkeypatch.setattr(runner.signal, "getsignal", lambda which: handlers[which])
    monkeypatch.setattr(
        runner.resource, "getrlimit", lambda _kind: tuple(address)
    )
    monkeypatch.setattr(runner.resource, "setrlimit", set_limit)
    token = object()
    envelope = runner.ResourceEnvelopeSeal(
        prior_address_space_limit=prior_address,
        prior_alarm_handler=prior_alarm,
        prior_prof_handler=prior_prof,
        prior_signal_mask=frozenset(),
        applied_address_space_soft_limit=runner.ADDRESS_SPACE_LIMIT_BYTES,
        installed_alarm_handler=installed_alarm,
        installed_prof_handler=installed_prof,
        triggered=[False, False],
        _sentinel=runner._RESOURCE_ENVELOPE_SEAL,
        _issuance_token=token,
    )
    envelope_holder.append(envelope)
    runner._ISSUED_RESOURCE_ENVELOPES[id(envelope)] = (envelope, token)
    phase = runner._enter_reserved_publication_phase(envelope)
    assert injected == [True]
    runner._validate_publication_phase(phase)
    assert mask == set(resource_signals)
    assert timers == {
        signal.ITIMER_REAL: (0.0, 0.0),
        signal.ITIMER_PROF: (0.0, 0.0),
    }
    assert handlers == {
        signal.SIGALRM: prior_alarm,
        signal.SIGPROF: prior_prof,
    }
    assert tuple(address) == prior_address
    terminal_payloads = {"terminal": b"fixed\n"}
    monkeypatch.setattr(
        runner,
        "_payloads_from_plan",
        lambda *_args: terminal_payloads,
    )
    assert runner._materialize_payload_plan_in_publication_phase(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        phase,
    ) == terminal_payloads


@pytest.mark.parametrize(
    "exception_type", (runner.WatchdogExpired, runner.CpuTimerExpired)
)
@pytest.mark.parametrize("injection_point", ("before_block", "after_block"))
def test_preanchor_publication_transition_recovers_block_boundary(
    exception_type: type[BaseException],
    injection_point: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_signals = frozenset((int(signal.SIGALRM), int(signal.SIGPROF)))
    mask: set[int] = set()
    injected = [False]
    handlers = {
        signal.SIGALRM: signal.SIG_DFL,
        signal.SIGPROF: signal.SIG_DFL,
    }

    def pthread_sigmask(how: int, values: Any) -> frozenset[int]:
        old = frozenset(mask)
        values_set = {int(value) for value in values}
        transition = how == signal.SIG_BLOCK and values_set == set(resource_signals)
        if transition and injection_point == "before_block" and not injected[0]:
            injected[0] = True
            raise exception_type()
        if how == signal.SIG_BLOCK:
            mask.update(values_set)
        elif how == signal.SIG_SETMASK:
            mask.clear()
            mask.update(values_set)
        if transition and injection_point == "after_block" and not injected[0]:
            injected[0] = True
            raise exception_type()
        return old

    monkeypatch.setattr(runner.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(runner.signal, "sigpending", lambda: set())
    monkeypatch.setattr(runner.signal, "sigwait", lambda _values: None)
    monkeypatch.setattr(
        runner.signal, "getitimer", lambda _which: (0.0, 0.0)
    )
    monkeypatch.setattr(
        runner.signal, "setitimer", lambda _which, _seconds: (0.0, 0.0)
    )
    monkeypatch.setattr(runner.signal, "getsignal", lambda which: handlers[which])
    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda which, value: handlers.__setitem__(which, value),
    )
    prior_address = (30_000_000_000, 40_000_000_000)
    monkeypatch.setattr(runner.resource, "getrlimit", lambda _kind: prior_address)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_args: None)
    phase = runner._enter_reserved_publication_phase(None)
    assert injected == [True]
    runner._validate_publication_phase(phase)
    assert mask == set(resource_signals)


def test_preanchor_transition_failure_restores_mask_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_signals = frozenset((int(signal.SIGALRM), int(signal.SIGPROF)))
    mask: set[int] = set()
    fail_once = [True]
    handlers = {
        signal.SIGALRM: signal.SIG_DFL,
        signal.SIGPROF: signal.SIG_DFL,
    }

    def pthread_sigmask(how: int, values: Any) -> frozenset[int]:
        old = frozenset(mask)
        values_set = {int(value) for value in values}
        if how == signal.SIG_BLOCK:
            mask.update(values_set)
        elif how == signal.SIG_SETMASK:
            mask.clear()
            mask.update(values_set)
        return old

    def setitimer(_which: int, _seconds: float) -> tuple[float, float]:
        if fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("synthetic pre-anchor transition failure")
        return 0.0, 0.0

    prior_address = (30_000_000_000, 40_000_000_000)
    monkeypatch.setattr(runner.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(runner.signal, "sigpending", lambda: set())
    monkeypatch.setattr(runner.signal, "sigwait", lambda _values: None)
    monkeypatch.setattr(
        runner.signal, "getitimer", lambda _which: (0.0, 0.0)
    )
    monkeypatch.setattr(runner.signal, "setitimer", setitimer)
    monkeypatch.setattr(runner.signal, "getsignal", lambda which: handlers[which])
    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda which, value: handlers.__setitem__(which, value),
    )
    monkeypatch.setattr(runner.resource, "getrlimit", lambda _kind: prior_address)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_args: None)
    with pytest.raises(runner.ResourceEnvelopeFailure):
        runner._enter_reserved_publication_phase(None)
    assert mask == set()
    phase = runner._enter_reserved_publication_phase(None)
    runner._validate_publication_phase(phase)
    assert mask == set(resource_signals)


@pytest.mark.parametrize(
    ("exception_type", "failure_stage"),
    (
        (runner.WatchdogExpired, "WATCHDOG_TIMEOUT"),
        (runner.CpuTimerExpired, "CPU_TIMER_TIMEOUT"),
    ),
)
def test_wall_and_cpu_timeout_after_anchor_produce_truthful_reserved_plan(
    exception_type: type[BaseException],
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumption = SimpleNamespace()
    phase = object()
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_resource_envelope", lambda *_args: None)

    def consume(*_args: Any, progress: runner.ConsumptionProgress) -> Any:
        progress.anchor_durable_entry_created_or_partial = True
        progress.consumption = consumption
        return consumption

    monkeypatch.setattr(runner, "consume_authorization", consume)
    monkeypatch.setattr(
        runner,
        "read_localize_full_pool_inventory",
        lambda *_args: (_ for _ in ()).throw(exception_type()),
    )
    monkeypatch.setattr(
        runner, "_enter_reserved_publication_phase", lambda *_args: phase
    )
    plan = runner._compute_payload_plan(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        resource_envelope=SimpleNamespace(),
        progress=runner.ConsumptionProgress(),
    )
    assert plan.failure_stage == failure_stage
    assert plan.consumption is consumption
    assert plan.anchor_durable_entry_created_or_partial is True
    assert plan.resource_envelope_enforced is True


def test_watchdog_is_fixed_plan_before_any_core_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_resource_envelope", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "consume_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runner.WatchdogExpired()),
    )
    phase = object()
    monkeypatch.setattr(
        runner, "_enter_reserved_publication_phase", lambda *_args: phase
    )
    plan = runner._compute_payload_plan(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        resource_envelope=SimpleNamespace(),
        progress=runner.ConsumptionProgress(),
    )
    assert plan.failure_stage == "WATCHDOG_TIMEOUT"
    assert plan.resource_envelope_enforced is True


def test_public_aggregate_terminal_ledger_and_lock_are_value_free() -> None:
    aggregate = runner._terminal_aggregate_value(
        failure_stage="WATCHDOG_TIMEOUT"
    )
    runner._assert_aggregate_safe(aggregate)
    assert aggregate["failure_stage"] == "WATCHDOG_TIMEOUT"
    encoded = runner.canonical_bytes(aggregate).decode("utf-8")
    for forbidden in (
        "case_id",
        "count",
        "elapsed",
        "identifier",
        "ordinal",
        '"path"',
        "rss",
        "size_bytes",
        "timing",
    ):
        assert forbidden not in encoded
    assert set(runner._payloads.__annotations__) >= {
        "consumption",
        "outcome",
        "failure_stage",
        "resource_envelope_enforced",
    }


def test_external_auth_anchor_and_atomic_publication_contracts_are_closed() -> None:
    auth_source = inspect.getsource(runner.authorize_preflight_once)
    anchor_source = inspect.getsource(runner.consume_authorization)
    publication_source = inspect.getsource(runner.publish_atomically)
    assert "CANDIDATE13_EXTERNAL_AUTHORIZATION" in auth_source
    assert "O_EXCL" in anchor_source
    assert "CONSUMED_BEFORE_MANIFEST_OPEN" in anchor_source
    assert "_renameat_noreplace" in publication_source
    assert "os.unlink" not in publication_source
    assert "rmtree" not in publication_source
    assert {
        "pool_adapter_stage_localizer_aggregate.json",
        "pool_adapter_stage_localizer_lock.json",
        "pool_adapter_stage_localizer_read_ledger.json",
    } <= set(publication_source.split('"'))


def _anchor_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, Path]:
    root = tmp_path / "repo"
    research = root / "research"
    research.mkdir(parents=True)
    anchor = (
        research
        / "f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1.json"
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "C13_ANCHOR", anchor)
    monkeypatch.setattr(runner, "CACHE_MANIFEST", root / "cache/manifest.json")
    _patch_output_roots(monkeypatch, root)
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_authorization", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_qualification", lambda *_args: None)
    monkeypatch.setattr(
        runner, "_validate_preanchor_reader_state", lambda *_args: None
    )
    monkeypatch.setattr(
        runner, "build_qualification_record", lambda *_args: b"qualification"
    )
    reader = runner.GuardedReader()
    token = object()
    context = SimpleNamespace(
        source_commitment_sha256="1" * 64,
        _issuance_token=token,
    )
    qualification = SimpleNamespace(
        reader=reader,
        control_binding_commitment_sha256="2" * 64,
    )
    authorization = SimpleNamespace(
        reader=reader,
        qualification_bytes=b"qualification",
        control_binding_commitment_sha256="2" * 64,
        authorization_sha256="3" * 64,
        qualification_sha256="4" * 64,
        anchor_parent_identity=runner._identity(os.lstat(research)),
    )
    return context, authorization, qualification, anchor


def _terminal_ledger(
    context: SimpleNamespace,
    authorization: SimpleNamespace,
    qualification: SimpleNamespace,
    *,
    consumption: Any,
    failure_stage: str,
    durable: bool,
) -> dict[str, Any]:
    payloads = runner._payloads(
        context,
        authorization,
        qualification,
        consumption=consumption,
        outcome=None,
        failure_stage=failure_stage,
        anchor_durable_entry_created_or_partial=durable,
        resource_envelope_enforced=True,
    )
    return runner._decode(
        payloads["pool_adapter_stage_localizer_read_ledger.json"],
        name="synthetic ledger",
    )


@pytest.mark.parametrize(
    "injection",
    ("pre_open", "post_o_excl", "mid_write", "post_fsync"),
)
def test_anchor_watchdog_injection_preserves_durable_truth_and_retry_barrier(
    injection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, authorization, qualification, anchor = _anchor_fixture(
        monkeypatch, tmp_path
    )
    original_open = runner.os.open
    original_write = runner.os.write
    original_fsync = runner.os.fsync
    state: dict[str, Any] = {"leaf_fd": None, "post_open_seen": False}

    def injected_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        is_anchor_create = (
            path == anchor.name and bool(flags & runner.os.O_CREAT)
        )
        if is_anchor_create and injection == "pre_open":
            raise runner.WatchdogExpired()
        descriptor = original_open(path, flags, *args, **kwargs)
        if is_anchor_create:
            state["leaf_fd"] = descriptor
        return descriptor

    def injected_write(descriptor: int, value: Any) -> int:
        if injection == "mid_write" and descriptor == state["leaf_fd"]:
            original_write(descriptor, memoryview(value)[:1])
            raise runner.WatchdogExpired()
        return original_write(descriptor, value)

    def injected_fsync(descriptor: int) -> None:
        if state["leaf_fd"] is not None and injection == "post_o_excl":
            if not state["post_open_seen"]:
                state["post_open_seen"] = True
                raise runner.WatchdogExpired()
        if injection == "post_fsync" and descriptor == state["leaf_fd"]:
            original_fsync(descriptor)
            raise runner.WatchdogExpired()
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "open", injected_open)
    monkeypatch.setattr(runner.os, "write", injected_write)
    monkeypatch.setattr(runner.os, "fsync", injected_fsync)
    if injection == "pre_open":
        with pytest.raises(runner.WatchdogExpired):
            runner.consume_authorization(context, authorization, qualification)
        assert not anchor.exists()
        ledger = _terminal_ledger(
            context,
            authorization,
            qualification,
            consumption=None,
            failure_stage="WATCHDOG_TIMEOUT",
            durable=False,
        )
        assert ledger["anchor_consumed_before_manifest"] is False
        assert ledger["anchor_durable_entry_created_or_partial"] is False
        assert ledger["anchor_receipt_available"] is False
        return

    with pytest.raises(runner.AnchorCreationFailure):
        runner.consume_authorization(context, authorization, qualification)
    assert anchor.exists()
    ledger = _terminal_ledger(
        context,
        authorization,
        qualification,
        consumption=None,
        failure_stage="ANCHOR_CONSUMPTION",
        durable=True,
    )
    assert ledger["anchor_consumed_before_manifest"] is True
    assert ledger["anchor_durable_entry_created_or_partial"] is True
    assert ledger["anchor_receipt_available"] is False
    monkeypatch.setattr(runner.os, "open", original_open)
    monkeypatch.setattr(runner.os, "write", original_write)
    monkeypatch.setattr(runner.os, "fsync", original_fsync)
    with pytest.raises(PermissionError, match="already consumed"):
        runner.consume_authorization(context, authorization, qualification)


@pytest.mark.parametrize(
    "failure_stage", ("WATCHDOG_TIMEOUT", "CPU_TIMER_TIMEOUT")
)
def test_timer_after_anchor_before_manifest_has_receipt_and_blocks_retry(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, authorization, qualification, anchor = _anchor_fixture(
        monkeypatch, tmp_path
    )
    consumption = runner.consume_authorization(
        context, authorization, qualification
    )
    assert anchor.exists()
    anchor_value = runner._decode(anchor.read_bytes(), name="synthetic anchor")
    assert anchor_value["allowed_output_namespace_set_commitment_sha256"] == (
        runner._allowed_output_namespace_set_commitment()
    )
    ledger = _terminal_ledger(
        context,
        authorization,
        qualification,
        consumption=consumption,
        failure_stage=failure_stage,
        durable=False,
    )
    assert ledger["anchor_consumed_before_manifest"] is True
    assert ledger["anchor_durable_entry_created_or_partial"] is True
    assert ledger["anchor_receipt_available"] is True
    with pytest.raises(PermissionError, match="already consumed"):
        runner.consume_authorization(context, authorization, qualification)


def test_anchor_revalidation_rejects_ordered_output_binding_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, authorization, qualification, anchor = _anchor_fixture(
        monkeypatch, tmp_path
    )
    consumption = runner.consume_authorization(
        context, authorization, qualification
    )
    changed = runner._decode(consumption.bytes_value, name="anchor mutation")
    changed["allowed_output_namespace_set_commitment_sha256"] = "f" * 64
    os.chmod(anchor, 0o600)
    anchor.write_bytes(runner.canonical_bytes(changed))
    os.chmod(anchor, 0o400)
    with pytest.raises(RuntimeError, match="anchor bytes"):
        runner.revalidate_anchor(context, consumption)


def test_ordered_output_binding_mutations_and_fallback_substitution_reject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "research").mkdir(parents=True)
    (root / "output/research").mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    primary, fallback = _patch_output_roots(monkeypatch, root)
    monkeypatch.setattr(
        runner,
        "C13_ANCHOR",
        root
        / "research/f1_aug30_train18_candidate13_pool_adapter_stage_localizer_consumption_v1.json",
    )
    monkeypatch.setattr(runner, "CACHE_MANIFEST", root / "cache/manifest.json")
    monkeypatch.setattr(runner, "_validate_context", lambda *_args: None)
    context = SimpleNamespace(source_commitment_sha256="1" * 64)
    authorization_value = runner.build_authorization_contract(
        context,
        qualification_record_sha256="2" * 64,
        control_binding_commitment_sha256="3" * 64,
    )
    for role in ("authorization", "consumption anchor", "publication lock"):
        changed = dict(authorization_value)
        changed["allowed_output_namespace_set_commitment_sha256"] = "f" * 64
        with pytest.raises(PermissionError, match="ordered output namespaces"):
            runner._validate_allowed_output_namespace_binding(changed, role=role)

    primary_freshness = runner.prevalidate_fresh_output(primary)
    fallback_freshness = runner.prevalidate_fresh_output(fallback)
    monkeypatch.setattr(runner, "revalidate_fresh_output", lambda *_args: None)
    with pytest.raises(PermissionError, match="freshness set mismatch"):
        runner.authorize_preflight_once(
            context,
            authorization_sha256="4" * 64,
            authorization_size_bytes=1,
            output_freshness=fallback_freshness,
            fallback_output_freshness=primary_freshness,
        )

    monkeypatch.undo()
    root = tmp_path / "parent-replacement"
    (root / "output/research").mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    _primary, fallback = _patch_output_roots(monkeypatch, root)
    fallback_freshness = runner.prevalidate_fresh_output(fallback)
    moved = fallback.parent.with_name("research-moved")
    os.rename(fallback.parent, moved)
    fallback.parent.mkdir()
    with pytest.raises(RuntimeError, match="parent identity"):
        runner.revalidate_fresh_output(fallback_freshness)


@pytest.mark.parametrize(
    ("exception_type", "failure_stage"),
    (
        (runner.WatchdogExpired, "WATCHDOG_TIMEOUT"),
        (runner.CpuTimerExpired, "CPU_TIMER_TIMEOUT"),
    ),
)
def test_preanchor_timer_failure_is_published_with_durable_false(
    exception_type: type[BaseException],
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Core:
        RESULT_KIND = _core_result()["kind"]

    context = SimpleNamespace(c13_core=Core())
    authorization = SimpleNamespace(
        reader=SimpleNamespace(),
        qualification_bytes=b"qualification",
        control_binding_commitment_sha256="2" * 64,
    )
    qualification = SimpleNamespace(control_binding_commitment_sha256="2" * 64)
    primary = SimpleNamespace(path=runner.OUTPUT_ROOT)
    fallback = SimpleNamespace(path=runner.FALLBACK_OUTPUT_ROOT)
    phase = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        runner,
        "prevalidate_fresh_output",
        lambda path: primary if path == runner.OUTPUT_ROOT else fallback,
    )
    monkeypatch.setattr(
        runner, "authorize_preflight_once", lambda *_args, **_kwargs: authorization
    )
    monkeypatch.setattr(
        runner, "qualify_source_controls", lambda *_args, **_kwargs: qualification
    )
    monkeypatch.setattr(
        runner, "build_qualification_record", lambda *_args: b"qualification"
    )
    monkeypatch.setattr(
        runner,
        "_arm_resource_envelope",
        lambda: (_ for _ in ()).throw(exception_type()),
    )
    monkeypatch.setattr(
        runner, "_enter_reserved_publication_phase", lambda value: phase
    )

    def materialize(
        _context: Any,
        _authorization: Any,
        _qualification: Any,
        plan: runner.PayloadPlan,
        publication_phase: Any,
    ) -> dict[str, bytes]:
        assert publication_phase is phase
        captured["plan"] = plan
        return {
            "pool_adapter_stage_localizer_aggregate.json": runner.canonical_bytes(
                runner._terminal_aggregate_value(failure_stage=failure_stage)
            ),
            "pool_adapter_stage_localizer_lock.json": b'{}\n',
            "pool_adapter_stage_localizer_read_ledger.json": b'{}\n',
        }

    monkeypatch.setattr(
        runner, "_materialize_payload_plan_in_publication_phase", materialize
    )
    monkeypatch.setattr(runner, "_publish_with_fallback", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        runner, "_audit_allowed_output_inventory", lambda *_args, **_kwargs: "f" * 64
    )
    result = runner.run_one_shot(
        context,
        authorization_sha256="3" * 64,
        authorization_size_bytes=1,
    )
    plan = captured["plan"]
    assert plan.failure_stage == failure_stage
    assert plan.anchor_durable_entry_created_or_partial is False
    assert plan.consumption is None
    assert result["diagnostic_complete"] is False
    assert result["publication_fallback_used"] is False


@pytest.mark.parametrize(
    ("raised", "failure_stage"),
    (
        (runner.WatchdogExpired, "WATCHDOG_TIMEOUT"),
        (runner.CpuTimerExpired, "CPU_TIMER_TIMEOUT"),
        (RuntimeError, "PAYLOAD_MATERIALIZATION"),
    ),
)
def test_postanchor_payload_materialization_failure_transitions_before_terminal_publication(
    raised: type[BaseException],
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Core:
        RESULT_KIND = _core_result()["kind"]

    context = SimpleNamespace(c13_core=Core())
    authorization = SimpleNamespace(
        reader=SimpleNamespace(),
        qualification_bytes=b"qualification",
        control_binding_commitment_sha256="2" * 64,
    )
    qualification = SimpleNamespace(control_binding_commitment_sha256="2" * 64)
    envelope = object()
    phase = object()
    consumption = object()
    events: list[str] = []
    primary = SimpleNamespace(path=runner.OUTPUT_ROOT)
    fallback = SimpleNamespace(path=runner.FALLBACK_OUTPUT_ROOT)
    monkeypatch.setattr(
        runner,
        "prevalidate_fresh_output",
        lambda path: primary if path == runner.OUTPUT_ROOT else fallback,
    )
    monkeypatch.setattr(
        runner, "authorize_preflight_once", lambda *_args, **_kwargs: authorization
    )
    monkeypatch.setattr(
        runner, "qualify_source_controls", lambda *_args, **_kwargs: qualification
    )
    monkeypatch.setattr(
        runner, "build_qualification_record", lambda *_args: b"qualification"
    )
    monkeypatch.setattr(runner, "_arm_resource_envelope", lambda: envelope)
    monkeypatch.setattr(runner, "_validate_resource_envelope", lambda *_args: None)

    def compute(*_args: Any, progress: runner.ConsumptionProgress, **_kwargs: Any) -> Any:
        progress.anchor_durable_entry_created_or_partial = True
        progress.consumption = consumption
        return runner.PayloadPlan(
            consumption=consumption,
            outcome=SimpleNamespace(),
            failure_stage="CORE_LOCALIZER_INVOCATION",
            anchor_durable_entry_created_or_partial=True,
            resource_envelope_enforced=True,
        )

    monkeypatch.setattr(runner, "_compute_payload_plan", compute)
    monkeypatch.setattr(
        runner,
        "_materialize_payload_plan_under_compute_envelope",
        lambda *_args: (_ for _ in ()).throw(raised()),
    )
    monkeypatch.setattr(
        runner,
        "_enter_reserved_publication_phase",
        lambda *_args: (events.append("reserve") or phase),
    )

    def reserved(
        _context: Any,
        _authorization: Any,
        _qualification: Any,
        plan: runner.PayloadPlan,
        publication_phase: Any,
    ) -> dict[str, bytes]:
        assert events == ["reserve"]
        assert publication_phase is phase
        assert plan.consumption is consumption
        assert plan.anchor_durable_entry_created_or_partial is True
        assert plan.failure_stage == failure_stage
        events.append("terminal")
        return {
            "pool_adapter_stage_localizer_aggregate.json": runner.canonical_bytes(
                runner._terminal_aggregate_value(failure_stage=failure_stage)
            ),
            "pool_adapter_stage_localizer_lock.json": b'{}\n',
            "pool_adapter_stage_localizer_read_ledger.json": b'{}\n',
        }

    monkeypatch.setattr(
        runner, "_materialize_payload_plan_in_publication_phase", reserved
    )
    monkeypatch.setattr(
        runner,
        "_publish_with_fallback",
        lambda *_args, **_kwargs: (events.append("publish") or False),
    )
    monkeypatch.setattr(
        runner, "_audit_allowed_output_inventory", lambda *_args, **_kwargs: "f" * 64
    )
    result = runner.run_one_shot(
        context,
        authorization_sha256="3" * 64,
        authorization_size_bytes=1,
    )
    assert events == ["reserve", "terminal", "publish"]
    assert result["diagnostic_complete"] is False
    assert result["check_code"] == runner.ENGINEERING_FAILURE_CODES[failure_stage]


def test_successful_one_shot_accepts_exact_raw_c13_aggregate_without_fake_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    aggregate = _core_result()

    class Core:
        RESULT_KIND = aggregate["kind"]
        RESULT_KEYS = frozenset(aggregate)
        STAGE_CONTRACT = STAGE_CONTRACT

    context = SimpleNamespace(c13_core=Core())
    authorization = SimpleNamespace(
        reader=SimpleNamespace(),
        qualification_bytes=b"qualification",
        control_binding_commitment_sha256="2" * 64,
    )
    qualification = SimpleNamespace(control_binding_commitment_sha256="2" * 64)
    envelope = SimpleNamespace()
    phase = SimpleNamespace()
    plan = SimpleNamespace(failure_stage="CORE_LOCALIZER_INVOCATION")
    payloads = {
        "pool_adapter_stage_localizer_aggregate.json": runner.canonical_bytes(
            aggregate
        ),
        "pool_adapter_stage_localizer_lock.json": b'{"lock":true}\n',
        "pool_adapter_stage_localizer_read_ledger.json": b'{"ledger":true}\n',
    }
    freshness_by_path = {
        runner.OUTPUT_ROOT: SimpleNamespace(path=runner.OUTPUT_ROOT),
        runner.FALLBACK_OUTPUT_ROOT: SimpleNamespace(
            path=runner.FALLBACK_OUTPUT_ROOT
        ),
    }
    monkeypatch.setattr(
        runner, "prevalidate_fresh_output", lambda path: freshness_by_path[path]
    )
    monkeypatch.setattr(
        runner, "authorize_preflight_once", lambda *_args, **_kwargs: authorization
    )
    monkeypatch.setattr(
        runner, "qualify_source_controls", lambda *_args, **_kwargs: qualification
    )
    monkeypatch.setattr(
        runner, "build_qualification_record", lambda *_args: b"qualification"
    )
    monkeypatch.setattr(
        runner,
        "_arm_resource_envelope",
        lambda: (events.append("arm") or envelope),
    )
    monkeypatch.setattr(
        runner,
        "_validate_resource_envelope",
        lambda _seal: events.append("validate"),
    )
    monkeypatch.setattr(
        runner,
        "_compute_payload_plan",
        lambda *_args, **_kwargs: (events.append("compute") or plan),
    )
    monkeypatch.setattr(
        runner,
        "_materialize_payload_plan_under_compute_envelope",
        lambda *_args, **_kwargs: (events.append("materialize") or payloads),
    )
    monkeypatch.setattr(
        runner,
        "_enter_reserved_publication_phase",
        lambda *_args: (events.append("reserve") or phase),
    )

    def publish(*_args: Any, **kwargs: Any) -> None:
        assert kwargs["publication_phase"] is phase
        events.append("publish")

    monkeypatch.setattr(runner, "publish_atomically", publish)
    monkeypatch.setattr(
        runner,
        "_audit_allowed_output_inventory",
        lambda *_args, **_kwargs: "f" * 64,
    )
    value = runner.run_one_shot(
        context,
        authorization_sha256="3" * 64,
        authorization_size_bytes=1,
    )
    assert value["diagnostic_complete"] is True
    assert value["check_code"] == aggregate["check_code"]
    assert value["publication_fallback_used"] is False
    assert events.count("publish") == 1
    assert events.index("materialize") < events.index("publish")


def test_terminal_payload_lineage_is_exact_in_aggregate_ledger_and_lock() -> None:
    reader = runner.GuardedReader()
    context = SimpleNamespace(source_commitment_sha256="1" * 64)
    authorization = SimpleNamespace(
        authorization_sha256="2" * 64,
        qualification_sha256="3" * 64,
    )
    qualification = SimpleNamespace(
        reader=reader,
        control_binding_commitment_sha256="4" * 64,
    )
    payloads = runner._payloads(
        context,
        authorization,
        qualification,
        consumption=None,
        outcome=None,
        failure_stage="WATCHDOG_TIMEOUT",
        resource_envelope_enforced=True,
    )
    for filename in payloads:
        value = runner._decode(payloads[filename], name=filename)
        assert value["engineering_attempt_id"] == runner.ENGINEERING_ATTEMPT_ID
        assert (
            value["scientific_hypothesis_id"]
            == runner.SCIENTIFIC_HYPOTHESIS_ID
        )
    lock = runner._decode(
        payloads["pool_adapter_stage_localizer_lock.json"],
        name="synthetic lock",
    )
    assert lock["allowed_output_namespace_set_commitment_sha256"] == (
        runner._allowed_output_namespace_set_commitment()
    )


def _output_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "repo"
    parent = root / "output/research"
    parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    output, _fallback = _patch_output_roots(monkeypatch, root)
    payloads = {
        "pool_adapter_stage_localizer_aggregate.json": b'{"aggregate":true}\n',
        "pool_adapter_stage_localizer_lock.json": b'{"lock":true}\n',
        "pool_adapter_stage_localizer_read_ledger.json": b'{"ledger":true}\n',
    }

    def noreplace(parent_fd: int, source: str, destination: str) -> None:
        try:
            os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.rename(
                source,
                destination,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            raise FileExistsError(destination)

    monkeypatch.setattr(runner, "_renameat_noreplace", noreplace)
    monkeypatch.setattr(runner, "_validate_publication_phase", lambda *_args: None)
    return output, payloads


def test_atomic_publication_exact_inventory_and_no_clobber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, payloads = _output_fixture(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    runner.publish_atomically(
        output,
        payloads,
        freshness=freshness,
        publication_phase=SimpleNamespace(),
    )
    assert set(path.name for path in output.iterdir()) == set(payloads)
    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == payloads
    with pytest.raises(FileExistsError):
        runner.prevalidate_fresh_output()


def test_output_residue_and_parent_replacement_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, payloads = _output_fixture(monkeypatch, tmp_path)
    residue = output.parent / f".{output.name}.staging-synthetic"
    residue.mkdir()
    with pytest.raises(FileExistsError, match="residue"):
        runner.prevalidate_fresh_output()

    output2, payloads2 = _output_fixture(monkeypatch, tmp_path / "parent-race")
    freshness = runner.prevalidate_fresh_output()
    moved = output2.parent.with_name("research-moved")
    os.rename(output2.parent, moved)
    output2.parent.mkdir()
    with pytest.raises(RuntimeError, match="parent identity"):
        runner.publish_atomically(
            output2,
            payloads2,
            freshness=freshness,
            publication_phase=SimpleNamespace(),
        )


def test_destination_race_is_not_clobbered_and_residue_is_not_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, payloads = _output_fixture(monkeypatch, tmp_path)
    freshness = runner.prevalidate_fresh_output()
    original_verify = runner._verify_owned_directory
    raced = False

    def verify(*args: Any, **kwargs: Any) -> None:
        nonlocal raced
        original_verify(*args, **kwargs)
        if not raced and str(kwargs["name"]).startswith(f".{output.name}.staging-"):
            output.mkdir()
            raced = True

    monkeypatch.setattr(runner, "_verify_owned_directory", verify)
    with pytest.raises(FileExistsError, match="raced"):
        runner.publish_atomically(
            output,
            payloads,
            freshness=freshness,
            publication_phase=SimpleNamespace(),
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert any(
        path.name.startswith(f".{output.name}.staging-")
        for path in output.parent.iterdir()
    )


@pytest.mark.parametrize("failure_point", ("pre_rename", "post_rename"))
def test_primary_publication_failure_uses_exact_fallback_without_cleanup(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary, payloads = _output_fixture(monkeypatch, tmp_path)
    fallback = runner.FALLBACK_OUTPUT_ROOT
    primary_freshness = runner.prevalidate_fresh_output(primary)
    fallback_freshness = runner.prevalidate_fresh_output(fallback)
    normal_rename = runner._renameat_noreplace

    def injected_rename(parent_fd: int, source: str, destination: str) -> None:
        if destination == primary.name:
            if failure_point == "pre_rename":
                raise OSError("synthetic primary pre-rename failure")
            normal_rename(parent_fd, source, destination)
            raise OSError("synthetic primary post-rename failure")
        normal_rename(parent_fd, source, destination)

    monkeypatch.setattr(runner, "_renameat_noreplace", injected_rename)
    used = runner._publish_with_fallback(
        payloads,
        primary_freshness=primary_freshness,
        fallback_freshness=fallback_freshness,
        publication_phase=SimpleNamespace(),
    )
    assert used is True
    assert fallback.is_dir()
    assert {path.name: path.read_bytes() for path in fallback.iterdir()} == payloads
    if failure_point == "pre_rename":
        assert not primary.exists()
        assert any(
            path.name.startswith(f".{primary.name}.staging-")
            for path in primary.parent.iterdir()
        )
    else:
        assert {path.name: path.read_bytes() for path in primary.iterdir()} == payloads
    commitment = runner._audit_allowed_output_inventory(
        primary_freshness,
        fallback_freshness,
        payloads,
        fallback_used=True,
    )
    assert len(commitment) == 64


@pytest.mark.parametrize(
    "mutation", ("extra", "status", "bitset", "boundary", "kind")
)
def test_exact_public_aggregate_schema_rejects_extra_and_malformed(
    mutation: str,
) -> None:
    exact = _core_result()

    class Core:
        RESULT_KIND = exact["kind"]
        RESULT_KEYS = frozenset(exact)
        STAGE_CONTRACT = STAGE_CONTRACT

    context = SimpleNamespace(c13_core=Core())
    runner._validate_core_aggregate(context, exact)
    drifted = dict(exact)
    if mutation == "extra":
        drifted["extra"] = False
    elif mutation == "status":
        drifted["status"] = "PASS"
    elif mutation == "bitset":
        drifted["stage_passed_all_bitset"] = "1" * 9
    elif mutation == "boundary":
        drifted["metric_computed"] = True
    else:
        drifted["kind"] = runner.TERMINAL_KIND
    with pytest.raises(ValueError):
        runner._validate_core_aggregate(context, drifted)


def test_terminal_public_aggregate_schema_rejects_extra_and_malformed() -> None:
    terminal = runner._terminal_aggregate_value(
        failure_stage="WATCHDOG_TIMEOUT"
    )
    context = SimpleNamespace(
        c13_core=SimpleNamespace(RESULT_KIND=_core_result()["kind"])
    )
    assert runner._validate_published_aggregate(context, terminal) is False
    for key, value in (
        ("extra", False),
        ("check_code", "DRIFT"),
        ("stage_reached_all_bitset", "1" * 10),
        ("engineering_attempt_id", "DRIFT"),
    ):
        drifted = dict(terminal)
        drifted[key] = value
        with pytest.raises((ValueError, KeyError)):
            runner._validate_published_aggregate(context, drifted)
